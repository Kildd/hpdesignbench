"""
Deterministic in-memory DuckDB engine over sealed brute-force Parquet parts.

Scientific identity comparisons use relative tolerance ``1e-12`` (same default
as ``scientific_values_equal``). Sentinel utilization equality uses absolute
tolerance ``1e-9``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core.ioh_core.workflows.brute_force.analysis.columns import (
    BruteForceColumnFamilies,
    discover_column_families,
)
from core.ioh_core.workflows.brute_force.analysis.context import (
    BruteForceAnalysisError,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.deps import import_duckdb

# Relative tolerance for objective/penalty scientific identities.
SCIENTIFIC_RTOL = 1e-12
# Absolute tolerance for opt_invalid_utilization sentinel equality checks.
SENTINEL_ATOL = 1e-9

_DEFAULT_VIEW_NAME = "bf_parts"
_MAX_RESULT_ROWS = 10_000


class BruteForceEngineError(BruteForceAnalysisError):
    """
    Raised when the DuckDB analysis engine cannot be constructed or queried.
    """


def quote_ident(name: str) -> str:
    """
    Quote a DuckDB identifier; only sealed schema names should be passed.
    """
    if not isinstance(name, str) or not name:
        raise BruteForceEngineError(f"identifier must be a non-empty string, got {name!r}")
    return '"' + name.replace('"', '""') + '"'


def quote_string(value: str) -> str:
    """
    Quote a DuckDB string literal (used only for SET paths / validated options).
    """
    if not isinstance(value, str):
        raise BruteForceEngineError(f"string literal must be str, got {type(value)!r}")
    return "'" + value.replace("'", "''") + "'"


def quote_path(path: Path | str) -> str:
    """
    Quote a filesystem path as a DuckDB string literal.
    """
    return quote_string(str(path))


@dataclass
class BruteForceQueryEngine:
    """
    Read-only DuckDB session bound to one sealed brute-force dataset.

    The engine never materializes the full base relation into Python. Callers
    must use aggregate SQL or bounded ``LIMIT`` queries through
    :meth:`fetch_rows` / :meth:`fetch_one`.
    """

    context: BruteForceDatasetContext
    columns: BruteForceColumnFamilies
    connection: Any
    view_name: str = _DEFAULT_VIEW_NAME
    scientific_rtol: float = SCIENTIFIC_RTOL
    sentinel_atol: float = SENTINEL_ATOL

    def close(self) -> None:
        """
        Close the DuckDB connection.
        """
        connection = self.connection
        self.connection = None
        if connection is not None:
            connection.close()

    def __enter__(self) -> BruteForceQueryEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def tolerances_metadata(self) -> dict[str, float]:
        """
        Return the locked numerical tolerances for report metadata.
        """
        return {
            "scientific_rtol": float(self.scientific_rtol),
            "sentinel_atol": float(self.sentinel_atol),
        }

    def fetch_one(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
    ) -> tuple[Any, ...] | None:
        """
        Execute SQL and return at most one row.
        """
        rows = self.fetch_rows(sql, params=params, max_rows=1)
        return rows[0] if rows else None

    def fetch_rows(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        max_rows: int = _MAX_RESULT_ROWS,
    ) -> list[tuple[Any, ...]]:
        """
        Execute SQL and return up to ``max_rows`` result tuples.

        Raises
        ------
        BruteForceEngineError
            If ``max_rows`` is invalid or the result exceeds ``max_rows``.
        """
        if self.connection is None:
            raise BruteForceEngineError("DuckDB connection is closed")
        if (
            not isinstance(max_rows, int)
            or isinstance(max_rows, bool)
            or max_rows < 1
        ):
            raise BruteForceEngineError(
                f"max_rows must be an int >= 1, got {max_rows!r}"
            )
        try:
            if params is None:
                result = self.connection.execute(sql)
            else:
                result = self.connection.execute(sql, list(params))
            rows = result.fetchmany(max_rows + 1)
        except Exception as exc:  # noqa: BLE001 - surface as analysis error
            raise BruteForceEngineError(f"DuckDB query failed: {exc}") from exc
        if len(rows) > max_rows:
            raise BruteForceEngineError(
                f"query returned more than max_rows={max_rows}; "
                "narrow the SQL with aggregates or LIMIT"
            )
        return list(rows)

    def fetch_dicts(
        self,
        sql: str,
        params: Sequence[Any] | None = None,
        *,
        max_rows: int = _MAX_RESULT_ROWS,
    ) -> list[dict[str, Any]]:
        """
        Execute SQL and return up to ``max_rows`` row dictionaries.
        """
        if self.connection is None:
            raise BruteForceEngineError("DuckDB connection is closed")
        try:
            if params is None:
                result = self.connection.execute(sql)
            else:
                result = self.connection.execute(sql, list(params))
            description = result.description
            rows = result.fetchmany(max_rows + 1)
        except Exception as exc:  # noqa: BLE001
            raise BruteForceEngineError(f"DuckDB query failed: {exc}") from exc
        if len(rows) > max_rows:
            raise BruteForceEngineError(
                f"query returned more than max_rows={max_rows}; "
                "narrow the SQL with aggregates or LIMIT"
            )
        if description is None:
            return []
        keys = [str(col[0]) for col in description]
        return [dict(zip(keys, row)) for row in rows]

    def count_rows(self) -> int:
        """
        Return ``COUNT(*)`` over the sealed parts view.
        """
        row = self.fetch_one(f"SELECT COUNT(*) FROM {quote_ident(self.view_name)}")
        if row is None:
            raise BruteForceEngineError("COUNT(*) returned no row")
        return int(row[0])


def open_query_engine(
    context: BruteForceDatasetContext,
    *,
    options: DatasetReportOptions | None = None,
) -> BruteForceQueryEngine:
    """
    Open an in-memory DuckDB engine over the sealed Parquet glob.

    Parameters
    ----------
    context : BruteForceDatasetContext
        Sealed dataset context from :func:`open_brute_force_dataset`.
    options : DatasetReportOptions or None, optional
        Thread / memory / spill settings. Defaults to one DuckDB thread.

    Returns
    -------
    BruteForceQueryEngine
        Connected engine with a registered parts view.

    Raises
    ------
    BruteForceEngineError
        If DuckDB setup or column discovery fails.
    """
    options = options or DatasetReportOptions()
    try:
        columns = discover_column_families(
            context.schema_doc,
            expected_constraint_names=context.constraint_names,
        )
    except Exception as exc:  # noqa: BLE001
        raise BruteForceEngineError(str(exc)) from exc

    duckdb = import_duckdb()
    try:
        connection = duckdb.connect(database=":memory:")
    except Exception as exc:  # noqa: BLE001
        raise BruteForceEngineError(f"could not open DuckDB connection: {exc}") from exc

    try:
        connection.execute(f"SET threads TO {int(options.duckdb_threads)}")
        if options.duckdb_memory_limit is not None:
            connection.execute(
                f"SET memory_limit TO {quote_string(str(options.duckdb_memory_limit))}"
            )
        if options.duckdb_temp_directory is not None:
            temp_dir = Path(options.duckdb_temp_directory)
            temp_dir.mkdir(parents=True, exist_ok=True)
            connection.execute(
                f"SET temp_directory TO {quote_path(temp_dir.resolve())}"
            )

        # DuckDB rejects prepared parameters in CREATE VIEW, so quote the
        # sealed parts glob as a string literal (never concatenate CLI text).
        connection.execute(
            f"CREATE VIEW {quote_ident(_DEFAULT_VIEW_NAME)} AS "
            f"SELECT * FROM read_parquet({quote_path(context.parts_glob)})"
        )
    except Exception as exc:  # noqa: BLE001
        try:
            connection.close()
        except Exception:
            pass
        raise BruteForceEngineError(
            f"could not configure DuckDB over {context.parts_glob!r}: {exc}"
        ) from exc

    return BruteForceQueryEngine(
        context=context,
        columns=columns,
        connection=connection,
        view_name=_DEFAULT_VIEW_NAME,
        scientific_rtol=SCIENTIFIC_RTOL,
        sentinel_atol=SENTINEL_ATOL,
    )
