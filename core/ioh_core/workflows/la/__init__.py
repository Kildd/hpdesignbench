"""
Landscape analysis package (sampled ELA plus exhaustive catalogue families).

Consumes sealed brute-force or sampling Parquet. Does not evaluate HP-shell
candidates, does not register as ``hdb-experiment``, and must not import
DuckDB, pflacco, or SALib at package import time.

Optional dependency:

- Parquet / DuckDB / numdifftools: ``pip install -e ".[la]"``
- pflacco (sampled ELA only, not in the extra): ``pip install pflacco==1.2.2 --no-deps``

Importing this package must not load DuckDB, PyArrow, pflacco, numdifftools,
or SALib.
Call :func:`core.ioh_core.workflows.la.deps.import_pyarrow` and
:func:`core.ioh_core.workflows.la.deps.import_duckdb` only from code
paths that need those libraries.
"""
