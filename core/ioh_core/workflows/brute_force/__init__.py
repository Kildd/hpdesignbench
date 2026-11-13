"""
Brute-force sharding workflow (Parquet datasets).

Optional dependencies:
- Parquet I/O: ``pip install -e ".[bruteforce]"``
- Dataset analysis toolbox: ``pip install -e ".[bruteforce-analysis]"``

Importing this package must not load PyArrow or DuckDB; call
:func:`core.ioh_core.workflows.brute_force.deps.import_pyarrow` or
:func:`core.ioh_core.workflows.brute_force.deps.import_duckdb` only from
code paths that need those libraries.
"""
