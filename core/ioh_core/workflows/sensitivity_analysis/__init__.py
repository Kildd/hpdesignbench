"""
Sobol sensitivity-analysis workflow (Saltelli / Jansen via SALib).

Optional dependency:
- Sampling, analyse, and Parquet I/O: ``pip install -e ".[sensitivity]"``

Importing this package must not load SALib or PyArrow; call
:func:`core.ioh_core.workflows.sensitivity_analysis.deps.import_salib` or
:func:`core.ioh_core.workflows.sensitivity_analysis.deps.import_pyarrow`
only from code that needs those libraries.
"""
