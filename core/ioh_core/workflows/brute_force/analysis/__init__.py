"""
Read-only post-collection analysis toolbox for sealed brute-force datasets.

Install optional DuckDB support with::

    pip install -e ".[bruteforce-analysis]"

Importing this package must not load DuckDB or PyArrow.
"""
from __future__ import annotations

from core.ioh_core.workflows.brute_force.analysis.best_designs import (
    run_best_designs_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.constraint_decision_tree import (
    DecisionTreeOptions,
    run_and_write_constraint_decision_tree,
    run_constraint_decision_tree_analysis,
    write_constraint_decision_tree_artifacts,
)
from core.ioh_core.workflows.brute_force.analysis.constraints import (
    run_constraints_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.context import (
    BruteForceAnalysisError,
    open_brute_force_dataset,
)
from core.ioh_core.workflows.brute_force.analysis.coverage import (
    run_coverage_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.engine import (
    BruteForceEngineError,
    BruteForceQueryEngine,
    open_query_engine,
)
from core.ioh_core.workflows.brute_force.analysis.feasibility import (
    run_feasibility_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.factorial_sensitivity import (
    assess_factorial_sensitivity_eligibility,
    compute_factorial_sobol_indices,
    exact_sobol_from_tensor,
    run_and_write_factorial_sensitivity,
    run_factorial_sensitivity_analysis,
    write_factorial_sensitivity_artifacts,
)
from core.ioh_core.workflows.brute_force.analysis.integrity import (
    run_integrity_analysis,
)
from core.ioh_core.workflows.brute_force.analysis.dataset_report import (
    refresh_near_optimum_bands_report,
    run_and_write_dataset_report,
    run_dataset_report,
)
from core.ioh_core.workflows.brute_force.analysis.models import (
    BruteForceDatasetContext,
    DatasetReportOptions,
)
from core.ioh_core.workflows.brute_force.analysis.report import (
    empty_dataset_report,
    write_dataset_reports,
    write_integrity_reports,
)
from core.ioh_core.workflows.brute_force.analysis.variable_cooccurrence import (
    run_and_write_variable_cooccurrence,
    run_variable_cooccurrence_analysis,
    write_variable_cooccurrence_artifacts,
)

__all__ = [
    "BruteForceAnalysisError",
    "BruteForceDatasetContext",
    "BruteForceEngineError",
    "BruteForceQueryEngine",
    "DatasetReportOptions",
    "DecisionTreeOptions",
    "assess_factorial_sensitivity_eligibility",
    "compute_factorial_sobol_indices",
    "empty_dataset_report",
    "exact_sobol_from_tensor",
    "open_brute_force_dataset",
    "open_query_engine",
    "run_and_write_constraint_decision_tree",
    "run_and_write_dataset_report",
    "run_and_write_factorial_sensitivity",
    "refresh_near_optimum_bands_report",
    "run_and_write_variable_cooccurrence",
    "run_best_designs_analysis",
    "run_constraint_decision_tree_analysis",
    "run_constraints_analysis",
    "run_coverage_analysis",
    "run_feasibility_analysis",
    "run_factorial_sensitivity_analysis",
    "run_integrity_analysis",
    "run_dataset_report",
    "run_variable_cooccurrence_analysis",
    "write_constraint_decision_tree_artifacts",
    "write_dataset_reports",
    "write_factorial_sensitivity_artifacts",
    "write_integrity_reports",
    "write_variable_cooccurrence_artifacts",
]
