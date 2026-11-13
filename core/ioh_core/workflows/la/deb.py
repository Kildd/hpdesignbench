"""
Fieldsend Eq. (6) / Deb (2000) comparator for constrained search.

Minimisation height ``h`` is ``(0, y)`` on analysis-local **F** and
``(1, phi)`` off **F**. Equal-phi infeasible designs are tied regardless
of ``y``. Linear id is not a component of ``h``.

``phi`` and **F** are the analysis-local pair from
:func:`core.ioh_core.workflows.la.preprocess.constraint_violations` on
the same selected constraint names as GBLT. This module does not build a
plateau graph, CoLON CLI, or escape edges.
"""
from __future__ import annotations


def deb_key(
    *,
    feasible: bool,
    y: float,
    phi: float,
) -> tuple[int, float]:
    """
    Return Fieldsend Eq. (6) height ``h`` (lexicographic; smaller is better).

    Parameters
    ----------
    feasible : bool
        Analysis-local **F** (all selected ``v_i == 0``).
    y : float
        Objective used only on **F**.
    phi : float
        Analysis-local sum of violations; used only off **F**.
    """
    if bool(feasible):
        return (0, float(y))
    return (1, float(phi))


def deb_compare(
    *,
    feasible_a: bool,
    y_a: float,
    phi_a: float,
    feasible_b: bool,
    y_b: float,
    phi_b: float,
) -> int:
    """
    Compare two designs under Deb / Fieldsend Eq. (6).

    Returns
    -------
    int
        ``-1`` if ``a`` is better, ``1`` if ``b`` is better, ``0`` if tied.
    """
    left = deb_key(feasible=feasible_a, y=y_a, phi=phi_a)
    right = deb_key(feasible=feasible_b, y=y_b, phi=phi_b)
    if left < right:
        return -1
    if left > right:
        return 1
    return 0
