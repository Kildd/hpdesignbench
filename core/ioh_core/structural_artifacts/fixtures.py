"""
Synthetic StructuralArtifact fixtures for storage round-trip tests.
"""
from __future__ import annotations

from core.ioh_core.structural_artifacts.artifact import StructuralArtifact
from slab_construction.slab_construction import MKCurveKey, SectionResultKey


def make_success_artifact(*, seed: int = 0) -> StructuralArtifact:
    """
    Build a small successful artifact with typed M–κ arrays and metadata.
    """
    section_key = SectionResultKey(
        result_type="M_u",
        x=0.0,
        limit_state="ULS",
        n=float(seed),
        flipped=False,
    )
    mk_key = MKCurveKey(
        limit_state="SLS",
        constitutive_law="linear",
        method="incremental",
        simplification=True,
        softening=None,
        n=float(seed),
        m_k_num_points=5,
    )
    m_y = tuple(float(seed + i) for i in range(5))
    chi_y = tuple(0.001 * float(seed + i) for i in range(5))
    payload = (
        ("m", float(10 + seed)),
        ("ok", True),
        ("note", f"seed-{seed}"),
        ("nested", (("a", 1), ("b", float("nan")))),
    )
    key = (
        ("snapshot_hash", f"snap-{seed}"),
        ("n", float(seed)),
        ("params", (("t", int(seed % 7)),)),
    )
    return StructuralArtifact(
        key=key,
        snapshot_hash=f"snap-{seed}",
        n=float(seed),
        m_u=((section_key, payload),),
        m_cr=(),
        mk_curves=((mk_key, ((0.0, (m_y, chi_y)), (0.5, (m_y, chi_y)))),),
        mk_metadata=((mk_key, (("status", "ok"), ("iters", int(seed)))),),
        mk_cleared=False,
        build_failure_result=None,
    )


def make_mk_cleared_artifact(*, seed: int = 1) -> StructuralArtifact:
    """
    Build an artifact that records cleared M–κ curves.
    """
    art = make_success_artifact(seed=seed)
    return StructuralArtifact(
        key=art.key,
        snapshot_hash=art.snapshot_hash,
        n=art.n,
        m_u=art.m_u,
        m_cr=art.m_cr,
        mk_curves=(),
        mk_metadata=(),
        mk_cleared=True,
        build_failure_result=None,
    )


def make_build_failure_artifact(*, seed: int = 2) -> StructuralArtifact:
    """
    Build a structural build-failure artifact.
    """
    key = (
        ("snapshot_hash", f"fail-{seed}"),
        ("n", float(seed)),
        ("params", (("t", int(seed)),)),
    )
    return StructuralArtifact(
        key=key,
        snapshot_hash=f"fail-{seed}",
        n=float(seed),
        m_u=(),
        m_cr=(),
        mk_curves=(),
        mk_metadata=(),
        mk_cleared=True,
        build_failure_result={
            "analysis_ok": False,
            "analysis_error": {"code": "materials", "seed": int(seed)},
            "y": float("inf"),
            "params": {"t": int(seed)},
        },
    )
