"""Integrator carried from HPShell through SLS rebuild and section flip."""
from __future__ import annotations

import unittest

from structuralcodes.sections.section_integrators import (
    FiberIntegrator,
    MarinIntegrator,
)

from core.analysis_core.section_methods import flipped_section, sls_section_EC
from _mains.Testing.fixtures.materials import concrete_c50_uls, solidian_Q95_pre_20
from slab_construction.slabs.hp_slab.hp_model.hp_geometry import HPGeometry
from slab_construction.slabs.hp_slab.hp_model.hp_shell import HPShell


def _shell(**kwargs) -> HPShell:
    geometry = HPGeometry(B=1200, L=6750, Hx=40, Hy=160, t=50, dy=100, nt=7)
    return HPShell(
        geometry,
        concrete_c50_uls,
        solidian_Q95_pre_20,
        reinf_area=50,
        name="integrator-hook",
        **kwargs,
    )


class TestSectionIntegratorHook(unittest.TestCase):
    """Default fiber, explicit Marin, and rebuilds that copy the source."""

    def test_default_shell_uses_fiber(self) -> None:
        section = _shell().section_at(0.5)
        self.assertIsInstance(
            section.section_calculator.integrator, FiberIntegrator
        )
        self.assertEqual(section.section_calculator.mesh_size, 0.01)

    def test_explicit_fiber_shell_uses_fiber(self) -> None:
        section = _shell(integrator="fiber").section_at(0.5)
        self.assertIsInstance(
            section.section_calculator.integrator, FiberIntegrator
        )

    def test_marin_shell_section_uses_marin(self) -> None:
        section = _shell(integrator="marin", mesh_size=0.01).section_at(0.5)
        self.assertIsInstance(
            section.section_calculator.integrator, MarinIntegrator
        )
        self.assertEqual(section.section_calculator.mesh_size, 0.01)

    def test_sls_rebuild_keeps_marin(self) -> None:
        uls = _shell(integrator="marin").section_at(0.5)
        sls = sls_section_EC(uls, "TENSTIFF_PARABOLIC")
        self.assertIsInstance(
            sls.section_calculator.integrator, MarinIntegrator
        )
        self.assertEqual(sls.section_calculator.mesh_size, 0.01)

    def test_flipped_section_keeps_marin(self) -> None:
        uls = _shell(integrator="marin").section_at(0.5)
        flipped = flipped_section(uls)
        self.assertIsInstance(
            flipped.section_calculator.integrator, MarinIntegrator
        )
        self.assertEqual(flipped.section_calculator.mesh_size, 0.01)

    def test_unknown_integrator_raises(self) -> None:
        with self.assertRaises(ValueError):
            _shell(integrator="quadrature")

    def test_mesh_size_outside_unit_interval_raises(self) -> None:
        with self.assertRaises(ValueError):
            _shell(mesh_size=0.0)
        with self.assertRaises(ValueError):
            _shell(mesh_size=1.1)
