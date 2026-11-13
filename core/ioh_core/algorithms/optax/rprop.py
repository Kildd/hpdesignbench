"""
RProp on the shared discrete finite-difference gradient.
"""
from __future__ import annotations

from typing import Any

from core.ioh_core.algorithms.optax.engine import OptaxDiscreteGradientAlgorithm

RPROP_INITIAL_STEP = 0.1
RPROP_ETA_PLUS = 1.2
RPROP_ETA_MINUS = 0.5
RPROP_MIN_STEP = 1e-6
RPROP_MAX_STEP = 50.0


class OptaxRProp(OptaxDiscreteGradientAlgorithm):
    """
    RProp on ``y_p`` with the same raw finite-difference gradient as Adam.

    ``learning_rate`` is the initial step. Box clipping does not reset the
    RProp state and does not shrink the proposed step. Frozen constants:
    initial step 0.1, eta+ 1.2, eta- 0.5, min 1e-6, max 50.
    """

    name = "OPTAX_RPROP"
    info = (
        "RProp on y_p; raw discrete finite-difference gradient in "
        "catalogue-index space; continuous state with integer evaluation "
        "projection; initial step=0.1, eta+=1.2, eta-=0.5, min=1e-6, max=50"
    )

    def build_optimizer(self, optax_module: Any, jnp_module: Any) -> Any:
        """
        Build ``optax.rprop`` with the frozen Riedmiller constants.

        Parameters
        ----------
        optax_module : module
            Imported ``optax`` module.
        jnp_module : module
            Imported ``jax.numpy`` module. Unused; RProp has no moment dtype
            argument.

        Returns
        -------
        GradientTransformation
            RProp update on the raw gradient.
        """
        del jnp_module
        return optax_module.rprop(
            learning_rate=RPROP_INITIAL_STEP,
            eta_minus=RPROP_ETA_MINUS,
            eta_plus=RPROP_ETA_PLUS,
            min_step_size=RPROP_MIN_STEP,
            max_step_size=RPROP_MAX_STEP,
        )
