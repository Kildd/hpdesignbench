"""
Adam on the shared discrete finite-difference gradient.
"""
from __future__ import annotations

from typing import Any

from core.ioh_core.algorithms.optax.engine import OptaxDiscreteGradientAlgorithm

ADAM_LEARNING_RATE = 0.3
ADAM_B1 = 0.9
ADAM_B2 = 0.999
ADAM_EPS = 1e-8
ADAM_EPS_ROOT = 0.0


class OptaxAdam(OptaxDiscreteGradientAlgorithm):
    """
    Adam on ``y_p`` with a discrete finite-difference gradient in index space.

    The continuous state is projected to a catalogue-index vector only when
    ``problem`` is called. Adam receives the raw finite-difference gradient.
    Frozen constants: lr=0.3, b1=0.9, b2=0.999, eps=1e-8.
    """

    name = "OPTAX_ADAM"
    info = (
        "Adam on y_p; discrete finite-difference gradient in catalogue-index "
        "space; continuous state with integer evaluation projection; "
        "lr=0.3, b1=0.9, b2=0.999, eps=1e-8"
    )

    def build_optimizer(self, optax_module: Any, jnp_module: Any) -> Any:
        """
        Build ``optax.adam`` with the frozen index-grid constants.

        Parameters
        ----------
        optax_module : module
            Imported ``optax`` module.
        jnp_module : module
            Imported ``jax.numpy`` module.

        Returns
        -------
        GradientTransformation
            Adam update. Not NAdam.
        """
        return optax_module.adam(
            learning_rate=ADAM_LEARNING_RATE,
            b1=ADAM_B1,
            b2=ADAM_B2,
            eps=ADAM_EPS,
            eps_root=ADAM_EPS_ROOT,
            mu_dtype=jnp_module.float64,
            nesterov=False,
        )
