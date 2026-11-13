"""
Sign-normalised Nesterov SGD on the shared discrete gradient.
"""
from __future__ import annotations

from typing import Any, Sequence

from core.ioh_core.algorithms.optax.engine import (
    OptaxDiscreteGradientAlgorithm,
    sign_gradient,
)

SGD_LEARNING_RATE = 0.03
SGD_MOMENTUM = 0.9
SGD_NESTEROV = True


class OptaxSGDNesterov(OptaxDiscreteGradientAlgorithm):
    """
    Sign-normalised Nesterov SGD on ``y_p``.

    The finite-difference gradient is the same raw vector Adam and RProp
    see. This token passes its component-wise sign to ``optax.sgd``.
    Frozen constants: lr=0.03, momentum=0.9, Nesterov enabled.
    """

    name = "OPTAX_SGD_NESTEROV"
    info = (
        "Sign-normalised Nesterov SGD on y_p; sign of the discrete "
        "finite-difference gradient in catalogue-index space; continuous "
        "state with integer evaluation projection; lr=0.03, momentum=0.9, "
        "Nesterov=True"
    )

    def build_optimizer(self, optax_module: Any, jnp_module: Any) -> Any:
        """
        Build ``optax.sgd`` with Nesterov momentum.

        Parameters
        ----------
        optax_module : module
            Imported ``optax`` module.
        jnp_module : module
            Imported ``jax.numpy`` module.

        Returns
        -------
        GradientTransformation
            Nesterov SGD. Not ``optax.sign_sgd``.
        """
        return optax_module.sgd(
            learning_rate=SGD_LEARNING_RATE,
            momentum=SGD_MOMENTUM,
            nesterov=SGD_NESTEROV,
            accumulator_dtype=jnp_module.float64,
        )

    def preprocess_gradient(self, gradient: Sequence[float]) -> list[float]:
        """
        Return the component-wise sign of the raw finite-difference gradient.

        Parameters
        ----------
        gradient : sequence of float
            Raw partial derivatives.

        Returns
        -------
        list of float
            ``-1.0``, ``0.0``, or ``1.0`` per component.
        """
        return sign_gradient(gradient)
