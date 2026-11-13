"""
ghalton-compatible Halton sequencer backed by SciPy.

DFLINT calls ``ghalton.Halton(n).get(N)``. PyPI ``ghalton`` ships only an
sdist with a C++ extension; this shim keeps the same call shape using
``scipy.stats.qmc.Halton`` (already a core dependency).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import qmc


class Halton:
    """
    Minimal Halton generator matching ``ghalton.Halton``.

    Parameters
    ----------
    dim : int
        Number of dimensions.
    """

    def __init__(self, dim: int) -> None:
        self._engine = qmc.Halton(d=int(dim), scramble=False)

    def get(self, n: int = 1) -> list[Any]:
        """
        Return ``n`` Halton points in the unit hypercube.

        Parameters
        ----------
        n : int, optional
            Number of points. Default is ``1``.

        Returns
        -------
        list
            Nested list of floats with shape ``(n, dim)`` (list-of-lists),
            matching ghalton's return style used by DFLINT indexing.
        """
        points = np.asarray(self._engine.random(int(n)), dtype=float)
        return points.tolist()
