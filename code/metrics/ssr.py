"""Canonical Subspace-Corrected Ranking score."""

from __future__ import annotations

import numpy as np


def ssr_dagger(reff: float, sa_k: float, n_class: int, eps: float = 1e-12) -> float:
    """Return SSR‡ = (log r_eff - log SA_k) / log C.

    ``reff`` must be the RankMe effective rank computed with normalized
    singular values, not the squared-singular-value VNE effective rank.
    """
    if n_class <= 1:
        raise ValueError(f"n_class must be greater than one, got {n_class}")
    return float(
        (np.log(max(float(reff), eps)) - np.log(max(float(sa_k), eps)))
        / np.log(int(n_class))
    )
