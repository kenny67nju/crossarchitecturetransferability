"""Final camera-ready scoring kernels; no implicit centering."""
import numpy as np
from typing import Dict, Tuple

def effective_rank_from_singular_values(sigma: np.ndarray) -> float:
    """Compute RankMe effective rank from singular values.

    RankMe uses ``p_i = sigma_i / sum_j sigma_j``.  Centralizing the
    definition here prevents experiments from silently switching to the
    squared-singular-value distribution used by VNE.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = sigma[np.isfinite(sigma) & (sigma > 1e-12)]
    if len(sigma) == 0:
        return 1.0
    p = sigma / sigma.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

def effective_rank(H: np.ndarray) -> float:
    """
    计算特征矩阵 H 的谱熵有效秩。

    r_eff = exp(-Σ pᵢ log pᵢ)，其中 pᵢ = σᵢ / Σσⱼ（奇异值归一化概率）

    Args:
        H: (N, d) 特征矩阵

    Returns:
        r_eff: 有效秩标量
    """
    N, d = H.shape
    if N <= d:
        G = (H @ H.T) / N
    else:
        G = (H.T @ H) / N

    eigvals = np.linalg.eigvalsh(G)[::-1]
    eigvals = np.maximum(eigvals, 0)
    eigvals = eigvals[eigvals > 1e-10]

    if len(eigvals) == 0:
        return 1.0

    sigma = np.sqrt(eigvals)
    return effective_rank_from_singular_values(sigma)
