"""Final camera-ready scoring kernels; no implicit centering."""
import numpy as np
from typing import Dict, Tuple

def principal_angles_from_matrices(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算两个特征矩阵对应的 top-k 子空间之间的主角。

    步骤：
        1. SVD(H_A) → V_A[:k]；SVD(H_B) → V_B[:k]
        2. 交叉积 M = V_A.T @ V_B
        3. SVD(M) 的奇异值 = cos(θᵢ)

    Args:
        H_A: (N_A, d) 特征矩阵
        H_B: (N_B, d) 特征矩阵
        k:   子空间维度

    Returns:
        cos_angles: (k_eff,) cos(θᵢ) ∈ [0,1]，降序
        angles_rad: (k_eff,) θᵢ ∈ [0, π/2]，升序（对应 cos 降序）
    """
    N_A, d_A = H_A.shape
    N_B, d_B = H_B.shape
    if d_A != d_B:
        raise ValueError(f"特征维度不匹配: {d_A} vs {d_B}")

    k_eff = max(1, min(k, N_A, N_B, d_A))

    _, _, Vt_A = np.linalg.svd(H_A, full_matrices=False)
    _, _, Vt_B = np.linalg.svd(H_B, full_matrices=False)

    V_A = Vt_A[:k_eff, :].T   # (d, k_eff)
    V_B = Vt_B[:k_eff, :].T   # (d, k_eff)

    M_cross = V_A.T @ V_B     # (k_eff, k_eff)
    cos_angles = np.linalg.svd(M_cross, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)

    angles_rad = np.arccos(cos_angles)

    return cos_angles, angles_rad

def compute_subspace_alignment(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> Dict:
    """
    计算 SA_k：两个特征矩阵的 top-k 子空间对齐度。

    Args:
        H_A: (N_A, d) 输入特征矩阵；本函数不额外中心化或归一化
        H_B: (N_B, d) 特征矩阵
        k:   主子空间维度

    Returns:
        dict: {
            "sa_k":           float  ∈ [0,1]，主指标
            "grassmann_dist": float  Grassmann 测地距离 √(Σθᵢ²)
            "cos_angles":     list   k 个 cos(θᵢ) 值（降序）
            "k_used":         int    实际有效 k
        }
    """
    cos_angles, angles_rad = principal_angles_from_matrices(H_A, H_B, k)

    sa_k = float(np.mean(cos_angles ** 2))
    grassmann_dist = float(np.sqrt(np.sum(angles_rad ** 2)))
    k_eff = len(cos_angles)

    return {
        "sa_k":           sa_k,
        "grassmann_dist": grassmann_dist,
        "cos_angles":     cos_angles.tolist(),
        "k_used":         k_eff,
    }
