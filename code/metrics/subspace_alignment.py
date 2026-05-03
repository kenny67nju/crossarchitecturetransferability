"""
Paper06 — Grassmann 统一框架
子空间对齐度 SA_k（Subspace Alignment）

核心公式：
    SA_k(S_A, S_B) = (1/k) Σᵢ cos²(θᵢ)

    θ₁,...,θₖ 为 S_A 和 S_B 的主角（principal angles）
    SA_k = 1 → 完全对齐（方向冗余）
    SA_k = 0 → 完全正交（理想互补）

与 Paper05/sa_metric.py 的关系：
    本模块是独立的纯 numpy 计算核，去除了 DataLoader 依赖，
    适用于直接传入特征矩阵的调用场景。
    SAMetric 类（含特征提取）请参见 sa_metric_full.py。
"""

import numpy as np
from typing import Dict, Tuple


# ─────────────────────────────────────────────────────────────
# 核心计算
# ─────────────────────────────────────────────────────────────

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
        H_A: (N_A, d) 特征矩阵（已建议 per-sample L2 归一化）
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


def compute_multiscale_sa(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k_values: Tuple[int, ...] = (10, 20, 50),
) -> Dict:
    """
    多尺度 SA_k，对多个 k 值分别计算并汇总。

    Returns:
        dict: {
            "sa_k10": float, "sa_k20": float, "sa_k50": float,
            "mean_sa_multiscale": float
        }
    """
    results = {}
    sa_list = []
    for k in k_values:
        r = compute_subspace_alignment(H_A, H_B, k=k)
        results[f"sa_k{k}"] = r["sa_k"]
        sa_list.append(r["sa_k"])
    results["mean_sa_multiscale"] = float(np.mean(sa_list))
    return results


def top_k_basis(H: np.ndarray, k: int) -> np.ndarray:
    """
    提取特征矩阵的 top-k 右奇异向量（Grassmann 流形上的点表示）。

    Returns:
        V_k: (d, k) 正交矩阵，列向量为 top-k 主方向
    """
    k_eff = max(1, min(k, min(H.shape)))
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k_eff, :].T   # (d, k_eff)


# ─────────────────────────────────────────────────────────────
# 单元测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d = 200, 512

    def _norm(H):
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    # 测试 1：相同矩阵 → SA_k ≈ 1.0
    H = _norm(rng.standard_normal((N, d)))
    r = compute_subspace_alignment(H, H.copy(), k=20)
    assert r["sa_k"] > 0.99, f"同矩阵 SA_k={r['sa_k']:.4f} 应≈1"
    print(f"[测试1] 同矩阵  SA_20={r['sa_k']:.4f}  ✓")

    # 测试 2：独立随机矩阵 → SA_k ≈ k/d（接近 0）
    H2 = _norm(rng.standard_normal((N, d)))
    r2 = compute_subspace_alignment(H, H2, k=20)
    print(f"[测试2] 随机矩阵 SA_20={r2['sa_k']:.4f}  (期望接近 0)")

    # 测试 3：轻度扰动 → SA_k 介于两者之间
    H3 = _norm(H + 0.1 * rng.standard_normal((N, d)))
    r3 = compute_subspace_alignment(H, H3, k=20)
    assert r3["sa_k"] > r2["sa_k"], "扰动版 SA_k 应高于随机"
    print(f"[测试3] 扰动矩阵 SA_20={r3['sa_k']:.4f}  ✓ > 随机")

    # 测试 4：多尺度
    ms = compute_multiscale_sa(H, H3)
    print(f"[测试4] 多尺度: " + "  ".join(f"k{k}={ms[f'sa_k{k}']:.4f}" for k in (10, 20, 50)))

    print("\n所有测试通过 ✓")
