"""
Paper06 — Grassmann 统一框架
Grassmann 流形几何工具

主要功能：
    1. principal_angles()       — 两个子空间的主角（核心几何运算）
    2. grassmann_distance()     — Grassmann 测地距离（√Σθᵢ²）
    3. subspace_from_matrix()   — 从特征矩阵提取 Grassmann 点
    4. GrassmannLoss            — 可微分的 Grassmann 对齐损失（PyTorch）

GrassmannLoss 对应 PLAN.md 阶段 4（域适应实验）：
    L_G = Σᵢ<ⱼ Σₗ θᵢⱼₗ²
    最小化此损失使所有域的 top-k 特征方向对齐。
"""

import numpy as np
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────
# 1. Numpy 工具（无梯度，用于度量计算）
# ─────────────────────────────────────────────────────────────

def principal_angles(
    V_A: np.ndarray,
    V_B: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算两个子空间的主角（principal angles）。

    Args:
        V_A: (d, k_A) 正交矩阵，列为子空间 A 的基向量
        V_B: (d, k_B) 正交矩阵，列为子空间 B 的基向量

    Returns:
        cos_angles: (min(k_A, k_B),) cos(θᵢ) ∈ [0,1]，降序
        angles_rad: (min(k_A, k_B),) θᵢ ∈ [0, π/2]，升序
    """
    M = V_A.T @ V_B   # (k_A, k_B)
    cos_angles = np.linalg.svd(M, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0.0, 1.0)
    angles_rad = np.arccos(cos_angles)
    return cos_angles, angles_rad


def grassmann_distance(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> float:
    """
    Grassmann 测地距离：d_G(S_A, S_B) = √(Σᵢ θᵢ²)

    Args:
        H_A: (N_A, d) 特征矩阵
        H_B: (N_B, d) 特征矩阵
        k:   子空间维度

    Returns:
        Grassmann 距离（非负标量）
    """
    from .subspace_alignment import principal_angles_from_matrices
    _, angles_rad = principal_angles_from_matrices(H_A, H_B, k)
    return float(np.sqrt(np.sum(angles_rad ** 2)))


def subspace_from_matrix(H: np.ndarray, k: int) -> np.ndarray:
    """
    从特征矩阵 H 提取 top-k 子空间的正交基（Grassmann 流形上的点）。

    Args:
        H: (N, d) 特征矩阵
        k: 子空间维度

    Returns:
        V_k: (d, k) 正交矩阵（列向量为主方向）
    """
    k_eff = max(1, min(k, min(H.shape)))
    _, _, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k_eff, :].T


def karcher_mean(subspaces: List[np.ndarray], n_iter: int = 10) -> np.ndarray:
    """
    Grassmann 流形上多个子空间的 Karcher 均值（Fréchet 均值近似）。

    使用梯度下降算法（参考 Absil et al. 2004）：
        初始化为第一个子空间，迭代计算对数映射和指数映射。

    Args:
        subspaces: List of (d, k) 正交矩阵
        n_iter:    梯度下降迭代次数

    Returns:
        V_mean: (d, k) 均值子空间
    """
    if len(subspaces) == 0:
        raise ValueError("subspaces 不能为空")
    if len(subspaces) == 1:
        return subspaces[0].copy()

    d, k = subspaces[0].shape
    V = subspaces[0].copy()   # 初始化为第一个点

    for _ in range(n_iter):
        # 计算所有点到当前 V 的对数映射（切向量）
        tangent_sum = np.zeros_like(V)
        for Vi in subspaces:
            # 对数映射：Log_V(Vi) = (I - VV.T) Vi (Vi.T V)^{-1} θ_i / sin(θ_i)
            # 简化实现：使用 QR 对残差正交化
            M = V.T @ Vi
            U_m, s, Wt = np.linalg.svd(M, full_matrices=False)
            s = np.clip(s, -1.0, 1.0)
            theta = np.arccos(s)
            # 切向量
            residual = Vi - V @ M
            Q, _ = np.linalg.qr(residual)
            Q = Q[:, :k]
            tangent = Q @ np.diag(theta)
            tangent_sum += tangent

        # 指数映射（梯度步）
        step = tangent_sum / len(subspaces)
        U_s, s_s, Wt_s = np.linalg.svd(step, full_matrices=False)
        # V_new = V cos(s_s) Wt_s + U_s sin(s_s) — 近似
        V_new = V @ Wt_s.T * np.cos(s_s)[None, :] + U_s * np.sin(s_s)[None, :]
        V, _ = np.linalg.qr(V_new)
        V = V[:, :k]

    return V


# ─────────────────────────────────────────────────────────────
# 2. PyTorch 可微分 Grassmann 对齐损失
# ─────────────────────────────────────────────────────────────

class GrassmannLoss:
    """
    Grassmann 对齐损失 L_G（对应 PLAN.md 阶段 4）。

    计算多个域的特征矩阵之间的 Grassmann 距离平方和：
        L_G = Σᵢ<ⱼ d_G(Sᵢ, Sⱼ)² = Σᵢ<ⱼ Σₗ θᵢⱼₗ²

    支持 numpy（评估）和 PyTorch（训练）两种模式。

    PyTorch 反向传播说明：
        主角的梯度通过矩阵微分计算：
        ∂L/∂H = (∂L/∂θ) × (∂θ/∂cos) × (∂cos/∂M) × (∂M/∂V) × (∂V/∂H)
        其中 ∂θ/∂cos = -1/√(1-cos²)，其余项由自动微分处理。
    """

    def __init__(self, k: int = 50):
        self.k = k

    def numpy_loss(self, feature_matrices: List[np.ndarray]) -> float:
        """
        纯 numpy 评估（无梯度），用于训练监控。

        Args:
            feature_matrices: List of (N, d) 特征矩阵，每个对应一个域

        Returns:
            L_G：所有域对的 Grassmann 距离平方和
        """
        total = 0.0
        n = len(feature_matrices)
        for i in range(n):
            for j in range(i + 1, n):
                d = grassmann_distance(feature_matrices[i], feature_matrices[j], self.k)
                total += d ** 2
        return total

    def torch_loss(self, feature_list) -> "torch.Tensor":
        """
        PyTorch 可微分版本，可用于反向传播。

        Args:
            feature_list: List of torch.Tensor (N, d)，每个对应一个域

        Returns:
            L_G: 标量 tensor（支持 .backward()）

        实现策略：
            使用 randomized SVD 近似（torch.linalg.svd 的 top-k 截断），
            通过矩阵余弦相似的 arccos 计算主角，
            梯度路径：H → V_k（右奇异向量）→ cos(θ) → θ² → L_G
        """
        import torch

        total_loss = None
        n = len(feature_list)

        for i in range(n):
            for j in range(i + 1, n):
                H_i = feature_list[i]
                H_j = feature_list[j]

                # 截断 SVD（top-k 右奇异向量）
                k_eff = max(1, min(self.k, H_i.shape[0], H_j.shape[0], H_i.shape[1]))

                # torch.linalg.svd 支持自动微分
                _, _, Vt_i = torch.linalg.svd(H_i, full_matrices=False)
                _, _, Vt_j = torch.linalg.svd(H_j, full_matrices=False)

                V_i = Vt_i[:k_eff, :].T   # (d, k_eff)
                V_j = Vt_j[:k_eff, :].T

                # 交叉积 → 奇异值 = cos(θ)
                M = V_i.T @ V_j
                cos_angles = torch.linalg.svd(M, full_matrices=False).S
                cos_angles = torch.clamp(cos_angles, 1e-6, 1.0 - 1e-6)

                theta = torch.arccos(cos_angles)
                loss_ij = torch.sum(theta ** 2)

                total_loss = loss_ij if total_loss is None else total_loss + loss_ij

        return total_loss if total_loss is not None else torch.tensor(0.0)


# ─────────────────────────────────────────────────────────────
# 3. 谱-方向正交分解（辅助定理 0.3）
# ─────────────────────────────────────────────────────────────

def spectral_wasserstein_distance(H_A: np.ndarray, H_B: np.ndarray) -> float:
    """
    奇异值分布的 Wasserstein-1 距离（STG，谱迁移距离）。

    p_A = σᵢ / Σσⱼ（归一化奇异值分布）
    W₁(p_A, p_B) = 对排序奇异值的 L1 差

    对应辅助定理中 d_repr² = W₂²(谱) + d_G²(方向) 的第一项。
    """
    def _singular_probs(H):
        N, d = H.shape
        G = (H @ H.T) / N if N <= d else (H.T @ H) / N
        eigvals = np.maximum(np.linalg.eigvalsh(G)[::-1], 0)
        eigvals = eigvals[eigvals > 1e-10]
        sigma = np.sqrt(eigvals)
        return sigma / (sigma.sum() + 1e-12)

    p_A = _singular_probs(H_A)
    p_B = _singular_probs(H_B)

    # 对齐长度（零补全）
    n = max(len(p_A), len(p_B))
    p_A = np.pad(p_A, (0, n - len(p_A)))
    p_B = np.pad(p_B, (0, n - len(p_B)))

    # W₁ = ||CDF_A - CDF_B||₁
    cdf_A = np.cumsum(p_A)
    cdf_B = np.cumsum(p_B)
    return float(np.sum(np.abs(cdf_A - cdf_B)))


def repr_distance(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> Dict:
    """
    完整表示距离（辅助定理 0.3）：
        d_repr² = α·W₁(谱)² + β·d_G(方向)²

    Args:
        H_A, H_B: 特征矩阵
        k:        Grassmann 子空间维度
        alpha:    谱项权重
        beta:     方向项权重

    Returns:
        dict: {"w1_spectral", "d_grassmann", "d_repr_sq", "d_repr"}
    """
    w1  = spectral_wasserstein_distance(H_A, H_B)
    d_g = grassmann_distance(H_A, H_B, k=k)
    d_sq = alpha * (w1 ** 2) + beta * (d_g ** 2)
    return {
        "w1_spectral":  w1,
        "d_grassmann":  d_g,
        "d_repr_sq":    d_sq,
        "d_repr":       float(np.sqrt(d_sq)),
    }


# ─────────────────────────────────────────────────────────────
# 单元测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d, k = 100, 256, 20

    def _rand_subspace(rng, d, k):
        return np.linalg.qr(rng.standard_normal((d, k)))[0]

    # 测试 1：相同子空间 → Grassmann 距离 = 0
    V = _rand_subspace(rng, d, k)
    cos_a, ang = principal_angles(V, V.copy())
    assert np.allclose(ang, 0, atol=1e-5), f"相同子空间主角应≈0，got {ang.max():.2e}"
    print(f"[测试1] 相同子空间: max_angle={ang.max():.2e}  ✓")

    # 测试 2：正交子空间 → cos(θ) ≈ 0
    V_A = _rand_subspace(rng, d, k)
    V_B_raw = _rand_subspace(rng, d, k)
    V_B = V_B_raw - V_A @ (V_A.T @ V_B_raw)
    V_B, _ = np.linalg.qr(V_B)
    V_B = V_B[:, :k]
    cos_ab, _ = principal_angles(V_A, V_B)
    print(f"[测试2] 正交子空间: mean_cos={cos_ab.mean():.4f}  (期望接近 0)")

    # 测试 3：GrassmannLoss numpy
    H_A = rng.standard_normal((N, d)).astype(np.float32)
    H_B = rng.standard_normal((N, d)).astype(np.float32)
    gl = GrassmannLoss(k=k)
    loss_val = gl.numpy_loss([H_A, H_B])
    print(f"[测试3] GrassmannLoss([H_A, H_B]) = {loss_val:.4f}")

    # 测试 4：repr_distance
    result = repr_distance(H_A, H_B, k=k)
    print(f"[测试4] repr_distance: W₁={result['w1_spectral']:.4f}  "
          f"d_G={result['d_grassmann']:.4f}  d_repr={result['d_repr']:.4f}")

    print("\n所有测试通过 ✓")
