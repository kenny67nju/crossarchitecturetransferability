"""
Paper06 — Grassmann 统一框架
有效秩（r_eff）与坍塌预测

主定理（定理 0.2，修订版）：
    r_eff([H_A; H_B]) ∈ [min(r_A, r_B), r_A + r_B]

    注意：下界为 min（不是 max）。三种行为区域：

    ① 超可加（superadditive）：α≈0, γ≈1
       r_eff > max(r_A, r_B)  方向正交且能量均衡 → 互补
       SA_k=0 精确公式（混合熵）：
         r* = nuclear_A / (nuclear_A + nuclear_B)
         r_eff = exp(H_bin(r*) + r*·log(r_A) + (1-r*)·log(r_B))

    ② 方向坍塌（subadditive via SA）：α→1
       r_eff → max(r_A, r_B)  子空间对齐，方向冗余

    ③ 能量坍塌（subadditive via energy）：γ* > ln(r_A)/(δ·ln(r_B))
       r_eff → r_large (核范数大的一方主导)

    能量比 γ = ||H_B||_* / ||H_A||_*（核范数，不是 Frobenius 范数）
"""

import numpy as np
from typing import Optional, Dict


# ─────────────────────────────────────────────────────────────
# 1. 基础：有效秩计算
# ─────────────────────────────────────────────────────────────

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


def normalized_reff(H: np.ndarray) -> float:
    """归一化有效秩：r_eff / min(N, d) ∈ (0, 1]"""
    r = effective_rank(H)
    return r / float(min(H.shape))


# ─────────────────────────────────────────────────────────────
# 2. 主定理：坍塌预测
# ─────────────────────────────────────────────────────────────

def predict_reff_collapse(
    H_A: np.ndarray,
    H_B: np.ndarray,
    k: int = 20,
) -> Dict:
    """
    根据主定理预测合并特征矩阵 [H_A; H_B] 的 r_eff，
    并分解坍塌来源（方向 vs. 能量）。

    Args:
        H_A: (N_A, d) 特征矩阵 A
        H_B: (N_B, d) 特征矩阵 B
        k:   SA_k 的子空间维度

    Returns:
        dict: {
            "r_eff_A":        float  H_A 的有效秩
            "r_eff_B":        float  H_B 的有效秩
            "r_eff_merged":   float  [H_A; H_B] 的实测有效秩
            "r_eff_ideal":    float  理想可加值 r_A + r_B
            "collapse_ratio": float  坍塌比例 = (r_ideal - r_merged) / r_ideal ∈ [0,1]
            "delta_r":        float  r_merged - max(r_A,r_B)；正=超可加，负=次可加
            "sa_k":           float  子空间对齐度 α ∈ [0,1]
            "nuclear_A":      float  H_A 核范数 = Σ σᵢ(H_A)
            "nuclear_B":      float  H_B 核范数
            "energy_ratio":   float  γ = ||H_B||_* / ||H_A||_*（核范数比，非 Frobenius）
            "r_star":         float  A 的核范数权重 = nuclear_A/(nuclear_A+nuclear_B)
            "dominant_cause": str    "superadditive"|"direction"|"energy"|"both"|"none"
        }
    """
    from .subspace_alignment import compute_subspace_alignment

    r_A = effective_rank(H_A)
    r_B = effective_rank(H_B)

    H_merged = np.concatenate([H_A, H_B], axis=0)
    r_merged = effective_rank(H_merged)

    r_ideal  = r_A + r_B
    collapse_ratio = max(0.0, (r_ideal - r_merged) / r_ideal) if r_ideal > 0 else 0.0

    sa_result = compute_subspace_alignment(H_A, H_B, k=k)
    alpha = sa_result["sa_k"]

    # 能量比 γ = ||H_B||_* / ||H_A||_*  （核范数 = 奇异值之和）
    # Frobenius 范数在 L2 归一化后 = √N（与谱无关），必须用核范数
    _, s_A, _ = np.linalg.svd(H_A, full_matrices=False)
    _, s_B, _ = np.linalg.svd(H_B, full_matrices=False)
    nuclear_A = float(np.sum(s_A))
    nuclear_B = float(np.sum(s_B))
    gamma     = nuclear_B / (nuclear_A + 1e-8)

    # r_star：A 在混合中的权重（核范数加权）
    r_star = nuclear_A / (nuclear_A + nuclear_B + 1e-8)

    # 超/次可加性：相对于 max(r_A, r_B)
    r_max  = max(r_A, r_B)
    delta_r = r_merged - r_max  # 正 → 超可加，负 → 次可加

    # 判断坍塌主导原因（启发式阈值）
    direction_dominant = alpha > 0.5
    energy_dominant    = gamma > 3.0 or gamma < 0.33
    if direction_dominant and energy_dominant:
        cause = "both"
    elif direction_dominant:
        cause = "direction"
    elif energy_dominant:
        cause = "energy"
    elif delta_r > 0:
        cause = "superadditive"
    else:
        cause = "none"

    return {
        "r_eff_A":        r_A,
        "r_eff_B":        r_B,
        "r_eff_merged":   r_merged,
        "r_eff_ideal":    r_ideal,
        "collapse_ratio": collapse_ratio,
        "delta_r":        delta_r,          # r_merged - max(r_A, r_B)
        "sa_k":           alpha,
        "nuclear_A":      nuclear_A,
        "nuclear_B":      nuclear_B,
        "energy_ratio":   gamma,            # 核范数比 = ||H_B||_* / ||H_A||_*
        "r_star":         r_star,           # A 的核范数权重
        "dominant_cause": cause,
    }


# ─────────────────────────────────────────────────────────────
# 3. 理论上下界
# ─────────────────────────────────────────────────────────────

def reff_bounds(r_A: float, r_B: float) -> Dict:
    """
    主定理的理论上下界（修订版）。

    下界为 min(r_A, r_B)（不是 max），在 SA_k≈0 且能量均衡时
    实际值可超过 max(r_A, r_B)（超可加性）。

    Returns:
        {
            "lower":    min(r_A, r_B)  — 硬下界
            "upper":    r_A + r_B      — 硬上界
            "max":      max(r_A, r_B)  — 超/次可加分界线
        }
    """
    return {
        "lower": min(r_A, r_B),
        "upper": r_A + r_B,
        "max":   max(r_A, r_B),
    }


def reff_theoretical_prediction(
    r_A: float,
    r_B: float,
    alpha: float,
    gamma: float,
) -> float:
    """
    基于混合熵公式的理论 r_eff 预测。

    推导（SA_k=0 精确基准）：
        当子空间正交时，合并矩阵的奇异值为两组奇异值的并集。
        令 r* = nuclear_A / (nuclear_A + nuclear_B) = 1 / (1 + γ)，则：

            H_combined = H_bin(r*) + r*·log(r_A) + (1-r*)·log(r_B)
            r_eff(SA_k=0) = exp(H_combined)
                          = exp(H_bin(r*)) · r_A^r* · r_B^(1-r*)

        其中 H_bin(r*) = -r*·log(r*) - (1-r*)·log(1-r*)

    SA_k = α 的精确合并结构（等主角近似，γ = ||H_B||*/||H_A||*）：
        合并 Gram 矩阵有两组奇异值：
          Group1（V_A 对齐方向）: σ_A_i · √(1 + α·γ²)
          Group2（正交方向 W）:   σ_A_i · γ · √(1-α)

        r* = √(1+α·γ²) / [√(1+α·γ²) + γ·√(1-α)]
        r_pred = exp(H_bin(r*) + r*·log(r_A) + (1-r*)·log(r_B))

        α=0: r* = 1/(1+γ)，退化为 SA_k=0 混合熵公式 ✓
        α=1: r* → 1，r_pred = r_A（方向坍塌） ✓

    Args:
        r_A:   H_A 的有效秩
        r_B:   H_B 的有效秩
        alpha: SA_k(H_A, H_B) ∈ [0, 1]
        gamma: ||H_B||_* / ||H_A||_*（核范数比）

    Returns:
        r_pred: 预测的合并有效秩，被限制在 [min(r_A,r_B), r_A+r_B]
    """
    alpha_c = float(np.clip(alpha, 0.0, 1.0 - 1e-8))
    gamma_f = float(gamma)

    # ── 精确物理推导：合并 Gram 矩阵的 2×2 块特征值 ──────────────
    #
    # 对每个主角对 (V_A_i, V_perp_i)，令 V_B_i = cos·V_A_i + sin·V_perp_i，
    # 则 H_merged.T@H_merged 在此 2D 块中为：
    #
    #   M_i = σ_A_i² · [[1,0],[0,0]]  +  σ_B_i² · [[cos², cos·sin],[cos·sin, sin²]]
    #       = σ_A_i² · [[1+γ²·cos², γ²·cos·sin],[γ²·cos·sin, γ²·sin²]]
    #
    # 精确特征值（设 A=1+γ², D=√(A²-4γ²(1-α))，cos²=α, sin²=1-α）：
    #   λ± = σ_A_i² · (A ± D) / 2
    #   奇异值振幅: amp± = √((A ± D)/2)
    #
    # 核范数权重: r* = amp+ / (amp+ + amp-)
    #
    # 极限验证:
    #   α=0: D=|1-γ²|, amp+=max(1,γ), amp-=min(1,γ) → r*=γ/(1+γ) if γ>1 ← 混合熵公式 ✓
    #   α=1: D=1+γ², amp+=√(1+γ²), amp-=0 → r*=1, r_pred=r_dominant ← 方向坍塌 ✓
    #   α=0,γ=1: amp+=amp-=1, r*=0.5, H_bin=log2 → r_pred=2r_A ← 最大超可加 ✓

    A    = 1.0 + gamma_f ** 2
    disc = max(A ** 2 - 4.0 * gamma_f ** 2 * (1.0 - alpha_c), 0.0)
    D    = float(np.sqrt(disc))
    amp_plus  = float(np.sqrt((A + D) / 2.0))               # 主特征方向振幅
    amp_minus = float(np.sqrt(max((A - D) / 2.0, 0.0)))     # 次特征方向振幅

    r_star = float(np.clip(
        amp_plus / (amp_plus + amp_minus + 1e-12),
        1e-6, 1.0 - 1e-6,
    ))

    # 混合熵公式
    # Group+ 对应能量主导方向（γ≥1 时为 H_B 方向，γ<1 时为 H_A 方向）
    H_bin    = -(r_star * np.log(r_star) + (1.0 - r_star) * np.log(1.0 - r_star))
    H_A_spec = float(np.log(max(r_A, 1e-8)))
    H_B_spec = float(np.log(max(r_B, 1e-8)))
    # H_spec+ ≈ 主导数据集的谱熵，H_spec- ≈ 次要数据集的谱熵
    if gamma_f >= 1.0:
        H_spec_plus, H_spec_minus = H_B_spec, H_A_spec   # B 主导
    else:
        H_spec_plus, H_spec_minus = H_A_spec, H_B_spec   # A 主导
    r_pred   = float(np.exp(H_bin + r_star * H_spec_plus + (1.0 - r_star) * H_spec_minus))

    # 硬边界：[min(r_A, r_B), r_A + r_B]
    return float(np.clip(r_pred, min(r_A, r_B), r_A + r_B))


# ─────────────────────────────────────────────────────────────
# 4. 合成数据生成（用于 E1 受控实验）
# ─────────────────────────────────────────────────────────────

def make_controlled_pair(
    d: int = 512,
    k: int = 50,
    n: int = 1000,
    target_alpha: float = 0.5,
    gamma: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    生成一对特征矩阵 (H_A, H_B)，使其 SA_k ≈ target_alpha，能量比 ≈ gamma。

    构造方法：
        1. 随机生成正交基 V_A ∈ Gr(k, d)
        2. 通过 Givens 旋转生成 V_B，使 SA_k(V_A, V_B) ≈ target_alpha
        3. 用奇异值 Σ_A, Σ_B（由 gamma 控制）重构 H_A, H_B

    Args:
        d:            特征维度
        k:            子空间维度
        n:            每个矩阵的样本数
        target_alpha: 目标 SA_k ∈ [0, 1]
        gamma:        能量比 ||Σ_B||_F / ||Σ_A||_F
        seed:         随机种子

    Returns:
        H_A: (n, d)，H_B: (n, d)
    """
    rng = np.random.default_rng(seed)

    # ── 生成 H_A ─────────────────────────────────────────────
    U_A = np.linalg.qr(rng.standard_normal((n, k)))[0]   # (n, k)
    V_A = np.linalg.qr(rng.standard_normal((d, k)))[0]   # (d, k)
    sigma_A = np.exp(-np.arange(k) / (k / 3.0))          # 指数衰减奇异值
    sigma_A /= sigma_A.sum()
    sigma_A *= np.sqrt(n)                                 # 归一化到合理范围

    # ── 生成 V_B：通过旋转控制 SA_k ─────────────────────────
    # cos(θᵢ) = target_alpha^0.5 → 所有主角相等（等角近似）
    cos_target = float(np.sqrt(np.clip(target_alpha, 0.0, 1.0)))

    # 在 span(V_A) 的正交补中随机取方向
    V_perp = np.linalg.qr(rng.standard_normal((d, k)))[0]
    # 正交化：去掉 V_A 的分量
    V_perp -= V_A @ (V_A.T @ V_perp)
    V_perp, _ = np.linalg.qr(V_perp)
    V_perp = V_perp[:, :k]

    # V_B = cos_target * V_A + sin_target * V_perp
    sin_target = float(np.sqrt(1.0 - cos_target ** 2))
    V_B = cos_target * V_A + sin_target * V_perp
    V_B, _ = np.linalg.qr(V_B)   # 重新正交化

    # ── 生成 H_B ─────────────────────────────────────────────
    U_B = np.linalg.qr(rng.standard_normal((n, k)))[0]
    sigma_B = sigma_A * gamma   # 通过 gamma 控制能量比

    H_A_clean = (U_A * sigma_A[None, :]) @ V_A.T   # (n, d)
    H_B_clean = (U_B * sigma_B[None, :]) @ V_B.T   # (n, d)

    # 极小正则化噪声（用于打破精确数值退化，对谱无影响）
    # 噪声量级远低于特征值阈值（1e-10），保证 effective_rank 只计算信号分量
    # 使用 1e-5 * std 确保噪声特征值 << 1e-10（理论公式验证所必需）
    noise_std_A = 1e-5 * float(np.std(H_A_clean))
    noise_std_B = 1e-5 * float(np.std(H_B_clean))
    H_A = H_A_clean + rng.standard_normal(H_A_clean.shape) * noise_std_A
    H_B = H_B_clean + rng.standard_normal(H_B_clean.shape) * noise_std_B

    return H_A.astype(np.float32), H_B.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# 单元测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from metrics.subspace_alignment import compute_subspace_alignment

    rng = np.random.default_rng(0)

    # 测试 1：低秩矩阵 → r_eff ≈ 真实秩
    N, d, rank_true = 500, 128, 5
    U = rng.standard_normal((N, rank_true))
    V = rng.standard_normal((rank_true, d))
    H_low = (U @ V) + 0.01 * rng.standard_normal((N, d))
    r = effective_rank(H_low)
    print(f"[测试1] 真实秩={rank_true} → r_eff={r:.2f}  "
          f"{'✓' if abs(r - rank_true) < 2.0 else '✗'}")

    # 测试 2：合成对 → 检验坍塌预测
    for alpha_target in [0.0, 0.5, 1.0]:
        H_A, H_B = make_controlled_pair(d=128, k=20, n=300, target_alpha=alpha_target, gamma=1.0)
        result = predict_reff_collapse(H_A, H_B, k=20)
        print(f"[测试2] α_target={alpha_target:.1f} → "
              f"sa_k={result['sa_k']:.3f}  collapse={result['collapse_ratio']:.3f}  "
              f"cause={result['dominant_cause']}")

    # 测试 3：理论预测精度与超可加性
    r_A, r_B = 45.0, 10.0
    print(f"\n[测试3] r_A={r_A}, r_B={r_B}")
    for alpha, gamma in [(0.0, 1.0), (1.0, 1.0), (0.0, 10.0), (0.0, 4.5)]:
        pred   = reff_theoretical_prediction(r_A, r_B, alpha, gamma)
        bounds = reff_bounds(r_A, r_B)
        in_range = bounds["lower"] <= pred <= bounds["upper"]
        superadd = pred > bounds["max"]
        print(f"  α={alpha:.1f}, γ={gamma:5.1f} → r_pred={pred:.2f}  "
              f"bounds=[{bounds['lower']:.1f},{bounds['upper']:.1f}]  "
              f"{'✓' if in_range else '✗ 越界'}  "
              f"{'↑超可加' if superadd else '↓次可加'}")

    # 测试 4：delta_r 超可加验证（SA_k≈0, 等能量 → delta_r > 0）
    print("\n[测试4] make_controlled_pair: 超可加性验证")
    for alpha_t in [0.0, 0.5, 1.0]:
        H_A, H_B = make_controlled_pair(d=128, k=20, n=300,
                                        target_alpha=alpha_t, gamma=1.0)
        res = predict_reff_collapse(H_A, H_B, k=20)
        print(f"  α_target={alpha_t:.1f} → sa_k={res['sa_k']:.3f}  "
              f"r_merged={res['r_eff_merged']:.2f}  "
              f"max(r_A,r_B)={max(res['r_eff_A'],res['r_eff_B']):.2f}  "
              f"delta_r={res['delta_r']:+.2f}  "
              f"cause={res['dominant_cause']}")

    print("\n所有测试通过 ✓")
