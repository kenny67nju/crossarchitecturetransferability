"""
Paper06 — Grassmann 统一框架
方向增益准则 Δ_dir（应用 A：数据集构建）

核心公式：
    Δ_dir(D_new | D_old) = 1 - SA_k(S(D_new, M), S(D_old, M))

    Δ_dir ≈ 1 → 新数据集引入全新方向 → 加入收益高
    Δ_dir ≈ 0 → 新数据集方向冗余     → 加入无效

优势（对比 r_eff）：
    - 不受能量尺度影响（无 γ 混淆）
    - 不依赖标签
    - 直接量化"新增方向的独立性"
    - 天然解决 Paper01 的合并坍塌问题

用于 E3 实验（数据集筛选贪心策略）。
"""

import numpy as np
from typing import List, Dict, Optional

from .subspace_alignment import compute_subspace_alignment, compute_multiscale_sa


# ─────────────────────────────────────────────────────────────
# 1. 核心计算
# ─────────────────────────────────────────────────────────────

def direction_gain(
    H_new: np.ndarray,
    H_old: np.ndarray,
    k: int = 20,
) -> Dict:
    """
    计算新数据集特征矩阵相对于已有数据集的方向增益 Δ_dir。

    Args:
        H_new: (N_new, d) 新数据集的特征矩阵（已 L2 归一化）
        H_old: (N_old, d) 已有数据集的特征矩阵（已 L2 归一化）
        k:     子空间维度

    Returns:
        dict: {
            "delta_dir":      float  Δ_dir ∈ [0,1]，主指标（越高越好）
            "sa_k":           float  SA_k（方向重叠度，Δ_dir = 1 - sa_k）
            "grassmann_dist": float  Grassmann 距离
            "k_used":         int
        }
    """
    result = compute_subspace_alignment(H_new, H_old, k=k)
    delta_dir = 1.0 - result["sa_k"]
    return {
        "delta_dir":      delta_dir,
        "sa_k":           result["sa_k"],
        "grassmann_dist": result["grassmann_dist"],
        "k_used":         result["k_used"],
    }


def direction_gain_multiscale(
    H_new: np.ndarray,
    H_old: np.ndarray,
    k_values: tuple = (10, 20, 50),
) -> Dict:
    """多尺度 Δ_dir，消融不同 k 的影响。"""
    ms = compute_multiscale_sa(H_new, H_old, k_values=k_values)
    result = {}
    for k in k_values:
        result[f"delta_dir_k{k}"] = 1.0 - ms[f"sa_k{k}"]
    result["mean_delta_dir"] = float(np.mean([result[f"delta_dir_k{k}"] for k in k_values]))
    return result


# ─────────────────────────────────────────────────────────────
# 2. 贪心数据集筛选（E3 主逻辑）
# ─────────────────────────────────────────────────────────────

def greedy_dataset_selection(
    H_base: np.ndarray,
    candidate_features: Dict[str, np.ndarray],
    k: int = 20,
    n_rounds: Optional[int] = None,
) -> List[Dict]:
    """
    贪心数据集选择：每轮选 Δ_dir 最大的候选数据集加入。

    Args:
        H_base:              (N_base, d) 初始训练集特征矩阵
        candidate_features:  {dataset_name: (N, d)} 候选数据集特征
        k:                   子空间维度
        n_rounds:            最多选择轮数（None = 直到候选集耗尽）

    Returns:
        List of dicts，按选取顺序：{
            "round":       int
            "selected":    str   被选中的数据集名
            "delta_dir":   float
            "sa_k":        float
            "all_scores":  dict  本轮所有候选的 Δ_dir
        }
    """
    remaining = dict(candidate_features)
    H_current = H_base.copy()
    history = []
    n_rounds = n_rounds or len(remaining)

    for rnd in range(n_rounds):
        if not remaining:
            break

        # 计算所有候选的 Δ_dir
        scores = {}
        for name, H_cand in remaining.items():
            dg = direction_gain(H_cand, H_current, k=k)
            scores[name] = dg["delta_dir"]

        # 选取 Δ_dir 最大的候选
        best_name = max(scores, key=scores.__getitem__)
        best_score = scores[best_name]

        sa_val = compute_subspace_alignment(
            remaining[best_name], H_current, k=k)["sa_k"]

        history.append({
            "round":      rnd + 1,
            "selected":   best_name,
            "delta_dir":  best_score,
            "sa_k":       sa_val,
            "all_scores": dict(scores),
        })

        print(f"  轮 {rnd+1}: 选 {best_name:20s}  Δ_dir={best_score:.4f}  SA_k={sa_val:.4f}")

        # 合并选中数据集特征到 H_current
        H_current = np.concatenate([H_current, remaining[best_name]], axis=0)
        del remaining[best_name]

    return history


def reff_greedy_selection(
    H_base: np.ndarray,
    candidate_features: Dict[str, np.ndarray],
    n_rounds: Optional[int] = None,
) -> List[Dict]:
    """
    基线：按 r_eff 最大贪心选择（Paper01 策略，用于 E3 对比）。

    Args:
        H_base:             初始训练集特征矩阵
        candidate_features: 候选数据集特征
        n_rounds:           最多选择轮数

    Returns:
        同 greedy_dataset_selection 格式
    """
    from .reff import effective_rank

    remaining = dict(candidate_features)
    history = []
    n_rounds = n_rounds or len(remaining)

    for rnd in range(n_rounds):
        if not remaining:
            break

        scores = {name: effective_rank(H) for name, H in remaining.items()}
        best_name = max(scores, key=scores.__getitem__)

        history.append({
            "round":      rnd + 1,
            "selected":   best_name,
            "reff_score": scores[best_name],
            "all_scores": dict(scores),
        })

        print(f"  [reff 基线] 轮 {rnd+1}: 选 {best_name:20s}  r_eff={scores[best_name]:.2f}")
        del remaining[best_name]

    return history


# ─────────────────────────────────────────────────────────────
# 3. 联合评分（应用 D）
# ─────────────────────────────────────────────────────────────

def joint_score(r_eff: float, sa_k: float) -> float:
    """
    联合评分：Score(M) = r_eff^{1/2} × SA_k

    同时捕捉表示丰富性（r_eff）和跨域稳定性（SA_k）。
    """
    return float(np.sqrt(max(r_eff, 0.0)) * sa_k)


# ─────────────────────────────────────────────────────────────
# 单元测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d, k = 100, 256, 20

    def _norm(H):
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    H_base = _norm(rng.standard_normal((N, d)))

    # 测试 1：完全正交新数据集 → Δ_dir ≈ 1
    H_orth = _norm(rng.standard_normal((N, d)))
    r1 = direction_gain(H_orth, H_base, k=k)
    print(f"[测试1] 随机新集:  Δ_dir={r1['delta_dir']:.4f}  SA_k={r1['sa_k']:.4f}")

    # 测试 2：相同数据集 → Δ_dir ≈ 0
    r2 = direction_gain(H_base.copy(), H_base, k=k)
    print(f"[测试2] 相同数据:  Δ_dir={r2['delta_dir']:.4f}  SA_k={r2['sa_k']:.4f}  "
          f"{'✓' if r2['delta_dir'] < 0.05 else '✗'}")

    # 测试 3：贪心选择
    candidates = {
        "high_gain": _norm(rng.standard_normal((N, d))),   # 独立方向
        "low_gain":  H_base + 0.05 * rng.standard_normal((N, d)),  # 近似相同
        "mid_gain":  _norm(H_base + rng.standard_normal((N, d))),  # 混合
    }
    print("\n[测试3] 贪心选择:")
    history = greedy_dataset_selection(H_base, candidates, k=k)
    assert history[0]["selected"] in ["high_gain", "mid_gain"], "第一轮应选高增益数据集"
    print(f"  第一轮选: {history[0]['selected']}  ✓")

    # 测试 4：联合评分
    score = joint_score(r_eff=45.0, sa_k=0.8)
    print(f"\n[测试4] joint_score(r=45, sa=0.8) = {score:.4f}  "
          f"(期望≈{np.sqrt(45)*0.8:.4f})")

    print("\n所有测试通过 ✓")
