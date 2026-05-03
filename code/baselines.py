"""
Paper06 — Grassmann 统一框架
迁移性预测基线方法

实现方法：
    - LogME:     Log Marginal Evidence（Zhang et al. 2021）
    - H-score:   特征与标签相关性的谱度量（Bao et al. 2019）
    - TransRate: 基于编码率的迁移性估计（Huang et al. 2022）

用于 E2 实验与 SA_k 的对比评估。

参考实现：
    LogME    — github.com/thuml/LogME
    H-score  — 原论文 Algorithm 1
    TransRate — 原论文 Appendix
"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 1. LogME（Log Marginal Evidence）
# ─────────────────────────────────────────────────────────────

class LogME:
    """
    LogME 迁移性评分（Zhang et al., ICML 2021）。

    核心思想：在特征 H 上建立贝叶斯线性回归，
    最大化标签 y 的对数边际似然 log p(y | H)。

    LogME 越高 → 特征与标签的线性关系越紧密 → 迁移性越好。

    用法：
        score = LogME().fit(H, y)
    """

    def __init__(self, regression: bool = False):
        """
        Args:
            regression: True 用于回归任务，False 用于分类（逐类 one-vs-rest）
        """
        self.regression = regression

    def fit(self, H: np.ndarray, y: np.ndarray) -> float:
        """
        计算 LogME 评分。

        Args:
            H: (N, d) 特征矩阵（已 L2 归一化）
            y: (N,) 标签数组（分类为整数，回归为浮点）

        Returns:
            LogME 评分（越高越好）
        """
        if self.regression:
            return self._fit_single(H, y.astype(np.float64))
        else:
            classes = np.unique(y)
            scores = []
            for c in classes:
                y_binary = (y == c).astype(np.float64)
                scores.append(self._fit_single(H, y_binary))
            return float(np.mean(scores))

    def _fit_single(self, H: np.ndarray, y: np.ndarray) -> float:
        """单目标 LogME 计算（EM 算法优化 α, β）。"""
        N, d = H.shape
        H = H.astype(np.float64)
        y = y.astype(np.float64)

        # SVD of H
        U, s, Vt = np.linalg.svd(H, full_matrices=False)
        # s: (min(N,d),)

        # 初始化超参数
        alpha = 1.0
        beta  = 1.0
        evidence = -np.inf

        for _ in range(100):  # EM 迭代
            s2 = s ** 2

            # M 步：更新 w = (alpha I + beta H^T H)^{-1} beta H^T y
            # 利用 SVD 高效计算
            Uy = U.T @ y       # (min_dim,)
            m  = beta * s * Uy / (alpha + beta * s2)   # (min_dim,) — 对角矩阵简化

            # 计算对数边际似然
            # log p(y | H, alpha, beta) ≈ -N/2 log(2π) + N/2 log beta
            #   - beta/2 ||y - H m||^2 - alpha/2 ||m||^2
            #   + 1/2 log |alpha I + beta H^T H|^{-1}
            log_det = float(np.sum(np.log(alpha + beta * s2)))
            y_pred  = U @ (s * m)   # H w 的重构
            res_sq  = float(np.sum((y - y_pred) ** 2))
            new_evidence = (
                0.5 * d * np.log(alpha)
                + 0.5 * N * np.log(beta)
                - 0.5 * beta * res_sq
                - 0.5 * alpha * float(np.sum(m ** 2))
                - 0.5 * log_det
                - 0.5 * N * np.log(2 * np.pi)
            )

            # E 步：更新 alpha, beta
            gamma = float(np.sum(beta * s2 / (alpha + beta * s2)))
            alpha = gamma / (float(np.sum(m ** 2)) + 1e-10)
            beta  = (N - gamma) / (res_sq + 1e-10)

            if abs(new_evidence - evidence) < 1e-5 * abs(new_evidence):
                evidence = new_evidence
                break
            evidence = new_evidence

        return float(evidence / N)   # 归一化，与数据集大小无关


# ─────────────────────────────────────────────────────────────
# 2. H-score（Bao et al. 2019）
# ─────────────────────────────────────────────────────────────

def compute_hscore(H: np.ndarray, y: np.ndarray) -> float:
    """
    H-score 迁移性评分（Bao et al., AAAI 2019）。

    核心思想：
        类间协方差与类内协方差的比值：
        H-score = tr(Σ_B^{-1} Σ_W)（越大越好）

        其中 Σ_B = 类间协方差，Σ_W = 类内协方差（加正则化）

    Args:
        H: (N, d) 特征矩阵（已 L2 归一化）
        y: (N,) 整数标签

    Returns:
        H-score（越高 → 特征类间分离性越好 → 迁移性越好）
    """
    H = H.astype(np.float64)
    classes = np.unique(y)

    # 全局均值
    mu_global = H.mean(axis=0)   # (d,)

    # 类内协方差 Σ_W
    Sigma_W = np.zeros((H.shape[1], H.shape[1]))
    for c in classes:
        H_c = H[y == c]
        if len(H_c) < 2:
            continue
        mu_c = H_c.mean(axis=0)
        diff = H_c - mu_c[None, :]
        Sigma_W += diff.T @ diff

    n_total = len(H)
    Sigma_W = Sigma_W / n_total + 1e-4 * np.eye(H.shape[1])  # 正则化

    # 类间协方差 Σ_B
    Sigma_B = np.zeros_like(Sigma_W)
    for c in classes:
        H_c = H[y == c]
        n_c = len(H_c)
        mu_c = H_c.mean(axis=0)
        diff = (mu_c - mu_global)[:, None]
        Sigma_B += n_c * (diff @ diff.T)
    Sigma_B /= n_total

    # H-score = tr(Σ_W^{-1} Σ_B)
    try:
        Sigma_W_inv = np.linalg.inv(Sigma_W)
        score = float(np.trace(Sigma_W_inv @ Sigma_B))
    except np.linalg.LinAlgError:
        # 奇异时使用伪逆
        Sigma_W_inv = np.linalg.pinv(Sigma_W)
        score = float(np.trace(Sigma_W_inv @ Sigma_B))

    return score


# ─────────────────────────────────────────────────────────────
# 3. TransRate（Huang et al. 2022）
# ─────────────────────────────────────────────────────────────

def compute_transrate(
    H: np.ndarray,
    y: np.ndarray,
    eps: float = 0.5,
) -> float:
    """
    TransRate 迁移性评分（Huang et al., ICML 2022）。

    基于最大编码率差：
        TransRate = R(Z) - R(Z | C)

    其中 R(Z) = 编码率（整体），R(Z|C) = 每类条件编码率之和
    高 TransRate → 整体分散但类内紧凑 → 好的迁移表示

    Args:
        H:   (N, d) 特征矩阵
        y:   (N,) 整数标签
        eps: 编码精度（正则化参数，原论文建议 0.5）

    Returns:
        TransRate 评分（越高越好）
    """
    H = H.astype(np.float64)
    N, d = H.shape

    def coding_rate(Z: np.ndarray, eps_: float) -> float:
        """编码率 R(Z, ε²) = (d/2) log det(I + d/(N ε²) ZZᵀ)"""
        n, p = Z.shape
        C = (Z.T @ Z) / n
        _, s, _ = np.linalg.svd(C, full_matrices=False)
        # log det = Σ log(1 + p/(ε²) σᵢ)
        rate = 0.5 * float(np.sum(np.log(1.0 + p / (eps_ ** 2) * s)))
        return rate

    # 整体编码率
    R_total = coding_rate(H, eps)

    # 条件编码率
    classes = np.unique(y)
    R_cond = 0.0
    for c in classes:
        H_c = H[y == c]
        n_c = len(H_c)
        if n_c < 2:
            continue
        R_c = coding_rate(H_c, eps)
        R_cond += (n_c / N) * R_c

    return float(R_total - R_cond)


# ─────────────────────────────────────────────────────────────
# 4. Linear Probe Accuracy（ground truth）
# ─────────────────────────────────────────────────────────────

def linear_probe_accuracy(
    H_train: np.ndarray,
    y_train: np.ndarray,
    H_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> float:
    """
    线性探针精度（作为 E2 实验的 ground truth 之一）。

    使用 sklearn LogisticRegression，L2 正则化。

    Args:
        H_train/H_test: (N, d) 训练/测试特征
        y_train/y_test: (N,) 标签
        C:              正则化强度的倒数
        max_iter:       最大迭代次数

    Returns:
        top-1 准确率 ∈ [0, 1]
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    H_tr = scaler.fit_transform(H_train)
    H_te = scaler.transform(H_test)

    clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs",
                              multi_class="multinomial", n_jobs=-1)
    clf.fit(H_tr, y_train)
    return float(clf.score(H_te, y_test))


# ─────────────────────────────────────────────────────────────
# 5. 统一接口
# ─────────────────────────────────────────────────────────────

def compute_all_baselines(
    H: np.ndarray,
    y: np.ndarray,
    H_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> dict:
    """
    统一计算所有基线迁移性指标。

    Args:
        H:      (N, d) 训练集特征（已 L2 归一化）
        y:      (N,) 标签
        H_test: 测试集特征（计算 LP 准确率时需要）
        y_test: 测试集标签

    Returns:
        dict: {
            "logme":     float  LogME 评分
            "hscore":    float  H-score
            "transrate": float  TransRate
            "lp_acc":    float  Linear Probe 准确率（若提供测试集）
        }
    """
    result = {}

    result["logme"]     = LogME().fit(H, y)
    result["hscore"]    = compute_hscore(H, y)
    result["transrate"] = compute_transrate(H, y)

    if H_test is not None and y_test is not None:
        result["lp_acc"] = linear_probe_accuracy(H, y, H_test, y_test)

    return result


# ─────────────────────────────────────────────────────────────
# 5. STG：谱迁移差距（Spectral Transfer Gap）
#    双通道分解的谱维度指标
# ─────────────────────────────────────────────────────────────

def compute_stg(H_src: np.ndarray, H_tgt: np.ndarray) -> float:
    """
    谱迁移差距（STG）：源域与目标域归一化奇异值分布的 Wasserstein-1 距离。

    核心思想：
        SVD 将 H = U Σ V^T 拆分为谱维度（Σ）和方向维度（V）。
        STG 度量"能量分布有多不同"（谱通道），
        SA_k 度量"激活方向有多不同"（方向通道）。
        两者正交互补 —— 实验显示 corr(STG, SA_k) ≈ 0.3。

    数学定义：
        p_src = σ(H_src) / ||σ(H_src)||₁     归一化奇异值分布
        p_tgt = σ(H_tgt) / ||σ(H_tgt)||₁
        STG   = W₁(p_src, p_tgt) = ||CDF_src - CDF_tgt||₁

    返回值：
        float: STG ≥ 0；值越大表示谱分布差异越大

    参数：
        H_src: (N_src, d) 源域特征矩阵（L2 归一化）
        H_tgt: (N_tgt, d) 目标域特征矩阵（L2 归一化）
    """
    # 计算奇异值
    _, s_src, _ = np.linalg.svd(H_src, full_matrices=False)
    _, s_tgt, _ = np.linalg.svd(H_tgt, full_matrices=False)

    # 截断到共同维度（取较小的奇异值数量）
    k = min(len(s_src), len(s_tgt))
    s_src = s_src[:k]
    s_tgt = s_tgt[:k]

    # 归一化为概率分布
    p_src = s_src / (s_src.sum() + 1e-12)
    p_tgt = s_tgt / (s_tgt.sum() + 1e-12)

    # Wasserstein-1 距离 = L1 范数的 CDF 差
    cdf_src = np.cumsum(p_src)
    cdf_tgt = np.cumsum(p_tgt)
    w1 = float(np.sum(np.abs(cdf_src - cdf_tgt)))

    return w1


def compute_dual_channel_distance(
    H_src: np.ndarray,
    H_tgt: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    k: int = 20,
) -> dict:
    """
    双通道表示距离（Dual-Channel Representation Distance）。

    d_repr = α × STG + β × (1 - SA_k)

    返回：
        dict:
            "stg":    float   谱迁移差距
            "sa_k":   float   子空间对齐度（来自 metrics.subspace_alignment）
            "d_repr": float   加权双通道距离
            "d_dir":  float   方向差异分量 = 1 - SA_k
    """
    from metrics.subspace_alignment import compute_subspace_alignment
    sa_result = compute_subspace_alignment(H_src, H_tgt, k=k)
    sa_k = sa_result["sa_k"]
    stg  = compute_stg(H_src, H_tgt)
    d_dir = 1.0 - sa_k
    d_repr = alpha * stg + beta * d_dir

    return {
        "stg":    stg,
        "sa_k":   sa_k,
        "d_dir":  d_dir,
        "d_repr": d_repr,
    }


# ─────────────────────────────────────────────────────────────
# 单元测试
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    N, d, K = 300, 128, 5

    # 生成可分特征（高迁移性）
    H_good = np.zeros((N, d))
    y = np.repeat(np.arange(K), N // K)
    for c in range(K):
        H_good[y == c] = rng.standard_normal(d)[None, :] + 0.1 * rng.standard_normal(
            (N // K, d))
    H_good /= np.linalg.norm(H_good, axis=1, keepdims=True) + 1e-8

    # 生成不可分特征（低迁移性）
    H_bad = rng.standard_normal((N, d))
    H_bad /= np.linalg.norm(H_bad, axis=1, keepdims=True) + 1e-8

    for label, H in [("可分（高迁移）", H_good), ("随机（低迁移）", H_bad)]:
        r = compute_all_baselines(H, y)
        print(f"\n{label}:")
        for k, v in r.items():
            print(f"  {k:12s} = {v:.4f}")

    print("\n基线计算完成 ✓")
