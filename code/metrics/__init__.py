"""
Paper06 metrics package — Grassmann 统一框架核心度量

模块：
    subspace_alignment  — SA_k：子空间对齐度（主指标）
    reff                — r_eff：有效秩与坍塌预测（主定理）
    grassmann           — Grassmann 几何工具（距离、主角度、Karcher 均值）
    direction_gain      — Δ_dir：数据集方向增益准则
"""
from .subspace_alignment import compute_subspace_alignment, compute_multiscale_sa
from .reff import (effective_rank, effective_rank_from_singular_values,
                   predict_reff_collapse,
                   reff_bounds, reff_theoretical_prediction)
from .ssr import ssr_dagger
from .grassmann import grassmann_distance, principal_angles, GrassmannLoss
from .direction_gain import direction_gain
