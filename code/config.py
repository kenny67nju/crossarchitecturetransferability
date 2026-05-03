"""
Paper06 — Grassmann 统一框架：数据-模型二元性
全局实验配置

关联论文：Paper01（数据集评估）、Paper05（模型泛化谱度量）
核心对象：S_k(D, M) ∈ Gr(k, d) — 每个（数据集, 模型）对在 Grassmann 流形上的点
"""
from pathlib import Path

# ── 路径 ──────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "data"
RESULT_DIR = ROOT / "results"
FIG_DIR    = ROOT / "figures"

for d in [DATA_DIR, RESULT_DIR, FIG_DIR]:
    d.mkdir(exist_ok=True)

# ── 通用参数 ──────────────────────────────────────────────────
BATCH_SIZE  = 256
N_SAMPLES   = 2000    # 每个数据集提取的特征样本数（mini 实验用 500）
SEED        = 42

# ── Grassmann 子空间维度 k ─────────────────────────────────────
# 主实验使用 k=20，消融使用 K_VALUES
K_DEFAULT   = 20
K_VALUES    = (10, 20, 50)   # 多尺度消融

# ── 快速调试模式 ───────────────────────────────────────────────
DEBUG_MODE  = False
if DEBUG_MODE:
    N_SAMPLES = 300

# ═══════════════════════════════════════════════════════════════
# E1：r_eff 坍塌定理验证（合成 + 真实数据集对）
# 基于修订定理（v2）重新设计：
#   下界为 min，三行为区域：超可加 / 方向坍塌 / 能量坍塌
#   核心新指标：delta_r = r_merged - max(r_A, r_B)
#   能量比 γ 用核范数（非 Frobenius 范数）
# ═══════════════════════════════════════════════════════════════

# 合成实验网格（v2）
# α 网格在 0 附近加密 → 精确捕捉超可加性边界
# γ 网格覆盖能量坍塌阈值（理论 γ* ≈ 3.0 ~ 10.0）
E1_ALPHA_GRID  = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]   # SA_k；在 0 附近加密
E1_GAMMA_GRID  = [0.1, 0.33, 0.5, 1.0, 2.0, 3.0, 10.0]        # 核范数比 γ；覆盖阈值
E1_SYNTH_D     = 512   # 特征维度
E1_SYNTH_K     = 50    # 子空间维度（决定 r_eff 量级）
E1_SYNTH_N     = 1000  # 每个矩阵的样本数
E1_N_REPEAT    = 5     # 每个 (α, γ) 重复次数（消除随机误差）

# 三行为区域预期（用于测试用例标注）
# 区域①超可加：α < 0.1  AND  0.5 ≤ γ ≤ 2.0  → delta_r > 0
# 区域②方向坍塌：α > 0.6                       → delta_r < 0
# 区域③能量坍塌：γ > 3.0 OR γ < 0.33           → delta_r << 0

# 真实数据集对（覆盖三行为区域）
# 注：基于已缓存特征 resnet50 {cifar10, stl10, svhn}
#                      vit_b16  {cifar10, stl10, svhn}
E1_REAL_PAIRS = [
    # 区域②预期（方向坍塌）——同类视觉数据集，SA_k 高
    ("cifar10",  "svhn"),    # Paper01 反例：45+10→20，SA_k 驱动
    ("cifar10",  "stl10"),   # CIFAR-10 的高分辨率版本，高相似度
    # 区域①预期（超可加）——不同语义域，SA_k 低
    ("svhn",     "stl10"),   # 街景数字 vs 物体，域差异大
]

# 真实对实验使用的探针（已有缓存特征）
E1_PROBES = ["resnet50", "vit_b16"]

# ═══════════════════════════════════════════════════════════════
# E2：SA_k 作为迁移性预测指标（主实验）
# ═══════════════════════════════════════════════════════════════

# 探针模型（12 架构，覆盖 CNN / ViT / 混合）
E2_PROBES_MINI = ["resnet50", "vit_b16"]
E2_PROBES_FULL = [
    # CNN 家族
    "resnet50", "resnet101", "efficientnet_b4", "convnext_tiny", "densenet121",
    # ViT 家族
    "vit_b16", "vit_l16", "deit_b", "swin_tiny", "beit_base",
    # 混合架构
    "regnet_y", "maxvit_tiny",
]

# 下游数据集（评估迁移性预测的目标任务）
# 使用 torchvision 可自动下载的数据集
E2_TARGET_DATASETS_MINI = ["cifar10", "stl10", "svhn"]
E2_TARGET_DATASETS_FULL = [
    "cifar10", "cifar100", "stl10", "svhn",
    "mnist", "fashion_mnist", "kmnist",
    "eurosat", "oxford_pets", "flowers102", "food101", "dtd",
]

# Linear probe 训练配置
E2_LP_LR          = 1e-3
E2_LP_EPOCHS      = 50
E2_LP_BATCH_SIZE  = 256

# ═══════════════════════════════════════════════════════════════
# E3：Δ_dir 数据集筛选（方向增益准则）
# ═══════════════════════════════════════════════════════════════

# 初始训练集 + 候选池
E3_BASE_DATASET     = "cifar10"            # 初始训练集（代理：固定 ResNet50 特征）
E3_CANDIDATE_POOL   = [
    "stl10", "svhn", "mnist", "fashion_mnist",
    "kmnist", "eurosat", "cifar100",
]
E3_PROBE_FOR_DELTA  = "resnet50"           # 计算 Δ_dir 使用的探针
E3_N_ROUNDS         = len(E3_CANDIDATE_POOL)  # 贪心选取轮数

# ═══════════════════════════════════════════════════════════════
# E4：L_Grassmann 域不变表示学习
# ═══════════════════════════════════════════════════════════════

# DomainNet 的 6 个域（需手动下载数据，使用时跳过）
E4_SOURCE_DOMAIN    = "real"
E4_TARGET_DOMAINS   = ["clipart", "infograph", "painting", "quickdraw", "sketch"]
E4_LAMBDA_VALUES    = [0.01, 0.1, 1.0]     # L_Grassmann 的权重 λ
E4_EPOCHS           = 30
E4_K_ALIGN          = 50                   # 对齐损失的子空间维度

# ═══════════════════════════════════════════════════════════════
# 探针配置（快速 / 完整）
# ═══════════════════════════════════════════════════════════════
PROBES_MINI = ["resnet50"]
PROBES_FULL = E2_PROBES_FULL

# ── 数据集语义域标签（用于可视化分组）────────────────────────
DATASET_DOMAIN = {
    "mnist":         "handwritten",
    "kmnist":        "handwritten",
    "fashion_mnist": "handwritten",
    "cifar10":       "natural",
    "cifar100":      "natural",
    "stl10":         "natural",
    "svhn":          "digit_scene",
    "eurosat":       "satellite",
    "oxford_pets":   "finegrained",
    "flowers102":    "finegrained",
    "food101":       "finegrained",
    "dtd":           "texture",
}

DOMAIN_COLORS = {
    "handwritten":  "#4C72B0",
    "natural":      "#55A868",
    "digit_scene":  "#8172B2",
    "satellite":    "#64B5CD",
    "finegrained":  "#937860",
    "texture":      "#C44E52",
}
