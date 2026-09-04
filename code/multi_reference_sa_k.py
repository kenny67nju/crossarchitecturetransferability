"""
multi_reference_sa_k.py
A3：多参考共识 SA_k
对每个 (架构, 目标任务) 对，使用多个参考任务计算 SA_k，取中位数作为 ensemble 结果
对比单参考（STL-10）vs 多参考（STL-10, CIFAR-10, Flowers-102 等）的稳健性
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from metrics.reff import effective_rank_from_singular_values

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "results" / "vtab_features"
EPS = 1e-12
K = 20

ARCHS = [
    'convnext_base','convnext_small','convnext_tiny',
    'deit_small_patch16_224','deit_tiny_patch16_224',
    'densenet121','densenet169',
    'efficientnet_b0','efficientnet_b2','efficientnet_b3',
    'mobilenetv3_large_100','mobilenetv3_small_100',
    'regnetx_032','regnety_032',
    'resnet18','resnet34','resnet50','resnet101','resnet152',
    'swin_small_patch4_window7_224','swin_tiny_patch4_window7_224',
    'vit_base_patch16_224','vit_small_patch16_224','vit_tiny_patch16_224',
    'wide_resnet50_2',
]
ALL_TASKS = ['cifar10','cifar100','dtd','fashion_mnist','flowers102','pets','stl10','svhn']
# 候选参考任务集合（不重叠的多参考）
REF_SETS = {
    'STL-10 only':           ['stl10'],
    'STL-10 + CIFAR-10':     ['stl10', 'cifar10'],
    'STL-10 + Flowers':      ['stl10', 'flowers102'],
    '4-ref ensemble':        ['stl10', 'cifar10', 'flowers102', 'pets'],
    '5-ref ensemble':        ['stl10', 'cifar10', 'flowers102', 'pets', 'fashion_mnist'],
}


def load_Hy(arch, task):
    d = np.load(FEAT / f"{arch}__{task}.npz", allow_pickle=True)
    H = d['features'] if 'features' in d else d['H']
    y = d['labels']   if 'labels'   in d else d['y']
    return H.astype(np.float64), y.astype(int)


def top_k_Vt(H, k):
    H = H - H.mean(0)
    _, sv, Vt = np.linalg.svd(H, full_matrices=False)
    return Vt[:k], sv


def reff_sv(sv):
    return effective_rank_from_singular_values(sv)


def sa_k_vt(Vt1, Vt2, k):
    k1 = min(k, Vt1.shape[0], Vt2.shape[0])
    return float(((Vt1[:k1] @ Vt2[:k1].T)**2).sum() / k1)


# 加载所有参考子空间
print("Loading reference subspaces for all candidate refs...")
ref_Vt = {}      # (arch, ref_task) -> Vt
for arch in ARCHS:
    for ref in ALL_TASKS:
        try:
            H, _ = load_Hy(arch, ref)
            Vt, _ = top_k_Vt(H, K)
            ref_Vt[(arch, ref)] = Vt
        except FileNotFoundError:
            pass
print(f"  {len(ref_Vt)} reference subspaces loaded.\n")

# 加载 LP 结果
df_lp = pd.read_csv(ROOT / "results" / "vtab_25arch_8tasks.csv")
lp_lookup = {(r.arch, r.task): r.lp_acc for _, r in df_lp.iterrows()}

# 主循环：对每个目标任务，针对每种参考组合计算 SCR-spe
print("="*70)
print("Computing SCR-spe with single vs multi-reference SA_k ensemble")
print("="*70)

results = {ref_name: {} for ref_name in REF_SETS}
for arch in ARCHS:
    for target in ALL_TASKS:
        if target == 'stl10': continue   # stl10 有时是参考
        if (arch, target) not in [(a, t) for (a, t) in [(arch, target)]]:
            pass
        try:
            H, y = load_Hy(arch, target)
        except FileNotFoundError:
            continue

        Vt_t, _ = top_k_Vt(H, K)
        # 总 SVD 用于 r_eff
        H_c = H - H.mean(0)
        sv = np.linalg.svd(H_c, compute_uv=False)
        r_eff = reff_sv(sv)
        log_r = float(np.log(r_eff + EPS))
        C = int(len(np.unique(y)))
        log_C = float(np.log(max(C, 2)))
        lp = lp_lookup.get((arch, target), np.nan)

        for ref_name, ref_list in REF_SETS.items():
            sa_values = []
            for ref in ref_list:
                if ref == target: continue   # 排除目标任务作参考
                if (arch, ref) not in ref_Vt: continue
                sa_values.append(sa_k_vt(Vt_t, ref_Vt[(arch, ref)], K))
            if len(sa_values) == 0: continue
            # 中位数 ensemble
            sa_med = float(np.median(sa_values))
            # SCR-spe (additive, β=15)
            scr = log_r + 15.0 * (1.0 - sa_med)
            # SSR^‡ form
            ssr = (log_r - np.log(sa_med + EPS)) / log_C
            results[ref_name].setdefault(target, []).append({
                'arch': arch, 'lp_acc': lp, 'sa_med': sa_med,
                'scr': scr, 'ssr_dagger': ssr, 'n_refs': len(sa_values),
            })


# 汇总：每参考方案的遗憾值与 ρ
def compute_metrics(per_target):
    """返回 (mean_regret, mean_rho, n_tasks)"""
    regs = []; rhos = []
    for t, rows in per_target.items():
        df = pd.DataFrame(rows).dropna(subset=['lp_acc', 'scr'])
        if len(df) < 5: continue
        oracle = df.lp_acc.max()
        sel = df.loc[df.scr.idxmax(), 'lp_acc']
        regs.append(oracle - sel)
        rho, _ = spearmanr(df.scr, df.lp_acc)
        rhos.append(rho)
    return float(np.mean(regs)), float(np.mean(rhos)), len(regs)


def compute_metrics_ssr(per_target):
    regs = []; rhos = []
    for t, rows in per_target.items():
        df = pd.DataFrame(rows).dropna(subset=['lp_acc', 'ssr_dagger'])
        if len(df) < 5: continue
        oracle = df.lp_acc.max()
        sel = df.loc[df.ssr_dagger.idxmax(), 'lp_acc']
        regs.append(oracle - sel)
        rho, _ = spearmanr(df.ssr_dagger, df.lp_acc)
        rhos.append(rho)
    return float(np.mean(regs)), float(np.mean(rhos)), len(regs)


print(f"\n{'Reference set':25s}  {'#refs':>6s}  {'SCR-spe reg':>11s}  {'SCR-spe ρ':>10s}  {'SSR reg':>9s}  {'SSR ρ':>8s}  {'n_tasks':>8s}")
print("-"*90)
for ref_name in REF_SETS:
    per = results[ref_name]
    n_refs = max(REF_SETS[ref_name].__len__(), 1)
    reg_scr, rho_scr, n_t = compute_metrics(per)
    reg_ssr, rho_ssr, _   = compute_metrics_ssr(per)
    print(f"{ref_name:25s}  {n_refs:>6d}  {reg_scr:+11.4f}  {rho_scr:+10.4f}"
          f"  {reg_ssr:+9.4f}  {rho_ssr:+8.4f}  {n_t:>8d}")

# ─────────────────────────────────────────────────────────────
# 稳健性指标：跨参考方案的标准差
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("跨参考方案稳健性：每个 (arch, target) 对的 SA_k 在不同参考下的方差")
print(f"{'='*70}")
all_arch_target = set()
for ref_name in REF_SETS:
    for t, rows in results[ref_name].items():
        for r in rows:
            all_arch_target.add((r['arch'], t))

robustness_data = []
for (arch, target) in sorted(all_arch_target):
    sa_per_ref = {}
    for ref_name in REF_SETS:
        if target in results[ref_name]:
            for r in results[ref_name][target]:
                if r['arch'] == arch:
                    sa_per_ref[ref_name] = r['sa_med']
                    break
    if len(sa_per_ref) >= 3:
        sa_vals = list(sa_per_ref.values())
        cv = float(np.std(sa_vals) / (np.mean(sa_vals) + EPS))
        robustness_data.append({'arch': arch, 'target': target,
                                'sa_mean': np.mean(sa_vals), 'sa_std': np.std(sa_vals),
                                'cv': cv})

df_r = pd.DataFrame(robustness_data)
print(f"\n  全池均值 CV(SA_k) 跨参考方案 = {df_r.cv.mean():.3f}  (期望小：参考方案选择不敏感)")
print(f"  中位数 CV                       = {df_r.cv.median():.3f}")
print(f"\n  CV < 0.10 的对：{(df_r.cv < 0.10).mean()*100:.1f}%")
print(f"  CV < 0.20 的对：{(df_r.cv < 0.20).mean()*100:.1f}%")
print(f"  CV < 0.30 的对：{(df_r.cv < 0.30).mean()*100:.1f}%")

print(f"\n  按目标任务分解：")
for t in df_r.target.unique():
    sub = df_r[df_r.target == t]
    print(f"    {t:15s}  CV 均值 = {sub.cv.mean():.3f}  中位数 = {sub.cv.median():.3f}")

print(f"\n保存结果：")
out = ROOT / "results" / "multi_ref_sa_k.csv"
df_r.to_csv(out, index=False)
print(f"  {out}")
