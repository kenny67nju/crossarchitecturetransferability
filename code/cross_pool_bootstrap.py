"""
cross_pool_bootstrap.py
跨池组装 23 任务级自举数据集，收紧 Flat 比较的 CI
跨 Pool B（5）+ VTAB（8）+ Extended（10）= 23 task units
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
EPS = 1e-12
RNG = np.random.default_rng(42)
N_BOOT = 10000


def per_task_regret(df, score_col):
    """每任务取 score 最大变体的 oracle gap"""
    out = {}
    for t in sorted(df.task.unique()):
        sub = df[df.task == t].dropna(subset=[score_col, 'lp_acc'])
        if len(sub) < 2: continue
        oracle = sub.lp_acc.max()
        sel = sub.loc[sub[score_col].idxmax(), 'lp_acc']
        out[t] = float(oracle - sel)
    return out


# ──────────────────────────────────────────
# 加载三池数据并统一格式
# ──────────────────────────────────────────
print("Assembling cross-pool bootstrap dataset...")

# Pool B (multi-variant subset)
pb = pd.read_csv(ROOT / "results" / "pool_b_final.csv")
pb['log_reff'] = np.log(np.maximum(pb['reff'], EPS))
pb['log_SAk']  = np.log(np.maximum(pb['sa_k'],  EPS))
pb['log_C']    = np.log(np.maximum(pb['n_class'].astype(float), 2.0))
pb['ssr_dagger'] = (pb['log_reff'] - pb['log_SAk']) / pb['log_C']
pb['scr_spe_15'] = pb['log_reff'] + 15.0 * (1.0 - pb['sa_k'])
multi_archs = pb.groupby('arch')['variant'].nunique()
multi = multi_archs[multi_archs >= 2].index
pb_m = pb[pb.arch.isin(multi)].copy()
pb_m['pool'] = 'Pool B'

# VTAB
vt = pd.read_csv(ROOT / "results" / "vtab_25arch_8tasks.csv")
vt['log_reff'] = np.log(np.maximum(vt['reff'], EPS))
vt['scr_spe_15'] = vt['log_reff'] + 15.0 * (1.0 - vt['sa_k'])
vt['pool'] = 'VTAB'

# Extended
ex = pd.read_csv(ROOT / "results" / "extended_full_sota.csv")
ex['log_reff'] = ex['rankme']  # rankme 已是 log r_eff
ex['scr_spe_15'] = ex['log_reff'] + 15.0 * (1.0 - ex['sa_k'])
ex['pool'] = 'Extended'

# 计算每池每任务的遗憾值
print("\n" + "="*70)
print(f"{'Pool':10s}  {'Task':18s}  {'Random':>9s}  {'RankMe':>9s}  {'DiSCO':>9s}  {'SSR^‡':>9s}  {'SCR15':>9s}  {'Oracle':>8s}")
print("-"*86)

records = []
for pool_name, df in [('Pool B', pb_m), ('VTAB', vt), ('Extended', ex)]:
    score_cols = {
        'random': None,    # 用 mean LP
        'rankme': 'log_reff' if 'log_reff' in df.columns else 'rankme',
        'disco':  'disco' if 'disco' in df.columns else None,
        'ssr':    'ssr_dagger',
        'scr15':  'scr_spe_15',
    }
    for t in sorted(df.task.unique()):
        sub = df[df.task == t].dropna(subset=['lp_acc'])
        if len(sub) < 5: continue
        oracle = sub.lp_acc.max()
        rec = {'pool': pool_name, 'task': t, 'oracle': oracle, 'random_mean': sub.lp_acc.mean()}
        rec['random_regret'] = oracle - sub.lp_acc.mean()
        for name, col in score_cols.items():
            if name == 'random': continue
            if col is None or col not in df.columns:
                rec[f'{name}_regret'] = np.nan; continue
            sub_v = sub.dropna(subset=[col])
            if len(sub_v) == 0:
                rec[f'{name}_regret'] = np.nan; continue
            sel_idx = sub_v[col].idxmax()
            sel_lp  = sub_v.loc[sel_idx, 'lp_acc']
            rec[f'{name}_regret'] = oracle - sel_lp
        records.append(rec)
        print(f"{pool_name:10s}  {t:18s}  {rec['random_regret']:+9.4f}"
              f"  {rec['rankme_regret']:+9.4f}"
              f"  {rec.get('disco_regret', float('nan')):+9.4f}"
              f"  {rec['ssr_regret']:+9.4f}"
              f"  {rec['scr15_regret']:+9.4f}"
              f"  {oracle:8.4f}")

df_all = pd.DataFrame(records)
print(f"\nTotal task units: {len(df_all)}")
df_all.to_csv(ROOT / "results" / "cross_pool_regrets.csv", index=False)


# ──────────────────────────────────────────
# 跨池自举
# ──────────────────────────────────────────
print("\n" + "="*70)
print(f"跨池自举 ({N_BOOT} 次重采样)")
print("="*70)

methods = {
    'Random':   df_all['random_regret'].values,
    'RankMe':   df_all['rankme_regret'].values,
    'DiSCO':    df_all['disco_regret'].values,
    'SSR^‡':    df_all['ssr_regret'].values,
    'SCR-spe(β=15)': df_all['scr15_regret'].values,
}

n_tasks = len(df_all)
print(f"\n  方法                  均值遗憾值  95% CI            CI 宽度")
print("-"*70)
for name, vals in methods.items():
    valid = ~np.isnan(vals)
    v = vals[valid]
    if len(v) < 5:
        print(f"  {name:20s}  N/A")
        continue
    boot_means = np.array([
        np.mean(RNG.choice(v, size=len(v), replace=True))
        for _ in range(N_BOOT)
    ])
    mean = float(np.mean(v))
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    print(f"  {name:20s}  {mean:+.4f}     [{ci_lo:+.4f}, {ci_hi:+.4f}]   {ci_hi-ci_lo:.4f}")

# ──────────────────────────────────────────
# 配对检验：SSR^‡ vs DiSCO（仅在两者都有的任务上）
# ──────────────────────────────────────────
print("\n" + "="*70)
print("配对自举：SSR^‡ vs DiSCO  (仅在两者都有定义的任务上)")
print("="*70)

paired_mask = ~(np.isnan(df_all.ssr_regret) | np.isnan(df_all.disco_regret))
n_paired = int(paired_mask.sum())
print(f"\n  配对任务数: {n_paired} / {len(df_all)}")
if n_paired >= 5:
    diffs = df_all.loc[paired_mask, 'ssr_regret'].values - df_all.loc[paired_mask, 'disco_regret'].values
    boot_diffs = np.array([
        np.mean(RNG.choice(diffs, size=len(diffs), replace=True))
        for _ in range(N_BOOT)
    ])
    mean_diff = float(np.mean(diffs))
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_two = 2 * min((boot_diffs > 0).mean(), (boot_diffs < 0).mean())
    print(f"  Δ(SSR - DiSCO) = {mean_diff:+.4f}")
    print(f"  95% CI         = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  p-value (two)  = {p_two:.4f}")
    print(f"  Tasks SSR < DiSCO: {(diffs < 0).sum()} / {len(diffs)}")

# ──────────────────────────────────────────
# 配对检验：SCR-spe vs SSR^‡
# ──────────────────────────────────────────
print("\n" + "="*70)
print("配对自举：SCR-spe(β=15) vs SSR^‡")
print("="*70)

paired_mask = ~(np.isnan(df_all.scr15_regret) | np.isnan(df_all.ssr_regret))
n_paired = int(paired_mask.sum())
if n_paired >= 5:
    diffs = df_all.loc[paired_mask, 'scr15_regret'].values - df_all.loc[paired_mask, 'ssr_regret'].values
    boot_diffs = np.array([
        np.mean(RNG.choice(diffs, size=len(diffs), replace=True))
        for _ in range(N_BOOT)
    ])
    mean_diff = float(np.mean(diffs))
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_two = 2 * min((boot_diffs > 0).mean(), (boot_diffs < 0).mean())
    print(f"\n  配对任务数: {n_paired}")
    print(f"  Δ(SCR15 - SSR) = {mean_diff:+.4f}")
    print(f"  95% CI         = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  p-value (two)  = {p_two:.4f}")

# ──────────────────────────────────────────
# 配对检验：SSR^‡ vs RankMe（应有显著优势）
# ──────────────────────────────────────────
print("\n" + "="*70)
print("配对自举：SSR^‡ vs RankMe")
print("="*70)

paired_mask = ~(np.isnan(df_all.ssr_regret) | np.isnan(df_all.rankme_regret))
n_paired = int(paired_mask.sum())
if n_paired >= 5:
    diffs = df_all.loc[paired_mask, 'ssr_regret'].values - df_all.loc[paired_mask, 'rankme_regret'].values
    boot_diffs = np.array([
        np.mean(RNG.choice(diffs, size=len(diffs), replace=True))
        for _ in range(N_BOOT)
    ])
    mean_diff = float(np.mean(diffs))
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_two = 2 * min((boot_diffs > 0).mean(), (boot_diffs < 0).mean())
    print(f"\n  配对任务数: {n_paired}")
    print(f"  Δ(SSR - RankMe) = {mean_diff:+.4f}")
    print(f"  95% CI          = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  p-value (two)   = {p_two:.4f}")
    print(f"  Tasks SSR better: {(diffs < 0).sum()} / {len(diffs)}")

print("\nDONE")
