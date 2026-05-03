"""Run two-stage strategies on extended pool 51v × 10t.
Includes 3-way decisive: SSR^‡ vs reff_top256 vs DiSCO.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-12
rng = np.random.default_rng(42)

d = pd.read_csv('results/extended_full_sota.csv')
print(f"Loaded {len(d)} rows, {d.variant.nunique()} variants, {d.task.nunique()} tasks")

# Filter to multi-variant archs
arch_var = d.groupby('arch').variant.nunique()
multi = arch_var[arch_var >= 2].index.tolist()
d_multi = d[d.arch.isin(multi)].copy()
TASKS = sorted(d_multi.task.unique())
multi_archs = d_multi.arch.unique().tolist()
print(f"Multi-variant: {d_multi.variant.nunique()}v, {d_multi.arch.nunique()} archs ({multi_archs})")

oracle_lp = float(np.mean([d_multi[d_multi.task==t].lp_acc.max() for t in TASKS]))
rand_lp = float(np.mean([d_multi[d_multi.task==t].lp_acc.mean() for t in TASKS]))

# Canonical = alphabetically first variant per arch
canonicals = {a: sorted(d_multi[d_multi.arch==a].variant.unique())[0] for a in multi_archs}

def flat(score):
    lps = [d_multi[d_multi.task==t].loc[d_multi[d_multi.task==t][score].idxmax()].lp_acc for t in TASKS]
    return float(np.mean(lps)), lps

def two_stage(s1, s2):
    lps = []
    for t in TASKS:
        dt = d_multi[d_multi.task == t]
        cscores = {}
        for a in multi_archs:
            cv = canonicals[a]
            row = dt[dt.variant == cv]
            if len(row) == 0: continue
            v = row[s1].iloc[0]
            if pd.isna(v): continue
            cscores[a] = float(v)
        if not cscores: continue
        best = max(cscores, key=cscores.get)
        sub = dt[dt.arch == best]
        if len(sub) == 0 or sub[s2].notna().sum()==0: continue
        lps.append(sub.loc[sub[s2].idxmax()].lp_acc)
    cost = len(multi_archs) + np.mean([len(d_multi[d_multi.arch==a].variant.unique()) for a in multi_archs])
    return float(np.mean(lps)), lps, float(cost)

print(f"\n{'='*92}")
print(f"Two-stage strategies on extended pool ({d_multi.variant.nunique()}v × {len(TASKS)}t)")
print(f"{'='*92}")
print(f"  Oracle = {oracle_lp:.3f}    Random ≈ {rand_lp:.3f}")
print(f"  {'strategy':45s}   mean LP   regret    cost")
print("  " + "-"*78)

# Flat baselines
for col, label in [('rankme','Flat-RankMe'),
                    ('rankme_top256','Flat-RankMe-top256'),
                    ('ssr_dagger','Flat-SSR^‡'),
                    ('disco','Flat-DiSCO'),
                    ('logme','Flat-LogME (label)'),
                    ('hscore','Flat-H-score (label)'),
                    ('leep','Flat-LEEP (label)'),
                    ('ncti','Flat-NCTI (label)')]:
    if col not in d_multi.columns: continue
    lp, _ = flat(col)
    flat_cost = float(np.mean([len(d_multi[d_multi.task==t]) for t in TASKS]))
    print(f"  {label:45s}   {lp:.3f}    {oracle_lp-lp:+.3f}    {flat_cost:5.1f}")

# Two-stage variants
print()
for s1, s2, label in [
    ('rankme','rankme','2S RankMe→RankMe'),
    ('rankme_top256','rankme','2S RankMe-top256→RankMe'),
    ('ssr_dagger','rankme','2S SSR^‡→RankMe (proposed)'),
    ('ssr_dagger','rankme_top256','2S SSR^‡→RankMe-top256'),
    ('disco','rankme','2S DiSCO→RankMe'),
    ('disco','disco','2S DiSCO→DiSCO'),
    ('logme','rankme','2S LogME→RankMe (mixed)'),
    ('logme','logme','2S LogME→LogME'),
    ('hscore','rankme','2S H-score→RankMe (mixed)'),
    ('leep','rankme','2S LEEP→RankMe (mixed)'),
]:
    if s1 not in d_multi.columns or s2 not in d_multi.columns: continue
    lp, lps, cost = two_stage(s1, s2)
    print(f"  {label:45s}   {lp:.3f}    {oracle_lp-lp:+.3f}    {cost:5.1f}")

# Decisive 3-way
print(f"\n{'='*92}")
print(f"Decisive 3-way comparison: SSR^‡ vs RankMe-top256 vs DiSCO (extended pool, {len(TASKS)} tasks)")
print(f"{'='*92}")
ssr_lp, ssr_lps = flat('ssr_dagger')
top_lp, top_lps = flat('rankme_top256')
dsc_lp, dsc_lps = flat('disco')

print(f"\n  Per-task LP:")
print(f"  {'task':16s}  Oracle  Random  SSR^‡   RM-top256  DiSCO")
for i, t in enumerate(TASKS):
    o = d_multi[d_multi.task==t].lp_acc.max()
    r = d_multi[d_multi.task==t].lp_acc.mean()
    print(f"  {t:16s}  {o:.3f}   {r:.3f}   {ssr_lps[i]:.3f}   {top_lps[i]:.3f}      {dsc_lps[i]:.3f}")

print(f"\n  Mean: SSR^‡ regret={oracle_lp-ssr_lp:+.3f}  RM-top256 regret={oracle_lp-top_lp:+.3f}  DiSCO regret={oracle_lp-dsc_lp:+.3f}")

# Paired bootstrap
ssr_arr = np.array(ssr_lps); top_arr = np.array(top_lps); dsc_arr = np.array(dsc_lps)
n = len(ssr_arr)
print(f"\n  Paired bootstrap (n_boot=10000, n_tasks={n}):")
for label, diff in [('SSR^‡ - RM-top256', ssr_arr - top_arr),
                     ('SSR^‡ - DiSCO', ssr_arr - dsc_arr),
                     ('RM-top256 - DiSCO', top_arr - dsc_arr)]:
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p2 = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
    sig = '✅ significant' if (lo > 0 or hi < 0) else '🟠 n.s.'
    print(f"    {label:25s}  Δ = {diff.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  p≈{p2:.3f}  {sig}")

# Two-stage paired test S5 vs S2
s5_lp, s5_lps, _ = two_stage('ssr_dagger', 'rankme')
s2_lp, s2_lps = flat('rankme')
diff = np.array(s5_lps) - np.array(s2_lps)
boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(10000)])
lo, hi = np.percentile(boots, [2.5, 97.5])
p2 = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
print(f"\n  2S SSR^‡→RankMe vs Flat-RankMe (paired test):")
print(f"    Δ = {diff.mean():+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], p≈{p2:.4f}")
