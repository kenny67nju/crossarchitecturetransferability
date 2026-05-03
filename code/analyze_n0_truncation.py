"""Analyze N0 truncation data — does dim-truncation fix the VGG bias?"""
import pandas as pd
import numpy as np

d = pd.read_csv('results/truncation_ablation.csv')
print(f"N0 data: {len(d)} rows, {d.variant.nunique()} variants, {d.task.nunique()} tasks")
print(f"  Tasks: {sorted(d.task.unique())}")

# For each task, find Flat-RankMe pick at each truncation
TASKS = sorted(d.task.unique())
COLS = ['reff_full', 'reff_trunc_768', 'reff_trunc_512', 'reff_trunc_256', 'reff_trunc_128']

print(f"\n{'='*82}")
print(f"Per-task Flat-RankMe pick at each truncation cap")
print(f"{'='*82}")

oracle = float(np.mean([d[d.task==t].lp_acc.max() for t in TASKS]))
rand = float(np.mean([d[d.task==t].lp_acc.mean() for t in TASKS]))
print(f"\n  Oracle = {oracle:.3f}    Random = {rand:.3f}    Random regret = {oracle-rand:+.3f}")

print(f"\n  {'task':12s}  Oracle  Random  full(LP/pick)        T=768               T=512               T=256              T=128")
for t in TASKS:
    gt = d[d.task == t]
    if len(gt) < 5: continue
    o = gt.lp_acc.max(); r = gt.lp_acc.mean()
    row_str = f"  {t:12s}  {o:.3f}   {r:.3f}  "
    for col in COLS:
        if col not in gt.columns: continue
        pick = gt.loc[gt[col].idxmax()]
        row_str += f"  {pick.lp_acc:.3f} {pick.arch[:12]:12s}"
    print(row_str)

print(f"\n  Mean LP across {len(TASKS)} tasks (and regret):")
for col in COLS:
    if col not in d.columns: continue
    lps = [d[d.task==t].loc[d[d.task==t][col].idxmax()].lp_acc for t in TASKS]
    lp_mu = float(np.mean(lps))
    print(f"    Flat-RankMe ({col:18s}): LP={lp_mu:.3f}    regret={oracle-lp_mu:+.3f}    Δ Random={lp_mu-rand:+.3f}")

# Also: who's selected at each truncation level (frequency)
print(f"\n  Most-frequently-selected architecture at each truncation level:")
for col in COLS:
    if col not in d.columns: continue
    picks = []
    for t in TASKS:
        gt = d[d.task == t]
        picks.append(gt.loc[gt[col].idxmax()].arch)
    print(f"    {col:18s}: picks across 5 tasks: {picks}")

# Spearman correlation of reff vs feat_dim at each truncation
from scipy.stats import spearmanr
print(f"\n  Per-task Spearman ρ(reff_X, feat_dim):")
print(f"  {'task':12s}    full      T=768     T=512     T=256     T=128")
for t in TASKS:
    gt = d[d.task == t]
    row = f"  {t:12s}  "
    for col in COLS:
        if col not in gt.columns: continue
        r, _ = spearmanr(gt[col], gt['feat_dim'])
        row += f"  {r:+.3f}  "
    print(row)

# Verdict
lp_full = float(np.mean([d[d.task==t].loc[d[d.task==t]['reff_full'].idxmax()].lp_acc for t in TASKS]))
lp_t512 = float(np.mean([d[d.task==t].loc[d[d.task==t]['reff_trunc_512'].idxmax()].lp_acc for t in TASKS]))
lp_t256 = float(np.mean([d[d.task==t].loc[d[d.task==t]['reff_trunc_256'].idxmax()].lp_acc for t in TASKS]))
print(f"\n  Verdict:")
print(f"    Flat-RankMe (full)      regret = {oracle-lp_full:+.3f}    {'WORSE' if lp_full<rand else 'BETTER'} than Random")
print(f"    Flat-RankMe (trunc 512) regret = {oracle-lp_t512:+.3f}    {'WORSE' if lp_t512<rand else 'BETTER'} than Random   gain over full: {lp_t512-lp_full:+.3f}")
print(f"    Flat-RankMe (trunc 256) regret = {oracle-lp_t256:+.3f}    {'WORSE' if lp_t256<rand else 'BETTER'} than Random   gain over full: {lp_t256-lp_full:+.3f}")
if lp_t512 - lp_full > 0.10:
    print(f"\n    ✅ Truncation FIXES the bias substantially.")
    print(f"       → Bias is largely dimensional/feature-width.")
elif lp_t512 - lp_full > 0.03:
    print(f"\n    🟠 Truncation HELPS but is not a complete fix.")
else:
    print(f"\n    ❌ Truncation does NOT fix; bias is in higher-mass spectrum.")
