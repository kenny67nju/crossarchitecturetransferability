"""
P1 — Hierarchical variance decomposition for v1.5 Claim 1.

Decompose Var(LP_acc) and Var(SCR*) and Var(SSR^‡) into three nested levels:
  V_family    = Var( mean_LP[family, task] ) over families, averaged over tasks
  V_arch      = Var( mean_LP[arch, task] − mean_LP[family, task] ) within family
  V_variant   = Var( LP[variant, task] − mean_LP[arch, task] ) within arch
  V_total     = sum of the three

Report the % of total variance at each level. This grounds the hierarchical
claim that family-level differences dominate, justifying the two-stage design.

Also test the same decomposition on score columns to verify:
  - SSR^‡ should have HIGH family-level variance (it discriminates families)
  - SCR* should have RELATIVELY HIGHER within-arch variance (it discriminates variants)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-12

FAMILY_MAP = {
    'resnet18':'F1_ResNet','resnet34':'F1_ResNet','resnet50':'F1_ResNet',
    'resnet101':'F1_ResNet','resnet152':'F1_ResNet',
    'resnext50':'F1_ResNet','wide_resnet50_2':'F1_ResNet',
    'vit_base':'F2_ViT','vit_small':'F2_ViT','vit_tiny':'F2_ViT',
    'vit_b32':'F2_ViT','vit_large':'F2_ViT','vit_b32_clip':'F2_ViT',
    'vit_large_clip':'F2_ViT','vit_dinov2':'F2_ViT','vit_b_16':'F2_ViT',
    'deit_base':'F2_ViT','deit_small':'F2_ViT','deit_tiny':'F2_ViT',
    'deit_b':'F2_ViT','beit_base':'F2_ViT','beit_b':'F2_ViT',
    'mae_b':'F2_ViT','dino_s':'F2_ViT',
    'convnext_small':'F3_ConvNeXt','convnext_tiny':'F3_ConvNeXt','convnext_base':'F3_ConvNeXt',
    'efficientnet_b0':'F4_Efficient','efficientnet_b2':'F4_Efficient',
    'efficientnet_b3':'F4_Efficient','efficientnet_b4':'F4_Efficient',
    'tf_efficientnet':'F4_Efficient','mnasnet':'F4_Efficient',
    'mobilenet_v3_large':'F4_Efficient','mobilenet_v3_small':'F4_Efficient',
    'mobilenet_v2':'F4_Efficient',
    'swin_small':'F5_Hierarchical','swin_tiny':'F5_Hierarchical','swin_base':'F5_Hierarchical',
    'maxvit':'F5_Hierarchical','mixer_b16':'F5_Hierarchical',
    'inception_v3':'F7_Misc','vgg16':'F7_Misc','vgg16_bn':'F7_Misc','regnet':'F7_Misc',
    'densenet121':'F7_Misc','densenet169':'F7_Misc','densenet201':'F7_Misc','googlenet':'F7_Misc',
}

d = pd.read_csv('results/pool_b_final.csv')
d['scr_star']   = np.log(np.maximum(d['reff'], EPS)) / np.log(d['n_class'])
d['ssr_dagger'] = (np.log(np.maximum(d['reff'], EPS)) -
                   np.log(np.maximum(d['sa_k'], EPS))) / np.log(d['n_class'])
d['rankme_log'] = np.log(np.maximum(d['reff'], EPS))
d['family'] = d['arch'].map(lambda a: FAMILY_MAP.get(a, 'Unknown'))

# Restrict to multi-variant archs so within-arch variance is meaningful
arch_var = d.groupby('arch').variant.nunique()
multi = arch_var[arch_var >= 2].index.tolist()
d = d[d.arch.isin(multi)].copy()

print(f"Pool B multi-variant: {len(d)} rows, {d.variant.nunique()} variants, "
      f"{d.arch.nunique()} archs, {d.family.nunique()} families, {d.task.nunique()} tasks")
print(f"  family distribution:")
for f, g in d.groupby('family'):
    print(f"    {f:18s}  archs={g.arch.nunique():2d}  variants={g.variant.nunique():2d}")


def variance_decomp(d, value_col):
    """Decompose Var(value_col) into family / arch (within-family) / variant (within-arch)
    components, averaged across tasks. Returns dict with raw vars and percentages."""
    rows = []
    for task, g in d.groupby('task'):
        # Family-level mean per task
        fam_mean = g.groupby('family')[value_col].mean()
        # Arch-level mean within family per task
        arch_mean = g.groupby(['family', 'arch'])[value_col].mean()
        # Variant within arch per task: just the residual

        # V_family: variance of family means weighted by family size
        v_fam = fam_mean.var(ddof=0)

        # V_arch_within_family: pooled variance of (arch_mean - family_mean)
        arch_resid = arch_mean.copy()
        for fam in fam_mean.index:
            arch_resid.loc[fam] = arch_resid.loc[fam] - fam_mean[fam]
        v_arch = arch_resid.var(ddof=0)

        # V_within_arch: pooled variance of (variant - arch_mean)
        v_within = 0.0
        n_within = 0
        for arch_key, gg in g.groupby('arch'):
            am = gg[value_col].mean()
            r = gg[value_col].values - am
            v_within += np.sum(r**2)
            n_within += len(r)
        v_within = v_within / max(n_within, 1)

        v_total = v_fam + v_arch + v_within
        rows.append({
            'task': task, 'V_family': v_fam, 'V_arch': v_arch,
            'V_within': v_within, 'V_total': v_total,
        })

    df = pd.DataFrame(rows)
    avg = df[['V_family', 'V_arch', 'V_within', 'V_total']].mean()
    total = avg['V_total']
    return {
        'V_family': avg['V_family'], 'V_arch': avg['V_arch'],
        'V_within': avg['V_within'], 'V_total': total,
        'pct_family': 100 * avg['V_family'] / total,
        'pct_arch':   100 * avg['V_arch'] / total,
        'pct_within': 100 * avg['V_within'] / total,
        'per_task': df,
    }


print("\n" + "="*82)
print("P1. Variance decomposition (averaged across 5 tasks, multi-variant pool)")
print("="*82)

print(f"\n  {'quantity':18s}  {'V_family':>10s}  {'V_arch':>10s}  {'V_within':>10s}  "
      f"{'V_total':>10s}  | {'%family':>8s}  {'%arch':>8s}  {'%within':>8s}")
print("  " + "-"*92)

for col, label in [
    ('lp_acc',      'LP-acc (ground)'),
    ('rankme_log',  'log r_eff (RankMe)'),
    ('scr_star',    'SCR* score'),
    ('ssr_dagger',  'SSR^‡ score'),
]:
    r = variance_decomp(d, col)
    print(f"  {label:18s}  {r['V_family']:10.4g}  {r['V_arch']:10.4g}  "
          f"{r['V_within']:10.4g}  {r['V_total']:10.4g}  | "
          f"{r['pct_family']:7.1f}%  {r['pct_arch']:7.1f}%  {r['pct_within']:7.1f}%")

# Per-task LP-acc decomposition table
print("\n  Per-task LP-acc variance decomposition:")
print(f"    {'task':12s}  {'V_family':>10s}  {'V_arch':>10s}  {'V_within':>10s}  "
      f"{'V_total':>10s}  | {'%family':>8s}  {'%arch':>8s}  {'%within':>8s}")
r_lp = variance_decomp(d, 'lp_acc')
for _, row in r_lp['per_task'].iterrows():
    pct_f = 100 * row['V_family'] / row['V_total']
    pct_a = 100 * row['V_arch'] / row['V_total']
    pct_w = 100 * row['V_within'] / row['V_total']
    print(f"    {row['task']:12s}  {row['V_family']:10.4g}  {row['V_arch']:10.4g}  "
          f"{row['V_within']:10.4g}  {row['V_total']:10.4g}  | "
          f"{pct_f:7.1f}%  {pct_a:7.1f}%  {pct_w:7.1f}%")

# ════════════════════════════════════════════════════════════════════════
# Score-LP variance share alignment
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*82)
print("Score variance share vs LP variance share")
print("(For each score, does its variance concentration MATCH the level it's good at?)")
print("="*82)

r_lp = variance_decomp(d, 'lp_acc')
r_scr = variance_decomp(d, 'scr_star')
r_ssr = variance_decomp(d, 'ssr_dagger')

print(f"\n  Layer        LP-acc share   SCR* share    SSR^‡ share    Δ(SSR-SCR)")
print(f"  family       {r_lp['pct_family']:8.1f}%      "
      f"{r_scr['pct_family']:8.1f}%       {r_ssr['pct_family']:8.1f}%       "
      f"{r_ssr['pct_family']-r_scr['pct_family']:+.1f}%")
print(f"  arch_in_fam  {r_lp['pct_arch']:8.1f}%      "
      f"{r_scr['pct_arch']:8.1f}%       {r_ssr['pct_arch']:8.1f}%       "
      f"{r_ssr['pct_arch']-r_scr['pct_arch']:+.1f}%")
print(f"  within_arch  {r_lp['pct_within']:8.1f}%      "
      f"{r_scr['pct_within']:8.1f}%       {r_ssr['pct_within']:8.1f}%       "
      f"{r_ssr['pct_within']-r_scr['pct_within']:+.1f}%")

# Verdict
print("\n  Interpretation:")
if r_ssr['pct_family'] > r_scr['pct_family'] + 5:
    print(f"    ✅ SSR^‡ has HIGHER family-level variance share than SCR* "
          f"(+{r_ssr['pct_family']-r_scr['pct_family']:.1f}% pts).")
    print(f"        This supports Claim 2: SSR^‡ is a family discriminator.")
else:
    print(f"    🟠 SSR^‡ does NOT have higher family-level variance share.")
    print(f"        Claim 2 needs reframing.")

if r_scr['pct_within'] > r_ssr['pct_within'] + 2:
    print(f"    ✅ SCR* has HIGHER within-arch variance share than SSR^‡ "
          f"(+{r_scr['pct_within']-r_ssr['pct_within']:.1f}% pts).")
    print(f"        This supports Claim 3: SCR* is a within-family ranker.")
else:
    print(f"    🟠 SCR* within-arch variance share not clearly higher.")

# ════════════════════════════════════════════════════════════════════════
# Family-level ranking accuracy
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*82)
print("Family-level ranking accuracy (does picking right family matter?)")
print("="*82)

# For each task, compute family-mean LP. Best family per task = oracle family.
# How much LP do we lose by picking the wrong family?
print(f"\n  {'task':12s}  {'best_family':18s}  {'oracle_LP':>10s}  "
      f"{'mean_LP':>10s}  {'worst_LP':>10s}  {'oracle−mean':>12s}")
for task, g in d.groupby('task'):
    fam_mean = g.groupby('family').lp_acc.mean()
    best_fam = fam_mean.idxmax()
    print(f"  {task:12s}  {best_fam:18s}  "
          f"{fam_mean.max():10.3f}  {fam_mean.mean():10.3f}  "
          f"{fam_mean.min():10.3f}  {fam_mean.max()-fam_mean.mean():12.3f}")

# ════════════════════════════════════════════════════════════════════════
# Save summary CSV
# ════════════════════════════════════════════════════════════════════════
summary = pd.DataFrame([
    {'quantity': 'LP-acc',     **{k: v for k, v in r_lp.items() if k != 'per_task'}},
    {'quantity': 'SCR*',       **{k: v for k, v in r_scr.items() if k != 'per_task'}},
    {'quantity': 'SSR^‡',      **{k: v for k, v in r_ssr.items() if k != 'per_task'}},
])
summary.to_csv('results/p1_variance_decomp.csv', index=False)
print(f"\nSaved → results/p1_variance_decomp.csv")
