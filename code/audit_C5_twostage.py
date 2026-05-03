"""
C5. Two-stage pipeline real test on Pool B (35 multi-variant archs, 132 variants, 5 tasks).

Strategies (per target task T):
  S0  Oracle:           argmax_v LP(v, T) over all 132 variants                     [upper bound]
  S1  Random:           mean LP over 100 random picks from 132 variants              [lower bound]
  S2  Flat-SCR*:        argmax_v SCR*(v, T) over all 132 variants
  S3  Flat-SSR^‡:       argmax_v SSR^‡(v, T) over all 132 variants
  S4  Two-stage SCR*:   stage-1 pick arch via SCR* on representative; stage-2 SCR* on variants
  S5  Two-stage SSR^‡:  stage-1 pick arch via SSR^‡ on representative; stage-2 SCR* on variants
  S6  Two-stage Adaptive (proposed):
        if pool contains ViT-family arch: stage-1 = SCR*
        else:                              stage-1 = SSR^‡
        stage-2 always SCR*

For each (strategy, task), we compute:
  - selected variant's LP accuracy
  - Regret = Oracle_LP − Strategy_LP
  - Cost (forward passes): Flat = 132; Two-stage = 35 + |winning arch variants|

Aggregates:
  - Mean Regret across 5 tasks
  - Mean Cost
  - Pareto frontier (Regret vs Cost)
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

EPS = 1e-12
rng = np.random.default_rng(42)

VIT_FAMILY_KEYWORDS = ['vit_', 'deit', 'swin', 'beit', 'mae', 'dino']

def is_vit_arch(arch):
    return any(p in arch.lower() for p in VIT_FAMILY_KEYWORDS)

# Load and prepare
d = pd.read_csv('results/pool_b_final.csv')
d['scr_star']   = np.log(np.maximum(d['reff'], EPS)) / np.log(d['n_class'])
d['ssr_dagger'] = (np.log(np.maximum(d['reff'], EPS)) -
                   np.log(np.maximum(d['sa_k'], EPS))) / np.log(d['n_class'])

print(f"Pool B: {len(d)} rows, {d.variant.nunique()} variants, "
      f"{d.arch.nunique()} archs, {d.task.nunique()} tasks")

arch_var = d.groupby('arch').variant.nunique()
multi_archs = arch_var[arch_var >= 2].index.tolist()
print(f"Multi-variant archs (≥2 variants): {len(multi_archs)}")

# Restrict to multi-variant archs (so "two-stage" is meaningful)
d_multi = d[d.arch.isin(multi_archs)].copy()
print(f"After multi-variant filter: {d_multi.variant.nunique()} variants, "
      f"{d_multi.arch.nunique()} archs")
print(f"  ViT-family archs in pool: "
      f"{sum(1 for a in multi_archs if is_vit_arch(a))}")

TASKS = sorted(d_multi.task.unique())
ALL_VARIANTS = d_multi.variant.unique()

# ── Compute per-arch representative score (mean across variants for each arch) ──
# Strategy: stage-1 picks arch by averaging SCR* / SSR^‡ across that arch's variants
#           (this simulates "pick one canonical per arch and score it")
# In real deployment, stage-1 uses ONE variant per arch — but pool data has multiple per arch.
# We simulate by taking the FIRST variant per arch as "canonical".

def get_canonical_variant(arch_in_d):
    """Pick canonical variant per arch (alphabetically first) — deterministic."""
    return sorted(arch_in_d.variant.unique())[0]

canonicals = {a: get_canonical_variant(d_multi[d_multi.arch == a]) for a in multi_archs}


def regret_strategy(d_t, strategy, task):
    """For a given task d_t (filtered to one task), run strategy. Return (selected_lp, n_forwards)."""
    available_variants = d_t.variant.tolist()

    if strategy == 'S0_oracle':
        # Best LP over all variants
        best = d_t.loc[d_t['lp_acc'].idxmax()]
        return best['lp_acc'], len(available_variants)

    if strategy == 'S1_random':
        # Mean over 100 random picks
        picks = rng.choice(available_variants, 100, replace=True)
        lps = [d_t[d_t.variant == p]['lp_acc'].iloc[0] for p in picks]
        return float(np.mean(lps)), 1  # cost = 1 forward (pick blindly)

    if strategy == 'S2_flat_scr':
        best = d_t.loc[d_t['scr_star'].idxmax()]
        return best['lp_acc'], len(available_variants)

    if strategy == 'S3_flat_ssr':
        best = d_t.loc[d_t['ssr_dagger'].idxmax()]
        return best['lp_acc'], len(available_variants)

    if strategy.startswith('S4') or strategy.startswith('S5') or strategy.startswith('S6'):
        # Stage-1: score each arch's canonical variant
        s1_score_col = {
            'S4_twostage_scr':      'scr_star',
            'S5_twostage_ssr':      'ssr_dagger',
            'S6_twostage_adaptive': None,  # decided below
        }[strategy]

        if strategy == 'S6_twostage_adaptive':
            archs_in_pool = list(d_t.arch.unique())
            has_vit = any(is_vit_arch(a) for a in archs_in_pool)
            s1_score_col = 'scr_star' if has_vit else 'ssr_dagger'

        canonical_scores = {}
        for a in multi_archs:
            cv = canonicals[a]
            row = d_t[d_t.variant == cv]
            if len(row) == 0: continue
            canonical_scores[a] = float(row[s1_score_col].iloc[0])

        if not canonical_scores:
            return float('nan'), 0
        best_arch = max(canonical_scores, key=canonical_scores.get)

        # Stage-2: pick variant in best_arch by SCR*
        arch_variants_d = d_t[d_t.arch == best_arch]
        if len(arch_variants_d) == 0:
            return float('nan'), len(canonical_scores)
        best_var = arch_variants_d.loc[arch_variants_d['scr_star'].idxmax()]
        n_stage1 = len(canonical_scores)
        n_stage2 = len(arch_variants_d)
        return float(best_var['lp_acc']), n_stage1 + n_stage2

    raise ValueError(strategy)


STRATEGIES = ['S0_oracle', 'S1_random', 'S2_flat_scr', 'S3_flat_ssr',
              'S4_twostage_scr', 'S5_twostage_ssr', 'S6_twostage_adaptive']

print()
print("=" * 92)
print("C5. Two-stage pipeline — per-task selected LP and regret")
print("=" * 92)
print(f"{'task':14s} " + " ".join(f"{s[:10]:>10s}" for s in STRATEGIES))

per_task = {s: [] for s in STRATEGIES}
per_task_cost = {s: [] for s in STRATEGIES}
for task in TASKS:
    d_t = d_multi[d_multi.task == task]
    if len(d_t) == 0: continue
    row_lp = []
    for s in STRATEGIES:
        lp, cost = regret_strategy(d_t, s, task)
        per_task[s].append(lp)
        per_task_cost[s].append(cost)
        row_lp.append(lp)
    print(f"  {task:12s} " + " ".join(f"{lp:>10.3f}" for lp in row_lp))

print()
print("=" * 92)
print("Aggregate (mean across 5 tasks)")
print("=" * 92)
print(f"{'strategy':24s}  mean_LP   mean_Regret   mean_Cost   ΔLP_vs_Random")
print("-" * 75)
oracle_mean = float(np.mean(per_task['S0_oracle']))
random_mean = float(np.mean(per_task['S1_random']))

for s in STRATEGIES:
    lp_mu = float(np.mean(per_task[s]))
    regret = oracle_mean - lp_mu
    cost = float(np.mean(per_task_cost[s]))
    delta_random = lp_mu - random_mean
    star = ''
    if s == 'S6_twostage_adaptive': star = '  ← proposed'
    elif s == 'S0_oracle': star = '  (upper bound)'
    elif s == 'S1_random': star = '  (lower bound)'
    print(f"  {s:24s}  {lp_mu:.4f}    {regret:+.4f}     {cost:6.1f}      {delta_random:+.4f}{star}")

# Bootstrap CI for proposed strategy regret
print()
print("=" * 92)
print("Bootstrap CI on Regret per strategy (task-level resampling, n_boot=2000)")
print("=" * 92)

for s in STRATEGIES:
    if s == 'S0_oracle': continue
    lps = np.array(per_task[s])
    oracle_lps = np.array(per_task['S0_oracle'])
    regrets = oracle_lps - lps
    n = len(regrets)
    if n < 2: continue
    boots = np.array([regrets[rng.integers(0, n, n)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  {s:24s}  Regret = {regrets.mean():+.4f}   95% CI = [{lo:+.4f}, {hi:+.4f}]")

# Paired comparison: S6 vs S2 (proposed vs flat-SCR*)
print()
print("=" * 92)
print("Paired bootstrap: S6 (Two-stage Adaptive) vs S2 (Flat-SCR*)")
print("=" * 92)
diff = np.array(per_task['S6_twostage_adaptive']) - np.array(per_task['S2_flat_scr'])
n = len(diff)
boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
lo, hi = np.percentile(boots, [2.5, 97.5])
p2 = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
print(f"  ΔLP (S6 − S2) per task: {diff}")
print(f"  Mean ΔLP = {diff.mean():+.4f}, 95% CI = [{lo:+.4f}, {hi:+.4f}], p≈{p2:.3f}")
if lo > 0:
    print("  ✅ S6 SIGNIFICANTLY BEATS S2")
elif hi < 0:
    print("  ❌ S6 LOSES TO S2")
else:
    print("  🟠 S6 ≈ S2 (CI crosses 0)")

# Save aggregate to CSV
out_rows = []
for s in STRATEGIES:
    out_rows.append({
        'strategy': s,
        'mean_lp': float(np.mean(per_task[s])),
        'mean_regret': oracle_mean - float(np.mean(per_task[s])),
        'mean_cost_forwards': float(np.mean(per_task_cost[s])),
        **{f'lp_{t}': lp for t, lp in zip(TASKS, per_task[s])},
    })
pd.DataFrame(out_rows).to_csv('results/c5_twostage_pipeline.csv', index=False)
print(f"\nSaved → results/c5_twostage_pipeline.csv")
