"""ARCHIVED: legacy 22-variant two-stage experiment using the obsolete SSR.

Two-stage pipeline comparison vs SOTA SF-MTE methods
=====================================================

Pool: 22 variants × 5 architectures (Phase A + Phase B)
Common tasks: cifar10, cifar100, dtd, eurosat, svhn

Strategies compared:
  Flat (evaluate score on all 22 candidates):
    - Random
    - Flat RankMe (label-free)
    - Flat SSR (label-free, ours)
    - Flat SCR (label-free, ours)
    - Flat LogME (label-required)
    - Flat LEEP (label-required)
    - Flat H-score (label-required)
    - Flat NCTI (label-required)

  Two-stage (stage 1: 5 architecture canonicals → pick a*; stage 2: |V_a*| variants):
    - 2S: LogME → LogME (label-required, our pipeline applied with LogME)
    - 2S: LogME → SCR (mixed)
    - 2S: SSR → SCR (label-free, ours)  ← our main proposal
    - 2S: NCTI → SCR (mixed)
    - 2S: RankMe → RankMe (label-free, naïve)

  Oracle: argmax of LP

Reads: results/within_arch_baselines.csv (22v×5t with all baselines computed)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

CSV = "results/within_arch_baselines.csv"

CANONICAL = {
    'resnet50':                       'resnet50.tv_in1k',
    'vit_base_patch16_224':           'vit_base_patch16_224.augreg_in1k',
    'convnext_small':                 'convnext_small.fb_in1k',
    'convnext_tiny':                  'convnext_tiny.fb_in1k',
    'swin_small':                     'swin_small_patch4_window7_224.ms_in1k',
}


def flat(d, target, score_col):
    sub = d[d.task == target].copy()
    if len(sub) == 0: return None, None, 0
    idx = sub[score_col].idxmax()
    return sub.loc[idx, 'variant'], sub.loc[idx, score_col], len(sub)


def two_stage(d, target, stage1_col, stage2_col):
    sub = d[d.task == target]
    canonicals = list(CANONICAL.values())
    canon_sub = sub[sub.variant.isin(canonicals)]
    if len(canon_sub) == 0: return None, None, 0
    winner_idx = canon_sub[stage1_col].idxmax()
    winner_arch = canon_sub.loc[winner_idx, 'arch']
    n1 = len(canon_sub)
    arch_sub = sub[sub.arch == winner_arch]
    if len(arch_sub) == 0: return None, None, n1
    idx = arch_sub[stage2_col].idxmax()
    return arch_sub.loc[idx, 'variant'], arch_sub.loc[idx, stage2_col], n1 + len(arch_sub)


def random_pick(d, target, seed=42):
    sub = d[d.task == target]
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(sub), size=100, replace=True)
    return None, sub.iloc[picks]['lp_acc'].mean(), 1


def main():
    d = pd.read_csv(CSV)
    print(f"Pool: {d.variant.nunique()} variants, {d.task.nunique()} tasks, "
          f"{d.arch.nunique()} archs", flush=True)

    # Define strategies
    strategies = []

    # Flat methods
    strategies.append(('Random', 'random', None, False, None))
    strategies.append(('Flat: RankMe',  'flat', 'rankme',  False, None))
    strategies.append(('Flat: SSR (ours)', 'flat', 'ssr', False, None))   # SSR present in scr_source CSVs
    strategies.append(('Flat: SCR (ours)', 'flat', 'scr', False, None))
    strategies.append(('Flat: LogME',  'flat', 'logme', True, None))
    strategies.append(('Flat: LEEP',   'flat', 'leep',  True, None))
    strategies.append(('Flat: H-score','flat', 'hscore',True, None))
    strategies.append(('Flat: NCTI',   'flat', 'ncti',  True, None))

    # Two-stage methods
    strategies.append(('2S: RankMe→RankMe',  'two_stage', ('rankme','rankme'), False, None))
    strategies.append(('2S: SSR→SCR (ours)', 'two_stage', ('ssr','scr'),       False, None))
    strategies.append(('2S: LogME→SCR',      'two_stage', ('logme','scr'),     'mixed', None))
    strategies.append(('2S: LogME→LogME',    'two_stage', ('logme','logme'),   True, None))
    strategies.append(('2S: NCTI→SCR',       'two_stage', ('ncti','scr'),      'mixed', None))

    # Oracle
    strategies.append(('Oracle', 'flat', 'lp_acc', True, None))

    # Add SSR column from existing data (it's in scr_source_*.csv but not within_arch_baselines.csv yet)
    # If 'ssr' missing, compute from sa_k+uid+reff... actually only Phase A has sa_k. Skip 'ssr' for Phase B.
    if 'ssr' not in d.columns:
        # Phase A SSR is in scr_source CSVs; Phase B in phase_b_scores.csv
        ssr_table = {}
        for fn in ['results/scr_source_ResNet50.csv', 'results/scr_source_ViTB_16.csv',
                    'results/phase_b_scores.csv']:
            try:
                pa = pd.read_csv(fn)
                for _, r in pa.iterrows():
                    ssr_table[(r['variant_name'], r['task'])] = r.get('ssr', float('nan'))
            except Exception: pass
        d['ssr'] = d.apply(lambda r: ssr_table.get((r['variant'], r['task']), float('nan')),
                            axis=1)

    rows = []
    for target in sorted(d.task.unique()):
        sub = d[d.task == target]
        oracle_lp = sub['lp_acc'].max()
        oracle_var = sub.loc[sub['lp_acc'].idxmax(), 'variant']

        for label, kind, cols, needs_labels, _ in strategies:
            if kind == 'random':
                _, mean_lp, n_eval = random_pick(d, target)
                rows.append({'target': target, 'strategy': label,
                              'picked': '<rand>', 'picked_lp': float(mean_lp),
                              'oracle_lp': oracle_lp, 'regret': oracle_lp - mean_lp,
                              'top1_hit': False, 'n_eval': n_eval,
                              'needs_labels': needs_labels})
                continue
            if kind == 'flat':
                picked, _, n_eval = flat(d, target, cols)
            elif kind == 'two_stage':
                picked, _, n_eval = two_stage(d, target, cols[0], cols[1])
            else: continue
            picked_lp = float(sub[sub.variant == picked]['lp_acc'].iloc[0]) if picked else np.nan
            rows.append({'target': target, 'strategy': label,
                          'picked': picked, 'picked_lp': picked_lp,
                          'oracle_lp': oracle_lp, 'regret': oracle_lp - picked_lp,
                          'top1_hit': (picked == oracle_var),
                          'n_eval': n_eval, 'needs_labels': needs_labels})

    eval_df = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    eval_df.to_csv("results/two_stage_sota_eval.csv", index=False)
    print(f"\nSaved: results/two_stage_sota_eval.csv ({len(eval_df)} rows)\n")

    # Summary table
    print("=" * 90)
    print(f"{'strategy':28s}  needs_lab   top1   mean_regret   mean_LP   mean_n_eval")
    print("-" * 90)
    summary_rows = []
    for label, kind, cols, needs_labels, _ in strategies:
        g = eval_df[eval_df.strategy == label]
        if len(g) == 0: continue
        top1 = g['top1_hit'].mean()
        regret = g['regret'].mean()
        lp = g['picked_lp'].mean()
        ne = g['n_eval'].mean()
        nl = '✓' if needs_labels is True else ('mixed' if needs_labels=='mixed' else '✗')
        print(f"  {label:28s}  {nl:>6s}     {top1:.0%}    {regret:+.4f}      "
              f"{lp:.4f}     {ne:.1f}")
        summary_rows.append({
            'strategy': label, 'needs_labels': nl,
            'top1': top1, 'mean_regret': regret, 'mean_LP': lp,
            'mean_n_eval': ne,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/two_stage_sota_summary.csv", index=False)
    print(f"\nSaved summary: results/two_stage_sota_summary.csv\n")


if __name__ == "__main__":
    main()
