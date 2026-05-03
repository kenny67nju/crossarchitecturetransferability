"""
Compute SOTA baselines for within-architecture variant selection
=================================================================

For each (variant, task), compute:
  Label-free:    SSR, SCR (=reff/(UID·logC)), RankMe, reff, UID
  Label-needed:  LogME, H-score, LEEP, NCTI, DISCO

Pool: 22 variants × 5 architectures × 5 common tasks
      (cifar10, cifar100, dtd, eurosat, svhn)

Output:
  results/within_arch_baselines.csv
  Plus printed table of mean ρ per arch and joint
"""
from __future__ import annotations

import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# Reuse implementations from run_final_benchmark.py
import sys
sys.path.insert(0, '.')
from run_final_benchmark import (compute_rankme, compute_ncti, compute_disco,
                                  compute_leep)
from baselines import LogME, compute_hscore

CACHE = Path("results/vtab_features")
PHASE_A_VIT = "results/scr_source_ViTB_16.csv"
PHASE_A_RN  = "results/scr_source_ResNet50.csv"
PHASE_B_CSV = "results/phase_b_scores.csv"
OUT = Path("results/within_arch_baselines.csv")

VARIANTS = []
# Phase A
for fn, arch_label in [(PHASE_A_VIT, 'vit_base_patch16_224'),
                        (PHASE_A_RN, 'resnet50')]:
    d = pd.read_csv(fn)
    for v in d.variant_name.unique():
        VARIANTS.append((v, arch_label))
# Phase B
for vn, group, _ in [
    ('convnext_small.fb_in1k',           'convnext_small',  None),
    ('convnext_small.in12k_ft_in1k',     'convnext_small',  None),
    ('convnext_tiny.fb_in1k',            'convnext_tiny',   None),
    ('convnext_tiny.in12k_ft_in1k',      'convnext_tiny',   None),
    ('swin_small_patch4_window7_224.ms_in1k',           'swin_small',  None),
    ('swin_small_patch4_window7_224.ms_in22k_ft_in1k',  'swin_small',  None),
]:
    VARIANTS.append((vn, group))

# Common tasks across all 22 variants
COMMON_TASKS = ['cifar10', 'cifar100', 'dtd', 'eurosat', 'svhn']
N_CLASS = {'cifar10': 10, 'cifar100': 100, 'dtd': 47, 'eurosat': 10, 'svhn': 10}


def fp(s): return s.replace('.', '_dot_')


def load_feat(variant, ds):
    p = CACHE / f"{fp(variant)}__{ds}.npz"
    if not p.exists(): return None, None
    f = np.load(p)
    keys = set(f.keys())
    if {'H', 'y'} <= keys:    return f['H'], f['y']
    if {'features', 'labels'} <= keys: return f['features'], f['labels']
    return None, None


def effective_rank(H):
    s = np.linalg.svd(H, full_matrices=False, compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def uid_twonn(H, sample=500):
    from sklearn.neighbors import NearestNeighbors
    if len(H) > sample:
        idx = np.random.default_rng(42).choice(len(H), sample, replace=False)
        H = H[idx]
    nn = NearestNeighbors(n_neighbors=3).fit(H)
    d, _ = nn.kneighbors(H)
    r1, r2 = d[:, 1], d[:, 2]
    mu = r2 / np.maximum(r1, 1e-9)
    mu = mu[mu > 1.001]
    if len(mu) < 5: return float('nan')
    log_mu = np.log(mu)
    F = (np.arange(1, len(log_mu) + 1)) / (len(log_mu) + 1)
    return float(-np.sum(np.log(1 - F)) / np.sum(log_mu))


def linear_probe(H, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    (tr, te), = sss.split(H, y)
    sc = StandardScaler().fit(H[tr])
    clf = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(H[tr]), y[tr])
    return float(clf.score(sc.transform(H[te]), y[te]))


def main():
    rows = []
    for variant, arch in VARIANTS:
        # Load REF features (use stl10 for SA_k computation if needed)
        H_ref, _ = load_feat(variant, 'stl10')
        for task in COMMON_TASKS:
            H, y = load_feat(variant, task)
            if H is None: continue
            n_c = N_CLASS[task]
            log_c = np.log(max(n_c, 2))
            try:
                lp = linear_probe(H, y)
                reff = effective_rank(H)
                uid = uid_twonn(H)
                rankme = compute_rankme(H)
                logme = LogME().fit(H, y)
                hscore = compute_hscore(H, y)
                leep = compute_leep(H, y)
                ncti = compute_ncti(H, y)
                disco = compute_disco(H, y)
            except Exception as e:
                print(f"  err {variant}/{task}: {e}")
                continue

            scr = reff / (max(uid, 1e-12) * log_c)
            rows.append({
                'arch': arch, 'variant': variant, 'task': task,
                'n_class': n_c, 'lp_acc': lp,
                'reff': reff, 'uid': uid,
                'scr': scr,         # within-arch SCR
                'rankme': rankme,   # label-free
                'logme': logme,
                'hscore': hscore,
                'leep': leep,
                'ncti': ncti,
                'disco': disco,
            })
            print(f"  {variant:48s} {task:10s} lp={lp:.3f} reff={reff:.0f} "
                  f"rankme={rankme:.2f} logme={logme:+.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}: {len(df)} rows", flush=True)

    # Per-arch and combined per-task ρ
    print("\n" + "=" * 90)
    print("Within-architecture per-task mean ρ (LP truth):")
    print("=" * 90)
    metrics = ['scr', 'rankme', 'reff', 'uid', 'logme', 'hscore', 'leep', 'ncti', 'disco']
    label_free = {'scr', 'rankme', 'reff', 'uid'}

    print(f"{'arch':30s}  " + "  ".join(f"{m:>8s}" for m in metrics))
    print("-" * 110)
    for arch, g in df.groupby('arch'):
        row = f"  {arch:28s}  "
        for m in metrics:
            rhos = []
            for t, gt in g.groupby('task'):
                if gt[m].nunique() <= 1: continue
                r, _ = spearmanr(gt[m], gt['lp_acc'])
                if not np.isnan(r): rhos.append(r)
            mu = np.mean(rhos) if rhos else float('nan')
            row += f"  {mu:+.3f}  "
        print(row)

    # Combined (per-task across all archs, then mean)
    row = f"  {'COMBINED (5 tasks)':28s}  "
    for m in metrics:
        per_task_rho = []
        for t, gt in df.groupby('task'):
            if gt[m].nunique() <= 1: continue
            r, _ = spearmanr(gt[m], gt['lp_acc'])
            if not np.isnan(r): per_task_rho.append(r)
        mu = np.mean(per_task_rho) if per_task_rho else float('nan')
        row += f"  {mu:+.3f}  "
    print(row)

    # Joint pooled per-task across archs (16 var × 5 task SCR-style)
    print()
    print(f"{'metric':12s}  joint_mean_rho   pos    LF?")
    for m in metrics:
        rhos = []
        for (a, t), g in df.groupby(['arch', 'task']):
            if g[m].nunique() <= 1: continue
            r, _ = spearmanr(g[m], g['lp_acc'])
            if not np.isnan(r): rhos.append(r)
        rhos = np.array(rhos)
        n_pos = (rhos > 0).sum()
        lf = '✓' if m in label_free else '×'
        print(f"  {m:10s}    {rhos.mean():+.3f}        {n_pos}/{len(rhos)}    {lf}")


if __name__ == "__main__":
    main()
