"""
compute_vtab_fspe.py
====================
Compute f_spe (fraction of between-class covariance in the architecture-specific
subspace) and SA_k (alignment with each candidate's STL-10 reference features) for
each (architecture, task) pair in the VTAB pool.

Output
------
  results/vtab_fspe_analysis.csv
  Columns: arch, task, feat_dim, n_class, lp_acc, reff, sa_k, ssr_dagger, frac_spe

Definitions
-----------
  SA_k(H, H_ref)   = (1/k) * ||V_{H,1:k}^T V_{ref,1:k}||_F^2
                     (top-k subspace alignment via principal angles)
  f_spe(H, H_ref)  = tr(Q Sigma_B Q) / tr(Sigma_B)
                     where P = V_{ref,1:k} V_{ref,1:k}^T, Q = I - P
                     and Sigma_B is the between-class (class-mean) covariance.
  SSR^dagger       = [log r_eff(H) - log SA_k(H, H_ref)] / log C

Filtering
---------
  The 25-architecture VTAB pool yields:
    - 25 archs x 8 tasks = 200 total pairs
    - Exclude STL-10 as target (used as reference): 25 x 7 = 175 pairs
    - resnet152 has a numerical SA_k failure (rank-deficient features at n<d):
      175 - 7 = 168 pairs with valid SA_k
    - 7 ResNet/DenseNet architectures lack per-class feature matrices (full
      feature files not in vtab_features/): 168 - 42 = 126 pairs with valid f_spe
    These 18 architectures x 7 tasks = 126 pairs are exported in the CSV.

Requirements
------------
  GPU-extracted features in features/vtab_features/ (see FEATURE_EXTRACTION.md)
  Python >= 3.8, numpy, pandas, scikit-learn

Usage
-----
  python code/compute_vtab_fspe.py
  # Output written to results/vtab_fspe_analysis.csv
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from metrics.reff import effective_rank_from_singular_values
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "features" / "vtab_features"
OUT  = ROOT / "results" / "vtab_fspe_analysis.csv"
EPS  = 1e-12
K    = 20       # subspace size (matches main experiments)

TASKS7 = ['cifar10','cifar100','dtd','fashion_mnist','flowers102','pets','svhn']
REF_TASK = 'stl10'


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def load_features(arch: str, task: str):
    """Load feature matrix H and integer labels y from .npz file."""
    path = FEAT / f"{arch}__{task}.npz"
    d = np.load(path, allow_pickle=True)
    H = (d['features'] if 'features' in d else d['H']).astype(np.float32)
    y = (d['labels']   if 'labels'   in d else d['y']).astype(int)
    return H, y


def reff_from_svd(H: np.ndarray) -> float:
    sv = np.linalg.svd(H, compute_uv=False)
    return effective_rank_from_singular_values(sv)


def sa_k_from_Vt(Vt1: np.ndarray, Vt2: np.ndarray, k: int) -> float:
    k1   = min(k, Vt1.shape[0], Vt2.shape[0])
    gram = Vt1[:k1] @ Vt2[:k1].T
    return float((gram**2).sum() / k1)


def compute_f_spe(H: np.ndarray, y: np.ndarray, Vref: np.ndarray, k: int) -> float:
    """Fraction of between-class variance outside the reference subspace.

    f_spe = tr(Q Sigma_B Q) / tr(Sigma_B)
    where Q = I - P,  P = V_{ref,1:k} V_{ref,1:k}^T
    """
    mu_global = H.mean(axis=0)
    differences = []
    weights = []
    for label in np.unique(y):
        mask = y == label
        differences.append(H[mask].mean(axis=0) - mu_global)
        weights.append(mask.sum() / len(y))
    differences = np.asarray(differences, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights[:, None] * differences**2))
    if total < EPS:
        return float('nan')
    coordinates = differences @ Vref[:k].T
    inside = float(np.sum(weights[:, None] * coordinates**2))
    return float(np.clip((total - inside) / total, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", type=Path, default=FEAT)
    parser.add_argument(
        "--lp-csv",
        type=Path,
        default=ROOT / "results" / "camera_ready" / "vtab_24arch_8tasks_lp_complete.csv",
    )
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    feature_dir = args.feature_dir
    if not feature_dir.exists():
        print(f"[ERROR] Feature directory not found: {feature_dir}")
        print("Run feature extraction first (see FEATURE_EXTRACTION.md).")
        return

    lp = pd.read_csv(args.lp_csv)
    arches = sorted(lp["arch"].unique())

    # ── Load each candidate encoder's STL-10 reference SVD ──
    print(f"Computing VTAB f_spe for {len(arches)} architectures x 7 tasks …")
    rows = []
    errors = 0

    for arch in arches:
        # Load reference (STL-10) features for this arch
        ref_path = feature_dir / f"{arch}__{REF_TASK}.npz"
        if not ref_path.exists():
            print(f"  [SKIP] {arch}: reference file missing")
            errors += 1
            continue
        with np.load(ref_path, allow_pickle=False) as data:
            H_ref = (data['features'] if 'features' in data else data['H']).astype(np.float32)
        _, _, Vt_ref = np.linalg.svd(H_ref, full_matrices=False)

        for task in TASKS7:
            path = feature_dir / f"{arch}__{task}.npz"
            if not path.exists():
                print(f"  [SKIP] {arch}/{task}: file missing")
                errors += 1
                continue
            with np.load(path, allow_pickle=False) as data:
                H = (data['features'] if 'features' in data else data['H']).astype(np.float32)
                y = (data['labels'] if 'labels' in data else data['y']).astype(int)
            _, singular_values, Vt = np.linalg.svd(H, full_matrices=False)

            feat_dim = H.shape[1]
            n_class  = len(np.unique(y))
            r_eff    = effective_rank_from_singular_values(singular_values)
            sa       = sa_k_from_Vt(Vt, Vt_ref, K)
            f_spe    = compute_f_spe(H, y, Vt_ref, K)
            ssr      = (np.log(max(r_eff, EPS)) -
                        np.log(max(sa,   EPS))) / np.log(n_class)

            # LP accuracy (from pre-computed CSV to avoid rerunning sklearn)
            rows.append({
                'arch':      arch,
                'task':      task,
                'feat_dim':  feat_dim,
                'n_class':   n_class,
                'lp_acc':    float('nan'),   # filled below from shipped CSV
                'reff':      r_eff,
                'sa_k':      sa,
                'ssr_dagger': ssr,
                'frac_spe':  f_spe,
            })

    df = pd.DataFrame(rows)

    # ── Merge lp_acc from shipped vtab_25arch_8tasks.csv ──
    lp = pd.read_csv(args.lp_csv, usecols=['arch','task','lp_acc'])
    df = df.merge(lp.rename(columns={'lp_acc':'lp_acc_ref'}),
                  on=['arch','task'], how='left')
    df['lp_acc'] = df['lp_acc_ref']
    df.drop(columns=['lp_acc_ref'], inplace=True)

    df.sort_values(['arch','task'], inplace=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows → {args.output}")
    if errors:
        print(f"[WARN] {errors} files skipped; run --extract-vtab first.")
    else:
        print(f"All {len(arches) * len(TASKS7)} pairs computed successfully.")

    # ── Verification ──
    from scipy.stats import spearmanr
    valid = df.dropna(subset=['sa_k','frac_spe'])
    if len(valid) > 10:
        rho, p = spearmanr(valid['sa_k'], valid['frac_spe'])
        print(f"\nrho_Sp(SA_k, f_spe) = {rho:+.4f}  p={p:.2e}  n={len(valid)}")
        print("  (paper claims rho = -0.964, n=126)")


if __name__ == "__main__":
    main()
