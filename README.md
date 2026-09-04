# Cross-Architecture Transferability is Width-Confounded: Diagnosis and Subspace Correction

Reproduction package for the NeurIPS 2026 submission.

## Repository layout

```
repository/
├── code/
│   ├── metrics/                   # Core metric implementations
│   │   ├── subspace_alignment.py  # SA_k: subspace alignment (principal angles)
│   │   ├── reff.py                # r_eff: effective rank (RankMe)
│   │   ├── ssr.py                 # Canonical SSR† definition
│   │   ├── grassmann.py           # Grassmann geometry utilities
│   │   └── direction_gain.py      # Direction-gain auxiliary metric
│   ├── analysis/
│   │   ├── a123_analysis.py       # A1/A2/A3: extended pool, truncation, MP check
│   │   └── b5_c78_fast.py         # B5/C7/C8: VTAB ref-task & k ablations
│   ├── baselines.py               # LogME, H-score, TransRate, linear probe
│   ├── run_final_benchmark.py     # Compute all baselines on a feature cache
│   ├── score_pool_b_final.py      # MV pool (132v × 5t) scoring → pool_b_final.csv
│   ├── analyze_n0_truncation.py   # Truncation ablation → truncation_ablation.csv
│   ├── score_extended_pool.py     # Extended 51v × 10t → extended_full_sota.csv
│   ├── audit_C5_twostage.py       # Two-stage pipeline (MV pool) → two_stage_sota_*.csv
│   ├── twostage_extended.py       # Extended two-stage & 3-way comparison
│   ├── audit_paper14_crossdomain.py # SSL/multimodal pool → paper14_crossdomain.csv
│   ├── cross_pool_bootstrap.py    # Cross-pool CI bootstrap → cross_pool_regrets.csv
│   ├── audit_P1_variance_decomp.py  # Mixed-budget sweep → p2_label_budget_sweep.csv
│   ├── multi_reference_sa_k.py    # Multi-reference SA_k → multi_ref_sa_k.csv
│   ├── extract_phase_b_features.py  # Feature extraction: MV pool (timm variants)
│   ├── extract_phase_d_features.py  # Feature extraction: VTAB / extended pool
│   ├── extract_phase_e_torchvision.py # Feature extraction: torchvision variants
│   ├── extract_phase_f_dino_clip.py   # Feature extraction: SSL encoders
│   ├── figures_pareto.py          # Figure: cost-regret Pareto
│   ├── figures_costsplit.py       # Figure: compute vs label budget
│   ├── figure_sigma_b_flow.py     # Figure: SA_k vs frac_spe scatter
│   ├── config.py                  # Shared constants
│   ├── datasets.py                # Dataset loaders
│   └── reproduce_all.py           # Master reproduction script
├── results/                       # Pre-computed result CSVs (shipped)
│   ├── pool_b_final.csv           # MV pool: all scores + LP accuracy
│   ├── vtab_25arch_8tasks.csv     # VTAB pool: flat comparison
│   ├── vtab_ssr_results.csv       # VTAB pool: SSR^‡ details
│   ├── extended_full_sota.csv     # Extended pool: all methods
│   ├── truncation_ablation.csv    # Truncation ablation
│   ├── c5_twostage_pipeline.csv   # MV pool two-stage details
│   ├── paper14_crossdomain.csv    # SSL/multimodal pool
│   ├── cross_pool_regrets.csv     # Cross-pool bootstrap CIs
│   ├── p2_label_budget_sweep.csv  # Mixed-budget K=1..35 sweep
│   ├── c8_k_ablation.csv          # VTAB k ablation
│   ├── within_arch_baselines.csv  # Within-arch baselines (Phase A+B)
│   ├── baselines_raw.csv          # Raw baseline scores
│   ├── p1_variance_decomp.csv     # Variance decomposition
│   └── vtab_fspe_analysis.csv     # VTAB f_spe: 126 arch-task pairs (§5 geometric premise)
├── features/                      # Feature .npz files (NOT shipped, see below)
├── paper/                         # LaTeX source + compiled PDFs
│   ├── main_v18.tex / main_v18.pdf
│   ├── main_v18_zh.tex / main_v18_zh.pdf
│   ├── references.bib
│   └── fig_*.pdf                  # All paper figures
├── config.yaml                    # Path / hyperparameter configuration
└── requirements.txt               # Python dependencies
```

## Quick start: verify numbers without GPU

All result CSVs are pre-computed and shipped. Run the verifier:

```bash
cd repository
pip install -r requirements.txt
python code/reproduce_all.py --verify
```

Expected output:
```
MV pool  Flat-RankMe regret               0.319  (paper: 0.319)  ✓
MV pool  Flat-SSR^‡ regret                0.098  (paper: 0.098)  ✓
Extended pool  mean Kendall τ(RankMe, d)  0.617  (paper: 0.617)  ✓
Extended pool  Flat-SSR^‡ regret          0.060  (paper: 0.060)  ✓
Mixed budget K=35 regret                  0.018  (paper: 0.018)  ✓
```

## Full reproduction (GPU required for feature extraction)

```bash
# Step 0: extract features (one-time, ~6h on a V100 for all three pools)
python code/reproduce_all.py --extract-mv    # MV pool (132 models × 6 tasks)
python code/reproduce_all.py --extract-vtab  # VTAB pool (25 models × 8 tasks)
python code/reproduce_all.py --extract-ssl   # SSL pool (8 encoders × 6 tasks)

# Steps 1-11: scoring, analysis, figures
python code/reproduce_all.py --all
```

Or run individual steps:
```bash
python code/reproduce_all.py --step 1   # MV pool scoring
python code/reproduce_all.py --step 6   # Cross-pool bootstrap CI
python code/reproduce_all.py --step 11  # Figures
```

See `FEATURE_EXTRACTION.md` for detailed GPU setup instructions.

## SSR^‡ formula

```
SSR^‡(H, H_ref) = [log r_eff(H) - log SA_k(H, H_ref)] / log C
```

where `r_eff` is RankMe effective rank with
`p_i = sigma_i / sum_j sigma_j`. `SA_k` compares the target features with
features extracted by the same candidate encoder on a fixed reference
dataset; no cross-encoder feature-space projection is performed. `C` is the
number of classes. Implemented in `code/metrics/`.

Superseded UID-based SSR and 22-variant two-stage experiments are retained
under `archive/` and excluded from reproduction.

## Audited camera-ready results

| Method          | MV pool regret | VTAB regret | Extended regret |
|-----------------|:--------------:|:-----------:|:---------------:|
| RankMe (flat)   | +0.319         | +0.097      | +0.144          |
| SSR^‡ (flat, zero label) | **+0.098** | **+0.053** | **+0.060** |
| Official DiSCO G=6 (target labels) | +0.146 | +0.061 | +0.045 |
| Official DiSCO G=8 (target labels) | +0.146 | +0.044 | +0.049 |
| 2S SSR^‡→RankMe | +0.102 (3.1× cheaper) | —    | —               |
| Mixed K=35      | +0.018 (5.7× better)  | —    | —               |

The primary equal-budget result uses the common support of 22 MV/VTAB/EX
pool–task units (STL-10 excluded): SSR^‡ regret is 0.066 and RankMe regret is
0.169, a paired difference of -0.1025. See
`results/camera_ready/headline_metrics.json` for both unit-level and
target-name-clustered uncertainty estimates.
