"""
reproduce_all.py — Reproduce all main results for the paper
"Cross-Architecture Transferability is Width-Confounded: Diagnosis and Subspace Correction"

Usage
-----
    python reproduce_all.py [--step STEP] [--all] [--verify]

Steps
-----
  1  Score MV pool (132v × 5t)          → results/pool_b_final.csv
  2  Truncation ablation                 → results/truncation_ablation.csv
  3  Extended 51v × 10t pool             → results/extended_full_sota.csv
  4  Two-stage pipeline (MV pool)        → results/c5_twostage_pipeline.csv
  5  SSL/multimodal pool (8 encoders)    → results/paper14_crossdomain.csv
  6  Cross-pool bootstrap CI             → results/cross_pool_regrets.csv
  7  Mixed-budget label sweep            → results/p2_label_budget_sweep.csv
  8  VTAB 25-arch × 8-task pool          → results/vtab_25arch_8tasks.csv
  9  VTAB reference-task & k ablations   → results/c8_k_ablation.csv
 10  A1/A2/A3 analysis (extended pool,   → results/multi_ref_sa_k.csv
     truncation-as-stage-1, MP check)
 11  Figures                             → paper/fig_pareto.pdf
                                            paper/fig_costsplit.pdf
                                            paper/fig_sigma_b_flow.pdf

Feature extraction (GPU required, run before steps 1-3, 5, 8-9)
-----
  --extract-mv     Extract MV pool features   (pool_b / vtab_features/)
  --extract-vtab   Extract VTAB pool features (vtab_features/)
  --extract-ssl    Extract SSL pool features  (paper14_features/)

Pre-computed result CSVs are shipped in results/ so reviewers can verify
numbers without re-extracting features.  Feature .npz files are not shipped
(too large); see FEATURE_EXTRACTION.md for instructions.
"""
import subprocess, sys, os, argparse, json
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent   # repository root
CODE  = ROOT / "code"
RES   = ROOT / "results"
PAPER = ROOT / "paper"

# ─── scoring / analysis steps ────────────────────────────────────────────────
STEPS = {
    1:  ("Score MV pool variants (132v × 5t)",
         f"python {CODE}/score_pool_b_final.py"),
    2:  ("Truncation ablation (reff_trunc at D=128/256/512/768)",
         f"python {CODE}/analyze_n0_truncation.py"),
    3:  ("Extended 51-variant × 10-task pool scoring",
         f"python {CODE}/score_extended_pool.py"),
    4:  ("Two-stage pipeline evaluation (MV pool, 35 archs × 5 tasks)",
         f"python {CODE}/audit_C5_twostage.py"),
    5:  ("SSL/multimodal pool scoring (8 encoders × 29 tasks)",
         f"python {CODE}/audit_paper14_crossdomain.py"),
    6:  ("Audited common-support statistics and official DiSCO summaries",
         (f"python {CODE}/summarize_official_disco.py && "
          f"python {CODE}/build_camera_ready_results.py")),
    7:  ("Mixed-budget label sweep (H-score→RankMe, K=1..35)",
         f"python {CODE}/audit_P1_variance_decomp.py"),
    8:  ("VTAB pool scoring (25 archs × 8 tasks, flat + two-stage)",
         f"python {CODE}/analysis/b5_c78_fast.py"),
    9:  ("VTAB reference-task ablation & k ablation (C7/C8)",
         f"python {CODE}/multi_reference_sa_k.py"),
    10: ("A1/A2/A3 analysis (extended pool flat, trunc stage-1, MP validation)",
         f"python {CODE}/analysis/a123_analysis.py"),
    11: ("VTAB f_spe analysis (rho(SA_k, f_spe)=-0.964, 126 pairs)",
         f"python {CODE}/compute_vtab_fspe.py"),
    12: ("Generate figures (Pareto, cost-split, sigma-b flow)",
         (f"python {CODE}/figures_pareto.py && "
          f"python {CODE}/figures_costsplit.py && "
          f"python {CODE}/figure_sigma_b_flow.py")),
}

# ─── feature extraction helpers ──────────────────────────────────────────────
EXTRACT = {
    "mv":   ("Extract MV pool features (timm/torchvision, GPU required)",
             [f"python {CODE}/extract_phase_b_features.py",
              f"python {CODE}/extract_phase_d_features.py",
              f"python {CODE}/extract_phase_e_torchvision.py"]),
    "vtab": ("Extract VTAB 25-arch features (GPU required)",
             [f"python {CODE}/extract_phase_d_features.py"]),
    "ssl":  ("Extract SSL/multimodal pool features (GPU required)",
             [f"python {CODE}/extract_phase_f_dino_clip.py"]),
}


def run_step(n):
    label, cmd = STEPS[n]
    print(f"\n{'='*70}")
    print(f"Step {n}: {label}")
    print(f"{'='*70}")
    print(f"$ {cmd}\n")
    ret = subprocess.run(cmd, shell=True, cwd=ROOT)
    if ret.returncode != 0:
        print(f"[ERROR] Step {n} failed (exit code {ret.returncode})")
        sys.exit(1)
    print(f"[OK] Step {n} completed")


def run_extract(pool):
    label, cmds = EXTRACT[pool]
    print(f"\n{'='*70}")
    print(f"Feature extraction: {label}")
    print(f"{'='*70}")
    for cmd in cmds:
        print(f"$ {cmd}\n")
        ret = subprocess.run(cmd, shell=True, cwd=ROOT)
        if ret.returncode != 0:
            print(f"[ERROR] extraction failed (exit code {ret.returncode})")
            sys.exit(1)
    print(f"[OK] feature extraction completed")


def verify_results():
    """Cross-check headline numbers against shipped result CSVs."""
    import pandas as pd, numpy as np
    from scipy.stats import kendalltau

    print("\n" + "="*70)
    print("Verifying headline numbers against results/")
    print("="*70)
    EPS = 1e-12
    ok = True

    def check(label, got, expected, tol=0.005):
        nonlocal ok
        flag = "✓" if abs(got - expected) < tol else "✗ MISMATCH"
        if flag.startswith("✗"): ok = False
        print(f"  {label:55s}  {got:.3f}  (paper: {expected:.3f})  {flag}")

    # ── MV pool flat regrets ─────────────────────────────────────
    pb = pd.read_csv(RES / "pool_b_final.csv")
    pb["ssr_dagger"] = (np.log(np.maximum(pb["reff"], EPS)) -
                        np.log(np.maximum(pb["sa_k"], EPS))) / np.log(pb["n_class"])
    multi = pb.groupby("arch")["variant"].nunique()
    pb_m  = pb[pb.arch.isin(multi[multi > 1].index)].copy()
    tasks = sorted(pb_m.task.unique())
    oracle = {t: pb_m[pb_m.task == t]["lp_acc"].max() for t in tasks}

    rm_reg  = np.mean([oracle[t] - pb_m[pb_m.task == t].loc[
                pb_m[pb_m.task == t]["rankme"].idxmax(), "lp_acc"] for t in tasks])
    ssr_reg = np.mean([oracle[t] - pb_m[pb_m.task == t].loc[
                pb_m[pb_m.task == t]["ssr_dagger"].idxmax(), "lp_acc"] for t in tasks])
    check("MV pool  Flat-RankMe regret",  rm_reg,  0.319)
    check("MV pool  Flat-SSR^‡ regret",   ssr_reg, 0.098)

    # ── Extended pool ────────────────────────────────────────────
    df_ext = pd.read_csv(RES / "extended_full_sota.csv")
    taus = []
    for t in df_ext.task.unique():
        sub = df_ext[df_ext.task == t].dropna(subset=["rankme", "feat_dim"])
        if len(sub) > 5:
            tau, _ = kendalltau(sub["rankme"], sub["feat_dim"])
            taus.append(tau)
    check("Extended pool  mean Kendall τ(RankMe, d)", np.mean(taus), 0.617, tol=0.01)

    regs = []
    for t in df_ext.task.unique():
        sub = df_ext[df_ext.task == t].dropna(subset=["ssr_dagger", "lp_acc"])
        if len(sub):
            regs.append(sub["lp_acc"].max() -
                        sub.loc[sub["ssr_dagger"].idxmax(), "lp_acc"])
    check("Extended pool  Flat-SSR^‡ regret",   np.mean(regs), 0.060)

    # ── Two-stage pipeline ───────────────────────────────────────
    if (RES / "c5_twostage_pipeline.csv").exists():
        ts = pd.read_csv(RES / "c5_twostage_pipeline.csv")
        row = ts[ts.strategy == "S5_twostage_ssr"]
        if len(row):
            check("Two-stage 2S SSR^‡→RankMe regret",
                  float(row.iloc[0]["mean_regret"]), 0.102)

    # ── Audited common support ───────────────────────────────────
    headline_path = RES / "camera_ready" / "headline_metrics.json"
    if headline_path.exists():
        headline = json.loads(headline_path.read_text())
        common = headline["common_support"]
        check("Common support (22 units) SSR^‡ mean regret",
              float(common["ssr_mean_regret"]), 0.06636, tol=0.0001)
        check("Common support (22 units) RankMe mean regret",
              float(common["rankme_mean_regret"]), 0.16886, tol=0.0001)
        check("Common support paired difference SSR^‡ - RankMe",
              float(common["mean_difference_ssr_minus_rankme"]),
              -0.1025, tol=0.0001)

        disco = headline["official_label_dependent_disco"]
        check("Official DiSCO MV G=6 regret (label-dependent)",
              float(disco["mv_g6_regret"]), 0.146, tol=0.0001)
        check("Official DiSCO VTAB G=8 regret (STL-10 excluded)",
              float(disco["vtab_g8_regret"]), 0.04357, tol=0.0001)
        check("Official DiSCO EX G=6 regret (label-dependent)",
              float(disco["ex_g6_regret"]), 0.045, tol=0.0001)

    # ── Mixed budget ─────────────────────────────────────────────
    if (RES / "p2_label_budget_sweep.csv").exists():
        p2 = pd.read_csv(RES / "p2_label_budget_sweep.csv")
        k35 = p2[p2.budget == 35]
        if len(k35):
            check("Mixed budget K=35 regret", float(k35.iloc[0]["regret"]), 0.018, tol=0.003)

    # ── VTAB f_spe geometric premise ─────────────────────────────
    if (RES / "vtab_fspe_analysis.csv").exists():
        from scipy.stats import spearmanr as _sp
        fspe = pd.read_csv(RES / "vtab_fspe_analysis.csv")
        valid = fspe.dropna(subset=["sa_k", "frac_spe"])
        if len(valid) >= 100:
            rho, _ = _sp(valid["sa_k"], valid["frac_spe"])
            check("VTAB rho_Sp(SA_k, f_spe)  (n=126)", rho, -0.964, tol=0.01)

    print()
    print("All checks passed ✓" if ok else "Some checks FAILED — see ✗ above")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument("--step",  type=int, choices=list(STEPS),
                        help="Run a single step")
    parser.add_argument("--all",   action="store_true",
                        help="Run all scoring + analysis steps (1-12)")
    parser.add_argument("--verify",action="store_true",
                        help="Verify headline numbers without re-running")
    parser.add_argument("--extract-mv",   action="store_true",
                        help="Extract MV pool features (GPU)")
    parser.add_argument("--extract-vtab", action="store_true",
                        help="Extract VTAB pool features (GPU)")
    parser.add_argument("--extract-ssl",  action="store_true",
                        help="Extract SSL pool features (GPU)")
    args = parser.parse_args()

    if args.extract_mv:
        run_extract("mv")
    elif args.extract_vtab:
        run_extract("vtab")
    elif args.extract_ssl:
        run_extract("ssl")
    elif args.verify:
        verify_results()
    elif args.all:
        for n in sorted(STEPS):
            run_step(n)
        verify_results()
    elif args.step:
        run_step(args.step)
    else:
        print(__doc__)
        print("Available steps:")
        for n, (label, _) in STEPS.items():
            print(f"  {n:2d}: {label}")
