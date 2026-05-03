"""
A1/A2/A3 analysis for NeurIPS paper revisions.

A1: Flat-score comparison extended to 10 tasks (extended_full_sota.csv)
    - SSR^‡ vs DiSCO vs RankMe flat regret, n=10 task bootstrap
A2: reff_trunc@256 as stage-1 in two-stage pipeline (pool_b_final + truncation_ablation)
    - Compare Trunc256→RankMe vs SSR^‡→RankMe vs DiSCO→RankMe on 5 tasks
A3: Marchenko-Pastur empirical validation
    - reff/d ratio per architecture across tasks
    - Kendall tau: RankMe-rank vs d-rank on each pool
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parent.parent.parent
RES  = ROOT / "results"
OUT  = Path(__file__).resolve().parent
RNG  = np.random.RandomState(42)
N_BOOT = 10_000


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────
def oracle(df: pd.DataFrame, task: str) -> float:
    return df[df.task == task]["lp_acc"].max()


def flat_pick(df: pd.DataFrame, task: str, score_col: str) -> float:
    sub = df[df.task == task]
    idx = sub[score_col].idxmax()
    return sub.loc[idx, "lp_acc"]


def random_pick(df: pd.DataFrame, task: str, rng: np.random.RandomState) -> float:
    sub = df[df.task == task]
    return float(sub["lp_acc"].sample(1, random_state=rng.randint(0, 2**31)).values[0])


def regret_series(df, tasks, score_col, mode="flat", rng=None):
    """Return array of per-task regrets for a flat or random strategy."""
    regs = []
    for t in tasks:
        oracle_lp = oracle(df, t)
        if mode == "flat":
            picked = flat_pick(df, t, score_col)
        elif mode == "random":
            picked = random_pick(df, t, rng or RNG)
        regs.append(oracle_lp - picked)
    return np.array(regs)


def bootstrap_mean_ci(vec, n=N_BOOT, rng=RNG):
    """Bootstrap CI on mean of a vector by task resampling."""
    samples = [vec[rng.choice(len(vec), len(vec), replace=True)].mean() for _ in range(n)]
    return float(np.mean(vec)), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_bootstrap_delta(vec_a, vec_b, n=N_BOOT, rng=RNG):
    """Δ = mean(a) - mean(b), bootstrap CI and p-value."""
    diffs = []
    for _ in range(n):
        idx = rng.choice(len(vec_a), len(vec_a), replace=True)
        diffs.append(vec_a[idx].mean() - vec_b[idx].mean())
    diffs = np.array(diffs)
    p = float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    obs = float(vec_a.mean() - vec_b.mean())
    return obs, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975)), p


# ═══════════════════════════════════════════════════════════════
# A1: FLAT SCORES ON 10 TASKS
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("A1: FLAT SCORE COMPARISON — 10 TASKS (extended_full_sota.csv)")
print("═"*70)

df10 = pd.read_csv(RES / "extended_full_sota.csv")
tasks10 = sorted(df10["task"].unique())
print(f"Tasks ({len(tasks10)}): {tasks10}")
print(f"Variants: {df10['variant'].nunique()}, Archs: {df10['arch'].nunique()}")

# Drop rows with missing scores
score_cols_10 = ["ssr_dagger", "disco", "rankme", "leep", "ncti", "hscore", "logme"]
print("\nMissing values per score column:")
for c in score_cols_10:
    if c in df10.columns:
        print(f"  {c}: {df10[c].isna().sum()} missing")

# For each task, restrict to variants with non-null scores
strategies = {
    "Flat-RankMe":   "rankme",
    "Flat-SSR^‡":    "ssr_dagger",
    "Flat-DiSCO":    "disco",
    "Flat-LEEP":     "leep",
    "Flat-H-score":  "hscore",
}

# Collect per-task regrets
results10 = {}
for name, col in strategies.items():
    regs = []
    for t in tasks10:
        sub = df10[df10.task == t].dropna(subset=[col, "lp_acc"])
        if len(sub) == 0:
            continue
        oracle_lp = sub["lp_acc"].max()
        picked = sub.loc[sub[col].idxmax(), "lp_acc"]
        regs.append(oracle_lp - picked)
    results10[name] = np.array(regs)

# Random baseline (per-task average over 1000 random picks)
random_regs = []
for t in tasks10:
    sub = df10[df10.task == t].dropna(subset=["lp_acc"])
    if len(sub) == 0:
        continue
    oracle_lp = sub["lp_acc"].max()
    picks = [sub["lp_acc"].sample(1, random_state=i).values[0] for i in range(1000)]
    random_regs.append(oracle_lp - np.mean(picks))
results10["Random"] = np.array(random_regs)

print(f"\n{'Strategy':20s}  {'n_tasks':>7s}  {'Mean Regret':>12s}  {'95% CI':>22s}")
print("-" * 70)
for name, regs in results10.items():
    mean_r, lo, hi = bootstrap_mean_ci(regs)
    print(f"{name:20s}  {len(regs):7d}  {mean_r:+12.4f}  [{lo:+.3f}, {hi:+.3f}]")

# Paired bootstrap: SSR^‡ vs DiSCO
if "Flat-SSR^‡" in results10 and "Flat-DiSCO" in results10:
    v_ssr = results10["Flat-SSR^‡"]
    v_disco = results10["Flat-DiSCO"]
    min_n = min(len(v_ssr), len(v_disco))
    v_ssr, v_disco = v_ssr[:min_n], v_disco[:min_n]
    delta, lo, hi, p = paired_bootstrap_delta(v_ssr, v_disco)
    print(f"\nPaired SSR^‡ - DiSCO (regret Δ; negative = SSR^‡ better):")
    print(f"  Δ = {delta:+.4f}  CI = [{lo:+.3f}, {hi:+.3f}]  p = {p:.3f}")

# Also: per-task breakdown
print("\nPer-task regret (Flat-RankMe / Flat-SSR^‡ / Flat-DiSCO):")
print(f"{'Task':15s}  {'Oracle':>7s}  {'RankMe':>9s}  {'SSR^‡':>9s}  {'DiSCO':>9s}  {'RankMe picks':20s}")
print("-" * 80)
for t in tasks10:
    sub = df10[df10.task == t].dropna(subset=["lp_acc", "ssr_dagger", "disco", "rankme"])
    if len(sub) == 0:
        continue
    ora = sub["lp_acc"].max()
    rm_pick = sub.loc[sub["rankme"].idxmax()]
    ssr_pick = sub.loc[sub["ssr_dagger"].idxmax()]
    dis_pick = sub.loc[sub["disco"].idxmax()]
    rm_reg = ora - rm_pick["lp_acc"]
    ssr_reg = ora - ssr_pick["lp_acc"]
    dis_reg = ora - dis_pick["lp_acc"]
    rm_name = rm_pick["variant"] if "variant" in rm_pick.index else "?"
    rm_d = int(rm_pick["feat_dim"]) if "feat_dim" in rm_pick.index else 0
    print(f"{t:15s}  {ora:.3f}  {rm_reg:+9.3f}  {ssr_reg:+9.3f}  {dis_reg:+9.3f}  {rm_name[:18]}(d={rm_d})")


# ═══════════════════════════════════════════════════════════════
# A2: reff_trunc@256 AS STAGE-1
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("A2: reff_trunc@256 AS STAGE-1 IN TWO-STAGE PIPELINE")
print("═"*70)

df_pb  = pd.read_csv(RES / "pool_b_final.csv")
df_tr  = pd.read_csv(RES / "truncation_ablation.csv")

# Restrict pool_b to 35 multi-variant archs (matching paper's main experiment)
ref_task = df_pb["task"].unique()[0]
vc = df_pb[df_pb.task == ref_task].groupby("arch")["variant"].nunique()
multi_archs = vc[vc > 1].index.tolist()
df_pb = df_pb[df_pb.arch.isin(multi_archs)].copy()
print(f"Multi-variant pool: {df_pb['arch'].nunique()} archs, "
      f"{df_pb['variant'].nunique()} variants, {df_pb['task'].nunique()} tasks")

# Merge truncation data into pool_b — key = (variant, task)
trunc_merge = df_tr[["variant", "task", "reff_trunc_256", "feat_dim"]].copy()
df_pb2 = df_pb.merge(trunc_merge, on=["variant", "task"], how="left")
print(f"Merged shape: {df_pb2.shape}")
print(f"reff_trunc_256 missing: {df_pb2['reff_trunc_256'].isna().sum()}")

# Build derived score: rankme_trunc256 = log(reff_trunc_256)
df_pb2["rankme_trunc256"] = np.log(df_pb2["reff_trunc_256"].clip(1e-9))

tasks5 = sorted(df_pb2["task"].unique())
print(f"Tasks: {tasks5}")
archs5 = sorted(df_pb2["arch"].unique())
print(f"Multi-variant archs: {len(archs5)}")

def two_stage_pipeline(df, tasks, s1_col, s2_col, canonical="first"):
    """
    Two-stage pipeline:
    Stage 1: for each arch, score its canonical variant (first alphabetically)
             → select best arch
    Stage 2: among variants of selected arch, pick by s2_col
    Returns per-task array of lp_acc and regret.
    """
    picks = []
    for t in tasks:
        sub = df[df.task == t].dropna(subset=[s1_col, s2_col, "lp_acc"])
        if len(sub) == 0:
            picks.append(np.nan)
            continue
        # Stage 1: one representative per arch
        reps = (sub.groupby("arch")
                   .apply(lambda g: g.sort_values("variant").iloc[0])
                   .reset_index(drop=True))
        best_arch = reps.loc[reps[s1_col].idxmax(), "arch"]
        # Stage 2: all variants of best arch
        variants = sub[sub.arch == best_arch]
        if len(variants) == 0:
            # Fallback: flat pick on s2
            picks.append(sub.loc[sub[s2_col].idxmax(), "lp_acc"])
        else:
            picks.append(variants.loc[variants[s2_col].idxmax(), "lp_acc"])
    return np.array(picks)


oracles5 = np.array([df_pb2[df_pb2.task == t]["lp_acc"].max() for t in tasks5])

# Define strategies
pipelines = {
    "Oracle":              None,
    "Random":             "random",
    "Flat-RankMe":        ("rankme", None),
    "Flat-SSR^‡":         ("ssr", None),
    "Flat-DiSCO":         ("disco", None),
    "Flat-Trunc256":      ("trunc256", None),
    "2S RankMe→RankMe":   ("rankme", "rankme"),
    "2S SSR^‡→RankMe":    ("ssr", "rankme"),
    "2S DiSCO→RankMe":    ("disco", "rankme"),
    "2S Trunc256→RankMe": ("trunc256", "rankme"),
    "2S SSR^‡→DiSCO":     ("ssr", "disco"),
}

col_map = {
    "rankme":   "rankme",
    "ssr":      "ssr",
    "disco":    "disco",
    "trunc256": "rankme_trunc256",
}

def flat_pick_pb(df, task, col):
    sub = df[df.task == task].dropna(subset=[col, "lp_acc"])
    return sub.loc[sub[col].idxmax(), "lp_acc"]

def two_stage_pick_pb(df, task, s1_col, s2_col):
    sub = df[df.task == task].dropna(subset=[s1_col, s2_col, "lp_acc"])
    reps = sub.groupby("arch").apply(lambda g: g.sort_values("variant").iloc[0]).reset_index(drop=True)
    best_arch = reps.loc[reps[s1_col].idxmax(), "arch"]
    variants = sub[sub.arch == best_arch]
    if len(variants) == 0:
        return sub.loc[sub[s2_col].idxmax(), "lp_acc"]
    return variants.loc[variants[s2_col].idxmax(), "lp_acc"]


strat_regs = {}
strat_picks = {}
for name, spec in pipelines.items():
    regs = []
    picks_list = []
    for i, t in enumerate(tasks5):
        ora = oracles5[i]
        if spec is None:
            p = ora
        elif spec == "random":
            sub = df_pb2[df_pb2.task == t]
            p = float(sub["lp_acc"].mean())
        elif spec[1] is None:
            # flat
            c = col_map[spec[0]]
            p = flat_pick_pb(df_pb2, t, c)
        else:
            # two-stage
            c1 = col_map[spec[0]]
            c2 = col_map[spec[1]]
            p = two_stage_pick_pb(df_pb2, t, c1, c2)
        picks_list.append(p)
        regs.append(ora - p)
    strat_regs[name] = np.array(regs)
    strat_picks[name] = picks_list

print(f"\n{'Strategy':25s}  {'Mean LP':>8s}  {'Regret':>8s}  {'95% CI':>22s}")
print("-" * 72)
for name, regs in strat_regs.items():
    mean_lp = np.mean(strat_picks[name])
    mean_r, lo, hi = bootstrap_mean_ci(regs)
    print(f"{name:25s}  {mean_lp:8.3f}  {mean_r:+8.4f}  [{lo:+.3f}, {hi:+.3f}]")

# Paired comparisons
print("\n--- Paired bootstrap comparisons (regret Δ; negative = A better than B) ---")
pairs = [
    ("2S SSR^‡→RankMe",    "2S Trunc256→RankMe",  "SSR^‡ vs Trunc256 (stage-1)"),
    ("2S Trunc256→RankMe", "2S RankMe→RankMe",    "Trunc256 vs RankMe (stage-1)"),
    ("2S SSR^‡→RankMe",    "2S DiSCO→RankMe",     "SSR^‡ vs DiSCO (stage-1)"),
    ("Flat-SSR^‡",         "Flat-DiSCO",          "Flat SSR^‡ vs DiSCO"),
    ("Flat-Trunc256",      "Flat-DiSCO",          "Flat Trunc256 vs DiSCO"),
    ("2S SSR^‡→RankMe",    "Flat-RankMe",         "S5 vs Flat-RankMe (main result)"),
]
for a, b, label in pairs:
    if a not in strat_regs or b not in strat_regs:
        continue
    delta, lo, hi, p = paired_bootstrap_delta(strat_regs[a], strat_regs[b])
    print(f"  {label:40s}  Δ={delta:+.4f}  CI=[{lo:+.3f},{hi:+.3f}]  p={p:.3f}")


# ═══════════════════════════════════════════════════════════════
# A3: MARCHENKO-PASTUR EMPIRICAL VALIDATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "═"*70)
print("A3: MARCHENKO-PASTUR EMPIRICAL VALIDATION")
print("═"*70)

# A3a: reff/d ratio per architecture (all three pools)
print("\n--- A3a: reff / feat_dim per variant (extended 10-task pool) ---")
df10_a = df10.copy()
df10_a["reff_over_d"] = df10_a["reff"] / df10_a["feat_dim"].clip(1)
ratio_by_arch = (df10_a.groupby("arch")["reff_over_d"]
                  .agg(["mean","std","min","max","count"])
                  .sort_values("mean"))
print(ratio_by_arch.round(3).to_string())
print(f"\nAcross all rows:  mean={df10_a['reff_over_d'].mean():.3f}  "
      f"std={df10_a['reff_over_d'].std():.3f}  "
      f"CV={df10_a['reff_over_d'].std()/df10_a['reff_over_d'].mean():.3f}")

# A3b: Kendall tau between RankMe-rank and d-rank per task (Pool B)
print("\n--- A3b: Kendall tau(RankMe_rank, d_rank) per task — Pool B ---")
for t in tasks5:
    sub = df_pb2[df_pb2.task == t].dropna(subset=["rankme", "feat_dim"]).copy()
    tau, p = kendalltau(sub["rankme"], sub["feat_dim"])
    print(f"  {t:15s}  tau={tau:+.3f}  p={p:.4f}  n={len(sub)}")

# Same for extended pool (10 tasks)
print("\n--- A3b: Kendall tau(RankMe_rank, d_rank) per task — extended 10-task pool ---")
all_taus = []
for t in tasks10:
    sub = df10[df10.task == t].dropna(subset=["rankme", "feat_dim"])
    if len(sub) < 5:
        continue
    tau, p = kendalltau(sub["rankme"], sub["feat_dim"])
    all_taus.append(tau)
    print(f"  {t:15s}  tau={tau:+.3f}  p={p:.4f}  n={len(sub)}")
print(f"  MEAN tau={np.mean(all_taus):+.3f}")

# A3c: Spearman rho(RankMe, d) for the VTAB pool (vtab data)
print("\n--- A3c: reff/d per architecture — Pool B (pool_b_final.csv) ---")
df_pb2["reff_over_d"] = df_pb2["reff"] / df_pb2["feat_dim"].clip(1)
df_pb2 = df_pb2.copy()  # suppress SettingWithCopyWarning
by_arch_pb = (df_pb2.groupby("arch")
              .agg(feat_dim=("feat_dim","first"),
                   reff_mean=("reff","mean"),
                   reff_over_d_mean=("reff_over_d","mean"),
                   reff_over_d_std=("reff_over_d","std"))
              .sort_values("feat_dim"))
print(by_arch_pb.round(3).head(20).to_string())

# Coefficient of variation of reff/d across pool
cv = df_pb2["reff_over_d"].std() / df_pb2["reff_over_d"].mean()
print(f"\nPool B reff/d: mean={df_pb2['reff_over_d'].mean():.3f}  std={df_pb2['reff_over_d'].std():.3f}  CV={cv:.3f}")

# A3d: Within-arch CV of reff/d vs across-arch variance
print("\n--- A3d: Within-arch CV of reff/d (across tasks) ---")
within = (df_pb2.groupby(["arch","variant"])["reff_over_d"]
          .agg(["mean","std"])
          .dropna())
within["cv"] = within["std"] / within["mean"]
print(f"Median within-arch CV of reff/d: {within['cv'].median():.3f}")
print(f"Mean:  {within['cv'].mean():.3f}")
print(f"Max:   {within['cv'].max():.3f}")

print("\n" + "═"*70)
print("DONE — A1/A2/A3 analysis complete")
print("═"*70)
