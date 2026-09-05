"""Build the canonical camera-ready headline results from audited CSV files.

This script is the single source of truth for numbers quoted in the revised
paper. It never reads the superseded legacy DiSCO column from the top-level
pool CSVs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "camera_ready"
N_BOOT = 100_000
SEED = 42


def sign_p(samples: np.ndarray) -> float:
    return float(min(1.0, 2 * min(np.mean(samples >= 0), np.mean(samples <= 0))))


def main() -> None:
    source_rows = []
    for pool, filename in [
        ("MV", "pool_b_final_score_complete.csv"),
        ("EX", "extended_full_sota_score_complete.csv"),
    ]:
        table = pd.read_csv(OUTPUT / filename)
        if pool == "MV":
            variant_counts = table.groupby("arch")["variant"].nunique()
            table = table[table["arch"].isin(variant_counts[variant_counts >= 2].index)]
        if table[["lp_acc", "rankme", "ssr_dagger"]].isna().any().any():
            raise RuntimeError(f"{filename} is not score-complete")
        for task, group in table.groupby("task"):
            oracle = float(group["lp_acc"].max())
            rankme_lp = float(group.loc[group["rankme"].idxmax(), "lp_acc"])
            ssr_lp = float(group.loc[group["ssr_dagger"].idxmax(), "lp_acc"])
            random_mean = float(group["lp_acc"].mean())
            source_rows.append(
                {
                    "pool": pool,
                    "task": task,
                    "oracle": oracle,
                    "random_mean": random_mean,
                    "random_regret": oracle - random_mean,
                    "rankme_regret": oracle - rankme_lp,
                    "ssr_regret": oracle - ssr_lp,
                }
            )
    source = pd.DataFrame(source_rows)
    vtab_lp = pd.read_csv(OUTPUT / "vtab_24arch_8tasks_lp_complete.csv")
    vtab_rankme = pd.read_csv(
        OUTPUT / "vtab_reference_24arch_uncentered" / "rankme_score_per_task.csv"
    ).set_index("task")
    vtab_ssr = pd.read_csv(
        OUTPUT / "vtab_reference_24arch_uncentered" / "stl10_only_ssr_per_task.csv"
    ).set_index("task")
    vtab_rows = []
    for task, group in vtab_lp[vtab_lp["task"] != "stl10"].groupby("task"):
        oracle = float(group["lp_acc"].max())
        random_mean = float(group["lp_acc"].mean())
        vtab_rows.append(
            {
                "pool": "VTAB",
                "task": task,
                "oracle": oracle,
                "random_mean": random_mean,
                "random_regret": oracle - random_mean,
                "rankme_regret": float(vtab_rankme.loc[task, "regret"]),
                "ssr_regret": float(vtab_ssr.loc[task, "regret"]),
            }
        )
    source = pd.concat([source, pd.DataFrame(vtab_rows)], ignore_index=True)
    source["task"] = source["task"].str.lower()
    common = source[source["task"] != "stl10"].dropna(
        subset=["ssr_regret", "rankme_regret"]
    ).copy()
    common["difference"] = common["ssr_regret"] - common["rankme_regret"]

    unit_rng = np.random.default_rng(SEED)
    values = common["difference"].to_numpy()
    unit_boot = unit_rng.choice(
        values, size=(N_BOOT, len(values)), replace=True
    ).mean(1)

    targets = np.array(sorted(common["task"].unique()))
    clusters = {
        target: common.loc[common.task == target, "difference"].to_numpy()
        for target in targets
    }
    cluster_rng = np.random.default_rng(SEED)
    cluster_boot = np.empty(N_BOOT)
    for index in range(N_BOOT):
        sampled = cluster_rng.choice(targets, size=len(targets), replace=True)
        cluster_boot[index] = np.concatenate([clusters[t] for t in sampled]).mean()

    disco = pd.read_csv(RESULTS / "disco" / "official_disco_regret_summary.csv")
    reference = json.loads(
        (
            OUTPUT
            / "vtab_reference_24arch_uncentered"
            / "vtab_reference_audit_summary.json"
        ).read_text()
    )
    reference_scores = {
        row["score"]: row["mean_regret"] for row in reference["scores"]
    }

    wilcoxon_result = wilcoxon(values, alternative="two-sided")
    summary = {
        "protocol": {
            "effective_rank_probabilities": "p_i = sigma_i / sum_j sigma_j",
            "ssr": "(log(r_eff) - log(max(SA_k, 1e-12))) / log(C)",
            "subspace_k": 20,
            "svd_centered": False,
            "reference_dataset": "STL-10",
            "common_support_excludes": ["STL-10"],
        },
        "common_support": {
            "n_pool_task_units": int(len(common)),
            "n_unique_targets": int(len(targets)),
            "pool_counts": common.groupby("pool").size().to_dict(),
            "ssr_mean_regret": float(common.ssr_regret.mean()),
            "rankme_mean_regret": float(common.rankme_regret.mean()),
            "mean_difference_ssr_minus_rankme": float(values.mean()),
            "unit_bootstrap_ci_95": np.percentile(unit_boot, [2.5, 97.5]).tolist(),
            "unit_bootstrap_two_sided_sign_p": sign_p(unit_boot),
            "target_clustered_ci_95": np.percentile(
                cluster_boot, [2.5, 97.5]
            ).tolist(),
            "target_clustered_two_sided_sign_p": sign_p(cluster_boot),
            "paired_wilcoxon_statistic": float(wilcoxon_result.statistic),
            "paired_wilcoxon_p": float(wilcoxon_result.pvalue),
        },
        "vtab_reference_ablation": {
            "n_architectures": 24,
            "n_tasks": 7,
            "stl10_single_reference_regret": reference_scores["stl10_only_ssr"],
            "four_reference_loo_regret": reference_scores[
                "four_reference_loo_ssr"
            ],
        },
        "official_label_dependent_disco": {
            f"{row.pool.lower()}_g{int(row.groups)}_regret": float(row.mean_regret)
            for row in disco.itertuples(index=False)
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    common.to_csv(OUTPUT / "common_support_per_task.csv", index=False)
    pd.DataFrame({"mean_difference": unit_boot}).to_csv(
        OUTPUT / "unit_bootstrap_distribution.csv", index=False
    )
    pd.DataFrame({"mean_difference": cluster_boot}).to_csv(
        OUTPUT / "target_cluster_bootstrap_distribution.csv", index=False
    )
    (OUTPUT / "headline_metrics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
