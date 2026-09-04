"""Clustered sensitivity analysis for SSR-ddagger versus RankMe.

The sampling unit is the canonical target dataset name.  Every pool-task row
sharing a target name is resampled together, which preserves cross-pool
dependence for repeated downstream datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-boot", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df["task"] = df["task"].str.lower()
    df = df[df["task"] != "stl10"].copy()
    df = df.dropna(subset=["ssr_regret", "rankme_regret"])
    df["difference"] = df["ssr_regret"] - df["rankme_regret"]

    targets = np.array(sorted(df["task"].unique()))
    clusters = {
        target: df.loc[df["task"] == target, "difference"].to_numpy()
        for target in targets
    }
    rng = np.random.default_rng(args.seed)
    boot = np.empty(args.n_boot, dtype=np.float64)
    for index in range(args.n_boot):
        sampled_targets = rng.choice(targets, size=len(targets), replace=True)
        sampled_values = np.concatenate(
            [clusters[target] for target in sampled_targets]
        )
        boot[index] = sampled_values.mean()

    observed = float(df["difference"].mean())
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_two_sided = float(
        min(1.0, 2.0 * min(np.mean(boot >= 0), np.mean(boot <= 0)))
    )

    mapping = (
        df.groupby("task")["pool"]
        .agg(lambda values: ",".join(sorted(values)))
        .rename("pools")
        .reset_index()
    )
    mapping["n_pool_task_units"] = mapping["pools"].str.count(",") + 1

    summary = {
        "n_pool_task_units": int(len(df)),
        "n_unique_targets": int(len(targets)),
        "n_bootstrap": args.n_boot,
        "seed": args.seed,
        "mean_difference_ssr_minus_rankme": observed,
        "clustered_ci_95_low": float(ci_low),
        "clustered_ci_95_high": float(ci_high),
        "clustered_two_sided_sign_p": p_two_sided,
        "ssr_mean_regret_pool_task_weighted": float(df["ssr_regret"].mean()),
        "rankme_mean_regret_pool_task_weighted": float(
            df["rankme_regret"].mean()
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(args.output_dir / "target_cluster_mapping.csv", index=False)
    pd.DataFrame({"bootstrap_mean_difference": boot}).to_csv(
        args.output_dir / "target_cluster_bootstrap_distribution.csv",
        index=False,
    )
    (args.output_dir / "target_cluster_bootstrap_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    print(json.dumps(summary, indent=2))
    print("\nTarget clusters:")
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()
