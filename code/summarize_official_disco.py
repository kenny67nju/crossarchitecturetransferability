"""Reproduce the corrected official-DiSCO regret values from local CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DISCO = RESULTS / "disco"


def load_scores(pool: str, groups: int, key: str) -> pd.DataFrame:
    filename = f"{pool}_official_pca128_g{groups}.csv"
    path = DISCO / filename
    if not path.exists():
        path = RESULTS / filename
    scores = pd.read_csv(path)
    split = scores["file"].str.removesuffix(".npz").str.split(
        "__", n=1, expand=True
    )
    scores[key] = split[0].str.replace("_dot_", ".", regex=False)
    scores["task"] = split[1]
    return scores[[key, "task", "disco"]].rename(
        columns={"disco": "official_disco"}
    )


def regret(
    ground_truth: pd.DataFrame,
    scores: pd.DataFrame,
    key: str,
    excluded_tasks: set[str] | None = None,
) -> tuple[float, int, int]:
    merged = ground_truth.merge(scores, on=[key, "task"], how="inner")
    merged = merged.dropna(subset=["lp_acc", "official_disco"])
    if excluded_tasks:
        merged = merged[~merged["task"].isin(excluded_tasks)]
    oracle = merged.groupby("task")["lp_acc"].max()
    selected = merged.loc[merged.groupby("task")["official_disco"].idxmax()]
    selected = selected.set_index("task")["lp_acc"]
    per_task = oracle - selected
    return float(per_task.mean()), len(per_task), len(merged)


def main() -> None:
    pools = [
        (
            "mv",
            pd.read_csv(DISCO / "pool_b_final_lp_complete.csv"),
            "variant",
            set(),
        ),
        (
            "vtab",
            pd.read_csv(RESULTS / "vtab_25arch_8tasks.csv"),
            "arch",
            {"stl10"},
        ),
        (
            "ex",
            pd.read_csv(DISCO / "extended_full_sota_lp_complete.csv"),
            "variant",
            set(),
        ),
    ]
    rows = []
    for pool, truth, key, excluded in pools:
        for groups in (6, 8):
            value, tasks, cells = regret(
                truth, load_scores(pool, groups, key), key, excluded
            )
            rows.append(
                {
                    "pool": pool.upper(),
                    "groups": groups,
                    "excluded_tasks": ",".join(sorted(excluded)),
                    "n_tasks": tasks,
                    "matched_lp_score_cells": cells,
                    "mean_regret": value,
                }
            )
    output = pd.DataFrame(rows)
    output.to_csv(DISCO / "official_disco_regret_summary.csv", index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
