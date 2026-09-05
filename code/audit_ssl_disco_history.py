"""Audit every retained SSL DiSCO result against one LP support.

This distinguishes the superseded proxy score from the official full-feature
and official PCA-128 variants and makes historical transcription errors
detectable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from summarize_official_disco import load_ssl_scores, regret


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    truth = pd.read_csv(RESULTS / "paper14_crossdomain.csv")
    task_oracle = truth.groupby("task")["lp_acc"].max()
    rows: list[dict[str, object]] = []

    legacy = truth[["model", "task", "disco"]].rename(
        columns={"disco": "official_disco"}
    )
    variants: list[tuple[str, pd.DataFrame]] = [("legacy_proxy", legacy)]
    for groups in (6, 8):
        for prefix in ("ssl_official", "ssl_official_pca128"):
            source = RESULTS / f"{prefix}_g{groups}.csv"
            scores = load_ssl_scores(groups, source)
            variants.append((f"{prefix}_g{groups}", scores))

    summaries = []
    for name, scores in variants:
        merged = truth.merge(scores, on=["model", "task"], how="inner")
        selected = merged.loc[
            merged.groupby("task")["official_disco"].idxmax()
        ].set_index("task")
        per_task = task_oracle - selected["lp_acc"]
        summaries.append(
            {
                "variant": name,
                "mean_regret": float(per_task.mean()),
                "n_tasks": int(len(per_task)),
                "n_cells": int(len(merged)),
            }
        )
        for task, value in per_task.items():
            rows.append(
                {
                    "variant": name,
                    "task": task,
                    "regret": float(value),
                    "selected_model": selected.loc[task, "model"],
                }
            )

    output = RESULTS / "camera_ready"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "ssl_disco_history_per_task.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "ssl_disco_history_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
