"""Reproduce the corrected official-DiSCO regret values from local CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from final_selection import multi


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DISCO = RESULTS / "disco"
SSL_MODELS = (
    "clip_vit_l14",
    "convnext_base",
    "dinov2_vit_b14",
    "dino_vit_s16",
    "mocov3_resnet50",
    "resnet101",
    "resnet50",
    "vit_b16",
)


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


def load_ssl_scores(groups: int, path: Path | None = None) -> pd.DataFrame:
    """Parse ``<task>_<model>_N1000.npz`` SSL score filenames."""
    if path is None:
        path = RESULTS / f"ssl_official_pca128_g{groups}.csv"
    scores = pd.read_csv(path)
    rows = []
    for row in scores.itertuples(index=False):
        stem = row.file.removesuffix("_N1000.npz")
        for model in SSL_MODELS:
            suffix = f"_{model}"
            if stem.endswith(suffix):
                rows.append(
                    {
                        "model": model,
                        "task": stem[: -len(suffix)],
                        "official_disco": row.disco,
                    }
                )
                break
    return pd.DataFrame(rows)


def regret(
    ground_truth: pd.DataFrame,
    scores: pd.DataFrame,
    key: str,
    excluded_tasks: set[str] | None = None,
) -> tuple[float, int, int]:
    merged = ground_truth.merge(scores, on=[key, "task"], how="inner", validate="one_to_one")
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
            multi(pd.read_csv(RESULTS / "p0_vgg_pair_20261104" / "p0_aggregate_v2" /
                              "single_preprocess_4096_pool132_scores.csv")),
            "variant",
            set(),
        ),
        (
            "vtab",
            pd.read_csv(
                RESULTS / "camera_ready" / "vtab_24arch_8tasks_lp_complete.csv"
            ),
            "arch",
            {"stl10"},
        ),
        (
            "ex",
            pd.read_csv(RESULTS / "camera_ready" / "extended_full_sota_score_complete.csv"),
            "variant",
            set(),
        ),
        (
            "ssl",
            pd.read_csv(RESULTS / "paper14_crossdomain.csv"),
            "model",
            set(),
        ),
    ]
    rows = []
    for pool, truth, key, excluded in pools:
        for groups in (6, 8):
            if pool == "mv":
                scores = truth[[key, "task", f"disco_official_g{groups}"]].rename(
                    columns={f"disco_official_g{groups}": "official_disco"})
            else:
                scores = load_ssl_scores(groups) if pool == "ssl" else load_scores(pool, groups, key)
            value, tasks, cells = regret(
                truth, scores, key, excluded
            )
            expected = {"mv": (5, 595), "vtab": (7, 168), "ex": (10, 510), "ssl": (29, 209)}
            assert (tasks, cells) == expected[pool], (pool, groups, tasks, cells)
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
    # Appendix diagnostic, separate from the PCA-128 baseline.
    ssl_truth = pools[-1][1]
    for groups in (6, 8):
        scores = load_ssl_scores(groups, RESULTS / f"ssl_official_g{groups}.csv")
        value, tasks, cells = regret(ssl_truth, scores, "model")
        assert (tasks, cells) == (29, 209)
        print(f"SSL full-feature appendix diagnostic G={groups}: {value:.6f} ({tasks} tasks, {cells} cells)")


if __name__ == "__main__":
    main()
