"""Recompute VTAB RankMe and target-disjoint reference ensembles from features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_SETS = {
    "stl10_only": ["stl10"],
    "four_reference_loo": ["stl10", "cifar10", "flowers102", "pets"],
    "five_reference_loo": [
        "stl10",
        "cifar10",
        "flowers102",
        "pets",
        "fashion_mnist",
    ],
}


def load_features(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    key = "features" if "features" in data else "H"
    return data[key].astype(np.float64)


def effective_rank(singular_values: np.ndarray) -> float:
    p = singular_values / singular_values.sum()
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("lp_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center each feature matrix before SVD.",
    )
    args = parser.parse_args()

    lp = pd.read_csv(args.lp_csv)
    keys = list(zip(lp["arch"], lp["task"]))
    cache: dict[tuple[str, str], tuple[np.ndarray, float]] = {}
    for arch, task in keys:
        path = args.feature_dir / f"{arch}__{task}.npz"
        if not path.exists():
            continue
        features = load_features(path)
        if args.center:
            features = features - features.mean(axis=0, keepdims=True)
        _, singular_values, vt = np.linalg.svd(features, full_matrices=False)
        cache[(arch, task)] = (
            vt[: args.k].copy(),
            effective_rank(singular_values),
        )

    rows = []
    for record in lp.itertuples(index=False):
        key = (record.arch, record.task)
        if key not in cache or record.task == "stl10":
            continue
        target_vt, reff = cache[key]
        row = {
            "arch": record.arch,
            "task": record.task,
            "lp_acc": record.lp_acc,
            "n_class": record.n_class,
            "reff_rankme": reff,
            "rankme_score": np.log(reff),
        }
        for name, references in REFERENCE_SETS.items():
            alignments = []
            references_used = []
            for reference in references:
                if reference == record.task:
                    continue
                reference_key = (record.arch, reference)
                if reference_key not in cache:
                    continue
                reference_vt, _ = cache[reference_key]
                k_eff = min(len(target_vt), len(reference_vt), args.k)
                alignment = float(
                    np.square(
                        target_vt[:k_eff] @ reference_vt[:k_eff].T
                    ).sum()
                    / k_eff
                )
                alignments.append(alignment)
                references_used.append(reference)
            if not alignments:
                continue
            median_alignment = float(np.median(alignments))
            row[f"{name}_n_refs"] = len(alignments)
            row[f"{name}_references"] = ",".join(references_used)
            row[f"{name}_sa"] = median_alignment
            row[f"{name}_ssr"] = (
                np.log(reff) - np.log(max(median_alignment, 1e-12))
            ) / np.log(max(float(record.n_class), 2.0))
        rows.append(row)

    output = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for score_name in ["rankme_score"] + [
        f"{name}_ssr" for name in REFERENCE_SETS
    ]:
        per_task = []
        for task, group in output.groupby("task"):
            valid = group.dropna(subset=["lp_acc", score_name])
            selected = valid.loc[valid[score_name].idxmax(), "lp_acc"]
            per_task.append(
                {
                    "task": task,
                    "oracle": float(valid["lp_acc"].max()),
                    "selected_lp": float(selected),
                    "regret": float(valid["lp_acc"].max() - selected),
                }
            )
        per_task_frame = pd.DataFrame(per_task)
        per_task_frame.to_csv(
            args.output_dir / f"{score_name}_per_task.csv", index=False
        )
        summaries.append(
            {
                "score": score_name,
                "mean_regret": float(per_task_frame["regret"].mean()),
                "n_tasks": len(per_task_frame),
            }
        )

    output.to_csv(args.output_dir / "vtab_reference_audit_cells.csv", index=False)
    summary = {
        "centered": args.center,
        "k": args.k,
        "n_cached_svd": len(cache),
        "n_scored_cells": len(output),
        "scores": summaries,
    }
    (args.output_dir / "vtab_reference_audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
