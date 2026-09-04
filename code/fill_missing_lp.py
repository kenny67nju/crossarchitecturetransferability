"""Fill missing linear-probe cells using the paper's original LP protocol."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def feature_filename(variant: str, task: str) -> str:
    return f"{variant.replace('.', '_dot_')}__{task}.npz"


def linear_probe(features: np.ndarray, labels: np.ndarray, max_iter: int) -> float:
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    (train, test), = split.split(features, labels)
    scaler = StandardScaler().fit(features[train])
    classifier = LogisticRegression(
        max_iter=max_iter, C=1.0, n_jobs=1
    ).fit(scaler.transform(features[train]), labels[train])
    return float(classifier.score(scaler.transform(features[test]), labels[test]))


def score_cell(arguments: tuple[str, str, str, int, int]) -> dict[str, object]:
    variant, task, feature_dir, max_iter, samples = arguments
    path = Path(feature_dir) / feature_filename(variant, task)
    with np.load(path, allow_pickle=False) as data:
        features = data["H"] if "H" in data else data["features"]
        labels = data["y"] if "y" in data else data["labels"]
    if samples and samples < len(labels):
        indices = np.random.default_rng(42).choice(
            len(labels), samples, replace=False
        )
        features, labels = features[indices], labels[indices]
    accuracy = linear_probe(features, labels, max_iter)
    return {
        "variant": variant,
        "task": task,
        "lp_acc": accuracy,
        "n_samples": len(labels),
        "feature_dim": features.shape[1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("existing_results", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("missing_output", type=Path)
    parser.add_argument("completed_output", type=Path)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    parser.add_argument("--samples", type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    existing = pd.read_csv(args.existing_results)
    keys = ["variant", "task"]
    missing = (
        manifest.merge(existing[keys], on=keys, how="left", indicator=True)
        .query("_merge == 'left_only'")[keys]
        .reset_index(drop=True)
    )
    print(f"missing cells: {len(missing)}", flush=True)

    work = [
        (row.variant, row.task, str(args.feature_dir), args.max_iter, args.samples)
        for row in missing.itertuples(index=False)
    ]
    if args.parallel_jobs == 1:
        results = map(score_cell, work)
    else:
        executor = ProcessPoolExecutor(max_workers=args.parallel_jobs)
        results = executor.map(score_cell, work)

    rows: list[dict[str, object]] = []
    for index, result in enumerate(results):
        rows.append(result)
        print(
            f"[{index + 1}/{len(missing)}] {result['variant']}/{result['task']}: "
            f"{result['lp_acc']:.6f}",
            flush=True,
        )
    if args.parallel_jobs != 1:
        executor.shutdown()

    added = pd.DataFrame(rows)
    args.missing_output.parent.mkdir(parents=True, exist_ok=True)
    added.to_csv(args.missing_output, index=False)

    completed = manifest.merge(
        pd.concat([existing, added], ignore_index=True),
        on=keys,
        how="left",
        suffixes=("", "_result"),
    )
    # Preserve the complete original schema while using manifest order.
    result_columns = list(existing.columns)
    normalized = pd.DataFrame({column: completed.get(column) for column in result_columns})
    for column in result_columns:
        result_column = f"{column}_result"
        if result_column in completed:
            normalized[column] = normalized[column].combine_first(completed[result_column])
    normalized.to_csv(args.completed_output, index=False)
    print(f"wrote {args.missing_output} and {args.completed_output}", flush=True)


if __name__ == "__main__":
    main()
