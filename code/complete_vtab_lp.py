"""Complete missing VTAB linear probes from cached features.

The output retains only architectures with cached features for every requested
task, so every per-task comparison uses the same candidate pool.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def load_cell(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        features = data["H"] if "H" in data else data["features"]
        labels = data["y"] if "y" in data else data["labels"]
    return features, labels.astype(int)


def linear_probe(features: np.ndarray, labels: np.ndarray) -> float:
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=0
    )
    (train, test), = splitter.split(features, labels)
    scaler = StandardScaler().fit(features[train])
    classifier = LogisticRegression(
        C=1.0, max_iter=1000, n_jobs=1
    ).fit(scaler.transform(features[train]), labels[train])
    return float(classifier.score(scaler.transform(features[test]), labels[test]))


def score(argument: tuple[str, str, str]) -> tuple[str, str, float]:
    arch, task, feature_dir = argument
    features, labels = load_cell(Path(feature_dir) / f"{arch}__{task}.npz")
    return arch, task, linear_probe(features, labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--parallel-jobs", type=int, default=1)
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    # Historical placeholder rows used n_class=0.  Infer task metadata from
    # the valid rows before carrying those placeholders into the complete table.
    class_counts = {
        task: int(group.loc[group["n_class"] > 1, "n_class"].mode().iloc[0])
        for task, group in table.groupby("task")
    }
    table["n_class"] = [
        class_counts[task] if value <= 1 else int(value)
        for task, value in zip(table["task"], table["n_class"])
    ]
    tasks = sorted(table["task"].unique())
    feature_sets = {
        arch: {
            task
            for task in tasks
            if (args.feature_dir / f"{arch}__{task}.npz").exists()
        }
        for arch in sorted(table["arch"].unique())
    }
    complete_arches = sorted(
        arch for arch, available in feature_sets.items() if available == set(tasks)
    )
    output = table[table["arch"].isin(complete_arches)].copy()
    missing = output[output["lp_acc"].isna()][["arch", "task"]]
    work = [
        (row.arch, row.task, str(args.feature_dir))
        for row in missing.itertuples(index=False)
    ]

    if args.parallel_jobs == 1:
        results = map(score, work)
    else:
        executor = ProcessPoolExecutor(max_workers=args.parallel_jobs)
        results = executor.map(score, work)

    for index, (arch, task, accuracy) in enumerate(results, start=1):
        mask = (output["arch"] == arch) & (output["task"] == task)
        output.loc[mask, "lp_acc"] = accuracy
        print(f"[{index}/{len(work)}] {arch}/{task}: {accuracy:.6f}", flush=True)

    if args.parallel_jobs != 1:
        executor.shutdown()

    if output["lp_acc"].isna().any():
        raise RuntimeError("completed common-pool table still contains missing LP cells")
    expected = len(complete_arches) * len(tasks)
    if len(output) != expected:
        raise RuntimeError(f"expected {expected} rows, found {len(output)}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    print(
        f"wrote {args.output_csv}: {len(complete_arches)} architectures x "
        f"{len(tasks)} tasks = {len(output)} cells"
    )


if __name__ == "__main__":
    main()
