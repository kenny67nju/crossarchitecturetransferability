#!/usr/bin/env python3
"""Score cached target features with the official classification DiSCO metric."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Keep process-level parallelism predictable on large multicore servers.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

from metrics.disco import disco_classification


def _score_file(job: tuple[str, int, int, float]) -> dict[str, object]:
    filename, n_groups, pca_components, rcond = job
    path = Path(filename)
    started = time.perf_counter()
    with np.load(path, allow_pickle=True) as cache:
        feature_key = "H" if "H" in cache else "features" if "features" in cache else None
        label_key = "y" if "y" in cache else "labels" if "labels" in cache else None
        if feature_key is None or label_key is None:
            raise KeyError(f"{path} must contain H/y or features/labels arrays")
        features = cache[feature_key]
        labels = cache[label_key]
    details = disco_classification(
        features,
        labels,
        n_groups=n_groups,
        pca_components=pca_components,
        rcond=rcond,
        return_details=True,
    )
    return {
        "file": path.name,
        "n_samples": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "n_classes": int(np.unique(labels).size),
        "n_groups": n_groups,
        "pca_components": pca_components,
        "disco": details.score,
        "group_scores": json.dumps(details.group_scores.tolist()),
        "spectral_ratios": json.dumps(details.spectral_ratios.tolist()),
        "group_bounds": json.dumps(details.group_bounds),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("feature_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--groups", type=int, default=8, choices=(6, 8))
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--rcond", type=float, default=1e-10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    files = sorted(path for path in args.feature_dir.glob("*.npz") if not path.name.startswith("._"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .npz files found in {args.feature_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, object]] = []
    jobs = [(str(path), args.groups, args.pca_components, args.rcond) for path in files]
    started = time.perf_counter()

    if args.workers == 1:
        iterator = map(_score_file, jobs)
        for index, result in enumerate(iterator, 1):
            completed.append(result)
            print(f"[{index}/{len(jobs)}] {result['file']}: {result['disco']:.8f}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_score_file, job): job[0] for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                completed.append(result)
                print(f"[{index}/{len(jobs)}] {result['file']}: {result['disco']:.8f}", flush=True)

    completed.sort(key=lambda row: str(row["file"]))
    temporary = args.output_csv.with_suffix(args.output_csv.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)
    temporary.replace(args.output_csv)
    elapsed = time.perf_counter() - started
    print(f"Wrote {len(completed)} scores to {args.output_csv} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
