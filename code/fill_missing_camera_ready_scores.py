#!/usr/bin/env python3
"""Fill missing RankMe/SA/SSR cells on an existing complete LP support table.

The target and STL-10 matrices are scored with the camera-ready convention:
uncentered SVD, RankMe singular-value normalization, k=20, and
SSR=(log(r_eff)-log(SA_k))/log(C).  Covariances are accumulated on an NPU so
large full-split EX matrices need not undergo a prohibitively expensive tall
CPU SVD; the resulting d-by-d symmetric eigendecomposition is exact up to the
usual floating-point arithmetic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_npu  # noqa: F401


EPS = 1e-12


def feature_path(directory: Path, variant: str, task: str) -> Path:
    stem = variant.replace(".", "_dot_")
    return directory / f"{stem}__{task}.npz"


def load_features(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as data:
        for key in ("H", "features"):
            if key in data:
                return np.asarray(data[key], dtype=np.float32)
    raise KeyError(f"no feature matrix in {path}")


def spectrum_and_basis(features: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    matrix = torch.from_numpy(features).to(device)
    gram = (matrix.T @ matrix).cpu().numpy().astype(np.float64, copy=False)
    del matrix
    torch.npu.empty_cache()
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    singular_values = np.sqrt(np.maximum(eigenvalues[order], 0.0))
    right_basis = eigenvectors[:, order].T
    return singular_values, right_basis


def effective_rank(singular_values: np.ndarray) -> float:
    sigma = singular_values[singular_values > EPS]
    probabilities = sigma / sigma.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def alignment(target_basis: np.ndarray, reference_basis: np.ndarray, k: int) -> float:
    k_used = min(k, target_basis.shape[0], reference_basis.shape[0])
    cosines = np.linalg.svd(
        target_basis[:k_used] @ reference_basis[:k_used].T,
        compute_uv=False,
    )
    return float(np.mean(np.clip(cosines, 0.0, 1.0) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--target-features", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--npu", type=int, default=5)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    device = f"npu:{args.npu}"
    torch.npu.set_device(device)
    frame = pd.read_csv(args.input_csv)
    valid = frame.dropna(subset=["reff", "rankme"])
    raw_error = np.median(np.abs(valid["rankme"] - valid["reff"]))
    log_error = np.median(np.abs(valid["rankme"] - np.log(valid["reff"])))
    if log_error < raw_error:
        frame.loc[frame["rankme"].notna(), "rankme"] = np.exp(
            frame.loc[frame["rankme"].notna(), "rankme"]
        )
        print("normalized historical log-RankMe column to raw effective rank", flush=True)
    missing = frame[frame["rankme"].isna() | frame["ssr_dagger"].isna()]
    print(f"device={device} missing={len(missing)}", flush=True)

    reference_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for position, (index, row) in enumerate(missing.iterrows(), 1):
        variant, task = str(row["variant"]), str(row["task"])
        print(f"[{position}/{len(missing)}] {variant}/{task}", flush=True)
        if variant not in reference_cache:
            reference_cache[variant] = spectrum_and_basis(
                load_features(feature_path(args.reference_features, variant, "stl10")),
                device,
            )
        _, reference_basis = reference_cache[variant]
        target_features = load_features(feature_path(args.target_features, variant, task))
        singular_values, target_basis = spectrum_and_basis(target_features, device)
        reff = effective_rank(singular_values)
        sa_k = max(alignment(target_basis, reference_basis, args.k), EPS)
        rankme = reff
        n_class = row["n_class"]
        if pd.isna(n_class):
            candidates = frame.loc[
                (frame["task"] == task) & frame["n_class"].notna(), "n_class"
            ]
            if not len(candidates):
                raise ValueError(f"cannot infer class count for {task}")
            n_class = candidates.iloc[0]
            frame.loc[index, "n_class"] = n_class
        ssr = float((np.log(rankme) - np.log(sa_k)) / np.log(float(n_class)))

        frame.loc[index, "reff"] = reff
        frame.loc[index, "rankme"] = rankme
        frame.loc[index, "sa_k"] = sa_k
        frame.loc[index, "ssr_dagger"] = ssr
        if "feat_dim" in frame.columns:
            frame.loc[index, "feat_dim"] = target_features.shape[1]
        if pd.isna(row.get("arch", np.nan)):
            candidates = frame.loc[
                (frame["variant"] == variant) & frame["arch"].notna(), "arch"
            ]
            frame.loc[index, "arch"] = (
                candidates.iloc[0] if len(candidates) else variant.split(".")[0]
            )
        del target_features, singular_values, target_basis

    # Existing rows store the correct RankMe effective rank in ``rankme`` but
    # may retain the superseded squared-spectrum value in ``reff``/SSR.
    frame["reff"] = frame["rankme"]
    frame["ssr_dagger"] = (
        np.log(np.maximum(frame["rankme"], EPS))
        - np.log(np.maximum(frame["sa_k"], EPS))
    ) / np.log(frame["n_class"])
    remaining = int(frame["rankme"].isna().sum() + frame["ssr_dagger"].isna().sum())
    if remaining:
        raise RuntimeError(f"score completion left {remaining} missing values")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    print(f"saved {len(frame)} rows -> {args.output_csv}", flush=True)


if __name__ == "__main__":
    main()
