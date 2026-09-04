"""Official classification DiSCO score (Zhang et al., AAAI 2025).

The implementation follows Eqs. (5)--(9) of
"Assessing Pre-Trained Models for Transfer Learning Through Distribution of
Spectral Components":

1. apply the paper's experimental PCA preprocessing to 128 dimensions;
2. compute the SVD of the PCA target feature matrix;
3. split the descending spectrum into ``n_groups`` consecutive groups;
4. measure each group's target-label performance with the probabilistic
   nearest-centroid classifier defined by class-specific Mahalanobis distance;
5. weight that confidence by the group's share of the singular-value sum.

DiSCO uses target ground-truth labels and is therefore label-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DiscoDetails:
    """Auditable intermediate values returned by :func:`disco_classification`."""

    score: float
    group_scores: np.ndarray
    spectral_ratios: np.ndarray
    group_bounds: tuple[tuple[int, int], ...]


def _validate_inputs(features: np.ndarray, labels: np.ndarray, n_groups: int) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features)
    labels = np.asarray(labels).reshape(-1)
    if features.ndim != 2:
        raise ValueError(f"features must be a 2-D matrix, got shape {features.shape}")
    if len(features) != len(labels):
        raise ValueError(f"features and labels differ in length: {len(features)} != {len(labels)}")
    if len(features) == 0:
        raise ValueError("features must not be empty")
    if not np.isfinite(features).all():
        raise ValueError("features contain NaN or infinity")
    if n_groups < 1:
        raise ValueError(f"n_groups must be positive, got {n_groups}")
    if np.unique(labels).size < 2:
        raise ValueError("DiSCO classification requires at least two classes")
    return features.astype(np.float64, copy=False), labels


def _group_bounds(rank: int, n_groups: int) -> tuple[tuple[int, int], ...]:
    """Split all singular directions into consecutive, near-equal groups."""
    groups = min(rank, n_groups)
    indices = np.array_split(np.arange(rank), groups)
    return tuple((int(index[0]), int(index[-1]) + 1) for index in indices if len(index))


def _ncc_confidence(component_coordinates: np.ndarray, labels: np.ndarray, rcond: float) -> float:
    """Eq. (8): mean ground-truth posterior under class Gaussian models.

    ``component_coordinates`` equals ``U_g Sigma_g``.  It is isometric to the
    paper's reconstruction ``U_g Sigma_g V_g^T``, so Mahalanobis distances and
    hence the score are identical while avoiding a large dense reconstruction.
    """
    classes, counts = np.unique(labels, return_counts=True)
    n_samples, dimension = component_coordinates.shape
    logits = np.empty((n_samples, len(classes)), dtype=np.float64)

    for column, (class_id, count) in enumerate(zip(classes, counts)):
        class_features = component_coordinates[labels == class_id]
        mean = class_features.mean(axis=0)
        centered = class_features - mean
        delta = component_coordinates - mean
        # If A is the centered class matrix, pinv(A.T A/(n-1)) can be
        # evaluated as (n-1) pinv(A) pinv(A).T.  Working through A's thin
        # pseudoinverse is exactly equivalent and much faster when a class has
        # fewer samples than the spectral-group dimension.
        centered_pinv = np.linalg.pinv(centered, rcond=rcond)
        whitened_delta = delta @ centered_pinv
        squared_mahalanobis = max(int(count) - 1, 1) * np.square(whitened_delta).sum(axis=1)
        logits[:, column] = -0.5 * squared_mahalanobis + np.log(count / n_samples)

    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    class_to_column = {class_id: column for column, class_id in enumerate(classes)}
    target_columns = np.fromiter((class_to_column[value] for value in labels), dtype=np.int64)
    return float(probabilities[np.arange(n_samples), target_columns].mean())


def disco_classification(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_groups: int = 8,
    pca_components: int = 128,
    rcond: float = 1e-10,
    return_details: bool = False,
) -> float | DiscoDetails:
    """Compute the official label-dependent DiSCO classification score.

    Args:
        features: Target feature matrix ``Z`` with shape ``(N, d)``.
        labels: Target ground-truth labels with shape ``(N,)``.
        n_groups: Number ``G`` of consecutive spectral groups.  The paper
            recommends 6 or 8; 8 is the default here.
        pca_components: Uniform feature dimension used in the paper's
            experiments. Set to 0 only for a formula-level ablation.
        rcond: Relative cutoff used for the Moore--Penrose covariance inverse.
        return_details: Return group-level values for auditing when true.
    """
    features, labels = _validate_inputs(features, labels, n_groups)
    if pca_components < 0:
        raise ValueError(f"pca_components must be non-negative, got {pca_components}")
    if pca_components and features.shape[1] > pca_components:
        from sklearn.decomposition import PCA

        features = PCA(
            n_components=min(pca_components, features.shape[0], features.shape[1]),
            svd_solver="randomized",
            random_state=0,
        ).fit_transform(features)
    left_vectors, singular_values, _ = np.linalg.svd(features, full_matrices=False)
    positive_rank = int(np.count_nonzero(singular_values > np.finfo(np.float64).eps * singular_values[0]))
    if positive_rank == 0:
        raise ValueError("feature matrix has zero numerical rank")

    left_vectors = left_vectors[:, :positive_rank]
    singular_values = singular_values[:positive_rank]
    bounds = _group_bounds(positive_rank, n_groups)
    denominator = float(singular_values.sum())

    group_scores = []
    spectral_ratios = []
    for start, stop in bounds:
        coordinates = left_vectors[:, start:stop] * singular_values[start:stop]
        group_scores.append(_ncc_confidence(coordinates, labels, rcond))
        spectral_ratios.append(float(singular_values[start:stop].sum() / denominator))

    group_scores_array = np.asarray(group_scores)
    spectral_ratios_array = np.asarray(spectral_ratios)
    score = float(group_scores_array @ spectral_ratios_array)
    if return_details:
        return DiscoDetails(score, group_scores_array, spectral_ratios_array, bounds)
    return score
