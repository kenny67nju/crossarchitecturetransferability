"""Scoring kernels used by the camera-ready release."""
from .reff import effective_rank, effective_rank_from_singular_values
from .subspace_alignment import compute_subspace_alignment, principal_angles_from_matrices
from .ssr import ssr_dagger
