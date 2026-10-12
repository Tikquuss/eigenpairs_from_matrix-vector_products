"""Matrix-free eigensolvers from the blog post
"Eigenpairs from Matrix-Vector Products: Power Iteration, Deflation, and Lanczos"
(https://tikquuss.github.io/blog/eigenpairs-from-matrix-vector-products/).

Every solver touches the matrix only through a function ``apply_a(x)`` that
returns ``A @ x`` for a real symmetric matrix ``A``.
"""

from .lanczos import (
    certify_top_k,
    complete_decomposition,
    explicit_residuals,
    lanczos,
    ritz_pairs,
    tridiagonal,
)
from .operators import Counter, bipartite_graph, prescribed_spectrum, random_orthogonal, spiked_covariance
from .power import (
    Stage,
    algorithm1_stage,
    deflated_eigenpairs,
    deflated_power_stage,
    power_iteration,
    project_out,
)

__all__ = [
    "Counter",
    "Stage",
    "algorithm1_stage",
    "bipartite_graph",
    "certify_top_k",
    "complete_decomposition",
    "deflated_eigenpairs",
    "deflated_power_stage",
    "explicit_residuals",
    "lanczos",
    "power_iteration",
    "prescribed_spectrum",
    "project_out",
    "random_orthogonal",
    "ritz_pairs",
    "spiked_covariance",
    "tridiagonal",
]
