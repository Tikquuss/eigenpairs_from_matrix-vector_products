"""Symmetric matrices of Section 2 of the post, each known through a product routine.

Every constructor returns an ``apply(x)`` routine that never forms the matrix,
and a dense copy used only as a reference for measuring errors.
"""

from typing import Callable, Tuple

import numpy as np

MatVec = Callable[[np.ndarray], np.ndarray]


class Counter:
    """Wrap a product routine and count its calls."""

    def __init__(self, apply_a: MatVec):
        self.apply_a = apply_a
        self.calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        return self.apply_a(x)


def random_orthogonal(dimension: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-distributed orthogonal matrix."""
    q, r = np.linalg.qr(rng.standard_normal((dimension, dimension)))
    return q * np.sign(np.diag(r))


def prescribed_spectrum(eigenvalues: np.ndarray, rng: np.random.Generator):
    """A = U diag(eigenvalues) U^T with random orthonormal eigenvectors U."""
    u = random_orthogonal(len(eigenvalues), rng)
    a = (u * eigenvalues) @ u.T
    return 0.5 * (a + a.T), u


def bipartite_graph(left: Tuple[int, int], right: Tuple[int, int], p_within: float,
                    p_across: float, rng: np.random.Generator):
    """Adjacency matrix of a random bipartite graph with two planted communities.

    The ``left`` nodes form two groups of the given sizes, and so do the
    ``right`` nodes. An edge joins a left node and a right node with probability
    ``p_within`` when their groups match and ``p_across`` otherwise; there are
    no edges inside a side. Nodes are ordered left first. The product is one
    pass over the edge list, (W x)_i = sum over the edges {i, j} of x_j.
    """
    n_left, n_right = sum(left), sum(right)
    group_left = np.repeat([0, 1], left)
    group_right = np.repeat([0, 1], right)
    probability = np.where(group_left[:, None] == group_right[None, :], p_within, p_across)
    rows, cols = np.nonzero(rng.random((n_left, n_right)) < probability)
    heads, tails = rows, n_left + cols
    dimension = n_left + n_right

    def apply_w(x: np.ndarray) -> np.ndarray:
        return (np.bincount(heads, weights=x[tails], minlength=dimension)
                + np.bincount(tails, weights=x[heads], minlength=dimension))

    dense = np.zeros((dimension, dimension))
    dense[heads, tails] = dense[tails, heads] = 1.0
    return apply_w, dense, np.column_stack([heads, tails])


def spiked_covariance(n: int, spikes: np.ndarray, dimension: int, rng: np.random.Generator):
    """Sample covariance (1/n) X^T X of n centered Gaussian observations.

    The population covariance has the eigenvalues ``spikes`` along random
    orthonormal directions and 1 along the others. The product
    X^T (X v) / n costs O(nd) operations and never forms X^T X.
    """
    directions = random_orthogonal(dimension, rng)[:, : len(spikes)]
    scales = np.sqrt(np.asarray(spikes, dtype=float) - 1.0)
    x = rng.standard_normal((n, dimension)) + (rng.standard_normal((n, len(spikes))) * scales) @ directions.T
    x -= x.mean(axis=0)

    def apply_c(v: np.ndarray) -> np.ndarray:
        return x.T @ (x @ v) / n

    return apply_c, x.T @ x / n, x
