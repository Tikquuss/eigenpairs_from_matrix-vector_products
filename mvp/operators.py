"""Matrices of Sections 2 and 5 of the post, each known through a product routine.

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


def directed_community_walk(n: int, groups: int, forward: int, other: int, rng: np.random.Generator):
    """Random walk on a random directed graph whose groups of nodes form a cycle.

    The n nodes form ``groups`` equal groups, numbered around a cycle. Each node
    has ``forward`` outgoing edges to distinct random nodes of the next group and
    ``other`` outgoing edges to distinct random nodes anywhere else. The walk
    leaves a node along one of its outgoing edges, chosen uniformly, so the
    transition matrix is P = W / (forward + other), where W_ij = 1 when the
    graph has the edge i -> j. P is not symmetric. The distribution p_k of the
    walker after k steps evolves as p_{k+1} = P^T p_k, and the product P^T x is
    one pass over the edges.

    Returns the product routine x -> P^T x, the dense P (a reference only), the
    group of every node and the edges as rows (tail, head).
    """
    group = np.repeat(np.arange(groups), n // groups)
    tails, heads = [], []
    for i in range(n):
        targets = list(rng.choice(np.flatnonzero(group == (group[i] + 1) % groups), forward, replace=False))
        targets += list(rng.choice(np.setdiff1d(np.arange(n), targets + [i]), other, replace=False))
        tails += [i] * len(targets)
        heads += targets
    tails, heads = np.array(tails), np.array(heads)
    degree = forward + other

    def apply_pt(x: np.ndarray) -> np.ndarray:
        return np.bincount(heads, weights=x[tails], minlength=n) / degree

    dense = np.zeros((n, n))
    dense[tails, heads] = 1.0 / degree
    return apply_pt, dense, group, np.column_stack([tails, heads])


def convection_diffusion(dimension: int, gamma: float):
    """The operator u -> -u'' + c u' on [0, 1] with zero boundary values,
    discretized by centered differences and scaled by h^2:
    (A x)_i = -(1 + gamma) x_{i-1} + 2 x_i - (1 - gamma) x_{i+1}, gamma = c h / 2.

    gamma = 0 gives the symmetric matrix of -u'' (Section 2.2). For
    0 < gamma < 1 the matrix is nonsymmetric with the real eigenvalues
    2 - 2 sqrt(1 - gamma^2) cos(k pi / (d + 1)), and A = S T S^{-1} with
    S = diag(s^0, ..., s^{d-1}), s = sqrt((1 + gamma) / (1 - gamma)), and T
    symmetric, so its eigenvector matrix S U has condition number s^{d-1}.
    Returns the product routine, the dense matrix, the exact eigenvalues and
    that condition number.
    """
    def apply_a(x: np.ndarray) -> np.ndarray:
        y = 2.0 * x
        y[1:] -= (1.0 + gamma) * x[:-1]
        y[:-1] -= (1.0 - gamma) * x[1:]
        return y

    dense = (np.diag(np.full(dimension, 2.0)) + np.diag(np.full(dimension - 1, -(1.0 - gamma)), 1)
             + np.diag(np.full(dimension - 1, -(1.0 + gamma)), -1))
    k = np.arange(1, dimension + 1)
    exact = 2.0 - 2.0 * np.sqrt(1.0 - gamma ** 2) * np.cos(k * np.pi / (dimension + 1))
    condition = np.sqrt((1.0 + gamma) / (1.0 - gamma)) ** (dimension - 1)
    return apply_a, dense, exact, condition
