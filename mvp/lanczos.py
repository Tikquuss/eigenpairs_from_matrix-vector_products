"""The Lanczos method with full reorthogonalization (Section 4 of the post)."""

from typing import Callable, List

import numpy as np

MatVec = Callable[[np.ndarray], np.ndarray]


def lanczos(apply_a: MatVec, q1: np.ndarray, steps: int, reorthogonalize: bool = True):
    """Run up to ``steps`` Lanczos steps (Definition 2), stopping early at an
    exact breakdown. Returns (alphas, betas, basis) with ``basis`` holding
    q_1, ..., q_m as columns; ``betas[m-1]`` is the norm of the last residual.
    """
    dimension = q1.size
    basis = np.zeros((dimension, steps))
    alphas, betas = np.zeros(steps), np.zeros(steps)
    q_previous, beta_previous = np.zeros(dimension), 0.0
    q = q1 / np.linalg.norm(q1)
    for j in range(steps):
        basis[:, j] = q
        z = apply_a(q) - beta_previous * q_previous
        alphas[j] = q @ z
        r = z - alphas[j] * q
        if reorthogonalize:
            for _ in range(2):
                r -= basis[:, : j + 1] @ (basis[:, : j + 1].T @ r)
        betas[j] = np.linalg.norm(r)
        if betas[j] == 0.0:
            return alphas[: j + 1], betas[: j + 1], basis[:, : j + 1]
        q_previous, q, beta_previous = q, r / betas[j], betas[j]
    return alphas, betas, basis


def tridiagonal(alphas: np.ndarray, betas: np.ndarray, m: int) -> np.ndarray:
    """T_m, with diagonal alpha_1..alpha_m and off-diagonal beta_1..beta_{m-1}."""
    return np.diag(alphas[:m]) + np.diag(betas[: m - 1], 1) + np.diag(betas[: m - 1], -1)


def ritz_pairs(alphas: np.ndarray, betas: np.ndarray, m: int):
    """Eigenpairs (theta, y) of the tridiagonal T_m (Lemma 4), ascending."""
    return np.linalg.eigh(tridiagonal(alphas, betas, m))


def _passes(beta: float, y_last: np.ndarray, theta: np.ndarray, eps_abs: float, eps_rel: float):
    """Inexpensive residual test of Algorithm 2: beta_m |(y_i)_m| (Lemma 4)."""
    return beta * np.abs(y_last) <= eps_abs + eps_rel * np.abs(theta)


def explicit_residuals(apply_a: MatVec, vectors: np.ndarray):
    """Step 5 of Algorithm 2: one product per candidate. Returns the Rayleigh
    quotients and the residual norms ||A x - (x^T A x) x|| of the normalized
    columns of ``vectors``."""
    values, residuals = [], []
    for x in vectors.T:
        x = x / np.linalg.norm(x)
        ax = apply_a(x)
        values.append(x @ ax)
        residuals.append(np.linalg.norm(ax - values[-1] * x))
    return np.array(values), np.array(residuals)


def certify_top_k(apply_a: MatVec, q1: np.ndarray, k: int, max_steps: int,
                  eps_abs: float = 1e-10, eps_rel: float = 1e-10):
    """Algorithm 2 for the k Ritz pairs of largest absolute value.

    Extends the basis one product at a time and returns ``first_step``, where
    ``first_step[i]`` is the first step m at which the i+1 Ritz pairs of largest
    absolute value all pass the inexpensive test (``-1`` if never), together
    with the k candidates at the last step: Rayleigh quotients, explicit
    residuals (k more products) and vectors.
    """
    alphas, betas, basis = lanczos(apply_a, q1, max_steps)
    first_step = np.full(k, -1)
    m = alphas.size
    for step in range(1, alphas.size + 1):
        theta, y = ritz_pairs(alphas, betas, step)
        order = np.argsort(-np.abs(theta))
        passed = _passes(betas[step - 1], y[-1, order], theta[order], eps_abs, eps_rel)
        for count in range(1, min(k, step) + 1):
            if first_step[count - 1] < 0 and passed[:count].all():
                first_step[count - 1] = step
        if (first_step >= 0).all():
            m = step
            break
    theta, y = ritz_pairs(alphas, betas, m)
    order = np.argsort(-np.abs(theta))[:k]
    vectors = basis[:, :m] @ y[:, order]
    values, residuals = explicit_residuals(apply_a, vectors)
    return first_step, values, residuals, vectors


def complete_decomposition(apply_a: MatVec, dimension: int, rng: np.random.Generator,
                           breakdown: float = 1e-10, eps_abs: float = 1e-10, eps_rel: float = 1e-10):
    """Proposition 4: Lanczos runs continued with new Gaussian starting vectors,
    orthogonal to every basis vector built so far, until the basis is complete.

    A run stops when beta_m <= ``breakdown`` * (largest |alpha| or beta seen),
    a numerical breakdown. Returns the number of certified Ritz pairs after
    each product (``certified[p-1]`` after p products), the Ritz values of the
    final block-diagonal projection, and the run lengths.
    """
    basis = np.zeros((dimension, 0))
    blocks: List[np.ndarray] = []
    certified: List[int] = []
    run_lengths: List[int] = []
    scale = 0.0
    while basis.shape[1] < dimension:
        g = rng.standard_normal(dimension)
        for _ in range(2):
            g -= basis @ (basis.T @ g)
        q = g / np.linalg.norm(g)
        alphas, betas = [], []
        q_previous, beta_previous = np.zeros(dimension), 0.0
        done_before = sum(block.shape[0] for block in blocks)
        while True:
            basis = np.column_stack([basis, q])
            z = apply_a(q) - beta_previous * q_previous
            alphas.append(q @ z)
            r = z - alphas[-1] * q
            for _ in range(2):
                r -= basis @ (basis.T @ r)
            betas.append(np.linalg.norm(r))
            scale = max(scale, abs(alphas[-1]), betas[-1])
            stop = betas[-1] <= breakdown * scale or basis.shape[1] == dimension
            m = len(alphas)
            theta, y = np.linalg.eigh(tridiagonal(np.array(alphas), np.array(betas), m))
            beta = 0.0 if stop else betas[-1]
            certified.append(done_before + int(_passes(beta, y[-1], theta, eps_abs, eps_rel).sum()))
            if stop:
                break
            q_previous, q, beta_previous = q, r / betas[-1], betas[-1]
        blocks.append(tridiagonal(np.array(alphas), np.array(betas), len(alphas)))
        run_lengths.append(len(alphas))
    t = np.zeros((dimension, dimension))
    start = 0
    for block in blocks:
        size = block.shape[0]
        t[start:start + size, start:start + size] = block
        start += size
    theta, y = np.linalg.eigh(t)
    return np.array(certified), theta, basis @ y, run_lengths
