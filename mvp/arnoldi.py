"""The Arnoldi method for nonsymmetric matrices (Section 5 of the post)."""

from typing import Callable

import numpy as np

MatVec = Callable[[np.ndarray], np.ndarray]


def arnoldi(apply_a: MatVec, q1: np.ndarray, steps: int):
    """Run up to ``steps`` Arnoldi steps (Definition 3), orthogonalizing each new
    vector against every basis vector, twice. Returns (H, Q): H is the
    (m+1) x m upper Hessenberg matrix of coefficients and Q holds q_1, ..., q_{m+1}
    as columns, so that A Q[:, :m] = Q H. Stops early at an exact breakdown."""
    dimension = q1.size
    basis = np.zeros((dimension, steps + 1))
    hessenberg = np.zeros((steps + 1, steps))
    basis[:, 0] = q1 / np.linalg.norm(q1)
    for j in range(steps):
        w = apply_a(basis[:, j])
        for _ in range(2):
            coefficients = basis[:, : j + 1].T @ w
            w = w - basis[:, : j + 1] @ coefficients
            hessenberg[: j + 1, j] += coefficients
        hessenberg[j + 1, j] = np.linalg.norm(w)
        if hessenberg[j + 1, j] == 0.0:
            return hessenberg[: j + 2, : j + 1], basis[:, : j + 2]
        basis[:, j + 1] = w / hessenberg[j + 1, j]
    return hessenberg, basis


def ritz_pairs(hessenberg: np.ndarray, m: int):
    """Ritz values (possibly complex), unit eigenvectors y of H_m, and the
    residuals h_{m+1,m} |y_m| of the Ritz pairs (Proposition 5)."""
    values, vectors = np.linalg.eig(hessenberg[:m, :m])
    vectors = vectors / np.linalg.norm(vectors, axis=0)
    return values, vectors, hessenberg[m, m - 1] * np.abs(vectors[-1])


def field_of_values(dense: np.ndarray, angles: int = 720) -> np.ndarray:
    """Points on the boundary of the field of values {x* A x : ||x|| = 1}, which
    contains every Ritz value (Proposition 5). For each direction t, the unit
    eigenvector x of the largest eigenvalue of the Hermitian part of
    exp(i t) A gives the boundary point x* A x farthest in that direction."""
    points = []
    for angle in np.linspace(0.0, 2.0 * np.pi, angles, endpoint=False):
        rotated = np.exp(1j * angle) * dense
        _, vectors = np.linalg.eigh((rotated + rotated.conj().T) / 2.0)
        x = vectors[:, -1]
        points.append(x.conj() @ dense @ x)
    points.append(points[0])
    return np.array(points)
