"""power-iteration.svg: Proposition 1, Remarks 2, 3 and 5, and Proposition 2.

(a) Strict dominance with a negative dominant eigenvalue of multiplicity 3.
(b) A tie between +1 and -1: plain power iteration against squared power
iteration with sign separation, run as in Algorithm 1 without deflation.
Matrices of size 60 with prescribed spectra and random eigenvectors; medians
and interquartile ranges over 200 Gaussian starting vectors.
"""

import matplotlib.pyplot as plt
import numpy as np

from mvp import prescribed_spectrum

from .common import BACKGROUND, BLUE, DASHED, GREY, ORANGE, band, finish, save, unit_gaussian

NAME = "power-iteration"
SEED = 20261012
DIMENSION = 60
N_STARTS = 200
N_PRODUCTS = 60


def strict_dominance_curves(rng: np.random.Generator):
    """Proposition 1 and Remarks 2-3 with lambda_star = -1 (multiplicity 3)."""
    lam_star, multiplicity, rho_2 = -1.0, 3, 0.8
    eigenvalues = np.concatenate(
        [np.full(multiplicity, lam_star), np.linspace(-rho_2, rho_2, DIMENSION - multiplicity)]
    )
    a, u = prescribed_spectrum(eigenvalues, rng)
    u_star = u[:, :multiplicity]

    k = np.arange(N_PRODUCTS + 1)
    signed, raw, rayleigh, bound_vector, bound_value = (
        np.empty((N_STARTS, k.size)) for _ in range(5)
    )
    for start in range(N_STARTS):
        v = unit_gaussian(DIMENSION, rng)
        p_star = u_star @ (u_star.T @ v)
        q_star = p_star / np.linalg.norm(p_star)
        tan_phi = np.linalg.norm(v - p_star) / np.linalg.norm(p_star)
        for step in k:
            av = a @ v
            signed[start, step] = np.linalg.norm((-1.0) ** step * v - q_star)
            raw[start, step] = np.linalg.norm(v - q_star)
            rayleigh[start, step] = abs(v @ av - lam_star)
            v = av / np.linalg.norm(av)
        ratio = rho_2 / abs(lam_star)
        bound_vector[start] = ratio**k * tan_phi
        bound_value[start] = 2 * abs(lam_star) * ratio ** (2 * k) * tan_phi**2
    return k, signed, raw, rayleigh, bound_vector, bound_value


def tie_curves(rng: np.random.Generator):
    """Remark 5 (+1 and -1 tie) against Proposition 2 run as in Algorithm 1."""
    eigenvalues = np.concatenate([[1.0, -1.0], np.linspace(-0.8, 0.8, DIMENSION - 2)])
    a, _ = prescribed_spectrum(eigenvalues, rng)

    products_plain = np.arange(1, N_PRODUCTS + 1)
    products_squared = np.arange(2, N_PRODUCTS + 1, 2)
    plain = np.empty((N_STARTS, products_plain.size))
    squared = np.empty((N_STARTS, products_squared.size))
    for start in range(N_STARTS):
        v0 = unit_gaussian(DIMENSION, rng)

        # Plain power iteration: residual of (v_k^T A v_k, v_k) after k+1 products.
        v = v0.copy()
        for index in range(products_plain.size):
            av = a @ v
            plain[start, index] = np.linalg.norm(av - (v @ av) * v)
            v = av / np.linalg.norm(av)

        # Squared power iteration with sign separation (no deflation, P = I).
        w = v0.copy()
        for index in range(products_squared.size):
            h1 = a @ w
            rho = np.linalg.norm(h1)
            y = h1 / rho
            h2 = a @ y
            s = 1.0 if w @ y >= 0 else -1.0
            n = np.linalg.norm(w + s * y)
            q = (w + s * y) / n
            aq = (h1 + s * h2) / n
            squared[start, index] = np.linalg.norm(aq - (q @ aq) * q)
            w = h2 / np.linalg.norm(h2)
    return products_plain, plain, products_squared, squared


def make(out_dir, preview=None) -> dict:
    rng = np.random.default_rng(SEED)
    k, signed, raw, rayleigh, bound_vector, bound_value = strict_dominance_curves(rng)
    products_plain, plain, products_squared, squared = tie_curves(rng)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)

    axis = axes[0]
    band(axis, k, signed, BLUE, r"$\|(-1)^k\mathbf{v}_k-\mathbf{q}_\star\|$")
    # v_k costs k products; its Rayleigh quotient needs A v_k, one more.
    band(axis, k + 1, rayleigh, ORANGE, r"$|\mathbf{v}_k^\top\mathbf{A}\mathbf{v}_k-\lambda_\star|$")
    axis.plot(k[1:], np.median(bound_vector, axis=0)[1:], color=BLUE, linewidth=1.2,
              linestyle=DASHED)
    axis.plot(k[1:] + 1, np.median(bound_value, axis=0)[1:], color=ORANGE, linewidth=1.2,
              linestyle=DASHED)
    axis.plot([], [], color=GREY, linewidth=1.2, linestyle=DASHED,
              label="bounds of Remark 3")
    axis.plot(k, np.median(raw, axis=0), linestyle="none", marker="o", markersize=2.6,
              color=GREY, label=r"raw $\|\mathbf{v}_k-\mathbf{q}_\star\|$")
    axis.set_title(r"(a) Strict dominance, $\lambda_\star=-1$, $\rho_2=0.8$", pad=8)
    axis.set_ylim(1e-12, 30)
    axis.legend(loc="lower left")

    axis = axes[1]
    band(axis, products_plain, plain, ORANGE, "plain power iteration")
    band(axis, products_squared, squared, BLUE, "squared, with sign separation",
         marker="o", markersize=2.6)
    axis.set_title(r"(b) Tie between $+1$ and $-1$", pad=8)
    axis.set_ylim(1e-8, 30)
    axis.legend(loc="lower left")
    axis.set_ylabel(r"residual $\|\mathbf{A}\mathbf{q}-(\mathbf{q}^\top\mathbf{A}\mathbf{q})\mathbf{q}\|$")

    axes[0].set_ylabel("error (log scale)")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlim(0, N_PRODUCTS)
        axis.set_xlabel("matrix–vector products")
    finish(axes)
    save(fig, NAME, out_dir, preview)
    return {}
