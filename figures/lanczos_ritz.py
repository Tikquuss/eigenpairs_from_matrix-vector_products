"""lanczos.svg: Lemma 5 and Proposition 3.

One Lanczos run on a matrix of size 60 with 12 distinct eigenvalues, the
extreme Ritz values against power iteration at equal numbers of products, and
the exact breakdown at the number of distinct eigenvalues. Medians and
interquartile ranges over 200 Gaussian starting vectors.
"""

import matplotlib.pyplot as plt
import numpy as np

from mvp import lanczos, prescribed_spectrum, ritz_pairs

from .common import BACKGROUND, GREEN, GREY, INK, LIGHT_GREY, ORANGE, band, finish, save, unit_gaussian

NAME = "lanczos"
SEED = 1950
DIMENSION = 60
N_STARTS = 200

# 12 distinct eigenvalues with multiplicities summing to 60.
EIGENVALUES = np.array([-1.5, -1.0, -0.6, -0.3, 0.0, 0.25, 0.5, 0.8, 1.1, 1.4, 1.7, 2.0])
MULTIPLICITIES = np.array([3, 6, 2, 9, 8, 4, 5, 7, 1, 6, 4, 5])


def power_rayleigh_quotients(a: np.ndarray, v0: np.ndarray, products: int) -> np.ndarray:
    """Entry m-1 is v_{m-1}^T A v_{m-1}, which costs m products in total."""
    values = np.empty(products)
    v = v0.copy()
    for index in range(products):
        av = a @ v
        values[index] = v @ av
        v = av / np.linalg.norm(av)
    return values


def make(out_dir, preview=None) -> dict:
    rng = np.random.default_rng(SEED)
    a, _ = prescribed_spectrum(np.repeat(EIGENVALUES, MULTIPLICITIES), rng)
    s = EIGENVALUES.size
    lam_max = EIGENVALUES.max()
    m = np.arange(1, s + 1)

    errors_lanczos = np.empty((N_STARTS, s))
    errors_power = np.empty((N_STARTS, s))
    betas_all = np.empty((N_STARTS, s))
    final_errors = np.empty(N_STARTS)
    example = None
    for start in range(N_STARTS):
        q1 = unit_gaussian(DIMENSION, rng)
        alphas, betas, _ = lanczos(lambda x: a @ x, q1, s)
        ritz = [ritz_pairs(alphas, betas, step)[0] for step in m]
        rayleigh = power_rayleigh_quotients(a, q1, s)
        errors_lanczos[start] = [abs(lam_max - values[-1]) for values in ritz]
        errors_power[start] = abs(lam_max - rayleigh)
        betas_all[start] = betas
        final_errors[start] = np.abs(ritz[-1] - EIGENVALUES).max()
        if example is None:
            example = (ritz, rayleigh)

    fig, axes = plt.subplots(
        1, 3, figsize=(14.0, 4.5), layout="constrained",
        gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]},
    )
    fig.patch.set_facecolor(BACKGROUND)

    # (a) Ritz values of one run.
    axis = axes[0]
    ritz, rayleigh = example
    for value, multiplicity in zip(EIGENVALUES, MULTIPLICITIES):
        axis.axhline(value, color=LIGHT_GREY, linewidth=0.9, zorder=1)
        axis.text(s + 0.55, value, rf"$\times{multiplicity}$", va="center", ha="left",
                  fontsize=9, color=GREY)
    axis.axvspan(s - 0.4, s + 0.4, color="#e8efe9", zorder=0)
    for step, values in zip(m, ritz):
        axis.scatter(np.full(values.size, step), values, s=16, color=INK, zorder=3,
                     label="Ritz values" if step == 1 else None)
    axis.plot(m, [values[-1] for values in ritz], color=GREEN, linewidth=1.3, zorder=2,
              label="extreme Ritz values")
    axis.plot(m, [values[0] for values in ritz], color=GREEN, linewidth=1.3, zorder=2)
    axis.plot(m, rayleigh, color=ORANGE, linewidth=1.6, marker="o", markersize=3.2,
              zorder=4, label="power iteration")
    axis.set_xlim(0.4, s + 1.4)
    axis.set_xticks(m)
    axis.set_ylim(-1.75, 2.75)
    axis.set_yticks(np.arange(-1.5, 2.01, 0.5))
    axis.set_xlabel(r"Lanczos step $m$")
    axis.set_ylabel("eigenvalue")
    axis.set_title(r"(a) One run: Ritz values, exact at $m=s=12$", pad=8)
    axis.legend(loc="upper center", ncol=3, fontsize=9, handlelength=1.6,
                columnspacing=1.2, borderaxespad=0.35)

    # (b) Distance to the largest eigenvalue at equal numbers of products.
    axis = axes[1]
    floor = 1e-17
    band(axis, m, np.maximum(errors_power, floor), ORANGE, "power iteration",
         marker="o", markersize=3)
    band(axis, m, np.maximum(errors_lanczos, floor), GREEN, "largest Ritz value",
         marker="o", markersize=3)
    axis.set_yscale("log")
    axis.set_ylim(1e-17, 10)
    axis.set_xticks(m)
    axis.set_xlabel(r"matrix–vector products $m$")
    axis.set_ylabel(r"distance to $\lambda_{\max}(\mathbf{A})=2$")
    axis.set_title("(b) Largest eigenvalue", pad=8)
    axis.legend(loc="lower left")

    # (c) The coefficient beta_m.
    axis = axes[2]
    band(axis, m, betas_all, GREEN, r"$\beta_m$", marker="o", markersize=3)
    axis.set_yscale("log")
    axis.set_ylim(1e-17, 10)
    axis.set_xticks(m)
    axis.set_xlabel(r"Lanczos step $m$")
    axis.set_ylabel(r"$\beta_m$")
    axis.set_title(r"(c) Exact breakdown at $m=s$", pad=8)
    axis.legend(loc="lower left")

    finish(axes)
    save(fig, NAME, out_dir, preview)
    return {
        "median_beta_11": np.median(betas_all[:, s - 2]),
        "median_beta_12": np.median(betas_all[:, s - 1]),
        "max_ritz_error_at_s": final_errors.max(),
    }
