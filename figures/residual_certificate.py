"""residual-certificate.svg: Lemma 2 and Remark 11 on a sample covariance matrix.

The matrix is the sample covariance (1/n) X^T X of n = 400 centered
observations of d = 60 variables whose population covariance has three
leading eigenvalues 9, 5 and 3 (Section 2.2). It is applied as X^T (X v) / n.
Power iteration estimates the variance along the first principal component,
and the residual certifies the estimate.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from mvp import power_iteration, spiked_covariance

from .common import BACKGROUND, BLUE, DASHED, GREY, INK, ORANGE, band, finish, save, stage_colors, unit_gaussian

NAME = "residual-certificate"
SEED = 400
N, DIMENSION = 400, 60
SPIKES = [9.0, 5.0, 3.0]
N_STARTS = 200
STEPS = 40
SHOWN = [0, 1, 2, 4, 8]


def make(out_dir, preview=None) -> dict:
    rng = np.random.default_rng(SEED)
    apply_c, dense, _ = spiked_covariance(N, SPIKES, DIMENSION, rng)
    exact = np.linalg.eigvalsh(dense)
    lam_1, lam_2 = exact[-1], exact[-2]

    rayleigh = np.empty((N_STARTS, STEPS))
    residual = np.empty((N_STARTS, STEPS))
    example = None
    for start in range(N_STARTS):
        values, norms, iterates = power_iteration(apply_c, unit_gaussian(DIMENSION, rng), STEPS)
        rayleigh[start], residual[start] = values, norms
        if example is None:
            example = iterates
    nearest = np.abs(rayleigh[..., None] - exact).min(axis=-1)
    to_top = np.abs(rayleigh - lam_1)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.0]})
    fig.patch.set_facecolor(BACKGROUND)

    # (a) The certificate mu -> ||A q - mu q|| for several iterates q = v_k.
    axis = axes[0]
    mu = np.linspace(0.0, 11.0, 2201)
    distance = np.abs(mu[:, None] - exact).min(axis=1)
    axis.fill_between(mu, 0, distance, color="#e4e9e6", linewidth=0, zorder=1)
    axis.plot(mu, distance, color=INK, linewidth=1.4, zorder=3,
              label=r"$\min_{\lambda\in\Lambda(\mathbf{A})}|\mu-\lambda|$")
    colors = stage_colors(len(SHOWN))
    labelled = []
    for color, k in zip(colors, SHOWN):
        q = example[:, k]
        aq = apply_c(q)
        quotient = q @ aq
        minimum = np.linalg.norm(aq - quotient * q)
        curve = np.sqrt(np.maximum((aq @ aq) - 2 * mu * quotient + mu**2, 0.0))
        axis.plot(mu, curve, color=color, linewidth=1.8, zorder=4, label=rf"$k={k}$")
        axis.plot([quotient], [minimum], marker="o", markersize=5,
                  color=color, markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        # The minimum projects onto the Rayleigh quotient, whose value is written
        # under the axis; values too close to share a row are stacked.
        axis.plot([quotient, quotient], [0, minimum], color=color, linewidth=1.0,
                  linestyle=DASHED, zorder=4)
        row = sum(abs(quotient - other) < 0.9 for other in labelled)
        labelled.append(quotient)
        axis.annotate(f"{quotient:.2f}", xy=(quotient, 0), xycoords=("data", "axes fraction"),
                      xytext=(0, -7 - 11.5 * row), textcoords="offset points", ha="center", va="top",
                      fontsize=9.5, color=color, fontweight="semibold")
    axis.plot(exact, np.zeros_like(exact), linestyle="none", marker="|", markersize=9,
              color=ORANGE, markeredgewidth=1.1, zorder=6, clip_on=False, label="eigenvalues")
    axis.set_xlim(0, 11)
    axis.set_xticks([0, 2, 4, 6, 8, 10], labels=["0", "", "4", "6", "", ""])
    axis.set_ylim(0, 5.4)
    axis.set_xlabel(r"scalar $\mu$")
    axis.set_ylabel(r"$\|\mathbf{A}\mathbf{v}_k-\mu\,\mathbf{v}_k\|$")
    axis.set_title(r"(a) Certificate for every scalar $\mu$", pad=8)
    axis.legend(loc="upper left", fontsize=8.5, ncol=2, handlelength=1.3, columnspacing=0.8,
                borderaxespad=0.3)

    # (b) Residual against the true errors along power iteration.
    axis = axes[1]
    products = np.arange(1, STEPS + 1)
    floor = 1e-15
    band(axis, products, residual, BLUE, r"residual $\|\mathbf{r}_k\|$")
    band(axis, products, np.maximum(nearest, floor), ORANGE,
         r"$\min_{\lambda\in\Lambda(\mathbf{A})}|\widehat{\lambda}_k-\lambda|$")
    axis.plot(products, np.maximum(np.median(to_top, axis=0), floor), color=GREY, linewidth=1.2,
              linestyle=(0, (1.2, 2.2)), label=r"$|\widehat{\lambda}_k-\lambda_{\max}(\mathbf{A})|$")
    axis.set_yscale("log")
    axis.set_ylim(3e-16, 30)
    axis.set_xlim(0, STEPS)
    axis.set_xlabel(r"matrix–vector products ($k+1$)")
    axis.set_ylabel("error (log scale)")
    axis.set_title("(b) Along power iteration", pad=8)
    axis.legend(loc="lower left", fontsize=9)

    # (c) Every (residual, distance) pair lies below the diagonal.
    axis = axes[2]
    shown_starts = 50
    x = residual[:shown_starts].ravel()
    y = np.maximum(nearest[:shown_starts].ravel(), floor)
    k_index = np.tile(np.arange(STEPS), shown_starts)
    cmap = mpl.colormaps["viridis_r"]
    scatter = axis.scatter(x, y, c=k_index, cmap=cmap, vmin=0, vmax=STEPS - 1, s=7, alpha=0.75,
                           linewidths=0, zorder=3, rasterized=True)
    line = np.array([1e-16, 30])
    axis.fill_between(line, line, 30, color="#f3e1d6", linewidth=0, zorder=1)
    axis.plot(line, line, color=INK, linewidth=1.3, zorder=4)
    axis.text(1e-10, 1e-4, "excluded by Lemma 2", ha="center", va="center", fontsize=9.5, color=ORANGE)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-16, 30)
    axis.set_ylim(1e-16, 30)
    axis.set_xlabel(r"residual $\|\mathbf{r}_k\|$")
    axis.set_ylabel(r"$\min_{\lambda\in\Lambda(\mathbf{A})}|\widehat{\lambda}_k-\lambda|$")
    axis.set_title("(c) The bound holds for every iterate", pad=8)
    colorbar = fig.colorbar(scatter, ax=axis, pad=0.01, fraction=0.05)
    colorbar.set_label(r"iteration $k$")

    finish(axes)
    save(fig, NAME, out_dir, preview)

    ratio = nearest / residual
    k_cross = int(np.argmax(np.median(residual, axis=0) < 1e-10))
    return {
        "eigenvalues_top": exact[::-1][:4],
        "bulk_range": [exact[0], exact[-4]],
        "ratio_lambda2_lambda1": lam_2 / lam_1,
        "max_distance_over_residual": ratio.max(),
        "example_k_rayleigh_residual": [[k, (example[:, k] @ apply_c(example[:, k])),
                                          np.linalg.norm(apply_c(example[:, k])
                                                         - (example[:, k] @ apply_c(example[:, k])) * example[:, k])]
                                         for k in SHOWN],
        "median_residual_at_products": {str(p): np.median(residual[:, p - 1]) for p in (10, 20, 30, 40)},
        "median_nearest_at_products": {str(p): np.median(nearest[:, p - 1]) for p in (10, 20, 30, 40)},
        "first_products_median_residual_below_1e-10": k_cross + 1,
    }
