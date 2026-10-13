"""arnoldi.svg: the Arnoldi method on two nonsymmetric matrices (Section 5).

(a) The random walk of directed-graph.svg, through products with P^T:
    eigenvalues, Ritz values and the field of values in the complex plane.
(b) Distance to the eigenvalue 1 and to the complex pair against the number
    of products, for the Arnoldi method and for power iteration.
(c) The discretized convection-diffusion operator, whose eigenvalues are all
    real: its Ritz values leave the real axis but stay in the field of values.
(d) Distance from each Ritz value to the spectrum against its residual, for
    the symmetric operator -u'' and for the convection-diffusion operator.
"""

import matplotlib.pyplot as plt
import numpy as np

from mvp import arnoldi, convection_diffusion
from mvp.arnoldi import field_of_values, ritz_pairs

from .common import (BACKGROUND, BLUE, DASHED, DOTTED, GREEN, GREY, INK, LIGHT_GREY, ORANGE, band, finish, save,
                     stage_colors, unit_gaussian)
from .directed_graph import walk

NAME = "arnoldi"
WALK_STEPS = (5, 10, 20)
PRODUCTS, STARTS = 40, 200
D, GAMMA, STEPS = 40, 0.2, 30
SHOWN = (10, 20, 30)
SCATTER_STARTS = 5
FLOOR = 1e-16


def ritz_statistics(apply_a, dense, exact, steps, rng):
    """Explicit residual and distance to the spectrum of every Ritz pair, m = 1..steps."""
    hessenberg, basis = arnoldi(apply_a, rng.standard_normal(dense.shape[0]), steps)
    rows, values_by_m = [], {}
    for m in range(1, steps + 1):
        values, vectors, cheap = ritz_pairs(hessenberg, m)
        values_by_m[m] = values
        for value, y, c in zip(values, vectors.T, cheap):
            x = basis[:, :m] @ y
            x = x / np.linalg.norm(x)
            rows.append((np.linalg.norm(dense @ x - value * x), np.abs(exact - value).min(), c))
    return np.array(rows), values_by_m


def complex_plane(axis, exact, values_by_m, shown, fov):
    axis.plot(fov.real, fov.imag, color=GREY, linewidth=1.1, zorder=1, label="field of values")
    for color, m in zip(stage_colors(len(shown)), shown):
        values = values_by_m[m]
        axis.scatter(values.real, values.imag, s=30, facecolors="none", edgecolors=[color], linewidths=1.3,
                     label=rf"Ritz values, $m={m}$", zorder=3)
    axis.set_xlabel("real part")
    axis.set_ylabel("imaginary part")


def make(out_dir, preview=None) -> dict:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    summary = {}

    # (a) Ritz values of the random walk.
    apply_pt, dense_p, _, _ = walk()
    a = dense_p.T
    exact_p = np.linalg.eigvals(a)
    order = np.argsort(-np.abs(exact_p))
    pair = exact_p[order[1]] if exact_p[order[1]].imag > 0 else exact_p[order[2]]
    hessenberg, _ = arnoldi(apply_pt, np.random.default_rng(1).standard_normal(a.shape[0]), max(WALK_STEPS))
    values_by_m = {m: ritz_pairs(hessenberg, m)[0] for m in WALK_STEPS}
    axis = axes[0, 0]
    t = np.linspace(0, 2 * np.pi, 400)
    axis.plot(np.cos(t), np.sin(t), color=LIGHT_GREY, linewidth=1.0, linestyle=DASHED, zorder=1)
    axis.scatter(exact_p.real, exact_p.imag, s=14, color=GREY, alpha=0.75, linewidths=0, zorder=2,
                 label="eigenvalues")
    fov_p = field_of_values(a)
    complex_plane(axis, exact_p, values_by_m, WALK_STEPS, fov_p)
    axis.set_aspect("equal")
    axis.set_xlim(-1.12, 1.12)
    axis.set_ylim(-1.12, 1.12)
    axis.set_title(r"(a) Random walk: Ritz values from $\mathbf{P}^\top$", pad=8)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8.5, framealpha=0.95, borderaxespad=0)
    for m in WALK_STEPS:
        summary[f"walk_m{m}_distance_to_1"] = np.abs(values_by_m[m] - 1).min()
        summary[f"walk_m{m}_distance_to_pair"] = np.abs(values_by_m[m] - pair).min()
    summary["walk_m5_ritz_values"] = [[v.real, v.imag] for v in values_by_m[5]]
    summary["walk_fov_real_range"] = [fov_p.real.min(), fov_p.real.max()]
    summary["walk_fov_max_imag"] = np.abs(fov_p.imag).max()
    summary["walk_eigenvector_condition"] = np.linalg.cond(np.linalg.eig(a)[1])

    # (b) Convergence against products, Arnoldi against power iteration.
    rng = np.random.default_rng(2)
    to_one, to_pair, power = [], [], []
    for _ in range(STARTS):
        v = unit_gaussian(a.shape[0], rng)
        hessenberg, _ = arnoldi(apply_pt, v, PRODUCTS)
        ritz = [ritz_pairs(hessenberg, m)[0] for m in range(1, PRODUCTS + 1)]
        to_one.append([np.abs(r - 1).min() for r in ritz])
        to_pair.append([np.abs(r - pair).min() for r in ritz])
        errors = []
        for _ in range(PRODUCTS):
            w = apply_pt(v)
            errors.append(abs(v @ w - 1.0))
            v = w / np.linalg.norm(w)
        power.append(errors)
    to_one, to_pair, power = (np.maximum(np.array(x), FLOOR) for x in (to_one, to_pair, power))
    products = np.arange(1, PRODUCTS + 1)
    axis = axes[0, 1]
    band(axis, products, power, ORANGE, r"power iteration, $|\mathbf{v}_k^\top\mathbf{P}^\top\mathbf{v}_k-1|$")
    band(axis, products, to_one, GREEN, r"Arnoldi, distance to $1$")
    band(axis, products, to_pair, GREEN, r"Arnoldi, distance to $\lambda_2$", linestyle=DASHED)
    axis.plot(products, np.abs(pair) ** products, color=GREY, linewidth=1.1, linestyle=DOTTED,
              label=rf"$|\lambda_2|^k={abs(pair):.2f}^k$")
    axis.set_yscale("log")
    axis.set_ylim(FLOOR / 3, 10)
    axis.set_xlim(0, PRODUCTS + 0.5)
    axis.set_xlabel("matrix–vector products")
    axis.set_ylabel("distance to the eigenvalue")
    axis.set_title("(b) Arnoldi against power iteration", pad=8)
    axis.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    for k in (10, 20, 30, 40):
        summary[f"median_power_k{k}"] = np.median(power[:, k - 1])
        summary[f"median_arnoldi_to_one_m{k}"] = np.median(to_one[:, k - 1])
        summary[f"median_arnoldi_to_pair_m{k}"] = np.median(to_pair[:, k - 1])

    # (c) and (d) The convection-diffusion operator against -u''.
    rows_sym = []
    rows_ns = []
    for seed in range(SCATTER_STARTS):
        rows, _ = ritz_statistics(*convection_diffusion(D, 0.0)[:3], STEPS, np.random.default_rng(10 + seed))
        rows_sym.append(rows)
        rows, by_m = ritz_statistics(*convection_diffusion(D, GAMMA)[:3], STEPS, np.random.default_rng(10 + seed))
        rows_ns.append(rows)
        if seed == 0:
            values_by_m = by_m
    rows_sym, rows_ns = np.vstack(rows_sym), np.vstack(rows_ns)
    _, dense_a, exact_a, condition = convection_diffusion(D, GAMMA)

    axis = axes[1, 0]
    fov_a = field_of_values(dense_a)
    axis.axhline(0, color=LIGHT_GREY, linewidth=0.9, zorder=1)
    axis.plot(exact_a, np.zeros_like(exact_a), linestyle="none", marker="|", markersize=11, color=INK,
              markeredgewidth=1.1, label="eigenvalues (all real)", zorder=2)
    complex_plane(axis, exact_a, values_by_m, SHOWN, fov_a)
    axis.set_xlim(-0.25, 4.25)
    axis.set_ylim(-0.5, 0.82)
    axis.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4])
    axis.set_title(rf"(c) Convection–diffusion, $\gamma={GAMMA}$: Ritz values", pad=8)
    axis.legend(loc="upper center", fontsize=8.5, ncol=3, framealpha=0.95)

    axis = axes[1, 1]
    line = np.array([1e-6, 10.0])
    axis.fill_between(line, line, 100, color="#f6e4d8", linewidth=0, zorder=1)
    axis.plot(line, line, color=INK, linewidth=1.2, zorder=2)
    axis.scatter(rows_sym[:, 0], np.maximum(rows_sym[:, 1], FLOOR), s=8, color=BLUE, alpha=0.55, linewidths=0,
                 zorder=3, label=r"$-u''$ (symmetric, $\gamma=0$)", rasterized=True)
    axis.scatter(rows_ns[:, 0], np.maximum(rows_ns[:, 1], FLOOR), s=8, color=ORANGE, alpha=0.55, linewidths=0,
                 zorder=4, label=rf"$-u''+cu'$ (nonsymmetric, $\gamma={GAMMA}$)", rasterized=True)
    axis.text(0.06, 0.93, "excluded by Lemma 2\nfor symmetric matrices", transform=axis.transAxes, fontsize=9.5,
              color=ORANGE, ha="left", va="top")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1e-3, 5)
    axis.set_ylim(1e-5, 5)
    axis.set_xlabel(r"residual $\|\mathbf{A}\mathbf{x}-\vartheta\mathbf{x}\|$")
    axis.set_ylabel(r"distance from $\vartheta$ to the spectrum")
    axis.set_title("(d) The residual certificate without symmetry", pad=8)
    axis.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    finish(axes.ravel())
    save(fig, NAME, out_dir, preview)

    ratio_sym = rows_sym[:, 1] / rows_sym[:, 0]
    ratio_ns = rows_ns[:, 1] / rows_ns[:, 0]
    summary.update({
        "pairs_per_operator": len(rows_ns),
        "symmetric_above_diagonal": int(np.sum(ratio_sym > 1)),
        "symmetric_max_ratio": ratio_sym.max(),
        "nonsymmetric_above_diagonal": int(np.sum(ratio_ns > 1)),
        "nonsymmetric_max_ratio": ratio_ns.max(),
        "nonsymmetric_bauer_fike_min_slack": (condition * rows_ns[:, 0] / np.maximum(rows_ns[:, 1], 1e-300)).min(),
        "nonsymmetric_condition": condition,
        "nonsymmetric_complex_ritz_shown": {m: int(np.sum(np.abs(values_by_m[m].imag) > 1e-12)) for m in SHOWN},
        "nonsymmetric_max_imag_shown": max(np.abs(values_by_m[m].imag).max() for m in SHOWN),
        "nonsymmetric_fov_max_imag": np.abs(fov_a.imag).max(),
        "nonsymmetric_fov_real_range": [fov_a.real.min(), fov_a.real.max()],
        "eigenvalue_range": [exact_a.min(), exact_a.max()],
        "cheap_vs_explicit_residual": float(np.abs(rows_ns[:, 2] / rows_ns[:, 0] - 1).max()),
        "residual_range": [float(min(rows_sym[:, 0].min(), rows_ns[:, 0].min())),
                           float(max(rows_sym[:, 0].max(), rows_ns[:, 0].max()))],
        "distance_min": float(min(rows_sym[:, 1].min(), rows_ns[:, 1].min())),
    })
    return summary
