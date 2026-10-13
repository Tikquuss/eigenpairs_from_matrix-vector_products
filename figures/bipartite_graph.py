"""bipartite-graph.svg: the graph whose adjacency matrix complete-eigenbasis.svg decomposes.

(a) The graph: 30 left and 20 right nodes, two planted communities on each side.
(b) Its adjacency matrix W, ordered left then right and by community.
(c) The eigenvalues of W: +sigma and -sigma for every singular value sigma of the
    biadjacency block, and 0 with multiplicity |30 - 20| = 10.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from . import complete_eigenbasis
from .common import BACKGROUND, DASHED, GREY, INK, LIGHT_GREY, finish, save

NAME = "bipartite-graph"
TEAL = "#0d9488"
GOLD = "#ca8a04"
COMMUNITY_COLORS = [TEAL, GOLD]


def node_positions(sizes, x, gap=1.6):
    """Evenly spaced nodes of each community, stacked from top to bottom; both
    sides use the same vertical interval for a given community."""
    spans = [(0.0, -14.0), (-14.0 - gap, -28.0 - gap)]
    ys = [np.linspace(top, bottom, size) for (top, bottom), size in zip(spans, sizes)]
    return np.column_stack([np.full(sum(sizes), x), np.concatenate(ys)])


def make(out_dir, preview=None) -> dict:
    _, dense, edges = complete_eigenbasis.graph()
    left, right = complete_eigenbasis.LEFT, complete_eigenbasis.RIGHT
    n_left = sum(left)
    group = np.concatenate([np.repeat([0, 1], left), np.repeat([0, 1], right)])
    positions = np.vstack([node_positions(left, 0.0), node_positions(right, 2.4)])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.05, 1.0, 1.15]})
    fig.patch.set_facecolor(BACKGROUND)

    # (a) The graph.
    axis = axes[0]
    within = across = 0
    for head, tail in edges:
        (x0, y0), (x1, y1) = positions[head], positions[tail]
        if group[head] == group[tail]:
            within += 1
            axis.plot([x0, x1], [y0, y1], color=COMMUNITY_COLORS[group[head]], linewidth=0.8, alpha=0.45, zorder=1)
        else:
            across += 1
            axis.plot([x0, x1], [y0, y1], color=GREY, linewidth=0.8, alpha=0.75, linestyle=DASHED, zorder=1)
    for g in (0, 1):
        members = group == g
        axis.scatter(positions[members, 0], positions[members, 1], s=34, color=COMMUNITY_COLORS[g],
                     edgecolors="white", linewidths=0.8, zorder=3)
    axis.text(0.0, 2.2, f"{n_left} nodes", ha="center", va="bottom", fontsize=10, color=INK)
    axis.text(2.4, 2.2, f"{sum(right)} nodes", ha="center", va="bottom", fontsize=10, color=INK)
    axis.set_xlim(-0.6, 3.0)
    axis.set_ylim(-31.5, 4.2)
    axis.axis("off")
    axis.set_title(f"(a) The graph: {len(edges)} edges", pad=8)

    # (b) The adjacency matrix.
    axis = axes[1]
    axis.imshow(dense, cmap=ListedColormap([BACKGROUND, INK]), interpolation="nearest")
    for boundary in (left[0], n_left, n_left + right[0]):
        axis.axhline(boundary - 0.5, color=LIGHT_GREY, linewidth=0.9)
        axis.axvline(boundary - 0.5, color=LIGHT_GREY, linewidth=0.9)
    axis.axhline(n_left - 0.5, color=GREY, linewidth=1.4)
    axis.axvline(n_left - 0.5, color=GREY, linewidth=1.4)
    for (row, col), label in {(n_left / 2, n_left / 2): "0", (n_left + sum(right) / 2, n_left + sum(right) / 2): "0",
                              (n_left / 2, n_left + sum(right) / 2): r"$\mathbf{B}$",
                              (n_left + sum(right) / 2, n_left / 2): r"$\mathbf{B}^\top$"}.items():
        axis.text(col, row, label, ha="center", va="center", fontsize=15, color=INK,
                  bbox=dict(boxstyle="round,pad=0.25", facecolor=BACKGROUND, edgecolor=LIGHT_GREY, alpha=0.92))
    axis.set_xticks([n_left / 2 - 0.5, n_left + sum(right) / 2 - 0.5])
    axis.set_xticklabels(["left", "right"])
    axis.set_yticks([n_left / 2 - 0.5, n_left + sum(right) / 2 - 0.5])
    axis.set_yticklabels(["left", "right"], rotation=90, va="center")
    axis.tick_params(length=0)
    axis.set_title(r"(b) Adjacency matrix $\mathbf{W}$", pad=8)

    # (c) The spectrum.
    axis = axes[2]
    eigenvalues = np.sort(np.linalg.eigvalsh(dense))[::-1]
    zero = np.abs(eigenvalues) < 1e-9
    index = np.arange(1, eigenvalues.size + 1)
    axis.axhline(0, color=LIGHT_GREY, linewidth=0.9, zorder=1)
    axis.vlines(index, 0, eigenvalues, color=INK, linewidth=1.0, zorder=2)
    axis.plot(index[~zero], eigenvalues[~zero], linestyle="none", marker="o", markersize=4, color=INK, zorder=3)
    axis.plot(index[zero], eigenvalues[zero], linestyle="none", marker="o", markersize=4, color=GREY, zorder=3)
    z0, z1 = index[zero][0], index[zero][-1]
    axis.annotate(f"$0$, repeated {zero.sum()} times", xy=((z0 + z1) / 2, 0.15), xytext=((z0 + z1) / 2, 2.6),
                  ha="center", fontsize=10, color=GREY,
                  arrowprops=dict(arrowstyle="-[", color=GREY, linewidth=0.9, mutation_scale=4.5))
    axis.set_xlim(0, eigenvalues.size + 1)
    axis.set_xlabel("index, in decreasing order")
    axis.set_ylabel(r"eigenvalue of $\mathbf{W}$")
    axis.set_title(r"(c) Eigenvalues: $\pm\sigma$ pairs and zeros", pad=8)
    finish(axis)

    save(fig, NAME, out_dir, preview)

    singular_values = np.linalg.svd(dense[:n_left, n_left:], compute_uv=False)
    nonzero = np.sort(eigenvalues[~zero])
    return {
        "edges": len(edges),
        "within_edges": within,
        "across_edges": across,
        "zero_multiplicity": int(zero.sum()),
        "rank_of_B": int(np.sum(singular_values > 1e-9)),
        "max_pair_asymmetry": float(np.abs(nonzero + nonzero[::-1]).max()),
        "max_mismatch_with_plus_minus_singular_values": float(np.abs(
            np.sort(np.concatenate([singular_values, -singular_values])) - nonzero).max()),
    }
