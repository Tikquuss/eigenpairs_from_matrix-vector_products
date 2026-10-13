"""directed-graph.svg: the directed graph whose random walk arnoldi.svg analyzes.

(a) The graph: three groups of 20 nodes in a cycle; every node has four edges
    into the next group and one edge to a random node elsewhere.
(b) Its transition matrix P, ordered by group.
(c) The distribution of the walker, started in group 1: the mass of each group
    rotates through the cycle while it settles.
(d) The eigenvalues of P in the complex plane.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from mvp import directed_community_walk

from .bipartite_graph import GOLD, TEAL
from .common import BACKGROUND, DASHED, GREY, INK, LIGHT_GREY, finish, save

NAME = "directed-graph"
ROSE = "#db2777"
GROUP_COLORS = [TEAL, GOLD, ROSE]
N, GROUPS, FORWARD, OTHER, SEED = 60, 3, 4, 1, 0
STEPS = 30


def walk():
    """The walk of Section 5: (x -> P^T x, P, group of each node, edges)."""
    return directed_community_walk(N, GROUPS, FORWARD, OTHER, np.random.default_rng(SEED))


def node_positions(group):
    """Each group on a sunflower disc; the discs sit on a triangle, so that the
    cycle 1 -> 2 -> 3 -> 1 runs counterclockwise."""
    positions = np.zeros((group.size, 2))
    for g in range(GROUPS):
        members = np.flatnonzero(group == g)
        angle = np.pi / 2 + 2 * np.pi * g / GROUPS
        centre = 2.35 * np.array([np.cos(angle), np.sin(angle)])
        k = np.arange(members.size)
        radius = 0.9 * np.sqrt((k + 0.5) / members.size)
        golden = k * np.pi * (3 - np.sqrt(5))
        positions[members] = centre + np.column_stack([radius * np.cos(golden), radius * np.sin(golden)])
    return positions


def make(out_dir, preview=None) -> dict:
    apply_pt, dense, group, edges = walk()
    positions = node_positions(group)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.6), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)

    # (a) The graph.
    axis = axes[0, 0]
    forward_edges = other_edges = 0
    for tail, head in edges:
        is_forward = group[head] == (group[tail] + 1) % GROUPS
        forward_edges += is_forward
        other_edges += not is_forward
        style = dict(color=GROUP_COLORS[group[tail]], alpha=0.38, linewidth=0.7) if is_forward else \
            dict(color=INK, alpha=0.75, linewidth=0.8, linestyle=DASHED)
        axis.add_patch(FancyArrowPatch(positions[tail], positions[head], arrowstyle="-|>", mutation_scale=7,
                                       shrinkA=3, shrinkB=4, connectionstyle="arc3,rad=0.12", zorder=1, **style))
    for g in range(GROUPS):
        members = group == g
        axis.scatter(positions[members, 0], positions[members, 1], s=30, color=GROUP_COLORS[g],
                     edgecolors="white", linewidths=0.7, zorder=3)
        angle = np.pi / 2 + 2 * np.pi * g / GROUPS
        axis.text(*(3.62 * np.array([np.cos(angle), np.sin(angle)])), f"group {g + 1}", ha="center", va="center",
                  fontsize=11, color=GROUP_COLORS[g], fontweight="semibold")
        # A guide arrow along the outside of the triangle, from this group to the next.
        start, stop = angle + np.radians(32), angle + 2 * np.pi / GROUPS - np.radians(32)
        arc = np.linspace(start, stop, 60)
        axis.plot(3.45 * np.cos(arc[:-3]), 3.45 * np.sin(arc[:-3]), color=GROUP_COLORS[g], linewidth=2.2,
                  alpha=0.75, solid_capstyle="round", zorder=0)
        axis.add_patch(FancyArrowPatch((3.45 * np.cos(arc[-4]), 3.45 * np.sin(arc[-4])),
                                       (3.45 * np.cos(arc[-1]), 3.45 * np.sin(arc[-1])), arrowstyle="-|>",
                                       mutation_scale=16, color=GROUP_COLORS[g], alpha=0.75, linewidth=0, zorder=0))
    axis.set_xlim(-4.1, 4.1)
    axis.set_ylim(-3.75, 4.0)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_title(f"(a) The graph: {N} nodes, {len(edges)} directed edges", pad=8)

    # (b) The transition matrix.
    axis = axes[0, 1]
    image = np.ones((N, N, 3)) * np.array(plt.matplotlib.colors.to_rgb(BACKGROUND))
    for tail, head in edges:
        is_forward = group[head] == (group[tail] + 1) % GROUPS
        image[tail, head] = plt.matplotlib.colors.to_rgb(GROUP_COLORS[group[tail]] if is_forward else INK)
    axis.imshow(image, interpolation="nearest")
    size = N // GROUPS
    for boundary in range(size, N, size):
        axis.axhline(boundary - 0.5, color=LIGHT_GREY, linewidth=0.9)
        axis.axvline(boundary - 0.5, color=LIGHT_GREY, linewidth=0.9)
    centres = [size * g + size / 2 - 0.5 for g in range(GROUPS)]
    axis.set_xticks(centres)
    axis.set_xticklabels([f"group {g + 1}" for g in range(GROUPS)])
    axis.set_yticks(centres)
    axis.set_yticklabels([f"group {g + 1}" for g in range(GROUPS)], rotation=90, va="center")
    for labels in (axis.get_xticklabels(), axis.get_yticklabels()):
        for g, label in enumerate(labels):
            label.set_color(GROUP_COLORS[g])
    axis.tick_params(length=0)
    axis.set_xlabel(r"to node $j$")
    axis.set_ylabel(r"from node $i$")
    axis.set_title(r"(b) Transition matrix $\mathbf{P}$, entries $P_{ij}=1/5$", pad=8)

    # (c) The distribution of the walker.
    axis = axes[1, 0]
    p = (group == 0).astype(float) / size
    masses = [np.bincount(group, weights=p, minlength=GROUPS)]
    for _ in range(STEPS):
        p = apply_pt(p)
        masses.append(np.bincount(group, weights=p, minlength=GROUPS))
    masses = np.array(masses)
    eigenvalues, vectors = np.linalg.eig(dense.T)
    stationary = np.real(vectors[:, np.argmin(np.abs(eigenvalues - 1))])
    stationary /= stationary.sum()
    stationary_masses = np.bincount(group, weights=stationary, minlength=GROUPS)
    axis.axhline(1 / 3, color=GREY, linewidth=1.0, linestyle=DASHED, zorder=1)
    steps = np.arange(STEPS + 1)
    for g in range(GROUPS):
        axis.plot(steps, masses[:, g], color=GROUP_COLORS[g], linewidth=1.6, marker="o", markersize=3.5,
                  label=f"group {g + 1}", zorder=2)
    axis.set_xlim(-0.5, STEPS + 0.5)
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel(r"step $k$")
    axis.set_ylabel(r"probability that the walker is in the group")
    axis.set_title(r"(c) The walker's distribution $\mathbf{p}_k=(\mathbf{P}^\top)^k\mathbf{p}_0$", pad=8)
    axis.legend(loc="upper right", fontsize=9.5)
    finish(axis)

    # (d) The spectrum.
    axis = axes[1, 1]
    order = np.argsort(-np.abs(eigenvalues))
    t = np.linspace(0, 2 * np.pi, 400)
    axis.plot(np.cos(t), np.sin(t), color=LIGHT_GREY, linewidth=1.0, linestyle=DASHED, zorder=1)
    axis.scatter(eigenvalues.real, eigenvalues.imag, s=16, color=GREY, linewidths=0, zorder=2)
    outliers = eigenvalues[order[:3]]
    axis.scatter(outliers.real, outliers.imag, s=46, color=INK, linewidths=0, zorder=3)
    pair = eigenvalues[order[1]] if eigenvalues[order[1]].imag > 0 else eigenvalues[order[2]]
    line = dict(arrowstyle="-", color=INK, linewidth=0.8, shrinkB=5)
    axis.annotate("$1$\nstationary\ndistribution", xy=(1, 0), xytext=(0.66, -0.6), fontsize=9.5, color=INK,
                  ha="center", va="center", arrowprops=line)
    axis.annotate(rf"$\lambda_2={abs(pair):.2f}\,e^{{i\,{np.degrees(np.angle(pair)):.0f}^\circ}}$",
                  xy=(pair.real, pair.imag), xytext=(-0.62, 0.98), fontsize=10, color=INK, ha="center",
                  va="center", arrowprops=line)
    axis.annotate(r"$\bar\lambda_2$", xy=(pair.real, -pair.imag), xytext=(-0.72, -0.95), fontsize=10, color=INK,
                  ha="center", va="center", arrowprops=line)
    axis.set_aspect("equal")
    axis.set_xlim(-1.12, 1.12)
    axis.set_ylim(-1.12, 1.12)
    axis.set_xlabel("real part")
    axis.set_ylabel("imaginary part")
    axis.set_title(r"(d) Eigenvalues of $\mathbf{P}$", pad=8)
    finish(axis)

    save(fig, NAME, out_dir, preview)

    deviation = np.abs(masses - stationary_masses).max(axis=1)
    bulk = np.abs(eigenvalues[order[3:]])
    in_degree = np.bincount(edges[:, 1], minlength=N)
    return {
        "edges": len(edges),
        "forward_edges": int(forward_edges),
        "other_edges": int(other_edges),
        "other_edges_within_group": int(sum(group[h] == group[t] for t, h in edges)),
        "in_degree_range": [int(in_degree.min()), int(in_degree.max())],
        "pair": [float(pair.real), float(pair.imag)],
        "pair_modulus": abs(pair),
        "pair_angle_degrees": np.degrees(np.angle(pair)),
        "bulk_max_modulus": bulk.max(),
        "stationary_group_masses": stationary_masses,
        "stationary_range": [stationary.min(), stationary.max()],
        "mass_group1_steps_0_to_6": masses[:7, 0],
        "deviation_at_10_20_30": deviation[[10, 20, 30]],
        "observed_decay_per_step_10_to_30": (deviation[30] / deviation[10]) ** (1 / 20),
    }
