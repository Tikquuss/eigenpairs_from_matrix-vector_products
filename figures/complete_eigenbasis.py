"""complete-eigenbasis.svg: Corollary 2 on the adjacency matrix of a bipartite graph.

The graph has 30 + 20 nodes in two planted communities, and its adjacency
matrix is applied by one pass over the edge list (Section 2.2). Because the
graph is bipartite, every nonzero eigenvalue comes with its opposite, and 0 is
an eigenvalue of multiplicity 10. Three ways of computing all 50 eigenpairs one
deflation stage at a time are compared: plain deflated power iteration, power
iteration on A - lambda_min(A) I with deflation (Corollary 1 and Lemma 1), and
Algorithm 1. As in Corollary 2, each stage is taken to its limit: it runs until
the residual of its candidate for the deflated matrix B_j falls below
``LIMIT``, and the residual for A is then measured.
"""

import matplotlib.pyplot as plt
import numpy as np

from mvp import Counter, algorithm1_stage, bipartite_graph, deflated_eigenpairs, deflated_power_stage

from .common import BACKGROUND, BLUE, DASHED, GREY, LIGHT_GREY, ORANGE, PURPLE, finish, save

NAME = "complete-eigenbasis"
GRAPH_SEED = 7
STAGE_SEED = 1
LEFT, RIGHT = (15, 15), (10, 10)
P_WITHIN, P_ACROSS = 0.45, 0.08
LIMIT = 1e-13
PLAIN_BUDGET = 20000


def graph():
    return bipartite_graph(LEFT, RIGHT, P_WITHIN, P_ACROSS, np.random.default_rng(GRAPH_SEED))


def run(method: str, apply_w, exact: np.ndarray):
    """All eigenpairs with one of the three methods; returns the stages and the
    total number of products. The reference spectrum ``exact`` provides the
    most favorable valid shift, lambda_min(A), and the constant tolerance
    10^-10 ||A|| (Algorithm 1 finds ||A|| = |lambda_1| at its first stage)."""
    dimension = exact.size
    tolerance = dict(eps_abs=1e-10 * np.abs(exact).max(), eps_rel=0.0, limit=LIMIT)
    counter = Counter(apply_w)
    rng = np.random.default_rng(STAGE_SEED)
    if method == "algorithm1":
        stages = deflated_eigenpairs(counter, dimension, dimension, rng, stage=algorithm1_stage,
                                     max_steps=20000, **tolerance)
    elif method == "shifted":
        stages = deflated_eigenpairs(counter, dimension, dimension, rng, stage=deflated_power_stage,
                                     shift=exact.min(), max_products=400000, **tolerance)
    else:
        stages = deflated_eigenpairs(counter, dimension, dimension, rng, stage=deflated_power_stage,
                                     shift=0.0, max_products=PLAIN_BUDGET, **tolerance)
    return stages, counter.calls


def certified_curve(stages):
    """Cumulative products after each certified stage, starting at (0, 0)."""
    costs = np.cumsum([stage.cost for stage in stages if stage.accepted])
    return np.concatenate([[0], costs]), np.arange(costs.size + 1)


def make(out_dir, preview=None) -> dict:
    apply_w, dense, edges = graph()
    dimension = dense.shape[0]
    exact = np.linalg.eigvalsh(dense)

    norm = np.abs(exact).max()
    results = {method: run(method, apply_w, exact) for method in ("algorithm1", "shifted", "plain")}
    alg1, _ = results["algorithm1"]
    shifted, _ = results["shifted"]
    plain, plain_products = results["plain"]
    assert all(stage.accepted for stage in alg1) and len(alg1) == dimension
    assert all(stage.accepted for stage in shifted) and len(shifted) == dimension

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})
    fig.patch.set_facecolor(BACKGROUND)
    stages = np.arange(1, dimension + 1)

    # (a) Eigenvalue found at each stage.
    axis = axes[0]
    for value in np.unique(np.round(exact, 9)):
        axis.axhline(value, color=LIGHT_GREY, linewidth=0.5, alpha=0.6, zorder=1)
    axis.plot(stages, [s.eigenvalue for s in shifted], linestyle="none", marker="s", markersize=5.2,
              markerfacecolor="none", markeredgecolor=PURPLE, markeredgewidth=1.1, zorder=3,
              label=r"shift $\mathbf{A}-\lambda_{\min}(\mathbf{A})\,\mathbb{I}_d$ and deflation")
    axis.plot(stages, [s.eigenvalue for s in alg1], linestyle="none", marker="o", markersize=4.0,
              color=BLUE, zorder=4, label="Algorithm 1")
    axis.set_xlim(0, dimension + 1)
    axis.set_ylim(-7.4, 9.6)
    axis.set_xlabel(r"stage $j$")
    axis.set_ylabel(r"eigenvalue $\lambda_j$ found at stage $j$")
    axis.set_title("(a) Order in which eigenvalues are found", pad=8)
    axis.legend(loc="upper right", fontsize=9)

    # (b) Certified eigenpairs against products.
    axis = axes[1]
    for stages_run, color, label in [(alg1, BLUE, "Algorithm 1"),
                                     (shifted, PURPLE, r"shift and deflation")]:
        x, y = certified_curve(stages_run)
        axis.step(np.maximum(x, 1), y, where="post", color=color, linewidth=2.0, label=label)
    axis.plot([1, plain_products], [0, 0], color=ORANGE, linewidth=2.4,
              label="plain deflation (stalls at stage 1)")
    axis.set_xscale("log")
    axis.set_xlim(8, 3e5)
    axis.set_ylim(-1.5, dimension + 2)
    axis.set_xlabel("matrix–vector products")
    axis.set_ylabel("eigenpairs found")
    axis.set_title("(b) Cost of the complete eigenbasis", pad=8)
    axis.legend(loc="upper left", fontsize=9)

    # (c) Accuracy of the eigenbasis found by Algorithm 1.
    axis = axes[2]
    q = np.column_stack([s.vector for s in alg1])
    found = np.array([s.eigenvalue for s in alg1])
    residuals = np.array([s.residuals[-1] for s in alg1])
    value_errors = np.array([np.abs(exact - value).min() for value in found])
    gram = np.abs(q.T @ q - np.eye(dimension))
    orthogonality = np.array([gram[:j, j].max() if j else np.nan for j in range(dimension)])
    floor = 1e-17
    axis.plot(stages, np.maximum(residuals, floor), linestyle="none", marker="o", markersize=4.0,
              color=BLUE, label=r"residual $\|\mathbf{A}\mathbf{q}_j-\lambda_j\mathbf{q}_j\|$")
    axis.plot(stages, np.maximum(value_errors, floor), linestyle="none", marker="s", markersize=3.6,
              markerfacecolor="none", markeredgecolor=PURPLE, markeredgewidth=1.0,
              label=r"distance from $\lambda_j$ to $\Lambda(\mathbf{A})$")
    axis.plot(stages, np.maximum(orthogonality, floor), linestyle="none", marker="^", markersize=3.6,
              color=GREY, label=r"$\max_{i<j}|\mathbf{q}_i^\top\mathbf{q}_j|$")
    axis.axhline(1e-10 * norm, color=GREY, linewidth=1.1, linestyle=DASHED)
    axis.text(dimension, 1.5e-10 * norm, r"tolerance $10^{-10}\|\mathbf{A}\|$", ha="right", va="bottom",
              fontsize=9, color=GREY)
    axis.set_yscale("log")
    axis.set_ylim(3e-18, 1e-5)
    axis.set_xlim(0, dimension + 1)
    axis.set_xlabel(r"stage $j$")
    axis.set_ylabel("error (log scale)")
    axis.set_title("(c) Accuracy of Algorithm 1", pad=8)
    axis.legend(loc="upper left", fontsize=9)

    finish(axes)
    save(fig, NAME, out_dir, preview)

    # Algorithm 1 as stated: each stage stops as soon as its residual for A is
    # below the same constant tolerance (no inner limit).
    as_stated = deflated_eigenpairs(Counter(apply_w), dimension, dimension, np.random.default_rng(STAGE_SEED),
                                    stage=algorithm1_stage, eps_abs=1e-10 * norm, eps_rel=0.0, max_steps=20000)
    stalled = as_stated[-1]

    reconstruction = np.linalg.norm(dense - (q * found) @ q.T, 2)
    zeros = int(np.sum(np.abs(exact) < 1e-9))
    return {
        "edges": len(edges),
        "dimension": dimension,
        "zero_multiplicity": zeros,
        "distinct_eigenvalues": int(np.unique(np.round(exact, 9)).size),
        "lambda_max": exact.max(),
        "algorithm1_products": sum(s.cost for s in alg1),
        "algorithm1_zero_stage_costs": [s.cost for s in alg1[-zeros:]],
        "algorithm1_order_by_magnitude": bool(np.all(np.diff(np.abs(found)) <= 1e-9)),
        "shifted_products": sum(s.cost for s in shifted),
        "plain_products": plain_products,
        "plain_min_residual": min(plain[0].residuals),
        "plain_final_residual": plain[0].residuals[-1],
        "max_residual": residuals.max(),
        "max_value_error": value_errors.max(),
        "max_orthogonality": np.nanmax(orthogonality),
        "reconstruction_error": reconstruction,
        "shifted_max_residual": max(s.residuals[-1] for s in shifted),
        "as_stated_certified": sum(s.accepted for s in as_stated),
        "as_stated_stalled_stage": None if stalled.accepted else len(as_stated),
        "as_stated_stalled_eigenvalue": stalled.eigenvalue,
        "as_stated_min_residual": min(stalled.residuals),
        "as_stated_final_inherited": stalled.inherited[-1],
        "as_stated_min_deflated": min(stalled.deflated_residuals),
        "tolerance": 1e-10 * norm,
    }
