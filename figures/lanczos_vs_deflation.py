"""lanczos-vs-deflation.svg: Algorithm 2 against Algorithm 1 on two operators of Section 2.

(a) The 8 eigenpairs of largest absolute value of the network Hessian of
hessian-deflation.svg, certified with the constant tolerance 1e-10 ||A||:
products used by Algorithm 1 (cumulative over its stages) and by Algorithm 2
(Lanczos steps, plus one explicit residual check per pair), medians and
interquartile ranges over 20 starting vectors.
(b) The complete eigenbasis of the bipartite graph of complete-eigenbasis.svg:
Algorithm 1 with each stage taken to its limit, against Lanczos runs continued
with new starting vectors (Proposition 4).
"""

import matplotlib.pyplot as plt
import numpy as np

from mvp import Counter, certify_top_k, complete_decomposition, explicit_residuals

from . import complete_eigenbasis, hessian_deflation
from .common import BACKGROUND, BLUE, GREEN, finish, quartiles, save

NAME = "lanczos-vs-deflation"
LANCZOS_SEED = 500
CONTINUATION_SEED = 9


def make(out_dir, preview=None) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    summary = {}

    # (a) Top 8 pairs of the network Hessian.
    loss, theta = hessian_deflation.problem()
    apply_h = loss.hvp_routine(theta)
    exact = np.linalg.eigvalsh(loss.dense_hessian(theta))
    norm = np.abs(exact).max()
    tolerance = 1e-10 * norm
    k_values = np.arange(1, hessian_deflation.K + 1)
    alg1_runs = hessian_deflation.runs(apply_h, theta.size, dict(eps_abs=tolerance, eps_rel=0.0), 100)
    alg1 = np.array([np.cumsum([stage.cost for stage in run]) for run in alg1_runs])
    lanczos_costs, explicit = [], []
    for start in range(hessian_deflation.N_STARTS):
        q1 = np.random.default_rng(LANCZOS_SEED + start).standard_normal(theta.size)
        first, values, residuals, _ = certify_top_k(apply_h, q1, hessian_deflation.K, theta.size,
                                                    eps_abs=tolerance, eps_rel=0.0)
        lanczos_costs.append(first + k_values)
        explicit.append(residuals.max())
    lanczos_costs = np.array(lanczos_costs)

    axis = axes[0]
    for costs, color, marker, label in [(alg1, BLUE, "o", "Algorithm 1"),
                                        (lanczos_costs, GREEN, "s", "Algorithm 2 (Lanczos)")]:
        median, low, high = quartiles(costs)
        axis.fill_between(k_values, low, high, color=color, alpha=0.18, linewidth=0)
        axis.plot(k_values, median, color=color, linewidth=2.0, marker=marker, markersize=4.5, label=label)
    axis.set_yscale("log")
    axis.set_ylim(5, 5000)
    axis.set_xticks(k_values)
    axis.set_xlabel(r"eigenpairs of largest $|\lambda|$ certified")
    axis.set_ylabel("Hessian–vector products")
    axis.set_title("(a) Network Hessian, 8 largest pairs", pad=8)
    axis.legend(loc="upper left")
    summary["hessian_algorithm1_median"] = np.median(alg1, axis=0)
    summary["hessian_lanczos_median"] = np.median(lanczos_costs, axis=0)
    summary["hessian_lanczos_max_explicit_residual"] = max(explicit)
    summary["hessian_tolerance"] = tolerance

    # (b) Complete eigenbasis of the bipartite graph.
    apply_w, dense, _ = complete_eigenbasis.graph()
    dimension = dense.shape[0]
    graph_exact = np.linalg.eigvalsh(dense)
    stages, _ = complete_eigenbasis.run("algorithm1", apply_w, graph_exact)
    x_alg1, y_alg1 = complete_eigenbasis.certified_curve(stages)
    counter = Counter(apply_w)
    certified, theta_values, vectors, run_lengths = complete_decomposition(
        counter, dimension, np.random.default_rng(CONTINUATION_SEED))
    lanczos_products = counter.calls
    values, residuals = explicit_residuals(apply_w, vectors)
    x_lanczos = np.arange(0, certified.size + 1)
    y_lanczos = np.concatenate([[0], certified])

    axis = axes[1]
    axis.step(np.maximum(x_alg1, 1), y_alg1, where="post", color=BLUE, linewidth=2.0, label="Algorithm 1")
    axis.step(np.maximum(x_lanczos, 1), y_lanczos, where="post", color=GREEN, linewidth=2.0,
              label="Lanczos, continued")
    axis.plot([lanczos_products + dimension], [dimension], marker="s", markersize=6, color=GREEN,
              markerfacecolor="white", markeredgewidth=1.4, linestyle="none",
              label=rf"+ {dimension} explicit checks")
    axis.set_xscale("log")
    axis.set_xlim(1, 1e6)
    axis.set_ylim(-1.5, dimension + 2)
    axis.set_xlabel("matrix–vector products")
    axis.set_ylabel("eigenpairs certified")
    axis.set_title("(b) Bipartite graph, complete eigenbasis", pad=8)
    axis.legend(loc="lower right", fontsize=9.5)

    finish(axes)
    save(fig, NAME, out_dir, preview)

    summary.update({
        "graph_algorithm1_products": int(x_alg1[-1]),
        "graph_lanczos_products": lanczos_products,
        "graph_run_lengths": run_lengths,
        "graph_lanczos_max_explicit_residual": residuals.max(),
        "graph_lanczos_max_value_error": np.abs(np.sort(values) - graph_exact).max(),
    })
    return summary
