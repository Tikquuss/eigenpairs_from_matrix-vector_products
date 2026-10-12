"""hessian-deflation.svg: Algorithm 1 on the Hessian of a neural-network training loss.

A fully connected tanh network 4-16-16-1 (369 parameters) is trained by
full-batch gradient descent on n = 256 inputs labelled by a random teacher
network, and its Hessian at the final point is applied by automatic
differentiation (Section 2.1). Algorithm 1 computes the 8 eigenpairs of
largest absolute value with two choices of tolerances:

* relative: eps_abs = eps_rel = 1e-10, so that stage j accepts when
  ||A q - lam q|| <= 1e-10 (1 + ||A q||);
* constant: eps_abs = 1e-10 ||A||, eps_rel = 0, where ||A|| = |lambda_1| is the
  absolute value of the eigenvalue found at the first stage.
"""

import matplotlib.pyplot as plt
import numpy as np

from mvp import deflated_eigenpairs
from mvp.hessian import teacher_student_problem

from .common import BACKGROUND, BLUE, DASHED, DOTTED, GREY, INK, ORANGE, PURPLE, finish, save, stage_colors

NAME = "hessian-deflation"
WIDTHS, TEACHER = [4, 16, 16, 1], [4, 8, 1]
N, STEPS, LEARNING_RATE, SEED = 256, 4000, 0.1, 3
K = 8
N_STARTS = 20
MAX_STEPS = 3000


def problem():
    loss, trajectory = teacher_student_problem(WIDTHS, n=N, steps=STEPS, learning_rate=LEARNING_RATE,
                                               seed=SEED, teacher_widths=TEACHER)
    theta = trajectory[-1]
    return loss, theta


def runs(apply_h, dimension, options, seed_offset):
    return [deflated_eigenpairs(apply_h, dimension, K, np.random.default_rng(seed_offset + start),
                                max_steps=MAX_STEPS, **options)
            for start in range(N_STARTS)]


def make(out_dir, preview=None) -> dict:
    loss, theta = problem()
    apply_h = loss.hvp_routine(theta)
    dense = loss.dense_hessian(theta)
    exact = np.linalg.eigvalsh(dense)
    top = exact[np.argsort(-np.abs(exact))[:K]]
    norm = np.abs(exact).max()
    dimension = theta.size

    relative = dict(eps_abs=1e-10, eps_rel=1e-10)
    constant = dict(eps_abs=1e-10 * norm, eps_rel=0.0)
    runs_relative = runs(apply_h, dimension, relative, 100)
    runs_constant = runs(apply_h, dimension, constant, 100)

    def certified(run):
        return sum(stage.accepted for stage in run)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.3, 1.0, 0.9]})
    fig.patch.set_facecolor(BACKGROUND)

    # (a) One complete run with the constant tolerance.
    axis = axes[0]
    example = runs_constant[0]
    colors = stage_colors(K)
    offset = 0
    for j, (stage, color) in enumerate(zip(example, colors)):
        x = offset + np.array(stage.products)
        axis.plot(x, stage.residuals, color=color, linewidth=1.9, marker="o", markersize=2.2,
                  label=rf"$\lambda_{j + 1}={stage.eigenvalue:.2f}$")
        offset += stage.cost
    axis.axhline(constant["eps_abs"], color=GREY, linewidth=1.1, linestyle=DASHED)
    axis.text(8, constant["eps_abs"] * 1.8, r"tolerance $10^{-10}\|\mathbf{A}\|$", ha="left",
              va="bottom", fontsize=9, color=GREY)
    axis.set_yscale("log")
    axis.set_ylim(1e-11, 3e5)
    axis.set_xlim(0, offset * 1.01)
    axis.set_xlabel("Hessian–vector products")
    axis.set_ylabel(r"residual $\|\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}\|$")
    axis.set_title("(a) One run, stage by stage", pad=8)
    axis.legend(loc="upper center", fontsize=8.6, ncol=4, handlelength=1.4, columnspacing=0.9)

    # (b) A stage that cannot be accepted with the relative tolerance.
    axis = axes[1]
    stalled_run = next(run for run in runs_relative if certified(run) < K)
    stage = stalled_run[-1]
    j = len(stalled_run)
    x = np.array(stage.products)
    tolerance = relative["eps_abs"] + relative["eps_rel"] * abs(stage.eigenvalue)
    axis.plot(x, stage.residuals, color=BLUE, linewidth=2.0,
              label=r"$\|\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}\|$ (checked)")
    axis.plot(x, np.maximum(stage.deflated_residuals, 1e-17), color=ORANGE, linewidth=1.6,
              linestyle=DASHED, label=r"$\|\mathbf{B}\mathbf{q}-\widehat{\lambda}\mathbf{q}\|$")
    axis.plot(x, stage.inherited, color=PURPLE, linewidth=1.8, linestyle=DOTTED,
              label=r"$\|\mathbf{Q}^\top\mathbf{A}\mathbf{q}\|$ (inherited)")
    axis.axhline(tolerance, color=GREY, linewidth=1.1, linestyle=DASHED)
    plateau = stage.residuals[-1]
    note = (rf"stays at ${plateau * 1e10:.2f}\cdot10^{{-10}}$," "\n"
            rf"above the tolerance ${tolerance * 1e10:.2f}\cdot10^{{-10}}$")
    axis.annotate(note, xy=(300, plateau), xytext=(395, 2e-6), ha="right", va="bottom", fontsize=9,
                  color=INK, arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    axis.set_yscale("log")
    axis.set_ylim(1e-17, 30)
    axis.set_xlim(0, 400)
    axis.set_xlabel(rf"Hessian–vector products in stage {j}")
    axis.set_ylabel("residual (log scale)")
    axis.set_title(rf"(b) Stage {j} with the relative tolerance", pad=8)
    axis.legend(loc="upper right", fontsize=9)

    # (c) Fraction of runs that certify the j largest pairs.
    axis = axes[2]
    js = np.arange(1, K + 1)
    for runs_, color, marker, label in [(runs_constant, BLUE, "o", "constant tolerance"),
                                        (runs_relative, ORANGE, "s", "relative tolerance")]:
        counts = np.array([certified(run) for run in runs_])
        axis.plot(js, [(counts >= j).mean() for j in js], color=color, linewidth=2.0, marker=marker,
                  markersize=4.5, label=label)
    axis.set_ylim(-0.04, 1.06)
    axis.set_xticks(js)
    axis.set_xlabel(r"number of eigenpairs $j$")
    axis.set_ylabel(rf"fraction of {N_STARTS} runs certifying $j$ pairs")
    axis.set_title("(c) Runs that do not stall", pad=8)
    axis.legend(loc="lower left", fontsize=9)

    finish(axes)
    save(fig, NAME, out_dir, preview)

    stalls = {}
    for run in runs_relative:
        if certified(run) < K:
            stalls[len(run)] = stalls.get(len(run), 0) + 1
    return {
        "parameters": dimension,
        "train_loss": float(loss(__import__("torch").as_tensor(theta))),
        "top_eigenvalues": top,
        "negative_eigenvalues": int(np.sum(exact < 0)),
        "min_eigenvalue": exact.min(),
        "norm": norm,
        "relative_completed": sum(certified(run) == K for run in runs_relative),
        "relative_stalls_by_stage": stalls,
        "constant_completed": sum(certified(run) == K for run in runs_constant),
        "constant_median_cumulative_products": np.median(
            [np.cumsum([s.cost for s in run]) for run in runs_constant if certified(run) == K], axis=0),
        "example_stage_costs": [s.cost for s in example],
        "example_eigenvalue_errors": [np.abs(exact - s.eigenvalue).min() for s in example],
        "stalled_stage": j,
        "stalled_eigenvalue": stage.eigenvalue,
        "stalled_tolerance": tolerance,
        "stalled_min_residual": min(stage.residuals),
        "stalled_final_inherited": stage.inherited[-1],
        "stalled_min_deflated": min(stage.deflated_residuals),
        "stalled_products": stage.cost,
        "stalled_previous_residuals": [s.residuals[-1] for s in stalled_run[:-1]],
    }
