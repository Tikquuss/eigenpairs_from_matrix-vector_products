"""hessian-deflation.svg: Algorithm 1 on the Hessian of a neural-network training loss.

A fully connected tanh network 4-16-16-1 (369 parameters) is trained by
full-batch gradient descent on n = 256 inputs labelled by a random teacher
network, and its Hessian at the final point is applied by automatic
differentiation (Section 2.1). The top row shows the matrix: (a) the training
and test losses along gradient descent, where the test loss uses 1,024 fresh
inputs labelled by the same teacher, (b) the Hessian spectrum every
SPECTRUM_EVERY steps of the same run, from the dense matrix, and (c) the
spectrum at the final point. The bottom row shows Algorithm 1, which computes
the 8 eigenpairs of largest absolute value with two choices of tolerances:

* relative: eps_abs = eps_rel = 1e-10, so that stage j accepts when
  ||A q - lam q|| <= 1e-10 (1 + ||A q||);
* constant: eps_abs = 1e-10 ||A||, eps_rel = 0, where ||A|| = |lambda_1| is the
  absolute value of the eigenvalue found at the first stage.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from mvp import deflated_eigenpairs
from mvp.hessian import teacher_student_problem

from .common import BACKGROUND, BLUE, DASHED, DOTTED, GREY, INK, ORANGE, PURPLE, finish, save, stage_colors

NAME = "hessian-deflation"
WIDTHS, TEACHER = [4, 16, 16, 1], [4, 8, 1]
N, STEPS, LEARNING_RATE, SEED = 256, 8000, 0.1, 3
K = 8
N_STARTS = 20
MAX_STEPS = 3000
SPECTRUM_EVERY = 80


def problem(return_trajectory: bool = False):
    loss, trajectory = teacher_student_problem(WIDTHS, n=N, steps=STEPS, learning_rate=LEARNING_RATE,
                                               seed=SEED, teacher_widths=TEACHER)
    theta = trajectory[-1]
    return (loss, theta, trajectory) if return_trajectory else (loss, theta)


def runs(apply_h, dimension, options, seed_offset):
    return [deflated_eigenpairs(apply_h, dimension, K, np.random.default_rng(seed_offset + start),
                                max_steps=MAX_STEPS, **options)
            for start in range(N_STARTS)]


def make(out_dir, preview=None) -> dict:
    loss, theta, trajectory = problem(return_trajectory=True)
    with torch.no_grad():
        train_curve = np.array([float(loss(torch.as_tensor(t))) for t in trajectory])
        test_curve = np.array([float(loss.test(torch.as_tensor(t))) for t in trajectory])
    apply_h = loss.hvp_routine(theta)
    dense = loss.dense_hessian(theta)
    exact = np.linalg.eigvalsh(dense)
    sampled = np.arange(0, STEPS + 1, SPECTRUM_EVERY)
    spectra = np.array([np.linalg.eigvalsh(loss.dense_hessian(trajectory[step])) for step in sampled])
    top = exact[np.argsort(-np.abs(exact))[:K]]
    norm = np.abs(exact).max()
    dimension = theta.size

    relative = dict(eps_abs=1e-10, eps_rel=1e-10)
    constant = dict(eps_abs=1e-10 * norm, eps_rel=0.0)
    runs_relative = runs(apply_h, dimension, relative, 100)
    runs_constant = runs(apply_h, dimension, constant, 100)

    def certified(run):
        return sum(stage.accepted for stage in run)

    fig, axes = plt.subplots(2, 3, figsize=(16.0, 8.8), layout="constrained")
    fig.patch.set_facecolor(BACKGROUND)
    axes = axes.ravel()
    colors = stage_colors(K)
    # Several line samples in one legend entry, one per colour.
    by_colour = {tuple: HandlerTuple(ndivide=None, pad=0.0)}

    # (a) Training and test losses along gradient descent.
    axis = axes[0]
    steps = np.arange(train_curve.size)
    axis.plot(steps, train_curve, color=BLUE, linewidth=2.0, label="training loss")
    axis.plot(steps, test_curve, color=ORANGE, linewidth=2.0, linestyle=DASHED, label="test loss")
    axis.plot([steps[-1]], [train_curve[-1]], marker="o", markersize=6, color=INK, zorder=4)
    axis.annotate("Hessian taken here", xy=(steps[-1], train_curve[-1]), xytext=(steps[-1] * 0.55, train_curve[-1] * 0.62),
                  fontsize=9, color=INK, arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    axis.set_yscale("log")
    axis.set_ylim(train_curve[-1] * 0.45, max(train_curve.max(), test_curve.max()) * 1.6)
    axis.set_xlim(0, steps[-1] * 1.03)
    axis.set_xlabel("gradient-descent step")
    axis.set_ylabel(r"squared loss $\frac{1}{2n}\sum_i(z_i-y_i)^2$")
    axis.set_title("(a) Training of the network", pad=8)
    axis.legend(loc="upper right", fontsize=9)

    # (b) The spectrum along the same run: the K largest eigenvalues in the stage
    # colours of panel (d), and a band holding all the others.
    axis = axes[1]
    descending = spectra[:, ::-1]
    axis.fill_between(sampled, spectra[:, 0], descending[:, K], color="#e4e9e6", linewidth=0, zorder=1)
    smallest, = axis.plot(sampled, spectra[:, 0], color=GREY, linewidth=1.3, zorder=2)
    largest = [axis.plot(sampled, descending[:, j], color=colors[j], linewidth=1.7, zorder=3)[0]
               for j in range(K)]
    axis.axhline(0, color=INK, linewidth=0.8, zorder=2)
    band = plt.Rectangle((0, 0), 1, 1, color="#e4e9e6")
    axis.legend([tuple(largest), band, smallest],
                [rf"the {K} largest, light to dark", f"the other {dimension - K}", "the smallest"],
                handler_map=by_colour, loc="lower right", fontsize=9)
    axis.set_yscale("symlog", linthresh=1e-2, linscale=0.6)
    axis.set_yticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10])
    axis.set_yticklabels(["$-1$", "$-0.1$", "$-0.01$", "$0$", "$0.01$", "$0.1$", "$1$", "$10$"])
    axis.set_ylim(-1.6, 12)
    axis.set_xlim(0, sampled[-1] * 1.03)
    axis.set_xlabel("gradient-descent step")
    axis.set_ylabel(r"eigenvalue (log scale beyond $\pm0.01$)")
    axis.set_title("(b) Spectrum during training", pad=8)

    # (c) The spectrum at the final point, by order of magnitude: positive
    # eigenvalues above the axis, negative ones below.
    axis = axes[2]
    magnitude = np.abs(exact)
    edges = 10.0 ** np.arange(np.floor(2 * np.log10(magnitude.min())) / 2, 1.01, 0.5)
    positive, _ = np.histogram(exact[exact > 0], edges)
    negative, _ = np.histogram(-exact[exact < 0], edges)
    bars = dict(width=np.diff(edges), align="edge", edgecolor=BACKGROUND, linewidth=0.8, zorder=2)
    axis.bar(edges[:-1], positive, color=BLUE, label=f"{int((exact > 0).sum())} positive eigenvalues", **bars)
    axis.bar(edges[:-1], -negative, color=ORANGE, label=f"{int((exact < 0).sum())} negative eigenvalues", **bars)
    axis.axhline(0, color=INK, linewidth=0.9, zorder=3)
    marks = []
    for value, color in zip(descending[-1, :K], colors):
        axis.plot([value, value], [0.9, 0.98], transform=axis.get_xaxis_transform(), color=color,
                  linewidth=2.2, solid_capstyle="butt", zorder=4)
        marks.append(Line2D([], [], color=color, marker="|", markersize=9, markeredgewidth=2.2, linestyle="none"))
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles + [tuple(marks)], labels + [rf"the {K} largest, computed in (d)"],
                handler_map=by_colour, loc="lower left", fontsize=9)
    axis.set_xscale("log")
    axis.set_xlim(edges[0], edges[-1])
    height = max(positive.max(), negative.max()) * 1.45
    axis.set_ylim(-height, height)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{abs(value):g}"))
    axis.set_xlabel(r"absolute value $|\lambda|$, in bins of half a decade")
    axis.set_ylabel("number of eigenvalues")
    axis.set_title("(c) Spectrum at the last step", pad=8)

    # (d) One complete run with the constant tolerance.
    axis = axes[3]
    example = runs_constant[0]
    offset = 0
    for j, (stage, color) in enumerate(zip(example, colors)):
        x = offset + np.array(stage.products)
        axis.plot(x, stage.residuals, color=color, linewidth=1.9, marker="o", markersize=2.2,
                  label=rf"$\lambda_{j + 1}={stage.eigenvalue:.2f}$")
        offset += stage.cost
    axis.axhline(constant["eps_abs"], color=GREY, linewidth=1.1, linestyle=DASHED)
    axis.text(8, constant["eps_abs"] * 1.8, r"threshold $10^{-10}\|\mathbf{A}\|$", ha="left",
              va="bottom", fontsize=9, color=GREY)
    axis.set_yscale("log")
    axis.set_ylim(1e-11, 3e5)
    axis.set_xlim(0, offset * 1.01)
    axis.set_xlabel("Hessian–vector products, all stages")
    axis.set_ylabel(r"residual $\|\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}\|$")
    axis.set_title("(d) One run with the constant threshold", pad=8)
    axis.legend(loc="upper center", fontsize=8.6, ncol=4, handlelength=1.4, columnspacing=0.9)

    # (e) A stage that cannot be accepted with the relative tolerance.
    axis = axes[4]
    # The first run that stalls at the stage where stalls are most common.
    stall_stages = [len(run) for run in runs_relative if certified(run) < K]
    common = max(sorted(set(stall_stages)), key=stall_stages.count)
    stalled_run = next(run for run in runs_relative if certified(run) < K and len(run) == common)
    stage = stalled_run[-1]
    j = len(stalled_run)
    x = np.array(stage.products)
    tolerance = relative["eps_abs"] + relative["eps_rel"] * abs(stage.eigenvalue)
    axis.plot(x, stage.residuals, color=BLUE, linewidth=2.0,
              label=r"residual for $\mathbf{A}$, $\|\mathbf{A}\mathbf{q}-\widehat{\lambda}\mathbf{q}\|$ (checked)")
    axis.plot(x, np.maximum(stage.deflated_residuals, 1e-17), color=ORANGE, linewidth=1.6,
              linestyle=DASHED, label=r"residual for $\mathbf{B}$, $\|\mathbf{B}\mathbf{q}-\widehat{\lambda}\mathbf{q}\|$")
    axis.plot(x, stage.inherited, color=PURPLE, linewidth=1.8, linestyle=DOTTED,
              label=r"inherited part, $\|\mathbf{Q}^\top\mathbf{A}\mathbf{q}\|$")
    axis.axhline(tolerance, color=GREY, linewidth=1.1, linestyle=DASHED)
    plateau = stage.residuals[-1]
    note = (rf"stays at ${plateau * 1e10:.2f}\cdot10^{{-10}}$," "\n"
            rf"above the threshold ${tolerance * 1e10:.2f}\cdot10^{{-10}}$")
    axis.annotate(note, xy=(300, plateau), xytext=(395, 2e-6), ha="right", va="bottom", fontsize=9,
                  color=INK, arrowprops=dict(arrowstyle="->", color=INK, linewidth=0.9))
    axis.set_yscale("log")
    axis.set_ylim(1e-17, 30)
    axis.set_xlim(0, 400)
    axis.set_xlabel(rf"Hessian–vector products in stage {j}")
    axis.set_ylabel("residual and its two parts (log scale)")
    axis.set_title(rf"(e) Stage {j} with the relative threshold", pad=8)
    axis.legend(loc="upper right", fontsize=9)

    # (f) Fraction of runs that certify the j largest pairs.
    axis = axes[5]
    js = np.arange(1, K + 1)
    for runs_, color, marker, label in [(runs_constant, BLUE, "o", "constant threshold"),
                                        (runs_relative, ORANGE, "s", "relative threshold")]:
        counts = np.array([certified(run) for run in runs_])
        axis.plot(js, [(counts >= j).mean() for j in js], color=color, linewidth=2.0, marker=marker,
                  markersize=4.5, label=label)
    axis.set_ylim(-0.04, 1.06)
    axis.set_xticks(js)
    axis.set_xlabel(r"number $j$ of largest eigenpairs")
    axis.set_ylabel(rf"fraction of {N_STARTS} runs certifying the $j$ largest")
    axis.set_title("(f) Runs that do not stall", pad=8)
    axis.legend(loc="lower left", fontsize=9)

    finish(axes)
    save(fig, NAME, out_dir, preview)

    stalls = {}
    for run in runs_relative:
        if certified(run) < K:
            stalls[len(run)] = stalls.get(len(run), 0) + 1
    return {
        "parameters": dimension,
        "train_loss": train_curve[-1],
        "test_loss": test_curve[-1],
        "test_loss_minimum": test_curve.min(),
        "test_loss_minimum_step": int(test_curve.argmin()),
        "initial_train_loss": train_curve[0],
        "initial_test_loss": test_curve[0],
        "top_eigenvalues": top,
        "negative_eigenvalues": int(np.sum(exact < 0)),
        "min_eigenvalue": exact.min(),
        "min_abs_eigenvalue": magnitude.min(),
        "eigenvalues_abs_in_1e-5_1e-2": int(np.sum((magnitude >= 1e-5) & (magnitude < 1e-2))),
        "spectrum_every": SPECTRUM_EVERY,
        "initial_largest": descending[0, :K],
        "initial_min_eigenvalue": spectra[0, 0],
        "initial_negative_eigenvalues": int(np.sum(spectra[0] < 0)),
        "final_largest": descending[-1, :K],
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
