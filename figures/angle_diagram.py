"""angle-to-eigenspace.svg: the right triangle behind tan(phi_0) in Theorem 1.

The starting vector v_0 splits into its projection p onto the eigenspace E and
the remainder v_0 - p, orthogonal to E. The colors follow the proof of
Proposition 1: the component along the dominant eigenspace in blue, the
remainder in orange.
"""

import matplotlib.pyplot as plt
import numpy as np

from .common import BLUE, GREY, INK, ORANGE, save

NAME = "angle-to-eigenspace"
PHI = np.deg2rad(33.0)


def arrow(axis, start, end, color, width):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops=dict(arrowstyle="-|>", color=color, linewidth=width,
                                  shrinkA=0, shrinkB=0, mutation_scale=13))


def make(out_dir, preview=None) -> dict:
    v0 = np.array([np.cos(PHI), np.sin(PHI)])
    p = np.array([v0[0], 0.0])

    fig, axis = plt.subplots(figsize=(3.4, 1.86))
    fig.patch.set_alpha(0.0)
    axis.set_facecolor("none")
    axis.set_aspect("equal")
    axis.axis("off")

    # The eigenspace, drawn as a line through the origin.
    axis.plot([-0.12, 1.22], [0, 0], color=GREY, linewidth=1.1, zorder=1)
    axis.text(1.24, 0.0, r"$E_{\lambda_\star}$", color=GREY, fontsize=12, va="center", ha="left")
    axis.plot([0], [0], marker="o", markersize=3.2, color=INK, zorder=4)

    # Right angle at the foot of the perpendicular, and the angle phi_0 at the origin.
    s = 0.055
    axis.plot([p[0] - s, p[0] - s, p[0]], [0, s, s], color=GREY, linewidth=0.9, zorder=2)
    t = np.linspace(0, PHI, 40)
    axis.plot(0.24 * np.cos(t), 0.24 * np.sin(t), color=INK, linewidth=1.0, zorder=2)
    axis.text(0.31 * np.cos(PHI / 2), 0.31 * np.sin(PHI / 2), r"$\phi_0$", color=INK, fontsize=12,
              va="center", ha="left")

    # The three sides.
    arrow(axis, (0, 0), p, BLUE, 2.4)
    arrow(axis, p, v0, ORANGE, 2.0)
    arrow(axis, (0, 0), v0, INK, 2.0)
    axis.text(0.36, 0.35, r"$\mathbf{v}_0$", color=INK, fontsize=13, ha="right", va="bottom")
    axis.text(p[0] / 2, -0.07, r"$\mathbf{p}_{\lambda_\star}$", color=BLUE, fontsize=13, ha="center", va="top")
    axis.text(p[0] + 0.04, v0[1] / 2, r"$\mathbf{v}_0-\mathbf{p}_{\lambda_\star}$", color=ORANGE, fontsize=13,
              ha="left", va="center")

    axis.set_xlim(-0.14, 1.5)
    axis.set_ylim(-0.24, 0.66)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    save(fig, NAME, out_dir, preview)
    return {"phi_0_degrees": float(np.rad2deg(PHI))}
