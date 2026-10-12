"""Shared style and helpers for the figures of the post."""

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE = "#2563eb"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
GREEN = "#16a34a"
GREY = "#7d8783"
LIGHT_GREY = "#b9c2be"
INK = "#27322f"
BACKGROUND = "#fbfaf7"
DASHED = (0, (4, 3))
DOTTED = (0, (1.2, 2.2))


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": 11,
            "axes.edgecolor": "#56615d",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "xtick.color": "#44504c",
            "ytick.color": "#44504c",
            "grid.color": "#d6ddda",
            "grid.linewidth": 0.7,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.edgecolor": "#d6ddda",
            "legend.fontsize": 9.5,
            "svg.hashsalt": "matrix-vector-products",
        }
    )


def stage_colors(count: int):
    """Sequential colours for time-ordered overlays (first stage lightest)."""
    cmap = mpl.colormaps["viridis"]
    return [cmap(0.88 - 0.8 * i / max(count - 1, 1)) for i in range(count)]


def unit_gaussian(dimension: int, rng: np.random.Generator) -> np.ndarray:
    g = rng.standard_normal(dimension)
    return g / np.linalg.norm(g)


def quartiles(values: np.ndarray):
    """Median and interquartile range along the first axis."""
    return (
        np.median(values, axis=0),
        np.quantile(values, 0.25, axis=0),
        np.quantile(values, 0.75, axis=0),
    )


def band(axis, x, values, color, label, **line_kwargs):
    median, low, high = quartiles(values)
    axis.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    axis.plot(x, median, color=color, linewidth=2.0, label=label, **line_kwargs)


def finish(axes) -> None:
    for axis in np.atleast_1d(axes):
        axis.set_facecolor(BACKGROUND)
        axis.grid(True, which="major", alpha=0.8)


def save(fig, name: str, out_dir: Path, preview: Path = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.svg", format="svg", facecolor=fig.get_facecolor(),
                metadata={"Date": None}, dpi=200)  # dpi only affects rasterized artists
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}.png", format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(name: str, out_dir: Path, summary: dict) -> None:
    """Numbers quoted in the caption of the figure, for checking."""
    def plain(value):
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [plain(item) for item in value]
        if isinstance(value, (np.floating, float)):
            return float(f"{float(value):.6g}")
        if isinstance(value, np.integer):
            return int(value)
        return value

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(plain(summary), indent=2) + "\n", encoding="utf-8")
