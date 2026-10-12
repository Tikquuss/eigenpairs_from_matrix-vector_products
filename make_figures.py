"""Reproduce the figures of the blog post
"Eigenpairs from Matrix-Vector Products: Power Iteration, Deflation, and Lanczos".

    python make_figures.py                      # all figures, SVG in figures/output
    python make_figures.py --only lanczos       # one figure
    python make_figures.py --preview previews   # PNG copies as well

Each figure also writes a JSON file with the numbers quoted in its caption.
"""

import argparse
import importlib
import time
from pathlib import Path

from figures.common import style, write_summary

FIGURES = [
    "power_iteration",
    "complete_eigenbasis",
    "residual_certificate",
    "hessian_deflation",
    "lanczos_ritz",
    "lanczos_vs_deflation",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "figures" / "output",
                        help="directory for the SVG files and the JSON summaries")
    parser.add_argument("--preview", type=Path, default=None, help="directory for PNG copies")
    parser.add_argument("--only", nargs="*", default=None, choices=FIGURES + [
        module.replace("_", "-") for module in FIGURES], help="figures to make (default: all)")
    args = parser.parse_args()

    style()
    selected = FIGURES if not args.only else [name.replace("-", "_") for name in args.only]
    for module_name in selected:
        start = time.time()
        module = importlib.import_module(f"figures.{module_name}")
        summary = module.make(args.out, args.preview)
        write_summary(module.NAME, args.out, summary)
        print(f"{module.NAME}.svg written in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
