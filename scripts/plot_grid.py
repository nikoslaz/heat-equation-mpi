"""Draws a grid written by one of the programs (grid.csv) as an image with a temperature scale.

Usage: python3 scripts/plot_grid.py [grid.csv] [output.png] [--theme light|dark]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from theme import HEAT_COLORMAP, PX, style  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid", nargs="?", default="grid.csv", type=Path)
    parser.add_argument("output", nargs="?", type=Path, help="default: the grid file with a .png suffix")
    parser.add_argument("--theme", choices=["light", "dark"], default="light")
    parser.add_argument("--title", help="default: the grid size")
    args = parser.parse_args()

    grid = np.loadtxt(args.grid, delimiter=",")
    t = style(args.theme)
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_axes((0.05, 0.05, 0.72, 0.78))
    image = ax.imshow(grid, cmap=HEAT_COLORMAP, vmin=0, vmax=100, interpolation="bilinear")
    ax.set_axis_off()
    bar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.04)
    bar.set_label("temperature", color=t["secondary"])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=0, colors=t["muted"])
    title = args.title or f"{grid.shape[0]} × {grid.shape[1]} grid, top edge held at 100"
    fig.text(0.05, 0.95, title, fontsize=11, fontweight="bold", color=t["primary"], va="top")
    for spine in ax.spines.values():
        spine.set_linewidth(1 * PX)

    output = args.output or args.grid.with_suffix(".png")
    fig.savefig(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
