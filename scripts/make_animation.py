"""Animates the heat spreading from the hot top edge, using the project's own CPU program for every frame.

Writes docs/images/heat.gif and docs/images/social-preview.png (the image GitHub shows when the repository is
shared: Settings -> General -> Social preview). Needs the CPU program built:
    cmake --preset release && cmake --build --preset release

Usage: python3 scripts/make_animation.py [path/to/heat]      (default: build/release/heat)
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from theme import HEAT_COLORMAP, THEMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "docs" / "images"
N = 160            # grid size
LAST_STEP = 12000  # by then the heat has reached the whole grid
FRAMES = 56
SIZE = 400         # pixels


def solve(heat, steps, workdir):
    """The grid after `steps` time steps, computed by the CPU program."""
    csv = Path(workdir) / "grid.csv"
    command = [str(heat), str(steps), str(N), "--csv", str(csv)]
    mpirun = shutil.which("mpirun")
    if mpirun:
        command = [mpirun, "-np", str(min(4, os.cpu_count() or 1)), *command]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    return np.loadtxt(csv, delimiter=",")


def font(size):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def frame(grid, step):
    colors = matplotlib.colormaps[HEAT_COLORMAP](np.clip(grid / 100.0, 0, 1))
    image = Image.fromarray((colors[:, :, :3] * 255).astype(np.uint8)).resize((SIZE, SIZE), Image.BILINEAR)
    draw = ImageDraw.Draw(image)
    text = f"time step {step:,}"
    for dx, dy in ((1, 1), (0, 0)):  # a soft shadow keeps the label readable on the bright top rows
        draw.text((14 + dx, SIZE - 30 + dy), text, font=font(15), fill=(0, 0, 0) if dx else (255, 255, 255))
    return image


def social_preview(grid):
    t = THEMES["dark"]
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100, facecolor=t["surface"])
    ax = fig.add_axes((0.62, 0.1, 0.34, 0.8))
    ax.imshow(grid, cmap=HEAT_COLORMAP, vmin=0, vmax=100, interpolation="bilinear")
    ax.set_axis_off()
    lines = [
        (0.70, "heat-equation-mpi", 40, "bold", t["primary"]),
        (0.58, "The 2D heat equation in C++17", 22, "normal", t["primary"]),
        (0.42, "MPI on CPUs · CUDA on one GPU · MPI + CUDA on several", 16, "normal", t["secondary"]),
        (0.34, "Temporal-blocking kernels, overlapped halo exchange", 16, "normal", t["secondary"]),
        (0.26, "Tested in CI without a GPU, on a CPU emulator of CUDA", 16, "normal", t["secondary"]),
    ]
    for y, text, size, weight, color in lines:
        fig.text(0.06, y, text, fontsize=size, fontweight=weight, color=color, va="center")
    fig.savefig(IMAGES / "social-preview.png", facecolor=t["surface"])
    plt.close(fig)


def main():
    heat = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "build" / "release" / "heat"
    if not heat.exists():
        sys.exit(f"{heat} not found: build the CPU program first (cmake --preset release && "
                 "cmake --build --preset release)")
    IMAGES.mkdir(parents=True, exist_ok=True)
    # Quadratic spacing in time: diffusion spreads like sqrt(t), so the front moves at a steady pace.
    steps = sorted({round(LAST_STEP * (f / (FRAMES - 1)) ** 2) for f in range(FRAMES)})
    with tempfile.TemporaryDirectory() as workdir:
        grids = [solve(heat, s, workdir) for s in steps]

    images = [frame(g, s).convert("P", palette=Image.ADAPTIVE, colors=128) for g, s in zip(grids, steps)]
    durations = [70] * (len(images) - 1) + [2000]  # hold the last frame
    images[0].save(IMAGES / "heat.gif", save_all=True, append_images=images[1:], duration=durations, loop=0,
                   optimize=True)
    social_preview(grids[-1])
    print(f"Wrote {IMAGES / 'heat.gif'} ({len(images)} frames, "
          f"{(IMAGES / 'heat.gif').stat().st_size // 1024} KB) and {IMAGES / 'social-preview.png'}")


if __name__ == "__main__":
    main()
