"""Draws the performance charts from the measurements in results/*.csv.

Writes a light and a dark version of each chart to docs/images/:
  performance-*.png   speed of the CPU and GPU programs on the 4096 x 4096 benchmark
  mpi-scaling-*.png   speedup of the CPU program with the number of MPI processes
  gpu-kernels-*.png   the GPU kernels side by side (once results with several kernels exist)

Usage: python3 scripts/make_charts.py
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from theme import PX, THEMES, style  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
IMAGES = ROOT / "docs" / "images"
BENCH_N, BENCH_K = 4096, 200  # the benchmark problem of the performance chart


def read(pattern):
    rows = []
    for path in sorted(RESULTS.glob(pattern)):
        with path.open(newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def best(rows):
    """The fastest of several runs of the same configuration."""
    return min(rows, key=lambda r: float(r["seconds"]))


# ---- Drawing helpers -------------------------------------------------------------------------------------


def rounded_hbar(ax, y, width, height, color, radius_px=4):
    """A horizontal bar from x = 0 with a rounded end at the data value and a square end at the baseline.
    Call after the axes limits and the figure layout are final."""
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    px_per_x, px_per_y = abs(x1 - x0), abs(y1 - y0)
    r = radius_px * 2 / px_per_x  # in x data units (2 image pixels per screen pixel)
    r = min(r, width / 2)
    ax.add_patch(
        FancyBboxPatch(
            (0, y - height / 2), width, height,
            boxstyle=f"round,pad=0,rounding_size={r}",
            mutation_aspect=px_per_x / px_per_y,
            facecolor=color, edgecolor="none", zorder=3,
        )
    )
    ax.add_patch(Rectangle((0, y - height / 2), width - r, height, facecolor=color, edgecolor="none", zorder=3))


def header(fig, t, title, subtitle, left):
    fig.text(left, 0.985, title, fontsize=13, fontweight="bold", color=t["primary"], va="top")
    fig.text(left, 0.985 - 0.30 / fig.get_figheight(), subtitle, fontsize=9.5, color=t["muted"], va="top")


def hbar_chart(theme, bars, series_names, title, subtitle, xlabel, path, references=()):
    """bars: (label, value, tip text, series index) from top to bottom.
    references: (x, label) vertical reference lines, drawn as muted hairlines with a direct label."""
    t = style(theme)
    row_in, bar_in = 0.55, 0.24  # bars are 24 screen pixels thick
    height = 1.45 + row_in * len(bars)
    fig = plt.figure(figsize=(8, height))
    left, right = 0.30, 0.90
    top = 1 - 0.8 / height
    bottom = 0.55 / height
    ax = fig.add_axes((left, bottom, right - left, top - bottom))

    vmax = max(max(v for _, v, _, _ in bars), max((x for x, _ in references), default=0))
    ax.set_xlim(0, vmax * 1.12)
    ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.set_yticks(range(len(bars)), [label for label, _, _, _ in bars])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    ax.tick_params(length=0, pad=6)
    ax.set_xlabel(xlabel, fontsize=9.5, labelpad=8)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_linewidth(1 * PX)
    ax.xaxis.grid(True, color=t["grid"], linewidth=1 * PX)
    ax.set_axisbelow(True)

    fig.canvas.draw()  # fix the layout before drawing the rounded bars in data coordinates
    bar_height = bar_in / row_in
    for i, (_, value, tip, s) in enumerate(bars):
        rounded_hbar(ax, i, value, bar_height, t["series"][s])
        ax.text(value + vmax * 0.012, i, tip, va="center", ha="left", fontsize=9.5, color=t["secondary"])
    for x, label in references:
        ax.axvline(x, color=t["muted"], linewidth=1 * PX, zorder=2)
        ax.text(x, -0.62, label, ha="center", va="bottom", fontsize=8.5, color=t["muted"])

    header(fig, t, title, subtitle, left=0.03)
    if len(series_names) > 1:
        handles = [Patch(facecolor=t["series"][i], label=name) for i, name in enumerate(series_names)]
        fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.97, 0.995), ncol=len(handles),
                   frameon=False, fontsize=9.5, handlelength=1.0, handleheight=1.0, labelcolor=t["secondary"])
    fig.savefig(path)
    plt.close(fig)


# ---- The charts ------------------------------------------------------------------------------------------


def one_process_baseline(cpu):
    """The fastest one-process run of the benchmark problem. With one process there is no neighbour, so both
    halo-exchange modes run exactly the same code and the best of both is the least noisy baseline."""
    return best([r for r in cpu if int(r["N"]) == BENCH_N and int(r["K"]) == BENCH_K and r["processes"] == "1"])


def performance_chart(cpu, gpu):
    """CPU (one process and the best MPI run) and every GPU configuration on the benchmark problem."""
    on_bench = lambda r: int(r["N"]) == BENCH_N and int(r["K"]) == BENCH_K  # noqa: E731
    cpu_runs = [r for r in cpu if on_bench(r) and r["overlap"] == "on"]
    if not cpu_runs:
        return False
    machine = cpu_runs[0]["machine"]
    single = one_process_baseline(cpu)
    fastest = best(cpu_runs)
    base = float(single["seconds"])

    bars = [
        ("CPU, 1 process", float(single["glups"]), "1.0×", 0),
        (f"CPU, {fastest['processes']} MPI processes", float(fastest["glups"]),
         f"{base / float(fastest['seconds']):.1f}×", 0),
    ]
    gpu_runs = [r for r in gpu if on_bench(r) and r["program"] == "heat_cuda"]
    device = gpu_runs[0]["device"] if gpu_runs else None
    for precision in ("double", "float"):
        for kernel in ("naive", "shared", "temporal"):
            runs = [r for r in gpu_runs if r["kernel"] == kernel and r["precision"] == precision]
            if runs:
                run = best(runs)
                label = f"GPU, {kernel} kernel" + (f" (S={run['tsteps']})" if kernel == "temporal" else "")
                label += f", {precision}"
                bars.append((label, float(run["glups"]), f"{base / float(run['seconds']):.1f}×", 1))

    names = [f"CPU: {machine}"] + ([f"GPU: {device}"] if device else [])
    subtitle = (f"{BENCH_N} × {BENCH_N} grid, {BENCH_K} time steps. Labels: speedup over one CPU process. "
                "CPU and GPU are different machines.")
    for theme in THEMES:
        hbar_chart(theme, bars, names, "Grid-point updates per second", subtitle,
                   "billion grid-point updates per second (higher is faster)",
                   IMAGES / f"performance-{theme}.png")
    return True


def gpu_kernels_chart(gpu):
    """All kernels and precisions on the largest grid that has them, against the memory-bandwidth limit of a
    kernel that reads and writes the grid once per step."""
    runs = [r for r in gpu if r["program"] == "heat_cuda"]
    if len({r["kernel"] for r in runs}) < 2:
        return False
    n = max(int(r["N"]) for r in runs)
    runs = [r for r in runs if int(r["N"]) == n]
    peak = float(runs[0]["peak_gbs"])
    bars = []
    for precision, series in (("double", 0), ("float", 1)):
        for kernel in ("naive", "shared", "temporal"):
            match = [r for r in runs if r["kernel"] == kernel and r["precision"] == precision]
            if match:
                run = best(match)
                label = f"{kernel}" + (f" (S={run['tsteps']})" if kernel == "temporal" else "") + f", {precision}"
                bars.append((label, float(run["glups"]), f"{float(run['glups']):.0f}", series))
    references = [(peak / 16, "double: memory limit"), (peak / 8, "float: memory limit")]
    for theme in THEMES:
        hbar_chart(theme, bars, ["double", "float"], f"GPU kernels on {runs[0]['device']}",
                   f"{n} × {n} grid. Lines: the fastest a kernel that reads and writes the grid once per step "
                   f"can go at the {peak:.0f} GB/s peak.",
                   "billion grid-point updates per second (higher is faster)",
                   IMAGES / f"gpu-kernels-{theme}.png", references=references)
    return True


def mpi_scaling_chart(cpu):
    """Speedup over one process for each process count: the best run with either halo-exchange mode."""
    runs = [r for r in cpu if int(r["N"]) == BENCH_N and int(r["K"]) == BENCH_K]
    if len({r["processes"] for r in runs}) < 2:
        return False
    machine = runs[0]["machine"]
    by_np = {}
    for r in runs:
        by_np.setdefault(int(r["processes"]), []).append(r)
    procs = sorted(by_np)
    base = float(best(by_np[1])["seconds"])
    speedup = [base / float(best(by_np[p])["seconds"]) for p in procs]

    for theme in THEMES:
        t = style(theme)
        fig = plt.figure(figsize=(8, 4.2))
        ax = fig.add_axes((0.08, 0.14, 0.88, 0.62))
        top = max(4.0, max(speedup) * 1.4)
        ax.set_xlim(0.5, procs[-1] + 0.5)
        ax.set_ylim(0, top)
        ax.plot([1, top], [1, top], color=t["muted"], linewidth=1 * PX, zorder=2)
        ax.text(top * 0.86 + 0.3, top * 0.86, "ideal", color=t["muted"], fontsize=9, ha="left", va="center")
        ax.plot(procs, speedup, color=t["series"][0], linewidth=2 * PX, solid_capstyle="round",
                solid_joinstyle="round", marker="o", markersize=8 * PX, markerfacecolor=t["series"][0],
                markeredgecolor=t["surface"], markeredgewidth=2 * PX, zorder=3)
        peak_i = max(range(len(procs)), key=lambda i: speedup[i])
        ax.annotate(f"{speedup[peak_i]:.2f}× with {procs[peak_i]} processes", (procs[peak_i], speedup[peak_i]),
                    xytext=(0, 12), textcoords="offset points", ha="center", fontsize=9.5, color=t["secondary"])
        ax.set_xticks(procs)
        ax.set_xlabel("MPI processes", fontsize=9.5)
        ax.set_ylabel("speedup over 1 process", fontsize=9.5)
        ax.tick_params(length=0, pad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_linewidth(1 * PX)
        ax.yaxis.grid(True, color=t["grid"], linewidth=1 * PX)
        ax.set_axisbelow(True)
        header(fig, t, f"CPU strong scaling on an {machine}",
               f"{BENCH_N} × {BENCH_N} grid, {BENCH_K} time steps, best of several runs.\n"
               "Memory bandwidth caps the speedup; past 4 processes the slower efficiency cores join in.",
               left=0.03)
        fig.savefig(IMAGES / f"mpi-scaling-{theme}.png")
        plt.close(fig)
    return True


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    cpu, gpu = read("cpu-*.csv"), read("gpu-*.csv")
    for name, made in (
        ("performance", performance_chart(cpu, gpu)),
        ("mpi-scaling", mpi_scaling_chart(cpu)),
        ("gpu-kernels", gpu_kernels_chart(gpu)),
    ):
        print(f"{name}: {'written' if made else 'skipped (not enough data yet)'}")


if __name__ == "__main__":
    main()
