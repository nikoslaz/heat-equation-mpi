"""Colours and sizes shared by the figure scripts (make_charts.py, make_animation.py, plot_grid.py).

Every chart is drawn twice, for GitHub's light and dark themes; the README picks the right one with <picture>.
The two series colours were checked for colour-blind separation and contrast in both themes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834"],  # blue, orange
    },
    "dark": {
        "surface": "#1a1a19",
        "primary": "#f0efec",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": ["#3987e5", "#d95926"],
    },
}

DPI = 200
# The images are shown at about half their pixel size, so one "screen pixel" is two image pixels.
PX = 2 * 72 / DPI  # points per screen pixel

HEAT_COLORMAP = "inferno"  # temperature: a "heat" colour scale, always shown with a scale or a caption


def style(theme):
    """Matplotlib settings for a theme."""
    t = THEMES[theme]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 10,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "figure.facecolor": t["surface"],
            "axes.facecolor": t["surface"],
            "savefig.facecolor": t["surface"],
            "text.color": t["primary"],
            "axes.labelcolor": t["secondary"],
            "xtick.color": t["muted"],
            "ytick.color": t["secondary"],
            "xtick.labelcolor": t["muted"],
            "ytick.labelcolor": t["secondary"],
            "axes.edgecolor": t["axis"],
        }
    )
    return t
