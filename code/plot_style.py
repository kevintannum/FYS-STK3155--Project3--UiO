"""Shared figure style for the Project 3 report.

One place to set colours, fonts and axis chrome, so every PNG that goes into the
report looks like it came from the same study.

Colour policy
-------------
* **Categorical** (identity: which activation, which regulariser) -- a fixed
  hue order, assigned by slot and never cycled. The first three slots are
  verified colourblind-safe against each other for *all* pairs, so comparisons
  stay readable in deuteranopia/tritanopia and in greyscale print.
* **Sequential** (magnitude: a heat-map of validation F2) -- a single blue hue
  running light to dark. Never a rainbow: rainbow maps invent boundaries that
  are not in the data.
* **Diverging** (polarity: residuals above/below zero) -- blue to red through a
  neutral grey midpoint.
* Colour is only ever used when it *encodes* something. A bar chart whose
  categories are already named on the x-axis gets a single colour.

Because some of these hues sit below 3:1 contrast against a white page, every
categorical chart also carries a legend and direct value labels, and every
figure has a matching CSV in ``results/`` -- identity is never colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Categorical slots (fixed order) ---------------------------------------
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
]

# --- Status (reserved; never reused as "series 4") -------------------------
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# --- Chart chrome and ink --------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"  # white: these figures are printed on paper, not a screen

# --- Sequential (single hue, light -> dark) --------------------------------
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", _BLUE_RAMP)
SEQ_CMAP_R = SEQ_CMAP.reversed()

# --- Diverging (two poles + neutral grey midpoint) -------------------------
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red",
    ["#0d366b", "#3987e5", "#9ec5f4", "#f0efec", "#f0a0a0", "#e34948", "#8f2020"],
)


def apply_style() -> None:
    """Install the report style into matplotlib's rcParams."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10.5,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            # Recessive chrome: the data should be the darkest thing on the page.
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 2.0,     # 2px lines
            "lines.markersize": 5.0,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "figure.constrained_layout.use": True,
        }
    )


def grid_y_only(ax) -> None:
    """Horizontal gridlines only -- vertical ones fight with grouped bars."""
    ax.grid(axis="y", which="major")
    ax.grid(axis="x", visible=False)


def grid_x_only(ax) -> None:
    ax.grid(axis="x", which="major")
    ax.grid(axis="y", visible=False)


def bar_labels(ax, bars, fmt="{:.3f}", fontsize=8.5, color=INK_SECONDARY, padding=2) -> None:
    """Direct value labels on bars.

    Mandatory on any categorical chart here: it is the secondary encoding that
    keeps the comparison readable when two hues are hard to tell apart, and it
    removes the need to read values off a gridline.
    """
    ax.bar_label(bars, fmt=fmt.format if callable(fmt) else fmt,
                 fontsize=fontsize, color=color, padding=padding)


def heatmap(ax, values, xticklabels, yticklabels, *, cmap=SEQ_CMAP, fmt="{:.3f}",
            cbar_label=None, annotate=True, vmin=None, vmax=None, errors=None):
    """Annotated heat map with readable in-cell labels.

    Label ink flips to white on the dark end of the ramp so the number stays
    legible at both extremes.
    """
    im = ax.imshow(values, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xticklabels)), xticklabels)
    ax.set_yticks(range(len(yticklabels)), yticklabels)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    if annotate:
        norm = im.norm
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if v != v:  # NaN
                    continue
                shade = norm(v)
                txt = fmt.format(v)
                if errors is not None:
                    txt += f"\n±{errors[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if shade > 0.55 else INK)

    if cbar_label:
        cb = ax.figure.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(cbar_label, color=INK_SECONDARY, fontsize=10)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=9, colors=INK_MUTED)
    return im


def save_fig(fig, name: str, results_dir: Path | str) -> Path:
    """Save a figure as PNG into ``results_dir/figures`` and report the path."""
    figures = Path(results_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    path = figures / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path)
    print(f"  saved {path}")
    return path


def save_table(df, name: str, results_dir: Path | str) -> Path:
    """Save a results table as CSV into ``results_dir/tables``.

    Every figure in the report has a CSV twin, which doubles as the accessible
    table view of the chart.
    """
    tables = Path(results_dir) / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    path = tables / (name if name.endswith(".csv") else f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  saved {path}")
    return path
