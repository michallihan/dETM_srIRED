"""
Fitness replicate correlations.

Plots the three sequencing replicates of the low-confidence fitness dataset against each
other as a 3 x 3 grid: scatter plots below the diagonal, Pearson r, p and R-squared above
it, replicate names on the diagonal.

Each replicate's value is the mean of its lenient- and stringent-gate fitness for that
variant. Where only one gate survived QC, that gate's value is used.

Output, written next to this script:
    replicate_plot.svg
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import FITNESS_LOW_CONF_CSV  # noqa: E402

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable

REPLICATES = [1, 2, 3]
SAMPLE_NAMES = ["Rep. 1", "Rep. 2", "Rep. 3"]

POINT_COLOR = "#2a6f97"
POINT_SIZE = 18
POINT_ALPHA = 0.45
POINT_EDGECOLOR = "black"
POINT_LINEWIDTH = 0.25
SHOW_STATS = True
SHOW_BEST_FIT = True

XLIM = (-3.5, 6.5)
YLIM = (-3.5, 6.5)
XTICKS = [6, 3, 0, -3]
YTICKS = [6, 3, 0, -3]
AXIS_LABEL_SIZE_PCT = 2.0
TICK_LABEL_SIZE_PCT = 2.0

OUTPUT_SVG = THIS_DIR / "replicate_plot.svg"


def load_replicate_fitness():
    """One column per replicate: the mean of its lenient- and stringent-gate fitness."""
    data = pd.read_csv(FITNESS_LOW_CONF_CSV)
    out = pd.DataFrame(index=data.index)
    for n in REPLICATES:
        gates = [f"fitness{n}_uncorr_lenient", f"fitness{n}_uncorr_stringent"]
        missing = [c for c in gates if c not in data.columns]
        if missing:
            raise ValueError(f"Input file is missing required columns: {missing}")
        out[f"rep{n}"] = data[gates].mean(axis=1)
    return out


def pct_to_points(fig, pct):
    return fig.get_figheight() * 72 * (pct / 100.0)


def pct_to_pixels(fig, pct):
    return fig.get_figheight() * fig.dpi * (pct / 100.0)


def style_ticks(ax, fig):
    ax.tick_params(
        labelsize=pct_to_points(fig, TICK_LABEL_SIZE_PCT),
        length=pct_to_pixels(fig, TICK_LABEL_SIZE_PCT) * 0.35,
        width=ax.spines["bottom"].get_linewidth(),
    )


def stats_text(x, y):
    r, p = pearsonr(x, y)
    p_display = f"p = {p:.1e}" if p >= 1e-300 else "p < 1e-300"
    return f"r = {r:.2f}\n{p_display}\nR² = {r * r:.2f}"


def main():
    df = load_replicate_fitness()
    columns = list(df.columns)
    n = len(columns)
    print(f"loaded {len(df)} variants from {FITNESS_LOW_CONF_CSV.name}")

    fig, axes = plt.subplots(n, n, figsize=(9, 9))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    axis_label_size = pct_to_points(fig, AXIS_LABEL_SIZE_PCT)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i == j:  # diagonal: replicate name
                ax.text(0.5, 0.5, SAMPLE_NAMES[i], ha="center", va="center",
                        fontsize=22, fontweight="bold", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_facecolor("none")
                continue

            pair = df[[columns[j], columns[i]]].dropna()

            if i > j:  # lower triangle: scatter
                ax.scatter(pair[columns[j]], pair[columns[i]], s=POINT_SIZE,
                           c=POINT_COLOR, alpha=POINT_ALPHA,
                           edgecolors=POINT_EDGECOLOR, linewidths=POINT_LINEWIDTH)

                if SHOW_BEST_FIT and len(pair) >= 2:
                    slope, intercept = np.polyfit(pair[columns[j]].to_numpy(),
                                                  pair[columns[i]].to_numpy(), 1)
                    x_fit = np.array([XLIM[0], XLIM[1]])
                    ax.plot(x_fit, slope * x_fit + intercept,
                            color="black", linestyle=":", linewidth=1.5)

                ax.set_xlim(*XLIM)
                ax.set_ylim(*YLIM)
                ax.set_xticks(XTICKS)
                ax.set_yticks(YTICKS)
                style_ticks(ax, fig)

                if i < n - 1:
                    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                else:
                    ax.set_xlabel(SAMPLE_NAMES[j], fontsize=axis_label_size)

                if j > 0:
                    ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                else:
                    ax.set_ylabel(SAMPLE_NAMES[i], fontsize=axis_label_size)
                continue

            # upper triangle: statistics
            if SHOW_STATS:
                ax.text(0.5, 0.5, stats_text(pair[columns[j]], pair[columns[i]]),
                        ha="center", va="center", fontsize=12, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(OUTPUT_SVG, format="svg")
    plt.close(fig)
    print(f"  wrote {OUTPUT_SVG.name}")


if __name__ == "__main__":
    main()
