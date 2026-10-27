import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import FITNESS_HIGH_CONF_CSV  # noqa: E402

MUTATIONS = ["H244Y", "P203Q", "A72T", "G37C"]
ORDER = ["WT", "T241A", "S68P-T241A"]
COLOURS = {"WT": "#7F7F7F", "T241A": "#92C4D2", "S68P-T241A": "#F5A2A7"}

FIGURE_STEM = "background_mutation_frequency"


def load_variants():
    """Variant identifiers from the high-confidence fitness table, wild-type excluded."""
    with open(FITNESS_HIGH_CONF_CSV, encoding="utf-8-sig") as fh:
        return [row["mut_id"] for row in csv.DictReader(fh) if row["mut_id"] != "WT"]


def split_backgrounds(variants):
    """Assign each variant to the background of the last founder it carries."""
    mutations = {v: set(v.split("-")) for v in variants}
    return {
        "WT": [v for v in variants if not mutations[v] & {"T241A", "S68P"}],
        "T241A": [v for v in variants
                  if "T241A" in mutations[v] and "S68P" not in mutations[v]],
        "S68P-T241A": [v for v in variants if {"S68P", "T241A"} <= mutations[v]],
    }


def frequencies(backgrounds):
    counts, totals = {}, {}
    for label, group in backgrounds.items():
        totals[label] = len(group)
        counts[label] = [
            100 * sum(1 for v in group if m in set(v.split("-"))) / len(group)
            for m in MUTATIONS
        ]
    return counts, totals


def plot(freqs, totals):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    x = np.arange(len(MUTATIONS))
    width = 0.26

    for i, label in enumerate(ORDER):
        values = freqs[label]
        positions = x + (i - 1) * width
        ax.bar(positions, values, width * 0.92, color=COLOURS[label],
               label=f"{label} (n = {totals[label]})", zorder=3)
        # mark true zeros so an absent bar is not read as missing data
        for xi, value in zip(positions, values):
            if value == 0:
                ax.text(xi, 0.12, "0", ha="center", va="bottom",
                        fontsize=7, color=COLOURS[label], fontweight="bold")

    ax.set_ylabel("Mutation frequency [%]", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(MUTATIONS, fontsize=11)
    ax.set_ylim(0, 14)
    ax.set_yticks(np.arange(0, 15, 2))
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", length=0)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.1)

    legend = ax.legend(title="Background of:", frameon=False, fontsize=9,
                       loc="upper right", bbox_to_anchor=(1.02, 1.04),
                       handlelength=1.1, handleheight=1.1, labelspacing=0.45)
    legend.get_title().set_fontsize(9.5)
    legend.get_title().set_fontweight("bold")
    legend._legend_box.align = "left"

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(THIS_DIR / f"{FIGURE_STEM}.{ext}", dpi=400)
    plt.close(fig)


def main():
    variants = load_variants()
    backgrounds = split_backgrounds(variants)
    freqs, totals = frequencies(backgrounds)

    assigned = sum(totals.values())
    print(f"loaded {len(variants)} variants from {FITNESS_HIGH_CONF_CSV.name}")
    print(f"assigned to a background: {assigned}"
          f"  (carrying S68P without T241A, so outside all three: {len(variants) - assigned})")

    plot(freqs, totals)

    for label in ORDER:
        print(f"  {label:12s} n={totals[label]:3d} " +
              "  ".join(f"{m} {v:5.1f}%" for m, v in zip(MUTATIONS, freqs[label])))
    print(f"  wrote {FIGURE_STEM}.pdf and .png")


if __name__ == "__main__":
    main()
