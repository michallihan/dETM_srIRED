"""
How the pool improved round by round.

Bars give the share of each round's scored variants that beat the wild type, judged
conservatively: a variant counts as improved only if the bottom of its fitness interval
clears the top of the wild type's. The line over them is the median kcat(app) of that round's
ten best variants, which is what turns a share into a magnitude -- the pool does not only
become more often better, it becomes better by more.

Below the axis sits the table the panel carries: how many variants were screened each round,
how many distinct variants the nanopore sequencing recovered from the sorted pool, and the
average number of substitutions the scored ones had come to hold. The screened and recovered
counts describe the whole round; the bars and the line describe the part of it that has a
fitness score, which is smaller.

kcat(app) comes from the conversion fitted in 01_fitness_score_processing/03_deriving_kcatapp,
read from its kcat_app_fit.json rather than restated here.

Outputs, written next to this script:
    campaign_progression.csv
    top_variants_by_round.csv
    campaign_progression.svg / .png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import (  # noqa: E402
    COLLAPSED_UNIQUE_VARIANTS_CSV,
    FITNESS_HIGH_CONF_CSV,
    KCAT_APP_FIT_JSON,
)
from variant_rounds import ROUNDS, variants_per_round  # noqa: E402

TOP_N = 10  # variants per round behind the median kcat(app) line

# Library size screened in each round, from the campaign records. Not derivable from the
# sequencing, which only sees what came back.
SCREENED_PER_ROUND = ("60K", "250K", "220K", "230K", "190K")

IMPROVED_COLOUR = "#9fcfc4"
NEUTRAL_COLOUR = "#cfcfcf"
BAR_AXIS_COLOUR = "#007f73"
LINE_COLOUR = "#df1f25"

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable


def kcat_app_conversion():
    fit = json.loads(KCAT_APP_FIT_JSON.read_text())
    return lambda fitness: fit["a"] * np.exp(fit["b"] * fitness)


def scored_per_round():
    """Every scored variant, once per round it appears in, with its fitness and kcat(app)."""
    fitness = pd.read_csv(FITNESS_HIGH_CONF_CSV)[["mut_id", "fitness_final", "sigma_final"]]
    rounds = variants_per_round()[["round", "mut_id"]]
    scored = rounds.merge(fitness, on="mut_id", how="inner")
    scored["kcat_app"] = kcat_app_conversion()(scored["fitness_final"])
    return scored


def recovered_per_round():
    """Distinct variants the nanopore sequencing saw in each round, scored or not."""
    collapsed = pd.read_csv(COLLAPSED_UNIQUE_VARIANTS_CSV)
    return {r: int((collapsed[f"round_{r}"] > 0).sum()) for r in ROUNDS}


def progression(scored):
    """Per round: the share of variants beating the wild type, and how mutated they are."""
    wild_type = scored[scored["mut_id"] == "WT"].iloc[0]
    beaten = wild_type["fitness_final"] + wild_type["sigma_final"]
    tenfold = beaten + np.log(10.0)

    recovered = recovered_per_round()
    rows = []
    for round_number, group in scored.groupby("round"):
        lower_bound = group["fitness_final"] - group["sigma_final"]
        improved = lower_bound > beaten
        mutations = group["mut_id"].map(lambda m: 0 if m == "WT" else len(m.split("-")))
        rows.append({
            "round": int(round_number),
            "n_variants_recovered": recovered[int(round_number)],
            "n_variants_scored": len(group),
            "mean_mutations": mutations.mean(),
            "improved_percent": improved.mean() * 100,
            "not_improved_percent": (~improved).mean() * 100,
            "n_variants_10x_wild_type": int((lower_bound > tenfold).sum()),
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def top_variants(scored):
    """Per round: the best variants by kcat(app), and the median across them."""
    rows = []
    for round_number, group in scored.groupby("round"):
        best = group.sort_values("kcat_app", ascending=False).head(TOP_N)
        rows.append({
            "round": int(round_number),
            "top_n": len(best),
            "top_median_kcat_app": best["kcat_app"].median(),
            "top_min_kcat_app": best["kcat_app"].min(),
            "top_max_kcat_app": best["kcat_app"].max(),
            "best_variant": best.iloc[0]["mut_id"],
            "best_variant_kcat_app": best.iloc[0]["kcat_app"],
        })
    return pd.DataFrame(rows).sort_values("round").reset_index(drop=True)


def plot(summary, best):
    x = summary["round"].to_numpy()
    improved = summary["improved_percent"].to_numpy()

    figure, ax = plt.subplots(figsize=(4.5, 4.0))
    figure.subplots_adjust(left=0.24, right=0.84, top=0.90, bottom=0.28)

    ax.bar(x, improved, color=IMPROVED_COLOUR, width=0.82, label="Improved variants")
    ax.bar(x, summary["not_improved_percent"], bottom=improved, color=NEUTRAL_COLOUR,
           width=0.82, label="Neutral & deteriorated variants")
    ax.set_ylabel("Share of variants [%]", color=BAR_AXIS_COLOUR, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)

    line_axis = ax.twinx()
    line_axis.plot(x, best["top_median_kcat_app"], color=LINE_COLOUR, marker="o",
                   linewidth=1.2, markersize=3.2, label=f"Top {TOP_N} median kcat(app)")
    line_axis.set_ylabel(f"Top {TOP_N} median\nkcat(app) [s$^{{-1}}$]", color=LINE_COLOUR,
                         fontsize=9)
    line_axis.tick_params(axis="y", labelsize=8, colors="black")
    line_axis.set_ylim(0, 100)
    line_axis.set_yticks(range(0, 101, 20))
    line_axis.spines["top"].set_visible(False)

    bar_handles, bar_labels = ax.get_legend_handles_labels()
    line_handles, line_labels = line_axis.get_legend_handles_labels()
    ax.legend(bar_handles + line_handles, bar_labels + line_labels, loc="upper left",
              fontsize=7.2, frameon=True, borderpad=0.25, handlelength=1.4)

    ax.add_patch(Rectangle((-0.24, -0.39), 1.24, 0.30, transform=ax.transAxes,
                           facecolor="#eeeeee", edgecolor="none", clip_on=False, zorder=-1))
    ax.text(-0.03, -0.09, "Evolution round", transform=ax.transAxes, ha="right", va="center",
            fontsize=7.2, clip_on=False)

    data_to_axes = blended_transform_factory(ax.transData, ax.transAxes)
    table = (
        ("Screened variants", list(SCREENED_PER_ROUND), -0.18),
        ("Selected variants", [str(n) for n in summary["n_variants_recovered"]], -0.26),
        ("Mutation average", [f"{v:.1f}" for v in summary["mean_mutations"]], -0.34),
    )
    for label, values, height in table:
        ax.text(-0.03, height, label, transform=ax.transAxes, ha="right", va="center",
                fontsize=7.2, clip_on=False)
        for position, value in zip(x, values):
            ax.text(position, height, value, transform=data_to_axes, ha="center", va="center",
                    fontsize=8.0, clip_on=False)

    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"campaign_progression.{extension}", dpi=400)
    plt.close(figure)


def main():
    if len(SCREENED_PER_ROUND) != len(ROUNDS):
        raise ValueError("SCREENED_PER_ROUND does not cover every round")

    scored = scored_per_round()
    summary = progression(scored)
    best = top_variants(scored)

    summary.to_csv(THIS_DIR / "campaign_progression.csv", index=False)
    best.to_csv(THIS_DIR / "top_variants_by_round.csv", index=False)
    plot(summary, best)

    print(summary.to_string(index=False))
    print(best[["round", "top_median_kcat_app", "best_variant", "best_variant_kcat_app"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
