import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import FITNESS_HIGH_CONF_CSV  # noqa: E402
from variant_rounds import ROUNDS, variants_per_round  # noqa: E402

PERCENTILE = 85
MEAN_COLOUR = "darkblue"
PERCENTILE_COLOUR = "darkgreen"

# Left out of the mutation-count panel: 0 is the wild type, and 7 holds a single variant.
# Both were left out of the published panel; they are reported below rather than dropped
# silently.
EXCLUDED_MUTATION_NUMBERS = (0, 7)

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable


def load_rounds():
    """Every round's variants, in round order, one table per round."""
    rounds = variants_per_round()
    return [table.drop(columns="round_label") for _, table in rounds.groupby("round")]


def fitness_per_round(rounds):
    """Scored variants present in each round, one row per variant per round."""
    fitness = pd.read_csv(FITNESS_HIGH_CONF_CSV)[["mut_id", "fitness_final", "sigma_final"]]
    scored = pd.concat(rounds, ignore_index=True).merge(fitness, on="mut_id", how="left")
    scored = scored[np.isfinite(scored["fitness_final"])]
    return scored[["round", "mut_id", "fitness_final", "sigma_final"]].reset_index(drop=True)


def plot_fitness(scored):
    """Swarm of fitness score against round, with each round's mean and percentile."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.swarmplot(x="round", y="fitness_final", data=scored, color="black", size=3,
                  order=list(ROUNDS), ax=ax)

    summary = []
    for position, round_number in enumerate(ROUNDS):
        values = scored.loc[scored["round"] == round_number, "fitness_final"]
        mean = round(float(np.mean(values)), 1)
        percentile = round(float(np.percentile(values, PERCENTILE)), 1)
        summary.append((round_number, len(values), mean, percentile))

        span = ((position + 0.2) / len(ROUNDS), (position + 1) / len(ROUNDS))
        for value, colour in ((mean, MEAN_COLOUR), (percentile, PERCENTILE_COLOUR)):
            ax.axhline(value, color=colour, linestyle="--", linewidth=1,
                       xmin=span[0], xmax=span[1])
            ax.text(position + 0.35, value, f"{value:.1f}", color=colour,
                    ha="center", va="bottom")
        ax.text(position, ax.get_ylim()[0] + 0.1, f"n={len(values)}",
                color="black", ha="center", va="bottom")

    handles = [plt.Line2D([], [], color=MEAN_COLOUR, linestyle="--", label="Mean"),
               plt.Line2D([], [], color=PERCENTILE_COLOUR, linestyle="--",
                          label=f"Upper {PERCENTILE}th percentile")]
    ax.legend(handles=handles, loc="upper left", fontsize=10)

    ax.set_xticks(range(len(ROUNDS)))
    ax.set_xticklabels([str(r) for r in ROUNDS])
    ax.set_xlabel("Round number")
    ax.set_ylabel("Fitness score")

    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(THIS_DIR / f"fitness_score_by_round.{extension}", dpi=400)
    plt.close(fig)
    return summary


def first_appearance(rounds):
    """Each distinct variant counted once, in the round it is first seen."""
    seen, counts = set(), []
    for table in rounds:
        fresh = table[~table["mut_id"].isin(seen)]
        seen.update(fresh["mut_id"])
        tally = fresh["Nham_aa"].value_counts().rename_axis("mutation_number").reset_index(
            name="variants")
        tally["round"] = table["round"].iloc[0]
        counts.append(tally)
    return pd.concat(counts, ignore_index=True)


def plot_mutation_numbers(counts):
    """Stacked bars: variants by mutation count, coloured by the round they first appear in."""
    shown = counts[~counts["mutation_number"].isin(EXCLUDED_MUTATION_NUMBERS)]
    colours = plt.get_cmap("viridis")(np.linspace(0, 1, len(ROUNDS)))

    fig, ax = plt.subplots(figsize=(4, 4))
    bottoms = {n: 0.0 for n in shown["mutation_number"].unique()}
    for colour, round_number in zip(colours, ROUNDS):
        subset = shown[shown["round"] == round_number]
        ax.bar(subset["mutation_number"], subset["variants"], color=colour,
               label=f"r{round_number}",
               bottom=[bottoms[n] for n in subset["mutation_number"]])
        for n, v in zip(subset["mutation_number"], subset["variants"]):
            bottoms[n] += v

    ax.set_xlabel("Mutation number")
    ax.set_ylabel("Count of variants per round")
    ax.legend(title="Round")

    fig.tight_layout()
    for extension in ("svg", "png"):
        fig.savefig(THIS_DIR / f"mutation_number_by_round.{extension}", dpi=400)
    plt.close(fig)


def main():
    rounds = load_rounds()
    for table in rounds:
        print(f"round {table['round'].iloc[0]}: {len(table)} distinct amino-acid variants")

    scored = fitness_per_round(rounds)
    scored.to_csv(THIS_DIR / "fitness_per_round.csv", index=False)
    print(f"\nwrote fitness_per_round.csv: {len(scored)} scored variant-round pairs")

    print(f"\nfitness_score_by_round: mean and upper {PERCENTILE}th percentile")
    for round_number, n, mean, percentile in plot_fitness(scored):
        print(f"  round {round_number}: n={n:4d}   mean {mean:4.1f}   "
              f"p{PERCENTILE} {percentile:4.1f}")

    counts = first_appearance(rounds)
    plot_mutation_numbers(counts)
    totals = counts.groupby("mutation_number")["variants"].sum()
    print(f"\nmutation_number_by_round: {int(totals.sum())} distinct variants across the campaign")
    for number, total in totals.items():
        note = "  (left out of the panel)" if number in EXCLUDED_MUTATION_NUMBERS else ""
        print(f"  {int(number)} mutations: {int(total):4d}{note}")


if __name__ == "__main__":
    raise SystemExit(main())
