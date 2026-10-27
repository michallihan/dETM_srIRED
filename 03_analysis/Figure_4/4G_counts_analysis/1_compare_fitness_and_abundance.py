import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import DIMSUM_ROUND_COUNTS_TSV, FITNESS_HIGH_CONF_CSV  # noqa: E402

ROUNDS = tuple(sorted(DIMSUM_ROUND_COUNTS_TSV))
PANEL_ROUNDS = (3, 5)

PANEL_VARIANTS = ("T241A", "S68P-T241A", "S68P-P203Q-T241A-H244Y")
ADDITIONAL_VARIANTS = ("S68P-P203Q-T241A", "S68P-T241A-H244Y")
REPORTED = PANEL_VARIANTS + ADDITIONAL_VARIANTS

# Rows DiMSum flags that the population-dynamics analysis drops: column -> value removed.
DISCARD = {"STOP": True, "STOP_readthrough": True, "barcode_valid": False, "indel": True}

FITNESS_COLOUR = "#CACACA"
ABUNDANCE_COLOUR = "#ABCDC2"
FITNESS_LABEL_COLOUR = "#7E7E7E"
ABUNDANCE_LABEL_COLOUR = "#679888"

# Fixed on both panels so the rounds can be read against each other.
FITNESS_LIMIT = 5
FITNESS_TICKS = np.arange(0, 6, 1)
ABUNDANCE_LIMIT = 16
ABUNDANCE_TICKS = np.arange(0, 17, 4)


def mutation_ids(sequences, wild_type):
    """Name each sequence by its differences from the wild type, N-terminal end first."""
    names = []
    for sequence in sequences:
        if len(sequence) != len(wild_type):
            names.append(None)
            continue
        differences = [f"{wild_type[i]}{i + 1}{sequence[i]}"
                       for i in range(len(wild_type)) if wild_type[i] != sequence[i]]
        names.append("WT" if not differences else "-".join(differences))
    return names


def round_abundance(round_number):
    """Each variant's share of one round's input pool, keyed by mutation id."""
    table = pd.read_csv(DIMSUM_ROUND_COUNTS_TSV[round_number], sep="\t")
    for column, discarded in DISCARD.items():
        if column in table.columns:
            table = table[table[column] != discarded]

    inputs = [c for c in table.columns if c.startswith("input")]
    collapsed = table.groupby("aa_seq", as_index=False).agg(
        {**{c: "sum" for c in inputs}, "WT": "first"}
    )

    fractions = collapsed[inputs] / collapsed[inputs].sum()
    collapsed["abundance"] = fractions.mean(axis=1)

    wild_type = collapsed.loc[collapsed["WT"] == True, "aa_seq"]  # noqa: E712
    if len(wild_type) != 1:
        raise ValueError(f"round {round_number} has {len(wild_type)} wild-type rows, expected 1")
    collapsed["mut_id"] = mutation_ids(collapsed["aa_seq"], wild_type.iloc[0])
    return dict(zip(collapsed["mut_id"], collapsed["abundance"]))


def build_table():
    """One row per reported variant per round, with its fitness score and abundance."""
    fitness = pd.read_csv(FITNESS_HIGH_CONF_CSV)
    scores = dict(zip(fitness["mut_id"], fitness["fitness_final"]))

    abundance = {r: round_abundance(r) for r in ROUNDS}

    rows = []
    for variant in REPORTED:
        for round_number in ROUNDS:
            share = float(abundance[round_number].get(variant, 0.0))
            rows.append({
                "mut_id": variant,
                "in_panels": variant in PANEL_VARIANTS,
                "round": round_number,
                "fitness_final": scores.get(variant, np.nan),
                "abundance_fraction": share,
                "abundance_percent": share * 100,
            })
    return pd.DataFrame(rows)


def wrap_label(mut_id):
    """Break a mutation id after each hyphen, as the published panels do."""
    parts = mut_id.split("-")
    return "\n".join(p + "-" for p in parts[:-1]) + ("\n" if len(parts) > 1 else "") + parts[-1]


def plot_round(table, round_number):
    """One panel: grey fitness bars on the left axis, green abundance bars on the right."""
    panel = table[(table["round"] == round_number)
                  & table["in_panels"]
                  & (table["abundance_fraction"] > 0)]
    shown = set(panel["mut_id"])
    panel = panel.set_index("mut_id").loc[[v for v in PANEL_VARIANTS if v in shown]]

    # Arial to match the published panels, with fallbacks so this still runs where it
    # is not installed.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11,
        "svg.fonttype": "none",
    })
    positions = np.arange(len(panel))
    width = 0.42

    fig, fitness_axis = plt.subplots(figsize=(1.25 * len(panel) + 1.5, 3.4))
    abundance_axis = fitness_axis.twinx()

    # Above the spines, which matplotlib draws at zorder 2.5. The bottom spine is wide
    # enough to cover the first few pixels of a bar, which costs a tall bar nothing and
    # hides most of a very short one — S68P-P203Q-T241A-H244Y in round 5 is 0.12%, six
    # pixels at print resolution, and four of them sit under the axis line.
    fitness_axis.bar(positions - width / 2, panel["fitness_final"], width,
                     color=FITNESS_COLOUR, zorder=3)
    abundance_axis.bar(positions + width / 2, panel["abundance_percent"], width,
                       color=ABUNDANCE_COLOUR, zorder=3)

    fitness_axis.set_ylabel("Fitness score", color=FITNESS_LABEL_COLOUR,
                            fontweight="bold", fontsize=14)
    fitness_axis.set_ylim(0, FITNESS_LIMIT)
    fitness_axis.set_yticks(FITNESS_TICKS)

    # Both axis labels read bottom to top, as in the published panels.
    abundance_axis.set_ylabel("Relative abundance [%]", color=ABUNDANCE_LABEL_COLOUR,
                              fontweight="bold", fontsize=14, labelpad=10)
    abundance_axis.set_ylim(0, ABUNDANCE_LIMIT)
    abundance_axis.set_yticks(ABUNDANCE_TICKS)

    fitness_axis.set_xticks(positions)
    fitness_axis.set_xticklabels([wrap_label(v) for v in panel.index],
                                 fontsize=11, fontweight="bold")
    fitness_axis.set_xlim(-0.65, len(panel) - 0.35)
    fitness_axis.tick_params(axis="x", length=0)

    fitness_axis.set_title(f"Round {round_number}", fontweight="bold", fontsize=16, pad=10)
    for axis in (fitness_axis, abundance_axis):
        axis.spines["top"].set_visible(False)
        for side in ("left", "right", "bottom"):
            axis.spines[side].set_linewidth(1.6)
        plt.setp(axis.get_yticklabels(), fontsize=12, fontweight="bold")
        axis.tick_params(axis="y", width=1.6, length=5)

    fig.tight_layout()
    stem = f"abundance_vs_fitness_round{round_number}"
    for extension in ("pdf", "png"):
        fig.savefig(THIS_DIR / f"{stem}.{extension}", dpi=400)
    plt.close(fig)
    return stem, panel


def main():
    table = build_table()
    table.to_csv(THIS_DIR / "fitness_abundance_comparison.csv", index=False)
    print(f"wrote fitness_abundance_comparison.csv: {len(table)} rows, "
          f"{len(REPORTED)} variants x {len(ROUNDS)} rounds")

    for round_number in PANEL_ROUNDS:
        stem, panel = plot_round(table, round_number)
        print(f"\n{stem}")
        for variant, row in panel.iterrows():
            print(f"  {variant:26s} fitness {row['fitness_final']:.3f}"
                  f"   abundance {row['abundance_percent']:6.3f}%")


if __name__ == "__main__":
    raise SystemExit(main())
