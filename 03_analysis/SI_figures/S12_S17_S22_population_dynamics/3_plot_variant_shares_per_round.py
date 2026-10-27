import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import DIMSUM_NAIVE_COUNTS_XLSX, DIMSUM_ROUND_COUNTS_TSV  # noqa: E402

ROUND_TABLES = {0: DIMSUM_NAIVE_COUNTS_XLSX, **DIMSUM_ROUND_COUNTS_TSV}

# Rows DiMSum flags: column -> value removed.
DISCARD = {"STOP": True, "STOP_readthrough": True, "barcode_valid": False, "indel": True}

EXTRA_PER_ROUND = 2  # largest variants outside the named set to show per round

NAMED_COLOURS = {
    "WT": "dimgray",
    "T241A": "#D72631",
    "S68P-T241A": "#FF6F3C",
    "S68P-P203Q-T241A": "#FFA500",
    "S68P-T241A-H244Y": "#FFD93D",
    "S68P-P203Q-T241A-H244Y": "#FFB07C",
}
EXTRA_COLOURS = ["#1B3B6F", "#3399FF", "#008080", "#40E0D0", "#2E8B57", "#32CD32",
                 "#98FF98", "#00CED1", "#4169E1", "#1E90FF", "#3CB371", "#7FFFD4"]
REST_COLOUR = "lightgrey"

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 16


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


def read_round(path):
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t")


def round_shares(round_number):
    """One round's variants with each one's mean share of the replicate read pools."""
    table = read_round(ROUND_TABLES[round_number])
    for column, discarded in DISCARD.items():
        if column in table.columns:
            table = table[table[column] != discarded]

    inputs = [c for c in table.columns if c.startswith("input")]
    collapsed = table.groupby("aa_seq", as_index=False).agg(
        {**{c: "sum" for c in inputs}, "WT": "first"}
    )

    wild_type = collapsed.loc[collapsed["WT"] == True, "aa_seq"]  # noqa: E712
    if len(wild_type) != 1:
        raise ValueError(f"round {round_number} has {len(wild_type)} wild-type rows, expected 1")
    collapsed["mut_id"] = mutation_ids(collapsed["aa_seq"], wild_type.iloc[0])

    fractions = collapsed[inputs] / collapsed[inputs].sum()
    return pd.DataFrame({"round": round_number,
                         "mut_id": collapsed["mut_id"],
                         "share": fractions.mean(axis=1)})


def bar_rows(shares, round_number):
    """The named variants, that round's largest others, and everything else as one bar."""
    named = shares[shares["mut_id"].isin(NAMED_COLOURS)]
    others = shares[~shares["mut_id"].isin(NAMED_COLOURS)]
    extra = others.nlargest(EXTRA_PER_ROUND, "share")

    shown = pd.concat([named, extra])
    rest = pd.DataFrame([{"round": round_number, "mut_id": "rest",
                          "share": shares["share"].sum() - shown["share"].sum()}])
    return pd.concat([shown, rest], ignore_index=True)


def stack_order(pivot):
    """Named variants first, then the extras least-frequent-first, then the remainder."""
    extras = [c for c in pivot.columns if c not in NAMED_COLOURS and c != "rest"]
    extras.sort(key=lambda column: pivot[column].sum())
    return [c for c in list(NAMED_COLOURS) + extras + ["rest"] if c in pivot.columns]


def plot(pivot):
    colours = []
    for index, column in enumerate(pivot.columns):
        if column in NAMED_COLOURS:
            colours.append(NAMED_COLOURS[column])
        elif column == "rest":
            colours.append(REST_COLOUR)
        else:
            colours.append(EXTRA_COLOURS[index % len(EXTRA_COLOURS)])

    ax = pivot.plot(kind="bar", stacked=True, figsize=(12, 8), color=colours, zorder=3)
    ax.set_xlabel("Round of evolution")
    ax.set_ylabel("Fraction of variant pool")
    ax.set_xticklabels([f"r{r}" for r in pivot.index], rotation=0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    order = [c for c in pivot.columns if c in by_label]
    ax.legend([by_label[c] for c in order], order, title="Mutation ID",
              bbox_to_anchor=(1.05, 1), loc="upper left")

    figure = ax.get_figure()
    figure.tight_layout()
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"variant_share_per_round.{extension}", dpi=400)
    plt.close(figure)


def main():
    bars = []
    for round_number in sorted(ROUND_TABLES):
        shares = round_shares(round_number)
        bars.append(bar_rows(shares, round_number))
        print(f"round {round_number}: {len(shares)} variants, "
              f"wild type {shares.loc[shares['mut_id'] == 'WT', 'share'].iloc[0]:.3%}")

    pivot = pd.concat(bars, ignore_index=True).pivot(
        index="round", columns="mut_id", values="share").fillna(0)
    pivot = pivot[stack_order(pivot)]

    pivot.to_csv(THIS_DIR / "variant_share_per_round.csv")
    plot(pivot)


if __name__ == "__main__":
    main()
