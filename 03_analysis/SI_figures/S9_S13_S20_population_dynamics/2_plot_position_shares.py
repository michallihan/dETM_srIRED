"""
How each position's mutation share moved across the five rounds.

One line per position, drawn from the summary 1_ writes, starting at zero for the naive
library. Eight views of the same table: everything at once, the positions that reach the
top ten in any round, those split by whether they peak in round 5 or earlier, and four
hand-picked position sets - the second NADPH loop, the positions that at some point out-run
68, and the neighbourhoods of 68 and 203.

The four hand-picked views and the two peak-split views keep the 0-10 % window the original
used. Any position that runs past it is clipped rather than rescaled, and the run prints
which ones, so the choice is visible rather than silent.

Outputs, written next to this script, as .svg and .png:
    all_positions
    leading_positions
    leading_positions_peaking_in_round_5
    leading_positions_peaking_earlier
    nadph_loop_positions
    positions_outrunning_68
    positions_around_68
    positions_around_203
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

SUMMARY_CSV = THIS_DIR / "mutation_share_per_position.csv"

ROUNDS = (1, 2, 3, 4, 5)
ROUND_COLUMNS = [f"round_{r}" for r in ROUNDS]

LEADING_PER_ROUND = 10  # positions taken from each round to form the "leading" set
LEGEND_PER_ROUND = 7  # positions listed per round in the all-positions text legend

SELECTIONS = {
    "nadph_loop_positions": (list(range(30, 43)) + [68],
                             "Second NADPH loop, with 68 for scale"),
    "positions_outrunning_68": ([5, 24, 68, 96, 97, 103, 120, 143, 175, 203, 241, 244],
                                "Positions that out-run 68 in at least one round"),
    "positions_around_68": (list(range(64, 76)), "Positions neighbouring 68"),
    "positions_around_203": ([p for p in range(197, 227) if p != 209],
                             "Positions neighbouring 203"),
}

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable


def load_summary():
    if not SUMMARY_CSV.exists():
        raise SystemExit(f"Run 1_summarise_mutation_occurrence.py first: {SUMMARY_CSV} is missing.")
    return pd.read_csv(SUMMARY_CSV).set_index("Position")


def save(figure, name):
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"{name}.{extension}", dpi=400, bbox_inches="tight")
    plt.close(figure)


def trace(summary, position):
    """One position's share across rounds, prefixed with zero for the naive library."""
    return [0.0] + summary.loc[position, ROUND_COLUMNS].tolist()


def new_axes(title, upper):
    figure, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel("Round of evolution", fontsize=12)
    ax.set_ylabel("Share of variant pool [%]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(len(ROUNDS) + 1))
    ax.set_xticklabels([0, *ROUNDS], fontsize=10)
    ax.set_ylim(0, upper)
    ax.grid(True, linestyle="--", alpha=0.5)
    return figure, ax


def report_clipped(summary, positions, upper, name):
    over = [p for p in positions if summary.loc[p, ROUND_COLUMNS].max() > upper]
    if over:
        print(f"{name}: clipped at {upper}% - {', '.join(str(p) for p in over)}")


def plot_all_positions(summary):
    """Every position, with the leading few per round listed alongside."""
    figure, ax = new_axes("Mutation share per position across rounds of evolution", 100)
    for position in summary.index:
        ax.plot(range(len(ROUNDS) + 1), trace(summary, position), alpha=0.5, linewidth=0.8)

    entries = []
    for round_number, column in zip(ROUNDS, ROUND_COLUMNS):
        entries.append(f"Top {LEGEND_PER_ROUND} for round {round_number}:")
        # Stable sort so that positions tied on share are listed lowest-first.
        top = summary[column].sort_values(ascending=False, kind="stable").head(LEGEND_PER_ROUND)
        entries += [f"{position}: {value:.2f}%" for position, value in top.items()]

    ax.text(1.05, 0.5, "\n".join(entries), transform=ax.transAxes, fontsize=10,
            verticalalignment="center",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9))
    figure.tight_layout()
    save(figure, "all_positions")


def plot_selection(summary, positions, title, name, upper):
    """A named set of positions, one labelled line each."""
    positions = [p for p in positions if p in summary.index]
    report_clipped(summary, positions, upper, name)
    figure, ax = new_axes(title, upper)
    for position in positions:
        ax.plot(range(len(ROUNDS) + 1), trace(summary, position),
                alpha=0.8, linewidth=1.5, label=f"Position {position}")
    ax.legend(title="Positions", fontsize=10, bbox_to_anchor=(1.05, 1), loc="upper left")
    figure.tight_layout()
    save(figure, name)


def main():
    summary = load_summary()

    plot_all_positions(summary)

    leading = set()
    for column in ROUND_COLUMNS:
        leading.update(summary[column].nlargest(LEADING_PER_ROUND).index)
    leading = sorted(leading)
    print(f"leading positions ({len(leading)}): {', '.join(str(p) for p in leading)}")

    plot_selection(summary, leading, "Leading positions across rounds of evolution",
                   "leading_positions", 100)

    peaks_last = [p for p in leading
                  if summary.loc[p, ROUND_COLUMNS].to_numpy().argmax() == len(ROUNDS) - 1]
    peaks_early = [p for p in leading if p not in peaks_last]

    plot_selection(summary, peaks_last, "Leading positions peaking in round 5",
                   "leading_positions_peaking_in_round_5", 10)
    plot_selection(summary, peaks_early, "Leading positions peaking before round 5",
                   "leading_positions_peaking_earlier", 10)

    for name, (positions, title) in SELECTIONS.items():
        plot_selection(summary, positions, title, name, 10)


if __name__ == "__main__":
    main()
