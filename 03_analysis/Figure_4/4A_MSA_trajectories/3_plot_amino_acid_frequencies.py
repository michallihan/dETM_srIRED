"""
Amino-acid frequency at each position of the trajectory, within its background.

One bar chart per panel: the wild-type residue, the two mutations of interest and
everything else pooled as "Other", as a percentage of the sequences in that background.
The wild-type share is the remainder for the campaign and a counted share for the
alignment, because the two sets are called differently; see path_dependency.py.

Outputs, per branch:
    <branch>/<position>_in_<background>.pdf
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import path_dependency as pd_  # noqa: E402


def percentages(branch, position, background, residues):
    """Percentages for the four bars of one panel, and the number of sequences."""
    table = pd.read_csv(pd_.called_mutations_csv(branch, background))
    counts = pd_.residue_counts(table["mutation"], position)
    total = len(table)

    wild_type, first, second = residues
    first_share = counts.get(first, 0) / total * 100
    second_share = counts.get(second, 0) / total * 100
    other_share = sum(
        count for residue, count in counts.items() if residue not in residues
    ) / total * 100

    if pd_.WILD_TYPE_SHARE[branch] == "remainder":
        wild_type_share = 100 - first_share - second_share - other_share
    else:
        wild_type_share = counts.get(wild_type, 0) / total * 100

    return [wild_type_share, first_share, second_share, other_share], total


def plot(branch, position, background, residues, shares):
    labels = list(residues) + ["Other"]
    plt.rcParams.update(pd_.PLOT_FONT)

    fig, ax = plt.subplots(figsize=(3, 6))
    ax.bar(labels, shares, color=pd_.FREQUENCY_COLOURS)
    ax.set_ylabel("Amino acid frequency [%]")
    ax.set_xlabel(str(position))
    ax.set_ylim(0, 100)

    name = f"{position}_in_{background or 'WT'}.pdf"
    fig.savefig(pd_.branch_dir(branch) / name, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return name


def main():
    for branch in pd_.BRANCHES:
        print(branch)
        for position, background, residues in pd_.FREQUENCY_PANELS[branch]:
            shares, total = percentages(branch, position, background, residues)
            name = plot(branch, position, background, residues, shares)
            breakdown = "  ".join(
                f"{label} {share:6.3f}%"
                for label, share in zip(list(residues) + ["Other"], shares)
            )
            print(f"  {name:20s} n={total:5d}  {breakdown}")


if __name__ == "__main__":
    main()
