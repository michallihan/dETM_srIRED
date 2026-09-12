"""
Per-position mutation occurrence across a sequence set, stacked by amino acid.

Each bar is one position of the IRED; its segments are the residues found there, as a
percentage of the sequences in the set. Drawn for the campaign as a whole and for the
alignment both as a whole and within each background, following OCCURRENCE_SETS in
path_dependency.py.

Residues are ordered alphabetically, so a segment sits in the same place in every bar and
in the legend. Left to the order residues are first encountered, the stack would depend on
the row order of the input and on the pandas version.

Outputs, per branch:
    <branch>/mutation_occurrence_all.pdf / .csv
    <branch>/mutation_occurrence_in_<background>.pdf / .csv   (alignment only)
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import path_dependency as pd_  # noqa: E402

POSITIONS = range(1, 292)
STOP_LABEL = "premature stop codon"


def occurrence_matrix(table):
    """Percentage of the set carrying each residue, per position."""
    counts = {}
    total = len(table)

    for mutations in table["mutation"]:
        if isinstance(mutations, float):  # empty cell
            continue
        mutation_string = mutations.strip()
        if mutation_string in pd_.NON_MUTATION_VALUES:
            continue
        for _, position, residue in pd_.parse_mutation_string(mutation_string):
            if residue == "*":
                residue = STOP_LABEL
            counts.setdefault(position, {})
            counts[position][residue] = counts[position].get(residue, 0) + 1

    rows = [
        {residue: count / total * 100 for residue, count in counts.get(position, {}).items()}
        for position in POSITIONS
    ]
    matrix = pd.DataFrame(rows, index=list(POSITIONS))
    return matrix.reindex(sorted(matrix.columns), axis=1), total


def plot(matrix, stem, branch):
    colours = [pd_.AMINO_ACID_COLOURS.get(column, "black") for column in matrix.columns]
    ax = matrix.plot(kind="bar", stacked=True, figsize=(15, 5), color=colours)
    ax.set_xlabel("Amino acid position")
    ax.set_ylabel("Occurrence of amino acid mutation (%)")
    ax.set_xticks(np.arange(-1, 290, step=5))
    ax.set_xticklabels(np.arange(0, 292, step=5))
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_ylim([0, 60])
    ax.set_yticks(np.arange(0, 61, step=5))
    ax.set_title("")

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = zip(*sorted(zip(handles, labels), key=lambda pair: pair[1]))
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5))

    figure = ax.get_figure()
    figure.savefig(pd_.branch_dir(branch) / f"{stem}.pdf", format="pdf", bbox_inches="tight")
    plt.close(figure)


def main():
    for branch in pd_.BRANCHES:
        print(branch)
        for background in pd_.OCCURRENCE_SETS[branch]:
            table = pd.read_csv(pd_.called_mutations_csv(branch, background))
            matrix, total = occurrence_matrix(table)
            stem = pd_.occurrence_stem(background)
            matrix.to_csv(pd_.branch_dir(branch) / f"{stem}.csv", index=False)
            plot(matrix, stem, branch)
            positions = int((matrix.fillna(0).sum(axis=1) > 0).sum())
            print(f"  {stem:28s} n={total:5d}  {positions} positions mutated")


if __name__ == "__main__":
    main()
