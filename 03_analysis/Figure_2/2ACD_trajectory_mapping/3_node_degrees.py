"""
Node degree per round: which variants have the most descendants in the trajectory network.

The network built by 1_build_trajectory_network.py joins two variants when one carries exactly
one more amino-acid substitution than the other and the smaller variant's substitution set is
contained in the larger. A variant's node degree in that network is the number of one-step
neighbours it has - the immediate parents it could have come from plus the immediate
descendants that extend it by one substitution. A high degree marks a variant at a branch
point: many single-step moves pass through it.

Degree alone favours variants that were simply around longer and so had more chances to spawn
neighbours. Dividing by the number of rounds a variant was seen in corrects for that, giving
descendants per round of presence, which is what the table ranks on.

Degrees are read straight from the graph 1_build_trajectory_network.py writes, so this is the
same network as Figure 2A/C. Run that script first. Round membership is not in the graph and
is taken from the round_1..round_5 columns of the nanopore dictionary: a variant "occurs" in a
round when its count there is greater than zero. Round 0, the naive library, is not a selection
round and is not counted.

Outputs, written next to this script:
    node_degrees.xlsx   every variant, ranked by node degree per round, with a Legend sheet
"""

import ast
import sys
from pathlib import Path

import pandas as pd
import networkx as nx

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import COLLAPSED_UNIQUE_VARIANTS_CSV  # noqa: E402

WT_LABEL = "WT"
ROUND_COLUMNS = [f"round_{i}" for i in range(1, 6)]

GRAPH_IN = THIS_DIR / "all_unique_variants_CHXCPA_graph.gml"
OUT_XLSX = THIS_DIR / "node_degrees.xlsx"

COLUMN_NOTES = [
    ("variant", "Variant identity, substitutions ordered N-terminal end first, joined with "
                "dashes. The wild type is WT."),
    ("n_mutations", "Number of amino-acid substitutions relative to the wild type."),
    ("node_degree", "One-step neighbours in the trajectory network: variants that differ by "
                    "exactly one added or removed substitution, with the smaller substitution "
                    "set contained in the larger. Read from the graph this folder builds, the "
                    "same network as Figure 2A/C."),
    ("n_rounds", "Number of selection rounds (1-5) the variant was seen in, i.e. rounds whose "
                 "count for it is greater than zero."),
    ("node_degree_per_round", "node_degree divided by n_rounds. The ranking column: descendants "
                              "per round of presence, so variants are not rewarded merely for "
                              "persisting across many rounds."),
    ("rounds_present", "Which of rounds 1-5 the variant was seen in."),
]


def graph_name(mutations):
    """Node id as 1_build_trajectory_network.py writes it: 'A100S_P138L_...', WT for none."""
    return "_".join("".join(triplet) for triplet in mutations) or WT_LABEL


def display_name(mutations):
    """Same variant with dashes, matching the fitness tables: 'A100S-P138L-...', WT for none."""
    return "-".join("".join(triplet) for triplet in mutations) or WT_LABEL


def write_workbook(path, table, notes):
    """One data sheet ranked as given, and one sheet defining its columns."""
    legend = [("Column", "Definition"), *notes]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Node_degrees", index=False)
        pd.DataFrame(legend).to_excel(writer, sheet_name="Legend", index=False, header=False)
        data_sheet = writer.sheets["Node_degrees"]
        for column in data_sheet.columns:
            longest = max(len(str(cell.value)) for cell in column)
            data_sheet.column_dimensions[column[0].column_letter].width = min(longest + 2, 34)
        legend_sheet = writer.sheets["Legend"]
        for letter, width in zip("AB", (24, 104)):
            legend_sheet.column_dimensions[letter].width = width


def main():
    if not GRAPH_IN.exists():
        raise FileNotFoundError(
            f"{GRAPH_IN.name} not found - run 1_build_trajectory_network.py first")

    graph = nx.read_gml(GRAPH_IN)
    degree = dict(graph.degree())
    print(f"Loaded network: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    df = pd.read_csv(COLLAPSED_UNIQUE_VARIANTS_CSV)
    mutations = [ast.literal_eval(m) for m in df["aa_mutations"]]
    node_ids = [graph_name(m) for m in mutations]

    missing = set(node_ids) - set(degree)
    if missing:
        raise ValueError(f"{len(missing)} variants are not nodes in the graph, e.g. "
                         f"{sorted(missing)[:3]} - is the graph built from this dictionary?")

    present = df[ROUND_COLUMNS].gt(0)
    rounds_present = present.apply(
        lambda row: ", ".join(str(i) for i, seen in enumerate(row, start=1) if seen), axis=1
    )

    table = pd.DataFrame({
        "variant": [display_name(m) for m in mutations],
        "n_mutations": df["aa_mutation_number"].astype(int),
        "node_degree": [degree[node] for node in node_ids],
        "n_rounds": present.sum(axis=1).astype(int),
        "rounds_present": rounds_present,
    })

    seen_in_a_round = table["n_rounds"] > 0
    dropped = int((~seen_in_a_round).sum())
    if dropped:
        print(f"note: {dropped} variants occur in no round 1-5 and are left out")
    table = table[seen_in_a_round].copy()

    table["node_degree_per_round"] = table["node_degree"] / table["n_rounds"]
    table = table.sort_values(
        ["node_degree_per_round", "node_degree"], ascending=False, kind="stable"
    ).reset_index(drop=True)

    table = table[["variant", "n_mutations", "node_degree", "n_rounds",
                   "node_degree_per_round", "rounds_present"]]

    write_workbook(OUT_XLSX, table, COLUMN_NOTES)
    print(f"wrote {OUT_XLSX.name}: {len(table)} variants ranked by node degree per round")

    top = table.head(12).copy()
    top["node_degree_per_round"] = top["node_degree_per_round"].round(2)
    print("\nTop 12 by node degree per round:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
