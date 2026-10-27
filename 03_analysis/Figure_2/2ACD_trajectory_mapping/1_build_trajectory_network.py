import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import COLLAPSED_UNIQUE_VARIANTS_CSV  # noqa: E402

WT_LABEL = "WT"

MATRIX_OUT = THIS_DIR / "all_unique_variants_CHXCPA_distance_matrix.csv"
NODES_OUT = THIS_DIR / "all_unique_variants_CHXCPA_nodes.csv"
GRAPH_OUT = THIS_DIR / "all_unique_variants_CHXCPA_graph.gml"


def variant_name(mutations):
    """Join [['A','100','S'], ...] into 'A100S_...'. Wild-type has no mutations."""
    return "_".join("".join(triplet) for triplet in mutations) or WT_LABEL


def build_adjacency(mutation_sets):
    """1 where two variants differ by exactly one added mutation, else 0."""
    n = len(mutation_sets)
    sizes = np.array([len(s) for s in mutation_sets])
    matrix = np.zeros((n, n), dtype=np.int8)

    for i in range(n):
        set_i, size_i = mutation_sets[i], sizes[i]
        # only variants one mutation away in either direction can be neighbours
        for j in np.flatnonzero(np.abs(sizes - size_i) == 1):
            set_j = mutation_sets[j]
            smaller, larger = (set_i, set_j) if size_i < sizes[j] else (set_j, set_i)
            if smaller <= larger:
                matrix[i, j] = 1

    return matrix


def main():
    df = pd.read_csv(COLLAPSED_UNIQUE_VARIANTS_CSV)
    print(f"Loaded {len(df)} variants from {COLLAPSED_UNIQUE_VARIANTS_CSV.name}")

    mutations = [ast.literal_eval(m) for m in df["aa_mutations"]]
    names = [variant_name(m) for m in mutations]
    mutation_sets = [frozenset(tuple(t) for t in m) for m in mutations]

    if len(set(names)) != len(names):
        raise ValueError("variant names are not unique - cannot use them as node ids")

    matrix = build_adjacency(mutation_sets)
    print(f"Edges: {matrix.sum() // 2} (matrix is symmetric)")

    pd.DataFrame(matrix, columns=names).to_csv(MATRIX_OUT, index=False)
    pd.DataFrame({"name": names}).to_csv(NODES_OUT, index=False)

    graph = nx.Graph()
    graph.add_nodes_from(names)
    rows, cols = np.nonzero(matrix)
    graph.add_edges_from((names[i], names[j]) for i, j in zip(rows, cols))
    nx.write_gml(graph, GRAPH_OUT)

    print(f"Nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    for path in (MATRIX_OUT, NODES_OUT, GRAPH_OUT):
        print(f"  wrote {path.name}")


if __name__ == "__main__":
    main()
