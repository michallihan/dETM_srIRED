# Figure 2A / 2C - trajectory mapping

Builds the mutational trajectory network for the CHX/CPA evolution campaign, and the
per-round selections used to colour it.

## Input

```
02_long_read_processing/Rike_IRED_evolution_CHX_CPA_dictionary_collapsed_unique_variants.csv
```

Located through `repo_paths.py`, so the scripts run from any directory.

## Scripts

```
python3 1_build_trajectory_network.py
python3 2_make_round_selections.py
```

`1_build_trajectory_network.py` builds the graph. Two variants are connected when one
carries exactly one more mutation than the other **and** the smaller variant's mutation set
is contained in the larger - a "one mutation added" relation rather than a Hamming
distance, which is what makes the result a trajectory map. Produces 1,342 nodes and 1,362
edges.

`2_make_round_selections.py` splits the variants by the round in which each is first
observed, ordered within each round by that round's occurrence count.

## Outputs

| File | Contents |
| --- | --- |
| `all_unique_variants_CHXCPA_distance_matrix.csv` | adjacency matrix |
| `all_unique_variants_CHXCPA_nodes.csv` | node names |
| `all_unique_variants_CHXCPA_graph.gml` | the graph, for import into Cytoscape |
| `variants_round{1..5}.csv` | variants first seen in each round |
| `round{1..5}.txt` | their node names, for selection in Cytoscape |

The `.gml` is imported into Cytoscape, where the layout, colouring, path highlighting and
export of the published panels are done.
