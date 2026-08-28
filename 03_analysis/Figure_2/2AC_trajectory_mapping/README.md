# Figure 2A / 2C - trajectory mapping

Builds the mutational trajectory network for the CHX/CPA evolution campaign and the
per-round and per-mutation selections used to colour it.

## Input

One file, the backbone dataset from `02_long_read_processing`:

```
Rike_IRED_evolution_CHX_CPA_dictionary_collapsed_unique_variants.csv
```

1,342 unique amino-acid variants (1,341 mutants plus wild-type) with their occurrence
counts in each of the five rounds. The scripts locate it through `repo_paths.py`, so they
run correctly from any directory and from wherever the repo is cloned.

## Scripts

Run in order; each writes its outputs into this folder.

```
python3 1_build_trajectory_network.py
python3 2_make_round_selections.py
python3 3_make_mutation_selection.py T241A
```

`1_build_trajectory_network.py` builds the graph. Two variants are connected when one
carries exactly one more mutation than the other **and** the smaller variant's mutation
set is entirely contained in the larger. This is a "one mutation added" parent-child
relation, not a Hamming distance, which is what makes the result a trajectory map rather
than a similarity network. Produces 1,342 nodes and 1,362 edges.

`2_make_round_selections.py` splits the variants by the round in which each is **first**
observed - a variant seen in rounds 1 and 3 counts only towards round 1 - ordered within
each round by that round's occurrence count, descending.

`3_make_mutation_selection.py` lists the variants carrying a given mutation, by exact
substitution (`T241A`) or by position (`241`).

## Outputs

| File | Contents |
| --- | --- |
| `all_unique_variants_CHXCPA_distance_matrix.csv` | 1342 x 1342 adjacency matrix |
| `all_unique_variants_CHXCPA_nodes.csv` | node names, one per row |
| `all_unique_variants_CHXCPA_graph.gml` | the graph, for import into Cytoscape |
| `variants_round{1..5}.csv` | variants first seen in each round, full columns |
| `round{1..5}.txt` | their node names, for selection in Cytoscape |
| `colour_T241A.txt` | variants carrying T241A (779 of 1342) |

## Where reproducibility stops

These scripts reproduce the **data** behind panels 2A and 2C, up to and including the
graph. They do not draw the figures, and neither did the originals.

Every published panel is a Cytoscape rendering: the layout, colouring, path highlighting
and export were interactive work, recorded only in the `.cys` session files that live with
the source data. Re-running a force-directed layout will not reproduce the published node
positions - the algorithm is stochastic, and the panels show hand-rearranged layouts. Load
`all_unique_variants_CHXCPA_graph.gml` into Cytoscape and apply the round or mutation
selections above to reproduce the analysis; consult the `.cys` sessions for the appearance.

## Verification

The outputs here were checked against the original files from
`5_Data used in this paper/1_Nanopore dataset_1342 variants/03_Network graph_Figure 2ABC`:

- `..._distance_matrix.csv` - identical, header and all 1342 x 1342 cells
- `..._nodes.csv` - identical, all 1342 entries
- `..._graph.gml` - identical node set and identical edge set (1342 nodes, 1362 edges)
- `round{1..5}.txt` - identical line content for all five (mine add a trailing newline)
- `variants_round{1..5}.csv` - identical row order for all five

## Changes from the original scripts

Replaces `distance_matrix_network.py`, `network_graphs.py` (both Friederike Nintzel,
14/11/2023) and `convert_lists_into_mutations.py`. The analysis is unchanged; what changed
is that steps previously done by hand are now in code.

Wild-type has an empty mutation list, so the name-building join produced an empty string
and `WT` had to be typed into the last cell of the node list by hand. It is now labelled
in code. The distance matrix was written with a pandas index whose column then had to be
deleted manually - otherwise every edge is shifted by one position - and is now written
without one. The per-round script handled one hard-coded round per run and was edited and
re-run five times; it now does all five.

The round-assignment rule was not recorded anywhere. No script for the per-round files
survives, and they appear to have been made by hand. The rule stated above was recovered
by matching against those files and reproduces all five exactly.

`3_make_mutation_selection.py` is the one script preserved in function rather than form.
The original `convert_list_mutation_format.py` did not select anything - it reformatted an
already-selected list, reading a hard-coded `colour_T241_red.csv` that is absent from the
source data and was evidently exported by hand from Cytoscape. Neither its input nor its
output survives, so this replacement could not be checked against it.
