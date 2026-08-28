# Figure 2B - fitness landscape

Renders the mutational network of the 594-variant dataset three ways, with modelled
apparent kcat as the value axis throughout.

## Input

```
01_fitness_score_processing/02a_high_confidence_fitness_scores/
    final_fitness_scores_ETM_with_kcat_app.csv
```

Located through `repo_paths.py`, so the script runs from any directory.

## Running

```
python3 1_build_fitness_landscape.py                  # reuses the cached graph
python3 1_build_fitness_landscape.py --rebuild-graph  # rebuilds it from scratch (~50 s)
```

## Method

Each variant is a node. An edge joins two variants one amino-acid mutation apart, drawn
from the lower to the higher mutation count. Nodes with no edges, and nodes unreachable
from wild-type, are dropped, leaving **459 nodes and 600 edges**. A spring layout (seed
5000) places each variant in x and y, and `kcat_app` gives the value.

For the two landscape views the surface is interpolated onto a 200 x 200 grid, with a
square collar of zero-valued points added around the data so it falls to zero at the edges.
Four variants are labelled: WT, T241A, S68P-T241A, S68P-P203Q-T241A-H244Y.

`fitness_graph.pickle` caches the graph and is gitignored; it rebuilds deterministically.

## Outputs

All written to `figures_fitness_landscape/`.

| File | Contents |
| --- | --- |
| `network_graph_for_fitness_landscape.svg` | the network, nodes coloured by kcat_app |
| `2D_contour_fitness_landscape.svg` / `.png` | the landscape seen from above |
| `3D_final_fitness_landscape.svg` / `.png` | the 3D surface |
