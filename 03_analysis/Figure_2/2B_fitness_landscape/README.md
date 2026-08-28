# Figure 2B - 3D fitness landscape

Renders the mutational network of the 594-variant dataset as a 3D surface, with modelled
apparent kcat as the height and colour axis.

## Input

```
01_fitness_score_processing/02a_high_confidence_fitness_scores/
    final_fitness_scores_ETM_with_kcat_app.csv
```

The high-confidence fitness table with two columns joined on: `kcat_app` and
`kcat_app_error`. Located through `repo_paths.py`, so the script runs from any directory.

## Running

```
python3 1_build_fitness_landscape.py                  # reuses the cached graph
python3 1_build_fitness_landscape.py --rebuild-graph  # rebuilds it from scratch (~35 s)
```

## Method

Each variant is a node. An edge joins two variants one amino-acid mutation apart, drawn
from the lower to the higher mutation count. Nodes with no edges, and nodes unreachable
from wild-type, are dropped, leaving **459 nodes and 600 edges**. A spring layout (seed
5000) places each variant in x and y, `kcat_app` gives z, and the surface is interpolated
onto a 200 x 200 grid. A square collar of zero-valued points is added around the data so
the surface falls to zero at the edges instead of ending ragged. Four variants are
labelled: WT, T241A, S68P-T241A, S68P-P203Q-T241A-H244Y.

`fitness_graph.pickle` is a cache, not a data file - it is gitignored because the graph
rebuilds deterministically (verified below).

## Outputs

```
figures_fitness_landscape/3D_final_fitness_landscape.svg
figures_fitness_landscape/3D_final_fitness_landscape.png   (600 dpi)
```

## Verification

- The graph rebuilt from scratch is **identical to the original `fitness_graph.pickle`** -
  same 459 nodes, same 600 edges, and the same node order. The cached graph therefore
  carries no hidden state, and the 459/600 figure matches the original run's stats file.
- Rendering from this merged table gives a figure **pixel-identical** to rendering from
  the original `Rike_IRED_evolution_CHX_CPA_fitness_with_kcat_app.csv`: the PNGs are
  md5-identical and the raster embedded in the SVGs decodes byte-identical. Only the
  timestamp in the SVG metadata differs. Swapping in the high-confidence table changes
  nothing about the figure.

## What was removed

Derived from `1_process_fitness_scores_lrDMS_V4.py` (Mink Sieders, 11/05/2024; edited by
Friederike Nintzel, 28/08/2024), reduced to the path producing the published panel.
Removed: the 2D contour landscape, the network-graph SVG, the interactive plotly viewer
(a 6 MB HTML file), the node-degree CSV, the stats text file, and a dead
annotation-expansion block that would have raised a `NameError` had its input list not
been empty. The surface computation, layout seed, view angle and styling are unchanged.

The ten alternative-colourmap PNGs in the original folder
(`3D_final_fitness_landscape_viridis.png` and similar) were produced by an earlier version
of the script - the version preserved there has no colourmap loop and cannot generate them.

## A caveat on the source table

The `kcat_app` values come from `Rike_IRED_evolution_CHX_CPA_fitness_with_kcat_app.csv`.
That file also carries its own `fitness_final` and `sigma_final` columns, and **those
disagree with the current high-confidence pipeline output**: 576 of 594 rows differ, by up
to 1.06 in `fitness_final`, and `sigma_final` correlates only 0.67. It was evidently built
against an earlier version of the fitness calculation.

This does not affect this figure, which uses only `kcat_app`, `Nham_aa`, `WT` and `aa_seq`
- and `Nham_aa` agrees exactly between the two tables. Only `kcat_app` and
`kcat_app_error` were taken across into the merged file; every other column is the current
high-confidence one, unchanged. Anything else drawing fitness values from that kcat table
should be checked.
