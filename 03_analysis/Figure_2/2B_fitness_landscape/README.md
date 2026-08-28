# Figure 2B - fitness landscape

Renders the mutational network of the 594-variant dataset three ways, with modelled
apparent kcat as the value axis throughout.

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
python3 1_build_fitness_landscape.py --rebuild-graph  # rebuilds it from scratch (~50 s)
```

## Method

Each variant is a node. An edge joins two variants one amino-acid mutation apart, drawn
from the lower to the higher mutation count. Nodes with no edges, and nodes unreachable
from wild-type, are dropped, leaving **459 nodes and 600 edges**. A spring layout (seed
5000) places each variant in x and y, and `kcat_app` gives the value.

For the two landscape views the surface is interpolated onto a 200 x 200 grid, with a
square collar of zero-valued points added around the data so it falls to zero at the edges
instead of ending ragged. Four variants are labelled throughout: WT, T241A, S68P-T241A,
S68P-P203Q-T241A-H244Y.

`fitness_graph.pickle` is a cache, not a data file - it is gitignored because the graph
rebuilds deterministically (verified below).

## Outputs

All written to `figures_fitness_landscape/`.

| File | Contents |
| --- | --- |
| `network_graph_for_fitness_landscape.svg` | the network itself, nodes coloured by kcat_app |
| `2D_contour_fitness_landscape.svg` / `.png` | the landscape seen from above |
| `3D_final_fitness_landscape.svg` / `.png` | the published 3D surface |

## Verification

- The graph rebuilt from scratch is **identical to the original `fitness_graph.pickle`** -
  same 459 nodes, same 600 edges, and the same node order. The cached graph therefore
  carries no hidden state, and the 459/600 figure matches the original run's stats file.
- Rendering from this merged table gives a 3D figure **pixel-identical** to rendering from
  the original `Rike_IRED_evolution_CHX_CPA_fitness_with_kcat_app.csv`: the PNGs are
  md5-identical and the raster embedded in the SVGs decodes byte-identical. Swapping in the
  high-confidence table changes nothing about the figure.
- Against the **published** figures, all three regenerate faithfully but not byte-exactly:
  the 2D contour differs in 1.5% of pixels, the network graph in 4.0%, the 3D surface in
  10.6%. In every case the differences are confined to text antialiasing, line rendering
  and sub-pixel positioning; topology, contours, peaks, colour mapping, colourbars and
  labels all match.

### Why the pixels differ

This is a matplotlib version artefact, not a data or code difference. The published panels
were rendered with **matplotlib 3.5.2** (recorded in the original SVG metadata); the
figures here with 3.10.9.

The 3D surface differs most because `kcat_app` reaches 100.189 and the two versions
autoscale the 3D z-axis differently - 3.5.2 ticks to 80, 3.10.9 extends to 100 - which
rescales the whole projection slightly. The 2D views have no such autoscaling and differ
only in glyph rendering.

Reproducing the published panels pixel-for-pixel would need matplotlib 3.5.2, which has no
wheel for current Python versions. The figures committed here are what the script produces
on a current toolchain.

## What was removed from the original

Derived from `1_process_fitness_scores_lrDMS_V4.py` (Mink Sieders, 11/05/2024; edited by
Friederike Nintzel, 28/08/2024). Removed: the interactive plotly viewer (a 6 MB HTML file),
the node-degree CSV, the stats text file, and a dead annotation-expansion block that would
have raised a `NameError` had its input list not been empty. Surface computation, layout
seed, view angle and styling are unchanged.

The ten alternative-colourmap PNGs in the original folder
(`3D_final_fitness_landscape_viridis.png` and similar) were produced by an earlier version
of the script - the version preserved there has no colourmap loop and cannot generate them.

## A caveat on the source table

The `kcat_app` values come from `Rike_IRED_evolution_CHX_CPA_fitness_with_kcat_app.csv`.
That file also carries its own `fitness_final` and `sigma_final` columns, and **those
disagree with the current high-confidence pipeline output**: 576 of 594 rows differ, by up
to 1.06 in `fitness_final`, and `sigma_final` correlates only 0.67. It was evidently built
against an earlier version of the fitness calculation.

This does not affect these figures, which use only `kcat_app`, `Nham_aa`, `WT` and
`aa_seq` - and `Nham_aa` agrees exactly between the two tables. Only `kcat_app` and
`kcat_app_error` were taken across into the merged file; every other column is the current
high-confidence one, unchanged. Anything else drawing fitness values from that kcat table
should be checked.
