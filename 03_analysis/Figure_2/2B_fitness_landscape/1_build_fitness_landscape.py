"""
Figure 2B - fitness landscape.

Builds the mutational network from the high-confidence fitness table and renders it three
ways, with modelled apparent kcat (kcat_app) as the value axis throughout.

Method: variants become nodes, and an edge joins two variants one amino-acid mutation
apart where the second carries one more mutation than the first. Nodes with no edges, and
nodes unreachable from wild-type, are dropped - leaving 459 nodes and 600 edges. A spring
layout gives each variant an (x, y) position and kcat_app gives the value. For the two
landscape views the surface is interpolated onto a 200 x 200 grid, with a square collar of
zero-valued points added around the data so it falls to zero at the edges rather than
ending ragged.

Input: 01_fitness_score_processing/02a_high_confidence_fitness_scores/
       final_fitness_scores_ETM_with_kcat_app.csv

Outputs, written to figures_fitness_landscape/ next to this script:
    network_graph_for_fitness_landscape.svg   the network itself, nodes coloured by kcat_app
    2D_contour_fitness_landscape.svg / .png   the landscape seen from above
    3D_final_fitness_landscape.svg / .png     the published 3D surface

Derived from 1_process_fitness_scores_lrDMS_V4.py (Mink Sieders, 11/05/2024; edited by
Friederike Nintzel, 28/08/2024). Removed from that original: the interactive plotly viewer,
the node-degree CSV, the stats text file, and a dead annotation-expansion block that would
have raised a NameError had its input list not been empty. Surface computation, layout
seed, view angle and styling are unchanged.

NOTE ON EXACT REPRODUCTION. The published panels were rendered with matplotlib 3.5.2.
kcat_app reaches 100.189, and newer matplotlib autoscales the 3D z-axis differently
(3.5.2 ticks to 80, 3.10 extends to 100), which rescales the projection slightly. The
surface, peaks, colour mapping and labels are unaffected. See README.md.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import proj3d  # noqa: F401  (registers the 3d projection)
from scipy.interpolate import griddata

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(THIS_DIR))

from repo_paths import FITNESS_HIGH_CONF_KCAT_CSV  # noqa: E402
import function_library_lrDMS_analysis as lrDMSa  # noqa: E402

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable

VALUE_COLUMN = "kcat_app"
ERROR_COLUMN = "kcat_app_error"
VALUE_LABEL = "kcat_app"
COLORMAP = "Spectral_r"
LABEL_NODES = ["WT", "T241A", "S68P-T241A", "S68P-P203Q-T241A-H244Y"]
LABEL_DOT_OFFSET_FRACTION = 0.025
LABEL_TEXT_OFFSET_FRACTION = 0.1
CONTOUR_LABEL_DOT_SIZE = 55
CONTOUR_LABEL_FONT_SIZE = 13
STATIC_POLYGON_EDGE_COLOR = (0.45, 0.45, 0.45, 0.65)
STATIC_POLYGON_EDGE_WIDTH = 0.1125
SEED_SPRING_LAYOUT = 5000
VIEW_ELEV, VIEW_AZIM = 24.99, -160.27
GRID_POINTS = 200
COLLAR_POINTS_PER_SIDE = 50

OUTPUT_DIR = THIS_DIR / "figures_fitness_landscape"
GRAPH_CACHE = THIS_DIR / "fitness_graph.pickle"


def open_data(input_csv):
    data = pd.read_csv(input_csv)
    required = ["aa_seq", "WT", "Nham_aa", VALUE_COLUMN, ERROR_COLUMN]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    data["WT"] = data["WT"].fillna(False).astype(bool)
    data = data.dropna(subset=["aa_seq", "Nham_aa", VALUE_COLUMN]).copy()
    data["Nham_aa"] = data["Nham_aa"].astype(int)

    if "mut_id" not in data.columns:
        data["mut_id"] = lrDMSa.define_mutation_ids(data)

    return data.sort_values(by=["WT", VALUE_COLUMN], ascending=False).reset_index(drop=True)


def build_graph(dataframe, rebuild=False):
    """Nodes are variants; edges join variants one mutation apart, low to high Nham_aa."""
    data = dataframe.copy(deep=True)
    data["mut_id"] = data["mut_id"].apply(lambda x: x.split("-"))
    data["name"] = data["mut_id"].apply(lambda x: "WT" if "WT" in x else "-".join(x) if x else "WT")

    if GRAPH_CACHE.exists() and not rebuild:
        print(f"using {GRAPH_CACHE.name}")
        graph = pickle.load(open(GRAPH_CACHE, "rb"))
    else:
        print("building node embedding")
        graph = nx.MultiGraph()
        for _, row in data.iterrows():
            graph.add_node(row["name"], mutations=row["mut_id"])

        for i, row1 in data.iterrows():
            for j, row2 in data.iterrows():
                if data.loc[i, "Nham_aa"] < data.loc[j, "Nham_aa"]:
                    if lrDMSa.is_one_mutation_apart(row1["aa_seq"], row2["aa_seq"]):
                        graph.add_edge(row1["name"], row2["name"])

        graph.remove_nodes_from([n for n, d in dict(graph.degree()).items() if d == 0])

        if "WT" not in graph:
            raise ValueError("No WT node in the graph")
        graph = lrDMSa.remove_unreachable_nodes(graph, "WT")

        pickle.dump(graph, open(GRAPH_CACHE, "wb"))

    values = data.set_index("name")[VALUE_COLUMN].to_dict()
    for node in graph.nodes():
        graph.nodes[node][VALUE_COLUMN] = values[node]

    print(f"graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    return graph, [values[node] for node in graph.nodes()]


def draw_network_graph(graph, values, pos):
    """The network itself: spring layout, nodes coloured by kcat_app."""
    value_min, value_max = min(values), max(values)
    span = value_max - value_min
    cmap = plt.get_cmap(COLORMAP)
    colors = [cmap(0.5 if span == 0 else (v - value_min) / span) for v in values]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(graph, pos, edge_color=(0, 0, 0, 1), alpha=0.5, width=0.5,
                           style="-", connectionstyle="arc3")
    nx.draw_networkx_nodes(graph, pos, node_size=10, node_color=colors)

    for node in LABEL_NODES:
        if node in pos:
            x, y = pos[node]
            plt.text(x, y - 0.045, node, fontsize=8, ha="center", va="center", color="white",
                     bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))

    scalar_map = plt.cm.ScalarMappable(cmap=cmap)
    scalar_map.set_array([value_min, value_max])
    plt.colorbar(scalar_map, ax=plt.gca(), label=VALUE_LABEL, shrink=1, pad=0.01,
                 orientation="vertical")

    plt.axis("on")
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_DIR / "network_graph_for_fitness_landscape.svg")
    plt.close()
    print("  wrote network_graph_for_fitness_landscape.svg")


def square_collar(x_coords, y_coords):
    """Points around a square enclosing the data, so the surface falls to zero at the edge."""
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    side = max(max_x - min_x, max_y - min_y)
    mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    min_x, max_x = mid_x - side / 2, mid_x + side / 2
    min_y, max_y = mid_y - side / 2, mid_y + side / 2

    n = COLLAR_POINTS_PER_SIDE
    xs = np.concatenate([np.linspace(min_x, max_x, n), np.full(n, max_x),
                         np.linspace(max_x, min_x, n), np.full(n, min_x)])
    ys = np.concatenate([np.full(n, min_y), np.linspace(min_y, max_y, n),
                         np.full(n, max_y), np.linspace(max_y, min_y, n)])
    return xs, ys


def project_to_figure(fig, ax, x, y, z):
    xp, yp, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    dx, dy = ax.transData.transform((xp, yp))
    return fig.transFigure.inverted().transform((dx, dy))


def add_projected_label_overlays(fig, ax, label_positions_3d):
    for label, x, y, dot_z, text_z in label_positions_3d:
        dot_x, dot_y = project_to_figure(fig, ax, x, y, dot_z)
        text_x, text_y = project_to_figure(fig, ax, x, y, text_z)
        fig.patches.append(Circle((dot_x, dot_y), radius=0.004, transform=fig.transFigure,
                                  facecolor="#444444", edgecolor="#444444",
                                  clip_on=False, zorder=1000))
        fig.text(text_x, text_y, label, fontsize=8, ha="center", va="center", color="white",
                 zorder=1001, clip_on=False,
                 bbox=dict(facecolor="black", alpha=0.55, edgecolor="none"))


def draw_2d_contour(grid_x, grid_y, grid_z, pos):
    """The landscape seen from above."""
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.contourf(grid_x, grid_y, grid_z, levels=20, cmap=COLORMAP)
    cbar = fig.colorbar(contour)
    cbar.set_label(VALUE_LABEL)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    right=False, left=False, labelbottom=False)

    for node in LABEL_NODES:
        if node in pos:
            x, y = pos[node]
            ax.scatter(x, y, c="black", s=CONTOUR_LABEL_DOT_SIZE, edgecolors="black", zorder=5)
    for node in LABEL_NODES:
        if node in pos:
            x, y = pos[node]
            ax.text(x, y - 0.04, node, fontsize=CONTOUR_LABEL_FONT_SIZE, ha="center",
                    va="center", color="white",
                    bbox=dict(facecolor="black", alpha=0.72, edgecolor="none"))

    plt.savefig(OUTPUT_DIR / "2D_contour_fitness_landscape.svg")
    plt.savefig(OUTPUT_DIR / "2D_contour_fitness_landscape.png", dpi=600)
    plt.close()
    print("  wrote 2D_contour_fitness_landscape.svg and .png")


def draw_3d_surface(grid_x, grid_y, grid_z, x_coords, y_coords, label_positions_3d):
    """The published 3D surface."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(grid_x, grid_y, grid_z, cmap=COLORMAP, rstride=1, cstride=1,
                              edgecolor=STATIC_POLYGON_EDGE_COLOR,
                              linewidth=STATIC_POLYGON_EDGE_WIDTH, antialiased=True)
    surface.set_rasterized(True)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axes.set_xlim3d(left=min(x_coords), right=max(x_coords))
    ax.axes.set_ylim3d(bottom=min(y_coords), top=max(y_coords))
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    right=False, left=False, labelbottom=False)
    ax.set_zlabel(VALUE_LABEL)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
    fig.canvas.draw()
    add_projected_label_overlays(fig, ax, label_positions_3d)

    plt.savefig(OUTPUT_DIR / "3D_final_fitness_landscape.svg")
    plt.savefig(OUTPUT_DIR / "3D_final_fitness_landscape.png", dpi=600)
    plt.close()
    print("  wrote 3D_final_fitness_landscape.svg and .png")


def fitness_landscape(graph, values, pos):
    positions = np.array(list(pos.values()))
    x_coords, y_coords = positions[:, 0], positions[:, 1]

    collar_x, collar_y = square_collar(x_coords, y_coords)
    x_coords = np.append(x_coords, collar_x)
    y_coords = np.append(y_coords, collar_y)
    values = np.append(values, np.zeros(len(collar_x)))

    grid_x, grid_y = np.meshgrid(np.linspace(min(x_coords), max(x_coords), GRID_POINTS),
                                 np.linspace(min(y_coords), max(y_coords), GRID_POINTS))
    grid_z = griddata((x_coords, y_coords), values, (grid_x, grid_y), method="linear")

    span = np.nanmax(values) - np.nanmin(values)
    dot_offset = span * LABEL_DOT_OFFSET_FRACTION
    text_offset = span * LABEL_TEXT_OFFSET_FRACTION

    label_positions_3d = []
    for node in LABEL_NODES:
        if node in pos:
            x, y = pos[node]
            z = values[list(graph.nodes).index(node)]
            label_positions_3d.append((node, x, y, z + dot_offset, z + text_offset))

    OUTPUT_DIR.mkdir(exist_ok=True)
    draw_2d_contour(grid_x, grid_y, grid_z, pos)
    draw_3d_surface(grid_x, grid_y, grid_z, x_coords, y_coords, label_positions_3d)


def main():
    rebuild = "--rebuild-graph" in sys.argv
    data = open_data(FITNESS_HIGH_CONF_KCAT_CSV)
    print(f"loaded {len(data)} variants from {FITNESS_HIGH_CONF_KCAT_CSV.name}")
    graph, values = build_graph(data, rebuild=rebuild)
    pos = nx.spring_layout(graph, seed=SEED_SPRING_LAYOUT)
    draw_network_graph(graph, values, pos)
    fitness_landscape(graph, values, pos)


if __name__ == "__main__":
    main()
