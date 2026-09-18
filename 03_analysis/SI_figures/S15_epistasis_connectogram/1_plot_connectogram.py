"""
Every measured step through the campaign's sequence space, and what each one did to kcat(app).

Nodes are variants, arranged in columns by how many substitutions they carry from the wild
type. An edge joins two variants one substitution apart, drawn from the lower column to the
higher, so a path from left to right is a mutational trajectory. Node colour is the variant's
own kcat(app). Edge colour is epsilon, how far the variant's kcat(app) sits from what the step
promised: the parent and the added substitution each contribute their fold-change over the
wild type, and the expectation is their product. Red is a step that beat that expectation,
blue one that fell short, pale grey one that met it. A black dashed edge is a step between two
measured variants whose epsilon could not be computed, because the substitution being added
was never measured on its own.

kcat(app) is multiplicative, so the arithmetic is done on m = log10(kcat_app / kcat_app of the
wild type), the log10 fold-change over wild type, which puts the wild type at zero and makes
the expectation a sum:

    expected m(daughter) = m(parent) + m(added substitution)
    epsilon              = m(observed) - m(expected)

Epsilon is therefore log10 of the ratio of observed to expected kcat(app); +0.3 is a step that
did twice what it promised.

Two decompositions produce an epsilon. A double mutant is predicted from its two singles, and
carries one epsilon drawn as two edges, one from each single. A variant with three or more
substitutions is predicted from a measured variant one substitution smaller plus the
substitution that was added, and every such split is enumerated, so a variant reached by
several routes carries one epsilon per route. Those routes routinely disagree, which is the
point of the figure.

Both need the added substitution measured on its own, and the campaign scored only 90 single
mutants against the 361 substitutions its variants carry. The 1,122 singles in
01_fitness_score_processing/max_single_mutants.csv, from Max's deep mutational scan of the
same enzyme, supply the rest; they are what lift the analysis from 184 epsilon values to 461.
Their kcat(app) is computed with the conversion fitted in
01_fitness_score_processing/03_deriving_kcatapp, the same conversion behind the campaign
table's own column.

Colour scales span the image of fitness -4 to +4 under that conversion, for both the nodes and
epsilon, so the two halves of the figure are on comparable footing and the scale covers the
data without being set by it.

Which variants appear. A variant with two or more substitutions becomes a node only if at
least one route into it could be scored, and a single mutant only if it sits inside one of
those variants -- most of Max's 1,122 singles are unrelated to this campaign and would
otherwise hang off the wild type unconnected. Nodes left isolated or unreachable from the wild
type once the edges are drawn are then dropped.

Note on the gate: GATE = "kcat" admits every higher-order variant, because kcat_app_error is
smaller than kcat_app everywhere in the table. GATE = "fitness" applies fitness_final >
sigma_final, which keeps 377 of 503. Neither tests whether an individual epsilon differs from
zero -- no such test is made here, and "quantified" in the summary means computed.

Outputs, written next to this script:
    epistasis_interactions.csv     one row per epsilon, with both parts of the expectation
    connectogram.svg / .png
"""

import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import (  # noqa: E402
    FITNESS_HIGH_CONF_KCAT_CSV,
    KCAT_APP_FIT_JSON,
    MAX_SINGLE_MUTANTS_CSV,
)

COLUMNS = ["aa_seq", "mut_id", "Nham_aa", "fitness_final", "sigma_final"]

# Which higher-order variants become nodes. "kcat" is kcat_app > kcat_app_error, which every
# variant passes; "fitness" is fitness_final > sigma_final, which keeps 377 of 503.
GATE = "kcat"

# The trajectory whose sub-combinations carry a text label and a thicker edge.
LABELLED_TRAJECTORY = "A28T-S68P-P203Q-T241A-H244Y"

# Colour windows, as fitness scores; converted to the kcat(app) axis below.
FITNESS_COLOUR_WINDOW = 4.0

MIN_DEGREE = 0  # nodes at or below this degree are dropped after the graph is built

NODE_CMAP = "jet"
EDGE_CMAP = LinearSegmentedColormap.from_list("epistasis", ["blue", "lightgrey", "red"])
KCAT_TICKS = [0.5, 1, 2, 5, 10, 20, 50, 100]

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 11


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def variants():
    """The campaign's scored variants and Max's single mutants, with kcat(app) on both."""
    fit = json.loads(KCAT_APP_FIT_JSON.read_text())

    campaign = pd.read_csv(FITNESS_HIGH_CONF_KCAT_CSV)[COLUMNS + ["kcat_app", "kcat_app_error"]]
    campaign["source"] = "campaign"

    singles = pd.read_csv(MAX_SINGLE_MUTANTS_CSV)[COLUMNS]
    singles["kcat_app"] = fit["a"] * np.exp(fit["b"] * singles["fitness_final"])
    singles["kcat_app_error"] = abs(fit["b"]) * singles["kcat_app"] * singles["sigma_final"]
    singles["source"] = "max_singles"

    overlap = set(campaign["mut_id"]) & set(singles["mut_id"])
    if overlap:
        raise ValueError(f"{len(overlap)} variants are in both tables, e.g. {sorted(overlap)[:3]}")

    table = pd.concat([campaign, singles], ignore_index=True)
    table["n_mut"] = table["Nham_aa"].astype(int)
    table["mut_list"] = [
        [] if mut_id == "WT" else mut_id.split("-") for mut_id in table["mut_id"]
    ]
    print(f"{len(campaign)} campaign variants and {len(singles)} of Max's singles")
    return table, fit


def one_substitution_apart(seq1, seq2):
    return len(seq1) == len(seq2) and sum(a != b for a, b in zip(seq1, seq2)) == 1


# --------------------------------------------------------------------------------------
# Expectations and epsilon
# --------------------------------------------------------------------------------------
def interactions(table):
    """One record per epsilon: a daughter, the two parts that predict it, and the shortfall.

    Doubles are predicted from their two singles and yield one record drawn as two edges;
    everything larger is predicted from each measured (n-1) parent plus the added substitution
    and yields one record per route.
    """
    metric = dict(zip(table["mut_id"], table["metric"]))
    kcat = dict(zip(table["mut_id"], table["kcat_app"]))
    source = dict(zip(table["mut_id"], table["source"]))
    singles = {m for m, n in zip(table["mut_id"], table["n_mut"]) if n == 1}

    records = []
    for muts, daughter, n in zip(table["mut_list"], table["mut_id"], table["n_mut"]):
        if n < 2:
            continue

        if n == 2:
            if not all(m in singles for m in muts):
                continue
            background, added = sorted(muts)
            edges = [tuple(sorted((m, daughter))) for m in muts]
            records.append(_record(daughter, n, "pair", background, added,
                                   metric, kcat, source, edges))
            continue

        for parent_muts in combinations(muts, n - 1):
            background = "-".join(parent_muts)
            remaining = set(muts) - set(parent_muts)
            added = remaining.pop()
            if background not in metric or added not in singles:
                continue
            edges = [tuple(sorted((background, daughter)))]
            records.append(_record(daughter, n, "background", background, added,
                                   metric, kcat, source, edges))

    return pd.DataFrame(records)


def _record(daughter, n, decomposition, background, added, metric, kcat, source, edges):
    expected_m = metric[background] + metric[added]
    epsilon = metric[daughter] - expected_m
    return {
        "daughter": daughter,
        "n_mutations": n,
        "decomposition": decomposition,
        "background": background,
        "added_mutation": added,
        "background_source": source[background],
        "added_mutation_source": source[added],
        "kcat_app_observed": kcat[daughter],
        "kcat_app_background": kcat[background],
        "kcat_app_added_mutation": kcat[added],
        "kcat_app_expected": 10 ** (expected_m + np.log10(kcat["WT"])),
        "epsilon_log10": epsilon,
        "fold_change_over_expected": 10 ** epsilon,
        "n_graph_edges": len(edges),
        "graph_edges": edges,
    }


# --------------------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------------------
def labelled_nodes():
    names = {"WT"}
    parts = LABELLED_TRAJECTORY.split("-")
    for size in range(1, len(parts) + 1):
        names.update("-".join(combo) for combo in combinations(parts, size))
    return names


def build_graph(nodes, epsilon_by_edge, eps_window):
    labelled = labelled_nodes()
    graph = nx.MultiGraph()
    for mut_id in nodes["mut_id"]:
        graph.add_node(mut_id)

    rows = nodes.reset_index(drop=True)
    for i, row_i in rows.iterrows():
        for j, row_j in rows.iterrows():
            if row_i["n_mut"] >= row_j["n_mut"]:
                continue
            if not one_substitution_apart(row_i["aa_seq"], row_j["aa_seq"]):
                continue
            edge = tuple(sorted((row_i["mut_id"], row_j["mut_id"])))
            if edge in epsilon_by_edge:
                shade = (epsilon_by_edge[edge] + eps_window) / (2 * eps_window)
                colour, style = EDGE_CMAP(shade), "solid"
                width = 2.0 if edge[0] in labelled and edge[1] in labelled else 0.6
            else:
                colour, style, width = (0, 0, 0, 1), "dashed", 0.5
            graph.add_edge(*edge, colour=colour, width=width, style=style)

    graph.remove_nodes_from([n for n, d in dict(graph.degree()).items() if d == 0])
    if "WT" not in graph:
        raise RuntimeError("The wild type is not in the graph.")
    reachable = set(nx.single_source_shortest_path_length(graph, "WT"))
    graph.remove_nodes_from(set(graph.nodes()) - reachable)
    graph.remove_nodes_from([n for n, d in dict(graph.degree()).items() if d <= MIN_DEGREE])
    return graph


def draw(graph, kcat_by_node, node_window, eps_window, kcat_wt):
    labelled = labelled_nodes()
    node_vmin, node_vmax = node_window
    cmap = plt.colormaps.get_cmap(NODE_CMAP)

    positions = nx.bfs_layout(graph, start="WT")
    positions = {node: (x * 3, -y * 3) for node, (x, y) in positions.items()}

    node_colours = [
        cmap((np.log10(kcat_by_node[node]) - node_vmin) / (node_vmax - node_vmin))
        if node in kcat_by_node else (0.5, 0.5, 0.5, 1)
        for node in graph.nodes()
    ]

    plt.close()
    plt.figure(figsize=(25, 15))
    nx.draw_networkx_edges(
        graph, positions,
        edge_color=list(nx.get_edge_attributes(graph, "colour").values()),
        width=list(nx.get_edge_attributes(graph, "width").values()),
        style=list(nx.get_edge_attributes(graph, "style").values()),
        alpha=0.5, connectionstyle="arc3",
    )
    nx.draw_networkx_nodes(graph, positions, node_size=10, node_color=node_colours)
    nx.draw_networkx_labels(
        graph, {k: (x - 0.001, y) for k, (x, y) in positions.items()},
        {node: node for node in graph.nodes() if node in labelled},
        font_size=8, horizontalalignment="right", verticalalignment="center",
    )

    epsilon_bar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=EDGE_CMAP, norm=plt.Normalize(-eps_window, eps_window)),
        ax=plt.gca(), orientation="vertical", shrink=0.25, pad=0.02,
        label=r"$\varepsilon$ (log$_{10}$ observed / expected $k_\mathrm{cat,app}$)",
    )
    epsilon_bar.ax.axhline(0, color="black", linewidth=0.6)

    kcat_bar = plt.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(node_vmin, node_vmax)),
        ax=plt.gca(), orientation="vertical", shrink=0.25, pad=0.002,
        label=r"Observed $k_\mathrm{cat,app}$ (s$^{-1}$)",
    )
    ticks = [t for t in KCAT_TICKS if node_vmin <= np.log10(t) <= node_vmax]
    kcat_bar.set_ticks(np.log10(ticks))
    kcat_bar.set_ticklabels([f"{t:g}" for t in ticks])
    kcat_bar.ax.axhline(np.log10(kcat_wt), color="black", linewidth=0.6)

    plt.axis("off")
    for suffix in ("svg", "png"):
        plt.savefig(THIS_DIR / f"connectogram.{suffix}", dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------------------
def main():
    table, fit = variants()

    kcat_wt = float(table.loc[table["mut_id"] == "WT", "kcat_app"].iloc[0])
    table["metric"] = np.log10(table["kcat_app"] / kcat_wt)

    # The conversion is log10(kcat_app) = (b / ln 10) * fitness + log10(a), so the fitness
    # colour window maps onto the kcat(app) axis without restating any constant.
    slope = fit["b"] / np.log(10)
    node_window = (
        -FITNESS_COLOUR_WINDOW * slope + np.log10(fit["a"]),
        FITNESS_COLOUR_WINDOW * slope + np.log10(fit["a"]),
    )
    eps_window = FITNESS_COLOUR_WINDOW * slope
    print(
        f"colour windows: kcat(app) {10 ** node_window[0]:.2f} to {10 ** node_window[1]:.1f} s-1, "
        f"epsilon +/- {eps_window:.3f} (wild type {kcat_wt:.2f} s-1)"
    )

    computed = interactions(table)
    pairs = computed[computed["decomposition"] == "pair"]
    substitutions = {s for m in computed["daughter"] for s in m.split("-")}
    possible = int(table.loc[table["n_mut"] >= 2, "n_mut"].sum())
    print(
        f"epsilon quantified for {len(computed)} steps "
        f"({len(pairs)} pairwise, {len(computed) - len(pairs)} into a measured background) "
        f"over {computed['daughter'].nunique()} variants and {len(substitutions)} substitutions; "
        f"{int(computed['n_graph_edges'].sum())} of {possible} possible splits resolve"
    )

    higher_order = table[table["n_mut"] >= 2]
    if GATE == "kcat":
        passes = higher_order["kcat_app"] > higher_order["kcat_app_error"]
    else:
        passes = higher_order["fitness_final"] > higher_order["sigma_final"]
    print(f'gate "{GATE}": {int(passes.sum())} of {len(higher_order)} higher-order variants pass')

    # A higher-order variant is a node only if at least one route into it could be scored;
    # a single mutant only if it sits inside one of those variants. Max's scan contributes
    # 1,122 singles and all but a handful are unrelated to this campaign.
    gated = set(higher_order.loc[passes, "mut_id"])
    kept = gated & set(computed["daughter"])
    inside_a_kept_double = {
        substitution
        for mut_id in kept
        if mut_id.count("-") == 1
        for substitution in mut_id.split("-")
    }
    kept |= inside_a_kept_double | {"WT"}
    drawable = computed[
        computed["daughter"].isin(kept) & computed["background"].isin(kept)
    ]
    epsilon_by_edge = {
        edge: epsilon
        for edges, epsilon in zip(drawable["graph_edges"], drawable["epsilon_log10"])
        for edge in edges
    }

    nodes = table[table["mut_id"].isin(kept)]
    graph = build_graph(nodes, epsilon_by_edge, eps_window)
    drawn = {
        tuple(sorted((u, v)))
        for u, v, data in graph.edges(data=True)
        if data["style"] == "solid"
    }
    print(
        f"graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges "
        f"({len(drawn)} carrying epsilon, {len(graph.edges) - len(drawn)} dashed); "
        "nodes per column "
        + str(sorted(Counter(0 if n == "WT" else n.count("-") + 1 for n in graph.nodes).items()))
    )

    computed = computed.copy()
    computed["in_figure"] = [
        all(edge in drawn for edge in edges) for edges in computed["graph_edges"]
    ]
    computed = computed.drop(columns="graph_edges").sort_values(
        ["n_mutations", "daughter", "background"]
    )
    computed.to_csv(THIS_DIR / "epistasis_interactions.csv", index=False)
    print(f"{int(computed['in_figure'].sum())} of {len(computed)} epsilon values are drawn")

    draw(graph, dict(zip(table["mut_id"], table["kcat_app"])), node_window, eps_window, kcat_wt)
    print("wrote connectogram.svg, connectogram.png and epistasis_interactions.csv")


if __name__ == "__main__":
    main()
