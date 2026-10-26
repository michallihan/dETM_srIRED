"""
Figure 4D: the measured network beside what the models made of it.

Left, every single-mutation step in the network coloured by the direction the measurement
says it went, green up and red down. Right, how often each method called that direction
correctly, against the 0.5 line it would reach by guessing. Nine methods, from zero-shot
protein language models through the MSA-based ones to models trained on the campaign's own
early rounds, and none of them clears the line.

Reads the tables 11_ writes; the greedy-walk row is dropped, being 1.0 by construction.

Outputs, written next to this script:
    figure_4d_model_accuracy.pdf / .png
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

SUMMARY_CSV = THIS_DIR / "edge_prediction_summary.csv"
EDGE_CSV = THIS_DIR / "edge_predictions.csv"
OUTPUT_PDF = THIS_DIR / "figure_4d_model_accuracy.pdf"
OUTPUT_PNG = THIS_DIR / "figure_4d_model_accuracy.png"

SPRING_SEED = 5000


def main():
    for required in (SUMMARY_CSV, EDGE_CSV):
        if not required.exists():
            raise SystemExit(f"Run 11_benchmark_edges.py first: {required.name} is missing.")

    summary = pd.read_csv(SUMMARY_CSV).query("method != 'kcat_app' and n_edges_scored > 0").copy()
    edges = pd.read_csv(EDGE_CSV)
    graph = nx.from_pandas_edgelist(edges, "source", "target")
    layout = nx.spring_layout(graph, seed=SPRING_SEED)

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 9

    fig, (network_ax, ax) = plt.subplots(
        1, 2, figsize=(11.2, 4.0), gridspec_kw={"width_ratios": [1.05, 1]}
    )

    edge_colors = ["#55a868" if positive else "#c44e52" for positive in edges["kcat_app_delta"].gt(0)]
    nx.draw_networkx_edges(graph, layout, ax=network_ax, edge_color=edge_colors, alpha=0.38, width=0.7)
    nx.draw_networkx_nodes(graph, layout, ax=network_ax, node_color="#8a8a8a", node_size=9, linewidths=0)
    network_ax.set_title("Observed direction of Δkcat(app)")
    network_ax.set_axis_off()

    ax.bar(range(len(summary)), summary["accuracy"], color="#5b84b1")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1, label="Random baseline")
    ax.set_xticks(range(len(summary)), summary["label"], rotation=40, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy: correct Δkcat(app) direction")
    ax.grid(True, axis="y", color="#E7E7E7", linewidth=1)
    ax.tick_params(axis="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUTPUT_PDF)
    fig.savefig(OUTPUT_PNG, dpi=300)
    plt.close(fig)

    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_PNG}")
    print(summary[["label", "n_edges_scored", "accuracy"]].to_string(index=False))


if __name__ == "__main__":
    main()
