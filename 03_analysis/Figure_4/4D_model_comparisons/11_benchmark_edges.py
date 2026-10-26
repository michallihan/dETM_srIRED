"""
Can any of these models tell which way a single mutation moves the enzyme?

Every pair of variants in the low-confidence set that differ by one substitution, where one's
mutations are a subset of the other's, is an edge: a single step the campaign could have taken.
The measured fitness says which way that step went. Each method scores both ends, and the sign
of its difference is its prediction of the direction. Accuracy is the fraction of steps it
gets right, over the 984 edges connected to wild type.

Two controls sit beside each accuracy. The random-sign baseline is 0.5. The permutation
baseline reshuffles a method's own scores across the network 2000 times and re-scores, which
asks whether the network's shape alone could produce the result; it lands at 0.500 for every
method, so it does not.

The supervised methods of 05_ to 07_ train on variants with at most two substitutions, so
edges touching that pool are excluded from their accuracy and they are scored on fewer edges.
The "Greedy walk" row scores the measured kcat(app) against itself and is 1.0 by construction:
it is a check that the edge bookkeeping is right, not a result, and 12_ leaves it out.

Reads whichever score tables are present in model_scores/, so it can be rerun after any single
scoring script. 01_ must have run.

Outputs, written next to this script:
    variant_table_with_model_scores.csv
    edge_predictions.csv
    edge_prediction_summary.csv
    method_roc.csv
    method_sign_agreement.csv / method_sign_agreement_n_edges.csv
    greedy_trajectories.csv
    edge_predictability_network.pdf / .png / .svg
    network_panels/<method>.pdf / .png / .svg
    method_roc.pdf / .png / .svg
    method_sign_agreement.pdf / .png / .svg
    greedy_trajectories.pdf / .png / .svg
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import argparse
import math

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

import model_comparison as mc  # noqa: E402

ESM_SCORE_CSV = mc.ESM2_SCORES_CSV
ESMC_SCORE_CSV = mc.ESMC_SCORES_CSV
PROTEINMPNN_SCORE_CSV = mc.PROTEINMPNN_SCORES_CSV
EVMUTATION_SCORE_CSV = mc.EVMUTATION_SCORES_CSV
THERMOMPNN_SCORE_CSV = mc.THERMOMPNN_SCORES_CSV
MULTIEVOLVE_SCORE_CSV = mc.MULTIEVOLVE_SCORES_CSV
EVOLVEPRO_SCORE_CSV = mc.EVOLVEPRO_ONEHOT_SCORES_CSV
EVOLVEPRO_ESM_SCORE_CSV = mc.EVOLVEPRO_ESM2_SCORES_CSV
GEMME_OFFICIAL_SCORE_CSV = mc.GEMME_SCORES_CSV

NETWORK_PANELS_DIR = THIS_DIR / "network_panels"
OUTPUT_DATASET = THIS_DIR / "variant_table_with_model_scores.csv"
OUTPUT_EDGE_CSV = THIS_DIR / "edge_predictions.csv"
OUTPUT_SUMMARY_CSV = THIS_DIR / "edge_prediction_summary.csv"
OUTPUT_NETWORK_STEM = THIS_DIR / "edge_predictability_network"
OUTPUT_TRAJECTORY_CSV = THIS_DIR / "greedy_trajectories.csv"
OUTPUT_TRAJECTORY_STEM = THIS_DIR / "greedy_trajectories"
OUTPUT_ROC_CSV = THIS_DIR / "method_roc.csv"
OUTPUT_ROC_STEM = THIS_DIR / "method_roc"
OUTPUT_AGREEMENT_CSV = THIS_DIR / "method_sign_agreement.csv"
OUTPUT_AGREEMENT_STEM = THIS_DIR / "method_sign_agreement"

SPRING_SEED = 5000
RANDOM_SEED = 20260727

METHOD_SPECS = {
    "kcat_app": {
        "label": "Greedy walk",
        "score_col": "kcat_app",
        "higher_is_better": True,
    },
    "esm_ppl_star": {
        "label": "ESM PPL*",
        "score_col": "esm_ppl_naturalness",
        "higher_is_better": True,
    },
    "esmc_ppl_star": {
        "label": "ESM-C PPL*",
        "score_col": "esmc_ppl_naturalness",
        "higher_is_better": True,
    },
    "proteinmpnn": {
        "label": "ProteinMPNN",
        "score_col": "proteinmpnn_naturalness",
        "higher_is_better": True,
    },
    "evmutation": {
        "label": "EVmutation",
        "score_col": "evmutation_naturalness",
        "higher_is_better": True,
    },
    "thermompnn": {
        "label": "ThermoMPNN",
        "score_col": "thermompnn_naturalness",
        "higher_is_better": True,
    },
    "multievolve_fcn": {
        "label": "MULTI-evolve FCN",
        "score_col": "multievolve_fcn_log_kcat_pred",
        "higher_is_better": True,
        "exclude_edges_touching_training_load_le": 2,
    },
    "evolvepro_rf_onehot": {
        "label": "EVOLVEpro RF (one-hot)",
        "score_col": "evolvepro_rf_onehot_log_kcat_pred",
        "higher_is_better": True,
        "exclude_edges_touching_training_load_le": 2,
    },
    "evolvepro_rf_esm2_650m": {
        "label": "EVOLVEpro RF (ESM2-650M)",
        "score_col": "evolvepro_rf_esm2_650m_log_kcat_pred",
        "higher_is_better": True,
        "exclude_edges_touching_training_load_le": 2,
    },
    "gemme_official": {
        "label": "GEMME",
        "score_col": "gemme_official_evolcombi_score",
        "higher_is_better": True,
    },
    "msa_conservation": {
        "label": "MSA conservation",
        "score_col": "msa_conservation_log_prob",
        "higher_is_better": True,
    },
}

SIGN_AGREEMENT_METHOD_GROUPS = [
    ("PLMs zero-shot", ["esm_ppl_star", "esmc_ppl_star"]),
    ("MSA-based methods", ["msa_conservation", "evmutation", "gemme_official"]),
    ("Structure-sequence-function", ["thermompnn"]),
    ("Supervised", ["multievolve_fcn", "evolvepro_rf_onehot", "evolvepro_rf_esm2_650m"]),
]

FOUNDER_NODES = [
    "WT",
    "T241A",
    "S68P-T241A",
    "S68P-P203Q-T241A",
    "S68P-T241A-H244Y",
    "S68P-P203Q-T241A-H244Y",
    "A28T-S68P-P203Q-T241A-H244Y",
]
FOUNDER_EDGES = [
    ("WT", "T241A"),
    ("T241A", "S68P-T241A"),
    ("S68P-T241A", "S68P-P203Q-T241A"),
    ("S68P-T241A", "S68P-T241A-H244Y"),
    ("S68P-P203Q-T241A", "S68P-P203Q-T241A-H244Y"),
    ("S68P-T241A-H244Y", "S68P-P203Q-T241A-H244Y"),
    ("S68P-P203Q-T241A-H244Y", "A28T-S68P-P203Q-T241A-H244Y"),
]
FOUNDER_TRAJECTORY = [
    "WT",
    "T241A",
    "S68P-T241A",
    "S68P-T241A-H244Y",
    "S68P-P203Q-T241A-H244Y",
    "A28T-S68P-P203Q-T241A-H244Y",
]
def build_graph(df: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    mutation_sets: dict[str, frozenset[str]] = {}
    for row in df.itertuples(index=False):
        name = str(row.mut_id)
        muts = mc.mutation_set(name)
        mutation_sets[name] = muts
        graph.add_node(
            name,
            fitness_final=float(row.fitness_final),
            sigma_final=float(row.sigma_final),
            Nham_aa=int(row.Nham_aa),
        )

    by_size: dict[int, list[str]] = {}
    for name, muts in mutation_sets.items():
        by_size.setdefault(len(muts), []).append(name)

    for size, smaller_nodes in by_size.items():
        for smaller in smaller_nodes:
            smaller_set = mutation_sets[smaller]
            for larger in by_size.get(size + 1, []):
                larger_set = mutation_sets[larger]
                if smaller_set < larger_set:
                    graph.add_edge(smaller, larger)

    if "WT" not in graph:
        raise KeyError("WT is not present in the low-confidence dataset")
    return graph.subgraph(nx.node_connected_component(graph, "WT")).copy()


def add_optional_score_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if ESM_SCORE_CSV.exists():
        esm = pd.read_csv(ESM_SCORE_CSV)
        if "single_pass_mean_log_probability" in esm.columns:
            esm["esm_ppl_naturalness"] = esm["single_pass_mean_log_probability"]
        elif "single_pass_ppl_star" in esm.columns:
            esm["esm_ppl_naturalness"] = -esm["single_pass_ppl_star"]
        out = out.merge(
            esm[["variant_id", "esm_ppl_naturalness"]].rename(columns={"variant_id": "mut_id"}),
            on="mut_id",
            how="left",
        )
    if ESMC_SCORE_CSV.exists():
        esmc = pd.read_csv(ESMC_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in esmc.columns else "mut_id"
        score_candidates = [
            "esmc_ppl_naturalness",
            "single_pass_mean_log_probability",
            "mean_log_probability",
            "score",
        ]
        score_col = next((col for col in score_candidates if col in esmc.columns), None)
        if score_col is None:
            if "single_pass_ppl_star" in esmc.columns:
                esmc["esmc_ppl_naturalness"] = -esmc["single_pass_ppl_star"]
                score_col = "esmc_ppl_naturalness"
            else:
                raise ValueError(f"No supported ESM-C score column found in {ESMC_SCORE_CSV}")
        esmc = esmc[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "esmc_ppl_naturalness"}
        )
        out = out.merge(esmc, on="mut_id", how="left")
    if PROTEINMPNN_SCORE_CSV.exists():
        mpnn = pd.read_csv(PROTEINMPNN_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in mpnn.columns else "mut_id"
        score_candidates = [
            "proteinmpnn_naturalness",
            "mean_log_probability",
            "score",
            "global_score",
            "sequence_score",
        ]
        score_col = next((col for col in score_candidates if col in mpnn.columns), None)
        if score_col is None:
            raise ValueError(f"No supported ProteinMPNN score column found in {PROTEINMPNN_SCORE_CSV}")
        mpnn = mpnn[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "proteinmpnn_naturalness"}
        )
        out = out.merge(mpnn, on="mut_id", how="left")
    if EVMUTATION_SCORE_CSV.exists():
        evmutation = pd.read_csv(EVMUTATION_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in evmutation.columns else "mut_id"
        score_candidates = [
            "evmutation_naturalness",
            "delta_E",
            "delta_e",
            "effect_prediction_epistatic",
            "evmutation_score",
            "score",
        ]
        score_col = next((col for col in score_candidates if col in evmutation.columns), None)
        if score_col is None:
            raise ValueError(f"No supported EVmutation score column found in {EVMUTATION_SCORE_CSV}")
        evmutation = evmutation[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "evmutation_naturalness"}
        )
        out = out.merge(evmutation, on="mut_id", how="left")
    if THERMOMPNN_SCORE_CSV.exists():
        thermompnn = pd.read_csv(THERMOMPNN_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in thermompnn.columns else "mut_id"
        score_candidates = [
            "thermompnn_naturalness",
            "negative_sum_ddg_pred",
            "score",
        ]
        score_col = next((col for col in score_candidates if col in thermompnn.columns), None)
        if score_col is None:
            raise ValueError(f"No supported ThermoMPNN score column found in {THERMOMPNN_SCORE_CSV}")
        thermompnn = thermompnn[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "thermompnn_naturalness"}
        )
        out = out.merge(thermompnn, on="mut_id", how="left")
    if MULTIEVOLVE_SCORE_CSV.exists():
        multievolve = pd.read_csv(MULTIEVOLVE_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in multievolve.columns else "mut_id"
        score_candidates = [
            "multievolve_fcn_log_kcat_pred",
            "multievolve_fcn_kcat_pred",
            "score",
        ]
        score_col = next((col for col in score_candidates if col in multievolve.columns), None)
        if score_col is None:
            raise ValueError(f"No supported MULTI-evolve score column found in {MULTIEVOLVE_SCORE_CSV}")
        multievolve = multievolve[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "multievolve_fcn_log_kcat_pred"}
        )
        out = out.merge(multievolve, on="mut_id", how="left")
    if EVOLVEPRO_SCORE_CSV.exists():
        evolvepro = pd.read_csv(EVOLVEPRO_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in evolvepro.columns else "mut_id"
        score_candidates = [
            "evolvepro_rf_onehot_log_kcat_pred",
            "evolvepro_rf_onehot_kcat_pred",
            "score",
        ]
        score_col = next((col for col in score_candidates if col in evolvepro.columns), None)
        if score_col is None:
            raise ValueError(f"No supported EVOLVEpro score column found in {EVOLVEPRO_SCORE_CSV}")
        evolvepro = evolvepro[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "evolvepro_rf_onehot_log_kcat_pred"}
        )
        out = out.merge(evolvepro, on="mut_id", how="left")
    if EVOLVEPRO_ESM_SCORE_CSV.exists():
        evolvepro_esm = pd.read_csv(EVOLVEPRO_ESM_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in evolvepro_esm.columns else "mut_id"
        score_candidates = [
            "evolvepro_rf_esm2_650m_log_kcat_pred",
            "evolvepro_rf_esm2_650m_kcat_pred",
            "score",
        ]
        score_col = next((col for col in score_candidates if col in evolvepro_esm.columns), None)
        if score_col is None:
            raise ValueError(f"No supported EVOLVEpro ESM score column found in {EVOLVEPRO_ESM_SCORE_CSV}")
        evolvepro_esm = evolvepro_esm[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "evolvepro_rf_esm2_650m_log_kcat_pred"}
        )
        out = out.merge(evolvepro_esm, on="mut_id", how="left")
    if GEMME_OFFICIAL_SCORE_CSV.exists():
        gemme = pd.read_csv(GEMME_OFFICIAL_SCORE_CSV)
        id_col = "variant_id" if "variant_id" in gemme.columns else "mut_id"
        score_candidates = [
            "gemme_official_evolcombi_score",
            "gemme_score",
            "score",
        ]
        score_col = next((col for col in score_candidates if col in gemme.columns), None)
        if score_col is None:
            raise ValueError(f"No supported GEMME score column found in {GEMME_OFFICIAL_SCORE_CSV}")
        gemme = gemme[[id_col, score_col]].rename(
            columns={id_col: "mut_id", score_col: "gemme_official_evolcombi_score"}
        )
        out = out.merge(gemme, on="mut_id", how="left")
    return out


def random_permutation_baseline(
    graph: nx.Graph,
    scores: dict[str, float],
    true_signs: list[int],
    directed_edges: list[tuple[str, str]],
    n_permutations: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    nodes = list(scores)
    values = np.array([scores[node] for node in nodes], dtype=float)
    accs = []
    for _ in range(n_permutations):
        rng.shuffle(values)
        perm = dict(zip(nodes, values))
        correct = 0
        usable = 0
        for true_sign, (source, target) in zip(true_signs, directed_edges):
            pred_delta = perm[target] - perm[source]
            pred_sign = int(np.sign(pred_delta))
            if pred_sign == 0:
                continue
            usable += 1
            correct += pred_sign == true_sign
        if usable:
            accs.append(correct / usable)
    return float(np.mean(accs)), float(np.std(accs, ddof=1))


def edge_prediction_table(graph: nx.Graph, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    node_rows = df.set_index("mut_id").to_dict("index")
    directed_edges: list[tuple[str, str]] = []
    for a, b in graph.edges:
        source, target = (a, b) if len(mc.mutation_set(a)) <= len(mc.mutation_set(b)) else (b, a)
        directed_edges.append((source, target))

    rows = []
    for source, target in directed_edges:
        true_delta = node_rows[target]["fitness_final"] - node_rows[source]["fitness_final"]
        true_sign = int(np.sign(true_delta))
        if true_sign == 0:
            continue
        row = {
            "source": source,
            "target": target,
            "source_distance": len(mc.mutation_set(source)),
            "target_distance": len(mc.mutation_set(target)),
            "delta_fitness": true_delta,
            "fitness_direction": "increase" if true_delta > 0 else "decrease",
        }
        for method, spec in METHOD_SPECS.items():
            score_col = spec["score_col"]
            source_score = node_rows[source].get(score_col, np.nan)
            target_score = node_rows[target].get(score_col, np.nan)
            max_training_load = spec.get("exclude_edges_touching_training_load_le")
            if max_training_load is not None and (
                row["source_distance"] <= max_training_load or row["target_distance"] <= max_training_load
            ):
                row[f"{method}_delta"] = np.nan
                row[f"{method}_predictable"] = pd.NA
                continue
            if pd.isna(source_score) or pd.isna(target_score):
                row[f"{method}_delta"] = np.nan
                row[f"{method}_predictable"] = pd.NA
                continue
            pred_delta = float(target_score) - float(source_score)
            pred_sign = int(np.sign(pred_delta))
            row[f"{method}_delta"] = pred_delta
            row[f"{method}_predictable"] = bool(pred_sign == true_sign) if pred_sign else pd.NA
        rows.append(row)

    edge_df = pd.DataFrame(rows)
    summary_rows = []
    true_signs = [int(np.sign(row["delta_fitness"])) for _, row in edge_df.iterrows()]
    for method, spec in METHOD_SPECS.items():
        pred_col = f"{method}_predictable"
        delta_col = f"{method}_delta"
        sub = edge_df.dropna(subset=[pred_col, delta_col]).copy()
        if sub.empty:
            summary_rows.append(
                {
                    "method": method,
                    "label": spec["label"],
                    "n_edges_scored": 0,
                    "accuracy": np.nan,
                    "random_sign_baseline": 0.5,
                    "permutation_baseline_mean": np.nan,
                    "permutation_baseline_sd": np.nan,
                    "accuracy_minus_random_sign": np.nan,
                }
            )
            continue
        method_scores = df.dropna(subset=[spec["score_col"]]).set_index("mut_id")[spec["score_col"]].to_dict()
        sub_edges = list(zip(sub["source"], sub["target"]))
        sub_true = [int(np.sign(v)) for v in sub["delta_fitness"]]
        perm_mean, perm_sd = random_permutation_baseline(graph, method_scores, sub_true, sub_edges)
        accuracy = float(sub[pred_col].astype(bool).mean())
        summary_rows.append(
            {
                "method": method,
                "label": spec["label"],
                "n_edges_scored": int(len(sub)),
                "accuracy": accuracy,
                "random_sign_baseline": 0.5,
                "permutation_baseline_mean": perm_mean,
                "permutation_baseline_sd": perm_sd,
                "accuracy_minus_random_sign": accuracy - 0.5,
            }
        )
    return edge_df, pd.DataFrame(summary_rows)


def normalized_layout(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    pos = nx.spring_layout(graph, seed=SPRING_SEED, k=0.11, iterations=120)
    xs = [xy[0] for xy in pos.values()]
    ys = [xy[1] for xy in pos.values()]
    x_mid = (min(xs) + max(xs)) / 2
    y_mid = (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    return {node: ((x - x_mid) / span * 2.0, (y - y_mid) / span * 2.0) for node, (x, y) in pos.items()}


def draw_network_panels(graph: nx.Graph, edge_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    methods = [row.method for row in summary_df.itertuples(index=False) if row.n_edges_scored > 0]
    if not methods:
        return
    pos = normalized_layout(graph)
    ncols = 2
    nrows = math.ceil(len(methods) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 5 * nrows), squeeze=False)
    plt.rcParams["font.family"] = "Arial"
    edge_lookup = {(row.source, row.target): row for row in edge_df.itertuples(index=False)}
    directed_edges = list(edge_lookup)

    def draw_founders(ax: plt.Axes, method: str) -> None:
        founder_edges = [edge for edge in FOUNDER_EDGES if graph.has_edge(*edge)]
        founder_nodes = [node for node in FOUNDER_NODES if node in graph]
        pred_col = f"{method}_predictable"
        predictable_edges = []
        unpredictable_edges = []
        unscored_edges = []
        for edge in founder_edges:
            source, target = edge if edge in edge_lookup else (edge[1], edge[0])
            row = edge_lookup.get((source, target))
            if row is None:
                unscored_edges.append(edge)
                continue
            val = getattr(row, pred_col)
            if pd.isna(val):
                unscored_edges.append(edge)
            elif bool(val):
                predictable_edges.append(edge)
            else:
                unpredictable_edges.append(edge)
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=unscored_edges,
            edge_color="#8e8e8e",
            width=2.6,
            alpha=1.0,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=predictable_edges,
            edge_color="#25824f",
            width=2.6,
            alpha=1.0,
        )
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            edgelist=unpredictable_edges,
            edge_color="#c73b3d",
            width=2.6,
            alpha=1.0,
        )
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            nodelist=founder_nodes,
            node_size=32,
            node_color="black",
            linewidths=0,
        )
        for node in founder_nodes:
            x, y = pos[node]
            label = node.replace("S68P-P203Q-T241A-H244Y", "S68P-P203Q-\nT241A-H244Y")
            label = label.replace("A28T-S68P-P203Q-T241A-H244Y", "A28T-S68P-P203Q-\nT241A-H244Y")
            ax.annotate(
                label,
                xy=(x, y),
                xytext=(5, 5),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="black",
                linespacing=0.85,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.74},
            )

    def draw_method(ax: plt.Axes, method: str) -> None:
        pred_col = f"{method}_predictable"
        predictable = []
        unpredictable = []
        unscored = []
        for edge in directed_edges:
            val = getattr(edge_lookup[edge], pred_col)
            if pd.isna(val):
                unscored.append(edge)
            elif bool(val):
                predictable.append(edge)
            else:
                unpredictable.append(edge)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=unscored, edge_color="#d7d7d7", width=0.25, alpha=0.30)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=predictable, edge_color="#25824f", width=0.55, alpha=0.85)
        nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=unpredictable, edge_color="#c73b3d", width=0.55, alpha=0.82)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=5, node_color="#6f6f6f", alpha=0.82, linewidths=0)
        draw_founders(ax, method)
        row = summary_df.loc[summary_df["method"].eq(method)].iloc[0]
        ax.set_title(f"{row['label']}\naccuracy={row['accuracy']:.3f}, n={int(row['n_edges_scored'])}", fontsize=14)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax, method in zip(axes.flat, methods):
        draw_method(ax, method)
    for ax in axes.flat[len(methods):]:
        ax.axis("off")
    fig.tight_layout()
    for suffix, kwargs in {".pdf": {}, ".png": {"dpi": 300}, ".svg": {}}.items():
        fig.savefig(OUTPUT_NETWORK_STEM.with_suffix(suffix), **kwargs)
    plt.close(fig)

    for method in methods:
        fig, ax = plt.subplots(figsize=(6, 5.5))
        draw_method(ax, method)
        fig.tight_layout()
        method_stem = NETWORK_PANELS_DIR / method
        for suffix, kwargs in {".pdf": {}, ".png": {"dpi": 300}, ".svg": {}}.items():
            fig.savefig(method_stem.with_suffix(suffix), **kwargs)
        plt.close(fig)


def greedy_trajectory(graph: nx.Graph, node_rows: dict[str, dict], method: str, max_steps: int = 5) -> list[dict]:
    spec = METHOD_SPECS[method]
    score_col = spec["score_col"]
    current = "WT"
    rows = []
    for step in range(max_steps + 1):
        current_row = node_rows[current]
        rows.append(
            {
                "strategy": method,
                "label": spec["label"],
                "step": step,
                "variant": current,
                "selection_score": current_row.get(score_col, np.nan),
                "kcat_app": current_row["kcat_app"],
                "fitness_final": current_row["fitness_final"],
            }
        )
        if step == max_steps:
            break
        current_size = len(mc.mutation_set(current))
        candidates = [
            neighbor
            for neighbor in graph.neighbors(current)
            if len(mc.mutation_set(neighbor)) == current_size + 1
            and neighbor in node_rows
            and not pd.isna(node_rows[neighbor].get(score_col, np.nan))
        ]
        if not candidates:
            break
        current = max(candidates, key=lambda node: node_rows[node][score_col])
    return rows


def founder_trajectory(node_rows: dict[str, dict], max_steps: int = 5) -> list[dict]:
    rows = []
    for step, variant in enumerate(FOUNDER_TRAJECTORY[: max_steps + 1]):
        if variant not in node_rows:
            break
        current_row = node_rows[variant]
        rows.append(
            {
                "strategy": "founder",
                "label": "Founder trajectory",
                "step": step,
                "variant": variant,
                "selection_score": np.nan,
                "kcat_app": current_row["kcat_app"],
                "fitness_final": current_row["fitness_final"],
            }
        )
    return rows


def write_and_plot_trajectories(graph: nx.Graph, df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    node_rows = df.set_index("mut_id").to_dict("index")
    methods = [
        method
        for method in [
            "esm_ppl_star",
            "esmc_ppl_star",
            "msa_conservation",
            "evmutation",
            "thermompnn",
            "multievolve_fcn",
            "evolvepro_rf_onehot",
            "evolvepro_rf_esm2_650m",
            "gemme_official",
            "kcat_app",
        ]
        if int(summary_df.loc[summary_df["method"].eq(method), "n_edges_scored"].iloc[0]) > 0
    ]
    rows: list[dict] = []
    for method in methods:
        rows.extend(greedy_trajectory(graph, node_rows, method))
    rows.extend(founder_trajectory(node_rows))
    trajectory_df = pd.DataFrame(rows)
    trajectory_df.to_csv(OUTPUT_TRAJECTORY_CSV, index=False)

    colors = {
        "esm_ppl_star": "#3b6fb6",
        "esmc_ppl_star": "#00a1a7",
        "msa_conservation": "#7a4da3",
        "evmutation": "#c0782f",
        "thermompnn": "#c44e8b",
        "kcat_app": "#25824f",
        "founder": "black",
    }
    markers = {
        "esm_ppl_star": "o",
        "esmc_ppl_star": "v",
        "msa_conservation": "s",
        "evmutation": "P",
        "thermompnn": "X",
        "kcat_app": "^",
        "founder": "D",
    }

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for strategy, sub in trajectory_df.groupby("strategy", sort=False):
        ax.plot(
            sub["step"],
            sub["kcat_app"],
            marker=markers.get(strategy, "o"),
            linewidth=2.0,
            markersize=5.5,
            color=colors.get(strategy, "#555555"),
            label=sub["label"].iloc[0],
        )
        for row in sub.itertuples(index=False):
            if row.step in {0, int(sub["step"].max())}:
                label = row.variant.replace("A28T-S68P-P203Q-T241A-H244Y", "A28T+4")
                label = label.replace("S68P-P203Q-T241A-H244Y", "4-mut founder")
                ax.annotate(
                    label,
                    xy=(row.step, row.kcat_app),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7.5,
                    color=colors.get(strategy, "#555555"),
                )
    ax.set_xlabel("Evolution step")
    ax.set_ylabel("True kcat_app [s$^{-1}$]")
    ax.set_title("Greedy in silico trajectories evaluated by true kcat_app")
    ax.set_xlim(-0.15, 5.15)
    ax.set_xticks(range(0, 6))
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for suffix, kwargs in {".pdf": {}, ".png": {"dpi": 300}, ".svg": {}}.items():
        fig.savefig(OUTPUT_TRAJECTORY_STEM.with_suffix(suffix), **kwargs)
    plt.close(fig)


def scored_methods(summary_df: pd.DataFrame) -> list[str]:
    return [
        row.method
        for row in summary_df.itertuples(index=False)
        if int(row.n_edges_scored) > 0
    ]


def independent_scored_methods(summary_df: pd.DataFrame) -> list[str]:
    return [method for method in scored_methods(summary_df) if method != "kcat_app"]


def grouped_sign_agreement_methods(summary_df: pd.DataFrame) -> tuple[list[str], list[int]]:
    available = set(independent_scored_methods(summary_df))
    methods: list[str] = []
    boundaries: list[int] = []
    for _, group_methods in SIGN_AGREEMENT_METHOD_GROUPS:
        group_present = [method for method in group_methods if method in available]
        if not group_present:
            continue
        if methods:
            boundaries.append(len(methods) + 1)
        methods.extend(group_present)
    remaining = [method for method in independent_scored_methods(summary_df) if method not in methods]
    if remaining and methods:
        boundaries.append(len(methods) + 1)
    methods.extend(remaining)
    return methods, boundaries


def plot_method_rocs(edge_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    methods = independent_scored_methods(summary_df)
    y_true_all = (edge_df["kcat_app_delta"] > 0).astype(int)
    roc_rows = []

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color="#8a8a8a", linewidth=1.0, linestyle="--", label="Random")

    colors = {
        "kcat_app": "#25824f",
        "esm_ppl_star": "#3b6fb6",
        "esmc_ppl_star": "#00a1a7",
        "proteinmpnn": "#777777",
        "evmutation": "#c0782f",
        "thermompnn": "#c44e8b",
        "msa_conservation": "#7a4da3",
    }

    for method in methods:
        delta_col = f"{method}_delta"
        valid = edge_df[[delta_col]].join(edge_df["kcat_app_delta"].rename("true_kcat_app_delta")).dropna()
        if valid.empty:
            continue
        y_true = (valid["true_kcat_app_delta"] > 0).astype(int).to_numpy()
        scores = valid[delta_col].astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        label = METHOD_SPECS[method]["label"]
        ax.plot(fpr, tpr, linewidth=2.0, color=colors.get(method), label=f"{label} (AUC={roc_auc:.3f})")
        for f, t, threshold in zip(fpr, tpr, thresholds):
            roc_rows.append(
                {
                    "method": method,
                    "label": label,
                    "n_edges": len(valid),
                    "auc": roc_auc,
                    "fpr": f,
                    "tpr": t,
                    "threshold": threshold,
                }
            )

    pd.DataFrame(roc_rows).to_csv(OUTPUT_ROC_CSV, index=False)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Prediction of positive true Δkcat_app")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color="#e7e7e7", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    for suffix, kwargs in {".pdf": {}, ".png": {"dpi": 300}, ".svg": {}}.items():
        fig.savefig(OUTPUT_ROC_STEM.with_suffix(suffix), **kwargs)
    plt.close(fig)


def plot_sign_agreement_grid(edge_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    methods, boundaries = grouped_sign_agreement_methods(summary_df)
    sign_df = pd.DataFrame(index=edge_df.index)
    labels = {"true_kcat_app": "True Δkcat_app"}
    sign_df["true_kcat_app"] = np.sign(edge_df["kcat_app_delta"].astype(float))
    for method in methods:
        sign_df[method] = np.sign(edge_df[f"{method}_delta"].astype(float))
        labels[method] = METHOD_SPECS[method]["label"]

    cols = ["true_kcat_app"] + methods
    agreement = pd.DataFrame(index=cols, columns=cols, dtype=float)
    n_overlap = pd.DataFrame(index=cols, columns=cols, dtype=int)
    for a in cols:
        for b in cols:
            if a == b:
                sub_self = sign_df[[a]].replace(0, np.nan).dropna()
                n_overlap.loc[a, b] = len(sub_self)
                agreement.loc[a, b] = 1.0 if len(sub_self) else np.nan
                continue
            sub = sign_df[[a, b]].replace(0, np.nan).dropna()
            n_overlap.loc[a, b] = len(sub)
            agreement.loc[a, b] = float((sub[a] == sub[b]).mean()) if len(sub) else np.nan

    agreement_labeled = agreement.rename(index=labels, columns=labels)
    n_labeled = n_overlap.rename(index=labels, columns=labels)
    agreement_labeled.to_csv(OUTPUT_AGREEMENT_CSV)
    n_labeled.to_csv(OUTPUT_AGREEMENT_CSV.with_name(f"{OUTPUT_AGREEMENT_CSV.stem}_n_edges.csv"))

    plt.rcParams["font.family"] = "Arial"
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    matrix = agreement_labeled.to_numpy(dtype=float)
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(agreement_labeled.columns)))
    ax.set_yticks(range(len(agreement_labeled.index)))
    ax.set_xticklabels(agreement_labeled.columns, rotation=35, ha="right")
    ax.set_yticklabels(agreement_labeled.index)
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="#1f1f1f", linewidth=1.0)
        ax.axvline(boundary - 0.5, color="#1f1f1f", linewidth=1.0)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                text = "NA"
            else:
                text = f"{value:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color="black")
    ax.set_title("Pairwise agreement of edge-direction signs")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction same sign")
    fig.tight_layout()
    for suffix, kwargs in {".pdf": {}, ".png": {"dpi": 300}, ".svg": {}}.items():
        fig.savefig(OUTPUT_AGREEMENT_STEM.with_suffix(suffix), **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-network", action="store_true", help="Do not render network graph panels.")
    args = parser.parse_args()

    NETWORK_PANELS_DIR.mkdir(parents=True, exist_ok=True)

    df = mc.load_variant_table()
    for column in ("kcat_app", "kcat_app_error", "msa_conservation_log_prob"):
        if column not in df.columns:
            raise SystemExit(f"{mc.VARIANT_TABLE_CSV.name} lacks {column!r}; rerun 01_prepare_variant_table.py.")
    df = add_optional_score_columns(df)
    df.to_csv(OUTPUT_DATASET, index=False)

    graph = build_graph(df)
    edge_df, summary_df = edge_prediction_table(graph, df)
    edge_df.to_csv(OUTPUT_EDGE_CSV, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    plot_method_rocs(edge_df, summary_df)
    plot_sign_agreement_grid(edge_df, summary_df)
    write_and_plot_trajectories(graph, df, summary_df)
    if not args.skip_network:
        draw_network_panels(graph, edge_df, summary_df)

    print(f"Wrote {OUTPUT_DATASET} ({len(df)} rows)")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} one-AA edges connected to WT")
    print(f"Wrote {OUTPUT_EDGE_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_CSV}")
    print(f"Wrote {OUTPUT_ROC_CSV}")
    print(f"Wrote {OUTPUT_ROC_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_AGREEMENT_CSV}")
    print(f"Wrote {OUTPUT_AGREEMENT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_TRAJECTORY_CSV}")
    print(f"Wrote {OUTPUT_TRAJECTORY_STEM.with_suffix('.pdf')}")
    if not args.skip_network:
        print(f"Wrote {OUTPUT_NETWORK_STEM.with_suffix('.pdf')}")
        print(f"Wrote {OUTPUT_NETWORK_STEM.with_suffix('.png')}")
        print(f"Wrote {OUTPUT_NETWORK_STEM.with_suffix('.svg')}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
