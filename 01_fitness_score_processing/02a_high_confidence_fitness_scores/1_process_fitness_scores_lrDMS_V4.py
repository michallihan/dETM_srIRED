"""
Created on 11-05-2024
Author: Mink Sieders
Generates a .csv file with fitness scores from two files with different sorting gates,
directly derived from dimsum R data file (via csv writer)

Edited on 28-08-2024
Author: Friederike Nintzel

This is the high-confidence version of the QC filter: all four QC steps are applied (input-count, mean-input-count, and sigma cutoffs).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import pearsonr
from itertools import combinations
from matplotlib_venn import venn2

import function_library_lrDMS_analysis as lrDMSa

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable (e.g. in Affinity/Illustrator)

# Anchor all paths to this script's own location (not the working directory), so the
# repo works no matter where it's cloned/downloaded to or what directory it's run from.
THIS_DIR = Path(__file__).resolve().parent
output_dir = THIS_DIR / "diagnostic_plots"
output_dir.mkdir(exist_ok=True)


def log_average(values):
    """Average a list of log-scale values by exponentiating, averaging, then taking the log again.
    NaNs are ignored; a single valid value is returned as-is; an all-NaN input returns NaN."""
    valid_values = [v for v in values if not pd.isna(v)]

    if len(valid_values) == 1:
        return valid_values[0]
    if len(valid_values) == 0:
        return np.nan

    exp_values = np.exp(valid_values)
    avg_exp_value = np.mean(exp_values)
    return np.log(avg_exp_value)


def open_data(dms1, dms2, str_gat):
    files = [dms1, dms2]
    if str_gat == 1 or str_gat == 2:
        idx_str = str_gat - 1
        idx_len = 1 if str_gat == 1 else 0
        raw_data_str = pd.read_csv(f"{files[idx_str]}")
        raw_data_len = pd.read_csv(f"{files[idx_len]}")
    else:
        raise ValueError("Sorry, stringent_gate only accepts integers 1 or 2")

    return raw_data_str, raw_data_len


def get_reps_num(data):
    if len(data.columns) == 21:
        return 2
    elif len(data.columns) == 25:
        return 3
    else:
        raise Exception(
            "Sorry, the number of columns in the dimsum output file seems to be off, or you used fewer than 2 or "
            "more than 3 replicates")


def filter_lrdms(raw_data_file_dms, cutoff_count_input=None, step_three=None, cutoff_sigma=None):
    """
    QC-filter a single-gate dimsum fitness table.

    Replicate-level fitness{n}_uncorr / sigma{n}_uncorr columns are never dropped here -
    only rows are removed.

    cutoff_count_input : drop rows where every input replicate has <= this many counts (step 2). None/False skips.
    step_three          : drop rows where the mean input count across replicates is below this value (step 3). None/False skips.
    cutoff_sigma        : drop rows with a pooled sigma >= this value (step 4). None/False skips.
    """
    removed_data = pd.DataFrame({})
    data = raw_data_file_dms
    reps_num = get_reps_num(data)

    # STEP 1: remove rows with indels, premature stop codons, and readthrough; drop columns not needed downstream.
    filtered_data = data[~(data["indel"] | data["STOP"] | data["STOP_readthrough"])]
    drop_columns = ["nt_seq", "Nmut_codons", "Nham_nt", "indel", "STOP", "STOP_readthrough", "Unnamed: 0"]
    dropped_data = filtered_data.drop(columns=drop_columns)

    # STEP 2: remove rows with an input count at or below cutoff_count_input in every replicate
    if cutoff_count_input:
        columns_to_check_input = ["count_e" + str(num) + "_s0" for num in range(1, reps_num + 1)]
        conditions_greater = [(dropped_data[column] > cutoff_count_input) for column in columns_to_check_input]
        combined_condition_greater = pd.concat(conditions_greater, axis=1).any(axis=1)
        filtered_data2 = dropped_data[combined_condition_greater]
        step2_label = f"2: Any rep input < {cutoff_count_input} counts"
    else:
        filtered_data2 = dropped_data
        step2_label = "2: Input count filter (not applied)"

    # STEP 3: remove rows with a mean input count (across replicates) below step_three
    if step_three:
        columns_to_check_input = ["count_e" + str(num) + "_s0" for num in range(1, reps_num + 1)]
        mean_counts = filtered_data2[columns_to_check_input].mean(axis=1)
        filtered_data3 = filtered_data2[mean_counts >= step_three]
        step3_label = f"3: Mean input < {step_three} counts"
    else:
        filtered_data3 = filtered_data2
        step3_label = "3: Mean input count filter (not applied)"

    # STEP 4: remove rows with a pooled sigma at or above cutoff_sigma
    if cutoff_sigma:
        filtered_data5 = filtered_data3[filtered_data3["sigma"] < cutoff_sigma]
        step4_label = f"4: Sigma cutoff {cutoff_sigma}"
    else:
        filtered_data5 = filtered_data3
        step4_label = "4: Sigma filter (not applied)"

    removed_data["1: Indels, frameshift, stop"] = [len(data) - len(dropped_data)]
    removed_data[step2_label] = [len(dropped_data) - len(filtered_data2)]
    removed_data[step3_label] = [len(filtered_data2) - len(filtered_data3)]
    removed_data[step4_label] = [len(filtered_data3) - len(filtered_data5)]
    removed_data["Kept"] = [len(filtered_data5)]

    return filtered_data5, removed_data


def figure_correlation_stringent_lenient(data_lenient, data_stringent):
    merged_df = pd.merge(data_lenient, data_stringent, on="aa_seq", suffixes=("_x", "_y"))
    merged_df.rename(columns={"fitness_x": "fitness_lenient", "fitness_y": "fitness_stringent"}, inplace=True)
    merged_df.rename(columns={"sigma_x": "sigma_lenient", "sigma_y": "sigma_stringent"}, inplace=True)
    final_df = merged_df[["aa_seq", "fitness_lenient", "fitness_stringent", "sigma_lenient", "sigma_stringent"]]
    pearson_corr, _ = stats.pearsonr(final_df["fitness_lenient"], final_df["fitness_stringent"])
    model = np.polyfit(final_df["fitness_lenient"], final_df["fitness_stringent"], 1)

    x_val_line, y_val_line = lrDMSa.generate_linear_sequence(model[0], model[1],
                                                             min([min(final_df["fitness_lenient"]),
                                                                  min(final_df["fitness_stringent"])]),
                                                             max([max(final_df["fitness_lenient"]),
                                                                  max(final_df["fitness_stringent"])]),
                                                             100)

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.scatter(final_df["fitness_lenient"], final_df["fitness_stringent"],
                color="black", alpha=0.3)
    ax1.plot(x_val_line, y_val_line, color="black", linewidth=2)
    ax1.text(0.9, 0.85, f"ρ: {pearson_corr:.2f}", transform=ax1.transAxes,
             fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    ax1.set_xlabel("Fitness score lenient gate")
    ax1.set_ylabel("Fitness score stringent gate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_lenient_stringent_gates.svg"))
    plt.close()

    fitness_lenient = data_lenient["fitness"].copy()
    fitness_stringent = data_stringent["fitness"].copy()

    count_lenient = fitness_lenient.count()
    count_stringent = fitness_stringent.count()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(fitness_lenient, bins=30, color='#1a9850', label=f'lenient (n={count_lenient})', alpha=0.5, edgecolor='black')
    ax.hist(fitness_stringent, bins=30, color='#fd8d3c', label=f'stringent (n={count_stringent})', alpha=0.5, edgecolor='black')

    ax.set_xlabel("Fitness Score")
    ax.set_ylabel("Frequency")
    ax.legend(loc='upper right')
    plt.title("Histogram of Fitness Scores")
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "histogram_fitness_scores.svg"))
    plt.close()


def combine_fitness_mappings(lenient_gate_df, stringent_gate_df, reps_num):
    """
    Combine the lenient- and stringent-gate fitness tables for each unique variant.

    Step 1 combines gates at the replicate level: for each replicate n, the lenient-gate
    and stringent-gate fitness{n}_uncorr values (and sigma{n}_uncorr values) for that
    variant are combined into fitness{n}_uncorr_combined / sigma{n}_uncorr_combined.

    Step 2 only then averages those per-replicate combined values into the overall
    fitness_final / sigma_final for the variant.
    """
    merged_df = pd.merge(lenient_gate_df, stringent_gate_df, on="aa_seq", how="outer",
                         suffixes=("_lenient", "_stringent"))

    # STEP 1: combine matching replicates across gates
    for n in range(1, reps_num + 1):
        fit_col = f"fitness{n}_uncorr"
        sig_col = f"sigma{n}_uncorr"
        merged_df[f"{fit_col}_combined"] = merged_df.apply(
            lambda row, fc=fit_col: log_average([row[f"{fc}_lenient"], row[f"{fc}_stringent"]]), axis=1)
        merged_df[f"{sig_col}_combined"] = merged_df.apply(
            lambda row, sc=sig_col: log_average([row[f"{sc}_lenient"], row[f"{sc}_stringent"]]), axis=1)

    # STEP 2: combine the per-replicate combined values into the final fitness/sigma
    combined_fitness_cols = [f"fitness{n}_uncorr_combined" for n in range(1, reps_num + 1)]
    combined_sigma_cols = [f"sigma{n}_uncorr_combined" for n in range(1, reps_num + 1)]
    merged_df["fitness_final"] = merged_df[combined_fitness_cols].apply(lambda row: log_average(row.tolist()), axis=1)
    merged_df["sigma_final"] = merged_df[combined_sigma_cols].apply(lambda row: log_average(row.tolist()), axis=1)

    # decision: which gate(s) contributed a pooled fitness value for this variant
    conditions = [merged_df["fitness_lenient"].isna(), merged_df["fitness_stringent"].isna()]
    choices = ["only stringent gate available", "only lenient gate available"]
    merged_df["decision"] = np.select(conditions, choices, default="average lenient & stringent gates")

    merged_df["Nham_aa"] = merged_df["Nham_aa_lenient"].combine_first(merged_df["Nham_aa_stringent"])
    merged_df = merged_df.drop(columns=["Nham_aa_lenient", "Nham_aa_stringent", "WT_stringent"])
    return merged_df


def figure_final_distribution_fitness(dataframe):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    unique_decisions = dataframe["decision"].unique()
    data = [dataframe[dataframe["decision"] == decision]["fitness_final"] for decision in unique_decisions]
    ax[0].hist(data, bins=np.arange(-4, 5, 0.5), label=unique_decisions, stacked=True,
               edgecolor="white")
    ax[0].set_xlabel("Fitness score")
    ax[0].set_ylabel("Number of unique variants")
    ax[0].set_ylim(0, 150)
    ax[0].set_yticks([0, 25, 50, 75, 100, 125, 150])
    ax[0].legend(title="Decision", loc="upper center")

    filtered_data = dataframe[(dataframe["Nham_aa"] >= 1) & (dataframe["Nham_aa"] <= 6)]
    mutation_frequencies = filtered_data["Nham_aa"].value_counts().sort_index()
    ax[1].bar(
        mutation_frequencies.index,
        mutation_frequencies.values,
        color="grey",
        edgecolor="white"
    )
    ax[1].set_xlabel("Number of mutations")
    ax[1].set_ylabel("Number of unique variants")
    ax[1].set_ylim(0, 250)
    ax[1].set_xticks(range(1, 7))
    ax[1].set_xticklabels([1, 2, 3, 4, 5, 6])
    plt.xticks(rotation=0)
    ax[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}".format(x)))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_fitness_mutation_distribution.svg"))
    plt.close()


def figures_quality_replicates(dataframe):
    df_filtered = dataframe[dataframe["decision"] == "average lenient & stringent gates"]
    a = (dataframe.filter(regex=r"^count_e[12345]_s0_lenient").to_numpy(), "input_lenient")
    b = (dataframe.filter(regex=r"^count_e[12345]_s1_lenient").to_numpy(), "output_lenient")
    c = (df_filtered.filter(regex=r"^count_e[12345]_s0_stringent").to_numpy(), "input_stringent")
    d = (df_filtered.filter(regex=r"^count_e[12345]_s1_stringent").to_numpy(), "output_stringent")
    data = [a, b, c, d]
    new_data = []
    add_factor = 1
    for ar, ar_name in data:
        new_ar = ar[~np.isnan(ar).any(axis=1), :]
        new_ar = np.add(new_ar, add_factor)
        new_ar = np.log10(new_ar)
        new_data.append((ar_name, new_ar))
    for ar_name, ar in new_data:
        df_ar = pd.DataFrame(ar, columns=[f"col_{i}" for i in range(ar.shape[1])])
        col_pairs = list(combinations(df_ar.columns, 2))
        num_pairs = len(col_pairs)
        fig, axs = plt.subplots(int(np.ceil(num_pairs / 3)), 3, figsize=(15, 5 * int(np.ceil(num_pairs / 3))))
        fig.suptitle(f"{ar_name}", fontsize=16)
        for (col1, col2), ax in zip(col_pairs, axs.flatten()):
            sns.regplot(x=df_ar[col1], y=df_ar[col2], ax=ax, scatter_kws={"s": 10, "color": "black", "alpha": 0.4},
                        line_kws={"color": "black", "linewidth": 1})
            X = df_ar[col1].values
            y = df_ar[col2].values
            r, p_value = pearsonr(X, y)
            ax.text(0.05, 0.95, f"ρ: {r:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment="top")
            ax.set_xlabel(f"log($counts_{int(col1[len(col1) - 1]) + 1}$+1)")
            ax.set_ylabel(f"log($counts_{int(col2[len(col2) - 1]) + 1}$+1)")
            ax.set_title(f"Replicate {int(col1[len(col1) - 1]) + 1} vs Replicate {int(col2[len(col2) - 1]) + 1}")
        if num_pairs % 3 != 0:
            for i in range(3 - num_pairs % 3):
                fig.delaxes(axs.flatten()[-(i + 1)])
        plt.savefig(os.path.join(output_dir, f"replicates_correlations_{ar_name}.svg"))
        plt.close()


def figure_final_replicate_correlations(dataframe, reps_num):
    """
    Pairwise correlation between the final, cross-gate-combined per-replicate fitness
    values (fitness{n}_uncorr_combined - replicate n after lenient+stringent were
    already combined for that replicate).

    
    """
    columns = [f"fitness{n}_uncorr_combined" for n in range(1, reps_num + 1)]
    sample_names = [f"Rep. {n}" for n in range(1, reps_num + 1)]
    n = reps_num

    df = dataframe[columns]
    global_min = df.min().min()
    global_max = df.max().max()

    from matplotlib.ticker import MaxNLocator
    fig_tmp, ax_tmp = plt.subplots()
    ax_tmp.set_xlim(global_min, global_max)
    ax_tmp.set_ylim(global_min, global_max)
    ax_tmp.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ticks = ax_tmp.get_xticks()
    plt.close(fig_tmp)

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j] if n > 1 else axes

            # diagonal and upper triangle: left blank, matching the trimmed paper figure
            if i <= j:
                ax.axis("off")
                continue

            # lower triangle: scatter of replicate j (x) vs replicate i (y)
            col_x, col_y = columns[j], columns[i]
            pair = dataframe[[col_x, col_y]].dropna()

            ax.scatter(pair[col_x], pair[col_y], s=18, c="#2a6f97", alpha=0.45,
                       edgecolors="black", linewidths=0.25)

            slope, intercept = np.polyfit(pair[col_x], pair[col_y], 1)
            x_fit = np.array([global_min, global_max])
            ax.plot(x_fit, slope * x_fit + intercept, color="black", linestyle=":", linewidth=1.5)

            r, p = pearsonr(pair[col_x], pair[col_y])
            p_display = f"p = {p:.1e}" if p >= 1e-300 else "p < 1e-300"
            ax.text(0.97, 0.97, f"r = {r:.2f}\n{p_display}", ha="right", va="top",
                    fontsize=11, transform=ax.transAxes,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)

            if i < n - 1:
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            else:
                ax.set_xlabel(sample_names[j])

            if j > 0:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            else:
                ax.set_ylabel(sample_names[i])

    plt.savefig(os.path.join(output_dir, "final_replicate_correlations.svg"))
    plt.close()


def figure_filtering_report():
    unique_seqs_unfilt = pd.concat([raw_len, raw_str])["aa_seq"].nunique()
    unique_seqs_filt = pd.concat([stringent_data, lenient_data])["aa_seq"].nunique()

    str_df = removed_data_str.iloc[:, ::-1]
    len_df = removed_data_len.iloc[:, ::-1]

    plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    ax4 = plt.subplot2grid((2, 2), (1, 1))

    colors = ["#016450", "#1a9850", "#a6d96a", "#fde725", "#fd8d3c", "#e31a1c", "#91003f"]

    y_limit = 200 + len_df.sum().sum()
    str_df.plot.bar(ax=ax1, stacked=True, width=0.8, color=colors)
    ax1.set_xlabel("stringent gate")
    ax1.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles1 = handles1[::-1]
    labels1 = labels1[::-1]
    ax1.legend(handles1, labels1, loc="upper right", fontsize=9)
    ax1.set_ylim(0, y_limit)
    ax1.set_ylabel("Number of unique sequences")

    len_df.plot.bar(ax=ax2, stacked=True, width=0.8, color=colors)
    ax2.set_xlabel("lenient gate")
    ax2.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2 = handles2[::-1]
    labels2 = labels2[::-1]
    ax2.legend(handles2, labels2, loc="upper right", fontsize=9)
    ax2.set_ylim(0, y_limit)

    len_match = len(pd.merge(lenient_data, stringent_data, on="aa_seq"))
    len_match_u = len(pd.merge(lenient_data_unfiltered, stringent_data_unfiltered, on="aa_seq"))
    venn2(subsets=(len(lenient_data) - len_match, len(stringent_data) - len_match, len_match),
          set_labels=("lenient", "stringent"), ax=ax4, set_colors=("#1a9850", "#fd8d3c"), alpha=0.5)
    ax4.set_title(f"Unique sequences after filtering (n = {unique_seqs_filt})")
    venn2(subsets=(
        len(lenient_data_unfiltered) - len_match_u, len(stringent_data_unfiltered) - len_match_u, len_match_u),
        set_labels=("lenient", "stringent"), set_colors=("#1a9850", "#fd8d3c"), alpha=0.5, ax=ax3)
    ax3.set_title(f"Unique sequences before filtering (n = {unique_seqs_unfilt})")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fitness_score_filtering_summary.svg"))
    plt.close()


if __name__ == "__main__":
    dimsum_file_1 = THIS_DIR / "lrDMS_all_variants_stringent.csv"
    dimsum_file_2 = THIS_DIR / "lrDMS_all_variants_lenient.csv"
    stringent_gate = 1
    output = THIS_DIR / "final_fitness_scores_ETM.csv"

    # --- high-confidence QC settings: set a cutoff to None to skip that filtering step ---
    STRINGENT_CUTOFF_COUNT_INPUT = 10
    STRINGENT_STEP_THREE = 100
    STRINGENT_CUTOFF_SIGMA = 1.0
    LENIENT_CUTOFF_COUNT_INPUT = 10
    LENIENT_STEP_THREE = 100
    LENIENT_CUTOFF_SIGMA = 1.5

    raw_str, raw_len = open_data(dimsum_file_1, dimsum_file_2, stringent_gate)

    stringent_data, removed_data_str = filter_lrdms(raw_str, cutoff_count_input=STRINGENT_CUTOFF_COUNT_INPUT,
                                                     step_three=STRINGENT_STEP_THREE,
                                                     cutoff_sigma=STRINGENT_CUTOFF_SIGMA)
    lenient_data, removed_data_len = filter_lrdms(raw_len, cutoff_count_input=LENIENT_CUTOFF_COUNT_INPUT,
                                                  step_three=LENIENT_STEP_THREE,
                                                  cutoff_sigma=LENIENT_CUTOFF_SIGMA)
    stringent_data_unfiltered = raw_str[stringent_data.columns.tolist()]
    lenient_data_unfiltered = raw_len[lenient_data.columns.tolist()]

    reps_num = get_reps_num(raw_str)
    complete_set = combine_fitness_mappings(lenient_data, stringent_data, reps_num).sort_values(
        by=["WT_lenient", "fitness_final"],
        ascending=False).reset_index(drop=True)
    complete_set = complete_set.rename(columns={"WT_lenient": "WT"})
    complete_set["mut_id"] = lrDMSa.define_mutation_ids(complete_set)
    complete_set.to_csv(output, index=False)

    figure_filtering_report()
    figure_correlation_stringent_lenient(lenient_data, stringent_data)
    figures_quality_replicates(complete_set)
    figure_final_distribution_fitness(complete_set)
    figure_final_replicate_correlations(complete_set, reps_num)
