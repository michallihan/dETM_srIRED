"""
Identifies the variants that selection removed completely — the inactive set.

A variant is called inactive when it was clearly present going into the selection and is
entirely absent coming out of it:

    input count  > 10  in ALL three replicates
    output count == 0  in ALL three replicates

Both conditions are per replicate, not on the average: a variant with inputs 47/37/8 fails
even though its mean is well above the threshold, and a single output read in one replicate
is enough to exclude it. The input requirement is what makes a zero output meaningful — it
guarantees the variant was sampled often enough that its absence is depletion rather than a
variant that was barely in the library to begin with.

These variants have no fitness score and cannot have one. DiMSum's fitness is a log ratio of
output to input counts, which is -Inf when the output is zero. The pipeline's
`fitnessDropoutPseudocount` — the one option that would assign a dropout a finite value — is
left at its default of 0 in dimsum_settings.py, so these variants simply do not reach the
fitness tables in 02a_/02b_. That is why this folder reports counts and nothing else: a
pseudocount-derived score would be a bound set by the pseudocount and the sequencing depth,
not a measurement of the variant.

Source: the LENIENT gate's variant_data_merge.tsv, which is DiMSum's merged per-variant count
table. It has to be that file rather than the gate's all_variants.csv that 02a_/02b_ read:
all_variants.csv is written after DiMSum's fitness stage and holds none of these 128 variants
at all, precisely because their fitness is not finite. The lenient gate is the published
definition, and it is the conservative choice — the lenient sort is the more permissive of the
two, so finding nothing in its output is stronger evidence of inactivity than finding nothing
in the stringent one. For reference, the same criterion applied to the stringent gate returns
420 variants, 96 of which are shared with the lenient set; switch MERGE_TSV below to see it.

Variants with indels, premature stop codons and stop-codon readthrough are removed first,
which is STEP 1 of the QC filter in 02a_/02b_, applied here in exactly the same way and for
the same reason. A frameshift or a truncation is trivially inactive and says nothing about
which substitutions the enzyme cannot tolerate. All 128 variants that survive are missense.

Counts are summed across barcodes by grouping on aa_seq before the criterion is applied, so
the thresholds act on the pooled per-variant count rather than on individual barcodes.

Outputs, written next to this script:
    inactive_variants.csv
    diagnostic_plots/count_inactive_per_pos.pdf / .png
    diagnostic_plots/count_all_single_per_pos.pdf / .png
"""

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable (e.g. in Affinity/Illustrator)

# Anchor all paths to this script's own location (not the working directory), so the
# repo works no matter where it's cloned/downloaded to or what directory it's run from.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import DIMSUM_DMS_LENIENT_MERGE_TSV  # noqa: E402

output_dir = THIS_DIR / "diagnostic_plots"
output_dir.mkdir(exist_ok=True)


def count_columns(data):
    """
    The input and output count columns of a variant_data_merge.tsv, ordered by replicate.

    The two gates number their samples differently — the lenient gate has input1/2/3 and the
    stringent gate input4/2/6 — so the columns are found by their experiment number (_e1_,
    _e2_, _e3_) rather than by name. Returns (input_columns, output_columns).
    """
    def by_experiment(pattern):
        matched = [c for c in data.columns if re.match(pattern, c)]
        return sorted(matched, key=lambda c: int(re.search(r"_e(\d+)_", c).group(1)))

    inputs = by_experiment(r"input\d+_e\d+_s0_")
    outputs = by_experiment(r"output\d+_e\d+_s1_")
    if len(inputs) != len(outputs) or not inputs:
        raise ValueError(f"Could not pair input and output columns: {inputs} / {outputs}")
    return inputs, outputs


def call_mutations(reference, target):
    """Position-wise comparison against the wild type. Returns a list of (position, wt, mut)."""
    if len(reference) != len(target):
        return None
    return [
        (i + 1, reference[i], target[i])
        for i in range(len(reference))
        if reference[i] != target[i]
    ]


def mutation_id(mutations):
    """'S68P-T241A' — the same identifier format the fitness tables in 02a_/02b_ use."""
    if not mutations:
        return "WT"
    return "-".join(f"{wt}{position}{mut}" for position, wt, mut in mutations)


def load_and_filter(merge_tsv):
    """
    Read the merged count table and apply STEP 1 of the 02a_/02b_ QC filter.

    Removes indels, premature stop codons and stop-codon readthrough, for the same reason
    those scripts do: they are trivially inactive and are not informative substitutions.
    """
    data = pd.read_csv(merge_tsv, sep="\t")
    kept = data[~(data["indel"] | data["STOP"] | data["STOP_readthrough"])]
    return data, kept


def wildtype_sequence(data):
    """The wild-type amino-acid sequence, taken from the table's own WT row."""
    wt_rows = data.loc[data["WT"] == True, "aa_seq"].unique()  # noqa: E712 - DiMSum writes a bool column
    if len(wt_rows) != 1:
        raise ValueError(f"Expected exactly one wild-type sequence, found {len(wt_rows)}.")
    return str(wt_rows[0])


def collapse_to_variants(kept, input_columns, output_columns):
    """Sum counts across barcodes so the thresholds act on the pooled per-variant count."""
    return kept.groupby("aa_seq", as_index=False).agg(
        {column: "sum" for column in input_columns + output_columns}
    )


def identify_inactive(variants, input_columns, output_columns, min_input, max_output):
    """input > min_input in every replicate AND output <= max_output in every replicate."""
    present_going_in = (variants[input_columns] > min_input).all(axis=1)
    absent_coming_out = (variants[output_columns] <= max_output).all(axis=1)
    return variants[present_going_in & absent_coming_out].copy()


def build_table(inactive, reference, input_columns, output_columns):
    """Assemble the output table in the column layout the circulated file uses."""
    rows = []
    for _, variant in inactive.iterrows():
        mutations = call_mutations(reference, str(variant["aa_seq"]))
        row = {
            "aa_seq": variant["aa_seq"],
            "Nham_aa": len(mutations),
            # Constant by construction after the STEP 1 filter, kept so the table matches the
            # layout of the file already in circulation.
            "WT": False,
            "indel": False,
            "STOP": False,
            "STOP_readthrough": False,
        }
        for replicate, column in enumerate(input_columns, start=1):
            row[f"count_e{replicate}_s0"] = variant[column]
        for replicate, column in enumerate(output_columns, start=1):
            row[f"count_e{replicate}_s1"] = variant[column]
        row["avg_input_count_on"] = variant[input_columns].sum() / len(input_columns)
        row["avg_output_count_on"] = variant[output_columns].sum() / len(output_columns)
        row["mut_id"] = mutation_id(mutations)
        rows.append(row)
    return pd.DataFrame(rows)


def single_mutant_positions(sequences, reference):
    """Positions of the substitution in every single mutant, stop codons excluded."""
    positions = []
    for sequence in sequences:
        mutations = call_mutations(reference, str(sequence))
        if mutations and len(mutations) == 1 and mutations[0][2] != "*":
            positions.append(mutations[0][0])
    return positions


def figure_per_position(positions, sequence_length, title, ylabel, filename):
    """
    Substitutions per position, as a bar per residue.

    The original drew these over a fixed range of 1-296, which overruns the 291-position
    sequence; the axis is set from the sequence itself here instead.
    """
    counts = [positions.count(position) for position in range(1, sequence_length + 1)]

    plt.figure(figsize=[15, 3])
    plt.bar(range(1, sequence_length + 1), counts, color="k")
    plt.xlim(1, sequence_length)
    for position in range(10, sequence_length + 1, 10):
        plt.axvline(x=position, color="k", alpha=0.6, linewidth=0.3)
    plt.locator_params(axis="x", nbins=29)
    plt.locator_params(axis="y", nbins=10)
    plt.title(title)
    plt.xlabel("position")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_dir / f"{filename}.pdf")
    plt.savefig(output_dir / f"{filename}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Switch to DIMSUM_DMS_STRINGENT_MERGE_TSV to apply the same criterion to the other gate.
    MERGE_TSV = DIMSUM_DMS_LENIENT_MERGE_TSV

    # --- inactivity criterion ---
    MIN_INPUT_COUNT = 10   # input must exceed this in every replicate
    MAX_OUTPUT_COUNT = 0   # output must not exceed this in any replicate

    output_csv = THIS_DIR / "inactive_variants.csv"

    raw, kept = load_and_filter(MERGE_TSV)
    reference = wildtype_sequence(raw)
    input_columns, output_columns = count_columns(raw)

    variants = collapse_to_variants(kept, input_columns, output_columns)
    inactive = identify_inactive(
        variants, input_columns, output_columns, MIN_INPUT_COUNT, MAX_OUTPUT_COUNT
    )
    table = build_table(inactive, reference, input_columns, output_columns)
    table.to_csv(output_csv, index=False)

    figure_per_position(
        single_mutant_positions(table["aa_seq"], reference),
        len(reference),
        "Inactive variants per position",
        "Count of inactive variants",
        "count_inactive_per_pos",
    )
    figure_per_position(
        single_mutant_positions(variants["aa_seq"], reference),
        len(reference),
        "Single mutants per position",
        "Count of variants",
        "count_all_single_per_pos",
    )

    print(f"{len(raw)} rows -> {len(kept)} after STEP 1 -> {len(variants)} variants "
          f"-> {len(table)} inactive")
    print(f"Wrote {output_csv}")
