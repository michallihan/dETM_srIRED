"""
How many clones would have to be screened to recover each variant of the winning lineage.

The population-dynamics tables say what fraction of each round's pool every variant made up.
Read as a screening problem that fraction is the chance of drawing the variant once, so the
number of clones that must be picked to see it at least once with probability P follows from
it directly:

    T = ln(1 - P) / ln(1 - F)

with F the variant's share of the pool and T rounded up to a whole clone. The 3/F rule of
thumb quoted alongside this expression is its large-library limit and runs a few per cent
high at the frequencies here, so it is not used: in round 5 it turns T241A's 23 clones into
24 and the quadruple's 2,565 into 2,566.

T is the cost of fishing a variant out of the pool as it stood, not a library-coverage
number. It assumes drawing with replacement from a population large enough that removing one
clone does not move F, which holds comfortably at these library sizes, and it says nothing
about whether a screen would recognise the variant once picked.

F comes from the input samples - the pool as it entered that round's selection, before
sorting - on the same filtering every other population-dynamics analysis in the repo uses:
rows with a stop codon, a read-through stop, an indel or an invalid barcode are dropped, rows
sharing an amino-acid sequence are summed so that synonymous nucleotide variants count once,
each replicate column is turned into a fraction of its own total, and the replicates are
averaged. The output samples are the post-sort pool and would give different numbers; they
are not what anyone picks colonies from.

Round 0 is the naive library. It sits in 01_lrDMS_pipeline as a workbook rather than a DiMSum
table because the run that produced it has not been located.

LINEAGE holds the five variants of the winning lineage and CITED_ROUND the round the main
text quotes. Both are configuration: nothing in the data picks them out, so changing either
changes what is reported without touching the calculation. A variant absent from a round has
no T - no depth of screening recovers something that is not there - and its cells are empty.

Outputs, written next to this script:
    oversampling_per_round.csv   every lineage variant in every round, machine-readable
    screening_burden.xlsx        the round the main text cites, then all rounds, then a
                                 sheet defining the columns
"""

import math
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import DIMSUM_NAIVE_COUNTS_XLSX, DIMSUM_ROUND_COUNTS_TSV  # noqa: E402
from variant_rounds import mutation_ids  # noqa: E402

ROUND_TABLES = {0: DIMSUM_NAIVE_COUNTS_XLSX, **DIMSUM_ROUND_COUNTS_TSV}

# Rows DiMSum flags that the population-dynamics analysis drops: column -> value removed.
DISCARD = {"STOP": True, "STOP_readthrough": True, "barcode_valid": False, "indel": True}

LINEAGE = ("T241A", "S68P-T241A", "S68P-P203Q-T241A", "S68P-T241A-H244Y",
           "S68P-P203Q-T241A-H244Y")

PROBABILITIES = (0.95, 0.99)
PLATE_WELLS = 96
CITED_ROUND = 5

SHEET_COLUMNS = ["round", "mut_id", "n_mutations", "abundance_percent",
                 "clones_95", "plates_95", "clones_99", "plates_99"]

COLUMN_NOTES = [
    ("round", "Selection round the pool was sampled from. Round 0 is the naive library, "
              "before any selection."),
    ("mut_id", "Variant identity, substitutions ordered by residue position."),
    ("n_mutations", "Number of amino-acid substitutions relative to the wild type."),
    ("abundance_fraction", "F: the variant's mean share of that round's input read pool, "
                           "averaged over the replicate samples. CSV only."),
    ("abundance_percent", "The same as a percentage."),
    ("clones_95", "T at P = 0.95: clones to pick for a 95% chance of recovering the variant "
                  "at least once, rounded up to a whole clone."),
    ("plates_95", "clones_95 expressed in 96-well plates, rounded up to a whole plate."),
    ("clones_99", "T at P = 0.99."),
    ("plates_99", "clones_99 in 96-well plates."),
]


def read_round(path):
    if path.suffix == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path, sep="\t")


def round_shares(round_number):
    """One round's variants with each one's mean share of the replicate read pools."""
    table = read_round(ROUND_TABLES[round_number])
    for column, discarded in DISCARD.items():
        if column in table.columns:
            table = table[table[column] != discarded]

    inputs = [c for c in table.columns if c.startswith("input")]
    collapsed = table.groupby("aa_seq", as_index=False).agg(
        {**{c: "sum" for c in inputs}, "WT": "first"}
    )

    wild_type = collapsed.loc[collapsed["WT"] == True, "aa_seq"]  # noqa: E712
    if len(wild_type) != 1:
        raise ValueError(f"round {round_number} has {len(wild_type)} wild-type rows, expected 1")
    collapsed["mut_id"] = mutation_ids(collapsed["aa_seq"], wild_type.iloc[0])

    fractions = collapsed[inputs] / collapsed[inputs].sum()
    shares = pd.Series(fractions.mean(axis=1).to_numpy(), index=collapsed["mut_id"])
    return shares[shares.index.notna()]


def clones_needed(share, probability):
    """Clones to pick for at least `probability` of drawing a variant held at `share`."""
    if share <= 0:
        return None
    if share >= 1:
        return 1
    return math.ceil(math.log(1 - probability) / math.log(1 - share))


def plates_needed(clones):
    return None if clones is None else math.ceil(clones / PLATE_WELLS)


def n_mutations(mut_id):
    return 0 if mut_id == "WT" else len(mut_id.split("-"))


def build():
    """Every lineage variant in every round, with its share and the screening depth it implies."""
    rows = []
    for round_number in sorted(ROUND_TABLES):
        shares = round_shares(round_number)
        for mut_id in LINEAGE:
            share = float(shares.get(mut_id, 0.0))
            row = {"round": round_number,
                   "mut_id": mut_id,
                   "n_mutations": n_mutations(mut_id),
                   "abundance_fraction": share,
                   "abundance_percent": 100 * share}
            for probability in PROBABILITIES:
                clones = clones_needed(share, probability)
                row[f"clones_{round(probability * 100)}"] = clones
                row[f"plates_{round(probability * 100)}"] = plates_needed(clones)
            rows.append(row)

    table = pd.DataFrame(rows)
    # Counts, not measurements: keep them whole so an unobserved variant reads as empty
    # rather than as a float with a missing value beside it.
    for column in table.columns:
        if column.startswith(("clones_", "plates_")):
            table[column] = table[column].astype("Int64")
    return table


def preamble():
    return [
        ("Quantity", "Clones that must be picked from a round's variant pool to recover a "
                     "given variant at least once with probability P."),
        ("Formula", "T = ln(1 - P) / ln(1 - F), rounded up to a whole clone."),
        ("F", "The variant's mean share of that round's input read pool "
              "(abundance_fraction in the CSV)."),
        ("P", "0.95 and 0.99, reported as clones_95 and clones_99."),
        ("Plate size", f"{PLATE_WELLS} wells; plates = ceil(T / {PLATE_WELLS})."),
        ("Frequency basis", "Input samples - the pool entering each round's selection, before "
                            "sorting. Rows with a stop codon, a read-through stop, an indel or "
                            "an invalid barcode are dropped, synonymous nucleotide variants are "
                            "summed into one amino-acid variant, each replicate column is taken "
                            "as a fraction of its own total and the replicates are averaged. "
                            "The same shares as Figure 4G and the population-dynamics SI."),
        ("Assumption", "Drawing with replacement from a pool large enough that removing a clone "
                       "does not change F. T is the depth needed to encounter the variant; it "
                       "does not model whether a screen would score it correctly."),
        ("Not used", "The 3/F approximation, which is the large-library limit of the formula "
                     "above and runs a few per cent high at these frequencies."),
        ("Empty cells", "The variant was not observed in that round's pool."),
        ("References", "Bosley & Ostermeier, Biomol. Eng. 2005, 22, 57-61; "
                       "Reetz, ChemBioChem 2008, DOI 10.1002/cbic.200800298."),
    ]


def write_workbook(path, sheets, notes, front):
    """The data sheets, then one sheet holding the parameters and the column definitions."""
    legend = [*front, ("", ""), ("Column", "Definition"), *notes]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet, table in sheets.items():
            table.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame(legend).to_excel(writer, sheet_name="Legend", index=False, header=False)
        for sheet in sheets:
            worksheet = writer.sheets[sheet]
            for column in worksheet.columns:
                longest = max(len(str(cell.value)) for cell in column)
                worksheet.column_dimensions[column[0].column_letter].width = min(longest + 2, 34)
        for letter, width in zip("AB", (18, 110)):
            writer.sheets["Legend"].column_dimensions[letter].width = width


def main():
    table = build()
    table.to_csv(THIS_DIR / "oversampling_per_round.csv", index=False)

    sheets = table[SHEET_COLUMNS].copy()
    sheets["abundance_percent"] = sheets["abundance_percent"].round(4)
    cited = sheets[sheets["round"] == CITED_ROUND].reset_index(drop=True)
    write_workbook(THIS_DIR / "screening_burden.xlsx",
                   {f"Round_{CITED_ROUND}": cited, "All_rounds": sheets},
                   COLUMN_NOTES, preamble())

    for _, row in cited.iterrows():
        observed = row["clones_95"] is not None and not pd.isna(row["clones_95"])
        plates = int(row["plates_95"]) if observed else 0
        depth = (f"{int(row['clones_95']):,} clones, {plates} plate{'s' * (plates != 1)}"
                 if observed else "not observed")
        print(f"round {CITED_ROUND}  {row['mut_id']:<24s} "
              f"{row['abundance_percent']:>8.4f}%  {depth}")


if __name__ == "__main__":
    main()
