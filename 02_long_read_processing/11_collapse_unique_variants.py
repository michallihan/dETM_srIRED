import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402

DICTIONARY_CSV = THIS_DIR / "Rike_IRED_evolution_CHX_CPA_dictionary.csv"
COLLAPSED_CSV = THIS_DIR / "Rike_IRED_evolution_CHX_CPA_dictionary_collapsed_unique_variants.csv"

PROTEIN_LENGTH = 291   # the reference translated, including the stop


def is_intact(row):
    seq = row["aa_sequence"]
    return (isinstance(seq, str)
            and len(seq) == PROTEIN_LENGTH          # no truncation, no frameshift
            and "'*'" not in str(row["aa_mutations"])  # no premature stop
            and row["insertions"] == 0
            and row["deletions"] == 0)


def collapse(dictionary):
    intact = dictionary[dictionary.apply(is_intact, axis=1)]
    rows = []
    for i, (seq, group) in enumerate(intact.groupby("aa_sequence", sort=False), start=1):
        rows.append({
            "name": f"var_{i}",
            "aa_mutations": group["aa_mutations"].iloc[0],
            "aa_mutation_number": int(group["aa_mutation_number"].iloc[0]),
            "aa_sequence": seq,
            "total_occurrences": len(group),
            **{f"round_{r}": int((group["round"] == r).sum()) for r in s.ROUNDS},
        })
    return pd.DataFrame(rows), len(dictionary) - len(intact)


def main():
    p = s.parser("Collapse the dictionary to one row per unique intact protein")
    p.add_argument("-i", "--input", default=DICTIONARY_CSV)
    p.add_argument("-o", "--output", default=COLLAPSED_CSV)
    args = p.parse_args()

    if not s.require(args.input):
        return 1
    print(f"  {args.input} -> {args.output}")
    if args.dry_run:
        return s.done("11_collapse_unique_variants", True)

    dictionary = pd.read_csv(args.input)
    collapsed, excluded = collapse(dictionary)
    collapsed.to_csv(args.output, index=False)
    print(f"  {len(dictionary)} dictionary rows, {excluded} excluded as not intact")
    print(f"  {len(collapsed)} unique variants, "
          f"{int(collapsed['total_occurrences'].sum())} occurrences")
    return s.done("11_collapse_unique_variants", len(collapsed) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
