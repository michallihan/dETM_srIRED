import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

import pipeline_settings as s  # noqa: E402

DICTIONARY_CSV = THIS_DIR / "Rike_IRED_evolution_CHX_CPA_dictionary.csv"


def build(tables):
    # Straight concatenation, round by round, with a round column in front. Nothing is
    # filtered: the dictionary keeps the caller's frameshift and no_translation rows, and
    # 11_collapse_unique_variants.py is what drops them. The published dictionary has one
    # round-3 row fewer than its source table (cluster_148, no_translation); no rule in
    # the data accounts for it, so it is not reproduced here. See README.
    frames = []
    for r, path in sorted(tables.items()):
        df = pd.read_csv(path)
        df.insert(0, "round", r)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    p = s.parser("Join the per-round mutation tables into the campaign dictionary")
    p.add_argument("--table", action="append", metavar="ROUND=PATH", default=[],
                   help="one round's full_table_barcodes_mutations.csv, e.g. 3=.../round_3.csv; "
                        "repeat per round. Default: work/9_mutations/round_<N>/")
    p.add_argument("-o", "--output", default=DICTIONARY_CSV)
    args = p.parse_args()

    if args.table:
        tables = {int(t.split("=", 1)[0]): t.split("=", 1)[1] for t in args.table}
    else:
        tables = {}
        for r in s.ROUNDS:
            fresh = s.WORK_DIR / f"9_mutations/round_{r}/full_table_barcodes_mutations.csv"
            tables[r] = fresh if fresh.exists() else s.PER_ROUND_TABLE[r]

    missing = sorted(set(s.ROUNDS) - set(tables))
    if missing:
        print(f"  no table given for round(s) {missing}")
    if not s.require(*tables.values()):
        return 1

    for r, path in sorted(tables.items()):
        print(f"  round {r}: {path}")
    if args.dry_run:
        return s.done("10_build_dictionary", not missing)

    joined = build(tables)
    joined.to_csv(args.output, index=False)
    print(f"  {len(joined)} rows written to {args.output}")
    print("  per round: " + ", ".join(
        f"{r}={int((joined['round'] == r).sum())}" for r in s.ROUNDS))
    return s.done("10_build_dictionary", not missing)


if __name__ == "__main__":
    raise SystemExit(main())
