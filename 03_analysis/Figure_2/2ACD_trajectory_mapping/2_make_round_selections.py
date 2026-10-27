import ast
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import COLLAPSED_UNIQUE_VARIANTS_CSV  # noqa: E402

WT_LABEL = "WT"
ROUNDS = [f"round_{i}" for i in range(1, 6)]


def variant_name(mutations):
    """Join [['A','100','S'], ...] into 'A100S_...'. Wild-type has no mutations."""
    return "_".join("".join(triplet) for triplet in mutations) or WT_LABEL


def main():
    df = pd.read_csv(COLLAPSED_UNIQUE_VARIANTS_CSV)
    df["node_name"] = [variant_name(ast.literal_eval(m)) for m in df["aa_mutations"]]

    seen = df[ROUNDS].gt(0)
    df["first_round"] = seen.idxmax(axis=1).where(seen.any(axis=1))

    unassigned = df["first_round"].isna().sum()
    if unassigned:
        print(f"note: {unassigned} variants occur in no round and are left out")

    for round_col in ROUNDS:
        n = round_col.split("_")[1]
        group = df[df["first_round"] == round_col].sort_values(
            round_col, ascending=False, kind="stable"
        )

        csv_out = THIS_DIR / f"variants_round{n}.csv"
        txt_out = THIS_DIR / f"round{n}.txt"

        group.drop(columns=["first_round"]).to_csv(csv_out, index=False)
        txt_out.write_text("\n".join(group["node_name"]) + "\n")

        print(f"round {n}: {len(group):4d} variants -> {csv_out.name}, {txt_out.name}")

    total = df["first_round"].notna().sum()
    print(f"total assigned: {total} of {len(df)} variants")


if __name__ == "__main__":
    main()
