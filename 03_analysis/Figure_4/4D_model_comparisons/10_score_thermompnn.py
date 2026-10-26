"""
ThermoMPNN's stability prediction, summed over each variant's substitutions.

The one method here that reads the structure rather than the sequence family: ThermoMPNN
predicts the change in folding free energy for every possible point substitution in 5OCM, and
a variant's score is the sum over the substitutions it carries, negated so that a more stable
variant scores higher like every other method in the comparison. Whether stability tracks
activity along these trajectories is the question 11_ puts to it.

The saturation mutagenesis table is the one ThermoMPNN's official inference workflow produced
for 5OCM and is kept next to this script; regenerate it with ThermoMPNN itself if needed.
This script only sums it, so it needs nothing beyond pandas. 01_ must have run.

Outputs, written to model_scores/ next to this script:
    thermompnn_scores.csv
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

import model_comparison as mc  # noqa: E402

THERMOMPNN_SSM = mc.THERMOMPNN_INFERENCE_CSV
OUTPUT_CSV = mc.THERMOMPNN_SCORES_CSV


def mutation_tokens(mut_id: str) -> list[str]:
    if not mut_id or mut_id == "WT" or str(mut_id).lower() == "nan":
        return []
    return str(mut_id).split("-")


def parse_mutation(token: str) -> tuple[str, int, str]:
    return token[0], int(token[1:-1]), token[-1]


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = mc.load_variant_table()
    ssm = pd.read_csv(THERMOMPNN_SSM)
    ssm["position"] = ssm["position"].astype(int)
    ssm["ddG_pred"] = pd.to_numeric(ssm["ddG_pred"])
    lookup = {
        (int(row.position), str(row.wildtype), str(row.mutation)): float(row.ddG_pred)
        for row in ssm.itertuples(index=False)
    }

    rows = []
    for variant in df["mut_id"].astype(str):
        ddg_sum = 0.0
        n_scored = 0
        missing = []
        for token in mutation_tokens(variant):
            wt_aa, one_based_pos, mut_aa = parse_mutation(token)
            key = (one_based_pos - 1, wt_aa, mut_aa)
            if key not in lookup:
                missing.append(token)
                continue
            ddg_sum += lookup[key]
            n_scored += 1
        rows.append(
            {
                "variant_id": variant,
                "thermompnn_sum_ddg_pred": ddg_sum if not missing else pd.NA,
                "thermompnn_naturalness": -ddg_sum if not missing else pd.NA,
                "n_mutations_scored": n_scored,
                "missing_mutations": ";".join(missing),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)
    n_missing = out["missing_mutations"].astype(str).ne("").sum()
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows, {n_missing} variants with missing ThermoMPNN terms)")


if __name__ == "__main__":
    main()
