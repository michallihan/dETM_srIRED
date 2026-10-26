"""
The variant table every model in this folder is scored against.

Takes the low-confidence fitness table, drops the rows the comparison cannot use, converts
fitness to apparent kcat with the repo's fitted conversion, and adds the MSA conservation
score that serves as the simplest baseline in the comparison. Nothing here involves a model:
it is built from the measured data and the alignment alone, which is what lets every scoring
script below read one fixed table rather than each filtering the dataset its own way.

The low-confidence table is used rather than the high-confidence one because the comparison
needs the connected network of single-mutation steps, and the high-confidence calls are too
sparse to connect.

Outputs, written next to this script:
    variant_table_with_kcat_app.csv
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import model_comparison as mc  # noqa: E402

from repo_paths import FITNESS_LOW_CONF_CSV  # noqa: E402

INPUT_CSV = FITNESS_LOW_CONF_CSV
OUTPUT_CSV = mc.VARIANT_TABLE_CSV


def main():
    df = mc.load_fitness_table(INPUT_CSV)
    df["kcat_app"] = mc.to_kcat_app(df["fitness_final"])
    df["kcat_app_error"] = mc.kcat_app_error(df["kcat_app"], df["sigma_final"])

    wt_seq = mc.wt_sequence(df)
    freqs = mc.msa_position_frequencies(wt_seq)
    df["msa_conservation_log_prob"] = df["mut_id"].map(lambda mut_id: mc.variant_msa_score(mut_id, freqs))

    df.to_csv(OUTPUT_CSV, index=False)

    a, b = mc.kcat_app_fit()
    print(f"Wrote {OUTPUT_CSV}")
    print(f"{len(df)} variants, mutation loads 0-{int(df['Nham_aa'].max())}, WT {len(wt_seq)} residues")
    print(f"kcat(app) conversion: kcat = {a:.6f} * exp({b:.6f} * fitness_final)")


if __name__ == "__main__":
    main()
