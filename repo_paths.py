"""
Locates the repository root and its key data files.

Scripts anywhere in the repo find the root by walking up from their own file location until
they reach the ".project-root" marker, then import the constants below. Paths therefore
resolve correctly regardless of where the repo is cloned and which directory a script is
run from.

Usage, at the top of a script:

    from pathlib import Path
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR
    while not (REPO_ROOT / ".project-root").exists():
        REPO_ROOT = REPO_ROOT.parent
    sys.path.insert(0, str(REPO_ROOT))

    from repo_paths import FITNESS_HIGH_CONF_CSV

Use THIS_DIR for files local to the script; use the constants below for the shared
datasets in 01_ and 02_.
"""

from pathlib import Path


def find_repo_root(start: Path = None, marker: str = ".project-root") -> Path:
    """Walk up from `start` (default: this file's folder) until `marker` is found."""
    here = start or Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    raise RuntimeError(f"Could not locate the repo root (no '{marker}' found above {here}).")


REPO_ROOT = find_repo_root()

# --- 01_fitness_score_processing: backbone fitness score datasets ---
FITNESS_SCORE_DIR = REPO_ROOT / "01_fitness_score_processing"
FITNESS_HIGH_CONF_DIR = FITNESS_SCORE_DIR / "02a_high_confidence_fitness_scores"
FITNESS_LOW_CONF_DIR = FITNESS_SCORE_DIR / "02b_low_confidence_fitness_scores"
FITNESS_HIGH_CONF_CSV = FITNESS_HIGH_CONF_DIR / "final_fitness_scores_ETM.csv"
FITNESS_LOW_CONF_CSV = FITNESS_LOW_CONF_DIR / "final_fitness_scores_ETM.csv"

# High-confidence table with modelled apparent kcat joined on (kcat_app, kcat_app_error).
# Source of the Figure 2B fitness landscape.
FITNESS_HIGH_CONF_KCAT_CSV = (
    FITNESS_HIGH_CONF_DIR / "final_fitness_scores_ETM_with_kcat_app.csv"
)

# --- 01_fitness_score_processing/01_lrDMS_pipeline: the DiMSum runs upstream of the
#     fitness scores. Reads are not held in the repo; see 01_NGS. ---
LRDMS_PIPELINE_DIR = FITNESS_SCORE_DIR / "01_lrDMS_pipeline"

# Barcode-identity map linking nucleotide barcodes to variants, from the nanopore long
# reads. Copied from the original runs; to be produced by the nanopore pipeline in time.
LRDMS_BARCODE_IDENTITY_TXT = LRDMS_PIPELINE_DIR / "VIF_all_rounds_upd.txt"

# DiMSum of the sorted droplet libraries, one run per FACS gate. These two tables are the
# lrDMS_all_variants_lenient.csv and _stringent.csv that 02a_ and 02b_ read.
DIMSUM_DMS_DIR = LRDMS_PIPELINE_DIR / "02_DiMSum_DMS"
DIMSUM_DMS_LENIENT_CSV = DIMSUM_DMS_DIR / "lenient_gate" / "all_variants.csv"
DIMSUM_DMS_STRINGENT_CSV = DIMSUM_DMS_DIR / "stringent_gate" / "all_variants.csv"

# DiMSum of each selection round, counting every variant in that round's read pool.
DIMSUM_POPULATION_DYNAMICS_DIR = LRDMS_PIPELINE_DIR / "03_DiMSum_population_dynamics"
DIMSUM_ROUND_COUNTS_TSV = {
    round_number: DIMSUM_POPULATION_DYNAMICS_DIR / f"round{round_number}" / "variant_data_merge.tsv"
    for round_number in range(1, 6)
}

# --- 02_long_read_processing: backbone nanopore long-read dataset(s) ---
LONG_READ_DIR = REPO_ROOT / "02_long_read_processing"

# Collapsed table of unique amino-acid variants across the five CHX/CPA rounds
# (1341 mutants + wild-type). Columns: name, aa_mutations, aa_mutation_number,
# aa_sequence, total_occurrences, round_1 ... round_5.
COLLAPSED_UNIQUE_VARIANTS_CSV = (
    LONG_READ_DIR / "Rike_IRED_evolution_CHX_CPA_dictionary_collapsed_unique_variants.csv"
)

# --- 04_biophysics: laboratory measurements on purified variants ---
# Measured by hand, not derived from anything else in the repo: melting temperature and
# Michaelis-Menten kinetics for the variants that were expressed and purified, keyed by the
# same mutation identifier the fitness table uses, with commas where that table uses dashes.
BIOPHYSICS_DIR = REPO_ROOT / "04_biophysics"
BIOPHYSICAL_DATA_CSV = BIOPHYSICS_DIR / "srired_chx_cpa_biophysical_data.csv"

# --- 03_analysis: per-figure analysis folders ---
ANALYSIS_DIR = REPO_ROOT / "03_analysis"
