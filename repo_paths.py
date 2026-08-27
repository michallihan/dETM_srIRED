"""
Central place for locating this repository's root and its key data folders.

Every script in this repo is meant to be runnable from anywhere — double-clicked,
run from an IDE, run from a terminal in any working directory — and to work no
matter where the repo itself was cloned or downloaded to on someone's machine.
That means paths must never be hardcoded absolute paths, and must never rely on
the current working directory (os.getcwd()).

The pattern: every script starts by walking up from its own file location until
it finds the ".project-root" marker file at the top of the repo. That gives an
absolute REPO_ROOT that is correct on any machine, which every other path is
then built from.

Usage, at the top of any script anywhere in the repo:

    from pathlib import Path
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR
    while not (REPO_ROOT / ".project-root").exists():
        REPO_ROOT = REPO_ROOT.parent
    sys.path.insert(0, str(REPO_ROOT))

    from repo_paths import FITNESS_HIGH_CONF_CSV, FITNESS_LOW_CONF_CSV

Use THIS_DIR for anything local to the script (its own inputs/outputs); use the
constants below for the shared backbone datasets produced in 01_/02_.
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
FITNESS_HIGH_CONF_DIR = FITNESS_SCORE_DIR / "high_confidence_fitness_scores"
FITNESS_LOW_CONF_DIR = FITNESS_SCORE_DIR / "low_confidence_fitness_scores"
FITNESS_HIGH_CONF_CSV = FITNESS_HIGH_CONF_DIR / "final_fitness_scores_ETM.csv"
FITNESS_LOW_CONF_CSV = FITNESS_LOW_CONF_DIR / "final_fitness_scores_ETM.csv"

# --- 02_long_read_processing: backbone nanopore long-read dataset(s) ---
# Populate with named constants as this section is filled in.
LONG_READ_DIR = REPO_ROOT / "02_long_read_processing"

# --- 03_analysis: per-figure analysis folders ---
ANALYSIS_DIR = REPO_ROOT / "03_analysis"
