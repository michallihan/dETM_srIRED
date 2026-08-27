# Rike et al. — evolutionary trajectory mapping

Code and processed data for the figures in the Rike et al. paper.

## Structure

```
01_fitness_score_processing/   backbone dataset: fitness scores from lrDMS
  high_confidence_fitness_scores/
  low_confidence_fitness_scores/
02_long_read_processing/       backbone dataset: nanopore long-read processing
03_analysis/                   one folder per figure/panel
  Figure_2/
    2AC_trajectory_mapping/
    2B_fitness_landscape/
    2D_hits_analysis/
    2E_mutational_frequency/
  Figure_3/
    3AB_global_epistasis/
    3C_mutational_frequency/
  Figure_4/
    4A_MSA_trajectories/
    4D_model_comparisons/
    4E_ESM_benchmarking/
    4G_counts_analysis/
  SI_figures/
```

`01_` and `02_` produce the backbone datasets everything else builds on. Each folder
under `03_analysis` is one figure panel: it either reads directly from `01_`/`02_`, or
processes its own intermediate data, and contains both the script that produces the
panel and the resulting figure file.

## Setting up

```
pip install -r requirements.txt
```

## Path conventions

Nothing in this repo uses absolute paths or relies on the working directory a script
happens to be run from (`os.getcwd()`). Every script anchors its paths to its own file
location instead, so the repo works correctly no matter where it's cloned or downloaded
to, and no matter what directory you run a script from.

A script that only reads/writes files in its own folder does this:

```python
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
```

A script that also needs the shared backbone datasets in `01_`/`02_` walks up to the
repo root (marked by the `.project-root` file at the top of the repo) and imports the
shared path constants from `repo_paths.py`:

```python
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import FITNESS_HIGH_CONF_CSV, FITNESS_LOW_CONF_CSV
```

Do not delete `.project-root` — it's how scripts find the repo root.

## Running `01_fitness_score_processing`

Each of `high_confidence_fitness_scores/` and `low_confidence_fitness_scores/` is
self-contained: run its `1_process_fitness_scores_lrDMS_V4.py` from anywhere and it
will read its own `lrDMS_all_variants_*.csv` inputs and write
`final_fitness_scores_ETM.csv` plus the diagnostic plots back into that same folder.
The two folders differ only in which QC cutoffs are applied (see the docstring at the
top of each script); the underlying `function_library_lrDMS_analysis.py` is identical.
