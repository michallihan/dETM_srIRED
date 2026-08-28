# Rike et al. — evolutionary trajectory mapping

Code and processed data for the figures in the Rike et al. paper.

## Structure

```
01_fitness_score_processing/   backbone dataset: fitness scores from lrDMS
  02a_high_confidence_fitness_scores/
  02b_low_confidence_fitness_scores/
02_long_read_processing/       backbone dataset: nanopore long-read processing
03_analysis/                   one folder per figure panel
  Figure_1/  Figure_2/  Figure_3/  Figure_4/  SI_figures/
```

`01_` and `02_` produce the backbone datasets everything else builds on. Each folder under
`03_analysis` is one figure panel: it either reads from `01_`/`02_` or processes its own
intermediate data, and holds the script, its outputs and a README describing them.

## Setting up

```
pip install -r requirements.txt
```

## Path conventions

No script uses an absolute path or depends on the directory it is run from. Each anchors
its paths to its own file location, so the repo works wherever it is cloned or downloaded.

A script working only within its own folder:

```python
from pathlib import Path
THIS_DIR = Path(__file__).resolve().parent
```

A script that also needs the backbone datasets walks up to the repo root, marked by the
`.project-root` file, and imports the shared constants from `repo_paths.py`:

```python
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import FITNESS_HIGH_CONF_CSV
```

Do not delete `.project-root` — it is how scripts find the repo root.
