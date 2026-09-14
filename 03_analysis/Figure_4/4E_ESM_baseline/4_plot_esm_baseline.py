"""
The panel: what ESM2 made of the evolved variants against what it made of random ones.

Two lines over the five rounds, both median PPL* -- the variants the campaign actually
selected, and the round-matched random null from 3_. Where the observed line sits below the
null, the selected variants look less surprising to the model than random variants carrying
the same number of substitutions from the same pool.

Runs from round_matched_null_summary.csv alone, so it redraws without torch or the ESM2
weights. The summary committed here is the one behind the published panel.

Outputs, written next to this script:
    esm_baseline.svg / .png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

SUMMARY_CSV = THIS_DIR / "round_matched_null_summary.csv"

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text editable
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})


def main():
    summary = pd.read_csv(SUMMARY_CSV)
    rounds = summary["round"].astype(str).str.extract(r"(\d+)")[0].astype(int)

    figure, ax = plt.subplots(figsize=(3.6, 3.0))
    ax.plot(rounds, summary["observed_median_ppl_star"], marker="o", label="Selected")
    ax.plot(rounds, summary["null_median_ppl_star"], marker="s", label="Round-matched random")
    ax.set_xticks(rounds)
    ax.set_xlabel("Round")
    ax.set_ylabel("Median PPL*")
    ax.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    for extension in ("svg", "png"):
        figure.savefig(THIS_DIR / f"esm_baseline.{extension}", dpi=400)
    plt.close(figure)

    print(summary[["round", "observed_n", "null_n", "observed_median_ppl_star",
                   "null_median_ppl_star", "median_delta_observed_minus_null",
                   "mann_whitney_u_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
