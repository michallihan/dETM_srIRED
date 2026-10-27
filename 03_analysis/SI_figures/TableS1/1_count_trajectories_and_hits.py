import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from repo_paths import (  # noqa: E402
    COLLAPSED_UNIQUE_VARIANTS_CSV,
    FITNESS_HIGH_CONF_KCAT_CSV,
)
from variant_rounds import mutation_ids, variants_per_round  # noqa: E402

FOLD_OVER_WILD_TYPE = 4  # the 5x cut was also drawn at one point; 4x is the one that is used

TRAJECTORY_NOTES = [
    ("round", "Round the variant was picked up in. A variant seen in several rounds has a row "
              "for each."),
    ("mut_id", "Variant identity, substitutions ordered by residue position."),
    ("n_mutations", "Number of amino-acid substitutions relative to the wild type."),
    ("fitness_final", "From final_fitness_scores_ETM_with_kcat_app.csv."),
    ("sigma_final", "From the same table."),
    ("kcat_app", "Apparent kcat modelled from the fitness score (s-1), from the same table."),
    ("kcat_app_error", "Its error (s-1), from the same table."),
    ("kcat_app_min", "kcat_app - kcat_app_error: the low end of the variant's apparent kcat."),
    ("n_trajectories", "Paths from the wild type adding one substitution at a time where every "
                       "intermediate is itself in the 594-variant fitness dataset. 0 means no "
                       "fully measured path exists; the wild type is 1; the upper bound is "
                       "n_mutations factorial."),
    ("n_trajectories_campaign", "The same count against the 1,342-variant nanopore dictionary, "
                                "which holds every variant seen at any point in the campaign "
                                "whether or not it was scored."),
]

# The hits sheet keeps one row per variant, so its round column means something narrower, and
# it carries one column the per-round sheet does not.
HIT_ROUND_NOTE = ("round", "Earliest round the variant was picked up in. Variants seen in "
                           "several rounds appear once, at their first appearance.")
FOLD_NOTE = ("fold_over_WT", "kcat_app(variant) / kcat_app(wild type), point estimates, for "
                             "reference. It is not what the hit call tests.")


def substitutions(mut_id):
    """A variant name as its substitutions, ordered by residue position."""
    if mut_id == "WT":
        return ()
    return tuple(sorted(mut_id.split("-"), key=lambda s: int(s[1:-1])))


def name(subs):
    return "WT" if not subs else "-".join(subs)


def trajectory_counter(reference):
    """Count paths from the wild type whose every intermediate is in `reference`."""

    @lru_cache(maxsize=None)
    def count(subs):
        if not subs:
            return 1
        total = 0
        for index in range(len(subs)):
            rest = subs[:index] + subs[index + 1:]
            if not rest or name(rest) in reference:
                total += count(rest)
        return total

    return count


def nanopore_variants():
    """The nanopore dictionary's variants, named the way the fitness table names them."""
    collapsed = pd.read_csv(COLLAPSED_UNIQUE_VARIANTS_CSV)
    wild_type = collapsed.loc[collapsed["aa_mutation_number"] == 0, "aa_sequence"]
    if len(wild_type) != 1:
        raise ValueError(f"the nanopore dictionary has {len(wild_type)} wild-type rows")
    # Its own aa_mutations column would do, but naming from the sequence keeps one definition.
    return set(mutation_ids(collapsed["aa_sequence"].str.rstrip("*"),
                            wild_type.iloc[0].rstrip("*")))


def write_workbook(path, table, sheet, notes, preamble=()):
    """One data sheet and one sheet defining its columns."""
    legend = [("Column", "Definition"), *notes]
    if preamble:
        legend = [*preamble, ("", ""), *legend]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame(legend).to_excel(writer, sheet_name="Legend", index=False, header=False)
        for worksheet, widths in ((writer.sheets[sheet], None),
                                  (writer.sheets["Legend"], (34, 110))):
            if widths is None:
                for column in worksheet.columns:
                    longest = max(len(str(cell.value)) for cell in column)
                    worksheet.column_dimensions[column[0].column_letter].width = min(longest + 2, 34)
            else:
                for letter, width in zip("AB", widths):
                    worksheet.column_dimensions[letter].width = width


def main():
    fitness = pd.read_csv(FITNESS_HIGH_CONF_KCAT_CSV)
    measured = set(fitness["mut_id"])
    campaign = nanopore_variants()
    print(f"{len(measured)} scored variants, {len(campaign)} in the nanopore dictionary, "
          f"{len(measured & campaign)} in both")

    count_measured = trajectory_counter(measured)
    count_campaign = trajectory_counter(campaign)

    scored = fitness[["mut_id", "Nham_aa", "fitness_final", "sigma_final",
                      "kcat_app", "kcat_app_error"]].rename(columns={"Nham_aa": "n_mutations"})
    scored["kcat_app_min"] = scored["kcat_app"] - scored["kcat_app_error"]
    scored["n_trajectories"] = [count_measured(substitutions(m)) for m in scored["mut_id"]]
    scored["n_trajectories_campaign"] = [count_campaign(substitutions(m)) for m in scored["mut_id"]]

    columns = ["round", "mut_id", "n_mutations", "fitness_final", "sigma_final",
               "kcat_app", "kcat_app_error", "kcat_app_min",
               "n_trajectories", "n_trajectories_campaign"]

    rounds = variants_per_round().rename(columns={"round_label": "round"})
    per_round = rounds[["round", "mut_id"]].merge(scored, on="mut_id", how="inner")
    per_round = per_round.sort_values(["n_trajectories", "fitness_final"], ascending=False,
                                      kind="stable").reset_index(drop=True)
    write_workbook(THIS_DIR / "trajectory_counts.xlsx", per_round[columns],
                   "Trajectory_counts", TRAJECTORY_NOTES)
    print(f"{len(per_round)} variant-round rows over {per_round['mut_id'].nunique()} variants")

    wild_type = fitness[fitness["mut_id"] == "WT"].iloc[0]
    threshold = FOLD_OVER_WILD_TYPE * (wild_type["kcat_app"] + wild_type["kcat_app_error"])

    earliest = per_round.groupby("mut_id", as_index=False)["round"].min()
    hits = scored[(scored["mut_id"] != "WT") & (scored["kcat_app_min"] > threshold)]
    hits = hits.merge(earliest, on="mut_id", how="left")
    hits["fold_over_WT"] = hits["kcat_app"] / wild_type["kcat_app"]
    hits = hits.sort_values(["n_trajectories_campaign", "fitness_final"], ascending=False,
                            kind="stable").reset_index(drop=True)

    preamble = [
        ("Hit criterion", "kcat_app(variant) - kcat_app_error(variant)  >  "
                          f"{FOLD_OVER_WILD_TYPE} x [ kcat_app(WT) + kcat_app_error(WT) ]"),
        ("kcat_app (WT)", wild_type["kcat_app"]),
        ("kcat_app_error (WT)", wild_type["kcat_app_error"]),
        ("Threshold", threshold),
        ("Variants passing", len(hits)),
        ("Variants tested", int((fitness["mut_id"] != "WT").sum())),
    ]
    hit_notes = [HIT_ROUND_NOTE if column == "round" else (column, definition)
                 for column, definition in TRAJECTORY_NOTES]
    hit_notes.insert(-2, FOLD_NOTE)
    write_workbook(THIS_DIR / "confident_hits.xlsx",
                   hits[[*columns[:-2], "fold_over_WT", *columns[-2:]]],
                   "Confident_hits", hit_notes, preamble)
    print(f"threshold {threshold:.6f} s-1; {len(hits)} of "
          f"{int((fitness['mut_id'] != 'WT').sum())} variants pass")


if __name__ == "__main__":
    main()
