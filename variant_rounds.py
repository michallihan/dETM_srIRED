import pandas as pd

from repo_paths import DIMSUM_ROUND_COUNTS_TSV

ROUNDS = tuple(sorted(DIMSUM_ROUND_COUNTS_TSV))

# Rows DiMSum flags that the population-dynamics analysis drops: column -> value removed.
DISCARD = {"STOP": True, "STOP_readthrough": True, "barcode_valid": False, "indel": True}


def mutation_ids(sequences, wild_type):
    """Name each sequence by its differences from the wild type, N-terminal end first."""
    names = []
    for sequence in sequences:
        if len(sequence) != len(wild_type):
            names.append(None)
            continue
        differences = [f"{wild_type[i]}{i + 1}{sequence[i]}"
                       for i in range(len(wild_type)) if wild_type[i] != sequence[i]]
        names.append("WT" if not differences else "-".join(differences))
    return names


def round_variants(round_number):
    """One round's distinct amino-acid variants, named against its own wild-type row."""
    table = pd.read_csv(DIMSUM_ROUND_COUNTS_TSV[round_number], sep="\t")
    for column, discarded in DISCARD.items():
        if column in table.columns:
            table = table[table[column] != discarded]

    collapsed = table.drop_duplicates(subset="aa_seq").reset_index(drop=True)
    wild_type = collapsed.loc[collapsed["WT"] == True, "aa_seq"]  # noqa: E712
    if len(wild_type) != 1:
        raise ValueError(f"round {round_number} has {len(wild_type)} wild-type rows, expected 1")

    return pd.DataFrame({
        "round": round_number,
        "round_label": f"r{round_number}",
        "mut_id": mutation_ids(collapsed["aa_seq"], wild_type.iloc[0]),
        "aa_seq": collapsed["aa_seq"],
        "Nham_aa": collapsed["Nham_aa"],
    })


def variants_per_round():
    """Every round's variants in round order, one row per variant per round it appears in."""
    return pd.concat([round_variants(r) for r in ROUNDS], ignore_index=True)


def first_round(mut_ids=None):
    """The earliest round each variant was seen in, as a mut_id -> round label mapping."""
    rounds = variants_per_round()
    earliest = rounds.groupby("mut_id")["round"].min()
    if mut_ids is not None:
        earliest = earliest.reindex(list(mut_ids))
    return earliest.map(lambda r: f"r{int(r)}" if pd.notna(r) else None)
