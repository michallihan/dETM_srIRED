"""
EVOLVEpro's top-layer model, trained on one-hot sequences.

EVOLVEpro fits a random forest on top of sequence features and uses it to propose the next
round. This runs that top layer on the campaign's own data: the forest is trained on the
variants carrying at most two substitutions and then asked to score the whole network, so the
question in 11_ is whether what it learned from the early rounds transfers to the deeper ones.

One-hot features here, the ESM2 embeddings of 04_ in 06_; everything else about the two runs is
identical, which isolates what the embedding contributes. 01_ must have run.

Outputs, written to model_scores/ next to this script:
    evolvepro_rf_onehot_scores.csv
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import model_comparison as mc  # noqa: E402

OUTPUT_CSV = mc.EVOLVEPRO_ONEHOT_SCORES_CSV

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


def one_hot_sequences(sequences: pd.Series) -> np.ndarray:
    seqs = sequences.astype(str).str.rstrip("*").tolist()
    n = len(seqs)
    length = len(seqs[0])
    x = np.zeros((n, length, len(AA_ALPHABET)), dtype=np.float32)
    for row, seq in enumerate(seqs):
        if len(seq) != length:
            raise ValueError(f"Sequence length mismatch at row {row}: {len(seq)} != {length}")
        for pos, aa in enumerate(seq):
            col = AA_TO_INDEX.get(aa)
            if col is not None:
                x[row, pos, col] = 1.0
    return x.reshape(n, -1)


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = mc.load_variant_table()
    df = df[df["kcat_app"].gt(0)].copy()
    df["log_kcat_app"] = np.log(df["kcat_app"])

    x_all = one_hot_sequences(df["aa_seq"])
    train_mask = df["Nham_aa"].le(2).to_numpy()
    y_train = df.loc[train_mask, "log_kcat_app"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=100,
        criterion="friedman_mse",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=1,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
    )
    model.fit(x_all[train_mask], y_train)
    pred = model.predict(x_all)

    out = pd.DataFrame(
        {
            "mut_id": df["mut_id"].to_numpy(),
            "evolvepro_rf_onehot_log_kcat_pred": pred,
            "evolvepro_rf_onehot_kcat_pred": np.exp(pred),
            "evolvepro_training_pool": train_mask,
            "evolvepro_embedding": "one_hot",
            "evolvepro_regressor": "RandomForestRegressor",
        }
    )
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows)")
    print(f"Training pool: {int(train_mask.sum())} WT/single/double variants")


if __name__ == "__main__":
    main()
