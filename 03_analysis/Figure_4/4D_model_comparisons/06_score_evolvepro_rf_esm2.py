"""
The same EVOLVEpro random forest, trained on ESM2 embeddings instead of one-hot features.

Identical to 05_ in every other respect: same training pool of variants with at most two
substitutions, same forest, same seed. The only change is the representation, so the gap
between the two accuracies in 11_ is what the protein language model's features are worth
to a supervised model on this landscape.

04_ must have run to produce the embeddings, and 01_ before it.

Outputs, written to model_scores/ next to this script:
    evolvepro_rf_esm2_650m_scores.csv
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

EMBEDDINGS_CSV = mc.ESM2_EMBEDDINGS_CSV
OUTPUT_CSV = mc.EVOLVEPRO_ESM2_SCORES_CSV


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not EMBEDDINGS_CSV.exists():
        raise SystemExit(f"Run 04_build_esm2_embeddings.py first: {EMBEDDINGS_CSV.name} is missing.")
    df = mc.load_variant_table()
    df = df[df["kcat_app"].gt(0)].copy()
    df["log_kcat_app"] = np.log(df["kcat_app"])

    embeddings = pd.read_csv(EMBEDDINGS_CSV, index_col=0)
    missing = sorted(set(df["mut_id"]) - set(embeddings.index))
    if missing:
        raise ValueError(f"Missing ESM embeddings for {len(missing)} variants, first: {missing[:5]}")
    x_all = embeddings.loc[df["mut_id"]].to_numpy(dtype=np.float32)
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
            "evolvepro_rf_esm2_650m_log_kcat_pred": pred,
            "evolvepro_rf_esm2_650m_kcat_pred": np.exp(pred),
            "evolvepro_training_pool": train_mask,
            "evolvepro_embedding": "esm2_t33_650M_UR50D_mean_layer33",
            "evolvepro_regressor": "RandomForestRegressor",
        }
    )
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows)")
    print(f"Training pool: {int(train_mask.sum())} WT/single/double variants")
    print(f"Embedding matrix: {x_all.shape}")


if __name__ == "__main__":
    main()
