"""
MULTI-evolve's fully connected network, trained the way its authors train it.

The third supervised method, and the one built specifically for combining mutations: a small
neural network on one-hot features, trained here on the same pool of variants with at most two
substitutions that 05_ and 06_ use, then asked to score the whole network. Run through the
official implementation rather than a reimplementation.

Clone the official MULTI-evolve repository into third_party/multievolve_official at the repo
root before running. Seeded with seed 2, though training still depends on the torch build.
01_ must have run.

Outputs, written to model_scores/ next to this script:
    multievolve_fcn_scores.csv
"""

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
while not (REPO_ROOT / ".project-root").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import model_comparison as mc  # noqa: E402

OFFICIAL = REPO_ROOT / "third_party" / "multievolve_official"
if not OFFICIAL.is_dir():
    raise SystemExit(f"Clone the official MULTI-evolve repository into {OFFICIAL} before running this step.")
sys.path.insert(0, str(OFFICIAL))

from multievolve.featurizers.base_featurizers import OneHotFeaturizer  # noqa: E402
from multievolve.predictors.neural_net_regressors import Fcn  # noqa: E402

OUTPUT_CSV = mc.MULTIEVOLVE_SCORES_CSV


@dataclass
class ExactSplit:
    splits: dict
    file_attrs: dict


def make_split(df: pd.DataFrame, train_df: pd.DataFrame, seed: int) -> ExactSplit:
    rng = np.random.default_rng(seed)
    wt_idx = train_df.index[train_df["Nham_aa"].eq(0)].tolist()
    candidate_idx = train_df.index[~train_df.index.isin(wt_idx)].to_numpy()
    val_size = max(1, int(round(len(train_df) * 0.15)))
    val_idx = set(rng.choice(candidate_idx, size=val_size, replace=False).tolist())
    train_idx = [idx for idx in train_df.index if idx not in val_idx]
    val_idx = sorted(val_idx)
    return ExactSplit(
        splits={
            "X_train": train_df.loc[train_idx, "aa_seq_model"].to_numpy(),
            "X_val": train_df.loc[val_idx, "aa_seq_model"].to_numpy(),
            "X_test": df["aa_seq_model"].to_numpy(),
            "y_train": train_df.loc[train_idx, "log_kcat_app"].to_numpy(),
            "y_val": train_df.loc[val_idx, "log_kcat_app"].to_numpy(),
            "y_test": df["log_kcat_app"].to_numpy(),
            "split_name": f"low_confidence_order_le_2_seed_{seed}",
        },
        file_attrs={
            "dataset_dir": str(THIS_DIR / "multievolve_fcn_model_cache"),
            "dataset_name": f"low_confidence_order_le_2_seed_{seed}",
        },
    )


def main() -> None:
    seed = 2
    config = {
        "layer_size": 100,
        "num_layers": 1,
        "learning_rate": 0.0001,
        "batch_size": 32,
        "optimizer": "adam",
        "epochs": 300,
    }

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = mc.load_variant_table()
    df = df[df["kcat_app"].gt(0)].copy()
    df["log_kcat_app"] = np.log(df["kcat_app"])
    df["aa_seq_model"] = df["aa_seq"].astype(str)

    train_df = df[df["Nham_aa"].le(2)].copy()
    split = make_split(df, train_df, seed=seed)
    feature = OneHotFeaturizer(protein="ired_low_confidence", use_cache=False)

    model = Fcn(split, feature, config=config, use_cache=False, show_plots=False)
    stats = model.run_model()
    predictions = model.predict(df["aa_seq_model"].to_numpy())

    out = pd.DataFrame(
        {
            "mut_id": df["mut_id"].to_numpy(),
            "multievolve_fcn_log_kcat_pred": predictions,
            "multievolve_fcn_kcat_pred": np.exp(predictions),
            "multievolve_training_pool": df["Nham_aa"].le(2).to_numpy(),
            "multievolve_seed": seed,
            "multievolve_layer_size": config["layer_size"],
            "multievolve_num_layers": config["num_layers"],
            "multievolve_learning_rate": config["learning_rate"],
            "multievolve_official_test_pearson_all_nodes": stats.get("Pearson r", np.nan),
            "multievolve_official_test_spearman_all_nodes": stats.get("Spearman r", np.nan),
        }
    )
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} ({len(out)} rows)")
    print(f"Training pool: {len(train_df)} WT/single/double variants")
    print(f"Scored variants: {len(df)}")


if __name__ == "__main__":
    main()
