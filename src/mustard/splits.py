"""Train / validation folds. Test folds are never used for tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import load_config
from .seed import DEFAULT_SEED


@dataclass
class FoldSplit:
    fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def stratified_cv_folds(
    df: pd.DataFrame,
    n_folds: int | None = None,
    seed: int = DEFAULT_SEED,
    val_size: float = 0.15,
) -> list[FoldSplit]:
    """5-fold stratified CV. Each test fold is held out; val is carved from train only."""
    cfg = load_config()
    n_folds = n_folds or int(cfg["train"]["n_folds"])
    y = df["sarcasm"].to_numpy()
    idx = np.arange(len(df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[FoldSplit] = []
    for k, (trainval, test) in enumerate(skf.split(idx, y)):
        y_tv = y[trainval]
        tr, va = train_test_split(
            trainval,
            test_size=val_size,
            stratify=y_tv,
            random_state=seed + k,
        )
        folds.append(
            FoldSplit(
                fold=k,
                train_idx=np.asarray(tr),
                val_idx=np.asarray(va),
                test_idx=np.asarray(test),
            )
        )
    return folds


def official_or_cv_folds(
    df: pd.DataFrame,
    n_folds: int | None = None,
    seed: int = DEFAULT_SEED,
    val_size: float = 0.15,
) -> list[FoldSplit]:
    """Use provided train/val/test labels when present; otherwise 5-fold CV."""
    if "split" in df.columns:
        labels = set(df["split"].astype(str).str.lower())
        if {"train", "val", "test"} <= labels:
            split = df["split"].astype(str).str.lower()
            train_idx = np.where(split == "train")[0]
            val_idx = np.where(split == "val")[0]
            test_idx = np.where(split == "test")[0]
            if min(len(train_idx), len(val_idx), len(test_idx)) > 0:
                return [
                    FoldSplit(fold=0, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
                ]
    return stratified_cv_folds(df, n_folds=n_folds, seed=seed, val_size=val_size)


def speaker_independent_split(
    df: pd.DataFrame,
    holdout_show: str | None = None,
    seed: int = DEFAULT_SEED,
    val_size: float = 0.15,
) -> FoldSplit:
    """Hold out one show as the test set (speaker/show-independent protocol)."""
    cfg = load_config()
    holdout = (holdout_show or cfg["train"]["speaker_independent_holdout_show"]).upper()
    shows = set(df["SHOW"].astype(str).str.upper())
    if holdout not in shows:
        # fall back to the largest show that is not the majority of the data
        counts = df["SHOW"].astype(str).str.upper().value_counts()
        holdout = counts.index[-1]
    test_mask = df["SHOW"].astype(str).str.upper() == holdout
    test_idx = np.where(test_mask)[0]
    trainval_idx = np.where(~test_mask)[0]
    y = df["sarcasm"].to_numpy()[trainval_idx]
    tr, va = train_test_split(
        trainval_idx,
        test_size=val_size,
        stratify=y,
        random_state=seed,
    )
    return FoldSplit(fold=-1, train_idx=np.asarray(tr), val_idx=np.asarray(va), test_idx=np.asarray(test_idx))
