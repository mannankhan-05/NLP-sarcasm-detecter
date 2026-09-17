#!/usr/bin/env python3
"""Train the UI text model on isolated utterances (how the web form is actually used)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    from mustard.config import get_paths
    from mustard.dataset import build_utterance_table, load_raw_csv
    from mustard.features import fit_standardizer, handcrafted_text_features, load_feature_pack
    from mustard.seed import seed_everything
    from mustard.text_utils import aggressive_clean

    seed_everything(42)
    paths = get_paths()
    pack = load_feature_pack()
    df = build_utterance_table(load_raw_csv())
    y = pack["y"]
    hands = np.stack([handcrafted_text_features(u, "") for u in df["utterance"]], axis=0)
    x = np.concatenate([pack["text_utt"], hands], axis=1).astype(np.float32)
    stats = fit_standardizer(x)
    xs = (x - stats["mean"]) / stats["std"]

    emb = LogisticRegression(max_iter=2500, C=0.4, class_weight="balanced", solver="liblinear")
    emb.fit(xs, y)

    tfidf = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=aggressive_clean,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=8000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(max_iter=2500, C=1.5, class_weight="balanced", solver="liblinear"),
            ),
        ]
    )
    tfidf.fit(df["utterance"].tolist(), y)

    out = paths["checkpoints"] / "text_ui"
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "emb": emb,
            "tfidf": tfidf,
            "stats": stats,
            "utt_dim": int(pack["text_utt"].shape[1]),
            "hand_dim": int(hands.shape[1]),
            "blend_emb": 0.6,
            "blend_tfidf": 0.4,
            "trained_on": "utterance_only",
        },
        out / "serve.joblib",
    )
    logging.info("Wrote %s", out / "serve.joblib")


if __name__ == "__main__":
    main()
