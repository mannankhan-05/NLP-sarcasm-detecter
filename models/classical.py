from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from mustard.text_utils import aggressive_clean


def make_tfidf_lr() -> Pipeline:
    return Pipeline(
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
                LogisticRegression(
                    max_iter=2000,
                    C=2.0,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def make_tfidf_svm() -> Pipeline:
    base = LinearSVC(C=1.0, class_weight="balanced", max_iter=4000)
    return Pipeline(
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
            ("clf", CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)),
        ]
    )


def predict_proba_safe(model, texts) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)
    scores = model.decision_function(texts)
    if scores.ndim == 1:
        scores = np.vstack([-scores, scores]).T
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)
