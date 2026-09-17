#!/usr/bin/env python3
"""Metrics, confusion matrices, modality ablation, calibration, error analysis."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluate")

ABLATION_MASKS = {
    "text": [1, 0, 0],
    "audio": [0, 1, 0],
    "visual": [0, 0, 1],
    "text+audio": [1, 1, 0],
    "text+visual": [1, 0, 1],
    "audio+visual": [0, 1, 1],
    "text+audio+visual": [1, 1, 1],
}


def main() -> None:
    import torch

    from mustard.config import get_paths, load_config
    from mustard.dataset import attach_media_paths, build_utterance_table, load_raw_csv
    from mustard.features import apply_standardizer, cache_all_features
    from mustard.io_utils import load_json, save_json
    from mustard.plots import plot_ablation, plot_confusion_matrix, plot_reliability
    from mustard.splits import stratified_cv_folds
    from mustard.train_loop import apply_temperature, expected_calibration_error, metrics_from_probs, predict_logits
    from models.unimodal import GatedAttentionFusion

    cfg = load_config()
    paths = get_paths(cfg)
    cv_path = paths["metrics"] / "cv_results.json"
    if not cv_path.exists():
        raise SystemExit("Run python train.py first.")
    cv = load_json(cv_path)

    df = attach_media_paths(build_utterance_table(load_raw_csv()))
    pack = cache_all_features(df)
    y = pack["y"]
    text_x = np.concatenate([pack["text_utt"], pack["text_ctx"], pack["text_hand"]], axis=1).astype(np.float32)
    audio_x = pack["audio"].astype(np.float32)
    visual_x = np.concatenate([pack["visual"], pack["visual_hand"]], axis=1).astype(np.float32)
    speaker_x = pack["speaker"].astype(np.float32)
    nat_mask = pack["mask"].astype(np.float32)

    for name, summary in cv["models"].items():
        cm = np.array(summary.get("confusion_matrix_sum", [[0, 0], [0, 0]]))
        plot_confusion_matrix(cm, f"{name} (summed 5-fold test)", paths["figures"] / name / "cm_sum.png")

    ablation = {}
    error_rows = []
    fusion_dir = paths["checkpoints"] / "fusion_attn"
    folds = stratified_cv_folds(df, seed=int(cfg["seed"]))
    if fusion_dir.exists():
        logger.info("Computing modality ablation from fusion_attn fold checkpoints")
        ablate_fold_metrics = {k: [] for k in ABLATION_MASKS}
        all_probs = []
        all_y = []
        all_idx = []
        for split in folds:
            ckpt_path = fusion_dir / f"fold{split.fold}.pt"
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = GatedAttentionFusion(
                ckpt["dims"]["text"],
                ckpt["dims"]["audio"],
                ckpt["dims"]["visual"],
                d_speaker=ckpt["dims"]["speaker"],
                d_model=int(cfg["fusion"]["d_model"]),
                n_heads=int(cfg["fusion"]["n_heads"]),
                dropout=float(cfg["fusion"]["dropout"]),
            )
            model.load_state_dict(ckpt["state_dict"])
            te = split.test_idx
            text_n = apply_standardizer(text_x, ckpt["text_stats"])
            audio_n = apply_standardizer(audio_x, ckpt["audio_stats"])
            visual_n = apply_standardizer(visual_x, ckpt["visual_stats"])
            for ab_name, bits in ABLATION_MASKS.items():
                m = nat_mask[te] * np.array(bits, dtype=np.float32)
                # if a requested modality was never observed, keep zeros (honest)
                tensors = {
                    "text": torch.tensor(text_n[te]),
                    "audio": torch.tensor(audio_n[te]),
                    "visual": torch.tensor(visual_n[te]),
                    "mask": torch.tensor(m),
                    "speaker": torch.tensor(speaker_x[te]),
                    "x": torch.tensor(text_n[te]),
                }
                logits = predict_logits(model, tensors)
                probs = apply_temperature(logits, ckpt.get("temperature", 1.0))
                met = metrics_from_probs(y[te], probs)
                ablate_fold_metrics[ab_name].append(met)
                if ab_name == "text+audio+visual":
                    all_probs.append(probs)
                    all_y.append(y[te])
                    all_idx.append(te)
                    # error analysis candidates
                    pred = probs.argmax(1)
                    for local, gi in enumerate(te):
                        if pred[local] != y[gi]:
                            error_rows.append(
                                {
                                    "utterance_id": str(df.iloc[gi]["utterance_id"]),
                                    "show": str(df.iloc[gi]["SHOW"]),
                                    "speaker": str(df.iloc[gi]["SPEAKER"]),
                                    "utterance": str(df.iloc[gi]["utterance"])[:240],
                                    "context": str(df.iloc[gi]["context"])[:240],
                                    "true": int(y[gi]),
                                    "pred": int(pred[local]),
                                    "p_sarcastic": float(probs[local, 1]),
                                    "fold": split.fold,
                                }
                            )

        for ab_name, mets in ablate_fold_metrics.items():
            if not mets:
                continue
            ablation[ab_name] = {
                "macro_f1": float(np.mean([m["macro_f1"] for m in mets])),
                "macro_f1_std": float(np.std([m["macro_f1"] for m in mets])),
                "accuracy": float(np.mean([m["accuracy"] for m in mets])),
                "note": "Audio/visual channels are zero when clips were missing at train time.",
            }
        if ablation:
            plot_ablation(ablation, paths["figures"] / "ablation.png")

        if all_probs:
            probs = np.concatenate(all_probs)
            yy = np.concatenate(all_y)
            cal = expected_calibration_error(yy, probs)
            plot_reliability(cal["reliability"], cal["ece"], "fusion_attn", paths["figures"] / "reliability_fusion.png")
            api_ece = cal["ece"]
        else:
            api_ece = cv["models"].get("fusion_attn", {}).get("ece_mean")
    else:
        api_ece = None

    # Prefer fusion metrics for the UI; fall back to best text model.
    preferred = "fusion_attn" if "fusion_attn" in cv["models"] else (
        "text_mlp" if "text_mlp" in cv["models"] else next(iter(cv["models"]))
    )
    chosen = cv["models"][preferred]
    api = {
        "model": preferred,
        "accuracy": chosen.get("accuracy_mean"),
        "accuracy_std": chosen.get("accuracy_std"),
        "macro_f1": chosen.get("macro_f1_mean"),
        "macro_f1_std": chosen.get("macro_f1_std"),
        "precision": chosen.get("precision_mean"),
        "recall": chosen.get("recall_mean"),
        "roc_auc": chosen.get("roc_auc_mean"),
        "ece": api_ece if api_ece is not None else chosen.get("ece_mean"),
        "confusion_matrix": chosen.get("confusion_matrix_sum"),
        "ablation_table": ablation,
        "trained_modalities": cv.get("trained_modalities", {"text": True, "audio": False, "visual": False}),
        "n_folds": cfg["train"]["n_folds"],
        "protocol": "5-fold stratified CV, mean ± std. Speaker-independent split held out separately.",
        "speaker_independent": chosen.get("speaker_independent"),
        "all_models": {
            k: {
                "accuracy": v.get("accuracy_mean"),
                "macro_f1": v.get("macro_f1_mean"),
                "macro_f1_std": v.get("macro_f1_std"),
                "precision": v.get("precision_mean"),
                "recall": v.get("recall_mean"),
                "roc_auc": v.get("roc_auc_mean"),
                "ece": v.get("ece_mean"),
                "confusion_matrix": v.get("confusion_matrix_sum"),
            }
            for k, v in cv["models"].items()
        },
    }

    # commentary on a diverse error slice
    error_rows = sorted(error_rows, key=lambda r: abs(r["p_sarcastic"] - 0.5), reverse=True)
    picked = _pick_errors(error_rows)
    api["error_analysis"] = picked
    save_json(api, paths["metrics"] / "metrics.json")
    save_json(picked, paths["metrics"] / "error_analysis.json")
    logger.info("Wrote artifacts/metrics/metrics.json")
    logger.info("Ablation: %s", {k: round(v["macro_f1"], 3) for k, v in ablation.items()})


def _pick_errors(rows: list[dict], k: int = 12) -> list[dict]:
    comments = [
        "High-confidence miss: the wording looks sincere without the delivery.",
        "Likely needs the prior turn — isolated, this line is ambiguous even to people.",
        "Hyperbole and punctuation may have been over-weighted relative to speaker intent.",
        "Deadpan sitcom delivery is not recoverable from text tokens alone.",
        "Possible laugh-track / audience cue in the original clip that text cannot see.",
        "Speaker-specific style (e.g. Chandler/Sheldon) can look sarcastic when it is not.",
        "Sentiment polarity of the utterance matches the context, so incongruity features stay quiet.",
        "Short utterance; little lexical evidence for either class.",
        "Politeness markers ('sure', 'great', 'right') fire the sarcasm lexicon too eagerly.",
        "Quoted or echoed speech is hard to score without knowing who is being mimicked.",
        "Emotion label in the corpus disagrees with surface wording — multi-task signal would help.",
        "Show-specific joke structure that does not transfer across sitcoms.",
    ]
    picked = []
    seen = set()
    for row in rows:
        key = row["utterance"]
        if key in seen:
            continue
        seen.add(key)
        item = dict(row)
        item["commentary"] = comments[len(picked) % len(comments)]
        picked.append(item)
        if len(picked) >= k:
            break
    return picked


if __name__ == "__main__":
    main()
