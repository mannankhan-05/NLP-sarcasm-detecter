#!/usr/bin/env python3
"""5-fold CV + speaker-independent training for all required model families."""

from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("train")


def _stack_text(pack) -> np.ndarray:
    return np.concatenate([pack["text_utt"], pack["text_ctx"], pack["text_hand"]], axis=1).astype(np.float32)


def _stack_visual(pack) -> np.ndarray:
    return np.concatenate([pack["visual"], pack["visual_hand"]], axis=1).astype(np.float32)


def _standardize_train(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    from mustard.features import apply_standardizer, fit_standardizer

    stats = fit_standardizer(train)
    return apply_standardizer(train, stats), apply_standardizer(other, stats), stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-classical", action="store_true")
    parser.add_argument("--skip-deep", action="store_true")
    parser.add_argument("--folds", type=int, default=None)
    args = parser.parse_args()

    import joblib
    import torch

    from mustard.config import get_paths, load_config
    from mustard.dataset import attach_media_paths, build_utterance_table, load_raw_csv
    from mustard.features import cache_all_features
    from mustard.io_utils import save_json
    from mustard.plots import plot_confusion_matrix
    from mustard.seed import seed_everything
    from mustard.splits import speaker_independent_split, stratified_cv_folds
    from mustard.train_loop import (
        apply_temperature,
        expected_calibration_error,
        metrics_from_probs,
        predict_logits,
        train_torch_classifier,
    )
    from models.classical import make_tfidf_lr, make_tfidf_svm, predict_proba_safe
    from models.unimodal import GatedAttentionFusion, LateFusion, MLPHead

    cfg = load_config()
    paths = get_paths(cfg)
    seed = int(cfg["seed"])
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    df = attach_media_paths(build_utterance_table(load_raw_csv()))
    pack = cache_all_features(df, force=False)
    y = pack["y"]
    texts = (df["context"].fillna("") + " [SEP] " + df["utterance"]).tolist()
    text_x = _stack_text(pack)
    audio_x = pack["audio"].astype(np.float32)
    visual_x = _stack_visual(pack)
    speaker_x = pack["speaker"].astype(np.float32)
    mask = pack["mask"].astype(np.float32)
    trained_modalities = {
        "text": True,
        "audio": bool(mask[:, 1].sum() > 20),
        "visual": bool(mask[:, 2].sum() > 20),
    }
    logger.info("Trained modalities: %s", trained_modalities)

    folds = stratified_cv_folds(df, n_folds=args.folds or cfg["train"]["n_folds"], seed=seed)
    si = speaker_independent_split(df, seed=seed)

    all_results: dict = {"seed": seed, "trained_modalities": trained_modalities, "models": {}}

    if not args.skip_classical:
        for name, factory in [("tfidf_lr", make_tfidf_lr), ("tfidf_svm", make_tfidf_svm)]:
            logger.info("=== %s ===", name)
            all_results["models"][name] = _run_classical(
                name, factory, texts, y, folds, si, paths, predict_proba_safe, joblib, df
            )

    if not args.skip_deep:
        hp = cfg["train"]
        deep_specs = _deep_specs(text_x, audio_x, visual_x, speaker_x, mask, trained_modalities, cfg)

        for spec in deep_specs:
            logger.info("=== %s ===", spec["name"])
            fold_metrics = []
            last_bundle = None
            for split in folds:
                bundle = _train_one_deep(spec, split, text_x, audio_x, visual_x, speaker_x, mask, y, hp, device, seed)
                m = bundle["test_metrics"]
                fold_metrics.append(m)
                cm = np.array(m["confusion_matrix"])
                plot_confusion_matrix(
                    cm,
                    f"{spec['name']} fold {split.fold}",
                    paths["figures"] / spec["name"] / f"cm_fold{split.fold}.png",
                )
                last_bundle = bundle
                ckpt_dir = paths["checkpoints"] / spec["name"]
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    bundle["ckpt"],
                    ckpt_dir / f"fold{split.fold}.pt",
                )
            summary = _summarize(fold_metrics)
            si_bundle = _train_one_deep(spec, si, text_x, audio_x, visual_x, speaker_x, mask, y, hp, device, seed)
            summary["speaker_independent"] = si_bundle["test_metrics"]
            all_results["models"][spec["name"]] = summary
            # persist the last fold as the serving checkpoint (plus mean-fold metrics)
            if last_bundle:
                serve = copy.deepcopy(last_bundle["ckpt"])
                serve["metrics"] = summary
                out = paths["checkpoints"] / spec["name"]
                out.mkdir(parents=True, exist_ok=True)
                torch.save(serve, out / "serve.pt")
            logger.info("%s CV macro-F1 %.3f ± %.3f", spec["name"], summary["macro_f1_mean"], summary["macro_f1_std"])

        # Retrain main fusion on all non-test data from fold 0's train+val+test? No.
        # Serve checkpoint is fold 0's best — instead retrain fusion on 85% of ALL data with a val split.
        from mustard.splits import FoldSplit
        from sklearn.model_selection import train_test_split

        idx = np.arange(len(y))
        tr, va = train_test_split(idx, test_size=0.15, stratify=y, random_state=seed)
        serve_split = FoldSplit(fold=99, train_idx=tr, val_idx=va, test_idx=va)
        fusion_spec = next(s for s in deep_specs if s["name"] == "fusion_attn")
        serve_bundle = _train_one_deep(
            fusion_spec, serve_split, text_x, audio_x, visual_x, speaker_x, mask, y, hp, device, seed
        )
        serve_ckpt = serve_bundle["ckpt"]
        serve_ckpt["metrics"] = all_results["models"].get("fusion_attn", {})
        serve_ckpt["trained_modalities"] = trained_modalities
        serve_ckpt["speaker_names"] = pack["meta"].get("speaker_names", [])
        fusion_dir = paths["checkpoints"] / "fusion_attn"
        fusion_dir.mkdir(parents=True, exist_ok=True)
        torch.save(serve_ckpt, fusion_dir / "serve.pt")
        logger.info("Wrote serving checkpoint (val macro-F1=%.3f)", serve_bundle["ckpt"]["val_macro_f1"])

    save_json(all_results, paths["metrics"] / "cv_results.json")
    logger.info("Wrote %s", paths["metrics"] / "cv_results.json")


def _deep_specs(text_x, audio_x, visual_x, speaker_x, mask, trained_modalities, cfg):
    from models.unimodal import GatedAttentionFusion, LateFusion, MLPHead
    d_t, d_a, d_v, d_s = text_x.shape[1], audio_x.shape[1], visual_x.shape[1], speaker_x.shape[1]
    hidden = int(cfg["train"]["hidden"])
    dropout = float(cfg["train"]["dropout"])
    specs = [
        {
            "name": "text_mlp",
            "kind": "uni",
            "modality": "text",
            "factory": lambda: MLPHead(d_t, hidden, dropout),
        },
    ]
    if trained_modalities["audio"]:
        specs.append({"name": "audio_mlp", "kind": "uni", "modality": "audio", "factory": lambda: MLPHead(d_a, hidden, dropout)})
    if trained_modalities["visual"]:
        specs.append({"name": "visual_mlp", "kind": "uni", "modality": "visual", "factory": lambda: MLPHead(d_v, hidden, dropout)})
    specs.extend(
        [
            {
                "name": "early_fusion",
                "kind": "early",
                "factory": lambda: MLPHead(d_t + d_a + d_v, hidden, dropout),
            },
            {
                "name": "late_fusion",
                "kind": "late",
                "factory": lambda: LateFusion(d_t, d_a, d_v, hidden, dropout),
            },
            {
                "name": "fusion_attn",
                "kind": "attn",
                "factory": lambda: GatedAttentionFusion(
                    d_t, d_a, d_v, d_speaker=d_s,
                    d_model=int(cfg["fusion"]["d_model"]),
                    n_heads=int(cfg["fusion"]["n_heads"]),
                    dropout=float(cfg["fusion"]["dropout"]),
                ),
            },
        ]
    )
    return specs


def _train_one_deep(spec, split, text_x, audio_x, visual_x, speaker_x, mask, y, hp, device, seed):
    import torch

    from mustard.features import apply_standardizer, fit_standardizer
    from mustard.train_loop import apply_temperature, metrics_from_probs, predict_logits, train_torch_classifier

    tr, va, te = split.train_idx, split.val_idx, split.test_idx
    t_stats = fit_standardizer(text_x[tr])
    a_stats = fit_standardizer(audio_x[tr])
    v_stats = fit_standardizer(visual_x[tr])
    text_n = apply_standardizer(text_x, t_stats)
    audio_n = apply_standardizer(audio_x, a_stats)
    visual_n = apply_standardizer(visual_x, v_stats)

    def slice_pack(idx):
        return {
            "text": torch.tensor(text_n[idx]),
            "audio": torch.tensor(audio_n[idx]),
            "visual": torch.tensor(visual_n[idx]),
            "mask": torch.tensor(mask[idx]),
            "speaker": torch.tensor(speaker_x[idx]),
            "x": torch.tensor(_x_for(spec, text_n, audio_n, visual_n, idx)),
            "y": torch.tensor(y[idx], dtype=torch.long),
        }

    model = spec["factory"]()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    drop = float(hp["modality_dropout"]) if spec["kind"] in {"attn", "late", "early"} else 0.0
    result = train_torch_classifier(
        model,
        slice_pack(tr),
        slice_pack(va),
        epochs=int(hp["epochs"]),
        lr=float(hp["lr"]),
        weight_decay=float(hp["weight_decay"]),
        batch_size=int(hp["batch_size"]),
        patience=int(hp["patience"]),
        modality_dropout=drop,
        seed=seed + split.fold,
        device=device,
    )
    model.load_state_dict(result.state_dict)
    test_logits = predict_logits(model, slice_pack(te), device=device)
    test_probs = apply_temperature(test_logits, result.temperature)
    test_metrics = metrics_from_probs(y[te], test_probs)
    test_metrics["ece"] = expected_calibration_error_safe(y[te], test_probs)
    ckpt = {
        "state_dict": result.state_dict,
        "spec": spec["name"],
        "kind": spec["kind"],
        "temperature": result.temperature,
        "val_macro_f1": result.best_val_f1,
        "n_params": n_params,
        "text_stats": t_stats,
        "audio_stats": a_stats,
        "visual_stats": v_stats,
        "dims": {
            "text": int(text_x.shape[1]),
            "audio": int(audio_x.shape[1]),
            "visual": int(visual_x.shape[1]),
            "speaker": int(speaker_x.shape[1]),
        },
        "hyperparameters": {
            "epochs": hp["epochs"],
            "lr": hp["lr"],
            "weight_decay": hp["weight_decay"],
            "dropout": hp["dropout"],
            "hidden": hp["hidden"],
            "batch_size": hp["batch_size"],
            "modality_dropout": hp["modality_dropout"],
            "seed": seed,
        },
        "history": result.history,
    }
    return {"ckpt": ckpt, "test_metrics": test_metrics, "probs": test_probs, "idx": te}


def expected_calibration_error_safe(y, probs):
    from mustard.train_loop import expected_calibration_error

    return expected_calibration_error(y, probs)["ece"]


def _x_for(spec, text_n, audio_n, visual_n, idx):
    if spec["kind"] == "uni":
        m = spec["modality"]
        if m == "text":
            return text_n[idx]
        if m == "audio":
            return audio_n[idx]
        return visual_n[idx]
    if spec["kind"] == "early":
        return np.concatenate([text_n[idx], audio_n[idx], visual_n[idx]], axis=1)
    return text_n[idx]


def _run_classical(name, factory, texts, y, folds, si, paths, predict_proba_safe, joblib, df):
    from mustard.plots import plot_confusion_matrix
    from mustard.train_loop import expected_calibration_error, metrics_from_probs

    fold_metrics = []
    ckpt_dir = paths["checkpoints"] / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for split in folds:
        model = factory()
        X_tr = [texts[i] for i in split.train_idx]
        # val unused for sklearn defaults; still not touching test for fitting
        model.fit(X_tr, y[split.train_idx])
        X_te = [texts[i] for i in split.test_idx]
        probs = predict_proba_safe(model, X_te)
        m = metrics_from_probs(y[split.test_idx], probs)
        m["ece"] = expected_calibration_error(y[split.test_idx], probs)["ece"]
        fold_metrics.append(m)
        plot_confusion_matrix(
            np.array(m["confusion_matrix"]),
            f"{name} fold {split.fold}",
            paths["figures"] / name / f"cm_fold{split.fold}.png",
        )
        joblib.dump(model, ckpt_dir / f"fold{split.fold}.joblib")
    # serve: fit on all but a small val is not needed; fit on 100% for demo serving of classical
    serve = factory()
    serve.fit(texts, y)
    joblib.dump(serve, ckpt_dir / "serve.joblib")

    si_model = factory()
    si_model.fit([texts[i] for i in np.concatenate([si.train_idx, si.val_idx])], y[np.concatenate([si.train_idx, si.val_idx])])
    si_probs = predict_proba_safe(si_model, [texts[i] for i in si.test_idx])
    si_metrics = metrics_from_probs(y[si.test_idx], si_probs)
    summary = _summarize(fold_metrics)
    summary["speaker_independent"] = si_metrics
    return summary


def _summarize(fold_metrics: list[dict]) -> dict:
    keys = ["accuracy", "precision", "recall", "macro_f1", "weighted_f1", "roc_auc", "ece"]
    out = {"folds": fold_metrics}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None and m[k] == m[k]]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else None
        out[f"{k}_std"] = float(np.std(vals)) if vals else None
    # summed confusion matrix
    cms = [np.array(m["confusion_matrix"]) for m in fold_metrics]
    out["confusion_matrix_sum"] = np.sum(cms, axis=0).astype(int).tolist()
    return out


if __name__ == "__main__":
    main()
