"""Token, modality, visual, and audio explanations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mustard.config import get_paths
from mustard.features import FrozenTextEncoder, extract_pitch_energy_contour
from mustard.text_utils import tokenize_words, preserve_text


def gradient_token_attributions(text: str, clf_weight: np.ndarray, encoder: FrozenTextEncoder | None = None) -> list[dict]:
    """Per-token alignment with the fusion text-projection sarcasm direction.

    CLS-only gradients cannot highlight words (the head consumes [CLS] alone),
    so we score each token hidden state against that direction. Lexicon fallback
    if the encoder is down or the direction is degenerate.
    """
    text = preserve_text(text)
    encoder = encoder or FrozenTextEncoder()
    if not encoder.available:
        return heuristic_token_attributions(text)

    import torch

    tokens, hidden, _ = encoder.encode_tokens(text)
    if hidden is None:
        return heuristic_token_attributions(text)

    h = hidden.detach()[0]
    w = torch.tensor(clf_weight.reshape(-1), dtype=h.dtype, device=h.device)
    w = w[: h.size(-1)]
    attr = (h * w).sum(dim=-1).cpu().numpy()
    skip = {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"}
    out = []
    for tok, a in zip(tokens, attr):
        if tok in skip:
            continue
        pretty = tok.replace("Ġ", "").replace("##", "")
        if not pretty:
            continue
        out.append({"token": pretty, "score": float(a)})
    aligned = _normalize_token_scores(out)
    if not aligned or max(abs(it["score"]) for it in aligned) < 1e-6:
        return heuristic_token_attributions(text)
    return aligned


def heuristic_token_attributions(text: str) -> list[dict]:
    from mustard.text_utils import HYPERBOLE, INTENSIFIERS
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    tokens = tokenize_words(text)
    scores = []
    for tok in tokens:
        low = tok.lower()
        s = 0.0
        if tok in {"!", "?", "..."} or tok == '"':
            s += 0.55
        if low in INTENSIFIERS:
            s += 0.45
        if low in HYPERBOLE:
            s += 0.4
        if tok.isupper() and len(tok) > 1:
            s += 0.3
        polar = analyzer.polarity_scores(tok)["compound"]
        s += abs(polar) * 0.5
        scores.append({"token": tok, "score": float(s)})
    return _normalize_token_scores(scores)


def _normalize_token_scores(items: list[dict]) -> list[dict]:
    if not items:
        return items
    mag = max(abs(it["score"]) for it in items) or 1.0
    for it in items:
        it["score"] = round(it["score"] / mag, 4)
    return items


def leave_one_modality_contributions(predict_fn, modalities: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float]:
    """predict_fn(text, audio, visual, mask) -> p_sarcastic."""
    names = ["text", "audio", "visual"]
    base = float(predict_fn(modalities["text"], modalities["audio"], modalities["visual"], mask))
    deltas = {}
    for i, name in enumerate(names):
        if mask[i] < 0.5:
            deltas[name] = 0.0
            continue
        m = mask.copy()
        m[i] = 0.0
        if m.sum() == 0:
            deltas[name] = abs(base - 0.5)
            continue
        p = float(predict_fn(modalities["text"], modalities["audio"], modalities["visual"], m))
        deltas[name] = abs(base - p)
    total = sum(deltas.values()) or 1.0
    return {k: round(v / total, 4) for k, v in deltas.items()}


def save_gradcam_keyframes(frames: list[np.ndarray], stem: str) -> list[dict]:
    """Save sampled (face-cropped) keyframes. Full Grad-CAM if a CNN backbone is loaded."""
    if not frames:
        return []
    try:
        import cv2
        from mustard.features import crop_face
    except Exception:
        return []

    out_dir = get_paths()["keyframes"]
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    heatmaps = _try_gradcam(frames)
    for i, frame in enumerate(frames[:6]):
        vis = crop_face(frame)
        if heatmaps is not None and i < len(heatmaps):
            vis = _overlay(vis, heatmaps[i])
        name = f"{stem}_{i}.jpg"
        path = out_dir / name
        cv2.imwrite(str(path), vis)
        items.append({"timestamp": round(i * 0.5, 2), "gradcam_url": f"/static/keyframes/{name}"})
    return items


def _try_gradcam(frames: list[np.ndarray]):
    try:
        import cv2
        import torch
        from torchvision import models, transforms
        from mustard.features import crop_face

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        target_layer = model.layer4
        activations = []
        gradients = []

        def fwd_hook(_m, _i, o):
            activations.append(o)

        def bwd_hook(_m, _gi, go):
            gradients.append(go[0])

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        maps = []
        for frame in frames[:6]:
            activations.clear()
            gradients.clear()
            rgb = cv2.cvtColor(cv2.resize(crop_face(frame), (224, 224)), cv2.COLOR_BGR2RGB)
            x = tfm(rgb).unsqueeze(0)
            x.requires_grad_(True)
            logits = model(x)
            score = logits.max()
            model.zero_grad()
            score.backward()
            act = activations[0][0].detach()
            grad = gradients[0][0].detach()
            weights = grad.mean(dim=(1, 2))
            cam = torch.relu((weights[:, None, None] * act).sum(0)).cpu().numpy()
            cam = cam / (cam.max() + 1e-8)
            cam = cv2.resize(cam, (vis_w := 224, 224))
            maps.append(cam)
        h1.remove()
        h2.remove()
        return maps
    except Exception:
        return None


def _overlay(bgr: np.ndarray, cam: np.ndarray) -> np.ndarray:
    import cv2

    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.resize(bgr, (heat.shape[1], heat.shape[0]))
    return cv2.addWeighted(base, 0.55, heat, 0.45, 0)


def audio_contour(wav_path: str | Path) -> dict:
    return extract_pitch_energy_contour(wav_path)
