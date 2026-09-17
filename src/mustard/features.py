"""Per-modality feature extraction with disk caching."""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .config import PROJECT_ROOT, get_paths, load_config
from .io_utils import load_json, save_json, save_npy
from .text_utils import HYPERBOLE, INTENSIFIERS, preserve_text, tokenize_words

logger = logging.getLogger(__name__)

_VADER = SentimentIntensityAnalyzer()

HANDCRAFTED_NAMES = [
    "n_chars",
    "n_tokens",
    "n_exclaim",
    "n_question",
    "n_quote",
    "n_ellipsis",
    "n_dots",
    "caps_ratio",
    "n_intensifiers",
    "n_hyperbole",
    "vader_compound",
    "vader_pos",
    "vader_neg",
    "ctx_vader_compound",
    "sentiment_incongruity",
    "n_context_tokens",
    "utt_ctx_len_ratio",
    "starts_with_yes_yeah_oh",
]


def handcrafted_text_features(utterance: str, context: str = "") -> np.ndarray:
    utt = preserve_text(utterance)
    ctx = preserve_text(context)
    tokens = tokenize_words(utt)
    words = [t.lower() for t in tokens if re.match(r"[A-Za-z]", t)]
    n_chars = max(len(utt), 1)
    letters = [c for c in utt if c.isalpha()]
    caps_ratio = (sum(c.isupper() for c in letters) / max(len(letters), 1)) if letters else 0.0
    vader_u = _VADER.polarity_scores(utt)
    vader_c = _VADER.polarity_scores(ctx) if ctx else {"compound": 0.0}
    low = utt.lower().strip()
    starts = 1.0 if re.match(r"^(yeah|yes|oh|sure|great|wow|right|okay|ok)\b", low) else 0.0
    n_tok = max(len(words), 1)
    n_ctx = len(ctx.split())
    vec = np.array(
        [
            float(n_chars),
            float(len(tokens)),
            float(utt.count("!")),
            float(utt.count("?")),
            float(utt.count('"') + utt.count("'")),
            float(utt.count("...")),
            float(utt.count(".")),
            caps_ratio,
            float(sum(w in INTENSIFIERS for w in words)),
            float(sum(w in HYPERBOLE for w in words)),
            vader_u["compound"],
            vader_u["pos"],
            vader_u["neg"],
            vader_c["compound"],
            abs(vader_u["compound"] - vader_c["compound"]),
            float(n_ctx),
            float(len(words) / max(n_ctx, 1)),
            starts,
        ],
        dtype=np.float32,
    )
    return vec


def extract_all_handcrafted(df: pd.DataFrame) -> np.ndarray:
    rows = [handcrafted_text_features(u, c) for u, c in zip(df["utterance"], df["context"])]
    return np.stack(rows, axis=0)


class FrozenTextEncoder:
    """Frozen DistilRoBERTa/BERT encoder used as a feature extractor only."""

    def __init__(self, model_name: str | None = None, max_length: int = 128):
        cfg = load_config()
        self.model_name = model_name or cfg["text"]["model_name"]
        self.max_length = max_length
        self.dim = int(cfg["text"]["embedding_dim"])
        self._ok = False
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._init()

    def _init(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            cache = str(get_paths()["hf_cache"])
            os.environ.setdefault("HF_HOME", cache)
            os.environ.setdefault("TRANSFORMERS_CACHE", cache)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache)
            self.model = AutoModel.from_pretrained(self.model_name, cache_dir=cache)
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            hidden = int(self.model.config.hidden_size)
            self.dim = hidden
            self._ok = True
            logger.info("Loaded frozen encoder %s on %s (dim=%d).", self.model_name, self.device, self.dim)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Transformer encoder unavailable (%s). Using hashing fallback.", exc)
            self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        if not self._ok:
            return _hashing_encode(texts, self.dim)
        import torch

        out = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = [preserve_text(t) or "[PAD]" for t in texts[i : i + batch_size]]
                enc = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                hidden = self.model(**enc).last_hidden_state
                cls = hidden[:, 0, :].detach().cpu().numpy().astype(np.float32)
                out.append(cls)
        return np.concatenate(out, axis=0)

    def encode_tokens(self, text: str):
        """Return tokens + last hidden states for attribution."""
        if not self._ok:
            return [], None, None
        import torch

        enc = self.tokenizer(
            preserve_text(text) or "[PAD]",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = self.tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
        enc = {k: v.to(self.device) for k, v in enc.items()}
        self.model.eval()
        outputs = self.model(**enc, output_hidden_states=True)
        return tokens, outputs.last_hidden_state, enc


def _hashing_encode(texts: list[str], dim: int) -> np.ndarray:
    """Deterministic hashed character-ngram bag — last-resort encoder."""
    from sklearn.feature_extraction.text import HashingVectorizer

    vec = HashingVectorizer(n_features=dim, alternate_sign=False, ngram_range=(2, 5), analyzer="char")
    x = vec.transform([preserve_text(t) for t in texts]).astype(np.float32).toarray()
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return (x / norms).astype(np.float32)


def extract_audio_features(wav_path: str | Path, sr: int = 16000) -> np.ndarray:
    """MFCC stats + pitch variance + energy contour. Zeros if the file is missing/corrupt."""
    cfg = load_config()
    dim = int(cfg["audio"]["feature_dim"])
    path = Path(wav_path)
    if not path.exists():
        return np.zeros(dim, dtype=np.float32)
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=sr, mono=True)
        y, _ = librosa.effects.trim(y, top_db=int(cfg["audio"]["top_db"]))
        if y.size == 0:
            return np.zeros(dim, dtype=np.float32)
        peak = np.max(np.abs(y)) + 1e-8
        y = y / peak
        n_mfcc = int(cfg["audio"]["n_mfcc"])
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=int(cfg["audio"]["n_fft"]), hop_length=int(cfg["audio"]["hop_length"]))
        delta = librosa.feature.delta(mfcc)
        rms = librosa.feature.rms(y=y)[0]
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        try:
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
            f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
        except Exception:  # noqa: BLE001
            f0 = np.array([])
        duration = len(y) / sr
        speaking_rate = (librosa.onset.onset_detect(y=y, sr=sr).size / duration) if duration > 0 else 0.0
        feats = np.concatenate(
            [
                mfcc.mean(axis=1),
                mfcc.std(axis=1),
                delta.mean(axis=1),
                np.array(
                    [
                        float(f0.mean()) if f0.size else 0.0,
                        float(f0.std()) if f0.size else 0.0,
                        float(rms.mean()),
                        float(rms.std()),
                        float(zcr.mean()),
                        float(speaking_rate),
                        float(duration),
                        float(np.percentile(rms, 90) - np.percentile(rms, 10)),
                    ],
                    dtype=np.float32,
                ),
            ]
        )
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, feats.size)
        out[:n] = feats[:n].astype(np.float32)
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Audio feature extraction failed for %s: %s", path, exc)
        return np.zeros(dim, dtype=np.float32)


def extract_pitch_energy_contour(wav_path: str | Path, sr: int = 16000) -> dict:
    path = Path(wav_path)
    empty = {"times": [], "f0": [], "energy": []}
    if not path.exists():
        return empty
    try:
        import librosa

        y, sr = librosa.load(str(path), sr=sr, mono=True)
        hop = 256
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
        try:
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr, hop_length=hop)
            f0 = np.nan_to_num(f0, nan=0.0)
        except Exception:  # noqa: BLE001
            f0 = np.zeros_like(rms)
        n = min(len(times), len(rms), len(f0), 200)
        return {
            "times": times[:n].tolist(),
            "f0": f0[:n].tolist(),
            "energy": rms[:n].tolist(),
        }
    except Exception:  # noqa: BLE001
        return empty


def demux_audio(video_path: str | Path, wav_path: str | Path, sr: int = 16000) -> bool:
    video_path, wav_path = Path(video_path), Path(wav_path)
    if not video_path.exists():
        return False
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", str(sr), "-acodec", "pcm_s16le",
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path.exists()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffmpeg demux failed for %s: %s", video_path, exc)
        return False


def extract_visual_features(video_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return (backbone_or_zeros, handcrafted, sampled_bgr_frames)."""
    cfg = load_config()
    dim_b = int(cfg["visual"]["backbone_dim"])
    dim_h = int(cfg["visual"]["handcrafted_dim"])
    frames = sample_frames(video_path, fps=float(cfg["visual"]["target_fps"]), max_frames=int(cfg["visual"]["max_frames"]))
    hand = _handcrafted_visual(frames, dim_h)
    backbone = _backbone_visual(frames, dim_b)
    return backbone, hand, frames


def sample_frames(video_path: str | Path, fps: float = 2.0, max_frames: int = 8) -> list[np.ndarray]:
    path = Path(video_path) if video_path else Path()
    if not path.exists():
        return []
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(int(round(native_fps / max(fps, 0.1))), 1)
        frames = []
        idx = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
        return frames
    except Exception as exc:  # noqa: BLE001
        logger.warning("Frame sampling failed for %s: %s", path, exc)
        return []


def crop_face(frame: np.ndarray) -> np.ndarray:
    try:
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        det = cv2.CascadeClassifier(cascade_path)
        faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        if len(faces) == 0:
            return frame
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.15 * max(w, h))
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
        return frame[y0:y1, x0:x1]
    except Exception:  # noqa: BLE001
        return frame


def _handcrafted_visual(frames: list[np.ndarray], dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if not frames:
        return out
    try:
        import cv2

        faces = [crop_face(f) for f in frames]
        resized = [cv2.resize(f, (64, 64)) for f in faces]
        stack = np.stack(resized).astype(np.float32) / 255.0
        means = stack.mean(axis=(0, 1, 2))
        stds = stack.std(axis=(0, 1, 2))
        gray = stack.mean(axis=-1)
        contrast = gray.std()
        motion = 0.0
        if len(gray) > 1:
            motion = np.mean(np.abs(np.diff(gray, axis=0)))
        brightness = gray.mean()
        face_fracs = []
        for frame, face in zip(frames, faces):
            face_fracs.append((face.shape[0] * face.shape[1]) / max(frame.shape[0] * frame.shape[1], 1))
        vec = np.array(
            [
                *means,
                *stds,
                contrast,
                motion,
                brightness,
                float(np.mean(face_fracs)),
                float(len(frames)),
            ],
            dtype=np.float32,
        )
        n = min(dim, vec.size)
        out[:n] = vec[:n]
        return out
    except Exception:  # noqa: BLE001
        return out


def _backbone_visual(frames: list[np.ndarray], dim: int) -> np.ndarray:
    """Optional frozen ResNet18; zeros if torchvision/weights are unavailable."""
    if not frames:
        return np.zeros(dim, dtype=np.float32)
    try:
        import torch
        import torch.nn.functional as F
        from torchvision import models, transforms
        import cv2

        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = torch.nn.Identity()
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        tfm = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        tensors = []
        for frame in frames:
            face = crop_face(frame)
            rgb = cv2.cvtColor(cv2.resize(face, (224, 224)), cv2.COLOR_BGR2RGB)
            tensors.append(tfm(rgb))
        batch = torch.stack(tensors)
        with torch.no_grad():
            feats = model(batch).mean(dim=0).cpu().numpy().astype(np.float32)
        out = np.zeros(dim, dtype=np.float32)
        n = min(dim, feats.size)
        out[:n] = feats[:n]
        return out
    except Exception:
        return np.zeros(dim, dtype=np.float32)


def build_speaker_onehot(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    speakers = sorted(df["SPEAKER"].astype(str).unique().tolist())
    index = {s: i for i, s in enumerate(speakers)}
    mat = np.zeros((len(df), len(speakers)), dtype=np.float32)
    for i, spk in enumerate(df["SPEAKER"].astype(str)):
        mat[i, index[spk]] = 1.0
    return mat, speakers


def fit_standardizer(x: np.ndarray) -> dict:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def apply_standardizer(x: np.ndarray, stats: dict) -> np.ndarray:
    return (x - stats["mean"]) / stats["std"]


def cache_all_features(df: pd.DataFrame, force: bool = False) -> dict[str, np.ndarray]:
    """Extract (or load) every modality cache. Never re-extract unless force=True."""
    paths = get_paths()
    feat_dir = paths["features"]
    meta_path = feat_dir / "meta.json"
    keys = {
        "text_utt": feat_dir / "text_utt.npy",
        "text_ctx": feat_dir / "text_ctx.npy",
        "text_hand": feat_dir / "text_hand.npy",
        "audio": feat_dir / "audio.npy",
        "visual": feat_dir / "visual.npy",
        "visual_hand": feat_dir / "visual_hand.npy",
        "speaker": feat_dir / "speaker.npy",
        "mask": feat_dir / "mask.npy",
        "y": feat_dir / "y.npy",
    }
    if not force and all(p.exists() for p in keys.values()) and meta_path.exists():
        logger.info("Loading cached features from %s", feat_dir)
        return {k: np.load(p) for k, p in keys.items()} | {"meta": load_json(meta_path)}

    encoder = FrozenTextEncoder()
    logger.info("Encoding utterance text (%d rows)...", len(df))
    text_utt = encoder.encode(df["utterance"].tolist())
    logger.info("Encoding context text...")
    text_ctx = encoder.encode(df["context"].tolist())
    text_hand = extract_all_handcrafted(df)
    speaker, speaker_names = build_speaker_onehot(df)

    cfg = load_config()
    audio = np.zeros((len(df), int(cfg["audio"]["feature_dim"])), dtype=np.float32)
    visual = np.zeros((len(df), int(cfg["visual"]["backbone_dim"])), dtype=np.float32)
    visual_hand = np.zeros((len(df), int(cfg["visual"]["handcrafted_dim"])), dtype=np.float32)
    mask = np.zeros((len(df), 3), dtype=np.float32)
    mask[:, 0] = 1.0  # text always present in this corpus

    for i, row in df.iterrows():
        if row.get("has_video") and row.get("video_path"):
            wav = Path(row["audio_path"])
            if not wav.exists():
                demux_audio(row["video_path"], wav)
            if wav.exists():
                audio[i] = extract_audio_features(wav)
                mask[i, 1] = 1.0
            v_b, v_h, _ = extract_visual_features(row["video_path"])
            visual[i] = v_b
            visual_hand[i] = v_h
            if np.abs(v_h).sum() > 0 or np.abs(v_b).sum() > 0:
                mask[i, 2] = 1.0
        if (i + 1) % 100 == 0:
            logger.info("Media features %d/%d", i + 1, len(df))

    y = df["sarcasm"].to_numpy(dtype=np.int64)
    arrays = {
        "text_utt": text_utt,
        "text_ctx": text_ctx,
        "text_hand": text_hand,
        "audio": audio,
        "visual": visual,
        "visual_hand": visual_hand,
        "speaker": speaker,
        "mask": mask,
        "y": y,
    }
    for k, arr in arrays.items():
        save_npy(arr, keys[k])

    meta = {
        "n": int(len(df)),
        "encoder": encoder.model_name if encoder.available else "hashing-fallback",
        "encoder_available": encoder.available,
        "dims": {k: list(v.shape) for k, v in arrays.items()},
        "speaker_names": speaker_names,
        "handcrafted_names": HANDCRAFTED_NAMES,
        "n_audio_present": int(mask[:, 1].sum()),
        "n_visual_present": int(mask[:, 2].sum()),
        "project_root": str(PROJECT_ROOT),
    }
    save_json(meta, meta_path)
    logger.info("Cached features: %s", meta["dims"])
    return arrays | {"meta": meta}


def load_feature_pack() -> dict:
    paths = get_paths()
    feat_dir = paths["features"]
    required = ["text_utt", "text_ctx", "text_hand", "audio", "visual", "visual_hand", "speaker", "mask", "y"]
    pack = {k: np.load(feat_dir / f"{k}.npy") for k in required}
    pack["meta"] = load_json(feat_dir / "meta.json")
    return pack
