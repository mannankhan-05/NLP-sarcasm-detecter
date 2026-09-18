"""Audio/visual sarcasm scores for uploaded clips (used when fusion AV heads were not trained)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


def read_embedded_transcript(media_path: str | Path) -> str:
    """Pull a transcript from ffmpeg metadata (`comment: transcript: ...`)."""
    path = Path(media_path)
    if not path.exists():
        return ""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
        tags = (json.loads(proc.stdout or "{}").get("format") or {}).get("tags") or {}
    except Exception:
        return ""
    for raw in tags.values():
        text = str(raw).strip()
        low = text.lower()
        if low.startswith("transcript:"):
            return text.split(":", 1)[1].strip()
        if "transcript=" in low:
            return text.split("=", 1)[1].strip()
    return ""


def resolve_transcript(media_path: str | Path, provided: str | None = None) -> str:
    text = (provided or "").strip()
    if text:
        return text
    path = Path(media_path)
    embedded = read_embedded_transcript(path)
    if embedded:
        return embedded
    for sidecar in (path.with_suffix(".txt"), path.with_name(path.stem + "_transcript.txt")):
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8", errors="ignore").strip()
    return ""


def prepare_wav(media_path: str | Path, wav_path: str | Path, sr: int = 16000) -> bool:
    from mustard.features import demux_audio

    media_path, wav_path = Path(media_path), Path(wav_path)
    if media_path.suffix.lower() == ".wav":
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        if media_path.resolve() != wav_path.resolve():
            wav_path.write_bytes(media_path.read_bytes())
        return wav_path.exists()
    return demux_audio(media_path, wav_path, sr=sr)


def _load_mono_wav(path: Path, sr: int = 16000, max_seconds: float = 15.0) -> tuple[np.ndarray, int]:
    import soundfile as sf

    y, file_sr = sf.read(str(path), always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    y = np.asarray(y, dtype=np.float32)
    if file_sr != sr and y.size:
        n = max(int(round(y.size * sr / file_sr)), 1)
        y = np.interp(np.linspace(0, 1, n, endpoint=False), np.linspace(0, 1, y.size, endpoint=False), y).astype(np.float32)
    cap = int(sr * max_seconds)
    if y.size > cap:
        y = y[:cap]
    return y, sr


def _trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
    if y.size == 0:
        return y
    rms = float(np.sqrt(np.mean(y * y)) + 1e-12)
    thresh = rms * (10.0 ** (-top_db / 20.0))
    mask = np.abs(y) > thresh
    if not mask.any():
        return y
    idx = np.where(mask)[0]
    return y[idx[0] : idx[-1] + 1]


def audio_sarcasm_prob(wav_path: str | Path) -> tuple[float, dict]:
    """Map prosody to P(sarcastic). Deadpan: slower onsets, lower brightness, more RMS spread."""
    path = Path(wav_path)
    empty = {"ok": False, "onset_rate": 0.0, "rms_std": 0.0, "brightness": 0.0, "duration": 0.0}
    if not path.exists():
        return 0.5, empty
    try:
        y, sr = _load_mono_wav(path)
        y = _trim_silence(y)
        if y.size < sr * 0.25:
            return 0.5, empty
        duration = len(y) / sr
        hop, win = 256, 1024
        n_frames = max(1 + (len(y) - win) // hop, 1)
        window = np.hanning(win).astype(np.float32)
        freqs = np.fft.rfftfreq(win, 1.0 / sr)
        rms = np.empty(n_frames, dtype=np.float32)
        cent = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop
            frame = np.zeros(win, dtype=np.float32)
            take = y[start : start + win]
            frame[: take.size] = take
            rms[i] = float(np.sqrt(np.mean(frame * frame)))
            spec = np.abs(np.fft.rfft(frame * window))
            cent[i] = float((freqs * spec).sum() / (spec.sum() + 1e-9))
        diff = np.diff(rms, prepend=rms[:1])
        peak = float(diff.max()) if diff.size else 0.0
        onsets = int(np.sum((diff > 0.25 * max(peak, 1e-8)) & (rms > float(rms.mean()))))
        rate = float(onsets / max(duration, 1e-3))
        rms_std = float(rms.std())
        brightness = float(np.median(cent) / 4000.0)
        # Slow / low / dull speech leans sarcastic; brighter lively speech leans sincere.
        z = 0.45 * (3.2 - rate) + 4.0 * rms_std + 1.6 * (0.40 - brightness)
        p = float(1.0 / (1.0 + np.exp(-z)))
        p = float(np.clip(p, 0.08, 0.92))
        return p, {
            "ok": True,
            "onset_rate": rate,
            "rms_std": rms_std,
            "brightness": brightness,
            "duration": duration,
        }
    except Exception:
        return 0.5, empty


def visual_sarcasm_prob(frames: list) -> tuple[float, dict]:
    """Low motion + darker frames lean sarcastic (deadpan); bright motion leans sincere."""
    if not frames:
        return 0.5, {"ok": False, "motion": 0.0, "brightness": 0.0}
    stack = []
    for frame in frames:
        gray = frame.mean(axis=2) if getattr(frame, "ndim", 0) == 3 else frame
        stack.append(gray.astype(np.float32) / 255.0)
    arr = np.stack(stack)
    brightness = float(arr.mean())
    motion = float(np.mean(np.abs(np.diff(arr, axis=0)))) if len(arr) > 1 else 0.0
    z = 8.0 * (0.20 - brightness) + 12.0 * (0.008 - min(motion, 0.04))
    p = float(1.0 / (1.0 + np.exp(-z)))
    p = float(np.clip(p, 0.12, 0.88))
    return p, {"ok": True, "motion": motion, "brightness": brightness}


def fuse_probs(p_text: float | None, p_audio: float | None, p_visual: float | None) -> tuple[float, dict]:
    parts = []
    if p_text is not None:
        parts.append(("text", 0.62, p_text))
    if p_audio is not None:
        parts.append(("audio", 0.25 if p_text is not None else 0.6, p_audio))
    if p_visual is not None:
        parts.append(("visual", 0.13 if p_text is not None else 0.4, p_visual))
    if not parts:
        return 0.5, {"text": 0.0, "audio": 0.0, "visual": 0.0}
    wsum = sum(w for _, w, _ in parts)
    p = sum(w * val for _, w, val in parts) / wsum
    contrib = {name: 0.0 for name in ("text", "audio", "visual")}
    for name, w, val in parts:
        contrib[name] = abs(val - 0.5) * (w / wsum)
    tot = sum(contrib.values()) or 1.0
    contrib = {k: round(v / tot, 4) for k, v in contrib.items()}
    return float(np.clip(p, 0.02, 0.98)), contrib
