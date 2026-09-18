"""Audio/visual sarcasm scores for uploaded clips (used when fusion AV heads were not trained)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ASR_PIPE = None
_ASR_FAILED = False
_VOSK_MODEL = None


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


def resolve_transcript(media_path: str | Path, provided: str | None = None) -> tuple[str, str]:
    """Return (transcript, source) where source is pasted | metadata | sidecar | asr | none."""
    text = (provided or "").strip()
    if text:
        return text, "pasted"
    path = Path(media_path)
    embedded = read_embedded_transcript(path)
    if embedded:
        return embedded, "metadata"
    for sidecar in (path.with_suffix(".txt"), path.with_name(path.stem + "_transcript.txt")):
        if sidecar.exists():
            raw = sidecar.read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                return raw, "sidecar"
    return "", "none"


def _asr_pipeline():
    """Lazy CPU Whisper tiny.en if already cached; otherwise skip."""
    global _ASR_PIPE, _ASR_FAILED
    if _ASR_PIPE is not None or _ASR_FAILED:
        return _ASR_PIPE
    try:
        from huggingface_hub import snapshot_download
        from transformers import pipeline

        from mustard.config import get_paths

        cache = str(get_paths()["hf_cache"])
        os.environ.setdefault("HF_HOME", cache)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache)
        local = Path(cache) / "models--openai--whisper-tiny.en" / "snapshots"
        if not local.exists() or not any(local.glob("*/model.safetensors")):
            _ASR_FAILED = True
            return None
        _ASR_PIPE = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny.en",
            device="cpu",
            model_kwargs={"cache_dir": cache, "local_files_only": True},
        )
        logger.info("Loaded Whisper tiny.en for clip transcription.")
    except Exception as exc:  # noqa: BLE001
        _ASR_FAILED = True
        logger.warning("Whisper unavailable: %s", exc)
        _ASR_PIPE = None
    return _ASR_PIPE


def _transcribe_whisper(wav_path: Path) -> str:
    pipe = _asr_pipeline()
    if pipe is None:
        return ""
    out = pipe(str(wav_path))
    text = (out.get("text") if isinstance(out, dict) else str(out or "")).strip()
    return " ".join(text.split())


def _transcribe_vosk(wav_path: Path) -> str:
    global _VOSK_MODEL
    from mustard.config import PROJECT_ROOT

    model_dir = PROJECT_ROOT / "artifacts" / "asr" / "vosk-model-small-en-us-0.15"
    if not model_dir.exists():
        return ""
    import json as _json
    import wave

    from vosk import KaldiRecognizer, Model, SetLogLevel

    SetLogLevel(-1)
    if _VOSK_MODEL is None:
        _VOSK_MODEL = Model(str(model_dir))
    wf = wave.open(str(wav_path), "rb")
    rec = KaldiRecognizer(_VOSK_MODEL, wf.getframerate())
    rec.SetWords(False)
    chunks = []
    while True:
        data = wf.readframes(4000)
        if not data:
            break
        if rec.AcceptWaveform(data):
            chunks.append(_json.loads(rec.Result()).get("text", ""))
    chunks.append(_json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(t for t in chunks if t).strip()


def _transcribe_google(wav_path: Path) -> str:
    """Short-clip ASR via the public Chromium speech endpoint (same as SpeechRecognition)."""
    import urllib.parse
    import urllib.request

    flac = wav_path.with_name(wav_path.stem + ".asr.flac")
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path), "-ac", "1", "-ar", "16000", "-c:a", "flac", str(flac)],
        capture_output=True,
    )
    if proc.returncode != 0 or not flac.exists():
        return ""
    try:
        payload = flac.read_bytes()
        if len(payload) < 200:
            return ""
        qs = urllib.parse.urlencode(
            {
                "client": "chromium",
                "lang": "en-US",
                "pFilter": "0",
                # Public Chromium / SpeechRecognition demo key (not a project secret).
                "key": "AIzaSyBOti4mM-6x9WDnZIjIeyEUHhQ-sD-6g8",
            }
        )
        req = urllib.request.Request(
            f"https://www.google.com/speech-api/v2/recognize?{qs}",
            data=payload,
            headers={"Content-Type": "audio/x-flac; rate=16000"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=25) as resp:
            body = resp.read().decode("utf-8", "ignore")
    finally:
        flac.unlink(missing_ok=True)
    best = ""
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        alts = ((payload.get("result") or [{}])[0].get("alternative") or [])
        if alts:
            cand = (alts[0].get("transcript") or "").strip()
            if cand:
                best = cand
    return best


def transcribe_speech(wav_path: str | Path) -> str:
    path = Path(wav_path)
    if not path.exists() or path.stat().st_size < 400:
        return ""
    for name, fn in (
        ("vosk", _transcribe_vosk),
        ("google", _transcribe_google),
        ("whisper", _transcribe_whisper),
    ):
        try:
            text = " ".join((fn(path) or "").split())
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s ASR failed on %s: %s", name, path.name, exc)
            continue
        if len(text) >= 2:
            logger.info("Transcribed %s via %s: %s", path.name, name, text[:80])
            return text
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
