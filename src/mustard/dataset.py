"""Load and normalize the MUStARD++ transcript table."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import get_paths, load_config

logger = logging.getLogger(__name__)

UTTERANCE_SUFFIX = "_u"


def load_raw_csv(csv_path: Path | None = None) -> pd.DataFrame:
    cfg = load_config()
    paths = get_paths(cfg)
    csv_path = csv_path or paths["raw_csv"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing {csv_path}. Run `python data/download.py` first."
        )
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["KEY"] = df["KEY"].astype(str)
    df["SCENE"] = df["SCENE"].astype(str)
    df["SENTENCE"] = df["SENTENCE"].fillna("").astype(str)
    df["SPEAKER"] = df["SPEAKER"].fillna("UNKNOWN").astype(str).str.strip()
    df["SHOW"] = df["SHOW"].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    return df


def _is_utterance(row: pd.Series) -> bool:
    """Utterances carry the sarcasm label; a few official rows have a broken KEY."""
    if pd.notna(row.get("Sarcasm")) and str(row.get("Sarcasm")).strip() != "":
        return True
    key = str(row.get("KEY", ""))
    return key.endswith(UTTERANCE_SUFFIX)


def build_utterance_table(raw: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per annotated utterance, with concatenated dialogue context."""
    raw = raw if raw is not None else load_raw_csv()
    is_utt = raw.apply(_is_utterance, axis=1)
    utt = raw.loc[is_utt].copy()
    ctx = raw.loc[~is_utt].copy()

    ctx["_ord"] = ctx["KEY"].str.extract(r"_c_?(\d+)$", expand=False).astype(float)
    ctx = ctx.sort_values(["SCENE", "_ord", "KEY"])

    ctx_agg = (
        ctx.groupby("SCENE", as_index=False)
        .agg(
            context=("SENTENCE", lambda s: " ".join(x.strip() for x in s if str(x).strip())),
            n_context=("SENTENCE", "size"),
            context_speakers=("SPEAKER", lambda s: " | ".join(s.astype(str))),
            context_duration=("END_TIME", "last"),
        )
    )

    merged = utt.merge(ctx_agg, on="SCENE", how="left")
    merged["context"] = merged["context"].fillna("")
    merged["n_context"] = merged["n_context"].fillna(0).astype(int)
    merged["utterance"] = merged["SENTENCE"].str.replace(r"\s+", " ", regex=True).str.strip()
    merged["context"] = merged["context"].str.replace(r"\s+", " ", regex=True).str.strip()

    merged["sarcasm"] = pd.to_numeric(merged["Sarcasm"], errors="coerce")
    merged = merged.dropna(subset=["sarcasm"])
    merged["sarcasm"] = merged["sarcasm"].astype(int)
    merged["sarcasm_type"] = merged["Sarcasm_Type"].fillna("NONE").astype(str).str.strip().str.upper()
    merged["implicit_emotion"] = merged["Implicit_Emotion"].fillna("Unknown").astype(str).str.strip()
    merged["explicit_emotion"] = merged["Explicit_Emotion"].fillna("Unknown").astype(str).str.strip()
    merged["valence"] = pd.to_numeric(merged["Valence"], errors="coerce")
    merged["arousal"] = pd.to_numeric(merged["Arousal"], errors="coerce")
    merged["utt_duration_sec"] = merged["END_TIME"].map(_parse_timestamp)

    merged["utterance_id"] = merged["SCENE"]
    merged["video_key"] = merged["KEY"]
    broken = ~merged["video_key"].astype(str).str.contains(r"_u", regex=True)
    merged.loc[broken, "video_key"] = merged.loc[broken, "SCENE"].astype(str) + "_u"
    merged = merged.reset_index(drop=True)
    merged["row_id"] = merged.index
    return merged


def _parse_timestamp(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        parts = text.split(":")
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return np.nan
    return np.nan


def attach_media_paths(df: pd.DataFrame, video_dir: Path | None = None) -> pd.DataFrame:
    """Resolve utterance/context clip paths; log missing files without crashing."""
    paths = get_paths()
    video_dir = video_dir or paths["videos"]
    audio_dir = paths["audio"]
    video_dir.mkdir(parents=True, exist_ok=True)

    existing = {p.name: p for p in video_dir.rglob("*") if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}}

    def resolve(key: str, scene: str) -> str:
        candidates = [
            f"{key}.mp4",
            f"{scene}_u.mp4",
            f"{scene}.mp4",
            f"{key}.mkv",
            f"{scene}_u.mkv",
        ]
        for name in candidates:
            if name in existing:
                return str(existing[name])
        # some dumps use the raw KEY with extra suffixes
        for name, path in existing.items():
            stem = Path(name).stem
            if stem == key or stem == scene or stem.startswith(scene):
                return str(path)
        return ""

    df = df.copy()
    df["video_path"] = [resolve(k, s) for k, s in zip(df["video_key"], df["utterance_id"])]
    df["audio_path"] = df["video_key"].map(lambda k: str(audio_dir / f"{k}.wav"))
    df["has_video"] = df["video_path"].astype(bool)
    df["has_audio"] = df["audio_path"].map(lambda p: Path(p).exists())
    n_missing = int((~df["has_video"]).sum())
    logger.info(
        "Media check: %d/%d utterances have a matching video clip (%d missing).",
        int(df["has_video"].sum()),
        len(df),
        n_missing,
    )
    return df


def verify_dataset(df: pd.DataFrame) -> dict:
    cfg = load_config()
    n = len(df)
    n_sarc = int((df["sarcasm"] == 1).sum())
    n_ns = int((df["sarcasm"] == 0).sum())
    report = {
        "n_utterances": n,
        "n_sarcastic": n_sarc,
        "n_non_sarcastic": n_ns,
        "balanced": n_sarc == n_ns,
        "expected_n": cfg["dataset"]["n_expected"],
        "n_matches_paper": n == cfg["dataset"]["n_expected"],
        "n_shows": int(df["SHOW"].nunique()),
        "n_speakers": int(df["SPEAKER"].nunique()),
        "n_with_video": int(df["has_video"].sum()) if "has_video" in df.columns else 0,
        "n_with_audio": int(df["has_audio"].sum()) if "has_audio" in df.columns else 0,
        "shows": df["SHOW"].value_counts().to_dict(),
    }
    return report


def load_processed() -> pd.DataFrame:
    paths = get_paths()
    parquet = paths["processed"] / "utterances.parquet"
    csv = paths["processed"] / "utterances.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    df = attach_media_paths(build_utterance_table())
    return df
