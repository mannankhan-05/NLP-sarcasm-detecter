from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mustard.config import get_paths
from mustard.dataset import attach_media_paths, build_utterance_table, load_raw_csv, verify_dataset
from mustard.io_utils import save_json

sns.set_theme(style="whitegrid", context="paper")


def generate_eda() -> dict:
    paths = get_paths()
    fig_dir = paths["figures"] / "eda"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = attach_media_paths(build_utterance_table(load_raw_csv()))
    df["utt_tokens"] = df["utterance"].str.split().str.len()
    df["ctx_tokens"] = df["context"].str.split().str.len()

    _bar(df["sarcasm"].map({0: "Non-sarcastic", 1: "Sarcastic"}), "Class distribution", fig_dir / "class_distribution.png")
    _hist(df["utt_tokens"], "Utterance length (tokens)", fig_dir / "utterance_length.png")
    _count(df, "sarcasm_type", "Sarcasm type", fig_dir / "sarcasm_type.png")
    _count(df, "implicit_emotion", "Implicit emotion", fig_dir / "implicit_emotion.png", rotate=True)
    _count(df, "explicit_emotion", "Explicit emotion", fig_dir / "explicit_emotion.png", rotate=True)
    _count(df, "SHOW", "Show distribution", fig_dir / "shows.png")
    top_spk = df["SPEAKER"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.barplot(x=top_spk.index, y=top_spk.values, ax=ax, color="#C9961A")
    ax.set_title("Top speakers")
    ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(fig_dir / "speakers.png", dpi=140)
    plt.close(fig)

    if df["utt_duration_sec"].notna().any():
        _hist(df["utt_duration_sec"].dropna(), "Utterance duration (s)", fig_dir / "duration.png")

    report = verify_dataset(df)
    report["mean_utt_tokens"] = float(df["utt_tokens"].mean())
    report["mean_ctx_tokens"] = float(df["ctx_tokens"].mean())
    report["sarcasm_types"] = df["sarcasm_type"].value_counts().to_dict()
    report["missing_video"] = int((~df["has_video"]).sum())
    save_json(report, paths["metrics"] / "eda_summary.json")
    return report


def _bar(series, title, path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    series.value_counts().plot(kind="bar", ax=ax, color=["#5c5346", "#C9961A"])
    ax.set_title(title)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _hist(series, title, path: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sns.histplot(series, bins=30, ax=ax, color="#C9961A")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _count(df: pd.DataFrame, col: str, title: str, path: Path, rotate: bool = False):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, ax=ax, color="#C9961A")
    ax.set_title(title)
    if rotate:
        ax.tick_params(axis="x", rotation=40)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
