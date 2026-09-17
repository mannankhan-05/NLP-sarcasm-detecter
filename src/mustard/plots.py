from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from mustard.config import get_paths


LABELS = ["Non-sarcastic", "Sarcastic"]


def plot_confusion_matrix(cm: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrBr",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=ax,
        cbar=False,
        linewidths=0.5,
        linecolor="#e8d9b0",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_reliability(reliability: list[dict], ece: float, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    confs, accs = [], []
    for row in reliability:
        if row["acc"] is None:
            continue
        confs.append(row["conf"])
        accs.append(row["acc"])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.plot([0, 1], [0, 1], ls="--", c="#888", label="perfect")
    if confs:
        ax.plot(confs, accs, marker="o", c="#c48a2a", label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE={ece:.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ablation(table: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(table.keys())
    vals = [table[n]["macro_f1"] for n in names]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = ["#c48a2a" if n == "text+audio+visual" else "#5c5346" for n in names]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Modality ablation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figures_dir() -> Path:
    return get_paths()["figures"]
