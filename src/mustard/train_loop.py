from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from torch.utils.data import DataLoader, TensorDataset

from mustard.seed import seed_everything


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(axis=1)
    p_pos = probs[:, 1]
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, p_pos))
    except ValueError:
        out["roc_auc"] = float("nan")
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    try:
        out["brier"] = float(brier_score_loss(y_true, p_pos))
    except ValueError:
        out["brier"] = float("nan")
    return out


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict:
    p_pos = probs[:, 1]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    reliability = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (p_pos >= lo) & (p_pos <= hi)
        else:
            mask = (p_pos >= lo) & (p_pos < hi)
        if mask.sum() == 0:
            reliability.append({"bin": i, "conf": float((lo + hi) / 2), "acc": None, "count": 0})
            continue
        acc = float(y_true[mask].mean())
        conf = float(p_pos[mask].mean())
        ece += (mask.mean()) * abs(acc - conf)
        reliability.append({"bin": i, "conf": conf, "acc": acc, "count": int(mask.sum())})
    return {"ece": float(ece), "reliability": reliability}


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 80) -> float:
    """Temperature scaling on a held-out validation set (never the test fold)."""
    z = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    log_t = nn.Parameter(torch.zeros(1))
    opt = torch.optim.LBFGS([log_t], lr=0.25, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        T = log_t.exp().clamp(min=1e-3, max=50.0)
        loss = F.cross_entropy(z / T, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(min=1e-3, max=50.0).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


@dataclass
class TrainResult:
    state_dict: dict
    best_val_f1: float
    history: list
    val_logits: np.ndarray
    val_y: np.ndarray
    temperature: float


def _n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_torch_classifier(
    model: nn.Module,
    train_tensors: dict[str, torch.Tensor],
    val_tensors: dict[str, torch.Tensor],
    epochs: int = 40,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 32,
    patience: int = 7,
    modality_dropout: float = 0.0,
    seed: int = 42,
    device: str = "cpu",
    multitask: bool = False,
    aux: dict | None = None,
) -> TrainResult:
    seed_everything(seed)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    keys = [k for k in train_tensors if k != "y"]
    train_ds = TensorDataset(*[train_tensors[k] for k in keys], train_tensors["y"])
    val_ds = TensorDataset(*[val_tensors[k] for k in keys], val_tensors["y"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    best_f1 = -1.0
    best_state = None
    stale = 0
    history = []
    best_val_logits = None
    best_val_y = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for batch in train_loader:
            *inputs, y = batch
            feed = {k: v.to(device) for k, v in zip(keys, inputs)}
            y = y.to(device)
            if "mask" in feed and modality_dropout > 0:
                feed["mask"] = _drop_modalities(feed["mask"], modality_dropout)
            opt.zero_grad()
            logits = _forward(model, feed)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.item()) * len(y)
            n_seen += len(y)

        val_logits, val_y = _collect_logits(model, val_loader, keys, device)
        val_probs = apply_temperature(val_logits, 1.0)
        m = metrics_from_probs(val_y, val_probs)
        history.append({"epoch": epoch, "train_loss": total_loss / max(n_seen, 1), **m})
        if m["macro_f1"] > best_f1 + 1e-4:
            best_f1 = m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_logits = val_logits
            best_val_y = val_y
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    temperature = fit_temperature(best_val_logits, best_val_y)
    return TrainResult(
        state_dict=best_state,
        best_val_f1=best_f1,
        history=history,
        val_logits=best_val_logits,
        val_y=best_val_y,
        temperature=temperature,
    )


def _drop_modalities(mask: torch.Tensor, p: float) -> torch.Tensor:
    """Randomly zero audio and/or visual (never drop every remaining modality)."""
    dropped = mask.clone()
    if mask.size(1) < 3:
        return dropped
    bern_a = (torch.rand(mask.size(0), device=mask.device) > p).float()
    bern_v = (torch.rand(mask.size(0), device=mask.device) > p).float()
    dropped[:, 1] = dropped[:, 1] * bern_a
    dropped[:, 2] = dropped[:, 2] * bern_v
    # keep text so the sample remains valid
    dropped[:, 0] = torch.clamp(dropped[:, 0], min=1.0)
    return dropped


def _forward(model: nn.Module, feed: dict) -> torch.Tensor:
    name = type(model).__name__
    if name == "GatedAttentionFusion" or hasattr(model, "encode_modalities"):
        return model(
            feed["text"],
            feed["audio"],
            feed["visual"],
            feed["mask"],
            feed.get("speaker"),
        )
    if name == "LateFusion":
        return model(feed["text"], feed["audio"], feed["visual"], feed["mask"])
    return model(feed["x"])


@torch.no_grad()
def _collect_logits(model, loader, keys, device):
    model.eval()
    logits_all = []
    y_all = []
    for batch in loader:
        *inputs, y = batch
        feed = {k: v.to(device) for k, v in zip(keys, inputs)}
        logits = _forward(model, feed)
        logits_all.append(logits.cpu().numpy())
        y_all.append(y.numpy())
    return np.concatenate(logits_all), np.concatenate(y_all)


@torch.no_grad()
def predict_logits(model, tensors: dict[str, torch.Tensor], device="cpu") -> np.ndarray:
    model = model.to(device)
    model.eval()
    keys = [k for k in tensors if k != "y"]
    ds = TensorDataset(*[tensors[k] for k in keys])
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    out = []
    for batch in loader:
        feed = {k: v.to(device) for k, v in zip(keys, batch)}
        out.append(_forward(model, feed).cpu().numpy())
    return np.concatenate(out)
