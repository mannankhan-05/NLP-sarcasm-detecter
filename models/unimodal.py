"""Lightweight classification heads on frozen multimodal features."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.45, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_classes),
        )
        self.emotion_head = nn.Linear(hidden // 2, 16)
        self.valence_head = nn.Linear(hidden // 2, 1)
        self.arousal_head = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        z = x
        for layer in list(self.net.children())[:-1]:
            z = layer(z)
        logits = self.net[-1](z)
        if return_aux:
            return logits, self.emotion_head(z), self.valence_head(z), self.arousal_head(z)
        return logits


class GatedAttentionFusion(nn.Module):
    """Cross-modal attention + learned gates, with an explicit modality mask.

    Missing modalities are zeroed and excluded from attention via key padding.
    """

    def __init__(
        self,
        d_text: int,
        d_audio: int,
        d_visual: int,
        d_speaker: int = 0,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.4,
        n_classes: int = 2,
        n_emotions: int = 16,
    ):
        super().__init__()
        self.text_proj = nn.Sequential(nn.LayerNorm(d_text), nn.Linear(d_text, d_model), nn.GELU(), nn.Dropout(dropout))
        self.audio_proj = nn.Sequential(nn.LayerNorm(d_audio), nn.Linear(d_audio, d_model), nn.GELU(), nn.Dropout(dropout))
        self.visual_proj = nn.Sequential(nn.LayerNorm(d_visual), nn.Linear(d_visual, d_model), nn.GELU(), nn.Dropout(dropout))
        self.attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.gate = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        fused_in = d_model + d_speaker
        self.classifier = nn.Sequential(
            nn.Linear(fused_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.emotion_head = nn.Linear(d_model, n_emotions)
        self.valence_head = nn.Linear(d_model, 1)
        self.arousal_head = nn.Linear(d_model, 1)
        self.d_speaker = d_speaker

    def encode_modalities(self, text, audio, visual, mask: torch.Tensor):
        t = self.text_proj(text)
        a = self.audio_proj(audio)
        v = self.visual_proj(visual)
        feats = torch.stack([t, a, v], dim=1)  # [B, 3, D]
        feats = feats * mask.unsqueeze(-1)
        padding = ~mask.bool()
        # MultiheadAttention requires at least one valid key; keep text if everything dropped
        all_dropped = padding.all(dim=1)
        if all_dropped.any():
            padding = padding.clone()
            padding[all_dropped, 0] = False
        attn_out, attn_w = self.attn(feats, feats, feats, key_padding_mask=padding, need_weights=True)
        attn_out = self.norm(attn_out + feats)
        gates = self.gate(attn_out).squeeze(-1)
        gates = gates.masked_fill(padding, -1e9)
        gates = torch.softmax(gates, dim=-1)
        fused = (attn_out * gates.unsqueeze(-1)).sum(dim=1)
        return fused, gates, attn_w

    def forward(self, text, audio, visual, mask, speaker=None, return_details: bool = False):
        fused, gates, attn_w = self.encode_modalities(text, audio, visual, mask)
        if speaker is not None and self.d_speaker > 0:
            clf_in = torch.cat([fused, speaker], dim=-1)
        else:
            clf_in = fused
        logits = self.classifier(clf_in)
        if return_details:
            aux = {
                "gates": gates,
                "attn": attn_w,
                "emotion": self.emotion_head(fused),
                "valence": self.valence_head(fused),
                "arousal": self.arousal_head(fused),
            }
            return logits, aux
        return logits


class LateFusion(nn.Module):
    def __init__(self, d_text, d_audio, d_visual, hidden=128, dropout=0.4):
        super().__init__()
        self.text = MLPHead(d_text, hidden, dropout)
        self.audio = MLPHead(d_audio, hidden, dropout)
        self.visual = MLPHead(d_visual, hidden, dropout)
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, text, audio, visual, mask):
        lt = self.text(text)
        la = self.audio(audio)
        lv = self.visual(visual)
        logits = torch.stack([lt, la, lv], dim=1)  # [B, 3, 2]
        w = self.weights.unsqueeze(0) * mask
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        return (logits * w.unsqueeze(-1)).sum(dim=1)
