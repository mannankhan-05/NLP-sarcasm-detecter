"""Load the serving checkpoint and run modality-aware inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from mustard.config import get_paths, load_config
from mustard.explain import (
    gradient_token_attributions,
    heuristic_token_attributions,
    leave_one_modality_contributions,
    save_gradcam_keyframes,
)
from mustard.features import (
    FrozenTextEncoder,
    apply_standardizer,
    extract_audio_features,
    extract_pitch_energy_contour,
    extract_visual_features,
    handcrafted_text_features,
)
from mustard.io_utils import load_json
from mustard.text_utils import apply_isolated_prior, preserve_text

logger = logging.getLogger(__name__)

LABELS = {0: "non_sarcastic", 1: "sarcastic"}


class SarcasmService:
    def __init__(self) -> None:
        self.cfg = load_config()
        self.paths = get_paths(self.cfg)
        self.encoder = FrozenTextEncoder()
        self.device = "cpu"
        self.model = None
        self.ckpt = None
        self.classical = None
        self.ui_bundle = None
        self.speaker_prior = None
        self.metrics = {}
        self.trained_modalities = {"text": True, "audio": False, "visual": False}
        self.speaker_dim = 1
        self._load()

    def _load(self) -> None:
        metrics_path = self.paths["metrics"] / "metrics.json"
        if metrics_path.exists():
            self.metrics = load_json(metrics_path)

        ckpt_path = self.paths["checkpoints"] / "fusion_attn" / "serve.pt"
        try:
            import torch
            from models.unimodal import GatedAttentionFusion

            if ckpt_path.exists():
                self.ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                dims = self.ckpt["dims"]
                self.model = GatedAttentionFusion(
                    dims["text"],
                    dims["audio"],
                    dims["visual"],
                    d_speaker=dims["speaker"],
                    d_model=int(self.cfg["fusion"]["d_model"]),
                    n_heads=int(self.cfg["fusion"]["n_heads"]),
                    dropout=0.0,
                )
                self.model.load_state_dict(self.ckpt["state_dict"])
                self.model.eval()
                self.trained_modalities = self.ckpt.get("trained_modalities", self.trained_modalities)
                self.speaker_dim = dims["speaker"]
                logger.info("Loaded fusion_attn serving checkpoint (%d params).", self.ckpt.get("n_params", 0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load fusion checkpoint: %s", exc)
            self.model = None

        try:
            import joblib

            ui_path = self.paths["checkpoints"] / "text_ui" / "serve.joblib"
            if ui_path.exists():
                self.ui_bundle = joblib.load(ui_path)
                logger.info("Loaded utterance-only UI text model.")
            clf_path = self.paths["checkpoints"] / "tfidf_lr" / "serve.joblib"
            if clf_path.exists():
                self.classical = joblib.load(clf_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Text classifiers unavailable: %s", exc)

        try:
            from mustard.features import load_feature_pack

            pack = load_feature_pack()
            self.speaker_prior = pack["speaker"].mean(axis=0).astype(np.float32)
        except Exception:
            self.speaker_prior = None

    def _ui_probs(self, text: str) -> tuple[np.ndarray, str] | tuple[None, str]:
        if not self.ui_bundle:
            return None, ""
        utt = self.encoder.encode([preserve_text(text)])[0]
        hand = handcrafted_text_features(text, "")
        vec = np.concatenate([utt, hand]).astype(np.float32).reshape(1, -1)
        vec = apply_standardizer(vec, self.ui_bundle["stats"])
        p_emb = self.ui_bundle["emb"].predict_proba(vec)[0]
        p_tfidf = self.ui_bundle["tfidf"].predict_proba([text])[0]
        w_e = float(self.ui_bundle.get("blend_emb", 0.6))
        w_t = float(self.ui_bundle.get("blend_tfidf", 0.4))
        p_sarc = w_e * float(p_emb[1]) + w_t * float(p_tfidf[1])
        p_sarc, extra = apply_isolated_prior(p_sarc, text)
        return np.array([1.0 - p_sarc, p_sarc], dtype=np.float32), extra

    def _text_vector(self, utterance: str, context: str = "") -> np.ndarray:
        utt = self.encoder.encode([preserve_text(utterance)])[0]
        ctx = self.encoder.encode([preserve_text(context)])[0]
        hand = handcrafted_text_features(utterance, context)
        vec = np.concatenate([utt, ctx, hand]).astype(np.float32)
        if self.ckpt is not None:
            vec = apply_standardizer(vec.reshape(1, -1), self.ckpt["text_stats"])[0]
        return vec

    def _zeros(self, key: str) -> np.ndarray:
        if self.ckpt is None:
            cfg = self.cfg
            if key == "audio":
                return np.zeros(int(cfg["audio"]["feature_dim"]), dtype=np.float32)
            if key == "visual":
                return np.zeros(
                    int(cfg["visual"]["backbone_dim"]) + int(cfg["visual"]["handcrafted_dim"]),
                    dtype=np.float32,
                )
            return np.zeros(1, dtype=np.float32)
        return np.zeros(self.ckpt["dims"][key], dtype=np.float32)

    def _predict_fusion(self, text, audio, visual, mask) -> tuple[np.ndarray, np.ndarray]:
        import torch

        t = torch.tensor(text.reshape(1, -1))
        a = torch.tensor(audio.reshape(1, -1))
        v = torch.tensor(visual.reshape(1, -1))
        m = torch.tensor(mask.reshape(1, -1).astype(np.float32))
        spk_vec = self.speaker_prior if self.speaker_prior is not None else np.zeros(self.speaker_dim, np.float32)
        spk = torch.tensor(spk_vec.reshape(1, -1).astype(np.float32))
        with torch.no_grad():
            logits, details = self.model(t, a, v, m, spk, return_details=True)
        temperature = float(self.ckpt.get("temperature", 1.0)) if self.ckpt else 1.0
        z = logits.numpy()[0] / max(temperature, 1e-3)
        z = z - z.max()
        e = np.exp(z)
        probs = e / e.sum()
        gates = details["gates"].numpy()[0]
        return probs, gates

    def _classical_probs(self, text: str) -> np.ndarray:
        if self.classical is None:
            return np.array([0.5, 0.5], dtype=np.float32)
        from models.classical import predict_proba_safe

        return predict_proba_safe(self.classical, [text])[0]

    def predict_text(self, text: str, context: str = "") -> dict:
        t0 = time.perf_counter()
        text = preserve_text(text)
        context = preserve_text(context)
        extra_note = ""
        ui, extra_note = self._ui_probs(text)
        if ui is not None:
            probs = ui
            gates = np.array([1.0, 0.0, 0.0])
            model_name = "text-ui-utterance"
        elif self.model is not None:
            tv = self._text_vector(text, context)
            audio = self._zeros("audio")
            visual = self._zeros("visual")
            if self.ckpt is not None:
                audio = apply_standardizer(audio.reshape(1, -1), self.ckpt["audio_stats"])[0]
                visual = apply_standardizer(visual.reshape(1, -1), self.ckpt["visual_stats"])[0]
            probs, gates = self._predict_fusion(tv, audio, visual, np.array([1.0, 0.0, 0.0], np.float32))
            model_name = "fusion-v1-textmasked"
        else:
            probs = self._classical_probs(text)
            gates = np.array([1.0, 0.0, 0.0])
            model_name = "tfidf-lr-fallback"

        label_id = int(probs.argmax())
        attr = self._attributions(text)
        note = "Text only — upload a clip for full multimodal analysis."
        if extra_note:
            note = extra_note + " " + note
        elif not context:
            note += " Adding the previous dialogue turn usually improves isolated lines."
        return {
            "label": LABELS[label_id],
            "confidence": float(probs[label_id]),
            "probabilities": {"non_sarcastic": float(probs[0]), "sarcastic": float(probs[1])},
            "modalities_used": ["text"],
            "token_attributions": attr,
            "modality_contributions": {"text": 1.0, "audio": 0.0, "visual": 0.0},
            "gates": {"text": float(gates[0]), "audio": float(gates[1]), "visual": float(gates[2])},
            "model": model_name,
            "inference_ms": int((time.perf_counter() - t0) * 1000),
            "note": note,
        }

    def predict_multimodal(self, video_path: str, transcript: str | None = None, context: str = "") -> dict:
        t0 = time.perf_counter()
        from mustard.av_infer import (
            audio_sarcasm_prob,
            fuse_probs,
            prepare_wav,
            resolve_transcript,
            transcribe_speech,
            visual_sarcasm_prob,
        )
        from mustard.features import sample_frames

        video_path = Path(video_path)
        wav_path = self.paths["audio"] / f"upload_{video_path.stem}.wav"
        transcript, transcript_source = resolve_transcript(video_path, transcript)
        transcript = preserve_text(transcript)
        context = preserve_text(context)
        prepare_wav(video_path, wav_path, sr=int(self.cfg["audio"]["sample_rate"]))
        if not transcript:
            heard = preserve_text(transcribe_speech(wav_path))
            if heard:
                transcript = heard
                transcript_source = "asr"

        video_exts = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
        frames = sample_frames(video_path) if video_path.suffix.lower() in video_exts else []
        av_trained = bool(self.trained_modalities.get("audio") and self.trained_modalities.get("visual") and self.model is not None)

        if av_trained:
            return self._predict_trained_fusion(video_path, wav_path, frames, transcript, context, t0)

        p_text = None
        extra_note = ""
        if transcript:
            ui, extra_note = self._ui_probs(transcript)
            if ui is not None:
                p_text = float(ui[1])
            elif self.classical is not None:
                p_text = float(self._classical_probs(transcript)[1])

        p_audio, audio_meta = audio_sarcasm_prob(wav_path)
        if not audio_meta.get("ok"):
            p_audio = None
        p_visual, visual_meta = visual_sarcasm_prob(frames)
        if not visual_meta.get("ok"):
            p_visual = None

        p_sarc, contrib = fuse_probs(p_text, p_audio, p_visual)
        probs = np.array([1.0 - p_sarc, p_sarc], dtype=np.float32)
        used = []
        if p_text is not None:
            used.append("text")
        if p_audio is not None:
            used.append("audio")
        if p_visual is not None:
            used.append("visual")

        gates = np.array(
            [
                float(contrib.get("text", 0.0)),
                float(contrib.get("audio", 0.0)),
                float(contrib.get("visual", 0.0)),
            ],
            dtype=np.float32,
        )
        if not used:
            probs = np.array([0.5, 0.5], dtype=np.float32)
            contrib = {"text": 0.0, "audio": 0.0, "visual": 0.0}
            gates = np.zeros(3, dtype=np.float32)

        label_id = int(probs.argmax())
        keyframes = save_gradcam_keyframes(frames, video_path.stem)
        contour = {"times": [], "f0": [], "energy": []}
        attr = self._attributions(transcript) if transcript else []

        notes = []
        if extra_note:
            notes.append(extra_note)
        if transcript_source == "asr":
            notes.append("Transcript was generated from the recording. Edit it in the box and re-upload if a word is wrong.")
        elif not transcript:
            notes.append("No speech could be transcribed and no transcript was pasted, so the text channel is masked.")
        notes.append(
            "MUStARD++ videos were not downloaded, so the gated AV heads stay untrained. "
            "This clip is scored with the utterance text model plus heuristic audio (prosody) "
            "and visual (brightness/motion) cues — not a sitcom-trained trimodal fusion."
        )
        if not used:
            notes.append("No usable text, audio, or video signal was decoded, so the output is an uninformative 50/50.")

        return {
            "label": LABELS[label_id],
            "confidence": float(probs[label_id]),
            "probabilities": {"non_sarcastic": float(probs[0]), "sarcastic": float(probs[1])},
            "modalities_used": used,
            "modality_contributions": contrib,
            "gates": {"text": float(gates[0]), "audio": float(gates[1]), "visual": float(gates[2])},
            "transcript": transcript,
            "transcript_source": transcript_source,
            "token_attributions": attr,
            "keyframes": keyframes,
            "audio_contour": contour,
            "heuristic_cues": {
                "audio": audio_meta,
                "visual": visual_meta,
                "disclaimer": "Exploratory AV descriptors fused with the text model; not MUStARD++-trained AV heads.",
            },
            "model": "clip-heuristic-av+text",
            "inference_ms": int((time.perf_counter() - t0) * 1000),
            "note": " ".join(notes),
        }

    def _predict_trained_fusion(self, video_path, wav_path, frames, transcript, context, t0) -> dict:
        used = []
        mask = np.zeros(3, dtype=np.float32)
        if transcript:
            mask[0] = 1.0
            used.append("text")
            tv = self._text_vector(transcript, context)
        else:
            tv = self._zeros("text") if self.ckpt is None else np.zeros(self.ckpt["dims"]["text"], np.float32)
            if self.ckpt is not None:
                tv = apply_standardizer(tv.reshape(1, -1), self.ckpt["text_stats"])[0]

        audio = extract_audio_features(wav_path)
        visual_b, visual_h, vis_frames = extract_visual_features(video_path)
        frames = vis_frames or frames
        visual = np.concatenate([visual_b, visual_h]).astype(np.float32)

        if np.abs(audio).sum() > 0:
            mask[1] = 1.0
            used.append("audio")
        if np.abs(visual).sum() > 0:
            mask[2] = 1.0
            used.append("visual")

        if self.ckpt is not None:
            audio_n = apply_standardizer(audio.reshape(1, -1), self.ckpt["audio_stats"])[0]
            visual_n = apply_standardizer(visual.reshape(1, -1), self.ckpt["visual_stats"])[0]
        else:
            audio_n, visual_n = audio, visual

        probs, gates = self._predict_fusion(tv, audio_n, visual_n, mask)

        def _pfn(t, a, v, m):
            p, _ = self._predict_fusion(t, a, v, m)
            return p[1]

        contrib = leave_one_modality_contributions(
            _pfn,
            {"text": tv, "audio": audio_n, "visual": visual_n},
            mask,
        )
        label_id = int(probs.argmax())
        return {
            "label": LABELS[label_id],
            "confidence": float(probs[label_id]),
            "probabilities": {"non_sarcastic": float(probs[0]), "sarcastic": float(probs[1])},
            "modalities_used": used or (["text"] if transcript else []),
            "modality_contributions": contrib,
            "gates": {"text": float(gates[0]), "audio": float(gates[1]), "visual": float(gates[2])},
            "transcript": transcript,
            "token_attributions": self._attributions(transcript) if transcript else [],
            "keyframes": save_gradcam_keyframes(frames, video_path.stem),
            "audio_contour": extract_pitch_energy_contour(wav_path),
            "heuristic_cues": _heuristic_av_cues(audio, visual_h, transcript),
            "model": "fusion-v1",
            "inference_ms": int((time.perf_counter() - t0) * 1000),
            "note": "Full multimodal path active (trained audio/visual heads).",
        }

    def _attributions(self, text: str) -> list[dict]:
        if not text:
            return []
        weight = None
        if self.ckpt is not None:
            w = self.ckpt["state_dict"].get("text_proj.1.weight")
            if w is not None:
                weight = w.detach().cpu().numpy()[:, : self.encoder.dim].mean(axis=0)
        if weight is not None and self.encoder.available:
            try:
                return gradient_token_attributions(text, weight, self.encoder)
            except Exception:
                return heuristic_token_attributions(text)
        return heuristic_token_attributions(text)


def _heuristic_av_cues(audio: np.ndarray, visual_h: np.ndarray, transcript: str) -> dict:
    """Unsupervised exploratory cues — never mixed into the official verdict unless AV heads were trained."""
    pitch_var = float(audio[27]) if audio.size > 27 else 0.0
    energy_std = float(audio[30]) if audio.size > 30 else 0.0
    motion = float(visual_h[7]) if visual_h.size > 7 else 0.0
    return {
        "pitch_variance": pitch_var,
        "energy_std": energy_std,
        "visual_motion": motion,
        "disclaimer": "Exploratory descriptors, not part of the calibrated class probability unless AV training data was present.",
    }


_SERVICE: SarcasmService | None = None


def get_service() -> SarcasmService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = SarcasmService()
    return _SERVICE
