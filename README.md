# MUSTARD — Multimodal Sarcasm Detection

End-to-end NLP system that classifies an utterance as **sarcastic** or **non-sarcastic** from **text, audio, and video**, with a calibrated confidence score and a modern web UI.

Primary corpus: **[MUStARD++](https://github.com/cfiltnlp/MUStARD_Plus_Plus)** (Bedi et al., LREC 2022) — 1,202 aligned sitcom utterances, 601 / 601.

```
Given (text T, audio A, visual V, dialogue context C, modality mask M)
    →  y ∈ {0 = non-sarcastic, 1 = sarcastic}
```

The fusion model is trained with **modality dropout**. Typing a sentence in the UI is an honest **text-only** inference path (audio and vision are masked). Uploading a clip activates every trained modality.

Realistic accuracy on this corpus is **65–80%**. If you see 95%, something leaked.

---

## Setup

Python 3.10+ (developed on 3.12), `ffmpeg` on `PATH`, ~2 GB disk for CPU PyTorch + DistilRoBERTa.

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU PyTorch (do this first — the default CUDA wheel is huge)
pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

pip install -r requirements.txt --no-cache-dir
```

GPU is optional. Text-only demo inference is designed to run on CPU.

Frontend (after the Python stack):

```bash
cd app/frontend
npm install
npm run build          # production bundle, served by FastAPI
# or: npm run dev      # Vite on :5173, proxies API to :8000
```

---

## Dataset

### Transcripts (required, automatic)

```bash
python data/download.py
```

This fetches [`mustard++_text.csv`](https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/main/mustard++_text.csv), verifies **1,202** utterances and the **601 / 601** split, and writes `data/processed/utterances.csv`.

### Videos (optional, large)

Raw utterance + context clips live on Google Drive (linked from the [MUStARD++ README](https://github.com/cfiltnlp/MUStARD_Plus_Plus)):

https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW

Place `.mp4` files in `data/videos/`, then:

```bash
python data/download.py --videos
python preprocess.py --demux    # ffmpeg: 16 kHz mono wav
```

There is **no separate audio archive** — audio is demuxed from the clips.

If clips are missing, training still runs. Audio/visual channels are logged as absent, the fusion head is trained with those modalities masked, and the UI will say so. That is intentional, not a silent fallback to a “text model labeled multimodal.”

Original **MUStARD** (Castro et al., ACL 2019) is the comparison corpus (690 clips, official 414 / 138 / 138 split, pre-extracted visual features). Prefer MUStARD++ as primary.

---

## Pipeline

```bash
python data/download.py
python preprocess.py
python features.py          # caches .npy under artifacts/features/
python train.py             # 5-fold CV + speaker-independent split
python evaluate.py          # metrics, CMs, ablation, calibration, ECE
python explain.py           # example token attributions
```

Never re-extract features unless you pass `--force` to `features.py`. Standardization is fit on the **training fold only**.

Serve:

```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 (after `npm run build`) or http://127.0.0.1:5173 during `npm run dev`.

---

## API

`POST /predict/text`

```json
{ "text": "oh great, another meeting that could have been an email", "context": "" }
```

`POST /predict/multimodal` — `multipart/form-data` with `video` plus optional `transcript`, `context`.

`GET /model/metrics` — accuracy, macro-F1, precision, recall, confusion matrix, ablation table, ECE.

`GET /examples` — one-click demo utterances.

---

## Modeling (frozen backbones)

| Model | What it is |
|---|---|
| `tfidf_lr` / `tfidf_svm` | Classical floor: aggressively cleaned TF-IDF |
| `text_mlp` / `audio_mlp` / `visual_mlp` | Unimodal heads on frozen features |
| `early_fusion` | Concatenate → MLP |
| `late_fusion` | Per-modality heads + learned mix |
| **`fusion_attn`** | Cross-modal attention + gates + modality mask (**main**) |

DistilRoBERTa, optional ResNet-18, and all video/audio extractors stay **frozen**. Only the fusion / MLP heads are trained (a few hundred thousand parameters). Seeds are pinned (`config.yaml` → `seed: 42`). Early stopping is on **validation macro-F1**. Test folds are never used for tuning. Temperature scaling is fit on validation logits.

---

## Project layout

```
data/download.py          dataset fetch + integrity
preprocess.py             text / audio / video + masks
features.py               cached per-modality vectors
models/                   classical, unimodal, fusion
train.py / evaluate.py / explain.py
app/backend               FastAPI
app/frontend              React + Tailwind
notebooks/eda.ipynb
REPORT.md                 nine-phase write-up
```

---

## Ethics snapshot

Acted US sitcom sarcasm ≠ spontaneous conversation. English-only. Narrow celebrity demographics. Laugh tracks can leak the label through audio. Isolated utterances are often genuinely ambiguous. Text-only UI inference is weaker than a true trimodal clip — the UI states which modalities were used.

Full discussion: [`REPORT.md`](REPORT.md).
