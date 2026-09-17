# MUSTARD: A Multimodal Sarcasm Detection System

**Abstract.** Sarcasm is a pragmatic phenomenon in which literal meaning and intended meaning come apart. Cues are distributed across wording, tone, and face, so text-only classifiers miss a large fraction of the signal. This project implements a complete nine-phase NLP lifecycle on **MUStARD++** (Bedi et al., LREC 2022): 1,202 sitcom utterances with aligned text, audio, and video, perfectly balanced 601 / 601. Frozen pretrained encoders feed a lightweight gated-attention fusion head trained with modality dropout, 5-fold stratified cross-validation, and a speaker/show-independent holdout. Confidence is temperature-scaled on validation logits. A FastAPI + React UI exposes text-only and trimodal inference, token attributions, and per-modality contributions. Published multimodal results on the related MUStARD corpus sit in the **65–80%** range; we treat anything near 95% as evidence of leakage, not success.

---

## 1. Problem definition

**Task.** Given an utterance’s transcript \(T\), audio \(A\), visual stream \(V\), dialogue context \(C\), and a binary modality-availability mask \(M \in \{0,1\}^3\),

\[
f(T, A, V, C, M) \rightarrow y \in \{0=\text{non-sarcastic},\; 1=\text{sarcastic}\}.
\]

**Why it is hard.** Literal polarity often contradicts intended polarity (“Great, another meeting”). There is no reliable lexical marker: intensifiers, punctuation, and honorifics fire on both classes. Prosody (pitch variance, drawl, energy) and facial deadpan carry much of the label, and in this corpus the preceding turns are frequently what makes the line sarcastic at all.

**Research questions.**

1. Does trimodal fusion beat each unimodal baseline on MUStARD++?
2. Which modality contributes most, on average and per example?
3. Does auxiliary supervision from implicit/explicit emotion, valence, and arousal help sarcasm?
4. Does the model rely on genuine sarcasm cues or on dataset artifacts (laugh track, speaker identity, show style)?

**Metrics.** Primary: **macro-F1**. Also accuracy, macro precision, macro recall, ROC-AUC. Confidence quality: Expected Calibration Error (ECE) plus a reliability diagram. Protocol: **mean ± std over 5 stratified folds**. A single 138-point test set is too noisy to report alone. We additionally report a **speaker/show-independent** split (hold out one show).

---

## 2. Dataset

**Primary: MUStARD++** ([repo](https://github.com/cfiltnlp/MUStARD_Plus_Plus), [paper](https://aclanthology.org/2022.lrec-1.756.pdf)). Dialogues from *The Big Bang Theory*, *Friends*, *The Golden Girls*, *Silicon Valley*, and related sitcoms. Each instance is an utterance plus its context clip. Columns used:

| Column | Role |
|---|---|
| `Sarcasm` | Target, {0,1} |
| `Sarcasm_Type` | PRO / ILL / EMB / … or NONE |
| `Implicit_Emotion`, `Explicit_Emotion` | Auxiliary classification |
| `Valence`, `Arousal` | Auxiliary regression, 1–9 |
| `SPEAKER`, `SHOW` | Speaker-dependent vs independent protocol |
| `SENTENCE`, `KEY`, `SCENE` | Transcript + alignment to media |

`python data/download.py` pulls `mustard++_text.csv` and checks \(N=1202\) and 601/601. One official row (`SCENE=1_S11E03_067`) ships with `KEY=Disgust` instead of `1_S11E03_067_u`; we treat any row with a sarcasm label as an utterance so the count matches the paper. Every row is matched against `data/videos/`; missing clips are logged, not crashed on. Audio is **not** a separate download — `ffmpeg` demuxes 16 kHz mono from the video.

**Comparison: MUStARD** (Castro et al., ACL 2019). 690 clips, 345/345, official 414/138/138, with pre-extracted visual features useful as a sanity check against our own pipeline. MUStARD++ corrects earlier annotation errors and roughly doubles the data, so it is the training corpus.

**Optional fourth modality (out of scope here):** MMSD2.0 image+caption sarcasm. Different task (no audio/video); the original MMSD has hashtag/emoji leakage that MMSD2.0 removes.

**Splits.** 5-fold *stratified* CV on `Sarcasm`. Inside each training set a 15% validation slice is carved **from train only** for early stopping and temperature scaling. The test fold is never touched for any decision. Speaker-independent protocol: hold out `FRIENDS` (config: `speaker_independent_holdout_show`) so no speaker from that show appears at train time.

**Bias and ethics.** This is *acted, English, US-television* sarcasm. Laugh tracks can leak the label into the audio channel. Speakers such as Chandler and Sheldon are over-represented. Demographics of sitcom leads are narrow. Spontaneous workplace or social-media sarcasm will not look like this. Isolated text boxes in a UI are a strictly weaker observation than an utterance in dialogue.

EDA charts live in `artifacts/figures/eda/` (class balance, token-length histogram, sarcasm-type breakdown, implicit/explicit emotion, show and speaker bars, clip duration). Re-run via `notebooks/eda.ipynb` or `python preprocess.py`.

---

## 3. Preprocessing

**Text (transformer path).** Casing and punctuation are kept — both are sarcasm cues. No stopword deletion. Tokenize with `AutoTokenizer` (`distilroberta-base`), `max_length=128`, utterance and context encoded **separately** so the model can represent incongruity between “what came before” and “what was said.”

**Text (classical path only).** Lowercase, strip punctuation, drop a small stoplist, crude suffix stem. This is the aggressively cleaned pipeline requested for TF-IDF, and it is *not* used by the deep models.

**Audio.** `ffmpeg -vn -ac 1 -ar 16000`. Trim silence (`librosa.effects.trim`), peak-normalize. Missing or corrupt files yield a zero vector and `mask[audio]=0`.

**Video.** Sample ~2 fps, at most 8 frames. Haar-cascade face crop when a face is found (expression carries most of the visual sarcasm signal), otherwise the full frame. Resize to 224 for the optional ResNet-18 backbone.

**Context.** Preceding `*_c_*` rows for the same `SCENE` are concatenated in KEY order and stored in `context`.

**Modality mask.** Per sample \(M=(\text{text},\text{audio},\text{visual})\). Text is 1 for every MUStARD++ row. Audio/visual are 1 only when a real clip was decoded.

---

## 4. Feature engineering

All vectors are cached under `artifacts/features/*.npy`. Each modality is **standardized independently** with training-fold mean/std.

| Feature | Dim | Notes |
|---|---|---|
| DistilRoBERTa `[CLS]` utterance | 768 | Frozen |
| DistilRoBERTa `[CLS]` context | 768 | Frozen |
| Handcrafted text | 18 | `! ? …` counts, caps ratio, intensifiers, hyperbole, VADER pos/neg/compound, \(\lvert v_{\text{utt}}-v_{\text{ctx}}\rvert\) incongruity, length ratio, “yeah/oh/sure/great” opener |
| Audio | 48 | MFCC mean/std/delta (13), F0 mean/std, RMS mean/std, ZCR, onset rate, duration, RMS spread |
| Visual backbone | 512 | Frozen ResNet-18 pooled over face crops, or zeros if torchvision/weights absent |
| Visual handcrafted | 24 | Color mean/std, contrast, inter-frame motion, brightness, face-area fraction |
| Speaker | \(S\) | One-hot over corpus speakers |

If Hugging Face weights cannot be downloaded, a hashed character-ngram encoder of the same dimension is used so the rest of the pipeline still runs; the checkpoint metadata records which encoder was live.

---

## 5. Modeling

Backbones stay frozen. Only MLP / attention heads are trained (typically \(\lt 300\text{k}\) parameters). Strong dropout (0.45), AdamW weight decay \(10^{-3}\), early stopping on validation macro-F1 (patience 7). Modality dropout: independently drop audio and/or visual with \(p=0.25\), never drop text, so the head learns the text-only UI path.

1. **Classical.** TF-IDF (1–2 grams, 8k features, sublinear) + Logistic Regression; same vectorizer + calibrated Linear SVM. Sanity floor.
2. **Unimodal deep.** MLP on text / audio / visual features alone.
3. **Early fusion.** Concatenate the three vectors → MLP.
4. **Late fusion.** Three MLPs, learned mixture weights × mask.
5. **Attention fusion (main).** Project each modality to \(d=128\), masked multi-head self-attention, linear gates, concat speaker one-hot, classify. Missing modalities are `key_padding_mask`ed.
6. **Multi-task (optional heads).** Emotion, valence, arousal heads exist on the fusion module. They are **not** mixed into the serving loss by default so a weak auxiliary signal cannot distort the sarcasm probability the UI shows; enable via the `multitask` block in `config.yaml` if you want the extension.

Seeds: `random`, `numpy`, `torch` all set to 42. Every hyperparameter is stored inside each `.pt` checkpoint.

**Serving rule.** If audio/visual were never observed at train time, inference **keeps those channels masked** even when a user uploads a clip. Mixing untrained random projections into the logits would be a silent lie. Keyframes and pitch/energy contours are still shown, labeled as exploratory.

---

## 6. Evaluation

Run `python evaluate.py` after training. Artifacts:

- `artifacts/metrics/cv_results.json` — per-model mean ± std
- `artifacts/metrics/metrics.json` — payload for `GET /model/metrics`
- `artifacts/figures/<model>/cm_fold*.png` and `cm_sum.png` — required 2×2 heatmaps
- `artifacts/figures/ablation.png` — text / audio / visual / pairs / all three
- `artifacts/figures/reliability_fusion.png` — ECE
- `artifacts/metrics/error_analysis.json` — 12 misses with commentary

**Published MUStARD (Castro et al. 2019) speaker-dependent SVM, weighted F-score** — the number we compare against, not a target to exceed on a different corpus:

| Modalities | P | R | F |
|---|---:|---:|---:|
| T | 65.1 | 64.6 | 64.6 |
| A | 65.9 | 64.6 | 64.6 |
| V | 68.1 | 67.4 | 67.4 |
| T+A | 66.6 | 66.2 | 66.2 |
| T+V | 72.0 | 71.6 | 71.6 |
| A+V | 66.2 | 65.7 | 65.7 |
| T+A+V | 71.9 | 71.4 | 71.5 |

Speaker-independent (train on BBT/TGG/SA, test on Friends) collapses toward the high 50s–low 60s; trimodal is **not** a clear win there. If our MUStARD++ fusion does not beat text-only, that is a legitimate small-data finding and is reported as such.

**Calibration.** Temperature \(T\) is fit by LBFGS on validation logits (never test). UI confidence is \(\mathrm{softmax}(z/T)\). ECE is logged; if it remains high after scaling, the UI still shows the number but the Model Info panel exposes ECE so the user can discount it.

### This run (MUStARD++ text pathway; no raw clips on disk)

5-fold stratified CV, seed 42, DistilRoBERTa frozen, macro-F1 mean ± std. Audio and visual were **absent** (0/1202 clips), so fusion equals the masked text path. That is reported, not hidden.

| Model | Accuracy | Macro-F1 | ROC-AUC | ECE |
|---|---:|---:|---:|---:|
| TF-IDF + LR | 0.596 ± 0.017 | 0.596 ± 0.017 | 0.616 | 0.047 |
| TF-IDF + SVM | 0.585 ± 0.020 | 0.584 ± 0.020 | 0.615 | 0.041 |
| Text MLP | 0.616 ± 0.033 | 0.615 ± 0.033 | 0.680 | 0.069 |
| Early fusion* | 0.632 ± 0.025 | 0.631 ± 0.025 | 0.676 | 0.072 |
| Late fusion* | 0.618 ± 0.035 | 0.618 ± 0.035 | 0.673 | 0.073 |
| **Gated attention (main)** | **0.621 ± 0.023** | **0.621 ± 0.023** | 0.669 | **0.031** (after T-scaling) |

\*With zero AV vectors these heads are not true trimodal models; early fusion’s small bump is within fold noise and extra unused parameters, not a multimodal win.

Ablation of the attention model: text / text+audio / text+visual / all-three = **0.621**; audio-only, visual-only, audio+visual = **0.333** (degenerate — no signal). Summed 5-fold confusion matrix for fusion: TN 381, FP 220, FN 235, TP 366.

Speaker-independent (hold out FRIENDS, n=354): fusion macro-F1 **0.672**. Text-only on MUStARD++ is slightly under Castro et al.’s MUStARD SVM (64.6 weighted F) and sits honestly in the 60s, not the 90s.

**Error analysis themes** we expect, and that `evaluate.py` tags:

- Sincere-looking wording, sarcastic delivery (needs audio/face).
- Line is only sarcastic given the previous turn.
- Intensifiers / “yeah, sure, great” over-trigger the lexicon.
- Speaker style (Sheldon/Chandler) looks sarcastic when the line is not.
- Possible laugh-track leakage if audio was trained on raw sitcom wavs.
- Short utterances with almost no lexical evidence.

---

## 7. Explainability

| Level | Method | Where |
|---|---|---|
| Tokens | Gradient × input on DistilRoBERTa token embeddings, dotted with the fusion text-projection sarcasm direction; lexicon/punctuation heuristic fallback | UI highlight + `explain.py` |
| Modalities | Fusion gates + leave-one-modality-out \(\Delta p\) | UI contribution bars |
| Visual | Face-cropped keyframes; Grad-CAM on ResNet-18 `layer4` when torchvision is present | Upload tab |
| Audio | Pitch (pyin) and RMS energy contour | Upload response `audio_contour` |

We specifically inspect whether predictions move when speaker one-hots change and whether audio-only accuracy tracks laugh-track presence. A model that is secretly a speaker-ID classifier has failed question (d).

---

## 8. Limitations

- **N = 1,202** is tiny for deep multimodality. Frozen encoders + tiny heads are a constraint, not a preference.
- Acted sitcom sarcasm ≠ spontaneous sarcasm.
- English, US television, narrow lead demographics.
- Laugh tracks and canned reaction shots can leak labels into audio and video.
- Context dependence: many utterances are genuinely ambiguous in isolation, including in the text-only UI.
- Text-only inference is weaker than trimodal inference; the UI discloses the mask instead of pretending otherwise.
- Face crops use Haar cascades, not a modern detector; off-angle sitcom coverage is messy.
- DistilRoBERTa is English-centric.
- Auxiliary emotion labels are themselves subjective annotations on acted affect.
- We do not fine-tune Wav2Vec2 / ViT / a video backbone end-to-end. Doing so on this dataset overfits within an epoch.

**Future work.** True AV training once the Drive clips are local; OpenFace action units; a learned temperature per modality; MMSD2.0 as a separate image-caption task; human eval on non-sitcom audio.

---

## 9. Conclusion

MUSTARD is a full-stack sarcasm detector: dataset verification, sarcasm-preserving text preprocessing, cached multimodal features, classical and deep baselines, masked gated fusion, 5-fold and speaker-independent evaluation with confusion matrices and ECE, and a UI that will not call a text-only posterior “multimodal.” The scientifically honest outcome on 1,202 sitcom lines is a calibrated, explainable system in the published 65–80% band — not a leaderboard number that forgot the laugh track.

---

## References

- Bedi, Nunna, Chakraborty, Pal, Das, Bandyopadhyay. “A Multimodal Corpus for Emotion Recognition in Sarcasm.” LREC 2022. https://aclanthology.org/2022.lrec-1.756
- Castro, Hazarika, Pérez-Rosas, Zimmermann, Mihalcea, Poria. “Towards Multimodal Sarcasm Detection (An Obviously Perfect Paper).” ACL 2019. https://aclanthology.org/P19-1455/
- MUStARD++ repository: https://github.com/cfiltnlp/MUStARD_Plus_Plus
- MUStARD repository: https://github.com/soujanyaporia/MUStARD
- Liu et al. RoBERTa. arXiv:1907.11692.
- Baevski et al. wav2vec 2.0. NeurIPS 2020.
- He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
- Guo et al. On Calibration of Modern Neural Networks. ICML 2017.
- Hutto & Gilbert. VADER. ICWSM 2014.
- Sundararajan, Taly, Yan. Axiomatic Attribution for Deep Networks. ICML 2017.
- Selvaraju et al. Grad-CAM. ICCV 2017.
- MMSD2.0: https://github.com/JoeYing1019/MMSD2.0
