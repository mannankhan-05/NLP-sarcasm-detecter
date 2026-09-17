#!/usr/bin/env python3
"""Text / audio / video preprocessing + modality masks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demux", action="store_true", help="Extract wav files from any available videos.")
    args = parser.parse_args()

    from mustard.config import get_paths, load_config
    from mustard.dataset import attach_media_paths, build_utterance_table, load_raw_csv, verify_dataset
    from mustard.features import demux_audio
    from mustard.text_utils import aggressive_clean, preserve_text

    cfg = load_config()
    paths = get_paths(cfg)
    df = attach_media_paths(build_utterance_table(load_raw_csv()))
    df["utterance_clean"] = df["utterance"].map(aggressive_clean)
    df["utterance_preserved"] = df["utterance"].map(preserve_text)
    df["context_preserved"] = df["context"].map(preserve_text)
    df["text_input"] = (df["context_preserved"] + " [SEP] " + df["utterance_preserved"]).str.strip()
    # modality mask columns (text, audio, visual)
    df["mask_text"] = 1
    df["mask_audio"] = df["has_audio"].astype(int)
    df["mask_visual"] = df["has_video"].astype(int)

    if args.demux:
        n_ok = 0
        for _, row in df.iterrows():
            if row["has_video"] and row["video_path"]:
                if demux_audio(row["video_path"], row["audio_path"], sr=cfg["audio"]["sample_rate"]):
                    n_ok += 1
        df["has_audio"] = df["audio_path"].map(lambda p: Path(p).exists())
        df["mask_audio"] = df["has_audio"].astype(int)
        logging.info("Demuxed %d audio files.", n_ok)

    out_csv = paths["processed"] / "utterances.csv"
    df.to_csv(out_csv, index=False)
    try:
        df.to_parquet(paths["processed"] / "utterances.parquet", index=False)
    except Exception:
        pass
    report = verify_dataset(df)
    try:
        from mustard.eda import generate_eda

        eda = generate_eda()
        logging.info("EDA figures written. Summary: %s", {k: eda[k] for k in ("n_utterances", "n_sarcastic", "n_non_sarcastic") if k in eda})
    except Exception as exc:  # noqa: BLE001
        logging.warning("EDA generation skipped: %s", exc)
    logging.info("Preprocessed %d utterances. Report: %s", len(df), report)
    logging.info(
        "Casing and punctuation are preserved for the transformer path; "
        "utterance_clean is the TF-IDF-only pipeline (lowercase, no punctuation, stemmed)."
    )


if __name__ == "__main__":
    main()
