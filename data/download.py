#!/usr/bin/env python3
"""Fetch MUStARD++ transcripts, optionally videos, and verify integrity."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

CSV_URL = "https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/main/mustard++_text.csv"
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1kUdT2yU7ERJ5KdauObTj5oQsBlSrvTlW"


def download_csv(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading mustard++_text.csv → {dest}")
    urllib.request.urlretrieve(CSV_URL, dest)
    return dest


def try_download_videos(video_dir: Path) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(video_dir).free / (1024**3)
    if free_gb < 8:
        print(
            f"Skipping video download: only {free_gb:.1f} GB free. "
            "The utterance clips typically need several GB.\n"
            f"Download manually from:\n  {DRIVE_FOLDER}\n"
            f"and place .mp4 files in {video_dir}"
        )
        return
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. pip install gdown, or download videos manually.")
        print(DRIVE_FOLDER)
        return
    print(f"Fetching Google Drive folder into {video_dir} ...")
    gdown.download_folder(url=DRIVE_FOLDER, output=str(video_dir), quiet=False, remaining_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MUStARD++ data.")
    parser.add_argument("--videos", action="store_true", help="Also attempt Google Drive video download.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    csv_path = ROOT / "data" / "raw" / "mustard++_text.csv"
    if args.force or not csv_path.exists():
        download_csv(csv_path)
    else:
        print(f"CSV already present: {csv_path}")

    from mustard.dataset import attach_media_paths, build_utterance_table, load_raw_csv, verify_dataset

    raw = load_raw_csv(csv_path)
    utt = attach_media_paths(build_utterance_table(raw))
    report = verify_dataset(utt)
    print("Integrity report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    if not report["n_matches_paper"]:
        print("WARNING: utterance count does not match the paper's 1,202.")
    if not report["balanced"]:
        print("WARNING: class split is not 601/601.")

    processed = ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    utt.to_csv(processed / "utterances.csv", index=False)
    try:
        utt.to_parquet(processed / "utterances.parquet", index=False)
    except Exception:
        pass

    if args.videos:
        try_download_videos(ROOT / "data" / "videos")
        utt = attach_media_paths(utt)
        print(f"Videos matched: {int(utt['has_video'].sum())}/{len(utt)}")
    else:
        print(
            "\nRaw videos are not fetched by default (large Drive folder).\n"
            f"  1. Open {DRIVE_FOLDER}\n"
            f"  2. Put utterance clips in {ROOT / 'data' / 'videos'}\n"
            "  3. Re-run: python data/download.py --videos\n"
            "Text-only training and the UI demo work without the clips.\n"
            "Audio is demuxed from those videos with ffmpeg; there is no separate audio archive."
        )


if __name__ == "__main__":
    main()
