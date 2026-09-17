#!/usr/bin/env python3
"""Extract and cache per-modality features."""

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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    from mustard.dataset import attach_media_paths, build_utterance_table, load_raw_csv
    from mustard.features import cache_all_features

    df = attach_media_paths(build_utterance_table(load_raw_csv()))
    pack = cache_all_features(df, force=args.force)
    meta = pack["meta"]
    print("Feature cache written to artifacts/features/")
    print("Dimensionality:")
    for name, shape in meta["dims"].items():
        print(f"  {name:14s} {shape}")
    print(f"  encoder: {meta['encoder']}")
    print(f"  audio present: {meta['n_audio_present']} / {meta['n']}")
    print(f"  visual present: {meta['n_visual_present']} / {meta['n']}")
    print(
        "Each modality is stored raw; train.py standardizes TEXT / AUDIO / VISUAL "
        "independently using training-fold statistics only."
    )


if __name__ == "__main__":
    main()
