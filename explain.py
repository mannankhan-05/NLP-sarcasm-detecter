#!/usr/bin/env python3
"""Generate token / modality / audio / visual explanations for the report and UI."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


EXAMPLES = [
    "Oh great, another meeting that could have been an email.",
    "Yeah, because that worked out so well last time.",
    "I love being stuck in traffic. Best part of my day.",
    "I'm really looking forward to dinner with you tonight.",
    "Sure, I completely trust the guy who just spilled coffee on the server.",
]


def main() -> None:
    from mustard.config import get_paths
    from mustard.explain import gradient_token_attributions, heuristic_token_attributions
    from mustard.features import FrozenTextEncoder
    from mustard.io_utils import save_json

    paths = get_paths()
    encoder = FrozenTextEncoder()
    weight = _text_direction()
    rows = []
    for text in EXAMPLES:
        if encoder.available and weight is not None:
            attr = gradient_token_attributions(text, weight, encoder)
        else:
            attr = heuristic_token_attributions(text)
        rows.append({"text": text, "token_attributions": attr})
        logging.info("%s -> %s", text, [(a["token"], round(a["score"], 2)) for a in attr[:8]])
    save_json(rows, paths["metrics"] / "example_explanations.json")
    logging.info("Wrote %s", paths["metrics"] / "example_explanations.json")
    logging.info(
        "Visual Grad-CAM and pitch/energy contours are produced at inference time "
        "for uploaded clips (see POST /predict/multimodal)."
    )


def _text_direction():
    import numpy as np
    import torch

    from mustard.config import get_paths

    ckpt_path = get_paths()["checkpoints"] / "fusion_attn" / "serve.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    w = sd.get("text_proj.1.weight")
    if w is None:
        return None
    # sarcasm direction in utterance embedding space ≈ first d_utt columns, mean over hidden
    utt_dim = 768
    mat = w.detach().cpu().numpy()
    direction = mat[:, :utt_dim].mean(axis=0)
    return direction.astype(np.float32)


if __name__ == "__main__":
    main()
