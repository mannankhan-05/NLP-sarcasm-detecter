#!/usr/bin/env python3
"""Build short MP4 demos with spoken audio + embedded transcripts (ffmpeg + espeak)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "demo"

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]

CLIPS = [
    {
        "stem": "sarcastic_oh_great",
        "label": "sarcastic",
        "seconds": 7,
        "transcript": "Oh great, another meeting that could have been an email.",
        "flite_voice": "awb",
        "audio_filter": "asetrate=16000*0.82,aresample=16000,atempo=0.88",
        "color": "0x2A2318",
        "caption": "SARCASTIC DEMO",
        "zoom": False,
    },
    {
        "stem": "sincere_dinner",
        "label": "non_sarcastic",
        "seconds": 3,
        "transcript": "I'm really looking forward to dinner with you tonight.",
        "flite_voice": "slt",
        "audio_filter": "asetrate=16000*1.08,aresample=16000,atempo=1.12",
        "color": "0x3D5A40",
        "caption": "NON-SARCASTIC DEMO",
        "zoom": True,
    },
]


def _font() -> str:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return p
    return ""


def _tts(text: str, wav: Path, spec: dict) -> None:
    """Speak via ffmpeg flite (no espeak binary required)."""
    script = wav.with_suffix(".tts.txt")
    script.write_text(text, encoding="utf-8")
    voice = spec.get("flite_voice", "slt")
    af = spec.get("audio_filter", "aresample=16000")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"flite=textfile={script}:voice={voice}",
        "-af", af,
        "-ar", "16000", "-ac", "1",
        str(wav),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not wav.exists():
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore")[-2000:])


def _stamp_wav_transcript(wav: Path, transcript: str) -> None:
    tagged = wav.with_suffix(".tagged.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(wav),
        "-metadata", f"comment=transcript: {transcript}",
        "-c", "copy",
        str(tagged),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode == 0 and tagged.exists():
        tagged.replace(wav)
    else:
        tagged.unlink(missing_ok=True)


def _mux(wav: Path, mp4: Path, spec: dict) -> None:
    seconds = int(spec["seconds"])
    font = _font()
    caption_file = wav.with_suffix(".caption.txt")
    caption_file.write_text(spec["caption"], encoding="utf-8")
    draw = f"drawtext=textfile='{caption_file}':fontcolor=0xE2B33A:fontsize=42:x=(w-text_w)/2:y=(h-text_h)/2"
    if font:
        draw += f":fontfile='{font}'"
    vf = draw
    if spec["zoom"]:
        vf = (
            "scale=1200:680,"
            "zoompan=z='min(zoom+0.0018,1.12)':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=960x540:fps=24,"
            f"{draw}"
        )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c={spec['color']}:s=960x540:d={seconds}:r=24",
        "-i", str(wav),
        "-af", f"apad=whole_dur={seconds}",
        "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        "-t", str(seconds),
        "-movflags", "+faststart",
        "-metadata", f"comment=transcript: {spec['transcript']}",
        "-metadata", f"title={spec['label']}",
        str(mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore")[-2500:])


def main() -> None:
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is required")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in CLIPS:
        wav = OUT / f"{spec['stem']}.wav"
        mp4 = OUT / f"{spec['stem']}.mp4"
        print(f"TTS {spec['stem']} ...")
        _tts(spec["transcript"], wav, spec)
        _stamp_wav_transcript(wav, spec["transcript"])
        print(f"mux {mp4.name} ...")
        _mux(wav, mp4, spec)
        sidecar = mp4.with_suffix(".txt")
        sidecar.write_text(spec["transcript"] + "\n", encoding="utf-8")
        manifest.extend(
            [
                {
                    "id": spec["stem"],
                    "kind": "video",
                    "file": mp4.name,
                    "url": f"/static/demo/{mp4.name}",
                    "label": spec["label"],
                    "seconds": spec["seconds"],
                    "transcript": spec["transcript"],
                    "format": "MP4 · H.264 + AAC 16 kHz",
                    "expected": spec["label"],
                    "how": "Audio/Video tab → drop the MP4. Transcript is embedded in metadata (no paste required).",
                },
                {
                    "id": f"{spec['stem']}_wav",
                    "kind": "audio",
                    "file": wav.name,
                    "url": f"/static/demo/{wav.name}",
                    "label": spec["label"],
                    "seconds": spec["seconds"],
                    "transcript": spec["transcript"],
                    "format": "WAV · 16 kHz PCM mono",
                    "expected": spec["label"],
                    "how": "Audio/Video tab → drop the WAV. Visual is masked; the Test button fills the transcript.",
                },
            ]
        )
        print(" wrote", mp4, mp4.stat().st_size, "bytes")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Done.", OUT)


if __name__ == "__main__":
    main()
