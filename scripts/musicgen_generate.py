#!/usr/bin/env python3
"""
MusicGen audio generator (standalone)

Runs in the dedicated audio venv. Usage:

  python scripts/musicgen_generate.py \
    --prompt "90s tech house groove, 126 BPM, G minor" \
    --duration 5 \
    --output /path/to/out.wav \
    --model facebook/musicgen-small

Available models (all from Hugging Face):
  - facebook/musicgen-small       (300M params, fast)
  - facebook/musicgen-medium      (1.5B params, balanced)
  - facebook/musicgen-large       (3.3B params, high quality)
  - facebook/musicgen-stereo-small   (stereo output)
  - facebook/musicgen-stereo-medium  (stereo output)
  - facebook/musicgen-stereo-large   (stereo output)
  - facebook/musicgen-melody      (melody-conditioned)
  - facebook/musicgen-melody-large   (melody-conditioned)

Requires: transformers>=4.45,<5.0, huggingface-hub>=0.34,<1.0, scipy>=1.11, torch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from transformers import pipeline


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate audio using MusicGen models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt", required=True, help="Text prompt describing the music")
    p.add_argument(
        "--duration", type=int, default=5, help="Duration in seconds (max 30)"
    )
    p.add_argument("--output", required=True, help="Output WAV file path")
    p.add_argument(
        "--model",
        default="facebook/musicgen-small",
        help="Hugging Face model ID (default: facebook/musicgen-small)",
    )
    args = p.parse_args()

    model_id = args.model

    # Try text-to-audio pipeline with selected model
    try:
        pipe = pipeline("text-to-audio", model=model_id, trust_remote_code=True)
    except Exception as pipeline_err:
        # Fallback to audiocraft if transformers pipeline fails
        try:
            from audiocraft.models import MusicGen as _AG

            gen = _AG.get_pretrained(model_id)
            gen.set_generation_params(duration=max(1, min(args.duration, 30)))
            wavs = gen.generate([args.prompt])
            audio = wavs[0].cpu().numpy().squeeze()
            sr = 32000
            audio_i16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(str(out_path), sr, audio_i16)
            print(str(out_path))
            return 0
        except Exception:
            print(
                f"Failed to initialize MusicGen with model {model_id}: {pipeline_err}",
                file=sys.stderr,
            )
            return 2

    max_tokens = max(50, min(args.duration * 50, 1500))
    out = pipe(args.prompt, forward_params={"max_new_tokens": max_tokens})
    if not out:
        print("Generation failed", file=sys.stderr)
        return 2
    if isinstance(out, dict):
        audio = out.get("audio")
        sr = out.get("sampling_rate", 32000)
    elif isinstance(out, list):
        audio = out[0].get("audio")
        sr = out[0].get("sampling_rate", 32000)
    else:
        print("Unexpected pipeline output type", file=sys.stderr)
        return 2
    if audio is None:
        print("Pipeline did not return audio array", file=sys.stderr)
        return 3

    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    audio_i16 = (audio * 32767.0).astype(np.int16)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(str(out_path), sr, audio_i16)

    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
