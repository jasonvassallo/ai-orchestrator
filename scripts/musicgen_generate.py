#!/usr/bin/env python3
"""
MusicGen audio generator (standalone)

Runs in the dedicated audio venv. Usage:

  python scripts/musicgen_generate.py \
    --prompt "90s tech house groove, 126 BPM, G minor" \
    --duration 5 \
    --output /path/to/out.wav

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
    tried = []
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--duration", type=int, default=5)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="facebook/musicgen-small")
    args = p.parse_args()

    # Use text-to-audio pipeline with MusicGen
    last_err: Exception | None = None
    for model_id in [
        args.model,
        "facebook/musicgen-small",
        "facebook/musicgen-stereo-small",
    ]:
        try:
            pipe = pipeline("text-to-audio", model=model_id, trust_remote_code=True)
            tried.append(model_id)
            break
        except Exception as e:
            last_err = e
            continue
    else:
        # Fallback to audiocraft if available
        try:
            from audiocraft.models import MusicGen as _AG

            gen = _AG.get_pretrained("facebook/musicgen-small")
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
            print(f"Failed to initialize MusicGen pipeline: {last_err}")
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
