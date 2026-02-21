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
  - facebook/musicgen-small
  - facebook/musicgen-medium
  - facebook/musicgen-large
  - facebook/musicgen-stereo-small
  - facebook/musicgen-stereo-medium
  - facebook/musicgen-stereo-large
  - facebook/musicgen-melody
  - facebook/musicgen-melody-large
  - facebook/musicgen-stereo-melody
  - facebook/musicgen-stereo-melody-large
  - facebook/musicgen-style

Requires: transformers>=5.1,<6.0, huggingface-hub>=1.4,<2.0, scipy>=1.11, torch
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from transformers import pipeline

WEIGHT_GLOBS = ("*.safetensors", "pytorch_model*.bin", "state_dict.bin")
INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _cache_roots() -> list[Path]:
    roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path.home() / "Library" / "Caches" / "huggingface" / "hub")

    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _is_model_cached(model_id: str) -> bool:
    folder = f"models--{model_id.replace('/', '--')}"
    for root in _cache_roots():
        repo_dir = root / folder
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue
        for snap in snapshots_dir.iterdir():
            if not snap.is_dir():
                continue
            for pattern in WEIGHT_GLOBS:
                if any(snap.glob(pattern)):
                    return True
            for index_file in INDEX_FILES:
                if (snap / index_file).exists():
                    return True
    return False


def _resolve_hf_cli() -> str:
    local_hf = Path(sys.executable).resolve().parent / (
        "hf.exe" if os.name == "nt" else "hf"
    )
    if local_hf.exists():
        return str(local_hf)

    hf_in_path = shutil.which("hf")
    if hf_in_path:
        return hf_in_path

    raise FileNotFoundError(
        "Could not find 'hf' CLI. Install with: pip install 'huggingface_hub[cli]'"
    )


def _ensure_model_available(model_id: str, revision: str | None) -> None:
    if _is_model_cached(model_id):
        print(f"[MusicGen] Using cached model: {model_id}", file=sys.stderr)
        return

    hf_cli = _resolve_hf_cli()
    cmd = [hf_cli, "download", model_id]
    if revision:
        cmd.extend(["--revision", revision])

    rev_text = revision if revision else "default"
    print(
        f"[MusicGen] Downloading model: {model_id} (revision: {rev_text})",
        file=sys.stderr,
    )
    print("[MusicGen] Download progress:", file=sys.stderr)

    res = subprocess.run(cmd, check=False)  # noqa: S603  # nosec B603
    if res.returncode != 0:
        raise RuntimeError(f"Failed to download model: {model_id}")

    if not _is_model_cached(model_id):
        raise RuntimeError(
            f"Model download did not produce local weights for: {model_id}"
        )

    print(f"[MusicGen] Model ready: {model_id}", file=sys.stderr)


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
    p.add_argument(
        "--revision",
        default="",
        help="Optional pinned Hugging Face revision SHA",
    )
    args = p.parse_args()

    model_id = args.model
    revision = args.revision.strip() or None

    try:
        _ensure_model_available(model_id, revision)
    except Exception as ensure_err:
        print(f"Failed to ensure model availability: {ensure_err}", file=sys.stderr)
        return 2

    # Try text-to-audio pipeline with selected model
    try:
        pipe = pipeline(
            "text-to-audio",
            model=model_id,
            revision=revision,
            trust_remote_code=True,
        )
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
