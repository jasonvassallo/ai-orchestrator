# MusicGen Audio (Separate Virtual Environment)

This project uses a dedicated virtual environment for MusicGen audio generation to avoid dependency conflicts with MLX and other providers.

## Why a separate venv?

- MLX / mlx-lm currently prefers specific Transformers/HF Hub versions
- MusicGen (Transformers-based) may require different versions
- A separate `.music-venv` keeps both stacks stable

## Create/Update the audio venv

```bash
python3 -m venv .music-venv
./.music-venv/bin/python -m pip install -U pip
# Preferred stack (Transformers 4.x)
./.music-venv/bin/python -m pip install \
  "transformers>=4.45,<5.0" \
  "huggingface-hub>=0.34,<1.0" \
  "scipy>=1.11.0" \
  "torch" \
  "accelerate>=0.26.0"
```

If you run into a `MusicgenDecoderConfig` error with the default model, try updating to the latest pre-release of Transformers (and keep HF Hub <1.0):

```bash
./.music-venv/bin/python -m pip install \
  "transformers==5.0.0rc1" \
  "huggingface-hub>=0.34,<1.0" \
  "scipy>=1.11.0" \
  "torch" \
  "accelerate>=0.26.0"
```

Note: The underlying HF model may change its config over time. If a particular combo fails, try the alternate set above.

## How it works in the app

- The orchestrator calls `scripts/musicgen_generate.py` via the audio venv to generate a WAV file
- The CLI/GUI/TUI then list the generated files under `~/Music/AI Orchestrator/`
- You can override the venv path by setting `MUSICGEN_VENV=/path/to/venv`

## Manual test

```bash
./.music-venv/bin/python scripts/musicgen_generate.py \
  --prompt "90s tech house groove, 126 BPM, G minor" \
  --duration 5 \
  --output /tmp/musicgen_test.wav
```

If successful, it prints the output path and writes `/tmp/musicgen_test.wav`.

## Troubleshooting

- `MusicgenDecoderConfig has no attribute 'decoder'`:
  - Switch the Transformers version (use either 4.45+ <5.0 or 5.0.0rc1) while keeping `huggingface-hub<1.0`.
- Model downloads are slow:
  - First run will download model weights to your HF cache. Re-uses cache afterward.
- Use a single HF cache:
  - `export HF_HOME="$HOME/Library/Caches/huggingface"`
