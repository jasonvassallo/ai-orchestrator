"""
Music Generation Module
=======================

Generates audio and MIDI files using AI models and music theory.
Supports:
- MIDI generation with separate tracks (drums, bass, chords)
- MusicGen for AI audio generation
- 90s tech-house and progressive house patterns
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# MIDI creation
try:
    from midiutil import MIDIFile

    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

# MusicGen (optional - requires torch + transformers)
MUSICGEN_AVAILABLE = False
MUSICGEN_MODEL = None
try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Music theory constants
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],  # Great for house music
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
}

# 90s Tech House chord progressions (minor key focused)
CHORD_PATTERNS = {
    "tech_house_90s": [
        # Classic minor progression with tension
        (0, 4),
        (5, 4),
        (3, 4),
        (4, 4),  # i - VI - iv - V
    ],
    "progressive_house": [
        (0, 8),
        (5, 4),
        (3, 4),
        (4, 8),
        (0, 8),  # Long builds
    ],
    "funky_house": [
        (0, 2),
        (0, 2),
        (3, 2),
        (3, 2),
        (5, 2),
        (5, 2),
        (4, 2),
        (4, 2),
    ],
    "deep_house": [
        (0, 4),
        (3, 4),
        (5, 4),
        (4, 4),
    ],
    "minimal_tech": [
        (0, 8),
        (0, 8),
        (3, 8),
        (0, 8),
    ],
}

# 90s Tech House drum patterns (16 steps = 1 bar at 4/4)
DRUM_PATTERNS = {
    "tech_house_90s": {
        # Classic 4-on-the-floor with syncopated kicks
        "kick": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
        "snare": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "clap": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hihat": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "open_hat": [
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            1,
        ],  # Offbeat open hats
        "rimshot": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    },
    "funky_90s": {
        # Funky syncopated pattern
        "kick": [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        "snare": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
        "clap": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        "hihat": [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        "open_hat": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        "shaker": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "conga": [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
    },
    "progressive_house": {
        "kick": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "clap": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hihat": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        "open_hat": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        "ride": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    },
    "minimal": {
        "kick": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        "rimshot": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        "hihat": [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    },
}

# General MIDI drum map (channel 10)
DRUM_NOTES = {
    "kick": 36,
    "kick2": 35,
    "snare": 38,
    "snare2": 40,
    "rimshot": 37,
    "clap": 39,
    "hihat": 42,
    "hihat_pedal": 44,
    "open_hat": 46,
    "shaker": 70,
    "tambourine": 54,
    "conga": 63,
    "conga_high": 62,
    "tom_low": 45,
    "tom_mid": 47,
    "tom_high": 50,
    "crash": 49,
    "ride": 51,
    "ride_bell": 53,
}


@dataclass
class MusicParameters:
    """Parameters for music generation."""

    prompt: str = ""
    key: str = "G"  # G minor is classic for tech house
    scale: str = "minor"
    genre: str = "tech_house_90s"
    mood: str = "groovy"
    energy: float = 0.75
    bpm: int = 126  # 124-128 range
    duration: int = 30  # seconds
    output_format: str = "all"  # midi, wav, mp3, all
    separate_tracks: bool = True  # Create separate files for each track

    @classmethod
    def from_dict(cls, data: dict) -> MusicParameters:
        """Create from dictionary."""
        # Parse key from "C Major" format
        key_str = data.get("key", "G Minor")
        if key_str and key_str != "Auto":
            parts = key_str.split()
            key = parts[0] if parts else "G"
            scale = parts[1].lower() if len(parts) > 1 else "minor"
        else:
            key = "G"
            scale = "minor"

        # Map genre to internal name
        genre_map = {
            "Electronic": "tech_house_90s",
            "Orchestral": "progressive_house",
            "Jazz": "funky_90s",
            "Rock": "progressive_house",
            "Pop": "progressive_house",
            "Hip Hop": "funky_90s",
            "Ambient": "minimal",
            "Classical": "progressive_house",
            "Folk": "minimal",
            "Blues": "funky_90s",
            "Country": "progressive_house",
            "R&B": "funky_90s",
            "Metal": "progressive_house",
            "Indie": "minimal",
            "Auto": "tech_house_90s",
        }
        genre = genre_map.get(data.get("genre", ""), "tech_house_90s")

        # Get BPM, default to 124-128 range for tech house
        bpm = data.get("bpm")
        if not bpm:
            bpm = random.randint(124, 128)  # noqa: S311

        return cls(
            prompt=data.get("prompt", ""),
            key=key,
            scale=scale,
            genre=genre,
            mood=data.get("mood", "groovy"),
            energy=data.get("energy", 0.75),
            bpm=bpm,
            duration=data.get("duration", 30),
            output_format=data.get("format", "all"),
            separate_tracks=True,
        )


def get_output_dir() -> Path:
    """Get the output directory for generated music."""
    output_dir = Path.home() / "Music" / "AI Orchestrator"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _find_project_root() -> Path | None:
    """Find the project root by looking for pyproject.toml or .git."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return None


def get_scale_notes(root: str, scale: str, octave: int = 4) -> list[int]:
    """Get MIDI note numbers for a scale starting at the given root."""
    root_idx = NOTES.index(root) if root in NOTES else 0
    intervals = SCALES.get(scale, SCALES["minor"])
    base_note = (octave + 1) * 12 + root_idx  # MIDI note number
    return [base_note + i for i in intervals]


def create_drums_midi(params: MusicParameters, filename_base: str) -> str:
    """Create a MIDI file with only drums."""
    if not MIDI_AVAILABLE:
        raise RuntimeError("midiutil not installed")

    midi = MIDIFile(1)
    midi.addTrackName(0, 0, "Drums")
    midi.addTempo(0, 0, params.bpm)

    # Calculate measures
    beats_per_measure = 4
    measures = (params.duration * params.bpm) // (60 * beats_per_measure)
    measures = max(4, min(measures, 64))

    # Get drum pattern
    pattern_name = params.genre if params.genre in DRUM_PATTERNS else "tech_house_90s"
    drum_pattern = DRUM_PATTERNS[pattern_name]
    steps_per_beat = 4
    total_steps = measures * beats_per_measure * steps_per_beat

    for drum_name, pattern in drum_pattern.items():
        note = DRUM_NOTES.get(drum_name, 36)
        pattern_len = len(pattern)

        for step in range(total_steps):
            if pattern[step % pattern_len]:
                time = step / steps_per_beat
                # Humanize velocity
                base_velocity = 80 + int(params.energy * 40)
                velocity = min(127, base_velocity + random.randint(-10, 10))  # noqa: S311
                # Slight timing humanization
                time_offset = random.uniform(-0.01, 0.01)  # noqa: S311
                midi.addNote(0, 9, note, max(0, time + time_offset), 0.2, velocity)

    output_path = get_output_dir() / f"{filename_base}_drums.mid"
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return str(output_path)


def create_bass_midi(params: MusicParameters, filename_base: str) -> str:
    """Create a MIDI file with only bass."""
    if not MIDI_AVAILABLE:
        raise RuntimeError("midiutil not installed")

    midi = MIDIFile(1)
    midi.addTrackName(0, 0, "Bass")
    midi.addTempo(0, 0, params.bpm)

    # Program change to synth bass
    midi.addProgramChange(0, 0, 0, 38)  # Synth Bass 1

    beats_per_measure = 4
    measures = (params.duration * params.bpm) // (60 * beats_per_measure)
    measures = max(4, min(measures, 64))

    scale_notes = get_scale_notes(params.key, params.scale, octave=2)

    # Get chord pattern
    pattern_name = params.genre if params.genre in CHORD_PATTERNS else "tech_house_90s"
    chord_pattern = CHORD_PATTERNS[pattern_name]

    current_beat = 0
    pattern_idx = 0

    while current_beat < measures * beats_per_measure:
        chord_root_offset, duration = chord_pattern[pattern_idx % len(chord_pattern)]
        bass_note = scale_notes[chord_root_offset % len(scale_notes)]

        # 90s tech house bass pattern: hits on beat with occasional syncopation
        for beat_offset in range(
            min(duration, measures * beats_per_measure - current_beat)
        ):
            beat = current_beat + beat_offset

            # Main bass hit on the beat
            velocity = min(127, 90 + int(params.energy * 30) + random.randint(-5, 5))  # noqa: S311
            midi.addNote(0, 0, bass_note, beat, 0.4, velocity)

            # Occasional offbeat ghost note (funky element)
            if random.random() < 0.3 and params.genre in [  # noqa: S311
                "funky_90s",
                "tech_house_90s",
            ]:
                ghost_velocity = velocity - 30
                midi.addNote(0, 0, bass_note, beat + 0.5, 0.2, ghost_velocity)

        current_beat += duration
        pattern_idx += 1

    output_path = get_output_dir() / f"{filename_base}_bass.mid"
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return str(output_path)


def create_chords_midi(params: MusicParameters, filename_base: str) -> str:
    """Create a MIDI file with only chords/pads."""
    if not MIDI_AVAILABLE:
        raise RuntimeError("midiutil not installed")

    midi = MIDIFile(1)
    midi.addTrackName(0, 0, "Chords")
    midi.addTempo(0, 0, params.bpm)

    # Program change to pad
    midi.addProgramChange(0, 0, 0, 89)  # Pad 2 (warm)

    beats_per_measure = 4
    measures = (params.duration * params.bpm) // (60 * beats_per_measure)
    measures = max(4, min(measures, 64))

    scale_notes = get_scale_notes(params.key, params.scale, octave=4)

    # Get chord pattern
    pattern_name = params.genre if params.genre in CHORD_PATTERNS else "tech_house_90s"
    chord_pattern = CHORD_PATTERNS[pattern_name]

    current_beat = 0
    pattern_idx = 0

    while current_beat < measures * beats_per_measure:
        chord_root_offset, duration = chord_pattern[pattern_idx % len(chord_pattern)]

        # Build minor chord (root, minor 3rd, 5th, optional 7th)
        root = scale_notes[chord_root_offset % len(scale_notes)]
        third = scale_notes[(chord_root_offset + 2) % len(scale_notes)]
        fifth = scale_notes[(chord_root_offset + 4) % len(scale_notes)]

        # Adjust octaves to keep voicing tight
        if third < root:
            third += 12
        if fifth < third:
            fifth += 12

        # Add 7th for jazzy tech house feel
        seventh = scale_notes[(chord_root_offset + 6) % len(scale_notes)]
        if seventh < fifth:
            seventh += 12

        chord_notes = [root, third, fifth]
        if params.energy > 0.5:
            chord_notes.append(seventh)

        velocity = min(127, 60 + int(params.energy * 30))

        # Pad-style sustained chords
        actual_duration = min(duration, measures * beats_per_measure - current_beat)
        for note in chord_notes:
            midi.addNote(0, 0, note, current_beat, actual_duration - 0.1, velocity)

        current_beat += duration
        pattern_idx += 1

    output_path = get_output_dir() / f"{filename_base}_chords.mid"
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return str(output_path)


def create_combined_midi(params: MusicParameters, filename_base: str) -> str:
    """Create a MIDI file with all tracks combined."""
    if not MIDI_AVAILABLE:
        raise RuntimeError("midiutil not installed")

    midi = MIDIFile(3)

    midi.addTrackName(0, 0, "Drums")
    midi.addTrackName(1, 0, "Bass")
    midi.addTrackName(2, 0, "Chords")

    for track in range(3):
        midi.addTempo(track, 0, params.bpm)

    # Program changes
    midi.addProgramChange(1, 0, 0, 38)  # Bass
    midi.addProgramChange(2, 0, 0, 89)  # Pad

    beats_per_measure = 4
    measures = (params.duration * params.bpm) // (60 * beats_per_measure)
    measures = max(4, min(measures, 64))

    # --- Drums (Track 0, Channel 9) ---
    pattern_name = params.genre if params.genre in DRUM_PATTERNS else "tech_house_90s"
    drum_pattern = DRUM_PATTERNS[pattern_name]
    steps_per_beat = 4
    total_steps = measures * beats_per_measure * steps_per_beat

    for drum_name, pattern in drum_pattern.items():
        note = DRUM_NOTES.get(drum_name, 36)
        pattern_len = len(pattern)

        for step in range(total_steps):
            if pattern[step % pattern_len]:
                time = step / steps_per_beat
                velocity = min(
                    127,
                    80 + int(params.energy * 40) + random.randint(-8, 8),  # noqa: S311
                )
                midi.addNote(0, 9, note, time, 0.2, velocity)

    # --- Bass (Track 1, Channel 0) ---
    scale_notes_bass = get_scale_notes(params.key, params.scale, octave=2)
    chord_pattern_name = (
        params.genre if params.genre in CHORD_PATTERNS else "tech_house_90s"
    )
    chord_pattern = CHORD_PATTERNS[chord_pattern_name]

    current_beat = 0
    pattern_idx = 0

    while current_beat < measures * beats_per_measure:
        chord_root_offset, duration = chord_pattern[pattern_idx % len(chord_pattern)]
        bass_note = scale_notes_bass[chord_root_offset % len(scale_notes_bass)]

        for beat_offset in range(
            min(duration, measures * beats_per_measure - current_beat)
        ):
            beat = current_beat + beat_offset
            velocity = min(127, 90 + int(params.energy * 30))
            midi.addNote(1, 0, bass_note, beat, 0.4, velocity)

        current_beat += duration
        pattern_idx += 1

    # --- Chords (Track 2, Channel 1) ---
    scale_notes_chords = get_scale_notes(params.key, params.scale, octave=4)

    current_beat = 0
    pattern_idx = 0

    while current_beat < measures * beats_per_measure:
        chord_root_offset, duration = chord_pattern[pattern_idx % len(chord_pattern)]

        root = scale_notes_chords[chord_root_offset % len(scale_notes_chords)]
        third = scale_notes_chords[(chord_root_offset + 2) % len(scale_notes_chords)]
        fifth = scale_notes_chords[(chord_root_offset + 4) % len(scale_notes_chords)]

        if third < root:
            third += 12
        if fifth < third:
            fifth += 12

        velocity = min(127, 60 + int(params.energy * 30))
        actual_duration = min(duration, measures * beats_per_measure - current_beat)

        for note in [root, third, fifth]:
            midi.addNote(2, 1, note, current_beat, actual_duration - 0.1, velocity)

        current_beat += duration
        pattern_idx += 1

    output_path = get_output_dir() / f"{filename_base}_full.mid"
    with open(output_path, "wb") as f:
        midi.writeFile(f)

    return str(output_path)


async def generate_audio_with_musicgen(
    params: MusicParameters, filename_base: str
) -> str | None:
    """Generate audio using MusicGen in a dedicated venv if available.

    Order:
      1) External venv runner (.music-venv) calling scripts/musicgen_generate.py
      2) Audiocraft (best in-process fallback)
      3) Transformers (in-process fallback)
    """
    # 0) Try dedicated audio venv runner
    try:
        project_root = _find_project_root()
        if project_root:
            venv = os.environ.get("MUSICGEN_VENV") or str(project_root / ".music-venv")
            py = Path(venv) / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            script = project_root / "scripts" / "musicgen_generate.py"
            if py.exists() and script.exists():
                prompt_parts = []
                if params.prompt:
                    prompt_parts.append(params.prompt)
                prompt_parts.append(f"{params.bpm} BPM")
                prompt_parts.append(f"{params.key} {params.scale}")
                genre_descriptions = {
                    "tech_house_90s": "90s tech house, funky beats, underground electronic",
                    "funky_90s": "funky house, groovy bassline, 90s electronic",
                    "progressive_house": "progressive house, atmospheric, building",
                    "deep_house": "deep house, soulful, smooth",
                    "minimal": "minimal techno, hypnotic, stripped back",
                }
                prompt_parts.append(
                    genre_descriptions.get(params.genre, "electronic dance music")
                )
                prompt = ", ".join(prompt_parts)

                output_path = get_output_dir() / f"{filename_base}_audio.wav"

                import subprocess

                cmd = [
                    str(py),
                    str(script),
                    "--prompt",
                    prompt,
                    "--duration",
                    str(max(1, min(params.duration, 30))),
                    "--output",
                    str(output_path),
                ]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0:
                    path_str = res.stdout.strip().splitlines()[-1]
                    if Path(path_str).exists():
                        return path_str
    except Exception:
        pass
    # 1) Try Audiocraft (preferred)
    try:
        import numpy as np
        import scipy.io.wavfile as wav
        from audiocraft.models import MusicGen

        prompt_parts = []
        if params.prompt:
            prompt_parts.append(params.prompt)
        prompt_parts.append(f"{params.bpm} BPM")
        prompt_parts.append(f"{params.key} {params.scale}")
        genre_descriptions = {
            "tech_house_90s": "90s tech house, funky beats, underground electronic",
            "funky_90s": "funky house, groovy bassline, 90s electronic",
            "progressive_house": "progressive house, atmospheric, building",
            "deep_house": "deep house, soulful, smooth",
            "minimal": "minimal techno, hypnotic, stripped back",
        }
        prompt_parts.append(
            genre_descriptions.get(params.genre, "electronic dance music")
        )
        prompt = ", ".join(prompt_parts)

        model = MusicGen.get_pretrained("facebook/musicgen-small")
        model.set_generation_params(duration=max(1, min(params.duration, 30)))
        wavs = model.generate([prompt])  # List[Tensor] with shape [1, T]
        audio = wavs[0].cpu().numpy().squeeze()
        sampling_rate = 32000  # MusicGen small default sample rate
        # Normalize to int16
        audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        output_path = get_output_dir() / f"{filename_base}_audio.wav"
        wav.write(str(output_path), rate=sampling_rate, data=audio_int16)
        return str(output_path)
    except Exception:
        pass

    # 2) Fallback to Transformers (when compatible)
    try:
        import scipy.io.wavfile as wav
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
    except ImportError:
        return None

    # Build prompt for MusicGen
    prompt_parts = []
    if params.prompt:
        prompt_parts.append(params.prompt)

    prompt_parts.append(f"{params.bpm} BPM")
    prompt_parts.append(f"{params.key} {params.scale}")

    genre_descriptions = {
        "tech_house_90s": "90s tech house, funky beats, underground electronic",
        "funky_90s": "funky house, groovy bassline, 90s electronic",
        "progressive_house": "progressive house, atmospheric, building",
        "deep_house": "deep house, soulful, smooth",
        "minimal": "minimal techno, hypnotic, stripped back",
    }
    prompt_parts.append(genre_descriptions.get(params.genre, "electronic dance music"))

    prompt = ", ".join(prompt_parts)

    try:
        # Load model (try local first to avoid downloads)
        model_id = "facebook/musicgen-small"
        try:
            processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_id, local_files_only=True
            )
        except OSError:
            # Fallback to downloading if not found locally
            print(f"Model {model_id} not found locally. Downloading...")
            processor = AutoProcessor.from_pretrained(model_id)
            model = MusicgenForConditionalGeneration.from_pretrained(model_id)

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt",
        )

        # Generate audio (duration in tokens, ~50 tokens per second)
        max_tokens = min(params.duration * 50, 1500)  # Cap at 30 seconds

        audio_values = model.generate(**inputs, max_new_tokens=max_tokens)

        # Save as WAV
        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_data = audio_values[0, 0].numpy()

        output_path = get_output_dir() / f"{filename_base}_audio.wav"
        wav.write(str(output_path), rate=sampling_rate, data=audio_data)

        return str(output_path)

    except Exception as e:
        print(f"MusicGen error: {e}")
        return None


async def generate_music(params: MusicParameters) -> dict[str, Any]:
    """Generate music based on parameters.

    Returns a dict with file paths and metadata.
    """
    results = {
        "success": True,
        "files": [],
        "message": "",
        "params": {
            "key": params.key,
            "scale": params.scale,
            "bpm": params.bpm,
            "duration": params.duration,
            "genre": params.genre,
        },
    }

    # Generate filename base
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(
        c for c in params.prompt[:20] if c.isalnum() or c == " "
    ).strip()
    safe_prompt = safe_prompt.replace(" ", "_") or "track"
    filename_base = (
        f"{safe_prompt}_{params.key}{params.scale[0]}_{params.bpm}bpm_{timestamp}"
    )

    try:
        if not MIDI_AVAILABLE:
            results["success"] = False
            results["message"] = (
                "MIDI generation requires midiutil. Install with: pip install midiutil"
            )
            return results

        # Generate separate MIDI files
        if params.separate_tracks:
            # Drums
            drums_path = create_drums_midi(params, filename_base)
            results["files"].append(
                {
                    "type": "midi",
                    "track": "drums",
                    "path": drums_path,
                    "filename": os.path.basename(drums_path),
                }
            )

            # Bass
            bass_path = create_bass_midi(params, filename_base)
            results["files"].append(
                {
                    "type": "midi",
                    "track": "bass",
                    "path": bass_path,
                    "filename": os.path.basename(bass_path),
                }
            )

            # Chords
            chords_path = create_chords_midi(params, filename_base)
            results["files"].append(
                {
                    "type": "midi",
                    "track": "chords",
                    "path": chords_path,
                    "filename": os.path.basename(chords_path),
                }
            )

        # Always create combined file too
        combined_path = create_combined_midi(params, filename_base)
        results["files"].append(
            {
                "type": "midi",
                "track": "full",
                "path": combined_path,
                "filename": os.path.basename(combined_path),
            }
        )

        # Try to generate audio with MusicGen
        if params.output_format in ("wav", "mp3", "all"):
            if TORCH_AVAILABLE:
                audio_path = await generate_audio_with_musicgen(params, filename_base)
                if audio_path:
                    results["files"].append(
                        {
                            "type": "wav",
                            "track": "audio",
                            "path": audio_path,
                            "filename": os.path.basename(audio_path),
                        }
                    )
                else:
                    results["message"] += (
                        "\nAudio generation requires: pip install transformers torch scipy"
                    )
            else:
                results["message"] += (
                    "\nFor AI audio generation, install: pip install torch transformers scipy"
                )

        results["message"] = (
            f"Generated {len(results['files'])} file(s) in ~/Music/AI Orchestrator/"
        )

    except Exception as e:
        results["success"] = False
        results["message"] = f"Error generating music: {str(e)}"

    return results


def format_music_result(result: dict) -> str:
    """Format music generation result for display."""
    lines = []

    if result["success"]:
        lines.append("**Music Generated Successfully!**\n")
        lines.append(
            f"**Key:** {result['params']['key']} {result['params']['scale'].title()}"
        )
        lines.append(f"**BPM:** {result['params']['bpm']}")
        lines.append(
            f"**Style:** {result['params']['genre'].replace('_', ' ').title()}"
        )
        lines.append("")

        if result["files"]:
            lines.append("**Files Created:**")

            # Group by type
            midi_files = [f for f in result["files"] if f["type"] == "midi"]
            audio_files = [f for f in result["files"] if f["type"] in ("wav", "mp3")]

            if midi_files:
                lines.append("\n*MIDI Tracks:*")
                for f in midi_files:
                    track_name = f.get("track", "unknown").title()
                    lines.append(f"- **{track_name}:** `{f['filename']}`")

            if audio_files:
                lines.append("\n*Audio Files:*")
                for f in audio_files:
                    lines.append(f"- **Audio:** `{f['filename']}`")

        if result.get("message"):
            lines.append(f"\n{result['message']}")

        lines.append("\n📁 _Files saved to: ~/Music/AI Orchestrator/_")
        lines.append("\n💡 _Open in Logic Pro, Ableton, or GarageBand!_")
    else:
        lines.append("**Music Generation Failed**")
        lines.append(result.get("message", "Unknown error"))

    return "\n".join(lines)


def get_capabilities() -> dict[str, bool]:
    """Get available music generation capabilities."""
    return {
        "midi": MIDI_AVAILABLE,
        "audio": TORCH_AVAILABLE,
        "musicgen": TORCH_AVAILABLE,
    }
