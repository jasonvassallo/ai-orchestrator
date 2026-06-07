"""
Music Generation Module
=======================

Generates audio and MIDI files using AI models and music theory.
Supports:
- MIDI generation with separate tracks (drums, bass, chords)
- MusicGen via MLX (Apple Silicon native, preferred)
- MusicGen via PyTorch (audiocraft / transformers fallback)
- Stable Audio via MLX (requires mlx-audiogen server)
- 90s tech-house and progressive house patterns
"""

from __future__ import annotations

import ipaddress
import logging
import os
import random
import socket
import subprocess  # nosec B404
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

logger = logging.getLogger(__name__)

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


# Available MusicGen models (from Hugging Face)
MUSICGEN_MODELS = {
    "musicgen-small": {
        "id": "facebook/musicgen-small",
        "revision": "4c8334b02c6ec4e8664a91979669a501ec497792",
        "description": "Fast, 300M params (recommended for quick generation)",
        "stereo": False,
        "melody": False,
    },
    "musicgen-medium": {
        "id": "facebook/musicgen-medium",
        "revision": "d3bd7b00761b78ad7a8a05145ee31e7832e9916c",
        "description": "Balanced, 1.5B params",
        "stereo": False,
        "melody": False,
    },
    "musicgen-large": {
        "id": "facebook/musicgen-large",
        "revision": "15ccdc92099879e47b6da12c350cdb71d4eab3ca",
        "description": "High quality, 3.3B params (slower)",
        "stereo": False,
        "melody": False,
    },
    "musicgen-stereo-small": {
        "id": "facebook/musicgen-stereo-small",
        "revision": "3a110c0092820d470372c880c5ea06ce21a78ddd",
        "description": "Stereo output, 300M params",
        "stereo": True,
        "melody": False,
    },
    "musicgen-stereo-medium": {
        "id": "facebook/musicgen-stereo-medium",
        "revision": "2747e613fbaadc43b14257224326ef58c5c11b81",
        "description": "Stereo output, 1.5B params",
        "stereo": True,
        "melody": False,
    },
    "musicgen-stereo-large": {
        "id": "facebook/musicgen-stereo-large",
        "revision": "bda8ec330c9c17e09728589eaf678f58f7d2d932",
        "description": "Stereo output, 3.3B params",
        "stereo": True,
        "melody": False,
    },
    "musicgen-melody": {
        "id": "facebook/musicgen-melody",
        "revision": "68d653a95788ec0d2b0abccab22c0b3a200c2d90",
        "description": "Melody-conditioned, 1.5B params",
        "stereo": False,
        "melody": True,
    },
    "musicgen-melody-large": {
        "id": "facebook/musicgen-melody-large",
        "revision": "6fdf8d3d815995108c9bdb5183414ff464b171ac",
        "description": "Melody-conditioned, 3.3B params",
        "stereo": False,
        "melody": True,
    },
    "musicgen-stereo-melody": {
        "id": "facebook/musicgen-stereo-melody",
        "revision": "022c3bd20a7d77e7c014082f0391bcbcd3940a7a",
        "description": "Stereo + melody-conditioned, 1.5B params",
        "stereo": True,
        "melody": True,
    },
    "musicgen-stereo-melody-large": {
        "id": "facebook/musicgen-stereo-melody-large",
        "revision": "eea14861e6c29c47aea1055982a217468cd1634d",
        "description": "Stereo + melody-conditioned, 3.3B params",
        "stereo": True,
        "melody": True,
    },
    "musicgen-style": {
        "id": "facebook/musicgen-style",
        "revision": "bd0bbe32a093ef751414cf2dd7a9a7e4fef16e96",
        "description": "Style-conditioned generation (experimental)",
        "stereo": False,
        "melody": False,
    },
}


def get_musicgen_model_choices() -> list[tuple[str, str]]:
    """Get list of (display_name, model_key) tuples for UI dropdowns."""
    return [(info["description"], key) for key, info in MUSICGEN_MODELS.items()]


def get_musicgen_model_id(model_key: str) -> str:
    """Get the Hugging Face model ID for a given model key."""
    return MUSICGEN_MODELS.get(model_key, MUSICGEN_MODELS["musicgen-small"])["id"]


def get_musicgen_model_revision(model_key: str) -> str:
    """Get the pinned Hugging Face revision SHA for a given model key."""
    return MUSICGEN_MODELS.get(model_key, MUSICGEN_MODELS["musicgen-small"])["revision"]


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
    musicgen_model: str = "musicgen-small"  # MusicGen model key

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
            bpm = random.randint(124, 128)  # noqa: S311  # nosec B311

        # Get MusicGen model (default to small for faster generation)
        musicgen_model = data.get("musicgen_model", "musicgen-small")
        if musicgen_model not in MUSICGEN_MODELS:
            musicgen_model = "musicgen-small"

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
            musicgen_model=musicgen_model,
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
                velocity = min(127, base_velocity + random.randint(-10, 10))  # noqa: S311  # nosec B311
                # Slight timing humanization
                time_offset = random.uniform(-0.01, 0.01)  # noqa: S311  # nosec B311
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
            velocity = min(127, 90 + int(params.energy * 30) + random.randint(-5, 5))  # noqa: S311  # nosec B311
            midi.addNote(0, 0, bass_note, beat, 0.4, velocity)

            # Occasional offbeat ghost note (funky element)
            if random.random() < 0.3 and params.genre in [  # noqa: S311  # nosec B311
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
                    80 + int(params.energy * 40) + random.randint(-8, 8),  # noqa: S311  # nosec B311
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


# ---------------------------------------------------------------------------
# MLX Audio Generation Backend
# ---------------------------------------------------------------------------

# MLX server address (mlx-audiogen-server). This integration targets a local
# mlx-audiogen server; the host is configurable but constrained to loopback.
MLX_SERVER_HOST = os.environ.get("MLX_SERVER_HOST", "127.0.0.1")
MLX_SERVER_PORT = int(os.environ.get("MLX_SERVER_PORT", "8420"))


def _resolve_loopback_ip(host: str) -> str | None:
    """Resolve *host* to a numeric loopback IP to connect to, or None.

    MLX_SERVER_HOST is operator-configurable via the environment. We refuse to
    issue requests to a non-loopback host so the MLX integration cannot be
    turned into an SSRF primitive. A literal loopback IP (127.0.0.0/8 or ::1) is
    returned as-is; any other name is resolved and accepted only if EVERY
    resolved address is loopback, in which case the numeric address is returned
    so callers connect to that PINNED IP instead of re-resolving the name on
    each request (which would reopen a DNS-rebinding hole). Resolution failures,
    empty results, or any non-loopback address yield None (fail closed).
    """
    normalized = host.strip().lower()
    # Pin the conventional loopback literal without any DNS lookup.
    if normalized == "localhost":
        return "127.0.0.1"
    # Bare IP literal: decide without a DNS syscall.
    try:
        return normalized if ipaddress.ip_address(normalized).is_loopback else None
    except ValueError:
        pass  # not a bare IP — resolve the hostname below
    try:
        infos = socket.getaddrinfo(normalized, None)
    except (socket.gaierror, UnicodeError, ValueError):
        return None
    if not infos:
        return None
    pinned: str | None = None
    for info in infos:
        addr = info[4][0]
        try:
            if not ipaddress.ip_address(addr).is_loopback:
                return None
        except ValueError:
            return None
        if pinned is None:
            pinned = addr
    return pinned


def _is_loopback_host(host: str) -> bool:
    """True if *host* resolves entirely to loopback (see _resolve_loopback_ip).

    Thin boolean wrapper kept for readability and direct unit testing; the
    request paths use _resolve_loopback_ip()/_MLX_LOOPBACK_IP for the pinned IP.
    """
    return _resolve_loopback_ip(host) is not None


# Resolved once at import (MLX_SERVER_HOST is a module-level env constant) so the
# request paths never call the blocking getaddrinfo() inside the async event loop
# AND always connect to the pinned numeric IP (no per-request re-resolution, so a
# rebinding-prone hostname cannot leave loopback after import). Because this is
# fixed at import, tests exercise other hosts by calling _resolve_loopback_ip()
# directly (and mocking socket.getaddrinfo), not by setting MLX_SERVER_HOST after.
_MLX_LOOPBACK_IP: str | None = _resolve_loopback_ip(MLX_SERVER_HOST)
_MLX_HOST_IS_LOOPBACK: bool = _MLX_LOOPBACK_IP is not None


def _mlx_base_url() -> str:
    """Base URL for the MLX server, built from the pinned loopback IP.

    Callers must guard on ``_MLX_HOST_IS_LOOPBACK`` first. Rather than silently
    falling back to 127.0.0.1 when no loopback IP was pinned (a latent footgun
    if a future caller forgets the guard), fail fast — a raise (not an assert,
    which ``python -O`` strips) keeps the loopback-only invariant explicit.
    """
    if _MLX_LOOPBACK_IP is None:
        raise RuntimeError(
            "_mlx_base_url() called without a pinned loopback IP; "
            "MLX_SERVER_HOST is not a loopback address."
        )
    host = _MLX_LOOPBACK_IP
    if ":" in host:  # IPv6 literal needs brackets in a URL
        host = f"[{host}]"
    return f"http://{host}:{MLX_SERVER_PORT}"


def _mlx_request(
    path: str,
    timeout: float,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    """Blocking HTTP request to the loopback-pinned MLX server.

    Returns ``(status_code, body_bytes)``. Async callers MUST run this via
    ``asyncio.to_thread`` so it never blocks the event loop. The base URL host
    is the import-time-pinned loopback IP (``_MLX_LOOPBACK_IP``) and the scheme
    is a fixed ``http://``, so this is not an SSRF / ``file://`` vector.
    """
    import urllib.request

    method = "POST" if data is not None else "GET"
    req = urllib.request.Request(  # nosec B310 — loopback-pinned base URL
        f"{_mlx_base_url()}{path}",
        data=data,
        headers=headers or {},
        method=method,
    )
    # Host is the pinned loopback IP from _mlx_base_url(); fixed http:// scheme.
    # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
        return resp.status, resp.read()


async def _generate_via_mlx_server(
    prompt: str,
    duration: int,
    output_path: Path,
    model_type: str = "musicgen",
    **kwargs: Any,
) -> str | None:
    """Generate audio via the mlx-audiogen HTTP server.

    Sends a POST request to the local MLX server, polls for completion,
    and downloads the resulting WAV file.

    Args:
        prompt: Text description of desired audio.
        duration: Duration in seconds.
        output_path: Where to save the WAV file.
        model_type: 'musicgen' or 'stable_audio'.
        **kwargs: Additional generation parameters (temperature, seed, etc.)

    Returns:
        Path to the generated WAV file, or None if server unavailable.
    """
    import asyncio
    import json

    if not _MLX_HOST_IS_LOOPBACK:
        logger.warning(
            "Refusing MLX server request to non-loopback host %r; "
            "set MLX_SERVER_HOST to a loopback address.",
            MLX_SERVER_HOST,
        )
        return None

    # Check if server is reachable. Blocking I/O runs in a worker thread so it
    # never blocks the async event loop. (urllib raises HTTPError for non-2xx
    # and URLError for connection failures; both subclass OSError, so a bad
    # status surfaces through the except branch rather than the status check.)
    try:
        status_code, _ = await asyncio.to_thread(_mlx_request, "/api/models", 2.0)
        if status_code != 200:
            return None
    except (TimeoutError, OSError):
        logger.debug("MLX server not reachable (host=%s)", MLX_SERVER_HOST)
        return None

    # Submit generation request
    body = {
        "model": model_type,
        "prompt": prompt,
        "seconds": float(min(duration, 300)),
    }
    # Pass through optional params
    for key in ("temperature", "top_k", "guidance_coef", "steps",
                "cfg_scale", "seed", "melody_path", "style_audio_path",
                "style_coef"):
        if key in kwargs and kwargs[key] is not None:
            body[key] = kwargs[key]

    try:
        payload = json.dumps(body).encode("utf-8")
        _, raw = await asyncio.to_thread(
            _mlx_request,
            "/api/generate",
            10.0,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        result = json.loads(raw)
        job_id = result["id"]
        logger.info("MLX generation submitted: job %s", job_id)
    except Exception as exc:
        logger.debug("MLX generation submit failed: %s", exc)
        return None

    # Poll for completion (up to 10 minutes)
    max_polls = 1200  # 10 minutes at 500ms intervals
    for _ in range(max_polls):
        await asyncio.sleep(0.5)
        try:
            _, raw = await asyncio.to_thread(
                _mlx_request, f"/api/status/{job_id}", 5.0
            )
            status = json.loads(raw)

            if status["status"] == "done":
                # Download the WAV (network + file write both off the event loop).
                _, audio = await asyncio.to_thread(
                    _mlx_request, f"/api/audio/{job_id}", 30.0
                )
                await asyncio.to_thread(output_path.write_bytes, audio)
                logger.info("MLX audio saved to %s", output_path)
                return str(output_path)

            if status["status"] == "error":
                logger.warning("MLX generation error: %s", status.get("error"))
                return None
        except Exception as exc:
            logger.debug("MLX status poll error: %s", exc)
            return None

    logger.warning("MLX generation timed out for job %s", job_id)
    return None


async def generate_audio_with_musicgen(
    params: MusicParameters, filename_base: str
) -> str | None:
    """Generate audio using MusicGen (or Stable Audio) backends.

    Cascade order:
      0) MLX server (mlx-audiogen-server on localhost:8420, Apple Silicon native)
      1) External venv runner (.music-venv) calling scripts/musicgen_generate.py
      2) Audiocraft (best in-process PyTorch fallback)
      3) Transformers (in-process PyTorch fallback)
    """
    # Build prompt
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

    # 0) Try MLX server (fastest on Apple Silicon, no PyTorch needed)
    try:
        mlx_result = await _generate_via_mlx_server(
            prompt=prompt,
            duration=max(1, min(params.duration, 30)),
            output_path=output_path,
            model_type="musicgen",
        )
        if mlx_result:
            return mlx_result
    except Exception as exc:
        logger.debug("MLX server generation failed: %s", exc)

    # 1) Try dedicated audio venv runner
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

                # Get the selected MusicGen model ID
                model_id = get_musicgen_model_id(params.musicgen_model)
                revision = get_musicgen_model_revision(params.musicgen_model)

                cmd = [
                    str(py),
                    str(script),
                    "--prompt",
                    prompt,
                    "--duration",
                    str(max(1, min(params.duration, 30))),
                    "--output",
                    str(output_path),
                    "--model",
                    model_id,
                    "--revision",
                    revision,
                ]
                logger.info(
                    "MusicGen model selected: %s (%s).",
                    params.musicgen_model,
                    model_id,
                )
                logger.info(
                    "If the model is not cached, download progress is shown below."
                )
                res = subprocess.run(  # noqa: S603
                    cmd, check=False
                )  # nosec B603
                if res.returncode == 0 and output_path.exists():
                    return str(output_path)
    except Exception as exc:
        logger.debug("MusicGen script invocation failed.", exc_info=exc)
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

        model_id = get_musicgen_model_id(params.musicgen_model)
        model = MusicGen.get_pretrained(model_id)
        model.set_generation_params(duration=max(1, min(params.duration, 30)))
        wavs = model.generate([prompt])  # List[Tensor] with shape [1, T]
        audio = wavs[0].cpu().numpy().squeeze()
        sampling_rate = 32000  # MusicGen small default sample rate
        # Normalize to int16
        audio_int16 = np.int16(np.clip(audio, -1.0, 1.0) * 32767)
        output_path = get_output_dir() / f"{filename_base}_audio.wav"
        wav.write(str(output_path), rate=sampling_rate, data=audio_int16)
        return str(output_path)
    except Exception as exc:
        logger.debug("Audiocraft MusicGen generation failed.", exc_info=exc)

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
        # Load selected model (try local first to avoid downloads)
        model_id = get_musicgen_model_id(params.musicgen_model)
        revision = get_musicgen_model_revision(params.musicgen_model)
        try:
            # Revisions are pinned to immutable commit SHAs via MUSICGEN_MODELS.
            # Bandit B615 cannot resolve dynamic values, so we explicitly suppress
            # these calls after enforcing commit pinning above.
            processor = AutoProcessor.from_pretrained(  # nosec B615
                model_id,
                revision=revision,
                local_files_only=True,
            )
            model = MusicgenForConditionalGeneration.from_pretrained(  # nosec B615
                model_id,
                revision=revision,
                local_files_only=True,
            )
        except OSError:
            # Fallback to downloading if not found locally
            print(f"Model {model_id} not found locally. Downloading...")
            processor = AutoProcessor.from_pretrained(  # nosec B615
                model_id, revision=revision
            )
            model = MusicgenForConditionalGeneration.from_pretrained(  # nosec B615
                model_id,
                revision=revision,
            )

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
            "musicgen_model": params.musicgen_model,
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
        # Show MusicGen model if present
        if result["params"].get("musicgen_model"):
            model_key = result["params"]["musicgen_model"]
            model_info = MUSICGEN_MODELS.get(model_key, {})
            model_desc = model_info.get("description", model_key)
            lines.append(f"**AI Model:** {model_desc}")
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
    # Check if MLX server is reachable (loopback-pinned; sync caller, no thread).
    mlx_available = False
    if _MLX_HOST_IS_LOOPBACK:
        try:
            status_code, _ = _mlx_request("/api/models", 1.0)
            mlx_available = status_code == 200
        except Exception as exc:
            logger.debug("MLX capability probe failed: %s", exc)

    return {
        "midi": MIDI_AVAILABLE,
        "audio": TORCH_AVAILABLE or mlx_available,
        "musicgen": TORCH_AVAILABLE or mlx_available,
        "mlx": mlx_available,
        "stable_audio": mlx_available,
    }
