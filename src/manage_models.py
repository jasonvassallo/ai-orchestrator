#!/usr/bin/env python3
"""
Model Management Script for AI Orchestrator
===========================================

Helps manage large local AI models to save disk space.
Features:
- Check for recommended MLX and MusicGen models.
- Download them if missing.
- List installed models and their sizes.
- Interactive cleanup of old models.

Usage:
    python -m src.manage_models
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Local models that require disk space
LOCAL_MODELS = {
    "MLX Llama 3.1 8B": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "MusicGen Small": "facebook/musicgen-small",
}


def get_cli_path() -> str:
    """Get the absolute path to the 'hf' CLI in the current venv."""
    # Prefer the modern 'hf' command
    hf_name = "hf.exe" if os.name == "nt" else "hf"
    hf_path = Path(sys.executable).parent / hf_name

    if hf_path.exists():
        return str(hf_path)

    # Fallback to old name
    cli_name = "huggingface-cli.exe" if os.name == "nt" else "huggingface-cli"
    return str(Path(sys.executable).parent / cli_name)


def check_huggingface_cli() -> bool:
    """Ensure huggingface-cli is installed and working."""
    cli_path = get_cli_path()
    if not os.path.exists(cli_path):
        return False

    try:
        # Check if it actually runs
        subprocess.run([cli_path, "--help"], capture_output=True, check=True)  # noqa: S603
        return True
    except (subprocess.CalledProcessError, Exception):
        return False


def get_cache_dirs() -> list[Path]:
    """Return all common cache directories for Hugging Face."""
    return [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path.home() / "Library" / "Caches" / "huggingface" / "hub",
    ]


def get_cache_size() -> str:
    """Get the total size of all detected huggingface caches."""
    total_size = 0
    for cache_dir in get_cache_dirs():
        if not cache_dir.exists():
            continue
        try:
            for dirpath, _, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
        except Exception as e:
            logger.debug(f"Error calculating cache size in {cache_dir}: {e}")

    return f"{total_size / (1024**3):.2f} GB"


def ensure_model_installed() -> None:
    """Check if recommended models are installed."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        print("❌ 'huggingface_hub' not found. Please run: pip install huggingface_hub")
        return

    print("\n🔍 Checking Local Models:")

    for name, repo_id in LOCAL_MODELS.items():
        print(f"\n--- {name} ---")

        # 1. Try standard API check
        cached = try_to_load_from_cache(repo_id=repo_id, filename="config.json")

        if cached:
            snapshot_path = Path(cached).parent
            print(f"✅ Installed at:\n   {snapshot_path}")
            continue

        # 2. Fallback: Manual folder search in all possible cache locations
        found_manually = False
        for cache_root in get_cache_dirs():
            alt_folder = f"models--{repo_id.replace('/', '--')}"
            full_path = cache_root / alt_folder

            if full_path.exists():
                # Find the newest snapshot if it exists
                snapshot_dir = full_path / "snapshots"
                display_path = full_path
                if snapshot_dir.exists():
                    snapshots = sorted(
                        snapshot_dir.iterdir(), key=os.path.getmtime, reverse=True
                    )
                    if snapshots:
                        display_path = snapshots[0]

                print(f"✅ Installed at:\n   {display_path}")
                found_manually = True
                break

        if found_manually:
            continue

        # 3. If not found anywhere
        print(f"⚠️  Not found in cache ({repo_id})")
        print("   Download now? (y/N): ", end="", flush=True)
        choice = input().strip().lower()

        if choice == "y":
            print("   🚀 Downloading... (This may take a while)")
            try:
                subprocess.run(  # noqa: S603
                    [get_cli_path(), "download", repo_id], check=True
                )
                print("   ✅ Download complete!")
            except subprocess.CalledProcessError as e:
                print(f"   ❌ Download failed: {e}")
            except Exception as e:
                print(f"   ❌ Error during download: {e}")
        else:
            print("   Skipping.")


def clean_cache() -> None:
    """Run the interactive cache cleanup for ALL detected cache locations."""
    print("\n🧹 Launching Hugging Face Cache Cleaner...")

    if not check_huggingface_cli():
        print("❌ 'hf' CLI is missing or broken.")
        print('Please run: pip install "huggingface_hub[cli]"')
        return

    print("   (Use arrow keys to select, Space to delete, Enter to confirm)")

    # Iterate over all known cache directories (e.g., ~/.cache AND ~/Library/Caches)
    for hub_path in get_cache_dirs():
        if not hub_path.exists():
            continue

        # HF_HOME should be the parent of 'hub' (e.g. ~/.cache/huggingface)
        hf_home = hub_path.parent

        print(f"\n📂 Scanning cache at: {hf_home}")
        try:
            # Run cache delete pointing to this specific home directory
            env = os.environ.copy()
            env["HF_HOME"] = str(hf_home)

            subprocess.run([get_cli_path(), "cache", "delete"], check=False, env=env)  # noqa: S603

        except KeyboardInterrupt:
            print("\nCancelled.")
            return
        except Exception as e:
            print(f"❌ An unexpected error occurred scanning {hf_home}: {e}")


def main() -> None:
    print("=" * 50)
    print("🤖 AI Orchestrator Model Manager")
    print("=" * 50)

    print(f"Current Cache Size: {get_cache_size()}")

    # 1. Ensure we have the right model
    ensure_model_installed()

    # 2. Offer to clean up
    print("\n" + "-" * 50)
    print("Would you like to manage/delete old models to save space?")
    print("Run cache cleaner? (y/N): ", end="", flush=True)
    try:
        choice = input().strip().lower()
        if choice == "y":
            clean_cache()
    except EOFError:
        pass

    print("\n✨ Done! You can now run the orchestrator with:")
    print("   python -m src.orchestrator 'Hello' --model mlx-llama8")


if __name__ == "__main__":
    main()
