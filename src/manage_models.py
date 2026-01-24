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
    python -m src.manage_models --yes
    python -m src.manage_models --yes --no-clean
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Local models that require disk space
LOCAL_MODELS = {
    "MLX Qwen3 4B": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "MLX Qwen 2.5 Coder 14B": "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
    "MLX Llama 3.2 11B Vision": "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit",
    "MLX Ministral 14B Reasoning": "mlx-community/Ministral-3-14B-Reasoning-2512-6bit",
    "MusicGen Small": "facebook/musicgen-small",
}

DOWNLOAD_RETRIES = 3
DOWNLOAD_BACKOFF_SECONDS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage local AI model cache.")
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Assume yes for downloads and cache cleanup.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip cache cleanup even when --yes is set.",
    )
    return parser.parse_args()


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


def enable_hf_transfer_if_available() -> None:
    if os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"):
        return

    try:
        import importlib.util
    except Exception:
        return

    if importlib.util.find_spec("hf_transfer") is None:
        return

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def download_with_retry(repo_id: str) -> bool:
    enable_hf_transfer_if_available()
    cli_path = get_cli_path()

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            subprocess.run([cli_path, "download", repo_id], check=True)  # noqa: S603
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Download failed (attempt {attempt}/{DOWNLOAD_RETRIES}): {e}")
            if attempt >= DOWNLOAD_RETRIES:
                return False
            delay = DOWNLOAD_BACKOFF_SECONDS * attempt
            print(f"   🔁 Retrying in {delay:.0f}s...")
            time.sleep(delay)
        except Exception as e:
            print(f"   ❌ Error during download: {e}")
            return False

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


def get_cache_command() -> str | None:
    cli_path = get_cli_path()
    try:
        result = subprocess.run(  # noqa: S603
            [cli_path, "cache", "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return None

    help_text = (result.stdout + result.stderr).lower()
    if "delete" in help_text:
        return "delete"
    if "prune" in help_text:
        return "prune"
    return None


def ensure_model_installed(auto_yes: bool) -> None:
    """Check if recommended models are installed."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        print("❌ 'huggingface_hub' not found. Please run: pip install huggingface_hub")
        return

    print("\n🔍 Checking Local Models:")

    def _find_snapshot_with_weights(repo_id: str) -> Path | None:
        weight_files = ("model.safetensors", "model.safetensors.index.json")
        for filename in weight_files:
            cached = try_to_load_from_cache(repo_id=repo_id, filename=filename)
            if cached:
                return Path(cached).parent

        for cache_root in get_cache_dirs():
            alt_folder = f"models--{repo_id.replace('/', '--')}"
            full_path = cache_root / alt_folder
            snapshot_dir = full_path / "snapshots"
            if not snapshot_dir.exists():
                continue
            snapshots = sorted(
                snapshot_dir.iterdir(), key=os.path.getmtime, reverse=True
            )
            for snap in snapshots:
                if list(snap.glob("*.safetensors")):
                    return snap
                if (snap / "model.safetensors.index.json").exists():
                    return snap

        return None

    for name, repo_id in LOCAL_MODELS.items():
        print(f"\n--- {name} ---")

        # 1. Check for actual weight files in cache
        snapshot_path = _find_snapshot_with_weights(repo_id)
        if snapshot_path:
            print(f"✅ Installed at:\n   {snapshot_path}")
            continue

        # 2. Fallback: Manual folder search for metadata-only cache
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

                print("⚠️  Cache metadata found but no weights.")
                print(f"   Cache path:\n   {display_path}")
                found_manually = True
                break

        if not found_manually:
            # 3. If not found anywhere
            print(f"⚠️  Not found in cache ({repo_id})")
        if auto_yes:
            choice = "y"
        else:
            print("   Download now? (y/N): ", end="", flush=True)
            try:
                choice = input().strip().lower()
            except EOFError:
                choice = ""

        if choice == "y":
            print("   🚀 Downloading... (This may take a while)")
            if download_with_retry(repo_id):
                print("   ✅ Download complete!")
            else:
                print("   ❌ Download failed after multiple attempts.")
        else:
            print("   Skipping.")


def clean_cache(auto_yes: bool) -> None:
    """Run the interactive cache cleanup for ALL detected cache locations."""
    print("\n🧹 Launching Hugging Face Cache Cleaner...")

    if not check_huggingface_cli():
        print("❌ 'hf' CLI is missing or broken.")
        print('Please run: pip install "huggingface_hub[cli]"')
        return

    cache_command = get_cache_command()
    if cache_command is None:
        print("❌ This version of 'hf' does not support cache cleanup commands.")
        print("Try `hf cache ls` and `hf cache rm <repo_id>` manually.")
        return

    if cache_command == "delete":
        print("   (Use arrow keys to select, Space to delete, Enter to confirm)")
    else:
        print("   Using 'hf cache prune' to remove detached revisions.")

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
            command = [get_cli_path(), "cache", cache_command]
            if cache_command == "prune" and auto_yes:
                command.append("--yes")
            subprocess.run(command, check=False, env=env)  # noqa: S603

        except KeyboardInterrupt:
            print("\nCancelled.")
            return
        except Exception as e:
            print(f"❌ An unexpected error occurred scanning {hf_home}: {e}")


def main() -> None:
    args = parse_args()
    print("=" * 50)
    print("🤖 AI Orchestrator Model Manager")
    print("=" * 50)

    print(f"Current Cache Size: {get_cache_size()}")

    # 1. Ensure we have the right model
    ensure_model_installed(args.yes)

    # 2. Offer to clean up
    if not args.no_clean:
        if args.yes:
            clean_cache(args.yes)
        else:
            print("\n" + "-" * 50)
            print("Would you like to manage/delete old models to save space?")
            print("Run cache cleaner? (y/N): ", end="", flush=True)
            try:
                choice = input().strip().lower()
                if choice == "y":
                    clean_cache(args.yes)
            except EOFError:
                pass

    print("\n✨ Done! You can now run the orchestrator with:")
    print("   python -m src.orchestrator 'Hello' --model mlx-llama-vision-11b")


if __name__ == "__main__":
    main()
