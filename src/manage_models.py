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
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Local models that require disk space
LOCAL_MODELS = {
    "MLX Llama 3.1 8B": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "MusicGen Small": "facebook/musicgen-small",
}

def get_cli_path() -> str:
    """Get the absolute path to huggingface-cli in the current venv."""
    cli_name = "huggingface-cli.exe" if os.name == "nt" else "huggingface-cli"
    cli_path = Path(sys.executable).parent / cli_name
    return str(cli_path)

def check_huggingface_cli() -> bool:
    """Ensure huggingface-cli is installed and working."""
    cli_path = get_cli_path()
    if not os.path.exists(cli_path):
        return False
    
    try:
        # Check if it actually runs (verifies dependencies like huggingface_hub[cli])
        subprocess.run([cli_path, "--help"], capture_output=True, check=True)  # noqa: S603
        return True
    except (subprocess.CalledProcessError, Exception):
        return False

def get_cache_size() -> str:
    """Get the size of the huggingface cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return "0 GB"

    total_size = 0
    try:
        for dirpath, _, filenames in os.walk(cache_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception as e:
        logger.debug(f"Error calculating cache size: {e}")

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
        
        # Check cache
        cached = try_to_load_from_cache(repo_id=repo_id, filename="config.json")

        if cached:
            print("✅ Installed at:")
            snapshot_path = Path(cached).parent
            print(f"   {snapshot_path}")
        else:
            print(f"⚠️  Not found in cache ({repo_id})")
            
            # Additional check for manual folder naming (common with MusicGen)
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
            alt_folder = f"models--{repo_id.replace('/', '--')}"
            if (cache_root / alt_folder).exists():
                print(f"   ✅ Found model folder but config check failed: {alt_folder}")
                print("   (The model is likely there but indexed differently)")
                continue

            print("   Download now? (y/N): ", end="", flush=True)
            choice = input().strip().lower()
            
            if choice == 'y':
                print("   🚀 Downloading... (This may take a while)")
                try:
                    subprocess.run(  # noqa: S603
                        [get_cli_path(), "download", repo_id],
                        check=True
                    )
                    print("   ✅ Download complete!")
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ Download failed: {e}")
                except Exception as e:
                    print(f"   ❌ Error during download: {e}")
            else:
                print("   Skipping.")

def clean_cache() -> None:
    """Run the interactive cache cleanup."""
    print("\n🧹 Launching Hugging Face Cache Cleaner...")
    
    if not check_huggingface_cli():
        print("❌ 'huggingface-cli' is missing or broken.")
        print("Please run: pip install \"huggingface_hub[cli]\"")
        print("Note: The TUI version requires 'urwid'.")
        return

    print("   (Use arrow keys to select, Space to delete, Enter to confirm)")
    try:
        # Use the modern 'hf cache delete' if available, otherwise fallback
        subprocess.run([get_cli_path(), "delete-cache"], check=False) # noqa: S603
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        print("Tip: Try running 'huggingface-cli delete-cache --disable-tui' if the TUI fails.")

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
        if choice == 'y':
            clean_cache()
    except EOFError:
        pass

    print("\n✨ Done! You can now run the orchestrator with:")
    print("   python -m src.orchestrator 'Hello' --model mlx-llama8")

if __name__ == "__main__":
    main()