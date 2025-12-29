#!/usr/bin/env python3
"""
Model Management Script for AI Orchestrator
===========================================

Helps manage large local AI models to save disk space.
Features:
- Check for the recommended MLX model.
- Download it if missing.
- List installed models and their sizes.
- interactive cleanup of old models.

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

def get_cli_path():
    """Get the absolute path to huggingface-cli in the current venv."""
    # Assuming huggingface-cli is in the same bin dir as python
    cli_name = "huggingface-cli.exe" if os.name == "nt" else "huggingface-cli"
    cli_path = Path(sys.executable).parent / cli_name
    return str(cli_path)

def check_huggingface_cli():
    """Ensure huggingface-cli is installed."""
    cli_path = get_cli_path()
    if not os.path.exists(cli_path):
        print("❌ 'huggingface-cli' not found in virtual environment.")
        print("Please install it: pip install huggingface_hub[cli]")
        return False
    return True

def get_cache_size():
    """Get the size of the huggingface cache."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return "0 GB"

    total_size = 0
    for dirpath, _, filenames in os.walk(cache_dir):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return f"{total_size / (1024**3):.2f} GB"

def ensure_model_installed():
    """Check if recommended models are installed."""
    from huggingface_hub import try_to_load_from_cache

    print("\n🔍 Checking Local Models:")
    
    for name, repo_id in LOCAL_MODELS.items():
        print(f"\n--- {name} ---")
        cached = try_to_load_from_cache(repo_id=repo_id, filename="config.json")

        if cached:
            print(f"✅ Installed at: {os.path.dirname(cached)}")
        else:
            print(f"⚠️  Not found ({repo_id})")
            choice = input(f"   Download {name} now? (y/N): ").strip().lower()
            if choice == 'y':
                print("   🚀 Downloading...")
                try:
                    subprocess.run(  # noqa: S603
                        [get_cli_path(), "download", repo_id],
                        check=True
                    )
                    print("   ✅ Download complete!")
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ Download failed: {e}")
            else:
                print("   Skipping.")

def clean_cache():
    """Run the interactive cache cleanup."""
    if not check_huggingface_cli():
        return

    print("\n🧹 Launching Hugging Face Cache Cleaner...")
    print("   (Use arrow keys to select, Space to delete, Enter to confirm)")
    try:
        subprocess.run([get_cli_path(), "delete-cache"], check=False)  # noqa: S603
    except KeyboardInterrupt:
        print("\nCancelled.")

def main():
    print("=" * 50)
    print("🤖 AI Orchestrator Model Manager")
    print("=" * 50)

    print(f"Current Cache Size: {get_cache_size()}")

    # 1. Ensure we have the right model
    ensure_model_installed()

    # 2. Offer to clean up
    print("\n" + "-" * 50)
    print("Would you like to manage/delete old models to save space?")
    choice = input("Run cache cleaner? (y/N): ").strip().lower()

    if choice == 'y':
        clean_cache()

    print("\n✨ Done! You can now run the orchestrator with:")
    print("   python -m src.orchestrator 'Hello' --model mlx-llama8")

if __name__ == "__main__":
    main()
