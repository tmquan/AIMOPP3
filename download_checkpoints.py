#!/usr/bin/env python3
"""
AIMO 3 Model Checkpoint Downloader

Downloads model weights for Llama-3.3-Nemotron-Super-49B and related models
in various precisions (FP8, BF16, FP16) for the H100 deployment.

Models:
1. nvidia/Llama-3.3-Nemotron-Super-49B-v1 (Main reasoning model)
2. nvidia/llama-embed-nemotron-8b (Embedding model for RAG)

Usage:
    python download_checkpoints.py --all
    python download_checkpoints.py --49b-fp8
    python download_checkpoints.py --embed-8b
    python download_checkpoints.py --checkpoints-dir /path/to/checkpoints
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# =============================================================================
# DEFAULT PATH CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

CHECKPOINTS_DIR = None


def parse_path_args():
    """Parse path arguments before importing heavy libraries."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help=f'Directory for model checkpoints (default: {DEFAULT_CHECKPOINTS_DIR})'
    )
    args, _ = parser.parse_known_args()
    return args


def setup_environment(checkpoints_dir: Path):
    """Set up HuggingFace environment variables."""
    global CHECKPOINTS_DIR
    
    CHECKPOINTS_DIR = checkpoints_dir
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)
    os.environ['HF_MODULES_CACHE'] = str(CHECKPOINTS_DIR / "modules")


# Setup environment before imports
_path_args = parse_path_args()
setup_environment(checkpoints_dir=Path(_path_args.checkpoints_dir))

from huggingface_hub import snapshot_download, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


def check_disk_space(path: Path, required_gb: float) -> bool:
    """Check if there's enough disk space."""
    import shutil
    
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024 ** 3)
    
    print(f"💾 Disk space check:")
    print(f"   Required: {required_gb:.1f} GB")
    print(f"   Available: {free_gb:.1f} GB")
    
    if free_gb < required_gb:
        print(f"⚠️  Warning: May not have enough disk space!")
        return False
    return True


def download_nemotron_49b_fp8():
    """Download Nemotron-Super-49B in FP8 precision (optimized for H100)."""
    print("=" * 80)
    print("🔽 Downloading Llama-3.3-Nemotron-Super-49B-v1 (FP8)...")
    print("   This is optimized for H100 GPUs with ~49GB VRAM usage")
    print("=" * 80)
    
    model_id = "nvidia/Llama-3.3-Nemotron-Super-49B-v1"
    local_dir = CHECKPOINTS_DIR / "nemotron-49b-fp8"
    
    # Check disk space (~50GB required)
    check_disk_space(CHECKPOINTS_DIR, 60.0)
    
    try:
        print(f"\n📥 Downloading from: {model_id}")
        print(f"📂 Target directory: {local_dir}")
        
        # Download the model
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            # For FP8, we want specific files
            ignore_patterns=["*.md", "*.txt", "*.json.lock"]
        )
        
        print(f"\n✅ Download complete!")
        print(f"   Location: {local_dir}")
        
        # List downloaded files
        files = list(local_dir.glob("**/*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"   Total size: {total_size / (1024**3):.2f} GB")
        print(f"   Files: {len([f for f in files if f.is_file()])}")
        
        return local_dir
        
    except HfHubHTTPError as e:
        print(f"❌ Error downloading model: {e}")
        print("   You may need to:")
        print("   1. Login: huggingface-cli login")
        print("   2. Accept model license on HuggingFace")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def download_nemotron_49b_bf16():
    """Download Nemotron-Super-49B in BF16 precision."""
    print("=" * 80)
    print("🔽 Downloading Llama-3.3-Nemotron-Super-49B-v1 (BF16)...")
    print("   ⚠️  This requires ~98GB VRAM (needs tensor parallelism)")
    print("=" * 80)
    
    model_id = "nvidia/Llama-3.3-Nemotron-Super-49B-v1"
    local_dir = CHECKPOINTS_DIR / "nemotron-49b-bf16"
    
    # Check disk space (~100GB required)
    check_disk_space(CHECKPOINTS_DIR, 110.0)
    
    try:
        print(f"\n📥 Downloading from: {model_id}")
        print(f"📂 Target directory: {local_dir}")
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.md", "*.txt", "*.json.lock"]
        )
        
        print(f"\n✅ Download complete!")
        print(f"   Location: {local_dir}")
        
        return local_dir
        
    except HfHubHTTPError as e:
        print(f"❌ Error downloading model: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def download_embed_nemotron_8b():
    """Download llama-embed-nemotron-8b for embeddings."""
    print("=" * 80)
    print("🔽 Downloading llama-embed-nemotron-8b...")
    print("   This is the embedding model for RAG pipeline")
    print("=" * 80)
    
    model_id = "nvidia/llama-embed-nemotron-8b"
    local_dir = CHECKPOINTS_DIR / "nemotron-embed-8b"
    
    # Check disk space (~16GB required)
    check_disk_space(CHECKPOINTS_DIR, 20.0)
    
    try:
        print(f"\n📥 Downloading from: {model_id}")
        print(f"📂 Target directory: {local_dir}")
        
        snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.md", "*.txt", "*.json.lock"]
        )
        
        print(f"\n✅ Download complete!")
        print(f"   Location: {local_dir}")
        
        # List downloaded files
        files = list(local_dir.glob("**/*"))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"   Total size: {total_size / (1024**3):.2f} GB")
        print(f"   Files: {len([f for f in files if f.is_file()])}")
        
        return local_dir
        
    except HfHubHTTPError as e:
        print(f"❌ Error downloading model: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def download_vllm_wheels():
    """Download vLLM wheels for offline installation."""
    print("=" * 80)
    print("🔽 Downloading vLLM wheels for offline installation...")
    print("=" * 80)
    
    import subprocess
    
    wheels_dir = CHECKPOINTS_DIR / "wheels"
    wheels_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        packages = [
            "vllm",
            "openai",
            "transformers",
            "torch",
            "sentencepiece",
            "tiktoken",
            "bitsandbytes"
        ]
        
        for package in packages:
            print(f"\n📥 Downloading {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "download",
                package,
                "-d", str(wheels_dir),
                "--no-deps"  # Don't download dependencies to avoid duplicates
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ {package} downloaded")
            else:
                print(f"   ⚠️  Could not download {package}: {result.stderr[:200]}")
        
        print(f"\n✅ Wheels saved to: {wheels_dir}")
        return wheels_dir
        
    except Exception as e:
        print(f"❌ Error downloading wheels: {e}")
        return None


def display_summary(downloads: dict):
    """Display summary of downloaded checkpoints."""
    print("\n" + "=" * 80)
    print("📋 CHECKPOINT DOWNLOAD SUMMARY")
    print("=" * 80)
    
    for name, path in downloads.items():
        if path:
            print(f"\n✅ {name}:")
            print(f"   Location: {path}")
            if isinstance(path, Path) and path.exists():
                files = list(path.glob("**/*"))
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                print(f"   Size: {total_size / (1024**3):.2f} GB")
        else:
            print(f"\n❌ {name}: Failed or skipped")
    
    print("\n" + "=" * 80)
    print(f"📁 Checkpoints location: {CHECKPOINTS_DIR}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints for AIMO 3 competition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  python download_checkpoints.py --all
  
  # Download only the 49B FP8 model (recommended for H100)
  python download_checkpoints.py --49b-fp8
  
  # Download only the embedding model
  python download_checkpoints.py --embed-8b
  
  # Download vLLM wheels for offline use
  python download_checkpoints.py --wheels
  
  # Custom output directory
  python download_checkpoints.py --all --checkpoints-dir /data/checkpoints

Model Sizes:
  - Nemotron-49B-FP8:  ~50 GB (fits on single H100-80GB)
  - Nemotron-49B-BF16: ~98 GB (requires tensor parallelism)
  - Embed-8B:          ~16 GB
        """
    )
    
    parser.add_argument(
        '--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help='Directory for model checkpoints'
    )
    parser.add_argument('--49b-fp8', dest='fp8_49b', action='store_true', help='Download 49B FP8 model (recommended)')
    parser.add_argument('--49b-bf16', dest='bf16_49b', action='store_true', help='Download 49B BF16 model')
    parser.add_argument('--embed-8b', dest='embed_8b', action='store_true', help='Download 8B embedding model')
    parser.add_argument('--wheels', action='store_true', help='Download vLLM wheels')
    parser.add_argument('--all', action='store_true', help='Download all models')
    
    args = parser.parse_args()
    
    # Default to FP8 + embed if nothing specified
    if not (args.all or args.fp8_49b or args.bf16_49b or args.embed_8b or args.wheels):
        args.fp8_49b = True
        args.embed_8b = True
    
    print("=" * 80)
    print("🚀 AIMO 3 Checkpoint Downloader")
    print("=" * 80)
    print(f"📁 Checkpoints directory: {CHECKPOINTS_DIR}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    downloads = {}
    
    if args.all or args.fp8_49b:
        downloads['Nemotron-49B-FP8'] = download_nemotron_49b_fp8()
    
    if args.all or args.bf16_49b:
        downloads['Nemotron-49B-BF16'] = download_nemotron_49b_bf16()
    
    if args.all or args.embed_8b:
        downloads['Embed-Nemotron-8B'] = download_embed_nemotron_8b()
    
    if args.all or args.wheels:
        downloads['vLLM Wheels'] = download_vllm_wheels()
    
    display_summary(downloads)
    
    print("\n✅ Checkpoint download complete!")
    print(f"📅 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

