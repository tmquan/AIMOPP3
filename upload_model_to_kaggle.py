#!/usr/bin/env python3
"""
Upload Nemotron-Super-49B-v1.5 to Kaggle Models

Usage:
    1. First authenticate: python upload_model_to_kaggle.py --login
    2. Then upload: python upload_model_to_kaggle.py --upload

Note: The model is ~98GB, upload will take significant time.
"""

import argparse
import kagglehub
from pathlib import Path

# Configuration
MODEL_DIR = "/localhome/local-tranminhq/AIMOPP3/checkpoints/nemotron-super-49b-v1_5"

# Kaggle model handle format: <username>/<model-slug>/<framework>/<variation>
# You need to replace YOUR_KAGGLE_USERNAME with your actual username
KAGGLE_USERNAME = "tmquan"  # <-- CHANGE THIS!
MODEL_SLUG = "nemotron-super-49b"
FRAMEWORK = "transformers"  # or "pytorch" 
VARIATION = "v1_5"

def login():
    """Authenticate with Kaggle."""
    print("🔐 Kaggle Authentication")
    print("=" * 60)
    print("You will be prompted to enter your Kaggle credentials.")
    print("Get your API token from: https://www.kaggle.com/settings")
    print("=" * 60)
    kagglehub.login()
    print("\n✅ Successfully authenticated with Kaggle!")

def check_model_dir():
    """Verify model directory exists and show contents."""
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        print(f"❌ Model directory not found: {MODEL_DIR}")
        return False
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
    total_gb = total_size / (1024**3)
    
    print(f"📁 Model directory: {MODEL_DIR}")
    print(f"📊 Total size: {total_gb:.2f} GB")
    print(f"📄 Files:")
    for f in sorted(model_path.glob('*'))[:15]:
        if f.is_file():
            size_mb = f.stat().st_size / (1024**2)
            print(f"   {f.name}: {size_mb:.1f} MB")
    
    return True

def upload():
    """Upload model to Kaggle."""
    if KAGGLE_USERNAME == "YOUR_KAGGLE_USERNAME":
        print("❌ ERROR: Please edit this script and set KAGGLE_USERNAME to your Kaggle username!")
        print("   Edit line 20 in upload_model_to_kaggle.py")
        return
    
    handle = f"{KAGGLE_USERNAME}/{MODEL_SLUG}/{FRAMEWORK}/{VARIATION}"
    
    print("=" * 60)
    print("🚀 Uploading Nemotron-Super-49B-v1.5 to Kaggle")
    print("=" * 60)
    print(f"📁 Source: {MODEL_DIR}")
    print(f"🎯 Target: https://www.kaggle.com/models/{KAGGLE_USERNAME}/{MODEL_SLUG}")
    print(f"📋 Handle: {handle}")
    print("=" * 60)
    
    if not check_model_dir():
        return
    
    print("\n⚠️  WARNING: This will upload ~98GB. This may take several hours!")
    confirm = input("Continue? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Aborted.")
        return
    
    print("\n📤 Starting upload...")
    
    # Ignore patterns for files we don't need to upload
    ignore_patterns = [
        "*.md",
        "*.txt", 
        "*.png",
        ".git/",
        ".cache/",
        "__pycache__/",
    ]
    
    try:
        kagglehub.model_upload(
            handle,
            MODEL_DIR,
            license_name="Apache 2.0",
            version_notes="Llama-3.3-Nemotron-Super-49B-v1.5 - BF16 weights for vLLM inference",
            ignore_patterns=ignore_patterns
        )
        print("\n✅ Upload complete!")
        print(f"🔗 View your model at: https://www.kaggle.com/models/{KAGGLE_USERNAME}/{MODEL_SLUG}")
    except Exception as e:
        print(f"\n❌ Upload failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload model to Kaggle")
    parser.add_argument('--login', action='store_true', help='Authenticate with Kaggle')
    parser.add_argument('--upload', action='store_true', help='Upload the model')
    parser.add_argument('--check', action='store_true', help='Check model directory')
    parser.add_argument('--username', type=str, help='Your Kaggle username')
    
    args = parser.parse_args()
    
    if args.username:
        global KAGGLE_USERNAME
        KAGGLE_USERNAME = args.username
    
    if args.login:
        login()
    elif args.check:
        check_model_dir()
    elif args.upload:
        upload()
    else:
        parser.print_help()
        print("\n📋 Quick Start:")
        print("   1. python upload_model_to_kaggle.py --login")
        print("   2. python upload_model_to_kaggle.py --username YOUR_USERNAME --upload")

if __name__ == "__main__":
    main()

