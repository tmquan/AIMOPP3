#!/usr/bin/env python3
"""
AIMO 3 Data Downloader

Downloads competition data and relevant mathematical datasets for the 
AI Mathematical Olympiad Progress Prize 3 competition.

Datasets:
1. AIMO 3 Competition Data (from Kaggle)
2. Nemotron Math datasets (for RAG/fine-tuning)
3. Mathematical problem datasets

Usage:
    python download_data.py --all
    python download_data.py --aimo3
    python download_data.py --nemotron-math
    python download_data.py --datasets-dir /path/to/datasets
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
DEFAULT_DATASETS_DIR = SCRIPT_DIR / "datasets"
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"

# Global variables
DATASETS_DIR = None
CHECKPOINTS_DIR = None


def parse_path_args():
    """Parse path arguments before importing heavy libraries."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--datasets-dir', type=str, default=str(DEFAULT_DATASETS_DIR),
        help=f'Directory for downloaded datasets (default: {DEFAULT_DATASETS_DIR})'
    )
    parser.add_argument(
        '--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help=f'Directory for model checkpoints/HF cache (default: {DEFAULT_CHECKPOINTS_DIR})'
    )
    args, _ = parser.parse_known_args()
    return args


def setup_environment(datasets_dir: Path, checkpoints_dir: Path):
    """Set up HuggingFace environment variables."""
    global DATASETS_DIR, CHECKPOINTS_DIR
    
    DATASETS_DIR = datasets_dir
    CHECKPOINTS_DIR = checkpoints_dir
    
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    
    os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)
    os.environ['HF_MODULES_CACHE'] = str(CHECKPOINTS_DIR / "modules")
    os.environ['HF_DATASETS_CACHE'] = str(DATASETS_DIR)


# Setup environment before imports
_path_args = parse_path_args()
setup_environment(
    datasets_dir=Path(_path_args.datasets_dir),
    checkpoints_dir=Path(_path_args.checkpoints_dir)
)


# AIMO Competition configurations
AIMO_COMPETITIONS = {
    "aimo1": {
        "name": "ai-mathematical-olympiad-prize",
        "display_name": "AIMO Progress Prize 1",
        "folder": "aimo1"
    },
    "aimo2": {
        "name": "ai-mathematical-olympiad-progress-prize-2",
        "display_name": "AIMO Progress Prize 2",
        "folder": "aimo2"
    },
    "aimo3": {
        "name": "ai-mathematical-olympiad-progress-prize-3",
        "display_name": "AIMO Progress Prize 3",
        "folder": "aimo3"
    }
}


def download_kaggle_competition(competition_key: str = None, download_all_aimo: bool = False):
    """Download AIMO competition data from Kaggle.
    
    Args:
        competition_key: One of 'aimo1', 'aimo2', 'aimo3', or None for aimo3 only
        download_all_aimo: If True, download all three AIMO competitions
    """
    import subprocess
    import zipfile
    
    # Check if kaggle CLI is available first
    result = subprocess.run(
        ["kaggle", "--version"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        print("❌ Kaggle CLI not found. Install with: pip install kaggle")
        print("   Also ensure ~/.config/kaggle/kaggle.json is configured")
        return None
    
    # Determine which competitions to download
    if download_all_aimo:
        competitions_to_download = list(AIMO_COMPETITIONS.keys())
    elif competition_key:
        competitions_to_download = [competition_key]
    else:
        competitions_to_download = ["aimo3"]  # Default
    
    results = {}
    
    for comp_key in competitions_to_download:
        if comp_key not in AIMO_COMPETITIONS:
            print(f"❌ Unknown competition key: {comp_key}")
            continue
            
        comp = AIMO_COMPETITIONS[comp_key]
        
        print("=" * 80)
        print(f"🔽 Downloading {comp['display_name']} from Kaggle...")
        print("=" * 80)
        
        try:
            output_dir = DATASETS_DIR / comp['folder']
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📂 Output directory: {output_dir}")
            
            # Download competition data
            subprocess.run([
                "kaggle", "competitions", "download",
                "-c", comp['name'],
                "-p", str(output_dir)
            ], check=True)
            
            # Unzip if needed
            for zip_file in output_dir.glob("*.zip"):
                print(f"📦 Extracting {zip_file.name}...")
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    zf.extractall(output_dir)
                zip_file.unlink()  # Remove zip after extraction
            
            print(f"\n✅ {comp['display_name']} downloaded to: {output_dir}")
            results[comp_key] = output_dir
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error downloading {comp['display_name']}: {e}")
            print("   Make sure you've accepted the competition rules on Kaggle:")
            print(f"   https://www.kaggle.com/competitions/{comp['name']}")
            results[comp_key] = None
        except Exception as e:
            print(f"❌ Error: {e}")
            results[comp_key] = None
    
    return results


def download_nemotron_math():
    """Download Nemotron Math datasets for mathematical reasoning."""
    print("=" * 80)
    print("🔽 Downloading Nemotron Math Datasets...")
    print("=" * 80)
    
    try:
        from datasets import load_dataset
        
        datasets_downloaded = {}
        
        # Nemotron Math v2
        print("\n📥 Downloading Nemotron-Math-v2...")
        try:
            dataset = load_dataset(
                "nvidia/Nemotron-Math-v2",
                cache_dir=str(DATASETS_DIR / "nemotron-math-v2")
            )
            datasets_downloaded['Nemotron-Math-v2'] = dataset
            print(f"   ✅ Splits: {list(dataset.keys())}")
            print(f"   ✅ Total samples: {sum(len(dataset[s]) for s in dataset.keys()):,}")
        except Exception as e:
            print(f"   ⚠️  Could not download Nemotron-Math-v2: {e}")
        
        # Nemotron Math Proofs
        print("\n📥 Downloading Nemotron-Math-Proofs-v1...")
        try:
            dataset = load_dataset(
                "nvidia/Nemotron-Math-Proofs-v1",
                cache_dir=str(DATASETS_DIR / "nemotron-math-proofs")
            )
            datasets_downloaded['Nemotron-Math-Proofs-v1'] = dataset
            print(f"   ✅ Splits: {list(dataset.keys())}")
            print(f"   ✅ Total samples: {sum(len(dataset[s]) for s in dataset.keys()):,}")
        except Exception as e:
            print(f"   ⚠️  Could not download Nemotron-Math-Proofs-v1: {e}")
        
        # Nemotron Competitive Programming (useful for olympiad-style problems)
        print("\n📥 Downloading Nemotron-Competitive-Programming-v1...")
        try:
            dataset = load_dataset(
                "nvidia/Nemotron-Competitive-Programming-v1",
                cache_dir=str(DATASETS_DIR / "nemotron-competitive-programming")
            )
            datasets_downloaded['Nemotron-Competitive-Programming-v1'] = dataset
            print(f"   ✅ Splits: {list(dataset.keys())}")
            print(f"   ✅ Total samples: {sum(len(dataset[s]) for s in dataset.keys()):,}")
        except Exception as e:
            print(f"   ⚠️  Could not download Nemotron-Competitive-Programming-v1: {e}")
        
        return datasets_downloaded
        
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        return None


def download_math_benchmarks():
    """Download mathematical benchmark datasets (MATH, GSM8K, etc.)."""
    print("=" * 80)
    print("🔽 Downloading Mathematical Benchmark Datasets...")
    print("=" * 80)
    
    try:
        from datasets import load_dataset
        
        datasets_downloaded = {}
        
        # GSM8K
        print("\n📥 Downloading GSM8K...")
        try:
            dataset = load_dataset(
                "openai/gsm8k",
                "main",
                cache_dir=str(DATASETS_DIR / "gsm8k")
            )
            datasets_downloaded['GSM8K'] = dataset
            print(f"   ✅ Splits: {list(dataset.keys())}")
        except Exception as e:
            print(f"   ⚠️  Could not download GSM8K: {e}")
        
        # MATH dataset
        print("\n📥 Downloading MATH dataset...")
        try:
            dataset = load_dataset(
                "lighteval/MATH",
                "all",
                cache_dir=str(DATASETS_DIR / "math-dataset"),
                trust_remote_code=True
            )
            datasets_downloaded['MATH'] = dataset
            print(f"   ✅ Splits: {list(dataset.keys())}")
        except Exception as e:
            print(f"   ⚠️  Could not download MATH: {e}")
        
        # Olympiad Bench (if available)
        print("\n📥 Attempting to download OlympiadBench...")
        try:
            dataset = load_dataset(
                "lmms-lab/OlympiadBench",
                cache_dir=str(DATASETS_DIR / "olympiad-bench"),
                trust_remote_code=True
            )
            datasets_downloaded['OlympiadBench'] = dataset
            print(f"   ✅ Splits: {list(dataset.keys())}")
        except Exception as e:
            print(f"   ⚠️  Could not download OlympiadBench: {e}")
        
        return datasets_downloaded
        
    except ImportError:
        print("❌ datasets library not found. Install with: pip install datasets")
        return None


def create_sample_data_csv():
    """Create a sample data.csv for testing the pipeline."""
    print("=" * 80)
    print("📝 Creating sample data.csv for testing...")
    print("=" * 80)
    
    import csv
    
    sample_problems = [
        {
            "id": 1,
            "problem": "Find all positive integers n such that n^2 + 1 is divisible by n + 1."
        },
        {
            "id": 2,
            "problem": "Let a, b, c be positive real numbers such that abc = 1. Prove that a^2 + b^2 + c^2 >= a + b + c."
        },
        {
            "id": 3,
            "problem": "In triangle ABC, let D be the foot of the altitude from A. If BD = 3, DC = 4, and AD = sqrt(12), find the area of triangle ABC."
        },
        {
            "id": 4,
            "problem": "How many ways can you arrange the letters of MATHEMATICS such that no two identical letters are adjacent?"
        },
        {
            "id": 5,
            "problem": "Find the remainder when 2^100 + 3^100 is divided by 5."
        }
    ]
    
    data_dir = DATASETS_DIR / "aimo3"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "data.csv"
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'problem'])
        writer.writeheader()
        writer.writerows(sample_problems)
    
    print(f"✅ Sample data.csv created at: {csv_path}")
    print(f"   Contains {len(sample_problems)} sample problems")
    
    return csv_path


def display_summary(downloads: dict):
    """Display summary of downloaded datasets."""
    print("\n" + "=" * 80)
    print("📋 DOWNLOAD SUMMARY")
    print("=" * 80)
    
    for name, result in downloads.items():
        if result:
            print(f"\n✅ {name}: Downloaded successfully")
            if hasattr(result, 'keys'):
                print(f"   Splits: {list(result.keys())}")
        else:
            print(f"\n❌ {name}: Failed or skipped")
    
    print("\n" + "=" * 80)
    print(f"📁 Datasets location: {DATASETS_DIR}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Download AIMO competition data and related datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything (all AIMO competitions + all datasets)
  python download_data.py --all
  
  # Download all three AIMO competitions (P1, P2, P3)
  python download_data.py --aimo-all
  
  # Download specific AIMO competition
  python download_data.py --aimo1
  python download_data.py --aimo2
  python download_data.py --aimo3
  
  # Download only Nemotron math datasets
  python download_data.py --nemotron-math
  
  # Download benchmark datasets (GSM8K, MATH)
  python download_data.py --benchmarks
  
  # Custom output directory
  python download_data.py --all --datasets-dir /data/aimo
        """
    )
    
    parser.add_argument(
        '--datasets-dir', type=str, default=str(DEFAULT_DATASETS_DIR),
        help='Directory for downloaded datasets'
    )
    parser.add_argument(
        '--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR),
        help='Directory for model checkpoints/HF cache'
    )
    # AIMO competition flags
    parser.add_argument('--aimo1', action='store_true', help='Download AIMO Progress Prize 1 data')
    parser.add_argument('--aimo2', action='store_true', help='Download AIMO Progress Prize 2 data')
    parser.add_argument('--aimo3', action='store_true', help='Download AIMO Progress Prize 3 data')
    parser.add_argument('--aimo-all', action='store_true', help='Download all AIMO competitions (P1, P2, P3)')
    # Other dataset flags
    parser.add_argument('--nemotron-math', action='store_true', help='Download Nemotron math datasets')
    parser.add_argument('--benchmarks', action='store_true', help='Download benchmark datasets (GSM8K, MATH)')
    parser.add_argument('--sample', action='store_true', help='Create sample data.csv for testing')
    parser.add_argument('--all', action='store_true', help='Download all available datasets (including all AIMO)')
    
    args = parser.parse_args()
    
    # Check if any specific AIMO flag is set
    any_aimo_flag = args.aimo1 or args.aimo2 or args.aimo3 or args.aimo_all
    any_dataset_flag = any_aimo_flag or args.nemotron_math or args.benchmarks or args.sample
    
    # If no specific flags, download all
    download_all = args.all or not any_dataset_flag
    
    print("=" * 80)
    print("🚀 AIMO Data Downloader")
    print("=" * 80)
    print(f"📁 Datasets directory: {DATASETS_DIR}")
    print(f"🤖 Checkpoints directory: {CHECKPOINTS_DIR}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    downloads = {}
    
    # Handle AIMO competition downloads
    if download_all or args.aimo_all:
        # Download all three AIMO competitions
        aimo_results = download_kaggle_competition(download_all_aimo=True)
        if aimo_results:
            for key, result in aimo_results.items():
                display_name = AIMO_COMPETITIONS[key]['display_name']
                downloads[display_name] = result
    else:
        # Download specific AIMO competitions
        if args.aimo1:
            result = download_kaggle_competition(competition_key="aimo1")
            if result:
                downloads['AIMO Progress Prize 1'] = result.get('aimo1')
        if args.aimo2:
            result = download_kaggle_competition(competition_key="aimo2")
            if result:
                downloads['AIMO Progress Prize 2'] = result.get('aimo2')
        if args.aimo3:
            result = download_kaggle_competition(competition_key="aimo3")
            if result:
                downloads['AIMO Progress Prize 3'] = result.get('aimo3')
    
    if download_all or args.nemotron_math:
        nemotron_datasets = download_nemotron_math()
        if nemotron_datasets:
            downloads.update(nemotron_datasets)
    
    if download_all or args.benchmarks:
        benchmark_datasets = download_math_benchmarks()
        if benchmark_datasets:
            downloads.update(benchmark_datasets)
    
    if download_all or args.sample:
        downloads['Sample Data'] = create_sample_data_csv()
    
    display_summary(downloads)
    
    print("\n✅ Data download complete!")
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

