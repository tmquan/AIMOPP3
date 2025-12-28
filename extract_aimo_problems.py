#!/usr/bin/env python3
"""
AIMO Problem Extractor and Embedder

Extracts problems and answers from AIMO 1, 2, and 3 datasets.
Embeds them using nvidia/llama-embed-nemotron-8b.
Saves to CSV with columns: problem, problem_embedding, answer, answer_embedding

Usage:
    python extract_aimo_problems.py
    python extract_aimo_problems.py --batch-size 8 --device cuda:0
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Path configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DATASETS_DIR = SCRIPT_DIR / "datasets"
OUTPUT_DIR = SCRIPT_DIR / "embeddings"

CHECKPOINTS_DIR = None


def parse_path_args():
    """Parse path arguments before heavy imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR))
    args, _ = parser.parse_known_args()
    return args


def setup_environment(checkpoints_dir: Path):
    """Setup environment variables."""
    global CHECKPOINTS_DIR
    
    CHECKPOINTS_DIR = checkpoints_dir
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)


_path_args = parse_path_args()
setup_environment(checkpoints_dir=Path(_path_args.checkpoints_dir))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_aimo_datasets() -> pd.DataFrame:
    """
    Load all AIMO 1, 2, and 3 datasets.
    
    Returns:
        DataFrame with columns: id, problem, answer, source, has_answer
    """
    all_data = []
    
    # ========== AIMO 1 ==========
    print("\n📂 Loading AIMO 1...")
    
    # Train set (has answers)
    aimo1_train = pd.read_csv(DATASETS_DIR / "aimo1" / "train.csv")
    for _, row in aimo1_train.iterrows():
        all_data.append({
            'id': f"aimo1_train_{row['id']}",
            'problem': str(row['problem']),
            'answer': str(row['answer']),
            'source': 'aimo1_train',
            'has_answer': True
        })
    print(f"   Train: {len(aimo1_train)} problems with answers")
    
    # Test set (no answers)
    aimo1_test = pd.read_csv(DATASETS_DIR / "aimo1" / "test.csv")
    for _, row in aimo1_test.iterrows():
        all_data.append({
            'id': f"aimo1_test_{row['id']}",
            'problem': str(row['problem']),
            'answer': 'UNANSWERED',
            'source': 'aimo1_test',
            'has_answer': False
        })
    print(f"   Test: {len(aimo1_test)} problems (unanswered)")
    
    # ========== AIMO 2 ==========
    print("\n📂 Loading AIMO 2...")
    
    # Reference set (has answers)
    aimo2_ref = pd.read_csv(DATASETS_DIR / "aimo2" / "reference.csv")
    for _, row in aimo2_ref.iterrows():
        all_data.append({
            'id': f"aimo2_ref_{row['id']}",
            'problem': str(row['problem']),
            'answer': str(row['answer']),
            'source': 'aimo2_reference',
            'has_answer': True
        })
    print(f"   Reference: {len(aimo2_ref)} problems with answers")
    
    # Test set (no answers)
    aimo2_test = pd.read_csv(DATASETS_DIR / "aimo2" / "test.csv")
    for _, row in aimo2_test.iterrows():
        all_data.append({
            'id': f"aimo2_test_{row['id']}",
            'problem': str(row['problem']),
            'answer': 'UNANSWERED',
            'source': 'aimo2_test',
            'has_answer': False
        })
    print(f"   Test: {len(aimo2_test)} problems (unanswered)")
    
    # ========== AIMO 3 ==========
    print("\n📂 Loading AIMO 3...")
    
    # Reference set (has answers)
    aimo3_ref = pd.read_csv(DATASETS_DIR / "aimo3" / "reference.csv")
    for _, row in aimo3_ref.iterrows():
        all_data.append({
            'id': f"aimo3_ref_{row['id']}",
            'problem': str(row['problem']),
            'answer': str(row['answer']),
            'source': 'aimo3_reference',
            'has_answer': True
        })
    print(f"   Reference: {len(aimo3_ref)} problems with answers")
    
    # Data.csv (no answers - practice problems)
    aimo3_data = pd.read_csv(DATASETS_DIR / "aimo3" / "data.csv")
    for _, row in aimo3_data.iterrows():
        all_data.append({
            'id': f"aimo3_data_{row['id']}",
            'problem': str(row['problem']),
            'answer': 'UNANSWERED',
            'source': 'aimo3_data',
            'has_answer': False
        })
    print(f"   Data: {len(aimo3_data)} problems (unanswered)")
    
    # Test set (no answers)
    aimo3_test = pd.read_csv(DATASETS_DIR / "aimo3" / "test.csv")
    for _, row in aimo3_test.iterrows():
        all_data.append({
            'id': f"aimo3_test_{row['id']}",
            'problem': str(row['problem']),
            'answer': 'UNANSWERED',
            'source': 'aimo3_test',
            'has_answer': False
        })
    print(f"   Test: {len(aimo3_test)} problems (unanswered)")
    
    df = pd.DataFrame(all_data)
    
    print(f"\n📊 Total: {len(df)} problems loaded")
    print(f"   With answers: {df['has_answer'].sum()}")
    print(f"   Unanswered: {(~df['has_answer']).sum()}")
    
    return df


class NemotronEmbedder:
    """
    Embedding extractor using nvidia/llama-embed-nemotron-8b.
    Uses last token pooling as recommended for Nemotron models.
    """
    
    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda",
        use_4bit: bool = False,
        max_length: int = 8192
    ):
        from transformers import AutoTokenizer, AutoModel
        
        # Use local checkpoint if available
        if model_path is None:
            local_path = CHECKPOINTS_DIR / "nemotron-embed-8b"
            if local_path.exists():
                model_path = str(local_path)
                print(f"🔄 Using local checkpoint: {model_path}")
            else:
                model_path = "nvidia/llama-embed-nemotron-8b"
                print(f"🔄 Using HuggingFace model: {model_path}")
        
        print(f"   Device: {device}")
        print(f"   4-bit quantization: {use_4bit}")
        
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=str(CHECKPOINTS_DIR),
            trust_remote_code=True
        )
        
        # Load model
        load_kwargs = {
            'cache_dir': str(CHECKPOINTS_DIR),
            'trust_remote_code': True,
            'torch_dtype': torch.float16
        }
        
        if use_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            load_kwargs['device_map'] = "auto"
        else:
            load_kwargs['device_map'] = device
        
        self.model = AutoModel.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
        
        print(f"✅ Model loaded successfully")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def embed_batch(
        self,
        texts: List[str],
        input_type: str = 'passage'
    ) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Uses the same approach as extract_embeddings_parallel_shards.py:
        - Adds 'query:' or 'passage:' prefix for Nemotron models
        - Uses mean pooling on the last hidden state
        - Normalizes embeddings to unit length
        
        Args:
            texts: List of input texts
            input_type: 'query' or 'passage' (default: 'passage' for documents)
        
        Returns:
            Array of normalized embeddings [batch_size, embedding_dim]
        """
        # Add instruction prefix for Nemotron model (matching parallel_shards approach)
        if 'nemotron' in self.tokenizer.name_or_path.lower():
            if input_type == 'query':
                formatted_texts = [f"query: {text}" for text in texts]
            else:
                formatted_texts = [f"passage: {text}" for text in texts]
        else:
            formatted_texts = texts
        
        # Tokenize with truncation for long texts
        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move inputs to device
        # When using device_map, get the device from the model's first parameter
        if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
            # Model is distributed - get device from first embedding layer
            target_device = next(self.model.parameters()).device
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        else:
            # Model is on a single device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass with mean pooling (matching parallel_shards approach)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling on the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
            # Normalize to unit length
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def embed_texts(
        self,
        texts: List[str],
        input_type: str = 'passage',
        batch_size: int = 8,
        desc: str = "Embedding"
    ) -> np.ndarray:
        """
        Embed a list of texts with progress bar.
        
        Args:
            texts: List of texts to embed
            input_type: 'query' or 'passage' (default: 'passage')
            batch_size: Batch size for processing
            desc: Description for progress bar
        
        Returns:
            Array of embeddings [num_texts, embedding_dim]
        """
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=desc):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_batch(batch, input_type)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings)


def embed_aimo_data(
    df: pd.DataFrame,
    embedder: NemotronEmbedder,
    batch_size: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Embed problems and answers from AIMO data.
    
    Uses the same approach as extract_embeddings_parallel_shards.py:
    - 'passage:' prefix for documents (problems and answers)
    - Mean pooling on last hidden state
    - L2 normalization
    
    Args:
        df: DataFrame with 'problem' and 'answer' columns
        embedder: NemotronEmbedder instance
        batch_size: Batch size for embedding
    
    Returns:
        Tuple of (problem_embeddings, answer_embeddings)
    """
    print("\n🔮 Embedding problems...")
    problem_embeddings = embedder.embed_texts(
        df['problem'].tolist(),
        input_type='passage',  # Problems are documents/passages
        batch_size=batch_size,
        desc="Problems"
    )
    
    print("\n🔮 Embedding answers...")
    # For answers, also use 'passage' type
    # For UNANSWERED, we still embed the placeholder text
    answer_embeddings = embedder.embed_texts(
        df['answer'].tolist(),
        input_type='passage',  # Answers are also documents
        batch_size=batch_size,
        desc="Answers"
    )
    
    return problem_embeddings, answer_embeddings


def save_to_csv(
    df: pd.DataFrame,
    problem_embeddings: np.ndarray,
    answer_embeddings: np.ndarray,
    output_path: Path
):
    """
    Save data with embeddings to CSV.
    
    Embeddings are stored as JSON-encoded lists.
    """
    print(f"\n💾 Saving to {output_path}...")
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'id': df['id'],
        'problem': df['problem'],
        'problem_embedding': [json.dumps(emb.tolist()) for emb in problem_embeddings],
        'answer': df['answer'],
        'answer_embedding': [json.dumps(emb.tolist()) for emb in answer_embeddings],
        'source': df['source'],
        'has_answer': df['has_answer']
    })
    
    output_df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(output_df)} records")
    
    # Also save embeddings as numpy files for efficient loading
    np_dir = output_path.parent / "numpy"
    np_dir.mkdir(exist_ok=True)
    
    np.save(np_dir / "problem_embeddings.npy", problem_embeddings)
    np.save(np_dir / "answer_embeddings.npy", answer_embeddings)
    df[['id', 'problem', 'answer', 'source', 'has_answer']].to_csv(
        np_dir / "metadata.csv", index=False
    )
    
    print(f"✅ Also saved numpy arrays to {np_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract and embed AIMO problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_aimo_problems.py
  python extract_aimo_problems.py --batch-size 16 --device cuda:0
  python extract_aimo_problems.py --4bit  # Use 4-bit quantization
        """
    )
    
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for embedding')
    parser.add_argument('--device', default='cuda:0', help='Device (cuda:0, cuda:1, etc.)')
    parser.add_argument('--4bit', action='store_true', dest='use_4bit', help='Use 4-bit quantization')
    parser.add_argument('--output', '-o', default=str(OUTPUT_DIR / "aimo_embeddings.csv"), help='Output CSV path')
    parser.add_argument('--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR))
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🧮 AIMO Problem Extractor and Embedder")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load all AIMO datasets
    df = load_aimo_datasets()
    
    # Initialize embedder
    print("\n" + "=" * 80)
    print("🔄 Initializing Embedding Model")
    print("=" * 80)
    
    embedder = NemotronEmbedder(
        device=args.device,
        use_4bit=args.use_4bit
    )
    
    # Embed problems and answers
    print("\n" + "=" * 80)
    print("🔮 Embedding Problems and Answers")
    print("=" * 80)
    
    problem_embeddings, answer_embeddings = embed_aimo_data(
        df, embedder, batch_size=args.batch_size
    )
    
    # Save results
    print("\n" + "=" * 80)
    print("💾 Saving Results")
    print("=" * 80)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_to_csv(df, problem_embeddings, answer_embeddings, output_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print(f"   Total problems: {len(df)}")
    print(f"   Problems with answers: {df['has_answer'].sum()}")
    print(f"   Unanswered problems: {(~df['has_answer']).sum()}")
    print(f"   Problem embedding shape: {problem_embeddings.shape}")
    print(f"   Answer embedding shape: {answer_embeddings.shape}")
    print(f"   Output CSV: {output_path}")
    print(f"   Numpy arrays: {output_path.parent / 'numpy'}")
    print("=" * 80)
    print(f"\n✅ Complete! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

