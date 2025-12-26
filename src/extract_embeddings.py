#!/usr/bin/env python3
"""
Embedding Extractor for AIMO 3

Extracts embeddings from mathematical problems using nvidia/llama-embed-nemotron-8b.
Supports parallel processing across multiple GPUs.

Usage:
    python extract_embeddings.py --input data.csv --output embeddings.parquet
    python extract_embeddings.py --input data.csv --num-gpus 4 --batch-size 32
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import json

# Path configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DEFAULT_EMBEDDINGS_DIR = PROJECT_DIR / "embeddings"

CHECKPOINTS_DIR = None
EMBEDDINGS_DIR = None


def parse_path_args():
    """Parse path arguments before heavy imports."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR))
    parser.add_argument('--embeddings-dir', type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    args, _ = parser.parse_known_args()
    return args


def setup_environment(checkpoints_dir: Path, embeddings_dir: Path):
    """Setup environment variables."""
    global CHECKPOINTS_DIR, EMBEDDINGS_DIR
    
    CHECKPOINTS_DIR = checkpoints_dir
    EMBEDDINGS_DIR = embeddings_dir
    
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    os.environ['HF_HOME'] = str(CHECKPOINTS_DIR)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(CHECKPOINTS_DIR)


_path_args = parse_path_args()
setup_environment(
    checkpoints_dir=Path(_path_args.checkpoints_dir),
    embeddings_dir=Path(_path_args.embeddings_dir)
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Simple dataset for text batching."""
    
    def __init__(self, texts: List[str], ids: List[int]):
        self.texts = texts
        self.ids = ids
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'text': self.texts[idx],
            'id': self.ids[idx]
        }


class EmbeddingExtractor:
    """
    Embedding extractor using nvidia/llama-embed-nemotron-8b.
    
    Features:
    - Last token pooling (as recommended for Nemotron)
    - Instruction-based embedding for queries vs documents
    - Memory-efficient batch processing
    - Multi-GPU support
    """
    
    def __init__(
        self,
        model_path: str = "nvidia/llama-embed-nemotron-8b",
        device: str = "cuda",
        use_4bit: bool = False,
        max_length: int = 8192
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to model or HuggingFace model ID
            device: Device to load model on
            use_4bit: Use 4-bit quantization to save VRAM
            max_length: Maximum sequence length
        """
        from transformers import AutoTokenizer, AutoModel
        
        print(f"🔄 Initializing Embedding Model...")
        print(f"   Model: {model_path}")
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
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        
        print(f"✅ Model loaded successfully")
        print(f"   Embedding dimension: {self.embedding_dim}")
    
    def _last_token_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract embeddings using last token pooling.
        This is the recommended method for Nemotron embedding models.
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]
    
    def get_embedding(
        self,
        text: str,
        is_query: bool = True,
        instruction: str = None
    ) -> np.ndarray:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text
            is_query: Whether this is a query (vs document)
            instruction: Custom instruction (optional)
        
        Returns:
            Normalized embedding vector
        """
        embeddings = self.get_embeddings_batch([text], is_query, instruction)
        return embeddings[0]
    
    def get_embeddings_batch(
        self,
        texts: List[str],
        is_query: bool = True,
        instruction: str = None
    ) -> np.ndarray:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of input texts
            is_query: Whether these are queries (vs documents)
            instruction: Custom instruction (optional)
        
        Returns:
            Array of normalized embeddings [batch_size, embedding_dim]
        """
        # Format texts with instruction prefix
        if instruction is None:
            if is_query:
                instruction = "Retrieve math problems similar to the query."
            else:
                instruction = ""  # Documents don't need instruction
        
        if instruction:
            formatted_texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        else:
            formatted_texts = texts
        
        # Tokenize
        inputs = self.tokenizer(
            formatted_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Move to device
        if not hasattr(self.model, 'hf_device_map'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._last_token_pool(
                outputs.last_hidden_state,
                inputs['attention_mask']
            )
            
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    def extract_from_csv(
        self,
        csv_path: str,
        output_path: str,
        text_column: str = 'problem',
        id_column: str = 'id',
        batch_size: int = 16,
        is_query: bool = False
    ) -> Dict:
        """
        Extract embeddings from a CSV file.
        
        Args:
            csv_path: Path to input CSV
            output_path: Path to output file (parquet or json)
            text_column: Column containing text to embed
            id_column: Column containing record IDs
            batch_size: Batch size for processing
            is_query: Whether texts are queries (vs documents)
        
        Returns:
            Dictionary with extraction statistics
        """
        import pandas as pd
        from tqdm import tqdm
        
        print(f"\n📂 Loading data from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        print(f"   Records: {len(df):,}")
        print(f"   Text column: {text_column}")
        print(f"   ID column: {id_column}")
        
        texts = df[text_column].astype(str).tolist()
        ids = df[id_column].tolist()
        
        # Extract embeddings in batches
        all_embeddings = []
        
        print(f"\n🔮 Extracting embeddings (batch_size={batch_size})...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Batches"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.get_embeddings_batch(batch_texts, is_query=is_query)
            all_embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(all_embeddings)
        
        print(f"\n✅ Extraction complete!")
        print(f"   Shape: {embeddings.shape}")
        
        # Save results
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            # Save as parquet
            result_df = pd.DataFrame({
                id_column: ids,
                'embeddings': embeddings.tolist()
            })
            result_df.to_parquet(output_path, index=False)
        elif output_path.suffix == '.npy':
            # Save embeddings as numpy array
            np.save(output_path, embeddings)
            # Also save IDs
            np.save(output_path.with_suffix('.ids.npy'), np.array(ids))
        else:
            # Save as JSON
            result = {
                'metadata': {
                    'model': 'nvidia/llama-embed-nemotron-8b',
                    'embedding_dim': self.embedding_dim,
                    'num_samples': len(texts),
                    'extraction_time': datetime.now().isoformat()
                },
                'embeddings': [
                    {
                        'id': ids[i],
                        'embedding': embeddings[i].tolist()
                    }
                    for i in range(len(ids))
                ]
            }
            with open(output_path, 'w') as f:
                json.dump(result, f)
        
        print(f"💾 Saved to: {output_path}")
        
        return {
            'num_samples': len(texts),
            'embedding_dim': self.embedding_dim,
            'output_path': str(output_path)
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from text data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from CSV
  python extract_embeddings.py --input data.csv --output embeddings.parquet
  
  # With custom columns
  python extract_embeddings.py --input problems.csv --text-column question --id-column qid
  
  # Use 4-bit quantization
  python extract_embeddings.py --input data.csv --output emb.parquet --4bit
  
  # Custom batch size
  python extract_embeddings.py --input data.csv --output emb.parquet --batch-size 32
        """
    )
    
    parser.add_argument('--input', '-i', required=True, help='Input CSV file')
    parser.add_argument('--output', '-o', required=True, help='Output file (parquet, json, or npy)')
    parser.add_argument('--text-column', default='problem', help='Column with text to embed')
    parser.add_argument('--id-column', default='id', help='Column with record IDs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--model', default='nvidia/llama-embed-nemotron-8b', help='Model to use')
    parser.add_argument('--4bit', action='store_true', dest='use_4bit', help='Use 4-bit quantization')
    parser.add_argument('--max-length', type=int, default=8192, help='Max sequence length')
    parser.add_argument('--device', default='cuda:0', help='Device (cuda:0, cuda:1, etc.)')
    parser.add_argument('--checkpoints-dir', type=str, default=str(DEFAULT_CHECKPOINTS_DIR))
    parser.add_argument('--embeddings-dir', type=str, default=str(DEFAULT_EMBEDDINGS_DIR))
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🔮 AIMO 3 Embedding Extractor")
    print("=" * 80)
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize extractor
    extractor = EmbeddingExtractor(
        model_path=args.model,
        device=args.device,
        use_4bit=args.use_4bit,
        max_length=args.max_length
    )
    
    # Extract embeddings
    stats = extractor.extract_from_csv(
        csv_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        id_column=args.id_column,
        batch_size=args.batch_size
    )
    
    print("\n" + "=" * 80)
    print("📋 EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"   Samples: {stats['num_samples']:,}")
    print(f"   Embedding dim: {stats['embedding_dim']}")
    print(f"   Output: {stats['output_path']}")
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

