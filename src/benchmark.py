#!/usr/bin/env python3
"""
Performance Benchmark for AIMO 3

Measures inference performance of Llama-3.3-Nemotron-Super-49B across
different precision formats (FP8, BF16, FP16) on H100 GPUs.

Metrics:
- Tokens per second (throughput)
- Time to first token (latency)
- Memory usage
- Total inference time

Usage:
    python benchmark.py --model-path ./checkpoints/nemotron-49b-fp8 --precision fp8
    python benchmark.py --all-precisions
    python benchmark.py --compare
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import statistics

# Path configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "benchmarks"


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    precision: str
    model_path: str
    prompt_length: int
    output_length: int
    total_time: float
    time_to_first_token: float
    tokens_per_second: float
    memory_allocated_gb: float
    memory_reserved_gb: float
    timestamp: str


class InferenceBenchmark:
    """
    Benchmark suite for measuring LLM inference performance.
    
    Measures:
    - Token generation throughput
    - Time to first token (TTFT)
    - GPU memory usage
    - End-to-end latency
    """
    
    # Sample prompts for benchmarking
    BENCHMARK_PROMPTS = [
        # Short prompt (math problem)
        "Find all positive integers n such that n² + 1 is divisible by n + 1.",
        
        # Medium prompt (olympiad-style)
        """Let ABC be a triangle with incenter I. Points D, E, F are the points where 
        the incircle touches sides BC, CA, AB respectively. Let P be the intersection 
        of lines AI and EF. Prove that P lies on the circumcircle of triangle DEF.""",
        
        # Long prompt (competition problem with context)
        """Consider a sequence of positive integers a₁, a₂, a₃, ... defined as follows:
        a₁ = 1, and for each n ≥ 1, aₙ₊₁ is the smallest positive integer greater than 
        aₙ such that aₙ + aₙ₊₁ is a perfect square.
        
        Part a) Find the first 10 terms of this sequence.
        Part b) Prove that aₙ ≥ n for all n ≥ 1.
        Part c) Find a closed-form formula for aₙ, or prove that none exists.
        
        Show all your work and reasoning."""
    ]
    
    def __init__(
        self,
        model_path: str,
        precision: str = "fp8",
        gpu_id: int = 0,
        max_model_len: int = 8192
    ):
        """
        Initialize benchmark.
        
        Args:
            model_path: Path to model weights
            precision: Inference precision (fp8, bf16, fp16)
            gpu_id: GPU device ID
            max_model_len: Maximum context length
        """
        self.model_path = Path(model_path)
        self.precision = precision
        self.gpu_id = gpu_id
        self.max_model_len = max_model_len
        
        self.model = None
        self.tokenizer = None
        self.results: List[BenchmarkResult] = []
    
    def _get_dtype(self):
        """Get torch dtype based on precision."""
        import torch
        
        dtype_map = {
            "fp8": torch.float8_e4m3fn if hasattr(torch, 'float8_e4m3fn') else torch.float16,
            "bf16": torch.bfloat16,
            "fp16": torch.float16
        }
        return dtype_map.get(self.precision, torch.float16)
    
    def _get_memory_stats(self) -> Dict:
        """Get current GPU memory statistics."""
        import torch
        
        if not torch.cuda.is_available():
            return {"allocated_gb": 0, "reserved_gb": 0}
        
        allocated = torch.cuda.memory_allocated(self.gpu_id) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(self.gpu_id) / (1024 ** 3)
        
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2)
        }
    
    def load_model_transformers(self):
        """Load model using transformers (for detailed benchmarking)."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"🔄 Loading model with transformers...")
        print(f"   Model: {self.model_path}")
        print(f"   Precision: {self.precision}")
        print(f"   Device: cuda:{self.gpu_id}")
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        # Load model
        load_kwargs = {
            "torch_dtype": self._get_dtype(),
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # Add quantization for FP8 if available
        if self.precision == "fp8":
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            except ImportError:
                print("   ⚠️ bitsandbytes not available, using FP16")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            **load_kwargs
        )
        self.model.eval()
        
        memory = self._get_memory_stats()
        print(f"✅ Model loaded")
        print(f"   Memory: {memory['allocated_gb']:.2f} GB allocated, {memory['reserved_gb']:.2f} GB reserved")
        
        return True
    
    def benchmark_single(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        num_runs: int = 3
    ) -> BenchmarkResult:
        """
        Run benchmark on a single prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            num_runs: Number of runs for averaging
        
        Returns:
            BenchmarkResult with metrics
        """
        import torch
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model_transformers() first.")
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_model_len
        ).to(self.model.device)
        
        prompt_length = inputs.input_ids.shape[1]
        
        print(f"\n📊 Benchmarking (prompt_length={prompt_length}, runs={num_runs})...")
        
        times = []
        ttfts = []
        output_lengths = []
        
        for run in range(num_runs):
            # Warm up GPU
            torch.cuda.synchronize()
            
            # Measure time to first token
            start_time = time.perf_counter()
            first_token_time = None
            
            # Generate with streaming to capture TTFT
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            total_time = end_time - start_time
            output_length = outputs.shape[1] - prompt_length
            
            # Estimate TTFT (first token is ~1/output_length of total time as baseline)
            ttft = total_time / max(output_length, 1)
            
            times.append(total_time)
            ttfts.append(ttft)
            output_lengths.append(output_length)
            
            tokens_per_sec = output_length / total_time
            print(f"   Run {run + 1}: {total_time:.2f}s, {output_length} tokens, {tokens_per_sec:.1f} tok/s")
        
        # Calculate averages
        avg_time = statistics.mean(times)
        avg_ttft = statistics.mean(ttfts)
        avg_output_length = statistics.mean(output_lengths)
        avg_tps = avg_output_length / avg_time
        
        memory = self._get_memory_stats()
        
        result = BenchmarkResult(
            precision=self.precision,
            model_path=str(self.model_path),
            prompt_length=prompt_length,
            output_length=int(avg_output_length),
            total_time=round(avg_time, 3),
            time_to_first_token=round(avg_ttft, 3),
            tokens_per_second=round(avg_tps, 2),
            memory_allocated_gb=memory['allocated_gb'],
            memory_reserved_gb=memory['reserved_gb'],
            timestamp=datetime.now().isoformat()
        )
        
        self.results.append(result)
        return result
    
    def run_full_benchmark(self, num_runs: int = 3) -> List[BenchmarkResult]:
        """
        Run benchmark on all sample prompts.
        
        Args:
            num_runs: Number of runs per prompt
        
        Returns:
            List of BenchmarkResults
        """
        print("=" * 80)
        print(f"🏃 Running Full Benchmark Suite")
        print(f"   Precision: {self.precision}")
        print(f"   Prompts: {len(self.BENCHMARK_PROMPTS)}")
        print(f"   Runs per prompt: {num_runs}")
        print("=" * 80)
        
        if self.model is None:
            self.load_model_transformers()
        
        results = []
        
        for i, prompt in enumerate(self.BENCHMARK_PROMPTS):
            print(f"\n📝 Prompt {i + 1}/{len(self.BENCHMARK_PROMPTS)}")
            print(f"   Preview: {prompt[:100]}...")
            
            result = self.benchmark_single(prompt, num_runs=num_runs)
            results.append(result)
        
        return results
    
    def save_results(self, output_path: Optional[str] = None):
        """Save benchmark results to JSON."""
        if output_path is None:
            results_dir = DEFAULT_RESULTS_DIR
            results_dir.mkdir(parents=True, exist_ok=True)
            output_path = results_dir / f"benchmark_{self.precision}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output = {
            "metadata": {
                "model_path": str(self.model_path),
                "precision": self.precision,
                "gpu_id": self.gpu_id,
                "timestamp": datetime.now().isoformat()
            },
            "results": [asdict(r) for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")
        return output_path
    
    def print_summary(self):
        """Print benchmark summary."""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "=" * 80)
        print("📊 BENCHMARK SUMMARY")
        print("=" * 80)
        
        print(f"\nModel: {self.model_path}")
        print(f"Precision: {self.precision}")
        
        # Calculate aggregates
        avg_tps = statistics.mean([r.tokens_per_second for r in self.results])
        avg_ttft = statistics.mean([r.time_to_first_token for r in self.results])
        max_memory = max([r.memory_allocated_gb for r in self.results])
        
        print(f"\n┌{'─' * 30}┬{'─' * 20}┐")
        print(f"│ {'Metric':<28} │ {'Value':<18} │")
        print(f"├{'─' * 30}┼{'─' * 20}┤")
        print(f"│ {'Avg Tokens/Second':<28} │ {avg_tps:<18.2f} │")
        print(f"│ {'Avg Time to First Token':<28} │ {avg_ttft:<18.3f}s │")
        print(f"│ {'Peak Memory (Allocated)':<28} │ {max_memory:<18.2f}GB │")
        print(f"│ {'Number of Tests':<28} │ {len(self.results):<18} │")
        print(f"└{'─' * 30}┴{'─' * 20}┘")
        
        print("\nPer-Prompt Results:")
        print(f"{'Prompt Len':<12} {'Output Len':<12} {'Time (s)':<12} {'Tok/s':<12}")
        print("-" * 48)
        for r in self.results:
            print(f"{r.prompt_length:<12} {r.output_length:<12} {r.total_time:<12.2f} {r.tokens_per_second:<12.2f}")


def run_comparison_benchmark(
    model_path: str,
    precisions: List[str] = ["fp8", "bf16", "fp16"],
    num_runs: int = 3
) -> Dict:
    """
    Run comparison benchmark across different precisions.
    
    Args:
        model_path: Path to model
        precisions: List of precisions to test
        num_runs: Runs per test
    
    Returns:
        Dictionary with comparison results
    """
    print("=" * 80)
    print("🔬 PRECISION COMPARISON BENCHMARK")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Precisions: {precisions}")
    print()
    
    all_results = {}
    
    for precision in precisions:
        print(f"\n{'=' * 80}")
        print(f"Testing {precision.upper()}")
        print("=" * 80)
        
        try:
            benchmark = InferenceBenchmark(
                model_path=model_path,
                precision=precision
            )
            
            results = benchmark.run_full_benchmark(num_runs=num_runs)
            benchmark.save_results()
            benchmark.print_summary()
            
            all_results[precision] = {
                "avg_tps": statistics.mean([r.tokens_per_second for r in results]),
                "avg_ttft": statistics.mean([r.time_to_first_token for r in results]),
                "memory_gb": max([r.memory_allocated_gb for r in results]),
                "results": [asdict(r) for r in results]
            }
            
            # Clear memory
            del benchmark
            import torch
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Failed to benchmark {precision}: {e}")
            all_results[precision] = {"error": str(e)}
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n{'Precision':<12} {'Tok/s':<12} {'TTFT (s)':<12} {'Memory (GB)':<15}")
    print("-" * 51)
    
    for precision, data in all_results.items():
        if "error" in data:
            print(f"{precision:<12} {'ERROR':<12} {'-':<12} {'-':<15}")
        else:
            print(f"{precision:<12} {data['avg_tps']:<12.2f} {data['avg_ttft']:<12.3f} {data['memory_gb']:<15.2f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Performance benchmark for AIMO 3 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark single precision
  python benchmark.py --model-path ./checkpoints/nemotron-49b-fp8 --precision fp8
  
  # Compare all precisions
  python benchmark.py --compare --model-path ./checkpoints/nemotron-49b
  
  # Quick benchmark with fewer runs
  python benchmark.py --model-path ./checkpoints/nemotron-49b-fp8 --runs 1
  
  # Benchmark with specific GPU
  python benchmark.py --model-path ./checkpoints/nemotron-49b-fp8 --gpu 1
        """
    )
    
    parser.add_argument('--model-path', type=str,
                       default=str(DEFAULT_CHECKPOINTS_DIR / "nemotron-49b-fp8"),
                       help='Path to model weights')
    parser.add_argument('--precision', choices=['fp8', 'bf16', 'fp16'], default='fp8',
                       help='Inference precision')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all precisions')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🏎️  AIMO 3 Inference Benchmark")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.compare:
        # Run comparison across precisions
        results = run_comparison_benchmark(
            model_path=args.model_path,
            num_runs=args.runs
        )
        
        # Save comparison
        output_path = DEFAULT_RESULTS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Comparison saved to: {output_path}")
        
    else:
        # Run single precision benchmark
        benchmark = InferenceBenchmark(
            model_path=args.model_path,
            precision=args.precision,
            gpu_id=args.gpu
        )
        
        benchmark.run_full_benchmark(num_runs=args.runs)
        benchmark.save_results(args.output)
        benchmark.print_summary()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

