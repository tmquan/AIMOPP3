#!/usr/bin/env python3
"""
AIMO 3 Submission Generator

Generates submission.csv from data.csv using Llama-3.3-Nemotron-Super-49B
with optimized system prompts for mathematical reasoning.

This script:
1. Loads problems from data.csv
2. Applies System 2 reasoning prompts
3. Generates solutions using the deployed model
4. Extracts answers and creates submission.csv

Usage:
    python generate.py --input data.csv --output submission.csv
    python generate.py --input data.csv --output submission.csv --strategy ensemble
    python generate.py --server-url http://localhost:8000/v1
"""

import os
import sys
import re
import csv
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time

# Path configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_CHECKPOINTS_DIR = SCRIPT_DIR / "checkpoints"
DEFAULT_DATA_DIR = SCRIPT_DIR / "datasets" / "aimo3"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "submissions"


# =============================================================================
# SYSTEM PROMPTS
# =============================================================================

SYSTEM_PROMPTS = {
    "reasoning": """You are an expert mathematician participating in the International Mathematical Olympiad. Reasoning mode: ON.

Your goal is to solve the given problem accurately. Follow this strict protocol:

1. **Exploration**: First, explore the problem space within <think> tags. Break the problem down into sub-components.
2. **Step-by-Step Execution**: Perform calculations explicitly. Show all work.
3. **Self-Correction**: After every major step, verify the intermediate result. If you encounter a contradiction, backtrack and try a different approach.
4. **Verification**: Before finalizing, check your answer by substitution or alternative method.
5. **Formatting**: Output the final answer as a non-negative integer between 0 and 99999 inside \\boxed{}. Do not output fractions, decimals, or equations in the box.

Remember: The answer MUST be a non-negative integer. If your calculation yields a non-integer, re-examine your approach.""",

    "pot": """You are a computational mathematician. To solve the problem, write a robust Python script that calculates the answer.

Follow these rules:
1. Define the variables and constraints clearly in comments.
2. Implement the logic in well-documented functions.
3. Use exact arithmetic where possible (fractions, integers).
4. Print the final answer at the end of the script.
5. The answer must be a non-negative integer between 0 and 99999.
6. Put your final integer answer in \\boxed{}.

Do not guess; calculate. Verify your implementation handles edge cases.""",

    "verification": """You are a mathematical verification expert. Your task is to verify a proposed solution.

Given:
- The original problem
- A proposed answer

Your job:
1. Check if the answer satisfies all conditions in the problem.
2. Try to find a counterexample or contradiction.
3. If the answer is correct, confirm it.
4. If incorrect, explain why and provide the correct answer.

Output the verified answer in \\boxed{}.""",

    "simple": """Solve the following math problem. Show your work step by step.
Output the final answer as a non-negative integer inside \\boxed{}."""
}


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_boxed_answer(text: str) -> Optional[int]:
    """
    Extract integer answer from \\boxed{} in text.
    
    Handles various formats:
    - \\boxed{42}
    - \\boxed{42.0}
    - \\boxed{-42} (takes absolute value)
    - Multiple boxed (takes last one)
    
    Args:
        text: Model output text
    
    Returns:
        Integer answer or None if not found
    """
    if not text:
        return None
    
    # Find all boxed expressions
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)
    
    if not matches:
        # Try alternative format: \boxed{...}
        matches = re.findall(r'boxed\{([^}]+)\}', text)
    
    if not matches:
        return None
    
    # Take the last match (final answer)
    answer_str = matches[-1].strip()
    
    # Clean up the answer
    # Remove LaTeX formatting
    answer_str = answer_str.replace('\\', '')
    answer_str = answer_str.replace(',', '')  # Remove thousands separators
    answer_str = answer_str.replace(' ', '')
    
    # Try to extract number
    try:
        # Try direct integer parse
        return abs(int(answer_str))
    except ValueError:
        pass
    
    try:
        # Try float and convert to int
        return abs(int(float(answer_str)))
    except ValueError:
        pass
    
    # Try to extract digits only
    digits = re.sub(r'[^\d]', '', answer_str)
    if digits:
        return int(digits)
    
    return None


def extract_answer_from_code(text: str) -> Optional[int]:
    """
    Extract answer from code output or print statements.
    
    Looks for patterns like:
    - print(42)
    - answer = 42
    - The answer is 42
    """
    patterns = [
        r'print\((\d+)\)',
        r'answer\s*=\s*(\d+)',
        r'result\s*=\s*(\d+)',
        r'The answer is (\d+)',
        r'Therefore.*?(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    
    return None


# =============================================================================
# GENERATION STRATEGIES
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for generation."""
    temperature: float = 0.6
    max_tokens: int = 4096
    top_p: float = 0.95
    num_samples: int = 1  # For self-consistency


class SubmissionGenerator:
    """
    Generator for AIMO 3 submissions.
    
    Strategies:
    - single: Single generation with reasoning prompt
    - pot: Program of Thought generation
    - ensemble: Multiple generations with majority voting
    - verify: Generate then verify
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000/v1",
        model_name: str = None,
        config: GenerationConfig = None
    ):
        """
        Initialize generator.
        
        Args:
            server_url: vLLM server URL
            model_name: Model name (optional, auto-detected)
            config: Generation configuration
        """
        self.server_url = server_url
        self.config = config or GenerationConfig()
        
        # Initialize OpenAI client
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=server_url,
                api_key="EMPTY"
            )
        except ImportError:
            print("❌ OpenAI client not installed. Run: pip install openai")
            sys.exit(1)
        
        # Get model name
        if model_name:
            self.model_name = model_name
        else:
            self.model_name = self._get_model_name()
    
    def _get_model_name(self) -> str:
        """Get model name from server."""
        try:
            models = self.client.models.list()
            if models.data:
                return models.data[0].id
            return "unknown"
        except Exception as e:
            print(f"⚠️ Could not get model name: {e}")
            return "unknown"
    
    def _generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = None,
        max_tokens: int = None
    ) -> str:
        """
        Generate a single response.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
        
        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens,
                top_p=self.config.top_p,
                stop=["<|eot_id|>"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""
    
    def solve_single(self, problem: str) -> Tuple[Optional[int], str]:
        """
        Solve problem with single reasoning generation.
        
        Returns:
            Tuple of (answer, raw_response)
        """
        prompt = f"Problem: {problem}\n\nSolve this problem step by step."
        response = self._generate(prompt, SYSTEM_PROMPTS["reasoning"])
        answer = extract_boxed_answer(response)
        return answer, response
    
    def solve_pot(self, problem: str) -> Tuple[Optional[int], str]:
        """
        Solve problem with Program of Thought.
        
        Returns:
            Tuple of (answer, raw_response)
        """
        prompt = f"Problem: {problem}\n\nWrite Python code to solve this."
        response = self._generate(prompt, SYSTEM_PROMPTS["pot"])
        
        # Try boxed first, then code extraction
        answer = extract_boxed_answer(response)
        if answer is None:
            answer = extract_answer_from_code(response)
        
        return answer, response
    
    def solve_ensemble(
        self,
        problem: str,
        num_samples: int = 5
    ) -> Tuple[Optional[int], List[str]]:
        """
        Solve problem with self-consistency (ensemble).
        
        Generates multiple solutions and takes majority vote.
        
        Returns:
            Tuple of (answer, list of responses)
        """
        responses = []
        answers = []
        
        # Generate multiple solutions
        for i in range(num_samples):
            prompt = f"Problem: {problem}\n\nSolve this problem step by step."
            # Use higher temperature for diversity
            response = self._generate(
                prompt,
                SYSTEM_PROMPTS["reasoning"],
                temperature=0.7
            )
            responses.append(response)
            
            answer = extract_boxed_answer(response)
            if answer is not None:
                answers.append(answer)
        
        # Majority vote
        if answers:
            from collections import Counter
            vote = Counter(answers)
            final_answer = vote.most_common(1)[0][0]
        else:
            final_answer = None
        
        return final_answer, responses
    
    def solve_with_verification(
        self,
        problem: str
    ) -> Tuple[Optional[int], Dict]:
        """
        Solve problem and then verify the answer.
        
        Returns:
            Tuple of (answer, {initial_response, verification_response})
        """
        # First generation
        initial_answer, initial_response = self.solve_single(problem)
        
        if initial_answer is None:
            return None, {"initial": initial_response, "verification": ""}
        
        # Verification
        verify_prompt = f"""Problem: {problem}

Proposed Answer: {initial_answer}

Verify this answer."""
        
        verify_response = self._generate(
            verify_prompt,
            SYSTEM_PROMPTS["verification"]
        )
        
        # Extract verified answer
        verified_answer = extract_boxed_answer(verify_response)
        
        # Use verified answer if available, otherwise initial
        final_answer = verified_answer if verified_answer is not None else initial_answer
        
        return final_answer, {
            "initial": initial_response,
            "verification": verify_response
        }
    
    def generate_submission(
        self,
        input_path: str,
        output_path: str,
        strategy: str = "single",
        problem_column: str = "problem",
        id_column: str = "id"
    ) -> Dict:
        """
        Generate submission CSV from input CSV.
        
        Args:
            input_path: Path to input CSV
            output_path: Path to output CSV
            strategy: Solving strategy (single, pot, ensemble, verify)
            problem_column: Column with problems
            id_column: Column with IDs
        
        Returns:
            Statistics dictionary
        """
        print("=" * 80)
        print("🚀 AIMO 3 Submission Generator")
        print("=" * 80)
        print(f"📂 Input: {input_path}")
        print(f"📂 Output: {output_path}")
        print(f"🎯 Strategy: {strategy}")
        print(f"🤖 Model: {self.model_name}")
        print(f"🌐 Server: {self.server_url}")
        print()
        
        # Load input
        import pandas as pd
        df = pd.read_csv(input_path)
        
        print(f"📊 Loaded {len(df)} problems")
        
        # Select strategy
        solve_func = {
            "single": self.solve_single,
            "pot": self.solve_pot,
            "ensemble": lambda p: (self.solve_ensemble(p, num_samples=5)[0], ""),
            "verify": lambda p: (self.solve_with_verification(p)[0], "")
        }.get(strategy, self.solve_single)
        
        # Generate solutions
        results = []
        start_time = time.time()
        
        for idx, row in df.iterrows():
            problem_id = row[id_column]
            problem_text = row[problem_column]
            
            print(f"\n📝 Problem {idx + 1}/{len(df)} (ID: {problem_id})")
            print(f"   {problem_text[:100]}...")
            
            try:
                answer, _ = solve_func(problem_text)
                
                # Default to 0 if no answer extracted
                if answer is None:
                    print(f"   ⚠️ No answer extracted, defaulting to 0")
                    answer = 0
                
                # Ensure answer is in valid range
                answer = answer % 100000
                
                print(f"   ✅ Answer: {answer}")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                answer = 0
            
            results.append({
                "id": problem_id,
                "answer": answer
            })
        
        elapsed = time.time() - start_time
        
        # Save submission
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        
        # Statistics
        stats = {
            "total_problems": len(df),
            "total_time": round(elapsed, 2),
            "avg_time_per_problem": round(elapsed / len(df), 2),
            "strategy": strategy,
            "model": self.model_name
        }
        
        print("\n" + "=" * 80)
        print("📊 GENERATION COMPLETE")
        print("=" * 80)
        print(f"   Total problems: {stats['total_problems']}")
        print(f"   Total time: {stats['total_time']}s")
        print(f"   Avg time/problem: {stats['avg_time_per_problem']}s")
        print(f"   Output: {output_path}")
        print("=" * 80)
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate AIMO 3 submission from problems CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic generation
  python generate.py --input data.csv --output submission.csv
  
  # With ensemble strategy (self-consistency)
  python generate.py --input data.csv --output submission.csv --strategy ensemble
  
  # With verification
  python generate.py --input data.csv --output submission.csv --strategy verify
  
  # Program of Thought
  python generate.py --input data.csv --output submission.csv --strategy pot
  
  # Custom server
  python generate.py --input data.csv --output submission.csv --server-url http://localhost:8001/v1

Strategies:
  single   - Single generation with reasoning mode (fastest)
  pot      - Program of Thought (code-based)
  ensemble - Multiple generations with majority voting (most accurate)
  verify   - Generate and then verify answer
        """
    )
    
    parser.add_argument('--input', '-i', type=str,
                       default=str(DEFAULT_DATA_DIR / "data.csv"),
                       help='Input CSV with problems')
    parser.add_argument('--output', '-o', type=str,
                       default=str(DEFAULT_OUTPUT_DIR / "submission.csv"),
                       help='Output submission CSV')
    parser.add_argument('--strategy', choices=['single', 'pot', 'ensemble', 'verify'],
                       default='single', help='Solving strategy')
    parser.add_argument('--server-url', type=str, default='http://localhost:8000/v1',
                       help='vLLM server URL')
    parser.add_argument('--model', type=str, help='Model name (optional)')
    parser.add_argument('--problem-column', default='problem', help='Problem text column')
    parser.add_argument('--id-column', default='id', help='ID column')
    parser.add_argument('--temperature', type=float, default=0.6, help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max tokens per response')
    
    args = parser.parse_args()
    
    # Create configuration
    config = GenerationConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Create generator
    generator = SubmissionGenerator(
        server_url=args.server_url,
        model_name=args.model,
        config=config
    )
    
    # Generate submission
    stats = generator.generate_submission(
        input_path=args.input,
        output_path=args.output,
        strategy=args.strategy,
        problem_column=args.problem_column,
        id_column=args.id_column
    )
    
    # Save stats
    stats_path = Path(args.output).with_suffix('.stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Stats saved to: {stats_path}")


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

