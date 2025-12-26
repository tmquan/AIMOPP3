#!/usr/bin/env python3
"""
UNIM (User-space NIM) Model Deployer for AIMO 3

User-space implementation of NIM-like deployment for restricted environments
like Kaggle notebooks where Docker access is unavailable.

This deployer:
- Runs vLLM as a native Python process
- Supports offline wheel installation
- Works without root/Docker access
- Provides the same OpenAI-compatible API

Usage:
    python UNIMModelDeployer.py --start --model-path ./checkpoints/nemotron-49b-fp8
    python UNIMModelDeployer.py --start --precision fp8 --port 8000
"""

import os
import sys
import subprocess
import time
import argparse
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Callable
import threading

# Path configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
DEFAULT_WHEELS_DIR = PROJECT_DIR / "checkpoints" / "wheels"


class UNIMModelDeployer:
    """
    User-space NIM Model Deployer.
    
    Designed for restricted environments (Kaggle, air-gapped systems)
    where Docker is unavailable. Uses native Python vLLM execution.
    
    Features:
    - Native Python execution (no Docker)
    - Offline wheel installation support
    - OpenAI-compatible API
    - Health monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        precision: str = "fp8",
        gpu_id: int = 0,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.95
    ):
        """
        Initialize UNIM deployer.
        
        Args:
            model_path: Path to model weights
            port: API port
            precision: Inference precision (fp8, bf16, fp16)
            gpu_id: GPU device ID
            max_model_len: Maximum context length
            gpu_memory_utilization: GPU memory fraction to use
        """
        self.model_path = Path(model_path).absolute()
        self.port = port
        self.precision = precision
        self.gpu_id = gpu_id
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        
        self.api_base = f"http://localhost:{port}/v1"
        self.process = None
        self.client = None
        self._log_thread = None
        self._shutdown_event = threading.Event()
        
        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
    
    @staticmethod
    def install_offline_packages(wheels_dir: str):
        """
        Install packages from local wheel files (for offline environments).
        
        Args:
            wheels_dir: Directory containing wheel files
        """
        wheels_dir = Path(wheels_dir)
        
        if not wheels_dir.exists():
            print(f"⚠️ Wheels directory not found: {wheels_dir}")
            return False
        
        print(f"📦 Installing packages from: {wheels_dir}")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "--no-index",
                "--find-links", str(wheels_dir),
                "vllm", "openai"
            ])
            print("✅ Offline installation complete")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def _get_dtype_arg(self) -> str:
        """Get dtype argument based on precision setting."""
        dtype_map = {
            "fp8": "float8",
            "bf16": "bfloat16",
            "fp16": "float16",
            "auto": "auto"
        }
        return dtype_map.get(self.precision, "auto")
    
    def start_server(self) -> bool:
        """
        Start the vLLM server as a native process.
        
        Returns:
            True if server started successfully
        """
        print("=" * 80)
        print("🚀 Starting UNIM (User-space NIM) Server")
        print("=" * 80)
        
        print(f"📂 Model: {self.model_path}")
        print(f"⚡ Precision: {self.precision}")
        print(f"🖥️  GPU: {self.gpu_id}")
        print(f"🌐 Port: {self.port}")
        print(f"📊 Context Length: {self.max_model_len}")
        print(f"💾 GPU Memory: {self.gpu_memory_utilization * 100:.0f}%")
        
        # Set GPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        
        # Build command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_path),
            "--port", str(self.port),
            "--dtype", self._get_dtype_arg(),
            "--tensor-parallel-size", "1",  # Single GPU for UNIM
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--trust-remote-code",
            "--disable-log-requests"  # Reduce log noise
        ]
        
        # Add FP8 specific flags if needed
        if self.precision == "fp8":
            cmd.extend(["--quantization", "fp8"])
        
        print(f"\n📋 Command: {' '.join(cmd)}")
        print("\n⏳ Starting server process...")
        
        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                bufsize=1,
                universal_newlines=True
            )
            
            # Start log streaming
            self._start_log_streaming()
            
            # Wait for server to be ready
            print("\n⏳ Waiting for server to initialize...")
            if self._wait_for_health():
                self._init_client()
                print(f"\n✅ UNIM Server ready!")
                print(f"   API: {self.api_base}")
                return True
            else:
                print("❌ Server failed to start")
                self.stop()
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def _start_log_streaming(self):
        """Stream server logs in background."""
        def stream_stderr():
            while not self._shutdown_event.is_set():
                if self.process and self.process.stderr:
                    line = self.process.stderr.readline()
                    if line:
                        # Filter out noisy log lines
                        if "INFO" in line or "WARNING" in line or "ERROR" in line:
                            print(f"   [vLLM] {line.strip()}")
                else:
                    time.sleep(0.1)
        
        self._log_thread = threading.Thread(target=stream_stderr, daemon=True)
        self._log_thread.start()
    
    def _wait_for_health(self, timeout: int = 600, interval: int = 5) -> bool:
        """
        Wait for server health check to pass.
        
        Args:
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds
        
        Returns:
            True if server is healthy
        """
        import requests
        
        start_time = time.time()
        health_url = f"http://localhost:{self.port}/health"
        
        while time.time() - start_time < timeout:
            # Check if process died
            if self.process and self.process.poll() is not None:
                print(f"❌ Process exited with code: {self.process.returncode}")
                return False
            
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"✅ Health check passed ({elapsed:.1f}s)")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"   ⚠️ {e}")
            
            elapsed = time.time() - start_time
            print(f"   ⏳ Still initializing... ({elapsed:.0f}s)")
            time.sleep(interval)
        
        return False
    
    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.api_base,
                api_key="EMPTY"  # vLLM doesn't require auth
            )
        except ImportError:
            print("⚠️ OpenAI client not available")
            self.client = None
    
    def stop(self):
        """Stop the server process."""
        print("\n🛑 Stopping UNIM server...")
        
        self._shutdown_event.set()
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("   ✅ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                self.process.kill()
                print("   ⚠️ Server force killed")
            self.process = None
        
        self.client = None
    
    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        stop: List[str] = None
    ) -> str:
        """
        Generate response using the deployed model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for reasoning mode
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            stop: Stop sequences
        
        Returns:
            Generated text
        """
        if not self.client:
            raise RuntimeError("Server not running. Call start_server() first.")
        
        # Default reasoning system prompt
        if system_prompt is None:
            system_prompt = self.get_reasoning_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=str(self.model_path),
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop or ["<|eot_id|>"]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ Generation error: {e}")
            return ""
    
    def generate_reasoning(
        self,
        problem: str,
        temperature: float = 0.6
    ) -> Dict:
        """
        Generate a reasoning response with structured output.
        
        Args:
            problem: Mathematical problem
            temperature: Sampling temperature
        
        Returns:
            Dict with 'thinking', 'answer', and 'raw' fields
        """
        system_prompt = self.get_reasoning_system_prompt()
        
        raw_response = self.generate(
            prompt=f"Problem: {problem}\n\nSolve this step by step.",
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # Parse response
        thinking = ""
        answer = None
        
        # Extract thinking
        import re
        think_match = re.search(r'<think>(.*?)</think>', raw_response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
        
        # Extract boxed answer
        box_match = re.findall(r'\\boxed\{([^}]+)\}', raw_response)
        if box_match:
            # Take the last boxed answer
            answer_str = box_match[-1]
            # Try to extract integer
            digits = re.sub(r'\D', '', answer_str)
            if digits:
                answer = int(digits)
        
        return {
            'thinking': thinking,
            'answer': answer,
            'raw': raw_response
        }
    
    @staticmethod
    def get_reasoning_system_prompt() -> str:
        """Get the default reasoning mode system prompt."""
        return """You are an expert mathematician participating in the International Mathematical Olympiad. Reasoning mode: ON.

Your goal is to solve the given problem accurately. Follow this strict protocol:

1. **Exploration**: First, explore the problem space within <think> tags. Break the problem down into sub-components.
2. **Step-by-Step Execution**: Perform calculations explicitly.
3. **Self-Correction**: After every major step, verify the intermediate result. If you encounter a contradiction, backtrack and try a different approach.
4. **Formatting**: Output the final answer as a non-negative integer between 0 and 99999 inside \\boxed{}. Do not output fractions or equations in the box."""
    
    @staticmethod
    def get_pot_system_prompt() -> str:
        """Get the Program of Thought system prompt."""
        return """You are a computational mathematician. To solve the problem, write a robust Python script that calculates the answer.

1. Define the variables and constraints clearly.
2. Implement the logic in a function.
3. Print the final answer at the end of the script.
4. Do not guess; calculate.
5. Put your final integer answer in \\boxed{}."""
    
    def get_status(self) -> Dict:
        """Get server status."""
        import requests
        
        status = {
            "running": False,
            "port": self.port,
            "model": str(self.model_path),
            "precision": self.precision,
            "gpu": self.gpu_id
        }
        
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            status["running"] = response.status_code == 200
            status["health"] = "healthy" if status["running"] else "unhealthy"
        except:
            status["health"] = "unreachable"
        
        return status


def main():
    parser = argparse.ArgumentParser(
        description="UNIM (User-space NIM) Deployer for AIMO 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server
  python UNIMModelDeployer.py --start --model-path ./checkpoints/nemotron-49b-fp8
  
  # Start with custom settings
  python UNIMModelDeployer.py --start --precision bf16 --port 8001 --gpu 1
  
  # Install offline packages first (for Kaggle)
  python UNIMModelDeployer.py --install-offline --wheels-dir ./checkpoints/wheels
  
  # Check status
  python UNIMModelDeployer.py --status
  
  # Stop server
  python UNIMModelDeployer.py --stop
        """
    )
    
    parser.add_argument('--start', action='store_true', help='Start the server')
    parser.add_argument('--stop', action='store_true', help='Stop the server')
    parser.add_argument('--status', action='store_true', help='Check server status')
    parser.add_argument('--install-offline', action='store_true', 
                       help='Install packages from offline wheels')
    parser.add_argument('--wheels-dir', type=str, default=str(DEFAULT_WHEELS_DIR),
                       help='Directory with wheel files')
    parser.add_argument('--model-path', type=str,
                       default=str(DEFAULT_CHECKPOINTS_DIR / "nemotron-49b-fp8"),
                       help='Path to model weights')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--precision', choices=['fp8', 'bf16', 'fp16'], default='fp8',
                       help='Inference precision')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--max-model-len', type=int, default=8192, help='Max context length')
    parser.add_argument('--memory-util', type=float, default=0.95, help='GPU memory utilization')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 UNIM (User-space NIM) Deployer for AIMO 3")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Handle offline installation
    if args.install_offline:
        UNIMModelDeployer.install_offline_packages(args.wheels_dir)
        return
    
    # Create deployer
    try:
        deployer = UNIMModelDeployer(
            model_path=args.model_path,
            port=args.port,
            precision=args.precision,
            gpu_id=args.gpu,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.memory_util
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run download_checkpoints.py first")
        sys.exit(1)
    
    # Handle commands
    if args.status:
        status = deployer.get_status()
        print("📊 UNIM Server Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    
    elif args.stop:
        deployer.stop()
    
    elif args.start:
        # Setup signal handlers
        def signal_handler(sig, frame):
            print("\n\n⚠️ Interrupt received...")
            deployer.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        if deployer.start_server():
            print("\n" + "=" * 80)
            print("✅ UNIM Server Running!")
            print(f"   API: {deployer.api_base}")
            print(f"   Health: http://localhost:{args.port}/health")
            print("\nPress Ctrl+C to stop")
            print("=" * 80)
            
            # Keep alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                deployer.stop()
        else:
            sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

