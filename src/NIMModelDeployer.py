#!/usr/bin/env python3
"""
NIM Model Deployer for AIMO 3

NVIDIA Inference Microservice (NIM) style deployer for Llama-3.3-Nemotron-Super-49B.
Uses vLLM as the inference backend with Docker containerization.

This deployer is for standard environments with Docker access.
For Kaggle/restricted environments, use UNIMModelDeployer.py instead.

Features:
- Docker-based deployment (full NIM style)
- OpenAI-compatible API
- FP8/BF16/FP16 precision support
- Health checks and monitoring
- Automatic GPU detection

Usage:
    python NIMModelDeployer.py --model-path ./checkpoints/nemotron-49b-fp8 --port 8000
    python NIMModelDeployer.py --start --precision fp8
    python NIMModelDeployer.py --stop
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
from typing import Optional, Dict, List
import threading

# Path configuration
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"


class NIMModelDeployer:
    """
    Docker-based NIM-style model deployer using vLLM.
    
    This class manages the lifecycle of a vLLM inference server
    running in a Docker container, providing an OpenAI-compatible API.
    """
    
    # Default container configuration
    DEFAULT_IMAGE = "vllm/vllm-openai:latest"
    DEFAULT_PORT = 8000
    CONTAINER_NAME_PREFIX = "aimo3-nim"
    
    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        precision: str = "fp8",
        gpu_ids: str = "0",
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1
    ):
        """
        Initialize the NIM deployer.
        
        Args:
            model_path: Path to model weights
            port: API port
            precision: Inference precision (fp8, bf16, fp16)
            gpu_ids: GPU IDs to use (e.g., "0" or "0,1")
            max_model_len: Maximum context length
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        self.model_path = Path(model_path).absolute()
        self.port = port
        self.precision = precision
        self.gpu_ids = gpu_ids
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        
        self.container_name = f"{self.CONTAINER_NAME_PREFIX}-{port}"
        self.api_base = f"http://localhost:{port}/v1"
        self.process = None
        self.client = None
        
        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_nvidia_docker(self) -> bool:
        """Check if NVIDIA Docker runtime is available."""
        try:
            result = subprocess.run(
                ["docker", "run", "--rm", "--gpus", "all", "nvidia/cuda:12.1.0-base-ubuntu22.04", "nvidia-smi"],
                capture_output=True, text=True, timeout=30
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _get_dtype_arg(self) -> str:
        """Get the dtype argument for vLLM based on precision."""
        dtype_map = {
            "fp8": "float8",
            "bf16": "bfloat16",
            "fp16": "float16",
            "auto": "auto"
        }
        return dtype_map.get(self.precision, "auto")
    
    def start_docker(self) -> bool:
        """
        Start the model server in a Docker container.
        
        Returns:
            True if server started successfully
        """
        print("=" * 80)
        print("🐳 Starting NIM Docker Container")
        print("=" * 80)
        
        # Check Docker availability
        if not self._check_docker():
            print("❌ Docker is not available. Please install Docker.")
            return False
        
        print(f"📦 Container: {self.container_name}")
        print(f"🔮 Model: {self.model_path}")
        print(f"⚡ Precision: {self.precision}")
        print(f"🖥️  GPUs: {self.gpu_ids}")
        print(f"🌐 Port: {self.port}")
        
        # Stop existing container if running
        self.stop()
        
        # Build Docker command
        docker_cmd = [
            "docker", "run",
            "-d",  # Detached mode
            "--name", self.container_name,
            "--gpus", f'"device={self.gpu_ids}"',
            "--shm-size", "16g",
            "-p", f"{self.port}:{self.port}",
            "-v", f"{self.model_path}:/model:ro",
            self.DEFAULT_IMAGE,
            "--model", "/model",
            "--port", str(self.port),
            "--dtype", self._get_dtype_arg(),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.95"
        ]
        
        print(f"\n🚀 Starting container...")
        print(f"   Command: {' '.join(docker_cmd)}")
        
        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Failed to start container: {result.stderr}")
                return False
            
            container_id = result.stdout.strip()[:12]
            print(f"✅ Container started: {container_id}")
            
            # Wait for health check
            if self._wait_for_health():
                self._init_client()
                print(f"\n✅ Server ready at: {self.api_base}")
                return True
            else:
                print("❌ Server failed to become healthy")
                return False
                
        except Exception as e:
            print(f"❌ Error starting container: {e}")
            return False
    
    def start_native(self) -> bool:
        """
        Start the model server as a native process (no Docker).
        
        Returns:
            True if server started successfully
        """
        print("=" * 80)
        print("🐍 Starting Native vLLM Server")
        print("=" * 80)
        
        print(f"🔮 Model: {self.model_path}")
        print(f"⚡ Precision: {self.precision}")
        print(f"🖥️  GPUs: {self.gpu_ids}")
        print(f"🌐 Port: {self.port}")
        
        # Set GPU visibility
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        
        # Build vLLM command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_path),
            "--port", str(self.port),
            "--dtype", self._get_dtype_arg(),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--max-model-len", str(self.max_model_len),
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.95"
        ]
        
        print(f"\n🚀 Starting server...")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # Start process in background
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            
            # Start log streaming in background
            self._start_log_stream()
            
            # Wait for health check
            if self._wait_for_health():
                self._init_client()
                print(f"\n✅ Server ready at: {self.api_base}")
                return True
            else:
                print("❌ Server failed to become healthy")
                self.stop()
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def _start_log_stream(self):
        """Start streaming server logs in background."""
        def stream_logs():
            if self.process and self.process.stderr:
                for line in iter(self.process.stderr.readline, b''):
                    if line:
                        print(f"   [vLLM] {line.decode().strip()}")
        
        log_thread = threading.Thread(target=stream_logs, daemon=True)
        log_thread.start()
    
    def _wait_for_health(self, timeout: int = 600) -> bool:
        """
        Wait for the server to become healthy.
        
        Args:
            timeout: Maximum wait time in seconds
        
        Returns:
            True if server is healthy
        """
        import requests
        
        print(f"\n⏳ Waiting for server health check (timeout: {timeout}s)...")
        
        start_time = time.time()
        check_interval = 5
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
                if response.status_code == 200:
                    elapsed = time.time() - start_time
                    print(f"✅ Health check passed in {elapsed:.1f}s")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            except Exception as e:
                print(f"   ⚠️ Health check error: {e}")
            
            time.sleep(check_interval)
            elapsed = time.time() - start_time
            print(f"   ⏳ Waiting... ({elapsed:.0f}s)")
        
        return False
    
    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.api_base,
                api_key="EMPTY"
            )
            print("✅ OpenAI client initialized")
        except ImportError:
            print("⚠️ OpenAI client not available (pip install openai)")
            self.client = None
    
    def stop(self):
        """Stop the server (Docker or native)."""
        print("\n🛑 Stopping server...")
        
        # Stop Docker container
        try:
            result = subprocess.run(
                ["docker", "stop", self.container_name],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                print(f"   ✅ Container {self.container_name} stopped")
            
            subprocess.run(
                ["docker", "rm", self.container_name],
                capture_output=True, text=True, timeout=10
            )
        except:
            pass
        
        # Stop native process
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            print("   ✅ Native process stopped")
        
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
        Generate a response using the deployed model.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (enables reasoning mode)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
        
        Returns:
            Generated text
        """
        if not self.client:
            raise RuntimeError("Server not running. Call start_docker() or start_native() first.")
        
        # Default system prompt for reasoning
        if system_prompt is None:
            system_prompt = (
                "You are an expert mathematician. Reasoning mode: ON.\n"
                "Think step-by-step within <think> tags before providing your final answer.\n"
                "Output the final answer as a non-negative integer inside \\boxed{}."
            )
        
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
    
    def get_status(self) -> Dict:
        """Get server status information."""
        import requests
        
        status = {
            "running": False,
            "port": self.port,
            "model": str(self.model_path),
            "precision": self.precision
        }
        
        try:
            response = requests.get(f"http://localhost:{self.port}/health", timeout=5)
            if response.status_code == 200:
                status["running"] = True
                status["health"] = "healthy"
        except:
            status["health"] = "unreachable"
        
        return status


def main():
    parser = argparse.ArgumentParser(
        description="NIM-style Model Deployer for AIMO 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with Docker
  python NIMModelDeployer.py --start --docker --model-path ./checkpoints/nemotron-49b-fp8
  
  # Start native server (no Docker)
  python NIMModelDeployer.py --start --model-path ./checkpoints/nemotron-49b-fp8
  
  # Stop server
  python NIMModelDeployer.py --stop
  
  # Check status
  python NIMModelDeployer.py --status
  
  # Custom configuration
  python NIMModelDeployer.py --start --precision bf16 --port 8001 --gpus "0,1"
        """
    )
    
    parser.add_argument('--start', action='store_true', help='Start the server')
    parser.add_argument('--stop', action='store_true', help='Stop the server')
    parser.add_argument('--status', action='store_true', help='Check server status')
    parser.add_argument('--docker', action='store_true', help='Use Docker deployment')
    parser.add_argument('--model-path', type=str, 
                       default=str(DEFAULT_CHECKPOINTS_DIR / "nemotron-49b-fp8"),
                       help='Path to model weights')
    parser.add_argument('--port', type=int, default=8000, help='API port')
    parser.add_argument('--precision', choices=['fp8', 'bf16', 'fp16'], default='fp8',
                       help='Inference precision')
    parser.add_argument('--gpus', type=str, default='0', help='GPU IDs (e.g., "0" or "0,1")')
    parser.add_argument('--max-model-len', type=int, default=8192, help='Max context length')
    parser.add_argument('--tensor-parallel', type=int, default=1, help='Tensor parallel size')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 NIM Model Deployer for AIMO 3")
    print("=" * 80)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create deployer
    try:
        deployer = NIMModelDeployer(
            model_path=args.model_path,
            port=args.port,
            precision=args.precision,
            gpu_ids=args.gpus,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run download_checkpoints.py first to download model weights")
        sys.exit(1)
    
    # Handle commands
    if args.status:
        status = deployer.get_status()
        print("📊 Server Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")
    
    elif args.stop:
        deployer.stop()
        print("✅ Server stopped")
    
    elif args.start:
        # Setup signal handlers
        def signal_handler(sig, frame):
            print("\n\n⚠️ Interrupt received, shutting down...")
            deployer.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start server
        if args.docker:
            success = deployer.start_docker()
        else:
            success = deployer.start_native()
        
        if success:
            print("\n" + "=" * 80)
            print("✅ Server is running!")
            print(f"   API Base: {deployer.api_base}")
            print(f"   Health: http://localhost:{args.port}/health")
            print(f"   Models: http://localhost:{args.port}/v1/models")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 80)
            
            # Keep running
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

