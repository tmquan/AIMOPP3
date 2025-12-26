# AIMO 3 Progress Prize - Nemotron-Super-49B Deployment

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model checkpoints (FP8 recommended for H100)
python download_checkpoints.py --49b-fp8 --embed-8b

# 3. Download competition data
python download_data.py --all

# 4. Start the inference server
python src/UNIMModelDeployer.py --start --precision fp8

# 5. Generate submissions
python generate.py --input datasets/aimo3/data.csv --output submissions/submission.csv
```

## Project Structure

```
AIMOPP3/
├── checkpoints/           # Model weights (FP8, BF16, embeddings)
├── datasets/              # Competition and training data
├── embeddings/            # Extracted embeddings
├── submissions/           # Generated submission files
├── visualizations/        # Embedding visualizations
├── src/
│   ├── NIMModelDeployer.py      # Docker-based deployment
│   ├── UNIMModelDeployer.py     # User-space deployment (Kaggle)
│   ├── extract_embeddings.py   # Embedding extraction
│   ├── visualize_embeddings.py # 2D/3D Plotly visualizations
│   └── benchmark.py            # Performance benchmarking
├── download_data.py       # Data downloader
├── download_checkpoints.py # Checkpoint downloader
├── generate.py            # Submission generator
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

# Architectural Optimization and Deployment Strategy for Llama-3.3-Nemotron-Super-49B in Constrained HPC Environments

## 1. Executive Summary

The third iteration of the AI Mathematical Olympiad (AIMO) Progress Prize introduces a paradigm shift in competitive AI reasoning by providing access to NVIDIA H100 Tensor Core GPUs. This hardware allocation necessitates a fundamental reevaluation of model selection and deployment architectures. The introduction of the nvidia/Llama-3.3-Nemotron-Super-49B-v1 model, specifically engineered through Neural Architecture Search (NAS) and knowledge distillation to inhabit the memory-throughput "sweet spot" of a single H100 accelerator, offers a distinct strategic advantage. This report provides an exhaustive technical analysis and operational roadmap for leveraging this architecture within the strict offline and temporal constraints of the Kaggle competition environment.

Our analysis indicates that the 49B parameter count is not an arbitrary design choice but a calculated derivation intended to maximize reasoning density—the amount of logical inference capability per gigabyte of VRAM—while remaining compatible with FP8 quantization on Hopper architecture. Unlike standard 70B models that require tensor parallelism across multiple devices, the Nemotron-Super-49B allows for single-device residence, thereby eliminating inter-chip communication latency and freeing up computational budget for extended test-time compute strategies such as self-consistency and verification.

This document details the implementation of a "NIM-like" deployment flow—referencing the conceptual frameworks of NIMModelDeployer—adapted for the udocker or native Python environments necessitated by Kaggle's restrictions. We provide a granular, step-by-step operational guide for hosting the model, bypassing standard internet restrictions through offline caching, and orchestrating a robust inference loop. Furthermore, we explore the implementation of a retrieval-augmented generation (RAG) pipeline utilizing llama-embed-nemotron-8b to ground mathematical reasoning in a corpus of known theorems and similar problem structures. Finally, we present a suite of prompt engineering strategies specifically tuned for the Nemotron's distinct "Reasoning Mode," designed to elicit high-fidelity Chain-of-Thought (CoT) traces essential for solving Olympiad-level problems.

---

## 2. The Computational Physics of AIMO 3

### 2.1 The Hardware Constraint: The H100-80GB Envelope

The defining constraint of the AIMO 3 competition is the hardware limitation: a single NVIDIA H100 GPU with 80GB of High Bandwidth Memory (HBM3). To understand the strategic necessity of the 49B model, one must first analyze the arithmetic intensity and memory footprint required by modern Large Language Models (LLMs).

A standard transformer model's memory footprint for weights can be approximated as:

$$M_{weights} \approx P \times B$$

where $P$ is the parameter count and $B$ is the number of bytes per parameter. For a 70 billion parameter model (Llama-3-70B) utilizing standard BFloat16 (16-bit) precision:

$$M_{70B, BF16} \approx 70 \times 10^9 \times 2 \text{ bytes} = 140 \text{ GB}$$

This exceeds the 80GB capacity of the H100, rendering the model impossible to load without quantization or model parallelism (splitting the model across multiple GPUs). While Kaggle provides H100s, the environment often restricts multi-GPU communication efficiency or simply provides single instances for inference notebooks to maximize concurrent user access. Even with 4-bit quantization (INT4), a 70B model occupies roughly 35-40GB. While this fits, the compression often degrades the subtle reasoning capabilities required for Olympiad mathematics, where a single token error in a logical chain leads to failure.

The Llama-3.3-Nemotron-Super-49B addresses this physics problem directly. It is designed to be the largest possible model that fits comfortably on a single H100 while maintaining high precision. Crucially, the model is optimized for FP8 (8-bit Floating Point) inference, a native capability of the Hopper architecture.

$$M_{49B, FP8} \approx 49 \times 10^9 \times 1 \text{ byte} = 49 \text{ GB}$$

This leaves approximately **31GB of VRAM** available for the Key-Value (KV) cache and runtime activations. This surplus is critical. The KV cache grows linearly with context length and batch size. In a competition with a 5-hour runtime limit, high-throughput batch processing is essential. The 31GB headroom allows for significant batch sizes (e.g., evaluating 16 or 32 candidate solutions in parallel) or extended context windows for complex problem decomposition.

### 2.2 Neural Architecture Search (NAS) and Distillation

The "Super" designation in the model name stems from its creation process. Unlike standard pruning, which often degrades performance by removing weights based on magnitude, NVIDIA employed **distillation-based Neural Architecture Search (NAS)**. The process involved identifying a subset of layers and attention heads from the Llama 3.3 70B teacher model that preserved the highest variance of information.

The resulting architecture is **heterogeneous**. Standard Transformers use identical blocks repeated $N$ times. The Nemotron-Super-49B, however, features non-repetitive blocks; some blocks skip attention layers entirely or use variable Feed-Forward Network (FFN) ratios.

- **Skip Attention**: Reduces memory bandwidth usage during inference, as attention operations are memory-bound.
- **Variable FFN**: Concentrates computational capacity in layers where semantic transformation is most critical.

This structural optimization means the model retains the "reasoning density" of the 70B teacher model while achieving the latency profile of a significantly smaller model. For AIMO 3 competitors, this implies that the 49B model is not a compromise; it is a specialized reasoning engine distilled specifically for the constraints of modern datacenter accelerators.

### 2.3 The Reasoning Toggle and Inference Modes

A distinct feature of the Nemotron series is the **dynamic reasoning toggle**. Controlled via the system prompt, the model can switch between a standard conversational mode and a "Reasoning Mode".

- **Reasoning Mode (ON)**: The model generates internal `<think>` tags (or analogous structural markers depending on the chat template) where it performs verbose intermediate computation before emitting the final answer. This is analogous to the "System 2" thinking process in human cognition.

**Implication**: For AIMO, this mode is mandatory. The "hidden" reasoning steps allow the model to error-correct and traverse the search space of the solution before committing to a result. The H100's FP8 throughput is specifically leveraged to generate these verbose traces quickly.

---

## 3. Deployment Methodologies: Native NIM vs. Custom vLLM

The user query references `NIMModelDeployer.py` and `UNIMModelDeployer.py`. These files represent a conceptual wrapper for deploying NVIDIA Inference Microservices (NIM). In a standard enterprise environment, a NIM is a Docker container that wraps an inference engine (like TensorRT-LLM) with a standardized API. However, the Kaggle notebook environment presents unique challenges that prevent the direct use of standard NIM containers.

### 3.1 The Constraints of Kaggle Deployment

- **No Internet Access**: During the submission run, the notebook cannot pull Docker images from the NVIDIA GPU Cloud (NGC) or download weights from Hugging Face. Everything must be pre-staged.
- **No Root Access/Docker Socket**: Standard `docker run` commands often require root privileges or access to the Docker socket, which is typically restricted in multi-tenant environments like Kaggle notebooks to prevent container escape vulnerabilities.
- **File System Persistence**: The writable directory (`/kaggle/working`) is ephemeral. Large model weights must reside in the read-only input directories (`/kaggle/input`).

### 3.2 The NIMModelDeployer Architectural Pattern

Given these constraints, the `NIMModelDeployer` class referenced likely implements a **"User-Space NIM."** Instead of running a full Docker container, it orchestrates the underlying Python processes directly. The architecture consists of three components:

1. **The Engine**: vLLM (Versatile Library for Large Models) is the industry standard for this task due to its PagedAttention mechanism, which manages the H100's memory efficiently. It mimics the performance profile of a compiled TensorRT-LLM engine but with greater flexibility.
2. **The Server**: An OpenAI-compatible API server running in a background subprocess (`subprocess.Popen`). This creates a local `localhost` endpoint.
3. **The Client**: A local Python client that sends HTTP requests to the server.

The distinction between `NIMModelDeployer` and `UNIMModelDeployer` (likely "Universal" or "User" NIM) typically lies in the containerization level. `NIMModelDeployer` might assume a standard Docker environment, while `UNIMModelDeployer` likely uses `udocker` or direct python execution to bypass root requirements. For AIMO 3, we will implement the **direct execution model**, which is the most robust method for Kaggle.

### 3.3 Offline Asset Management

To replicate the NIM experience offline, we must manually curate the dependencies.

- **Weights**: The model weights must be downloaded locally and uploaded as a private Kaggle dataset. For the 49B model, downloading the FP8 quantized version directly is recommended to ensure compatibility.
- **Libraries**: The inference engine (vLLM) has complex dependencies (Torch, Triton, Xformers). These cannot be `pip install`ed at runtime without internet. We must download the `.whl` files for these libraries and upload them as a dataset.

### 3.4 Operational Flow Strategy

The optimal flow to leverage the llama3-nemotron-super-49B follows this strictly ordered pipeline:

| Stage | Description |
|-------|-------------|
| **Stage 0 (Offline)** | Asset acquisition. Download standard BF16 or FP8 weights. Download llama-embed-nemotron-8b. Download vLLM wheels. |
| **Stage 1 (Runtime Init)** | Install Python libraries from local wheels. |
| **Stage 2 (Server Launch)** | Spin up the vLLM server in a background thread, pointing it to the local model weights. Wait for the health check to pass. |
| **Stage 3 (Inference Loop)** | Use the Kaggle evaluation API to fetch problems one by one. Send them to the local server. Parse the result. |
| **Stage 4 (Teardown)** | Ensure clean exit to avoid hanging processes. |

---

## 4. Full Instruction: Hosting Llama-3.3-Nemotron-Super-49B on H100

This section provides a comprehensive, reproducible guide to hosting the model. It assumes you are working within a Kaggle Notebook attached to an H100 accelerator.

### 4.1 Step 1: Dataset Preparation (The "Cold Storage")

Before opening the competition notebook, you must prepare your assets.

**Create a Local Script**: On your local machine (with internet), use the `huggingface_hub` library to snapshot the model.

```python
from huggingface_hub import snapshot_download

# Download the FP8 version to save download time and ensure H100 compatibility
snapshot_download(repo_id="nvidia/Llama-3.3-Nemotron-Super-49B-v1-FP8", local_dir="./nemotron_49b_fp8")
snapshot_download(repo_id="nvidia/llama-embed-nemotron-8b", local_dir="./nemotron_embed_8b")
```

**Download Wheels**: Download the vllm wheel compatible with the Kaggle environment (usually CUDA 12.1 or 11.8). You can find pre-compiled wheels or download them using `pip download vllm --dest ./wheels`.

**Upload to Kaggle**: Create a new Kaggle Dataset (e.g., `aimo3-nemotron-assets`) and upload these folders.

### 4.2 Step 2: Notebook Setup and Installation

In your competition notebook, attach the dataset. The first cell should handle the offline installation.

```python
import os
import subprocess
import sys
import time
import pandas as pd
import polars as pl
from pathlib import Path

# Configuration Constants
# Adjust these paths to match where your dataset is mounted
MODEL_PATH = "/kaggle/input/aimo3-nemotron-assets/nemotron_49b_fp8"
EMBED_MODEL_PATH = "/kaggle/input/aimo3-nemotron-assets/nemotron_embed_8b"
WHEELS_PATH = "/kaggle/input/aimo3-nemotron-assets/wheels"

def install_offline_packages():
    """Installs vLLM and dependencies from local wheel files."""
    print("Installing dependencies from local wheels...")
    # --no-index ensures pip doesn't try to reach PyPI
    # --find-links points to the directory containing our .whl files
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "--no-index",
        "--find-links", WHEELS_PATH,
        "vllm"
    ])
    print("Installation complete.")

# Trigger installation
install_offline_packages()
```

### 4.3 Step 3: The NIMModelDeployer Implementation

We implement a class that manages the lifecycle of the vLLM server. This encapsulates the logic of `NIMModelDeployer.py`, adapting it for the H100 constraints.

```python
class NIMModelDeployer:
    def __init__(self, model_path, port=8000):
        self.model_path = model_path
        self.port = port
        self.process = None
        self.client = None
        self.api_base = f"http://localhost:{self.port}/v1"

    def start_server(self):
        """
        Launches the vLLM server in a subprocess. 
        This mimics the 'docker run' behavior of a NIM but in user space.
        """
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--dtype", "float8",           # Enable FP8 on H100
            "--tensor-parallel-size", "1", # Single GPU
            "--max-model-len", "8192",     # Reasonable context for math problems
            "--gpu-memory-utilization", "0.95",
            "--trust-remote-code"
        ]
        
        print(f"Starting Server with command: {' '.join(cmd)}")
        # Redirect stdout/stderr to devnull to keep notebook clean, or capture for debug
        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("Waiting for server to initialize...")
        self._wait_for_health_check()
        
        # Initialize the OpenAI client wrapper
        from openai import OpenAI
        self.client = OpenAI(base_url=self.api_base, api_key="EMPTY")
        print("Server successfully initialized and ready for inference.")

    def _wait_for_health_check(self, timeout=600):
        """Polls the server health endpoint until it returns 200 OK."""
        import requests
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("vLLM server failed to start within timeout.")
            try:
                # vLLM exposes a /health endpoint
                response = requests.get(f"http://localhost:{self.port}/health")
                if response.status_code == 200:
                    break
            except requests.exceptions.ConnectionError:
                pass  # Server not yet listening
            time.sleep(5)

    def generate_reasoning(self, prompt, temperature=0.6):
        """
        Generates a response using the Nemotron Reasoning Mode.
        """
        try:
            # The system prompt is the switch for Reasoning Mode
            messages = [
                {
                    "role": "system",
                    "content": "You are a mathematical reasoning assistant. Reasoning mode: ON. "
                               "Think step-by-step within <think> tags before providing your final answer."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                max_tokens=4096,  # Allocate budget for verbose <think> traces
                stop=["<|eot_id|>"]  # Ensure clean stopping
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Inference Error: {e}")
            return ""

    def stop(self):
        """Terminates the background server process."""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("Server stopped.")
```

### 4.4 Step 4: The Evaluation Loop

Finally, we connect this deployer to the competition's evaluation API.

```python
import kaggle_evaluation.aimo_3_inference_server as aimo_api

def main():
    # Initialize the "Local NIM"
    deployer = NIMModelDeployer(MODEL_PATH)
    deployer.start_server()
    
    # Define the prediction callback required by the API
    def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
        problem_text = problem.item(0)
        uid = id_.item(0)
        
        # 1. (Optional) Retrieval Step would go here (See Section 5)
        
        # 2. Prompt Engineering
        full_prompt = f"Problem: {problem_text}\n\nSolve this carefully."
        
        # 3. Inference
        # We might use a higher temperature to encourage diverse reasoning paths
        raw_output = deployer.generate_reasoning(full_prompt)
        
        # 4. Answer Extraction (Critical Step)
        # Parse the integer from the \boxed{} output
        final_answer = extract_integer_answer(raw_output)
        
        return pl.DataFrame({'id': [uid], 'answer': [final_answer]})

    # Start the Kaggle Gateway
    inference_server = aimo_api.AIMO3InferenceServer(predict)

    # Differentiate between local testing and actual submission
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        # Local test mode using public data
        inference_server.run_local_gateway(
            ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
        )
    
    # Cleanup
    deployer.stop()

def extract_integer_answer(text):
    """
    Robustly extracts the final integer answer from the model output.
    Looks for \\boxed{...} and cleans up latex formatting.
    """
    import re
    # Pattern to find boxed content
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        # Take the last boxed answer as the final one
        candidate = matches[-1]
        # Remove non-digit characters (e.g., if it outputted "7\pi" or "approx 7")
        # For AIMO, answers are non-negative integers.
        digits = re.sub(r'\D', '', candidate)
        if digits:
            return int(digits) % 100000  # Modulo 100k per some AIMO rules
    return 0  # Fallback

if __name__ == "__main__":
    main()
```

---

## 5. Integrating RAG: Generating Embeddings with Llama-Embed-Nemotron-8B

The user query specifically requests a step to generate per-record embeddings using `nvidia/llama-embed-nemotron-8b` and references `extract_embeddings_parallel_shards.py`. This is part of a Retrieval-Augmented Generation (RAG) strategy where the model retrieves similar solved problems to use as few-shot examples.

### 5.1 Challenges of Concurrent Embedding on H100

Loading both a 49B generation model and an 8B embedding model into the same GPU memory is technically feasible on an 80GB H100 if quantization is used.

| Model | Precision | VRAM Usage |
|-------|-----------|------------|
| 49B (FP8) | 8-bit | ~50GB |
| 8B (FP16) | 16-bit | ~16GB |
| **Total** | - | **~66GB** |
| **Remaining** | - | ~14GB for KV Cache and Activations |

This is tight but manageable. Alternatively, one can use 4-bit quantization for the embedding model to reduce it to ~5GB, creating massive headroom.

### 5.2 Embedding Extraction Code

The following class mimics the logic of `extract_embeddings_parallel_shards.py`, providing a method to encode incoming queries. This code uses the `transformers` library rather than vLLM for the embedding part, as vLLM's embedding support can be heavier to configure alongside a generation server.

```python
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

class EmbeddingExtractor:
    def __init__(self, model_path):
        print("Initializing Embedding Model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load in 4-bit to save VRAM for the main 49B model
        # This requires bitsandbytes to be installed
        self.model = AutoModel.from_pretrained(
            model_path, 
            device_map="cuda", 
            torch_dtype=torch.float16,
            load_in_4bit=True 
        )
        self.model.eval()

    def get_embedding(self, text, is_query=True):
        """
        Generates embeddings using the specific instruction template required by Nemotron.
        """
        # The prompt template is strictly defined for this model
        if is_query:
            instruction = "Retrieve math problems similar to the query."
            formatted_text = f"Instruct: {instruction}\nQuery: {text}"
        else:
            # For documents/passages, no instruction is needed
            formatted_text = text
        
        inputs = self.tokenizer(
            formatted_text, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=512  # Standard context length for embedding models
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model(**inputs)
            # The Nemotron embedding model uses the last token's hidden state 
            # (EOS token pooling) to represent the sequence.
            embeddings = self._last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
        # Normalize embeddings for cosine similarity
        return F.normalize(embeddings, p=2, dim=1)

    def _last_token_pool(self, last_hidden_states, attention_mask):
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Usage in the main loop:
# embedder = EmbeddingExtractor(EMBED_MODEL_PATH)
# vector = embedder.get_embedding(problem_text)
# Use vector to query a local Faiss index of the training set
```

---

## 6. Advanced Prompt Engineering and Cognitive Strategies

Winning AIMO 3 requires more than just deploying the model; it requires coercing the Llama-3.3-Nemotron-Super-49B into a specific cognitive stance that minimizes hallucination and maximizes logical rigor. The "Reasoning Mode" is the primary lever for this.

### 6.1 The "System 2" Verification Prompt

This prompt template forces the model to engage in what cognitive scientists call "System 2" thinking—slow, deliberative, and verifiable. It leverages the `<think>` verification signal native to the Nemotron architecture.

**Template Structure:**

```
System: You are an expert mathematician participating in the International Mathematical Olympiad. Reasoning mode: ON.

Your goal is to solve the given problem accurately. Follow this strict protocol:

1. **Exploration**: First, explore the problem space within <think> tags. Break the problem down into sub-components.
2. **Step-by-Step Execution**: Perform calculations explicitly.
3. **Self-Correction**: After every major step, verify the intermediate result. If you encounter a contradiction, backtrack and try a different approach.
4. **Formatting**: Output the final answer as a non-negative integer between 0 and 99999 inside \boxed{}. Do not output fractions or equations in the box.

User: {problem_text}

Reasoning:
```

**Key Design Choices:**

- **"Reasoning mode: ON"**: This phrase is the activation key that switches the model's internal attention patterns to the RLHF-tuned reasoning distribution.
- **"Backtrack"**: Explicitly encouraging backtracking reduces the model's tendency to "double down" on early arithmetic errors, a common failure mode in CoT.
- **Constraint Reinforcement**: Reminding the model of the "0-99999" constraint acts as a heuristic filter, pruning impossible answers (like negatives or irrationals) early in the generation process.

### 6.2 The "Program of Thought" (PoT) Meta-Prompt

For combinatorics or number theory problems where the logic is sound but the arithmetic is prone to token-prediction errors (e.g., multiplying large primes), a **Program of Thought** strategy is superior. This involves prompting the model to write Python code rather than English text.

**Template Structure:**

```
System: You are a computational mathematician. To solve the problem, write a robust Python script that calculates the answer.

1. Define the variables and constraints clearly.
2. Implement the logic in a function.
3. Print the final answer at the end of the script.
4. Do not guess; calculate.

User: {problem_text}

Reasoning:
```

While the Kaggle inference server prevents executing code on the fly to feed back into the model (unless you build a complex re-entrant loop), simply asking the model to **write code forces it to formalize its logic**. The act of writing valid Python syntax acts as a regularizer for the model's reasoning. Even if you cannot execute it, the model's internal simulation of the code execution often yields a more accurate result than pure natural language generation.

### 6.3 The "Self-Consistency" Ensemble Prompt

If time permits (within the 5-hour budget), the most effective strategy is **Self-Consistency**.

1. Run the System 2 prompt **5 times** with `temperature=0.7` (higher entropy).
2. Run the PoT prompt **5 times**.
3. Extract all answers.
4. Perform a **majority vote**.

This leverages the "Wisdom of the Crowds" effect on the model's own distribution, significantly boosting performance on high-variance problems.

---

## 7. Strategic Conclusions and Future Outlook

The deployment of Llama-3.3-Nemotron-Super-49B on a single H100 for the AIMO 3 competition represents a convergence of cutting-edge model architecture and extreme system optimization.

### Key Takeaways

| Aspect | Insight |
|--------|---------|
| **Architecture-Hardware Fit** | The 49B parameter size is the exact upper bound for FP8 inference on an 80GB card, making it the most powerful single-device reasoner available. |
| **Deployment Rigor** | Success depends on the robust implementation of a "Local NIM" using subprocess and vLLM, strictly managing dependencies via offline wheels. |
| **Reasoning vs. Retrieval** | While the 49B model is potent, augmenting it with an 8B embedding model (running in 4-bit) provides the necessary "long-term memory" of mathematical theorems without exceeding VRAM limits. |

### Future Implications

The techniques described here—offline "NIM" construction, FP8 utilization, and hybrid reasoning/coding prompts—are not limited to this competition. They represent a **blueprint for deploying sovereign, high-reasoning AI agents in air-gapped or secure environments** where external API dependencies are unacceptable. As models like Nemotron continue to specialize via NAS, the ability to tailor deployment pipelines to specific hardware envelopes will become the primary differentiator in applied AI engineering.

