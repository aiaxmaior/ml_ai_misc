"""
LLM Benchmark Suite
Supports HuggingFace transformers, TensorRT-LLM engines, and API endpoints
Author: Arjun Joshi
Date: 2024-10-05
"""

import os
import argparse
import sys
import subprocess
from pathlib import Path
import requests
import json

# Multiple GPU Setup (for local models only)
def setup_gpu():
    print("Available GPUs:")
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpu_list = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, name = line.split(', ', 1)
                gpu_list.append((idx.strip(), name.strip()))
                print(f"  {idx.strip()}) {name.strip()}")
        if len(gpu_list)>1:
            gpu_choice = input("\nSelect GPU ID for quantization (default 0): ").strip()
        
        gpu_id = gpu_choice if gpu_choice else "0"
        
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES={gpu_id}\n")
        
    except Exception as e:
        print(f"[WARNING] Could not query GPUs: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# Parse args FIRST before any CUDA imports
parser = argparse.ArgumentParser(description="Benchmark Q-DRIVE LLM Model")
parser.add_argument("--model_path", type=str,
                   default="~/qdrive_alpha/LLM/model/Phi-3-mini-128k-instruct",
                   help="Path to model directory (HF) or engine directory (TRT)")
parser.add_argument("--model_name", type=str, default="FP16 Baseline",
                   help="Name/label for this model")
parser.add_argument("--device", type=str, default="cuda",
                   help="Device to run the model on")
parser.add_argument("--output_file", type=str,
                   default="qdrive_model_benchmark.txt",
                   help="File to save benchmark results")
parser.add_argument("--gpu_id", type=int, default=0,
                   help="GPU ID to use")
parser.add_argument("--append", action="store_true",
                   help="Append to existing output file instead of overwriting")
parser.add_argument("--use_trt", action="store_true",
                    help="Use TensorRT-LLM engine instead of HuggingFace model")
parser.add_argument("--tokenizer_path", type=str,
                    default="~/qdrive_alpha/LLM/model/Phi-3-mini-128k-instruct",
                    help="Path to tokenizer (for TRT mode)")
parser.add_argument("--skip_throughput", action="store_true", 
                    help="Skip throughput measurement")

# New API-related arguments
parser.add_argument("--use_api", action="store_true",
                    help="Use API endpoint instead of local model")
parser.add_argument("--api_url", type=str,
                    default="http://192.168.68.54:8080/v1/completions",
                    help="API endpoint URL (OpenAI-compatible format)")
parser.add_argument("--api_type", type=str, 
                    choices=["openai", "vllm", "ollama", "tgi", "custom"],
                    default="openai",
                    help="API type/format")
parser.add_argument("--api_key", type=str, default=None,
                    help="API key if required")
parser.add_argument("--api_model", type=str, default=None,
                    help="Model name for API calls (e.g., 'llama2', 'phi-3-mini')")

args = parser.parse_args()

# Only setup GPU for local models
if not args.use_api:
    setup_gpu()
    # Set CUDA_VISIBLE_DEVICES before any CUDA/torch imports
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Now import CUDA-dependent libraries (only if needed)
import time
import psutil
import numpy as np
import math
import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Conditional imports
if not args.use_api:
    import torch
    import GPUtil
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    
from rouge_score import rouge_scorer


class APIBenchmark:
    """Benchmark for models served through API endpoints"""
    
    def __init__(self, api_url, api_type="openai", api_key=None, model_name=None):
        self.api_url = api_url
        self.api_type = api_type
        self.api_key = api_key
        self.model_name = model_name
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test if API is reachable"""
        try:
            response = self._make_request("Test connection", max_tokens=10)
            print(f"[INFO] API connection successful to {self.api_url}")
        except Exception as e:
            print(f"[ERROR] Failed to connect to API: {e}")
            sys.exit(1)
    
    def _make_request(self, prompt, max_tokens=50, temperature=0.7):
        """Make API request based on API type"""
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        if self.api_type in ["openai", "vllm", "tgi"]:
            # OpenAI-compatible format
            data = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if self.model_name:
                data["model"] = self.model_name
                
        elif self.api_type == "ollama":
            # Ollama format
            data = {
                "model": self.model_name or "llama2",
                "prompt": prompt,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                }
            }
            # Adjust URL for Ollama
            if "/v1/completions" in self.api_url:
                self.api_url = self.api_url.replace("/v1/completions", "/api/generate")
                
        elif self.api_type == "custom":
            # Generic/custom format
            data = {
                "text": prompt,
                "max_length": max_tokens,
                "temperature": temperature,
            }
        
        response = requests.post(self.api_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text based on API type
        if self.api_type in ["openai", "vllm", "tgi"]:
            if "choices" in result:
                return result["choices"][0]["text"]
            else:
                return result.get("text", "")
        elif self.api_type == "ollama":
            return result.get("response", "")
        else:
            return result.get("generated_text", result.get("text", ""))
    
    def measure_perplexity(self, dataset_name="wikitext", num_samples=100):
        """API endpoints typically don't support perplexity calculation"""
        print("  Perplexity measurement not supported for API endpoints")
        return None
    
    def measure_latency(self, prompts, num_tokens=50, warmup=5):
        """Measure inference latency via API"""
        
        # Warmup
        print("  Warming up API...")
        for _ in range(warmup):
            try:
                _ = self._make_request(prompts[0], max_tokens=num_tokens)
            except:
                pass
        
        user_input = input("Do you want to measure latency now? (y/n): ")
        if user_input.lower() != 'y':
            print("Latency measurement skipped.")
            return {
                "mean_ms": 0,
                "median_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "std_ms": 0
            }
        
        latencies = []
        for prompt in prompts:
            start = time.perf_counter()
            
            try:
                _ = self._make_request(prompt, max_tokens=num_tokens)
            except Exception as e:
                print(f"  Request failed: {e}")
                continue
            
            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            print(f"  Request completed in {latency_ms:.1f}ms")
        
        if not latencies:
            return {
                "mean_ms": 0,
                "median_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "std_ms": 0
            }
        
        return {
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "std_ms": np.std(latencies)
        }
    
    def measure_throughput(self, prompt, duration_seconds=60):
        """Measure tokens/second throughput via API"""
        
        user_input = input("Do you want to measure throughput now? (y/n): ")
        if user_input.lower() != 'y':
            print("Throughput measurement skipped.")
            return 0
        
        tokens_generated = 0
        requests_completed = 0
        start_time = time.time()
        
        print(f"  Running throughput test for {duration_seconds} seconds...")
        while (time.time() - start_time) < duration_seconds:
            try:
                response = self._make_request(prompt, max_tokens=50)
                # Rough token estimation (words * 1.3)
                tokens_generated += len(response.split()) * 1.3
                requests_completed += 1
            except Exception as e:
                print(f"  Request failed: {e}")
                continue
        
        elapsed = time.time() - start_time
        throughput = tokens_generated / elapsed if elapsed > 0 else 0
        
        print(f"  Completed {requests_completed} requests")
        return throughput
    
    def measure_resource_utilization(self, prompt, duration_seconds=10, gpu_id=0):
        """Monitor local CPU usage during API inference (GPU not applicable)"""
        
        cpu_utils = []
        request_times = []
        
        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            req_start = time.time()
            try:
                _ = self._make_request(prompt, max_tokens=50)
            except:
                pass
            request_times.append(time.time() - req_start)
            cpu_utils.append(psutil.cpu_percent(interval=0.1))
        
        return {
            "gpu_utilization_mean": 0,  # N/A for API
            "gpu_memory_mb_mean": 0,     # N/A for API
            "gpu_memory_mb_peak": 0,      # N/A for API
            "cpu_utilization_mean": np.mean(cpu_utils) if cpu_utils else 0,
            "avg_request_time_s": np.mean(request_times) if request_times else 0
        }
    
    def measure_rouge(self, test_cases):
        """Measure ROUGE scores for Q-DRIVE coaching quality via API"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for prompt, reference in test_cases:
            try:
                generated = self._make_request(prompt, max_tokens=100)
                
                scores = scorer.score(reference, generated)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception as e:
                print(f"  Failed to get response for ROUGE: {e}")
                continue
        
        if not rouge1_scores:
            return {"rouge1": 0, "rouge2": 0, "rougeL": 0}
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores)
        }


class ModelBenchmark:
    """Benchmark for HuggingFace transformers models (FP16, INT8, etc.)"""
    
    def __init__(self, model_path, device="cuda", gpu_id=0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            os.path.expanduser(model_path),
            device_map=device, 
            torch_dtype=torch.float16,
            trust_remote_code=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.expanduser(model_path),
            trust_remote_code=False
        )
        self.device = device
        
    def measure_perplexity(self, dataset_name="wikitext", num_samples=100):
        """Measure perplexity on standard dataset"""
        import torch
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(dataset["text"][:num_samples])
        
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = encodings.input_ids.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = math.exp(loss.item())
        
        return perplexity
    
    def measure_latency(self, prompts, num_tokens=50, warmup=5):
        """Measure inference latency"""
        import torch
        
        # Warmup
        for _ in range(warmup):
            inputs = self.tokenizer(prompts[0], return_tensors="pt").to(self.device)
            self.model.generate(**inputs, max_new_tokens=num_tokens, use_cache=True)
        
        latencies = []
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = self.model.generate(**inputs, max_new_tokens=num_tokens, use_cache=True)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        return {
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "std_ms": np.std(latencies)
        }
    
    def measure_throughput(self, prompt, duration_seconds=60):
        """Measure tokens/second throughput"""
        user_input = input("Do you want to measure throughput now? (y/n): ")
        if user_input.lower() != 'y':
            print("Throughput measurement skipped.")
            return 0
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        tokens_generated = 0
        start_time = time.time()
        
        while (time.time() - start_time) < duration_seconds:
            outputs = self.model.generate(**inputs, max_new_tokens=50, use_cache=True)
            tokens_generated += outputs.shape[1] - inputs.input_ids.shape[1]
        
        elapsed = time.time() - start_time
        throughput = tokens_generated / elapsed
        
        return throughput
    
    def measure_resource_utilization(self, prompt, duration_seconds=10, gpu_id=0):
        """Monitor GPU and CPU usage during inference"""
        import GPUtil
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        gpu_utils = []
        gpu_mems = []
        cpu_utils = []
        
        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            self.model.generate(**inputs, max_new_tokens=50, use_cache=True)
            
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > gpu_id:
                gpu_utils.append(gpus[gpu_id].load * 100)
                gpu_mems.append(gpus[gpu_id].memoryUsed)
            cpu_utils.append(psutil.cpu_percent(interval=0.1))
        
        return {
            "gpu_utilization_mean": np.mean(gpu_utils) if gpu_utils else 0,
            "gpu_memory_mb_mean": np.mean(gpu_mems) if gpu_mems else 0,
            "gpu_memory_mb_peak": np.max(gpu_mems) if gpu_mems else 0,
            "cpu_utilization_mean": np.mean(cpu_utils)
        }
    
    def measure_rouge(self, test_cases):
        """Measure ROUGE scores for Q-DRIVE coaching quality"""
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for prompt, reference in test_cases:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=100, use_cache=True)
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the generated response (remove prompt)
            generated = generated[len(prompt):].strip()
            
            scores = scorer.score(reference, generated)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            "rouge1": np.mean(rouge1_scores),
            "rouge2": np.mean(rouge2_scores),
            "rougeL": np.mean(rougeL_scores)
        }


class TRTLLMBenchmark:
    """Benchmark for TensorRT-LLM engines"""
    
    def __init__(self, engine_dir, tokenizer_path, gpu_id=0):
        from tensorrt_llm.runtime import ModelRunnerCpp
        from transformers import AutoTokenizer
        import torch

        # Set the GPU device before initializing TRT runner
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        self.runner = ModelRunnerCpp.from_dir(
            engine_dir=os.path.expanduser(engine_dir),
            rank=0
        )
        # Load tokenizer from original model
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.expanduser(tokenizer_path),
            trust_remote_code=False
        )
        
    def measure_perplexity(self, dataset_name="wikitext", num_samples=100):
        """TRT-LLM doesn't support perplexity calculation"""
        print("  Perplexity measurement not supported for TRT engines (requires loss computation)")
        return None
    
    def measure_latency(self, prompts, num_tokens=50, warmup=5):
        """Measure inference latency for TRT engine"""

        # Warmup
        for _ in range(warmup):
            input_ids = self.tokenizer.encode(prompts[0])
            _ = self.runner.generate(
                batch_input_ids=[np.array(input_ids, dtype=np.int32)],
                max_new_tokens=num_tokens,
                end_id=self.tokenizer.eos_token_id or 2,
                pad_id=self.tokenizer.pad_token_id or 0
            )
        user_input = input("Do you want to measure latency now? (y/n): ")
        if user_input.lower() != 'y':
            print("Latency measurement skipped.")
            return {
                "mean_ms": 0,
                "median_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "std_ms": 0
            }
                
        latencies = []
        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt)

            start = time.perf_counter()

            _ = self.runner.generate(
                batch_input_ids=[np.array(input_ids, dtype=np.int32)],
                max_new_tokens=num_tokens,
                end_id=self.tokenizer.eos_token_id or 2,
                pad_id=self.tokenizer.pad_token_id or 0
            )

            end = time.perf_counter()
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
        
        return {
            "mean_ms": np.mean(latencies),
            "median_ms": np.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "std_ms": np.std(latencies)
        }
    
    def measure_throughput(self, prompt, duration_seconds=60):
        """Measure tokens/second throughput"""
        user_input = input("Do you want to measure throughput now? (y/n): ")
        if user_input.lower() != 'y':
            print("Throughput measurement skipped.")
            return 0
            
        input_ids = self.tokenizer.encode(prompt)
        input_ids_array = np.array(input_ids, dtype=np.int32)

        tokens_generated = 0
        start_time = time.time()

        while (time.time() - start_time) < duration_seconds:
            _ = self.runner.generate(
                batch_input_ids=[input_ids_array],
                max_new_tokens=50,
                end_id=self.tokenizer.eos_token_id or 2,
                pad_id=self.tokenizer.pad_token_id or 0
            )
            tokens_generated += 50
        
        elapsed = time.time() - start_time
        throughput = tokens_generated / elapsed
        
        return throughput
    
    def measure_resource_utilization(self, prompt, duration_seconds=10, gpu_id=0):
        """Monitor GPU and CPU usage during inference"""
        import GPUtil
        
        input_ids = self.tokenizer.encode(prompt)
        input_ids_array = np.array(input_ids, dtype=np.int32)

        gpu_utils = []
        gpu_mems = []
        cpu_utils = []

        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            self.runner.generate(
                batch_input_ids=[input_ids_array],
                max_new_tokens=50,
                end_id=self.tokenizer.eos_token_id or 2,
                pad_id=self.tokenizer.pad_token_id or 0
            )
            
            gpus = GPUtil.getGPUs()
            if gpus and len(gpus) > gpu_id:
                gpu_utils.append(gpus[gpu_id].load * 100)
                gpu_mems.append(gpus[gpu_id].memoryUsed)
            cpu_utils.append(psutil.cpu_percent(interval=0.1))
        
        return {
            "gpu_utilization_mean": np.mean(gpu_utils) if gpu_utils else 0,
            "gpu_memory_mb_mean": np.mean(gpu_mems) if gpu_mems else 0,
            "gpu_memory_mb_peak": np.max(gpu_mems) if gpu_mems else 0,
            "cpu_utilization_mean": np.mean(cpu_utils)
        }
    
    def measure_rouge(self, test_cases):
        """
        Measure ROUGE scores against reference responses
        """
        from rouge_score import rouge_scorer
        import torch
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for prompt, reference in test_cases:
            # Tokenize input
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to('cuda')
            
            # Generate
            output_ids = self.runner.generate(
                input_ids,
                max_new_tokens=100,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Handle different output formats
            if isinstance(output_ids, torch.Tensor):
                output_ids = output_ids[0].tolist()
            elif isinstance(output_ids, list):
                if output_ids and isinstance(output_ids[0], list):
                    output_ids = output_ids[0]
            
            try:
                generated = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            except TypeError:
                output_ids = [int(x) for x in output_ids]
                generated = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            
            generated = generated.replace(prompt, "").strip()
            
            scores = scorer.score(reference, generated)
            all_scores['rouge1'].append(scores['rouge1'].fmeasure)
            all_scores['rouge2'].append(scores['rouge2'].fmeasure)
            all_scores['rougeL'].append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': sum(all_scores['rouge1']) / len(all_scores['rouge1']),
            'rouge2': sum(all_scores['rouge2']) / len(all_scores['rouge2']),
            'rougeL': sum(all_scores['rougeL']) / len(all_scores['rougeL']),
        }


# Q-DRIVE test cases
qdrive_test_prompts = [
    "Driver exceeded safe following distance. Speed: 65mph. Following distance: 1.2s. Provide brief coaching tip:",
    "Lane departure detected without signal. Lateral offset: 0.8m. Speed: 45mph. Generate feedback:",
    "Hard braking event. Deceleration: 0.6g. Speed before: 50mph. Location: approaching intersection. Coaching:",
]

qdrive_reference_responses = [
    "Increase your following distance to at least 2 seconds. At 65mph, maintain approximately 190 feet from the vehicle ahead.",
    "You drifted out of your lane without signaling. Always check mirrors and signal before changing lanes.",
    "Hard braking detected approaching the intersection. Anticipate stops earlier and brake gradually."
]


if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Helper functions
    def get_unspecified_dir():
        """Find next available unspecified directory name"""
        base_path = Path("./unspecified")
        
        if not base_path.exists():
            return str(base_path)
        
        count = 1
        while True:
            new_path = Path(f"./unspecified_{count}")
            if not new_path.exists():
                return str(new_path)
            count += 1

    def prompt_for_model_path(default_path):
        while True:
            inp = input(f"Model Path [{default_path}] (or 'check <path>'): ").strip()
            if not inp: return default_path
            if inp.startswith('check'):
                p = Path(inp.split(maxsplit=1)[1] if ' ' in inp else '.').expanduser()
                [print(f"  {item.name}") for item in p.iterdir()] if p.exists() else print(f"Not found: {p}")
            elif (path := Path(inp).expanduser()).exists(): return str(path)
            else: print(f"Not found: {path}")
    
    # Model Data Source Selection - Always ask unless explicitly specified
    model_source = None
    
    # Check if source was explicitly specified via command line
    if args.use_api:
        model_source = "api"
    elif args.use_trt:
        model_source = "trt"
    elif args.model_path and not args.model_path.startswith("~"):  # If path was explicitly provided
        # Still ask for source type
        model_source = None
    
    # Interactive model source selection
    if model_source is None:
        print("\n" + "="*60)
        print("Q-DRIVE LLM Benchmark Suite")
        print("="*60)
        print("\nSelect Model Data Source:")
        print("1) HuggingFace Transformers (FP16, INT8, etc.)")
        print("2) TensorRT-LLM Engine")
        print("3) API Endpoint (vLLM, Ollama, TGI, etc.)")
        print("-"*40)
        
        while True:
            source_input = input("Enter your choice [1-3]: ").strip()
            if source_input == "1":
                model_source = "hf"
                args.use_api = False
                args.use_trt = False
                print("\n→ Selected: HuggingFace Transformers")
                break
            elif source_input == "2":
                model_source = "trt"
                args.use_api = False
                args.use_trt = True
                print("\n→ Selected: TensorRT-LLM Engine")
                break
            elif source_input == "3":
                model_source = "api"
                args.use_api = True
                args.use_trt = False
                print("\n→ Selected: API Endpoint")
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Configure based on model source
    if model_source == "api":
        # API Mode Setup
        print("\n" + "="*60)
        print("API Endpoint Configuration")
        print("="*60)
        
        if not args.api_url:
            print("\nCommon API endpoints:")
            print("  • vLLM:     http://localhost:8000/v1/completions")
            print("  • Ollama:   http://localhost:11434/api/generate")
            print("  • TGI:      http://localhost:8080/v1/completions")
            print("  • Custom:   http://your-server:port/endpoint")
            args.api_url = input("\nAPI URL [http://localhost:8000/v1/completions]: ").strip()
            args.api_url = args.api_url if args.api_url else "http://localhost:8000/v1/completions"
        
        if not args.api_type:
            print("\nSelect API Type:")
            print("1) OpenAI-compatible (vLLM, TGI, text-generation-webui)")
            print("2) Ollama (for GGUF models)")
            print("3) Custom format")
            type_choice = input("Enter choice [1]: ").strip()
            type_map = {"1": "openai", "2": "ollama", "3": "custom", "": "openai"}
            args.api_type = type_map.get(type_choice, "openai")
        
        if not args.api_model:
            args.api_model = input("\nModel name for API (e.g., 'llama2', 'phi-3-mini') [optional]: ").strip()
            args.api_model = args.api_model if args.api_model else None
        
        if not args.api_key:
            key_input = input("API key [press Enter if none]: ").strip()
            args.api_key = key_input if key_input else None
        
        if not args.model_name:
            args.model_name = input("\nModel label for benchmark results [API Model]: ").strip()
            args.model_name = args.model_name if args.model_name else "API Model"
        
        if not args.api_url:
            args.api_url = input("Enter the API url including port [default uses port http://127.0.0.1:8000]")
            args.api_url = args.api_url if args.api_url else None
    elif model_source == "hf":
        # HuggingFace Transformers Setup
        print("\n" + "="*60)
        print("HuggingFace Transformers Configuration")
        print("="*60)
        
        if not args.model_path:
            print("\nExamples:")
            print("  • ~/models/Phi-3-mini-128k-instruct")
            print("  • /opt/models/Llama-2-7b-hf")
            print("  • ./quantized/model_int8")
            default_path = "~/qdrive_alpha/LLM/model/Phi-3-mini-128k-instruct"
            input_path = prompt_for_model_path(default_path)
            args.model_path = input_path if input_path else default_path
        
        # Validate model path exists
        model_path = Path(args.model_path).expanduser()
        if not model_path.exists():
            print(f"[ERROR] Model path does not exist: {model_path}")
            sys.exit(1)
        
        args.model_path = str(model_path)
        
        # Handle model_name
        if not args.model_name:
            suggested_name = model_path.name if model_path.name else "HF_Model"
            input_name = input(f"\nModel label for benchmark results [{suggested_name}]: ").strip()
            args.model_name = input_name if input_name else suggested_name
        
        # Set GPU (respect CUDA_VISIBLE_DEVICES if set)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_id)
                print(f"\n[INFO] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
            else:
                print("\n[WARNING] CUDA not available, using CPU")
        else:
            print(f"\n[INFO] Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
            
    elif model_source == "trt":
        # TensorRT-LLM Engine Setup
        print("\n" + "="*60)
        print("TensorRT-LLM Engine Configuration")
        print("="*60)
        
        if not args.model_path:
            print("\nTRT Engine Directory Examples:")
            print("  • ~/engines/phi3_mini_trt")
            print("  • /opt/trt_engines/llama2_int8")
            print("  • ./trt_output/engine")
            default_path = get_unspecified_dir()
            input_path = prompt_for_model_path(default_path)
            args.model_path = input_path if input_path else default_path
        
        # Validate engine path exists
        engine_path = Path(args.model_path).expanduser()
        if not engine_path.exists():
            print(f"[ERROR] Engine path does not exist: {engine_path}")
            sys.exit(1)
        
        args.model_path = str(engine_path)
        
        # Handle tokenizer_path (required for TRT)
        if not args.tokenizer_path:
            print("\nTokenizer path (original HF model with tokenizer files)")
            print("Examples:")
            print("  • ~/models/Phi-3-mini-128k-instruct")
            print("  • Same as engine directory if tokenizer.json exists there")
            
            # Check if tokenizer exists in engine directory
            if (engine_path / "tokenizer.json").exists() or (engine_path / "tokenizer_config.json").exists():
                default_tok_path = str(engine_path)
                print(f"\n[Found tokenizer in engine directory]")
            else:
                default_tok_path = "~/qdrive_alpha/LLM/model/Phi-3-mini-128k-instruct"
            
            tok_input = input(f"Tokenizer path [{default_tok_path}]: ").strip()
            args.tokenizer_path = tok_input if tok_input else default_tok_path
        
        # Handle model_name
        if not args.model_name:
            suggested_name = engine_path.name if engine_path.name else "TRT_Engine"
            input_name = input(f"\nModel label for benchmark results [{suggested_name}]: ").strip()
            args.model_name = input_name if input_name else suggested_name
        
        # Set GPU (respect CUDA_VISIBLE_DEVICES if set)
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu_id)
                print(f"\n[INFO] Using GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
            else:
                print("\n[WARNING] CUDA not available, using CPU")
        else:
            print(f"\n[INFO] Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Handle output_file
    if not args.output_file:
        safe_name = args.model_name.replace(' ', '_').replace('/', '_')
        args.output_file = f"./benchmark_{safe_name}_{timestamp}.txt"
    
    # Prepare output header
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mode = "a" if args.append else "w"
    
    print("\n" + "="*60)
    print("Starting Benchmark")
    print("="*60)
    
    output = []
    output.append("=" * 60)
    output.append("Q-DRIVE Model Benchmark Report")
    output.append("=" * 60)
    output.append(f"Timestamp: {now}")
    output.append(f"Model: {args.model_name}")
    
    if model_source == "api":
        output.append(f"Source: API Endpoint")
        output.append(f"Type: {args.api_type.upper()}")
        output.append(f"URL: {args.api_url}")
        if args.api_model:
            output.append(f"API Model: {args.api_model}")
    elif model_source == "hf":
        output.append(f"Source: HuggingFace Transformers")
        output.append(f"Path: {args.model_path}")
        output.append(f"Device: {args.device.upper()}")
        output.append(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', args.gpu_id)}")
    elif model_source == "trt":
        output.append(f"Source: TensorRT-LLM Engine")
        output.append(f"Engine Path: {args.model_path}")
        output.append(f"Tokenizer Path: {args.tokenizer_path}")
        output.append(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', args.gpu_id)}")
    
    output.append("=" * 60)
    
    for line in output:
        print(line)
    
    # Initialize benchmark based on source
    print(f"\nInitializing benchmark...")
    
    if model_source == "api":
        print(f"Connecting to API at {args.api_url}...")
        bench = APIBenchmark(
            api_url=args.api_url,
            api_type=args.api_type,
            api_key=args.api_key,
            model_name=args.api_model
        )
        print("✓ API connection established.\n")
    elif model_source == "hf":
        print(f"Loading HuggingFace model from {args.model_path}...")
        bench = ModelBenchmark(args.model_path, device=args.device, gpu_id=args.gpu_id)
        print("✓ Model loaded successfully.\n")
    elif model_source == "trt":
        print(f"Loading TensorRT-LLM engine from {args.model_path}...")
        bench = TRTLLMBenchmark(args.model_path, args.tokenizer_path, gpu_id=args.gpu_id)
        print("✓ Engine loaded successfully.\n")
    
    # Run benchmarks (same for all modes)
    
    # 1. Perplexity
    print("Measuring perplexity...")
    ppl = bench.measure_perplexity()
    if ppl is not None:
        result = f"Perplexity: {ppl:.2f}"
        print(result)
        output.append(result)
    else:
        output.append("Perplexity: N/A (Not supported)")
    
    # 2. Latency
    print("\nMeasuring latency...")
    latency = bench.measure_latency(qdrive_test_prompts)
    output.append("\nLatency (50 tokens):")
    output.append(f"  Mean:   {latency['mean_ms']:.1f}ms")
    output.append(f"  Median: {latency['median_ms']:.1f}ms")
    output.append(f"  P95:    {latency['p95_ms']:.1f}ms")
    output.append(f"  P99:    {latency['p99_ms']:.1f}ms")
    output.append(f"  StdDev: {latency['std_ms']:.1f}ms")
    
    if latency['p99_ms'] < 800:
        status = "  Status: MEETS Q-DRIVE <800ms requirement"
    else:
        status = "  Status: EXCEEDS Q-DRIVE requirement"
    output.append(status)
    
    for line in output[-7:]:
        print(line)
    
    # 3. Throughput
    if not args.skip_throughput:
        print("\nMeasuring throughput (60s test)...")
        throughput = bench.measure_throughput(qdrive_test_prompts[0], duration_seconds=60)
        result = f"\nThroughput: {throughput:.1f} tokens/sec"
        print(result)
        output.append(result)
    else:
        print("\nSkipping throughput measurement...")
        output.append("\nThroughput: SKIPPED")
    
    # 4. Resource Utilization
    print("\nMeasuring resource utilization...")
    resources = bench.measure_resource_utilization(
        qdrive_test_prompts[0], 
        duration_seconds=10,
        gpu_id=args.gpu_id if not args.use_api else 0
    )
    output.append("\nResource Utilization:")
    
    if args.use_api:
        output.append(f"  CPU Util: {resources['cpu_utilization_mean']:.1f}%")
        output.append(f"  Avg Request Time: {resources.get('avg_request_time_s', 0):.2f}s")
        output.append("  GPU metrics: N/A (remote API)")
    else:
        output.append(f"  GPU Util: {resources['gpu_utilization_mean']:.1f}%")
        output.append(f"  GPU Mem:  {resources['gpu_memory_mb_mean']:.0f}MB (peak: {resources['gpu_memory_mb_peak']:.0f}MB)")
        output.append(f"  CPU Util: {resources['cpu_utilization_mean']:.1f}%")
    
    for line in output[-4:] if not args.use_api else output[-3:]:
        print(line)
    
    # 5. ROUGE (Quality)
    print("\nMeasuring ROUGE scores...")
    test_cases = list(zip(qdrive_test_prompts, qdrive_reference_responses))
    rouge = bench.measure_rouge(test_cases)
    output.append("\nROUGE Scores (vs reference responses):")
    output.append(f"  ROUGE-1: {rouge['rouge1']:.3f}")
    output.append(f"  ROUGE-2: {rouge['rouge2']:.3f}")
    output.append(f"  ROUGE-L: {rouge['rougeL']:.3f}")
    
    for line in output[-4:]:
        print(line)
    
    output.append("=" * 60)
    output.append("")
    
    # Write results
    with open(args.output_file, mode) as f:
        f.write("\n".join(output) + "\n")
    
    print(f"\n[DONE] Benchmark complete. Results saved to {args.output_file}")