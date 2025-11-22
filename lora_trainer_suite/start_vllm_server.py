#!/usr/bin/env python3
"""
vLLM Server Launcher for Qwen VL Models
Simplified launcher for running vLLM with Qwen vision models
"""

import argparse
import subprocess
import sys
import os
import signal


def launch_vllm_server(
    model: str,
    port: int = 8000,
    host: str = "0.0.0.0",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
):
    """
    Launch vLLM server for vision-language models

    Args:
        model: HuggingFace model name
        port: Server port
        host: Server host
        max_model_len: Maximum model context length
        gpu_memory_utilization: GPU memory utilization fraction (0.0-1.0)
        dtype: Model dtype (bfloat16, float16, float32)
        tensor_parallel_size: Number of GPUs for tensor parallelism
    """

    # Build vLLM command
    cmd = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--trust-remote-code",
    ]

    print("=" * 80)
    print("vLLM Server Launcher for Qwen VL Models")
    print("=" * 80)
    print(f"\nModel: {model}")
    print(f"Server: http://{host}:{port}")
    print(f"GPU Memory: {gpu_memory_utilization * 100}%")
    print(f"Max Context: {max_model_len} tokens")
    print(f"Dtype: {dtype}")
    print(f"Tensor Parallel: {tensor_parallel_size} GPU(s)")
    print("\nCommand:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)
    print("\nStarting server... (Press Ctrl+C to stop)\n")

    try:
        # Launch server
        process = subprocess.Popen(cmd)

        # Handle Ctrl+C gracefully
        def signal_handler(sig, frame):
            print("\n\nShutting down vLLM server...")
            process.terminate()
            process.wait()
            print("Server stopped.")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Wait for process
        process.wait()

    except FileNotFoundError:
        print("Error: vLLM not found. Install with: pip install vllm>=0.3.0")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Launch vLLM server for Qwen VL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default Qwen2.5-VL
  python start_vllm_server.py

  # Qwen3-VL (if available)
  python start_vllm_server.py --model Qwen/Qwen3-VL-8B

  # Abliterated version
  python start_vllm_server.py --model mlabonne/Qwen2-VL-7B-Instruct-abliterated

  # Custom port and memory
  python start_vllm_server.py --port 8001 --gpu-memory 0.7

  # Multi-GPU (if you have multiple GPUs)
  python start_vllm_server.py --tensor-parallel 2
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-VL-7B-Instruct)"
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )

    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.8,
        help="GPU memory utilization 0.0-1.0 (default: 0.8)"
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype (default: bfloat16)"
    )

    parser.add_argument(
        "--tensor-parallel",
        type=int,
        default=1,
        help="Tensor parallel size - number of GPUs (default: 1)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.gpu_memory < 0.0 or args.gpu_memory > 1.0:
        print("Error: --gpu-memory must be between 0.0 and 1.0")
        sys.exit(1)

    # Launch server
    launch_vllm_server(
        model=args.model,
        port=args.port,
        host=args.host,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel,
    )


if __name__ == "__main__":
    main()
