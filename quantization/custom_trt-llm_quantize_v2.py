"""
Author: Arjun Joshi
Date: 10.7.2025
Description: Quantize Using TensorRT-LLM 0.12.0, Default "Phi-3-mini-128k-instruct" Model
             Designed for Jetson/Ampere (W4A16 or W8A8 recommended)
Project: QDrive Ecosystem - Orin Jetson Nano Edge AI Platform
"""

import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Set CUDA_LAUNCH_BLOCKING for deterministic errors/debugging
if 'CUDA_LAUNCH_BLOCKING' not in os.environ:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
import sys
from pathlib import Path

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser(description="Quantize models for Jetson/Ampere with TensorRT-LLM")
parser.add_argument("--format",
                    type=str,
                    required=False,
                    default="W4A16",
                    choices=["W4A16", "W8A8", "W4A8", "FP8", "FP16"],
                    help="Quant format: W4A16 (INT4-AWQ), W8A8 (INT8-SQ), W4A8 (experimental), FP8 (Hopper+), FP16 (no quant)")
parser.add_argument("--gpu_arch",
                    type=str,
                    required=False,
                    default="ampere",
                    choices=["ampere", "ada", "hopper", "blackwell"],
                    help="Target GPU architecture for deployment")
parser.add_argument("--input_dir",
                    type=str,
                    required=True,
                    help="Path to input model directory")
parser.add_argument("--output_dir",
                    type=str,
                    required=True,
                    help="Output directory for TRT-LLM checkpoint")
parser.add_argument("--model_name",
                    type=str,
                    required=False,
                    default="phi",
                    help="Model architecture name (phi, llama, qwen, etc.)")
parser.add_argument("--calib_samples",
                    type=int,
                    default=512,
                    help="Number of calibration samples")
parser.add_argument("--calib_seq_len",
                    type=int,
                    default=512,
                    help="Calibration sequence length")
parser.add_argument("--dtype",
                    type=str,
                    default="float16",
                    choices=["float16", "bfloat16"],
                    help="Model dtype")
parser.add_argument("--tp_size",
                    type=int,
                    default=1,
                    help="Tensor parallelism size")
parser.add_argument("--pp_size",
                    type=int,
                    default=1,
                    help="Pipeline parallelism size")

args = parser.parse_args()

# Validate paths
input_path = Path(args.input_dir).expanduser()
output_path = Path(args.output_dir).expanduser()

if not input_path.exists():
    print(f"[ERROR] Input directory does not exist: {input_path}")
    sys.exit(1)

output_path.mkdir(parents=True, exist_ok=True)

# Check GPU
print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

# ---------------------------
# Map format to TRT-LLM qformat
# ---------------------------
format_map = {
    "W4A16": "int4_awq",
    "W8A8": "int8_sq",
    "W4A8": "w4a8_awq",
    "FP8": "fp8",
    "FP16": None  # No quantization
}

qformat = format_map.get(args.format.upper())

if args.format.upper() == "FP16":
    print("\n[INFO] FP16 selected: no quantization")
    print("[INFO] Use trtllm-build directly on the HuggingFace model")
    sys.exit(0)

# Validate compatibility
if args.format.upper() == "FP8" and args.gpu_arch in ["ampere", "ada"]:
    print("[ERROR] FP8 requires Hopper+ architecture")
    sys.exit(1)

if args.format.upper() == "W4A8" and args.gpu_arch == "ampere":
    print("[WARN] W4A8 is experimental on Ampere, recommend W4A16 or W8A8")

print(f"\n[INFO] Configuration:")
print(f"       Format: {args.format} ({qformat})")
print(f"       GPU Architecture: {args.gpu_arch}")
print(f"       Input: {input_path}")
print(f"       Output: {output_path}")
print(f"       Model: {args.model_name}")
print(f"       Calibration: {args.calib_samples} samples, {args.calib_seq_len} seq_len")

# ---------------------------
# Build TRT-LLM quantization command
# ---------------------------
print("\n[INFO] Using TensorRT-LLM quantization...")

# Use TRT-LLM's quantize.py script
trtllm_quantize_script = Path.home() / "Documents/tensorrt_llm/TensorRT-LLM/examples/quantization/quantize.py"

if not trtllm_quantize_script.exists():
    print(f"[ERROR] TRT-LLM quantize script not found at: {trtllm_quantize_script}")
    print("[INFO] Please install TensorRT-LLM or specify correct path")
    sys.exit(1)

cmd = [
    "python",
    str(trtllm_quantize_script),
    "--model_dir", str(input_path),
    "--output_dir", str(output_path),
    "--dtype", args.dtype,
    "--qformat", qformat,
    "--calib_size", str(args.calib_samples),
    "--tp_size", str(args.tp_size),
    "--pp_size", str(args.pp_size)
]

print("\n[CMD] " + " ".join(cmd))
print()

# ---------------------------
# Execute quantization
# ---------------------------
import subprocess

try:
    result = subprocess.run(cmd, check=True)
    
    print("\n" + "="*60)
    print("[DONE] Quantization complete!")
    print(f"       Checkpoint saved to: {output_path}")
    print("="*60)
    
    # Show next steps
    print("\n[NEXT STEPS]")
    print("1. Build TRT-LLM engine:")
    print(f"   trtllm-build --checkpoint_dir {output_path} \\")
    print(f"                --output_dir ./engine \\")
    print(f"                --gemm_plugin {args.dtype}")
    print("\n2. Transfer to Jetson and run inference")
    
except subprocess.CalledProcessError as e:
    print(f"\n[ERROR] Quantization failed with exit code {e.returncode}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n[INFO] Quantization interrupted by user")
    sys.exit(130)