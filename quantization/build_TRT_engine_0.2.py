#!/usr/bin/env python3
"""
TensorRT-LLM Engine Builder for Q-DRIVE
Converts quantized checkpoint to optimized TRT-LLM engine
Supports both x86_64 and ARM64 (Jetson) platforms

USAGE:
  Simple: python build_TRT_engine.py --checkpoint_dir /path/to/checkpoint
  Full:   python build_TRT_engine.py --checkpoint_dir /path/to/checkpoint --output_dir /path/to/output
"""

import argparse
import os
import subprocess
import json
import sys
import platform
from pathlib import Path


def check_environment():
    """Verify TensorRT-LLM environment is properly configured"""
    print("Checking environment...")
    
    # Check architecture
    arch = platform.machine()
    print(f"  Architecture: {arch}")
    if arch == 'aarch64':
        print("  Platform: Jetson (ARM64)")
    elif arch == 'x86_64':
        print("  Platform: Desktop/Server (x86_64)")
    else:
        print(f"  WARNING: Unsupported architecture: {arch}")
    
    # Check TensorRT-LLM
    try:
        import tensorrt_llm
        print(f"  TensorRT-LLM: v{tensorrt_llm.__version__}")
    except ImportError:
        print("  ERROR: TensorRT-LLM not installed!")
        print("  Install: pip install tensorrt_llm")
        return False
    
    # Check CUDA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_info = result.stdout.strip().split('\n')[0]
        print(f"  GPU: {gpu_info}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  WARNING: Could not query GPU (nvidia-smi failed)")
    
    print("  Environment check passed\n")
    return True


def auto_detect_output_dir(checkpoint_dir: str) -> str:
    """Generate default output directory name"""
    checkpoint_name = Path(checkpoint_dir).name
    parent_dir = Path(checkpoint_dir).parent
    
    # Remove common suffixes
    base_name = checkpoint_name.replace('-checkpoint', '').replace('_checkpoint', '')
    
    # Add -engine suffix
    output_name = f"{base_name}-engine"
    output_path = parent_dir / output_name
    
    return str(output_path)


def auto_detect_settings(checkpoint_dir: str) -> dict:
    """Auto-detect optimal settings from checkpoint config"""
    config_path = os.path.join(checkpoint_dir, "config.json")
    
    defaults = {
        "max_batch_size": 1,      # Single inference for Q-DRIVE
        "max_input_len": 2048,     # Reasonable for coaching prompts
        "max_num_tokens": 2304,    # Context + generation
        "max_beam_width": 1,
        "gemm_plugin": "float16"       # Greedy decoding
    }
    
    if not os.path.exists(config_path):
        return defaults
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Adjust based on model config
        if "max_position_embeddings" in config:
            max_seq = config["max_position_embeddings"]
            # Cap at reasonable values for inference
            defaults["max_input_len"] = min(512, max_seq // 2)
        
        print(f"Auto-detected settings from checkpoint config")
        
    except Exception as e:
        print(f"Warning: Could not parse config.json: {e}")
        print("Using default settings")
    
    return defaults


def build_trt_engine(
    checkpoint_dir: str,
    output_dir: str = None,
    max_batch_size: int = None,
    max_input_len: int = None,
    max_num_tokens: int = None,
    max_beam_width: int = None,
    gpu_id: int = 0,
    gemm_plugin: str = None,
    force: bool = False
):
    """
    Build TensorRT-LLM engine from quantized checkpoint
    
    Args:
        checkpoint_dir: Path to quantized checkpoint (REQUIRED)
        output_dir: Where to save engine (auto-generated if None)
        max_batch_size: Maximum batch size (auto-detected if None)
        max_input_len: Maximum input length (auto-detected if None)
        max_num_tokens: Maximum tokens (auto-detected if None)
        max_beam_width: Beam width (auto-detected if None)
        gemm_plugin: Precision for matrix multiplication (float16 by default)
        gpu_id: GPU to use for building
        force: Overwrite existing engine
    """
    
    checkpoint_dir = os.path.expanduser(checkpoint_dir)
    
    # Validate checkpoint directory
    if not os.path.exists(checkpoint_dir):
        print(f"ERROR: Checkpoint not found: {checkpoint_dir}")
        return False
    
    config_path = os.path.join(checkpoint_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"ERROR: config.json not found in {checkpoint_dir}")
        print("This doesn't appear to be a valid TensorRT-LLM checkpoint.")
        return False
    
    # Auto-detect output directory if not provided
    if output_dir is None:
        output_dir = auto_detect_output_dir(checkpoint_dir)
        print(f"Auto-detected output directory: {output_dir}")
    else:
        output_dir = os.path.expanduser(output_dir)
    
    # Auto-detect settings if not provided
    auto_settings = auto_detect_settings(checkpoint_dir)
    
    if max_batch_size is None:
        max_batch_size = auto_settings["max_batch_size"]
    if max_input_len is None:
        max_input_len = auto_settings["max_input_len"]
    if max_num_tokens is None:
        max_num_tokens = auto_settings["max_num_tokens"]
    if max_beam_width is None:
        max_beam_width = auto_settings["max_beam_width"]
    if gemm_plugin is None:
        gemm_plugin = auto_settings["gemm_plugin"]
    
    # Load and validate config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if "producer" in config:
        producer = config["producer"].get("name", "unknown")
        version = config["producer"].get("version", "unknown")
        print(f"Checkpoint producer: {producer} v{version}")
    
    if config.get("architecture"):
        print(f"Model architecture: {config['architecture']}")
    
    # Check if output already exists
    engine_path = os.path.join(output_dir, "rank0.engine")
    if os.path.exists(engine_path) and not force:
        print(f"\nERROR: Engine already exists at {engine_path}")
        print("Options:")
        print("  1. Use --force to overwrite")
        print("  2. Use --output_dir to specify different location")
        print("  3. Delete existing engine manually")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("TensorRT-LLM Engine Build Configuration")
    print("="*70)
    print(f"Checkpoint:       {checkpoint_dir}")
    print(f"Output:           {output_dir}")
    print(f"GPU:              {gpu_id}")
    print(f"Max Batch Size:   {max_batch_size}")
    print(f"Max Input Len:    {max_input_len}")
    print(f"Max Num Tokens:   {max_num_tokens}")
    print(f"Max Beam Width:   {max_beam_width}")
    print(f"GEMM          :   {gemm_plugin} ")
    print("="*70)
    
    # Build command
    cmd = [
        "trtllm-build",
        "--checkpoint_dir", checkpoint_dir,
        "--output_dir", output_dir,
        "--gemm_plugin", "auto",
        "--max_batch_size", str(max_batch_size),
        "--max_input_len", str(max_input_len),
        "--max_num_tokens", str(max_num_tokens),
        "--max_beam_width", str(max_beam_width),
        "--gemm_plugin", str(gemm_plugin)
    ]
    
    print("\nBuild command:")
    print(" ".join(cmd))
    print("\nBuilding engine (this may take 10-30 minutes)...")
    print("You can safely ignore warnings about 'Provided but not required tensors'")
    print()
    
    # Set environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Execute build
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Print output
        print(result.stdout)
        
        # Check if engine was actually created
        if os.path.exists(engine_path):
            size_mb = os.path.getsize(engine_path) / (1024*1024)
            
            print("\n" + "="*70)
            print("BUILD SUCCESSFUL")
            print("="*70)
            print(f"Engine:     {engine_path}")
            print(f"Size:       {size_mb:.1f} MB")
            
            # List all output files
            print("\nGenerated files:")
            for f in sorted(Path(output_dir).glob("*")):
                f_size_mb = f.stat().st_size / (1024*1024)
                print(f"  {f.name:<30} {f_size_mb:>8.1f} MB")
            
            print("\nNext steps:")
            print(f"  1. Benchmark: python benchmark_qdrive_LLM.py --model_path {output_dir} --use_trt")
            print(f"  2. Deploy to application")
            print("="*70)
            
            if result.returncode != 0:
                print("\nNote: Build completed successfully despite non-zero exit code.")
                print("This is a known TensorRT-LLM cleanup bug and can be ignored.")
            
            return True
        else:
            print("\n" + "="*70)
            print("BUILD FAILED")
            print("="*70)
            print(f"Engine file not found: {engine_path}")
            print("\nPossible issues:")
            print("  - Insufficient GPU memory (need ~8GB free)")
            print("  - Incompatible checkpoint format")
            print("  - TensorRT-LLM version mismatch")
            print("\nTroubleshooting:")
            print("  1. Check GPU memory: nvidia-smi")
            print("  2. Verify checkpoint: ls -lh {checkpoint_dir}")
            print("  3. Check logs above for specific errors")
            return False
        
    except Exception as e:
        print("\n" + "="*70)
        print("BUILD ERROR")
        print("="*70)
        print(f"Exception: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build TensorRT-LLM engine from quantized checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simplest usage (auto-detects everything):
  python build_TRT_engine.py --checkpoint_dir ~/phi3-checkpoint
  
  # Specify output location:
  python build_TRT_engine.py --checkpoint_dir ~/phi3-checkpoint --output_dir ~/my-engine
  
  # Override settings:
  python build_TRT_engine.py --checkpoint_dir ~/phi3-checkpoint --max_input_len 1024
  
  # Rebuild existing engine:
  python build_TRT_engine.py --checkpoint_dir ~/phi3-checkpoint --force
        """
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to quantized checkpoint directory (REQUIRED)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save engine (default: auto-generated from checkpoint name)"
    )
    
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximum batch size (default: 1)"
    )
    
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=None,
        help="Maximum input sequence length (default: 512)"
    )
    
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=None,
        help="Maximum total tokens in context (default: 128)"
    )
    
    parser.add_argument(
        "--max_beam_width",
        type=int,
        default=None,
        help="Beam width for decoding (default: 1 for greedy)"
    )
    
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to use for building (default: 0)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing engine without prompting"
    )
    
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        help="Skip environment validation checks"
    )

    parser.add_argument(
        "--gemm_plugin",
        type=str,
        default="float16",
        help="GEMM value [bfloat16, float16, auto]"
    )
    
    args = parser.parse_args()
    
    print("Q-DRIVE TensorRT-LLM Engine Builder")
    print("="*70)
    print()
    
    # Environment check
    if not args.skip_checks:
        if not check_environment():
            print("\nEnvironment check failed.")
            print("Fix the issues above or use --skip_checks to bypass (not recommended).")
            sys.exit(1)
    
    # Build engine
    success = build_trt_engine(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
        max_num_tokens=args.max_num_tokens,
        max_beam_width=args.max_beam_width,
        gpu_id=args.gpu_id,
        force=args.force
    )
    
    sys.exit(0 if success else 1)