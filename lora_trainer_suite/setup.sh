#!/bin/bash

# LoRA Trainer Suite Setup Script
# For RTX 3090, 128GB RAM, Ryzen 9 7900X

set -e

echo "========================================="
echo "LoRA Trainer Suite - Setup Script"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 10)' 2>/dev/null; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Check NVIDIA GPU
echo ""
echo "Checking NVIDIA GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "Warning: nvidia-smi not found. GPU support may not work."
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA support..."
read -p "Select CUDA version (11.8/12.1) [12.1]: " cuda_version
cuda_version=${cuda_version:-12.1}

if [ "$cuda_version" == "11.8" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [ "$cuda_version" == "12.1" ]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "Invalid CUDA version. Using 12.1..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

# Install core dependencies
echo ""
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install Flash Attention (optional)
echo ""
read -p "Install Flash Attention 2? (recommended, y/n) [y]: " install_flash
install_flash=${install_flash:-y}

if [ "$install_flash" == "y" ]; then
    echo "Installing Flash Attention 2..."
    pip install flash-attn --no-build-isolation || echo "Warning: Flash Attention installation failed. Continuing..."
fi

# Setup training frameworks
echo ""
echo "========================================="
echo "Training Framework Setup"
echo "========================================="
echo ""
echo "This suite supports multiple training frameworks:"
echo "  1. Kohya_ss (for Stable Diffusion)"
echo "  2. AI-Toolkit (for Flux)"
echo "  3. Both"
echo "  4. Skip (install later)"
echo ""
read -p "Select option [3]: " framework_option
framework_option=${framework_option:-3}

if [ "$framework_option" == "1" ] || [ "$framework_option" == "3" ]; then
    echo ""
    echo "Setting up Kohya_ss..."
    if [ -d "../kohya_ss" ]; then
        echo "Kohya_ss already exists. Skipping..."
    else
        git clone https://github.com/kohya-ss/sd-scripts.git ../kohya_ss
        cd ../kohya_ss
        pip install -r requirements.txt
        cd ../lora_trainer_suite
        echo "✓ Kohya_ss installed"
    fi
fi

if [ "$framework_option" == "2" ] || [ "$framework_option" == "3" ]; then
    echo ""
    echo "Setting up AI-Toolkit..."
    if [ -d "../ai-toolkit" ]; then
        echo "AI-Toolkit already exists. Skipping..."
    else
        git clone https://github.com/ostris/ai-toolkit.git ../ai-toolkit
        cd ../ai-toolkit
        git submodule update --init --recursive
        pip install -r requirements.txt
        cd ../lora_trainer_suite
        echo "✓ AI-Toolkit installed"
    fi
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p models/clip_interrogator
mkdir -p models/qwen2vl
mkdir -p models/flux
mkdir -p models/sd
mkdir -p output/loras
mkdir -p datasets
echo "✓ Directories created"

# Test installation
echo ""
echo "========================================="
echo "Testing Installation"
echo "========================================="
echo ""

python3 << 'EOF'
import sys

def test_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError:
        print(f"✗ {package_name or module_name} - MISSING")
        return False

print("Core Dependencies:")
test_import("torch", "PyTorch")
test_import("torchvision")
test_import("diffusers", "Diffusers")
test_import("transformers", "Transformers")
test_import("accelerate", "Accelerate")
test_import("bitsandbytes", "BitsAndBytes")

print("\nOptional Dependencies:")
test_import("flash_attn", "Flash Attention 2")
test_import("xformers", "xFormers")

print("\nGUI & Utilities:")
test_import("gradio", "Gradio")
test_import("PIL", "Pillow")
test_import("cv2", "OpenCV")

print("\nTesting GPU:")
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("✗ CUDA not available")
EOF

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Launch GUI: python lora_trainer_gui.py"
echo "  3. Open browser: http://localhost:7860"
echo ""
echo "Optional:"
echo "  - Download models will happen automatically on first use"
echo "  - Configure settings in the Settings tab"
echo "  - See README.md for detailed usage instructions"
echo ""
echo "Happy training! 🎨"
