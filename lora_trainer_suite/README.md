# LoRA Trainer GUI Suite

A comprehensive GUI-based LoRA training suite for Flux and Stable Diffusion models with automated tagging, validation, and quality assessment.

## Features

- 🎨 **Modern Web GUI** - Built with Gradio for easy access
- 🏷️ **Automated Tagging** - CLIP Interrogator + Qwen2-VL-8B (abliterated)
- 🚀 **Multi-Model Support** - Flux.1 (images), WAN I2V 2.1/2.2 (Qwen video diffusion)
- ⚡ **Optimized Performance** - Designed for RTX 3090 with 24GB VRAM
- 📊 **Real-time Monitoring** - Training progress, loss curves, sample generation
- ✅ **Built-in Validation** - Quality metrics and comparison tools

## Supported Models

### **Image Generation:**
- **Flux.1-dev**: High-quality image generation
- **Flux.1-schnell**: Fast image generation

### **Video Generation (WAN I2V by Qwen):**
- **Wan2.1-I2V-14B-720P**: Image-to-Video generation (14B parameters, 720p resolution)
- **Wan2.2-I2V-A14B**: Advanced I2V with MoE architecture (27B total, 14B active)

**Practical Workflow:** Flux generates image → WAN I2V animates it

WAN is Qwen's advanced video diffusion model built on the Diffusion Transformer paradigm with:
- Novel 3D causal VAE (Wan-VAE) for efficient video compression
- Flow Matching framework
- Mixture-of-Experts (MoE) in v2.2 for better quality
- I2V: Animate existing images with temporal consistency

## Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 128GB
- CPU: Ryzen 9 7900X (12-core) or equivalent
- Storage: 100GB+ SSD

**Recommended:**
- All of the above
- NVMe SSD for dataset storage
- High-speed internet for model downloads

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/lora-trainer-suite.git
cd lora-trainer-suite
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install PyTorch with CUDA

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Flash Attention (Optional but Recommended)

```bash
pip install flash-attn --no-build-isolation
```

### 6. Install Training Frameworks

#### Option A: Kohya_ss (for SD models)

```bash
git clone https://github.com/kohya-ss/sd-scripts.git ../kohya_ss
cd ../kohya_ss
pip install -r requirements.txt
cd ../lora-trainer-suite
```

#### Option B: AI-Toolkit (for Flux models)

```bash
git clone https://github.com/ostris/ai-toolkit.git ../ai-toolkit
cd ../ai-toolkit
git submodule update --init --recursive
pip install -r requirements.txt
cd ../lora-trainer-suite
```

### 7. Install Qwen2-VL (Abliterated Version)

For uncensored tagging, you'll need the abliterated version:

```bash
# Option 1: Use official Qwen2-VL
pip install qwen-vl-utils

# Option 2: Use abliterated version (if available from community)
# Check Hugging Face for abliterated versions like:
# - mlabonne/Qwen2-VL-7B-Instruct-abliterated
# - Your preferred uncensored variant
```

## Quick Start

### 1. Launch the GUI

```bash
python lora_trainer_gui.py
```

The GUI will open at `http://localhost:7860`

### 2. Prepare Your Dataset

1. Go to **Dataset Preparation** tab
2. Point to your image folder
3. Images should be in supported formats: PNG, JPG, JPEG, WEBP
4. Optionally resize/augment your dataset

### 3. (Optional) Start vLLM Server for Faster Tagging

For 3-5x faster inference with Qwen models:

```bash
# Start vLLM server in a separate terminal
python start_vllm_server.py

# Or with custom model
python start_vllm_server.py --model Qwen/Qwen2.5-VL-7B-Instruct
```

The GUI will automatically detect and use vLLM if running.

### 4. Auto-Tag Images

1. Go to **Auto Tagging** tab
2. Select your dataset path
3. Choose tagging methods:
   - **CLIP Interrogator**: Fast, good quality captions
   - **Qwen VL (Qwen2.5/Qwen3)**: Detailed, uncensored descriptions
4. Select backend:
   - **auto**: Try vLLM first (faster), fallback to direct
   - **vllm**: Force vLLM (requires server running)
   - **direct**: Use Transformers (slower but simpler)
5. Configure merge settings
6. Click **Start Auto-Tagging**

### 5. Train LoRA

1. Go to **LoRA Training** tab
2. Configure:
   - **Base Model**: Select Flux or SD model
   - **Output Name**: Name for your LoRA
   - **Dataset Path**: Point to tagged dataset
   - **Training Parameters**: Adjust as needed
3. Click **Start Training**

### 6. Validate Model

1. Go to **Validation** tab
2. Load your trained LoRA
3. Enter test prompts
4. Generate samples and view quality metrics

## Configuration

### Training Parameters Guide

#### For RTX 3090 (24GB VRAM):

**Flux Models:**
```
Batch Size: 1
Gradient Accumulation: 4-8
Learning Rate: 1e-4
LoRA Rank: 16-32
Mixed Precision: bf16
Gradient Checkpointing: True
```

**WAN I2V (Qwen) 2.1/2.2:**
```
Model: Wan-AI/Wan2.1-I2V-14B-720P or Wan-AI/Wan2.2-I2V-A14B
Batch Size: 1
Gradient Accumulation: 4-8
Learning Rate: 1e-4 to 5e-5
LoRA Rank: 16-32
Mixed Precision: bf16
Resolution: 512x512 or 720p (video frames)
Num Frames: 16-32 (depending on VRAM)
Input: Single image (generated by Flux)
```

**Note:** WAN 2.2 uses MoE (Mixture-of-Experts) architecture with 27B total parameters but only 14B active per step.
**I2V Workflow:** Train on image→video pairs. Use Flux-generated images as input for WAN I2V animation.

### Memory Optimization

If you encounter OOM errors:

1. **Reduce Batch Size**: Set to 1
2. **Increase Gradient Accumulation**: Compensates for smaller batch
3. **Enable Gradient Checkpointing**: Trades compute for memory
4. **Lower LoRA Rank**: 8-16 instead of 32+
5. **Use 4-bit Quantization**: For Qwen2-VL tagging

## Module Usage

### CLIP Interrogator

```python
from modules.clip_interrogator_module import CLIPInterrogatorModule

ci = CLIPInterrogatorModule(device='cuda')
caption = ci.interrogate('path/to/image.jpg', mode='best')
print(caption)
```

### Qwen2-VL Tagger

```python
from modules.qwen_tagger import QwenVLTagger

tagger = QwenVLTagger(use_4bit=True)
tags = tagger.tag_image(
    'path/to/image.jpg',
    prompt="Describe this image in detail with all visible elements."
)
print(tags)
```

### Dataset Manager

```python
from modules.dataset_manager import DatasetManager

dm = DatasetManager()
info = dm.load_dataset('/path/to/dataset')
dm.resize_images('/path/to/dataset', 512, 512)
dm.create_training_metadata('/path/to/dataset')
```

### Command-Line Tools

Each module includes CLI support:

```bash
# CLIP Interrogator
python modules/clip_interrogator_module.py image.jpg --mode best

# Qwen Tagger
python modules/qwen_tagger.py image.jpg --booru

# Dataset Manager
python modules/dataset_manager.py /path/to/dataset --analyze --validate

# Model Validator
python modules/validator.py \
  --lora /path/to/lora.safetensors \
  --base "stabilityai/stable-diffusion-2-1" \
  --prompt "a photo of sks person" \
  --samples 4
```

## vLLM High-Performance Backend

### What is vLLM?

vLLM is a high-performance inference server that provides:
- **3-5x faster** inference compared to direct Transformers
- Better GPU utilization with PagedAttention
- Automatic batching for multiple requests
- Lower memory footprint

### Starting vLLM Server

**Quick Start:**
```bash
python start_vllm_server.py
```

**Custom Configuration:**
```bash
# Use specific model
python start_vllm_server.py --model Qwen/Qwen2.5-VL-7B-Instruct

# Adjust GPU memory usage (0.0-1.0)
python start_vllm_server.py --gpu-memory 0.7

# Custom port
python start_vllm_server.py --port 8001

# Multi-GPU (if you have 2+ GPUs)
python start_vllm_server.py --tensor-parallel 2
```

**Manual vLLM Command:**
```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code
```

### Using vLLM in Code

```python
from modules.qwen_tagger import QwenVLTagger

# Auto mode: tries vLLM first, falls back to direct
tagger = QwenVLTagger(backend="auto")

# Force vLLM (fails if server not running)
tagger = QwenVLTagger(backend="vllm")

# Force direct mode (slower but simpler)
tagger = QwenVLTagger(backend="direct", use_4bit=True)

# Custom vLLM server
tagger = QwenVLTagger(
    backend="vllm",
    vllm_server_url="http://192.168.1.100",
    vllm_port=8000
)
```

### vLLM vs Direct Mode

| Feature | vLLM | Direct |
|---------|------|--------|
| **Speed** | 3-5x faster | Baseline |
| **Memory** | Lower (PagedAttention) | Higher |
| **Batching** | Automatic | Manual |
| **Setup** | Requires server | Just install packages |
| **4-bit Quant** | No | Yes |
| **Offline** | No (needs server) | Yes |

**Recommendation:**
- **Large datasets**: Use vLLM for batch tagging
- **Quick tests**: Use direct mode
- **Limited VRAM**: Direct mode with 4-bit quantization

## Advanced Features

### Custom Tagging Prompts

For Qwen2-VL, you can customize prompts:

```python
custom_prompt = """
Describe this image including:
1. Main subject characteristics
2. Clothing and accessories
3. Pose and expression
4. Background and environment
5. Artistic style and mood
6. Technical qualities (lighting, composition)

Format as comma-separated tags.
"""

tags = tagger.tag_image('image.jpg', prompt=custom_prompt)
```

### Booru-Style Tags

Generate Danbooru/e621 style tags:

```python
booru_tags = tagger.generate_booru_tags('image.jpg')
# Output: "1girl, solo, long_hair, blue_eyes, blonde_hair, standing, outdoors, day"
```

### Batch Processing

```python
image_paths = glob.glob('/dataset/*.jpg')
results = tagger.batch_tag(
    image_paths,
    progress_callback=lambda cur, total, tags: print(f"{cur}/{total}: {tags[:50]}...")
)
```

## Training Workflows

### Workflow 1: Character LoRA (Flux)

1. Collect 20-50 images of character
2. Auto-tag with CLIP + Qwen2-VL
3. Manual review and cleanup
4. Train with:
   - Steps: 1000-2000
   - Rank: 16-32
   - Learning Rate: 1e-4

### Workflow 2: Style LoRA (WAN I2V)

1. Generate 30-100 images with Flux in target style
2. Create video animations from those images (for training data)
3. Tag focusing on motion, style, and composition
4. Train with:
   - Steps: 1500-3000
   - Rank: 16-32
   - Learning Rate: 5e-5
   - Input: Image (from Flux) → Output: Video animation

### Workflow 3: Concept LoRA

1. Collect diverse examples
2. Detailed tagging with emphasis on concept
3. Train with higher steps (2000-4000)

## Troubleshooting

### CUDA Out of Memory

```
Error: CUDA out of memory
```

Solutions:
- Reduce batch size to 1
- Enable gradient checkpointing
- Increase gradient accumulation steps
- Use 4-bit quantization for tagging models
- Close other GPU applications

### CLIP Interrogator Not Found

```
ModuleNotFoundError: No module named 'clip_interrogator'
```

Solution:
```bash
pip install clip-interrogator
```

### Flash Attention Build Errors

If Flash Attention fails to install:

```bash
pip install flash-attn --no-build-isolation
```

Or skip it - the suite works without it (just slower).

### Qwen2-VL Model Download Issues

If model download is slow or fails:

1. Set HuggingFace cache:
```bash
export HF_HOME=/path/to/large/storage
```

2. Use mirror (if available):
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Performance Tips

1. **Use vLLM for Tagging**: 3-5x faster than direct mode for batch operations
2. **Use NVMe SSD**: For dataset storage and cache
3. **Enable Flash Attention**: 20-30% faster training
4. **Use bf16**: Better numerical stability than fp16
5. **Cache Latents**: Speeds up training (Kohya_ss)
6. **Multiple Workers**: For data loading (adjust based on CPU)

## File Structure

```
lora_trainer_suite/
├── lora_trainer_gui.py          # Main GUI application
├── modules/
│   ├── __init__.py
│   ├── clip_interrogator_module.py  # CLIP captioning
│   ├── qwen_tagger.py           # Qwen2-VL tagging
│   ├── dataset_manager.py       # Dataset utilities
│   ├── lora_trainer.py          # Training integration
│   ├── validator.py             # Model validation
│   └── config_manager.py        # Configuration
├── requirements.txt
├── README.md
└── models/                      # Downloaded models (created)
    ├── clip_interrogator/
    ├── qwen2vl/
    ├── flux/
    └── sd/
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Credits

- **Kohya_ss**: Training scripts for Stable Diffusion
- **AI-Toolkit**: Flux training framework
- **CLIP Interrogator**: Automated captioning
- **Qwen2-VL**: Vision-language model by Alibaba Cloud
- **Hugging Face**: Models and libraries

## Support

For issues and questions:
- GitHub Issues: [link]
- Discord: [link]
- Documentation: [link]

## Changelog

### v1.1.0 (2024-11-22)
- **vLLM Backend Support**: 3-5x faster inference for Qwen models
- **Qwen3-VL Support**: Updated to support Qwen2.5-VL and Qwen3-VL
- **Dual Backend System**: Auto-detect vLLM or fallback to direct mode
- **vLLM Server Launcher**: Easy-to-use script for starting vLLM
- **Enhanced GUI**: Backend selection in settings and tagging tabs
- **Updated Dependencies**: Latest transformers, vLLM, and accelerate

### v1.0.0 (2024-01-XX)
- Initial release
- CLIP Interrogator integration
- Qwen2-VL tagging support
- Flux and WAN I2V video diffusion training support
- Real-time validation
- Web-based GUI

## Roadmap

- [ ] SDXL support
- [ ] ControlNet integration
- [ ] Multi-GPU training
- [ ] Automatic hyperparameter tuning
- [ ] Dataset augmentation presets
- [ ] Experiment tracking (WandB/TensorBoard)
- [ ] Model merging utilities
- [ ] Batch validation tools
