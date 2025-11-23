#!/usr/bin/env python3
"""
LoRA Trainer GUI Suite
Main application for training LoRA models with automated tagging and validation
Supports Flux and Stable Diffusion 2.1/2.2
Optimized for RTX 3090, 128GB RAM, Ryzen 9 7900X
"""

import gradio as gr
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

# Add modules to path
sys.path.append(str(Path(__file__).parent))

from modules.clip_interrogator_module import CLIPInterrogatorModule
from modules.qwen_tagger import QwenVLTagger
from modules.dataset_manager import DatasetManager
from modules.lora_trainer import LoRATrainer
from modules.validator import ModelValidator
from modules.config_manager import ConfigManager


class LoRATrainerGUI:
    """Main GUI application for LoRA training suite"""

    def __init__(self):
        self.config = ConfigManager()
        self.dataset_manager = DatasetManager(self.config)
        self.clip_interrogator = None  # Lazy load
        self.qwen_tagger = None  # Lazy load
        self.lora_trainer = None  # Lazy load
        self.validator = None  # Lazy load
        self.training_thread = None

    def build_interface(self):
        """Build the Gradio interface"""

        with gr.Blocks(title="LoRA Trainer Suite", theme=gr.themes.Soft()) as app:
            gr.Markdown(
                """
                # 🎨 LoRA Trainer Suite
                Complete workflow for training high-quality LoRA models
                **Hardware:** RTX 3090 | 128GB RAM | Ryzen 9 7900X
                """
            )

            with gr.Tabs() as tabs:
                # Tab 1: Dataset Preparation
                with gr.Tab("📁 Dataset Preparation"):
                    self.build_dataset_tab()

                # Tab 2: Auto Tagging
                with gr.Tab("🏷️ Auto Tagging"):
                    self.build_tagging_tab()

                # Tab 3: LoRA Training
                with gr.Tab("🚀 LoRA Training"):
                    self.build_training_tab()

                # Tab 4: Validation
                with gr.Tab("✅ Validation"):
                    self.build_validation_tab()

                # Tab 5: Settings
                with gr.Tab("⚙️ Settings"):
                    self.build_settings_tab()

        return app

    def build_dataset_tab(self):
        """Build dataset preparation interface"""
        gr.Markdown("## Dataset Preparation")

        with gr.Row():
            with gr.Column():
                dataset_path = gr.Textbox(
                    label="Dataset Directory",
                    placeholder="/path/to/dataset",
                    value=self.config.get("dataset_path", "")
                )

                gr.Markdown("### Supported Formats")
                gr.Markdown("- Images: PNG, JPG, JPEG, WEBP\n- Captions: TXT files with same name as images")

                load_btn = gr.Button("Load Dataset", variant="primary")

            with gr.Column():
                dataset_info = gr.JSON(label="Dataset Information")
                image_preview = gr.Gallery(
                    label="Image Preview",
                    columns=4,
                    height=400
                )

        with gr.Row():
            with gr.Column():
                resize_width = gr.Number(label="Resize Width", value=512)
                resize_height = gr.Number(label="Resize Height", value=512)
                resize_btn = gr.Button("Batch Resize Images")

            with gr.Column():
                augment_flip = gr.Checkbox(label="Horizontal Flip", value=False)
                augment_rotate = gr.Checkbox(label="Random Rotation", value=False)
                augment_btn = gr.Button("Apply Augmentation")

        process_status = gr.Textbox(label="Status", interactive=False)

        # Event handlers
        load_btn.click(
            fn=self.load_dataset,
            inputs=[dataset_path],
            outputs=[dataset_info, image_preview, process_status]
        )

        resize_btn.click(
            fn=self.resize_images,
            inputs=[dataset_path, resize_width, resize_height],
            outputs=[process_status]
        )

        augment_btn.click(
            fn=self.augment_dataset,
            inputs=[dataset_path, augment_flip, augment_rotate],
            outputs=[process_status]
        )

    def build_tagging_tab(self):
        """Build auto-tagging interface"""
        gr.Markdown("## Automated Image Tagging")

        with gr.Row():
            with gr.Column():
                tag_dataset_path = gr.Textbox(
                    label="Dataset Directory",
                    placeholder="/path/to/dataset"
                )

                gr.Markdown("### Tagging Methods")

                use_clip = gr.Checkbox(label="CLIP Interrogator", value=True)
                clip_mode = gr.Dropdown(
                    label="CLIP Mode",
                    choices=["best", "fast", "classic", "negative"],
                    value="best"
                )

                use_qwen = gr.Checkbox(label="Qwen VL (Qwen2.5/Qwen3) - Uncensored", value=True)

                qwen_backend = gr.Radio(
                    label="Qwen Backend",
                    choices=["auto", "vllm", "direct"],
                    value="auto",
                    info="auto: try vLLM first (3-5x faster), fallback to direct | vllm: force vLLM | direct: transformers only"
                )

                qwen_prompt = gr.Textbox(
                    label="Qwen Custom Prompt",
                    value="Describe this image in detail, including all visible elements, style, composition, and artistic qualities.",
                    lines=3
                )

                merge_tags = gr.Checkbox(
                    label="Merge with Existing Tags",
                    value=True
                )

                tag_format = gr.Dropdown(
                    label="Tag Format",
                    choices=["comma_separated", "line_separated", "json"],
                    value="comma_separated"
                )

                start_tag_btn = gr.Button("Start Auto-Tagging", variant="primary", size="lg")
                stop_tag_btn = gr.Button("Stop", variant="stop")

            with gr.Column():
                tagging_progress = gr.Textbox(
                    label="Progress",
                    lines=15,
                    interactive=False
                )

                current_image = gr.Image(label="Current Image")
                current_tags = gr.Textbox(label="Generated Tags", lines=5)

        # Event handlers
        start_tag_btn.click(
            fn=self.start_auto_tagging,
            inputs=[
                tag_dataset_path, use_clip, clip_mode,
                use_qwen, qwen_backend, qwen_prompt, merge_tags, tag_format
            ],
            outputs=[tagging_progress, current_image, current_tags]
        )

    def build_training_tab(self):
        """Build LoRA training interface"""
        gr.Markdown("## LoRA Training Configuration")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Configuration")

                base_model = gr.Dropdown(
                    label="Base Model",
                    choices=[
                        "black-forest-labs/FLUX.1-dev",
                        "black-forest-labs/FLUX.1-schnell",
                        "Wan-AI/Wan2.1-T2V-14B",
                        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    ],
                    value="black-forest-labs/FLUX.1-dev",
                    info="Flux for images, WAN (Qwen video) for video generation"
                )

                output_name = gr.Textbox(
                    label="Output LoRA Name",
                    placeholder="my_lora_model"
                )

                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./output/loras"
                )

                gr.Markdown("### Training Parameters")

                train_dataset_path = gr.Textbox(
                    label="Training Dataset",
                    placeholder="/path/to/dataset"
                )

                with gr.Row():
                    batch_size = gr.Number(label="Batch Size", value=1)
                    gradient_accumulation = gr.Number(label="Gradient Accumulation Steps", value=4)

                with gr.Row():
                    learning_rate = gr.Number(label="Learning Rate", value=1e-4, step=1e-5)
                    max_train_steps = gr.Number(label="Max Train Steps", value=1000)

                with gr.Row():
                    lora_rank = gr.Number(label="LoRA Rank", value=16)
                    lora_alpha = gr.Number(label="LoRA Alpha", value=32)

                optimizer = gr.Dropdown(
                    label="Optimizer",
                    choices=["AdamW8bit", "AdamW", "Prodigy", "Lion"],
                    value="AdamW8bit"
                )

                with gr.Row():
                    mixed_precision = gr.Dropdown(
                        label="Mixed Precision",
                        choices=["no", "fp16", "bf16"],
                        value="bf16"
                    )
                    gradient_checkpointing = gr.Checkbox(
                        label="Gradient Checkpointing",
                        value=True
                    )

                with gr.Row():
                    save_every_n_steps = gr.Number(label="Save Every N Steps", value=100)
                    sample_every_n_steps = gr.Number(label="Sample Every N Steps", value=100)

                sample_prompt = gr.Textbox(
                    label="Sample Prompt",
                    placeholder="a photo of sks person",
                    lines=2
                )

                start_training_btn = gr.Button("Start Training", variant="primary", size="lg")
                stop_training_btn = gr.Button("Stop Training", variant="stop")

            with gr.Column():
                training_progress = gr.Textbox(
                    label="Training Progress",
                    lines=20,
                    interactive=False
                )

                loss_plot = gr.Plot(label="Training Loss")

                sample_images = gr.Gallery(
                    label="Sample Generations",
                    columns=2,
                    height=400
                )

        # Event handlers
        start_training_btn.click(
            fn=self.start_training,
            inputs=[
                base_model, output_name, output_dir, train_dataset_path,
                batch_size, gradient_accumulation, learning_rate,
                max_train_steps, lora_rank, lora_alpha, optimizer,
                mixed_precision, gradient_checkpointing,
                save_every_n_steps, sample_every_n_steps, sample_prompt
            ],
            outputs=[training_progress, loss_plot, sample_images]
        )

    def build_validation_tab(self):
        """Build validation interface"""
        gr.Markdown("## Model Validation")

        with gr.Row():
            with gr.Column():
                lora_model_path = gr.Textbox(
                    label="LoRA Model Path",
                    placeholder="/path/to/lora/model"
                )

                val_base_model = gr.Dropdown(
                    label="Base Model",
                    choices=[
                        "black-forest-labs/FLUX.1-dev",
                        "black-forest-labs/FLUX.1-schnell",
                        "Wan-AI/Wan2.1-T2V-14B",
                        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
                    ],
                    value="black-forest-labs/FLUX.1-dev"
                )

                validation_prompt = gr.Textbox(
                    label="Validation Prompt",
                    placeholder="a photo of sks person in a suit",
                    lines=3
                )

                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="blurry, low quality, distorted",
                    lines=2
                )

                with gr.Row():
                    num_samples = gr.Number(label="Number of Samples", value=4)
                    steps = gr.Number(label="Inference Steps", value=30)

                with gr.Row():
                    guidance_scale = gr.Number(label="Guidance Scale", value=7.5)
                    seed = gr.Number(label="Seed (-1 for random)", value=-1)

                gr.Markdown("### Video Settings (for WAN models)")

                with gr.Row():
                    num_frames = gr.Number(
                        label="Number of Frames",
                        value=16,
                        info="For video models only"
                    )
                    fps = gr.Number(
                        label="FPS",
                        value=8,
                        info="Frames per second for video"
                    )

                generate_btn = gr.Button("Generate Samples", variant="primary")

            with gr.Column():
                validation_images = gr.Gallery(
                    label="Generated Samples (Images/Videos)",
                    columns=2,
                    height=600,
                    type="filepath"
                )

                quality_metrics = gr.JSON(label="Quality Metrics")

                gr.Markdown(
                    """
                    **Note:** Videos are saved as animated GIFs.
                    Click on thumbnails to view full size/animation.
                    """
                )

        # Event handlers
        generate_btn.click(
            fn=self.validate_model,
            inputs=[
                lora_model_path, val_base_model, validation_prompt,
                negative_prompt, num_samples, steps, guidance_scale, seed,
                num_frames, fps
            ],
            outputs=[validation_images, quality_metrics]
        )

    def build_settings_tab(self):
        """Build settings interface"""
        gr.Markdown("## Application Settings")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Hardware Settings")

                device = gr.Dropdown(
                    label="Device",
                    choices=["cuda", "cpu"],
                    value="cuda"
                )

                vram_optimization = gr.Dropdown(
                    label="VRAM Optimization",
                    choices=["none", "low_vram", "medium_vram"],
                    value="none"
                )

                cpu_threads = gr.Number(
                    label="CPU Threads",
                    value=12,
                    info="Ryzen 9 7900X has 12 cores"
                )

                gr.Markdown("### Qwen VL Backend Settings")

                default_qwen_backend = gr.Radio(
                    label="Default Qwen Backend",
                    choices=["auto", "vllm", "direct"],
                    value="auto",
                    info="auto: try vLLM, fallback to direct"
                )

                vllm_server_url = gr.Textbox(
                    label="vLLM Server URL",
                    value="http://localhost:8000",
                    info="URL of vLLM server (if using vLLM backend)"
                )

                gr.Markdown("### Model Cache")

                cache_dir = gr.Textbox(
                    label="Model Cache Directory",
                    value="./models"
                )

                clear_cache_btn = gr.Button("Clear Model Cache")

            with gr.Column():
                gr.Markdown("### About")

                gr.Markdown(
                    """
                    **LoRA Trainer Suite v1.1**

                    Features:
                    - CLIP Interrogator for automated captioning
                    - Qwen2-VL / Qwen2.5-VL / Qwen3-VL (uncensored tagging)
                    - **Dual Backend:** vLLM (3-5x faster) + Direct mode
                    - Support for Flux (images) and WAN 2.1/2.2 (Qwen video)
                    - Integrated validation pipeline

                    Hardware Requirements:
                    - GPU: RTX 3090 (24GB VRAM)
                    - RAM: 128GB
                    - CPU: Ryzen 9 7900X (12-core)

                    Dependencies:
                    - Kohya_ss training scripts
                    - Hugging Face Diffusers
                    - BitsAndBytes for quantization
                    - CLIP Interrogator
                    - vLLM (optional, for faster inference)

                    **vLLM Server:**
                    Run `python start_vllm_server.py` to start
                    the high-performance inference server.

                    **WAN Video:** Qwen's video diffusion models
                    (Wan-AI/Wan2.1-T2V-14B, Wan-AI/Wan2.2-T2V-A14B)
                    """
                )

                save_settings_btn = gr.Button("Save Settings", variant="primary")

        settings_status = gr.Textbox(label="Status", interactive=False)

        save_settings_btn.click(
            fn=self.save_settings,
            inputs=[device, vram_optimization, cpu_threads, default_qwen_backend, vllm_server_url, cache_dir],
            outputs=[settings_status]
        )

    # Implementation methods

    def load_dataset(self, dataset_path: str) -> Tuple[Dict, List, str]:
        """Load and analyze dataset"""
        try:
            info = self.dataset_manager.load_dataset(dataset_path)
            preview_images = self.dataset_manager.get_preview_images(limit=12)
            return info, preview_images, "Dataset loaded successfully"
        except Exception as e:
            return {}, [], f"Error: {str(e)}"

    def resize_images(self, dataset_path: str, width: int, height: int) -> str:
        """Resize images in dataset"""
        try:
            self.dataset_manager.resize_images(dataset_path, width, height)
            return f"Images resized to {width}x{height}"
        except Exception as e:
            return f"Error: {str(e)}"

    def augment_dataset(self, dataset_path: str, flip: bool, rotate: bool) -> str:
        """Apply data augmentation"""
        try:
            self.dataset_manager.augment(dataset_path, flip, rotate)
            return "Augmentation applied successfully"
        except Exception as e:
            return f"Error: {str(e)}"

    def start_auto_tagging(
        self, dataset_path: str, use_clip: bool, clip_mode: str,
        use_qwen: bool, qwen_backend: str, qwen_prompt: str, merge_tags: bool, tag_format: str
    ):
        """Start automated tagging process"""
        try:
            # Lazy load models
            if use_clip and self.clip_interrogator is None:
                yield "Loading CLIP Interrogator...", None, ""
                self.clip_interrogator = CLIPInterrogatorModule()

            if use_qwen and self.qwen_tagger is None:
                yield f"Loading Qwen VL (backend: {qwen_backend})...\n", None, ""
                self.qwen_tagger = QwenVLTagger(backend=qwen_backend)

            # Get all images
            images = self.dataset_manager.get_all_images(dataset_path)
            total = len(images)

            for idx, img_path in enumerate(images):
                progress_text = f"Processing {idx+1}/{total}: {os.path.basename(img_path)}\n"

                tags = []

                # CLIP Interrogator
                if use_clip:
                    clip_tags = self.clip_interrogator.interrogate(img_path, mode=clip_mode)
                    tags.append(clip_tags)
                    progress_text += f"CLIP: {clip_tags}\n"

                # Qwen2-VL tagging
                if use_qwen:
                    qwen_tags = self.qwen_tagger.tag_image(img_path, qwen_prompt)
                    tags.append(qwen_tags)
                    progress_text += f"Qwen: {qwen_tags}\n"

                # Merge and save tags
                final_tags = self._merge_tags(tags, tag_format)
                self.dataset_manager.save_tags(img_path, final_tags, merge_tags)

                yield progress_text, img_path, final_tags

            yield f"Tagging complete! Processed {total} images.", None, ""

        except Exception as e:
            yield f"Error: {str(e)}", None, ""

    def start_training(self, *args):
        """Start LoRA training"""
        try:
            if self.lora_trainer is None:
                self.lora_trainer = LoRATrainer(self.config)

            # Parse training parameters
            training_config = {
                'base_model': args[0],
                'output_name': args[1],
                'output_dir': args[2],
                'dataset_path': args[3],
                'batch_size': int(args[4]),
                'gradient_accumulation_steps': int(args[5]),
                'learning_rate': float(args[6]),
                'max_train_steps': int(args[7]),
                'lora_rank': int(args[8]),
                'lora_alpha': int(args[9]),
                'optimizer': args[10],
                'mixed_precision': args[11],
                'gradient_checkpointing': args[12],
                'save_every_n_steps': int(args[13]),
                'sample_every_n_steps': int(args[14]),
                'sample_prompt': args[15],
            }

            # Start training in generator mode for progress updates
            for progress, loss_data, samples in self.lora_trainer.train(training_config):
                yield progress, loss_data, samples

        except Exception as e:
            yield f"Error: {str(e)}", None, []

    def validate_model(
        self, lora_path: str, base_model: str, prompt: str,
        negative_prompt: str, num_samples: int, steps: int,
        guidance_scale: float, seed: int, num_frames: int, fps: int
    ):
        """Validate trained LoRA model (images or videos)"""
        try:
            if self.validator is None:
                self.validator = ModelValidator(self.config)

            files, metrics = self.validator.validate(
                lora_path=lora_path,
                base_model=base_model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_samples=int(num_samples),
                steps=int(steps),
                guidance_scale=guidance_scale,
                seed=int(seed),
                num_frames=int(num_frames),
                fps=int(fps)
            )

            return files, metrics

        except Exception as e:
            return [], {"error": str(e)}

    def save_settings(self, device: str, vram_opt: str, cpu_threads: int,
                     qwen_backend: str, vllm_url: str, cache_dir: str) -> str:
        """Save application settings"""
        try:
            self.config.update({
                'device': device,
                'vram_optimization': vram_opt,
                'cpu_threads': int(cpu_threads),
                'qwen_backend': qwen_backend,
                'vllm_server_url': vllm_url,
                'cache_dir': cache_dir
            })
            self.config.save()
            return "Settings saved successfully"
        except Exception as e:
            return f"Error: {str(e)}"

    def _merge_tags(self, tags: List[str], format: str) -> str:
        """Merge tags from multiple sources"""
        if format == "comma_separated":
            return ", ".join(tags)
        elif format == "line_separated":
            return "\n".join(tags)
        elif format == "json":
            return json.dumps(tags, indent=2)
        return ", ".join(tags)

    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        app = self.build_interface()
        app.launch(**kwargs)


if __name__ == "__main__":
    gui = LoRATrainerGUI()
    gui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
