#!/usr/bin/env python3
"""
LoRA training GUI with automated tagging and validation
Supports Flux and WAN I2V models
"""

import gradio as gr
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

sys.path.append(str(Path(__file__).parent))

from modules.clip_interrogator_module import CLIPInterrogatorModule
from modules.qwen_tagger import QwenVLTagger
from modules.dataset_manager import DatasetManager
from modules.lora_trainer import LoRATrainer
from modules.validator import ModelValidator
from modules.config_manager import ConfigManager
from modules.video_inference import VideoInferenceModule


class LoRATrainerGUI:

    def __init__(self):
        self.config = ConfigManager()
        self.dataset_manager = DatasetManager(self.config)
        self.clip_interrogator = None
        self.qwen_tagger = None
        self.lora_trainer = None
        self.validator = None
        self.video_inference = VideoInferenceModule()
        self.training_thread = None

        # Centralized VLM Settings
        self.vlm_settings = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "backend": "auto",
            "api_url": "http://localhost",
            "api_port": 8000
        }

    def _get_qwen_tagger(self):
        """Get or initialize QwenVLTagger with current settings"""
        model = self.vlm_settings["model"]
        backend = self.vlm_settings["backend"]
        url = self.vlm_settings["api_url"]
        port = int(self.vlm_settings["api_port"])

        if (self.qwen_tagger is None or 
            self.qwen_tagger.model_name != model or 
            self.qwen_tagger.vllm_server_url != url or 
            self.qwen_tagger.vllm_port != port):
            
            if self.qwen_tagger:
                self.qwen_tagger.unload()
            
            self.qwen_tagger = QwenVLTagger(
                model_name=model,
                backend=backend,
                vllm_server_url=url,
                vllm_port=port
            )
        
        return self.qwen_tagger

    def build_vlm_settings_tab(self):
        gr.Markdown("## Centralized VLM Settings")
        gr.Markdown("Configure your Vision Language Model once for all features (Auto Tagging, Smart Crop, Video Inference).")
        
        with gr.Row():
            with gr.Column():
                vlm_model = gr.Dropdown(
                    label="VLM Model",
                    choices=["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-Thinking"],
                    value=self.vlm_settings["model"],
                    allow_custom_value=True
                )
                
                vlm_backend = gr.Radio(
                    label="Backend",
                    choices=["auto", "vllm", "direct"],
                    value=self.vlm_settings["backend"],
                    info="auto: try vLLM first, fallback to direct"
                )

            with gr.Column():
                api_url = gr.Textbox(
                    label="API URL (vLLM/KoboldCPP)",
                    value=self.vlm_settings["api_url"]
                )
                api_port = gr.Number(
                    label="API Port",
                    value=self.vlm_settings["api_port"],
                    precision=0
                )
                refresh_btn = gr.Button("🔄 Query API for Models")

        save_btn = gr.Button("Save VLM Settings", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)

        def update_settings(model, backend, url, port):
            self.vlm_settings["model"] = model
            self.vlm_settings["backend"] = backend
            self.vlm_settings["api_url"] = url
            self.vlm_settings["api_port"] = port
            return f"Settings updated! Model: {model}, Backend: {backend}"

        save_btn.click(
            fn=update_settings,
            inputs=[vlm_model, vlm_backend, api_url, api_port],
            outputs=[status]
        )

        refresh_btn.click(
            fn=self.refresh_models,
            inputs=[api_url, api_port],
            outputs=[vlm_model]
        )

    def build_video_inference_tab(self):
        gr.Markdown("## Video Inference & Tagging")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Input Video")
                
                video_metadata = gr.JSON(label="Video Metadata", visible=False)
                
                sample_strategy = gr.Dropdown(
                    label="Sampling Strategy",
                    choices=["uniform", "interval", "fps", "all_frames"],
                    value="uniform",
                    info="uniform: n frames evenly spaced | interval: every n seconds | fps: resample to specific FPS | all_frames: entire video"
                )
                
                with gr.Row():
                    num_frames = gr.Number(label="Number of Frames", value=8, precision=0, visible=True)
                    interval_sec = gr.Number(label="Interval/FPS", value=1.0, visible=False, info="Seconds for 'interval' or target FPS for 'fps'")

                prompt = gr.Textbox(
                    label="Prompt",
                    value="Describe this video in detail, focusing on the main action, setting, and atmosphere.",
                    lines=3
                )
                
                with gr.Accordion("Inference Parameters", open=False):
                    max_tokens = gr.Number(label="Max New Tokens", value=2000)
                    temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=0.7)
                    top_p = gr.Slider(label="Top P", minimum=0.1, maximum=1.0, value=0.9)
                    repetition_penalty = gr.Slider(label="Repetition Penalty", minimum=1.0, maximum=2.0, value=1.1)

                run_btn = gr.Button("Run Video Inference", variant="primary")

            with gr.Column():
                output_text = gr.Textbox(label="VLM Response", lines=10, interactive=False)
                extracted_frames = gr.Gallery(label="Extracted Frames", columns=4)

        def update_sampling_controls(strategy):
            if strategy == "uniform":
                return gr.update(visible=True), gr.update(visible=False)
            elif strategy in ["interval", "fps"]:
                return gr.update(visible=False), gr.update(visible=True)
            else:  # all_frames
                return gr.update(visible=False), gr.update(visible=False)

        def on_video_upload(video_path):
            if not video_path:
                return gr.update(visible=False), {}
            try:
                metadata = self.video_inference.get_video_info(video_path)
                return gr.update(visible=True), metadata
            except:
                return gr.update(visible=False), {}

        def run_video_inference(video_path, strategy, n_frames, interval, prompt, max_tok, temp, top_p, rep_pen):
            if not video_path:
                return "Please upload a video.", []
            
            try:
                frames = self.video_inference.extract_frames(
                    video_path, strategy, int(n_frames), interval
                )
                
                tagger = self._get_qwen_tagger()
                response = tagger.tag_video(
                    frames, prompt, int(max_tok), temp, top_p, 50, rep_pen
                )
                return response, frames
            except Exception as e:
                return f"Error: {str(e)}", []

        sample_strategy.change(
            fn=update_sampling_controls,
            inputs=[sample_strategy],
            outputs=[num_frames, interval_sec]
        )

        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[video_metadata, video_metadata]
        )

        run_btn.click(
            fn=run_video_inference,
            inputs=[video_input, sample_strategy, num_frames, interval_sec, prompt, max_tokens, temperature, top_p, repetition_penalty],
            outputs=[output_text, extracted_frames]
        )

    def build_interface(self):
        with gr.Blocks(title="LoRA Trainer Suite") as app:
            gr.Markdown(
                """
                # 🎨 LoRA Trainer Suite
                Complete workflow for training high-quality LoRA models
                **Hardware:** RTX 3090 | 128GB RAM | Ryzen 9 7900X
                """
            )

            with gr.Tabs():
                with gr.Tab("⚙️ VLM Settings"):
                    self.build_vlm_settings_tab()

                with gr.Tab("📁 Dataset Preparation"):
                    self.build_dataset_tab()

                with gr.Tab("🏷️ Auto Tagging"):
                    self.build_tagging_tab()

                with gr.Tab("🎥 Video Inference"):
                    self.build_video_inference_tab()

                with gr.Tab("🚀 LoRA Training"):
                    self.build_training_tab()

                with gr.Tab("✅ Validation"):
                    self.build_validation_tab()

                with gr.Tab("🔧 App Settings"):
                    self.build_settings_tab()

        return app

    def build_dataset_tab(self):
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
                gr.Markdown("### Smart Preprocessing")
                target_size = gr.Number(label="Target Size", value=512, info="Square crop size (512 or 1024)")
                yolo_model = gr.Dropdown(
                    label="YOLO Model",
                    choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
                    value="yolov8n.pt"
                )
                smart_prep_btn = gr.Button("Start Smart Preprocessing", variant="primary")
                
                gr.Markdown("*VLM settings are configured in the 'VLM Settings' tab*")

            with gr.Column():
                augment_flip = gr.Checkbox(label="Horizontal Flip", value=False)
                augment_rotate = gr.Checkbox(label="Random Rotation", value=False)
                augment_btn = gr.Button("Apply Augmentation")

        process_status = gr.Textbox(label="Status", interactive=False)
        
        with gr.Accordion("Review Smart Crops", open=False):
            with gr.Tabs():
                with gr.Tab("Processed (YOLO)"):
                    processed_gallery = gr.Gallery(label="Processed Images", columns=6, height=300)
                with gr.Tab("VLM Modified"):
                    vlm_gallery = gr.Gallery(label="VLM Modified (Outliers)", columns=4, height=300)
                with gr.Tab("Manual Review"):
                    manual_gallery = gr.Gallery(label="Manual Review Needed", columns=4, height=300)

        load_btn.click(
            fn=self.load_dataset,
            inputs=[dataset_path],
            outputs=[dataset_info, image_preview, process_status]
        )
        
        smart_prep_btn.click(
            fn=self.smart_preprocess_dataset,
            inputs=[dataset_path, target_size, yolo_model],
            outputs=[process_status, processed_gallery, vlm_gallery, manual_gallery]
        )
        
        augment_btn.click(
            fn=self.augment_dataset,
            inputs=[dataset_path, augment_flip, augment_rotate],
            outputs=[process_status]
        )

    def build_tagging_tab(self):
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
                
                gr.Markdown("*Configure Qwen Model, Backend, and API in 'VLM Settings' tab*")

                qwen_prompt = gr.Textbox(
                    label="Qwen Custom Prompt",
                    value="Describe this image in detail, including all visible elements, style, composition, and artistic qualities.",
                    lines=3
                )

                gr.Markdown("### LLM Inference Parameters")
                gr.Markdown("*Adjust these to control Qwen VL output quality and prevent repetition*")

                with gr.Row():
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.05,
                        info="Higher = more creative, lower = more focused"
                    )
                    top_p = gr.Slider(
                        label="Top P (Nucleus)",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        info="Nucleus sampling threshold"
                    )

                with gr.Row():
                    top_k = gr.Slider(
                        label="Top K",
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        info="Consider top K tokens"
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition Penalty",
                        minimum=1.0,
                        maximum=2.0,
                        value=1.15,
                        step=0.05,
                        info="Penalize repeated tokens (important for Qwen3-VL!)"
                    )

                with gr.Row():
                    presence_penalty = gr.Slider(
                        label="Presence Penalty",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        info="Penalize tokens that have appeared"
                    )
                    frequency_penalty = gr.Slider(
                        label="Frequency Penalty",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        info="Penalize based on token frequency"
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

        start_tag_btn.click(
            fn=self.start_auto_tagging,
            inputs=[
                tag_dataset_path, use_clip, clip_mode,
                use_qwen, qwen_prompt,
                temperature, top_p, top_k, repetition_penalty,
                presence_penalty, frequency_penalty,
                merge_tags, tag_format
            ],
            outputs=[tagging_progress, current_image, current_tags]
        )

    def build_training_tab(self):
        gr.Markdown("## LoRA Training Configuration")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model Configuration")

                base_model = gr.Dropdown(
                    label="Base Model",
                    choices=[
                        "black-forest-labs/FLUX.1-dev",
                        "black-forest-labs/FLUX.1-schnell",
                        "Wan-AI/Wan2.1-I2V-14B-720P",
                        "Wan-AI/Wan2.2-I2V-A14B",
                    ],
                    value="black-forest-labs/FLUX.1-dev",
                    info="Flux for images, WAN I2V (Qwen video) to animate images"
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
                        "Wan-AI/Wan2.1-I2V-14B-720P",
                        "Wan-AI/Wan2.2-I2V-A14B",
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
                    - Support for Flux (images) and WAN I2V 2.1/2.2 (Qwen video)
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

                    **WAN I2V:** Qwen's Image-to-Video diffusion models
                    (Wan-AI/Wan2.1-I2V-14B-720P, Wan-AI/Wan2.2-I2V-A14B)
                    Workflow: Flux generates image → WAN I2V animates it
                    """
                )

                save_settings_btn = gr.Button("Save Settings", variant="primary")

        settings_status = gr.Textbox(label="Status", interactive=False)

        save_settings_btn.click(
            fn=self.save_settings,
            inputs=[device, vram_optimization, cpu_threads, default_qwen_backend, vllm_server_url, cache_dir],
            outputs=[settings_status]
        )

    def load_dataset(self, dataset_path: str) -> Tuple[Dict, List, str]:
        try:
            info = self.dataset_manager.load_dataset(dataset_path)
            preview_images = self.dataset_manager.get_preview_images(limit=12)
            return info, preview_images, "Dataset loaded successfully"
        except Exception as e:
            return {}, [], f"Error: {str(e)}"

    def resize_images(self, dataset_path: str, width: int, height: int) -> str:
        try:
            self.dataset_manager.resize_images(dataset_path, width, height)
            return f"Images resized to {width}x{height}"
        except Exception as e:
            return f"Error: {str(e)}"

    def smart_preprocess_dataset(
        self, dataset_path: str, target_size: int, yolo_model: str
    ):
        try:
            # Initialize tagger from central settings
            self.qwen_tagger = self._get_qwen_tagger()
            
            results = self.dataset_manager.smart_preprocess(
                dataset_path=dataset_path,
                target_size=int(target_size),
                yolo_model=yolo_model,
                tagger=self.qwen_tagger
            )
            
            msg = (
                f"Smart preprocessing complete!\n"
                f"Processed: {len(results['processed'])}\n"
                f"VLM Modified: {len(results['vlm_modified'])}\n"
                f"Manual Review: {len(results['manual_review'])}"
            )
            
            return msg, results['processed'], results['vlm_modified'], results['manual_review']
            
        except Exception as e:
            return f"Error: {str(e)}", [], [], []

    def augment_dataset(self, dataset_path: str, flip: bool, rotate: bool) -> str:
        try:
            self.dataset_manager.augment(dataset_path, flip, rotate)
            return "Augmentation applied successfully"
        except Exception as e:
            return f"Error: {str(e)}"

    def start_auto_tagging(
        self, dataset_path: str, use_clip: bool, clip_mode: str,
        use_qwen: bool, qwen_prompt: str,
        temperature: float, top_p: float, top_k: int, repetition_penalty: float,
        presence_penalty: float, frequency_penalty: float,
        merge_tags: bool, tag_format: str
    ):
        try:
            # load qwen first to check backend
            qwen_using_vllm = False
            if use_qwen:
                self.qwen_tagger = self._get_qwen_tagger()
                yield f"Loading {self.qwen_tagger.model_name} (backend: {self.qwen_tagger.backend})...\n", None, ""
                qwen_using_vllm = self.qwen_tagger.active_backend == "vllm"
                
            if qwen_using_vllm:
                yield f"Using vLLM backend at {self.qwen_tagger.vllm_server_url}:{self.qwen_tagger.vllm_port}\n", None, ""
            else:
                yield "Using direct transformers backend\n", None, ""

            # skip CLIP if vLLM is running (VLM is good enough on its own)
            skip_clip = use_clip and qwen_using_vllm
            if skip_clip:
                yield "⚡ vLLM detected! Skipping CLIP Interrogator (using VLM only for faster processing)\n", None, ""
                use_clip = False

            if use_clip and self.clip_interrogator is None:
                yield "Loading CLIP Interrogator...", None, ""
                self.clip_interrogator = CLIPInterrogatorModule()

            images = self.dataset_manager.get_all_images(dataset_path)
            total = len(images)

            for idx, img_path in enumerate(images):
                progress_text = f"Processing {idx+1}/{total}: {os.path.basename(img_path)}\n"

                tags = []

                if use_clip:
                    clip_tags = self.clip_interrogator.interrogate(img_path, mode=clip_mode)
                    tags.append(clip_tags)
                    progress_text += f"CLIP: {clip_tags}\n"

                if use_qwen:
                    qwen_tags = self.qwen_tagger.tag_image(
                        img_path,
                        prompt=qwen_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=int(top_k),
                        repetition_penalty=repetition_penalty,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty
                    )
                    tags.append(qwen_tags)
                    progress_text += f"Qwen VL: {qwen_tags}\n"

                final_tags = self._merge_tags(tags, tag_format)
                self.dataset_manager.save_tags(img_path, final_tags, merge_tags)

                yield progress_text, img_path, final_tags

            yield f"Tagging complete! Processed {total} images.", None, ""

        except Exception as e:
            yield f"Error: {str(e)}\n", None, ""

    def refresh_models(self, api_url: str, api_port: int):
        """Query API for available models"""
        try:
            import requests
            url = f"{api_url}:{int(api_port)}/v1/models"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [m['id'] for m in data['data']]
                return gr.Dropdown(choices=models, value=models[0] if models else None)
            else:
                return gr.Dropdown(choices=["Error: API returned " + str(response.status_code)])
        except Exception as e:
            return gr.Dropdown(choices=["Error: " + str(e)])

    def start_training(self, *args):
        try:
            if self.lora_trainer is None:
                self.lora_trainer = LoRATrainer(self.config)

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

            for progress, loss_data, samples in self.lora_trainer.train(training_config):
                yield progress, loss_data, samples

        except Exception as e:
            yield f"Error: {str(e)}", None, []

    def validate_model(
        self, lora_path: str, base_model: str, prompt: str,
        negative_prompt: str, num_samples: int, steps: int,
        guidance_scale: float, seed: int, num_frames: int, fps: int
    ):
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
        if format == "comma_separated":
            return ", ".join(tags)
        elif format == "line_separated":
            return "\n".join(tags)
        elif format == "json":
            return json.dumps(tags, indent=2)
        return ", ".join(tags)

    def launch(self, **kwargs):
        # Check VLM directory configuration
        self._check_vlm_config()
        
        app = self.build_interface()
        
        # Add permitted paths
        allowed_paths = kwargs.pop('allowed_paths', [])
        allowed_paths.extend([
            self.config.get("dataset_path", ""),
            self.config.get("cache_dir", "./models"),
            os.getcwd()
        ])
        
        # Remove duplicates and empty strings
        allowed_paths = list(set([p for p in allowed_paths if p]))
        
        app.launch(allowed_paths=allowed_paths, **kwargs)

    def _check_vlm_config(self):
        """Check if VLM directory is configured, prompt if not"""
        # Check if vLLM is available (mock check for now, or check import)
        vllm_available = False
        try:
            import vllm
            vllm_available = True
        except ImportError:
            pass
            
        if not vllm_available:
            # Check config for VLM directory
            vlm_dir = self.config.get("vlm_dir")
            if not vlm_dir:
                print("\n" + "!"*80)
                print("vLLM not found or configured!")
                print("Please enter the directory where your VLM models are stored (or press Enter to skip):")
                user_input = input("VLM Directory > ").strip()
                if user_input:
                    self.config.update({"vlm_dir": user_input})
                    self.config.save()
                    print(f"Saved VLM directory: {user_input}")
                else:
                    print("Skipping VLM directory configuration.")
                print("!"*80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Trainer Suite GUI")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--allowed-paths", nargs="+", default=[], help="Additional allowed paths")

    args = parser.parse_args()

    gui = LoRATrainerGUI()
    gui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        allowed_paths=args.allowed_paths
    )
