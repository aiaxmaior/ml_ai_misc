"""
LoRA Trainer Module
Integrates with training frameworks for Flux and Stable Diffusion models
Supports Kohya_ss and direct diffusers training
Optimized for RTX 3090
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Generator, Tuple, List
import torch
import time
from datetime import datetime


class LoRATrainer:
    """LoRA training manager with Kohya_ss and diffusers support"""

    def __init__(self, config=None):
        """
        Initialize LoRA Trainer

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.training_process = None
        self.training_active = False
        self.logs = []

    def train(self, training_config: Dict) -> Generator:
        """
        Train LoRA model with progress updates

        Args:
            training_config: Training configuration dictionary

        Yields:
            Tuple of (progress_text, loss_plot_data, sample_images)
        """
        # Determine training method based on model
        base_model = training_config['base_model']

        if 'flux' in base_model.lower():
            yield from self._train_flux(training_config)
        elif 'stable-diffusion' in base_model.lower():
            yield from self._train_sd(training_config)
        else:
            yield f"Unsupported model: {base_model}", None, []

    def _train_flux(self, config: Dict) -> Generator:
        """Train Flux LoRA using ai-toolkit or Kohya_ss"""
        try:
            yield "Preparing Flux LoRA training...\n", None, []

            # Create output directory
            output_dir = Path(config['output_dir']) / config['output_name']
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save training config
            config_path = output_dir / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Check for ai-toolkit (preferred for Flux)
            if self._check_ai_toolkit_available():
                yield "Using ai-toolkit for Flux training...\n", None, []
                yield from self._train_flux_ai_toolkit(config, output_dir)
            else:
                # Fallback to Kohya_ss if available
                yield "ai-toolkit not found, using Kohya_ss...\n", None, []
                yield from self._train_flux_kohya(config, output_dir)

        except Exception as e:
            yield f"Error during Flux training: {str(e)}\n", None, []

    def _train_sd(self, config: Dict) -> Generator:
        """Train Stable Diffusion LoRA using Kohya_ss or diffusers"""
        try:
            yield "Preparing Stable Diffusion LoRA training...\n", None, []

            # Create output directory
            output_dir = Path(config['output_dir']) / config['output_name']
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save training config
            config_path = output_dir / "training_config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Check for Kohya_ss
            if self._check_kohya_available():
                yield "Using Kohya_ss for SD training...\n", None, []
                yield from self._train_sd_kohya(config, output_dir)
            else:
                # Fallback to direct diffusers training
                yield "Kohya_ss not found, using diffusers...\n", None, []
                yield from self._train_sd_diffusers(config, output_dir)

        except Exception as e:
            yield f"Error during SD training: {str(e)}\n", None, []

    def _check_ai_toolkit_available(self) -> bool:
        """Check if ai-toolkit is available"""
        # Check for ai-toolkit in common locations
        toolkit_paths = [
            Path.home() / "ai-toolkit",
            Path("./ai-toolkit"),
            Path("../ai-toolkit"),
        ]

        for path in toolkit_paths:
            if (path / "run.py").exists() or (path / "train.py").exists():
                return True

        return False

    def _check_kohya_available(self) -> bool:
        """Check if Kohya_ss is available"""
        # Check for Kohya_ss in common locations
        kohya_paths = [
            Path.home() / "kohya_ss",
            Path("./kohya_ss"),
            Path("../kohya_ss"),
        ]

        for path in kohya_paths:
            if (path / "train_network.py").exists():
                return True

        # Check if installed as package
        try:
            import kohya_ss
            return True
        except ImportError:
            return False

    def _train_flux_ai_toolkit(self, config: Dict, output_dir: Path) -> Generator:
        """Train Flux using ai-toolkit"""
        # Create ai-toolkit config
        toolkit_config = {
            "job": "train",
            "config": {
                "name": config['output_name'],
                "process": [
                    {
                        "type": "flux_lora",
                        "training_folder": str(output_dir),
                        "device": "cuda:0",
                        "network": {
                            "type": "lora",
                            "linear": config.get('lora_rank', 16),
                            "linear_alpha": config.get('lora_alpha', 32)
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": config.get('save_every_n_steps', 100),
                            "max_step_saves_to_keep": 5
                        },
                        "datasets": [
                            {
                                "folder_path": config['dataset_path'],
                                "caption_ext": "txt",
                                "caption_dropout_rate": 0.05,
                                "shuffle_tokens": False,
                                "cache_latents_to_disk": True,
                                "resolution": [512, 512]
                            }
                        ],
                        "train": {
                            "batch_size": config.get('batch_size', 1),
                            "steps": config.get('max_train_steps', 1000),
                            "gradient_accumulation_steps": config.get('gradient_accumulation_steps', 1),
                            "train_unet": True,
                            "train_text_encoder": False,
                            "gradient_checkpointing": config.get('gradient_checkpointing', True),
                            "noise_scheduler": "flowmatch",
                            "optimizer": config.get('optimizer', 'adamw8bit').lower(),
                            "lr": config.get('learning_rate', 1e-4),
                            "ema_config": {
                                "use_ema": True,
                                "ema_decay": 0.99
                            },
                            "dtype": "bf16"
                        },
                        "model": {
                            "name_or_path": config['base_model'],
                            "is_flux": True,
                            "quantize": True
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": config.get('sample_every_n_steps', 100),
                            "width": 512,
                            "height": 512,
                            "prompts": [config.get('sample_prompt', 'a photo')],
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 4,
                            "sample_steps": 20
                        }
                    }
                ]
            }
        }

        # Save toolkit config
        toolkit_config_path = output_dir / "ai_toolkit_config.yaml"
        import yaml
        with open(toolkit_config_path, 'w') as f:
            yaml.dump(toolkit_config, f)

        yield f"AI-toolkit config saved to {toolkit_config_path}\n", None, []
        yield "To train with ai-toolkit, run:\n", None, []
        yield f"  python /path/to/ai-toolkit/run.py {toolkit_config_path}\n", None, []
        yield "\nNote: Integration with ai-toolkit requires manual setup.\n", None, []

    def _train_flux_kohya(self, config: Dict, output_dir: Path) -> Generator:
        """Train Flux using Kohya_ss (if supported)"""
        yield "Note: Flux training with Kohya_ss requires latest version with Flux support.\n", None, []
        yield "Please use ai-toolkit or manual training setup for Flux models.\n", None, []

    def _train_sd_kohya(self, config: Dict, output_dir: Path) -> Generator:
        """Train SD using Kohya_ss"""
        try:
            # Find Kohya_ss installation
            kohya_path = self._find_kohya_path()
            if not kohya_path:
                yield "Kohya_ss not found. Please install from: https://github.com/kohya-ss/sd-scripts\n", None, []
                return

            # Build Kohya command
            train_script = kohya_path / "train_network.py"

            cmd = [
                sys.executable,
                str(train_script),
                f"--pretrained_model_name_or_path={config['base_model']}",
                f"--train_data_dir={config['dataset_path']}",
                f"--output_dir={output_dir}",
                f"--output_name={config['output_name']}",
                f"--train_batch_size={config.get('batch_size', 1)}",
                f"--learning_rate={config.get('learning_rate', 1e-4)}",
                f"--max_train_steps={config.get('max_train_steps', 1000)}",
                f"--network_module=networks.lora",
                f"--network_dim={config.get('lora_rank', 16)}",
                f"--network_alpha={config.get('lora_alpha', 32)}",
                f"--optimizer_type={config.get('optimizer', 'AdamW8bit')}",
                f"--gradient_accumulation_steps={config.get('gradient_accumulation_steps', 1)}",
                f"--mixed_precision={config.get('mixed_precision', 'bf16')}",
                f"--save_every_n_steps={config.get('save_every_n_steps', 100)}",
                "--save_model_as=safetensors",
                "--caption_extension=.txt",
                "--cache_latents",
                "--seed=42",
            ]

            if config.get('gradient_checkpointing', True):
                cmd.append("--gradient_checkpointing")

            # Add sample generation
            if config.get('sample_prompt'):
                cmd.extend([
                    f"--sample_every_n_steps={config.get('sample_every_n_steps', 100)}",
                    f"--sample_prompts={config.get('sample_prompt')}",
                    f"--sample_sampler=euler_a",
                ])

            yield f"Starting Kohya_ss training...\n", None, []
            yield f"Command: {' '.join(cmd)}\n\n", None, []

            # Run training process
            yield from self._run_training_process(cmd, output_dir)

        except Exception as e:
            yield f"Error: {str(e)}\n", None, []

    def _train_sd_diffusers(self, config: Dict, output_dir: Path) -> Generator:
        """Train SD using HuggingFace diffusers"""
        try:
            yield "Using HuggingFace diffusers for training...\n", None, []

            # Import required libraries
            from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
            from diffusers.optimization import get_scheduler
            from transformers import CLIPTextModel, CLIPTokenizer
            from peft import LoraConfig, get_peft_model
            import torch.nn.functional as F
            from torch.utils.data import Dataset, DataLoader
            from PIL import Image

            # This is a simplified version - full implementation would be more complex
            yield "Setting up diffusers training pipeline...\n", None, []

            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load models
            yield "Loading base model...\n", None, []

            # [Training code would go here - this is a placeholder]
            yield "Note: Direct diffusers training requires full implementation.\n", None, []
            yield "Consider using Kohya_ss or ai-toolkit for production training.\n", None, []

        except Exception as e:
            yield f"Error: {str(e)}\n", None, []

    def _find_kohya_path(self) -> Optional[Path]:
        """Find Kohya_ss installation path"""
        paths = [
            Path.home() / "kohya_ss",
            Path.home() / "sd-scripts",
            Path("./kohya_ss"),
            Path("./sd-scripts"),
            Path("../kohya_ss"),
            Path("../sd-scripts"),
        ]

        for path in paths:
            if (path / "train_network.py").exists():
                return path

        return None

    def _run_training_process(self, cmd: List[str], output_dir: Path) -> Generator:
        """
        Run training process and yield progress updates

        Args:
            cmd: Command to run
            output_dir: Output directory

        Yields:
            Training progress updates
        """
        try:
            self.training_active = True
            loss_history = []

            # Start process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            self.training_process = process

            # Monitor output
            for line in process.stdout:
                self.logs.append(line.strip())

                # Parse loss from output
                if "loss:" in line.lower() or "loss=" in line.lower():
                    try:
                        # Extract loss value (format varies by framework)
                        loss_val = self._extract_loss(line)
                        if loss_val is not None:
                            loss_history.append(loss_val)
                    except:
                        pass

                # Check for sample images
                sample_images = list(output_dir.glob("**/sample-*.png"))

                # Prepare loss plot data
                loss_plot = None
                if loss_history:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=loss_history,
                        mode='lines',
                        name='Training Loss'
                    ))
                    fig.update_layout(
                        title="Training Loss",
                        xaxis_title="Step",
                        yaxis_title="Loss",
                        height=300
                    )
                    loss_plot = fig

                # Yield progress
                yield line, loss_plot, [str(img) for img in sample_images[-4:]]

            # Wait for completion
            process.wait()

            if process.returncode == 0:
                yield "\n✓ Training completed successfully!\n", loss_plot, [str(img) for img in sample_images]
            else:
                yield f"\n✗ Training failed with code {process.returncode}\n", loss_plot, []

        except Exception as e:
            yield f"\nError during training: {str(e)}\n", None, []
        finally:
            self.training_active = False
            self.training_process = None

    def _extract_loss(self, line: str) -> Optional[float]:
        """Extract loss value from log line"""
        import re

        # Try various loss patterns
        patterns = [
            r"loss[:\s=]+([0-9.]+)",
            r"loss_total[:\s=]+([0-9.]+)",
            r"train_loss[:\s=]+([0-9.]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, line.lower())
            if match:
                return float(match.group(1))

        return None

    def stop_training(self):
        """Stop active training process"""
        if self.training_process:
            self.training_process.terminate()
            self.training_active = False
            return "Training stopped"
        return "No active training"

    def get_training_logs(self) -> List[str]:
        """Get training logs"""
        return self.logs


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LoRA Trainer CLI")
    parser.add_argument("--config", required=True, help="Path to training config JSON")

    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    trainer = LoRATrainer()

    print("Starting training...")
    for progress, loss_plot, samples in trainer.train(config):
        print(progress, end='')

    print("\nTraining complete!")
