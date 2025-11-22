"""
Model Validator Module
Validates trained LoRA models by generating test images and computing quality metrics
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random
from PIL import Image
import numpy as np


class ModelValidator:
    """Validate LoRA models with inference and quality metrics"""

    def __init__(self, config=None):
        """
        Initialize Model Validator

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.pipeline = None
        self.current_model = None

    def validate(
        self,
        lora_path: str,
        base_model: str,
        prompt: str,
        negative_prompt: str = "",
        num_samples: int = 4,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = -1,
        num_frames: int = 16,  # For video models
        fps: int = 8  # For video models
    ) -> Tuple[List[str], Dict]:
        """
        Validate LoRA model by generating samples (images or videos)

        Args:
            lora_path: Path to LoRA weights
            base_model: Base model name
            prompt: Generation prompt
            negative_prompt: Negative prompt
            num_samples: Number of samples to generate
            steps: Inference steps
            guidance_scale: Guidance scale
            seed: Random seed (-1 for random)
            num_frames: Number of frames for video models
            fps: Frames per second for video models

        Returns:
            Tuple of (file_paths, metrics)
        """
        try:
            # Determine if this is a video model
            is_video = 'wan' in base_model.lower()

            # Load pipeline if needed
            if self.pipeline is None or self.current_model != base_model:
                self._load_pipeline(base_model)

            # Load LoRA weights
            self._load_lora(lora_path)

            # Setup output directory
            output_dir = Path(lora_path).parent / "validation_samples"
            output_dir.mkdir(exist_ok=True, parents=True)

            file_paths = []
            samples = []

            if is_video:
                # Generate video samples
                videos = self._generate_video_samples(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_samples=num_samples,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    num_frames=num_frames,
                    fps=fps
                )

                # Save videos and extract preview frames
                for idx, video_frames in enumerate(videos):
                    # Save as GIF
                    gif_path = output_dir / f"video_{idx:04d}.gif"
                    video_frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=video_frames[1:],
                        duration=1000 // fps,
                        loop=0
                    )
                    file_paths.append(str(gif_path))

                    # Also save first frame as preview
                    preview_path = output_dir / f"video_{idx:04d}_frame0.png"
                    video_frames[0].save(preview_path)
                    samples.append(video_frames[0])

            else:
                # Generate image samples
                images = self._generate_samples(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_samples=num_samples,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )

                # Save images
                for idx, img in enumerate(images):
                    img_path = output_dir / f"sample_{idx:04d}.png"
                    img.save(img_path)
                    file_paths.append(str(img_path))
                    samples.append(img)

            # Compute quality metrics
            metrics = self._compute_metrics(samples, prompt)
            metrics['is_video'] = is_video
            if is_video:
                metrics['num_frames'] = num_frames
                metrics['fps'] = fps

            return file_paths, metrics

        except Exception as e:
            return [], {"error": str(e)}

    def _load_pipeline(self, base_model: str):
        """Load diffusion pipeline"""
        try:
            print(f"Loading base model: {base_model}")

            if 'flux' in base_model.lower():
                self._load_flux_pipeline(base_model)
            elif 'wan' in base_model.lower():
                self._load_wan_pipeline(base_model)
            elif 'stable-diffusion' in base_model.lower():
                self._load_sd_pipeline(base_model)
            else:
                raise ValueError(f"Unsupported model: {base_model}")

            self.current_model = base_model
            print("✓ Pipeline loaded")

        except Exception as e:
            raise RuntimeError(f"Failed to load pipeline: {str(e)}")

    def _load_flux_pipeline(self, model_name: str):
        """Load Flux pipeline"""
        from diffusers import FluxPipeline

        self.pipeline = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir="./models/flux"
        )

        # Enable optimizations
        if hasattr(self.pipeline, 'enable_model_cpu_offload'):
            self.pipeline.enable_model_cpu_offload()

    def _load_sd_pipeline(self, model_name: str):
        """Load Stable Diffusion pipeline"""
        from diffusers import StableDiffusionPipeline

        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir="./models/sd"
        )

        self.pipeline = self.pipeline.to("cuda")

        # Enable optimizations
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()

        # Enable xformers if available
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass

    def _load_wan_pipeline(self, model_name: str):
        """Load WAN video diffusion pipeline"""
        try:
            # WAN uses similar pipeline structure to SD but for video
            # Note: This assumes WAN models are compatible with diffusers
            # May need adjustment based on actual WAN implementation
            from diffusers import DiffusionPipeline

            self.pipeline = DiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir="./models/wan",
                trust_remote_code=True  # WAN may need custom code
            )

            # Enable optimizations
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()

            if hasattr(self.pipeline, 'enable_vae_slicing'):
                self.pipeline.enable_vae_slicing()

            # Enable xformers if available
            try:
                self.pipeline.enable_xformers_memory_efficient_attention()
            except:
                pass

            print("✓ WAN video pipeline loaded")

        except Exception as e:
            raise RuntimeError(f"Failed to load WAN pipeline: {str(e)}")

    def _load_lora(self, lora_path: str):
        """Load LoRA weights into pipeline"""
        lora_path = Path(lora_path)

        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA not found: {lora_path}")

        # Determine file type
        if lora_path.is_dir():
            # Load from directory
            self.pipeline.load_lora_weights(str(lora_path))
        elif lora_path.suffix in ['.safetensors', '.pt', '.bin']:
            # Load from file
            parent_dir = lora_path.parent
            weight_name = lora_path.name
            self.pipeline.load_lora_weights(str(parent_dir), weight_name=weight_name)
        else:
            raise ValueError(f"Unsupported LoRA format: {lora_path.suffix}")

        print(f"✓ LoRA loaded: {lora_path.name}")

    def _generate_samples(
        self,
        prompt: str,
        negative_prompt: str,
        num_samples: int,
        steps: int,
        guidance_scale: float,
        seed: int
    ) -> List[Image.Image]:
        """Generate sample images"""
        images = []

        for i in range(num_samples):
            # Set seed
            if seed >= 0:
                generator = torch.Generator(device="cuda").manual_seed(seed + i)
            else:
                generator = None

            # Generate
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=512,
                    width=512,
                )

            images.append(result.images[0])

        return images

    def _generate_video_samples(
        self,
        prompt: str,
        negative_prompt: str,
        num_samples: int,
        steps: int,
        guidance_scale: float,
        seed: int,
        num_frames: int = 16,
        fps: int = 8
    ) -> List[List[Image.Image]]:
        """Generate video samples (list of frame lists)"""
        videos = []

        for i in range(num_samples):
            # Set seed
            if seed >= 0:
                generator = torch.Generator(device="cuda").manual_seed(seed + i)
            else:
                generator = None

            # Generate video
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    height=512,
                    width=512,
                    num_frames=num_frames,
                )

            # Extract frames
            # WAN output format may vary - handle both video tensor and frame list
            if hasattr(result, 'frames'):
                frames = result.frames[0]  # First video in batch
            elif hasattr(result, 'images'):
                frames = result.images  # May be list of frames
            else:
                # Fallback: treat as video tensor
                frames = result[0] if isinstance(result, list) else [result]

            # Convert frames to PIL Images if needed
            pil_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    pil_frames.append(frame)
                elif isinstance(frame, torch.Tensor):
                    # Convert tensor to PIL
                    frame_array = frame.cpu().numpy()
                    if frame_array.ndim == 3:
                        # CHW to HWC
                        frame_array = frame_array.transpose(1, 2, 0)
                    # Normalize to 0-255
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame_array.astype(np.uint8)))
                else:
                    # Assume numpy array
                    frame_array = np.array(frame)
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    pil_frames.append(Image.fromarray(frame_array.astype(np.uint8)))

            videos.append(pil_frames)

        return videos

    def _compute_metrics(self, images: List[Image.Image], prompt: str) -> Dict:
        """
        Compute quality metrics for generated images

        Args:
            images: Generated images
            prompt: Generation prompt

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'num_samples': len(images),
            'prompt': prompt,
        }

        try:
            # Compute basic image statistics
            avg_brightness = []
            avg_contrast = []
            avg_saturation = []

            for img in images:
                # Convert to array
                img_array = np.array(img)

                # Brightness (mean of pixel values)
                brightness = img_array.mean()
                avg_brightness.append(brightness)

                # Contrast (std of pixel values)
                contrast = img_array.std()
                avg_contrast.append(contrast)

                # Saturation (for RGB images)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    # Convert to HSV
                    from PIL import ImageColor
                    hsv_img = img.convert('HSV')
                    hsv_array = np.array(hsv_img)
                    saturation = hsv_array[:, :, 1].mean()
                    avg_saturation.append(saturation)

            metrics['avg_brightness'] = round(float(np.mean(avg_brightness)), 2)
            metrics['avg_contrast'] = round(float(np.mean(avg_contrast)), 2)
            if avg_saturation:
                metrics['avg_saturation'] = round(float(np.mean(avg_saturation)), 2)

            # Compute diversity (variance in image features)
            if len(images) > 1:
                brightness_var = np.var(avg_brightness)
                metrics['diversity_score'] = round(float(brightness_var), 2)

        except Exception as e:
            metrics['metric_error'] = str(e)

        # Try to compute CLIP score if available
        try:
            metrics.update(self._compute_clip_score(images, prompt))
        except:
            pass

        return metrics

    def _compute_clip_score(self, images: List[Image.Image], prompt: str) -> Dict:
        """
        Compute CLIP score for text-image alignment

        Args:
            images: Generated images
            prompt: Text prompt

        Returns:
            Dictionary with CLIP scores
        """
        try:
            from transformers import CLIPProcessor, CLIPModel

            # Load CLIP
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            model = model.to("cuda" if torch.cuda.is_available() else "cpu")

            # Compute scores
            scores = []
            for img in images:
                inputs = processor(
                    text=[prompt],
                    images=img,
                    return_tensors="pt",
                    padding=True
                )

                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    score = logits_per_image.item()
                    scores.append(score)

            return {
                'clip_score_mean': round(float(np.mean(scores)), 4),
                'clip_score_std': round(float(np.std(scores)), 4),
            }

        except Exception as e:
            return {'clip_score_error': str(e)}

    def unload(self):
        """Unload pipeline to free VRAM"""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("✓ Pipeline unloaded")

    def compare_models(
        self,
        lora_paths: List[str],
        base_model: str,
        prompt: str,
        **kwargs
    ) -> Dict:
        """
        Compare multiple LoRA models

        Args:
            lora_paths: List of LoRA model paths
            base_model: Base model to use
            prompt: Test prompt
            **kwargs: Additional generation parameters

        Returns:
            Comparison results
        """
        results = {}

        for lora_path in lora_paths:
            model_name = Path(lora_path).stem

            print(f"Validating: {model_name}")

            images, metrics = self.validate(
                lora_path=lora_path,
                base_model=base_model,
                prompt=prompt,
                **kwargs
            )

            results[model_name] = {
                'images': images,
                'metrics': metrics
            }

        return results


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model Validator CLI")
    parser.add_argument("--lora", required=True, help="Path to LoRA weights")
    parser.add_argument("--base", required=True, help="Base model name")
    parser.add_argument("--prompt", required=True, help="Generation prompt")
    parser.add_argument("--negative", default="", help="Negative prompt")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples")
    parser.add_argument("--steps", type=int, default=30, help="Inference steps")
    parser.add_argument("--cfg", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")

    args = parser.parse_args()

    validator = ModelValidator()

    print("Generating validation samples...")
    images, metrics = validator.validate(
        lora_path=args.lora,
        base_model=args.base,
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_samples=args.samples,
        steps=args.steps,
        guidance_scale=args.cfg,
        seed=args.seed
    )

    print("\n=== Validation Results ===")
    print(f"Generated {len(images)} samples")
    print("\nMetrics:")
    import json
    print(json.dumps(metrics, indent=2))
    print(f"\nImages saved to: {Path(images[0]).parent if images else 'N/A'}")
