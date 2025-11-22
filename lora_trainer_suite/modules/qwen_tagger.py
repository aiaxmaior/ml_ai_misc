"""
Qwen2-VL Tagger Module
Uses Qwen2-VL-8B (Abliterated) with BitsAndBytes quantization for uncensored image tagging
Optimized for RTX 3090
"""

import os
import torch
from PIL import Image
from typing import Optional, Union, List
import gc
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)


class QwenVLTagger:
    """
    Qwen2-VL-8B Vision-Language Model for image tagging
    Uses BitsAndBytes 4-bit quantization for efficient VRAM usage on RTX 3090
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        abliterated_repo: Optional[str] = None,
        device: Optional[str] = None,
        use_4bit: bool = True,
        use_flash_attention: bool = True
    ):
        """
        Initialize Qwen2-VL tagger

        Args:
            model_name: HuggingFace model name
            abliterated_repo: Optional abliterated model repo (e.g., community versions)
            device: Device to use ('cuda' or 'cpu')
            use_4bit: Use 4-bit BitsAndBytes quantization
            use_flash_attention: Use Flash Attention 2 for faster inference
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = abliterated_repo or model_name
        self.use_4bit = use_4bit
        self.use_flash_attention = use_flash_attention

        self.model = None
        self.processor = None

        self._load_model()

    def _load_model(self):
        """Load Qwen2-VL model with optimizations"""
        try:
            print(f"Loading Qwen2-VL from {self.model_name}...")

            # Configure BitsAndBytes for 4-bit quantization
            if self.use_4bit and self.device == 'cuda':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                print("✓ Using 4-bit BitsAndBytes quantization")
            else:
                bnb_config = None

            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir="./models/qwen2vl"
            )

            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "cache_dir": "./models/qwen2vl",
                "torch_dtype": torch.bfloat16 if self.device == 'cuda' else torch.float32,
            }

            if bnb_config is not None:
                model_kwargs["quantization_config"] = bnb_config

            if self.use_flash_attention and self.device == 'cuda':
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("✓ Using Flash Attention 2")
                except:
                    print("⚠ Flash Attention 2 not available, using default")

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            print(f"✓ Qwen2-VL loaded on {self.device}")

            # Print VRAM usage
            if self.device == 'cuda':
                vram_used = torch.cuda.memory_allocated() / 1024**3
                print(f"  VRAM used: {vram_used:.2f} GB")

        except ImportError as e:
            raise ImportError(
                "Missing dependencies. Install with:\n"
                "pip install transformers accelerate bitsandbytes flash-attn --no-build-isolation"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Qwen2-VL: {str(e)}")

    def tag_image(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate tags/description for an image

        Args:
            image: Path to image or PIL Image
            prompt: Custom prompt (if None, uses default tagging prompt)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated tags/description
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                image = Image.open(image).convert('RGB')

            # Ensure RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Default tagging prompt
            if prompt is None:
                prompt = (
                    "Describe this image in detail. Include:\n"
                    "1. Main subject and their attributes\n"
                    "2. Clothing, pose, and expression\n"
                    "3. Background and setting\n"
                    "4. Artistic style and composition\n"
                    "5. Lighting and mood\n"
                    "6. Any notable details\n"
                    "Provide tags as comma-separated keywords."
                )

            # Prepare messages for chat template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Apply chat template
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Process inputs
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            # Decode output
            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # Extract only the generated part (after the prompt)
            # The output usually includes the input prompt
            if prompt in generated_text:
                generated_text = generated_text.split(prompt)[-1].strip()

            return generated_text

        except Exception as e:
            print(f"Error tagging image: {str(e)}")
            return ""

    def batch_tag(
        self,
        image_paths: List[str],
        prompt: Optional[str] = None,
        progress_callback=None
    ) -> dict:
        """
        Batch tag multiple images

        Args:
            image_paths: List of image paths
            prompt: Custom prompt for all images
            progress_callback: Optional callback(current, total, tags)

        Returns:
            Dictionary mapping image paths to tags
        """
        results = {}
        total = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            try:
                tags = self.tag_image(img_path, prompt=prompt)
                results[img_path] = tags

                if progress_callback:
                    progress_callback(idx + 1, total, tags)

                # Clear cache periodically
                if (idx + 1) % 10 == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results[img_path] = ""

        return results

    def analyze_style(self, image: Union[str, Image.Image]) -> str:
        """
        Analyze artistic style of image

        Args:
            image: Image path or PIL Image

        Returns:
            Style description
        """
        prompt = (
            "Analyze the artistic style of this image. Describe:\n"
            "- Art medium (photo, digital art, painting, etc.)\n"
            "- Style influences (realistic, anime, cartoon, abstract, etc.)\n"
            "- Color palette and mood\n"
            "- Technical qualities (lighting, composition, detail level)\n"
            "Provide concise style tags."
        )

        return self.tag_image(image, prompt=prompt)

    def generate_booru_tags(self, image: Union[str, Image.Image]) -> str:
        """
        Generate Danbooru/e621 style tags

        Args:
            image: Image path or PIL Image

        Returns:
            Booru-style tags
        """
        prompt = (
            "Generate detailed tags for this image in Danbooru/e621 format. Include:\n"
            "- Character features (hair color, eye color, species, etc.)\n"
            "- Clothing items\n"
            "- Pose and expression\n"
            "- Background elements\n"
            "- Artistic style\n"
            "- Quality indicators\n"
            "Format as comma-separated tags with underscores (e.g., 'red_hair, blue_eyes')."
        )

        return self.tag_image(image, prompt=prompt, temperature=0.5)

    def unload(self):
        """Unload model to free VRAM"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        print("✓ Qwen2-VL unloaded")

    def __del__(self):
        """Cleanup on deletion"""
        self.unload()


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen2-VL Tagger CLI")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--prompt", help="Custom prompt", default=None)
    parser.add_argument("--style", action="store_true", help="Analyze style")
    parser.add_argument("--booru", action="store_true", help="Generate booru tags")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    tagger = QwenVLTagger(
        device=args.device,
        use_4bit=not args.no_4bit
    )

    print(f"\nProcessing: {args.image}\n")

    if args.style:
        result = tagger.analyze_style(args.image)
        print("Style Analysis:")
    elif args.booru:
        result = tagger.generate_booru_tags(args.image)
        print("Booru Tags:")
    else:
        result = tagger.tag_image(args.image, prompt=args.prompt)
        print("Tags:")

    print(f"{result}\n")
