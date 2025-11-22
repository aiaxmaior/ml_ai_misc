"""
CLIP Interrogator Module
Wrapper for CLIP Interrogator for automated image captioning
"""

import os
import torch
from PIL import Image
from typing import Optional, Union
import gc


class CLIPInterrogatorModule:
    """Wrapper for CLIP Interrogator with optimizations for RTX 3090"""

    def __init__(self, device: Optional[str] = None):
        """
        Initialize CLIP Interrogator

        Args:
            device: Device to use ('cuda' or 'cpu'). Auto-detects if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ci = None
        self._load_model()

    def _load_model(self):
        """Load CLIP Interrogator model"""
        try:
            from clip_interrogator import Config, Interrogator

            # Configure for RTX 3090
            config = Config(
                device=self.device,
                clip_model_name="ViT-L-14/openai",
                caption_model_name="blip-large",
                cache_path="./models/clip_interrogator",
                chunk_size=2048,
                flavor_intermediate_count=2048,
                quiet=False
            )

            self.ci = Interrogator(config)
            print(f"✓ CLIP Interrogator loaded on {self.device}")

        except ImportError:
            raise ImportError(
                "CLIP Interrogator not installed. "
                "Install with: pip install clip-interrogator"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP Interrogator: {str(e)}")

    def interrogate(
        self,
        image: Union[str, Image.Image],
        mode: str = "best",
        max_flavors: int = 32
    ) -> str:
        """
        Generate caption for image using CLIP Interrogator

        Args:
            image: Path to image or PIL Image object
            mode: Interrogation mode ('best', 'fast', 'classic', 'negative')
            max_flavors: Maximum number of flavor terms to include

        Returns:
            Generated caption as string
        """
        try:
            # Load image if path provided
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                image = Image.open(image).convert('RGB')

            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Generate caption based on mode
            if mode == "best":
                caption = self.ci.interrogate(image, max_flavors=max_flavors)
            elif mode == "fast":
                caption = self.ci.interrogate_fast(image, max_flavors=max_flavors)
            elif mode == "classic":
                caption = self.ci.interrogate_classic(image, max_flavors=max_flavors)
            elif mode == "negative":
                caption = self.ci.interrogate(image, max_flavors=max_flavors)
                # Generate negative prompt (invert concepts)
                caption = self._generate_negative(caption)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            return caption.strip()

        except Exception as e:
            print(f"Error interrogating image: {str(e)}")
            return ""

    def _generate_negative(self, positive_caption: str) -> str:
        """
        Generate negative prompt from positive caption

        Args:
            positive_caption: Positive caption to invert

        Returns:
            Negative prompt
        """
        # Common negative terms
        negative_terms = [
            "blurry", "low quality", "distorted", "ugly",
            "bad anatomy", "poorly drawn", "deformed",
            "disfigured", "mutation", "mutated", "extra limbs"
        ]

        # You could use LLM to generate better negatives
        # For now, return common negative terms
        return ", ".join(negative_terms)

    def batch_interrogate(
        self,
        image_paths: list,
        mode: str = "best",
        progress_callback=None
    ) -> dict:
        """
        Batch interrogate multiple images

        Args:
            image_paths: List of image paths
            mode: Interrogation mode
            progress_callback: Optional callback(current, total, caption)

        Returns:
            Dictionary mapping image paths to captions
        """
        results = {}
        total = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            try:
                caption = self.interrogate(img_path, mode=mode)
                results[img_path] = caption

                if progress_callback:
                    progress_callback(idx + 1, total, caption)

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results[img_path] = ""

        return results

    def unload(self):
        """Unload model to free VRAM"""
        if self.ci is not None:
            del self.ci
            self.ci = None

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        print("✓ CLIP Interrogator unloaded")

    def __del__(self):
        """Cleanup on deletion"""
        self.unload()


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CLIP Interrogator CLI")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--mode", default="best", choices=["best", "fast", "classic", "negative"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])

    args = parser.parse_args()

    ci = CLIPInterrogatorModule(device=args.device)
    caption = ci.interrogate(args.image, mode=args.mode)

    print(f"\nImage: {args.image}")
    print(f"Caption: {caption}\n")
