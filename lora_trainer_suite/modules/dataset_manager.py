"""
Dataset Manager Module
Handles dataset loading, preprocessing, augmentation, and organization
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import random
from tqdm import tqdm


class DatasetManager:
    """Manage datasets for LoRA training"""

    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(self, config=None):
        """
        Initialize Dataset Manager

        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.current_dataset = None
        self.images = []
        self.captions = {}

    def load_dataset(self, dataset_path: str) -> Dict:
        """
        Load and analyze dataset

        Args:
            dataset_path: Path to dataset directory

        Returns:
            Dictionary with dataset information
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        if not dataset_path.is_dir():
            raise ValueError(f"Dataset path must be a directory: {dataset_path}")

        # Find all images
        self.images = []
        for ext in self.SUPPORTED_IMAGE_FORMATS:
            self.images.extend(dataset_path.glob(f"*{ext}"))
            self.images.extend(dataset_path.glob(f"**/*{ext}"))

        # Remove duplicates and sort
        self.images = sorted(list(set(self.images)))

        # Load existing captions
        self.captions = {}
        for img_path in self.images:
            caption_path = img_path.with_suffix('.txt')
            if caption_path.exists():
                with open(caption_path, 'r', encoding='utf-8') as f:
                    self.captions[str(img_path)] = f.read().strip()

        # Analyze dataset
        info = {
            'path': str(dataset_path),
            'total_images': len(self.images),
            'captioned_images': len(self.captions),
            'uncaptioned_images': len(self.images) - len(self.captions),
            'formats': self._count_formats(),
            'resolutions': self._analyze_resolutions(),
            'total_size_mb': self._calculate_total_size()
        }

        self.current_dataset = dataset_path
        return info

    def _count_formats(self) -> Dict[str, int]:
        """Count images by format"""
        formats = {}
        for img_path in self.images:
            ext = img_path.suffix.lower()
            formats[ext] = formats.get(ext, 0) + 1
        return formats

    def _analyze_resolutions(self) -> Dict:
        """Analyze image resolutions"""
        resolutions = []
        aspect_ratios = []

        # Sample up to 100 images for performance
        sample_size = min(100, len(self.images))
        sample_images = random.sample(self.images, sample_size)

        for img_path in sample_images:
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    resolutions.append((w, h))
                    aspect_ratios.append(w / h if h > 0 else 1.0)
            except:
                continue

        if resolutions:
            avg_w = sum(r[0] for r in resolutions) / len(resolutions)
            avg_h = sum(r[1] for r in resolutions) / len(resolutions)
            avg_aspect = sum(aspect_ratios) / len(aspect_ratios)

            return {
                'average': f"{int(avg_w)}x{int(avg_h)}",
                'average_aspect_ratio': f"{avg_aspect:.2f}",
                'min': f"{min(r[0] for r in resolutions)}x{min(r[1] for r in resolutions)}",
                'max': f"{max(r[0] for r in resolutions)}x{max(r[1] for r in resolutions)}"
            }

        return {}

    def _calculate_total_size(self) -> float:
        """Calculate total dataset size in MB"""
        total_bytes = sum(img.stat().st_size for img in self.images)
        return round(total_bytes / (1024 * 1024), 2)

    def get_preview_images(self, limit: int = 12) -> List[str]:
        """
        Get preview images for display

        Args:
            limit: Maximum number of images to return

        Returns:
            List of image paths
        """
        sample = self.images[:limit] if len(self.images) <= limit else random.sample(self.images, limit)
        return [str(img) for img in sample]

    def get_all_images(self, dataset_path: Optional[str] = None) -> List[str]:
        """
        Get all image paths

        Args:
            dataset_path: Optional dataset path (uses current if None)

        Returns:
            List of image paths
        """
        if dataset_path and dataset_path != str(self.current_dataset):
            self.load_dataset(dataset_path)

        return [str(img) for img in self.images]

    def resize_images(
        self,
        dataset_path: str,
        target_width: int,
        target_height: int,
        maintain_aspect: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Batch resize images

        Args:
            dataset_path: Dataset directory
            target_width: Target width
            target_height: Target height
            maintain_aspect: Whether to maintain aspect ratio
            output_dir: Output directory (creates resized/ subdir if None)
        """
        if not self.images or str(self.current_dataset) != dataset_path:
            self.load_dataset(dataset_path)

        # Setup output directory
        if output_dir is None:
            output_dir = Path(dataset_path) / "resized"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True, parents=True)

        # Resize images
        for img_path in tqdm(self.images, desc="Resizing images"):
            try:
                with Image.open(img_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    # Resize
                    if maintain_aspect:
                        img.thumbnail((target_width, target_height), Image.Resampling.LANCZOS)
                    else:
                        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)

                    # Save
                    output_path = output_dir / img_path.name
                    img.save(output_path, quality=95)

                    # Copy caption if exists
                    caption_path = img_path.with_suffix('.txt')
                    if caption_path.exists():
                        shutil.copy2(caption_path, output_dir / caption_path.name)

            except Exception as e:
                print(f"Error resizing {img_path}: {e}")

        print(f"✓ Resized {len(self.images)} images to {output_dir}")

    def augment(
        self,
        dataset_path: str,
        horizontal_flip: bool = True,
        random_rotation: bool = False,
        output_dir: Optional[str] = None
    ):
        """
        Apply data augmentation

        Args:
            dataset_path: Dataset directory
            horizontal_flip: Apply horizontal flip
            random_rotation: Apply random rotation (-10 to 10 degrees)
            output_dir: Output directory (creates augmented/ subdir if None)
        """
        if not self.images or str(self.current_dataset) != dataset_path:
            self.load_dataset(dataset_path)

        # Setup output directory
        if output_dir is None:
            output_dir = Path(dataset_path) / "augmented"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True, parents=True)

        # Augment images
        for img_path in tqdm(self.images, desc="Augmenting images"):
            try:
                with Image.open(img_path) as img:
                    # Original
                    base_name = img_path.stem
                    ext = img_path.suffix

                    # Save original
                    img.save(output_dir / f"{base_name}_orig{ext}")

                    # Copy original caption
                    caption_path = img_path.with_suffix('.txt')
                    if caption_path.exists():
                        with open(caption_path, 'r', encoding='utf-8') as f:
                            caption = f.read()
                        with open(output_dir / f"{base_name}_orig.txt", 'w', encoding='utf-8') as f:
                            f.write(caption)

                    # Horizontal flip
                    if horizontal_flip:
                        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped.save(output_dir / f"{base_name}_flip{ext}")

                        if caption_path.exists():
                            with open(output_dir / f"{base_name}_flip.txt", 'w', encoding='utf-8') as f:
                                f.write(caption)

                    # Random rotation
                    if random_rotation:
                        angle = random.uniform(-10, 10)
                        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=False)
                        rotated.save(output_dir / f"{base_name}_rot{ext}")

                        if caption_path.exists():
                            with open(output_dir / f"{base_name}_rot.txt", 'w', encoding='utf-8') as f:
                                f.write(caption)

            except Exception as e:
                print(f"Error augmenting {img_path}: {e}")

        print(f"✓ Augmented dataset saved to {output_dir}")

    def save_tags(
        self,
        image_path: str,
        tags: str,
        merge: bool = True
    ):
        """
        Save tags for an image

        Args:
            image_path: Path to image
            tags: Tags to save
            merge: Whether to merge with existing tags
        """
        image_path = Path(image_path)
        caption_path = image_path.with_suffix('.txt')

        existing_tags = ""
        if merge and caption_path.exists():
            with open(caption_path, 'r', encoding='utf-8') as f:
                existing_tags = f.read().strip()

        # Merge tags if requested
        if merge and existing_tags:
            # Simple merge - combine and deduplicate
            all_tags = set(existing_tags.split(', ')) | set(tags.split(', '))
            final_tags = ', '.join(sorted(all_tags))
        else:
            final_tags = tags

        # Save
        with open(caption_path, 'w', encoding='utf-8') as f:
            f.write(final_tags)

    def validate_dataset(self, dataset_path: str) -> Dict:
        """
        Validate dataset for training

        Args:
            dataset_path: Dataset directory

        Returns:
            Validation report
        """
        if not self.images or str(self.current_dataset) != dataset_path:
            self.load_dataset(dataset_path)

        issues = []
        warnings = []

        # Check for images without captions
        uncaptioned = [str(img) for img in self.images if str(img) not in self.captions]
        if uncaptioned:
            warnings.append(f"{len(uncaptioned)} images without captions")

        # Check for very small images
        small_images = []
        for img_path in self.images[:50]:  # Sample check
            try:
                with Image.open(img_path) as img:
                    if img.size[0] < 256 or img.size[1] < 256:
                        small_images.append(str(img_path))
            except:
                issues.append(f"Cannot open image: {img_path}")

        if small_images:
            warnings.append(f"{len(small_images)} images smaller than 256x256")

        # Check minimum dataset size
        if len(self.images) < 10:
            issues.append("Dataset has fewer than 10 images (recommended: 20+)")

        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'total_images': len(self.images),
            'captioned': len(self.captions)
        }

    def create_training_metadata(
        self,
        dataset_path: str,
        output_file: str = "metadata.jsonl"
    ):
        """
        Create metadata file for training (Kohya_ss format)

        Args:
            dataset_path: Dataset directory
            output_file: Output metadata filename
        """
        if not self.images or str(self.current_dataset) != dataset_path:
            self.load_dataset(dataset_path)

        output_path = Path(dataset_path) / output_file
        metadata = []

        for img_path in self.images:
            caption = self.captions.get(str(img_path), "")

            entry = {
                "file_name": img_path.name,
                "text": caption
            }

            metadata.append(entry)

        # Save as JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in metadata:
                f.write(json.dumps(entry) + '\n')

        print(f"✓ Created metadata file: {output_path}")
        return str(output_path)


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Manager CLI")
    parser.add_argument("dataset", help="Path to dataset directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset")
    parser.add_argument("--validate", action="store_true", help="Validate dataset")
    parser.add_argument("--resize", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                        help="Resize images to WIDTHxHEIGHT")
    parser.add_argument("--metadata", action="store_true", help="Create metadata file")

    args = parser.parse_args()

    dm = DatasetManager()

    if args.analyze:
        info = dm.load_dataset(args.dataset)
        print("\n=== Dataset Analysis ===")
        print(json.dumps(info, indent=2))

    if args.validate:
        report = dm.validate_dataset(args.dataset)
        print("\n=== Validation Report ===")
        print(json.dumps(report, indent=2))

    if args.resize:
        dm.resize_images(args.dataset, args.resize[0], args.resize[1])

    if args.metadata:
        dm.create_training_metadata(args.dataset)
