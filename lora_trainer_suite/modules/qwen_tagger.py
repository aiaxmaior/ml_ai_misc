"""
Qwen VL tagger with vLLM and direct mode support
"""

import os
import torch
from PIL import Image
from typing import Optional, Union, List
import gc
import base64
from io import BytesIO
import requests


class QwenVLTagger:
    """Qwen VL model wrapper - supports both direct transformers and vLLM backends"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        abliterated_repo: Optional[str] = None,
        device: Optional[str] = None,
        backend: str = "auto",  # auto, vllm, or direct
        use_4bit: bool = True,
        use_flash_attention: bool = True,
        vllm_server_url: Optional[str] = None,
        vllm_port: int = 8000
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = abliterated_repo or model_name
        self.use_4bit = use_4bit
        self.use_flash_attention = use_flash_attention
        self.vllm_server_url = vllm_server_url or "http://localhost"
        self.vllm_port = vllm_port

        self.backend = backend
        self.active_backend = None
        self.model = None
        self.processor = None

        self._initialize_backend()

    def _initialize_backend(self):
        if self.backend == "vllm":
            if self._check_vllm_available():
                self.active_backend = "vllm"
                print(f"Using vLLM backend at {self.vllm_server_url}:{self.vllm_port}")
            else:
                raise RuntimeError(
                    f"vLLM server not available at {self.vllm_server_url}:{self.vllm_port}"
                )

        elif self.backend == "direct":
            self.active_backend = "direct"
            self._load_model_direct()

        elif self.backend == "auto":
            if self._check_vllm_available():
                self.active_backend = "vllm"
                print("Using vLLM backend (auto-detected)")
            else:
                print("vLLM not available, using direct mode")
                self.active_backend = "direct"
                self._load_model_direct()

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _check_vllm_available(self) -> bool:
        """Check if vLLM server is available"""
        try:
            url = f"{self.vllm_server_url}:{self.vllm_port}/health"
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except:
            return False

    def _load_model_direct(self):
        """Load model using transformers"""
        try:
            from transformers import (
                Qwen2VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig
            )

            print(f"Loading {self.model_name}...")

            bnb_config = None
            if self.use_4bit and self.device == 'cuda':
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                print("Using 4-bit quantization")

            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir="./models/qwenvl"
            )

            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto",
                "cache_dir": "./models/qwenvl",
                "torch_dtype": torch.bfloat16 if self.device == 'cuda' else torch.float32,
            }

            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config

            if self.use_flash_attention and self.device == 'cuda':
                try:
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    print("Using Flash Attention 2")
                except:
                    pass  # fall back to default attention

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            if self.device == 'cuda':
                vram_used = torch.cuda.memory_allocated() / 1024**3
                print(f"Model loaded - VRAM: {vram_used:.2f} GB")

        except ImportError:
            raise ImportError(
                "Missing dependencies - install transformers, accelerate, bitsandbytes, flash-attn"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def tag_image(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        do_sample: bool = True
    ) -> str:
        """Generate tags for image using VLM"""
        if self.active_backend == "vllm":
            return self._tag_image_vllm(
                image, prompt, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, presence_penalty, frequency_penalty, do_sample
            )
        else:
            return self._tag_image_direct(
                image, prompt, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty, presence_penalty, frequency_penalty, do_sample
            )

    def _tag_image_direct(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        do_sample: bool
    ) -> str:
        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                image = Image.open(image).convert('RGB')

            if image.mode != 'RGB':
                image = image.convert('RGB')

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

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generated_text = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]

            # extract generated portion only
            if prompt in generated_text:
                generated_text = generated_text.split(prompt)[-1].strip()

            return generated_text

        except Exception as e:
            print(f"Error tagging image (direct): {str(e)}")
            return ""

    def _tag_image_vllm(
        self,
        image: Union[str, Image.Image],
        prompt: Optional[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        presence_penalty: float,
        frequency_penalty: float,
        do_sample: bool
    ) -> str:
        try:
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image not found: {image}")
                image = Image.open(image).convert('RGB')

            if image.mode != 'RGB':
                image = image.convert('RGB')

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

            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            url = f"{self.vllm_server_url}:{self.vllm_port}/v1/chat/completions"

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt
                            }
                        ]
                    }
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty,
            }

            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"Error tagging image (vLLM): {str(e)}")
            return ""

    def batch_tag(self, image_paths: List[str], prompt: Optional[str] = None, progress_callback=None):
        """Tag multiple images"""
        results = {}
        total = len(image_paths)

        for idx, img_path in enumerate(image_paths):
            try:
                tags = self.tag_image(img_path, prompt=prompt)
                results[img_path] = tags

                if progress_callback:
                    progress_callback(idx + 1, total, tags)

                if (idx + 1) % 10 == 0 and self.device == 'cuda':
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                results[img_path] = ""

        return results

    def analyze_style(self, image: Union[str, Image.Image]) -> str:
        """Analyze image style"""
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
        """Generate booru-style tags"""
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
        """Free VRAM"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    def __del__(self):
        self.unload()


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Qwen VL Tagger CLI (Qwen2-VL/Qwen3-VL)")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--prompt", help="Custom prompt", default=None)
    parser.add_argument("--style", action="store_true", help="Analyze style")
    parser.add_argument("--booru", action="store_true", help="Generate booru tags")
    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "direct"],
                        help="Backend: auto (try vLLM, fallback to direct), vllm, or direct")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Model name (Qwen2-VL or Qwen3-VL)")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization (direct mode)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--vllm-url", default="http://localhost", help="vLLM server URL")
    parser.add_argument("--vllm-port", type=int, default=8000, help="vLLM server port")

    args = parser.parse_args()

    tagger = QwenVLTagger(
        model_name=args.model,
        device=args.device,
        backend=args.backend,
        use_4bit=not args.no_4bit,
        vllm_server_url=args.vllm_url,
        vllm_port=args.vllm_port
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
