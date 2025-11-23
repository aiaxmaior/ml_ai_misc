#!/usr/bin/env python3
"""
Standalone Gradio Interface for Video Inference
Extracts frames from video and sends to KoboldCPP/vLLM for analysis
"""

import cv2
import base64
import requests
import gradio as gr
from pathlib import Path
import tempfile
from PIL import Image
import io
import json
from pathlib import Path

# Config file path
CONFIG_FILE = Path(__file__).parent / "video_inf_config.json"

DEFAULT_CONFIG = {
    "backend": "vLLM",
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model_name": "Qwen/Qwen2.5-VL-7B-Instruct",
    "processing_mode": "Frame Extraction (KoboldCPP/GGUF)",
    "sampling_mode": "fps",
    "interval": 2.0,
    "target_fps": 1.0,
    "max_frames_limit": 0,
    "image_width": 512,
    "image_height": 512,
    "prompt": "Analyze this video sequence and describe exactly what is happening.",
    "max_tokens": 2000,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.15,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0
}

def load_config():
    """Load config from JSON file"""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save config to JSON file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return f"✓ Config saved to {CONFIG_FILE}"
    except Exception as e:
        return f"✗ Error saving config: {str(e)}"

def encode_file_base64(file_path):
    """Encode entire file to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_frame(frame):
    """Encode OpenCV frame to base64 JPEG"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def extract_frames(video_path, sampling_mode="interval", interval=2.0, target_fps=1.0, 
                   max_frames=None, image_size=(512, 512)):
    """
    Extract frames from video
    sampling_mode: 'interval' (every N seconds), 'fps' (resample to target FPS), 'all' (every frame)
    """
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_images = []  # For preview (limited to 50 for gallery)
    
    if sampling_mode == "all":
        # Extract every frame
        step = 1
    elif sampling_mode == "fps":
        # Resample to target FPS
        step = max(1, int(fps / target_fps))
    else:  # interval
        # Every N seconds
        step = int(fps * interval)
    
    current_frame = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        # Capture frame based on sampling mode
        if current_frame % step == 0:
            # Resize to save bandwidth
            resized = cv2.resize(frame, image_size)
            frames.append(resized)
            
            # Convert to PIL for preview (limit preview to 50 frames)
            if len(frame_images) < 50:
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                frame_images.append(pil_image)
            
        current_frame += 1
        
        # Stop if max_frames reached (if specified)
        if max_frames and len(frames) >= max_frames:
            break
            
    video.release()
    return frames, frame_images

def get_video_info(video_path):
    """Get video metadata"""
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    video.release()
    
    return {
        "FPS": round(fps, 2),
        "Total Frames": total_frames,
        "Resolution": f"{width}x{height}",
        "Duration (sec)": round(duration, 2)
    }

def process_video(video_path, api_url, model_name, backend, processing_mode, prompt, sampling_mode, interval, target_fps,
                  max_frames_limit, image_width, image_height, max_tokens, temperature, top_p, top_k,
                  repetition_penalty, presence_penalty, frequency_penalty):
    """Process video and send to API"""
    if not video_path:
        return "Please upload a video.", [], {}
    
    if not model_name:
        return "Please enter a model name.", [], {}
    
    try:
        # Get video info
        info = get_video_info(video_path)
        
        messages = [{"role": "user", "content": []}]
        frame_images = []  # For preview
        
        # --- MODE 1: Native Video (send whole video file) ---
        if processing_mode == "Native Video (vLLM)":
            print("Processing in Native Video mode (sending full video file)...")
            video_b64 = encode_file_base64(video_path)
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                }
            ]
            # Extract a few frames just for UI preview
            video = cv2.VideoCapture(video_path)
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total_frames // 24)  # 24 preview frames
            count = 0
            while video.isOpened() and len(frame_images) < 24:
                ret, frame = video.read()
                if not ret:
                    break
                if count % step == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_frame)
                    frame_images.append(pil_image)
                count += 1
            video.release()
            
        # --- MODE 2: Frame Extraction (extract and send frames) ---
        else:
            print(f"Processing in Frame Extraction mode ({sampling_mode})...")
            # Determine max_frames (None = unlimited)
            max_frames = int(max_frames_limit) if max_frames_limit > 0 else None
            
            # Extract frames
            frames, frame_images = extract_frames(
                video_path,
                sampling_mode=sampling_mode,
                interval=float(interval),
                target_fps=float(target_fps),
                max_frames=max_frames,
                image_size=(int(image_width), int(image_height))
            )
            
            if len(frames) == 0:
                return "No frames extracted from video.", [], info
            
            # Prepare messages with frames
            messages[0]["content"].append({"type": "text", "text": prompt})
            for frame in frames:
                base64_image = encode_frame(frame)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
        
        # Build payload with all parameters
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repetition_penalty": float(repetition_penalty),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty)
        }
        
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        ai_response = result['choices'][0]['message']['content']
        return ai_response, frame_images, info
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"API Error: {e.response.status_code}\n"
        try:
            error_detail = e.response.json()
            error_msg += f"Details: {error_detail}"
        except:
            error_msg += f"Response: {e.response.text[:500]}"
        return error_msg, [], {}
    except Exception as e:
        return f"Error: {str(e)}", [], {}

def build_interface():
    # Load saved config
    config = load_config()
    
    with gr.Blocks(title="Video Inference Tool") as app:
        gr.Markdown("# 🎥 Video Inference Tool")
        gr.Markdown("Extract frames from video and send to vision language model API")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                video_info = gr.JSON(label="Video Metadata", visible=False)
                
                gr.Markdown("### Configuration")
                with gr.Row():
                    save_config_btn = gr.Button("💾 Save Config", scale=1)
                    load_config_btn = gr.Button("📂 Load Config", scale=1)
                config_status = gr.Markdown("")
                
                gr.Markdown("### API Configuration")
                
                backend = gr.Radio(
                    label="Backend",
                    choices=["vLLM", "KoboldCPP"],
                    value=config["backend"],
                    info="vLLM: supports more parameters | KoboldCPP: basic compatibility"
                )
                
                with gr.Row():
                    api_url = gr.Textbox(
                        label="API URL",
                        value=config["api_url"],
                        placeholder="http://localhost:8000/v1/chat/completions",
                        scale=3
                    )
                    refresh_models_btn = gr.Button("🔄", scale=0, size="sm")
                
                model_name = gr.Dropdown(
                    label="Model Name",
                    choices=["Qwen/Qwen2.5-VL-7B-Instruct", "Qwen/Qwen3-VL-Thinking"],
                    value=config["model_name"],
                    allow_custom_value=True,
                    info="Must match the model loaded in vLLM/KoboldCPP"
                )
                
                processing_mode = gr.Radio(
                    label="Processing Mode",
                    choices=["Native Video (vLLM)", "Frame Extraction (KoboldCPP/GGUF)"],
                    value=config.get("processing_mode", "Frame Extraction (KoboldCPP/GGUF)"),
                    info="Native: Send full video file (vLLM only) | Extraction: Extract frames (works with all backends)"
                )
                
                gr.Markdown("### Frame Extraction")
                
                sampling_mode = gr.Radio(
                    label="Sampling Mode",
                    choices=["interval", "fps", "all"],
                    value=config["sampling_mode"],
                    info="interval: every N seconds | fps: resample to target FPS | all: every frame"
                )
                
                with gr.Row():
                    interval = gr.Number(label="Interval (seconds)", value=config["interval"], precision=1, visible=False)
                    target_fps = gr.Number(label="Target FPS", value=config["target_fps"], precision=1, visible=True)
                
                max_frames_limit = gr.Number(
                    label="Max Frames (0 = unlimited)",
                    value=config["max_frames_limit"],
                    precision=0,
                    info="Processing entire videos may take time but gives complete context"
                )
                
                with gr.Row():
                    image_width = gr.Number(label="Frame Width", value=config["image_width"], precision=0)
                    image_height = gr.Number(label="Frame Height", value=config["image_height"], precision=0)
                
                gr.Markdown("### Inference Settings")
                prompt = gr.Textbox(
                    label="Prompt",
                    value=config["prompt"],
                    lines=3
                )
                
                with gr.Accordion("Advanced Parameters", open=False):
                    max_tokens = gr.Number(label="Max Tokens", value=config["max_tokens"], precision=0)
                    
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature", 
                            minimum=0.0, 
                            maximum=2.0, 
                            value=config["temperature"], 
                            step=0.05,
                            info="Higher = more creative"
                        )
                        top_p = gr.Slider(
                            label="Top P",
                            minimum=0.0,
                            maximum=1.0,
                            value=config["top_p"],
                            step=0.05,
                            info="Nucleus sampling"
                        )
                    
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top K",
                            minimum=1,
                            maximum=100,
                            value=config["top_k"],
                            step=1,
                            info="Consider top K tokens"
                        )
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            minimum=1.0,
                            maximum=2.0,
                            value=config["repetition_penalty"],
                            step=0.05,
                            info="Penalize repeated tokens"
                        )
                    
                    with gr.Row():
                        presence_penalty = gr.Slider(
                            label="Presence Penalty",
                            minimum=0.0,
                            maximum=2.0,
                            value=config["presence_penalty"],
                            step=0.1,
                            info="Penalize tokens that appeared"
                        )
                        frequency_penalty = gr.Slider(
                            label="Frequency Penalty",
                            minimum=0.0,
                            maximum=2.0,
                            value=config["frequency_penalty"],
                            step=0.1,
                            info="Penalize by frequency"
                        )
                
                run_btn = gr.Button("🚀 Run Video Inference", variant="primary", size="lg")
            
            with gr.Column():
                status_text = gr.Markdown("")
                output_text = gr.Textbox(label="AI Response", lines=15, interactive=False)
                extracted_frames = gr.Gallery(label="Extracted Frames (preview, max 50)", columns=5, height=300)
        
        # Config save/load handlers
        def save_current_config(backend, api_url, model_name, processing_mode, sampling_mode, interval, target_fps,
                                max_frames_limit, image_width, image_height, prompt, max_tokens,
                                temperature, top_p, top_k, repetition_penalty, presence_penalty, frequency_penalty):
            config = {
                "backend": backend,
                "api_url": api_url,
                "model_name": model_name,
                "processing_mode": processing_mode,
                "sampling_mode": sampling_mode,
                "interval": interval,
                "target_fps": target_fps,
                "max_frames_limit": max_frames_limit,
                "image_width": image_width,
                "image_height": image_height,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "presence_penalty": presence_penalty,
                "frequency_penalty": frequency_penalty
            }
            return save_config(config)
        
        def load_saved_config():
            config = load_config()
            return (
                config["backend"], config["api_url"], config["model_name"],
                config.get("processing_mode", "Frame Extraction (KoboldCPP/GGUF)"),
                config["sampling_mode"], config["interval"], config["target_fps"],
                config["max_frames_limit"], config["image_width"], config["image_height"],
                config["prompt"], config["max_tokens"], config["temperature"],
                config["top_p"], config["top_k"], config["repetition_penalty"],
                config["presence_penalty"], config["frequency_penalty"],
                f"✓ Config loaded from {CONFIG_FILE}"
            )
        
        save_config_btn.click(
            fn=save_current_config,
            inputs=[backend, api_url, model_name, processing_mode, sampling_mode, interval, target_fps,
                    max_frames_limit, image_width, image_height, prompt, max_tokens,
                    temperature, top_p, top_k, repetition_penalty, presence_penalty, frequency_penalty],
            outputs=[config_status]
        )
        
        load_config_btn.click(
            fn=load_saved_config,
            outputs=[backend, api_url, model_name, processing_mode, sampling_mode, interval, target_fps,
                     max_frames_limit, image_width, image_height, prompt, max_tokens,
                     temperature, top_p, top_k, repetition_penalty, presence_penalty, frequency_penalty,
                     config_status]
        )
        
        # Refresh models function
        def refresh_models(api_url):
            try:
                # Extract base URL (remove /v1/chat/completions)
                base_url = api_url.replace("/v1/chat/completions", "")
                models_url = f"{base_url}/v1/models"
                
                response = requests.get(models_url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    models = [m['id'] for m in data.get('data', [])]
                    if models:
                        return gr.Dropdown(choices=models, value=models[0])
                    return gr.Dropdown(choices=["No models found"])
                else:
                    return gr.Dropdown(choices=[f"Error: HTTP {response.status_code}"])
            except Exception as e:
                return gr.Dropdown(choices=[f"Error: {str(e)}"])
        
        refresh_models_btn.click(
            fn=refresh_models,
            inputs=[api_url],
            outputs=[model_name]
        )
        
        # Event handlers
        def update_sampling_controls(mode):
            if mode == "interval":
                return gr.update(visible=True), gr.update(visible=False)
            elif mode == "fps":
                return gr.update(visible=False), gr.update(visible=True)
            else:  # all
                return gr.update(visible=False), gr.update(visible=False)
        
        sampling_mode.change(
            fn=update_sampling_controls,
            inputs=[sampling_mode],
            outputs=[interval, target_fps]
        )
        
        # Event handlers
        def on_video_upload(video_path):
            if not video_path:
                return gr.update(visible=False), {}
            try:
                info = get_video_info(video_path)
                return gr.update(visible=True), info
            except:
                return gr.update(visible=False), {}
        
        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[video_info, video_info]
        )
        
        run_btn.click(
            fn=process_video,
            inputs=[
                video_input, api_url, model_name, backend, processing_mode, prompt, sampling_mode, interval, target_fps,
                max_frames_limit, image_width, image_height, max_tokens, temperature, top_p, top_k,
                repetition_penalty, presence_penalty, frequency_penalty
            ],
            outputs=[output_text, extracted_frames, video_info]
        )
    
    return app

if __name__ == "__main__":
    app = build_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        inbrowser=True
    )
