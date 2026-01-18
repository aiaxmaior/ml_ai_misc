#!/usr/bin/env python3
"""
🎬 VIDEO PROMO SIFTER v3.1 - "The Production Lab"
Optimized for Qwen3-VL 8B Thinking. 
Features: FFmpeg Pre-Processing, 40k Token Support, and Industry-Standard Thought Syntax.
"""

import cv2
import base64
import requests
import gradio as gr
from pathlib import Path
import subprocess
import tempfile
from PIL import Image
import io
import json
import time
import os

# Config file path
CONFIG_FILE = Path(__file__).parent / "video_inf_config.json"

DEFAULT_CONFIG = {
    "api_url": "http://localhost:8000/v1/chat/completions",
    "model_name": "Qwen-VL",
    "processing_mode": "Native Video (vLLM)",
    "sampling_mode": "fps",
    "interval": 2.0,
    "target_fps": 1.0,
    "max_frames_limit": 0,
    "image_width": 640,
    "image_height": 480,
    "prompt": "Identify the top cinematic segments in this video suitable for a promo. List timestamps and justify why they are high-energy.",
    "max_tokens": 40960,
    "temperature": 0.7,
    "top_p": 0.9,
    "min_p": 0.05,
    "top_k": 20,
    "repetition_penalty": 1.15,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "thought_syntax": "<|placeholder_thinking|>{content}</|placeholder_thinking|>",
    "vram_limit": 164000
}

# Industry Standard Syntaxes (Renamed for stability)
THOUGHT_SYNTAX_CHOICES = [
    ("<|placeholder_thinking|>{content}</|placeholder_thinking|>", "ChatML (Qwen Standard): <|thought|>..."),
    ("<placeholder_thinking>{content}</placeholder_thinking>", "XML (Open Source Standard): <thought>..."),
    ("[placeholder_thinking]{content}[/placeholder_thinking]", "BBCode: [THOUGHT]..."),
    ("(placeholder_thinking){content}", "Parentheses: (thinking)..."),
    ("", "No special tag")
]

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        return "✓ Settings Saved"
    except Exception as e:
        return f"✗ Error: {str(e)}"

def get_video_info(video_path):
    if not video_path: return {}
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    video.release()
    return {"fps": fps, "frames": total_frames, "res": f"{width}x{height}", "dur": duration}

def run_ffmpeg_process(input_path, start_t, end_t, width, height, crf=28):
    """Uses FFmpeg to create an optimized sub-clip for analysis"""
    output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_t), "-to", str(end_t),
        "-i", input_path, "-vf", f"scale={int(width)}:{int(height)}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", "veryfast", "-c:a", "copy", output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path
    except Exception as e:
        return f"FFmpeg Error: {str(e)}"

def update_token_estimate(dur, fps, w, h, limit):
    """Calculates estimated visual tokens vs context limit"""
    if not dur or not fps: return "### Token Status\nUpload video to estimate..."
    # Approximate visual token logic for VL models
    tokens_per_frame = (w * h) / 784
    total_visual_tokens = (dur * fps / 2) * tokens_per_frame
    remaining = limit - total_visual_tokens
    color = "green" if remaining > 20000 else "orange" if remaining > 0 else "red"
    status = f"### Token Status\n- **Visual Tokens:** {int(total_visual_tokens):,}\n"
    status += f"- **Thinking Buffer:** <span style='color:{color}'>{int(remaining):,}</span> / {limit:,}"
    return status

def refresh_models(api_url):
    try:
        base_url = api_url.split("/v1/")[0]
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models = [m['id'] for m in response.json().get('data', [])]
            return gr.update(choices=models, value=models[0] if models else "")
    except: pass
    return gr.update(choices=["Offline / Check URL"])

def make_markdown_instruction(thought_syntax):
    if "{content}" in thought_syntax and thought_syntax:
        open_tag = thought_syntax.split("{content}")[0]
        close_tag = thought_syntax.split("{content}")[1]
    else:
        open_tag, close_tag = "[placeholder_thinking]", "[/placeholder_thinking]"
    
    return f"""# SYSTEM INSTRUCTIONS
Analyze the video and provide high-quality marketing timestamps.
- CRITICAL: You MUST perform all your intermediate reasoning and timestamp calculations inside {open_tag} and {close_tag} tags.
- The final response outside the tags must be a clean, copy-pasteable Markdown table.
"""

def process_video_streaming(video_path, api_url, model_name, processing_mode, prompt, sampling_mode, interval, target_fps,
                           max_frames_limit, image_width, image_height, max_tokens, temperature, top_p, min_p, top_k,
                           rep_p, pres_p, freq_p, thought_syntax):
    if not video_path: yield "Error: No video uploaded.", [], {}; return
    info = get_video_info(video_path)
    
    user_instruction = make_markdown_instruction(thought_syntax)
    prompt_injected = user_instruction + "\n\n" + prompt
    
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt_injected}]}]
    frame_images = []
    
    if "Native Video" in processing_mode:
        messages[0]["content"].append({"type": "video_url", "video_url": {"url": f"file://{video_path}"}})
        _, frame_images = extract_frames_for_preview(video_path)
    else:
        frames, frame_images = extract_frames_manual(video_path, sampling_mode, interval, target_fps, max_frames_limit, (image_width, image_height))
        for f in frames:
            _, buf = cv2.imencode('.jpg', f, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64 = base64.b64encode(buf).decode('utf-8')
            messages[0]["content"].append({
                "type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

    payload = {
        "model": model_name, "messages": messages, "max_tokens": int(max_tokens), "temperature": float(temperature),
        "top_p": float(top_p), "min_p": float(min_p), "top_k": int(top_k), "presence_penalty": float(pres_p),
        "frequency_penalty": float(freq_p), "repetition_penalty": float(rep_p), "stream": True 
    }

    try:
        response = requests.post(api_url, json=payload, stream=True, timeout=600)
        response.raise_for_status()
        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8').replace('data: ', '')
                if decoded == '[DONE]': break
                try:
                    chunk = json.loads(decoded)
                    if 'choices' in chunk and len(chunk['choices']) > 0:
                        content = chunk['choices'][0].get('delta', {}).get('content', '')
                        full_response += content
                        yield full_response, frame_images, info
                except: continue
    except Exception as e: yield f"API Error: {str(e)}", [], info

def extract_frames_for_preview(video_path):
    video = cv2.VideoCapture(video_path)
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, frames_count // 24)
    frame_images = []
    for i in range(24):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, frame = video.read()
        if success:
            frame_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    video.release()
    return [], frame_images

def extract_frames_manual(video_path, sampling_mode, interval, target_fps, max_frames, size):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames, frame_images = [], []
    step = max(1, int(fps / target_fps)) if sampling_mode == "fps" else max(1, int(fps * interval))
    current = 0
    while video.isOpened():
        success, frame = video.read()
        if not success: break
        if current % step == 0:
            resized = cv2.resize(frame, (int(size[0]), int(size[1])))
            frames.append(resized)
            if len(frame_images) < 50:
                frame_images.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
        current += 1
        if max_frames and len(frames) >= int(max_frames): break
    video.release()
    return frames, frame_images

def build_interface():
    config = load_config()
    theme = gr.themes.Monochrome(primary_hue="neutral", radius_size="none", font=[gr.themes.GoogleFont("Source Serif Pro"), "serif"], font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"])
    
    with gr.Blocks(theme=theme, title="Promo Sifter Production Lab") as app:
        gr.Markdown("# 🎬 VIDEO PROMO SIFTER: PRODUCTION LAB")
        
        with gr.Row():
            # Left column for configuration and inputs
            with gr.Column(scale=1):
                raw_video = gr.Video(label="Original Source")
                
                with gr.Accordion("🛠 FFmpeg Pre-Process Lab", open=False):
                    with gr.Row():
                        start_time = gr.Number(label="Start (s)", value=0)
                        end_time = gr.Number(label="End (s)", value=60)
                    with gr.Row():
                        proc_w = gr.Number(label="Width", value=640)
                        proc_h = gr.Number(label="Height", value=480)
                    process_btn = gr.Button("✂️ GENERATE OPTIMIZED CLIP", variant="secondary")
                
                analysis_video = gr.Video(label="Current Analysis Target")
                token_status = gr.Markdown("### Token Status\nUpload video to estimate...")

                with gr.Accordion("🎞 Frame Sampling", open=True):
                    proc_mode = gr.Radio(["Native Video (vLLM)", "Extraction"], label="Mode", value=config["processing_mode"])
                    samp_mode = gr.Radio(["fps", "interval"], label="Method", value=config["sampling_mode"])
                    with gr.Row():
                        interval_in = gr.Number(label="Interval (s)", value=config["interval"], visible=(config["sampling_mode"] == "interval"))
                        target_fps = gr.Slider(0.1, 10.0, value=config["target_fps"], label="Target FPS", visible=(config["sampling_mode"] == "fps"))

                with gr.Accordion("⚙️ Inference Tuning", open=False):
                    max_t = gr.Slider(512, 40960, step=512, label="Max Output Tokens", value=config["max_tokens"])
                    temp = gr.Slider(0.0, 2.0, value=config["temperature"], label="Temp")
                    with gr.Row():
                        top_p = gr.Slider(0.0, 1.0, value=config["top_p"], label="Top P")
                        min_p = gr.Slider(0.0, 1.0, value=config["min_p"], label="Min P")
                    with gr.Row():
                        rep_p = gr.Slider(1.0, 2.0, value=config["repetition_penalty"], label="Rep Penalty")
                        freq_p = gr.Slider(0.0, 2.0, value=config["frequency_penalty"], label="Freq Penalty")
                        pres_p = gr.Slider(0.0, 2.0, value=config["presence_penalty"], label="Pres Penalty")
                    thought_syntax = gr.Dropdown(choices=[c[0] for c in THOUGHT_SYNTAX_CHOICES], value=config["thought_syntax"], label="Thought Syntax", allow_custom_value=True)

                with gr.Accordion("🔗 Connection", open=False):
                    with gr.Row():
                        api_url = gr.Textbox(label="URL", value=config["api_url"], scale=4)
                        refresh_btn = gr.Button("🔄", scale=1)
                    model_name = gr.Dropdown(label="Model ID", value=config["model_name"], allow_custom_value=True, choices=[config["model_name"]])
                save_btn = gr.Button("SAVE SETTINGS", variant="secondary")

            # Right column for prompt and output
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Analysis Goal", value=config["prompt"], lines=4)
                with gr.Row():
                    run_btn = gr.Button("RUN ANALYSIS", variant="primary", scale=4)
                    stop_btn = gr.Button("STOP", variant="stop", scale=1)
                
                output_text = gr.Textbox(label="AI Reasoning & Analysis", lines=30, interactive=False)
                
                with gr.Accordion("Visual Evidence", open=False):
                    gallery = gr.Gallery(columns=6, height="auto")

        # Logic
        def on_video_load(path):
            if not path: return 0, 0, "### Token Status\nNo video loaded."
            info = get_video_info(path)
            token_md = update_token_estimate(info['dur'], config['target_fps'], 640, 480, config['vram_limit'])
            return 0, info['dur'], token_md

        raw_video.change(on_video_load, inputs=raw_video, outputs=[start_time, end_time, token_status])
        
        def update_samp(m):
            return gr.update(visible=m=="interval"), gr.update(visible=m=="fps")
        samp_mode.change(update_samp, inputs=samp_mode, outputs=[interval_in, target_fps])
        
        process_btn.click(run_ffmpeg_process, [raw_video, start_time, end_time, proc_w, proc_h], analysis_video)
        refresh_btn.click(refresh_models, api_url, model_name)
        
        run_event = run_btn.click(process_video_streaming, 
            [analysis_video, api_url, model_name, proc_mode, prompt, samp_mode, interval_in, target_fps, gr.State(0), proc_w, proc_h, max_t, temp, top_p, min_p, gr.State(20), rep_p, pres_p, freq_p, thought_syntax], 
            [output_text, gallery, gr.State()])
        
        stop_btn.click(None, None, None, cancels=[run_event])
        
        save_btn.click(
            fn=lambda *args: save_config({
                "api_url": args[0], "model_name": args[1], "processing_mode": args[2],
                "sampling_mode": args[3], "target_fps": args[4], "image_width": args[5],
                "image_height": args[6], "prompt": args[7], "max_tokens": args[8],
                "temperature": args[9], "top_p": args[10], "min_p": args[11],
                "repetition_penalty": args[12], "presence_penalty": args[13],
                "frequency_penalty": args[14], "thought_syntax": args[15], "interval": args[16]
            }),
            inputs=[api_url, model_name, proc_mode, samp_mode, target_fps, proc_w, proc_h, prompt, max_t, temp, top_p, min_p, rep_p, pres_p, freq_p, thought_syntax, interval_in],
            outputs=[gr.Markdown("Status")]
        )
        
    return app

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7861)