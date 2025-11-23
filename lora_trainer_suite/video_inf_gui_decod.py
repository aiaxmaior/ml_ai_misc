#!/usr/bin/env python3
"""
Standalone Gradio Interface for Video Inference
Supports both manual frame extraction (KoboldCpp/GGUF) and native video (vLLM/Standard Models)
"""

import cv2
import base64
import requests
import gradio as gr
import os

def encode_file_base64(file_path):
    """Encode any file to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_frame(frame):
    """Encode OpenCV frame to base64 JPEG"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def extract_frames_preview(video_path, limit=24):
    """Extract a few frames just for UI preview"""
    video = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // limit)
    
    count = 0
    while video.isOpened() and len(frames) < limit:
        ret, frame = video.read()
        if not ret: break
        if count % step == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1
    video.release()
    return frames

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

def process_video(video_path, api_url, model_name, mode, prompt, max_tokens, temperature):
    if not video_path:
        return "Please upload a video.", []
    
    try:
        messages = [{"role": "user", "content": []}]
        
        # --- MODE 1: Native Video (vLLM Standard) ---
        if mode == "Native Video (vLLM)":
            print("Processing in Native Video mode...")
            video_b64 = encode_file_base64(video_path)
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}
                }
            ]
            
        # --- MODE 2: Frame Extraction (KoboldCpp / GGUF) ---
        else:
            print("Processing in Frame Extraction mode...")
            # Extract frames (1 fps default for simplicity in this mode)
            video = cv2.VideoCapture(video_path)
            fps = video.get(cv2.CAP_PROP_FPS)
            step = int(fps) # 1 frame per second
            
            messages[0]["content"].append({"type": "text", "text": prompt})
            
            count = 0
            frames_sent = 0
            while video.isOpened() and frames_sent < 16: # Limit to 16 frames for GGUF context safety
                ret, frame = video.read()
                if not ret: break
                if count % step == 0:
                    # Resize to 512x512 to save context
                    frame = cv2.resize(frame, (512, 512))
                    b64_img = encode_frame(frame)
                    messages[0]["content"].append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    })
                    frames_sent += 1
                count += 1
            video.release()

        # Send Request
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature)
        }

        print(f"Sending request to {api_url}...")
        response = requests.post(api_url, json=payload, timeout=300) # 5 min timeout for video
        response.raise_for_status()
        result = response.json()
        
        ai_response = result['choices'][0]['message']['content']
        
        # Generate preview frames for UI
        preview_frames = extract_frames_preview(video_path)
        
        return ai_response, preview_frames

    except Exception as e:
        return f"Error: {str(e)}", []

def build_interface():
    with gr.Blocks(title="Universal Video Inference") as app:
        gr.Markdown("# 🎥 Universal Video Inference (vLLM Native + GGUF)")
        
        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.Video(label="Input Video", format="mp4")
                video_info = gr.JSON(label="Metadata")
                
                api_url = gr.Textbox(label="API URL", value="http://localhost:8000/v1/chat/completions")
                model_name = gr.Textbox(label="Model Name", value="Qwen/Qwen2.5-VL-7B-Instruct")
                
                mode = gr.Radio(
                    label="Processing Mode",
                    choices=["Native Video (vLLM)", "Frame Extraction (GGUF/Kobold)"],
                    value="Native Video (vLLM)",
                    info="Use Native for full vLLM models. Use Extraction for GGUF/KoboldCpp."
                )
                
                prompt = gr.Textbox(label="Prompt", value="Describe this video in detail.", lines=2)
                max_tokens = gr.Number(label="Max Tokens", value=512)
                temperature = gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.1)
                
                btn = gr.Button("Run Inference", variant="primary")
            
            with gr.Column(scale=1):
                output = gr.Textbox(label="Analysis", lines=10)
                gallery = gr.Gallery(label="Video Preview Frames")

        # Update metadata on upload
        video_input.change(fn=get_video_info, inputs=video_input, outputs=video_info)
        
        btn.click(
            fn=process_video,
            inputs=[video_input, api_url, model_name, mode, prompt, max_tokens, temperature],
            outputs=[output, gallery]
        )

    return app

if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7861)