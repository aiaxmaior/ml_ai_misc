#!/usr/bin/env python3
"""
Pygame Model Selection GUI for TRT-LLM Quantization
Scans base directory, lets user pick model + format, then launches CLI quantization
"""

import os
import sys
import subprocess

# ---------------------------
# GPU Selection BEFORE any imports
# ---------------------------
print("Available GPUs:")
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name', '--format=csv,noheader'],
        capture_output=True,
        text=True,
        check=True
    )
    
    gpu_list = []
    for line in result.stdout.strip().split('\n'):
        if line:
            idx, name = line.split(', ', 1)
            gpu_list.append((idx.strip(), name.strip()))
            print(f"  {idx.strip()}) {name.strip()}")
    
    gpu_choice = input("\nSelect GPU ID for quantization (default 0): ").strip()
    gpu_id = gpu_choice if gpu_choice else "0"
    
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print(f"[INFO] Set CUDA_VISIBLE_DEVICES={gpu_id}\n")
    
except Exception as e:
    print(f"[WARNING] Could not query GPUs: {e}")
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# NOW import everything else (NO TORCH)
import pygame
import json
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
DEFAULT_BASE = "~/qdrive_alpha/jetson/quantize"
DEFAULT_QUANT = "./quants"

# Prompt for directories
base_input = input(f"Enter base model directory (default: {DEFAULT_BASE}): ").strip()
BASE_DIR = Path(base_input or DEFAULT_BASE).expanduser()

quant_input = input(f"Enter output quantized models directory (default: {DEFAULT_QUANT}): ").strip()
QUANT_DIR = Path(quant_input or DEFAULT_QUANT).expanduser()
QUANT_DIR.mkdir(exist_ok=True, parents=True)

# Validate base directory
if not BASE_DIR.exists():
    print(f"ERROR: Base directory does not exist: {BASE_DIR}")
    sys.exit(1)

print(f"[INFO] Scanning models in: {BASE_DIR}")
print(f"[INFO] Quantized models will be saved to: {QUANT_DIR}\n")

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BG_COLOR = (20, 20, 30)
TEXT_COLOR = (220, 220, 220)
ACCENT_COLOR = (0, 180, 255)
BUTTON_COLOR = (40, 40, 60)
BUTTON_HOVER = (60, 60, 80)
BUTTON_SELECTED = (0, 120, 200)

FORMATS = ["W4A16", "W8A8", "W4A8", "FP8"]
GPU_ARCHS = ["ampere", "ada", "hopper", "blackwell"]

# ---------------------------
# Scan for models
# ---------------------------
def scan_models(base_dir):
    """Find all model directories - torch-free scanning"""
    models = []
    if not base_dir.exists():
        return models
    
    for item in base_dir.iterdir():
        if item.is_dir():
            has_model = (item / "config.json").exists() or \
                       (item / "pytorch_model.bin").exists() or \
                       list(item.glob("*.safetensors"))
            
            if has_model:
                try:
                    size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024**3)
                except:
                    size = 0.0
                
                models.append({
                    "name": item.name,
                    "path": str(item),
                    "size": size
                })
    return sorted(models, key=lambda x: x["name"])

# ---------------------------
# UI Components
# ---------------------------
class TextInput:
    def __init__(self, x, y, width, height, label, default_text="", input_type="text"):
        self.rect = pygame.Rect(x, y, width, height)
        self.label = label
        self.text = str(default_text)
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0
        self.input_type = input_type
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                return True
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_TAB:
                return "tab"
            else:
                if self.input_type == "int":
                    if event.unicode.isdigit():
                        self.text += event.unicode
                else:
                    self.text += event.unicode
        return False
    
    def draw(self, screen, font, label_font):
        label_surf = label_font.render(self.label, True, (180, 180, 180))
        screen.blit(label_surf, (self.rect.x, self.rect.y - 25))
        
        color = ACCENT_COLOR if self.active else (80, 80, 90)
        pygame.draw.rect(screen, (35, 35, 45), self.rect, border_radius=5)
        pygame.draw.rect(screen, color, self.rect, 2, border_radius=5)
        
        text_surf = font.render(self.text, True, TEXT_COLOR)
        screen.blit(text_surf, (self.rect.x + 10, self.rect.y + 12))
        
        if self.active and self.cursor_visible:
            cursor_x = self.rect.x + 10 + text_surf.get_width() + 2
            pygame.draw.line(screen, TEXT_COLOR, 
                           (cursor_x, self.rect.y + 8),
                           (cursor_x, self.rect.bottom - 8), 2)
    
    def update(self, dt):
        self.cursor_timer += dt
        if self.cursor_timer > 500:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
    
    def get_value(self):
        if self.input_type == "int":
            return int(self.text) if self.text else 0
        return self.text

class ModelArchGrid:
    def __init__(self, x, y, width, height, architectures):
        self.rect = pygame.Rect(x, y, width, height)
        self.architectures = architectures
        self.scroll_offset = 0
        self.row_height = 40
        self.selected_idx = None
        
    def draw(self, screen, font, small_font):
        pygame.draw.rect(screen, (25, 25, 35), self.rect, border_radius=5)
        pygame.draw.rect(screen, ACCENT_COLOR, self.rect, 2, border_radius=5)
        
        header_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 45)
        pygame.draw.rect(screen, (40, 40, 50), header_rect)
        
        header_text = small_font.render("Index | Architecture | Model | Modality", True, ACCENT_COLOR)
        screen.blit(header_text, (self.rect.x + 10, self.rect.y + 12))
        
        y_pos = self.rect.y + 50 - self.scroll_offset
        
        for arch in self.architectures:
            if y_pos > self.rect.y + 45 and y_pos < self.rect.bottom:
                row_rect = pygame.Rect(self.rect.x + 5, y_pos, self.rect.width - 10, self.row_height - 2)
                
                if self.selected_idx == arch["index"]:
                    pygame.draw.rect(screen, BUTTON_SELECTED, row_rect, border_radius=3)
                elif row_rect.collidepoint(pygame.mouse.get_pos()):
                    pygame.draw.rect(screen, (45, 45, 55), row_rect, border_radius=3)
                
                idx_surf = small_font.render(f"{arch['index']}", True, (150, 200, 255))
                screen.blit(idx_surf, (row_rect.x + 10, row_rect.y + 10))
                
                name_surf = small_font.render(arch["name"], True, TEXT_COLOR)
                screen.blit(name_surf, (row_rect.x + 70, row_rect.y + 10))
                
                model_surf = small_font.render(arch["model"][:30], True, (150, 150, 150))
                screen.blit(model_surf, (row_rect.x + 450, row_rect.y + 10))
                
                mod_surf = small_font.render(arch["modality"], True, (150, 200, 255))
                screen.blit(mod_surf, (row_rect.x + 750, row_rect.y + 10))
            
            y_pos += self.row_height
        
        if len(self.architectures) * self.row_height > self.rect.height - 50:
            scrollbar_height = max(30, (self.rect.height - 50) * (self.rect.height - 50) / (len(self.architectures) * self.row_height))
            scrollbar_y = self.rect.y + 50 + (self.scroll_offset / (len(self.architectures) * self.row_height)) * (self.rect.height - 50)
            pygame.draw.rect(screen, (100, 100, 120), 
                           (self.rect.right - 10, scrollbar_y, 8, scrollbar_height), border_radius=4)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            relative_y = event.pos[1] - self.rect.y - 50 + self.scroll_offset
            idx = int(relative_y / self.row_height)
            if 0 <= idx < len(self.architectures):
                self.selected_idx = self.architectures[idx]["index"]
                return True
        
        elif event.type == pygame.MOUSEWHEEL and self.rect.collidepoint(pygame.mouse.get_pos()):
            max_scroll = max(0, len(self.architectures) * self.row_height - (self.rect.height - 50))
            self.scroll_offset = max(0, min(self.scroll_offset - event.y * 30, max_scroll))
        
        return False

class Button:
    def __init__(self, x, y, width, height, text, value=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.value = value
        self.selected = False
        self.hovered = False
    
    def draw(self, screen, font):
        color = BUTTON_SELECTED if self.selected else (BUTTON_HOVER if self.hovered else BUTTON_COLOR)
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, ACCENT_COLOR, self.rect, 2, border_radius=5)
        
        text_surf = font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False

class ScrollList:
    def __init__(self, x, y, width, height, items):
        self.rect = pygame.Rect(x, y, width, height)
        self.items = items
        self.selected_idx = 0
        self.scroll_offset = 0
        self.item_height = 60
    
    def draw(self, screen, font, small_font):
        pygame.draw.rect(screen, (30, 30, 40), self.rect, border_radius=5)
        pygame.draw.rect(screen, ACCENT_COLOR, self.rect, 2, border_radius=5)
        
        y_pos = self.rect.y + 10 - self.scroll_offset
        for idx, item in enumerate(self.items):
            if y_pos > self.rect.y - self.item_height and y_pos < self.rect.bottom:
                item_rect = pygame.Rect(self.rect.x + 10, y_pos, self.rect.width - 20, self.item_height - 5)
                
                if idx == self.selected_idx:
                    pygame.draw.rect(screen, BUTTON_SELECTED, item_rect, border_radius=3)
                
                name_surf = font.render(item["name"][:30], True, TEXT_COLOR)
                screen.blit(name_surf, (item_rect.x + 10, item_rect.y + 5))
                
                size_text = f"{item['size']:.2f} GB"
                size_surf = small_font.render(size_text, True, (150, 150, 150))
                screen.blit(size_surf, (item_rect.x + 10, item_rect.y + 30))
            
            y_pos += self.item_height
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                relative_y = event.pos[1] - self.rect.y + self.scroll_offset
                idx = int(relative_y / self.item_height)
                if 0 <= idx < len(self.items):
                    self.selected_idx = idx
                    return True
        elif event.type == pygame.MOUSEWHEEL:
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.scroll_offset = max(0, min(self.scroll_offset - event.y * 30, 
                                                 len(self.items) * self.item_height - self.rect.height))
        return False

# ---------------------------
# Main App
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("TRT-LLM Quantization - Model Selection")
    clock = pygame.time.Clock()
    
    title_font = pygame.font.Font(None, 48)
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 22)
    label_font = pygame.font.Font(None, 20)
    
    architectures = []
    try:
        with open("architectures.jsonl", 'r') as f:
            for line in f:
                if line.strip():
                    architectures.append(json.loads(line))
    except FileNotFoundError:
        print("WARNING: architectures.jsonl not found, using default phi3")
        architectures = [{"index": 19, "name": "Phi3ForCausalLM", "model": "Phi-4", "modality": "L"}]
    
    models = scan_models(BASE_DIR)
    if not models:
        print(f"ERROR: No models found in {BASE_DIR}")
        sys.exit(1)
    
    print(f"[INFO] Found {len(models)} models\n")
    
    model_list = ScrollList(50, 120, 500, 850, models)
    
    format_buttons = []
    for idx, fmt in enumerate(FORMATS):
        btn = Button(600, 120 + idx * 70, 200, 60, fmt, fmt)
        if idx == 0:
            btn.selected = True
        format_buttons.append(btn)
    
    arch_buttons = []
    for idx, arch in enumerate(GPU_ARCHS):
        btn = Button(850, 120 + idx * 70, 200, 60, arch, arch)
        if idx == 0:
            btn.selected = True
        arch_buttons.append(btn)
    
    calib_samples_input = TextInput(1100, 140, 180, 50, "Calib Samples", "256", "int")
    calib_seq_len_input = TextInput(1320, 140, 180, 50, "Seq Length", "512", "int")
    output_dir_input = TextInput(1100, 240, 580, 50, "Output Dir", str(QUANT_DIR))
    text_inputs = [calib_samples_input, calib_seq_len_input, output_dir_input]
    
    select_arch_btn = Button(1100, 340, 300, 60, "Select Architecture")
    quantize_btn = Button(SCREEN_WIDTH // 2 - 200, 980, 400, 70, "QUANTIZE")
    
    show_arch_selection = False
    arch_grid = None
    selected_model_name = "phi3"
    
    running = True
    while running:
        dt = clock.get_time()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if show_arch_selection:
                if arch_grid and arch_grid.handle_event(event):
                    selected_arch = next(a for a in architectures if a["index"] == arch_grid.selected_idx)
                    selected_model_name = selected_arch["name"]
                    show_arch_selection = False
                    print(f"[INFO] Selected: {selected_model_name}")
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    back_rect = pygame.Rect(SCREEN_WIDTH - 220, 20, 200, 50)
                    if back_rect.collidepoint(event.pos):
                        show_arch_selection = False
            
            else:
                model_list.handle_event(event)
                
                for btn in format_buttons:
                    if btn.handle_event(event):
                        for b in format_buttons:
                            b.selected = False
                        btn.selected = True
                
                for btn in arch_buttons:
                    if btn.handle_event(event):
                        for b in arch_buttons:
                            b.selected = False
                        btn.selected = True
                
                for inp in text_inputs:
                    result = inp.handle_event(event)
                    if result == "tab":
                        current_idx = text_inputs.index(inp)
                        next_idx = (current_idx + 1) % len(text_inputs)
                        for ti in text_inputs:
                            ti.active = False
                        text_inputs[next_idx].active = True
                
                if select_arch_btn.handle_event(event):
                    show_arch_selection = True
                    arch_grid = ModelArchGrid(100, 100, 1720, 880, architectures)
                
                if quantize_btn.handle_event(event):
                    selected_model = models[model_list.selected_idx]
                    selected_format = next(b.value for b in format_buttons if b.selected)
                    selected_arch = next(b.value for b in arch_buttons if b.selected) 
                    
                    cmd = [
                        "python", "custom_trt-llm_quantize_v2.py",
                        "--format", selected_format,
                        "--gpu_arch", selected_arch,
                        "--input_dir", selected_model["path"],
                        "--output_dir", output_dir_input.get_value(),
                        "--model_name", selected_model_name,
                        "--calib_samples", str(calib_samples_input.get_value()),
                        "--calib_seq_len", str(calib_seq_len_input.get_value())
                    ]

                    print("\n" + "="*60)
                    print("LAUNCHING QUANTIZATION")
                    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
                    print("="*60)
                    print(" ".join(cmd))
                    print("="*60 + "\n")
                    
                    pygame.quit()
                    subprocess.run(cmd)
                    return
        
        screen.fill(BG_COLOR)
        
        if show_arch_selection and arch_grid:
            title = title_font.render("Select Model Architecture", True, ACCENT_COLOR)
            screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 30))
            
            arch_grid.draw(screen, font, small_font)
            
            back_btn = Button(SCREEN_WIDTH - 220, 20, 200, 50, "Back")
            back_btn.draw(screen, small_font)
        
        else:
            title = title_font.render("TRT-LLM Quantization Setup", True, ACCENT_COLOR)
            screen.blit(title, (SCREEN_WIDTH // 2 - title.get_width() // 2, 30))
            
            labels = [
                ("Models:", 50, 90),
                ("Format:", 600, 90),
                ("GPU:", 850, 90),
                ("Settings:", 1100, 90)
            ]
            for text, x, y in labels:
                surf = font.render(text, True, TEXT_COLOR)
                screen.blit(surf, (x, y))
            
            model_list.draw(screen, font, small_font)
            for btn in format_buttons + arch_buttons:
                btn.draw(screen, font)
            
            for inp in text_inputs:
                inp.draw(screen, font, label_font)
                inp.update(dt)
            
            select_arch_btn.draw(screen, font)
            quantize_btn.draw(screen, font)
            
            selected = models[model_list.selected_idx]
            info_text = f"Model: {selected['name']} | Arch: {selected_model_name} | {selected['size']:.2f} GB"
            info_surf = small_font.render(info_text, True, (180, 180, 180))
            screen.blit(info_surf, (50, 1040))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()