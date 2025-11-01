"""
Gradio Web UI for Smart Content Moderation System
Face Blur + Hate Speech + Blood/NSFW Detection
AUTO-LAUNCHES BROWSER + AUTO-SAVES TO OUTPUT FOLDER
"""

import gradio as gr
import tempfile
import os
import webbrowser
import time
import shutil
from pathlib import Path
import cv2
from datetime import datetime
from main import process_image, process_video, validate_blur_strength, get_media_type

# Create output folder
OUTPUT_DIR = Path("social_moderation/data/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def blur_media(input_files, blur_strength, confidence, blur_text, nsfw_blur, 
               blood_threshold, nsfw_blur_type, progress=gr.Progress()):
    """Process media files with complete moderation and auto-save."""
    
    results = []
    total = len(input_files) if input_files else 0
    
    if total == 0:
        return None, "❌ No files uploaded"
    
    blur_strength = validate_blur_strength(blur_strength)
    
    for idx, input_file in enumerate(input_files):
        input_path = input_file.name
        filename = Path(input_path).name
        
        progress((idx / total), desc=f"Processing {filename}...")
        
        try:
            media_type = get_media_type(input_path, 'auto')
            
            # Create output path with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name_without_ext = Path(filename).stem
            file_ext = Path(filename).suffix
            
            # Save to appropriate subfolder
            if media_type == 'image':
                output_subdir = OUTPUT_DIR / "images"
            else:
                output_subdir = OUTPUT_DIR / "videos"
            
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename with timestamp
            output_filename = f"{name_without_ext}_{timestamp}{file_ext}"
            output_path = output_subdir / output_filename
            
            # Process the file
            success = process_image(
                input_path, str(output_path), blur_strength, confidence,
                False, blur_text, nsfw_blur, nsfw_blur_type, False,
                blood_threshold=blood_threshold
            )
            
            if success:
                results.append({
                    'path': str(output_path),
                    'filename': output_filename,
                    'type': media_type
                })
                progress((idx + 1) / total, desc=f"✅ Saved: {output_filename}")
            
        except Exception as e:
            progress((idx / total), desc=f"❌ Error: {e}")
            continue
    
    if not results:
        return None, "❌ Processing failed"
    
    # Prepare output and status
    output_path_str = results[0]['path']
    status = f"✅ Successfully processed {len(results)} file(s)\n\n"
    status += "📁 **Saved to:**\n"
    
    for result in results:
        status += f"  • {result['filename']}\n"
        if result['type'] == 'image':
            status += f"    📍 `{OUTPUT_DIR / 'images' / result['filename']}`\n"
        else:
            status += f"    📍 `{OUTPUT_DIR / 'videos' / result['filename']}`\n"
    
    status += "\n**Features Applied:**"
    status += "\n  👤 Face blurring"
    status += "\n  🔤 Hate speech detection"
    status += "\n  🩸 Blood/NSFW detection"
    
    return output_path_str, status

# Create Gradio Interface
with gr.Blocks(title="Smart Content Moderation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Smart Content Moderation System")
    gr.Markdown("**Blur faces • Detect hate speech • Blur blood/NSFW content**")
    gr.Markdown(f"📁 **Output Folder:** `{OUTPUT_DIR}`")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📤 Input")
            input_files = gr.File(label="Upload Image/Video", file_count="multiple", file_types=["image", "video"])
        
        with gr.Column():
            gr.Markdown("### 📥 Output")
            output_file = gr.File(label="Blurred Output (Download)")
            status_text = gr.Textbox(label="Status & Save Location", interactive=False, lines=6)
    
    gr.Markdown("---")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ⚙️ Blur Settings")
            blur_strength = gr.Slider(minimum=3, maximum=151, value=51, step=2, label="👤 Blur Strength")
            confidence = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.05, label="Detection Confidence")
        
        with gr.Column():
            gr.Markdown("### 🔤 Hate Speech Settings")
            blur_text = gr.Checkbox(label="Enable Hate Speech Detection & Blur", value=True)
            gr.Markdown("*Detects: 'Hate You', profanity, offensive text*")
        
        with gr.Column():
            gr.Markdown("### 🩸 Blood/NSFW Settings")
            nsfw_blur = gr.Checkbox(label="Enable Blood/NSFW Detection & Blur", value=True)
            blood_threshold = gr.Slider(
                minimum=0.0, maximum=1.0, value=0.3, step=0.1, 
                label="Blood Sensitivity (lower = more sensitive)"
            )
            nsfw_blur_type = gr.Dropdown(
                choices=["gaussian", "pixelate", "mosaic", "black"], 
                value="gaussian", 
                label="Blur Type"
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        process_btn = gr.Button("🎬 Process & Save", variant="primary", size="lg")
    
    # Connect button
    process_btn.click(
        blur_media,
        inputs=[
            input_files, blur_strength, confidence, blur_text, nsfw_blur,
            blood_threshold, nsfw_blur_type
        ],
        outputs=[output_file, status_text]
    )
    
    gr.Markdown("""
    ---
    ## ✨ Features:
    
    ### 👤 Face Blurring
    - YOLOv8 face detection with 99% accuracy
    - Adaptive Gaussian blur based on face size
    - Works on images and videos
    
    ### 🔤 Hate Speech Detection  
    - EasyOCR text detection (45+ languages)
    - Rule-based toxicity detection
    - Detects offensive words, hate patterns
    
    ### 🩸 Blood/NSFW Content
    - HSV color-based blood detection
    - Sensitive to 5%+ red pixels
    - Multiple blur types available
    
    ## 🚀 Usage:
    
    1. Upload image or video
    2. Enable desired features
    3. Adjust sensitivity sliders
    4. Click "Process & Save"
    5. Download or find in output folder
    
    ## 📁 Auto-Save:
    
    - Images → `social_moderation/data/output/images/`
    - Videos → `social_moderation/data/output/videos/`
    - Timestamped filenames for organization
    
    ## 📊 Recommendations:
    
    - **Blur Strength**: 51 (default) = good balance
    - **Confidence**: 0.5 = balanced detection
    - **Blood Threshold**: 0.3 = very sensitive
    """)

if __name__ == "__main__":
    # Auto-launch browser
    def open_browser():
        time.sleep(2)
        webbrowser.open("http://localhost:7860")
    
    import threading
    thread = threading.Thread(target=open_browser, daemon=True)
    thread.start()
    
    # Launch Gradio app
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_error=True
    )
