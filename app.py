import os
import shutil
import subprocess
import uuid
import time
from flask import Flask, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from gtts import gTTS

# Optional imports for Stable Diffusion. If present and configured, the server will use it.
try:
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False

# ------- Config -------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SD_MODEL_ID = os.environ.get('SD_MODEL_ID', 'runwayml/stable-diffusion-v1-5')
FPS_DEFAULT = 8

app = Flask(__name__, static_folder=os.path.join(BASE_DIR, '..', 'frontend'), static_url_path='')
CORS(app)

# Load SD pipelines lazily
sd_text2img_pipe = None
sd_img2img_pipe = None

def load_sd_pipelines():
    global sd_text2img_pipe, sd_img2img_pipe
    if not SD_AVAILABLE:
        return False
    if sd_text2img_pipe is None:
        print('Loading Stable Diffusion pipelines...')
        sd_text2img_pipe = StableDiffusionPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16)
        sd_img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            sd_text2img_pipe.to('cuda')
            sd_img2img_pipe.to('cuda')
        else:
            sd_text2img_pipe.to('cpu')
            sd_img2img_pipe.to('cpu')
    return True

@app.route('/')
def index():
    return send_from_directory(os.path.join(BASE_DIR, '..', 'frontend'), 'index.html')

# Helper to convert HSV to RGB
def ImageColorFromHSV(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v)
    return (r, g, b)

# Placeholder frame generator
def placeholder_generate_frames(prompt, frames, w=512, h=512):
    out = []
    for i in range(frames):
        img = Image.new('RGBA', (w, h), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)

        hue = int((i / max(1, frames)) * 360)
        color = tuple(int(x * 255) for x in ImageColorFromHSV(hue, 0.6, 0.95))
        draw.rectangle([0, 0, w, h], fill=color)

        # Main character
        cx = int(w * (0.2 + 0.6 * (i / max(1, frames))))
        cy = int(h * 0.5 + 20 * np.sin(i / max(1, frames) * 6.28))
        draw.ellipse([cx-60, cy-60, cx+60, cy+60], fill=(255, 220, 60))

        # Friend
        fx = int(w * (0.6 - 0.4 * (i / max(1, frames))))
        fy = int(h * 0.6 + 10 * np.cos(i / max(1, frames) * 6.28))
        draw.ellipse([fx-36, fy-36, fx+36, fy+36], fill=(255, 120, 160))

        try:
            font = ImageFont.truetype('arial.ttf', 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((12, 12), f"{prompt[:50]}", font=font, fill=(10, 10, 10))

        filename = f"frame_{i:04d}.png"
        path = os.path.join(OUTPUT_DIR, filename)
        img.convert('RGB').save(path)
        out.append(path)
    return out

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get('prompt', 'A colorful cartoon')[:512]
    duration = float(data.get('duration', 6))
    fps = int(data.get('fps', FPS_DEFAULT))
    use_tts = bool(data.get('use_tts', True))

    request_id = str(int(time.time())) + '_' + uuid.uuid4().hex[:8]
    job_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(job_dir, exist_ok=True)

    frames_count = max(1, int(duration * fps))

    # 1) Generate frames
    frames = []
    if SD_AVAILABLE:
        try:
            load_sd_pipelines()
            img = sd_text2img_pipe(prompt, guidance_scale=7.5).images[0]
            img.save(os.path.join(job_dir, 'frame_0000.png'))
            prev = img
            frames.append(os.path.join(job_dir, 'frame_0000.png'))
            for i in range(1, frames_count):
                new = sd_img2img_pipe(prompt, image=prev, strength=0.35, guidance_scale=7.5).images[0]
                p = os.path.join(job_dir, f'frame_{i:04d}.png')
                new.save(p)
                frames.append(p)
                prev = new
        except Exception as e:
            print('Stable Diffusion generation failed, using placeholder:', e)
            frames = placeholder_generate_frames(prompt, frames_count)
    else:
        frames = placeholder_generate_frames(prompt, frames_count)

    # 2) Generate TTS audio
    audio_path = os.path.join(job_dir, 'voice.mp3')
    if use_tts:
        try:
            tts = gTTS(text=prompt, lang='en')
            tts.save(audio_path)
        except Exception as e:
            print('TTS failed:', e)
            audio_path = None
    else:
        audio_path = None

    # 3) Assemble video
    video_path = os.path.join(job_dir, 'output.mp4')
    ffmpeg_cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(job_dir, 'frame_%04d.png')]
    if audio_path and os.path.exists(audio_path):
        ffmpeg_cmd += ['-i', audio_path, '-shortest']
    ffmpeg_cmd += ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path]

    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except Exception as e:
        print('ffmpeg failed, try moviepy fallback:', e)
        from moviepy.editor import ImageSequenceClip, AudioFileClip
        clip = ImageSequenceClip(frames, fps=fps)
        if audio_path and os.path.exists(audio_path):
            audio = AudioFileClip(audio_path)
            clip = clip.set_audio(audio.set_duration(clip.duration))
        clip.write_videofile(video_path, codec='libx264')

    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    print('Stable Diffusion available:', SD_AVAILABLE)
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

