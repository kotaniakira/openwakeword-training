import os
import random
import subprocess
import soundfile as sf
import librosa
import numpy as np
import torch
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages
from style_bert_vits2.tts_model import TTSModel
from tqdm import tqdm
import time

import traceback

from style_bert_vits2.nlp import bert_models

# --- Configuration ---
OUTPUT_DIR = "positive_clips"
TMP_DIR = "tmp_raw"
TARGET_COUNT = 3000
SAMPLE_RATE = 16000

# Text variations (Must include all)
TEXTS = [
    "空也よ", "空也よ！", "空也よ…", "空也よ。", "空也よ、", 
    "空也よ？", "空也よっ", "空也よー", "くうやよ", "くーやよ"
]

# Parameters
SPEEDS = [0.85, 1.0, 1.15]
MIN_DURATION = 0.6
MAX_DURATION = 2.0

# --- Setup ---
def setup_model():
    print("Loading Style-Bert-VITS2 JP-Extra model...")
    try:
        # Load BERT
        bert_path = os.path.abspath('bert_models/deberta-v2-large-japanese-char-wwm')
        print(f"Loading BERT from: {bert_path}")
        bert_models.load_tokenizer(Languages.JP, bert_path)
        bert_models.load_model(Languages.JP, bert_path)

        # Load VITS
        base_model_dir = "model_assets/jvnv-F1-jp/jvnv-F1-jp"
        model = TTSModel(
            model_path=os.path.join(base_model_dir, "jvnv-F1-jp_e160_s14000.safetensors"),
            config_path=os.path.join(base_model_dir, "config.json"),
            style_vec_path=os.path.join(base_model_dir, "style_vectors.npy"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"Error loading model 'jvnv-F1-jp': {e}")
        print("Please ensure you have the model downloaded or specify a valid path.")
        print("You can download the model using: style-bert-vits2-install-model jvnv-F1-jp")
        exit(1)
    
    return model

def process_audio(raw_path, out_path):
    """
    1. Load with librosa (to check silence/duration).
    2. Trim silence.
    3. Check duration.
    4. Save as 16k mono 16-bit using ffmpeg for robustness or soundfile.
    """
    try:
        # Load and Trim
        y, sr = librosa.load(raw_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
        
        if duration < MIN_DURATION or duration > MAX_DURATION:
            return False, f"Duration {duration:.2f}s out of range"

        # Save trimmed to temp
        tmp_trimmed = raw_path + ".trimmed.wav"
        sf.write(tmp_trimmed, y_trimmed, sr)

        # Convert with FFMPEG to ensure 16kHz, mono, 16-bit PCM
        # ffmpeg -i input -ar 16000 -ac 1 -c:a pcm_s16le output -y
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", tmp_trimmed,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-c:a", "pcm_s16le",
            out_path
        ]
        subprocess.run(cmd, check=True)
        
        # Cleanup temp
        if os.path.exists(tmp_trimmed):
            os.remove(tmp_trimmed)
            
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    model = setup_model()
    
    # Get available styles
    # The internal structure might vary, but usually model.style2id exists
    styles = list(model.style2id.keys()) if hasattr(model, 'style2id') else ["Neutral"]
    print(f"Available Styles: {styles}")

    # Get available speakers
    speakers = list(model.spk2id.keys()) if hasattr(model, 'spk2id') else [0]
    print(f"Available Speakers: {speakers}")

    generated_count = 0
    
    # Check existing files to not overwrite if restarting (simple logic: count files)
    existing = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".wav")])
    if existing > 0:
        print(f"Found {existing} existing files. Appending...")
        generated_count = existing

    pbar = tqdm(total=TARGET_COUNT, initial=generated_count)

    while generated_count < TARGET_COUNT:
        text = random.choice(TEXTS)
        style = random.choice(styles)
        speed = random.choice(SPEEDS)
        speaker_id = random.choice(speakers) if isinstance(speakers[0], int) else model.spk2id[random.choice(speakers)]
        
        # Unique ID for temp file
        uid = f"{time.time()}_{random.randint(1000,9999)}"
        raw_path = os.path.join(TMP_DIR, f"raw_{uid}.wav")
        final_filename = f"kuya_yo_{generated_count + 1:06d}.wav"
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        
        try:
            # Generate
            sr, audio_data = model.infer(
                text=text,
                style=style,
                speaker_id=speaker_id,
                length=1.0 / speed,
                noise=0.6,
                noise_w=0.8,
                sdp_ratio=0.2
            )
            
            # Save Raw
            sf.write(raw_path, audio_data, sr)
            
            # Process
            success, msg = process_audio(raw_path, final_path)
            
            if success:
                generated_count += 1
                pbar.update(1)
            else:
                # print(f"Skipped: {msg}") # Optional: comment out to reduce noise
                pass
                
        except Exception as e:
            print(f"Generation Error: {e}")
            time.sleep(1)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)

    print("Generation Complete.")

if __name__ == "__main__":
    main()
