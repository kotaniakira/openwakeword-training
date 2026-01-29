import os
import sys
import subprocess
import argparse
import random
import time
import shutil
from tqdm import tqdm

# ================= 設定 =================
# このスクリプトは phase1-audio-gen/main.py として配置される前提
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
VENV_DIR = os.path.join(BASE_DIR, "venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")

DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
SAMPLE_RATE = 16000
TARGET_COUNT = 3000
SPEEDS = [0.85, 1.0, 1.15]
# ========================================

def run_cmd(cmd, cwd=None, check=True):
    # print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=check, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def setup_environment():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # 1. venv作成
    if not os.path.exists(VENV_DIR):
        print(f"Creating Phase 1 virtual environment in {VENV_DIR}...")
        run_cmd(f"{sys.executable} -m venv {VENV_DIR}")
        
        # 2. 依存関係インストール
        print("Installing dependencies for Phase 1...")
        pip = os.path.join(VENV_DIR, "bin", "pip")
        run_cmd(f"{pip} install -U pip wheel setuptools")
        
        # PyTorch (CUDA 12.1)
        run_cmd(f"{pip} install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        # その他のライブラリ
        run_cmd(f"{pip} install -r {REQUIREMENTS_FILE}")

def process_audio(raw_path, out_path):
    """無音削除、長さチェック、16kHzモノラル変換"""
    import librosa
    import soundfile as sf
    try:
        y, sr = librosa.load(raw_path, sr=None)
        y_trimmed, _ = librosa.effects.trim(y, top_db=30)
        duration = librosa.get_duration(y=y_trimmed, sr=sr)
        
        if duration < 0.5 or duration > 2.5:
            return False
        
        # 16kHz, Mono, 16bit保存
        y_resampled = librosa.resample(y_trimmed, orig_sr=sr, target_sr=SAMPLE_RATE)
        sf.write(out_path, y_resampled, SAMPLE_RATE, subtype='PCM_16')
        return True
    except Exception as e:
        return False

def generate_audio(model_name, phrases, target_count):
    import torch
    import soundfile as sf
    from style_bert_vits2.tts_model import TTSModel
    from style_bert_vits2.nlp import bert_models
    from style_bert_vits2.constants import Languages
    from huggingface_hub import snapshot_download

    # 出力ディレクトリ設定
    output_dir = os.path.join(DATASET_ROOT, model_name, "positive_clips")
    raw_dir = os.path.join(DATASET_ROOT, model_name, "tmp_raw")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    print(f"Model Name: {model_name}")
    print(f"Phrases: {phrases}")
    print(f"Target Count: {target_count}")

    # モデルロード
    model_assets_dir = os.path.join(PROJECT_ROOT, "model_assets", "jvnv-F1-jp")
    bert_dir = os.path.join(PROJECT_ROOT, "bert_models", "deberta-v2-large-japanese-char-wwm")
    
    try:
        # BERTモデルの準備
        if not os.path.exists(bert_dir):
            print(f"Downloading BERT model to {bert_dir}...")
            snapshot_download(repo_id="ku-nlp/deberta-v2-large-japanese-char-wwm", local_dir=bert_dir)
        
        # BERTロード
        print(f"Loading BERT from {bert_dir}...")
        bert_models.load_tokenizer(Languages.JP, bert_dir)
        bert_models.load_model(Languages.JP, bert_dir)

        print("Loading TTS Model (JVNV-F1-JP)...")
        if not os.path.exists(model_assets_dir):
            print(f"Downloading model to {model_assets_dir}...")
            snapshot_download(repo_id="litagin/style_bert_vits2_jvnv", local_dir=model_assets_dir)
            
        # パスの特定
        config_path = None
        model_file = None
        style_vec = None
        
        for root, dirs, files in os.walk(model_assets_dir):
            for f in files:
                if f == "config.json":
                    config_path = os.path.join(root, f)
                elif f.endswith(".safetensors") and "jvnv-F1-jp" in f:
                    model_file = os.path.join(root, f)
                elif f == "style_vectors.npy":
                    style_vec = os.path.join(root, f)
        
        if not (config_path and model_file and style_vec):
            print("Error: Model files missing even after download.")
            return

        model = TTSModel(
            model_path=model_file,
            config_path=config_path,
            style_vec_path=style_vec,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"Model load failed: {e}")
        return

    styles = list(model.style2id.keys())
    
    current_count = len([f for f in os.listdir(output_dir) if f.endswith(".wav")])
    if current_count >= target_count:
        print(f"Target count already reached ({current_count}). Skipping generation.")
        return

    print(f"Generating {target_count - current_count} clips...")
    pbar = tqdm(total=target_count, initial=current_count)
    
    while current_count < target_count:
        text = random.choice(phrases)
        style = random.choice(styles)
        speed = random.choice(SPEEDS)
        
        tmp_path = os.path.join(raw_dir, f"temp_{time.time()}_{random.randint(0,9999)}.wav")
        final_filename = f"{model_name}_{current_count+1:06d}.wav"
        final_path = os.path.join(output_dir, final_filename)

        try:
            sr, audio_data = model.infer(
                text=text,
                style=style,
                length=1.0/speed,
                noise=0.6,
                noise_w=0.8,
                sdp_ratio=0.2
            )
            sf.write(tmp_path, audio_data, sr)
            
            if process_audio(tmp_path, final_path):
                current_count += 1
                pbar.update(1)
            
        except Exception as e:
            time.sleep(0.1)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    try:
        shutil.rmtree(raw_dir)
    except:
        pass

    print(f"\nPhase 1 Complete. Data saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate wake word audio clips.")
    parser.add_argument("model_name", type=str, help="Name of the model (e.g., 'kuya_yo')")
    parser.add_argument("phrases", type=str, help="Comma-separated list of phrases.")
    parser.add_argument("--count", type=int, default=TARGET_COUNT, help="Number of clips to generate")
    
    # 内部再実行かどうかを判定する隠し引数
    parser.add_argument("--internal_run", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # 1. 現在のPythonが専用venvでない場合、環境構築して再実行
    if sys.prefix != VENV_DIR and not args.internal_run:
        setup_environment()
        
        print("Re-launching script inside Phase 1 virtual environment...")
        # 引数をそのまま渡して再実行
        cmd = [VENV_PYTHON, __file__, args.model_name, args.phrases, "--count", str(args.count), "--internal_run"]
        subprocess.check_call(cmd)
        sys.exit(0)

    # 2. 専用venvの中で実行されるロジック
    phrases_list = [p.strip() for p in args.phrases.split(",") if p.strip()]
    if not phrases_list:
        print("Error: No phrases provided.")
        sys.exit(1)
        
    generate_audio(args.model_name, phrases_list, args.count)

if __name__ == "__main__":
    main()