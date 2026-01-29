import os
import sys
import shutil
import subprocess
import glob
import wave
import contextlib
import requests
import argparse
from tqdm import tqdm

# ================= 設定 ================
# このスクリプトは phase2-model-train/main.py として配置される前提
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_ROOT = os.path.join(PROJECT_ROOT, "datasets")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# Phase 2 専用のフォルダと環境
VENV_DIR = os.path.join(BASE_DIR, "venv")
VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python3")
REQUIREMENTS_FILE = os.path.join(BASE_DIR, "requirements.txt")

OPENWAKEWORD_REPO = "https://github.com/dscripka/openWakeWord.git"

# 必須リソース
RESOURCES = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "validation_set_features.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
}
ONNX_MODELS = {
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx"
}

# ================= ヘルパー関数 ================

def run_cmd(cmd, cwd=None, check=True):
    print(f"Running: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=check, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def setup_environment():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    # 1. venv作成
    if not os.path.exists(VENV_DIR):
        print(f"Creating Phase 2 virtual environment in {VENV_DIR}...")
        run_cmd(f"{sys.executable} -m venv {VENV_DIR}")
        
        # 2. 依存関係インストール
        print("Installing dependencies for Phase 2...")
        pip = os.path.join(VENV_DIR, "bin", "pip")
        run_cmd(f"{pip} install -U pip wheel setuptools")
        
        # requirements.txt からインストール
        # (PyTorch 2.1.2 指定済み)
        run_cmd(f"{pip} install -r {REQUIREMENTS_FILE}")

    # 3. openWakeWord クローン & インストール
    repo_dir = os.path.join(BASE_DIR, "openWakeWord")
    if not os.path.exists(repo_dir):
        run_cmd(f"git clone {OPENWAKEWORD_REPO}", cwd=BASE_DIR)
        
    # venv内に openWakeWord をインストール (editable)
    pip = os.path.join(VENV_DIR, "bin", "pip")
    run_cmd(f"{pip} install -e ./openWakeWord", cwd=BASE_DIR)

def download_file(url, path):
    if os.path.exists(path):
        return
    print(f"Downloading: {os.path.basename(path)}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as bar:
        for data in response.iter_content(1024):
            f.write(data)
            bar.update(len(data))

def prepare_resources():
    for name, url in RESOURCES.items():
        download_file(url, os.path.join(BASE_DIR, name))
    
    # openWakeWordのリソースディレクトリ特定
    try:
        site_packages = subprocess.check_output(
            f"{VENV_PYTHON} -c 'import site; print(site.getsitepackages()[0])'", shell=True
        ).decode().strip()
        target_res_dir = os.path.join(site_packages, "openwakeword", "resources", "models")
        os.makedirs(target_res_dir, exist_ok=True)
        
        # また、リポジトリ内のパスも必要になる場合があるため両方に配置
        repo_res_dir = os.path.join(BASE_DIR, "openWakeWord", "openwakeword", "resources", "models")
        os.makedirs(repo_res_dir, exist_ok=True)
        
        for name, url in ONNX_MODELS.items():
            # ダウンロード先
            dl_path = os.path.join(BASE_DIR, name)
            download_file(url, dl_path)
            
            # site-packages へコピー
            shutil.copy(dl_path, os.path.join(target_res_dir, name))
            # リポジトリ内へコピー
            shutil.copy(dl_path, os.path.join(repo_res_dir, name))
            
    except Exception as e:
        print(f"Warning setting up resources: {e}")

    os.makedirs(os.path.join(BASE_DIR, "audioset_16k"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "fma"), exist_ok=True)

def create_dummy_wav(path, duration=1.0):
    with contextlib.closing(wave.open(path, 'w')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        n_frames = int(duration * 16000)
        wf.writeframes(b'\x00' * n_frames * 2)

def prepare_training_data(model_name):
    src_dir = os.path.join(DATASET_ROOT, model_name, "positive_clips")
    if not os.path.exists(src_dir):
        print(f"Error: Dataset for '{model_name}' not found at {src_dir}")
        print("Run Phase 1 script first.")
        sys.exit(1)

    print(f"Structuring training data for {model_name}...")
    
    base_model_dir = os.path.join(BASE_DIR, "my_custom_model", model_name)
    if os.path.exists(base_model_dir):
        shutil.rmtree(base_model_dir) 
    
    train_dir = os.path.join(base_model_dir, "positive_train")
    test_dir = os.path.join(base_model_dir, "positive_test")
    neg_train = os.path.join(base_model_dir, "negative_train")
    neg_test = os.path.join(base_model_dir, "negative_test")
    
    for d in [train_dir, test_dir, neg_train, neg_test]:
        os.makedirs(d, exist_ok=True)

    src_files = glob.glob(os.path.join(src_dir, "*.wav"))
    if not src_files:
        print("Error: No wav files found in dataset.")
        sys.exit(1)
    
    import random
    random.shuffle(src_files)
    n_total = len(src_files)
    n_val = min(300, int(n_total * 0.1)) 
    
    test_files = src_files[:n_val]
    train_files = src_files[n_val:]
    
    print(f"Train: {len(train_files)}, Test: {len(test_files)}")
    
    for f in train_files:
        shutil.copy(f, train_dir)
    for f in test_files:
        shutil.copy(f, test_dir)
        
    create_dummy_wav(os.path.join(neg_train, "dummy.wav"))
    create_dummy_wav(os.path.join(neg_test, "dummy.wav") )
    
    piper_dummy = os.path.join(BASE_DIR, "piper-sample-generator")
    os.makedirs(piper_dummy, exist_ok=True)
    with open(os.path.join(piper_dummy, "generate_samples.py"), "w") as f:
        f.write("def generate_samples(*args, **kwargs): pass\n")
        
    return len(train_files) + len(test_files), n_val

def create_config(model_name, n_total, n_val):
    config_path = os.path.join(BASE_DIR, f"{model_name}.yaml")
    
    # 絶対パスで指定することでディレクトリが変わっても安心
    abs_base = os.path.abspath(BASE_DIR)
    
    yaml_content = f"""
model_name: "{model_name}"
target_phrase: ["{model_name}"] 
n_samples: {n_total}
n_samples_val: {n_val}
steps: 10000
output_dir: "{os.path.join(abs_base, 'my_custom_model')}"
background_paths: ["{os.path.join(abs_base, 'audioset_16k')}", "{os.path.join(abs_base, 'fma')}"]
background_paths_duplication_rate: [1, 1]
rir_paths: []
false_positive_validation_data_path: "{os.path.join(abs_base, 'validation_set_features.npy')}"
feature_data_files:
  ACAV100M_sample: "{os.path.join(abs_base, 'openwakeword_features_ACAV100M_2000_hrs_16bit.npy')}"
batch_n_per_class:
  ACAV100M_sample: 1024
  adversarial_negative: 50
  positive: 50
model_type: "dnn"
layer_size: 32
max_negative_weight: 1500
target_false_positives_per_hour: 0.2
augmentation_batch_size: 16
augmentation_rounds: 1
tts_batch_size: 1
piper_sample_generator_path: "{os.path.join(abs_base, 'piper-sample-generator')}"
    """
    with open(config_path, "w") as f:
        f.write(yaml_content)
    return config_path

def train_logic(model_name):
    prepare_resources()
    n_total, n_val = prepare_training_data(model_name)
    config_path = create_config(model_name, n_total, n_val)
    config_filename = os.path.basename(config_path)
    
    print("\n--- Running Augmentation ---")
    run_cmd(f"{VENV_PYTHON} openWakeWord/openwakeword/train.py --training_config {config_filename} --augment_clips", cwd=BASE_DIR)
    
    print("\n--- Running Training ---")
    onnx_path = os.path.join(BASE_DIR, "my_custom_model", model_name + ".onnx")
    
    # 常に学習実行
    try:
        run_cmd(f"{VENV_PYTHON} openWakeWord/openwakeword/train.py --training_config {config_filename} --train_model", cwd=BASE_DIR, check=False)
    except Exception as e:
        print(f"Warning: Training script had an issue: {e}")

    # TFLite Conversion & Save
    tflite_out_dir = os.path.join(BASE_DIR, "my_custom_model", f"{model_name}_tflite_tmp")
    final_model_dir = os.path.join(MODELS_ROOT, model_name)
    os.makedirs(final_model_dir, exist_ok=True)
    
    if os.path.exists(onnx_path):
        print("\n--- Saving & Converting Models ---")
        shutil.copy(onnx_path, os.path.join(final_model_dir, f"{model_name}.onnx"))
        print(f"ONNX model saved to: {os.path.join(final_model_dir, f'{model_name}.onnx')}")
        
        try:
            print("Converting ONNX to TFLite using onnx2tf...")
            run_cmd(f"{VENV_DIR}/bin/onnx2tf -i {onnx_path} -o {tflite_out_dir} --non_verbose", cwd=BASE_DIR)
            
            gen_tflite = os.path.join(tflite_out_dir, f"{model_name}_float32.tflite")
            if not os.path.exists(gen_tflite):
                possible_files = glob.glob(os.path.join(tflite_out_dir, "*.tflite"))
                if possible_files:
                    gen_tflite = possible_files[0]

            if os.path.exists(gen_tflite):
                shutil.copy(gen_tflite, os.path.join(final_model_dir, f"{model_name}.tflite"))
                print(f"\nSUCCESS! Models saved to: {final_model_dir}")
                print(f"Final TFLite model: {os.path.join(final_model_dir, f'{model_name}.tflite')}")
            else:
                print("Warning: TFLite file not found after conversion.")
        except Exception as e:
            print(f"TFLite conversion failed: {e}")
    else:
        print(f"Error: Training finished but ONNX not found at {onnx_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Train Wake Word Model")
    parser.add_argument("model_name", type=str, help="Model name (e.g. kuya_yo)")
    parser.add_argument("--internal_run", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    
    model_name = args.model_name
    
    print(f"=== Phase 2: Training Model '{model_name}' ===")
    
    # 1. 環境チェックと再実行
    if sys.prefix != VENV_DIR and not args.internal_run:
        setup_environment()
        print("Re-launching script inside Phase 2 virtual environment...")
        cmd = [VENV_PYTHON, __file__, model_name, "--internal_run"]
        subprocess.check_call(cmd)
        sys.exit(0)

    # 2. ロジック実行
    train_logic(model_name)

if __name__ == "__main__":
    main()
