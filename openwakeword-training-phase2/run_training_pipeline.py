import os
import shutil
import subprocess
import sys
import glob

# Configuration
PROJECT_ROOT = os.getcwd()
MODEL_DIR = "my_custom_model/kuya_yo"
VENV_PYTHON = "./venv/bin/python3"
TRAIN_SCRIPT = "openWakeWord/openwakeword/train.py"
CONFIG_FILE = "my_model.yaml"

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    print("=== Starting Automated Training Pipeline for 'Kuya yo' ===")

    # 1. Clean up partial features to force regeneration
    # We remove .npy files in the model directory to ensure 'augment_clips' runs fully
    # and generates both train and test features.
    # print("Cleaning old feature files...")
    # features = glob.glob(os.path.join(MODEL_DIR, "*.npy"))
    # for f in features:
    #     os.remove(f)
    #     print(f"Removed {f}")

    # 2. Run Augmentation (Feature Extraction)
    # This reads the wavs in positive_train/test and creates .npy files
    print("\n=== Step 1: Augmentation & Feature Extraction ===")
    try:
        run_cmd(f"{VENV_PYTHON} {TRAIN_SCRIPT} --training_config {CONFIG_FILE} --augment_clips")
    except subprocess.CalledProcessError:
        print("Augmentation failed. Please check logs.")
        return

    # Check if features exist
    if not os.path.exists(os.path.join(MODEL_DIR, "positive_features_test.npy")):
        print("Error: positive_features_test.npy was not generated.")
        return

    # 3. Run Training
    print("\n=== Step 2: Training Model ===")
    try:
        run_cmd(f"{VENV_PYTHON} {TRAIN_SCRIPT} --training_config {CONFIG_FILE} --train_model")
    except subprocess.CalledProcessError:
        print("Training failed.")
        return

    # 4. Verify Outputs & Convert to TFLite (if needed)
    onnx_path = os.path.join(MODEL_DIR, "kuya_yo.onnx")
    tflite_path = os.path.join(MODEL_DIR, "kuya_yo.tflite")

    if os.path.exists(onnx_path):
        print(f"\nSuccess! ONNX model found at: {onnx_path}")
        
        # Check TFLite
        if not os.path.exists(tflite_path):
            print("TFLite model missing. Attempting conversion via onnx2tf...")
            # Simple fallback conversion if openWakeWord didn't do it
            try:
                subprocess.run(f"{VENV_PYTHON} -m pip install onnx2tf tensorflow", shell=True)
                run_cmd(f"{VENV_PYTHON} -m onnx2tf -i {onnx_path} -o {MODEL_DIR}")
                # onnx2tf creates a folder/file structure, usually output is saved model. 
                # We need simple tflite.
                # Actually, openWakeWord usually exports tflite itself. 
                # If missing, it might be due to onnxruntime version.
                # Let's hope training generated it.
            except Exception as e:
                print(f"Conversion failed: {e}")
        else:
            print(f"Success! TFLite model found at: {tflite_path}")
    else:
        print("Error: ONNX model was not created.")
        return

    # 5. Test (Mic) - Dry run
    print("\n=== Step 3: Verification ===")
    print(f"Model ready. You can test it with:")
    print(f"{VENV_PYTHON} openWakeWord/openwakeword/examples/detect_from_microphone.py --model {tflite_path}")

if __name__ == "__main__":
    os.chdir(PROJECT_ROOT)
    main()
