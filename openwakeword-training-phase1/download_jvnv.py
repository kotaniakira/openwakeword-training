import os
import sys
from huggingface_hub import snapshot_download

def main():
    # スクリプトのあるディレクトリを基準にする
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Downloading models to {BASE_DIR} ...")

    # 1. BERT Model
    bert_repo = "ku-nlp/deberta-v2-large-japanese-char-wwm"
    bert_dir = os.path.join(BASE_DIR, "bert_models", "deberta-v2-large-japanese-char-wwm")
    
    print(f"\n--- Downloading BERT Model: {bert_repo} ---")
    if not os.path.exists(bert_dir):
        snapshot_download(repo_id=bert_repo, local_dir=bert_dir)
        print("BERT Download complete.")
    else:
        print("BERT directory already exists. Skipping.")

    # 2. Style-Bert-VITS2 Model (JVNV)
    jvnv_repo = "litagin/style_bert_vits2_jvnv"
    jvnv_dir = os.path.join(BASE_DIR, "model_assets", "jvnv-F1-jp")
    
    print(f"\n--- Downloading TTS Model: {jvnv_repo} ---")
    if not os.path.exists(jvnv_dir):
        snapshot_download(repo_id=jvnv_repo, local_dir=jvnv_dir)
        print("JVNV Download complete.")
    else:
        print("JVNV directory already exists. Skipping.")

    print("\nAll models are ready!")

if __name__ == "__main__":
    # 仮想環境の pip で huggingface_hub が入っていない場合のエラーハンドリング
    try:
        main()
    except ImportError:
        print("Error: 'huggingface_hub' library not found.")
        print("Please run this script inside the virtual environment or install the package:")
        print("pip install huggingface_hub")