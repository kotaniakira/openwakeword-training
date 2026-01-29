
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Style-Bert-VITS2 の推論に必要な、
- JVNVモデル (jvnv-F1-jp) を ./model_assets に配置
- 日本語BERT (ku-nlp/deberta-v2-large-japanese-char-wwm) を ./bert_models にキャッシュ
するためのセットアップスクリプト。

使い方:
    python setup_sbv2_local.py

必要:
    pip install style-bert-vits2 huggingface_hub soundfile simpleaudio
    (GPU加速するなら PyTorch を環境に応じて別途インストール)
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download
from style_bert_vits2.nlp import bert_models
from style_bert_vits2.constants import Languages


def main():
    # === 1) 保存先のベースディレクトリを決定（カレントを基準） ===
    cwd = Path.cwd()
    assets_root = cwd / "model_assets"
    jvnv_root   = assets_root / "jvnv-F1-jp"
    bert_root   = cwd / "bert_models"  # ← BERTのキャッシュ保存先

    # ディレクトリ作成
    jvnv_root.mkdir(parents=True, exist_ok=True)
    bert_root.mkdir(parents=True, exist_ok=True)

    # === 2) BERTのキャッシュ保存先を「HF_HOME」で強制的に指定 ===
    # ※ transformers / huggingface_hub 共通でここに保存されます
    os.environ["HF_HOME"] = str(bert_root)

    # === 3) jvnv-F1-jp の3ファイルを Hugging Face から ./model_assets へDL ===
    repo_id = "litagin/style_bert_vits2_jvnv"
    files = [
        "jvnv-F1-jp/jvnv-F1-jp_e160_s14000.safetensors",
        "jvnv-F1-jp/config.json",
        "jvnv-F1-jp/style_vectors.npy",
    ]
    for f in files:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=f,
            local_dir=str(assets_root),  # ./model_assets に保存
        )
        print(f"[OK] downloaded: {local_path}")

    # === 4) 日本語BERTをロード（初回は自動ダウンロード → ./bert_models にキャッシュ） ===
    print("[INFO] Loading Japanese BERT (this may take a while on first run)...")
    bert_models.load_model(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    bert_models.load_tokenizer(Languages.JP, "ku-nlp/deberta-v2-large-japanese-char-wwm")
    print(f"[OK] BERT cached under: {bert_root}")

    # === 5) 完了メッセージとパス表示 ===
    print("\n=== Setup Complete ===")
    print(f"JVNV model files  : {jvnv_root}")
    print(f"BERT cache (HF_HOME): {bert_root}")


if __name__ == "__main__":
    main()

