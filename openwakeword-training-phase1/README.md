# Wakeword Generation: "Kuya yo"

This folder contains scripts to generate a dataset for the wake word "空也よ" using Style-Bert-VITS2 (JP-Extra).

## Prerequisites
- Python 3.10+
- FFmpeg (installed and on system PATH)
- NVIDIA GPU (Recommended)

## Setup

1. **Create and Activate Virtual Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the JP-Extra Model**
   The script relies on the `jvnv-F1-jp` model. Download it using the library's CLI:
   ```bash
   style-bert-vits2-install-model jvnv-F1-jp
   ```
   *Note: If this command fails, please check the Style-Bert-VITS2 documentation for manual model download instructions.*

## Usage

1. **Generate Clips**
   Run the generator script. It will generate files until it hits the target (default 3000).
   ```bash
   python generate_clips.py
   ```
   - Output: `positive_clips/`
   - Temporary Raw files: `tmp_raw/` (cleaned up automatically)

2. **Verify Statistics**
   Check the distribution of the generated files.
   ```bash
   python stats.py
   ```

## Configuration
Edit `generate_clips.py` to change:
- `TARGET_COUNT`: Number of files to generate (currently 3000).
- `TEXTS`: The list of text prompts.
- `SPEEDS`: Speed variations.
