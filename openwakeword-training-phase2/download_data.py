import os
import requests
from tqdm import tqdm

FILES = {
    "openwakeword_features_ACAV100M_2000_hrs_16bit.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "validation_set_features.npy": "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy"
}

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping.")
        return
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()

if __name__ == "__main__":
    for name, url in FILES.items():
        download_file(url, name)
    
    # Create empty augmentation directories to prevent immediate crashes
    os.makedirs("audioset_16k", exist_ok=True)
    os.makedirs("fma", exist_ok=True)
    print("Directories created.")
