import os
import librosa
import numpy as np
import glob

DATA_DIR = "positive_clips"

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    if not files:
        print("No files found.")
        return

    print(f"Analyzing {len(files)} files...")
    
    durations = []
    rms_values = []
    
    for f in files:
        y, sr = librosa.load(f, sr=None)
        durations.append(librosa.get_duration(y=y, sr=sr))
        rms_values.append(np.sqrt(np.mean(y**2)))
        
    durations = np.array(durations)
    rms_values = np.array(rms_values)
    
    print("-" * 30)
    print(f"Total Files: {len(files)}")
    print(f"Duration (s): Mean={durations.mean():.3f}, Min={durations.min():.3f}, Max={durations.max():.3f}")
    print(f"RMS Amplitude: Mean={rms_values.mean():.3f}, Min={rms_values.min():.3f}, Max={rms_values.max():.3f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
