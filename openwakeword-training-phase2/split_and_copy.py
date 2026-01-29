import os
import glob
import shutil
import random

# Config
SRC_DIR = "positive_clips_raw"
BASE_OUT = "my_custom_model/kuya_yo"
TRAIN_DIR = os.path.join(BASE_OUT, "positive_train")
TEST_DIR = os.path.join(BASE_OUT, "positive_test")
VAL_COUNT = 300

def main():
    # Create directories
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Get files
    files = glob.glob(os.path.join(SRC_DIR, "*.wav"))
    random.shuffle(files)
    
    total = len(files)
    print(f"Total files: {total}")
    
    if total < VAL_COUNT:
        print("Error: Not enough files.")
        return

    # Split
    test_files = files[:VAL_COUNT]
    train_files = files[VAL_COUNT:]
    
    print(f"Copying {len(train_files)} to training...")
    for f in train_files:
        shutil.copy(f, TRAIN_DIR)
        
    print(f"Copying {len(test_files)} to testing...")
    for f in test_files:
        shutil.copy(f, TEST_DIR)
        
    print("Done.")

if __name__ == "__main__":
    main()
