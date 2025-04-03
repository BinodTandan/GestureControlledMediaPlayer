import os
import shutil
import random

# Paths
SOURCE_ROOT = "leapGestRecog"
DEST_ROOT = "data"
TRAIN_SPLIT = 0.8  # 80% for training

# Gesture folder mapping
gesture_map = {
    "01_palm": "pause",
    "03_fist": "play",
    "05_thumb": "volume_up",
    "08_palm_moved": "volume_down",
    "06_index": "next",
    "09_c": "previous",
    "07_ok": "mute",
    "02_l": "stop"
}

# Create destination subfolders
for split in ["train", "test"]:
    for gesture_folder in gesture_map.values():
        os.makedirs(os.path.join(DEST_ROOT, split, gesture_folder), exist_ok=True)

print("[INFO] Organizing dataset with train/test split...")
file_counter = 0

# Loop through subjects
subject_folders = [f for f in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, f))]

for original_gesture, target_folder in gesture_map.items():
    all_images = []

    # Collect all image paths for this gesture across subjects
    for subject in subject_folders:
        source_path = os.path.join(SOURCE_ROOT, subject, original_gesture)

        if os.path.exists(source_path):
            for filename in os.listdir(source_path):
                src_file = os.path.join(source_path, filename)
                all_images.append((src_file, f"{subject}_{filename}"))

    # Shuffle and split
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_SPLIT)
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    # Copy to destination
    for split_name, split_data in [("train", train_images), ("test", test_images)]:
        for src_file, dst_name in split_data:
            dst_folder = os.path.join(DEST_ROOT, split_name, target_folder)
            dst_file = os.path.join(dst_folder, dst_name)
            shutil.copyfile(src_file, dst_file)
            file_counter += 1

print(f"[DONE] Dataset organized with train/test split. Total images copied: {file_counter}")
