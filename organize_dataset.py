import os
import shutil

# Source folder containing LeapGestRecog data
SOURCE_ROOT = "leapGestRecog"
DEST_ROOT = "data"

# Mapping of original gesture folders → your custom media control classes
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

# Create destination folders
print("[INFO] Creating destination folders...")
for gesture_folder in gesture_map.values():
    os.makedirs(os.path.join(DEST_ROOT, gesture_folder), exist_ok=True)

# Loop through each subject folder (00 - 09)
subject_folders = [f for f in os.listdir(SOURCE_ROOT) if os.path.isdir(os.path.join(SOURCE_ROOT, f))]

print("[INFO] Organizing dataset...")
file_counter = 0

for subject in subject_folders:
    subject_path = os.path.join(SOURCE_ROOT, subject)

    for original_gesture, target_folder in gesture_map.items():
        source_path = os.path.join(subject_path, original_gesture)

        if os.path.exists(source_path):
            dest_path = os.path.join(DEST_ROOT, target_folder)

            for filename in os.listdir(source_path):
                src_file = os.path.join(source_path, filename)
                dst_file = os.path.join(dest_path, f"{subject}_{filename}")
                shutil.copyfile(src_file, dst_file)
                file_counter += 1

print(f" Dataset organized successfully! Total images copied: {file_counter}")
