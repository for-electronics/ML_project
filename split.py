# import os
#
# dataset_path = "C:/Users/ASUS/PyCharmMiscProject/jaad_dataset/"
#
# # Check if the dataset structure is correct
# print("Checking dataset structure...\n")
#
# # Check JAAD clips
# jaad_clips_path = os.path.join(dataset_path, "JAAD_clips")
# print(f"JAAD Clips Found: {os.path.exists(jaad_clips_path)}")
# print(f"Number of videos: {len(os.listdir(jaad_clips_path)) if os.path.exists(jaad_clips_path) else 0}")
#
# # Check annotations
# annotations_path = os.path.join(dataset_path, "JAAD-JAAD_2.0")
# print(f"JAAD Annotations Found: {os.path.exists(annotations_path)}")
# print(f"Files inside JAAD-JAAD_2.0: {os.listdir(annotations_path) if os.path.exists(annotations_path) else []}")

import os

# Paths to predefined splits
train_split_path = "jaad_dataset/JAAD-JAAD_2.0/split_ids/all_videos/train.txt"
test_split_path = "jaad_dataset/JAAD-JAAD_2.0/split_ids/all_videos/test.txt"

# Read train and test video IDs
def read_split_file(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]

train_videos = read_split_file(train_split_path)
test_videos = read_split_file(test_split_path)

# Print summary
print(f"Training videos: {len(train_videos)}")
print(f"Testing videos: {len(test_videos)}")

# Example output check
print("Sample train video IDs:", train_videos[:5])
print("Sample test video IDs:", test_videos[:5])
