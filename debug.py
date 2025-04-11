import os

# Path where all video frame folders are stored
base_path = "frames"  # Update this with your actual folder path

total_frames = 0

# Loop through each folder (each video)
for video_folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, video_folder)

    # Ensure it's a directory
    if os.path.isdir(folder_path):
        # Count image files inside the folder
        num_frames = len([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))])
        total_frames += num_frames
        print(f"📂 {video_folder}: {num_frames} frames")

print(f"\n🎯 Total frames across all videos: {total_frames}")
