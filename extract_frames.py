import cv2
import os
import glob

video_dir = "jaad_dataset/JAAD_clips"
frames_base_dir = "frames"

# Ensure the frames directory exists
os.makedirs(frames_base_dir, exist_ok=True)

# Find all videos in the folder
video_files = sorted(glob.glob(os.path.join(video_dir, "*.*")))  # Detects all formats

if not video_files:
    print("❌ No videos found in the 'videos/' folder!")
    exit()

print("🎥 Found video files:", video_files)

for video_path in video_files:
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(frames_base_dir, video_name)

    # Skip already extracted videos
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(f"⏭️ Skipping {video_name}, frames already extracted.")
        continue

    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if video ends

        frame_filename = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_id += 5  # Adjust frame step to avoid missing frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

    cap.release()
    print(f"✅ Extracted {frame_id} frames from {video_name}")

print("\n🎉 All videos processed successfully!")
