import os
import xml.etree.ElementTree as ET
from PIL import Image
import glob

# --- Configuration ---
annotation_dir = "jaad_dataset/JAAD-JAAD_2.0/annotations"  # Directory containing multiple XML files
frames_base_dir = "frames"  # Base directory containing frames for all videos
output_base_dir = "cropped_pedestrians"  # Base directory for cropped images

# Create output base directory if it doesn't exist
os.makedirs(output_base_dir, exist_ok=True)

# Get all annotation files (assumes one XML per video)
annotation_files = glob.glob(os.path.join(annotation_dir, "*.xml"))

for annotation_file in annotation_files:
    video_name = os.path.splitext(os.path.basename(annotation_file))[0]  # Extract video ID
    img_dir = os.path.join(frames_base_dir, video_name)  # Frames folder for this video
    output_dir = os.path.join(output_base_dir, video_name)  # Output folder for this video

    if not os.path.exists(img_dir):
        print(f"Skipping {video_name}, frames not found!")
        continue  # Skip if frames don't exist

    os.makedirs(output_dir, exist_ok=True)

    # Parse XML file
    tree = ET.parse(annotation_file)
    root = tree.getroot()

    # --- Extract Pedestrian Bounding Boxes ---
    bboxes = []
    frame_indices = []

    for track in root.findall("track"):
        if track.get("label") == "pedestrian":  # Only get pedestrian labels
            for box in track.findall("box"):
                x1, y1, x2, y2 = float(box.get("xtl")), float(box.get("ytl")), float(box.get("xbr")), float(
                    box.get("ybr"))
                frame_id = int(box.get("frame"))  # Get frame index from annotation
                bboxes.append((frame_id, x1, y1, x2, y2))
                frame_indices.append(frame_id)

    print(f"[{video_name}] Total pedestrian bounding boxes found: {len(bboxes)}")

    # Get available frames for this video
    available_frames = sorted(
        [int(f.split("_")[1].split(".")[0]) for f in os.listdir(img_dir) if f.startswith("frame_")])
    print(f"[{video_name}] Available frames: {available_frames[:10]}...")

    # --- Crop Pedestrians from Frames ---
    for i, (frame_id, x1, y1, x2, y2) in enumerate(bboxes):
        # Find the nearest available frame
        closest_frame = min(available_frames, key=lambda x: abs(x - frame_id))

        frame_path = os.path.join(img_dir, f"frame_{closest_frame:04d}.jpg")

        if not os.path.exists(frame_path):
            print(f"[{video_name}] Frame not found: {frame_path}")
            continue  # Skip missing frames

        # Open frame and crop pedestrian
        img = Image.open(frame_path)
        cropped_img = img.crop((x1, y1, x2, y2))

        # Save cropped pedestrian image
        cropped_img.save(os.path.join(output_dir, f"ped_{i:04d}.jpg"))

    print(f"[{video_name}] Cropping complete! Check {output_dir} for results.")

print("All videos processed successfully!")
