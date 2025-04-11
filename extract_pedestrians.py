import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Paths (Modify as needed)
ANNOTATIONS_DIR = "jaad_dataset/JAAD-JAAD_2.0/annotations"  # Path to XML files
VIDEO_FRAMES_DIR = "frames"  # Extracted video frames
OUTPUT_DIR = "cropped_pedestrians"  # Where cropped images are stored

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pedestrians():
    for xml_file in tqdm(os.listdir(ANNOTATIONS_DIR)):
        if not xml_file.endswith(".xml"):
            continue

        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
        root = tree.getroot()

        video_id = xml_file.replace(".xml", "")  # Extract video name

        for track in root.findall("track"):
            label = track.attrib["label"]

            # Only process "pedestrian" (ignoring "ped" and "people")
            if label != "pedestrian":
                continue

            for box in track.findall("box"):
                frame = int(box.attrib["frame"])
                xtl, ytl, xbr, ybr = (
                    int(float(box.attrib["xtl"])),
                    int(float(box.attrib["ytl"])),
                    int(float(box.attrib["xbr"])),
                    int(float(box.attrib["ybr"])),
                )

                # Load corresponding video frame
                frame_path = os.path.join(VIDEO_FRAMES_DIR, f"{video_id}_{frame:05d}.jpg")
                if not os.path.exists(frame_path):
                    continue

                img = cv2.imread(frame_path)
                cropped = img[ytl:ybr, xtl:xbr]  # Crop pedestrian

                # Resize for MobileNet
                cropped = cv2.resize(cropped, (224, 224))

                # Save cropped image
                save_path = os.path.join(OUTPUT_DIR, f"{video_id}_{frame:05d}.jpg")
                cv2.imwrite(save_path, cropped)

    print(f"✅ Cropped pedestrian images saved to {OUTPUT_DIR}")


# Run extraction
extract_pedestrians()
