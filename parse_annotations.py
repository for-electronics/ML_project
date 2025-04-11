import os
import xml.etree.ElementTree as ET
import cv2


def parse_jaad_annotations(xml_file):
    """
    Parses JAAD XML annotation file and extracts pedestrian bounding boxes.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = []

    for track in root.findall("track"):
        if track.get("label") == "pedestrian":  # We only extract pedestrians
            ped_id = track.get("id")

            for box in track.findall("box"):
                frame = int(box.get("frame"))  # Frame number
                x1 = float(box.get("xtl"))  # Top-left X
                y1 = float(box.get("ytl"))  # Top-left Y
                x2 = float(box.get("xbr"))  # Bottom-right X
                y2 = float(box.get("ybr"))  # Bottom-right Y
                occlusion = int(box.get("occluded"))  # Occlusion level

                annotations.append({
                    "ped_id": ped_id,
                    "frame": frame,
                    "bbox": (x1, y1, x2, y2),
                    "occlusion": occlusion
                })

    return annotations
