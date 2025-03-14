import torch
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection on a sample video
results = model("sample_video.mp4", save=True, conf=0.4)
