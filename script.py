from ultralytics import YOLO
import cv2
import mediapipe as mp
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Path to JAAD dataset
jaad_path = r"C:\Users\ASUS\PyCharmMiscProject\jaad_dataset\JAAD_clips"
video_files = [f for f in os.listdir(jaad_path) if f.endswith(".mp4")]

print(f"Processing {len(video_files)} videos...")

for video_file in video_files:
    video_path = os.path.join(jaad_path, video_file)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO on the frame
        results = model(frame)

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class ID
                if cls == 0:  # Class 0 is "person"
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box

                    # Extract pedestrian region
                    pedestrian_roi = frame[y1:y2, x1:x2]
                    pedestrian_roi_rgb = cv2.cvtColor(pedestrian_roi, cv2.COLOR_BGR2RGB)

                    # Apply pose estimation
                    result_pose = pose.process(pedestrian_roi_rgb)

                    if result_pose.pose_landmarks:
                        for lm in result_pose.pose_landmarks.landmark:
                            px, py = int(lm.x * pedestrian_roi.shape[1]), int(lm.y * pedestrian_roi.shape[0])
                            cv2.circle(frame, (x1 + px, y1 + py), 5, (0, 255, 0), -1)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.imshow("YOLO + Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
