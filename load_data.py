import os
from torch.utils.data import Dataset
from PIL import Image


class VideoDataset(Dataset):
    def __init__(self, video_folder, transform=None):
        self.video_folder = video_folder
        self.transform = transform
        self.video_paths = []
        self.labels = []

        # Loop through the folders inside the video folder
        for seq_name in os.listdir(video_folder):
            seq_path = os.path.join(video_folder, seq_name)
            if os.path.isdir(seq_path):  # Ensure it's a directory
                label = int(seq_name.split('_')[-1])  # Assuming folder name is the label like 'video_0001'
                self.video_paths.append(seq_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self._load_frames(video_path)

        if self.transform:
            frames = [self.transform(frame) for frame in frames]  # Apply transform to each frame

        # Return frames and their label
        return frames, label

    def _load_frames(self, video_path):
        frames = []
        # Load frames from video_path
        for frame_name in sorted(os.listdir(video_path)):
            frame_path = os.path.join(video_path, frame_name)
            frame = Image.open(frame_path)  # Load image frame
            frames.append(frame)

        return frames
