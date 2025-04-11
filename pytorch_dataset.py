import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Define transformations (resize, normalize for MobileNet)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class PedestrianDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = sorted(os.listdir(img_dir))  # Ensure consistent order
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image  # No labels yet (unsupervised or self-supervised)

# Test dataset
if __name__ == "__main__":
    dataset = PedestrianDataset("cropped_pedestrians/")
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
