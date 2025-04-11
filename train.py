import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Ensure the main function is called in a safe way for Windows
if __name__ == '__main__':
    # Set device to GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transformation for data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 to fit MobileNet V2 input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Same normalization as ImageNet
    ])

    # Load custom dataset (frames folder) from directory structure
    # Assuming the frames folder is structured with class subfolders inside
    train_dataset = datasets.ImageFolder(root='./frames/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='./frames/test', transform=transform)

    # Create DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load the MobileNet V2 model from torchvision and use the weights enum (recommended in latest versions)
    model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).to(device)

    # Modify the final fully connected layer to match the number of classes in your dataset
    # Ensure the number of classes in the dataset matches the output of this layer
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes)).to(device)

    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero out gradients from previous step

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss

            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights

            running_loss += loss.item()
            if batch_idx % 100 == 99:  # Print every 100 batches
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    # Evaluation on test set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # No need to compute gradients during evaluation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
