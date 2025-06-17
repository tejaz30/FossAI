import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import torch.nn as nn
import yaml
import os

# Load config
with open("best_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform (same as training)
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 test set
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

# Load model and modify final layer
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, config["num_classes"])
model = model.to(device)

# Load trained weights
checkpoint_path = config["checkpoint_path"]
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Inference loop
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"[INFO] Inference Accuracy on CIFAR-10 Test Set: {accuracy:.2f}%")
