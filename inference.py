import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
import torch.nn as nn
import yaml
import os
from sklearn.metrics import f1_score
from resnet import resnet20

# Load config
with open("best_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transform (same as training)
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load CIFAR-10 test set
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=2)

# Load model and modify final layer
model = resnet20(dropout=config['dropout'])
model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
model = model.to(device)


# Load trained weights
checkpoint_path = 'resnet_cifar10/final_model.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")


model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Inference loop
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
f1 = f1_score(all_labels, all_preds, average='macro')

print(f"[INFO] Inference Accuracy on CIFAR-10 Test Set: {accuracy:.2f}%")
print(f"[INFO] Macro F1 Score on CIFAR-10 Test Set: {f1:.4f}")
