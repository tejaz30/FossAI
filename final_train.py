import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import yaml
import os
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from resnet import resnet20

# Load best config
with open("best_config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Data Transformations
# Normalization values for CIFAR-10 dataset
mean = (0.4914, 0.4822, 0.4465)
std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Dataset Loading
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transform_train, download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    shuffle=True,
    num_workers=2
)

#Model Initialization
model = resnet20(dropout=config['dropout'])
model.fc = nn.Linear(model.fc.in_features, config['num_classes'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

lossfn = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config['step_size'],
    gamma=config['gamma']
)

#Training Loop
for epoch in range(config['epochs']):
    model.train()
    current_loss = 0
    correct_preds = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = lossfn(outputs, labels)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct_preds += predicted.eq(labels).sum().item()

    scheduler.step()
    accuracy = 100 * correct_preds / total
    print(f"Epoch [{epoch + 1}/{config['epochs']}], Loss: {current_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Saving the final model
checkpoint_path = 'resnet_cifar10/final_model.pth'
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)
print(f"[INFO] Final model saved to: {checkpoint_path}")
