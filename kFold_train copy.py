import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import yaml
import os
import numpy as np 
import matplotlib.pyplot as plt
from resnet import resnet20

with open("config.yaml","r") as file:
    config = yaml.safe_load(file)

transform = transforms.Compose([
    transforms.ToTensor()
    
])

#Loading the dataset
dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform=transform,download=True)

k = 5
kf = KFold(n_splits=k, shuffle=True,random_state=42)

indices = np.arange(len(dataset))
all_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n[INFO] Fold {fold+1}/5")

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist() # because Subset expectd a 1-d array and the pixels are 2D
    
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    trainloader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    #Model Loading
    model = resnet20()
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params=model.parameters(), lr = config['learning_rate'], weight_decay = config['weight_decay'])

    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size=config['step_size'],gamma=config['gamma'])

    train_loss_history = []
    train_acc_history=[]

    for eopch in range(config['epochs']):
        model.train()
        current_loss = 0
        correct_preds=0
        total=0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            _,predicted = outputs.max(1)
            total += labels.size(0)
            correct_preds += predicted.eq(labels).sum().item()

        epoch_loss = current_loss/len(trainloader)
        epoch_acc = 100. * correct_preds/total
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        scheduler.step()
    
    fold_name = f"fold{fold+1}"
    fold_dir = f"trials/{fold_name}"
    os.makedirs(fold_dir, exist_ok=True)

    torch.save(model.state_dict(),os.path.join(fold_dir, "model.pth"))

    #Plotting training loss and accuracy

    # Get all existing curve files
    existing_files = os.listdir('visualizations')
    count = sum(1 for f in existing_files if f.startswith("training_curve_") and f.endswith(".png"))

    # Create new filename with count
    filename = f"visualizations/training_curve_run{count+1}.png"


    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(epochs)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.xticks(epochs)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"[INFO] Saved training curve to: {filename}")



