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

with open("best_config.yaml","r") as file:
    config = yaml.safe_load(file)

transform = transforms.Compose([
    transforms.ToTensor()
    
])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
    ])

#Loading the dataset
train_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform=transform_train,download=True)
val_dataset = torchvision.datasets.CIFAR10(root = './data', train = True, transform=transform,download=True)

k = 3
kf = KFold(n_splits=k, shuffle=True,random_state=42)

indices = np.arange(len(train_dataset))
all_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
    print(f"\n[INFO] Fold {fold+1}/3")

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist() 

    #These were initially defined as numpy arrays which held the indices of each datapoint which were split into training ad validation for that fold
    #No we are converting them to a Python list because that is the datatype expected by the Subset function


    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    trainloader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
    valloader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    #Model Loading
    model = resnet20()
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(params=model.parameters(), lr = config['learning_rate'], weight_decay = config['weight_decay'],)

    scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size=config['step_size'],gamma=config['gamma'])

    train_loss_history = []
    train_acc_history=[]
    val_acc_history = []

    for epoch in range(config['epochs']):
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

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        val_acc_history.append(val_acc)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    fold_name = f"fold{fold+1}"
    fold_dir = f"trials/2_try/{fold_name}"
    os.makedirs(fold_dir, exist_ok=True)

    torch.save(model.state_dict(),os.path.join(fold_dir, "model.pth"))




    #Plotting training loss and validation accuracy

    # Get all existing curve files
    # existing_files = os.listdir('graphs')
    # count = sum(1 for f in existing_files if f.startswith("training_curve_") and f.endswith(".png"))

    # Create new filename with count


    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_history, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.xticks(epochs)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_acc_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.xticks(epochs)


    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc_history, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.xticks(epochs)

    plt.tight_layout()
    graph_path = os.path.join(fold_dir, f"training_curve.png")
    plt.savefig(graph_path)
    print(f"[INFO] Saved training curve to: {graph_path}")



