import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import yaml
import os
import numpy as np 
import matplotlib.pyplot as plt
import random
from resnet import resnet20

# Search space for random search
search_space = {
    'learning_rate': [1e-3, 5e-4, 1e-4],
    'batch_size': [32, 64, 128],
    'weight_decay': [1e-4, 5e-5, 1e-5],
    'step_size': [2, 3, 5],
    'gamma': [0.1, 0.5, 0.7]
}

# Function to sample one configuration
def sample_config(space):
    return {
        'learning_rate': random.choice(space['learning_rate']),
        'batch_size': random.choice(space['batch_size']),
        'weight_decay': random.choice(space['weight_decay']),
        'step_size': random.choice(space['step_size']),
        'gamma': random.choice(space['gamma']),
        'dropout': random.choice([0.3,0.5,0.7]),
        'epochs': 15,
        'num_classes': 10
    }

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor()
])
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

# Load CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train, download=True)
val_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)

k = 3
kf = KFold(n_splits=k, shuffle=True, random_state=42)
indices = np.arange(len(train_dataset))

n_trials = 5

best_trial = None
best_val = 0

for trial in range(n_trials):
    print(f"\n[INFO] Starting Trial {trial + 1}/{n_trials}")
    config = sample_config(search_space)  

    trial_dir = f"trials/trial_{trial + 1}"
    os.makedirs(trial_dir, exist_ok=True)

    # Save the used config
    with open(os.path.join(trial_dir, "config_used.yaml"), "w") as f:
        yaml.dump(config, f)

    all_val_accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n[INFO] Fold {fold + 1}/{k}")

        train_subset = Subset(train_dataset, train_idx.tolist())
        val_subset = Subset(val_dataset, val_idx.tolist())

        trainloader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
        valloader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

        model = resnet20(dropout=config['dropout'])
        model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

        train_loss_history = []
        train_acc_history = []
        val_acc_history = []

        for epoch in range(config['epochs']):
            model.train()
            current_loss = 0
            correct_preds = 0
            total = 0

            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                current_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct_preds += predicted.eq(labels).sum().item()

            epoch_loss = current_loss / len(trainloader)
            epoch_acc = 100. * correct_preds / total
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
            print(f"Epoch {epoch + 1}/{config['epochs']}, Train Acc: {epoch_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        all_val_accs.append(sum(val_acc_history) / len(val_acc_history))

        fold_name = f"fold{fold + 1}"
        fold_dir = os.path.join(trial_dir, fold_name)
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pth"))

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

    # Save average accuracy for trial
    avg_val_acc = sum(all_val_accs) / len(all_val_accs)
    with open(os.path.join(trial_dir, "val_score.txt"), "w") as f:
        f.write(f"Average Validation Accuracy: {avg_val_acc:.2f}%\n")



    # Check and update the best trial
    if avg_val_acc > best_val:
        best_val = avg_val_acc
        best_trial = config.copy()
        best_trial['trial_dir'] = trial_dir  # Optional: track where this config was run

# Save best config after all trials
if best_trial:
    with open("best_config.yaml", "w") as f:
        yaml.dump(best_trial, f)
    print(f"\n Best configuration saved to best_config.yaml with {best_val:.2f}% validation accuracy.")
