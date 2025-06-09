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

# Load config.yaml 
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)




# Data transformations  
transform_train = transforms.Compose([ # Compose builds a pipeline
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

# CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=config['batch_size'], shuffle=False, num_workers=2) 
#num_workers decides the number of subprocesses by the processes to load the dataset


# Model Loading 
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, config['num_classes']) 
#This is changing the architecture of the final fully connected layer of the neural network, 
# we need to this as the original resent model is trained on 1000 classes and we need to change it according to the cifaar-10 dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Loss Function - softmax
criterion = nn.CrossEntropyLoss() 

#Optimizer
optimizer = optim.Adam(params = model.parameters(), lr = config['learning_rate'], weight_decay=config['weight_decay']) 
#for sgd - the momentum doesnt really change as much as we learnt in the lectures
#for adam - we are also adding weight decay

#scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer= optimizer, step_size= config['step_size'],gamma = config['gama'])

#training portion

train_loss_history = []
train_acc_history = []

for epoch in range(config['epochs']):
    model.train()
    current_loss = 0
    correct_preds = 0
    total = 0

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device) # moving the data from out datafolder to the cpu or gpu for training
        optimizer.zero_grad() # rests the backpropogation values

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()
        _, predicted = outputs.max(1) # this is finding the highesh percentage prediction class for each of the example in the batch
        total += labels.size(0)
        correct_preds += predicted.eq(labels).sum().item()


    epoch_loss = current_loss / len(trainloader)
    epoch_acc = 100. * correct_preds / total
    train_loss_history.append(epoch_loss)
    train_acc_history.append(epoch_acc) 



# Saving model
checkpoint_dir = os.path.dirname(config['checkpoint_path'])

try:
    os.makedirs(checkpoint_dir, exist_ok=True)
except Exception as e:
    print(f"Failed to create directory '{checkpoint_dir}': {e}")



torch.save(model.state_dict(), config['checkpoint_path'])

# Plotting training loss and accuracy

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
