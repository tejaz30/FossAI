# 🧠 Residual Learning with ResNet-18 on CIFAR-10

This project demonstrates the training and evaluation of a **ResNet-18** model on the **CIFAR-10** dataset using PyTorch, focusing on **residual learning**, **model improvement through hyperparameter tuning**, and **training visualization**. It includes multiple experiments using learning rate scheduling, weight decay, and K-Fold Cross Validation.

---

## 📌 Project Overview

### ✅ What is Residual Learning?

Traditional deep networks often suffer from the **vanishing gradient problem**, making it hard to train deeper models effectively. **Residual learning**, introduced in the [ResNet paper (He et al., 2015)](https://arxiv.org/abs/1512.03385), solves this by introducing **skip connections**, allowing the network to learn residuals instead of full transformations. This increases the relevance of the data presented to the deeper layers of the network aswell as the gradient value presented to each weight and by extension to the changes to the neuron. 

**Without residuals:**
> H(x) = desired mapping

**With residuals (ResNet):**
> H(x) = F(x) + x  
> → The network learns the residual: F(x) = H(x) - x

This architecture allows training of very deep networks (50, 101, 152+ layers) without degradation.

---

### 🧪 Experiments Conducted

| Experiment              | Optimizer | Weight Decay | LR Scheduler | Accuracy ↑ | Notes                               |
|-------------------------|-----------|--------------|--------------|------------|-------------------------------------|
| `exp1_no_weight_decay`  | SGD       | ❌           | ❌           | ~58%       | Baseline                            |
| `exp2_with_wd_scheduler`| Adam      | ✅           | StepLR       | ~65%       | Improved generalization             |
| `exp3_kfold`            | Adam      | ✅           | StepLR       | ~75% (avg across folds) | Better model stability |
| 'exp4_random_search`    | Adam      | ✅           | StepLR       | ~78%       | Hyperparameter Tuning              |

📊 Training accuracy and loss curves are saved after each run in `/trials/1_try`.

📂 Checkpoints for trained models are saved in `/checkpoints`.

---

## 🔧 Project Structure
<Pre>

RESNET_CIFAR10/
│
├── checkpoints/                  # Final trained model weights
├── data/                         # CIFAR-10 dataset storage
├── graphs/                       # (Optional) Graph assets for reports
├── Intermediate_models/          # Models & configs saved during training
│   └── trial-1/
│       ├── config.yaml
│       ├── train.py
│       └── kFold_train_18.py
│
├── models/                       # Finalized or best models
├── results/                      # Metrics, logs, or analysis outputs
├── trials/                       # Organized experiment runs
│   ├── 1_try/
│   │   ├── trial_1/
│   │   ├── ...
│   │   └── trial_5/
│   └── 2_try/
│
├── inference.py                  # Model inference on test images
├── kFold_train_20.py             # K-Fold cross-validation training script
├── random_search_train.py        # Hyperparameter tuning using random search
├── resnet.py                     # Custom ResNet wrapper if used
├── best_config.yaml              # Best config found after tuning
├── requirements.txt              # Python dependencies
├── .gitignore                    # Files/folders excluded from Git
└── README.md                     # Project documentation
</Pre>



---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/yourusername/resnet-cifar10.git
cd resnet-cifar10
```
### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```
## Training the Model
To start training with current config:

```bash
python train.py
```
You can change the training configuration (batch size, learning rate, etc.) via config.yaml.

---

## Hyperparameter Tuning + K-Fold Cross Validation
Implemented features:

 Optimizer comparison (SGD vs Adam)

 Weight Decay (L2 regularization)

 Learning Rate Scheduling (StepLR)

 K-Fold Cross Validation (e.g., K=5) with average performance tracking

 ---

 ## Learning and Objectives

Residual learning helps deeper models converge faster and generalize better as the learning process for a particular neuron is made easier.

Weight decay reduced overfitting and improved test accuracy.

StepLR scheduler smoothed training and helped escape local minima.

K-Fold cross-validation gave more reliable performance evaluation.

Implementing Cross-validation further removes the problem of overfitting aswell as really generalizes the model over all combinations of data.

Deeper Networks would perform better but the most important is Hyperprarameter tuning.

Experiment tracking and visualization made model behavior transparent and reproducible.

---

## Author
Teja Bulusu
Learning Deep Neural Networks – 2025


