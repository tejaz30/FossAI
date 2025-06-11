import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_shortcut=True):
        super().__init__()
        self.use_shortcut = use_shortcut

        # First conv layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match dimensions incase of downsampling 
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=1, stride=stride),
                nn.ConstantPad3d((0, 0, 0, 0, 0, out_channels - in_channels), 0)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        shortcut = self.shortcut(x) if self.use_shortcut else x

        out += shortcut
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, use_shortcuts=True, dropout = 0.5):
        super().__init__()
        self.in_channels = 16
        self.use_shortcut = use_shortcuts

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Create stacks
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1) #defined next it mainly focuses on the hyperparameter definitions such as strides and layers
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s,
                                use_shortcut=self.use_shortcut))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

                #ask what fan_out does and the whole working of the intialize weights functioin in general

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out) #avg pooling the results instead of max pooling as the images of are not that huge to rule out some part of the image completely
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)

def resnet20(dropout = 0.5):
    # ResNet-20 for CIFAR-10 has 3 layers of 3 blocks each (total 20 layers)
    return ResNet(BasicBlock, [3, 3, 3])

