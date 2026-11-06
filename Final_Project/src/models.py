"""A CIFAR-sized ResNet-18 with a reusable 512-dimensional representation."""
import torch
from torch import nn
from torchvision.models import resnet18


class CIFARResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = resnet18(weights=None, num_classes=10)
        self.network.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.network.maxpool = nn.Identity()

    def forward_features(self, x):
        net = self.network
        x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        x = net.layer4(net.layer3(net.layer2(net.layer1(x))))
        return torch.flatten(net.avgpool(x), 1)

    def forward(self, x, return_features=False):
        features = self.forward_features(x)
        logits = self.network.fc(features)
        return (logits, features) if return_features else logits
