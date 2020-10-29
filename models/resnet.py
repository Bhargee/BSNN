from math import inf

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as deterministic_resnet
import sys

import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, device, stochastic, stride=1):
        super(BasicBlock, self).__init__()
        self.stochastic = stochastic
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        if not self.stochastic:
            out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        if not self.stochastic:
            out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_labels, device, stochastic):
        super(ResNet, self).__init__()
        self.stochastic = stochastic
        self.device = device
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.fc = nn.Linear(64, num_labels)

        self.apply(_weights_init)

        if stochastic:
            torch.nn.init.orthogonal_(self.fc.weight)
            self.fc.weight.requires_grad = False


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, device=self.device, stochastic=self.stochastic, stride=stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if not self.stochastic:
            x = F.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnet18(stochastic, num_labels, device):
    return ResNet(BasicBlock, [3, 3, 3], num_labels, device, stochastic)


def resnet34(stochastic, num_labels, device):
    return ResNet(BasicBlock, [5, 5, 5], num_labels, device, stochastic)


def resnet50(stochastic, num_labels, device):
    return ResNet(BasicBlock, [9, 9, 9], num_labels, device, stochastic)


def resnet101(stochastic, num_labels, device):
    return ResNet(BasicBlock, [18, 18, 18], num_labels, device, stochastic)
