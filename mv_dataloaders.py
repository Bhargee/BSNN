import logging
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10, MNIST, SVHN, STL10
from torchvision import transforms
from torchvision import datasets

def cifar10(batch_size, num_workers=1):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010])
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010])
    ])

    trainset = CIFAR10(root='./CIFAR10_DATA', train=True, download=True,
                        transform=train_transform)
    testset = CIFAR10(root='./CIFAR10_DATA', train=False, download=True,
                        transform=test_transform)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader

