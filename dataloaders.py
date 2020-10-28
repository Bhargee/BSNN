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


SPLIT_FILE = 'splits.json'


class _CIFAR10C(Dataset):
    def __init__(self, root='./CIFAR-10-C'):
        self.root = root
        self.cgroups = [f for f in os.listdir(self.root) if f != 'labels.npy']
        self.samples_per_cgroup = 50000

    def __len__(self):
        return self.samples_per_cgroup*len(self.cgroups)

    def __getitem__(self, i):
        cgroup_i = i // self.samples_per_cgroup
        if torch.is_tensor(i):
            cgroup = torch.from_numpy(np.load([self.cgroups[ci] for ci in cgroup_i]))
            im = cgroup[i - (self.samples_per_cgroup * cgroup_i), :, :, :]
            im = im.permute(0,3,1,2)
        else:
            p = os.path.join(self.root, self.cgroups[cgroup_i])
            cgroup = torch.from_numpy(np.load(p))
            im = cgroup[i - (self.samples_per_cgroup * cgroup_i), :, :, :]
            im = im.permute(2,0,1)

        labels = torch.from_numpy(np.load(os.path.join(self.root, 'labels.npy')))
        l = labels[i % self.samples_per_cgroup]
        return im, l
        

def _get_train_val_samplers(dataset, trainset, split=[.9,.1]):
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE, 'r') as fp:
            indexes = json.load(fp)
            if dataset in indexes.keys():
                s = indexes[dataset]
                val_inds, train_inds = s['val'], s['train']
                if len(val_inds)/len(trainset) == split[1]:
                    train_sampler = SubsetRandomSampler(train_inds)
                    val_sampler = SubsetRandomSampler(val_inds)
                    return train_sampler, val_sampler
                else:
                    logging.info('Recomputing train and val indexes')

    else:
        indexes = {}

    num_train_samples = len(trainset)
    indices = list(range(num_train_samples))
    random.shuffle(indices)
    num_val = int((split[1]*num_train_samples))
    val_inds, train_inds = indices[:num_val], indices[num_val:]
    indexes[dataset] = {'val': val_inds, 'train': train_inds}
    with open(SPLIT_FILE, 'w') as fp:
        json.dump(indexes, fp)
    train_sampler = SubsetRandomSampler(train_inds)
    val_sampler = SubsetRandomSampler(val_inds)

    return train_sampler, val_sampler


def mnist(resize=False, batch_size=1, num_workers=1):
    mnist_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]
    if resize:
        mnist_transforms = [transforms.Resize(32)] + mnist_transforms
    mnist_transforms = transforms.Compose(mnist_transforms)

    mnist_train = MNIST('/scratch/bsm92/MNIST_DATA/', train=True, transform=mnist_transforms, download=True)
    mnist_test = MNIST('/scratch/bsm92/MNIST_DATA/', train = False,transform=mnist_transforms)
    train_sampler, val_sampler = _get_train_val_samplers('mnist', mnist_train)

    train_loader = DataLoader(mnist_train, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
    val_loader = DataLoader(mnist_train, batch_size=batch_size,
            sampler=val_sampler, num_workers=num_workers)
    test_loader = DataLoader(mnist_test,  batch_size=batch_size, shuffle=True,
            num_workers=num_workers)

    return train_loader, val_loader, test_loader


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

    #train_sampler, val_sampler = _get_train_val_samplers('cifar10', trainset)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #valloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=val_sampler)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def cifar10c(batch_size, num_workers=1):
    root = './CIFAR-10-C'
    lf = 'labels.npy'
    assert os.path.exists(root)
    assert os.path.exists(os.path.join(root, lf))
    assert len(os.listdir(root)) > 1
    
    testset = _CIFAR10C()
    trainloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers)
    return trainloader



def svhn(batch_size, num_workers=1):
    ts = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.4377, .4438, .4782],
                          std=[.1282, .1315, .1123])])
    trainset = SVHN('/scratch/bsm92/SVHN', transform=ts, download=True)
    testset = SVHN('/scratch/bsm92/SVHN', split='test', download=True, transform=ts)
    train_sampler, val_sampler = _get_train_val_samplers('svhn', trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size,
            sampler=train_sampler, num_workers=num_workers)
    valloader = DataLoader(trainset, batch_size=batch_size,
            sampler=val_sampler, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)

    return trainloader, valloader, testloader


def tinyimagenet(batch_size, num_workers=5):
    num_workers=0
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataroot = '/scratch/bsm92/tiny-imagenet-200'
    train_val_dataset_dir = os.path.join(dataroot, "train")
    test_dataset_dir = os.path.join(dataroot, "val1")
    trainset = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train)
    valset = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test)
    testset  = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)
    index_path = os.path.join(dataroot, 'tinyimagenet_indices.pth')
    if os.path.exists(index_path):
        indices = torch.load(index_path)
        train_indices = indices[:len(indices) - 10000]
        valid_indices = indices[len(indices) - 10000:] 
    else:
        indices = list(range(200*500))
        random.shuffle(indices)
        train_indices = indices[:len(indices) - 10000]
        valid_indices = indices[len(indices) - 10000:] 
        torch.save(indices, index_path)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(train_indices),
                                           num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                           sampler=SubsetRandomSampler(valid_indices),
                                           num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, drop_last=False,
                                         num_workers=num_workers)
    return trainloader, valloader, testloader
