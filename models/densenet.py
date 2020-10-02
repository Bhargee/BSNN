from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.densenet as det_densenet

import layers as L


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, device):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features))
        self.add_module('conv1', L.Conv2d(num_input_features, bn_size * 
                                 growth_rate, 1, device, True, stride=1,
                                 bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('conv2', L.Conv2d(bn_size*growth_rate, growth_rate, 3,
                                device, True, stride=1, padding=1, bias=False))
    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, device):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features+i*growth_rate, growth_rate,
                    bn_size, device)
            self.add_module(f'denselayer{i+1}', layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, device):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('conv', L.Conv2d(num_input_features,
            num_output_features, 1, device, True, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module): #TODO make num_layers a param, currently=10
    def __init__(self, device, growth_rate=32, block_config=(6,12,24,16),
                 num_init_features=64, bn_size=4):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', L.Conv2d(3, num_init_features, 7, device, True, stride=2,
                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate,
                                device)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features//2, device)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2

        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, 10, bias=False)
        self.classifier.requires_grad = False

        for m in self.modules():
            if isinstance(m, L.Conv2d):
                nn.init.kaiming_normal_(m.inner.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1,1)).view(features.size(0), -1)
        return self.classifier(out)


def densenet121(stoch, device=None):
    if stoch:
        return DenseNet(device) # the right arguments are already defaults
    else:
        return det_densenet.densenet121(num_classes=10)


def densenet161(stoch, device=None):
    if stoch:
        return DenseNet(device, 48, (6, 12, 36, 24), 96)
    else:
        return det_densenet.densenet161(num_classes=10)

