import torch
import torch.nn as nn
import torchvision.models.vgg as deterministic_vgg

import layers as L


class VGG(nn.Module):
    def __init__(self, features, device, orthogonal, num_labels):
        super(VGG, self).__init__()
        self.features = features
        self.device = device
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            L.Linear(512*7*7, 4096, device, True, orthogonal),
            L.Linear(4096, 4096, device, True, orthogonal),
            nn.Linear(4096, num_labels, bias=False)
        )
        self._init_weights()


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, L.Conv2d):
                nn.init.kaiming_normal_(m.inner.weight, mode='fan_out', nonlinearity='relu')
                if m.inner.bias is not None:
                    nn.init.constant_(m.inner.bias, 0)
                nn.init.constant_(m.norm.weight, 1)
                nn.init.constant_(m.norm.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, L.Linear):
                nn.init.normal_(m.inner.weight, 0, 0.01)
                if m.inner.bias is not None:
                    nn.init.constant_(m.inner.bias, 0)
                nn.init.constant_(m.norm.weight, 1)
                nn.init.constant_(m.norm.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)


def _make_layers(cfg, device):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = L.Conv2d(in_channels, v, 3, device, True, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 
          512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
          'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(cfg, device, orthogonal, num_labels):
    return VGG(_make_layers(cfgs[cfg], device), device, orthogonal, num_labels)


def vgg16(stochastic, num_labels, device, orthogonal=False):
    if stochastic:
        return _vgg('D', device, orthogonal, num_labels)
    else:
        return deterministic_vgg.vgg16_bn(num_classes=num_labels)
