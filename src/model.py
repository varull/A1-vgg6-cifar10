"""
model.py: VGG6 network (Q2a).

"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

cfg_vgg6 = [64, 64, "M", 128, 128, "M"]


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Linear(128, num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=True, activation="relu"):
    act = ACTIVATIONS[activation]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), act()]
            else:
                layers += [conv2d, act()]
            in_channels = v
    return nn.Sequential(*layers)


def vgg6(activation="relu", num_classes=10, batch_norm=True):
    return VGG(make_layers(cfg_vgg6, batch_norm, activation), num_classes)


if __name__ == "__main__":
    x = torch.randn(8, 3, 32, 32)
    for name in ACTIVATIONS:
        model = vgg6(name)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:<8} output {tuple(out.shape)}  parameters {n_params:,}")