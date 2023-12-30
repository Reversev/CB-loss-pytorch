# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2023/11/30
# @author : ''
# @FileName: resnet.py
"""ResNet-18 Image classfication for cifar-10 with PyTorch
reference: https://github.com/richardaecn/class-balanced-loss
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, num_blocks=[2, 2, 2, 2], loss_type="focal"):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)

        if loss_type == "focal" or loss_type == "sigmoid":
            self._init_bias(value=-np.log(self.num_classes - 1))
        else:
            self._init_bias(value=0)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def _init_bias(self, value):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, value)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10, loss_type="focal"):
    return ResNet(ResidualBlock, num_classes=num_classes, loss_type=loss_type)


def ResNet32(num_classes=10, loss_type="focal"):
    return ResNet(ResidualBlock, num_classes=num_classes, num_blocks=[3, 4, 6, 3], loss_type=loss_type)


if __name__ == '__main__':
    from torchsummary import summary
    x = torch.randn((2, 3, 32, 32))
    model = ResNet32()
    summary(model, (3, 32, 32), device="cpu")
    print(model(x).shape)
