"""
ResNet-50 bottom-up network (C3, C4, C5) for RetinaNet.

Code : torchvision 0.20.1 torchvision/models/resnet.py (conv3x3, conv1x1, Bottleneck, ResNet trunk),
       extracted class by class; the classifier (avgpool, fc) is removed because the detector only
       uses res3 / res4 / res5 (detectron2 Base-RetinaNet.yaml: RESNETS.OUT_FEATURES).
Paper: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016), ResNet-50 = [3, 4, 6, 3].

[Unified] detectron2 trains RetinaNet from ImageNet weights with FrozenBatchNorm and frozen stem + res2
(FREEZE_AT = 2). This framework trains from scratch, so every layer and BatchNorm is trainable.
[Unified] torchvision places the stride-2 of a bottleneck in the 3x3 conv (ResNet v1.5); the MSRA weights
used by detectron2 put it in the 1x1 (STRIDE_IN_1X1). Without pretrained weights the torchvision layout
is kept as extracted.
"""

import torch.nn as nn

RESNET_DEPTH = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3]}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups,
                     bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """torchvision ResNet trunk without avgpool / fc. forward -> {"res3", "res4", "res5"}."""

    def __init__(self, layers, block=Bottleneck, norm_layer=None, dilate_last=False):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=dilate_last)   # DC5 when True
        self.out_channels = {"res3": 128 * block.expansion, "res4": 256 * block.expansion,
                             "res5": 512 * block.expansion}

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:                                   # torchvision ResNet._make_layer
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                        norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return {"res3": c3, "res4": c4, "res5": c5}


def build_resnet(depth=50, dilate_last=False):
    return ResNet(RESNET_DEPTH[depth], dilate_last=dilate_last)
