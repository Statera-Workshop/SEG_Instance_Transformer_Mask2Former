"""
Shared Darknet (.cfg) network builder.

The three legacy v3 projects each carried their own copy of models/models.py.
The parser, the convolution/route/shortcut/upsample construction and the
forward pass were identical; only the [yolo] block differed. Here the common
part exists once, and each detector passes a `yolo_factory` that builds its
own detection layer (the algorithm core).

    yolo_factory(block: dict, hyperparams: dict) -> nn.Module
        module.forward(x, targets, img_size) -> (output, loss)

[Unified] BatchNorm momentum is fixed to the PyTorch default 0.1 for every
task (legacy Complex/YOLO3D used 0.9, Image used 0.1).
"""

import torch
import torch.nn as nn


def weights_init_normal(m):
    """Legacy Darknet-PyTorch initialisation, applied to every task."""
    # [Unified] type checks instead of class-name matching (YOLOv5 has a "Conv" wrapper block)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
