"""Minimal stand-ins for detr/util/misc.py (single process, fixed-size batches without padding masks)."""
from typing import Optional

import torch
from torch import Tensor


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors, self.mask = tensors, mask

    def decompose(self):
        return self.tensors, self.mask


def interpolate(*args, **kwargs):
    return torch.nn.functional.interpolate(*args, **kwargs)
