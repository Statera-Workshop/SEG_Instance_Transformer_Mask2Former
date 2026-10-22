"""
[Unified] 2026-10-04: PreciseBN for models that declare `precise_bn = True`.

Why: every model is trained with micro batch 2, so BatchNorm running mean / var are a momentum-0.1 average of
2-image batch statistics. For the 18 YOLO models this does not matter (stored vs recomputed statistics differ
by -0.004 .. +0.014 mAP50 on FL36), but for RetinaNet (BN ResNet-50 followed by an un-normalised FPN and head)
the stored statistics gave test mAP50 0.148 and recomputed ones 0.849 for the same weights.

What: before an evaluation and before the final checkpoint is saved, every BN running mean / var is replaced by
a cumulative average over `precise_bn_images` training images (no gradient, no weight change). Same idea as
detectron2 hooks.PreciseBN (TEST.PRECISE_BN). Models without the flag are untouched.
"""

import time

import torch
import torch.nn as nn

from common.trainer import make_loader


def wants_precise_bn(model):
    return bool(getattr(model, "precise_bn", False))


@torch.no_grad()
def recompute_bn_stats(model, dataset, device, n_images=None, batch_size=None, workers=4, seed=0):
    n_images = n_images or getattr(model, "precise_bn_images", 1600)
    batch_size = batch_size or getattr(model, "precise_bn_batch_size", 16)
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    if not bns:
        return 0, 0
    t0 = time.time()
    was_training = model.training
    momenta = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                     # cumulative average over every batch seen
    model.train()
    seen = 0
    try:
        for _, images, _ in make_loader(dataset, batch_size, workers, True, seed):
            model(images.to(device))
            seen += images.shape[0]
            if seen >= n_images:
                break
    finally:
        for bn, mom in zip(bns, momenta):
            bn.momentum = mom
        model.train(was_training)
    print(f"PreciseBN    : {len(bns)} BN layers re-estimated on {seen} training images ({time.time() - t0:.0f} s)")
    return len(bns), seen
