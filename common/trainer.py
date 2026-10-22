"""
Common training loop for every detector.

The model contract is the only requirement:
    loss, raw = model(images, targets)
    model.yolo_layers[i].metrics  (optional, for logging)

Shared behaviour (identical for all tasks):
    optimizer / learning rate / schedule   -> common/config.py PROFILE
    micro batch + gradient accumulation     -> same effective batch (8)
    non-finite loss / gradient guard         -> skip the step (legacy Image v3)
    per-epoch log + `last` checkpoint
"""

import inspect
import math
import time
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from common.progress import Progress
from common.utils import seed_worker


def build_optimizer(model, recipe):
    name = recipe["optimizer"].lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=recipe["lr"], momentum=recipe.get("momentum", 0.9),
                               weight_decay=recipe.get("weight_decay", 5e-4))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=recipe["lr"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=recipe["lr"], weight_decay=recipe.get("weight_decay", 5e-4))
    raise ValueError(f"Unknown optimizer {recipe['optimizer']}")


def build_scheduler(optimizer, recipe):
    s = recipe.get("lr_schedule", "constant")
    if s == "constant":
        return None
    if s == "step10x0.8":  # legacy Complex/YOLO3D StepLR(10, 0.8), stepped per epoch
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    raise ValueError(f"Unknown lr schedule {s}")


def make_loader(dataset, batch_size, workers, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      collate_fn=dataset.collate_fn, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=seed_worker, generator=g, drop_last=False)


def _grads_finite(model):
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return False
    return True


def _targets_to(targets, device):
    """[Unified] 2026-10-11: detection targets are one float tensor [N, 6]; segmentation targets are a dict
    {"boxes": [N, 6] float, "masks": [N, S, S] uint8} (common/datasets/voc_seg.py). Masks stay uint8."""
    if isinstance(targets, dict):
        return {k: v.to(device, dtype=torch.float32 if k == "boxes" else v.dtype, non_blocking=True)
                for k, v in targets.items()}
    return targets.to(device, dtype=torch.float32, non_blocking=True)


def train(model, dataset, recipe, args, device, on_epoch_end=None):
    loader = make_loader(dataset, args.batch_size, args.workers, True, args.seed)
    optimizer = build_optimizer(model, recipe)
    scheduler = build_scheduler(optimizer, recipe)
    accum = max(1, args.accum)
    n_batches = len(loader)
    if n_batches == 0:
        raise RuntimeError("Training loader is empty.")

    print(f"Train images : {len(dataset)}")
    print(f"Batches/epoch: {n_batches}  (micro batch {args.batch_size} x accum {accum} "
          f"= effective {args.batch_size * accum})")
    print(f"Optimizer    : {recipe['optimizer']}  lr={recipe['lr']}  schedule={recipe.get('lr_schedule')}")
    print()

    history = []
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "set_epoch"):   # [Unified] epoch-dependent algorithms (YOLOv6 ATSS warm-up,
            if len(inspect.signature(model.set_epoch).parameters) >= 2:   # YOLO26 ProgLoss decay)
                model.set_epoch(epoch - 1, args.epochs)   # 0-based epoch, total epochs
            else:
                model.set_epoch(epoch - 1)
        t0 = time.time()
        total, valid, skipped = 0.0, 0, 0
        head_stats, head_cnt = defaultdict(float), defaultdict(int)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        bar = Progress(n_batches, f"{args.run_name} | epoch {epoch:02d}/{args.epochs}",
                       log_every=args.log_interval, extra_total=(args.epochs - epoch) * n_batches)
        for bi, (_, images, targets) in enumerate(loader, 1):
            images = images.to(device, dtype=torch.float32, non_blocking=True)
            targets = _targets_to(targets, device)
            loss, _ = model(images, targets)
            if not torch.isfinite(loss):
                skipped += 1
                print(f"  WARNING: non-finite loss at batch {bi}, skipped")
                continue
            (loss / accum).backward()
            pending += 1
            if pending == accum or bi == n_batches:
                if _grads_finite(model):
                    if getattr(model, "clip_max_norm", 0):   # [Unified] 2026-10-08: optional, declared by the model
                        torch.nn.utils.clip_grad_norm_(model.parameters(), model.clip_max_norm)   # (DETR recipe)
                    optimizer.step()
                else:
                    skipped += 1
                    print(f"  WARNING: non-finite gradient at batch {bi}, step skipped")
                optimizer.zero_grad(set_to_none=True)
                pending = 0

            lv = float(loss.detach())
            total += lv
            valid += 1
            # [Unified] logging fix: a scale without positives reports conf_obj = cls_acc = 0; those
            # layers are left out of the conf_obj / cls_acc means (they were averaged in before, which
            # understated both for every multi-scale model).
            for layer in getattr(model, "yolo_layers", []):
                has_pos = layer.metrics.get("conf_obj") not in (None, 0.0)
                for k in ("conf_obj", "conf_noobj", "cls_acc"):
                    v = layer.metrics.get(k)
                    if v is None or not math.isfinite(v) or (k != "conf_noobj" and not has_pos):
                        continue
                    head_stats[k] += v
                    head_cnt[k] += 1

            bar.update(loss=f"{total / max(valid, 1):.3f}")
        bar.close()

        if scheduler is not None:
            scheduler.step()
        avg = total / max(valid, 1)
        stats = {k: v / max(head_cnt[k], 1) for k, v in head_stats.items()}
        rec = {"epoch": epoch, "loss": avg, "time_sec": round(time.time() - t0, 1),
               "lr": optimizer.param_groups[0]["lr"], "skipped": skipped, **stats}
        history.append(rec)
        print(f"Epoch {epoch:02d} done  loss={avg:.4f}  time={rec['time_sec']}s  "
              + "  ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        elapsed = time.time() - t_train
        eta_h = elapsed / epoch * (args.epochs - epoch) / 3600
        print(f"Elapsed {elapsed / 3600:.2f} h  |  ETA for remaining {args.epochs - epoch} epoch(s): {eta_h:.2f} h")
        if on_epoch_end:
            on_epoch_end(epoch, history)
        print()

    total_min = (time.time() - t_train) / 60
    print(f"Training time: {total_min:.1f} min")
    if torch.cuda.is_available():
        print(f"Peak VRAM    : {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    return history
