"""
Common entry logic for the four scripts of every detector:

    train_<task>_<ver>.py            train 10 epochs, then evaluate on test
    evaluate_<task>_<ver>.py         evaluate on the test split only

Each entry script is three lines: it only names the task, the version and the
model builder. Arguments, paths, logs and outputs are therefore identical.
"""

import argparse
from pathlib import Path

from common import config as C
from common.darknet import weights_init_normal
from common.precise_bn import recompute_bn_stats, wants_precise_bn
from common.trainer import train
from common.progress import queue_tag
from common.utils import (banner, get_device, load_checkpoint, save_checkpoint, save_json, set_seed,
                          start_log, timestamp)


def get_task(name):
    from tasks.seg_task import SegTask
    return SegTask()


def parse_args(mode, task, version):
    spec = C.TASKS[task]
    P = C.PROFILE
    micro = spec["micro_batch"]
    p = argparse.ArgumentParser(description=f"{mode} {task} {version}")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--seed", type=int, default=P["seed"])
    p.add_argument("--img-size", type=int, default=spec["img_size"])
    p.add_argument("--batch-size", type=int, default=micro)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--weights", default=None, help="checkpoint (default: runs/<task>_<ver>/checkpoints/...)")
    p.add_argument("--max-test", type=int, default=None, help="limit test samples (smoke test)")
    if mode == "train":
        p.add_argument("--epochs", type=int, default=P["epochs"])
        p.add_argument("--accum", type=int, default=max(1, P["effective_batch"] // micro))
        p.add_argument("--recipe", choices=["controlled", "reference"], default="controlled",
                       help="controlled = common profile (Mode A); reference = legacy optimizer (Mode B)")
        p.add_argument("--lr", type=float, default=None, help="override learning rate")
        p.add_argument("--pretrained", default=None, help="Darknet backbone weights, e.g. darknet53.conv.74")
        p.add_argument("--max-train", type=int, default=None, help="limit train samples (smoke test)")
        p.add_argument("--log-interval", type=int, default=100)
        p.add_argument("--overwrite", action="store_true")
        p.add_argument("--skip-eval", action="store_true", help="do not evaluate on test after training")
        p.add_argument("--eval-interval", type=int, default=5,
                       help="also evaluate on the test split every N epochs (convergence curve; 0 = off)")
    args = p.parse_args()
    args.max_train = getattr(args, "max_train", None)
    args.tag = version
    args.run_name = f"{task.upper()} {version}"
    return args


def default_weights(task, version):
    """Profile checkpoint (e010) if present, else the newest final checkpoint (e.g. e008)."""
    path = C.checkpoint_path(task, version, C.PROFILE["epochs"])
    if path.exists():
        return path
    found = sorted(path.parent.glob(f"{task}_{version}_*_e[0-9][0-9][0-9].pth"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else path


def main(mode, task, version, build_model):
    args = parse_args(mode, task, version)
    run = C.run_dir(task, version)
    start_log(run, f"{mode}_{task}_{version}")
    banner(f"{queue_tag()}{task.upper()} {version}  |  {mode}")
    set_seed(args.seed)
    device = get_device(args.device)
    t = get_task(task)
    print(f"Device       : {device}")
    print(f"Profile      : {C.PROFILE['name']}  seed={args.seed}  img_size={args.img_size}")

    model = build_model(task, len(t.class_names)).to(device)
    print(f"Parameters   : {sum(p.numel() for p in model.parameters()):,}")

    if mode == "train":
        model.apply(weights_init_normal)
        if args.pretrained:
            model.load_darknet_weights(args.pretrained)
            print(f"Pretrained   : {args.pretrained}")
        recipe = dict(C.PROFILE) if args.recipe == "controlled" else {**C.PROFILE, **C.REFERENCE_RECIPE[task]}
        if args.lr is not None:
            recipe["lr"] = args.lr
        print(f"Recipe       : {args.recipe}")
        ckpt = C.checkpoint_path(task, version, args.epochs)
        if ckpt.exists() and not args.overwrite:
            raise FileExistsError(f"{ckpt} exists (use --overwrite).")
        train_set = t.train_set(args)
        meta = {"task": task, "version": version, "img_size": args.img_size, "classes": t.class_names,
                "recipe": recipe, "seed": args.seed, "dataset": C.TASKS[task]["dataset"]}

        eval_curve = []                       # [Unified] optional convergence curve (--eval-interval)
        test_set = t.test_set(args) if args.eval_interval > 0 else None

        def on_epoch_end(epoch, history):
            save_checkpoint(ckpt.with_name(ckpt.stem.rsplit("_e", 1)[0] + "_last.pth"), model,
                            {**meta, "epoch": epoch, "history": history}, overwrite=True)
            if test_set is not None and epoch % args.eval_interval == 0 and epoch < args.epochs:
                if wants_precise_bn(model):    # [Unified] see common/precise_bn.py
                    recompute_bn_stats(model, train_set, device, workers=args.workers, seed=args.seed + epoch)
                res = t.evaluate(model, test_set, args, device)
                point = {"epoch": epoch, **{k: v["summary"]["mAP"] for k, v in res.items()
                                            if isinstance(v, dict) and "summary" in v}}
                eval_curve.append(point)
                print("Eval curve   : " + "  ".join(f"{k}={v:.4f}" if k != "epoch" else f"epoch {v}"
                                                   for k, v in point.items()))
                save_json(run / "eval" / "eval_curve.json", eval_curve)
                model.train()

        history = train(model, train_set, recipe, args, device, on_epoch_end)
        if wants_precise_bn(model):            # [Unified] final weights are saved with recomputed BN statistics
            recompute_bn_stats(model, train_set, device, workers=args.workers, seed=args.seed + args.epochs)
        save_checkpoint(ckpt, model, {**meta, "epoch": args.epochs, "history": history}, overwrite=args.overwrite)
        if args.skip_eval:
            return
        banner(f"{task.upper()} {version}  |  test evaluation after training")
        result = t.evaluate(model, t.test_set(args), args, device)
        if eval_curve:
            eval_curve.append({"epoch": args.epochs, **{k: v["summary"]["mAP"] for k, v in result.items()
                                                        if isinstance(v, dict) and "summary" in v}})
        save_json(run / "eval" / f"test_after_train_{timestamp()}.json",
                  {"checkpoint": str(ckpt), "train_history": history, "eval_curve": eval_curve, **result})
        return

    weights = Path(args.weights) if args.weights else default_weights(task, version)
    load_checkpoint(weights, model, device)
    model.eval()
    if mode == "evaluate":
        result = t.evaluate(model, t.test_set(args), args, device)
        save_json(run / "eval" / f"test_{timestamp()}.json", {"checkpoint": str(weights), **result})
