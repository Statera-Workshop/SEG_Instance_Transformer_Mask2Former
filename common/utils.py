"""Seed, device, log files and checkpoints - identical for every detector."""

import datetime
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    """DataLoader worker seeding (reproducible augmentation)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def get_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


_TEE = None


def log_only(text):
    """Write a line to the log file without printing it to the terminal."""
    if _TEE is not None:
        _TEE.file.write(text + "\n")


class Tee:
    """Mirror stdout into a log file."""

    def __init__(self, path):
        global _TEE
        _TEE = self
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(path, "a", buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def start_log(run_dir, script_name):
    path = Path(run_dir) / "logs" / f"{script_name}_{timestamp()}.log"
    Tee(path)
    print(f"Log file     : {path}")
    return path


def banner(title):
    print("=" * 64)
    print(f" {title}")
    print("=" * 64)


def save_checkpoint(path, model, meta, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path} (use --overwrite).")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), **meta}, path)
    print(f"Checkpoint   : {path}")


def load_checkpoint(path, model, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    print(f"Weights      : {path}")
    return ckpt


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved        : {path}")
