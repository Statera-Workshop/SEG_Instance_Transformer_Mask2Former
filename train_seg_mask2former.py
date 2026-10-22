"""Train seg Mask2Former, then evaluate on the test split."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.mask2former.build import build_model

if __name__ == "__main__":
    main("train", task="seg", version="mask2former", build_model=build_model)
