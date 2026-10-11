"""Evaluate the stage-1 detector (mean per-image Dice) on a generated dataset.

Same metric as the autoresearch harness (``evaluate_dice``): the Dice score of each image,
averaged over the first ``n`` images of the dataset.

    uv run python stage1_linefilter/eval_detector.py --data data/val --n 256
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from csd import generate_dataset, load_dataset
from stage1_linefilter.detector import predict


def dice(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice score between two binary masks (1.0 when both are empty).

    Args:
        pred: Predicted mask.
        target: Target mask.

    Returns:
        ``2·|P∩T| / (|P| + |T|)``.
    """
    pred, target = pred > 0, target > 0
    total = pred.sum() + target.sum()
    return 1.0 if total == 0 else float(2 * (pred & target).sum() / total)


def ensure_dataset(data_dir: str | pathlib.Path, n: int, seed: int) -> pathlib.Path:
    """Generate ``data_dir`` with the organisers' generator unless it already holds ``n`` images.

    Args:
        data_dir: Dataset folder.
        n: Number of images.
        seed: Generator seed (tuning set: 1, validation: 2, test: 2026).

    Returns:
        The dataset folder.
    """
    data_dir = pathlib.Path(data_dir)
    if not (data_dir / "meta.json").exists():  # written last: absent if a run was interrupted
        generate_dataset(n=n, out_dir=data_dir, seed=seed, overwrite=True)
    return data_dir


def evaluate(data_dir: str | pathlib.Path, n: int | None = None) -> np.ndarray:
    """Per-image Dice of ``predict`` on the first ``n`` images of a dataset.

    Args:
        data_dir: Dataset folder (``images.npy``, ``masks.npy``, ...).
        n: Number of images to evaluate (all if ``None``).

    Returns:
        Array of per-image Dice scores.
    """
    ds = load_dataset(data_dir)
    images, masks = ds["images"][:n], ds["masks"][:n]
    return np.array([dice(predict(img), mask) for img, mask in zip(images, masks)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean per-image Dice of the detector.")
    parser.add_argument("--data", default="data/val", help="dataset folder")
    parser.add_argument("--n", type=int, default=None, help="first n images (default: all)")
    parser.add_argument(
        "--generate",
        nargs=2,
        type=int,
        metavar=("N", "SEED"),
        help="generate the dataset first if it is missing (e.g. --generate 2000 2026)",
    )
    args = parser.parse_args()

    if args.generate:
        ensure_dataset(args.data, *args.generate)
    scores = evaluate(args.data, args.n)
    print(f"dataset     : {args.data} ({len(scores)} images)")
    print(f"mean Dice   : {scores.mean():.6f}")
    print(f"median Dice : {np.median(scores):.6f}")
    print(f"5th pct     : {np.percentile(scores, 5):.6f}")


if __name__ == "__main__":
    main()
