"""Score the line-filter detector on the robustness suite of ``stage1_unet``.

Same 200 clean test scenes, same 25 perturbed sets (9 artefact families, rebuilt bit for bit
from their seeds by ``detection.robustness.perturb``) and same object F1 as the U-Net results,
so the three detectors are compared on identical images with an identical metric. The line
filter has no threshold to select: its output mask is scored as is.

    uv run python stage1_linefilter/robustness.py   # ~10 min on one CPU core
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage1_unet"))

from detection.metrics import object_scores
from detection.robustness import LEVELS, perturb, set_name

from stage1_linefilter.detector import predict

SUITE = ROOT / "stage1_unet" / "robustness_suite"
OUT = ROOT / "stage1_linefilter" / "results" / "robustness_linefilter.json"


def load_clean() -> tuple[np.ndarray, list[list[dict]]]:
    """The 200 clean test scenes and their sticks."""
    z = np.load(SUITE / "clean.npz")
    with open(SUITE / "clean_sticks.jsonl") as f:
        sticks = [json.loads(line)["sticks"] for line in f]
    return z["images"].astype(np.float64), sticks


def obj_f1(images: np.ndarray, sticks: list[list[dict]]) -> dict:
    """Object-level scores of the line filter on a stack of raw images."""
    pred = np.stack([predict(im) > 0 for im in images])
    h, w = images.shape[1:]
    s = object_scores(pred, sticks, h, w)
    return {k: s[k] for k in ("obj_f1", "obj_precision", "obj_recall")}


def main() -> None:
    images, sticks = load_clean()
    res = {"clean": obj_f1(images, sticks)}
    print(f"clean        obj F1 {res['clean']['obj_f1']:.3f}", flush=True)
    for si, (kind, levels) in enumerate(LEVELS.items()):
        for li, level in enumerate(levels):
            name = set_name(kind, level)
            rng = np.random.default_rng(50_000_000 + 1000 * si + li)  # seeds of robustness.make
            x = np.stack([perturb(im, kind, level, rng) for im in images]).astype(np.float32)
            res[name] = obj_f1(x.astype(np.float64), sticks)
            print(f"{name:12s} obj F1 {res[name]['obj_f1']:.3f}", flush=True)
    ratios = [res[set_name(k, lv)]["obj_f1"] / res["clean"]["obj_f1"]
              for k, v in LEVELS.items() for lv in v]  # fmt: skip
    res["_robustness"] = float(np.mean(ratios))
    no_pol = [r for (k, v), r in zip(((k, lv) for k, vs in LEVELS.items() for lv in vs), ratios)
              if k != "polarity"]  # fmt: skip
    res["_robustness_no_polarity"] = float(np.mean(no_pol))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(f"robustness {res['_robustness']:.3f} (without polarity "
          f"{res['_robustness_no_polarity']:.3f}) -> {OUT.relative_to(ROOT)}")  # fmt: skip


if __name__ == "__main__":
    main()
