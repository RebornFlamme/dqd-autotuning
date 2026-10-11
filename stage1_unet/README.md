# ML solution: interdot detection with a U-Net

Stage 1 with deep learning. Two detectors, each answering the failure of the previous one:

1. **U-Net** (1.9 M parameters) trained on an infinite simulator generator, with a log-uniform
   contrast law so that faint sticks (the regime of stage 2) are well represented.
2. **U-Net LOFO**: the same U-Net behind `PhysInput`, a parameter-free physical preprocessing
   (row median, MAD rescaling, despiking), trained on 8 randomized laboratory-artefact families
   and tested **leave-one-family-out**: every artefact is scored by a model that never saw it.

The walkthrough is [`walkthrough.ipynb`](walkthrough.ipynb).

| | U-Net | U-Net LOFO |
|---|---|---|
| obj F1, clean test (200 scenes) | **0.980** | 0.975 |
| obj F1, stage-2 frames | **0.964** | 0.951 |
| robustness score, 25 perturbed sets | 0.872 | **0.930** |
| same, without polarity | 0.892 | **0.964** |

*obj F1*: a stick is found if a predicted pixel lies within len/2 + 1 px of its centre.
*Robustness score*: mean of obj F1 (perturbed) / obj F1 (clean) over 9 artefact families × up to
3 levels; 1 means insensitive.

## Running the notebook

PyTorch is not in the project dependencies, so add it on the fly:

```bash
uv run --with torch jupyter nbconvert --to notebook --execute --inplace stage1_unet/walkthrough.ipynb
```

The U-Net runs live on CPU (about 1 min). The LOFO numbers are read from `results/lofo/`
because the 22 LOFO checkpoints are not versioned.

## Contents

| path | content |
|---|---|
| `detection/` | `data_gen.py` (geometry pool and eval sets), `synth.py` (GPU noise and artefacts), `models.py` (U-Net, `PhysInput`), `train.py`, `evalsets.py` / `metrics.py` (scoring), `evaluate.py`, `export_dl.py`, `robustness.py` (artefact suite), `lofo.py` + `lofo_job.sh` |
| `models/unet_lowsnr.pt` | frozen U-Net (EMA weights in float16 + val threshold) |
| `robustness_suite/` | the 200 clean test scenes of the robustness suite (the 25 perturbed sets are regenerated from them, bit for bit) |
| `results/` | training logs and evals of the U-Net, robustness summary, LOFO table and per-run evals |

```python
from detection.export_dl import load, predict_mask
model, thr = load("models/unet_lowsnr.pt")   # run from stage1_unet/
masks = predict_mask(model, thr, images)   # images: (N, H, W) raw CSDs -> boolean masks
```

## Reproducing the training (GPU)

From `stage1_unet/`, with `uv run --with torch`:

```bash
python -m detection.data_gen pool --n 30000 --out data/pool --workers 32
python -m detection.data_gen sets --out data/eval --workers 32
python -m detection.data_gen verify --pool data/pool --eval data/eval     # noise statistics must match the official ones
python -m detection.train --arch unet --seed 0 --steps 14000 --warmup 300 --eval-every 1000 --compile --intensity-law loguniform --out runs/fast_unet_lowsnr
python -m detection.export_dl runs/fast_unet_lowsnr models/unet_lowsnr.pt
python -m detection.robustness make --src data/eval/test --out data/robustness  # the artefact suite
python -m detection.lofo jobs                 # 22 lines "name|args"; each -> bash detection/lofo_job.sh name args
python -m detection.lofo report --out results/lofo
```
