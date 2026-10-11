"""Stage-2 optimiser: the 7-step virtual-gate procedure.

    uv run python stage2_optimizer/optimize_vg.py --n 20 --grid 5 --seed 1000

Steps: 1 reference image, 2 linear D (±30 mV), 3 first curvature (±120 mV),
4 n³ barrier grid with the depth table K, 5 candidates, 6 parabola refinement barrier by
barrier, 7 absolute contrast and answer. The algorithm only calls ``exp.measure``;
``exp.reveal()`` is used by :func:`check` after the run, for evaluation only.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import pathlib
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter
from scipy.ndimage import label as ndlabel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from csd import BARRIERS, CHALLENGE, new_experiment
from stage2_optimizer.virtual_gates import (
    REF_STEP,
    VirtualGates,
    Window,
    clean,
    measure,
    remove_rows,
    stick_depth,
    working_point,
)

H_LINEAR, H_CURV = 0.03, 0.12  # barrier steps of steps 2 and 3 [V]
H_SMALL, H_CROSS = 0.008, 0.1  # compressed calibration: linear step, cross-term points [V]
N_CALIB = 6  # sticks measured during the compressed calibration
LIGHT_SKELETON = False  # compressed grid: register 33 points instead of 53
CENTRED = True  # compressed grid: add the centre of every grid cell (body-centred lattice)
GRID_HALF = 0.8 * CHALLENGE.optimum_range  # grid on [-0.4, 0.4] V per barrier
N_FULL = 30  # grid points measured as full scenes before switching to per-stick patches
PATCH_STEP3_BELOW = 10  # with fewer full grid points than this, step 3 also uses patches
ZETA_MIN, SEPARATION, MAX_CANDIDATES, MIN_VALUES = 3.5, 0.3, 8, 10
REFINE_WIN, REFINE_STEP, REFINE_PTS, REFINE_HALF = 0.04, 1e-3, 9, (0.2, 0.1)
CONTRAST_WIN, CONTRAST_STEP, CONTRAST_REPEAT = 0.024, 0.25e-3, 2
# compressed refinement: strip (g2, g4) [V], its step, passes, repeats on the last pass
FAST_WIN, FAST_STEP, FAST_HALF, FAST_REPEAT = (0.04, 0.012), 2e-3, (0.2, 0.1, 0.05), 2
FAST_CONTRAST = (0.5e-3, 4)  # compressed absolute contrast: step [V], repeats
SHORTLIST = 0.75  # candidates within this fraction of the best contrast get a floor measurement


@dataclass
class Candidate:
    """A (grid point, stick) pair whose depth stands out, then its refined peak."""

    m: int  # grid point index (visit index in the grid)
    i: int  # stick index
    zeta: float  # score
    b_grid: np.ndarray  # barriers of the grid point
    b: np.ndarray | None = None  # refined barriers
    scans: list[tuple[int, np.ndarray, np.ndarray, float]] = field(default_factory=list)
    contrast: float = np.nan  # absolute contrast at the refined barriers
    gain: float = np.nan  # contrast / floor contrast of the same stick (compressed mode)
    floor: float = np.nan  # floor contrast of the stick, far from its peak (compressed mode)
    floor_imgs: list[tuple[np.ndarray, np.ndarray, Window]] = field(default_factory=list)
    scan_imgs: list[list[tuple[np.ndarray, Window]]] = field(default_factory=list)  # per scan
    summit: tuple[np.ndarray, Window] | None = None  # last 40 mV image nearest the vertex
    contrast_imgs: list[tuple[np.ndarray, np.ndarray, Window]] = field(default_factory=list)


@dataclass
class Run:
    """State and trace of one optimisation, filled step by step."""

    exp: object
    vg: VirtualGates
    budget: dict[str, int] = field(default_factory=dict)
    grid: np.ndarray | None = None  # (M, 3) barriers of the grid, in visit order
    grid_visits: slice | None = None  # their indices in vg.visits
    zeta: np.ndarray | None = None  # candidate scores ζ[m, i] (step 5)
    candidates: list[Candidate] = field(default_factory=list)
    final: dict | None = None
    _last: int = 0

    def spend(self, step: str) -> None:
        """Book the pixels spent since the previous call under ``step``."""
        self.budget[step] = self.budget.get(step, 0) + self.exp.n_pixels - self._last
        self._last = self.exp.n_pixels


# --------------------------------------------------------------------- steps 1-3
def step1_reference(exp) -> Run:
    """Reference image at b = 0, centre p0, 0.3 V, 1 mV; detect the constellation."""
    run = Run(exp, VirtualGates(exp))
    run.spend("1 reference")
    return run


def step2_linear(run: Run) -> np.ndarray:
    """±30 mV on each barrier (6 images); D column k = (t+ - t-) / 0.06 V.

    Returns:
        The finite-difference estimate of D, shape (2, 3).
    """
    D = np.zeros((2, 3))
    for k in range(3):
        t = []
        for s in (+1, -1):
            b = np.zeros(3)
            b[k] = s * H_LINEAR
            t.append(run.vg.visit(b).t)
        D[:, k] = (t[0] - t[1]) / (2 * H_LINEAR)
    run.spend("2 linear D")
    return D


def step3_curvature(run: Run, patches: bool = False) -> None:
    """±120 mV on each barrier, centred with the current model; refit D and Q.

    Args:
        run: The run.
        patches: Measure per-stick patches (``VirtualGates.visit_patches``) instead of scenes.
    """
    for k, s in itertools.product(range(3), (+1, -1)):
        b = np.zeros(3)
        b[k] = s * H_CURV
        run.vg.visit_patches(b) if patches else run.vg.visit(b)
    run.spend("3 curvature Q")


# ------------------------------------------------------------------------ step 4
def grid_points(n_grid: int = 5) -> np.ndarray:
    """n³ grid on [-0.4, 0.4]³, ordered from the centre outwards."""
    axis = np.linspace(-GRID_HALF, GRID_HALF, n_grid)
    pts = np.array(list(itertools.product(axis, axis, axis)))
    return pts[np.argsort(np.linalg.norm(pts, axis=1), kind="stable")]


def step4_grid(run: Run, n_grid: int = 5, n_full: int | None = None) -> None:
    """Visit every grid point (scene image, registration, refit, depth of every stick).

    Args:
        run: The run.
        n_grid: Grid points per barrier.
        n_full: Number of first points (nearest the centre) measured as full scenes; the
            following ones only measure a small patch around each predicted stick
            (``VirtualGates.visit_patches``). None = full scene everywhere.
    """
    run.grid = grid_points(n_grid)
    start = len(run.vg.visits)
    for m, b in enumerate(run.grid):
        if n_full is None or m < n_full:
            run.vg.visit(b)
        else:
            run.vg.visit_patches(b)
    run.grid_visits = slice(start, len(run.vg.visits))
    run.spend("4 grid")


# ------------------------------------------------- steps 2-4, compressed (patches only)
def calibrate_patches(run: Run) -> None:
    """Steps 2-3 without any scene image: D, Q_kk and Q_kl from per-stick patches.

    ±8 mV on each barrier (shift < 8 mV: a 40 mV patch is enough without any model), then
    ±120 mV (wide patches), then one point per pair of barriers at 100 mV (wide patches)
    so that every cross term is measured before the grid. Only the ``N_CALIB`` deepest
    sticks of the reference are measured.
    """
    vg = run.vg
    # the translation is common to all sticks: the deepest ones of the reference suffice
    ref_depth = stick_depth(clean(vg.ref_img, REF_STEP), vg.ref_win, vg.sticks.r)
    only = np.argsort(-np.nan_to_num(ref_depth))[:N_CALIB]
    for k, s in itertools.product(range(3), (+1, -1)):
        vg.visit_patches(s * H_SMALL * np.eye(3)[k], wide=False, force=True, only=only)
    run.spend("2 linear D")
    for k, s in itertools.product(range(3), (+1, -1)):
        vg.visit_patches(s * H_CURV * np.eye(3)[k], wide=True, force=True, only=only)
    for k, j in itertools.combinations(range(3), 2):
        b = H_CROSS * (np.eye(3)[k] + np.eye(3)[j])
        vg.visit_patches(b, wide=True, force=True, only=only)
    run.spend("3 curvature Q")


def grid_order(n_grid: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Grid points in the compressed visit order, and which ones are registered.

    First the inner points (every |b_k| <= one grid step, centre outwards), then the
    skeleton of the cube (corners, edge and face centres), then the rest. The inner
    points and the skeleton are registered (patches); the rest lies between registered
    points and is only read (strips). With ``CENTRED``, the centre of every grid cell is
    added (body-centred lattice, strips): a narrow peak is then never half a grid step
    away from the nearest point along every barrier at once.

    Returns:
        (points (M, 3), registered (M,) bool).
    """
    pts = grid_points(n_grid)
    step = 2 * GRID_HALF / (n_grid - 1)
    a = np.abs(pts)
    inner = np.all(a <= step + 1e-9, axis=1)
    skeleton = np.all((a < 1e-9) | (a > GRID_HALF - 1e-9), axis=1) & ~inner
    if LIGHT_SKELETON:  # register fewer points: no inner triple, no edge centre
        moved = np.sum(a > 1e-9, axis=1)
        inner &= moved <= 2
        skeleton &= moved != 2
    rank = np.where(inner, 0, np.where(skeleton, 1, 2))
    order = np.argsort(rank, kind="stable")
    pts, registered = pts[order], (rank < 2)[order]
    if CENTRED:
        mid = np.linspace(-GRID_HALF, GRID_HALF, n_grid)[:-1] + step / 2
        centres = np.array(list(itertools.product(mid, mid, mid)))
        pts = np.vstack([pts, centres])
        registered = np.append(registered, np.zeros(len(centres), bool))
    return pts, registered


def step4_grid_compressed(run: Run, n_grid: int = 5) -> None:
    """Grid of step 4 with patches on the registered points and strips on the others."""
    run.grid, registered = grid_order(n_grid)
    start = len(run.vg.visits)
    for b, reg in zip(run.grid, registered):
        if reg:
            run.vg.visit_patches(b, wide=False)
        else:
            run.vg.read_strips(b)
    run.grid_visits = slice(start, len(run.vg.visits))
    run.spend("4 grid")


def K_table(run: Run) -> np.ndarray:
    """Depth table K[m, i] restricted to the grid visits."""
    return run.vg.depth_table()[run.grid_visits]


# ------------------------------------------------------------------------ step 5
def step5_candidates(run: Run) -> list[Candidate]:
    """Pairs (m, i) whose depth exceeds the stick's own floor by ζ ≥ 3.5 robust spreads.

    A pair is kept if it is more than 0.3 V (along at least one barrier) from every pair
    already kept; at most 8 candidates. Only confirmed sticks with ≥ 10 depths count.
    """
    K = K_table(run)
    f = np.nanmedian(K, axis=0)
    s = 1.4826 * np.nanmedian(np.abs(K - f), axis=0)
    zeta = (K - f) / np.where(s > 0, s, np.nan)
    usable = run.vg.sticks.confirmed & (np.sum(np.isfinite(K), axis=0) >= MIN_VALUES)
    zeta[:, ~usable] = np.nan
    run.zeta = zeta
    order = np.argsort(np.nan_to_num(-zeta, nan=np.inf), axis=None)
    kept: list[Candidate] = []
    for flat in order:
        m, i = np.unravel_index(flat, zeta.shape)
        z = zeta[m, i]
        if not np.isfinite(z) or z < ZETA_MIN or len(kept) == MAX_CANDIDATES:
            break
        b = run.grid[m]
        if all(np.max(np.abs(b - c.b_grid)) > SEPARATION for c in kept):
            kept.append(Candidate(int(m), int(i), float(z), b.copy()))
    run.candidates = kept
    return kept


# ------------------------------------------------------------------------ step 6
def peak_position(x: np.ndarray, y: np.ndarray) -> float:
    """Vertex of the parabola through the deepest sample and its two neighbours.

    Falls back to the deepest sample if it is on the edge or the curvature is not < 0.
    """
    j = int(np.nanargmax(y))
    if j == 0 or j == len(y) - 1:
        return float(x[j])
    ym, y0, yp = y[j - 1], y[j], y[j + 1]
    curv = yp - 2 * y0 + ym
    if not np.isfinite(curv) or curv >= 0:
        return float(x[j])
    h = x[1] - x[0]
    return float(x[j] - 0.5 * h * (yp - ym) / curv)


def peak_fit(x: np.ndarray, y: np.ndarray, n_fit: int = 5) -> float:
    """Vertex of the least-squares parabola through the ``n_fit`` samples around the deepest.

    Falls back to :func:`peak_position` when fewer than ``n_fit`` samples surround the
    deepest one or the curvature is not < 0; the vertex is kept within one sample of it.
    """
    j, h = int(np.nanargmax(y)), n_fit // 2
    if j - h < 0 or j + h >= len(y) or not np.all(np.isfinite(y[j - h : j + h + 1])):
        return peak_position(x, y)
    a, bb, _ = np.polyfit(x[j - h : j + h + 1] - x[j], y[j - h : j + h + 1], 2)
    if a >= 0:
        return peak_position(x, y)
    return float(x[j] + np.clip(-bb / (2 * a), x[j - 1] - x[j], x[j + 1] - x[j]))


def local_depth(
    run: Run, b: np.ndarray, i: int, strip: bool = False
) -> tuple[float, np.ndarray, Window]:
    """Depth of stick ``i`` on a small image centred on its prediction.

    Args:
        run: The run.
        b: Barriers.
        i: Stick index.
        strip: Measure a ``FAST_WIN`` strip at ``FAST_STEP`` instead of the 40 mV, 1 mV square.

    Returns:
        (depth, cleaned image, window).
    """
    p = run.vg.sticks.r[i] + run.vg.predict(b)
    if strip:
        win = Window(tuple(p), FAST_WIN[0], FAST_STEP, FAST_WIN[1])
    else:
        win = Window(tuple(p), REFINE_WIN, REFINE_STEP)
    z = clean(measure(run.exp, b, win), win.step)
    return float(stick_depth(z, win, p[None])[0]), z, win


def step6_refine(run: Run, fast: bool = False) -> None:
    """One barrier at a time, 9 points over ±0.2 V then ±0.1 V, parabola vertex.

    Args:
        run: The run.
        fast: Compressed variant: strips instead of square images, passes ``FAST_HALF``
            (one more, tighter), each depth averaged over ``FAST_REPEAT`` strips on the
            last pass, vertex by a 5-point least-squares parabola.
    """
    halves = FAST_HALF if fast else REFINE_HALF
    for c in run.candidates:
        b = c.b_grid.copy()
        for n_pass, half in enumerate(halves):
            repeat = FAST_REPEAT if fast and n_pass == len(halves) - 1 else 1
            for k in range(3):
                xs = b[k] + np.linspace(-half, half, REFINE_PTS)
                ys, imgs = [], []
                for x in xs:
                    bb = b.copy()
                    bb[k] = x
                    reads = [local_depth(run, bb, c.i, strip=fast) for _ in range(repeat)]
                    d, z, win = np.mean([r[0] for r in reads]), reads[0][1], reads[0][2]
                    ys.append(d)
                    imgs.append((z, win))
                ys = np.array(ys)
                b[k] = peak_fit(xs, ys) if fast else peak_position(xs, ys)
                c.scans.append((k, xs, ys, b[k]))
                c.scan_imgs.append(imgs)
                c.summit = imgs[int(np.argmin(np.abs(xs - b[k])))]
        c.b = b
    run.spend("6 refinement")


# ------------------------------------------------------------------------ step 7
def contrast_core(img: np.ndarray) -> tuple[float, np.ndarray]:
    """Absolute contrast of one image: row medians removed, core of the stick, its contrast.

    The core is the connected set of pixels whose smoothed value is below half of the
    smoothed minimum, eroded by one pixel; the contrast is minus its mean.

    Returns:
        (contrast, core mask).
    """
    img = remove_rows(img)
    smooth = gaussian_filter(img, 2.0)
    core = smooth < 0.5 * smooth.min()
    labels, _ = ndlabel(core, structure=np.ones((3, 3)))
    core = labels == labels[np.unravel_index(np.argmin(smooth), smooth.shape)]
    core = binary_erosion(core) if binary_erosion(core).any() else core
    return float(-img[core].mean()), core


def stick_contrast(
    run: Run, b: np.ndarray, p: np.ndarray, trace: list | None = None, fast: bool = False
) -> float:
    """Absolute contrast of the stick at ``p`` (24 mV, 0.25 mV, mean of 2 images).

    Args:
        run: The run.
        b: Barriers.
        p: Predicted (g2, g4) of the stick.
        trace: If given, receives ``(row-corrected image, core mask, window)`` per image.
        fast: Compressed variant: ``FAST_CONTRAST`` (0.5 mV, mean of 4 images, half the pixels).
    """
    step, repeat = FAST_CONTRAST if fast else (CONTRAST_STEP, CONTRAST_REPEAT)
    win = Window(tuple(p), CONTRAST_WIN, step)
    values = []
    for _ in range(repeat):
        img = measure(run.exp, b, win)
        C, core = contrast_core(img)
        values.append(C)
        if trace is not None:
            trace.append((remove_rows(img), core, win))
    return float(np.mean(values))


def step7_answer(run: Run, fast: bool = False) -> dict:
    """Best refined candidate, then the most contrasted stick lit with it.

    Args:
        run: The run.
        fast: Find the sticks lit with the best candidate from per-stick patches instead
            of a scene image.

    Returns:
        ``{"gates": {g1..g5}, "b": ..., "stick": i, "contrast": C, "visit": Visit,
        "lit": [(i, C), ...]}``.
    """
    for c in run.candidates:
        p = run.vg.sticks.r[c.i] + run.vg.predict(c.b)
        c.contrast = stick_contrast(run, c.b, p, c.contrast_imgs, fast=fast)
    run.spend("7a candidate contrasts")
    best = max(run.candidates, key=lambda c: c.contrast)
    if fast:  # compare the peaks themselves: divide out each stick's own floor contrast
        floors: dict[int, float] = {}
        for c in run.candidates:
            if c.contrast >= SHORTLIST * best.contrast and c.i not in floors:
                b_far = -GRID_HALF * np.where(c.b >= 0, 1.0, -1.0)
                p = run.vg.sticks.r[c.i] + run.vg.predict(b_far)
                floors[c.i] = stick_contrast(run, b_far, p, c.floor_imgs, fast=True)
        for c in run.candidates:
            c.floor = floors.get(c.i, np.nan)
            c.gain = c.contrast / floors[c.i] if floors.get(c.i, 0) > 0 else 0.0
        run.spend("7a floor contrasts")
        best = max(run.candidates, key=lambda c: c.gain)

    if fast:
        n = len(run.vg.sticks.r)
        v = run.vg.visit_patches(best.b, wide=False)
        depth = np.pad(run.vg.depths[-1], (0, n - len(run.vg.depths[-1])), constant_values=np.nan)
    else:
        v = run.vg.visit(best.b)
        depth = stick_depth(v.z, v.win, run.vg.sticks.r + v.t)
    lit = [(best.i, best.contrast)]
    for i in np.flatnonzero(depth > 0.5 * np.nanmax(depth)):
        if i != best.i and run.vg.sticks.confirmed[i]:
            p_i = run.vg.sticks.r[i] + v.t
            lit.append((int(i), stick_contrast(run, best.b, p_i, fast=fast)))
    i_final, C = max(lit, key=lambda x: x[1])
    p = run.vg.sticks.r[i_final] + v.t
    run.spend("7b final choice")
    run.final = {
        "gates": working_point(best.b, p),
        "b": best.b,
        "stick": i_final,
        "contrast": C,
        "visit": v,
        "lit": lit,
    }
    return run.final


# ------------------------------------------------------------------ full procedure
def optimize(exp, n_grid: int = 5, mode: str = "compressed", n_full: int = N_FULL) -> Run:
    """Run the 7 steps on ``exp``; the answer is ``run.final["gates"]``.

    Args:
        exp: The experiment.
        n_grid: Grid points per barrier.
        mode: How steps 2-4 measure. ``"full"``: a scene image everywhere (reference procedure).
            ``"patches"``: scenes for steps 2-3 and the first ``n_full`` grid points, then
            per-stick patches. ``"compressed"``: no scene image after the reference
            (patches for the calibration and the registered grid points, strips elsewhere).
        n_full: See ``mode="patches"``.
    """
    run = step1_reference(exp)
    if mode == "compressed":
        calibrate_patches(run)
        step4_grid_compressed(run, n_grid)
    else:
        step2_linear(run)
        step3_curvature(run, patches=mode == "patches" and n_full < PATCH_STEP3_BELOW)
        step4_grid(run, n_grid, n_full if mode == "patches" else None)
    step5_candidates(run)
    if not run.candidates:  # nothing lit anywhere: keep the best grid point of any stick
        K = K_table(run)
        m, i = np.unravel_index(np.nanargmax(K / np.nanmedian(K, axis=0)), K.shape)
        run.candidates = [Candidate(int(m), int(i), np.nan, run.grid[m].copy())]
    step6_refine(run, fast=mode == "compressed")
    step7_answer(run, fast=mode == "compressed")
    return run


def check(exp, gates: dict[str, float]) -> dict:
    """Post-hoc evaluation (uses the hidden state, never called by the algorithm).

    Returns:
        quality = best contrast factor at the answer's barriers / maximal factor,
        dist = max |b - b*| [V], plus the budget.
    """
    rev = exp.reveal()
    wp = dict(exp.start, **gates)
    quality = exp._sim.contrast(wp) / rev["max_contrast_factor"]
    b_opt = np.array([rev["optimum_barriers"][g] for g in BARRIERS])
    b = np.array([gates[g] for g in BARRIERS])
    return {
        "quality": float(quality),
        "dist": float(np.max(np.abs(b - b_opt))),
        "n_pixels": rev["n_pixels"],
        "n_measurements": rev["n_measurements"],
    }


def baseline(exp) -> dict[str, float]:
    """The starter coordinate ascent (for comparison)."""
    sys.path.insert(
        0, str(pathlib.Path(__file__).resolve().parents[1] / "starter" / "stage2_optimization")
    )
    from optimize import optimize as starter_optimize

    return starter_optimize(exp)


def evaluate(
    seeds,
    n_grid: int = 5,
    with_baseline: bool = False,
    verbose: bool = True,
    mode: str = "compressed",
    n_full: int = N_FULL,
):
    """Run the optimiser on fresh devices and check each answer post hoc.

    Args:
        seeds: Device seeds (test seeds: 1000+, never used for tuning).
        n_grid: Grid points per barrier.
        mode: Measurement mode of steps 2-4, see :func:`optimize`.
        n_full: Full-scene grid points of ``mode="patches"``.
        with_baseline: Also run the starter coordinate ascent on the same devices.
        verbose: Print one line per device.

    Returns:
        One dict per device: quality, dist, reached, n_pixels, n_measurements, secs
        (+ baseline, baseline_pixels).
    """
    rows = []
    for seed in seeds:
        t0 = time.time()
        exp = new_experiment(seed=seed)
        run = optimize(exp, n_grid, mode, n_full)
        res = check(exp, run.final["gates"])
        refined = min(np.max(np.abs(c.b - _opt(exp))) for c in run.candidates)
        res.update(seed=seed, reached=bool(refined < 0.05), secs=time.time() - t0)
        res.update(
            grid_pixels=run.budget["4 grid"],
            refine_pixels=run.budget["6 refinement"],
            scene_visits=sum(v.patches is None for v in run.vg.visits),
            n_sticks=len(run.vg.sticks.r),
            n_candidates=len(run.candidates),
        )
        if with_baseline:
            exp_b = new_experiment(seed=seed)
            rb = check(exp_b, baseline(exp_b))
            res.update(baseline=rb["quality"], baseline_pixels=rb["n_pixels"])
        rows.append(res)
        if verbose:
            print(
                f"seed {seed}: quality {res['quality']:.3f}  |b-b*| {1e3 * res['dist']:4.0f} mV  "
                f"optimal peak refined: {'yes' if res['reached'] else 'no ':3s}  "
                f"{res['n_measurements']} images  {res['n_pixels'] / 1e6:.2f} M px"
                + (f"  baseline {res['baseline']:.3f}" if with_baseline else "")
                + f"  ({res['secs']:.0f} s)"
            )
    return rows


def summary(rows) -> str:
    """Median/mean quality, share >= 0.95, optimal peak refined, median pixels."""
    q = np.array([r["quality"] for r in rows])
    lines = [
        (
            f"quality      : median {np.median(q):.3f}, mean {q.mean():.3f}, "
            f">= 0.95: {np.mean(q >= 0.95):.0%}, min {q.min():.3f}"
        ),
        f"optimal peak refined: {np.mean([r['reached'] for r in rows]):.0%}",
        f"pixels       : median {np.median([r['n_pixels'] for r in rows]) / 1e6:.2f} M",
    ]
    if "baseline" in rows[0]:
        qb = np.array([r["baseline"] for r in rows])
        pb = np.median([r["baseline_pixels"] for r in rows]) / 1e6
        lines.append(
            f"baseline     : median {np.median(qb):.3f}, mean {qb.mean():.3f}, {pb:.2f} M px"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-2 optimiser (virtual gates + grid).")
    p.add_argument("--n", type=int, default=20, help="number of devices")
    p.add_argument("--grid", type=int, default=5, help="grid points per barrier")
    p.add_argument("--seed", type=int, default=1000, help="first seed")
    p.add_argument("--baseline", action="store_true", help="also run the starter optimiser")
    p.add_argument("--out", type=pathlib.Path, help="save the per-device results (CSV)")
    p.add_argument(
        "--mode",
        choices=("full", "patches", "compressed"),
        default="compressed",
        help="how steps 2-4 measure: scenes everywhere, patches after --full scenes, or no scene",
    )
    p.add_argument("--full", type=int, default=N_FULL, help="full-scene grid points (patches)")
    args = p.parse_args()

    seeds = range(args.seed, args.seed + args.n)
    rows = evaluate(seeds, args.grid, args.baseline, mode=args.mode, n_full=args.full)
    print("---")
    print(summary(rows))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def _opt(exp) -> np.ndarray:
    rev = exp.reveal()
    return np.array([rev["optimum_barriers"][g] for g in BARRIERS])


if __name__ == "__main__":
    main()
