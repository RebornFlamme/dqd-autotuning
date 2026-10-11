"""Measurement operators and virtual gates for the stage-2 optimiser.

Implements the image operators of the procedure (clean, detect, register, readout) and the translation model of steps 2-3 (D + Q, ridge fit), plus an incremental
stick discovery: sticks missed on the reference image are added whenever a later image
shows them, and their depth is back-filled in every stored image.

Built on the simulator's public API only: ``exp.measure`` (the only acquisition),
``exp.start`` (start working point), :class:`csd.ScanWindow` (span/step of a measurement),
``csd.BARRIERS`` / ``csd.PLUNGERS`` (gate names) and ``csd.CHALLENGE`` (scene size and step,
barrier range). Conventions: an image of a window centred on (g2c, g4c), span W and step s
has rows = g4 (row 0 = lowest g4) and columns = g2, and pixel ``(j, i)`` sits at
``(g2c - W/2 + i s, g4c - W/2 + j s)``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import correlate, gaussian_filter
from scipy.ndimage import label as ndlabel

from csd import BARRIERS, CHALLENGE, PLUNGERS, ScanWindow
from stage1_linefilter.detector import predict as stage1_predict

SCENE = CHALLENGE.generator.scene_span  # side of the scene [V] (0.3)
SCENE_STEP = CHALLENGE.generator.scene_step  # step of the scene images [V] (2 mV)
REF_STEP = SCENE_STEP / 2  # step of the reference image [V] (1 mV)
FILTER_LEN = 6e-3  # matched-filter length along the stick [V]
SEARCH = (0.03, 0.06, 0.12)  # registration search radii, retried in order [V]
LAMBDA = 1e-5  # ridge penalty on the quadratic terms
STICK_SIZE = 15e-3  # blobs closer than this are one stick (sticks are 8-16 mV long) [V]
DETECT_K = 4.5  # detection level in robust sigmas
PATCH = 0.04  # side of the per-stick patches of :meth:`VirtualGates.visit_patches` [V]
PATCH_SEARCH = 0.012  # registration search radius on the patches [V]
PATCH_YOUNG, SEARCH_YOUNG = 0.06, 0.02  # same, while the model has < N_YOUNG measurements
N_YOUNG = 40
STRIP, STRIP_V = 0.04, 0.012  # sides (g2, g4) of the strips of :meth:`VirtualGates.read_strips` [V]

Detector = Callable[[np.ndarray], np.ndarray]


# --------------------------------------------------------------------------- windows
@dataclass(frozen=True)
class Window:
    """A measurement window: a :class:`csd.ScanWindow` placed at a plunger centre.

    Attributes:
        center: (g2, g4) of the window centre [V].
        span: Side of the window along g2 [V] (and along g4 if ``span_v`` is None).
        step: Pixel size [V].
        span_v: Side along g4 [V] for a rectangular window (None = square).
    """

    center: tuple[float, float]
    span: float
    step: float
    span_v: float | None = None

    @property
    def spans(self) -> np.ndarray:
        """(span along g2, span along g4) [V]."""
        return np.array([self.span, self.span if self.span_v is None else self.span_v])

    @property
    def scan(self) -> ScanWindow:
        """The simulator's scan window (span/step, both axes)."""
        sh, sv = self.spans
        return ScanWindow(span_h=sh, span_v=sv, step_h=self.step, step_v=self.step)

    @property
    def n(self) -> int:
        """Number of pixels along g2."""
        return self.scan.n_h

    def to_px(self, volts: np.ndarray) -> np.ndarray:
        """(g2, g4) volts -> (col, row) fractional pixels."""
        lo = np.asarray(self.center) - self.spans / 2
        return (np.asarray(volts) - lo) / self.step

    def to_volts(self, px: np.ndarray) -> np.ndarray:
        """(col, row) pixels -> (g2, g4) volts."""
        lo = np.asarray(self.center) - self.spans / 2
        return lo + np.asarray(px) * self.step

    def extent(self) -> tuple[float, float, float, float]:
        """``imshow`` extent (g2_min, g2_max, g4_min, g4_max)."""
        lo = np.asarray(self.center) - self.spans / 2
        return (lo[0], lo[0] + self.spans[0], lo[1], lo[1] + self.spans[1])


def working_point(b: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Gate dict {g1..g5} from barriers ``b`` (order ``csd.BARRIERS``) and plungers ``p``."""
    wp = {g: float(x) for g, x in zip(BARRIERS, b)}
    wp.update({g: float(x) for g, x in zip(PLUNGERS, p)})
    return wp


def start_plungers(exp) -> np.ndarray:
    """p0: the start plungers (g2, g4) of the experiment (``exp.start``)."""
    return np.array([exp.start[g] for g in PLUNGERS])


def measure(exp, b: np.ndarray, win: Window) -> np.ndarray:
    """Acquire one image of ``win`` at barriers ``b`` (``exp.measure``)."""
    sw = win.scan
    return exp.measure(
        **working_point(b, np.asarray(win.center)),
        span_h=sw.span_h,
        span_v=sw.span_v,
        step_h=sw.step_h,
        step_v=sw.step_v,
    )


# ----------------------------------------------------------------- image operators
def remove_rows(img: np.ndarray) -> np.ndarray:
    """Subtract the median of each row (scan-line offsets)."""
    return img - np.median(img, axis=1, keepdims=True)


def clean(img: np.ndarray, step: float) -> np.ndarray:
    """Row medians removed, 45° matched filter, light Gaussian blur.

    Args:
        img: Raw image (rows = g4, cols = g2).
        step: Pixel size [V].

    Returns:
        Filtered image ``z`` (sticks are negative dips).
    """
    n = max(2, round(FILTER_LEN / step))
    z = correlate(remove_rows(img), np.eye(n) / n, mode="nearest")
    return gaussian_filter(z, 0.5)


def robust_sigma(z: np.ndarray) -> float:
    """Robust noise level: 1.4826 x median absolute deviation."""
    return float(1.4826 * np.median(np.abs(z - np.median(z))))


def detect(img: np.ndarray, win: Window, detector: Detector = stage1_predict) -> np.ndarray:
    """Stick positions in volts, from the stage-1 detector mask.

    Each connected component of the mask gives one stick, located at its barycentre
    weighted by the depth ``-z`` of the cleaned image. Components
    closer than ``STICK_SIZE`` are fragments of the same stick and are merged.

    Args:
        img: Raw image.
        win: Its window.
        detector: Function raw image -> binary mask.

    Returns:
        Array (n, 2) of (g2, g4) positions.
    """
    mask = np.asarray(detector(img)).astype(bool)
    z = clean(img, win.step)
    labels, n = ndlabel(mask, structure=np.ones((3, 3)))
    blobs = []  # (position [V], weight)
    for k in range(1, n + 1):
        rows, cols = np.nonzero(labels == k)
        w = np.clip(-z[rows, cols], 1e-9, None)
        pos = win.to_volts(np.array([np.average(cols, weights=w), np.average(rows, weights=w)]))
        for j, (q, wq) in enumerate(blobs):
            if np.hypot(*(pos - q)) < STICK_SIZE:
                blobs[j] = ((q * wq + pos * w.sum()) / (wq + w.sum()), wq + w.sum())
                break
        else:
            blobs.append((pos, w.sum()))
    return np.array([q for q, _ in blobs]).reshape(-1, 2)


def _box_max(img: np.ndarray, px: np.ndarray, half: int) -> np.ndarray:
    """Max of ``img`` in a (2 half + 1)^2 box around each (col, row) pixel (NaN if outside)."""
    out = np.full(len(px), np.nan)
    n_rows, n_cols = img.shape
    for k, (c, r) in enumerate(np.rint(px).astype(int)):
        if half <= c < n_cols - half and half <= r < n_rows - half:
            out[k] = img[r - half : r + half + 1, c - half : c + half + 1].max()
    return out


def stick_depth(
    z: np.ndarray, win: Window, positions: np.ndarray, half_mv: float = 4e-3
) -> np.ndarray:
    """Depth = max of ``-z`` in a small box around each predicted stick.

    Args:
        z: Cleaned image.
        win: Its window.
        positions: (n, 2) predicted (g2, g4) of the sticks.
        half_mv: Half side of the box [V] (±2 px at 2 mV).

    Returns:
        Depth per stick (NaN when outside the image).
    """
    half = max(1, round(half_mv / win.step))
    return _box_max(-z, win.to_px(positions), half)


# -------------------------------------------------------------------- registration
@dataclass
class Registration:
    """Result of :func:`register`."""

    t: np.ndarray  # measured translation (g2, g4) [V]
    ok: bool  # quality control passed
    score: np.ndarray  # score map S(u), rows = du_g4, cols = du_g2
    radius: int  # search radius [px]


def register(
    z: np.ndarray, win: Window, refs: np.ndarray, t_pred: np.ndarray, search: float
) -> Registration:
    """Translation that puts every reference stick on a dip at once.

    Args:
        z: Cleaned image.
        win: Its window.
        refs: (n, 2) reference positions r_i at b = 0 [V].
        t_pred: Predicted translation [V].
        search: Search radius [V].

    Returns:
        The registration (translation, quality flag, score map).
    """
    s = np.clip(-z / robust_sigma(z), 0.0, 6.0)
    R = round(search / win.step)
    pad = np.pad(s, R + 1)
    base = np.rint(win.to_px(refs + t_pred)).astype(int) + R + 1
    S = np.zeros((2 * R + 1, 2 * R + 1))
    for c, r in base:
        if R <= c < pad.shape[1] - R - 1 and R <= r < pad.shape[0] - R - 1:
            S += pad[r - R : r + R + 1, c - R : c + R + 1]
    S = gaussian_filter(S, 1.0)
    du, ok = _peak(S, R)
    return Registration(t=t_pred + du * win.step, ok=ok, score=S, radius=R)


def _peak(S: np.ndarray, R: int) -> tuple[np.ndarray, bool]:
    """Peak of a smoothed score map S(u) of radius ``R`` and its quality control.

    Returns:
        (shift (col, row) in pixels, refined by the barycentre within 2 px; QC passed).
    """
    j, i = np.unravel_index(np.argmax(S), S.shape)
    excess = np.clip(S - np.median(S), 0, None)
    yy, xx = np.mgrid[-2:3, -2:3]
    jj, ii = np.clip(j + yy, 0, 2 * R), np.clip(i + xx, 0, 2 * R)
    w = excess[jj, ii] * (yy**2 + xx**2 <= 4)
    du = np.array([(w * ii).sum(), (w * jj).sum()]) / max(w.sum(), 1e-12) - R

    far = np.hypot(*np.mgrid[-j : 2 * R + 1 - j, -i : 2 * R + 1 - i]) > 5
    second = excess[far].max() if far.any() else 0.0
    ok = bool(S.max() >= np.median(S) + 8 * robust_sigma(S) and second <= 0.7 * excess.max())
    return du, ok


# ------------------------------------------------------------- translation model
def features(b: np.ndarray) -> np.ndarray:
    """phi(b) = (b1, b2, b3, b1², b2², b3², 2b1b2, 2b1b3, 2b2b3)."""
    b1, b2, b3 = b
    return np.array([b1, b2, b3, b1 * b1, b2 * b2, b3 * b3, 2 * b1 * b2, 2 * b1 * b3, 2 * b2 * b3])


@dataclass
class TranslationModel:
    """t(b) = D b + [bᵀ Q_x b]_x, fitted by ridge regression (penalty on Q only)."""

    bs: list[np.ndarray] = field(default_factory=list)
    ts: list[np.ndarray] = field(default_factory=list)
    theta: np.ndarray = field(default_factory=lambda: np.zeros((9, 2)))

    def add(self, b: np.ndarray, t: np.ndarray) -> None:
        """Add one measured translation and refit."""
        self.bs.append(np.asarray(b, float))
        self.ts.append(np.asarray(t, float))
        self.fit()

    def fit(self, linear_only: bool = False) -> np.ndarray:
        """Ridge least squares (step 3).

        Args:
            linear_only: Fit D alone (Q = 0), for comparison.

        Returns:
            The (9, 2) coefficients.
        """
        Phi = np.array([features(b) for b in self.bs])
        T = np.array(self.ts)
        if linear_only:
            theta = np.zeros((9, 2))
            theta[:3] = np.linalg.lstsq(Phi[:, :3], T, rcond=None)[0]
            return theta
        # (ΦᵀΦ + λΛ)θ = ΦᵀT, solved as an augmented least-squares problem so that the
        # first images (fewer than 3 barriers moved) give the minimum-norm D.
        Lam = np.diag([0.0] * 3 + [1.0] * 6)
        A = np.vstack([Phi, np.sqrt(LAMBDA) * Lam])
        self.theta = np.linalg.lstsq(A, np.vstack([T, np.zeros((9, 2))]), rcond=None)[0]
        return self.theta

    def predict(self, b: np.ndarray, theta: np.ndarray | None = None) -> np.ndarray:
        """Predicted translation (g2, g4) at barriers ``b``."""
        th = self.theta if theta is None else theta
        return features(np.asarray(b, float)) @ th

    @property
    def D(self) -> np.ndarray:
        """Linear coefficients, shape (2, 3) (rows g2, g4; columns g1, g3, g5)."""
        return self.theta[:3].T

    def Q(self, x: int) -> np.ndarray:
        """Symmetric 3x3 quadratic matrix of axis ``x`` (0 = g2, 1 = g4)."""
        q = self.theta[3:, x]
        return np.array([[q[0], q[3], q[4]], [q[3], q[1], q[5]], [q[4], q[5], q[2]]])


# ------------------------------------------------------------------ virtual gates
@dataclass
class Visit:
    """One scene image, registered (or one set of per-stick patches: ``z`` is None)."""

    b: np.ndarray
    win: Window
    z: np.ndarray | None
    sigma: float
    t: np.ndarray
    t_pred: np.ndarray
    ok: bool
    n_tries: int
    patches: list[tuple[np.ndarray, Window]] | None = None  # (cleaned patch, window) per stick
    sticks: np.ndarray | None = None  # indices of the sticks of ``patches``
    registered: bool = True  # False for strips: ``t`` is the prediction, not a measurement


@dataclass
class Sticks:
    """The constellation: reference positions r_i (b = 0) and their status."""

    r: np.ndarray  # (n, 2) [V]
    confirmed: np.ndarray  # (n,) bool: usable for registration / candidates
    sightings: list[list[np.ndarray]]  # per stick, its r estimates
    found_at: list[int]  # index of the visit where the stick was found (-1 = reference)

    def add(self, r: np.ndarray, visit: int) -> None:
        """Append a tentative stick."""
        self.r = np.vstack([self.r, r])
        self.confirmed = np.append(self.confirmed, False)
        self.sightings.append([r])
        self.found_at.append(visit)


class VirtualGates:
    """Follow every stick whatever the barriers (steps 1-3 + discovery).

    Args:
        exp: The experiment (only ``exp.measure`` is called).
        detector: Raw image -> binary stick mask (stage-1 detector by default).
    """

    def __init__(self, exp, detector: Detector = stage1_predict):
        self.exp = exp
        self.detector = detector
        self.model = TranslationModel()
        self.visits: list[Visit] = []
        self.depths: list[np.ndarray] = []  # per visit, depth of every known stick
        self.p0 = start_plungers(exp)
        self.ref_win = Window(tuple(self.p0), SCENE, REF_STEP)
        self.ref_img = measure(exp, np.zeros(3), self.ref_win)
        r = detect(self.ref_img, self.ref_win, detector)
        self.sticks = Sticks(r, np.ones(len(r), bool), [[x] for x in r], [-1] * len(r))

    # -- tracking ------------------------------------------------------------
    def predict(self, b: np.ndarray) -> np.ndarray:
        """Predicted translation t(b) (zero before any measurement)."""
        return self.model.predict(b) if self.model.bs else np.zeros(2)

    def _first_search(self, b: np.ndarray) -> int:
        """Index in ``SEARCH`` of the first try at ``b``.

        A cross term Q_kl (k != l) is only measured once a visit has moved barriers k and
        l together; before that, the prediction can be off by tens of mV (up to 2|Q_kl|
        |b_k b_l|), more than the ±30 mV search. The first visit that needs a cross term
        never seen yet therefore starts at ±60 mV.
        """
        seen = np.zeros((3, 3), bool)
        for x in self.model.bs:
            seen |= np.abs(np.outer(x, x)) > 0
        need = (np.abs(np.outer(b, b)) > 0) & ~np.eye(3, dtype=bool)
        return int(np.any(need & ~seen))

    def visit(self, b: np.ndarray, record: bool = True) -> Visit:
        """Measure the scene at ``b``, register it, refit the model, discover sticks.

        The image is centred on p0 + t_pred(b) with a margin equal to the search radius.
        It is re-measured with a larger window and search (±30,
        then ±60, then ±120 mV) if quality control fails or if the peak lies at the
        edge of the search area (the true shift may be outside it).

        Args:
            b: Barriers (g1, g3, g5) [V].
            record: Store the visit (depth table, discovery).

        Returns:
            The visit. If every try failed, ``ok`` is False, the model is not updated and
            ``t`` is the prediction.
        """
        b = np.asarray(b, float)
        t_pred = self.predict(b)
        refs = self.sticks.r[self.sticks.confirmed]
        for n_try, search in enumerate(SEARCH[self._first_search(b) :], 1):
            win = Window(tuple(self.p0 + t_pred), SCENE + 2 * search, SCENE_STEP)
            img = measure(self.exp, b, win)
            z = clean(img, win.step)
            reg = register(z, win, refs, t_pred, search)
            # once D is known (step 2 done) a shift at the edge of the search is suspect
            inside = len(self.model.bs) < 6 or np.max(np.abs(reg.t - t_pred)) < 0.8 * search
            if reg.ok and inside:
                break
        ok = reg.ok and inside
        v = Visit(b, win, z, robust_sigma(z), reg.t if ok else t_pred, t_pred, ok, n_try)
        if v.ok:
            self.model.add(b, v.t)
        if record:
            self.visits.append(v)
            self.depths.append(stick_depth(z, win, self.sticks.r + v.t))
            if v.ok:
                self._discover(img, v)
            self._update_confirmations()
        return v

    def visit_patches(
        self,
        b: np.ndarray,
        wide: bool | None = None,
        force: bool = False,
        only: np.ndarray | None = None,
    ) -> Visit:
        """Like :meth:`visit`, but measure only a small patch around each predicted stick.

        Meant for barriers where the model is already accurate (a few mV): each known
        stick gets a ``PATCH`` window centred on r_i + t_pred(b) (``PATCH_YOUNG`` when
        ``wide``). As every patch is centred on its prediction, the registration score
        S(u) is the sum of the patches of the confirmed sticks. If quality control fails
        or the shift lies at the edge of the search, the measurement is retried with the
        wide patch, then with the full scene (:meth:`visit`). No stick is discovered on a
        patch visit.

        Args:
            b: Barriers (g1, g3, g5) [V].
            wide: Use the wide patch and search (None = while the model has fewer than
                ``N_YOUNG`` measurements).
            force: Use patches even if ``b`` needs a cross term never measured yet
                (otherwise such a point is measured as a full scene).
            only: Indices of the sticks to measure (None = all); the depth of the others
                is NaN for this visit.

        Returns:
            The visit (``z`` is None and ``patches`` is set unless it fell back).
        """
        b = np.asarray(b, float)
        if not force and self._first_search(b):  # prediction too uncertain
            return self.visit(b)
        if wide is None:
            wide = len(self.model.bs) < N_YOUNG
        span, search = (PATCH_YOUNG, SEARCH_YOUNG) if wide else (PATCH, PATCH_SEARCH)
        idx = np.arange(len(self.sticks.r)) if only is None else np.asarray(only)
        t_pred = self.predict(b)
        wins = [Window(tuple(r + t_pred), span, SCENE_STEP) for r in self.sticks.r[idx]]
        zs = [clean(measure(self.exp, b, w), w.step) for w in wins]
        sigma = robust_sigma(np.concatenate([z.ravel() for z in zs]))

        R = round(search / SCENE_STEP)
        c = wins[0].n // 2  # pixel of the patch centre
        S = np.zeros((2 * R + 1, 2 * R + 1))
        for z in (z for z, conf in zip(zs, self.sticks.confirmed[idx]) if conf):
            S += np.clip(-z / sigma, 0.0, 6.0)[c - R : c + R + 1, c - R : c + R + 1]
        du, ok = _peak(gaussian_filter(S, 1.0), R)
        if not ok or np.max(np.abs(du)) * SCENE_STEP >= 0.8 * search:
            if not wide:  # retry with the wide patch before paying for a scene
                return self.visit_patches(b, wide=True, force=True, only=only)
            return self.visit(b)

        t = t_pred + du * SCENE_STEP
        scene = Window(tuple(self.p0 + t_pred), SCENE, SCENE_STEP)
        v = Visit(b, scene, None, sigma, t, t_pred, True, 1, list(zip(zs, wins)), idx)
        self.model.add(b, t)
        self.visits.append(v)
        depth = np.full(len(self.sticks.r), np.nan)
        depth[idx] = [
            stick_depth(z, w, (r + t)[None])[0] for z, w, r in zip(zs, wins, self.sticks.r[idx])
        ]
        self.depths.append(depth)
        self._update_confirmations()
        return v

    def read_strips(self, b: np.ndarray) -> Visit:
        """Depth of every stick from a thin strip at its predicted position, no registration.

        Meant for barriers inside the region where the model has already been registered
        (interpolation): the strip (``STRIP`` along g2 x ``STRIP_V`` along g4) only holds
        the box where the depth is read. The model is not updated.

        Args:
            b: Barriers (g1, g3, g5) [V].

        Returns:
            The visit (``t`` is the prediction, ``z`` is None, ``patches`` holds the strips).
        """
        b = np.asarray(b, float)
        t = self.predict(b)
        wins = [Window(tuple(r + t), STRIP, SCENE_STEP, STRIP_V) for r in self.sticks.r]
        zs = [clean(measure(self.exp, b, w), w.step) for w in wins]
        sigma = robust_sigma(np.concatenate([z.ravel() for z in zs]))
        scene = Window(tuple(self.p0 + t), SCENE, SCENE_STEP)
        idx = np.arange(len(wins))
        v = Visit(b, scene, None, sigma, t, t, True, 1, list(zip(zs, wins)), idx, registered=False)
        self.visits.append(v)
        self.depths.append(
            np.array(
                [stick_depth(z, w, (r + t)[None])[0] for z, w, r in zip(zs, wins, self.sticks.r)]
            )
        )
        self._update_confirmations()
        return v

    def _update_confirmations(self) -> None:
        """Confirm a discovered stick once it is above detection level in 2 other images."""
        for k in np.flatnonzero(~self.sticks.confirmed):
            seen = [
                self.depths[n][k] / v.sigma > DETECT_K
                for n, v in enumerate(self.visits)
                if n != self.sticks.found_at[k] and v.ok and np.isfinite(self.depths[n][k])
            ]
            self.sticks.confirmed[k] = sum(seen) >= 2

    def _discover(self, img: np.ndarray, v: Visit) -> None:
        """Add the sticks seen in ``img`` that are not in the constellation yet."""
        idx = len(self.visits) - 1
        margin = 5 * v.win.step
        lo = np.asarray(v.win.center) - v.win.span / 2 + margin
        hi = lo + v.win.span - 2 * margin
        for p in detect(img, v.win, self.detector):
            if not np.all((p > lo) & (p < hi)):
                continue
            r = p - v.t
            d = np.hypot(*(self.sticks.r - r).T)
            k = int(np.argmin(d)) if len(d) else -1
            if k >= 0 and d[k] < STICK_SIZE:
                if self.sticks.found_at[k] >= 0:  # refine a discovered stick's position
                    self.sticks.sightings[k].append(r)
                    self.sticks.r[k] = np.mean(self.sticks.sightings[k], axis=0)
                continue
            if stick_depth(v.z, v.win, p[None])[0] <= DETECT_K * v.sigma:  # too faint: noise
                continue
            self.sticks.add(r, idx)
            self._backfill(len(self.sticks.r) - 1)

    def _backfill(self, k: int) -> None:
        """Depth of the new stick ``k`` in every stored visit (column k of K)."""
        for n, v in enumerate(self.visits):
            seen = v.ok and v.z is not None  # a patch visit has no image of the new stick
            d = stick_depth(v.z, v.win, self.sticks.r[k][None] + v.t)[0] if seen else np.nan
            if len(self.depths[n]) < len(self.sticks.r):
                self.depths[n] = np.append(self.depths[n], d)
            else:
                self.depths[n][k] = d

    def depth_table(self) -> np.ndarray:
        """K[m, i]: depth of stick i at visit m (NaN where unknown)."""
        n = len(self.sticks.r)
        return np.array([np.pad(d, (0, n - len(d)), constant_values=np.nan) for d in self.depths])
