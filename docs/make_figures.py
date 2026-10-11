"""Figures of the README.

    uv run python docs/make_figures.py            # every figure
    uv run python docs/make_figures.py device csd # a subset

Every figure is written to ``docs/figures/<name>.svg`` (text as paths, white background) and
``docs/figures/<name>.png`` (300 dpi). The data figures are computed here from the code of the
repository; the example device is ``new_experiment(seed=1007)`` throughout. Hidden simulator
state (``exp._sim``, ``exp.reveal()``) is only read to draw the ground truth of the problem
figure and to score the answers, never by the algorithms.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from functools import cache

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "figures"

# ----------------------------------------------------------------------------- style
INK = "#1a1a1a"
MUTED = "#6b6f76"
FAINT = "#d9dbdf"
BLUE = "#2B5C9E"  # categorical order: blue, vermilion, ochre, green (validated, CVD-safe)
RED = "#C8553D"
OCHRE = "#B8860B"
GREEN = "#13916B"
GRAY = "#8A8F98"
LIGHT_BLUE = "#dbe6f4"
LIGHT_RED = "#f6ded8"
LIGHT_OCHRE = "#f3e7c9"
LIGHT_GREEN = "#d3ece3"
PAPER = "#f7f5f0"

CSD_CMAP = LinearSegmentedColormap.from_list("csd", ["#0b1a33", "#2B5C9E", "#e9eef5", "#ffffff"])
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "depth", ["#ffffff", "#f3e7c9", "#d9a441", "#C8553D", "#5a1a10"]
)


def setup_style() -> None:
    """Computer Modern (Latin Modern) everywhere, thin recessive axes, white background."""
    for f in pathlib.Path("/usr/share/texmf/fonts/opentype/public/lm").glob("lmroman10-*.otf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update(
        {
            "font.family": "Latin Modern Roman",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "image.origin": "lower",
            "svg.fonttype": "path",
            "lines.linewidth": 1.2,
            "axes.titlelocation": "left",
            "axes.titlepad": 4,
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote docs/figures/{name}.svg / .png")


def tag(ax, letter: str, x: float = -0.02, y: float = 1.02) -> None:
    """Panel letter, bold, top-left."""
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom")  # fmt: skip


def bare(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def frame(ax, color=INK, lw=0.6) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(color)
        s.set_linewidth(lw)


def arrow(ax, p, q, color=INK, lw=0.9, style="-|>", ms=7, **kw):
    a = FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                        shrinkA=0, shrinkB=0, **kw)  # fmt: skip
    ax.add_patch(a)
    return a


def overlay(ax, mask, color, alpha=0.85, extent=None):
    m = np.ma.masked_where(~mask, mask)
    ax.imshow(m, cmap=ListedColormap([color]), alpha=alpha, interpolation="nearest",
              extent=extent, vmin=0, vmax=1)  # fmt: skip


# ----------------------------------------------------------------------------- data
@cache
def example_run():
    """The full optimisation of the example device (seed 1007), with its trace."""
    from csd import new_experiment
    from stage2_optimizer import optimize_vg as opt

    exp = new_experiment(seed=1007)
    run = opt.optimize(exp)
    return exp, run


def _json(path: pathlib.Path):
    with path.open() as f:
        return json.load(f)


def read_csv(name: str) -> list[dict]:
    path = ROOT / "stage2_optimizer" / "results" / name
    with path.open() as f:
        return [
            {k: (v == "True") if v in ("True", "False") else float(v) for k, v in r.items()}
            for r in csv.DictReader(f)
        ]


# ============================================================================ 1. device
def fig_device() -> None:
    """(a) the nanotube over five gates, (b) the double-well potential along it."""
    fig = plt.figure(figsize=(7.2, 2.35))
    ax = fig.add_axes([0.0, 0.0, 0.55, 1.0])
    bx = fig.add_axes([0.62, 0.14, 0.38, 0.78])

    # ---- (a) device
    ax.set_xlim(-0.6, 10.6)
    ax.set_ylim(-1.15, 3.9)
    ax.set_aspect("equal")
    bare(ax)
    ax.add_patch(Rectangle((-0.5, -0.55), 11.0, 0.55, fc="#ecebe6", ec="none"))
    ax.text(10.4, -0.32, "substrate", ha="right", va="center", fontsize=7, color=MUTED)
    gates = [("$g_1$", "barrier"), ("$g_2$", "plunger"), ("$g_3$", "barrier"),
             ("$g_4$", "plunger"), ("$g_5$", "barrier")]  # fmt: skip
    xs = np.linspace(1.6, 8.4, 5)
    for x, (name, kind) in zip(xs, gates):
        fc, ec = (LIGHT_RED, RED) if kind == "barrier" else (LIGHT_BLUE, BLUE)
        w = 0.62 if kind == "barrier" else 1.05
        ax.add_patch(Rectangle((x - w / 2, 0.0), w, 0.55, fc=fc, ec=ec, lw=0.9))
        ax.text(x, 0.27, name, ha="center", va="center", fontsize=9)
        ax.text(x, -0.82, kind, ha="center", va="center", fontsize=7.5,
                color=RED if kind == "barrier" else BLUE)  # fmt: skip
    # contacts
    for x0, lab in ((-0.3, "S"), (9.3, "D")):
        ax.add_patch(Rectangle((x0, 0.0), 1.0, 2.05, fc="#e3e4e8", ec=INK, lw=0.8))
        ax.text(x0 + 0.5, 1.0, lab, ha="center", va="center", fontsize=10)
    # suspended nanotube
    y0, h = 2.05, 0.42
    ax.add_patch(FancyBboxPatch((0.2, y0), 9.6, h, boxstyle="round,pad=0,rounding_size=0.21",
                                fc="#f4f4f2", ec=INK, lw=0.9, zorder=3))  # fmt: skip
    # honeycomb hint on the tube
    for k, xx in enumerate(np.arange(0.45, 9.7, 0.32)):
        yy = y0 + h / 2 + (0.08 if k % 2 else -0.08)
        ax.add_patch(Polygon(_hexagon(xx, yy, 0.085), closed=True, fc="none", ec="#9aa0a8",
                             lw=0.4, zorder=4))  # fmt: skip
    ax.annotate("carbon nanotube", xy=(5.0, y0 + h), xytext=(5.0, 3.25), ha="center",
                fontsize=8, arrowprops={"arrowstyle": "-", "lw": 0.6, "color": INK})  # fmt: skip
    # vacuum gap + field lines
    for x in xs:
        ax.plot([x, x], [0.6, y0 - 0.05], ls=(0, (1.5, 1.5)), lw=0.6, color=GRAY, zorder=1)
    # dots
    for x, lab in zip(xs[[1, 3]], ("dot 1", "dot 2")):
        ax.add_patch(FancyBboxPatch((x - 0.6, y0 - 0.07), 1.2, h + 0.14,
                                    boxstyle="round,pad=0,rounding_size=0.25", fc="none",
                                    ec=BLUE, lw=1.1, ls="--", zorder=5))  # fmt: skip
        ax.text(x, y0 + h + 0.22, lab, ha="center", va="bottom", fontsize=8, color=BLUE)
    ax.text(-0.4, 3.85, "(a)", fontsize=10, fontweight="bold", va="top")

    # ---- (b) potential
    x = np.linspace(0, 10, 800)
    centers = np.linspace(1.6, 8.4, 5)

    def U(barriers, wells):
        u = np.zeros_like(x)
        for c, a in zip(centers[[0, 2, 4]], barriers):
            u += a * np.exp(-((x - c) ** 2) / (2 * 0.42**2))
        for c, a in zip(centers[[1, 3]], wells):
            u -= a * np.exp(-((x - c) ** 2) / (2 * 0.75**2))
        return u

    u = U((1.0, 0.62, 1.0), (0.55, 0.55))
    u2 = U((1.0, 0.32, 1.0), (0.55, 0.55))
    m = np.abs(u - u2) > 1e-3
    bx.plot(x[m], u2[m], color=RED, lw=1.0, ls="--")
    bx.plot(x, u, color=INK, lw=1.3)
    bx.annotate("$t_c$ (set by $g_3$)", xy=(5.0, 0.22), xytext=(5.0, -0.45),
                color=RED, fontsize=7.5, ha="center", va="center",
                arrowprops={"arrowstyle": "-", "lw": 0.6, "color": RED})  # fmt: skip
    for c, lab in zip(centers[[1, 3]], (r"$|0\rangle$", r"$|1\rangle$")):
        bx.text(c, -0.62, lab, ha="center", va="top", fontsize=10)
    bx.plot(centers[1], u[np.argmin(abs(x - centers[1]))] + 0.08, "o", ms=6, color=BLUE, zorder=5)
    bx.text(centers[1] - 1.0, -0.38, "$e^-$", color=BLUE, fontsize=9)
    # detuning arrow
    bx.annotate("", xy=(centers[3] + 1.25, -0.42), xytext=(centers[3] + 1.25, -0.2),
                arrowprops={"arrowstyle": "<|-|>", "lw": 0.8, "color": INK, "mutation_scale": 7})  # fmt: skip
    bx.text(centers[3] + 1.4, -0.31, r"$\varepsilon$", fontsize=10, va="center")
    for c, lab in zip(centers, ("$g_1$", "$g_2$", "$g_3$", "$g_4$", "$g_5$")):
        bx.text(c, 1.13, lab, ha="center", fontsize=8.5,
                color=RED if lab in ("$g_1$", "$g_3$", "$g_5$") else BLUE)  # fmt: skip
    bx.set_xlim(0, 10)
    bx.set_ylim(-0.85, 1.3)
    bx.set_xticks([])
    bx.set_yticks([])
    bx.spines["bottom"].set_visible(True)
    bx.set_xlabel("position along the tube")
    bx.set_ylabel("potential energy")
    bx.text(-0.05, 1.31, "(b)", fontsize=10, fontweight="bold", va="bottom", ha="right")
    save(fig, "device")


def _hexagon(x, y, r):
    t = np.arange(6) * np.pi / 3 + np.pi / 6
    return np.c_[x + r * np.cos(t), y + r * np.sin(t)]


# ======================================================================= 2. stability
def cim_ground_state(v1, v2, Ec1=1.0, Ec2=1.0, Ecm=0.35, a12=0.12, a21=0.12, nmax=4):
    """Ground-state charges of the constant-interaction model on a voltage grid.

    F(N) = ½ Nᵀ E N − N·u with E = [[Ec1, Ecm], [Ecm, Ec2]] and u = α v (cross lever arms
    a12, a21), the free energy of the README (all energies in units of Ec1).
    """
    u1 = v1 + a12 * v2
    u2 = v2 + a21 * v1
    best = np.full(v1.shape, np.inf)
    n1 = np.zeros(v1.shape, int)
    n2 = np.zeros(v1.shape, int)
    for a in range(nmax + 1):
        for b in range(nmax + 1):
            F = 0.5 * Ec1 * a * a + 0.5 * Ec2 * b * b + Ecm * a * b - a * u1 - b * u2
            m = F < best
            best[m], n1[m], n2[m] = F[m], a, b
    return n1, n2


def fig_csd() -> None:
    """(a) the honeycomb of the constant-interaction model, (b) a simulated CSD, (c) zoom."""
    from csd import new_experiment

    fig = plt.figure(figsize=(7.2, 2.55))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.62], wspace=0.28)
    ax, bx, cx = (fig.add_subplot(gs[i]) for i in range(3))

    # ---- (a) honeycomb
    n = 900
    v = np.linspace(0.15, 3.05, n)
    V1, V2 = np.meshgrid(v, v)
    n1, n2 = cim_ground_state(V1, V2)
    state = n1 * 10 + n2
    shade = ((n1 + n2) % 2).astype(float)
    ax.imshow(shade, extent=(v[0], v[-1], v[0], v[-1]), cmap=ListedColormap([PAPER, "#ebe7dc"]),
              interpolation="nearest")  # fmt: skip
    # boundaries: charge lines (total N changes) vs interdot (N1+N2 constant)
    dx = np.diff(state, axis=1) != 0
    dy = np.diff(state, axis=0) != 0
    tot = n1 + n2
    inter_x = dx & (np.diff(tot, axis=1) == 0)
    inter_y = dy & (np.diff(tot, axis=0) == 0)
    line = np.zeros_like(shade, bool)
    inter = np.zeros_like(shade, bool)
    line[:, 1:] |= dx & ~inter_x
    line[1:, :] |= dy & ~inter_y
    inter[:, 1:] |= inter_x
    inter[1:, :] |= inter_y
    from scipy.ndimage import binary_dilation

    ext = (v[0], v[-1], v[0], v[-1])
    overlay(ax, binary_dilation(line, iterations=2), INK, 1.0, ext)
    overlay(ax, binary_dilation(inter, iterations=5), RED, 1.0, ext)
    # labels at the centre of each domain
    for a in range(4):
        for b in range(4):
            m = (n1 == a) & (n2 == b)
            if m.sum() > 3000:
                yy, xx = np.nonzero(m)
                cx_, cy_ = v[int(xx.mean())], v[int(yy.mean())]
                if 0.35 < cx_ < 2.85 and 0.35 < cy_ < 2.85:
                    ax.text(cx_, cy_, f"({a},{b})", ha="center", va="center", fontsize=7)
    ax.set_xlim(v[0], v[-1])
    ax.set_ylim(v[0], v[-1])
    ax.set_xlabel("plunger $V_2$")
    ax.set_ylabel("plunger $V_4$")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("constant-interaction model", color=MUTED, fontsize=8.5)
    ax.text(0.98, -0.2, "", transform=ax.transAxes)
    tag(ax, "a", -0.06)
    # legend
    ax.plot([], [], color=INK, lw=1.4, label="charge transition dot $\\leftrightarrow$ lead")
    ax.plot([], [], color=RED, lw=2.6, label="interdot transition (the 'stick')")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, handlelength=1.6,
              borderaxespad=0)  # fmt: skip

    # ---- (b) simulated CSD
    exp = new_experiment(seed=1007)
    step = 1e-3
    img = exp.measure(span_h=0.3, span_v=0.3, step_h=step, step_v=step)
    ext = (0.0, 0.3, 0.0, 0.3)
    lo, hi = np.percentile(img, [0.5, 99.5])
    bx.imshow(img, extent=ext, cmap=CSD_CMAP, vmin=lo, vmax=hi, interpolation="nearest")
    bx.set_xlabel("$g_2$ (V)")
    bx.set_ylabel("$g_4$ (V)")
    bx.set_xticks([0, 0.1, 0.2, 0.3])
    bx.set_yticks([0, 0.1, 0.2, 0.3])
    bx.set_title("simulated CSD (raw signal)", color=MUTED, fontsize=8.5)
    tag(bx, "a".replace("a", "b"), -0.12)
    # zoom box around the deepest stick of the reference
    sim = exp._sim
    mids = np.array([s.middle for s in sim.base_interdots])
    k = int(np.argmin(np.hypot(*(mids - [0.075, 0.2]).T)))
    c = mids[k]
    half = 0.022
    bx.add_patch(Rectangle(c - half, 2 * half, 2 * half, fc="none", ec=OCHRE, lw=1.1))

    # ---- (c) zoom
    j0, i0 = np.round((c - half) / step).astype(int)[::-1]
    nz = round(2 * half / step)
    z = img[j0 : j0 + nz, i0 : i0 + nz]
    cx.imshow(z, extent=(c[0] - half, c[0] + half, c[1] - half, c[1] + half), cmap=CSD_CMAP,
              vmin=lo, vmax=hi, interpolation="nearest")  # fmt: skip
    frame(cx, OCHRE, 1.1)
    s = sim.base_interdots[k]
    ux, uy = np.cos(s.theta), np.sin(s.theta)
    L = s.length / 2
    off = 0.0075
    cx.plot([c[0] - L * ux - off * uy * 0, c[0] + L * ux], [c[1] - L * uy + off, c[1] + L * uy + off],
            color=RED, lw=0)  # fmt: skip
    cx.text(0.5, -0.06, "one interdot: 1–2 px wide,\n4–8 px long, a few $\\sigma$ deep",
            transform=cx.transAxes, ha="center", va="top", fontsize=7.5)  # fmt: skip
    cx.set_title(f"zoom ({1e3 * 2 * half:.0f} mV)", color=MUTED, fontsize=8.5)
    tag(cx, "c", -0.05)
    save(fig, "csd")


# ========================================================================= 3. problem
def fig_problem() -> None:
    """(a) barriers translate every stick, (b) hidden contrast landscape, (c) start vs optimum."""
    from csd import BARRIERS, new_experiment

    exp = new_experiment(seed=1007)
    sim = exp._sim  # ground truth, for the illustration only
    rev = exp.reveal()
    b_star = np.array([rev["optimum_barriers"][g] for g in BARRIERS])

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.08, 0.5], height_ratios=[1, 1], wspace=0.42,
                          hspace=0.35)  # fmt: skip
    ax = fig.add_subplot(gs[:, 0])
    bx = fig.add_subplot(gs[:, 1])
    c0 = fig.add_subplot(gs[0, 2])
    c1 = fig.add_subplot(gs[1, 2])

    # ---- (a) drift
    step = 2e-3
    img = exp.measure(span_h=0.3, span_v=0.3, step_h=step, step_v=step)
    lo, hi = np.percentile(img, [0.5, 99.5])
    ax.imshow(img, extent=(0, 0.3, 0, 0.3), cmap=CSD_CMAP, vmin=lo, vmax=hi, alpha=0.55,
              interpolation="nearest")  # fmt: skip
    mids = np.array([s.middle for s in sim.base_interdots])
    db = 0.05
    cols = (RED, OCHRE, GREEN)
    for k, (g, col) in enumerate(zip(BARRIERS, cols)):
        e = np.zeros(3)
        e[k] = db
        d = sim.drift_matrix.drift(e)
        for m in mids:
            arrow(ax, m, m + d, color=col, lw=1.0, ms=6)
        ax.plot([], [], color=col, lw=1.2, label=f"${g[0]}_{g[1]}$ + {1e3 * db:.0f} mV")
    ax.plot(*mids.T, "o", ms=2.6, color=INK, zorder=5)
    ax.set(xlim=(0, 0.3), ylim=(0, 0.3), xlabel="$g_2$ (V)", ylabel="$g_4$ (V)")
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=3, fontsize=7,
              handlelength=1.0, columnspacing=0.8, borderaxespad=0)  # fmt: skip
    ax.set_title("barriers translate every stick", color=MUTED, fontsize=8.5)
    tag(ax, "a", -0.12)

    # ---- (b) contrast landscape (slice through the optimum)
    n = 161
    g = np.linspace(-0.5, 0.5, n)
    G1, G3 = np.meshgrid(g, g)
    F = np.zeros_like(G1)
    for i in range(n):
        for j in range(n):
            F[i, j] = sim.contrast({**exp.start, "g1": G1[i, j], "g3": G3[i, j], "g5": b_star[2]})
    im = bx.imshow(F, extent=(g[0], g[-1], g[0], g[-1]), cmap="cividis", origin="lower",
                   interpolation="bilinear")  # fmt: skip
    cb = fig.colorbar(im, ax=bx, fraction=0.046, pad=0.03)
    cb.set_label("best contrast factor", fontsize=8)
    cb.outline.set_linewidth(0.5)
    bx.plot(*b_star[:2], marker="*", ms=11, mec=INK, mfc="white", mew=0.8, ls="none")
    bx.plot(0, 0, marker="o", ms=5, mec=INK, mfc="none", mew=0.9, ls="none")
    bx.annotate("start", (0, 0), (0.04, 0.04), fontsize=7.5, color="white")
    bx.annotate("hidden optimum", b_star[:2], (b_star[0] + 0.08, b_star[1] - 0.12), fontsize=7.5,
                color="white")  # fmt: skip
    for r in sim.contrast_model.regions:
        if abs(r.center[2] - b_star[2]) < 1e-9:
            continue
    bx.set(xlabel="barrier $g_1$ (V)", ylabel="barrier $g_3$ (V)")
    bx.set_title(f"hidden contrast, slice $g_5$ = {1e3 * b_star[2]:.0f} mV", color=MUTED,
                 fontsize=8.5)  # fmt: skip
    tag(bx, "b", -0.14)

    # ---- (c) the same stick, start vs optimum
    k = int(
        np.argmax(sim._region_of == np.argmax([r.amplitude for r in sim.contrast_model.regions]))
    )
    best_region = int(np.argmax([r.amplitude for r in sim.contrast_model.regions]))
    k = int(np.flatnonzero(sim._region_of == best_region)[0])
    p0 = sim.base_interdots[k].middle + np.array([exp.start["g2"], exp.start["g4"]]) - 0.15
    span, st = 0.03, 0.5e-3
    a0 = exp.measure(g2=p0[0], g4=p0[1], span_h=span, span_v=span, step_h=st, step_v=st)
    d = sim.drift_matrix.drift(b_star)
    gates = dict(zip(BARRIERS, b_star))
    a1 = exp.measure(g2=p0[0] + d[0], g4=p0[1] + d[1], span_h=span, span_v=span, step_h=st,
                     step_v=st, **gates)  # fmt: skip
    vlo, vhi = np.percentile(a1, [0.5, 99.5])
    for cx, a, title in ((c0, a0, "start ($b = 0$)"), (c1, a1, "optimum ($b = b^\\star$)")):
        cx.imshow(a - np.median(a, axis=1, keepdims=True), cmap=CSD_CMAP, vmin=vlo - np.median(a1),
                  vmax=vhi - np.median(a1), interpolation="nearest")  # fmt: skip
        frame(cx, MUTED, 0.6)
        cx.set_title(title, color=MUTED, fontsize=8)
    c0.text(1.05, 0.5, f"contrast\n×{sim.contrast_model.base:.0f}", transform=c0.transAxes,
            fontsize=7.5, va="center")  # fmt: skip
    c1.text(1.05, 0.5, f"contrast\n×{rev['max_contrast_factor']:.0f}", transform=c1.transAxes,
            fontsize=7.5, va="center")  # fmt: skip
    tag(c0, "c", -0.08)
    save(fig, "problem")


# ======================================================================== 4. pipeline
def _box(ax, x, y, w, h, title, body, fc, ec, num=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=fc, ec=ec, lw=0.9))  # fmt: skip
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=8.2,
            fontweight="bold")  # fmt: skip
    ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=7, color=INK,
            linespacing=1.25)  # fmt: skip
    if num is not None:
        ax.add_patch(Circle((x + 0.13, y + h - 0.02), 0.11, fc=ec, ec="none", zorder=5))
        ax.text(x + 0.13, y + h - 0.02, num, ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=6)  # fmt: skip


def fig_pipeline() -> None:
    """Overview: stage 1 detector, then the 7 measurement steps with their pixel budget."""
    _, run = example_run()
    bud = run.budget
    px = {
        "1": bud["1 reference"],
        "2": bud["2 linear D"],
        "3": bud["3 curvature Q"],
        "4": bud["4 grid"],
        "5": 0,
        "6": bud["6 refinement"],
        "7": bud["7a candidate contrasts"] + bud["7a floor contrasts"] + bud["7b final choice"],
    }
    total = sum(px.values())

    fig = plt.figure(figsize=(7.2, 2.62))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 5.25)
    bare(ax)

    # stage 1 row
    _box(ax, 0.15, 3.95, 2.5, 1.1, "raw CSD", "2-D map $(g_2, g_4)$\nnoise, row offsets", "#f2f3f5",
         GRAY)  # fmt: skip
    _box(ax, 3.35, 3.95, 4.6, 1.1, "Stage 1 — interdot detector",
         "line filter (no training) or U-Net\nraw image $\\rightarrow$ binary mask of sticks",
         LIGHT_BLUE, BLUE)  # fmt: skip
    _box(ax, 8.65, 3.95, 5.6, 1.1, "Stage 2 — virtual-gate optimiser",
         "constant-interaction model: one translation $t(b)$ for all sticks\n"
         "$\\Rightarrow$ measure only where the sticks are predicted", LIGHT_RED, RED)  # fmt: skip
    arrow(ax, (2.68, 4.5), (3.3, 4.5))
    arrow(ax, (7.98, 4.5), (8.6, 4.5))

    steps = [
        ("1", "reference", "one scene\n0.3 V, 1 mV\n→ sticks $r_i$"),
        ("2", "linear $D$", "±8 mV / barrier\n40 mV patches\n→ lever arms"),
        ("3", "curvature $Q$", "±120 mV, pairs\nwide patches\n→ $t(b)$ fitted"),
        ("4", "explore", "189 barrier\npoints, patches\n+ thin strips"),
        ("5", "candidates", "robust score\n$\\zeta_{m,i}$ per stick\n(no measurement)"),
        ("6", "refine", "1-D sweeps\nparabola vertex\n3 passes"),
        ("7", "answer", "absolute contrast\n+ gain vs floor\n→ $g^\\star$"),
    ]
    w, gap, x0, y = 1.86, 0.2, 0.15, 1.3
    groups = {"track the sticks": (0, 3), "explore the cube": (3, 5), "conclude": (5, 7)}
    for k, (num, title, body) in enumerate(steps):
        x = x0 + k * (w + gap)
        _box(ax, x, y, w, 1.6, title, body, "white", RED, num)
        if k:
            arrow(ax, (x - gap + 0.01, y + 0.8), (x - 0.02, y + 0.8), ms=6)
        # budget bar
        frac = px[num] / total
        ax.add_patch(Rectangle((x, 0.42), w, 0.2, fc="#f1f1ef", ec="none"))
        ax.add_patch(Rectangle((x, 0.42), w * frac / max(px[n] / total for n in px), 0.22,
                               fc=RED, ec="none"))  # fmt: skip
        ax.text(x + w / 2, 0.3, f"{px[num] / 1e3:.0f} k px" if px[num] else "—", ha="center",
                va="top", fontsize=7, color=MUTED)  # fmt: skip
    for name, (a, b) in groups.items():
        xa, xb = x0 + a * (w + gap), x0 + b * (w + gap) - gap
        ax.plot([xa, xa, xb, xb], [3.05, 3.15, 3.15, 3.05], color=MUTED, lw=0.7)
        ax.text((xa + xb) / 2, 3.22, name, ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.text(0.15, 0.72, f"pixels measured per step on the example device (seed 1007), "
            f"{total / 1e6:.2f} M in all", ha="left", va="bottom", fontsize=7, color=MUTED)  # fmt: skip
    save(fig, "pipeline")


# ====================================================================== 5. line filter
def detector_stages(image):
    """Every intermediate result of ``stage1_linefilter.detector.predict``, in order."""
    from scipy.ndimage import binary_dilation, correlate
    from scipy.ndimage import label as ndlabel
    from scipy.ndimage import maximum as ndmaximum

    from stage1_linefilter.detector import PARAMS, filter_by_orientation, predict, robust_depth

    p = PARAMS
    rows = image - np.median(image, axis=1, keepdims=True)
    depth = robust_depth(rows)
    along = robust_depth(correlate(rows, np.eye(p["line_len"]) / p["line_len"], mode="nearest"))
    seeds = along > p["k_line"]
    cand = binary_dilation(seeds, iterations=p["dilate"]) & (depth > p["k_low"])
    labels, n = ndlabel(cand, structure=np.ones((3, 3)))
    smooth = correlate(depth, np.eye(2) / 2, mode="nearest")
    amp = np.concatenate([[0.0], ndmaximum(smooth, labels, np.arange(1, n + 1))])[labels]
    fwhm = cand & (depth > p["half"] * amp) & (amp > p["k_peak"])
    final = filter_by_orientation(fwhm, rows)
    assert (final == predict(image)).all()
    return {"raw": image, "rows": rows, "depth": depth, "along": along, "seeds": seeds, "cand": cand,
                "fwhm": fwhm, "final": final}  # fmt: skip


def fig_linefilter() -> None:
    """The five operations of the classical detector on one validation image."""
    from skimage.measure import label, regionprops

    from stage1_linefilter.eval_detector import ensure_dataset

    ds = ensure_dataset(ROOT / "data" / "val", n=256, seed=2)
    images = np.load(ds / "images.npy", mmap_mode="r")
    masks = np.load(ds / "masks.npy", mmap_mode="r")
    k = 10
    img, truth = np.asarray(images[k], float), np.asarray(masks[k]) > 0
    s = detector_stages(img)
    # crop: around a component removed by the orientation filter and its nearest stick
    regs = regionprops(label(s["fwhm"], connectivity=2))
    bad = max((r for r in regs if not s["final"][tuple(r.coords[0])]), key=lambda r: r.area)
    good = min((r for r in regs if s["final"][tuple(r.coords[0])]),
               key=lambda r: np.hypot(*np.subtract(r.centroid, bad.centroid)))  # fmt: skip
    c = (np.array(bad.centroid) + np.array(good.centroid)) / 2
    h = 17
    r0, c0 = np.clip(np.round(c).astype(int) - h, 0, 150 - 2 * h)
    Z = (slice(r0, r0 + 2 * h), slice(c0, c0 + 2 * h))

    fig, axes = plt.subplots(1, 6, figsize=(7.2, 1.75))
    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99, top=0.78, bottom=0.17)
    lo, hi = np.percentile(img, [1, 99])
    titles = [
        ("raw image", "row offsets + noise"),
        ("1 · row median", "offsets removed"),
        ("2 · robust depth", r"$d = (\mathrm{med}-a)/1.48\,\mathrm{MAD}$"),
        ("3 · line filter", "5-px mean at 45°, seeds"),
        ("4 · half-max cut", r"keep $d > 0.6\,A$"),
        ("5 · orientation", "8 directions, 45° ± 22.5°"),
    ]
    axes[0].imshow(s["raw"][Z], cmap=CSD_CMAP, vmin=lo, vmax=hi, interpolation="nearest")
    axes[1].imshow(s["rows"][Z], cmap=CSD_CMAP, vmin=lo - np.median(img), vmax=hi - np.median(img),
                   interpolation="nearest")  # fmt: skip
    axes[2].imshow(s["depth"][Z], cmap=DEPTH_CMAP, vmin=-1, vmax=7, interpolation="nearest")
    axes[3].imshow(s["along"][Z], cmap=DEPTH_CMAP, vmin=-1, vmax=7, interpolation="nearest")
    _outline(axes[3], s["seeds"][Z], BLUE)
    axes[4].imshow(s["rows"][Z], cmap="gray", vmin=lo - np.median(img), vmax=hi - np.median(img),
                   alpha=0.45, interpolation="nearest")  # fmt: skip
    overlay(axes[4], s["fwhm"][Z], RED, 0.9)
    axes[5].imshow(s["rows"][Z], cmap="gray", vmin=lo - np.median(img), vmax=hi - np.median(img),
                   alpha=0.45, interpolation="nearest")  # fmt: skip
    overlay(axes[5], (s["fwhm"] & ~s["final"])[Z], GRAY, 0.55)
    overlay(axes[5], s["final"][Z], RED, 0.9)
    _outline(axes[5], truth[Z], INK)
    for ax, (a, b) in zip(axes, titles):
        frame(ax, FAINT)
        ax.set_title(a, fontsize=8.2, loc="center", pad=11)
        ax.text(0.5, 1.03, b, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.6,
                color=MUTED)  # fmt: skip
    axes[5].text(0.5, -0.05, "red: kept   grey: removed\nblack outline: ground truth",
                 transform=axes[5].transAxes, ha="center", va="top", fontsize=6.4, color=MUTED)  # fmt: skip
    axes[3].text(0.5, -0.05, "blue outline: seeds", transform=axes[3].transAxes, ha="center",
                 va="top", fontsize=6.4, color=MUTED)  # fmt: skip
    save(fig, "linefilter")


def _outline(ax, mask, color, lw=0.8):
    """Draw the pixel outline of a boolean mask."""
    mask = np.asarray(mask, bool)
    H, W = mask.shape
    pad = np.pad(mask, 1)
    segs = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c]:
                continue
            if not pad[r + 2, c + 1]:
                segs.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
            if not pad[r, c + 1]:
                segs.append([(c - 0.5, r - 0.5), (c + 0.5, r - 0.5)])
            if not pad[r + 1, c]:
                segs.append([(c - 0.5, r - 0.5), (c - 0.5, r + 0.5)])
            if not pad[r + 1, c + 2]:
                segs.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
    from matplotlib.collections import LineCollection

    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))


# ================================================================ 6. detection results
FAMILIES = {
    "white": "white noise", "pink": "1/f noise", "drift": "drift", "jumps": "charge jumps",
    "stripes": "row stripes", "lowpass": "lock-in low-pass", "saturate": "saturation",
    "spikes": "outlier pixels", "polarity": "inverted polarity",
}  # fmt: skip


def robustness_table() -> dict[str, dict[str, float]]:
    """obj F1 of the three detectors on clean + the 25 perturbed sets (same 200 scenes)."""
    sys.path.insert(0, str(ROOT / "stage1_unet"))
    from detection.robustness import LEVELS, set_name

    res = ROOT / "stage1_unet" / "results"
    unet = _json(res / "robustness_dl_summary.json")["fast_unet_lowsnr"]["sets"]
    lf = _json(ROOT / "stage1_linefilter" / "results" / "robustness_linefilter.json")
    out = {"line filter": {}, "U-Net": {}, "U-Net LOFO": {}, "_clean": {}}
    lofo_clean = {}
    for f, lvs in LEVELS.items():
        ev = _json(res / f"lofo/per_run/lofo_u32_no-{f}/eval.json")["sets"]
        for lv in lvs:
            n = set_name(f, lv)
            out["line filter"][n] = lf[n]["obj_f1"]
            out["U-Net"][n] = unet[n]["obj_f1"][0]
            out["U-Net LOFO"][n] = ev[n]["obj_f1"]
            lofo_clean[n] = ev["clean"]["obj_f1"]
    out["_clean"] = {"line filter": lf["clean"]["obj_f1"], "U-Net": unet["clean"]["obj_f1"][0],
                     "U-Net LOFO": float(np.mean(list(lofo_clean.values())))}  # fmt: skip
    out["_lofo_clean"] = lofo_clean
    out["_levels"] = {f: [set_name(f, lv) for lv in lvs] for f, lvs in LEVELS.items()}
    return out


def robustness_scores(tab) -> dict[str, tuple[float, float]]:
    """Mean of obj F1(perturbed) / obj F1(clean), with and without polarity."""
    out = {}
    for m in ("line filter", "U-Net", "U-Net LOFO"):
        names = [n for lv in tab["_levels"].values() for n in lv]
        clean = {
            n: (tab["_lofo_clean"][n] if m == "U-Net LOFO" else tab["_clean"][m]) for n in names
        }
        r = [tab[m][n] / clean[n] for n in names]
        rn = [tab[m][n] / clean[n] for n in names if not n.startswith("polarity")]
        out[m] = (float(np.mean(r)), float(np.mean(rn)))
    return out


def fig_detection() -> None:
    """(a) Dice of the line filter on 2000 test images, (b) robustness of the three detectors."""
    scores = np.load(ROOT / "stage1_linefilter" / "results" / "dice_test_seed2026.npy")
    tab = robustness_table()
    rob = robustness_scores(tab)

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.9], wspace=0.25)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    bins = np.linspace(0, 1, 41)
    ax.hist(scores, bins, color=OCHRE, ec="white", lw=0.4)
    ax.set_yscale("log")
    ax.axvline(scores.mean(), color=INK, lw=1.0)
    ax.axvline(np.median(scores), color=MUTED, lw=0.8, ls="--")
    ax.text(scores.mean() - 0.02, 900, f"mean\n{scores.mean():.3f}", color=INK, ha="right",
            va="top", fontsize=7.5)  # fmt: skip
    ax.text(0.03, 900, f"{len(scores)} unseen images\nmedian {np.median(scores):.3f}\n"
            f"Dice $\\geq$ 0.9: {np.mean(scores >= 0.9):.1%}", ha="left", va="top", fontsize=7.5)  # fmt: skip
    ax.set(xlabel="Dice per image", ylabel="images", xlim=(0, 1), ylim=(0.8, 2000))
    ax.set_title("line filter, no training", color=MUTED, fontsize=8.5)
    tag(ax, "a", -0.14)

    fams = list(FAMILIES)
    methods = [("line filter", OCHRE), ("U-Net", BLUE), ("U-Net LOFO", RED)]
    x = np.arange(len(fams))
    wbar = 0.26
    for j, (m, col) in enumerate(methods):
        vals = [[tab[m][n] for n in tab["_levels"][f]] for f in fams]
        mean = np.array([np.mean(v) for v in vals])
        lo = np.array([np.min(v) for v in vals])
        hi = np.array([np.max(v) for v in vals])
        xx = x + (j - 1) * (wbar + 0.03)
        bx.bar(xx, mean, wbar, color=col, ec="white", lw=0.5,
               label=f"{m}  (clean {tab['_clean'][m]:.3f}, robustness {rob[m][0]:.3f})")  # fmt: skip
        bx.vlines(xx, lo, hi, color=INK, lw=0.6)
    bx.set_xticks(x)
    bx.set_xticklabels([FAMILIES[f] for f in fams], rotation=28, ha="right", fontsize=7.5)
    bx.set_ylim(0, 1.0)
    bx.set_ylabel("object F1 (mean over levels)")
    bx.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.27),
        ncol=3,
        fontsize=7,
        handlelength=1.0,
        columnspacing=1.2,
        borderaxespad=0,
    )
    bx.set_title("unseen measurement artefacts, 200 scenes (whiskers: min–max over levels)",
                 color=MUTED, fontsize=8.5)  # fmt: skip
    bx.grid(axis="y", color=FAINT, lw=0.5)
    bx.set_axisbelow(True)
    tag(bx, "b", -0.06)
    save(fig, "detection")


# =================================================================== 7. virtual gates
def fig_virtual_gates() -> None:
    """(a) reference constellation, (b) one later point: only the predicted patches/strips,
    (c) prediction error of t(b) before each registration, (d) where the cube is measured."""
    from stage2_optimizer.virtual_gates import clean

    exp, run = example_run()
    vg = run.vg
    ref = clean(vg.ref_img, vg.ref_win.step)
    s = np.percentile(np.abs(ref), 99.5)

    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 0.95], wspace=0.3,
                          hspace=0.42)  # fmt: skip
    ax, bx = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cx = fig.add_subplot(gs[1, 0])
    dx = fig.add_axes([0.56, 0.05, 0.31, 0.36], projection="3d")

    # ---- (a) reference
    ax.imshow(-ref, extent=vg.ref_win.extent(), cmap="Greys", vmin=-0.3 * s, vmax=s,
              interpolation="nearest")  # fmt: skip
    for i, r in enumerate(vg.sticks.r):
        ax.add_patch(Circle(r, 0.009, fc="none", ec=RED, lw=0.9))
        ax.text(r[0] + 0.011, r[1] + 0.006, str(i), color=RED, fontsize=6.5)
    ax.set(xlabel="$g_2$ (V)", ylabel="$g_4$ (V)")
    ax.set_title("step 1: one scene at $b = 0$ $\\rightarrow$ sticks $r_i$", color=MUTED,
                 fontsize=8.5)  # fmt: skip
    tag(ax, "a", -0.13)

    # ---- (b) a registered point far in the cube: patches at r_i + t(b)
    grid_v = vg.visits[run.grid_visits]
    far = max((v for v in grid_v if v.registered and v.patches is not None),
              key=lambda v: np.linalg.norm(v.t))  # fmt: skip
    strip = max((v for v in grid_v if not v.registered), key=lambda v: np.linalg.norm(v.t))
    bx.set_facecolor(PAPER)
    for v, col in ((far, RED), (strip, BLUE)):
        for z, w in v.patches:
            bx.imshow(-z, extent=w.extent(), cmap="Greys", vmin=-0.3 * s, vmax=s,
                      interpolation="nearest", zorder=2)  # fmt: skip
            e = w.extent()
            bx.add_patch(Rectangle((e[0], e[2]), e[1] - e[0], e[3] - e[2], fc="none", ec=col,
                                   lw=0.8, zorder=3))  # fmt: skip
    k = 0
    arrow(bx, vg.sticks.r[k], vg.sticks.r[k] + far.t, color=RED, lw=1.0, ms=7, zorder=4)
    bx.plot(*vg.sticks.r.T, "o", ms=2.2, color=MUTED, zorder=4)
    allp = np.vstack([vg.sticks.r, vg.sticks.r + far.t, vg.sticks.r + strip.t])
    lo, hi = allp.min(0) - 0.04, allp.max(0) + 0.04
    span = (hi - lo).max()
    lo[0] -= span - (hi - lo)[0]  # square window
    bx.set(xlim=(lo[0], lo[0] + span), ylim=(lo[1], lo[1] + span), xlabel="$g_2$ (V)",
           ylabel="$g_4$ (V)")  # fmt: skip
    bx.set_aspect("equal")
    fb = ", ".join(f"{1e3 * x:+.0f}" for x in far.b)
    sb = ", ".join(f"{1e3 * x:+.0f}" for x in strip.b)
    bx.plot([], [], "s", mfc="none", mec=RED, label=f"40 mV patches\n$b$ = ({fb}) mV")
    bx.plot([], [], "s", mfc="none", mec=BLUE, label=f"40×12 mV strips\n$b$ = ({sb}) mV")
    bx.plot([], [], "o", ms=2.5, color=MUTED, label="reference $r_i$")
    bx.set_anchor("W")
    bx.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), ncol=1, fontsize=6.5, handlelength=1.0, borderaxespad=0.3,
              facecolor="white", frameon=True, edgecolor=FAINT, framealpha=0.95)  # fmt: skip
    n_scene = round(0.3 / 2e-3) ** 2
    n_far = sum(z.size for z, _ in far.patches)
    bx.set_title(f"then: measure only at $r_i + t(b)$ ({n_far} px vs {n_scene} / scene)",
                 color=MUTED, fontsize=8.5)  # fmt: skip
    tag(bx, "b", -0.13)

    # ---- (c) prediction error before each registration
    reg = [v for v in vg.visits if v.registered and v.ok]
    err = np.array([1e3 * np.hypot(*(v.t - v.t_pred)) for v in reg])
    n_cal = 15
    idx = np.arange(1, len(err) + 1)
    cx.axvspan(0.5, n_cal + 0.5, color=LIGHT_RED, lw=0)
    cx.text(n_cal / 2 + 0.5, 0.036, "calibration\n(steps 2–3)", ha="center", va="bottom",
            fontsize=7, color=RED)  # fmt: skip
    cx.text(n_cal + 4, 30, "exploration (step 4)", ha="left", va="top", fontsize=7, color=MUTED)
    cx.plot(idx, np.maximum(err, 0.03), "o-", ms=3, lw=0.8, color=INK, mfc="white", mew=0.8)
    cx.axhline(2.0, color=BLUE, lw=0.8, ls="--")
    cx.text(len(err), 6.0, "1 pixel = 2 mV", color=BLUE, ha="right", va="bottom", fontsize=7,
            bbox={"fc": "white", "ec": "none", "pad": 0.5})  # fmt: skip
    cx.set_yscale("log")
    cx.set(xlabel="registered measurement", ylabel="$|t_{\\mathrm{meas}} - t_{\\mathrm{pred}}|$ (mV)",
           ylim=(0.03, 40), xlim=(0.5, len(err) + 0.5))  # fmt: skip
    cx.set_title(f"$t(b)$ predicts each new point to "
                 f"{np.median(err[n_cal:]):.1f} mV (median)", color=MUTED, fontsize=8.5)  # fmt: skip
    tag(cx, "c", -0.13)

    # ---- (d) barrier cube
    from csd import BARRIERS
    from stage2_optimizer import optimize_vg as opt

    pts, registered = opt.grid_order(5)
    centres = np.arange(len(pts)) >= 125
    for mask, col, lab, ms in ((registered, RED, "registered (patches)", 9),
                               (~registered & ~centres, BLUE, "read only (strips)", 6),
                               (centres, GRAY, "cell centres (strips)", 4)):  # fmt: skip
        dx.scatter(*pts[mask].T, s=ms, c=col, depthshade=False, label=lab, lw=0)
    rev = exp.reveal()
    b_star = np.array([rev["optimum_barriers"][g] for g in BARRIERS])
    for c in run.candidates:
        dx.plot(*np.c_[c.b_grid, c.b], color=OCHRE, lw=1.0)
    dx.scatter(*np.array([c.b for c in run.candidates]).T, s=14, c=OCHRE, marker="D",
               depthshade=False, label="refined candidates", lw=0)  # fmt: skip
    dx.scatter(*b_star, s=70, marker="*", c="white", edgecolors=INK, lw=0.8, depthshade=False,
               label="hidden optimum")  # fmt: skip
    dx.set_xlabel("$g_1$ (V)", labelpad=-6, fontsize=8)
    dx.set_ylabel("$g_3$ (V)", labelpad=-6, fontsize=8)
    dx.set_zlabel("$g_5$ (V)", labelpad=-7, fontsize=8)
    for a in (dx.xaxis, dx.yaxis, dx.zaxis):
        a.set_ticks([-0.4, 0, 0.4])
        a.set_tick_params(labelsize=6.5, pad=-2)
        a.pane.set_facecolor("white")
        a.pane.set_edgecolor(FAINT)
        a._axinfo["grid"]["color"] = FAINT
        a._axinfo["grid"]["linewidth"] = 0.4
    dx.view_init(elev=18, azim=-58)
    dx.set_box_aspect((1, 1, 1), zoom=0.9)
    dx.legend(loc="center left", bbox_to_anchor=(1.08, 0.5), fontsize=6.5, handletextpad=0.2,
              borderaxespad=0)  # fmt: skip
    y_title = cx.get_position().y1
    fig.text(0.585, y_title + 0.006, "steps 4–6: 189 points of the barrier cube", color=MUTED,
             fontsize=8.5, va="bottom")  # fmt: skip
    fig.text(0.578, y_title + 0.004, "(d)", fontsize=10, fontweight="bold", ha="right",
             va="bottom")  # fmt: skip
    save(fig, "virtual_gates")


# ===================================================================== 8. refine/answer
def fig_refine() -> None:
    """(a) 1-D sweeps of the winning candidate, (b) ranking of the candidates by gain."""
    from csd import BARRIERS

    exp, run = example_run()
    rev = exp.reveal()
    b_star = np.array([rev["optimum_barriers"][g] for g in BARRIERS])
    best = max(run.candidates, key=lambda c: c.gain)

    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.75], wspace=0.42)
    sweeps = [fig.add_subplot(gs[i]) for i in range(3)]
    gx = fig.add_subplot(gs[3])

    shades = ["#e7b9ad", "#d6826f", RED]
    for k, ax in enumerate(sweeps):
        scans = [s for s in best.scans if s[0] == k]
        for n, (_, xs, ys, vertex) in enumerate(scans):
            ax.plot(1e3 * xs, ys, "o-", ms=2.6, lw=0.8, color=shades[n],
                    label=f"pass {n + 1}" if k == 0 else None)  # fmt: skip
        ax.axvline(1e3 * scans[-1][3], color=INK, lw=0.8)
        ax.axvline(1e3 * b_star[k], color=OCHRE, lw=1.0, ls="--")
        ax.set_xlabel(f"barrier $g_{BARRIERS[k][1]}$ (mV)")
        if k == 0:
            ax.set_ylabel("stick depth (a.u.)")
            ax.legend(loc="lower center", fontsize=6.3, handlelength=1.0, borderaxespad=0.1)
        ax.tick_params(labelsize=7)
    sweeps[0].set_title("step 6: sweep one barrier at a time, parabola vertex", color=MUTED,
                        fontsize=8.5)  # fmt: skip
    sweeps[2].plot([], [], color=INK, lw=0.8, label="answer")
    sweeps[2].plot([], [], color=OCHRE, lw=1.0, ls="--", label="hidden optimum")
    sweeps[2].legend(loc="lower center", fontsize=6.3, handlelength=1.4, borderaxespad=0.1)
    tag(sweeps[0], "a", -0.2)

    # ---- (c) candidates: gain = C / C0
    cands = list(run.candidates)
    order = sorted(range(len(cands)), key=lambda n: -cands[n].gain)
    y = np.arange(len(order))[::-1]
    for yy_, n in zip(y, order):
        c = cands[n]
        short = np.isfinite(c.floor)
        gx.barh(yy_, c.gain if short else 0, 0.62, color=RED if c is best else "#e7b9ad")
        txt = (f"$C$ = {c.contrast:.1f}, $C_0$ = {c.floor:.1f}" if short
               else f"$C$ = {c.contrast:.1f} < 0.75 max: dropped")  # fmt: skip
        gx.text((c.gain if short else 0) + 0.3, yy_, txt, va="center", fontsize=6.5, color=INK)
    gx.set_yticks(y)
    gx.set_yticklabels([f"{chr(65 + n)} · stick {cands[n].i}" for n in order], fontsize=7)
    gx.set_xlabel("gain $C / C_0$")
    gx.set_xlim(0, 22)
    gx.set_title("step 7: rank by gain over the stick's floor", color=MUTED, fontsize=8.5)
    gx.tick_params(axis="y", length=0)
    gx.spines["left"].set_visible(False)
    tag(gx, "b", -0.25)
    save(fig, "refine")


# ========================================================================= 9. results
def fig_results() -> None:
    """Quality reached vs pixels measured on the 60 test devices, three modes and the starter."""
    comp = read_csv("test_seeds_1000_1059_compressed.csv")
    pat = read_csv("test_seeds_1000_1059_patches.csv")
    full = read_csv("test_seeds_1000_1059.csv")

    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.25)
    ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    series = [
        ("starter (coordinate ascent)", [r["baseline_pixels"] for r in full],
         [r["baseline"] for r in full], GRAY),
        ("scene at every grid point", [r["n_pixels"] for r in full], [r["quality"] for r in full],
         BLUE),
        ("scenes, then patches", [r["n_pixels"] for r in pat], [r["quality"] for r in pat], OCHRE),
        ("compressed (default)", [r["n_pixels"] for r in comp], [r["quality"] for r in comp], RED),
    ]  # fmt: skip
    LABEL_AT = {
        "starter (coordinate ascent)": (3.2, 0.24),
        "scene at every grid point": (3.7, 0.5),
        "scenes, then patches": (2.4, 0.8),
        "compressed (default)": (0.85, 0.66),
    }
    rng = np.random.default_rng(0)
    for name, px, q, col in series:
        px, q = np.array(px) / 1e6, np.array(q)
        ax.scatter(px * np.exp(rng.normal(0, 0.012, len(px))), q, s=11, color=col, lw=0.4,
                   ec="white", zorder=3)  # fmt: skip
        ax.plot(np.median(px), np.median(q), marker="o", ms=8, mfc=col, mec=INK, mew=0.9,
                zorder=4, ls="none")  # fmt: skip
        ax.annotate(f"{name}\nmedian {np.median(q):.3f}, {np.median(px):.2f} M px",
                    (np.median(px), np.median(q)), xytext=LABEL_AT[name], ha="center",
                    va="center", fontsize=6.6, color=INK,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": MUTED, "shrinkB": 5})  # fmt: skip
    ax.set_xscale("log")
    ax.set(xlabel="pixels measured per device (millions)", ylabel="quality = $C(b)/C_{\\max}$",
           ylim=(0, 1.05), xlim=(0.4, 12))  # fmt: skip
    ax.set_xticks([0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10"])
    ax.grid(color=FAINT, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_title("60 unseen devices (seeds 1000–1059)", color=MUTED, fontsize=8.5)
    tag(ax, "a", -0.1)

    q = np.array([r["quality"] for r in comp])
    qb = np.array([r["baseline"] for r in full])
    bins = np.linspace(0, 1, 41)
    bx.hist(qb, bins, color=GRAY, ec="white", lw=0.4, label=f"starter (median {np.median(qb):.3f})")
    bx.hist(q, bins, color=RED, ec="white", lw=0.4, label=f"ours (median {np.median(q):.3f})")
    bx.set(xlabel="quality", ylabel="devices", xlim=(0, 1.02))
    bx.legend(loc="upper center", fontsize=7, handlelength=1.0)
    bx.text(0.5, 0.62, f"quality $\\geq$ 0.95: {np.mean(q >= 0.95):.0%} of devices\n"
            f"worst device: {q.min():.3f}", transform=bx.transAxes, ha="center", fontsize=7,
            color=INK)  # fmt: skip
    bx.set_title("quality distribution", color=MUTED, fontsize=8.5)
    tag(bx, "b", -0.12)
    save(fig, "results")


# ============================================================================ 0. hero
def fig_hero() -> None:
    """Raw diagram at the start, its detected sticks, and the diagram at the answer."""
    from stage1_linefilter.detector import predict

    exp, run = example_run()
    step = 1e-3
    g = run.final["gates"]
    t = run.vg.predict(run.final["b"])
    c0 = np.array([exp.start["g2"], exp.start["g4"]])
    img0 = exp.measure(g2=c0[0], g4=c0[1], span_h=0.3, span_v=0.3, step_h=step, step_v=step)
    img1 = exp.measure(g1=g["g1"], g3=g["g3"], g5=g["g5"], g2=c0[0] + t[0], g4=c0[1] + t[1],
                       span_h=0.3, span_v=0.3, step_h=step, step_v=step)  # fmt: skip
    rows0 = img0 - np.median(img0, axis=1, keepdims=True)
    sim = exp._sim  # ground-truth contrast factors, for the caption only
    f0 = sim.contrast(dict(exp.start))
    f1 = sim.contrast({**exp.start, **g})
    vmin, vmax = -7, 3  # same colour scale for the two raw images (deep sticks saturate)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    fig.subplots_adjust(wspace=0.06, left=0.0, right=1.0, top=0.84, bottom=0.02)
    axes[0].imshow(img0, cmap=CSD_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].imshow(rows0, cmap="Greys_r", vmin=-4, vmax=3, interpolation="nearest", alpha=0.6)
    overlay(axes[1], predict(img0) > 0, RED, 1.0)
    axes[2].imshow(img1, cmap=CSD_CMAP, vmin=vmin, vmax=vmax, interpolation="nearest")
    titles = [
        ("1 · raw charge-stability diagram", f"start point $b = 0$: best contrast ×{f0:.1f}"),
        ("2 · interdots detected", "classical line filter, no training"),
        ("3 · at the gates found", f"same colour scale: best contrast ×{f1:.1f}"),
    ]  # fmt: skip
    for ax, (a, b) in zip(axes, titles):
        frame(ax, FAINT)
        ax.set_title(a, fontsize=9, loc="center", pad=12)
        ax.text(0.5, 1.02, b, transform=ax.transAxes, ha="center", va="bottom", fontsize=7.2,
                color=MUTED)  # fmt: skip
    save(fig, "hero")


# ============================================================================ main
FIGURES = {
    "hero": fig_hero,
    "device": fig_device,
    "csd": fig_csd,
    "problem": fig_problem,
    "pipeline": fig_pipeline,
    "linefilter": fig_linefilter,
    "detection": fig_detection,
    "virtual_gates": fig_virtual_gates,
    "refine": fig_refine,
    "results": fig_results,
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("names", nargs="*", choices=[*FIGURES, []], help="figures (default: all)")
    args = p.parse_args()
    setup_style()
    for name in args.names or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main()
