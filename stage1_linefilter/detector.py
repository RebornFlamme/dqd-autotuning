"""Stage-1 interdot detector, reused by the stage-2 optimisation.

No training, no weights: a classical pipeline whose six parameters (``PARAMS``) were tuned
by an automated search loop on a generated tuning set (``generate_dataset(seed=1)``) and
scored on a separate validation set (seed 2); results are reported on a third set (seed 2026).

Pipeline:
    1. per-row offset removal (scan-line drift), depth in noise units;
    2. seeds: pixels whose depth averaged along the stick direction (matched filter)
       is significant;
    3. around the seeds, keep pixels above half of their component's peak depth (FWHM);
    4. orientation filter: connected components whose local line direction (oriented
       filter bank) is not the sticks' (e.g. charge-transition fragments) are deleted.

Usage::

    from stage1_linefilter.detector import predict

    mask = predict(image)  # uint8 mask, same shape as the raw CSD image
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, correlate
from scipy.ndimage import label as ndlabel
from scipy.ndimage import maximum as ndmaximum
from skimage.measure import label, regionprops

STICK_THETA = np.pi / 4  # expected stick orientation [rad], image displayed with origin="lower"

# Parameters tuned on load_train() (grid / coordinate search).
PARAMS = {
    "k_line": 3.5,
    "line_len": 5,
    "dilate": 2,
    "k_low": 2.0,
    "half": 0.6,
    "k_peak": 3.0,
}
N_ORIENT = 8  # orientations of the filter bank used to check a component's direction
ORIENT_TOL = 1  # allowed deviation from STICK_THETA, in bins of pi / N_ORIENT


def robust_depth(a):
    """Dip depth in noise units: (median − a) / (1.4826·MAD)."""
    med = np.median(a)
    return (med - a) / (1.4826 * np.median(np.abs(a - med)))


def dip_mask(
    image,
    k_line=3.5,
    line_len=5,
    dilate=2,
    k_low=2.0,
    half=0.5,
    k_peak=3.5,
):
    """Predict an interdot mask from the dark dips of a CSD.

    Args:
        image: Row-corrected 2D CSD image (interdots are dark dips).
        k_line: Seed threshold (in MADs) on the depth averaged along the stick.
        line_len: Length [px] of the averaging along the stick (matched filter).
        dilate: Dilation of the seeds [px] before the depth thresholds.
        k_low: Minimum depth (in MADs) of kept pixels.
        half: Fraction of its component's amplitude a pixel must reach (~ FWHM).
        k_peak: Minimum amplitude (in MADs) of a kept component.

    Returns:
        Boolean mask with the same shape as ``image``.
    """
    depth = robust_depth(image)

    # Matched filter: averaging along the stick direction raises the SNR of faint
    # interdots (white noise averages out, the dip does not).
    along = robust_depth(correlate(image, np.eye(line_len) / line_len, mode="nearest"))
    seeds = along > k_line
    if dilate > 0:
        seeds = binary_dilation(seeds, iterations=dilate)
    cand = seeds & (depth > k_low)

    # An interdot dip has a roughly uniform amplitude along its length, while the
    # charge-transition lines attached to its ends are shallower and its thermal /
    # tunnel broadening forms tails across it: keep, in each component, the pixels
    # above ~half of its amplitude (full width at half maximum, in every direction).
    # The amplitude is the max of the depth averaged over 2 px along the stick: the
    # single-pixel max is biased upwards by the noise.
    labels, n = ndlabel(cand, structure=np.ones((3, 3)))
    smooth_depth = correlate(depth, np.eye(2) / 2, mode="nearest")
    amp = np.concatenate([[0.0], ndmaximum(smooth_depth, labels, np.arange(1, n + 1))])[labels]
    return cand & (depth > half * amp) & (amp > k_peak)


def line_kernel(theta, half_len):
    """Averaging kernel along a segment of angle ``theta`` (x-y, origin="lower").

    Args:
        theta: Segment orientation [rad].
        half_len: Half length of the segment [px] (sampled every 1 px, then rounded).

    Returns:
        Normalised 2D kernel (rows = y, cols = x).
    """
    kernel = np.zeros((2 * half_len + 1, 2 * half_len + 1))
    for t in range(-half_len, half_len + 1):
        kernel[half_len + round(t * np.sin(theta)), half_len + round(t * np.cos(theta))] = 1
    return kernel / kernel.sum()


def filter_by_orientation(mask, image, n_orient=N_ORIENT, tol=ORIENT_TOL, half_len=2):
    """Delete connected components whose local line direction is not the sticks'.

    Each component's direction is the darkest of ``n_orient`` oriented line averages of
    the image over its pixels (an oriented filter bank: robust even for components too
    small for a shape-based orientation). Fragments of charge-transition lines, which
    run in other directions, are rejected.

    Args:
        mask: Boolean mask (rows = y, cols = x, origin="lower").
        image: Row-corrected image the mask was computed from.
        n_orient: Number of orientations sampled over [0, pi).
        tol: Allowed deviation from ``STICK_THETA``, in orientation bins.
        half_len: Half length [px] of the oriented line averages.

    Returns:
        Filtered boolean mask.
    """
    thetas = np.arange(n_orient) * np.pi / n_orient
    resp = np.array([correlate(image, line_kernel(t, half_len), mode="nearest") for t in thetas])
    stick_bin = round(STICK_THETA / (np.pi / n_orient))
    out = mask.copy()
    for region in regionprops(label(mask, connectivity=2)):
        rr, cc = region.coords.T
        best = np.argmin(resp[:, rr, cc].mean(axis=1))
        if abs(best - stick_bin) > tol:
            out[rr, cc] = False
    return out


def predict(image):
    """Full pipeline: raw image -> binary interdot mask (uint8)."""
    image = np.asarray(image, dtype=float)
    # Charge-sensor drift between scan lines adds a per-row offset: remove it.
    image = image - np.median(image, axis=1, keepdims=True)
    mask = dip_mask(image, **PARAMS)
    mask = filter_by_orientation(mask, image)
    return mask.astype(np.uint8)
