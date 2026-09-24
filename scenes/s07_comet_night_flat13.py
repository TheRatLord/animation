"""Round-13 painted flattening for s07_comet_night land plates.

Judges: rock faces read as faceted low-poly planes with airbrushed bevel highlights, forest slopes as all-over
noise speckle, far range not aerial enough.  These post-passes repaint the existing plates:
  * ranges: mean-shift segmentation collapses the faceted shading into a few flat painted planes that follow
    the existing lit / shadow structure (snow planes stay crisp), then a light median cleans stray specks;
    back plates are pulled toward the sky haze colour with their value range compressed (aerial perspective),
  * forests: below a thin treeline band along each silhouette the stamped-tree speckle is replaced by a
    near-flat, gently graded value mass (the tree tips on the skyline stay as hand-placed edges).
"""
import numpy as np
import cv2

from lib import core as C


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _fill_bg(rgb, a):
    """Push colour outward into transparent pixels (no dark / grey fringe when filtering)."""
    w = (a > 0.5).astype(np.float32)
    out = rgb * w[..., None]
    for sg in (3, 9, 27):
        num = cv2.GaussianBlur(rgb * w[..., None], (0, 0), sg)
        den = cv2.GaussianBlur(w, (0, 0), sg)[..., None] + 1e-5
        out = np.where(w[..., None] > 0, out, np.where(den > 0.02, num / den, out))
        w = np.maximum(w, (den[..., 0] > 0.02).astype(np.float32))
    return out


def flatten_range(rgba, s, sp=9, sr=16, haze=None, haze_amt=0.0, contrast=1.0):
    rgb, a = rgba[..., :3], rgba[..., 3]
    f = _fill_bg(rgb, a)
    g = np.clip(f, 0, 1) ** (1 / 2.2)
    u8 = (g * 255 + 0.5).astype(np.uint8)
    ms = cv2.pyrMeanShiftFiltering(u8, max(int(sp * s), 3), sr, maxLevel=1)
    ms = cv2.medianBlur(ms, 3)
    out = (ms.astype(np.float32) / 255) ** 2.2
    # keep the thin bright crest rim of the original (crisp lit edge on the ridge line)
    rim = np.clip(rgb.max(-1) - out.max(-1), 0, None)
    top = _top_band(a, int(3 * s) + 2)
    out = out + (rgb - out) * (top * _ss(0.02, 0.1, rim))[..., None]
    if contrast != 1.0:
        m = (out * a[..., None]).sum((0, 1)) / (a.sum() + 1e-5)
        out = m + (out - m) * contrast
    if haze is not None:
        out = out * (1 - haze_amt) + np.asarray(haze, np.float32) * haze_amt
    return np.dstack([np.clip(out, 0, None), a]).astype(np.float32)


def _top_band(a, px):
    """1 within px rows below the local silhouette top (erosion difference, from above)."""
    m = (a > 0.5).astype(np.uint8)
    k = np.ones((px * 2 + 1, 1), np.uint8)
    # a pixel is in the top band if the pixel px rows above is outside
    sh = np.zeros_like(m)
    sh[px:] = m[:-px]
    return ((m > 0) & (sh == 0)).astype(np.float32)


def flatten_forest(rgba, H, s, keep_px=None, xmask=None, sigma=None):
    rgb, a = rgba[..., :3], rgba[..., 3]
    if keep_px is None:
        keep_px = 0.012 * H
    if sigma is None:
        sigma = 0.012 * H
    f = _fill_bg(rgb, a)
    # value mass: heavy blur, then a gentle quantisation into 3 soft bands (painted, near-flat)
    b = cv2.GaussianBlur(f, (0, 0), sigma)
    b2 = cv2.GaussianBlur(f, (0, 0), sigma * 3)
    b = (0.35 * b + 0.65 * b2) * 0.9
    # depth below the silhouette: distance to the transparent region measured upward
    m = (a > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    # soft treeline edge: the tips (first keep_px) keep their stamped detail, fading into the mass
    w = _ss(keep_px * 0.5, keep_px * 1.6, dist)
    if xmask is not None:
        w = w * xmask[None, :]
    out = rgb * (1 - w[..., None]) + b * w[..., None]
    return np.dstack([out, a]).astype(np.float32)
