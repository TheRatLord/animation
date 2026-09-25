"""Round-20 forest-ridge paint pass for s07_comet_night (midground + near shore ridges).

The judges read the forested ridges as felt/clay: the same frosty cotton-ball treeline fringe outlined every
ridge evenly.  Here the treeline detail is kept only where the silhouette catches sky light (a broken, slowly
varying stretch, strongest toward the afterglow), elsewhere it is painted back into the flat mass; the body
gets broad value masses and darkens into lost edges toward the lake (more contrast on the near ridge).
"""
import numpy as np
import cv2

from s07_comet_night_flat13 import _fill_bg


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _smooth1d(n, cells, rng):
    k = rng.random(cells + 3)
    xs = np.linspace(0, cells, n)
    v = np.interp(xs, np.arange(cells + 3), k)
    v = cv2.GaussianBlur(v.astype(np.float32)[None, :], (0, 0), n / cells * 0.35)[0]
    return (v - v.min()) / (v.max() - v.min() + 1e-6)


def paint_ridge(rgba, H, s, sun_x, seed=0, keep=0.6, dark=0.25, masses=0.07, rim=(0.0, 0.0, 0.0)):
    rgb, a = rgba[..., :3], rgba[..., 3]
    h, w = a.shape
    rng = np.random.default_rng(seed)
    m = (a > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    body = cv2.GaussianBlur(_fill_bg(rgb, a), (0, 0), 0.02 * H)
    xs = np.arange(w, dtype=np.float32)
    lightx = np.exp(-((xs - sun_x) / (0.35 * w)) ** 2)
    var = _smooth1d(w, 9, rng) * 0.7 + _smooth1d(w, 23, rng) * 0.3
    catch = np.clip(_ss(0.35, 0.7, var) * (0.45 + 0.55 * lightx) + 0.15 * lightx, 0, 1) * keep
    fringe = 1 - _ss(0.005 * H, 0.018 * H, dist)
    k = (fringe * (1 - catch[None, :]) * 0.85)[..., None]
    out = rgb * (1 - k) + body * k
    # a thin cool/warm light along the crest only where it catches the sky
    crest = (1 - _ss(0.0, 0.004 * H + 1.0, dist)) * catch[None, :]
    out = out + crest[..., None] * np.asarray(rim, np.float32)
    # broad painted value masses in the body + lost dark edges toward the lake
    n2 = cv2.resize(rng.standard_normal((6, 14)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    n2 = cv2.GaussianBlur(n2, (0, 0), 0.03 * H)
    deep = _ss(0.02 * H, 0.14 * H, dist)
    out = out * (1 + masses * np.clip(n2, -1.5, 1.5) * deep)[..., None]
    out = out * (1 - dark * deep)[..., None]
    return np.dstack([out, a]).astype(np.float32)
