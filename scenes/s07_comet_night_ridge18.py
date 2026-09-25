"""Round-18 ridgeline painting for s07_comet_night: crisp afterglow rim light on the crests.

warm_rim: the afterglow sits behind the ranges, so the crests that face it catch a thin, crisp, hot
orange rim (thicker and brighter near the glow, broken by a brushy along-ridge variation, stronger on the
slope that faces the glow).  Also lays a cool violet wash into the shadowed lower planes so the lit rim reads
against flat cool shadow instead of smooth clay shading.
Input / output: straight RGBA band plate (rows of the land band), float32.
"""
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def warm_rim(pl, sun_x, Wp, H, s, amt=1.0, width=2.2, seed=0, violet=0.12):
    h, w = pl.shape[:2]
    a = np.clip(pl[..., 3], 0, 1)
    # distance below the silhouette top (px), per column, via shifted alpha differences
    k = max(width * s, 1.0)
    rim = np.zeros_like(a)
    nstep = int(np.ceil(3 * k)) + 1
    for i in range(1, nstep + 1):
        ab = np.zeros_like(a)
        ab[i:] = a[:-i]
        edge = np.clip(a - ab, 0, 1)            # transparent i px above -> within i px of the top edge
        rim = np.maximum(rim, edge * np.exp(-((i - 1) / k) ** 2))
    # slope facing the glow: silhouette top descends toward the sun -> brighter
    xs = np.arange(w, dtype=np.float32)
    top = np.argmax(a > 0.5, axis=0).astype(np.float32)
    top[a.max(0) < 0.5] = h
    tsm = cv2.GaussianBlur(top[None, :], (0, 0), 6 * s + 1)[0]
    slope = np.gradient(tsm)                    # +: surface drops to the right
    toward = np.sign(sun_x - xs)                # +: sun to the right
    face = _ss(-0.4, 0.8, -slope * toward)      # rising toward the sun = facing it
    prox = np.exp(-((xs - sun_x) / (0.38 * Wp)) ** 2)
    rng = np.random.default_rng(seed)
    nz = cv2.resize(rng.random((1, max(w // 24, 4))).astype(np.float32), (w, 1), interpolation=cv2.INTER_CUBIC)[0]
    brk = 0.55 + 0.6 * _ss(0.25, 0.75, nz)
    wgt = (0.25 + 0.75 * prox) * (0.35 + 0.65 * face) * brk
    rim = rim * wgt[None, :] * amt
    rgb = pl[..., :3]
    col = np.array([1.25, 0.58, 0.3], np.float32)
    rgb = rgb * (1 - np.clip(rim, 0, 1)[..., None] * 0.6) + col * rim[..., None]
    # cool flat violet wash in the lower shadow planes (depth below the crest)
    ys = np.arange(h, dtype=np.float32)[:, None]
    dep = np.clip((ys - tsm[None, :]) / (0.08 * H), 0, 1)
    lum = rgb.max(-1)
    sh = (dep * (1 - _ss(0.35, 0.7, lum)) * violet)[..., None]
    rgb = rgb * (1 - sh) + np.array([0.2, 0.2, 0.42], np.float32) * sh
    out = pl.copy()
    out[..., :3] = rgb
    return out
