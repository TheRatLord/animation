"""Round-10 ridge lighting for s07_comet_night.

ridge_light8: post-pass on a range plate (straight RGBA band):
  * a thin, crisp, high-value cyan-white rim along the skyline where the ridge faces the comet (with an HDR
    core so the bloom catches it), broken along its length like a painted highlight,
  * small specular snow glints on the lit crests and a sparse few on bright snow faces,
  * the ridge's shadow side (faces turned away from the comet) falls to a darker violet, and the whole
    plate gets a deeper violet shadow value so the range silhouettes against the warm horizon.
"""
import math

import numpy as np
import cv2

from lib import core as C


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def skyline(a):
    """First row with alpha > 0.5 per column (Hh if none)."""
    m = a > 0.5
    has = m.any(0)
    top = np.where(has, m.argmax(0), a.shape[0]).astype(np.float32)
    return top, has


def ridge_light8(rgba, light_xy, Wp, H, s, seed=0, rim_amt=1.0, shadow_amt=0.5, glints=40, flat=0.45,
                 rim_col=(0.86, 0.98, 1.12), vio=(0.3, 0.22, 0.55), contrast=0.5, depth_frac=0.06):
    out = rgba.copy()
    a = rgba[..., 3]
    Hh, Ww = a.shape
    top, has = skyline(a)
    if not has.any():
        return out
    xs = np.arange(Ww, dtype=np.float32)
    # smooth the skyline a little for the slope, keep the crisp one for the rim
    ts = cv2.GaussianBlur(np.where(has, top, np.nan_to_num(top)).reshape(1, -1), (0, 0), 2.0 * s + 0.7)[0]
    slope = np.gradient(ts)
    nx, ny = slope, -np.ones_like(slope)
    nn = np.sqrt(nx * nx + ny * ny)
    nx, ny = nx / nn, ny / nn
    lx = light_xy[0] - xs
    ly = (light_xy[1] - top) * flat                      # flattened: side-facing matters more
    ln = np.sqrt(lx * lx + ly * ly) + 1e-3
    facing = np.clip((nx * lx + ny * ly) / ln, -1, 1)
    facing = cv2.GaussianBlur(facing.reshape(1, -1).astype(np.float32), (0, 0), 3.0 * s + 1.0)[0]
    lit = _ss(0.35, 0.8, facing)
    shade = _ss(0.25, -0.2, facing)                        # faces turned away
    # no rim on the lowest skyline stretches (the valley floor / shore edge of a plate)
    low = _ss(Hh - 0.02 * H, Hh - 0.06 * H, top)
    lit = lit * low
    rng = np.random.default_rng(seed)
    # break the rim along its length (painted highlight, not a tube)
    br = C.fbm(Ww, 4, Ww / (0.015 * Wp), 3, seed=seed + 3)[2]
    brk = 0.35 + 0.65 * _ss(0.35, 0.6, br)
    near = 0.55 + 0.45 * np.exp(-np.abs(xs - light_xy[0]) / (0.5 * Wp))
    ys = np.arange(Hh, dtype=np.float32)[:, None]
    dd = ys - top[None, :]                                  # depth below the skyline
    inside = (dd >= 0) & has[None, :]
    rw = (1.1 * s + 0.4) * (0.7 + 0.6 * C.fbm(Ww, 4, Ww / (0.03 * Wp), 2, seed=seed + 5)[2])[None, :]
    rim = _ss(rw * 1.9, rw * 0.7, dd) * inside * a
    rim_w = (lit * brk * near)[None, :] * rim * rim_amt
    rc = np.asarray(rim_col, np.float32)
    # the lit side just under the rim: a narrow pale band (snow catching the light), lost downward
    band = np.exp(-np.clip(dd, 0, None) / (5.0 * s + 1.0)) * inside * a * (lit * near)[None, :] * 0.25 * rim_amt
    rgb = out[..., :3]
    rgb = rgb * (1 - np.clip(band, 0, 1)[..., None]) + rc * 0.8 * band[..., None]
    rgb = rgb * (1 - np.clip(rim_w, 0, 0.95)[..., None]) + rc * np.clip(rim_w, 0, 0.95)[..., None]
    rgb += (rim_w * _ss(rw * 1.1, rw * 0.3, dd))[..., None] * rc * 0.6          # HDR core
    # shadow side: darker violet, strongest near the crest, falling off down the face
    D = depth_frac * H
    sh = cv2.GaussianBlur((shade[None, :] * np.exp(-np.clip(dd, 0, None) / D) * inside).astype(np.float32),
                          (0, 0), 4.0 * s + 1.0) * a * shadow_amt
    vc = np.asarray(vio, np.float32)
    rgb = rgb * (1 - sh[..., None] * 0.6) + vc * rgb.mean(-1, keepdims=True) * sh[..., None] * 0.6 * 1.6
    # plate-wide value structure: push the darker half of the plate toward a deep violet
    lum = rgb.mean(-1)
    msk = a > 0.5
    if msk.any():
        med = float(np.median(lum[msk]))
        dk = _ss(med * 1.05, med * 0.55, lum) * contrast * a
        rgb = rgb * (1 - dk[..., None]) + (rgb * np.array([0.72, 0.62, 0.95], np.float32) * 0.7) * dk[..., None]
    # specular glints on lit crests
    cand = np.nonzero((lit * brk > 0.6) & has)[0]
    if len(cand) and glints:
        pick = rng.choice(cand, min(glints, len(cand)), replace=False)
        for x in pick:
            y = top[x] + rng.uniform(0.5, 2.5) * s
            it = rng.uniform(0.4, 1.2)
            C.splat(rgb, float(x), float(y), 0.7 * s + 0.3, rc, 1.6 * it)
            C.splat(rgb, float(x), float(y), 3.0 * s + 0.5, rc, 0.08 * it)
    # a few glints on bright snow faces
    snow = (lum > np.percentile(lum[msk], 97) if msk.any() else lum > 1) & msk
    sy, sx = np.nonzero(snow)
    if len(sx) and glints:
        k = rng.choice(len(sx), min(glints // 2, len(sx)), replace=False)
        for i in k:
            C.splat(rgb, float(sx[i]), float(sy[i]), 0.6 * s + 0.3, rc, rng.uniform(0.5, 1.1))
    out[..., :3] = rgb
    return out


def twilight8(Wp, Hp, y_h, H, xs, ys, x_warm, seed=0):
    """Continuous kataware-doki grade behind the ranges: hot gold/orange right at the ridge line, through
    peach and pink into violet, with a soft, noise-free upper transition into the navy (no stripe).
    Returns (target colour (Hp, Wp, 3), mix (Hp, Wp))."""
    h = np.clip(y_h - ys, 0, None) / H
    n = C.fbm(max(Wp // 16, 8), max(Hp // 16, 8), 2.5, 3, seed=seed)
    n = cv2.resize(n, (Wp, Hp), interpolation=cv2.INTER_CUBIC)
    hh = h * (1 + 0.1 * (n - 0.5))
    stops = [(0.0, (1.0, 0.6, 0.24)), (0.1, (1.0, 0.58, 0.26)), (0.15, (1.0, 0.56, 0.32)),
             (0.2, (0.98, 0.58, 0.46)), (0.26, (0.82, 0.46, 0.56)), (0.33, (0.54, 0.31, 0.58)),
             (0.43, (0.32, 0.2, 0.5)), (0.6, (0.2, 0.15, 0.42))]
    xp = np.array([p for p, _ in stops], np.float32)
    T = np.dstack([np.interp(hh, xp, np.array([c[k] for _, c in stops], np.float32)) for k in range(3)])
    gx = np.exp(-((xs - x_warm) / (0.5 * Wp)) ** 2)
    cool = np.array([0.36, 0.26, 0.56], np.float32)
    # away from the warm side the glow cools to violet (warm/cool split along the horizon)
    T = T * gx[..., None] + (T * 0.35 + cool * 0.65) * (1 - gx[..., None])
    M = 0.95 * (1 - _ss(0.08, 0.58, hh)) ** 1.3 * (0.55 + 0.45 * gx)
    M = M * (ys <= y_h + 2)
    return T.astype(np.float32), M.astype(np.float32)
