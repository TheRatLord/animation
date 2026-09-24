"""Round-16 ridge lighting for s07_comet_night (reviewer: uniform glowing rim along every ridge top +
posterised dark snow blobs read as embossed).  Same contract as s07_comet_night_ridge8.ridge_light8, but:
the rim only lives on light-facing upper slopes near the crests and tapers away down the flanks (broken,
thinner), the lee side gets a lost (softened) skyline edge, and the dark rock / shade patches are flattened
into one cool shadow value instead of posterised dark blobs."""
import numpy as np
import cv2

from lib import core as C


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def skyline(a):
    m = a > 0.5
    has = m.any(0)
    top = np.where(has, m.argmax(0), a.shape[0]).astype(np.float32)
    return top, has


def ridge_light16(rgba, light_xy, Wp, H, s, seed=0, rim_amt=1.0, shadow_amt=0.5, glints=40, flat=0.45,
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
    lit = _ss(0.5, 0.9, facing)
    # taper: the rim lives on the upper slopes near each crest and dies away down the flanks
    kw = int(0.06 * Wp) | 1
    pk = cv2.erode(np.where(has, top, Hh).astype(np.float32).reshape(1, -1), np.ones((1, kw), np.uint8))[0]
    lit = lit * np.exp(-np.clip(top - pk, 0, None) / (0.035 * H))
    shade = _ss(0.25, -0.2, facing)                        # faces turned away
    # no rim on the lowest skyline stretches (the valley floor / shore edge of a plate)
    low = _ss(Hh - 0.02 * H, Hh - 0.06 * H, top)
    lit = lit * low
    rng = np.random.default_rng(seed)
    # break the rim along its length (painted highlight, not a tube)
    br = C.fbm(Ww, 4, Ww / (0.015 * Wp), 3, seed=seed + 3)[2]
    brk = 0.15 + 0.85 * _ss(0.4, 0.62, br)
    near = 0.55 + 0.45 * np.exp(-np.abs(xs - light_xy[0]) / (0.5 * Wp))
    ys = np.arange(Hh, dtype=np.float32)[:, None]
    dd = ys - top[None, :]                                  # depth below the skyline
    inside = (dd >= 0) & has[None, :]
    rw = (0.9 * s + 0.35) * (0.4 + 0.8 * lit[None, :]) * (0.7 + 0.6 * C.fbm(Ww, 4, Ww / (0.03 * Wp), 2, seed=seed + 5)[2])[None, :]
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
    # plate-wide value structure: the darker rock / shade patches become ONE flat cool shadow value
    # (painted plane), with a soft-ish boundary instead of posterised dark blobs
    lum = rgb.mean(-1)
    msk = a > 0.5
    if msk.any():
        lb = cv2.GaussianBlur(lum, (0, 0), 1.2 * s + 0.5)
        med = float(np.median(lum[msk]))
        dk = _ss(med * 1.0, med * 0.72, lb) * a
        dsel = (dk > 0.5) & msk
        if dsel.any():
            sc_ = np.median(rgb[dsel], axis=0) * np.array([0.92, 0.9, 1.08], np.float32)
            sc_ = sc_ * 0.5 + np.median(rgb[msk], axis=0) * np.array([0.55, 0.55, 0.8], np.float32) * 0.5
            k_ = (dk * contrast * 1.6).clip(0, 0.85)[..., None]
            rgb = rgb * (1 - k_) + sc_ * k_
    # lee side: lost edge (the skyline softens into the sky where the ridge faces away from the light)
    soft_e = _ss(0.0, 2.2 * s + 1.0, dd) * inside
    lee = (shade * low)[None, :]
    out[..., 3] = a * (1 - lee * 0.55 * (1 - soft_e))
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
