"""Round-21 ridge painting for s07_comet_night (reviewer: soft mushy ridgelines, blotchy snow faces, far range
flat lilac triangles, no lifted mist in front of the ridge bases).

harden   : re-cuts the silhouette of a range plate into a crisp ~1 px anti-aliased edge (the earlier paint
           passes blurred the alpha 2-3 px and left a glowing halo along the crest).
couloirs : painted snow structure - thin crisp gullies running down the fall line from just under the crest,
           tapering and ending at different depths, with lit snow ribs between them on the faces that catch
           the afterglow / comet light and deeper violet-blue in the gullies on the shadow faces.
lifted_mist : one soft horizontal band of lifted valley mist lying across the bases of the ranges (yn_01),
           lit warm near the afterglow and cool toward the comet, airbrushed top and bottom.
All operate on straight-RGBA band plates (rows b0:Hl of the land band), float32.
"""
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise1d(n, cell, rng):
    k = max(int(n / cell) + 3, 4)
    v = rng.standard_normal(k).astype(np.float32)
    return cv2.resize(v[None, :], (int(k * cell), 1), interpolation=cv2.INTER_CUBIC)[0][:n + 1]


def _top(a, s, sig=2.0):
    h = a.shape[0]
    top = np.argmax(a > 0.5, axis=0).astype(np.float32)
    top[a.max(0) < 0.5] = h
    return top, cv2.GaussianBlur(top[None, :], (0, 0), sig * s + 1)[0]


def crisp(pl, amt=0.6, sigma=2.0, s=1.0):
    """Unsharp mask on the premultiplied colour: crisper painted value edges / crest, no halo."""
    a = np.clip(pl[..., 3:4], 0, 1)
    pm = pl[..., :3] * a
    sg = sigma * s + 0.3
    bl = cv2.GaussianBlur(pm, (0, 0), sg) / np.maximum(cv2.GaussianBlur(a, (0, 0), sg)[..., None], 1e-4)
    rgb = pl[..., :3]
    out = pl.copy()
    out[..., :3] = np.clip(rgb + amt * (rgb - bl) * (a > 0.02), 0, None)
    return out


def harden(pl, gain=3.5):
    out = pl.copy()
    a = np.clip(pl[..., 3], 0, 1)
    out[..., 3] = np.clip((a - 0.5) * gain + 0.5, 0, 1) * _ss(0.02, 0.15, a)
    return out


def couloirs(pl, s, H, Wp, sun_x, light_x, seed=0, amt=1.0, depth=0.07, period=16.0, lit_amt=1.0):
    h, w = pl.shape[:2]
    rgb = pl[..., :3].copy()
    a = np.clip(pl[..., 3], 0, 1)
    top, tsm = _top(a, s, 2.0)
    tsl = cv2.GaussianBlur(top[None, :], (0, 0), 14 * s + 1)[0]
    slope = np.clip(np.gradient(tsl), -2.5, 2.5)          # +: surface drops to the right
    rng = np.random.default_rng(seed)
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    d = np.clip(ys - tsm[None, :], 0, None)                # depth below the crest (px)
    # fall line: gullies lean down-slope (away from the crest line)
    xf = xs + np.sign(slope)[None, :] * np.minimum(np.abs(slope), 1.2)[None, :] * d * 0.55
    xf = xf + _noise1d(h, 30 * s + 4, rng)[:h, None] * 3.0 * s       # hand-drawn wobble
    per = period * s + 2
    n1 = _noise1d(w + 400, per, rng)
    n2 = _noise1d(w + 400, per * 0.45, rng)
    xi = np.clip(xf + 200, 0, w + 399)
    g1 = 1 - np.abs(np.interp(xi, np.arange(len(n1)), n1))
    g2 = 1 - np.abs(np.interp(xi, np.arange(len(n2)), n2))
    # per-gully length: gullies end at different depths
    ln = np.interp(xi, np.arange(len(n1)), _noise1d(w + 400, per * 1.7, rng)[:len(n1)])
    D = depth * H * np.clip(0.7 + 0.45 * ln, 0.25, 1.4)
    env = _ss(1.5 * s, 6 * s, d) * (1 - _ss(0.35, 1.0, d / D))
    thr = 0.76 + 0.2 * _ss(0.0, 1.0, d / D)                 # V-shaped: wide at the notch, tapering down
    gul = _ss(thr, thr + 0.035, g1) * env
    gul2 = _ss(thr + 0.08, thr + 0.11, g2) * env * 0.45 * _ss(0.3, 0.6, ln)
    gul = np.clip(gul + gul2 * (1 - gul), 0, 1)
    rib = _ss(0.25, 0.55, 1 - g1) * env * (1 - gul)
    # faces: lit toward the afterglow (sun_x) and toward the comet (light_x)
    tw = np.sign(sun_x - xs[0])
    face_sun = _ss(-0.2, 0.6, -slope * tw)
    tc = np.sign(light_x - xs[0])
    face_cm = _ss(-0.2, 0.6, -slope * tc)
    lit = np.clip(0.6 * face_sun + 0.5 * face_cm, 0, 1)[None, :]
    lum = rgb.max(-1)
    snow = _ss(0.12, 0.35, lum)
    # gullies: cool violet-blue shadow cut into the snow (deeper on the shadow faces)
    gk = (gul * snow * amt * (0.55 + 0.45 * (1 - lit)))[..., None]
    gcol = rgb * np.array([0.66, 0.66, 0.84], np.float32) + np.array([0.0, 0.0, 0.02], np.float32)
    rgb = rgb * (1 - gk) + gcol * gk
    # lit snow ribs between them (cool-white on the comet side, warm on the afterglow side)
    warm = np.exp(-((xs[0] - sun_x) / (0.4 * Wp)) ** 2)[None, :]
    rc = np.array([0.1, 0.1, 0.13], np.float32) * (1 - warm[..., None]) + \
        np.array([0.16, 0.09, 0.07], np.float32) * warm[..., None]
    rk = (rib * snow * lit * 0.8 * lit_amt)[..., None]
    rgb = rgb + rk * rc * (0.6 + lum[..., None])
    out = pl.copy()
    out[..., :3] = rgb
    return out


def lifted_mist(Wp, Hb, base_rows, H, s, sun_x, light_x, seed=0, amt=0.5, lift=0.018, thick=0.016):
    """Straight RGBA band.  base_rows: per-column row (band coords) of the range bases (smoothed)."""
    rng = np.random.default_rng(seed)
    ys = np.arange(Hb, dtype=np.float32)[:, None]
    xs = np.arange(Wp, dtype=np.float32)
    yc = cv2.GaussianBlur(base_rows.astype(np.float32)[None, :], (0, 0), 0.06 * Wp)[0] - lift * H
    yc = yc + _noise1d(Wp, 0.1 * Wp, rng)[:Wp] * 0.004 * H
    th = thick * H * np.clip(1 + 0.35 * _noise1d(Wp, 0.07 * Wp, rng)[:Wp], 0.4, 1.7)
    dy = (ys - yc[None, :])
    # soft airbrushed top, a slightly firmer lifted underside
    prof = np.where(dy < 0, np.exp(-(dy / (1.2 * th[None, :])) ** 2), np.exp(-(dy / (0.7 * th[None, :])) ** 2))
    brk = 0.65 + 0.35 * _ss(-0.8, 0.8, _noise1d(Wp, 0.05 * Wp, rng)[:Wp])
    A = np.clip(prof * brk[None, :] * amt, 0, 1).astype(np.float32)
    warm = np.exp(-((xs - sun_x) / (0.33 * Wp)) ** 2)[None, :]
    cool = np.exp(-((xs - light_x) / (0.35 * Wp)) ** 2)[None, :]
    topk = _ss(0.5, -1.2, dy / th[None, :])                  # lighter on top (sky light)
    col = np.array([0.3, 0.34, 0.6], np.float32)[None, None, :] + \
        (warm * (0.4 + 0.6 * topk))[..., None] * np.array([0.34, 0.16, 0.1], np.float32) + \
        (cool * (0.4 + 0.6 * topk))[..., None] * np.array([0.05, 0.12, 0.18], np.float32) + \
        topk[..., None] * np.array([0.06, 0.07, 0.1], np.float32)
    return np.dstack([np.broadcast_to(col, (Hb, Wp, 3)), A]).astype(np.float32)
