"""s04_rain_street finishing passes (round-3 panel notes):
  * vp_tame   - the end-of-street glow becomes a coloured (magenta / cyan) haze with the far building silhouettes
                still readable, instead of a desaturated near-white blob.
  * fg_rain   - a handful of close, defocused streaks: thin, short, semi-transparent, lit only by the neon
                behind them (no constant white floor).
"""
import numpy as np
import cv2


def _vp_setup(sc):
    W, H = sc.W, sc.H
    cam = sc.cam
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    vx, vy = cam.cx + 0.005 * W, cam.cy + 0.01 * H
    # core: the far street + sky slot around the vanishing point; slightly taller than wide below the horizon
    # (wet road reflection of the same glow)
    d = ((xx - vx) / (0.13 * W)) ** 2 + ((yy - vy) / np.where(yy > vy, 0.16 * H, 0.2 * H)) ** 2
    core = np.exp(-d * 1.4)
    # the conbini spill on the right of the walker (white shop light) is part of the same blown patch
    d2 = ((xx - (vx + 0.11 * W)) / (0.05 * W)) ** 2 + ((yy - (vy - 0.02 * H)) / (0.1 * H)) ** 2
    spill = np.exp(-d2 * 1.6)
    # magenta on the left / centre, cool cyan toward the conbini on the right
    side = np.clip((xx - vx) / (0.12 * W) * 0.5 + 0.5, 0, 1)
    mag = np.float32([1.03, 0.98, 0.98])
    cya = np.float32([0.92, 1.00, 1.02])
    tint = mag * (1 - side[..., None]) + cya * side[..., None]
    sc._vpm = (np.maximum(core, spill * 0.8)[..., None].astype(np.float32), tint.astype(np.float32))


def vp_tame(sc, img):
    """Local exposure roll-off around the vanishing point (HDR, before bloom)."""
    if not hasattr(sc, '_vpm'):
        _vp_setup(sc)
    m, tint = sc._vpm
    lum = img.max(-1, keepdims=True)
    # soft rolloff: 0.5 -> 0.41, 1 -> 0.67, 2 -> 0.95 ... keeps building / figure silhouettes separated from
    # the glow behind them, while the glow itself stays below the white shoulder
    Lk = 1.45
    f = lum / (1.0 + lum / Lk)
    s = f / np.maximum(lum, 1e-4)
    # tint increases with brightness so the dark silhouettes stay neutral and the glow carries the colour
    w = np.clip(lum - 0.35, 0, 1.2) / 1.2
    col = 1.0 + (tint - 1.0) * w
    out = img * s * col
    img[:] = img * (1 - m) + out * m


def fg_rain_setup(sc):
    W, H = sc.W, sc.H
    rng = np.random.default_rng(4748)
    n = 22
    sc._fgr2 = dict(x=rng.uniform(-0.05 * W, 1.1 * W, n), ph=rng.uniform(0, 1, n),
                    L=rng.uniform(0.07, 0.13, n) * H, sp=rng.uniform(4.2, 5.2, n) * H,
                    wd=rng.uniform(1.0, 1.5, n) * H / 1080.0, a=rng.uniform(0.3, 0.6, n))


def fg_rain(sc, img, t, light):
    """Few close drops, defocused (soft, wide falloff but low opacity), coloured by the neon behind them."""
    if not hasattr(sc, '_fgr2'):
        fg_rain_setup(sc)
    W, H = sc.W, sc.H
    R_ = sc._fgr2
    slant = 0.09
    m = np.zeros((H, W), np.uint8)
    for x, ph, L, sp, wd, a in zip(R_['x'], R_['ph'], R_['L'], R_['sp'], R_['wd'], R_['a']):
        per = H + L
        y = (ph * per + sp * t) % per - L
        xs = x - slant * (y + L)
        p0 = (int(xs * 4), int(y * 4))
        p1 = (int((xs - slant * L) * 4), int((y + L) * 4))
        cv2.line(m, p0, p1, int(255 * a), max(1, int(round(wd))), cv2.LINE_AA, 2)
    mf = m.astype(np.float32) / 255.0
    mf = cv2.GaussianBlur(mf, (0, 0), 1.3 * H / 1080.0) * 1.6
    lum = light.max(-1, keepdims=True)
    chroma = light / np.maximum(lum, 1e-3)
    col = chroma * np.clip(lum * 1.5, 0.0, 1.3) * 0.7 + 0.015
    img += mf[..., None] * col
