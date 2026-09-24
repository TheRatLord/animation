"""Round-4 painted additions for s08_snow_station.

- wall_weather: visible painted wood grain (broad flowing grain bands, per-board tone, knots) and drip
  stains running down from the eave / sills for the waiting-room walls (world-space, per pixel).
- soft_night_clouds: a low overcast snow sky painted as a few very large soft value masses with smooth
  gradients (no scalloped stamp edges), a faint warm sodium glow on the undersides above the lit town /
  station and a subtle cool moonlit rim along one upper edge.
- small_cone: a compact amber beam + halation for a secondary downward lamp (plate-space volume).
"""
import math

import numpy as np
import cv2

import s08_snow_station_tex as TX


# ----------------------------------------------------------------------------- walls
def wall_weather(U, V, v_top, ppm, seed=0):
    """U along the boards (m), V height above the floor (m), v_top wall top (m), ppm px per metre.
    Returns (mult (H, W), stain (H, W)) : multiply the albedo by mult, then darken / tint by stain."""
    ppm = np.asarray(ppm, np.float32)
    per = 0.19
    board = np.floor(V / per)
    fv = np.mod(V / per, 1.0)
    # broad flowing grain bands (cathedral figure) readable at mid distance
    w1 = TX.sample(seed + 5, U * 0.35, board * 0.173 + fv * 0.05, cells=5, octaves=2)
    ph = U * 1.6 + 2.2 * w1 + board * 3.7 + 1.8 * np.abs(fv - 0.5)
    band = 0.5 + 0.5 * np.sin(ph * 2 * math.pi)
    band = np.clip(band, 0, 1) ** 3
    fine = TX.sample(seed + 6, U * 1.2, V * 14.0, cells=12, octaves=3)
    vis = np.clip((ppm - 25) / 60.0, 0, 1)
    mult = 1.0 + (0.22 * band - 0.08 + 0.1 * fine) * (0.5 + 0.5 * vis)
    # lighter worn lower lip of each board (sun-bleached) + dark lap shadow line
    mult = mult * (1 + 0.14 * np.clip((fv - 0.7) / 0.3, 0, 1))
    # drip stains below the eave: thin dark streaks of very different length hanging from the top,
    # a few tinted rusty where nails / the gutter bracket bleed
    rng = np.random.default_rng(seed + 40)
    nst = 90
    cu = rng.uniform(0, 1, nst)
    wd = rng.uniform(0.012, 0.05, nst)
    ln = rng.uniform(0.15, 1.3, nst) ** 1.3
    st = rng.uniform(0.25, 0.75, nst)
    # evaluate on the pixels near the top band only (cheap: separable in U via a 1D lookup table)
    tab_n = 4096
    ut = np.linspace(0, 1, tab_n, dtype=np.float32)
    colw = np.zeros((nst, tab_n), np.float32)
    for i in range(nst):
        d = np.minimum(np.abs(ut - cu[i]), 1 - np.abs(ut - cu[i]))
        colw[i] = np.exp(-(d / (wd[i] / 9.0)) ** 2)
    uu = np.mod(U / 9.0, 1.0)
    idx = np.clip((uu * (tab_n - 1)).astype(np.int32), 0, tab_n - 1)
    below = np.clip(v_top - V, 0, None)
    stain = np.zeros(U.shape, np.float32)
    for i in range(nst):
        prof = colw[i][idx]
        stain += st[i] * prof * np.clip(1 - below / ln[i], 0, 1) ** 1.3
    wob = TX.sample(seed + 7, U * 0.8, V * 0.4, cells=10, octaves=2)
    stain = stain * (0.75 + 0.25 * wob) + 0.35 * np.clip(1 - below / 0.18, 0, 1)
    return mult.astype(np.float32), np.clip(stain, 0, 1).astype(np.float32)


def apply_weather(col, mult, stain):
    """Apply wall_weather output to a colour array (..., 3)."""
    rust = np.array([0.55, 0.3, 0.16], np.float32)
    s_ = stain[..., None]
    out = col * mult[..., None]
    out = out * (1 - 0.7 * s_) + out * rust * 0.5 * s_
    return out


# ----------------------------------------------------------------------------- sky
def _fbm(rng, w, h, cells, octs=4):
    acc = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octs):
        cx = max(int(cells * 2 ** o), 2)
        cy = max(int(cells * 2 ** o * h / w), 2)
        g = rng.random((cy + 1, cx + 1)).astype(np.float32)
        acc += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    acc /= tot
    return (acc - acc.mean()) / (acc.std() + 1e-6)


def soft_night_clouds(img, horizon_v, s, glow_centres=(), seed=21):
    """Paint an overcast snow-cloud ceiling over the plate sky img (PH, PW, 3) above horizon_v (px).
    A few huge soft masses (large blurred silhouettes) with smooth vertical value gradients, darker
    slate tops, lighter lavender-grey bellies; warm sodium glow on the undersides over glow_centres
    [(u_px, strength)], a thin cool moonlit rim on the upper-left edges. Returns new img."""
    PH, PW = img.shape[:2]
    rng = np.random.default_rng(seed)
    q = 4
    w, h = PW // q, PH // q
    yy = np.linspace(0, PH, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, PW, w, dtype=np.float32)[None, :]
    hv = horizon_v
    # coverage: big low-frequency masses, gently stretched horizontally (stratocumulus banks)
    base = _fbm(rng, w, h, 3.0, 4)
    big = cv2.resize(_fbm(rng, w // 2 + 2, h + 2, 1.6, 3), (w, h), interpolation=cv2.INTER_CUBIC)
    cov = 0.45 * base + 0.8 * big
    yn = yy / hv
    bands = (0.9 * np.exp(-((yn - 0.12) / 0.2) ** 2) + 0.8 * np.exp(-((yn - 0.46) / 0.14) ** 2)
             + 0.6 * np.exp(-((yn - 0.74) / 0.1) ** 2))
    field = cov + 1.1 * bands - 0.15
    field = cv2.GaussianBlur(field.astype(np.float32), (0, 0), 2.0)
    m = np.clip((field + 0.05) / 0.45, 0, 1)
    m = m * m * (3 - 2 * m)
    m = m * np.clip((hv * 0.97 - yy) / (0.08 * hv), 0, 1)
    depth = cv2.GaussianBlur(np.clip(field, 0, 2), (0, 0), 5.0)
    k = np.clip(depth / 1.2, 0, 1)[..., None]
    gy = cv2.Sobel(field, cv2.CV_32F, 0, 1, ksize=5)
    gy = cv2.GaussianBlur(gy, (0, 0), 3.0)
    belly = np.clip(-gy * 0.3, 0, 1)[..., None]          # lower (downward-facing) parts of each mass
    topf = np.clip(gy * 0.3, 0, 1)[..., None]            # upper edges
    sky = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    # painted values relative to the sky behind: bodies a touch lighter/greyer (lit from the town),
    # upper parts darker slate, bellies lighter lavender
    grey = sky.mean(-1, keepdims=True)
    body = sky * 0.85 + grey * np.array([0.42, 0.4, 0.55], np.float32) + np.array([0.024, 0.024, 0.05], np.float32)
    col = body * (1.0 + 0.12 * k) * (1 - 0.22 * topf) + belly * np.array([0.045, 0.04, 0.07], np.float32)
    glow = np.zeros((h, w), np.float32)
    for (gu, gs, gw) in glow_centres:
        glow += gs * np.exp(-((xx - gu) / gw) ** 2) * np.clip((yn - 0.2) / 0.6, 0, 1) ** 1.3
    col = col + (glow[..., None] * (0.35 + 0.9 * belly)) * np.array([0.2, 0.11, 0.045], np.float32)
    # moonlit rim along upward-facing edges, strongest upper left
    rim = np.clip(gy * 0.6, 0, 1) * np.clip(1.3 - xx / (0.6 * PW), 0, 1) * np.clip(1.1 - yn, 0, 1)
    rim = rim * np.clip((field + 0.05) / 0.25, 0, 1) * np.clip(1 - (field - 0.2) / 0.4, 0, 1)
    col = col + rim[..., None] * np.array([0.07, 0.09, 0.16], np.float32)
    bv = cv2.GaussianBlur(_fbm(rng, w, h, 5, 2), (0, 0), 2.0)
    col = col * (1 + 0.05 * bv[..., None])
    col = cv2.resize(col.astype(np.float32), (PW, PH), interpolation=cv2.INTER_CUBIC)
    m = cv2.resize(m.astype(np.float32), (PW, PH), interpolation=cv2.INTER_CUBIC)[..., None]
    m = np.clip(m, 0, 1) * 0.95
    return (img * (1 - m) + col * m).astype(np.float32)


# ----------------------------------------------------------------------------- secondary lamp
def small_cone(PW, PH, lu, lv, ground_v, s, half_ang=0.36, amber=(1.0, 0.74, 0.36), strength=0.22):
    """Screen-space painted beam for a small downward lamp at (lu, lv) px whose light reaches down to
    about ground_v px: a tight cone with feathered edges, brightest near the head, fading with length,
    plus a soft halation disc round the head."""
    q = 2
    w, h = PW // q, PH // q
    yy = (np.arange(h, dtype=np.float32)[:, None] + 0.5) * q
    xx = (np.arange(w, dtype=np.float32)[None, :] + 0.5) * q
    dy = yy - lv
    L = max(ground_v - lv, 10.0)
    dx = xx - lu
    rel = np.abs(dx) / np.maximum(dy * math.tan(half_ang) + 6 * s, 1e-3)
    across = np.clip(1 - rel, 0, 1) ** 1.4
    along = np.clip(dy / (10 * s), 0, 1) * np.exp(-np.clip(dy, 0, None) / (0.55 * L)) * (dy > 0)
    beam = across * along
    d2 = dx * dx + dy * dy
    hal = 0.9 * np.exp(-d2 / (2 * (10 * s) ** 2)) + 0.35 * np.exp(-d2 / (2 * (40 * s) ** 2))
    v = cv2.GaussianBlur((beam * strength + hal * 0.35).astype(np.float32), (0, 0), 1.5 * s + 0.3)
    v = cv2.resize(v, (PW, PH), interpolation=cv2.INTER_LINEAR)
    return v[..., None] * np.asarray(amber, np.float32)
