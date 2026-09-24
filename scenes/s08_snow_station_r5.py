"""Round-5 painted helpers for s08_snow_station.

painted_overcast: the low snow sky as a hand-painted glowing overcast (after cm5_08): a clear vertical
value gradient (deep navy zenith -> lighter slate-lavender toward the horizon), a few broad horizontal
cloud-band masses laid in as flat value steps with soft lower edges, and lighter town-light scatter
bands just above the tree line (a touch warmer on the right, over the village).
lamp_air_glow: warm, wide glow of lit snowy air around a lamp head (screen space, additive)."""
import numpy as np
import cv2


def _fbm(rng, w, h, cx, cy, octs=4):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octs):
        gw, gh = int(cx * 2 ** o) + 2, int(cy * 2 ** o) + 2
        g = rng.random((gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
    out /= tot
    return (out - out.mean()) / (out.std() + 1e-6)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def painted_overcast(img, horizon_v, s, seed=51, amount=0.8):
    PH, PW = img.shape[:2]
    rng = np.random.default_rng(seed)
    q = 4
    w, h = PW // q, PH // q
    yy = np.linspace(0, PH, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    el = np.clip((horizon_v - yy) / horizon_v, -0.2, 1.2)
    # vertical value gradient (zenith -> horizon)
    top = np.array([0.05, 0.065, 0.155], np.float32)
    mid = np.array([0.11, 0.13, 0.265], np.float32)
    low = np.array([0.23, 0.26, 0.41], np.float32)
    k1 = _ss(0.0, 0.55, el)[..., None]
    k2 = _ss(0.45, 1.0, el)[..., None]
    base = low * (1 - k1) + mid * k1
    base = base * (1 - k2) + top * k2
    # broad horizontal band masses: stretched fbm, bent a little, laid in as three flat value steps
    n = _fbm(rng, w, h, 1.6, 5.0, 4)
    warp = cv2.resize(rng.random((4, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    n = cv2.remap(n, np.tile(np.arange(w, dtype=np.float32)[None, :], (h, 1)),
                  np.clip(np.arange(h, dtype=np.float32)[:, None] + (warp - 0.5) * h * 0.12, 0, h - 1).astype(np.float32),
                  cv2.INTER_LINEAR)
    n = n + 0.55 * np.sin(el * 11.0 + xx * 2.2)             # stacked stratus bands
    n = cv2.GaussianBlur(n.astype(np.float32), (0, 0), 1.2)
    # each step edge is crisp on the upper side, soft (lost) underneath: use the vertical gradient sign
    gy = cv2.Sobel(n, cv2.CV_32F, 0, 1, ksize=3)
    soft = np.where(gy < 0, 0.07, 0.25).astype(np.float32)
    step1 = _ss(0.1 - soft, 0.1 + soft, n)
    step2 = _ss(0.9 - soft, 0.9 + soft, n)
    val = -0.35 + 0.55 * step1 + 0.45 * step2                 # -0.35 (gaps) .. 0.65 (densest mass)
    band_col = np.array([0.075, 0.08, 0.13], np.float32)
    col = base + val[..., None] * band_col * (0.45 + 0.8 * (1 - _ss(0.2, 1.0, el)))[..., None]
    # light scatter from the town / station just above the tree line: lighter soft bands, warmer right
    scat = np.exp(-np.clip(el - 0.04, 0, None) / 0.2) * _ss(-0.05, 0.05, el)
    scat = scat * (0.75 + 0.25 * np.clip(val + 0.35, 0, 1))
    warm = np.exp(-((xx - 0.78) / 0.2) ** 2) + 0.6 * np.exp(-((xx - 0.45) / 0.14) ** 2)
    col = col + scat[..., None] * (np.array([0.06, 0.07, 0.1], np.float32) + warm[..., None] * np.array([0.07, 0.045, 0.02], np.float32))
    # painted mottling (large soft dabs), no pixel noise
    mot = cv2.GaussianBlur(_fbm(rng, w, h, 7, 5, 2), (0, 0), 1.5)
    col = col * (1 + 0.035 * mot[..., None])
    col = cv2.resize(col.astype(np.float32), (PW, PH), interpolation=cv2.INTER_CUBIC)
    kk = amount * _ss(-0.04, 0.06, (horizon_v - np.arange(PH, dtype=np.float32)) / horizon_v)[:, None, None]
    return (img * (1 - kk) + col * kk).astype(np.float32)


def lamp_air_glow(W, H, pts, s):
    """pts: [(u, v, rgb, radius_px, strength)] -> additive (H, W, 3) warm air glow (two gaussian lobes)."""
    q = 4
    w, h = max(W // q, 1), max(H // q, 1)
    yy = (np.arange(h, dtype=np.float32)[:, None] + 0.5) * (H / h) - 0.5
    xx = (np.arange(w, dtype=np.float32)[None, :] + 0.5) * (W / w) - 0.5
    out = np.zeros((h, w, 3), np.float32)
    for (u, v, c, r, k) in pts:
        d2 = ((xx - u) ** 2 + (yy - v) ** 2)
        g = 0.65 * np.exp(-d2 / (r * r)) + 0.35 * np.exp(-d2 / (0.3 * r) ** 2)
        out += (g * k)[..., None] * np.asarray(c, np.float32)
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
