"""s04_rain_street painterly passes (panel v3 notes):
  * grime      - world-anchored value breakup, rain-stain run-off streaks and wet vertical specular streaks
                 painted into the G-buffer (albedo / emission) of the facade + object canvases before lighting.
  * neon_spill - soft halation falloff of the neon signs onto the adjacent wet walls, poles and machines
                 (the sign emission, blurred wide, lights the albedo around it).
  * brush_smear- road reflections broken into vertical brush strokes of varying length (per-column stroke map)
"""
import numpy as np
import cv2

from lib import core as C

_TEX = {}


def _tile(seed, scale=16, octaves=5):
    if seed not in _TEX:
        _TEX[seed] = C.fbm(1024, 1024, scale, octaves, seed=seed, aspect=False).astype(np.float32)
    return _TEX[seed]


def _wn(seed, U, V, fu, fv, scale=16, octaves=5):
    """Noise sampled at world coords (U, V) with frequencies fu, fv (cells / m)."""
    t = _tile(seed, scale, octaves)
    k = 1024.0 / scale
    u = np.mod(U * (fu * k), 1024.0).astype(np.float32)
    v = np.mod(V * (fv * k), 1024.0).astype(np.float32)
    return cv2.remap(t, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _world(cv):
    cam = cv.cam
    s = cam.ss
    a = cv.a
    am = np.maximum(a, 1e-4)
    zr = cv.z / am
    Hs, Ws = a.shape
    ys, xs = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    x = (xs + 0.5) / s - 0.5
    y = (ys + 0.5) / s - 0.5
    X = (x - cam.pcx) * zr / cam.f
    Y = cam.h - (y - cam.pcy) * zr / cam.f
    nx = np.abs(cv.n[..., 0]) / am
    side = nx > 0.5
    U = np.where(side, zr, X)          # horizontal coordinate along the surface (Z on side walls, X on fronts)
    return U.astype(np.float32), Y.astype(np.float32), zr, side


def grime(cv, strength=1.0, seed=0, spec_light=None, emi_comp=0.0):
    """In place on a Canvas. strength scales everything; spec_light: optional (h, w, 3) light map (any size)
    that colours the wet streaks (defaults to the canvas' own blurred emission)."""
    a = cv.a
    if a.max() <= 0:
        return
    U, Y, zr, side = _world(cv)
    s = cv.cam.ss
    # emissive surfaces (lit sign faces, windows) receive much less grime
    e = cv.emi.max(-1) / np.maximum(a, 1e-4)
    keep = np.clip(1.0 - e * 1.2, 0.45, 1.0)
    # --- broad painted value masses + mid mottling (tile / panel dirt)
    b1 = _wn(seed + 1, U, Y, 0.35, 0.35)
    b2 = _wn(seed + 2, U, Y, 1.6, 1.6)
    b3 = _wn(seed + 3, U, Y, 6.0, 6.0, octaves=3)
    val = 1.0 + 0.42 * (b1 - 0.5) * 2 + 0.22 * (b2 - 0.5) * 2 + 0.08 * (b3 - 0.5) * 2
    # --- rain-stain run-off: long vertical streaks, strongest just below each floor slab (every ~3 m) and
    # thinning downwards; column pattern is fine along the surface and very long vertically
    st = _wn(seed + 4, U, Y, 5.0, 0.12, octaves=4)
    st2 = _wn(seed + 5, U, Y, 13.0, 0.25, octaves=3)
    fl = np.mod(Y, 3.0) / 3.0                       # 1 just below a slab, 0 at the next slab down
    run = 0.35 + 0.65 * fl ** 1.5
    streak = np.clip((st - 0.47) * 4.0, 0, 1) * run + np.clip((st2 - 0.55) * 5.0, 0, 1) * 0.8 * run
    streak = np.clip(streak, 0, 1)
    # top-down soot / damp near the ground (splash-back band 0-0.6 m)
    damp = np.clip(1.0 - Y / 0.7, 0, 1) * (0.6 + 0.4 * b2)
    mult = val * (1.0 - 0.6 * streak) * (1.0 - 0.35 * damp)
    k = (keep * strength).astype(np.float32)
    mult = 1.0 + (mult - 1.0) * k
    # desaturate the grime slightly toward a cool grey-green (no candy colours in the dirt)
    alb = cv.alb
    lum = (alb * np.float32([0.3, 0.55, 0.15])).sum(-1, keepdims=True)
    tint = np.float32([0.92, 1.0, 0.98])
    dmix = (0.25 * streak * k)[..., None]
    alb[:] = (alb * (1 - dmix) + lum * tint * dmix) * mult[..., None]
    # lit faces (lightboxes, glass) are dirty too: stains and mottling dim their emission a little
    em = 1.0 + (val * (1.0 - 0.55 * streak) - 1.0) * (1.0 * strength)
    if emi_comp > 0:
        # blown lightbox faces are pulled back under the shoulder so their painted dirt can read
        el = cv.emi.max(-1) / np.maximum(a, 1e-4)
        em = em / (1.0 + emi_comp * np.maximum(el - 0.8, 0))
    em = 1.0 + (np.clip(em, 0.3, 1.3) - 1.0) * np.clip(a, 0, 1)
    cv.emi *= em[..., None]
    # --- wet vertical specular streaks: thin bright runs catching the nearby light
    w1 = _wn(seed + 6, U, Y, 14.0, 0.5, octaves=2)
    w2 = _wn(seed + 7, U, Y, 0.6, 0.6, octaves=2)
    wet = np.clip((w1 - 0.64) * 7.0, 0, 1) * np.clip((w2 - 0.35) * 2.5, 0, 1)
    if spec_light is None:
        h4, w4 = a.shape[0] // 8, a.shape[1] // 8
        sm = cv2.resize(cv.emi, (w4, h4), interpolation=cv2.INTER_AREA)
        sl = cv2.GaussianBlur(sm, (0, 0), 0.02 * w4) * 0.8 + cv2.GaussianBlur(sm, (0, 0), 0.006 * w4) * 0.5
    else:
        sl = spec_light
    sl = cv2.resize(sl, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
    base = np.float32([0.05, 0.075, 0.08])
    cv.emi += (wet * keep * strength * 0.55 * a)[..., None] * (sl + base)


def spill_accum(acc, cv, f=8):
    """Accumulate a canvas' emission (downsampled by f) into acc (None -> new)."""
    Hs, Ws = cv.a.shape
    e = cv2.resize(cv.emi, (Ws // f, Hs // f), interpolation=cv2.INTER_AREA)
    return e if acc is None else acc + e


def spill_light(acc, W):
    """Wide soft halation field from the accumulated sign emission (1/f res). W = frame width (px)."""
    if acc is None:
        return None
    w = acc.shape[1]
    e = np.minimum(acc, 3.0)
    return (cv2.GaussianBlur(e, (0, 0), 0.012 * w) * 0.55 + cv2.GaussianBlur(e, (0, 0), 0.035 * w) * 0.45 +
            cv2.GaussianBlur(e, (0, 0), 0.08 * w) * 0.25).astype(np.float32)


def neon_spill(cv, light, amt=0.5):
    """Light the canvas albedo with the halation field (added as emission, i.e. after-the-fact spill)."""
    if light is None:
        return
    L = cv2.resize(light, (cv.a.shape[1], cv.a.shape[0]), interpolation=cv2.INTER_LINEAR)
    e = cv.emi.max(-1) / np.maximum(cv.a, 1e-4)
    keep = np.clip(1.0 - e * 1.2, 0.0, 1.0)
    cv.emi += cv.alb * L * (amt * keep)[..., None]


# ----------------------------------------------------------------------------- reflections
def brush_smear(sc, img):
    """Wet-road reflection as vertical brush strokes: each stroke column smears a different length, strokes
    are a few px wide with soft sides, and a light horizontal jitter breaks the mirrored lettering."""
    H, W = img.shape[:2]
    if not hasattr(sc, '_bsm'):
        W2, H2 = W // 2, H // 2
        rng = np.random.default_rng(6161)
        col = rng.random((max(H2 // 60, 2), max(W2 // 5, 4))).astype(np.float32)
        col = cv2.resize(col, (W2, H2), interpolation=cv2.INTER_CUBIC)
        col2 = rng.random((max(H2 // 25, 2), max(W2 // 2, 4))).astype(np.float32)
        col2 = cv2.resize(col2, (W2, H2), interpolation=cv2.INTER_CUBIC)
        m = np.clip(col * 0.7 + col2 * 0.3, 0, 1)
        w_long = np.clip((m - 0.35) * 2.2, 0, 1)
        sc._bsm = (w_long[..., None].astype(np.float32),)
    (wl,) = sc._bsm
    W2, H2 = W // 2, H // 2
    s2 = cv2.resize(img, (W2, H2), interpolation=cv2.INTER_AREA)
    b1 = cv2.GaussianBlur(s2, (0, 0), sigmaX=0.7, sigmaY=0.006 * H)
    b2 = cv2.GaussianBlur(s2, (0, 0), sigmaX=1.0, sigmaY=0.018 * H)
    mix = b1 * (1 - wl) + b2 * wl
    up = cv2.resize(mix, (W, H), interpolation=cv2.INTER_LINEAR)
    return (up * 0.82 + img * 0.18).astype(np.float32)
