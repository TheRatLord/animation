"""Round-3 painted additions for s08_snow_station.

- long_cone: a faint amber beam volume that carries the foreground lamp's cone all the way down to its
  pool on the platform (the tight single-scattering cone fades out ~1.5 m under the head).
- pole_light: warm lamp-side edge + thin specular line + chipped paint on the tall foreground lamp pole.
- cornice: the canopy's front snow load painted as an irregular cornice (uneven overhang, lumpy top,
  lamp-lit amber rim, blue-violet underside, sparkle glints, irregular icicles that catch the lamp).
- boot_prints: paired, elongated boot prints along curved trails (heel + sole), with blue-violet
  shadowed interiors, a lit far wall and kicked-up rims.
"""
import math

import numpy as np
import cv2
from numba import njit, prange


# ----------------------------------------------------------------------------- long beam
@njit(cache=True, fastmath=True, parallel=True)
def _beam(dx, dy, depth, px, py, pz, cin, cout, ymin, nsamp):
    H, W = dx.shape
    out = np.zeros((H, W), np.float32)
    for i in prange(H):
        for j in range(W):
            rx, ry = dx[i, j], dy[i, j]
            rn = math.sqrt(rx * rx + ry * ry + 1.0)
            ux, uy, uz = rx / rn, ry / rn, 1.0 / rn
            T = depth[i, j] * rn
            s0 = px * ux + py * uy + pz * uz
            sa = max(s0 - 4.5, 0.05)
            sb = min(s0 + 4.5, T)
            if sb <= sa:
                continue
            ds = (sb - sa) / nsamp
            acc = 0.0
            for m in range(nsamp):
                ss = sa + (m + 0.5) * ds
                wx = ss * ux - px
                wy = ss * uy - py
                wz = ss * uz - pz
                if ss * uy < ymin:
                    continue
                d = math.sqrt(wx * wx + wy * wy + wz * wz) + 1e-4
                cd = -wy / d
                c = (cd - cout) / (cin - cout)
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                c = c * c * (3.0 - 2.0 * c)
                acc += c / (0.6 + d) ** 2.0 * ds
            out[i, j] = acc
    return out


def long_cone(rx, ry, zbuf, lamp_pos, cam_x, y_floor, q=2):
    """Faint beam volume (plate size, scalar) for a downward lamp at lamp_pos (world)."""
    PH, PW = rx.shape
    w, h = PW // q, PH // q
    rxs = cv2.resize(rx, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    rys = cv2.resize(ry, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    zb = cv2.resize(cv2.erode(zbuf, np.ones((3, 3), np.uint8)), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    zb = np.minimum(zb, 200.0).astype(np.float32)
    out = _beam(rxs, rys, zb, np.float32(lamp_pos[0] - cam_x), np.float32(lamp_pos[1]), np.float32(lamp_pos[2]),
                np.float32(math.cos(math.radians(13))), np.float32(math.cos(math.radians(27))), np.float32(y_floor), 64)
    out = cv2.GaussianBlur(out, (0, 0), 1.2)
    return cv2.resize(out, (PW, PH), interpolation=cv2.INTER_LINEAR).astype(np.float32)


# ----------------------------------------------------------------------------- lamp pole
def pole_light(cv, cam, x, z, y0, y1, w, lamp_uv, s, warm, seed=3):
    """Paint the lit side of a square lamp pole (world x centre, depth z, from y0 to y1, half width w)
    onto canvas cv: a warm amber edge band on the lamp side (strongest near the head, again near the
    lit ground pool), a thin bright specular line, and chipped paint / rust flecks along the shaft."""
    rng = np.random.default_rng(seed)
    ua, va = cam.p(x - w, y1, z)
    ub, vb = cam.p(x + w, y0, z)
    X0, Y0 = int(math.floor(min(ua, ub))) - 2, int(math.floor(min(va, vb))) - 2
    X1, Y1 = int(math.ceil(max(ua, ub))) + 3, int(math.ceil(max(va, vb))) + 3
    X0, Y0 = max(X0, 0), max(Y0, 0)
    X1, Y1 = min(X1, cv.W), min(Y1, cv.H)
    if X1 <= X0 or Y1 <= Y0:
        return
    xx = np.arange(X0, X1, dtype=np.float32)[None, :]
    yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
    # world coords of the pixels on the pole face
    Yw = -(yy - cam.cy) * z / cam.f
    Xw = (xx - cam.cx) * z / cam.f + cam.ox
    # the pole tapers: width at this height
    tpr = (Yw - y0) / max(y1 - y0, 1e-3)
    ww = w * (1 - 0.3 * np.clip(tpr, 0, 1))
    rel = (Xw - x) / np.maximum(ww, 1e-4)            # -1 (lamp side, left) .. 1
    inside = (np.abs(rel) <= 1.0) & (Yw >= y0) & (Yw <= y1)
    a = inside.astype(np.float32)
    # warm lit band on the lamp side: strongest near the head, fading down, picking up again as bounce
    # from the lit snow pool near the ground
    head = np.clip(1 - (y1 - Yw) / 2.2, 0, 1) ** 1.5
    bounce = np.clip(1 - (Yw - y0) / 1.4, 0, 1) ** 1.2
    band = np.clip((-0.15 - rel) / 0.5, 0, 1)
    lit = band * (0.55 * head + 0.35 * bounce + 0.1)
    spec = np.exp(-((rel + 0.72) / 0.09) ** 2) * (0.5 * head + 0.35 * bounce + 0.12)
    # chipped paint / rust flecks
    nz = cv2.resize(rng.random((max((Y1 - Y0) // max(int(3 * s), 1), 2), max((X1 - X0) // max(int(2 * s), 1), 2))).astype(np.float32),
                    (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
    chip = (nz > 0.93).astype(np.float32) * 0.6
    rust = np.clip(1 - (Yw - y0) / 0.6, 0, 1) * (nz > 0.6) * 0.5
    base = np.array([0.045, 0.05, 0.075], np.float32)
    col = base * (1 + 0.5 * chip[..., None]) + np.array([0.12, 0.05, 0.02], np.float32) * rust[..., None]
    col = col + warm * (lit[..., None] * 0.42) + np.array([1.0, 0.82, 0.6], np.float32) * (spec[..., None] * 0.9)
    # cool sky rim on the right edge
    col = col + np.array([0.06, 0.08, 0.16], np.float32) * np.clip((rel - 0.6) / 0.4, 0, 1)[..., None]
    rgb = cv.rgb[Y0:Y1, X0:X1]
    al = cv.a[Y0:Y1, X0:X1]
    m = (a * (al > 0.5))[..., None]
    cv.rgb[Y0:Y1, X0:X1] = rgb * (1 - m) + col * m * al[..., None]


# ----------------------------------------------------------------------------- weathering
def _blob_noise(rng, h, w, cy, cx):
    g = rng.random((max(cy, 2), max(cx, 2))).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


def sign_grime(tex, seed=0):
    """Extra weathering for the enamel station-name board (h, w, 3 albedo): chipped paint (ragged flakes
    knocked off along the edges and round the bolts, showing dark rusty steel with a light chipped
    rim), water / rust streaks running down from the top edge under the snow cap and from the bolts,
    and a grimy bottom band."""
    h, w = tex.shape[:2]
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    out = tex.copy()
    # streaks from the top edge (melt water under the snow cap) - varied width / length / strength
    streak = np.zeros((h, w), np.float32)
    for _ in range(int(26 + w / 30)):
        cx = rng.uniform(0.01, 0.99)
        wd = rng.uniform(0.002, 0.009)
        ln = rng.uniform(0.15, 0.9)
        st = rng.uniform(0.25, 0.8)
        prof = np.exp(-((x - cx - 0.004 * np.sin(y * 30 + cx * 50)) / wd) ** 2)
        streak += st * prof * np.clip(1 - y / ln, 0, 1) ** 1.4
    # rust runs from the four bolts and the post brackets
    rust = np.zeros((h, w), np.float32)
    for bx in (0.035, 0.965):
        for by in (0.1, 0.88):
            rust += np.exp(-((x - bx) / 0.006) ** 2) * np.clip((y - by) / 0.02, 0, 1) * np.exp(-(y - by) / 0.18) * 0.9
            rust += np.exp(-(((x - bx) * w) ** 2 + ((y - by) * h) ** 2) / (0.02 * h + 1) ** 2) * 0.9
    # chipped paint: ragged patches near the edges / corners
    edge_d = np.minimum(np.minimum(x * w, (1 - x) * w), np.minimum(y * h, (1 - y) * h))
    n1 = _blob_noise(rng, h, w, h // 5 + 2, w // 5 + 2)
    n2 = _blob_noise(rng, h, w, h // 2 + 2, w // 2 + 2)
    chip_p = np.clip(1 - edge_d / (0.12 * h + 2), 0, 1) * 0.55 + 0.08
    chip = ((n1 * 0.6 + n2 * 0.4) > (1 - chip_p * 0.62)).astype(np.float32)
    chip = cv2.GaussianBlur(chip, (0, 0), 0.5)
    chip_rim = np.clip(cv2.dilate(chip, np.ones((3, 3), np.uint8)) - chip, 0, 1)
    steel = np.array([0.26, 0.17, 0.12], np.float32) * (0.7 + 0.6 * n2[..., None])
    grime_c = np.array([0.22, 0.17, 0.12], np.float32)
    rust_c = np.array([0.5, 0.22, 0.08], np.float32)
    s_ = np.clip(streak, 0, 1)[..., None]
    out = out * (1 - 0.35 * s_) + grime_c * 0.35 * s_
    r_ = np.clip(rust, 0, 1)[..., None]
    out = out * (1 - 0.65 * r_) + rust_c * 0.65 * r_
    out = out * (1 - chip[..., None]) + steel * chip[..., None]
    out = out + chip_rim[..., None] * 0.12
    # grimy bottom band + soot along the bottom edge
    gb = np.clip((y - 0.75) / 0.25, 0, 1) ** 1.6 * (0.6 + 0.4 * n1)
    out = out * (1 - 0.3 * gb[..., None])
    return np.clip(out, 0, 1.5).astype(np.float32)


def plank_detail(tex, Lm, joints=(), bolts=(), seed=0, grain_boost=1.8, top_wear=True):
    """Stronger painted wood for a slat texture (h, w, 3) spanning Lm metres along x: boost the grain
    contrast, dark butt-joint seams at `joints` (fractions of the length), bolt heads, a light worn
    top edge and dirt collecting along the bottom edge."""
    h, w = tex.shape[:2]
    rng = np.random.default_rng(seed)
    mean = tex.mean(axis=(0, 1), keepdims=True)
    out = mean + (tex - mean) * grain_boost
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    # long grain streaks along the board (dark fibre lines)
    fib = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigmaX=w * 0.03 + 1, sigmaY=0.6)
    fib = (fib - fib.mean()) / (fib.std() + 1e-6)
    out = out * (1 + 0.12 * np.clip(fib, -2, 2))[..., None]
    for j in joints:
        seam = np.exp(-((x - j) * w / 1.2) ** 2)
        out = out * (1 - 0.7 * seam[..., None])
        out = out + np.exp(-((x - j - 1.8 / w) * w / 1.0) ** 2)[..., None] * 0.05
    for (bx, by) in bolts:
        d = np.sqrt(((x - bx) * w) ** 2 + ((y - by) * h) ** 2)
        b = np.clip(1.6 - d, 0, 1)
        out = out * (1 - 0.6 * b[..., None]) + np.array([0.3, 0.14, 0.06], np.float32) * 0.5 * b[..., None]
    if top_wear:
        tw = np.clip(1 - y * h / 2.0, 0, 1) * (rng.random((1, w)) > 0.3)
        out = out + tw[..., None] * np.array([0.12, 0.09, 0.06], np.float32)
    dirt = np.clip((y - 0.7) / 0.3, 0, 1)
    out = out * (1 - 0.3 * dirt[..., None])
    return np.clip(out, 0, 1.5).astype(np.float32)
