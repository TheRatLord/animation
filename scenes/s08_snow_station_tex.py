"""Painted surface textures for s08_snow_station (world-space, sampled per pixel through the camera):
wood grain for clapboard / slats (per-board tone, flowing grain lines filtered by pixel footprint, butt
joints, knots), grime (rain-streaks running down from openings and eaves, dirt splash near the ground,
soot) and weathering for enamel signs (rust bleeding from bolts and edges, dirt collecting at the
bottom). Each returns multipliers / colours for the caller's existing shading."""
import numpy as np
import cv2

_CACHE = {}


def _noise(seed, n=512, cells=24, octaves=4):
    key = (seed, n, cells, octaves)
    if key not in _CACHE:
        rng = np.random.default_rng(seed)
        acc = np.zeros((n, n), np.float32)
        amp, tot = 1.0, 0.0
        for o in range(octaves):
            c = cells * 2 ** o
            g = rng.random((c + 1, c + 1)).astype(np.float32)
            g[-1] = g[0]
            g[:, -1] = g[:, 0]
            acc += amp * cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC)
            tot += amp
            amp *= 0.5
        acc /= tot
        acc = (acc - acc.mean()) / (acc.std() + 1e-6)
        _CACHE[key] = acc.astype(np.float32)
    return _CACHE[key]


def sample(seed, u, v, cells=24, octaves=4):
    """Tileable fbm (zero mean, unit std) sampled at float coords (u, v) in tile units."""
    T = _noise(seed, 512, cells, octaves)
    mu = (np.mod(u, 1.0) * 512).astype(np.float32)
    mv = (np.mod(v, 1.0) * 512).astype(np.float32)
    return cv2.remap(T, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def wood(U, V, per, px_per_m, seed=0, joints=True):
    """Clapboard / plank wood multiplier. U: along the boards (m), V: across (m), per: board width (m),
    px_per_m: screen pixels per metre (array or scalar) for filtering. Returns (H, W) ~ 0.6..1.25."""
    ppm = np.asarray(px_per_m, np.float32)
    board = np.floor(V / per)
    rng = np.random.default_rng(seed)
    tone_tab = rng.normal(0, 1, 4096).astype(np.float32)
    off_tab = rng.random(4096).astype(np.float32)
    bi = (board.astype(np.int64) % 4096)
    tone = tone_tab[bi]
    off = off_tab[bi]
    fv = np.mod(V / per, 1.0)
    # flowing grain lines along the board: phase modulated by low-frequency noise, fades when sub-pixel
    warp = sample(seed + 1, U * 0.18 + off * 7, fv * 0.25 + board * 0.37, cells=6, octaves=3)
    g_freq = 26.0                                           # lines per metre across the board
    gph = (V + 0.035 * warp) * g_freq + off * 10
    lines = np.abs(np.mod(gph, 1.0) - 0.5) * 2.0            # 0 at line centre
    vis = np.clip((ppm / g_freq - 1.2) / 2.0, 0, 1)          # need > ~1.2 px per line spacing
    grain = 1 - 0.3 * vis * np.clip(1 - lines / 0.35, 0, 1)
    # long fibre streaks (visible from further away)
    fib = sample(seed + 2, U * 0.9 + off * 3, V * 9.0, cells=10, octaves=3)
    fvis = np.clip((ppm - 6) / 20.0, 0, 1)
    mult = (1 + 0.12 * tone) * grain * (1 + 0.13 * fib * fvis)
    if joints:
        seg = 2.2 + 1.2 * off
        jp = np.mod(U + off * seg * 3.0, seg)
        jw = np.maximum(0.012, 0.7 / np.maximum(ppm, 1e-3))
        joint = np.clip(1 - np.minimum(jp, seg - jp) / jw, 0, 1)
        mult = mult * (1 - 0.45 * joint * np.clip(ppm / 40.0, 0, 1))
    # knots: a few dark ovals with a lighter ring
    kn = sample(seed + 3, U * 0.35, board * 0.21, cells=16, octaves=1)
    knot = np.clip((kn - 2.0) / 0.4, 0, 1) * np.clip((ppm - 30) / 40.0, 0, 1)
    mult = mult * (1 - 0.35 * knot)
    return mult.astype(np.float32)


def grime(U, V, y_top, y_bot, seed=0, openings=()):
    """Weathering multiplier for a wall: rain streaks running down (stronger just under the eave and
    under each opening), dirt splash along the bottom, blotchy soot. U horizontal, V height (m).
    openings: [(u0, u1, v_bottom)] - streaks hang below these (window sills)."""
    streak = sample(seed + 11, U * 1.6, V * 0.08, cells=40, octaves=3)          # vertical streaks
    blot = sample(seed + 12, U * 0.25, V * 0.25, cells=8, octaves=4)
    under_eave = np.clip(1 - (y_top - V) / 0.9, 0, 1)
    k = 0.4 * under_eave
    for (u0, u1, vb) in openings:
        inside = np.clip((U - u0) / 0.1, 0, 1) * np.clip((u1 - U) / 0.1, 0, 1)
        below = np.clip((vb - V) / 0.05, 0, 1) * np.clip(1 - (vb - V) / 1.1, 0, 1)
        k = k + 0.46 * inside * below
    splash = np.clip(1 - (V - y_bot) / 0.55, 0, 1) ** 1.5
    m = 1 - (k + 0.08) * np.clip(streak * 0.8 + 0.5, 0, 1.6) - 0.3 * splash * (0.7 + 0.3 * blot) - 0.08 * np.clip(blot, 0, 2)
    return np.clip(m, 0.35, 1.1).astype(np.float32)


def sign_weather(tex, seed=0):
    """Weather an enamel sign texture (h, w, 3): dirt collecting at the bottom and along the edges, rust
    bleeding down from the four bolts and the top edge, faint scratches."""
    h, w = tex.shape[:2]
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    rng = np.random.default_rng(seed)
    n = cv2.resize(rng.random((max(h // 6, 2), max(w // 6, 2))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    st = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=h * 0.05)
    st = (st - st.mean()) / (st.std() + 1e-6)
    edge = np.clip(1 - np.minimum(np.minimum(x, 1 - x) * w, np.minimum(y, 1 - y) * h) / (0.06 * min(h, w) + 1), 0, 1)
    dirt = 0.18 * np.clip((y - 0.55) / 0.45, 0, 1) ** 1.5 + 0.12 * edge + 0.05 * n
    rust_c = np.array([0.45, 0.2, 0.08], np.float32)
    rust = np.zeros((h, w), np.float32)
    for bx in (0.04, 0.96):
        for by in (0.1, 0.9):
            d = np.exp(-(((x - bx) * w / (0.02 * w + 1)) ** 2)) * np.clip((y - by) / 0.02, 0, 1) * np.exp(-(y - by) / 0.25)
            rust += d * (0.6 + 0.4 * np.clip(st, -1, 2))
            rust += np.exp(-(((x - bx) * w) ** 2 + ((y - by) * h) ** 2) / (0.012 * min(h, w) + 1) ** 2)
    rust += np.clip(1 - y / 0.05, 0, 1) * 0.25 * np.clip(st + 0.5, 0, 2)
    # rust blooms where the posts are bolted on (bottom corners) and brown water stains running down
    for bx in (0.07, 0.93):
        rust += 0.9 * np.exp(-((x - bx) * w / (0.035 * w + 1)) ** 2) * np.clip((y - 0.72) / 0.28, 0, 1) ** 1.5 * (0.7 + 0.3 * n)
    stain = np.clip(st * 0.5 + 0.2, 0, 1.5) * np.clip(1 - y / 0.7, 0, 1) * (0.5 + 0.5 * n)
    dirt = dirt + 0.12 * stain
    rust = np.clip(rust, 0, 1)
    out = tex * (1 - dirt[..., None]) * (1 - 0.04 * np.clip(st, -2, 2)[..., None])
    out = out * (1 - 0.6 * rust[..., None]) + rust_c * 0.6 * rust[..., None]
    return out.astype(np.float32)
