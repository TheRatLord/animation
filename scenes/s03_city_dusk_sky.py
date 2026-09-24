"""Painted afterglow sky for s03_city_dusk: long torn cloud streaks lit from a sun just below the horizon.

After sunset the high clouds still catch direct light (brilliant pink / gold undersides) while low clouds
sit in the earth's shadow (dark violet bodies with thin burning rims on the sunward edge). Each streak is
a horizontally stretched, noise-torn density field thresholded with a narrow edge (crisp painted
silhouettes); lighting = how open the path toward the sun is (density sampled along the sun direction),
which yields bright lit rims exactly on sun-facing edges and soft internal gradients.
"""
import math
import numpy as np
import cv2

from lib import core as C


def stretched_fbm(W, H, sx, sy, octaves, seed, angle=0.0):
    """fbm with feature size sx (px, horizontal) x sy (px, vertical). 0..1"""
    w = max(int(W / sx * 8), 8)
    h = max(int(H / sy * 8), 8)
    n = C.fbm(w, h, scale=w / 8.0, octaves=octaves, seed=seed, aspect=False)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
    return n.astype(np.float32)


def _noise1d(n, cells, seed, octaves=4):
    rng = np.random.default_rng(seed)
    out = np.zeros(n, np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        c = int(cells * 2 ** o) + 3
        g = rng.random(c).astype(np.float32)
        x = np.linspace(0, c - 3, n)
        out += amp * np.interp(x, np.arange(c), g).astype(np.float32)
        tot += amp
        amp *= 0.5
    out /= tot
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-9)


def streak_density(W, H, specs, seed=0):
    """specs: list of dict(cx, cy, L, T, tier, tilt, wave, sub) in px. Each streak is a long band whose
    centre line wanders and whose thickness varies strongly along its length (1D noise), tapering to wispy
    ends; the top edge is lumpier than the flatter base; stretched 2D noise tears it into fibres.
    returns density (H, W) and per-pixel tier map (0 low .. 1 high)."""
    rng = np.random.default_rng(seed)
    D = np.zeros((H, W), np.float32)
    TIER = np.zeros((H, W), np.float32)
    n_fib = stretched_fbm(W, H, 0.05 * W, 0.004 * H, 4, seed + 3)
    n_lump = stretched_fbm(W, H, 0.012 * W, 0.01 * H, 3, seed + 4)
    xs_full = np.arange(W, dtype=np.float32)
    for si, s in enumerate(specs):
        cx, cy, L, T = s['cx'], s['cy'], s['L'], s['T']
        for sub in range(s.get('sub', 2)):
            sd = seed * 100 + si * 10 + sub
            off_y = (sub * 1.15 - 0.3) * T * rng.uniform(0.8, 1.6) * (1 if sub % 2 else -1) if sub else 0.0
            off_x = rng.uniform(-0.15, 0.15) * L if sub else 0.0
            Ls = L * (1.0 if sub == 0 else rng.uniform(0.3, 0.6))
            Ts = T * (1.0 if sub == 0 else rng.uniform(0.5, 0.8))
            x0b, x1b = int(max(cx + off_x - Ls * 0.55, 0)), int(min(cx + off_x + Ls * 0.55, W))
            if x1b <= x0b:
                continue
            xs = xs_full[x0b:x1b]
            u = (xs - (cx + off_x)) / (Ls / 2)
            nth = _noise1d(len(xs), max(Ls / W * 9, 2), sd + 1)
            nwv = _noise1d(len(xs), max(Ls / W * 3, 1.5), sd + 2)
            env = np.clip(1 - np.abs(u) ** 2.2, 0, 1) ** 0.8
            thick = Ts * (0.32 + 1.2 * nth ** 1.5) * env ** 0.7 + 1e-3
            yc = cy + off_y + s.get('tilt', 0.0) * (xs - cx) + (nwv - 0.5) * Ts * s.get('wave', 3.0) * (1.0 if sub == 0 else 0.4)
            y0b = int(max(np.min(yc - thick * 2.2) - 2, 0))
            y1b = int(min(np.max(yc + thick * 1.6) + 2, H))
            if y1b <= y0b:
                continue
            ys = np.arange(y0b, y1b, dtype=np.float32)[:, None]
            v = (ys - yc[None, :]) / thick[None, :]
            lump = n_lump[y0b:y1b, x0b:x1b]
            fib = n_fib[y0b:y1b, x0b:x1b]
            # asymmetric profile: flat base, lumpy top
            vb = np.where(v > 0, v * 1.5, v * (0.8 + 0.6 * (1 - lump)))
            d = np.exp(-vb * vb) * (0.55 + 0.9 * fib) * (0.8 + 0.4 * env[None, :])
            sl = (slice(y0b, y1b), slice(x0b, x1b))
            np.maximum(D[sl], d, out=D[sl])
            tsub = TIER[sl]
            m = d > 0.25
            tsub[m] = np.maximum(tsub[m], s.get('tier', 0.5))
    return D, TIER


def paint(W, H, sky, specs, sun, seed=0, thresh=0.5, edge=0.035):
    """Returns RGBA straight-alpha plate (H, W, 4) of lit cloud streaks over `sky` (H, W, 3)."""
    D, TIER = streak_density(W, H, specs, seed)
    # crisp painted silhouette
    A = C.smoothstep(thresh - edge * 0.5, thresh + edge * 0.5, D)
    A = C.blur(A, 0.45)
    # internal density (for translucency near edges)
    dens = C.smoothstep(thresh, thresh + 0.6, D)
    # ---- lighting: openness toward the sun
    xs, ys = C.grid(W, H)
    dx, dy = sun[0] - xs, sun[1] - ys
    dist = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dist, dy / dist
    Ab = C.blur(A, 1.5)
    occ = np.zeros((H, W), np.float32)
    wsum = 0.0
    for k, wt in ((0.004, 1.0), (0.01, 0.8), (0.022, 0.6), (0.045, 0.4)):
        off = k * W
        smp = cv2.remap(Ab, (xs + ux * off).astype(np.float32), (ys + uy * off).astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        occ += smp * wt
        wsum += wt
    occ /= wsum
    lit = np.clip(1 - occ * 1.25, 0, 1) ** 1.3                       # 1 = fully exposed to the sun
    # rim (round 6): only on the sun-facing LOWER edges (the sun is below the horizon), with a width that
    # varies along each streak (1-4.5 px @1080p) and a strength that comes and goes - no outline all round
    sc = W / 1920.0
    wv = stretched_fbm(W, H, 0.06 * W, 0.05 * H, 2, seed + 51)
    wv2 = stretched_fbm(W, H, 0.025 * W, 0.05 * H, 2, seed + 52)
    off = (1.0 + 3.5 * wv) * sc
    near = cv2.remap(Ab, xs.astype(np.float32), (ys + off).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rim = np.clip(A - near, 0, 1) * 1.8 * C.smoothstep(0.25, 0.7, wv2) * np.clip(uy * 3.0, 0, 1)
    rim = np.clip(rim, 0, 1)
    # the upper (shadowed) edges are feathered into the sky instead of outlined
    up = cv2.remap(Ab, xs.astype(np.float32), (ys - 3.0 * sc).astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    top_edge = C.blur(np.clip(A - up, 0, 1), 1.5 * sc)
    # ---- colours
    # proximity to the sun (horizontal) -> gold/white; far -> pink/magenta
    prox = np.exp(-dist / (0.35 * W))
    elev = np.clip((sun[1] - ys) / (0.7 * H), 0, 1)                   # height above the sun
    tier = TIER
    # lit colour: high clouds hot pink-orange, near the sun gold-white
    lit_far = C.lerp(np.array([1.0, 0.4, 0.5], np.float32), np.array([1.05, 0.55, 0.4], np.float32),
                     (1 - elev)[..., None])
    lit_col = C.lerp(lit_far, np.array([1.2, 0.82, 0.48], np.float32), (prox ** 1.2)[..., None])
    # shadow body: low clouds in the earth's shadow = dark violet; high ones lavender-pink
    shd_low = np.array([0.27, 0.2, 0.42], np.float32)
    shd_high = np.array([0.62, 0.42, 0.66], np.float32)
    shd = C.lerp(shd_low, shd_high, tier[..., None])
    # blend a bit of the sky behind into the body (air between)
    shd = C.lerp(shd, sky, 0.18)
    # how strongly each tier receives direct light: high clouds fully, low clouds mostly rims
    recv = (0.25 + 0.75 * tier)[..., None]
    col = C.lerp(shd, lit_col, (np.clip(lit * recv[..., 0] * 1.25, 0, 1))[..., None])
    # internal soft gradient: slightly darker cores
    col = col * (0.9 + 0.1 * (1 - dens))[..., None]
    rimc = C.lerp(np.array([1.0, 0.45, 0.32], np.float32), np.array([1.3, 0.9, 0.5], np.float32), prox[..., None])
    col = col + rim[..., None] * rimc * (0.45 + 0.8 * prox)[..., None] * (0.85 - 0.3 * tier)[..., None]
    # underside glow: the lower (sun-facing) band of every streak burns gold/coral
    below = cv2.remap(Ab, xs.astype(np.float32), (ys + 0.006 * H).astype(np.float32), cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    under = np.clip(A - below, 0, 1)
    under = C.blur(under, 0.8) * 1.5 * (0.35 + 0.65 * C.smoothstep(0.2, 0.75, wv2))
    col = C.lerp(col, rimc * 1.1, np.clip(under * (0.5 + 0.5 * prox), 0, 0.85)[..., None])
    # fibrous brush texture along the streaks (painted, not plastic)
    fib = stretched_fbm(W, H, 0.09 * W, 0.0035 * H, 3, seed + 41)
    fib2 = stretched_fbm(W, H, 0.03 * W, 0.002 * H, 2, seed + 43)
    col = col * (0.86 + 0.2 * fib + 0.1 * fib2)[..., None]
    # thin edges are translucent (sky shows through, glow)
    alpha = A * (0.55 + 0.45 * dens) * (1 - 0.45 * top_edge)
    out = np.dstack([col, alpha]).astype(np.float32)
    return out


def fast_bloom(img, threshold=0.8, knee=0.3, strength=0.35, halation=0.2):
    """Cheaper equivalent of fx.bloom_soft: soft-knee bright pass at quarter resolution, 4-level pyramid
    blur, warm halation; returns img + bloom."""
    H, W = img.shape[:2]
    q = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    lum = q.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
    br = q * (k * k)
    acc = np.zeros_like(br)
    wts = (1.0, 0.8, 0.6, 0.45)
    cur = br
    lv = []
    for i in range(4):
        cur = cv2.GaussianBlur(cur, (0, 0), 1.6)
        lv.append(cur)
        if i < 3:
            cur = cv2.resize(cur, (max(cur.shape[1] // 2, 2), max(cur.shape[0] // 2, 2)), interpolation=cv2.INTER_AREA)
    h4, w4 = br.shape[:2]
    for L, wt in zip(lv, wts):
        acc += cv2.resize(L, (w4, h4), interpolation=cv2.INTER_LINEAR) * wt
    acc /= sum(wts)
    hal = cv2.GaussianBlur(br, (0, 0), max(0.006 * W / 4, 0.8))
    tot = acc * strength + hal * np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5
    tot = cv2.resize(tot, (W, H), interpolation=cv2.INTER_LINEAR)
    return (img + tot).astype(np.float32)


from numba import njit, prange  # noqa: E402


@njit(cache=True, parallel=True, fastmath=True)
def _radial_march(src, x0, y0, length, steps):
    h, w = src.shape
    out = np.zeros((h, w), np.float32)
    for y in prange(h):
        for x in range(w):
            acc = 0.0
            ws = 0.0
            for i in range(steps):
                k = 1.0 - length * (i / steps) ** 1.15
                sx = x0 + (x - x0) * k
                sy = y0 + (y - y0) * k
                wt = 1.0 - 0.5 * i / steps
                ws += wt
                if sx < 0 or sy < 0 or sx > w - 1.001 or sy > h - 1.001:
                    continue
                ix = int(sx)
                iy = int(sy)
                fx = sx - ix
                fy = sy - iy
                v = (src[iy, ix] * (1 - fx) + src[iy, ix + 1] * fx) * (1 - fy) + \
                    (src[iy + 1, ix] * (1 - fx) + src[iy + 1, ix + 1] * fx) * fy
                acc += v * wt
            out[y, x] = acc / ws
    return out


def shafts(W, H, lx, ly, occluder, strength=0.8, length=0.95, radius=0.12, tint=(1.0, 0.6, 0.38), t=0.0,
           streaks=0.5, n_beams=9, seed=4, q=4):
    """Fast crepuscular rays (numba radial march at 1/q res): bright horizon source masked by the occluder
    (towers + clouds), smeared radially away from the light. Returns additive (H, W, 3)."""
    w, h = W // q, H // q
    x0, y0 = lx / q, ly / q
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2) / w
    src = np.exp(-(d / radius) ** 2) * 0.7 + 0.2 / (1 + (d / 0.03) ** 2)
    occ = cv2.resize(np.clip(occluder, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    src = (src * (1 - occ) ** 1.5).astype(np.float32)
    acc = _radial_march(src, np.float32(x0), np.float32(y0), np.float32(length), 36)
    if streaks:
        ang = np.arctan2(ys - y0, xs - x0)
        rng = np.random.default_rng(seed)
        pat = np.zeros_like(ang)
        tot = 0.0
        for j in range(3):
            fq = n_beams * (1 + 0.6 * j) + rng.uniform(-1, 1)
            a = 1.0 / (1 + j)
            pat += a * np.sin(ang * round(fq) + rng.uniform(0, 6.28) + t * rng.uniform(0.15, 0.35) * (1 if j % 2 else -1))
            tot += a
        pat = pat / tot
        pat = 1 + streaks * C.smoothstep(-0.6, 0.9, pat) * 2 - streaks
        acc = acc * np.clip(pat, 0.1, 2.0)
    acc = acc * (1 + 0.06 * math.sin(t * 1.1 + seed))
    acc = cv2.GaussianBlur(acc.astype(np.float32), (0, 0), 1.2)
    acc = acc * (1 - 0.85 * occ)
    x = acc * strength
    x = x / (1 + 0.9 * x)
    out = cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return out[..., None] * np.asarray(tint, np.float32)
