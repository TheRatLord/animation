"""Refined optical FX for the Shinkai-style montage (softer, better tuned than core.lens_flare/god_rays).

All functions return ADDITIVE float32 (H, W, 3) images (add them to the frame before bloom/finish)
unless stated otherwise. Everything is deterministic for a given seed and smooth in the light
position, so a moving sun gives temporally coherent flares.

anime_flare(W, H, lx, ly, ...)          hot core + broad bloom + few soft spikes + faint chromatic ghosts
light_shafts(W, H, lx, ly, occluder)    crepuscular rays through gaps in an occluder alpha (clouds, trees)
bloom_soft(img, ...)                     multi-scale bloom with warm halation, soft knee (no hard clipping)
glints(W, H, xs, ys, ...)                4-point star specular glints (water, rails, glass, wires)
sun_visibility(alpha, lx, ly, r)         0..1 how much of the sun disc is unoccluded (dim the flare with it)
aerial(img, amount, color)               atmospheric-perspective tint for far layers
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from . import core as C


def _ang_profile(n_rays, seed, sharp=60.0):
    """Precomputed 1D angular ray profile: returns function(angle_array) -> (amp, length_scale)."""
    rng = np.random.default_rng(seed)
    N = 2048
    a = np.linspace(-math.pi, math.pi, N, endpoint=False)
    amp = np.zeros(N, np.float32)
    ln = np.zeros(N, np.float32)
    base = rng.uniform(0, 2 * math.pi)
    for i in range(n_rays):
        ang = base + i * 2 * math.pi / n_rays + rng.normal(0, 0.06)
        w = 1.0 / sharp * rng.uniform(0.6, 1.4)
        d = np.angle(np.exp(1j * (a - ang)))
        k = np.exp(-(d / w) ** 2)
        amp = np.maximum(amp, k * rng.uniform(0.4, 1.0))
        ln = np.maximum(ln, k * rng.uniform(0.35, 1.0))
    # many faint hair-thin rays
    for i in range(n_rays * 4):
        ang = rng.uniform(-math.pi, math.pi)
        w = 1.0 / (sharp * 2.5)
        d = np.angle(np.exp(1j * (a - ang)))
        k = np.exp(-(d / w) ** 2)
        amp = np.maximum(amp, k * rng.uniform(0.1, 0.35))
        ln = np.maximum(ln, k * rng.uniform(0.2, 0.6))
    return a, amp, ln


_PROFILE_CACHE = {}


def fast_blur(img, sigma):
    """Gaussian-like blur whose cost does not grow with sigma: downsample so the residual sigma is
    ~3 px, blur, upsample (bilinear). Good for bloom/glow; not for crisp detail."""
    if sigma <= 3.0:
        return C.blur(img, sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.0 * 0.85)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def anime_flare(W, H, lx, ly, intensity=1.0, tint=(1.0, 0.93, 0.8), seed=3, rays=6, ray_len=0.055,
                starburst=1.0, ghosts=1.0, halo=1.0, streak=0.2, glow=1.0, rot=0.0, spike_width=0.06, hair=0.0,
                center=None):
    """Anime lens flare around a light at (lx, ly) px (may be slightly off-frame).

    Shinkai-style: a compact hot core, a wide soft warm bloom, a few (4-6) SHORT soft low-opacity spikes,
    a faint rainbow halo ring, 5 faint chromatic ghosts (hexagons + rings) along the axis from the light
    through the frame centre (they slide as the camera/sun moves: pass the on-screen sun position each
    frame), and a subtle anamorphic streak. No hairline rays by default.
    center: optional (x, y) optical centre (default frame centre) - shift it slightly with the camera.
    params: intensity - global multiplier (use sun_visibility() to dim when the sun is occluded);
            tint - warm colour of the glow; rays - number of soft spikes (0 = none); ray_len - spike
            length as a fraction of W; spike_width - angular half-width of a spike (radians);
            starburst/ghosts/halo/streak/glow - component multipliers; hair - optional faint hairline
            rays (0 = off); rot - spike rotation in radians (animate VERY slowly, e.g. 0.02*t).
    returns: (H, W, 3) additive, HDR (core > 1). Half-res internally -> ~0.1-0.2 s at 1080p.
    """
    s = 2
    w, h = W // s, H // s
    x0, y0 = lx / s, ly / s
    xs, ys = C.grid(w, h)
    dx, dy = xs - x0, ys - y0
    d = np.sqrt(dx * dx + dy * dy) + 1e-3
    out = np.zeros((h, w, 3), np.float32)
    tint = np.asarray(tint, np.float32)
    dW = d / w
    if glow:
        core = 1.0 / (1.0 + (dW / 0.004) ** 2.4)
        broad = 1.0 / (1.0 + (dW / 0.022) ** 2) * 0.1 + np.exp(-dW / 0.12) * 0.06 + np.exp(-dW / 0.35) * 0.03
        out += core[..., None] * np.array([1.0, 0.98, 0.94], np.float32) * 0.9 * glow
        out += broad[..., None] * tint * glow
    if starburst and rays:
        rng = np.random.default_rng(seed)
        ang = np.arctan2(dy, dx) - rot
        sp = np.zeros_like(d)
        base = rng.uniform(0, 2 * math.pi)
        for i in range(rays):
            a0 = base + i * 2 * math.pi / rays + rng.normal(0, 0.08)
            da = np.angle(np.exp(1j * (ang - a0))).astype(np.float32)
            wdt = spike_width * rng.uniform(0.7, 1.3)
            L = ray_len * rng.uniform(0.55, 1.0)
            # spikes widen slightly with distance (soft wedge) and fade out smoothly
            wd = wdt * (1 + 2.5 * dW / L) * 0.5 + 0.004 / (dW + 0.004)
            sp += np.exp(-(da / wd) ** 2) * np.exp(-dW / (L * 0.45)) * rng.uniform(0.6, 1.0)
        if hair:
            for i in range(rays * 3):
                a0 = rng.uniform(-math.pi, math.pi)
                da = np.angle(np.exp(1j * (ang - a0))).astype(np.float32)
                sp += np.exp(-(da / 0.006) ** 2) * np.exp(-dW / (ray_len * 0.5 * rng.uniform(0.4, 1))) * hair * 0.3
        sp = sp * np.clip(dW / 0.006, 0, 1)
        out += sp[..., None] * (0.6 * tint + 0.4) * 0.07 * starburst
    if halo:
        R = 0.22
        rr = (dW - R) / 0.02
        ring = np.exp(-rr * rr)
        hue = np.stack([np.clip(0.55 + rr * 0.3, 0, 1), np.clip(1 - np.abs(rr) * 0.35, 0, 1),
                        np.clip(0.55 - rr * 0.3, 0, 1)], -1)
        out += ring[..., None] * hue * 0.018 * halo
    if ghosts:
        rng = np.random.default_rng(seed + 1)
        if center is None:
            cx, cy = w / 2, h / 2
        else:
            cx, cy = center[0] / s, center[1] / s
        vx, vy = cx - x0, cy - y0
        # (position along the axis, radius, colour, polygon sides (0 = disc), ring?)
        spec = [(0.38, 0.014, (0.55, 0.85, 1.0), 6, False), (0.7, 0.04, (0.55, 1.0, 0.72), 6, False),
                (1.08, 0.022, (1.0, 0.72, 0.5), 0, False), (1.42, 0.07, (0.6, 0.72, 1.0), 6, True),
                (1.8, 0.028, (1.0, 0.6, 0.85), 0, False)]
        for k, r, col, sides, ring_ in spec:
            gx, gy = x0 + vx * k, y0 + vy * k
            R = r * w
            bx0, bx1 = int(max(gx - R * 1.3 - 2, 0)), int(min(gx + R * 1.3 + 3, w))
            by0, by1 = int(max(gy - R * 1.3 - 2, 0)), int(min(gy + R * 1.3 + 3, h))
            if bx1 <= bx0 or by1 <= by0:
                continue
            ex, ey = xs[by0:by1, bx0:bx1] - gx, ys[by0:by1, bx0:bx1] - gy
            rad = np.sqrt(ex * ex + ey * ey)
            if sides:
                th = np.arctan2(ey, ex) + 0.3
                seg = 2 * math.pi / sides
                rad = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
            col = np.asarray(col, np.float32)
            q = rad / R
            # soft disc with a slightly brighter chromatic rim (R outside, B inside)
            disc = np.stack([C.smoothstep(1.08, 0.85, q), C.smoothstep(1.03, 0.8, q), C.smoothstep(0.98, 0.75, q)], -1)
            edge = C.smoothstep(0.5, 1.0, q)[..., None]
            amt = 0.028 * rng.uniform(0.7, 1.2) * (0.45 + 0.55 * edge)
            if ring_:
                amt = amt * np.exp(-((q - 0.93) / 0.07) ** 2)[..., None] * 2.2
            if r > 0.06:
                amt = amt * 0.5
            out[by0:by1, bx0:bx1] += C.blur(disc.astype(np.float32), max(R * 0.04, 0.6)) * col * amt * ghosts
    if streak:
        sy = np.exp(-(dy / (0.004 * h)) ** 2)
        sx = np.exp(-np.abs(dx) / (0.22 * w)) * 0.6 + np.exp(-np.abs(dx) / (0.05 * w)) * 0.5
        out += (sy * sx)[..., None] * np.array([0.8, 0.9, 1.0], np.float32) * 0.3 * streak
    out = cv2.resize(out.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return (out * intensity).astype(np.float32)


def sun_visibility(alpha, lx, ly, radius):
    """Fraction (0..1) of a disc at (lx, ly) with `radius` px NOT covered by `alpha` (H, W).
    Use it to fade the flare / sun glare when the sun slides behind clouds or buildings."""
    H, W = alpha.shape[:2]
    r = max(int(radius), 1)
    x0, x1 = int(max(lx - r, 0)), int(min(lx + r + 1, W))
    y0, y1 = int(max(ly - r, 0)), int(min(ly + r + 1, H))
    if x1 <= x0 or y1 <= y0:
        return 1.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = ((xx - lx) ** 2 + (yy - ly) ** 2) <= r * r
    if not m.any():
        return 1.0
    return float(1.0 - np.clip(alpha[y0:y1, x0:x1][m], 0, 1).mean())


def light_shafts(W, H, lx, ly, occluder=None, strength=0.6, length=0.9, tint=(1.0, 0.9, 0.72), radius=0.12,
                 streaks=0.0, seed=7, rot=0.0, source=None, t=None, n_beams=7, hollow=0.0):
    """Crepuscular rays: wide, soft light wedges emanating from (lx, ly), shaped by gaps in `occluder`.

    params: occluder - (H, W) alpha of things that block light (clouds, trees, buildings): the source
            (bright sky around the sun) is masked by it and radially smeared, so beams appear where light
            passes between clouds and shadow-wedges fan out behind them. None = open sky.
            strength - brightness; length - how far rays extend (0..1 of the distance to the light);
            tint - ray colour; radius - size (fraction of W) of the bright source region;
            streaks - extra synthetic angular beam structure (default 0: beams come only from the real
                      gaps/edges of the occluder, i.e. volumetric rays radiating past cloud edges);
            hollow - suppress the source right at the light (0..1) if the core would blow out;
            n_beams - rough number of distinct wedges; seed; rot - rotate the beam pattern;
            t - scene time: gives a slow, gentle shimmer (beams breathe/shift slightly), coherent;
            source - optional custom (H, W) source brightness image instead of the radial one.
    returns: (H, W, 3) additive, soft-compressed so it never blows out (x / (1 + x)). ~0.1 s at 1080p.
    """
    q = 4
    w, h = max(W // q, 8), max(H // q, 8)
    x0, y0 = lx / q, ly / q
    xs, ys = C.grid(w, h)
    if source is None:
        d = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2) / w
        src = np.exp(-(d / radius) ** 2) * 0.7 + 0.2 / (1 + (d / 0.03) ** 2)
        src = src * (1 - hollow * np.exp(-(d / (radius * 0.35)) ** 2))
    else:
        src = cv2.resize(source.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    src = src.astype(np.float32)
    occ = None
    if occluder is not None:
        occ = cv2.resize(np.clip(occluder, 0, 1).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        src = src * np.clip(1 - occ, 0, 1) ** 1.5      # INTER_AREA can overshoot 1 -> NaN
    steps = 36
    acc = np.zeros((h, w), np.float32)
    wsum = 0.0
    for i in range(steps):
        k = 1.0 - length * (i / steps) ** 1.15
        M = np.array([[k, 0, (1 - k) * x0], [0, k, (1 - k) * y0]], np.float32)
        wt = 1.0 - 0.5 * i / steps
        acc += cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE) * wt
        wsum += wt
    acc /= wsum
    if streaks:
        ang = np.arctan2(ys - y0, xs - x0) - rot
        rng = np.random.default_rng(seed)
        tt = 0.0 if t is None else float(t)
        pat = np.zeros_like(ang)
        tot = 0.0
        for j in range(3):
            f = n_beams * (1 + 0.6 * j) + rng.uniform(-1, 1)
            a = 1.0 / (1 + j)
            pat += a * np.sin(ang * round(f) + rng.uniform(0, 6.28) + tt * rng.uniform(0.15, 0.35) * (1 if j % 2 else -1))
            tot += a
        pat = pat / tot
        pat = 1 + streaks * C.smoothstep(-0.6, 0.9, pat) * 2 - streaks
        acc = acc * np.clip(pat, 0.1, 2.0)
    if t is not None:
        acc = acc * (1 + 0.06 * math.sin(t * 1.1 + seed))
    acc = C.blur(acc.astype(np.float32), 1.2)
    if occ is not None:
        acc = acc * (1 - 0.85 * occ)  # shafts live in the air, not on the cloud bodies
    x = acc * strength
    x = x / (1 + 0.9 * x)
    out = cv2.resize(x.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    return (np.clip(out, 0, None)[..., None] * np.asarray(tint, np.float32)).astype(np.float32)


def bloom_soft(img, threshold=0.72, knee=0.35, strength=0.45, halation=0.25, radii=(0.003, 0.01, 0.03, 0.08)):
    """Multi-scale bloom with a soft knee + warm halation (red-orange fringe around highlights, the
    'film glow' that makes anime highlights feel luminous). Returns the new image (H, W, 3)."""
    H, W = img.shape[:2]
    x = img[..., :3]
    lum = x.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
    k = k * k
    br = x * np.maximum(k, (lum > threshold + knee))
    small = cv2.resize(br, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    wts = [1.0, 0.8, 0.6, 0.45][:len(radii)]
    for r, wt in zip(radii, wts):
        acc += fast_blur(small, r * W / 2) * wt
    acc /= sum(wts)
    acc = cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)
    out = x + acc * strength
    if halation:
        hal = fast_blur(small, 0.006 * W / 2)
        hal = cv2.resize(hal, (W, H), interpolation=cv2.INTER_LINEAR)
        out = out + hal * np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5
    return out.astype(np.float32)


def glints(W, H, xs, ys, size=0.02, intensity=1.0, color=(1.0, 0.97, 0.9), angle=0.0, t=None, seed=0, arms=4):
    """Star-shaped specular glints (thin cross flares + soft core) at points - for water sparkle,
    rail/wire/glass highlights.

    params: xs, ys - arrays of px positions; size - arm length fraction of W (scalar or array);
            intensity - scalar or array; angle - arm rotation (rad); arms - 4 (cross) or 6/8;
            t - if given, each glint twinkles smoothly over time with its own phase (coherent).
    returns: (H, W, 3) additive.
    """
    xs = np.atleast_1d(np.asarray(xs, np.float32))
    ys = np.atleast_1d(np.asarray(ys, np.float32))
    n = len(xs)
    size = np.broadcast_to(np.atleast_1d(size), (n,)).astype(np.float32)
    inten = np.broadcast_to(np.atleast_1d(intensity), (n,)).astype(np.float32).copy()
    if t is not None:
        rng = np.random.default_rng(seed)
        ph = rng.uniform(0, 6.28, n)
        fr = rng.uniform(1.0, 3.0, n)
        inten = inten * np.clip(0.5 + 0.5 * np.sin(t * fr + ph), 0, 1) ** 2
    out = np.zeros((H, W, 3), np.float32)
    col = np.asarray(color, np.float32)
    for i in range(n):
        if inten[i] < 1e-3:
            continue
        L = size[i] * W
        R = int(L) + 2
        x0, y0 = int(xs[i]) - R, int(ys[i]) - R
        x1, y1 = int(xs[i]) + R + 1, int(ys[i]) + R + 1
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        dx, dy = xx - xs[i], yy - ys[i]
        k = np.zeros_like(dx)
        for a in range(arms // 2):
            th = angle + a * math.pi / (arms // 2)
            u = dx * math.cos(th) + dy * math.sin(th)
            v = -dx * math.sin(th) + dy * math.cos(th)
            wdt = 0.6 + 0.02 * L
            k += np.exp(-(v / wdt) ** 2) * np.clip(1 - np.abs(u) / L, 0, 1) ** 2.5 * (1.0 if a == 0 else 0.7)
        r2 = (dx * dx + dy * dy) / (max(L * 0.08, 0.8) ** 2)
        k += np.exp(-r2) * 1.5
        out[cy0:cy1, cx0:cx1] += k[..., None] * col * inten[i]
    return out


def aerial(img, amount, color):
    """Atmospheric perspective: blend img toward haze `color` by `amount` (scalar or (H, W) map).
    Distant layers: amount 0.3-0.7 with the sky's horizon colour."""
    a = np.asarray(amount, np.float32)
    if a.ndim == 2:
        a = a[..., None]
    return img[..., :3] * (1 - a) + np.asarray(color, np.float32) * a


@njit(cache=True, fastmath=True, parallel=True)
def _shoulder(x, s, desat):
    H, W = x.shape[0], x.shape[1]
    out = np.empty((H, W, 3), np.float32)
    k = 1.0 - s
    for i in prange(H):
        for j in range(W):
            m = 0.0
            for c in range(3):
                v = max(x[i, j, c], 0.0)
                if v > m:
                    m = v
            wd = min(max((m - 1.0) / 1.5, 0.0), 1.0) * desat
            ym = 0.0
            for c in range(3):
                v = max(x[i, j, c], 0.0)
                if v > s:
                    o = v - s
                    v = s + k * o / (o + k)
                out[i, j, c] = v
                if v > ym:
                    ym = v
            if wd > 0.0:
                for c in range(3):
                    out[i, j, c] = out[i, j, c] * (1.0 - wd) + ym * wd
    return out


def shoulder(img, start=0.8, desat=0.35):
    """Soft highlight roll-off (call just before core.finish): values above `start` are compressed
    smoothly toward 1.0 (rational curve, slope 1 at `start`) instead of hard-clipping, so big bright
    areas (sun glare, lit clouds) keep gradation. Very bright pixels drift toward white (desat) like
    film. Returns (H, W, 3) float32. ~0.03 s at 1080p (numba)."""
    x = np.ascontiguousarray(img[..., :3], dtype=np.float32)
    return _shoulder(x, np.float32(start), np.float32(desat))


# ----------------------------------------------------------------------------- fast compositing / finishing

@njit(cache=True, fastmath=True, parallel=True)
def _over_rgba(dst, src):
    H, W = dst.shape[0], dst.shape[1]
    out = np.empty((H, W, 3), np.float32)
    for i in prange(H):
        for j in range(W):
            a = src[i, j, 3]
            if a < 0.0:
                a = 0.0
            elif a > 1.0:
                a = 1.0
            for c in range(3):
                out[i, j, c] = dst[i, j, c] * (1.0 - a) + src[i, j, c] * a
    return out


def over_rgba(dst, rgba):
    """Fast (numba, parallel) straight-alpha 'over': composite an (H, W, 4) plate onto (H, W, 3).
    Same result as core.over(dst, rgba[..., :3], rgba[..., 3]) but ~10x faster at 1080p."""
    return _over_rgba(np.ascontiguousarray(dst[..., :3], dtype=np.float32), np.ascontiguousarray(rgba, dtype=np.float32))


@njit(cache=True, fastmath=True, parallel=True)
def _finish_kernel(x, vig, sat, gr, gamt):
    H, W = x.shape[0], x.shape[1]
    out = np.empty((H, W, 3), np.float32)
    for i in prange(H):
        for j in range(W):
            r, g, b = x[i, j, 0], x[i, j, 1], x[i, j, 2]
            l = 0.2126 * r + 0.7152 * g + 0.0722 * b
            r = l + (r - l) * sat
            g = l + (g - l) * sat
            b = l + (b - l) * sat
            v = vig[i, j]
            lw = min(max(l, 0.0), 1.0)
            n = gr[i, j] * gamt * (0.25 + 3.0 * lw * (1.0 - lw))
            out[i, j, 0] = min(max(r * (1 - v) + n, 0.0), 1.0)
            out[i, j, 1] = min(max(g * (1 - v) + n, 0.0), 1.0)
            out[i, j, 2] = min(max(b * (1 - v) + n, 0.0), 1.0)
    return out


_VIG_CACHE = {}


def finish_fast(img, t=0.0, exposure=1.0, sat=1.08, grain_amt=0.005, vig=0.3, ca=0.0012):
    """Faster equivalent of core.finish (tonemap off): exposure, saturation, chromatic aberration,
    vignette and per-frame film grain. Grain is luminance-weighted (strongest in mid-tones, ~4x weaker
    in deep shadows / bright highlights and flat haze) - keep grain_amt around 0.004-0.006.
    ~0.1 s at 1080p vs ~0.45 s for core.finish."""
    H, W = img.shape[:2]
    x = img[..., :3].astype(np.float32) * np.float32(exposure)
    if ca:
        x = C.chromatic_aberration(x, ca)
    key = (W, H, float(vig))
    if key not in _VIG_CACHE:
        xs, ys = C.grid(W, H)
        d = np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
        _VIG_CACHE[key] = (d ** 2.2 * vig).astype(np.float32)
    if grain_amt:
        rng = np.random.default_rng(int(t * C.FPS) + 7)
        n = rng.standard_normal((H // 2 + 1, W // 2 + 1)).astype(np.float32)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        n = np.zeros((H, W), np.float32)
    return _finish_kernel(np.ascontiguousarray(x), _VIG_CACHE[key], np.float32(sat), n, np.float32(grain_amt))
