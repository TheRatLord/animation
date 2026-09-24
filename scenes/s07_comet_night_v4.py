"""Round-4 painting helpers for s07_comet_night.

- milky_way4: the Milky Way built in band-aligned coordinates from layered star clouds (big soft clouds,
  mid clumps, fine granular knots) with warm peach/gold core patches, cool teal/violet edges and 3 long,
  feathered dark rifts that run WITH the band; tapers into the horizon glow instead of ending in a blob.
- comet_core: hot nucleus with a short cross-flare + tiny fragment sparkles (Your Name split).
- glitter_column: rippled vertical light path on the lake below the comet.
"""
import math
import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def _points_field(Lp, Wd, pts, sig_u, sig_v):
    """Sum of anisotropic gaussians centred at pts (u_px, v_px, amp) on a (Wd, Lp) straight texture."""
    acc = np.zeros((Wd, Lp), np.float32)
    ui = np.clip(pts[:, 0].astype(int), 0, Lp - 1)
    vi = np.clip(pts[:, 1].astype(int), 0, Wd - 1)
    np.add.at(acc, (vi, ui), pts[:, 2].astype(np.float32))
    return cv2.GaussianBlur(acc, (0, 0), sigmaX=sig_u, sigmaY=sig_v)


def milky_way4(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0, s=1.0):
    """Returns (glow RGB additive, dust (H,W), density (H,W)). p0 = horizon end, p1 = far end."""
    rng = np.random.default_rng(seed)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    Lp = int(L) + 2
    half = int(4.2 * width)
    Wd = 2 * half + 1

    # ---- straight (u, v) texture space: columns = along (px), rows = across (px, centred)
    uu = np.arange(Lp, dtype=np.float32)[None, :] / L            # 0..1
    vv = (np.arange(Wd, dtype=np.float32) - half)[:, None]         # px
    wu = width * (1.0 + 0.5 * np.clip(1 - uu, 0, 1) ** 2)          # broader toward the horizon end
    # axis wobble (the band is not a ruler-straight tube)
    wob = P.fbm1d(uu[0], 2.5, 3, seed + 3) * 0.35 * width
    vb = (vv - wob[None, :]) / wu
    env = C.smoothstep(0.0, 0.3, uu) * (1 - 0.35 * C.smoothstep(0.75, 1.0, uu))

    def scatter(n, spread, amp_sigma):
        u_ = rng.uniform(0, 1, n)
        wloc = width * (1.0 + 0.5 * (1 - u_) ** 2)
        v_ = rng.normal(0, spread, n) * wloc + half + np.interp(u_, uu[0], wob)
        a_ = rng.lognormal(0, amp_sigma, n)
        return np.stack([u_ * L, v_, a_], 1)

    c1 = _points_field(Lp, Wd, scatter(int(0.05 * L), 0.5, 0.5), 0.05 * L, 0.32 * width)
    c2 = _points_field(Lp, Wd, scatter(int(0.5 * L), 0.55, 0.6), 0.012 * L, 0.1 * width)
    c3 = _points_field(Lp, Wd, scatter(int(4.0 * L), 0.6, 0.7), 1.6 * s + 0.6, 1.2 * s + 0.5)
    for c in (c1, c2, c3):
        c /= (np.percentile(c, 99.5) + 1e-6)
    clouds = np.clip(c1 * 0.55 + c2 * 0.45, 0, 1.4)
    band = np.exp(-(vb ** 2) * 1.0)
    halo = np.exp(-(vb / 2.4) ** 2)
    along = 0.55 + 0.45 * (P.fbm1d(uu[0], 3.0, 3, seed + 41)[None, :] * 0.5 + 0.5) * 1.4
    lum = band * (0.06 + 1.45 * clouds ** 2.0) * along + band * c3 * 0.12 + halo * 0.07

    # ---- dark rifts: long feathered lanes that run with the band
    dust = np.zeros_like(vb)
    rifts = [  # (u0, u1, centre (band units), half-width (band units), opacity, wiggle, seed)
        (0.02, 0.8, 0.1, 0.2, 0.95, 0.38, 1),
        (0.32, 0.98, -0.45, 0.11, 0.85, 0.3, 2),
        (0.06, 0.42, -0.35, 0.08, 0.75, 0.25, 3),
        (0.52, 0.9, 0.6, 0.07, 0.7, 0.22, 4),
        (0.15, 0.55, 0.45, 0.05, 0.6, 0.3, 5),
        (0.6, 0.95, -0.05, 0.05, 0.6, 0.35, 6),
        (0.25, 0.7, -0.7, 0.06, 0.5, 0.2, 7),
    ]
    for (a, b, cc, hw, op, wg, sd) in rifts:
        q = np.clip((uu - a) / (b - a), 0, 1)
        taper = np.clip(np.sin(np.pi * q), 0, 1) ** 0.7 * ((uu >= a) & (uu <= b))
        cl = cc + 1.5 * wg * P.fbm1d(uu[0], 6.0, 4, seed + 100 + sd)[None, :]
        wdt = hw * (0.55 + 0.45 * (P.fbm1d(uu[0], 7.0, 3, seed + 200 + sd)[None, :] * 0.5 + 0.5) * 1.6)
        d = (vb - cl) / np.maximum(wdt, 1e-3)
        brk = C.smoothstep(-0.25, 0.15, P.fbm1d(uu[0], 12.0, 3, seed + 300 + sd))[None, :]
        prof_ = C.smoothstep(1.25, 0.55, np.abs(d))            # crisp-edged lane
        dust = np.maximum(dust, prof_ * taper * op * (0.15 + 0.85 * brk))
    # torn fingers / fine structure inside the lanes (elongated along the band)
    nz = np.random.default_rng(seed + 9).random((Wd // 3 + 2, Lp // 12 + 2)).astype(np.float32)
    nz = cv2.resize(cv2.GaussianBlur(nz, (0, 0), 1.0), (Lp, Wd), interpolation=cv2.INTER_CUBIC)[:Wd, :Lp]
    nz = (nz - nz.mean()) / (nz.std() + 1e-6)
    dust = np.clip(dust * (0.85 + 0.35 * nz), 0, 0.92)
    # a few small dark knots (globules) inside the brightest region
    knots = scatter(14, 0.35, 0.3)
    kn = _points_field(Lp, Wd, knots, 0.012 * L, 0.06 * width)
    kn /= kn.max() + 1e-6
    dust = np.maximum(dust, kn * 0.55 * C.smoothstep(0.15, 0.7, uu))
    dust = 0.55 * cv2.GaussianBlur(dust, (0, 0), 1.2 * s + 0.4) + 0.3 * cv2.GaussianBlur(dust, (0, 0), 4 * s + 0.5) + \
        0.15 * cv2.GaussianBlur(dust, (0, 0), 12 * s + 1)
    dust = np.clip(dust * 1.1, 0, 0.9) * env

    glow = lum * (1 - dust) * env
    # ---- colour: blue-white body, warm peach/gold core patches, cool teal / violet edges
    core_zone = np.exp(-((uu - 0.38) / 0.22) ** 2) * np.exp(-(vb / 0.7) ** 2)
    warmth = np.clip(core_zone * (0.6 + 1.4 * c1), 0, 1.0)[..., None]
    body = np.array([0.72, 0.82, 1.0], np.float32)
    warm = np.array([1.0, 0.68, 0.4], np.float32)
    col = body * (1 - warmth) + warm * warmth
    img = glow[..., None] * col * 0.75
    teal = np.exp(-((vb - 1.15) / 0.55) ** 2) * (0.5 + 0.5 * c1)
    viol = np.exp(-((vb + 1.2) / 0.6) ** 2) * (0.5 + 0.5 * c1)
    img += (teal * env)[..., None] * np.array([0.1, 0.34, 0.42], np.float32) * 0.16
    img += (viol * env)[..., None] * np.array([0.32, 0.16, 0.48], np.float32) * 0.13
    # warm-lit rift margins
    edge = np.clip(cv2.GaussianBlur(dust, (0, 0), 5 * s + 0.5) - dust * 0.85, 0, 1) * band * env
    img += edge[..., None] * np.array([1.0, 0.66, 0.45], np.float32) * 0.22
    img *= strength
    dens = np.clip(band * (1 - dust) ** 1.6 * (0.2 + 0.9 * clouds + 0.5 * c3) * env, 0, 1)

    # ---- map straight texture into the plate
    xs, ys = C.grid(W, H)
    u = ((xs - p0[0]) * ux + (ys - p0[1]) * uy)
    v = (-(xs - p0[0]) * uy + (ys - p0[1]) * ux)
    v = v - bend * L * (((u / L) - 0.5) ** 2 * 4 - 1)
    mx = u.astype(np.float32)
    my = (v + half).astype(np.float32)
    rm = lambda a: cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_LINEAR,  # noqa: E731
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rm(img), rm(dust), rm(dens)


# ----------------------------------------------------------------------------- comet nucleus

def cross_flare(img, x, y, L, wd, col, amt, rot=0.0):
    H, W = img.shape[:2]
    R = int(L) + 2
    x0, y0 = int(x) - R, int(y) - R
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
    dx_, dy_ = xx - x, yy - y
    ca, sa = math.cos(rot), math.sin(rot)
    a_, b_ = dx_ * ca + dy_ * sa, -dx_ * sa + dy_ * ca
    k = np.exp(-(b_ / wd) ** 2) * np.clip(1 - np.abs(a_) / L, 0, 1) ** 2.5
    k += np.exp(-(a_ / wd) ** 2) * np.clip(1 - np.abs(b_) / L, 0, 1) ** 2.5
    img[cy0:cy1, cx0:cx1] += k[..., None] * np.asarray(col, np.float32) * amt


# ----------------------------------------------------------------------------- lake glitter

def glitter_column(Hlk, Wp, x0, y_top, y_bot, H, s, col_core, col_soft):
    """Vertical path of comet light on the water from the far shore (y_top) down to the mirrored head
    (y_bot) and beyond; narrow near the horizon, widening toward the viewer."""
    xs, ys = C.grid(Wp, Hlk)
    d = np.clip(ys - y_top, 0, None)
    span = max(y_bot - y_top, 1.0)
    q = d / span
    wd = (1.5 * s + 0.5) + 0.012 * H * q ** 1.2
    prof = C.smoothstep(-1, 3, ys - y_top) * (0.7 + 0.8 * np.exp(-((q - 1.0) / 0.55) ** 2) + 0.25 * q)
    core = np.exp(-((xs - x0) / wd) ** 2) * prof
    soft = np.exp(-((xs - x0) / (wd * 4 + 0.01 * H)) ** 2) * prof
    return (core[..., None] * np.asarray(col_core, np.float32) +
            soft[..., None] * np.asarray(col_soft, np.float32)).astype(np.float32)


def tail_glitter(Hlk, Wp, head, d, L, y_h, lk0, H, s, col, amt=0.12, n=48, bend=0.0):
    """Reflection of the tail: each point of the tail throws its own (dim) vertical glitter column from the
    shore down to its mirrored position -> a cool, streaky wash beside the main comet path."""
    out = np.zeros((Hlk, Wp, 3), np.float32)
    col = np.asarray(col, np.float32)
    for i in range(1, n + 1):
        u = i / n * 0.7
        b = math.exp(-u / 0.25) * (1 - u) ** 1.4
        off = bend * L * u * u
        x = head[0] + d[0] * u * L - d[1] * off
        y = head[1] + d[1] * u * L + d[0] * off
        ym = 2 * y_h - y - lk0
        wd = (2.0 + 10.0 * u) * s + 1.0
        x0, x1 = int(max(x - 4 * wd, 0)), int(min(x + 4 * wd + 1, Wp))
        if x1 <= x0:
            continue
        y1 = int(min(ym + 0.06 * H, Hlk))
        if y1 <= 0:
            continue
        yy = np.arange(y1, dtype=np.float32)[:, None]
        xx = np.arange(x0, x1, dtype=np.float32)[None, :]
        prof = (0.3 + 0.7 * np.clip(yy / max(ym, 1.0), 0, 1) ** 1.5) * np.exp(-np.clip(yy - ym, 0, None) / (0.03 * H))
        k = np.exp(-((xx - x) / wd) ** 2) * prof
        out[:y1, x0:x1] += k[..., None] * col * (amt * b)
    return out
