"""Round-8 Milky Way for s07_comet_night: structured, not a noise band.

  * a bright warm cream-peach core bulge,
  * mottled star clouds (multi-scale, clumpy, granular),
  * dark rift lanes: branching, filament-edged dust clouds, crisp on one side and feathered on the other,
  * small bright knots (clusters / emission patches),
  * band edges that break up into wisps instead of a smooth gaussian falloff.
"""
import math
import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def stretched_fbm(Wd, Lp, cells_u, cells_v, octaves, seed, stretch=4.0, gain=0.55):
    """fbm on a (Wd rows = across, Lp cols = along) texture, features elongated along the columns."""
    lw = max(int(Lp / stretch), 16)
    n = np.zeros((Wd, lw), np.float32)
    amp, tot = 1.0, 0.0
    cu, cv = cells_u, cells_v
    for o in range(octaves):
        n += amp * C.value_noise(lw, Wd, max(1, cu), max(1, cv), seed + 57 * o)
        tot += amp
        amp *= gain
        cu *= 2.0
        cv *= 2.0
    n /= tot
    n = cv2.resize(n, (Lp, Wd), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(n, 1), np.percentile(n, 99)
    return np.clip((n - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def milky_way6(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0, s=1.0, core_u=0.3):
    """Returns (glow RGB additive, dust (H,W), density (H,W)). p0 = horizon end, p1 = far end."""
    rng = np.random.default_rng(seed)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    Lp = int(L) + 2
    half = int(3.6 * width)
    Wd = 2 * half + 1
    uu = np.arange(Lp, dtype=np.float32)[None, :] / L
    vv = (np.arange(Wd, dtype=np.float32) - half)[:, None]
    wu = width * (1.0 + 0.3 * np.clip(1 - uu, 0, 1) ** 2 + 0.35 * np.exp(-((uu - core_u) / 0.12) ** 2))
    wob = P.fbm1d(uu[0], 2.5, 3, seed + 3)[None, :] * 0.3 * width
    vb = (vv - wob) / wu                                           # band units (0 = axis)
    env = C.smoothstep(0.0, 0.22, uu) * (1 - 0.4 * C.smoothstep(0.7, 1.0, uu))

    # ---- noise fields (texture space)
    wn1 = stretched_fbm(Wd, Lp, 6, 10, 4, seed + 10, stretch=3.0)
    wn2 = stretched_fbm(Wd, Lp, 14, 22, 4, seed + 11, stretch=2.5)
    fine = stretched_fbm(Wd, Lp, 40, 60, 3, seed + 12, stretch=1.6)
    # band edge breaking into wisps: |vb| perturbed by elongated noise
    avb = np.abs(vb)
    vw = avb * (1 + 0.55 * (wn1 - 0.5)) - 0.45 * (wn2 - 0.5) * C.smoothstep(0.4, 1.2, avb)
    edge = C.smoothstep(1.45, 0.55, vw)
    wisps = C.smoothstep(0.55, 0.8, wn2) * C.smoothstep(2.3, 1.0, avb) * (1 - edge)
    clouds = np.clip(0.45 * wn1 + 0.35 * wn2 + 0.35 * fine - 0.2, 0, 1) ** 1.4
    gr = rng.random((Wd, Lp)).astype(np.float32)
    gr = cv2.GaussianBlur(gr, (0, 0), 0.8 * s + 0.3)
    gr = np.clip((gr - gr.mean()) / (gr.std() + 1e-6), -3, 3)
    core = np.exp(-((uu - core_u) / 0.13) ** 2) * np.exp(-(vb / 0.75) ** 2)
    core_hot = np.exp(-((uu - core_u) / 0.06) ** 2) * np.exp(-(vb / 0.35) ** 2)
    lum = (edge * (0.3 + 0.75 * clouds) + wisps * 0.25 * (0.4 + clouds)) * (1 + 0.12 * gr)
    lum = lum + core * (0.75 + 0.7 * clouds) + core_hot * 0.45
    along = 0.7 + 0.3 * (P.fbm1d(uu[0], 3.0, 3, seed + 41)[None, :] * 0.5 + 0.5)
    lum *= along

    # ---- dark rift lanes
    dust = np.zeros_like(vb)
    lanes = [  # (u0, u1, centre, half-width, opacity, wiggle, crisp side +1/-1, seed)
        (0.03, 0.92, 0.1, 0.13, 1.0, 0.22, 1, 1),
        (0.2, 0.98, -0.5, 0.08, 0.9, 0.25, -1, 2),
        (0.05, 0.45, -0.3, 0.06, 0.85, 0.2, 1, 3),
        (0.45, 0.88, 0.6, 0.05, 0.75, 0.18, -1, 4),
        (0.12, 0.5, 0.52, 0.04, 0.7, 0.25, 1, 5),
    ]
    fil = stretched_fbm(Wd, Lp, 30, 50, 4, seed + 70, stretch=2.2)
    fib = stretched_fbm(Wd, Lp, 8, 40, 4, seed + 71, stretch=5.0)          # dark fibres along the band
    for (a, b, cc, hw, opa, wg, side, sd) in lanes:
        q = np.clip((uu - a) / (b - a), 0, 1)
        taper = np.clip(np.sin(np.pi * q), 0, 1) ** 0.6 * ((uu >= a) & (uu <= b))
        cl = cc + wg * P.fbm1d(uu[0], 5.0, 4, seed + 100 + sd)[None, :]
        wdt = hw * (0.5 + 1.0 * (P.fbm1d(uu[0], 8.0, 3, seed + 200 + sd)[None, :] * 0.5 + 0.5))
        dd = (vb - cl) / np.maximum(wdt, 1e-3)
        dd = dd + 0.4 * (fil - 0.5)
        crisp = C.smoothstep(1.0, 0.8, dd * side)
        soft = C.smoothstep(1.8, 0.4, -dd * side)
        prof = np.where(dd * side > 0, crisp, soft)
        brk = C.smoothstep(-0.05, 0.3, P.fbm1d(uu[0], 14.0, 3, seed + 300 + sd))[None, :]
        clump = C.smoothstep(0.35, 0.6, fil + 0.25 * brk)                 # lanes made of dust clouds
        dust = np.maximum(dust, prof * taper * opa * brk * (0.35 + 0.65 * clump))
        br = C.smoothstep(0.66, 0.74, fib) * np.exp(-np.abs(dd) / 2.5) * taper * opa * 0.7
        dust = np.maximum(dust, br)
    for _ in range(0):
        cu_ = rng.uniform(0.1, 0.9)
        cv_ = rng.normal(0, 0.5)
        r_ = rng.uniform(0.003, 0.007)
        g_ = np.exp(-(((uu - cu_) / r_) ** 2 + ((vb - cv_) / (r_ * L / width * 0.6)) ** 2))
        dust = np.maximum(dust, g_ * rng.uniform(0.5, 0.85))
    dust = np.clip(dust * 1.25, 0, 0.97)
    dust = 0.8 * dust + 0.2 * cv2.GaussianBlur(dust, (0, 0), 2.5 * s + 0.4)
    dust *= env * C.smoothstep(1.7, 0.7, avb)          # dust only inside the band

    glow = lum * (1 - dust) * env
    # ---- bright knots
    knc = np.zeros(vb.shape + (3,), np.float32)
    for i in range(16):
        cu_ = rng.uniform(0.12, 0.85)
        cv_ = rng.normal(0, 0.45)
        r_ = rng.uniform(0.003, 0.009)
        g_ = np.exp(-(((uu - cu_) / r_) ** 2 + ((vb - cv_) / (r_ * L / width)) ** 2)) * rng.uniform(0.4, 1.0)
        c_ = np.array([1.0, 0.55, 0.7] if rng.random() < 0.35 else [0.9, 0.95, 1.0], np.float32)
        knc += g_[..., None] * c_
    # ---- colour
    warm = np.clip(core * 1.3 + core_hot, 0, 1)[..., None]
    body = np.array([0.78, 0.85, 1.0], np.float32)
    peach = np.array([1.0, 0.78, 0.55], np.float32)
    col = body * (1 - warm) + peach * warm
    e3 = env[..., None]
    img = glow[..., None] * col * 1.0
    img += core_hot[..., None] * (1 - dust[..., None]) * np.array([1.0, 0.9, 0.75], np.float32) * 0.18 * e3
    img += knc * (1 - dust[..., None]) * 0.35 * e3
    halo = np.exp(-(vb / 2.0) ** 2) * env
    img += halo[..., None] * np.array([0.05, 0.05, 0.1], np.float32)
    rim = np.clip(cv2.GaussianBlur(dust, (0, 0), 2.0 * s + 0.5) - dust, -1, 0) * -1
    img += (rim * lum * env)[..., None] * np.array([1.0, 0.75, 0.55], np.float32) * 0.5
    img *= strength
    dens = np.clip((edge * (0.25 + 0.9 * clouds) + wisps * 0.3 + core * 0.6) * (1 - dust) ** 1.5 * env, 0, 1)

    xs, ys = C.grid(W, H)
    u = ((xs - p0[0]) * ux + (ys - p0[1]) * uy)
    v = (-(xs - p0[0]) * uy + (ys - p0[1]) * ux)
    v = v - bend * L * (((u / L) - 0.5) ** 2 * 4 - 1)
    mx = u.astype(np.float32)
    my = (v + half).astype(np.float32)

    def rm(a):
        return cv2.remap(a.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    return rm(img), rm(dust), rm(dens)
