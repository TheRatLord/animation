"""Round-9 Milky Way for s07_comet_night: a structured band, not a streak texture.

  * clumped star-cloud masses (round-ish, multi-scale) with brighter knots strung along the core,
  * dark rifts that BRANCH and NARROW (a trunk rift splitting into two branches, side rifts peeling off
    at an angle, every lane tapering to a point), crisp on one side and feathered on the other,
  * a warm peach/gold core against cool blue-white edges,
  * the band edge breaking into clumps instead of a smooth gaussian or parallel smeared strokes.
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P
from s07_comet_night_mw6 import stretched_fbm


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def milky_way7(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0, s=1.0, core_u=0.3):
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
    vb = (vv - wob) / wu
    env = _ss(0.0, 0.22, uu) * (1 - 0.4 * _ss(0.7, 1.0, uu))
    aniso = L / width                    # a round feature of radius r (u units) is r * aniso in vb units

    # ---- clumpy star clouds (nearly isotropic noise at several scales)
    c1 = stretched_fbm(Wd, Lp, 5, 9, 4, seed + 10, stretch=1.3, gain=0.6)
    c2 = stretched_fbm(Wd, Lp, 12, 20, 4, seed + 11, stretch=1.15, gain=0.6)
    c3 = stretched_fbm(Wd, Lp, 34, 55, 3, seed + 12, stretch=1.0)
    avb = np.abs(vb)
    vw = avb * (1 + 0.5 * (c1 - 0.5)) - 0.5 * (c2 - 0.5) * _ss(0.3, 1.2, avb)
    edge = _ss(1.5, 0.45, vw)
    clump = np.clip(0.5 * c1 + 0.35 * c2 + 0.25 * c3 - 0.22, 0, 1) ** 1.6
    clump = _ss(0.05, 0.55, clump)                          # masses with defined edges
    gr = rng.random((Wd, Lp)).astype(np.float32)
    gr = cv2.GaussianBlur(gr, (0, 0), 0.8 * s + 0.3)
    gr = np.clip((gr - gr.mean()) / (gr.std() + 1e-6), -3, 3)
    core = np.exp(-((uu - core_u) / 0.13) ** 2) * np.exp(-(vb / 0.75) ** 2)
    core_hot = np.exp(-((uu - core_u) / 0.06) ** 2) * np.exp(-(vb / 0.35) ** 2)
    spine = np.exp(-(vb / 0.5) ** 2)
    lum = edge * (0.18 + 0.95 * clump * (0.5 + 0.5 * spine)) * (1 + 0.14 * gr)
    # bright star-cloud knots strung along the core
    knots = np.zeros_like(vb)
    for i in range(46):
        cu_ = rng.uniform(0.08, 0.9)
        cv_ = rng.normal(0, 0.35)
        r_ = rng.uniform(0.004, 0.016) * (1.5 if abs(cu_ - core_u) < 0.12 else 1.0)
        g_ = np.exp(-(((uu - cu_) / r_) ** 2 + ((vb - cv_) / (r_ * aniso)) ** 2))
        knots += g_ * rng.uniform(0.25, 0.8)
    knots = knots * (0.6 + 0.8 * c3)
    lum = lum + knots * edge * 0.7 + core * (0.5 + 0.5 * clump) + core_hot * 0.2

    # ---- branching, narrowing rifts
    fil = stretched_fbm(Wd, Lp, 30, 50, 4, seed + 70, stretch=1.4)

    def lane(ua, ub, c0, slope, hw, opa, side, sd, curve=0.0):
        q = np.clip((uu - ua) / (ub - ua), 0, 1)
        inside = (uu >= ua) & (uu <= ub)
        taper = np.clip(q / 0.12, 0, 1) * (1 - q) ** 0.8                     # blunt start, narrows out
        cl = (c0 + slope * (uu - ua) * 2.0 + curve * q * q +
              0.12 * P.fbm1d(uu[0], 6.0, 3, seed + 100 + sd)[None, :])
        wdt = hw * taper * (0.7 + 0.6 * (P.fbm1d(uu[0], 9.0, 2, seed + 200 + sd)[None, :] * 0.5 + 0.5))
        dd = (vb - cl) / np.maximum(wdt, 1e-3)
        dd = dd + 0.75 * (fil - 0.5)
        crisp = _ss(1.0, 0.75, dd * side)
        soft = _ss(1.9, 0.3, -dd * side)
        prof = np.where(dd * side > 0, crisp, soft)
        brk = _ss(-0.2, 0.25, P.fbm1d(uu[0], 11.0, 3, seed + 300 + sd))[None, :]
        return prof * inside * opa * (0.35 + 0.65 * brk) * (wdt > 0.004)

    dust = np.zeros_like(vb)
    # the great rift: one trunk that splits into two branches
    dust = np.maximum(dust, lane(0.1, 0.55, 0.05, 0.0, 0.34, 1.0, 1, 1))
    dust = np.maximum(dust, lane(0.4, 0.97, 0.1, 0.5, 0.24, 0.95, 1, 2, curve=0.1))
    dust = np.maximum(dust, lane(0.42, 0.92, -0.02, -0.8, 0.18, 0.9, -1, 3, curve=-0.15))
    # side rifts peeling off at an angle, narrowing as they leave the band
    for k, (ua, ub, c0, sl, hw, op, sd_) in enumerate(((0.12, 0.38, -0.25, -1.4, 0.12, 0.85, 4),
                                                       (0.2, 0.42, 0.35, 1.2, 0.1, 0.8, 5),
                                                       (0.55, 0.78, -0.45, -1.1, 0.09, 0.75, 6),
                                                       (0.62, 0.85, 0.5, 1.5, 0.08, 0.7, 7),
                                                       (0.28, 0.5, -0.05, -2.0, 0.07, 0.65, 8))):
        dust = np.maximum(dust, lane(ua, ub, c0, sl, hw, op, 1 if k % 2 else -1, sd_))
    # small dark globules
    for _ in range(0):
        cu_ = rng.uniform(0.1, 0.9)
        cv_ = rng.normal(0, 0.5)
        r_ = rng.uniform(0.002, 0.005)
        g_ = np.exp(-(((uu - cu_) / r_) ** 2 + ((vb - cv_) / (r_ * aniso)) ** 2))
        dust = np.maximum(dust, _ss(0.3, 0.6, g_) * rng.uniform(0.5, 0.85))
    dust = np.clip(dust * 1.15, 0, 0.95)
    dust = 0.8 * dust + 0.2 * cv2.GaussianBlur(dust, (0, 0), 2.0 * s + 0.4)
    dust *= env * _ss(1.25, 0.6, vw) * _ss(0.2, 0.42, uu)

    glow = lum * (1 - dust) * env
    # ---- colour: warm peach/gold core, cool blue-white edges
    warm = np.clip(core * 1.3 + core_hot + 0.45 * spine * clump, 0, 1)[..., None]
    cool = np.array([0.66, 0.78, 1.0], np.float32)
    peach = np.array([1.0, 0.8, 0.58], np.float32)
    col = cool * (1 - warm) + peach * warm
    e3 = env[..., None]
    img = glow[..., None] * col
    img += core_hot[..., None] * (1 - dust[..., None]) * np.array([1.0, 0.88, 0.7], np.float32) * 0.2 * e3
    halo = np.exp(-(vb / 2.0) ** 2) * env
    img += halo[..., None] * np.array([0.04, 0.05, 0.1], np.float32)
    # warm-lit rims along the rift edges (the glow wraps the dust)
    rim = np.clip(cv2.GaussianBlur(dust, (0, 0), 2.0 * s + 0.5) - dust, -1, 0) * -1
    img += (rim * lum * env)[..., None] * np.array([1.0, 0.72, 0.5], np.float32) * 0.6
    img *= strength
    dens = np.clip((edge * (0.2 + 0.9 * clump) + knots * 0.5 + core * 0.6) * (1 - dust) ** 1.5 * env, 0, 1)

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
