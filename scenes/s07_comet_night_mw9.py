"""Round-11 Milky Way for s07_comet_night.

Built directly in pixel space with ISOTROPIC noise (no noise stretched along the band -> no parallel scratch
strokes):
  * a broad, soft glow band whose width breathes, with mottled star clouds (multi-scale isotropic fbm masses)
    and a bulging warm-cream core that fades through pale violet to cool blue at the edges,
  * the Great Rift: one dark lane that meanders along the band and SPLITS into two branches, plus a short
    third lane on the other flank, each with organic ragged edges (domain-warped, fine-noise eroded), inner
    mottling and a faint warm rim where the glow wraps the dust,
  * scattered dark globules / wisps inside the band,
  * a density map (for the point-star layers) that concentrates stars toward the core,
  * a soft brightness cap so the horizon end never clips to white.
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _fbm(W, H, px_scale, octaves, seed):
    """Isotropic fbm whose base feature size is ~px_scale pixels."""
    return C.fbm(W, H, max(W / px_scale, 1.0), octaves, seed=seed).astype(np.float32)


def milky_way9(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0, s=1.0, core_u=0.35, cap=0.42):
    """Returns (glow RGB additive, dust (H,W), density (H,W)). p0 = horizon end, p1 = far end."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    xs, ys = C.grid(W, H)
    u = ((xs - p0[0]) * ux + (ys - p0[1]) * uy) / L
    v = (-(xs - p0[0]) * uy + (ys - p0[1]) * ux)
    v = v - bend * L * ((u - 0.5) ** 2 * 4 - 1)

    def n1(q, f, sd):
        return P.fbm1d(np.clip(q, -0.5, 1.5).ravel(), f, 3, sd).reshape(q.shape).astype(np.float32)

    # breathing width, wider at the core bulge
    wv = width * (1.0 + 0.45 * np.exp(-((u - core_u) / 0.16) ** 2) + 0.2 * np.clip(0.2 - u, 0, 1) * 3)
    wv = wv * (1.0 + 0.18 * n1(u, 3.0, seed + 1))
    # organic warp (isotropic, pixel space)
    wa = _fbm(W, H, 0.9 * width, 4, seed + 2) - 0.5
    wb = _fbm(W, H, 0.9 * width, 4, seed + 3) - 0.5
    vw = v + wa * 0.5 * width
    uw = u + wb * 0.5 * width / L
    vb = vw / wv
    env = _ss(-0.04, 0.12, u) * (1 - 0.55 * _ss(0.65, 1.05, u))

    # ---- glow + star clouds
    prof = np.exp(-(vb / 1.05) ** 2)
    wide = np.exp(-(vb / 2.2) ** 2)
    c1 = _fbm(W, H, 0.7 * width, 5, seed + 10)
    c2 = _fbm(W, H, 0.25 * width, 4, seed + 11)
    c3 = _fbm(W, H, 0.08 * width, 3, seed + 12)
    clump = np.clip(0.55 * c1 + 0.3 * c2 + 0.15 * c3, 0, 1)
    clump = _ss(0.36, 0.72, clump)
    core = np.exp(-((uw - core_u) / 0.2) ** 2) * np.exp(-(vb / 0.9) ** 2)
    core_hot = np.exp(-((uw - core_u) / 0.07) ** 2) * np.exp(-(vb / 0.38) ** 2)
    rng = np.random.default_rng(seed)
    gr = rng.random((H, W)).astype(np.float32)
    gr = cv2.GaussianBlur(gr, (0, 0), 0.7 * s + 0.3)
    gr = (gr - gr.mean()) / (gr.std() + 1e-6)
    lum = prof * (0.25 + 0.85 * clump) * (1 + 0.1 * np.clip(gr, -2.5, 2.5)) + wide * 0.14
    lum = lum + core * (0.35 + 0.35 * clump) + core_hot * 0.25

    # ---- dust lanes (Great Rift splitting into two branches + a short flank lane)
    fe = _fbm(W, H, 0.12 * width, 4, seed + 20)          # fine erosion of the lane edges
    fm = _fbm(W, H, 0.3 * width, 3, seed + 21)

    def lane(c, hw, u0, u1_, sd, opa=0.92):
        taper = _ss(u0, u0 + 0.12, uw) * (1 - _ss(u1_ - 0.18, u1_, uw))
        h_ = hw * width * (0.45 + 0.55 * taper) * (0.7 + 0.6 * (n1(uw, 7.0, sd) * 0.5 + 0.5))
        d = np.abs(vw - c) / h_
        d = d + 0.6 * (fe - 0.5) + 0.45 * (fm - 0.5)
        o = _ss(1.1, 0.35, d) * taper
        return o * opa * (0.72 + 0.28 * _ss(0.3, 0.7, fm))

    c_main = width * (0.08 + 0.28 * n1(uw, 2.2, seed + 30))
    split = _ss(0.42, 0.98, uw) ** 1.2
    c_a = c_main + width * 0.55 * split
    c_b = c_main - width * 0.45 * split
    dust = lane(c_main, 0.2, 0.06, 0.55, seed + 31)
    dust = np.maximum(dust, lane(c_a, 0.15, 0.4, 1.0, seed + 32))
    dust = np.maximum(dust, lane(c_b, 0.11, 0.44, 0.92, seed + 33, opa=0.8))
    c_f = -width * (0.62 + 0.15 * n1(uw, 3.0, seed + 34))
    dust = np.maximum(dust, lane(c_f, 0.09, 0.15, 0.5, seed + 35, opa=0.7))
    # small branch twigs peeling off the main lane
    for k, (ua, ub, side) in enumerate(((0.22, 0.4, 1), (0.3, 0.5, -1), (0.62, 0.8, 1))):
        q = np.clip((uw - ua) / (ub - ua), 0, 1)
        ct = c_main + side * width * 0.5 * q ** 1.3
        dust = np.maximum(dust, lane(ct, 0.08, ua, ub, seed + 40 + k, opa=0.6))
    # scattered dark globules / wisps in the band
    gl = _fbm(W, H, 0.18 * width, 4, seed + 50)
    dust = np.maximum(dust, _ss(0.66, 0.76, gl) * 0.55 * prof)
    dust = np.clip(dust, 0, 0.94) * env * _ss(1.9, 0.9, np.abs(vb))

    glow = lum * (1 - dust) * env
    # ---- colour: warm cream core -> pale violet -> cool blue edges
    warm = np.clip(core * 1.4 + core_hot, 0, 1)[..., None]
    cool = np.array([0.38, 0.56, 1.0], np.float32)
    vio = np.array([0.86, 0.66, 1.0], np.float32)
    cream = np.array([1.0, 0.84, 0.62], np.float32)
    pv = prof[..., None]
    col = cool * (1 - pv) + vio * pv
    col = col * (1 - warm) + cream * warm
    img = glow[..., None] * col
    rim = np.clip(cv2.GaussianBlur(dust, (0, 0), 2.5 * s + 0.5) - dust, 0, 1)
    img += (rim * lum * env)[..., None] * np.array([1.0, 0.75, 0.55], np.float32) * 0.5
    # pink HII knots
    for i in range(18):
        cu_ = rng.uniform(0.12, 0.85)
        cv_ = rng.normal(0, 0.5) * width
        r_ = rng.uniform(3, 9) * s
        cx_ = p0[0] + ux * cu_ * L - uy * cv_
        cy_ = p0[1] + uy * cu_ * L + ux * cv_
        C.splat(img, cx_, cy_, r_, np.array([1.0, 0.45, 0.65], np.float32), rng.uniform(0.04, 0.1))
    img *= strength
    # soft cap: never clips (the core near the horizon stays a cream glow)
    m = img.max(-1, keepdims=True) + 1e-6
    img = img * (cap * np.tanh(m / cap) / m)
    dens = np.clip((prof * (0.25 + 0.9 * clump) + core * 0.8) * (1 - dust) ** 1.5 * env, 0, 1)
    return img.astype(np.float32), dust.astype(np.float32), dens.astype(np.float32)
