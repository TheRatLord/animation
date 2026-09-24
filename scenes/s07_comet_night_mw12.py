"""Round-12 Milky Way (broad soft violet-brown dust clouds, star density rising along the lanes; from
round-11 for s07_comet_night.

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


def milky_way12(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0, s=1.0, core_u=0.35, cap=0.42):
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

    # ---- dust lanes: BROAD, soft, feathered dust clouds (not crack lines). Each lane is a wide gaussian-ish
    # band whose edge is feathered by medium-scale mottling; inside it the opacity is patchy (dust clouds
    # with thinner gaps), so it reads as violet-brown smoke lying over the star clouds.
    fm = _fbm(W, H, 0.45 * width, 4, seed + 21)            # medium mottling (cloud-scale)
    fe = _fbm(W, H, 0.16 * width, 3, seed + 20)            # fine feathering

    def lane(c, hw, u0, u1_, sd, opa=1.0):
        taper = _ss(u0, u0 + 0.16, uw) * (1 - _ss(u1_ - 0.22, u1_, uw))
        h_ = hw * width * (0.55 + 0.45 * taper) * (0.75 + 0.5 * (n1(uw, 4.0, sd) * 0.5 + 0.5))
        d = np.abs(vw - c) / h_
        d = d + 0.55 * (fm - 0.5) + 0.18 * (fe - 0.5)
        o = _ss(1.25, 0.25, d) * taper
        return o * opa * (0.6 + 0.4 * _ss(0.3, 0.7, fm))

    c_main = width * (0.05 + 0.25 * n1(uw, 1.6, seed + 30))
    split = _ss(0.42, 0.98, uw) ** 1.2
    c_a = c_main + width * 0.6 * split
    c_b = c_main - width * 0.5 * split
    dust = lane(c_main, 0.42, 0.04, 0.58, seed + 31)
    dust = 1 - (1 - dust) * (1 - lane(c_a, 0.36, 0.38, 1.0, seed + 32))
    dust = 1 - (1 - dust) * (1 - lane(c_b, 0.26, 0.44, 0.94, seed + 33, opa=0.6))
    c_f = -width * (0.7 + 0.15 * n1(uw, 2.0, seed + 34))
    dust = 1 - (1 - dust) * (1 - lane(c_f, 0.22, 0.12, 0.52, seed + 35, opa=0.45))
    # a few soft dark dust wisps / globules in the band
    gl = _fbm(W, H, 0.3 * width, 4, seed + 50)
    dust = 1 - (1 - dust) * (1 - _ss(0.6, 0.78, gl) * 0.35 * prof)
    dust = cv2.GaussianBlur(dust, (0, 0), 1.2 * s + 0.4)
    dust = np.clip(dust * 1.1, 0, 0.92) * env * _ss(2.2, 1.0, np.abs(vb))

    glow = lum * (1 - 0.92 * dust) * env
    # ---- colour: warm cream core -> pale violet -> cool blue edges
    warm = np.clip(core * 1.4 + core_hot, 0, 1)[..., None]
    cool = np.array([0.38, 0.56, 1.0], np.float32)
    vio = np.array([0.86, 0.66, 1.0], np.float32)
    cream = np.array([1.0, 0.84, 0.62], np.float32)
    pv = prof[..., None]
    col = cool * (1 - pv) + vio * pv
    col = col * (1 - warm) + cream * warm
    img = glow[..., None] * col
    # the dust itself is faintly lit: violet-brown smoke (thicker = browner), not a black void
    # round 13: warm-violet / brown tint VARIATION inside the dust clouds (not a uniform grey)
    tv = _ss(0.3, 0.7, _fbm(W, H, 0.6 * width, 3, seed + 60))[..., None]
    tv2 = _ss(0.35, 0.75, _fbm(W, H, 0.25 * width, 3, seed + 61))[..., None]
    c_vio = np.array([0.44, 0.24, 0.46], np.float32)
    c_brn = np.array([0.42, 0.24, 0.13], np.float32)
    c_mag = np.array([0.5, 0.22, 0.34], np.float32)
    dcol = c_vio * (1 - tv) + c_brn * tv
    dcol = dcol * (1 - 0.35 * tv2) + c_mag * 0.35 * tv2
    img += (dust * (lum * 0.55 + 0.07) * env)[..., None] * dcol * 0.6
    # warm glow where the star clouds wrap the dust edges
    rim = np.clip(cv2.GaussianBlur(dust, (0, 0), 0.12 * width) - dust, 0, 1)
    img += (rim * lum * env)[..., None] * np.array([1.0, 0.78, 0.6], np.float32) * 0.35
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
    edge = np.clip(cv2.GaussianBlur(dust, (0, 0), 0.1 * width) * 1.6, 0, 1) * (1 - _ss(0.25, 0.6, dust))
    dens = np.clip((prof * (0.25 + 0.9 * clump) + core * 0.8 + 1.8 * edge * (0.4 + prof) + 0.55 * np.exp(-((np.abs(vb) - 1.1) / 0.45) ** 2) * (0.5 + clump)) * (1 - 0.75 * dust) * env, 0, 1)
    return img.astype(np.float32), dust.astype(np.float32), dens.astype(np.float32)
