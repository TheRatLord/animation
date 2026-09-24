"""Round-7 painting helpers for s07_comet_night (Your Name comet night).

- comet_tail5: the Your-Name dust tail -- broad, curved, with a hard cyan-white core line along its
  leading edge and a spectral spread toward the diffuse side (cyan -> teal -> faint gold -> magenta ->
  violet haze), feathered lengthwise brush striae and glittering particles; plus a separate thin straight
  blue ion tail diverging from the nucleus.
- twilight: a warm kataware-doki glow along the horizon (gold -> rose -> lilac) under a navy sky.
"""
import math
import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def _g(x, c, w):
    return np.exp(-((x - c) / w) ** 2)


def _streak_tex(u, vn, seed, fu=3.0, fv=38.0):
    """Lengthwise brush striae: noise that is long along u and fine across vn (0..1)."""
    rng = np.random.default_rng(seed)
    nv, nu = 512, 64
    t = rng.random((nv, nu)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), sigmaX=1.2, sigmaY=0.7)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    mx = np.mod(u * fu * nu / 8.0, nu).astype(np.float32)
    my = np.mod((vn * fv + 256.0), nv).astype(np.float32)
    return cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def comet_tail5(Wp, Hp, head, d, L, W, s, bend=0.16, w0=0.0035, w1=0.085, seed=3, strength=1.0):
    """Returns (rgb additive, u, vn, body) for the curved dust tail."""
    xs, ys = C.grid(Wp, Hp)
    dx, dy = d
    px, py = xs - head[0], ys - head[1]
    u = (px * dx + py * dy) / L
    across = -px * dy + py * dx
    across = across - bend * L * np.clip(u, 0, None) ** 2
    uc = np.clip(u, 0, 1)
    wu = (w0 + (w1 - w0) * uc ** 0.8) * W
    vn = across / wu
    # along-tail falloff
    fall = C.smoothstep(-0.003, 0.012, u) * np.clip(1 - u, 0, 1) ** 1.2 * (0.35 + 0.65 * np.exp(-uc / 0.35))
    spec = C.smoothstep(0.02, 0.22, uc)                   # spectral bands open out away from the head
    st = _streak_tex(u, vn, seed)
    st2 = _streak_tex(u * 0.7 + 3.1, vn * 0.55, seed + 1, fv=22.0)
    stria = 0.35 + 1.3 * (0.6 * st + 0.4 * st2)           # 0.35..1.65
    stria = 1 + (stria - 1) * spec
    # hard leading edge (vn < -0.62) and ragged diffuse trailing side
    lead = C.smoothstep(-0.9, -0.62, vn + 0.05 * (st - 0.5))
    hair = np.clip(st * 1.4 - 0.2, 0, 1) ** 2
    trail = np.exp(-np.clip(vn - 0.55, 0, None) ** 2 / (0.22 + 1.1 * hair * spec) ** 2)
    env = lead * trail * fall
    core = _g(vn, -0.56, 0.06 + 0.05 * spec) * (1.0 + 0.6 * np.exp(-uc / 0.1))
    bands = [  # centre, width, colour, gain
        (-0.38, 0.2, (0.2, 0.85, 1.0), 1.1),
        (-0.1, 0.16, (0.15, 1.0, 0.7), 0.55),
        (0.12, 0.09, (1.0, 0.78, 0.32), 0.42),
        (0.34, 0.18, (1.0, 0.25, 0.72), 0.85),
        (0.66, 0.34, (0.45, 0.3, 1.0), 0.4),
        (-0.78, 0.05, (0.9, 0.35, 1.0), 0.3),
    ]
    img = np.zeros(xs.shape + (3,), np.float32)
    for (c, w, col, g) in bands:
        k = _g(vn, c, w) * g
        img += k[..., None] * np.asarray(col, np.float32)
    # near the nucleus everything collapses into one cyan-white jet
    near = 1 - spec
    jet = _g(vn, -0.3, 0.45) * near
    img = img * spec[..., None] + jet[..., None] * np.array([0.55, 0.9, 1.0], np.float32)
    img = img * (env * stria)[..., None]
    img += (core * fall * lead)[..., None] * np.array([0.82, 0.97, 1.0], np.float32) * 1.1
    # soft overall veil (the tail glows into the sky)
    veil = np.exp(-(np.clip(vn, -3, 3) / 1.2) ** 2) * fall
    img += veil[..., None] * np.array([0.05, 0.12, 0.22], np.float32)
    img *= strength
    body = (np.exp(-(vn + 0.2) ** 2 / 0.4) * fall).astype(np.float32)
    return img.astype(np.float32), u.astype(np.float32), vn.astype(np.float32), body


def tail_particles(Wp, Hp, head, d, L, W, s, bend, w0, w1, n, seed):
    """Glittering dust grains scattered through the spectral side of the tail (additive)."""
    rng = np.random.default_rng(seed)
    uq = rng.uniform(0.04, 0.75, n) ** 1.1
    vq = rng.normal(0.35, 0.45, n)
    wu = (w0 + (w1 - w0) * uq ** 0.8) * W
    acr = vq * wu + bend * L * uq ** 2
    x = head[0] + d[0] * uq * L - d[1] * acr
    y = head[1] + d[1] * uq * L + d[0] * acr
    cols = np.array([[0.7, 1.0, 1.0], [1.0, 0.55, 0.95], [0.8, 0.7, 1.0], [1.0, 1.0, 1.0], [0.5, 0.9, 1.0]],
                    np.float32)
    ci = rng.integers(0, len(cols), n)
    m = (rng.random(n) ** 5 * 1.6 + 0.08) * np.clip(1 - uq, 0.1, 1) ** 0.8
    acc = np.zeros((Hp, Wp, 3), np.float32)
    ok = (x >= 0) & (x < Wp) & (y >= 0) & (y < Hp)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    for c in range(3):
        np.add.at(acc[..., c], (yi, xi), m[ok] * cols[ci[ok], c])
    return C.blur(acc, 0.5 * s + 0.2) * 3.2


def ion_tail(Wp, Hp, head, d, L, W, s, seed=7):
    """Thin straight blue ion tail with a bright root, faint streamers, fading out."""
    xs, ys = C.grid(Wp, Hp)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    a = -px * d[1] + py * d[0]
    uc = np.clip(u, 0, 1)
    w = (0.0009 + 0.004 * uc) * W
    fall = C.smoothstep(-0.002, 0.01, u) * np.clip(1 - u, 0, 1) ** 2.2 * (0.3 + 0.7 * np.exp(-uc / 0.18))
    core = np.exp(-(a / w) ** 2)
    glow = np.exp(-(a / (w * 4.5)) ** 2) * 0.3
    st = _streak_tex(u, a / w, seed, fu=5.0, fv=10.0)
    k = (core * (0.7 + 0.5 * st) + glow) * fall
    img = k[..., None] * np.array([0.35, 0.58, 1.0], np.float32)
    img += (core * fall * np.exp(-uc / 0.06))[..., None] * np.array([0.6, 0.85, 1.0], np.float32)
    return img.astype(np.float32)


def twilight(Wp, Hp, y_h, H, xs, ys, x_warm, seed=0):
    """Additive warm horizon glow: gold-peach band hugging the horizon, rose above, lilac veil.
    Rises high enough (~0.25 H) to read above the mountain silhouettes."""
    h = np.clip(y_h - ys, 0, None)
    gx = np.exp(-((xs - x_warm) / (0.34 * Wp)) ** 2)
    gxb = 0.35 + 0.65 * gx
    n = P.fbm_lowres(Wp, Hp, 3, 3, seed, q=8)
    # thin warm cloud streaks lying in the glow (stratus bars, lit from below)
    bars = P.rot_fbm(Wp, Hp, 0.0, 12.0, 45.0, 4, seed + 5, warp_amt=0.03)
    bars = C.smoothstep(0.58, 0.72, bars) * np.exp(-((h - 0.1 * H) / (0.05 * H)) ** 2) * gx
    gold = np.exp(-h / (0.085 * H)) * gxb * gx
    rose = np.exp(-h / (0.17 * H)) * gxb * (0.8 + 0.4 * n)
    lil = np.exp(-h / (0.34 * H)) * (0.6 + 0.4 * gx)
    below = (ys <= y_h + 2)
    out = (gold[..., None] * np.array([0.95, 0.46, 0.1], np.float32) +
           rose[..., None] * np.array([0.26, 0.08, 0.1], np.float32) +
           lil[..., None] * np.array([0.07, 0.0, 0.08], np.float32) +
           bars[..., None] * np.array([0.3, 0.12, 0.12], np.float32)) * below[..., None]
    mix = np.clip(gold * 1.1 + rose * 0.25, 0, 0.85) * below
    return out.astype(np.float32), mix.astype(np.float32)
