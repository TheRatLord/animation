"""Round-11 comet for s07_comet_night (Your Name yn_05).

comet_tail9: the comet8 hard white core + striated cyan dust fan (convex side feathering out in brushed
hair-streaks), with the concave side re-painted as the yn_05 spectral ribbons running parallel to the core:
thin gold/green fringe -> broad saturated magenta/pink band (~half the cyan body's width) -> dark blue gap
with sparkle -> a fainter violet/magenta second ribbon diverging outward, then feathered striae bleeding out.
ion_tail9: thin hard blue ion tail that curves away from the dust tail.
star_flare: 6-point star with halation for the nucleus.
"""
import math

import numpy as np
import cv2

from lib import core as C
from s07_comet_night_comet8 import _ss, _striae8, comet_head8  # noqa: F401


def comet_tail9(Wp, Hp, head, d, L, W, s, bend=0.2, w0=0.003, w1=0.13, wexp=0.85, seed=3, n_dust=3000,
                sec=1.0):
    """Returns (rgb additive, U, VN, BODY) full plate."""
    pts = []
    for uq in np.linspace(0, 1.05, 16):
        wq = (w0 + (w1 - w0) * uq ** wexp) * W
        for v in (-2.3, 3.4):
            acr = v * wq + bend * L * uq * uq
            pts.append((head[0] + d[0] * uq * L - d[1] * acr, head[1] + d[1] * uq * L + d[0] * acr))
    pts = np.array(pts)
    bx0, by0 = np.maximum(pts.min(0) - 0.02 * W, 0).astype(int)
    bx1, by1 = pts.max(0) + 0.02 * W
    bx1, by1 = int(min(bx1, Wp)), int(min(by1, Hp))
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    acr = -px * d[1] + py * d[0] - bend * L * np.clip(u, 0, None) ** 2
    uc = np.clip(u, 0, 1.2)
    wu = (w0 + (w1 - w0) * uc ** wexp) * W
    vn = acr / wu                                         # +0.5 core line, ~-1 convex edge
    start = _ss(-0.002, 0.012, u)
    fall = start * np.clip(1.03 - u, 0, 1) ** 1.2 * (0.35 + 0.65 * np.exp(-uc / 0.3))

    st_f = _striae8(u, vn, seed, fu=90.0, fv=70.0, blur_u=5.0, blur_v=0.6)
    st_m = _striae8(u, vn, seed + 1, fu=40.0, fv=22.0, blur_u=4.0, blur_v=0.8)
    st = 0.55 * st_f + 0.45 * st_m
    img = np.zeros(u.shape + (3,), np.float32)

    # ---- hard core line on the concave edge
    cw = 0.035 + 0.025 * np.clip(uc / 0.5, 0, 1)
    cpos = 0.5 + 0.03 * (st_m - 0.5)
    core = _ss(cw * 1.0, cw * 0.35, np.abs(vn - cpos))
    core_env = start * (np.exp(-uc / 0.45) * 0.9 + 0.35 * np.exp(-uc / 0.05)) * (0.8 + 0.4 * st_m)
    cc = np.clip(uc / 0.6, 0, 1)[..., None]
    core_col = np.array([1.0, 1.0, 1.0], np.float32) * (1 - cc) + np.array([0.55, 0.95, 1.0], np.float32) * cc
    img += (core * core_env)[..., None] * core_col * 2.2
    cg = np.exp(-((vn - cpos) / (cw * 2.5)) ** 2) * core_env
    img += cg[..., None] * np.array([0.35, 0.8, 1.0], np.float32) * 0.5

    # ---- dust body: cyan next to the core -> blue at the convex edge; strongly striated
    q = np.clip((cpos - vn) / 1.5, 0, 1)
    lead = _ss(cpos + 0.02, cpos - 0.04, vn)
    edge_c = -1.0 - 0.35 * (st - 0.5) - 0.25 * (st_m - 0.5)
    outer = _ss(edge_c - 0.18, edge_c + 0.25, vn)
    stria = 0.7 + 0.35 * st_f ** 1.4 + 0.45 * st_m
    body = lead * outer * (1.0 - 0.55 * q) * stria * fall
    c1 = np.array([0.62, 0.97, 1.0], np.float32)
    c2 = np.array([0.18, 0.78, 1.0], np.float32)
    c3 = np.array([0.16, 0.38, 1.0], np.float32)
    qq = q[..., None]
    col = np.where(qq < 0.3, c1 + (c2 - c1) * (qq / 0.3), c2 + (c3 - c2) * np.clip((qq - 0.3) / 0.7, 0, 1))
    img += col * body[..., None] * 1.5

    # ---- brushed hair-streaks feathering out of the convex edge at slight angles
    rng = np.random.default_rng(seed + 7)
    for k, ang in enumerate((5.0, 9.0, 13.0)):
        ta = math.tan(math.radians(ang))
        a2 = (acr + ta * uc * L) / W
        sf = _striae8(u, a2, seed + 20 + k, fu=30.0, fv=1400.0, blur_u=6.0, blur_v=0.55, nv=4096)
        hair = np.clip((sf - 0.6) / 0.4, 0, 1) ** 1.5
        zone = _ss(cpos - 0.2, -0.4, vn) * np.exp(-np.clip(edge_c - vn, 0, None) / (0.45 + 0.4 * uc))
        img += (hair * zone * fall * _ss(0.03, 0.2, uc))[..., None] * \
            np.array([0.3, 0.62, 1.0], np.float32) * rng.uniform(0.6, 0.9)

    # ---- spectral ribbons on the concave side
    grow = _ss(0.02, 0.14, uc)
    fg_c = cpos + 0.1 + 0.03 * uc
    sfg = _striae8(u, vn, seed + 45, fu=60.0, fv=40.0, blur_u=5.0, blur_v=0.7)
    fg = np.exp(-((vn - fg_c) / (0.05 + 0.03 * uc)) ** 2) * grow * fall
    img += (fg * (0.55 + 0.6 * sfg))[..., None] * np.array([0.5, 1.0, 0.65], np.float32) * 1.9 * sec
    gold_c = fg_c + 0.075
    gd = np.exp(-((vn - gold_c) / (0.03 + 0.015 * uc)) ** 2) * grow * fall
    img += (gd * (0.4 + 0.7 * sfg))[..., None] * np.array([1.0, 0.8, 0.4], np.float32) * 0.9 * sec
    mw_ = 0.32 + 0.4 * uc                                         # magenta band full width (vn units)
    m_c = gold_c + 0.04 + mw_ / 2
    sm = _striae8(u, vn, seed + 50, fu=55.0, fv=34.0, blur_u=5.0, blur_v=0.7)
    md = (vn - m_c) / (mw_ / 2) + 0.35 * (sm - 0.5)
    mag_b = _ss(1.15, 0.45, np.abs(md)) * (0.55 + 0.75 * sm ** 1.3)
    brk = _ss(-0.2, 0.3, (_striae8(u, vn * 0.02, seed + 60, fu=10.0, fv=1.0, blur_u=3.0) - 0.4) * 2.0)
    mag_b = mag_b * (0.5 + 0.5 * brk) * grow * fall
    mcol = np.array([1.0, 0.3, 0.8], np.float32)
    pcol = np.array([0.8, 0.28, 1.0], np.float32)
    mcc = np.clip(md * 0.5 + 0.5, 0, 1)[..., None]
    img += mag_b[..., None] * (mcol * (1 - mcc) + pcol * mcc) * 2.1 * sec
    # second, fainter diverging ribbon (violet -> magenta with a green thread on its inner edge)
    s2c = m_c + mw_ / 2 + 0.22 + 0.45 * uc
    s2w = 0.22 + 0.3 * uc
    s2 = _striae8(u, vn, seed + 70, fu=45.0, fv=26.0, blur_u=5.0, blur_v=0.7)
    d2_ = (vn - s2c) / s2w + 0.4 * (s2 - 0.5)
    b2 = _ss(1.1, 0.3, np.abs(d2_)) * (0.3 + 0.9 * s2 ** 1.5)
    brk2 = _ss(-0.1, 0.35, (_striae8(u, vn * 0.02, seed + 61, fu=8.0, fv=1.0, blur_u=3.0) - 0.45) * 2.0)
    b2 = b2 * (0.25 + 0.75 * brk2) * _ss(0.08, 0.3, uc) * fall
    c2c = np.clip(d2_ * 0.5 + 0.5, 0, 1)[..., None]
    img += b2[..., None] * (np.array([0.55, 0.3, 1.0], np.float32) * (1 - c2c) +
                            np.array([0.95, 0.3, 0.8], np.float32) * c2c) * 1.3 * sec
    g2 = np.exp(-((vn - (s2c - s2w * 0.9)) / (0.03 + 0.02 * uc)) ** 2) * _ss(0.1, 0.3, uc) * fall * brk2
    img += g2[..., None] * np.array([0.5, 1.0, 0.7], np.float32) * 0.35 * sec
    # feathered striae bleeding outward past the ribbons
    ta = math.tan(math.radians(-6.0))
    a2 = (acr + ta * uc * L) / W
    sf = _striae8(u, a2, seed + 80, fu=26.0, fv=1400.0, blur_u=6.0, blur_v=0.55, nv=4096)
    hair = np.clip((sf - 0.62) / 0.38, 0, 1) ** 1.5
    zone = _ss(m_c - 0.1, m_c + 0.2, vn) * np.exp(-np.clip(vn - (s2c + s2w), 0, None) / (0.4 + 0.3 * uc))
    img += (hair * zone * fall * _ss(0.05, 0.25, uc))[..., None] * np.array([0.45, 0.4, 1.0], np.float32) * \
        0.5 * sec
    # soft blue veil across the ribbon zone so the bands sit in light, not on black
    vz = _ss(cpos, cpos + 0.3, vn) * np.exp(-np.clip(vn - s2c, 0, None) / 0.6) * fall
    img += vz[..., None] * np.array([0.05, 0.12, 0.3], np.float32)

    veil = np.exp(-((vn + 0.2) / 1.5) ** 2) * fall
    img += veil[..., None] * np.array([0.02, 0.06, 0.14], np.float32)

    out = np.zeros((Hp, Wp, 3), np.float32)
    out[by0:by1, bx0:bx1] = img
    # ---- sparkle dust: on the ribbons (magenta, white, gold), a few in the body (cyan)
    rng = np.random.default_rng(seed + 31)
    uq = rng.uniform(0.04, 0.9, n_dust) ** 1.1
    wq = (w0 + (w1 - w0) * uq ** wexp) * W
    kind = rng.random(n_dust)
    vq = np.where(kind < 0.55, 0.95 + 0.9 * uq + rng.normal(0, 1, n_dust) * (0.25 + 0.3 * uq),
                  np.where(kind < 0.75, 0.63 + rng.normal(0, 0.05, n_dust), rng.uniform(-1.1, 0.45, n_dust)))
    aq = vq * wq + bend * L * uq ** 2
    x = head[0] + d[0] * uq * L - d[1] * aq
    y = head[1] + d[1] * uq * L + d[0] * aq
    cols = np.array([[1.0, 0.4, 0.95], [1.0, 0.9, 1.0], [1.0, 0.78, 0.45], [0.6, 0.95, 1.0]], np.float32)
    ci = np.where(kind < 0.75, rng.choice(3, n_dust, p=[0.5, 0.35, 0.15]), 3)
    m = (rng.random(n_dust) ** 5 * 1.5 + 0.05) * np.clip(1.0 - uq, 0.15, 1)
    acc = np.zeros((Hp, Wp, 3), np.float32)
    ok = (x >= 0) & (x < Wp - 1) & (y >= 0) & (y < Hp - 1)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    for c in range(3):
        np.add.at(acc[..., c], (yi, xi), m[ok] * cols[ci[ok], c])
    out += C.blur(acc, 0.5 * s + 0.2) * 3.0

    U = np.full((Hp, Wp), 5.0, np.float32)
    VN = np.full((Hp, Wp), 9.0, np.float32)
    B = np.zeros((Hp, Wp), np.float32)
    U[by0:by1, bx0:bx1] = u
    VN[by0:by1, bx0:bx1] = vn
    B[by0:by1, bx0:bx1] = (np.exp(-((vn + 0.3) / 0.8) ** 2) * fall).astype(np.float32)
    return out, U, VN, B


def ion_tail9(Wp, Hp, head, d, L, W, s, curve=0.0, seed=7):
    """Thin, hard-edged blue ion tail with fine parallel filaments; bends by `curve` (fraction of L across,
    sign picks the side) so it visibly peels away from the dust tail."""
    ex, ey = head[0] + d[0] * L, head[1] + d[1] * L
    pad = 0.03 * W + abs(curve) * L
    bx0, bx1 = int(max(min(head[0], ex) - pad, 0)), int(min(max(head[0], ex) + pad, Wp))
    by0, by1 = int(max(min(head[1], ey) - pad, 0)), int(min(max(head[1], ey) + pad, Hp))
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    uc = np.clip(u, 0, 1)
    a = -px * d[1] + py * d[0] - curve * L * uc ** 2
    w = (0.6 + 1.8 * uc) * s + 0.35
    fall = _ss(-0.002, 0.01, u) * np.clip(1 - u, 0, 1) ** 1.3 * (0.5 + 0.5 * np.exp(-uc / 0.3))
    st = _striae8(u, a / W, seed, fu=40.0, fv=900.0, blur_u=6.0, blur_v=0.6, nv=2048)
    core = _ss(w * 1.2, w * 0.4, np.abs(a))
    glow = np.exp(-(a / (w * 4.0)) ** 2) * 0.3 + np.exp(-(a / (w * 14.0)) ** 2) * 0.08
    fil = np.zeros_like(a)
    rng = np.random.default_rng(seed)
    for k in range(3):
        off = rng.uniform(2.5, 5.0) * w * (1 if k % 2 else -1)
        fil += _ss(w * 0.8, w * 0.15, np.abs(a - off)) * rng.uniform(0.25, 0.45) * _ss(0.05, 0.25, uc)
    k = (core * (0.75 + 0.5 * st) + glow + fil * (0.5 + st)) * fall
    img = k[..., None] * np.array([0.3, 0.55, 1.0], np.float32) * 1.4
    img += (core * fall * np.exp(-uc / 0.1))[..., None] * np.array([0.7, 0.9, 1.0], np.float32) * 1.2
    out = np.zeros((Hp, Wp, 3), np.float32)
    out[by0:by1, bx0:bx1] = img
    return out


def star_flare(img, x, y, s, amt=1.0, rot=0.3, col=(0.85, 0.95, 1.0)):
    """6-point star (4 long + 2 short spikes) with a soft halation disc and faint ring, added in place."""
    H, W = img.shape[:2]
    R = int(70 * s) + 4
    x0, y0 = int(x) - R, int(y) - R
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return
    yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
    dx, dy = xx - x, yy - y
    r = np.sqrt(dx * dx + dy * dy) + 1e-3
    k = np.zeros_like(r)
    for (ang, ln, wd, it) in ((rot, 62, 0.75, 1.0), (rot + math.pi / 2, 46, 0.75, 0.8),
                              (rot + math.pi / 4, 20, 0.6, 0.4), (rot - math.pi / 4, 20, 0.6, 0.4)):
        ca, sa = math.cos(ang), math.sin(ang)
        a_ = dx * ca + dy * sa
        b_ = -dx * sa + dy * ca
        Ls = ln * s
        wdt = wd * s + 0.35
        k += np.exp(-(b_ / (wdt * (1 + 1.5 * np.abs(a_) / Ls))) ** 2) * np.clip(1 - np.abs(a_) / Ls, 0, 1) ** 2.2 * it
    halo = np.exp(-(r / (8 * s)) ** 2) * 0.4 + np.exp(-r / (20 * s)) * 0.12
    ring = np.exp(-((r - 17 * s) / (2.0 * s)) ** 2) * 0.05
    c = np.asarray(col, np.float32)
    img[cy0:cy1, cx0:cx1] += (k * 1.3 + halo)[..., None] * c * amt
    img[cy0:cy1, cx0:cx1] += ring[..., None] * np.array([0.6, 0.8, 1.0], np.float32) * amt
