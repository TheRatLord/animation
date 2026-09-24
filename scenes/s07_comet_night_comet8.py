"""Round-10 comet for s07_comet_night (Your Name yn_05).

comet_tail8: +across = concave side (the tail bends toward it). Across the tail (concave -> convex):
  a faint, broken magenta/gold secondary band (separated by a dark gap, diverging with distance), a thin
  broken magenta fringe, the HARD white-cyan core line, the cyan dust body fading to blue with strong
  longitudinal striations, and brushed blue hair-streaks feathering out of the ragged convex edge.
ion_tail8: straight, thin, hard-edged blue ion tail with fine parallel filaments.
comet_head8: small hard nucleus with a tight bloom.
"""
import math

import numpy as np
import cv2

from lib import core as C


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _striae8(u, vn, seed, fu, fv, blur_u=4.0, blur_v=0.7, nv=1024, nu=256):
    """Longitudinal brush striae 0..1 (long along u, fine across vn)."""
    rng = np.random.default_rng(seed)
    t = rng.random((nv, nu)).astype(np.float32)
    t = cv2.GaussianBlur(t, (0, 0), sigmaX=blur_u, sigmaY=blur_v)
    t = (t - t.min()) / (t.max() - t.min() + 1e-6)
    mx = np.mod(u * fu, nu).astype(np.float32)
    my = np.mod(vn * fv + nv / 2, nv).astype(np.float32)
    return cv2.remap(t, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def comet_tail8(Wp, Hp, head, d, L, W, s, bend=0.2, w0=0.003, w1=0.13, wexp=0.85, seed=3, n_dust=3000,
                sec=1.0):
    """Returns (rgb additive, U, VN, BODY) full plate."""
    pts = []
    for uq in np.linspace(0, 1.05, 16):
        wq = (w0 + (w1 - w0) * uq ** wexp) * W
        for v in (-2.2, 2.6):
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
    stria = 0.45 + 0.75 * st_f ** 1.4 + 0.45 * st_m
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

    # ---- thin broken magenta fringe just outside the core (concave side)
    brk = _ss(-0.15, 0.3, (_striae8(u, vn * 0.05, seed + 40, fu=12.0, fv=1.0, blur_u=3.0) - 0.5) * 2.0)
    fr = np.exp(-((vn - (cpos + 0.15 + 0.05 * uc)) / (0.06 + 0.03 * uc)) ** 2) * _ss(0.04, 0.15, uc) * fall
    img += (fr * (0.35 + 0.65 * brk) * (0.5 + 0.8 * st_f))[..., None] * \
        np.array([1.0, 0.2, 0.75], np.float32) * 1.4 * sec

    # ---- separated secondary band: diverges with distance, broken, streaky magenta/violet with gold
    sc = cpos + 0.32 + 0.55 * uc + 0.08 * (st_m - 0.5)
    sw = 0.1 + 0.12 * uc
    sd = (vn - sc) / sw
    sb = _striae8(u, vn, seed + 50, fu=50.0, fv=30.0, blur_u=5.0, blur_v=0.7)
    band = _ss(1.0, 0.25, np.abs(sd) + 0.6 * (sb - 0.5)) * (0.25 + 1.0 * sb ** 1.5)
    brk2 = _ss(-0.1, 0.35, (_striae8(u, vn * 0.02, seed + 60, fu=9.0, fv=1.0, blur_u=3.0) - 0.45) * 2.0)
    band = band * (0.2 + 0.8 * brk2) * _ss(0.1, 0.3, uc) * fall
    mag = np.array([0.95, 0.22, 0.85], np.float32)
    vio = np.array([0.55, 0.3, 1.0], np.float32)
    gold = np.array([1.0, 0.7, 0.3], np.float32)
    sdc = np.clip(sd, 0, 1)[..., None]
    bc = mag * (1 - sdc) + vio * sdc
    goldness = (_ss(0.55, 0.8, sb) * _ss(0.2, 0.5, uc) + _ss(-0.2, -0.6, sd) * 0.6)[..., None]
    goldness = np.clip(goldness, 0, 1)
    bc = bc * (1 - goldness * 0.6) + gold * (goldness * 0.6)
    img += band[..., None] * bc * 2.2 * sec

    veil = np.exp(-((vn + 0.2) / 1.5) ** 2) * fall
    img += veil[..., None] * np.array([0.02, 0.06, 0.14], np.float32)

    out = np.zeros((Hp, Wp, 3), np.float32)
    out[by0:by1, bx0:bx1] = img
    # ---- sparkle dust: in the secondary band / fringe (magenta, white, gold), a few in the body (cyan)
    rng = np.random.default_rng(seed + 31)
    uq = rng.uniform(0.04, 0.9, n_dust) ** 1.1
    wq = (w0 + (w1 - w0) * uq ** wexp) * W
    kind = rng.random(n_dust)
    vq = np.where(kind < 0.55, 0.82 + 0.55 * uq + rng.normal(0, 1, n_dust) * (0.18 + 0.12 * uq),
                  np.where(kind < 0.75, 0.63 + rng.normal(0, 0.05, n_dust), rng.uniform(-1.1, 0.45, n_dust)))
    aq = vq * wq + bend * L * uq ** 2
    x = head[0] + d[0] * uq * L - d[1] * aq
    y = head[1] + d[1] * uq * L + d[0] * aq
    cols = np.array([[1.0, 0.4, 0.95], [1.0, 0.9, 1.0], [1.0, 0.78, 0.45], [0.6, 0.95, 1.0]], np.float32)
    ci = np.where(kind < 0.75, rng.choice(3, n_dust, p=[0.55, 0.3, 0.15]), 3)
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


def ion_tail8(Wp, Hp, head, d, L, W, s, seed=7):
    """Straight, thin, hard-edged blue ion tail with 2-3 fine parallel filaments; clearly visible."""
    ex, ey = head[0] + d[0] * L, head[1] + d[1] * L
    pad = 0.03 * W
    bx0, bx1 = int(max(min(head[0], ex) - pad, 0)), int(min(max(head[0], ex) + pad, Wp))
    by0, by1 = int(max(min(head[1], ey) - pad, 0)), int(min(max(head[1], ey) + pad, Hp))
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    a = -px * d[1] + py * d[0]
    uc = np.clip(u, 0, 1)
    w = (0.7 + 2.2 * uc) * s + 0.35
    fall = _ss(-0.002, 0.01, u) * np.clip(1 - u, 0, 1) ** 1.3 * (0.5 + 0.5 * np.exp(-uc / 0.3))
    st = _striae8(u, a / W, seed, fu=40.0, fv=900.0, blur_u=6.0, blur_v=0.6, nv=2048)
    core = _ss(w * 1.2, w * 0.4, np.abs(a))
    glow = np.exp(-(a / (w * 4.0)) ** 2) * 0.3 + np.exp(-(a / (w * 14.0)) ** 2) * 0.07
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


def comet_head8(Wp, Hp, head, s, W, amt=1.0):
    """Small, hard nucleus with a tight bloom."""
    x0, y0 = int(max(head[0] - 0.12 * W, 0)), int(max(head[1] - 0.12 * W, 0))
    x1, y1 = int(min(head[0] + 0.12 * W, Wp)), int(min(head[1] + 0.12 * W, Hp))
    out = np.zeros((Hp, Wp, 3), np.float32)
    if x1 <= x0 or y1 <= y0:
        return out
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dd = np.sqrt((xs - head[0]) ** 2 + (ys - head[1]) ** 2) / W
    img = (1.0 / (1.0 + (dd / 0.0011) ** 2.6))[..., None] * np.array([1.0, 1.0, 1.0], np.float32) * 3.0
    img += (1.0 / (1.0 + (dd / 0.0035) ** 2))[..., None] * np.array([0.6, 0.9, 1.0], np.float32) * 0.45
    img += np.exp(-dd / 0.015)[..., None] * np.array([0.25, 0.55, 1.0], np.float32) * 0.14
    img += np.exp(-dd / 0.05)[..., None] * np.array([0.1, 0.25, 0.55], np.float32) * 0.06
    out[y0:y1, x0:x1] = img * amt
    return out
