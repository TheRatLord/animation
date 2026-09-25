"""Round-18 comet additions for s07_comet_night (yn_05 warm/cool split).

warm_dust_tail: a SEPARATE curved dust tail diverging from the cool cyan tail on its concave side:
  crisp inner edge with a thin spectral fringe (teal -> green -> gold) facing the cyan tail, a magenta body
  that turns pink and then peach toward the feathered outer edge (and further from the nucleus), long
  brushed striae along its length, and a scatter of sparkle dust drifting off the outer edge.
Returns additive RGB on the full plate (Wp, Hp), same (u, n) frame convention as comet_tail9:
  n = (-d[1], d[0]); the tail bends toward +n.
"""
import math

import numpy as np
import cv2

from lib import core as C
from s07_comet_night_comet8 import _ss, _striae8


def warm_dust_tail(Wp, Hp, head, d, L, W, s, bend=0.3, extra=0.16, w0=0.004, w1=0.07, off0=0.012,
                   seed=18, n_dust=900, amt=1.0):
    n = (-d[1], d[0])

    def centre(u):
        return off0 * W * np.clip(u / 0.15, 0, 1) + (bend + extra) * L * u * u

    pts = []
    for uq in np.linspace(0, 1.0, 20):
        wq = (w0 + (w1 - w0) * uq ** 0.8) * W
        for v in (-2.0, 3.5):
            a = centre(uq) + v * wq
            pts.append((head[0] + d[0] * uq * L + n[0] * a, head[1] + d[1] * uq * L + n[1] * a))
    pts = np.array(pts)
    bx0, by0 = np.maximum(pts.min(0) - 0.02 * W, 0).astype(int)
    bx1, by1 = pts.max(0) + 0.02 * W
    bx1, by1 = int(min(bx1, Wp)), int(min(by1, Hp))
    out = np.zeros((Hp, Wp, 3), np.float32)
    if bx1 <= bx0 or by1 <= by0:
        return out
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    uc = np.clip(u, 0, 1.1)
    a = px * n[0] + py * n[1] - centre(uc)
    wu = (w0 + (w1 - w0) * uc ** 0.8) * W
    vn = a / wu                      # -1 inner (cyan side) ... +1 outer
    start = _ss(-0.002, 0.03, u)
    fall = start * np.clip(1.0 - u / 0.92, 0, 1) ** 1.3 * (0.45 + 0.55 * np.exp(-uc / 0.35))
    st1 = _striae8(u, vn, seed, fu=70.0, fv=55.0, blur_u=5.0, blur_v=0.6)
    st2 = _striae8(u, vn, seed + 1, fu=30.0, fv=18.0, blur_u=4.0, blur_v=0.8)
    st = 0.55 * st1 + 0.45 * st2
    img = np.zeros(u.shape + (3,), np.float32)
    # body: crisp inner edge, broad feathered outer edge (with hair striae)
    inner = _ss(-1.08, -0.9, vn + 0.06 * (st2 - 0.5))
    edge_o = 0.85 + 0.5 * (st - 0.5)
    outer = _ss(edge_o + 0.45, edge_o - 0.35, vn)
    body = inner * outer * (0.55 + 0.6 * st ** 1.2) * fall
    q = np.clip((vn + 1) / 2, 0, 1)
    qq = (q * 0.75 + 0.35 * np.clip(uc / 0.8, 0, 1))[..., None]
    c_mag = np.array([1.0, 0.24, 0.78], np.float32)
    c_pink = np.array([1.0, 0.4, 0.58], np.float32)
    c_peach = np.array([1.0, 0.64, 0.42], np.float32)
    col = np.where(qq < 0.5, c_mag + (c_pink - c_mag) * (qq / 0.5),
                   c_pink + (c_peach - c_pink) * np.clip((qq - 0.5) / 0.5, 0, 1))
    img += col * body[..., None] * 1.25
    # hot inner lip (just inside the fringe): whiter magenta
    lip = np.exp(-((vn + 0.86) / 0.09) ** 2) * fall * (0.6 + 0.6 * st1)
    img += lip[..., None] * np.array([1.0, 0.55, 0.9], np.float32) * 0.8
    # spectral fringe between the warm and cool tails: teal -> green -> gold, thin, broken
    brk = _ss(-0.2, 0.35, (_striae8(u, vn * 0.02, seed + 5, fu=9.0, fv=1.0, blur_u=3.0) - 0.4) * 2.0)
    grow = _ss(0.02, 0.12, uc)
    for (c_, wdt, colr, it) in ((-1.28, 0.07, (0.25, 0.85, 1.0), 0.8), (-1.16, 0.06, (0.45, 1.0, 0.55), 1.1),
                                (-1.05, 0.05, (1.0, 0.85, 0.35), 0.8)):
        f = np.exp(-((vn - c_) / (wdt + 0.03 * uc)) ** 2) * grow * fall * (0.45 + 0.55 * brk)
        img += f[..., None] * np.array(colr, np.float32) * it
    # feathered hair streaks bleeding off the outer edge at a small angle
    ta = math.tan(math.radians(-5.0))
    a2 = (a + ta * uc * L) / W
    sf = _striae8(u, a2, seed + 9, fu=26.0, fv=1400.0, blur_u=6.0, blur_v=0.55, nv=4096)
    hair = np.clip((sf - 0.6) / 0.4, 0, 1) ** 1.5
    zone = _ss(0.2, 0.8, vn) * np.exp(-np.clip(vn - 1.0, 0, None) / 0.9)
    img += (hair * zone * fall * _ss(0.05, 0.2, uc))[..., None] * np.array([1.0, 0.5, 0.75], np.float32) * 0.6
    # faint warm veil so the band sits in light
    veil = np.exp(-(vn / 1.8) ** 2) * fall
    img += veil[..., None] * np.array([0.08, 0.03, 0.08], np.float32)
    out[by0:by1, bx0:bx1] = img * amt
    # sparkle dust drifting off the outer edge
    rng = np.random.default_rng(seed + 31)
    uq = rng.uniform(0.05, 0.85, n_dust) ** 1.1
    wq = (w0 + (w1 - w0) * uq ** 0.8) * W
    vq = 0.7 + np.abs(rng.normal(0, 1, n_dust)) * (0.5 + 0.8 * uq)
    aq = centre(uq) + vq * wq
    x = head[0] + d[0] * uq * L + n[0] * aq
    y = head[1] + d[1] * uq * L + n[1] * aq
    cols = np.array([[1.0, 0.45, 0.85], [1.0, 0.92, 0.95], [1.0, 0.75, 0.45]], np.float32)
    ci = rng.choice(3, n_dust, p=[0.5, 0.32, 0.18])
    m = (rng.random(n_dust) ** 5 * 1.6 + 0.06) * np.clip(1.0 - uq, 0.15, 1) * np.exp(-(vq - 0.7) / 2.0)
    acc = np.zeros((Hp, Wp, 3), np.float32)
    ok = (x >= 0) & (x < Wp - 1) & (y >= 0) & (y < Hp - 1)
    xi, yi = x[ok].astype(int), y[ok].astype(int)
    for c in range(3):
        np.add.at(acc[..., c], (yi, xi), m[ok] * cols[ci[ok], c])
    out += C.blur(acc, 0.5 * s + 0.2) * 3.0 * amt
    return out
