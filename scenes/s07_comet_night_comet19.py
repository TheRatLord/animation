"""Round-19 comet for s07_comet_night: a true SPLIT tail (Your Name yn_05 / brief).

Round 18 still read as one wide flat fan with rainbow bands (a broad cyan dust fan with the warm tail glued
to it).  Now two distinct filaments diverge from the nucleus with dark sky between them:
  ion_tail19   narrow, nearly straight, COOL: hard cyan-white core line -> cyan -> blue, fine parallel
               filaments, slowly widening and tapering to nothing (no solid wedge).
  dust_tail19  wider, curved, WARM: magenta at the inner edge -> pink -> gold toward the tip/outer edge
               (built on s07_comet_night_comet18.warm_dust_tail: spectral fringe facing the ion tail,
               brushed striae, sparkle debris only along this curved tail), tapered at the tip.
  head_fringe  small spectral fringe around the hard white core.
Also returns (U, VN, B) aux fields in the dust frame for the per-frame growth/streamer flow.
"""
import math

import numpy as np
import cv2

from lib import core as C
from s07_comet_night_comet8 import _ss, _striae8
import s07_comet_night_comet18 as CM18


def ion_tail19(Wp, Hp, head, d, L, W, s, curve=-0.03, seed=7, w_end=0.022):
    n = (-d[1], d[0])
    ex, ey = head[0] + d[0] * L, head[1] + d[1] * L
    pad = w_end * W * 3 + abs(curve) * L + 0.02 * W
    bx0, bx1 = int(max(min(head[0], ex) - pad, 0)), int(min(max(head[0], ex) + pad, Wp))
    by0, by1 = int(max(min(head[1], ey) - pad, 0)), int(min(max(head[1], ey) + pad, Hp))
    out = np.zeros((Hp, Wp, 3), np.float32)
    if bx1 <= bx0 or by1 <= by0:
        return out
    ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    uc = np.clip(u, 0, 1.1)
    a = px * n[0] + py * n[1] - curve * L * uc ** 2
    wu = (0.0012 + (w_end - 0.0012) * uc ** 0.9) * W + 0.6 * s
    vn = a / wu
    start = _ss(-0.002, 0.01, u)
    fall = start * np.clip(1.0 - u / 0.95, 0, 1) ** 1.4 * (0.4 + 0.6 * np.exp(-uc / 0.3))
    stf = _striae8(u, vn, seed, fu=60.0, fv=40.0, blur_u=6.0, blur_v=0.6)
    stm = _striae8(u, vn, seed + 1, fu=25.0, fv=14.0, blur_u=5.0, blur_v=0.8)
    edge = 1.0 + 0.25 * (stm - 0.5)
    body = _ss(edge + 0.15, edge - 0.45, np.abs(vn)) * (0.45 + 0.55 * stf ** 1.2) * fall
    q = np.clip(np.abs(vn), 0, 1)[..., None]
    c0 = np.array([0.7, 1.0, 1.0], np.float32)
    c1 = np.array([0.2, 0.8, 1.0], np.float32)
    c2 = np.array([0.15, 0.42, 1.0], np.float32)
    col = np.where(q < 0.4, c0 + (c1 - c0) * (q / 0.4), c1 + (c2 - c1) * np.clip((q - 0.4) / 0.6, 0, 1))
    img = col * body[..., None] * 1.45
    # hard core line (white near the head, cyan further out)
    cw = 0.12 + 0.05 * uc
    core = _ss(cw, cw * 0.3, np.abs(vn - 0.04 * (stm - 0.5))) * start * (np.exp(-uc / 0.2) + 0.3 * np.exp(-uc / 0.03))
    cc = np.clip(uc / 0.5, 0, 1)[..., None]
    img += core[..., None] * (np.array([1.0, 1.0, 1.0], np.float32) * (1 - cc) +
                              np.array([0.5, 0.95, 1.0], np.float32) * cc) * 2.0
    # fine filaments feathering off both edges at small angles
    for k, ang in enumerate((3.0, -3.5)):
        ta = math.tan(math.radians(ang))
        a2 = (a + ta * uc * L) / W
        sf = _striae8(u, a2, seed + 20 + k, fu=24.0, fv=1400.0, blur_u=6.0, blur_v=0.55, nv=4096)
        hair = np.clip((sf - 0.62) / 0.38, 0, 1) ** 1.5
        zone = np.exp(-np.clip(np.abs(vn) - 0.7, 0, None) / 1.2) * _ss(0.4, 0.9, np.abs(vn) + 0.2)
        img += (hair * zone * fall * _ss(0.04, 0.2, uc))[..., None] * np.array([0.3, 0.65, 1.0], np.float32) * 0.5
    # narrow cool glow (kept tight so the gap to the warm tail stays dark)
    glow = np.exp(-(vn / 2.0) ** 2) * fall
    img += glow[..., None] * np.array([0.04, 0.14, 0.32], np.float32)
    out[by0:by1, bx0:bx1] = img
    return out


def dust_frame(Wp, Hp, head, d, L, W, bend, extra, off0, w0, w1):
    """(U, VN, B) of the curved dust tail on the full plate (U = 5 / VN = 9 outside the box)."""
    n = (-d[1], d[0])
    ys, xs = np.mgrid[0:Hp, 0:Wp].astype(np.float32)
    px, py = xs - head[0], ys - head[1]
    u = (px * d[0] + py * d[1]) / L
    uc = np.clip(u, 0, 1.1)
    cen = off0 * W * np.clip(uc / 0.15, 0, 1) + (bend + extra) * L * uc * uc
    a = px * n[0] + py * n[1] - cen
    wu = (w0 + (w1 - w0) * uc ** 0.8) * W
    vn = (a / wu).astype(np.float32)
    B = (np.exp(-((vn - 0.8) / 1.2) ** 2) * _ss(-0.002, 0.03, u) * np.clip(1 - u / 0.92, 0, 1)).astype(np.float32)
    U = np.where((u > -0.05) & (np.abs(vn) < 6), u, 5.0).astype(np.float32)
    VN = np.where((u > -0.05) & (np.abs(vn) < 6), vn, 9.0).astype(np.float32)
    return U, VN, B


def dust_tail19(Wp, Hp, head, d, L, W, s, bend=0.3, extra=0.3, w0=0.004, w1=0.085, off0=0.006, seed=19,
                n_dust=1100):
    t = CM18.warm_dust_tail(Wp, Hp, head, d, L, W, s, bend=bend, extra=extra, w0=w0, w1=w1, off0=off0,
                            seed=seed, n_dust=n_dust)
    U, VN, B = dust_frame(Wp, Hp, head, d, L, W, bend, extra, off0, w0, w1)
    # pink-magenta -> GOLD toward the tip and the outer edge; taper the tip so it never reads as a wedge
    g = np.clip(0.75 * _ss(0.2, 0.8, U) + 0.35 * _ss(0.3, 1.6, VN), 0, 1)[..., None]
    lum = t.max(-1, keepdims=True)
    gold = np.array([1.0, 0.72, 0.34], np.float32)
    t = t * (1 - 0.6 * g) + lum * gold * (0.6 * g)
    t = t * (1 - 0.5 * _ss(0.55, 0.95, U))[..., None]
    # keep the root slim and unsaturated (no blown white wedge right behind the nucleus)
    t = t * (0.6 + 0.4 * _ss(0.02, 0.3, U))[..., None]
    return t, U, VN, B


def head_fringe(Wp, Hp, head, W, s):
    """Tiny spectral fringe (cyan inside -> green -> magenta outside) around the hard white core."""
    out = np.zeros((Hp, Wp, 3), np.float32)
    R = int(0.02 * W) + 4
    x0, y0 = int(head[0]) - R, int(head[1]) - R
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + 2 * R, Wp), min(y0 + 2 * R, Hp)
    ys, xs = np.mgrid[y0c:y1c, x0c:x1c].astype(np.float32)
    r = np.sqrt((xs - head[0]) ** 2 + (ys - head[1]) ** 2) / W
    img = np.zeros(r.shape + (3,), np.float32)
    for (rc, wd, col, it) in ((0.0028, 0.0009, (0.3, 0.95, 1.0), 0.35), (0.0042, 0.0009, (0.5, 1.0, 0.55), 0.22),
                              (0.0058, 0.0012, (1.0, 0.35, 0.85), 0.2)):
        img += np.exp(-((r - rc) / wd) ** 2)[..., None] * np.array(col, np.float32) * it
    out[y0c:y1c, x0c:x1c] = img
    return out
