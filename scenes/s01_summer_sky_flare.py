"""s01 lens-flare pieces: a visible chain of tinted ghosts (rounded hexagons + discs + one ring) along the
axis from the sun through the optical centre, and a faint anamorphic streak. Work at reduced res."""
import math
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


# (position along sun->centre axis, radius as frac of w, tint, shape, strength)
CHAIN = [
    (0.55, 0.008, (1.0, 0.8, 0.5), 'disc', 0.08),     # small amber
    (1.0, 0.018, (0.7, 0.88, 1.0), 'hex', 0.025),     # very faint pale hex
    (1.45, 0.012, (0.88, 0.75, 1.0), 'disc', 0.025),  # faint violet dot
]


def ghost_chain(w, h, lx, ly, cx, cy, amt=1.0, rot=0.3):
    out = np.zeros((h, w, 3), np.float32)
    vx, vy = cx - lx, cy - ly
    for k, r, col, shape, st in CHAIN:
        gx, gy = lx + vx * k, ly + vy * k
        R = max(r * w, 1.5)
        bx0, bx1 = int(max(gx - R * 1.5 - 2, 0)), int(min(gx + R * 1.5 + 3, w))
        by0, by1 = int(max(gy - R * 1.5 - 2, 0)), int(min(gy + R * 1.5 + 3, h))
        if bx1 <= bx0 or by1 <= by0:
            continue
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        ex, ey = xx - gx, yy - gy
        rad = np.sqrt(ex * ex + ey * ey)
        if shape == 'hex':
            th = np.arctan2(ey, ex) + rot
            seg = math.pi / 3
            hx = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
            rad = rad * 0.2 + hx * 0.8
        q = rad / R
        c = np.asarray(col, np.float32)
        if shape == 'ring':
            m = np.stack([np.exp(-((q - 0.98) / 0.05) ** 2), np.exp(-((q - 0.94) / 0.05) ** 2),
                          np.exp(-((q - 0.9) / 0.05) ** 2)], -1) + _ss(1.0, 0.5, q)[..., None] * 0.15
        else:
            # filled body brighter toward the rim, with a thin chromatic fringe (R outside, B inside)
            # soft body (no flat fill): a gentle gradient that is brightest just inside a soft rim
            m = np.stack([_ss(1.12, 0.8, q), _ss(1.08, 0.76, q), _ss(1.04, 0.72, q)], -1)
            m = m * (0.3 + 0.7 * _ss(0.2, 0.95, q))[..., None]
        out[by0:by1, bx0:bx1] += m * c * st * amt * 2.2
    return cv2.GaussianBlur(out, (0, 0), max(1.0, 0.003 * w))


def anamorphic(w, h, lx, ly, amt=1.0, xs=None, ys=None):
    if xs is None:
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xs - lx, ys - ly
    core = np.exp(-(dy / (0.0035 * h)) ** 2) * (np.exp(-np.abs(dx) / (0.3 * w)) * 0.7 + np.exp(-np.abs(dx) / (0.06 * w)))
    wide = np.exp(-(dy / (0.012 * h)) ** 2) * np.exp(-np.abs(dx) / (0.15 * w)) * 0.25
    return ((core + wide)[..., None] * np.array([0.62, 0.8, 1.0], np.float32) * 0.22 * amt).astype(np.float32)
