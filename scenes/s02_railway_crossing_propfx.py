"""s02 helper: painted finishing passes for the near props (vending machine, barrier gearbox) drawn into the
supersampled vector canvas: alpha-BLENDED polygons (the canvas primitives overwrite), stepped vertical value
gradients (lit top -> darker, cooler base), diagonal glass glare streaks and crisp sun-side edge highlights.
All inputs are screen-space points (1x px); the canvas is premultiplied RGBA uint8 at cv.ss supersampling."""
import numpy as np
import cv2


def blend_poly(cv, pts, rgb, a):
    """Alpha-blend a filled polygon (premultiplied, keeps the existing coverage) onto the canvas."""
    if a <= 0:
        return
    ss = cv.ss
    p = np.asarray(pts, np.float64) * ss
    x0, y0 = np.floor(p.min(0) - 2).astype(int)
    x1, y1 = np.ceil(p.max(0) + 2).astype(int) + 1
    Hb, Wb = cv.buf.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, Wb), min(y1, Hb)
    if x1 <= x0 or y1 <= y0:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    q = np.round((p - [x0, y0]) * 16).astype(np.int32)
    cv2.fillPoly(m, [q], 255, cv2.LINE_8, shift=4)
    reg = cv.buf[y0:y1, x0:x1].astype(np.float32)
    cov = reg[..., 3:4] / 255.0                            # only over what is already painted
    k = (m.astype(np.float32) / 255.0 * min(a, 1.0))[..., None] * cov
    col = np.asarray(np.clip(rgb, 0, 1), np.float32) * 255.0
    rgb_ = reg[..., :3] * (1 - k) + col * cov * k
    cv.buf[y0:y1, x0:x1, :3] = np.clip(rgb_ + 0.5, 0, 255).astype(np.uint8)


def _lerp(p, q, t):
    return (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)


def quad_pt(quad, u, v):
    """Point at (u, v) in a screen quad (tl, tr, br, bl); u 0..1 left->right, v 0..1 top->bottom."""
    tl, tr, br, bl = quad
    return _lerp(_lerp(tl, tr, u), _lerp(bl, br, u), v)


def sub_quad(quad, u0, v0, u1, v1):
    return [quad_pt(quad, u0, v0), quad_pt(quad, u1, v0), quad_pt(quad, u1, v1), quad_pt(quad, u0, v1)]


def vgrad(cv, quad, rgb, a_top, a_bot, v0=0.0, v1=1.0, n=8):
    """Stepped vertical gradient (n painted bands) of `rgb` from alpha a_top (at v0) to a_bot (at v1)."""
    for i in range(n):
        va, vb = v0 + (v1 - v0) * i / n, v0 + (v1 - v0) * (i + 1) / n
        a = a_top + (a_bot - a_top) * (i + 0.5) / n
        blend_poly(cv, sub_quad(quad, 0.0, va, 1.0, vb), rgb, a)


def glare(cv, quad, u, w, slant=0.35, a=0.35, rgb=(1.0, 1.0, 1.0), v0=0.0, v1=1.0):
    """Diagonal glare streak across a glass panel: a band starting at u (top) of width w, leaning left."""
    pts = [quad_pt(quad, u, v0), quad_pt(quad, u + w, v0), quad_pt(quad, u + w - slant, v1),
           quad_pt(quad, u - slant, v1)]
    blend_poly(cv, pts, rgb, a)


def edge(cv, p, q, width, rgb, a=1.0):
    """Crisp highlight line (drawn opaque when a >= 1)."""
    if a >= 0.999:
        cv.line([p, q], width, rgb)
        return
    d = np.array(q, np.float64) - np.array(p, np.float64)
    n_ = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9) * width * 0.5
    blend_poly(cv, [tuple(p + n_), tuple(np.array(q) + n_), tuple(np.array(q) - n_), tuple(p - n_)], rgb, a)
