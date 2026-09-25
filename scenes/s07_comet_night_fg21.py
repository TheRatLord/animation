"""Round-21 near-foreground wipe-by for s07_comet_night (reviewer: for the first ~3 s the tilt moves the whole
frame as one painting - bring a near layer in early).

A near lakeside cedar standing on the reed bank, painted as a near-black silhouette: a slightly leaning trunk
and tiers of drooping, needle-fringed branch pads (sky shows between the tiers), a thin cool comet rim on the
edges that face the comet (upper-left) and a faint warm twilight kiss on the far-left fringe.  It rises from
the bottom-right corner by ~1.3 s and, being far nearer than anything else, climbs ~1.4x faster than the town
and slides out of frame right with the truck before the end (a Shinkai foreground wipe-by).
Returns a premultiplied RGBA sprite + its anchor (crown tip) in sprite px.
"""
import math

import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def cedar_sprite(W, H, seed=7, height=0.72, width=0.3):
    s = H / 1080.0
    q = 3
    sw, sh = int(width * W), int(height * H)
    Wq, Hq = sw * q, sh * q
    rng = np.random.default_rng(seed)
    m = np.zeros((Hq, Wq), np.uint8)
    cx = 0.5 * sw
    top = 0.012 * H
    lean = 0.035                                    # trunk leans slightly left toward the lake
    def trunk_x(y):
        return cx - lean * (y - top)
    # trunk (tapered)
    tw0, tw1 = 0.004 * W, 0.016 * W
    pts = [(trunk_x(top) - tw0 * 0.3, top), (trunk_x(top) + tw0 * 0.3, top), (trunk_x(sh) + tw1, sh),
           (trunk_x(sh) - tw1, sh)]
    cv2.fillPoly(m, [np.array([(p[0] * q, p[1] * q) for p in pts], np.int32)], 255, cv2.LINE_AA)
    # tiers of drooping needle pads, conical envelope, gaps between them
    y = top
    n = 0
    while y < sh * 0.97:
        f = (y - top) / (sh - top)
        half = (0.02 + 0.44 * f ** 0.7) * sw * rng.uniform(0.75, 1.1)
        th = (0.016 + 0.045 * f ** 0.6) * H * rng.uniform(0.8, 1.15)
        for side in (-1, 1):
            if n > 1 and rng.random() < 0.12:
                continue
            L = half * rng.uniform(0.7, 1.05)
            x0 = trunk_x(y + th * 0.3)
            droop = th * rng.uniform(0.35, 0.8)
            upper, lower = [], []
            K = 64
            kern = np.array([0.25, 0.5, 0.25])
            FU = np.convolve(rng.uniform(0.0, 1.0, K + 1) ** 3, kern, 'same') * 1.5 * 0.55 * th
            FL = np.convolve(rng.uniform(0.0, 1.0, K + 1) ** 3, kern, 'same') * 1.5 * 0.45 * th
            for k in range(K + 1):
                u = k / K
                xx = x0 + side * L * u
                # a drooping branch arc; the pad is thick near the trunk and thins to a tuft at the tip
                yc = y + th * 0.35 + droop * u ** 1.6
                thick = th * (0.55 + 0.45 * (1 - u)) * (0.4 + 0.6 * math.sin(math.pi * min(u * 1.1 + 0.08, 1.0)))
                fr_u = FU[k]                                         # needle tufts on the fringe
                fr_l = FL[k]
                upper.append((xx, yc - thick * 0.45 - fr_u))
                lower.append((xx, yc + thick * 0.55 + fr_l))
            poly = upper + lower[::-1]
            cv2.fillPoly(m, [np.array([(p[0] * q, p[1] * q) for p in poly], np.int32)], 255, cv2.LINE_AA)
        # crown tip spike
        if n == 0:
            tip = [(trunk_x(top) - 0.002 * W, top + th * 0.8), (trunk_x(top), top - 0.012 * H),
                   (trunk_x(top) + 0.002 * W, top + th * 0.8)]
            cv2.fillPoly(m, [np.array([(p[0] * q, p[1] * q) for p in tip], np.int32)], 255, cv2.LINE_AA)
        y += th * rng.uniform(0.75, 1.15) + (0.002 + 0.012 * f) * H * rng.uniform(0.2, 1.0)
        n += 1
    a = cv2.resize(m.astype(np.float32) / 255.0, (sw, sh), interpolation=cv2.INTER_AREA)
    # rim: edges facing the comet (up-left) catch a thin cool light; the far-left fringe a faint warm kiss
    d = max(int(round(1.8 * s)), 1)
    sh_ = np.zeros_like(a)
    sh_[d:, d:] = a[:-d, :-d]                        # alpha shifted down-right -> edges facing up-left
    rim = np.clip(a - sh_, 0, 1)
    xs = np.arange(sw, dtype=np.float32)[None, :]
    warm = _ss(0.55 * sw, 0.15 * sw, xs)
    col = np.array([0.01, 0.016, 0.045], np.float32)[None, None, :] * np.ones((sh, sw, 1), np.float32)
    rc = np.array([0.22, 0.4, 0.72], np.float32) * (1 - 0.5 * warm[..., None]) + \
        np.array([0.5, 0.3, 0.3], np.float32) * (0.5 * warm[..., None])
    col = col + rim[..., None] * rc * 0.6
    # near-lens softness (very slight)
    pm = cv2.GaussianBlur(col * a[..., None], (0, 0), 0.7 * s + 0.2)
    a = cv2.GaussianBlur(a, (0, 0), 0.7 * s + 0.2)
    return np.dstack([pm, a]).astype(np.float32), (cx, top)


def composite(img, spr, x_tip, y_tip, anchor):
    """Premultiplied sprite over img so its anchor lands at (x_tip, y_tip) (sub-pixel)."""
    H, W = img.shape[:2]
    sh, sw = spr.shape[:2]
    ox, oy = x_tip - anchor[0], y_tip - anchor[1]
    x0, y0 = int(math.floor(ox)), int(math.floor(oy))
    fx, fy = ox - x0, oy - y0
    M = np.float32([[1, 0, fx], [0, 1, fy]])
    sp = cv2.warpAffine(spr, M, (sw + 1, sh + 1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    xa, xb = max(x0, 0), min(x0 + sw + 1, W)
    ya, yb = max(y0, 0), min(y0 + sh + 1, H)
    if xb <= xa or yb <= ya:
        return img
    s_ = sp[ya - y0:yb - y0, xa - x0:xb - x0]
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - s_[..., 3:4]) + s_[..., :3]
    # the trunk continues below the sprite
    if y0 + sh < H:
        tb = s_[-1:, :, :]
        img[y0 + sh + 1:H, xa:xb] = img[y0 + sh + 1:H, xa:xb] * (1 - tb[..., 3:4]) + tb[..., :3]
    return img
