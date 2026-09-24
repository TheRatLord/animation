"""Dust motes for s09_classroom: particles drifting in the air, glittering only inside the sunbeams."""
import math
import numpy as np
from numba import njit, prange

import s09_classroom_rt as R


def make_dust(n, seed=11):
    rng = np.random.default_rng(seed)
    P = np.zeros((n, 10), np.float64)
    P[:, 0] = rng.uniform(0.8, 8.5, n)
    P[:, 1] = rng.uniform(0.3, 2.8, n)
    P[:, 2] = rng.uniform(0.3, 6.2, n)
    P[:, 3:6] = rng.normal(0, 1, (n, 3)) * np.array([0.012, 0.006, 0.012])   # slow drift m/s
    P[:, 3] -= 0.01                                                            # gentle breeze
    P[:, 4] += 0.004                                                           # warm air rising
    P[:, 6:9] = rng.uniform(0, 6.28, (n, 3))                                   # wobble phases
    P[:, 9] = rng.uniform(0.3, 1.0, n) ** 2                                    # brightness
    return P


@njit(fastmath=True, cache=True)
def dust(img, depth, cam, fpx, P, t, L, VB, HB, CM, cm_ppm, cm_y0, sunc, k):
    H, W = img.shape[0], img.shape[1]
    for i in range(P.shape[0]):
        px = P[i, 0] + P[i, 3] * t + 0.03 * math.sin(t * 0.7 + P[i, 6])
        py = P[i, 1] + P[i, 4] * t + 0.02 * math.sin(t * 0.9 + P[i, 7])
        pz = P[i, 2] + P[i, 5] * t + 0.03 * math.sin(t * 0.6 + P[i, 8])
        if pz <= 0.05:
            continue
        tw = -pz / L[2]
        xw = px + L[0] * tw
        yw = py + L[1] * tw
        a = R.aperture(xw, yw, 0.004 + tw * 0.012, VB, HB)
        if a <= 0.01:
            continue
        a *= R.curtain_T(xw, yw, CM, cm_ppm, cm_y0)
        rx, ry, rz = px - cam[0], py - cam[1], pz - cam[2]
        zc = rx * cam[3] + ry * cam[4] + rz * cam[5]
        if zc < 0.3:
            continue
        sx = W * 0.5 + (rx * cam[6] + ry * cam[7] + rz * cam[8]) / zc * fpx
        sy = H * 0.5 - (rx * cam[9] + ry * cam[10] + rz * cam[11]) / zc * fpx
        if sx < -20 or sy < -20 or sx > W + 20 or sy > H + 20:
            continue
        dist = math.sqrt(rx * rx + ry * ry + rz * rz)
        # size: a tiny point far away, a soft out-of-focus disc when close to the lens
        rad = max(0.6, 0.0011 * W * (1.0 / zc) * 1.6)
        blurr = abs(zc - 3.5) * 0.0009 * W
        rad = max(rad, blurr)
        inten = a * P[i, 9] * k * min(1.0, 1.2 / (rad * rad) * 1.0 + 0.0) * 1.4
        if rad > 2.0:
            inten = a * P[i, 9] * k * 0.55 * (2.0 / rad) ** 1.2
        tw_ = 0.6 + 0.4 * math.sin(t * 3.0 + P[i, 6] * 5.0)        # glitter as they tumble
        inten *= tw_
        R_ = int(rad * 2.0) + 2
        x0, y0 = int(sx), int(sy)
        for yy in range(max(0, y0 - R_), min(H, y0 + R_ + 1)):
            for xx in range(max(0, x0 - R_), min(W, x0 + R_ + 1)):
                if depth[yy, xx] < dist - 0.05:
                    continue
                d2 = ((xx + 0.5 - sx) ** 2 + (yy + 0.5 - sy) ** 2) / (rad * rad)
                if rad > 2.0:
                    w = min(1.0, max(0.0, (1.15 - math.sqrt(d2)) * 4.0)) * (0.75 + 0.25 * d2)
                else:
                    w = math.exp(-d2 * 1.5)
                if w <= 0:
                    continue
                img[yy, xx, 0] += w * inten * sunc[0]
                img[yy, xx, 1] += w * inten * sunc[1]
                img[yy, xx, 2] += w * inten * sunc[2]


@njit(parallel=True, fastmath=True, cache=True)
def edges(col, thr, mask):
    """mark pixels whose log-luminance differs from a neighbour by > thr (dilated by 1 px)."""
    H, W = col.shape[0], col.shape[1]
    for y in prange(H):
        for x in range(W):
            mask[y, x] = 0
    for y in prange(H):
        for x in range(W):
            l0 = math.log((col[y, x, 0] + col[y, x, 1] + col[y, x, 2]) / 3 + 0.02)
            if x + 1 < W:
                l1 = math.log((col[y, x + 1, 0] + col[y, x + 1, 1] + col[y, x + 1, 2]) / 3 + 0.02)
                if abs(l1 - l0) > thr:
                    mask[y, x] = 1
                    mask[y, x + 1] = 1
            if y + 1 < H:
                l1 = math.log((col[y + 1, x, 0] + col[y + 1, x, 1] + col[y + 1, x, 2]) / 3 + 0.02)
                if abs(l1 - l0) > thr:
                    mask[y, x] = 1
                    mask[y + 1, x] = 1


@njit(parallel=True, fastmath=True, cache=True)
def combine1(col, reflb, rwb, rw, gi, alb, out):
    H, W = col.shape[0], col.shape[1]
    for y in prange(H):
        for x in range(W):
            w = rw[y, x]
            d = max(rwb[y, x], 1e-4)
            for c in range(3):
                v = col[y, x, c] + gi[y, x, c] * max(alb[y, x, c], 0.0)
                if w > 0:
                    v += reflb[y, x, c] / d * w
                out[y, x, c] = v


@njit(parallel=True, fastmath=True, cache=True)
def combine2(img, vol, fl, fk, vk, vt, grade):
    """+ volumetric shafts (protected over bright pixels) + flare + colour grade (cool shadows)."""
    H, W = img.shape[0], img.shape[1]
    for y in prange(H):
        for x in range(W):
            r, g, b = img[y, x, 0], img[y, x, 1], img[y, x, 2]
            l = (r + g + b) / 3
            q = min(1.0, max(0.0, (l - 0.3) / 0.6))
            p = 1.0 - 0.93 * q * q * (3 - 2 * q)
            v = max(vol[y, x], 0.0) * p * vk
            r += v * vt[0] + fl[y, x, 0] * fk
            g += v * vt[1] + fl[y, x, 1] * fk
            b += v * vt[2] + fl[y, x, 2] * fk
            # grade: shadows lean teal-blue, mids/highlights warm
            l = 0.2126 * r + 0.7152 * g + 0.0722 * b
            s = max(0.0, 1.0 - l / 0.45)
            s = s * s
            r = r * (1 - grade * 0.10 * s)
            g = g * (1 + grade * 0.02 * s)
            b = b * (1 + grade * 0.14 * s) + grade * 0.012 * s
            h = min(1.0, max(0.0, (l - 0.45) / 0.6))
            r = r * (1 + grade * 0.05 * h)
            b = b * (1 - grade * 0.05 * h)
            # gentle luminance contrast around a mid pivot (deeper shadows, brighter lights)
            l = max(0.2126 * r + 0.7152 * g + 0.0722 * b, 1e-4)
            l2 = 0.4 * (l / 0.4) ** (1.0 + 0.14 * grade)
            m = l2 / l
            img[y, x, 0] = r * m
            img[y, x, 1] = g * m
            img[y, x, 2] = b * m


def bloom_half(img, threshold=0.9, knee=0.4, strength=0.35, halation=0.2, radii=(0.003, 0.01, 0.03, 0.08)):
    """Same look as lib.fx.bloom_soft but the bright-pass is extracted at half resolution (2x faster)."""
    import cv2
    from lib import fx as F
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    lum = small.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
    k = k * k
    br = small * np.maximum(k, (lum > threshold + knee))
    acc = np.zeros_like(br)
    wts = [1.0, 0.8, 0.6, 0.45]
    for r, wt in zip(radii, wts):
        acc += F.fast_blur(br, r * W / 2) * wt
    acc *= strength / sum(wts)
    if halation:
        acc += F.fast_blur(br, 0.006 * W / 2) * (np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5)
    return img + cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def starburst(W, H, lx, ly, seed=21, n_long=8, n_short=16, tint=(1.0, 0.9, 0.75)):
    """Crisp diffraction starburst around the sun: long hard spikes + many short fine ones and a white
    hot core (HDR additive, (H, W, 3)). Rendered once and translated each frame."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32) + 0.5, np.arange(H, dtype=np.float32) + 0.5)
    dx, dy = xs - lx, ys - ly
    d = np.sqrt(dx * dx + dy * dy) + 1e-3
    dW = d / W
    ang = np.arctan2(dy, dx)
    out = np.zeros((H, W), np.float32)
    base = rng.uniform(0, math.pi)
    for i in range(n_long):
        a0 = base + i * 2 * math.pi / n_long + rng.normal(0, 0.03)
        da = np.abs(np.angle(np.exp(1j * (ang - a0)))).astype(np.float32)
        wd = (0.9 / W) / np.maximum(dW, 1e-4) * 1.6 + 0.004     # ~constant pixel width, tapering
        ln = 0.16 * rng.uniform(0.6, 1.0)
        out += np.exp(-(da / wd) ** 2) * np.exp(-dW / (ln * 0.35)) * rng.uniform(0.6, 1.0) * 1.2
    for i in range(n_short):
        a0 = rng.uniform(-math.pi, math.pi)
        da = np.abs(np.angle(np.exp(1j * (ang - a0)))).astype(np.float32)
        wd = (0.7 / W) / np.maximum(dW, 1e-4) + 0.002
        ln = 0.05 * rng.uniform(0.4, 1.0)
        out += np.exp(-(da / wd) ** 2) * np.exp(-dW / (ln * 0.4)) * rng.uniform(0.3, 0.7)
    out *= np.clip(dW / 0.004, 0, 1) ** 0.5
    core = 1.0 / (1.0 + (dW / 0.0045) ** 3) * 3.0 + np.exp(-dW / 0.012) * 0.6
    t = np.asarray(tint, np.float32)
    img = out[..., None] * (0.55 + 0.45 * t) + core[..., None] * np.array([1.0, 0.97, 0.9], np.float32)
    return img.astype(np.float32)


def hex_ghosts(W, H, lx, ly, seed=5):
    """Crisp hexagonal aperture ghosts on the line from the sun through the frame centre (additive)."""
    import cv2
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W, 3), np.float32)
    cx, cy = W / 2, H / 2
    vx, vy = cx - lx, cy - ly
    spec = [(0.45, 0.016, (0.5, 0.9, 1.0), 0.10), (0.78, 0.034, (0.6, 1.0, 0.7), 0.07),
            (1.25, 0.022, (1.0, 0.7, 0.45), 0.10), (1.6, 0.06, (0.55, 0.65, 1.0), 0.05),
            (1.95, 0.03, (1.0, 0.55, 0.85), 0.08), (2.35, 0.012, (0.8, 1.0, 0.9), 0.12)]
    ss = 2
    for k, r, col, amt in spec:
        gx, gy = lx + vx * k, ly + vy * k
        R = r * W
        m = np.zeros((H * ss // 4, W * ss // 4), np.float32)   # quarter-res x ss canvas
        sc = ss / 4
        pts = [(int((gx + R * math.cos(a + 0.3)) * sc * 8), int((gy + R * math.sin(a + 0.3)) * sc * 8))
               for a in np.linspace(0, 2 * math.pi, 7)[:-1]]
        cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA, shift=3)
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
        m = cv2.GaussianBlur(m, (0, 0), max(0.6, R * 0.03))
        # brighter chromatic rim, fainter body
        rim = np.clip(m - cv2.GaussianBlur(m, (0, 0), max(1.0, R * 0.12)) * 1.0, 0, 1)
        out += (m * 0.8 + rim * 2.4)[..., None] * np.asarray(col, np.float32) * amt
    return out
