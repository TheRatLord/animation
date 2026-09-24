"""Fused numba kernels for s10's per-frame compositing (keeps frame(t) well under budget at 1080p)."""
import numpy as np
from numba import njit, prange
import math


@njit(cache=True, fastmath=True, parallel=True)
def slice_over(img, rgb, al, Z, lo, hi, y0):
    """In place: img[y] = over(img, rgb) where the deck depth Z is in [lo, hi) (rows >= y0 only)."""
    H, W = Z.shape
    for r in prange(y0, H):
        for c in range(W):
            z = Z[r, c]
            if z >= lo and z < hi:
                a = al[r, c]
                if a > 0.0:
                    for k in range(3):
                        img[r, c, k] = img[r, c, k] * (1.0 - a) + rgb[r, c, k] * a


@njit(cache=True, fastmath=True, parallel=True)
def shadow_tint(rgb, shd, col, amt):
    """In place: pull the deck toward the cool shadow colour where the heads cast shadows."""
    H, W = shd.shape
    for r in prange(H):
        for c in range(W):
            s = shd[r, c] * amt
            if s > 0.001:
                m = (rgb[r, c, 0] + rgb[r, c, 1] + rgb[r, c, 2]) / 3.0
                f = 0.75 + 0.25 * m
                for k in range(3):
                    rgb[r, c, k] = rgb[r, c, k] * (1.0 - s) + col[k] * f * s


@njit(cache=True, fastmath=True, parallel=True)
def over_rows(img, L, y0, sea_a):
    """In place: img[y0:] = over(img[y0:], L) and sea_a[y0:] = max(sea_a, alpha)."""
    h, W = L.shape[0], L.shape[1]
    for i in prange(h):
        r = y0 + i
        for c in range(W):
            a = L[i, c, 3]
            if a <= 0.0:
                continue
            if a > 1.0:
                a = 1.0
            for k in range(3):
                img[r, c, k] = img[r, c, k] * (1.0 - a) + L[i, c, k] * a
            if a > sea_a[r, c]:
                sea_a[r, c] = a


@njit(cache=True, fastmath=True, parallel=True)
def grade_and_light(img, g, fg, shafts, sea_a, tw_a, add, add_k):
    """In place: foreground violet grade per row (g), light shafts masked by the clouds, plus an
    additive image (sun / flare) scaled by add_k."""
    H, W = sea_a.shape
    for r in prange(H):
        gr = g[r]
        for c in range(W):
            m = sea_a[r, c]
            t = 0.9 * tw_a[r, c]
            if t > m:
                m = t
            sm = 1.0 - 0.85 * m
            for k in range(3):
                v = img[r, c, k]
                if gr > 0.0:
                    v = v * (1.0 - gr) + fg[k] * (0.6 + 0.6 * v) * gr
                img[r, c, k] = v + shafts[r, c, k] * sm + add[r, c, k] * add_k


def bloom(img, threshold=1.0, knee=0.3, strength=0.2, halation=0.18, radii=(0.003, 0.01, 0.03, 0.08)):
    """Cheaper equivalent of lib.fx.bloom_soft: the bright pass is taken at half resolution and the
    bloom + warm halation are summed at half resolution before one upsample. Returns a new image."""
    import cv2
    from lib.fx import fast_blur
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    lum = small.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
    k = np.maximum(k * k, (lum > threshold + knee).astype(np.float32))
    br = (small * k).astype(np.float32)
    acc = np.zeros_like(br)
    wts = [1.0, 0.8, 0.6, 0.45][:len(radii)]
    for r, wt in zip(radii, wts):
        acc += fast_blur(br, r * W / 2) * wt
    acc *= strength / sum(wts)
    if halation:
        acc += fast_blur(br, 0.006 * W / 2) * (np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5)
    return img + cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


@njit(cache=True, fastmath=True, parallel=True)
def occ_raw(zc, zmin):
    """Contact-shadow seed: 1 where a nearer silhouette sits just below this (farther) billow."""
    H, W = zc.shape
    out = np.zeros((H, W), np.float32)
    for i in prange(H):
        for j in range(W):
            z = zc[i, j]
            if z <= 0.0:
                continue
            v = (z - zmin[i, j]) / z
            v = (v - 0.03) / 0.17
            if v <= 0.0:
                continue
            if v > 1.0:
                v = 1.0
            fz = 1.0 - (z - 40.0) / 80.0
            if fz > 1.0:
                fz = 1.0
            elif fz < 0.0:
                fz = 0.0
            out[i, j] = v * v * (3 - 2 * v) * fz
    return out


@njit(cache=True, fastmath=True, parallel=True)
def compose(img, rgb, al, occ, lit, tw_a, bird_a, bird_c, bm, shafts, hy, sx, light, kbeam, hband):
    """Sea over sky (+ contact shade), birds, horizon haze band, radiating beams, light shafts.
    Returns (img, occ_all)."""
    H, W = al.shape
    out = np.empty((H, W, 3), np.float32)
    occ_all = np.empty((H, W), np.float32)
    deep = (0.12, 0.1, 0.36)
    hcol = (1.0, 0.86, 0.62)
    bcol = (1.0, 0.8, 0.55)
    for i in prange(H):
        y = i + 0.5
        b1 = (y - hy + 0.004 * H) / (0.024 * H)
        b2 = (y - hy) / (0.07 * H)
        band = math.exp(-b1 * b1) + 0.35 * math.exp(-b2 * b2)
        for j in range(W):
            a = al[i, j]
            ba = bird_a[i, j]
            ta = tw_a[i, j]
            o = occ[i, j] * (1.0 - lit[i, j]) * 0.4
            wxx = (j + 0.5 - sx) / (0.3 * W)
            hb = band * (math.exp(-wxx * wxx) * 0.75 + 0.25) * hband
            oa = a if a > ta else ta
            occ_all[i, j] = oa
            bmk = bm[i, j] * (1.0 - 0.92 * oa) * kbeam
            sm = 1.0 - 0.95 * oa
            for c in range(3):
                s = rgb[i, j, c]
                s = s + (deep[c] - s) * o
                v = img[i, j, c]
                v = v + (bird_c[c] - v) * ba
                v = v * (1.0 - a) + s * a
                v += hcol[c] * hb + bcol[c] * bmk + shafts[i, j, c] * sm
                out[i, j, c] = v
    return out, occ_all
