"""s06_seaside: overhead wires drawn per frame as clean catenaries (2x supersampled, anti-aliased).

The wires are stored as 3D polylines (plate px + depth); each frame they are displaced by the camera
truck point by point (x += pan + K / Z), so every span stays a smooth sagging curve between its two
crossarm attachments whatever the parallax - no per-pixel warp, no kinks. A thin warm sheen rides on the
sun-facing underside, strongest near the sun."""
import numpy as np
import cv2

SS = 2


def draw(img, wires, pan, K, ox, oy, f, u, color=(0.17, 0.155, 0.28), sheen=(1.7, 1.12, 0.66), D=0.0,
         c=(0.0, 0.0), V=(0.0, 0.0)):
    H, W = img.shape[:2]
    segs = []
    ylo, yhi = H, 0
    for (px, py, zz, kk) in wires:
        s = 1.0 / np.maximum(1.0 - D / zz, 0.2)
        x = c[0] + (px - ox - c[0] + K / zz) * s + pan
        y = c[1] + (py - oy - c[1]) * s + V[0] + V[1] / zz
        vis = (x > -60) & (x < W + 60) & (y > -60) & (y < H + 60)
        if vis.sum() < 2:
            continue
        idx = np.nonzero(vis)[0]
        a0, a1 = max(idx[0] - 1, 0), min(idx[-1] + 2, len(x))
        x, y, zz_, kk_ = x[a0:a1], y[a0:a1], zz[a0:a1], kk[a0:a1]
        wd = np.maximum(f * 0.02 / zz_ * s[a0:a1], 0.6 * u)
        segs.append((x, y, wd, kk_))
        ylo = min(ylo, float(y.min() - wd.max() - 2))
        yhi = max(yhi, float(y.max() + wd.max() + 2))
    if not segs:
        return img
    r0, r1 = int(max(ylo, 0)), int(min(yhi + 1, H))
    if r1 <= r0:
        return img
    hh = r1 - r0
    dark = np.zeros((hh * SS, W * SS), np.uint8)
    lite = np.zeros((hh * SS, W * SS), np.uint8)
    for (x, y, wd, kk) in segs:
        n = len(x)
        # chunk length adapts to how fast the width changes (near the camera the width grows quickly)
        step = 6 if wd.max() > 3 * wd.min() else 30
        P = np.stack([x * SS, (y - r0) * SS], 1)
        for a in range(0, n - 1, step):
            b = min(a + step + 1, n)
            w_ = float(wd[a] + wd[b - 1]) * 0.5 * SS
            th = max(int(round(w_)), 1)
            val = int(np.clip(255 * min(w_ / th, 1.0), 30, 255))
            pts = P[a:b]
            cv2.polylines(dark, [np.round(pts * 16).astype(np.int32)], False, val, th, cv2.LINE_AA, shift=4)
            k = float(kk[a] + kk[b - 1]) * 0.5
            if k > 0.08:
                ts = max(int(round(w_ * 0.45)), 1)
                off = pts + [0, w_ * 0.12]
                cv2.polylines(lite, [np.round(off * 16).astype(np.int32)], False, int(255 * k * min(w_ * 0.45 / ts, 1.0)),
                              ts, cv2.LINE_AA, shift=4)
    m = cv2.resize(dark, (W, hh), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255.0)
    ml = cv2.resize(lite, (W, hh), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255.0) * m
    c = np.asarray(color, np.float32)
    reg = img[r0:r1]
    reg *= (1 - m)[..., None]
    reg += c * m[..., None]
    reg += (np.asarray(sheen, np.float32) - c) * (ml * 0.6)[..., None]
    return img
