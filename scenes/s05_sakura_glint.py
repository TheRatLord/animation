"""s05_sakura round 8: coherent specular glitter band on the canal under the sun.

The sun's reflection path on rippled water is painted as ONE band: a soft warm sheen column below the sun
(widening toward the camera) filled with many short horizontal ripple highlights (1-2 px tall dashes, longer
near the camera).  Every dash twinkles on its own smooth slow cycle (continuous in t -> temporally coherent,
no per-frame noise) and drifts slightly with the current, so the band shimmers while its overall brightness
stays constant.  Outside the band only a handful of faint sky glints remain.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def draw_dashes(img, xs, ys, L, hgt, a, col):
    H, W = img.shape[0], img.shape[1]
    for i in range(xs.shape[0]):
        if a[i] < 0.004:
            continue
        cx, cy, ln, hh = xs[i], ys[i], L[i], hgt[i]
        x0 = max(0, int(cx - ln - 1))
        x1 = min(W, int(cx + ln + 2))
        y0 = max(0, int(cy - 2 * hh - 1))
        y1 = min(H, int(cy + 2 * hh + 2))
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / hh
            wy = math.exp(-dy * dy)
            if wy < 0.01:
                continue
            for x in range(x0, x1):
                dx = abs(x + 0.5 - cx) / ln
                if dx >= 1.0:
                    continue
                wx = (1.0 - dx * dx) ** 2
                k = a[i] * wx * wy
                img[y, x, 0] += col[0] * k
                img[y, x, 1] += col[1] * k
                img[y, x, 2] += col[2] * k


class Glitter:
    def __init__(self, W, H, w_x, w_y, w_Z, w_a, emx, sun_x, hy, seed=21, n=420):
        """w_* : water pixel lists of the far plate (plate coords, x offset by emx)."""
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        u = W / 1920.0
        dy = np.maximum(w_y - hy, 1.0)
        # band half-width grows toward the camera (perspective of the glitter path)
        sig = (0.008 * W + 0.07 * dy)
        gx = (w_x - (sun_x + emx)) / sig
        wgt = np.exp(-gx * gx) * w_a * np.clip((w_Z - 5.0) / 8.0, 0.1, 1.0)
        p = wgt / wgt.sum()
        idx = rng.choice(len(p), n, replace=True, p=p)
        self.x = w_x[idx].astype(np.float64) + rng.uniform(-0.5, 0.5, n)
        self.y = w_y[idx].astype(np.float64) + rng.uniform(-0.5, 0.5, n)
        self.px, self.py = w_x[idx], w_y[idx]
        z = w_Z[idx].astype(np.float64)
        near = np.clip(8.0 / z, 0.15, 1.0)
        self.L = u * (2.0 + 9.0 * near) * rng.uniform(0.6, 1.4, n)
        self.h = np.maximum(0.55, u * (0.5 + 0.6 * near)) * np.ones(n)
        self.inten = (0.7 + 1.1 * np.exp(-gx[idx] ** 2 * 1.5)) * rng.uniform(0.5, 1.1, n) * (0.6 + 0.4 * near)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.fr = rng.uniform(1.2, 2.6, n)
        self.dr = rng.uniform(-1.0, 1.0, n) * u * 3.0
        # soft sheen column (static, additive; follows the plate parallax through its own shift)
        self.sun_x, self.hy, self.emx = sun_x, hy, emx
        self.col = np.array([1.15, 0.98, 0.78])

    def sheen(self, water_a, shift_cols):
        """static additive warm sheen under the sun (plate res), returns (H, Wc, 3)"""
        H, Wc = water_a.shape
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(Wc, dtype=np.float32)[None, :]
        dy = np.maximum(ys - self.hy, 1.0)
        sig = 0.012 * self.W + 0.09 * dy
        g = np.exp(-((xs - (self.sun_x + self.emx)) / sig) ** 2)
        fall = np.clip(dy / (0.05 * self.H), 0, 1) * np.exp(-dy / (0.6 * self.H))
        k = (g * fall * water_a * 0.2).astype(np.float32)
        return k[..., None] * np.array([1.0, 0.86, 0.66], np.float32)

    def draw(self, img, t, shift):
        """shift: per-glint horizontal parallax shift (screen px)"""
        tw = np.clip(np.sin(t * self.fr + self.ph), 0.0, 1.0) ** 2
        a = self.inten * tw
        x = self.x - self.emx + shift + self.dr * t
        draw_dashes(img, x, self.y, self.L, self.h, a, self.col)
