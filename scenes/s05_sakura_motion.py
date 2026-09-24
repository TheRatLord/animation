"""s05_sakura round 6: secondary motion that makes the multi-plane depth read at the slower camera speed.

* canopy sway: every near tree card is composited with a per-row horizontal shift = camera parallax + a gentle
  wind sway that is zero at the trunk foot and grows toward the crown (smooth sinusoids, per-tree phase), so
  the crowns rock slowly against the far bank / sky while the feet stay planted;
* drifting dapple: the sun flecks that fall through the canopy onto the path slide and breathe with the
  swaying crowns (the static fleck map is re-sampled through a smooth, slowly animated displacement field and
  the ground is re-lit by the exact sun/shade ratio -> no popping, no flicker).
"""
import math
import numpy as np
import cv2
from numba import njit, prange


@njit(cache=True, parallel=True, fastmath=True)
def over_rows_acc(dst, acc, src, ox, oy, shift, sway, foot, top):
    """dst (H,W,3) <- straight RGBA src placed at (ox + shift + sway * w(row), oy), bilinear in x;
    w = 0 at/below the foot row, rising (smooth, ^1.6) to 1 at the top row.  Coverage into acc."""
    H, W = dst.shape[0], dst.shape[1]
    h, w = src.shape[0], src.shape[1]
    y0 = max(0, oy)
    y1 = min(H, oy + h)
    span = max(foot - top, 1.0)
    for y in prange(y0, y1):
        sy = y - oy
        q = (foot - sy) / span
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        fx0 = ox + shift + sway * q ** 1.6
        x0 = max(0, int(math.floor(fx0)))
        x1 = min(W, int(math.ceil(fx0 + w)))
        for x in range(x0, x1):
            sx = x - fx0
            ix = int(math.floor(sx))
            fr = sx - ix
            if ix < -1 or ix >= w:
                continue
            a0 = src[sy, ix, 3] if ix >= 0 else 0.0
            a1 = src[sy, ix + 1, 3] if ix + 1 < w else 0.0
            a = a0 * (1 - fr) + a1 * fr
            if a <= 1e-4:
                continue
            acc[y, x] = acc[y, x] * (1 - a) + a
            for c in range(3):
                v0 = src[sy, ix, c] * a0 if ix >= 0 else 0.0
                v1 = src[sy, ix + 1, c] * a1 if ix + 1 < w else 0.0
                dst[y, x, c] = dst[y, x, c] * (1 - a) + v0 * (1 - fr) + v1 * fr


def sway_px(t, z, W, phase):
    """horizontal crown-top sway in px (1/Z falloff, two incommensurate slow sinusoids)"""
    A = (W / 1920.0) * 70.0 / z
    return A * (math.sin(0.55 * t + phase) + 0.35 * math.sin(1.27 * t + 2.1 * phase + 0.7))


class Dapple:
    """Animated sun-fleck dapple on the ground plate."""

    def __init__(self, spot, hy, H, W, emx, ratio=(2.3, 1.95, 1.14), q=2, seed=5):
        self.q = q
        self.H, self.W, self.emx = H, W, emx
        Hq, Wq = H // q, spot.shape[1] // q
        self.spot = cv2.resize(spot.astype(np.float32), (Wq, Hq), interpolation=cv2.INTER_AREA)
        self.r0 = int(hy) + 2
        self.ratio = np.array(ratio, np.float32) - 1.0
        rng = np.random.default_rng(seed)
        h, w = (H - self.r0) // q + 1, W // q + 1
        self.h, self.w = h, w

        def smooth(cell):
            g = rng.standard_normal((int(h / cell) + 3, int(w / cell) + 3)).astype(np.float32)
            return cv2.resize(g, (int(g.shape[1] * cell), int(g.shape[0] * cell)),
                              interpolation=cv2.INTER_CUBIC)[:h, :w]
        c = 26.0 * (W / 1920.0) / q
        self.n = [smooth(c), smooth(c), smooth(c * 0.6), smooth(c * 0.6)]
        yy = (np.arange(h, dtype=np.float32) * q + self.r0 - hy) / max(H - hy, 1.0)
        self.persp = np.clip(yy, 0.02, 1.0)[:, None]        # nearer ground -> bigger displacement
        self.ys = (np.arange(h, dtype=np.float32) * q + self.r0) / q
        self.xs = np.arange(w, dtype=np.float32)
        base = self.spot
        self._den = None

    def apply(self, img, t, shifts):
        q = self.q
        amp = 7.0 * (self.W / 1920.0) / q * self.persp
        dx = amp * (self.n[0] * math.sin(0.62 * t + 0.3) + 0.6 * self.n[2] * math.sin(1.13 * t + 1.9))
        dy = 0.5 * amp * (self.n[1] * math.sin(0.51 * t + 2.2) + 0.6 * self.n[3] * math.sin(0.97 * t + 0.4))
        rows = (np.arange(self.h) * q + self.r0).clip(0, self.H - 1)
        sh = (shifts[rows] / q).astype(np.float32)[:, None]
        mx0 = (self.xs[None, :] + (self.emx) / q - sh).astype(np.float32)
        my0 = np.broadcast_to(self.ys[:, None], mx0.shape).astype(np.float32)
        s0 = cv2.remap(self.spot, mx0, my0, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        s1 = cv2.remap(self.spot, mx0 + dx.astype(np.float32), my0 + dy.astype(np.float32), cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT)
        s0 = np.clip(s0, 0, 1)[..., None]
        s1 = np.clip(s1, 0, 1)[..., None]
        mult = (1.0 + self.ratio * s1) / (1.0 + self.ratio * s0)
        mult = cv2.resize(mult.astype(np.float32), (self.W, (self.h) * q), interpolation=cv2.INTER_LINEAR)
        n = self.H - self.r0
        img[self.r0:] *= mult[:n]
        return img
