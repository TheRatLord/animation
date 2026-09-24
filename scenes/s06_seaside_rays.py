"""s06_seaside: painted god-ray wedges (cm5_05 style).

A handful of broad, soft-edged light bands radiating from the low sun: each is a wedge in polar
coordinates about the sun with feathered sides, a faint streaky inner texture (static, so no flicker),
and a radial envelope (starts a little way off the sun, peaks, fades out across the sky). Laid additively
over sky + clouds, slightly weaker over the cloud bodies, faded out below the horizon."""
import math
import numpy as np
import cv2

from lib import core as C

# (angle deg: screen space, 0 = right, -90 = straight up; half width deg; strength; reach (x W))
BEAMS = [(-170.0, 3.4, 0.8, 1.0), (-153.0, 5.5, 1.0, 1.05), (-128.0, 3.6, 0.6, 0.75),
         (-44.0, 4.6, 0.8, 0.75), (-19.0, 3.0, 0.65, 0.6)]


class RayFan:
    def __init__(self, W, H, seed=5, q=3, strength=0.16, tint=(1.0, 0.76, 0.5)):
        self.W, self.H, self.q = W, H, q
        self.w, self.h = max(W // q, 8), max(H // q, 8)
        rng = np.random.default_rng(seed)
        # static angular streak texture (per beam): 1D noise over angle
        n = 2048
        tex = np.cumsum(rng.standard_normal(n)).astype(np.float32)
        tex = cv2.GaussianBlur(tex[None], (0, 0), sigmaX=6)[0]
        tex = tex - cv2.GaussianBlur(tex[None], (0, 0), sigmaX=60)[0]
        self.tex = (tex / (np.abs(tex).max() + 1e-6)).astype(np.float32)
        self.ph = rng.uniform(0, 6.28, len(BEAMS))
        self.strength = strength
        self.tint = np.asarray(tint, np.float32)
        self.xs, self.ys = C.grid(self.w, self.h)

    def render(self, sx, sy, hz, t, occ=None):
        q, w, h = self.q, self.w, self.h
        dx = self.xs - sx / q
        dy = self.ys - sy / q
        r = np.sqrt(dx * dx + dy * dy) * q / self.W            # distance in frame widths
        th = np.degrees(np.arctan2(dy, dx))
        acc = np.zeros((h, w), np.float32)
        n = len(self.tex)
        for i, (a, hw, st, reach) in enumerate(BEAMS):
            d = (th - a + 180.0) % 360.0 - 180.0
            # feathered wedge: soft sides, slightly harder inner core
            prof = np.exp(-(d / hw) ** 4) * 0.7 + np.exp(-(d / (hw * 1.9)) ** 2) * 0.3
            idx = ((d / (hw * 3.0) + 0.5) * 300 + i * 311).astype(np.int32) % n
            prof = prof * (0.82 + 0.18 * self.tex[idx])
            env = C.smoothstep(0.02, 0.14, r) * np.exp(-np.maximum(r - 0.12, 0) / (0.45 * reach))
            env = env * C.smoothstep(reach * 1.25, reach * 0.55, r)
            br = 0.88 + 0.12 * math.sin(t * 0.55 + self.ph[i])
            acc += prof * env * st * br
        # fade below the horizon (a little still reaches the sea haze)
        acc *= 0.25 + 0.75 * C.smoothstep(hz / q + 6, hz / q - 10, self.ys)
        if occ is not None:
            o = cv2.resize(occ.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            acc *= 1 - 0.45 * np.clip(o, 0, 1)
        out = cv2.resize(acc, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return out[..., None] * (self.tint * self.strength)
