"""Cloud test A-1: a big summer cumulonimbus tower with its anvil against a deep blue sky, the sun at
upper right partly hidden behind the anvil's edge. Technique A (distance-field painter, lib/clouds_A).
Slow drift + multi-plane parallax, gentle tilt-up (3 s)."""
import math
import time
import numpy as np
import cv2
from lib import core as C, sky as SK, fx as F, clouds_A as CA

DURATION = 3.0


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        pw, ph = int(W * 1.12), int(H * 1.12)
        self.pw, self.ph = pw, ph
        ox, oy = (pw - W) / 2, (ph - H) / 2
        self.ox, self.oy = ox, oy
        hz = 0.965
        self.sun_p = (0.672 * pw, 0.232 * ph)
        hzp = (hz * H + oy) / ph
        self.sky = SK.sky_gradient(pw, ph, 'summer_noon', horizon=hzp, sun=self.sun_p, sun_radius=0.4)
        L = (0.6, -0.7, 0.45)
        # hero cumulonimbus
        cb = CA.cumulonimbus_groups(cx=0.5 * pw, base_y=1.02 * ph, width=0.42 * pw, height=0.86 * ph, seed=3,
                                    wind=-1, lean=-0.06)
        self.hero = CA.render_cloud(pw, ph, cb, light=L, pal='summer', sun=self.sun_p, seed=3, sky=self.sky,
                                    backlit=0.25, terminator=-0.06)
        # distant cumulus along the horizon (hazier, smaller)
        far = []
        for i, (x, w, h) in enumerate([(0.08, 0.16, 0.12), (0.9, 0.2, 0.16), (0.22, 0.1, 0.07), (0.72, 0.09, 0.06)]):
            far.append(CA.cumulus_groups(x * pw, 0.975 * ph, w * pw, h * ph, seed=20 + i, towers=3, haze=0.35))
        self.far = CA.render_clouds(pw, ph, far, light=L, pal='summer', sun=self.sun_p, seed=9, sky=self.sky,
                                   rim=0.7)
        print('plates %.1fs' % (time.time() - t0))

    def _shift(self, plate, dx, dy, zoom=1.0):
        W, H = self.W, self.H
        M = np.array([[zoom, 0, W / 2 - zoom * (self.pw / 2 + dx)], [0, zoom, H / 2 - zoom * (self.ph / 2 + dy)]],
                     np.float32)
        return cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
                              if plate.shape[2] == 3 else cv2.BORDER_CONSTANT)

    def frame(self, t):
        W, H = self.W, self.H
        u = C.ease_in_out_sine(t / DURATION)
        tilt = (0.5 - u) * 0.035 * H          # slow tilt up
        z = 1.0 + 0.02 * u
        sky = self._shift(self.sky, 0, tilt * 0.3, 1 + (z - 1) * 0.3)
        # sun lives on the sky plane
        sx = (self.sun_p[0] - self.pw / 2) * (1 + (z - 1) * 0.3) + W / 2
        sy = (self.sun_p[1] - self.ph / 2 - tilt * 0.3) * (1 + (z - 1) * 0.3) + H / 2
        img = sky + SK.sun(W, H, sx, sy, radius=0.012, color=(1.0, 0.95, 0.85))
        far = self._shift(self.far, 0.004 * W * t, tilt * 0.55, 1 + (z - 1) * 0.55)
        hero = self._shift(self.hero, 0.0045 * W * t, tilt * 0.7, 1 + (z - 1) * 0.7)
        img = F.over_rgba(img, far)
        img = F.over_rgba(img, hero)
        occ = hero[..., 3]
        vis = F.sun_visibility(occ, sx, sy, 0.012 * W)
        img = img + F.light_shafts(W, H, sx, sy, occluder=occ, strength=0.22, length=0.9, radius=0.08,
                                   tint=(1.0, 0.95, 0.85), t=t)
        img = F.bloom_soft(img, threshold=1.0, knee=0.25, strength=0.12, halation=0.04)
        img = img + F.anime_flare(W, H, sx, sy, intensity=0.2 + 0.7 * vis, rot=0.02 * t, tint=(1.0, 0.95, 0.86))
        img = F.shoulder(img, 0.82)
        return F.finish_fast(img, t, sat=1.08, grain_amt=0.003, vig=0.18, ca=0.0005)
