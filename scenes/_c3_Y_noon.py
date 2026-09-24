"""clouds3_Y demo: towering summer cumulonimbus against a deep blue sky, sun upper right."""
import time
import numpy as np

from lib import core as C
from lib import sky as S
from lib import fx as F
from lib import clouds3_Y as K

DURATION = 6.0

SKY = dict(stops=[(0.0, '#123f8e'), (0.25, '#1d56a8'), (0.5, '#3474c2'), (0.72, '#5d9bd6'),
                  (0.88, '#94c4e8'), (1.0, '#c4e2f2')],
           sun_glow='#fff6e0', sun_glow_amt=0.22, below='#cfeaf6', band=('#e6f6fb', 0.5))


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        m = 0.08
        self.PW, self.PH = int(W * (1 + 2 * m)), int(H * (1 + 2 * m))
        ox, oy = W * m, H * m
        self.sun = (1.02 * W, -0.12 * H)                  # just off frame, upper right
        sun_p = (self.sun[0] + ox, self.sun[1] + oy)
        far_sun = (sun_p[0] + 3 * W, sun_p[1] - 2.2 * W)  # directional light for the clouds
        self.hz = 0.93
        self.sky = S.sky_gradient(W, H, SKY, horizon=self.hz, sun=self.sun, sun_radius=0.5, variation=0.015,
                                  seed=2, horizon_glow=0.8)
        PW, PH = self.PW, self.PH
        self.cirrus = K.cirrus_plate(PW, PH, far_sun, 'noon', seed=4, region=(0.0, 0.45), angle=-14,
                                     density=0.4, mackerel=0.55, opacity=0.85)
        self.strat = K.horizon_bank_plate(PW, PH, 0.875 * H + oy, 0.05 * H, far_sun, 0.2, 'noon', seed=12,
                                          rows=2, haze=0.12, value_bias=-0.12, flat=0.95, spread=(3.0, 7.0))
        self.bank = K.horizon_bank_plate(PW, PH, self.hz * H + oy + 0.01 * H, 0.045 * H, far_sun, 0.35, 'noon',
                                         seed=8, rows=3, haze=0.6)
        self.tower = K.cumulonimbus_plate(PW, PH, 0.33 * W + ox, 0.86 * H + oy, 0.95 * H, far_sun, sun_z=0.12,
                                          pal='noon', seed=21, width=0.36 * W, turrets=3, lean=0.12, anvil=0.0,
                                          cap_plateau=0.25, cap_fall=1.6, paint_r=3.5, band_soft=0.12, crease=0.4,
                                          cap_gain=0.6, sub_cap=0.6)
        self.heaps = K.cumulus_plate(PW, PH, [(0.8 * W + ox, 0.8 * H + oy, 0.28 * W, 0.2 * H, 5, 0.1),
                                              (0.95 * W + ox, 0.64 * H + oy, 0.12 * W, 0.08 * H, 6, 0.15)],
                                     far_sun, 0.3, 'noon', cap_plateau=0.25, cap_fall=1.6, paint_r=3.5, band_soft=0.12,
                                     crease=0.4)
        self.setup_s = time.time() - t0
        print(f'plates {self.setup_s:.1f}s')

    def frame(self, t):
        W, H = self.W, self.H
        p = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.03 * p
        cam = (0.03 * W * p, -0.01 * H * p)
        img = self.sky.copy()
        img = img + S.sun(W, H, self.sun[0], self.sun[1], radius=0.02, intensity=0.9, glow_size=1.0)
        L = [K.drift(self.cirrus, W, H, t, speed=0.002, cam=cam, zoom=zoom, depth=0.15),
             K.drift(self.bank, W, H, t, speed=0.0025, cam=cam, zoom=zoom, depth=0.3),
             K.drift(self.tower, W, H, t, speed=0.004, cam=cam, zoom=zoom, depth=0.5, billow=0.0012, seed=3),
             K.drift(self.strat, W, H, t, speed=0.0045, cam=cam, zoom=zoom, depth=0.52),
             K.drift(self.heaps, W, H, t, speed=0.007, cam=cam, zoom=zoom, depth=0.7, billow=0.0015, seed=4)]
        img = K.over(img, *L)
        img = img + K.halo(L[2], 0.3) + K.halo(L[4], 0.3)
        img = img + F.anime_flare(W, H, self.sun[0], self.sun[1], intensity=0.6, rays=6)
        img = F.bloom_soft(img, threshold=1.0, knee=0.25, strength=0.25, halation=0.1)
        img = F.shoulder(img, 0.92, 0.05)
        return F.finish_fast(img, t, sat=1.05, grain_amt=0.004, vig=0.22, ca=0.0006)
