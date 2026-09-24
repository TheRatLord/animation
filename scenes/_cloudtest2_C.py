"""_cloudtest2_C - technique C test 2: a sea of clouds seen from above at sunrise. The sun sits low over
the horizon; the billow tops are lit gold near the sun and pink away from it, the valleys between the
billows fall into deep indigo-violet shadow, and the sea flattens into a luminous haze toward the horizon.
Two towering cumulus rise out of the sea (the near one casts its long shadow toward the camera).
Motion: slow lateral glide with true per-row parallax (near billows slide faster than far ones).

Clouds: lib/clouds_C (heads in painter's order + heightfield single-scattering sun march + painterly
posterised paint-over; towers = 2.5D relief slab + light march + paint-over).
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import core as C, sky as S, fx as F, clouds_C as K  # noqa: E402

DURATION = 3.0
HORIZON = 0.42
SKY = dict(stops=[(0.0, '#18206a'), (0.25, '#2f3c96'), (0.5, '#6258b4'), (0.68, '#b074bc'),
                  (0.8, '#ee92aa'), (0.9, '#ffb088'), (1.0, '#ffcc88')],
           sun_glow='#ffb060', sun_glow_amt=0.2, below='#ffd0a0', band=('#ffd090', 0.3))
PAN = 0.05          # world units / s (lateral glide)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        s = H / 1080.0
        self.s = s
        self.sun = (0.585 * W, 0.372 * H)
        # towers (world X, Z, radius, height)
        self.towers = [dict(X=-5.4, Z=17.0, r=1.8, h=3.9, seed=21, wf=0.55),
                       dict(X=9.5, Z=29.0, r=1.6, h=3.6, seed=33, wf=0.45)]
        splits = [t['Z'] for t in self.towers]
        sea = K.sea_plates(W, H, horizon=HORIZON, seed=4, sun_px=self.sun, sun_elev=4.0, splits=splits,
                           towers=[(t['X'], t['Z'], t['r'] * 0.8, t['h'] * 0.6) for t in self.towers])
        self.f = sea['f']
        self.zrow = sea['zrow']
        self.sea = [K.premul(l) for l in sea['layers']]          # far .. near
        # tower plates
        self.tw = []
        for t in self.towers:
            x_base, y_base = sea['project'](t['X'], 0.15, t['Z'])
            _, y_top = sea['project'](t['X'], t['h'], t['Z'])
            th = y_base - y_top
            ph = int(th / 0.84)
            pw = int(ph * 1.25)
            sun_plate = (self.sun[0] - (x_base - pw * 0.46), self.sun[1] - (y_base - 0.9 * ph))
            keys = [(0.0, 0.5, 0.0), (0.15, 0.52, 0.0), (0.35, 0.42, 0.02), (0.55, 0.34, 0.04),
                    (0.75, 0.28, 0.05), (0.9, 0.22, 0.06), (1.0, 0.1, 0.06)]
            keys = [(k[0], k[1] * t['wf'] / 0.5, k[2]) for k in keys]
            rgba = K.tower_plate(pw, ph, seed=t['seed'], pal=K.PAL_TOWER_DAWN,
                                 sun_dir=(0.9 if self.sun[0] > x_base else -0.9, 0.12, 0.25),
                                 sun_px=sun_plate, base_row=0.9 * ph, top_row=0.06 * ph, cx0=0.46 * pw,
                                 keys=keys, anvil=False, scale=ph / 1000.0, tear=0.0, glow=0.6, backlit=0.06,
                                 haze=0.3, haze_top=0.55, bands=(0.55,), mid_w=0.035,
                                 head_dir=(0.55 if self.sun[0] > x_base else -0.55, 0.85), sil_detail=0.35)
            # aerial perspective toward the dawn haze
            fog = 1 - math.exp(-t['Z'] / 45.0 / 0.45)
            rgba[..., :3] = K._lerp(rgba[..., :3], K._hex('#f6c2b4'), 0.35 * fog)
            self.tw.append(dict(pm=K.premul(rgba), x=x_base - pw * 0.46, y=y_base - 0.9 * ph, Z=t['Z']))
        sky = S.sky_gradient(W, H, SKY, horizon=HORIZON, sun=self.sun, sun_radius=0.45, seed=3)
        self.sky = sky
        # per-row parallax gain (px per world unit of camera travel)
        self.gain = (self.f / np.maximum(self.zrow, 0.5)).astype(np.float32)
        self.xs, self.ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    def _sea_layer(self, pm, travel):
        dx = (travel * self.gain)[:, None]
        mx = self.xs + dx
        return cv2.remap(pm, mx, self.ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def frame(self, t):
        W, H, s = self.W, self.H, self.s
        u = C.ease_in_out_sine(t / DURATION)
        travel = PAN * DURATION * (u - 0.5)
        img = self.sky.copy()
        img += S.sun(W, H, self.sun[0], self.sun[1], radius=0.01, color=(1.0, 0.62, 0.3), intensity=0.6,
                     glow_size=0.7)
        occ = np.zeros((H, W), np.float32)
        layers = self.sea
        for i, pm in enumerate(layers):
            L = self._sea_layer(pm, travel)
            a = L[..., 3:4]
            img = img * (1 - a) + L[..., :3]
            occ = np.maximum(occ, a[..., 0])
            # tower i sits between sea layer i (farther) and i+1 (nearer); towers are sorted near -> far
            ti = len(self.tw) - 1 - i
            if 0 <= ti < len(self.tw) and i < len(layers) - 1:
                tw = self.tw[ti]
                a_t = K.place(img, tw['pm'], tw['x'] + travel * self.f / tw['Z'], tw['y'])
                occ = np.maximum(occ, a_t)
        vis = F.sun_visibility(occ, self.sun[0], self.sun[1], 0.013 * W)
        img += F.light_shafts(W, H, self.sun[0], self.sun[1], occluder=occ, strength=0.18, length=0.6, t=t,
                              tint=(1.0, 0.8, 0.55), radius=0.08)
        img += F.anime_flare(W, H, self.sun[0], self.sun[1], intensity=0.15 + 0.3 * vis, tint=(1.0, 0.7, 0.45),
                             rays=6, ray_len=0.04, ghosts=0.5, streak=0.3, rot=0.01 * t, glow=0.6)
        img = F.bloom_soft(img, threshold=1.0, knee=0.15, strength=0.3, halation=0.15)
        return F.finish_fast(img, t, sat=1.05, grain_amt=0.004, vig=0.22, ca=0.0008)
