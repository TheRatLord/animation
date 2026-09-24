"""clouds3_Y demo: sea of clouds at sunrise seen from above - towers, deep indigo valleys, gold/pink lit tops."""
import time
import numpy as np

from lib import core as C
from lib import sky as S
from lib import fx as F
from lib import clouds3_Y as K

DURATION = 6.0

SKY = dict(stops=[(0.0, '#1d3580'), (0.3, '#2f55a6'), (0.52, '#5f78b8'), (0.7, '#a888b4'),
                  (0.83, '#ec9e92'), (0.93, '#ffbd7c'), (1.0, '#ffd898')],
           sun_glow='#ffd08a', sun_glow_amt=0.25, below='#f4b89c', band=('#ffe2b0', 0.35))


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        m = 0.07
        self.PW, self.PH = PW, PH = int(W * (1 + 2 * m)), int(H * (1 + 2 * m))
        ox, oy = W * m, H * m
        self.hz = 0.36
        self.sun = (0.6 * W, 0.33 * H)
        sun_p = (self.sun[0] + ox, self.sun[1] + oy)
        hy = self.hz * H + oy
        self.sky = S.sky_gradient(W, H, SKY, horizon=self.hz, sun=self.sun, sun_radius=0.3, variation=0.012,
                                  seed=5, horizon_glow=0.8)
        air = K._c('#e2ae96')
        # sea split at the towers' depths: far | tower A | mid | tower B | near
        self.sea = K.sea_of_clouds_plate(PW, PH, hy, sun_p, sun_z=-0.35, pal='sunrise', seed=3, splits=(14.0, 30.0),
                                         cam_h=2.5, relief=0.8, air=air, haze_far=0.9,
                                         plate_kw=[{}, dict(paint_r=9), dict(paint_r=18, paint_aniso=4.0, value_bias=-0.12, crest=0.55, brush_angle=-25)])
        self.towers = [
            K.sea_tower_plate(PW, PH, hy, 0.8, 30.0, 5.2, sun_p, sun_z=-0.1, pal='sunrise', seed=31, haze=0.35,
                              width=1.0, turrets=5, anvil=0.4, air=air, crest=0.9, crest_drop=0.25, cap=0.16, body=0.12, form_w=0.45, crease=0.35, sub_cap=0.5, scallop=0.5, paint_r=4, cap_plateau=0.55, cap_fall=0.8, band_soft=0.08, cap_gain=0.75),
            K.sea_tower_plate(PW, PH, hy, 0.18, 14.0, 5.0, sun_p, sun_z=-0.05, pal='sunrise', seed=33, haze=0.1,
                              width=1.05, turrets=5, lean=0.05, air=air, crest=0.9, crest_drop=0.25, cap=0.16, body=0.12, form_w=0.45, crease=0.35, sub_cap=0.5, scallop=0.5, paint_r=4, cap_plateau=0.55, cap_fall=0.8, band_soft=0.08, cap_gain=0.75),
        ]
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        self.fg_shade = (1 - 0.32 * np.clip((yy - 0.55) / 0.45, 0, 1) ** 1.2
                         * np.array([1.0, 0.9, 0.7], np.float32)).astype(np.float32)
        self.setup_s = time.time() - t0
        print(f'plates {self.setup_s:.1f}s')

    def frame(self, t):
        W, H = self.W, self.H
        p = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.04 * p
        cam = (0.03 * W * p, -0.006 * H * p)
        img = self.sky.copy()
        img = img + S.sun(W, H, self.sun[0], self.sun[1], radius=0.012, color=(1.0, 0.8, 0.55), intensity=1.2,
                          glow_size=0.35, core=2.0)
        far, mid, near = self.sea
        tB, tA = self.towers
        Ls = [K.drift(far, W, H, t, speed=0.0015, cam=cam, zoom=zoom, depth=0.2),
              K.drift(tB, W, H, t, speed=0.0018, cam=cam, zoom=zoom, depth=0.22, billow=0.0008, seed=1),
              K.drift(mid, W, H, t, speed=0.003, cam=cam, zoom=zoom, depth=0.45, billow=0.001, seed=2),
              K.drift(tA, W, H, t, speed=0.0035, cam=cam, zoom=zoom, depth=0.5, billow=0.0012, seed=3),
              K.drift(near, W, H, t, speed=0.007, cam=cam, zoom=zoom, depth=1.0, billow=0.0015, seed=4)]
        img = K.over(img, *Ls)
        img = img * self.fg_shade
        for L in Ls:
            img = img + K.halo(L, 0.25)
        occ = np.clip(sum(L[..., 3] for L in Ls), 0, 1)
        vis = F.sun_visibility(occ, self.sun[0], self.sun[1], 0.01 * W)
        img = img + F.light_shafts(W, H, self.sun[0], self.sun[1], occluder=occ, strength=0.25, length=0.9,
                                   tint=(1.0, 0.8, 0.58), radius=0.14, t=t)
        img = img + F.anime_flare(W, H, self.sun[0], self.sun[1], intensity=0.35 + 0.65 * vis, tint=(1.0, 0.8, 0.55),
                                  rays=6)
        img = F.bloom_soft(img, threshold=1.0, knee=0.25, strength=0.3, halation=0.12)
        img = F.shoulder(img, 0.9, 0.12)
        return F.finish_fast(img, t, sat=1.05, grain_amt=0.004, vig=0.25, ca=0.0008)
