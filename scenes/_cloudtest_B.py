"""Cloud test B1: summer cumulonimbus tower with anvil (painted cut-paper technique, lib/clouds2)."""
import time
import numpy as np

from lib import core as C
from lib import sky as S
from lib import fx as F
from lib import clouds2 as K

DURATION = 3.0


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        u = W / 1920.0
        m = 0.08                                   # plate margin for drift / camera
        self.PW, self.PH = int(W * (1 + 2 * m)), int(H * (1 + 2 * m))
        ox, oy = W * m, H * m                      # frame origin inside plates
        self.sun = (0.83 * W, 0.1 * H)             # on-screen sun (t=0): open sky just above the anvil
        sun_p = (self.sun[0] + ox, self.sun[1] + oy)
        self.sky = S.sky_gradient(W, H, 'summer_noon', horizon=1.02, sun=self.sun, sun_radius=0.5,
                                  variation=0.012, seed=3)
        PW, PH = self.PW, self.PH

        # far: low hazy bank along the horizon, bases dissolving into a pale haze band
        self.far = K.horizon_bank_plate(PW, PH, 0.99 * H + oy, 0.09 * H, sun=sun_p, preset='noon', seed=41,
                                        rows=6, haze=0.25, haze_color='#d6eef8',
                                        shrink=0.3, layer_haze=0.07)
        # aerial haze wash between the far bank and the nearer clouds
        self.haze = K.haze_plate(PW, PH, 0.62 * H + oy, 1.0 * H + oy, '#d2ecf8', 0.35)

        # mid: two side cumulus, each with its own light / composition, hazier with distance
        self.mid = K.cumulus_plate(PW, PH, [(0.12 * W + ox, 0.71 * H + oy, 0.22 * W, 0.12 * W, 7, 0.35),
                                            (0.95 * W + ox, 0.77 * H + oy, 0.14 * W, 0.07 * W, 8, 0.6)],
                                   sun=sun_p, preset='noon', n=3, sun_jitter=30.0)

        # hero tower (backlit edges near the sun)
        self.tower = K.cumulonimbus_plate(PW, PH, 0.5 * W + ox, 0.9 * H + oy, 0.84 * H, sun=sun_p, preset='noon',
                                          seed=12, anvil=True, anvil_dir=1.0, backlit=0.5, width=1.05, form=0.55, sun_z=0.45,
                                          haze_band=(0.66 * H + oy, 0.97 * H + oy, K._c('#c4e2f5'), 0.3))
        # high cirrus, faint, drifting slowly across the upper-left sky
        self.cirrus = K.cirrus_plate(PW, PH, preset='noon', seed=21, region=(0.04, 0.3), angle=-6.0, density=0.35,
                                     opacity=0.35)
        self.setup_s = time.time() - t0

    def frame(self, t):
        W, H = self.W, self.H
        p = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.04 * p
        cam = (-0.035 * W * p, 0.012 * H * p)
        img = self.sky.copy()
        # far layers slow, near layers faster (parallax); slow coherent billowing on the cumulus
        ci = K.drift(self.cirrus, W, H, t, speed=0.01, cam=cam, zoom=zoom, depth=0.15)
        far = K.drift(self.far, W, H, t, speed=0.003, cam=cam, zoom=zoom, depth=0.25, billow=0.0008,
                      billow_scale=0.02, billow_rate=0.3, seed=1)
        hz = K.drift(self.haze, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=0.3)
        mid = K.drift(self.mid, W, H, t, speed=0.012, cam=cam, zoom=zoom, depth=0.6, billow=0.0018,
                      billow_scale=0.03, billow_rate=0.35, seed=2)
        tow = K.drift(self.tower, W, H, t, speed=0.006, cam=cam, zoom=zoom, depth=0.45, billow=0.0016,
                      billow_scale=0.035, billow_rate=0.3, seed=3)
        # sun (behind everything): on-screen position with small parallax (depth 0.1)
        lx, ly = K.screen_pos(self.sun, W, H, cam=cam, zoom=zoom, depth=0.1)
        img = img + S.sun(W, H, lx, ly, radius=0.011, color=(1.0, 0.96, 0.88), intensity=1.0, glow_size=1.0)
        img = K.composite(img, ci, far, hz, mid, tow)
        occ = K.occluder(tow, mid)
        vis = F.sun_visibility(occ, lx, ly, 0.012 * W)
        img = img + F.light_shafts(W, H, lx, ly, occluder=occ, strength=0.22, length=0.8,
                                   tint=(1.0, 0.95, 0.85), radius=0.1, t=t)
        shim = 1.0 + 0.06 * np.sin(2.1 * t) + 0.03 * np.sin(5.3 * t + 1.0)     # gentle flare shimmer
        img = img + F.anime_flare(W, H, lx, ly, intensity=(0.25 + 0.75 * vis) * shim, tint=(1.0, 0.93, 0.8), rays=6,
                                  ghosts=0.7, streak=0.15, rot=0.02 * t)
        img = F.bloom_soft(img, threshold=1.12, knee=0.12, strength=0.5, halation=0.08, radii=(0.0015, 0.005, 0.016, 0.06))
        img = F.shoulder(img, 0.92, 0.03)
        return F.finish_fast(img, t, sat=1.06, grain_amt=0.004, vig=0.22, ca=0.0008)
