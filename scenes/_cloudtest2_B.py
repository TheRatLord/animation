"""Cloud test B2: sea of clouds from above at sunrise, two towering cumulus, painted cut-paper technique (lib/clouds2)."""
import time
import numpy as np

from lib import core as C
from lib import sky as S
from lib import fx as F
from lib import clouds2 as K

DURATION = 3.0

SKY = dict(stops=[(0.0, '#1b2a78'), (0.25, '#33429a'), (0.48, '#6a58ae'), (0.66, '#b570b0'),
                  (0.8, '#e8849a'), (0.91, '#fb9a78'), (1.0, '#ffb277')],
           sun_glow='#ffa050', sun_glow_amt=0.1, below='#f7b894', band=('#ffc088', 0.14))


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        u = W / 1920.0
        self.u = u
        m = 0.07
        self.PW, self.PH = int(W * (1 + 2 * m)), int(H * (1 + 2 * m))
        ox, oy = W * m, H * m
        self.hz = 0.4                                   # horizon (fraction of H)
        self.sun = (0.66 * W, 0.385 * H)
        sun_p = (self.sun[0] + ox, self.sun[1] + oy)
        self.sky = S.sky_gradient(W, H, SKY, horizon=self.hz, sun=self.sun, sun_radius=0.35, variation=0.012,
                                  seed=5, horizon_glow=0.6)
        hy = self.hz * H + oy
        # three parallax plates (far sea / mid sea + towers / near sea), rows split at 7 and 11
        towers = [dict(row=5, plate=1, cx=0.87 * W + ox, base_off=0.03 * H, height=0.3 * H, seed=31,
                       preset='sunrise', haze=0.25, width=0.8, backlit=0.3, sun_z=0.25, bounce_group=0.5),
                  dict(row=8, plate=1, cx=0.24 * W + ox, base_off=0.02 * H, height=0.62 * H, seed=33,
                       preset='sunrise', width=0.85, backlit=0.2, sun_z=0.25, bounce_group=0.6)]
        self.plates = K.sea_of_clouds_plate(self.PW, self.PH, hy, sun=sun_p, preset='sunrise', seed=3, rows=16,
                                            splits=(7, 11), towers=towers, horizon_haze=(K._c('#ffc4a0'), 0.7),
                                            near_w=0.42, far_w=0.018, aspect_near=0.34, aspect_far=0.15,
                                            haze_far=0.75, persp=2.1, valley=0.6)
        # soft additive horizon glow band (drawn over the far sea, under the mid/near plates)
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]
        d = (ys - self.hz * H) / H
        g = np.exp(-(d / 0.035) ** 2) * (0.35 + 0.65 * np.exp(-((xs - self.sun[0]) / (0.35 * W)) ** 2))
        self.hglow = (g[..., None] * np.array([1.0, 0.7, 0.45], np.float32) * 0.22).astype(np.float32)
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        self.fg_shade = (1 - 0.22 * K._ss(0.62, 1.0, yy) * np.array([1.0, 0.85, 0.55], np.float32)).astype(np.float32)
        self.setup_s = time.time() - t0

    def frame(self, t):
        W, H = self.W, self.H
        p = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.045 * p
        cam = (0.04 * W * p, -0.008 * H * p)
        img = self.sky.copy()
        z0 = 1 + (zoom - 1) * 0.05
        lx = W / 2 + (self.sun[0] - W / 2) * z0 - cam[0] * 0.05
        ly = H / 2 + (self.sun[1] - H / 2) * z0
        img = img + S.sun(W, H, lx, ly, radius=0.013, color=(1.0, 0.68, 0.38), intensity=1.4, glow_size=0.45, core=1.8)
        depths = (0.2, 0.55, 1.1)
        speeds = (0.0015, 0.005, 0.011)
        Ls = [K.drift(pl, W, H, t, speed=sp, cam=cam, zoom=zoom, depth=d, billow=b, billow_scale=0.03,
                      billow_rate=0.3, seed=k) for k, (pl, sp, d, b) in
              enumerate(zip(self.plates, speeds, depths, (0.0008, 0.0014, 0.002)))]
        img = F.over_rgba(img, Ls[0])
        img = img + self.hglow
        for L in Ls[1:]:
            img = F.over_rgba(img, L)
        img = img * self.fg_shade                    # foreground rows sink into deeper, cooler shadow
        occ = np.clip(Ls[0][..., 3] + Ls[1][..., 3] + Ls[2][..., 3], 0, 1)
        vis = F.sun_visibility(occ, lx, ly, 0.012 * W)
        img = img + F.light_shafts(W, H, lx, ly, occluder=occ, strength=0.3, length=0.9,
                                   tint=(1.0, 0.78, 0.55), radius=0.14, t=t)
        shim = 1.0 + 0.06 * np.sin(1.9 * t) + 0.03 * np.sin(4.7 * t + 0.6)     # gentle flare shimmer
        img = img + F.anime_flare(W, H, lx, ly, intensity=(0.3 + 0.7 * vis) * shim, tint=(1.0, 0.75, 0.5), rays=6, glow=0.35,
                                  ghosts=0.6, streak=0.3, rot=0.02 * t)
        img = F.bloom_soft(img, threshold=1.0, knee=0.25, strength=0.3, halation=0.1, radii=(0.004, 0.012, 0.035, 0.09))
        img = F.shoulder(img, 0.88, 0.2)
        return F.finish_fast(img, t, sat=1.05, grain_amt=0.004, vig=0.25, ca=0.0008)
