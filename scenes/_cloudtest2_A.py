"""Cloud test A-2: a sea of clouds seen from above at sunrise, two towering cumulus rising out of it,
the sun low near the horizon; gold/pink lit tops, deep indigo-violet valleys, the sea flattening into
haze toward the horizon. Technique A (lib/clouds_A.cloud_sea). Slow drift + parallax, gentle push (3 s)."""
import time
import numpy as np
import cv2
from lib import core as C, sky as SK, fx as F, clouds_A as CA

DURATION = 3.0

SKY = dict(stops=[(0.0, '#1b2466'), (0.25, '#33388a'), (0.5, '#6a4f9e'), (0.7, '#c46a9a'), (0.84, '#f59088'),
                  (0.94, '#ffc08a'), (1.0, '#ffe0a8')],
           sun_glow='#ffc47a', sun_glow_amt=0.35, below='#f7b08c', band=('#fff0c8', 0.45))


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        pw, ph = int(W * 1.1), int(H * 1.1)
        self.pw, self.ph = pw, ph
        self.hz = 0.44 * ph
        self.sun_p = (0.64 * pw, self.hz - 0.035 * ph)
        self.sky = SK.sky_gradient(pw, ph, SKY, horizon=self.hz / ph, sun=self.sun_p, sun_radius=0.7)
        towers = [dict(x=0.22 * pw, z=2.4, height=0.5 * ph, width=0.3 * pw, seed=4),
                  dict(x=0.82 * pw, z=5.0, height=0.3 * ph, width=0.17 * pw, seed=9)]
        self.layers = CA.cloud_sea(pw, ph, self.hz, self.sun_p, seed=21, pal='sunrise', n_rows=12, z_far=16.0,
                                   layer_splits=(5.0, 2.0), towers=towers, sky=self.sky, haze_col='#f2a8a8',
                                   amp=0.3 * (ph - self.hz), feat=0.33 * pw, rim=1.0, lz=-0.25)
        print('plates %.1fs' % (time.time() - t0))

    def _shift(self, plate, dx, dy, zoom=1.0):
        W, H = self.W, self.H
        M = np.array([[zoom, 0, W / 2 - zoom * (self.pw / 2 + dx)], [0, zoom, H / 2 - zoom * (self.ph / 2 + dy)]],
                     np.float32)
        return cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT if plate.shape[2] == 3 else cv2.BORDER_CONSTANT)

    def frame(self, t):
        W, H = self.W, self.H
        u = C.ease_in_out_sine(t / DURATION)
        z = 1.0 + 0.025 * u
        pan = (u - 0.5) * 0.02 * W
        img = self._shift(self.sky, pan * 0.05, 0, 1 + (z - 1) * 0.05)
        sx = (self.sun_p[0] - self.pw / 2 - pan * 0.05) * (1 + (z - 1) * 0.05) + W / 2
        sy = (self.sun_p[1] - self.ph / 2) * (1 + (z - 1) * 0.05) + H / 2
        img = img + SK.sun(W, H, sx, sy, radius=0.011, color=(1.0, 0.72, 0.4), glow_size=1.4)
        occ = np.zeros((H, W), np.float32)
        for plate, zd in self.layers:
            d = 1.0 / zd                      # parallax ~ 1/depth
            drift = -0.004 * W * t * d        # clouds drift slowly left, nearer ones faster
            L = self._shift(plate, pan * d - drift, 0, 1 + (z - 1) * d)
            img = F.over_rgba(img, L)
            occ = np.maximum(occ, L[..., 3])
        vis = F.sun_visibility(occ, sx, sy, 0.011 * W)
        img = img + F.light_shafts(W, H, sx, sy, occluder=occ, strength=0.25, length=0.8, radius=0.1,
                                   tint=(1.0, 0.75, 0.5), t=t)
        img = F.bloom_soft(img, threshold=0.9, knee=0.3, strength=0.18, halation=0.12)
        img = img + F.anime_flare(W, H, sx, sy, intensity=0.25 + 0.6 * vis, rot=0.02 * t, tint=(1.0, 0.8, 0.55))
        img = F.shoulder(img, 0.82)
        return F.finish_fast(img, t, sat=1.06, grain_amt=0.003, vig=0.22, ca=0.0005)
