"""_cloudtest_C - technique C test: a colossal summer cumulonimbus (anvil spreading downwind) against a
deep cerulean sky; the sun sits at the upper right, half hidden behind the crown's edge and slowly slides
out as the cloud drifts. Low hazy cumulus along the horizon, a couple of small fair-weather puffs nearer
the camera drift faster (parallax), slow push-in.

Clouds: lib/clouds_C (2.5D relief slab + single-scattering light march + painterly posterised paint-over).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import core as C, sky as S, fx as F, clouds_C as K  # noqa: E402

DURATION = 3.0

SKY = dict(stops=[(0.0, '#0b32a0'), (0.2, '#1446bb'), (0.42, '#2166d2'), (0.62, '#3b8ae0'),
                  (0.8, '#72b8ee'), (0.92, '#ace0f6'), (1.0, '#d6f2fa')],
           sun_glow='#fff6e0', sun_glow_amt=0.28, below='#e2f6fb', band=('#eefafd', 0.6))

HORIZON = 0.9


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        s = H / 1080.0
        self.s = s
        # --- hero tower plate (bigger than needed so it can drift)
        pw, ph = int(1.02 * W), int(1.04 * H)
        self.tx0, self.ty0 = -0.06 * W, -0.02 * H          # plate top-left in the frame at t=0
        base_row, top_row = 0.9 * ph, 0.05 * ph
        cx0 = 0.44 * pw
        # sun: upper right, tucked just behind the right edge of the crown
        sun_dir = (0.72, 0.5, 0.42)
        tower, sp = K.tower_plate(pw, ph, seed=11, sun_dir=sun_dir, sun_px=('edge', 0.2 * H - self.ty0, 0.006 * W),
                                  base_row=base_row, top_row=top_row, cx0=cx0, anvil=True, anvil_dir=-1.0, scale=s,
                                  haze=0.55, haze_top=0.62, return_sun=True, meander_px=25.0)
        self.sun0 = (sp[0] + self.tx0, sp[1] + self.ty0)
        self.tower = K.premul(tower)
        # --- horizon bank: small, hazy cumulus
        bw, bh = int(1.25 * W), int(0.3 * H)
        bank = np.zeros((bh, bw, 4), np.float32)
        rng = np.random.default_rng(5)
        xs = np.linspace(0.0, 0.95, 9) + rng.uniform(-0.02, 0.02, 9)
        for i, xf in enumerate(xs):
            cw = int(rng.uniform(0.12, 0.2) * W)
            chh = int(cw * rng.uniform(0.25, 0.34))
            p = K.cumulus_plate(cw, chh, seed=40 + i, sun_dir=sun_dir, scale=s * 0.3, haze=0.9, haze_top=0.0,
                                tear=0.5, rim=0.6, keys=[(0.0, 1.15, 0.0), (0.35, 1.0, 0.05), (0.7, 0.62, 0.1),
                                                         (1.0, 0.25, 0.12)])
            p[..., :3] = K._lerp(p[..., :3], K._hex('#bfe0f6'), 0.4)
            x = int(xf * bw)
            y = int(bh - chh * 0.9)
            _paste(bank, p, x, y)
        self.bank = K.premul(bank)
        # --- two small nearer puffs (drift faster)
        self.puffs = []
        for i, (xf, yf, wf) in enumerate([(0.72, 0.6, 0.24), (-0.05, 0.63, 0.2)]):
            cw = int(wf * W)
            chh = int(cw * 0.42)
            p = K.cumulus_plate(cw, chh, seed=70 + i, sun_dir=sun_dir, scale=s * 0.5,
                                keys=[(0.0, 0.95, 0.0), (0.3, 0.85, 0.06), (0.6, 0.55, 0.12), (0.85, 0.3, 0.2),
                                      (1.0, 0.1, 0.22)],
                                haze=0.3, haze_top=0.3, tear=0.8)
            self.puffs.append((K.premul(p), xf * W, yf * H))
        self.sky = S.sky_gradient(W, H, SKY, horizon=HORIZON, sun=self.sun0, sun_radius=0.5, seed=2)

    def frame(self, t):
        W, H, s = self.W, self.H, self.s
        u = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.018 * u
        piv = (0.6 * W, 0.45 * H)
        sun = (piv[0] + (self.sun0[0] - piv[0]) * zoom, piv[1] + (self.sun0[1] - piv[1]) * zoom)
        img = self.sky.copy()
        img += S.sun(W, H, sun[0], sun[1], radius=0.011, color=(1.0, 0.9, 0.7), intensity=0.8)
        # far bank (slowest), hero tower, near puffs (fastest)
        K.place(img, self.bank, -0.1 * W - 6 * s * t, H * HORIZON - self.bank.shape[0] + 8 * s, zoom=1 + (zoom - 1) * 0.3,
                pivot=piv)
        a_t = K.place(img, self.tower, self.tx0 - 11 * s * t, self.ty0, zoom=zoom, pivot=piv)
        occ = a_t.copy()
        for pm, x, y in self.puffs:
            a = K.place(img, pm, x - 26 * s * t, y, zoom=1 + (zoom - 1) * 1.6, pivot=piv)
            occ = np.maximum(occ, a)
        vis = F.sun_visibility(occ, sun[0], sun[1], 0.014 * W)
        # light: shafts from the edge of the tower, flare, bloom
        img += F.light_shafts(W, H, sun[0], sun[1], occluder=occ, strength=0.22, length=0.7, t=t,
                              tint=(1.0, 0.93, 0.8), radius=0.08)
        img += F.anime_flare(W, H, sun[0], sun[1], intensity=0.25 + 0.5 * vis, tint=(1.0, 0.92, 0.78),
                             rays=6, ray_len=0.05, ghosts=0.6, streak=0.15, rot=0.01 * t)
        img = F.bloom_soft(img, threshold=1.0, knee=0.15, strength=0.3, halation=0.1)
        return F.finish_fast(img, t, sat=1.06, grain_amt=0.004, vig=0.18, ca=0.0008)


def _paste(dst, src, x, y):
    """Straight-alpha 'over' of src into dst (both RGBA straight) at integer (x, y), clipped."""
    H, W = dst.shape[:2]
    h, w = src.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    a = s[..., 3:4]
    ao = a + d[..., 3:4] * (1 - a)
    rgb = (s[..., :3] * a + d[..., :3] * d[..., 3:4] * (1 - a)) / np.maximum(ao, 1e-5)
    d[..., :3] = rgb
    d[..., 3:4] = ao
