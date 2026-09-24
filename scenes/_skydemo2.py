"""Sky library demo 2: magic hour - a low sun near the horizon at lower-left; clouds near it are backlit
violet silhouettes with blazing gold rims, the others are lit peach-orange from the lower-left with lavender
shadow sides and pink bounce light. One big hero bank, varied mid clouds, many small fragments, low
stratocumulus streaks, hooked pink/gold cirrus, a contrail and a small flock. Slow push-in + pan with
parallax and coherent billowing (3 s)."""
import numpy as np
from lib import core as C, sky as S, fx as F

DURATION = 3.0


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        hz = 0.93
        pw, ph = int(W * 1.16), int(H * 1.12)
        self.pw, self.ph = pw, ph
        ox, oy = (pw - W) / 2, (ph - H) / 2
        self.sun_p = (0.24 * W + ox, 0.8 * H + oy)
        hzp = (hz * H + oy) / ph
        hy = hzp * ph
        self.sky = S.sky_gradient(pw, ph, 'magic_hour', horizon=hzp, sun=self.sun_p, sun_radius=0.4)
        # designed layout: (cx, base_y, width, height, dist, kind, flat) in plate px
        hero = [(0.68 * pw, 0.6 * ph, 0.48 * pw, 0.3 * ph, 2.0, 'cumulus', 1.0)]
        mids = [(0.2 * pw, 0.66 * ph, 0.2 * pw, 0.11 * ph, 3.0, 'cumulus', 0.9),      # near the sun: backlit
                (0.38 * pw, 0.36 * ph, 0.14 * pw, 0.08 * ph, 3.4, 'cumulus', 0.9),
                (0.95 * pw, 0.22 * ph, 0.2 * pw, 0.1 * ph, 2.6, 'stratocumulus', 0.8)]
        frags = [s for s in S.cloud_specs(pw, ph, seed=21, kind='cumulus', coverage=0.4, horizon=hzp,
                                          fragments=1.4, strato=1.0)[1:]
                 if s[2] < 0.1 * pw or s[5] == 'stratocumulus']
        kw = dict(seed=21, sun_pos=self.sun_p, sun_z=0.12, palette='magic_hour', sky=self.sky, rim=1.3,
                  backlit=0.85, backlit_radius=0.3)
        near = S.cumulus_plate(pw, ph, clouds=hero + mids, layer=(0, 3.2), **kw)
        mid = S.cumulus_plate(pw, ph, clouds=hero + mids + frags, layer=(3.2, 6.0), **kw)
        far = S.cumulus_plate(pw, ph, clouds=frags, layer=(6.0, 99), **kw)
        bank = S.horizon_bank(pw, ph, seed=9, palette='magic_hour', horizon=hzp, sun_pos=self.sun_p, sky=self.sky,
                              height=0.045, sun_z=0.05, backlit=0.8, rows=3)
        cir = S.cirrus_plate(pw, ph, seed=8, color=(1.0, 0.86, 0.8), under=(1.0, 0.62, 0.55), angle=-6,
                             density=0.55, region=(0.03, 0.34), opacity=0.6, clusters=2)
        trail = S.contrail(pw, ph, (0.98 * pw, 0.1 * ph), (0.7 * pw, 0.2 * ph), color=(1.0, 0.9, 0.82),
                           width=0.0012, spread=5.0, opacity=0.45, seed=3, curve=-0.01 * pw)
        self.layers = S.CloudDrift([(cir, -0.002, 0.08), (trail, -0.002, 0.08), (bank, -0.0005, 0.1, 0.0005),
                                    (far, -0.0012, 0.16, 0.0008), (mid, -0.0022, 0.28, 0.0012),
                                    (near, -0.004, 0.45, 0.0016)])
        self.bird_col = S.bird_color(self.sky[int(0.35 * ph), int(0.5 * pw)])

    def _cam(self, t):
        u = C.ease_in_out_sine(t / DURATION)
        return (u - 0.5) * 0.05 * self.W, (0.5 - u) * 0.02 * self.H, 1.0 + 0.05 * u

    def frame(self, t):
        W, H = self.W, self.H
        cam = self._cam(t)
        dp = 0.06
        z = 1 + (cam[2] - 1) * dp
        lx = (self.sun_p[0] - (self.pw / 2 + cam[0] * dp)) * z + W / 2
        ly = (self.sun_p[1] - (self.ph / 2 + cam[1] * dp)) * z + H / 2
        img = S.drift(self.sky, W, H, t, 0.0, cam=cam[:2], zoom=cam[2], depth=dp)
        img = img + S.sun(W, H, lx, ly, radius=0.013, color=(1.0, 0.74, 0.4), core=0.9)
        Ls = self.layers.render_layers(W, H, t, *cam)
        for L in Ls:
            img = F.over_rgba(img, L)
        occ = np.maximum.reduce([Ls[i][..., 3] for i in (2, 3, 4, 5)])
        vis = F.sun_visibility(occ, lx, ly, 0.013 * W)
        img = img + F.light_shafts(W, H, lx, ly, occluder=occ, strength=0.28, length=0.95, radius=0.07,
                                   tint=(1.0, 0.72, 0.42), t=t)
        b = S.birds(W, H, t, seed=2, n=7, center=(0.5, 0.3), velocity=(0.02, -0.004), size=0.017, spread=0.07)
        img = C.over(img, self.bird_col, b)
        img = F.bloom_soft(img, threshold=1.0, knee=0.3, strength=0.2, halation=0.12)
        img = img + F.anime_flare(W, H, lx, ly, intensity=0.3 + 0.6 * vis, tint=(1.0, 0.76, 0.5), rot=0.02 * t,
                                  center=(W / 2 + cam[0] * 0.3, H / 2))
        img = F.shoulder(img, 0.82, desat=0.15)
        return F.finish_fast(img, t, sat=1.06, grain_amt=0.004, vig=0.22, ca=0.0006)
