"""Sky library demo 1: a towering summer cumulonimbus with its anvil sliding past a high sun - silver lining,
translucent sun-side edges, volumetric shafts radiating past the anvil, soft chromatic flare, hooked cirrus,
receding horizon banks. Slow push-in + tilt-up with 5 parallax planes and coherent lobe billowing (3 s)."""
import math
import numpy as np
from lib import core as C, sky as S, fx as F

DURATION = 3.0


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        hz = 0.97
        pw, ph = int(W * 1.18), int(H * 1.16)
        self.pw, self.ph = pw, ph
        ox, oy = (pw - W) / 2, (ph - H) / 2
        self.sun_p = (0.715 * W + ox, 0.085 * H + oy)            # sun in plate px
        hzp = (hz * H + oy) / ph
        self.sky = S.sky_gradient(pw, ph, 'summer_noon', horizon=hzp, sun=self.sun_p)
        specs = S.cloud_specs(pw, ph, seed=11, kind='cumulonimbus', coverage=0.35, horizon=hzp, hero_x=0.42,
                              fragments=0.8, strato=0.4)
        kw = dict(seed=11, sun_pos=self.sun_p, sun_z=0.22, palette='summer_noon', clouds=specs, sky=self.sky,
                  backlit=0.35, backlit_radius=0.12, wind=1.0)
        hero = S.cumulus_plate(pw, ph, layer=(0, 2.5), **kw)      # the tower (dist 2)
        mid = S.cumulus_plate(pw, ph, layer=(2.5, 5.5), **kw)
        far = S.cumulus_plate(pw, ph, layer=(5.5, 99), **kw)
        bank = S.horizon_bank(pw, ph, seed=4, palette='summer_noon', horizon=hzp, sun_pos=self.sun_p, sky=self.sky,
                              height=0.05, rows=3)
        cir = S.cirrus_plate(pw, ph, seed=4, angle=-10, density=0.5, region=(0.02, 0.4), opacity=0.55,
                             color=(1.0, 1.0, 1.0), under=(0.86, 0.92, 1.0), clusters=2)
        # (plate, speed fraction-of-W/s, parallax depth, billow)
        self.layers = S.CloudDrift([(cir, 0.003, 0.1), (bank, 0.0008, 0.12, 0.0005), (far, 0.0015, 0.2, 0.0008),
                                    (mid, 0.0028, 0.32, 0.0012), (hero, 0.0045, 0.5, 0.0016)])

    def _cam(self, t):
        u = C.ease_in_out_sine(t / DURATION)
        zoom = 1.0 + 0.07 * u                          # slow push-in
        cam_y = (0.5 - u) * 0.07 * self.H              # tilt up
        cam_x = (u - 0.5) * 0.03 * self.W              # slight pan right
        return cam_x, cam_y, zoom

    def _to_screen(self, p, t, cam, depth, speed=0.0):
        cam_x, cam_y, zoom = cam
        z = 1 + (zoom - 1) * depth
        cx = self.pw / 2 - speed * self.W * t + cam_x * depth
        cy = self.ph / 2 + cam_y * depth
        return (p[0] - cx) * z + self.W / 2, (p[1] - cy) * z + self.H / 2

    def frame(self, t):
        W, H = self.W, self.H
        cam = self._cam(t)
        img = S.drift(self.sky, W, H, t, 0.0, cam=cam[:2], zoom=cam[2], depth=0.06)
        lx, ly = self._to_screen(self.sun_p, t, cam, 0.06)
        img = img + S.sun(W, H, lx, ly, radius=0.011, color=(1.0, 0.95, 0.85))
        Ls = self.layers.render_layers(W, H, t, *cam)
        for L in Ls:
            img = F.over_rgba(img, L)
        occ = np.maximum(np.maximum(Ls[3][..., 3], Ls[4][..., 3]), Ls[2][..., 3])
        vis = F.sun_visibility(occ, lx, ly, 0.011 * W)
        img = img + F.light_shafts(W, H, lx, ly, occluder=occ, strength=0.3, length=0.95, radius=0.09,
                                   tint=(1.0, 0.95, 0.85), t=t)
        img = F.bloom_soft(img, threshold=1.05, knee=0.25, strength=0.14, halation=0.05)
        img = img + F.anime_flare(W, H, lx, ly, intensity=0.25 + 0.75 * vis, rot=0.02 * t, tint=(1.0, 0.95, 0.86),
                                  center=(W / 2 + cam[0] * 0.3, H / 2))
        img = F.shoulder(img, 0.8)
        return F.finish_fast(img, t, sat=1.1, grain_amt=0.004, vig=0.2, ca=0.0006)
