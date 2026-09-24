"""s10 - FINALE: sunrise over the sea of clouds, seen from a summit shrine (round 17: reconceived).

We stand on a mountain summit at dawn. A weathered vermilion torii with a shimenawa and a wooden summit
marker (雲見岳山頂) stand on the rocky crest; a lone Japanese black pine frames the upper right. Below us
the sea of clouds (lib.clouds3 ray-marched deck: broad banks near, compressing into thin gold-rimmed
rows at the horizon, deep indigo valleys) runs out to a crisp white-hot horizon line where a small hot
sun has just cleared the cloud sea. Distant peaks stand in the clouds like islands; a nearer forested
ridge rises out of them on the right.

Planes (parallax factor = 1.5 / distance): sky + sun (fixed, the sun climbs slowly), altocumulus streets,
far peaks, deck (per-pixel depth re-projection), forested ridge, summit foreground (1.0) and an
out-of-focus grass fringe (1.35). Camera: eased crane-up + truck right + slight push, so the foreground
drops and slides against the clouds while the horizon barely moves. Secondary motion: grass, pine pads
and shide papers sway, mist drifts, clouds drift. The light swells through the shot (sun, sky glow,
rim light, shafts) and peaks in a warm halation at the very end.
"""
import math
import os
import sys
import time

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import core as C, fx as F, sky as S, clouds3 as K  # noqa: E402
import s10_sea_of_clouds_summit as SM  # noqa: E402

DURATION = 6.0
FOV = 55.0
F32 = np.float32
DK = 1.5            # parallax factor = DK / Z


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        sc = W / 1920.0
        self.sc = sc
        m = 0.08
        PW, PH = int(round(W * (1 + 2 * m))), int(round(H * (1 + 2 * m)))
        ox, oy = (PW - W) / 2.0, (PH - H) / 2.0
        self.PW, self.PH, self.ox, self.oy = PW, PH, ox, oy
        self.hz = 0.415
        hy = self.hz * H + oy
        self.hy = hy
        self.sun0 = (0.615 * W, 0.392 * H)            # frame px at t=0 (climbs to ~0.37H)
        sun_p = (self.sun0[0] + ox, self.sun0[1] + oy - 0.01 * H)
        f = (PW / 2.0) / math.tan(math.radians(FOV) / 2)
        self.f = f
        # ---- sky (frame-sized, fixed)
        self.sky = SM.sky_plate(W, H, self.hz * H, self.sun0)
        # ---- altocumulus streets, lit gold / pink on the sun side
        self.flecks = K.sky_rows_plate(PW, PH, hy, seed=21, fov=FOV, alt=4.0, cell=0.34, cell_stretch=1.3,
                                       row_l=0.9, patch=6.0, patch_thr=0.16, cover=0.5, warp=1.6, angle=14.0,
                                       region=(0.0, 0.93), z_haze=50.0, lit=(1.35, 0.86, 0.55),
                                       body=(0.52, 0.42, 0.66), under=(0.3, 0.27, 0.52), a_lo=0.1, a_hi=0.3,
                                       lumps=0.8, lit_screen=(0.0, 3.5), opacity=0.9, haze_col=(1.0, 0.74, 0.62),
                                       haze_fade=0.25, veil=0.3, edge_px=1.0)
        # ---- far peaks standing in the cloud sea
        def base_at(Z):
            return hy + f * 1.0 / Z
        self.peaks = []
        pk = [  # (x0, x1 frame frac, Z, height units, peaks, seed, haze)
            (-0.1, 0.45, 26.0, 2.6, [(0.35, 1.0, 0.33), (0.72, 0.55, 0.22), (0.1, 0.35, 0.2)], 4, 0.38),
            (0.62, 1.12, 34.0, 1.6, [(0.35, 0.8, 0.3), (0.7, 1.0, 0.25)], 6, 0.5),
            (0.28, 0.75, 55.0, 1.1, [(0.3, 0.7, 0.3), (0.62, 1.0, 0.26)], 8, 0.62),
        ]
        for (a, b, Z, hu, pks, sd, hzv) in pk:
            by = base_at(Z) + 0.35 * f / Z
            P = SM.peak_layer(PW, PH, ox + a * W, ox + b * W, by, hu * f / Z, pks, sd, sun_p,
                              lit_col=(0.9, 0.55, 0.5), shade_col=np.array([0.24, 0.21, 0.40], F32),
                              haze_col=(0.66, 0.52, 0.66), haze=hzv, rim_col=(1.4, 0.95, 0.55),
                              rim_px=1.6 * sc, snow=0.0)
            self.peaks.append((P, Z))
        # ---- deck
        cols = [((ox + (a + b) / 2 * W - PW / 2) / f * Z, Z, 0.25 * (b - a) * W / f * Z, hu * 0.7)
                for (a, b, Z, hu, _, _, _) in pk[:2]]
        self.deck_R = K.deck_fields(PW, PH, hy, sun_p, fov=FOV, seed=3, towers=cols, stretch=2.2,
                                    bank=(0.45, 9.0, 3.2), p1=(0.36, 1.6), p3=(0.06, 0.18), p4=(0.012, 0.065))
        self.deck = K.paint_deck(self.deck_R, PW, PH, hy, sun_p[0], deep=(0.09, 0.08, 0.24),
                                 body=(0.36, 0.33, 0.58), near_dark=0.45, near_rim=0.15, near_z=(1.5, 9.0),
                                 haze_far=(0.84, 0.7, 0.86), haze_sun=(1.2, 0.9, 0.66), haze_amt=0.65,
                                 z_haze=50.0)
        x0, y0, x1, y1 = self.deck_R['box']
        self.deck_rgba = np.zeros((PH, PW, 4), F32)
        self.deck_rgba[y0:y1, x0:x1, :3] = self.deck
        self.deck_rgba[y0:y1, x0:x1, 3] = 1.0
        self.deck_rgba[:y0 + 3, :, 3] = 0.0
        self.deck_rgba[..., :3] = K._bleed(self.deck_rgba[..., :3], self.deck_rgba[..., 3], 3.0)
        Zm = np.full((PH, PW), 200.0, F32)
        Zm[y0:y1, x0:x1] = np.where(self.deck_R['A'] > 0.3, self.deck_R['Z'], 200.0)
        self.deckZ = Zm
        # ---- forested ridge rising out of the clouds (right of centre)
        Zr = 6.0
        self.ridge_Z = Zr
        rx = ox + np.array([-0.5, 0.5, 0.58, 0.66, 0.74, 0.8, 0.86, 0.95, 1.1, 1.5]) * W
        ry = oy + np.array([0.9, 0.9, 0.66, 0.61, 0.575, 0.555, 0.56, 0.585, 0.62, 0.62]) * H
        self.ridge = SM.ridge_plate(PW, PH, rx, ry, base_at(Zr) + 0.5 * f / Zr, 12, sun_p, sc,
                                    body=(0.2, 0.18, 0.32), top_col=(0.16, 0.15, 0.26), rim_col=(1.3, 0.8, 0.45),
                                    haze_col=(0.62, 0.5, 0.66), haze=0.12)
        # ---- summit foreground
        sm = SM.Summit(PW, PH, W, H, ox, oy, sun_p)
        self.fgS, self.fgD, self.fgW = sm.build()
        self._grid = np.mgrid[0:H, 0:W].astype(F32)
        self._pgrid = np.mgrid[0:PH, 0:PW].astype(F32)
        self.flare = K.SunFlare(W, H, tint=(1.0, 0.8, 0.55), star_len=0.075, bloom=0.6, ring=0.5, chain=1.0,
                                vertical=0.0)
        # horizon: thin white-hot line + narrow warm bloom
        Y = np.arange(H, dtype=F32)[:, None]
        X = np.arange(W, dtype=F32)[None, :]
        d = (Y - self.hz * H) / H
        wide = np.exp(-(d / 0.02) ** 2) * (0.25 + 0.75 * np.exp(-((X - self.sun0[0]) / (0.28 * W)) ** 2))
        line = np.exp(-(d / 0.0018) ** 2) * (0.3 + 0.7 * np.exp(-((X - self.sun0[0]) / (0.32 * W)) ** 2))
        self.hglow = (wide[..., None] * np.array([1.0, 0.7, 0.42], F32) * 0.25 +
                      line[..., None] * np.array([1.0, 0.95, 0.86], F32) * 0.85).astype(F32)
        # mist streaks drifting over the near clouds (between ridge and summit)
        ys = np.arange(PH, dtype=F32)[:, None] / PH
        n2 = K._noise(PW, PH, 4.0, 78, 4, stretch=9.0, angle=-3)
        band = SM.ss(0.52, 0.62, ys) * (1 - SM.ss(0.8, 0.9, ys))
        mist = np.clip(n2 * 1.4 - 0.1, 0, 1) * band * 0.3
        xs_ = (np.arange(PW, dtype=F32)[None, :] - sun_p[0]) / W
        warm = np.exp(-(xs_ / 0.3) ** 2)[..., None]
        mcol = np.array([0.95, 0.66, 0.72], F32) * (1 - warm) + np.array([1.35, 0.95, 0.55], F32) * warm
        self.mist = (mcol * mist[..., None]).astype(F32)          # premultiplied, additive
        # sun star: long thin rays (vertical / horizontal longest, diagonals shorter), cached 2x sprite
        cw, ch = 2 * W, 2 * H
        yy_, xx_ = np.mgrid[0:ch, 0:cw].astype(F32)
        dx_, dy_ = (xx_ - W) / W, (yy_ - H) / W
        r_ = np.sqrt(dx_ * dx_ + dy_ * dy_) + 1e-6
        th_ = np.arctan2(dy_, dx_)
        star = np.zeros((ch, cw), F32)
        for k, (ang, ln, wd) in enumerate([(0.08, 0.34, 0.0011), (0.08 + math.pi / 2, 0.30, 0.0011),
                                           (0.08 + math.pi / 4, 0.12, 0.0009), (0.08 - math.pi / 4, 0.12, 0.0009)]):
            perp = np.abs(dx_ * math.sin(ang) - dy_ * math.cos(ang))
            along = np.abs(dx_ * math.cos(ang) + dy_ * math.sin(ang))
            wdt = wd * (1 + along / 0.05)
            star += np.exp(-(perp / wdt) ** 2) * np.exp(-along / (ln * 0.35)) * (1 / (1 + along / 0.01))
        glow_ = np.exp(-r_ / 0.02) * 0.5 + np.exp(-r_ / 0.07) * 0.18
        self.star = (star[..., None] * np.array([1.0, 0.9, 0.72], F32) * 2.2 +
                     glow_[..., None] * np.array([1.0, 0.78, 0.5], F32)).astype(F32)
        self.setup_s = time.time() - t0

    # ------------------------------------------------------------------------------------------------
    def _warp_deck(self, cam, zoom):
        W, H = self.W, self.H
        yy, xx = self._grid
        d0 = 0.2
        z0 = 1 + (zoom - 1) * d0
        px = (xx - W / 2) / z0 + self.PW / 2 + cam[0] * d0
        py = (yy - H / 2) / z0 + self.PH / 2 + cam[1] * d0
        Zs = cv2.remap(self.deckZ, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dep = np.clip(DK / np.maximum(Zs, 0.5), 0.0, 1.2)
        z = 1 + (zoom - 1) * dep
        px = (xx - W / 2) / z + self.PW / 2 + cam[0] * dep
        py = (yy - H / 2) / z + self.PH / 2 + cam[1] * dep
        rgba = cv2.remap(self.deck_rgba, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Zw = cv2.remap(self.deckZ, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rgba, Zw

    def _bloom(self, img, thr, strength, halation):
        """Fast multi-scale bloom + warm halation (quarter-res bright pass)."""
        H, W = self.H, self.W
        q = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        lum = q.max(-1, keepdims=True)
        k = np.clip((lum - thr + 0.3) / 0.6, 0, 1)
        br = q * (k * k)
        s = W / 1920.0
        b4 = cv2.resize(br, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
        big = cv2.GaussianBlur(b4, (0, 0), 4 * s) * 0.6 + cv2.GaussianBlur(b4, (0, 0), 10 * s) * 0.45
        big = cv2.resize(big, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR)
        acc = (cv2.GaussianBlur(br, (0, 0), 1.5 * s) * 1.0 + cv2.GaussianBlur(br, (0, 0), 5 * s) * 0.8 + big) / 2.85
        hal = cv2.GaussianBlur(br, (0, 0), 3 * s) * np.array([1.0, 0.45, 0.2], F32) * (halation * 0.5)
        add = cv2.resize(acc * strength + hal, (W, H), interpolation=cv2.INTER_LINEAR)
        img += add
        return img

    def _atmo(self, sx, sy):
        if not hasattr(self, '_atm'):
            W, H = self.W, self.H
            yy, xx = np.mgrid[0:2 * H, 0:2 * W].astype(F32)
            d = np.sqrt((xx - W) ** 2 + ((yy - H) * 1.6) ** 2) / W
            g = np.exp(-d / 0.05) * 0.3 + np.exp(-d / 0.16) * 0.12 + np.exp(-d / 0.45) * 0.06
            self._atm = (g[..., None] * np.array([1.0, 0.62, 0.3], F32)).astype(F32)
        M = np.array([[1, 0, sx - self.W], [0, 1, sy - self.H]], np.float32)
        return cv2.warpAffine(self._atm, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _star(self, sx, sy, rot):
        c, s_ = math.cos(rot), math.sin(rot)
        W, H = self.W, self.H
        M = np.array([[c, -s_, sx - (c * W - s_ * H)], [s_, c, sy - (s_ * W + c * H)]], np.float32)
        return cv2.warpAffine(self.star, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _sway(self, t):
        """Grass / needles / papers: smooth travelling gusts, displacement scaled by the sway weight."""
        PH, PW = self.PH, self.PW
        yy, xx = self._pgrid
        sc = self.sc
        g = 0.6 + 0.4 * math.sin(0.9 * t + 0.4)
        ph = xx / (260 * sc) - 1.7 * t
        dx = (np.sin(ph) * 0.7 + 0.3 * np.sin(2.3 * ph + 1.1 + 0.8 * t)) * g * 3.2 * sc * self.fgW
        dy = np.abs(dx) * 0.25
        return cv2.remap(self.fgD, xx - dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def frame(self, t):
        W, H = self.W, self.H
        u = t / DURATION
        p = C.ease_in_out_sine(u)
        zoom = 1.0 + 0.045 * p
        cam = (0.045 * W * p, -0.07 * H * p)
        light = 0.78 + 0.27 * C.ease_in_out_sine(min(u * 1.1, 1.0))       # light swells
        # sun climbs slowly (fixed at infinity: tiny parallax)
        sx = self.sun0[0] + cam[0] * 0.0
        sy = self.sun0[1] - 0.018 * H * p
        img = self.sky * (0.92 + 0.1 * light)
        img = img + S.sun(W, H, sx, sy, radius=0.0065, color=(1.0, 0.82, 0.55), intensity=1.1 * light,
                          glow_size=1.3, core=2.4)
        Lk = K.drift(self.flecks, W, H, t, speed=0.0025, cam=cam, zoom=zoom, depth=0.04)
        img = K.composite(img, Lk)
        img += self.hglow * light
        deck, Zw = self._warp_deck(cam, zoom)
        # far peaks, occluded by nearer deck crests
        layers = []
        for i, (P, Z) in enumerate(self.peaks):
            L = K.drift(P, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=DK / Z)
            L[..., 3] *= np.clip((Zw - (Z - 1.5)) / 3.0, 0, 1)
            layers.append((Z, L))
        img = K.composite(img, deck)
        for Z, L in sorted(layers, key=lambda q: -q[0]):
            img = K.composite(img, L)
        Lr = K.drift(self.ridge, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=DK / self.ridge_Z)
        Lr[..., 3] *= np.clip((Zw - (self.ridge_Z - 0.8)) / 1.6, 0, 1)
        img = K.composite(img, Lr)
        Lm = K.drift(self.mist, W, H, t, speed=0.004, cam=cam, zoom=zoom, depth=0.35)
        img += Lm * (0.6 + 0.4 * light)
        # foreground
        D = self._sway(t)
        fs = K.drift(self.fgS, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=1.0)
        fd = K.drift(D, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=1.0)
        # rim light swells with the light
        # warm light-filled atmosphere behind the foreground (grows with the light)
        atm = self._atmo(sx, sy)
        img += atm * (0.55 * light ** 2)
        img = K.composite(img, fs, fd)
        img += atm * (0.18 * light ** 2)            # veiling glare over the silhouettes near the sun
        occ = K.occluder(fs, fd)
        img = img + F.light_shafts(W, H, sx, sy, occluder=occ, strength=0.07 * light, length=0.95,
                                   tint=(1.0, 0.72, 0.45), radius=0.06, t=t)
        vis = F.sun_visibility(occ, sx, sy, 0.008 * W)
        shim = 1.0 + 0.04 * math.sin(1.9 * t) + 0.02 * math.sin(4.3 * t + 0.6)
        img = img + self._star(sx, sy, 0.01 * t) * (0.5 + 0.5 * vis) * shim * light ** 1.5
        img = img + self.flare.render(sx, sy, intensity=(0.4 + 0.6 * vis) * shim * light, t=t,
                                      center=(W / 2 + cam[0] * 0.3, H / 2))
        # end: warm halation swell
        e = C.smoothstep(0.82, 1.0, u)
        img = self._bloom(img, 1.0 - 0.08 * e, 0.2 + 0.15 * e, 0.12 + 0.08 * e)
        if e > 0:
            img += atm * (0.45 * e)
        img = F.shoulder(img, 0.88, 0.05)
        return F.finish_fast(img, t, exposure=0.97 + 0.06 * light, sat=1.06, grain_amt=0.004, vig=0.28, ca=0.0008)
