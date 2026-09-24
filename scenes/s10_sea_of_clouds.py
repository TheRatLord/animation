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
import s10_sea_of_clouds_floor as FL  # noqa: E402

DURATION = 6.0
FOV = 55.0
F32 = np.float32
DK = 1.5            # parallax factor = DK / Z


def _add(img, x, k=1.0):
    """img += x * k in place (multi-threaded)."""
    if x.dtype != np.float32:
        x = x.astype(F32)
    cv2.scaleAdd(np.ascontiguousarray(x), float(k), img, dst=img)
    return img


def _cached_deck(PW, PH, hy, sun_p, shape):
    """Ray-march the deck (optionally cached on disk while iterating: env S10_CACHE=1)."""
    if not os.environ.get('S10_CACHE'):
        return K.deck_fields(PW, PH, hy, sun_p, **shape)
    import hashlib
    import pickle
    key = hashlib.md5(repr((PW, PH, round(hy, 3), tuple(round(v, 3) for v in sun_p), sorted(
        (k, repr(v)) for k, v in shape.items()))).encode()).hexdigest()[:12]
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out', 'cache')
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, f's10deck_{key}.pkl')
    if os.path.exists(fn):
        with open(fn, 'rb') as fh:
            return pickle.load(fh)
    R = K.deck_fields(PW, PH, hy, sun_p, **shape)
    with open(fn, 'wb') as fh:
        pickle.dump(R, fh)
    return R


class Scene:
    def __init__(self, W, H):
        t0 = time.time()
        self.W, self.H = W, H
        sc = W / 1920.0
        self.sc = sc
        m = 0.11
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
        self.sun_p = sun_p
        self.build_sky()
        self.build_peaks()
        self.build_deck()
        self.ridge_Z = 6.0
        self.build_near()
        self.build_fg()
        self.build_fx()
        self.setup_s = time.time() - t0

    def build_sky(self):
        W, H, PW, PH, ox, oy, hy, f, sun_p = self.W, self.H, self.PW, self.PH, self.ox, self.oy, self.hy, self.f, self.sun_p
        self.sky = SM.sky_plate(W, H, self.hz * H, self.sun0)
        # ---- altocumulus streets, lit gold / pink on the sun side
        # mackerel sky: rows of small broken cloudlets on a high plane, the rows receding toward the sun's
        # azimuth (they converge on it in perspective); pink / gold undersides, cool lilac tops
        az = math.degrees(math.atan2(f, (sun_p[0] - PW / 2)))
        # two interleaved layers (different cloudlet sizes / seeds / patchiness) so the cloudlets vary in size
        # and gather into curved, meandering mackerel bands that compress toward the horizon
        # Two palettes per layer (same cloudlets): hot gold near the sun, cool pink / violet undersides far
        # from it, blended by the distance to the sun so the sun area holds the value focus.
        warm = dict(lit=(1.7, 1.0, 0.52), body=(0.95, 0.56, 0.5), under=(0.62, 0.36, 0.5), haze_col=(1.1, 0.7, 0.5))
        cool = dict(lit=(1.15, 0.68, 0.76), body=(0.58, 0.44, 0.7), under=(0.36, 0.27, 0.58), haze_col=(0.85, 0.55, 0.62))
        fk = dict(fov=FOV, alt=3.2, cell_stretch=0.8, warp=1.7, angle=az, region=(0.0, 0.9), z_haze=45.0,
                  a_lo=0.12, a_hi=0.34, lumps=1.0, lit_screen=(0.0, 4.0), opacity=0.95, haze_fade=0.35,
                  veil=0.12, edge_px=1.1)
        l1 = dict(seed=21, cell=0.26, row_l=0.55, patch=3.2, patch_thr=-0.12, cover=0.55)
        l2 = dict(seed=33, cell=0.5, row_l=0.9, patch=4.5, patch_thr=-0.1, cover=0.55, angle=az + 12, warp=2.2)
        A1w = K.sky_rows_plate(PW, PH, hy, **dict(fk, **l1, **warm))
        A1c = K.sky_rows_plate(PW, PH, hy, **dict(fk, **l1, **cool))
        A2w = K.sky_rows_plate(PW, PH, hy, **dict(fk, **l2, **warm))
        A2c = K.sky_rows_plate(PW, PH, hy, **dict(fk, **l2, edge_px=2.2, **cool))
        yy_, xx_ = np.mgrid[0:PH, 0:PW].astype(F32)
        dsn = np.sqrt(((xx_ - sun_p[0]) / W) ** 2 + ((yy_ - sun_p[1]) / W * 1.4) ** 2)
        pr = np.exp(-(dsn / 0.3) ** 2)[..., None]
        # the cloudlets gather into 3 drifting bands (on the sky plane, so they compress toward the horizon)
        # with gaps of clean gradient between them; bigger and softer near the top, fine near the horizon
        dyp = np.maximum(hy - yy_, 1.0)
        Zs = 3.2 * f / dyp
        Xs = (xx_ - PW / 2) / f * Zs
        wn = K._noise(PW, PH, 3.0, 61, 3, stretch=2.5, angle=-8)
        q = np.log(np.maximum(Zs + 0.28 * Xs, 4.0)) + 0.16 * wn
        en = K._noise(PW, PH, 9.0, 62, 3, stretch=2.0, angle=-8)
        band = np.zeros_like(q)
        for (c_, hw_, amp_) in ((2.72, 0.3, 1.0), (3.34, 0.2, 1.0), (3.98, 0.19, 0.9), (4.65, 0.3, 0.5)):
            band = np.maximum(band, amp_ * C.smoothstep(hw_ * (1.0 + 0.5 * en), hw_ * 0.35, np.abs(q - c_)))
        band = band[..., None]
        blend = []
        for (Aw, Ac) in ((A1w, A1c), (A2w, A2c)):
            rgb_ = Ac[..., :3] * (1 - pr) + Aw[..., :3] * pr
            # value focus: far from the sun the cloudlets sit lower in value (cooler, dimmer)
            rgb_ = rgb_ * (0.88 + 0.12 * pr)
            blend.append((rgb_, Aw[..., 3:] * band))
        (r1, a1), (r2, a2) = blend
        a1 = a1 * (0.35 + 0.65 * C.smoothstep(hy - 0.02 * H, hy - 0.2 * H, yy_[..., None]))   # sparser at the horizon
        a2 = a2 * (0.4 + 0.6 * C.smoothstep(hy - 0.08 * H, hy - 0.35 * H, yy_[..., None]))   # big ones up high
        ao = a1 + a2 * (1 - a1)
        rgb = (r1 * a1 + r2 * a2 * (1 - a1)) / np.maximum(ao, 1e-4)
        self.flecks = np.dstack([rgb, ao]).astype(F32)

    def base_at(self, Z):
        return self.hy + self.f * 1.0 / Z

    def build_peaks(self):
        W, H, PW, PH, ox, oy, hy, f, sun_p, sc = (self.W, self.H, self.PW, self.PH, self.ox, self.oy, self.hy,
                                                  self.f, self.sun_p, self.sc)
        # ---- far peaks standing in the cloud sea
        def base_at(Z):
            return hy + f * 1.0 / Z
        self.peaks = []
        pk = [  # (x0, x1 frame frac, Z, height units, peaks, seed, colour, lit flank, haze)
            (-0.22, 0.26, 26.0, 2.25, [(0.52, 1.0, 0.62), (0.8, 0.62, 0.3), (0.22, 0.66, 0.4)], 4,
             (0.22, 0.2, 0.46), (0.4, 0.29, 0.52), (0.62, 0.43, 0.62)),
            (0.7, 1.16, 34.0, 1.8, [(0.3, 0.8, 0.5), (0.62, 1.0, 0.45)], 6,
             (0.27, 0.24, 0.5), (0.44, 0.32, 0.54), (0.68, 0.48, 0.64)),
            (0.38, 0.58, 55.0, 2.0, [(0.42, 1.0, 0.6), (0.75, 0.75, 0.45)], 8,
             (0.4, 0.31, 0.55), (0.6, 0.4, 0.55), (0.85, 0.58, 0.62)),
        ]
        for (a, b, Z, hu, pks, sd, c0, c1, hc) in pk:
            by = base_at(Z) + 0.35 * f / Z
            # the big left massif is far off the sun axis: hazed toward the horizon colour, a faint rim only
            left = (b < 0.4)
            P = SM.far_peak_plate(PW, PH, ox + a * W, ox + b * W, by, hu * f / Z, pks, sd, sun_p, sc,
                                  col=c0, lit=c1, haze_col=(0.74, 0.52, 0.64) if left else hc,
                                  rim_amt=0.0 if left else (0.9 if Z < 50 else 0.6),
                                  aerial=0.42 if left else 0.0, face_blur=40.0 if left else 2.0)
            self.peaks.append((P, Z))
        self.pk = pk

    def build_deck(self):
        W, H, PW, PH, ox, oy, hy, f, sun_p, sc = (self.W, self.H, self.PW, self.PH, self.ox, self.oy, self.hy,
                                                  self.f, self.sun_p, self.sc)
        pk = self.pk
        # ---- deck
        cols = []            # no cloud 'towers' at the peaks: the painted silhouettes stand in the deck
        # broad banks (big lobes, deep gaps) that compress into thin rows toward the horizon
        # lobed banks with varied spacing: big banks (bank) carry scalloped heads (p2 / p3), separated by
        # deep valleys (gap), compressing into thin gold-rimmed rows toward the horizon
        dk_shape = dict(fov=FOV, seed=3, towers=cols, stretch=1.6, bank=(0.9, 16.0, 5.0), p1=(0.35, 5.0),
                        p2=(0.2, 1.8), p3=(0.1, 0.55), p4=(0.035, 0.16), gap=(0.9, 9.0), amb_k=6.0,
                        soft=0.015)
        self.deck_R = _cached_deck(PW, PH, hy, sun_p, dk_shape)
        dp = dict(deep=(0.01, 0.01, 0.06), body=(0.2, 0.14, 0.4), near_dark=0.7, near_rim=0.85, near_z=(1.8, 12.0),
                  haze_far=(0.72, 0.52, 0.8), haze_sun=(1.25, 0.76, 0.46), haze_amt=0.38, z_haze=60.0,
                  shadow_col=(0.05, 0.04, 0.18), near_soft=0.0, kuwa=2, rim_focus=0.85, k_sun=0.42,
                  crown_col=(0.9, 0.46, 0.5), crown_amt=0.55, crown_px=30.0, valley_amt=0.8,
                  valley_col=(0.03, 0.025, 0.12), under_col=(0.1, 0.08, 0.3), under_amt=0.4, patches=0.5)
        full = K.paint_deck(self.deck_R, PW, PH, hy, sun_p[0], **dp)
        # the lit crest lines as broken, dashed sun-facing highlight fragments (not continuous contour
        # lines) that thin out with distance: base deck with very little rim + a dashed rim layer
        base = K.paint_deck(self.deck_R, PW, PH, hy, sun_p[0], **dict(dp, k_sun=0.42 * 0.3))
        x0, y0, x1, y1 = self.deck_R['box']
        bw_, bh_ = x1 - x0, y1 - y0
        dsh = K._noise(bw_, bh_, max(bw_ / (26.0 * sc), 4), 91, 3, stretch=6.0, angle=-2)
        dsh2 = K._noise(bw_, bh_, max(bw_ / (70.0 * sc), 3), 92, 2, stretch=4.0)
        Zd = self.deck_R['Z']
        xs_ = (np.arange(x0, x1, dtype=F32)[None, :] - sun_p[0]) / W
        near_sun = np.exp(-(xs_ / 0.18) ** 2)
        # far (large Z) -> sparser, thinner fragments; the sun column keeps more of them
        thr = -0.35 + 0.55 * C.smoothstep(8.0, 60.0, Zd) - 0.35 * near_sun
        keep = C.smoothstep(thr, thr + 0.12, dsh + 0.35 * dsh2)
        keep = np.clip(keep * (1.0 - 0.45 * C.smoothstep(15.0, 80.0, Zd)) + 0.12 * near_sun, 0, 1)
        # the crest light belongs to the sun column: it falls off fast into the cool left third and more
        # slowly to the right (no uniform rim on every row)
        sw_ = np.exp(-(xs_ / np.where(xs_ < 0, 0.2, 0.32)) ** 2)
        keep = keep * sw_
        self.deck = (base + np.clip(full - base, 0, None) * keep[..., None]).astype(F32)
        dp_hot = dict(dp, k_sun=0.42 * 1.5, near_rim=1.0, rim_focus=0.25)
        hot = np.clip(K.paint_deck(self.deck_R, PW, PH, hy, sun_p[0], **dp_hot) - full, 0, None) * keep[..., None]
        self.deck_rgba = np.zeros((PH, PW, 4), F32)
        self.deck_rgba[y0:y1, x0:x1, :3] = self.deck
        self.deck_rgba[y0:y1, x0:x1, 3] = 1.0
        self.deck_rgba[:y0 + 3, :, 3] = 0.0
        # the deck seen between the nearest banks (below the summit lip) lies in their shadow: deep indigo
        yv = np.arange(PH, dtype=F32)[:, None, None]
        yf = C.smoothstep(oy + 0.6 * H, oy + 0.78 * H, yv)
        self.deck_rgba[..., :3] = self.deck_rgba[..., :3] * (1 - 0.8 * yf) + np.array([0.04, 0.025, 0.14], F32) * 0.8 * yf
        self.deck_rgba[..., :3] = K._bleed(self.deck_rgba[..., :3], self.deck_rgba[..., 3], 3.0)
        self.deck_hot = np.zeros((PH, PW, 3), F32)
        self.deck_hot[y0:y1, x0:x1] = hot
        self.deck_hot[:y0 + 3] = 0.0
        self.deck_hot *= (1 - yf)
        Zm = np.full((PH, PW), 200.0, F32)
        Zm[y0:y1, x0:x1] = np.where(self.deck_R['A'] > 0.3, self.deck_R['Z'], 200.0)
        self.deckZ = Zm

    def build_ridge(self):
        W, H, PW, PH, ox, oy, hy, f, sun_p, sc = (self.W, self.H, self.PW, self.PH, self.ox, self.oy, self.hy,
                                                  self.f, self.sun_p, self.sc)
        base_at = self.base_at
        # ---- forested ridge rising out of the clouds (right of centre)
        Zr = 6.0
        self.ridge_Z = Zr
        rx = ox + np.array([-0.5, 0.5, 0.58, 0.66, 0.74, 0.8, 0.86, 0.95, 1.1, 1.5]) * W
        ry = oy + np.array([0.9, 0.9, 0.66, 0.61, 0.575, 0.555, 0.56, 0.585, 0.62, 0.62]) * H
        self.ridge = SM.ridge_plate(PW, PH, rx, ry, base_at(Zr) + 0.5 * f / Zr, 12, sun_p, sc)

    def build_near(self):
        W, H, PW, PH, ox, oy, hy, f, sun_p, sc = (self.W, self.H, self.PW, self.PH, self.ox, self.oy, self.hy,
                                                  self.f, self.sun_p, self.sc)
        # the painted cloud floor: rows in perspective (scalloped heads growing toward the camera), split
        # into three depth groups for parallax (far -> near)
        fl = FL.Floor(PW, PH, W, hy, sun_p, sc, y_start=hy + 0.11 * H, seed=7)
        self.floor = []
        for (P, Rr, q, _rows) in fl.paint(groups=((0.0, 0.14), (0.14, 0.34), (0.34, 0.62), (0.62, 1.01))):
            # physical parallax of the group's mean row line (depth = DK / Z, Z = f / (y - horizon))
            ya = P[..., 3].sum(1)
            ym = float((np.arange(PH) * ya).sum() / max(ya.sum(), 1e-3))
            dep = float(np.clip(DK * 1.5 * (ym - hy) / f, 0.08, 0.7))
            self.floor.append((P, Rr, dep, 0.0004 + 0.0075 * dep))
        # far-left cloud bank in front of the big massif's base (lit top edge, lost base)
        fb = FL.Floor(PW, PH, W, hy, sun_p, sc, y_start=hy + 0.11 * H, seed=19)
        self.bank = fb.bank(ox - 0.04 * W, ox + 0.36 * W, oy + 0.515 * H, 16.0, 34.0, 0.05, seed=23, tint=(0.86, 0.84, 0.9), lit_col=(0.8, 0.6, 0.66))
        yb_ = oy + 0.5 * H
        self.bank_dep = float(np.clip(DK * 1.5 * (yb_ - hy) / f, 0.05, 0.7))
        self.wisp = SM.wisp_plate(PW, PH, ox + 0.5 * W, PW, oy + 0.7 * H, 0.03 * H, 41, sun_p, sc)
        # mist streaks drifting over the near clouds (between ridge and summit)
        ys = np.arange(PH, dtype=F32)[:, None] / PH
        n2 = K._noise(PW, PH, 4.0, 78, 4, stretch=9.0, angle=-3)
        band = SM.ss(0.52, 0.62, ys) * (1 - SM.ss(0.8, 0.9, ys))
        mist = np.clip(n2 * 1.6 - 0.3, 0, 1) * band * 0.05
        xs_ = (np.arange(PW, dtype=F32)[None, :] - sun_p[0]) / W
        warm = np.exp(-(xs_ / 0.3) ** 2)[..., None]
        mcol = np.array([0.95, 0.66, 0.72], F32) * (1 - warm) + np.array([1.35, 0.95, 0.55], F32) * warm
        self.mist = (mcol * mist[..., None]).astype(F32)          # premultiplied, additive

    def build_fg(self):
        W, H, PW, PH, ox, oy, sun_p = self.W, self.H, self.PW, self.PH, self.ox, self.oy, self.sun_p
        sm = SM.Summit(PW, PH, W, H, ox, oy, sun_p)
        self.fgS, self.fgD, self.fgW, self.fgRs, self.fgRd = sm.build()
        self._grid = np.mgrid[0:H, 0:W].astype(F32)
        self._pgrid = np.mgrid[0:PH, 0:PW].astype(F32)

    def build_fx(self):
        W, H = self.W, self.H
        self.flare = K.SunFlare(W, H, tint=(1.0, 0.8, 0.55), star_len=0.075, bloom=0.2, ring=0.0, chain=0.0,
                                vertical=0.0)
        # horizon: thin white-hot line + narrow warm bloom
        Y = np.arange(H, dtype=F32)[:, None]
        X = np.arange(W, dtype=F32)[None, :]
        d = (Y - self.hz * H) / H
        # thin white-gold line with a hot core under the sun + a narrow warm skirt (yn_02)
        wide = np.exp(-(d / 0.009) ** 2) * (0.15 + 0.85 * np.exp(-((X - self.sun0[0]) / (0.22 * W)) ** 2))
        line = np.exp(-(d / 0.0011) ** 2) * (0.35 + 0.65 * np.exp(-((X - self.sun0[0]) / (0.3 * W)) ** 2))
        core = np.exp(-(d / 0.0016) ** 2) * np.exp(-((X - self.sun0[0]) / (0.07 * W)) ** 2)
        # thin bright horizon cloud band (yn_02): stacked hairline streaks of lit cloud tops just below the
        # horizon line, compressed by perspective, brightest under the sun and broken along their length
        nb = K._noise(W, H, max(W / (40.0 * self.sc), 4), 95, 3, stretch=14.0)
        nb2 = K._noise(W, H, max(W / (160.0 * self.sc), 3), 96, 2, stretch=6.0)
        dd = np.maximum(d, 0.0)
        u_ = np.sqrt(dd * H / (0.05 * H))                      # perspective-compressed row coordinate
        rows_ = (0.5 + 0.5 * np.cos(u_ * 2 * math.pi * 3.2 + 0.8 * nb2)) ** 6
        band_ = rows_ * C.smoothstep(-0.2, 0.35, nb) * C.smoothstep(0.0, 0.004, d) * C.smoothstep(0.075, 0.01, d)
        band_ = band_ * (0.25 + 0.75 * np.exp(-((X - self.sun0[0]) / (0.3 * W)) ** 2))
        self.hband = (band_[..., None] * np.array([1.0, 0.82, 0.62], F32) * 0.3).astype(F32)
        self.hglow = (wide[..., None] * np.array([1.0, 0.62, 0.3], F32) * 0.18 +
                      line[..., None] * np.array([1.0, 0.93, 0.78], F32) * 0.95 +
                      core[..., None] * np.array([1.0, 0.95, 0.85], F32) * 0.9).astype(F32)
        # sun star: long thin rays (vertical / horizontal longest, diagonals shorter), cached 2x sprite
        cw, ch = 2 * W, 2 * H
        yy_, xx_ = np.mgrid[0:ch, 0:cw].astype(F32)
        dx_, dy_ = (xx_ - W) / W, (yy_ - H) / W
        r_ = np.sqrt(dx_ * dx_ + dy_ * dy_) + 1e-6
        th_ = np.arctan2(dy_, dx_)
        star = np.zeros((ch, cw), F32)
        for k, (ang, ln, wd) in enumerate([(0.0, 0.16, 0.0011), (math.pi / 2, 0.24, 0.0011),
                                           (math.pi / 4, 0.12, 0.0009), (-math.pi / 4, 0.12, 0.0009)]):
            perp = np.abs(dx_ * math.sin(ang) - dy_ * math.cos(ang))
            along = np.abs(dx_ * math.cos(ang) + dy_ * math.sin(ang))
            wdt = wd * (1 + along / 0.05)
            star += np.exp(-(perp / wdt) ** 2) * np.exp(-along / (ln * 0.35)) * (1 / (1 + along / 0.01))
        glow_ = np.exp(-r_ / 0.007) * 0.9 + np.exp(-r_ / 0.025) * 0.12 + np.exp(-r_ / 0.08) * 0.03
        # anamorphic horizontal streak
        beads = 0.35 + 0.65 * (0.5 + 0.5 * np.cos(dx_ / 0.022 * 2 * math.pi)) ** 10 * (np.abs(dx_) > 0.03)
        glow_ = glow_ + np.exp(-(dy_ / 0.0011) ** 2) * (np.exp(-np.abs(dx_) / 0.2) * 0.3 * beads +
                                                      np.exp(-np.abs(dx_) / 0.08) * 0.4)
        glow_ = glow_ + np.exp(-(dy_ / 0.005) ** 2) * np.exp(-np.abs(dx_) / 0.3) * 0.08
        self.star = (star[..., None] * np.array([1.0, 0.9, 0.72], F32) * 2.2 +
                     glow_[..., None] * np.array([1.0, 0.78, 0.5], F32)).astype(F32)
        rng = np.random.default_rng(17)
        gcols = [np.array(c, F32) for c in ((1.0, 0.72, 0.4), (1.0, 0.5, 0.75), (0.5, 0.9, 1.0), (0.9, 1.0, 0.6),
                                            (1.0, 0.85, 0.6))]
        self.ghost_spec = []
        # ONE restrained ghost chain: a few small soft discs along the sun -> centre axis (no hexagons)
        # two small beads close to the sun only - nothing lands on the torii / marker / pine
        for i, k in enumerate((0.3, 0.52)):
            R = W * rng.uniform(0.0035, 0.006)
            a = 0.1 * rng.uniform(0.8, 1.1)
            self.ghost_spec.append((float(k), R, a, gcols[i % 5], False))

        self.build_grade()

    def build_grade(self):
        """Screen-space warm / cool split over the cloud sea (applied before the foreground): teal-blue in the
        far left shadows, pinkish gold along the sun path, the front row pushed darkest and coolest."""
        W, H = self.W, self.H
        yy, xx = np.mgrid[0:H, 0:W].astype(F32)
        sx = self.sun0[0]
        hzy = self.hz * H
        below = C.smoothstep(hzy, hzy + 0.01 * H, yy)
        k_t = np.exp(-(((xx - 0.12 * W) / (0.3 * W)) ** 2 + ((yy - 0.62 * H) / (0.17 * H)) ** 2))
        k_t = np.maximum(k_t, 0.6 * np.exp(-(((xx - 0.98 * W) / (0.18 * W)) ** 2 + ((yy - 0.58 * H) / (0.12 * H)) ** 2)))
        k_w = np.exp(-(((xx - sx - 0.02 * W) / (0.17 * W)) ** 2 + ((yy - 0.5 * H) / (0.085 * H)) ** 2))
        # the sun column carries the warmth down the deck (pink -> peach gold), fading toward the front
        k_w = np.maximum(k_w, 0.6 * np.exp(-(((xx - sx) / (0.12 * W)) ** 2 + ((yy - 0.64 * H) / (0.12 * H)) ** 2)))
        k_f = C.smoothstep(0.7 * H, 0.93 * H, yy)
        prox = np.exp(-((xx - sx) / (0.3 * W)) ** 2)
        self.g_t = (k_t * below).astype(F32)
        self.g_w = (k_w * below).astype(F32)
        self.g_f = k_f.astype(F32)
        self.g_c = ((1 - prox) * below * (1 - k_f)).astype(F32)

    def _grade(self, img):
        lum = img.mean(-1)
        shadow = C.smoothstep(0.5, 0.08, lum)
        mid = C.smoothstep(0.05, 0.25, lum) * C.smoothstep(1.2, 0.5, lum)
        kt = (self.g_t * shadow)[..., None]
        img = img * (1 - kt * np.array([0.32, 0.02, -0.04], F32)) + kt * np.array([0.0, 0.018, 0.03], F32)
        kw = (self.g_w * mid)[..., None]
        img = img * (1 + kw * np.array([0.22, 0.05, -0.16], F32))
        kf = self.g_f[..., None]
        img = img * (1 - kf * np.array([0.34, 0.3, 0.12], F32))
        kc = (self.g_c * (0.55 + 0.45 * shadow))[..., None]
        img = img * (1 - kc * np.array([0.12, 0.03, -0.04], F32))
        return img.astype(F32)

    # ------------------------------------------------------------------------------------------------
    def _warp_deck(self, cam, zoom, want_hot=False, t=0.0):
        W, H = self.W, self.H
        yy, xx = self._grid
        d0 = 0.2
        z0 = 1 + (zoom - 1) * d0
        px = (xx - W / 2) / z0 + self.PW / 2 + cam[0] * d0
        py = (yy - H / 2) / z0 + self.PH / 2 + cam[1] * d0
        Zs = cv2.remap(self.deckZ, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dep = np.clip(DK / np.maximum(Zs, 0.5), 0.0, 1.2)
        z = 1 + (zoom - 1) * dep
        # slow drift of the cloud sea (rate grows with nearness: per-depth parallax drift)
        px = (xx - W / 2) / z + self.PW / 2 + cam[0] * dep + 0.006 * W * t * dep
        py = (yy - H / 2) / z + self.PH / 2 + cam[1] * dep
        rgba = cv2.remap(self.deck_rgba, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        Zw = cv2.remap(self.deckZ, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        hot = cv2.remap(self.deck_hot, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if want_hot else None
        return rgba, Zw, hot

    def _bloom(self, img, thr, strength, halation, occ=None):
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
        if occ is not None:
            # silhouettes: keep the halation from outlining the dark foreground on every side
            oq = cv2.resize(occ, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
            hal = hal * (1 - 0.97 * oq)[..., None]
            acc = acc * (1 - 0.9 * oq)[..., None]
        add = cv2.resize(acc * strength + hal, (W, H), interpolation=cv2.INTER_LINEAR)
        img += add
        return img

    def _atmo(self, sx, sy):
        if not hasattr(self, '_atm'):
            W, H = self.W, self.H
            yy, xx = np.mgrid[0:2 * H, 0:2 * W].astype(F32)
            d = np.sqrt((xx - W) ** 2 + ((yy - H) * 1.6) ** 2) / W
            g = np.exp(-d / 0.035) * 0.14 + np.exp(-d / 0.13) * 0.055 + np.exp(-d / 0.45) * 0.03
            self._atm = (g[..., None] * np.array([1.0, 0.62, 0.3], F32)).astype(F32)
        M = np.array([[1, 0, sx - self.W], [0, 1, sy - self.H]], np.float32)
        return cv2.warpAffine(self._atm, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _ghosts(self, sx, sy, cx, cy, amt):
        """Lens ghost chain along the axis sun -> optical centre and beyond (yn_02): small hot beads near
        the sun, larger soft discs / hexagons further out, gold / magenta / teal tints. Additive."""
        W, H = self.W, self.H
        out = np.zeros((H, W, 3), F32)
        vx, vy = cx - sx, cy - sy
        for (k, R, a, col, hexa) in self.ghost_spec:
            gx, gy = sx + vx * k, sy + vy * k
            b0x, b1x = int(max(gx - R * 1.6, 0)), int(min(gx + R * 1.6 + 1, W))
            b0y, b1y = int(max(gy - R * 1.6, 0)), int(min(gy + R * 1.6 + 1, H))
            if b1x <= b0x or b1y <= b0y:
                continue
            ey, ex = np.mgrid[b0y:b1y, b0x:b1x].astype(F32)
            ex, ey = ex - gx, ey - gy
            rad = np.sqrt(ex * ex + ey * ey)
            if hexa:
                th = np.arctan2(ey, ex) + 0.3
                seg = 2 * math.pi / 6
                rad = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
            q = rad / R
            disc = C.smoothstep(1.05, 0.8, q) * (0.45 + 0.55 * C.smoothstep(0.2, 1.0, q)) +                 0.35 * np.exp(-((q - 1.0) / 0.25) ** 2) * (q < 1.6)
            out[b0y:b1y, b0x:b1x] += disc[..., None] * col * a
        return out * amt

    def _fan(self, sx, sy):
        """Painted crepuscular wedges radiating from the sun and fanning DOWN across the cloud sea: broad
        soft light wedges alternating with cooler shadow wedges (cast by unseen towers on the horizon).
        Returns (additive light, shadow factor); cached on a 2x canvas and translated with the sun."""
        if not hasattr(self, '_fan_c'):
            W, H = self.W, self.H
            yy, xx = np.mgrid[0:2 * H, 0:2 * W].astype(F32)
            dx, dy = (xx - W) / W, (yy - H) / W
            r = np.sqrt(dx * dx + dy * dy) + 1e-6
            th = np.arctan2(dy, dx)
            rng = np.random.default_rng(12)
            n = 1440
            k = np.arange(n)
            lit = np.zeros(n, F32)
            shd = np.zeros(n, F32)
            # wedges only in the lower half-plane (angles 0..pi = down), a few reaching up faintly
            for _ in range(11):
                c = rng.uniform(0.08, 0.92) * n / 2
                wd = rng.uniform(6, 22)
                d = np.abs(k - c)
                lit += (np.exp(-(d / wd) ** 4) * rng.uniform(0.5, 1.0)).astype(F32)
            for _ in range(7):
                c = rng.uniform(0.1, 0.9) * n / 2
                wd = rng.uniform(4, 14)
                d = np.abs(k - c)
                shd += (np.exp(-(d / wd) ** 4) * rng.uniform(0.5, 1.0)).astype(F32)
            lit = np.clip(lit, 0, 1.1)
            shd = np.clip(shd, 0, 1) * (1 - np.clip(lit, 0, 1))
            idx = ((th % (2 * math.pi)) / (2 * math.pi) * n).astype(np.int32) % n
            # soft painted edges: blur the angular profile a little in screen space
            down = C.smoothstep(-0.01, 0.06, dy)
            fall = (np.exp(-r / 0.55) * 0.8 + 0.2 * np.exp(-r / 1.2)) * C.smoothstep(0.015, 0.09, r)
            g = lit[idx] * down * fall
            sh = shd[idx] * down * C.smoothstep(0.05, 0.25, r) * np.exp(-r / 0.9)
            g = cv2.GaussianBlur(g.astype(F32), (0, 0), 3.0 * W / 1920)
            sh = cv2.GaussianBlur(sh.astype(F32), (0, 0), 5.0 * W / 1920)
            self._fan_c = (g[..., None] * np.array([1.0, 0.7, 0.42], F32) * 0.3).astype(F32)
            self._fan_s = sh.astype(F32)
        M = np.array([[1, 0, sx - self.W], [0, 1, sy - self.H]], np.float32)
        a = cv2.warpAffine(self._fan_c, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        s_ = cv2.warpAffine(self._fan_s, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return a, s_

    def _star(self, sx, sy, rot):
        c, s_ = math.cos(rot), math.sin(rot)
        W, H = self.W, self.H
        M = np.array([[c, -s_, sx - (c * W - s_ * H)], [s_, c, sy - (s_ * W + c * H)]], np.float32)
        return cv2.warpAffine(self.star, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _sway(self, t, plates):
        """Grass / needles / papers: smooth travelling gusts, displacement scaled by the sway weight."""
        PH, PW = self.PH, self.PW
        yy, xx = self._pgrid
        sc = self.sc
        g = 0.6 + 0.4 * math.sin(0.9 * t + 0.4)
        ph = np.arange(PW, dtype=F32) / (260 * sc) - 1.7 * t
        wave = ((np.sin(ph) * 0.7 + 0.3 * np.sin(2.3 * ph + 1.1 + 0.8 * t)) * (g * 3.2 * sc)).astype(F32)
        dx = self.fgW * wave[None, :]
        dy = np.abs(dx)
        dy *= 0.25
        dx = xx - dx
        dy += yy
        return [cv2.remap(P, dx, dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) for P in plates]

    def _rows(self, plate):
        """Plate rows holding any alpha (cached per plate object)."""
        if not hasattr(self, '_row_cache'):
            self._row_cache = {}
        k = id(plate)
        if k not in self._row_cache:
            r = np.nonzero(plate[..., 3].max(1) > 1e-4)[0]
            self._row_cache[k] = (int(r[0]), int(r[-1]) + 1) if len(r) else (0, 0)
        return self._row_cache[k]

    def _band(self, plate, t, speed, cam, zoom, depth, rows_of=None):
        """drift() restricted to the frame rows the plate's content can reach: returns (y0, sub-layer)."""
        W, H = self.W, self.H
        r0, r1 = self._rows(plate if rows_of is None else rows_of)
        h, w = plate.shape[:2]
        z = 1 + (zoom - 1) * depth
        cx = w / 2 - speed * W * t + cam[0] * depth
        cy = h / 2 + cam[1] * depth
        y0 = int(max(math.floor(H / 2 + (r0 - cy) * z) - 2, 0))
        y1 = int(min(math.ceil(H / 2 + (r1 - cy) * z) + 2, H))
        if y1 <= y0:
            return 0, None
        M = np.array([[z, 0, W / 2 - cx * z], [0, z, H / 2 - cy * z - y0]], np.float32)
        L = cv2.warpAffine(plate, M, (W, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))
        return y0, L

    def _floor_comp(self, img, P, Rr, t, spd, cam, zoom, dep, light):
        y0, L = self._band(P, t, spd, cam, zoom, dep)
        if L is None:
            return img
        img = self._comp(img, y0, L)
        y0r, Lr = self._band(Rr, t, spd, cam, zoom, dep, rows_of=P)
        _add(img[y0r:y0r + Lr.shape[0]], Lr, 0.75 + 0.35 * light)
        return img

    def _comp(self, img, y0, L):
        if L is not None:
            img[y0:y0 + L.shape[0]] = K.composite(np.ascontiguousarray(img[y0:y0 + L.shape[0]]), L)
        return img

    def frame(self, t):
        W, H = self.W, self.H
        u = t / DURATION
        # eased start, but the move is still travelling at the end (it keeps running under the fade-out)
        p = C.ease_in_out_sine(u * 0.82) / C.ease_in_out_sine(0.82)
        zoom = 1.0 + 0.07 * p
        # truck right + crane up (cresting the summit): the foreground drops and slides against the clouds
        # eased crane-up + truck left toward the torii (+ push): the summit (depth 1) travels ~190 px,
        # the near cloud floor ~1/3 of that, the horizon band stays put
        cam = (-0.085 * W * p, -0.105 * H * p)
        light = 0.78 + 0.27 * C.ease_in_out_sine(min(u * 1.1, 1.0))       # light swells
        e = C.smoothstep(0.7, 1.0, u)                                      # final swell
        e2 = C.smoothstep(0.75, 1.0, u)                                    # triumphant peak (last 1.5 s)
        # sun climbs slowly (fixed at infinity: tiny parallax)
        sx = self.sun0[0]
        sy = self.sun0[1] - 0.024 * H * p
        img = self.sky * np.float32(0.95 + 0.05 * light)
        _add(img, S.sun(W, H, sx, sy, radius=0.0065, color=(1.0, 0.82, 0.55), intensity=1.1 * light,
                        glow_size=0.8, core=2.6))
        img = self._comp(img, *self._band(self.flecks, t, 0.0025, cam, zoom, 0.04))
        _add(img, self.hglow, light)
        deck, Zw, dhot = self._warp_deck(cam, zoom, e > 0, t)
        if dhot is not None:
            deck[..., :3] += dhot * (0.5 * e)
        img = K.composite(img, deck)
        _add(img, self.hband, light)
        for P, Z in sorted(self.peaks, key=lambda q: -q[1]):
            y0, L = self._band(P, t, 0.0, cam, zoom, DK / Z)
            if L is not None:
                L[..., 3] *= np.clip((Zw[y0:y0 + L.shape[0]] - (Z - 1.5)) / 3.0, 0, 1)
                img = self._comp(img, y0, L)
        img = self._comp(img, *self._band(self.bank, t, 0.0004 + 0.0075 * self.bank_dep, cam, zoom, self.bank_dep))
        if self.floor:
            P, Rr, dep, spd = self.floor[0]
            img = self._floor_comp(img, P, Rr, t, spd, cam, zoom, dep, light)
        img = self._comp(img, *self._band(self.wisp, t, 0.003, cam, zoom, DK / (self.ridge_Z - 0.6)))
        for gi, (P, Rr, dep, spd) in enumerate(self.floor[1:]):
            img = self._floor_comp(img, P, Rr, t, spd, cam, zoom, dep, light)
        Lm = K.drift(self.mist, W, H, t, speed=0.004, cam=cam, zoom=zoom, depth=0.35)
        _add(img, Lm, 0.18 + 0.12 * light)
        img = self._grade(img)
        # god-ray fan spreading from the sun over the cloud sea (swells at the end; behind the summit)
        # the shafts breathe slowly (two incommensurate slow sines, smooth, no flicker)
        breath = 1.0 + 0.14 * math.sin(2 * math.pi * t / 3.3 + 0.7) + 0.07 * math.sin(2 * math.pi * t / 1.9 + 2.1)
        fa, fsh = self._fan(sx, sy)
        img *= (1.0 - (0.16 + 0.06 * e) * fsh)[..., None]
        _add(img, fa * (1.0 - 0.55 * self.g_f)[..., None], (0.5 + 0.3 * e) * light * breath)
        # foreground (+ its rim light, which swells with the light)
        D, Rd = self._sway(t, (self.fgD, self.fgRd))
        fs = K.drift(self.fgS, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=1.0)
        fd = K.drift(D, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=1.0)
        rs = K.drift(self.fgRs, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=1.0)
        rd = K.drift(Rd, W, H, t, speed=0.0, cam=cam, zoom=zoom, depth=1.0)
        atm = self._atmo(sx, sy)
        _add(img, atm, 0.55 * light ** 2 * (1 + 0.25 * e2))
        img = K.composite(img, fs, fd)
        rim_g = (0.8 + 0.25 * light) * (1.0 + 0.45 * e)
        _add(img, np.ascontiguousarray(rs[..., :3]), rim_g)
        _add(img, np.ascontiguousarray(rd[..., :3]), rim_g)
        _add(img, atm, 0.1 * light ** 2)            # veiling glare over the silhouettes near the sun
        occ = K.occluder(fs, fd)
        _add(img, F.light_shafts(W, H, sx, sy, occluder=occ, strength=(0.07 + 0.11 * e) * light * breath, length=0.95,
                                   tint=(1.0, 0.72, 0.45), radius=0.06, t=t))
        vis = F.sun_visibility(occ, sx, sy, 0.008 * W)
        shim = 1.0 + 0.04 * math.sin(1.9 * t) + 0.02 * math.sin(4.3 * t + 0.6)
        _add(img, self._star(sx, sy, 0.0), (0.5 + 0.5 * vis) * shim * light ** 1.5 * (1 + 0.15 * e2))
        cen = (W / 2 + cam[0] * 0.3, H / 2)
        _add(img, self.flare.render(sx, sy, intensity=(0.4 + 0.6 * vis) * shim * light * (1 + 0.05 * e + 0.1 * e2), t=t,
                                    center=cen))
        _add(img, self._ghosts(sx, sy, cen[0], cen[1], (0.5 + 0.5 * vis) * light * (1 + 0.4 * e)))
        img = self._bloom(img, 1.0 - 0.03 * e, 0.15 + 0.03 * e + 0.02 * e2, 0.12 + 0.05 * e + 0.08 * e2, occ=occ)
        img = F.shoulder(img, 0.88, 0.05)
        return F.finish_fast(img, t, exposure=0.97 + 0.06 * light, sat=1.06, grain_amt=0.004, vig=0.28, ca=0.0008)
