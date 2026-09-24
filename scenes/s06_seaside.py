"""s06_seaside - coastal road at sunset.

A two-lane road on a sea cliff curves away to the left toward a headland with a small white lighthouse;
a white W-beam guardrail runs along the sea side, utility poles and wires follow the hillside. The sun
hangs low over the sea, laying a glittering path of sparkles to the foot of the cliff; distant islands,
painted sunset cumulus (lib/clouds2) with gold linings and violet shadows. The camera trucks slowly
sideways (eased) past a near utility pole, a pipe fence and backlit pampas grass: every land element carries
its own inverse depth, so the plates are warped per pixel (true truck parallax: ground plane sheared, poles
/ rail / hill each at their own depth) while the sky and the islands stay put and a slight counter-pan pivots
the move about the middle distance.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import core as C, sky as S, fx as F, clouds2 as K  # noqa: E402
import s06_seaside_paint as P  # noqa: E402
import s06_seaside_land as L  # noqa: E402
import s06_seaside_sea as SEA  # noqa: E402
import s06_seaside_far as FAR  # noqa: E402
import s06_seaside_fx as FX  # noqa: E402
import s06_seaside_near as NEAR  # noqa: E402
import s06_seaside_sky as SKY  # noqa: E402
import s06_seaside_sky3 as SK2  # noqa: E402
import s06_seaside_wires as WI  # noqa: E402

DURATION = 5.0

HZ = 0.53        # horizon (fraction of H)
SUN = (0.655, 0.445)   # sun position in frame at cam=0 (fractions)
TRUCK = 1.4      # lateral camera travel over the shot (m)
PAN = 0.028      # counter-pan (fraction of W) - pivots the move around ~35 m
Z_FAR = 350.0    # headland / lighthouse distance (m)
# deeper sunset cumulus palette (clouds2 'sunset' pushed toward orange / rose, violet shadows)
SUNSET = dict(hi=(1.0, 0.84, 0.62), lit=(1.0, 0.64, 0.4), lit_lo=(0.96, 0.5, 0.5), mid='#d86a92', shade='#8270c2',
              deep='#40367e', refl='#a08ad0', bounce='#f09a86', rim=(1.4, 1.02, 0.62), haze='#e8a2b6',
              edge_dark=0.07)


def ease(x):
    """quintic smootherstep: a slow start and a soft landing, most of the travel mid-shot."""
    x = min(max(x, 0.0), 1.0)
    return 0.75 * x * x * x * (x * (6 * x - 15) + 10) + 0.25 * x


def cc(h):
    return C.hex2rgb(h)


def _over4(a, b):
    """straight-alpha RGBA b over a -> straight RGBA."""
    ab = b[..., 3:4]
    aa = a[..., 3:4]
    al = ab + aa * (1 - ab)
    rgb = (b[..., :3] * ab + a[..., :3] * aa * (1 - ab)) / np.maximum(al, 1e-5)
    return np.concatenate([rgb, al], -1).astype(np.float32)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.HZ = HZ
        self.pw, self.ph = pw, ph = int(W * 1.2), int(H * 1.08)
        self.ox, self.oy = ox, oy = (pw - W) / 2, (ph - H) / 2
        self.hzp = HZ * H + oy
        self.f = 0.95 * W
        self.hc = 5.0
        s = W / 1920.0
        self.s = s
        self.sun_p = (SUN[0] * W + ox, SUN[1] * H + oy)
        self._build_sky()
        self._build_sea_static()
        self.seaR = SEA.Sea(W, H, seed=3)
        self.sky_sun = self.sky + S.sun(self.pw, self.ph, self.sun_p[0], self.sun_p[1], radius=0.014 * W / self.pw,
                                        color=(1.0, 0.62, 0.3), core=1.25, intensity=0.62)
        self._build_far()
        self._build_land()

    # ------------------------------------------------------------------ projection (plate px)
    def proj(self, X, Y, Z):
        X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64); Z = np.asarray(Z, np.float64)
        x = self.ox + self.W / 2 + self.f * X / Z
        y = self.hzp + self.f * (self.hc - Y) / Z
        return x, y

    # ------------------------------------------------------------------ sky
    def _build_sky(self):
        W, H, pw, ph = self.W, self.H, self.pw, self.ph
        hzp = self.hzp / ph
        preset = dict(stops=[(0.0, '#1b2f86'), (0.2, '#2b48a8'), (0.4, '#5a62bc'), (0.56, '#9a74c0'),
                             (0.7, '#e880a4'), (0.82, '#ff9478'), (0.92, '#ffac66'), (1.0, '#ffc47e')],
                      sun_glow='#ffb05a', sun_glow_amt=0.15, below='#ffcc80', band=('#ffe0a0', 0.2))
        self.sky = S.sky_gradient(pw, ph, preset, horizon=hzp, sun=self.sun_p, sun_radius=0.5)
        sp = self.sun_p
        u = W / 1920.0
        # painted cumulus (lib/clouds2 Painter, mass by mass): the low sun lies between the two heaps, so
        # the left heap is lit gold on its lower-right flank, the right one on its lower-left flank
        ox, oy, Hs = self.ox, self.oy, H / 1080.0
        # painted sunset cumulus (lib/clouds2 Painter, mass by mass; s06_seaside_sky2)
        ox, oy, Hs = self.ox, self.oy, H / 1080.0
        self.cl_near = SK2.cumulus(pw, ph, sp, W, H, ox, oy)
        # thin horizon stratus: soft brushed streaks with torn ends and backlit gold undersides
        self.strat = SKY.streaks(pw, ph, sp, [(0.44 * W + ox, 0.78 * W + ox, 0.382 * H + oy, 11 * Hs, 5),
                                              (0.76 * W + ox, 1.04 * W + ox, 0.352 * H + oy, 9 * Hs, 6),
                                              (0.3 * W + ox, 0.53 * W + ox, 0.425 * H + oy, 8 * Hs, 7),
                                              (0.58 * W + ox, 0.72 * W + ox, 0.412 * H + oy, 5 * Hs, 8),
                                              (0.0 * W + ox, 0.22 * W + ox, 0.395 * H + oy, 7 * Hs, 9),
                                              (0.7 * W + ox, 0.9 * W + ox, 0.44 * H + oy, 5 * Hs, 10)], u)
        self.cl_far = self.strat
        self.bank = K.horizon_bank_plate(pw, ph, self.hzp + 0.002 * ph, 0.03 * ph, sun=sp, preset='sunset', seed=12,
                                         rows=3, haze=0.35, haze_color='#ffc890', sun_z=-0.05, shrink=0.5)
        self.far_sky = _over4(self.bank, self.cl_far)      # horizon bank + stratus drift together
        self.cirrus = K.cirrus_plate(pw, ph, preset='sunset', seed=4, region=(0.02, 0.3), angle=-4, density=0.45,
                                     opacity=0.45)

    # ------------------------------------------------------------------ sea (static parts)
    def _build_sea_static(self):
        W, H, pw, ph = self.W, self.H, self.pw, self.ph
        # reflection of the sky: mirror about the horizon, heavily blurred horizontally/vertically
        hz = int(self.hzp)
        ys = np.arange(ph, dtype=np.float32)
        yr = np.clip(2 * self.hzp - ys, 0, ph - 1)
        mapx = np.tile(np.arange(pw, dtype=np.float32)[None], (ph, 1))
        mapy = np.tile(yr[:, None], (1, pw)).astype(np.float32)
        comp = self.sky.copy()
        for pl in (self.bank, self.cl_far, self.cl_near):
            comp = F.over_rgba(comp, pl)
        refl = cv2.remap(comp, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        refl = cv2.GaussianBlur(refl, (0, 0), sigmaX=0.02 * W, sigmaY=0.01 * H)
        self.refl = refl.astype(np.float32)

    # ------------------------------------------------------------------ far: islands, headland, lighthouse
    def _build_far(self):
        fb = FAR.Far(self)
        self.islands = fb.islands()
        self.headland = fb.headland()
        self.head_refl = fb.reflection(self.headland)
        self.surf = fb.surf()
        self.lamp_p = fb.lamp
        self.farp = self.islands.copy()
        for pl in (self.head_refl, self.headland):
            a = pl[..., 3:4]
            rgb = self.farp[..., :3] * self.farp[..., 3:4] * (1 - a) + pl[..., :3] * a
            al = self.farp[..., 3:4] * (1 - a) + a
            self.farp = np.concatenate([rgb / np.maximum(al, 1e-4), al], -1).astype(np.float32)

    def _build_land(self):
        W, H = self.W, self.H
        self.landb = L.Land(self)
        d = self.landb.build()
        self.land = d
        nb = NEAR.Near(self)
        self.nearb = nb
        self.near = nb.build()
        self.rows = {k: self.rows_of(d[k], self.oy, H) for k in ('road', 'grass', 'rail', 'hill', 'poles')}
        self.rows.update({k: self.rows_of(self.near[k], 0, H) for k in ('back', 'front')})
        # compositing darkening of the lower-left foreground (frames the luminous background)
        xs_, ys_ = C.grid(self.W, self.H)
        dd = np.sqrt((xs_ / self.W / 0.75) ** 2 + ((self.H - ys_) / self.H / 0.75) ** 2)
        self.fg_dark = (1 - 0.3 * (1 - C.smoothstep(0.15, 1.0, dd)))[..., None].astype(np.float32)
        self.fg_dark *= (1 - 0.1 * C.smoothstep(0.8, 1.0, ys_ / self.H))[..., None]
        self.mote_r = np.random.default_rng(99).random((26, 7))
        self.bird_col = S.bird_color(self.sky[int(0.35 * self.ph), int(0.55 * self.pw)], darkness=0.7)
        self.flare = FX.Flare(self.W, self.H, tint=(1.0, 0.72, 0.45))
        self.flare.amts = [a * 2.3 for a in self.flare.amts]          # a readable (still subtle) ghost chain
        # upper-left sky mask for the wide shafts (above the horizon, left of the sun)
        xs_, ys_ = C.grid(W, H)
        sxm, sym = SUN[0] * W, SUN[1] * H
        self.shaft_mask = (C.smoothstep(HZ * H, 0.3 * H, ys_) * C.smoothstep(sxm + 0.05 * W, sxm - 0.25 * W, xs_)
                           )[..., None].astype(np.float32)
        # static paper / paint texture (screen space, identical every frame: no flicker)
        n1 = C.fbm(W, H, 90, 3, seed=711)
        n2 = C.fbm(W, H, 9, 3, seed=712)
        fib = cv2.GaussianBlur(np.random.default_rng(713).standard_normal((H, W)).astype(np.float32), (0, 0),
                               sigmaX=2.2 * W / 1920, sigmaY=0.5 * W / 1920)
        self.paper = (1 + 0.016 * (n1 - 0.5) + 0.012 * (n2 - 0.5) + 0.01 * fib)[..., None].astype(np.float32)

    # ------------------------------------------------------------------ camera
    def cam(self, t):
        """-> (pan px, K px*m, T m): a point at inverse depth iz is displaced by pan + K * iz."""
        u = ease(t / DURATION)
        T = (u - 0.5) * TRUCK
        return (u - 0.5) * PAN * self.W, -self.f * T, T

    @staticmethod
    def rows_of(plate, oy, H):
        """frame rows [r0, r1) where a plate has any coverage (no vertical camera motion)."""
        r = np.where(plate[..., 3].max(1) > 1e-3)[0]
        if len(r) == 0:
            return (0, 0)
        return (int(max(r[0] - oy - 2, 0)), int(min(r[-1] - oy + 3, H)))

    def comp(self, img, plate, iz, pan, K, ox, oy, rows, sway=None, t=0.0):
        """warp a plate (only its rows) and composite it over img in place."""
        r0, r1 = rows
        if r1 <= r0:
            return img
        if sway is None:
            L_ = self.warp(plate, iz, pan, K, ox, oy, r=(r0, r1))
        else:
            wmap, kw = sway
            L_, wm = self.warp(plate, iz, pan, K, ox, oy, extra=(wmap,), r=(r0, r1))
            L_ = self.sway(L_, wm, t, y0=r0, **kw)
        img[r0:r1] = F.over_rgba(img[r0:r1], L_)
        return img

    def warp(self, plate, iz, pan, K, ox, oy, extra=(), r=None):
        """per-pixel truck warp of a plate with its inverse-depth field (2 fixed-point iterations)."""
        W = self.W
        r0, r1 = (0, self.H) if r is None else r
        H = r1 - r0
        oxi, oyi = int(round(ox)), int(round(oy)) + r0
        xs = np.arange(W, dtype=np.float32)[None] + ox
        ys = np.ascontiguousarray(np.broadcast_to(np.arange(r0, r1, dtype=np.float32)[:, None] + oy, (H, W)))
        if np.isscalar(iz):
            mx = np.ascontiguousarray(np.broadcast_to(xs - pan - K * iz, (H, W)), np.float32)
        else:
            iz0 = iz[oyi:oyi + H, oxi:oxi + W]
            mx = (xs - pan - K * iz0).astype(np.float32)
            for _ in range(1):
                izs = cv2.remap(iz, mx, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                mx = (xs - pan - K * izs).astype(np.float32)
        out = [cv2.remap(p, mx, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
               for p in (plate,) + tuple(extra)]
        return out if extra else out[0]

    def sample(self, plate, shift, depth, zoom, dy=0.0):
        W, H = self.W, self.H
        z = 1 + (zoom - 1) * depth
        # plate px -> frame px: x_f = (x_p - ox + shift*depth - W/2) * z + W/2
        tx = (-self.ox + shift * depth - W / 2) * z + W / 2
        ty = (-self.oy + dy * depth - H / 2) * z + H / 2
        M = np.array([[z, 0, tx], [0, z, ty]], np.float32)
        bv = (0, 0, 0, 0) if plate.shape[2] == 4 else (0, 0, 0)
        return cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=bv)

    def to_frame(self, xp, yp, shift, depth, zoom):
        z = 1 + (zoom - 1) * depth
        return ((xp - self.ox + shift * depth - self.W / 2) * z + self.W / 2,
                (yp - self.oy - self.H / 2) * z + self.H / 2)

    def sway(self, rgba, wmap, t, amp=0.0045, freq=1.7, seed=0.0, y0=0):
        W, H = self.W, rgba.shape[0]
        if wmap.ndim == 3:
            wmap = wmap[..., 0]
        xs = np.arange(W, dtype=np.float32)[None]
        ys = np.arange(H, dtype=np.float32)[:, None]
        ph = t * freq + seed + xs * (6.0 / W) + 0.3 * np.sin(t * 0.6 + xs * (2.0 / W))
        gust = 0.6 + 0.4 * math.sin(t * 0.9 + 0.5 + seed)
        dx = (np.sin(ph) * 0.7 + 0.3 * np.sin(ph * 2.3 + 1.0)) * gust * amp * W
        w = np.clip(wmap, 0, 1) ** 1.5
        mx = xs - dx * w
        my = ys + np.abs(dx) * w * 0.15
        my = np.minimum(my, H - 1)
        return cv2.remap(rgba, mx.astype(np.float32), np.broadcast_to(my, mx.shape).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def halation_src(self, img):
        """bright-pass of the background at 1/4 res, spread wide (two radii), warm tinted -> full-res add."""
        W, H = self.W, self.H
        q = 4
        sm = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
        lum = sm.max(-1, keepdims=True)
        b = np.minimum(sm, 1.6) * C.smoothstep(1.05, 1.8, lum)
        u = W / 1920.0 / q
        h1 = cv2.GaussianBlur(b, (0, 0), 9 * u)
        h2 = cv2.GaussianBlur(b, (0, 0), 34 * u)
        h = (h1 * 0.13 + h2 * 0.2) * np.array([1.0, 0.74, 0.5], np.float32)
        return cv2.resize(h, (W, H), interpolation=cv2.INTER_LINEAR)

    def streak_at(self, sx, sy):
        W, H = self.W, self.H
        if not hasattr(self, '_streak'):
            u = W / 1920.0
            xs = np.arange(-W, W + 1, dtype=np.float32)[None]
            ys = np.arange(-40, 41, dtype=np.float32)[:, None] * u
            prof = np.exp(-(ys / (1.6 * u)) ** 2) * 0.7 + np.exp(-(ys / (6 * u)) ** 2) * 0.3
            fall = 0.6 * np.exp(-np.abs(xs) / (0.12 * W)) + 0.4 * np.exp(-np.abs(xs) / (0.4 * W))
            self._streak = (prof * fall)[..., None] * np.array([0.5, 0.34, 0.22], np.float32) * 0.8
            self._streak = self._streak.astype(np.float32)
        out = np.zeros((H, W, 3), np.float32)
        st = self._streak
        hh = st.shape[0] // 2
        yi = int(round(sy))
        x0 = W - sx      # column of st that lands on frame x=0
        xi = int(round(x0))
        y0, y1 = max(yi - hh, 0), min(yi + hh + 1, H)
        if y1 > y0:
            out[y0:y1] = st[y0 - (yi - hh):y1 - (yi - hh), xi:xi + W]
        return out

    def motes(self, img, t, shift):
        W, H = self.W, self.H
        r = self.mote_r
        x = (r[:, 0] * 1.3 - 0.15) * W + shift * 1.3 + np.sin(t * r[:, 2] + r[:, 3] * 6) * 0.01 * W + t * r[:, 4] * 0.01 * W
        y = r[:, 1] * H - t * r[:, 5] * 0.012 * H + np.cos(t * r[:, 2] * 0.7 + r[:, 3] * 5) * 0.008 * H
        tw = 0.5 + 0.5 * np.sin(t * (1 + r[:, 2]) + r[:, 3] * 9)
        # brighter toward the sun side
        sunw = np.exp(-((x - SUN[0] * W) / (0.35 * W)) ** 2)
        rad = (0.0015 + 0.004 * r[:, 6] ** 2) * W
        inten = (0.1 + 0.3 * tw) * (0.3 + 0.7 * sunw) * 0.6
        C.splat(img, x, y, rad, np.array([1.0, 0.8, 0.55], np.float32), inten, soft=True)

    # ------------------------------------------------------------------ sea per frame
    def sea(self, t, shift, zoom, sunx, suny):
        W, H = self.W, self.H
        z0 = 1 + (zoom - 1) * 0.7
        hzf = (HZ * H - H / 2) * z0 + H / 2
        y0 = int(math.ceil(hzf))
        ys = np.arange(y0, H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None]
        dyy = np.maximum(ys - hzf, 0.35)
        drow = lambda d: shift * (0.7 + 0.25 * np.clip(d / (H - hzf), 0, 1))
        mx = (xs - W / 2) / z0 + W / 2 + self.ox - drow(dyy)
        my = (ys - H / 2) / z0 + H / 2 + self.oy
        refl = cv2.remap(self.refl, mx.astype(np.float32), np.broadcast_to(my, mx.shape).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        col, y0 = self.seaR.render(t, hzf, drow, sunx, refl, cc('#1c2c78'), cc('#ffb060'))
        return col, y0, hzf

    # ------------------------------------------------------------------ frame
    def frame(self, t):
        W, H = self.W, self.H
        pan, Kt, T = self.cam(t)
        img = self.sample(self.sky_sun, pan, 1.0, 1.0)
        occ = None
        for pl, sp, is_occ in ((self.cirrus, 0.002, False), (self.far_sky, 0.0008, True),
                               (self.cl_near, 0.0022, True)):
            Ls = self.sample(pl, pan + sp * W * t, 1.0, 1.0)
            img = F.over_rgba(img, Ls)
            if is_occ:
                occ = Ls[..., 3] if occ is None else np.maximum(occ, Ls[..., 3])
        sx, sy = self.sun_p[0] - self.ox + pan, self.sun_p[1] - self.oy
        # crepuscular rays fanning out through the gaps between clouds
        img += F.light_shafts(W, H, sx, sy, occluder=occ, strength=0.2, length=0.95, radius=0.09,
                              tint=(1.0, 0.7, 0.45), t=t)
        # faint wide shafts reaching up-left past the left cumulus (static pattern, slow breathing)
        img += self.shaft_mask * (0.85 + 0.15 * math.sin(t * 0.7)) * F.light_shafts(
            W, H, sx, sy, occluder=occ, strength=0.15, length=1.0, radius=0.3, tint=(1.0, 0.72, 0.5), t=t,
            streaks=0.35, n_beams=6, hollow=0.6)
        hzf = HZ * H
        img = np.ascontiguousarray(img)
        self.seaR.render_fast(img, t, hzf, pan, 1.0, self.ox, self.oy, sx, self.refl, cc('#1c2c78'), cc('#ffb060'),
                              shear=-T / self.hc)
        gx, gy, gs, gi = self.seaR.glint_params(t, sx, hzf, H * 0.95)
        FX.add_glints(img, gx, gy, gs, gi * 1.05, color=(1.0, 0.88, 0.66))
        # halation source: the luminous sky / sun path BEFORE any silhouette is laid over it; spread and
        # added back after the foreground so the light wraps round every edge near the sun
        hal = self.halation_src(img)
        dfar = pan + Kt / Z_FAR
        img = F.over_rgba(img, self.sample(self.farp, dfar, 1.0, 1.0))
        sf = self.sample(self.surf, dfar, 1.0, 1.0)
        sf[..., 3] *= 0.7 + 0.3 * math.sin(t * 1.3)
        img = F.over_rgba(img, sf)
        # lighthouse lamp: slow rotating-beam flash
        lx, ly = self.lamp_p[0] - self.ox + dfar, self.lamp_p[1] - self.oy
        ph = (t + 1.1) % 3.2
        flash = 0.25 + 1.2 * math.exp(-((ph - 1.6) / 0.28) ** 2)
        FX.add_glints(img, [lx], [ly], 0.006 + 0.012 * (flash - 0.25), 0.5 * flash, color=(1.0, 0.9, 0.7))
        # a few gulls gliding over the sea, in front of the glow
        b = FX.birds_local(W, H, t, seed=5, n=4, center=(0.53 + pan * 0.9 / W, 0.3), velocity=(-0.012, -0.0015),
                           size=0.013, spread=0.045, flap=1.6, heading=-1.0, size_var=0.35)
        if b is not None:
            m, bx0, by0 = b
            sl = img[by0:by0 + m.shape[0], bx0:bx0 + m.shape[1]]
            sl[:] = sl * (1 - m[..., None]) + self.bird_col * m[..., None]
        d = self.land
        ox, oy = self.ox, self.oy
        img = np.ascontiguousarray(img)
        rw = self.rows
        self.comp(img, d['road'], d['iz_road'], pan, Kt, ox, oy, rw['road'])
        self.comp(img, d['grass'], d['iz_grass'], pan, Kt, ox, oy, rw['grass'], sway=(d['grass_w'], {}), t=t)
        self.comp(img, d['rail'], d['iz_rail'], pan, Kt, ox, oy, rw['rail'])
        self.comp(img, d['hill'], d['iz_hill'], pan, Kt, ox, oy, rw['hill'])
        self.comp(img, d['poles'], d['iz_poles'], pan, Kt, ox, oy, rw['poles'])
        WI.draw(img, self.landb.wires, pan, Kt, ox, oy, self.f, self.s)
        # specular glints travelling along the guardrail top / wires as they line up with the sun
        gl = self.landb.rail_glints + self.landb.wire_glints
        if gl:
            ga = np.array([g[:5] for g in gl], np.float64)
            gx_ = ga[:, 0] - ox + pan + Kt * ga[:, 2]
            gy_ = ga[:, 1] - oy
            al = np.exp(-((gx_ - sx) / (0.1 * W)) ** 2)
            gi = ga[:, 3] * al * (0.45 + 0.12 * np.sin(t * 1.1 + ga[:, 4]))
            FX.add_glints(img, gx_, gy_, 0.007 + 0.006 * al, gi, color=(1.0, 0.86, 0.62))
        # curve-mirror glint
        mg = self.landb.mirror_glint
        mgx, mgy = mg[0] - ox + pan + Kt * self.landb.mirror_iz, mg[1] - oy
        FX.add_glints(img, [mgx], [mgy], 0.012, 0.5 + 0.2 * math.sin(t * 2.0), color=(1.0, 0.92, 0.8))
        img *= self.fg_dark
        img = np.ascontiguousarray(img)
        # ---- near foreground (strongest parallax)
        nd, nb = self.near, self.nearb
        self.comp(img, nd['back'], nd['iz_back'], pan, Kt, nb.mx, 0, rw['back'])
        dn = pan + Kt / NEAR.Z_GRASS
        self.comp(img, nd['front'], nd['iz_front'], pan, Kt, nb.mx, 0, rw['front'],
                  sway=(nd['front_w'], dict(amp=0.007, freq=1.3, seed=1.7)), t=t)
        img = np.ascontiguousarray(img)
        # street lamp (already lit at dusk) + glint travelling along the fence top rail
        dp = pan + Kt / NEAR.Z_POLE
        lpx, lpy = nb.lamp[0] - nb.mx + dp, nb.lamp[1]
        FX.add_glints(img, [lpx], [lpy], 0.01, 0.35, color=(1.0, 0.85, 0.6))
        fgx = nb.fence_glint[0] - nb.mx + dp + (0.5 - ease(t / DURATION)) * 0.12 * W
        FX.add_glints(img, [fgx], [nb.fence_glint[1]], 0.016, 0.55, color=(1.0, 0.88, 0.68))
        img += hal
        # drifting motes of light near the lens
        self.motes(img, t, dn)
        img *= self.paper
        FX.fast_bloom(img, threshold=1.05, knee=0.3, strength=0.26, halation=0.12)
        vis = F.sun_visibility(occ, sx, sy, 0.012 * W) if occ is not None else 1.0
        # ghost chain across the frame toward the lower left; its axis swings with the truck
        uu = ease(t / DURATION)
        self.flare.add(img, sx, sy, intensity=0.4 + 0.3 * vis,
                       center=(W * (0.36 - 0.12 * (uu - 0.5)), H * (0.7 + 0.04 * (uu - 0.5))))
        # thin anamorphic streak through the sun (photographic, subtle) - static shape, follows the sun
        img += self.streak_at(sx, sy) * (0.55 + 0.45 * vis)
        img = F.shoulder(img, 0.9, desat=0.04)
        return F.finish_fast(img, t, sat=1.05, grain_amt=0.0, vig=0.22, ca=0.0006)
