"""s02_railway_crossing - rural railway crossing at summer noon.

A Japanese level crossing in the paddies: the striped crossing signal with alternating blinking red lamps
and a lowered black-yellow barrier stands in the right third; single-track rails gleam as they recede to
the vanishing point between forested hills; concrete utility poles march along the line carrying sagging
wires; lush rice paddies ripple under wind and drifting cloud shadows; a towering cumulonimbus over blue
mountains. Camera: a slow dolly-in (true perspective: the ground plane is re-projected with a homography,
near structures are re-projected from 3D every frame) with nostalgic heat haze over the far rails.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import core as C, sky as S, fx as F, clouds2 as K  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s02_railway_crossing_paint as P  # noqa: E402
import s02_railway_crossing_art as A  # noqa: E402
import s02_railway_crossing_props as PR  # noqa: E402
import s02_railway_crossing_fx2 as FX  # noqa: E402
import s02_railway_crossing_leaf as LF  # noqa: E402
import s02_railway_crossing_arch as AR  # noqa: E402
import s02_railway_crossing_tower as TW  # noqa: E402
import s02_railway_crossing_propfx as PF  # noqa: E402

DURATION = 5.0
hx = P.hexc

# ------------------------------------------------------------------------------------------ palette
COL = dict(
    rice_a=hx('#62a832'), rice_b=hx('#88c23a'), rice_c=hx('#3f8e3c'), rice_gap=hx('#1a5c48'),
    rice_hi=hx('#d6ee7a'), rice_far=hx('#9fd07a'),
    levee=hx('#b4d468'), levee_dirt=hx('#cfc08e'),
    verge=hx('#5e9c38'), verge_dk=hx('#2f6a3a'),
    ballast=hx('#9c968c'), ballast_dk=hx('#5e5850'), ballast_lt=hx('#d2cdc2'), ballast_oil=hx('#6e5f52'),
    sleeper=hx('#c9c5ba'), sleeper_sh=hx('#8a8a8e'),
    rail_head=hx('#e4ebef'), rail_face=hx('#7a4f3e'), rail_face_sh=hx('#553a33'),
    asphalt=hx('#8a8789'), asphalt_dk=hx('#5d5e68'), asphalt_lt=hx('#b8b4b0'), paint=hx('#f4f3ee'),
    panel=hx('#5a5e66'),
    haze=hx('#cfe6f0'), haze_far=hx('#d9eef4'),
    water=hx('#bfe6fa'), concrete=hx('#c8c6be'),
    shadow=hx('#3d5b8a'),
    yellow=hx('#ffc81e'), black=hx('#1d1f26'), pole_c=hx('#d5d2c8'), wood=hx('#6b5a4c'),
    wire=hx('#2a303c'),
)


def shade(base, ndl, lit_gain=(1.06, 1.02, 0.94), sh_mul=(0.56, 0.62, 0.8), sh_add=(0.04, 0.07, 0.13)):
    """Cel-ish lighting of a surface colour: shadow side is cool (sky fill), lit side warm."""
    base = np.asarray(base, np.float32)
    lit = base * np.asarray(lit_gain, np.float32)
    sh = base * np.asarray(sh_mul, np.float32) + np.asarray(sh_add, np.float32)
    k = float(P.sstep(-0.08, 0.1, ndl))
    return sh + (lit - sh) * k


def box_cov(c, a, b, fp):
    """Box-filtered coverage of interval [a, b] at coordinate c with footprint fp."""
    return np.clip((np.minimum(c + fp * 0.5, b) - np.maximum(c - fp * 0.5, a)) / fp, 0, 1)


def per_cov(c, period, duty, fp, phase=0.0):
    """Box-filtered coverage of a periodic box pattern (duty fraction of each period is 1)."""
    def Fi(z):
        z = z - phase
        k = np.floor(z / period)
        return k * duty * period + np.minimum(z - k * period, duty * period)
    return np.clip((Fi(c + fp * 0.5) - Fi(c - fp * 0.5)) / np.maximum(fp, 1e-6), 0, 1)


def lerp3(a, b, t):
    t = np.asarray(t, np.float32)
    if t.ndim >= 1 and t.shape[-1] != 3:
        t = t[..., None]
    return a + (b - a) * t


def glints_inplace(img, xs, ys, size=0.02, intensity=1.0, color=(1.0, 0.97, 0.9), t=None, seed=0):
    """Star glints (cross flare + soft core) added in place on small patches (fx.glints look, no full-frame
    allocation). Coherent twinkle over t."""
    H, W = img.shape[:2]
    xs = np.atleast_1d(np.asarray(xs, np.float32))
    ys = np.atleast_1d(np.asarray(ys, np.float32))
    n = len(xs)
    size = np.broadcast_to(np.atleast_1d(size), (n,)).astype(np.float32)
    inten = np.broadcast_to(np.atleast_1d(intensity), (n,)).astype(np.float32).copy()
    if t is not None:
        rng = np.random.default_rng(seed)
        ph = rng.uniform(0, 6.28, n)
        fr = rng.uniform(1.0, 3.0, n)
        inten = inten * np.clip(0.5 + 0.5 * np.sin(t * fr + ph), 0, 1) ** 2
    col = np.asarray(color, np.float32)
    for i in range(n):
        if inten[i] < 1e-3:
            continue
        L = size[i] * W
        R = int(L) + 2
        cx0, cy0 = max(int(xs[i]) - R, 0), max(int(ys[i]) - R, 0)
        cx1, cy1 = min(int(xs[i]) + R + 1, W), min(int(ys[i]) + R + 1, H)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        dx, dy = xx - xs[i], yy - ys[i]
        wdt = 0.6 + 0.02 * L
        k = (np.exp(-(dy / wdt) ** 2) * np.clip(1 - np.abs(dx) / L, 0, 1) ** 2.5 +
             0.7 * np.exp(-(dx / wdt) ** 2) * np.clip(1 - np.abs(dy) / L, 0, 1) ** 2.5)
        k += np.exp(-(dx * dx + dy * dy) / (max(L * 0.08, 0.8) ** 2)) * 1.5
        img[cy0:cy1, cx0:cx1] += k[..., None] * col * inten[i]


def flare_ghosts(W, H, lx, ly):
    """Extra soft lens-flare ghosts (discs + a faint hexagon + rainbow ring) on the line from the sun through
    the frame centre - the aperture reflections of an anime photographic flare."""
    out = np.zeros((H // 2, W // 2, 3), np.float32)
    ys, xs = np.mgrid[0:H // 2, 0:W // 2].astype(np.float32)
    xs = xs * 2 + 1
    ys = ys * 2 + 1
    cx, cy = W / 2, H / 2
    vx, vy = cx - lx, cy - ly
    for (k, r, col, a, kind) in [(0.42, 0.018, (1.0, 0.85, 0.6), 0.10, 0), (0.7, 0.05, (0.55, 0.9, 0.75), 0.05, 1),
                                 (1.05, 0.028, (0.6, 0.75, 1.0), 0.08, 0), (1.32, 0.085, (0.7, 0.6, 1.0), 0.04, 2),
                                 (1.62, 0.04, (1.0, 0.7, 0.8), 0.06, 1), (1.9, 0.12, (0.6, 0.85, 1.0), 0.035, 0)]:
        gx, gy = lx + vx * k, ly + vy * k
        R = r * W
        if kind == 1:
            ang = np.arctan2(ys - gy, xs - gx)
            dd = np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2) * (np.cos(np.pi / 6) / np.cos(((ang + np.pi / 6) % (np.pi / 3)) - np.pi / 6))
        else:
            dd = np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2)
        d = dd / R
        if kind == 2:
            m = np.exp(-((d - 1.0) / 0.06) ** 2)
            hue = np.stack([0.6 + 0.4 * np.cos(d * 14), 0.7 + 0.3 * np.cos(d * 14 + 2.1), 0.8 + 0.2 * np.cos(d * 14 + 4.2)], -1)
            out += m[..., None] * hue * a
        else:
            m = P.sstep(1.0, 0.8, d) * (0.55 + 0.45 * P.sstep(0.3, 1.0, d))
            out += m[..., None] * np.array(col, np.float32) * a
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def over_straight(bot, top):
    """Straight-alpha RGBA 'over' (both (H, W, 4))."""
    ta = top[..., 3:4]
    ba = bot[..., 3:4] * (1 - ta)
    a = ta + ba
    rgb = (top[..., :3] * ta + bot[..., :3] * ba) / np.maximum(a, 1e-5)
    return np.concatenate([rgb, a], -1).astype(np.float32)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.f = 0.82 * W
        self.px = 0.42 * W
        self.hz = 0.6 * H
        self.h = 1.6
        self.sc = W / 1920.0
        L = np.array([0.72, 0.6, 0.36])
        self.L = L / np.linalg.norm(L)
        # ground cast-shadow direction: the sun sits up-right and AHEAD (as seen on screen), so the posts,
        # poles and machines throw diagonal shadows toward the camera-left across the road
        Ls = np.array([0.5, 0.74, 0.46])
        self.Lsh = Ls / np.linalg.norm(Ls)
        # sun sits just outside the top-right corner (glare + flare leak into frame)
        self.sun_screen = (0.905 * W, 0.075 * H)
        self.Xt = -1.0
        self.road = (9.3, 14.3)
        self.rng = np.random.default_rng(2)
        self.noise_a = P.tile_noise(256, seed=11, octaves=5, base=4)
        self.noise_b = P.tile_noise(256, seed=12, octaves=4, base=2)
        self.noise_c = P.tile_noise(256, seed=13, octaves=3, base=8)
        self._build_world()
        for nm in ('sky', 'far', 'mid', 'ground', 'grass'):
            self._stage(nm, getattr(self, '_build_' + nm))
        self.canvas = P.Canvas(W, H, ss=3)
        # painted prop textures (face rasters, blitted per frame with a perspective warp)
        v = self.vend
        self.tx_vend = PR.vending_front(v['X1'] - v['X0'], v['H'])
        self.tx_vside = PR.vending_side(v['D'], v['H'])
        self.tx_bin = PR.bin_front()
        self.tx_gb = {False: PR.gearbox_front(mirror=False), True: PR.gearbox_front(mirror=True)}
        self.tx_pole = PR.pole_texture(self.poles[0]['H'], seed=4)
        self._haze_tex = P.tile_noise(128, seed=21, octaves=3, base=4)
        self.mid = self._rim_plate(self.mid, 2.2 * self.sc, 0.5)
        self.mount = self._rim_plate(self.mount, 1.6 * self.sc, 0.3)
        self._build_glare()

    def _rim_plate(self, plate, r, amt):
        """Sun-side rim light on a straight-alpha plate: pixels whose neighbour toward the sun (upper right)
        is empty sky get a warm bright edge."""
        a = plate[..., 3]
        dx, dy = r * 0.85, -r * 0.55
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        nb = cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), borderValue=0)
        rim = np.clip(a - nb, 0, 1) * a
        rim = cv2.GaussianBlur(rim, (0, 0), 0.4)
        out = plate.copy()
        out[..., :3] = out[..., :3] + (np.array([1.0, 0.95, 0.82], np.float32) - out[..., :3]) * (rim * amt)[..., None]
        return out

    def _build_glare(self):
        """Static sun glare: a warm haze wedge washing down from the sun over the right third and along the
        horizon (screen-blended per frame)."""
        W, H = self.W, self.H
        q = 4
        ys, xs = np.mgrid[0:H // q, 0:W // q].astype(np.float32)
        xs, ys = xs * q, ys * q
        sx, sy = self.sun_screen
        d = np.sqrt(((xs - sx) / W) ** 2 + ((ys - sy) / W) ** 2)
        ang = np.arctan2(ys - sy, -(xs - sx))           # 0 = toward the left, pi/2 = straight down
        wedge = np.exp(-((ang - 1.05) / 0.55) ** 2)       # a fan pointing down-left toward the vanishing point
        g = np.exp(-d / 0.16) * 0.34 + np.exp(-d / 0.45) * 0.12 * (0.5 + 0.5 * wedge) + wedge * np.exp(-d / 0.7) * 0.1
        hzb = np.exp(-((ys - self.hz) / (0.07 * H)) ** 2) * (0.05 + 0.95 * P.sstep(0.1 * W, 0.95 * W, xs)) * 0.2
        g = g + hzb
        col = np.array([1.0, 0.95, 0.84], np.float32)
        gl = (g[..., None] * col).astype(np.float32)
        self._glare = cv2.resize(gl, (W, H), interpolation=cv2.INTER_CUBIC)

    def _stage(self, name, fn):
        """Dev-only plate cache (env S02_CACHE=dir; S02_FRESH=comma list of stages to rebuild)."""
        d = os.environ.get('S02_CACHE')
        if not d:
            return fn()
        import pickle
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'{name}_{self.W}.pkl')
        if os.path.exists(path) and name not in os.environ.get('S02_FRESH', '').split(','):
            with open(path, 'rb') as fh:
                self.__dict__.update(pickle.load(fh))
            return None
        before = dict(self.__dict__)
        fn()
        new = {k: v for k, v in self.__dict__.items() if k not in before or before[k] is not v}
        with open(path, 'wb') as fh:
            pickle.dump(new, fh)
        return None

    # ------------------------------------------------------------------ camera
    def cam(self, t):
        u = min(max(t / DURATION, 0.0), 1.0)
        s = 0.4 * u + 0.6 * C.ease_in_out_sine(u)
        return dict(cz=1.1 * s, cx=-0.32 * s, fz=1.0 + 0.02 * s)

    def proj(self, X, Y, Z, c):
        X, Y, Z = np.asarray(X, np.float64), np.asarray(Y, np.float64), np.asarray(Z, np.float64)
        Zr = np.maximum(Z - c['cz'], 0.05)
        F_ = self.f * c['fz']
        return self.px + F_ * (X - c['cx']) / Zr, self.hz - F_ * (Y - self.h) / Zr

    def kscale(self, Z, c):
        return self.f * c['fz'] / max(Z - c['cz'], 0.05)

    # ------------------------------------------------------------------ world layout
    def _build_world(self):
        rng = np.random.default_rng(5)
        self.sigA = dict(X=2.35, Z=8.7, arm=5.0, arm_dir=1, scale=1.0)
        self.sigB = dict(X=-4.5, Z=15.0, arm=5.0, arm_dir=-1, scale=1.0, mirror=True)
        # concrete utility poles along the right of the line
        self.poles = [dict(X=7.3, Z=15.9, H=11.8, kind='conc', tr=False, guy=True, near=True, sign=True)]
        self.vend = dict(X0=5.05, X1=6.05, Z=15.45, D=0.72, H=1.83)
        z = 24.0
        k = 0
        while z < 700:
            self.poles.append(dict(X=3.9 + rng.normal(0, 0.05), Z=z, H=10.8 + rng.normal(0, 0.25), kind='conc',
                                   tr=(k in (0, 3))))
            z += 34.0 + rng.uniform(-2, 2)
            k += 1
        # wooden telephone poles along the left
        self.tpoles = []
        z = 31.0
        while z < 700:
            self.tpoles.append(dict(X=-5.6 + rng.normal(0, 0.05), Z=z, H=8.2 + rng.normal(0, 0.2)))
            z += 42.0 + rng.uniform(-3, 3)

    # ------------------------------------------------------------------ sky
    def _build_sky(self):
        W, H = self.W, self.H
        pw, ph = int(W * 1.12), int(H * 1.1)
        self.pw, self.ph = pw, ph
        ox, oy = (pw - W) / 2, (ph - H) / 2
        hzp = (self.hz + oy) / ph
        sun_p = (self.sun_screen[0] + ox, self.sun_screen[1] + oy)
        self.sun_p = sun_p
        sky = S.sky_gradient(pw, ph, 'summer_noon', horizon=hzp, sun=sun_p, sun_radius=0.28)
        hy = hzp * ph
        u = W / 1920.0
        # --- painted clouds (lib/clouds2): hero cumulonimbus tower rising behind the hills left of the
        # vanishing point (sun up-right, slightly ahead: warm lit right flank, lavender left side, silver rims),
        # its anvil shearing right; a pair of fair-weather cumulus over the right fields; a far hazy bank
        # along the horizon; faint high cirrus.
        sky_h = K.haze_plate(pw, ph, hy - 0.2 * ph, hy + 0.01 * ph, '#d4ecf8', 0.4)
        bank = K.horizon_bank_plate(pw, ph, hy + 0.004 * ph, 0.075 * ph, sun=sun_p, preset='noon', seed=61, rows=4,
                                    haze=0.3, haze_color='#d4ecf8', shrink=0.4, layer_haze=0.12)
        far = K.cumulus_plate(pw, ph, [(0.66 * pw, hy - 0.065 * ph, 0.13 * pw, 0.05 * pw, 23, 0.75),
                                       (0.98 * pw, hy - 0.11 * ph, 0.12 * pw, 0.045 * pw, 24, 0.7),
                                       (0.03 * pw, hy - 0.07 * ph, 0.1 * pw, 0.04 * pw, 26, 0.8)],
                              sun=sun_p, preset='noon', n=3, sun_jitter=20.0, sun_z=0.25)
        mid = K.cumulus_plate(pw, ph, [(0.8 * pw, hy - 0.2 * ph, 0.2 * pw, 0.085 * pw, 7, 0.4),
                                       (0.55 * pw, 0.27 * ph, 0.085 * pw, 0.03 * pw, 8, 0.5)],
                              sun=sun_p, preset='noon', n=4, sun_jitter=25.0, sun_z=0.16, backlit=0.25)
        # aerial perspective: the right cumulus sinks into the horizon haze toward its base
        mid = TW.haze_gradient(mid, hy - 0.42 * ph, hy - 0.16 * ph, '#cfe7f6', 0.55, x_range=(0.66 * pw, 1.0 * pw, 0.05 * pw))
        near, _ = TW.tower(pw, ph, 0.25 * pw, hy - 0.03 * ph, 0.486 * ph, W, seed=12)
        cir = K.cirrus_plate(pw, ph, preset='noon', seed=21, region=(0.02, 0.26), angle=-7.0, density=0.35,
                             opacity=0.35)
        self.bird_col = S.bird_color(sky[int(0.35 * ph), int(0.6 * pw)])
        # the faint cirrus, the far horizon bank and the haze wash are baked into the sky plate (they barely
        # move over 5 s); far -> near drifting layers: far cumulus, mid cumulus, hero tower (occluders = last two)
        self.sky = np.ascontiguousarray(K.composite(sky, cir, bank, sky_h), dtype=np.float32)
        self.clouds = S.CloudDrift([(far, 0.002, 0.1), (mid, 0.006, 0.16, 0.0015), (near, 0.0035, 0.2, 0.0014)])
        self.n_occ = 2

    # ------------------------------------------------------------------ mountains / hills
    def _profile(self, W, seed, base, amp, freq, sharp=1.6):
        rng = np.random.default_rng(seed)
        x = np.arange(W, dtype=np.float64) / W
        p = np.zeros(W)
        a = 1.0
        fr = freq
        for o in range(6):
            ph = rng.uniform(0, 100)
            n = np.interp(x * fr + ph, np.arange(int(fr + ph) + 3), rng.random(int(fr + ph) + 3))
            n = cv2.GaussianBlur(n.astype(np.float32)[None, :], (0, 0), sigmaX=max(W / fr / 6, 0.5))[0]
            p += a * (1 - np.abs(n * 2 - 1)) ** sharp
            a *= 0.48
            fr *= 2.1
        p = (p - p.min()) / (p.max() - p.min() + 1e-9)
        return base - amp * p

    def _build_far(self):
        """Distant mountains: a ridged-multifractal terrain rendered voxel-style (true ridges, gullies,
        valleys), painted in crisp value bands with blue-lavender atmospheric tint and white base haze;
        then bluish wooded hills / tree rows along the horizon (painted foliage masses)."""
        W, H = self.W, self.H
        hz = self.hz
        mw, mh = int(W * 1.08), H
        self.far_off = ((mw - W) / 2, 0)
        ox = self.far_off[0]
        N = 2048
        r = A.ridged_mf(N, 7, base=12, octaves=9, gain=0.6)
        macro = C.fbm(N, N, scale=3, octaves=3, seed=9)
        x0w, z0w = -40000.0, 4000.0
        cell = 80000.0 / N
        jj, ii = np.mgrid[0:N, 0:N].astype(np.float32)
        wz = z0w + jj * cell
        wx = x0w + ii * cell
        env = np.clip((wz - 10000) / 5000, 0, 1) ** 0.6
        # lower saddle behind the vanishing point so the line of the rails leads into open sky/haze
        vpx = (wx / np.maximum(wz, 1)) * self.f
        env = env * (1 - 0.45 * np.exp(-((vpx + 0.02 * W) / (0.12 * W)) ** 2))
        hm = (r ** 1.2 * 2400 + macro ** 2 * 1500) * env
        hm = cv2.GaussianBlur(hm.astype(np.float32), (0, 0), 0.5)
        gy, gx = np.gradient(hm, cell)
        nx, ny, nz = -gx, np.ones_like(hm), -gy
        nl = np.sqrt(nx * nx + ny * ny + nz * nz)
        Lm = np.array([0.85, 0.45, -0.15])
        Lm = Lm / np.linalg.norm(Lm)
        shd = ((nx * Lm[0] + ny * Lm[1] + nz * Lm[2]) / nl).astype(np.float32)
        sc_ = W / 1920.0
        Sv, D, E, Al = A._voxel(mw, mh, hm, x0w, z0w, cell, self.f, self.px + ox, hz, 60.0, 5000.0, 60000.0,
                                0.00025, 1 / (2 * 6.4e6), shd, hm)
        v = cv2.GaussianBlur(Sv, (0, 0), 0.6 * sc_ + 0.2)
        # painted ridge detail: fine downslope gully striations (vertical brush streaks) break the facets
        import s02_railway_crossing_cb2 as _CB2
        gst = _CB2.streak_noise(mh, mw, 71, cx=3.0, cy=int(160 * sc_ + 60), octaves=3).T
        gst2 = _CB2.streak_noise(mh, mw, 72, cx=5.0, cy=int(420 * sc_ + 100), octaves=2).T
        v = v + ((gst - 0.5) * 0.07 + (gst2 - 0.5) * 0.05) * (Al > 0.5)
        # ridge / gully painting: deep blue gullies, mid-blue shadow faces, pale sunlit faces, bright crests
        col = hx('#3f5aa6') + (hx('#5f7cc6') - hx('#3f5aa6')) * P.sstep(0.28, 0.34, v)[..., None]
        col = col + (hx('#8fabe2') - col) * P.sstep(0.4, 0.44, v)[..., None]
        col = col + (hx('#b9cff2') - col) * P.sstep(0.55, 0.59, v)[..., None]
        col = col + (hx('#dce8fa') - col) * (P.sstep(0.74, 0.78, v) * 0.85)[..., None]
        # snowfields / bare rock on the high summits: snow keeps to the gullies and the shaded couloirs,
        # sun-bleached white on lit faces, pale blue in shade
        sn = C.fbm(mw // 2, mh // 2, scale=40, octaves=3, seed=77)
        sn = cv2.resize(sn, (mw, mh), interpolation=cv2.INTER_LINEAR)
        snow = P.sstep(2300, 2800, E + (sn - 0.5) * 900) * (E < 1e5)
        snow_c = hx('#a9c0ec') + (hx('#f6f9ff') - hx('#a9c0ec')) * P.sstep(0.44, 0.5, v)[..., None]
        col = col + (snow_c - col) * (snow * 0.85)[..., None]
        fog = 1 - np.exp(-D / 90000)
        col = col + (hx('#cfe0f4') - col) * (fog * 0.5)[..., None]
        low = np.exp(-E / 260)
        col = col + (hx('#dcebf8') - col) * (low * 0.8)[..., None]
        ys = np.arange(mh, dtype=np.float32)[:, None]
        Al = Al * (ys < hz + 1)
        # slightly softer skyline (atmosphere), without losing the silhouette
        Al = np.maximum(Al * 0.0, cv2.GaussianBlur(Al.astype(np.float32), (0, 0), 0.7 * sc_ + 0.3))
        col = col + (hx('#c4d6f2') - col) * (P.sstep(0.95, 0.4, Al) * 0.5)[..., None]
        # strong atmospheric fade toward pale cyan at the base (the range sits behind the noon haze)
        yf_ = P.sstep(hz - 0.16 * H, hz + 0.005 * H, ys)
        col = col * (1 - 0.1 * (1 - yf_))[..., None] + 0  # slightly deeper blue up high
        col = col + (hx('#d4ecf6') - col) * (yf_ ** 1.3 * 0.82)[..., None]
        self.mount = np.dstack([col, Al]).astype(np.float32)
        # ---- bluish wooded hills and tree rows along the horizon (gap around the vanishing point)
        rng = np.random.default_rng(17)
        hl = []
        vp = self.px + ox
        sc = self.sc
        for (x0, x1, peak, seed, fog_, rm) in [(-0.05 * mw, vp - 0.09 * W, 0.07 * H, 21, 0.55, 5.5),
                                               (vp + 0.07 * W, 1.05 * mw, 0.055 * H, 22, 0.55, 5.5),
                                               (0.05 * mw, vp - 0.03 * W, 0.035 * H, 23, 0.38, 4.5),
                                               (vp + 0.1 * W, 0.8 * mw, 0.03 * H, 24, 0.36, 4.5),
                                               (-0.05 * mw, 1.05 * mw, 0.012 * H, 25, 0.28, 3.5)]:
            prof = A.profile(int(x1 - x0) + 4, seed, 2.2, 1.2)
            n = len(prof)
            u = np.linspace(0, 1, n)
            env_ = np.sin(np.clip(u, 0, 1) * math.pi) ** 0.5
            topf = lambda x, prof=prof, env_=env_, x0=x0, peak=peak, n=n: hz - peak * (0.3 + 0.7 * prof[
                int(np.clip(x - x0, 0, n - 1))]) * env_[int(np.clip(x - x0, 0, n - 1))] - 0.003 * H
            hl += LF.hedge_row(rng, x0, x1, hz + 0.004 * H, topf, rm * sc * 0.75, -1e5 * fog_, fog_, tone_sd=0.25)
        pal_far = dict(gap=hx('#1f4c5c'), shd=hx('#2f6272'), shd_w=hx('#386c6c'), mid=hx('#4f8a64'),
                       mid_w=hx('#5c9660'), lit=hx('#8cbd6c'), hi=hx('#b4d88a'), rim=hx('#dcefc0'),
                       haze=hx('#a3c1e6'))
        hills, _ = LF.render(mw, mh, np.array(hl, np.float32), pal=pal_far, ss=2, holes=0.3, seed=3, wl=0.3,
                             rim_amt=0.4)
        self.mount = over_straight(self.mount, hills)

    # ------------------------------------------------------------------ midground (houses, groves)
    def _build_mid(self):
        """Midground farmsteads and groves (Z ~ 100..600): painted broadleaf crowns (camphor / keyaki /
        shrubs), sugi cedar windbreaks behind the houses, kawara-roofed farmhouses with sheds, block walls
        and cast shadows. Rendered once for the t=0 camera, in depth buckets (far -> near)."""
        W, H = self.W, self.H
        c0 = dict(cz=0.0, cx=0.0, fz=1.0)
        mw, mh = int(W * 1.08), H
        ox = (mw - W) / 2
        self.mid_off = (ox, 0)
        self.mid_Z = 140.0
        rng = np.random.default_rng(41)
        items = []
        # (kind, X, Z, params)
        # --- left farmstead (near): a two-storey house with balcony laundry, AC units, antenna and a block
        # wall, a kei truck in the drive, a tin shed, an older one-storey house behind a hedge, a roadside
        # vegetable stand with nobori banners; camphor grove + cedar windbreak behind
        items += [('house', -19.0, 80.0, dict(wd=10, dp=7.5, floors=2, wall='siding', tile=hx('#56657f'), seed=1,
                                              balcony=True, antenna=True, block=True, entrance=0.3,
                                              ac=[('front', 2.92, 3.3), ('side', 1.2, 3.25), ('side', 4.6, 0.3)])),
                  ('truck', -18.0, 73.8, dict(seed=2)),
                  ('shed', -10.8, 77.0, dict(wd=3.6, dp=3.0, h=2.4, roof=hx('#4d7fb0'))),
                  ('stand', -8.2, 61.0, dict(seed=5)),
                  ('upole', -13.2, 78.0, dict(drops=[(-14.0, 5.9, 79.0), (-9.5, 2.5, 77.5)])),
                  ('house', -37.0, 98.0, dict(wd=11, dp=8, floors=1, wall='wood', tile=hx('#6d5a52'), seed=2,
                                              posts=5, hedge=True, yard=2.6, ac=[('front', 9.0, 0.05)])),
                  ('grove', -27.0, 96.0, dict(n=2, span=6, hgt=(12, 15), kind='round')),
                  ('grove', -12.0, 92.0, dict(n=1, span=3, hgt=(9, 11), kind='round')),
                  ('cedars', -40.0, 125.0, dict(n=9, span=30, hgt=(15, 22))),
                  ('grove', -55.0, 128.0, dict(n=4, span=22, hgt=(12, 16), kind='vase')),
                  ('house', -64.0, 150.0, dict(wd=10, dp=7, floors=2, wall='plaster', tile=hx('#5f6c88'), seed=6,
                                               antenna=True, ac=[('front', 7.5, 0.05)])),
                  ('grove', -95.0, 190.0, dict(n=6, span=46, hgt=(12, 17), kind='round')),
                  ('cedars', -130.0, 260.0, dict(n=10, span=60, hgt=(18, 24)))]
        # --- right farmstead
        items += [('house', 27.0, 114.0, dict(wd=11, dp=8, floors=2, wall='plaster', tile=hx('#667490'), seed=3,
                                              antenna=True, block=True, ac=[('front', 3.2, 3.4), ('front', 6.75, 3.4), ('side', 2.2, 0.3)])),
                  ('shed', 36.8, 110.0, dict(wd=3.0, dp=2.4, h=2.2, roof=hx('#b0563c'))),
                  ('house', 60.0, 190.0, dict(wd=12, dp=8, floors=1, wall='wood', tile=hx('#5a5a66'), seed=4, posts=5,
                                              hedge=True)),
                  ('house', 90.0, 290.0, dict(wd=12, dp=8, floors=2, wall='plaster', tile=hx('#5d6a86'), seed=7)),
                  ('grove', 30.0, 132.0, dict(n=4, span=26, hgt=(11, 15), kind='round')),
                  ('cedars', 50.0, 150.0, dict(n=6, span=18, hgt=(16, 21))),
                  ('grove', 44.0, 118.0, dict(n=2, span=6, hgt=(8, 10), kind='vase')),
                  ('grove', 110.0, 300.0, dict(n=7, span=70, hgt=(12, 16), kind='round')),
                  ('cedars', 150.0, 250.0, dict(n=8, span=50, hgt=(18, 23))),
                  ('grove', 20.0, 420.0, dict(n=6, span=46, hgt=(11, 15), kind='round')),
                  ('grove', -26.0, 520.0, dict(n=7, span=60, hgt=(11, 14), kind='round')),
                  ('cedars', -60.0, 430.0, dict(n=8, span=40, hgt=(17, 22)))]
        items.sort(key=lambda it: -it[2])
        buckets = [(360, 1e9), (220, 360), (160, 220), (118, 160), (85, 118), (0, 85)]
        out = np.zeros((mh, mw, 4), np.float32)
        sway = np.zeros((mh, mw), np.float32)
        folown = np.zeros((mh, mw), np.float32)
        hzc = hx('#a9cde2')
        for (za, zb) in buckets:
            its = [it for it in items if za <= it[2] < zb]
            if not its:
                continue
            cvA = P.Canvas(mw, mh, ss=3)
            cvB = P.Canvas(mw, mh, ss=3)
            clumps = []
            for (kind, X, Z, prm) in its:
                fog = float(np.clip((Z - 60) / 480, 0, 0.62)) ** 0.9
                if kind == 'grove':
                    n = int(round(prm['n'] * 1.7))
                    for i in range(n):
                        Xi = X - prm['span'] / 2 + (i + rng.uniform(0.2, 0.8)) * prm['span'] / n
                        Zi = Z + rng.uniform(-4, 6)
                        th = rng.uniform(*prm['hgt'])
                        kd = prm['kind'] if rng.random() < 0.7 else ('vase' if prm['kind'] == 'round' else 'round')
                        cw = th * (rng.uniform(0.85, 1.15) if kd == 'round' else rng.uniform(0.9, 1.2))
                        xb, yb = self.proj(Xi, 0, Zi, c0)
                        xb, yb = float(xb) + ox, float(yb)
                        k = self.f / Zi
                        trunk_h = th * rng.uniform(0.16, 0.28)
                        tw = max(0.28 * k, 0.8)
                        tc = hx('#3a3434') * (1 - fog) + hzc * fog
                        cvA.poly([(xb - tw, yb), (xb + tw, yb), (xb + tw * 0.6, yb - trunk_h * k * 1.3),
                                  (xb - tw * 0.6, yb - trunk_h * k * 1.3)], tc)
                        for bI in range(3):
                            ang = rng.uniform(-0.9, 0.9)
                            L_ = th * k * rng.uniform(0.25, 0.4)
                            y0_ = yb - trunk_h * k * rng.uniform(0.8, 1.2)
                            cvA.line([(xb, y0_), (xb + math.sin(ang) * L_, y0_ - math.cos(ang) * L_)],
                                     max(tw * 0.7, 0.6), tc)
                        wpx = cw * k
                        clumps += LF.crown(rng, xb, yb - trunk_h * k * 0.9, wpx, (th - trunk_h * 0.9) * k,
                                           -Zi * 50.0, fog, tone0=rng.uniform(-0.2, 0.3), kind=kd,
                                           leaf_r=max(wpx / 13.0, 2.0 * self.sc), tree=float(len(clumps) % 997))
                elif kind == 'cedars':
                    n = prm['n']
                    pal_c = [hx('#133a3a'), hx('#24604c'), hx('#4f8e44'), hx('#9cc65a')]
                    gaps = rng.gamma(1.6, 1.0, n)
                    xs_ = np.cumsum(gaps)
                    xs_ = X - prm['span'] / 2 + (xs_ - xs_[0]) / max(xs_[-1] - xs_[0], 1e-3) * prm['span']
                    trees = []
                    for i in range(n):
                        Zi = Z + rng.uniform(-6, 10)
                        th = rng.uniform(*prm['hgt']) * rng.choice([0.62, 0.8, 1.0, 1.0, 1.18])
                        wr = rng.uniform(0.24, 0.36)
                        trees.append((Zi, xs_[i], th, wr))
                    trees.sort(key=lambda q_: -q_[0])
                    for (Zi, Xi, th, wr) in trees:
                        xb, yb = self.proj(Xi, 0, Zi, c0)
                        k = self.f / Zi
                        fog_i = float(np.clip((Zi - 60) / 420, 0, 0.7)) ** 0.85
                        FX.draw_cedar3(cvA, rng, float(xb) + ox, float(yb), th * wr * k, th * k, pal_c, fog=fog_i,
                                       hazec=hx('#9dbfe4'))
                elif kind == 'house':
                    AR.house(self, cvB, X, Z, prm, ox, fog)
                elif kind == 'shed':
                    AR.shed(self, cvB, X, Z, prm, ox, fog)
                elif kind == 'truck':
                    AR.kei_truck(self, cvB, X, Z, ox, fog, seed=prm['seed'])
                elif kind == 'stand':
                    AR.farm_stand(self, cvB, X, Z, ox, fog, seed=prm['seed'])
                elif kind == 'upole':
                    AR.service_pole(self, cvB, X, Z, ox, fog, drops=prm['drops'])
            rgbA, aA = cvA.resolve_premult()
            layer = np.zeros((mh, mw, 4), np.float32)
            layer[..., :3] = rgbA / np.maximum(aA, 1e-4)[..., None]
            layer[..., 3] = aA
            fshare = np.zeros((mh, mw), np.float32)
            fsw = np.zeros((mh, mw), np.float32)
            if clumps:
                fol, fsw = LF.render(mw, mh, np.array(clumps, np.float32), ss=2, seed=int(za) % 97)
                layer = over_straight(layer, fol)
                fshare = fol[..., 3]
            rgbB, aB = cvB.resolve_premult()
            lb = np.dstack([rgbB / np.maximum(aB, 1e-4)[..., None], aB]).astype(np.float32)
            layer = over_straight(layer, lb)
            fshare = fshare * (1 - aB)
            out = over_straight(out, layer)
            sway = sway * (1 - layer[..., 3]) + fshare * fsw
            folown = folown * (1 - layer[..., 3]) + fshare
        self.mid = out
        # wind-sway weight: foliage only (crown tops most), grown a few px outward so the swaying edges can
        # move out into the sky, zero on houses / trunks (they stay rigid)
        r_ = max(int(round(3 * self.sc)), 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1))
        sw_ = cv2.GaussianBlur(cv2.dilate(sway, ker), (0, 0), 1.0 * self.sc + 0.3)
        rigid = np.clip(out[..., 3] - folown, 0, 1)
        self.mid_sway = (sw_ * (1 - rigid)).astype(np.float32)

    def _draw_shed(self, cv, X, Z, prm, ox, c0, fog):
        hz_c = hx('#a9cde2')

        def fc(c):
            return np.asarray(c, np.float32) * (1 - fog) + hz_c * fog

        def pr(x, y, z):
            a, b = self.proj(x, y, z, c0)
            return (float(a) + ox, float(b))
        wd, dp, h = prm['wd'], prm['dp'], prm['h']
        x0, x1 = X - wd / 2, X + wd / 2
        xs = x0 if X > 0 else x1
        Lx, Ly, Lz = self.L
        self._ground_shadow(cv, [(x0, Z), (x1, Z), (x1, Z + dp), (x0, Z + dp)], h, fc)
        cv.poly([pr(xs, 0, Z), pr(xs, 0, Z + dp), pr(xs, h, Z + dp), pr(xs, h, Z)],
                fc(shade(hx('#c9c6bc'), Lx * (1 if X < 0 else -1))))
        cv.poly([pr(x0, 0, Z), pr(x1, 0, Z), pr(x1, h, Z), pr(x0, h, Z)], fc(shade(hx('#d8d4c8'), -0.2)))
        # sliding door + ribbing
        cv.poly([pr(x0 + wd * 0.15, 0, Z), pr(x0 + wd * 0.6, 0, Z), pr(x0 + wd * 0.6, h * 0.8, Z),
                 pr(x0 + wd * 0.15, h * 0.8, Z)], fc(shade(hx('#9aa2a6'), -0.2)))
        for i in range(6):
            xx = x0 + wd * (0.1 + 0.8 * i / 5)
            cv.line([pr(xx, 0.05, Z), pr(xx, h * 0.95, Z)], 0.7, fc(shade(hx('#b4b0a4'), -0.3)))
        # corrugated mono-pitch roof overhang
        rc = prm['roof']
        cv.poly([pr(x0 - 0.3, h + 0.35, Z - 0.4), pr(x1 + 0.3, h + 0.35, Z - 0.4), pr(x1 + 0.3, h + 0.05, Z + dp + 0.3),
                 pr(x0 - 0.3, h + 0.05, Z + dp + 0.3)], fc(shade(rc, 0.6)))
        cv.poly([pr(x0 - 0.3, h + 0.35, Z - 0.4), pr(x1 + 0.3, h + 0.35, Z - 0.4), pr(x1 + 0.3, h + 0.2, Z - 0.4),
                 pr(x0 - 0.3, h + 0.2, Z - 0.4)], fc(shade(rc, -0.3)))
        cv.poly([pr(x0, h, Z), pr(x1, h, Z), pr(x1, h - 0.25, Z), pr(x0, h - 0.25, Z)], fc(hx('#3c4150')))

    def _ground_shadow(self, cv, footprint, h, fc, alpha=0.5):
        """Cast shadow of a prism (footprint (X, Z) list, height h) on the ground, drawn in the canvas."""
        Lx, Ly, Lz = self.L
        c0 = dict(cz=0.0, cx=0.0, fz=1.0)
        ox = self.mid_off[0]
        pts = list(footprint) + [(x - h * Lx / Ly, z - h * Lz / Ly) for (x, z) in footprint]
        P2 = np.array([[float(v) for v in self.proj(x, 0, z, c0)] for (x, z) in pts], np.float32)
        P2[:, 0] += ox
        hull = cv2.convexHull(P2)[:, 0, :]
        cv.poly([tuple(p) for p in hull], fc(hx('#2c4a52')), alpha)

    def _draw_house2(self, cv, X, Z, prm, ox, c0, fog):
        """Rural farmhouse: kawara gable roof (ridge cap + onigawara, tile columns, eave tile ends, eave
        shadow), plaster or charred-wood walls, aluminium sash windows reflecting the sky, entrance, block
        wall, ground shadow."""
        hz_c = hx('#a9cde2')
        rng = np.random.default_rng(prm['seed'])

        def fc(c):
            return np.asarray(c, np.float32) * (1 - fog) + hz_c * fog

        def pr(x, y, z):
            a, b = self.proj(x, y, z, c0)
            return (float(a) + ox, float(b))
        wd, dp, hw, ht = prm['wd'], prm['dp'], prm['hw'], prm['ht']
        x0, x1 = X - wd / 2, X + wd / 2
        Zf, Zb = Z, Z + dp
        left_side = X > 0            # we see the -X face of houses right of the camera
        xs = x0 if left_side else x1
        Lx, Ly, Lz = self.L
        side_ndl = -Lx if left_side else Lx
        k = self.f / Z
        # ground shadow
        self._ground_shadow(cv, [(x0 - 0.8, Zf - 0.8), (x1 + 0.8, Zf - 0.8), (x1 + 0.8, Zb + 0.8), (x0 - 0.8, Zb + 0.8)],
                            ht * 0.8, fc, 0.55)
        plaster = hx('#ece4d2') if prm['wall'] == 'plaster' else hx('#4a3a34')
        lower = hx('#5a4438') if prm['wall'] == 'plaster' else hx('#3e302c')
        # --- side wall + gable
        cv.poly([pr(xs, 0, Zf), pr(xs, 0, Zb), pr(xs, hw, Zb), pr(xs, hw, Zf)], fc(shade(plaster, side_ndl)))
        cv.poly([pr(xs, 0, Zf), pr(xs, 0, Zb), pr(xs, hw * 0.35, Zb), pr(xs, hw * 0.35, Zf)], fc(shade(lower, side_ndl)))
        cv.poly([pr(xs, hw, Zf), pr(xs, hw, Zb), pr(xs, ht - 0.3, (Zf + Zb) / 2)], fc(shade(plaster, side_ndl) * 0.96))
        # side window
        wz0, wz1 = Zf + dp * 0.35, Zf + dp * 0.65
        cv.poly([pr(xs, hw * 0.4, wz0), pr(xs, hw * 0.4, wz1), pr(xs, hw * 0.75, wz1), pr(xs, hw * 0.75, wz0)],
                fc(hx('#c8ccd0')))
        cv.poly([pr(xs, hw * 0.43, wz0 + 0.1), pr(xs, hw * 0.43, wz1 - 0.1), pr(xs, hw * 0.72, wz1 - 0.1),
                 pr(xs, hw * 0.72, wz0 + 0.1)], fc(hx('#7fa8d4')))
        # --- front wall (in shade: sun is ahead) with lower wainscot
        front = [pr(x0, 0, Zf), pr(x1, 0, Zf), pr(x1, hw, Zf), pr(x0, hw, Zf)]
        cv.poly(front, fc(shade(plaster, -0.25)))
        cv.poly([pr(x0, 0, Zf), pr(x1, 0, Zf), pr(x1, hw * 0.3, Zf), pr(x0, hw * 0.3, Zf)], fc(shade(lower, -0.25)))
        # timber posts
        for i in range(5):
            xx = x0 + wd * i / 4
            cv.poly([pr(xx - 0.08, 0, Zf), pr(xx + 0.08, 0, Zf), pr(xx + 0.08, hw, Zf), pr(xx - 0.08, hw, Zf)],
                    fc(shade(hx('#4a3a30'), -0.25)))
        # windows: aluminium frames, sky reflections, some with curtains; entrance door
        nwin = 3
        for i in range(nwin):
            a = x0 + wd * (0.08 + i * 0.3)
            b = a + wd * 0.22
            if i == 1:
                # entrance: sliding frosted-glass doors under a small canopy
                cv.poly([pr(a, 0, Zf), pr(b, 0, Zf), pr(b, hw * 0.72, Zf), pr(a, hw * 0.72, Zf)], fc(hx('#b8bcc0')))
                cv.poly([pr(a + 0.1, 0.05, Zf), pr(b - 0.1, 0.05, Zf), pr(b - 0.1, hw * 0.68, Zf),
                         pr(a + 0.1, hw * 0.68, Zf)], fc(hx('#dfe6ea')))
                cv.line([pr((a + b) / 2, 0.05, Zf), pr((a + b) / 2, hw * 0.68, Zf)], max(0.6, 0.06 * k), fc(hx('#9aa0a6')))
                cv.poly([pr(a - 0.3, hw * 0.78, Zf - 0.9), pr(b + 0.3, hw * 0.78, Zf - 0.9), pr(b + 0.3, hw * 0.84, Zf),
                         pr(a - 0.3, hw * 0.84, Zf)], fc(shade(hx('#6f7a90'), 0.5)))
                continue
            ya, yb = hw * 0.32, hw * 0.78
            cv.poly([pr(a, ya, Zf), pr(b, ya, Zf), pr(b, yb, Zf), pr(a, yb, Zf)], fc(hx('#c6cacc')))
            # glass: sky reflection gradient (pale top -> deeper blue), diagonal glare
            m = 0.07
            gl = [pr(a + m, ya + m, Zf), pr(b - m, ya + m, Zf), pr(b - m, yb - m, Zf), pr(a + m, yb - m, Zf)]
            cv.poly(gl, fc(hx('#5e86bc')))
            cv.poly([pr(a + m, (ya + yb) / 2, Zf), pr(b - m, (ya + yb) / 2, Zf), pr(b - m, yb - m, Zf),
                     pr(a + m, yb - m, Zf)], fc(hx('#9cc2e6')))
            cv.poly([pr(a + (b - a) * 0.15, yb - m, Zf), pr(a + (b - a) * 0.35, yb - m, Zf),
                     pr(a + (b - a) * 0.2, ya + m, Zf), pr(a + m, ya + m, Zf), pr(a + m, ya + (yb - ya) * 0.3, Zf)],
                    fc(hx('#e4f2fc')))
            if rng.random() < 0.5:
                cv.poly([pr(b - (b - a) * 0.3, ya + m, Zf), pr(b - m, ya + m, Zf), pr(b - m, yb - m, Zf),
                         pr(b - (b - a) * 0.3, yb - m, Zf)], fc(hx('#e8dcc0')))
            cv.line([pr((a + b) / 2, ya, Zf), pr((a + b) / 2, yb, Zf)], max(0.5, 0.05 * k), fc(hx('#b0b4b8')))
        # eave shadow on the wall
        cv.poly([pr(x0, hw, Zf), pr(x1, hw, Zf), pr(x1, hw - 0.45, Zf), pr(x0, hw - 0.45, Zf)],
                fc(hx('#3a4254') * 0.7 + shade(plaster, -0.25) * 0.3))
        # --- roof: front slope (lit), barge edge, eave tile ends, ridge cap + onigawara
        ov = 0.8
        eY = hw + 0.15
        rz = (Zf + Zb) / 2
        rx0, rx1 = x0 - ov, x1 + ov
        e0, e1 = pr(rx0, eY, Zf - ov), pr(rx1, eY, Zf - ov)
        r0, r1 = pr(rx0, ht, rz), pr(rx1, ht, rz)
        tile = hx('#6f7f9e')
        cv.poly([e0, e1, r1, r0], fc(shade(tile, 0.55)))
        # tile columns running down the slope + subtle course lines
        ncol = int(wd / 0.3)
        step = 1 if 0.3 * k >= 3.0 else 2
        for i in range(0, ncol + 1, step):
            xx = rx0 + (rx1 - rx0) * i / ncol
            cv.line([pr(xx, eY, Zf - ov), pr(xx, ht, rz)], max(0.45, 0.04 * k),
                    fc(hx('#4c5a78') * 0.55 + shade(tile, 0.55) * 0.45))
        for j in range(1, 7):
            f_ = j / 7
            yy = eY + (ht - eY) * f_
            zz = (Zf - ov) + (rz - (Zf - ov)) * f_
            cv.line([pr(rx0, yy, zz), pr(rx1, yy, zz)], max(0.4, 0.03 * k),
                    fc(hx('#a8b6d2') * 0.35 + shade(tile, 0.55) * 0.65))
        # sun glint band near the ridge
        cv.poly([pr(rx0, ht - 0.5, rz - 0.9), pr(rx1, ht - 0.5, rz - 0.9), r1, r0], fc(hx('#aebbd8')))
        # eave: tile-end row (bright lip) with a dark underside line
        cv.poly([e0, e1, pr(rx1, eY - 0.25, Zf - ov), pr(rx0, eY - 0.25, Zf - ov)], fc(hx('#2e3446')))
        cv.poly([e0, e1, pr(rx1, eY + 0.12, Zf - ov + 0.1), pr(rx0, eY + 0.12, Zf - ov + 0.1)], fc(hx('#c0cae0')))
        # barge board (gable edge) on the visible side
        bx = rx0 if left_side else rx1
        cv.line([pr(bx, eY, Zf - ov), pr(bx, ht, rz), pr(bx, eY, Zb + ov)], max(0.8, 0.14 * k), fc(hx('#3a4050')))
        # ridge cap with lit top and onigawara at the ends
        cv.poly([pr(rx0 + 0.2, ht - 0.05, rz), pr(rx1 - 0.2, ht - 0.05, rz), pr(rx1 - 0.2, ht + 0.35, rz),
                 pr(rx0 + 0.2, ht + 0.35, rz)], fc(hx('#3c4458')))
        cv.poly([pr(rx0 + 0.2, ht + 0.25, rz), pr(rx1 - 0.2, ht + 0.25, rz), pr(rx1 - 0.2, ht + 0.38, rz),
                 pr(rx0 + 0.2, ht + 0.38, rz)], fc(hx('#d4ddf0')))
        for ex in (rx0 + 0.2, rx1 - 0.2):
            cv.poly([pr(ex - 0.25, ht - 0.05, rz), pr(ex + 0.25, ht - 0.05, rz), pr(ex + 0.3, ht + 0.6, rz),
                     pr(ex - 0.3, ht + 0.6, rz)], fc(hx('#353c4e')))
        # block wall in front of the plot (lit top edge, block joints)
        if prm.get('block'):
            bz = Zf - 3.0
            bx0, bx1 = x0 - 2.0, x1 + 2.5
            gap0, gap1 = x0 + wd * 0.3, x0 + wd * 0.55
            for (u0, u1) in ((bx0, gap0), (gap1, bx1)):
                cv.poly([pr(u0, 0, bz), pr(u1, 0, bz), pr(u1, 1.2, bz), pr(u0, 1.2, bz)], fc(shade(hx('#bdb8ac'), -0.2)))
                cv.poly([pr(u0, 1.2, bz), pr(u1, 1.2, bz), pr(u1, 1.3, bz + 0.15), pr(u0, 1.3, bz + 0.15)],
                        fc(hx('#f2eee2')))
                for j in range(1, 6):
                    cv.line([pr(u0, 0.2 * j, bz), pr(u1, 0.2 * j, bz)], 0.5,
                            fc(hx('#8e8a80') * 0.5 + shade(hx('#bdb8ac'), -0.2) * 0.5))
            self._ground_shadow(cv, [(bx0, bz), (bx1, bz), (bx1, bz + 0.15), (bx0, bz + 0.15)], 1.2, fc, 0.45)

    # ------------------------------------------------------------------ ground plane plate
    def _build_ground(self):
        W, H = self.W, self.H
        q = 1.5
        self.q = q
        mx, my = 0.04 * W, 2.0
        self.gmx, self.gmy = mx, my
        hz, f, h, px = self.hz, self.f, self.h, self.px
        Wp = int((W + 2 * mx) * q)
        Hp = int((H - hz + my) * q) + 2
        J, I = np.meshgrid(np.arange(Wp, dtype=np.float64), np.arange(Hp, dtype=np.float64))
        x0 = (J + 0.5) / q - mx
        y0 = hz - my + (I + 0.5) / q
        v = np.maximum(y0 - hz, 0.03)
        Z = np.minimum(f * h / v, 6000.0)
        X = (x0 - px) * Z / f
        fpX = Z / (f * q)
        fpZ = np.minimum(Z * Z / (f * h * q), 4000)
        Xt = self.Xt
        dxt = X - Xt
        rng = np.random.default_rng(71)

        def tex(t, sx, sz, X_=X, Z_=Z):
            return P.sample_tile(t, (X_ / sx) % 256, (Z_ / sz) % 256)

        def amp_for(fp, wl):
            return P.sstep(1.2, 0.3, fp / wl)

        # ---------------- paddies. Two kinds of plot: lush summer rice (a raised canopy plane 0.42 m high whose
        # near edges show stalk walls) and flooded young-rice plots (a water mirror at ground level with rows of
        # tufts converging to the vanishing point). All plots are framed by raised earth ridges (aze, 0.22 m)
        # traced in true 3D: lit grassy top, shadowed face toward the camera, lit/shaded side faces.
        ze = [16.2]
        while ze[-1] < 7000:
            ze.append(ze[-1] + rng.uniform(14, 26) * (1 + ze[-1] / 400))
        ze = np.array(ze)
        offs = rng.uniform(0, 40, len(ze) + 1)
        spans = rng.uniform(11, 20, len(ze) + 1)
        lw = 0.5
        TRK = 3.6

        def hsh(v, a, b=43758.5453):
            return ((np.sin(v * a) * b) % 1.0 + 1.0) % 1.0

        def field_info(Xw, Zw):
            iz_ = np.clip(np.searchsorted(ze, Zw), 1, len(ze) - 1)
            dz_ = np.minimum(Zw - ze[iz_ - 1], ze[iz_] - Zw)
            adx = np.abs(Xw - Xt)
            xf_ = (adx - TRK + offs[iz_]) / spans[iz_]
            ix_ = np.floor(xf_)
            dx_ = np.minimum(xf_ - ix_, 1 - (xf_ - ix_)) * spans[iz_]
            dx_ = np.where(adx < TRK + lw + 0.1, np.abs(adx - TRK), dx_)
            inf = (dx_ > lw) & (dz_ > lw) & (adx > TRK) & (Zw > ze[0])
            fid_ = (iz_ * 131 + ix_ * 17 + (Xw > Xt) * 7).astype(np.int64)
            return inf, dx_, dz_, fid_, iz_

        def flooded(fid_, iz_, Xw):
            r = hsh(fid_.astype(np.float64), 3.7713, 9173.31)
            near_force = (iz_ <= 2) & (np.abs(Xw - Xt) > TRK) & (np.abs(Xw - Xt) < TRK + 30)
            return (r < 0.42) | (near_force & (r < 0.75))

        hr = 0.42
        kc = self.h / (self.h - hr)
        hv = 0.22
        kv = self.h / (self.h - hv)
        Xc, Zc = X / kc, Z / kc
        inC, dxC, dzC, fidC, izC = field_info(Xc, Zc)
        inG, dx_lev, dz_lev, fidG, iz = field_info(X, Z)
        tallC = inC & ~flooded(fidC, izC, Xc)
        tallG = inG & ~flooded(fidG, iz, X)
        face = (~tallC) & tallG
        fid = fidC
        frand = hsh(fid.astype(np.float64), 12.9898)
        frand2 = hsh(fid.astype(np.float64), 78.233, 12345.678)
        frand3 = hsh(fid.astype(np.float64), 39.3468, 24634.63)
        fpXc, fpZc = fpX / kc, fpZ / kc
        fpx1 = fpXc * q
        fpz1 = fpZc * q
        # per-plot hue & value: yellow-green <-> blue-green, some bleached / some deep
        rice = lerp3(hx('#a4d040'), hx('#3f9a58'), frand ** 1.2)
        rice = lerp3(rice, hx('#c2dc5a'), np.clip(0.3 - frand2, 0, 1) * 2.0)
        rice = lerp3(rice, hx('#2f7a4c'), np.clip(frand2 - 0.72, 0, 1) * 2.2)
        # planted rows along Z (lines converging on the vanishing point); gaps are deep blue-green shadow
        row_p = 0.5
        adxc = np.abs(Xc - Xt)
        fpXa = np.sqrt(fpXc ** 2 + (Xc / np.maximum(Zc, 1e-3) * fpZc) ** 2)
        rowcov = per_cov(adxc, row_p, 0.55, fpXa, phase=frand * 3)
        row_amp = P.sstep(0.6, 0.25, fpXa * q / row_p)
        tuftz = per_cov(Zc, 0.2, 0.7, fpZc, phase=frand2 * 2)
        tz_amp = P.sstep(0.5, 0.2, fpz1 / 0.2)
        sv1 = C.value_noise(Wp, Hp, Wp / 90, Hp / 2.0, seed=81)
        sv2 = C.value_noise(Wp, Hp, Wp / 30, Hp / 1.1, seed=82)
        sv3 = C.value_noise(Wp, Hp, Wp / 6, Hp / 1.6, seed=83)
        streak = (sv1 - 0.5) * 0.5 + (sv2 - 0.5) * 0.3 + (sv3 - 0.5) * 0.25
        leaf = tex(self.noise_a, 0.05, 0.12, X_=Xc, Z_=Zc)
        a_leaf = P.sstep(0.6, 0.2, np.maximum(fpx1 / 0.05, fpz1 / 0.4))
        n2 = tex(self.noise_a, 1.2, 1.2, X_=Xc, Z_=Zc)
        n3 = tex(self.noise_b, 9.0, 9.0, X_=Xc, Z_=Zc)
        a2 = P.sstep(0.5, 0.15, np.maximum(fpx1, fpz1) / 1.2)
        gapc = lerp3(hx('#16503f'), hx('#1f5a5a'), frand)
        # rice density / hue varies inside each plot (sparse sunny yellow-green <-> dense deep emerald)
        dens1 = tex(self.noise_b, 7.0, 9.0, X_=Xc, Z_=Zc)
        dens2 = tex(self.noise_a, 2.2, 3.0, X_=Xc, Z_=Zc)
        dns = np.clip((dens1 - 0.5) * 1.6 + (dens2 - 0.5) * 0.7, -1, 1)
        rice = lerp3(rice, hx('#c6dc50'), np.clip(-dns, 0, 1) * 0.8)
        rice = lerp3(rice, hx('#1f7048'), np.clip(dns, 0, 1) * 0.8)
        rice = lerp3(rice, gapc, (1 - rowcov) * row_amp * (0.5 + 0.25 * np.clip(-dns, 0, 1)))
        rice = lerp3(rice, hx('#d8ee78'), rowcov * row_amp * (0.12 + 0.2 * tuftz * tz_amp))
        rice = lerp3(rice, gapc, rowcov * row_amp * (1 - tuftz) * tz_amp * 0.25)
        # warm near -> cool far colour temperature across the fields
        rice = lerp3(rice, rice * np.array([0.72, 0.92, 1.08], np.float32) + np.array([0.0, 0.02, 0.06], np.float32),
                     P.sstep(25, 320, Zc))
        rice = lerp3(rice, rice * np.array([1.08, 1.04, 0.86], np.float32), P.sstep(60, 12, Zc) * 0.6)
        rice = rice * (0.86 + 0.26 * frand3)[..., None]
        rice = lerp3(rice, hx('#d4d85a'), np.clip(frand3 - 0.8, 0, 1) * 2.0)
        rice = rice * (1 + streak * 0.18 + (n2 - 0.5) * 0.16 * a2 + (n3 - 0.5) * 0.2)[..., None]
        rice = lerp3(rice, COL['rice_hi'], np.clip(streak * 2.2 - 0.25, 0, 1) * 0.35)
        rice = lerp3(rice, COL['rice_hi'], np.clip((leaf - 0.58) * 3, 0, 1) * a_leaf * 0.5)
        rice = lerp3(rice, COL['rice_gap'], np.clip((0.4 - leaf) * 3, 0, 1) * a_leaf * 0.45)
        # canopy darkens toward its edges (the ridge gap) - soft AO
        edge = np.minimum(dxC, dzC)
        rice = rice * (1 - 0.3 * P.sstep(lw + 0.7, lw, edge))[..., None]

        # stalk walls of the tall plots: where the view ray enters the plot below the canopy top
        lo = np.full(Z.shape, 1.0 / kc)
        hi = np.ones(Z.shape)
        fm = face
        for _ in range(7):
            mid = 0.5 * (lo + hi)
            ii_ = field_info(X * mid, Z * mid)
            inside = ii_[0] & ~flooded(ii_[3], ii_[4], X * mid)
            hi = np.where(fm & inside, mid, hi)
            lo = np.where(fm & ~inside, mid, lo)
        se = np.where(face, 0.5 * (lo + hi), 9.0)
        yfrac = np.clip(self.h * (1 - se) / hr, 0, 1)
        Xe, Ze = X * np.minimum(se, 1), Z * np.minimum(se, 1)
        srng = np.random.default_rng(99)
        sg = np.tile(srng.random((6, 160)).astype(np.float32), (3, 3))
        stex = cv2.resize(sg, (3 * 256, 3 * 256), interpolation=cv2.INTER_CUBIC)[256:512, 256:512]
        stex = np.clip((stex - stex.min()) / (stex.max() - stex.min()), 0, 1)
        stex = cv2.resize(stex, (256, 256))
        ucoord = (Xe + Ze * 0.25) / 0.0022 + yfrac * 9.0
        stalk = P.sample_tile(stex, ucoord % 256, (yfrac * 60.0) % 256)
        stalk = stalk * P.sstep(1.0, 0.3, fpX * q / 0.02) + 0.5 * (1 - P.sstep(1.0, 0.3, fpX * q / 0.02))
        wall = lerp3(hx('#1d4a3c'), hx('#3f8040'), P.sstep(0.0, 0.75, yfrac))
        wall = lerp3(wall, hx('#7cb83e'), P.sstep(0.6, 1.0, yfrac) * (0.4 + 0.6 * stalk))
        wall = lerp3(wall, hx('#d2ea72'), P.sstep(0.85, 1.0, yfrac) * np.clip((stalk - 0.45) * 3, 0, 1))
        wall = wall * (0.72 + 0.56 * stalk)[..., None]

        # ---- aze ridges, traced along each view ray between the ridge-top height and the ground
        Zs, Ze_ = Z / kv, Z
        # (a) ridges running across the view (constant Z): first ridge whose far edge lies beyond Zs
        jx = np.clip(np.searchsorted(ze + lw, Zs), 0, len(ze) - 1)
        zf = ze[jx] - lw
        hitX = zf <= Ze_
        sX = np.where(zf <= Zs, 1.0 / kv, zf / np.maximum(Z, 1e-3))
        okX = hitX & (np.abs(X * sX - Xt) > TRK - lw)
        sX = np.where(okX, sX, 9.0)
        # (b) ridges running along the view (constant |X - Xt|) incl. the track-side ridge at TRK
        Xs_ = X / kv
        adx_a = np.abs(Xs_ - Xt)
        adx_b = np.abs(X - Xt)
        izs = np.clip(np.searchsorted(ze, np.maximum(Zs, ze[0] + 1e-3)), 1, len(ze) - 1)
        spn, ofs = spans[izs], offs[izs]
        xfa = (adx_a - TRK + ofs) / spn
        xfb = (adx_b - TRK + ofs) / spn
        hw_ = lw / spn
        near_i = np.round(xfa)
        in_a = (np.abs(xfa - near_i) < hw_) & (adx_a > TRK)
        nxt = np.ceil(xfa)
        st_ = nxt - hw_
        hitZ = (st_ <= xfb) & (adx_b > TRK)
        adx_hit = np.where(in_a, adx_a, st_ * spn + TRK - ofs)
        # track-side ridge [TRK - lw*0.1, TRK + 2 lw]
        t0_, t1_ = TRK - 0.05, TRK + 2 * lw * 0.55
        in_t = (adx_a >= t0_) & (adx_a <= t1_)
        hit_t = (adx_a < t0_) & (adx_b >= t0_)
        adx_hit = np.where(in_t | (hit_t & ~in_a), np.where(in_t, adx_a, t0_), adx_hit)
        hitZ = hitZ | in_t | hit_t
        in_a = in_a | in_t
        X_hit = np.where(X > Xt, Xt + adx_hit, Xt - adx_hit)
        sZ = np.where(in_a, 1.0 / kv, X_hit / np.where(np.abs(X) < 1e-3, 1e-3, X))
        okZ = hitZ & (sZ >= 1.0 / kv - 1e-6) & (sZ <= 1.0) & (Z * sZ > ze[0] - lw) & (np.abs(X) > 0.5)
        sZ = np.where(okZ, sZ, 9.0)
        sL = np.minimum(sX, sZ)
        lev = (sL <= 1.0) & (sL < se) & ~tallC
        lev_top = lev & (sL <= 1.0 / kv + 1e-6)
        lev_face = lev & ~lev_top
        side = lev_face & (sZ < sX)
        yl = np.clip((1 - sL) * self.h / hv, 0, 1)          # 0 at the base .. 1 at the ridge top
        Xl, Zl = X * np.minimum(sL, 1), Z * np.minimum(sL, 1)
        gtex = P.sample_tile(stex, ((Xl + Zl) / 0.004) % 256, (yl * 30.0) % 256)
        gres = P.sstep(1.0, 0.3, fpX * q / 0.02)
        gtex = gtex * gres + 0.5 * (1 - gres)
        # face toward the camera: in shade (sun ahead), cool; side faces: lit on the left plots, shaded right
        f_sh = lerp3(hx('#244c44'), hx('#4c8a4e'), P.sstep(0.0, 1.0, yl))
        f_lit = lerp3(hx('#4c8a3a'), hx('#a6d25a'), P.sstep(0.0, 1.0, yl))
        lit_side = side & (X < Xt)
        fcol = np.where(lit_side[..., None], f_lit, f_sh)
        fcol = fcol * (0.8 + 0.4 * gtex)[..., None]
        # bright sunlit lip along the top edge of every face (grass tips catching the sun)
        lipn = tex(self.noise_a, 0.25, 0.25, X_=Xl, Z_=Zl)
        fcol = lerp3(fcol, hx('#c2e070'), P.sstep(0.8, 0.98, yl) * (0.2 + 0.4 * np.clip((lipn - 0.35) * 2.5, 0, 1)))
        # wet dark contact line where the ridge meets the water / canopy
        fcol = lerp3(fcol, hx('#1a3a3a'), P.sstep(0.18, 0.0, yl) * 0.6)
        # ridge top: sun-baked grass with a worn dirt path along its crown
        ltn = tex(self.noise_a, 0.3, 0.3, X_=Xl, Z_=Zl)
        tcol = lerp3(hx('#8cbc4c'), hx('#b8c880'), np.clip((ltn - 0.45) * 2.5, 0, 1) * 0.5)
        tcol = lerp3(tcol, hx('#4e8a3c'), np.clip((0.45 - ltn) * 3.0, 0, 1) * 0.6)
        tcol = tcol * (0.9 + 0.2 * gtex)[..., None]
        lev_rgb = np.where(lev_top[..., None], tcol, fcol)

        # ---- flooded plots: tufts over a mirror of the sky; plate keeps the tuft/mud colour and a
        # per-pixel reflectance (the reflection itself is added per frame from the rendered sky)
        wet = inG & ~tallG & ~lev & ~face
        adxg = np.abs(X - Xt)
        fpXg = np.sqrt(fpX ** 2 + (X / np.maximum(Z, 1e-3) * fpZ) ** 2)
        wrow = per_cov(adxg, row_p, 0.28, fpXg, phase=hsh(fidG.astype(np.float64), 12.9898) * 3)
        wdot = per_cov(Z, 0.18, 0.55, fpZ, phase=hsh(fidG.astype(np.float64), 78.233) * 2)
        wres = P.sstep(0.6, 0.25, fpXg * q / row_p)
        dres = P.sstep(0.5, 0.2, fpZ * q / 0.18)
        tcov = wrow * (dres * wdot + (1 - dres) * 0.55)
        tcov = tcov * wres + (1 - wres) * 0.16
        # tufts stand up: at grazing angles (far) and toward the sides they hide more of the water
        graze = P.sstep(30, 400, Z) * 0.4 + P.sstep(8, 40, np.abs(X)) * 0.2
        tcov = np.clip(tcov + (1 - tcov) * graze, 0, 1)
        fr2g = hsh(fidG.astype(np.float64), 39.3468, 24634.63)
        tuftc = lerp3(hx('#5ea43c'), hx('#8cc44a'), fr2g)
        tuftc = lerp3(tuftc, hx('#c6e470'), np.clip(wrow * wdot * dres, 0, 1) * 0.35)
        mud = lerp3(hx('#56705e'), hx('#6f8a86'), P.sstep(20, 200, Z))
        wrgb = lerp3(mud, tuftc, tcov)
        fres = 0.62 + 0.3 * P.sstep(15, 250, Z)
        wa = (wet * (1 - tcov) * fres).astype(np.float32)

        # ---- assemble: stalk wall / ridge / water under the canopy
        fC = cv2.GaussianBlur(tallC.astype(np.float32), (0, 0), 0.5)
        fF = cv2.GaussianBlur((face & ~lev).astype(np.float32), (0, 0), 0.5)
        fL = cv2.GaussianBlur(lev.astype(np.float32), (0, 0), 0.5)
        ground = lerp3(hx('#8aba50'), wrgb, wet.astype(np.float32))
        ground = lerp3(ground, lev_rgb, fL)
        ground = lerp3(ground, wall, fF)
        wa = wa * (1 - fL) * (1 - fF)
        canopy_rgb, canopy_a = rice, fC
        rice_mask = np.zeros_like(Z)
        wet_c = wet
        self.wet_frac = wet
        wy_, wx_ = np.nonzero(wet & (wa > 0.3) & (Z < 700) & (Z > 18))
        if len(wy_):
            sel_ = np.random.default_rng(77).choice(len(wy_), size=min(70, len(wy_)), replace=False)
            self.water_pts = np.stack([X[wy_[sel_], wx_[sel_]], Z[wy_[sel_], wx_[sel_]]], 1)
        else:
            self.water_pts = np.zeros((0, 2))
        lev_col = ground
        # ---------------- trackside grass strips & ditches
        strip = box_cov(np.abs(dxt), 1.55, 3.0, fpX)
        grass_col = lerp3(COL['verge'], COL['levee'], tex(self.noise_a, 0.4, 0.4) * 0.6)
        grass_col = lerp3(grass_col, COL['verge_dk'], np.clip(tex(self.noise_a, 0.05, 0.05) - 0.55, 0, 1) * 1.5 *
                          amp_for(np.maximum(fpX, fpZ), 0.2))
        ground = lerp3(ground, grass_col, strip)
        rice_mask = rice_mask * (1 - strip)
        # concrete irrigation ditch along the right of the line: water reflects the sky
        dch = box_cov(dxt, 3.0, 3.55, fpX)
        wat = box_cov(dxt, 3.1, 3.45, fpX)
        ground = lerp3(ground, COL['concrete'], dch)
        wcol = lerp3(COL['water'], hx('#7fb0d8'), np.clip(tex(self.noise_b, 0.8, 3.0) * 0.8, 0, 1) * 0.5)
        ground = lerp3(ground, hx('#4c6a70'), wat)
        rice_mask = rice_mask * (1 - dch)
        wa = wa * (1 - strip) * (1 - dch)
        self.water_mask_world = None

        # ---------------- ballast, sleepers, rails
        # chunky crushed stone (~7 cm) with strong lit-top / dark-gap contrast; mottled clusters far away
        bal = box_cov(np.abs(dxt), -1, 1.55, fpX)
        g1 = tex(self.noise_a, 0.035, 0.035)
        g2 = tex(self.noise_c, 0.3, 0.3)
        fpm = np.maximum(fpX, fpZ)
        ag = amp_for(fpm, 0.1)
        bt = A.ballast_tile(512, seed=3, stones=1700)
        bs_ = 0.0052
        stones = np.dstack([P.sample_tile(bt[..., i].copy(), (X / bs_) % 512, (Z / bs_) % 512) for i in range(3)])
        a_st = amp_for(fpm, 0.05)
        bavg = bt.reshape(-1, 3).mean(0)
        bbase = hx('#ae9e8c') * (1 + (g2 - 0.5) * 0.22)[..., None]
        st_ = (bavg + (stones - bavg) * 1.45) * (bbase / np.maximum(bavg, 1e-3))
        st_ = st_ * np.array([1.04, 1.0, 0.94], np.float32)
        bcol = lerp3(bbase, st_, a_st)
        mott = (1 - a_st) * ag
        bcol = lerp3(bcol, hx('#4c4846'), np.clip(0.44 - g1, 0, 1) * 2.4 * mott)
        bcol = lerp3(bcol, hx('#d6d0c6'), np.clip(g1 - 0.6, 0, 1) * 2.4 * mott)
        # rust / brake-dust brown between and around the rails
        bcol = lerp3(bcol, bcol * np.array([0.8, 0.66, 0.54], np.float32) + np.array([0.05, 0.02, 0.0], np.float32),
                     np.exp(-(dxt / 0.5) ** 2) * 0.35)
        # shoulder slopes down to the verge: darker, cooler
        bcol = bcol * (1 - 0.25 * P.sstep(1.05, 1.55, np.abs(dxt)))[..., None]
        # --- wooden sleepers (creosoted, weathered), bedded with 4.5 cm standing proud of the ballast:
        # lit top face (seen through the height homography Z/ks), shaded camera-facing front face,
        # contact occlusion in the ballast around each sleeper
        hs = 0.045
        ks = self.h / (self.h - hs)
        per, dep, ph0 = 0.6, 0.22, 0.1
        slx_t = box_cov(np.abs(X / ks - Xt), -1, 1.0, fpX / ks)
        slx_g = box_cov(np.abs(dxt), -1, 1.0, fpX)
        top_c = per_cov(Z / ks, per, dep / per, fpZ / ks, phase=ph0) * slx_t
        flen = np.clip(Z * (ks - 1) / ks, 1e-3, per * 0.999)
        front_c = per_cov(Z, per, flen / per, fpZ, phase=ph0) * slx_g * (1 - top_c)
        ao_c = per_cov(Z, per, (dep + 0.16) / per, fpZ, phase=ph0 - 0.08) * box_cov(np.abs(dxt), -1, 1.09, fpX)
        bcol = bcol * (1 - 0.38 * ao_c)[..., None] + hx('#10183a') * (0.04 * ao_c)[..., None]
        sid = np.floor((Z / ks - ph0) / per)
        srnd = ((np.sin(sid * 91.7) * 4375.85) % 1.0 + 1.0) % 1.0
        srnd2 = ((np.sin(sid * 37.1 + 3.0) * 2375.85) % 1.0 + 1.0) % 1.0
        wood = lerp3(hx('#7d6450'), hx('#8a7a6a'), srnd)                   # brown <-> sun-bleached grey
        wood = lerp3(wood, hx('#5a4436'), np.clip(srnd2 - 0.7, 0, 1) * 2.5)  # a few dark fresh ones
        grain = tex(self.noise_c, 0.25, 0.008, X_=X / ks, Z_=Z / ks)        # grain runs across the track
        wood = wood * (0.86 + 0.26 * (grain - 0.5) * amp_for(fpZ / ks, 0.02) + 0.07)[..., None]
        # sunlit top: warm, with a bright far arris (edge facing the sun) and oily dark centre
        zin = ((Z / ks - ph0) % per) / dep
        wood = lerp3(wood, hx('#d8c2a0'), (P.sstep(0.8, 0.97, zin) * P.sstep(1.0, 0.97, zin) * 0.55 *
                                           amp_for(fpZ / ks, 0.03)))
        wood = lerp3(wood, hx('#3e3028'), np.exp(-((X / ks - Xt) / 0.35) ** 2) * 0.3)
        # steel tie plates + spikes under each rail
        tp = np.zeros_like(Z)
        for xr_ in (Xt - 0.5335, Xt + 0.5335):
            tp = np.maximum(tp, box_cov(X / ks, xr_ - 0.12, xr_ + 0.12, fpX / ks))
        wood = lerp3(wood, hx('#4a4644'), tp * 0.9)
        wood = lerp3(wood, hx('#b8b4ae'), tp * per_cov(X / ks, 0.08, 0.3, fpX / ks, phase=0.02) *
                     P.sstep(0.25, 0.4, zin) * P.sstep(0.75, 0.6, zin) * 0.6 * amp_for(fpX / ks, 0.02))
        frontc = hx('#2e2826') * (0.9 + 0.2 * srnd)[..., None]
        bcol = lerp3(bcol, frontc, front_c)
        bcol = lerp3(bcol, wood, top_c)
        self.sleeper_top = top_c * bal
        ground = lerp3(ground, bcol, bal)
        rice_mask = rice_mask * (1 - bal)

        # ---------------- road (crossing): hot noon asphalt
        r0, r1 = self.road
        road = box_cov(Z, r0, r1, fpZ)
        a_n = tex(self.noise_a, 0.012, 0.012)
        a_f = tex(self.noise_c, 0.006, 0.006)
        a_m = tex(self.noise_b, 1.5, 1.2)
        a_s = amp_for(fpZ, 0.02)
        acol = hx('#978b82') * (1 + (a_m - 0.5) * 0.42 + (tex(self.noise_c, 0.25, 0.25) - 0.5) * 0.2)[..., None]
        # bleached lighter toward the sun side (right) and the far lane
        bleach = P.sstep(Xt - 6, Xt + 14, X) * 0.35 + P.sstep(r0, r1, Z) * 0.15
        acol = lerp3(acol, hx('#c8b9a6'), bleach)
        acol = lerp3(acol, hx('#baa58e'), np.clip((tex(self.noise_b, 3.0, 2.0) - 0.55) * 2.5, 0, 1) * 0.5)
        # oil / tyre drips: dark soft blotches concentrated along the lane centres, bluish-black with a sheen
        lane = np.exp(-((Z - 10.85) / 0.45) ** 2) + np.exp(-((Z - 12.75) / 0.45) ** 2)
        oil = P.sstep(0.5, 0.66, tex(self.noise_b, 0.55, 0.35) * 0.7 + tex(self.noise_a, 0.12, 0.09) * 0.3) * lane
        oil = oil + 0.5 * P.sstep(0.7, 0.8, tex(self.noise_c, 1.4, 0.9)) * P.sstep(r0 + 0.3, r0 + 0.8, Z) * P.sstep(r1 - 0.3, r1 - 0.8, Z)
        acol = lerp3(acol, hx('#4c4852'), np.clip(oil, 0, 1) * 0.55)
        # aggregate: light quartz chips and dark stones
        acol = lerp3(acol, hx('#48464e'), np.clip(0.4 - a_n, 0, 1) * 3.6 * a_s)
        acol = lerp3(acol, hx('#ece4d8'), np.clip(a_n - 0.6, 0, 1) * 4.0 * a_s)
        acol = lerp3(acol, hx('#f2ece2'), np.clip(a_f - 0.8, 0, 1) * 4.0 * amp_for(fpZ, 0.01))
        acol = lerp3(acol, hx('#3e3c44'), np.clip(0.18 - a_f, 0, 1) * 4.0 * amp_for(fpZ, 0.01))
        # fine aggregate speckle (white-noise chips, ~1.5 cm) - bright quartz and dark basalt
        spk = np.random.default_rng(5).random((256, 256)).astype(np.float32)
        sp1 = P.sample_tile(spk, (X / 0.02) % 256, (Z / 0.02) % 256)
        a_sp = amp_for(np.maximum(fpX, fpZ), 0.06)
        acol = lerp3(acol, hx('#f8f3ea'), np.clip((sp1 - 0.68) * 4.0, 0, 1) * a_sp * 0.9)
        acol = lerp3(acol, hx('#34323a'), np.clip((0.3 - sp1) * 4.0, 0, 1) * a_sp * 0.7)
        # tyre-worn bands (darker, polished)
        wn = tex(self.noise_b, 0.9, 0.3)
        for zc in (10.4, 11.3, 12.3, 13.2):
            tb_ = box_cov(Z, zc - 0.3, zc + 0.3, fpZ) * (0.7 + 0.6 * wn)
            acol = lerp3(acol, hx('#5c5a62'), np.clip(tb_, 0, 1) * 0.5)
            # polished centre of each wheel path catches the sky (thin paler streak)
            acol = lerp3(acol, hx('#b8b6bc'), box_cov(Z, zc - 0.05, zc + 0.06, fpZ) * 0.25 * a_s)
        # repair patches: fresher, darker asphalt with a glossy tar border
        tar = np.zeros_like(Z)
        for (xa, xb, za, zb) in ((Xt + 9.6, Xt + 12.8, 9.9, 11.4), (Xt - 9.5, Xt - 6.1, 12.1, 13.9),
                                  (Xt + 11.0, Xt + 13.0, 11.8, 13.6), (Xt - 3.6, Xt - 2.2, 9.6, 10.9),
                                  (Xt + 2.4, Xt + 3.5, 12.6, 14.0), (Xt - 14.0, Xt - 11.0, 9.5, 11.0),
                                  (Xt + 5.2, Xt + 7.9, 10.0, 11.2)):
            pm = box_cov(X, xa, xb, fpX) * box_cov(Z, za, zb, fpZ)
            pin = box_cov(X, xa + 0.07, xb - 0.07, fpX) * box_cov(Z, za + 0.07, zb - 0.07, fpZ)
            acol = lerp3(acol, hx('#5e5c64') * (1.0 + (a_m - 0.5) * 0.2 + (a_n - 0.5) * 0.25 * a_s)[..., None], pm * 0.85)
            tar = np.maximum(tar, pm - pin)
        # longitudinal construction joint + transverse seams, and a network of sealed cracks (tar snakes)
        tar = np.maximum(tar, box_cov(Z, 11.78, 11.86, fpZ) * P.sstep(0.3, 0.5, tex(self.noise_b, 2.0, 2.0)))
        for xs_ in (Xt - 7.3, Xt + 4.6, Xt + 10.2):
            wig = (tex(self.noise_a, 0.6, 0.6) - 0.5) * 0.25
            tar = np.maximum(tar, box_cov(X + wig, xs_, xs_ + 0.06, fpX))
        crk = tex(self.noise_c, 0.35, 0.35)
        crk2 = tex(self.noise_b, 0.9, 0.5)
        cl = np.exp(-((crk - 0.5) / 0.03) ** 2) * P.sstep(0.45, 0.65, crk2) * amp_for(fpZ, 0.25)
        tar = np.maximum(tar, np.clip(cl, 0, 1))
        acol = lerp3(acol, hx('#26272e'), tar * 0.85)
        # crossing panels between/around the rails: moulded rubber modules (1 m along the road), anti-slip
        # ribbing, dark seams with a lit lip, steel edge angles, dusty wheel paths, bolt heads
        pan = box_cov(np.abs(dxt), -1, 1.45, fpX)
        pn_ = tex(self.noise_c, 0.3, 0.3)
        pcol = hx('#4a4a50') * (1 + (pn_ - 0.5) * 0.18)[..., None]
        fpXa_ = np.sqrt(fpX ** 2 + (X / np.maximum(Z, 1e-3) * fpZ) ** 2)
        rib = per_cov(X - Xt, 0.06, 0.45, fpXa_, phase=0.01)
        ribz = per_cov(Z, 0.06, 0.45, fpZ, phase=0.02)
        mid_ = box_cov(np.abs(dxt), -1, 0.47, fpX)          # inner panel (between rails): ribs along X
        rib_amp = amp_for(fpXa_, 0.05)
        ribz_amp = amp_for(fpZ, 0.05)
        pat = mid_ * ribz * ribz_amp + (1 - mid_) * rib * rib_amp
        pcol = lerp3(pcol, hx('#6c6c72'), pat * 0.55)
        # ribs catch the high sun: bright specular crest on each rib, strongest toward the sun side / far lane
        rib_hi = per_cov(X - Xt, 0.06, 0.12, fpXa_, phase=0.035) * (1 - mid_) * rib_amp +             per_cov(Z, 0.06, 0.12, fpZ, phase=0.045) * mid_ * ribz_amp
        rsh = 0.45 + 0.55 * P.sstep(r0, r1, Z)
        pcol = lerp3(pcol, hx('#d6d6dc'), rib_hi * rsh * 0.75)
        self._panel_spec = rib_hi * rsh
        pcol = lerp3(pcol, hx('#2e2e34'), (mid_ * (1 - ribz) * ribz_amp + (1 - mid_) * (1 - rib) * rib_amp) * 0.25)
        # far-lit average when the ribs are unresolved
        pcol = lerp3(pcol, hx('#58585e'), (1 - np.maximum(rib_amp, ribz_amp)) * 0.3)
        # dust + wheel-worn lighter paths
        for zc in (10.4, 11.3, 12.3, 13.2):
            pcol = lerp3(pcol, hx('#8a8680'), box_cov(Z, zc - 0.3, zc + 0.3, fpZ) * (0.25 + 0.25 * pn_))
        pcol = lerp3(pcol, hx('#9a948a'), np.clip((tex(self.noise_a, 0.4, 0.4) - 0.55) * 2.5, 0, 1) * 0.35)
        # module seams: dark gap, lit far lip (faces the sun ahead)
        seam = per_cov(Z, 1.0, 0.025, fpZ, phase=r0 + 0.5)
        lip = per_cov(Z, 1.0, 0.02, fpZ, phase=r0 + 0.525)
        pcol = lerp3(pcol, hx('#141418'), seam * 0.95)
        pcol = lerp3(pcol, hx('#a8a8ae'), lip * 0.7)
        for xs_ in (0.47, 1.43):
            pcol = lerp3(pcol, hx('#141418'), box_cov(np.abs(dxt), xs_ - 0.015, xs_ + 0.01, fpX) * 0.9)
        # bolt heads near the module corners
        bz = per_cov(Z, 1.0, 0.06, fpZ, phase=r0 + 0.62) * per_cov(np.abs(dxt), 0.48, 0.12, fpX, phase=0.1)
        pcol = lerp3(pcol, hx('#c4c0b8'), bz * 0.7 * amp_for(fpm, 0.03))
        # steel edge angles along the road edges
        pcol = lerp3(pcol, hx('#c8c4bc'), np.maximum(box_cov(Z, r0 + 0.02, r0 + 0.1, fpZ),
                                                    box_cov(Z, r1 - 0.1, r1 - 0.02, fpZ)))
        pcol = lerp3(pcol, hx('#1a1a20'), np.maximum(box_cov(Z, r0, r0 + 0.02, fpZ), box_cov(Z, r1 - 0.02, r1, fpZ)))
        acol = lerp3(acol, pcol, pan)
        panel_spec = self._panel_spec * pan * road
        tar = tar * (1 - pan)
        # faded, chipped road paint: edge lines, stop bars, a dashed centre line
        chip = np.clip((tex(self.noise_c, 0.04, 0.04) - 0.22) * 4.0, 0, 1) * np.clip((a_m - 0.08) * 3, 0, 1)
        chip = chip * a_s + (1 - a_s) * 0.8
        pcol_w = hx('#f1efe8') * (0.93 + 0.07 * a_n)[..., None]
        brk = 1 - box_cov(np.abs(dxt), -1, 2.2, fpX)
        paint = np.zeros_like(Z)
        for (za, zb) in ((r0 + 0.2, r0 + 0.35), (r1 - 0.35, r1 - 0.2)):
            paint = np.maximum(paint, box_cov(Z, za, zb, fpZ) * brk)
        paint = np.maximum(paint, box_cov(Z, 11.72, 11.88, fpZ) * per_cov(X, 5.0, 0.55, fpX, phase=0.8) *
                           (1 - box_cov(np.abs(dxt), -1, 5.0, fpX)))
        for (xa, xb) in ((Xt + 3.8, Xt + 4.3), (Xt - 4.3, Xt - 3.8)):
            zlo, zhi = (r0 + 0.4, 11.7) if xa > Xt else (11.9, r1 - 0.4)
            paint = np.maximum(paint, box_cov(X, xa, xb, fpX) * box_cov(Z, zlo, zhi, fpZ))
        acol = lerp3(acol, pcol_w, paint * chip * 0.92)
        # hot-noon sheen: grazing glare toward the sun (upper right) - brightest far and right
        sx_scr = (X / np.maximum(Z, 0.1)) * f + px
        sheen = np.exp(-((sx_scr - self.sun_screen[0]) / (0.28 * W)) ** 2) * P.sstep(r0, r1 + 1, Z)
        acol = lerp3(acol, hx('#e4e2e2'), sheen * 0.14)
        self.road_tar = tar * road
        self.road_sheen = sheen * road
        # soft road shoulders
        sh_z = np.maximum(box_cov(Z, r0 - 0.5, r0, fpZ), box_cov(Z, r1, r1 + 0.9, fpZ))
        ground = lerp3(ground, lerp3(COL['verge'], COL['ballast'], 0.35), sh_z * (1 - bal))
        ground = lerp3(ground, acol, road)
        rice_mask = rice_mask * (1 - road) * (1 - sh_z)
        # near verge in front of the road (grassy bank)
        near = box_cov(Z, -10, r0 - 0.5, fpZ) * (1 - bal)
        vcol = lerp3(COL['verge'], COL['verge_dk'], np.clip(tex(self.noise_a, 0.3, 0.3) - 0.3, 0, 1))
        vcol = lerp3(vcol, COL['levee'], np.clip(tex(self.noise_a, 0.07, 0.07) - 0.6, 0, 1) * 2)
        ground = lerp3(ground, vcol, near)
        rice_mask = rice_mask * (1 - near)

        # rails (drawn with their height: ground-equivalent positions scale by k about the camera).
        # Both rails: identical steel - dark rusty web on the visible side face, a lighter head side with a
        # sharp bright arris, a narrow mirror-bright running band on top; each casts a shadow on the bed.
        kz = self.h / (self.h - 0.16)
        spec = np.zeros_like(Z)
        for xr in (Xt - 0.5335, Xt + 0.5335):
            hw = 0.034
            rsh = box_cov(X, xr - hw - 0.2, xr - hw, fpX) * (1 - road)
            ground = lerp3(ground, ground * np.array([0.42, 0.46, 0.62], np.float32), rsh * 0.9)
            face = xr + hw
            fa, fb = sorted((face * kz, face))
            fc_ = box_cov(X, fa, fb, fpX) * (1 - road)
            Xs = np.where(np.abs(X) < 1e-4, -1e-4, X)
            yf = np.clip((1 - face / Xs) / (1 - 1 / kz), 0, 1)
            fcol = lerp3(hx('#6a4c3c'), hx('#3a2e2a'), P.sstep(0.08, 0.16, yf))          # foot -> web (rust)
            fcol = lerp3(fcol, hx('#6e6a68'), P.sstep(0.66, 0.72, yf))                     # head side (steel)
            fcol = lerp3(fcol, hx('#dfe4e8'), P.sstep(0.9, 0.97, yf))                      # lit arris
            # at distance the face is sub-pixel: average
            fres = amp_for(fpX, 0.02)
            fcol = lerp3(hx('#4a3c36'), fcol, fres)
            ground = lerp3(ground, fcol, fc_)
            # flangeway groove through the crossing
            grv = box_cov(X, (xr + hw) * kz, (xr + hw) * kz + 0.07, fpX) * road
            ground = lerp3(ground, hx('#141418'), grv * 0.9)
            hc = box_cov(X, (xr - hw) * kz, (xr + hw) * kz, fpX)
            band = box_cov(X, (xr - hw + 0.02) * kz, (xr + hw - 0.01) * kz, fpX)
            gl = np.clip(np.log(Z / 8.0) / 3.5, 0, 1)
            head = lerp3(hx('#8a8e94'), hx('#b6bcc2'), gl)
            head = lerp3(head, lerp3(hx('#e8eef2'), hx('#fbfdff'), gl), np.clip(band / np.maximum(hc, 1e-3), 0, 1))
            ground = lerp3(ground, head, hc)
            spec = np.maximum(spec, band * (0.5 + 0.5 * gl))
        self.rail_kz = kz

        # dappled shadow of an off-screen roadside tree over the near-left ground
        self.near_shadow = self._near_shadow(X, Z)
        # painterly leaf-tip strokes over the canopy: perspective-scaled dashes, light and dark
        canopy_rgb = self._rice_strokes(canopy_rgb, fC, Zc, kc, frand, strength=0.5)
        gmask = np.clip(near + strip, 0, 1) * (1 - bal) * (1 - road)
        ground = self._rice_strokes(ground, gmask, Z, 1.0, frand * 0 + 0.3, seed=321, density=16, length=0.22,
                                    lit=hx('#b8dc5a'), dark_mul=(0.45, 0.6, 0.6), vertical=True)
        # rice canopy occludes whatever lies behind/below it
        ground = lerp3(ground, canopy_rgb, canopy_a)
        rice_mask = np.maximum(rice_mask, canopy_a)

        # ---------------- baked cast shadows of the static structures (sun high, ahead-right)
        sh = np.zeros((Hp, Wp), np.float32)
        self._bake_shadows(sh)
        ground = lerp3(ground, ground * np.array([0.36, 0.5, 0.64], np.float32) + hx('#1c4f66') * 0.16, sh * 0.9)
        if os.environ.get('S02_DBG'):
            cv2.imwrite(os.environ['S02_DBG'] + '/shmap.png', (sh * 255).astype(np.uint8))

        # ---------------- aerial perspective
        aer = 1 - np.exp(-Z / 800.0)
        aer = aer ** 1.25
        ground = lerp3(ground, lerp3(COL['haze'], COL['haze_far'], P.sstep(400, 3000, Z)), aer * 0.9)
        # faint bright band just below the horizon (distant paddies glowing in the haze)
        ground = ground + (hx('#f3fbff') - ground) * (np.exp(-(Z / 2500.0) ** -2) * 0.35)[..., None]
        ground = cv2.GaussianBlur(ground.astype(np.float32), (0, 0), 0.45)
        rice_mask = rice_mask * (1 - aer * 0.5)
        wa = wa * (1 - bal) * (1 - road) * (1 - sh_z) * (1 - near) * (1 - canopy_a)
        wa = np.maximum(wa, wat * 0.7 * (1 - bal) * (1 - road) * (1 - sh_z) * (1 - near))
        wa = wa * (1 - aer * 0.6) * (1 - sh * 0.35)
        spec = np.maximum(spec, self.road_tar * 0.18 + self.road_sheen * 0.08 + panel_spec * 0.3)
        self.gplate = np.dstack([ground, rice_mask, spec, wa]).astype(np.float32)
        self.gWp, self.gHp = Wp, Hp

    def _road_text(self, acol, X, Z, fpX, fpZ, Xt):
        """Painted road lettering 'tomare' (stop) in the near lane, glyphs stretched along the direction of
        travel (-X), read top-to-bottom by an approaching driver."""
        masks = [A.text_mask(ch, 160) for ch in (chr(0x3068), chr(0x307e), chr(0x308c))]
        if any(m is None for m in masks):
            return acol
        Lc, Wc = 1.3, 1.2
        Xs = Xt + 4.95
        zc = 0.5 * (self.road[0] + 0.5 * (self.road[0] + self.road[1]))
        cov = np.zeros(X.shape, np.float32)
        for i, m in enumerate(masks):
            m = cv2.copyMakeBorder(m, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
            gh, gw = m.shape
            u = (X - (Xs + i * Lc * 1.12)) / Lc
            v = (zc - Z) / Wc + 0.5
            mu = (u * (gh - 1)).astype(np.float32)
            mv = (v * (gw - 1)).astype(np.float32)
            smp = cv2.remap(m.astype(np.float32), mv, mu, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0)
            cov = np.maximum(cov, smp)
        cov = np.clip((cov - 0.35) * 3.0, 0, 1)
        wear = np.clip((P.sample_tile(self.noise_c, (X / 0.05) % 256, (Z / 0.05) % 256) - 0.25) * 3, 0.55, 1)
        return lerp3(acol, COL['paint'] * 0.97, cov * wear * 0.92)

    def _rice_strokes(self, rgb, mask, Zc, kc, frand, seed=123, density=22, length=0.3, lit=None,
                      dark_mul=(0.55, 0.68, 0.72), vertical=False, strength=1.0):
        Hp, Wp = mask.shape
        rng = np.random.default_rng(seed)
        ss = 2
        n = int(Wp * Hp / density)
        xs = rng.uniform(0, Wp, n)
        ys = rng.uniform(0, Hp, n)
        ix, iy = xs.astype(int), ys.astype(int)
        keep = mask[iy, ix] > 0.5
        xs, ys, ix, iy = xs[keep], ys[keep], ix[keep], iy[keep]
        Zp = Zc[iy, ix]
        L = np.clip(length * self.f * self.q / (Zp * kc) * (self.h - 0.7 * (kc > 1.01)) / self.h * 3.0, 1.2, 22.0)
        # keep density roughly constant in screen space: thin the near strokes less
        kind = rng.random(len(xs))
        ang = np.radians(rng.normal(-8, 18, len(xs))) if not vertical else np.radians(rng.normal(-80, 14, len(xs)))
        th = np.clip(L * 0.16, 0.7, 2.6)
        acc_l = np.zeros((Hp * ss, Wp * ss), np.float32)
        acc_d = np.zeros((Hp * ss, Wp * ss), np.float32)
        for i in range(len(xs)):
            x0, y0 = xs[i], ys[i]
            dx, dy = math.cos(ang[i]) * L[i] * 0.5, math.sin(ang[i]) * L[i] * 0.5
            p0 = (int((x0 - dx) * ss * 16), int((y0 - dy) * ss * 16))
            p1 = (int((x0 + dx) * ss * 16), int((y0 + dy) * ss * 16))
            tgt = acc_l if kind[i] < 0.55 else acc_d
            cv2.line(tgt, p0, p1, float(0.5 + 0.5 * rng.random()), max(1, int(th[i] * ss)), cv2.LINE_AA, shift=4)
        acc_l = cv2.resize(acc_l, (Wp, Hp), interpolation=cv2.INTER_AREA) * mask
        acc_d = cv2.resize(acc_d, (Wp, Hp), interpolation=cv2.INTER_AREA) * mask
        fade = P.sstep(1500, 300, Zc)
        litc = lerp3(COL['rice_hi'], hx('#e8f59a'), frand * 0.5) if lit is None else lit
        rgb = lerp3(rgb, litc, np.clip(acc_l * 0.55 * fade * strength, 0, 1))
        rgb = lerp3(rgb, rgb * np.array(dark_mul, np.float32), np.clip(acc_d * 0.6 * fade * strength, 0, 1))
        return rgb

    def _near_shadow(self, X, Z):
        """Dappled shadow of an off-screen roadside tree over the near-left ground (world space)."""
        X = np.atleast_2d(X)
        Z = np.atleast_2d(Z)
        dap = P.sample_tile(self.noise_a, (X / 0.11 + 207) % 256, (Z / 0.11 + 40) % 256)
        dap = 0.5 + (dap - 0.5) * 1.6 + (P.sample_tile(self.noise_c, (X / 0.05) % 256, (Z / 0.05) % 256) - 0.5) * 0.35
        v = dap * 0.55 + 0.4 * P.sstep(0.0, -5.0, X) + 0.2 * P.sstep(12.8, 8.0, Z)
        left = P.sstep(0.47, 0.52, v) * P.sstep(13.2, 10.5, Z) * P.sstep(1.0, -1.5, X)
        # big dappled crown shadow of an off-screen roadside tree (right, beyond the road) over the right lane,
        # with its trunk's shadow leading in from the right edge
        u = (X - 9.0) / 2.4
        w = (Z - 12.6) / 1.8
        big = P.sample_tile(self.noise_b, (X / 0.1 + 90) % 256, (Z / 0.1 + 13) % 256)
        dap2 = P.sample_tile(self.noise_a, (X / 0.05 + 30) % 256, (Z / 0.05 + 77) % 256)
        dap2 = 0.5 + (dap2 - 0.5) * 1.6 + (P.sample_tile(self.noise_c, (X / 0.03) % 256, (Z / 0.03) % 256) - 0.5) * 0.35
        blob = 1 - np.sqrt(u * u + w * w) + (big - 0.5) * 1.2
        crown = P.sstep(0.0, 0.04, blob) * P.sstep(0.4, 0.45, dap2 + np.clip(blob, 0, 1) * 0.1)
        tr_d = np.abs((Z - 12.4) - (X - 9.6) * 0.48) / np.sqrt(1 + 0.48 ** 2)
        trunk = P.sstep(0.2, 0.14, tr_d) * P.sstep(9.2, 9.8, X) * P.sstep(16.5, 15.5, X)
        return np.maximum(left, np.maximum(crown, trunk)).astype(np.float32)

    def _gproj(self, Xw, Zw):
        """world ground point -> ground plate pixel (t=0 camera)."""
        x = self.px + self.f * Xw / Zw
        y = self.hz + self.f * self.h / Zw
        return (x + self.gmx) * self.q, (y - self.hz + self.gmy) * self.q

    def _shadow_poly(self, pts3):
        Lx, Ly, Lz = self.Lsh
        out = []
        for (X, Y, Z) in pts3:
            gx, gz = X - Y * Lx / Ly, Z - Y * Lz / Ly
            if gz < 0.6:
                out.append((np.nan, np.nan))
                continue
            out.append(self._gproj(gx, gz))
        return out

    def _pole_ins_world(self, p):
        if p.get('kind') == 'conc':
            out = []
            for (ya, half, nins) in [(p['H'] - 0.62, 1.0, 3), (p['H'] - 1.7, 0.75, 2)]:
                for j in range(nins):
                    xo = -half * 0.85 + j * (2 * half * 0.85) / max(nins - 1, 1)
                    out.append((p['X'] + xo, ya + 0.22))
            return out
        return [(p['X'] - 0.45, p['H'] - 0.36), (p['X'] + 0.45, p['H'] - 0.36)]

    def _wire_spans_world(self):
        spans = []
        for poles in (self.poles, self.tpoles):
            ins = [self._pole_ins_world(p) for p in poles]
            for i in range(len(poles) - 1):
                if poles[i]['Z'] > 140:
                    break
                for (a, b) in zip(ins[i], ins[i + 1]):
                    spans.append(((a[0], a[1], poles[i]['Z']), (b[0], b[1], poles[i + 1]['Z'])))
            for a in ins[0]:
                spans.append(((a[0] + 0.1, a[1] + 0.2, poles[0]['Z'] - 34.0), (a[0], a[1], poles[0]['Z'])))
        return spans

    def _bake_shadows(self, sh):
        ss = 2
        Hp, Wp = sh.shape
        m = np.zeros((Hp * ss, Wp * ss), np.uint8)

        def fill(pts):
            p = np.round(np.array(pts) * ss * 16).astype(np.int32)
            cv2.fillPoly(m, [p], 255, cv2.LINE_8, shift=4)

        def column(X, Z, r, Ht):
            pts = [(X - r, 0, Z), (X + r, 0, Z), (X + r, Ht, Z), (X - r, Ht, Z)]
            fill(self._shadow_poly(pts))

        for s_ in (self.sigA, self.sigB):
            X, Z = s_['X'], s_['Z']
            column(X, Z, 0.08, 3.4)
            # crossbuck + lamp bar blobs
            fill(self._shadow_poly([(X - 0.62, 2.85, Z), (X + 0.62, 2.85, Z), (X + 0.62, 3.45, Z), (X - 0.62, 3.45, Z)]))
            fill(self._shadow_poly([(X - 0.55, 2.3, Z), (X + 0.55, 2.3, Z), (X + 0.55, 2.62, Z), (X - 0.55, 2.62, Z)]))
            # barrier machine
            bx = X + 0.33 * (1 if not s_.get('mirror') else -1)
            fill(self._shadow_poly([(bx - 0.2, 0, Z), (bx + 0.2, 0, Z), (bx + 0.2, 1.05, Z), (bx - 0.2, 1.05, Z)]))
        for p in self.poles:
            column(p['X'], p['Z'], 0.15, p['H'])
            fill(self._shadow_poly([(p['X'] - 1.0, p['H'] - 0.7, p['Z']), (p['X'] + 1.0, p['H'] - 0.7, p['Z']),
                                    (p['X'] + 1.0, p['H'] - 0.55, p['Z']), (p['X'] - 1.0, p['H'] - 0.55, p['Z'])]))
        for p in self.tpoles:
            column(p['X'], p['Z'], 0.12, p['H'])
        v = self.vend
        fill(self._shadow_poly([(v['X0'], 0, v['Z']), (v['X1'], 0, v['Z']), (v['X1'], v['H'], v['Z']),
                                (v['X0'], v['H'], v['Z'])]))
        fill(self._shadow_poly([(v['X0'], v['H'], v['Z']), (v['X1'], v['H'], v['Z']), (v['X1'], v['H'], v['Z'] + v['D']),
                                (v['X0'], v['H'], v['Z'] + v['D'])]))
        # guy wire of the hero pole
        hp = self.poles[0]
        gx0, gz0 = hp['X'] + 3.0, hp['Z'] + 1.0
        ptsg = self._shadow_poly([(hp['X'], hp['H'] * 0.7, hp['Z']), (gx0, 0.0, gz0)])
        cv2.polylines(m, [np.round(np.array(ptsg) * ss * 16).astype(np.int32)], False, 255, 2, cv2.LINE_AA, shift=4)
        mm = cv2.resize(m.astype(np.float32) / 255.0, (Wp, Hp), interpolation=cv2.INTER_AREA)
        sh[:] = np.maximum(sh, C.blur(mm, 0.6))
        # overhead wires: thin, slightly softer shadows (sun penumbra)
        wm = np.zeros((Hp * ss, Wp * ss), np.uint8)
        for (A3, B3) in self._wire_spans_world():
            n = 48
            tt = np.linspace(0, 1, n)
            pts3 = [(A3[0] + (B3[0] - A3[0]) * u, A3[1] + (B3[1] - A3[1]) * u - 0.45 * 4 * u * (1 - u),
                     A3[2] + (B3[2] - A3[2]) * u) for u in tt]
            pp = np.array(self._shadow_poly(pts3))
            ok = np.isfinite(pp).all(1)
            if ok.sum() < 2:
                continue
            cv2.polylines(wm, [np.round(pp[ok] * ss * 16).astype(np.int32)], False, 255, 2, cv2.LINE_AA, shift=4)
        wmm = cv2.resize(wm.astype(np.float32) / 255.0, (Wp, Hp), interpolation=cv2.INTER_AREA)
        sh[:] = np.maximum(sh, C.blur(wmm, 0.7) * 0.6)
        sh[:] = np.maximum(sh, self.near_shadow)

    # ------------------------------------------------------------------ foreground grass blades
    def _build_grass(self):
        """Tufts of summer grass on the verges: blades fan out from tuft centres; a few foxtails and
        white fleabane flowers. Precomputed geometry, animated with a travelling wind wave."""
        rng = np.random.default_rng(91)
        Xt = self.Xt
        tufts = []
        # dense verge in front of the road, both sides of the track
        dens = self.noise_b
        for i in range(900):
            Z = 5.4 + rng.random() ** 0.8 * 3.9
            X = rng.uniform(-7.5, 6.5)
            if abs(X - Xt) < 1.85:
                continue
            d = P.sample_tile(dens, np.array([[X / 0.02 % 256]], np.float32), np.array([[Z / 0.02 % 256]], np.float32))[0, 0]
            if rng.random() > 0.55 + 0.9 * d:
                continue
            shrink = 0.55 if (3.2 < X < 7.2 and Z > 7.0) else 1.0      # keep the road lettering readable
            tufts.append((X, Z, rng.uniform(0.5, 1.2) * (0.6 + 0.8 * d) * shrink))
        # tall framing weeds close to the lens at both edges
        for i in range(26):
            side = -1 if i % 2 else 1
            Zf = rng.uniform(5.3, 6.4)
            X = (-3.25 + rng.uniform(0, 0.4)) if side < 0 else (3.0 + rng.uniform(0, 1.2))
            tufts.append((X, Zf, rng.uniform(1.4, 2.1)))
        # along the track beyond the road
        for i in range(160):
            Z = rng.uniform(14.9, 30)
            side = 1 if rng.random() < 0.5 else -1
            X = Xt + side * rng.uniform(1.8, 2.9)
            tufts.append((X, Z, rng.uniform(0.5, 1.0)))
        blades = []
        for (X, Z, sz) in tufts:
            nb = int(rng.integers(5, 13))
            lean0 = rng.normal(0, 0.15)
            for j in range(nb):
                ht = sz * rng.uniform(0.12, 0.42) * (1.5 if rng.random() < 0.08 else 1.0)
                blades.append(dict(X=X + rng.normal(0, 0.05), Z=Z + rng.normal(0, 0.04), h=ht,
                                   w=rng.uniform(0.008, 0.017), lean=lean0 + rng.normal(0, 0.35),
                                   ph=rng.uniform(0, 6.28), tone=rng.uniform(0, 1), kind=0))
            r = rng.random()
            if r < 0.12:
                blades.append(dict(X=X, Z=Z, h=sz * rng.uniform(0.4, 0.62), w=0.004, lean=lean0 + rng.normal(0, 0.2),
                                   ph=rng.uniform(0, 6.28), tone=0.5, kind=1))          # foxtail
            elif r < 0.2:
                blades.append(dict(X=X + 0.03, Z=Z, h=sz * rng.uniform(0.3, 0.5), w=0.003, lean=rng.normal(0, 0.1),
                                   ph=rng.uniform(0, 6.28), tone=0.5, kind=2))          # fleabane
        blades.sort(key=lambda b: -b['Z'])
        self.blades = blades

    # ================================================================== per-frame drawing
    def _ground_frame(self, c):
        W, H = self.W, self.H
        f, h, px, hz, q = self.f, self.h, self.px, self.hz, self.q
        F_ = f * c['fz']
        y_start = int(math.floor(hz))
        A = np.array([[f * h, f * c['cx'], 0], [0, f * h, 0], [0, c['cz'], F_ * h]], np.float64)
        T = np.array([[1, 0, -px], [0, 1, y_start - hz], [0, 0, 1]], np.float64)
        Sm = np.array([[q, 0, q * (px + self.gmx)], [0, q, q * self.gmy], [0, 0, 1]], np.float64)
        M = Sm @ A @ T
        Hg = H - y_start
        g = cv2.warpPerspective(self.gplate[..., :4], M, (W, Hg), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REPLICATE)
        sw = cv2.warpPerspective(np.ascontiguousarray(self.gplate[..., 4:6]), M, (W, Hg),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        return y_start, g, sw[..., 0], sw[..., 1]

    def _world_lowres(self, c, y_start, Hg, qd=4):
        W = self.W
        w, hh = W // qd, Hg
        xs = (np.arange(w, dtype=np.float64) + 0.5) * qd
        ys = y_start + (np.arange(hh, dtype=np.float64) + 0.5)
        Xs, Ys = np.meshgrid(xs, ys)
        F_ = self.f * c['fz']
        v = np.maximum(Ys - self.hz, 0.3)
        Zt = F_ * self.h / v
        X = c['cx'] + (Xs - self.px) * Zt / F_
        return X, Zt + c['cz'], w, hh

    def _ground_dynamics(self, g, c, t, y_start):
        """Wind: gust bands travelling across the paddies (leaves flip, a silvery-yellow sheen rolls over the
        rice and roughens the water) + slowly drifting cloud shadows (world space, temporally coherent)."""
        W = self.W
        Hg = g.shape[0]
        X, Z, w, hh = self._world_lowres(c, y_start, Hg, qd=6)
        # gust bands: fronts perpendicular to the wind, warped by noise, travelling toward the camera-left
        wv = 4.2
        u = X * 0.42 + Z * 0.91
        warp = (P.sample_tile(self.noise_b, (X / 0.6) % 256, (Z / 0.9) % 256) - 0.5) * 9.0
        ph = (u + wv * t + warp) / 16.0
        band = (0.5 + 0.5 * np.cos(ph * 2 * np.pi)) ** 4
        env = P.sample_tile(self.noise_a, ((X + 1.8 * t) / 1.6) % 256, ((Z + 3.8 * t) / 2.4) % 256)
        band = band * np.clip((env - 0.3) * 2.2, 0, 1)
        n2 = P.sample_tile(self.noise_c, ((X + 1.5 * t) / 0.08) % 256, ((Z + 3.4 * t) / 0.16) % 256)
        sheen = band * 0.9 + np.clip((n2 - 0.6) * 2.0, 0, 1) * 0.2 * (0.3 + band)
        sheen = sheen * P.sstep(900, 200, Z)
        # cloud shadows (large, soft) drifting slowly across the fields
        edge = P.sample_tile(self.noise_a, ((X - 2.5 * t) / 0.3) % 256, ((Z - 1.0 * t) / 0.5) % 256) - 0.5
        v = np.zeros_like(X)
        for (bx, bz, rx, rz) in ((22, 44, 34, 16), (-60, 75, 45, 28), (70, 260, 70, 90), (-30, 520, 130, 160),
                                 (140, 700, 160, 200), (-160, 330, 60, 80)):
            d = np.sqrt(((X - bx - 2.5 * t) / rx) ** 2 + ((Z - bz - 1.0 * t) / rz) ** 2)
            v = np.maximum(v, 1 - d)
        cshadow = P.sstep(0.0, 0.1, v + edge * 0.45) * P.sstep(24, 34, Z)
        sheen = cv2.resize(sheen.astype(np.float32), (W, Hg), interpolation=cv2.INTER_LINEAR)
        cshadow = cv2.resize(cshadow.astype(np.float32), (W, Hg), interpolation=cv2.INTER_LINEAR)
        self._band = cv2.resize((band * P.sstep(600, 60, Z)).astype(np.float32), (W, Hg),
                                interpolation=cv2.INTER_LINEAR)
        self._cshadow = cshadow
        rm = g[..., 3]
        rgb = g[..., :3]
        # leaves flip in the gust: a silvery-yellow crest with a darker trough following it
        trough = (0.5 + 0.5 * np.cos(ph * 2 * np.pi + 1.2)) ** 6 * np.clip((env - 0.3) * 2.2, 0, 1) * P.sstep(900, 200, Z)
        trough = cv2.resize(trough.astype(np.float32), (W, Hg), interpolation=cv2.INTER_LINEAR)
        rgb = rgb + (hx('#eef6a8') - rgb) * np.clip(sheen * rm * 0.95, 0, 1)[..., None]
        rgb = rgb * (1 - trough * rm * 0.32)[..., None]
        rgb = rgb * (1 - cshadow[..., None] * np.array([0.36, 0.3, 0.14], np.float32))
        return rgb

    def _water(self, img, grgb, wa, c, t, y0):
        """Flooded paddies / ditch: mirror the already-rendered sky, clouds, mountains and treeline about the
        horizon, with coherent wind ripples; gust bands roughen the surface (softer, paler reflection)."""
        W, H = self.W, self.H
        rows = np.nonzero(wa.max(1) > 0.01)[0]
        if len(rows) == 0:
            return grgb
        ra, rb = int(rows[0]), int(rows[-1]) + 1
        hh = rb - ra
        if not hasattr(self, '_wgrid') or self._wgrid[0].shape != (hh, W):
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(ra, rb, dtype=np.float32) + y0)
            self._wgrid = (xs, ys)
        xs, ys = self._wgrid
        xh, yh = xs[::2, ::2], ys[::2, ::2]
        deph = np.clip((yh - self.hz) / (0.3 * H), 0.02, 1.5)
        n1 = P.sample_tile(self._haze_tex, (xh / (0.02 * W)) % 128, (yh / (0.0035 * W) / deph ** 0.3 - t * 1.3) % 128)
        n2 = P.sample_tile(self._haze_tex, (xh / (0.006 * W) + 50) % 128, (yh / (0.0015 * W) - t * 2.1) % 128)
        n1 = cv2.resize(n1, (W, hh), interpolation=cv2.INTER_LINEAR)
        n2 = cv2.resize(n2, (W, hh), interpolation=cv2.INTER_LINEAR)
        dep = np.clip((ys - self.hz) / (0.3 * H), 0.02, 1.5)
        amp = (0.8 + 4.0 * dep) * self.sc
        band = self._band[ra:rb]
        dx = ((n1 - 0.5) * 0.6 + (n2 - 0.5) * 0.4) * amp * (1 + band * 2)
        dy = (n2 - 0.5) * amp * 0.6 * (1 + band)
        my = np.clip(2 * self.hz - ys + dy, 0, self.hz - 1)
        refl = cv2.remap(img, xs + dx, my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # water tints and darkens the mirror a little; wind-roughened patches scatter the sky light
        refl = refl * np.array([0.84, 0.92, 0.98], np.float32) + np.array([0.0, 0.01, 0.03], np.float32)
        rough = np.clip(band * 0.8, 0, 1)[..., None]
        refl = refl + (hx('#b4d6ec') - refl) * rough * 0.55
        # near water is darker / more saturated (looking down into it), far water a pale sky mirror
        refl = refl * (0.8 + 0.2 * np.clip(1.2 - dep, 0, 1))[..., None]
        # ripple glints: small sparkles riding the ripples, concentrated in the column under the sun
        if not hasattr(self, '_wglint_col'):
            self._wglint_col = np.exp(-((np.arange(W, dtype=np.float32)[None, :] - self.sun_screen[0]) /
                                        (0.3 * W)) ** 2).astype(np.float32)
        gl_ = P.sstep(0.74, 0.86, n2 * 0.65 + n1 * 0.35) * (0.25 + 0.75 * self._wglint_col) * (0.6 + band)
        refl = refl + (gl_ * 0.9)[..., None] * np.array([1.0, 0.97, 0.9], np.float32)
        k = wa[ra:rb, :, None]
        out = grgb.copy()
        out[ra:rb] = grgb[ra:rb] * (1 - k) + refl * k * (1 - self._cshadow[ra:rb, :, None] * 0.25)
        return out

    def _heat_haze(self, img, t):
        """Nostalgic heat shimmer. (1) A temporally coherent refraction warp (~2 px at 1080p) rising slowly,
        strongest in a band just above the far rails around the vanishing point, weaker (~1 px) over the hot
        asphalt. (2) An inferior mirage on the far track bed: shimmering silver pools that mirror the sky and
        the approaching railcar just below the vanishing point."""
        W, H = self.W, self.H
        y0 = int(self.hz - 0.06 * H)
        y1 = int(self.hz + 0.3 * H)
        band = img[y0:y1]
        hh = y1 - y0
        if not hasattr(self, '_hz_grid'):
            xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(hh, dtype=np.float32))
            yy = ys + y0
            prof = np.exp(-((yy - (self.hz + 0.006 * H)) / (0.045 * H)) ** 2)
            prof = prof * (0.35 + 0.65 * np.exp(-((xs - self.px) / (0.2 * W)) ** 2))
            yr0, yr1 = self.hz + self.f * self.h / self.road[1], self.hz + self.f * self.h / self.road[0]
            road = P.sstep(yr0 - 0.02 * H, yr0 + 0.01 * H, yy) * P.sstep(yr1 + 0.01 * H, yr1 - 0.03 * H, yy)
            prof = np.maximum(prof, road * 0.6)
            self._hz_grid = (xs, ys, yy, prof.astype(np.float32))
        xs, ys, yy, prof = self._hz_grid
        xh, yh = xs[::2, ::2], yy[::2, ::2]
        n1 = P.sample_tile(self._haze_tex, (xh / (0.045 * W)) % 128, (yh / (0.008 * W) + t * 4.0) % 128)
        n2 = P.sample_tile(self._haze_tex, (xh / (0.016 * W) + 40) % 128, (yh / (0.004 * W) + t * 7.0) % 128)
        n1 = cv2.resize(n1, (W, hh), interpolation=cv2.INTER_LINEAR)
        n2 = cv2.resize(n2, (W, hh), interpolation=cv2.INTER_LINEAR)
        d = (n1 - 0.5) + 0.6 * (n2 - 0.5)
        amp = 6.5 * self.sc
        dx = d * amp * prof
        dy = (n2 - 0.5) * amp * 0.9 * prof
        out = cv2.remap(band, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # inferior mirage: mirror about a line just below the horizon, in noisy pools over the track bed
        ym = self.hz + 0.0045 * H
        mh = 0.026 * H
        if not hasattr(self, '_mir'):
            ry0, ry1 = int(ym) - y0, int(ym + mh) + 2 - y0
            self._mir = (ry0, ry1)
        ry0, ry1 = self._mir
        mx_, my_ = xs[ry0:ry1], yy[ry0:ry1]
        pool = P.sample_tile(self._haze_tex, (mx_ / (0.018 * W) + t * 0.6) % 128, (my_ / (0.004 * W) + t * 2.5) % 128)
        amask = (np.clip((pool - 0.3) * 3.5, 0, 1) * np.exp(-((mx_ - self.px) / (0.1 * W)) ** 2) *
                 P.sstep(ym - 1, ym + 0.15 * mh, my_) * P.sstep(ym + mh, ym + 0.35 * mh, my_))
        srcy = np.clip(2 * ym - my_ - y0 + (pool - 0.5) * 2.0 * self.sc, 0, hh - 1).astype(np.float32)
        srcx = (mx_ + dx[ry0:ry1] * 1.5).astype(np.float32)
        mir = cv2.remap(out, srcx, srcy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mir = mir * np.array([0.96, 1.0, 1.04], np.float32) + 0.05
        k = (amask * 0.95)[..., None]
        out[ry0:ry1] = out[ry0:ry1] * (1 - k) + mir * k
        # faint bright shimmering sheen right above the far rails + mirage brightening at the vanishing point
        out = out + (np.clip(d, 0, None) * prof * 0.1)[..., None] * np.array([1.0, 0.98, 0.94], np.float32)
        if not hasattr(self, '_mirglow'):
            self._mirglow = (np.exp(-((xs - self.px) / (0.075 * W)) ** 2 - ((yy - (self.hz + 0.003 * H)) / (0.012 * H)) ** 2)
                             ).astype(np.float32)
        fl = 0.85 + 0.15 * math.sin(t * 2.3) * math.sin(t * 1.7 + 1.0)
        gk = (self._mirglow * (0.26 + 0.14 * (n1 - 0.5)) * fl)[..., None]
        out = out + (np.array([0.93, 0.97, 1.0], np.float32) - out) * gk
        img = img.copy()
        img[y0:y1] = out
        return img

    # ------------------------------------------------------------------ structures
    def _cyl(self, cv, xc, yb, yt, rb, rt, colfn, bands=(-1.0, -0.35, 0.2, 0.55, 0.82, 1.0), stripes=None):
        """Draw a vertical cylinder in screen space with cel bands. colfn(ndl, k) -> rgb, k=stripe index."""
        Lx, Ly, Lz = self.L
        segs = stripes if stripes is not None else [(yb, yt, 0, 0.0)]
        for (ya, yb_, k, sl) in segs:
            for i in range(len(bands) - 1):
                a0, a1 = bands[i], bands[i + 1]
                am = 0.5 * (a0 + a1)
                ndl = am * Lx - math.sqrt(max(1 - am * am, 0)) * Lz
                col = colfn(ndl, k, am)

                def pt(a, y):
                    tt = (y - yb) / (yt - yb + 1e-9)
                    r = rb + (rt - rb) * tt
                    return (xc + a * r, y - sl * a * r)
                cv.poly([pt(a0, ya), pt(a1, ya), pt(a1, yb_), pt(a0, yb_)], col)

    def _draw_signal(self, cv, sg, c, t, lamp):
        """Japanese level-crossing signal in local metres (x right, y up) on the plane Z = sg['Z']."""
        Z = sg['Z']
        k = self.kscale(Z, c)
        mir = -1 if sg.get('mirror') else 1
        X0 = sg['X']

        def T(x, y):
            xs, ys = self.proj(X0 + x, y, Z, c)
            return (float(xs), float(ys))

        def Tl(pts):
            return [T(x, y) for (x, y) in pts]
        Lx, Ly, Lz = self.L
        yel, blk = COL['yellow'], COL['black']
        face_ndl = -Lz * 0.9 + 0.12
        # ---- barrier machine housing: a real box (lit top, shaded side, painted front) with contact shadow
        bx = 0.33 * mir
        bw, bh, bd = 0.19, 1.0, 0.3

        def P3(x, y, z):
            xs, ys = self.proj(X0 + x, y, z, c)
            return (float(xs), float(ys))
        self._contact_ao(cv, X0 + bx - bw, X0 + bx + bw, Z, Z + bd, c, grow=(0.2, 0.12, 0.05), alphas=(0.15, 0.28, 0.45))
        xside = bx - bw if (X0 + bx) > c['cx'] else bx + bw
        side_col = shade(hx('#c89a18'), -0.3)
        cv.poly([P3(xside, 0, Z), P3(xside, 0, Z + bd), P3(xside, bh, Z + bd), P3(xside, bh, Z)], side_col)
        for i in range(-2, 6):
            y = i * 0.26
            st = [(Z, y), (Z + bd, y + 0.08), (Z + bd, y + 0.21), (Z, y + 0.13)]
            st = [(z_, min(max(y_, 0.0), bh)) for (z_, y_) in st]
            if st[0][1] >= bh or st[2][1] <= 0:
                continue
            cv.poly([P3(xside, y_, z_) for (z_, y_) in st], shade(blk, -0.3))
        cv.line([P3(xside, 0, Z), P3(xside, bh, Z)], 0.9, hx('#141620'))
        # front (painted texture)
        gq = [P3(bx - bw, bh, Z), P3(bx + bw, bh, Z), P3(bx + bw, 0, Z), P3(bx - bw, 0, Z)]
        PR.blit_quad(cv, self.tx_gb[bool(sg.get('mirror'))], gq)
        # painted finish: warm lit top, cool dark base, glossy sheen streak, sun-side arris
        PF.vgrad(cv, gq, hx('#fff2c8'), 0.22, 0.0, 0.0, 0.4, n=5)
        PF.vgrad(cv, gq, hx('#141a2c'), 0.0, 0.5, 0.35, 1.0, n=8)
        PF.glare(cv, gq, 0.62, 0.14, slant=0.25, a=0.14, v0=0.02, v1=0.7)
        kg = self.kscale(Z, c)
        PF.edge(cv, P3(bx + bw, bh, Z), P3(bx + bw, 0.03, Z), max(0.016 * kg, 1.0), hx('#fff0c8'))
        # cap: dark lid with a sunlit top face seen from above
        cv.poly([P3(bx - bw - 0.02, bh, Z - 0.02), P3(bx + bw + 0.02, bh, Z - 0.02), P3(bx + bw + 0.02, bh + 0.06, Z - 0.02),
                 P3(bx - bw - 0.02, bh + 0.06, Z - 0.02)], hx('#2a2c36'))
        cv.poly([P3(bx - bw - 0.02, bh + 0.06, Z - 0.02), P3(bx + bw + 0.02, bh + 0.06, Z - 0.02),
                 P3(bx + bw + 0.02, bh + 0.06, Z + bd + 0.02), P3(bx - bw - 0.02, bh + 0.06, Z + bd + 0.02)],
                hx('#a4acbc'))
        cv.poly([P3(bx + bw * 0.2, bh + 0.061, Z), P3(bx + bw + 0.02, bh + 0.061, Z),
                 P3(bx + bw + 0.02, bh + 0.061, Z + bd), P3(bx + bw * 0.2, bh + 0.061, Z + bd)], hx('#eef0f4'))
        cv.line([P3(bx - bw - 0.02, bh + 0.06, Z - 0.02), P3(bx + bw + 0.02, bh + 0.06, Z - 0.02)], 0.8, hx('#fff2d0'))
        # ---- concrete footing
        cv.poly(Tl([(-0.2, -0.02), (0.2, -0.02), (0.2, 0.1), (-0.2, 0.1)]), shade(hx('#bdb9ae'), face_ndl))
        cv.poly(Tl([(-0.2, 0.08), (0.2, 0.08), (0.2, 0.12), (-0.2, 0.12)]), shade(hx('#e6e2d8'), 0.5))
        # ---- striped pole
        xc, yb = T(0, 0.1)
        _, yt = T(0, 3.42)
        r = 0.07 * k
        n_st = 13
        segs = []
        for i in range(n_st):
            ya = yb + (yt - yb) * i / n_st
            yb2 = yb + (yt - yb) * (i + 1) / n_st
            segs.append((ya, yb2, i % 2, 0.35))

        def pcol(ndl, kk, am):
            base = yel if kk == 0 else blk
            cc = shade(base, ndl)
            if am > 0.85:
                cc = cc + (np.array([1.0, 0.98, 0.9]) - cc) * (0.35 if kk == 0 else 0.25)
            return cc
        self._cyl(cv, xc, yb, yt, r, r, pcol, stripes=segs)
        # cable conduit running down the pole
        cv.line([T(0.075, 0.4), T(0.075, 2.3)], max(1.0, 0.018 * k), shade(hx('#30343c'), face_ndl))
        # small white ID plate with illegible characters
        cv.poly(Tl([(-0.07, 1.35), (0.07, 1.35), (0.07, 1.62), (-0.07, 1.62)]), shade(hx('#f4f4f0'), face_ndl))
        for j in range(4):
            yy = 1.58 - j * 0.06
            cv.poly(Tl([(-0.045, yy - 0.025), (0.045, yy - 0.025), (0.045, yy), (-0.045, yy)]),
                    shade(hx('#2c3a6a'), face_ndl))
        # ---- direction indicator box
        cy_ = 2.0
        cv.poly(Tl([(-0.2, cy_ - 0.09), (0.2, cy_ - 0.09), (0.2, cy_ + 0.09), (-0.2, cy_ + 0.09)]),
                shade(blk, face_ndl))
        cv.poly(Tl([(-0.2, cy_ + 0.07), (0.2, cy_ + 0.07), (0.2, cy_ + 0.09), (-0.2, cy_ + 0.09)]),
                shade(hx('#5a5e6a'), 0.5))
        arrow_on = 0.85 + 0.15 * math.sin(t * 3)
        for sgn in (-1, 1):
            ax = 0.1 * sgn
            cv.poly(Tl([(ax - 0.06 * sgn, cy_ - 0.05), (ax + 0.06 * sgn, cy_), (ax - 0.06 * sgn, cy_ + 0.05)]),
                    hx('#ffb43c') * (arrow_on if sgn == 1 else 0.35) + hx('#302018') * (0 if sgn == 1 else 0.6))
        # ---- lamp bar and lamps
        ly = 2.5
        cv.poly(Tl([(-0.48, ly - 0.035), (0.48, ly - 0.035), (0.48, ly + 0.035), (-0.48, ly + 0.035)]),
                shade(blk, face_ndl))
        cv.poly(Tl([(-0.48, ly + 0.02), (0.48, ly + 0.02), (0.48, ly + 0.035), (-0.48, ly + 0.035)]),
                shade(hx('#6a6e7a'), 0.5))
        lamp_pos = []
        for j, sgn in enumerate((-1, 1)):
            lx = 0.37 * sgn
            cx_, cy2 = T(lx, ly)
            Rb = 0.215 * k
            cv.circle((cx_, cy2), Rb, shade(blk, face_ndl))
            # backboard rim catches the light (top-right)
            cv.ellipse((cx_, cy2), (Rb, Rb), 0, 200, 340, shade(hx('#70747e'), 0.5))
            cv.circle((cx_, cy2 + 0.01 * k), Rb * 0.93, shade(blk, face_ndl))
            on = lamp[j]
            Rl = 0.11 * k
            off_col = hx('#5b1512')
            on_col = hx('#ff4a2a')
            cv.circle((cx_, cy2), Rl * 1.12, hx('#101116'))
            cv.circle((cx_, cy2), Rl, off_col + (on_col - off_col) * on)
            cv.circle((cx_ - 0.2 * Rl, cy2 + 0.15 * Rl), Rl * 0.55, off_col * 1.2 + (hx('#ffb09a') - off_col * 1.2) * on)
            # hood (visor) over the lens
            cv.ellipse((cx_, cy2 - Rl * 0.15), (Rl * 1.35, Rl * 1.25), 0, 180, 360, shade(hx('#22242c'), face_ndl))
            cv.ellipse((cx_, cy2 - Rl * 0.05), (Rl * 1.12, Rl * 0.95), 0, 180, 360,
                       off_col * 0.6 + (on_col * 0.75 - off_col * 0.6) * on)
            cv.ellipse((cx_, cy2 - Rl * 0.2), (Rl * 1.4, Rl * 0.24), 0, 180, 360, shade(hx('#6a6e78'), 0.5))
            # glass specular dot (sky reflection)
            cv.circle((cx_ + 0.35 * Rl, cy2 - 0.3 * Rl), max(0.6, 0.16 * Rl), hx('#ffe4dc') * (0.5 + 0.5 * on) + 0.2)
            lamp_pos.append((cx_, cy2, Rl))
        # ---- crossbuck (X sign) with diagonal yellow/black stripes and black border
        cyc = 3.12
        for ang in (30.0, -30.0):
            a = math.radians(ang)
            ux, uy = math.cos(a), math.sin(a)
            vx, vy = -uy, ux
            Lb, Wb = 0.64, 0.085
            corners = [(-Lb * ux - Wb * vx, cyc - Lb * uy - Wb * vy), (Lb * ux - Wb * vx, cyc + Lb * uy - Wb * vy),
                       (Lb * ux + Wb * vx, cyc + Lb * uy + Wb * vy), (-Lb * ux + Wb * vx, cyc - Lb * uy + Wb * vy)]
            cv.poly(Tl(corners), shade(blk, face_ndl))
            inner = [((x - 0) * 0.94, cyc + (y - cyc) * 0.8) for (x, y) in corners]
            inner = [(-Lb * 0.96 * ux - Wb * 0.72 * vx, cyc - Lb * 0.96 * uy - Wb * 0.72 * vy),
                     (Lb * 0.96 * ux - Wb * 0.72 * vx, cyc + Lb * 0.96 * uy - Wb * 0.72 * vy),
                     (Lb * 0.96 * ux + Wb * 0.72 * vx, cyc + Lb * 0.96 * uy + Wb * 0.72 * vy),
                     (-Lb * 0.96 * ux + Wb * 0.72 * vx, cyc - Lb * 0.96 * uy + Wb * 0.72 * vy)]
            for (poly, kk) in P.stripes_on_quad(inner, 7, phase=0.0, slant=0.12 * (1 if ang > 0 else -1)):
                cv.poly(Tl(poly), shade(yel if kk == 0 else blk, face_ndl + 0.04))
            # lit top edge (rim light from the high sun)
            top_edge = [corners[3], corners[2], (corners[2][0] - 0.02 * vx, corners[2][1] - 0.025),
                        (corners[3][0] - 0.02 * vx, corners[3][1] - 0.025)]
            cv.poly(Tl(top_edge), hx('#fff4c8') * 0.95)
        # centre bolt plate
        cv.circle(T(0, cyc), 0.05 * k, shade(hx('#3a3c44'), face_ndl))
        # ---- alarm bell / speaker on top
        cv.poly(Tl([(-0.03, 3.38), (0.03, 3.38), (0.03, 3.5), (-0.03, 3.5)]), shade(blk, face_ndl))
        bx_, by_ = T(0, 3.56)
        cv.ellipse((bx_, by_), (0.12 * k, 0.08 * k), 0, 0, 360, shade(hx('#3a3e48'), face_ndl))
        cv.ellipse((bx_, by_), (0.12 * k, 0.08 * k), 0, 180, 300, shade(hx('#9aa0ac'), 0.6))
        cv.ellipse((bx_ + 0.02 * k, by_ - 0.01 * k), (0.07 * k, 0.045 * k), 0, 200, 320, hx('#e6ecf4'))
        cv.poly(Tl([(-0.01, 3.62), (0.01, 3.62), (0.004, 3.78), (-0.004, 3.78)]), shade(blk, face_ndl))
        return lamp_pos

    def arm_angle(self, t):
        """Barrier arms lower from raised (~80 deg) to horizontal during the shot, with a small settle bounce."""
        u = min(max((t - 0.35) / 2.6, 0.0), 1.0)
        e = 1 - (1 - u) ** 3
        ang = 80.0 * (1 - e)
        if u >= 1.0:
            tb = t - 2.95
            ang = -2.2 * math.exp(-tb * 4.0) * math.sin(tb * 13.0)
        return math.radians(max(ang, -1.5) if u >= 1.0 else ang)

    def _draw_arm(self, cv, sg, c, t, lamp_on):
        """Barrier arm pivoting at the machine: a clean round pipe with crisp black/yellow bands, dark contour,
        warm lit top edge and cool underside, red reflector lamps (blink with the signal), rounded end cap,
        counterweight, and its shadow on the ground."""
        mir = -1 if sg.get('mirror') else 1
        X = sg['X'] + 0.33 * mir
        d = sg['arm_dir']
        Z0 = sg['Z'] + 0.15 * d
        L_ = sg['arm']
        th = self.arm_angle(t)
        py = 0.9
        ux, uy = math.cos(th) * d, math.sin(th)          # (dz, dy) along the arm
        nx, ny = -math.sin(th) * d, math.cos(th)         # (dz, dy) across the arm (up when lowered)
        r = 0.055
        yel, blk = COL['yellow'], COL['black']

        def P3(s_, o, xo=0.0):
            return (X + xo, py + uy * s_ + ny * o, Z0 + ux * s_ + nx * o)

        def pr(q):
            a_, b_ = self.proj(q[0], q[1], q[2], c)
            return (float(a_), float(b_))
        # shadow on the ground (drawn translucent before the arm)
        Lx, Ly, Lz = self.L
        sh_pts = []
        for s_ in np.linspace(0.0, L_, 6):
            for o in (-r, r):
                q = P3(s_, o)
                gz = q[2] - q[1] * Lz / Ly
                if gz - c['cz'] < 0.6:
                    continue
                sh_pts.append(pr((q[0] - q[1] * Lx / Ly, 0.0, gz)))
        if len(sh_pts) >= 3:
            hull = cv2.convexHull(np.array(sh_pts, np.float32))[:, 0, :]
            cv.poly([tuple(v) for v in hull], hx('#1e2c44'), 0.5)
        # counterweight
        cw = [pr(P3(-0.55, -0.1)), pr(P3(0.0, -0.1)), pr(P3(0.0, 0.11)), pr(P3(-0.55, 0.11))]
        cv.poly(cw, hx('#1a1c24'))
        cw2 = [pr(P3(-0.53, -0.08)), pr(P3(-0.02, -0.08)), pr(P3(-0.02, 0.09)), pr(P3(-0.53, 0.09))]
        cv.poly(cw2, shade(hx('#3a3e48'), -0.2))
        cv.poly([pr(P3(-0.53, 0.06)), pr(P3(-0.02, 0.06)), pr(P3(-0.02, 0.09)), pr(P3(-0.53, 0.09))], hx('#9aa0ae'))
        # contour: the whole pipe slightly fatter in near-black (reads as a 1px outline)
        kk0 = self.kscale(Z0, c)
        o_m = 1.1 / max(kk0, 1.0)
        n = 24
        segs = []
        for i in range(n):
            s0, s1 = L_ * i / n, L_ * (i + 1) / n
            segs.append((s0, s1, 0.5 * (s0 + s1)))
        order = sorted(range(n), key=lambda i: -(Z0 + ux * segs[i][2]))
        for i in order:
            s0, s1, _ = segs[i]
            om = 1.1 / max(self.kscale(Z0 + ux * s0, c), 1.0)
            cv.poly([pr(P3(s0, -r - om)), pr(P3(s1 + om, -r - om)), pr(P3(s1 + om, r + om)), pr(P3(s0, r + om))],
                    hx('#101118'))
        # striped pipe: bands of ~0.33 m, each with cool underside, body and warm lit top
        nb = max(int(round(L_ / 0.33)), 4)
        for i in sorted(range(nb), key=lambda i: -(Z0 + ux * (i + 0.5) * L_ / nb)):
            s0, s1 = L_ * i / nb, L_ * (i + 1) / nb
            base = yel if i % 2 == 0 else blk
            cv.poly([pr(P3(s0, -r)), pr(P3(s1, -r)), pr(P3(s1, r)), pr(P3(s0, r))], shade(base, -0.2))
            cv.poly([pr(P3(s0, -r * 0.2)), pr(P3(s1, -r * 0.2)), pr(P3(s1, r)), pr(P3(s0, r))], shade(base, 0.15))
            top = hx('#fff4c4') if i % 2 == 0 else hx('#8a8e9c')
            cv.poly([pr(P3(s0, r * 0.45)), pr(P3(s1, r * 0.45)), pr(P3(s1, r * 0.85)), pr(P3(s0, r * 0.85))], top)
        # rounded end cap (red reflective tip)
        tip = []
        for k_ in range(11):
            a_ = -math.pi / 2 + math.pi * k_ / 10
            tip.append(pr(P3(L_ + math.cos(a_) * r * 1.15, math.sin(a_) * r * 1.15)))
        cv.poly([pr(P3(L_, -r)), pr(P3(L_, r))] + tip[::-1], hx('#b82a22'))
        cv.poly([pr(P3(L_, r * 0.3)), pr(P3(L_, r))] + tip[::-1][:4], hx('#ff8a6a'))
        # red reflector lamps on small black housings along the top of the arm (blink with the signal)
        out = []
        for j, s_ in enumerate((L_ * 0.3, L_ * 0.62, L_ * 0.94)):
            on = lamp_on[j % 2]
            kk = self.kscale(Z0 + ux * s_, c)
            rr = max(0.05 * kk, 1.0)
            q = pr(P3(s_, r + 0.05, xo=-0.04 * mir))
            cv.circle(q, rr * 1.35, hx('#111218'))
            cv.circle(q, rr, hx('#5a1410') + (hx('#ff4a2a') - hx('#5a1410')) * on)
            cv.circle((q[0] + rr * 0.3, q[1] - rr * 0.3), rr * 0.35, hx('#ffd0c0') * (0.5 + 0.5 * on))
            out.append((q, kk))
        return out

    def _contact_ao(self, cv, x0, x1, z0, z1, c, grow=(0.3, 0.18, 0.09, 0.03), alphas=(0.12, 0.22, 0.34, 0.5)):
        """Soft contact shadow (ambient occlusion) on the ground around a footprint: nested translucent quads."""
        for e, a_ in zip(grow, alphas):
            pts = [(x0 - e, z0 - e), (x1 + e, z0 - e), (x1 + e, z1 + e), (x0 - e, z1 + e)]
            pp = [tuple(float(v_) for v_ in self.proj(x, 0.0, z, c)) for (x, z) in pts]
            cv.poly(pp, hx('#1c2638'), a_)

    def _draw_vending(self, cv, c, t):
        """Drink vending machine + recycling bin on the shoulder beyond the crossing: painted face textures
        (lit display, glass sky reflection, coin panel, stickers, grime, contour + warm sun-side rim), a
        visible side, and contact shadows."""
        v = self.vend
        X0, X1, Z, D, Hh = v['X0'], v['X1'], v['Z'], v['D'], v['H']

        def pr(x, y, z):
            a_, b_ = self.proj(x, y, z, c)
            return (float(a_), float(b_))
        b0, b1 = X1 + 0.08, X1 + 0.5
        self._contact_ao(cv, X0, b1, Z, Z + D, c)
        # side (-X) face in shade
        PR.blit_quad(cv, self.tx_vside, [pr(X0, Hh, Z + D), pr(X0, Hh, Z), pr(X0, 0, Z), pr(X0, 0, Z + D)])
        # front face
        fq = [pr(X0, Hh, Z), pr(X1, Hh, Z), pr(X1, 0, Z), pr(X0, 0, Z)]
        PR.blit_quad(cv, self.tx_vend, fq)
        # painted finish: sky-lit top -> cooler, darker base; glass glare over the display window; crisp
        # sun-side arris and top edge
        PF.vgrad(cv, fq, hx('#eaf4ff'), 0.2, 0.0, 0.0, 0.35, n=5)
        PF.vgrad(cv, fq, hx('#1a2238'), 0.0, 0.45, 0.3, 1.0, n=8)
        gq = PF.sub_quad(fq, 0.06, 0.18, 0.94, 0.52)
        PF.glare(cv, gq, 0.5, 0.16, slant=0.3, a=0.32)
        PF.glare(cv, gq, 0.72, 0.05, slant=0.3, a=0.28)
        PF.glare(cv, PF.sub_quad(fq, 0.06, 0.78, 0.94, 0.9), 0.55, 0.12, slant=0.2, a=0.25)
        kv = self.kscale(Z, c)
        PF.edge(cv, pr(X1, Hh, Z), pr(X1, 0.05, Z), max(0.018 * kv, 1.0), hx('#fff4dc'))
        PF.edge(cv, pr(X0, Hh, Z), pr(X1, Hh, Z), max(0.014 * kv, 1.0), hx('#ffffff'), 0.85)
        # recycling bin: lit lid (seen from above) + painted front
        zb = Z + 0.05
        cv.poly([pr(b0 - 0.01, 0.82, zb), pr(b1 + 0.01, 0.82, zb), pr(b1 - 0.02, 0.84, zb + 0.34),
                 pr(b0 + 0.02, 0.84, zb + 0.34)], hx('#1a2a44'))
        cv.poly([pr(b0 + 0.005, 0.825, zb + 0.005), pr(b1 - 0.005, 0.825, zb + 0.005), pr(b1 - 0.025, 0.84, zb + 0.33),
                 pr(b0 + 0.025, 0.84, zb + 0.33)], hx('#86b8f0'))
        cv.poly([pr(b1 - 0.1, 0.83, zb + 0.01), pr(b1 - 0.01, 0.83, zb + 0.01), pr(b1 - 0.03, 0.84, zb + 0.3),
                 pr(b1 - 0.1, 0.84, zb + 0.3)], hx('#d8ecff'))
        PR.blit_quad(cv, self.tx_bin, [pr(b0, 0.82, zb), pr(b1, 0.82, zb), pr(b1, 0, zb), pr(b0, 0, zb)])
        wx0, wx1, wy0, wy1 = X0 + 0.06, X1 - 0.06, 0.96, Hh - 0.33
        return pr((wx0 + wx1) / 2, (wy0 + wy1) / 2, Z), (wx1 - wx0) * self.kscale(Z, c)

    def _draw_pole(self, cv, p, c):
        Z = p['Z']
        if Z - c['cz'] < 1.0:
            return None
        k = self.kscale(Z, c)
        xb, yb = self.proj(p['X'], 0, Z, c)
        xt, yt = self.proj(p['X'], p['H'], Z, c)
        xb, yb, yt = float(xb), float(yb), float(yt)
        fog = float(np.clip((Z - 20) / 800, 0, 0.7))
        hzc = hx('#c4e0ee')

        def fg(cc):
            return np.asarray(cc) * (1 - fog) + hzc * fog
        if p.get('kind') == 'conc':
            base = COL['pole_c']
            r0, r1 = 0.17 * k, 0.11 * k
            hs_ = [0.0, 0.3, 0.8, 1.6, p['H'] * 0.55, p['H']]
            grime = [0.6, 0.36, 0.16, 0.05, 0.0]
            gcol = hx('#766a5a')
            segs = []
            for j in range(5):
                _, ya_ = self.proj(p['X'], hs_[j], Z, c)
                _, yb_ = self.proj(p['X'], hs_[j + 1], Z, c)
                segs.append((float(ya_), float(yb_), j, 0.0))

            def colfn(ndl, kk, am):
                cc = shade(base, ndl)
                cc = cc * (1 - grime[kk]) + shade(gcol, ndl) * grime[kk]
                if am > 0.82:
                    cc = cc + (np.array([1.0, 0.98, 0.92]) - cc) * (0.5 - 0.3 * grime[kk])
                return fg(cc)
            if p.get('near'):
                PR.blit_pole(cv, self.tx_pole, xb, yb, yt, r0, r1, fog=fog, hazec=hzc)
            else:
                self._cyl(cv, xb, yb, yt, r0, r1, colfn, stripes=segs)
            if k > 12 and not p.get('near'):
                # rain stains running down from the cross-arm bolts / form seams
                for (xo, h0, h1) in ((-0.05, p['H'] - 0.8, p['H'] - 3.2), (0.07, p['H'] - 1.9, p['H'] - 4.6)):
                    x_, y0_ = self.proj(p['X'] + xo, h0, Z, c)
                    _, y1_ = self.proj(p['X'] + xo, h1, Z, c)
                    cv.poly([(float(x_) - 0.012 * k, float(y0_)), (float(x_) + 0.012 * k, float(y0_)),
                             (float(x_), float(y1_))], fg(shade(hx('#a8a498'), 0.0)))
                for hseam in (3.0, 6.0, 9.0):
                    if hseam < p['H'] - 2:
                        _, ys_ = self.proj(p['X'], hseam, Z, c)
                        rr = (0.17 - 0.06 * hseam / p['H']) * k
                        cv.line([(xb - rr, float(ys_)), (xb + rr, float(ys_))], max(0.5, 0.012 * k),
                                fg(shade(hx('#9c988e'), 0.0)))
            if p.get('sign') and k > 30:
                # vertical enamel advertising plate strapped to the pole (typical rural Japan)
                sx0, sx1 = p['X'] - 0.19, p['X'] + 0.19
                sy0, sy1 = 2.35, 3.55
                q_ = [self.proj(sx0, sy0, Z - 0.2, c), self.proj(sx1, sy0, Z - 0.2, c),
                      self.proj(sx1, sy1, Z - 0.2, c), self.proj(sx0, sy1, Z - 0.2, c)]
                cv.poly([(float(a), float(b)) for a, b in q_], fg(shade(hx('#f2efe4'), -0.15)))
                q2 = [self.proj(sx0, sy1 - 0.28, Z - 0.2, c), self.proj(sx1, sy1 - 0.28, Z - 0.2, c),
                      self.proj(sx1, sy1, Z - 0.2, c), self.proj(sx0, sy1, Z - 0.2, c)]
                cv.poly([(float(a), float(b)) for a, b in q2], fg(shade(hx('#1f5fae'), -0.15)))
                for j in range(4):
                    yy0 = sy1 - 0.4 - j * 0.19
                    q3 = [self.proj(p['X'] - 0.1, yy0 - 0.12, Z - 0.2, c), self.proj(p['X'] + 0.1, yy0 - 0.12, Z - 0.2, c),
                          self.proj(p['X'] + 0.1, yy0, Z - 0.2, c), self.proj(p['X'] - 0.1, yy0, Z - 0.2, c)]
                    cv.poly([(float(a), float(b)) for a, b in q3], fg(shade(hx('#2a2a30') if j != 2 else hx('#c83a2e'), -0.15)))
                # rust drip at the lower edge + straps
                for yy_ in (sy0 + 0.1, sy1 - 0.1):
                    a_, b_ = self.proj(p['X'] - 0.2, yy_, Z - 0.21, c)
                    a2_, b2_ = self.proj(p['X'] + 0.2, yy_, Z - 0.21, c)
                    cv.line([(float(a_), float(b_)), (float(a2_), float(b2_))], max(0.6, 0.02 * k), fg(hx('#5a5048')))
                a_, b_ = self.proj(p['X'] + 0.1, sy0, Z - 0.2, c)
                cv.poly([(float(a_) - 0.015 * k, float(b_)), (float(a_) + 0.015 * k, float(b_)),
                         (float(a_), float(b_) + 0.25 * k)], fg(hx('#9a6a48')))
            if p.get('guy'):
                # guy wire (shibu-sen) to a ground anchor, with the yellow/black guard sleeve
                gx0, gz0 = p['X'] + 3.0, Z + 1.0
                top_ = self.proj(p['X'], p['H'] * 0.7, Z, c)
                bot_ = self.proj(gx0, 0.0, gz0, c)
                cv.line([(float(top_[0]), float(top_[1])), (float(bot_[0]), float(bot_[1]))],
                        max(0.6, 0.012 * self.kscale(gz0, c)), fg(hx('#2a2e36')))
                n_g = 6
                for j in range(n_g):
                    u0, u1 = j / n_g, (j + 1) / n_g
                    Pa = [(gx0 + (p['X'] - gx0) * u * 0.2, p['H'] * 0.7 * u * 0.2, gz0 + (Z - gz0) * u * 0.2)
                          for u in (u0, u1)]
                    pa = [tuple(float(v) for v in self.proj(*q_, c)) for q_ in Pa]
                    wg = 0.045 * self.kscale(gz0, c)
                    cv.line(pa, max(1.0, wg), shade(hx('#f2c21c') if j % 2 == 0 else hx('#23252c'), 0.3))
                a_ = self.proj(gx0, 0.0, gz0, c)
                cv.ellipse((float(a_[0]), float(a_[1])), (0.12 * self.kscale(gz0, c), 0.03 * self.kscale(gz0, c)), 0,
                           hx('#8a8680'))
            # number plate (white/red) and step bolts
            if k > 20:
                _, y1 = self.proj(p['X'], 1.8, Z, c)
                _, y2 = self.proj(p['X'], 2.1, Z, c)
                cv.poly([(xb - 0.1 * k, float(y2)), (xb + 0.1 * k, float(y2)), (xb + 0.1 * k, float(y1)),
                         (xb - 0.1 * k, float(y1))], fg(shade(hx('#f0f0ea'), -0.1)))
                cv.poly([(xb - 0.1 * k, float(y2)), (xb + 0.1 * k, float(y2)), (xb + 0.1 * k, float(y2) + 0.06 * k),
                         (xb - 0.1 * k, float(y2) + 0.06 * k)], fg(hx('#d8412e')))
                for j in range(9):
                    yy = 2.6 + j * 0.45
                    _, ys_ = self.proj(p['X'], yy, Z, c)
                    sgn = 1 if j % 2 else -1
                    rr = (0.17 - 0.06 * yy / p['H'])
                    wb = max(0.5, (0.035 if p.get('near') else 0.025) * k)
                    cv.line([(xb + sgn * rr * 0.8 * k, float(ys_)), (xb + sgn * (rr + 0.13) * k, float(ys_) - 0.02 * k)],
                            wb + 1.0, fg(hx('#1c1e26')))
                    cv.line([(xb + sgn * rr * 0.8 * k, float(ys_)), (xb + sgn * (rr + 0.13) * k, float(ys_) - 0.02 * k)],
                            wb, fg(shade(hx('#6a6e78'), 0.2 * sgn)))
                    if sgn > 0:
                        cv.line([(xb + rr * k, float(ys_) - wb * 0.35), (xb + (rr + 0.13) * k, float(ys_) - 0.02 * k - wb * 0.35)],
                                max(wb * 0.35, 0.5), fg(hx('#fff0d0')))
            # cross arms with insulators
            arms = [(p['H'] - 0.62, 1.0, 3), (p['H'] - 1.7, 0.75, 2)]
            ins = []
            for (ya, half, nins) in arms:
                q = [self.proj(p['X'] - half, ya, Z, c), self.proj(p['X'] + half, ya, Z, c),
                     self.proj(p['X'] + half, ya + 0.1, Z, c), self.proj(p['X'] - half, ya + 0.1, Z, c)]
                cv.poly([(float(a), float(b)) for a, b in q], fg(shade(hx('#6e7480'), -0.1)))
                q2 = [q[3], q[2], (q[2][0], q[2][1] + 0.03 * k), (q[3][0], q[3][1] + 0.03 * k)]
                cv.poly([(float(a), float(b)) for a, b in q2], fg(hx('#d8dde6')))
                # diagonal brace
                cv.line([self.proj(p['X'] - half * 0.6, ya, Z, c), self.proj(p['X'], ya - 0.55, Z, c)],
                        max(0.5, 0.03 * k), fg(shade(hx('#6e7480'), -0.1)))
                for j in range(nins):
                    xo = -half * 0.85 + j * (2 * half * 0.85) / max(nins - 1, 1)
                    ix_, iy_ = self.proj(p['X'] + xo, ya + 0.1, Z, c)
                    ix_, iy_ = float(ix_), float(iy_)
                    cv.ellipse((ix_, iy_ - 0.08 * k), (0.055 * k, 0.1 * k), 0, 0, 360, fg(shade(hx('#e8e4dc'), -0.1)))
                    cv.ellipse((ix_ + 0.015 * k, iy_ - 0.1 * k), (0.03 * k, 0.07 * k), 0, 0, 360,
                               fg(hx('#fffaf0')))
                    ins.append((p['X'] + xo, ya + 0.22))
            if p.get('tr'):
                # pole-top transformer can
                tx, ty = self.proj(p['X'] + 0.32, p['H'] - 3.2, Z, c)
                tx, ty = float(tx), float(ty)
                cw, ch = 0.28 * k, 0.75 * k
                cv.poly([(tx - cw, ty - ch), (tx + cw, ty - ch), (tx + cw, ty), (tx - cw, ty)],
                        fg(shade(hx('#9aa2ae'), -0.15)))
                cv.poly([(tx + cw * 0.4, ty - ch), (tx + cw, ty - ch), (tx + cw, ty), (tx + cw * 0.4, ty)],
                        fg(shade(hx('#b8c0cc'), 0.5)))
                cv.ellipse((tx, ty - ch), (cw, 0.07 * k), 0, 0, 360, fg(hx('#dfe4ea')))
            return ins
        else:
            base = COL['wood']
            r0, r1 = 0.13 * k, 0.1 * k

            def colfn(ndl, kk, am):
                cc = shade(base, ndl)
                if am > 0.82:
                    cc = cc + (np.array([1.0, 0.95, 0.85]) - cc) * 0.4
                return fg(cc)
            self._cyl(cv, xb, yb, yt, r0, r1, colfn)
            q = [self.proj(p['X'] - 0.55, p['H'] - 0.5, Z, c), self.proj(p['X'] + 0.55, p['H'] - 0.5, Z, c),
                 self.proj(p['X'] + 0.55, p['H'] - 0.42, Z, c), self.proj(p['X'] - 0.55, p['H'] - 0.42, Z, c)]
            cv.poly([(float(a), float(b)) for a, b in q], fg(shade(hx('#5a4c40'), -0.1)))
            return [(p['X'] - 0.45, p['H'] - 0.36), (p['X'] + 0.45, p['H'] - 0.36)]

    def _wires(self, cv, poles, ins_list, c, sag, col, width_m, extend_back=True):
        """Catenary wires between consecutive poles (3D) + toward the camera from the first pole."""
        spans = []
        for i in range(len(poles) - 1):
            a, b = ins_list[i], ins_list[i + 1]
            if a is None or b is None:
                continue
            for (pa, pb) in zip(a, b):
                spans.append(((pa[0], pa[1], poles[i]['Z']), (pb[0], pb[1], poles[i + 1]['Z'])))
        if extend_back and ins_list and ins_list[0] is not None:
            for pa in ins_list[0]:
                spans.append(((pa[0] + 0.1, pa[1] + 0.2, poles[0]['Z'] - 34.0), (pa[0], pa[1], poles[0]['Z'])))
        for (A, B) in spans:
            n = 40
            s = np.linspace(0, 1, n)
            Xs = A[0] + (B[0] - A[0]) * s
            Ys = A[1] + (B[1] - A[1]) * s - sag * 4 * s * (1 - s)
            Zs = A[2] + (B[2] - A[2]) * s
            ok = Zs - c['cz'] > 0.6
            if ok.sum() < 2:
                continue
            xs, ys = self.proj(Xs[ok], Ys[ok], Zs[ok], c)
            zmid = float(np.median(Zs[ok]))
            fog = float(np.clip((zmid - 20) / 900, 0, 0.6))
            wcol = col * (1 - fog) + hx('#c4e0ee') * fog
            # split into chunks so near parts are thicker
            m = len(xs)
            chunks = max(1, min(8, m // 5))
            idx = np.linspace(0, m - 1, chunks + 1).astype(int)
            for j in range(chunks):
                i0, i1 = idx[j], idx[j + 1]
                zc = float(np.mean(Zs[ok][i0:i1 + 1]))
                wpx = width_m * self.kscale(zc, c)
                alpha = 1.0 if wpx >= 0.7 else max(wpx / 0.7, 0.25)
                pts = np.stack([xs[i0:i1 + 1], ys[i0:i1 + 1]], 1)
                cv.line(pts, max(wpx, 0.34), wcol, alpha)

    def _grass_arrays(self):
        if hasattr(self, '_ga'):
            return self._ga
        B = self.blades
        g = {k_: np.array([b[k_] for b in B], np.float64) for k_ in ('X', 'Z', 'h', 'w', 'lean', 'ph', 'tone', 'kind')}
        dark = [hx('#1f5a3e'), hx('#2c6e3a'), hx('#3f8a36')]
        lit = [hx('#8cc43c'), hx('#b4dc52'), hx('#d4ec7c')]
        ti = np.minimum((g['tone'] * 3).astype(int), 2)
        shd = self._near_shadow(g['X'][None, :], g['Z'][None, :])[0]
        shd = np.maximum(shd, 0.55 * P.sstep(7.2, 5.6, g['Z']))
        smul = np.array([0.5, 0.6, 0.85])
        sadd = np.array([0.02, 0.05, 0.1])
        g['dark'] = [P.Canvas._col(dark[i] * (1 - sh_ * (1 - smul)) + sadd * sh_) for i, sh_ in zip(ti, shd)]
        g['lit'] = [P.Canvas._col(lit[i] * (1 - sh_ * (1 - smul * 0.9)) + sadd * sh_) for i, sh_ in zip(ti, shd)]
        self._ga = g
        return g

    def _draw_grass(self, cv, c, t):
        g = self._grass_arrays()
        W = self.W
        ssf = cv.ss * 16
        X, Z = g['X'], g['Z']
        Zr = Z - c['cz']
        F_ = self.f * c['fz']
        k = F_ / np.maximum(Zr, 0.05)
        xb = self.px + k * (X - c['cx'])
        yb = self.hz + k * self.h
        hpx = g['h'] * k
        vis = (Zr > 0.5) & (xb > -hpx) & (xb < W + hpx) & (hpx >= 1.0)
        gust = 0.5 + 0.5 * np.sin(t * 1.3 - X * 0.45 - Z * 0.2)
        lean = g['lean'] + 0.1 * gust ** 2 + 0.04 * np.sin(t * 2.7 + g['ph'])
        s_ = np.linspace(0, 1, 6)[None, :]
        ang = lean[:, None] * s_ ** 1.3
        cx_ = xb[:, None] + np.sin(ang) * hpx[:, None] * s_
        cy_ = yb[:, None] - np.cos(ang) * hpx[:, None] * s_
        wpx = np.maximum(g['w'] * k, 0.5)[:, None] * (1 - s_ ** 1.1)
        Lp = np.stack([cx_ - wpx, cy_], -1)
        Rp = np.stack([cx_ + wpx, cy_], -1)
        Mp = np.stack([cx_, cy_], -1)
        blade = np.round(np.concatenate([Lp, Rp[:, ::-1]], 1) * ssf).astype(np.int32)
        litp = np.round(np.concatenate([Mp[:, 2:], Rp[:, 2:][:, ::-1]], 1) * ssf).astype(np.int32)
        stem = np.round(Mp * ssf).astype(np.int32)
        buf = cv.buf
        kind, tone = g['kind'], g['tone']
        dk, lt = g['dark'], g['lit']
        c_stem1, c_head1, c_head2 = P.Canvas._col(hx('#6f9a3a')), P.Canvas._col(hx('#a6b85c')), P.Canvas._col(hx('#eef2b8'))
        c_stem2, c_pet, c_cen = P.Canvas._col(hx('#4f8a3a')), P.Canvas._col(hx('#fbfbf6')), P.Canvas._col(hx('#f2c83a'))
        for i in np.nonzero(vis)[0]:
            kd = kind[i]
            if kd == 0:
                cv2.fillPoly(buf, [blade[i]], dk[i], cv2.LINE_8, shift=4)
                if tone[i] > 0.3:
                    cv2.fillPoly(buf, [litp[i]], lt[i], cv2.LINE_8, shift=4)
            elif kd == 1:
                ki = k[i]
                cv2.polylines(buf, [stem[i]], False, c_stem1, max(1, int(0.006 * ki * cv.ss)), cv2.LINE_8, shift=4)
                hx_, hy_ = Mp[i, -1]
                a_ = lean[i]
                cv.ellipse((hx_ + math.sin(a_) * 0.035 * ki, hy_ + math.cos(a_) * 0.02 * ki),
                           (0.016 * ki + 0.4, 0.05 * ki + 0.4), math.degrees(a_) + 15, hx('#a6b85c'))
                cv.ellipse((hx_ + math.sin(a_) * 0.035 * ki + 0.006 * ki, hy_ + math.cos(a_) * 0.02 * ki - 0.005 * ki),
                           (0.008 * ki + 0.3, 0.042 * ki + 0.3), math.degrees(a_) + 15, hx('#eef2b8'))
            else:
                ki = k[i]
                cv2.polylines(buf, [stem[i]], False, c_stem2, max(1, int(0.004 * ki * cv.ss)), cv2.LINE_8, shift=4)
                fx_, fy_ = Mp[i, -1]
                rr = 0.011 * ki + 0.4
                cv.ellipse((fx_, fy_), (rr * 1.25, rr * 0.7), 0, hx('#fbfbf6'))
                cv.circle((fx_, fy_ - rr * 0.1), rr * 0.4, hx('#f2c83a'))

    def _draw_train(self, img, c, t):
        """A one-car diesel railcar approaching from the far distance (drawn before the heat haze)."""
        Zt = 430.0 - 13.0 * t
        k = self.kscale(Zt, c)
        X0 = self.Xt
        xc, yb = self.proj(X0, 0.0, Zt, c)
        xc, yb = float(xc), float(yb)
        hw, ht = 1.45 * k, 3.9 * k
        pad = 6
        x0, x1 = int(xc - hw - pad), int(xc + hw + pad) + 1
        y0, y1 = int(yb - ht - pad), int(yb + pad) + 1
        cv = P.Canvas(x1 - x0, y1 - y0, ss=4)
        fog = 0.42
        hzc = hx('#d2e8f2')

        def fc(cc):
            return np.asarray(cc) * (1 - fog) + hzc * fog

        def R(xa, ya, xb, yb_, col):
            cv.poly([(xc - x0 + xa * k, yb - y0 - ya * k), (xc - x0 + xb * k, yb - y0 - ya * k),
                     (xc - x0 + xb * k, yb - y0 - yb_ * k), (xc - x0 + xa * k, yb - y0 - yb_ * k)], fc(col))
        R(-1.45, 0.35, 1.45, 3.75, hx('#e9ebe6'))            # body
        R(-1.45, 3.55, 1.45, 3.9, hx('#b8bcc4'))             # roof
        R(-1.45, 0.35, 1.45, 1.0, hx('#2e8a6e'))             # livery band
        R(-1.45, 1.0, 1.45, 1.15, hx('#e6a13a'))
        R(-1.25, 1.6, -0.2, 2.9, hx('#3a4a5e'))              # windows
        R(0.2, 1.6, 1.25, 2.9, hx('#3a4a5e'))
        R(-0.15, 1.4, 0.15, 3.0, hx('#4a5566'))
        R(-0.5, 3.05, 0.5, 3.4, hx('#2a2e36'))               # destination sign
        R(-1.4, 0.0, 1.4, 0.35, hx('#2a2c30'))               # skirt
        rgb, a = cv.resolve()
        sub = img[max(y0, 0):y1, max(x0, 0):x1]
        rgb = rgb[max(0, -y0):max(0, -y0) + sub.shape[0], max(0, -x0):max(0, -x0) + sub.shape[1]]
        a = a[max(0, -y0):max(0, -y0) + sub.shape[0], max(0, -x0):max(0, -x0) + sub.shape[1]]
        img[max(y0, 0):y1, max(x0, 0):x1] = sub * (1 - a[..., None]) + rgb
        # headlights (glow added later so bloom picks them up)
        return [(xc - 1.0 * k, yb - 0.75 * k), (xc + 1.0 * k, yb - 0.75 * k)], k

    def _particles(self, img, c, t):
        """Sunlit fluff / pollen drifting in the warm air; near ones are soft out-of-focus discs."""
        rng = np.random.default_rng(404)
        n = 34
        X0 = rng.uniform(-6, 7, n)
        Y0 = rng.uniform(0.5, 4.0, n)
        Z0 = rng.uniform(3.0, 16.0, n)
        ph = rng.uniform(0, 6.28, n)
        X = X0 - 0.35 * t + 0.25 * np.sin(t * 0.7 + ph)
        Y = Y0 + 0.12 * np.sin(t * 0.9 + ph * 1.7) + 0.05 * t
        Z = Z0
        ok = Z - c['cz'] > 0.8
        xs, ys = self.proj(X[ok], Y[ok], Z[ok], c)
        k = self.f * c['fz'] / (Z[ok] - c['cz'])
        r = np.clip(0.012 * k, 0.8, 9.0)
        inten = np.clip(0.9 - (r - 2.0) * 0.06, 0.25, 0.9) * (0.6 + 0.4 * np.sin(t * 2.0 + ph[ok]) ** 2)
        C.splat(img, xs, ys, r, np.array([1.0, 0.97, 0.88], np.float32), inten * 0.55)
        return img

    # ------------------------------------------------------------------ frame
    def _affine_layer(self, plate, off, Zref, c):
        """Place a plate rendered for the t=0 camera: scale about the VP by fz*Zref/(Zref-cz) and shift for cx."""
        W, H = self.W, self.H
        s = c['fz'] * (Zref / (Zref - c['cz']) if Zref else 1.0)
        dx = -self.f * c['fz'] * c['cx'] / (Zref - c['cz']) if Zref else 0.0
        # plate px -> screen: x = px + (xp - off - px) * s + dx
        M = np.array([[s, 0, self.px - (off[0] + self.px) * s + dx], [0, s, self.hz - (off[1] + self.hz) * s]],
                     np.float32)
        return cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))

    def _bloom(self, img, threshold=1.0, knee=0.15, strength=0.22, halation=0.1, radii=(0.003, 0.01, 0.03, 0.08)):
        """fx.bloom_soft equivalent with the soft-knee threshold taken on the half-res image and the bloom
        + warm halation accumulated at half res (one upsample, one full-res add)."""
        H, W = img.shape[:2]
        small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        lum = small.max(-1, keepdims=True)
        k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
        br = small * np.maximum(k * k, (lum > threshold + knee))
        acc = np.zeros_like(br)
        wts = (1.0, 0.8, 0.6, 0.45)
        for r, wt in zip(radii, wts):
            acc += F.fast_blur(br, r * W / 2) * (wt * strength / sum(wts))
        acc += F.fast_blur(br, 0.006 * W / 2) * (np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5)
        img += cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)
        return img

    def _paper_texture(self):
        """Subtle watercolour-paper tooth: fine fibre speckle + soft mottling, ~+-1.2 % multiplicative."""
        W, H = self.W, self.H
        rng = np.random.default_rng(77)
        fine = rng.standard_normal((H, W)).astype(np.float32)
        fine = cv2.GaussianBlur(fine, (0, 0), 0.6 * self.sc + 0.2)
        fib = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), sigmaX=2.2 * self.sc,
                               sigmaY=0.5 * self.sc + 0.2)
        mot = C.fbm(W // 4, H // 4, scale=9, octaves=4, seed=78)
        mot = cv2.resize(mot, (W, H), interpolation=cv2.INTER_CUBIC)
        fine /= fine.std() + 1e-6
        fib /= fib.std() + 1e-6
        # the tooth is gentler over the sky (no blotchy mottle inside the painted cloud masses)
        ys = np.arange(H, dtype=np.float32)[:, None]
        sk = 0.45 + 0.55 * P.sstep(self.hz - 0.12 * H, self.hz + 0.02 * H, ys)
        p = 1.0 + (0.005 * fine + 0.004 * fib + 0.01 * (mot - 0.5)) * sk
        return p.astype(np.float32)

    def _mid_layer(self, c, t):
        """Midground plate placed like _affine_layer, plus a subtle wind sway of the tree crowns: a slow
        travelling gust wave (smooth in space and time, no flicker) displaces the crown pixels by ~1 px at
        1080p, the crown tops most."""
        md = self._affine_layer(self.mid, self.mid_off, self.mid_Z, c)
        sw = getattr(self, 'mid_sway', None)
        if sw is None:
            return md
        W, H = self.W, self.H
        if not hasattr(self, '_sway_rows'):
            rows = np.nonzero(sw.max(1) > 0.02)[0]
            self._sway_rows = (int(rows.min()), int(rows.max()) + 1) if len(rows) else (0, 0)
        r0, r1 = self._sway_rows
        if r1 <= r0:
            return md
        Zr = self.mid_Z
        s = c['fz'] * Zr / (Zr - c['cz'])
        dx = -self.f * c['fz'] * c['cx'] / (Zr - c['cz'])
        off = self.mid_off
        # plate rows r0..r1 -> screen rows
        ya = max(int((r0 - off[1] - self.hz) * s + self.hz) - 3, 0)
        yb = min(int((r1 - off[1] - self.hz) * s + self.hz) + 4, H)
        if yb <= ya:
            return md
        xs = np.arange(W, dtype=np.float32)[None, :]
        ys = np.arange(ya, yb, dtype=np.float32)[:, None]
        mx = ((xs - self.px - dx) / s + off[0] + self.px).astype(np.float32)
        my = ((ys - self.hz) / s + off[1] + self.hz).astype(np.float32)
        mx = np.broadcast_to(mx, (yb - ya, W))
        my = np.ascontiguousarray(np.broadcast_to(my, (yb - ya, W)))
        wgt = cv2.remap(sw, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        sc_ = self.sc
        ph = mx / (260.0 * sc_)
        g = (0.65 * np.sin(2 * math.pi * 0.32 * t - ph) + 0.35 * np.sin(2 * math.pi * 0.71 * t - 1.7 * ph + 1.3) +
             0.25 * np.sin(my / (37.0 * sc_) + 2 * math.pi * 0.9 * t))
        amp = 1.3 * sc_ * wgt
        mx2 = (mx + amp * g).astype(np.float32)
        my2 = (my + 0.35 * amp * np.sin(2 * math.pi * 0.45 * t - ph * 1.3 + 0.7)).astype(np.float32)
        band = cv2.remap(self.mid, mx2, my2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))
        md[ya:yb] = band
        return md

    def frame(self, t):
        W, H = self.W, self.H
        c = self.cam(t)
        # --- sky & clouds (camera push: tiny zoom about the frame, clouds drift)
        zoom = c['fz']
        img = S.drift(self.sky, W, H, t, 0.0, cam=(0, 0), zoom=zoom, depth=1.0)
        Ls = self.clouds.render_layers(W, H, t, 0.0, 0.0, zoom)
        for Lr in Ls:
            img = F.over_rgba(img, Lr)
        # soft silver-gold glow spilling off the hero tower's sunlit rim into the blue (quarter res)
        tw_ = Ls[-1][::4, ::4]
        hot_ = np.clip(tw_[..., :3].max(-1) - 1.02, 0, 1) * tw_[..., 3]
        glow_ = cv2.GaussianBlur(hot_, (0, 0), 0.005 * W / 4) * 0.7 + cv2.GaussianBlur(hot_, (0, 0), 0.018 * W / 4) * 0.3
        img += cv2.resize(glow_, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * np.array(
            [1.0, 0.95, 0.82], np.float32)
        # --- mountains + hills, midground
        mt = self._affine_layer(self.mount, self.far_off, 0, c)
        img = F.over_rgba(img, mt)
        md = self._mid_layer(c, t)
        img = F.over_rgba(img, md)
        # --- ground
        y0, g, spec, wa = self._ground_frame(c)
        grgb = self._ground_dynamics(g, c, t, y0)
        grgb = self._water(img, grgb, wa, c, t, y0)
        # ground rows: keep whatever is above the horizon (mountain bases) where the ground is not yet
        yy = np.arange(y0, H, dtype=np.float32)[:, None]
        gm = np.clip(yy - self.hz + 0.5, 0, 1)[..., None]
        img[y0:y0 + 2] = img[y0:y0 + 2] * (1 - gm[:2]) + grgb[:2] * gm[:2]
        img[y0 + 2:] = grgb[2:]
        # hills and the mid plate (houses/groves) stay in front of the far ground they stand on
        img[y0:] = F.over_rgba(img[y0:], mt[y0:])
        img[y0:] = F.over_rgba(img[y0:], md[y0:])
        # --- distant railcar approaching down the line (wobbles in the heat haze)
        heads, tk = self._draw_train(img, c, t)
        # --- rail head sun-streak (continuous polished-steel highlight; shimmers faintly)
        if not hasattr(self, '_vpw'):
            yy_ = np.arange(y0, H, dtype=np.float32)[:, None]
            xx_ = np.arange(W, dtype=np.float32)[None, :]
            self._vpw = (0.7 + 1.6 * np.exp(-(yy_ - self.hz) / (0.07 * H)) *
                         np.exp(-((xx_ - self.px) / (0.25 * W)) ** 2)).astype(np.float32)
        # the polished running band brightens where the sun's reflection falls: a soft hot spot that sits
        # a fixed distance ahead of the camera and so slides along the rails during the push-in
        if not hasattr(self, '_rail_rows'):
            self._rail_rows = np.arange(y0, H, dtype=np.float32)[:, None]
        Zrow = self.f * c['fz'] * self.h / np.maximum(self._rail_rows - self.hz, 0.3) + c['cz']
        dz_ = (Zrow - c['cz'] - 11.0) / 7.0
        hot = 1.0 + 1.3 * np.exp(-dz_ * dz_)
        streak = spec * 0.62 * self._vpw * hot
        img[y0:] += streak[..., None] * np.array([1.0, 0.97, 0.9], np.float32) * 0.85
        # --- far structures (poles + wires beyond ~40 m) then the heat haze over them and the far rails
        cv = self.canvas
        cv.clear()
        ZF = 40.0
        tfar = [i for i, p in enumerate(self.tpoles) if p['Z'] >= ZF]
        pfar = [i for i, p in enumerate(self.poles) if p['Z'] >= ZF]
        ins_t = [None] * len(self.tpoles)
        ins_p = [None] * len(self.poles)
        for i in tfar[::-1]:
            ins_t[i] = self._draw_pole(cv, self.tpoles[i], c)
        for i in pfar[::-1]:
            ins_p[i] = self._draw_pole(cv, self.poles[i], c)
        self._wires(cv, [self.tpoles[i] for i in tfar], [ins_t[i] for i in tfar], c, sag=0.35, col=COL['wire'],
                    width_m=0.012, extend_back=False)
        self._wires(cv, [self.poles[i] for i in pfar], [ins_p[i] for i in pfar], c, sag=0.45, col=COL['wire'],
                    width_m=0.016, extend_back=False)
        rgbf, af = cv.resolve(int(self.hz - 0.45 * H), int(self.hz + 0.12 * H))
        yf0 = int(self.hz - 0.45 * H)
        sub = np.ascontiguousarray(img[yf0:yf0 + rgbf.shape[0]], dtype=np.float32)
        P.over_premult(sub, np.ascontiguousarray(rgbf, dtype=np.float32), np.ascontiguousarray(af, dtype=np.float32))
        img[yf0:yf0 + rgbf.shape[0]] = sub
        img = self._heat_haze(img, t)
        # --- a few swallows high over the fields
        # (rendered at full resolution inside a band around the flock so the swallows keep crisp wings)
        by0, by1 = int(0.2 * H), int(0.5 * H)
        bb = S.birds(W, by1 - by0, t, seed=4, n=4, center=(0.66, (0.36 * H - by0) / (by1 - by0)),
                     velocity=(-0.022, -0.004 * H / (by1 - by0)), size=0.013, spread=0.05, heading=-1.0)
        b = np.zeros((H, W), np.float32)
        b[by0:by1] = bb
        img = np.ascontiguousarray(img, dtype=np.float32)
        P.over_color(img, np.asarray(self.bird_col, np.float32), np.ascontiguousarray(b, dtype=np.float32))
        # --- near structures (per-frame 3D)
        cv.clear()
        tn = [i for i in range(len(self.tpoles)) if i not in tfar]
        pn = [i for i in range(len(self.poles)) if i not in pfar]
        for i in tn[::-1]:
            ins_t[i] = self._draw_pole(cv, self.tpoles[i], c)
        for i in pn[::-1]:
            ins_p[i] = self._draw_pole(cv, self.poles[i], c)
        kt = (tn[-1] + 2) if tn else 1
        kp = (pn[-1] + 2) if pn else 1
        self._wires(cv, self.tpoles[:kt], ins_t[:kt], c, sag=0.35, col=COL['wire'], width_m=0.012)
        self._wires(cv, self.poles[:kp], ins_p[:kp], c, sag=0.45, col=COL['wire'], width_m=0.016)
        ph = (t / 1.2) % 1.0
        e = 0.05
        lamp_a = float(P.sstep(0.0, e, ph) * (1 - P.sstep(0.5, 0.5 + e, ph)))
        lamp_b = 1.0 - lamp_a
        lampsB = self._draw_signal(cv, self.sigB, c, t, (lamp_b, lamp_a))
        armB = self._draw_arm(cv, self.sigB, c, t, (lamp_b, lamp_a))
        vend_c, vend_w = self._draw_vending(cv, c, t)
        armA = self._draw_arm(cv, self.sigA, c, t, (lamp_a, lamp_b))
        lampsA = self._draw_signal(cv, self.sigA, c, t, (lamp_a, lamp_b))
        self._draw_grass(cv, c, t)
        rgb, a = cv.resolve()
        img = np.ascontiguousarray(img, dtype=np.float32)
        # halation bloom in the bright sky behind the crossbucks (the silhouettes are then cut against it)
        q4 = 4
        if not hasattr(self, '_q4grid'):
            yq, xq = np.mgrid[0:H // q4, 0:W // q4].astype(np.float32)
            self._q4grid = (xq * q4 + q4 * 0.5, yq * q4 + q4 * 0.5)
        xq, yq = self._q4grid
        hal = np.zeros((H // q4, W // q4, 3), np.float32)
        for sg_, gain in ((self.sigA, 1.0), (self.sigB, 0.55)):
            hx_, hy_ = self.proj(sg_['X'], 3.1, sg_['Z'], c)
            rr = 1.5 * self.kscale(sg_['Z'], c)
            dd = ((xq - float(hx_)) ** 2 + (yq - float(hy_)) ** 2) / (rr * rr)
            hal += (np.exp(-dd * 1.2) * 0.045)[..., None] * np.array(
                [1.0, 0.96, 0.86], np.float32) * gain
        img += cv2.resize(hal, (W, H), interpolation=cv2.INTER_LINEAR)
        bg = img.copy()
        P.over_premult(img, np.ascontiguousarray(rgb, dtype=np.float32), np.ascontiguousarray(a, dtype=np.float32))
        # light wrap: the bright sky bleeds over the edges of the structures (stronger toward the sun)
        wrap = FX.light_wrap(img, bg, a, 0, 0, W, H, sigma=0.006 * W)
        if not hasattr(self, '_wrapw'):
            sx_, sy_ = self.sun_screen
            yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
            self._wrapw = (0.5 + 0.25 * np.exp(-(((xx_ - sx_) / W) ** 2 + ((yy_ - sy_) / W) ** 2) / 0.18)).astype(np.float32)
        a4 = cv2.resize(a, (W // q4, H // q4), interpolation=cv2.INTER_AREA)
        edge = np.clip((1 - cv2.resize(cv2.GaussianBlur(a4, (0, 0), 0.004 * W / q4), (W, H))) * 2.5, 0, 1)
        img += np.minimum(wrap, 0.5) * (a * edge * self._wrapw)[..., None] * 0.28
        # crisp sun-side rim light on poles, signals, arms, grass (neighbour toward the sun is background)
        rr_ = 2.2 * self.sc
        Mr = np.float32([[1, 0, -rr_ * 0.85], [0, 1, rr_ * 0.55]])
        nb_ = cv2.warpAffine(a, Mr, (W, H), borderValue=0)
        rim_ = np.clip(a - nb_, 0, 1) * a
        img += (rim_ * 0.5 * self._wrapw)[..., None] * np.array([1.0, 0.93, 0.78], np.float32)
        # --- emissive lamps: tight glow + wide soft halo, red spill onto the backboard / pole / crossbuck,
        # and a faint warm spill on the surroundings (wide terms accumulated at quarter resolution)
        add = img
        wide = np.zeros((H // q4, W // q4, 3), np.float32)
        for lamps, st, gain in ((lampsA, (lamp_a, lamp_b), 1.0), (lampsB, (lamp_b, lamp_a), 0.75)):
            for (lx, ly, Rl), on in zip(lamps, st):
                if on < 0.02:
                    continue
                Rl = max(Rl, 1.0)
                R = int(Rl * 14) + 4
                x0, x1 = max(int(lx - R), 0), min(int(lx + R), W)
                y0_, y1 = max(int(ly - R), 0), min(int(ly + R), H)
                if x1 > x0 and y1 > y0_:
                    yy_, xx_ = np.mgrid[y0_:y1, x0:x1].astype(np.float32)
                    d = np.sqrt((xx_ - lx) ** 2 + (yy_ - ly) ** 2) / Rl
                    gl = 1.0 / (1 + (d / 1.2) ** 3) * 0.95 + np.exp(-d / 4.5) * 0.4
                    core = np.exp(-(d / 0.75) ** 4) * 0.9
                    spill = np.exp(-d / 6.0) * a[y0_:y1, x0:x1] * 0.75
                    base = add[y0_:y1, x0:x1]
                    add[y0_:y1, x0:x1] = (base + base * spill[..., None] * np.array([1.7, -0.2, -0.35]) * on * gain +
                                          (gl[..., None] * np.array([1.0, 0.26, 0.14]) +
                                           core[..., None] * np.array([1.0, 0.78, 0.62])) * on * gain)
                dq = np.sqrt((xq - lx) ** 2 + (yq - ly) ** 2) / Rl
                wide += (np.exp(-dq / 8.0) * 0.09 + np.exp(-dq / 24.0) * 0.012)[..., None] * np.array(
                    [1.0, 0.3, 0.16], np.float32) * on * gain
        img += cv2.resize(wide, (W, H), interpolation=cv2.INTER_LINEAR)
        # arm reflector lamps: small glows
        for arm, st in ((armA, (lamp_a, lamp_b)), (armB, (lamp_b, lamp_a))):
            for j, ((ax_, ay_), kk) in enumerate(arm):
                on = st[j % 2]
                if on > 0.05:
                    C.splat(img, np.array([ax_]), np.array([ay_]), max(0.07 * kk, 1.5),
                            np.array([1.0, 0.3, 0.15], np.float32), 0.8 * on)
        # water glints on the flooded paddies (world-fixed points, coherent twinkle)
        if len(self.water_pts):
            wp = self.water_pts
            okw = wp[:, 1] - c['cz'] > 5
            wx_, wy_ = self.proj(wp[okw, 0], 0.0, wp[okw, 1], c)
            dist = np.clip(wp[okw, 1] / 400.0, 0, 1)
            glints_inplace(img, wx_.astype(np.float32), wy_.astype(np.float32), size=0.005 * (1.3 - dist),
                           intensity=0.5, t=t * 1.0, seed=9, color=(1.0, 0.98, 0.92))
        # sun glint on the vending machine's glass front (fixed specular point, steady)
        gvx, gvy = float(vend_c[0]) + 0.3 * vend_w, float(vend_c[1]) - 0.22 * vend_w
        if 0 <= gvx < W and 0 <= gvy < H:
            glints_inplace(img, np.array([gvx], np.float32), np.array([gvy], np.float32),
                           size=np.array([0.012], np.float32), intensity=np.array([0.75], np.float32),
                           color=(1.0, 0.98, 0.92))
        # train headlights
        for (hx_, hy_) in heads:
            rr = max(tk * 0.16, 1.2)
            C.splat(img, np.array([hx_]), np.array([hy_]), rr * 2.2, np.array([1.0, 0.95, 0.8], np.float32), 1.6)
            C.splat(img, np.array([hx_]), np.array([hy_]), rr * 7, np.array([1.0, 0.85, 0.6], np.float32), 0.25)
        img = self._particles(img, c, t)
        # --- rail glints: star-like specular points riding along both rail heads. The specular spot sits
        # at a fixed-ish distance ahead of the moving camera, so it slides along the rails during the push-in;
        # each fades in/out on its own slow cycle (coherent, no flicker); occluded by near structures.
        gxs, gys, gsz, gin = [], [], [], []
        for ri, xr in enumerate((self.Xt - 0.5335, self.Xt + 0.5335)):
            for j, d0 in enumerate((6.5, 9.5, 16.0, 24.0, 38.0, 60.0, 95.0)):
                ph_ = (t * 0.23 + 0.37 * j + 0.5 * ri) % 1.0
                env_ = math.sin(math.pi * ph_) ** 2
                dd_ = d0 * (1.0 + 0.35 * ph_)
                Zg = c['cz'] + dd_
                xg, yg = self.proj(xr, 0.16, Zg, c)
                xg, yg = float(xg), float(yg)
                if not (0 <= xg < W and 0 <= yg < H):
                    continue
                occ_ = float(a[int(yg), int(xg)])
                k_ = self.kscale(Zg, c)
                gxs.append(xg)
                gys.append(yg)
                gsz.append(float(np.clip(0.2 * k_, 7.0, 34.0)) * self.sc / W)
                gin.append(env_ * (1 - occ_) * (1.5 if d0 < 30 else 1.0))
        if gxs:
            glints_inplace(img, np.array(gxs, np.float32), np.array(gys, np.float32), size=np.array(gsz, np.float32),
                           intensity=np.array(gin, np.float32), color=(1.0, 0.97, 0.9))
        if os.environ.get('S02_NOPOST'):
            return F.finish_fast(img, t, sat=1.08, grain_amt=0.0, vig=0.14, ca=0.0)
        # --- sun glare from the top-right corner + flare
        lx, ly = self.sun_screen
        if not hasattr(self, '_sunglow'):
            self._sunglow = S.sun(W, H, lx, ly, radius=0.016, color=(1.0, 0.93, 0.8), intensity=0.6, disc=False,
                                  glow_size=0.7)
            self._ghosts = FX.anamorphic_flare(W, H, lx, ly) + flare_ghosts(W, H, lx, ly) * 1.3
            self._flare = F.anime_flare(W, H, lx, ly, intensity=1.0, tint=(1.0, 0.92, 0.8), rot=0.0,
                                        center=(W / 2, H / 2), ghosts=1.0, starburst=0.9, halo=0.8, streak=0.0)
        img += self._sunglow
        # the glare veils the open sky / far field; the solid midground (houses, groves) keeps its values
        gk = (0.85 * (1.0 - 0.5 * md[..., 3]))[..., None]
        img = img + (1.0 - np.minimum(img, 1.0)) * self._glare * gk
        occ = np.maximum.reduce([Lr[::4, ::4, 3] for Lr in Ls[-self.n_occ:]] + [a[::4, ::4]])
        occ = cv2.resize(np.ascontiguousarray(occ), (W, H), interpolation=cv2.INTER_LINEAR)
        img += F.light_shafts(W, H, lx, ly, occluder=occ, strength=0.1, length=0.95,
                                   radius=0.14, tint=(1.0, 0.95, 0.85), t=t)
        # diffusion: soft veil of the blurred frame (Shinkai-style photographic glow)
        sm = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        gl = cv2.resize(F.fast_blur(sm, 0.01 * W / 4), (W, H), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img, dtype=np.float32)
        P.diffuse(img, np.ascontiguousarray(gl, dtype=np.float32), 1.0, 0.0, 0.92, 0.09)
        img = self._bloom(img, threshold=1.02, knee=0.15, strength=0.22, halation=0.1)
        img = img + self._flare * 0.85
        img += self._ghosts
        img = F.shoulder(img, 0.93, desat=0.03)
        # static paint / paper texture (fixed in screen space: no per-frame grain, no flicker)
        if not hasattr(self, '_paper'):
            self._paper = self._paper_texture()
        img *= self._paper[..., None]
        return F.finish_fast(img, t, sat=1.08, grain_amt=0.0, vig=0.14, ca=0.0006)
