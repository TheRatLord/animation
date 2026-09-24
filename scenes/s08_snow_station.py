"""s08_snow_station - a small rural train station at night in heavy snowfall (5 cm per Second homage).

One-point perspective down the platform: a line of warm lamps recedes past a snow-laden canopy with a
lit waiting room, the track runs beside it into the snowy haze, utility poles and wires lead the eye to
the vanishing point, snow-laden pines and distant houses with warm windows sit in the deep blue night.
Snow falls in true 3D (sized/defocused by depth, lit by the lamps it passes, occluded by the scene),
lamps scatter volumetric light cones into the snowy air. Camera: slow truck right + gentle push-in with
three parallax planes (far / station / foreground book)."""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import core as C          # noqa: E402
from lib import fx as F            # noqa: E402
import s08_snow_station_lib as L   # noqa: E402
import s08_snow_station_pine as PN  # noqa: E402
import s08_snow_station_sky as SK   # noqa: E402
import s08_snow_station_spruce as SP  # noqa: E402
import s08_snow_station_fir as FR  # noqa: E402
import s08_snow_station_paint as PT  # noqa: E402
import s08_snow_station_bough4 as BG  # noqa: E402
import s08_snow_station_vol as VL  # noqa: E402
import s08_snow_station_arch as AR  # noqa: E402
import s08_snow_station_tex as TX  # noqa: E402
import s08_snow_station_clouds as CL  # noqa: E402
import s08_snow_station_bokeh as BK  # noqa: E402
import s08_snow_station_tex2 as T2  # noqa: E402
import s08_snow_station_r3 as R3  # noqa: E402
import s08_snow_station_r4 as R4  # noqa: E402
import s08_snow_station_r5 as R5  # noqa: E402

DURATION = 5.0
ZSPLIT = 40.0
ZREF_MID, ZREF_FAR = 22.0, 70.0

# ----------------------------------------------------------------------------- world (metres)
# camera at the origin, 1.5 m above the platform, looking down +z (the track direction)
Y_PLAT = -1.5
CAM_X = 0.9
Y_BED = -2.6
PLAT_X0, PLAT_X1, PLAT_Z1 = -6.4, 2.2, 64.0
RAILS = (3.55, 4.62)
CAN_X0, CAN_X1, CAN_Z0, CAN_Z1, CAN_Y = -5.9, 2.75, 16.0, 30.0, 1.6
BLD_X1, BLD_Z0, BLD_Z1 = -3.3, 18.0, 29.0

WARM = np.array([1.0, 0.6, 0.28], np.float32)        # sodium-ish post lamps
WHITE_W = np.array([1.0, 0.76, 0.46], np.float32)     # canopy lamps
AIR_AMBER = np.array([1.0, 0.74, 0.36], np.float32)   # lamp light scattered in the snowy air (pure amber)
WIN = np.array([1.0, 0.72, 0.38], np.float32)        # windows
VEND = np.array([1.0, 0.5, 0.36], np.float32)       # red machine body + warm-white panel

SKY_TOP = C.hex2rgb('#050b22')
SKY_MID = C.hex2rgb('#11245a')
SKY_LOW = C.hex2rgb('#284a8c')
SKY_HZ = C.hex2rgb('#4b6fae')
FOG = C.hex2rgb('#35558f')
AMB_SNOW = np.array([0.27, 0.34, 0.64], np.float32)  # snow lit by the night sky


_RAMP_E = np.array([0.0, 0.12, 0.35, 0.8, 1.6, 3.0], np.float32)
_RAMP_C = np.array([[0.3, 0.3, 0.6], [0.5, 0.42, 0.62], [0.92, 0.6, 0.44], [1.2, 0.86, 0.56],
                    [1.5, 1.2, 0.85], [1.9, 1.75, 1.5]], np.float32)


def snow_lit(amb, E, bump=1.0):
    """Painterly snow colour under night ambient + warm lamp irradiance E (..., 3): the pool passes
    blue -> violet -> salmon -> gold -> cream instead of mixing to grey."""
    e = E.mean(-1) * 1.0
    tgt = np.stack([np.interp(e, _RAMP_E, _RAMP_C[:, c]) for c in range(3)], -1).astype(np.float32)
    w = (1 - np.exp(-e * 3.0))[..., None]
    return (amb * (1 - w) + tgt * w) * bump


def paint_E(E, step=2.3, soft=0.1, mix=0.7):
    """Hand-painted light: the irradiance magnitude is laid in as a few flat value plateaus (a warm
    pool with a readable edge, then a cool shade value) instead of a smooth CG falloff. Hue is kept."""
    E = np.asarray(E, np.float32)
    e = E.mean(-1, keepdims=True)
    l = np.log(e + 0.02) / math.log(step)
    fl = np.floor(l)
    fr = l - fl
    q = fl + np.clip((fr - (1 - soft)) / soft, 0, 1) ** 2 * (3 - 2 * np.clip((fr - (1 - soft)) / soft, 0, 1)) + 0.5
    eb = np.exp(q * math.log(step)) - 0.02
    eb = np.maximum(eb, 0) * mix + e * (1 - mix)
    return (E * (eb / np.maximum(e, 1e-5))).astype(np.float32)


def cone_params(lamp):
    """Shaped spotlight cone of a lamp for the air / the flakes: (cos inner, cos outer, omni, power, ext)."""
    (p, col, I, cone, cpow, sc) = lamp
    if cone >= 0.94:                                   # shaded post lamps: a clean amber beam
        return (math.cos(math.radians(9)), math.cos(math.radians(32)), 0.01, 1.4, 2.0)
    if cone > 0:                                       # canopy tubes: broad
        return (math.cos(math.radians(35)), math.cos(math.radians(75)), 0.12, 1.0, 3.0)
    return (1.0, 1.0, 1.0, 1.0, 2.0)


def lamps():
    """(pos, colour, intensity, cone, cone_pow, has_scatter)"""
    Ls = []
    for z, sc, I in ((6.5, 1.0, 20.0), (36.0, 0.28, 11.0), (47.0, 0.3, 10.0), (58.0, 0.32, 10.0)):
        Ls.append(((1.45, 2.1, z), WARM, I, 0.95, 1.4, sc))
    for z in (19.0, 23.5, 28.0):
        Ls.append(((0.2, 1.38, z), WHITE_W, 5.5, 0.9, 1.2, 0.22))
    Ls.append(((-2.3, 1.38, 23.5), WHITE_W, 4.0, 0.9, 1.2, 0.22))
    Ls.append(((-3.0, 0.2, 21.5), WIN, 1.6, 0.0, 1.0, False))      # waiting-room windows
    Ls.append(((-2.3, -0.6, 24.0), VEND, 1.4, 0.0, 1.0, False))    # vending machine
    Ls.append(((6.2, 4.6, 70.0), WARM, 5.0, 0.85, 1.4, 0.5))      # street lamp on a pole
    Ls.append(((-5.15, 2.1, 13.8), WARM, 14.0, 0.95, 1.4, 1.1))    # second platform lamp (left)
    return Ls


class Scene:
    FOG_C = FOG

    def __init__(self, W, H):
        self.W, self.H = W, H
        s = W / 1920.0
        self.s = s
        mx, my = int(0.06 * W), int(0.06 * H)
        self.mx, self.my = mx, my
        PW, PH = W + 2 * mx, H + 2 * my
        self.PW, self.PH = PW, PH
        f = 0.78 * W
        self.cam = L.Cam(f, 0.54 * W + mx, 0.57 * H + my, ox=CAM_X)
        self.lamps = lamps()
        self.rng = np.random.default_rng(8)

        us, vs = C.grid(PW, PH)
        self.rx = (us - self.cam.cx) / f
        self.ry = -(vs - self.cam.cy) / f

        self.far = L.Canvas(PW, PH, self._sky())
        self.gnd = L.Canvas(PW, PH)          # ground planes: warped exactly (plane homography) per frame
        self.mid = L.Canvas(PW, PH)          # station + near objects (z < ZSPLIT)
        self.midf = L.Canvas(PW, PH)         # village, far trees, far poles (z >= ZSPLIT)
        # foreground sprites, each on its own parallax plane at its true depth
        self.fgs = {k: L.Canvas(PW, PH) for k in ('tree', 'bench', 'sign', 'lamp')}
        self._far_scene()
        self._ground()
        for cvx in (self.mid, self.midf):
            cvx.z = self.gnd.z.copy()
            cvx.ztest = True
        self._mid_objects()
        self._station()
        self._foreground()
        self._wire_prep()
        zb = np.minimum(self.far.z, np.where(self.gnd.a > 0.5, self.gnd.z, 1e4))
        for cvx in [self.mid, self.midf]:
            zb = np.minimum(zb, np.where(cvx.a > 0.5, cvx.z, 1e4))
        self.zbuf_bg = zb.copy()          # scene depth without the foreground sprites (veil mask)
        for cvx in list(self.fgs.values()):
            zb = np.minimum(zb, np.where(cvx.a > 0.5, cvx.z, 1e4))
        self.zbuf = zb
        self._volumetrics()
        self.L_far = self.far.rgba()
        self.L_gnd = self.gnd.rgba()
        self.gnd_id = self.plat_id
        self.L_mid = self.mid.rgba()
        self._aerial_fade(self.midf)
        self._aerial_fade(self.mid, near=True)
        self.L_midf = self.midf.rgba()
        self.sprites = []
        # parallax depths of the sprite planes (the framing pine is pulled nearer than it is drawn, so
        # it reads as the true foreground plane: ~4x the canopy's screen speed, ~12x the village's)
        for k, zr in (('tree', 4.8), ('bench', 10.3), ('sign', 8.2), ('lamp', 6.5)):
            self.sprites.append((k, zr) + self._crop(self.fgs[k]))
        del self.rx, self.ry
        self._flakes()
        self.bk = BK.Bokeh(W, H)
        self._glows()
        self._halos()
        self._veil()
        self._streaks()
        self._sparkles()

    # ------------------------------------------------------------------ helpers
    def _aerial_fade(self, cv, near=False):
        """Aerial perspective into the snowfall: the village and treeline beyond the platform sink into a
        luminous blue snow haze (lower contrast than the midground), lit windows keep a little punch."""
        z = np.where(cv.a > 0.02, cv.z, 1e4)
        z = cv2.erode(np.minimum(z, 400.0).astype(np.float32), np.ones((3, 3), np.uint8))
        if near:
            k = 0.16 * C.smoothstep(26.0, 42.0, z)
        else:
            k = 0.46 * C.smoothstep(34.0, 54.0, z) + 0.2 * C.smoothstep(58.0, 110.0, z)
        lum = cv.rgb.max(-1)
        k = k * (1 - 0.45 * C.smoothstep(0.55, 1.4, lum))
        haze = FOG * 1.18 + np.array([0.01, 0.015, 0.03], np.float32)
        cv.rgb = (cv.rgb * (1 - k[..., None]) + haze * k[..., None]).astype(np.float32)
        if not near:
            # round 5: the village sits inside the snowfall - its blacks lift toward the fog and its
            # silhouettes (roof edges, trim) go slightly soft, so the near pine / lamp pole separate
            dark = C.smoothstep(0.35, 0.05, lum) * C.smoothstep(38.0, 50.0, z)
            cv.rgb = (cv.rgb + (haze - cv.rgb) * (0.15 * dark)[..., None]).astype(np.float32)
            sb = 1.3 * self.s
            kb = C.smoothstep(40.0, 52.0, cv2.dilate(z, np.ones((5, 5), np.uint8)))[..., None]
            rgb_b = cv2.GaussianBlur(cv.rgb, (0, 0), sb)
            a_b = cv2.GaussianBlur(cv.a, (0, 0), sb)
            cv.rgb = (cv.rgb * (1 - kb) + rgb_b * kb).astype(np.float32)
            cv.a = (cv.a * (1 - kb[..., 0]) + a_b * kb[..., 0]).astype(np.float32)

    def _crop(self, cv):
        a = cv.a
        ys, xs = np.nonzero(a > 0.002)
        if len(ys) == 0:
            return (0, 0, np.zeros((1, 1, 4), np.float32))
        y0, y1 = max(ys.min() - 2, 0), min(ys.max() + 3, cv.H)
        x0, x1 = max(xs.min() - 2, 0), min(xs.max() + 3, cv.W)
        return (x0, y0, np.dstack([cv.rgb[y0:y1, x0:x1], a[y0:y1, x0:x1]]).astype(np.float32))

    def _sprite_over(self, img, x0, y0, rgba, M):
        """Composite a cropped plate sprite through the layer affine M (scale + translate)."""
        sc, ox, oy = float(M[0, 0]), float(M[0, 2]), float(M[1, 2])
        h, w = rgba.shape[:2]
        X0 = int(math.floor(sc * x0 + ox)) - 1
        Y0 = int(math.floor(sc * y0 + oy)) - 1
        X1 = int(math.ceil(sc * (x0 + w) + ox)) + 1
        Y1 = int(math.ceil(sc * (y0 + h) + oy)) + 1
        X0c, Y0c, X1c, Y1c = max(X0, 0), max(Y0, 0), min(X1, self.W), min(Y1, self.H)
        if X1c <= X0c or Y1c <= Y0c:
            return img
        A = np.array([[sc, 0, sc * x0 + ox - X0c], [0, sc, sc * y0 + oy - Y0c]], np.float32)
        spr = cv2.warpAffine(rgba, A, (X1c - X0c, Y1c - Y0c), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a = spr[..., 3:4]
        img[Y0c:Y1c, X0c:X1c] = img[Y0c:Y1c, X0c:X1c] * (1 - a) + spr[..., :3]
        return img

    def fog_amt(self, z, k=1.0):
        z = np.clip(np.nan_to_num(np.asarray(z, np.float32), nan=1e4, posinf=1e4, neginf=0.0), 0.0, 1e4)
        return 1.0 - np.exp(-z / (68.0 / k))

    def fog_col(self, v):
        """Fog colour = the sky colour just above the horizon at screen row v (plate px)."""
        return FOG

    def light_at(self, P, N=None, spec=False, vis0=None):
        """Irradiance (rgb) at world points P (..., 3) with optional normals N (..., 3).
        vis0: optional visibility (shadow) factor for lamp 0 (the foreground lamp)."""
        P = np.asarray(P, np.float32)
        out = np.zeros(P.shape[:-1] + (3,), np.float32)
        for li, (lp, col, I, cone, cpow, _) in enumerate(self.lamps):
            d = np.asarray(lp, np.float32) - P
            d2 = (d ** 2).sum(-1)
            dist = np.sqrt(d2) + 1e-4
            e = I / (d2 + 0.35)
            if N is not None:
                e = e * np.clip((d * N).sum(-1) / dist, 0, 1)
            if cone > 0:
                cd = np.clip(d[..., 1] / dist, -1, 1)    # light travels down: lamp above point
                e = e * ((1 - cone) + cone * np.clip((cd - 0.3) / 0.55, 0, 1) ** cpow)
            if li == 0 and vis0 is not None:
                e = e * vis0
            out += e[..., None] * col
        return out

    def door_light(self, X, Z):
        """Warm light from the lit waiting-room doorway spilling onto the platform in front of it (the
        interior light shines through the door opening; soft edges, mullion shadows)."""
        lx, ly, lz = -4.4, 0.2, 21.0
        zd = BLD_Z0
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (zd - lz) / (Z - lz)
        ok = (Z < zd) & (t > 0) & (t < 1)
        xi = lx + t * (X - lx)
        yi = ly + t * (Y_PLAT - ly)
        pen = 0.03 + 0.04 * np.clip(zd - Z, 0, 10) / 10
        ap = C.smoothstep(-5.0 - pen, -5.0 + pen, xi) * C.smoothstep(-3.8 + pen, -3.8 - pen, xi) *             C.smoothstep(Y_PLAT + 0.15 - pen, Y_PLAT + 0.15 + pen, yi) * C.smoothstep(Y_PLAT + 1.95 + pen, Y_PLAT + 1.95 - pen, yi)
        ap = ap * (1 - 0.8 * C.smoothstep(0.05, 0.02, np.abs(xi + 4.4))) * (1 - 0.7 * C.smoothstep(0.05, 0.02, np.abs(yi - (Y_PLAT + 0.95))))
        d2 = (X - lx) ** 2 + (Y_PLAT - ly) ** 2 + (Z - lz) ** 2
        cos = (ly - Y_PLAT) / np.sqrt(d2)
        e = np.where(ok, ap * 9.0 * cos / d2 * 10.0, 0.0)
        return (e[..., None] * WIN).astype(np.float32)

    def window_light(self, X, Z):
        """Warm window-shaped patches cast on the platform by the waiting-room pendants through the side
        windows (sash bars and tied-back curtains print as shadows)."""
        out = np.zeros(X.shape, np.float32)
        ok0 = X > BLD_X1 + 0.02
        for (za, zb) in ((19.2, 21.6), (22.4, 24.8), (26.0, 28.2)):
            lx, ly, lz = -4.55, Y_PLAT + 2.2, (za + zb) / 2
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (BLD_X1 - lx) / (X - lx)
            yi = ly + t * (Y_PLAT - ly)
            zi = lz + t * (Z - lz)
            pen = 0.02 + 0.05 * np.clip(X - BLD_X1, 0, 2)
            cz = (zi - za) / (zb - za)
            ap = C.smoothstep(Y_PLAT + 1.0 - pen, Y_PLAT + 1.0 + pen, yi) * C.smoothstep(Y_PLAT + 2.3 + pen, Y_PLAT + 2.3 - pen, yi)
            ap = ap * C.smoothstep(0.16 - pen, 0.16 + pen, cz) * C.smoothstep(0.84 + pen, 0.84 - pen, cz)
            mul = C.smoothstep(0.02, 0.04 + pen, np.abs(yi - (Y_PLAT + 1.65)))
            for q in (1, 2):
                mul = mul * C.smoothstep(0.02, 0.045 + pen, np.abs(zi - (za + (zb - za) * q / 3)))
            d2 = (X - lx) ** 2 + (Y_PLAT - ly) ** 2 + (Z - lz) ** 2
            cos = (ly - Y_PLAT) / np.sqrt(d2)
            out += np.where(ok0, ap * (0.3 + 0.7 * mul) * 6.5 * cos / d2, 0.0)
        return (out[..., None] * WIN).astype(np.float32)

    SIGN = dict(z=8.2, xc=-2.4, w=1.7, yb=Y_PLAT + 1.0, yt=Y_PLAT + 1.62)

    BENCH = dict(x0=-1.8, x1=-1.28, z0=9.3, z1=11.3, ys=Y_PLAT + 0.44)

    def sign_shadow(self, X, Z):
        """Visibility of the foreground lamp from floor points (X, Z)."""
        return (1 - 0.93 * self._occ(X, Z, 0)).astype(np.float32)

    def _occ(self, X, Z, li):
        """Occlusion (0..1) of lamp li seen from floor points (X, Z) by the things standing on the platform:
        station sign board + legs, the bench (seat + backrest), the two bins and the foreground lamp pole.
        Crisp near the contact, penumbra widening with distance from the occluder."""
        lx, ly, lz = self.lamps[li][0]
        X = np.asarray(X, np.float32)
        Z = np.asarray(Z, np.float32)
        occ = np.zeros_like(X)

        def zrect(zr, xa, xb, ya, yb):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (zr - Z) / (lz - Z)
            ok = (t > 0) & (t < 0.999)
            xi = X + t * (lx - X)
            yi = Y_PLAT + t * (ly - Y_PLAT)
            pen = 0.008 + 0.035 * np.clip(np.abs(Z - zr), 0, 8) / 8
            m = C.smoothstep(xa - pen, xa + pen, xi) * C.smoothstep(xb + pen, xb - pen, xi) * \
                C.smoothstep(ya - pen, ya + pen, yi) * C.smoothstep(yb + pen, yb - pen, yi)
            return np.where(ok, m, 0.0)

        def yrect(yr, xa, xb, za, zb):
            t = (yr - Y_PLAT) / (ly - Y_PLAT)
            xi = X + t * (lx - X)
            zi = Z + t * (lz - Z)
            pen = 0.01 + 0.03 * t
            return C.smoothstep(xa - pen, xa + pen, xi) * C.smoothstep(xb + pen, xb - pen, xi) * \
                C.smoothstep(za - pen, za + pen, zi) * C.smoothstep(zb + pen, zb - pen, zi)

        def xrect(xr, ya, yb, za, zb):
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (xr - X) / (lx - X)
            ok = (t > 0) & (t < 0.999)
            yi = Y_PLAT + t * (ly - Y_PLAT)
            zi = Z + t * (lz - Z)
            pen = 0.01 + 0.03 * np.clip(np.abs(xr - X), 0, 4) / 4
            m = C.smoothstep(ya - pen, ya + pen, yi) * C.smoothstep(yb + pen, yb - pen, yi) * \
                C.smoothstep(za - pen, za + pen, zi) * C.smoothstep(zb + pen, zb - pen, zi)
            return np.where(ok, m, 0.0)
        S = self.SIGN
        occ = np.maximum(occ, zrect(S['z'], S['xc'] - S['w'] / 2 - 0.06, S['xc'] + S['w'] / 2 + 0.06, S['yb'] - 0.04, S['yt'] + 0.2))
        for xp in (S['xc'] - S['w'] / 2 + 0.12, S['xc'] + S['w'] / 2 - 0.12):
            occ = np.maximum(occ, zrect(S['z'], xp - 0.04, xp + 0.04, Y_PLAT - 0.1, S['yb']))
        B = self.BENCH
        occ = np.maximum(occ, yrect(B['ys'] + 0.08, B['x0'], B['x1'] + 0.04, B['z0'], B['z1']))
        occ = np.maximum(occ, 0.85 * xrect(B['x0'] + 0.03, B['ys'] + 0.12, B['ys'] + 0.52, B['z0'], B['z1']))
        for zl in (B['z0'] + 0.12, B['z1'] - 0.12):
            occ = np.maximum(occ, zrect(zl, B['x0'], B['x1'], Y_PLAT - 0.1, B['ys']) * 0.8)
        for xa in (1.05, 1.55):
            occ = np.maximum(occ, zrect(15.3, xa, xa + 0.42, Y_PLAT - 0.1, Y_PLAT + 0.95))
        occ = np.maximum(occ, zrect(6.5, 1.98, 2.12, Y_PLAT - 0.1, 2.45))          # foreground lamp pole
        return np.clip(occ, 0, 1).astype(np.float32)

    def _lamp_E(self, li, P, N):
        lp, col, I, cone, cpow, _ = self.lamps[li]
        d = np.asarray(lp, np.float32) - P
        d2 = (d ** 2).sum(-1)
        dist = np.sqrt(d2) + 1e-4
        e = I / (d2 + 0.35) * np.clip((d * N).sum(-1) / dist, 0, 1)
        if cone > 0:
            cd = np.clip(d[..., 1] / dist, -1, 1)
            e = e * ((1 - cone) + cone * np.clip((cd - 0.3) / 0.55, 0, 1) ** cpow)
        return e

    def P(self, X, Y, Z):
        return self.cam.p(X, Y, Z)

    def quad(self, cv, P4, col, z=None, ss=3):
        pts = self.cam.pts(P4)
        x0, y0, m = L.poly_local(pts, ss=ss)
        zz = float(np.mean(np.asarray(P4)[:, 2])) if z is None else z
        cv.paint(x0, y0, m, col, zz)

    # ------------------------------------------------------------------ sky & far
    def _sky(self):
        img = SK.snow_night(self.PW, self.PH, self.cam.f, self.cam.cx, self.cam.cy, seed=5, glow_x=0.22)
        add, self.cloud_d = SK.snow_clouds(self.PW, self.PH, self.cam.f, self.cam.cx, self.cam.cy, seed=5, glow_x=0.22,
                                           strength=0.6)
        # cool, low snow-cloud ceiling: the cloud masses stay slate-indigo (no warm/magenta town glow in
        # the sky); only a thin amber light-pollution band remains right on the horizon (snow_night)
        lum = add.mean(-1, keepdims=True)
        add = lum * np.array([0.78, 0.9, 1.32], np.float32) * 0.55
        img = img + add
        # painted snow-cloud value masses (lib/clouds2), lit faintly from the town below; a lighter hazy
        # band behind the far tree line and a faint warm town glow behind the houses on the right
        # round 4: a few large, soft overcast snow-cloud masses with smooth value gradients (no stamped
        # scallop edges), warm sodium glow on the undersides above the town (right) and the station
        img = img - add * 0.7
        PW = self.PW
        img = R4.soft_night_clouds(img, self.cam.cy, self.s,
                                   glow_centres=((0.78 * PW, 1.0, 0.22 * PW), (0.45 * PW, 0.7, 0.2 * PW), (0.2 * PW, 0.45, 0.12 * PW)))
        img = img + CL.night_glow(self.PW, self.PH, self.cam.cy, self.cam.f, self.s)
        # round 5: hand-painted glowing overcast (vertical gradient, flat band masses, town scatter)
        return R5.painted_overcast(img, self.cam.cy, self.s, amount=0.78)

    def _far_scene(self):
        cam, cv, s = self.cam, self.far, self.s
        PW, PH = self.PW, self.PH
        rng = np.random.default_rng(3)
        # distant mountains (z ~ 900) - barely darker than the sky
        xs = np.linspace(-0.05, 1.05, 400) * PW
        u = xs / PW
        ridge = (0.06 + 0.05 * np.sin(u * 5.3 + 1.0) + 0.03 * np.sin(u * 13 + 2) + 0.012 * np.sin(u * 37))
        ridge = ridge * (0.7 + 0.6 * C.smoothstep(0.3, 0.0, u)) + 0.03
        ys = cam.cy - ridge * 0.35 * PH
        poly = np.concatenate([np.stack([xs, ys], 1), [[xs[-1], cam.cy + 5], [xs[0], cam.cy + 5]]])
        x0, y0, m = L.poly_local(poly, ss=2)
        col = C.lerp(SKY_LOW, SKY_HZ, 0.35) * 0.82
        cv.paint(x0, y0, m, col, 900.0)
        # snowy forest band (z 160-320): rows of little firs
        for row, (zr, dens) in enumerate([(420, 1.3), (340, 1.1), (280, 1.0), (220, 1.0), (170, 0.8), (140, 0.5)]):
            fa = self.fog_amt(zr)
            X = -zr * 1.2
            while X < zr * 1.2:
                X += rng.uniform(2.5, 6.0) / dens
                if rng.random() < 0.18:
                    X += rng.uniform(8, 30)
                    continue
                if abs(X - 3.6) < 12 and zr < 200:
                    pass
                hgt = rng.uniform(7, 13) if rng.random() < 0.55 else rng.uniform(13, 24)
                Z = zr + rng.uniform(-15, 15)
                bu, bv = cam.p(X, Y_BED, Z)
                hpx = cam.f * hgt / Z
                if bu < -20 or bu > PW + 20:
                    continue
                FR.fir(cv, bu, bv, hpx, Z, int(rng.integers(1e9)), fol=(0.05, 0.09, 0.16),
                       snow_top=(0.55, 0.66, 0.9), snow_shd=(0.2, 0.28, 0.48), fog=FOG * 1.02,
                       fog_amt=float(np.clip(fa + 0.02, 0, 0.9)), width=rng.uniform(0.2, 0.33), ss=2,
                       detail=0.5 if hpx > 30 else 0, rim_amt=0.5)
            # far houses tucked into the forest line with lit windows
        for (X, Z, w, d, h) in [(-70, 150, 9, 7, 5.5), (-38, 128, 8, 6, 5), (46, 140, 10, 7, 6),
                                (88, 190, 9, 8, 6), (-120, 210, 10, 8, 6), (24, 118, 7, 6, 4.8)]:
            self._house(cv, X, Z, w, d, h, int(X * 7 + Z), far=True)
        # scattered distant lights (windows, street lamps) of the village in the snow
        glow = np.zeros((PH, PW, 3), np.float32)
        for i in range(46):
            Z = rng.uniform(110, 420)
            X = rng.uniform(-1.0, 1.0) * Z * 0.9
            if abs(X - 3.6) < Z * 0.05:
                continue
            Y = Y_BED + rng.uniform(0.6, 5.0)
            u, v = cam.p(X, Y, Z)
            att = math.exp(-Z / 260.0)
            c = WIN * rng.uniform(0.8, 1.5) if rng.random() < 0.8 else np.array([0.7, 0.85, 1.0], np.float32)
            C.splat(glow, np.array([u]), np.array([v]), max(0.7 * s, 0.5), c, 2.2 * att)
            C.splat(glow, np.array([u]), np.array([v]), 5.0 * s, c, 0.1 * att)
        self._far_glow = glow
        # ground below the forest line: snow field to the horizon (drawn in _ground for near part)
        zf = np.full((PH, PW), 1e4, np.float32)
        ry = self.ry
        sel = ry < -1e-4
        Zb = np.where(sel, Y_BED / np.minimum(ry, -1e-4), 1e4)
        field = sel & (Zb > 110)
        a = field.astype(np.float32)
        fa = self.fog_amt(Zb)[..., None]
        base = AMB_SNOW * 0.85
        colf = base * (1 - fa) + FOG * 1.05 * fa
        cv.paint(0, 0, a * 0 + field, colf, np.where(field, Zb, 1e4))
        cv.rgb = cv.rgb + self._far_glow

    def _house(self, cv, X, Z, w, d, h, seed, far=False, lit=1.0, gable_front=False):
        """Japanese rural house: box + snow-heavy gable roof + warm windows. X, Z = front-left corner."""
        cam, s = self.cam, self.s
        rng = np.random.default_rng(abs(int(seed)))
        y0 = Y_BED
        fa = float(np.clip(self.fog_amt(Z + d * 0.5) * (1.05 if far else 0.72), 0, 0.96))
        wall = np.array([0.1, 0.075, 0.085], np.float32)
        if not hasattr(self, 'win_spill'):
            self.win_spill = []
        wall_s = np.array([0.065, 0.06, 0.085], np.float32)

        def fogc(c, extra=0.0):
            return c * (1 - fa) + FOG * fa * (1 + extra)
        front = [(X, y0, Z), (X + w, y0, Z), (X + w, y0 + h, Z), (X, y0 + h, Z)]
        fp = cam.pts(np.array(front))
        tw_ = int(max(abs(fp[1, 0] - fp[0, 0]), 2)) + 2
        th_ = int(max(abs(fp[0, 1] - fp[3, 1]), 2)) + 2
        wt = PT.wall_tex(tw_ * 2, th_ * 2, int(h / 0.3), wall * 1.25, abs(int(seed)), AMB_SNOW)
        PT.paste_quad(cv, fp, fogc(wt), Z)
        if X > 0:
            side = [(X, y0, Z), (X, y0, Z + d), (X, y0 + h, Z + d), (X, y0 + h, Z)]
            sx = X
        else:
            side = [(X + w, y0, Z), (X + w, y0, Z + d), (X + w, y0 + h, Z + d), (X + w, y0 + h, Z)]
            sx = X + w
        self.quad(cv, side, fogc(wall_s), Z + d / 2)
        # snow drift against the walls
        self.quad(cv, [(X - 0.3, y0 - 0.05, Z - 0.3), (X + w + 0.3, y0 - 0.05, Z - 0.3), (X + w + 0.3, y0 + 0.45, Z - 0.02),
                       (X - 0.3, y0 + 0.5, Z - 0.02)], fogc(AMB_SNOW * 0.95), Z - 0.1)
        floors = [0.9] if h < 4.2 else [0.9, 3.0]
        nwin = max(1, int(w / 2.4))
        wins = []
        for i in range(nwin):
            wx = X + (i + 0.5) * w / nwin
            for fl in floors:
                if rng.random() < 0.35 and not (i == nwin // 2 and fl == floors[-1]):
                    continue
                wins.append((wx, fl, rng.uniform(0.9, 1.5), rng.uniform(0.75, 1.0)))
        for (wx, fl, ww, wh) in wins:
            zq = Z - 0.02
            bright = rng.uniform(1.0, 1.8) * lit
            self.quad(cv, [(wx - ww / 2 - 0.08, y0 + fl - 0.08, zq), (wx + ww / 2 + 0.08, y0 + fl - 0.08, zq),
                           (wx + ww / 2 + 0.08, y0 + fl + wh + 0.08, zq), (wx - ww / 2 - 0.08, y0 + fl + wh + 0.08, zq)],
                      fogc(np.array([0.42, 0.44, 0.5], np.float32) * AMB_SNOW * 0.7 + 0.01), Z)
            wp = cam.pts(np.array([(wx - ww / 2, y0 + fl, zq), (wx + ww / 2, y0 + fl, zq), (wx + ww / 2, y0 + fl + wh, zq),
                                   (wx - ww / 2, y0 + fl + wh, zq)]))
            ww_px = int(abs(wp[1, 0] - wp[0, 0])) + 1
            wh_px = int(abs(wp[0, 1] - wp[3, 1])) + 1
            wtex = PT.house_window(ww_px * 2, wh_px * 2, abs(int(seed * 31 + wx * 17 + fl * 5)), bright * 0.85)
            PT.paste_quad(cv, wp, wtex * (1 - fa * 0.45) + FOG * fa * 0.3, Z)
            self.quad(cv, [(wx - ww / 2 - 0.14, y0 + fl - 0.1, zq - 0.02), (wx + ww / 2 + 0.14, y0 + fl - 0.1, zq - 0.02),
                           (wx + ww / 2 + 0.12, y0 + fl + 0.02, zq - 0.02), (wx - ww / 2 - 0.12, y0 + fl + 0.02, zq - 0.02)],
                      fogc(AMB_SNOW * 1.25 + WIN * 0.25 * bright), Z - 0.05)
            if not far:
                self.win_spill.append((wx, y0 + fl, Z, ww, bright * (1 - fa * 0.5)))
        if not far:
            AR.house_details(self, cv, X, Z, w, d, h, seed, fa, fogc, wins, gable_front=gable_front, amb=AMB_SNOW,
                             Y0=Y_BED, snow_col=np.array([0.52, 0.62, 0.9], np.float32))
            if (X, Z) == (14.0, 52.0):
                AR.shop_signs(self, cv, X, Z, w, h, fa, fogc, AMB_SNOW, Y0=Y_BED)
        # side window
        if rng.random() < 0.7:
            zc = Z + d * 0.5
            sp = cam.pts(np.array([(sx, y0 + 1.0, zc - 0.6), (sx, y0 + 1.0, zc + 0.6), (sx, y0 + 1.8, zc + 0.6), (sx, y0 + 1.8, zc - 0.6)]))
            stex = PT.house_window(int(abs(sp[1, 0] - sp[0, 0])) * 2 + 4, int(abs(sp[0, 1] - sp[3, 1])) * 2 + 4, abs(int(seed)) + 99, 1.0 * lit)
            PT.paste_quad(cv, sp, stex * (1 - fa * 0.45) + FOG * fa * 0.3, zc)
        if gable_front:
            self._gable_roof(cv, X, Z, w, d, h, fa, fogc, wall, seed)
            return
        # gable roof (ridge parallel to the front): front slope visible, heavy snow on it
        rh = h * 0.45
        ov = 0.55
        ridge_z = Z + d / 2
        eave_y = y0 + h
        snow_t = 0.5
        # fascia under the eave
        self.quad(cv, [(X - ov, eave_y - 0.1, Z - ov), (X + w + ov, eave_y - 0.1, Z - ov), (X + w + ov, eave_y + 0.12, Z - ov),
                       (X - ov, eave_y + 0.12, Z - ov)], fogc(wall * 0.5), Z - ov)
        roofp = [(X - ov, eave_y + snow_t, Z - ov), (X + w + ov, eave_y + snow_t, Z - ov),
                 (X + w + ov, eave_y + rh + snow_t, ridge_z), (X - ov, eave_y + rh + snow_t, ridge_z)]
        pts = cam.pts(np.array(roofp))
        x0, y0_, m = L.poly_local(pts)
        hh = m.shape[0]
        g = np.linspace(1.0, 0.95, hh, dtype=np.float32)[:, None, None]
        snow_roof = np.array([0.52, 0.62, 0.9], np.float32)
        cv.paint(x0, y0_, m, fogc(snow_roof) * g * np.ones((1, m.shape[1], 1), np.float32), Z)
        # rounded snow overhang along the eave
        n = 30
        xs = np.linspace(X - ov - 0.1, X + w + ov + 0.1, n)
        top = np.stack([xs, eave_y + snow_t + 0.02 * np.sin(xs * 3 + seed), np.full(n, Z - ov - 0.05)], 1)
        bot = np.stack([xs, eave_y + 0.05 - 0.1 * np.abs(np.sin(xs * 1.3 + seed)) ** 3, np.full(n, Z - ov - 0.05)], 1)
        x0, y0_, m = L.poly_local(cam.pts(np.concatenate([top, bot[::-1]])))
        hh = m.shape[0]
        g = np.where(np.linspace(0, 1, hh, dtype=np.float32) < 0.3, 1.0, 0.62).astype(np.float32)[:, None, None]
        cv.paint(x0, y0_, m, fogc(snow_roof * 0.95) * g * np.ones((1, m.shape[1], 1), np.float32), Z - ov)
        lw = max(cam.f * 0.06 / Z, 0.6 * s)
        # crisp lit rim along the snow edge and the ridge (sky light), dark gutter under the eave
        x0, y0_, m = L.line_local(cam.pts(top), lw)
        cv.paint(x0, y0_, m * 0.9, fogc(snow_roof * 1.35), Z - ov - 0.06)
        ridge = np.array([[X - ov, eave_y + rh + snow_t, ridge_z], [X + w + ov, eave_y + rh + snow_t, ridge_z]])
        x0, y0_, m = L.line_local(cam.pts(ridge), lw)
        cv.paint(x0, y0_, m * 0.9, fogc(snow_roof * 1.3), Z)
        gut = np.array([[X - ov + 0.1, eave_y - 0.02, Z - ov + 0.05], [X + w + ov - 0.1, eave_y - 0.02, Z - ov + 0.05]])
        x0, y0_, m = L.line_local(cam.pts(gut), max(cam.f * 0.1 / Z, 0.7 * s))
        cv.paint(x0, y0_, m, fogc(wall * 0.35), Z - ov + 0.05)
        irng = np.random.default_rng(abs(int(seed)) + 17)
        for i in range(int(w * 2.2)):
            xi = X - ov + 0.2 + irng.random() * (w + 2 * ov - 0.4)
            ln = 0.08 + 0.55 * irng.random() ** 2.5
            wd = 0.03 + 0.05 * ln
            P = np.array([[xi - wd, eave_y - 0.05, Z - ov + 0.04], [xi + wd, eave_y - 0.05, Z - ov + 0.04],
                          [xi, eave_y - 0.05 - ln, Z - ov + 0.04]])
            x0, y0_, m = L.poly_local(cam.pts(P), ss=4)
            cv.paint(x0, y0_, m * 0.9, fogc(snow_roof * 1.15 + WIN * 0.1), Z - ov + 0.04)
        # entrance lamp
        if rng.random() < 0.6 and not far:
            ex = X + w * rng.uniform(0.2, 0.8)
            u, v = cam.p(ex, y0 + 2.2, Z - 0.1)
            gl = np.zeros((1, 1), np.float32)
            x0, y0_, m = L.poly_local(np.array([[u - 1.2 * s, v - 1.2 * s], [u + 1.2 * s, v - 1.2 * s], [u + 1.2 * s, v + 1.2 * s],
                                               [u - 1.2 * s, v + 1.2 * s]]))
            cv.paint(x0, y0_, m, WIN * 3.0, Z - 0.1)

    def _gable_roof(self, cv, X, Z, w, d, h, fa, fogc, wall, seed):
        """Gable end facing the camera: triangular gable wall, two snow-laden slopes (the one facing the
        camera axis visible), thick rounded snow edges and icicles."""
        cam, s = self.cam, self.s
        y0 = Y_BED + h
        rh = w * 0.42
        xm = X + w / 2
        ov = 0.5
        # gable wall
        self.quad(cv, [(X, y0, Z), (X + w, y0, Z), (xm, y0 + rh, Z)], fogc(wall * 0.9), Z)
        # visible slope: faces the camera axis (+x side if the house is left of camera)
        sgn = 1 if X < 0 else -1
        xe = X + w + ov if sgn > 0 else X - ov
        snow = np.array([0.52, 0.62, 0.9], np.float32)
        yr = y0 + rh + 0.45
        slope = [(xm, yr, Z - ov), (xe, y0 - 0.2 + 0.45, Z - ov), (xe, y0 - 0.2 + 0.45, Z + d + ov), (xm, yr, Z + d + ov)]
        x0, y0_, m = L.poly_local(cam.pts(np.array(slope)))
        gg = np.linspace(1.0, 0.82, m.shape[1], dtype=np.float32)[None, :, None]
        if sgn < 0:
            gg = gg[:, ::-1]
        cv.paint(x0, y0_, m, fogc(snow) * gg * np.ones((m.shape[0], 1, 1), np.float32), Z + d / 2)
        # thick snow along both front rakes (seen edge-on as a band) with a rounded top
        n = 24
        for side in (-1, 1):
            xs_ = np.linspace(xm, xm + side * (w / 2 + ov), n)
            tt = np.linspace(0, 1, n)
            ytop = yr - (rh + 0.2) * tt + 0.05 * np.sin(tt * 9 + seed)
            ybot = ytop - 0.5 + 0.08 * np.abs(np.sin(tt * 7 + seed)) ** 2
            P = np.concatenate([np.stack([xs_, ytop, np.full(n, Z - ov)], 1), np.stack([xs_[::-1], ybot[::-1], np.full(n, Z - ov)], 1)])
            x0, y0_, m = L.poly_local(cam.pts(P))
            vv = np.where(np.linspace(0, 1, m.shape[0], dtype=np.float32) < 0.35, 1.02, 0.68).astype(np.float32)[:, None, None]
            cv.paint(x0, y0_, m, fogc(snow) * vv * np.ones((1, m.shape[1], 1), np.float32), Z - ov)
            x0, y0_, m = L.line_local(cam.pts(np.stack([xs_, ytop, np.full(n, Z - ov - 0.02)], 1)), max(cam.f * 0.06 / Z, 0.6 * s))
            cv.paint(x0, y0_, m * 0.9, fogc(snow * 1.35), Z - ov - 0.02)
            x0, y0_, m = L.line_local(cam.pts(np.stack([xs_, ybot + 0.02, np.full(n, Z - ov - 0.01)], 1)), max(cam.f * 0.07 / Z, 0.6 * s))
            cv.paint(x0, y0_, m * 0.8, fogc(wall * 0.4), Z - ov - 0.01)
        # icicles along the lower rake ends
        rng = np.random.default_rng(seed + 3)
        for i in range(10):
            xi = X + rng.uniform(0.05, 0.95) * w
            ln = rng.uniform(0.1, 0.5)
            P = np.array([[xi - 0.04, y0 - 0.05, Z - 0.05], [xi + 0.04, y0 - 0.05, Z - 0.05], [xi, y0 - 0.05 - ln, Z - 0.05]])
            x0, y0_, m = L.poly_local(cam.pts(P))
            cv.paint(x0, y0_, m * 0.8, fogc(snow * 0.9), Z - 0.05)
        # small attic window glowing in the gable
        self.quad(cv, [(xm - 0.35, y0 + 0.4, Z - 0.02), (xm + 0.35, y0 + 0.4, Z - 0.02), (xm + 0.35, y0 + 1.0, Z - 0.02),
                       (xm - 0.35, y0 + 1.0, Z - 0.02)], WIN * 1.3 * (1 - fa * 0.45) + FOG * fa * 0.3, Z)

    # ------------------------------------------------------------------ ground (raycast planes)
    def _snow_tex(self, X, Z, seed, scale=0.06):
        """Sample world-space fbm texture at floor coords (perspective correct)."""
        key = ('tex', seed)
        if not hasattr(self, '_tex'):
            self._tex = {}
        if key not in self._tex:
            self._tex[key] = C.fbm(768, 768, 12, 5, seed=seed).astype(np.float32)
        T = self._tex[key]
        mx = (X / scale).astype(np.float32) + 384
        mz = (Z / scale).astype(np.float32) + 100
        return cv2.remap(T, mx, mz, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _ground(self):
        cam, cv, s = self.cam, self.gnd, self.s
        PW, PH = self.PW, self.PH
        rx, ry = self.rx, self.ry
        down = ry < -1e-5
        ryc = np.minimum(ry, -1e-5)
        # platform plane
        Zp = Y_PLAT / ryc
        Xp = rx * Zp + CAM_X
        plat = down & (Xp > PLAT_X0) & (Xp < PLAT_X1) & (Zp < PLAT_Z1)
        # bed / field plane
        Zb = Y_BED / ryc
        Xb = rx * Zb + CAM_X
        bed = down & ~plat & (Zb < 130)
        self.plat_id = plat.astype(np.float32)

        # ---------- platform surface
        Z = np.where(plat, Zp, 1.0)
        X = np.where(plat, Xp, 0.0)
        Pw = np.stack([X, np.full_like(X, Y_PLAT), Z], -1)
        Nup = np.array([0, 1, 0], np.float32)
        E = (self.light_at(Pw, Nup, vis0=self.sign_shadow(X, Z)) + self.door_light(X, Z) + self.window_light(X, Z)) * 0.72
        E = paint_E(E, step=2.1, soft=0.14, mix=0.6)       # painted warm pool plateaus, not a CG falloff
        under = ((X > CAN_X0 - 0.2) & (X < CAN_X1) & (Z > CAN_Z0 - 0.3) & (Z < CAN_Z1 + 0.3)).astype(np.float32)
        under = C.blur(under, 2.0 * s)
        # canopy occlusion of the sky ambient, drifting snow reaching in at the canopy edges
        t1 = self._snow_tex(X, Z, 5, 0.05)
        t2 = self._snow_tex(X * 0.3, Z * 0.3, 6, 0.05)
        edge_d = np.minimum.reduce([np.abs(Z - CAN_Z0), np.abs(Z - CAN_Z1), np.abs(X - CAN_X1)])
        drift = C.smoothstep(1.6, 0.0, edge_d + (t2 - 0.5) * 1.6) * under
        snowcov = np.clip(1 - under + drift, 0, 1)
        # wall AO
        ao = 1 - 0.5 * C.smoothstep(1.2, 0.0, np.abs(X - BLD_X1)) * ((Z > BLD_Z0) & (Z < BLD_Z1) & (X > BLD_X1))
        amb = AMB_SNOW * (1 - 0.8 * under[..., None]) * ao[..., None]
        # snow undulation (bump lighting from the gradient of a world-space texture)
        tb = self._snow_tex(X, Z + 0.25, 5, 0.05)
        # wind-sculpted drift ripples: world texture stretched across the platform (reads as soft
        # horizontal brush bands that tighten with distance), lit from the lamp side
        big = self._snow_tex(X * 0.3, Z * 0.5, 16, 0.05)
        wv = self._snow_tex(X * 0.5, Z * 0.5, 15, 0.05)
        ph = Z / 0.55 * 2 * np.pi + 7.0 * wv + 1.5 * X
        # asymmetric wind ripple profile: gentle windward slope, sharper lit crest
        rip = np.sin(ph) + 0.35 * np.sin(2 * ph + 0.8)
        rip = rip * C.smoothstep(0.25, 0.75, big)
        near = C.smoothstep(30.0, 3.0, Z) * C.smoothstep(2.0, 6.0, cam.f / np.maximum(Z, 1) * 0.55 / 12 * 12)
        bump = 1 + 0.6 * (tb - t1) + 0.08 * (t2 - 0.5) + (0.055 * rip + 0.06 * (big - 0.5)) * near
        # broad wind-drift undulations: slopes facing the lamp/camera catch more light
        dr1 = self._snow_tex(X * 0.35, Z * 0.35 + 0.25, 18, 0.05) - self._snow_tex(X * 0.35, Z * 0.35, 18, 0.05)
        dr2 = self._snow_tex(X * 0.35 + 0.25, Z * 0.35, 18, 0.05) - self._snow_tex(X * 0.35, Z * 0.35, 18, 0.05)
        bump = bump - 2.2 * dr1 + 0.8 * dr2
        # fine painted value texture: small wind-scoured crust flecks and soft brush strokes
        fine = self._snow_tex(X * 3.0, Z * 3.0, 27, 0.05)
        fine2 = self._snow_tex(X * 3.0, Z * 3.0 + 0.08, 27, 0.05)
        stroke = self._snow_tex(X * 0.8, Z * 3.2, 28, 0.05)
        nf = C.smoothstep(24.0, 2.0, Z)
        bump = bump + (2.6 * (fine2 - fine) + 0.16 * (stroke - 0.5)) * nf
        # drifted ridge along the platform edge (wind piles snow against the lip): crest lit by the
        # lamps, the lee toward the track in cool blue
        dx = PLAT_X1 - X
        rc = 0.42 + 0.22 * (t2 - 0.5) + 0.1 * np.sin(Z * 0.9)
        rw = 0.2 + 0.05 * np.sin(Z * 1.7 + 1.0)
        hh = np.exp(-((dx - rc) / rw) ** 2) * (1 - under)
        slope = -2 * (dx - rc) / (rw * rw) * hh          # d h / d dx
        bump = bump * (1 + 0.1 * hh + 0.06 * np.clip(slope, -2.5, 2.5))
        self._edge_ridge = (hh, slope)
        # footprints trail
        fp = self._footprints(X, Z)
        # snow colour
        alb_snow = 0.92
        snow_col = snow_lit(amb * alb_snow, E * alb_snow * (1 - 0.45 * under[..., None]), bump[..., None])
        # painted cast shadows: every lamp that reaches the snow throws crisp blue-violet shadows of the
        # sign legs, bench, bins and lamp pole away from itself (deeper and more violet than open snow)
        shd_col = snow_lit(amb * alb_snow, E * 0.0, bump[..., None]) * np.array([0.6, 0.54, 1.0], np.float32)
        for li, kmin in ((0, 0.3), (len(self.lamps) - 1, 0.6), (4, 0.5)):
            occ = self._occ(X, Z, li)
            El = self._lamp_E(li, Pw, Nup)
            lp = self.lamps[li][0]
            dl = np.sqrt((X - lp[0]) ** 2 + (Z - lp[2]) ** 2)
            k = np.clip(kmin * 1.3 * np.exp(-dl / 9.0) + El * 2.4, 0, 0.95) * occ * (1 - under)
            snow_col = C.lerp(snow_col, shd_col, k[..., None])
        # footprints: the lamp light cannot reach into the prints -> cool blue-violet hollows with a lit
        # crescent on the far wall
        # (cycle 7) painted prints: a mid cool hollow, ONE darker cool shadow value on the far inner wall
        # (it faces away from the lamps), and a lit crescent on the near lip
        pk = fp[0][..., None]
        hollow = snow_col * np.array([0.66, 0.64, 0.84], np.float32) + AMB_SNOW * np.array([0.03, 0.02, 0.1], np.float32)
        wallc = snow_col * np.array([0.4, 0.4, 0.66], np.float32) + AMB_SNOW * np.array([0.02, 0.01, 0.08], np.float32)
        rimc = snow_lit(amb * alb_snow, E * alb_snow * 1.35, 1.1)
        snow_col = C.lerp(snow_col, hollow, np.clip(pk * 1.05, 0, 1))
        snow_col = C.lerp(snow_col, wallc, (fp[1] * 0.85)[..., None])
        snow_col = C.lerp(snow_col, rimc * 1.1, (fp[2] * 0.8)[..., None])
        # concrete under the canopy (wet, dark, reflective)
        conc = np.array([0.25, 0.22, 0.2], np.float32) * (0.8 + 0.4 * t1[..., None])
        # each canopy tube throws its own soft oval pool onto the concrete below it
        pools = np.zeros(X.shape, np.float32)
        for (lp, lc_, I_, cone_, cp_, sc_) in self.lamps:
            if lp[1] < CAN_Y and CAN_Z0 < lp[2] < CAN_Z1 and cone_ > 0:
                pools += I_ / 5.5 * np.exp(-((X - lp[0]) / 1.0) ** 2 - ((Z - lp[2]) / 0.95) ** 2)
        conc_col = conc * (amb * 0.8 + E * 0.7 + pools[..., None] * WHITE_W * 1.6)
        # glossy reflections of lamps on the wet concrete
        refl = self._wet_reflections(rx, ry, Pw)
        conc_col = conc_col + refl * 0.55
        tact = C.smoothstep(1.28, 1.31, X) * C.smoothstep(1.62, 1.59, X)
        tex_t = 0.8 + 0.2 * (np.cos(Z / 0.3 * 2 * np.pi) > 0.85)
        tcol = np.array([0.62, 0.48, 0.08], np.float32) * (amb * 0.5 + E * 0.9) * tex_t[..., None]
        conc_col = C.lerp(conc_col, tcol + refl * 0.3, tact[..., None])
        col = C.lerp(conc_col, snow_col, snowcov[..., None])
        # tactile paving strip under thin snow: a faint yellow line, bumps showing where it is trodden
        tstrip = C.smoothstep(1.2, 1.24, X) * C.smoothstep(1.56, 1.52, X) * snowcov
        thin = C.smoothstep(0.45, 0.7, t2 * 0.7 + t1 * 0.3)
        pxm = cam.f / np.maximum(Z, 0.5)
        dots = ((np.cos(X / 0.075 * 2 * np.pi) * np.cos(Z / 0.075 * 2 * np.pi)) > 0.55).astype(np.float32)
        dots = dots * C.smoothstep(25, 60, pxm * 0.6)
        ylw = np.array([0.75, 0.55, 0.1], np.float32) * (amb * 0.8 + E * 0.75)
        tk = tstrip * (0.22 + 0.4 * thin) * (1 - 0.25 * dots)
        col = C.lerp(col, ylw, tk[..., None])
        col = col + (tstrip * (0.4 + thin) * dots * 0.3)[..., None] * (AMB_SNOW * 0.5 + E * 0.5)
        # lee of the edge ridge: soft blue shadow
        hh, slope = self._edge_ridge
        lee = np.clip(-slope, 0, 3) / 3 * (1 - under)
        col = C.lerp(col, col * np.array([0.62, 0.72, 1.0], np.float32), (lee * 0.55)[..., None])
        # wind-drifted snow banked against the back fence (a soft ridge catching the light)
        drift_f = C.smoothstep(PLAT_X0 + 1.4, PLAT_X0 + 0.15, X) * (Z > 3)
        col = col * (1 + 0.18 * drift_f[..., None] * (0.6 + 0.8 * t2[..., None]))
        # platform edge snow lip
        lip = C.smoothstep(PLAT_X1 - 0.25, PLAT_X1 - 0.07, X) * C.smoothstep(PLAT_X1 - 0.01, PLAT_X1 - 0.04, X) * plat
        col = col + lip[..., None] * (AMB_SNOW * 0.3 + E * 0.45)
        # cornice underside: crisp cool shadow right at the drop
        esh = C.smoothstep(PLAT_X1 - 0.045, PLAT_X1 - 0.015, X) * plat
        col = C.lerp(col, AMB_SNOW * 0.35 + E * 0.05, (esh * 0.85)[..., None])
        # tactile strip hint under snow
        fa = self.fog_amt(Z)[..., None]
        col = col * (1 - fa) + FOG * fa
        a = plat.astype(np.float32)
        a = np.where(plat, np.maximum(cv2.GaussianBlur(a, (0, 0), 0.6), 0.5), cv2.GaussianBlur(a, (0, 0), 0.6))
        cv.paint(0, 0, a, col, np.where(plat, Zp, 1e4))

        # ---------- track bed & field
        Z = np.where(bed, Zb, 1.0)
        X = np.where(bed, Xb, 0.0)
        Pw = np.stack([X, np.full_like(X, Y_BED), Z], -1)
        E = self.light_at(Pw, Nup)
        t1 = self._snow_tex(X, Z, 7, 0.06)
        tb = self._snow_tex(X, Z + 0.3, 7, 0.06)
        t3 = self._snow_tex(X * 0.2, Z * 0.2, 8, 0.05)
        wv = self._snow_tex(X * 0.5, Z * 0.5, 17, 0.05)
        ph = Z / 0.8 * 2 * np.pi + 8.0 * wv + 0.8 * X
        rip = (np.sin(ph) + 0.35 * np.sin(2 * ph + 0.8)) * C.smoothstep(3.0, 8.0, cam.f / np.maximum(Z, 1) * 0.8)
        bump = 1 + 0.7 * (tb - t1) + 0.12 * (t3 - 0.5)
        # shadow of the platform wall on the bed next to it
        ao = 1 - 0.6 * C.smoothstep(1.5, 0.9, X - PLAT_X1) * (Z < PLAT_Z1)
        # buried sleepers: subtle ripples between the rails
        # snow-capped sleepers: rhythmic dark front faces + bright caps + shadowed troughs (filtered by
        # the pixel footprint so they fade cleanly into the distance instead of aliasing)
        per = 0.62
        dz = Z * Z / (cam.f * abs(Y_BED)) / per          # footprint in periods per pixel
        pp = np.mod(Z / per + 0.3, 1.0)
        fw = np.clip(dz, 0.004, 1.0)

        def band(p0, p1):
            return C.smoothstep(p0 - fw, p0 + fw, pp) * C.smoothstep(p1 + fw, p1 - fw, pp)
        sx = C.smoothstep(RAILS[0] - 0.62, RAILS[0] - 0.5, X) * C.smoothstep(RAILS[1] + 0.62, RAILS[1] + 0.5, X)
        vis_ = C.smoothstep(0.45, 0.12, dz) * sx
        face = band(0.0, 0.085) * vis_
        cap = band(0.085, 0.36) * vis_
        trough = (1 - band(0.0, 0.36)) * vis_ * C.smoothstep(0.4, 1.0, pp)
        bump = bump * (1 + 0.07 * cap - 0.14 * trough) - 0.03 * vis_ * (1 - C.smoothstep(0.45, 0.12, dz))
        groove = np.exp(-((X - RAILS[0] - 0.09) / 0.06) ** 2) + np.exp(-((X - RAILS[1] + 0.09) / 0.06) ** 2) +             np.exp(-((X - RAILS[0] + 0.09) / 0.06) ** 2) * 0.6 + np.exp(-((X - RAILS[1] - 0.09) / 0.06) ** 2) * 0.6
        bump = bump * (1 - 0.3 * np.clip(groove, 0, 1))
        col = snow_lit(AMB_SNOW * 0.92 * ao[..., None], E * 0.9, bump[..., None])
        wood = np.array([0.05, 0.04, 0.05], np.float32) + E * 0.06 + AMB_SNOW * 0.1
        # sleepers buried: only a soft blue dent and a sliver of dark wood where the snow broke off
        wn = self._snow_tex(X * 2.0, Z * 2.0, 44, 0.05)
        bare = C.smoothstep(0.62, 0.75, wn)
        col = C.lerp(col, col * np.array([0.66, 0.74, 0.96], np.float32), (face * 0.4 * (0.5 + 0.8 * wn))[..., None])
        col = C.lerp(col, wood, (face * 0.5 * bare)[..., None])
        col = self._field_detail(col, X, Z, E, bed)
        fa = self.fog_amt(Z)[..., None]
        col = col * (1 - fa) + FOG * 1.02 * fa
        a = bed.astype(np.float32) * C.smoothstep(130, 105, Zb)
        col = self._rails_rc(col, rx, ry)
        cv.paint(0, 0, a, col, np.where(bed, Zb, 1e4))

    def _field_detail(self, col, X, Z, E, sel):
        """Painted texture for the open snowfield right of the tracks: long wind drifts with soft blue
        shadow sides and lamp-lit crests, sastrugi ripples, a pair of old tyre ruts and a line of
        footprints heading to the village."""
        cam = self.cam
        fr = C.smoothstep(RAILS[1] + 0.9, RAILS[1] + 2.2, X) * sel * C.smoothstep(110, 60, Z)
        if not fr.any():
            return col
        pxm = cam.f / np.maximum(Z, 0.5)                       # pixels per metre on the ground
        # drift heightfield: long ridges running roughly along the track, warped
        w1 = self._snow_tex(X * 0.25, Z * 0.12, 41, 0.05)
        w2 = self._snow_tex(X * 0.6, Z * 0.3, 42, 0.05)

        def hgt(Xq, Zq):
            ph = Xq / 2.6 * 2 * np.pi + 5.0 * self._snow_tex(Xq * 0.25, Zq * 0.12, 41, 0.05) + 0.12 * Zq
            return 0.5 + 0.5 * np.sin(ph) + 0.35 * np.sin(2 * ph + 1.3)
        e = 0.12
        dhx = (hgt(X + e, Z) - hgt(X - e, Z)) / (2 * e)
        dhz = (hgt(X, Z + e) - hgt(X, Z - e)) / (2 * e)
        # light from the station side (-x) and from the camera side (-z): lit windward faces, blue lees
        sl = -dhx * 0.55 - dhz * 0.35
        amp = C.smoothstep(0.6, 3.0, pxm * 0.35) * (0.55 + 0.45 * w1)
        sl = np.clip(sl * 0.45 * amp, -1, 1)
        shadow = np.clip(-sl, 0, 1)[..., None]
        lit = np.clip(sl, 0, 1)[..., None]
        blue = col * np.array([0.6, 0.7, 1.02], np.float32)
        out = C.lerp(col, blue, shadow * 0.9)
        out = out * (1 + lit * np.array([0.22, 0.2, 0.18], np.float32)) + lit * E * 0.25
        # fine sastrugi (only where resolvable)
        rip = np.sin(Z / 0.35 * 2 * np.pi + 9 * w2 + 1.4 * X)
        rv = C.smoothstep(3.0, 9.0, pxm * 0.35 / 2.5)
        out = out * (1 + 0.035 * rip * rv * (0.4 + w2))[..., None]
        # tyre ruts: two parallel grooves along a gentle curve toward the village
        Xc = 5.8 + 0.1 * Z + 0.0025 * Z * Z
        for off in (-0.8, 0.8):
            d = X - (Xc + off)
            hw = 0.16
            fw = np.maximum(1.0 / np.maximum(pxm, 1e-3), 0.02)
            g = C.smoothstep(hw + fw, hw * 0.4, np.abs(d)) * C.smoothstep(55, 30, Z)
            rim = C.smoothstep(hw + 0.08 + fw, hw, np.abs(d)) * (d > 0) - g * 0.0
            gcol = out * np.array([0.52, 0.6, 0.9], np.float32)
            out = C.lerp(out, gcol, (g * 0.55 * fr)[..., None])
            out = out + (np.clip(rim - g, 0, 1) * 0.1 * fr)[..., None] * (AMB_SNOW + E * 0.8)
        # footprints: alternate left/right every 0.36 m along another path
        Xf = 6.3 + 0.06 * Z + 0.004 * Z * Z
        k = np.floor(Z / 0.36)
        side = np.where(np.mod(k, 2) > 0.5, 0.12, -0.12)
        zc = (k + 0.5) * 0.36
        d = np.sqrt(((X - Xf - side) / 0.075) ** 2 + ((Z - zc) / 0.15) ** 2)
        vis = C.smoothstep(2.0, 5.0, pxm * 0.07) * C.smoothstep(40, 25, Z)
        fp = C.smoothstep(1.1, 0.7, d) * vis * fr
        out = C.lerp(out, out * np.array([0.5, 0.58, 0.88], np.float32), (fp * 0.8)[..., None])
        return C.lerp(col, out, fr[..., None])

    def _footprints(self, X, Z):
        """Boot prints in the platform snow: paired left/right prints (sole + separate heel, toes turned
        slightly out) along curved trails - from the camera to the bench, from the bench on to the
        waiting-room door, and a short detour out to the platform edge under the lamp and back.
        Returns (depression, lit far wall, kicked-up rim)."""
        rng = np.random.default_rng(21)
        dep = np.zeros_like(X)
        wall = np.zeros_like(X)
        rim = np.zeros_like(X)
        prints = []
        trails = ((np.array([0.75, 1.2]), np.array([0.45, 5.6]), np.array([-0.95, 9.4]), 0.36, 0),
                  (np.array([-1.05, 10.9]), np.array([-1.3, 15.0]), np.array([-4.3, 17.7]), 0.36, 1),
                  (np.array([-0.85, 8.6]), np.array([0.6, 7.2]), np.array([1.55, 7.9]), 0.34, 0),
                  (np.array([1.45, 7.25]), np.array([0.8, 5.6]), np.array([0.55, 3.2]), 0.35, 1))
        for (p0, p1, p2, step, off) in trails:
            # arc length -> evenly spaced steps along the quadratic curve
            tt = np.linspace(0, 1, 400)
            P = (1 - tt)[:, None] ** 2 * p0 + 2 * ((1 - tt) * tt)[:, None] * p1 + (tt * tt)[:, None] * p2
            seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
            # (cycle 7) uneven stride: a real walk, not a stamped row
            sls = [0.15]
            while sls[-1] < seg[-1] - 0.1 - step:
                sls.append(sls[-1] + step * rng.uniform(0.72, 1.3))
            for k, sl in enumerate(sls):
                t = float(np.interp(sl, seg, tt))
                p = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2
                dp = 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)
                dp = dp / np.linalg.norm(dp)
                nrm = np.array([dp[1], -dp[0]])
                side = 1.0 if (k + off) % 2 else -1.0
                ang = -side * rng.uniform(0.08, 0.2)                  # toes turned slightly out
                ca, sa = math.cos(ang), math.sin(ang)
                d2 = np.array([dp[0] * ca - dp[1] * sa, dp[0] * sa + dp[1] * ca])
                prints.append((p + nrm * side * rng.uniform(0.07, 0.12) + rng.normal(0, 0.015, 2), d2, rng.uniform(0.7, 1.0),
                               rng.uniform(0, 6.28, 3), rng.uniform(0.9, 1.15)))
        sel0 = (Z > -1) & (Z < 19) & (X > -4.8) & (X < 2.0)
        ii = np.nonzero(sel0)
        Xa, Za = X[ii], Z[ii]
        depa = np.zeros_like(Xa)
        walla = np.zeros_like(Xa)
        rima = np.zeros_like(Xa)
        for (c, dv, depth, ph, ln) in prints:
            ex, ez = Xa - c[0], Za - c[1]
            near = (np.abs(ex) < 0.35) & (np.abs(ez) < 0.35)
            if not near.any():
                continue
            j = np.nonzero(near)[0]
            ex, ez = ex[j], ez[j]
            a_ = ex * dv[1] - ez * dv[0]          # across the boot
            b_ = ex * dv[0] + ez * dv[1]          # along the boot (toe +)
            sole = np.sqrt((a_ / 0.05) ** 2 + ((b_ - 0.06) / (0.14 * ln)) ** 2)
            heel = np.sqrt((a_ / 0.045) ** 2 + ((b_ + 0.14 * ln) / 0.05) ** 2)
            d = np.minimum(sole, heel * 1.04)
            # crumbly, irregular outline (snow breaks unevenly at the edge of a print)
            d = d + 0.07 * np.sin(a_ * 95 + ph[0]) * np.sin(b_ * 70 + ph[1]) + 0.05 * np.sin(b_ * 160 + a_ * 40 + ph[2])
            # deeper toward the toe / heel centres, soft crumbly edge
            m = C.smoothstep(1.15, 0.65, d) * depth * (0.75 + 0.25 * C.smoothstep(0.9, 0.3, d))
            depa[j] = np.maximum(depa[j], m)
            # the inner wall on the far side (facing the camera) catches the light
            walla[j] = np.maximum(walla[j], C.smoothstep(0.35, 0.8, d) * C.smoothstep(1.12, 0.95, d) * C.smoothstep(-0.01, 0.06, ez) * depth)
            # little ridge of kicked-up snow round the print (strongest at the toe)
            rr = C.smoothstep(1.0, 1.12, d) * C.smoothstep(1.55, 1.2, d) * (0.5 + 0.5 * C.smoothstep(-0.05, 0.12, b_))
            rr = rr * C.smoothstep(0.03, -0.04, ez)        # (cycle 7) the NEAR lip faces the lamps -> lit
            rima[j] = np.maximum(rima[j], rr * depth)
        dep[ii] = depa
        wall[ii] = walla
        rim[ii] = rima
        return dep, wall, rim

    def _wet_reflections(self, rx, ry, Pw):
        out = np.zeros(rx.shape + (3,), np.float32)
        # reflected view ray direction
        n = np.sqrt(rx * rx + ry * ry + 1)
        R = np.stack([rx / n, -ry / n, 1 / n], -1)
        for (lp, col, I, cone, cpow, _) in self.lamps:
            d = np.asarray(lp, np.float32) - Pw
            dist = np.sqrt((d ** 2).sum(-1)) + 1e-3
            dn = d / dist[..., None]
            # anisotropic glossy lobe: narrow sideways, long toward the camera (a vertical streak)
            ex = R[..., 0] - dn[..., 0]
            ey = R[..., 1] - dn[..., 1]
            lobe = np.exp(-(ex / 0.012) ** 2 - (ey / 0.05) ** 2) * (dn[..., 2] > 0)
            if lp[1] > CAN_Y:
                # the reflected ray toward a lamp above the roof is blocked by the canopy
                tc = (CAN_Y - Pw[..., 1]) / (lp[1] - Pw[..., 1])
                xc = Pw[..., 0] + tc * (lp[0] - Pw[..., 0])
                zc = Pw[..., 2] + tc * (lp[2] - Pw[..., 2])
                blk = (xc > CAN_X0) & (xc < CAN_X1) & (zc > CAN_Z0) & (zc < CAN_Z1)
                lobe = np.where(blk, 0.0, lobe)
            out += (lobe * min(I, 6) * 0.16)[..., None] * col
        return out

    def _rails_rc(self, col, rx, ry):
        """Ray-cast rails over the bed: dark rail body (web side) + a polished head that mirrors the cool
        sky glow along its whole length and flashes warm where the lamps glint on it."""
        cam = self.cam
        ryc = np.minimum(ry, -1e-5)
        out = col.copy()
        for rxw in RAILS:
            for (yh, hw, kind) in ((0.07, 0.045, 0), (0.15, 0.03, 1)):
                Zh = (Y_BED + yh) / ryc
                Xh = rx * Zh + CAM_X
                fw = Zh / cam.f * 1.2 + 0.004
                m = C.smoothstep(hw + fw, hw - fw * 0.3, np.abs(Xh - rxw)) * (Zh < 140) * (ry < -1e-5)
                m = m * np.clip(hw * 2 / (fw * 2), 0, 1) ** 0.5 if kind == 1 else m * np.clip(hw * 2 / fw, 0, 1)
                # snow drifted over the rail in patches (the line is half buried, not freshly cleared)
                zz_ = Zh + 0.37 * rxw
                cover = C.smoothstep(0.5, 0.78, 0.5 + 0.28 * np.sin(zz_ * 1.3 + rxw) + 0.2 * np.sin(zz_ * 3.7 + 2 * rxw)
                                     + 0.12 * np.sin(zz_ * 9.1 + 5 * rxw))
                m = m * (1 - 0.9 * cover)
                fa = self.fog_amt(Zh)[..., None]
                if kind == 0:
                    c = np.array([0.03, 0.035, 0.07], np.float32) * (1 - fa) + FOG * fa
                    out = C.lerp(out, c, (m * 0.6)[..., None])
                else:
                    sel = m > 0.003
                    P = np.stack([np.full_like(Xh, rxw), np.full_like(Xh, Y_BED + yh), Zh], -1)[sel].astype(np.float32)
                    E = self.light_at(P, np.array([0, 1, 0], np.float32))
                    V = np.array([CAM_X, 0.0, 0.0], np.float32) - P
                    V /= np.linalg.norm(V, axis=1, keepdims=True)
                    G = np.zeros_like(E)
                    for (lp, lcol, I, cone, cpow, _) in self.lamps:
                        Ld = np.asarray(lp, np.float32) - P
                        dl = np.linalg.norm(Ld, axis=1, keepdims=True)
                        Hh = V + Ld / dl
                        Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
                        g = np.exp(-(Hh[:, 2] / 0.06) ** 2) * (Hh[:, 1] > 0.2)
                        G += (g * I / (dl[:, 0] ** 2 + 1.0))[:, None] * lcol
                    hi = np.zeros(col.shape, np.float32)
                    zz = Zh[sel][:, None]
                    # thin cool steel sheen reflecting the night sky; warm only where a lamp pool is
                    sky = np.array([0.3, 0.42, 0.8], np.float32) * (0.75 + 0.35 * C.smoothstep(5, 40, zz))
                    Ew = np.clip(E - 0.08, 0, None)
                    hi[sel] = sky * 0.55 + Ew * 1.1 * np.array([1.0, 0.78, 0.55], np.float32) + np.minimum(G * 30.0, 1.6)
                    hi = hi * (1 - fa) + FOG * 1.1 * fa
                    out = C.lerp(out, hi, m[..., None])
        return out

    def _rails(self):
        cam, cv, s = self.cam, self.mid, self.s
        for rxw in RAILS:
            zs = np.geomspace(1.0, 125, 500)
            P = np.stack([np.full_like(zs, rxw), np.full_like(zs, Y_BED + 0.1), zs], 1)
            uv = cam.pts(P)
            E = self.light_at(P.astype(np.float32), np.array([0, 1, 0], np.float32))
            # specular glints of the lamps on the polished rail heads: for a cylinder along z, a glint
            # appears where the half vector between view and light has no z component
            Pf = P.astype(np.float32)
            V = np.array([CAM_X, 0.0, 0.0], np.float32) - Pf
            V /= np.linalg.norm(V, axis=1, keepdims=True)
            G = np.zeros((len(zs), 3), np.float32)
            for (lp, lcol, I, cone, cpow, _) in self.lamps:
                Ld = np.asarray(lp, np.float32) - Pf
                dl = np.linalg.norm(Ld, axis=1, keepdims=True)
                Hh = V + Ld / dl
                Hh /= np.linalg.norm(Hh, axis=1, keepdims=True)
                g = np.exp(-(Hh[:, 2] / 0.07) ** 2) * (Hh[:, 1] > 0.2)
                G += (g * I / (dl[:, 0] ** 2 + 1.0))[:, None] * lcol
            E = E + G * 30.0
            lum = E.mean(-1)
            for i in range(len(zs) - 1):
                z = zs[i]
                wpx = max(cam.f * 0.075 / z, 0.7 * s)
                fa = float(self.fog_amt(z))
                base = np.array([0.025, 0.03, 0.06], np.float32)
                hi = np.array([0.32, 0.46, 0.85], np.float32) * (0.8 + 0.4 * (z > 20)) + E[i] * 1.0
                c = base * (1 - fa) + FOG * fa
                x0, y0, m = L.line_local(uv[i:i + 2], wpx)
                cv.paint(x0, y0, m, c, z)
                # top highlight
                x0, y0, m = L.line_local(uv[i:i + 2] - [0, wpx * 0.25], max(wpx * 0.35, 0.6 * s))
                ch = hi * (1 - fa) + FOG * fa * 1.1
                cv.paint(x0, y0, m * 0.9, ch, z)

    # ------------------------------------------------------------------ mid objects (trees, poles, fence)
    def _mid_objects(self):
        cam, cv, s = self.cam, self.mid, self.s
        rng = np.random.default_rng(12)
        objs = []
        # trees (x, z, height)
        right = [(17, 22, 16.5), (14.8, 25.5, 12.5), (9.8, 41, 9), (33, 46, 18), (14.5, 66, 11), (8.8, 55, 7),
                 (31, 118, 13), (45, 128, 15), (19, 124, 10), (38, 104, 11)]
        left = [(-15, 40, 13), (-19, 47, 16), (-11, 58, 9), (-24, 30, 17), (-30, 62, 14)]
        for (x, z, h) in right + left:
            objs.append(('tree', z, (x, z, h)))
        # utility poles along the right of the track
        self.poles = [(6.9, z) for z in (8.0, 36.0, 64.0, 92.0, 120.0)]
        for (x, z) in self.poles:
            objs.append(('pole', z, (x, z)))
        # village cluster across the tracks
        objs.append(('house', 66, (-27.0, 66.0, 8.0, 6.0, 4.6)))
        for hs in [(14.0, 52.0, 8.0, 6.0, 5.2), (23.5, 61.0, 8.0, 7.0, 6.6), (36.0, 68.0, 8.0, 6.0, 5.0),
                   (18.5, 71.0, 7.0, 6.0, 4.4), (28.0, 79.0, 10.0, 7.0, 6.2), (41.0, 76.0, 8.0, 6.0, 5.2),
                   (21.0, 94.0, 9.0, 7.0, 5.5), (33.0, 98.0, 8.0, 6.0, 4.8)]:
            objs.append(('house', hs[1], hs))
        self.streetlights = [(20.0, 49.0), (30.5, 56.0), (9.5, 70.0), (38.0, 88.0)]
        for (x, z) in self.streetlights:
            objs.append(('slight', z, (x, z)))
        objs.append(('signal', 60.0, (5.5, 60.0)))
        objs.append(('gable', 34, (-17.5, 34.0, 7.0, 7.0, 4.4)))
        objs.append(('crossing', 96, (5.4, 96.0)))
        # the open field right of the track: a second pole line, a half-buried fence and a relay box
        self.poles2 = [(9.8, 19.0), (12.6, 45.0), (15.4, 71.0), (18.2, 97.0)]
        for (x, z) in self.poles2:
            objs.append(('pole', z, (x, z)))
        for z in np.arange(17.0, 90.0, 2.9):
            objs.append(('post', float(z), (12.2 + 0.035 * (z - 17), float(z))))
        objs.append(('relay', 14.5, (7.3, 14.5)))
        objs.sort(key=lambda o: -o[1])
        for kind, z, p in objs:
            cv = self.midf if z >= ZSPLIT else self.mid
            if kind == 'tree':
                self._tree(cv, *p)
            elif kind == 'pole':
                self._pole(cv, *p)
            elif kind == 'post':
                self._fence_post(cv, p[0], p[1], p[1] + 2.9)
            elif kind == 'relay':
                self._relay_box(cv, *p)
            elif kind == 'slight':
                self._streetlight(cv, *p)
            elif kind == 'signal':
                self._signal(cv, *p)
            elif kind == 'crossing':
                self._crossing_post(cv, *p)
            elif kind == 'gable':
                self._house(cv, *p, seed=11, gable_front=True)
            elif kind == 'house':
                self._house(cv, *p, seed=int(p[0] * 13 + p[1]) + 5, gable_front=p[0] in (23.5, 33.0))
        self._wire_geometry()

    def _streetlight(self, cv, x, z):
        cam, s = self.cam, self.s
        fa = float(self.fog_amt(z))
        c = np.array([0.05, 0.055, 0.09], np.float32) * (1 - fa) + FOG * fa
        wpx = max(cam.f * 0.12 / z, 0.8 * s)
        x0, y0, m = L.line_local(cam.pts(np.array([[x, Y_BED, z], [x, Y_BED + 5.0, z], [x - 0.6, Y_BED + 5.2, z]])), wpx)
        cv.paint(x0, y0, m, c, z)

    def _signal(self, cv, x, z):
        cam, s = self.cam, self.s
        fa = float(self.fog_amt(z))
        c = np.array([0.04, 0.045, 0.07], np.float32) * (1 - fa) + FOG * fa
        wpx = max(cam.f * 0.12 / z, 0.8 * s)
        x0, y0, m = L.line_local(cam.pts(np.array([[x, Y_BED, z], [x, Y_BED + 4.3, z]])), wpx)
        cv.paint(x0, y0, m, c, z)
        self.quad(cv, [(x - 0.28, Y_BED + 3.2, z - 0.05), (x + 0.28, Y_BED + 3.2, z - 0.05), (x + 0.28, Y_BED + 4.4, z - 0.05),
                       (x - 0.28, Y_BED + 4.4, z - 0.05)], c * 0.8, z - 0.05)
        self.quad(cv, [(x - 0.32, Y_BED + 4.4, z - 0.1), (x + 0.32, Y_BED + 4.4, z - 0.1), (x + 0.28, Y_BED + 4.55, z - 0.1),
                       (x - 0.28, Y_BED + 4.55, z - 0.1)], AMB_SNOW * 1.3 * (1 - fa) + FOG * fa * 1.1, z - 0.1)

    def _tree(self, cv, x, z, h):
        cam = self.cam
        bu, bv = cam.p(x, Y_BED, z)
        hpx = cam.f * h / z
        fa = float(self.fog_amt(z))
        # warm light from the station lamps on the side facing them
        E = self.light_at(np.array([[x, Y_BED + h * 0.4, z]], np.float32))[0]
        side = -1.0 if x > 0 else 1.0
        wamt = float(np.clip(E.mean() * 6, 0, 0.8))
        FR.fir(cv, bu, bv, hpx, z, abs(int(x * 100 + z * 7)), fol=(0.035, 0.065, 0.11),
               snow_top=(0.64, 0.74, 0.98), snow_shd=(0.19, 0.25, 0.52), warm=E / (E.max() + 1e-6) * 0.6,
               warm_side=side, warm_amt=wamt, fog=FOG, fog_amt=min(fa * 1.35 + 0.08, 0.92), width=0.3 + 0.08 * ((x * 7.3) % 1.0))

    def _pole(self, cv, x, z):
        cam, s = self.cam, self.s
        fa = float(self.fog_amt(z))
        top = 6.6
        c = np.array([0.05, 0.055, 0.09], np.float32) * (1 - fa) + FOG * fa
        wpx = max(cam.f * 0.28 / z, 1.0 * s)
        u0, v0 = cam.p(x, Y_BED, z)
        u1, v1 = cam.p(x, top, z)
        x0, y0, m = L.poly_local(np.array([[u0 - wpx / 2, v0], [u0 + wpx / 2, v0], [u1 + wpx * 0.4, v1],
                                          [u1 - wpx * 0.4, v1]]))
        cv.paint(x0, y0, m, c, z)
        # rim of light on the side facing the station
        E = self.light_at(np.array([[x, 1.0, z]], np.float32))[0]
        x0, y0, m = L.line_local(np.array([[u0 - wpx * 0.35, v0], [u1 - wpx * 0.3, v1]]), max(wpx * 0.25, 0.6 * s))
        cv.paint(x0, y0, m * 0.8, c + E * 0.8 + AMB_SNOW * 0.05, z)
        # cross arms with snow
        for ya, half in ((6.1, 0.9), (5.4, 0.7)):
            pa = cam.pts(np.array([[x - half, ya, z], [x + half, ya, z], [x + half, ya - 0.12, z], [x - half, ya - 0.12, z]]))
            x0, y0, m = L.poly_local(pa)
            cv.paint(x0, y0, m, c, z)
            ps = cam.pts(np.array([[x - half, ya + 0.07, z], [x + half, ya + 0.07, z], [x + half, ya - 0.01, z],
                                   [x - half, ya - 0.01, z]]))
            x0, y0, m = L.poly_local(ps)
            cv.paint(x0, y0, m, (AMB_SNOW * 1.5) * (1 - fa) + FOG * fa * 1.1, z)
        # transformer can on one pole
        if abs(z - 36) < 1:
            pb = cam.pts(np.array([[x + 0.15, 4.3, z], [x + 0.65, 4.3, z], [x + 0.65, 3.3, z], [x + 0.15, 3.3, z]]))
            x0, y0, m = L.poly_local(pb)
            cv.paint(x0, y0, m, c * 1.3, z)
        if abs(z - 64) < 1:
            # street lamp arm (its light is lamp #9 at (6.2, 4.6, 70) -> put it at this pole instead)
            pass

    def _relay_box(self, cv, x, z):
        """Trackside relay cabinet: grey steel box with a thick snow cap overhanging it, a drift banked
        against its side and a tiny green status lamp."""
        cam = self.cam
        w, d, hgt = 0.9, 0.6, 1.25
        x0w, x1w, z0, z1 = x - w / 2, x + w / 2, z - d / 2, z + d / 2
        y0, y1 = Y_BED, Y_BED + hgt
        fa = float(self.fog_amt(z))
        steel = np.array([0.55, 0.58, 0.62], np.float32)
        Ef = self.light_at(np.array([[x, y0 + 0.7, z0 - 0.05]], np.float32), np.array([0, 0, -1], np.float32))[0]
        Es = self.light_at(np.array([[x0w - 0.05, y0 + 0.7, z]], np.float32), np.array([-1, 0, 0], np.float32))[0]

        def fogc(c):
            return c * (1 - fa) + FOG * fa
        # soft cool contact shadow on the snow around the base
        P = np.array([(x0w - 0.5, y0, z0 - 0.35), (x1w + 0.6, y0, z0 - 0.35), (x1w + 0.9, y0, z1 + 0.9), (x0w - 0.2, y0, z1 + 0.9)])
        X0, Y0, m = L.poly_local(cam.pts(P), pad=8)
        m = cv2.GaussianBlur(m, (0, 0), max(cam.f * 0.12 / z, 1.0))
        cv.paint(X0, Y0, m * 0.45, fogc(AMB_SNOW * np.array([0.35, 0.42, 0.75], np.float32)), None)
        # painted-steel faces: grime runs, rust from the snow line, chipped edges, louvres, warning label
        def tex_face(P4, Ew, seed, **kw):
            pts = cam.pts(np.array(P4))
            tw = max(int((pts[:, 0].max() - pts[:, 0].min()) * 3), 8)
            th = max(int((pts[:, 1].max() - pts[:, 1].min()) * 3), 8)
            alb = T2.steel_cabinet(tw, th, seed=seed, **kw)
            vv = np.linspace(0, 1, th, dtype=np.float32)[:, None, None]
            light = AMB_SNOW * (0.6 + 0.6 * vv) + Ew * 0.9
            PT.paste_quad(cv, pts, fogc(alb * light * 1.05), P4[0][2] if P4[0][2] == P4[1][2] else z)
        # side facing the station (-x) - catches the lamp
        tex_face([(x0w, y0, z1), (x0w, y0, z0), (x0w, y1, z0), (x0w, y1, z1)], Es * 1.4, 71, louvres=False, label=False)
        # front (faces camera), lit by bounce from the snow below
        tex_face([(x0w, y0, z0), (x1w, y0, z0), (x1w, y1, z0), (x0w, y1, z0)], Ef, 72)
        # snow drift banked against the base
        n = 24
        xs = np.linspace(x0w - 0.4, x1w + 0.3, n)
        top = np.stack([xs, y0 + 0.28 * np.sin(np.linspace(0, np.pi, n)) ** 0.8 + 0.03 * np.sin(xs * 9), np.full(n, z0 - 0.15)], 1)
        bot = np.stack([xs, np.full(n, y0 - 0.05), np.full(n, z0 - 0.15)], 1)
        X0, Y0, m = L.poly_local(cam.pts(np.concatenate([top, bot[::-1]])))
        vv = np.linspace(1.0, 0.0, m.shape[0], dtype=np.float32)[:, None, None]
        cc = AMB_SNOW * (0.75 + 0.35 * vv) + Ef * (0.3 + 0.5 * vv)
        cv.paint(X0, Y0, m, fogc(cc * np.ones((1, m.shape[1], 1), np.float32)), z0 - 0.15)
        # snow cap: thick rounded pillow overhanging front and side
        n = 30
        xs = np.linspace(x0w - 0.08, x1w + 0.06, n)
        tt = np.linspace(0, 1, n)
        ytop = y1 + 0.05 + 0.17 * np.clip(np.sin(np.pi * tt) * 1.6, 0, 1) ** 0.5 + 0.015 * np.sin(tt * 17)
        ybot = y1 - 0.06 - 0.05 * np.abs(np.sin(tt * 5.0 + 1)) ** 3
        P = np.concatenate([np.stack([xs, ytop, np.full(n, z0 - 0.06)], 1), np.stack([xs[::-1], ybot[::-1], np.full(n, z0 - 0.06)], 1)])
        X0, Y0, m = L.poly_local(cam.pts(P))
        hh = m.shape[0]
        vv = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
        top_c = AMB_SNOW * 1.5 + Ef * 0.4
        und = AMB_SNOW * np.array([0.62, 0.66, 0.95], np.float32) + Ef * 0.3
        kk = C.smoothstep(0.5, 0.54, vv)
        cc = top_c * (1 - kk) + und * kk
        cv.paint(X0, Y0, m, fogc(cc * np.ones((1, m.shape[1], 1), np.float32)), z0 - 0.06)
        # lamp-side rim of the cap
        rim = np.stack([xs[: n // 2], ytop[: n // 2], np.full(n // 2, z0 - 0.07)], 1)
        X0, Y0, m = L.line_local(cam.pts(rim), max(cam.f * 0.018 / z, 0.7 * self.s))
        cv.paint(X0, Y0, m * 0.8, fogc(AMB_SNOW * 1.2 + Ef * 1.4 + np.array([0.25, 0.18, 0.1], np.float32)), z0 - 0.07)
        # status lamp
        u, v = cam.p(x1w - 0.15, y1 - 0.2, z0 - 0.02)
        X0, Y0, m = L.poly_local(np.array([[u - 1.2 * self.s, v - 1.2 * self.s], [u + 1.2 * self.s, v - 1.2 * self.s],
                                           [u + 1.2 * self.s, v + 1.2 * self.s], [u - 1.2 * self.s, v + 1.2 * self.s]]))
        cv.paint(X0, Y0, m, np.array([0.4, 2.2, 1.0], np.float32), z0 - 0.02)

    def _crossing_post(self, cv, x, z):
        cam, s = self.cam, self.s
        fa = float(self.fog_amt(z))
        c = np.array([0.08, 0.08, 0.1], np.float32) * (1 - fa) + FOG * fa
        wpx = max(cam.f * 0.1 / z, 0.8 * s)
        u0, v0 = cam.p(x, Y_BED, z)
        u1, v1 = cam.p(x, Y_BED + 3.2, z)
        x0, y0, m = L.line_local(np.array([[u0, v0], [u1, v1]]), wpx)
        cv.paint(x0, y0, m, c, z)
        # crossbuck (X) and the lamp bar
        for (a0, a1) in (((-0.5, 2.6), (0.5, 3.2)), ((-0.5, 3.2), (0.5, 2.6)), ((-0.45, 2.3), (0.45, 2.3))):
            P = np.array([[x + a0[0], Y_BED + a0[1], z], [x + a1[0], Y_BED + a1[1], z]])
            x0, y0, m = L.line_local(cam.pts(P), max(cam.f * 0.07 / z, 0.7 * s))
            cv.paint(x0, y0, m, c * 1.4 if a0[1] != 2.3 else c, z)

    def _wire_geometry(self):
        """Power-line spans (world points) - drawn per frame so they can sway in the wind."""
        poles = [(6.9, 8.0 - 28.0)] + self.poles
        self.wires = []
        k = 0
        for (xo, ya, sag, wd) in ((-0.8, 6.1, 0.55, 0.018), (0.0, 6.1, 0.5, 0.018), (0.8, 6.1, 0.55, 0.018),
                                  (-0.6, 5.4, 0.7, 0.03), (0.6, 5.4, 0.6, 0.02)):
            for i in range(len(poles) - 1):
                (x0_, z0), (x1_, z1) = poles[i], poles[i + 1]
                t = np.linspace(0, 1, 120)
                X = x0_ + xo + (x1_ - x0_) * t
                Z = z0 + (z1 - z0) * t
                Y = ya - sag * 4 * t * (1 - t)
                ok = Z > 0.6
                if ok.sum() < 2:
                    continue
                bell = 4 * t * (1 - t)
                self.wires.append((X[ok], Y[ok], Z[ok], bell[ok], wd, 0.7 * k + 0.9 * i))
                k += 1
        poles2 = [(7.0, -6.0)] + self.poles2
        for (xo, ya, sag, wd) in ((-0.8, 6.1, 0.6, 0.016), (0.8, 6.1, 0.65, 0.016), (0.0, 5.4, 0.8, 0.024)):
            for i in range(len(poles2) - 1):
                (x0_, z0), (x1_, z1) = poles2[i], poles2[i + 1]
                t = np.linspace(0, 1, 120)
                X = x0_ + xo + (x1_ - x0_) * t
                Z = z0 + (z1 - z0) * t
                Y = ya - sag * 4 * t * (1 - t)
                ok = Z > 0.6
                if ok.sum() < 2:
                    continue
                bell = 4 * t * (1 - t)
                self.wires.append((X[ok], Y[ok], Z[ok], bell[ok], wd, 0.7 * k + 0.9 * i + 2.0))
                k += 1

    def _wire_prep(self):
        """Static occluder mask: mid-layer surfaces nearer than the wire at each pixel."""
        wdep = np.full((self.PH, self.PW), 1e4, np.float32)
        for (X, Y, Z, bell, wd, ph) in sorted(self.wires, key=lambda w: -float(np.median(w[2]))):
            uv = self.cam.pts(np.stack([X, Y, Z], 1))
            for seg in np.array_split(np.arange(len(uv)), 30):
                idx = np.arange(seg[0], min(seg[-1] + 2, len(uv)))
                if len(idx) < 2:
                    continue
                zz = float(Z[idx].mean())
                wpx = max(self.cam.f * wd / zz, 1.0) + 8 * self.s
                cv2.polylines(wdep, [np.round(uv[idx]).astype(np.int32)], False, zz, int(math.ceil(wpx)))
        occ = np.maximum(self.mid.a * (self.mid.z < wdep - 1.0), self.midf.a * (self.midf.z < wdep - 1.0)) * (wdep < 1e3)
        self.wire_occ = cv2.GaussianBlur(occ.astype(np.float32), (0, 0), 0.7)

    def _proj(self, X, Y, Z, tx, ty, tz):
        """Exact projection into the frame for the moving camera."""
        f = self.cam.f
        Zc = np.maximum(np.asarray(Z) - tz, 0.25)
        return ((self.cam.cx - self.mx) + f * (np.asarray(X) - CAM_X - tx) / Zc,
                (self.cam.cy - self.my) - f * (np.asarray(Y) - ty) / Zc)

    def _draw_wires(self, img, t, M, ct=(0.0, 0.0, 0.0)):
        W, H, cam, s = self.W, self.H, self.cam, self.s
        segs = []
        for (X, Y, Z, bell, wd, ph) in self.wires:
            sway = 0.09 * math.sin(1.3 * t + ph) + 0.04 * math.sin(2.9 * t + 1.7 * ph)
            pu, pv = self._proj(X + bell * sway * 0.5, Y + bell * sway, Z, *ct)
            pts = np.stack([pu, pv], 1)
            for seg in np.array_split(np.arange(len(pts)), 12):
                idx = np.arange(seg[0], min(seg[-1] + 2, len(pts)))
                if len(idx) >= 2:
                    segs.append((pts[idx], float(Z[idx].mean()), wd))
        allp = np.concatenate([p for p, _, _ in segs])
        x0 = int(max(np.floor(allp[:, 0].min()) - 8, 0))
        y0 = int(max(np.floor(allp[:, 1].min()) - 8, 0))
        x1 = int(min(np.ceil(allp[:, 0].max()) + 8, W))
        y1 = int(min(np.ceil(allp[:, 1].max()) + 8, H))
        if x1 <= x0 or y1 <= y0:
            return img
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        fg = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for (pp, zz, wd) in segs:
            wpx = max(cam.f * wd / max(zz - 0.3, 0.5) * M[0, 0], 0.8 * s)
            fa = float(self.fog_amt(zz))
            q = np.round((pp - [x0, y0]) * 16).astype(np.int32)
            th = max(1, int(round(wpx)))
            cv2.polylines(m, [q], False, int(255 * min(0.55 + 0.45 * wpx, 1.0)), th, cv2.LINE_AA, shift=4)
            cv2.polylines(fg, [q], False, int(fa * 255), th + 1, cv2.LINE_8, shift=4)
        a = m.astype(np.float32) / 255
        fa = np.clip(fg.astype(np.float32) / 255, 0, 1)[..., None]
        occ = cv2.warpAffine(self.wire_occ, M, (W, H), flags=cv2.INTER_LINEAR)[y0:y1, x0:x1]
        a = (a * (1 - occ))[..., None]
        col = np.array([0.03, 0.04, 0.08], np.float32) * (1 - fa) + FOG * 0.9 * fa
        img = img.copy()
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + col * a
        return img

    def _bloom(self, img, threshold=0.7, knee=0.35, strength=0.5, halation=0.35):
        """bloom_soft equivalent with the bright pass done at half resolution (cheaper at 1080p)."""
        H, W = img.shape[:2]
        small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        lum = small.max(-1, keepdims=True)
        k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
        br = small * np.maximum(k * k, (lum > threshold + knee))
        acc = np.zeros_like(br)
        wts = (1.0, 0.8, 0.6, 0.45)
        for r, wt in zip((0.003, 0.01, 0.03, 0.08), wts):
            acc += F.fast_blur(br, r * W / 2) * wt
        acc = acc * (strength / sum(wts))
        hal = F.fast_blur(br, 0.006 * W / 2) * (np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5)
        add = cv2.resize(acc + hal, (W, H), interpolation=cv2.INTER_LINEAR)
        return img + add

    def _fence_post(self, cv, x, z, zn=None):
        cam, s = self.cam, self.s
        fa = float(self.fog_amt(z))
        c = np.array([0.06, 0.055, 0.08], np.float32) * (1 - fa) + FOG * fa
        if zn is not None:
            for yw in (0.85, 0.45):
                P = np.array([[x, Y_BED + yw, zn], [x, Y_BED + yw - 0.02, (z + zn) / 2], [x, Y_BED + yw, z]])
                faw = float(self.fog_amt((z + zn) / 2))
                cw = np.array([0.05, 0.05, 0.08], np.float32) * (1 - faw) + FOG * faw
                x0, y0, m = L.line_local(cam.pts(P), max(cam.f * 0.008 / z, 0.6 * s))
                cv.paint(x0, y0, m * 0.8, cw, (z + zn) / 2)
        w = 0.035
        pts = cam.pts(np.array([[x - w, Y_BED, z], [x + w, Y_BED, z], [x + w * 0.8, Y_BED + 0.95, z], [x - w * 0.8, Y_BED + 0.95, z]]))
        x0, y0, m = L.poly_local(pts)
        cv.paint(x0, y0, m, c, z)
        cap = cam.pts(np.array([[x - w * 1.4, Y_BED + 0.93, z], [x + w * 1.4, Y_BED + 0.93, z],
                                [x + w * 0.9, Y_BED + 1.04, z], [x - w * 0.9, Y_BED + 1.04, z]]))
        x0, y0, m = L.poly_local(cap)
        E = self.light_at(np.array([[x, Y_BED + 1.0, z]], np.float32), np.array([0, 1, 0], np.float32))[0]
        cv.paint(x0, y0, m, snow_lit(AMB_SNOW * 1.2, E[None])[0] * (1 - fa) + FOG * fa * 1.1, z)

    # ------------------------------------------------------------------ station
    def _station(self):
        cam, cv, s = self.cam, self.mid, self.s
        # far lamp posts (beyond canopy), back to front
        for z in (58.0, 47.0, 36.0):
            self._lamp_post(self.midf if z >= ZSPLIT else cv, z)
        self._lamp_post(cv, 13.8, x=-5.75, d=1.0)            # second platform lamp, left side
        self._fence_left(cv)
        self._platform_end()
        self._building(cv)
        self._canopy(cv)
        self._bench_fg(self.fgs['bench'])

    def _lamp_post(self, cv, z, x=2.05, d=-1.0):
        cam, s = self.cam, self.s
        fa = float(self.fog_amt(z))
        dark = np.array([0.05, 0.055, 0.085], np.float32)
        c = dark * (1 - fa) + FOG * fa
        w = 0.07
        pts = cam.pts(np.array([[x - w, Y_PLAT, z], [x + w, Y_PLAT, z], [x + w * 0.7, 2.45, z], [x - w * 0.7, 2.45, z]]))
        x0, y0, m = L.poly_local(pts)
        cv.paint(x0, y0, m, c, z)
        # lit edge facing the lamp above / camera
        E = self.light_at(np.array([[x + d * 0.1, -0.5, z - 0.1]], np.float32))[0]
        e = cam.pts(np.array([[x + d * w * 0.8, Y_PLAT, z], [x + d * w * 0.6, 2.2, z]]))
        x0, y0, m = L.line_local(e, max(cam.f * 0.018 / z, 0.6 * s))
        cv.paint(x0, y0, m * 0.6, c + np.minimum(E, 0.8) * 0.12, z)
        # arm
        arm = np.array([[x, 2.45, z], [x + d * 0.15, 2.52, z], [x + d * 0.4, 2.5, z], [x + d * 0.6, 2.4, z]])
        x0, y0, m = L.line_local(cam.pts(arm), max(cam.f * 0.05 / z, 0.8 * s))
        cv.paint(x0, y0, m, c, z)
        # shade: dark enamel cone seen from below, glowing opening, snow cap on top
        lx, ly = x + d * 0.6, 2.2
        shade = np.array([[lx - 0.3, ly, z], [lx - 0.08, ly + 0.17, z], [lx + 0.08, ly + 0.17, z], [lx + 0.3, ly, z]])
        x0, y0, m = L.poly_local(cam.pts(shade))
        cv.paint(x0, y0, m, np.array([0.05, 0.07, 0.07], np.float32) * (1 - fa) + FOG * fa, z)
        cap = np.array([[lx - 0.22, ly + 0.07, z], [lx - 0.12, ly + 0.2, z], [lx, ly + 0.24, z],
                        [lx + 0.12, ly + 0.2, z], [lx + 0.22, ly + 0.07, z], [lx + 0.08, ly + 0.14, z], [lx - 0.08, ly + 0.14, z]])
        x0, y0, m = L.poly_local(cam.pts(cap))
        cv.paint(x0, y0, m, (AMB_SNOW * 1.5) * (1 - fa) + FOG * fa * 1.2, z)
        # opening ellipse (inner surface lit by the bulb) and the bulb
        uo, vo = cam.p(lx, ly, z)
        rxp = cam.f * 0.21 / z
        ryp = max(rxp * (ly / math.hypot(ly, z)) * 1.1, 1.2 * s)
        ss = 4
        wbox, hbox = int(rxp * 2 + 6), int(ryp * 2 + 6)
        mm = np.zeros((hbox * ss, wbox * ss), np.uint8)
        cv2.ellipse(mm, (int(wbox * ss / 2), int(hbox * ss / 2)), (int(rxp * ss), int(ryp * ss)), 0, 0, 360, 255, -1, cv2.LINE_AA)
        mm = cv2.resize(mm.astype(np.float32) / 255, (wbox, hbox), interpolation=cv2.INTER_AREA)
        yy, xx = np.mgrid[0:hbox, 0:wbox].astype(np.float32)
        d = np.sqrt(((xx - wbox / 2) / max(rxp, 1)) ** 2 + ((yy - hbox / 2) / max(ryp, 1)) ** 2)
        inner = np.array([1.1, 0.62, 0.28], np.float32) + np.array([3.2, 2.6, 1.9], np.float32) * np.exp(-d * d * 6)[..., None]
        cv.paint(int(uo - wbox / 2), int(vo - hbox / 2), mm, inner * (1 - fa * 0.6), z)
    def _platform_end(self):
        """Railing across the far end of the platform with snow on its top rail (seen through the canopy)."""
        cam, s = self.cam, self.s
        cv = self.midf
        z = PLAT_Z1 - 0.3
        fa = float(self.fog_amt(z))
        c = np.array([0.06, 0.06, 0.09], np.float32) * (1 - fa) + FOG * fa
        E = self.light_at(np.array([[0.0, Y_PLAT + 1.0, z - 0.2]], np.float32), np.array([0, 0, -1], np.float32))[0]
        for xp in np.linspace(PLAT_X0 + 0.2, PLAT_X1 - 0.2, 7):
            self.quad(cv, [(xp - 0.05, Y_PLAT, z), (xp + 0.05, Y_PLAT, z), (xp + 0.05, Y_PLAT + 1.1, z), (xp - 0.05, Y_PLAT + 1.1, z)], c, z)
        for yr in (Y_PLAT + 1.05, Y_PLAT + 0.55):
            self.quad(cv, [(PLAT_X0, yr - 0.06, z), (PLAT_X1, yr - 0.06, z), (PLAT_X1, yr, z), (PLAT_X0, yr, z)], c, z)
        sn = (AMB_SNOW * 1.25 + E * 0.8) * (1 - fa) + FOG * fa * 1.1
        self.quad(cv, [(PLAT_X0, Y_PLAT + 1.05, z - 0.02), (PLAT_X1, Y_PLAT + 1.05, z - 0.02), (PLAT_X1, Y_PLAT + 1.13, z - 0.02),
                       (PLAT_X0, Y_PLAT + 1.12, z - 0.02)], sn, z - 0.02)

    def _fence_left(self, cv):
        cam, s = self.cam, self.s
        x = PLAT_X0 + 0.15
        for z in np.arange(4.0, PLAT_Z1, 2.2):
            fa = float(self.fog_amt(z))
            c = np.array([0.07, 0.06, 0.08], np.float32) * (1 - fa) + FOG * fa
            pts = cam.pts(np.array([[x - 0.06, Y_PLAT, z], [x + 0.06, Y_PLAT, z], [x + 0.06, Y_PLAT + 1.1, z],
                                    [x - 0.06, Y_PLAT + 1.1, z]]))
            x0, y0, m = L.poly_local(pts)
            cv.paint(x0, y0, m, c, z)
        for yr, th in ((Y_PLAT + 1.05, 0.08), (Y_PLAT + 0.55, 0.06)):
            zs = np.geomspace(4.0, PLAT_Z1, 90)
            for i in range(len(zs) - 1):
                z = zs[i]
                fa = float(self.fog_amt(z))
                c = np.array([0.07, 0.06, 0.08], np.float32) * (1 - fa) + FOG * fa
                P = np.array([[x, yr, zs[i]], [x, yr, zs[i + 1]], [x, yr - th, zs[i + 1]], [x, yr - th, zs[i]]])
                x0, y0, m = L.poly_local(cam.pts(P))
                cv.paint(x0, y0, m, c, z)
                if yr > Y_PLAT + 1:
                    P = np.array([[x, yr + 0.07, zs[i]], [x, yr + 0.07, zs[i + 1]], [x, yr, zs[i + 1]], [x, yr, zs[i]]])
                    x0, y0, m = L.poly_local(cam.pts(P))
                    E = self.light_at(np.array([[x, yr, z]], np.float32), np.array([0, 1, 0], np.float32))[0]
                    cv.paint(x0, y0, m, (AMB_SNOW * 1.3 + E) * (1 - fa) + FOG * fa * 1.1, z)

    def _plane_region(self, axis, val, bounds):
        """Raycast an axis-aligned plane (axis 'x' or 'y' or 'z' = val); returns mask, X, Y, Z."""
        rx, ry = self.rx, self.ry
        if axis == 'x':
            Z = (val - CAM_X) / np.where(np.abs(rx) < 1e-6, 1e-6, rx)
        elif axis == 'y':
            Z = val / np.where(np.abs(ry) < 1e-6, 1e-6, ry)
        else:
            Z = np.full_like(rx, val)
        X, Y = rx * Z + CAM_X, ry * Z
        (x0, x1), (y0, y1), (z0, z1) = bounds
        m = (Z > 0.1) & (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
        return m, X, Y, Z

    def _paint_region(self, cv, m, col, Z, ss_blur=0.0):
        """Paint a raycast region with a soft 1px anti-aliased edge (mask blurred slightly)."""
        a = m.astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 0.6)
        a = np.where(m, np.maximum(a, 0.5), a)
        ys, xs = np.nonzero(a > 0.002)
        if len(ys) == 0:
            return
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cc = col[y0:y1, x0:x1] if (isinstance(col, np.ndarray) and col.ndim == 3) else col
        cv.paint(x0, y0, a[y0:y1, x0:x1], cc, Z[y0:y1, x0:x1] if isinstance(Z, np.ndarray) else Z)

    def _hand_paint_wall(self, col, m, pool, shade_col, sig=7.0):
        """Hand-painted wall (round 5): outside the warm light pool the wall is ONE flat cool shade value
        (texture suppressed); the plank / brick detail is only allowed to read inside the pool."""
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return col
        pad = int(4 * sig * self.s) + 2
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, m.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, m.shape[1])
        mf = m[y0:y1, x0:x1].astype(np.float32)
        c = np.nan_to_num(col[y0:y1, x0:x1]).astype(np.float32)
        sg = sig * self.s
        den = np.maximum(cv2.GaussianBlur(mf, (0, 0), sg), 1e-3)[..., None]
        cs = cv2.GaussianBlur(c * mf[..., None], (0, 0), sg) / den
        pk = np.clip(pool[y0:y1, x0:x1], 0, 1)[..., None]
        lit = cs + (c - cs) * (0.35 + 0.65 * pk)
        # (cycle 7) the cool shade side is no longer a dead flat fill: it keeps the painted board /
        # grain / streak pattern as a relative value modulation, and is lifted a little so it reads
        ratio = np.clip(c.mean(-1, keepdims=True) / np.maximum(cs.mean(-1, keepdims=True), 1e-4), 0.35, 1.8)
        shade = np.asarray(shade_col, np.float32) * 1.55 * (1 + 0.75 * (ratio - 1))
        # a broad warm spill gradient bridges the pool and the shade (no hard CG falloff)
        spill = np.clip(cv2.GaussianBlur(pk[..., 0], (0, 0), 38 * self.s + 1) * 1.6, 0, 1)[..., None]
        shade = shade * (1 - 0.6 * spill) + shade * np.array([2.1, 1.45, 0.85], np.float32) * 0.6 * spill
        out = lit * pk + shade * (1 - pk)
        col = col.copy()
        col[y0:y1, x0:x1] = np.where(mf[..., None] > 0, out, c)
        return col

    def _building(self, cv):
        cam, s = self.cam, self.s
        # side wall x = BLD_X1 (faces +x, toward the track)
        m, X, Y, Z = self._plane_region('x', BLD_X1, ((-99, 99), (Y_PLAT, CAN_Y), (BLD_Z0, BLD_Z1)))
        P = np.stack([np.full_like(X, BLD_X1), Y, Z], -1)
        E = paint_E(self.light_at(P, np.array([1, 0, 0], np.float32)), step=2.0, soft=0.12, mix=0.75)
        tex = self._snow_tex(Z * 3, Y * 0.2, 9, 0.05)
        plank = self._clapboard(Y, Z)
        streak = self._snow_tex(Z * 7.0, Y * 0.35, 23, 0.05)
        stain = self._snow_tex(Z * 0.8, Y * 0.8, 24, 0.05)
        wood = np.array([0.3, 0.2, 0.14], np.float32) * (0.85 + 0.3 * tex[..., None]) * plank[..., None]
        wood = wood * (0.82 + 0.3 * streak[..., None]) * (0.9 + 0.2 * stain[..., None])
        # painted wood grain, butt joints, knots + rain-streak grime under the windows / eave, splash dirt
        ppm = cam.f / np.maximum(Z, 1.0)
        wood = wood * TX.wood(Z, Y - Y_PLAT, 0.19, ppm, seed=31)[..., None]
        wood = wood * TX.grime(Z, Y - Y_PLAT, CAN_Y - Y_PLAT, 0.0, seed=32,
                               openings=[(19.2, 21.6, 0.92), (22.4, 24.8, 0.92), (26.0, 28.2, 0.92)])[..., None]
        # darker weathered skirt where melt water splashes, cool bounce from the snow
        wood = wood * (1 - 0.25 * C.smoothstep(Y_PLAT + 0.45, Y_PLAT, Y))[..., None]
        wm_, ws_ = R4.wall_weather(Z, Y - Y_PLAT, CAN_Y - Y_PLAT, ppm, seed=61)
        wood = R4.apply_weather(wood * 1.45, wm_, ws_)
        bounce = C.smoothstep(Y_PLAT + 1.6, Y_PLAT, Y)[..., None]
        col = wood * (AMB_SNOW * (0.95 + 0.7 * bounce) + E * 0.9)
        # round 5: warm light pool that falls off from the lit windows and the lamps, flat cool shade
        glow = np.zeros_like(Z)
        for (za, zb) in ((19.2, 21.6), (22.4, 24.8), (26.0, 28.2)):
            dz = np.maximum(np.maximum(za - Z, Z - zb), 0)
            dy = np.maximum(np.maximum(Y_PLAT + 1.0 - Y, Y - (Y_PLAT + 2.3)), 0)
            glow = np.maximum(glow, np.exp(-(np.sqrt(dz ** 2 + dy ** 2 * 2.2) / 0.42) ** 1.4))
        jit = 0.06 * np.sin(Y * 6.0 + Z * 2.3)
        pool = C.smoothstep(0.25, 0.6, np.clip(C.smoothstep(0.3, 1.4, E.mean(-1)) + 0.9 * glow + jit, 0, 1.5))
        col = col + wood * WIN[None, None] * (0.35 * glow)[..., None]
        base_w = np.array([0.3, 0.2, 0.14], np.float32) * 1.45
        shade_c = base_w * AMB_SNOW * 0.9 + np.array([0.004, 0.008, 0.03], np.float32)
        col = self._hand_paint_wall(col, m, pool, shade_c)
        # windows (warm, frosted, curtains)
        for (za, zb) in ((19.2, 21.6), (22.4, 24.8), (26.0, 28.2)):
            wm = (Z > za) & (Z < zb) & (Y > Y_PLAT + 1.0) & (Y < Y_PLAT + 2.3)
            fr = (Z > za - 0.08) & (Z < zb + 0.08) & (Y > Y_PLAT + 0.92) & (Y < Y_PLAT + 2.38)
            gy = (Y - (Y_PLAT + 1.0)) / 1.3
            glass = self._interior(Z) * (0.92 + 0.12 * tex[..., None])
            # curtains tied back at both sides of each window (folds, lit from inside)
            cz = (Z - za) / (zb - za)
            cw_ = 0.16 + 0.04 * np.sin(gy * 5.0 + za)
            curt = (cz < cw_) | (cz > 1 - cw_)
            fold = 0.72 + 0.28 * np.cos(Z / 0.045 * np.pi) ** 2
            ccol = np.array([1.05, 0.55, 0.3], np.float32) * fold[..., None] * (0.7 + 0.4 * gy[..., None])
            glass = np.where(curt[..., None], ccol, glass)
            # frost creeping in from the lower corners of the panes
            frost = np.clip(1 - gy / 0.3 - 0.8 * np.minimum(cz, 1 - cz), 0, 1) ** 1.3 * (0.7 + 0.6 * tex)
            glass = glass * (1 - 0.55 * frost[..., None]) + frost[..., None] * np.array([0.62, 0.66, 0.8], np.float32) * 0.8
            glass = glass + C.smoothstep(0.3, 0.0, gy)[..., None] * np.array([0.12, 0.12, 0.16], np.float32)
            # mullions: sash grid (3 x 2 panes)
            zw = (zb - za)
            mul = (np.abs(Y - (Y_PLAT + 1.65)) < 0.03)
            for q in (1, 2):
                mul = mul | (np.abs(Z - (za + zw * q / 3)) < 0.035)
            glass = np.where(mul[..., None], np.array([0.12, 0.08, 0.06], np.float32), glass)
            col = np.where(fr[..., None], np.array([0.1, 0.07, 0.06], np.float32), col)
            col = np.where(wm[..., None], glass, col)
            # snow on the sill + lit sill edge
            sill = (Z > za - 0.12) & (Z < zb + 0.12) & (Y > Y_PLAT + 0.86) & (Y < Y_PLAT + 0.94)
            col = np.where(sill[..., None], AMB_SNOW * 1.2 + E * 0.9, col)
        # timetable board and travel posters between the windows (real printed boards)
        lt = AMB_SNOW * 0.35 + E * 1.1 + 0.12
        if not hasattr(self, '_boards'):
            self._boards = dict(tt=PT.timetable(90, 150), p1=PT.poster(110, 160, 0), p2=PT.poster(110, 160, 2))
        for key, (za_, zb_, ya_, yb_) in (('tt', (21.72, 22.28, Y_PLAT + 1.0, Y_PLAT + 1.95)),
                                          ('p1', (25.05, 25.75, Y_PLAT + 1.05, Y_PLAT + 2.05)),
                                          ('p2', (18.35, 18.95, Y_PLAT + 1.0, Y_PLAT + 1.85))):
            tex_b = self._boards[key]
            th_, tw_ = tex_b.shape[:2]
            inb = (Z > za_) & (Z < zb_) & (Y > ya_) & (Y < yb_)
            frm = (Z > za_ - 0.035) & (Z < zb_ + 0.035) & (Y > ya_ - 0.035) & (Y < yb_ + 0.035)
            mu = ((Z - za_) / (zb_ - za_) * tw_).astype(np.float32)
            mv = ((yb_ - Y) / (yb_ - ya_) * th_).astype(np.float32)
            smp = cv2.remap(tex_b, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            col = np.where(frm[..., None], np.array([0.08, 0.06, 0.05], np.float32) * (AMB_SNOW + E), col)
            col = np.where(inb[..., None], smp * lt, col)
        # the vending machine blocks part of the wall -> drawn after
        fa = self.fog_amt(Z)[..., None]
        col = np.nan_to_num(col * (1 - fa) + FOG * fa)
        self._paint_region(cv, m, col, Z)
        # front wall z = BLD_Z0 (faces the camera) with a lit sliding door
        m, X, Y, Z = self._plane_region('z', BLD_Z0, ((CAN_X0, BLD_X1), (Y_PLAT, CAN_Y), (0, 99)))
        P = np.stack([X, Y, Z], -1)
        E = paint_E(self.light_at(P, np.array([0, 0, -1], np.float32)), step=2.0, soft=0.12, mix=0.75)
        tex = self._snow_tex(X * 3, Y * 0.3, 10, 0.05)
        bounce = C.smoothstep(Y_PLAT + 1.6, Y_PLAT, Y)[..., None]
        streak = self._snow_tex(X * 7.0, Y * 0.35, 25, 0.05)
        col = np.array([0.3, 0.21, 0.14], np.float32) * (0.85 + 0.3 * tex[..., None]) * (AMB_SNOW * (0.5 + 0.6 * bounce) + E)
        col = col * (0.82 + 0.3 * streak[..., None])
        col = col * self._clapboard(Y, Z)[..., None]
        ppm = cam.f / np.maximum(Z, 1.0)
        col = col * TX.wood(X, Y - Y_PLAT, 0.19, ppm, seed=33)[..., None]
        col = col * TX.grime(X, Y - Y_PLAT, CAN_Y - Y_PLAT, 0.0, seed=34, openings=[(-5.1, -3.7, 2.1), (-5.0, -3.8, 2.22), (-3.66, -3.36, 1.02)])[..., None]
        wm_, ws_ = R4.wall_weather(X * 1.3 + 3.0, Y - Y_PLAT, CAN_Y - Y_PLAT, ppm, seed=62)
        col = R4.apply_weather(col * 2.3, wm_, ws_) + np.array([0.014, 0.014, 0.034], np.float32)
        # hand-painted warm/cool split: the left platform lamp throws a warm, soft-edged oval pool low
        # across the facade; above and to the right the wall stays in one cool blue shade value
        jit = 0.08 * np.sin(Y * 7.0 + X * 3.0)
        dpool = np.sqrt(((X + 5.2) / 1.9) ** 2 + ((Y - (Y_PLAT + 0.55)) / 1.25) ** 2) + jit
        pool = C.smoothstep(1.0, 0.82, dpool)
        shade = 1 - pool
        col = col * (1 + pool[..., None] * np.array([1.5, 0.95, 0.35], np.float32))             + shade[..., None] * np.array([0.0, 0.006, 0.02], np.float32)
        # round 5: brick / plank detail only inside the lamp pool + door spill; one flat cool value elsewhere
        dd_ = np.sqrt((np.maximum(np.maximum(-5.1 - X, X + 3.7), 0)) ** 2 + np.maximum(Y - (Y_PLAT + 2.05), 0) ** 2)
        pool2 = np.clip(pool + 0.8 * np.exp(-(dd_ / 0.5) ** 1.4), 0, 1)
        base_f = np.array([0.3, 0.21, 0.14], np.float32) * 2.3
        col = self._hand_paint_wall(col, m, C.smoothstep(0.15, 0.6, pool2),
                                    base_f * AMB_SNOW * 0.55 + np.array([0.014, 0.016, 0.04], np.float32))
        door = (X > -5.1) & (X < -3.7) & (Y < Y_PLAT + 2.05)
        dg = (X > -5.0) & (X < -3.8) & (Y < Y_PLAT + 1.95) & (Y > Y_PLAT + 0.15)
        col = np.where(door[..., None], np.array([0.12, 0.09, 0.07], np.float32), col)
        gy = (Y - Y_PLAT) / 2.0
        inner = self._interior(Z)
        frame_ = (np.abs(X - (-4.4)) < 0.035) | (np.abs(Y - (Y_PLAT + 0.95)) < 0.03)
        inner = np.where(frame_[..., None], np.array([0.1, 0.07, 0.05], np.float32), inner)
        inner = inner + C.smoothstep(0.25, 0.05, gy)[..., None] * np.array([0.2, 0.18, 0.22], np.float32)
        col = np.where(dg[..., None], inner, col)
        # little sign above the door
        sg = (X > -5.0) & (X < -3.8) & (Y > Y_PLAT + 2.25) & (Y < Y_PLAT + 2.6)
        stex = PT.text_board(240, 70, [('待 合 室', 0.52, 0.62, (24, 26, 34), True)], bg=(236, 234, 226))
        mu = ((X + 5.0) / 1.2 * stex.shape[1]).astype(np.float32)
        mv = ((Y_PLAT + 2.6 - Y) / 0.35 * stex.shape[0]).astype(np.float32)
        smp = cv2.remap(stex, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        col = np.where(sg[..., None], smp * (AMB_SNOW * 0.45 + E * 1.2 + 0.1), col)
        # a small lit route-map board right of the door
        rm = (X > -3.66) & (X < -3.36) & (Y > Y_PLAT + 1.05) & (Y < Y_PLAT + 1.75)
        rtex = PT.poster(70, 150, 1)
        mu = ((X + 3.66) / 0.3 * rtex.shape[1]).astype(np.float32)
        mv = ((Y_PLAT + 1.75 - Y) / 0.7 * rtex.shape[0]).astype(np.float32)
        smp = cv2.remap(rtex, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        col = np.where(rm[..., None], smp * (AMB_SNOW * 0.45 + E * 1.2 + 0.1), col)
        self._paint_region(cv, m, col, Z)
        # benches along the side wall, bins by the canopy entrance
        self._bench(cv)
        self._bench(cv, 25.9, 27.9)
        self._vending(cv)
        self._bins(cv)

    def _interior(self, Zs):
        """Interior mapping of the lit waiting room seen through windows / the door: continue each view ray
        past the glass (depth Zs) into the room box and paint its walls, floor, ceiling light, bench,
        posters and the stove."""
        rx, ry = self.rx, self.ry
        big = 1e9
        with np.errstate(divide='ignore', invalid='ignore'):
            zb = np.where(rx < -1e-6, (CAN_X0 + 0.1 - CAM_X) / rx, big)          # back wall (left)
            zr = np.where(rx > 1e-6, (BLD_X1 - CAM_X) / rx, big)                  # right wall inner
            zf = np.where(ry < -1e-6, Y_PLAT / ry, big)
            zc = np.where(ry > 1e-6, (CAN_Y - 0.05) / ry, big)
        ze = np.full_like(rx, BLD_Z1)
        cands = np.stack([zb, zr, zf, zc, ze], 0)
        cands = np.where(cands > Zs[None] + 1e-3, cands, big)
        k = np.argmin(cands, 0)
        Zi = np.min(cands, 0)
        Xi = rx * Zi + CAM_X
        Yi = ry * Zi
        wall_hi = np.array([1.25, 0.95, 0.62], np.float32)
        wall_lo = np.array([0.55, 0.32, 0.17], np.float32)
        col = np.where((Yi < Y_PLAT + 0.9)[..., None], wall_lo, wall_hi)
        # posters / timetable on the back wall
        post = (k == 0) & (((Zi > 20.0) & (Zi < 20.7)) | ((Zi > 25.2) & (Zi < 26.4))) & (Yi > Y_PLAT + 1.2) & (Yi < Y_PLAT + 2.0)
        col = np.where(post[..., None], np.array([1.4, 1.35, 1.25], np.float32), col)
        post2 = (k == 0) & (Zi > 22.2) & (Zi < 22.8) & (Yi > Y_PLAT + 1.3) & (Yi < Y_PLAT + 2.1)
        col = np.where(post2[..., None], np.array([0.9, 0.4, 0.35], np.float32), col)
        # bench along the back wall
        bench = (k == 0) & (Yi < Y_PLAT + 0.5) & (Yi > Y_PLAT + 0.38)
        col = np.where(bench[..., None], np.array([0.35, 0.18, 0.08], np.float32), col)
        # floor / ceiling
        col = np.where((k == 2)[..., None], np.array([0.45, 0.27, 0.13], np.float32), col)
        ceil = (k == 3)
        lamp = np.exp(-((Xi + 4.6) / 0.18) ** 2)
        col = np.where(ceil[..., None], np.array([1.1, 0.9, 0.65], np.float32) + lamp[..., None] * 3.0, col)
        # the stove: dark body with an orange window, pipe up to the ceiling
        stove = (np.abs(Zi - 24.0) < 0.3) & (k == 0) & (Yi < Y_PLAT + 0.8)
        col = np.where(stove[..., None], np.array([0.08, 0.06, 0.05], np.float32), col)
        pipe = (np.abs(Zi - 24.0) < 0.07) & (k == 0)
        col = np.where(pipe[..., None], np.array([0.1, 0.08, 0.07], np.float32), col)
        fire = (np.abs(Zi - 24.0) < 0.14) & (k == 0) & (Yi > Y_PLAT + 0.3) & (Yi < Y_PLAT + 0.5)
        col = np.where(fire[..., None], np.array([2.4, 0.9, 0.2], np.float32), col)
        # soft light falloff toward the room corners
        vign = 1.0 - 0.25 * np.clip(np.abs(Yi - (Y_PLAT + 1.5)) / 1.5, 0, 1)
        col = col * vign[..., None]
        # hanging pendant lamps (enamel shade + glowing bulb) on cords, ray-traced as spheres
        for pz in (20.4, 23.6, 27.1):
            pcx, pcy, pr = -4.55, Y_PLAT + 2.25, 0.2
            ox = CAM_X - pcx
            a_ = rx * rx + ry * ry + 1
            b_ = 2 * (rx * ox + ry * (-pcy) + (-pz))
            c_ = ox * ox + pcy * pcy + pz * pz - pr * pr
            disc = b_ * b_ - 4 * a_ * c_
            hit = disc > 0
            Zh = (-b_ - np.sqrt(np.maximum(disc, 0))) / (2 * a_)
            hit = hit & (Zh > Zs) & (Zh < Zi)
            Yh = ry * Zh
            shade = hit & (Yh > pcy - 0.03)
            bulb = hit & ~shade
            sh_c = np.array([0.1, 0.22, 0.16], np.float32) * (1.0 + 2.5 * C.smoothstep(pcy + 0.05, pcy - 0.03, Yh))[..., None]
            col = np.where(shade[..., None], sh_c, col)
            col = np.where(bulb[..., None], np.array([3.2, 2.6, 1.7], np.float32), col)
            # cord
            Xc = rx * pz + CAM_X
            Yc = ry * pz
            cord = (np.abs(Xc - pcx) < 0.012) & (Yc > pcy) & (Yc < CAN_Y) & (Zs < pz)
            col = np.where(cord[..., None], np.array([0.08, 0.06, 0.05], np.float32), col)
            # warm hot spot on the ceiling/walls around each pendant
            d2 = (Xi - pcx) ** 2 + (Yi - pcy) ** 2 + (Zi - pz) ** 2
            col = col + (0.35 / (d2 + 0.4))[..., None] * np.array([0.6, 0.4, 0.2], np.float32) * (~hit)[..., None]
        return col.astype(np.float32)

    def _paste(self, cv, P4, tex, z):
        """World quad (bl, br, tr, tl) textured with tex (top row = tl..tr)."""
        tex = np.ascontiguousarray(tex, np.float32)
        PT.paste_quad(cv, self.cam.pts(np.array(P4, np.float64)), tex, z)

    def _clapboard(self, Y, Z):
        """Horizontal clapboard siding: each board lit on its lower lip, dark shadow line under it."""
        per = 0.19
        fp = np.clip(np.abs(Z) / self.cam.f / per * 1.5, 0.01, 0.5)
        ph = np.mod((Y - Y_PLAT) / per, 1.0)
        shadow = C.smoothstep(0.12 + fp, 0.0, ph) * (1 - C.smoothstep(0.2, 0.45, fp))
        lip = C.smoothstep(0.14, 0.16 + fp, ph) * C.smoothstep(0.34 + fp, 0.2, ph) * (1 - C.smoothstep(0.2, 0.45, fp))
        grad = 0.85 + 0.25 * ph
        return (grad * (1 - 0.7 * shadow) * (1 + 0.6 * lip)).astype(np.float32)

    def _bins(self, cv):
        """Two station rubbish bins (blue / grey) with snow caps, just outside the canopy."""
        for k, (xa, colb) in enumerate(((1.55, np.array([0.16, 0.3, 0.55], np.float32)), (1.05, np.array([0.35, 0.36, 0.4], np.float32)))):
            z0 = 15.3
            xb, hgt, dp = xa + 0.42, 0.85, 0.42
            E = self.light_at(np.array([[xa + 0.2, Y_PLAT + 0.5, z0 - 0.1]], np.float32), np.array([0, 0, -1], np.float32))[0]
            Es = self.light_at(np.array([[xa, Y_PLAT + 0.5, z0 + 0.2]], np.float32), np.array([-1, 0, 0], np.float32))[0]
            self.quad(cv, [(xa, Y_PLAT, z0), (xa, Y_PLAT, z0 + dp), (xa, Y_PLAT + hgt, z0 + dp), (xa, Y_PLAT + hgt, z0)],
                      colb * (AMB_SNOW * 0.35 + Es * 0.9) * 0.8, z0 + dp / 2)
            self.quad(cv, [(xa, Y_PLAT, z0), (xb, Y_PLAT, z0), (xb, Y_PLAT + hgt, z0), (xa, Y_PLAT + hgt, z0)],
                      colb * (AMB_SNOW * 0.4 + E * 1.0), z0)
            # slot + label
            self.quad(cv, [(xa + 0.08, Y_PLAT + hgt - 0.22, z0 - 0.01), (xb - 0.08, Y_PLAT + hgt - 0.22, z0 - 0.01),
                           (xb - 0.08, Y_PLAT + hgt - 0.12, z0 - 0.01), (xa + 0.08, Y_PLAT + hgt - 0.12, z0 - 0.01)],
                      np.array([0.02, 0.02, 0.03], np.float32), z0 - 0.01)
            self.quad(cv, [(xa + 0.1, Y_PLAT + 0.35, z0 - 0.01), (xb - 0.1, Y_PLAT + 0.35, z0 - 0.01),
                           (xb - 0.1, Y_PLAT + 0.5, z0 - 0.01), (xa + 0.1, Y_PLAT + 0.5, z0 - 0.01)],
                      np.array([0.9, 0.9, 0.85], np.float32) * (AMB_SNOW * 0.4 + E * 1.0), z0 - 0.01)
            self._drift_along(cv, xa - 0.06, z0 - 0.04, xb + 0.04, z0 - 0.04, 0.1, 70 + k)
            # snow cap
            n = 16
            xs = np.linspace(xa - 0.03, xb + 0.03, n)
            top = np.stack([xs, Y_PLAT + hgt + 0.1 * np.sin(np.linspace(0, np.pi, n)) ** 0.6, np.full(n, z0 - 0.02)], 1)
            bot = np.stack([xs, np.full(n, Y_PLAT + hgt - 0.03), np.full(n, z0 - 0.02)], 1)
            x0, y0, mm = L.poly_local(self.cam.pts(np.concatenate([top, bot[::-1]])))
            vv = np.where(np.linspace(0, 1, mm.shape[0], dtype=np.float32) < 0.45, 1.0, 0.0).astype(np.float32)[:, None, None]
            cc = (AMB_SNOW * (0.85 + 0.7 * vv) + E * (0.45 + 0.4 * vv)) * np.ones((1, mm.shape[1], 1), np.float32)
            cv.paint(x0, y0, mm, cc.astype(np.float32), z0 - 0.02)

    def _bench(self, cv, z0=19.4, z1=21.8):
        cam = self.cam
        x0w, x1w = BLD_X1 + 0.05, BLD_X1 + 0.5
        ys = Y_PLAT + 0.45
        E = self.light_at(np.array([[x1w, ys, (z0 + z1) / 2]], np.float32))[0]
        wood = np.array([0.35, 0.2, 0.12], np.float32)
        # legs
        for z in (z0 + 0.1, z1 - 0.1):
            self.quad(cv, [(x1w - 0.05, Y_PLAT, z), (x1w - 0.05, ys, z), (x1w - 0.05, ys, z + 0.06), (x1w - 0.05, Y_PLAT, z + 0.06)],
                      np.array([0.04, 0.04, 0.05], np.float32), z)
        # seat top (visible from above) and front edge
        self.quad(cv, [(x0w, ys, z0), (x1w, ys, z0), (x1w, ys, z1), (x0w, ys, z1)], wood * (0.2 + E * 1.1), (z0 + z1) / 2)
        self.quad(cv, [(x1w, ys, z0), (x1w, ys - 0.06, z0), (x1w, ys - 0.06, z1), (x1w, ys, z1)], wood * (0.12 + E * 0.6), (z0 + z1) / 2)
        # backrest slats against wall
        for yb in (ys + 0.25, ys + 0.45):
            self.quad(cv, [(x0w + 0.02, yb, z0), (x0w + 0.02, yb + 0.1, z0), (x0w + 0.02, yb + 0.1, z1), (x0w + 0.02, yb, z1)],
                      wood * (0.18 + E * 0.9), (z0 + z1) / 2)

    def _bench_fg(self, cv):
        """Snow-covered wooden platform bench under the foreground lamp pool (faces the track): iron
        leg frames, a thick rounded snow cushion on the seat, snow along the backrest, lamp-lit front edge."""
        cam, s = self.cam, self.s
        B = self.BENCH
        x0w, x1w, z0, z1, ys = B['x0'], B['x1'], B['z0'], B['z1'], B['ys']
        iron = np.array([0.03, 0.03, 0.045], np.float32)
        wood = np.array([0.42, 0.24, 0.12], np.float32)
        Nx = np.array([1, 0, 0], np.float32)
        Nz = np.array([0, 0, -1], np.float32)
        E_front = self.light_at(np.array([[x1w + 0.05, ys, (z0 + z1) / 2]], np.float32), Nx)[0]
        E_end = self.light_at(np.array([[(x0w + x1w) / 2, ys, z0 - 0.05]], np.float32), Nz)[0]
        E_top = self.light_at(np.array([[(x0w + x1w) / 2, ys + 0.2, (z0 + z1) / 2]], np.float32), np.array([0, 1, 0], np.float32))[0]
        # contact shadow / snow banked under the seat
        P = np.array([(x0w - 0.1, Y_PLAT, z0 - 0.1), (x1w + 0.12, Y_PLAT, z0 - 0.1), (x1w + 0.12, Y_PLAT, z1 + 0.1), (x0w - 0.1, Y_PLAT, z1 + 0.1)])
        X0, Y0, m = L.poly_local(cam.pts(P), pad=6)
        m = cv2.GaussianBlur(m, (0, 0), max(cam.f * 0.04 / z0, 1.0))
        cv.paint(X0, Y0, m * 0.55, AMB_SNOW * np.array([0.3, 0.36, 0.7], np.float32) + E_top * 0.08, (z0 + z1) / 2)
        # leg frames (near and far): front leg, back leg, cross bar
        for zl in (z1 - 0.12, z0 + 0.12):
            for (xa, ya, xb, yb) in ((x1w - 0.05, Y_PLAT, x1w - 0.05, ys), (x0w + 0.05, Y_PLAT, x0w + 0.03, ys + 0.5),
                                     (x0w + 0.05, Y_PLAT + 0.18, x1w - 0.05, Y_PLAT + 0.18)):
                x0_, y0_, m = L.line_local(cam.pts(np.array([[xa, ya, zl], [xb, yb, zl]])), max(cam.f * 0.045 / zl, 0.8 * s))
                El = self.light_at(np.array([[xa, (ya + yb) / 2, zl - 0.05]], np.float32), Nz)[0]
                cv.paint(x0_, y0_, m, iron + El * 0.08, zl)
        # seat slats: front face (toward the track) and the near end - painted wood (grain, worn edges)
        def wood_tex(Lm, Hm, lit, seed, per):
            tw_, th_ = 480, max(int(480 * Hm / Lm), 12)
            c = T2.painted_wood(tw_, th_, Lm, Hm, seed=seed, base=wood * 1.1, per=None, along_x=True)
            c = R3.plank_detail(c, Lm, joints=(0.47,) if Lm > 1.0 else (), bolts=((0.06, 0.5), (0.94, 0.5)) if Lm > 1.0 else (),
                                seed=seed, grain_boost=2.0)
            return (c * lit * 1.7).astype(np.float32)
        PT.paste_quad(cv, cam.pts(np.array([(x1w, ys - 0.07, z0), (x1w, ys - 0.07, z1), (x1w, ys, z1), (x1w, ys, z0)])),
                      wood_tex(z1 - z0, 0.07, AMB_SNOW * 0.4 + E_front * 1.0, 51, 0.07), (z0 + z1) / 2)
        PT.paste_quad(cv, cam.pts(np.array([(x0w, ys - 0.07, z0), (x1w, ys - 0.07, z0), (x1w, ys, z0), (x0w, ys, z0)])),
                      wood_tex(x1w - x0w, 0.07, AMB_SNOW * 0.35 + E_end * 1.0, 57, 0.07), z0)
        # backrest slats
        for yb_ in (ys + 0.2, ys + 0.38):
            PT.paste_quad(cv, cam.pts(np.array([(x0w + 0.02, yb_, z0), (x0w + 0.02, yb_, z1), (x0w + 0.02, yb_ + 0.1, z1),
                                                 (x0w + 0.02, yb_ + 0.1, z0)])),
                          wood_tex(z1 - z0, 0.1, (AMB_SNOW * 0.4 + E_front * 0.9) * 0.85, 52 + int(yb_ * 10), 0.1), (z0 + z1) / 2)
            self.quad(cv, [(x0w - 0.01, yb_, z0), (x0w + 0.05, yb_, z0), (x0w + 0.05, yb_ + 0.1, z0), (x0w - 0.01, yb_ + 0.1, z0)],
                      wood * (AMB_SNOW * 0.35 + E_end * 0.9), z0)
        rng = np.random.default_rng(66)

        def pillow(xa, xb, ybase, hgt, zc, za, zb, lit_col, shd_col, n=60, lip=0.05):
            """Painted snow SLAB on the seat (round 5): one flat pale top plane, one flat cool front face,
            a hard lit top edge, squared (slightly broken) ends and a few small drip overhangs - no
            rounded cushion, no gradient, no bevel."""
            zs = np.linspace(za, zb, n)
            tt = (zs - za) / (zb - za)
            # squared ends: the slab stops almost vertically, with a tiny chipped corner
            end = np.ones(n)
            jag = rng.uniform(-1, 1, n)
            jag = np.convolve(jag, np.ones(3) / 3, mode='same')
            ytop = ybase + hgt * (0.92 + 0.08 * jag) * end
            back = np.stack([np.full(n, xa), ytop, zs], 1)
            front = np.stack([np.full(n, xb + 0.02), ytop, zs], 1)
            X0, Y0, m = L.poly_local(cam.pts(np.concatenate([back, front[::-1]])), ss=4)
            cv.paint(X0, Y0, m, lit_col, zc)
            # near end face (faces the camera / the foreground lamp): one flat mid value
            h0 = float(ytop[0] - ybase)
            self.quad(cv, [(xa, ybase - lip * 0.35, za), (xb + 0.022, ybase - lip * 0.35, za), (xb + 0.02, ybase + h0, za),
                           (xa, ybase + h0, za)], lit_col * 0.3 + shd_col * 0.7, za - 0.01)
            # front face: vertical, ONE flat cool value, a few small drips hanging past the slat edge
            drp = np.zeros(n)
            for _ in range(5):
                c_ = rng.uniform(0.08, 0.92)
                drp += rng.uniform(0.5, 1.0) * np.exp(-((tt - c_) / rng.uniform(0.01, 0.025)) ** 2)
            lipd = (lip * 0.35 + lip * 1.1 * np.clip(drp, 0, 1)) * (end > 0.99)
            ft = np.stack([np.full(n, xb + 0.02), ytop, zs], 1)
            fb = np.stack([np.full(n, xb + 0.022), np.minimum(ybase - lipd, ytop - 0.005), zs], 1)
            X0, Y0, m = L.poly_local(cam.pts(np.concatenate([ft, fb[::-1]])), ss=4)
            cv.paint(X0, Y0, m, shd_col, zc - 0.05)
            # thin flat cast shadow of the slab on the slat face right under it
            sb = np.stack([np.full(n, xb + 0.005), ybase - lipd - 0.02 * end, zs], 1)
            st2 = np.stack([np.full(n, xb + 0.005), ybase - lipd * 0.5, zs], 1)
            X0, Y0, m = L.poly_local(cam.pts(np.concatenate([st2, sb[::-1]])), ss=4)
            cv.paint(X0, Y0, m * 0.5, AMB_SNOW * np.array([0.18, 0.2, 0.42], np.float32), zc - 0.04)
            # hard lit edge along the top-front corner
            X0, Y0, m = L.line_local(cam.pts(ft), max(cam.f * 0.008 / zc, 0.7 * s))
            cv.paint(X0, Y0, m, lit_col * 1.2 + np.array([0.06, 0.04, 0.02], np.float32), zc - 0.06)
        self._drift_along(cv, x1w + 0.04, z1 + 0.05, x1w + 0.04, z0 - 0.02, 0.05, 81, zdraw=(z0 + z1) / 2 - 0.1)
        self._drift_along(cv, x0w - 0.1, z0 - 0.1, x1w + 0.1, z0 - 0.1, 0.07, 82, zdraw=z0 - 0.15)
        lit = snow_lit(AMB_SNOW * 1.15, E_top[None] * 0.8)[0]
        lit = lit * 1.12 + np.array([0.05, 0.05, 0.05], np.float32)     # (c7) flat, no sheen; round 5: the top plane is the palest value
        shd = AMB_SNOW * np.array([0.55, 0.62, 0.95], np.float32) + E_front * 0.25
        shd = AMB_SNOW * np.array([0.62, 0.6, 0.86], np.float32) + np.array([0.03, 0.02, 0.04], np.float32) + E_front * 0.16
        pillow(x0w + 0.04, x1w, ys, 0.07, (z0 + z1) / 2, z0 - 0.02, z1 + 0.02, lit, shd, lip=0.04)
        pillow(x0w - 0.02, x0w + 0.07, ys + 0.48, 0.045, (z0 + z1) / 2, z0, z1, lit * 1.02, shd, lip=0.022)

    def _vending(self, cv):
        cam = self.cam
        x0w, x1w, z0, z1 = BLD_X1, BLD_X1 + 0.75, 23.4, 24.5
        yt = Y_PLAT + 1.83
        # side (faces camera, -z): red body lit by the canopy tubes, a white stripe and a lit brand panel
        E = self.light_at(np.array([[(x0w + x1w) / 2, Y_PLAT + 1.0, z0 - 0.05]], np.float32), np.array([0, 0, -1], np.float32))[0]
        # front (faces the camera, -z): lit display window with three rows of drinks, price tags, button
        # lamps, logo band and coin panel; the display glows (emissive) and the red body is lit
        sp = cam.pts(np.array([(x0w, Y_PLAT, z0), (x1w, Y_PLAT, z0), (x1w, yt, z0), (x0w, yt, z0)]))
        tex = PT.vending_front(96, 234, 4)
        th_ = tex.shape[0]
        vv = np.linspace(0, 1, th_, dtype=np.float32)[:, None, None]
        emis = (tex.max(-1, keepdims=True) > 0.9).astype(np.float32)
        body = tex * (AMB_SNOW * (0.5 + 0.5 * vv) + E * 1.1 + 0.25)
        tex = body * (1 - emis) + tex * 0.9 * emis
        tex[:, -2:] = tex[:, -2:] * 0.5 + np.array([0.7, 0.3, 0.25], np.float32)   # lit edge
        PT.paste_quad(cv, sp, tex, z0)
        # side (faces +x, toward the track): red body with the advert panel
        P4 = [(x1w, Y_PLAT, z0), (x1w, Y_PLAT, z1), (x1w, yt, z1), (x1w, yt, z0)]
        stx = PT.vending_side(60, 140)
        vv = np.linspace(0, 1, 140, dtype=np.float32)[:, None, None]
        PT.paste_quad(cv, cam.pts(P4), stx * (AMB_SNOW * (0.5 + 0.5 * vv) + E * 0.8 + 0.1), (z0 + z1) / 2)
        # snow cap on top of the machine
        self.quad(cv, [(x0w - 0.03, yt, z0 - 0.03), (x1w + 0.04, yt, z0 - 0.03), (x1w + 0.03, yt + 0.07, z0 - 0.03),
                       (x0w - 0.02, yt + 0.06, z0 - 0.03)], AMB_SNOW * 1.0 + E * 0.6, z0 - 0.03)

    def _canopy(self, cv):
        cam, s = self.cam, self.s
        # underside y = CAN_Y
        m, X, Y, Z = self._plane_region('y', CAN_Y, ((CAN_X0, CAN_X1), (-99, 99), (CAN_Z0, CAN_Z1)))
        P = np.stack([X, np.full_like(X, CAN_Y), Z], -1)
        # the canopy lamps light the ceiling around them (they hang just below it)
        E = np.zeros(X.shape + (3,), np.float32)
        for (lp, col, I, cone, cpow, _) in self.lamps:
            if lp[1] > 1.0 and CAN_Z0 < lp[2] < CAN_Z1:
                d2 = (X - lp[0]) ** 2 + (Z - lp[2]) ** 2 + 0.05
                E += (I * 0.08 / d2)[..., None] * col
        E += self.light_at(P, np.array([0, -1, 0], np.float32)) * 0.4
        tex = self._snow_tex(X, Z * 4, 13, 0.05)
        panel = 0.8 + 0.2 * (np.cos(X / 0.9 * 2 * np.pi) > 0.96)
        boards = 1 - 0.18 * C.smoothstep(0.9, 0.99, np.cos(Z / 0.3 * 2 * np.pi)) * C.smoothstep(26, 18, Z)
        stain = self._snow_tex(X * 1.2, Z * 1.2, 26, 0.05)
        base = np.array([0.18, 0.15, 0.14], np.float32) * (0.85 + 0.3 * tex[..., None]) * panel[..., None]
        base = base * (boards * (0.85 + 0.3 * stain))[..., None]
        # (cycle 7) warm bounce off the lit platform snow under each tube, fading into a cool shade
        # toward the open end; painted board seams stay readable in both
        bnc = np.zeros(X.shape, np.float32)
        for (lp, col_, I, cone, cpow, _) in self.lamps:
            if lp[1] > 1.0 and CAN_Z0 < lp[2] < CAN_Z1:
                bnc += np.exp(-(((X - lp[0]) / 1.6) ** 2 + ((Z - lp[2]) / 2.2) ** 2))
        bnc = np.clip(bnc, 0, 1.2)
        seam = 1 - 0.35 * C.smoothstep(0.93, 0.99, np.cos(X / 0.24 * 2 * np.pi))
        base = base * seam[..., None]
        col = (base * (AMB_SNOW * 0.42 + E) + np.array([0.06, 0.035, 0.015], np.float32) * C.smoothstep(30, 17, Z)[..., None]
               + base * bnc[..., None] * np.array([0.95, 0.6, 0.3], np.float32) * 0.9)
        fa = self.fog_amt(Z)[..., None]
        col = col * (1 - fa) + FOG * fa
        self._paint_region(cv, m, col, Z)
        # transverse beams (front face visible)
        tubes = [lp for (lp, _c, _I, _cn, _cp, _s) in self.lamps if lp[1] > 1.0 and CAN_Z0 < lp[2] < CAN_Z1]
        for k, zb in enumerate((CAN_Z1 - 0.2, 23.5 + 2.2, 21.2, 18.6)):
            E = self.light_at(np.array([[0.0, CAN_Y - 0.15, zb - 0.1]], np.float32), np.array([0, 0, -1], np.float32))[0]
            # (cycle 7) painted timber beam: grain + weathering, a warm bounce lip picked up from the
            # nearest tubes (strongest on the bottom edge), cool shade elsewhere
            tex = T2.painted_wood(640, 24, CAN_X1 - CAN_X0, 0.22, seed=140 + k, base=(0.3, 0.22, 0.16))
            xw = np.linspace(CAN_X0, CAN_X1, 640, dtype=np.float32)[None, :]
            yv = np.linspace(0, 1, 24, dtype=np.float32)[:, None]          # 0 top .. 1 bottom
            wb = np.zeros_like(xw)
            for lp in tubes:
                wb = wb + np.exp(-((xw - lp[0]) / 1.8) ** 2 - ((zb - lp[2]) / 3.0) ** 2)
            wb = np.clip(wb, 0, 1.2) * (0.35 + 0.65 * yv)
            lt = AMB_SNOW * 0.42 + E * 1.1 + wb[..., None] * np.array([1.1, 0.7, 0.36], np.float32) * 0.9
            self._paste(cv, [(CAN_X0, CAN_Y - 0.22, zb), (CAN_X1, CAN_Y - 0.22, zb), (CAN_X1, CAN_Y, zb), (CAN_X0, CAN_Y, zb)],
                        tex * lt, zb)
        # canopy lamps: fluorescent-like fixtures
        for (lp, col, I, cone, cpow, _) in self.lamps:
            if lp[1] > 1.0 and CAN_Z0 < lp[2] < CAN_Z1:
                x, y, z = lp
                self.quad(cv, [(x - 0.35, y + 0.06, z), (x + 0.35, y + 0.06, z), (x + 0.35, y - 0.03, z), (x - 0.35, y - 0.03, z)],
                          np.array([0.1, 0.1, 0.12], np.float32), z)
                self.quad(cv, [(x - 0.32, y - 0.03, z - 0.05), (x + 0.32, y - 0.03, z - 0.05), (x + 0.32, y - 0.07, z - 0.05),
                               (x - 0.32, y - 0.07, z - 0.05)], col * 4.0, z)
        self._clock(cv)
        # pillars
        for zp in (29.6, 23.5, 17.2):
            for xp in (CAN_X1 - 0.35,):
                self._pillar(cv, xp, zp)
        # front fascia + snow slab + icicles
        z = CAN_Z0
        E = self.light_at(np.array([[0.0, CAN_Y + 0.2, z - 0.1]], np.float32), np.array([0, 0, -1], np.float32))[0]
        # (cycle 7) painted fascia board: plank seams + weathering streaks, the main lamp's warm spill
        # on its right end breaking into cool blue shade toward the left
        tex = T2.painted_wood(900, 40, CAN_X1 - CAN_X0, 0.4, seed=150, base=(0.42, 0.38, 0.36))
        xw = np.linspace(CAN_X0, CAN_X1, 900, dtype=np.float32)[None, :]
        seam = 1 - 0.45 * (np.abs(np.mod(xw + 0.3, 1.8) - 0.9) > 0.885)
        lx0 = self.lamps[0][0][0]
        wsp = np.exp(-((xw - lx0) / 3.2) ** 2) * 0.9 + np.exp(-((xw - self.lamps[-1][0][0]) / 1.6) ** 2) * 0.5
        lt = AMB_SNOW * 0.5 + E * 0.6 + wsp[..., None] * np.array([1.0, 0.62, 0.3], np.float32) * 0.7
        rs = np.random.default_rng(151)
        strk = cv2.GaussianBlur(rs.random((40, 900)).astype(np.float32), (0, 0), sigmaX=1.2, sigmaY=12)
        strk = 1 - 0.5 * np.clip((strk - strk.mean()) / (strk.std() + 1e-6), 0, 2) * np.linspace(0.3, 1, 40, dtype=np.float32)[:, None]
        self._paste(cv, [(CAN_X0, CAN_Y - 0.05, z), (CAN_X1, CAN_Y - 0.05, z), (CAN_X1, CAN_Y + 0.35, z), (CAN_X0, CAN_Y + 0.35, z)],
                    tex * (seam * strk)[..., None] * lt, z)
        self._cornice(cv, z)

    def _cornice(self, cv, z):
        """Front snow load of the canopy painted as an irregular cornice: lumpy wind-shaped top, uneven
        overhang (heavy lobes curling over the fascia, a few places where a chunk has already dropped),
        warm amber rim / face where the platform lamps catch it, sky-lit pale crown, blue-violet shadowed
        underside, compressed-layer brush texture, stable ice glints and irregular icicles."""
        cam, s = self.cam, self.s
        rng = np.random.default_rng(55)
        n = 320
        xs = np.linspace(CAN_X0 - 0.18, CAN_X1 + 0.16, n)
        q = (xs - xs[0]) / (xs[-1] - xs[0])

        def bumps(k, w0, w1, a0, a1):
            out = np.zeros(n)
            for _ in range(k):
                c = rng.uniform(xs[0], xs[-1])
                out += rng.uniform(a0, a1) * np.exp(-((xs - c) / rng.uniform(w0, w1)) ** 2)
            return out
        # top silhouette: broad swells + smaller lumps; rounded, uneven ends
        top = CAN_Y + 0.66 + bumps(6, 0.6, 1.8, -0.06, 0.1) + bumps(14, 0.12, 0.35, -0.03, 0.05)
        top = top + 0.012 * np.sin(xs * 17.0 + 1.3) + 0.008 * np.sin(xs * 31.0)
        endl = np.clip(q / 0.035, 0, 1) ** 0.55
        endr = np.clip((1 - q) / 0.05, 0, 1) ** 0.7
        top = CAN_Y + 0.3 + (top - CAN_Y - 0.3) * endl * endr
        # overhang: heavy lobes curling over the fascia (varied width / depth), a few broken-off gaps
        droop = bumps(9, 0.18, 0.6, 0.04, 0.2) + bumps(10, 0.05, 0.14, 0.02, 0.09)
        for _ in range(3):
            c = rng.uniform(xs[0] + 0.8, xs[-1] - 0.8)
            droop -= 0.2 * np.exp(-((xs - c) / rng.uniform(0.2, 0.45)) ** 4)
        bot = CAN_Y + 0.2 - np.clip(droop, -0.12, 0.3)
        bot = np.minimum(bot, top - 0.1)
        zf = z - 0.12
        pt_top = cam.pts(np.stack([xs, top, np.full(n, zf)], 1))
        pt_bot = cam.pts(np.stack([xs, bot, np.full(n, zf)], 1))
        # cool shadow cast by the cornice on the fascia below it
        shb = np.stack([xs, bot + 0.03, np.full(n, z - 0.02)], 1)
        shl = np.stack([xs, bot - 0.13 - 0.4 * np.clip(droop, 0, None), np.full(n, z - 0.02)], 1)
        x0, y0, mm = L.poly_local(cam.pts(np.concatenate([shb, shl[::-1]])))
        mm = mm * np.linspace(0.25, 0.75, mm.shape[0], dtype=np.float32)[::-1, None]
        cv.paint(x0, y0, mm, np.array([0.05, 0.06, 0.18], np.float32), z - 0.02)
        x0, y0, mm = L.poly_local(np.concatenate([pt_top, pt_bot[::-1]]), ss=4)
        hh, ww = mm.shape
        uu = np.arange(ww, dtype=np.float32)[None, :] + x0
        rows = np.arange(hh, dtype=np.float32)[:, None] + y0
        ytp = np.interp(uu[0], pt_top[:, 0], pt_top[:, 1])[None, :]
        ybt = np.interp(uu[0], pt_bot[:, 0], pt_bot[:, 1])[None, :]
        rv = np.clip((rows - ytp) / np.maximum(ybt - ytp, 1.0), 0, 1)
        Xw = (uu[0] - cam.cx) * zf / cam.f + CAM_X
        Ew = self.light_at(np.stack([Xw, np.full_like(Xw, CAN_Y + 0.4), np.full_like(Xw, zf)], -1).astype(np.float32),
                           np.array([0, 0, -1], np.float32))
        Ew = Ew[None, :, :]
        # value masses: pale sky-lit crown -> front face (lamp-lit) -> blue-violet curl-under
        rng2 = np.random.default_rng(56)
        jag = (0.06 * np.sin(uu * (0.05 / s) + 1.0) + 0.04 * np.sin(uu * (0.13 / s) + 2.0))
        # painted as two flat planes: a pale sky-lit top plane with a crisp jagged lower edge, and ONE
        # cool blue-violet front value (warmed only where the lamp pool actually reaches) - no bevel,
        # no graded curl-under, no layer striping
        crown = 1 - C.smoothstep(0.3, 0.315, rv + jag)
        crown_c = AMB_SNOW * np.array([1.62, 1.58, 1.4], np.float32) + np.array([0.05, 0.06, 0.1], np.float32) + Ew * 0.3
        ewb = paint_E(Ew)
        face = np.array([0.25, 0.3, 0.5], np.float32) + ewb * np.array([0.5, 0.36, 0.22], np.float32)
        face = np.broadcast_to(face, (hh, ww, 3))
        col = face * (1 - crown[..., None]) + crown_c * crown[..., None]
        grain = cv2.GaussianBlur(rng2.random((hh, ww)).astype(np.float32), (0, 0), sigmaX=7.0 * s + 1, sigmaY=0.7 * s + 0.3)
        grain = (grain - grain.mean()) / (grain.std() + 1e-6)
        col = col * (1 + 0.012 * grain)[..., None]
        cv.paint(x0, y0, mm, col.astype(np.float32), zf)
        # silhouette rims: pale cool sky rim along the top, amber rim where the lamps catch the edges
        x0r, y0r, mr = L.line_local(pt_top + [0, 0.8 * s], max(cam.f * 0.022 / z, 0.8 * s))
        Er = self.light_at(np.stack([xs, top, np.full(n, zf - 0.1)], 1).astype(np.float32), np.array([0, 0.5, -0.8], np.float32))
        cv.paint(x0r, y0r, mr * 0.7, AMB_SNOW * 1.9 + np.array([0.05, 0.05, 0.08], np.float32), zf - 0.01)
        wr = np.clip(np.interp(np.arange(mr.shape[1]) + x0r, pt_top[:, 0], Er.mean(-1)) * 1.2, 0, 1)[None, :]
        cv.paint(x0r, y0r, mr * 0.9 * wr, np.array([1.35, 0.92, 0.5], np.float32) * np.ones((mr.shape[0], mr.shape[1], 1), np.float32), zf - 0.012)
        Eb = self.light_at(np.stack([xs, bot, np.full(n, zf - 0.1)], 1).astype(np.float32), np.array([0, -0.3, -0.95], np.float32))
        x0b, y0b, mb = L.line_local(pt_bot - [0, 1.2 * s], max(cam.f * 0.016 / z, 0.7 * s))
        wb = np.clip(np.interp(np.arange(mb.shape[1]) + x0b, pt_bot[:, 0], Eb.mean(-1)) * 1.3, 0, 1)[None, :]
        cv.paint(x0b, y0b, mb * 0.0 * wb, np.array([1.2, 0.72, 0.36], np.float32) * np.ones((mb.shape[0], mb.shape[1], 1), np.float32), zf - 0.012)
        # stable ice glints on the crown / face (tiny, warm near the lamps)
        gl = np.random.default_rng(57)
        ng = 90
        gx = gl.uniform(xs[0] + 0.1, xs[-1] - 0.1, ng)
        gt = np.interp(gx, xs, top)
        gb = np.interp(gx, xs, bot)
        gy = gt - (gt - gb) * gl.uniform(0.03, 0.55, ng) ** 1.6
        Pg = np.stack([gx, gy, np.full(ng, zf - 0.02)], 1)
        Eg = self.light_at(Pg.astype(np.float32), np.array([0, 0, -1], np.float32)).mean(-1)
        ug, vg = cam.p(gx, gy, np.full(ng, zf - 0.02))
        for i in range(ng):
            r = max(cam.f * gl.uniform(0.006, 0.013) / z, 0.6 * s)
            wk = float(np.clip(Eg[i] * 1.5, 0, 1))
            if wk < 0.35 or gl.random() < 0.6:          # round 5: sparse, only in the lamp pool
                continue
            cc = np.array([1.2, 1.3, 1.6], np.float32) * (1 - wk) + np.array([1.7, 1.4, 0.95], np.float32) * wk
            x0g, y0g, mg = L.poly_local(np.array([[ug[i] - r, vg[i]], [ug[i], vg[i] - r * 1.6], [ug[i] + r, vg[i]], [ug[i], vg[i] + r * 1.6]]), ss=4)
            cv.paint(x0g, y0g, mg * gl.uniform(0.5, 1.0), cc, zf - 0.02)
        # icicles: clustered under the lobes (longer under heavy ones), varied length / thickness, a
        # slight kink; lamp-lit edge + bright tip where the lamps catch them
        centers = gl.uniform(CAN_X0 + 0.2, CAN_X1 - 0.1, 10)
        for i in range(70):
            x = float(np.clip(gl.choice(centers) + gl.normal(0, 0.3), CAN_X0, CAN_X1))
            dr = float(np.interp(x, xs, np.clip(droop, 0, None)))
            yb0 = float(np.interp(x, xs, bot)) + 0.015
            ln = 0.03 + (0.25 + 1.6 * dr) * gl.random() ** 2.6
            wd = 0.008 + 0.06 * ln * gl.uniform(0.6, 1.4) + gl.uniform(0, 0.008)
            kink = gl.uniform(-0.25, 0.25) * wd
            P = np.array([[x - wd, yb0, zf - 0.02], [x + wd, yb0, zf - 0.02], [x + wd * 0.4 + kink, yb0 - ln * 0.5, zf - 0.02],
                          [x + kink * 1.6, yb0 - ln, zf - 0.02], [x - wd * 0.3 + kink, yb0 - ln * 0.45, zf - 0.02]])
            E = self.light_at(np.array([[x, yb0 - ln / 2, zf - 0.1]], np.float32))[0]
            x0i, y0i, mi = L.poly_local(cam.pts(P), ss=4)
            vv = np.linspace(0, 1, mi.shape[0], dtype=np.float32)[:, None, None]
            cc = (AMB_SNOW * np.array([0.55, 0.6, 0.85], np.float32) + E * 0.7) * (0.75 + 0.55 * vv) * np.ones((1, mi.shape[1], 1), np.float32)
            cv.paint(x0i, y0i, mi * 0.85, cc.astype(np.float32), zf - 0.03)
            lk = float(np.clip(E.mean() * 1.4, 0, 1))
            e_ = cam.pts(np.array([[x - wd * 0.35, yb0, zf - 0.03], [x + kink * 1.5, yb0 - ln * 0.93, zf - 0.03]]))
            x0i, y0i, mi = L.line_local(e_, max(cam.f * 0.005 / z, 0.5 * s))
            ec = AMB_SNOW * 1.3 * (1 - lk) + np.array([1.5, 1.05, 0.6], np.float32) * lk + 0.12
            cv.paint(x0i, y0i, mi * 0.75, ec.astype(np.float32), zf - 0.035)
            if ln > 0.07:
                tu, tv = cam.p(x + kink * 1.6, yb0 - ln + 0.008, zf - 0.03)
                rr = max(cam.f * 0.01 / z, 0.7 * s)
                x0i, y0i, mi = L.poly_local(np.array([[tu - rr, tv], [tu, tv - rr], [tu + rr, tv], [tu, tv + rr]]), ss=4)
                cv.paint(x0i, y0i, mi, np.array([1.3, 1.25, 1.3], np.float32) * (1 - lk) + np.array([2.0, 1.5, 0.9], np.float32) * lk, zf - 0.04)

    def _pillar(self, cv, x, z, w=0.12):
        cam = self.cam
        c = np.array([0.42, 0.42, 0.45], np.float32)
        for face in ('front', 'side'):
            if face == 'front':
                P4 = [(x - w, Y_PLAT, z - w), (x + w, Y_PLAT, z - w), (x + w, CAN_Y, z - w), (x - w, CAN_Y, z - w)]
                N = np.array([0, 0, -1], np.float32)
                zz = z - w
            else:
                P4 = [(x - w, Y_PLAT, z - w), (x - w, Y_PLAT, z + w), (x - w, CAN_Y, z + w), (x - w, CAN_Y, z - w)]
                N = np.array([-1, 0, 0], np.float32)
                zz = z
            x0, y0, m = L.poly_local(cam.pts(np.array(P4)))
            hh = m.shape[0]
            ys = np.linspace(CAN_Y, Y_PLAT, hh, dtype=np.float32)
            Pw = np.stack([np.full_like(ys, x - (w if face == 'side' else 0)), ys, np.full_like(ys, zz - 0.05)], -1)
            E = self.light_at(Pw, N)
            bounce = C.smoothstep(Y_PLAT + 1.8, Y_PLAT, ys)[:, None]
            col = (AMB_SNOW * (0.3 + 0.5 * bounce) + E * 0.9 + bounce * 0.15 * WARM * (face == 'front')) * 1.15
            tw = max(m.shape[1] * 2, 8)
            alb = T2.painted_wood(tw, hh, 2 * w, CAN_Y - Y_PLAT, seed=int(abs(x * 31 + z * 7)) + (face == 'side'),
                                  base=(0.4, 0.34, 0.3), along_x=False, grime_bottom=1.0)
            alb = cv2.resize(alb, (m.shape[1], hh), interpolation=cv2.INTER_AREA)
            cv.paint(x0, y0, m, (alb * col[:, None, :]).astype(np.float32), zz)

    def _clock(self, cv):
        cam, s = self.cam, self.s
        x, y, z = 0.9, 1.02, 18.5
        u, v = cam.p(x, y, z)
        R = cam.f * 0.2 / z
        # hanging rod
        x0, y0, m = L.line_local(cam.pts(np.array([[x, 1.38, z], [x, y + 0.2, z]])), max(cam.f * 0.02 / z, 0.7 * s))
        cv.paint(x0, y0, m, np.array([0.06, 0.06, 0.07], np.float32), z)
        ss = 4
        B = int(R * 1.3) + 3
        img = np.zeros((2 * B * ss, 2 * B * ss, 3), np.float32)
        a = np.zeros((2 * B * ss, 2 * B * ss), np.float32)
        cc = (B * ss, B * ss)
        cv2.circle(a, cc, int(R * ss), 1.0, -1, cv2.LINE_AA)
        E = self.light_at(np.array([[x, y, z - 0.1]], np.float32), np.array([0, 0, -1], np.float32))[0]
        face = np.array([0.92, 0.9, 0.85], np.float32) * (AMB_SNOW * 0.4 + E * 1.1 + 0.25)
        cv2.circle(img, cc, int(R * ss), tuple(float(q) for q in np.array([0.08, 0.08, 0.1])), -1, cv2.LINE_AA)
        cv2.circle(img, cc, int(R * 0.86 * ss), tuple(float(q) for q in face), -1, cv2.LINE_AA)
        for k in range(12):
            ang = k / 12 * 2 * math.pi
            r0, r1 = R * 0.7, R * 0.8
            cv2.line(img, (int(cc[0] + math.sin(ang) * r0 * ss), int(cc[1] - math.cos(ang) * r0 * ss)),
                     (int(cc[0] + math.sin(ang) * r1 * ss), int(cc[1] - math.cos(ang) * r1 * ss)),
                     (0.1, 0.1, 0.1), max(1, int(R * 0.05 * ss)), cv2.LINE_AA)
        for ang, ln, wd in ((2 * math.pi * (11 + 52 / 60) / 12, 0.45, 0.07), (2 * math.pi * 52 / 60, 0.68, 0.045)):
            cv2.line(img, cc, (int(cc[0] + math.sin(ang) * R * ln * ss), int(cc[1] - math.cos(ang) * R * ln * ss)),
                     (0.05, 0.05, 0.06), max(1, int(R * wd * ss)), cv2.LINE_AA)
        img = cv2.resize(img, (2 * B, 2 * B), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (2 * B, 2 * B), interpolation=cv2.INTER_AREA)
        cv.paint(int(u - B), int(v - B), a, img / np.maximum(a, 1e-3)[..., None], z)

    # ------------------------------------------------------------------ foreground book
    def _foreground(self):
        self._tree_fg(self.fgs['tree'])
        self._lamp_post(self.fgs['lamp'], 6.5)
        R3.pole_light(self.fgs['lamp'], self.cam, 2.05, 6.5, Y_PLAT, 2.45, 0.07, None, self.s, WARM)
        self._sign(self.fgs['sign'])

    def _tree_fg(self, cv):
        """The big foreground fir framing the left edge (own parallax plane: moves fastest)."""
        x, z, h = -7.9, 11.5, 12.9
        cam = self.cam
        tu, _ = cam.p(x, 0.0, z)
        _, tv = cam.p(x, Y_BED - 1.5 + h, z)
        _, bv = cam.p(x, Y_BED - 0.3, z)
        l0 = cam.p(*self.lamps[0][0])
        l1 = cam.p(*self.lamps[-1][0])
        s = self.s
        lamps_uv = [(l0[0], l0[1], WARM, 900 * s, 0.22), (l1[0], l1[1] + 40 * s, WARM, 300 * s, 1.0)]
        tu = 150 * s                                   # trunk visible just inside the left edge
        rgb, a = BG.bough_fir(cv.W, cv.H, tu, tv, bv, lambda v: 0.38 * (v - tv), s, seed=4242, lamps=lamps_uv,
                              fog=FOG, avoid=(415 * s, 430 * s, 790 * s))
        cv.rgb = rgb
        cv.a = a
        cv.z = np.where(a > 0.5, z, cv.z).astype(np.float32)

    def _sign(self, cv):
        cam, s = self.cam, self.s
        S_ = self.SIGN
        z, xc, wdt, yb, yt = S_['z'], S_['xc'], S_['w'], S_['yb'], S_['yt']
        dark = np.array([0.06, 0.06, 0.09], np.float32)
        def wood_quad(P4, seed, Lm, Hm, along_x, zq, base=(0.24, 0.18, 0.15)):
            pts = cam.pts(np.array(P4))
            tw = max(int((pts[:, 0].max() - pts[:, 0].min()) * 3), 8)
            th = max(int((pts[:, 1].max() - pts[:, 1].min()) * 3), 8)
            alb = T2.painted_wood(tw, th, Lm, Hm, seed=seed, base=base, along_x=along_x, grime_bottom=0.8 if not along_x else 0.0)
            if along_x:
                alb = R3.plank_detail(alb, Lm, joints=(0.5,), bolts=((0.03, 0.5), (0.97, 0.5)), seed=seed, grain_boost=1.7)
            else:
                alb = R3.plank_detail(alb.transpose(1, 0, 2).copy(), Hm, bolts=((0.12, 0.5), (0.2, 0.5)), seed=seed,
                                      grain_boost=1.7, top_wear=False).transpose(1, 0, 2).copy()
            Pc = np.array(P4, np.float32).mean(0)
            Eq = self.light_at(Pc[None] + np.array([[0, 0, -0.05]], np.float32), np.array([0, 0, -1], np.float32))[0]
            PT.paste_quad(cv, pts, (alb * (AMB_SNOW * 0.55 + Eq * 1.0)).astype(np.float32), zq)
        for xp in (xc - wdt / 2 + 0.12, xc + wdt / 2 - 0.12):
            wood_quad([(xp - 0.04, Y_PLAT, z + 0.02), (xp + 0.04, Y_PLAT, z + 0.02), (xp + 0.04, yb, z + 0.02),
                       (xp - 0.04, yb, z + 0.02)], int(xp * 100) % 97 + 30, 0.08, yb - Y_PLAT, False, z)
        for xp in (xc - wdt / 2 + 0.12, xc + wdt / 2 - 0.12):
            self._drift_along(cv, xp - 0.22, z - 0.06, xp + 0.18, z - 0.06, 0.07, abs(int(xp * 100)) + 5, zdraw=z - 0.07)
        # board frame
        wood_quad([(xc - wdt / 2 - 0.04, yb - 0.04, z), (xc + wdt / 2 + 0.04, yb - 0.04, z),
                   (xc + wdt / 2 + 0.04, yt + 0.04, z), (xc - wdt / 2 - 0.04, yt + 0.04, z)], 41, wdt + 0.08, yt - yb + 0.08,
                  True, z, base=(0.2, 0.15, 0.13))
        # face texture
        P4 = [(xc - wdt / 2, yb, z), (xc + wdt / 2, yb, z), (xc + wdt / 2, yt, z), (xc - wdt / 2, yt, z)]
        pts = cam.pts(np.array(P4))
        tw = int(max(pts[:, 0]) - min(pts[:, 0])) * 2 + 4
        th = int(tw * (yt - yb) / wdt)
        clean = L.sign_texture(tw, th)
        wth = TX.sign_weather(clean, seed=8)
        # keep the enamel face clean and white: weathering only at a fraction, stronger at the edges
        tex = clean * 0.45 + (R3.sign_grime(wth, seed=9) * 0.5 + wth * 0.5) * 0.55
        ink = (clean.max(-1, keepdims=True) < 0.3).astype(np.float32)
        tex = tex * (1 - ink) + clean * ink          # crisp black kana stay pure black
        # extra weathering: grime runs from the top edge / snow cap, faint scratches, worn enamel edges
        grn = np.random.default_rng(81)
        st_ = cv2.GaussianBlur(grn.random((th, tw)).astype(np.float32), (0, 0), sigmaX=max(tw * 0.004, 0.6), sigmaY=th * 0.18)
        st_ = (st_ - st_.mean()) / (st_.std() + 1e-6)
        yy_ = np.linspace(0, 1, th, dtype=np.float32)[:, None]
        runs = np.clip(st_ * 0.7 + 0.1, 0, 2) * np.exp(-yy_ / 0.22)
        tex = tex * (1 - 0.16 * runs[..., None]) + np.array([0.3, 0.24, 0.17], np.float32) * 0.1 * runs[..., None]
        # lighting across the face (world positions of texels)
        gx = np.linspace(xc - wdt / 2, xc + wdt / 2, tw, dtype=np.float32)
        gy = np.linspace(yt, yb, th, dtype=np.float32)
        GX, GY = np.meshgrid(gx, gy)
        Pw = np.stack([GX, GY, np.full_like(GX, z)], -1)
        E = self.light_at(Pw, np.array([0, 0, -1], np.float32))
        # white enamel reads lit: cool snow-bounce floor + lamp light + a warm spill falling from the
        # left platform lamp across the upper-left of the board
        l1 = self.lamps[-1][0]
        dl = np.sqrt((GX - l1[0]) ** 2 + (GY - l1[1]) ** 2 * 1.4)
        spill = np.exp(-(dl - 1.8) / 1.1).clip(0, 1.6)[..., None] * np.array([1.0, 0.66, 0.32], np.float32) * 0.55
        bnc = (0.5 + 0.18 * (1 - (GY - yb) / (yt - yb)))[..., None] * np.array([0.62, 0.68, 0.9], np.float32)
        lit = tex * (bnc + E * 0.6 + spill)
        src = np.float32([[0, th], [tw, th], [tw, 0], [0, 0]])
        dst = np.float32(pts)
        M = cv2.getPerspectiveTransform(src, dst)
        x0 = int(dst[:, 0].min()) - 2
        y0 = int(dst[:, 1].min()) - 2
        w = int(dst[:, 0].max()) - x0 + 3
        h = int(dst[:, 1].max()) - y0 + 3
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64) @ M
        img = cv2.warpPerspective(lit.astype(np.float32), T, (w, h), flags=cv2.INTER_AREA)
        a = cv2.warpPerspective(np.ones((th, tw), np.float32), T, (w, h), flags=cv2.INTER_LINEAR)
        cv.paint(x0, y0, a, img, z)
        # rust where the posts are bolted to the frame (bleeding down the posts)
        for xp in (xc - wdt / 2 + 0.12, xc + wdt / 2 - 0.12):
            for (ya_, yb_, al) in ((yb - 0.16, yb - 0.04, 0.8), (yb - 0.34, yb - 0.16, 0.35)):
                x0_, y0_, m_ = L.poly_local(cam.pts(np.array([(xp - 0.04, ya_, z + 0.01), (xp + 0.04, ya_, z + 0.01),
                                                            (xp + 0.04, yb_, z + 0.01), (xp - 0.04, yb_, z + 0.01)])))
                Er = self.light_at(np.array([[xp, yb - 0.1, z - 0.05]], np.float32), np.array([0, 0, -1], np.float32))[0]
                cv.paint(x0_, y0_, m_ * al, np.array([0.42, 0.17, 0.07], np.float32) * (AMB_SNOW * 0.35 + Er * 0.9), z + 0.01)
        # snow cap on top of the board: flat-topped cushion with rounded ends, a drooping overhang lip
        # curling over the front edge (blue-violet underside) and a few icicles hanging from it
        rng = np.random.default_rng(88)
        n = 220
        xs = np.linspace(xc - wdt / 2 - 0.09, xc + wdt / 2 + 0.09, n)
        tt = (xs - xs[0]) / (xs[-1] - xs[0])
        end = np.clip(np.minimum(tt, 1 - tt) / 0.07, 0, 1) ** 0.5
        lum_ = np.zeros(n)
        for _ in range(7):
            c_ = rng.uniform(xs[0], xs[-1])
            lum_ += rng.uniform(-0.035, 0.05) * np.exp(-((xs - c_) / rng.uniform(0.06, 0.3)) ** 2)
        # (cycle 7) squared, slightly chipped ends and a near-flat top (a slab, not a cushion)
        end = np.clip(np.minimum(tt / 0.012, (1 - tt) / 0.02), 0, 1)
        prof = np.clip(0.1 + 0.4 * lum_ + 0.004 * np.sin(xs * 23.0), 0.05, 0.16) * end
        top = np.stack([xs, yt + 0.07 + prof * 0.85, np.full_like(xs, z - 0.05)], 1)
        drp = np.zeros(n)
        for _ in range(8):
            c_ = rng.uniform(xs[0], xs[-1])
            drp += rng.uniform(0.01, 0.05) * np.exp(-((xs - c_) / rng.uniform(0.02, 0.09)) ** 2)
        lipd = (0.008 + 0.006 * np.abs(np.sin(xs * 2.3 + 0.6)) + np.clip(drp * 0.45, 0, 0.028)) * end
        bot = np.stack([xs, yt + 0.035 - lipd, np.full_like(xs, z - 0.05)], 1)
        pts = cam.pts(np.concatenate([top, bot[::-1]]))
        x0, y0, mm = L.poly_local(pts, ss=4)
        hh, ww = mm.shape
        ptop = cam.pts(top)
        pbot = cam.pts(bot)
        colx = np.arange(ww, dtype=np.float32) + x0
        ytop_px = np.interp(colx, ptop[:, 0], ptop[:, 1])[None, :]
        ybot_px = np.interp(colx, pbot[:, 0], pbot[:, 1])[None, :]
        rows = np.arange(hh, dtype=np.float32)[:, None] + y0
        rv = np.clip((rows - ytop_px) / np.maximum(ybot_px - ytop_px, 1e-3), 0, 1)
        Et = self.light_at(np.array([[xc, yt + 0.1, z - 0.1]], np.float32))[0]
        lit_c = snow_lit(AMB_SNOW * 1.1, Et[None] * 0.8)[0]
        shd_c = AMB_SNOW * np.array([0.5, 0.52, 0.95], np.float32) + Et * 0.12
        jg = 0.05 * np.sin(colx * 0.19 / s + 0.4)[None, :] + 0.03 * np.sin(colx * 0.47 / s)[None, :]
        # (cycle 7) flat planes only: pale top plane | crisp lit front lip | ONE flat cool front face
        stp = C.smoothstep(0.42, 0.44, rv + 0.5 * jg)
        lit_c = lit_c * 1.12 + np.array([0.03, 0.04, 0.05], np.float32)
        shd_c = AMB_SNOW * np.array([0.62, 0.62, 0.9], np.float32) + Et * 0.1
        colsl = lit_c * (1 - stp[..., None]) + shd_c * stp[..., None]
        lip = C.smoothstep(0.34, 0.37, rv + 0.5 * jg) * (1 - stp)
        colsl = colsl * (1 - lip[..., None]) + (lit_c * 1.25 + np.array([0.08, 0.05, 0.02], np.float32)) * lip[..., None]
        cv.paint(x0, y0, mm, colsl.astype(np.float32), z - 0.05)
        # icicles
        ic_c0 = rng.uniform(xs[0] + 0.15, xs[-1] - 0.15, 4)
        for k in range(11):
            xi = float(np.clip(rng.choice(ic_c0) + rng.normal(0, 0.12), xs[0] + 0.08, xs[-1] - 0.08))
            ld = float(np.interp(xi, xs, lipd))
            ld = ld - 0.035
            ln = (0.01 + 0.05 * rng.random() ** 2.5) * (0.6 + max(ld + 0.035, 0) / 0.06)
            wd_ = rng.uniform(0.01, 0.02)
            ic = np.array([(xi - wd_, yt - ld + 0.01, z - 0.06), (xi + wd_, yt - ld + 0.01, z - 0.06),
                           (xi + 0.002, yt - ld - ln, z - 0.06)])
            x0_, y0_, m_ = L.poly_local(cam.pts(ic), ss=4)
            vv_ = np.linspace(0, 1, m_.shape[0], dtype=np.float32)[:, None, None]
            ic_c = (np.array([0.55, 0.66, 0.95], np.float32) * (1 - 0.3 * vv_) + Et * 0.5) * np.ones((1, m_.shape[1], 1), np.float32)
            cv.paint(x0_, y0_, m_ * 0.9, ic_c.astype(np.float32), z - 0.06)
            # glint on the icicle
            x0_, y0_, m_ = L.line_local(cam.pts(np.array([(xi - wd_ * 0.3, yt - ld, z - 0.065), (xi, yt - ld - ln * 0.6, z - 0.065)])),
                                         max(cam.f * 0.004 / z, 0.5 * s))
            cv.paint(x0_, y0_, m_ * 0.8, np.array([1.1, 1.05, 1.0], np.float32) + Et * 0.6, z - 0.065)


    def _drift_along(self, cv, xa, za, xb, zb, hgt, seed, zdraw=None):
        """Snow piled against the base of an object: a lumpy rounded bank along the floor segment
        (xa, za) -> (xb, zb), lit on top by the lamps, cool blue at the foot."""
        cam = self.cam
        rng = np.random.default_rng(seed)
        n = 36
        t = np.linspace(0, 1, n)
        xs = xa + (xb - xa) * t
        zs = za + (zb - za) * t
        prof = hgt * (np.sin(np.pi * t) ** 0.6) * (0.8 + 0.2 * np.sin(t * 11 + rng.uniform(0, 6)))
        top = np.stack([xs, Y_PLAT + prof, zs], 1)
        bot = np.stack([xs, np.full(n, Y_PLAT - 0.03), zs], 1)
        x0, y0, mm = L.poly_local(cam.pts(np.concatenate([top, bot[::-1]])), ss=4)
        E = self.light_at(np.array([[(xa + xb) / 2, Y_PLAT + hgt, (za + zb) / 2]], np.float32), np.array([0, 1, 0], np.float32),
                          vis0=self.sign_shadow(np.array([(xa + xb) / 2]), np.array([(za + zb) / 2 - 0.3]))[0])[0]
        vv = np.linspace(0, 1, mm.shape[0], dtype=np.float32)[:, None, None]
        # same colour as the platform snow around it at the top, only a little cooler at the foot
        top_c = snow_lit(AMB_SNOW * 0.92, E[None] * 0.92 * 0.72)[0] * 1.04
        cc = top_c * (1 - 0.3 * vv) + top_c * np.array([0.7, 0.78, 1.0], np.float32) * 0.3 * vv
        cv.paint(x0, y0, mm, (cc * np.ones((1, mm.shape[1], 1), np.float32)).astype(np.float32),
                 (za + zb) / 2 if zdraw is None else zdraw)

    def _snow_mound(self, cv, xc, z, w):
        cam = self.cam
        n = 30
        xs = np.linspace(xc - w, xc + w, n)
        h = 0.12 * np.sin(np.linspace(0, np.pi, n)) ** 0.7
        top = np.stack([xs, Y_PLAT + h, np.full_like(xs, z)], 1)
        bot = np.stack([xs, np.full_like(xs, Y_PLAT - 0.02), np.full_like(xs, z)], 1)
        x0, y0, mm = L.poly_local(cam.pts(np.concatenate([top, bot[::-1]])))
        E = self.light_at(np.array([[xc, Y_PLAT + 0.1, z]], np.float32), np.array([0, 1, 0], np.float32))[0]
        cv.paint(x0, y0, mm, AMB_SNOW * 0.95 + E * 0.9, z)

    # ------------------------------------------------------------------ volumetric light in the snowy air
    def _volumetrics(self):
        q = 2
        PW, PH = self.PW, self.PH
        w, h = PW // q, PH // q
        rx = cv2.resize(self.rx, (w, h), interpolation=cv2.INTER_AREA)
        ry = cv2.resize(self.ry, (w, h), interpolation=cv2.INTER_AREA)
        # min-pool depth so light does not leak over thin near objects
        zb = cv2.erode(self.zbuf, np.ones((3, 3), np.uint8))
        zb = cv2.resize(zb, (w, h), interpolation=cv2.INTER_NEAREST)
        def scatter(sel):
            lp, lc, cin, cout, om, pw, ex = [], [], [], [], [], [], []
            for li, lamp in enumerate(self.lamps):
                (p, col, I, cone, cpow, sc) = lamp
                if not sc or not sel(li, p):
                    continue
                ci, co, o, pp, e = cone_params(lamp)
                lp.append((p[0] - CAM_X, p[1], p[2]))
                cc = AIR_AMBER if col is WARM else col
                lc.append(cc * I * float(sc))
                cin.append(ci); cout.append(co); om.append(o); pw.append(pp); ex.append(e)
            if not lp:
                return np.zeros((PH, PW, 3), np.float32)
            f32 = lambda q: np.array(q, np.float32)
            out = VL.inscatter(rx.astype(np.float32), ry.astype(np.float32), zb.astype(np.float32), f32(lp), f32(lc),
                               f32(cin), f32(cout), f32(om), f32(pw), f32(ex), np.float32(150.0), 40)
            out = cv2.GaussianBlur(out, (0, 0), 0.7)
            out = cv2.resize(out, (PW, PH), interpolation=cv2.INTER_LINEAR)
            v = np.clip(out, 0, None).astype(np.float32) * 0.05
            lum = v.max(-1, keepdims=True)
            return (v / (1 + 0.5 * lum)).astype(np.float32)
        # three parallax groups: the foreground lamp, station lamps, far lamps
        self.vol0 = scatter(lambda li, p: li == 0)
        # compress the hot core under the lamp head so the beam reads as an even amber cone, not a blob
        l0 = self.vol0.max(-1, keepdims=True)
        self.vol0 = (self.vol0 / (1 + 0.9 * l0)).astype(np.float32) * 1.15
        # feathered beam edges + soft halation round the lamp head (the beam is brightest at the head
        # and falls off lengthwise; no secondary hot spot inside it)
        self.vol0 = cv2.GaussianBlur(self.vol0, (0, 0), 2.5 * self.s + 0.5)
        lu, lv = self.cam.p(*self.lamps[0][0])
        us_, vs_ = C.grid(PW, PH)
        # (cycle 7) a soft volumetric SHAFT, not a spotlight mesh: the edges feather out widely (wider
        # blur across than along), and the density thins with distance from the head
        dv_ = np.clip(vs_ - lv, 0, None)
        fall = (0.35 + 0.65 * np.exp(-dv_ / (150 * self.s))).astype(np.float32)[..., None]
        self.vol0 = cv2.GaussianBlur(self.vol0 * fall, (0, 0), sigmaX=11 * self.s + 1, sigmaY=5 * self.s + 1) * 0.72
        # a wide, low-density light volume around the tight core (cm5_08: broad feathered cones)
        l0p = self.lamps[0][0]
        _, gv0 = self.cam.p(l0p[0], Y_PLAT, l0p[2])
        self.vol0 = self.vol0 + R4.small_cone(PW, PH, lu, lv + 6 * self.s, gv0, self.s, half_ang=0.5, amber=AIR_AMBER, strength=0.5, halo=0.0)
        d2 = (us_ - lu) ** 2 + (vs_ - (lv + 4 * self.s)) ** 2
        hal = (0.35 * np.exp(-d2 / (2 * (22 * self.s) ** 2)) + 0.14 * np.exp(-d2 / (2 * (70 * self.s) ** 2))
               + 0.05 * np.exp(-d2 / (2 * (170 * self.s) ** 2)))
        self.vol0 = self.vol0 + hal[..., None].astype(np.float32) * AIR_AMBER
        # faint long beam volume carrying the cone all the way down into its pool on the platform
        beam = R3.long_cone(self.rx, self.ry, self.zbuf, self.lamps[0][0], CAM_X, Y_PLAT)
        beam = cv2.GaussianBlur(beam, (0, 0), sigmaX=14 * self.s + 1, sigmaY=6 * self.s + 1)
        self.vol0 = self.vol0 + (beam * 0.26)[..., None] * AIR_AMBER
        self.beam0 = beam
        self.vol = scatter(lambda li, p: li != 0 and p[2] < ZSPLIT)
        # round 4: the left platform lamp gets its own small amber cone + head halation
        l1 = self.lamps[-1][0]
        lu1, lv1 = self.cam.p(l1[0], l1[1] - 0.05, l1[2])
        _, gv1 = self.cam.p(l1[0], Y_PLAT, l1[2])
        self.vol = self.vol + R4.small_cone(PW, PH, lu1, lv1, gv1, self.s, half_ang=0.55, amber=AIR_AMBER, strength=0.5)
        self.vol_f = scatter(lambda li, p: li != 0 and p[2] >= ZSPLIT)

    # ------------------------------------------------------------------ snow flakes
    def _flakes(self):
        rng = np.random.default_rng(77)
        W, H, cam = self.W, self.H, self.cam
        bands = [(1.6, 5.0, 520), (5.0, 14.0, 34000), (14.0, 40.0, 72000), (40.0, 90.0, 30000)]
        Z, U, Vv = [], [], []
        for z0, z1, n in bands:
            zz = (rng.random(n) * (z1 ** 3 - z0 ** 3) + z0 ** 3) ** (1 / 3)
            Z.append(zz)
            U.append(rng.uniform(-0.15, 1.15, n))
            Vv.append(rng.uniform(-0.3, 1.3, n))
        Z = np.concatenate(Z)
        U = np.concatenate(U)
        f = cam.f
        cxF, cyF = cam.cx - self.mx, cam.cy - self.my
        X = (U * W - cxF) * Z / f + CAM_X
        Ytop = ((cyF - (-0.3) * H) * Z / f)
        Ybot = np.maximum((cyF - 1.3 * H) * Z / f, Y_BED - 0.2)
        # extra population inside the foreground lamp's light cone: the cone visibly fills with snow
        nfree = len(Z)
        ccx, ccr = [np.zeros(nfree)], [np.zeros(nfree)]
        for (li, nc, R0) in ((0, 1400, 1.35), (len(self.lamps) - 1, 520, 1.1)):
            lx, ly, lz = self.lamps[li][0]
            rr = np.sqrt(rng.random(nc)) * R0
            ang = rng.uniform(0, 2 * np.pi, nc)
            X = np.concatenate([X, lx + rr * np.cos(ang)])
            Z = np.concatenate([Z, lz + rr * np.sin(ang) * 1.3])
            Ytop = np.concatenate([Ytop, np.full(nc, ly + 0.2)])
            Ybot = np.concatenate([Ybot, np.full(nc, max(ly - 3.6, Y_PLAT))])
            ccx.append(np.full(nc, lx))
            ccr.append(np.full(nc, R0))
        self.fcx = np.concatenate(ccx).astype(np.float32)
        self.fcr = np.concatenate(ccr).astype(np.float32)      # 0: free flake, >0: cone flake (wraps in x)
        self.fX = X.astype(np.float32)
        self.fY0 = (Ytop + rng.random(len(Z)) * (Ybot - Ytop)).astype(np.float32)
        self.fYt = Ytop.astype(np.float32)
        self.fYr = np.maximum(Ytop - Ybot, 0.1).astype(np.float32)
        self.fZ = Z.astype(np.float32)
        n = len(Z)
        self.fvy = rng.uniform(0.7, 1.3, n).astype(np.float32) * 1.15
        self.fswA = rng.uniform(0.05, 0.22, n).astype(np.float32)
        self.fswW = rng.uniform(0.6, 1.6, n).astype(np.float32)
        self.fswP = rng.uniform(0, 6.28, n).astype(np.float32)
        self.fsize = np.clip(rng.lognormal(np.log(0.0042), 0.55, n), 0.0014, 0.022).astype(np.float32)
        self.fglw = rng.uniform(1.5, 5.0, n).astype(np.float32)
        self.fglp = rng.uniform(0, 6.28, n).astype(np.float32)
        # large out-of-focus flakes right in front of the lens (screen space, slow drift)
        # (a handful of big soft discs spread over the whole frame, incl. the foreground platform/pine)
        rb = np.random.default_rng(4040)
        nb = 14
        self.bk_u = ((np.arange(nb) + rb.uniform(-0.35, 0.35, nb)) / nb * 1.3 - 0.15)[rb.permutation(nb)].astype(np.float32)
        self.bk_v = ((np.arange(nb) + rb.uniform(-0.3, 0.3, nb)) / nb * 1.3 - 0.15).astype(np.float32)
        self.bk_r = (20 + 40 * rb.random(nb)).astype(np.float32)          # radius at 1080p (40-120 px wide)
        self.bk_a = rb.uniform(0.3, 0.46, nb).astype(np.float32)
        self.bk_vy = (0.045 + 0.0012 * self.bk_r).astype(np.float32)      # frame heights per second
        self.bk_vx = rb.uniform(0.012, 0.035, nb).astype(np.float32)
        self.bk_ph = rb.uniform(0, 6.28, nb).astype(np.float32)
        self.bk_par = rb.uniform(0.8, 1.25, nb).astype(np.float32)
        self.bk_k = (rb.random(nb) < 0.45).astype(np.int32) + 1            # 1: rimmed disc, 2: soft disc
        # snow clumps sliding off pine branches: (X, Y, Z, start time)
        self.clumps = [(-5.9, 1.3, 12.0, 1.2), (13.3, Y_BED + 7.4, 25.0, 0.7), (-7.6, Y_BED + 8.3, 13.3, 2.4), (8.9, Y_BED + 5.6, 40.5, 3.6),
                       (15.6, Y_BED + 4.6, 24.8, 1.6)]
        self.clump_rng = [np.random.default_rng(900 + k).normal(0, 1, (60, 3)).astype(np.float32) for k in range(len(self.clumps))]

    def _glows(self):
        """Static additive light: signal lamp, distant street lights with small pools, vending spill."""
        PW, PH, cam, s = self.PW, self.PH, self.cam, self.s
        g = np.zeros((PH, PW, 3), np.float32)
        us, vs = C.grid(PW, PH)

        def ok(u, v, z, r=4):
            ui, vi = int(u), int(v)
            if not (0 <= ui < PW and 0 <= vi < PH):
                return 0.0
            return float(self.zbuf[vi, ui] > z - 1.0)
        # signal lamps
        for (dy, col, I) in ((4.05, np.array([0.25, 1.0, 0.55], np.float32), 1.6), (3.6, np.array([1.0, 0.15, 0.08], np.float32), 0.25)):
            x, y, z = 5.5, Y_BED + dy, 59.9
            u, v = cam.p(x, y, z)
            vis = ok(u, v, z)
            C.splat(g, np.array([u]), np.array([v]), max(1.1 * s, 0.7), col, 4.0 * I * vis)
            C.splat(g, np.array([u]), np.array([v]), 9.0 * s, col, 0.22 * I * vis)
        # street lights
        for (x, z) in self.streetlights:
            hx, hy = x - 0.6, Y_BED + 5.1
            u, v = cam.p(hx, hy, z)
            fa = float(self.fog_amt(z))
            vis = ok(u, v, z)
            C.splat(g, np.array([u]), np.array([v]), max(1.3 * s, 0.7), WARM * 1.2, 4.0 * vis * (1 - fa * 0.5))
            C.splat(g, np.array([u]), np.array([v]), 12.0 * s, WARM, 0.18 * vis * (1 - fa * 0.5))
            # pool of light on the snow below
            gu, gv = cam.p(hx, Y_BED, z)
            rx = cam.f * 3.2 / z
            ry = abs(cam.f * Y_BED / (z - 3.2) - cam.f * Y_BED / (z + 3.2)) / 2
            x0, x1 = int(max(gu - 3 * rx, 0)), int(min(gu + 3 * rx, PW))
            y0, y1 = int(max(gv - 3 * ry, 0)), int(min(gv + 3 * ry, PH))
            if x1 > x0 and y1 > y0:
                d2 = ((us[y0:y1, x0:x1] - gu) / rx) ** 2 + ((vs[y0:y1, x0:x1] - gv) / max(ry, 0.5)) ** 2
                occ = (self.zbuf[y0:y1, x0:x1] > z - 4).astype(np.float32)
                g[y0:y1, x0:x1] += (np.exp(-d2 * 1.5) * occ * 0.3 * (1 - fa * 0.6))[..., None] * WARM
        # warm window light spilling onto the snow in front of the houses
        for (wx, wy, wz, ww, br) in getattr(self, 'win_spill', []):
            gu, gv = cam.p(wx, Y_BED, wz - 1.1)
            rx = cam.f * (ww * 0.75 + 0.3) / wz
            ry = max(abs(cam.f * Y_BED / (wz - 2.2) - cam.f * Y_BED / wz) / 2, 0.8)
            x0, x1 = int(max(gu - 3 * rx, 0)), int(min(gu + 3 * rx, PW))
            y0, y1 = int(max(gv - 3 * ry, 0)), int(min(gv + 3 * ry, PH))
            if x1 > x0 and y1 > y0:
                d2 = ((us[y0:y1, x0:x1] - gu) / rx) ** 2 + ((vs[y0:y1, x0:x1] - gv) / ry) ** 2
                occ = (np.abs(self.zbuf[y0:y1, x0:x1] - (wz - 1.1)) < 3.0).astype(np.float32)
                g[y0:y1, x0:x1] += (np.exp(-d2 * 1.3) * occ * 0.16 * br)[..., None] * WIN
        self.vol_f = self.vol_f + g
        # vending machine spill (pulses gently in frame())
        vx, vz = BLD_X1 + 0.75, 23.95
        u, v = cam.p(vx + 0.1, Y_PLAT + 1.1, vz)
        vg = np.zeros((PH, PW, 3), np.float32)
        C.splat(vg, np.array([u]), np.array([v]), cam.f * 0.9 / vz, VEND, 0.08)
        gu, gv = cam.p(vx + 0.55, Y_PLAT, vz)
        rx = cam.f * 1.1 / vz
        ry = abs(cam.f * Y_PLAT / (vz - 1.3) - cam.f * Y_PLAT / (vz + 1.3)) / 2
        d2 = ((us - gu) / rx) ** 2 + ((vs - gv) / max(ry, 0.5)) ** 2
        vg += (np.exp(-d2 * 1.2) * 0.3)[..., None] * VEND + (np.exp(-d2 * 3.0) * 0.35)[..., None] * np.array([1.0, 0.28, 0.16], np.float32)
        self.vend_glow = vg.astype(np.float32)

    def _halos(self):
        """Volumetric halation of every emissive surface (windows, lamp heads, vending machine) in the
        snowy air, plus a blurred light map used to tint the flakes that pass in front of them."""
        PW, PH = self.PW, self.PH
        comp = self.L_far[..., :3].copy()
        for Lr in (self.L_gnd, self.L_midf, self.L_mid):
            comp = comp * (1 - Lr[..., 3:]) + Lr[..., :3] * Lr[..., 3:]
        for k in ('tree', 'bench', 'sign', 'lamp'):
            cvx = self.fgs[k]
            comp = comp * (1 - cvx.a[..., None]) + (cvx.rgb if k != 'lamp' else 0.0)
        lamp = self.fgs['lamp']
        q = 4

        def halo_of(c):
            lum = c.max(-1, keepdims=True)
            em = np.minimum(c * np.clip((lum - 0.95) / 0.8, 0, 1), 3.0)
            sm = cv2.resize(em, (PW // q, PH // q), interpolation=cv2.INTER_AREA)
            h1 = F.fast_blur(sm, 0.004 * PW / q)
            h2 = F.fast_blur(sm, 0.012 * PW / q)
            h3 = F.fast_blur(sm, 0.035 * PW / q)
            halo = (h1 * 0.35 + h2 * 0.55 + h3 * 0.5) * np.array([1.0, 0.85, 0.7], np.float32)
            return cv2.resize(halo, (PW, PH), interpolation=cv2.INTER_LINEAR) * 0.8, sm, h3
        # soft highlight cap for the deep view through the canopy: haze there may glow but never clip
        clum = comp.max(-1)
        us_, vs_ = C.grid(PW, PH)
        rxx = (us_ - self.cam.cx) / self.cam.f
        ryy = -(vs_ - self.cam.cy) / self.cam.f
        Yo, Xo = ryy * CAN_Z1, rxx * CAN_Z1 + CAM_X             # where each ray leaves the canopy tunnel
        tunnel = C.smoothstep(CAN_Y + 0.4, CAN_Y - 0.3, Yo) * C.smoothstep(CAN_X0 - 0.5, CAN_X0 + 0.5, Xo) *             C.smoothstep(CAN_X1 + 1.5, CAN_X1 + 0.5, Xo)
        capm = C.smoothstep(28.0, 46.0, self.zbuf) * C.smoothstep(0.75, 0.45, clum) * tunnel
        self.cap_mask = cv2.GaussianBlur(capm.astype(np.float32), (0, 0), 3.0 * self.s + 0.5)
        farm = C.smoothstep(ZSPLIT - 6, ZSPLIT + 6, self.zbuf)[..., None]
        hn, smn, h3n = halo_of(comp * (1 - farm))
        hf, smf, h3f = halo_of(comp * farm)
        hl, sml, h3l = halo_of(lamp.rgb)
        self.vol = self.vol + hn
        self.vol_f = self.vol_f + hf
        self.vol0 = self.vol0 + hl
        # light map for flakes: includes the volumetric cones (fog light) as well
        sm = smn + smf + sml
        h3 = h3n + h3f + h3l
        vq = cv2.resize(self.vol + self.vol_f + self.vol0, (PW // q, PH // q), interpolation=cv2.INTER_AREA)
        self.fglow = (F.fast_blur(sm, 0.008 * PW / q) * 0.9 + h3 * 0.5 + vq * 0.35).astype(np.float32)
        self.fglow_q = q

    def _veil(self):
        """Mid-depth curtain of snowfall between the station and the village: wind-blown sheets of
        denser snow (soft diagonal streaks) that drift across everything deeper than ~25 m."""
        PW, PH, W, H = self.PW, self.PH, self.W, self.H
        q = 4
        vw, vh = (2 * W) // q, H // q + 8
        n1 = C.fbm(vw, vh, 6.0, 4, seed=81)
        n2 = C.fbm(vw, vh, 14.0, 3, seed=82)
        v = 0.65 * n1 + 0.35 * n2
        # shear into slanted sheets (wind from the left, snow falling)
        sh = np.float32([[1, 0.45, 0], [0, 1, 0]])
        v = cv2.warpAffine(v.astype(np.float32), sh, (vw, vh), borderMode=cv2.BORDER_WRAP)
        v = cv2.GaussianBlur(v, (0, 0), sigmaX=2.5, sigmaY=0.8)
        v = np.clip((v - v.mean()) / (v.std() + 1e-6) * 0.5 + 0.5, 0, 1)
        self.veil_tex = np.concatenate([v, v], 1).astype(np.float32)
        self.veil_q = q
        # the foreground sprites are composited after the veil, so they must not cut holes in its mask
        zb = self.zbuf_bg
        m = C.smoothstep(18.0, 45.0, zb) * (1 - 0.5 * C.smoothstep(150.0, 400.0, zb))
        self.veil_mask = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0 * self.s)

    def _streaks(self):
        """Far snowfall veil: two wrapping tiles of fine motion-blurred streaks (thousands of distant
        flakes) that thicken the air over everything deeper than ~15 m - the house and trees on the
        right, the view through the canopy tunnel - for aerial depth."""
        s = self.s
        T = int(round(384 * s)) // 2 * 2
        self.stk_T = T
        tiles = []
        for k, (n, ln, seed) in enumerate(((1600, 4.5, 91), (2800, 2.6, 92))):
            rng = np.random.default_rng(seed)
            ss = 3
            big = np.zeros((T * ss, T * ss), np.float32)
            x = rng.uniform(0, T, n)
            y = rng.uniform(0, T, n)
            L_ = ln * s * rng.uniform(0.6, 1.4, n)
            it = rng.uniform(0.35, 1.0, n)
            dx, dy = 0.14, 1.0
            nrm = math.hypot(dx, dy)
            for i in range(n):
                for ox in (-T, 0, T):
                    for oy in (-T, 0, T):
                        x0, y0 = x[i] + ox, y[i] + oy
                        x1, y1 = x0 + dx / nrm * L_[i], y0 + dy / nrm * L_[i]
                        if max(x0, x1) < -2 or min(x0, x1) > T + 2 or max(y0, y1) < -2 or min(y0, y1) > T + 2:
                            continue
                        cv2.line(big, (int(x0 * ss), int(y0 * ss)), (int(x1 * ss), int(y1 * ss)), float(it[i]),
                                 max(int(round(ss * max(0.9 * s, 0.6))), 1), cv2.LINE_AA)
            tl = cv2.resize(big, (T, T), interpolation=cv2.INTER_AREA)
            tl = cv2.GaussianBlur(tl, (0, 0), 0.35 + 0.3 * k)
            tiles.append(tl.astype(np.float32))
        self.stk_tiles = tiles
        zb = cv2.erode(self.zbuf, np.ones((5, 5), np.uint8))      # incl. the foreground sprites: no streaks over the pine
        m = C.smoothstep(14.0, 40.0, zb) * (1 - 0.55 * C.smoothstep(150.0, 400.0, zb))
        self.stk_mask = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.0 * s + 0.3)

    def _draw_streaks(self, img, t, M, tx=0.0):
        W, H, T = self.W, self.H, self.stk_T
        m = cv2.warpAffine(self.stk_mask, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        acc = np.zeros((T, T), np.float32)
        for k, (tl, vy, amt, par) in enumerate(zip(self.stk_tiles, (0.55, 0.32), (0.16, 0.13), (0.5, 0.25))):
            oy = (t * vy * H) % T
            ox = (t * vy * 0.14 * H - tx * self.cam.f / 30.0 * par) % T
            A = np.float32([[1, 0, ox], [0, 1, oy]])
            acc += cv2.warpAffine(tl, A, (T, T), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * amt
        full = np.tile(acc, (H // T + 2, W // T + 2))[:H, :W]
        a = (full * m)[..., None]
        c = np.array([0.62, 0.7, 0.95], np.float32)
        return img + (c - img * 0.5) * a * 0.9

    def _draw_veil(self, img, t, M):
        W, H, q = self.W, self.H, self.veil_q
        vw = self.veil_tex.shape[1] // 2
        off = (t * 0.035 * W / q) % vw
        dy = (t * 0.02 * H / q)
        A = np.float32([[1, 0, -off], [0, 1, -dy % 4]])
        vs = cv2.warpAffine(self.veil_tex, A, (vw, self.veil_tex.shape[0] - 4), flags=cv2.INTER_LINEAR)
        wq, hq = W // q, H // q
        vs = cv2.resize(vs[:, : W // q + 1], (wq, hq), interpolation=cv2.INTER_LINEAR)
        # plate -> frame affine at 1/q resolution: only the translation scales (the old version divided
        # the whole matrix, which pasted a quarter-size copy of the depth mask into the top-left sky)
        Mq = M.copy()
        Mq[:, 2] = M[:, 2] / q
        if not hasattr(self, '_veil_mq'):
            self._veil_mq = cv2.resize(self.veil_mask, (self.PW // q, self.PH // q), interpolation=cv2.INTER_AREA)
        m = cv2.warpAffine(self._veil_mq, Mq, (wq, hq), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.resize((m * (0.09 + 0.15 * vs)).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        ys = np.nonzero(a.max(1) > 0.004)[0]
        if len(ys) == 0:
            return img
        y0, y1 = ys.min(), ys.max() + 1
        a = a[y0:y1, :, None]
        img = img.copy()
        c = FOG * 1.25 + np.array([0.03, 0.03, 0.04], np.float32)
        img[y0:y1] = img[y0:y1] + (c - img[y0:y1]) * a
        return img

    def _sparkles(self):
        """Static glints in the snow surface (ice crystals catching the lamps), twinkling coherently."""
        rng = np.random.default_rng(31)
        cam = self.cam
        n = 22000
        X = rng.uniform(-6.0, 2.1, n)
        Z = np.exp(rng.uniform(np.log(1.2), np.log(62), n))
        npool = 9000
        rr = np.sqrt(rng.random(npool)) * 4.5
        aa = rng.uniform(0, 2 * np.pi, npool)
        X = np.concatenate([X, np.clip(1.2 + rr * np.cos(aa) * 0.8, -6.0, 2.1)])
        Z = np.concatenate([Z, np.clip(7.5 + rr * np.sin(aa) * 1.4, 1.2, 62)])
        # round 4: extra cold glints in the blue-violet shadowed snow of the near-left platform
        nsh = 6000
        X = np.concatenate([X, rng.uniform(-6.0, -0.5, nsh)])
        Z = np.concatenate([Z, np.exp(rng.uniform(np.log(1.4), np.log(10.0), nsh))])
        # glitter along the platform edge (the wind-packed lip catches every lamp)
        ne = 7000
        X = np.concatenate([X, rng.uniform(1.55, 2.17, ne)])
        Z = np.concatenate([Z, np.exp(rng.uniform(np.log(1.5), np.log(40.0), ne))])
        n = len(X)
        under = (X > CAN_X0) & (X < CAN_X1) & (Z > CAN_Z0 - 0.5) & (Z < CAN_Z1 + 0.5)
        P = np.stack([X, np.full(n, Y_PLAT + 0.01), Z], -1).astype(np.float32)
        E = self.light_at(P, np.array([0, 1, 0], np.float32), vis0=self.sign_shadow(X, Z)).mean(-1)
        u, v = cam.p(X, Y_PLAT, Z)
        ui, vi = u.astype(int), v.astype(int)
        inb = (ui >= 0) & (ui < self.PW) & (vi >= 0) & (vi < self.PH)
        zb = np.zeros(n)
        zb[inb] = self.zbuf[vi[inb], ui[inb]]
        edge = (X > 1.55).astype(np.float32)
        keep = inb & ~under & (zb > Z - 0.5 - 0.02 * Z) & (rng.random(n) < np.clip(E * 0.28 - 0.04, 0, 0.4) * np.clip(12.0 / Z, 0.15, 1.0))
        Yk = np.full(int(keep.sum()), Y_PLAT + 0.01)
        # second population: the open snowfield right of the track (cool glints in the blue shadow)
        nf = 26000
        Xf = np.concatenate([rng.uniform(2.7, 30.0, nf - 8000), rng.uniform(2.3, 9.0, 8000)])
        Zf = np.exp(rng.uniform(np.log(4.0), np.log(70.0), nf))
        Pf = np.stack([Xf, np.full(nf, Y_BED + 0.01), Zf], -1).astype(np.float32)
        Ef = self.light_at(Pf, np.array([0, 1, 0], np.float32)).mean(-1)
        uf, vf = cam.p(Xf, Y_BED, Zf)
        ufi, vfi = uf.astype(int), vf.astype(int)
        inf_ = (ufi >= 0) & (ufi < self.PW) & (vfi >= 0) & (vfi < self.PH)
        zbf = np.zeros(nf)
        zbf[inf_] = self.zbuf[vfi[inf_], ufi[inf_]]
        keepf = inf_ & (zbf > Zf - 0.5 - 0.02 * Zf) & (rng.random(nf) < np.clip(Ef * 0.5 - 0.02, 0.0, 0.3) * np.clip(9.0 / Zf, 0.15, 1.0))
        # lower snow specular: only ~half the glints survive (painted snow is matte, a few glints read)
        # round 5: sparse glints, local to the lamp pools only (powder, not wet plastic)
        keep = keep & (rng.random(n) < 0.4)
        keepf = keepf & (rng.random(nf) < 0.3)
        Yk = np.full(int(keep.sum()), Y_PLAT + 0.01)
        self.sp_u = np.concatenate([u[keep], uf[keepf]]).astype(np.float32)
        self.sp_v = np.concatenate([v[keep], vf[keepf]]).astype(np.float32)
        Ek = np.concatenate([E[keep], Ef[keepf]])
        m = len(Ek)
        self.sp_I = (np.clip(Ek, 0.4, 2.0) * rng.uniform(0.6, 1.4, m)).astype(np.float32)
        self.sp_ph = rng.uniform(0, 6.28, m).astype(np.float32)
        self.sp_w = rng.uniform(1.0, 2.6, m).astype(np.float32)
        # a few bigger, brighter glints (ice crystals facing the light) among the fine sparkle
        self.sp_R = np.where(rng.random(m) < 0.14, 1.8, 1.0).astype(np.float32) * rng.uniform(0.8, 1.2, m).astype(np.float32)
        self.sp_z = np.concatenate([Z[keep], Zf[keepf]]).astype(np.float32)
        self.sp_x = np.concatenate([X[keep], Xf[keepf]]).astype(np.float32)
        self.sp_y = np.concatenate([Yk, np.full(int(keepf.sum()), Y_BED + 0.01)]).astype(np.float32)
        # warm glints inside the lamp pools, cold blue-white ice glints in the shadowed snow
        wk = np.clip((Ek - 0.1) / 0.5, 0, 1)[:, None]
        self.sp_col = (np.array([1.3, 1.55, 2.3], np.float32) * (1 - wk) + np.array([2.2, 1.9, 1.5], np.float32) * wk).astype(np.float32)

    def _draw_sparkles(self, img, t, M, ct=(0.0, 0.0, 0.0)):
        # slow, smooth twinkle (no frame-to-frame flicker): each glint swells and fades over ~2-6 s
        # stable glints (they only breathe slowly, never pop on/off between frames)
        tw = 0.75 + 0.25 * np.sin(self.sp_w * 0.35 * t + self.sp_ph)
        sel = tw > 0.02
        if not sel.any():
            return img
        u, v = self._proj(self.sp_x[sel], self.sp_y[sel], self.sp_z[sel], *ct)
        n = int(sel.sum())
        R = (max(0.95 * self.s, 0.55) * self.sp_R[sel]).astype(np.float32)
        col = self.sp_col[sel] * 0.85 * (1 + 0.25 * (self.sp_R[sel] > 1.4))[:, None]
        a = np.clip(tw[sel] * (0.55 + self.sp_I[sel] * 0.8), 0, 1).astype(np.float32)
        z0 = np.zeros(n, np.float32)
        L.splat_flakes(img, u.astype(np.float32), v.astype(np.float32), z0, z0, R, col, a, np.zeros(n, np.int32))
        return img

    def _cam_at(self, t):
        # lateral truck (camera slides right along the platform) with only a slight push: the near pine
        # races across the frame, the canopy drifts, the village barely moves
        u = C.ease_in_out_sine(t / DURATION)
        tx = -0.36 + 0.72 * u
        tz = 0.25 * u
        ty = 0.02 * math.sin(t * 0.9)
        return tx, ty, tz

    def _layer_affine(self, zref, tx, ty, tz):
        f = self.cam.f
        cxP, cyP = self.cam.cx, self.cam.cy
        cxF, cyF = cxP - self.mx, cyP - self.my
        sc = zref / max(zref - tz, 1e-3)
        ox = cxF - sc * (cxP + f * tx / zref)
        oy = cyF - sc * (cyP - f * ty / zref)
        return np.array([[sc, 0, ox], [0, sc, oy]], np.float32)

    def _warp(self, img, M):
        return cv2.warpAffine(img, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _warp_ground(self, tx, ty, tz):
        """Ground planes (platform at Y_PLAT, track bed / field at Y_BED) re-projected exactly for the
        moving camera (per-pixel plane homography), so everything standing on them stays planted."""
        W, H, cam = self.W, self.H, self.cam
        if not hasattr(self, '_gdu'):
            us, vs = C.grid(W, H)
            self._gdu = (us - (cam.cx - self.mx)).astype(np.float32)
            self._gdv = (vs - (cam.cy - self.my)).astype(np.float32)
            self._gsrc = np.ascontiguousarray(np.dstack([self.L_gnd, self.gnd_id]).astype(np.float32))
        res = []
        for Y0 in (Y_PLAT, Y_BED):
            mu, mv = L.ground_maps(self._gdu, self._gdv, np.float32(cam.f), np.float32(cam.cx), np.float32(cam.cy),
                                   np.float32(tx), np.float32(ty), np.float32(tz), np.float32(Y0))
            res.append(cv2.remap(self._gsrc, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        w = np.clip(res[0][..., 4:5], 0, 1)
        return res[0][..., :4] * w + res[1][..., :4] * (1 - w)

    def frame(self, t):
        W, H, cam, s = self.W, self.H, self.cam, self.s
        tx, ty, tz = self._cam_at(t)
        ct = (tx, ty, tz)
        Mf = self._layer_affine(400.0, tx, ty, tz)
        Mmf = self._layer_affine(ZREF_FAR, tx, ty, tz)
        Mm = self._layer_affine(ZREF_MID, tx, ty, tz)
        img = self._warp(self.L_far[..., :3], Mf)
        img = F.over_rgba(img, self._warp_ground(tx, ty, tz))
        img = F.over_rgba(img, self._warp(self.L_midf, Mmf))
        img = F.over_rgba(img, self._warp(self.L_mid, Mm))
        img = self._draw_veil(img, t, Mm)
        img = self._draw_wires(img, t, Mm, ct)
        img = np.ascontiguousarray(img, np.float32)
        Ml = None
        for (k, zr, x0, y0, rgba) in self.sprites:
            M = self._layer_affine(zr, tx, ty, tz)
            if k == 'lamp':
                Ml = M
            img = self._sprite_over(img, x0, y0, rgba, M)
        vsum = self._warp(self.vol + self.vend_glow, Mm) + self._warp(self.vol_f, Mmf) + self._warp(self.vol0, Ml)
        img = self._deep_cap(img, vsum, Mm)
        img = self._crossing(img, t, Mmf)
        img = np.ascontiguousarray(img, np.float32)
        img = self._draw_sparkles(img, t, Mm, ct)
        img = self._snow(img, t, tx, ty, tz)
        img = self._clumps(img, t, tx, ty, tz)
        img = self._draw_streaks(img, t, Mm, tx)
        # soft anime flare on the foreground lamp bulb
        lx, ly, lz = self.lamps[0][0]
        u, v = cam.p(lx, ly - 0.02, lz)
        lu = Ml[0, 0] * u + Ml[0, 2]
        lv = Ml[1, 1] * v + Ml[1, 2]
        img = img + self._flare_at(lu, lv)
        # round 5: warm glow of the lit snowy air around both lamp heads (painted, wide and soft)
        u2, v2 = cam.p(*self.lamps[-1][0])
        lu2, lv2 = Mm[0, 0] * u2 + Mm[0, 2], Mm[1, 1] * v2 + Mm[1, 2]
        img = img + R5.lamp_air_glow(W, H, [(lu, lv, WARM, 330 * s, 0.04), (lu2, lv2, WARM, 190 * s, 0.033)], s)
        # (cycle 7) s08 -> s10 light leak, built in-shot so it is an optical event on the lamp head: an
        # additive warm bloom that starts ~6 frames (0.25 s) before the cut at the end of the edit window
        # (source frame 96 = t 4.0 s) and peaks on the cut (assemble adds its own 0.30 leak on top)
        pk_ = C.smoothstep(3.75, 4.0, t) * C.smoothstep(4.3, 4.0, t)
        if pk_ > 1e-3:
            img = img + R5.lamp_air_glow(W, H, [(lu, lv + 20 * s, np.array([1.0, 0.8, 0.5], np.float32), 300 * s, 0.13 * pk_ ** 1.5),
                                                (lu, lv, np.array([1.0, 0.9, 0.72], np.float32), 80 * s, 0.2 * pk_ ** 1.5)], s)
        img = self._bloom(img, strength=0.45, halation=0.3)
        img = self.bk.frame(np.ascontiguousarray(img, np.float32), t, tx, vsum, cam.f)
        img = self._lens_flakes(img, t, tx, vsum)
        img = F.shoulder(img, 0.82, 0.3)
        img = img * self._paper()
        return F.finish_fast(img, t, sat=1.08, grain_amt=0.0022, vig=0.35, ca=0.0008)

    def _lens_flakes(self, img, t, tx, light):
        """Two or three huge defocused flakes a metre or two from the lens: they cross the frame at camera
        speed (true world positions, projected with the truck), the strongest depth cue in the shot."""
        W, H, f = self.W, self.H, self.cam.f
        # (X world rel. to camera start, Y, Z, fall m/s, radius m)
        LF = ((0.5, 0.35, 1.5, 0.2, 0.07), (-0.1, 0.05, 2.1, 0.12, 0.08), (0.9, 0.3, 1.25, 0.16, 0.06))
        xs, ys, rs, cs, als = [], [], [], [], []
        for i, (X, Y, Z, vy, r) in enumerate(LF):
            Yt = Y - vy * t + 0.03 * math.sin(0.7 * t + i)
            Xt = X + 0.05 * t + 0.02 * math.sin(0.5 * t + 2 * i)
            u = W * 0.54 + f * (Xt - tx) / Z
            v = H * 0.57 - f * Yt / Z
            R = f * r / Z
            if u < -R * 2 or u > W + R * 2:
                continue
            ui, vi = int(np.clip(u, 0, W - 1)), int(np.clip(v, 0, H - 1))
            wk = float(np.clip(light[vi, ui].max() * 3.0, 0, 1))
            c = np.array([0.95, 1.05, 1.35], np.float32) * (1 - wk) + np.array([1.6, 1.1, 0.55], np.float32) * wk
            xs.append(u); ys.append(v); rs.append(R); cs.append(c); als.append(0.27 + 0.06 * wk)
        if not xs:
            return img
        img = np.ascontiguousarray(img, np.float32)
        n = len(xs)
        BK.screen_discs(img, np.array(xs, np.float32), np.array(ys, np.float32), np.array(rs, np.float32),
                        np.full(n, 0.9, np.float32), np.array(cs, np.float32), np.array(als, np.float32),
                        np.array([0.3, 0.9, 0.1][:n], np.float32))
        return img

    def _paper(self):
        """Static paint/paper tooth (screen-locked, identical every frame): soft mottling + fine fibres."""
        if not hasattr(self, '_paper_m'):
            W, H = self.W, self.H
            rng = np.random.default_rng(1234)
            mot = cv2.resize(rng.random((H // 24 + 2, W // 24 + 2)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
            fib = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), sigmaX=2.2 * self.s + 0.3, sigmaY=0.5)
            tooth = cv2.GaussianBlur(rng.random((H, W)).astype(np.float32), (0, 0), 0.7)
            fib = (fib - fib.mean()) / (fib.std() + 1e-6)
            tooth = (tooth - tooth.mean()) / (tooth.std() + 1e-6)
            m = 1 + 0.012 * (mot - 0.5) + 0.006 * fib + 0.004 * tooth
            self._paper_m = m[..., None].astype(np.float32)
        return self._paper_m

    def _deep_cap(self, img, vsum, M):
        """Beyond the canopy the warm lamp scatter falls off into cool blue snow haze: compress highlights
        there toward ~0.78 and cool them, so the far end reads as a lit blue fog, not a clipped hole."""
        m = cv2.warpAffine(self.cap_mask, M, (self.W, self.H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        ys, xs = np.nonzero(m[::4, ::4] > 0.003)
        # warm scattered light replaces some of the blue night behind it (reads amber, never pink)
        cut = np.clip(vsum.max(-1, keepdims=True) * 1.6, 0, 0.6) * np.array([0.0, 0.22, 0.6], np.float32)
        img = img * (1 - cut) + vsum
        if len(ys) == 0:
            return img
        y0, y1 = max(ys.min() * 4 - 4, 0), min(ys.max() * 4 + 8, self.H)
        x0, x1 = max(xs.min() * 4 - 4, 0), min(xs.max() * 4 + 8, self.W)
        sub = img[y0:y1, x0:x1]
        img = img.copy()
        img[y0:y1, x0:x1] = self._cap_block(sub, vsum[y0:y1, x0:x1], m[y0:y1, x0:x1])
        return img

    def _cap_block(self, img, vs, m):
        """img already includes the scatter vs; in the deep region pull the warm scatter down and cool it,
        then softly cap highlights (~0.8) so lamps glow through blue haze instead of clipping."""
        a = m[..., None]
        vl = vs.mean(-1, keepdims=True)
        vcool = vl * np.array([0.7, 0.88, 1.25], np.float32)
        img = img - vs * (0.6 * a) + vcool * (0.3 * a)
        lum = img.max(-1, keepdims=True)
        k = 0.62
        comp = np.where(lum > k, k + (lum - k) / (1 + (lum - k) / 0.2), lum)
        capped = img * (comp / np.maximum(lum, 1e-4))
        return img * (1 - a) + capped * a

    def _flare_at(self, lu, lv):
        """The lamp flare is rendered once (padded plate) and translated with the tiny camera drift."""
        W, H = self.W, self.H
        if not hasattr(self, '_flare'):
            p = int(0.05 * W)
            self._fp = p
            self._flo = (lu, lv)
            self._flare = F.anime_flare(W + 2 * p, H + 2 * p, lu + p, lv + p, intensity=0.32, tint=(1.0, 0.72, 0.42),
                                        rays=6, ray_len=0.05 * W / (W + 2 * p), ghosts=0.0, halo=0.2, streak=0.35,
                                        glow=0.7, seed=8, center=(W / 2 + p, H / 2 + p))
        p = self._fp
        M = np.array([[1, 0, lu - self._flo[0] - p], [0, 1, lv - self._flo[1] - p]], np.float32)
        return cv2.warpAffine(self._flare, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _crossing(self, img, t, M):
        """Level crossing far down the line: two red lamps flashing alternately (~1 Hz)."""
        cam, s = self.cam, self.s
        z, x, y = 96.0, 5.4, Y_BED + 2.3
        ph = (t * 1.1) % 1.0
        for k, dx in enumerate((-0.35, 0.35)):
            on = C.smoothstep(0.0, 0.06, (ph + 0.5 * k) % 1.0) * C.smoothstep(0.5, 0.42, (ph + 0.5 * k) % 1.0)
            u, v = cam.p(x + dx, y, z)
            uf = M[0, 0] * u + M[0, 2]
            vf = M[1, 1] * v + M[1, 2]
            col = np.array([1.0, 0.12, 0.06], np.float32)
            C.splat(img, np.array([uf]), np.array([vf]), max(1.0 * s, 0.6), col, 3.0 * on)
            C.splat(img, np.array([uf]), np.array([vf]), 7.0 * s, col, 0.25 * on)
        return img

    def _tube_flicker(self, img, t, M):
        """The last fluorescent tube under the canopy is failing: it stutters off a few times."""
        fi = int(t * C.FPS)
        off = 0.0
        for (a, b) in ((1.55, 1.95), (3.6, 3.75)):
            if a <= t < b:
                h = (fi * 2654435761) % 1000 / 1000.0
                off = 1.0 if h < 0.6 else 0.25
        if off <= 0:
            return img
        x, y, z = 0.2, 1.38, 28.0
        cam = self.cam
        n = 9
        us, vs = cam.p(np.linspace(x - 0.3, x + 0.3, n), np.full(n, y - 0.05), np.full(n, z))
        uf = M[0, 0] * us + M[0, 2]
        vf = M[1, 1] * vs + M[1, 2]
        col = -WHITE_W * 0.9 * off
        C.splat(img, uf, vf, max(1.6 * self.s, 0.8), col, 1.0)
        C.splat(img, np.array([uf.mean()]), np.array([vf.mean() + 6 * self.s]), 30 * self.s, col, 0.05)
        return np.maximum(img, 0.0)

    def _bokeh(self, img, t, tx=0.0, light=None):
        """Big out-of-focus flakes right in front of the lens: soft bright discs drifting down across the
        whole frame with strong parallax (they sit ~2 m from the camera); amber where they cross a
        lamp's light (sampled from the scattered-light map), cool white-blue elsewhere."""
        W, H, s = self.W, self.H, self.s
        par = tx * self.cam.f / 2.2 / W * self.bk_par
        u = np.mod(self.bk_u + 0.15 + self.bk_vx * t + 0.012 * np.sin(0.6 * t + self.bk_ph) - par, 1.3) - 0.15
        v = np.mod(self.bk_v + 0.15 + self.bk_vy * t, 1.3) - 0.15
        R = self.bk_r * (W / 1920.0)
        uu, vv = u * W, v * H
        wk = np.zeros(len(u), np.float32)
        if light is not None:
            ui = np.clip(uu.astype(int), 0, W - 1)
            vi = np.clip(vv.astype(int), 0, H - 1)
            wk = np.clip(light[vi, ui].max(-1) * 2.2, 0, 1).astype(np.float32)
        wk = wk[:, None]
        col = np.array([0.95, 1.0, 1.18], np.float32) * (1 - wk) + np.array([1.4, 1.0, 0.5], np.float32) * wk
        n = len(u)
        z0 = np.zeros(n, np.float32)
        L.splat_flakes(img, uu.astype(np.float32), vv.astype(np.float32), z0, z0, R.astype(np.float32),
                       col.astype(np.float32), (self.bk_a * (1 + 0.2 * wk[:, 0])).astype(np.float32), self.bk_k)
        return img

    def _clumps(self, img, t, tx, ty, tz):
        """Snow clumps sliding off pine branches: a falling core that breaks into a drifting powder veil."""
        cam, f = self.cam, self.cam.f
        us, vs, rs, cs, als = [], [], [], [], []
        for k, (X0, Y0, Z0, t0) in enumerate(self.clumps):
            dt = t - t0
            if dt < 0 or dt > 2.4:
                continue
            N = self.clump_rng[k]
            n = len(N)
            fall = 0.5 * 7.0 * np.minimum(dt, 1.4) ** 2 + 4.0 * max(dt - 1.4, 0)
            spread = 0.05 + 0.5 * dt
            X = X0 + N[:, 0] * spread * 0.8 + 0.3 * dt
            Y = Y0 - fall * (0.55 + 0.45 * np.clip(1 - np.abs(N[:, 1]) * 0.3, 0, 1)) + N[:, 1] * spread * 0.4
            Z = Z0 + N[:, 2] * spread * 0.3
            ok = Y > Y_BED
            Zc = np.maximum(Z - tz, 0.5)
            u = (cam.cx - self.mx) + f * (X - CAM_X - tx) / Zc
            v = (cam.cy - self.my) - f * (Y - ty) / Zc
            core = np.arange(n) < 12
            r = f * np.where(core, 0.16, 0.09) / Zc * (1 + dt * 0.8 * ~core)
            a = np.where(core, 0.9 * np.clip(1.2 - dt, 0, 1), 0.35 * np.clip(dt * 3, 0, 1) * np.clip(1 - dt / 2.4, 0, 1))
            a = a * ok
            E = self.light_at(np.stack([X, Y, Z], -1).astype(np.float32))
            c = AMB_SNOW * 1.1 + np.array([0.08, 0.09, 0.12], np.float32) + E.mean(-1, keepdims=True) * 0.8
            fa = self.fog_amt(Z)[:, None]
            c = c * (1 - fa) + FOG * fa
            us.append(u); vs.append(v); rs.append(np.maximum(r, 0.7 * self.s)); cs.append(c); als.append(a)
        if not us:
            return img
        u = np.concatenate(us).astype(np.float32)
        v = np.concatenate(vs).astype(np.float32)
        z0 = np.zeros(len(u), np.float32)
        L.splat_flakes(img, u, v, z0, z0, np.concatenate(rs).astype(np.float32), np.concatenate(cs).astype(np.float32),
                       np.concatenate(als).astype(np.float32), np.zeros(len(u), np.int32))
        return img

    def _snow(self, img, t, tx, ty, tz):
        W, H, cam, s = self.W, self.H, self.cam, self.s
        f = cam.f
        X = self.fX + 0.35 * t
        cone = self.fcr > 0
        rel = np.mod(X - self.fcx + self.fcr, 2 * np.maximum(self.fcr, 1e-3)) - self.fcr
        X = np.where(cone, self.fcx + rel, X) + self.fswA * np.sin(self.fswW * t + self.fswP)
        ctaper = np.where(cone, np.clip((self.fcr - np.abs(rel)) / (0.25 * np.maximum(self.fcr, 1e-3)), 0, 1), 1.0)
        Y = self.fYt - np.mod(self.fYt - self.fY0 + self.fvy * t, self.fYr)
        Z = self.fZ
        # occlusion vs the static scene (plate camera)
        up = (cam.cx + f * (X - CAM_X) / Z).astype(np.int32)
        vp = (cam.cy - f * Y / Z).astype(np.int32)
        inb = (up >= 0) & (up < self.PW) & (vp >= 0) & (vp < self.PH)
        zb = np.full(len(Z), 1e4, np.float32)
        zb[inb] = self.zbuf[vp[inb], up[inb]]
        # no snow under the canopy
        shelter = (X > CAN_X0) & (X < CAN_X1 - 0.3) & (Z > CAN_Z0 + 0.4) & (Z < CAN_Z1) & (Y < CAN_Y)
        vis = inb & (Z < zb) & ~shelter
        Zc = np.maximum(Z - tz, 0.25)
        u = (cam.cx - self.mx) + f * (X - CAM_X - tx) / Zc
        v = (cam.cy - self.my) - f * (Y - ty) / Zc
        # velocity (px per frame, 0.5 shutter)
        dvy = f * self.fvy / Zc / C.FPS * 0.2
        dvx = f * (0.35 + self.fswA * self.fswW * np.cos(self.fswW * t + self.fswP)) / Zc / C.FPS * 0.2
        r = f * self.fsize / Zc
        coc = (0.009 + 0.014 * C.smoothstep(4.5, 1.5, Zc)) * f * np.abs(1 / Zc - 1 / 8.0)
        R = np.sqrt(r * r + coc * coc) + 0.35 * s
        R = np.maximum(R, 0.55 * max(s, 0.6))
        # lighting
        Pw = np.stack([X, Y, Z], -1).astype(np.float32)
        if not hasattr(self, '_lamp_arr'):
            cp = np.array([cone_params(l)[:4] for l in self.lamps], np.float32)
            self._lamp_arr = (np.array([l[0] for l in self.lamps], np.float32), np.array([l[1] for l in self.lamps], np.float32),
                              np.array([l[2] for l in self.lamps], np.float32), cp[:, 0].copy(), cp[:, 1].copy(),
                              cp[:, 2].copy(), cp[:, 3].copy())
        la = self._lamp_arr
        E, fs = VL.flake_light(np.ascontiguousarray(Pw), np.array([CAM_X + tx, ty, tz], np.float32), la[0], la[1], la[2],
                               la[3], la[4], la[5], la[6], np.float32(0.7))
        # flakes take on the colour of the light they pass through: sodium amber inside the lamp cones,
        # warm gold in front of the lit windows (screen-space glow map), cool blue-white in the open night
        # light confined to the cone volume near each lamp head: brightness fades with distance
        if not hasattr(self, '_lamp_pos'):
            self._lamp_pos = np.array([l[0] for l in self.lamps], np.float32)
        lpv = self._lamp_pos
        dmin = np.full(len(Z), 1e3, np.float32)
        for k in range(len(lpv)):
            dk = (X - lpv[k, 0]) ** 2 + (Y - lpv[k, 1]) ** 2 + (Z - lpv[k, 2]) ** 2
            dmin = np.minimum(dmin, dk)
        att = C.smoothstep(4.8, 1.0, np.sqrt(dmin)).astype(np.float32)
        # the post lamps' glints stay inside the visible beam (tighter than the lighting cone)
        tight = np.ones(len(Z), np.float32)
        for k in (0, len(lpv) - 1):
            dx_, dy_, dz_ = X - lpv[k, 0], lpv[k, 1] - Y, Z - lpv[k, 2]
            dk = np.sqrt(dx_ * dx_ + dy_ * dy_ + dz_ * dz_) + 1e-4
            nearest = np.abs(dk * dk - dmin) < 1e-4
            cone_k = C.smoothstep(math.cos(math.radians(27)), math.cos(math.radians(15)), dy_ / dk)
            tight = np.where(nearest, cone_k, tight)
        # canopy tubes only light the air under the roof, not the open night in front of it
        for k in range(len(lpv)):
            if lpv[k, 1] < CAN_Y and CAN_Z0 < lpv[k, 2] < CAN_Z1:
                dk2 = (X - lpv[k, 0]) ** 2 + (Y - lpv[k, 1]) ** 2 + (Z - lpv[k, 2]) ** 2
                nearest = np.abs(dk2 - dmin) < 1e-4
                tight = np.where(nearest & ((Z < CAN_Z0) | (X > CAN_X1)), 0.3 * tight, tight)
        att = att * tight
        E = E * att[:, None]
        fs = fs * att[:, None]
        Ed = E.mean(-1, keepdims=True)
        lit_k = C.smoothstep(0.35, 1.4, Ed[:, 0])
        R = R * (1 + 0.45 * lit_k * C.smoothstep(2.5, 5.0, Zc))
        r = r * (1 + 0.45 * lit_k * C.smoothstep(2.5, 5.0, Zc))
        wk = C.smoothstep(0.35, 1.4, Ed)                       # lamp tint only inside the lit cones
        Ew = Ed * np.array([1.0, 0.8, 0.48], np.float32) * 1.05 * wk
        fsd = fs * np.array([1.05, 0.85, 0.6], np.float32) * 0.9 * C.smoothstep(0.25, 1.2, fs.mean(-1, keepdims=True))
        # flakes glint as they tumble through the light (tiny facets flashing)
        gl = np.clip(np.sin(self.fglw * t * 2.0 + self.fglp), 0, 1) ** 24
        Ew = Ew * (1 + 2.5 * gl[:, None] * wk)
        gm = np.zeros((len(Z), 3), np.float32)
        gq = self.fglow_q
        gi = np.clip(vp // gq, 0, self.fglow.shape[0] - 1)
        gj = np.clip(up // gq, 0, self.fglow.shape[1] - 1)
        gm[inb] = self.fglow[gi[inb], gj[inb]]
        base = (AMB_SNOW * 1.05 + np.array([0.12, 0.14, 0.18], np.float32)) * (1 - wk * np.array([0.0, 0.2, 0.55], np.float32))
        col = base + Ew + fsd + gm * 0.22 * np.array([1.0, 0.92, 0.82], np.float32)
        cm = col.max(-1, keepdims=True)
        col = col / np.maximum(cm / 1.8, 1.0)
        fa = self.fog_amt(Z)[:, None]
        col = col * (1 - fa) + FOG * 1.1 * fa
        alpha = np.clip((r / R) ** 2 * 1.4, 0.02, 0.85)
        alpha = np.where(R > 2.5 * s, np.minimum(alpha, 0.42), alpha)
        bok = R > 5 * s
        alpha = np.where(bok, np.clip(0.7 * np.sqrt(np.clip(r / np.maximum(R, 1e-3), 0, 1)), 0.04, 0.2), alpha)
        alpha = np.clip(alpha * (1 + 0.7 * lit_k), 0, 0.95) * ctaper
        alpha = np.where(bok, np.minimum(alpha, 0.07), alpha)
        kind = bok.astype(np.int32) * 2
        # flakes crossing a lamp cone streak visibly (longer effective shutter where they are lit)
        # lit flakes stay round/short soft flakes (no dash streaks)
        stk = 1.0 - 0.45 * wk[:, 0]
        dvx = dvx * stk
        dvy = dvy * stk
        # near flakes (within a few metres of the lens) are soft round blobs drifting gently, not
        # straight diagonal dashes (which read as rain over the dark pine / trees)
        nearf = C.smoothstep(7.0, 2.0, Zc)
        dvx = dvx * (1 - 0.75 * nearf)
        dvy = dvy * (1 - 0.75 * nearf)
        R = R * (1 + 0.3 * nearf)
        # defocused discs: the shutter streak is lost in the blur (no pill shapes)
        dvx = np.where(bok, dvx * 0.25, dvx)
        dvy = np.where(bok, dvy * 0.25, dvy)
        sel = vis & (u > -R * 3) & (u < W + R * 3) & (v > -R * 3) & (v < H + R * 3)
        order = np.argsort(-Z[sel])
        idx = np.nonzero(sel)[0][order]
        img = np.ascontiguousarray(img, np.float32)
        L.splat_flakes(img, u[idx].astype(np.float32), v[idx].astype(np.float32), dvx[idx].astype(np.float32),
                       dvy[idx].astype(np.float32), R[idx].astype(np.float32), col[idx].astype(np.float32),
                       alpha[idx].astype(np.float32), kind[idx].astype(np.int32))
        return img
