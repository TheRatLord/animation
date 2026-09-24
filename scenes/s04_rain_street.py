"""s04_rain_street - night, heavy rain, narrow Tokyo side street with a crosswalk.
Slow push-in down the street; neon, vending machines, wet asphalt reflections, rain at several depths.
"""
import math
import os
import sys

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

from lib import core as C          # noqa: E402
from lib import fx as F            # noqa: E402
from lib import sky as S           # noqa: E402
import s04_rain_street_r as R      # noqa: E402
import s04_rain_street_art as A    # noqa: E402
import s04_rain_street_fx as FX    # noqa: E402
import s04_rain_street_art2 as A2   # noqa: E402
import s04_rain_street_detail as DT  # noqa: E402
import s04_rain_street_layers as LY  # noqa: E402
import s04_rain_street_v5 as V5     # noqa: E402
import s04_rain_street_fin as FN    # noqa: E402

DURATION = 5.0
hexc = A.hexc

# ----------------------------------------------------------------------------- world layout constants
ROAD_L, ROAD_R = -3.1, 3.5          # asphalt extent
X_CROSS0, X_CROSS1 = 21.0, 28.0     # cross street (Z range)
FOCUS = 21.0
SIGN_TEXT = {12.2: 'ラーメン', 9.0: '旅館', 17.2: '喫茶店', 20.5: 'カラオケ', 7.2: '酒場', 6.6: '洋食', 16.5: '焼鳥',
             18.5: 'くすり'}
ROAD_MARKING = False                # 止まれ is cut off by the frame bottom during the push-in: dropped
CARD_Z = 60.0                       # signs nearer than this get their own card plate
AMBIENT = np.array([0.050, 0.060, 0.115], np.float32)
PUD_T = 0.625                       # puddle threshold on the puddle noise (lower = more standing water)
R.MIRROR_K = 0.5                   # reflected heights squashed -> hanging signs land in frame, readable


FIG_POSES = 12                      # walk cycle drawings (on twos -> one full stride cycle per second)
FIG_CYCLE = FIG_POSES * 2.0 / C.FPS  # seconds per cycle (two steps)
FIG_SPEED = 1.0                     # m/s, walking away from camera into the conbini light
FIG_X, FIG_Z = 2.35, 19.6


def fig_state(t):
    k = int(math.floor(t * C.FPS / 2.0 + 1e-6)) % FIG_POSES
    return k, FIG_SPEED * t


def cam_path(t):
    u = C.ease_in_out_sine(t / DURATION)
    return 0.26 * u, 0.06 * u, 1.35 * u          # dX, dY, dZ (metres)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.cam = cam = R.Cam(W, H, f_rel=0.66, h=1.4, pp=(0.54, 0.6), margin=0.035, ss=2)
        self.rng = np.random.default_rng(404)
        self.lights = []
        self.flick = []
        self._layout()
        self._build()

    # ========================================================================= layout (pure data)
    def _layout(self):
        rng = self.rng
        # ---- buildings: (side, z0, z1, xface, height, style, ground, sign list)
        B = []
        # near block, left
        B += [dict(side=-1, z0=0.3, z1=8.0, x=-3.55, h=10.5, style='apt', ground='drug', seed=1),
              dict(side=-1, z0=8.0, z1=14.0, x=-3.45, h=17.0, style='bar', ground='bar', seed=2),
              dict(side=-1, z0=14.0, z1=21.0, x=-3.6, h=13.0, style='balcony', ground='entrance', seed=3)]
        # near block, right
        B += [dict(side=1, z0=0.3, z1=6.5, x=3.95, h=9.5, style='apt', ground='bar', seed=4),
              dict(side=1, z0=6.5, z1=13.0, x=3.85, h=14.0, style='apt', ground='shop', seed=5),
              dict(side=1, z0=13.0, z1=21.0, x=4.0, h=20.0, style='office', ground='bar', seed=6)]
        # corner buildings (far block starts at 28)
        B += [dict(side=-1, z0=28.0, z1=36.0, x=-3.5, h=15.0, style='bar', ground='bar', seed=7, corner=True),
              dict(side=1, z0=28.0, z1=40.0, x=3.9, h=8.5, style='apt', ground='conbini', seed=8, corner=True)]
        # far block: procedural
        for side in (-1, 1):
            z = 36.0 if side < 0 else 40.0
            k = 0
            while z < 230:
                wz = rng.uniform(6, 13)
                hh = rng.uniform(8, 20) * (1.0 if rng.random() < 0.85 else 1.35)
                st = rng.choice(['apt', 'bar', 'office', 'balcony', 'bar'])
                gr = rng.choice(['shop', 'bar', 'shutter', 'entrance', 'shop'])
                B.append(dict(side=side, z0=z, z1=z + wz, x=side * rng.uniform(3.5, 4.0) + (0.2 if side > 0 else -0.1),
                              h=hh, style=st, ground=gr, seed=100 + 13 * k + (side > 0) * 7))
                z += wz
                k += 1
        self.buildings = B

        # ---- vertical projecting signs: (side, z, y0, height, width, style)
        # spaced (in depth / height) so that no sign hides another sign's lettering from the camera path
        SG = [(-1, 9.0, 3.3, 3.2, 0.75, 0), (-1, 12.2, 2.6, 3.6, 0.8, 3), (-1, 16.5, 4.6, 5.5, 1.0, 2),
              (-1, 18.5, 2.6, 1.8, 0.7, 6), (1, 7.2, 3.4, 2.2, 0.85, 1), (1, 11.5, 3.6, 5.2, 0.9, 4),
              (1, 17.2, 5.3, 6.5, 1.1, 5), (1, 20.5, 2.6, 3.0, 0.7, 7), (-1, 6.6, 2.5, 2.4, 0.8, 7),
              (-1, 30.0, 3.0, 9.0, 1.2, 3), (-1, 33.5, 3.4, 6.0, 1.0, 0)]
        z = 40.0
        while z < 180:
            side = rng.choice([-1, 1])
            hh = rng.uniform(3, 8) * (1 + z / 150)
            SG.append((side, z, rng.uniform(3, 5), hh, rng.uniform(0.8, 1.3) * (1 + z / 200), int(rng.integers(8))))
            z += rng.uniform(3, 9)
        self.signs = SG

        # ---- utility poles: (X, Z, height, lamp)
        self.poles = [(3.35, 6.2, 11.0, False), (2.95, 13.0, 11.5, True), (-2.85, 22.5, 12.5, True),
                      (3.0, 26.0, 12.5, False), (3.3, 36.0, 11.5, True),
                      (-3.1, 52.0, 11.0, False), (3.4, 66.0, 11.0, True), (-3.1, 84.0, 11.0, True),
                      (3.4, 100.0, 11.5, False), (-3.1, 120.0, 11.0, True), (3.4, 142.0, 11.0, False),
                      (-3.1, 165.0, 11.0, True), (3.4, 190.0, 11.0, False)]

    # ========================================================================= build all plates
    def _build(self):
        cam = self.cam
        W, H = self.W, self.H
        self._make_lights()
        # fog colour field: brighter bluish-magenta glow toward the vanishing point
        xs, ys = cam.grid(1)
        d = np.sqrt(((xs - cam.pcx) / W) ** 2 + ((ys - cam.pcy) / H * 1.6) ** 2)
        base = np.array([0.10, 0.10, 0.22], np.float32)
        glow = np.array([0.34, 0.30, 0.52], np.float32)
        self.fogcol = (base + glow * np.exp(-d / 0.18)[..., None] * 0.95).astype(np.float32)
        fog = (self.fogcol, 38.0, 0.95, 12.0, 0.7)
        dof = (FOCUS, 10.0 * W / 1920.0)
        dof_n = (FOCUS, 3.5 * W / 1920.0)   # near set stays crisp: painted detail, not CG mush
        gl = (0.75, 0.2, (0.005, 0.015, 0.04))

        self.sky = self._sky()
        planes = []
        # far block (z >= 28 incl corners)
        cv = R.Canvas(cam)
        self._paint_buildings(cv, lambda b: b['z0'] >= 27.9)
        self._paint_signs(cv, lambda s: s[1] >= CARD_Z)
        self._paint_poles(cv, lambda p: p[1] >= 30.0)
        self._paint_wires(cv, (30.0, 1e9))
        self._paint_traffic(cv)
        self._paint_cones(cv, lambda z: z >= 30.0)
        self._mist_bands(cv)
        planes.append(R.bake(cv, self.lights, AMBIENT, fog=fog, dof=dof, name='far', glow=gl))
        del cv
        # near block walls + signs
        cv = R.Canvas(cam)
        self._defer_tenant = True          # near tenant boxes become cards (crisp copy, clean parallax)
        self._paint_buildings(cv, lambda b: b['z0'] < 27.9)
        occ = np.full(cv.a.shape, 1e9, np.float32)

        def acc_occ(c):
            m = c.a > 0.5
            zr = c.z / np.maximum(c.a, 1e-6)
            occ[m] = np.minimum(occ[m], zr[m])
        acc_occ(cv)
        planes.append(R.bake(cv, self.lights, AMBIENT, fog=fog, dof=dof_n, name='near', glow=gl))
        del cv
        # signs + standees: one cropped single-depth card each (clean parallax, no glyph smearing)
        cards = []
        for sg in [s_ for s_ in self.signs if s_[1] < CARD_Z]:
            cv = R.Canvas(cam)
            self._paint_signs(cv, lambda s_, sg=sg: s_ is sg)
            acc_occ(cv)
            c = LY.bake_card(cv, self.lights, AMBIENT, fog=fog, dof=None, glow=gl, name='sign%.1f' % sg[1])
            if c is not None:
                c.zkey = sg[1]
                cards.append(c)
            del cv
        for k in range(2):
            cv = R.Canvas(cam)
            self._paint_standsign(cv, only=k)
            acc_occ(cv)
            c = LY.bake_card(cv, self.lights, AMBIENT, fog=fog, dof=None, glow=gl, name='stand%d' % k)
            if c is not None:
                c.zkey = (9.2, 16.5)[k]
                cards.append(c)
            del cv
        for b in self.buildings:
            if b['style'] == 'bar' and b['z0'] < 27.9 and b['z0'] > 1.0:
                cv = R.Canvas(cam)
                self._tenant_boxes(cv, b)
                acc_occ(cv)
                c = LY.bake_card(cv, self.lights, AMBIENT, fog=fog, dof=None, glow=gl, name='tenant%d' % b['seed'])
                if c is not None:
                    c.zkey = b['z0'] + 0.3
                    cards.append(c)
                del cv
        cards.sort(key=lambda c: -c.zkey)
        self.cards = cards
        self.card_mirror = LY.merge_mirrors(cards, cam)
        # near objects: poles, wires, vending machines
        cv = R.Canvas(cam)
        self._paint_poles(cv, lambda p: p[1] < 30.0)
        self._paint_vending(cv)
        self._paint_lanterns(cv)
        self._paint_cones(cv, lambda z: z < 30.0)
        acc_occ(cv)
        planes.append(R.bake(cv, self.lights, AMBIENT, fog=fog, dof=dof_n, name='objects', glow=gl))
        del cv
        for p in planes:
            p.make_mirror()
        # walking figure: one cropped card per stride pose (animated on threes), re-projected per frame at its
        # walked distance, with its own reflection in the wet road
        self.fig_cards = []
        for k in range(FIG_POSES):
            cv = R.Canvas(cam)
            self._paint_figure(cv, ph=2 * np.pi * k / FIG_POSES)
            c = LY.bake_card(cv, self.lights, AMBIENT, fog=fog, dof=dof, glow=gl, name='fig%d' % k)
            self.fig_cards.append(c)
            del cv
        # overhead wires: own plates binned by depth (smooth depth -> clean re-projection during the push-in),
        # with occlusion by nearer walls/signs/poles baked in; far bin first
        for (lo, hi) in ((17.0, 30.0), (0.0, 17.0)):
            cv = R.Canvas(cam)
            self._paint_wires(cv, (lo, hi))
            zr = cv.z / np.maximum(cv.a, 1e-6)
            vis = (occ > zr - 0.25).astype(np.float32)
            vis = cv2.GaussianBlur(vis, (0, 0), 0.7)
            cv.alb *= vis[..., None]
            cv.emi *= vis[..., None]
            cv.n *= vis[..., None]
            cv.a *= vis
            cv.z *= vis
            planes.append(R.bake(cv, self.lights, AMBIENT, fog=fog, dof=dof, name='wires%d' % int(lo)))
            del cv
        del occ
        self.planes = planes
        self._ground(fog, dof)
        self._rain_setup()
        self._ripple_setup()
        self._bokeh_setup()

    # ------------------------------------------------------------------------- mist bands
    def _mist_bands(self, cv):
        """Low-lying mist layers drifting between the building rows: emissive-only haze painted into the far
        canvas where there is nothing opaque in front (sky gap and far street), banded in world height."""
        cam = self.cam
        s = cam.ss
        Hs, Ws = cv.H, cv.W
        ys, xs = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
        ys = ys / s
        xs = xs / s
        a = cv.a
        zr = np.where(a > 0.02, cv.z / np.maximum(a, 1e-4), 400.0)
        zr = cv2.GaussianBlur(zr, (0, 0), 6 * s)
        Y = cam.h - (ys - cam.pcy) * zr / cam.f
        n = C.fbm(Ws // 4, Hs // 4, 3.0, 5, seed=515)
        n = cv2.resize(n, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
        n2 = C.fbm(Ws // 8, Hs // 24, 5.0, 4, seed=516)
        n2 = cv2.resize(n2, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
        band = (np.exp(-((Y - 1.2) / 1.6) ** 2) * 0.9 + np.exp(-((Y - 9.5) / 1.3) ** 2) * 0.6 +
                np.exp(-((Y - 17.0) / 2.0) ** 2) * 0.45)
        band *= np.clip((n2 - 0.35) * 2.2, 0, 1) * (0.6 + 0.6 * n)
        dist = np.clip((zr - 30.0) / 60.0, 0, 1)
        col = np.array([0.30, 0.26, 0.46], np.float32)
        m = (band * dist * 0.22)[..., None] * col
        cv.emi += m * (1.0 - 0.6 * a[..., None])

    # ------------------------------------------------------------------------- lights
    def _make_lights(self):
        L = self.lights
        # vending machines (cool white, strong) - positions set in _paint_vending too
        for z in (4.95, 5.95, 6.95):
            L.append(dict(pos=(-1.5, 1.0, z), color=hexc('#dff0ff'), power=1.6, radius=0.9, wrap=0.4))
        # signs: coloured lights in front of each projecting sign
        for (side, z, y0, hh, w, st) in getattr(self, 'signs', []):
            if z > 70:
                continue
            col = hexc(A.SIGN_STYLES[st]['bg'] if A.SIGN_STYLES[st]['pe'] > 0 else A.SIGN_STYLES[st]['fg'])
            xf = side * 3.5 - side * 0.5
            L.append(dict(pos=(xf - side * 0.6, y0 + hh / 2, z - 0.3), color=col, power=0.6 * hh * w, radius=1.2,
                          wrap=0.5))
        # street lamps on poles (cool LED, spot down)
        for (X, Z, hh, lamp) in self.poles:
            if lamp and Z < 120:
                lx = X - np.sign(X) * 1.0
                L.append(dict(pos=(lx, 5.6, Z), color=hexc('#cfe8ff'), power=12.0, radius=0.4, dir=(0, -1, 0),
                              cone=(0.15, 0.7), wrap=0.3))
        # shop fronts / bars (warm), conbini (white)
        for b in self.buildings:
            if b['ground'] in ('bar', 'shop') and b['z0'] < 120:
                zc = (b['z0'] + b['z1']) / 2
                col = hexc('#ffb070') if b['ground'] == 'bar' else hexc('#ffd8a0')
                L.append(dict(pos=(b['x'] - b['side'] * 0.8, 1.6, zc), color=col, power=1.4, radius=1.5, wrap=0.5))
        # conbini: very bright white spill across the intersection
        for x in (5.0, 9.0, 13.0):
            L.append(dict(pos=(x, 1.8, 27.2), color=hexc('#eef6ff'), power=4.5, radius=1.5, wrap=0.5))
        L.append(dict(pos=(3.2, 1.8, 31.0), color=hexc('#eef6ff'), power=3.0, radius=1.5, wrap=0.5))
        # lanterns at the bar (right, z~14)
        for (xw, z) in [(3.85, 15.0), (3.85, 17.8), (-3.35, 10.2), (-3.35, 12.4), (-3.4, 30.5), (-3.4, 32.2)]:
            L.append(dict(pos=(xw - np.sign(xw) * 0.6, 2.0, z - 0.2), color=hexc('#ff5028'), power=0.35, radius=0.5,
                          wrap=0.5))
        L.append(dict(pos=(2.6, 0.7, 8.9), color=hexc('#fff0e0'), power=0.35, radius=0.4, wrap=0.4))
        L.append(dict(pos=(-2.3, 0.7, 16.2), color=hexc('#ff4040'), power=0.3, radius=0.4, wrap=0.4))
        # traffic light red
        L.append(dict(pos=(1.5, 5.0, 27.3), color=hexc('#ff2a1a'), power=0.8, radius=0.8, wrap=0.3))

    # ------------------------------------------------------------------------- sky
    def _sky(self):
        """Rain-night ceiling over the city: a low, structured nimbostratus deck lit from below by the
        city (pink-magenta undersides, violet bodies), darker gaps showing the deep navy upper sky, and ragged
        scud with crisp lit lower edges. Plate is wider than the frame so it can drift."""
        cam = self.cam
        PW, PH = cam.PW + int(0.08 * cam.W), cam.PH
        hz = cam.pcy / PH
        preset = dict(stops=[(0.0, '#080a26'), (0.2, '#141848'), (0.4, '#2a2468'), (0.58, '#553c8a'),
                             (0.78, '#9a5aa0'), (1.0, '#e08ab8')],
                      sun_glow='#ffb0c0', sun_glow_amt=0.1, below='#d890b8', band=('#e8a8c8', 0.35))
        sky = S.sky_gradient(PW, PH, preset, horizon=hz, variation=0.03, seed=4)
        ys = np.linspace(0, 1, PH, dtype=np.float32)[:, None]
        lit = np.clip(ys / hz, 0, 1) ** 1.2
        # deck: horizontally stretched billows; defined lobes via thresholded fbm with warped edges
        def field(seed, sx, sy, oct_):
            f = C.fbm(max(PW // sx, 8), max(PH // sy, 8), 4.0, oct_, seed=seed)
            return cv2.resize(f, (PW, PH), interpolation=cv2.INTER_CUBIC)
        base = field(41, 7, 2, 5)
        det = field(42, 2, 1, 4)
        v = base * 0.78 + det * 0.22
        dens = C.smoothstep(0.47, 0.53, v)
        # underside light: distance to the lower edge of each lobe (city glow hits the bottom)
        sh = max(2, int(PH * 0.012))
        below = np.vstack([dens[sh:], np.repeat(dens[-1:], sh, 0)])
        under = np.clip(dens - below, 0, 1)
        under = cv2.GaussianBlur(under, (0, 0), PH * 0.004) * 0.9
        sh2 = max(3, int(PH * 0.06))
        below2 = np.vstack([dens[sh2:], np.repeat(dens[-1:], sh2, 0)])
        broad = cv2.GaussianBlur(np.clip(dens - below2, 0, 1), (0, 0), PH * 0.025) * 1.4
        under = under * 0.5 + broad
        inner = cv2.GaussianBlur(dens, (0, 0), PH * 0.02)
        body = hexc('#4a3478') * (0.65 + 1.0 * lit[..., None]) * (0.75 + 0.35 * det[..., None])
        body = body + hexc('#b0609a') * (0.35 * lit * np.clip(1 - inner, 0, 1) ** 0.5)[..., None]
        gap = sky * 0.8
        out = gap * (1 - dens[..., None]) + body * dens[..., None]
        out = out + hexc('#ff9ad0') * (np.clip(under, 0, 1) * (0.25 + 0.75 * lit))[..., None] * 0.55
        # ragged scud below the deck: darker, crisp, stretched fragments with bright lower rims
        sc = field(43, 9, 2, 6)
        d2 = C.smoothstep(0.6, 0.64, sc * 0.85 + det * 0.15) * np.clip(1.25 - ys / hz, 0, 1)
        b2 = np.vstack([d2[sh:], np.repeat(d2[-1:], sh, 0)])
        rim = cv2.GaussianBlur(np.clip(d2 - b2, 0, 1), (0, 0), 1.0)
        out = out * (1 - 0.8 * d2[..., None]) + hexc('#1c1840') * (0.8 + 0.6 * lit[..., None]) * 0.8 * d2[..., None]
        out = out + hexc('#ffb0d8') * (rim * (0.3 + 0.7 * lit))[..., None] * 0.3
        # city light pollution glow rising from the street canyon
        xs = np.linspace(0, 1, PW, dtype=np.float32)[None, :]
        cx = cam.pcx / PW
        glow = np.exp(-((xs - cx) / 0.18) ** 2) * lit ** 2
        out = out + hexc('#c070b0') * (glow * 0.45)[..., None]
        return out.astype(np.float32)

    # ------------------------------------------------------------------------- buildings
    def _paint_buildings(self, cv, pred):
        cam = self.cam
        bl = [b for b in self.buildings if pred(b)]
        bl.sort(key=lambda b: -b['z0'])
        for b in bl:
            rng = np.random.default_rng(b['seed'])
            side, z0, z1, xf, hh = b['side'], b['z0'], b['z1'], b['x'], b['h']
            wm = z1 - z0
            znear = max(z0, 1.0)
            ppm = min(cam.px_per_m(znear), 260.0 * cam.ss * cam.W / 1920)
            # far facades get less density (cheaper)
            T = A.facade(rng, wm, hh, ppm, b['style'], ground=b['ground'], lit_p=0.5 if b['style'] != 'office' else 0.4,
                         near=z0 < 1.0, spandrel=z0 < 28.0)
            if z0 < 1.0:
                # the nearest walls frame the shot: keep them dark and cool (night, wet) so the neon reads
                T['alb'] = (T['alb'] * np.array([0.6, 0.56, 0.84], np.float32)).astype(np.float32)
            if side < 0:
                P = [(xf, hh, z0), (xf, hh, z1), (xf, 0, z1), (xf, 0, z0)]
                n = (1, 0, 0)
            else:
                P = [(xf, hh, z1), (xf, hh, z0), (xf, 0, z0), (xf, 0, z1)]
                n = (-1, 0, 0)
            # near-side wall facing the camera (visible above lower neighbours / across the intersection)
            depth = 14.0 if not b.get('corner') else 16.0
            if z0 > 1.0:
                sw = depth
                sppm = min(cam.px_per_m(z0), 260.0 * cam.ss * cam.W / 1920)
                srng = np.random.default_rng(b['seed'] + 999)
                if b.get('corner'):
                    Ts = A.facade(srng, sw, hh, sppm, 'bar' if side < 0 else 'apt',
                                  ground='bar' if side < 0 else 'conbini', lit_p=0.55)
                else:
                    # plain shutter (no signboard): only a sliver of this wall shows at the corner, and a
                    # partial sign there reads as a torn white edge next to the neighbour's fascia
                    Ts = A.facade(srng, sw, hh, sppm, 'apt', ground='shutter_plain', lit_p=0.3)
                    Ts['alb'] *= 0.8
                if side < 0:
                    Ps = [(xf - sw, hh, z0), (xf, hh, z0), (xf, 0, z0), (xf - sw, 0, z0)]
                else:
                    Ps = [(xf, hh, z0), (xf + sw, hh, z0), (xf + sw, 0, z0), (xf, 0, z0)]
                cv.quad(Ps, Ts['alb'], Ts['a'], Ts['emi'], normal=(0, 0, -1))
                if b.get('corner') and side > 0:
                    self._conbini_front(cv, xf, z0)
            cv.quad(P, T['alb'], T['a'], T['emi'], normal=n)
            if z0 < 70:
                self._clutter(cv, b)
            DT.add(cv, b, cam)
            # rooftop details: water tank / railing / billboard silhouettes
            if rng.random() < 0.6 and 20 < z0 < 80:
                self._rooftop(cv, rng, side, xf, z0, z1, hh)

    def _box(self, cv, s, xf, zc, y0, w, h, d, Tfront, Tside):
        """Box protruding from a wall at xf (side s) centred at zc: front face + camera-facing side."""
        xo = xf - s * d
        z0, z1 = zc - w / 2, zc + w / 2
        if s < 0:
            Pf = [(xo, y0 + h, z0), (xo, y0 + h, z1), (xo, y0, z1), (xo, y0, z0)]
            Ps = [(xf, y0 + h, z0), (xo, y0 + h, z0), (xo, y0, z0), (xf, y0, z0)]
        else:
            Pf = [(xo, y0 + h, z1), (xo, y0 + h, z0), (xo, y0, z0), (xo, y0, z1)]
            Ps = [(xo, y0 + h, z0), (xf, y0 + h, z0), (xf, y0, z0), (xo, y0, z0)]
        cv.quad(Ps, Tside['alb'], Tside['a'], Tside.get('emi'), normal=(0, 0, -1))
        cv.quad(Pf, Tfront['alb'], Tfront['a'], Tfront.get('emi'), normal=(-s, 0, 0))

    def _clutter(self, cv, b):
        """3D facade clutter: AC units, balconies with laundry, pipes, small lit tenant boxes."""
        cam = self.cam
        s, z0, z1, xf, hh = b['side'], b['z0'], b['z1'], b['x'], b['h']
        rng = np.random.default_rng(b['seed'] + 4242)
        zmid = max((z0 + z1) / 2, 2.0)
        ppm = min(cam.px_per_m(max(z0, 2.0)), 300 * cam.ss * cam.W / 1920)
        fh = 3.0 if b['style'] != 'office' else 3.6
        floors = int((hh - 3.4) // fh)
        # balconies ------------------------------------------------------------------------
        if b['style'] == 'balcony':
            for f in range(floors):
                ys = 3.4 + f * fh
                d = 0.95
                L = z1 - z0 - 0.4
                T = A.tex(L * ppm * 0.5, 1.15 * ppm, A.hexc('#a8aeb8'))
                hp, wp = T['a'].shape
                # frosted panel railing with top rail and posts
                T['alb'][:] = A.hexc('#8a93a4') * np.linspace(0.8, 1.05, hp, dtype=np.float32)[:, None, None]
                A.rect(T['alb'], 0, 0, wp, hp * 0.07, A.hexc('#d8dde4'))
                for xk in np.linspace(0, wp, int(L / 1.8) + 2):
                    A.rect(T['alb'], xk - 1, 0, xk + 1, hp, A.hexc('#5a606c'))
                A.rect(T['alb'], 0, hp * 0.88, wp, hp, A.hexc('#6a707c'))
                # laundry / futon over some railings
                for k in range(int(rng.integers(0, 3))):
                    lx = rng.uniform(0.05, 0.8) * wp
                    c = A.hexc(rng.choice(['#e8e0d0', '#7aa0d8', '#e89aa8', '#f0f0f0', '#a8c890']))
                    A.rect(T['alb'], lx, hp * 0.02, lx + rng.uniform(0.06, 0.15) * wp, hp * rng.uniform(0.5, 0.8), c)
                zz0, zz1 = z0 + 0.2, z1 - 0.2
                xo = xf - s * d
                if s < 0:
                    Pf = [(xo, ys + 1.15, zz0), (xo, ys + 1.15, zz1), (xo, ys, zz1), (xo, ys, zz0)]
                else:
                    Pf = [(xo, ys + 1.15, zz1), (xo, ys + 1.15, zz0), (xo, ys, zz0), (xo, ys, zz1)]
                # slab underside edge + end panel facing the camera
                Te = A.tex(d * ppm, 1.3 * ppm, A.hexc('#7a8292'))
                A.rect(Te['alb'], 0, Te['a'].shape[0] * 0.88, Te['a'].shape[1], Te['a'].shape[0], A.hexc('#b0b6c0'))
                if zz0 > 1.0:
                    Pe = ([(xf, ys + 1.15, zz0), (xo, ys + 1.15, zz0), (xo, ys - 0.15, zz0), (xf, ys - 0.15, zz0)]
                          if s < 0 else
                          [(xo, ys + 1.15, zz0), (xf, ys + 1.15, zz0), (xf, ys - 0.15, zz0), (xo, ys - 0.15, zz0)])
                    cv.quad(Pe, Te['alb'], Te['a'], None, normal=(0, 0, -1))
                cv.quad(Pf, T['alb'], T['a'], None, normal=(-s, 0, 0))
        # AC units -------------------------------------------------------------------------
        n_ac = int(rng.integers(1, 3)) * max(1, floors)
        for k in range(n_ac):
            zc = rng.uniform(z0 + 0.6, z1 - 0.6)
            yb = 3.4 + rng.integers(0, max(1, floors)) * fh + rng.uniform(0.1, 1.8)
            if b['style'] == 'balcony':
                continue
            Tf = A.tex(0.8 * ppm, 0.6 * ppm, A.hexc('#c4c8ce'))
            hp, wp = Tf['a'].shape
            cv2.circle(Tf['alb'], (int(wp * 0.38), int(hp * 0.5)), max(1, int(hp * 0.34)),
                       tuple(float(c) for c in A.hexc('#4a4e58')), -1, cv2.LINE_AA)
            for j in range(4):
                cv2.circle(Tf['alb'], (int(wp * 0.38), int(hp * 0.5)), max(1, int(hp * 0.08 * (j + 1))),
                           tuple(float(c) for c in A.hexc('#8a8e98')), 1, cv2.LINE_AA)
            A.rect(Tf['alb'], wp * 0.75, hp * 0.15, wp * 0.92, hp * 0.85, A.hexc('#9a9ea8'))
            Ts = A.tex(0.3 * ppm, 0.6 * ppm, A.hexc('#9aa0a8'))
            self._box(cv, s, xf, zc, yb, 0.8, 0.6, 0.3, Tf, Ts)
            # bracket + drain hose
            Tb = A.tex(max(2, 0.03 * ppm), 0.5 * ppm, A.hexc('#3a3c44'))
            cv.sprite(xf - s * 0.28, yb - 0.5, zc - 0.3, 0.03, 0.5, Tb['alb'], Tb['a'], None)
        # pipes ------------------------------------------------------------------------------
        for k in range(int(rng.integers(1, 3))):
            zc = rng.uniform(z0 + 0.3, z1 - 0.3)
            Tp = A.tex(max(2, 0.12 * ppm), (hh - 3.6) * ppm * 0.25, A.hexc('#8a9098'))
            xx = np.linspace(-1, 1, Tp['a'].shape[1], dtype=np.float32)[None, :, None]
            Tp['alb'] *= 0.75 + 0.35 * (1 - xx ** 2)
            # downpipes run from the roof to the ground-floor fascia top (never across the shop sign)
            cv.sprite(xf - s * 0.1, 3.3, zc, 0.11, hh - 3.6, Tp['alb'], Tp['a'], None, normal=(-s * 0.7, 0, -0.7))
        # small lit tenant boxes stacked on the building edge (bar buildings) -----------------
        if b['style'] == 'bar' and not (z0 < 27.9 and getattr(self, '_defer_tenant', False)):
            self._tenant_boxes(cv, b)

    def _tenant_boxes(self, cv, b):
        cam = self.cam
        s, z0, z1, xf, hh = b['side'], b['z0'], b['z1'], b['x'], b['h']
        rng = np.random.default_rng(b['seed'] + 4343)
        ppm = min(cam.px_per_m(max(z0, 2.0)), 300 * cam.ss * cam.W / 1920)
        fh = 3.0 if b['style'] != 'office' else 3.6
        floors = int((hh - 3.4) // fh)
        if True:
            zc = z0 + 0.35
            for f in range(max(1, floors)):
                y = 3.6 + f * fh
                st = int(rng.integers(8))
                Tf = A.sign_panel(rng, 0.8 * ppm, 0.55 * ppm, style=st, vertical=False, count=2, bright=0.8)
                Ts = A.sign_panel(rng, 0.25 * ppm, 0.55 * ppm, style=st, vertical=True, count=1, bright=0.8)
                self._box(cv, s, xf, zc + 0.4, y, 0.8, 0.55, 0.25, Tf, Ts)

    def _conbini_front(self, cv, xf, z0):
        """Bright convenience-store front on the corner, facing the camera across the intersection."""
        cam = self.cam
        w, hgt = 11.0, 3.4
        ppm = cam.px_per_m(z0)
        T = A.tex(w * ppm, hgt * ppm, hexc('#e8eef4'))
        h, wp = T['a'].shape
        P = lambda m: m * ppm
        E = T['emi']
        # colour stripes band (generic store livery)
        A.rect(T['alb'], 0, 0, wp, P(0.22), hexc('#f4f4f4'))
        A.rect(E, 0, 0, wp, P(0.22), hexc('#ffffff') * 2.2)
        for i, c in enumerate(['#1aa24a', '#f08a1c', '#1a64d0']):
            A.rect(T['alb'], 0, P(0.22 + 0.1 * i), wp, P(0.32 + 0.1 * i), hexc(c))
            A.rect(E, 0, P(0.22 + 0.1 * i), wp, P(0.32 + 0.1 * i), hexc(c) * 2.0)
        # glass front with interior
        gl = np.zeros((h, wp), np.float32)
        A.rect(gl, P(0.2), P(0.62), wp - P(0.2), h - P(0.05), 1.0)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        inter = hexc('#f2f8ff') * (2.6 + 0.6 * yy)
        rng = np.random.default_rng(77)
        m = np.zeros((h, wp), np.float32)
        x = P(0.4)
        while x < wp - P(0.6):
            ww = P(rng.uniform(0.8, 1.6))
            A.rect(m, x, h - P(rng.uniform(1.2, 1.7)), x + ww, h - P(0.05), 1.0)
            for k in range(4):
                yk = h - P(0.3 + 0.35 * k)
                A.rect(m, x, yk, x + ww, yk + P(0.05), 0.0)
            x += ww + P(rng.uniform(0.4, 0.9))
        prod = A.noise(wp, h, max(8, wp // 12), 5, 2)
        shelf_col = hexc('#ff9a6a') * (0.6 + 0.8 * prod[..., None]) + hexc('#6ac0ff') * (1 - prod[..., None]) * 0.6
        inter = inter * (1 - m[..., None]) + shelf_col * 1.4 * m[..., None]
        for i in range(8):
            xm = P(0.2) + (wp - P(0.4)) * i / 7
            A.rect(gl, xm - P(0.04), P(0.62), xm + P(0.04), h, 0.25)
        E[:] = E + inter * gl[..., None]
        T['alb'] = T['alb'] * (1 - gl[..., None]) + 0.2 * gl[..., None]
        zz = z0 - 0.02
        cv.quad([(xf + 0.3, hgt, zz), (xf + 0.3 + w, hgt, zz), (xf + 0.3 + w, 0, zz), (xf + 0.3, 0, zz)],
                T['alb'], T['a'], T['emi'], normal=(0, 0, -1))
        # the side of the store along our street: also glass
        w2 = 9.0
        T2 = A.tex(w2 * ppm * 0.5, hgt * ppm * 0.5, hexc('#e8eef4'))
        h2, wp2 = T2['a'].shape
        g2 = np.zeros((h2, wp2), np.float32)
        A.rect(g2, 0, h2 * 0.2, wp2 * 0.7, h2, 1.0)
        yy2 = np.linspace(0, 1, h2, dtype=np.float32)[:, None, None]
        T2['emi'] = (hexc('#f2f8ff') * (2.0 + 0.5 * yy2) * g2[..., None]).astype(np.float32)
        A.rect(T2['emi'], 0, 0, wp2, h2 * 0.06, hexc('#ffffff') * 1.8)
        for i, c in enumerate(['#1aa24a', '#f08a1c', '#1a64d0']):
            A.rect(T2['emi'], 0, h2 * (0.06 + 0.03 * i), wp2, h2 * (0.09 + 0.03 * i), hexc(c) * 1.8)
        xs2 = xf - 0.02
        cv.quad([(xs2, hgt, z0 + w2), (xs2, hgt, z0), (xs2, 0, z0), (xs2, 0, z0 + w2)],
                T2['alb'], T2['a'], T2['emi'], normal=(-1, 0, 0))

    def _rooftop(self, cv, rng, side, xf, z0, z1, hh):
        cam = self.cam
        k = rng.random()
        zc = z0 + 1.5
        ppm = cam.px_per_m(zc)
        if k < 0.45:
            # water tank on legs
            w, h = 2.2, 1.6
            T = A.tex(w * ppm, (h + 1.0) * ppm, hexc('#6a6e7a'))
            hp, wp = T['a'].shape
            a = np.zeros((hp, wp), np.float32)
            A.rect(a, 0, 0, wp, hp * 0.62, 1.0)
            for lx in (0.1, 0.45, 0.8):
                A.rect(a, wp * lx, hp * 0.62, wp * lx + max(1, wp * 0.06), hp, 1.0)
            T['a'] = a
            T['alb'] *= np.linspace(1.1, 0.8, hp, dtype=np.float32)[:, None, None]
            X = xf + side * rng.uniform(2.0, 4.5)
            cv.sprite(X, hh, zc + 2, w, h + 1.0, T['alb'], T['a'], None)
        elif k < 0.8:
            # rooftop billboard facing the camera
            w, h = rng.uniform(3, 6), rng.uniform(1.5, 3.0)
            T = A.sign_panel(rng, w * ppm, h * ppm, style=int(rng.choice([0, 1, 2, 5, 6])), vertical=False,
                             count=int(rng.integers(3, 6)))
            X = xf - side * (w / 2 + rng.uniform(1.0, 3.0))
            X = xf + side * (w / 2 + rng.uniform(0.5, 2.5))
            # frame legs
            L = A.tex(w * ppm, 1.2 * ppm, hexc('#2a2c34'))
            hp, wp = L['a'].shape
            a = np.zeros((hp, wp), np.float32)
            for lx in np.linspace(0.05, 0.9, 4):
                A.rect(a, wp * lx, 0, wp * lx + max(1, wp * 0.03), hp, 1.0)
            A.rect(a, 0, hp * 0.4, wp, hp * 0.45, 1.0)
            cv.sprite(X, hh, zc + 1, w, 1.2, L['alb'], a, None)
            cv.sprite(X, hh + 1.2, zc + 1, w, h, T['alb'], T['a'], T['emi'])
        else:
            # antenna mast
            w, h = 0.15, rng.uniform(3, 6)
            T = A.tex(max(2, w * ppm), h * ppm, hexc('#303440'))
            X = xf + side * rng.uniform(1, 4)
            cv.sprite(X, hh, zc + 1, w, h, T['alb'], T['a'], None)
            # red aviation light
            e = A.tex(max(3, 0.3 * ppm), max(3, 0.3 * ppm), hexc('#ff3020'))
            e['emi'][:] = hexc('#ff3020') * 3.0
            cv.sprite(X, hh + h, zc + 1, 0.3, 0.3, e['alb'], None, e['emi'])

    # ------------------------------------------------------------------------- signs
    def _paint_signs(self, cv, pred):
        cam = self.cam
        sl = [s for s in self.signs if pred(s)]
        sl.sort(key=lambda s: -s[1])
        for i, (side, z, y0, hh, w, st) in enumerate(sl):
            rng = np.random.default_rng(int(z * 100) + st)
            xw = self._wall_x(side, z)
            ppm = min(cam.px_per_m(z), 400 * cam.ss * cam.W / 1920)
            T = A.sign_panel(rng, w * ppm, hh * ppm, style=st, vertical=True, bright=0.8 if z < 12 else (1.0 if z < 60 else 0.8),
                             text=SIGN_TEXT.get(z))
            if rng.random() < 0.35:
                A.neon_outline(T, hexc(A.SIGN_STYLES[(st + 3) % 8]['fg']), gain=2.5, inset=0.04, width=0.025)
            gap = 0.25
            if side < 0:
                x0, x1 = xw + gap, xw + gap + w
            else:
                x0, x1 = xw - gap - w, xw - gap
            depth = 0.3
            # side edge (casing), facing the street centre
            xe = x1 if side < 0 else x0
            Te = A.tex(max(2, depth * ppm), hh * ppm, hexc('#20222a'))
            Te['emi'][:] = np.asarray(T['emi'].mean((0, 1)), np.float32) * 0.15
            if side < 0:
                cv.quad([(xe, y0 + hh, z), (xe, y0 + hh, z + depth), (xe, y0, z + depth), (xe, y0, z)],
                        Te['alb'], Te['a'], Te['emi'], normal=(1, 0, 0))
            else:
                cv.quad([(xe, y0 + hh, z + depth), (xe, y0 + hh, z), (xe, y0, z), (xe, y0, z + depth)],
                        Te['alb'], Te['a'], Te['emi'], normal=(-1, 0, 0))
            # brackets
            Tb = A.tex(max(2, gap * ppm), max(2, 0.08 * ppm), hexc('#3a3c44'))
            for yb in (y0 + 0.3, y0 + hh - 0.3):
                xa, xb = (xw, x0) if side < 0 else (x1, xw)
                cv.quad([(xa, yb + 0.04, z + 0.15), (xb, yb + 0.04, z + 0.15), (xb, yb - 0.04, z + 0.15),
                         (xa, yb - 0.04, z + 0.15)], Tb['alb'], Tb['a'], None)
            cv.quad([(x0, y0 + hh, z), (x1, y0 + hh, z), (x1, y0, z), (x0, y0, z)], T['alb'], T['a'], T['emi'],
                    normal=(0, 0, -1))

    def _wall_x(self, side, z):
        for b in self.buildings:
            if b['side'] == side and b['z0'] <= z < b['z1']:
                return b['x']
        return side * 3.7

    # ------------------------------------------------------------------------- poles & wires
    def _paint_poles(self, cv, pred):
        cam = self.cam
        pl = [p for p in self.poles if pred(p)]
        pl.sort(key=lambda p: -p[1])
        for (X, Z, hh, lamp) in pl:
            rng = np.random.default_rng(int(Z * 10))
            ppm = min(cam.px_per_m(Z), 500 * cam.ss * cam.W / 1920)
            dia = 0.34
            T = A.pole_tex(rng, max(3, dia * ppm), hh * ppm, hh)
            if Z > 14:   # mid-distance poles: read as dark silhouettes against the lit haze
                T['alb'] = (T['alb'] * 0.42).astype(np.float32)
            cv.sprite(X, 0, Z, dia, hh, T['alb'], T['a'], None, normal=(-np.sign(X) * 0.6, 0, -0.8))
            # crossarms (perpendicular to the street -> face the camera) with diagonal braces + insulators
            arm_col = hexc('#1e2028')
            for ya, wa in ((hh - 0.45, 2.0), (hh - 1.25, 1.6), (hh - 2.0, 1.2)):
                Ta = A.tex(wa * ppm, max(2, 0.11 * ppm), arm_col)
                hp_, wp_ = Ta['a'].shape
                Ta['alb'][: max(1, hp_ // 3)] = hexc('#6a6e7a')          # lit top edge
                cv.sprite(X, ya, Z - 0.05, wa, 0.11, Ta['alb'], Ta['a'], None)
                for sgn in (-1, 1):
                    P0 = (X + sgn * wa * 0.32, ya, Z - 0.06)
                    P1 = (X, ya - 0.45, Z - 0.06)
                    cv.polyline3d(np.array([P0, P1]), 0.035, hexc('#2e3038'))
                for xi in np.linspace(-wa / 2 + 0.08, wa / 2 - 0.08, 4 if wa > 1.5 else 3):
                    Ti = A.tex(max(2, 0.09 * ppm), max(3, 0.17 * ppm), hexc('#d8dce4'))
                    hi_, wi_ = Ti['a'].shape
                    for k in range(3):   # ribbed porcelain
                        Ti['alb'][int(hi_ * (0.2 + 0.28 * k)):int(hi_ * (0.3 + 0.28 * k))] *= 0.55
                    Ti['emi'][:] = hexc('#ffd0f0') * 0.08
                    cv.sprite(X + xi, ya + 0.055, Z - 0.1, 0.09, 0.17, Ti['alb'], Ti['a'], Ti['emi'])
            # transformer cans (one or two) on a bracket, with bushings and a drop lead
            if rng.random() < 0.75 or Z < 32:
                ncan = 2 if Z < 32 else 1
                for k in range(ncan):
                    xo = X + (-np.sign(X) * 0.5 if k == 0 else np.sign(X) * 0.45)
                    Tt = A.tex(0.55 * ppm, 0.95 * ppm, hexc('#5a606a'))
                    hp_, wp_ = Tt['a'].shape
                    xx = np.linspace(-1, 1, wp_, dtype=np.float32)[None, :, None]
                    yy = np.linspace(0, 1, hp_, dtype=np.float32)[:, None, None]
                    Tt['alb'] *= (0.55 + 0.5 * np.clip(1 - (xx + 0.35) ** 2, 0, 1)) * (0.8 + 0.25 * yy)
                    for r_ in (0.12, 0.5, 0.88):
                        Tt['alb'][int(hp_ * r_):int(hp_ * r_) + max(1, hp_ // 40)] *= 0.6
                    Tt['alb'][: max(1, hp_ // 14)] = hexc('#5a5e68')
                    Tt['emi'][:] = (hexc('#ff80d0') * 0.05 * np.clip(-xx, 0, 1) +
                                    hexc('#80d8ff') * 0.05 * np.clip(xx, 0, 1))
                    cv.sprite(xo, hh - 3.3 - 0.1 * k, Z - 0.22, 0.55, 0.95, Tt['alb'], Tt['a'], Tt['emi'])
                    for bx in (-0.12, 0.12):
                        Tb_ = A.tex(max(2, 0.06 * ppm), max(2, 0.14 * ppm), hexc('#c8ccd4'))
                        cv.sprite(xo + bx, hh - 2.36 - 0.1 * k, Z - 0.23, 0.06, 0.14, Tb_['alb'], Tb_['a'], None)
                    lead = np.array([(xo + 0.12, hh - 2.25 - 0.1 * k, Z - 0.23), (xo + 0.3, hh - 1.8, Z - 0.1),
                                     (X + 0.4, hh - 1.25, Z - 0.06)])
                    cv.polyline3d(lead, 0.018, hexc('#15161c'))
                Tb2 = A.tex(1.2 * ppm, max(2, 0.08 * ppm), hexc('#2a2c34'))
                cv.sprite(X, hh - 2.45, Z - 0.21, 1.2, 0.08, Tb2['alb'], Tb2['a'], None)
            # step bolts up the pole
            for k in range(8):
                yb = 2.6 + k * 0.45
                sgn = 1 if k % 2 else -1
                Ts_ = A.tex(max(2, 0.2 * ppm), max(2, 0.03 * ppm), hexc('#50545e'))
                cv.sprite(X + sgn * 0.2, yb, Z - 0.12, 0.2, 0.03, Ts_['alb'], Ts_['a'], None)
            # junction box
            Tj = A.tex(0.32 * ppm, 0.42 * ppm, hexc('#5a5e68'))
            Tj['alb'][: max(1, Tj['a'].shape[0] // 10)] *= 1.5
            cv.sprite(X - np.sign(X) * 0.1, hh - 4.6, Z - 0.2, 0.32, 0.42, Tj['alb'], Tj['a'], None)
            # a few small boxes / signs on the pole
            Tb = A.tex(0.3 * ppm, 0.5 * ppm, hexc('#6a6e78'))
            cv.sprite(X, 4.3, Z - 0.18, 0.3, 0.5, Tb['alb'], Tb['a'], None)
            if lamp:
                # street lamp arm + LED head
                lx = X - np.sign(X) * 1.0
                Ta = A.tex(1.1 * ppm, max(2, 0.06 * ppm), hexc('#3a3c44'))
                cv.sprite((X + lx) / 2, 5.75, Z - 0.1, 1.1, 0.06, Ta['alb'], Ta['a'], None)
                Th = A.tex(0.5 * ppm, max(2, 0.14 * ppm), hexc('#50545e'))
                hp, wp = Th['a'].shape
                A.rect(Th['emi'], 0, hp * 0.55, wp, hp, hexc('#e0f0ff') * 6.0)
                cv.sprite(lx, 5.62, Z - 0.12, 0.5, 0.14, Th['alb'], Th['a'], Th['emi'])

    def _wire_segments(self):
        """Overhead lines as drooping catenary BUNDLES strung pole-to-pole along each kerb, and across the street
        at (nearly) constant depth, so they read as sagging curves over the street instead of lines radiating
        from one point. (p0, p1, sag, diameter)."""
        if hasattr(self, '_wsegs'):
            return self._wsegs
        rng = np.random.default_rng(55)
        poles = sorted(self.poles, key=lambda p: p[1])
        left = [p for p in poles if p[0] < 0]
        right = [p for p in poles if p[0] > 0]
        segs = []

        def bundle(p0, p1, n, sag, dia, dy=0.1, dx=0.0):
            for j in range(n):
                segs.append(((p0[0] + dx * j, p0[1] - dy * j, p0[2]), (p1[0] + dx * j, p1[1] - dy * j, p1[2]),
                             sag * (1 + 0.06 * j), dia))
        # along each kerb: power lines on the crossarms + fat telecom cables lower down
        for side_poles in (left, right):
            X0 = side_poles[0][0]
            chain = [(X0, -18.0, side_poles[0][2])] + side_poles
            for a, b in zip(chain[:-1], chain[1:]):
                span = abs(b[1] - a[1])
                for k in range(6):
                    ya = a[2] - 0.5 - 0.8 * (k // 3)
                    yb = b[2] - 0.5 - 0.8 * (k // 3)
                    off = (-0.7 + 0.45 * (k % 3))
                    segs.append(((a[0] + off, ya, a[1]), (b[0] + off, yb, b[1]),
                                 0.025 * span + rng.uniform(0.1, 0.35), 0.012 if k < 3 else 0.018))
                for k in range(2):
                    segs.append(((a[0] - np.sign(a[0]) * (0.2 + 0.15 * k), 6.0 - 0.35 * k, a[1]),
                                 (b[0] - np.sign(b[0]) * (0.2 + 0.15 * k), 6.1 - 0.35 * k, b[1]),
                                 0.04 * span + rng.uniform(0.1, 0.3), 0.05 - 0.015 * k))
        # across the street from every pole: a bundle to a bracket on the opposite facade, same depth
        for (X, Z, hh, _) in poles:
            xt = -np.sign(X) * 3.7
            n = int(rng.integers(2, 5))
            bundle((X, hh - 1.3, Z), (xt, hh - 2.6 - rng.uniform(0, 1.0), Z + rng.uniform(-0.6, 0.6)), n,
                   rng.uniform(0.7, 1.2), 0.016, dy=rng.uniform(0.09, 0.16))
            # short service drops to the facades on the same side, drooping
            for k in range(2):
                zt = Z + rng.uniform(1.5, 5.0) * (1 if k == 0 else -1)
                segs.append(((X, hh - 1.5 - 0.3 * k, Z), (np.sign(X) * 3.6, rng.uniform(5.0, 6.8), zt),
                             rng.uniform(0.35, 0.6), 0.011))
        # extra facade-to-facade bundles across the street (telecom / cable TV), roughly perpendicular
        trng = np.random.default_rng(909)
        for zc in (8.6, 11.2, 15.5, 18.2, 24.0, 31.5, 34.5, 44.0):
            z0 = zc + trng.uniform(-0.4, 0.4)
            z1 = zc + trng.uniform(-0.8, 0.8)
            y0 = trng.uniform(5.4, 8.6) if zc > 10 else trng.uniform(4.6, 5.4)
            y1 = y0 + trng.uniform(-0.8, 0.6)
            n = int(trng.integers(2, 5))
            bundle((-3.5, y0, z0), (3.9, y1, z1), n, trng.uniform(0.6, 1.3) * (0.8 if zc < 10 else 1.0),
                   0.014 + 0.012 * (trng.random() < 0.4), dy=trng.uniform(0.07, 0.14))
        self._wsegs = segs
        return segs

    def _paint_wires(self, cv, zsel):
        wire_col = hexc('#101218')
        grng = np.random.default_rng(4040 + int(zsel[0]))
        for (p0, p1, sag, dia) in self._wire_segments():
            zmid = 0.5 * (p0[2] + p1[2])
            if not (zsel[0] <= zmid < zsel[1]):
                continue
            n = 96
            t = np.linspace(0, 1, n)
            # catenary-ish droop (cosh profile normalised to the requested mid-span sag)
            u = 2 * t - 1
            c = 2.2
            sagc = (np.cosh(c) - np.cosh(c * u)) / (np.cosh(c) - 1)
            P = np.stack([p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t - sag * sagc,
                          p0[2] + (p1[2] - p0[2]) * t], 1)
            keep = P[:, 2] > 0.6
            if keep.sum() < 2:
                continue
            cv.polyline3d(P[keep], dia, wire_col)
            if zmid > 60:
                continue
            # neon-lit underside: a thin continuous specular line, pink toward the left signs, cyan to the right
            Pu = P[keep].copy()
            Pu[:, 1] -= dia * 0.3
            xm = float(np.clip((Pu[:, 0].mean() + 3.5) / 7.4, 0, 1))
            tint = hexc('#ff5fb0') * (1 - xm) + hexc('#50e0ff') * xm
            tint = tint * 0.6 + hexc('#e8eeff') * 0.4
            cv.polyline3d(Pu, dia * 0.45, tint * 0.5, emi=tint * grng.uniform(0.35, 0.6), min_px=0.45, alpha=0.7)
            # rain beads hanging under the wire
            V5.wire_drops(cv, P[keep], dia, tint, grng)

    # ------------------------------------------------------------------------- vending machines
    def _paint_vending(self, cv):
        cam = self.cam
        rng = np.random.default_rng(31)
        xw = -3.5
        depth = 0.75
        brands = [hexc('#e4002b'), hexc('#f0f0f0'), hexc('#0a58c8')]
        zs = [4.5, 5.5, 6.5]
        for i in range(len(zs) - 1, -1, -1):
            z0 = zs[i]
            z1 = z0 + 0.98
            hh = 1.83
            ppm = min(cam.px_per_m(z0), 700 * cam.ss * cam.W / 1920)
            xf = xw + depth
            T = A2.vending_front(rng, 0.98 * ppm, hh * ppm, brands[i])
            T['alb'] = (T['alb'] * 0.35).astype(np.float32)
            T['emi'] = (T['emi'] * 0.5).astype(np.float32)   # self-lit panel: keep it from washing out
            V5.vend_extras(T, rng, brands[i])
            cv.quad([(xf, hh, z0), (xf, hh, z1), (xf, 0.02, z1), (xf, 0.02, z0)], T['alb'], T['a'], T['emi'],
                    normal=(1, 0, 0))
            # side panel facing the camera (only the first one is really visible)
            Ts = A2.vending_side(rng, depth * ppm, hh * ppm, brands[i])
            cv.quad([(xw, hh, z0), (xf, hh, z0), (xf, 0.02, z0), (xw, 0.02, z0)], Ts['alb'], Ts['a'], Ts['emi'],
                    normal=(0, 0, -1))
            # top
            Tt = A.tex(depth * ppm, ppm, brands[i] * 0.6)
            cv.quad([(xw, hh, z1), (xf, hh, z1), (xf, hh, z0), (xw, hh, z0)], Tt['alb'], Tt['a'], None,
                    normal=(0, 1, 0))
            # wet specular edges on the cabinet corners, catching the neon (front-near vertical + top edges)
            cv.polyline3d(np.array([(xf, 0.03, z0 - 0.005), (xf, hh, z0 - 0.005)]), 0.018, hexc('#e8f0ff'),
                          emi=hexc('#ffd0f0') * 1.1, min_px=0.8)
            cv.polyline3d(np.array([(xf, hh + 0.005, z0), (xf, hh + 0.005, z1)]), 0.016, hexc('#e8f0ff'),
                          emi=hexc('#e8f4ff') * 0.9, min_px=0.8)
            cv.polyline3d(np.array([(xw, hh + 0.005, z0 - 0.005), (xf, hh + 0.005, z0 - 0.005)]), 0.014,
                          hexc('#e8f0ff'), emi=hexc('#d0e8ff') * 0.8, min_px=0.8)

    # ------------------------------------------------------------------------- figure with umbrella
    def _paint_figure(self, cv, ph=0.0, X=FIG_X, Z=FIG_Z):
        """Lone figure with a clear vinyl umbrella, back-lit by the convenience store, walking away from
        camera. ph = stride phase (0..2pi covers one left + one right step). Seen from behind: the swing leg
        bends and kicks its heel up (the pale sole flashes), the body bobs and sways, the bag swings against the
        stride, the umbrella bobs a beat late and rocks with the shoulders."""
        cam = self.cam
        ppm = cam.px_per_m(Z) * 2.0
        w, h = 1.5, 2.5
        wp, hp = int(w * ppm), int(h * ppm)
        q = 4
        bob = 0.03 * math.cos(2 * ph) - 0.012
        sway = 0.024 * math.sin(ph)
        ub = 0.034 * math.cos(2 * ph - 0.7)
        tilt = 0.07 * math.sin(ph - 0.5)
        P0 = lambda x, y: (int((wp / 2 + x * ppm) * q), int((hp - y * ppm) * q))
        P = lambda x, y: P0(x + sway, y + bob)

        def PU(x, y):
            dx, dy = x, y - 1.3
            ca, sa = math.cos(tilt), math.sin(tilt)
            return P0(dx * ca - dy * sa + sway, 1.3 + dx * sa + dy * ca + bob + ub)

        def poly(m, pts, fn=None):
            fn = fn or P
            cv2.fillPoly(m, [np.array([fn(*p) for p in pts], np.int32)], 255, cv2.LINE_AA, shift=2)

        def seg(m, a_, b_, wa, wb):
            (xa, ya), (xb, yb) = a_, b_
            d = np.array([yb - ya, -(xb - xa)], np.float64)
            d /= max(np.hypot(*d), 1e-6)
            pts = [(xa + d[0] * wa, ya + d[1] * wa), (xb + d[0] * wb, yb + d[1] * wb),
                   (xb - d[0] * wb, yb - d[1] * wb), (xa - d[0] * wa, ya - d[1] * wa)]
            cv2.fillPoly(m, [np.array([P0(*p) for p in pts], np.int32)], 255, cv2.LINE_AA, shift=2)
        body = np.zeros((hp, wp), np.uint8)
        sole = np.zeros((hp, wp), np.uint8)
        # coat: slightly flared hem that swings with the stride
        hem = 0.02 * math.sin(ph)
        poly(body, [(-0.19, 1.40), (-0.08, 1.45), (0.08, 1.45), (0.2, 1.40), (0.23, 1.0), (0.27 + hem, 0.6),
                    (-0.26 + hem, 0.6), (-0.22, 1.0)])
        for side, lx in ((-1, -0.1), (1, 0.1)):
            lift = max(0.0, math.sin(ph) * side)
            li = lift ** 0.85
            hip = (lx + sway, 0.64 + bob)
            fx = lx + 0.02 * side * li
            fy = 0.25 * li                                  # heel kicks up behind as the leg swings through
            knee = (lx * 0.95 + sway * 0.5, 0.34 + 0.07 * li)
            ank = (fx, fy + 0.07 - 0.03 * li)
            seg(body, hip, knee, 0.068, 0.056)
            seg(body, knee, ank, 0.056, 0.042)
            if li < 0.15:
                poly(body, [(fx - 0.065, 0.075), (fx + 0.065, 0.075), (fx + 0.07, 0.0), (fx - 0.07, 0.0)], fn=P0)
            else:
                # the lifted shoe shows its pale sole to the camera
                poly(body, [(fx - 0.06, fy + 0.1), (fx + 0.06, fy + 0.1), (fx + 0.058, fy + 0.0),
                            (fx - 0.058, fy + 0.0)], fn=P0)
                poly(sole, [(fx - 0.05, fy + 0.07), (fx + 0.05, fy + 0.07), (fx + 0.048, fy + 0.01),
                            (fx - 0.048, fy + 0.01)], fn=P0)
        poly(body, [(-0.05, 1.42), (0.05, 1.42), (0.05, 1.5), (-0.05, 1.5)])
        cv2.ellipse(body, P(0.0, 1.58), (int(0.1 * ppm * q), int(0.12 * ppm * q)), 0, 0, 360, 255, -1, cv2.LINE_AA, 2)
        # bag in the right hand swings against the stride
        bsw = 0.055 * math.sin(ph + 0.6)
        poly(body, [(0.2 + bsw, 1.05), (0.36 + bsw, 1.05), (0.35 + bsw * 1.5, 0.78), (0.21 + bsw * 1.5, 0.78)])
        poly(body, [(0.19, 1.3), (0.24, 1.3), (0.24 + bsw, 1.05), (0.2 + bsw, 1.05)])
        poly(body, [(-0.2, 1.38), (-0.1, 1.28), (0.02, 1.36), (0.0, 1.43), (-0.12, 1.37)])
        # umbrella
        ux, uy, R_, Hc = 0.02, 2.02, 0.56, 0.3
        can = np.zeros((hp, wp), np.uint8)
        n_rib = 8
        ang = np.linspace(np.pi, 0, 60)
        top = [(ux + R_ * np.cos(a), uy + Hc * np.sin(a)) for a in ang]
        edge = []
        for k in range(n_rib):
            xa = ux + R_ - 2 * R_ * k / n_rib
            xb = ux + R_ - 2 * R_ * (k + 1) / n_rib
            for tt in np.linspace(0, 1, 8):
                xx = xa + (xb - xa) * tt
                edge.append((xx, uy - 0.04 + 0.04 * np.sin(np.pi * tt)))
        poly(can, top + edge, fn=PU)
        ribs = np.zeros((hp, wp), np.uint8)
        for k in range(n_rib + 1):
            xk = ux + R_ - 2 * R_ * k / n_rib
            pts = []
            for tt in np.linspace(0, 1, 12):
                xx = ux + (xk - ux) * np.sin(tt * np.pi / 2)
                yy = uy + Hc * np.cos(tt * np.pi / 2) - 0.02
                pts.append(PU(xx, yy))
            cv2.polylines(ribs, [np.array(pts, np.int32)], False, 255, max(1, int(0.008 * ppm)), cv2.LINE_AA, 2)
        rim = np.zeros((hp, wp), np.uint8)
        cv2.polylines(rim, [np.array([PU(*p) for p in top], np.int32)], False, 255, max(1, int(0.012 * ppm)),
                      cv2.LINE_AA, 2)
        cv2.polylines(rim, [np.array([PU(*p) for p in edge], np.int32)], False, 255, max(1, int(0.01 * ppm)),
                      cv2.LINE_AA, 2)
        shaft = np.zeros((hp, wp), np.uint8)
        cv2.line(shaft, PU(ux, uy + Hc + 0.06), PU(0.0, 1.3), 255, max(1, int(0.018 * ppm)), cv2.LINE_AA, 2)
        cv2.line(shaft, PU(0.0, 1.3), PU(-0.03, 1.22), 255, max(1, int(0.03 * ppm)), cv2.LINE_AA, 2)
        f = lambda m: m.astype(np.float32) / 255.0
        bf, cf, rf, rbf, sf, so = f(body), f(can), f(rim), f(ribs), f(shaft), f(sole)
        alpha = np.clip(np.maximum(bf, np.maximum(sf, rf * 0.95)) + cf * 0.45 + rbf * 0.5, 0, 1)
        alb = np.ones((hp, wp, 3), np.float32) * hexc('#0c0e14')
        vinyl = hexc('#d8ecff')
        alb = alb * (1 - cf[..., None]) + hexc('#8a9ab0') * cf[..., None]
        alb = alb * (1 - rbf[..., None]) + hexc('#6a7080') * rbf[..., None]
        alb = alb * (1 - so[..., None]) + hexc('#b8bcc4') * so[..., None]
        E = np.zeros((hp, wp, 3), np.float32)
        yy, xx = np.mgrid[0:hp, 0:wp].astype(np.float32)
        gl = np.clip(0.5 + 0.5 * (xx / wp - 0.3), 0, 1)
        E += vinyl * (cf * (0.35 + 0.5 * gl))[..., None]
        E += hexc('#f2f8ff') * (rf * (1.4 + 1.8 * gl))[..., None]
        E += hexc('#c8dcff') * (rbf * 0.3)[..., None]
        E += hexc('#e8eeff') * (so * 0.9)[..., None]
        rng = np.random.default_rng(5)
        dm = np.zeros((hp, wp), np.float32)
        for _ in range(60):
            a_ = rng.uniform(0.05, np.pi - 0.05)
            r_ = rng.uniform(0.15, 0.97) ** 0.7
            px_, py_ = PU(ux + R_ * r_ * math.cos(a_), uy - 0.02 + Hc * r_ * math.sin(a_))
            cv2.circle(dm, (px_ >> 2, py_ >> 2), max(1, int(0.008 * ppm)), 1.0, -1, cv2.LINE_AA)
        E += hexc('#ffffff') * (dm * 1.2)[..., None]
        er = cv2.erode(body, np.ones((3, 3), np.uint8), iterations=max(1, int(0.012 * ppm)))
        edge_m = (body.astype(np.float32) - er.astype(np.float32)) / 255.0
        E += hexc('#e4eeff') * (edge_m * (0.3 + 1.4 * np.clip(xx / wp * 2 - 0.8, 0, 1)))[..., None]
        cv.sprite(X, 0, Z, w, h, alb, alpha, E, normal=(0, 0, -1))
        self.figure = (X, Z)

    def _paint_lanterns(self, cv):
        """Red paper lanterns hanging at the izakaya fronts."""
        cam = self.cam
        spots = [(3.85, 15.0), (3.85, 17.8), (-3.35, 10.2), (-3.35, 12.4), (-3.4, 30.5), (-3.4, 32.2)]
        for i, (xw, z) in enumerate(spots):
            rng = np.random.default_rng(90 + i)
            ppm = cam.px_per_m(z) * 1.5
            T = A.lantern_tex(0.34 * ppm, 0.55 * ppm, col='#e0301c', glow=1.6, rng=rng)
            X = xw - np.sign(xw) * 0.35
            # hung low enough that lantern + cord stay clear of the fascia sign lettering behind them
            cv.sprite(X, 1.72, z, 0.34, 0.55, T['alb'], T['a'], T['emi'])
            Th = A.tex(max(2, 0.02 * ppm), 0.14 * ppm, hexc('#202020'))
            cv.sprite(X, 2.27, z, 0.02, 0.12, Th['alb'], Th['a'], None)

    def _paint_cones(self, cv, pred):
        """Volumetric light cones under the street lamps (additive, lives in the humid air)."""
        cam = self.cam
        s = cam.ss
        for (X, Z, hh, lamp) in self.poles:
            if not lamp or not pred(Z) or Z > 130:
                continue
            lx = X - np.sign(X) * 1.0
            xa, ya = cam.proj(lx, 5.55, Z)
            xg, yg = cam.proj(lx, 0.0, Z)
            xa, ya, xg, yg = float(xa), float(ya), float(xg), float(yg)
            rw = cam.f * s * 2.3 / Z
            ra = cam.f * s * 0.2 / Z
            x0, x1 = int(max(xg - rw * 1.6, 0)), int(min(xg + rw * 1.6, cv.W))
            y0, y1 = int(max(ya, 0)), int(min(yg + 2, cv.H))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            t = (yy - ya) / max(yg - ya, 1.0)
            hw = ra + (rw - ra) * t
            cx = xa + (xg - xa) * t
            d = (xx - cx) / np.maximum(hw, 1.0)
            inten = np.exp(-d * d * 1.6) * (1 - 0.45 * t) * C.smoothstep(0.0, 0.1, t) * C.smoothstep(1.02, 0.85, t)
            # a brighter core line and soft streaks (rain inside the beam)
            rng = np.random.default_rng(int(Z))
            k = (0.4 if Z < 60 else 0.25) * (1.0 + 0.25 * np.sin(d * 9 + rng.uniform(0, 6)))
            cv.emi[y0:y1, x0:x1] += (inten * k)[..., None] * hexc('#cfe6ff')

    def _paint_standsign(self, cv, only=None):
        """Lit standing signboard on the street (right, foreground)."""
        cam = self.cam
        for k, (X, Z, st, seed) in enumerate(((3.05, 9.2, 0, 71), (-2.75, 16.5, 2, 72))):
            if only is not None and k != only:
                continue
            rng = np.random.default_rng(seed)
            ppm = cam.px_per_m(Z) * 1.3
            T = A.sign_panel(rng, 0.45 * ppm, 0.95 * ppm, style=st, vertical=True, count=3, bright=0.9)
            cv.sprite(X, 0.18, Z, 0.45, 0.95, T['alb'], T['a'], T['emi'], normal=(-np.sign(X) * 0.5, 0, -0.85))
            L = A.tex(0.5 * ppm, 0.2 * ppm, hexc('#2a2c34'))
            cv.sprite(X, 0.0, Z + 0.02, 0.5, 0.2, L['alb'], L['a'], None)

    # ------------------------------------------------------------------------- traffic signal
    def _paint_traffic(self, cv):
        cam = self.cam
        Z = 27.6
        ppm = cam.px_per_m(Z) * 1.5
        # arm from the right corner pole over the road
        Ta = A.tex(3.0 * ppm, max(2, 0.08 * ppm), hexc('#40444e'))
        cv.sprite(2.6, 5.4, Z + 0.1, 2.4, 0.08, Ta['alb'], Ta['a'], None)
        Tp = A.tex(max(2, 0.2 * ppm), 5.8 * ppm, hexc('#50545e'))
        cv.sprite(3.75, 0, Z + 0.1, 0.2, 5.8, Tp['alb'], Tp['a'], None)
        # signal head: 3 lamps, red lit
        w, h = 1.25, 0.42
        T = A.tex(w * ppm, h * ppm, hexc('#2a2d36'))
        hp, wp = T['a'].shape
        for i, (c, on) in enumerate((('#20ff90', 0), ('#ffb020', 0), ('#ff2a1a', 1))):
            cx, cy, r = int(wp * (0.18 + 0.32 * i)), hp // 2, int(hp * 0.33)
            m = np.zeros((hp, wp), np.uint8)
            cv2.circle(m, (cx, cy), r, 255, -1, cv2.LINE_AA)
            mf = m.astype(np.float32)[..., None] / 255
            col = hexc(c)
            T['alb'] = T['alb'] * (1 - mf) + col * 0.3 * mf
            if on:
                T['emi'] += col * mf * 6.0
            else:
                T['emi'] += col * mf * 0.08
        cv.sprite(1.5, 5.0, Z, w, h, T['alb'], T['a'], T['emi'], anchor='center')
        # second signal far away
        Z2 = 96.0
        ppm2 = cam.px_per_m(Z2) * 2
        T2 = A.tex(w * ppm2, h * ppm2, hexc('#2a2d36'))
        hp, wp = T2['a'].shape
        A.rect(T2['emi'], wp * 0.72, hp * 0.2, wp * 0.92, hp * 0.8, hexc('#ff2a1a') * 6)
        cv.sprite(1.4, 5.0, Z2, w, h, T2['alb'], T2['a'], T2['emi'], anchor='center')

    # ------------------------------------------------------------------------- ground
    def _ground(self, fog, dof):
        cam = self.cam
        s = cam.ss
        cv = R.Canvas(cam)
        Hs, Ws = cv.H, cv.W
        ys, xs = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
        xs = (xs + 0.5) / s - 0.5
        ys = (ys + 0.5) / s - 0.5
        v = ys - cam.pcy
        below = v > 0.5
        Z = np.where(below, cam.f * cam.h / np.maximum(v, 0.5), 5000).astype(np.float32)
        X = ((xs - cam.pcx) * Z / cam.f).astype(np.float32)
        # --- world-space textures (sampled via remap from world-aligned grids)
        rng = np.random.default_rng(12)
        # asphalt base
        alb = np.ones((Hs, Ws, 3), np.float32) * hexc('#1c1e24')
        # large patches (repairs)
        patch = self._world_noise(X, Z, 0.25, 0.08, seed=3)
        alb *= (0.8 + 0.45 * patch)[..., None]
        # fine aggregate, fading with distance
        fine = self._world_noise(X, Z, 6.0, 6.0, seed=4)
        fade = np.clip(8.0 / Z, 0, 1)
        alb *= (1 + (fine - 0.5) * 0.5 * fade)[..., None]
        # gutter strips + concrete edges
        inroad = (X > ROAD_L) & (X < ROAD_R)
        gut = ((X > ROAD_L - 0.45) & (X < ROAD_L)) | ((X > ROAD_R) & (X < ROAD_R + 0.45))
        alb[gut] = hexc('#3a3c42')
        # side street asphalt in the intersection
        # paint: white edge lines, stop line, crosswalk
        white = hexc('#d8dde4')
        paint = np.zeros((Hs, Ws), np.float32)
        def band(v0, v1, x):
            return np.clip(np.minimum(x - v0, v1 - x) * 30 / np.maximum(Z, 1) * 0 + (x > v0) * (x < v1), 0, 1)
        edgeL = (X > ROAD_L + 0.25) & (X < ROAD_L + 0.4) & ((Z < X_CROSS0 - 3.5) | (Z > X_CROSS1 + 0.5))
        edgeR = (X > ROAD_R - 0.4) & (X < ROAD_R - 0.25) & ((Z < X_CROSS0 - 3.5) | (Z > X_CROSS1 + 0.5))
        paint = np.maximum(paint, (edgeL | edgeR).astype(np.float32))
        # dashed centre line far beyond the intersection
        cz = (X > 0.1) & (X < 0.25) & (Z > X_CROSS1 + 3) & ((Z % 9.0) < 5.0)
        paint = np.maximum(paint, cz.astype(np.float32))
        # stop line
        sl = (Z > 9.2) & (Z < 9.65) & (X > ROAD_L + 0.45) & (X < 0.1)
        paint = np.maximum(paint, sl.astype(np.float32))
        # zebra crossing (stripes elongated along Z), mid-block, the hero leading element
        zc = (Z > 10.2) & (Z < 14.4) & (X > ROAD_L + 0.15) & (X < ROAD_R - 0.1) & (((X - ROAD_L + 0.1) % 0.9) < 0.46)
        paint = np.maximum(paint, zc.astype(np.float32))
        zebra = zc.astype(np.float32)
        zc0 = (Z > 17.4) & (Z < 20.4) & (X > ROAD_L + 0.2) & (X < ROAD_R - 0.1) & (((X - ROAD_L) % 0.9) < 0.45)
        paint = np.maximum(paint, zc0.astype(np.float32) * 0.8)
        # zebra on the far side too
        zc2 = (Z > X_CROSS1 + 0.3) & (Z < X_CROSS1 + 3.4) & (X > ROAD_L + 0.2) & (X < ROAD_R - 0.1) & \
            (((X - ROAD_L) % 0.9) < 0.45)
        paint = np.maximum(paint, zc2.astype(np.float32))
        # wear on paint
        wear = self._world_noise(X, Z, 3.0, 1.0, seed=8)
        wear2 = self._world_noise(X, Z, 9.0, 9.0, seed=18)
        paint *= np.clip(0.75 + 0.7 * wear, 0, 1) * np.clip(0.8 + 0.5 * wear2, 0, 1)
        alb = alb * (1 - paint[..., None]) + white * paint[..., None]
        # manhole covers
        for (mx, mz) in ((1.2, 9.5), (-1.4, 34.0), (0.9, 58.0)):
            r = np.sqrt((X - mx) ** 2 + (Z - mz) ** 2)
            m = (r < 0.32).astype(np.float32)
            ring = ((r > 0.26) & (r < 0.32)).astype(np.float32)
            pat = (np.sin((X - mx) * 60) * np.sin((Z - mz) * 60) > 0).astype(np.float32)
            mc = hexc('#2e2c2a') * (0.8 + 0.3 * pat)[..., None] * (1 - 0.3 * ring[..., None])
            alb = alb * (1 - m[..., None]) + mc * m[..., None]
        # grates along the gutters
        gr = gut & ((Z % 7.0) < 0.6)
        alb[gr] = hexc('#15161a')
        # wet darkening (paint keeps more of its value)
        alb *= (0.62 + 0.4 * paint)[..., None]
        alpha = below.astype(np.float32)
        cv.alb = (alb * alpha[..., None]).astype(np.float32)
        cv.a = alpha
        cv.z = (Z * alpha).astype(np.float32)
        cv.n = np.zeros((Hs, Ws, 3), np.float32)
        cv.n[..., 1] = alpha
        # road marking 止まれ in our (left) lane before the stop line: real glyphs, elongated along Z as
        # painted on Japanese roads, first character farthest so it reads upright from the driver's seat
        grng = np.random.default_rng(3)
        gw, gh = 480, 1440
        Tm = A.tex(gw, gh, white, a=0.0)
        m = A.glyph_column(grng, gw, gh, 3, weight=0.13, margin=0.02, text='止まれ')
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        wr = self._world_noise(np.linspace(0, 3, gw, dtype=np.float32)[None, :].repeat(gh, 0),
                               np.linspace(0, 9, gh, dtype=np.float32)[:, None].repeat(gw, 1), 3.0, 1.0, seed=8)
        Tm['a'] = m * np.clip(0.45 + 0.6 * wr, 0, 0.8)
        if ROAD_MARKING:
            cv.quad([(-2.0, 0.001, 9.0), (-0.8, 0.001, 9.0), (-0.8, 0.001, 6.55), (-2.0, 0.001, 6.55)],
                    Tm['alb'] * 0.5, Tm['a'], None, normal=(0, 1, 0))
        # ---- extra channels at supersampled res: puddles & reflectance
        pud = self._world_noise(X, Z, 0.35, 0.12, seed=5)
        pud2 = self._world_noise(X, Z, 1.2, 0.5, seed=6)
        # puddles gather at the gutters and in the wheel ruts
        edge_bias = np.exp(-((X - ROAD_L) / 0.8) ** 2) + np.exp(-((X - ROAD_R) / 0.8) ** 2) + \
            0.5 * np.exp(-((X + 1.2) / 0.5) ** 2) + 0.5 * np.exp(-((X - 1.5) / 0.5) ** 2)
        pv = pud * 0.7 + pud2 * 0.3 + 0.18 * edge_bias
        puddle = C.smoothstep(PUD_T, PUD_T + 0.02, pv) * below
        paintmask = paint
        puddle *= (1 - 0.8 * paintmask)
        # reflectance: Schlick fresnel on a water film
        cos = v / np.sqrt(v * v + cam.f * cam.f)
        cos = np.clip(cos, 0, 1)
        Fr = 0.03 + 0.97 * (1 - cos) ** 5
        rough = self._world_noise(X, Z, 2.5, 0.35, seed=7)
        stria = self._world_noise(X, Z, 1.6, 1.6, seed=9)
        smod = np.clip((stria - 0.3) * 1.6, 0, 1) ** 1.3 * (0.5 + rough)
        refl = (np.clip(Fr * 1.3, 0, 1) * (0.2 + 0.9 * smod) + 0.03) * (1 - 0.35 * paintmask)
        refl_p = np.clip(0.25 + 0.9 * Fr, 0, 0.95)
        # reflection break-up: wind ripples and surface texture chop the long neon streaks into bands
        bu = self._world_noise(X, Z, 0.5, 2.6, seed=21) * 0.65 + self._world_noise(X, Z, 1.5, 7.0, seed=22) * 0.35
        brk = np.clip((bu - 0.3) * 2.4, 0, 1) ** 1.2
        # dry-ish patches (crown of the road, fresh asphalt): only there the reflections smear into streaks
        dn = self._world_noise(X, Z, 0.42, 0.16, seed=31) * 0.75 + self._world_noise(X, Z, 1.4, 0.8, seed=32) * 0.25
        dry = C.smoothstep(0.5, 0.64, dn) * (1 - puddle) * (1 - 0.6 * np.exp(-((X - ROAD_L) / 0.6) ** 2)
                                                             - 0.6 * np.exp(-((X - ROAD_R) / 0.6) ** 2))
        dry = np.clip(dry, 0, 1)
        # crisp mirror on the wet film: strong everywhere (anime wet-road look), a little less on paint
        rcr = np.clip(0.3 + 0.75 * Fr, 0, 0.92) * (0.72 + 0.28 * smod) * (1 - 0.4 * paintmask)
        ex = np.dstack([puddle, refl * below, refl_p * below, brk * below, dry * below,
                        rcr * below]).astype(np.float32)
        ex = cv2.resize(ex, (cam.PW, cam.PH), interpolation=cv2.INTER_AREA)
        paint_1x = cv2.resize(paint * (0.35 + 0.65 * zebra), (cam.PW, cam.PH), interpolation=cv2.INTER_AREA)
        # light bleeding from the vending machines / shop glass onto the wet pavement (emissive decal)
        cv.emi += V5.ground_pools(X, Z, below) * alpha[..., None]
        # bake lighting (no DOF blur on the ground so it keeps crisp; mild fog)
        plate = R.bake(cv, self.lights, AMBIENT * 1.3 * (1 + 3.8 * paint_1x[..., None]), fog=fog, dof=None, name='ground')
        # analytic depth above horizon: far
        self.ground = plate
        self.ground_ex = np.ascontiguousarray(ex)
        del cv

    def _world_noise(self, X, Z, fx, fz, seed):
        """Smooth noise evaluated at world coords (X,Z) with frequencies fx, fz (cycles/m)."""
        key = (seed,)
        if not hasattr(self, '_wn'):
            self._wn = {}
        if key not in self._wn:
            n = C.fbm(1024, 1024, 16, 5, seed=seed + 300, aspect=False)
            self._wn[key] = cv2.copyMakeBorder(n, 0, 0, 0, 0, cv2.BORDER_WRAP)
        n = self._wn[key]
        u = (X * fx * 64.0) % 1024
        v = (Z * fz * 64.0) % 1024
        return cv2.remap(n, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)

    # ------------------------------------------------------------------------- rain
    def _rain_setup(self):
        W, H = self.W, self.H
        sc = H / 1080.0
        rng = np.random.default_rng(77)
        tiles, speeds, slants, gains, modes = [], [], [], [], []
        # (count, length px, width px, alpha, blur, speed px/s, gain, mode, slant)
        # three depth planes, deliberately sparse: rain reads as fewer, longer, brighter streaks that pick up
        # the neon, not as a full-screen grain.
        #   far: fine, slow, soft, dissolving into the haze toward the vanishing point
        #   mid: longer single-pixel streaks
        #   near: few long, thick, defocused streaks (+ a handful of very close out-of-focus ones)
        # round 3: every layer is 1 px, short and semi-transparent; brightness comes only from the light behind
        specs = [(2600, 13, 1, 0.16, 0.5, 1100, 1.0, 1, 0.055), (1000, 28, 1, 0.36, 0.5, 2200, 1.0, 1, 0.065),
                 (200, 66, 1, 0.52, 0.7, 3900, 1.0, 0, 0.08)]
        for (cnt, ln, wd, al, bl, sp, gn, md, slant) in specs:
            th = H * 2
            tile = np.zeros((th, W), np.uint8)
            n = int(cnt * W / 1920)
            xs = rng.uniform(0, W, n)
            ys = rng.uniform(0, th, n)
            L = ln * sc * rng.uniform(0.7, 1.3, n)
            for x, y, l in zip(xs, ys, L):
                for dy in (0, -th):        # draw wrapped copies so the vertical seam is invisible
                    p0 = (int(x * 4), int((y + dy) * 4))
                    p1 = (int((x - slant * l) * 4), int((y + dy + l) * 4))
                    cv2.line(tile, p0, p1, int(255 * al * rng.uniform(0.5, 1.0)), max(1, int(round(wd * sc))),
                             cv2.LINE_AA, shift=2)
            tf = tile.astype(np.float32) / 255.0
            if bl > 0:
                tf = cv2.GaussianBlur(tf, (0, 0), bl * sc)
            tiles.append(tf)
            speeds.append(sp * sc)
            slants.append(slant)
            gains.append(gn)
            modes.append(md)
        slant = 0.07
        # rain curtains: sheets of rain that only show against bright sources (doorway, signs)
        th = H * 2
        fine = A.noise(max(W // 2, 8), 10, max(8, W // 7), 6061, 3)
        fine = cv2.resize(fine, (W, th), interpolation=cv2.INTER_CUBIC)
        cur = np.clip((fine - 0.42) * 2.6, 0, 1) ** 1.5
        # broad sheet modulation (slow wind drift), applied to the light map at 1/8 res
        self._sheets = A.noise(W // 8 * 2, H // 8, 9, 6062, 3).astype(np.float32)
        # make it tile vertically: cross-fade the two halves
        wv = (0.5 - 0.5 * np.cos(np.linspace(0, 2 * np.pi, th, dtype=np.float32)))[:, None]
        cur = cur * wv + np.roll(cur, th // 2, 0) * (1 - wv)
        tiles.append(cur.astype(np.float32))
        speeds.append(2800 * sc)
        slants.append(slant)
        gains.append(0.08)
        modes.append(2)
        self.rain_tiles = np.ascontiguousarray(np.stack(tiles).astype(np.float32))
        self.rain_speed = speeds
        self.rain_slant = slants
        self.rain_gain = np.array(gains, np.float32)
        self.rain_mode = np.array(modes, np.int64)
        # far mask: the fine rain dissolves into haze near the vanishing point
        cam = self.cam
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt(((xx - cam.cx) / W) ** 2 + ((yy - cam.cy) / H * 1.3) ** 2)
        self.rain_far = np.ascontiguousarray((0.25 + 0.75 * np.clip(d / 0.32, 0, 1) ** 1.2).astype(np.float32))

    def _ripple_setup(self):
        """Deterministic rain impacts: each event = a 2-frame splash crown followed by an expanding
        elliptical ripple ring (6-12 frames). Denser/brighter on puddles, fainter on the asphalt film."""
        rng = np.random.default_rng(88)
        n = 7000
        X = rng.uniform(ROAD_L - 0.3, ROAD_R + 0.3, n)
        Z = 2.2 + rng.power(0.55, n) * 34.0
        pud = self._world_noise(X[None, :].astype(np.float32), Z[None, :].astype(np.float32), 0.35, 0.12, seed=5)[0]
        pud2 = self._world_noise(X[None, :].astype(np.float32), Z[None, :].astype(np.float32), 1.2, 0.5, seed=6)[0]
        edge_bias = np.exp(-((X - ROAD_L) / 0.8) ** 2) + np.exp(-((X - ROAD_R) / 0.8) ** 2) + \
            0.5 * np.exp(-((X + 1.2) / 0.5) ** 2) + 0.5 * np.exp(-((X - 1.5) / 0.5) ** 2)
        onp = (pud * 0.7 + pud2 * 0.3 + 0.18 * edge_bias) > PUD_T
        keep = onp | (rng.random(n) < 0.45)
        X, Z, onp = X[keep], Z[keep], onp[keep]
        m = len(X)
        self.rip = dict(X=X, Z=Z, t0=rng.uniform(-0.6, DURATION, m), dur=rng.integers(6, 13, m) / C.FPS,
                        r=rng.uniform(0.1, 0.24, m) * np.where(onp, 1.0, 0.7), amp=np.where(onp, 1.0, 0.45),
                        jit=rng.uniform(0, 2 * np.pi, m))
        # splashes on the vending machine tops (y = 1.83 m)
        k = 360
        self.vtop = dict(X=rng.uniform(-3.45, -2.8, k), Z=rng.uniform(4.55, 7.4, k), t0=rng.uniform(-0.2, DURATION, k),
                         jit=rng.uniform(0, 2 * np.pi, k))
        # animated ripple-wobble fields for the reflections (half res, scrolled over time)
        W2, H2 = self.W // 2, self.H // 2
        w1 = A.noise(max(W2 // 10, 8), H2 * 2 // 3, max(4, W2 // 90), 7070, 4)
        w1 = cv2.resize(w1, (W2, H2 * 2), interpolation=cv2.INTER_CUBIC)
        w2 = A.noise(max(W2 // 4, 8), H2 * 2 // 2, max(6, W2 // 40), 7071, 3)
        w2 = cv2.resize(w2, (W2, H2 * 2), interpolation=cv2.INTER_CUBIC)
        self._wob = (w1 - w1.mean()).astype(np.float32), (w2 - w2.mean()).astype(np.float32)
        yy = np.arange(H2, dtype=np.float32)
        self._wob_gain = (np.clip((yy * 2 - self.cam.cy) / (self.H - self.cam.cy), 0, 1) ** 1.3)[:, None].astype(np.float32)
        gx, gy = np.meshgrid(np.arange(W2, dtype=np.float32), yy)
        self._g2 = (gx, gy)

    # ------------------------------------------------------------------------- bokeh
    def _bokeh_setup(self):
        cam = self.cam
        W, H = self.W, self.H
        rng = np.random.default_rng(606)
        self.bokeh = []
        # far point lights beyond focus -> small soft discs
        far = self.planes[0]
        pts = FX.find_lights(far.rgba, far.depth, 40.0, 0.6, 12, 0.05 * W, point=True)
        for (x, y, z, c) in pts:
            X, Y = cam.world_from_plate(x, y, z)
            m = float(c.max())
            col = c / max(m, 1e-3) * min(m, 2.0)
            r = W * min(0.003 + 0.08 * abs(1 / z - 1 / FOCUS), 0.007)
            self.bokeh.append((X, Y, z, col.astype(np.float32) * 0.16, r))
        # (no near-lens discs: they read as dirty blobs over the signage.) Two soft, strongly tinted discs
        # low in the corners only, where they sit over the vending-machine glow and the wet road.
        for (u, v, cc, rr) in ((0.03, 0.93, '#ffb050', 0.05), (0.975, 0.9, '#b080ff', 0.055)):
            Zb = 2.6
            X, Y = cam.world_from_plate(u * W + cam.mx, v * H + cam.my, Zb)
            self.bokeh.append((X, Y, Zb, hexc(cc) * 0.07, rr * W))

    def _draw_bokeh(self, img, dX, dY, dZ):
        cam = self.cam
        W, H = self.W, self.H
        for (X, Y, z, col, r) in self.bokeh:
            zc = z - dZ
            if zc < 0.3:
                continue
            x = cam.cx + cam.f * (X - dX) / zc
            y = cam.cy - cam.f * (Y - cam.h - dY) / zc
            rr = r * z / zc if z < 10 else r
            if x < -rr * 2 or x > W + rr * 2 or y < -rr * 2 or y > H + rr * 2:
                continue
            ox, oy = (x - W / 2) / (W / 2), (y - H / 2) / (H / 2)
            FX.draw_bokeh(img, rr, ox, oy, 6 if rr > 6 else 0, x, y, col)

    # ========================================================================= frame
    def frame(self, t):
        W, H = self.W, self.H
        cam = self.cam
        dX, dY, dZ = cam_path(t)
        if not hasattr(self, '_mir'):
            vy = np.arange(H, dtype=np.float32)
            self._mir = (np.ascontiguousarray(np.arange(W, dtype=np.float32)[None, :].repeat(H, 0)),
                         np.ascontiguousarray((2 * cam.cy - vy)[:, None].repeat(W, 1)))
            self._gy = np.ascontiguousarray(vy[:, None].repeat(W, 1))
        mx, my = self._mir
        # sky: fixed in the frame except a slow cloud drift
        drift = 0.004 * W * t
        M = np.array([[1, 0, cam.cx - cam.pcx - drift], [0, 1, cam.cy - cam.pcy]], np.float32)
        sky = cv2.warpAffine(self.sky, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        refl = cv2.remap(sky, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        g, gex = self.ground.warp(W, H, dX, dY, dZ, extra=self.ground_ex)
        layers = [p.warp(W, H, dX, dY, dZ) for p in self.planes]
        for i, p in enumerate(self.planes):
            if p.mirror is not None:
                R.over_pm(refl, p.mirror.warp(W, H, dX, dY, dZ))
            if i == 1:
                R.over_pm(refl, self.card_mirror.warp(W, H, dX, dY, dZ))
            if i == 2:
                fk, fw = fig_state(t)
                fc = self.fig_cards[fk]
                if fc is not None and fc.mirror is not None:
                    LY.over_card(refl, fc.mirror, cam, dX, dY, dZ - fw)
        # ---- water surface motion: rain rings + wind-ripple wobble displace the reflections
        rings, crowns = self._rings(t, dX, dY, dZ)
        W2, H2 = W // 2, H // 2
        # slow horizontal wobble: world-anchored swell rows drifting toward the camera (smooth over ~10-30 px
        # rows, so mirrored lettering stays readable, just gently wavering)
        dxw = V5.wobble(self, t, dZ)                      # (H2, W2) in full-res px
        rs = cv2.resize(rings, (W2, H2), interpolation=cv2.INTER_AREA)
        rs = cv2.GaussianBlur(rs, (0, 0), 0.8)
        k = 5.0 * W / 1920
        dd = cv2.resize(np.dstack([cv2.Sobel(rs, cv2.CV_32F, 1, 0, ksize=3) * k + dxw,
                                   cv2.Sobel(rs, cv2.CV_32F, 0, 1, ksize=3) * k]), (W, H))
        ddx, ddy = dd[..., 0], dd[..., 1]
        sharp = cv2.remap(refl, mx + ddx, self._gy + ddy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # wet film: the same crisp mirror with a slight vertical softening (water film, not glass)
        soft = cv2.GaussianBlur(sharp, (0, 0), sigmaX=0.35 * W / 1920, sigmaY=1.6 * H / 1080)
        # dry-ish patches: vertically smeared reflections
        s2 = cv2.resize(refl, (W2, H2), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(s2, (0, 0), sigmaX=0.5, sigmaY=0.0055 * H)
        s4 = cv2.resize(s2, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        b2 = cv2.GaussianBlur(s4, (0, 0), sigmaX=0.8, sigmaY=0.012 * H)
        glow = cv2.GaussianBlur(s4, (0, 0), 2.0)
        hi = np.clip(s4 - 0.3, 0, None)
        h1 = cv2.GaussianBlur(hi, (0, 0), sigmaX=0.5, sigmaY=0.032 * H / 4)
        h2 = cv2.GaussianBlur(hi, (0, 0), sigmaX=1.0, sigmaY=0.018 * H / 4)
        hs = cv2.resize(0.6 * h1 + 0.5 * h2, (W2, H2), interpolation=cv2.INTER_LINEAR)
        streak = 0.45 * b1 + 0.35 * cv2.resize(b2, (W2, H2)) + hs * 0.75
        if not hasattr(self, '_rowgain'):
            yy = np.arange(H2, dtype=np.float32) * 2
            gnr = 1.0 + 1.6 * np.clip((yy - cam.cy) / (H - cam.cy), 0, 1) ** 1.2
            self._rowgain = gnr[:, None, None].astype(np.float32)
        gx, gy = self._g2
        streak = cv2.remap(streak, gx + dxw * 0.5, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        streak = cv2.resize(streak * self._rowgain, (W, H), interpolation=cv2.INTER_LINEAR)
        glow = cv2.resize(glow, (W, H), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(sky)
        R.ground_combine2(img, g, gex, streak, sharp, soft, rings * 1.3, glow, crowns * 1.5)   # (+ splash crowns)
        V5.hotspots(self, img, gex, dX, dY, dZ)
        for i, layer in enumerate(layers):
            R.over_pm(img, layer)
            if i == 0:
                V5.mist(self, img, t)
            if i == 1:
                for c in self.cards:
                    LY.over_card(img, c, cam, dX, dY, dZ)
            if i == 2:      # after the near objects: the walker passes in front of the far corner pole
                fk, fw = fig_state(t)
                if self.fig_cards[fk] is not None:
                    LY.over_card(img, self.fig_cards[fk], cam, dX, dY, dZ - fw)
        self._vtop_splashes(img, t, dX, dY, dZ)
        V5.splashes(self, img, t, dX, dY, dZ)
        V5.foot_splashes(self, img, t, dX, dY, dZ, FIG_X, FIG_Z, FIG_SPEED, FIG_CYCLE)
        self._draw_bokeh(img, dX, dY, dZ)
        FN.vp_tame(self, img)
        # ---- bloom / halation on the lit set only (before the rain), so the glow never pulses with the
        # per-frame rain streaks; it only evolves smoothly with the camera move
        FX.fast_bloom(img, threshold=0.65, knee=0.4, strength=0.5, halation=0.22,
                      radii=(0.003, 0.01, 0.035, 0.1), mist=0.14, mist_sigma=0.05)
        V5.rain_halo(self, img, t)
        # ---- rain (layers + curtains), lit by the bloomed scene
        img = self._rain(img, t)
        # ---- post
        img = F.shoulder(img, 0.78, 0.3)
        R.grade(img, 1.12, np.array([0.0, 0.012, 0.035], np.float32), 1.06)
        img = F.finish_fast(img, 0.0, exposure=1.0, sat=1.12, grain_amt=0.0, vig=0.35, ca=0.0012)
        img *= self._paper()
        return img

    def _paper(self):
        """Static, screen-fixed paint/paper tooth (multiplicative, ~+-1.5 %): fibrous low-contrast texture that
        never changes from frame to frame (replaces per-frame grain / dithering)."""
        if getattr(self, '_paper_tex', None) is None:
            W, H = self.W, self.H
            s = W / 1920.0
            a = A.noise(W, H, max(8, int(W / (6 * s))), 9191, 3)
            b = A.noise(max(W // 3, 4), max(H * 2, 4), max(8, int(W / (40 * s))), 9192, 3)
            b = cv2.resize(b, (W, H), interpolation=cv2.INTER_AREA)
            c = A.noise(max(W // 12, 4), max(H // 12, 4), 6, 9193, 3)
            c = cv2.resize(c, (W, H), interpolation=cv2.INTER_CUBIC)
            p = (a - 0.5) * 0.010 + (b - 0.5) * 0.012 + (c - 0.5) * 0.022
            self._paper_tex = (1.0 + p)[..., None].astype(np.float32)
        return self._paper_tex

    def _crown(self, m, x, y, s, age, jit, val):
        """Tiny splash crown at (x, y) px, height ~s px. age in [0, 2) frames."""
        k = 1.0 + 0.5 * age
        a = 1.0 if age < 1 else 0.55
        v = int(val * a)
        if v < 3:
            return
        X4, Y4 = int(x * 4), int(y * 4)
        cv2.ellipse(m, (X4, Y4), (max(1, int(s * 4 * 0.9 * k)), max(1, int(s * 4 * 0.22 * k))), 0, 0, 360, v // 2,
                    -1, cv2.LINE_AA, 2)
        for i, ang in enumerate((-1.0, -0.5, 0.0, 0.5, 1.0)):
            an = ang + 0.15 * math.sin(jit + i * 1.7)
            ln = s * (1.1 + 0.35 * math.cos(jit * 2 + i)) * k
            x1 = x + math.sin(an) * ln * 0.8
            y1 = y - math.cos(an) * ln
            cv2.line(m, (X4, Y4), (int(x1 * 4), int(y1 * 4)), v, 1, cv2.LINE_AA, 2)
            if age >= 1:   # detached droplets
                cv2.circle(m, (int((x1 + math.sin(an) * s * 0.4) * 4), int((y1 - s * 0.3) * 4)),
                           max(1, int(s * 0.9)), v, -1, cv2.LINE_AA, 2)

    def _rings(self, t, dX, dY, dZ):
        """Ripple rings (float map) and splash crowns (float map), both deterministic in t."""
        W, H = self.W, self.H
        cam = self.cam
        rp = self.rip
        m = np.zeros((H, W), np.uint8)
        cr = np.zeros((H, W), np.uint8)
        el = t - rp['t0']
        age = el / rp['dur']
        act = (el > -1.5 / C.FPS) & (age < 1)
        for X, Z, a, e, rr, amp, jit in zip(rp['X'][act], rp['Z'][act], age[act], el[act], rp['r'][act],
                                             rp['amp'][act], rp['jit'][act]):
            zc = Z - dZ
            if zc < 0.6:
                continue
            x = cam.cx + cam.f * (X - dX) / zc
            y = cam.cy + cam.f * (cam.h + dY) / zc
            if x < -20 or x > W + 20 or y > H + 20:
                continue
            fage = e * C.FPS          # frames since impact
            if 0 <= fage < 2 and zc < 22:
                s = cam.f * 0.028 / zc
                if s > 0.7:
                    self._crown(cr, x, y, s, fage, jit, 255 * amp)
            if a <= 0:
                continue
            r = rr * a ** 0.55
            ax = cam.f * r / zc
            ay = ax * (cam.h + dY) / zc
            val = int(255 * amp * (1 - a) ** 1.3)
            if ax < 0.8 or val < 4:
                continue
            th = 1 if ax < 25 else 2
            for kk, rk in enumerate((1.0, 0.62, 0.3)):
                if kk == 2 and a > 0.5:
                    continue
                cv2.ellipse(m, (int(x * 4), int(y * 4)), (max(1, int(ax * rk * 4)), max(1, int(ay * rk * 4))), 0, 0,
                            360, int(val * (1 - 0.35 * kk)), th, cv2.LINE_AA, shift=2)
        return m.astype(np.float32) / 255.0, cr.astype(np.float32) / 255.0

    def _vtop_splashes(self, img, t, dX, dY, dZ):
        """Splash crowns on the flat tops of the vending machines (lit by their own glow)."""
        cam = self.cam
        v = self.vtop
        fa = (t - v['t0']) * C.FPS
        act = (fa >= 0) & (fa < 2)
        if not act.any():
            return
        H, W = img.shape[:2]
        pts = []
        for X, Z, f_, jit in zip(v['X'][act], v['Z'][act], fa[act], v['jit'][act]):
            zc = Z - dZ
            x = cam.cx + cam.f * (X - dX) / zc
            y = cam.cy - cam.f * (1.83 - cam.h - dY) / zc
            pts.append((x, y, cam.f * 0.042 / zc, f_, jit))
        pad = int(0.03 * W)
        x0 = int(max(min(p[0] for p in pts) - pad, 0)); x1 = int(min(max(p[0] for p in pts) + pad, W))
        y0 = int(max(min(p[1] for p in pts) - pad, 0)); y1 = int(min(max(p[1] for p in pts) + pad, H))
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for (x, y, sz, f_, jit) in pts:
            self._crown(m, x - x0, y - y0, sz, f_, jit, 230)
        img[y0:y1, x0:x1] += (m.astype(np.float32) / 255.0)[..., None] * np.float32([0.95, 1.0, 1.15])

    def _rain(self, img, t):
        W, H = self.W, self.H
        # light the drops catch: blurred scene colour; curtains only catch the bright sources
        small = cv2.resize(img, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        lm = cv2.GaussianBlur(small, (0, 0), 3.0)
        hs = cv2.GaussianBlur(np.clip(small - 0.35, 0, None), (0, 0), 5.0) * 2.2
        w8 = small.shape[1]
        fx_ = 0.05 * w8 * t
        ox = int(fx_)
        ax = np.float32(fx_ - ox)
        sh = (self._sheets[:, w8 // 2 - ox: w8 // 2 - ox + w8] * (1 - ax) +
              self._sheets[:, w8 // 2 - ox - 1: w8 // 2 - ox - 1 + w8] * ax)
        hs = hs * np.clip((sh - 0.36) * 2.4, 0, 1)[..., None]
        both = cv2.resize(np.dstack([lm * 2.0 + 0.012, hs]), (W, H), interpolation=cv2.INTER_LINEAR)
        light, hi = both[..., :3], both[..., 3:]
        th = self.rain_tiles.shape[1]
        offs = np.array([int(sp * t) % th for sp in self.rain_speed], np.int64)
        shifts = np.array([int(sp * t * sl) % W for sp, sl in zip(self.rain_speed, self.rain_slant)], np.int64)
        img = np.ascontiguousarray(img, dtype=np.float32)
        R.rain_add2(img, np.ascontiguousarray(light, dtype=np.float32), np.ascontiguousarray(hi, dtype=np.float32),
                    self.rain_far, self.rain_tiles, offs, shifts, self.rain_gain, self.rain_mode)
        FN.fg_rain(self, img, t, light)
        return img
