"""s03_city_dusk - Tokyo skyline at magic hour from a high vantage.

Telephoto view over a dense city just after sunset: a Shinjuku-like cluster of towers on the right (a
twin-topped Metropolitan-Government-like hall, a Cocoon-like ovoid, stepped and crowned tops, fin / band /
glass facades), a Docomo-tower-like clock tower with twin needles on the left third against the brightest
afterglow, a street-gridded ocean of mid-rise blocks with setbacks, rooftop clutter and glyph signs, warm
sodium glow rising out of the street canyons. A towering cumulonimbus glows over the cluster. Floors of
office lights switch on in waves, red aircraft-warning lights blink, a lit commuter train slides along an
elevated viaduct. Foreground: a rooftop with railing, water tank and a lattice mast whose cables sag across
the lower frame. Camera: slow lateral truck + slight pan with true perspective parallax (the city is a 3D
box model rendered into depth bands) and a painted near plane.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import core as C, sky as S, fx as F  # noqa: E402
import s03_city_dusk_city as K  # noqa: E402
import s03_city_dusk_sky as SK  # noqa: E402
import s03_city_dusk_cb8 as CB4  # noqa: E402  (round 9: lobe-following planes, convex-only gold lining)
import s03_city_dusk_fg as FGH  # noqa: E402
import s03_city_dusk_text as TXT  # noqa: E402
import s03_city_dusk_paper as PAPER  # noqa: E402

DURATION = 5.5
# round 10: palette family per hero tower (screen x -> s03_city_dusk_city.PALS row)
HERO_FAM = {0.575: 0, 0.79: 5, 0.87: 2, 0.95: 7, 0.755: 4, 0.84: 1, 0.965: 0, 0.53: 3}
NAME_P = 0.2       # share of near mid-rise roofs with a rooftop name board

HZ = 0.56           # horizon (fraction of H)
FOC = 2.2           # focal length (fraction of W): telephoto compression
CAM_H = 170.0       # camera height (m)
TRUCK = 57.0        # max |camera lateral offset| (m): travel runs TRUCK0 -> TRUCK
TRUCK0 = -45.0
YAW = 0.002         # extra pan half-range (fraction of W): moves everything incl. sky
SUN_X = 0.405       # afterglow centre (fraction of W)
FOG_Z = 4200.0
TR_A, TR_B = 1650.0, 0.33   # elevated railway: z = TR_A + TR_B * x (m)
TR_BAND = 1400.0
DECK0, DECK1 = 15.0, 17.5   # viaduct deck heights (m)
BANDS = [900.0, 1400.0, 2000.0, 2800.0, 3600.0, 5000.0, 8000.0, 40000.0]
# street grid (m): x-period, street width, x offset, z-period, street width, z offset
STREET = (84.0, 12.0, -46.0, 62.0, 10.0, 0.0)
SOD = (0.8, 0.38, 0.26)
AVENUE = (-54.0, -26.0)   # round 6: a wide boulevard (widened grid street) running straight toward the afterglow


def row_ox(zb):
    """round 10: x offset of the cross streets in the block row starting at z = zb (same formula as the
    ground shader in s03_city_dusk_city.render_band): Tokyo blocks do not line up row to row"""
    PX, OX, PZ, OZ = STREET[0], STREET[2], STREET[3], STREET[5]
    row = math.floor((zb - OZ) / PZ + 1e-6)
    fr = (row * 0.618034 + 0.3) % 1.0
    return OX + PX * (0.55 * fr - 0.275)


def _sky_preset():
    return dict(stops=[(0.0, '#16277a'), (0.22, '#283a9c'), (0.42, '#574cb2'), (0.58, '#9c5cb6'),
                       (0.70, '#dc5f9e'), (0.80, '#ff6a70'), (0.88, '#ff6e3e'), (0.95, '#ff8124'),
                       (1.0, '#ff9530')],
                sun_glow='#ffa04a', sun_glow_amt=0.22, below='#ff9530', band=('#ffab48', 0.14))


class City:
    """Procedural building layout (world metres)."""

    def __init__(self, W, H, seed=3):
        self.W, self.H = W, H
        self.f = FOC * W
        self.hy = HZ * H
        self.rng = np.random.default_rng(seed)
        self.rng2 = np.random.default_rng(seed + 1000)   # round-5 extra clutter (keeps the layout stream)
        self.rng3 = np.random.default_rng(seed + 2000)   # round-6 tower-top clutter (keeps both streams)
        self.rng4 = np.random.default_rng(seed + 3000)   # round-7 denser mid-distance roof dressing
        self.rows = []
        self.lights = []     # aircraft warning lights (x, y, z, phase)
        self.spires = []     # (x, z, y0, y1, w0) thin spires drawn onto the band plates
        self.clocks = []     # (x, y, z, r) clock faces
        self.name_cands = []  # roofs that may carry a rooftop name board (added after generation)
        self.cur_pal = 0
        self.ntower = 0
        self.rng5 = np.random.default_rng(seed + 4000)   # round-10 tower palettes (keeps the layout streams)
        self._generate()
        self._name_boards()

    # ------------------------------------------------------------------ helpers
    def xr(self, z, margin_px):
        return (self.W / 2 + margin_px) * z / self.f

    def add(self, x0, x1, z0, z1, h, mat, alb, flh=3.5, bay=3.0, wwf=0.6, whf=0.5, lit0=0.0, litadd=0.0,
            litcol=0.5, glass=1.0, base=0.0, grp=1, sign=0.0, runf=1, rooft=0, crown=0.0, parent=None, seed=None,
            pal=None):
        rec = np.zeros(K.NB)
        if pal is None:
            pal = self.cur_pal if parent is not None else 0
        else:
            self.cur_pal = pal
        # round 10: painted tower palette (1 + family + 16 * tower id); only glass-family materials use it
        rec[K.PAL] = pal if mat in (K.M_GLASS, K.M_CURTAIN, K.M_FINS, K.M_BAND, K.M_DARK) else 0
        rec[[K.BX0, K.BX1, K.BZ0, K.BZ1, K.BH, K.MAT]] = x0, x1, z0, z1, h, mat
        rec[K.SEED] = self.rng.integers(1, 10 ** 6) if seed is None else seed
        rec[[K.AR, K.AG, K.AB]] = alb
        rec[[K.FLH, K.BAYW, K.WWF, K.WHF]] = flh, bay, wwf, whf
        if lit0 < 0.99:     # round 4: make the lights-coming-on progression read across the shot
            lit0, litadd = lit0 * 0.45, min(litadd * 1.45 + lit0 * 0.55, 0.95)
        rec[[K.LIT0, K.LITADD, K.LITCOL, K.GLASS, K.BASE]] = lit0, litadd, litcol, glass, base
        rec[[K.GRP, K.SIGN, K.RUNF, K.ROOFT, K.CROWN]] = grp, sign, runf, rooft, crown
        sk = (z0 if parent is None else parent[0], 0 if parent is None else parent[1] + 1)
        self.rows.append((sk, rec))
        return (z0 if parent is None else parent[0], sk[1])

    def albedo(self, kind):
        r = self.rng
        if kind == 'tower':
            c = [(0.75, 0.85, 1.05), (1.05, 1.02, 1.0), (0.62, 0.68, 0.88), (1.1, 1.0, 0.9), (0.5, 0.55, 0.72)][
                r.integers(0, 5)]
        else:
            c = [(1.12, 1.0, 0.84),    # concrete beige
                 (1.3, 1.27, 1.22),    # white tile
                 (0.85, 0.87, 0.95),   # grey
                 (0.62, 0.52, 0.5),    # dark brown
                 (1.0, 0.64, 0.54),    # brick
                 (1.22, 1.12, 0.92),   # cream
                 (0.55, 0.6, 0.78),    # dark blue-grey
                 (0.95, 0.95, 1.05)][r.integers(0, 8)]
        return np.array(c) * r.uniform(0.8, 1.2)

    def pal_code(self, fam=None, mat=None):
        """round 10: palette family per tower (see s03_city_dusk_city.PALS) -> record code"""
        r = self.rng5
        if fam is None:
            if mat == K.M_DARK:
                fam = int(r.choice([4, 4, 5, 0]))
            elif mat == K.M_FINS:
                fam = int(r.choice([6, 3, 1, 7]))
            elif mat == K.M_BAND:
                fam = int(r.choice([1, 7, 3, 2]))
            else:
                fam = int(r.choice([0, 0, 2, 5, 1, 6, 4]))
        self.ntower += 1
        return 1 + fam + 16 * self.ntower

    @staticmethod
    def track_z(x):
        return TR_A + TR_B * x

    def clearance(self, x0, x1, z0, z1):
        """Max height so the building does not hide the railway deck; None if it sits on the corridor."""
        hm = 1e9
        for x in (x0, x1):
            zt = self.track_z(x)
            if z0 - 14 <= zt <= z1 + 14:
                return None
            for zz in (z0, z1):
                zt2 = TR_A / max(1 - TR_B * x / zz, 1e-3)
                if zz < zt2:
                    hm = min(hm, CAM_H - (CAM_H - DECK0 + 1.0) * zz / zt2)
        if (self.track_z(x0) - z1) * (self.track_z(x1) - z0) < 0 and min(z1, self.track_z(x1)) >= max(z0, self.track_z(x0)) - 14:
            return None
        return hm

    def facade_params(self, mat):
        r = self.rng
        if mat == K.M_CURTAIN:
            return dict(flh=r.uniform(3.8, 4.2), bay=r.uniform(1.5, 2.2), wwf=0.9, whf=0.7,
                        lit0=r.uniform(0.04, 0.08), litadd=r.uniform(0.35, 0.5), litcol=0.3, grp=6, runf=1)
        if mat == K.M_DARK:
            return dict(flh=r.uniform(3.8, 4.3), bay=r.uniform(2.4, 3.4), wwf=0.5, whf=0.45,
                        lit0=r.uniform(0.03, 0.06), litadd=r.uniform(0.25, 0.4), litcol=0.7, grp=1, runf=1)
        if mat == K.M_GLASS:
            return dict(flh=r.uniform(3.8, 4.3), bay=r.uniform(1.6, 3.2), wwf=r.uniform(0.8, 0.92),
                        whf=r.uniform(0.6, 0.78), lit0=r.uniform(0.03, 0.07), litadd=r.uniform(0.3, 0.45),
                        litcol=0.15, grp=int(r.choice([4, 8, 12])), runf=int(r.integers(1, 3)))
        if mat == K.M_APT:
            return dict(flh=r.uniform(2.8, 3.1), bay=r.uniform(3.5, 6.0), wwf=r.uniform(0.6, 0.8),
                        whf=r.uniform(0.45, 0.6), lit0=r.uniform(0.08, 0.16), litadd=r.uniform(0.3, 0.45),
                        litcol=0.75, grp=2, runf=1)
        if mat == K.M_FINS:
            return dict(flh=r.uniform(3.6, 4.2), bay=r.uniform(1.2, 1.8), wwf=r.uniform(0.5, 0.62),
                        whf=r.uniform(0.82, 0.92), lit0=r.uniform(0.03, 0.07), litadd=r.uniform(0.3, 0.45),
                        litcol=0.2, grp=int(r.choice([6, 12, 20])), runf=int(r.integers(1, 3)))
        if mat == K.M_BAND:
            return dict(flh=r.uniform(3.4, 3.9), bay=r.uniform(1.6, 2.4), wwf=0.96, whf=r.uniform(0.42, 0.55),
                        lit0=r.uniform(0.04, 0.08), litadd=r.uniform(0.3, 0.45), litcol=0.3,
                        grp=int(r.choice([8, 16])), runf=int(r.integers(1, 3)))
        return dict(flh=r.uniform(3.2, 3.8), bay=r.uniform(2.6, 5.0), wwf=r.uniform(0.45, 0.7),
                    whf=r.uniform(0.4, 0.6), lit0=r.uniform(0.06, 0.14), litadd=r.uniform(0.3, 0.45),
                    litcol=0.55, grp=int(r.choice([3, 8, 20])), runf=1)

    def roof_clutter(self, pid, x0, x1, z0, z1, h, hm, alb):
        """Stair housings, water tanks, AC units, lattice antennas, fences, billboard frames."""
        r = self.rng
        w, d = x1 - x0, z1 - z0
        if w < 7 or d < 7:
            return
        self.name_cands.append((pid, x0, x1, z0, z1, h, hm))
        dark = np.array([0.8, 0.8, 0.85]) * r.uniform(0.7, 1.1)
        pale = np.array([1.25, 1.2, 1.15]) * r.uniform(0.8, 1.1)

        def item(ix0, ix1, iz0, iz1, top, mat, a, base=h, **kw):
            if top > hm:
                return
            self.add(ix0, ix1, iz0, iz1, top, mat, a, base=base, parent=pid, **kw)

        # stair / elevator housing
        if r.random() < 0.8:
            sw_, sd_ = min(r.uniform(3.5, 6.0), w * 0.45), min(r.uniform(3.5, 6.0), d * 0.45)
            sx_ = x0 + r.uniform(0.08, 0.92 - sw_ / w) * w
            sz_ = z0 + r.uniform(0.15, 0.9 - sd_ / d) * d
            item(sx_, sx_ + sw_, sz_, sz_ + sd_, h + r.uniform(2.8, 4.0), K.M_PLAIN, alb * 0.95)
        # water tank on a stand
        if r.random() < 0.35:
            tw = r.uniform(2.2, 3.4)
            tx = x0 + r.uniform(0.1, 0.9 - tw / w) * w
            tz = z0 + r.uniform(0.2, 0.9 - tw / d) * d
            item(tx + 0.3, tx + tw - 0.3, tz + 0.3, tz + tw - 0.3, h + 1.3, K.M_FENCE, dark)
            item(tx, tx + tw, tz, tz + tw, h + 1.3 + r.uniform(2.0, 3.2), K.M_TANK, pale, base=h + 1.3)
        # AC units
        for _ in range(int(r.integers(0, 6))):
            aw, ad = r.uniform(1.0, 2.2), r.uniform(0.8, 1.4)
            ax = x0 + r.uniform(0.05, 0.95 - aw / w) * w
            az = z0 + r.uniform(0.05, 0.95 - ad / d) * d
            item(ax, ax + aw, az, az + ad, h + r.uniform(0.9, 1.5), K.M_PLAIN, pale * r.uniform(0.8, 1.0))
        # lattice antenna
        if r.random() < 0.18 and h + 8 < hm:
            ax = x0 + r.uniform(0.2, 0.8) * w
            az = z0 + r.uniform(0.3, 0.8) * d
            self.spires.append((ax, az, h, h + r.uniform(6, 14), 0.9))
        # safety fence along the front edge / fenced roof
        if r.random() < 0.3:
            item(x0 + 0.3, x1 - 0.3, z0 + 0.3, z0 + 0.6, h + 1.6, K.M_FENCE, dark)
        # round 5: denser rooftop dressing from a separate stream - whip / lattice antennas, a second
        # water tank, rows of condenser units, and red aviation lamps on the taller mid-rises
        r2 = self.rng2
        for _ in range(int(r2.integers(0, 3))):
            if h + 5 < hm and r2.random() < 0.55:
                ax = x0 + r2.uniform(0.1, 0.9) * w
                az = z0 + r2.uniform(0.2, 0.9) * d
                self.spires.append((ax, az, h, h + r2.uniform(3.5, 9.0), r2.uniform(0.35, 0.7)))
        if r2.random() < 0.22 and w > 10:
            tw = r2.uniform(2.0, 3.0)
            tx = x0 + r2.uniform(0.05, 0.95 - tw / w) * w
            tz = z0 + r2.uniform(0.2, 0.9 - tw / d) * d
            item(tx + 0.3, tx + tw - 0.3, tz + 0.3, tz + tw - 0.3, h + 1.0, K.M_FENCE, dark)
            item(tx, tx + tw, tz, tz + tw, h + 1.0 + r2.uniform(1.8, 2.8), K.M_TANK, pale * 0.95, base=h + 1.0)
        if r2.random() < 0.4 and w > 12:
            n_ = int(r2.integers(2, 6))
            az = z0 + r2.uniform(0.05, 0.4) * d
            ax = x0 + r2.uniform(0.05, 0.3) * w
            for k_ in range(n_):
                if ax + 1.4 > x1 - 0.5:
                    break
                item(ax, ax + 1.3, az, az + 0.9, h + 1.1, K.M_PLAIN, pale * r2.uniform(0.85, 1.0))
                ax += 1.7
        if h > 48 and z0 < 2600 and r2.random() < 0.08:
            self.lights.append((x0 + 0.8, h + 0.6, z0, float(r2.uniform(0, 1))))
        # round 7: denser dressing (own stream): second stair housing, water tanks on stands, AC rows,
        # perimeter railings on the side edges, small sheds / cooling towers
        r4 = self.rng4
        if r4.random() < 0.5 and w > 9:
            sw_, sd_ = min(r4.uniform(2.5, 4.5), w * 0.35), min(r4.uniform(2.5, 4.5), d * 0.35)
            sx_ = x0 + r4.uniform(0.05, 0.95 - sw_ / w) * w
            sz_ = z0 + r4.uniform(0.1, 0.9 - sd_ / d) * d
            item(sx_, sx_ + sw_, sz_, sz_ + sd_, h + r4.uniform(2.2, 3.4), K.M_PLAIN, alb * r4.uniform(0.85, 1.05))
        for _ in range(int(r4.integers(0, 3))):
            if w > 8:
                tw = r4.uniform(1.6, 2.8)
                tx = x0 + r4.uniform(0.05, 0.95 - tw / w) * w
                tz = z0 + r4.uniform(0.1, 0.9 - tw / d) * d
                sh_ = r4.uniform(0.8, 1.6)
                item(tx + 0.25, tx + tw - 0.25, tz + 0.25, tz + tw - 0.25, h + sh_, K.M_FENCE, dark)
                item(tx, tx + tw, tz, tz + tw, h + sh_ + r4.uniform(1.6, 2.6), K.M_TANK,
                     pale * r4.uniform(0.85, 1.05), base=h + sh_)
        for _ in range(int(r4.integers(1, 4))):
            if w > 8:
                n_ = int(r4.integers(2, 5))
                az = z0 + r4.uniform(0.05, 0.85) * d
                ax = x0 + r4.uniform(0.05, 0.5) * w
                for k_ in range(n_):
                    if ax + 1.3 > x1 - 0.4:
                        break
                    item(ax, ax + 1.2, az, az + 0.8, h + r4.uniform(0.9, 1.2), K.M_PLAIN, pale * r4.uniform(0.8, 1.0))
                    ax += 1.5
        if r4.random() < 0.55:
            item(x0 + 0.2, x0 + 0.45, z0 + 0.3, z1 - 0.3, h + 1.3, K.M_FENCE, dark)
            item(x1 - 0.45, x1 - 0.2, z0 + 0.3, z1 - 0.3, h + 1.3, K.M_FENCE, dark)
            item(x0 + 0.3, x1 - 0.3, z0 + 0.2, z0 + 0.45, h + 1.3, K.M_FENCE, dark)
        if r4.random() < 0.25 and w > 12:
            cw_ = r4.uniform(3.0, 5.0)
            cx_ = x0 + r4.uniform(0.1, 0.9 - cw_ / w) * w
            cz_ = z0 + r4.uniform(0.3, 0.9 - cw_ / d) * d if d > cw_ / 0.6 else z0 + 0.2 * d
            item(cx_, cx_ + cw_, cz_, cz_ + cw_ * 0.8, h + r4.uniform(2.5, 3.5), K.M_TANK, dark * 1.2)
        # billboard frame facing the camera
        if r.random() < 0.06 and w > 12:
            bw = w * r.uniform(0.5, 0.8)
            bx = x0 + r.uniform(0.05, 0.95 - bw / w) * w
            bz = z0 + d * r.uniform(0.3, 0.6)
            bh = r.uniform(5, 8)
            item(bx, bx + bw, bz, bz + 0.8, h + 2.2, K.M_FENCE, dark)
            item(bx, bx + bw, bz, bz + 0.5, h + 2.2 + bh, K.M_BILL, pale, base=h + 2.2,
                 lit0=1.0 if r.random() < 0.6 else 0.0)

    def building(self, x0, x1, z0, z1, h, clutter=True):
        if x1 > AVENUE[0] - 1.0 and x0 < AVENUE[1] + 1.0 and z0 < 14000:
            return None     # keep the boulevard open (before any RNG draw: the layout stream is unchanged)
        r = self.rng
        hm = self.clearance(x0, x1, z0, z1)
        if hm is None:
            return None
        if hm < 4.0:
            return None
        h = min(max(h, 5.0), hm)
        if h > 75:
            u = r.random()
            mat = K.M_CURTAIN if u < 0.25 else (K.M_GLASS if u < 0.42 else (K.M_FINS if u < 0.62 else (
                K.M_DARK if u < 0.84 else (K.M_BAND if u < 0.94 else K.M_CONC))))
            kind = 'tower'
        else:
            u = r.random()
            mat = K.M_CONC if u < 0.38 else (K.M_APT if u < 0.62 else (K.M_BAND if u < 0.8 else
                                                                     (K.M_GLASS if u < 0.9 else K.M_FINS)))
            kind = 'mid'
        alb = self.albedo(kind)
        if mat == K.M_DARK:
            alb = alb * 0.55
        fp = self.facade_params(mat)
        if kind == 'mid':
            # round 6: fewer lit windows in the mid-rise sea (it read as speckle noise)
            fp['lit0'] *= 0.35
            fp['litadd'] *= 0.4
        sign = 0.0
        if z0 < 3000 and h < 70 and r.random() < 0.22:
            sign = float(r.integers(1, 999))
        rooft = int(r.integers(0, 5))
        near = z0 < 3600
        heli = 2.0 if (near and 60 < h < CAM_H - 20 and x1 - x0 > 30 and z1 - z0 > 28 and r.random() < 0.5) else 0.0
        # stepped setback: main body lower, upper tier inset (mostly from the front)
        setback = near and h > 18 and r.random() < 0.38
        h_main = h * r.uniform(0.6, 0.8) if setback else h
        pid = self.add(x0, x1, z0, z1, h_main, mat, alb, rooft=rooft, sign=sign,
                       crown=heli if not setback else 0.0, pal=self.pal_code(mat=mat) if h > 60 else 0, **fp)
        top_box = (x0, x1, z0, z1, h_main)
        if setback:
            w, d = x1 - x0, z1 - z0
            ix0 = x0 + w * (r.uniform(0.0, 0.25) if r.random() < 0.6 else 0.0)
            ix1 = x1 - w * (r.uniform(0.0, 0.25) if r.random() < 0.6 else 0.0)
            iz0 = z0 + d * r.uniform(0.2, 0.45)
            if ix1 - ix0 > 5 and z1 - iz0 > 5:
                self.add(ix0, ix1, iz0, z1, h, mat, alb * 0.97, base=h_main, rooft=rooft, parent=pid, **fp)
                top_box = (ix0, ix1, iz0, z1, h)
                if near and clutter:
                    # clutter on the lower terrace too
                    self.roof_clutter(pid, x0, x1, z0, iz0, h_main, hm, alb)
        if near and clutter and h < CAM_H - 4 and r.random() < 0.88:
            self.roof_clutter(pid, top_box[0], top_box[1], top_box[2], top_box[3], top_box[4], hm, alb)
        if h > 95 and z0 < 2200:
            self.lights.append((0.5 * (x0 + x1), h + 0.8, z0, float(r.uniform(0, 1))))
        return pid

    def tower(self, sxf, z, w, d, h, tiers=(), mat=K.M_GLASS, alb=None, fp=None, lights=True, crown=None,
              masts=0, crown_lit=False, fam=None):
        """Hero tower at screen x fraction sxf; tiers: list of (height_add, width_frac[, x_offset_frac[,
        depth_frac]]). crown: None | 'mech' | 'chamfer'."""
        r = self.rng
        xc = (sxf - 0.5) * self.W * z / self.f
        alb = self.albedo('tower') if alb is None else np.array(alb)
        fp = dict(self.facade_params(mat)) if fp is None else fp
        x0, x1 = xc - w / 2, xc + w / 2
        pid = self.add(x0, x1, z, z + d, h, mat, alb, crown=1.0 if (crown_lit and not tiers) else 0.0,
                       pal=self.pal_code(fam, mat), **fp)
        top = h
        cw, cd, cx_ = w, d, xc
        for k, tr in enumerate(tiers):
            dh, wf = tr[0], tr[1]
            off = tr[2] if len(tr) > 2 else 0.0
            df = tr[3] if len(tr) > 3 else wf
            nw, nd = w * wf, d * df
            ncx = xc + off * w
            last = k == len(tiers) - 1
            self.add(ncx - nw / 2, ncx + nw / 2, z + (d - nd) / 2, z + (d + nd) / 2, top + dh, mat, alb * 0.97,
                     base=top, parent=pid, crown=1.0 if (crown_lit and last) else 0.0, **fp)
            top += dh
            cw, cd, cx_ = nw, nd, ncx
        zc = z + (d - cd) / 2
        if crown == 'mech':
            nw, nd = cw * 0.8, cd * 0.8
            self.add(cx_ - nw / 2, cx_ + nw / 2, z + (d - nd) / 2, z + (d + nd) / 2, top + 9.0, K.M_PLAIN,
                     alb * 0.8, base=top, parent=pid)
            # lattice screen around the machine room
            self.add(cx_ - cw / 2 + 0.5, cx_ + cw / 2 - 0.5, zc + 0.5, zc + cd - 0.5, top + 7.0, K.M_FENCE,
                     alb * 0.7, base=top, parent=pid)
            top += 9.0
            cw, cd = nw, nd
            zc = z + (d - cd) / 2
        elif crown == 'chamfer':
            # chamfered top: several short, successively narrower tiers
            for k in range(5):
                nw, nd = cw * (0.93 - 0.02 * k), cd * (0.93 - 0.02 * k)
                self.add(cx_ - nw / 2, cx_ + nw / 2, z + (d - nd) / 2, z + (d + nd) / 2, top + 2.2, mat,
                         alb * 0.95, base=top, parent=pid, **fp)
                top += 2.2
                cw, cd = nw, nd
            zc = z + (d - cd) / 2
        if len(tiers) < 6 and cw > 12:
            top, cw, cd, cx_ = self.tower_top(pid, mat, alb, fp, cx_, zc, cw, cd, top, crown, lights)
            zc = z + (d - cd) / 2
        for k in range(masts):
            mxp = cx_ + (k - (masts - 1) / 2) * cw * 0.45
            mh = r.uniform(14, 30)
            self.spires.append((mxp, zc + cd * 0.5, top, top + mh, 1.1))
            if lights:
                self.lights.append((mxp, top + mh, zc + cd * 0.5, r.uniform(0, 1)))
        if lights:
            ph = r.uniform(0, 1)
            self.lights.append((cx_ - cw / 2 + 1, top + 0.8, zc, ph))
            self.lights.append((cx_ + cw / 2 - 1, top + 0.8, zc, ph))
        return xc, top, pid

    def tower_top(self, pid, mat, alb, fp, cx_, zc, cw, cd, top, crown, lights):
        """Round 6: a readable, cluttered tower-top silhouette - an off-centre stepped setback, a machine
        penthouse, water tanks, a row of condenser units along the parapet, lattice / whip antennas and a
        window-cleaning (BMU) crane with a lattice boom. Uses its own RNG (layout streams unchanged)."""
        r = self.rng3
        sd = lambda: int(r.integers(1, 10 ** 6))
        dark = np.array([0.75, 0.75, 0.85]) * r.uniform(0.7, 1.0)
        pale = np.array([1.2, 1.15, 1.1]) * r.uniform(0.8, 1.0)
        z0c, z1c = zc, zc + cd
        # stepped setback (only where the top is still a plain box)
        if crown is None and r.random() < 0.7:
            wf = r.uniform(0.55, 0.75)
            off = float(r.choice([-1.0, 1.0])) * (1 - wf) * 0.5 * cw
            dh = r.uniform(8.0, 16.0)
            nw = cw * wf
            self.add(cx_ + off - nw / 2, cx_ + off + nw / 2, z0c + cd * 0.1, z1c - cd * 0.1, top + dh, mat,
                     alb * 0.96, base=top, parent=pid, seed=sd(), **fp)
            # the lower terrace keeps a railing
            self.add(cx_ - cw / 2 + 0.5, cx_ + cw / 2 - 0.5, z0c + 0.4, z0c + 0.8, top + 1.4, K.M_FENCE, dark,
                     base=top, parent=pid, seed=sd())
            top += dh
            cw, cx_ = nw, cx_ + off
            z0c, z1c = z0c + cd * 0.1, z1c - cd * 0.1
            cd = z1c - z0c
        # machine penthouse (off-centre)
        if crown != 'mech':
            pw_ = cw * r.uniform(0.35, 0.55)
            px_ = cx_ + r.uniform(-0.25, 0.25) * cw
            self.add(px_ - pw_ / 2, px_ + pw_ / 2, z0c + cd * 0.3, z1c - cd * 0.15, top + r.uniform(4.5, 7.0),
                     K.M_PLAIN, alb * 0.75, base=top, parent=pid, seed=sd())
        # condenser / AC units along the front parapet (small irregular bumps on the silhouette)
        x = cx_ - cw / 2 + r.uniform(0.5, 2.0)
        while x < cx_ + cw / 2 - 2.5:
            aw = r.uniform(1.6, 2.6)
            if r.random() < 0.7:
                self.add(x, x + aw, z0c + 0.6, z0c + 2.0, top + r.uniform(1.2, 2.2), K.M_PLAIN, pale * 0.8,
                         base=top, parent=pid, seed=sd())
            x += aw + r.uniform(0.4, 2.5)
        # water tanks on stands
        for k in range(int(r.integers(1, 3))):
            tw = r.uniform(3.5, 5.5)
            tx = cx_ + r.uniform(-0.4, 0.4) * cw - tw / 2
            tz = z0c + cd * r.uniform(0.35, 0.6)
            self.add(tx + 0.4, tx + tw - 0.4, tz + 0.4, tz + tw - 0.4, top + 1.6, K.M_FENCE, dark, base=top,
                     parent=pid, seed=sd())
            self.add(tx, tx + tw, tz, tz + tw, top + 1.6 + r.uniform(3.0, 4.5), K.M_TANK, pale, base=top + 1.6,
                     parent=pid, seed=sd())
        # window-cleaning crane: mast + lattice boom reaching out over one edge
        if r.random() < 0.65:
            sgn = float(r.choice([-1.0, 1.0]))
            mxc = cx_ + sgn * r.uniform(0.1, 0.3) * cw
            mz = z0c + cd * 0.45
            mh = r.uniform(7.0, 11.0)
            self.add(mxc - 0.6, mxc + 0.6, mz, mz + 1.2, top + mh, K.M_PLAIN, dark * 0.9, base=top, parent=pid,
                     seed=sd())
            bl = r.uniform(12.0, 20.0)
            bx0, bx1 = (mxc - 1.0, mxc + bl) if sgn > 0 else (mxc - bl, mxc + 1.0)
            self.add(bx0, bx1, mz + 0.1, mz + 1.1, top + mh + 1.4, K.M_FENCE, dark * 0.9, base=top + mh - 0.2,
                     parent=pid, seed=sd())
        # lattice / whip antennas
        for k in range(int(r.integers(1, 4))):
            ax = cx_ + r.uniform(-0.4, 0.4) * cw
            ah = r.uniform(6.0, 22.0)
            self.spires.append((ax, z0c + cd * r.uniform(0.3, 0.7), top, top + ah, r.uniform(0.5, 1.4)))
            if lights and ah > 12:
                self.lights.append((ax, top + ah, z0c + cd * 0.5, float(r.uniform(0, 1))))
        return top, cw, cd, cx_

    # ------------------------------------------------------------------ layout
    def _generate(self):
        r = self.rng
        W, f = self.W, self.f
        PX, SWX, OX, PZ, SWZ, OZ = STREET
        # ---------------- street-grid blocks (near/mid distance)
        zb = OZ + math.floor((1000 - OZ) / PZ) * PZ
        while zb < 6500:
            za0, za1 = zb + SWZ, zb + PZ
            dz = za1 - za0
            xm = self.xr(za0, 0.14 * W + f * TRUCK / max(za0, 1) + YAW * W)
            OX = row_ox(zb)      # round 10: each block row has its own cross-street offset (no grid)
            jb0 = math.floor((-xm - OX) / PX) - 1
            jb1 = math.ceil((xm - OX) / PX) + 1
            for j in range(jb0, jb1):
                bx0 = OX + j * PX + SWX
                bx1 = OX + (j + 1) * PX
                # an alley splits some blocks
                cuts = [bx0, bx1]
                if r.random() < 0.4:
                    c = bx0 + (bx1 - bx0) * r.uniform(0.35, 0.65)
                    cuts = [bx0, c - 2.5, c + 2.5, bx1]
                for a0, a1 in zip(cuts[::2], cuts[1::2]):
                    # front row and back row of buildings in the block
                    dfront = dz * r.uniform(0.45, 0.65)
                    for (rz0, rz1) in ((za0, za0 + dfront), (za0 + dfront + r.uniform(0.5, 3), za1)):
                        if rz1 - rz0 < 8:
                            continue
                        x = a0 + r.uniform(0, 1.0)
                        while x < a1 - 6:
                            w = min((r.uniform(9, 34) if r.random() < 0.55 else r.uniform(24, 48)) * (1 + max(za0 - 3500, 0) / 5000), a1 - x)
                            if a1 - (x + w) < 7:
                                w = a1 - x
                            cl = 0.5 + 0.5 * math.sin(x / 700 + 1.3) * math.sin(za0 / 900 + 0.4) + \
                                0.3 * math.sin(x / 260 - za0 / 330)
                            h = math.exp(r.normal(math.log(22), 0.5)) * (1 + 1.2 * max(cl, 0) ** 2)
                            if za0 > 2300 and r.random() < 0.012 + 0.03 * max(cl, 0):
                                h = r.uniform(80, 150)
                            h = float(np.clip(h, 7, 65 if za0 < 2300 else 150))
                            dd = (rz1 - rz0) * r.uniform(0.85, 1.0)
                            zz = rz0 + r.uniform(0, (rz1 - rz0) - dd)
                            if r.random() < 0.3 and w > 16 and dd > 16 and za0 < 3600:
                                # L-shape in plan: a full-depth wing + a lower (or taller) shallow wing
                                c = x + w * r.uniform(0.35, 0.65)
                                left_full = r.random() < 0.5
                                fx0, fx1 = (x, c) if left_full else (c, x + w)
                                sx0, sx1 = (c + 0.2, x + w) if left_full else (x, c - 0.2)
                                self.building(fx0, fx1, zz, zz + dd, h)
                                sd = dd * r.uniform(0.4, 0.65)
                                hh = h * r.uniform(0.45, 0.8) if r.random() < 0.75 else h * r.uniform(1.1, 1.4)
                                if r.random() < 0.5:
                                    self.building(sx0, sx1, zz + dd - sd, zz + dd, min(hh, 65))
                                else:
                                    self.building(sx0, sx1, zz, zz + sd, min(hh, 65))
                            else:
                                self.building(x, x + w, zz, zz + dd, h)
                            x += w + r.uniform(0.3, 2.5)
            zb += PZ
        # ---------------- far city: random rows (cheap)
        z = 6500.0
        while z < 26000:
            xm = self.xr(z, 0.14 * W + f * TRUCK / max(z, 1) + YAW * W)
            x = -xm - r.uniform(0, 30)
            dz = max(26.0, 0.045 * z)
            while x < xm:
                w = r.uniform(12, 42) * (1 + max(z - 6000, 0) / 12000)
                d = r.uniform(12, max(14, min(dz * 1.2, 40)))
                cl = 0.5 + 0.5 * math.sin(x / 700 + 1.3) * math.sin(z / 900 + 0.4) + 0.3 * math.sin(x / 260 - z / 330)
                h = math.exp(r.normal(math.log(20), 0.5)) * (1 + 1.2 * max(cl, 0) ** 2)
                if r.random() < 0.01 + 0.03 * max(cl, 0):
                    h = r.uniform(70, 140)
                h = float(np.clip(h, 7, 150))
                zz = z + r.uniform(0, dz)
                self.building(x, x + w, zz, zz + d, h, clutter=False)
                x += w + r.uniform(0.5, 6.0)
            z += dz * r.uniform(0.85, 1.15)
        # ---------------- hero towers
        # Docomo-like clock tower: slender body, stepped crown with clock faces, twin needle spires
        zd = 2150.0
        fpd = dict(flh=4.4, bay=2.2, wwf=0.55, whf=0.86, lit0=0.15, litadd=0.4, litcol=0.2, grp=40, runf=2)
        xc, top, pid = self.tower(0.30, zd, 40.0, 40.0, 190.0, tiers=[(16, 0.9), (14, 0.8), (12, 0.66), (10, 0.52),
                                                                       (8, 0.4)],
                                  mat=K.M_FINS, alb=(1.02, 0.98, 1.1), fp=fpd, lights=False, fam=6)
        self.clocks.append((xc, 190.0 + 16 + 14 * 0.5, zd + (40 - 40 * 0.8) / 2, 7.5))
        for dxs in (-6.0, 6.0):
            self.spires.append((xc + dxs, zd + 20.0, top, top + 58.0, 2.2))
            self.lights.append((xc + dxs, top + 58.0, zd + 20, 0.2))
        self.spires.append((xc, zd + 20.0, top, top + 20.0, 3.0))
        self.docomo = (xc, zd, top)
        for dxs in (-12.0, 12.0):
            self.lights.append((xc + dxs, 190.0 + 0.8, zd + 4.0, 0.7))
        # Shinjuku-like cluster on the right
        # front row
        # Tocho-like: wide base block + two towers with chamfered crowns, stone fins
        fpt = dict(flh=4.0, bay=1.5, wwf=0.52, whf=0.88, lit0=0.15, litadd=0.4, litcol=0.25, grp=100, runf=2)
        zt = 2500.0
        xt = (0.705 - 0.5) * W * zt / f
        albt = (1.15, 1.1, 1.08)
        # round 7: the wide podium block is a reflective curtain wall (mullions, floor bands, the sunset sky
        # and the cumulus mirrored across it) instead of a flat dark slab
        fpb = dict(flh=4.2, bay=3.2, wwf=0.9, whf=0.7, lit0=0.1, litadd=0.35, litcol=0.3, grp=6, runf=1, glass=0.5)
        pidt = self.add(xt - 48, xt + 48, zt, zt + 40, 170.0, K.M_CURTAIN, np.array(albt), pal=self.pal_code(3), **fpb)
        for sx0 in (xt - 48, xt + 48 - 36):
            self.add(sx0, sx0 + 36, zt + 1, zt + 39, 225.0, K.M_FINS, np.array(albt) * 0.98, base=170.0,
                     parent=pidt, **fpt)
            ctop = 225.0
            cw_ = 36.0
            for k in range(4):
                nw = cw_ - 4.5
                self.add(sx0 + (36 - nw) / 2, sx0 + (36 + nw) / 2, zt + 1 + (38 - (38 - (36 - nw))) / 2,
                         zt + 39 - (36 - nw) / 2, ctop + 4.0, K.M_FINS, np.array(albt) * 0.95, base=ctop,
                         parent=pidt, **fpt)
                ctop += 4.0
                cw_ = nw
            self.lights.append((sx0 + 18 - cw_ / 2 + 1, ctop + 0.6, zt + 5, 0.4))
            self.lights.append((sx0 + 18 + cw_ / 2 - 1, ctop + 0.6, zt + 5, 0.4))
            self.spires.append((sx0 + 18, zt + 20, ctop, ctop + 12, 0.9))
        specs = [  # sxf, z, w, d, h, tiers, kind, crown, masts, mat
            (0.575, 2350, 38, 36, 232, [(12, 0.8)], None, 'chamfer', 1, K.M_CURTAIN),
            (0.645, 2300, 42, 40, 200, [], 'cocoon', None, 0, K.M_GLASS),
            (0.79, 2250, 50, 44, 250, [(10, 0.9), (10, 0.78), (8, 0.62)], None, None, 0, K.M_DARK),  # Sompo-like
            (0.87, 2400, 40, 36, 236, [(14, 0.7, -0.15), (12, 0.45, -0.27)], None, None, 0, K.M_CURTAIN),
            (0.95, 2300, 34, 30, 205, [], None, 'mech', 0, K.M_FINS),
            # back row (taller, hazier)
            (0.61, 3400, 46, 40, 280, [], 'park', None, 0, K.M_GLASS),     # Park-tower-like 3 pyramids
            (0.755, 3500, 60, 50, 320, [], None, 'mech', 2, K.M_DARK),
            (0.84, 3900, 52, 50, 300, [(18, 0.62)], None, 'chamfer', 1, K.M_BAND),
            (0.965, 3300, 42, 40, 268, [], None, 'mech', 1, K.M_CURTAIN),
            (0.53, 4300, 40, 40, 290, [(12, 0.75)], None, None, 0, K.M_FINS),
        ]
        for sxf, z, w, d, h, tiers, kind, crown, masts, mat in specs:
            if kind == 'cocoon':
                n = 26
                prof = []
                for k in range(n):
                    u0 = (k + 1) / n
                    wf = max(0.12, math.sqrt(max(1 - u0 ** 2.2, 0.0)))
                    prof.append((70.0 / n, wf))
                fpc = dict(self.facade_params(K.M_GLASS))
                fpc.update(bay=1.8, flh=4.0)
                self.tower(sxf, float(z), w, d, h, tiers=prof, mat=K.M_GLASS, fp=fpc, alb=(0.7, 0.8, 1.0), fam=1)
            elif kind == 'park':
                # three blocks of different heights, each with a pyramid top
                xc0 = (sxf - 0.5) * self.W * z / self.f
                fpp = self.facade_params(K.M_GLASS)
                for k, (dx_, hh) in enumerate(((-24, h - 30), (0, h), (24, h - 15))):
                    bx = xc0 + dx_
                    pidp = self.add(bx - 13, bx + 13, z + k * 2, z + 38, hh, K.M_GLASS, np.array((0.8, 0.85, 1.0)),
                                    pal=self.pal_code(6),
                                    **fpp)
                    top = hh
                    for s_ in range(12):
                        wf = 1 - (s_ + 1) / 13
                        self.add(bx - 13 * wf, bx + 13 * wf, z + k * 2 + (36 - k * 2) * (1 - wf) / 2,
                                 z + 38 - (36 - k * 2) * (1 - wf) / 2, top + 2.4, K.M_PLAIN,
                                 np.array((0.8, 0.85, 1.0)), base=top, parent=pidp)
                        top += 2.4
                    self.lights.append((bx, top + 0.5, z + 20, r.uniform(0, 1)))
            else:
                self.tower(sxf, float(z), w, d, h, tiers=tiers, mat=mat, crown=crown, masts=masts,
                           crown_lit=(crown is None and r.random() < 0.6), fam=HERO_FAM.get(sxf))
        # distant high-rises melting into the haze (a second, paler skyline)
        for k in range(14):
            z = float(r.uniform(3800, 11000))
            sxf = float(r.uniform(0.52, 1.08)) if k < 11 else float(r.uniform(-0.08, 0.12))
            if 0.5 < sxf < 1.0 and z < 5000:
                continue
            h = float(r.uniform(150, 265))
            w = float(r.uniform(30, 70))
            self.tower(sxf, z, w, w * r.uniform(0.7, 1.0), h, tiers=[] if r.random() < 0.6 else [(10, 0.7)],
                       mat=[K.M_GLASS, K.M_FINS, K.M_CURTAIN, K.M_DARK][int(r.integers(0, 4))], lights=z < 7000)
        for sxf, z, w, h in [(0.035, 2900, 44, 196), (0.115, 4200, 40, 190), (0.19, 3300, 34, 150)]:
            self.tower(sxf, float(z), w, w * 0.9, h, tiers=[] if r.random() < 0.5 else [(12, 0.75)],
                       mat=[K.M_CURTAIN, K.M_FINS, K.M_DARK][int(r.integers(0, 3))], crown='chamfer')

    def _name_boards(self):
        """Rooftop company-name boards (real Japanese lettering, see s03_city_dusk_text.NAMES) along the
        front parapet of some nearer mid-rises. Uses its own RNG so the city layout is unchanged."""
        nn = len(TXT.NAMES)
        order = np.random.default_rng(5).permutation(nn)
        cnt = 0
        for (pid, x0, x1, z0, z1, h, hm) in self.name_cands:
            if z0 > 2700 or h > 70:
                continue
            q = np.random.default_rng(int(abs(x0 * 131.0 + z0 * 17.0 + h * 7.0)) % (2 ** 31))
            if q.random() > NAME_P:
                continue
            w = x1 - x0
            bw = min(w * q.uniform(0.55, 0.85), 26.0)
            bh = min(max(bw * q.uniform(0.16, 0.24), 2.2), 4.5)
            if h + 0.9 + bh > hm:
                continue
            bx = x0 + q.uniform(0.05, 0.95 - bw / w) * w
            bz = z0 + 0.4
            dark = np.array([0.8, 0.8, 0.85]) * q.uniform(0.7, 1.0)
            self.add(bx + 0.4, bx + bw - 0.4, bz + 0.1, bz + 0.5, h + 0.9, K.M_FENCE, dark, base=h,
                     parent=pid, seed=int(q.integers(1, 10 ** 6)))
            # seed encodes (name, style): name = (seed // 6) % nn (each name once before any repeats), style = seed % 6
            sd = 6 * (int(order[cnt % nn]) + nn * int(q.integers(0, 1000))) + int(q.integers(0, 6))
            cnt += 1
            self.add(bx, bx + bw, bz, bz + 0.35, h + 0.9 + bh, K.M_NAME, np.ones(3), base=h + 0.9,
                     parent=pid, seed=sd, lit0=1.0 if q.random() < 0.75 else 0.0)

    def arrays(self):
        self.rows.sort(key=lambda it: (-it[0][0], it[0][1]))
        return np.array([rec for _, rec in self.rows], np.float64)


class Scene:
    FG_Z = 712.0     # parallax-equivalent depth of the painted near plane (rooftops): ~3.5x the skyline
    MAST_Z = 820.0   # the lattice mast + its cables (a building behind the near rooftop)
    MAST_OFF = 98.6  # px (1080p): mast plane placement so it ends where it was designed

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.f = FOC * W
        self.hy = HZ * H
        # ---------------- sky
        self.mx_sky = int(0.04 * W)
        pw, ph = W + 2 * self.mx_sky, H
        self.sky_pw = pw
        self.sun_p = (SUN_X * W + self.mx_sky, self.hy + 0.035 * H)
        self.sky = S.sky_gradient(pw, ph, _sky_preset(), horizon=HZ, sun=self.sun_p, sun_radius=0.32)
        self.sky = self.sky + S.sun(pw, ph, self.sun_p[0], self.sun_p[1], radius=0.02, color=(1.0, 0.62, 0.3),
                                    disc=False, intensity=0.55, glow_size=1.3)
        self._build_clouds()
        # haze colour image in frame coords (for the city's aerial perspective): the sky above the horizon;
        # below it graded from the horizon colour toward warm peach near the sun / blue-violet at the edges
        sky_frame = self.sky[:, self.mx_sky:self.mx_sky + W]
        haze = sky_frame.copy()
        hyi = int(self.hy) - 2
        haze[hyi:] = haze[hyi]
        xs = np.arange(W, dtype=np.float32)
        near_sun = np.exp(-np.abs(xs - SUN_X * W) / (0.22 * W))[:, None]
        tint = (np.array([0.25, 0.32, 0.56], np.float32) * (1 - near_sun) +
                np.array([0.98, 0.56, 0.48], np.float32) * near_sun)
        ky = C.smoothstep(self.hy, self.hy + 0.3 * H, np.arange(H, dtype=np.float32))[:, None, None] * 0.75
        haze = haze * (1 - ky) + tint[None] * ky
        haze = cv2.GaussianBlur(haze, (0, 0), 0.01 * W)
        self.haze = np.ascontiguousarray(haze.astype(np.float64))
        # ---------------- city
        city = City(W, H)
        self.city = city
        B = city.arrays()
        self.B = B
        self._render_bands(B)
        self._build_foreground()
        self.glare_pad = int(YAW * W + 4)
        gp = self.glare_pad
        self.glare = np.ascontiguousarray(self._glare(W + 2 * gp, H, SUN_X * W + gp, self.hy + 0.004 * H))
        self.paper = PAPER.paper(W, H)     # static paint / paper surface (identical every frame)

    # ------------------------------------------------------------------ clouds
    def _build_clouds(self):
        W, H = self.W, self.H
        pw, ph = self.sky_pw + int(0.04 * W), H
        self.cloud_pw = pw
        ox = (pw - W) / 2
        sun = (self.sun_p[0] + (pw - self.sky_pw) / 2, self.sun_p[1])
        self.sun_c = sun
        skyc = cv2.resize(self.sky, (pw, ph))
        # hero cumulonimbus behind the Shinjuku cluster (painted lobes, crisp gold rims)
        self.hero, self.bank = CB4.paint(pw, ph, W, H, ox, sun, seed=5, sky=skyc)
        # shift the plate so the cloud sits at the same frame position as designed (plate is wider)
        f = lambda x: x * W + ox
        high = [
            dict(cx=f(0.26), cy=0.08 * H, L=0.8 * W, T=0.034 * H, tier=1.0, tilt=0.03, sub=3),
            dict(cx=f(0.5), cy=0.035 * H, L=0.28 * W, T=0.016 * H, tier=0.95, tilt=-0.02, sub=2),
            dict(cx=f(0.14), cy=0.24 * H, L=0.46 * W, T=0.028 * H, tier=0.85, sub=2, wave=2.2),
        ]
        low = [
            dict(cx=f(0.38), cy=0.34 * H, L=0.44 * W, T=0.016 * H, tier=0.55, sub=2),
            dict(cx=f(0.36), cy=0.435 * H, L=0.5 * W, T=0.016 * H, tier=0.15, sub=2, wave=2.0),
            dict(cx=f(0.10), cy=0.395 * H, L=0.32 * W, T=0.016 * H, tier=0.2, sub=2, wave=2.0),
            dict(cx=f(0.22), cy=0.495 * H, L=0.42 * W, T=0.011 * H, tier=0.0, sub=1, wave=1.5),
        ]
        hp = SK.paint(pw, ph, skyc, high, sun, seed=11)
        lp = SK.paint(pw, ph, skyc, low, sun, seed=23)
        self.clouds = S.CloudDrift([(self.bank, 0.0008, 0.15), (self.hero, 0.0012, 0.2), (lp, 0.0045, 0.35),
                                    (hp, 0.009, 0.5)])

    # ------------------------------------------------------------------ city bands
    def _render_bands(self, B):
        W, H, f, hy = self.W, self.H, self.f, self.hy
        cx = W / 2
        ss = 2
        x0, x1, z0, z1, hb = B[:, K.BX0], B[:, K.BX1], B[:, K.BZ0], B[:, K.BZ1], B[:, K.BH]
        base = B[:, K.BASE]
        sxa = cx + f * np.minimum(x0 / z0, x0 / z1)
        sxb = cx + f * np.maximum(x1 / z0, x1 / z1)
        sya = hy + f * np.minimum((CAM_H - hb) / z0, (CAM_H - hb) / z1)
        syb = hy + f * np.maximum((CAM_H - base) / z0, (CAM_H - base) / z1)
        bb = np.stack([sxa - 1, sya - 1, sxb + 1, syb + 1], 1).astype(np.float64)
        self.bands = []
        jobs = []
        for bi in range(len(BANDS) - 1):
            za, zb = BANDS[bi], BANDS[bi + 1]
            sel = (z0 >= za) & (z0 < zb)
            if za == TR_BAND:
                behind = z0 >= TR_A + TR_B * 0.5 * (x0 + x1)
                jobs.append((za, zb, sel & behind, 1))
                jobs.append((za, zb, sel & ~behind, 2))
            else:
                jobs.append((za, zb, sel, 0))
        jobs = sorted(jobs, key=lambda j: (-j[0], j[3]))
        out = []
        for j in jobs:
            out.append(j)
            if j[0] == TR_BAND and j[3] == 1:
                out.append(('train',))
        sod = np.array(SOD, np.float64)
        for j in out:
            if j[0] == 'train':
                vp = self._viaduct_plate(int(f * TRUCK / TR_BAND + YAW * W + 6))
                vp['train'] = True
                self.bands.append(vp)
                continue
            za, zb, sel, side = j
            idx = np.nonzero(sel)[0]
            zrep = 2.0 / (1.0 / za + 1.0 / min(zb, 12000.0))
            if za == TR_BAND:
                zrep = TR_A
            mx = int(f * TRUCK / za + YAW * W + 6)
            top = int(max(0, np.floor(bb[idx, 1].min()))) if len(idx) else int(hy)
            for (sxw, szw, y0w, y1w, w0) in self.city.spires:
                if za <= szw < zb:
                    top = min(top, int(hy + f * (CAM_H - y1w) / szw) - 4)
            top = max(0, min(top, int(hy + f * CAM_H / zb) - 2))
            bot = int(min(H, math.ceil(hy + f * CAM_H / za) + 2))
            if bot <= top:
                continue
            Ws, Hs = (W + 2 * mx) * ss, (bot - top) * ss
            order = idx.astype(np.int64)
            rgb, al, E, Eon, iz = K.render_band(B, order, bb, Hs, Ws, float(ss), float(-mx), float(top), cx, hy, f,
                                            CAM_H, za, zb, self.haze, hy, FOG_Z, 6500.0, True,
                                            np.array(STREET + AVENUE, np.float64), TR_A, TR_B, side, (SUN_X - 0.5) * W / f, sod,
                                            *TXT.atlases())
            wd, hd = W + 2 * mx, bot - top
            pm = cv2.resize(np.dstack([rgb * al[..., None], al]), (wd, hd), interpolation=cv2.INTER_AREA)
            Ed = cv2.resize(E, (wd, hd), interpolation=cv2.INTER_AREA)
            Eond = cv2.resize(Eon, (wd, hd), interpolation=cv2.INTER_AREA)
            self._izd = cv2.resize(iz, (wd, hd), interpolation=cv2.INTER_AREA)
            lum = 0.3 * Ed[..., 0] + 0.55 * Ed[..., 1] + 0.15 * Ed[..., 2]
            on = np.where(lum > 1e-5, Eond / np.maximum(lum, 1e-5), 1e3).astype(np.float32)
            pm = pm.astype(np.float32)
            Ed = Ed.astype(np.float32)
            for (sxw, szw, y0w, y1w, w0) in self.city.spires:
                if za <= szw < zb:
                    if side:
                        # round 10: the railway band is split in two plates; draw each antenna only into the
                        # plate that holds its building (it was drawn into both -> stray floating streaks)
                        bh_ = szw >= TR_A + TR_B * sxw
                        if bh_ != (side == 1):
                            continue
                    self._draw_spire(pm, sxw, szw, y0w, y1w, w0, mx, top)
            for (cxw, cyw, czw, rw) in self.city.clocks:
                if za <= czw < zb:
                    self._draw_clock(pm, Ed, on, cxw, cyw, czw, rw, mx, top)
            if za >= 2000.0:
                # round 9: cool aerial haze stepping up with depth (strongest on the sun axis, where the
                # blocks otherwise repeat at one value) -> the depth steps of the city read
                amt = 0.1 + 0.3 * min((math.log(za) - math.log(2000.0)) / math.log(4.0), 1.0)
                xs_ = np.arange(wd, dtype=np.float32) - mx
                ax_ = np.exp(-((xs_ - SUN_X * W) / (0.22 * W)) ** 2)
                k_ = (amt * (0.55 + 0.45 * ax_))[None, :, None].astype(np.float32)
                hc_ = np.array([0.56, 0.5, 0.74], np.float32)
                pm[..., :3] = pm[..., :3] * (1 - k_) + hc_ * pm[..., 3:4] * k_
                Ed = Ed * (1 - 0.6 * k_)
            self.bands.append(dict(za=za, zb=zb, zrep=zrep, mx=mx, top=top, bot=bot, pm=pm,
                                   E=np.ascontiguousarray(Ed), on=np.ascontiguousarray(on)))

    def _draw_spire(self, pm, xw, zw, y0w, y1w, w0, mx, top):
        """Tapered antenna / needle spire drawn into a band plate (premultiplied)."""
        f, hy, W = self.f, self.hy, self.W
        h, w = pm.shape[:2]
        X = lambda xx: W / 2 + f * xx / zw + mx
        Y = lambda yy: hy + f * (CAM_H - yy) / zw - top
        pts = [(X(xw - w0 / 2), Y(y0w)), (X(xw + w0 / 2), Y(y0w)), (X(xw + 0.12), Y(y1w)), (X(xw - 0.12), Y(y1w))]
        bx0 = int(max(min(p[0] for p in pts) - 8, 0)); bx1 = int(min(max(p[0] for p in pts) + 8, w))
        by0 = int(max(min(p[1] for p in pts) - 8, 0)); by1 = int(min(max(p[1] for p in pts) + 8, h))
        if bx1 <= bx0 or by1 <= by0:
            return
        sw_, sh_ = bx1 - bx0, by1 - by0
        loc = [(p[0] - bx0, p[1] - by0) for p in pts]
        m = C.polygon_mask(sw_, sh_, loc, ss=4)
        # round 10: depth test against the band plate (nearer facades hide the antenna; it was drawn over them)
        izs = self._izd[by0:by1, bx0:bx1]
        vis = np.clip((1.0 / (zw - 2.0) - izs) * zw * zw * 0.05 + 0.5, 0.0, 1.0).astype(np.float32)
        if w0 > 1.5:
            for yy, ww in ((y0w + (y1w - y0w) * 0.25, w0 * 1.7), (y0w + (y1w - y0w) * 0.55, w0 * 1.25)):
                q = [(X(xw - ww / 2) - bx0, Y(yy) - by0), (X(xw + ww / 2) - bx0, Y(yy) - by0),
                     (X(xw + ww / 2) - bx0, Y(yy - 2.0) - by0), (X(xw - ww / 2) - bx0, Y(yy - 2.0) - by0)]
                m = np.maximum(m, C.polygon_mask(sw_, sh_, q, ss=4))
        fog = 1 - math.exp(-max(zw - 700, 0) / (FOG_Z * 2.5))
        hz = self.haze[int(hy) - 30, int(np.clip(X(xw) - mx, 0, W - 1))]
        col = np.array([0.1, 0.12, 0.25]) * (1 - fog) + hz * fog
        rim = C.line_mask(sw_, sh_, (loc[0][0] + 0.3, loc[0][1]), (loc[3][0] + 0.2, loc[3][1]), width=0.7) * m
        c = np.broadcast_to(col, (sh_, sw_, 3)) + rim[..., None] * np.array([1.2, 0.6, 0.35]) * 0.7
        sub = pm[by0:by1, bx0:bx1]
        m = m * vis
        sub[..., :3] = sub[..., :3] * (1 - m[..., None]) + c * m[..., None]
        sub[..., 3] = sub[..., 3] * (1 - m) + m

    def _draw_clock(self, pm, E, on, xw, yw, zw, rw, mx, top):
        """Lit clock face (pale warm disc, dark hands) on the Docomo-like crown."""
        f, hy, W = self.f, self.hy, self.W
        cxp = W / 2 + f * xw / zw + mx
        cyp = hy + f * (CAM_H - yw) / zw - top
        R = f * rw / zw
        h, w = pm.shape[:2]
        x0, x1 = int(max(cxp - R - 3, 0)), int(min(cxp + R + 4, w))
        y0, y1 = int(max(cyp - R - 3, 0)), int(min(cyp + R + 4, h))
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d = np.sqrt((xx - cxp) ** 2 + (yy - cyp) ** 2)
        disc = np.clip(R - d + 0.5, 0, 1)
        ring = np.clip(1 - np.abs(d - R * 0.92) / max(0.12 * R, 0.6), 0, 1)
        # hands: ~6:25
        hands = np.zeros_like(d)
        for ang, ln in ((math.radians(90 + 12), 0.5), (math.radians(90 - 150 + 360), 0.78)):
            ux, uy = math.cos(ang), math.sin(ang)
            px_, py_ = xx - cxp, yy - cyp
            t_ = np.clip(px_ * ux + py_ * uy, 0, ln * R)
            dd = np.sqrt((px_ - t_ * ux) ** 2 + (py_ - t_ * uy) ** 2)
            hands = np.maximum(hands, np.clip(1 - dd / max(0.07 * R, 0.5), 0, 1))
        face = disc * (1 - np.maximum(hands, ring * 0.7))
        sub = pm[y0:y1, x0:x1]
        sub[..., :3] = sub[..., :3] * (1 - disc[..., None]) + np.array([0.4, 0.36, 0.34], np.float32) * disc[..., None]
        sub[..., 3] = np.maximum(sub[..., 3], disc)
        E[y0:y1, x0:x1] = E[y0:y1, x0:x1] * (1 - disc[..., None]) + \
            np.array([1.2, 1.05, 0.85], np.float32) * face[..., None] * 0.55
        on[y0:y1, x0:x1] = np.where(disc > 0.01, -10.0, on[y0:y1, x0:x1])

    # ------------------------------------------------------------------ railway + train
    def _proj(self, x, y, z, truck, yaw):
        return (self.W / 2 + self.f * (x - truck) / z - yaw, self.hy + self.f * (CAM_H - y) / z)

    class _Canvas:
        """Supersampled drawing region (x0..x0+w, y0..y0+h in frame px) with colour/alpha/emission."""

        def __init__(self, x0, y0, w, h, ss):
            self.x0, self.y0, self.w, self.h, self.ss = x0, y0, w, h, ss
            self.col = np.zeros((h * ss, w * ss, 3), np.float32)
            self.al = np.zeros((h * ss, w * ss), np.float32)
            self.emi = np.zeros((h * ss, w * ss, 3), np.float32)

        def P(self, p):
            return ((p[0] - self.x0) * self.ss, (p[1] - self.y0) * self.ss)

        def _stamp(self, m, x0, y0, c, a, e, add):
            h, w = m.shape
            mb = m.astype(bool)
            sc = self.col[y0:y0 + h, x0:x0 + w]
            sa = self.al[y0:y0 + h, x0:x0 + w]
            se = self.emi[y0:y0 + h, x0:x0 + w]
            sc[mb] = sc[mb] * (1 - a) + np.asarray(c, np.float32) * a
            sa[mb] = sa[mb] * (1 - a) + a
            if e is not None:
                if add:
                    se[mb] = se[mb] + np.asarray(e, np.float32)
                else:
                    se[mb] = np.asarray(e, np.float32)
            else:
                se[mb] *= (1 - a)

        def poly(self, pts, c, a=1.0, e=None):
            Hc, Wc = self.al.shape
            p = np.array([self.P(q) for q in pts], np.float64).round().astype(np.int32)
            x0, y0 = max(p[:, 0].min() - 1, 0), max(p[:, 1].min() - 1, 0)
            x1, y1 = min(p[:, 0].max() + 2, Wc), min(p[:, 1].max() + 2, Hc)
            if x1 <= x0 or y1 <= y0:
                return
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [p - [x0, y0]], 1)
            self._stamp(m, x0, y0, c, a, e, False)

        def line(self, p0, p1, c, width, a=1.0, e=None):
            Hc, Wc = self.al.shape
            wpx = max(1, int(round(width * self.ss)))
            q = np.array([self.P(p0), self.P(p1)], np.float64).round().astype(np.int32)
            x0, y0 = max(q[:, 0].min() - wpx - 1, 0), max(q[:, 1].min() - wpx - 1, 0)
            x1, y1 = min(q[:, 0].max() + wpx + 2, Wc), min(q[:, 1].max() + wpx + 2, Hc)
            if x1 <= x0 or y1 <= y0:
                return
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.line(m, (int(q[0, 0] - x0), int(q[0, 1] - y0)), (int(q[1, 0] - x0), int(q[1, 1] - y0)), 1, wpx)
            self._stamp(m, x0, y0, c, a, e, True)

        def result(self):
            pm = cv2.resize(np.dstack([self.col * self.al[..., None], self.al]), (self.w, self.h),
                            interpolation=cv2.INTER_AREA)
            E = cv2.resize(self.emi, (self.w, self.h), interpolation=cv2.INTER_AREA)
            return pm.astype(np.float32), E.astype(np.float32)

    def _rail_frame(self):
        L = math.sqrt(1 + TR_B ** 2)
        return L, 1 / L, TR_B / L, TR_B / L, -1 / L

    def _rail_fog(self):
        W, H, hy = self.W, self.H, self.hy
        fog = 1 - math.exp(-(TR_A - 700) / (FOG_Z * 2.2))
        hz = self.haze[min(int(hy) + 40, H - 1), W // 2]
        return lambda c: np.asarray(c, np.float32) * (1 - fog) + hz.astype(np.float32) * fog

    def _viaduct_plate(self, mx):
        """Static elevated viaduct (piers, deck, parapet, rails, catenary masts + wires, lamps) rendered once
        into a band-like plate (camera centred) that is shifted with the TR_A parallax per frame."""
        W, H, f, hy = self.W, self.H, self.f, self.hy
        L, ux, uz, nx, nz = self._rail_frame()
        xr0, xr1 = -560.0, 560.0
        y_top = int(min(hy + f * (CAM_H - DECK1 - 12) / (TR_A + TR_B * xr1),
                        hy + f * (CAM_H - DECK1 - 12) / (TR_A + TR_B * xr0))) - 3
        y_bot = int(max(hy + f * CAM_H / (TR_A + TR_B * xr0 - 6), hy + f * CAM_H / (TR_A + TR_B * xr1 - 6))) + 3
        y_top, y_bot = max(y_top, 0), min(y_bot, H)
        cv_ = self._Canvas(-mx, y_top, W + 2 * mx, y_bot - y_top, 3)
        P = lambda x, y, z: self._proj(x, y, z, 0.0, 0.0)
        fg = self._rail_fog()
        xs = np.arange(xr0, xr1 + 1, 8.0)
        for x in np.arange(xr0 - 10, xr1 + 11, 26.0):
            z = TR_A + TR_B * x
            a0 = (x + nx * 2.5 - ux * 1.4, z + nz * 2.5 - uz * 1.4)
            a1 = (x + nx * 2.5 + ux * 1.4, z + nz * 2.5 + uz * 1.4)
            cv_.poly([P(a0[0], 0, a0[1]), P(a1[0], 0, a1[1]), P(a1[0], DECK0, a1[1]), P(a0[0], DECK0, a0[1])],
                     fg((0.1, 0.12, 0.22)))
            cv_.poly([P(a0[0], 0, a0[1]), P(a1[0], 0, a1[1]), P(a1[0], 4.0, a1[1]), P(a0[0], 4.0, a0[1])],
                     fg((0.55, 0.3, 0.25)), a=0.6)
        fr = [(x + nx * 5.0, TR_A + TR_B * x + nz * 5.0) for x in xs]
        bk = [(x - nx * 5.0, TR_A + TR_B * x - nz * 5.0) for x in xs]
        for i in range(len(xs) - 1):
            (xa, za), (xb, zb) = fr[i], fr[i + 1]
            (xc, zc), (xd, zd) = bk[i], bk[i + 1]
            cv_.poly([P(xa, DECK1, za), P(xb, DECK1, zb), P(xd, DECK1, zd), P(xc, DECK1, zc)], fg((0.3, 0.26, 0.42)))
            cv_.poly([P(xa, DECK0, za), P(xb, DECK0, zb), P(xb, DECK1, zb), P(xa, DECK1, za)], fg((0.2, 0.24, 0.36)))
        for i in range(len(xs) - 1):
            (xa, za), (xb, zb) = fr[i], fr[i + 1]
            cv_.line(P(xa, DECK1 + 0.9, za), P(xb, DECK1 + 0.9, zb), fg((0.26, 0.28, 0.42)), 0.9)
            cv_.line(P(xa, DECK1 + 1.4, za), P(xb, DECK1 + 1.4, zb), fg((1.0, 0.6, 0.55)), 0.4)
        for off in (-2.6, -1.1, 1.1, 2.6):
            pts = [P(x + nx * off, DECK1 + 0.25, TR_A + TR_B * x + nz * off) for x in xs[::4]]
            for i in range(len(pts) - 1):
                cv_.line(pts[i], pts[i + 1], fg((1.0, 0.72, 0.62)), 0.3, a=0.9, e=(0.5, 0.26, 0.2))
        for x in np.arange(xr0, xr1 + 1, 45.0):
            z = TR_A + TR_B * x - nz * 4.4
            xx = x - nx * 4.4
            cv_.line(P(xx, DECK1, z), P(xx, DECK1 + 9.0, z), fg((0.1, 0.1, 0.2)), 0.6)
            cv_.line(P(xx, DECK1 + 8.3, z), P(xx + nx * 7.5, DECK1 + 8.3, z + nz * 7.5), fg((0.1, 0.1, 0.2)), 0.45)
            cv_.line(P(xx - 0.3, DECK1, z), P(xx - 0.3, DECK1 + 9.0, z), fg((1.0, 0.55, 0.4)), 0.2, a=0.8)
        for off in (-1.8, 1.8):
            pts = [P(x + nx * off, DECK1 + 7.4 - 0.4 * math.sin(math.pi * ((x - xr0) % 45.0) / 45.0),
                     TR_A + TR_B * x + nz * off) for x in xs[::2]]
            for i in range(len(pts) - 1):
                cv_.line(pts[i], pts[i + 1], fg((0.3, 0.25, 0.4)), 0.22, a=0.8, e=(0.12, 0.06, 0.05))
        pm, E = cv_.result()
        lp = []
        for x in np.arange(xr0 - 40, xr1 + 41, 22.0):
            z = TR_A + TR_B * x + nz * 5.4
            sx, sy = P(x + nx * 5.4, DECK1 + 1.2, z)
            lp.append((sx + mx, sy - y_top))
        lp = np.array(lp)
        C.splat(E, lp[:, 0], lp[:, 1], 0.0009 * W, (1.0, 0.62, 0.28), 1.6)
        C.splat(E, lp[:, 0], lp[:, 1], 0.0035 * W, (1.0, 0.55, 0.25), 0.12)
        on = np.full(pm.shape[:2], -10.0, np.float32)
        return dict(za=TR_BAND, zb=TR_BAND, zrep=TR_A, mx=mx, top=y_top, bot=y_bot, pm=np.ascontiguousarray(pm),
                    E=np.ascontiguousarray(E), on=on)

    def _train(self, img, em, t, truck, yaw, tA=TR_A, tB=TR_B, dk=DECK1, v=42.0, head0=250.0, ncar=8, far=False, side=1.9, glint=True):
        """The lit commuter train (8 cars, moving -x toward the afterglow), drawn per frame in a crop."""
        W, H, f, hy = self.W, self.H, self.f, self.hy
        L, ux, uz, nx, nz = self._rail_frame()
        if far:
            L = math.sqrt(1 + tB ** 2)
            ux, uz, nx, nz = 1 / L, tB / L, tB / L, -1 / L
            fo = 1 - math.exp(-(tA - 700) / (FOG_Z * 2.2))
            hz0 = self.haze[min(int(hy) + 20, H - 1), W // 2].astype(np.float32)
            fg = lambda c: np.asarray(c, np.float32) * (1 - fo) + hz0 * fo
        else:
            fg = self._rail_fog()
        head = head0 * L - v * t * L
        car, gap = 20.0, 1.4
        y0, y1 = dk + 0.8, dk + 8.0
        o = side
        # the band shifts with the tA parallax -> draw the train with the same camera mapping
        sh = -f * truck / tA - yaw
        P = lambda x, y, z: (W / 2 + f * x / z + sh, hy + f * (CAM_H - y) / z)
        s_end = head + ncar * (car + gap)
        xa_, xb_ = head / L - 3, s_end / L + 3
        pts = [P(xa_, y1 + 2, tA + tB * xa_), P(xb_, y1 + 2, tA + tB * xb_),
               P(xa_, y0 - 2, tA + tB * xa_), P(xb_, y0 - 2, tA + tB * xb_)]
        cx0 = int(max(min(p[0] for p in pts) - 4, 0)); cx1 = int(min(max(p[0] for p in pts) + 4, W))
        cy0 = int(max(min(p[1] for p in pts) - 4, 0)); cy1 = int(min(max(p[1] for p in pts) + 4, H))
        if cx1 <= cx0 or cy1 <= cy0:
            return
        cv_ = self._Canvas(cx0, cy0, cx1 - cx0, cy1 - cy0, 3)
        head_pt = None
        tail_pt = None
        body = fg((0.2, 0.22, 0.36))
        roof = fg((0.42, 0.34, 0.52))
        stripe = fg((1.0, 0.42, 0.14))
        for ci in range(ncar):
            s0 = head + ci * (car + gap)
            s1 = s0 + car
            xa, xb = s0 / L, s1 / L
            za, zb = tA + tB * xa, tA + tB * xb
            A = (xa + nx * o, za + nz * o)
            Bp = (xb + nx * o, zb + nz * o)
            Ab = (xa - nx * o, za - nz * o)
            Bb = (xb - nx * o, zb - nz * o)
            # roof (catches the violet sky) with a warm rim along its sunward (left/back) edge
            cv_.poly([P(A[0], y1, A[1]), P(Bp[0], y1, Bp[1]), P(Bb[0], y1, Bb[1]), P(Ab[0], y1, Ab[1])], roof)
            cv_.line(P(Ab[0], y1, Ab[1]), P(Bb[0], y1, Bb[1]), fg((1.2, 0.72, 0.62)), 0.3)
            # side (facing camera): dark steel body
            cv_.poly([P(A[0], y0, A[1]), P(Bp[0], y0, Bp[1]), P(Bp[0], y1, Bp[1]), P(A[0], y1, A[1])], body)
            # rounded car ends: darker end caps
            for (ea, eb) in ((s0, s0 + 0.6), (s1 - 0.6, s1)):
                pa = (ea / L + nx * (o + 0.02), tA + tB * ea / L + nz * (o + 0.02))
                pb = (eb / L + nx * (o + 0.02), tA + tB * eb / L + nz * (o + 0.02))
                cv_.poly([P(pa[0], y0, pa[1]), P(pb[0], y0, pb[1]), P(pb[0], y1, pb[1]), P(pa[0], y1, pa[1])],
                         fg((0.08, 0.08, 0.16)))
            # orange line stripe under the windows (JR-like livery)
            cv_.poly([P(A[0], y0 + 1.3, A[1]), P(Bp[0], y0 + 1.3, Bp[1]), P(Bp[0], y0 + 1.9, Bp[1]),
                      P(A[0], y0 + 1.9, A[1])], stripe, e=(0.12, 0.04, 0.0))
            # window row: individual lit panes separated by pillars, doors darker
            pitch = 2.3
            k = 0
            u = s0 + 1.4
            while u + 1.5 < s1 - 1.0:
                ua, ub = u, u + 1.5
                door = (k % 4 == 1)
                wa, wb = ua / L, ub / L
                pa = (wa + nx * (o + 0.05), tA + tB * wa + nz * (o + 0.05))
                pb = (wb + nx * (o + 0.05), tA + tB * wb + nz * (o + 0.05))
                yl, yh = (y0 + 0.6, y0 + 5.9) if door else (y0 + 2.4, y0 + 5.6)
                em_ = (2.4, 1.75, 0.95) if not door else (1.3, 0.95, 0.55)
                cv_.poly([P(pa[0], yl, pa[1]), P(pb[0], yl, pb[1]), P(pb[0], yh, pb[1]), P(pa[0], yh, pa[1])],
                         (1.0, 0.86, 0.62), e=em_)
                u += pitch
                k += 1
            if ci == 0:
                head_pt = (xa + nx * o, za + nz * o)
            if ci == ncar - 1:
                tail_pt = (xb + nx * o, zb + nz * o)
            # pantograph on every other car
            if ci % 2 == 1:
                pm_ = ((xa + xb) / 2, (za + zb) / 2)
                cv_.line(P(pm_[0], y1, pm_[1]), P(pm_[0] + 1.2, y1 + 1.4, pm_[1]), fg((0.1, 0.1, 0.2)), 0.25)
                cv_.line(P(pm_[0] + 1.2, y1 + 1.4, pm_[1]), P(pm_[0] - 0.4, y1 + 2.0, pm_[1]), fg((0.1, 0.1, 0.2)), 0.25)
        pm, E = cv_.result()
        dirn = 1.0 if v > 0 else -1.0
        if v < 0:
            head_pt, tail_pt = tail_pt, head_pt
        a = pm[..., 3:4]
        sl = (slice(cy0, cy1), slice(cx0, cx1))
        img[sl] = img[sl] * (1 - a) + pm[..., :3]
        em[sl] = em[sl] * (1 - a) + E
        if head_pt is not None:
            hp = P(head_pt[0], dk + 2.0, head_pt[1])
            C.splat(em, [hp[0]], [hp[1]], 0.0013 * W, (1.0, 0.97, 0.88), 5.0)
            C.splat(em, [hp[0]], [hp[1]], 0.007 * W, (1.0, 0.85, 0.6), 0.45)
            # headlight beam along the track ahead + its reflection sliding along the rails
            for k in range(1, 14):
                xk = head_pt[0] - dirn * k * 2.2
                zk = tA + tB * xk
                bp = P(xk, dk + 1.6, zk + nz * 0.3)
                C.splat(em, [bp[0]], [bp[1]], (0.002 + 0.0006 * k) * W, (1.0, 0.85, 0.6), 0.22 * math.exp(-k / 6.0))
            for k, off in enumerate((-0.75, 0.75)):
                xg = head_pt[0] - dirn * 9.0
                gp = P(xg + nx * off, dk + 0.3, tA + tB * xg + nz * off)
                C.splat(em, [gp[0]], [gp[1]], 0.0012 * W, (1.0, 0.92, 0.8), 3.0)
                C.splat(em, [gp[0]], [gp[1]], 0.005 * W, (1.0, 0.8, 0.6), 0.2)
        gk = 0.55 if far else 1.0
        for ci in range(ncar):
            for fr_ in (0.2, 0.5, 0.8):
                s_ = head + ci * (car + gap) + fr_ * car
                xw = s_ / L + nx * o
                zw = tA + tB * s_ / L + nz * o
                q = P(xw, dk + 4.0, zw)
                C.splat(em, [q[0]], [q[1]], 0.006 * W, (1.0, 0.66, 0.34), 0.1 * gk)
                if not far:
                    for off in ((1.1, 2.6) if side > 0 else (-1.1, -2.6)):
                        g2 = P(s_ / L + nx * off, dk + 0.3, tA + tB * s_ / L + nz * off)
                        C.splat(em, [g2[0]], [g2[1]], 0.0011 * W, (1.0, 0.8, 0.55), 0.55)
        if tail_pt is not None:
            # warm glow trail left behind along the deck (the lit train just passed)
            for k in range(1, 12):
                xk = tail_pt[0] + dirn * k * 3.0
                q = P(xk, dk + 2.5, tA + tB * xk)
                C.splat(em, [q[0]], [q[1]], (0.004 + 0.0004 * k) * W, (1.0, 0.55, 0.3),
                        0.16 * gk * math.exp(-k / 4.0))
            tp = P(tail_pt[0], dk + 1.6, tail_pt[1])
            C.splat(em, [tp[0]], [tp[1]], 0.001 * W, (1.0, 0.1, 0.06), 2.5)
            C.splat(em, [tp[0]], [tp[1]], 0.004 * W, (1.0, 0.15, 0.1), 0.25)
        if far or not glint:
            return
        # afterglow glints travelling along the rails as the camera trucks (specular points on the steel)
        for j, xg0 in enumerate((-60.0, 40.0, 150.0)):
            xg = xg0 + 5.5 * truck + 9.0 * t
            for off in (-2.6, -1.1, 1.1, 2.6):
                gp = P(xg + nx * off, dk + 0.3, tA + tB * xg + nz * off)
                if 0 <= gp[0] < W:
                    C.splat(em, [gp[0]], [gp[1]], 0.0008 * W, (1.0, 0.7, 0.45), 0.9)
                    C.splat(em, [gp[0]], [gp[1]], 0.004 * W, (1.0, 0.6, 0.4), 0.06)

    # ------------------------------------------------------------------ foreground (painted near plane)
    def _build_foreground(self):
        """Near rooftops framing the vista: lower-left a rooftop with parapet, railing, water tank on a
        steel stand, AC units and a lattice telecom mast; lower-right a lower rooftop corner with a fence
        and the back of a billboard frame. Power / telecom cables sag from the mast across the lower frame.
        Warm rim light on sunward (left / top) edges, cool teal ambient on camera-facing faces."""
        W, H = self.W, self.H
        mx = int(self.f * TRUCK / self.FG_Z + YAW * W + 8)
        pw = W + 2 * mx
        self.fg_mx_tmp = mx
        top = int(0.18 * H)
        h = H - top
        ss = 3
        col = np.zeros((h * ss, pw * ss, 3), np.float32)
        al = np.zeros((h * ss, pw * ss), np.float32)
        emi = np.zeros((h * ss, pw * ss, 3), np.float32)
        onm = np.full((h * ss, pw * ss), -10.0, np.float32)

        def Q(x, y):          # frame-fraction coords -> plate supersampled px
            return (int(round((x * W + mx) * ss)), int(round((y * H - top) * ss)))

        HH, WW = al.shape

        def _apply(m, ox, oy, c, a, e=None):
            mb = m.astype(bool)
            if not mb.any():
                return
            sl = (slice(oy, oy + m.shape[0]), slice(ox, ox + m.shape[1]))
            sc, sa, se = col[sl], al[sl], emi[sl]
            sc[mb] = sc[mb] * (1 - a) + np.asarray(c, np.float32) * a
            sa[mb] = sa[mb] * (1 - a) + a
            if e is not None:
                se[mb] += np.asarray(e, np.float32)
            else:
                se[mb] *= (1 - a)

        def _box(qs, pad):
            qs = np.asarray(qs)
            x0, y0 = max(int(qs[:, 0].min()) - pad, 0), max(int(qs[:, 1].min()) - pad, 0)
            x1, y1 = min(int(qs[:, 0].max()) + pad + 1, WW), min(int(qs[:, 1].max()) + pad + 1, HH)
            return x0, y0, x1, y1

        def fill(pts, c, a=1.0):
            qs = np.array([Q(*p) for p in pts], np.int32)
            x0, y0, x1, y1 = _box(qs, 2)
            if x1 <= x0 or y1 <= y0:
                return
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillPoly(m, [qs - [x0, y0]], 1, cv2.LINE_8)
            _apply(m, x0, y0, c, a)

        def ln(p0, p1, c, wd, a=1.0, e=None):
            t_ = max(1, int(round(wd * H / 1080 * ss)))
            qs = np.array([Q(*p0), Q(*p1)], np.int32)
            x0, y0, x1, y1 = _box(qs, t_ + 2)
            if x1 <= x0 or y1 <= y0:
                return
            m = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.line(m, tuple(qs[0] - [x0, y0]), tuple(qs[1] - [x0, y0]), 1, t_)
            _apply(m, x0, y0, c, a, e)

        def rect(q0, q1):
            return (slice(max(min(q0[1], q1[1]), 0), min(max(q0[1], q1[1]) + 1, HH)),
                    slice(max(min(q0[0], q1[0]), 0), min(max(q0[0], q1[0]) + 1, WW)))

        def pl(pts, c, wd, e=None):
            for p0, p1 in zip(pts[:-1], pts[1:]):
                ln(p0, p1, c, wd, e=e)

        def stamp_text(text, fx0, fy0, fw, fh, c, e=None, a=1.0, vertical=False, replace_e=False, glow=0.0):
            """render real Japanese text fitted into the box (fractions of W/H), painted into the canvas"""
            wpx, hpx = int(fw * W * ss), int(fh * H * ss)
            m = FGH.text_mask(text, (wpx if vertical else hpx) * 1.0, vertical=vertical)
            m = FGH.fit(m, wpx, hpx) * a
            x0q, y0q = Q(fx0, fy0)
            ys_ = slice(max(y0q, 0), min(y0q + hpx, HH))
            xs_ = slice(max(x0q, 0), min(x0q + wpx, WW))
            if ys_.stop <= ys_.start or xs_.stop <= xs_.start:
                return
            mm = m[ys_.start - y0q:ys_.stop - y0q, xs_.start - x0q:xs_.stop - x0q]
            col[ys_, xs_] = col[ys_, xs_] * (1 - mm[..., None]) + np.asarray(c, np.float32) * mm[..., None]
            al[ys_, xs_] = np.maximum(al[ys_, xs_], mm)
            if e is not None:
                ee = np.asarray(e, np.float32) * mm[..., None]
                if replace_e:
                    emi[ys_, xs_] = emi[ys_, xs_] * (1 - mm[..., None]) + ee
                else:
                    emi[ys_, xs_] += ee
                if glow > 0:
                    gb = cv2.GaussianBlur(mm, (0, 0), glow * H * ss)
                    emi[ys_, xs_] += np.asarray(e, np.float32) * gb[..., None] * 0.35
            onm[ys_, xs_] = np.where(mm > 0.05, -10.0, onm[ys_, xs_])

        rng = np.random.default_rng(91)
        wall = np.array([0.06, 0.08, 0.15], np.float32)       # camera-facing wall: cool teal-navy
        wall2 = np.array([0.04, 0.045, 0.1], np.float32)
        roof = np.array([0.13, 0.12, 0.24], np.float32)       # roof deck reflecting the violet sky
        steel = np.array([0.05, 0.055, 0.11], np.float32)
        rim = np.array([1.25, 0.66, 0.4], np.float32)
        rimp = np.array([0.95, 0.5, 0.55], np.float32)
        # ---------------- left rooftop
        xl0, xl1 = -0.25, 0.3
        yr = 0.855        # parapet top (front edge)
        yb = 0.835        # far roof edge (roof deck seen slightly from above)
        fill([(xl0, yb), (xl1 - 0.012, yb), (xl1, yr), (xl1, 1.01), (xl0, 1.01)], wall)
        fill([(xl0, yb), (xl1 - 0.012, yb), (xl1, yr), (xl0, yr)], roof)
        # side wall (facing right, away from the glow)
        fill([(xl1, yr), (xl1 + 0.022, yr - 0.012), (xl1 + 0.022, 1.01), (xl1, 1.01)], wall2)
        # wall panel seams and floor slab lines (catching a little sky light)
        for x in np.arange(xl0, xl1, 0.034):
            ln((x, yr + 0.004), (x, 1.01), wall * 1.3, 0.8)
        for fy in (0.897, 0.957):
            ln((xl0, fy), (xl1, fy), wall * 1.45, 1.2)
        # office windows: panes in tenant runs - warm tungsten / fluorescent / a few cool, dark ones mirror
        # the dusk sky (lighter at the top); blinds half-drawn on some; a few runs switch on during the shot
        pane = 0.0155
        for fy in (0.905, 0.965):
            run, state = 0, None
            for x in np.arange(xl0 + 0.01, xl1 - 0.014, pane):
                if run <= 0:
                    u_ = rng.random()
                    state = 'warm' if u_ < 0.3 else ('fluo' if u_ < 0.5 else ('cool' if u_ < 0.56 else 'dark'))
                    run = int(rng.integers(2, 8))
                    ont = -10.0 if rng.random() < 0.55 else float(rng.uniform(0.6, 4.8))
                    blind = rng.uniform(0.15, 0.6) if rng.random() < 0.4 else 0.0
                run -= 1
                x0q, x1q = x + 0.0012, x + pane - 0.0012
                y0q, y1q = fy + 0.002, fy + 0.03
                fill([(x0q, y0q), (x1q, y0q), (x1q, y1q), (x0q, y1q)], np.array([0.1, 0.1, 0.22], np.float32))
                fill([(x0q, y0q), (x1q, y0q), (x1q, y0q + 0.01), (x0q, y0q + 0.01)],
                     np.array([0.26, 0.18, 0.34], np.float32))
                fill([(x0q, y1q - 0.006), (x1q, y1q - 0.006), (x1q, y1q), (x0q, y1q)],
                     np.array([0.05, 0.05, 0.12], np.float32))
                if state != 'dark':
                    c_ = {'warm': np.array([1.1, 0.6, 0.28]), 'fluo': np.array([1.0, 0.94, 0.8]),
                          'cool': np.array([0.62, 0.8, 1.05])}[state] * rng.uniform(0.45, 0.6)
                    yb_ = y0q + (y1q - y0q) * blind
                    mb = rect(Q(x0q, yb_), Q(x1q, y1q))
                    emi[mb] = c_
                    onm[mb] = ont
                    if blind > 0:
                        mb2 = rect(Q(x0q, y0q), Q(x1q, yb_))
                        emi[mb2] = c_ * 0.35
                        onm[mb2] = ont
        # mullions over the windows
        for fy in (0.905, 0.965):
            for x in np.arange(xl0 + 0.01, xl1 - 0.012, pane):
                ln((x, fy), (x, fy + 0.03), wall, 1.4)
        # company sign on the parapet wall: lit white panel with real Japanese lettering (red)
        sx0_, sx1_, sy0_, sy1_ = 0.03, 0.19, 0.864, 0.891
        fill([(sx0_, sy0_), (sx1_, sy0_), (sx1_, sy1_), (sx0_, sy1_)], np.array([0.2, 0.2, 0.26], np.float32))
        mb = rect(Q(sx0_ + 0.0015, sy0_ + 0.002), Q(sx1_ - 0.0015, sy1_ - 0.002))
        emi[mb] = np.array([0.95, 0.93, 0.86], np.float32) * 0.5
        col[mb] = np.array([0.5, 0.5, 0.5], np.float32)
        stamp_text('東雲ビルディング', sx0_ + 0.006, sy0_ + 0.004, sx1_ - sx0_ - 0.012, sy1_ - sy0_ - 0.008,
                   c=(0.5, 0.05, 0.04), e=(0.75, 0.06, 0.04), replace_e=True)
        # parapet coping with a hot rim along the top edge
        ln((xl0, yr), (xl1, yr), wall * 1.3, 3.0)
        ln((xl0, yr - 0.0015), (xl1 - 0.002, yr - 0.0015), rim, 2.2, e=np.array([0.5, 0.22, 0.1], np.float32))
        ln((xl1, yr), (xl1 + 0.022, yr - 0.012), rim * 0.6, 1.2)
        # railing along the roof edge (posts + two rails), rim-lit tops
        rh = 0.045
        for x in np.arange(xl0, xl1, 0.018):
            ln((x, yr), (x, yr - rh), steel, 2.0)
            ln((x - 0.0008, yr), (x - 0.0008, yr - rh), rim * 0.55, 0.8)
        ln((xl0, yr - rh), (xl1, yr - rh), steel, 2.2)
        ln((xl0, yr - rh - 0.0014), (xl1, yr - rh - 0.0014), rim, 1.1)
        ln((xl0, yr - rh * 0.5), (xl1, yr - rh * 0.5), steel, 1.4)
        # AC units on the roof deck (boxes with fan grilles and slats)
        for (ax, aw, ah) in ((0.04, 0.03, 0.03), (0.075, 0.025, 0.026), (0.225, 0.035, 0.034), (-0.08, 0.04, 0.03)):
            fill([(ax, yb - ah), (ax + aw, yb - ah), (ax + aw, yb + 0.004), (ax, yb + 0.004)], wall * 1.25)
            ln((ax, yb - ah), (ax + aw, yb - ah), rimp, 1.2)
            ln((ax - 0.0006, yb - ah), (ax - 0.0006, yb + 0.004), rim * 0.8, 1.0)
            for yy in np.linspace(yb - ah + 0.006, yb, 4):
                ln((ax + 0.003, yy), (ax + aw - 0.003, yy), wall * 0.8, 0.8)
        # water tank on a steel stand
        tx0, tx1 = 0.235, 0.29
        ty_top, ty_bot = 0.63, 0.725
        for x in (tx0 + 0.004, tx1 - 0.004, (tx0 + tx1) / 2):
            ln((x, ty_bot), (x, yb), steel, 3.0)
        ln((tx0 + 0.004, ty_bot + 0.03), (tx1 - 0.004, yb - 0.01), steel, 1.3)
        ln((tx1 - 0.004, ty_bot + 0.03), (tx0 + 0.004, yb - 0.01), steel, 1.3)
        ln((tx0 + 0.004, ty_bot + 0.03), (tx1 - 0.004, ty_bot + 0.03), steel, 1.8)
        tank = [(tx0, ty_top + 0.012), (tx0 + 0.006, ty_top), (tx1 - 0.006, ty_top), (tx1, ty_top + 0.012),
                (tx1, ty_bot), (tx0, ty_bot)]
        fill(tank, np.array([0.1, 0.12, 0.22], np.float32))
        for yy in np.linspace(ty_top + 0.02, ty_bot - 0.01, 5):
            ln((tx0, yy), (tx1, yy), np.array([0.07, 0.085, 0.17]), 1.2)
        ln((tx0 - 0.0005, ty_top + 0.012), (tx0 - 0.0005, ty_bot), rim, 1.6)      # sunward rim
        ln((tx0, ty_top + 0.012), (tx0 + 0.006, ty_top), rim, 1.4)
        ln((tx0 + 0.006, ty_top), (tx1 - 0.006, ty_top), rimp, 1.2)
        # ladder
        for yy in np.arange(ty_bot, yb, 0.012):
            ln((tx1 + 0.004, yy), (tx1 + 0.011, yy), steel, 1.0)
        ln((tx1 + 0.004, ty_top + 0.01), (tx1 + 0.004, yb), steel, 1.0)
        ln((tx1 + 0.011, ty_top + 0.01), (tx1 + 0.011, yb), steel, 1.0)
        # ---- rooftop clutter on the near roof deck: stair house with a lit door, pipes, vents, dish
        roofc = np.array([0.2, 0.17, 0.33], np.float32)       # top faces catch the violet zenith
        sh0, sh1, shy = -0.036, 0.034, 0.755
        fill([(sh0, shy), (sh1, shy), (sh1, yb + 0.003), (sh0, yb + 0.003)], wall * 1.15)
        fill([(sh0 - 0.004, shy - 0.006), (sh1 + 0.004, shy - 0.006), (sh1 + 0.004, shy), (sh0 - 0.004, shy)], roofc)
        ln((sh0 - 0.004, shy - 0.006), (sh1 + 0.004, shy - 0.006), rim, 1.6)
        ln((sh0 - 0.0045, shy - 0.006), (sh0 - 0.0045, yb), rim * 0.8, 1.3)
        # lit door (warm, half open) + lamp
        dx0, dx1 = sh0 + 0.036, sh0 + 0.054
        mb = rect(Q(dx0, shy + 0.03), Q(dx1, yb))
        col[mb] = np.array([0.4, 0.3, 0.25], np.float32)
        emi[mb] = np.array([1.1, 0.7, 0.35], np.float32) * 0.7
        mb = rect(Q(dx0 + 0.011, shy + 0.03), Q(dx1, yb))
        emi[mb] = np.array([1.1, 0.7, 0.35], np.float32) * 0.28
        mb = rect(Q(dx0 + 0.006, shy + 0.02), Q(dx0 + 0.012, shy + 0.025))
        emi[mb] = np.array([1.3, 1.1, 0.8], np.float32) * 1.5
        # signage tube on the stair house: small vertical neon
        stamp_text('屋上', sh0 + 0.008, shy + 0.012, 0.014, 0.05, c=(0.9, 0.9, 1.0), e=(0.45, 0.9, 1.3),
                   vertical=True, glow=0.004)
        # pipes running along the deck with rim-lit tops and saddle supports
        for py_, pw_ in ((yb - 0.004, 2.6), (yb - 0.011, 2.0)):
            ln((-0.25, py_), (0.23, py_), wall * 1.1, pw_)
            ln((-0.25, py_ - 0.0017), (0.23, py_ - 0.0017), rim * 0.75, 0.9)
            for x in np.arange(-0.24, 0.23, 0.045):
                ln((x, py_), (x, yb + 0.002), steel, 1.4)
        # vent stacks with caps
        for vx, vh in ((0.115, 0.03), (0.128, 0.022), (0.142, 0.026)):
            ln((vx, yb), (vx, yb - vh), wall * 1.2, 3.0)
            ln((vx - 0.0012, yb), (vx - 0.0012, yb - vh), rim * 0.8, 1.0)
            fill([(vx - 0.004, yb - vh - 0.004), (vx + 0.004, yb - vh - 0.004), (vx + 0.003, yb - vh),
                  (vx - 0.003, yb - vh)], wall * 1.3)
            ln((vx - 0.004, yb - vh - 0.004), (vx + 0.004, yb - vh - 0.004), rim, 1.0)
        # small satellite dish on a post
        ln((0.2, yb), (0.2, yb - 0.022), steel, 1.6)
        cv2.ellipse(col, Q(0.2, yb - 0.03), (int(0.006 * W * ss), int(0.011 * H * ss)), -25, 0, 360,
                    tuple(float(c) for c in wall * 1.6), -1)
        cv2.ellipse(al, Q(0.2, yb - 0.03), (int(0.006 * W * ss), int(0.011 * H * ss)), -25, 0, 360, 1.0, -1)
        cv2.ellipse(col, Q(0.2, yb - 0.03), (int(0.006 * W * ss), int(0.011 * H * ss)), -25, 110, 240,
                    tuple(float(c) for c in rim), max(1, int(1.3 * ss * H / 1080)))
        # AC unit fan grilles (lighter discs) + top faces
        for (ax, aw, ah) in ((0.04, 0.03, 0.03), (0.075, 0.025, 0.026), (0.225, 0.035, 0.034), (-0.08, 0.04, 0.03)):
            fill([(ax, yb - ah - 0.004), (ax + aw, yb - ah - 0.004), (ax + aw, yb - ah), (ax, yb - ah)], roofc)
            ln((ax, yb - ah - 0.004), (ax + aw, yb - ah - 0.004), rim, 1.1)
            cv2.circle(col, Q(ax + aw * 0.5, yb - ah * 0.5), int(min(aw * W, ah * H) * 0.36 * ss),
                       tuple(float(c) for c in wall * 1.8), max(1, int(1.2 * ss * H / 1080)))
        # vertical neon sign hung on the front wall (karaoke), pink tubes + warm backing
        nx0, nx1, ny0, ny1 = 0.262, 0.286, 0.868, 1.0
        fill([(nx0, ny0), (nx1, ny0), (nx1, ny1), (nx0, ny1)], np.array([0.08, 0.05, 0.12], np.float32))
        ln((nx0, ny0), (nx1, ny0), rim, 1.4)
        ln((nx0 - 0.0006, ny0), (nx0 - 0.0006, ny1), rim * 0.8, 1.2)
        stamp_text('カラオケ', nx0 + 0.004, ny0 + 0.008, nx1 - nx0 - 0.008, ny1 - ny0 - 0.012,
                   c=(1.0, 0.75, 0.95), e=(1.6, 0.35, 1.0), vertical=True, glow=0.006)
        for yy in (ny0 + 0.003, ny1 - 0.004):
            ln((nx0 + 0.002, yy), (nx1 - 0.002, yy), np.array([1.0, 0.8, 0.9]), 0.9, e=np.array([1.2, 0.3, 0.8]))
        # wall details: drain pipes, wall-mounted AC units, slab glints
        for x in (-0.19, 0.005, 0.215):
            ln((x, yr + 0.004), (x, 1.01), wall * 1.5, 2.2)
            ln((x - 0.001, yr + 0.004), (x - 0.001, 1.01), rimp * 0.45, 0.8)
        for (ux_, uy_) in ((-0.15, 0.94), (0.1, 0.94), (0.14, 1.0), (-0.05, 1.0)):
            fill([(ux_, uy_ - 0.022), (ux_ + 0.026, uy_ - 0.022), (ux_ + 0.026, uy_), (ux_, uy_)], wall * 1.7)
            ln((ux_, uy_ - 0.022), (ux_ + 0.026, uy_ - 0.022), rimp * 0.8, 1.0)
            cv2.circle(col, Q(ux_ + 0.009, uy_ - 0.011), int(0.008 * H * ss), tuple(float(c) for c in wall * 1.1), -1)
        # warm glints on the water-tank ladder rungs (sunward ends)
        for yy in np.arange(0.725, yb, 0.012):
            mb = rect(Q(0.294 - 0.0008, yy - 0.001), Q(0.294 + 0.001, yy + 0.0006))
            emi[mb] += np.array([1.2, 0.7, 0.4], np.float32) * 0.9
        for yy in np.linspace(0.64, 0.72, 5):
            mb = rect(Q(0.2345, yy), Q(0.2355, yy + 0.004))
            emi[mb] += np.array([1.2, 0.72, 0.45], np.float32) * 0.8
        self._paint_near_facade(W, H, ss, col, al, emi, onm, Q, rect, fill, ln, stamp_text, xl0, xl1, yr, rh,
                                wall, rim, rimp, steel, nx0, nx1, ny0, ny1)
        self._paint_near_roofs(W, H, ss, col, al, emi, onm, Q, rect, fill, ln, stamp_text, wall, roof, rim, rimp,
                               steel)
        # ---- switch to the second (farther) near plane: the lattice mast on the building behind, its cables
        P1 = (col, al, emi, onm)
        col = np.zeros_like(col); al = np.zeros_like(al); emi = np.zeros_like(emi)
        onm = np.full_like(onm, -10.0)
        # lattice telecom mast (its base hides behind the near rooftop)
        mxf, mbase, mtip = 0.17, 0.9, 0.3
        wb_, wt_ = 0.022, 0.006
        lv = np.linspace(mbase, mtip, 16)
        for k in range(len(lv) - 1):
            ya_, yb2 = lv[k], lv[k + 1]
            ua = (mbase - ya_) / (mbase - mtip)
            ub = (mbase - yb2) / (mbase - mtip)
            wa, wbb = wb_ + (wt_ - wb_) * ua, wb_ + (wt_ - wb_) * ub
            ln((mxf - wa, ya_), (mxf - wbb, yb2), steel, 2.4)
            ln((mxf + wa, ya_), (mxf + wbb, yb2), steel, 2.4)
            ln((mxf - wa, ya_), (mxf + wbb, yb2), steel, 1.0)
            ln((mxf + wa, ya_), (mxf - wbb, yb2), steel, 1.0)
            ln((mxf - wbb, yb2), (mxf + wbb, yb2), steel, 1.0)
            ln((mxf - wa - 0.0012, ya_), (mxf - wbb - 0.0012, yb2), rim, 1.0)
        ln((mxf, mtip), (mxf, mtip - 0.06), steel, 1.6)
        ln((mxf - 0.0008, mtip), (mxf - 0.0008, mtip - 0.06), rim, 0.7)
        # antenna panels / dish on platforms
        for yy, side in ((0.42, -1), (0.5, 1), (0.6, -1)):
            u_ = (mbase - yy) / (mbase - mtip)
            wv = wb_ + (wt_ - wb_) * u_
            ln((mxf - wv - 0.008, yy), (mxf + wv + 0.008, yy), steel, 2.0)
            px_ = mxf + side * (wv + 0.006)
            fill([(px_ - 0.003, yy - 0.03), (px_ + 0.003, yy - 0.03), (px_ + 0.003, yy - 0.003),
                  (px_ - 0.003, yy - 0.003)], steel * 1.3)
            ln((px_ - 0.0032, yy - 0.03), (px_ - 0.0032, yy - 0.003), rim, 0.9)
        cv2.ellipse(col, Q(mxf + 0.016, 0.545), (int(0.01 * W * ss), int(0.012 * W * ss)), 0, 0, 360,
                    tuple(float(c) for c in steel * 1.4), -1)
        cv2.ellipse(al, Q(mxf + 0.016, 0.545), (int(0.01 * W * ss), int(0.012 * W * ss)), 0, 0, 360, 1.0, -1)
        cv2.ellipse(col, Q(mxf + 0.016, 0.545), (int(0.01 * W * ss), int(0.012 * W * ss)), 0, 100, 250,
                    tuple(float(c) for c in rim), max(1, int(1.2 * ss * H / 1080)))
        self.mast_top = (mxf, mtip - 0.06)
        P2 = (col, al, emi, onm)
        col, al, emi, onm = P1
        # ---------------- right rooftop (lower, closer to the frame edge)
        xr0_ = 0.86
        yr2 = 0.95
        fill([(xr0_ + 0.02, yr2 - 0.012), (1.2, yr2 - 0.012), (1.2, 1.01), (xr0_, 1.01), (xr0_, yr2)], wall)
        fill([(xr0_, yr2), (xr0_ + 0.02, yr2 - 0.012), (1.2, yr2 - 0.012), (1.2, yr2)], roof)
        ln((xr0_, yr2), (1.2, yr2), rim * 0.8, 1.4)
        ln((xr0_ - 0.0006, yr2), (xr0_ - 0.0006, 1.01), rim * 0.7, 1.2)
        for fy in (0.955,):
            x = xr0_ + 0.01
            while x < 1.18:
                wdt = rng.uniform(0.02, 0.06)
                on_ = rng.random() < 0.5
                c_ = np.array([1.1, 0.93, 0.75]) if rng.random() < 0.6 else np.array([1.15, 0.6, 0.3])
                mb = rect(Q(x, fy), Q(x + wdt, fy + 0.028))
                col[mb] = np.array([0.08, 0.1, 0.2], np.float32)
                if on_:
                    emi[mb] = c_ * 0.5
                    onm[mb] = -10.0 if rng.random() < 0.5 else rng.uniform(1.0, 4.8)
                x += wdt + rng.uniform(0.006, 0.03)
        # fence on the right roof
        fh = 0.04
        for x in np.arange(xr0_ + 0.01, 1.2, 0.014):
            ln((x, yr2 - 0.012), (x, yr2 - 0.012 - fh), steel, 1.6)
        for yy in (yr2 - 0.012 - fh, yr2 - 0.012 - fh * 0.5):
            ln((xr0_ + 0.01, yy), (1.2, yy), steel, 1.4)
        ln((xr0_ + 0.01, yr2 - 0.012 - fh - 0.0012), (1.2, yr2 - 0.012 - fh - 0.0012), rimp, 0.9)
        # rooftop billboard facing the camera: internally lit poster (sky-blue gradient, dark lettering),
        # steel frame + walkway + lattice legs, floodlight arms on top
        bx0_, bx1_ = 0.9, 1.14
        by0_, by1_ = 0.745, 0.84
        fill([(bx0_ - 0.004, by0_ - 0.006), (bx1_ + 0.004, by0_ - 0.006), (bx1_ + 0.004, by1_ + 0.004),
              (bx0_ - 0.004, by1_ + 0.004)], steel)
        # poster face: vertical gradient, painted as emission (it glows against the dusk)
        q0, q1 = Q(bx0_, by0_), Q(bx1_, by1_)
        hh_ = q1[1] - q0[1]
        gy_ = np.linspace(0, 1, hh_, dtype=np.float32)[:, None, None]
        top_c = np.array([0.55, 0.78, 1.05], np.float32)
        bot_c = np.array([1.0, 0.95, 0.9], np.float32)
        grad = top_c * (1 - gy_) + bot_c * gy_
        sl_ = (slice(q0[1], q1[1]), slice(q0[0], min(q1[0], WW)))
        wv_ = sl_[1].stop - sl_[1].start
        col[sl_] = grad[:, :, :] * 0.35
        al[sl_] = 1.0
        emi[sl_] = np.broadcast_to(grad * 0.62, (hh_, wv_, 3))
        onm[sl_] = -10.0
        # a white jet-contrail swoosh on the poster
        for k in range(3):
            pts_ = C.catenary((bx0_ + 0.02, by0_ + 0.06 - 0.006 * k), (bx1_ - 0.01, by0_ + 0.012 - 0.004 * k), -0.02, n=30)
            for p0_, p1_ in zip(pts_[:-1], pts_[1:]):
                ln(tuple(p0_), tuple(p1_), np.array([1.0, 1.0, 1.0]), 1.4 - 0.4 * k, e=np.array([0.5, 0.5, 0.5]))
        stamp_text('いつか、あの空へ。', bx0_ + 0.014, by0_ + 0.016, 0.15, 0.03, c=(0.03, 0.06, 0.2),
                   e=(0.0, 0.0, 0.0), replace_e=True)
        stamp_text('東雲エアライン', bx0_ + 0.016, by1_ - 0.03, 0.07, 0.016, c=(0.6, 0.08, 0.08),
                   e=(0.28, 0.02, 0.02), replace_e=True)
        # frame edges catching the afterglow
        ln((bx0_ - 0.004, by0_ - 0.006), (bx1_ + 0.004, by0_ - 0.006), rim, 1.6)
        ln((bx0_ - 0.0045, by0_ - 0.006), (bx0_ - 0.0045, by1_ + 0.004), rim, 1.4)
        # walkway + rails under the poster
        ln((bx0_ - 0.01, by1_ + 0.008), (bx1_, by1_ + 0.008), steel, 2.2)
        for x in np.arange(bx0_ - 0.01, bx1_, 0.008):
            ln((x, by1_ + 0.008), (x, by1_ - 0.006), steel, 0.9)
        # lattice legs
        for x in np.linspace(bx0_ + 0.01, bx1_ - 0.01, 5):
            ln((x, by1_ + 0.008), (x, yr2 - 0.012), steel, 2.4)
            ln((x - 0.001, by1_ + 0.008), (x - 0.001, yr2 - 0.012), rim * 0.5, 0.8)
        for x0_, x1_ in zip(np.linspace(bx0_ + 0.01, bx1_ - 0.01, 5)[:-1], np.linspace(bx0_ + 0.01, bx1_ - 0.01, 5)[1:]):
            ln((x0_, by1_ + 0.008), (x1_, yr2 - 0.012), steel, 1.0)
            ln((x1_, by1_ + 0.008), (x0_, yr2 - 0.012), steel, 1.0)
        # floodlight arms with lamps (bright warm heads)
        self.flood = []
        for x in np.linspace(bx0_ + 0.02, bx1_ - 0.02, 4):
            ln((x, by0_ - 0.006), (x, by0_ - 0.02), steel, 1.3)
            ln((x, by0_ - 0.02), (x + 0.008, by0_ - 0.024), steel, 1.3)
            mb = rect(Q(x + 0.006, by0_ - 0.027), Q(x + 0.012, by0_ - 0.022))
            col[mb] = np.array([0.9, 0.9, 0.85], np.float32)
            al[mb] = 1.0
            emi[mb] = np.array([1.5, 1.4, 1.2], np.float32) * 1.5
        P1 = (col, al, emi, onm)
        col, al, emi, onm = P2
        # ---------------- cables from the mast sagging across the lower frame (catenary)
        cab = []
        for k, (ya, yb3, sag, wd) in enumerate(((0.43, 0.6, 0.11, 1.5), (0.5, 0.64, 0.12, 1.3),
                                                 (0.6, 0.7, 0.1, 1.6), (0.415, 0.52, 0.09, 1.1))):
            p0 = (mxf + 0.004, ya)
            p1 = (1.2, yb3)
            pts = C.catenary(p0, p1, sag, n=90)
            cab.append((pts, wd))
            pl([tuple(p) for p in pts], np.array([0.05, 0.05, 0.11]), wd)
            # specular: the upper edge of each wire catches the sky; brightest where it faces the sun
            for i in range(len(pts) - 1):
                x_ = pts[i][0]
                g = math.exp(-abs(x_ - SUN_X) / 0.14)
                c_ = rimp * (0.35 + 0.9 * g)
                ln((pts[i][0], pts[i][1] - 0.0009), (pts[i + 1][0], pts[i + 1][1] - 0.0009), c_, 0.6,
                   e=np.array([0.6, 0.35, 0.25], np.float32) * g * 0.8)
        # insulators / small birds on a wire
        for (bxw, k) in ((0.5, 0), (0.522, 0), (0.68, 1)):
            pts, _ = cab[k]
            j = int(np.argmin(np.abs(pts[:, 0] - bxw)))
            bx_, by_ = pts[j]
            cv2.ellipse(col, Q(bx_, by_ - 0.006), (int(0.0035 * W * ss), int(0.006 * H * ss)), 0, 0, 360,
                        (0.04, 0.04, 0.09), -1)
            cv2.ellipse(al, Q(bx_, by_ - 0.006), (int(0.0035 * W * ss), int(0.006 * H * ss)), 0, 0, 360, 1.0, -1)
            cv2.circle(col, Q(bx_ + 0.003, by_ - 0.013), int(0.0022 * W * ss), (0.04, 0.04, 0.09), -1)
            cv2.circle(al, Q(bx_ + 0.003, by_ - 0.013), int(0.0022 * W * ss), 1.0, -1)
            ln((bx_ - 0.004, by_ - 0.0075), (bx_ - 0.0005, by_ - 0.0125), rim * 0.8, 0.8)
        self.fg_mx = mx
        self.fg_top = top
        P2 = (col, al, emi, onm)

        def _pack(Pk):
            c_, a_, e_, o_ = Pk
            pm = cv2.resize(np.dstack([c_ * a_[..., None], a_]), (pw, h), interpolation=cv2.INTER_AREA)
            E = cv2.resize(e_, (pw, h), interpolation=cv2.INTER_AREA)
            on_ = cv2.resize(o_, (pw, h), interpolation=cv2.INTER_NEAREST)
            return (np.ascontiguousarray(pm.astype(np.float32)), np.ascontiguousarray(E.astype(np.float32)),
                    np.ascontiguousarray(on_.astype(np.float32)))

        self.fg, self.fg_E, self.fg_on = _pack(P1)
        self.mast, self.mast_E, self.mast_on = _pack(P2)
        self.fg_light = ((mxf * W + mx), (mtip - 0.06) * H - top)

    # ------------------------------------------------------------------ round 6: painted near facade + roofs
    def _paint_near_facade(self, W, H, ss, col, al, emi, onm, Q, rect, fill, ln, stamp_text, xl0, xl1, yr, rh,
                           wall, rim, rimp, steel, nx0, nx1, ny0, ny1):
        """The 東雲ビル / karaoke facade was a flat dark box: give it a painted value gradient (warm afterglow
        bounce under the parapet, cooler and darker toward the street), rain-streak weathering, a warm-lit
        sunward side wall with a hot corner edge, pilasters and sills catching the light, glinting railing
        posts, textured wall AC units, and the pink glow of the カラオケ sign spilling onto the wall."""
        rng = np.random.default_rng(606)
        # value gradient over the front wall (not the lit sign panels: those carry emission)
        sl = rect(Q(xl0, yr + 0.003), Q(xl1, 1.01))
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh > 1 and ww > 1:
            gy = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
            x0f = (sl[1].start / ss - self.fg_mx_tmp) / W
            gx = np.clip((np.linspace(x0f, x0f + ww / ss / W, ww, dtype=np.float32)[None, :, None] - xl0) /
                         (xl1 - xl0), 0, 1)
            m = ((al[sl] > 0.5) & (emi[sl].max(-1) < 0.05))[..., None]
            warm = np.array([0.2, 0.085, 0.08], np.float32) * (1 - gy) ** 2.2 * (0.35 + 0.65 * gx ** 1.5)
            cool = 0.9 + 0.2 * (1 - gy)
            col[sl] = np.where(m, col[sl] * cool + warm, col[sl])
        # weathering: faint darker rain streaks running down from the parapet and from the window sills
        for _ in range(70):
            x = rng.uniform(xl0, xl1)
            y0 = float(rng.choice([yr + 0.004, 0.936, 0.996]))
            ln((x, y0), (x, y0 + rng.uniform(0.01, 0.05)), wall * 0.55, rng.uniform(0.8, 2.2), a=0.22)
        # pilasters with a warm edge on their sunward (right) side; window sills catching light
        for x in np.arange(xl0 + 0.055, xl1 - 0.01, 0.11):
            ln((x, yr + 0.004), (x, 1.01), wall * 1.35, 3.2)
            ln((x + 0.0028, yr + 0.004), (x + 0.0028, 1.01), rim * 0.42, 0.9)
        for fy in (0.905, 0.965):
            ln((xl0, fy + 0.0315), (xl1, fy + 0.0315), rimp * 0.55, 1.1)
            ln((xl0, fy + 0.0005), (xl1, fy + 0.0005), wall * 0.5, 1.4)
        # sunward side wall: warm-lit, darkening down, hot corner edge, slab lines, a narrow window
        sw0, sw1 = xl1, xl1 + 0.022
        fill([(sw0, yr), (sw1, yr - 0.012), (sw1, 1.01), (sw0, 1.01)], np.array([0.42, 0.2, 0.24], np.float32))
        sl = rect(Q(sw0, yr - 0.012), Q(sw1, 1.01))
        hh = sl[0].stop - sl[0].start
        if hh > 1:
            gy = np.linspace(0, 1, hh, dtype=np.float32)[:, None, None]
            m = (al[sl] > 0.5)[..., None]
            col[sl] = np.where(m, col[sl] * (1.15 - 0.75 * gy), col[sl])
        for fy in (0.897, 0.957):
            ln((sw0, fy), (sw1, fy - 0.004), np.array([0.62, 0.3, 0.26], np.float32), 1.3)
        fill([(sw0 + 0.008, 0.91), (sw0 + 0.015, 0.908), (sw0 + 0.015, 0.94), (sw0 + 0.008, 0.941)],
             np.array([0.16, 0.09, 0.18], np.float32))
        ln((sw0 + 0.0085, 0.911), (sw0 + 0.0145, 0.909), np.array([1.0, 0.62, 0.45], np.float32), 0.9,
           e=np.array([0.5, 0.25, 0.12], np.float32))
        ln((sw0, yr), (sw0, 1.01), np.array([1.35, 0.8, 0.5], np.float32), 1.8,
           e=np.array([0.55, 0.28, 0.12], np.float32))
        ln((sw0, yr), (sw1, yr - 0.012), np.array([1.4, 0.85, 0.5], np.float32), 1.6,
           e=np.array([0.6, 0.3, 0.12], np.float32))
        # railing glints: the top rail and the post heads flare toward the sun
        for x in np.arange(xl0, xl1, 0.018):
            g = math.exp(-abs(SUN_X - x) / 0.22)
            ln((x, yr - rh - 0.0014), (min(x + 0.018, xl1), yr - rh - 0.0014), rim * (0.6 + 0.4 * g), 1.1,
               e=np.array([0.3, 0.14, 0.05], np.float32) * g * g)
            mb = rect(Q(x - 0.0012, yr - rh - 0.003), Q(x + 0.0008, yr - rh + 0.0005))
            emi[mb] += np.array([1.6, 1.05, 0.55], np.float32) * (0.25 + 1.4 * g)
        # the coping of the parapet: a continuous specular line brightening toward the sun
        for x in np.arange(xl0, xl1, 0.02):
            g = math.exp(-abs(SUN_X - x) / 0.2)
            ln((x, yr - 0.0022), (min(x + 0.02, xl1), yr - 0.0022), np.array([1.3, 0.8, 0.5], np.float32) * (0.5 + 0.5 * g),
               1.0, e=np.array([0.35, 0.18, 0.07], np.float32) * g * g)
        # textured wall AC units: slats, fan guard, rim, refrigerant pipe and a grime streak below
        for (ux_, uy_) in ((-0.15, 0.94), (0.1, 0.94), (0.14, 1.0), (-0.05, 1.0)):
            for k in range(5):
                yy = uy_ - 0.019 + k * 0.0038
                ln((ux_ + 0.017, yy), (ux_ + 0.0245, yy), wall * 1.1, 0.8)
            ln((ux_ + 0.026, uy_ - 0.022), (ux_ + 0.026, uy_), rim * 0.55, 1.0)
            ln((ux_ + 0.001, uy_ - 0.0215), (ux_ + 0.025, uy_ - 0.0215), np.array([1.1, 0.66, 0.5], np.float32), 0.8)
            ln((ux_ + 0.004, uy_), (ux_ + 0.004, uy_ + 0.03), steel * 1.4, 1.4)
            ln((ux_ + 0.0048, uy_), (ux_ + 0.0048, uy_ + 0.03), rimp * 0.35, 0.6)
            ln((ux_ + 0.013, uy_), (ux_ + 0.013, uy_ + 0.025), wall * 0.5, 4.0, a=0.3)
            cv2.circle(col, Q(ux_ + 0.009, uy_ - 0.011), int(0.0065 * H * ss), tuple(float(c) for c in wall * 0.7),
                       max(1, int(0.8 * ss * H / 1080)))
            for r_ in (0.0045, 0.003):
                cv2.circle(col, Q(ux_ + 0.009, uy_ - 0.011), int(r_ * H * ss), tuple(float(c) for c in wall * 1.6),
                           max(1, int(0.6 * ss * H / 1080)))
        # the pink karaoke glow spilling onto the wall around the sign
        sl = rect(Q(nx0 - 0.07, ny0 - 0.01), Q(min(nx1 + 0.04, xl1 + 0.022), 1.01))
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh > 1 and ww > 1:
            yy, xx = np.mgrid[sl[0].start:sl[0].stop, sl[1].start:sl[1].stop].astype(np.float32)
            qa, qb = Q(nx0, ny0), Q(nx1, ny1)
            dx_ = np.maximum(np.maximum(qa[0] - xx, xx - qb[0]), 0)
            dy_ = np.maximum(np.maximum(qa[1] - yy, yy - qb[1]), 0)
            d = np.sqrt(dx_ ** 2 + dy_ ** 2) / (ss * W)
            spill = (np.exp(-(d / 0.012) ** 2) * 0.7 + np.exp(-d / 0.03) * 0.3) * (d > 0)
            m = (al[sl] > 0.5) & (emi[sl].max(-1) < 0.3)
            spill = (spill * m).astype(np.float32)[..., None]
            emi[sl] += spill * np.array([0.75, 0.13, 0.45], np.float32) * 0.55
            col[sl] = col[sl] * (1 - 0.25 * spill) + spill * 0.25 * np.array([0.6, 0.2, 0.45], np.float32)

    def _paint_near_roofs(self, W, H, ss, col, al, emi, onm, Q, rect, fill, ln, stamp_text, wall, roof, rim, rimp,
                          steel):
        """Two lower near roofs across the bottom centre (the mid-rise sea there read as voxel noise): an
        apartment roof with laundry poles, a futon over the railing, a stair house with a lit door, AC units
        and its name plate (青空ハイツ); a shop-building roof with a water tank, condenser units, potted
        plants, a dish and a lit rooftop sign (喫茶ひまわり). Warm rims on the sun-facing edges."""
        rng = np.random.default_rng(707)
        roofc = np.array([0.2, 0.17, 0.33], np.float32)
        winw = np.array([0.08, 0.08, 0.17], np.float32)

        def box_roof(x0, x1, yt, deck=0.01):
            fill([(x0, yt), (x1, yt), (x1, 1.01), (x0, 1.01)], np.array([0.13, 0.12, 0.22], np.float32))
            fill([(x0 + 0.004, yt - deck), (x1 - 0.004, yt - deck), (x1, yt), (x0, yt)],
                 np.array([0.32, 0.26, 0.44], np.float32))
            # warm bounce on the upper wall under the coping
            fill([(x0, yt + 0.002), (x1, yt + 0.002), (x1, yt + 0.01), (x0, yt + 0.01)],
                 np.array([0.36, 0.2, 0.26], np.float32), a=0.5)
            # windows: a lit row, some curtained
            x = x0 + 0.008
            while x < x1 - 0.02:
                wd = rng.uniform(0.014, 0.024)
                fill([(x, yt + 0.012), (x + wd, yt + 0.012), (x + wd, yt + 0.04), (x, yt + 0.04)], winw)
                if rng.random() < 0.55:
                    c_ = np.array([1.1, 0.66, 0.32] if rng.random() < 0.6 else [1.0, 0.95, 0.82], np.float32)
                    mb = rect(Q(x + 0.0015, yt + 0.013), Q(x + wd - 0.0015, yt + 0.039))
                    emi[mb] = c_ * rng.uniform(0.35, 0.55)
                    onm[mb] = -10.0 if rng.random() < 0.6 else float(rng.uniform(0.8, 4.5))
                ln((x, yt + 0.0405), (x + wd, yt + 0.0405), rimp * 0.5, 0.9)
                x += wd + rng.uniform(0.008, 0.016)
            # parapet coping + railing with rim-lit tops and glints toward the sun
            ln((x0, yt), (x1, yt), wall * 1.4, 2.4)
            ln((x0, yt - 0.0012), (x1, yt - 0.0012), rim, 1.3, e=np.array([0.35, 0.16, 0.06], np.float32))
            ln((x0 - 0.0006, yt), (x0 - 0.0006, 1.01), rim * 0.6, 1.0)
            for x in np.arange(x0 + 0.003, x1, 0.011):
                ln((x, yt), (x, yt - 0.024), steel, 1.3)
                g = math.exp(-abs(SUN_X - x) / 0.18)
                mb = rect(Q(x - 0.001, yt - 0.0255), Q(x + 0.0006, yt - 0.0232))
                emi[mb] += np.array([1.5, 0.95, 0.5], np.float32) * (0.2 + 1.2 * g)
            ln((x0, yt - 0.024), (x1, yt - 0.024), steel, 1.5)
            ln((x0, yt - 0.0252), (x1, yt - 0.0252), rim * 0.9, 0.8)
            ln((x0, yt - 0.012), (x1, yt - 0.012), steel, 1.0)

        # ---------------- roof A: apartment (青空ハイツ)
        ax0, ax1, ayt = 0.345, 0.49, 0.962
        # stair house with a lit door (behind the railing)
        fill([(ax0 + 0.008, 0.912), (ax0 + 0.04, 0.912), (ax0 + 0.04, ayt - 0.008), (ax0 + 0.008, ayt - 0.008)],
             np.array([0.2, 0.17, 0.3], np.float32))
        fill([(ax0 + 0.006, 0.907), (ax0 + 0.042, 0.907), (ax0 + 0.042, 0.912), (ax0 + 0.006, 0.912)], roofc)
        ln((ax0 + 0.006, 0.907), (ax0 + 0.042, 0.907), rim, 1.2)
        ln((ax0 + 0.0418, 0.907), (ax0 + 0.0418, ayt - 0.008), rim * 0.7, 1.0)
        mb = rect(Q(ax0 + 0.024, 0.925), Q(ax0 + 0.034, ayt - 0.008))
        col[mb] = np.array([0.4, 0.3, 0.25], np.float32)
        emi[mb] = np.array([1.1, 0.72, 0.38], np.float32) * 0.6
        mb = rect(Q(ax0 + 0.027, 0.917), Q(ax0 + 0.031, 0.921))
        emi[mb] = np.array([1.3, 1.1, 0.8], np.float32) * 1.4
        # laundry: two poles, a sagging line, shirts / towels / a sheet
        p0x, p1x, ptop = ax0 + 0.06, ax0 + 0.125, 0.918
        for px_ in (p0x, p1x):
            ln((px_, ptop), (px_, ayt - 0.006), steel * 1.3, 1.6)
            ln((px_ - 0.0006, ptop), (px_ - 0.0006, ayt - 0.006), rim * 0.6, 0.6)
        pts = C.catenary((p0x, ptop + 0.002), (p1x, ptop + 0.002), 0.004, n=20)
        for a_, b_ in zip(pts[:-1], pts[1:]):
            ln(tuple(a_), tuple(b_), np.array([0.5, 0.45, 0.55], np.float32), 0.6)
        cloth = [(0.92, 0.9, 0.86), (0.55, 0.72, 0.95), (0.95, 0.62, 0.7), (0.95, 0.92, 0.78), (0.4, 0.5, 0.75),
                 (0.9, 0.85, 0.9)]
        x = p0x + 0.004
        k = 0
        while x < p1x - 0.008:
            wd = rng.uniform(0.006, 0.012)
            ht = rng.uniform(0.012, 0.024)
            yc = ptop + 0.003 + 0.004 * math.sin(math.pi * (x - p0x) / (p1x - p0x))
            c_ = np.array(cloth[k % len(cloth)], np.float32) * 0.85
            if rng.random() < 0.4:     # a shirt: shoulders wider than the body
                fill([(x - 0.002, yc), (x + wd + 0.002, yc), (x + wd, yc + 0.006), (x + wd, yc + ht),
                      (x, yc + ht), (x, yc + 0.006)], c_)
            else:
                fill([(x, yc), (x + wd, yc), (x + wd, yc + ht), (x, yc + ht)], c_)
            ln((x, yc + 0.0008), (x + wd, yc + 0.0008), np.array([1.2, 0.8, 0.6], np.float32) * 0.8, 0.8)
            ln((x + wd - 0.0006, yc), (x + wd - 0.0006, yc + ht), c_ * 0.7, 0.8)
            x += wd + rng.uniform(0.002, 0.006)
            k += 1
        # AC outdoor units on the deck
        for (ux_, uw_) in ((ax0 + 0.135, 0.018), (ax0 + 0.114, 0.016)):
            fill([(ux_, ayt - 0.022), (ux_ + uw_, ayt - 0.022), (ux_ + uw_, ayt - 0.004), (ux_, ayt - 0.004)],
                 np.array([0.34, 0.32, 0.44], np.float32))
            ln((ux_, ayt - 0.022), (ux_ + uw_, ayt - 0.022), rimp, 1.0)
            cv2.circle(col, Q(ux_ + uw_ * 0.4, ayt - 0.013), int(0.0055 * H * ss), tuple(float(c) for c in wall * 0.9),
                       max(1, int(0.8 * ss * H / 1080)))
        box_roof(ax0, ax1, ayt)
        # futon over the railing (patterned) and the name plate on the parapet
        fx0 = ax0 + 0.075
        fill([(fx0, ayt - 0.025), (fx0 + 0.026, ayt - 0.025), (fx0 + 0.026, ayt + 0.008), (fx0, ayt + 0.008)],
             np.array([0.55, 0.62, 0.8], np.float32))
        for k in range(3):
            ln((fx0 + 0.005 + k * 0.008, ayt - 0.023), (fx0 + 0.005 + k * 0.008, ayt + 0.006),
               np.array([0.85, 0.85, 0.95], np.float32), 0.9)
        ln((fx0, ayt - 0.025), (fx0 + 0.026, ayt - 0.025), np.array([1.3, 0.9, 0.75], np.float32), 1.2)
        ln((fx0 + 0.0258, ayt - 0.025), (fx0 + 0.0258, ayt + 0.008), np.array([0.3, 0.3, 0.5], np.float32), 1.0)
        fill([(ax1 - 0.06, ayt + 0.002), (ax1 - 0.008, ayt + 0.002), (ax1 - 0.008, ayt + 0.0105),
              (ax1 - 0.06, ayt + 0.0105)], np.array([0.85, 0.83, 0.78], np.float32))
        stamp_text('青空ハイツ', ax1 - 0.057, ayt + 0.0028, 0.046, 0.0068, c=(0.1, 0.2, 0.5),
                   e=(0.02, 0.05, 0.14), replace_e=True)
        mb = rect(Q(ax1 - 0.06, ayt + 0.002), Q(ax1 - 0.008, ayt + 0.0105))
        emi[mb] += np.array([0.4, 0.38, 0.33], np.float32) * 0.5
        # ---------------- roof B: shop building (lower), sign 喫茶ひまわり
        bx0, bx1, byt = 0.505, 0.69, 0.974
        # water tank on a stand
        tx0, tx1 = bx0 + 0.13, bx0 + 0.162
        for x in (tx0 + 0.003, tx1 - 0.003):
            ln((x, 0.935), (x, byt - 0.006), steel, 2.0)
        ln((tx0 + 0.003, 0.95), (tx1 - 0.003, byt - 0.008), steel, 1.0)
        ln((tx1 - 0.003, 0.95), (tx0 + 0.003, byt - 0.008), steel, 1.0)
        fill([(tx0, 0.9), (tx1, 0.9), (tx1, 0.936), (tx0, 0.936)], np.array([0.12, 0.13, 0.24], np.float32))
        for yy in (0.91, 0.92, 0.93):
            ln((tx0, yy), (tx1, yy), np.array([0.09, 0.1, 0.2]), 1.0)
        ln((tx0, 0.9), (tx1, 0.9), rim, 1.3, e=np.array([0.3, 0.14, 0.05], np.float32))
        ln((tx0 - 0.0005, 0.9), (tx0 - 0.0005, 0.936), rim * 0.9, 1.3)
        # potted plants
        for (px_, r_) in ((bx0 + 0.03, 0.007), (bx0 + 0.043, 0.006), (bx0 + 0.055, 0.0075)):
            fill([(px_ - 0.004, byt - 0.009), (px_ + 0.004, byt - 0.009), (px_ + 0.003, byt - 0.002),
                  (px_ - 0.003, byt - 0.002)], np.array([0.35, 0.18, 0.16], np.float32))
            cv2.circle(col, Q(px_, byt - 0.013), int(r_ * H * ss), (0.06, 0.16, 0.14), -1)
            cv2.circle(al, Q(px_, byt - 0.013), int(r_ * H * ss), 1.0, -1)
            cv2.ellipse(col, Q(px_ - 0.001, byt - 0.016), (int(r_ * 0.8 * H * ss), int(r_ * 0.5 * H * ss)), 0, 180,
                        360, (0.5, 0.42, 0.25), max(1, int(1.0 * ss * H / 1080)))
        # lit rooftop sign on a lattice frame
        sx0_, sx1_, sy0_, sy1_ = bx0 + 0.02, bx0 + 0.09, 0.925, 0.948
        for x in (sx0_ + 0.01, sx1_ - 0.01):
            ln((x, sy1_), (x, byt - 0.004), steel, 1.8)
        ln((sx0_ + 0.01, sy1_ + 0.004), (sx1_ - 0.01, byt - 0.006), steel, 0.9)
        fill([(sx0_ - 0.002, sy0_ - 0.003), (sx1_ + 0.002, sy0_ - 0.003), (sx1_ + 0.002, sy1_ + 0.003),
              (sx0_ - 0.002, sy1_ + 0.003)], steel)
        mb = rect(Q(sx0_, sy0_), Q(sx1_, sy1_))
        col[mb] = np.array([0.5, 0.42, 0.2], np.float32)
        emi[mb] = np.array([1.05, 0.8, 0.35], np.float32) * 0.38
        onm[mb] = -10.0
        stamp_text('喫茶ひまわり', sx0_ + 0.005, sy0_ + 0.005, sx1_ - sx0_ - 0.01, sy1_ - sy0_ - 0.01,
                   c=(0.35, 0.08, 0.05), e=(0.05, 0.01, 0.0), replace_e=True)
        ln((sx0_ - 0.002, sy0_ - 0.003), (sx1_ + 0.002, sy0_ - 0.003), rim, 1.2)
        # condenser units
        for ux_ in (bx1 - 0.05, bx1 - 0.032):
            fill([(ux_, byt - 0.02), (ux_ + 0.016, byt - 0.02), (ux_ + 0.016, byt - 0.003), (ux_, byt - 0.003)],
                 np.array([0.34, 0.32, 0.44], np.float32))
            ln((ux_, byt - 0.02), (ux_ + 0.016, byt - 0.02), rimp, 1.0)
            cv2.circle(col, Q(ux_ + 0.007, byt - 0.0115), int(0.005 * H * ss), tuple(float(c) for c in wall * 0.9),
                       max(1, int(0.8 * ss * H / 1080)))
        box_roof(bx0, bx1, byt, deck=0.008)
        # a small dish on the parapet corner
        cv2.ellipse(col, Q(bx1 - 0.012, byt - 0.03), (int(0.005 * W * ss), int(0.009 * H * ss)), -25, 0, 360,
                    tuple(float(c) for c in np.array([0.34, 0.32, 0.44], np.float32)), -1)
        cv2.ellipse(al, Q(bx1 - 0.012, byt - 0.03), (int(0.005 * W * ss), int(0.009 * H * ss)), -25, 0, 360, 1.0, -1)
        cv2.ellipse(col, Q(bx1 - 0.012, byt - 0.03), (int(0.005 * W * ss), int(0.009 * H * ss)), -25, 110, 240,
                    tuple(float(c) for c in rim), max(1, int(1.2 * ss * H / 1080)))
        ln((bx1 - 0.012, byt - 0.024), (bx1 - 0.012, byt), steel, 1.4)

    # ------------------------------------------------------------------ sky life: jet + contrail, birds
    def _jet(self, img, em, t, yaw):
        W, H = self.W, self.H
        hx0, hy0 = 0.47 * W, 0.155 * H
        vx, vy = -0.011 * W, -0.0026 * W
        hx, hy_ = hx0 + vx * t - yaw, hy0 + vy * t
        dx, dy = -vx, -vy
        n = math.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        Lc = 0.2 * W
        ss = 3
        x0 = int(max(min(hx, hx + dx * Lc) - 0.02 * W, 0)); x1 = int(min(max(hx, hx + dx * Lc) + 0.02 * W, W))
        y0 = int(max(min(hy_, hy_ + dy * Lc) - 0.02 * W, 0)); y1 = int(min(max(hy_, hy_ + dy * Lc) + 0.02 * W, H))
        if x1 <= x0 or y1 <= y0:
            return
        h, w = y1 - y0, x1 - x0
        acc = np.zeros((h * ss, w * ss), np.float32)
        nseg = 48
        for i in range(nseg - 1, -1, -1):
            s0, s1 = i / nseg, (i + 1) / nseg
            age = s0
            wd = (0.0007 + 0.0022 * age) * W
            a = (1 - age) ** 1.6 * min(1.0, s0 * 30 + 0.1)
            if s0 < 0.012:
                a *= 0.3
            p0 = ((hx + dx * Lc * s0 - x0) * ss, (hy_ + dy * Lc * s0 - y0) * ss)
            p1 = ((hx + dx * Lc * s1 - x0) * ss, (hy_ + dy * Lc * s1 - y0) * ss)
            cv2.line(acc, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), float(a), max(1, int(wd * ss)))
        acc = cv2.resize(acc, (w, h), interpolation=cv2.INTER_AREA)
        acc = C.blur(acc, 0.0008 * W)
        col = np.array([1.1, 0.72, 0.62], np.float32)
        sl = (slice(y0, y1), slice(x0, x1))
        img[sl] = img[sl] * (1 - acc[..., None] * 0.75) + col * acc[..., None] * 0.75
        C.splat(em, [hx], [hy_], 0.0012 * W, (1.0, 0.85, 0.7), 0.9)
        b = math.exp(-(((t * 1.1) % 1.0 - 0.3) / 0.06) ** 2)
        C.splat(em, [hx + 0.002 * W], [hy_ + 0.0006 * W], 0.0009 * W, (1.0, 0.15, 0.1), 1.6 * b)

    def _birds(self, img, t, yaw):
        W, H = self.W, self.H
        x0, y0 = 0, int(0.12 * H)
        Wc, Hc = int(0.5 * W), int(0.34 * H)
        a = S.birds(Wc, Hc, t, seed=5, n=6, center=(0.12 * W / Wc, (0.33 * H - y0) / Hc),
                    velocity=(0.022 * W / Wc, -0.003 * H / Hc), size=0.012 * W / Wc, spread=0.04 * W / Wc,
                    flap=2.6, heading=1.0)
        M = np.float32([[1, 0, -yaw], [0, 1, 0]])
        a = cv2.warpAffine(a, M, (Wc, Hc), flags=cv2.INTER_LINEAR)
        col = np.array([0.16, 0.12, 0.3], np.float32)
        sl = (slice(y0, y0 + Hc), slice(x0, x0 + Wc))
        img[sl] = img[sl] * (1 - a[..., None]) + col * a[..., None]
        return img

    # ------------------------------------------------------------------ aircraft warning lights
    def _warning_lights(self, t, truck, yaw, fg_shift, mast_shift):
        """Screen positions + blink intensities of the red aircraft-warning lights (slow, out of sync).
        Lights hidden behind the near planes are dropped."""
        W, H = self.W, self.H
        out = []
        for k, (x, y, z, ph) in enumerate(self.city.lights):
            sx, sy = self._proj(x, y, z, truck, yaw)
            if not (-10 < sx < W + 10 and 0 <= sy < H):
                continue
            # occluded by the near planes?
            yy = int(sy) - self.fg_top
            if 0 <= yy < self.fg.shape[0]:
                xp = int(sx - fg_shift)
                xq = int(sx - mast_shift)
                if 0 <= xp < self.fg.shape[1] and self.fg[yy, xp, 3] > 0.5:
                    continue
                if 0 <= xq < self.mast.shape[1] and self.mast[yy, xq, 3] > 0.5:
                    continue
            per = 1.7 + 1.1 * ((k * 0.618034) % 1.0)
            u = (t / per + ph + 0.37 * k) % 1.0
            b = float(C.smoothstep(0.0, 0.08, u) * C.smoothstep(0.5, 0.4, u))
            fog = math.exp(-max(z - 1500, 0) / 12000.0)
            out.append((sx, sy, b, fog, 1.0 if z < 3000 else 0.8))
        # the lattice mast top (near plane) and a second lamp half-way down it
        fsx = self.fg_light[0] + mast_shift
        fsy = self.fg_light[1] + self.fg_top
        u = (t / 2.2 + 0.55) % 1.0
        b = float(C.smoothstep(0.0, 0.08, u) * C.smoothstep(0.5, 0.4, u))
        out.append((fsx, fsy, b, 1.0, 1.6))
        u = (t / 2.9 + 0.1) % 1.0
        b = float(C.smoothstep(0.0, 0.08, u) * C.smoothstep(0.5, 0.4, u))
        out.append((fsx + 0.0125 * W, 0.42 * H, b, 1.0, 1.25))
        return out

    def _draw_beacons(self, img, beacons):
        """crisp red cores (3-4 px, #ff2a1a) painted OVER the graded image + soft red bloom halos (the halos are
        accumulated at quarter resolution and added once: cheap even with ~100 lamps)"""
        W, H = self.W, self.H
        red = np.array([1.0, 0.165, 0.10], np.float32)
        sc = H / 1080.0
        q = 4
        hw, hh = W // q + 1, H // q + 1
        acc = np.zeros((hh, hw), np.float32)
        for (sx, sy, b, fog, size) in beacons:
            b = 0.3 + 0.7 * b      # slow flash over a steady glow: the lamp never vanishes
            Rh = 0.0036 * W * size / q      # round 9: tight glow (~40 % of the old halo), red point reads
            cx, cy = sx / q, sy / q
            x0, x1 = int(max(cx - Rh * 4.5, 0)), int(min(cx + Rh * 4.5 + 1, hw))
            y0, y1 = int(max(cy - Rh * 4.5, 0)), int(min(cy + Rh * 4.5 + 1, hh))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            acc[y0:y1, x0:x1] += (np.exp(-(d / Rh) ** 2) * 0.3 + np.exp(-(d / max(Rh * 0.4, 0.6)) ** 2) * 0.3 +
                                  np.exp(-(d / (Rh * 2.2)) ** 2) * 0.04) * b * fog
        halo = cv2.resize(acc, (W + q, H + q), interpolation=cv2.INTER_LINEAR)[q // 2:q // 2 + H, q // 2:q // 2 + W]
        img += red * halo[..., None]
        for (sx, sy, b, fog, size) in beacons:
            b = 0.3 + 0.7 * b
            R = 2.2 * sc * size
            x0, x1 = int(max(sx - R - 2, 0)), int(min(sx + R + 3, W))
            y0, y1 = int(max(sy - R - 2, 0)), int(min(sy + R + 3, H))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
            core = np.clip(R + 0.5 - d, 0, 1) * b * (0.55 + 0.45 * fog)
            sub = img[y0:y1, x0:x1]
            sub[:] = sub * (1 - core[..., None]) + red * 1.05 * core[..., None]
            # tiny hot centre so it reads as a lamp
            hc = np.clip(R * 0.45 + 0.5 - d, 0, 1) * b * 0.5
            sub[:] = sub * (1 - hc[..., None]) + np.array([1.0, 0.55, 0.45], np.float32) * hc[..., None]

    # ------------------------------------------------------------------ camera
    def cam(self, t):
        # slow lateral truck concentrated in the part of the shot the edit uses (~1.8-5.0 s): eased in,
        # settling on the billboard framing on the last beat
        s = min(max((t - 0.9) / 4.6, 0.0), 1.0)
        u = 0.15 * (t / DURATION) + 0.85 * s * s * s * (s * (s * 6 - 15) + 10)
        truck = TRUCK0 + (TRUCK - TRUCK0) * u
        yaw = (u - 0.5) * 2 * YAW * self.W
        return truck, yaw

    def _sun_rim(self, img, occ, occ_sky, lx, ly):
        """Warm 2-3 px rim on the tower edges that face the afterglow + halation spilling over them."""
        W, H = self.W, self.H
        x0, x1 = int(max(lx - 0.34 * W, 0)), int(min(lx + 0.34 * W, W))
        y0, y1 = int(0.22 * H), int(min(self.hy + 0.1 * H, H))
        city = np.clip(occ[y0:y1, x0:x1] - occ_sky[y0:y1, x0:x1], 0, 1)
        xs = np.arange(x0, x1, dtype=np.float32)[None, :]
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        dx, dy = lx - xs, ly - ys
        d = np.sqrt(dx * dx + dy * dy) + 1e-3
        off = 2.6 * H / 1080
        mx = (xs - x0 + dx / d * off).astype(np.float32)
        my = (ys - y0 + dy / d * off).astype(np.float32)
        mx, my = np.broadcast_to(mx, city.shape).copy(), np.broadcast_to(my, city.shape).copy()
        nb = cv2.remap(city, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rim = np.clip(city - nb, 0, 1)
        w = np.exp(-d / (0.16 * W)) * C.smoothstep(self.hy + 0.06 * H, self.hy - 0.02 * H, ys)
        rw = rim * w
        glow = cv2.GaussianBlur(rw, (0, 0), 0.004 * W)
        add = rw[..., None] * np.array([1.6, 0.9, 0.5], np.float32) * 1.3 +             glow[..., None] * np.array([1.0, 0.55, 0.3], np.float32) * 1.4
        img[y0:y1, x0:x1] += add

    def _sun_compress(self, img, lx, ly, keep=None):
        W, H = self.W, self.H
        if getattr(self, '_scw', None) is None:
            # radial weight precomputed on a padded canvas, slid with the pan (it is very soft)
            gp = self.glare_pad
            xs = np.arange(W + 2 * gp, dtype=np.float32) - (SUN_X * W + gp)
            ys = np.arange(H, dtype=np.float32) - (self.hy + 0.004 * H)
            r = np.sqrt(xs[None, :] ** 2 + (ys[:, None] * 1.3) ** 2)
            self._scw = np.ascontiguousarray(np.exp(-(r / (0.42 * W)) ** 2).astype(np.float32))
        gx0 = int(round(self.glare_pad - (SUN_X * W - lx)))
        wm = self._scw[:, gx0:gx0 + W]
        if keep is not None:
            wm = wm * (1.0 - 0.85 * np.clip(keep, 0, 1))     # the painted cloud keeps its own values
        # per channel (not hue-preserving): the hottest core converges to a pale warm white (~0.93 after
        # the shoulder) while the orange halation below ~0.75 is untouched
        k0, top = 0.75, 0.42
        d = np.maximum(img - k0, 0.0)
        f = img - d + d / (1.0 + d / top)
        wm3 = wm[..., None]
        return img * (1 - wm3) + f * wm3

    def _glare(self, W, H, lx, ly):
        """Sun glare at the horizon: hot core, wide warm bloom, anamorphic streaks, ghost flares."""
        out = F.anime_flare(W, H, lx, ly, intensity=1.0, tint=(1.0, 0.6, 0.32), rays=6, ray_len=0.07,
                            starburst=0.8, ghosts=3.5, halo=1.6, streak=1.2, glow=1.0)
        # long thin anamorphic line through the whole frame + a second, softer band
        ys = np.arange(H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None, :]
        dy = ys - ly
        dx = np.abs(xs - lx)
        s1 = np.exp(-(dy / (0.0018 * H)) ** 2) * (np.exp(-dx / (0.45 * W)) * 0.5 + np.exp(-dx / (0.08 * W)) * 0.6)
        s2 = np.exp(-(dy / (0.012 * H)) ** 2) * np.exp(-dx / (0.3 * W)) * 0.12
        # wide halation core: the afterglow wraps the silhouettes of the towers nearest the sun
        r = np.sqrt((xs - lx) ** 2 + ((ys - ly) * 1.25) ** 2)
        hal = np.exp(-(r / (0.065 * W)) ** 2) * 0.55 + np.exp(-r / (0.15 * W)) * 0.22
        out += hal[..., None] * np.array([1.0, 0.6, 0.34], np.float32)
        out += (s1[..., None] * np.array([0.75, 0.82, 1.0], np.float32) +
                s2[..., None] * np.array([1.0, 0.65, 0.45], np.float32))
        return out.astype(np.float32)

    def frame(self, t):
        W, H = self.W, self.H
        truck, yaw = self.cam(t)
        M = np.float32([[1, 0, -self.mx_sky - yaw], [0, 1, 0]])
        img = cv2.warpAffine(self.sky, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        lx, ly = SUN_X * W - yaw, self.hy + 0.004 * H
        img = np.ascontiguousarray(img)
        cls = self.clouds.render_layers(W, H, t, cam_x=yaw + 0.6 * truck * W / 1920.0, cam_y=0.0)
        occ = np.zeros((H, W), np.float32)
        for L in cls:
            img = F.over_rgba(img, L)
            occ = np.maximum(occ, L[..., 3])
        em = np.zeros((H, W, 3), np.float32)
        self._jet(img, em, t, yaw)
        vx_, vy_ = 0.335 * W - yaw, 0.165 * H
        tw = 0.95 + 0.05 * math.sin(t * 1.1 + 1.0)   # slow, gentle (no twinkle flicker)
        C.splat(em, [vx_], [vy_], 0.0011 * W, (1.0, 0.95, 0.9), 1.4 * tw)
        C.splat(em, [vx_], [vy_], 0.004 * W, (1.0, 0.8, 0.8), 0.08 * tw)
        img = np.ascontiguousarray(self._birds(img, t, yaw), dtype=np.float32)
        occ_sky = occ.copy()
        self.clouds_a = cls[0][..., 3] * 0 + np.maximum(cls[0][..., 3], cls[1][..., 3])
        for bd in self.bands:
            shift = -self.f * truck / bd['zrep'] - yaw - bd['mx']
            K.composite_band(img, em, occ, bd['pm'], bd['E'], bd['on'], bd['top'], float(shift), float(t))
            if 'train' in bd:
                # outbound train on the far track (moving right), then the inbound one on the near track
                self._train(img, em, t, truck, yaw, v=-38.0, head0=-330.0, ncar=6, side=-1.9, glint=False)
                self._train(img, em, t, truck, yaw)

        self._sun_rim(img, occ, occ_sky, lx, ly)
        shift2 = -self.f * truck / self.MAST_Z - yaw - self.fg_mx + self.MAST_OFF * W / 1920.0
        K.composite_band(img, em, occ, self.mast, self.mast_E, self.mast_on, self.fg_top, float(shift2), float(t))
        shift = -self.f * truck / self.FG_Z - yaw - self.fg_mx
        K.composite_band(img, em, occ, self.fg, self.fg_E, self.fg_on, self.fg_top, float(shift), float(t))
        beacons = self._warning_lights(t, truck, yaw, shift, shift2)
        # lights glow: bloom the emission layer on its own (windows, train, signs, beacons)
        q = cv2.resize(em, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(q, (0, 0), 0.0025 * W) * 0.9 + cv2.GaussianBlur(q, (0, 0), 0.009 * W) * 0.7
        img += em
        img += cv2.resize(g * 0.3, (W, H), interpolation=cv2.INTER_LINEAR)
        # crepuscular rays fanning up from the horizon glow, cut by towers and cloud edges
        sh_ = SK.shafts(W, H, lx, ly + 0.01 * H, occ, strength=0.8, length=0.95, radius=0.12,
                        tint=(1.0, 0.6, 0.38), t=t, streaks=0.55, n_beams=9, seed=4)
        # rays pass behind the storm cloud: keep its painted pinks saturated
        img += sh_ * (1 - 0.75 * self.clouds_a)[..., None]
        # glare: precomputed on a padded canvas, slid with the pan (integer px; it is soft)
        gx0 = int(round(self.glare_pad - (SUN_X * W - lx)))
        img += self.glare[:, gx0:gx0 + W] * (0.45 * (1 - 0.6 * self.clouds_a))[..., None]
        # bloom: the painted cloud interior is held out of the bloom source (no airbrushed glow outline);
        # its HDR gold lining still blooms
        hero_vis = cls[1][..., 3] * (1 - np.clip(occ - occ_sky, 0, 1))
        bsrc = CB4.bloom_src(img, hero_vis)
        img = img + (SK.fast_bloom(bsrc, threshold=0.8, knee=0.3, strength=0.4, halation=0.25) - bsrc)
        # highlight roll-off: the painted cloud keeps its hot pink-gold (almost no desaturation toward white)
        # round 9: the sun core must not clip to a flat white disk - hue-preserving HDR compression of the
        # afterglow region (max channel rolls off toward ~1.15 before the shoulder -> peak ~0.94), the wide
        # halation keeps carrying the brightness around it
        img = self._sun_compress(img, lx, ly, hero_vis)
        hv = cv2.GaussianBlur(hero_vis, (0, 0), 1.5)[..., None]
        img = F.shoulder(img, 0.82, desat=0.15) * (1 - hv) + F.shoulder(img, 0.86, desat=0.02) * hv
        self._draw_beacons(img, beacons)
        # paint / paper surface only on the painted city + near planes; the sky gradient stays clean
        cm = np.clip(occ - occ_sky, 0, 1)
        img = PAPER.apply(img, 1.0 + (self.paper - 1.0) * (0.12 + 0.88 * cm))
        return F.finish_fast(img, t, sat=1.06, grain_amt=0.0025, vig=0.22, ca=0.0)
