"""s09_classroom - empty high-school classroom, late afternoon.

Rows of desks and chairs, big windows with white curtains billowing, warm sun streaming in with light
shafts and floating dust, long window-frame shadows across floor and desks, blackboard, blue sky with
cumulus and green trees outside. Camera: slow push-in toward the windows.

Rendered with a small numba ray tracer (scenes/s09_classroom_rt.py): true perspective/parallax for the
push-in, analytic soft window shadows, glossy reflections; painted sky/tree plates outside
(s09_classroom_outside.py); curtains are cloth meshes rasterised on top (s09_classroom_cloth.py);
dust motes / compositing kernels in s09_classroom_fx.py.
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import core as C, fx as F  # noqa: E402
import s09_classroom_rt as R  # noqa: E402
import s09_classroom_outside as O  # noqa: E402
import s09_classroom_cloth as K  # noqa: E402
import s09_classroom_fx as X  # noqa: E402

DURATION = 5.0


def _norm(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


class _Boxes:
    def __init__(self):
        self.rows = []
        self.groups = []
        self._g0 = 0

    def begin(self):
        self._g0 = len(self.rows)

    def box(self, c, h, mat, var=0.0, yaw=0.0, noshadow=False, shape=0):
        self.rows.append([c[0], c[1], c[2], h[0], h[1], h[2], math.cos(yaw), math.sin(yaw), mat, var,
                          1.0 if noshadow else 0.0, float(shape)])

    def part(self, origin, yaw, off, h, mat, var=0.0, noshadow=False, shape=0):
        """box at local offset `off` of an object at `origin` rotated by yaw."""
        c, s = math.cos(yaw), math.sin(yaw)
        wx = origin[0] + c * off[0] + s * off[2]
        wz = origin[2] - s * off[0] + c * off[2]
        self.box((wx, origin[1] + off[1], wz), h, mat, var, yaw, noshadow, shape)

    def end(self):
        i0, i1 = self._g0, len(self.rows)
        if i1 <= i0:
            return
        lo = np.array([1e9] * 3)
        hi = -lo
        for r in self.rows[i0:i1]:
            c, h, cs, sn = np.array(r[:3]), np.array(r[3:6]), abs(r[6]), abs(r[7])
            ex = np.array([cs * h[0] + sn * h[2], h[1], sn * h[0] + cs * h[2]])
            lo = np.minimum(lo, c - ex)
            hi = np.maximum(hi, c + ex)
        self.groups.append(list(lo - 1e-3) + list(hi + 1e-3) + [i0, i1])

    def arrays(self):
        return np.array(self.rows, np.float64), np.array(self.groups, np.float64)


JACKET_AT = {(3, 3)}


def build_room(seed=5):
    rng = np.random.default_rng(seed)
    Bx = _Boxes()
    LX, LZ, HC = R.LX, R.LZ, R.HC
    # ---- window wall: pillars, sill, valance
    pillars = [2.55, 4.6, 6.65]
    Bx.begin()
    for pxc in pillars:
        Bx.box((pxc, HC / 2, 0.09), (0.16, HC / 2, 0.09), R.M_PILLAR, noshadow=True)
    Bx.box(((R.AX0) / 2, HC / 2, 0.09), (R.AX0 / 2, HC / 2, 0.09), R.M_PILLAR, noshadow=True)
    Bx.box(((LX + R.AX1) / 2, HC / 2, 0.09), ((LX - R.AX1) / 2, HC / 2, 0.09), R.M_PILLAR, noshadow=True)
    Bx.box((LX / 2, R.AY0 - 0.03, 0.11), (LX / 2, 0.03, 0.11), R.M_SILL, noshadow=True)
    Bx.box((LX / 2, R.AY1 + 0.14, 0.07), (LX / 2, 0.14, 0.07), R.M_VALANCE, noshadow=True)
    Bx.end()
    bays = [(R.AX0, pillars[0] - 0.16), (pillars[0] + 0.16, pillars[1] - 0.16),
            (pillars[1] + 0.16, pillars[2] - 0.16), (pillars[2] + 0.16, R.AX1)]
    # ---- aluminium sash frames (also the analytic shadow bars VB / HB)
    VB, HB = [], []
    fw = 0.028
    Bx.begin()
    for (x0, x1) in bays:
        xm = (x0 + x1) / 2
        for xc in (x0 + fw, x1 - fw):
            Bx.box((xc, (R.AY0 + R.AY1) / 2, -0.04), (fw, (R.AY1 - R.AY0) / 2, 0.035), R.M_FRAME, noshadow=True)
            VB.append((xc, fw))
        for xc, zc in ((xm - 0.025, -0.02), (xm + 0.025, -0.065)):
            Bx.box((xc, (R.AY0 + R.AY1) / 2, zc), (0.022, (R.AY1 - R.AY0) / 2, 0.022), R.M_FRAME, noshadow=True)
        VB.append((xm, 0.05))
        for xc in ((x0 + xm) / 2, (xm + x1) / 2):     # transom windows split again
            Bx.box((xc, (2.3 + R.AY1) / 2, -0.04), (0.018, (R.AY1 - 2.3) / 2, 0.03), R.M_FRAME, noshadow=True)
    for yc, hh in ((R.AY0 + 0.03, 0.03), (R.AY1 - 0.03, 0.03), (2.3, 0.032), (1.02, 0.018)):
        Bx.box((LX / 2, yc, -0.04), ((R.AX1 - R.AX0) / 2, hh, 0.035), R.M_FRAME, noshadow=True)
        HB.append((yc, hh))
    Bx.end()
    # ---- balcony outside
    Bx.begin()
    Bx.box((LX / 2, 0.36, -1.62), (LX / 2 + 3, 0.42, 0.07), R.M_PARAPET, noshadow=True)
    Bx.box((LX / 2, -0.06, -0.8), (LX / 2 + 3, 0.06, 0.8), R.M_BALC, noshadow=True)
    Bx.box((LX / 2, 1.08, -1.62), (LX / 2 + 3, 0.02, 0.025), R.M_RAIL, noshadow=True)
    Bx.box((LX / 2, 0.93, -1.62), (LX / 2 + 3, 0.012, 0.015), R.M_RAIL, noshadow=True)
    for xp in np.arange(-2.5, LX + 3, 1.5):
        Bx.box((xp, 0.93, -1.62), (0.018, 0.15, 0.018), R.M_RAIL, noshadow=True)
    Bx.end()
    # ---- front wall: blackboard, tray, clock, notices, speaker, platform, lectern
    Bx.begin()
    Bx.box((LX - 0.012, 1.55, 3.5), (0.012, 0.62, 2.35), R.M_BOARDFRAME)
    Bx.box((LX - 0.02, 1.55, 3.5), (0.012, 0.58, 2.3), R.M_BOARD)
    Bx.box((LX - 0.07, 0.92, 3.5), (0.06, 0.015, 2.3), R.M_TRAY)
    # erasers + chalk sticks on the tray
    Bx.box((LX - 0.075, 0.935 + 0.022, 2.05), (0.026, 0.022, 0.065), R.M_ERASER, yaw=0.08)
    Bx.box((LX - 0.07, 0.935 + 0.022, 4.95), (0.026, 0.022, 0.065), R.M_ERASER, yaw=-0.12)
    for i, (zc, cv_) in enumerate(((2.35, 0.1), (2.44, 0.6), (2.5, 0.1), (2.62, 0.85), (4.6, 0.3))):
        Bx.box((LX - 0.06 - 0.012 * (i % 2), 0.935 + 0.006, zc), (0.006, 0.006, 0.035), R.M_CHALK,
               var=cv_, yaw=0.2 * (i - 2), noshadow=True)
    Bx.box((LX - 0.03, 2.5, 3.5), (0.03, 0.15, 0.15), R.M_CLOCK)
    Bx.box((LX - 0.08, 2.72, 6.3), (0.08, 0.14, 0.2), R.M_SPEAKER)
    for i in range(7):
        rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02), rng.random()     # keep the rng stream
    for i in range(5):
        z = 6.05 + (i % 3) * 0.3
        y = 1.9 - (i // 3) * 0.42
        Bx.box((LX - 0.004, y, z), (0.003, 0.18, 0.12), R.M_PAPER, var=rng.random())
    Bx.box((LX - 0.55, 0.1, 3.5), (0.55, 0.1, 2.6), R.M_PLATFORM)
    # teacher's lectern (kyoutaku): panelled wooden body on a dark kick plinth, an overhanging top board,
    # the attendance binder, a chalk box and a stack of printouts on top
    lo_ = (LX - 0.75, 0.0, 3.4)
    Bx.box((lo_[0], 0.2 + 0.035, lo_[2]), (0.235, 0.035, 0.435), R.M_LECTERN, var=0.05)
    Bx.box((lo_[0], 0.27 + 0.47, lo_[2]), (0.25, 0.47, 0.45), R.M_LECTERN, var=0.5)
    Bx.box((lo_[0] - 0.01, 1.21 + 0.016, lo_[2]), (0.285, 0.016, 0.49), R.M_LECTERN, var=0.95)
    Bx.box((lo_[0] - 0.05, 1.242 + 0.012, lo_[2] - 0.22), (0.11, 0.012, 0.155), R.M_BOOK, var=0.0, yaw=0.12)
    Bx.box((lo_[0] - 0.02, 1.242 + 0.005, lo_[2] + 0.16), (0.105, 0.005, 0.15), R.M_PAPER, var=0.02, yaw=-0.2)
    Bx.box((lo_[0] - 0.02, 1.252 + 0.004, lo_[2] + 0.17), (0.105, 0.004, 0.15), R.M_NOTE, var=0.1, yaw=-0.1)
    Bx.box((lo_[0] + 0.12, 1.242 + 0.025, lo_[2] + 0.02), (0.045, 0.025, 0.08), R.M_PCASE, var=0.4, yaw=0.3)
    Bx.end()
    # teacher's desk near the window
    Bx.begin()
    tdo = (LX - 1.35, 0.0, 0.95)
    Bx.part(tdo, 0.0, (0, 0.72, 0), (0.4, 0.02, 0.62), R.M_DESKTOP, 0.3)
    Bx.part(tdo, 0.0, (0.05, 0.4, 0.35), (0.32, 0.3, 0.25), R.M_DESKMETAL)
    for sx in (-1, 1):
        for sz in (-1, 1):
            Bx.part(tdo, 0.0, (sx * 0.36, 0.35, sz * 0.58), (0.02, 0.35, 0.02), R.M_DESKMETAL)
    Bx.part(tdo, 0.0, (0.1, 0.76, -0.3), (0.12, 0.02, 0.16), R.M_BOOK, 0.1)
    Bx.part(tdo, 0.0, (0.05, 0.795, -0.28), (0.11, 0.015, 0.15), R.M_BOOK, 0.6)
    Bx.end()
    # ceiling beams (aligned with the window pillars) and fluorescent lights between them
    Bx.begin()
    for xb in pillars:
        Bx.box((xb, HC - 0.14, LZ / 2), (0.16, 0.14, LZ / 2), R.M_PILLAR, noshadow=True)
    Bx.end()
    for xl in (1.55, 3.58, 5.62, 7.66):
        Bx.begin()
        for zl in (1.6, 3.6, 5.6):
            Bx.box((xl, HC - 0.09, zl), (0.62, 0.035, 0.09), R.M_LIGHT, noshadow=True)
        Bx.end()
    # ---- student desks + chairs (students face +x)
    rows = [2.2, 3.3, 4.4, 5.5, 6.6]
    cols = [1.4, 2.45, 3.5, 4.55, 5.6]
    skip = {(0, 4), (0, 3), (0, 2), (1, 4)}
    for ri, xd in enumerate(rows):
        for ci, zd in enumerate(cols):
            if (ri, ci) in skip:
                continue
            Bx.begin()
            yaw = float(np.clip(rng.normal(0, 0.065), -0.11, 0.11))           # +-3..6 deg
            o = (xd + rng.normal(0, 0.05), 0.0, zd + rng.normal(0, 0.06))
            v = rng.random()
            Bx.part(o, yaw, (0, 0.705, 0), (0.225, 0.016, 0.315), R.M_DESKTOP, v)
            Bx.part(o, yaw, (0.02, 0.64, 0), (0.19, 0.045, 0.28), R.M_DESKMETAL)
            for sx in (-1, 1):
                for sz in (-1, 1):
                    Bx.part(o, yaw, (sx * 0.2, 0.345, sz * 0.285), (0.012, 0.345, 0.012), R.M_DESKMETAL)
                Bx.part(o, yaw, (sx * 0.2, 0.12, 0), (0.01, 0.01, 0.285), R.M_DESKMETAL)
            if rng.random() < 0.35:        # school bag on the hook
                Bx.part(o, yaw, (0.0, 0.5, 0.34), (0.16, 0.13, 0.05), R.M_BAG, rng.random())
            if rng.random() < 0.2:          # a book left on the desk
                Bx.part(o, yaw + rng.normal(0, 0.3), (rng.uniform(-0.05, 0.05), 0.73, rng.uniform(-0.1, 0.1)),
                        (0.1, 0.008, 0.14), R.M_BOOK, rng.random())
            # lived-in clutter (own rng stream so the layout above stays put)
            cr = np.random.default_rng(1000 + ri * 10 + ci)
            sb = 1 if cr.random() < 0.5 else -1
            if cr.random() < 0.45:          # bag hanging from the side hook, strap up to the hook
                bv = cr.random()
                Bx.part(o, yaw, (cr.uniform(-0.04, 0.04), 0.47, sb * 0.345), (0.15, 0.11, 0.045), R.M_BAG, bv)
                Bx.part(o, yaw, (0.0, 0.62, sb * 0.335), (0.014, 0.055, 0.006), R.M_BAG, bv, noshadow=True)
            if cr.random() < 0.4:           # notebooks, sometimes a small stack
                nn = 1 + int(cr.random() < 0.35)
                ny_ = 0.721
                for k_ in range(nn):
                    Bx.part(o, yaw + cr.normal(0, 0.25), (cr.uniform(-0.08, 0.06), ny_ + 0.004,
                                                          cr.uniform(-0.14, 0.14)),
                            (0.09, 0.004, 0.125), R.M_NOTE, cr.random())
                    ny_ += 0.008
            if cr.random() < 0.25:          # pencil case
                Bx.part(o, yaw + cr.normal(0, 0.4), (cr.uniform(-0.12, 0.1), 0.721 + 0.017,
                                                     cr.uniform(-0.2, 0.2)), (0.026, 0.017, 0.085),
                        R.M_PCASE, cr.random(), shape=2)
            if cr.random() < 0.62:          # textbook / printouts poking out of the book-box
                zz = cr.uniform(-0.12, 0.12)
                Bx.part(o, yaw + cr.normal(0, 0.1), (-0.19, 0.612, zz), (0.045, 0.01, 0.105), R.M_NOTE,
                        cr.random(), noshadow=True)
                if cr.random() < 0.7:
                    Bx.part(o, yaw + cr.normal(0, 0.2), (-0.2, 0.627, zz + cr.uniform(-0.08, 0.08)),
                            (0.05, 0.0015, 0.1), R.M_PAPER, 0.05, noshadow=True)
            if cr.random() < 0.12:          # a bag dumped on the floor beside the desk
                Bx.part(o, yaw + cr.normal(0, 0.5), (cr.uniform(-0.3, -0.1), 0.1, -sb * 0.42),
                        (0.17, 0.1, 0.06), R.M_BAG, cr.random())
            # chair: behind the desk (lower x), sometimes pushed in, sometimes pulled out / turned
            pull = rng.choice([0.0, 0.08, 0.2, 0.32], p=[0.35, 0.3, 0.2, 0.15])
            cy = rng.normal(0, 0.16) + (rng.choice([0.55, -0.45, 0.3]) if rng.random() < 0.25 else 0)
            if (ri, ci) in ((2, 4), (3, 1)):        # a couple of chairs left turned sideways, pulled out
                cy, pull = (1.25 if ri == 2 else -1.1), 0.3
            co = (o[0] - 0.36 - pull, 0.0, o[2] + rng.normal(0, 0.05))
            cyaw = yaw + cy
            cv = rng.random()
            Bx.part(co, cyaw, (0, 0.42, 0), (0.19, 0.012, 0.19), R.M_CHAIRWOOD, cv)
            # bent plywood back (SDF panel: curved in plan, arched top edge, rounded corners)
            Bx.part(co, cyaw, (-0.2, 0.72, 0), (0.012 + R.BACK_BEND * 0.5 + 0.002, 0.1, 0.18), R.M_CHAIRWOOD, cv,
                    shape=1)
            for sz in (-1, 1):
                Bx.part(co, cyaw, (-0.205, 0.6, sz * 0.17), (0.011, 0.19, 0.011), R.M_CHAIRMETAL)
                for sx in (-1, 1):
                    Bx.part(co, cyaw, (sx * 0.17, 0.2, sz * 0.17), (0.011, 0.2, 0.011), R.M_CHAIRMETAL)
            if (ri, ci) in JACKET_AT:
                # blazer thrown over the chair back: shoulders on the rail, body hanging behind,
                # a sleeve dangling, the lining side toward the seat
                jv = 0.3
                Bx.part(co, cyaw, (-0.2, 0.835, 0.0), (0.03, 0.014, 0.2), R.M_JACKET, jv)
                Bx.part(co, cyaw, (-0.232, 0.62, 0.01), (0.009, 0.22, 0.205), R.M_JACKET, jv)
                Bx.part(co, cyaw, (-0.17, 0.72, -0.01), (0.008, 0.11, 0.19), R.M_JACKET, jv + 0.2)
                Bx.part(co, cyaw, (-0.25, 0.5, 0.17), (0.02, 0.2, 0.035), R.M_JACKET, jv + 0.4)
            Bx.end()
    B, G = Bx.arrays()
    return B, G, np.array(VB, np.float64), np.array(HB, np.float64)


def floor_ao(B, ppm):
    LX, LZ = R.LX, R.LZ
    w, h = int(LX * ppm), int(LZ * ppm)
    occ = np.zeros((h, w), np.float32)
    cont = np.zeros((h, w), np.float32)
    for r in B:
        mat = int(r[8])
        if mat in (R.M_FRAME, R.M_LIGHT, R.M_PARAPET, R.M_BALC, R.M_RAIL, R.M_PAPER, R.M_CLOCK, R.M_SPEAKER,
                   R.M_VALANCE, R.M_SILL, R.M_BOARD, R.M_BOARDFRAME, R.M_TRAY):
            continue
        c, hh, cs, sn = r[:3], r[3:6], r[6], r[7]
        if c[2] < 0 or c[1] - hh[1] > 1.5:
            continue
        ht = c[1] - hh[1]
        wgt = math.exp(-ht / 0.5) * (0.6 if mat != R.M_DESKTOP else 0.35)
        pts = []
        for sx, sz in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            lx, lz = sx * hh[0], sz * hh[2]
            pts.append(((c[0] + cs * lx + sn * lz) * ppm, (c[2] - sn * lx + cs * lz) * ppm))
        m = np.zeros_like(occ)
        cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0)
        occ = np.maximum(occ, m * wgt)
        if ht < 0.01 and max(hh[0], hh[2]) < 0.03:
            # leg foot: tight contact shadow
            cv2.circle(cont, (int(c[0] * ppm), int(c[2] * ppm)), max(1, int(0.035 * ppm)), 1.0, -1, cv2.LINE_AA)
    occ = C.blur(occ, 0.06 * ppm) * 0.6 + C.blur(occ, 0.25 * ppm) * 0.6 + C.blur(cont, 0.018 * ppm) * 0.75
    xs, zs = C.grid(w, h)
    xs, zs = xs / ppm, zs / ppm
    wall = np.exp(-np.minimum(np.minimum(xs, LX - xs), LZ - zs) / 0.25) * 0.35 + np.exp(-zs / 0.3) * 0.25
    return np.clip(1 - occ - wall, 0.25, 1).astype(np.float32)


def board_tex(ppm, seed=3):
    """Chalk on the blackboard (coverage 0..1 over wall coords: x = z * ppm, y = (HC - y) * ppm):
    a maths lesson in jittered Japanese glyphs, a parabola sketch, the date and the day-duty name
    written vertically at the right edge, and half-erased smudges."""
    from PIL import Image, ImageDraw, ImageFont
    rng = np.random.default_rng(seed)
    w, h = int(R.LZ * ppm), int(R.HC * ppm)
    ss = 2
    img = Image.new('L', (w * ss, h * ss), 0)
    dr = ImageDraw.Draw(img)
    fpath = None
    for f in ('C:/Windows/Fonts/YuGothM.ttc', 'C:/Windows/Fonts/msgothic.ttc'):
        if os.path.exists(f):
            fpath = f
            break

    def P(z, y):
        return z * ppm * ss, (R.HC - y) * ppm * ss

    def text(z, y, s_, size, vertical=False, val=225):
        px = int(size * 1.25 * ppm * ss)
        font = ImageFont.truetype(fpath, px)
        x0, y0 = P(z, y)
        for ch in s_:
            if ch == ' ':
                if vertical:
                    y0 += px * 0.5
                else:
                    x0 += px * 0.45
                continue
            # each glyph slightly rotated / offset / sized: hand-written feel
            g = Image.new('L', (px * 2, px * 2), 0)
            ImageDraw.Draw(g).text((px * 0.5, px * 0.35), ch, fill=int(val * rng.uniform(0.8, 1.0)), font=font)
            g = g.rotate(rng.normal(0, 4), resample=Image.BILINEAR)
            sc = rng.uniform(0.92, 1.06)
            g = g.resize((int(px * 2 * sc), int(px * 2 * sc)), Image.BILINEAR)
            ox = int(x0 - px * 0.5 + rng.normal(0, px * 0.03))
            oy = int(y0 - px * 0.35 + rng.normal(0, px * 0.04))
            cur = np.array(img.crop((ox, oy, ox + g.size[0], oy + g.size[1])))
            img.paste(Image.fromarray(np.maximum(cur, np.array(g))), (ox, oy))
            adv = dr.textlength(ch, font=font)
            if vertical:
                y0 += px * 1.02
            else:
                x0 += adv * rng.uniform(0.98, 1.06)

    if fpath:
        text(1.42, 2.08, '\u4e8c\u6b21\u95a2\u6570\u306e\u30b0\u30e9\u30d5', 0.1)
        text(1.5, 1.88, 'y = a(x \u2212 p)\u00b2 + q', 0.085)
        text(1.5, 1.7, '\u9802\u70b9 (p, q)   \u8ef8 x = p', 0.075)
        text(1.5, 1.5, '\u4f8b)  y = 2x\u00b2 \u2212 8x + 5', 0.075)
        text(1.8, 1.32, '= 2(x \u2212 2)\u00b2 \u2212 3', 0.075)
        text(1.5, 1.12, '\u2192 \u9802\u70b9 (2, \u22123)', 0.075)
        text(5.62, 2.08, '9\u670822\u65e5(\u706b)', 0.07, vertical=True)
        text(5.44, 2.08, '\u65e5\u76f4 \u4f50\u85e4', 0.07, vertical=True)
        text(3.5, 2.08, '\u5bbf\u984c P.48 \u554f3', 0.065, val=200)
    arr = cv2.dilate(np.array(img), np.ones((3, 3), np.uint8))
    # parabola sketch with axes
    lw = max(1, int(0.007 * ppm * ss))
    cz, cy = 4.62, 1.45

    def ip(z, y):
        a, b = P(z, y)
        return int(a), int(b)
    cv2.line(arr, ip(cz - 0.42, cy - 0.15), ip(cz + 0.45, cy - 0.15), 210, lw, cv2.LINE_AA)
    cv2.line(arr, ip(cz - 0.3, cy - 0.35), ip(cz - 0.3, cy + 0.42), 210, lw, cv2.LINE_AA)
    xs = np.linspace(-0.3, 0.35, 40)
    pts = np.array([ip(cz + x, cy - 0.28 + 5.0 * (x - 0.05) ** 2) for x in xs], np.int32)
    cv2.polylines(arr, [pts], False, 230, lw, cv2.LINE_AA)
    cv2.circle(arr, ip(cz + 0.05, cy - 0.28), lw * 2, 230, -1, cv2.LINE_AA)
    T = cv2.resize(arr.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    # chalk grain: strokes break up a little
    grain = C.fbm(w, h, max(4.0, w / (0.012 * ppm)), 2, seed=seed + 4)
    T = T * (0.65 + 0.5 * grain)
    # eraser smudges: wide soft horizontal swipes
    sm = np.zeros((h, w), np.float32)
    for i in range(30):
        z0 = rng.uniform(1.2, 5.4)
        y0 = rng.uniform(1.0, 2.1)
        ln = rng.uniform(0.3, 1.0)
        pts = [(int((z0 + ln * u) * ppm), int((R.HC - y0 - 0.05 * math.sin(u * 3 + i)) * ppm))
               for u in np.linspace(0, 1, 12)]
        cv2.polylines(sm, [np.array(pts, np.int32)], False, float(rng.uniform(0.04, 0.1)), int(0.1 * ppm),
                      cv2.LINE_AA)
    # a few broad, half-erased eraser arcs (brighter chalk-dust clouds) and dust along the bottom edge
    for i in range(7):
        z0 = rng.uniform(1.3, 5.6)
        y0 = rng.uniform(1.0, 2.1)
        ln = rng.uniform(0.3, 0.7)
        amp = rng.uniform(0.03, 0.08)
        pts = [(int((z0 + ln * u) * ppm), int((R.HC - y0 - amp * math.sin(u * 3.14)) * ppm))
               for u in np.linspace(0, 1, 16)]
        cv2.polylines(sm, [np.array(pts, np.int32)], False, float(rng.uniform(0.12, 0.22)), int(0.11 * ppm),
                      cv2.LINE_AA)
    sm = C.blur(sm, 0.03 * ppm)
    ys = (np.arange(h, dtype=np.float32)[:, None] + 0.5) / ppm
    yy = R.HC - ys
    dustb = np.exp(-np.maximum(yy - 0.95, 0) / 0.09) * (yy > 0.93) * 0.22
    dustb = dustb * (0.5 + 0.9 * C.fbm(w, h, max(4.0, w / (0.08 * ppm)), 3, seed=seed + 9))
    sm = sm * (0.6 + 0.7 * C.fbm(w, h, max(4.0, w / (0.05 * ppm)), 3, seed=seed + 7))
    return np.clip(T * 0.9 + sm + dustb, 0, 1).astype(np.float32)


class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        k = W / 1920.0
        self.fpx = 0.72 * W
        # sun: late afternoon, low, seen through the transom of the third window bay
        # painter's cheat: the visible sun disc sits in the transom (Lvis) while the key light comes from
        # a little higher (L), so the window grid lands on the floor as long parallelograms in view.
        self.Lvis = _norm((0.60437538, 0.17109619, -0.77811085))
        self.L = _norm((0.60437538, 0.36, -0.77811085))
        L = self.L
        self.sun_uv = (self.Lvis[0] / -self.Lvis[2], self.Lvis[1] / -self.Lvis[2])
        self.B, self.G, self.VB, self.HB = build_room()
        self.fao_ppm = 110.0
        self.FAO = floor_ao(self.B, self.fao_ppm)
        self.bt_ppm = 300.0
        chalk = board_tex(self.bt_ppm)
        import s09_classroom_wall as WD
        dec = WD.wall_decals(self.bt_ppm, R.LZ, R.HC)
        hh_, ww_ = chalk.shape
        dec = cv2.resize(dec, (ww_, hh_), interpolation=cv2.INTER_AREA) if dec.shape[:2] != (hh_, ww_) else dec
        self.BT = np.ascontiguousarray(np.concatenate([chalk[..., None], dec], -1).astype(np.float32))
        self.sky_ppu = 1250.0 * k
        self.SKY, _ = O.sky_plate(self.sky_ppu, self.sun_uv)
        self.tppm = 130.0 * k
        self.TREE = O.tree_plate(self.tppm, sun_dir_screen=(0.3, -1.0))
        self.cm_ppm = 100.0
        self.cm_y0 = R.HC
        self.CM = np.zeros((int(R.HC * self.cm_ppm), int(R.LX * self.cm_ppm)), np.float32)
        sun = np.array([1.0, 0.7, 0.42]) * 3.0
        amb = np.array([0.38, 0.45, 0.7]) * 1.55
        self.P = np.array([L[0], L[1], L[2], sun[0], sun[1], sun[2], amb[0], amb[1], amb[2], 0.8, 0.35, 0.0],
                          np.float64)
        self.jit = ((np.add.outer(np.arange(64), np.arange(64) * 0.618) * 0.754877) % 1.0)
        self.vol_k = 0.24
        self.vol_c, self.vol_base, self.vol_hp = 2.0, 0.5, 3.2
        self.sub = None
        self.emask = None
        self.flare = None
        self.dust = self.spawn_dust()
        self.curtains = [K.Curtain(0.55, 0.98, 5, 0.05, 0.14, 1, 0.3),
                         K.Curtain(2.76, 3.75, 7, 0.03, 0.8, 1, 0.0),
                         K.Curtain(4.8, 5.3, 5, 0.03, 0.55, 1, 1.7),
                         K.Curtain(6.02, 6.46, 5, 0.05, 0.2, -1, 2.6),
                         K.Curtain(6.85, 7.28, 5, 0.05, 0.16, 1, 4.1),
                         K.Curtain(8.26, 8.68, 5, 0.05, 0.12, -1, 5.3)]
        tris, uvs, base = [], [], 0
        for c in self.curtains:
            tris.append(K.grid_tris(c.nv, c.nu, base))
            uu, vv = np.meshgrid(np.linspace(0, 1, c.nu), np.linspace(0, 1, c.nv))
            uvs.append(np.stack([uu.ravel(), vv.ravel()], 1))
            base += c.nu * c.nv
        self.tris = np.concatenate(tris, 0).astype(np.int64)
        self.UV = np.concatenate(uvs, 0).astype(np.float64)
        xs, ys = C.grid(W, H)
        sxp, syp = xs + 0.5 - W / 2, H / 2 - (ys + 0.5)
        self.zfac = (self.fpx / np.sqrt(self.fpx ** 2 + sxp ** 2 + syp ** 2)).astype(np.float32)

    def spawn_dust(self, n_beam=620, n_amb=140, seed=11):
        """Motes seeded where they will glitter: inside the sunbeams (mostly the shafts from the centre
        window toward the blackboard), plus a few dim ones at the beam fringes."""
        rng = np.random.default_rng(seed)
        N = 60000
        pos = np.stack([rng.uniform(0.8, 8.6, N), rng.uniform(0.2, 2.8, N), rng.uniform(0.25, 6.0, N)], 1)
        L = self.L
        tw = -pos[:, 2] / L[2]
        xw = pos[:, 0] + L[0] * tw
        yw = pos[:, 1] + L[1] * tw
        a = np.array([R.aperture(float(x), float(y), 0.004 + float(t_) * 0.012, self.VB, self.HB)
                      for x, y, t_ in zip(xw, yw, tw)])
        cam = self.camera(DURATION / 2)
        rel = pos - cam[:3]
        zc = rel @ cam[3:6]
        sx = 0.5 + (rel @ cam[6:9]) / np.maximum(zc, 1e-3) * self.fpx / self.W
        sy = 0.5 - (rel @ cam[9:12]) / np.maximum(zc, 1e-3) * self.fpx / self.W * (self.W / self.H)
        ok = (zc > 0.8) & (sx > -0.02) & (sx < 1.02) & (sy > -0.02) & (sy < 1.02)
        reg = np.exp(-((sx - 0.68) / 0.16) ** 2 - ((sy - 0.46) / 0.22) ** 2)
        wb = np.where(ok & (a > 0.45), (0.25 + 3.0 * reg) * a, 0.0)
        ib = rng.choice(N, n_beam, replace=False, p=wb / wb.sum())
        wa = np.where(ok & (a > 0.05) & (a <= 0.45), 1.0, 0.0)
        ia = rng.choice(N, n_amb, replace=False, p=wa / wa.sum())
        P = X.make_dust(0, seed=seed, pos=pos[np.concatenate([ib, ia])])
        P[:, 3:6] *= 2.2                                          # visible drift / tumbling through the beam
        P[:, 9] = 0.35 + 0.65 * rng.random(len(P)) ** 1.5
        return P

    def shape_vol(self, vol):
        # compress the hot core near the sun, then boost the beam structure (shafts with dark gaps)
        W = self.W
        c = self.vol_c
        v = vol / (1.0 + vol / c)
        base = F.fast_blur(v, 0.035 * W)
        return (self.vol_base * v + self.vol_hp * np.maximum(v - 0.8 * base, 0.0)).astype(np.float32)

    def camera(self, t):
        u = C.ease_in_out_sine(t / DURATION)
        p0 = np.array([0.95, 1.18, 6.1])
        tgt = np.array([6.4, 0.95, 0.2])
        f = _norm(tgt - p0)
        rt_ = _norm(np.cross(f, (0, 1, 0)))
        # slow eased lateral dolly (left -> right across the aisle) + a push-in toward the windows
        pos = p0 + f * (0.95 * u) + rt_ * (0.85 * (u - 0.5))
        pos = pos + np.array([0.0, 0.01 * math.sin(t * 0.6), 0.0])
        yaw_extra = 0.05 * (u - 0.5)
        c, s = math.cos(yaw_extra), math.sin(yaw_extra)
        f = _norm((f[0] * c + f[2] * s, f[1] - 0.012 * u, -f[0] * s + f[2] * c))
        r = _norm(np.cross(f, (0, 1, 0)))
        up = np.cross(r, f)
        return np.concatenate([pos, f, r, up]).astype(np.float64)

    def frame(self, t):
        W, H = self.W, self.H
        cam = self.camera(t)
        self.P[11] = t
        meshes = [c.verts(t) for c in self.curtains]
        K.shadow_mask(meshes, self.L, self.CM, self.cm_ppm, self.cm_y0)
        col = np.zeros((H, W, 3), np.float32)
        alb = np.zeros_like(col)
        direct = np.zeros_like(col)
        refl = np.zeros_like(col)
        rw = np.zeros((H, W), np.float32)
        depth = np.zeros((H, W), np.float32)
        oid = np.zeros((H, W), np.float32)
        args = (self.B, self.G, self.VB, self.HB, self.CM, self.cm_ppm, self.cm_y0, self.SKY, self.sky_ppu, O.U0,
                O.V1, self.TREE, self.tppm, O.TX0, O.TY1, self.BT, self.bt_ppm, self.FAO, self.fao_ppm, self.P)
        R.render(W, H, cam, self.fpx, *args, col, alb, direct, refl, rw, depth, oid)
        # adaptive AA on edges / shadow boundaries
        if self.emask is None:
            self.emask = np.zeros((H, W), np.uint8)
        X.edges(col, 0.15, self.emask)
        ys, xs = np.nonzero(self.emask)
        R.render_aa(W, H, cam, self.fpx, *args, ys.astype(np.int64), xs.astype(np.int64), col, alb, direct, rw)
        # glossy reflections: blurred at half res (stretched vertically, like a polished floor)
        k = W / 1920.0
        h2, w2 = H // 2, W // 2
        rwh = cv2.resize(rw, (w2, h2), interpolation=cv2.INTER_AREA)
        np.multiply(refl, rw[..., None], out=refl)
        rfh = cv2.resize(refl, (w2, h2), interpolation=cv2.INTER_AREA)
        rfh = cv2.GaussianBlur(rfh, (0, 0), sigmaX=1.1 * k, sigmaY=11.0 * k)
        rwh = cv2.GaussianBlur(rwh, (0, 0), sigmaX=1.1 * k, sigmaY=11.0 * k)
        reflb = cv2.resize(rfh, (W, H), interpolation=cv2.INTER_LINEAR)
        rwb = cv2.resize(rwh, (W, H), interpolation=cv2.INTER_LINEAR)
        # screen-space bounce light (sunlit patches glow onto everything around them)
        gi = F.fast_blur(direct, 0.06 * W) * 0.75 + F.fast_blur(direct, 0.015 * W) * 0.3
        img = np.empty_like(col)
        X.combine1(col, reflb, rwb, rw, gi, alb, img)
        # curtains (rasterised cloth, 4x coverage AA, z-tested against the ray-traced scene)
        V = np.concatenate([m.reshape(-1, 3) for m in meshes], 0)
        N = np.concatenate([K.normals(m).reshape(-1, 3) for m in meshes], 0)
        zs = depth * self.zfac
        if self.sub is None:
            self.sub = (np.zeros((H, W, 4, 3), np.float32), np.zeros((H, W, 4), np.float32),
                        np.zeros((H, W, 4), np.float32), np.zeros((H, W), np.float32))
        sub_c, sub_a, sub_z, cov = self.sub
        K.init_bufs(zs, sub_a, sub_z)
        K.raster(W, H, cam, self.fpx, V, N, self.UV, self.tris, zs, self.P, self.VB, self.HB, sub_c, sub_a, sub_z)
        K.resolve(img, sub_c, sub_a, sub_z, depth, self.zfac, cov)
        # volumetric light shafts (quarter res)
        q = 4
        wq, hq = W // q, H // q
        dlo = cv2.resize(depth, (wq, hq), interpolation=cv2.INTER_NEAREST)
        vol = np.zeros((hq, wq), np.float32)
        R.volume(wq, hq, cam, self.fpx / q, dlo, self.L[0], self.L[1], self.L[2], self.VB, self.HB, self.CM,
                 self.cm_ppm, self.cm_y0, self.B, self.G, self.jit, t, 36, vol)
        # fill the thin sash-bar shadows inside the beams (a closing): occluded air must read as missing
        # light, never as dark bars of smoke across the windows
        kq = max(3, int(0.022 * wq) | 1)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kq, kq))
        vol = cv2.morphologyEx(vol, cv2.MORPH_CLOSE, ker)
        vol = cv2.GaussianBlur(vol, (0, 0), 1.2)
        vol = cv2.resize(vol, (W, H), interpolation=cv2.INTER_CUBIC)
        np.maximum(vol, 0.0, out=vol)
        self.last_vol = vol
        vol = self.shape_vol(vol)
        # sun flare: visibility = fraction of a small disc around the sun that shows the open sky
        f, r_, u_ = cam[3:6], cam[6:9], cam[9:12]
        Ld = self.Lvis @ f
        lx = W / 2 + self.fpx * (self.Lvis @ r_) / Ld
        ly = H / 2 - self.fpx * (self.Lvis @ u_) / Ld
        rr = max(2, int(0.008 * W))
        x0, y0 = int(lx), int(ly)
        if rr <= x0 < W - rr and rr <= y0 < H - rr:
            vis = float(((oid[y0 - rr:y0 + rr + 1, x0 - rr:x0 + rr + 1] < 0) *
                         (1 - cov[y0 - rr:y0 + rr + 1, x0 - rr:x0 + rr + 1])).mean())
        else:
            vis = 0.0
        self.sun_vis = vis
        if self.flare is None:
            # the sun barely moves on screen: render the flare once, then translate it
            c2 = self.camera(DURATION / 2)
            Ld2 = self.Lvis @ c2[3:6]
            self.flare_xy = (W / 2 + self.fpx * (self.Lvis @ c2[6:9]) / Ld2,
                             H / 2 - self.fpx * (self.Lvis @ c2[9:12]) / Ld2)
            self.flare = F.anime_flare(W, H, self.flare_xy[0], self.flare_xy[1], intensity=1.0,
                                       tint=(1.0, 0.8, 0.55), rays=12, ray_len=0.11, spike_width=0.022,
                                       starburst=2.0, hair=0.0, ghosts=0.0, halo=1.15, glow=1.0, streak=0.35)
            self.flare += X.starburst(W, H, self.flare_xy[0], self.flare_xy[1]) * 1.1
            self.flare += X.hex_ghosts(W, H, self.flare_xy[0], self.flare_xy[1])
        M = np.float32([[1, 0, lx - self.flare_xy[0]], [0, 1, ly - self.flare_xy[1]]])
        fl = cv2.warpAffine(self.flare, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        X.combine2(img, vol, fl, 0.4 + 0.6 * vis, self.vol_k, np.array([1.0, 0.78, 0.5]), 1.0)
        X.dust(img, depth, cam, self.fpx, self.dust, t, self.L, self.VB, self.HB, self.CM, self.cm_ppm, self.cm_y0,
               np.array([1.0, 0.85, 0.62]), 2.6)
        img = X.bloom_half(img, threshold=0.85, knee=0.4, strength=0.38, halation=0.45)
        img = F.shoulder(img, 0.8, desat=0.2)
        img = F.finish_fast(img, t, sat=1.1, grain_amt=0.004, vig=0.25, ca=0.0)
        return self.lift_darks(img)

    def lift_darks(self, img):
        """Painted shadows never go dead: lift the darkest values into a saturated blue-violet, warmed
        to ochre where the shadows sit next to sunlit floor / desks (bounce light)."""
        H, W = img.shape[:2]
        lum = img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722
        sm = cv2.resize(lum, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
        hot = np.clip((sm - 0.45) * 2.5, 0, 1)
        warm = cv2.GaussianBlur(hot, (0, 0), 0.012 * W / 8 * 4)
        warm = np.clip(cv2.resize(warm, (W, H), interpolation=cv2.INTER_LINEAR) * 3.0, 0, 1)
        k = np.clip(1.0 - lum / 0.3, 0, 1) ** 2
        vio = np.array([0.085, 0.07, 0.15], np.float32)
        och = np.array([0.15, 0.095, 0.05], np.float32)
        lift = vio[None, None] + (och - vio)[None, None] * (0.75 * warm)[..., None]
        return np.clip(img + lift * k[..., None], 0, 1).astype(np.float32)
