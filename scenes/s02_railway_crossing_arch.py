"""Rural architecture for s02_railway_crossing (round 3): detailed Japanese farmhouses for the midground plate.

Drawn in true perspective into the supersampled vector canvas (hard clean silhouettes, flat painted values
with a few stepped gradients): kawara tile roofs (tile columns with lit crests, course lines, round eave-tile
ends, ridge + onigawara, sky-reflection sheen), a skirt roof (hisashi) between floors, eave occlusion bands
and the verge shadow on the sunlit gable wall, aluminium sash windows with sky / cloud reflections, curtains
and storm-shutter boxes, air-conditioner outdoor units with fan grilles and pipe covers, gutters and
downpipes, TV antennas, a balcony with laundry and a futon over the rail, block walls with openwork blocks,
mesh fences, hedges, a corrugated shed, a farm-stand with nobori banners and a hand-painted sign (real
Japanese text rendered with a system Japanese font).
"""
import math

import numpy as np
import cv2

import s02_railway_crossing_paint as P
import s02_railway_crossing_props as PR
import s02_railway_crossing_art as A

hx = P.hexc


def _shade(base, ndl, lit_gain=(1.06, 1.02, 0.94), sh_mul=(0.62, 0.68, 0.84), sh_add=(0.04, 0.07, 0.13)):
    base = np.asarray(base, np.float32)
    lit = base * np.asarray(lit_gain, np.float32)
    sh = base * np.asarray(sh_mul, np.float32) + np.asarray(sh_add, np.float32)
    k = float(P.sstep(-0.08, 0.1, ndl))
    return np.clip(sh + (lit - sh) * k, 0, 1.2)


class Ctx:
    """Projection + fog helpers for one building (plate coordinates of the t=0 camera)."""

    def __init__(self, sc, cv, ox, fog, hazec=hx('#a9cde2')):
        self.sc, self.cv, self.ox, self.fog = sc, cv, ox, fog
        self.hazec = np.asarray(hazec, np.float32)
        self.c0 = dict(cz=0.0, cx=0.0, fz=1.0)
        self.L = sc.L

    def pr(self, x, y, z):
        a, b = self.sc.proj(x, y, z, self.c0)
        return (float(a) + self.ox, float(b))

    def k(self, z):
        return self.sc.f / max(z, 0.1)

    def fc(self, c):
        return np.asarray(c, np.float32) * (1 - self.fog) + self.hazec * self.fog

    def _blend(self, pts2, col, a, width=None):
        """Alpha-blended poly / polyline on the canvas (the canvas primitives overwrite pixels)."""
        cv = self.cv
        ss = cv.ss
        p = np.asarray(pts2, np.float64) * ss
        pad = (width or 0) * ss + 2
        x0, y0 = np.floor(p.min(0) - pad).astype(int)
        x1, y1 = np.ceil(p.max(0) + pad).astype(int) + 1
        Hb, Wb = cv.buf.shape[:2]
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, Wb), min(y1, Hb)
        if x1 <= x0 or y1 <= y0:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        q = np.round((p - [x0, y0]) * 16).astype(np.int32)
        if width is None:
            cv2.fillPoly(m, [q], 255, cv2.LINE_8, shift=4)
        else:
            cv2.polylines(m, [q], False, 255, max(1, int(round(width * ss))), cv2.LINE_8, shift=4)
        k = (m.astype(np.float32) * (a / 255.0))[..., None]
        reg = cv.buf[y0:y1, x0:x1].astype(np.float32)
        c = np.array([*np.clip(col, 0, 1) * 255.0, 255.0], np.float32)
        cv.buf[y0:y1, x0:x1] = np.clip(reg * (1 - k) + c * k + 0.5, 0, 255).astype(np.uint8)

    def quad(self, pts3, col, a=1.0):
        pts = [self.pr(*p) for p in pts3]
        if a >= 0.999:
            self.cv.poly(pts, self.fc(col), a)
        else:
            self._blend(pts, self.fc(col), a)

    def line(self, pts3, w_m, col, a=1.0, min_px=0.5):
        z = float(np.mean([p[2] for p in pts3]))
        pts = [self.pr(*p) for p in pts3]
        w = max(min_px, w_m * self.k(z))
        if a >= 0.999:
            self.cv.line(pts, w, self.fc(col), a)
        else:
            self._blend(pts, self.fc(col), a, width=w)

    def rect(self, F, u0, v0, u1, v1, col, a=1.0):
        self.quad([F(u0, v0), F(u1, v0), F(u1, v1), F(u0, v1)], col, a)

    def circle3(self, F, u, v, r, col, n=20, a=1.0):
        pts = [F(u + r * math.cos(t), v + r * math.sin(t)) for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]
        self.quad(pts, col, a)


def ground_shadow(C, footprint, h, alpha=0.5):
    """Cast shadow of a prism (footprint [(X, Z)], height h) on the ground, alpha-blended."""
    Lx, Ly, Lz = C.L
    pts = list(footprint) + [(x - h * Lx / Ly, z - h * Lz / Ly) for (x, z) in footprint]
    P2 = np.array([C.pr(x, 0.0, z) for (x, z) in pts], np.float32)
    hull = cv2.convexHull(P2)[:, 0, :]
    C._blend([tuple(p) for p in hull], C.fc(hx('#2c4a52')), alpha)


def _face_front(x0, z):
    return lambda u, v, d=0.0: (x0 + u, v, z - d)


def _face_side(xs, z0, sgn):
    """Side face at x = xs, u runs along +Z; sgn = outward normal sign (+1 = faces +X)."""
    return lambda u, v, d=0.0: (xs + sgn * d, v, z0 + u)


# ------------------------------------------------------------------------------------------ components

def window(C, F, u0, v0, u1, v1, rng, lit=False, frame=hx('#c4c8cc'), curtain=None, shutter_box=0, shoji=False,
           lattice=False):
    """Aluminium sliding sash: frame, two glass panes reflecting the sky (pale top, deep blue body, a cloud
    smudge and a diagonal glare), central meeting rails, sill, optional curtain / shoji inside, optional
    storm-shutter box (tobukuro) beside it (-1 left / +1 right)."""
    fr = _shade(frame, 0.4 if lit else -0.3)
    m = 0.05
    if shutter_box:
        bw = (u1 - u0) * 0.28
        ua, ub = (u1 + 0.02, u1 + 0.02 + bw) if shutter_box > 0 else (u0 - 0.02 - bw, u0 - 0.02)
        C.rect(F, ua, v0 - 0.05, ub, v1 + 0.08, _shade(hx('#b9b4a6'), 0.4 if lit else -0.3))
        for j in range(1, 5):
            uu = ua + (ub - ua) * j / 5
            C.line([F(uu, v0), F(uu, v1 + 0.05)], 0.012, _shade(hx('#8e8a7e'), -0.3))
    C.rect(F, u0, v0, u1, v1, fr)
    # glass: sky reflection (bright on the lit side; on the shaded front the open sky still mirrors pale)
    top = hx('#bcdcf4') if lit else hx('#8fb8e0')
    mid_ = hx('#5a8cc8') if lit else hx('#3b64a0')
    low = hx('#2f5590') if lit else hx('#1f3a6a')
    g0, g1 = u0 + m, u1 - m
    h0, h1 = v0 + m, v1 - m
    hh = h1 - h0
    C.rect(F, g0, h0, g1, h1, low)
    C.rect(F, g0, h0 + hh * 0.38, g1, h1, mid_)
    C.rect(F, g0, h0 + hh * 0.78, g1, h1, top)
    # cloud reflection smudge on one pane
    if rng.random() < 0.7:
        cu = rng.uniform(g0 + 0.15 * (g1 - g0), g1 - 0.25 * (g1 - g0))
        cw = (g1 - g0) * rng.uniform(0.18, 0.3)
        C.quad([F(cu, h0 + hh * 0.5), F(cu + cw, h0 + hh * 0.56), F(cu + cw * 1.2, h0 + hh * 0.72),
                F(cu + cw * 0.3, h0 + hh * 0.76), F(cu - cw * 0.2, h0 + hh * 0.62)], hx('#eef6fc'), 0.8)
    if curtain is not None:
        side = rng.random() < 0.5
        cu0, cu1 = (g0, g0 + (g1 - g0) * rng.uniform(0.2, 0.4)) if side else (g1 - (g1 - g0) * rng.uniform(0.2, 0.4), g1)
        C.rect(F, cu0, h0, cu1, h1, curtain, 0.85)
        C.line([F(cu0 + (cu1 - cu0) * 0.3, h0), F(cu0 + (cu1 - cu0) * 0.3, h1)], 0.015, curtain * 0.8, 0.6)
    if shoji:
        C.rect(F, g0 + 0.02, h0 + 0.02, (g0 + g1) / 2, h1 - 0.02, hx('#ece6d6'), 0.75)
        for j in range(1, 4):
            uu = g0 + ((g0 + g1) / 2 - g0) * j / 4
            C.line([F(uu, h0), F(uu, h1)], 0.012, hx('#b0a48c'), 0.8)
        for j in range(1, 6):
            vv = h0 + hh * j / 6
            C.line([F(g0, vv), F((g0 + g1) / 2, vv)], 0.012, hx('#b0a48c'), 0.8)
    # diagonal glare streaks
    for (a_, b_, al) in ((0.12, 0.26, 0.7), (0.32, 0.37, 0.5)):
        C.quad([F(g0 + (g1 - g0) * a_, h1), F(g0 + (g1 - g0) * b_, h1), F(g0 + (g1 - g0) * (b_ - 0.12), h0),
                F(g0 + (g1 - g0) * max(a_ - 0.12, 0), h0)], hx('#f2f9ff'), al)
    if lattice:
        for j in range(1, 6):
            uu = u0 + (u1 - u0) * j / 6
            C.line([F(uu, v0, 0.04), F(uu, v1, 0.04)], 0.03, _shade(hx('#b8bcc0'), 0.3 if lit else -0.3))
    # meeting rails + frame (dark reveal line along the top inside the frame)
    C.rect(F, g0, h1 - 0.05, g1, h1, hx('#18263e'), 0.7)
    C.line([F((u0 + u1) / 2, v0), F((u0 + u1) / 2, v1)], 0.045, fr)
    C.line([F(u0, v1 - m * 0.5), F(u1, v1 - m * 0.5)], 0.035, _shade(frame, 0.9 if lit else 0.0))
    # sill (lit top line) + shadow under it
    C.rect(F, u0 - 0.04, v0 - 0.05, u1 + 0.04, v0, _shade(hx('#d0d2d0'), 0.6))
    C.rect(F, u0 - 0.02, v0 - 0.11, u1 + 0.02, v0 - 0.05, hx('#000000'), 0.1)


def ac_unit(C, F, u, v, face_n, rng, scale=1.0, lit=False):
    """Air-conditioner outdoor unit standing against a wall (u = left edge, v = base height), with a fan grille,
    side vent, lit top, pipe cover running up the wall and a contact shadow."""
    w, h, d = 0.8 * scale, 0.56 * scale, 0.3 * scale
    body = hx('#f2f2ea')
    L = C.L
    ndl_front = -L[2] if face_n == 'z' else (L[0] if face_n == '+x' else -L[0])
    # contact shadow on the wall
    C.rect(F, u + 0.04, v - 0.06, u + w + 0.16, v + h + 0.02, hx('#1a2434'), 0.55)
    # box: front face (offset d), top, visible side (toward +x when on a front wall)
    fr = [F(u, v, d), F(u + w, v, d), F(u + w, v + h, d), F(u, v + h, d)]
    C.quad([F(u, v + h, 0.0), F(u + w, v + h, 0.0), F(u + w, v + h, d), F(u, v + h, d)], _shade(body, 0.8))
    C.quad([F(u + w, v, 0.0), F(u + w, v, d), F(u + w, v + h, d), F(u + w, v + h, 0.0)],
           _shade(body, 0.5 if face_n == 'z' else -0.2))
    C.quad(fr, _shade(body, ndl_front))
    # fan grille: dark disc, lighter rings and a cross guard
    fu, fv, fr_ = u + w * 0.4, v + h * 0.5, h * 0.36
    Ff = lambda a, b: F(a, b, d + 0.001)
    C.circle3(Ff, fu, fv, fr_, hx('#2e323a'))
    C.circle3(Ff, fu, fv, fr_ * 0.66, hx('#454a54'))
    C.circle3(Ff, fu, fv, fr_ * 0.3, hx('#6c717a'))
    C.line([Ff(fu - fr_, fv), Ff(fu + fr_, fv)], 0.012, hx('#9aa0a8'))
    C.line([Ff(fu, fv - fr_), Ff(fu, fv + fr_)], 0.012, hx('#9aa0a8'))
    # side vent slats on the right of the front
    for j in range(6):
        vv = v + h * (0.2 + 0.1 * j)
        C.line([Ff(u + w * 0.78, vv), Ff(u + w * 0.94, vv)], 0.015, _shade(hx('#9c9e9a'), ndl_front))
    # lit top edge
    C.line([F(u, v + h, d), F(u + w, v + h, d)], 0.02, _shade(hx('#fafaf4'), 1.0))
    # feet
    C.rect(F, u + 0.06, v - 0.08, u + 0.14, v, hx('#6a6c70'))
    C.rect(F, u + w - 0.14, v - 0.08, u + w - 0.06, v, hx('#6a6c70'))
    # pipe cover (beige duct) from the unit's right side up the wall to a wall cap
    pc = _shade(hx('#d9cfb4'), ndl_front if face_n == 'z' else 0.4)
    top = v + h + rng.uniform(0.6, 1.4)
    C.line([F(u + w + 0.08, v + h * 0.4, 0.06), F(u + w + 0.08, top, 0.06)], 0.1, pc)
    C.line([F(u + w + 0.08, top, 0.06), F(u + w - 0.25, top, 0.06)], 0.1, pc)
    C.rect(F, u + w - 0.35, top - 0.1, u + w - 0.2, top + 0.1, _shade(hx('#e8e2d0'), ndl_front))


def gutter(C, x0, x1, y, z, lit=0.6):
    """Half-round eave gutter with a lit top highlight."""
    C.line([(x0, y, z), (x1, y, z)], 0.13, _shade(hx('#8e8e88'), -0.2))
    C.line([(x0, y + 0.05, z - 0.02), (x1, y + 0.05, z - 0.02)], 0.035, _shade(hx('#e8e6dc'), lit))


def downpipe(C, x, y0, y1, z):
    C.line([(x, y0, z), (x, y1, z)], 0.08, _shade(hx('#8e8e88'), -0.2))
    C.line([(x + 0.025, y0, z - 0.01), (x + 0.025, y1, z - 0.01)], 0.02, _shade(hx('#c8c8c0'), 0.3))
    for yy in np.arange(y0 + 0.6, y1, 1.2):
        C.line([(x - 0.06, yy, z), (x + 0.06, yy, z)], 0.04, hx('#6c6e70'))
    C.line([(x, y1, z), (x - 0.2, y1 + 0.15, z + 0.3)], 0.08, _shade(hx('#8e8e88'), -0.2))


def antenna(C, x, y, z, rng, h=2.6):
    """Rooftop TV antenna: mast with guy wires, Yagi boom with director elements and a UHF reflector."""
    dk = hx('#3a3e48')
    top = y + h
    C.line([(x, y, z), (x, top, z)], 0.05, dk)
    for (dx, dz) in ((-1.3, 0.8), (1.2, 0.9), (0.3, -1.0)):
        C.line([(x, top - 0.8, z), (x + dx, y - 0.1, z + dz)], 0.012, dk, 0.6)
    bl = 1.9
    C.line([(x - bl * 0.4, top - 0.25, z), (x + bl * 0.6, top - 0.25, z)], 0.035, dk)
    for j in range(9):
        xx = x - bl * 0.35 + bl * 0.9 * j / 8
        e = 0.34 - 0.015 * j
        C.line([(xx, top - 0.25, z - e), (xx, top - 0.25, z + e)], 0.022, dk)
    # reflector
    C.line([(x - bl * 0.4, top - 0.55, z), (x - bl * 0.4, top + 0.05, z)], 0.03, dk)
    C.line([(x - 0.3, top + 0.2, z), (x + 0.5, top + 0.2, z)], 0.03, dk)


def laundry(C, x0, x1, y, z, rng):
    """Laundry pole (monohoshi-zao) with towels and shirts hanging in the still air, pegs hanger."""
    C.line([(x0, y, z), (x1, y, z)], 0.04, _shade(hx('#c8ccd2'), 0.6))
    cols = [hx('#f6f4ee'), hx('#b9d4ee'), hx('#f2c6c8'), hx('#fbfbf6'), hx('#9fb8d8'), hx('#f1e3b0'),
            hx('#ffffff'), hx('#d4e6c4')]
    x = x0 + 0.2
    while x < x1 - 0.5:
        w = rng.uniform(0.35, 0.7)
        c = cols[int(rng.integers(0, len(cols)))]
        kind = rng.random()
        ndl = -0.1 + 0.2 * rng.random()
        if kind < 0.45:          # towel
            hgt = rng.uniform(0.5, 0.8)
            C.quad([(x, y, z), (x + w, y, z), (x + w - 0.02, y - hgt, z - 0.03), (x + 0.02, y - hgt, z - 0.03)],
                   _shade(c, ndl))
            C.line([(x, y - 0.03, z - 0.01), (x + w, y - 0.03, z - 0.01)], 0.04, _shade(c, 0.9))
        elif kind < 0.8:         # T-shirt on a hanger
            hgt = rng.uniform(0.6, 0.72)
            cx = x + w / 2
            C.line([(cx, y, z), (cx, y - 0.08, z)], 0.015, hx('#50545c'))
            C.quad([(cx - 0.3, y - 0.1, z), (cx + 0.3, y - 0.1, z), (cx + 0.42, y - 0.28, z), (cx + 0.2, y - 0.3, z),
                    (cx + 0.2, y - hgt, z), (cx - 0.2, y - hgt, z), (cx - 0.2, y - 0.3, z), (cx - 0.42, y - 0.28, z)],
                   _shade(c, ndl))
            w = 0.8
        else:                    # pinch hanger with small items
            C.line([(x, y - 0.12, z), (x + 0.55, y - 0.12, z)], 0.02, hx('#9aa0aa'))
            for j in range(4):
                xx = x + 0.04 + j * 0.13
                cc = cols[int(rng.integers(0, len(cols)))]
                C.rect(lambda u, v, d=0.0: (u, v, z), xx, y - 0.4, xx + 0.1, y - 0.14, _shade(cc, ndl))
            w = 0.6
        x += w + rng.uniform(0.08, 0.25)


def block_wall(C, x0, x1, z, h=1.25, gap=None, lit_ndl=-0.2):
    """Concrete-block wall with a lit cap, block joints and a row of openwork (decorative) blocks."""
    base = hx('#c4bfb2')
    segs = [(x0, x1)] if gap is None else [(x0, gap[0]), (gap[1], x1)]
    F = lambda u, v, d=0.0: (u, v, z - d)
    for (u0, u1) in segs:
        C.rect(F, u0, 0, u1, h, _shade(base, lit_ndl))
        C.rect(F, u0, 0, u1, 0.25, _shade(hx('#a8a498'), lit_ndl))       # damp / grime at the foot
        for j in range(1, int(h / 0.2)):
            C.line([F(u0, 0.2 * j), F(u1, 0.2 * j)], 0.012, _shade(hx('#8c887c'), lit_ndl), 0.8)
        nb = int((u1 - u0) / 0.4)
        for i in range(nb + 1):
            for j in range(int(h / 0.2)):
                uu = u0 + 0.4 * i + (0.2 if j % 2 else 0.0)
                if uu < u1:
                    C.line([F(uu, 0.2 * j), F(uu, 0.2 * j + 0.2)], 0.01, _shade(hx('#8c887c'), lit_ndl), 0.6)
        # openwork blocks in the second-highest row
        vv = h - 0.4
        i = 0
        while u0 + 0.4 * (i + 1) < u1:
            if i % 3 == 1:
                a_ = u0 + 0.4 * i + 0.06
                C.rect(F, a_, vv + 0.04, a_ + 0.28, vv + 0.16, hx('#2c3440'))
                C.rect(F, a_ + 0.12, vv + 0.04, a_ + 0.16, vv + 0.16, _shade(base, lit_ndl))
            i += 1
        # cap
        C.quad([(u0 - 0.02, h, z), (u1 + 0.02, h, z), (u1 + 0.02, h + 0.06, z + 0.12), (u0 - 0.02, h + 0.06, z + 0.12)],
               _shade(hx('#e6e2d6'), 0.8))
        C.rect(F, u0 - 0.02, h - 0.05, u1 + 0.02, h + 0.02, _shade(hx('#d8d4c8'), lit_ndl + 0.2))


def mesh_fence(C, x0, x1, z, h=1.1, post=2.0):
    """Green-coated wire-mesh fence: posts, top rail, diamond mesh (thin semi-transparent lines)."""
    col = hx('#3e6a58')
    F = lambda u, v, d=0.0: (u, v, z)
    C.line([F(x0, h), F(x1, h)], 0.04, _shade(col, 0.2))
    x = x0
    while x <= x1 + 1e-3:
        C.line([F(x, 0), F(x, h + 0.05)], 0.05, _shade(col, 0.0))
        x += post
    k = C.k(z)
    step = max(0.15, 2.2 / k)
    x = x0
    while x < x1:
        C.line([F(x, 0), F(min(x + h, x1), min(h, x1 - x))], 0.01, col, 0.35)
        C.line([F(x + h, 0), F(max(x, x0), h - max(0, x0 - x))], 0.01, col, 0.35)
        x += step


def hedge(C, x0, x1, z, h, rng, lit_ndl=0.3):
    """Clipped hedge (painted): dark body, mid band, scalloped lit top edge with leaf flecks."""
    F = lambda u, v, d=0.0: (u, v, z)
    C.rect(F, x0, 0, x1, h, hx('#1f5046'))
    C.rect(F, x0, h * 0.45, x1, h, hx('#2f6c44'))
    k = C.k(z)
    n = max(4, int((x1 - x0) * 5))
    for i in range(n):
        u = x0 + (x1 - x0) * (i + rng.uniform(0.1, 0.9)) / n
        r = rng.uniform(0.12, 0.2)
        yy = h - rng.uniform(0.0, 0.08)
        C.cv.ellipse(C.pr(u, yy, z), (r * k, r * 0.6 * k), 0, hx('#4e9640') * (1 - C.fog) + C.hazec * C.fog)
        C.cv.ellipse(C.pr(u + 0.03, yy + 0.05, z), (r * 0.7 * k, r * 0.35 * k), 0,
                     hx('#9dcc4c') * (1 - C.fog) + C.hazec * C.fog)


# ------------------------------------------------------------------------------------------ roof

def tile_roof(C, rx0, rx1, eY, eZ, ht, rz, tile, lit_ndl, seed=0, sheen=True):
    """Front slope of a kawara gable roof from the eave line (eY at z=eZ) up to the ridge (ht at rz)."""
    base = _shade(tile, lit_ndl)
    C.quad([(rx0, eY, eZ), (rx1, eY, eZ), (rx1, ht, rz), (rx0, ht, rz)], base)
    # sky-reflection sheen: stepped lighter bands toward the ridge (painted, 2 steps)
    if sheen:
        for f0, amt in ((0.55, 0.12), (0.82, 0.22)):
            y0 = eY + (ht - eY) * f0
            z0 = eZ + (rz - eZ) * f0
            C.quad([(rx0, y0, z0), (rx1, y0, z0), (rx1, ht, rz), (rx0, ht, rz)], hx('#dce8f8'), amt)
    k = C.k(eZ)
    ncol = int((rx1 - rx0) / 0.28)
    step = 1 if 0.28 * k >= 3.5 else (2 if 0.28 * k >= 1.8 else 3)
    groove = base * 0.45 + np.asarray(hx('#1a2032'), np.float32) * 0.55
    crest = np.clip(base * 1.3 + 0.07, 0, 1)
    for i in range(0, ncol + 1, step):
        xx = rx0 + (rx1 - rx0) * i / ncol
        C.line([(xx, eY, eZ), (xx, ht, rz)], 0.05, groove, 0.9)
        C.line([(xx + 0.12, eY, eZ), (xx + 0.12, ht, rz)], 0.035, crest, 0.7)
    nc = int(math.hypot(ht - eY, rz - eZ) / 0.3)
    for j in range(1, nc):
        f_ = j / nc
        C.line([(rx0, eY + (ht - eY) * f_, eZ + (rz - eZ) * f_), (rx1, eY + (ht - eY) * f_, eZ + (rz - eZ) * f_)],
               0.025, groove, 0.6)
    # eave: dark underside, row of round tile ends with a lit upper lip
    C.quad([(rx0, eY, eZ), (rx1, eY, eZ), (rx1, eY - 0.28, eZ), (rx0, eY - 0.28, eZ)], hx('#1c2232'))
    F = lambda u, v, d=0.0: (u, v, eZ - 0.02)
    if 0.14 * k >= 1.2:
        for i in range(0, ncol + 1, step):
            xx = rx0 + (rx1 - rx0) * i / ncol + 0.12
            C.circle3(F, xx, eY - 0.06, 0.12, _shade(tile, -0.4) * 0.8, n=10)
            C.circle3(F, xx - 0.02, eY - 0.03, 0.06, _shade(tile, 0.6), n=8)
    C.line([(rx0, eY + 0.03, eZ), (rx1, eY + 0.03, eZ)], 0.06, np.clip(base * 1.3 + 0.08, 0, 1))


def ridge(C, rx0, rx1, ht, rz, tile, big=True):
    h = 0.45 if big else 0.3
    C.quad([(rx0 + 0.15, ht - 0.05, rz), (rx1 - 0.15, ht - 0.05, rz), (rx1 - 0.15, ht + h, rz), (rx0 + 0.15, ht + h, rz)],
           _shade(tile, -0.3) * 0.85)
    C.quad([(rx0 + 0.15, ht + h - 0.12, rz), (rx1 - 0.15, ht + h - 0.12, rz), (rx1 - 0.15, ht + h, rz),
            (rx0 + 0.15, ht + h, rz)], _shade(hx('#c8d4ea'), 0.8))
    for j in range(1, 3):
        yy = ht + h * j / 3
        C.line([(rx0 + 0.15, yy, rz), (rx1 - 0.15, yy, rz)], 0.02, _shade(tile, -0.6) * 0.7, 0.8)
    for ex in (rx0 + 0.15, rx1 - 0.15):
        C.quad([(ex - 0.28, ht - 0.05, rz), (ex + 0.28, ht - 0.05, rz), (ex + 0.36, ht + h + 0.35, rz),
                (ex, ht + h + 0.5, rz), (ex - 0.36, ht + h + 0.35, rz)], _shade(tile, -0.4) * 0.75)
        C.line([(ex - 0.3, ht + h + 0.3, rz), (ex, ht + h + 0.46, rz), (ex + 0.3, ht + h + 0.3, rz)], 0.04,
               _shade(hx('#c8d4ea'), 0.8), 0.8)


# ------------------------------------------------------------------------------------------ building

def house(sc, cv, X, Z, prm, ox, fog):
    """Japanese rural house (1 or 2 storeys) with gable kawara roof and dense detail."""
    C = Ctx(sc, cv, ox, fog)
    rng = np.random.default_rng(prm.get('seed', 0))
    wd, dp = prm['wd'], prm['dp']
    floors = prm.get('floors', 1)
    hw = 2.9 * floors + 0.3                       # wall height under the main eave
    ht = hw + prm.get('pitch', 0.42) * dp / 2 + 0.2
    x0, x1 = X - wd / 2, X + wd / 2
    Zf, Zb = Z, Z + dp
    see_plus_x = X < 0
    xs = x1 if see_plus_x else x0
    Lx = sc.L[0]
    side_ndl = Lx if see_plus_x else -Lx
    side_lit = side_ndl > 0
    front_ndl = -0.3
    tile = prm.get('tile', hx('#5d6a86'))
    wall = prm.get('wall', 'plaster')
    wall_c = {'plaster': hx('#efe6d2'), 'siding': hx('#dcdcd2'), 'wood': hx('#4e3e36'), 'cream': hx('#f2e8cc')}[wall]
    low_c = hx('#5a4a40') if wall != 'wood' else hx('#3a2e2a')
    # ground shadow
    ground_shadow(C, [(x0 - 0.9, Zf - 0.9), (x1 + 0.9, Zf - 0.9), (x1 + 0.9, Zb + 0.9), (x0 - 0.9, Zb + 0.9)],
                      ht * 0.8, 0.55)
    ssgn = 1.0 if see_plus_x else -1.0
    Fs = _face_side(xs, Zf, ssgn)
    Ff = _face_front(x0, Zf)
    # ---- side (gable) wall: stepped value bands (ground bounce low, darker under the verge)
    sw = _shade(wall_c, side_ndl)
    C.quad([Fs(0, 0), Fs(dp, 0), Fs(dp, hw), Fs(0, hw)], sw)
    C.quad([Fs(0, hw), Fs(dp, hw), Fs(dp / 2, ht - 0.25)], sw * 0.97)
    C.quad([Fs(0, 0), Fs(dp, 0), Fs(dp, 0.45), Fs(0, 0.45)], _shade(low_c, side_ndl))
    if wall == 'siding':
        for j in range(2, int(hw / 0.25)):
            C.line([Fs(0, 0.25 * j), Fs(dp, 0.25 * j)], 0.012, sw * 0.85, 0.7)
    if wall == 'wood':
        if True:
            for j in range(1, int(dp / 0.2)):
                C.line([Fs(0.2 * j, 0), Fs(0.2 * j, hw)], 0.012, sw * 0.75, 0.7)
    if side_lit:
        # verge shadow: a band parallel to the gable slope + eave-corner shadow
        ov = 0.6
        t_ = 0.83 * ov
        C.quad([Fs(0, hw), Fs(dp / 2, ht - 0.25), Fs(dp / 2 + 0.5 * ov, ht - 0.25 - t_), Fs(0.5 * ov, hw - t_)],
               hx('#34455e'), 0.45)
        C.quad([Fs(dp / 2, ht - 0.25), Fs(dp, hw), Fs(dp + 0.1, hw - t_), Fs(dp / 2 + 0.5 * ov, ht - 0.25 - t_)],
               hx('#34455e'), 0.45)
        C.quad([Fs(0, hw), Fs(dp, hw), Fs(dp, hw - t_), Fs(0, hw - t_)], hx('#34455e'), 0.45)
    else:
        C.quad([Fs(0, hw), Fs(dp, hw), Fs(dp, hw - 0.55), Fs(0, hw - 0.55)], hx('#1e2838'), 0.5)
    # side windows
    for j in range(floors):
        vb = 0.9 + 2.9 * j
        u0 = dp * rng.uniform(0.3, 0.4)
        window(C, lambda u, v, d=0.0: Fs(u, v, d + 0.01), u0, vb, u0 + 1.2, vb + 1.0, rng, lit=side_lit,
               curtain=hx('#efe8d6') if rng.random() < 0.6 else None)
    # ---- front wall (in shade, sun ahead): stepped bands, cool under the eaves, warm bounce low
    fw = _shade(wall_c, front_ndl) * 0.74
    C.rect(Ff, 0, 0, wd, hw, fw)
    C.rect(Ff, 0, 0, wd, 0.9, fw * 0.96 + np.asarray(hx('#c8b89a'), np.float32) * 0.06)
    C.rect(Ff, 0, 0, wd, 0.45, _shade(low_c, front_ndl))
    if wall == 'siding':
        for j in range(2, int(hw / 0.25)):
            C.line([Ff(0, 0.25 * j), Ff(wd, 0.25 * j)], 0.012, fw * 0.88, 0.7)
    if wall == 'wood':
        for j in range(1, int(wd / 0.2)):
            C.line([Ff(0.2 * j, 0.45), Ff(0.2 * j, hw)], 0.012, fw * 0.75, 0.7)
    # posts
    for i in range(prm.get('posts', 0)):
        uu = wd * i / max(prm['posts'] - 1, 1)
        C.rect(Ff, uu - 0.07, 0, uu + 0.07, hw, _shade(hx('#4a3a30'), front_ndl))
    # ground-floor openings: entrance + windows; upper-floor windows
    ent = prm.get('entrance', 0.36)
    ea, eb = wd * ent, wd * ent + 1.8
    for (a_, b_) in [(0.5, ea - 0.6), (eb + 0.5, wd - 0.6)]:
        n = max(1, int((b_ - a_) / 2.4))
        wwid = (b_ - a_) / n
        for i in range(n):
            ua = a_ + i * wwid + 0.15
            ub = ua + wwid - 0.3
            window(C, Ff, ua, 0.75, ub, 2.35, rng, lit=False, curtain=hx('#e8e2d2') if rng.random() < 0.5 else None,
                   shutter_box=(1 if i == n - 1 else 0), shoji=rng.random() < 0.3, lattice=(i == 0 and rng.random() < 0.5))
    # entrance: sliding frosted doors, lamp, canopy, name plate
    C.rect(Ff, ea, 0, eb, 2.1, _shade(hx('#8a8c90'), front_ndl))
    C.rect(Ff, ea + 0.08, 0.05, eb - 0.08, 2.02, hx('#d9e2e8'))
    for j in range(1, 6):
        C.line([Ff(ea + 0.08, 2.0 * j / 6), Ff(eb - 0.08, 2.0 * j / 6)], 0.02, hx('#9aa2aa'), 0.8)
    C.line([Ff((ea + eb) / 2, 0.05), Ff((ea + eb) / 2, 2.02)], 0.05, hx('#8a9098'))
    C.rect(Ff, eb + 0.15, 1.3, eb + 0.33, 1.62, hx('#6a5440'))                      # name plate
    C.rect(Ff, eb + 0.18, 1.34, eb + 0.3, 1.58, hx('#e8dcc0'))
    C.circle3(lambda u, v, d=0.0: Ff(u, v, 0.05), ea - 0.25, 2.2, 0.1, hx('#fff4d8'))  # porch lamp
    C.quad([Ff(ea - 0.4, 2.35, 0.9), Ff(eb + 0.4, 2.35, 0.9), Ff(eb + 0.4, 2.5, 0.0), Ff(ea - 0.4, 2.5, 0.0)],
           _shade(hx('#6e7688'), 0.6))
    C.rect(Ff, ea - 0.4, 2.1, eb + 0.4, 2.35, hx('#28303c'), 0.45)
    if floors >= 2:
        for i in range(3):
            ua = wd * (0.08 + 0.32 * i)
            window(C, Ff, ua, 3.75, ua + wd * 0.2, 5.0, rng, lit=False,
                   curtain=hx('#f2e6cc') if rng.random() < 0.6 else hx('#d8e4ee'), shutter_box=(1 if i == 2 else 0))
    # eave occlusion band on the front wall
    C.rect(Ff, 0, hw - 0.75, wd, hw, hx('#26324a'), 0.45)
    C.rect(Ff, 0, hw - 0.4, wd, hw, hx('#1c2638'), 0.6)
    C.rect(Ff, 0, hw - 0.15, wd, hw, hx('#121a28'), 0.6)
    # AC units
    for (face, au, av) in prm.get('ac', []):
        if face == 'front':
            ac_unit(C, Ff, au, av, 'z', rng, scale=1.2)
        else:
            ac_unit(C, lambda u, v, d=0.0: Fs(u, v, d), au, av, '+x' if see_plus_x else '-x', rng, lit=side_lit,
                    scale=1.2)
    # downpipes at the front corners
    downpipe(C, x0 + 0.12, 0.0, hw, Zf - 0.1)
    downpipe(C, x1 - 0.12, 0.0, hw, Zf - 0.1)
    # ---- skirt roof between the floors (2-storey)
    if floors >= 2:
        sy = 3.05
        tile_roof(C, x0 - 0.3, x1 + 0.3, sy - 0.1, Zf - 0.95, sy + 0.45, Zf, tile, 0.5, sheen=False)
        C.rect(Ff, 0, sy - 0.75, wd, sy - 0.2, hx('#26324a'), 0.5)
        C.rect(Ff, 0, sy - 0.45, wd, sy - 0.2, hx('#141c2a'), 0.55)
        gutter(C, x0 - 0.3, x1 + 0.3, sy - 0.2, Zf - 1.0)
        if prm.get('balcony'):
            # balcony on the upper floor: slab, railing, laundry, futon
            ba, bb = x0 + wd * 0.45, x0 + wd * 0.95
            bz = Zf - 1.0
            C.quad([(ba, sy + 0.45, bz), (bb, sy + 0.45, bz), (bb, sy + 0.45, Zf), (ba, sy + 0.45, Zf)],
                   _shade(hx('#c8c4b8'), 0.6))
            C.quad([(ba, sy + 0.25, bz), (bb, sy + 0.25, bz), (bb, sy + 0.5, bz), (ba, sy + 0.5, bz)],
                   _shade(hx('#b8b4a8'), -0.3))
            laundry(C, ba + 0.2, bb - 0.2, sy + 2.2, bz + 0.45, rng)
            for u in np.arange(ba, bb + 0.01, 0.12):
                C.line([(u, sy + 0.5, bz), (u, sy + 1.55, bz)], 0.02, _shade(hx('#9aa0a8'), 0.2), 0.9)
            C.line([(ba, sy + 1.55, bz), (bb, sy + 1.55, bz)], 0.06, _shade(hx('#c8ccd2'), 0.8))
            C.line([(ba, sy + 1.0, bz), (bb, sy + 1.0, bz)], 0.03, _shade(hx('#9aa0a8'), 0.2))
            fa = ba + (bb - ba) * rng.uniform(0.05, 0.3)
            fb = fa + 1.6
            C.quad([(fa, sy + 1.6, bz - 0.02), (fb, sy + 1.6, bz - 0.02), (fb, sy + 0.75, bz - 0.06),
                    (fa, sy + 0.75, bz - 0.06)], _shade(hx('#f2d6d0'), -0.05))
            for j in range(4):
                C.circle3(lambda u, v, d=0.0: (u, v, bz - 0.07), fa + 0.25 + 0.38 * j, sy + 1.15 + 0.12 * (j % 2),
                          0.1, hx('#e49aa0'), n=8)
            C.line([(fa, sy + 1.6, bz - 0.03), (fb, sy + 1.6, bz - 0.03)], 0.05, hx('#fff6f0'))
    # ---- main roof
    ov = 0.75
    eY = hw + 0.1
    rz = (Zf + Zb) / 2
    rx0, rx1 = x0 - ov * 0.8, x1 + ov * 0.8
    # gable barge + visible roof thickness on the seen side
    bx = rx1 if see_plus_x else rx0
    tile_roof(C, rx0, rx1, eY, Zf - ov, ht, rz, tile, 0.45, seed=prm.get('seed', 0))
    gutter(C, rx0, rx1, eY - 0.18, Zf - ov - 0.05)
    C.line([(bx, eY, Zf - ov), (bx, ht + 0.05, rz), (bx, eY, Zb + ov)], 0.2, _shade(hx('#f0ece2'), side_ndl))
    C.line([(bx, eY - 0.12, Zf - ov), (bx, ht - 0.1, rz), (bx, eY - 0.12, Zb + ov)], 0.08, hx('#2c3444'))
    ridge(C, rx0, rx1, ht, rz, tile)
    if prm.get('antenna'):
        antenna(C, x0 + wd * rng.uniform(0.2, 0.35), ht + 0.3, rz + 0.2, rng)
    # ---- yard: block wall / mesh fence / hedge / AC on the ground
    if prm.get('block'):
        bz = Zf - prm.get('yard', 3.2)
        block_wall(C, x0 - 2.0, x1 + 2.5, bz, gap=(x0 + wd * ent - 0.3, x0 + wd * ent + 2.4))
        ground_shadow(C, [(x0 - 2.0, bz), (x1 + 2.5, bz), (x1 + 2.5, bz + 0.15), (x0 - 2.0, bz + 0.15)], 1.25, 0.45)
    if prm.get('hedge'):
        hz_ = Zf - prm.get('yard', 3.2)
        hedge(C, x0 - 1.5, x1 + 1.5, hz_, 1.1, rng)


def shed(sc, cv, X, Z, prm, ox, fog):
    """Corrugated-iron farm shed / tractor garage: ribbed walls, rust streaks, open bay with a dark interior,
    mono-pitch roof with a sunlit ribbed top."""
    C = Ctx(sc, cv, ox, fog)
    wd, dp, h = prm['wd'], prm['dp'], prm['h']
    x0, x1 = X - wd / 2, X + wd / 2
    see_plus_x = X < 0
    xs = x1 if see_plus_x else x0
    ssgn = 1.0 if see_plus_x else -1.0
    side_ndl = sc.L[0] * ssgn
    ground_shadow(C, [(x0, Z), (x1, Z), (x1, Z + dp), (x0, Z + dp)], h)
    wallc = prm.get('wall', hx('#a7aeb0'))
    Fs = _face_side(xs, Z, ssgn)
    Ff = _face_front(x0, Z)
    C.quad([Fs(0, 0), Fs(dp, 0), Fs(dp, h), Fs(0, h + 0.3)], _shade(wallc, side_ndl))
    for j in range(1, int(dp / 0.15)):
        C.line([Fs(0.15 * j, 0), Fs(0.15 * j, h + 0.3 * (1 - 0.15 * j / dp))], 0.02, _shade(wallc, side_ndl) * 0.82, 0.8)
    C.rect(Ff, 0, 0, wd, h + 0.3, _shade(wallc, -0.3))
    C.rect(Ff, wd * 0.12, 0, wd * 0.72, h * 0.86, hx('#232a34'))                     # open bay
    C.rect(Ff, wd * 0.14, 0, wd * 0.34, h * 0.35, hx('#4a5058'))                     # stuff inside
    C.rect(Ff, wd * 0.4, 0, wd * 0.62, h * 0.22, hx('#7a4a38'))
    for j in range(int(wd / 0.15)):
        uu = 0.15 * j
        if wd * 0.12 < uu < wd * 0.72:
            continue
        C.line([Ff(uu, 0), Ff(uu, h + 0.3)], 0.02, _shade(wallc, -0.3) * 0.82, 0.8)
    for uu in (wd * 0.05, wd * 0.8, wd * 0.9):
        C.quad([Ff(uu, h), Ff(uu + 0.12, h), Ff(uu + 0.08, h * 0.4), Ff(uu + 0.03, h * 0.5)], hx('#8a5a3c'), 0.45)
    rc = prm.get('roof', hx('#4d7fb0'))
    C.quad([(x0 - 0.35, h + 0.4, Z - 0.45), (x1 + 0.35, h + 0.4, Z - 0.45), (x1 + 0.35, h + 0.05, Z + dp + 0.3),
            (x0 - 0.35, h + 0.05, Z + dp + 0.3)], _shade(rc, 0.6))
    for j in range(int((wd + 0.7) / 0.2)):
        xx = x0 - 0.35 + 0.2 * j
        C.line([(xx, h + 0.4, Z - 0.45), (xx, h + 0.05, Z + dp + 0.3)], 0.03, _shade(rc, 0.9), 0.6)
    C.quad([(x0 - 0.35, h + 0.4, Z - 0.45), (x1 + 0.35, h + 0.4, Z - 0.45), (x1 + 0.35, h + 0.28, Z - 0.45),
            (x0 - 0.35, h + 0.28, Z - 0.45)], _shade(rc, -0.4))
    C.rect(Ff, 0, h, wd, h + 0.3, hx('#1c2430'), 0.5)


# ------------------------------------------------------------------------------------------ signs

def _text_tex(text, wm, hm, bg, fg, vertical=False, ppm=120, border=None, sub=None):
    T = PR.Tex(wm, hm, ppm)
    T.fill(T.mask_rect(0, 0, wm, hm), bg)
    if border is not None:
        T.fill(T.mask_rect(0, 0, wm, hm) * (1 - T.mask_rect(0.04, 0.04, wm - 0.04, hm - 0.04)), border)
    chars = list(text) if vertical else [text]
    n = len(chars)
    for i, ch in enumerate(chars):
        m = A.text_mask(ch, 96)
        if m is None:
            continue
        if vertical:
            cell_h = (hm - 0.12) / n
            size = min(wm * 0.78, cell_h * 0.9)
            u0 = (wm - size) / 2
            v1 = hm - 0.06 - cell_h * i - (cell_h - size) / 2
            box = (u0, v1 - size, u0 + size, v1)
        else:
            asp = m.shape[1] / m.shape[0]
            th = hm * (0.5 if sub else 0.66)
            tw = min(th * asp, wm * 0.9)
            th = tw / asp
            v0 = hm * (0.38 if sub else 0.5) - th / 2 + (0.08 * hm if sub else 0)
            box = ((wm - tw) / 2, v0, (wm + tw) / 2, v0 + th)
        x0, y0 = T.px(box[0], box[3])
        x1, y1 = T.px(box[2], box[1])
        w_, h_ = max(int(x1 - x0), 1), max(int(y1 - y0), 1)
        mm = cv2.resize(m, (w_, h_), interpolation=cv2.INTER_AREA)
        full = np.zeros((T.h, T.w), np.float32)
        xa, ya = int(x0), int(y0)
        full[max(ya, 0):ya + h_, max(xa, 0):xa + w_] = mm[max(-ya, 0):min(h_, T.h - ya), max(-xa, 0):min(w_, T.w - xa)]
        T.fill(full, fg)
    if sub:
        m = A.text_mask(sub, 64)
        if m is not None:
            asp = m.shape[1] / m.shape[0]
            th = hm * 0.2
            tw = min(th * asp, wm * 0.8)
            th = tw / asp
            x0, y0 = T.px((wm - tw) / 2, hm * 0.2 + th)
            w_, h_ = max(int(tw * ppm), 1), max(int(th * ppm), 1)
            mm = cv2.resize(m, (w_, h_), interpolation=cv2.INTER_AREA)
            full = np.zeros((T.h, T.w), np.float32)
            full[int(y0):int(y0) + h_, int(x0):int(x0) + w_] = mm[:T.h - int(y0), :T.w - int(x0)]
            T.fill(full, fg)
    return T.premult()


def _blit_face(C, tex, F, u0, v0, u1, v1):
    q = [C.pr(*F(u0, v1)), C.pr(*F(u1, v1)), C.pr(*F(u1, v0)), C.pr(*F(u0, v0))]
    t = tex.copy()
    if C.fog > 0:
        t[..., :3] = t[..., :3] * (1 - C.fog) + C.hazec * C.fog * t[..., 3:4]
    PR.blit_quad(C.cv, t, q)


def farm_stand(sc, cv, X, Z, ox, fog, seed=0):
    """Unmanned roadside vegetable stand (mujin hanbaijo): a small timber booth with a tin roof, shelves
    with produce, a painted sign '野菜直売所' and two nobori banners 'とれたて野菜' / '新米'."""
    C = Ctx(sc, cv, ox, fog)
    rng = np.random.default_rng(seed)
    wd, dp, h = 2.2, 1.0, 2.0
    x0, x1 = X - wd / 2, X + wd / 2
    Ff = _face_front(x0, Z)
    ground_shadow(C, [(x0, Z), (x1, Z), (x1, Z + dp), (x0, Z + dp)], h, 0.5)
    wood = hx('#8a6a4c')
    C.rect(lambda u, v, d=0.0: (x0 + u, v, Z + dp), 0, 0, wd, h, _shade(wood, -0.5) * 0.7)   # back wall
    for u in (0.05, wd - 0.05):
        C.line([Ff(u, 0), Ff(u, h)], 0.1, _shade(wood, -0.2))
    C.rect(lambda u, v, d=0.0: (x0 + u, v, Z + 0.3), 0.1, 0.7, wd - 0.1, 0.78, _shade(wood, 0.5))
    C.rect(lambda u, v, d=0.0: (x0 + u, v, Z + 0.5), 0.1, 1.25, wd - 0.1, 1.31, _shade(wood, 0.5))
    prod = [hx('#e8452e'), hx('#3e8a36'), hx('#f0a830'), hx('#7a3a8a'), hx('#e8e0c0'), hx('#2e6e2e')]
    for (vy, zz) in ((0.78, 0.3), (1.31, 0.5)):
        u = 0.2
        while u < wd - 0.35:
            cc = prod[int(rng.integers(0, len(prod)))]
            C.rect(lambda uu, v, d=0.0: (x0 + uu, v, Z + zz - 0.05), u, vy, u + 0.28, vy + 0.1, hx('#d8c8a0'))
            C.circle3(lambda uu, v, d=0.0: (x0 + uu, v, Z + zz - 0.06), u + 0.14, vy + 0.13, 0.08, _shade(cc, 0.3), n=10)
            u += 0.36
    # tin roof
    C.quad([(x0 - 0.2, h + 0.25, Z - 0.35), (x1 + 0.2, h + 0.25, Z - 0.35), (x1 + 0.2, h + 0.05, Z + dp + 0.1),
            (x0 - 0.2, h + 0.05, Z + dp + 0.1)], _shade(hx('#b04a3a'), 0.6))
    C.quad([(x0 - 0.2, h + 0.25, Z - 0.35), (x1 + 0.2, h + 0.25, Z - 0.35), (x1 + 0.2, h + 0.15, Z - 0.35),
            (x0 - 0.2, h + 0.15, Z - 0.35)], _shade(hx('#b04a3a'), -0.5))
    # sign board under the roof
    if not hasattr(sc, '_tx_stand'):
        sc._tx_stand = (_text_tex('野菜直売所', 2.0, 0.42, hx('#f4efe0'), hx('#2a3a8a'), border=hx('#2a3a8a')),
                        _text_tex('とれたて野菜', 0.42, 1.9, hx('#2f8f4e'), hx('#fbfbf2'), vertical=True),
                        _text_tex('新米', 0.42, 1.4, hx('#d8352a'), hx('#fff8e8'), vertical=True))
    ts, tn1, tn2 = sc._tx_stand
    _blit_face(C, ts, lambda u, v, d=0.0: (x0 + u, v, Z - 0.36), 0.1, h - 0.3, wd - 0.1, h + 0.12)
    # nobori banners on poles, left and right of the stand
    for (bx, tex, hh) in ((x0 - 0.7, tn1, 1.9), (x1 + 0.5, tn2, 1.4)):
        C.line([(bx, 0, Z - 0.3), (bx, hh + 0.9, Z - 0.3)], 0.04, hx('#e8e8e4'))
        C.line([(bx, hh + 0.85, Z - 0.3), (bx + 0.45, hh + 0.85, Z - 0.3)], 0.03, hx('#e8e8e4'))
        _blit_face(C, tex, lambda u, v, d=0.0: (bx + u, v, Z - 0.31), 0.03, 0.8, 0.45, 0.8 + hh)


def kei_truck(sc, cv, X, Z, ox, fog, seed=0, facing=-1):
    """White kei truck (light pickup) parked side-on (along X): cab with side glass reflecting the sky,
    flat bed with drop sides and a headboard guard, dark wheels, lit roof / rail edges, contact shadow.
    X = centre, Z = the near side plane; facing -1 = cab on the left."""
    C = Ctx(sc, cv, ox, fog)
    Lg, Wd = 3.4, 1.45
    x0 = X - Lg / 2
    ground_shadow(C, [(x0, Z), (x0 + Lg, Z), (x0 + Lg, Z + Wd), (x0, Z + Wd)], 1.1, 0.5)
    body = hx('#f2f2ee')
    sh = _shade(body, -0.3)
    lit = _shade(body, 1.0)

    def F(u, v, d=0.0):
        uu = u if facing < 0 else Lg - u
        return (x0 + uu, v, Z - d)
    # far-side hint (roof of the cab seen slightly from below is hidden; bed interior: thin dark strip)
    C.rect(F, 1.42, 0.9, 3.4, 0.98, hx('#5a5e66'))
    # underbody + bumpers
    C.rect(F, -0.02, 0.28, 3.42, 0.5, hx('#3a3e46'))
    # bed drop side with hinge lines
    C.rect(F, 1.42, 0.45, 3.4, 0.95, sh)
    C.line([F(1.42, 0.94), F(3.4, 0.94)], 0.035, lit)
    C.line([F(1.42, 0.7), F(3.4, 0.7)], 0.015, sh * 0.85)
    for u in (2.05, 2.75):
        C.line([F(u, 0.47), F(u, 0.93)], 0.015, sh * 0.85)
    # headboard guard (torii frame) behind the cab
    C.rect(F, 1.36, 0.95, 1.44, 1.85, _shade(hx('#c8ccd0'), -0.2))
    for j in range(3):
        C.line([F(1.36, 1.1 + 0.25 * j), F(1.5, 1.1 + 0.25 * j)], 0.02, hx('#8a8e96'))
    # cab
    cab = [F(0.0, 0.35), F(0.0, 1.05), F(0.1, 1.12), F(0.32, 1.8), F(1.34, 1.83), F(1.36, 0.35)]
    C.quad(cab, sh)
    C.quad([F(0.36, 1.13), F(0.47, 1.72), F(1.28, 1.74), F(1.28, 1.13)], hx('#4e76a8'))
    C.quad([F(0.4, 1.3), F(0.46, 1.72), F(1.28, 1.74), F(1.28, 1.55)], hx('#bcdaf2'))
    C.quad([F(0.7, 1.13), F(0.8, 1.13), F(1.05, 1.74), F(0.95, 1.74)], hx('#f0f8ff'), 0.6)
    C.line([F(0.33, 1.8), F(1.34, 1.83)], 0.04, lit)
    C.line([F(0.9, 0.4), F(0.9, 1.12)], 0.012, sh * 0.8)
    C.line([F(1.3, 0.4), F(1.3, 1.74)], 0.012, sh * 0.8)
    C.rect(F, 1.0, 0.95, 1.12, 0.99, hx('#50545c'))                          # door handle
    C.rect(F, 0.02, 0.62, 0.1, 0.8, hx('#f4e8c0'))                             # head lamp edge
    C.rect(F, 0.3, 1.28, 0.36, 1.42, hx('#2c3038'))                            # mirror
    C.rect(F, 3.34, 0.55, 3.41, 0.72, hx('#d83a2a'))                           # tail lamp
    # wheel arches + wheels
    for u in (0.62, 2.78):
        C.circle3(lambda a, b, d=0.0: F(a, b, 0.01), u, 0.34, 0.36, hx('#262a30'), n=16)
        C.circle3(lambda a, b, d=0.0: F(a, b, 0.02), u, 0.3, 0.29, hx('#15171c'), n=16)
        C.circle3(lambda a, b, d=0.0: F(a, b, 0.03), u, 0.3, 0.13, hx('#9aa0a8'), n=10)


def service_pole(sc, cv, X, Z, ox, fog, drops=(), h=8.5, seed=0):
    """Wooden / concrete utility pole by the houses with a crossarm, insulators, a pole transformer and
    service-drop wires sagging to the house eaves (drops = [(x, y, z), ...])."""
    C = Ctx(sc, cv, ox, fog)
    k = C.k(Z)
    pc = _shade(hx('#b8b2a4'), -0.2)
    C.line([(X, 0, Z), (X, h, Z)], 0.28, pc)
    C.line([(X + 0.07, 0.3, Z - 0.01), (X + 0.06, h - 0.1, Z - 0.01)], 0.07, _shade(hx('#d8d2c4'), 0.8), 0.9)
    C.line([(X - 0.9, h - 0.5, Z), (X + 0.9, h - 0.5, Z)], 0.1, hx('#5a5c62'))
    for dx in (-0.8, -0.3, 0.3, 0.8):
        C.line([(X + dx, h - 0.5, Z), (X + dx, h - 0.3, Z)], 0.07, hx('#e8e8e0'))
    # pole-mounted transformer can
    C.quad([(X + 0.18, h - 2.4, Z - 0.3), (X + 0.62, h - 2.4, Z - 0.3), (X + 0.62, h - 1.3, Z - 0.3),
            (X + 0.18, h - 1.3, Z - 0.3)], _shade(hx('#9aa2a8'), -0.2))
    C.line([(X + 0.2, h - 1.35, Z - 0.31), (X + 0.6, h - 1.35, Z - 0.31)], 0.04, _shade(hx('#e4e8ec'), 0.9))
    # step bolts
    for yy in np.arange(2.2, h - 1.0, 0.45):
        side = 1 if int(yy / 0.45) % 2 else -1
        C.line([(X, yy, Z), (X + side * 0.2, yy, Z)], 0.03, hx('#6a6c70'))
    wc = hx('#2a303c')
    for (hx_, hy_, hz_) in drops:
        a = np.array([X + 0.6, h - 0.5, Z])
        b = np.array([hx_, hy_, hz_])
        pts = []
        for f in np.linspace(0, 1, 16):
            p = a + (b - a) * f
            p[1] -= 0.35 * 4 * f * (1 - f)
            pts.append(tuple(p))
        C.line(pts, 0.02, wc, 0.9)
