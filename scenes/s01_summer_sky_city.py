"""Foreground plates for s01 (round 3): Japanese residential rooftops, trees, utility pole and wires.

Painted element by element into straight-alpha RGBA plates with supersampled masks (clean hard-edged
shapes), but every surface carries painted structure:
  * kawara tile roofs: tile courses (a lit lip + a dark under-shadow per course), vertical tile ribs,
    per-tile value jitter, a broad sky-reflection sheen, ridge caps / hip ridges with a hot specular edge,
    fascia + gutter with a glint, a dark eave shadow on the wall below;
  * walls: plaster with a soft vertical gradient, windows with sky reflections + glint streak + sash bars,
    balconies with rails and hanging laundry, AC outdoor units, downpipes;
  * trees: clusters of scalloped leaf masses (unions of leaf-clump circles), dark blue-green undersides,
    mid-tone body, a bright yellow-green crescent on the top-right sun side, lit leaf speckles;
  * atmospheric perspective: each row further away is lighter / bluer / lower contrast.
The sun is high ahead of the camera (behind the cloud): faces toward us are in cool sky-lit shade and
every top edge catches a warm rim.
"""
import math
import numpy as np
import cv2

from s01_summer_sky_town import Canvas, antenna, solar_heater  # noqa: E402  (own shot's helper module)
from PIL import Image, ImageDraw, ImageFont


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _c(*v):
    return np.array(v, np.float32)


def _h(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


RIM = _c(1.4, 1.2, 0.92)
GLINTS = []          # (x, y, size, phase) specular points collected while painting (plate coords)
HAZE = _h('#b8d6f0')

# roof palettes: (top/ridge side, eave side, sky sheen, ridge cap)
ROOF_PALS = [
    (_h('#44557e'), _h('#1e2842'), _h('#9fb6dc'), _h('#1a2238')),   # blue-grey kawara
    (_h('#3d5a6a'), _h('#182a36'), _h('#9cc4d4'), _h('#14222c')),   # teal
    (_h('#56506a'), _h('#262236'), _h('#b2aed0'), _h('#1e1a2c')),   # violet-grey
    (_h('#5a4a52'), _h('#2a1e26'), _h('#c0a8b4'), _h('#20161c')),   # brown
    (_h('#4a5670'), _h('#222a3e'), _h('#a8b8d8'), _h('#1c2234')),   # silver
]
# walls in the cool sky-lit shade, but each house its own paint: cream siding, peach mortar, pale mint,
# blue-grey, warm beige, white tile (warm/cool variation between neighbours)
WALL_PALS = [_h('#b8aa9c'), _h('#a8b4c4'), _h('#c0a898'), _h('#9cb4b0'), _h('#b4a4b0'), _h('#c4bcb0'),
             _h('#8e9cb8'), _h('#bca690')]


def hz(c, k):
    return c * (1 - k) + HAZE * k


def _vnoise(n, seed):
    rng = np.random.default_rng(seed)
    return rng.random(n).astype(np.float32)


# =============================================================================================== roofs
def roof_face(cv, pts, y_top, y_bot, pal, cs, hk, seed, sheen_x=None, lit=1.0):
    """Tile roof face polygon `pts` spanning y_top (ridge) .. y_bot (eave). cs = tile course spacing px."""
    top, bot, sheen, _ = pal
    top, bot, sheen = hz(top, hk), hz(bot, hk), hz(sheen, hk * 0.7)
    jit = _vnoise(4096, seed)
    sx = sheen_x

    def color(gx, gy):
        v = np.clip((gy - y_top) / max(y_bot - y_top, 1), 0, 1)
        c = top + (bot - top) * (v ** 0.7)[..., None]
        # broad sky reflection sheen (upper part of the face, drifting diagonally)
        sh = np.exp(-((v - 0.25) / 0.22) ** 2)
        if sx is not None:
            sh = sh * (0.55 + 0.45 * np.exp(-((gx - sx) / (8 * cs * 6)) ** 2))
        c = c + (sheen - c) * (sh * 0.45 * lit)[..., None]
        # tile courses: lit lip + shadow beneath
        f = np.mod((gy - y_top) / cs, 1.0)
        row = np.floor((gy - y_top) / cs).astype(np.int32)
        lip = _ss(0.0, 0.08, f) * _ss(0.3, 0.14, f)
        und = _ss(0.3, 0.45, f) * _ss(0.75, 0.5, f)
        kc = 1.0 + 0.5 * np.clip((cs - 4.0) / 4.0, 0, 1)       # nearer rows: stronger tile courses
        kn = float(np.clip((cs - 5.5) / 2.5, 0, 1))              # nearest row: full kawara modelling
        c = c * (1 + (0.28 + 0.2 * kn) * kc * lip * lit - (0.2 + 0.22 * kn) * kc * und)[..., None]
        # tile ribs (vertical), staggered per course, with per-tile jitter
        tw = cs * 1.15
        g = (gx / tw + 0.5 * (row % 2) * (1 - kn))
        rib = np.abs(np.mod(g, 1.0) - 0.5) * 2
        c = c * (1 - 0.1 * _ss(0.75, 1.0, rib))[..., None]
        if kn > 0:
            # rounded kawara: each tile column is a barrel (lit crown, shaded troughs), and the glaze on
            # every barrel crown mirrors the bright sky just below each course lip
            ph_ = np.mod(g, 1.0)
            barrel = np.cos((ph_ - 0.4) * 2 * math.pi)
            c = c * (1 + kn * (0.13 * barrel - 0.08 * _ss(0.8, 0.98, ph_)))[..., None]
            glaze = _ss(0.55, 0.95, barrel) * _ss(0.05, 0.12, f) * _ss(0.42, 0.2, f)
            c = c + (hz(_h('#cfe2f6'), hk) - c) * np.clip(kn * glaze * 0.4 * (0.6 + 0.4 * lit), 0, 0.6)[..., None]
            # a thin dark line where each course overlaps the next (per-row shadow line)
            c = c * (1 - kn * 0.35 * np.exp(-((f - 0.97) / 0.035) ** 2))[..., None]
        idx = (np.floor(g).astype(np.int64) * 131 + row * 17) % 4096
        c = c * (0.95 + 0.1 * jit[idx])[..., None]
        # backlit: the tile courses nearest the ridge catch a specular glaze (each lip a bright line)
        spec = np.exp(-v / 0.12) * (0.35 + 0.65 * lip) * lit
        if sx is not None:
            spec = spec * (0.6 + 0.4 * np.exp(-((gx - sx) / (8 * cs * 10)) ** 2))
        c = c + (hz(_h('#fff2da'), hk * 0.5) - c) * np.clip(spec * 0.55, 0, 0.8)[..., None]
        return c
    r_ = cv.poly(pts, color)
    if hk < 0.35 and lit > 0.9 and seed % 3 == 0:
        rng_ = np.random.default_rng(seed + 5)
        P_ = np.asarray(pts, np.float64)
        for _ in range(1):
            u_, v_ = rng_.uniform(0.2, 0.8), rng_.uniform(0.1, 0.35)
            gx_ = P_[:, 0].min() + (P_[:, 0].max() - P_[:, 0].min()) * u_
            GLINTS.append((gx_, y_top + (y_bot - y_top) * v_, 0.6, rng_.uniform(0, 6.28)))
    return r_


def ridge(cv, p0, p1, w, pal, hk, rim_k=1.0):
    cap = hz(pal[3], hk)
    cv.lines([[p0, p1]], max(1.2, w), cap)
    off = w * 0.32
    cv.lines([[(p0[0], p0[1] - off), (p1[0], p1[1] - off)]], max(0.7, w * 0.3), hz(RIM, hk * 0.5), 0.95 * rim_k)
    # a hotter rim segment on the sun-facing half of the ridge
    xm = (p0[0] + p1[0]) / 2
    ym = (p0[1] + p1[1]) / 2
    cv.lines([[(p0[0] + (xm - p0[0]) * 0.3, p0[1] - off + (ym - p0[1]) * 0.3), (xm, ym - off)]], max(0.7, w * 0.4),
             hz(RIM * 1.15, hk * 0.4), rim_k)
    if hk < 0.4 and rim_k > 0.8:
        GLINTS.append((xm - (xm - p0[0]) * 0.2, ym - off, 0.9, (p0[0] * 0.37) % 6.28))


def house(cv, x, y_eave, w, s, rng, hk, kind, wall_h, cs, detail=1.0):
    """One house seen across the rooftops (roof facing us). Returns the roof top y."""
    pal = ROOF_PALS[int(rng.integers(0, len(ROOF_PALS)))]
    wall = hz(WALL_PALS[int(rng.integers(0, len(WALL_PALS)))] * rng.uniform(0.92, 1.05), hk)
    roof_h = w * rng.uniform(0.2, 0.26)
    y_r = y_eave - roof_h
    ov = w * 0.04                                  # eave overhang
    xl, xr = x - w / 2 + ov, x + w / 2 - ov
    y_bot = y_eave + wall_h
    seed = int(rng.integers(1 << 30))

    # ---- wall
    siding = rng.random() < 0.6
    wjit = _vnoise(4096, seed + 77)
    seed_w = int(seed % 997)
    lap = max(cs * 1.1, 2.0)

    def wcol(gx, gy):
        v = np.clip((gy - y_eave) / max(wall_h, 1), 0, 1)
        c = wall * (0.8 + 0.2 * _ss(0.0, 0.35, v))[..., None]
        # sky bounce brightens the left (sun-side) half, the right edge turns cooler
        f = np.clip((gx - xl) / max(xr - xl, 1), 0, 1)
        c = c * (1.04 - 0.12 * f)[..., None] + _h('#9ab4e0') * (0.08 * f)[..., None]
        # weathering: rain stains running down from the eave and from the window sills (per-column
        # streaks), a grimy band along the foot and a soft dirt gradient into the corners
        xq = np.mod((gx - xl) / max(2.6 * s, 1.0) + seed_w, 4000.0)
        st = np.interp(xq, np.arange(4096, dtype=np.float32), wjit).astype(np.float32)
        st = _ss(0.7, 0.97, st) * np.exp(-np.clip(gy - y_eave, 0, None) / max(roof_h * 1.2, 1))
        grime = 0.1 * _ss(0.72, 1.0, v) + 0.06 * (_ss(0.12, 0.0, f) + _ss(0.88, 1.0, f))
        c = c * (1 - 0.1 * st - grime)[..., None]
        if siding:
            ph_ = np.mod((gy - y_eave) / lap, 1.0)
            c = c * (1 - 0.07 * _ss(0.7, 0.95, ph_) + 0.05 * _ss(0.0, 0.12, ph_) * _ss(0.3, 0.12, ph_))[..., None]
        return c
    cv.poly([(xl, y_eave), (xr, y_eave), (xr, y_bot), (xl, y_bot)], wcol)
    # upper floor windows / balcony
    nwin = 2 if w > 14 * cs else 1
    wh = roof_h * rng.uniform(0.75, 0.95)
    wy = y_eave + roof_h * 0.45
    for k in range(nwin):
        wx = x + (k - (nwin - 1) / 2) * w * 0.4 + rng.uniform(-0.04, 0.04) * w
        ww = w * rng.uniform(0.16, 0.24)
        window(cv, wx, wy, ww, wh, s, hk, rng)
        if detail > 0.5 and rng.random() < 0.55:
            balcony(cv, wx, wy + wh * 0.5, ww * 1.5, wh * 0.62, s, hk, rng)
        if detail > 0.5 and rng.random() < 0.6:
            ac_unit(cv, wx + ww * 0.62 + w * 0.02, wy + wh * 0.35, w * 0.1, s, hk)
    # lower floor windows (partly visible)
    if wall_h > roof_h * 2.2:
        wy2 = wy + roof_h * 1.6
        for k in range(nwin):
            wx = x + (k - (nwin - 1) / 2) * w * 0.38 + rng.uniform(-0.05, 0.05) * w
            window(cv, wx, wy2, w * rng.uniform(0.18, 0.26), wh * 1.1, s, hk, rng)
    # downpipes (both corners) + a hose from an AC unit
    for px2 in (xl + w * 0.025,):
        cv.lines([[(px2, y_eave + 0.2 * roof_h), (px2, y_bot)]], max(1.0, 0.9 * s), hz(_h('#5a6680'), hk))
        cv.lines([[(px2 - 0.3 * s, y_eave + 0.2 * roof_h), (px2 - 0.3 * s, y_bot)]], max(0.5, 0.3 * s),
                 hz(_h('#d8e0ee'), hk), 0.7)
    px_ = xr - w * 0.03
    cv.lines([[(px_, y_eave + 0.2 * roof_h), (px_, y_bot)]], max(1.0, 0.9 * s), hz(_h('#5a6680'), hk))
    cv.lines([[(px_ - 0.3 * s, y_eave + 0.2 * roof_h), (px_ - 0.3 * s, y_bot)]], max(0.5, 0.3 * s),
             hz(_h('#c8d2e4'), hk), 0.6)
    # eave shadow on the wall
    cv.poly([(xl, y_eave), (xr, y_eave), (xr, y_eave + roof_h * 0.28), (xl, y_eave + roof_h * 0.28)],
            lambda gx, gy: hz(_h('#141a2c'), hk) * np.ones_like(gx)[..., None], 0.55)
    cv.poly([(xl, y_eave), (xr, y_eave), (xr, y_eave + roof_h * 0.1), (xl, y_eave + roof_h * 0.1)],
            hz(_h('#141a2c'), hk), 0.5)
    # ---- roof
    sxh = x + rng.uniform(-0.3, 0.3) * w
    if kind == 'hip':
        rl = w * rng.uniform(0.2, 0.3)
        face = [(x - w / 2, y_eave), (x + w / 2, y_eave), (x + w / 2 - rl, y_r), (x - w / 2 + rl, y_r)]
        roof_face(cv, face, y_r, y_eave, pal, cs, hk, seed, sheen_x=sxh)
        # hip ends slightly darker (they face sideways)
        for sgn in (-1, 1):
            # hip ends are separate roof planes: the sun-side (left) one lit warm, the other in shadow
            tri = [(x + sgn * w / 2, y_eave), (x + sgn * (w / 2 - rl), y_r),
                   (x + sgn * (w / 2 - rl * (1.15 if sgn < 0 else 0.5)), y_eave)]
            if sgn < 0:
                pl = (pal[0] * 1.25 + _h('#3a2c20') * 0.5, pal[1] * 1.3 + _h('#302418') * 0.4, pal[2] * 1.1, pal[3])
                roof_face(cv, tri, y_r, y_eave, pl, cs, hk, seed + 7, lit=1.2)
            else:
                pl = (pal[0] * 0.7, pal[1] * 0.75, pal[2] * 0.75, pal[3])
                roof_face(cv, tri, y_r, y_eave, pl, cs, hk, seed + 9, lit=0.3)
            ridge(cv, (x + sgn * (w / 2 - rl), y_r), (x + sgn * w / 2, y_eave), 0.9 * cs, pal, hk,
                  1.0 if sgn < 0 else 0.55)
        ridge(cv, (x - w / 2 + rl, y_r), (x + w / 2 - rl, y_r), 1.1 * cs, pal, hk)
        top_y = y_r
    elif kind == 'gable':
        face = [(x - w / 2, y_eave), (x + w / 2, y_eave), (x + w / 2 - w * 0.02, y_r), (x - w / 2 + w * 0.02, y_r)]
        roof_face(cv, face, y_r, y_eave, pal, cs, hk, seed, sheen_x=sxh)
        ridge(cv, (x - w / 2 + w * 0.02, y_r), (x + w / 2 - w * 0.02, y_r), 1.2 * cs, pal, hk)
        for sgn in (-1, 1):
            cv.lines([[(x + sgn * (w / 2 - w * 0.02), y_r), (x + sgn * w / 2, y_eave)]], max(1.2, 0.8 * cs),
                     hz(pal[3], hk))
            cv.ellipse(x + sgn * (w / 2 - w * 0.02), y_r - 0.2 * cs, 0.9 * cs, 0.7 * cs, hz(pal[3], hk))
            cv.ellipse(x + sgn * (w / 2 - w * 0.02) - 0.2 * cs, y_r - 0.5 * cs, 0.45 * cs, 0.3 * cs, hz(RIM, hk * 0.5), 0.8)
        top_y = y_r
    else:  # gable end facing us: triangular wall + two thin roof slopes
        apex = (x, y_r - roof_h * 0.3)
        el = (x - w * 0.5, y_eave + roof_h * 0.08)
        er = (x + w * 0.5, y_eave + roof_h * 0.08)
        cv.poly([(xl, y_eave + roof_h * 0.1), apex, (xr, y_eave + roof_h * 0.1)], wcol)
        # soffit shadow under the barge boards (the overhang shades the gable wall)
        sd = roof_h * 0.16
        for p0 in ((xl, y_eave + roof_h * 0.1), (xr, y_eave + roof_h * 0.1)):
            cv.poly([p0, apex, (apex[0], apex[1] + sd * 1.4), (p0[0], p0[1] + sd)],
                    hz(_h('#1a2238'), hk), 0.42)
        # upper-floor windows in the gable wall (frames + sky reflections), an AC unit, a floor trim band
        gy_ = y_eave - roof_h * 0.62
        gwh = roof_h * 0.5
        ngw = 2 if w > 12 * cs else 1
        for k in range(ngw):
            gwx = x + (k - (ngw - 1) / 2) * w * 0.3
            window(cv, gwx, gy_, w * 0.15, gwh, s, hk, rng)
        if detail > 0.5 and rng.random() < 0.7:
            ac_unit(cv, x + w * (0.22 if ngw == 2 else 0.14), gy_ + gwh * 0.45, w * 0.09, s, hk)
        cv.poly([(xl, y_eave + roof_h * 0.1), (xr, y_eave + roof_h * 0.1), (xr, y_eave + roof_h * 0.2),
                 (xl, y_eave + roof_h * 0.2)], hz(_h('#6a7490'), hk), 0.8)
        cv.lines([[(xl, y_eave + roof_h * 0.1), (xr, y_eave + roof_h * 0.1)]], max(0.6, 0.6 * s), hz(RIM, hk * 0.6), 0.6)
        # vent
        vy = apex[1] + roof_h * 0.32
        cv.poly([(x - w * 0.05, vy), (x + w * 0.05, vy), (x + w * 0.05, vy + roof_h * 0.22),
                 (x - w * 0.05, vy + roof_h * 0.22)], hz(_h('#2a3044'), hk))
        cv.lines([[(x - w * 0.05, vy + roof_h * 0.22 * i / 5), (x + w * 0.05, vy + roof_h * 0.22 * i / 5)]
                  for i in range(1, 5)], max(0.6, 0.4 * s), hz(_h('#8894ac'), hk))
        # barge boards with tile edge + rim
        for p0, p1, rk in ((el, apex, 1.0), (apex, er, 0.5)):
            cv.lines([[p0, apex if p1 is apex else p1]], max(2.0, 1.6 * cs), hz(pal[1], hk))
            cv.lines([[(p0[0], p0[1] - 0.9 * cs), (p1[0], p1[1] - 0.9 * cs)]], max(0.8, 0.35 * cs), hz(RIM, hk * 0.5), rk)
        # a sliver of the side roof slopes with courses
        for sgn in (-1, 1):
            q = [(x, apex[1]), (x + sgn * w * 0.5, el[1]), (x + sgn * w * 0.53, el[1] - roof_h * 0.1), (x, apex[1] - roof_h * 0.1)]
            roof_face(cv, q, apex[1] - roof_h * 0.1, el[1], pal, cs * 0.7, hk, seed + 3, lit=0.6)
        top_y = apex[1]
    # fascia + gutter along the eave with glint
    if kind != 'gable_end':
        cv.lines([[(x - w / 2, y_eave + 0.5 * cs), (x + w / 2, y_eave + 0.5 * cs)]], max(1.2, 1.1 * cs),
                 hz(_h('#3a4460'), hk))
        cv.lines([[(x - w / 2, y_eave + 1.1 * cs), (x + w / 2, y_eave + 1.1 * cs)]], max(1.0, 0.8 * cs),
                 hz(_h('#6a7896'), hk))
        cv.lines([[(x - w / 2, y_eave + 0.8 * cs), (x + w / 2, y_eave + 0.8 * cs)]], max(0.5, 0.25 * cs),
                 hz(_h('#e6eefa'), hk * 0.6), 0.8)
        cv.lines([[(x - w / 2, y_eave - 0.1 * cs), (x + w / 2, y_eave - 0.1 * cs)]], max(0.6, 0.3 * cs),
                 hz(RIM, hk * 0.6), 0.55)
        if cs > 6 * s:
            # half-round gutter: dark underside, bright rolled lip, hangers and a sky-lit top edge
            gy_ = y_eave + 1.6 * cs
            cv.lines([[(x - w / 2, gy_), (x + w / 2, gy_)]], 1.3 * cs, hz(_h('#8a94a8'), hk))
            cv.lines([[(x - w / 2, gy_ + 0.45 * cs), (x + w / 2, gy_ + 0.45 * cs)]], 0.5 * cs, hz(_h('#3e465c'), hk))
            cv.lines([[(x - w / 2, gy_ - 0.45 * cs), (x + w / 2, gy_ - 0.45 * cs)]], 0.3 * cs, hz(_h('#eef4fc'), hk), 0.9)
            nh = max(int(w / (6 * cs)), 2)
            cv.lines([[(x - w / 2 + (k + 0.5) * w / nh, gy_ - 0.9 * cs), (x - w / 2 + (k + 0.5) * w / nh, gy_ + 0.6 * cs)]
                      for k in range(nh)], max(0.8, 0.25 * cs), hz(_h('#4a5268'), hk), 0.9)
    return top_y


def window(cv, wx, wy, ww, wh, s, hk, rng):
    x0, x1, y0, y1 = wx - ww / 2, wx + ww / 2, wy, wy + wh
    warm = rng.random() < 0.3
    cur = rng.uniform(0.3, 0.6)
    cph = 0.1 + 0.6 * ((math.sin(wx * 3.7131 + wy * 11.117) * 24634.6345) % 1.0)

    def gcol(gx, gy):
        v = np.clip((gy - y0) / wh, 0, 1)
        u = np.clip((gx - x0) / ww, 0, 1)
        c = _h('#a8d0f2') * (1 - v[..., None]) + _h('#3a5480') * v[..., None]     # sky reflection
        c = c * (0.85 + 0.15 * u)[..., None]
        streak = np.exp(-(((u + v * 0.7) - 0.55) / 0.07) ** 2) * 0.5 + np.exp(-(((u + v * 0.7) - 0.8) / 0.025) ** 2) * 0.4
        c = c + streak[..., None] * _h('#eaf6ff') * 0.8
        # reflected cumulus: a soft, bright cloud shape in the upper glass (mirrored sky), shifted per
        # window, and a darker reflected-roofline band along the bottom
        cu = cph + 0.25 * np.sin(v * 7.0 + cph * 9.0) * 0.1
        cl = np.exp(-((u - cu) / 0.28) ** 2 - ((v - 0.28) / 0.16) ** 2) +             0.7 * np.exp(-((u - cu - 0.22) / 0.16) ** 2 - ((v - 0.2) / 0.12) ** 2)
        c = c + (_h('#f4f6fa') - c) * np.clip(cl * 0.75, 0, 0.8)[..., None]
        c = c * (1 - 0.25 * _ss(0.8, 0.95, v))[..., None]
        if warm:
            k = _ss(cur, cur + 0.05, v) * 0.55
            c = c + (_h('#c8a890') - c) * k[..., None]
        return hz(c, hk * 0.8)
    cv.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], gcol)
    hsh = (math.sin(wx * 12.9898 + wy * 78.233) * 43758.5453) % 1.0
    if hk < 0.45 and hsh < 0.3:
        GLINTS.append((x0 + ww * 0.78, y0 + wh * 0.2, 0.7 + 0.5 * (1 - hk), hsh * 20.0))
    fr = hz(_h('#c4ccd8'), hk)
    lw = max(0.8, 0.9 * s)
    cv.lines([[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)], [(wx, y0), (wx, y1)]], lw, fr, 0.95)
    cv.lines([[(x0 - ww * 0.06, y1 + lw), (x1 + ww * 0.06, y1 + lw)]], lw * 1.3, hz(_h('#d8dfe8'), hk))
    cv.lines([[(x0, y0 + lw), (x1, y0 + lw)]], lw * 1.6, hz(_h('#20283c'), hk), 0.6)


def balcony(cv, bx, by, bw, bh, s, hk, rng):
    x0, x1 = bx - bw / 2, bx + bw / 2
    lw = max(0.6, 0.55 * s)
    # laundry pole + hanging clothes
    ly = by - bh * 0.9
    cv.lines([[(x0 + bw * 0.04, ly), (x1 - bw * 0.04, ly)]], max(0.7, 0.6 * s), hz(_h('#c8ccd4'), hk))
    cols = [_h('#f4f2ee'), _h('#8fb8e0'), _h('#f0c8b0'), _h('#e8e4f4'), _h('#b0d0b8'), _h('#f6e2a0')]
    xx = x0 + bw * 0.1
    while xx < x1 - bw * 0.2:
        cw = bw * rng.uniform(0.09, 0.16)
        ch = bh * rng.uniform(0.4, 0.75)
        c = cols[int(rng.integers(0, len(cols)))] * 0.78
        cv.poly([(xx, ly), (xx + cw, ly), (xx + cw * 0.95, ly + ch), (xx + cw * 0.05, ly + ch)],
                lambda gx, gy, xx=xx, cw=cw, c=c: hz(c * (0.8 + 0.35 * np.clip(1 - (gx - xx) / cw, 0, 1))[..., None], hk))
        xx += cw + bw * rng.uniform(0.01, 0.05)
    # slab + rail
    cv.poly([(x0, by), (x1, by), (x1, by + bh * 0.16), (x0, by + bh * 0.16)], hz(_h('#6c7690'), hk))
    rails = [[(x0 + bw * i / 18, by - bh * 0.5), (x0 + bw * i / 18, by)] for i in range(19)]
    cv.lines(rails, lw, hz(_h('#4a5470'), hk), 0.9)
    cv.lines([[(x0, by - bh * 0.5), (x1, by - bh * 0.5)]], lw * 2, hz(_h('#5a6480'), hk))
    cv.lines([[(x0, by - bh * 0.5 - lw), (x1, by - bh * 0.5 - lw)]], lw, hz(RIM, hk * 0.5), 0.8)
    cv.lines([[(x0, by), (x1, by)]], lw, hz(_h('#dfe6f2'), hk), 0.8)


def ac_unit(cv, ax, ay, aw, s, hk):
    ah = aw * 0.7
    # cast shadow on the wall below/right (sun behind-left, soft sky shade)
    cv.poly([(ax + aw * 0.08, ay + ah), (ax + aw * 1.06, ay + ah), (ax + aw * 1.06, ay + ah * 1.22),
             (ax + aw * 0.14, ay + ah * 1.22)], hz(_h('#1a2032'), hk), 0.35)
    cv.poly([(ax, ay), (ax + aw, ay), (ax + aw, ay + ah), (ax, ay + ah)],
            lambda gx, gy: hz(_h('#b8bcc8') * (0.8 + 0.2 * np.clip((ay + ah - gy) / ah, 0, 1))[..., None], hk))
    cv.ellipse(ax + aw * 0.38, ay + ah * 0.5, ah * 0.36, ah * 0.36, hz(_h('#5c6274'), hk))
    cv.lines([[(ax + aw * 0.38 - ah * 0.34, ay + ah * 0.5), (ax + aw * 0.38 + ah * 0.34, ay + ah * 0.5)],
              [(ax + aw * 0.38, ay + ah * 0.16), (ax + aw * 0.38, ay + ah * 0.84)]], max(0.5, 0.4 * s),
             hz(_h('#9aa0b0'), hk))
    # side grille: horizontal slats, each with a dark shadow line under a lit edge
    gx0, gx1 = ax + aw * 0.72, ax + aw * 0.94
    ns = max(int(ah / max(2.2 * s, 1.5)), 3)
    cv.poly([(gx0, ay + ah * 0.12), (gx1, ay + ah * 0.12), (gx1, ay + ah * 0.88), (gx0, ay + ah * 0.88)],
            hz(_h('#4c5264'), hk))
    cv.lines([[(gx0, ay + ah * (0.12 + 0.76 * (k + 0.5) / ns)), (gx1, ay + ah * (0.12 + 0.76 * (k + 0.5) / ns))]
              for k in range(ns)], max(0.5, 0.45 * s), hz(_h('#c4c8d4'), hk), 0.9)
    cv.lines([[(ax, ay + ah), (ax + aw, ay + ah)]], max(0.6, 0.6 * s), hz(_h('#3a4052'), hk), 0.9)
    cv.lines([[(ax, ay), (ax + aw, ay)]], max(0.6, 0.5 * s), hz(RIM, hk * 0.5), 0.9)


# =============================================================================================== trees
def tree(cv, x, y, r, s, rng, hk, lean=0.0):
    """Broadleaf tree painted from leaf-cluster dabs (s01_summer_sky_trees)."""
    import s01_summer_sky_trees as TT
    return TT.tree(cv, x, y, r, s, rng, hk, hz, lean)


def tree_old(cv, x, y, r, s, rng, hk, lean=0.0):
    """Broadleaf tree: clusters of scalloped leaf masses with a lit top-right crescent."""
    rng = np.random.default_rng(int(rng.integers(1 << 30)))
    L = np.array([0.3, -0.95])
    dark = hz(_h('#06121f'), hk)           # cool blue-green core shadow
    mid = hz(_h('#173d3a'), hk)
    lit = hz(_h('#5a9c56'), hk * 0.85)
    hot = hz(_h('#e6f7a0') * 1.1, hk * 0.5)
    trunk = hz(_h('#1a1e2c'), hk)
    cv.lines([[(x, y + r * 1.2), (x + lean * r, y), (x - r * 0.3, y - r * 0.4)],
              [(x + lean * r, y + r * 0.1), (x + r * 0.4, y - r * 0.35)]], max(1.0, r * 0.07), trunk)
    n = int(rng.integers(5, 8))
    clumps = []
    for k in range(n):
        a = -math.pi / 2 + (k / max(n - 1, 1) - 0.5) * 2.4 + rng.normal(0, 0.15)
        d = rng.uniform(0.3, 0.7) * r
        rr = r * rng.uniform(0.32, 0.46)
        clumps.append((x + lean * r + math.cos(a) * d * 1.1, y - r * 0.25 + math.sin(a) * d * 0.8, rr))
    clumps.append((x + lean * r, y - r * 0.4, r * 0.5))
    clumps.sort(key=lambda c: -c[1])
    for (cx, cy, rr) in clumps:
        # leaf-clump circles filling an ellipse -> scalloped outline
        m = int(10 + 14 * rng.random())
        ang = rng.uniform(0, 2 * math.pi, m)
        rad = np.sqrt(rng.uniform(0.0, 1.0, m)) * rr * 0.72
        circ = [(cx + math.cos(a_) * d_, cy + math.sin(a_) * d_ * 0.78, rr * rng.uniform(0.2, 0.34))
                for a_, d_ in zip(ang, rad)]
        circ.append((cx, cy, rr * 0.62))
        allp = np.array([[c[0] - c[2], c[1] - c[2]] for c in circ] + [[c[0] + c[2], c[1] + c[2]] for c in circ])
        bb = cv._bbox(allp, 3)
        if bb is None:
            continue
        x0, y0, x1, y1 = bb
        ssf = 4

        def disc_mask(cs_, off=(0, 0), scale=1.0):
            mm = np.zeros(((y1 - y0) * ssf, (x1 - x0) * ssf), np.uint8)
            for (qx, qy, qr) in cs_:
                cv2.circle(mm, (int(((qx + off[0]) - x0) * ssf * 16), int(((qy + off[1]) - y0 - cv.y0) * ssf * 16)),
                           max(1, int(qr * scale * ssf * 16)), 255, -1, cv2.LINE_AA, shift=4)
            return cv2.resize(mm.astype(np.float32) / 255.0, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        m0 = disc_mask(circ)
        cv.put(bb, m0, dark)
        m1 = disc_mask(circ, off=(L[0] * rr * 0.2, L[1] * rr * 0.2), scale=0.85) * m0
        cv.put(bb, m1, mid, 0.95)
        shf = disc_mask(circ, off=(-L[0] * rr * 0.3, -L[1] * rr * 0.3))
        cres = np.clip(m0 - shf, 0, 1)
        cv.put(bb, cres, lit, 0.95)
        shf2 = disc_mask(circ, off=(-L[0] * rr * 0.1, -L[1] * rr * 0.1))
        cv.put(bb, np.clip(m0 - shf2, 0, 1), hot, 0.85)
        # lit leaf clusters: small scalloped bumps along the sun side, following the outline
        sp = []
        for _ in range(int(9 + 7 * rng.random())):
            a_ = rng.uniform(-2.6, -0.3)
            d_ = rr * rng.uniform(0.55, 0.85)
            sp.append((cx + math.cos(a_) * d_, cy + math.sin(a_) * d_ * 0.78, rr * rng.uniform(0.1, 0.17)))
        ms = disc_mask(sp) * m0
        msh = disc_mask(sp, off=(-L[0] * rr * 0.05, -L[1] * rr * 0.05))
        cv.put(bb, ms, lit, 0.85)
        cv.put(bb, np.clip(ms - msh, 0, 1), hot, 0.9)


_FONTS = [r'C:\Windows\Fonts\YuGothB.ttc', r'C:\Windows\Fonts\msgothic.ttc']


def text_mask(text, h_px):
    f = None
    for p_ in _FONTS:
        try:
            f = ImageFont.truetype(p_, max(int(h_px * 4), 8))
            break
        except OSError:
            f = None
    if f is None:
        return None
    bb = f.getbbox(text)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    im = Image.new('L', (w + 8, h + 8), 0)
    ImageDraw.Draw(im).text((4 - bb[0], 4 - bb[1]), text, fill=255, font=f)
    m = np.asarray(im, np.float32) / 255.0
    return cv2.resize(m, (max(int(m.shape[1] / 4), 1), max(int(m.shape[0] / 4), 1)), interpolation=cv2.INTER_AREA)


def rooftop_sign(cv, cx, y_roof, w, h, s, hk, text):
    """Rooftop billboard on a steel frame: white panel, blue lettering + band, lit top edge."""
    y1 = y_roof - 0.35 * h
    y0 = y1 - h
    fr = hz(_h('#3a4460'), hk)
    legs = [[(cx - w * 0.4, y1), (cx - w * 0.4, y_roof)], [(cx + w * 0.4, y1), (cx + w * 0.4, y_roof)],
            [(cx - w * 0.4, y_roof), (cx + w * 0.4, y1)]]
    cv.lines(legs, max(0.8, 1.2 * s), fr)
    cv.poly([(cx - w / 2, y0), (cx + w / 2, y0), (cx + w / 2, y1), (cx - w / 2, y1)],
            lambda gx, gy: hz(_h('#eef2f8'), hk * 0.8) * (0.9 + 0.1 * np.clip((gx - cx + w / 2) / w, 0, 1))[..., None])
    cv.poly([(cx - w / 2, y1 - h * 0.18), (cx + w / 2, y1 - h * 0.18), (cx + w / 2, y1), (cx - w / 2, y1)],
            hz(_h('#2c62c8'), hk * 0.7))
    m = text_mask(text, h * 0.55)
    if m is not None:
        th, tw = m.shape
        sc = min((w * 0.86) / tw, (h * 0.62) / th)
        m = cv2.resize(m, (max(int(tw * sc), 1), max(int(th * sc), 1)), interpolation=cv2.INTER_AREA)
        th, tw = m.shape
        x0 = int(cx - tw / 2)
        y0i = int(y0 + h * 0.08) - cv.y0
        if 0 <= x0 and x0 + tw <= cv.W and 0 <= y0i and y0i + th <= cv.H:
            cv.put((x0, y0i, x0 + tw, y0i + th), m, hz(_h('#1d3f9a'), hk * 0.7))
    cv.lines([[(cx - w / 2, y0), (cx + w / 2, y0)]], max(1.0, 1.4 * s), hz(RIM, 0.2))


# =============================================================================================== far fill
def hills(cv, ox, W, H, y_base, rng):
    """Two layers of haze-blue distant hills along the horizon (very low contrast, lit ridge line)."""
    xs = np.linspace(ox - 0.05 * W, ox + 1.1 * W, 400)
    for k, (amp, col, hk2, off) in enumerate([(0.05, _h('#a6c6ea'), 0.0, -0.012), (0.03, _h('#8fb2dc'), 0.0, 0.0)]):
        n = np.zeros_like(xs)
        for f, a in ((1.3, 1.0), (3.1, 0.45), (7.3, 0.18), (15.0, 0.07)):
            ph = rng.uniform(0, 6.28)
            n += a * np.sin(xs / W * f * 6.28 + ph)
        n = (n - n.min()) / (n.max() - n.min() + 1e-6)
        ys = y_base + off * H - amp * H * n
        poly = np.concatenate([np.stack([xs, ys], 1), [[xs[-1], y_base + 0.08 * H], [xs[0], y_base + 0.08 * H]]])
        top_ = ys.min()

        def hc(gx, gy, col=col, top_=top_):
            v = np.clip((gy - top_) / (0.07 * H), 0, 1)
            return col * (1.02 - 0.06 * v)[..., None] + HAZE * (0.25 * v)[..., None]
        cv.poly(poly, hc)
        cv.lines([np.stack([xs, ys - 0.6], 1)], max(0.6, 0.0006 * W), _h('#d8ecfa'), 0.5)


def water_tank(cv, tx, ty, r, s, hk, legs=True):
    """Elevated cylindrical rooftop water tank (lit left, sky rim right) on a steel stand."""
    h = r * 1.3
    if legs:
        cv.lines([[(tx - r * 0.8, ty), (tx - r * 0.8, ty + r * 0.9)], [(tx + r * 0.8, ty), (tx + r * 0.8, ty + r * 0.9)],
                  [(tx - r * 0.8, ty + r * 0.9), (tx + r * 0.8, ty)], [(tx + r * 0.8, ty + r * 0.9), (tx - r * 0.8, ty)]],
                 max(0.6, 0.9 * s), hz(_h('#4a5470'), hk))

    def tc(gx, gy):
        f = np.clip((gx - (tx - r)) / (2 * r), 0, 1)
        nx = f * 2 - 1
        key = np.clip(-nx * 0.7 + 0.45, 0, 1) ** 1.4
        c = _h('#6a7490') + (_h('#e2e6ee') - _h('#6a7490')) * key[..., None]
        c = c + (_h('#a8c4ea') - c) * (_ss(0.8, 0.98, f) * 0.5)[..., None]
        return hz(c, hk)
    cv.poly([(tx - r, ty - h), (tx + r, ty - h), (tx + r, ty), (tx - r, ty)], tc)
    cv.ellipse(tx, ty - h, r, r * 0.28, hz(_h('#c8ccd8'), hk))
    cv.ellipse(tx - r * 0.25, ty - h - r * 0.04, r * 0.6, r * 0.12, hz(RIM, hk * 0.5), 0.8)
    for j in (0.3, 0.65):
        cv.lines([[(tx - r, ty - h * j), (tx + r, ty - h * j)]], max(0.5, 0.5 * s), hz(_h('#58627e'), hk), 0.6)
    GLINTS.append((tx - r * 0.45, ty - h * 0.8, 0.8, (tx * 0.11) % 6.28))


def apartment(cv, x0, x1, by, base, s, hk, rng, fh, tank=True, pal=None):
    """Mid-rise apartment block: flat roof with parapet, floor bands with balconies, window grid."""
    wallc = hz(pal if pal is not None else _h('#8c96b0'), hk)
    cv.poly([(x0, by), (x1, by), (x1, base), (x0, base)],
            lambda gx, gy: wallc * (1.02 - 0.1 * np.clip((gx - x0) / (x1 - x0), 0, 1))[..., None])
    ncol = max(int((x1 - x0) / (fh * 1.25)), 2)
    nfl = int((base - by) / fh) + 1
    for k in range(nfl):
        yy = by + 0.35 * fh + k * fh
        for j in range(ncol):
            wx = x0 + (j + 0.5) * (x1 - x0) / ncol
            window(cv, wx, yy, (x1 - x0) / ncol * 0.62, fh * 0.5, s, hk, rng)
        cv.poly([(x0, yy + fh * 0.58), (x1, yy + fh * 0.58), (x1, yy + fh * 0.7), (x0, yy + fh * 0.7)],
                hz(_h('#b4bccc'), hk))
        cv.lines([[(x0, yy + fh * 0.58), (x1, yy + fh * 0.58)]], max(0.6, 0.6 * s), hz(RIM, hk * 0.6), 0.6)
    cv.lines([[(x0, by), (x1, by)]], max(1.0, 1.3 * s), hz(RIM, hk * 0.4))
    cv.lines([[(x0 + 1.5 * s, by), (x0 + 1.5 * s, base)]], max(0.6, 0.8 * s), hz(RIM, hk * 0.6), 0.5)
    if tank:
        water_tank(cv, x0 + (x1 - x0) * rng.uniform(0.6, 0.8), by - fh * 0.35, fh * 0.45, s, hk)
        # stair / elevator housing
        sx = x0 + (x1 - x0) * rng.uniform(0.1, 0.3)
        cv.poly([(sx - fh * 0.5, by), (sx + fh * 0.5, by), (sx + fh * 0.5, by - fh * 0.9), (sx - fh * 0.5, by - fh * 0.9)],
                hz(_h('#a0a8bc'), hk))
        cv.lines([[(sx - fh * 0.5, by - fh * 0.9), (sx + fh * 0.5, by - fh * 0.9)]], max(0.8, 0.9 * s), hz(RIM, hk * 0.4))


def far_poles(cv, ox, W, H, y_ground, s, hk, rng, n=6):
    """A receding line of small utility poles with sagging wires across the far rooftops."""
    xs = np.sort(ox + rng.uniform(-0.05, 1.05, n) * W)
    tops = []
    for x in xs:
        h = H * rng.uniform(0.06, 0.08)
        top = y_ground - h
        col = hz(_h('#3a4460'), hk)
        cv.lines([[(x, y_ground), (x, top)]], max(0.9, 1.3 * s), col)
        cv.lines([[(x - 0.4 * s, y_ground), (x - 0.4 * s, top)]], max(0.5, 0.4 * s), hz(RIM, hk * 0.6), 0.6)
        cv.lines([[(x - 0.009 * W, top + 0.004 * H), (x + 0.009 * W, top + 0.004 * H)]], max(0.8, 1.0 * s), col)
        tops.append((x, top + 0.004 * H))
    for a, b in zip(tops[:-1], tops[1:]):
        for dy in (0.0, 0.006 * H):
            t = np.linspace(0, 1, 60)
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + dy + (b[1] - a[1]) * t + 0.012 * H * 4 * t * (1 - t)
            cv.lines([np.stack([x, y], 1)], max(0.5, 0.6 * s), hz(_h('#2a3450'), hk), 0.8)


# =============================================================================================== layers
# screen-x centres / half widths (fractions of W) of the far rooftop signs
SIGN_BANDS = [(0.44 + 0.38 * 0.22, 0.05), (0.905, 0.045), (0.07, 0.065)]


def town_near(W, H, pw, ph, y_top0, seed=11):
    rng = np.random.default_rng(seed)
    y0 = int(y_top0 + 0.5 * H)
    cv = Canvas(pw, ph - y0, y0)
    ox = (pw - W) / 2
    base = y_top0 + H * 1.3
    s = W / 1920
    rows = [(0.822, 0.13, 0.3, 4.2), (0.87, 0.18, 0.14, 5.6), (0.95, 0.27, 0.0, 9.0)]
    import s01_summer_sky_roofdetail as RD
    drng = np.random.default_rng(seed + 100)
    for ri, (fy, fw, hk, cs) in enumerate(rows):
        x = ox - rng.uniform(0.02, 0.1) * W
        items = []
        while x < ox + 1.1 * W:
            w = fw * W * rng.uniform(0.8, 1.15)
            ye = y_top0 + (fy + rng.uniform(-0.012, 0.012)) * H
            items.append((x, ye, w))
            x += w * rng.uniform(0.98, 1.12)
        # trees behind this row
        for k in range(2 if ri < 2 else 1):
            tx = ox + rng.uniform(0.05, 0.95) * W
            tree(cv, tx, y_top0 + (fy - 0.005) * H, fw * W * rng.uniform(0.3, 0.42), s, rng, hk + 0.05)
        for (x, ye, w) in items:
            kind = ['hip', 'gable', 'gable_end', 'hip', 'gable'][int(rng.integers(0, 5))]
            if ri == 2 and kind == 'gable':
                kind = 'hip'          # nearest row: hipped roofs show a lit and a shadowed plane
            yr = house(cv, x, ye, w, s, rng, hk, kind, base - ye, cs * s, detail=1.0)
            RD.roof_clutter(cv, x, w, yr, ye, kind, s, hk, drng, big=1.0 if ri > 0 else 0.7)
            if ri == 2 and drng.random() < 0.85:
                ax2 = x + drng.uniform(-0.25, 0.25) * w
                if not any(abs(ax2 - (ox + sx_ * W)) < hw_ * W for sx_, hw_ in SIGN_BANDS):
                    antenna(cv, ax2, yr + 0.5 * s, H * drng.uniform(0.07, 0.1), s * 3.0, drng, HAZE, hk)
            if ri < 2 and rng.random() < 0.2:
                solar_heater(cv, x + rng.uniform(-0.2, 0.2) * w, yr + 0.004 * H, w * 0.28, s, HAZE, hk)
            if ri < 2 and rng.random() < 0.55:
                arng = np.random.default_rng(int(rng.integers(1 << 30)))
                ax_ = x + arng.uniform(-0.25, 0.25) * w
                # keep the far rooftop signs readable: no antenna mast in front of them
                if not any(abs(ax_ - (ox + sx_ * W)) < hw_ * W for sx_, hw_ in SIGN_BANDS):
                    antenna(cv, ax_, yr + 0.5 * s, H * arng.uniform(0.05, 0.08) * (0.7 + 0.2 * ri),
                            s * 2.4, arng, HAZE, hk)
    return cv.rgba(), y0


def town_far(W, H, pw, ph, y_top0, seed=5):
    rng = np.random.default_rng(seed)
    y0 = int(y_top0 + 0.5 * H)
    cv = Canvas(pw, ph - y0, y0)
    s = W / 1920
    ox = (pw - W) / 2
    base = y_top0 + 1.2 * H
    hk = 0.5
    import s01_summer_sky_roofdetail as RD
    drng = np.random.default_rng(seed + 200)
    hills(cv, ox, W, H, y_top0 + 0.8 * H, rng)
    # more distant apartment blocks + water tanks filling the upper-middle distance
    for (a0, a1, top_, pal, hk2) in [(0.0, 0.13, 0.735, _h('#9aa2b8'), 0.62), (0.17, 0.27, 0.768, _h('#a4a0b4'), 0.58),
                                     (0.71, 0.8, 0.772, _h('#96a0b6'), 0.6), (0.82, 0.99, 0.742, _h('#8c98b2'), 0.64),
                                     (0.33, 0.4, 0.778, _h('#aab0c2'), 0.62)]:
        apartment(cv, ox + a0 * W, ox + a1 * W, y_top0 + top_ * H, base, s, hk2, rng, 0.03 * H, tank=True, pal=pal)
        fh_ = 0.03 * H
        RD.facade_life(cv, ox + a0 * W, ox + a1 * W, y_top0 + top_ * H, base, s, hk2, drng, fh_,
                       max(int((a1 - a0) * W / (fh_ * 1.25)), 2))
        RD.flat_roof_kit(cv, ox + (a0 + 0.01) * W, ox + (a0 + (a1 - a0) * 0.5) * W, y_top0 + top_ * H, s, hk2, drng, fh_)
    # distant tree line (very hazy) along the horizon
    for k in range(10):
        tree(cv, ox + rng.uniform(-0.02, 1.02) * W, y_top0 + H * rng.uniform(0.79, 0.8), W * rng.uniform(0.025, 0.045),
             s, rng, 0.62)
    # mid-rise apartment block (flat roof, balconies, water tank, stair tower)
    bx0, bx1 = ox + 0.44 * W, ox + 0.66 * W
    by = y_top0 + 0.762 * H
    wallc = hz(_h('#8c96b0'), hk)
    cv.poly([(bx0, by), (bx1, by), (bx1, base), (bx0, base)],
            lambda gx, gy: wallc * (0.92 + 0.08 * np.clip((gx - bx0) / (bx1 - bx0), 0, 1))[..., None])
    fh = 0.042 * H
    for k in range(6):
        yy = by + 0.018 * H + k * fh
        for j in range(8):
            wx = bx0 + (j + 0.5) * (bx1 - bx0) / 8
            window(cv, wx, yy + fh * 0.2, (bx1 - bx0) / 8 * 0.7, fh * 0.55, s, hk, rng)
        cv.poly([(bx0, yy + fh * 0.72), (bx1, yy + fh * 0.72), (bx1, yy + fh * 0.95), (bx0, yy + fh * 0.95)],
                hz(_h('#a8b0c4'), hk))
        cv.lines([[(bx0, yy + fh * 0.72), (bx1, yy + fh * 0.72)]], max(0.7, 0.8 * s), hz(RIM, hk * 0.6), 0.7)
    cv.lines([[(bx0, by), (bx1, by)]], max(1.2, 1.5 * s), hz(RIM, 0.25))
    rail = [[(bx0 + (bx1 - bx0) * j / 39, by), (bx0 + (bx1 - bx0) * j / 39, by - 0.012 * H)] for j in range(40)]
    cv.lines(rail, max(0.6, 0.4 * s), hz(_h('#5a6680'), hk))
    cv.lines([[(bx0, by - 0.012 * H), (bx1, by - 0.012 * H)]], max(0.8, 0.6 * s), hz(RIM, 0.3))
    tx = bx0 + 0.72 * (bx1 - bx0)
    cv.poly([(tx - 0.03 * W, by - 0.03 * H), (tx + 0.03 * W, by - 0.03 * H), (tx + 0.03 * W, by - 0.065 * H),
             (tx - 0.03 * W, by - 0.065 * H)],
            lambda gx, gy: hz(_h('#c4c8d4') * (0.75 + 0.3 * np.exp(-((gx - tx + 0.018 * W) / (0.012 * W)) ** 2))[..., None], hk))
    cv.lines([[(tx - 0.03 * W, by - 0.065 * H), (tx + 0.03 * W, by - 0.065 * H)]], max(1, 1.2 * s), hz(RIM, 0.2))
    cv.lines([[(tx - 0.025 * W, by - 0.03 * H), (tx - 0.025 * W, by)], [(tx + 0.025 * W, by - 0.03 * H), (tx + 0.025 * W, by)],
              [(tx - 0.025 * W, by), (tx + 0.025 * W, by - 0.03 * H)]], max(0.8, 0.8 * s), hz(_h('#5a6680'), hk))
    for j in range(1, 6):
        xx = tx - 0.03 * W + 0.06 * W * j / 6
        cv.lines([[(xx, by - 0.065 * H), (xx, by - 0.03 * H)]], max(0.6, 0.7 * s), hz(_h('#8a92a8'), hk), 0.7)
    for yy in (by - 0.053 * H, by - 0.042 * H):
        cv.lines([[(tx - 0.03 * W, yy), (tx + 0.03 * W, yy)]], max(0.6, 0.7 * s), hz(_h('#8a92a8'), hk), 0.7)
    rooftop_sign(cv, bx0 + 0.38 * (bx1 - bx0), by, 0.065 * W, 0.03 * H, s, hk, 'コーポ青空')
    RD.flat_roof_kit(cv, bx0 + 0.52 * (bx1 - bx0), bx0 + 0.62 * (bx1 - bx0), by, s, hk, drng, fh)
    rooftop_sign(cv, ox + 0.905 * W, y_top0 + 0.742 * H, 0.06 * W, 0.026 * H, s, 0.6, 'さくら歯科')
    rooftop_sign(cv, ox + 0.07 * W, y_top0 + 0.735 * H, 0.095 * W, 0.028 * H, s, 0.62, '田中クリーニング')
    sx = bx0 + 0.08 * (bx1 - bx0)
    cv.poly([(sx - 0.025 * W, by), (sx + 0.025 * W, by), (sx + 0.025 * W, by - 0.05 * H), (sx - 0.025 * W, by - 0.05 * H)],
            hz(_h('#9aa2b8'), hk))
    cv.lines([[(sx - 0.025 * W, by - 0.05 * H), (sx + 0.025 * W, by - 0.05 * H)]], max(1, 1.2 * s), hz(RIM, 0.2))
    far_poles(cv, ox, W, H, y_top0 + 0.8 * H, s, 0.45, rng, n=7)
    # hazier rooftops row
    x = ox - 0.05 * W
    while x < ox + 1.08 * W:
        w = W * rng.uniform(0.09, 0.14)
        if bx0 - 0.03 * W < x < bx1 + 0.03 * W:
            x += w * 0.8
            continue
        ye = y_top0 + H * rng.uniform(0.792, 0.805)
        yr = house(cv, x, ye, w, s, rng, hk, ['hip', 'gable', 'gable_end'][int(rng.integers(0, 3))], base - ye,
                   3.0 * s, detail=0.3)
        if rng.random() < 0.25:
            antenna(cv, x + rng.uniform(-0.3, 0.3) * w, yr, 0.045 * H * rng.uniform(0.7, 1.2), s * 2, rng, HAZE, hk)
        x += w * rng.uniform(0.88, 1.04)
    return cv.rgba(), y0


# =============================================================================================== pole
def pole_and_wires(W, H, pw, ph, y_top0, T, seed=7, sun_x=None):
    """Concrete utility pole (right) with crossarms, porcelain insulators, a pole transformer with bolted
    lid and bushings; a few thin catenary power lines with a sun glint along each."""
    rng = np.random.default_rng(seed)
    cv = Canvas(pw, ph, 0)
    ox = (pw - W) / 2
    sun_x = ox + 0.47 * W if sun_x is None else sun_x
    s = W / 1920
    px = ox + 0.845 * W
    top = y_top0 - T + 0.72 * H
    bottom = ph
    wt, wb = 0.0105 * W, 0.0145 * W
    dark = _h('#1a2034')
    sky_rim = _h('#8fb2e6')

    def pole_col(gx, gy):
        half = wt + (wb - wt) * np.clip((gy - top) / max(bottom - top, 1), 0, 1)
        f = np.clip((gx - (px - half)) / (2 * half), 0, 1)
        nx = f * 2 - 1                                   # cylinder normal x
        nz = np.sqrt(np.clip(1 - nx * nx, 0, 1))
        # sun is up-left: warm key on the left flank, cool sky fill on the right, dark core
        key = np.clip(-nx * 0.8 + nz * 0.25, 0, 1) ** 2
        c = dark + (_h('#56607a') - dark) * key[..., None]
        c = c + (sky_rim - c) * (_ss(0.75, 0.98, f) * 0.45)[..., None]
        rim = np.exp(-((f - 0.06) / 0.05) ** 2)
        c = c + (RIM - c) * (rim * 0.95)[..., None]
        # concrete texture: faint vertical streaks + horizontal casting marks
        c = c * (1 + 0.04 * np.sin(gx * 0.9) * np.sin(gy * 0.013))[..., None]
        return c
    cv.poly([(px - wt, top), (px + wt, top), (px + wb, bottom), (px - wb, bottom)], pole_col)
    cv.ellipse(px, top, wt, wt * 0.32, _h('#c8c0b0'))
    cv.ellipse(px - wt * 0.2, top - wt * 0.05, wt * 0.7, wt * 0.18, RIM, 0.8)
    # step bolts
    for k in range(18):
        yy = top + 0.1 * H + k * 0.035 * H
        sgn = 1 if k % 2 else -1
        half = wt + (wb - wt) * (yy - top) / max(bottom - top, 1)
        p0, p1 = (px + sgn * half * 0.8, yy), (px + sgn * (half + 0.011 * W), yy - 0.003 * H)
        cv.lines([[p0, p1]], max(1.2, 2.4 * s), dark)
        cv.lines([[(p0[0], p0[1] - 1.0 * s), (p1[0], p1[1] - 1.0 * s)]], max(0.6, 0.8 * s), RIM, 0.75)
    # number plate
    py_ = top + 0.52 * H
    cv.poly([(px - wt * 0.7, py_), (px + wt * 0.5, py_), (px + wt * 0.5, py_ + 0.05 * H), (px - wt * 0.7, py_ + 0.05 * H)],
            _h('#9aa6c0'))
    for j in range(3):
        cv.lines([[(px - wt * 0.5, py_ + (0.012 + 0.012 * j) * H), (px + wt * 0.3, py_ + (0.012 + 0.012 * j) * H)]],
                 max(0.8, 1.2 * s), _h('#2a3a70'), 0.8)
    wires = []
    arms = [(0.018, 0.11, 3), (0.06, 0.09, 3)]
    ins = []
    for (dy, half, n) in arms:
        ay = top + dy * H
        x0a, x1a = px - half * W, px + half * W * 0.7
        cv.poly([(x0a, ay - 0.003 * H), (x1a, ay - 0.003 * H), (x1a, ay + 0.004 * H), (x0a, ay + 0.004 * H)],
                lambda gx, gy, ay=ay: dark + (_h('#4a5470') - dark) * _ss(ay + 0.004 * H, ay - 0.003 * H, gy)[..., None] * 0.6)
        cv.lines([[(x0a, ay - 0.003 * H), (x1a, ay - 0.003 * H)]], max(1, 1.4 * s), RIM, 0.9)
        cv.lines([[(px - half * W * 0.6, ay + 0.004 * H), (px - wt, ay + 0.03 * H)],
                  [(px + half * W * 0.45, ay + 0.004 * H), (px + wt, ay + 0.03 * H)]], max(1, 2.0 * s), dark)
        for i in range(n):
            ix = x0a + (x1a - x0a) * (i + 0.5) / n
            for j in range(3):
                iy = ay - 0.006 * H - j * 0.0055 * H
                rr = 0.0055 * W * (1 - 0.12 * j)
                cv.ellipse(ix, iy, rr, 0.0025 * H, _h('#6a7898'))
                cv.ellipse(ix - rr * 0.25, iy - 0.001 * H, rr * 0.6, 0.001 * H, _h('#f4f0ea'), 0.9)
            ins.append((ix, ay - 0.021 * H))
    # pole transformer (single can, bolted lid, bushings) on a bracket
    ty = top + 0.17 * H
    tx = px - 0.034 * W
    tw, th = 0.017 * W, 0.068 * H

    def tcol(gx, gy):
        f = np.clip((gx - (tx - tw)) / (2 * tw), 0, 1)
        nx = f * 2 - 1
        nz = np.sqrt(np.clip(1 - nx * nx, 0, 1))
        key = np.clip(-nx * 0.7 + nz * 0.35, 0, 1) ** 1.5
        c = _h('#2a3248') + (_h('#8a94ac') - _h('#2a3248')) * key[..., None]
        c = c + (sky_rim - c) * (_ss(0.8, 0.98, f) * 0.4)[..., None]
        c = c + (RIM - c) * (np.exp(-((f - 0.05) / 0.04) ** 2) * 0.9)[..., None]
        band = np.exp(-((gy - (ty + th * 0.2)) / (0.002 * H)) ** 2) + np.exp(-((gy - (ty + th * 0.8)) / (0.002 * H)) ** 2)
        c = c * (1 - 0.3 * band)[..., None]
        return c
    cv.poly([(tx - tw, ty), (tx + tw, ty), (tx + tw, ty + th), (tx - tw, ty + th)], tcol)
    cv.ellipse(tx, ty + th, tw, tw * 0.3, _h('#1e2436'))
    # lid: slightly wider rim with bolts, lit top
    cv.ellipse(tx, ty, tw * 1.08, tw * 0.3, _h('#3a4460'))
    cv.ellipse(tx - tw * 0.1, ty - tw * 0.05, tw * 0.95, tw * 0.22, _h('#b4bccc'))
    cv.ellipse(tx - tw * 0.3, ty - tw * 0.08, tw * 0.5, tw * 0.1, RIM, 0.8)
    for b in range(7):
        a = math.pi * (0.1 + 0.8 * b / 6)
        cv.ellipse(tx - math.cos(a) * tw * 1.0, ty + math.sin(a) * tw * 0.28, 1.2 * s + 0.4, 1.0 * s + 0.4, _h('#e8e4dc'), 0.9)
    # bushings (small insulator stacks) on the lid + drop leads
    for bxo in (-0.45, 0.0, 0.45):
        bx = tx + bxo * tw
        for j in range(3):
            cv.ellipse(bx, ty - 0.004 * H - j * 0.0035 * H, 0.0028 * W, 0.0012 * H, _h('#7a86a4'))
            cv.ellipse(bx - 0.0008 * W, ty - 0.0045 * H - j * 0.0035 * H, 0.0016 * W, 0.0005 * H, _h('#f2eee6'), 0.9)
        wires.append((np.array([(bx, ty - 0.014 * H), (bx - 0.004 * W, top + 0.08 * H), (px - 0.05 * W, top + 0.064 * H)]),
                      1.0 * s))
    # bracket bands
    for yy in (ty + th * 0.25, ty + th * 0.75):
        cv.lines([[(tx + tw, yy), (px - wt, yy)]], max(1.2, 2.2 * s), dark)
        cv.lines([[(tx + tw, yy - 1.2 * s), (px - wt, yy - 1.2 * s)]], max(0.5, 0.6 * s), RIM, 0.6)
    # far pole (receding to the left, smaller + hazier)
    fx, ftop = ox + 0.2 * W, top + 0.3 * H
    fwt = wt * 0.4
    fcol = hz(_h('#2c3650'), 0.35)
    cv.poly([(fx - fwt, ftop), (fx + fwt, ftop), (fx + fwt * 1.3, ph), (fx - fwt * 1.3, ph)],
            lambda gx, gy: fcol + (hz(RIM, 0.3) - fcol) * np.exp(-((gx - fx + fwt * 0.7) / (fwt * 0.3)) ** 2)[..., None] * 0.8)
    far_ins = []
    for (dy, half, n) in arms:
        ay = ftop + dy * 0.4 * H
        cv.lines([[(fx - half * 0.4 * W, ay), (fx + half * 0.28 * W, ay)]], max(1, 2.2 * s), fcol)
        for i in range(n):
            far_ins.append((fx - half * 0.4 * W + half * 0.68 * W * (i + 0.5) / n, ay - 0.008 * H))
    # spans: to the far pole (these cross in front of the tower -> keep few), one pair off to the left
    # at a steep sag, two to the right
    spans = []
    # every line is anchored insulator-to-insulator, grouped by crossarm level: upper arm -> far pole's
    # upper arm, lower arm -> far pole's lower arm, then each group continues off the left edge at its
    # own level (a gentle, consistent sag); the upper arm also feeds a span off to the right
    for a, b in zip(ins[:3], far_ins[:3]):
        spans.append((a, b, 0.05 * H * rng.uniform(0.9, 1.1), 1.9 * s))
    for a, b in zip(ins[3:6], far_ins[3:6]):
        spans.append((a, b, 0.055 * H * rng.uniform(0.9, 1.1), 1.9 * s))
    for i, b in enumerate(far_ins[:6]):
        lvl = 0 if i < 3 else 1
        spans.append((b, (b[0] - 0.55 * W, b[1] + (0.05 + 0.02 * lvl) * H + (i % 3) * 0.004 * H), 0.03 * H, 1.5 * s))
    for i, a in enumerate(ins[:3]):
        spans.append((a, (a[0] + 0.5 * W, a[1] - 0.05 * H + i * 0.01 * H), 0.05 * H, 2.2 * s))
    # communication cable on its own low bracket: pole -> far pole -> off left, same level
    cy = top + 0.33 * H
    fcy = ftop + 0.2 * H
    cv.lines([[(px - wb * 1.1, cy), (px + wb * 1.1, cy)]], max(1.2, 3.0 * s), dark)
    cv.lines([[(fx - fwt * 2.5, fcy), (fx + fwt * 2.5, fcy)]], max(1.0, 2.0 * s), fcol)
    spans.append(((px - wb, cy), (fx, fcy), 0.07 * H, 3.6 * s))
    spans.append(((fx, fcy), (fx - 0.6 * W, fcy + 0.06 * H), 0.04 * H, 3.0 * s))
    for (a, b, sag, wd) in spans:
        t = np.linspace(0, 1, 200)
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t + sag * 4 * t * (1 - t)
        wires.append((np.stack([x, y], 1), wd))
    for pts, wd in wires:
        cv.lines([pts], max(0.8, wd), _h('#161c2c'))

        def spec(gx, gy, sx=sun_x):
            k = 0.25 + 1.1 * np.exp(-((gx - sx) / (0.28 * W)) ** 2)
            return _c(1.35, 1.25, 1.05) * k[..., None]
        cv.lines([pts - [0, wd * 0.35]], max(0.45, wd * 0.35), spec, 0.85)
    return cv.rgba(), wires
