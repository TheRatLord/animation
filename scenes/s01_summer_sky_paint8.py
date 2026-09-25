"""s01 helper (round 14): the hero cumulonimbus painted as NESTED FRACTAL MASSES.

Round 13 (paint7) read as a rectangular block of same-size cotton balls with a stamped bright cap /
lilac belly each, floating mid-sky. This painter follows wwy_01 / cm5_05 / gow_02:

  * SHAPE: a tapering, right-leaning column that rises from a broad hazy foot on the horizon into a
    broad flat ANVIL spreading downwind (right), ~3x the width of the neck, with an overshooting dome
    near the up-sun end. Primary masses are few and big, and their sizes vary a lot.
  * NESTED DETAIL: each primary mass = core ellipse + level-1 heads on its upper rim + level-2 heads on
    the outward rims of those + fine cauliflower buds on the final silhouette (lit side only). Detail
    is shown only in light; shade is a broad continuous plane.
  * VALUE: a big-form field (sun upper-left: left / upper-left faces warm white, a continuous saturated
    blue-grey shadow down the right side and through the lower third) + each mass's own big lambert +
    bud lambert only where the mass is lit. Undersides of masses are LOST (fade into the body).
  * EDGES: crisp on the lit silhouette, soft/lost where the shade side meets the sky; no dark contour.
  * FOOT: lower tower aerially faded, flat darker base dissolving into horizon haze / rain-foot.
Deterministic and resolution independent (sizes in units of the tower height Hc).
"""
import math

import numpy as np
import cv2

from lib import clouds3 as K3

F32 = np.float32


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise(w, h, px, seed, octaves=3, stretch=1.0, angle=0.0):
    return K3._noise(w, h, max(w / max(px, 1.0), 2.0), seed, octaves, stretch=stretch, angle=angle).astype(F32)


def _c(v):
    return np.asarray(v, F32)


LEAN = 0.1
_CFG = dict(lean=LEAN, hwk=1.0, hw=None)
HW0 = [0.4, 0.36, 0.29, 0.23, 0.2, 0.2, 0.2]
# hero (round 15): a broad-footed tower that tapers steadily into a domed crown (no top-heavy hammer)
HW_HERO = [0.6, 0.54, 0.44, 0.35, 0.28, 0.25, 0.22]


def axis_x(s):
    ln = _CFG['lean']
    return ln * s + 0.5 * ln * s * s


def half_w(s):
    return _CFG['hwk'] * np.interp(s, [0.0, 0.1, 0.3, 0.55, 0.72, 0.8, 1.0], _CFG['hw'] or HW0)


# primary masses: (x rel. column axis in Hc, s = centre height, rx, ry, kind). Painted in list order
# (first = behind). Upper masses sit behind lower ones so each lit crown shows against the shaded belly
# of the mass above it.
MASSES = [
    # anvil + overshooting dome are painted separately (see _anvil)
    (-0.1, 0.855, 0.2, 0.05, 'flare'),
    (0.2, 0.86, 0.26, 0.045, 'flare'),
    (-0.05, 0.79, 0.16, 0.08, 'neck'),
    (0.15, 0.76, 0.13, 0.09, 'neck'),
    (-0.14, 0.68, 0.12, 0.09, 'body'),
    (0.1, 0.64, 0.15, 0.12, 'body'),
    (-0.2, 0.58, 0.09, 0.07, 'body'),
    (-0.05, 0.54, 0.14, 0.11, 'body'),
    (0.21, 0.5, 0.13, 0.13, 'shade'),
    (-0.15, 0.43, 0.13, 0.11, 'body'),
    (0.05, 0.37, 0.19, 0.15, 'body'),
    (0.25, 0.29, 0.14, 0.13, 'shade'),
    (-0.19, 0.27, 0.14, 0.12, 'body'),
    (0.02, 0.19, 0.22, 0.15, 'body'),
    (-0.24, 0.1, 0.14, 0.08, 'skirt'),
    (0.26, 0.11, 0.17, 0.09, 'shade'),
    (-0.02, 0.07, 0.26, 0.1, 'skirt'),
]


# hero (round 15): FOUR big stacked round lobe masses (crown / upper / middle / lower) on a wide low
# skirt, plus two broad shade-side masses down the right. Each lobe is a large value mass with a lit
# upper-left crest and a smooth blue-grey underside; small cauliflower only along the lit silhouette.
HERO_MASSES = [
    # broad flat foot (the column rises out of a wide plateau) + irregular side heaps
    (0.14, 0.07, 0.68, 0.095, 'skirt'),
    (-0.44, 0.1, 0.16, 0.065, 'skirt'),
    (0.64, 0.09, 0.2, 0.065, 'shade'),
    # shade-side masses down the right
    (0.22, 0.73, 0.12, 0.1, 'shade'),
    (0.31, 0.5, 0.15, 0.15, 'shade'),
    (0.4, 0.26, 0.21, 0.15, 'shade'),
    # big lit value masses (crown / upper / middle / lower pair)
    (-0.02, 0.83, 0.16, 0.12, 'body'),
    (-0.12, 0.67, 0.18, 0.14, 'body'),
    (0.07, 0.6, 0.13, 0.1, 'body'),
    (-0.02, 0.44, 0.26, 0.18, 'body'),
    (0.14, 0.2, 0.29, 0.15, 'body'),
    (-0.25, 0.24, 0.29, 0.16, 'body'),
    # mid-size lobes on the fronts of the big masses (lobes on lobes, several scales)
    (-0.21, 0.75, 0.07, 0.055, 'lobe'),
    (-0.16, 0.53, 0.1, 0.075, 'lobe'),
    (0.07, 0.5, 0.09, 0.065, 'lobe'),
    (-0.3, 0.38, 0.09, 0.07, 'lobe'),
    (-0.4, 0.29, 0.11, 0.08, 'lobe'),
    (-0.19, 0.33, 0.12, 0.08, 'lobe'),
    (-0.31, 0.16, 0.13, 0.07, 'lobe'),
    (0.05, 0.27, 0.11, 0.075, 'lobe'),
    (-0.06, 0.16, 0.09, 0.06, 'lobe'),
]


def _dome_field(ww, hh, cell, rr, seed, keep=0.85):
    """Height field of random round puffs on a jittered grid (max of hemispheres), px units."""
    rng = np.random.default_rng(seed)
    Z = np.zeros((hh, ww), F32)
    nx, ny = int(ww / cell) + 2, int(hh / cell) + 2
    for j in range(ny):
        for i in range(nx):
            if rng.random() > keep:
                continue
            x = (i + rng.random()) * cell
            y = (j + rng.random()) * cell
            r = cell * rng.uniform(*rr)
            x0, x1 = int(max(x - r, 0)), int(min(x + r + 1, ww))
            y0, y1 = int(max(y - r, 0)), int(min(y + r + 1, hh))
            if x1 <= x0 or y1 <= y0:
                continue
            yy = np.arange(y0, y1, dtype=F32)[:, None] - y
            xx = np.arange(x0, x1, dtype=F32)[None, :] - x
            z = np.sqrt(np.clip(1 - (xx * xx + yy * yy) / (r * r), 0, 1)) * r * rng.uniform(0.5, 0.9)
            np.maximum(Z[y0:y1, x0:x1], z, out=Z[y0:y1, x0:x1])
    return Z


class Comp:
    __slots__ = ('x', 'y', 'rx', 'ry', 'rot', 'm', 'lvl', 'zk')

    def __init__(self, x, y, rx, ry, m, lvl, rot=0.0, zk=0.7):
        self.x, self.y, self.rx, self.ry = float(x), float(y), float(rx), float(ry)
        self.m, self.lvl, self.rot, self.zk = m, lvl, float(rot), zk


def _bb(c, x0, y0, ww, hh, pad=1.15):
    r = max(c.rx, c.ry) * pad + 2
    bx0, bx1 = int(max(c.x - r - x0, 0)), int(min(c.x + r - x0 + 1, ww))
    by0, by1 = int(max(c.y - r - y0, 0)), int(min(c.y + r - y0 + 1, hh))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return (slice(by0, by1), slice(bx0, bx1))


def _ell(X, Y, c, bb):
    cr, sr = math.cos(math.radians(c.rot)), math.sin(math.radians(c.rot))
    dx = X[bb] - c.x
    dy = Y[bb] - c.y
    u = (dx * cr + dy * sr) / c.rx
    v = (-dx * sr + dy * cr) / c.ry
    q2 = u * u + v * v
    q = np.sqrt(q2)
    a = np.clip((1 - q) * min(c.rx, c.ry) + 0.5, 0, 1)
    z = np.sqrt(np.clip(1 - q2, 0, 1)) * min(c.rx, c.ry) * c.zk
    return a, z, u, v


def _heads(rng, comps, c, m, lvl, n, rfrac, arc, push, Ld, lit_bias):
    """Buds along the rim of comp c inside the angular arc (deg, screen: -90 = up)."""
    out = []
    if n <= 0:
        return out
    angs = np.sort(rng.uniform(arc[0], arc[1], n))
    for ad in angs:
        th = math.radians(ad)
        nx, ny = math.cos(th), math.sin(th)
        lit = -(nx * Ld[0] + ny * Ld[1])          # >0 facing the sun
        r = min(c.rx, c.ry) * rng.uniform(*rfrac) * (1.0 + lit_bias * 0.0)
        px = c.x + nx * c.rx * push
        py = c.y + ny * c.ry * push
        k = Comp(px, py, r * rng.uniform(1.0, 1.25), r, m, lvl, rng.uniform(-20, 20), 0.75)
        out.append(k)
    comps.extend(out)
    return out


def build(rng, cx, base_y, Hc, Ld, mdef=None, big=False):
    masses, comps = [], []
    for mi, (mx, s, rx, ry, kind) in enumerate(MASSES if mdef is None else mdef):
        s = s + rng.uniform(-0.008, 0.008)
        x = cx + (axis_x(s) + mx) * Hc + rng.uniform(-0.01, 0.01) * Hc
        y = base_y - s * Hc
        rxp, ryp = rx * Hc, ry * Hc
        masses.append(dict(x=x, y=y, rx=rxp, ry=ryp, kind=kind, s=s, mx=mx))
        rot = rng.uniform(-6, 6)
        core = Comp(x, y + 0.08 * ryp, rxp * 0.9, ryp * 0.85, mi, 0, rot, 0.5)
        comps.append(core)
        if kind == 'flare':
            l1 = _heads(rng, comps, core, mi, 1, 3, (0.5, 0.8), (-175, -5), 0.7, Ld, 0)
        elif kind == 'skirt':
            l1 = _heads(rng, comps, core, mi, 1, int(rng.integers(4, 6)), (0.4, 0.62), (-165, -15), 0.62, Ld, 0)
        elif big:
            # big lobes: a few broad heads on the upper rim only (the lobe stays ONE round mass)
            nh = 3 if kind == 'body' else 2
            l1 = _heads(rng, comps, core, mi, 1, nh, (0.45, 0.62) if kind != 'lobe' else (0.4, 0.55),
                        (-165, -40), 0.55, Ld, 0)
        else:
            nh = int(rng.integers(3, 6))
            l1 = _heads(rng, comps, core, mi, 1, nh, (0.3, 0.62), (-178, -4), 0.82, Ld, 0)
        for c in l1:
            # outward direction from the core
            ang = math.degrees(math.atan2(c.y - core.y, c.x - core.x))
            n2 = int(rng.integers(2, 5)) if not big else 2
            _heads(rng, comps, c, mi, 2, n2, (0.26, 0.5) if not big else (0.3, 0.45), (ang - 80, ang + 80),
                   0.85, Ld, 0)
    return masses, comps


def _anvil_field(xs, ys, cx, base_y, Hc, seed):
    """Anvil coverage (soft) + a thickness coordinate: a broad flat plate spreading downwind (right)."""
    xr = (xs - cx) / Hc                                   # -> right
    sh = (base_y - ys) / Hc
    left, right = -0.36, 1.05
    # top: nearly flat, a gentle dome over the updraft, sinking a little downwind
    top = 0.985 - 0.02 * xr - 0.035 * _ss(0.35, 1.05, xr) + 0.018 * np.exp(-((xr - 0.05) / 0.25) ** 2)
    # underside: flat and fairly low near the column, rising (thinning) downwind
    bot = 0.84 + 0.03 * _ss(-0.2, 0.2, xr) + 0.075 * _ss(0.25, 1.1, xr) + 0.03 * _ss(-0.1, -0.36, xr)
    n1 = _noise(xs.shape[1], ys.shape[0], 0.08 * Hc, seed + 3, 3)
    top = top + 0.008 * n1
    bot = bot + 0.012 * n1
    ends = _ss(left - 0.01, left + 0.05, xr) * _ss(right + 0.01, right - 0.25, xr)
    inside = _ss(bot - 0.004, bot + 0.012, sh) * _ss(top + 0.003, top - 0.003, sh)
    v = np.clip((sh - bot) / np.maximum(top - bot, 1e-3), 0, 1)       # 0 underside -> 1 top
    return inside, ends, v, xr, top, bot


def _shelf_field(xs, ys, cx, base_y, Hc, seed):
    """Hero anvil (round 16): a wide, flat, WEDGE-shaped shelf sheared off the crown downwind (right):
    thin where it leaves the crown, thickest (~0.05 Hc) at the downwind end; a dead-flat top (crisp), a
    flat underside that sags downwind; the far end frays into fibres. Same signature as _anvil_field."""
    xr = (xs - cx) / Hc
    sh = (base_y - ys) / Hc
    left, right = 0.1, 0.93
    n1 = _noise(xs.shape[1], ys.shape[0], 0.09 * Hc, seed + 3, 2)
    n2 = _noise(xs.shape[1], ys.shape[0], 0.025 * Hc, seed + 4, 2, stretch=6.0)
    top = 0.95 + 0.022 * _ss(0.0, 0.9, xr) + 0.0015 * n1
    thick = 0.022 + 0.036 * np.clip((xr - 0.15) / 0.68, 0, 1) ** 0.8
    thick = thick + 0.012 * np.exp(-((xr - 0.2) / 0.08) ** 2)            # fillet where it leaves the crown
    # underside hangs in small rounded scallops (fading out toward the fibrous end)
    sc_ = _noise(xs.shape[1], ys.shape[0], 0.035 * Hc, seed + 6, 2, stretch=0.6)
    bot = top - thick + 0.004 * n1 - 0.007 * (1 - np.abs(sc_)) * _ss(0.8, 0.5, xr)
    # trailing end frays into horizontal fibres of different lengths
    fl = _noise(xs.shape[1], ys.shape[0], 0.012 * Hc, seed + 5, 3, stretch=14.0, angle=0.5)
    xe = xr + 0.07 * fl
    ends = _ss(left - 0.02, left + 0.05, xr) * _ss(right + 0.04, right - 0.06, xe)
    inside = _ss(bot - 0.005, bot + 0.004, sh) * _ss(top + 0.0011, top - 0.0011, sh)
    v = np.clip((sh - bot) / np.maximum(top - bot, 1e-3), 0, 1)
    return inside, ends, v, xr, top, bot


def _wedge_hero(xs, ys, cx, base_y, Hc, seed):
    """Hero anvil (round 17): a flat WEDGE that grows out of the tower top and is sheared downwind
    (right). Broadest where it leaves the column, its flat underside rises to meet the near-flat top
    downwind, and the thin trailing end frays into long cirrus strands (wwy_02).
    Returns (solid coverage, total alpha incl. strands, v 0 underside -> 1 top, xr, top, bot, rgb)."""
    h, w = ys.shape[0], xs.shape[1]
    xr = (xs - cx) / Hc + 0 * ys
    sh = (base_y - ys) / Hc + 0 * xs
    n1 = _noise(w, h, 0.09 * Hc, seed + 3, 2)
    # near-flat top (tropopause): a faint dome over the updraft, very slowly rising downwind
    top = (0.972 + 0.012 * _ss(0.1, 1.1, xr) + 0.014 * np.exp(-((xr - 0.02) / 0.2) ** 2) + 0.0015 * n1)
    # thickness: broad at the root, tapering downwind
    tk = 1.0 - _ss(-0.05, 1.02, xr)
    thick = 0.012 + 0.125 * tk ** 1.25
    # underside: small rounded scallops near the root (fading out downwind), soft wobble
    sc_ = _noise(w, h, 0.03 * Hc, seed + 6, 2, stretch=0.7)
    bot = top - thick + 0.004 * n1 - 0.008 * (1 - np.abs(sc_)) * _ss(0.75, 0.35, xr)
    # root: the underside dips down into the column top (the wedge flares out of the tower)
    bot = bot - 0.045 * np.exp(-((xr - 0.08) / 0.16) ** 2)
    # rounded up-sun end: the left boundary curls back at top and bottom
    mid = 0.5 * (top + bot)
    hv = np.clip((sh - mid) / np.maximum(0.5 * (top - bot), 1e-3), -1.5, 1.5)
    # the up-sun end is a diagonal flare: it leaves the column's left shoulder and spreads outward
    # (up-left) into the flat top, with a small rounded top corner
    xl = -0.03 - 0.12 * _ss(-1.0, 0.75, hv) + 0.035 * _ss(0.55, 1.05, hv) ** 2 + 0.012 * n1
    endl = _ss(xl - 0.004, xl + 0.006, xr)
    # solid plate; trailing end eroded by long horizontal fibres
    fl = _noise(w, h, 0.012 * Hc, seed + 5, 3, stretch=16.0, angle=0.6)
    fl2 = _noise(w, h, 0.004 * Hc, seed + 7, 2, stretch=20.0, angle=0.6)
    fr = 0.7 * fl + 0.3 * fl2
    right = 0.86 + 0.08 * fl
    inside = _ss(bot - 0.004, bot + 0.003, sh) * _ss(top + 0.0012, top - 0.0012, sh)
    ero = _ss(0.5, 0.95, xr)
    keep = _ss(-0.6 + 1.3 * ero - 0.1, -0.6 + 1.3 * ero + 0.1, fr + 0.2)
    solid = inside * endl * _ss(right + 0.03, right - 0.05, xr) * (1 - ero * (1 - keep))
    v = np.clip((sh - bot) / np.maximum(top - bot, 1e-3), 0, 1)
    # cirrus strands trailing off the end: thin bright threads fanning out and drooping a little
    yc = 0.5 * (top + bot) - 0.018 * _ss(0.7, 1.3, xr) ** 1.5
    spread = 0.01 + 0.028 * _ss(0.6, 1.3, xr)
    sd = (sh - yc) / spread
    s1 = _noise(w, h, 0.003 * Hc, seed + 8, 3, stretch=40.0, angle=1.2)
    s2 = _noise(w, h, 0.02 * Hc, seed + 9, 2, stretch=12.0, angle=1.2)
    thread = _ss(0.15, 0.6, s1 * 0.8 + 0.35 * s2 + 0.1)
    strands = (thread * np.exp(-sd * sd) * _ss(0.55, 0.85, xr) * _ss(1.32, 0.95, xr + 0.12 * s2)
               * (0.55 - 0.25 * _ss(0.9, 1.3, xr)))
    alpha = np.clip(np.maximum(solid, strands), 0, 1)
    # colour: flat painted planes: crisp warm-white lit top, a pale blue body, a darker flat underside
    fb = _noise(w, h, 0.03 * Hc, seed + 81, 3, stretch=10.0, angle=0.8)
    vv = np.clip(v + 0.1 * fb, 0, 1)
    c_un = _c((0.46, 0.53, 0.75)) - _c((0.03, 0.03, 0.0)) * _ss(0.3, 0.0, xr)[..., None]
    c_md = _c((0.78, 0.83, 0.93))
    c_tp = _c((1.04, 1.0, 0.93))
    k1 = _ss(0.1, 0.55, vv)[..., None]
    # the lit top band is thicker at the root (the rounded shoulder faces the sun) and thins downwind
    tb = 0.8 - 0.1 * tk
    k2 = _ss(tb - 0.06, tb + 0.04, v + 0.03 * fl)[..., None]
    col = c_un * (1 - k1) + c_md * k1
    col = col * (1 - k2) + c_tp * k2
    # thin downwind part is translucent / sky-lit (lighter, cooler); horizontal fibrous brush streaks
    thin = _ss(0.35, 0.9, xr)[..., None]
    col = col + (_c((0.84, 0.89, 0.97)) - col) * (0.45 * thin) * (1 - k2)
    col = col + 0.035 * (fb * (0.4 + 0.6 * _ss(0.2, 0.9, xr)))[..., None]
    # sun-facing up-wind end: a warm lit face (the sun is low to the left, at the same height)
    lf = _ss(xl + 0.06, xl + 0.005, xr) * _ss(0.15, 0.5, v)
    col = col + (_c((1.05, 1.0, 0.92)) - col) * (0.7 * lf)[..., None]
    # strands: pale, bright threads
    ks = np.clip(strands - solid, 0, 1)[..., None] * 2
    col = col + (_c((0.92, 0.95, 1.0)) - col) * np.clip(ks, 0, 1)
    return solid.astype(F32), alpha.astype(F32), v.astype(F32), xr.astype(F32), top, bot, col.astype(F32)


LAST = {}


def paint(pw, ph, cx, base_y, Hc, W, seed=4, Ld=(-0.62, -0.78), dbg=None, mdef=None, anvil=True, lean=LEAN, hwk=1.0,
          box=(0.75, 1.25, 1.1, 0.12), haze=0.0, haze_col=(0.8, 0.87, 0.96), tex=1.0, hero=False):
    _CFG['lean'], _CFG['hwk'] = lean, hwk
    _CFG['hw'] = HW_HERO if hero else None
    rng = np.random.default_rng(seed)
    Ld = np.asarray(Ld, F32) / np.linalg.norm(Ld)
    L3 = np.array([Ld[0] * 0.78, Ld[1] * 0.78, 0.62], F32)
    L3 /= np.linalg.norm(L3)
    big = hero
    masses, comps = build(rng, cx, base_y, Hc, Ld, HERO_MASSES if hero else mdef, big=hero)
    if hero:
        # shade-side masses: break each smooth sphere outline into smaller cauliflower sub-lobes
        # (separate rng so the lit side keeps its exact silhouette)
        rng2 = np.random.default_rng(seed + 991)
        for mi, M in enumerate(masses):
            if M['mx'] < 0.05 or M['s'] < 0.15:
                continue
            core = [c for c in comps if c.m == mi and c.lvl == 0][0]
            l1 = _heads(rng2, comps, core, mi, 1, 6, (0.22, 0.36), (-150, 35), 0.86, Ld, 0)
            for c in l1:
                ang = math.degrees(math.atan2(c.y - core.y, c.x - core.x))
                _heads(rng2, comps, c, mi, 2, 2, (0.3, 0.5), (ang - 70, ang + 70), 0.85, Ld, 0)
    x0 = max(int(cx - box[0] * Hc), 0)
    x1 = min(int(cx + box[1] * Hc), pw)
    y0 = max(int(base_y - box[2] * Hc), 0)
    y1 = min(int(base_y + box[3] * Hc), ph)
    ww, hh = x1 - x0, y1 - y0
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    sh = (base_y - ys) / Hc
    # brush wobble on every edge
    amp = 0.004 * Hc
    wx = _noise(ww, hh, 0.02 * Hc, seed + 1, 3) * amp
    wy = _noise(ww, hh, 0.02 * Hc, seed + 2, 3) * amp
    X = xs + wx
    Y = ys + wy
    # ---------------- anvil (behind everything)
    if hero:
        wd_solid, wd_alpha, an_v, xr, an_top, an_bot, wd_col = _wedge_hero(xs, ys, cx, base_y, Hc, seed)
        an_in, an_ends = wd_solid, np.ones_like(wd_solid)
    else:
        an_in, an_ends, an_v, xr, an_top, an_bot = _anvil_field(X, Y, cx, base_y, Hc, seed)
    # fibrous downwind tail: streaks eat into the plate from the right
    fib = _noise(ww, hh, 0.05 * Hc, seed + 11, 4, stretch=8.0, angle=2.0)
    fib2 = _noise(ww, hh, 0.012 * Hc, seed + 12, 3, stretch=10.0, angle=2.0)
    tail = _ss(0.55, 1.05, xr) if not hero else _ss(0.62, 1.0, xr)
    edgev = 1 - 4 * an_v * (1 - an_v)                    # 1 at the top / underside, 0 mid-plate
    ero = tail * (0.45 + 0.55 * edgev)
    if hero:
        # the flat top stays intact; fibres eat the underside and the trailing end
        ero = tail * (0.35 + 0.65 * _ss(0.5, 0.0, an_v))
    fthr = -0.9 + 1.25 * ero
    an_a = an_in * an_ends * _ss(fthr - 0.05 - 0.2 * ero, fthr + 0.05 + 0.2 * ero, 0.7 * fib + 0.3 * fib2 + 0.25)
    if hero:
        an_a = wd_solid.copy()
    # the left (up-sun) end curls into cauliflower heads: added as comps
    anvil_heads = []
    for (xk, rk, dy) in () if hero else ((-0.27, 0.03, 0.012), (-0.19, 0.024, 0.0), (-0.1, 0.034, -0.004), (0.18, 0.026, 0.01),
                         (0.3, 0.02, 0.016)):
        topk = 0.985 - 0.02 * xk + 0.018 * math.exp(-((xk - 0.05) / 0.25) ** 2)
        r = (rk + rng.uniform(-0.004, 0.004)) * Hc
        yk = base_y - (topk - 0.02 - dy + rng.uniform(-0.006, 0.004)) * Hc
        anvil_heads.append(Comp(cx + xk * Hc, yk, r * 1.35, r, -1, 1, rng.uniform(-10, 10), 0.75))
    # left end: one rounded head at the tip
    if not hero:
        anvil_heads.append(Comp(cx - 0.32 * Hc, base_y - 0.925 * Hc, 0.065 * Hc, 0.042 * Hc, -1, 1, -8, 0.75))
    for c in list(anvil_heads):
        ang = math.degrees(math.atan2(c.y - (base_y - 0.93 * Hc), c.x - (cx - 0.1 * Hc)))
        _heads(rng, anvil_heads, c, -1, 2, 2, (0.3, 0.5), (-160, -30), 0.8, Ld, 0)
    # overshooting dome over the updraft
    od = (Comp(cx + 0.02 * Hc, base_y - 0.975 * Hc, 0.13 * Hc, 0.05 * Hc, -2, 0, 0, 0.6) if not hero else
          Comp(cx + 0.1 * Hc, base_y - 0.93 * Hc, 0.14 * Hc, 0.045 * Hc, -2, 0, -4, 0.6))
    od_heads = [od]
    _heads(rng, od_heads, od, -2, 1, 6, (0.35, 0.55), (-175, -5), 0.62, Ld, 0)
    if hero:
        od_heads = []          # no separate knobbly cap: the column flows straight into the wedge
    for c in list(od_heads[1:]):
        ang = math.degrees(math.atan2(c.y - od.y, c.x - od.x))
        _heads(rng, od_heads, c, -2, 2, 3, (0.3, 0.45), (ang - 70, ang + 70), 0.6, Ld, 0)
    ah_cov = np.zeros((hh, ww), F32)
    ah_z = np.zeros((hh, ww), F32)
    for c in anvil_heads + od_heads:
        bb = _bb(c, x0, y0, ww, hh)
        if bb is None:
            continue
        a, z, _, _ = _ell(X, Y, c, bb)
        np.maximum(ah_cov[bb], a, out=ah_cov[bb])
        np.maximum(ah_z[bb], z * (a > 0.5), out=ah_z[bb])
    shelf_a = an_a.copy()
    an_a = np.maximum(an_a, ah_cov)
    if not anvil:
        an_a = an_a * 0
        ah_cov = ah_cov * 0
        ah_z = ah_z * 0
    # ---------------- column masses
    nm = len(masses)
    mcov = [None] * nm
    mz = [None] * nm
    mbb = [None] * nm
    for mi, M in enumerate(masses):
        r = max(M["rx"], M["ry"]) * 3.2
        bx0, bx1 = int(max(M['x'] - r - x0, 0)), int(min(M['x'] + r - x0, ww))
        by0, by1 = int(max(M['y'] - r - y0, 0)), int(min(M['y'] + r - y0, hh))
        mbb[mi] = (slice(by0, by1), slice(bx0, bx1))
        mcov[mi] = np.zeros((by1 - by0, bx1 - bx0), F32)
        mz[mi] = np.zeros((by1 - by0, bx1 - bx0), F32)
    for c in comps:
        bb = _bb(c, x0, y0, ww, hh)
        if bb is None:
            continue
        a, z, _, _ = _ell(X, Y, c, bb)
        mb = mbb[c.m]
        sy = slice(max(bb[0].start, mb[0].start), min(bb[0].stop, mb[0].stop))
        sx = slice(max(bb[1].start, mb[1].start), min(bb[1].stop, mb[1].stop))
        if sy.stop <= sy.start or sx.stop <= sx.start:
            continue
        la = a[sy.start - bb[0].start:sy.stop - bb[0].start, sx.start - bb[1].start:sx.stop - bb[1].start]
        lz = z[sy.start - bb[0].start:sy.stop - bb[0].start, sx.start - bb[1].start:sx.stop - bb[1].start]
        tgt = (slice(sy.start - mb[0].start, sy.stop - mb[0].start), slice(sx.start - mb[1].start, sx.stop - mb[1].start))
        np.maximum(mcov[c.m][tgt], la, out=mcov[c.m][tgt])
        np.maximum(mz[c.m][tgt], lz, out=mz[c.m][tgt])
    colcov = np.zeros((hh, ww), F32)
    for mi in range(nm):
        np.maximum(colcov[mbb[mi]], mcov[mi], out=colcov[mbb[mi]])
    # fill the core of the column (no sky holes): closing + hole fill at low res
    kq = 4
    small = cv2.resize(colcov, (ww // kq, hh // kq), interpolation=cv2.INTER_AREA)
    kr = max(int(0.04 * Hc / kq), 2)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ker)
    bg = (closed < 0.5).astype(np.uint8)
    ff = bg.copy()
    msk = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    for (px_, py_) in ((0, 0), (ff.shape[1] - 1, 0), (0, ff.shape[0] - 1), (ff.shape[1] - 1, ff.shape[0] - 1)):
        if ff[py_, px_] == 1:
            cv2.floodFill(ff, msk, (px_, py_), 2)
    holes = ((bg == 1) & (ff == 1)).astype(F32)
    closed = np.maximum(closed, holes)
    closed = cv2.resize(cv2.GaussianBlur(closed, (0, 0), 1.0), (ww, hh), interpolation=cv2.INTER_LINEAR)
    fill = np.clip((closed - 0.5) * 3 + 0.5, 0, 1)
    colcov = np.maximum(colcov, fill * (_ss(0.9, 0.84, sh) if anvil and not hero else 1.0))
    # the neck joins the anvil underside
    # flat base: cut the column at the (slightly stepped) base line
    bn = _noise(ww, 1, 0.25 * Hc, seed + 9, 2)[0][None, :]
    base_off = 0.012 * bn
    if hero:
        base_off = base_off + 0.004 * _noise(ww, 1, 0.03 * Hc, seed + 10, 2)[0][None, :]
    basecut = _ss(-0.004, 0.008, sh - base_off)
    colcov = colcov * basecut
    if hero:
        # nothing of the column pokes above the anvil's flat top
        colcov = colcov * _ss(an_top + 0.004, an_top - 0.004, sh)
    cov = np.maximum(colcov, an_a)
    # ---------------- fine cauliflower buds on the lit silhouette
    buds = []
    for rr_, ks_, gp_ in (((0.018, 0.034), 0.15, (1.6, 3.2)), ((0.005, 0.012), 0.2, (1.0, 1.6))):
        bl = _cauliflower(rng, cov, x0, y0, Hc, Ld, base_y, cx, rr_rng=rr_, keep_shade=ks_, gap=gp_, fix_lit=hero)
        if hero:
            bl = [c for c in bl if (base_y - c.y) / Hc < 0.855]
        bcov = np.zeros((hh, ww), F32)
        for c in bl:
            bb = _bb(c, x0, y0, ww, hh)
            if bb is None:
                continue
            a, _, _, _ = _ell(X, Y, c, bb)
            np.maximum(bcov[bb], a, out=bcov[bb])
        cov = np.maximum(cov, bcov * _ss(0.02, 0.06, sh))
        buds += bl
    mkf = (cov > 0.5).astype(F32)
    # ---------------- BIG FORM value: whole silhouette dome lit from the upper left + across-the-column
    # position + height; a continuous core shadow in the lower right
    hgt = cv2.GaussianBlur(mkf, (0, 0), 0.05 * Hc) * 0.4 + cv2.GaussianBlur(mkf, (0, 0), 0.14 * Hc) * 0.6
    gy_, gx_ = np.gradient(hgt * 0.35 * Hc)
    nz = np.sqrt(gx_ ** 2 + gy_ ** 2 + 0.6 ** 2)
    dome = (-gx_ * L3[0] - gy_ * L3[1] + 0.6 * L3[2]) / nz
    sc = np.clip(sh, 0, 1)
    ax = cx + axis_x(sc) * Hc
    u = (xs - ax) / (half_w(sc) * Hc)
    gn = _noise(ww, hh, 0.12 * Hc, seed + 21, 3)
    G = 0.47 - 0.25 * np.clip(u, -1.2, 1.4) + 0.3 * (dome - 0.55) + 0.14 * (np.clip(sh, 0, 1) - 0.5) + 0.03 * gn
    core = np.exp(-((u - 0.55) / 0.65) ** 2 - ((sh - 0.3) / 0.28) ** 2)
    G = G - 0.17 * core - 0.1 * _ss(0.25, 0.0, sh)
    # anvil: lit flat top, shadowed underside, sheltered where it overhangs the column
    anG = 0.5 + 0.34 * an_v - 0.1 * _ss(0.2, 1.0, xr) + 0.03 * gn
    if hero:
        # flat painted planes: a crisp warm-white lit top, a mid body, a slightly darker flat underside
        anG = (0.43 + 0.12 * _ss(0.15, 0.45, an_v) + 0.27 * _ss(0.7, 0.86, an_v) - 0.06 * _ss(0.3, 1.0, xr)
               + 0.02 * gn).astype(F32)
    G = np.where(an_a > colcov, anG, G).astype(F32)
    V = (G - 0.16 - 0.08 * _ss(-0.3, 0.8, u)).astype(F32)       # body under everything (shade plane)
    V = np.where(an_a > colcov, anG - 0.02, V).astype(F32)

    def _lam(z, sig):
        zb = cv2.GaussianBlur(z, (0, 0), sig) if sig > 0.3 else z
        gyy, gxx = np.gradient(zb)
        n_ = np.sqrt(gxx ** 2 + gyy ** 2 + 1.0)
        return (-gxx * L3[0] - gyy * L3[1] + L3[2]) / n_, -gyy / n_

    # anvil heads (lit cauliflower on its up-sun end + overshooting dome)
    lamh, nyh = _lam(ah_z, 0.003 * Hc)
    wh = ah_cov * _ss(0.0, 0.25, ah_z / (0.02 * Hc))
    Vh = anG + 0.35 * (lamh - 0.55)
    wh = np.clip(ah_cov * 0.95, 0, 1) * _ss(-0.9, 0.2, nyh + 0.0)
    V = V * (1 - wh) + Vh * wh
    # ---------------- primary masses back -> front: big lambert (terminator wraps each mass), bud
    # lambert only in the light, crisp lit crown, lost underside
    crest_sh = np.zeros((hh, ww), F32)          # lobe crests of the shade-side masses (a few break through)
    crest_lt = np.zeros((hh, ww), F32)          # warm-cream crests on the lit side
    pocket = np.zeros((hh, ww), F32)            # cool secondary shadow above each lit lobe's crest
    for mi, M in enumerate(masses):
        bb = mbb[mi]
        m = mcov[mi] * colcov[bb]
        if m.max() < 0.01:
            continue
        r = min(M['rx'], M['ry'])
        zb = cv2.GaussianBlur(m, (0, 0), 0.45 * r) * r * 0.9
        lam_b, _ = _lam(zb, 0)
        lam_f, nyf = _lam(mz[mi], 0.02 * r + 0.5)
        # position of the mass in the big light (shade-side masses are low contrast)
        shade = float(np.clip((M['mx'] + 0.05) / 0.35, 0, 1)) * (1.0 if M['kind'] != 'shade' else 1.3)
        shade = min(shade, 1.0)
        kb = 0.46 - 0.22 * shade
        shk = float(_ss(0.0, 0.25, M['mx'])) if big else 0.0
        kb = kb * (1 - 0.6 * shk)
        g = G[bb]
        Vm = g + kb * (lam_b - 0.5)
        det = _ss(0.45, 0.75, lam_b) * (1 - 0.6 * shade)
        Vm = Vm + 0.42 * det * (lam_f - lam_b)
        # lost underside: the lower part of the mass dissolves into the body value
        yy = (Y[bb] - M['y']) / M['ry']
        brk = 0.25 * _noise(m.shape[1], m.shape[0], 0.05 * Hc, seed + 100 + mi, 2)
        lost = _ss(0.55, -0.15, yy + brk)
        if M['kind'] == 'skirt':
            lost = np.maximum(lost, 0.7)
        if big:
            # a hard, warm-white lit crest along the sun-facing (upper-left) rim of every big mass / lobe
            mb_ = cv2.GaussianBlur(m, (0, 0), max(0.004 * Hc, 1.0))
            gyy, gxx = np.gradient(mb_)
            gl_ = np.sqrt(gxx * gxx + gyy * gyy) + 1e-6
            face = np.clip(-(gxx * Ld[0] + gyy * Ld[1]) / gl_, 0, 1)
            din_m = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 3).astype(F32)
            band = (1 - _ss(0.0025 * Hc, 0.007 * Hc, din_m)) * (m > 0.5)
            crest = band * _ss(0.25, 0.75, face) * (1 - 0.75 * shade)
            Vm = Vm + 0.13 * crest * (1 - shk)
            a_ = m * lost
            crest_sh[bb] = crest_sh[bb] * (1 - a_) + crest * shk * a_
            crest_lt[bb] = crest_lt[bb] * (1 - a_) + crest * (1 - shk) * (1 - shade) * a_
            pocket[bb] = pocket[bb] * (1 - (m > 0.5))
            if shk < 0.5 and M['kind'] in ('body', 'lobe') and M['s'] > 0.12:
                # the lobe above/behind this one sits in a cool pocket just over its lit crest
                din_o = cv2.distanceTransform((m <= 0.5).astype(np.uint8), cv2.DIST_L2, 3).astype(F32)
                above = _ss(0.2, -0.4, (Y[bb] - M['y']) / M['ry'])
                pk_ = np.exp(-din_o / (0.011 * Hc)) * (m <= 0.5) * above * _ss(0.2, 0.8, face + 0.3)
                np.maximum(pocket[bb], pk_ * (1 - shade), out=pocket[bb])
            if M['kind'] == 'lobe':
                # cool underside on each mid-size lobe before it dissolves into the mass below
                Vm = Vm - 0.07 * _ss(0.05, 0.6, yy) * _ss(-0.2, 0.3, lam_b)
        a = m * lost
        V[bb] = V[bb] * (1 - a) + Vm * a
    if big:
        # ONE blue-grey shadow mass down the right: the shade-side lobes are merged (big soft blur of the
        # value) and only a few lighter lobe crests break through; painterly directional strokes
        zone = (_ss(0.02, 0.4, u) * _ss(0.12, 0.2, sh) * (colcov > 0.5)).astype(F32)
        zone = cv2.GaussianBlur(zone, (0, 0), 0.02 * Hc)
        Vbl = cv2.GaussianBlur(V, (0, 0), 0.05 * Hc)
        V = V + (Vbl - V) * (0.75 * zone)
        V = V + 0.13 * np.clip(cv2.GaussianBlur(crest_sh, (0, 0), 0.003 * Hc) * 1.4, 0, 1) * zone
        stk = (_noise(ww, hh, 0.028 * Hc, seed + 401, 3, stretch=5.0, angle=-62) * 0.7
               + _noise(ww, hh, 0.01 * Hc, seed + 402, 2, stretch=7.0, angle=-55) * 0.3)
        V = V + 0.05 * stk * zone
        # lit side: cool secondary shadows between lobes (pocket over each crest)
        litz = (1 - zone) * (colcov > 0.5)
        V = V - 0.075 * cv2.GaussianBlur(pocket, (0, 0), 0.003 * Hc) * litz
        LAST['zone'] = zone
    # ---------------- small cauliflower texture inside the LIGHT (grey-blue crevices between bright puffs);
    # nearly absent in the shade (broad plane there)
    Zt = (_dome_field(ww, hh, max(0.04 * Hc, 6.0), (0.45, 0.85), seed + 300) +
          0.55 * _dome_field(ww, hh, max(0.017 * Hc, 3.5), (0.45, 0.9), seed + 301))
    lt, _ = _lam(Zt, 0.0015 * Hc)
    lt = lt - float(np.mean(lt))
    Vb0 = cv2.GaussianBlur(V, (0, 0), 0.01 * Hc)
    litm = _ss(0.42, 0.72, Vb0)
    V = V + tex * (0.16 * litm + 0.04 * (1 - litm)) * lt * _ss(0.05, 0.15, sh) * (1 - 0.6 * (an_a > colcov) * (ah_cov < 0.5)) * (
        (1 - 0.85 * LAST['zone']) if big else 1.0)
    # ---------------- fine buds: tiny lit tops on the silhouette
    for c in buds:
        bb = _bb(c, x0, y0, ww, hh, 1.05)
        if bb is None:
            continue
        a, z, uu, vv = _ell(X, Y, c, bb)
        f = -(uu * Ld[0] + vv * Ld[1])
        Vl = np.maximum(V[bb], G[bb]) + 0.1 * np.clip(-f, -1, 1)
        la = a * _ss(-0.3, 0.5, -vv * 0.8 - f * 0.4) * 0.9
        V[bb] = V[bb] * (1 - la) + Vl * la
    # ---------------- value -> painted colour ramp (soft value steps, brush-warped)
    wn = _noise(ww, hh, 0.035 * Hc, seed + 5, 3) * 0.03 + _noise(ww, hh, 0.01 * Hc, seed + 6, 2, stretch=3, angle=-30) * 0.012
    Vw = V + wn
    steps = [(0.0, (0.36, 0.47, 0.7)), (0.15, (0.43, 0.54, 0.77)), (0.28, (0.52, 0.63, 0.83)),
             (0.4, (0.64, 0.73, 0.88)), (0.5, (0.79, 0.83, 0.91)), (0.58, (0.91, 0.915, 0.93)),
             (0.68, (1.0, 0.975, 0.935)), (0.85, (1.04, 1.01, 0.95))]
    lv = np.array([s_[0] for s_ in steps], F32)
    Vs = np.zeros_like(Vw)
    for i in range(1, len(lv)):
        mid = 0.5 * (lv[i - 1] + lv[i])
        Vs = Vs + (lv[i] - lv[i - 1]) * _ss(mid - 0.025, mid + 0.025, Vw)
    Vs = Vs + lv[0]
    litq = _ss(0.45, 0.6, Vw)
    pk = 0.3 + 0.35 * litq
    Vf = np.clip(Vw * (1 - pk) + Vs * pk, -0.05, 1)
    cs = np.array([s_[1] for s_ in steps], F32)
    col = np.stack([np.interp(Vf, lv, cs[:, k]) for k in range(3)], -1).astype(F32)
    # sky-bounce: up-facing shade planes a touch bluer, down-facing ones a faint warm bounce
    lowsh = _ss(0.45, 0.25, Vf) * _ss(0.55, 0.2, sh)
    col = col + (_c((0.78, 0.74, 0.8)) - col) * (0.12 * lowsh)[..., None]
    # flat, darker underside band at the base, then aerial perspective on the lower tower
    if hero:
        # a darker, FLAT base band (one broad blue-grey plane, slightly warm bounce from the ground)
        # value break-up: horizontal darker slabs and lighter torn scud streaks across the base plane
        bn1 = _noise(ww, hh, 0.06 * Hc, seed + 71, 3, stretch=6.0, angle=0.0)
        bn2 = _noise(ww, hh, 0.015 * Hc, seed + 72, 2, stretch=8.0, angle=0.0)
        bb_ = _ss(0.085 + 0.02 * bn1, 0.025, sh - base_off) * (colcov > 0.3)
        bcol = _c((0.37, 0.43, 0.62)) + _c((0.07, 0.07, 0.06)) * np.clip(0.8 * bn1 + 0.4 * bn2, -1, 1)[..., None]
        col = col + (bcol - col) * (0.88 * bb_)[..., None]
        # thin lighter scud lines hanging along the base (sky light catching torn fragments)
        scud = _ss(0.35, 0.8, bn2 + 0.4 * bn1) * _ss(0.0, 0.012, sh - base_off) * _ss(0.05, 0.02, sh - base_off)
        col = col + (_c((0.7, 0.76, 0.89)) - col) * (0.5 * scud * (colcov > 0.3))[..., None]
        hzk = _ss(0.3, 0.0, sh) * 0.16
    else:
        bb_ = _ss(0.05, 0.0, sh - base_off) * (colcov > 0.3)
        col = col + (_c((0.55, 0.62, 0.77)) - col) * (0.5 * bb_)[..., None]
        hzk = _ss(0.42, 0.0, sh) * 0.5
    col = col + (_c((0.8, 0.87, 0.96)) - col) * hzk[..., None]
    if hero:
        # a few warm-cream crest highlights along the lit lobes (form inside the warm-white mass)
        cr = cv2.GaussianBlur(crest_lt, (0, 0), 0.004 * Hc)
        cr = np.clip(cr * 1.6, 0, 1) * _ss(0.5, 0.7, Vf)
        col = col + (_c((1.07, 0.985, 0.85)) - col) * (0.55 * cr)[..., None]
    # painterly simplification: flat brush patches with crisp borders (generalised Kuwahara)
    kr = max(2, int(round(0.0045 * Hc)))
    col = K3.kuwahara(col, kr, q=4.0 if not hero else 2.0)
    if haze > 0:
        hk = haze * (0.55 + 0.45 * _ss(0.6, 0.0, sh))
        col = col + (_c(haze_col) - col) * hk[..., None]
    # dry-brush grain
    st = _noise(ww, hh, 0.01 * Hc, seed + 41, 2, stretch=5.0, angle=-35)
    col = col * (1 + 0.01 * st[..., None])
    # ---------------- shade-side silhouette: thin vapour lit by the sky, never a dark contour
    din = cv2.distanceTransform((cov > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    edge = np.exp(-din / (0.012 * Hc))
    Vb = cv2.GaussianBlur(V, (0, 0), 0.015 * Hc)
    shd_e = _ss(0.5, 0.3, Vb)
    ek = np.clip(edge * shd_e, 0, 1) * 0.5
    col = col + (_c((0.7, 0.79, 0.93)) - col) * ek[..., None]
    if hero:
        # faint warm bounce / iridescence where the shaded right side meets the sky
        eb = np.exp(-din / (0.02 * Hc)) * (1 - np.exp(-din / (0.004 * Hc))) * shd_e * _ss(0.08, 0.2, sh)
        hue = _ss(0.25, 0.85, sh)[..., None]
        tint = _c((0.97, 0.84, 0.82)) * (1 - hue) + _c((0.86, 0.82, 0.97)) * hue
        col = col + (tint - col) * (0.3 * eb)[..., None]
    # ---------------- alpha: crisp on the lit silhouette, soft on the shade side
    litness = _ss(0.42, 0.6, Vb)
    Asoft = cv2.GaussianBlur(cov, (0, 0), (0.006 if not hero else 0.011) * Hc)
    Asoft = np.minimum(np.clip(Asoft * 1.15, 0, 1), cv2.dilate(cov, np.ones((3, 3), np.uint8)))
    litx = cv2.dilate(litness, np.ones((5, 5), np.uint8))
    if hero:
        litx = np.maximum(litx, _ss(0.05, 0.02, sh - base_off) * 0.85)
    A = cov * litx + Asoft * (1 - litx)
    # anvil underside edge is soft; tail is wispy
    A = A * (1 - 0.3 * _ss(0.6, 1.05, xr) * (an_a > colcov))
    if hero and anvil:
        # the anvil wedge is painted as flat planes over the column top (its own crisp alpha); where it
        # leaves the column its underside melts softly into the tower's shade
        und = _ss(0.05, 0.5, an_v)
        over_col = cv2.GaussianBlur((colcov > 0.5).astype(F32), (0, 0), 0.02 * Hc)
        anm = np.clip(wd_alpha * (1 - over_col * (1 - und)), 0, 1)
        anm = np.maximum(anm, wd_solid * und)
        col = col * (1 - anm[..., None]) + wd_col * anm[..., None]
        A = np.maximum(A * (1 - wd_alpha) + wd_alpha, A)
        A = A * (1 - anm) + np.maximum(wd_alpha, A) * anm
    # foot: sink into haze (alpha fade at the base) + a streaky rain-foot / haze veil below it
    below = -(sh - base_off)
    colb = cv2.GaussianBlur(colcov[np.clip(int(base_y - y0 - 0.03 * Hc), 0, hh - 1)][None, :], (0, 0), 0.02 * Hc)
    if hero:
        # the flat base stays solid (it is a hard, dark plane); below it a lighter, soft, slanting rain veil
        # hangs from the shade-side half of the base and dissolves downward
        # the base plane keeps a crisp flat bottom edge (litness forced on near the base)
        xb = (xs - cx) / Hc
        slant = (ys - base_y) * 0.14                                 # streaks lean with the downdraft
        strk = _noise(ww, hh, 0.02 * Hc, seed + 61, 3, stretch=12.0, angle=82)
        strk2 = _noise(ww, hh, 0.006 * Hc, seed + 62, 2, stretch=16.0, angle=82)
        xs_ = xb + slant / Hc
        cn = _ss(-0.5, 0.5, _noise(ww, 1, 0.06 * Hc, seed + 63, 2)[0][None, :])
        curtain = _ss(-0.02, 0.1, xs_) * _ss(0.62, 0.42, xs_) * (0.7 + 0.3 * cn)
        # the veil hangs from the whole underside above it, clearly visible (~25-30 %) right under the
        # base and thinning with aerial fade toward the horizon
        top_e = _ss(-0.003, 0.006, below)
        fall = top_e * (0.55 + 0.45 * np.exp(-np.clip(below, 0, None) / 0.07))
        va = colb * curtain * fall * (0.6 + 0.3 * _ss(-0.5, 0.5, strk) + 0.15 * strk2)
        va = np.clip(va * 0.62, 0, 0.55)
        lite = _ss(0.1, 0.8, strk + 0.5 * strk2)                   # lighter shafts inside the veil
        aer = _ss(0.0, 0.1, below)[..., None]
        vcol = _c((0.5, 0.57, 0.76)) * (1 - lite[..., None]) + _c((0.76, 0.82, 0.94)) * lite[..., None]
        vcol = vcol + (_c((0.8, 0.86, 0.95)) - vcol) * (0.5 * aer)
        veil = np.zeros((ph, pw, 4), F32)
        veil[y0:y1, x0:x1, :3] = vcol
        veil[y0:y1, x0:x1, 3] = va
        veil[..., :3] = K3._bleed(veil[..., :3], (veil[..., 3] > 0.01).astype(F32), 4.0)
        LAST['veil'] = veil
        sa = np.zeros_like(A)
        scol = vcol
    else:
        A = A * (1 - 0.35 * _ss(0.06, -0.01, sh - base_off))
        strk = _ss(-0.2, 0.6, _noise(ww, hh, 0.05 * Hc, seed + 61, 3, stretch=7.0, angle=86))
        sa = np.clip(colb * np.exp(-np.clip(below, 0, None) / 0.05) * (0.25 + 0.25 * strk) * _ss(-0.01, 0.01, below), 0, 0.5)
        scol = _c((0.72, 0.79, 0.9))
    Ao = A + sa * (1 - A)
    col = (col * A[..., None] + scol * (sa * (1 - A))[..., None]) / np.maximum(Ao, 1e-4)[..., None]
    if hero:
        Ao = np.clip(Ao, 0, 1)
    A = Ao
    out = np.zeros((ph, pw, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = np.clip(A, 0, 1)
    out[..., :3] = K3._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * W))
    if dbg is not None:
        dbg.update(V=V, G=G, cov=cov, box=(x0, y0, x1, y1))
    return out


def _cauliflower(rng, cov, x0, y0, Hc, Ld, base_y, cx, q=2, rr_rng=(0.006, 0.016), keep_shade=0.25, gap=(1.1, 1.8),
                 fix_lit=False):
    m = cv2.resize(cov, (cov.shape[1] // q, cov.shape[0] // q), interpolation=cv2.INTER_AREA)
    cs, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=len)[:, 0, :].astype(F32)
    mb = cv2.GaussianBlur(m.astype(F32), (0, 0), max(0.012 * Hc / q, 1.0))
    gy, gx = np.gradient(mb)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    out = []
    t = 0.0
    while t < d[-1]:
        k = min(int(np.searchsorted(d, t)), len(c) - 1)
        px, py = c[k]
        nx, ny = -gx[int(py), int(px)], -gy[int(py), int(px)]
        nl = math.hypot(nx, ny) + 1e-9
        nx, ny = nx / nl, ny / nl
        X, Y = px * q + x0, py * q + y0
        s = (base_y - Y) / Hc
        xr = (X - cx) / Hc
        lit = -(nx * Ld[0] + ny * Ld[1])
        if fix_lit:
            lit = -lit                      # (n is outward, Ld points TOWARD the sun)
        rr = max(rng.uniform(*rr_rng) * Hc, 1.6)
        t += rr * rng.uniform(*gap) / q
        if ny > 0.45 or s < 0.12 or (s > 0.82 and xr > 0.3) or (s > 0.9 and abs(xr - 0.02) > 0.16 and xr > -0.25):
            continue
        if lit < -0.15 and rng.random() > keep_shade:
            continue
        rr *= (0.65 if lit < 0.1 else 1.0)
        dd = rr * rng.uniform(0.1, 0.45)
        out.append(Comp(X - nx * dd, Y - ny * dd, rr * rng.uniform(1.0, 1.3), rr, -1, 3, rng.uniform(-25, 25), 0.8))
    return out


def rim_light(P, Hc, sun, W, reach=0.35):
    """Backlit rim: the up-sun silhouette nearest the sun catches a warm, slightly blown edge."""
    h, w = P.shape[:2]
    sc = W / 1920.0
    r = int(reach * 2.5 * Hc)
    x0, x1 = max(int(sun[0]) - r, 0), min(int(sun[0]) + r, w)
    y0, y1 = max(int(sun[1]) - r, 0), min(int(sun[1]) + r, h)
    c = P[y0:y1, x0:x1, :3]
    a = P[y0:y1, x0:x1, 3]
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    d = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    near = np.exp(-(d / reach) ** 2)
    m = (a > 0.5).astype(np.uint8)
    ins = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(F32)
    # only on edges facing the sun
    ab = cv2.GaussianBlur(a, (0, 0), 3.0 * sc + 1)
    gy, gx = np.gradient(ab)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    vx, vy = (sun[0] - xs), (sun[1] - ys)
    vl = np.sqrt(vx * vx + vy * vy) + 1e-6
    face = np.clip(-(gx * vx + gy * vy) / (gl * vl), 0, 1)
    face = cv2.dilate(face, np.ones((5, 5), np.uint8))
    rw = (1.5 + 4.0 * near) * sc + 0.8
    band = (1 - _ss(rw * 0.5, rw * 1.4, ins)) * (m > 0)
    k = np.clip(band * (0.25 + 0.75 * near) * face, 0, 1)
    c += (_c((1.2, 1.13, 0.98)) - c) * k[..., None]
    # a broader warm glaze on the lit side near the sun
    k2 = np.exp(-(d / (reach * 1.3)) ** 2) * (1 - _ss(0.0, 0.05 * Hc, ins)) * 0.35 * (m > 0) * face
    c += (_c((1.15, 1.05, 0.9)) - c) * k2[..., None]
    P[y0:y1, x0:x1, :3] = c
    return P


def hero(pw, ph, cx, base_y, Hc, W, seed=4, dbg=None):
    """rgba plate + sun position (plate px): the sun sits just off the anvil's up-sun (left) tip, a little
    below it, so its glow wraps the tip and the column's upper-left shoulder."""
    P = paint(pw, ph, cx, base_y, Hc, W, seed=seed, dbg=dbg, hero=True, tex=0.4, box=(0.9, 1.1, 1.1, 0.16))
    # the sun sits well clear of the cloud, up and left in open sky (no tangent with the crown)
    sun = np.array([cx - 0.6 * Hc, base_y - 0.97 * Hc], np.float32)
    P = rim_light(P, Hc, sun, W, reach=0.6)
    return P, sun


CU_MASSES = [
    (-0.05, 0.74, 0.26, 0.2, 'body'),
    (-0.42, 0.46, 0.28, 0.22, 'body'),
    (0.32, 0.52, 0.3, 0.24, 'shade'),
    (-0.08, 0.42, 0.36, 0.28, 'body'),
    (0.64, 0.3, 0.24, 0.17, 'shade'),
    (-0.72, 0.2, 0.24, 0.13, 'skirt'),
    (0.02, 0.15, 0.55, 0.15, 'skirt'),
    (0.55, 0.13, 0.34, 0.12, 'shade'),
]


def cumulus(pw, ph, cx, base_y, Hc, W, seed=0, haze=0.3, haze_col=(0.8, 0.87, 0.96), flip=False):
    """A mid-distance summer cumulus painted with the SAME brush as the hero tower (nested heads, lit
    upper-left, lost blue-grey shade, cauliflower only on the lit silhouette), aerially faded."""
    md = [((-m[0] if flip else m[0]),) + tuple(m[1:]) for m in CU_MASSES]
    return paint(pw, ph, cx, base_y, Hc, W, seed=seed, mdef=md, anvil=False, lean=0.0, hwk=2.3,
                 box=(1.25, 1.25, 1.15, 0.1), haze=haze, haze_col=haze_col, tex=0.8)


if __name__ == '__main__':
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib import core as C, sky as S
    W, H = (1920, 1080) if 'full' in sys.argv else (960, 540)
    pw, ph = int(W * 1.12), int(H * 1.12)
    mg = 0.03 * H
    base_y = mg + 0.8 * H
    top_y = mg + 0.06 * H
    Hc = (base_y - top_y)
    t0 = time.time()
    P, sun = hero(pw, ph, (pw - W) / 2 + 0.42 * W, base_y, Hc, W, seed=4)
    print('hero', time.time() - t0, sun)
    sky = S.sky_gradient(pw, ph, dict(stops=[(0.0, '#0a45b8'), (0.16, '#155dcd'), (0.34, '#2e80de'), (0.5, '#58a2e9'),
                                             (0.64, '#8ec4f0'), (0.78, '#b8dcf6'), (0.9, '#d6ecf9'), (1.0, '#ebf8fc')],
                                      sun_glow='#fff6e0', sun_glow_amt=0.3, below='#eef9fc'), horizon=0.83)[..., :3]
    img = sky * (1 - P[..., 3:4]) + P[..., :3] * P[..., 3:4]
    ox = (pw - W) // 2
    img = img[int(mg):int(mg) + H, ox:ox + W]
    tag = [a for a in sys.argv[1:] if a != 'full']
    C.save_png(os.path.join('out', 'compare', 'p8_%s.png' % (tag[0] if tag else 'a')), np.clip(img, 0, 1))
