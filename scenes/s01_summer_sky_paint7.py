"""s01 helper (round 13): the hero cumulonimbus painted as a MASS HIERARCHY.

Round 12 (paint6) read as a rectangular cork with uniform 'bubble-wrap' dimples and a milky shade side.
This painter designs the tower like a background artist blocks it in:

  * SILHOUETTE: a leaning, bulging stack. ~17 designed masses (skirt, lower body, mid, upper body,
    crown).  Big masses bulge out past the column at staggered heights on the down-sun (right) side and
    the crown spreads into a sheared, flattened anvil ~1.5x the width of the mid column, trailing right.
  * THREE SCALES ONLY: major masses (250-400 px at 1080p) each get a crisp sunlit top and a broad, soft,
    LOST underside (the mass fades into the darker body beneath it); secondary heads (80-150 px) only
    along the crowns of the masses (fewer and weaker on the shade side); fine cauliflower only on the
    lit silhouette.  No small interior crescents.
  * VALUE: body = big form (up-sun left bright, down-sun lower right deep) + each mass's own lambert, so
    the terminator wraps around each mass.  Mapped to 4 painted value steps:
    warm white lit (#FFF8EC) -> pale half tone -> blue-grey mid shade (#B9C6DC) -> core shadow (#8FA0C0),
    plus warm peach bounce on the down-facing undersides of the low shaded masses.
  * EDGES: crisp on the lit silhouette, soft/lost where the shade side meets the sky; a stepped flat base
    with a few skirt lobes at different heights that dissolves into haze / rain-shaft streaks.
Deterministic, resolution independent (sizes in units of the tower height Hc).
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
    return K3._noise(w, h, max(w / max(px, 1.0), 2.0), seed, octaves, stretch=stretch, angle=angle)


def _c(v):
    return np.asarray(v, F32)


# designed masses: (x rel. axis, s = height of centre, rx, ry, kind) in units of Hc; painted in this
# order (crown first = behind, skirt last = in front)
MASSES = [
    # (x, s, rx, ry, kind, group). groups = the 6-7 MAJOR value masses; members are its secondary heads
    # crown / young anvil: wide, flattened, sheared toward the right
    (-0.4, 0.9, 0.13, 0.062, 'crown', 0),
    (-0.18, 0.935, 0.16, 0.078, 'crown', 0),
    (0.07, 0.95, 0.16, 0.075, 'crown', 0),
    (0.33, 0.935, 0.16, 0.065, 'crown', 0),
    (0.55, 0.91, 0.16, 0.05, 'crown', 0),
    (0.72, 0.895, 0.12, 0.03, 'crown', 0),
    (-0.27, 0.835, 0.12, 0.085, 'big', 0),
    (0.27, 0.85, 0.13, 0.075, 'bulge', 0),
    # rounded heaps, staggered diagonally (never in rows), each its own major value mass
    (0.06, 0.77, 0.14, 0.15, 'big', 1),
    (-0.14, 0.7, 0.12, 0.13, 'big', 2),
    (0.37, 0.745, 0.12, 0.1, 'bulge', 3),
    (0.04, 0.56, 0.15, 0.17, 'big', 4),
    (0.45, 0.53, 0.14, 0.12, 'bulge', 5),
    (-0.25, 0.49, 0.12, 0.14, 'big', 6),
    (0.24, 0.43, 0.12, 0.13, 'big', 7),
    (0.36, 0.27, 0.12, 0.1, 'bulge', 8),
    (-0.15, 0.27, 0.15, 0.14, 'big', 9),
    (0.1, 0.23, 0.16, 0.15, 'big', 10),
    # skirt: low, flat-bottomed lobes at slightly different heights
    (-0.3, 0.09, 0.12, 0.06, 'skirt', 11),
    (-0.1, 0.07, 0.17, 0.075, 'skirt', 11),
    (0.16, 0.075, 0.16, 0.07, 'skirt', 11),
    (0.33, 0.13, 0.12, 0.075, 'skirt', 11),
    (0.42, 0.06, 0.12, 0.05, 'skirt', 11),
]
LEAN = 0.08


class Lobe:
    __slots__ = ('x', 'y', 'rx', 'ry', 'rot', 'mass', 'lvl', 'jit')

    def __init__(self, x, y, rx, ry, mass, lvl, rot=0.0, jit=0.0):
        self.x, self.y, self.rx, self.ry = float(x), float(y), float(rx), float(ry)
        self.mass, self.lvl, self.rot, self.jit = mass, lvl, rot, jit


def _ell(X, Y, lb, bb):
    """Signed ellipse coverage (anti-aliased) + local normalised coords, inside bbox slices bb."""
    cr, sr = math.cos(math.radians(lb.rot)), math.sin(math.radians(lb.rot))
    dx = X[bb] - lb.x
    dy = Y[bb] - lb.y
    u = (dx * cr + dy * sr) / lb.rx
    v = (-dx * sr + dy * cr) / lb.ry
    q = np.sqrt(u * u + v * v)
    a = np.clip((1 - q) * min(lb.rx, lb.ry) + 0.5, 0, 1)
    return a, u, v, q


def _bbox(lb, x0, y0, ww, hh, pad=1.2):
    r = max(lb.rx, lb.ry) * pad
    bx0, bx1 = int(max(lb.x - r - x0, 0)), int(min(lb.x + r - x0 + 1, ww))
    by0, by1 = int(max(lb.y - r - y0, 0)), int(min(lb.y + r - y0 + 1, hh))
    if bx1 <= bx0 or by1 <= by0:
        return None
    return (slice(by0, by1), slice(bx0, bx1))


def build(rng, cx, base_y, Hc, Ld):
    masses = []
    lobes = []
    for mi, (mx, s, rx, ry, kind, grp) in enumerate(MASSES):
        x = cx + (mx + LEAN * s) * Hc + rng.uniform(-0.01, 0.01) * Hc
        y = base_y - s * Hc
        rxp, ryp = rx * Hc, ry * Hc
        masses.append(dict(x=x, y=y, rx=rxp, ry=ryp, kind=kind, s=s, mx=mx, grp=grp))
        # core body: two overlapping ellipses (never a compass circle)
        rot = rng.uniform(-8, 8) if kind != 'crown' else rng.uniform(-3, 3)
        lobes.append(Lobe(x - 0.18 * rxp, y + 0.1 * ryp, rxp * 0.82, ryp * 0.88, mi, 0, rot))
        lobes.append(Lobe(x + 0.2 * rxp, y + 0.05 * ryp, rxp * 0.8, ryp * 0.86, mi, 0, -rot))
        # secondary heads along the crown of the mass (upper arc), sized 0.3-0.45 of the mass
        side_lit = mx < 0.25
        n = int(rng.integers(4, 7)) if kind in ('big', 'bulge') else int(rng.integers(3, 6))
        if kind == 'skirt':
            n = int(rng.integers(3, 5))
        angs = np.sort(rng.uniform(-172, -8, n))
        for k, ad in enumerate(angs):
            th = math.radians(ad)
            if kind == 'crown':
                rr = ryp * rng.uniform(0.55, 0.8)
            elif kind == 'skirt':
                rr = ryp * rng.uniform(0.6, 0.85)
            else:
                rr = min(ryp * rng.uniform(0.36, 0.52), 0.08 * Hc)
            # heads on the down-sun side are broader and flatter (less definition in the shade)
            fl = 1.0 if (ad < -90 or side_lit) else 1.25
            px = x + math.cos(th) * rxp * 0.8
            py = y + math.sin(th) * ryp * 0.8 + 0.35 * rr
            lobes.append(Lobe(px, py, rr * rng.uniform(1.05, 1.3) * fl, rr, mi, 1, rng.uniform(-18, 18),
                              rng.uniform(-1, 1)))
        # small side heads low on the flanks (bulging outward, flat)
        for sd in (-1, 1):
            if rng.random() < 0.6 and kind in ('big', 'bulge'):
                rr = ryp * rng.uniform(0.3, 0.42)
                lobes.append(Lobe(x + sd * rxp * 0.88, y + ryp * rng.uniform(-0.05, 0.3), rr * 1.2, rr * 0.9, mi, 1,
                                  rng.uniform(-10, 10), rng.uniform(-1, 1)))
    return masses, lobes


def _cauliflower(rng, cov, x0, y0, Hc, Ld, base_y, q=4):
    """Fine cauliflower buds along the lit part of the silhouette only."""
    m = cv2.resize(cov, (cov.shape[1] // q, cov.shape[0] // q), interpolation=cv2.INTER_AREA)
    cs, _ = cv2.findContours((m > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=len)[:, 0, :].astype(F32)
    mb = cv2.GaussianBlur(m.astype(F32), (0, 0), max(0.02 * Hc / q, 1.0))
    gy, gx = np.gradient(mb)
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    step = 0.016 * Hc / q
    out = []
    for t in np.arange(0, d[-1], step):
        k = min(int(np.searchsorted(d, t)), len(c) - 1)
        px, py = c[k]
        nx, ny = -gx[int(py), int(px)], -gy[int(py), int(px)]
        nl = math.hypot(nx, ny) + 1e-9
        nx, ny = nx / nl, ny / nl
        X, Y = px * q + x0, py * q + y0
        s = (base_y - Y) / Hc
        lit = nx * Ld[0] + ny * Ld[1]
        if ny > 0.35 or s < 0.1:
            continue
        # dense on the lit side, sparse + small on the shade side
        if lit < -0.1 and rng.random() < 0.8:
            continue
        if rng.random() < 0.2:
            continue
        rr = rng.uniform(0.009, 0.02) * Hc * (0.7 if lit < 0.1 else 1.0)
        dd = rr * rng.uniform(0.15, 0.5)
        out.append(Lobe(X - nx * dd, Y - ny * dd, rr * rng.uniform(1.0, 1.3), rr, -1, 2, rng.uniform(-25, 25),
                        rng.uniform(-1, 1)))
    return out


def paint(pw, ph, cx, base_y, Hc, W, seed=4, Ld=(-0.55, -0.83), dbg=None):
    rng = np.random.default_rng(seed)
    Ld = np.asarray(Ld, F32) / np.linalg.norm(Ld)
    masses, lobes = build(rng, cx, base_y, Hc, Ld)
    x0 = max(int(cx - 1.0 * Hc), 0)
    x1 = min(int(cx + 1.25 * Hc), pw)
    y0 = max(int(base_y - 1.12 * Hc), 0)
    y1 = min(int(base_y + 0.3 * Hc), ph)
    ww, hh = x1 - x0, y1 - y0
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    sh = (base_y - ys) / Hc                                         # height in Hc (0 = base)
    # painterly domain warp: edges wobble like a loaded brush
    amp = 0.005 * Hc
    wx = _noise(ww, hh, 0.025 * Hc, seed + 1, 3) * amp + _noise(ww, hh, 0.007 * Hc, seed + 2, 2) * amp * 0.4
    wy = _noise(ww, hh, 0.025 * Hc, seed + 3, 3) * amp + _noise(ww, hh, 0.007 * Hc, seed + 4, 2) * amp * 0.4
    X = xs + wx
    Y = ys + wy
    # ---------------- per-mass coverage
    nm = len(masses)
    mcov = []
    cov = np.zeros((hh, ww), F32)
    for mi in range(nm):
        m = np.zeros((hh, ww), F32)
        for lb in lobes:
            if lb.mass != mi:
                continue
            bb = _bbox(lb, x0, y0, ww, hh)
            if bb is None:
                continue
            a, _, _, _ = _ell(X, Y, lb, bb)
            np.maximum(m[bb], a, out=m[bb])
        mcov.append(m)
        np.maximum(cov, m, out=cov)
    # fill the gaps between masses (a cloud has no sky holes through its core): closing of the union
    kq = 4
    small = cv2.resize(cov, (ww // kq, hh // kq), interpolation=cv2.INTER_AREA)
    kr = max(int(0.035 * Hc / kq), 2)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))
    closed = cv2.morphologyEx(small, cv2.MORPH_CLOSE, ker)
    # + fill every enclosed sky hole (background not connected to the window border)
    bg = (closed < 0.5).astype(np.uint8)
    ff = bg.copy()
    msk = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
    for (px_, py_) in ((0, 0), (ff.shape[1] - 1, 0), (0, ff.shape[0] - 1), (ff.shape[1] - 1, ff.shape[0] - 1)):
        if ff[py_, px_] == 1:
            cv2.floodFill(ff, msk, (px_, py_), 2)
    holes = ((bg == 1) & (ff == 1)).astype(F32)
    closed = np.maximum(closed, cv2.GaussianBlur(holes, (0, 0), 1.0))
    closed = cv2.resize(closed, (ww, hh), interpolation=cv2.INTER_LINEAR)
    fillm = np.clip((closed - 0.5) * 3 + 0.5, 0, 1) * _ss(0.02, 0.1, sh)
    cov = np.maximum(cov, fillm)
    # crown: flattened, sheared top edge that sinks gently to the right (anvil shear)
    xr = (xs - cx) / Hc
    top_line = base_y - Hc * (1.03 - 0.035 * np.clip(xr, -0.6, 1.0) - 0.05 * _ss(0.3, 0.95, xr)
                              + 0.022 * _noise(ww, 1, 0.07 * Hc, seed + 7, 3)[0][None, :])
    topcut = _ss(top_line - 0.004 * Hc, top_line + 0.006 * Hc, ys)
    # stepped flat base: 3-4 levels, each skirt lobe's bottom at its own height
    bn = _noise(ww, 1, 0.3 * Hc, seed + 9, 2)[0][None, :]
    steps_x = np.array([-0.8, -0.38, 0.02, 0.36, 0.7, 1.2]) + LEAN * 0.05
    steps_h = np.array([0.03, 0.012, 0.0, 0.018, 0.034, 0.05])
    base_off = np.interp(xr[0], steps_x, steps_h).astype(F32)[None, :]
    base_off = cv2.GaussianBlur(base_off, (0, 0), 0.02 * Hc)
    cut = base_y - Hc * (base_off + 0.004 * bn)
    basecut = _ss(cut + 0.005 * Hc, cut - 0.006 * Hc, ys)
    # anvil tail: the downwind end of the crown shreds into fibrous streaks (no smooth capsule end)
    tn = _noise(ww, hh, 0.03 * Hc, seed + 71, 3, stretch=6.0, angle=3)
    tailw = _ss(0.42, 0.75, xr - 0.1 * LEAN) * _ss(0.75, 0.84, sh)
    shred = 1 - tailw * (1 - _ss(-0.25 + 0.4 * _ss(0.5, 0.95, xr), 0.2 + 0.4 * _ss(0.5, 0.95, xr), tn))
    cut_all = topcut * basecut * shred
    cov = cov * cut_all
    mcov = [m * cut_all for m in mcov]
    # fine cauliflower buds on the lit silhouette
    buds = _cauliflower(rng, cov, x0, y0, Hc, Ld, base_y)
    bcov = np.zeros((hh, ww), F32)
    for lb in buds:
        bb = _bbox(lb, x0, y0, ww, hh)
        if bb is None:
            continue
        a, _, _, _ = _ell(X, Y, lb, bb)
        np.maximum(bcov[bb], a, out=bcov[bb])
    bcov = bcov * topcut
    cov = np.maximum(cov, bcov)
    mk = cov > 0.5
    mkf = mk.astype(F32)
    # ---------------- big form value G: dome of the whole silhouette lit from the upper left +
    # across-the-mass position (down-sun lower right deep) + height
    hgt = cv2.GaussianBlur(mkf, (0, 0), 0.06 * Hc) * 0.5 + cv2.GaussianBlur(mkf, (0, 0), 0.16 * Hc) * 0.5
    gy_, gx_ = np.gradient(hgt * 0.3 * Hc)
    nz = np.sqrt(gx_ ** 2 + gy_ ** 2 + 0.5 ** 2)
    dome = (-gx_ * Ld[0] - gy_ * Ld[1]) / nz
    # across position relative to the (leaning) column
    axl = cx + (0.1 * np.clip(sh, 0, 1) + 0.1) * Hc
    u = (xs - axl) / (0.5 * Hc)
    gn = _noise(ww, hh, 0.15 * Hc, seed + 11, 3)
    G = (0.41 - 0.27 * np.clip(u, -1.2, 1.5) + 0.22 * dome + 0.16 * (np.clip(sh, 0, 1) - 0.5) - 0.1 * _ss(0.2, 0.0, sh)
         + 0.05 * gn)
    # the core shadow: deepest in the lower-right third
    core = np.exp(-(((xs - (cx + 0.36 * Hc)) / (0.28 * Hc)) ** 2 + ((sh - 0.32) / 0.26) ** 2))
    G = (G - 0.3 * core).astype(F32)
    # ---------------- body under everything (seen through the lost undersides): a half step darker on
    # the lit side, a full step darker in the shade half
    shx = _ss(-0.3, 0.9, u)
    V = (G - 0.2 - 0.14 * shx).astype(F32)
    Bm = np.zeros((hh, ww), F32)                     # warm bounce weight
    L3 = np.array([Ld[0] * 0.8, Ld[1] * 0.8, 0.45], F32)
    L3 /= np.linalg.norm(L3)

    def _normals(mm, sig_big, sig_small, k_big, k_small):
        mb1 = cv2.GaussianBlur(mm, (0, 0), sig_big)
        mb2 = cv2.GaussianBlur(mm, (0, 0), sig_small)
        g1y, g1x = np.gradient(mb1 * k_big)
        g2y, g2x = np.gradient(mb2 * k_small)
        nx_, ny_ = -(g1x + g2x), -(g1y + g2y)
        nl = np.sqrt(nx_ ** 2 + ny_ ** 2 + 0.55 ** 2)
        lam = (nx_ * L3[0] + ny_ * L3[1] + 0.55 * L3[2]) / nl
        return lam, nx_ / nl, ny_ / nl

    # ---------------- MAJOR masses (groups), back to front: each one value shape with its own lambert
    # (the terminator wraps each mass), a crisp lit top and a broad LOST underside (down-facing part fades
    # into the body value beneath it)
    ngrp = max(M['grp'] for M in masses) + 1
    for g in range(ngrp):
        mem = [i for i, M in enumerate(masses) if M['grp'] == g]
        gm = np.zeros((hh, ww), F32)
        for i in mem:
            np.maximum(gm, mcov[i], out=gm)
        ys_, xs_ = np.nonzero(gm > 0.01)
        if len(ys_) == 0:
            continue
        pad = int(0.08 * Hc)
        by0, by1 = max(ys_.min() - pad, 0), min(ys_.max() + pad, hh)
        bx0, bx1 = max(xs_.min() - pad, 0), min(xs_.max() + pad, ww)
        sl = (slice(by0, by1), slice(bx0, bx1))
        mm = gm[sl]
        gh = (ys_.max() - ys_.min()) + 1.0
        lam, nxn, nyn = _normals(mm, 0.22 * gh, 0.05 * gh, 0.9 * gh, 0.2 * gh)
        sx = float(np.mean([masses[i]['mx'] for i in mem]))
        s_g = float(np.mean([masses[i]['s'] for i in mem]))
        shade_side = float(np.clip((sx - 0.05) / 0.45, 0, 1))
        kl = 0.6 - 0.3 * shade_side
        Vg = G[sl] + kl * (lam - 0.15)
        # lost underside: where the mass turns down, it goes transparent (soft, brush-broken)
        brk = 0.25 * _noise(bx1 - bx0, by1 - by0, 0.06 * Hc, seed + 50 + g, 2)
        gy_c = 0.5 * (ys_.min() + ys_.max()) + y0
        vv = (Y[sl] - gy_c) / (0.5 * gh)
        fade = _ss(0.45, -0.45, 0.55 * nyn + 0.45 * vv + brk)
        if masses[mem[0]]['kind'] == 'skirt':
            fade = np.maximum(fade, 0.6)
        a = mm * fade
        V[sl] = V[sl] * (1 - a) + Vg * a
        # warm bounce on the down-facing undersides of the low shaded masses
        if s_g < 0.6 and g != 0:
            bw = np.clip(nyn, 0, 1) * mm * (0.4 + 0.6 * shade_side) * _ss(0.75, 0.2, s_g)
            Bm[sl] = np.maximum(Bm[sl], bw)
    # ---------------- secondary masses inside each group: crisp lit top rim, transparent lower half
    for mi, M in enumerate(masses):
        if len([1 for N in masses if N['grp'] == M['grp']]) < 2:
            continue
        m = mcov[mi]
        pad = 1.3
        bx0 = int(max(M['x'] - M['rx'] * pad - x0, 0))
        bx1 = int(min(M['x'] + M['rx'] * pad - x0 + 1, ww))
        by0 = int(max(M['y'] - M['ry'] * 1.8 - y0, 0))
        by1 = int(min(M['y'] + M['ry'] * 1.5 - y0 + 1, hh))
        if bx1 <= bx0 or by1 <= by0:
            continue
        sl = (slice(by0, by1), slice(bx0, bx1))
        mm = m[sl]
        if mm.max() < 0.01:
            continue
        lam, nxn, nyn = _normals(mm, 0.3 * M['ry'], 0.1 * M['ry'], 1.2 * M['ry'], 0.3 * M['ry'])
        shade_side = float(np.clip((M['mx'] - 0.05) / 0.45, 0, 1))
        vv = (Y[sl] - M['y']) / M['ry']
        fade = _ss(0.1, -0.6, vv + 0.3 * _noise(bx1 - bx0, by1 - by0, 0.1 * Hc, seed + 150 + mi, 2))
        a = mm * fade * (0.85 - 0.35 * shade_side)
        Vm = V[sl] + (0.2 - 0.08 * shade_side) * (lam - 0.1)
        V[sl] = V[sl] * (1 - a) + Vm * a
    # ---------------- secondary heads: lit cap + lost underside, only strong on the lit crowns
    for lb in lobes:
        if lb.lvl != 1:
            continue
        M = masses[lb.mass]
        bb = _bbox(lb, x0, y0, ww, hh, 1.1)
        if bb is None:
            continue
        a, uu, vv, q = _ell(X, Y, lb, bb)
        a = a * mcov[lb.mass][bb]
        f = uu * Ld[0] + vv * Ld[1]
        shade_side = float(np.clip((M['mx'] - 0.1 * M['s'] - 0.05) / 0.45, 0, 1))
        ks = 0.2 * (1 - 0.6 * shade_side)
        Vl = V[bb] + ks * (f + 0.1) + 0.02 * lb.jit
        up = -vv * 0.9 + f * 0.3
        la = a * _ss(-0.15, 0.55, up) * (0.9 - 0.4 * shade_side)
        V[bb] = V[bb] * (1 - la) + Vl * la
    # ---------------- fine buds: tiny lit tops on the silhouette (a few px of crisp cream)
    for lb in buds:
        bb = _bbox(lb, x0, y0, ww, hh, 1.1)
        if bb is None:
            continue
        a, uu, vv, q = _ell(X, Y, lb, bb)
        f = uu * Ld[0] + vv * Ld[1]
        Vl = np.maximum(V[bb], G[bb]) + 0.14 * f + 0.02 * lb.jit
        la = a * _ss(-0.2, 0.5, -vv * 0.8 + f * 0.4)
        V[bb] = V[bb] * (1 - la) + Vl * la
    # ---------------- value -> 4 painted steps (soft-snapped, brush-warped borders)
    wn = _noise(ww, hh, 0.04 * Hc, seed + 5, 3) * 0.045 + _noise(ww, hh, 0.012 * Hc, seed + 6, 2, stretch=3, angle=-25) * 0.012
    Vw = V + wn
    steps = [(0.0, (0.5, 0.57, 0.71)), (0.14, (0.56, 0.627, 0.753)), (0.28, (0.64, 0.7, 0.81)),
             (0.4, (0.725, 0.776, 0.863)), (0.5, (0.83, 0.855, 0.9)), (0.6, (0.94, 0.935, 0.925)),
             (0.72, (1.0, 0.973, 0.925)), (0.88, (1.04, 1.01, 0.95))]
    lv = np.array([s_[0] for s_ in steps], F32)
    Vs = np.zeros_like(Vw)
    for i in range(1, len(lv)):
        mid = 0.5 * (lv[i - 1] + lv[i])
        Vs = Vs + (lv[i] - lv[i - 1]) * _ss(mid - 0.03, mid + 0.03, Vw)
    Vs = Vs + lv[0]
    Vf = np.clip(0.7 * Vw + 0.3 * Vs, 0, 1)
    cs = np.array([s_[1] for s_ in steps], F32)
    col = np.stack([np.interp(Vf, lv, cs[:, k]) for k in range(3)], -1).astype(F32)
    # warm reflected light (peach) inside the core shadow
    Bm = cv2.GaussianBlur(Bm, (0, 0), 0.012 * Hc) * _ss(0.5, 0.3, Vf)
    col = col + (_c((0.93, 0.8, 0.74)) - col) * (0.6 * np.clip(Bm * 1.4, 0, 1))[..., None]
    # base band: a flat, darker blue-grey underside
    bb_ = _ss(0.06, 0.0, sh - base_off)
    col = col + (_c((0.58, 0.64, 0.76)) - col) * (0.55 * bb_)[..., None]
    # aerial perspective on the low tower (slight)
    hzk = _ss(0.35, 0.0, sh) * 0.14
    col = col + (_c((0.78, 0.86, 0.96)) - col) * hzk[..., None]
    # dry-brush grain
    st = _noise(ww, hh, 0.012 * Hc, seed + 41, 2, stretch=5.0, angle=-35)
    col = col * (1 + 0.012 * st[..., None])
    # ---------------- shade-side silhouette: the edge turns toward the sky (thin vapour lit by the
    # blue sky + forward scatter), never a dark contour line
    din = cv2.distanceTransform((cov > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    edge = np.exp(-din / (0.018 * Hc))
    shd_e = 1 - _ss(0.4, 0.6, cv2.GaussianBlur(V, (0, 0), 0.02 * Hc))
    ek = np.clip(edge * shd_e * _ss(0.03, 0.12, sh), 0, 1) * 0.55
    col = col + (_c((0.74, 0.81, 0.93)) - col) * ek[..., None]
    # ---------------- alpha: crisp on the lit silhouette, soft / lost on the shade side
    litness = _ss(0.45, 0.62, cv2.GaussianBlur(V, (0, 0), 0.01 * Hc))
    Asoft = cv2.GaussianBlur(cov, (0, 0), 0.01 * Hc)
    Asoft = np.minimum(np.clip(Asoft * 1.2, 0, 1), cv2.dilate(cov, np.ones((3, 3), np.uint8)))
    Asoft = np.maximum(Asoft * 0.9, cov * _ss(0.5, 0.95, Asoft))
    litx = cv2.dilate(litness, np.ones((7, 7), np.uint8))
    A = cov * litx + Asoft * (1 - litx)
    # base: lower edge sinks softly into the haze
    A = A * (1 - 0.35 * _ss(0.035, -0.01, sh - base_off))
    # ---------------- under-base: rain-shaft / haze streaks hanging from the base (soft, vertical)
    below = (ys - cut) / Hc                                          # >0 under the base
    colbase = cv2.GaussianBlur(cov[np.clip((cut - y0 - 0.02 * Hc).astype(int), 0, hh - 1), np.arange(ww)[None, :]],
                               (0, 0), 0.02 * Hc)[0][None, :]
    strk = _noise(ww, hh, 0.06 * Hc, seed + 61, 3, stretch=6.0, angle=84)
    strk = _ss(-0.1, 0.7, strk)
    shaft = np.clip(colbase, 0, 1) * np.exp(-np.clip(below, 0, None) / 0.07) * (below > -0.01)
    sa = np.clip(shaft * (0.22 + 0.22 * strk) * _ss(-0.01, 0.01, below), 0, 0.6)
    scol = _c((0.66, 0.73, 0.86))
    Ao = A + sa * (1 - A)
    col = (col * A[..., None] + scol * (sa * (1 - A))[..., None]) / np.maximum(Ao, 1e-4)[..., None]
    A = Ao
    # ---------------- sheared cirrus veil trailing right off the anvil top
    n1 = _noise(ww, hh, 0.06 * Hc, seed + 81, 4, stretch=7.0, angle=4)
    n2 = _noise(ww, hh, 0.015 * Hc, seed + 82, 3, stretch=9.0, angle=4)
    fib = _ss(0.0, 0.6, 0.65 * n1 + 0.35 * n2)
    tl_y = top_line + 0.03 * Hc
    band = np.exp(-((ys - tl_y) / (0.035 * Hc)) ** 2)
    where = _ss(0.35, 0.65, xr) * _ss(1.15, 0.8, xr)
    wa = np.clip(band * fib * where * 0.75, 0, 1) * (1 - A)
    wcol = _c((0.95, 0.955, 0.975))
    Ao = A + wa
    col = (col * A[..., None] + wcol * wa[..., None]) / np.maximum(Ao, 1e-4)[..., None]
    A = Ao
    out = np.zeros((ph, pw, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = np.clip(A, 0, 1)
    out[..., :3] = K3._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * W))
    if dbg is not None:
        dbg.update(V=V, G=G, cov=cov, box=(x0, y0, x1, y1))
    return out


def rim_light(P, Hc, sun, W, reach=0.3):
    """Backlit crown: the silhouette nearest the sun catches a thin warm rim (kept narrow so the lit
    silhouette stays a crisp readable edge)."""
    h, w = P.shape[:2]
    sc = W / 1920.0
    r = int(reach * 2.2 * Hc)
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
    rw = (1.2 + 3.0 * near) * sc + 0.6
    band = (1 - _ss(rw * 0.5, rw * 1.3, ins)) * (m > 0)
    k = np.clip(band * near, 0, 1)
    c += (_c((1.45, 1.36, 1.15)) - c) * k[..., None]
    P[y0:y1, x0:x1, :3] = c
    return P


def hero(pw, ph, cx, base_y, Hc, W, hz_y=None, seed=4, dbg=None):
    """Returns (rgba plate (ph, pw, 4), sun position in plate px). The sun sits just behind the crown's
    up-sun shoulder."""
    P = paint(pw, ph, cx, base_y, Hc, W, seed=seed, dbg=dbg)
    a_ = P[..., 3] > 0.5
    rows = np.nonzero(a_.any(1))[0]
    ytop = rows[0]
    xs_top = np.nonzero(a_[ytop + int(0.012 * Hc)])[0]
    sun = np.array([xs_top.mean() - 0.12 * Hc, ytop + 0.035 * Hc], np.float32)
    col = a_[:, int(sun[0])]
    yy = np.nonzero(col)[0]
    if len(yy):
        sun[1] = yy[0] + 0.02 * Hc
    P = rim_light(P, Hc, sun, W)
    return P, sun


if __name__ == '__main__':
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib import core as C, sky as S
    W, H = (1920, 1080) if 'full' in sys.argv else (960, 540)
    pw, ph = int(W * 1.12), int(H * 1.26)
    mg = 0.03 * H
    base_y = mg + 0.845 * H
    top_y = mg + 0.15 * H
    Hc = (base_y - top_y) / 1.01
    t0 = time.time()
    dbg = {}
    P, sun = hero(pw, ph, (pw - W) / 2 + 0.38 * W, base_y, Hc, W, seed=4, dbg=dbg)
    print('hero', time.time() - t0, sun)
    sky = S.sky_gradient(pw, ph, dict(stops=[(0.0, '#0a45b8'), (0.16, '#155dcd'), (0.34, '#2e80de'), (0.5, '#58a2e9'),
                                             (0.64, '#8ec4f0'), (0.78, '#b8dcf6'), (0.9, '#d6ecf9'), (1.0, '#ebf8fc')],
                                      sun_glow='#fff6e0', sun_glow_amt=0.3, below='#eef9fc'), horizon=0.8)[..., :3]
    img = sky * (1 - P[..., 3:4]) + P[..., :3] * P[..., 3:4]
    ox = (pw - W) // 2
    img = img[int(mg):int(mg) + H, ox:ox + W]
    tag = [a for a in sys.argv[1:] if a != 'full']
    C.save_png(os.path.join('out', 'compare', 'p7_%s.png' % (tag[0] if tag else 'a')), np.clip(img, 0, 1))
