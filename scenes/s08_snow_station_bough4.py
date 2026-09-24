"""Painted foreground fir for s08_snow_station (v4): layered, drooping, snow-laden boughs.

Round-3 repaint of the framing conifer on the left edge, the way a background artist lays in a snowy
fir: overlapping tiers of drooping boughs (irregular spacing, some hazy back boughs filling the dark
gaps so they read as depth instead of a ladder), each bough a needle mass built from hundreds of
brushed needle strokes (teal-lit top, blue-green body, blue bounce underside, ragged hanging fringe).

Snow sits ONLY on the upper surface of each bough and follows its sagging curve: loads of very
different length and thickness (a long heavy blanket on one bough, a thin broken dusting on the next,
a stubby cushion near the trunk), some loads bridging two or three tiers near the trunk where the
boughs crowd together. Painted values: a lit top plane (warm amber near the platform lamp), a crisp
jagged step to a blue-violet shadow face lit by bounce, a deeper violet belly, a thin bright rim and a
few stable ice glints; needle tips break both the upper and the lower snow edges.
Deterministic per seed."""
import math

import numpy as np
import cv2

from s08_snow_station_bough3 import _poly_mask, _smooth, _sstep, _over, _ip, PAL


SNOW_TOP = np.array([0.8, 0.85, 1.04], np.float32)      # sky-lit top plane
SNOW_FACE = np.array([0.36, 0.38, 0.68], np.float32)      # blue-violet shadow face (bounce-lit)
SNOW_BELLY = np.array([0.17, 0.16, 0.42], np.float32)    # deep violet underside
RIM = np.array([0.86, 0.94, 1.2], np.float32)
AMBER = np.array([1.0, 0.68, 0.34], np.float32)


def _strokes_mask(segs, X0, Y0, w, h, ss=3):
    """Anti-aliased brush strokes: segs = [(x0, y0, x1, y1, width)] in px."""
    m = np.zeros((h * ss, w * ss), np.uint8)
    for (xa, ya, xb, yb, wd) in segs:
        p0 = (int((xa - X0) * ss), int((ya - Y0) * ss))
        p1 = (int((xb - X0) * ss), int((yb - Y0) * ss))
        cv2.line(m, p0, p1, 255, max(int(round(wd * ss)), 1), cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


def _curves_mask(curves, X0, Y0, w, h, ss=3):
    """Anti-aliased brushed polylines: curves = [(points (k, 2), width)] in px."""
    m = np.zeros((h * ss, w * ss), np.uint8)
    for (P, wd) in curves:
        q = ((np.asarray(P, np.float64) - [X0, Y0]) * ss).astype(np.int32)
        cv2.polylines(m, [q], False, 255, max(int(round(wd * ss)), 1), cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


def _bumps(rng, n, t, k0, k1, w0, w1, a0, a1):
    out = np.zeros_like(t)
    for _ in range(int(rng.integers(k0, k1 + 1))):
        c = rng.uniform(0.0, 1.0)
        out += rng.uniform(a0, a1) * np.exp(-((t - c) / rng.uniform(w0, w1)) ** 2)
    return out


class _Bough:
    """Geometry of one bough (px): centreline, needle band edges, thickness."""

    def __init__(self, u0, v0, L, s, rng, kind, dirn):
        self.kind, self.dirn, self.L, self.s = kind, dirn, L, s
        n = 180
        self.n = n
        t = np.linspace(0, 1, n)
        self.t = t
        a0 = rng.uniform(-0.18, 0.15) if kind != 'fore' else rng.uniform(0.02, 0.25)
        droop = rng.uniform(0.12, 0.7)
        curl = rng.uniform(0.04, 0.3)
        self.x = u0 + dirn * L * t * (1 - 0.2 * droop * t)
        y = v0 + L * a0 * t + L * droop * t ** 2 + L * curl * np.clip(t - 0.6, 0, None) ** 2 * 5
        y = y + L * 0.015 * np.sin(t * rng.uniform(5, 10) + rng.uniform(0, 6)) * t
        self.y = y
        thick = rng.uniform(0.1, 0.15) if kind != 'fore' else rng.uniform(0.13, 0.18)
        self.T = L * thick * (1 - 0.8 * t) ** 1.1 + 1.4 * s
        self.yu = y - self.T * 0.35
        self.yl = y + self.T * 0.6
        self.snow_top = None       # filled when snow is painted (for bridges)


def bough_fir(W, H, trunk_u, top_v, base_v, reach, s, seed=0, lamps=(), fog=(0.2, 0.33, 0.56), avoid=None):
    """Paint the fir into a fresh (H, W) RGBA plate (same contract as bough3.bough_fir)."""
    rng = np.random.default_rng(seed)
    rgb = np.zeros((H, W, 3), np.float32)
    a = np.zeros((H, W), np.float32)
    fog = np.asarray(fog, np.float32)
    lamps = [(float(u), float(v), np.asarray(c, np.float32), float(r), float(k)) for (u, v, c, r, k) in lamps]

    def lamp_light(xx, yy):
        out = np.zeros(np.broadcast(xx, yy).shape + (3,), np.float32)
        for (u, v, c, r, k) in lamps:
            d2 = ((xx - u) ** 2 + (yy - v) ** 2) / (r * r)
            out += (k * np.exp(-d2))[..., None] * c
        return out

    # directional brush noise (for needle / snow texture)
    nz = rng.random((H // 2 + 1, W // 2 + 1)).astype(np.float32)
    k = max(int(5 * s) | 1, 3)
    ker = np.zeros((k, k), np.float32)
    for i in range(k):
        ker[i, min(int(i * 0.45 + k * 0.3), k - 1)] = 1.0
    ker /= ker.sum()
    st = cv2.resize(cv2.filter2D(nz, -1, ker), (W, H), interpolation=cv2.INTER_LINEAR)
    st = np.clip((st - st.mean()) / (st.std() + 1e-6), -2.2, 2.2).astype(np.float32)

    span = base_v - top_v
    # ---- deep interior: dark blue-green depth around the trunk, a little wider toward the base
    vs = np.linspace(top_v, base_v + 0.4 * span, 200)
    rr = np.array([reach(v) for v in vs]) * 0.42
    rr = rr * (1 + 0.35 * np.sin(vs * 0.045 / s + 1.0) * np.sin(vs * 0.017 / s)) + rng.normal(0, 1, len(vs)) * 0.08 * rr
    rr = np.maximum(rr, 5 * s)
    poly = np.concatenate([np.stack([trunk_u + rr, vs], 1), np.stack([trunk_u - rr[::-1], vs[::-1]], 1)])
    X0, Y0 = max(int(poly[:, 0].min()) - 2, 0), max(int(poly[:, 1].min()) - 2, 0)
    w2, h2 = min(int(poly[:, 0].max()) + 3, W) - X0, min(int(poly[:, 1].max()) + 3, H) - Y0
    if w2 > 0 and h2 > 0:
        m = _poly_mask([poly], X0, Y0, w2, h2, ss=2)
        m = cv2.GaussianBlur(m, (0, 0), 6.0 * s + 0.3)
        col = (PAL['NEED_LO'] * 0.55 + fog * 0.08) * (1 + 0.06 * st[Y0:Y0 + h2, X0:X0 + w2, None])
        _over(rgb, a, X0, Y0, m * 0.96, col.astype(np.float32))

    # ---- trunk
    tv0 = top_v + 0.03 * span
    tv1 = base_v + 0.45 * span
    n = 60
    tt = np.linspace(0, 1, n)
    wd = (3.0 + 21.0 * tt ** 0.8) * s
    xc = trunk_u + 5 * s * np.sin(tt * 5.0 + 1.0) * tt
    yv = tv0 + (tv1 - tv0) * tt
    poly = np.concatenate([np.stack([xc - wd, yv], 1), np.stack([xc[::-1] + wd[::-1], yv[::-1]], 1)])
    X0, Y0 = max(int(poly[:, 0].min()) - 2, 0), max(int(poly[:, 1].min()) - 2, 0)
    X1, Y1 = min(int(poly[:, 0].max()) + 3, W), min(int(poly[:, 1].max()) + 3, H)
    if X1 > X0 and Y1 > Y0:
        w2, h2 = X1 - X0, Y1 - Y0
        m = _poly_mask([poly], X0, Y0, w2, h2, ss=3)
        yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
        xx = np.arange(X0, X1, dtype=np.float32)[None, :]
        cx_ = np.interp(yy[:, 0], yv, xc)[:, None]
        wx_ = np.interp(yy[:, 0], yv, wd)[:, None]
        rel = np.clip((xx - cx_) / np.maximum(wx_, 1e-3), -1, 1)
        bark = cv2.resize(rng.random((max(h2 // int(14 * s + 1), 2), max(w2 // int(3 * s + 1), 2))).astype(np.float32),
                          (w2, h2), interpolation=cv2.INTER_CUBIC)
        base = np.array([0.085, 0.06, 0.075], np.float32)
        col = base * (0.65 + 0.7 * bark[..., None]) * (1 - 0.4 * np.abs(rel)[..., None])
        col = col + np.array([0.2, 0.11, 0.05], np.float32) * np.clip(rel, 0, 1)[..., None] ** 2 * 0.5
        blot = cv2.resize(rng.random((max(h2 // int(40 * s + 1), 2), 2)).astype(np.float32), (w2, h2), interpolation=cv2.INTER_CUBIC)
        snow = (_sstep(-0.2, -0.75, rel) * _sstep(0.45, 0.7, bark) * _sstep(0.55, 0.75, blot)).astype(np.float32)
        col = col * (1 - snow[..., None] * 0.9) + SNOW_FACE * 1.1 * snow[..., None] * 0.9
        _over(rgb, a, X0, Y0, m, col.astype(np.float32))

    # ---- bough layout: irregular tiers, each a main bough + optional back / fore boughs
    items = []
    v = top_v + 0.015 * span
    tier = 0
    Lmax = max(reach(base_v), 1)
    while v < base_v + 0.12 * span:
        L = reach(v)
        spacing = max(0.06 * span * (0.45 + 0.55 * min(L / Lmax, 1)), 7 * s) * rng.choice([rng.uniform(0.4, 0.7), rng.uniform(0.9, 1.7)])
        items.append((tier * 10.0, trunk_u + rng.uniform(-0.02, 0.03) * L, v, L * rng.uniform(0.68, 1.0), 0.0, tier,
                      'main', int(rng.integers(1 << 30)), 1.0))
        # hazy back boughs filling the gap below (depth, not a ladder)
        for _ in range(int(rng.integers(1, 3))):
            dv = rng.uniform(0.25, 0.8) * spacing
            items.append((tier * 10.0 - 3.0 + rng.uniform(-1, 1), trunk_u + rng.uniform(-0.02, 0.02) * L, v + dv,
                          L * rng.uniform(0.45, 0.85), rng.uniform(0.25, 0.45), tier, 'back',
                          int(rng.integers(1 << 30)), 1.0 if rng.random() < 0.8 else -1.0))
        if rng.random() < 0.5:
            dv = rng.uniform(0.3, 0.7) * spacing
            items.append((tier * 10.0 + 7.0, trunk_u, v + dv, L * rng.uniform(0.25, 0.5), 0.0, tier, 'fore',
                          int(rng.integers(1 << 30)), 1.0))
        v += spacing
        tier += 1
    items.sort(key=lambda q: q[0])

    geo = {}
    for it in items:
        (_, u0, v0, Lb, haze, tr, kind, sd, dirn) = it
        if Lb < 8 * s:
            continue
        if avoid is not None and dirn > 0:
            # keep the bough tips off a region (the station-name board): (x_max, v0, v1) in px
            xm, va, vb = avoid
            if va - 0.3 * Lb < v0 < vb:
                Lb = min(Lb, max((xm - u0) / 0.92, 8 * s))
        geo[id(it)] = _Bough(u0, v0, Lb, s, np.random.default_rng(sd), kind, dirn)
        geo[id(it)].lamps = lamps

    # bridges: a main bough's load spills down onto the next 1-2 tiers near the trunk
    mains = [it for it in items if it[6] == 'main' and id(it) in geo]
    bridge_after = {}
    brng = np.random.default_rng(seed + 7)
    i = 0
    while i < len(mains) - 1:
        if brng.random() < 0.4:
            span_n = 2 if brng.random() < 0.3 and i + 2 < len(mains) else 1
            bridge_after[id(mains[i])] = [mains[i + q] for q in range(1, span_n + 1)]
            i += span_n + 1
        else:
            i += 1

    for it in items:
        if id(it) not in geo:
            continue
        g = geo[id(it)]
        (_, u0, v0, Lb, haze, tr, kind, sd, dirn) = it
        if id(it) in bridge_after:
            # the spanning load sits behind this bough's fringe and tucks under the lower tier(s)
            lows = [geo[id(q)] for q in bridge_after[id(it)]]
            _paint_bridge(rgb, a, g, lows, np.random.default_rng(sd + 2), st, lamp_light, W, H)
        brng_i = np.random.default_rng(sd + 1)
        _paint_needles(rgb, a, g, brng_i, st, lamp_light, fog, haze, W, H)
        _paint_snow(rgb, a, g, brng_i, st, lamp_light, fog, haze, W, H, tr)
    return rgb, a


def _box(polys_or_pts, W, H, pad_top=0, pad_bot=0, pad=4):
    allp = np.concatenate(polys_or_pts, 0)
    x0 = int(math.floor(allp[:, 0].min())) - pad
    y0 = int(math.floor(allp[:, 1].min())) - pad - pad_top
    x1 = int(math.ceil(allp[:, 0].max())) + pad
    y1 = int(math.ceil(allp[:, 1].max())) + pad + pad_bot
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    return X0, Y0, X1, Y1


def _paint_needles(rgb, a, g, rng, st, lamp_light, fog, haze, W, H):
    """Needle mass: a core band plus hundreds of brushed needle strokes (hanging fringe below, short
    tufts along the top), coloured by position across the bough."""
    NEED, NEED_HI, NEED_LO = PAL['NEED'], PAL['NEED_HI'], PAL['NEED_LO']
    s, n, dirn = g.s, g.n, g.dirn
    x, y, T, yu, yl = g.x, g.y, g.T, g.yu, g.yl
    body = np.concatenate([np.stack([x, yu + T * 0.1], 1), np.stack([x[::-1], yl[::-1] - T[::-1] * 0.1], 1)])
    curves = []          # (points, width, group) : group 0 normal, 1 light, 2 dark
    nstr = int(90 + g.L / (1.0 * s))
    for _ in range(nstr):
        ti = rng.uniform(0.0, 1.0) ** 0.85
        j = min(int(ti * (n - 1)), n - 1)
        # hanging branchlet: leaves the band outward, bends down under its own weight
        ln = T[j] * rng.uniform(0.35, 1.1) * (0.6 + 0.8 * math.sin(math.pi * min(ti + 0.1, 1.0))) + 1.5 * s
        ang = math.radians(rng.uniform(15, 75))
        bend = math.radians(rng.uniform(20, 60))
        bx, by = x[j], y[j] + T[j] * rng.uniform(-0.1, 0.4)
        mx_, my_ = bx + dirn * math.cos(ang) * ln * 0.5, by + math.sin(ang) * ln * 0.5
        a2 = min(ang + bend, math.radians(110))
        ex, ey = mx_ + dirn * math.cos(a2) * ln * 0.5, my_ + math.sin(a2) * ln * 0.5
        r = rng.random()
        curves.append((np.array([[bx, by], [mx_, my_], [ex, ey]]), max(rng.uniform(1.0, 2.4) * s, 0.8),
                       1 if r < 0.22 else (2 if r < 0.5 else 0)))
    for _ in range(int(30 + g.L / (5 * s))):
        ti = rng.uniform(0.02, 0.98)
        j = min(int(ti * (n - 1)), n - 1)
        ln = T[j] * rng.uniform(0.2, 0.45) + 1.2 * s
        ang = math.radians(rng.uniform(-60, -15))
        bx, by = x[j], yu[j] + T[j] * 0.25
        curves.append((np.array([[bx, by], [bx + dirn * math.cos(ang) * ln, by + math.sin(ang) * ln]]),
                       max(1.0 * s, 0.8), 1))
    for _ in range(9):
        j = n - 1 - int(rng.integers(0, 14))
        ln = T[0] * rng.uniform(0.2, 0.45)
        ang = rng.uniform(-0.3, 1.4)
        curves.append((np.array([[x[j], y[j]], [x[j] + dirn * math.cos(ang) * ln, y[j] + math.sin(ang) * ln]]),
                       max(1.3 * s, 0.8), 0))
    pts = [body] + [c[0] for c in curves]
    X0, Y0, X1, Y1 = _box(pts, W, H, pad_top=int(T.max() * 3.2) + 6, pad_bot=int(T.max()) + 4)
    if X1 <= X0 or Y1 <= Y0:
        return
    w, h = X1 - X0, Y1 - Y0
    g.box = (X0, Y0, X1, Y1)
    m = np.maximum(_poly_mask([body], X0, Y0, w, h, ss=3), _curves_mask([(c[0], c[1]) for c in curves], X0, Y0, w, h))
    light = _curves_mask([(c[0], c[1] * 0.8) for c in curves if c[2] == 1], X0, Y0, w, h)
    dark = _curves_mask([(c[0], c[1]) for c in curves if c[2] == 2], X0, Y0, w, h)
    xx = np.arange(X0, X1, dtype=np.float32)[None, :]
    yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
    ycx = _ip(xx[0], x, y)[None, :]
    Tx = _ip(xx[0], x, T)[None, :]
    rel = (yy - ycx) / np.maximum(Tx, 1e-3)
    # painted tier MASS (round 5): one dark blue-green silhouette value per bough, a flat cool moonlit
    # plane on the upper surface split by a soft-but-readable step, a slightly bluer underside value.
    # Needle strokes only break the SILHOUETTE; inside the mass the texture is suppressed.
    MASS = np.array([0.03, 0.075, 0.1], np.float32)
    MOON = np.array([0.08, 0.16, 0.23], np.float32)
    UNDER = np.array([0.035, 0.06, 0.12], np.float32)
    jag = 0.12 * np.sin(xx * 0.11 / s + x[0]) + 0.08 * np.sin(xx * 0.29 / s + y[0])
    topk = _sstep(0.05, -0.12, rel + jag)[..., None]
    undk = _sstep(0.55, 0.8, rel - jag)[..., None]
    fol = MASS * (1 - topk) + MOON * topk
    fol = fol * (1 - undk) + UNDER * undk
    core = cv2.GaussianBlur(m, (0, 0), 2.5 * s + 0.5)
    core = _sstep(0.82, 0.97, core)
    edge = np.clip(m - core, 0, 1)[..., None]            # silhouette zone only
    fol = fol * (1 + 0.05 * st[Y0:Y1, X0:X1, None])
    fol = fol * (1 + edge * (0.45 * light[..., None] * (1 - undk) - 0.3 * dark[..., None]))
    E = lamp_light(xx, yy)
    fol = fol + E * np.array([0.2, 0.12, 0.05], np.float32) * topk * 0.35
    # warm lamp-side rim: only the silhouette pixels whose outward normal faces a lamp
    mb = cv2.GaussianBlur(m, (0, 0), 4.5 * s + 0.5)
    gx = cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(2.2 * s) * 2 + 1, 3),) * 2)
    band = np.clip(m - cv2.erode(m, ker), 0, 1) * _sstep(0.25, 0.5, mb)
    rimw = np.zeros((h, w), np.float32)
    for (lu, lv, lc, lr, lk) in getattr(g, 'lamps', ()):
        dx, dy = lu - xx, lv - yy
        dd = np.sqrt(dx * dx + dy * dy) + 1e-3
        face_ = np.clip((-gx * dx - gy * dy) / (gn * dd), 0, 1)
        rimw += face_ ** 1.5 * lk * np.exp(-(dd / (lr * 1.1)) ** 2)
    rimw = np.clip(rimw * 2.2, 0, 1) * band
    fol = fol * (1 - rimw[..., None] * 0.55) + np.array([0.7, 0.42, 0.17], np.float32) * rimw[..., None] * 0.55
    fol = fol * (1 - haze) + fog * 0.36 * haze
    _over(rgb, a, X0, Y0, m, fol.astype(np.float32))
    g.E = E
    g.fol_box = (X0, Y0, X1, Y1)


def _paint_snow(rgb, a, g, rng, st, lamp_light, fog, haze, W, H, tier):
    """Snow loads lying on the upper surface of the bough, following its curve."""
    if not hasattr(g, 'box'):
        return
    s, n, dirn, t = g.s, g.n, g.dirn, g.t
    x, T, yu = g.x, g.T, g.yu
    # load character per bough: heavy blanket / broken cushions / thin dusting
    r = rng.random()
    if g.kind == 'back':
        mode = 'dust' if r < 0.45 else ('clumps' if r < 0.75 else 'broken')
    else:
        mode = 'blanket' if r < 0.22 else ('broken' if r < 0.55 else ('clumps' if r < 0.82 else 'dust'))
    # per-bough load scale varies a lot (one heavy load, the next a thin crust)
    vscale = float(np.clip(rng.lognormal(0.0, 0.35), 0.5, 1.7))
    Smax = g.L * {'blanket': rng.uniform(0.08, 0.14), 'broken': rng.uniform(0.06, 0.11), 'clumps': rng.uniform(0.07, 0.12),
                  'dust': rng.uniform(0.015, 0.03)}[mode] * vscale + 1.2 * s
    segs = []
    if mode == 'blanket':
        segs.append((rng.uniform(0.02, 0.25), rng.uniform(0.45, 0.95), 1.0))
    elif mode == 'clumps':
        # a row of small separate cushions of very different size along the bough
        tc = rng.uniform(0.0, 0.2)
        while tc < 0.92:
            ln = rng.uniform(0.04, 0.16)
            segs.append((tc, min(tc + ln, 0.97), rng.uniform(0.3, 1.1)))
            tc = tc + ln + rng.uniform(0.03, 0.18)
    else:
        tc = rng.uniform(0.0, 0.15)
        while tc < 0.9:
            ln = rng.uniform(0.06, 0.45)
            segs.append((tc, min(tc + ln, 0.96), rng.uniform(0.25, 1.0) if mode == 'broken' else rng.uniform(0.6, 1.0)))
            tc = tc + ln + rng.uniform(0.02, 0.28)
    X0, Y0, X1, Y1 = g.box
    w, h = X1 - X0, Y1 - Y0
    xx = np.arange(X0, X1, dtype=np.float32)[None, :]
    yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
    E = g.E
    Eh = E.mean(-1, keepdims=True)
    warmk = np.clip(Eh * 1.6, 0, 1)
    sm_all = np.zeros((h, w), np.float32)
    col_all = np.zeros((h, w, 3), np.float32)
    rim_all = np.zeros((h, w), np.float32)
    face_all = np.zeros((h, w), np.float32)
    top_full = np.full(n, np.nan)
    bot_full = np.full(n, np.nan)
    for (ta, tb, big) in segs:
        sel = (t >= ta) & (t <= tb)
        if sel.sum() < 5:
            continue
        tt = (t[sel] - ta) / max(tb - ta, 1e-3)
        xs_, yus, Ts = x[sel], yu[sel], T[sel]
        # thickness: an irregular swell (not a pill): envelope with steep-ish ends + random bumps
        # (cycle 7) a thin SLAB lying along the bough: steep squared ends, near-constant thickness,
        # only small irregular steps - no dome / cushion swell
        env = np.clip(np.sin(np.pi * np.clip(tt, 0, 1)) * 3.0, 0, 1) ** 0.6
        prof = env * (0.8 + 0.5 * _bumps(rng, 0, tt, 2, 4, 0.05, 0.15, 0.2, 0.7))
        prof = prof * (1 - 0.45 * t[sel])                      # thinner toward the tip
        prof = prof + 0.05 * np.sin(tt * rng.uniform(25, 45) + rng.uniform(0, 6)) * env
        prof = _smooth(np.clip(prof, 0, None), 2)
        prof = prof / max(prof.max(), 1e-3)
        Sk = 0.62 * Smax * big * (1 - 0.3 * t[sel].mean()) * float(np.clip((tb - ta) / 0.35, 0.3, 1.0))
        top = yus - Sk * prof * 1.05 + Ts * 0.15
        # crisp, slightly jagged hand-cut top edge (small irregular notches, not a smooth roll)
        zz = rng.uniform(-1, 1, len(tt))
        zz = np.convolve(zz, np.ones(3) / 3, mode='same')
        top = top + zz * 1.3 * s * env
        # bottom follows the bough's upper edge, slumping a little over the front in a few lobes
        slump = _bumps(rng, 0, tt, 1, 4, 0.03, 0.1, 0.2, 0.9)
        bot = yus + Ts * (0.2 + 0.08 * rng.random()) * env ** 0.5 + Sk * 0.2 * np.clip(slump, 0, 1.2) * env + 0.5 * s
        # hard, slightly ragged underside: small chipped steps + a few short drips (not a smooth roll)
        rg = rng.uniform(-1, 1, len(tt))
        rg = np.convolve(rg, np.ones(2) / 2, mode='same')
        drip = (rng.random(len(tt)) < 0.04).astype(np.float64)
        drip = np.convolve(drip, np.ones(3), mode='same') * rng.uniform(1.5, 3.5) * s
        bot = bot + rg * 1.1 * s * env + drip * env
        bot = np.maximum(bot, top + 1.6 * s)
        top_full[sel] = top
        bot_full[sel] = bot
        poly = np.concatenate([np.stack([xs_, top], 1), np.stack([xs_[::-1], bot[::-1]], 1)])
        m = _poly_mask([poly], X0, Y0, w, h, ss=3)
        ttop = _ip(xx[0], xs_, top)[None, :]
        tbot = _ip(xx[0], xs_, bot)[None, :]
        rv = np.clip((yy - ttop) / np.maximum(tbot - ttop, 1e-3), 0, 1)
        ts_ = rng.uniform(0.38, 0.6) if mode != 'dust' else 0.6
        jag = 0.08 * np.sin(xx * (0.08 / s) + rng.uniform(0, 6)) + 0.04 * np.sin(xx * (0.23 / s) + rng.uniform(0, 6))
        # painted, not modelled: ONE flat pale lit plane over ONE flat cool shadow value, split by a
        # crisp hand-drawn (slightly jagged) edge - no bevel gradient, no belly darkening, no crown glow
        # (cycle 7) ONE flat pale-blue plane for the whole slab; the light is carried only by a crisp
        # 2-3 px warm-white top edge (drawn below as the rim), no internal gradient or value split
        body = SNOW_TOP * 0.56 + SNOW_FACE * 0.44
        body = body * rng.uniform(0.97, 1.02) + Eh * AMBER * 0.1
        step = np.zeros(rv.shape + (1,), np.float32)
        col = body * np.ones_like(rv)[..., None]
        col = col * (1 + 0.012 * st[Y0:Y1, X0:X1, None])
        col_all = col_all * (1 - m[..., None]) + col * m[..., None]
        face_all = np.maximum(face_all * (1 - m), step[..., 0] * m)
        sm_all = np.maximum(sm_all, m)
        rm = np.zeros((h * 3, w * 3), np.uint8)
        q = ((np.stack([xs_, top + 1.1 * s], 1) - [X0, Y0]) * 3).astype(np.int32)
        cv2.polylines(rm, [q], False, 255, max(int(round(2.4 * s * 3)), 3), cv2.LINE_AA)
        rim_all = np.maximum(rim_all, cv2.resize(rm.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA) * m)
    g.snow_top = top_full
    if sm_all.max() <= 0:
        return
    # darker needle-mass shadow under each load: the snow shades the needles right beneath it
    # (a soft cool-dark band hugging the load's lower edge, strongest near the trunk), painted onto the
    # existing needles only, so every tier separates from the one below it as depth
    # (round 5) no AO/contact darkening under the loads: the snow's lower edge is lost into the needles
    # falling-powder puffs: a few soft translucent wisps shed from the load's lower edge / tip
    if haze < 0.2 and rng.random() < 0.55:
        pm = np.zeros((h, w), np.float32)
        for _ in range(int(rng.integers(1, 4))):
            j = int(rng.integers(0, n))
            if np.isnan(bot_full[j]):
                continue
            cx_, cy_ = x[j] - X0, bot_full[j] - Y0 + rng.uniform(2, 14) * s
            for q in range(int(rng.integers(3, 8))):
                px_ = cx_ + rng.normal(0, 3.5 * s)
                py_ = cy_ + q * rng.uniform(2.5, 6) * s
                rr_ = rng.uniform(1.2, 3.2) * s * (1 + 0.15 * q)
                if 0 <= px_ < w and 0 <= py_ < h:
                    cv2.circle(pm, (int(px_), int(py_)), max(int(rr_), 1), float(rng.uniform(0.35, 0.8) * (1 - q / 9)), -1, cv2.LINE_AA)
        pm = cv2.GaussianBlur(pm, (0, 0), 1.6 * s + 0.4)
        if pm.max() > 0:
            _over(rgb, a, X0, Y0, np.clip(pm, 0, 0.7), (SNOW_TOP * 0.8 + Eh * AMBER * 0.5).astype(np.float32))
    tf = 1.0 - 0.1 * min(tier / 14.0, 1.0)
    col_all = col_all * tf
    col_all = col_all * (1 - haze * 0.65) + fog * 1.15 * haze * 0.65
    # lost lower edge: the shadow face dissolves into the needle mass (the lit top edge stays crisp)
    # (cycle 7) hard ragged underside instead: keep the anti-aliased polygon edge as painted
    _over(rgb, a, X0, Y0, sm_all, col_all.astype(np.float32))
    # needle tips breaking the snow edges: dark strokes hanging over the lower edge, a few poking up
    segs2 = []
    for _ in range(int(20 + g.L / (3.5 * s))):
        ti = rng.uniform(0.03, 0.97)
        j = min(int(ti * (n - 1)), n - 1)
        if np.isnan(bot_full[j]):
            continue
        # short needle tufts hanging from just inside the lower snow edge (breaks the clean curve)
        yb = bot_full[j] - rng.uniform(0.5, 3.5) * s
        ln = T[j] * rng.uniform(0.2, 0.7) + 2.0 * s
        ang = math.radians(rng.uniform(50, 110))
        segs2.append((x[j], yb, x[j] + dirn * math.cos(ang) * ln, yb + math.sin(ang) * ln, max(rng.uniform(0.9, 1.6) * s, 0.8)))
    for _ in range(int(1 + g.L / (60 * s))):
        ti = rng.uniform(0.05, 0.95)
        j = min(int(ti * (n - 1)), n - 1)
        if np.isnan(top_full[j]) or (yu[j] - top_full[j]) > 6 * s:
            continue                                   # only through thin snow
        ln = rng.uniform(2.0, 4.5) * s
        ang = math.radians(rng.uniform(-80, -40))
        yb = top_full[j] + 1.5 * s
        segs2.append((x[j], yb, x[j] + dirn * math.cos(ang) * ln, yb + math.sin(ang) * ln, max(0.9 * s, 0.8)))
    if segs2:
        pm = _strokes_mask(segs2, X0, Y0, w, h, ss=3)
        ncol = (PAL['NEED'] * 1.1 + E * np.array([0.1, 0.06, 0.03], np.float32)) * (1 - haze) + fog * 0.36 * haze
        _over(rgb, a, X0, Y0, pm * 0.95, ncol.astype(np.float32))
    rim_c = np.array([1.08, 1.04, 1.02], np.float32) * (1 - warmk) + (np.array([0.75, 0.7, 0.66], np.float32) + AMBER * 0.75) * warmk
    _over(rgb, a, X0, Y0, rim_all * (0.95 - 0.55 * haze), rim_c.astype(np.float32))
    # a few stable ice glints on the lit tops (static: part of the painted plate)
    grng = np.random.default_rng(int(g.x[0] * 13 + g.y[0] * 7) & 0x7fffffff)
    ys_, xs2 = np.nonzero((rim_all > 0.5) & (grng.random(rim_all.shape) < 0.01))
    if len(xs2) and haze < 0.2:
        sp = np.zeros((h, w), np.float32)
        sp[ys_, xs2] = 1.0
        sp = np.clip(cv2.GaussianBlur(sp, (0, 0), 0.55 * s + 0.2) * 3.0, 0, 1)
        _over(rgb, a, X0, Y0, sp, (np.array([1.3, 1.35, 1.55], np.float32) + Eh * AMBER).astype(np.float32))


def _paint_bridge(rgb, a, g, lows, rng, st, lamp_light, W, H):
    """Snow load spanning from bough g down over the next tier(s) near the trunk: a single heavy mass
    whose top is g's snow top and whose foot tucks behind the lowest bough's load."""
    s = g.s
    low = lows[-1]
    xa = max(g.x[0], low.x[0]) + g.L * rng.uniform(0.02, 0.06)
    xb_lim = min(g.x[-1], low.x[-1])
    xb = xa + (xb_lim - xa) * rng.uniform(0.35, 0.6)
    if xb - xa < 10 * s:
        return
    xs = np.linspace(xa, xb, 90)
    top_up = _ip(xs, g.x, g.yu) - _ip(xs, g.x, g.T) * 0.2
    foot = _ip(xs, low.x, low.yu) + _ip(xs, low.x, low.T) * 0.3
    q = (xs - xa) / (xb - xa)
    # sloping outer end (the load slumps down toward the lower tier), soft lumps along the top
    env = _sstep(1.0, 0.55, q) * _sstep(0.0, 0.08, q) ** 0.5
    top = top_up + (foot - top_up) * (1 - env) + 0.06 * (foot - top_up) * np.sin(q * rng.uniform(8, 14) + rng.uniform(0, 6)) * env
    if np.all(foot - top < 3 * s) or np.median(foot - top_up) > 0.26 * g.L:
        return
    bot = np.maximum(foot, top + 2 * s)
    poly = np.concatenate([np.stack([xs, top], 1), np.stack([xs[::-1], bot[::-1]], 1)])
    X0, Y0, X1, Y1 = _box([poly], W, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    w, h = X1 - X0, Y1 - Y0
    m = _poly_mask([poly], X0, Y0, w, h, ss=3)
    xx = np.arange(X0, X1, dtype=np.float32)[None, :]
    yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
    ttop = _ip(xx[0], xs, top)[None, :]
    tbot = _ip(xx[0], xs, bot)[None, :]
    rv = np.clip((yy - ttop) / np.maximum(tbot - ttop, 1e-3), 0, 1)
    E = lamp_light(xx, yy)
    Eh = E.mean(-1, keepdims=True)
    warmk = np.clip(Eh * 1.6, 0, 1)
    # the lit cap is thin (the mass mostly faces away from the sky): violet face with vertical
    # slump strokes, deeper toward the foot
    capd = (6 + 4 * np.sin(xx * 0.05 / s)) * s
    cap = np.clip(1 - (yy - ttop) / np.maximum(capd, 1), 0, 1)
    cap = (cap > 0.35).astype(np.float32) * 0.85 + 0.15 * cap
    lit = SNOW_TOP * (1 - warmk) + (SNOW_TOP * 0.55 + AMBER * 0.75 * np.clip(Eh * 1.4, 0, 1.2)) * warmk
    vst = cv2.GaussianBlur(np.random.default_rng(int(xa)).random((h, w)).astype(np.float32), (0, 0),
                           sigmaX=1.2 * s + 0.3, sigmaY=6 * s + 1)
    vst = (vst - vst.mean()) / (vst.std() + 1e-6)
    # uneven drape: a lit ledge where it rests on the bough in between, slump strokes, darker foot
    lv = rng.uniform(0.35, 0.6)
    ledge = np.exp(-((rv - lv - 0.05 * np.sin(xx * 0.06 / s)) / 0.06) ** 2)
    face = SNOW_FACE * 0.8 * (1 + 0.02 * vst[..., None]) + Eh * AMBER * 0.12
    cap = (cap > 0.5).astype(np.float32)
    col = lit * cap[..., None] + face * (1 - cap[..., None])
    _over(rgb, a, X0, Y0, m, col.astype(np.float32))
    rm = np.zeros((h * 3, w * 3), np.uint8)
    qq = ((np.stack([xs, top + 0.6 * s], 1) - [X0, Y0]) * 3).astype(np.int32)
    cv2.polylines(rm, [qq], False, 255, max(int(round(1.2 * s * 3 * 0.5)), 2), cv2.LINE_AA)
    rim = cv2.resize(rm.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA) * m
    _over(rgb, a, X0, Y0, rim * 0.7, (RIM * (1 - warmk) + (RIM * 0.5 + AMBER * 1.1) * warmk).astype(np.float32))
