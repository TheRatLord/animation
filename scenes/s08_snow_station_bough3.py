"""Painted foreground fir for s08_snow_station (v3): irregular, overlapping, drooping snow-laden boughs.

A framing conifer painted the way a background artist does it: a visible tapering trunk with bark and
wind-plastered snow, a sparse dark interior right around it, then boughs of very different length, start
angle, sag and spacing, painted in an interleaved order so some hang in front of the ones below and some
tuck behind the ones above. Each bough is a dark needle mass (teal top, blue-green body, a cool blue
underside lit by bounce from the snow - never black) with a ragged fringe of hanging branchlets, and a
snow load broken into separate lumpy clumps (gaps where the needles show) that drape over the front in
rounded lobes: flat painted values (lit top, crisp step to a blue-violet shadow face, deep shadow under
the lobes), a bright rim on the upper silhouette, warmed near the lamps. Deterministic per seed."""
import math

import numpy as np
import cv2


def _poly_mask(polys, x0, y0, w, h, ss=3):
    m = np.zeros((h * ss, w * ss), np.uint8)
    for p in polys:
        q = ((np.asarray(p, np.float64) - [x0, y0]) * ss).astype(np.int32)
        cv2.fillPoly(m, [q], 255, lineType=cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


def _smooth(x, k):
    if k < 1:
        return x
    ker = np.hanning(2 * k + 3)[1:-1]
    ker /= ker.sum()
    xp = np.pad(x, k, mode='edge')
    return np.convolve(xp, ker, mode='valid')


def _sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _over(rgb, a, x0, y0, m, col):
    H, W = a.shape
    h, w = m.shape
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    mm = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0][..., None]
    c = col[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] if (isinstance(col, np.ndarray) and col.ndim == 3) else np.asarray(col, np.float32)
    rgb[Y0:Y1, X0:X1] = rgb[Y0:Y1, X0:X1] * (1 - mm) + c * mm
    a[Y0:Y1, X0:X1] = a[Y0:Y1, X0:X1] * (1 - mm[..., 0]) + mm[..., 0]


PAL = dict(
    NEED=np.array([0.035, 0.09, 0.105], np.float32),       # needle body: deep blue-green
    NEED_HI=np.array([0.075, 0.16, 0.165], np.float32),    # sky-lit upper needles (teal)
    NEED_LO=np.array([0.04, 0.06, 0.13], np.float32),      # underside: cool blue bounce, not black
    SNOW_HI=np.array([0.8, 0.88, 1.08], np.float32),
    SNOW_LIT=np.array([0.52, 0.62, 0.92], np.float32),
    SNOW_SHD=np.array([0.2, 0.22, 0.52], np.float32),
    SNOW_DEEP=np.array([0.11, 0.12, 0.33], np.float32))


def bough_fir(W, H, trunk_u, top_v, base_v, reach, s, seed=0, lamps=(), fog=(0.2, 0.33, 0.56)):
    """Paint the fir into a fresh (H, W) RGBA plate. trunk_u: trunk x (px), top_v / base_v: apex and
    lowest-bough rows (px), reach(v) -> bough length (px) at row v. lamps: (u, v, rgb, radius_px, k)
    screen-space warm sources. Returns rgb (H, W, 3), alpha (H, W)."""
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

    nz = rng.random((H // 2 + 1, W // 2 + 1)).astype(np.float32)
    k = max(int(5 * s) | 1, 3)
    ker = np.zeros((k, k), np.float32)
    for i in range(k):
        ker[i, min(int(i * 0.45 + k * 0.3), k - 1)] = 1.0
    ker /= ker.sum()
    st = cv2.filter2D(nz, -1, ker)
    st = cv2.resize(st, (W, H), interpolation=cv2.INTER_LINEAR)
    st = np.clip((st - st.mean()) / (st.std() + 1e-6), -2.2, 2.2).astype(np.float32)

    span = base_v - top_v
    # ---- sparse dark interior right around the trunk
    vs = np.linspace(top_v, base_v + 0.4 * span, 160)
    rr = np.array([reach(v) for v in vs]) * 0.24
    rr = rr * (1 + 0.5 * np.sin(vs * 0.05 / s + 1.0) * np.sin(vs * 0.021 / s)) + rng.normal(0, 1, len(vs)) * 0.1 * rr
    rr = np.maximum(rr, 4 * s)
    poly = np.concatenate([np.stack([trunk_u + rr, vs], 1), np.stack([trunk_u - rr[::-1], vs[::-1]], 1)])
    X0, Y0 = max(int(poly[:, 0].min()) - 2, 0), max(int(poly[:, 1].min()) - 2, 0)
    w2, h2 = min(int(poly[:, 0].max()) + 3, W) - X0, min(int(poly[:, 1].max()) + 3, H) - Y0
    if w2 > 0 and h2 > 0:
        m = _poly_mask([poly], X0, Y0, w2, h2, ss=2)
        m = cv2.GaussianBlur(m, (0, 0), 2.0 * s + 0.3)
        col = PAL['NEED_LO'] * 0.75 * (1 + 0.25 * st[Y0:Y0 + h2, X0:X0 + w2, None])
        _over(rgb, a, X0, Y0, m * 0.97, col.astype(np.float32))

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
        col = col * (1 - snow[..., None] * 0.9) + PAL['SNOW_LIT'] * 0.75 * snow[..., None] * 0.9
        _over(rgb, a, X0, Y0, m, col.astype(np.float32))

    # ---- boughs
    items = []
    v = top_v + 0.02 * span
    tier = 0
    Lmax = max(reach(base_v), 1)
    while v < base_v + 0.12 * span:
        L = reach(v)
        spacing = max(0.05 * span * (0.5 + 0.5 * min(L / Lmax, 1)), 6 * s) * rng.uniform(0.6, 1.4)
        nb = 1 + int(rng.random() < 0.6) + int(rng.random() < 0.3)
        for b in range(nb):
            kind = 'main' if b == 0 else ('back' if rng.random() < 0.55 else 'fore')
            lf = {'main': rng.uniform(0.6, 1.15), 'back': rng.uniform(0.45, 0.8), 'fore': rng.uniform(0.28, 0.55)}[kind]
            dv = {'main': 0.0, 'back': rng.uniform(-0.5, -0.15), 'fore': rng.uniform(0.15, 0.6)}[kind] * spacing
            haze = 0.3 if kind == 'back' else 0.0
            order = tier + {'main': 0.0, 'back': -1.5, 'fore': 1.2}[kind] + rng.uniform(-0.9, 0.9)
            dirn = 1.0 if (b == 0 or rng.random() < 0.55) else -1.0
            if dirn < 0:
                lf *= 0.95
            items.append((order, trunk_u + rng.uniform(-0.03, 0.03) * L, v + dv, L * lf, haze, tier, kind,
                          int(rng.integers(1 << 30)), dirn))
        v += spacing
        tier += 1
    items.sort(key=lambda q: q[0])
    for (_, u0, v0, Lb, haze, tr, kind, sd, dirn) in items:
        if Lb < 8 * s:
            continue
        _bough(rgb, a, u0, v0, Lb, s, np.random.default_rng(sd), st, lamp_light, fog, haze, W, H, tr, kind, dirn)
    return rgb, a


def _ip(xq, x, y):
    if x[-1] < x[0]:
        return np.interp(xq, x[::-1], y[::-1])
    return np.interp(xq, x, y)


def _bough(rgb, a, u0, v0, L, s, rng, st, lamp_light, fog, haze, W, H, tier, kind, dirn=1.0):
    """One drooping bough: a tapering, sagging needle mass with a ragged hanging fringe, carrying a few
    separate snow clumps of very different size (dark needles showing between them). Each clump has an
    irregular sagging silhouette (lumpy top, drips hanging from the underside), a flat lit top plane with
    a thin cool rim and a few ice sparkles, a jagged step to a blue-violet shadow face and a deep
    underside - painted values, not a shaded tube."""
    NEED, NEED_HI, NEED_LO = PAL['NEED'], PAL['NEED_HI'], PAL['NEED_LO']
    SNOW_LIT, SNOW_SHD, SNOW_DEEP = PAL['SNOW_LIT'], PAL['SNOW_SHD'], PAL['SNOW_DEEP']
    n = 120
    t = np.linspace(0, 1, n)
    a0 = rng.uniform(-0.2, 0.18) if kind != 'fore' else rng.uniform(0.05, 0.35)
    droop = rng.uniform(0.2, 0.62)
    curl = rng.uniform(0.05, 0.35)
    x = u0 + dirn * L * t * (1 - 0.22 * droop * t)
    y = v0 + L * a0 * t + L * droop * t ** 2 + L * curl * np.clip(t - 0.62, 0, None) ** 2 * 6
    y = y + L * 0.018 * np.sin(t * rng.uniform(5, 11) + rng.uniform(0, 6)) * t
    thick = rng.uniform(0.12, 0.17) if kind != 'fore' else rng.uniform(0.15, 0.22)
    T = L * thick * (1 - 0.82 * t) ** 1.15 + 1.2 * s
    yu = y - T * 0.35
    yl = y + T * 0.65
    body = np.concatenate([np.stack([x, yu], 1), np.stack([x[::-1], yl[::-1]], 1)])
    polys = [body]
    # hanging branchlets under the bough (drooping more toward the tip), a few tufts poking up at the back
    nstr = int(18 + L / (3.5 * s))
    for i in range(nstr):
        ti = rng.uniform(0.03, 0.99)
        j = min(int(ti * (n - 1)), n - 1)
        ln = T[j] * rng.uniform(0.5, 1.4) * (1.3 - 0.4 * ti) + 2 * s
        wd = max(T[j] * rng.uniform(0.22, 0.5), 1.3 * s)
        ang = rng.uniform(-0.55, 0.25)
        bx, by = x[j], yl[j] - T[j] * 0.3
        tip = (bx + dirn * math.sin(ang) * ln, by + math.cos(ang) * ln)
        mid = (bx + dirn * math.sin(ang) * ln * 0.55, by + math.cos(ang) * ln * 0.55)
        polys.append(np.array([[bx - wd, by], [bx + wd, by], [mid[0] + wd * 0.5, mid[1]], [tip[0] + wd * 0.1, tip[1]],
                               [tip[0] - wd * 0.2, tip[1] - wd * 0.2], [mid[0] - wd * 0.6, mid[1]]]))
    for i in range(int(4 + L / (40 * s))):
        ti = rng.uniform(0.1, 0.95)
        j = min(int(ti * (n - 1)), n - 1)
        ln = T[j] * rng.uniform(0.4, 0.9)
        wd = max(T[j] * 0.25, 1.2 * s)
        ang = rng.uniform(-0.6, 0.6) + math.pi
        bx, by = x[j], yu[j] + T[j] * 0.2
        polys.append(np.array([[bx - wd, by], [bx + wd, by], [bx + dirn * math.sin(ang) * ln, by + math.cos(ang) * ln]]))
    for i in range(5):
        j = n - 1 - int(rng.integers(0, 14))
        ln = T[0] * rng.uniform(0.25, 0.6)
        ang = rng.uniform(0.0, 1.3)
        wd = max(T[j] * 0.45, 1.3 * s)
        polys.append(np.array([[x[j] - dirn * wd, y[j] - wd], [x[j] - dirn * wd, y[j] + wd],
                               [x[j] + dirn * math.cos(ang) * ln, y[j] + math.sin(ang) * ln]]))
    allp = np.concatenate(polys, 0)
    x0 = int(math.floor(allp[:, 0].min())) - 3
    y0 = int(math.floor(allp[:, 1].min())) - int(T.max() * 3.0) - 10
    x1 = int(math.ceil(allp[:, 0].max())) + 4
    y1 = int(math.ceil(allp[:, 1].max())) + int(T.max() * 1.2) + 4
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    w, h = X1 - X0, Y1 - Y0
    fm = _poly_mask(polys, X0, Y0, w, h, ss=3)
    xx = np.arange(X0, X1, dtype=np.float32)[None, :]
    yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
    ycx = _ip(xx[0], x, y)[None, :]
    Tx = _ip(xx[0], x, T)[None, :]
    rel = (yy - ycx) / np.maximum(Tx, 1e-3)
    shade = np.clip((rel + 0.35) / 1.3, 0, 1)[..., None]
    fol = NEED_HI * (1 - shade) ** 2 + NEED * (1 - (1 - shade) ** 2)
    und = np.clip((rel - 0.3) / 0.6, 0, 1)[..., None]
    fol = fol * (1 - und) + NEED_LO * und
    fol = fol * (1 + 0.38 * st[Y0:Y1, X0:X1, None])
    E = lamp_light(xx, yy)
    fol = fol + E * np.array([0.22, 0.14, 0.07], np.float32) * (1 - shade) * 0.5
    fol = fol * (1 - haze) + fog * 0.35 * haze
    _over(rgb, a, X0, Y0, fm, fol.astype(np.float32))

    # ---- snow: a few separate clumps of different size along the bough
    Smax = L * rng.uniform(0.045, 0.075) + 1.5 * s
    if kind == 'fore':
        Smax *= 1.05
    clumps = []
    tc = rng.uniform(0.02, 0.1)
    tend = rng.uniform(0.82, 0.96)
    while tc < tend:
        ln = rng.uniform(0.12, 0.42) * (1.0 - 0.4 * tc)
        te = min(tc + ln, 0.97)
        big = rng.uniform(0.3, 0.95) * (1 - 0.5 * tc)          # clumps get smaller toward the tip
        if rng.random() < 0.18:
            big *= 0.35                                           # a thin dusting here and there
        clumps.append((tc, te, big))
        tc = te + rng.uniform(0.025, 0.12)
    if not clumps:
        return
    Eh = E.mean(-1, keepdims=True)
    warm = np.array([1.0, 0.66, 0.36], np.float32)
    amb = np.array([0.95, 0.68, 0.22], np.float32)
    sm_all = np.zeros((h, w), np.float32)
    col_all = np.zeros((h, w, 3), np.float32)
    rim_all = np.zeros((h, w), np.float32)
    for (ta, tb, big) in clumps:
        sel = (t >= ta) & (t <= tb)
        if sel.sum() < 4:
            continue
        tt = (t[sel] - ta) / max(tb - ta, 1e-3)
        xs_, yus, Ts = x[sel], yu[sel], T[sel]
        # lumpy, asymmetric top: a main swell plus 2-4 smaller bumps, steep ends
        prof = np.sin(np.pi * np.clip(tt, 0, 1)) ** rng.uniform(0.35, 0.6)
        pk = rng.uniform(0.25, 0.75)
        prof = prof * (0.75 + 0.35 * np.exp(-((tt - pk) / rng.uniform(0.15, 0.35)) ** 2))
        for _ in range(int(rng.integers(2, 5))):
            c = rng.uniform(0.1, 0.9)
            prof = prof + rng.uniform(-0.12, 0.2) * np.exp(-((tt - c) / rng.uniform(0.05, 0.12)) ** 2)
        prof = np.clip(prof, 0, None)
        prof = _smooth(prof, 2)
        Sk = Smax * big
        top = yus - Sk * prof + Ts * 0.12
        # underside: sagging belly with a few drips / lobes hanging lower, of varied size
        belly = np.sin(np.pi * np.clip(tt, 0, 1)) ** 0.3
        drip = np.zeros_like(tt)
        for _ in range(int(rng.integers(1, 5))):
            c = rng.uniform(0.1, 0.9)
            drip += rng.uniform(0.3, 1.2) * np.exp(-((tt - c) / rng.uniform(0.03, 0.09)) ** 2)
        Tb = np.minimum(Ts, Smax * 1.6)
        bot = yus + (Tb * (0.25 + 0.25 * big) + Sk * 0.2) * belly + Tb * 0.4 * np.clip(drip, 0, 1.4) * belly + 0.6 * s
        bot = np.maximum(bot, top + 1.0 * s)
        poly = np.concatenate([np.stack([xs_, top], 1), np.stack([xs_[::-1], bot[::-1]], 1)])
        m = _poly_mask([poly], X0, Y0, w, h, ss=3)
        ttop = _ip(xx[0], xs_, top)[None, :]
        tbot = _ip(xx[0], xs_, bot)[None, :]
        rv = np.clip((yy - ttop) / np.maximum(tbot - ttop, 1e-3), 0, 1)
        # flat lit top plane -> jagged crisp step to the shadow face -> deep underside
        ts_ = rng.uniform(0.32, 0.55)
        jag = 0.07 * np.sin(xx * (0.07 / s) + rng.uniform(0, 6)) + 0.035 * np.sin(xx * (0.19 / s) + rng.uniform(0, 6))
        step = np.clip((rv - ts_ - jag) / 0.08, 0, 1)
        step = step * step * (3 - 2 * step)
        lv = rng.uniform(0.9, 1.08)
        col = SNOW_LIT * lv * (1 - step[..., None]) + SNOW_SHD * rng.uniform(0.9, 1.1) * step[..., None]
        dd = np.clip((rv - 0.8) / 0.2, 0, 1)[..., None]
        col = col * (1 - 0.5 * dd) + SNOW_DEEP * 0.5 * dd
        col = col * (1 + 0.035 * st[Y0:Y1, X0:X1, None])
        col = col + Eh * amb * (1 - step[..., None]) * 0.55 + Eh * warm * step[..., None] * 0.25
        col = col * (1 - np.array([0.0, 0.1, 0.4], np.float32) * np.clip(Eh * 1.2, 0, 1))
        col_all = col_all * (1 - m[..., None]) + col * m[..., None]
        sm_all = np.maximum(sm_all, m)
        # thin cool rim on the top edge
        rm = np.zeros((h * 3, w * 3), np.uint8)
        q = ((np.stack([xs_, top + 0.5 * s], 1) - [X0, Y0]) * 3).astype(np.int32)
        cv2.polylines(rm, [q], False, 255, max(int(round(1.3 * s * 3 * 0.5)), 2), cv2.LINE_AA)
        rim_all = np.maximum(rim_all, cv2.resize(rm.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA) * m)
    tf = 1.0 - 0.12 * min(tier / 14.0, 1.0)
    col_all = col_all * tf
    col_all = col_all * (1 - haze * 0.6) + fog * 1.2 * haze * 0.6
    _over(rgb, a, X0, Y0, sm_all, col_all.astype(np.float32))
    # needle tips poking through the snow edge (breaks the clean silhouette)
    for _ in range(int(rng.integers(2, 6))):
        ti = rng.uniform(0.05, 0.9)
        j = min(int(ti * (n - 1)), n - 1)
        bx, by = x[j], yu[j] - T[j] * rng.uniform(0.0, 0.4)
        ln = T[j] * rng.uniform(0.3, 0.7)
        wd = max(0.9 * s, T[j] * 0.08)
        pm = _poly_mask([np.array([[bx - wd, by], [bx + wd, by], [bx + dirn * ln * 0.4, by - ln]])], X0, Y0, w, h, ss=3)
        _over(rgb, a, X0, Y0, pm * 0.9, (NEED_HI * 0.9 * (1 - haze) + fog * 0.35 * haze).astype(np.float32))
    rim_c = np.array([0.72, 0.84, 1.15], np.float32) * 1.05 + Eh * amb * 1.3
    _over(rgb, a, X0, Y0, rim_all * (0.8 - 0.45 * haze), rim_c.astype(np.float32))
    # a few ice sparkles on the lit tops
    ys_, xs2 = np.nonzero((rim_all > 0.5) & (rng.random(rim_all.shape) < 0.012))
    if len(xs2):
        sp = np.zeros((h, w), np.float32)
        sp[ys_, xs2] = 1.0
        sp = cv2.GaussianBlur(sp, (0, 0), 0.6 * s + 0.2) * 3.0
        sp = np.clip(sp, 0, 1) * (1 - haze)
        _over(rgb, a, X0, Y0, sp, (np.array([1.35, 1.4, 1.6], np.float32) + Eh * amb).astype(np.float32))
