"""Painted foreground fir for s08_snow_station: layered, drooping, snow-laden boughs.

The tree is painted the way a background artist paints a framing conifer: a dark cool interior mass
near the trunk, then tier after tier of boughs from the top down, each bough back-to-front within its
tier. A bough is a long drooping arc (rises a little from the trunk, then sags under its snow load and
the tip curls down), a dark blue-green needle mass with a ragged fringe of hanging branchlets under it,
and a heavy lumpy snow pillow riding its top that follows the arc, drapes over the front edge in rounded
lobes and thins out toward the tip. Snow is painted in flat values: lit top (cool sky white, warmed by
the lamps near them), a crisp step to a blue-violet shadow on the draped front face, a bright rim on the
upper silhouette. Everything is deterministic for a seed and painted in plate pixels."""
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


def _over(rgb, a, x0, y0, m, col):
    """Paint premultiplied-free colour col (h,w,3 or 3) with coverage m at (x0, y0) into rgb/a."""
    H, W = a.shape
    h, w = m.shape
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    mm = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0][..., None]
    c = col[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] if (isinstance(col, np.ndarray) and col.ndim == 3) else np.asarray(col, np.float32)
    rgb[Y0:Y1, X0:X1] = rgb[Y0:Y1, X0:X1] * (1 - mm) + c * mm
    a[Y0:Y1, X0:X1] = a[Y0:Y1, X0:X1] * (1 - mm[..., 0]) + mm[..., 0]


def bough_fir(W, H, trunk_u, top_v, base_v, reach, s, seed=0, lamps=(), fog=(0.2, 0.33, 0.56)):
    """Paint the fir into a fresh (H, W) RGBA plate. trunk_u: trunk x (px, may be off-plate), top_v /
    base_v: apex and lowest-bough rows (px), reach(v) -> bough length (px) at row v. lamps: list of
    (u, v, rgb, radius_px, strength) screen-space warm sources that light nearby snow and rim the tips.
    Returns rgb (H, W, 3) (straight colour), alpha (H, W)."""
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

    NEED = np.array([0.036, 0.07, 0.1], np.float32)       # needle body (deep blue-green)
    NEED_HI = np.array([0.075, 0.13, 0.17], np.float32)      # sky-lit upper needles
    NEED_LO = np.array([0.012, 0.022, 0.045], np.float32)    # shadowed underside
    SNOW_HI = np.array([0.76, 0.86, 1.06], np.float32)
    SNOW_LIT = np.array([0.5, 0.61, 0.9], np.float32)
    SNOW_SHD = np.array([0.17, 0.2, 0.47], np.float32)       # blue-violet shadow
    SNOW_DEEP = np.array([0.1, 0.11, 0.3], np.float32)
    # needle stroke texture (fine directional streaks, drooping diagonally)
    nz = rng.random((H // 2 + 1, W // 2 + 1)).astype(np.float32)
    k = max(int(5 * s) | 1, 3)
    ker = np.zeros((k, k), np.float32)
    for i in range(k):
        ker[i, min(int(i * 0.45 + k * 0.3), k - 1)] = 1.0
    ker /= ker.sum()
    st = cv2.filter2D(nz, -1, ker)
    st = cv2.resize(st, (W, H), interpolation=cv2.INTER_LINEAR)
    st = np.clip((st - st.mean()) / (st.std() + 1e-6), -2.2, 2.2).astype(np.float32)

    # ---- interior mass near the trunk (dark, cool): fills behind the boughs so gaps near the trunk
    # read as deep tree, while gaps between the outer tips show sky
    vs = np.linspace(top_v, base_v + 0.4 * (base_v - top_v), 160)
    rr = np.array([reach(v) for v in vs]) * 0.42
    rr = rr * (1 + 0.25 * np.sin(vs * 0.09 + 1.0) * np.sin(vs * 0.031)) + rng.normal(0, 1, len(vs)) * 0.08 * rr
    poly = np.concatenate([np.stack([trunk_u + rr, vs], 1), [[trunk_u - 50, vs[-1]], [trunk_u - 50, vs[0]]]])
    x0, y0 = int(poly[:, 0].min()) - 2, int(poly[:, 1].min()) - 2
    w, h = int(poly[:, 0].max()) - x0 + 4, int(poly[:, 1].max()) - y0 + 4
    X0, Y0 = max(x0, 0), max(y0, 0)
    w2, h2 = min(x0 + w, W) - X0, min(y0 + h, H) - Y0
    if w2 > 0 and h2 > 0:
        m = _poly_mask([poly], X0, Y0, w2, h2, ss=2)
        m = cv2.GaussianBlur(m, (0, 0), 1.2 * s + 0.3)
        _over(rgb, a, X0, Y0, m, NEED_LO * 1.1)

    # ---- tiers of boughs, top to bottom
    span = base_v - top_v
    v = top_v + 0.02 * span
    tier = 0
    while v < base_v + 0.12 * span:
        L = reach(v)
        spacing = max(0.06 * span * (0.55 + 0.45 * min(L / max(reach(base_v), 1), 1)), 6 * s)
        # back bough (behind, shorter, a touch higher, hazier) then the main front bough, then an
        # occasional short foreshortened bough reaching toward the camera
        subs = [(-0.35, 0.72, 0.35), (0.0, 1.0, 0.0)]
        if rng.random() < 0.45:
            subs.append((0.4, rng.uniform(0.38, 0.55), 0.0))
        for (dv, lf, haze) in subs:
            Lb = L * lf * rng.uniform(0.78, 1.12)
            if Lb < 8 * s:
                continue
            _bough(rgb, a, trunk_u + rng.uniform(-0.04, 0.02) * Lb, v + dv * spacing, Lb, s, rng, st, lamp_light,
                   NEED, NEED_HI, NEED_LO, SNOW_HI, SNOW_LIT, SNOW_SHD, SNOW_DEEP, fog, haze, W, H, tier)
        v += spacing * rng.uniform(0.85, 1.15)
        tier += 1
    return rgb, a


def _bough(rgb, a, u0, v0, L, s, rng, st, lamp_light, NEED, NEED_HI, NEED_LO, SNOW_HI, SNOW_LIT, SNOW_SHD,
           SNOW_DEEP, fog, haze, W, H, tier):
    n = 90
    t = np.linspace(0, 1, n)
    droop = rng.uniform(0.2, 0.34)
    rise = rng.uniform(0.06, 0.12)
    curl = rng.uniform(0.05, 0.14)
    x = u0 + L * t
    y = v0 - L * rise * t + L * droop * t ** 2 + L * curl * np.clip(t - 0.75, 0, None) ** 2 * 12
    y = y + L * 0.012 * np.sin(t * 9 + rng.uniform(0, 6))
    T = L * rng.uniform(0.1, 0.13) * (1 - 0.7 * t) + 1.5 * s          # needle mass thickness
    yu = y - T * 0.35
    yl = y + T * 0.65
    # ---- needle mass + ragged hanging fringe (union of polygons)
    body = np.concatenate([np.stack([x, yu], 1), np.stack([x[::-1], yl[::-1]], 1)])
    polys = [body]
    nstr = int(10 + L / (7 * s))
    for i in range(nstr):
        ti = rng.uniform(0.04, 0.98)
        j = min(int(ti * (n - 1)), n - 1)
        ln = T[j] * rng.uniform(0.35, 0.95) * (1.2 - 0.5 * ti)
        wd = max(T[j] * rng.uniform(0.3, 0.6), 1.5 * s)
        ang = rng.uniform(-0.45, 0.1)                      # hanging, trailing back toward the trunk
        bx, by = x[j], yl[j] - T[j] * 0.3
        tip = (bx + math.sin(ang) * ln, by + math.cos(ang) * ln)
        mid = (bx + math.sin(ang) * ln * 0.55, by + math.cos(ang) * ln * 0.55)
        polys.append(np.array([[bx - wd, by], [bx + wd, by], [mid[0] + wd * 0.55, mid[1]], [tip[0] + wd * 0.12, tip[1]],
                               [tip[0] - wd * 0.2, tip[1] - wd * 0.2], [mid[0] - wd * 0.6, mid[1]]]))
    # spiky tip spray
    for i in range(3):
        j = n - 1 - int(rng.integers(0, 10))
        ln = T[0] * rng.uniform(0.3, 0.6)
        ang = rng.uniform(-0.2, 1.0)
        wd = max(T[j] * 0.45, 1.5 * s)
        polys.append(np.array([[x[j] - wd, y[j] - wd], [x[j] - wd, y[j] + wd],
                               [x[j] + math.cos(ang) * ln, y[j] + math.sin(ang) * ln]]))
    allp = np.concatenate(polys, 0)
    x0 = int(math.floor(allp[:, 0].min())) - 3
    y0 = int(math.floor(allp[:, 1].min())) - int(T.max() * 1.6) - 6
    x1 = int(math.ceil(allp[:, 0].max())) + 4
    y1 = int(math.ceil(allp[:, 1].max())) + 4
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    w, h = X1 - X0, Y1 - Y0
    fm = _poly_mask(polys, X0, Y0, w, h, ss=3)
    xx = np.arange(X0, X1, dtype=np.float32)[None, :]
    yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
    # column-wise param along the bough (monotonic x)
    tc = np.clip((xx - u0) / L, 0, 1)
    ycx = np.interp(xx[0], x, y)[None, :]
    Tx = np.interp(xx[0], x, T)[None, :]
    rel = (yy - ycx) / np.maximum(Tx, 1e-3)                  # -0.35 top .. 0.65 bottom (+ fringe)
    shade = np.clip((rel + 0.35) / 1.4, 0, 1)
    fol = NEED_HI * (1 - shade[..., None]) ** 2 + NEED * (1 - (1 - shade[..., None]) ** 2)
    fol = fol * (1 - 0.55 * np.clip((rel - 0.5) / 0.8, 0, 1))[..., None] + NEED_LO * 0.0
    fol = fol * (1 + 0.28 * st[Y0:Y1, X0:X1, None])
    # light from the lamps on the needles (only a little: they are dark and matte)
    E = lamp_light(xx, yy)
    fol = fol + E * np.array([0.22, 0.14, 0.07], np.float32) * (1 - shade[..., None]) * 0.5
    # outer tips a bit lighter/cooler (air between viewer and far boughs), back boughs hazier
    fol = fol * (1 - haze) + fog * 0.35 * haze
    _over(rgb, a, X0, Y0, fm, fol.astype(np.float32))
    # ---- snow pillow along the top of the bough
    env = np.clip(t / 0.1, 0, 1) ** 0.6 * np.clip((rng.uniform(0.86, 0.98) - t) / 0.3, 0, 1) ** 0.8
    # the snow sits in a few heavy clumps: occasional breaks where needles show through
    for k_ in range(int(rng.integers(0, 3))):
        c = rng.uniform(0.3, 0.85)
        env = env * (1 - 0.97 * np.exp(-((t - c) / rng.uniform(0.012, 0.03)) ** 2))
    S = L * rng.uniform(0.06, 0.08) * (1 - 0.4 * t) + 1.0 * s
    lump = np.zeros(n)
    for k_ in range(int(rng.integers(3, 6))):
        c = rng.uniform(0.05, 0.95)
        wdt = rng.uniform(0.06, 0.16)
        lump += rng.uniform(0.3, 0.7) * np.exp(-((t - c) / wdt) ** 2)
    lump = 0.6 + np.clip(lump, 0, 0.7)
    lump = _smooth(lump, 3)
    top = yu - S * env * lump + T * 0.1
    # draped front face: rounded lobes hanging over the needles
    drape = np.zeros(n)
    for k_ in range(int(rng.integers(3, 7))):
        c = rng.uniform(0.08, 0.9)
        wdt = rng.uniform(0.03, 0.08)
        drape += rng.uniform(0.3, 1.0) * np.exp(-((t - c) / wdt) ** 2)
    bot = yu + T * (0.3 + 0.75 * np.clip(drape, 0, 1.2)) * env + 0.5 * s
    bot = np.maximum(bot, top + 0.8 * s)
    keep = env > 0.02
    xs_, ts_, bs_ = x[keep], top[keep], bot[keep]
    if len(xs_) < 4:
        return
    sp = np.concatenate([np.stack([xs_, ts_], 1), np.stack([xs_[::-1], bs_[::-1]], 1)])
    sm = _poly_mask([sp], X0, Y0, w, h, ss=3)
    ttop = np.interp(xx[0], xs_, ts_)[None, :]
    tbot = np.interp(xx[0], xs_, bs_)[None, :]
    rv = np.clip((yy - ttop) / np.maximum(tbot - ttop, 1e-3), 0, 1)        # 0 top .. 1 bottom
    # flat painted values: lit top, crisp step into the blue-violet draped face, deep shadow under lobes
    step = np.clip((rv - 0.5 - 0.08 * st[Y0:Y1, X0:X1] * 0.3) / 0.07, 0, 1)
    step = step * step * (3 - 2 * step)
    col = SNOW_LIT * (1 - step[..., None]) + SNOW_SHD * step[..., None]
    col = col * (1 - 0.35 * np.clip((rv - 0.8) / 0.2, 0, 1))[..., None] + SNOW_DEEP * 0.35 * np.clip((rv - 0.8) / 0.2, 0, 1)[..., None]
    # a lighter band on the very top (sky light) and a subtle value variation along the pillow
    hi = np.clip(1 - rv / 0.18, 0, 1) ** 1.5
    col = col + (SNOW_HI - SNOW_LIT) * hi[..., None] * 0.8
    col = col * (1 + 0.04 * st[Y0:Y1, X0:X1, None] * 0.5)
    # warm lamp light on the lit top and a warm reflected tint in the shadow nearest a lamp
    Eh = E.mean(-1, keepdims=True)
    warm = np.array([1.0, 0.66, 0.36], np.float32)
    col = col + E * (1 - step[..., None]) * 0.9 + Eh * warm * step[..., None] * 0.25
    # lower boughs sit under the canopy of the ones above: a little less sky light
    col = col * (1.0 - 0.14 * min(tier / 14.0, 1.0))
    # distance haze for back boughs
    col = col * (1 - haze * 0.6) + fog * 1.2 * haze * 0.6
    _over(rgb, a, X0, Y0, sm, col.astype(np.float32))
    # crisp rim on the upper silhouette of the snow
    rim_pts = np.stack([xs_, ts_ + 0.6 * s], 1)
    rm = np.zeros((h * 3, w * 3), np.uint8)
    q = ((rim_pts - [X0, Y0]) * 3).astype(np.int32)
    cv2.polylines(rm, [q], False, 255, max(int(round(2.2 * s * 3 * 0.5)), 2), cv2.LINE_AA)
    rm = cv2.resize(rm.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA) * sm
    rim_c = SNOW_HI * 1.12 + E * 1.3
    _over(rgb, a, X0, Y0, rm * (0.75 - 0.4 * haze), rim_c.astype(np.float32))
    # a few dark needle tufts poking through the snow top
    for i in range(0):
        ti = rng.uniform(0.15, 0.85)
        j = int(ti * (n - 1))
        if env[j] < 0.3:
            continue
        bx, by = x[j], top[j] + S[j] * 0.35
        ln = S[j] * rng.uniform(0.35, 0.7)
        ang = rng.uniform(-2.4, -1.4)
        wd = max(S[j] * 0.22, 1.5 * s)
        tp = np.array([[bx - wd, by], [bx + wd, by], [bx + math.cos(ang) * ln + wd * 0.3, by + math.sin(ang) * ln],
                       [bx + math.cos(ang) * ln - wd * 0.3, by + math.sin(ang) * ln + wd * 0.2]])
        tm = _poly_mask([tp], X0, Y0, w, h, ss=3)
        _over(rgb, a, X0, Y0, tm * 0.9, NEED * 1.1)
