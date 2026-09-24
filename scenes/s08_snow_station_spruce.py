"""Painted snow-laden spruce for s08_snow_station (replaces the faceted blob pine).

Drooping branch tiers with serrated hanging needle fringes, each carrying a continuous snow pillow shaded
per pixel: a crisp bright rim along its top edge, a lit body and a cool blue-violet underside with a dark
crevice where it meets the needles. Painted bottom to top so each tier's dark fringe overlaps the pillow
below (the stacked-layer look of painted anime backgrounds). The side facing the lamp (warm_side sign)
gets warm rims on caps and silhouette."""
import math
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def pine2(cv, bu, bv, hpx, z, seed, fol=(0.035, 0.07, 0.11), snow_top=(0.66, 0.76, 0.98),
          snow_shd=(0.2, 0.26, 0.5), rim=(0.8, 0.9, 1.15), warm=None, warm_side=0.0, warm_amt=0.0,
          fog=None, fog_amt=0.0, width=0.27, snow=1.0, ss=3, rim_amt=1.0, detail=1.0, lean=None):
    rng = np.random.default_rng(seed)
    if hpx < 3:
        return
    S = int(ss if hpx >= 40 else max(ss, 4))
    hw0 = hpx * width * rng.uniform(0.9, 1.1)
    x0 = int(bu - hw0 * 1.5) - 4
    y0 = int(bv - hpx * 1.05) - 4
    w = int(hw0 * 3.0) + 8
    h = int(hpx * 1.08) + 8
    Hh, Ww = h * S, w * S
    col = np.zeros((Hh, Ww, 3), np.float32)
    alp = np.zeros((Hh, Ww), np.float32)
    snowm = np.zeros((Hh, Ww), np.float32)
    rimm = np.zeros((Hh, Ww), np.float32)
    cx = (bu - x0) * S
    top = (bv - hpx - y0) * S
    H = hpx * S
    hw = hw0 * S
    fol = np.asarray(fol, np.float32)
    s_top = np.asarray(snow_top, np.float32)
    s_shd = np.asarray(snow_shd, np.float32)
    rimc = np.asarray(rim, np.float32)
    wdir = float(np.sign(warm_side)) if (warm is not None and warm_amt > 0.02) else 0.0
    lean = rng.uniform(-0.02, 0.02) if lean is None else lean

    def region(poly):
        p = np.round(poly).astype(np.int32)
        bx0, by0 = max(p[:, 0].min(), 0), max(p[:, 1].min(), 0)
        bx1, by1 = min(p[:, 0].max() + 1, Ww), min(p[:, 1].max() + 1, Hh)
        if bx1 <= bx0 or by1 <= by0:
            return None
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cv2.fillPoly(m, [p - [bx0, by0]], 1)
        mb = m.astype(bool)
        if not mb.any():
            return None
        return bx0, by0, bx1, by1, mb

    def put(poly, colfn, sn=0.0, rimfn=None):
        r = region(poly)
        if r is None:
            return
        bx0, by0, bx1, by1, mb = r
        X = np.arange(bx0, bx1, dtype=np.float32)[None, :]
        Y = np.arange(by0, by1, dtype=np.float32)[:, None]
        c = np.broadcast_to(colfn(X, Y), (by1 - by0, bx1 - bx0, 3))
        col[by0:by1, bx0:bx1][mb] = c[mb]
        alp[by0:by1, bx0:bx1][mb] = 1.0
        snowm[by0:by1, bx0:bx1][mb] = sn
        if rimfn is not None:
            rv = np.broadcast_to(rimfn(X, Y), (by1 - by0, bx1 - bx0))
            rimm[by0:by1, bx0:bx1][mb] = rv[mb]
        else:
            rimm[by0:by1, bx0:bx1][mb] = 0.0

    # trunk
    tw = max(hpx * 0.017, 0.6) * S
    put(np.array([[cx - tw * 0.5, top + H * 0.25], [cx + tw * 0.5, top + H * 0.25], [cx + tw * 1.3, top + H],
                  [cx - tw * 1.3, top + H]]), lambda X, Y: fol * 0.55)

    ph1, ph2 = rng.uniform(0, 6.28, 2)

    def env(r):
        return hw * (0.05 + 0.95 * r ** 0.88) * (1 + 0.09 * math.sin(r * 10 + ph1) + 0.05 * math.sin(r * 23 + ph2))

    def tier(xc_, y, eL, eR, th, droop, samt, front=False):
        m = 64
        t = np.linspace(-1, 1, m)
        at = np.abs(t)
        xs = xc_ + np.where(t < 0, t * eL, t * eR)
        nb = int(rng.integers(2, 5))
        bump = th * 0.12 * np.abs(np.sin(np.pi * at * nb + rng.uniform(0, 6)))
        dome = th * rng.uniform(0.1, 0.35) * (1 - at ** 2)
        yu = y + droop * at ** 1.7 - dome - bump
        nt = int(rng.integers(5, 11))
        saw = np.abs(((at * nt + rng.uniform(0, 1)) % 1.0) - 0.5) * 2
        jag = rng.uniform(0.4, 1.0, m)
        yl = yu + th * (0.55 + 0.45 * (1 - at) ** 0.6) + th * 0.42 * saw ** 1.8 * jag * (0.3 + 0.7 * at)
        yl[0] = yu[0] + th * 0.12
        yl[-1] = yu[-1] + th * 0.12
        sh = rng.uniform(0.85, 1.15) * (1.08 if front else 1.0)

        def needles(X, Y):
            yu_ = np.interp(X, xs, yu)
            yl_ = np.interp(X, xs, yl)
            v = np.clip((Y - yu_) / np.maximum(yl_ - yu_, 1), 0, 1)[..., None]
            return fol * sh * (0.6 + 0.9 * v ** 2) + s_shd * 0.08 * v ** 3
        put(np.concatenate([np.stack([xs, yu], 1), np.stack([xs[::-1], yl[::-1]], 1)]), needles)
        if samt <= 0.02:
            return
        tl, tr = rng.uniform(0.72, 0.97), rng.uniform(0.72, 0.97)
        sel = (t >= -tl) & (t <= tr)
        if sel.sum() < 4:
            return
        u = np.where(t < 0, -t / tl, t / tr)
        prof = np.clip(1 - np.clip(u, 0, 1) ** 2.4, 0, 1) ** 0.5
        nl = int(rng.integers(2, 5))
        lump = 0.78 + 0.28 * np.abs(np.sin(np.pi * t * nl + rng.uniform(0, 6)))
        sth = th * samt * rng.uniform(1.0, 1.4) * prof * lump
        st = yu - sth * 0.8
        sb = yu + sth * 0.38 + th * 0.12 * np.abs(np.sin(np.pi * t * (nl + 3) + rng.uniform(0, 6))) ** 3 * prof
        ts = t[sel]
        xs_s, st_s, sb_s = xs[sel], st[sel], sb[sel]
        crev = sb_s + max(th * 0.1, S * 1.0)
        put(np.concatenate([np.stack([xs_s, sb_s - 1], 1), np.stack([xs_s[::-1], crev[::-1]], 1)]),
            lambda X, Y: fol * 0.35)
        rw = max(float(sth.max()) * 0.13, S * 1.1)
        shade_k = rng.uniform(0.94, 1.05)

        def snowc(X, Y):
            st_ = np.interp(X, xs_s, st_s)
            sb_ = np.interp(X, xs_s, sb_s)
            tt = np.interp(X, xs_s, ts)
            v = np.clip((Y - st_) / np.maximum(sb_ - st_, 1), 0, 1)
            k = _ss(0.18, 0.95, v)[..., None]
            c = s_top * (1 - k) + s_shd * k
            side = np.clip(tt * (wdir if wdir != 0 else 1.0), -1, 1)[..., None]
            return c * (0.94 + 0.08 * side) * shade_k

        def rimf(X, Y):
            st_ = np.interp(X, xs_s, st_s)
            tt = np.interp(X, xs_s, ts)
            d = Y - st_
            lampside = np.clip(0.55 + 0.45 * tt * wdir, 0, 1) if wdir != 0 else 0.6
            wid = rw * (0.7 + 0.8 * lampside)
            return np.clip(1 - d / wid, 0, 1) ** 0.7
        put(np.concatenate([np.stack([xs_s, st_s], 1), np.stack([xs_s[::-1], sb_s[::-1]], 1)]), snowc, 1.0, rimf)

    n = int(np.clip(5 + hpx / 38, 5, 17) * rng.uniform(0.92, 1.1))
    rk = np.cumsum(rng.uniform(0.6, 1.4, n))
    rk = 0.07 + 0.86 * (rk - rk[0]) / (rk[-1] - rk[0])
    for k in range(n - 1, -1, -1):
        r = rk[k]
        e = env(r)
        th = max(H * 1.0 / n * rng.uniform(0.8, 1.2), S * 1.8)
        y = top + r * H
        xc_ = cx + lean * H * (1 - r) + rng.normal(0, 0.04) * e
        eL = e * rng.uniform(0.7, 1.15) * (rng.uniform(0.5, 0.75) if rng.random() < 0.12 else 1.0)
        eR = e * rng.uniform(0.7, 1.15) * (rng.uniform(0.5, 0.75) if rng.random() < 0.12 else 1.0)
        tier(xc_, y, eL, eR, th, th * rng.uniform(0.9, 1.7) * (0.45 + 0.7 * r), snow * (1.0 if rng.random() > 0.07 else 0.35))
        if detail > 0 and r > 0.25 and rng.random() < 0.5 * detail:
            sd = rng.choice([-1.0, 1.0])
            tier(xc_ + sd * e * rng.uniform(0.2, 0.5), y + th * rng.uniform(0.25, 0.55), e * rng.uniform(0.22, 0.38),
                 e * rng.uniform(0.22, 0.38), th * 0.75, th * 0.5, snow * rng.uniform(0.8, 1.05), front=True)
    # spire with a small snow cap
    tipw = max(hw * 0.03, S * 0.8)
    put(np.array([[cx - tipw, top + H * 0.09], [cx + lean * H * 0.1, top - H * 0.025], [cx + tipw, top + H * 0.09]]),
        lambda X, Y: fol)
    capp = np.array([[cx - tipw * 1.6, top + H * 0.07], [cx - tipw * 0.5, top + H * 0.035], [cx + tipw * 0.3, top + H * 0.03],
                     [cx + tipw * 1.6, top + H * 0.07], [cx, top + H * 0.078]])
    put(capp, lambda X, Y: s_top, 1.0, lambda X, Y: np.float32(0.5))

    # needle texture on the dark foliage
    nt = rng.random((max(Hh // (3 * S), 2), max(Ww // (3 * S), 2))).astype(np.float32)
    nt = cv2.resize(nt, (Ww, Hh), interpolation=cv2.INTER_LINEAR)
    nt = cv2.GaussianBlur(nt, (0, 0), sigmaX=0.6 * S, sigmaY=1.4 * S)
    col = col * (1 + ((1 - snowm) * (nt - 0.5) * 0.6)[..., None])
    # painterly brush strokes inside the snow (soft, horizontal)
    bt = rng.random((max(Hh // (4 * S), 2), max(Ww // (12 * S), 2))).astype(np.float32)
    bt = cv2.resize(bt, (Ww, Hh), interpolation=cv2.INTER_CUBIC)
    bt = cv2.GaussianBlur(bt, (0, 0), sigmaX=2.5 * S, sigmaY=0.7 * S)
    col = col * (1 + (snowm * (1 - rimm) * (bt - 0.5) * 0.12)[..., None])
    # cap rims: cool bright
    col = col * (1 - rimm[..., None] * rim_amt) + rimc * (rimm * rim_amt)[..., None]

    colL = cv2.resize(col * alp[..., None], (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alp, (w, h), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(snowm * alp, (w, h), interpolation=cv2.INTER_AREA)
    rm = cv2.resize(rimm * alp, (w, h), interpolation=cv2.INTER_AREA)
    colL = colL / np.maximum(a, 1e-4)[..., None]
    sm = sm / np.maximum(a, 1e-4)
    rm = rm / np.maximum(a, 1e-4)
    yy = np.clip((np.arange(h, dtype=np.float32)[:, None] - (bv - hpx - y0)) / max(hpx, 1), 0, 1)
    colL = colL * (1 + (0.1 - 0.26 * yy) * sm)[..., None]
    rwp = max(1, int(round(max(hpx * 0.004, 1.0))))
    up = np.zeros_like(a)
    up[rwp:] = a[:-rwp]
    et = np.clip(a - up, 0, 1) * (0.3 + 0.7 * sm) * rim_amt
    colL = colL + et[..., None] * rimc * 0.25
    if warm is not None and warm_amt > 0:
        warm = np.asarray(warm, np.float32)
        xx = (np.arange(w, dtype=np.float32)[None, :] - (bu - x0)) / max(hw0, 1)
        side = np.clip(0.5 + 0.6 * xx * warm_side, 0, 1) ** 1.6
        colL = colL + (side * warm_amt)[..., None] * warm * (0.12 + 0.8 * sm[..., None])
        colL = colL + (rm * side * warm_amt)[..., None] * warm * 1.8
        sh_ = np.zeros_like(a)
        if warm_side > 0:
            sh_[:, :-rwp] = a[:, rwp:]
        else:
            sh_[:, rwp:] = a[:, :-rwp]
        el = np.clip(a - sh_, 0, 1) * (0.3 + 0.7 * sm)
        colL = colL + (el * warm_amt * 1.4)[..., None] * warm
    if fog is not None and fog_amt > 0:
        colL = colL * (1 - fog_amt) + np.asarray(fog, np.float32) * fog_amt
    cv.paint(x0, y0, np.clip(a, 0, 1), colL.astype(np.float32), z)
