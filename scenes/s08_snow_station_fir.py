"""Painted snow-laden fir for s08_snow_station (BG-painting style, replaces the faceted tier pines).

The tree is built the way a background painter blocks it in:
  * a ragged dark core silhouette around the trunk,
  * individually placed drooping branches (left / right / toward camera, irregular spacing, gaps, a few
    over-long limbs) whose needle masses end in hooked tips and hang serrated fringes,
  * separate hand-shaped snow caps sitting on each branch (1-3 clumps, lumpy domed tops, drooping lobes
    at the front edge) with a crisp bright top rim (warm on the lamp side, cool on the sky side), a lit
    body and a soft blue underside, plus a dark crevice where the cap meets the needles.
Painted bottom-to-top so each branch's needle fringe overlaps the snow of the branch below.
All geometry is in plate pixels; the local canvas is clipped to the plate so huge foreground trees
stay cheap."""
import math
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def fir(cv, bu, bv, hpx, z, seed, fol=(0.03, 0.055, 0.095), snow_top=(0.64, 0.74, 0.97),
        snow_shd=(0.2, 0.26, 0.52), rim=(0.86, 0.94, 1.15), warm=None, warm_side=0.0, warm_amt=0.0,
        fog=None, fog_amt=0.0, width=0.3, snow=1.0, ss=3, detail=1.0, lean=None, rim_amt=1.0,
        return_layer=False, mound=True):
    rng = np.random.default_rng(seed)
    if hpx < 3:
        return None
    S = int(ss if hpx >= 60 else max(ss, 4))
    if hpx > 500:
        S = 2
    hw0 = hpx * width * rng.uniform(0.92, 1.08)
    PW, PH = cv.W, cv.H
    gx0 = int(math.floor(bu - hw0 * 1.45)) - 4
    gy0 = int(math.floor(bv - hpx * 1.06)) - 4
    gx1 = int(math.ceil(bu + hw0 * 1.45)) + 4
    gy1 = int(math.ceil(bv + hpx * 0.1)) + 4
    x0, y0 = max(gx0, 0), max(gy0, 0)
    x1, y1 = min(gx1, PW), min(gy1, PH)
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = x1 - x0, y1 - y0
    Hh, Ww = h * S, w * S
    col = np.zeros((Hh, Ww, 3), np.float32)
    alp = np.zeros((Hh, Ww), np.float32)
    snowm = np.zeros((Hh, Ww), np.float32)
    rimm = np.zeros((Hh, Ww), np.float32)

    def L(px, py):          # plate px -> local supersampled coords
        return (np.asarray(px, np.float64) - x0) * S, (np.asarray(py, np.float64) - y0) * S

    cx, top = L(bu, bv - hpx)
    cx, top = float(cx), float(top)
    H = hpx * S
    hw = hw0 * S
    fol = np.asarray(fol, np.float32)
    s_top = np.asarray(snow_top, np.float32)
    s_shd = np.asarray(snow_shd, np.float32)
    rimc = np.asarray(rim, np.float32)
    wdir = float(np.sign(warm_side)) if (warm is not None and warm_amt > 0.02) else 0.0
    wcol = np.asarray(warm, np.float32) if warm is not None else np.zeros(3, np.float32)
    lean = rng.uniform(-0.025, 0.025) if lean is None else lean
    bend = rng.uniform(-0.04, 0.04)

    def axis_x(r):
        return cx + lean * H * (1 - r) + bend * H * math.sin(math.pi * r)

    def put(poly, colfn, sn=0.0, rimfn=None):
        p = np.round(np.asarray(poly)).astype(np.int32)
        bx0, by0 = max(int(p[:, 0].min()), 0), max(int(p[:, 1].min()), 0)
        bx1, by1 = min(int(p[:, 0].max()) + 1, Ww), min(int(p[:, 1].max()) + 1, Hh)
        if bx1 <= bx0 or by1 <= by0:
            return
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cv2.fillPoly(m, [p - [bx0, by0]], 1)
        mb = m.astype(bool)
        if not mb.any():
            return
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

    ph = rng.uniform(0, 6.28, 3)

    def env(r):
        return hw * (0.04 + 0.96 * r ** 0.82) * (1 + 0.08 * math.sin(r * 11 + ph[0]) + 0.05 * math.sin(r * 27 + ph[1]))

    # ---- trunk
    tw = max(hpx * 0.014, 0.7) * S
    put(np.array([[axis_x(0.3) - tw * 0.5, top + H * 0.3], [axis_x(0.3) + tw * 0.5, top + H * 0.3],
                  [axis_x(1.0) + tw * 1.4, top + H * 1.0], [axis_x(1.0) - tw * 1.4, top + H * 1.0]]),
        lambda X, Y: fol * 0.5)

    # ---- ragged dark core (fills the centre so gaps only open toward the outer edge)
    rs = np.linspace(0.05, 1.0, 60)
    cw = np.array([env(r) * (0.42 + 0.12 * math.sin(r * 31 + ph[2])) for r in rs])
    jag = 1 + 0.25 * rng.random(60)
    ys = top + rs * H
    xl = np.array([axis_x(r) for r in rs]) - cw * jag
    xr = np.array([axis_x(r) for r in rs]) + cw * jag[::-1]
    put(np.concatenate([np.stack([xl, ys], 1), np.stack([xr[::-1], ys[::-1]], 1)]),
        lambda X, Y: fol * 0.62)

    # ---- branches
    n = int(np.clip(6 + hpx / 30, 6, 22) * rng.uniform(0.9, 1.1))
    th0 = H / n
    branches = []
    for side in (-1, 1):
        for i in range(n):
            r = 0.07 + 0.86 * (i + rng.uniform(0.15, 0.85)) / n
            if rng.random() < 0.07 and r > 0.25:
                continue                                   # gap
            Lb = env(r) * rng.uniform(0.72, 1.08)
            if rng.random() < 0.08 and r > 0.3:
                Lb *= rng.uniform(1.12, 1.3)               # an over-long limb
            branches.append((r, side, Lb, False))
    # branches reaching toward the camera: low snow mounds over the core
    if detail > 0:
        for i in range(int(n * 0.6 * detail)):
            r = rng.uniform(0.15, 0.97)
            branches.append((r + 0.004, float(rng.uniform(-0.75, 0.75)), env(r) * rng.uniform(0.2, 0.42), True))
    branches.sort(key=lambda b: -b[0])                     # bottom first

    def branch(r, side, Lb, front):
        th = th0 * rng.uniform(0.75, 1.05) * (0.75 + 0.35 * r)
        xr0 = axis_x(r)
        yr0 = top + r * H
        m = 28
        t = np.linspace(0, 1, m)
        if front:
            # a short limb pointing at the camera: seen end-on as a drooping lobe centred at xr0 + side*env
            xc_ = xr0 + side * env(r) * 0.8
            half = Lb
            xs = xc_ + (t * 2 - 1) * half
            at = np.abs(t * 2 - 1)
            yc = yr0 + th * 0.25 + th * 0.5 * at ** 1.6
            yu = yc - th * 0.25 * (1 - at ** 2)
            yl = yc + th * (0.55 + 0.35 * (1 - at)) + th * 0.3 * np.abs(np.sin(t * rng.uniform(9, 16) + ph[0])) ** 2
            tt_side = (t * 2 - 1)
            poly = np.concatenate([np.stack([xs, yu], 1), np.stack([xs[::-1], yl[::-1]], 1)])
        else:
            droop = Lb * rng.uniform(0.2, 0.42) * (0.55 + 0.6 * r)
            xr0 = xr0 - side * th * rng.uniform(0.3, 0.8)
            Lb = Lb + th * 0.5
            arch = Lb * rng.uniform(0.0, 0.12)
            xs = xr0 + side * Lb * t
            yc = yr0 + droop * t ** 1.8 - arch * np.sin(np.pi * t) + droop * 0.35 * np.clip((t - 0.8) / 0.2, 0, 1) ** 2
            thu = th * (0.22 * (1 - t) + 0.06)
            thl = th * (0.48 * (1 - t) ** 0.7 + 0.2)
            taper = 0.35 + 0.65 * _ss(0.0, 0.14, t)
            yu = yc - thu * taper
            yl = yc + thl * taper
            # serrated hanging fringe: asymmetric spikes (steep on the outer side) pointing down, longer
            # toward the tip; built as a function of t so the outline never self-intersects
            m2 = 160
            tq = np.linspace(0, 1, m2)
            ycq = np.interp(tq, t, yc)
            lowq = np.interp(tq, t, yl)
            upq = np.interp(tq, t, yu)
            k = max(int(Lb / max(th * 0.3, 2 * S)), 3)
            sp = np.zeros(m2)
            for tj in rng.uniform(0.03, 0.99, k):
                ln = th * rng.uniform(0.12, 0.6) * (0.4 + 0.8 * tj)
                wj = rng.uniform(0.35, 0.8) * th / max(Lb, 1)
                d = tq - tj
                prof_ = np.where(d < 0, 1 + d / (wj * 1.6), 1 - d / (wj * 0.5))
                sp = np.maximum(sp, ln * np.clip(prof_, 0, 1) ** 1.3)
            lowq = lowq + sp
            tuf = np.zeros(m2)
            for tj in rng.uniform(0.5, 0.97, int(rng.integers(2, 5))):
                d = tq - tj
                wj = rng.uniform(0.25, 0.5) * th / max(Lb, 1)
                tuf = np.maximum(tuf, th * rng.uniform(0.06, 0.2) * np.clip(1 - np.abs(d) / wj, 0, 1))
            upq = upq - tuf
            xq_ = xr0 + side * Lb * tq
            tipx = xr0 + side * Lb * 1.04
            tipy = ycq[-1] + th * rng.uniform(0.2, 0.5)
            poly = np.concatenate([np.stack([xq_, upq], 1), [[tipx, tipy]], np.stack([xq_[::-1], lowq[::-1]], 1)])
            tt_side = t * side
        shade = rng.uniform(0.85, 1.15) * (1.08 if front else 1.0)

        yu_i, yl_i, xs_i = yu, yl, xs
        order = np.argsort(xs_i)

        def needles(X, Y):
            yu_ = np.interp(X, xs_i[order], yu_i[order])
            yl_ = np.interp(X, xs_i[order], yl_i[order])
            v = np.clip((Y - yu_) / np.maximum(yl_ - yu_, 1), 0, 1)[..., None]
            # cool sky light on the upper needles, darker hanging undersides
            return fol * shade * (1.35 - 0.55 * v) + s_shd * 0.06 * (1 - v)
        put(poly, needles)

        if snow <= 0.02 or rng.random() < 0.03:
            return
        # ---- snow caps: 1-3 separate hand-shaped clumps along the branch
        if front:
            segs = [(0.12, 0.88)]
        else:
            t1 = rng.uniform(0.72, 0.95)
            nc = 1 + int(Lb > th * 2.5) + int(Lb > th * 5.5 and rng.random() < 0.7)
            cuts = np.sort(rng.uniform(0.02, t1, nc - 1)) if nc > 1 else np.array([])
            edges = np.concatenate([[rng.uniform(0.0, 0.08)], cuts, [t1]])
            segs = []
            for a_, b_ in zip(edges[:-1], edges[1:]):
                gap = rng.uniform(0.015, 0.05)
                if b_ - a_ - gap > 0.08:
                    segs.append((a_ + gap * 0.5, b_ - gap * 0.5))
        for (ta, tb) in segs:
            mm = 30
            u = np.linspace(0, 1, mm)
            tq = ta + (tb - ta) * u
            xq = np.interp(tq, t, xs)
            ycq = np.interp(tq, t, yu)
            if front:
                ycq = np.interp(tq, t, yu)
            L_ = abs(xq[-1] - xq[0])
            hc = th * snow * rng.uniform(1.05, 1.6) * (1 - 0.3 * (tq.mean() if not front else 0.3))
            hc = min(hc, L_ * 0.6)
            asym = rng.uniform(0.35, 0.65)
            uu = np.where(u < asym, u / asym * 0.5, 0.5 + (u - asym) / (1 - asym) * 0.5)
            prof = np.sin(np.pi * uu) ** 0.35
            nl = int(rng.integers(2, 5))
            lump = 0.8 + 0.25 * np.abs(np.sin(np.pi * u * nl + rng.uniform(0, 6))) + rng.normal(0, 0.03, mm)
            st = ycq - hc * prof * lump
            # front edge: the pillow bulges over and hangs a few rounded lobes
            nd = int(rng.integers(1, 4))
            drip = np.zeros(mm)
            for _ in range(nd):
                c0 = rng.uniform(0.15, 0.85)
                wd = rng.uniform(0.06, 0.16)
                drip += np.exp(-((u - c0) / wd) ** 2) * rng.uniform(0.15, 0.45)
            sb = ycq + th * (0.14 + drip) * prof ** 0.35
            sb[0] = st[0] = (st[0] + sb[0]) / 2
            sb[-1] = st[-1] = (st[-1] + sb[-1]) / 2
            # crevice shadow on the needles below the cap
            crev = sb + max(th * 0.1, S * 1.2)
            put(np.concatenate([np.stack([xq, sb - 1], 1), np.stack([xq[::-1], crev[::-1]], 1)]),
                lambda X, Y: fol * 0.3)
            o = np.argsort(xq)
            xo, sto, sbo = xq[o], st[o], sb[o]
            tso = (tt_side if front else tq * side)
            tso = np.interp(xo, xq[o], (np.linspace(-1, 1, mm) if front else tq * side)[o])
            rw = max(hc * rng.uniform(0.05, 0.08), S * 1.0)
            shade_k = rng.uniform(0.95, 1.05)

            def snowc(X, Y, xo=xo, sto=sto, sbo=sbo, tso=tso, shade_k=shade_k):
                st_ = np.interp(X, xo, sto)
                sb_ = np.interp(X, xo, sbo)
                v = np.clip((Y - st_) / np.maximum(sb_ - st_, 1), 0, 1)
                # flat painted planes: pale lit top / one cool shadow value, crisp jagged split
                jg = 0.07 * np.sin(X * 0.21 + sto[0]) + 0.05 * np.sin(X * 0.53 + 1.7)
                k = _ss(0.5, 0.53, v + jg)[..., None]
                c = s_top * (1 - k) + (s_shd * 1.1) * k
                return c * shade_k

            def rimf(X, Y, xo=xo, sto=sto, rw=rw):
                st_ = np.interp(X, xo, sto)
                d = Y - st_
                return _ss(rw, rw * 0.55, d) * 0.85
            put(np.concatenate([np.stack([xq, st], 1), np.stack([xq[::-1], sb[::-1]], 1)]), snowc, 1.0, rimf)

    for (r, side, Lb, front) in branches:
        branch(r, side, Lb, front)

    # ---- snow drift banked around the foot of the tree (buries the lowest drooping branches)
    if snow > 0.02 and mound:
        bx, by = L(bu, bv)
        bx, by = float(bx), float(by)
        m = 40
        uu = np.linspace(-1, 1, m)
        wd = hw * rng.uniform(0.85, 1.05)
        hgt = H * rng.uniform(0.05, 0.08)
        ytop = by - hgt * (1 - np.abs(uu) ** 3) ** 0.5 * (0.85 + 0.2 * np.sin(uu * 5 + ph[1]) ** 2)
        ybot = by + H * 0.035 * (1 - np.abs(uu) ** 3) ** 0.5
        ytop = np.minimum(ytop, ybot - 0.5)
        ytop[0] = ytop[-1] = ybot[0]
        xsm = bx + uu * wd
        o_x, o_t = xsm, ytop

        def mound(X, Y):
            st_ = np.interp(X, o_x, o_t)
            v = np.clip((Y - st_) / max(hgt * 1.6, 1), 0, 1)[..., None]
            k = _ss(0.42, 0.46, v)
            return s_top * 0.78 * (1 - k) + s_shd * 1.1 * k

        def mrim(X, Y):
            st_ = np.interp(X, o_x, o_t)
            return np.clip(1 - (Y - st_) / max(hgt * 0.12, S), 0, 1) * 0.7
        put(np.concatenate([np.stack([xsm, ytop], 1), np.stack([xsm[::-1], ybot], 1)]), mound, 1.0, mrim)

    # ---- leader and top cap
    tipw = max(hw * 0.018, S * 0.9)
    ax = axis_x(0.0)
    put(np.array([[ax - tipw, top + H * 0.1], [ax + lean * H * 0.05 + tipw * 0.3, top - H * 0.03], [ax + tipw, top + H * 0.1]]),
        lambda X, Y: fol)
    tt_ = np.linspace(0, np.pi, 16)
    capp = np.concatenate([np.stack([ax - tipw * 2.4 * np.cos(tt_) , top + H * 0.07 - tipw * 2.2 * np.sin(tt_) ** 0.7
                                     * (1 + 0.15 * np.sin(tt_ * 3))], 1),
                           [[ax + tipw * 2.2, top + H * 0.085], [ax, top + H * 0.095], [ax - tipw * 2.2, top + H * 0.085]]])
    cy0, cy1 = top + H * 0.07 - tipw * 2.4, top + H * 0.095
    put(capp, lambda X, Y: s_top * (1 - _ss(0.5, 0.56, (Y - cy0) / max(cy1 - cy0, 1))[..., None]) + s_shd * 1.1 * _ss(0.5, 0.56, (Y - cy0) / max(cy1 - cy0, 1))[..., None], 1.0,
        lambda X, Y: np.clip(1 - (Y - cy0) / max(tipw * 0.9, 1), 0, 1))

    # ---- texture: vertical needle streaks in the dark masses, soft brush strokes inside the snow
    nt = rng.random((max(Hh // (3 * S), 2), max(Ww // (2 * S), 2))).astype(np.float32)
    nt = cv2.resize(nt, (Ww, Hh), interpolation=cv2.INTER_LINEAR)
    nt = cv2.GaussianBlur(nt, (0, 0), sigmaX=0.5 * S, sigmaY=1.6 * S)
    col = col * (1 + ((1 - snowm) * (nt - 0.5) * 0.7)[..., None])
    if hpx > 400:
        # close trees: finer needle strokes (short diagonal hatching) over the dark masses
        nf = rng.random((max(Hh // S, 2), max(Ww // S, 2))).astype(np.float32)
        nf = cv2.resize(nf, (Ww, Hh), interpolation=cv2.INTER_NEAREST)
        nf = cv2.GaussianBlur(nf, (0, 0), sigmaX=0.4 * S, sigmaY=2.2 * S)
        col = col * (1 + ((1 - snowm) * (nf - 0.5) * 1.1)[..., None])
    bt = rng.random((max(Hh // (5 * S), 2), max(Ww // (14 * S), 2))).astype(np.float32)
    bt = cv2.resize(bt, (Ww, Hh), interpolation=cv2.INTER_CUBIC)
    bt = cv2.GaussianBlur(bt, (0, 0), sigmaX=2.5 * S, sigmaY=0.8 * S)
    col = col * (1 + (snowm * (1 - rimm) * (bt - 0.5) * 0.03)[..., None])

    # ---- rims on the cap tops: cool sky rim everywhere, warm lamp rim on the lamp side
    xxS = (np.arange(Ww, dtype=np.float32)[None, :] - cx) / max(hw, 1)
    if wdir != 0:
        lampside = np.clip(0.35 + 0.75 * xxS * wdir, 0, 1) ** 1.3
    else:
        lampside = np.zeros_like(xxS)
    rim_col = rimc[None, None, :] * (1 - lampside[..., None] * warm_amt * 0.8) + \
        (wcol * 2.2 + 0.25)[None, None, :] * (lampside[..., None] * warm_amt * 0.8)
    ra = (rimm * rim_amt)[..., None]
    col = col * (1 - ra) + rim_col * ra
    # warm light washing the lamp side of the snow bodies
    if wdir != 0:
        col = col + (snowm * lampside * warm_amt * 0.35)[..., None] * wcol

    colL = cv2.resize(col * alp[..., None], (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alp, (w, h), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(snowm * alp, (w, h), interpolation=cv2.INTER_AREA)
    colL = colL / np.maximum(a, 1e-4)[..., None]
    sm = sm / np.maximum(a, 1e-4)
    # silhouette rim: a thin edge light on the needle tips facing the lamp / sky
    rwp = max(1, int(round(max(hpx * 0.0025, 1.0))))
    upv = np.zeros_like(a)
    upv[rwp:] = a[:-rwp]
    et = np.clip(a - upv, 0, 1) * (1 - sm)
    colL = colL + et[..., None] * rimc * 0.12 * rim_amt
    if wdir != 0:
        sh_ = np.zeros_like(a)
        if warm_side > 0:
            sh_[:, :-rwp] = a[:, rwp:]
        else:
            sh_[:, rwp:] = a[:, :-rwp]
        xx = (np.arange(w, dtype=np.float32)[None, :] - (bu - x0)) / max(hw0, 1)
        side = np.clip(0.5 + 0.6 * xx * warm_side, 0, 1) ** 1.6
        el = np.clip(a - sh_, 0, 1)
        colL = colL + (el * side * warm_amt * 0.9)[..., None] * wcol
    if fog is not None and fog_amt > 0:
        colL = colL * (1 - fog_amt) + np.asarray(fog, np.float32) * fog_amt
    a = np.clip(a, 0, 1)
    if return_layer:
        return x0, y0, a, colL.astype(np.float32)
    cv.paint(x0, y0, a, colL.astype(np.float32), z)
    return None
