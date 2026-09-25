"""Round-18 night clouds for s07_comet_night: SOFT PAINTED moonlit clouds (yn_05) + rim-lit twilight banks (yn_02).

Why a rewrite: round 17 thresholded lattice value-noise (axis-aligned cells) and hard-stepped value planes, which
read as stair-stepped voxel sprites / posterised contour plates.  Here every field is smooth and isotropic and
evaluated at full plate resolution:
  * silhouette = metaball field.  Big core lobes are SUMMED (one soft body), cauliflower florets and buds are
    grown on the up-facing arc and MAX-unioned (crisp scalloped top edge with sharp valleys).  Streaks are a
    dense chain of long overlapping ellipses along a meandering spine (a band, not beads), torn apart by
    wind-stretched noise, with 1-3 lobed heads riding on them.
  * noise = Gaussian-blurred white noise upsampled with cubic interpolation (isotropic, no lattice steps),
    used for a smooth domain warp and for tearing the underside only.
  * edge = analytic signed distance (F - thr) / |grad F|: ~1 px anti-aliased on the lit top, feathered
    2-6 px (more on the warm banks) where the silhouette faces down -> lost soft undersides.
  * interior = 2-3 SOFT value masses from occlusion toward the light: pale cyan-white lit tops on every
    lobe (lobe-on-lobe), a blue-grey mid, a navy-violet body, translucent bluish base (stars show through).
  * warm banks: violet-grey bodies, a thin hot orange/pink rim on the underside + sun side, soft lost
    bottoms that melt into the afterglow.
Returns straight RGBA (Hp, Wp, 4) in plate px, colour bled into the transparent margin (no dark fringes).
"""
import math

import numpy as np
import cv2

LN2 = math.log(2.0)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise(w, h, cx, cy, rng):
    """Smooth isotropic(ish) noise, zero mean / unit std, feature size ~cx, cy px."""
    fx, fy = max(cx / 3.0, 1.0), max(cy / 3.0, 1.0)
    sw, sh = int(w / fx) + 6, int(h / fy) + 6
    g = rng.standard_normal((sh, sw)).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.3)
    big = cv2.resize(g, (int(math.ceil(sw * fx)), int(math.ceil(sh * fy))), interpolation=cv2.INTER_CUBIC)
    ox, oy = int(2 * fx), int(2 * fy)
    out = big[oy:oy + h, ox:ox + w]
    if out.shape != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return ((out - out.mean()) / (out.std() + 1e-6)).astype(np.float32)


def _fbm(w, h, cx, cy, octaves, rng, gain=0.5):
    out = np.zeros((h, w), np.float32)
    a = 1.0
    for o in range(octaves):
        out += a * _noise(w, h, cx / 2 ** o, cy / 2 ** o, rng)
        a *= gain
    return (out - out.mean()) / (out.std() + 1e-6)


K_EDGE = 0.5412   # compact kernel (1 - (q k)^2)^2 is 0.5 at q = 1


def _lobe(F, x, y, rx, ry, ang, wgt=1.0, k=0.2):
    """Union an elliptical compact lobe (field 0.5 at its radius, 0 beyond 1.85 r) into F with a
    polynomial smooth-max of blend width k (k -> 0: hard union -> crisp creases between florets)."""
    h, w = F.shape
    R = 1.9 * max(rx, ry)
    x0, x1 = int(max(x - R, 0)), int(min(x + R + 1, w))
    y0, y1 = int(max(y - R, 0)), int(min(y + R + 1, h))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - x, yy - y
    c, s = math.cos(ang), math.sin(ang)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    q2 = ((u / rx) ** 2 + (v / ry) ** 2) * K_EDGE ** 2
    g = wgt * np.clip(1 - q2, 0, 1) ** 2
    a = F[y0:y1, x0:x1]
    if k <= 1e-4:
        F[y0:y1, x0:x1] = np.maximum(a, g)
    else:
        hh = np.clip(k - np.abs(a - g), 0, None) / k
        F[y0:y1, x0:x1] = np.maximum(a, g) + hh * hh * k * 0.25


def _shift(img, dx, dy):
    """Sample img at p + (dx, dy) (zero outside)."""
    M = np.array([[1, 0, -dx], [0, 1, -dy]], np.float32)
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def _florets(rng, cx, cy, rx, ry, ang, n, rel, spread, depth, out, level=0, bias=0.0):
    """Grow florets on the up-facing arc of an ellipse (world coords, y down)."""
    c, s = math.cos(ang), math.sin(ang)
    for _ in range(n):
        th = math.radians(-90 + bias + rng.uniform(-1, 1) * spread)
        d = (math.cos(th), math.sin(th))
        du, dv = d[0] * c + d[1] * s, -d[0] * s + d[1] * c
        rr = 1.0 / math.sqrt((du / rx) ** 2 + (dv / ry) ** 2)
        rf = min(rx, ry) * rng.uniform(*rel)
        px = cx + d[0] * (rr - 0.5 * rf)
        py = cy + d[1] * (rr - 0.5 * rf)
        out.append((px, py, rf * rng.uniform(1.0, 1.2), rf, 0.0, level))
        if depth > 0 and rng.random() < 0.8:
            _florets(rng, px, py, rf, rf, 0.0, int(rng.integers(1, 4)), (0.34, 0.55), 70, depth - 1,
                     out, level + 1, bias * 0.5)


def _puff(rng, X, Y, r, a, core, flo, warm, flat=1.35):
    """A small cumulus puff: 1-3 core lobes + florets."""
    c, sn = math.cos(a), math.sin(a)
    n = int(rng.integers(1, 4))
    for j in range(n):
        u_ = (j - (n - 1) / 2) * r * rng.uniform(0.7, 1.1)
        rr = r * rng.uniform(0.7, 1.0) * (1 - 0.25 * abs(j - (n - 1) / 2))
        v_ = -0.3 * rr
        wx, wy = X + u_ * c - v_ * sn, Y + u_ * sn + v_ * c
        core.append((wx, wy, rr * flat, rr, a))
        _florets(rng, wx, wy, rr * flat, rr, a, int(rng.integers(2, 4)), (0.36, 0.56), 62,
                 0 if warm else 1, flo)


def _shape(cl, rng, s):
    """Metaball field of one cloud in its bbox.  Returns F, (bx0, by0), local v (below-base > 0)."""
    T, L = cl['T'], cl['L']
    kind = cl['kind']
    warm = cl.get('warm', 0) > 0
    a = math.radians(cl['ang'])
    c, sn = math.cos(a), math.sin(a)
    X, Y = cl['x'], cl['y']
    core, flo, band = [], [], []
    if kind == 'cu':
        n = int(rng.integers(3, 6))
        us = np.sort(rng.uniform(-0.36, 0.36, n)) * L
        for u_ in us:
            q = abs(u_) / (0.5 * L)
            r = T * rng.uniform(0.36, 0.52) * (1 - 0.5 * q ** 1.5)
            v_ = -0.5 * r + rng.normal(0, 0.03 * T)
            wx, wy = X + u_ * c - v_ * sn, Y + u_ * sn + v_ * c
            core.append((wx, wy, r * rng.uniform(1.2, 1.5), r, a))
            _florets(rng, wx, wy, r * 1.35, r, a, int(rng.integers(3, 6)), (0.3, 0.48), 70,
                     1 if warm else 2, flo)
        # low flat skirt the lobes sit on
        if not warm:
            band.append((X, Y + 0.02 * T, 0.5 * L, 0.13 * T, a))
    else:
        N = max(int(L / (0.05 * T + 3)), 8)
        uu = np.linspace(-0.5, 0.5, N) * L
        m1 = _noise(N + 8, 1, 7, 1, rng)[0, 4:4 + N]
        m2 = _noise(N + 8, 1, 3, 1, rng)[0, 4:4 + N]
        env = np.clip(1 - np.abs(uu / (0.5 * L)) ** 1.6, 0, 1) ** 0.8
        spacing = L / (N - 1)
        for i in range(N):
            th = T * 0.1 * env[i] * (0.6 + 0.35 * m2[i])
            if th < 0.015 * T:
                continue
            wx = X + uu[i] * c - 0.15 * T * m1[i] * sn
            wy = Y + uu[i] * sn + 0.15 * T * m1[i] * c
            band.append((wx, wy, spacing * 3.0 + th, max(th, 1.0), a))
        hk = cl.get('heads', (1, 3))
        npf = int(rng.integers(hk[0], hk[1] + 1)) + int(L / (0.45 * T))
        for j in range(npf):
            u_ = rng.uniform(-0.42, 0.3) * L
            i = int(np.clip((u_ / L + 0.5) * (N - 1), 0, N - 1))
            big = j < hk[1]
            r = T * (rng.uniform(0.17, 0.3) if big else rng.uniform(0.08, 0.15)) * (0.4 + 0.6 * env[i])
            v_ = 0.15 * T * m1[i] - 0.2 * r
            _puff(rng, X + u_ * c - v_ * sn, Y + u_ * sn + v_ * c, r, a, core, flo, warm)
    allp = core + band + [(p[0], p[1], p[2], p[3], 0.0) for p in flo]
    pad = int(0.3 * T + 12)
    rmax = max(max(p[2], p[3]) for p in allp)
    bx0 = int(min(p[0] for p in allp) - 1.2 * rmax - pad)
    bx1 = int(max(p[0] for p in allp) + 1.2 * rmax + pad)
    by0 = int(min(p[1] for p in allp) - 1.2 * rmax - pad)
    by1 = int(max(p[1] for p in allp) + 1.2 * rmax + pad)
    w, h = bx1 - bx0, by1 - by0
    F = np.zeros((h, w), np.float32)
    for (wx, wy, rx, ry, aa) in band:
        _lobe(F, wx - bx0, wy - by0, rx, ry, aa, 0.9, 0.25)
    for (wx, wy, rx, ry, aa) in core:
        _lobe(F, wx - bx0, wy - by0, rx, ry, aa, 1.0, 0.18)
    for (px, py, rx, ry, aa, lev) in flo:
        _lobe(F, px - bx0, py - by0, rx, ry, aa, 1.0 - 0.05 * lev, 0.05 if lev == 0 else 0.02)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vloc = -(xx + bx0 - X) * sn + (yy + by0 - Y) * c
    return F, (bx0, by0), vloc


def night_clouds18(Wp, Hp, clusters, H, light_xy, tail_glow=None, mw_glow=None, seed=18, sun_x=None, pal=None):
    s = H / 1080.0
    P = dict(top=(0.84, 0.95, 1.06), mid=(0.42, 0.56, 0.86), body=(0.27, 0.32, 0.63), base=(0.25, 0.32, 0.68),
             vtop=(0.66, 0.6, 0.82), vmid=(0.4, 0.36, 0.56), vbody=(0.3, 0.26, 0.46),
             rim=(1.6, 0.62, 0.34), rim2=(1.35, 0.5, 0.6), glow=(0.95, 0.5, 0.4))
    if pal:
        P.update(pal)
    cP = {k: np.array(v, np.float32) for k, v in P.items()}
    out = np.zeros((Hp, Wp, 4), np.float32)
    tg = mg = None
    if tail_glow is not None:
        q = 8
        sm = cv2.resize(tail_glow, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        tg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 0.03 * H / q), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    if mw_glow is not None:
        q = 8
        sm = cv2.resize(mw_glow.max(-1), (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        mg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 0.06 * H / q), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        mg = mg / (mg.max() + 1e-6)
    if sun_x is None:
        sun_x = 0.3 * Wp

    order = sorted(range(len(clusters)), key=lambda i: (-clusters[i].get('far', 0.0), clusters[i]['y']))
    for ic in order:
        cl = clusters[ic]
        rng = np.random.default_rng(seed * 1000 + ic * 7 + 3)
        T = cl['T']
        st = cl['kind'] == 'st'
        wm = float(cl.get('warm', 0.0))
        haze = float(cl.get('haze', 0.0))
        F, (bx0, by0), vloc = _shape(cl, rng, s)
        h, w = F.shape
        # smooth domain warp (irregular big forms) + a finer one (irregular florets); no lattice steps
        dxl = _noise(w, h, 0.35 * T, 0.35 * T, rng) * 0.05 * T + _noise(w, h, 0.07 * T + 2, 0.07 * T + 2, rng) * 0.004 * T
        dyl = _noise(w, h, 0.35 * T, 0.35 * T, rng) * 0.04 * T + _noise(w, h, 0.07 * T + 2, 0.07 * T + 2, rng) * 0.004 * T
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        F = cv2.remap(F, xx + dxl, yy + dyl, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        vloc = cv2.remap(vloc, xx + dxl, yy + dyl, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # lower part: torn by wind-stretched noise, flat-ish base that dissolves
        low = _ss(-0.35 * T, 0.12 * T, vloc)
        tear = _fbm(w, h, 0.45 * T + 6, 0.07 * T + 3, 3, rng)
        tear2 = _fbm(w, h, 0.12 * T + 3, 0.12 * T + 3, 2, rng)
        F = F + (0.1 if st else 0.06) * tear * (low if not st else 0.4 + 0.6 * low) + 0.04 * tear2 * low
        # cauliflower: billow octaves (|noise|: rounded bumps, sharp creases), strongest on the sky side
        bil = (0.5 * np.abs(_noise(w, h, 0.2 * T + 3, 0.18 * T + 3, rng)) +
               0.32 * np.abs(_noise(w, h, 0.085 * T + 2, 0.08 * T + 2, rng)) +
               0.18 * np.abs(_noise(w, h, 0.04 * T + 1.5, 0.038 * T + 1.5, rng)))
        F = F + (0.27 if not wm else 0.18) * (bil - 0.8) * (1 - 0.7 * low)
        if st:
            F = F + 0.025 * _fbm(w, h, 0.7 * T + 10, 0.1 * T + 2, 2, rng)     # fibrous along the wind
        F = F * (1 - 0.8 * _ss(0.06 * T, 0.3 * T, vloc))                        # flat, soft base
        thr = 0.5
        # --- remove crumbs smaller than a brush dab
        ins = (F > thr - 0.08).astype(np.uint8)
        n_, lab, stt, _ = cv2.connectedComponentsWithStats(ins, connectivity=8)
        if n_ > 2:
            keep = stt[:, cv2.CC_STAT_AREA] >= (0.09 * T) ** 2 + 16
            keep[0] = True
            kill = (~keep)[lab] & (ins > 0)
            kill = cv2.dilate(kill.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            F = np.where(kill, np.minimum(F, thr - 0.25), F)
        # --- signed distance + facing
        gx = cv2.Sobel(F, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(F, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        gn = np.sqrt(gx * gx + gy * gy) + 1e-5
        sd = np.clip((F - thr) / gn, -60, 60)
        Gb = cv2.GaussianBlur(F, (0, 0), 0.05 * T + 1)
        sy = cv2.Sobel(Gb, cv2.CV_32F, 0, 1, ksize=3)
        sx = cv2.Sobel(Gb, cv2.CV_32F, 1, 0, ksize=3)
        sgn = np.sqrt(sx * sx + sy * sy) + 1e-6
        down = np.clip(-sy / sgn, 0, 1) ** 1.2
        # coverage + occlusion toward the light
        Cc = _ss(thr - 0.06, thr + 0.06, F)
        lx_ = 1.0 if light_xy[0] >= cl['x'] else -1.0
        L1 = np.array([0.35 * lx_, -1.0])
        L1 /= np.linalg.norm(L1)
        Cb1 = cv2.GaussianBlur(Cc, (0, 0), 0.018 * T + 0.6)
        Cb2 = cv2.GaussianBlur(Cc, (0, 0), 0.07 * T + 1.0)
        jit = _fbm(w, h, 0.09 * T + 3, 0.07 * T + 3, 2, rng)
        d1 = (0.06 if st else 0.11 if not wm else 0.06) * T
        d1m = d1 * np.clip(0.9 + 0.55 * _noise(w, h, 0.4 * T + 4, 0.3 * T + 4, rng), 0.25, 1.9)
        occ1 = cv2.remap(Cb1, (xx + L1[0] * d1m).astype(np.float32), (yy + L1[1] * d1m).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        lit = 1 - _ss(0.3, 0.52, occ1 + 0.18 * jit)
        occ2 = _shift(Cb2, L1[0] * 0.28 * T, L1[1] * 0.28 * T)
        body = _ss(0.35, 0.9, occ2 + 0.08 * jit)
        under = _shift(Cb2, 0.0, 0.16 * T)
        basen = 1 - _ss(0.25, 0.8, under)
        # --- colour: soft painted value masses
        if wm > 0:
            top = cP['top'] * (1 - haze) + cP['vtop'] * haze
            mid = cP['mid'] * (1 - haze) + cP['vmid'] * haze
            bod = cP['body'] * (1 - haze) + cP['vbody'] * haze
            bas = bod
        else:
            top, mid, bod, bas = cP['top'], cP['mid'], cP['body'], cP['base']
        col = np.ones((h, w, 1), np.float32) * mid
        col = col * (1 - (0.85 * body)[..., None]) + bod * (0.85 * body)[..., None]
        bk = (basen * (1 - lit) * 0.6)[..., None]
        col = col * (1 - bk) + bas * bk
        if wm > 0:
            lk_ = (lit * 0.7)[..., None]
        else:
            lk_ = lit[..., None]
        # painted mid-value patches inside the shadowed body (soft-edged brush masses, not a gradient)
        pp = _fbm(w, h, 0.22 * T + 3, 0.15 * T + 3, 2, rng)
        occm = _shift(cv2.GaussianBlur(Cc, (0, 0), 0.035 * T + 0.8), L1[0] * 0.17 * T, L1[1] * 0.17 * T)
        pk = (_ss(0.38, 0.62, 1 - occm + 0.32 * pp) * (1 - lit) * (1 - 0.5 * basen) * 0.6)[..., None]
        col = col * (1 - pk) + (mid * 0.4 + top * 0.6) * pk
        col = col * (1 - lk_) + top * lk_
        # faint broad colour variation inside the body (painted, not flat)
        col = col * (1 + 0.035 * _fbm(w, h, 0.3 * T, 0.2 * T, 2, rng))[..., None]
        xg = xx + bx0
        A_extra = 1.0
        if wm > 0:
            # thin hot rim on the underside + sun side, glow melting the lost bottom into the afterglow
            rdx = np.clip((sun_x - cl['x']) / (0.6 * Wp), -1, 1)
            Rd = np.array([0.7 * rdx, 1.0])
            Rd /= np.linalg.norm(Rd)
            d3 = 0.05 * T + 2.0 * s
            occR = _shift(Cb1, Rd[0] * d3, Rd[1] * d3)
            rim = (1 - _ss(0.18, 0.6, occR + 0.08 * jit)) * _ss(0.0, 0.5, Cb1 + Cc)
            # broad underglow: the lower body warms toward the lit underside (not a line)
            occW = _shift(Cb2, Rd[0] * 0.2 * T, Rd[1] * 0.2 * T)
            bnc = (1 - _ss(0.05, 0.9, occW + 0.12 * jit)) ** 1.6 * Cc
            sunk = np.exp(-((xg - sun_x) / (0.45 * Wp)) ** 2)
            rimc = cP['rim'] * (1 - basen[..., None] * 0.0)
            rimc = rimc * (0.6 + 0.4 * sunk[..., None]) + cP['rim2'] * (1 - sunk[..., None]) * 0.3
            rk = np.clip(rim * wm * (0.7 + 0.5 * sunk), 0, 1)[..., None]
            bk2 = np.clip(bnc * wm * 0.6 * (0.6 + 0.4 * sunk), 0, 1)[..., None]
            col = col * (1 - bk2) + np.array([0.86, 0.46, 0.56], np.float32) * bk2
            col = col * (1 - rk) + rimc * rk
            # bounce: bodies warm up slightly toward the base
            bw = (basen * 0.35 * wm)[..., None]
            col = col * (1 - bw) + cP['glow'] * 0.6 * bw
        # --- alpha: crisp AA top, feathered lost underside
        if wm > 0:
            wid = 0.6 + down * (0.018 * T + 2.5 * s)
        else:
            wid = 0.6 + down * (1.5 + 4.5 * s) * (1.6 if st else 1.0)
        A = _ss(-1.0, 1.0, sd / wid)
        # translucent base (stars show through), streaks lighter overall
        trn = 1 - (0.38 if not st else 0.45) * basen * (1 - lit)
        if wm > 0:
            trn = 1 - 0.1 * basen * (1 - lit)
        A = A * trn * A_extra
        # drop floating crumbs / hair-thin shreds (judged as noise specks)
        am = (A > 0.05).astype(np.uint8)
        n_, lab, stt, _ = cv2.connectedComponentsWithStats(am, connectivity=8)
        if n_ > 2:
            big_ = (stt[:, cv2.CC_STAT_AREA] >= (0.13 * T) ** 2 + 40) &                    (np.minimum(stt[:, cv2.CC_STAT_WIDTH], stt[:, cv2.CC_STAT_HEIGHT]) >= 0.05 * T + 3)
            big_[0] = False
            kp = cv2.GaussianBlur(cv2.dilate(big_[lab].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32),
                                  (0, 0), 1.0)
            A = A * kp
        # --- composite
        xa, xb = max(bx0, 0), min(bx0 + w, Wp)
        ya, yb = max(by0, 0), min(by0 + h, Hp)
        if xb <= xa or yb <= ya:
            continue
        sl = (slice(ya - by0, yb - by0), slice(xa - bx0, xb - bx0))
        cc = col[sl]
        if tg is not None:
            tpk = np.clip(tg[ya:yb, xa:xb], 0, 0.35)
            cc = cc + tpk * (0.25 + 0.75 * lit[sl])[..., None] * 0.55
        if mg is not None:
            cc = cc * (1 + mg[ya:yb, xa:xb][..., None] * np.array([0.06, 0.02, -0.03], np.float32))
        aa = (A[sl] * cl['amt'])[..., None]
        dst = out[ya:yb, xa:xb]
        a0 = dst[..., 3:4]
        na = aa + a0 * (1 - aa)
        dst[..., :3] = (cc * aa + dst[..., :3] * a0 * (1 - aa)) / np.maximum(na, 1e-5)
        dst[..., 3:4] = na
    # bleed colour into the transparent margin so bilinear resampling never darkens the edge
    a = out[..., 3]
    q = 4
    pm = cv2.resize(out[..., :3] * a[..., None], (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
    am = cv2.resize(a, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
    pm = cv2.GaussianBlur(pm, (0, 0), 3)
    am = cv2.GaussianBlur(am, (0, 0), 3)[..., None]
    bl = cv2.resize(pm / np.maximum(am, 1e-4), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    wgt = _ss(0.0, 0.2, a)[..., None]
    out[..., :3] = out[..., :3] * wgt + bl * (1 - wgt)
    return out
