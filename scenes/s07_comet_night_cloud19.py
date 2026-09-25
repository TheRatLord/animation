"""Round-19 night clouds for s07_comet_night: SOFT PAINTED clouds (yn_05) + rim-lit twilight banks (yn_02).

Round 18 still read as clay: every floret carried its own lit cap (lobe-on-lobe emboss), the lit band ran round
the whole silhouette and the underside frayed into ragged crumbs.  Here the silhouette and the paint are
decoupled:
  * silhouette: a few big Gaussian masses (smooth union) + one ring of cauliflower florets on the sky side only,
    domain-warped; streaks are long torn bands; cirrus are wind-stretched fibrous veils.
  * lighting is COLUMN based, not per-lobe: the moonlit tone is a painted plane hanging from the top edge
    (distance to the sky above, measured straight up), so it only exists on the top / comet-facing edge; the
    body below is one flat mid value that darkens slightly toward the base.  No outline, no emboss.
  * the underside is a LOST edge: alpha falls off over 20-40 px measured up from the bottom of each column,
    broken by wind-stretched noise into soft wisps (no crumbs: components smaller than a brush dab are dropped).
  * twilight banks: violet body, a hot rim ONLY on the horizon-facing underside that fades inward as a gradient
    (orange -> pink -> violet), cauliflower top with a softer shadow edge, hazed toward the sky colour
    (aerial perspective).
Returns straight RGBA (Hp, Wp, 4) in plate px.
"""
import math

import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise(w, h, cx, cy, rng):
    """Smooth isotropic noise, zero mean / unit std, feature size ~cx, cy px."""
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


def _rot_noise(w, h, lx, ly, ang, octaves, rng):
    """fbm stretched along angle ang (radians): features lx long, ly across."""
    D = int(math.hypot(w, h)) + 4
    n = _fbm(D, D, lx, ly, octaves, rng)
    M = cv2.getRotationMatrix2D((D / 2, D / 2), -math.degrees(ang), 1.0)
    n = cv2.warpAffine(n, M, (D, D), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    y0, x0 = (D - h) // 2, (D - w) // 2
    return np.ascontiguousarray(n[y0:y0 + h, x0:x0 + w])


def _gauss(F, x, y, rx, ry, ang, wgt=1.0, mode='sum'):
    h, w = F.shape
    R = 3.0 * max(rx, ry)
    x0, x1 = int(max(x - R, 0)), int(min(x + R + 1, w))
    y0, y1 = int(max(y - R, 0)), int(min(y + R + 1, h))
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - x, yy - y
    c, s = math.cos(ang), math.sin(ang)
    u = dx * c + dy * s
    v = -dx * s + dy * c
    g = wgt * np.exp(-((u / rx) ** 2 + (v / ry) ** 2) * 0.6931)       # 0.5 at the radius
    if mode == 'sum':
        F[y0:y1, x0:x1] += g
    else:
        F[y0:y1, x0:x1] = np.maximum(F[y0:y1, x0:x1], g)


def _col_dist(mask, up):
    """Per-column run length inside `mask` to the nearest outside pixel above (up=True) or below."""
    h, w = mask.shape
    d = np.zeros((h, w), np.float32)
    m = mask.astype(np.float32)
    rng_ = range(h) if up else range(h - 1, -1, -1)
    prev = np.zeros(w, np.float32)
    for y in rng_:
        prev = (prev + 1.0) * m[y]
        d[y] = prev
    return d


def _row_dist(mask, from_right):
    return _col_dist(mask.T, not from_right).T


def _shape(cl, rng):
    """Density field (0.5 = silhouette) of one cloud in its bbox; returns F, (bx0, by0)."""
    T, L = cl['T'], cl['L']
    kind = cl['kind']
    a = math.radians(cl['ang'])
    c, sn = math.cos(a), math.sin(a)
    X, Y = cl['x'], cl['y']
    big, flo = [], []

    def wpos(u_, v_):
        return X + u_ * c - v_ * sn, Y + u_ * sn + v_ * c

    if kind == 'cu':
        warm_ = cl.get('warm', 0) > 0
        n = int(rng.integers(5, 8)) if warm_ else int(rng.integers(3, 6))
        us = (np.linspace(-0.36, 0.36, n) + rng.uniform(-0.05, 0.05, n)) * L if warm_ else             np.sort(rng.uniform(-0.34, 0.34, n)) * L
        for u_ in us:
            q = abs(u_) / (0.5 * L)
            r = T * rng.uniform(0.34, 0.5) * (1 - 0.45 * q ** 1.4)
            v_ = -0.35 * r + rng.normal(0, 0.03 * T)
            wx, wy = wpos(u_, v_)
            big.append((wx, wy, r * rng.uniform(1.3, 1.7), r, a))
            # one ring of cauliflower florets on the sky side (silhouette only; paint ignores them)
            for _ in range(int(rng.integers(3, 6))):
                th = math.radians(-90 + rng.uniform(-68, 68))
                rf = r * rng.uniform(0.3, 0.5)
                rx_, ry_ = r * 1.5, r
                du, dv = math.cos(th), math.sin(th)
                rr = 1.0 / math.sqrt((du / rx_) ** 2 + (dv / ry_) ** 2)
                fx_, fy_ = wpos(u_ + du * (rr - 0.55 * rf), v_ + dv * (rr - 0.55 * rf))
                flo.append((fx_, fy_, rf * rng.uniform(1.0, 1.25), rf, 0.0))
        # a broad flat skirt the masses sit on (night cumulus only)
        if not warm_:
            big.append((X, Y + 0.02 * T, 0.46 * L, 0.16 * T, a))
        else:
            big.append((X, Y + 0.0 * T, 0.4 * L, 0.26 * T, a))
    elif kind == 'st':
        N = max(int(L / (0.12 * T + 3)), 6)
        uu = np.linspace(-0.5, 0.5, N) * L
        env = np.clip(1 - np.abs(uu / (0.5 * L)) ** 1.8, 0, 1) ** 0.7
        m1 = rng.normal(0, 1, N).cumsum()
        m1 = (m1 - m1.mean()) / (m1.std() + 1e-6)
        for i in range(N):
            th = T * (0.16 + 0.1 * rng.random()) * env[i]
            if th < 0.03 * T:
                continue
            wx, wy = wpos(uu[i], 0.12 * T * m1[i])
            big.append((wx, wy, L / N * 1.6 + th, th, a))
        hk = cl.get('heads', (1, 2))
        for j in range(int(rng.integers(hk[0], hk[1] + 1))):
            u_ = rng.uniform(-0.35, 0.25) * L
            i = int(np.clip((u_ / L + 0.5) * (N - 1), 0, N - 1))
            r = T * rng.uniform(0.2, 0.32) * (0.5 + 0.5 * env[i])
            wx, wy = wpos(u_, 0.12 * T * m1[i] - 0.3 * r)
            big.append((wx, wy, r * 1.4, r, a))
            for _ in range(int(rng.integers(2, 4))):
                th = math.radians(-90 + rng.uniform(-60, 60))
                rf = r * rng.uniform(0.35, 0.55)
                flo.append((wx + math.cos(th) * r * 1.1, wy + math.sin(th) * r * 0.8, rf * 1.15, rf, 0.0))
    else:  # 'ci' cirrus veil: handled by the caller (no metaballs)
        big.append((X, Y, 0.5 * L, 0.5 * T, a))
    allp = big + flo
    pad = int(0.25 * T + 24)
    rmax = max(max(p[2], p[3]) for p in allp)
    bx0 = int(min(p[0] for p in allp) - 1.3 * rmax - pad)
    bx1 = int(max(p[0] for p in allp) + 1.3 * rmax + pad)
    by0 = int(min(p[1] for p in allp) - 1.3 * rmax - pad)
    by1 = int(max(p[1] for p in allp) + 1.3 * rmax + pad)
    w, h = bx1 - bx0, by1 - by0
    F = np.zeros((h, w), np.float32)
    if kind == 'ci':
        return F, (bx0, by0)
    for (wx, wy, rx, ry, aa) in big:
        _gauss(F, wx - bx0, wy - by0, rx, ry, aa, 0.62, 'sum')
    for (wx, wy, rx, ry, aa) in flo:
        _gauss(F, wx - bx0, wy - by0, rx, ry, aa, 0.62, 'max')
    return F, (bx0, by0)


def night_clouds19(Wp, Hp, clusters, H, light_xy, sky, tail_glow=None, seed=19, sun_x=None):
    s = H / 1080.0
    out = np.zeros((Hp, Wp, 4), np.float32)
    tg = None
    if tail_glow is not None:
        q = 8
        sm = cv2.resize(tail_glow, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        tg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 0.03 * H / q), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    if sun_x is None:
        sun_x = 0.3 * Wp
    # night palette (yn_05): pale cyan-white moonlit top plane, flat blue-grey body, a slightly deeper base
    TOP = np.array([0.8, 0.9, 1.0], np.float32)
    TOP2 = np.array([0.62, 0.74, 0.92], np.float32)
    MID = np.array([0.4, 0.49, 0.74], np.float32)
    LOW = np.array([0.3, 0.36, 0.64], np.float32)
    # twilight banks (yn_02)
    VTOP = np.array([0.36, 0.31, 0.52], np.float32)
    VBODY = np.array([0.5, 0.33, 0.5], np.float32)
    PINK = np.array([0.92, 0.5, 0.6], np.float32)
    HOT = np.array([1.35, 0.66, 0.36], np.float32)

    order = sorted(range(len(clusters)), key=lambda i: (-clusters[i].get('far', 0.0), clusters[i]['y']))
    for ic in order:
        cl = clusters[ic]
        rng = np.random.default_rng(seed * 1000 + ic * 7 + 3)
        T = cl['T']
        kind = cl['kind']
        wm = float(cl.get('warm', 0.0))
        haze = float(cl.get('haze', 0.0))
        F, (bx0, by0) = _shape(cl, rng)
        h, w = F.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        ang = math.radians(cl['ang'])
        if kind == 'ci':
            # ---- cirrus: wind-stretched fibrous veil, soft everywhere, lighter along its upper side
            c, sn = math.cos(ang), math.sin(ang)
            u = ((xx + bx0 - cl['x']) * c + (yy + by0 - cl['y']) * sn) / (0.5 * cl['L'])
            v = (-(xx + bx0 - cl['x']) * sn + (yy + by0 - cl['y']) * c) / (0.5 * T)
            bend = 0.35 * u * u
            env = np.clip(1 - np.abs(u) ** 2, 0, 1) * np.exp(-((v - bend) / 0.55) ** 2)
            fib = _rot_noise(w, h, 0.4 * cl['L'], 0.14 * T + 3, ang, 3, rng)
            fib2 = _rot_noise(w, h, 0.15 * cl['L'], 0.3 * T + 3, ang, 2, rng)
            dens = env * (0.6 + 0.3 * fib + 0.3 * fib2)
            A = _ss(0.2, 0.85, dens) * cl['amt'] * 0.55
            lt = _ss(0.2, -0.6, v - bend)             # upper side a touch brighter
            col = MID[None, None, :] * (1 - 0.5 * lt[..., None]) + TOP2 * (0.5 * lt[..., None])
            col = col + 0.06 * fib[..., None] * np.array([0.6, 0.8, 1.0], np.float32)
        else:
            # ---- silhouette: big irregular forms (domain warp) + cauliflower scallops on the sky side
            dxl = _noise(w, h, 0.4 * T, 0.4 * T, rng) * 0.06 * T
            dyl = _noise(w, h, 0.4 * T, 0.4 * T, rng) * 0.04 * T
            F = cv2.remap(F, xx + dxl, yy + dyl, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            Fb = cv2.GaussianBlur(F, (0, 0), 0.12 * T + 2)
            gyb = cv2.Sobel(Fb, cv2.CV_32F, 0, 1, ksize=3)
            gxb = cv2.Sobel(Fb, cv2.CV_32F, 1, 0, ksize=3)
            gnb = np.sqrt(gxb * gxb + gyb * gyb) + 1e-7
            upf = np.clip(gyb / gnb, 0, 1)            # edge faces up (inside is below)
            dnf = np.clip(-gyb / gnb, 0, 1)
            bil = (0.55 * np.abs(_noise(w, h, 0.14 * T + 3, 0.12 * T + 3, rng)) +
                   0.3 * np.abs(_noise(w, h, 0.06 * T + 2, 0.055 * T + 2, rng)) +
                   0.15 * np.abs(_noise(w, h, 0.028 * T + 1.5, 0.026 * T + 1.5, rng)))
            F = F + (0.21 if wm > 0 else 0.16) * (bil - 0.75) * (0.3 + 0.7 * upf)
            if kind == 'st':
                F = F + 0.07 * _rot_noise(w, h, 0.5 * T + 8, 0.06 * T + 3, ang, 2, rng)
            # wind-torn underside (stretched noise, only where the edge faces down)
            tear = _rot_noise(w, h, 0.35 * T + 8, 0.1 * T + 3, ang, 3, rng)
            F = F + 0.04 * tear * dnf
            thr = 0.5
            ins = (F > thr).astype(np.uint8)
            n_, lab, stt, _ = cv2.connectedComponentsWithStats(ins, connectivity=8)
            if n_ > 2:
                keep = stt[:, cv2.CC_STAT_AREA] >= max((0.14 * T) ** 2, 60 * s * s)
                keep[0] = True
                ins = (keep[lab] & (ins > 0)).astype(np.uint8)
                F = np.where(ins > 0, F, np.minimum(F, thr - 0.02))
            # fill pin-holes so the lit plane never shows through the body
            ins = cv2.morphologyEx(ins, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            gx = cv2.Sobel(F, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(F, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            sd = np.clip((F - thr) / (np.sqrt(gx * gx + gy * gy) + 1e-5), -8, 8)
            Aedge = _ss(-0.9, 0.9, sd)
            # ---- column distances: to the sky above (lit plane) and below (lost underside)
            d_up = _col_dist(ins, True)
            d_dn = _col_dist(ins, False)
            # smooth the column fields across columns (no vertical streaks under scallops)
            sgx = 0.035 * T + 2.0
            d_up = cv2.GaussianBlur(d_up, (0, 0), sigmaX=sgx, sigmaY=0.3 * sgx)
            d_dn = cv2.GaussianBlur(d_dn, (0, 0), sigmaX=1.0 * sgx, sigmaY=0.3 * sgx)
            lx_ = light_xy[0] - cl['x']
            d_side = _row_dist(ins, lx_ > 0)          # distance to the comet-facing side
            nz = _fbm(w, h, 0.25 * T + 4, 0.12 * T + 3, 2, rng)
            if wm <= 0:
                # lost underside: 20-40 px, broken into soft wind wisps
                Wd = float(np.clip(0.3 * T, 20 * s, 42 * s))
                fade = _ss(0.0, 1.0, d_dn / Wd + 0.18 * tear + 0.1)
                A = Aedge * fade
                # moonlit plane hanging from the top edge (painted: soft-edged, varying depth)
                Lw = (0.17 if kind == 'cu' else 0.24) * T * np.clip(1.0 + 0.3 * nz, 0.45, 1.6)
                lit_t = 1 - _ss(0.8, 1.1, d_up / Lw)
                side_ok = _ss(0.25, 0.6, abs(lx_) / (0.5 * cl['L'] + 1))
                lit_s = (1 - _ss(0.4, 1.2, d_side / (0.5 * Lw + 2))) * 0.55 * side_ok * _ss(0.0, 0.6, 1 - d_up / (T + 1))
                lit = np.maximum(lit_t, lit_s)
                lit = cv2.GaussianBlur(lit, (0, 0), 1.2 * s + 0.4)
                # body: one flat mid value, slightly deeper toward the base
                vpos = d_up / (d_up + d_dn + 1.0)
                vpos = cv2.GaussianBlur(vpos, (0, 0), 0.05 * T + 1)
                dk = _ss(0.45, 1.0, vpos)[..., None] * 0.7
                col = MID[None, None, :] * (1 - dk) + LOW * dk
                # a second, broad half-light just under the lit plane (painted transition, not emboss)
                hl = (1 - _ss(1.1, 2.6, d_up / Lw)) * (1 - lit)
                col = col * (1 - 0.35 * hl[..., None]) + TOP2 * (0.35 * hl[..., None])
                col = col * (1 - lit[..., None]) + TOP * lit[..., None]
                col = col * (1 + 0.03 * _fbm(w, h, 0.4 * T, 0.25 * T, 2, rng))[..., None]
                # translucent base: stars glow through the thin lower body
                A = A * (1 - 0.25 * _ss(0.5, 1.0, vpos))
            else:
                # ---- twilight bank: hot rim on the horizon-facing underside, fading inward
                Rw = 0.2 * T * np.clip(1.0 + 0.25 * nz, 0.5, 1.5)
                rimq = d_dn / Rw
                rdx = np.clip((sun_x - cl['x']) / (0.5 * Wp), -1, 1)
                sunk = math.exp(-((cl['x'] - sun_x) / (0.45 * Wp)) ** 2)
                side_d = _row_dist(ins, rdx > 0)
                side_d = cv2.GaussianBlur(side_d, (0, 0), sigmaX=0.3 * sgx, sigmaY=sgx)
                vpos = d_up / (d_up + d_dn + 1.0)
                vpos = cv2.GaussianBlur(vpos, (0, 0), 0.05 * T + 1)
                # the sun-side flank only warms in the lower half (never an outline over the top)
                rimq = np.minimum(rimq, side_d / (0.6 * Rw) + 3.0 * (1 - _ss(0.35, 0.8, vpos)) + 0.3)
                g_hot = (1 - _ss(0.0, 0.5, rimq)) ** 1.5
                g_pink = 1 - _ss(0.1, 2.2, rimq)
                col = VTOP[None, None, :] * (1 - vpos[..., None]) + VBODY * vpos[..., None]
                # faint cool moonlit plane on the cauliflower tops (reads the lobes, no outline)
                ltw = 1 - _ss(0.3, 1.3, d_up / (0.22 * T * np.clip(1.0 + 0.4 * nz, 0.4, 1.6)))
                ltw = cv2.GaussianBlur(ltw, (0, 0), 1.0 * s + 0.4)[..., None] * 0.28
                col = col * (1 - ltw) + np.array([0.52, 0.5, 0.74], np.float32) * ltw
                warmth = wm * (0.55 + 0.45 * sunk)
                pk = (g_pink * warmth * 0.85)[..., None]
                col = col * (1 - pk) + PINK * pk
                hk = (g_hot * warmth)[..., None]
                hotc = HOT * (0.75 + 0.25 * sunk) + np.array([0.0, -0.1, 0.2], np.float32) * (1 - sunk)
                col = col * (1 - hk) + hotc * hk
                col = cv2.GaussianBlur(col, (0, 0), 0.8 * s + 0.3)
                # top: cauliflower, but the shadow side is soft (lost into the sky)
                Wt = 3.0 * s + 1.0
                A = Aedge * _ss(0.0, 1.0, d_up / Wt + 0.2) * (0.3 + 0.7 * Aedge)
                A = np.maximum(A * 0.0, A)
        # ---- aerial perspective: haze toward the local sky colour
        xa, xb = max(bx0, 0), min(bx0 + w, Wp)
        ya, yb = max(by0, 0), min(by0 + h, Hp)
        if xb <= xa or yb <= ya:
            continue
        sl = (slice(ya - by0, yb - by0), slice(xa - bx0, xb - bx0))
        cc = col[sl]
        if haze > 0:
            skc = sky[ya:yb, xa:xb]
            cc = cc * (1 - haze * 0.4) + (skc * 1.05 + 0.02) * (haze * 0.4)
        if tg is not None:
            tpk = np.clip(tg[ya:yb, xa:xb], 0, 0.35)
            cc = cc + tpk * 0.4
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
