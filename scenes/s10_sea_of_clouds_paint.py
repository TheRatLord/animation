"""s10 painterly cloud-sea renderer (round 5 rewrite).

The sea of clouds is an anime multiplane: ~60 ROWS of painted cloud billboards standing at world depths
z in front of the camera (plus a few rows of translucent wisps).  Each row's top silhouette is the hard
max of hierarchical arcs (broad mounds -> rounded lobes -> small cauliflower bumps), generated
procedurally per cell from a hash so rows are infinite and deterministic.  Every frame each row is
projected with a pinhole camera (true parallax: near rows sweep by fast, far rows barely move) and
painted per pixel directly at screen resolution (crisp at any size):

  * BACKLIT lighting: the sun sits behind the sea, so the camera-facing faces are cool blue-violet
    shadow; only the upward-facing crowns pick up warm cream/peach light, strongest near the sun's
    azimuth; a thin 1.5-3 px silver-gold rim runs along every silhouette facing the sun, with a
    translucent forward-scatter glow creeping in from it.
  * no noise in the shading: soft internal gradients come from the lobe normals only.
  * deep valley shadow below each row's tops falls off smoothly into indigo.
  * atmospheric perspective: near rows saturated, mid rows paler/bluer, far rows pink/peach haze,
    compressing into thin bands at the horizon; under the sun the far sea turns to gold.

Towers are painted once as plates (paint_tower) and placed per frame at their depth.
"""
import math
import numpy as np
import cv2
from numba import njit, prange


# ============================================================================ hashing

@njit(cache=True, inline='always')
def _hash(a, b, c):
    h = (a * 374761393 + b * 668265263 + c * 2147483647 + 1013904223) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return h / 4294967296.0


@njit(cache=True, inline='always')
def _vnoise(x, seed):
    i = math.floor(x)
    fr = x - i
    ii = int(i)
    a = _hash(ii, seed, 7)
    b = _hash(ii + 1, seed, 7)
    s = fr * fr * (3.0 - 2.0 * fr)
    return a + (b - a) * s


@njit(cache=True, inline='always')
def _ss(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, inline='always')
def _cl(x, a, b):
    return a if x < a else (b if x > b else x)


# ============================================================================ row silhouettes

@njit(cache=True)
def _arc(T, D, P, c0, c1, cxs, camx, s, xc, r, a, base, pw, lev, rad):
    """Rasterise one arc (half-ellipse-ish dome) into the 1D top buffer T (max), keep the winner's
    parameters P[c] = (xc, r, a, base, lev, rad) and the slope D[c]."""
    W = T.shape[0]
    x0 = int(math.floor(cxs + (xc - r - camx) * s))
    x1 = int(math.ceil(cxs + (xc + r - camx) * s)) + 1
    if x0 < c0:
        x0 = c0
    if x1 > c1:
        x1 = c1
    for c in range(x0, x1):
        xw = camx + (c + 0.5 - cxs) / s
        u = (xw - xc) / r
        q = 1.0 - u * u
        if q <= 0.0:
            continue
        sq = q ** pw
        y = base + a * sq
        if y > T[c]:
            T[c] = y
            # dy/dx = a * pw * q^(pw-1) * (-2u / r)
            D[c] = -a * pw * (q ** (pw - 1.0)) * 2.0 * u / r
            P[c, 0] = xc
            P[c, 1] = r
            P[c, 2] = a
            P[c, 3] = base
            P[c, 4] = lev
            P[c, 5] = rad


@njit(cache=True)
def _row_tops(T, D, P, cxs, camx, s, cw, A, yb, seed, W, asp, empty):
    """Generate a row's cells (hash-deterministic) visible in [0, W) and rasterise all arcs."""
    xw0 = camx + (0.0 - cxs) / s
    xw1 = camx + (W - cxs) / s
    j0 = int(math.floor(xw0 / cw)) - 2
    j1 = int(math.floor(xw1 / cw)) + 2
    for j in range(j0, j1 + 1):
        if _hash(j, seed, 1) < empty:
            continue
        xc = (j + 0.5 + 0.5 * (_hash(j, seed, 2) - 0.5)) * cw
        rm = cw * (0.55 + 0.45 * _hash(j, seed, 3))
        hm = A * (0.3 + 0.7 * _hash(j, seed, 4) ** 0.8)
        b0 = yb + A * 0.25 * (_hash(j, seed, 5) - 0.5)
        _arc(T, D, P, 0, W, cxs, camx, s, xc, rm, hm, b0, 0.6, 0.0, hm)
        nl = 3 + int(_hash(j, seed, 6) * 4.0)
        for k in range(nl):
            kk = 100 + k * 17
            ul = -0.82 + 1.64 * (k + 0.2 + 0.6 * _hash(j, seed, kk)) / nl
            xl = xc + rm * ul
            rl = rm * (0.2 + 0.2 * _hash(j, seed, kk + 1))
            al = rl * asp * (0.8 + 0.25 * _hash(j, seed, kk + 2))
            qm = 1.0 - ul * ul
            if qm <= 0.02:
                continue
            hb = b0 + hm * qm ** 0.6
            lb = hb - al * (0.35 + 0.25 * _hash(j, seed, kk + 3))
            _arc(T, D, P, 0, W, cxs, camx, s, xl, rl, al, lb, 0.5, 1.0, al)
            nb = 1 + int(_hash(j, seed, kk + 4) * 3.0)
            for m in range(nb):
                mm = kk + 5 + m * 5
                ub = -0.7 + 1.4 * _hash(j, seed, mm)
                xb = xl + rl * ub
                rb = rl * (0.24 + 0.2 * _hash(j, seed, mm + 1))
                ab = rb * asp * (0.8 + 0.2 * _hash(j, seed, mm + 2))
                qb = 1.0 - ub * ub
                if qb <= 0.05:
                    continue
                hbb = lb + al * qb ** 0.5
                _arc(T, D, P, 0, W, cxs, camx, s, xb, rb, ab, hbb - ab * 0.55, 0.5, 2.0, ab)



# ============================================================================ 2D lobed rows (ellipsoid z-buffer)

@njit(cache=True, inline='always')
def _put(E, n, x, y, r, a, z, par, lev, ph):
    E[n, 0] = x
    E[n, 1] = y
    E[n, 2] = r
    E[n, 3] = a
    E[n, 4] = z
    E[n, 5] = par
    E[n, 6] = lev
    E[n, 7] = r * 0.8
    E[n, 8] = ph


@njit(cache=True)
def _gen_row(E, cw, A, yb, seed, xw0, xw1, asp, empty):
    """Fill E[k] = (xc, yc, r, a, zc, parent, lev, bz, phase) (world units, y up) for the row's cells
    within [xw0, xw1]: wide flat mounds -> overlapping front billows (scallops) -> crown lobes along the
    upper contour -> small cauliflower bumps on the lobes' upper contours."""
    n = 0
    NE = E.shape[0]
    j0 = int(math.floor(xw0 / cw)) - 2
    j1 = int(math.floor(xw1 / cw)) + 2
    for j in range(j0, j1 + 1):
        if _hash(j, seed, 1) < empty:
            continue
        if n + 90 >= NE:
            break
        xc = (j + 0.5 + 0.6 * (_hash(j, seed, 2) - 0.5)) * cw
        rm = cw * (0.75 + 0.55 * _hash(j, seed, 3))
        hm = A * (0.3 + 0.7 * _hash(j, seed, 4) ** 0.8)
        am = hm * 1.3
        yc = yb + hm - am + A * 0.2 * (_hash(j, seed, 5) - 0.5)
        zc = rm * 0.3 * (_hash(j, seed, 8) - 0.5)
        m = n
        _put(E, n, xc, yc, rm, am, zc, -1.0, 0.0, _hash(j, seed, 10) * 6.28)
        n += 1
        # front billows: overlapping flat scallops across the upper face
        nfb = 3 + int(_hash(j, seed, 9) * 4.0)
        for k in range(nfb):
            kk = 300 + k * 11
            Xf = -0.7 + 1.4 * (k + 0.2 + 0.6 * _hash(j, seed, kk)) / nfb
            Yf = -0.1 + 0.75 * _hash(j, seed, kk + 1)
            qf = 1.0 - Xf * Xf - Yf * Yf
            if qf <= 0.05:
                continue
            rf = rm * (0.3 + 0.25 * _hash(j, seed, kk + 2))
            af = rf * (0.38 + 0.14 * _hash(j, seed, kk + 3))
            _put(E, n, xc + Xf * rm, yc + Yf * am, rf, af, zc + rm * 0.8 * math.sqrt(qf) - rf * 0.5, m, 1.0,
                 _hash(j, seed, kk + 4) * 6.28)
            f_i = n
            n += 1
            # a couple of bumps along the scallop's upper contour
            for q in range(3):
                mm = kk + 5 + q * 3
                pb = math.pi * (0.2 + 0.6 * _hash(j, seed, mm))
                rb = rf * (0.25 + 0.15 * _hash(j, seed, mm + 1))
                _put(E, n, xc + Xf * rm + math.cos(pb) * rf * 0.9, yc + Yf * am + math.sin(pb) * af * 0.9,
                     rb, rb * 0.88, E[f_i, 4] + rf * 0.8 * 0.35 * math.sin(pb) - rb * 0.3, f_i, 2.0,
                     _hash(j, seed, mm + 2) * 6.28)
                n += 1
        # crown lobes along the upper contour
        nl = 7 + int(_hash(j, seed, 6) * 6.0)
        for k in range(nl):
            kk = 100 + k * 17
            ph = math.pi * (0.06 + 0.88 * (k + 0.15 + 0.7 * _hash(j, seed, kk)) / nl)
            rl = rm * (0.1 + 0.15 * _hash(j, seed, kk + 1)) * (0.6 + 0.4 * math.sin(ph))
            al = rl * (0.5 + 0.2 * _hash(j, seed, kk + 2))
            rad = 0.9 + 0.1 * _hash(j, seed, kk + 3)
            xl = xc + math.cos(ph) * rm * rad
            yl = yc + math.sin(ph) * am * rad
            kf = _hash(j, seed, kk + 4)
            zl = zc + rm * 0.8 * math.sqrt(max(1.0 - rad * rad, 0.0)) * kf * 1.3 - rl * 0.25 + rl * 0.5 * kf
            _put(E, n, xl, yl, rl, al, zl, m, 1.0, _hash(j, seed, kk + 5) * 6.28)
            li = n
            n += 1
            nb = 1 + int(_hash(j, seed, kk + 6) * 3.5)
            for q in range(nb):
                mm = kk + 7 + q * 5
                pb = math.pi * (0.12 + 0.76 * _hash(j, seed, mm))
                rb = rl * (0.3 + 0.2 * _hash(j, seed, mm + 1))
                ab = rb * 0.9
                xb = xl + math.cos(pb) * rl * 0.93
                ybb = yl + math.sin(pb) * al * 0.93
                zb = zl + rl * 0.8 * 0.5 * _hash(j, seed, mm + 2) * math.sin(pb) - rb * 0.3
                _put(E, n, xb, ybb, rb, ab, zb, li, 2.0, _hash(j, seed, mm + 3) * 6.28)
                n += 1
    return n


@njit(cache=True, inline='always')
def _cmod(X, Y, ph):
    """Irregular (hand-painted) contour: radius modulation by angle."""
    th = math.atan2(Y, X)
    return 1.0 + 0.045 * math.sin(3.0 * th + ph) + 0.03 * math.sin(5.0 * th + 1.9 * ph) + \
        0.016 * math.sin(11.0 * th + 2.7 * ph)


@njit(cache=True, inline='always')
def _enorm(E, i, xw, yw):
    X = (xw - E[i, 0]) / E[i, 2]
    Y = (yw - E[i, 1]) / E[i, 3]
    q = X * X + Y * Y
    nz = math.sqrt(max(1.0 - q, 0.0))
    return X, Y, nz


@njit(cache=True, fastmath=True, parallel=True)
def _paint_row(img, cov, E, n, s, z, hy, cxs, camx, camy, A, yb, sx, sy, W0, pal, fz1, fz2, light, wr, wg):
    H, W = img.shape[0], img.shape[1]
    if n == 0:
        return
    ytop = 1e9
    ybot = -1e9
    for i in range(n):
        yt = hy + (camy - (E[i, 1] + E[i, 3])) * s
        ybm = hy + (camy - (E[i, 1] - E[i, 3])) * s
        if yt < ytop:
            ytop = yt
        if ybm > ybot:
            ybot = ybm
    r0 = int(max(math.floor(ytop) - 1, 0))
    r1 = int(min(math.ceil(ybot) + 1, H))
    if r0 >= r1:
        return
    c_face = pal[0]
    c_fill = pal[1]
    c_val = pal[2]
    c_pink = pal[3]
    c_warm = pal[4]
    c_rim = pal[5]
    c_glow = pal[6]
    c_mid = pal[7]
    c_far_lo = pal[8]
    c_far_hi = pal[9]
    for r in prange(r0, r1):
        ys = r + 0.5
        yw = camy - (ys - hy) / s
        h1 = np.full(W, -1e18)
        h2 = np.full(W, -1e18)
        i1 = np.full(W, -1, np.int64)
        i2 = np.full(W, -1, np.int64)
        e1 = np.zeros(W)
        w1 = np.zeros(W)
        ca = np.zeros(W)
        ic = np.full(W, -1, np.int64)
        for i in range(n):
            a = E[i, 3]
            Y = (yw - E[i, 1]) / a
            my = 1.1 + 1.0 / (a * s)
            if Y <= -my or Y >= my:
                continue
            rr = E[i, 2]
            Ys = Y / 1.1
            qy = 1.0 - Ys * Ys
            hw = rr * 1.1 * math.sqrt(qy) if qy > 0 else 0.0
            hw += 1.0 / s
            c0 = int(math.floor(cxs + (E[i, 0] - hw - camx) * s))
            c1 = int(math.ceil(cxs + (E[i, 0] + hw - camx) * s)) + 1
            if c0 < 0:
                c0 = 0
            if c1 > W:
                c1 = W
            for c in range(c0, c1):
                xw = camx + (c + 0.5 - cxs) / s
                X = (xw - E[i, 0]) / rr
                rho = math.sqrt(X * X + Y * Y) / _cmod(X, Y, E[i, 8]) + 1e-9
                q = rho * rho
                gx = X / (rr * rho)
                gy = Y / (a * rho)
                gl = math.sqrt(gx * gx + gy * gy) + 1e-9
                dpx = (1.0 - rho) / gl * s
                # crisp upper contours, soft melting lower edges (clouds, not pebbles)
                sf = _ss(0.15, -0.75, Y)
                soft_px = 1.0 + sf * 0.55 * a * s
                cv = (dpx + 0.5) / soft_px
                if cv <= 0.0:
                    continue
                if cv > 1.0:
                    cv = 1.0
                if cv > ca[c]:
                    ca[c] = cv
                    ic[c] = i
                if q < 1.0:
                    h = E[i, 4] + E[i, 7] * math.sqrt(1.0 - q)
                    if h > h1[c]:
                        h2[c] = h1[c]
                        i2[c] = i1[c]
                        h1[c] = h
                        i1[c] = i
                        e1[c] = dpx
                        w1[c] = cv
                    elif h > h2[c]:
                        h2[c] = h
                        i2[c] = i
        dsy = sy - ys
        valk = _ss(yb + 0.15 * A, yb - 0.9 * A, yw) * 0.9
        for c in range(W):
            al = ca[c]
            if al <= 0.0:
                continue
            xs = c + 0.5
            xw = camx + (xs - cxs) / s
            wx = (xs - sx) / (0.3 * W0)
            wpath = 0.05 * W0 + 0.55 * (ys - hy)
            if wpath < 0.03 * W0:
                wpath = 0.03 * W0
            wq = (xs - sx) / wpath
            path = math.exp(-wq * wq)
            prox = math.exp(-wx * wx) * 0.6 + 0.4 * path
            proxw = math.exp(-wx * wx * 0.3)
            dsx = sx - xs
            dl = math.sqrt(dsx * dsx + dsy * dsy) + 1e-3
            ux = dsx / dl
            uy = -dsy / dl
            a_ = i1[c]
            b_ = i2[c]
            wa = 1.0
            if a_ < 0:
                a_ = ic[c]
                b_ = -1
                ed = 0.0
            else:
                ed = e1[c]
                if b_ >= 0:
                    wa = w1[c]
                    if wa > 1.0:
                        wa = 1.0
            col0 = 0.0
            col1 = 0.0
            col2 = 0.0
            for pass_ in range(2):
                if pass_ == 0:
                    ii = a_
                    ww = wa
                else:
                    ii = b_
                    ww = 1.0 - wa
                if ii < 0 or ww <= 0.0:
                    continue
                X = (xw - E[ii, 0]) / E[ii, 2]
                Y = (yw - E[ii, 1]) / E[ii, 3]
                rho = math.sqrt(X * X + Y * Y) / _cmod(X, Y, E[ii, 8]) + 1e-9
                gx = X / (E[ii, 2] * rho)
                gy = Y / (E[ii, 3] * rho)
                gln = math.sqrt(gx * gx + gy * gy) + 1e-9
                if pass_ == 0 and i1[c] >= 0:
                    edpx = ed
                elif pass_ == 0:
                    edpx = 0.0
                else:
                    edpx = (1.0 - rho) / gln * s
                if edpx < 0.0:
                    edpx = 0.0
                # contour normal (2D) and how much it faces the sun on screen
                facing = (gx * ux + gy * uy) / gln
                if facing < 0.0:
                    facing = 0.0
                # painted value: the ROW is one mass (valley indigo -> blue face -> lighter lavender near
                # its top level); each lobe only adds a subtle lift toward its own crown
                mrow = _ss(yb - 0.9 * A, yb + 0.9 * A, yw)
                Yc = Y / math.sqrt(max(1.0 - X * X * 0.85, 0.08))
                lift = _ss(-0.2, 0.9, Yc) * 0.22
                # warm light only on the crowns (narrow band along each lobe's upper contour); gold near
                # the sun's azimuth, peach-pink away from it; a narrow pink transition just below
                top = _ss(0.72 - 0.25 * path, 0.97, Yc) * (0.4 + 0.6 * facing)
                wl = top * (0.3 + 0.7 * prox) * light
                wp = _ss(0.5, 0.72, Yc) * (1.0 - _ss(0.72, 0.92, Yc)) * 0.3 * (0.35 + 0.65 * prox)
                wmix = _cl(prox * 1.3, 0.0, 1.0)
                inner = 1.0 if (pass_ == 0 and b_ >= 0) else 0.0
                rk = 1.0 - 0.4 * inner
                rim = math.exp(-edpx / wr) * facing ** 0.7 * (0.3 + 0.7 * prox) * rk
                tr = (math.exp(-edpx / wg) * 0.45 + math.exp(-edpx / (wg * 4.0)) * 0.35 * path) * facing * (0.2 + 0.8 * prox) * rk
                for k in range(3):
                    cc = c_val[k] + (c_face[k] - c_val[k]) * (0.25 + 0.75 * mrow)
                    cc = cc + (c_fill[k] - cc) * (_ss(0.55, 1.0, mrow) * 0.3 + lift)
                    cc = cc + (c_pink[k] - cc) * wp
                    cwk = c_pink[k] + (c_warm[k] - c_pink[k]) * (0.45 + 0.55 * wmix)
                    cc = cc + (cwk - cc) * _cl(wl, 0.0, 1.0)
                    cc = cc + (c_glow[k] - cc) * _cl(tr * light, 0.0, 0.85)
                    cm = cc + (c_mid[k] - cc) * fz1
                    hz = c_far_lo[k] + (c_far_hi[k] - c_far_lo[k]) * proxw
                    cc = cm + (hz - cm) * fz2
                    cc = cc + (c_rim[k] * light - cc) * _cl(rim * (1.0 - 0.55 * fz2), 0.0, 1.0)
                    if k == 0:
                        col0 += cc * ww
                    elif k == 1:
                        col1 += cc * ww
                    else:
                        col2 += cc * ww
            img[r, c, 0] = img[r, c, 0] * (1.0 - al) + col0 * al
            img[r, c, 1] = img[r, c, 1] * (1.0 - al) + col1 * al
            img[r, c, 2] = img[r, c, 2] * (1.0 - al) + col2 * al
            if al > cov[r, c]:
                cov[r, c] = al


# ============================================================================ per-frame sea

@njit(cache=True, fastmath=True, parallel=True)
def deck(img, hy, f, camy, ydeck, sx, W0, col_near, col_mid, col_far_lo, col_far_hi, zf, zf2):
    """The cloud-deck plane far below the tops (only visible through gaps between far rows)."""
    H, W = img.shape[0], img.shape[1]
    r0 = int(max(math.floor(hy), 0))
    for r in prange(r0, H):
        dy = r + 0.5 - hy
        if dy <= 0.05:
            dy = 0.05
        z = f * (camy - ydeck) / dy
        f1 = 1.0 - math.exp(-z / zf)
        f2 = 1.0 - math.exp(-z / zf2)
        for c in range(W):
            wx = (c + 0.5 - sx) / (0.3 * W0)
            prox = math.exp(-wx * wx)
            for k in range(3):
                cm = col_near[k] + (col_mid[k] - col_near[k]) * f1
                hz = col_far_lo[k] + (col_far_hi[k] - col_far_lo[k]) * prox
                img[r, c, k] = cm + (hz - cm) * f2


@njit(cache=True, fastmath=True, parallel=True)
def _shade_row(img, cov, T, D, P, hy, f, s, z, cxs, camx, camy, A, sx, sy, W0, pal, fz1, fz2, light, wr, wg,
               is_wisp, wseed, wdrift, yb):
    H, W = img.shape[0], img.shape[1]
    # vertical extent on screen
    tmax = -1e9
    for c in range(W):
        if T[c] > tmax:
            tmax = T[c]
    if tmax < -1e8:
        return
    ytop = int(math.floor(hy + (camy - tmax) * s)) - 2
    if is_wisp:
        ytop = int(math.floor(hy + (camy - (tmax + 3.0 * A)) * s)) - 2
        ybot = int(math.ceil(hy + (camy - (yb - 0.25)) * s)) + 2
    else:
        ybot = int(math.ceil(hy + (camy - (yb - 3.0 * A)) * s)) + 2
    if ytop < 0:
        ytop = 0
    if ybot > H:
        ybot = H
    if ytop >= ybot:
        return
    # palette rows
    c_face = pal[0]
    c_fill = pal[1]
    c_val = pal[2]
    c_pink = pal[3]
    c_warm = pal[4]
    c_rim = pal[5]
    c_glow = pal[6]
    c_mid = pal[7]
    c_far_lo = pal[8]
    c_far_hi = pal[9]
    c_wisp = pal[10]
    for r in prange(ytop, ybot):
        ys = r + 0.5
        yw = camy - (ys - hy) / s
        dsy = sy - ys
        for c in range(W):
            Tc = T[c]
            if Tc < -1e8:
                continue
            xs = c + 0.5
            xw = camx + (xs - cxs) / s
            wx = (xs - sx) / (0.3 * W0)
            prox = math.exp(-wx * wx)
            proxw = math.exp(-wx * wx * 0.3)
            if is_wisp:
                # translucent wisp: gaussian around the band centre, combed along x
                cy_ = Tc
                th = P[c, 2]
                dyw = (yw - cy_) / th
                g = math.exp(-dyw * dyw)
                if g < 0.01:
                    continue
                n1 = _vnoise(xw * 0.8 + wdrift, wseed) * 0.65 + _vnoise(xw * 2.3 + wdrift * 1.7, wseed + 3) * 0.35
                fib = _vnoise(xw * 1.0 + yw * 0.0 + wdrift * 0.5, wseed + 11) * 0.5 + \
                    _vnoise((yw - cy_) / th * 4.0 + xw * 0.12, wseed + 17) * 0.5
                al = _ss(0.42, 0.78, n1) * g * (0.55 + 0.45 * fib) * P[c, 5]
                if al < 0.003:
                    continue
                up = _cl(-dyw * 0.8 + 0.5, 0.0, 1.0)
                for k in range(3):
                    cc = c_wisp[k] + (c_glow[k] - c_wisp[k]) * (0.25 + 0.75 * prox) * (0.35 + 0.65 * up) * light
                    cm = cc + (c_mid[k] - cc) * fz1 * 0.5
                    hz = c_far_lo[k] + (c_far_hi[k] - c_far_lo[k]) * prox
                    cc = cm + (hz - cm) * fz2
                    img[r, c, k] = img[r, c, k] * (1.0 - al) + cc * al
                continue
            sw = Tc - yw
            sl = D[c]
            sp = sw * s / math.sqrt(1.0 + sl * sl)
            al = sp + 0.5
            if al <= 0.0:
                continue
            if al > 1.0:
                al = 1.0
            xc = P[c, 0]
            rr_ = P[c, 1]
            aa = P[c, 2]
            bs = P[c, 3]
            lev = P[c, 4]
            u = (xw - xc) / rr_
            v = (yw - bs) / aa
            # lobe normal (x right, y up, z toward camera)
            if v > 0.0:
                q = u * u + v * v
                if q < 1.0:
                    nz = math.sqrt(1.0 - q)
                    nx = u
                    ny = v
                else:
                    iq = 1.0 / math.sqrt(q)
                    nx = u * iq
                    ny = v * iq
                    nz = 0.0
            else:
                nx = u * 0.45
                ny = v * 0.35
                nz = 1.0
                ln = math.sqrt(nx * nx + ny * ny + nz * nz)
                nx /= ln
                ny /= ln
                nz /= ln
            # sun direction on screen from this pixel (y up)
            dsx = sx - xs
            dl = math.sqrt(dsx * dsx + dsy * dsy) + 1e-3
            ux = dsx / dl
            uy = -dsy / dl
            n2 = math.sqrt(nx * nx + ny * ny) + 1e-5
            facing = (nx * ux + ny * uy) / n2
            if facing < 0.0:
                facing = 0.0
            crown = ny + 0.3 * facing * (1.0 - nz)
            # ---- base: cool faces, sky-lit upward planes
            kf = _cl(ny * 0.9 + 0.1, 0.0, 1.0) * 0.55
            wl = _ss(0.42, 0.9, crown) * (0.3 + 0.7 * prox) * light
            wp = _ss(0.2, 0.55, crown) * (1.0 - _ss(0.6, 0.95, crown)) * 0.3 * (0.4 + 0.6 * prox)
            q2 = sw / (A + 1e-4)
            val = _ss(0.12, 1.25, q2) * 0.92
            rim = math.exp(-sp / wr) * facing ** 0.6 * (0.35 + 0.65 * prox) * (1.0 - 0.3 * lev * 0.0)
            tr = math.exp(-sp / wg) * facing * (0.2 + 0.8 * prox) * 0.5
            for k in range(3):
                cc = c_face[k] + (c_fill[k] - c_face[k]) * kf
                cc = cc + (c_pink[k] - cc) * wp
                cc = cc + (c_warm[k] - cc) * wl
                cc = cc + (c_val[k] - cc) * val
                cc = cc + (c_glow[k] - cc) * _cl(tr * light, 0.0, 0.85)
                # atmospheric perspective: mid (paler/bluer) then far haze (pink -> gold near the sun)
                cm = cc + (c_mid[k] - cc) * fz1
                hz = c_far_lo[k] + (c_far_hi[k] - c_far_lo[k]) * proxw
                cc = cm + (hz - cm) * fz2
                cc = cc + (c_rim[k] * light - cc) * _cl(rim * (1.0 - 0.55 * fz2), 0.0, 1.0)
                img[r, c, k] = img[r, c, k] * (1.0 - al) + cc * al
            if al > cov[r, c]:
                cov[r, c] = al


@njit(cache=True)
def draw_rows(img, cov, rows, i0, i1, d, hy, f, cxs, camx, camy, sx, sy, W0, pal, light, wr, wg, zf, zf2, t):
    """Draw rows[i0:i1] (sorted far -> near) into img in place. rows columns:
    z, cw, A, yb, seed, type(0 cloud / 1 wisp), asp, empty, drift."""
    W = img.shape[1]
    T = np.empty(W, np.float64)
    D = np.empty(W, np.float64)
    P = np.empty((W, 6), np.float64)
    E = np.empty((12000, 9), np.float64)
    for i in range(i0, i1):
        z = rows[i, 0] - d
        if z < 0.5:
            continue
        s = f / z
        cw = rows[i, 1]
        A = rows[i, 2]
        yb = rows[i, 3]
        seed = int(rows[i, 4])
        typ = int(rows[i, 5])
        fz1 = 1.0 - math.exp(-z / zf)
        fz2 = (1.0 - math.exp(-z / zf2)) ** 1.3
        for c in range(W):
            T[c] = -1e9
            D[c] = 0.0
        if typ == 0:
            xw0 = camx + (0.0 - cxs) / s
            xw1 = camx + (W - cxs) / s
            n = _gen_row(E, cw, A, yb, seed, xw0, xw1, rows[i, 6], rows[i, 7])
            _paint_row(img, cov, E, n, s, z, hy, cxs, camx, camy, A, yb, sx, sy, W0, pal, fz1 * 0.6, fz2,
                       light, wr, wg)
        else:
            # wisp band: centre height yb, thickness A; per-column fade envelope
            for c in range(W):
                xw = camx + (c + 0.5 - cxs) / s
                env = _vnoise(xw * 0.18 + rows[i, 8] * t * 0.05, seed + 5)
                T[c] = yb + 0.04 * math.sin(xw * 0.7 + seed)
                P[c, 2] = A * (0.7 + 0.6 * _vnoise(xw * 0.9, seed + 9))
                P[c, 5] = _ss(0.3, 0.65, env) * 0.85
            _shade_row(img, cov, T, D, P, hy, f, s, z, cxs, camx, camy, A, sx, sy, W0, pal, fz1 * 0.8, fz2,
                       light, wr, wg, True, seed, rows[i, 8] * t, yb)


def make_rows(seed=7):
    """World rows: (z, cw, A, yb, seed, type, asp, empty, drift)."""
    rng = np.random.default_rng(seed)
    out = []
    z = 1.6
    k = 0
    while z < 900:
        big = max(1.0, z / 60.0) ** 0.85
        cw = rng.uniform(1.4, 2.2) * big
        A = rng.uniform(0.3, 0.5) * big ** 0.95
        yb = rng.uniform(-0.12, 0.06) * big
        empty = 0.12 if z < 200 else 0.25
        asp = rng.uniform(0.62, 0.8)
        out.append((z, cw, A, yb, 1000 + k, 0, asp, empty, 0.0))
        k += 1
        z *= rng.uniform(1.09, 1.16) if z < 120 else rng.uniform(1.12, 1.2)
    # a final, very far row of tall cumulus bumps that break the horizon line
    for i, zz in enumerate((1300.0, 1900.0)):
        out.append((zz, 38.0 + 10 * i, 26.0 + 6 * i, -12.0, 5000 + i, 0, 0.75, 0.45, 0.0))
    # wisps skimming the sea surface
    for i, (zz, yy) in enumerate(((6.5, 0.62), (8.5, 0.66), (11.0, 0.7), (15.0, 0.75), (21.0, 0.85))):
        out.append((zz, 1.0, 0.035 + 0.01 * i, yy, 7000 + i, 1, 0.0, 0.0, 1.0 + 0.4 * i))
    out.sort(key=lambda r: -r[0])
    return np.array(out, np.float64)


# ============================================================================ towers

@njit(cache=True, fastmath=True)
def _raster_domes(Hs, Ws, S):
    """Hard z-buffer of spheres S[i] = (cx, cy, cz, r, lev, parent_i, ...) in plate px (y down).
    Returns h (front surface z), id."""
    h = np.full((Hs, Ws), -1e12, np.float32)
    ids = np.full((Hs, Ws), -1, np.int32)
    for i in range(S.shape[0]):
        cx, cy, cz, r = S[i, 0], S[i, 1], S[i, 2], S[i, 3]
        x0 = max(0, int(cx - r - 1))
        x1 = min(Ws, int(cx + r + 2))
        y0 = max(0, int(cy - r - 1))
        y1 = min(Hs, int(cy + r + 2))
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / r
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                q = 1.0 - dx * dx - dy * dy
                if q <= 0.0:
                    continue
                zz = cz + r * math.sqrt(q)
                if zz > h[y, x]:
                    h[y, x] = zz
                    ids[y, x] = i
    return h, ids


@njit(cache=True, fastmath=True, parallel=True)
def _tower_normals(ids, S, Hs, Ws, fx, fy, fr, fh, base_y):
    """Blend each pixel's dome normal with its parent's and with the tower form normal."""
    N = np.zeros((Hs, Ws, 3), np.float32)
    for y in prange(Hs):
        for x in range(Ws):
            i = ids[y, x]
            if i < 0:
                continue
            acc0 = 0.0
            acc1 = 0.0
            acc2 = 0.0
            j = i
            wt = 1.0
            tot = 0.0
            for lv in range(3):
                cx, cy, r = S[j, 0], S[j, 1], S[j, 3]
                dx = (x + 0.5 - cx) / r
                dy = (y + 0.5 - cy) / r
                q = 1.0 - dx * dx - dy * dy
                if q < 0.0:
                    q = 0.0
                nz = math.sqrt(q)
                ln = math.sqrt(dx * dx + dy * dy + nz * nz) + 1e-6
                acc0 += wt * dx / ln
                acc1 += wt * (-dy) / ln
                acc2 += wt * nz / ln
                tot += wt
                p = int(S[j, 5])
                if p < 0:
                    break
                j = p
                wt *= 0.8
            # form normal: vertical column / rounded top
            v = (base_y - (y + 0.5)) / fh
            dx = (x + 0.5 - fx) / fr
            ftop = 0.0
            if v > 0.75:
                ftop = (v - 0.75) / 0.25
            fnx = dx * 0.9
            fny = 0.15 + 0.8 * ftop
            fnz = math.sqrt(max(1.0 - fnx * fnx, 0.05))
            ln = math.sqrt(fnx * fnx + fny * fny + fnz * fnz)
            lvl = S[i, 4]
            wf = 0.55 if lvl < 0.5 else 0.35
            n0 = acc0 / tot * (1 - wf) + fnx / ln * wf
            n1 = acc1 / tot * (1 - wf) + fny / ln * wf
            n2 = acc2 / tot * (1 - wf) + fnz / ln * wf
            ln = math.sqrt(n0 * n0 + n1 * n1 + n2 * n2) + 1e-6
            N[y, x, 0] = n0 / ln
            N[y, x, 1] = n1 / ln
            N[y, x, 2] = n2 / ln
    return N


def tower_domes(rng, cx, base_y, width, height, lean=0.0, crown_w=1.0):
    """Cumulus tower as a hierarchy of spheres (plate px, y down). Returns list of
    (cx, cy, cz, r, lev, parent)."""
    S = []
    # level 0: big billows stacked in 3 columns under a top-heavy envelope
    cols = [(-0.52, 0.55), (-0.12, 0.93), (0.25, 1.0), (0.55, 0.7)]
    for (u, ht) in cols:
        T = height * ht * rng.uniform(0.92, 1.0)
        yc = base_y - width * 0.12
        R0 = width * (0.26 - 0.08 * abs(u))
        while True:
            vv = (base_y - yc) / height
            R = R0 * (1 - 0.25 * vv) * rng.uniform(0.9, 1.08)
            xc = cx + u * width * 0.5 * (1 + 0.15 * vv) + lean * vv * width + rng.uniform(-0.03, 0.03) * width
            zc = -abs(u) * width * 0.25 + rng.uniform(-0.05, 0.05) * R
            S.append([xc, yc, zc, R, 0, -1])
            if yc - R <= base_y - T:
                break
            yc -= R * rng.uniform(0.62, 0.8)
    n0 = len(S)
    # level 1/2: lobes grown on the upper & outer silhouette only (cauliflower crown), never face-on dots
    def grow(parents, lev, count, ratio, upness):
        out = []
        for pi in parents:
            px, py, pz, pr = S[pi][0], S[pi][1], S[pi][2], S[pi][3]
            vv = (base_y - py) / height
            n = int(round(count * (0.5 + vv)))
            for _ in range(n):
                a = rng.uniform(-math.pi * 0.95, -math.pi * 0.05) if rng.random() < upness else \
                    rng.choice([rng.uniform(-math.pi, -math.pi * 0.6), rng.uniform(-math.pi * 0.4, 0.0)])
                r = pr * ratio * rng.uniform(0.75, 1.2)
                d = pr - r * rng.uniform(0.3, 0.55)
                x = px + math.cos(a) * d
                y = py + math.sin(a) * d
                if y > base_y - r * 0.6:
                    continue
                z = pz - r * 0.35
                S.append([x, y, z, r, lev, pi])
                out.append(len(S) - 1)
        return out
    l1 = grow(list(range(n0)), 1, 5, 0.42, 0.8)
    l2 = grow(l1, 2, 3, 0.45, 0.85)
    grow(l2, 3, 2, 0.45, 0.9)
    return S


def paint_tower(W, H, pw, ph, spec, pal, sun_p, ss=2, seed=0):
    """Paint one cumulus tower plate (pw x ph, straight-alpha RGBA).

    spec: dict(cx, base_y, width, height, lean, haze (0..1), haze_col, lit (0..1))
    sun_p: sun position in plate px. Lighting: warm cream crown with crisp lobes, one cool shadow mass
    below, thin gold rims on the sun-facing silhouette, torn base wisps."""
    rng = np.random.default_rng(seed)
    Hs, Ws = ph * ss, pw * ss
    cx, by, wd, ht = spec['cx'] * ss, spec['base_y'] * ss, spec['width'] * ss, spec['height'] * ss
    S = tower_domes(rng, cx, by, wd, ht, spec.get('lean', 0.0))
    Sa = np.array(S, np.float64)
    h, ids = _raster_domes(Hs, Ws, Sa)
    N = _tower_normals(ids, Sa, Hs, Ws, cx, 0.0, wd * 0.6, ht, by)
    a = (ids >= 0).astype(np.float32)
    ys = (np.arange(Hs, dtype=np.float32)[:, None] + 0.5)
    xs = (np.arange(Ws, dtype=np.float32)[None, :] + 0.5)
    v = np.clip((by - ys) / ht, 0, 1)                         # 0 base .. 1 top
    lev = np.where(ids >= 0, Sa[np.maximum(ids, 0), 4], 0).astype(np.float32)
    # light: toward the sun (to the side, up, and from behind)
    sx, sy = sun_p[0] * ss, sun_p[1] * ss
    side = 1.0 if sx > cx else -1.0
    L = np.array([0.62 * side, 0.62, -0.25])
    L = L / np.linalg.norm(L)
    ndl = N[..., 0] * L[0] + N[..., 1] * L[1] + N[..., 2] * L[2]
    bias = -0.75 + 0.95 * C_ss(0.3, 0.9, v)                  # crown lit, the lower mass in shadow
    le = ndl + bias
    lit = C_ss(-0.04, 0.1, le)
    cap = C_ss(0.12, 0.4, le)
    band = np.exp(-((le + 0.01) / 0.07) ** 2)
    P = {k: _rgb(c) for k, c in pal.items()}
    ny = N[..., 1:2]
    upf = np.clip(ny * 0.8 + 0.25, 0, 1)
    shd = P['shd_low'] + (P['shd_high'] - P['shd_low']) * upf
    shd = shd + (P['shd_sky'] - shd) * (np.clip(ny, 0, 1) * 0.5 * v[..., None])
    shd = shd + (P['bounce'] - shd) * (np.clip(-ny, 0, 1) * 0.5)
    shd = shd + (P['shd_deep'] - shd) * (C_ss(0.45, 0.0, v) * 0.5)[..., None]
    col = shd + (P['pink'] - shd) * (band * 0.55)[..., None]
    litc = P['lit_low'] + (P['lit'] - P['lit_low']) * cap[..., None]
    col = col + (litc - col) * (lit * spec.get('lit', 1.0))[..., None]
    # recess lines where a lobe sits over another (crisp dome overlaps inside the light)
    hb = cv2.GaussianBlur(np.where(ids >= 0, h, 0).astype(np.float32), (0, 0), 1.5 * ss)
    rec = np.clip((hb - h) / (0.004 * wd), 0, 1) * a
    col = col + (P['shd_high'] - col) * (rec * 0.35 * lit)[..., None]
    # silhouette rim toward the sun + translucent glow
    m8 = (a > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(m8, cv2.DIST_L2, 5).astype(np.float32)
    ab = cv2.GaussianBlur(a, (0, 0), 0.01 * wd)
    gx = -cv2.Sobel(ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = -cv2.Sobel(ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    dx, dy = sx - xs, sy - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    face = np.clip((gx * dx / dl + gy * dy / dl) / gl, 0, 1)
    face = cv2.GaussianBlur(face * (gl > 1e-4), (0, 0), 0.004 * wd)
    face = C_ss(0.05, 0.6, face)
    prox = 0.45 + 0.55 * np.exp(-dl / (0.6 * W * ss))
    wr = max(1.4 * ss * W / 1920, 1.0 * ss)
    rim = np.exp(-din / wr) * face * prox * spec.get('rim', 1.0)
    glow = np.exp(-din / (0.018 * wd)) * face * prox * 0.45 * spec.get('rim', 1.0)
    col = col + (P['glow'] - col) * np.clip(glow, 0, 0.8)[..., None]
    col = col + (P['rim'] - col) * np.clip(rim, 0, 1)[..., None]
    # haze
    hz = spec.get('haze', 0.0)
    if hz:
        hcol = np.asarray(spec['haze_col'], np.float32)
        hv = hz * (0.6 + 0.4 * (1 - v))
        col = col + (hcol - col) * hv[..., None]
    # torn base: horizontally combed alpha fade into the sea
    rng2 = np.random.default_rng(seed + 5)
    g = rng2.random((int(Hs / (0.012 * ht)) + 3, 12)).astype(np.float32)
    st = cv2.resize(g, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    fade = C_ss(0.0, 0.16, v + (st - 0.5) * 0.12)
    alpha = a * fade
    rgba = np.dstack([col, alpha]).astype(np.float32)
    pm = rgba.copy()
    pm[..., :3] *= pm[..., 3:4]
    pm = cv2.resize(pm, (pw, ph), interpolation=cv2.INTER_AREA)
    out = np.dstack([pm[..., :3] / np.maximum(pm[..., 3:4], 1e-5), pm[..., 3]])
    return out.astype(np.float32)


def C_ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _rgb(c):
    if isinstance(c, str):
        h = c.lstrip('#')
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    return np.asarray(c, np.float32)
