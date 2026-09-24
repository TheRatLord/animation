"""s05_sakura helpers: camera, static environment ray-caster + painted shading, per-frame water, railing,
fast shifted compositing."""
import math
import numpy as np
import cv2
from numba import njit, prange

from s05_sakura_raster import hash2, vnoise, fbm2, sstep

# ============================================================================ camera

EYE = 1.5          # eye height above the path
Y_PATH = -1.5
Y_WATER = -2.9
Y_FAR = -1.3
U_RAIL = -4.45     # railing line (near bank edge)
U_CURB0, U_CURB1 = -4.6, -4.3
U_PATH1 = 1.2      # right edge of the path
U_FARWALL = -30.0


class Cam:
    """Pinhole camera: x right, y up, z forward.  World 'path frame' (u, Y, s): u lateral (right), Y up
    (eye = 0), s along the path.  Path heads to the right of the view axis by `yaw`."""

    def __init__(self, W, H, yaw_deg=15.0):
        self.W, self.H = W, H
        self.f = 0.8 * W
        self.cx = W / 2
        self.hy = 0.54 * H
        a = math.radians(yaw_deg)
        self.ca, self.sa = math.cos(a), math.sin(a)

    def to_cam(self, u, Y, s):
        u = np.asarray(u, np.float64)
        s = np.asarray(s, np.float64)
        return u * self.ca + s * self.sa, np.asarray(Y, np.float64) + 0 * u, -u * self.sa + s * self.ca

    def proj(self, X, Y, Z, tx=0.0, mx=0.0, ss=1.0):
        Z = np.maximum(Z, 1e-3)
        return (self.cx + mx + self.f * (X - tx) / Z) * ss, (self.hy - self.f * Y / Z) * ss

    def dir_to_screen(self, d):
        """screen position of a direction (x, y, z camera coords)"""
        return self.cx + self.f * d[0] / d[2], self.hy - self.f * d[1] / d[2]

    def screen_to_dir(self, x, y):
        d = np.array([(x - self.cx) / self.f, (self.hy - y) / self.f, 1.0])
        return d / np.linalg.norm(d)


# ============================================================================ fast shifted compositing

@njit(cache=True, parallel=True, fastmath=True)
def over_shift(dst, src, ox, oy, shift):
    """dst (H,W,3) <- src RGBA (h,w,4) placed with its (0,0) at dst (ox + shift, oy), bilinear in x."""
    H, W = dst.shape[0], dst.shape[1]
    h, w = src.shape[0], src.shape[1]
    y0 = max(0, oy)
    y1 = min(H, oy + h)
    fx0 = ox + shift
    x0 = max(0, int(math.floor(fx0)))
    x1 = min(W, int(math.ceil(fx0 + w)))
    for y in prange(y0, y1):
        sy = y - oy
        for x in range(x0, x1):
            sx = x - fx0
            ix = int(math.floor(sx))
            fr = sx - ix
            if ix < -1 or ix >= w:
                continue
            a0 = src[sy, ix, 3] if ix >= 0 else 0.0
            a1 = src[sy, ix + 1, 3] if ix + 1 < w else 0.0
            a = a0 * (1 - fr) + a1 * fr
            if a <= 1e-4:
                continue
            for c in range(3):
                v0 = src[sy, ix, c] * a0 if ix >= 0 else 0.0
                v1 = src[sy, ix + 1, c] * a1 if ix + 1 < w else 0.0
                pv = v0 * (1 - fr) + v1 * fr          # premultiplied
                dst[y, x, c] = dst[y, x, c] * (1 - a) + pv


@njit(cache=True, parallel=True, fastmath=True)
def over_shift_acc(dst, acc, src, ox, oy, shift):
    """over_shift that also accumulates coverage into acc (H, W) (occluder mask for sun / light shafts)."""
    H, W = dst.shape[0], dst.shape[1]
    h, w = src.shape[0], src.shape[1]
    y0 = max(0, oy)
    y1 = min(H, oy + h)
    fx0 = ox + shift
    x0 = max(0, int(math.floor(fx0)))
    x1 = min(W, int(math.ceil(fx0 + w)))
    for y in prange(y0, y1):
        sy = y - oy
        for x in range(x0, x1):
            sx = x - fx0
            ix = int(math.floor(sx))
            fr = sx - ix
            if ix < -1 or ix >= w:
                continue
            a0 = src[sy, ix, 3] if ix >= 0 else 0.0
            a1 = src[sy, ix + 1, 3] if ix + 1 < w else 0.0
            a = a0 * (1 - fr) + a1 * fr
            if a <= 1e-4:
                continue
            acc[y, x] = acc[y, x] * (1 - a) + a
            for c in range(3):
                v0 = src[sy, ix, c] * a0 if ix >= 0 else 0.0
                v1 = src[sy, ix + 1, c] * a1 if ix + 1 < w else 0.0
                dst[y, x, c] = dst[y, x, c] * (1 - a) + v0 * (1 - fr) + v1 * fr


@njit(cache=True, parallel=True, fastmath=True)
def over_shift_rows(dst, src, ox, oy, shifts):
    """Like over_shift but with a per-row shift (shifts[h])."""
    H, W = dst.shape[0], dst.shape[1]
    h, w = src.shape[0], src.shape[1]
    y0 = max(0, oy)
    y1 = min(H, oy + h)
    for y in prange(y0, y1):
        sy = y - oy
        fx0 = ox + shifts[sy]
        x0 = max(0, int(math.floor(fx0)))
        x1 = min(W, int(math.ceil(fx0 + w)))
        for x in range(x0, x1):
            sx = x - fx0
            ix = int(math.floor(sx))
            fr = sx - ix
            if ix < -1 or ix >= w:
                continue
            a0 = src[sy, ix, 3] if ix >= 0 else 0.0
            a1 = src[sy, ix + 1, 3] if ix + 1 < w else 0.0
            a = a0 * (1 - fr) + a1 * fr
            if a <= 1e-4:
                continue
            for c in range(3):
                v0 = src[sy, ix, c] * a0 if ix >= 0 else 0.0
                v1 = src[sy, ix + 1, c] * a1 if ix + 1 < w else 0.0
                dst[y, x, c] = dst[y, x, c] * (1 - a) + v0 * (1 - fr) + v1 * fr


@njit(cache=True, parallel=True, fastmath=True)
def over_shift_map(dst, src, ox, oy, invz, k):
    """Per-pixel parallax: dst pixel x samples src at x - ox + k*invz(dst) (one fixed-point iteration)."""
    H, W = dst.shape[0], dst.shape[1]
    h, w = src.shape[0], src.shape[1]
    y0 = max(0, oy)
    y1 = min(H, oy + h)
    for y in prange(y0, y1):
        sy = y - oy
        for x in range(W):
            sx0 = x - ox
            if sx0 < 0 or sx0 >= w:
                continue
            s1 = sx0 + k * invz[sy, sx0]
            j = int(s1)
            if j < 0 or j >= w:
                continue
            sx = sx0 + k * invz[sy, j]
            ix = int(math.floor(sx))
            fr = sx - ix
            if ix < 0 or ix + 1 >= w:
                continue
            a0 = src[sy, ix, 3]
            a1 = src[sy, ix + 1, 3]
            a = a0 * (1 - fr) + a1 * fr
            if a <= 1e-4:
                continue
            for c in range(3):
                dst[y, x, c] = dst[y, x, c] * (1 - a) + src[sy, ix, c] * a0 * (1 - fr) + src[sy, ix + 1, c] * a1 * fr


# ============================================================================ static ray caster

# box columns: u0 u1 Y0 Y1 s0 s1 type r g b seed layer | bbox x0 x1 y0 y1 (ss px)
NB = 16


@njit(cache=True, fastmath=True)
def hedge_top(u, s, Y1):
    """Top surface height of the clipped hedge: overlapping rounded leaf-clump lobes."""
    cell = 0.62
    k = math.floor(s / cell)
    best = Y1 - 0.62
    for j in range(-1, 2):
        kk = k + j
        c = (kk + 0.5) * cell + (hash2(kk, 1, 301) - 0.5) * 0.3
        rr = 0.3 + 0.2 * hash2(kk, 2, 301)
        amp = 0.34 + 0.28 * hash2(kk, 3, 301)
        d = (s - c) / rr
        if d * d < 1.0:
            v = Y1 - 0.62 + amp * math.sqrt(1.0 - d * d)
            if v > best:
                best = v
    return best + 0.06 * (vnoise(s * 7.0, u * 7.0, 302) - 0.5)


@njit(cache=True, parallel=True, fastmath=True)
def raycast(Hs, Ws, ss, f, cxc, hy, ca, sa, B, mat, Zo, uo, Yo, so, face, bid):
    nbx = B.shape[0]
    for y in prange(Hs):
        for x in range(Ws):
            rx = ((x + 0.5) / ss - cxc) / f
            ry = (hy - (y + 0.5) / ss) / f
            ru = rx * ca - sa
            rs = rx * sa + ca
            best = 1e9
            m = 0
            fc = 0
            bi = -1
            if ry < 0:
                Z = Y_PATH / ry
                u = Z * ru
                if u > U_CURB0 and u < 60.0:
                    best = Z
                    m = 1
                Z = Y_WATER / ry
                u = Z * ru
                if u <= U_CURB0 and u >= U_FARWALL and Z < best:
                    best = Z
                    m = 2
                Z = Y_FAR / ry
                u = Z * ru
                if u < U_FARWALL and Z < best:
                    best = Z
                    m = 4
            if ru < 0:
                Z = U_FARWALL / ru
                Yh = Z * ry
                if Yh >= Y_WATER and Yh <= Y_FAR and Z < best:
                    best = Z
                    m = 3
            px = (x + 0.5)
            py = (y + 0.5)
            for b in range(nbx):
                if px < B[b, 12] or px > B[b, 13] or py < B[b, 14] or py > B[b, 15]:
                    continue
                tmin = 0.0
                tmax = 1e9
                ax = 0
                ok = True
                for k in range(3):
                    if k == 0:
                        d = ru
                        b0 = B[b, 0]
                        b1 = B[b, 1]
                    elif k == 1:
                        d = ry
                        b0 = B[b, 2]
                        b1 = B[b, 3]
                    else:
                        d = rs
                        b0 = B[b, 4]
                        b1 = B[b, 5]
                    if abs(d) < 1e-12:
                        if b0 > 0 or b1 < 0:
                            ok = False
                            break
                        continue
                    t0 = b0 / d
                    t1 = b1 / d
                    sgn = -1
                    if t0 > t1:
                        t0, t1 = t1, t0
                        sgn = 1
                    if t0 > tmin:
                        tmin = t0
                        ax = (k + 1) * sgn
                    if t1 < tmax:
                        tmax = t1
                    if tmax < tmin:
                        ok = False
                        break
                if ok and (B[b, 6] == 12.0 or B[b, 6] == 13.0):
                    # pitched roof: box clipped by the two roof planes (ridge along s for 12, along u for 13)
                    k_ = (B[b, 3] - B[b, 2])
                    if B[b, 6] == 12.0:
                        hw_ = 0.5 * (B[b, 1] - B[b, 0])
                        c_ = 0.5 * (B[b, 0] + B[b, 1])
                        dh = ru
                    else:
                        hw_ = 0.5 * (B[b, 5] - B[b, 4])
                        c_ = 0.5 * (B[b, 4] + B[b, 5])
                        dh = rs
                    k_ = k_ / hw_
                    for sg_ in range(2):
                        sgn2 = 1.0 if sg_ == 0 else -1.0
                        # half-space: Y + sgn2 * k * h <= Y1 + sgn2 * k * c
                        den = ry + sgn2 * k_ * dh
                        dd = B[b, 3] + sgn2 * k_ * c_
                        if abs(den) < 1e-12:
                            if dd < 0:
                                ok = False
                            continue
                        tp_ = dd / den
                        if den < 0:
                            if tp_ > tmin:
                                tmin = tp_
                                ax = 4 if sg_ == 0 else -4
                        else:
                            if tp_ < tmax:
                                tmax = tp_
                    if tmax < tmin:
                        ok = False
                if ok and tmin > 0 and tmin < best:
                    if B[b, 6] == 4.0:
                        # hedge: lumpy height field inside its box (scalloped clump silhouette)
                        hit = -1.0
                        nst = 16
                        tp = tmin
                        for q in range(nst + 1):
                            tq = tmin + (tmax - tmin) * q / nst
                            if tq * ry <= hedge_top(tq * ru, tq * rs, B[b, 3]):
                                if q == 0:
                                    hit = tq
                                else:
                                    lo = tp
                                    hi = tq
                                    for it in range(5):
                                        md = 0.5 * (lo + hi)
                                        if md * ry <= hedge_top(md * ru, md * rs, B[b, 3]):
                                            hi = md
                                        else:
                                            lo = md
                                    hit = hi
                                break
                            tp = tq
                        if hit > 0 and hit < best:
                            best = hit
                            m = 5
                            fc = ax if hit == tmin else 2
                            bi = b
                        continue
                    best = tmin
                    m = 5
                    fc = ax
                    bi = b
            mat[y, x] = m
            if m > 0:
                Zo[y, x] = best
                uo[y, x] = best * ru
                Yo[y, x] = best * ry
                so[y, x] = best * rs
            else:
                Zo[y, x] = 1e5
            face[y, x] = fc
            bid[y, x] = bi


# ============================================================================ static shading

@njit(cache=True, fastmath=True)
def _petal_cells(u, s, fp, cell, dens, seed):
    """Coverage (0..1) of scattered fallen petals around world point (u, s); fp = pixel footprint (m)."""
    if fp > cell * 0.8:
        return dens * 0.55
    iu = math.floor(u / cell)
    is_ = math.floor(s / cell)
    cov = 0.0
    for du in range(-1, 2):
        for ds in range(-1, 2):
            cu = iu + du
            cs = is_ + ds
            if hash2(cu, cs, seed) > dens:
                continue
            pu = (cu + hash2(cu, cs, seed + 1)) * cell
            ps = (cs + hash2(cu, cs, seed + 2)) * cell
            ang = hash2(cu, cs, seed + 3) * 6.283
            r = cell * (0.28 + 0.18 * hash2(cu, cs, seed + 4))
            dx = u - pu
            dy = s - ps
            a = dx * math.cos(ang) + dy * math.sin(ang)
            b = -dx * math.sin(ang) + dy * math.cos(ang)
            q = math.sqrt((a / r) ** 2 + (b / (0.62 * r)) ** 2)
            c = 1.0 - sstep(1.0 - fp / r, 1.0 + fp / r, q)
            if c > cov:
                cov = c
    return cov


@njit(cache=True, fastmath=True)
def _ground_sun(u, s, fp, Lu, Ls):
    """Sunlit amount of the near-bank ground (railing shadows + dappled canopy shade)."""
    # --- railing shadows (posts every 2 m, rails at 1.0 / 0.55 m)
    kk = 2.1
    du_h = 0.66 * kk
    ds_h = -0.75 * kk
    lit = 1.0
    if u > U_RAIL and u < U_RAIL + du_h * 1.1 + 0.3:
        for hr in (1.0, 0.55):
            uc = U_RAIL + du_h * hr
            w = 0.028 + 0.012 * hr
            pen = 0.01 + 0.02 * hr + fp
            d = abs(u - uc)
            lit = min(lit, sstep(w - pen, w + pen, d))
        # posts
        tpar = (u - U_RAIL) / du_h
        if tpar > -0.05 and tpar < 1.08:
            sp = s - ds_h * tpar           # post s this pixel's shadow would come from
            k = math.floor((sp - 0.4) / 2.0 + 0.5)
            ps = 0.4 + 2.0 * k
            d = abs(sp - ps) * 0.75
            w = 0.03
            pen = 0.008 + 0.02 * tpar + fp
            lit = min(lit, 1.0 - (1.0 - sstep(w - pen, w + pen, d)) * (1.0 - sstep(1.0, 1.07, tpar)))
    # --- canopy shade (the blossom tunnel over the right of the path)
    edge = -0.4 - 2.6 * math.exp(-max(s, 0.0) / 5.0) + 1.3 * (fbm2(s * 0.09, 3.1, 11, 3) - 0.5) \
        + 0.6 * (fbm2(s * 0.45, u * 0.2, 12, 2) - 0.5) + 0.45 * (fbm2(s * 1.7, u * 1.7, 13, 2) - 0.5)
    sh = sstep(edge - 0.22 - fp, edge + 0.22 + fp, u)
    # isolated shadow blobs of overhanging clumps on the lit side of the edge
    ob = fbm2(u * 0.8 + 3.0, s * 0.5, 14, 3) + 0.3 * (fbm2(u * 3.0, s * 2.0, 15, 2) - 0.5)
    sh = max(sh, sstep(0.64 - fp, 0.66 + fp, ob) * sstep(edge - 2.2, edge - 0.6, u))
    spot = 0.0
    if sh > 0.0:
        # dapples = pinhole images of the sun through canopy gaps: soft ellipses stretched along the
        # sun's azimuth, brightest in the centre, gathered in clusters; a few larger merged patches.
        su = 0.66
        sv = -0.75
        a_ = u * su + s * sv            # along the light direction
        b_ = -u * sv + s * su           # across it
        cl = fbm2(u * 0.35 + 2.0, s * 0.22, 20, 3)
        dens = sstep(0.38, 0.62, cl)
        # the shade edge breaks up into many small coins of light
        dens = max(dens, 0.85 * (1.0 - sstep(0.1, 1.1, abs(u - edge))))
        fleck = 0.0
        cell = 0.34
        ga = a_ / (cell * 2.2)
        gb = b_ / cell
        ia = math.floor(ga)
        ib = math.floor(gb)
        for da in range(-1, 2):
            for db in range(-1, 2):
                ca_ = ia + da
                cb_ = ib + db
                hh = hash2(ca_, cb_, 25)
                if hh > 0.15 + 0.7 * dens:
                    continue
                pa = ca_ + 0.2 + 0.6 * hash2(ca_, cb_, 26)
                pb = cb_ + 0.2 + 0.6 * hash2(ca_, cb_, 27)
                h8 = hash2(ca_, cb_, 28)
                rr = 0.07 + 0.5 * h8 * h8          # many small pin-points, a few big flecks
                qa = (ga - pa) * 2.2 / (rr * (1.6 + 1.4 * hash2(ca_, cb_, 29)))
                qb = (gb - pb) / rr
                q = math.sqrt(qa * qa + qb * qb)
                # irregular (lobed) outline: sun images overlapping through leaf gaps
                q *= 1.0 + 0.16 * math.sin(3.0 * math.atan2(qb, qa) + 6.283 * hash2(ca_, cb_, 30))
                pix = fp / (cell * rr) + 0.03
                c = (1.0 - sstep(1.0 - pix - 0.06, 1.0 + pix, q)) * (0.85 + 0.25 * (1.0 - q * q))
                if c > fleck:
                    fleck = c
        # larger irregular gaps
        n = fbm2(a_ * 0.9, b_ * 1.6, 21, 3) + (cl - 0.5) * 0.3
        soft = 0.03 + min(fp * 3.0, 0.12)
        big = sstep(0.66 - soft, 0.66 + soft, n)
        fleck = max(fleck, big * 0.95)
        # far away the flecks merge into a soft mottled shade
        far = sstep(0.025, 0.09, fp)
        fleck = fleck * (1.0 - far) + far * (0.3 * dens + 0.5 * big)
        spot = sh * fleck
        lit = min(lit, 1.0 - sh * (1.0 - fleck)) if fleck < 1.0 else lit * (1.0 + sh * (fleck - 1.0))
    return lit, spot


@njit(cache=True, fastmath=True)
def _cells(x, y, seed):
    """Worley F1/F2 plus the offset from the nearest feature point (cell units)."""
    ix = math.floor(x)
    iy = math.floor(y)
    d1 = 9.0
    d2 = 9.0
    ox = 0.0
    oy = 0.0
    hid = 0.0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            cx = ix + dx
            cy = iy + dy
            px = cx + 0.1 + 0.8 * hash2(cx, cy, seed)
            py = cy + 0.1 + 0.8 * hash2(cx, cy, seed + 1)
            d = math.sqrt((x - px) ** 2 + (y - py) ** 2)
            if d < d1:
                d2 = d1
                d1 = d
                ox = x - px
                oy = y - py
                hid = hash2(cx, cy, seed + 2)
            elif d < d2:
                d2 = d
    return d1, d2, ox, oy, hid


@njit(cache=True, fastmath=True)
def _pal4(v, c0, c1, c2, c3):
    v = min(max(v, 0.0), 0.999) * 3.0
    i = int(v)
    f = v - i
    if i == 0:
        a, b = c0, c1
    elif i == 1:
        a, b = c1, c2
    else:
        a, b = c2, c3
    return a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f


@njit(cache=True, fastmath=True)
def _hedge_shade(u, Yh, s, fc, Y1, Y0, fp, Lu, LY, Ls):
    """Clipped azalea/box hedge: individually lit leaf clumps, shadowed underside, blossom litter."""
    vy = Yh - Y0
    if fc == 2:
        x = s / 0.3
        y = (u - 6.0) / 0.26 + 17.0
    else:
        x = s / 0.3
        y = Yh / 0.24
    # two scales of clumps
    d1, d2, ox, oy, hid = _cells(x, y, 311)
    e1, e2, px_, py_, hid2 = _cells(x * 2.3 + 5.0, y * 2.3, 313)
    k = 1.0 - sstep(0.1, 0.35, fp)          # fine detail fades with distance
    ny = oy / 0.55
    nx = ox / 0.55
    if fc == 2:
        ny = 0.6 + 0.4 * ny
    lam = 0.35 + 0.55 * ny * 0.8 - 0.25 * nx + 0.2 * (py_ / 0.55) * k
    crev = (1.0 - sstep(0.02, 0.16, d2 - d1)) * k
    tl = sstep(-0.4, -0.02, Yh - hedge_top(u, s, Y1))
    under = 1.0 - sstep(0.05, 0.55, vy)
    v = 0.38 + 0.42 * lam + 0.28 * tl - 0.35 * under - 0.3 * crev + 0.12 * (hid - 0.5)
    # dappled canopy shade from the cherry row
    mo = sstep(0.46, 0.56, fbm2(s * 0.45, Yh * 0.6 + u * 0.3, 63, 3) + 0.15 * (fbm2(s * 2.2, Yh * 2.2, 64, 2) - 0.5))
    v = v * (0.62 + 0.38 * mo) + 0.08 * mo * tl
    c0 = (0.03, 0.1, 0.13)
    c1 = (0.08, 0.25, 0.24)
    c2 = (0.25, 0.52, 0.25)
    c3 = (0.7, 0.86, 0.34)
    r, g, b = _pal4(v, c0, c1, c2, c3)
    # tiny leaf glints on lit clump tops
    if fp < 0.02:
        lf = hash2(int(math.floor(x * 9.0)), int(math.floor(y * 9.0)), 317)
        gl = sstep(0.9, 0.97, lf) * sstep(0.55, 0.8, v) * (1 - fp / 0.02) * 0.35
        r += (0.95 - r) * gl
        g += (1.0 - g) * gl
        b += (0.7 - b) * gl
    # fallen blossom litter caught on the hedge (denser on the top)
    dens = 0.12 + 0.35 * tl
    pc = _petal_cells(s, Yh * 1.3 + u, fp, 0.045, dens, 319)
    pr = 0.8 + 0.2 * mo
    r += (pr - r) * pc
    g += (pr * 0.8 - g) * pc
    b += (pr * 0.87 - b) * pc
    return r, g, b


@njit(cache=True, fastmath=True)
def _facade(typ, br, bg, bb, lit, hc, vy, top, hB, sd, fp, ax, shade_r, shade_g, shade_b):
    """Painted apartment-block / house facade: balcony bands with rails, varied windows (sky reflections,
    dark interiors, a few warm lit or glinting panes), AC units, vertical value gradient, lit parapet."""
    isd = int(sd)
    h1 = hash2(isd, 1, 11)
    h2 = hash2(isd, 2, 11)
    h3 = hash2(isd, 3, 11)
    # soft vertical gradient: a touch darker/cooler at street level, lighter toward the sky
    gv = 0.84 + 0.2 * sstep(0.0, max(hB, 1.0), vy)
    r = br * (shade_r + (1.1 - shade_r) * lit) * gv
    g = bg * (shade_g + (0.97 - shade_g) * lit) * gv
    b = bb * (shade_b + (0.82 - shade_b) * lit) * (0.97 + 0.03 * gv)
    # warm bounce from the sunlit ground on the lower floors of shaded facades
    wb_ = (1.0 - sstep(0.0, 7.0, vy)) * (1.0 - lit) * 0.16
    r += wb_ * 0.9
    g += wb_ * 0.5
    b -= wb_ * 0.2
    if typ == 0:
        fh = 2.8 + 0.5 * h1
        wsp = 1.7 + 1.9 * h2
        wwf = 0.34 + 0.34 * h3
        apt = h2 > 0.08
    else:
        fh = 2.7
        wsp = 2.6 + 1.2 * h2
        wwf = 0.3 + 0.2 * h3
        apt = h1 > 0.5
    fl = vy / fh
    fi = math.floor(fl)
    ff = fl - fi
    hcc = (hc + sd * 3.1) / wsp
    hi_ = math.floor(hcc)
    hf = hcc - hi_
    pixv = fp / fh
    pixh = fp / wsp
    fade = 1.0 - sstep(0.16, 0.42, pixv)
    # round 6: weathering - faint vertical rain streaks / stains and a darker plinth (no clean CG planes)
    if fp < 0.08:
        stn = fbm2(hc * 2.2 + sd, vy * 0.22, 63, 3)
        stv = sstep(0.55, 0.8, stn) * 0.07 * (1.0 - fp / 0.08)
        pl = (1.0 - sstep(0.35, 0.45 + fp, vy)) * 0.12
        r *= 1.0 - stv - pl
        g *= 1.0 - stv - pl
        b *= 1.0 - stv * 0.6 - pl * 0.6
    if typ == 5 or (typ == 0 and hB < 8.5 and h1 > 0.35):
        # horizontal lap siding lines on the houses
        if fp < 0.05:
            sl = vy / 0.22 - math.floor(vy / 0.22)
            sline = (1.0 - sstep(0.0, 0.12 + fp / 0.22, sl)) * (1.0 - fp / 0.05) * 0.12
            r *= 1.0 - sline
            g *= 1.0 - sline
            b *= 1.0 - sline * 0.7
    if vy > 0.9 and top > 0.8 and fade > 0.0:
        wy0 = 0.34 if apt else 0.3
        wy1 = 0.86 if apt else 0.78
        wx0 = 0.5 - wwf * 0.5
        wx1 = 0.5 + wwf * 0.5
        # per-cell window size variety (sliding doors on balconies, small toilet windows ...)
        hv = hash2(int(hi_), int(fi), isd)
        hw = hash2(int(hi_), int(fi), isd + 7)
        if hw < 0.18:
            wx0 = 0.5 - wwf * 0.28
            wx1 = 0.5 + wwf * 0.28
            wy0 = wy0 + 0.18
        elif hw > 0.8 and apt:
            wx0 = 0.12
            wx1 = 0.88
            wy0 = 0.3
        win = sstep(wx0 - pixh, wx0 + pixh, hf) * (1 - sstep(wx1 - pixh, wx1 + pixh, hf)) \
            * sstep(wy0 - pixv, wy0 + pixv, ff) * (1 - sstep(wy1 - pixv, wy1 + pixv, ff))
        if typ == 5 and hash2(int(hi_), int(fi), isd + 5) > 0.6:
            win = 0.0
        # frame: thin lighter mullion around the pane
        fyy = (ff - wy0) / (wy1 - wy0)
        if hv < 0.4:
            # pane reflecting the sky: pale cyan-blue sky at the top with a bright diagonal streak, the
            # reflected blue deepening toward the bottom
            k = 1.0 - fyy
            dg_ = sstep(0.55, 0.68, k + 0.35 * (hf - wx0) / max(wx1 - wx0, 1e-3)) *                 (1.0 - sstep(0.78, 0.9, k + 0.35 * (hf - wx0) / max(wx1 - wx0, 1e-3)))
            wr, wg, wb = 0.2 + 0.36 * k + 0.3 * dg_, 0.36 + 0.36 * k + 0.24 * dg_, 0.62 + 0.3 * k + 0.1 * dg_
            # reflected cumulus / neighbouring facade (round 8): a pale cloud band + a darker lower corner
            xw_ = (hf - wx0) / max(wx1 - wx0, 1e-3)
            cl_ = sstep(0.5, 0.62, 0.5 + 0.5 * math.sin(xw_ * 5.0 + hi_ * 1.7 + fi * 2.3) - 0.9 * (1.0 - fyy) + 0.35)
            wr += (0.88 - wr) * cl_ * 0.7
            wg += (0.9 - wg) * cl_ * 0.7
            wb += (0.96 - wb) * cl_ * 0.7
            dk_ = sstep(0.3, 0.0, fyy) * sstep(0.5, 0.0, xw_) * 0.5
            wr *= 1.0 - dk_
            wg *= 1.0 - dk_
            wb *= 1.0 - dk_ * 0.6
        elif hv < 0.9:
            # interior (round 8): varied curtains - one side drawn, both sides, full lace, or blinds; varied
            # curtain colours; dark room behind
            hc9 = hash2(int(hi_), int(fi), isd + 9)
            hcc_ = hash2(int(hi_), int(fi), isd + 13)
            xw = (hf - wx0) / max(wx1 - wx0, 1e-3)
            cr_, cg_, cb_ = 0.86, 0.84, 0.78
            if hcc_ < 0.25:
                cr_, cg_, cb_ = 0.9, 0.78, 0.8
            elif hcc_ < 0.45:
                cr_, cg_, cb_ = 0.74, 0.84, 0.78
            elif hcc_ < 0.6:
                cr_, cg_, cb_ = 0.72, 0.78, 0.9
            if hc9 < 0.3:
                cur = sstep(0.55, 0.6, xw)
            elif hc9 < 0.55:
                cur = max(1.0 - sstep(0.22, 0.27, xw), sstep(0.73, 0.78, xw))
            elif hc9 < 0.8:
                cur = 0.55 + 0.1 * math.sin(xw * 40.0)
                cr_, cg_, cb_ = 0.8, 0.83, 0.88
            else:
                bl_ = fyy * 9.0 - math.floor(fyy * 9.0)
                cur = 0.6 + 0.3 * sstep(0.0, 0.3, bl_) * (1.0 - sstep(0.02, 0.06, fp))
                cr_, cg_, cb_ = 0.82, 0.82, 0.8
            fold = 1.0 - 0.12 * (0.5 + 0.5 * math.sin(xw * 28.0 + hc9 * 6.0)) * (1.0 - sstep(0.02, 0.05, fp))
            wr = 0.12 + (cr_ * fold - 0.12) * cur
            wg = 0.16 + (cg_ * fold - 0.16) * cur
            wb = 0.3 + (cb_ * fold - 0.3) * cur
        elif hv < 0.95:
            # glinting pane catching the sun
            wr, wg, wb = 1.0, 0.95, 0.86
        else:
            # warm interior light
            wr, wg, wb = 0.9, 0.66, 0.42
        # window frame: a light aluminium surround just outside the pane
        fw_ = 0.05 / wsp
        fwv = 0.05 / fh
        wout = sstep(wx0 - fw_ - pixh, wx0 - fw_ + pixh, hf) * (1 - sstep(wx1 + fw_ - pixh, wx1 + fw_ + pixh, hf)) \
            * sstep(wy0 - fwv - pixv, wy0 - fwv + pixv, ff) * (1 - sstep(wy1 + fwv - pixv, wy1 + fwv + pixv, ff))
        frm = max(wout - win, 0.0) * fade * (1.0 - sstep(0.02, 0.06, fp))
        if typ == 5 and hash2(int(hi_), int(fi), isd + 5) > 0.6:
            frm = 0.0
        fv_ = 0.8 + 0.2 * lit
        r += (fv_ * 0.96 - r) * frm * 0.8
        g += (fv_ * 0.97 - g) * frm * 0.8
        b += (fv_ * 1.0 - b) * frm * 0.8
        # central mullion + a transom on the taller windows (round 8: sashes read clearly)
        mid = (wx0 + wx1) * 0.5
        mul = 1.0 - sstep(0.0, 0.016 + pixh, abs(hf - mid))
        if wy1 - wy0 > 0.45:
            mul = max(mul, 1.0 - sstep(0.0, 0.012 + pixv, abs(fyy - 0.28) * (wy1 - wy0)))
        wa = win * fade * 0.92
        r += (wr - r) * wa
        g += (wg - g) * wa
        b += (wb - b) * wa
        mm = mul * win * fade * 0.8
        r += (0.85 - r) * mm
        g += (0.86 - g) * mm
        b += (0.9 - b) * mm
        # rain stains running down from the window-sill corners (round 8 weathering)
        if fp < 0.05 and ff < wy0 and ff > wy0 - 0.45:
            dc = min(abs(hf - wx0), abs(hf - wx1))
            stn_ = (1.0 - sstep(0.0, 0.03 + pixh, dc)) * (1.0 - (wy0 - ff) / 0.45) * (1.0 - fp / 0.05)
            stn_ *= 0.5 + 0.5 * (hash2(int(hi_), int(fi), isd + 41) > 0.4)
            r *= 1.0 - 0.12 * stn_
            g *= 1.0 - 0.12 * stn_
            b *= 1.0 - 0.08 * stn_
        if apt and fade > 0.2 and hash2(int(hi_), int(fi), isd + 51) > 0.5 and ff > 0.3 and ff < 0.84:
            # laundry hung from the balcony pole: shirts / towels of varied colour and length
            pj = (hf - 0.08) / 0.105
            ji = math.floor(pj)
            jf = pj - ji
            if ji >= 0 and ji < 8:
                hp = hash2(int(hi_) * 16 + int(ji), int(fi), isd + 57)
                if hp > 0.35:
                    ln_ = 0.2 + 0.3 * hash2(int(hi_) * 16 + int(ji), int(fi), isd + 59)
                    wdt = 0.62 + 0.3 * hp
                    inx = sstep(0.5 - wdt * 0.5 - pixh / 0.105, 0.5 - wdt * 0.5 + pixh / 0.105, jf) *                         (1.0 - sstep(0.5 + wdt * 0.5 - pixh / 0.105, 0.5 + wdt * 0.5 + pixh / 0.105, jf))
                    iny = sstep(0.82 - ln_ - pixv, 0.82 - ln_ + pixv, ff) * (1.0 - sstep(0.82 - pixv, 0.82 + pixv, ff))
                    lc = hash2(int(hi_) * 16 + int(ji), int(fi), isd + 61)
                    lr, lg, lb = 0.95, 0.95, 0.93
                    if lc < 0.2:
                        lr, lg, lb = 0.55, 0.7, 0.92
                    elif lc < 0.35:
                        lr, lg, lb = 0.98, 0.74, 0.78
                    elif lc < 0.48:
                        lr, lg, lb = 0.98, 0.9, 0.6
                    elif lc < 0.58:
                        lr, lg, lb = 0.35, 0.4, 0.55
                    sh_ = (0.62 + 0.38 * lit) * (0.9 + 0.1 * math.cos(jf * 12.0))
                    la = inx * iny * fade
                    r += (lr * sh_ - r) * la
                    g += (lg * sh_ - g) * la
                    b += (lb * sh_ * (1.0 + 0.1 * (1.0 - lit)) - b) * la
        if apt:
            # balcony: slab underside shadow, parapet panel with handrail + balusters
            shadow = sstep(0.88, 0.9 + pixv, ff) * fade
            r *= 1.0 - 0.35 * shadow
            g *= 1.0 - 0.32 * shadow
            b *= 1.0 - 0.2 * shadow
            if ff < 0.3:
                par = sstep(0.0, 0.02 + pixv, ff) * (1 - sstep(0.28, 0.3 + pixv, ff))
                # balusters (fine vertical bars) or a solid frosted panel
                if h3 > 0.45:
                    bx = hc / 0.14 - math.floor(hc / 0.14)
                    bar = (1.0 - sstep(0.18, 0.3 + fp / 0.14, bx)) * (1.0 - sstep(0.03, 0.08, fp))
                    pv = 0.55 + 0.45 * bar
                else:
                    pv = 0.9
                pr_, pg_, pb_ = 0.93 * gv, 0.93 * gv, 0.96
                if lit < 0.5:
                    pr_, pg_, pb_ = 0.62, 0.66, 0.84
                a_ = par * fade * pv
                r += (pr_ - r) * a_
                g += (pg_ - g) * a_
                b += (pb_ - b) * a_
                # handrail catches the light
                hr_ = sstep(0.24, 0.265 + pixv, ff) * (1 - sstep(0.28, 0.3 + pixv, ff)) * fade
                r += (1.05 - r) * hr_ * 0.8
                g += (1.0 - g) * hr_ * 0.8
                b += (0.95 - b) * hr_ * 0.8
                # AC outdoor unit on some balconies
                if hash2(int(hi_), int(fi), isd + 21) > 0.55:
                    ac = sstep(0.62, 0.63 + pixh, hf) * (1 - sstep(0.86, 0.87 + pixh, hf)) * \
                        sstep(0.03, 0.04 + pixv, ff) * (1 - sstep(0.2, 0.21 + pixv, ff)) * fade
                    gr = 0.0
                    if fp < 0.04:
                        cx_ = (hf - 0.68) / 0.08
                        cy_ = (ff - 0.115) / 0.06
                        gr = (1.0 - sstep(0.7, 0.9, math.sqrt(cx_ * cx_ + cy_ * cy_))) * 0.4
                    av = 0.92 * (0.75 + 0.25 * lit) * (1 - gr)
                    r += (av - r) * ac
                    g += (av - g) * ac
                    b += (av * 1.03 - b) * ac
        else:
            # plain floor-slab line
            slab = sstep(0.0, 0.02 + pixv, ff) * (1 - sstep(0.06, 0.08 + pixv, ff)) * fade * 0.35
            r += (0.97 - r) * slab
            g += (0.96 - g) * slab
            b += (0.97 - b) * slab
    if typ == 0 and fp < 0.08:
        # rain downpipes on some window-column boundaries + a gutter / coping shadow line under the roof
        fd = 1.0 - sstep(0.04, 0.08, fp)
        if hash2(int(hi_), 5, isd + 31) > 0.72:
            pw_ = 0.06 / wsp
            dd = abs(hf - 0.03)
            pipe = (1.0 - sstep(pw_, pw_ + pixh, dd)) * fd
            hl_ = (1.0 - sstep(0.0, pw_ * 0.5 + pixh, abs(hf - 0.03 + pw_ * 0.4))) * fd
            pv_ = 0.5 + 0.25 * lit
            r += (pv_ * 0.95 - r) * pipe * 0.85 + (1.0 - r) * hl_ * 0.3 * lit
            g += (pv_ * 0.97 - g) * pipe * 0.85 + (1.0 - g) * hl_ * 0.3 * lit
            b += (pv_ * 1.08 - b) * pipe * 0.85 + (1.0 - b) * hl_ * 0.3 * lit
        gut = sstep(0.3, 0.32 + fp, top) * (1.0 - sstep(0.42, 0.44 + fp, top)) * fd
        r *= 1.0 - 0.3 * gut
        g *= 1.0 - 0.3 * gut
        b *= 1.0 - 0.2 * gut
    # cast shadow under the eaves / parapet coping (round 5)
    eav = (1.0 - sstep(0.3, 0.75 + fp, top)) * sstep(0.12, 0.3, top) * (0.55 if typ == 0 else 0.4)
    r *= 1.0 - eav * 0.42
    g *= 1.0 - eav * 0.4
    b *= 1.0 - eav * 0.22
    rim = (1 - sstep(0.0, 0.25 + fp * 1.5, top))
    r += (1.05 - r) * rim * 0.8
    g += (0.97 - g) * rim * 0.8
    b += (0.9 - b) * rim * 0.8
    return r, g, b


@njit(cache=True, parallel=True, fastmath=True)
def shade_env(mat, Zo, uo, Yo, so, face, bid, B, f, ss, Lu, LY, Ls, haze, haze_k, haze_max, out, layer, sunx,
              spot_out):
    Hs, Ws = mat.shape
    for y in prange(Hs):
        for x in range(Ws):
            m = mat[y, x]
            if m == 0:
                for c in range(4):
                    out[y, x, c] = 0.0
                layer[y, x] = 0
                continue
            Z = Zo[y, x]
            u = uo[y, x]
            Yh = Yo[y, x]
            s = so[y, x]
            fp = Z / (f * ss)             # pixel footprint in metres
            r = 0.5
            g = 0.5
            b = 0.5
            lay = 0
            if m == 1:
                lay = 1
                lit, spot = _ground_sun(u, s, fp, Lu, Ls)
                spot_out[y, x] = spot
                if u < U_CURB1:
                    # concrete curb under the railing
                    tex = 0.9 + 0.1 * fbm2(u * 9, s * 3, 5, 2)
                    tex -= 0.12 * (1.0 - sstep(0.0, 0.02 + fp, abs(s / 1.8 - math.floor(s / 1.8 + 0.5)) * 1.8)) *                         (1.0 - sstep(0.03, 0.08, fp))
                    sr, sg, sb = 0.86 * tex, 0.82 * tex, 0.76 * tex
                    hr, hg, hb = 0.55, 0.56, 0.72
                elif u < U_PATH1:
                    # paved path: warm ochre concrete slabs, expansion joints, painted cracks, gravel grain
                    gr = fbm2(u * 2.2, s * 2.2, 3, 3)
                    fine = 0.0
                    if fp < 0.01:
                        fine = (hash2(int(math.floor(u * 160)), int(math.floor(s * 160)), 9) - 0.5) * (1 - fp / 0.01)
                    blot = fbm2(u * 0.7, s * 0.25, 4, 3)
                    # slab joints: across the path every 3.2 m + one longitudinal joint
                    sj = s / 3.2
                    jd = abs(sj - math.floor(sj + 0.5)) * 3.2           # metres to the nearest cross joint
                    jw = 0.012 + fp * 0.9
                    joint = (1.0 - sstep(jw * 0.4, jw, jd)) * (1.0 - sstep(0.03, 0.09, fp))
                    jlit = (1.0 - sstep(jw, jw * 2.2, abs(jd - jw * 1.6))) * (sj - math.floor(sj + 0.5) > 0) \
                        * (1.0 - sstep(0.02, 0.06, fp))
                    ljd = abs(u + 1.62)
                    joint = max(joint, (1.0 - sstep(jw * 0.4, jw, ljd)) * (1.0 - sstep(0.03, 0.09, fp)) * 0.8)
                    # slab-to-slab tone variation
                    sid = math.floor(sj + 0.5) * 2 + (1 if u > -1.62 else 0)
                    slab_v = (hash2(int(sid), 5, 407) - 0.5) * 0.06
                    # painted cracks: sparse segments of a cell network, thin and dark with a lit lip
                    crack = 0.0
                    if fp < 0.02:
                        c1, c2, cox, coy, chd = _cells(u / 0.9 + 3.3, s / 1.3, 409)
                        keep = sstep(0.55, 0.65, fbm2(u * 0.9, s * 0.6, 410, 2))
                        wn = 0.02 + fp / 0.9 * 1.2
                        crack = (1.0 - sstep(wn * 0.3, wn, c2 - c1)) * keep * (1 - fp / 0.02)
                        ww = fbm2(u * 6.0, s * 6.0, 411, 2)
                        crack *= sstep(0.35, 0.5, ww)
                    peb = 0.0
                    if fp < 0.012:
                        e1, e2, pox, poy, phd = _cells(u / 0.04, s / 0.04, 401)
                        if phd > 0.45:
                            pr_ = 0.2 + 0.14 * phd
                            peb = (1.0 - sstep(pr_ - 0.08, pr_ + 0.05 + fp / 0.04, e1)) * (1 - fp / 0.012)
                            peb = peb * (0.7 if poy > 0 else -0.9) * 0.6
                    stain = sstep(0.55, 0.75, fbm2(u * 0.45 + 9.0, s * 0.3, 408, 3)) * 0.07
                    tex = 0.95 + 0.07 * (gr - 0.5) + 0.07 * fine + 0.07 * (blot - 0.5) + slab_v \
                        - 0.3 * crack - 0.4 * joint + 0.06 * peb - stain + 0.22 * jlit
                    # foot-worn centre slightly lighter
                    wear = math.exp(-((u + 1.5) / 1.4) ** 2) * 0.03
                    # warm cream concrete near the camera cooling to a pale lilac-grey toward the vanishing point
                    gd = sstep(4.0, 70.0, s)
                    sr = (1.0 - 0.05 * gd + wear) * tex
                    sg = (0.89 + 0.0 * gd + wear) * tex
                    sb = (0.75 + 0.08 * gd + wear) * tex
                    sv_ = 0.92 + 0.16 * fbm2(u * 0.35 + 5.0, s * 0.12, 24, 3)
                    # cobalt / teal-blue shade (sky-lit), lighter and cooler with distance
                    sv2 = 0.84 + 0.3 * fbm2(u * 0.22 + 1.0, s * 0.09, 26, 3)
                    hr = (0.4 + 0.1 * gd) * tex * sv_ * sv2
                    hg = (0.43 + 0.08 * gd) * tex * sv_ * sv2
                    hb = (0.66 + 0.06 * gd) * tex * (0.96 + 0.08 * sv_) * (0.9 + 0.1 * sv2)
                    # warm bounce light from the sunlit path / blossoms inside the shade
                    wbn = sstep(0.45, 0.75, fbm2(u * 0.3 + 4.0, s * 0.15, 27, 3)) * 0.55
                    hr += (0.66 - hr) * wbn * 0.45
                    hg += (0.52 - hg) * wbn * 0.45
                    hb += (0.6 - hb) * wbn * 0.45
                    # shallow puddles left by the morning watering: sky reflection + bright sheen
                    pcell = 3.6
                    pci = math.floor(u / pcell)
                    pcs = math.floor(s / pcell)
                    for dpu in range(-1, 2):
                        for dps in range(-1, 2):
                            ku = pci + dpu
                            ks = pcs + dps
                            if hash2(ku, ks, 420) > 0.22:
                                continue
                            pu_ = (ku + 0.2 + 0.6 * hash2(ku, ks, 421)) * pcell
                            ps_ = (ks + 0.2 + 0.6 * hash2(ku, ks, 422)) * pcell
                            prr = 0.18 + 0.3 * hash2(ku, ks, 423)
                            du_ = (u - pu_) / (prr * 0.8)
                            ds_ = (s - ps_) / (prr * 1.6)
                            qq = math.sqrt(du_ * du_ + ds_ * ds_) + 0.35 * (fbm2(u * 3.0, s * 3.0, 424, 2) - 0.5)
                            pm_ = 1.0 - sstep(0.85 - fp * 2.0 / prr, 0.95 + fp * 2.0 / prr, qq)
                            if pm_ > 0.0:
                                # reflect the pale sky / blossom; bright sheen toward the far edge
                                sk = 0.55 + 0.45 * sstep(-1.0, 1.0, ds_)
                                rr_ = 0.72 + 0.26 * sk
                                rg_ = 0.8 + 0.18 * sk
                                rb_ = 0.95 + 0.06 * sk
                                rim_ = sstep(0.6, 0.9, qq) * sstep(0.0, 1.0, ds_) * 0.5
                                sr += (rr_ * 0.85 + rim_ - sr) * pm_ * 0.55
                                sg += (rg_ * 0.85 + rim_ - sg) * pm_ * 0.55
                                sb += (rb_ * 0.9 + rim_ * 0.8 - sb) * pm_ * 0.55
                                hr += (0.5 - hr) * pm_ * 0.6
                                hg += (0.58 - hg) * pm_ * 0.6
                                hb += (0.82 - hb) * pm_ * 0.6
                    # darker curb gutter line
                    gut = math.exp(-((u - U_CURB1) / (0.04 + fp)) ** 2) * 0.35
                    sr *= 1 - gut
                    sg *= 1 - gut
                    sb *= 1 - gut * 0.7
                    hr *= 1 - gut
                    hg *= 1 - gut
                    hb *= 1 - gut * 0.7
                else:
                    # ---- raised granite edge-stone strip (shadowed riser + lit top face) with grass spilling
                    #      over it, then the grass bank painted as lit blade clumps
                    cw = 0.012 * (vnoise(s * 0.7, 0.3, 430) - 0.5)          # slight irregular settling
                    e0 = U_PATH1 + cw
                    e1 = e0 + 0.1                     # riser -> top arris
                    e2 = e1 + 0.2                     # top face -> grass
                    # grass overhang: spiky tuft profile along the edge
                    bc = s / 0.055
                    bi0 = math.floor(bc)
                    tuftp = 0.0
                    for db in range(-1, 2):
                        kb = bi0 + db
                        hb0 = hash2(kb, 1, 431)
                        cc_ = kb + 0.5 + 0.4 * (hash2(kb, 2, 431) - 0.5)
                        tri = max(0.0, 1.0 - abs(bc - cc_) / (0.8 + 0.6 * hb0))
                        tuftp = max(tuftp, hb0 * tri ** 0.7)
                    env_ = sstep(0.35, 0.7, fbm2(s * 1.6, 2.0, 432, 2))
                    gb = e2 - (0.05 + 0.16 * env_) * tuftp * (1.0 - sstep(0.012, 0.05, fp)) \
                        - 0.03 * env_ * sstep(0.012, 0.05, fp)
                    if u < e1:
                        # riser: vertical face seen edge-on-ish, cooler and darker, dark contact line
                        k = math.floor(s / 0.9)
                        jd_ = abs(s - (k + 0.5) * 0.9)
                        jt = 1.0 - sstep(0.43, 0.45, jd_) * 0.35
                        v = (0.92 + 0.08 * hash2(int(k), 3, 4)) * jt
                        fr_ = (u - e0) / 0.1
                        v *= 0.8 + 0.2 * fr_
                        sr, sg, sb = 0.62 * v, 0.58 * v, 0.56 * v
                        hr, hg, hb = 0.3 * v, 0.32 * v, 0.46 * v
                        cl_ = (1.0 - sstep(0.0, 0.25 + fp * 20.0, fr_)) * 0.45
                        sr *= 1 - cl_
                        sg *= 1 - cl_
                        sb *= 1 - cl_ * 0.7
                        hr *= 1 - cl_
                        hg *= 1 - cl_
                        hb *= 1 - cl_ * 0.7
                    elif u < gb:
                        # top face: bright, with a lit arris line and block joints
                        k = math.floor(s / 0.9)
                        jd_ = abs(s - (k + 0.5) * 0.9)
                        jt = 1.0 - sstep(0.43, 0.45, jd_) * 0.4
                        v = (0.94 + 0.06 * hash2(int(k), 3, 4)) * jt * (0.97 + 0.06 * (fbm2(u * 20, s * 6, 433, 2) - 0.5))
                        sr, sg, sb = 1.0 * v, 0.95 * v, 0.86 * v
                        hr, hg, hb = 0.5 * v, 0.52 * v, 0.7 * v
                        ar_ = (1.0 - sstep(0.0, 0.015 + fp, u - e1)) * 0.25
                        sr += ar_
                        sg += ar_ * 0.95
                        sb += ar_ * 0.85
                        # shadow of the overhanging grass on the stone
                        so_ = (1.0 - sstep(0.0, 0.05 + fp, gb - u)) * 0.45
                        sr *= 1 - so_
                        sg *= 1 - so_
                        sb *= 1 - so_ * 0.6
                    else:
                        # grass bank: clumps lit on their far/upper side, dark crevices, blade strokes
                        k_ = 1.0 - sstep(0.01, 0.06, fp)
                        d1, d2, ox_, oy_, hid = _cells(u / 0.22, s / 0.4, 441)
                        e1_, e2_, ox2, oy2, hid2 = _cells(u / 0.09 + 3.0, s / 0.16, 442)
                        lam_ = 0.5 + 0.5 * (oy_ / 0.6) * 0.9 - 0.2 * (ox_ / 0.6)
                        lam2 = 0.5 + 0.5 * (oy2 / 0.6)
                        crev = (1.0 - sstep(0.02, 0.14, d2 - d1)) * k_
                        crev2 = (1.0 - sstep(0.02, 0.12, e2_ - e1_)) * k_ * 0.6
                        strk = 0.0
                        if fp < 0.02:
                            strk = (vnoise(u * 90.0, s * 9.0, 443) - 0.5) * (1 - fp / 0.02)
                        v = 0.45 + 0.3 * (lam_ - 0.5) * k_ + 0.15 * (lam2 - 0.5) * k_ - 0.32 * crev - 0.15 * crev2 \
                            + 0.25 * strk + 0.1 * (hid - 0.5) + 0.1 * (fbm2(u * 0.8, s * 0.5, 7, 3) - 0.5)
                        v = min(max(v, 0.0), 1.0)
                        sr, sg, sb = _pal4(v, (0.1, 0.25, 0.17), (0.28, 0.5, 0.2), (0.55, 0.76, 0.28),
                                           (0.86, 0.95, 0.52))
                        hr, hg, hb = _pal4(v, (0.04, 0.14, 0.17), (0.1, 0.27, 0.28), (0.2, 0.42, 0.36),
                                           (0.36, 0.56, 0.44))
                        # the curb's short cast shadow on the grass
                        cs_ = (1.0 - sstep(0.0, 0.07 + fp, u - gb)) * 0.5
                        sr *= 1 - cs_
                        sg *= 1 - cs_
                        sb *= 1 - cs_ * 0.5
                        lit *= 0.35 + 0.65 * sstep(0.35, 0.75, fbm2(u * 0.6, s * 0.4, 31, 2))
                # warm saturated penumbra edge (painted look)
                lc = min(lit, 1.0)
                pen = 4.0 * lc * (1 - lc)
                hot = max(lit - 1.0, 0.0)
                r = hr + (sr - hr) * lc + pen * 0.16 + hot * 0.9
                g = hg + (sg - hg) * lc + pen * 0.05 + hot * 0.75
                b = hb + (sb - hb) * lc - pen * 0.07 + hot * 0.45
                # fallen petals: denser along the curb, the edging and on the grass
                # petals gather in drifts along the curb and the path edges (wind-swept), a light scatter
                # elsewhere; drift outlines wander along the path
                wob = (fbm2(s * 0.35, 1.7, 42, 3) - 0.5) * 1.0
                wob2 = (fbm2(s * 0.3, 8.1, 43, 3) - 0.5) * 1.0
                drift = 1.0 * (1.0 - sstep(0.12, 0.6 + 0.3 * wob, u - U_CURB1))
                if u < 1.45:
                    drift += 0.7 * (1.0 - sstep(0.1, 0.5 + 0.25 * wob2, U_PATH1 - u))
                drift += 0.45 * sstep(1.0, 1.6, u)
                patches = sstep(0.58, 0.72, fbm2(u * 0.7, s * 0.28, 41, 3))
                dens = 0.2 + drift * (0.55 + 0.45 * fbm2(u * 2.0, s * 1.2, 44, 2)) + 0.45 * patches
                if u < U_CURB1:
                    dens *= 0.5
                if u > U_PATH1 and u < U_PATH1 + 0.32:
                    dens *= 0.35
                pc = _petal_cells(u, s, fp, 0.042, min(dens, 0.95), 77)
                if dens > 0.7:
                    pc = max(pc, _petal_cells(u + 0.021, s + 0.013, fp, 0.042, min(dens - 0.3, 0.9), 78))
                pr = 0.74 + (1.02 - 0.74) * lc
                pg = 0.52 + (0.78 - 0.52) * lc
                pb = 0.76 + (0.86 - 0.76) * lc
                r += (pr - r) * pc
                g += (pg - g) * pc
                b += (pb - b) * pc
            elif m == 2:
                lay = 3
                r, g, b = 0.2, 0.4, 0.45
            elif m == 3:
                # far embankment wall: stone courses, in shade (faces away from the sun)
                cy = (Yh - Y_WATER) / 0.42
                row = math.floor(cy)
                off = 0.5 * (row % 2)
                cs = s / 0.9 + off
                col = math.floor(cs)
                jy = abs(cy - row - 0.5)
                jx = abs(cs - col - 0.5)
                pix = fp / 0.4
                joint = max(sstep(0.44 - pix, 0.5, jy), sstep(0.46 - pix, 0.5, jx) * 0.8)
                v = 0.93 + 0.08 * hash2(int(row), int(col), 13) + 0.1 * (fbm2(s * 0.2, Yh * 0.5, 14, 3) - 0.5)
                r, g, b = 0.6 * v, 0.63 * v, 0.76 * v
                jf = joint * (1 - sstep(0.02, 0.08, fp)) * 0.14
                r *= 1 - jf
                g *= 1 - jf
                b *= 1 - jf
                # wet dark band + moss at the waterline, lit coping on top
                wet = 1.0 - sstep(Y_WATER + 0.1, Y_WATER + 0.6, Yh)
                r += (0.26 - r) * wet
                g += (0.36 - g) * wet
                b += (0.4 - b) * wet
                top = sstep(Y_FAR - 0.18, Y_FAR - 0.06, Yh)
                r += (1.0 - r) * top
                g += (0.93 - g) * top
                b += (0.86 - b) * top
                streak = (fbm2(s * 1.5, Yh * 0.2, 17, 2) - 0.5) * 0.1
                r *= 1 + streak
                g *= 1 + streak
                b *= 1 + streak
            elif m == 4:
                # far bank ground / city ground
                n = fbm2(u * 0.3, s * 0.3, 19, 2)
                r, g, b = 0.72 + 0.05 * n, 0.72 + 0.04 * n, 0.8
                if u > -31.5:
                    r, g, b = 0.86, 0.8, 0.74
            else:
                bi = bid[y, x]
                typ = int(B[bi, 6])
                br, bg, bb = B[bi, 7], B[bi, 8], B[bi, 9]
                sd = B[bi, 10]
                lay = int(B[bi, 11])
                fc = face[y, x]
                ax = abs(fc)
                # normal (path frame) and light
                nu = 0.0
                nY = 0.0
                ns = 0.0
                if ax == 1:
                    nu = float(fc)
                elif ax == 2:
                    nY = float(fc)
                else:
                    ns = float(fc)
                if ax == 4:
                    # pitched roof plane
                    kk_ = (B[bi, 3] - B[bi, 2]) / (0.5 * ((B[bi, 1] - B[bi, 0]) if typ == 12 else (B[bi, 5] - B[bi, 4])))
                    nn_ = math.sqrt(1.0 + kk_ * kk_)
                    nY = 1.0 / nn_
                    if typ == 12:
                        nu = (1.0 if fc > 0 else -1.0) * kk_ / nn_
                        ns = 0.0
                    else:
                        ns = (1.0 if fc > 0 else -1.0) * kk_ / nn_
                        nu = 0.0
                lam = nu * Lu + nY * LY + ns * Ls
                # facade coordinates
                if ax == 1:
                    hc = s
                else:
                    hc = u
                vy = Yh - B[bi, 2]
                top = B[bi, 3] - Yh
                lit = sstep(-0.05, 0.25, lam)
                if typ == 5:
                    # land-side houses sit behind the blossom row: mottled canopy shade on the facades
                    # round 11: one clean painted canopy-shadow line across the upper floor (no mottle camo)
                    sl = 3.9 + 0.5 * math.sin(hc * 0.33 + sd) + 0.2 * math.sin(hc * 1.1 + 2.0 * sd)
                    lit *= 1.0 - sstep(sl - 0.04, sl + 0.04, vy)
                shade_r, shade_g, shade_b = 0.44, 0.5, 0.74     # cool sky-lit shadow multiplier (keeps local colour)
                if lay == 0:
                    # far bank (round 5): a warmer, lighter shade so each material keeps its hue
                    shade_r, shade_g, shade_b = 0.62, 0.6, 0.74
                if typ == 0 or typ == 5:
                    r, g, b = _facade(typ, br, bg, bb, lit, hc, vy, top, B[bi, 3] - B[bi, 2], sd, fp, ax,
                                      shade_r, shade_g, shade_b)
                elif typ == 1 or typ == 6:
                    # poles / thin rails: dark, blue in shade
                    r = br * (0.6 + 0.4 * lit)
                    g = bg * (0.65 + 0.35 * lit)
                    b = bb * (0.8 + 0.2 * lit)
                elif typ == 2 or typ == 3:
                    # bridge
                    r = br * (shade_r + (1.0 - shade_r) * lit)
                    g = bg * (shade_g + (0.98 - shade_g) * lit)
                    b = bb * (shade_b + (0.95 - shade_b) * lit)
                    if typ == 2 and top < 0.12 + fp:
                        r, g, b = 1.0, 0.95, 0.88
                    if typ == 3 and top < 0.1 + fp:
                        r, g, b = 1.0, 0.96, 0.9
                    if ax == 2 and fc < 0:
                        r, g, b = r * 0.5, g * 0.5, b * 0.6
                elif typ == 4:
                    r, g, b = _hedge_shade(u, Yh, s, fc, B[bi, 3], B[bi, 2], fp, Lu, LY, Ls)
                elif typ == 7:
                    # signboard: saturated panel with pale lettering blocks and a lit frame
                    r = br * (0.62 + 0.4 * lit)
                    g = bg * (0.66 + 0.36 * lit)
                    b = bb * (0.8 + 0.22 * lit)
                    hS = B[bi, 3] - B[bi, 2]
                    wS = max(B[bi, 1] - B[bi, 0], B[bi, 5] - B[bi, 4])
                    tall = hS > wS
                    fr_ = min(vy, top)
                    if fr_ < 0.08 + fp:
                        r, g, b = r * 0.55 + 0.4, g * 0.55 + 0.4, b * 0.55 + 0.42
                    elif fp < 0.0:
                        if tall:
                            k2 = math.floor(vy / 0.55)
                            ch = sstep(0.12, 0.16, vy / 0.55 - k2) * (1 - sstep(0.84, 0.88, vy / 0.55 - k2))
                        else:
                            k2 = math.floor(hc / 0.6)
                            ch = sstep(0.12, 0.16, hc / 0.6 - k2) * (1 - sstep(0.84, 0.88, hc / 0.6 - k2))
                            ch *= sstep(0.25, 0.3, vy / hS) * (1 - sstep(0.7, 0.75, vy / hS))
                        ch *= (hash2(int(k2), int(sd), 71) > 0.2) * (1 - sstep(0.03, 0.12, fp))
                        r += (0.98 - r) * ch * 0.85
                        g += (0.97 - g) * ch * 0.85
                        b += (0.94 - b) * ch * 0.85
                elif typ == 9:
                    # roof / eave / balcony slab: lit top, mid front edge, dark underside
                    r = br * (shade_r + (1.02 - shade_r) * lit)
                    g = bg * (shade_g + (0.98 - shade_g) * lit)
                    b = bb * (shade_b + (0.95 - shade_b) * lit)
                    if ax == 2 and fc < 0:
                        r, g, b = r * 0.45, g * 0.47, b * 0.6
                    elif ax == 2 and fc > 0:
                        r, g, b = r * 1.15, g * 1.12, b * 1.08
                    if ax != 2 and top < 0.05 + fp:
                        r, g, b = r * 0.6 + 0.42, g * 0.6 + 0.4, b * 0.6 + 0.38
                    if ax == 1 and fp < 0.05:
                        # tile courses on the roof front
                        tc = (vy / 0.12) - math.floor(vy / 0.12)
                        tl_ = (1.0 - sstep(0.0, 0.25, tc)) * 0.18 * (1.0 - fp / 0.05)
                        r *= 1 - tl_
                        g *= 1 - tl_
                        b *= 1 - tl_ * 0.6
                elif typ == 10:
                    # frosted balcony panel / railing
                    r = br * (0.72 + 0.3 * lit)
                    g = bg * (0.74 + 0.28 * lit)
                    b = bb * (0.86 + 0.16 * lit)
                    if top < 0.06 + fp:
                        r, g, b = 1.02, 0.99, 0.95
                    if fp < 0.05:
                        bx_ = hc / 0.12 - math.floor(hc / 0.12)
                        bar_ = (1.0 - sstep(0.15, 0.3, bx_)) * 0.18 * (1.0 - fp / 0.05)
                        r *= 1 - bar_
                        g *= 1 - bar_
                        b *= 1 - bar_ * 0.7
                elif typ == 11:
                    # vending machine: coloured body, glowing display with rows of bottles, dark slot
                    r = br * (0.7 + 0.3 * lit)
                    g = bg * (0.7 + 0.3 * lit)
                    b = bb * (0.78 + 0.22 * lit)
                    if ax == 1:
                        hS = B[bi, 3] - B[bi, 2]
                        fy_ = vy / hS
                        fx_ = (hc - B[bi, 4]) / max(B[bi, 5] - B[bi, 4], 0.1)
                        disp = sstep(0.5, 0.52, fy_) * (1 - sstep(0.9, 0.92, fy_)) * sstep(0.06, 0.08, fx_) * \
                            (1 - sstep(0.92, 0.94, fx_))
                        if disp > 0.0:
                            rowp = (fy_ - 0.5) / 0.4 * 3.0
                            colp = fx_ * 7.0
                            bot = (1.0 - sstep(0.25, 0.35, abs(colp - math.floor(colp) - 0.5))) * \
                                sstep(0.25, 0.45, rowp - math.floor(rowp)) * (1.0 - sstep(0.02, 0.05, fp))
                            hb_ = hash2(int(math.floor(colp)), int(math.floor(rowp)), int(sd))
                            cr_, cg_, cb_ = 0.98, 0.98, 0.96
                            if hb_ < 0.3:
                                cr_, cg_, cb_ = 0.95, 0.35, 0.3
                            elif hb_ < 0.55:
                                cr_, cg_, cb_ = 0.3, 0.55, 0.95
                            elif hb_ < 0.75:
                                cr_, cg_, cb_ = 0.98, 0.85, 0.3
                            rr_ = 1.02 + (cr_ - 1.02) * bot
                            gg_ = 1.0 + (cg_ - 1.0) * bot
                            bb_ = 0.97 + (cb_ - 0.97) * bot
                            r += (rr_ - r) * disp
                            g += (gg_ - g) * disp
                            b += (bb_ - b) * disp
                        slot = sstep(0.1, 0.12, fy_) * (1 - sstep(0.24, 0.26, fy_)) * sstep(0.15, 0.17, fx_) * \
                            (1 - sstep(0.85, 0.87, fx_))
                        r += (0.15 - r) * slot
                        g += (0.16 - g) * slot
                        b += (0.22 - b) * slot
                    if top < 0.05 + fp:
                        r, g, b = 1.0, 0.97, 0.93
                elif typ == 12 or typ == 13:
                    if ax == 4:
                        # kawara tile roof: courses running along the ridge, lit / shaded plane, glossy sky
                        # sheen on the plane facing the viewer, bright ridge line and dark eave shadow line
                        hc2 = s if typ == 12 else u
                        along = u if typ == 12 else s
                        lo_ = 0.5 * ((B[bi, 1] - B[bi, 0]) if typ == 12 else (B[bi, 5] - B[bi, 4]))
                        ctr = 0.5 * ((B[bi, 0] + B[bi, 1]) if typ == 12 else (B[bi, 4] + B[bi, 5]))
                        dist_r = abs(along - ctr) / lo_                     # 0 ridge .. 1 eave
                        lt2 = sstep(-0.1, 0.35, lam)
                        r = br * (0.55 + 0.6 * lt2)
                        g = bg * (0.6 + 0.52 * lt2)
                        b = bb * (0.78 + 0.34 * lt2)
                        # sky sheen: glossy tiles mirror the pale sky toward the lower part of the plane
                        sh2 = (1.0 - lt2) * sstep(0.2, 1.0, dist_r) * 0.35
                        r += (0.72 - r) * sh2
                        g += (0.84 - g) * sh2
                        b += (0.98 - b) * sh2
                        if fp < 0.06:
                            # round 6: legible kawara courses (dark shadow line under each course, lit lip above
                            # it), vertical tile joints and a per-tile tone variation (no flat CG plane)
                            tc = (dist_r * lo_) / 0.3
                            fct = tc - math.floor(tc)
                            fdm = 1.0 - fp / 0.06
                            cl2 = (1.0 - sstep(0.0, 0.34 + fp / 0.3, fct)) * 0.5 * fdm
                            lip = sstep(0.55, 0.95, fct) * 0.14 * fdm * (0.4 + 0.6 * lt2)
                            vl = hc2 / 0.3 - math.floor(hc2 / 0.3)
                            cl2 += (1.0 - sstep(0.0, 0.15 + fp / 0.3, vl)) * 0.16 * fdm
                            tv = (hash2(int(math.floor(hc2 / 0.3)), int(math.floor(tc)), int(sd)) - 0.5) * 0.12 * fdm
                            r = r * (1.0 - cl2 + tv) + lip
                            g = g * (1.0 - cl2 + tv) + lip
                            b = b * (1.0 - cl2 * 0.7 + tv) + lip * 0.9
                        rid = 1.0 - sstep(0.02, 0.06 + fp / lo_, dist_r)
                        r += (1.02 - r) * rid * 0.7
                        g += (0.98 - g) * rid * 0.7
                        b += (0.95 - b) * rid * 0.7
                        eav = sstep(0.93 - fp / lo_, 0.97, dist_r)
                        r += (0.97 - r) * eav * 0.6 * lt2
                        g += (0.94 - g) * eav * 0.6 * lt2
                        b += (0.9 - b) * eav * 0.6 * lt2
                    elif ax == 2:
                        r, g, b = br * 0.35, bg * 0.38, bb * 0.55      # eave soffit
                    else:
                        # gable-end triangle: plaster wall, board trim under the verge
                        r = 0.93 * (0.5 + 0.55 * lit)
                        g = 0.9 * (0.55 + 0.46 * lit)
                        b = 0.86 * (0.8 + 0.2 * lit)
                        if top < 0.1 + fp:
                            r, g, b = br * 0.5, bg * 0.5, bb * 0.62
                elif typ == 15:
                    # noren: split cloth panels (dark slits), a pale hem band and a white crest in the middle
                    lw_ = 0.4
                    ph_ = (hc - B[bi, 4]) / lw_
                    slit = sstep(0.0, 0.06 + fp / lw_, min(ph_ - math.floor(ph_), 1.0 - (ph_ - math.floor(ph_))))
                    fold = 0.9 + 0.1 * math.sin(hc * 30.0 + sd)
                    r = br * (0.7 + 0.35 * lit) * fold
                    g = bg * (0.7 + 0.35 * lit) * fold
                    b = bb * (0.8 + 0.25 * lit) * fold
                    hS = B[bi, 3] - B[bi, 2]
                    fy_ = vy / hS
                    top_b = sstep(0.82, 0.86, fy_)
                    r += (0.9 - r) * top_b * 0.5
                    g += (0.88 - g) * top_b * 0.5
                    b += (0.86 - b) * top_b * 0.5
                    cx_ = (hc - 0.5 * (B[bi, 4] + B[bi, 5])) / 0.28
                    cy_ = (fy_ - 0.5) * hS / 0.28
                    crest = (1.0 - sstep(0.75, 1.0, math.sqrt(cx_ * cx_ + cy_ * cy_))) * (1.0 - sstep(0.03, 0.08, fp))
                    r += (0.95 - r) * crest * 0.8
                    g += (0.94 - g) * crest * 0.8
                    b += (0.92 - b) * crest * 0.8
                    r *= 0.35 + 0.65 * slit
                    g *= 0.35 + 0.65 * slit
                    b *= 0.4 + 0.6 * slit
                elif typ == 14:
                    # laundry / futon hung over a balcony rail: soft folds, lit and shaded
                    fold = 0.9 + 0.1 * math.sin(hc * 25.0 + sd)
                    r = br * (0.66 + 0.4 * lit) * fold
                    g = bg * (0.68 + 0.36 * lit) * fold
                    b = bb * (0.8 + 0.24 * lit) * fold
                elif typ == 8:
                    # rooftop water tank / plant box: pale, banded, lit rim on top
                    r = br * (shade_r + (1.02 - shade_r) * lit)
                    g = bg * (shade_g + (0.98 - shade_g) * lit)
                    b = bb * (shade_b + (0.94 - shade_b) * lit)
                    band = math.floor(vy / 0.45)
                    if vy / 0.45 - band < 0.12 and fp < 0.1:
                        r, g, b = r * 0.82, g * 0.82, b * 0.86
                    if top < 0.08 + fp:
                        r, g, b = 1.02, 0.97, 0.92
            if lay == 5:
                # round 5: land-side houses sit under the blossom tunnel - their upper floors are in the
                # canopy's cool shade (the lacy blossom edge reads cleanly against it)
                cs_ = sstep(Y_PATH + 2.6, Y_PATH + 4.6, Yh) * 0.4
                r *= 1.0 - cs_ * 1.0
                g *= 1.0 - cs_ * 0.95
                b *= 1.0 - cs_ * 0.55
            # aerial perspective
            hz = haze_max * (1.0 - math.exp(-Z / haze_k))
            if m == 1:
                hz *= 0.4
            hr_, hg_, hb_ = haze[0], haze[1], haze[2]
            if m == 5 and lay == 0 and bid[y, x] >= 0:
                tyb = int(B[bid[y, x], 6])
                if tyb == 0 or tyb == 7 or tyb == 8 or tyb == 1 or tyb >= 9:
                    # city: bluer and lighter with distance (aerial perspective)
                    hz = 0.62 * (1.0 - math.exp(-Z / 330.0))
                    hr_, hg_, hb_ = 0.78, 0.86, 0.97
            r += (hr_ - r) * hz
            g += (hg_ - g) * hz
            b += (hb_ - b) * hz
            out[y, x, 0] = r
            out[y, x, 1] = g
            out[y, x, 2] = b
            out[y, x, 3] = 1.0
            layer[y, x] = lay


def box_bboxes(cam, B, mx, ss):
    """Screen bbox (ss px) of each box for culling."""
    for i in range(B.shape[0]):
        us = [B[i, 0], B[i, 1]]
        Ys = [B[i, 2], B[i, 3]]
        sv = [B[i, 4], B[i, 5]]
        xs, ys = [], []
        bad = False
        for u in us:
            for Y in Ys:
                for s in sv:
                    X, Yc, Z = cam.to_cam(u, Y, s)
                    if Z < 0.2:
                        bad = True
                        continue
                    x, y = cam.proj(X, Yc, Z, 0, mx, ss)
                    xs.append(float(x))
                    ys.append(float(y))
        if bad or not xs:
            B[i, 12:16] = (-1e9, 1e9, -1e9, 1e9)
        else:
            B[i, 12:16] = (min(xs) - 2, max(xs) + 2, min(ys) - 2, max(ys) + 2)
    return B


SIGN_COLS = [(0.86, 0.22, 0.24), (0.2, 0.42, 0.82), (0.97, 0.8, 0.22), (0.18, 0.6, 0.42), (0.95, 0.95, 0.93),
             (0.93, 0.45, 0.2)]


def rooftops(B, seed=17):
    """Break the city silhouettes: water tanks, stair houses, antennas, rooftop billboards and vertical
    signboards on the nearer far-bank buildings (call after the building heights are final)."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(u0, u1, Y0, Y1, s0, s1, typ, col, layer=0):
        rows.append([u0, u1, Y0, Y1, s0, s1, typ, col[0], col[1], col[2], rng.integers(0, 1000), layer,
                     0, 0, 0, 0])

    for i in range(B.shape[0]):
        if B[i, 6] != 0:
            continue
        u0, u1, Y0, Y1, s0, s1 = B[i, :6]
        if u1 < -100 or s0 > 700:
            continue
        if Y1 - Y0 < 8.5 and u1 > -42 and s0 <= 300:
            continue                       # low houses get pitched roofs instead (riverside_details)
        w = s1 - s0
        d = u1 - u0
        # stair / lift housing
        if rng.random() < 0.55:
            a = s0 + rng.uniform(0.15, 0.6) * w
            b0 = u0 + rng.uniform(0.2, 0.5) * d
            if rng.random() < 0.35:
                add(b0, b0 + 2.6, Y1, Y1 + 2.6, a, a + 2.8, 8, (0.9, 0.9, 0.92))
        # water tanks on legs
        for k in range(int(rng.integers(1, 3))):
            a = s0 + rng.uniform(0.05, 0.8) * w
            b0 = u0 + rng.uniform(0.1, 0.8) * d
            hL = rng.uniform(0.8, 1.6)
            add(b0 + 0.2, b0 + 0.3, Y1, Y1 + hL, a + 0.2, a + 0.3, 1, (0.5, 0.52, 0.58))
            add(b0 + 1.7, b0 + 1.8, Y1, Y1 + hL, a + 1.7, a + 1.8, 1, (0.5, 0.52, 0.58))
            add(b0, b0 + 2.0, Y1 + hL, Y1 + hL + 1.8, a, a + 2.0, 8, (0.92, 0.94, 0.97))
        # antennas
        for k in range(int(rng.integers(1, 4))):
            a = s0 + rng.uniform(0.1, 0.9) * w
            b0 = u0 + rng.uniform(0.1, 0.9) * d
            if rng.random() < 0.6:
                add(b0, b0 + 0.12, Y1, Y1 + rng.uniform(2.5, 6.0), a, a + 0.12, 1, (0.55, 0.57, 0.62))
            if rng.random() < 0.6:
                hh = Y1 + rng.uniform(2.0, 4.0)
                add(b0 - 0.8, b0 + 0.9, hh, hh + 0.08, a, a + 0.1, 1, (0.55, 0.57, 0.62))
        # rooftop billboard facing the river
        if rng.random() < 0.28 and w > 7:
            bw = min(w - 2.0, rng.uniform(6, 11))
            a = s0 + rng.uniform(0.5, w - bw - 0.5)
            hb = rng.uniform(2.0, 3.2)
            add(u1 - 1.2, u1 - 0.9, Y1 + 0.6, Y1 + 0.6 + hb, a, a + bw, 7,
                SIGN_COLS[int(rng.integers(len(SIGN_COLS)))])
            add(u1 - 1.3, u1 - 1.2, Y1, Y1 + 0.7, a + 0.5, a + 0.6, 1, (0.45, 0.47, 0.52))
            add(u1 - 1.3, u1 - 1.2, Y1, Y1 + 0.7, a + bw - 0.6, a + bw - 0.5, 1, (0.45, 0.47, 0.52))
        # vertical (tategaki) signboard sticking out of the river-facing facade
        if rng.random() < 0.35 and Y1 - Y0 > 9:
            a = s0 + rng.uniform(0.1, 0.4) * w
            hs_ = rng.uniform(4.0, min(9.0, Y1 - Y0 - 3))
            ys = Y0 + rng.uniform(2.5, Y1 - Y0 - hs_ - 0.5)
            add(u1, u1 + 1.1, ys, ys + hs_, a, a + 0.35, 7, SIGN_COLS[int(rng.integers(len(SIGN_COLS)))])
        # rooftop parapet rail
        if rng.random() < 0.5:
            add(u1 - 0.1, u1, Y1, Y1 + 1.0, s0, s1, 1, (0.62, 0.64, 0.7))
    if not rows:
        return B
    return np.concatenate([B, np.array(rows, np.float64)], 0)


LAUNDRY = [(0.97, 0.97, 0.96), (0.95, 0.95, 0.92), (0.55, 0.7, 0.92), (0.98, 0.72, 0.76), (0.98, 0.9, 0.6),
           (0.62, 0.84, 0.72), (0.96, 0.96, 0.98)]
ROOF_COLS = [(0.3, 0.36, 0.5), (0.26, 0.3, 0.38), (0.55, 0.36, 0.32), (0.3, 0.42, 0.44), (0.42, 0.4, 0.46)]

NOREN_COLS = [(0.16, 0.22, 0.46), (0.62, 0.16, 0.18), (0.2, 0.36, 0.3), (0.3, 0.2, 0.36)]
VEND_COLS = [(0.92, 0.2, 0.22), (0.2, 0.42, 0.85), (0.96, 0.96, 0.95), (0.95, 0.55, 0.2)]


def riverside_details(B, seed=23):
    """Street-level life on the front row of far-bank buildings: pitched eaves / roofs, balcony slabs with
    frosted panels, AC units, small shop signs and vending machines."""
    rng = np.random.default_rng(seed)
    rows = []

    def add(u0, u1, Y0, Y1, s0, s1, typ, col, layer=0):
        rows.append([u0, u1, Y0, Y1, s0, s1, typ, col[0], col[1], col[2], rng.integers(0, 1000), layer,
                     0, 0, 0, 0])

    roof_cols = ROOF_COLS
    nv = 0
    for i in range(B.shape[0]):
        if B[i, 6] != 0:
            continue
        u0, u1, Y0, Y1, s0, s1 = B[i, :6]
        if u1 < -42 or s0 > 300:
            continue
        w = s1 - s0
        d = u1 - u0
        hB = Y1 - Y0
        low = hB < 8.5
        rc = roof_cols[int(rng.integers(len(roof_cols)))]
        if low:
            # house: pitched kawara-tile roof (ridge along or across the river), overhanging eaves
            rh = rng.uniform(1.6, 2.6)
            if rng.random() < 0.55:
                add(u0 - 0.5, u1 + 0.8, Y1 - 0.05, Y1 + rh, s0 - 0.5, s1 + 0.5, 12, rc)
            else:
                add(u0 - 0.5, u1 + 0.8, Y1 - 0.05, Y1 + rh * 1.2, s0 - 0.5, s1 + 0.5, 13, rc)
            if rng.random() < 0.6:
                # ground-floor pent roof (hisashi) over the entrance
                add(u1, u1 + 0.9, Y0 + 2.5, Y0 + 2.62, s0 + 0.4, s1 - 0.4, 9, rc)
            # small balcony with laundry on the upper floor
            if rng.random() < 0.7 and hB > 5.0:
                a0 = s0 + rng.uniform(0.3, 0.3 * w)
                a1 = min(s1 - 0.3, a0 + rng.uniform(2.5, 4.5))
                yf = Y0 + 2.8
                add(u1, u1 + 0.8, yf - 0.14, yf, a0, a1, 9, (0.9, 0.9, 0.93))
                add(u1 + 0.72, u1 + 0.8, yf, yf + 0.95, a0, a1, 10, (0.86, 0.9, 0.98))
                xx = a0 + 0.2
                while xx < a1 - 0.5:
                    lw = rng.uniform(0.35, 0.7)
                    add(u1 + 0.5, u1 + 0.55, yf + 0.35, yf + 1.6, xx, xx + lw, 14,
                        LAUNDRY[int(rng.integers(len(LAUNDRY)))])
                    xx += lw + rng.uniform(0.05, 0.4)
        else:
            add(u1 - 0.1, u1 + 0.25, Y1 - 0.1, Y1 + 0.15, s0 - 0.15, s1 + 0.15, 9, (0.9, 0.9, 0.92))
        # balcony slabs + frosted panels on some storeys
        if rng.random() < 0.65 and hB > 5.5:
            fh = 2.8
            k = 1
            a0 = s0 + rng.uniform(0.3, 0.2 * w)
            a1 = s1 - rng.uniform(0.3, 0.2 * w)
            while Y0 + k * fh + 1.0 < Y1 - 0.3:
                yf = Y0 + k * fh
                add(u1, u1 + 1.0, yf - 0.16, yf, a0, a1, 9, (0.9, 0.9, 0.93))
                add(u1 + 0.92, u1 + 1.0, yf, yf + 1.0, a0, a1, 10, (0.86, 0.9, 0.98))
                if rng.random() < 0.4:
                    xx = a0 + rng.uniform(0.2, 2.0)
                    for m_ in range(int(rng.integers(1, 4))):
                        lw = rng.uniform(0.4, 0.8)
                        if xx + lw > a1:
                            break
                        add(u1 + 0.6, u1 + 0.65, yf + 0.5, yf + 1.9, xx, xx + lw, 14,
                            LAUNDRY[int(rng.integers(len(LAUNDRY)))])
                        xx += lw + rng.uniform(0.05, 0.3)
                if rng.random() < 0.5:
                    a = a0 + rng.uniform(0.2, max(0.3, a1 - a0 - 1.2))
                    add(u1 + 0.1, u1 + 0.45, yf, yf + 0.6, a, a + 0.8, 8, (0.93, 0.93, 0.94))
                k += 1
        for kk in range(int(rng.integers(0, 3))):
            a = s0 + rng.uniform(0.3, max(0.4, w - 1.2))
            yy = Y0 + rng.choice([0.1, 3.1, 6.0])
            if yy + 0.6 < Y1:
                add(u1, u1 + 0.32, yy, yy + 0.62, a, a + 0.82, 8, (0.93, 0.93, 0.95))
        if rng.random() < 0.35:
            # round 5: projecting shop sign facing the camera (big enough to letter legibly), on a bracket
            a = s0 + rng.uniform(0.3, max(0.4, w - 3.5))
            wl_ = rng.uniform(2.2, 3.2)
            add(u1 + 0.1, u1 + 0.1 + 1.2 * wl_, Y0 + 2.6, Y0 + 3.85, a, a + 0.2, 7,
                SIGN_COLS[int(rng.integers(len(SIGN_COLS)))])
            add(u1 - 0.05, u1 + 0.15, Y0 + 3.6, Y0 + 3.7, a + 0.05, a + 0.15, 1, (0.45, 0.47, 0.52))
            # a noren curtain over the shop door under it
            add(u1 + 0.02, u1 + 0.08, Y0 + 1.35, Y0 + 2.35, a + 0.4, a + 2.0, 15,
                NOREN_COLS[int(rng.integers(len(NOREN_COLS)))])
        # round 3: legible lettering - signboards that FACE THE CAMERA (normal -s), big enough to read
        if not low and rng.random() < 0.35:
            # rooftop billboard standing across the roof, on two steel legs
            bw = min(d - 1.0, rng.uniform(6.5, 8.5))
            hb = rng.uniform(3.2, 3.9)
            uu0 = u1 - 0.4 - bw
            a = s0 + rng.uniform(0.6, 2.0)
            add(uu0, uu0 + bw, Y1 + 0.9, Y1 + 0.9 + hb, a, a + 0.22, 7,
                SIGN_COLS[int(rng.integers(len(SIGN_COLS)))])
            for lx in (uu0 + 0.5, uu0 + bw - 0.6):
                add(lx, lx + 0.1, Y1, Y1 + 0.95, a + 0.05, a + 0.15, 1, (0.45, 0.47, 0.52))
        if rng.random() < 0.35:
            # tategaki signboard projecting from the river facade, facing the camera
            hs_ = rng.uniform(3.2, min(5.0, max(3.3, hB - 1.0)))
            ys = Y0 + rng.uniform(1.0, max(1.1, hB - hs_ - 0.3))
            a = s0 + rng.uniform(0.3, 0.3 * w)
            add(u1, u1 + 1.5, ys, ys + hs_, a, a + 0.22, 7, SIGN_COLS[int(rng.integers(len(SIGN_COLS)))])
            add(u1 - 0.1, u1 + 0.05, ys + 0.3, ys + 0.4, a + 0.05, a + 0.15, 1, (0.45, 0.47, 0.52))
        if rng.random() < 0.4:
            # horizontal shop fascia across the camera-facing (downstream) end of the building
            add(u0 + 0.4, u1 - 0.4, Y0 + 2.6, Y0 + 3.5, s0 - 0.15, s0, 7,
                SIGN_COLS[int(rng.integers(len(SIGN_COLS)))])
        if rng.random() < 0.3 and nv < 6:
            # noren curtain over a ground-floor shop entrance
            a = s0 + rng.uniform(0.3, max(0.4, w - 2.2))
            add(u1 + 0.02, u1 + 0.08, Y0 + 1.35, Y0 + 2.35, a, a + 1.6, 15,
                NOREN_COLS[int(rng.integers(len(NOREN_COLS)))])
        if rng.random() < 0.45 and nv < 10:
            a = s0 + rng.uniform(0.3, max(0.4, w - 2.4))
            for m in range(int(rng.integers(1, 3))):
                add(u1 + 0.05, u1 + 0.8, Y0, Y0 + 1.85, a + m * 1.05, a + m * 1.05 + 1.0, 11,
                    VEND_COLS[int(rng.integers(len(VEND_COLS)))])
            nv += 1
    if not rows:
        return B
    return np.concatenate([B, np.array(rows, np.float64)], 0)


def make_boxes(seed=5):
    rng = np.random.default_rng(seed)
    boxes = []

    def add(u0, u1, Y0, Y1, s0, s1, typ, col, layer=0):
        boxes.append([u0, u1, Y0, Y1, s0, s1, typ, col[0], col[1], col[2], rng.integers(0, 1000), layer,
                      0, 0, 0, 0])

    # round 5: varied materials - cream / beige mortar, tan tile, terracotta, grey-green, brick, white
    facade = [(0.97, 0.93, 0.84), (0.96, 0.84, 0.66), (0.8, 0.84, 0.86), (0.9, 0.74, 0.56), (0.84, 0.56, 0.44),
              (0.66, 0.76, 0.68), (0.99, 0.97, 0.93), (0.74, 0.5, 0.42), (0.93, 0.88, 0.72)]
    # far-bank city rows: (u_front, depth, s range, height range, width range, gap)
    rows = [(-37.5, 9, (12, 420), (5.5, 12), (7, 15), (1.0, 4.0)),
            (-52, 12, (30, 520), (9, 22), (10, 20), (3, 10)),
            (-78, 18, (60, 700), (14, 36), (12, 24), (8, 30)),
            (-130, 25, (160, 1100), (30, 75), (15, 30), (25, 80)),
            (-240, 40, (400, 1600), (50, 120), (20, 40), (60, 180))]
    for uf, dp, (s0, s1), (h0, h1), (w0, w1), (g0, g1) in rows:
        s = s0 + rng.uniform(0, 5)
        while s < s1:
            w = rng.uniform(w0, w1)
            h = rng.uniform(h0, h1)
            if rng.random() < 0.15:
                h *= 1.5
            u0 = uf - rng.uniform(0, 3)
            add(u0 - dp, u0, Y_FAR, Y_FAR + h, s, s + w, 0, facade[rng.integers(len(facade))])
            s += w + rng.uniform(g0, g1)
    # land side: hedge + houses behind the right tree row
    add(6.0, 6.8, Y_PATH, Y_PATH + 1.5, 3.0, 400.0, 4, (0.2, 0.4, 0.3), 4)
    s = 4.0
    while s < 380:
        w = rng.uniform(7, 11)
        h = rng.uniform(5.0, 7.5)
        u0 = rng.uniform(8.0, 10.0)
        pale = [(0.97, 0.94, 0.88), (0.95, 0.9, 0.8), (0.88, 0.9, 0.9), (0.99, 0.97, 0.94), (0.92, 0.86, 0.78)]
        add(u0, u0 + 9, Y_PATH, Y_PATH + h, s, s + w, 5, pale[int(rng.integers(len(facade))) % len(pale)], 5)
        rc = ROOF_COLS[int(rng.integers(len(ROOF_COLS)))]
        rh = rng.uniform(1.8, 2.8)
        # pitched tile roof with overhanging eaves (ridge along the path or across it)
        add(u0 - 0.6, u0 + 9.6, Y_PATH + h - 0.05, Y_PATH + h + rh, s - 0.6, s + w + 0.6,
            12 if rng.random() < 0.6 else 13, rc, 5)
        yr = Y_PATH + h
        # path-facing upper-floor balcony: slab, frosted panel, handrail, laundry, AC unit
        a0 = s + rng.uniform(0.4, 1.5)
        a1 = min(s + w - 0.4, a0 + rng.uniform(3.0, 5.5))
        yf = Y_PATH + 2.8
        add(u0 - 0.9, u0, yf - 0.15, yf, a0, a1, 9, (0.9, 0.9, 0.93), 5)
        add(u0 - 0.9, u0 - 0.82, yf, yf + 1.0, a0, a1, 10, (0.86, 0.9, 0.98), 5)
        xx = a0 + 0.3
        while xx < a1 - 0.5 and rng.random() < 0.8:
            lw = rng.uniform(0.35, 0.8)
            add(u0 - 0.6, u0 - 0.55, yf + 0.45, yf + 1.75, xx, xx + lw, 14, LAUNDRY[int(rng.integers(len(LAUNDRY)))],
                5)
            xx += lw + rng.uniform(0.05, 0.5)
        if rng.random() < 0.7:
            a = a1 - 1.0
            add(u0 - 0.45, u0, yf, yf + 0.62, a, a + 0.82, 8, (0.93, 0.93, 0.95), 5)
        # ground-floor AC unit and a pent roof over the door
        a = s + rng.uniform(0.5, max(0.6, w - 1.5))
        add(u0 - 0.35, u0, Y_PATH, Y_PATH + 0.62, a, a + 0.82, 8, (0.9, 0.9, 0.92), 5)
        if rng.random() < 0.6:
            a = s + rng.uniform(0.3, max(0.4, w - 2.5))
            add(u0 - 0.8, u0, Y_PATH + 2.3, Y_PATH + 2.42, a, a + 2.0, 9, rc, 5)
        s += w + rng.uniform(2, 5)
    # utility poles on the far bank (overlay layer 2), cross-arms
    poles = []
    for sp in np.arange(24.0, 420.0, 34.0):
        u = -31.0
        add(u - 0.14, u + 0.14, Y_FAR, Y_FAR + 10.5, sp - 0.14, sp + 0.14, 1, (0.55, 0.55, 0.6), 2)
        add(u - 0.9, u + 0.9, Y_FAR + 9.6, Y_FAR + 9.75, sp - 0.06, sp + 0.06, 1, (0.45, 0.45, 0.5), 2)
        add(u - 0.7, u + 0.7, Y_FAR + 8.6, Y_FAR + 8.72, sp - 0.05, sp + 0.05, 1, (0.45, 0.45, 0.5), 2)
        poles.append((u, sp))
    # far railing (overlay)
    add(-30.25, -30.2, Y_FAR + 0.95, Y_FAR + 1.02, 5.0, 400.0, 6, (0.9, 0.92, 0.95), 2)
    add(-30.25, -30.2, Y_FAR + 0.5, Y_FAR + 0.54, 5.0, 400.0, 6, (0.9, 0.92, 0.95), 2)
    for sp in np.arange(6.0, 200.0, 2.0):
        add(-30.26, -30.19, Y_FAR, Y_FAR + 1.02, sp - 0.03, sp + 0.03, 6, (0.9, 0.92, 0.95), 2)
    # bridge near the vanishing point
    s0 = 128.0
    add(-31.0, -4.6, Y_PATH - 0.9, Y_PATH, s0, s0 + 8.0, 2, (0.9, 0.9, 0.9))
    add(-31.0, -4.6, Y_PATH, Y_PATH + 1.0, s0, s0 + 0.25, 3, (0.92, 0.94, 0.96))
    add(-31.0, -4.6, Y_PATH, Y_PATH + 1.0, s0 + 7.75, s0 + 8.0, 3, (0.92, 0.94, 0.96))
    B = np.array(boxes, np.float64)
    return B, poles


# ============================================================================ per-frame water

@njit(cache=True, parallel=True, fastmath=True)
def water_kernel(out, wy, wx, wu, ws, wZ, wa, wry, refl, ywb, t, f, sunx, suny_m, deep, sun_col, invz, kf):
    """Shade water pixels (listed) into out (H,W,3) premultiplied-add by coverage wa."""
    n = wy.shape[0]
    H, W = refl.shape[0], refl.shape[1]
    for k in prange(n):
        y = wy[k]
        x = wx[k]
        u = wu[k]
        s = ws[k]
        Z = wZ[k]
        fp = Z / f
        # ripple slope field (world space, flowing toward the camera)
        sf = s + 0.25 * t
        a1 = math.sin(s * 3.1 + u * 0.7 - t * 1.9) + 0.6 * math.sin(s * 5.3 - u * 1.3 + t * 2.3)
        a2 = fbm2(u * 0.9, sf * 2.2, 51, 3) - 0.5
        a3 = fbm2(u * 0.35, sf * 0.5 - t * 0.05, 52, 2) - 0.5
        slope = 0.35 * a1 * 0.1 + a2 * 0.9 + a3 * 0.6
        # screen offsets of the reflection: ripples slice it into horizontal bands that shift sideways
        dyr = slope * 0.05 * f / Z + slope * 0.6
        band = fbm2(u * 0.25 + 7, sf * 3.2 - t * 0.4, 53, 3) - 0.5
        dxr = (band * 0.085 + (fbm2(u * 1.3, sf * 1.1, 56, 2) - 0.5) * 0.04 + 0.012 * math.sin(sf * 6.0 + u * 0.8 - t * 1.5)) * f / Z
        ry_ = 2.0 * ywb[x] - y + dyr
        # parallax: the water plate moves with the water's depth, the mirrored objects with their own
        iyo = min(max(int(2.0 * ywb[x] - y), 0), H - 1)
        rx_ = x + dxr - kf * (invz[y, x] - invz[iyo, x])
        if ry_ < 0:
            ry_ = 0.0
        if ry_ > H - 2:
            ry_ = H - 2.0
        if rx_ < 0:
            rx_ = 0.0
        if rx_ > W - 2:
            rx_ = W - 2.0
        iy = int(ry_)
        ix = int(rx_)
        fy = ry_ - iy
        fx = rx_ - ix
        cr = (refl[iy, ix, 0] * (1 - fx) + refl[iy, ix + 1, 0] * fx) * (1 - fy) + \
             (refl[iy + 1, ix, 0] * (1 - fx) + refl[iy + 1, ix + 1, 0] * fx) * fy
        cg = (refl[iy, ix, 1] * (1 - fx) + refl[iy, ix + 1, 1] * fx) * (1 - fy) + \
             (refl[iy + 1, ix, 1] * (1 - fx) + refl[iy + 1, ix + 1, 1] * fx) * fy
        cb = (refl[iy, ix, 2] * (1 - fx) + refl[iy, ix + 1, 2] * fx) * (1 - fy) + \
             (refl[iy + 1, ix, 2] * (1 - fx) + refl[iy + 1, ix + 1, 2] * fx) * fy
        # fresnel
        cth = min(abs(wry[k]), 1.0)
        F = 0.05 + 0.95 * (1 - cth) ** 4
        F = min(0.62 + F * 0.6, 0.9)
        # near-bank shadow of the embankment on the water (cool)
        nb = sstep(-9.5, -5.0, u)
        dr = deep[0] * (1 - 0.5 * nb)
        dg = deep[1] * (1 - 0.5 * nb)
        db = deep[2] * (1 - 0.4 * nb)
        # darker, deeper water in the shadow of the near wall (reflection dimmed too)
        rk = 1.0 - 0.3 * nb
        r = dr + (cr * 0.9 * rk - dr) * F
        g = dg + (cg * 0.94 * rk - dg) * F
        b = db + (cb * 0.98 * rk - db) * F
        # the far wall's dark wet foot line where wall meets its reflection
        wl = math.exp(-((y - ywb[x]) / (1.2 + 0.25 * f / Z)) ** 2) * 0.55
        r *= 1 - wl
        g *= 1 - wl * 0.9
        b *= 1 - wl * 0.75
        # ripple crest highlights reflecting the bright sky: thin horizontal streaks
        sw = math.sin(sf * 9.0 + 3.0 * fbm2(u * 0.35, sf * 0.8, 54, 2) * 6.283 - t * 1.2)
        wid = min(1.0, 0.25 + fp * 30.0)
        streak = sstep(1.0 - 0.12 * wid, 1.0, sw) * sstep(0.35, 0.65, fbm2(u * 0.6 + 3.0, sf * 0.9, 55, 2))
        cr_ = sstep(0.14, 0.3, slope) * 0.06 + streak * 0.22 * (1 - sstep(0.02, 0.06, fp))
        # darker ripple troughs (cool) - gives the surface a painted, banded look
        tr = sstep(0.05, 0.3, -band) * 0.36
        r -= r * tr * 0.9
        g -= g * tr * 0.6
        b -= b * tr * 0.4
        r += cr_ * 1.0
        g += cr_ * 0.95
        b += cr_ * 0.86
        # sun glitter below the sun
        gx = (x - sunx) / (0.07 * W)
        gy = (y - suny_m) / (0.2 * H)
        gw = math.exp(-gx * gx - gy * gy * 0.6)
        if gw > 0.01:
            cell = 0.35 + fp * 3.0
            cu = math.floor(u / cell)
            cs = math.floor(sf / (cell * 0.5))
            hh = hash2(int(cu), int(cs), 61)
            ph = hash2(int(cu), int(cs), 62) * 6.283
            tw = max(0.0, math.sin(t * (2.0 + 3.0 * hh) + ph)) ** 6
            pu = (cu + 0.5) * cell
            ps = (cs + 0.5) * cell * 0.5
            dd = ((u - pu) / (cell * 0.18)) ** 2 + ((sf - ps) / (cell * 0.07)) ** 2
            sp = math.exp(-dd * 1.5) * tw * (hh > 0.35)
            gl = gw * (sp * 1.2 + 0.1 * sstep(0.0, 0.3, slope))
            r += sun_col[0] * gl
            g += sun_col[1] * gl
            b += sun_col[2] * gl
        # petal rafts drifting on the water (hanaikada)
        sr_ = s + 0.35 * t      # rafts drift downstream (toward the camera) slowly
        d1 = fbm2(u * 0.3, sr_ * 0.045, 71, 3)
        d2 = fbm2(u * 1.1, sr_ * 0.25, 72, 2)
        # hanaikada: long ribbons of petals hugging both banks (wandering width, torn into pieces) + a few
        # loose rafts drifting mid-stream
        w1 = 0.5 + 1.4 * fbm2(sr_ * 0.07, 4.1, 74, 2)
        c1 = -29.9 + 0.5 * w1 + (fbm2(sr_ * 0.05, 1.3, 73, 2) - 0.5) * 1.2
        w2 = 0.3 + 0.9 * fbm2(sr_ * 0.09, 7.7, 76, 2)
        c2 = -4.75 - 0.5 * w2 + (fbm2(sr_ * 0.06, 2.9, 77, 2) - 0.5) * 0.6
        rib = max(1.0 - sstep(0.6, 1.0, abs(u - c1) / w1), 1.0 - sstep(0.6, 1.0, abs(u - c2) / w2))
        brk = fbm2(u * 0.4, sr_ * 0.1, 75, 3)
        rib *= sstep(0.25, 0.36, brk)
        dens = max(sstep(0.3, 0.45, rib * (0.7 + 0.6 * d2)), sstep(0.62, 0.72, d1 + (d2 - 0.5) * 0.3))
        sf = sr_
        if dens > 0.0:
            cov = 0.0
            if fp < 0.025:
                cell = 0.06
                iu = math.floor(u / cell)
                is_ = math.floor(sf / cell)
                for du in range(-1, 2):
                    for ds in range(-1, 2):
                        cu2 = iu + du
                        cs2 = is_ + ds
                        if hash2(cu2, cs2, 81) > dens * 0.95:
                            continue
                        pu = (cu2 + hash2(cu2, cs2, 82)) * cell
                        ps = (cs2 + hash2(cu2, cs2, 83)) * cell
                        rr = cell * (0.25 + 0.12 * hash2(cu2, cs2, 84))
                        q = math.sqrt(((u - pu) / rr) ** 2 + ((sf - ps) / (rr * 0.7)) ** 2)
                        c = 1 - sstep(1 - fp / rr, 1 + fp / rr, q)
                        if c > cov:
                            cov = c
                far = sstep(0.008, 0.025, fp)
                cov = cov * (1 - far) + dens * 0.9 * far
            else:
                cov = dens * 0.9
            cov = max(cov, dens * 0.72)
            # raft body: pale pink with a lit rim, slightly shadowed in the middle
            pr = 1.0 - 0.08 * dens
            r += (pr - r) * cov
            g += (0.82 * pr - g) * cov
            b += (0.88 * pr - b) * cov
        a = wa[k]
        out[y, x, 0] += r * a
        out[y, x, 1] += g * a
        out[y, x, 2] += b * a


# ============================================================================ per-frame railing

@njit(cache=True, fastmath=True)
def draw_capsules(img, C, col_dark, col_lit, spec_col, Lx, Ly):
    """Draw metal pipes (capsules, far->near order) with cylindrical shading + specular line, AA.
    C columns: x0 y0 r0 x1 y1 r1 spec(0..1)"""
    H, W = img.shape[0], img.shape[1]
    for i in range(C.shape[0]):
        ax, ay, ar, bx, by, br, spk = C[i, 0], C[i, 1], C[i, 2], C[i, 3], C[i, 4], C[i, 5], C[i, 6]
        rm = max(ar, br) + 2
        x0 = max(0, int(min(ax, bx) - rm))
        x1 = min(W, int(max(ax, bx) + rm + 1))
        y0 = max(0, int(min(ay, by) - rm))
        y1 = min(H, int(max(ay, by) + rm + 1))
        ex = bx - ax
        ey = by - ay
        L2 = ex * ex + ey * ey + 1e-6
        ln = math.sqrt(L2)
        # unit normal of the segment (pointing "up-left")
        nx = -ey / ln
        ny = ex / ln
        if ny > 0:
            nx = -nx
            ny = -ny
        for y in range(y0, y1):
            for x in range(x0, x1):
                px = x + 0.5 - ax
                py = y + 0.5 - ay
                t = (px * ex + py * ey) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = px - t * ex
                qy = py - t * ey
                rr = ar + (br - ar) * t
                d = math.sqrt(qx * qx + qy * qy)
                rv = max(rr, 0.5)
                cov = min(max(rv + 0.5 - d, 0.0), 1.0)
                if rr < 0.5:
                    cov *= rr / 0.5 * 0.9 + 0.1
                if cov <= 0.0:
                    continue
                sgn = (qx * nx + qy * ny) / rv     # -1..1 across the pipe (+ = toward the light side)
                sgn = min(max(sgn, -1.0), 1.0)
                k = sstep(-0.6, 0.7, sgn)
                r = col_dark[0] + (col_lit[0] - col_dark[0]) * k
                g = col_dark[1] + (col_lit[1] - col_dark[1]) * k
                b = col_dark[2] + (col_lit[2] - col_dark[2]) * k
                sp = math.exp(-((sgn - 0.55) / 0.18) ** 2) * spk
                r += spec_col[0] * sp
                g += spec_col[1] * sp
                b += spec_col[2] * sp
                img[y, x, 0] = img[y, x, 0] * (1 - cov) + r * cov
                img[y, x, 1] = img[y, x, 1] * (1 - cov) + g * cov
                img[y, x, 2] = img[y, x, 2] * (1 - cov) + b * cov
