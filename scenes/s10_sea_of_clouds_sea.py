"""Sea-of-clouds renderer for s10 (dome heightfield + column 'voxel-space' perspective march).

World: x right, y up, z forward.  The cloud sea is a heightfield h(x, z) = max of rounded DOMES at 4
levels (masses ~13 units wide, lobes ~2-7, bumps ~1-2, pebbles ~0.4) over a floor, tiled with period P,
plus large non-periodic swells.  Every child dome is embedded in its parent (its rim never floats above
the parent surface), so the field is continuous: crisp creases where lobes meet, no cliffs.

Because the camera has no roll/pitch (the horizon is placed with a lens shift), every image column is a
vertical plane: each column is marched front-to-back once, filling pixels upward (y-buffer); steep
faces are refined adaptively.  Near the camera the field is evaluated analytically (crisp creases,
slowly breathing lobes); farther away it is read from a pre-rasterised tile pyramid (fast, filtered ->
no shimmer).  Cast shadows (low sun -> long shadows into the valleys) and cavity occlusion are baked
into the same tile maps.
"""
import math
import numpy as np
import cv2
from numba import njit, prange


EPS = 0.15
_SE = math.sqrt(EPS)


def prof(x):
    """Dome profile 0..1 for x = d / r in [0, 1]: round crown, finite (steep) slope at the rim."""
    return (math.sqrt(max(1.0 - (1.0 - EPS) * x * x, 0.0)) - _SE) / (1.0 - _SE)


# ----------------------------------------------------------------------------------------------- layout

def build_domes(seed=5, P=96.0, spacing=8.0, cell=1.5, turrets=5):
    rng = np.random.default_rng(seed)
    D = []   # cx, cz, cy, r, ay, level, phase, tone

    def surf(dm, x, z):
        cx, cz, cy, r, ay = dm[:5]
        d = math.hypot(x - cx, z - cz) / r
        return cy + ay * r * prof(min(d, 1.0))

    def embed(parent, x, z, cy, r, ay, amax=1.1):
        # lower the child's base so its rim never floats above the parent's surface (no cliffs)
        top = cy + ay * r
        mn = cy
        for k in range(32):
            a = k * math.pi / 16
            for fr in (1.0, 0.85):
                mn = min(mn, surf(parent, x + math.cos(a) * r * fr, z + math.sin(a) * r * fr))
        ay2 = min((top - mn) / r, amax)
        return min(mn, top - ay2 * r), ay2

    n = int(P / spacing)
    masses = []
    for i in range(n):
        for j in range(n):
            if rng.random() < 0.18:
                continue
            cx = (i + 0.5 + rng.uniform(-0.4, 0.4)) * P / n
            cz = (j + 0.5 + rng.uniform(-0.4, 0.4)) * P / n
            r = rng.uniform(6.0, 9.5)
            top = rng.uniform(0.4, 1.3)
            m = [cx, cz, 0.0, r, top / r, 0, rng.uniform(0, 6.28), rng.uniform(-1, 1)]
            masses.append(m)
            D.append(m)
    for k in range(turrets):
        cx, cz = rng.uniform(0, P), rng.uniform(0, P)
        r = rng.uniform(3.5, 5.0)
        m = [cx, cz, 0.0, r, rng.uniform(0.5, 0.65), 0, rng.uniform(0, 6.28), rng.uniform(-1, 1)]
        masses.append(m)
        D.append(m)
    lobes = []
    for m in masses:
        k = int(rng.integers(40, 56) * (m[3] / 6.0) ** 2)
        for _ in range(k):
            a = rng.uniform(0, 2 * math.pi)
            rr = m[3] * 0.85 * math.sqrt(rng.uniform(0.0, 1.0))
            x, z = m[0] + math.cos(a) * rr, m[1] + math.sin(a) * rr
            r1 = (0.9 + 1.4 * rng.random() ** 1.5) * (1.0 - 0.3 * rr / m[3])
            ay1 = rng.uniform(0.3, 0.42)
            cy = surf(m, x, z) - ay1 * r1 * rng.uniform(0.4, 0.62)
            cy, ay1 = embed(m, x, z, cy, r1, ay1, 0.75)
            L = [x, z, cy, r1, ay1, 1, rng.uniform(0, 6.28), rng.uniform(-1, 1)]
            lobes.append(L)
            D.append(L)
    bumps = []
    for L in lobes:
        k = int(rng.integers(6, 10))
        for _ in range(k):
            a = rng.uniform(0, 2 * math.pi)
            rr = L[3] * 0.8 * math.sqrt(rng.uniform(0.0, 1.0))
            x, z = L[0] + math.cos(a) * rr, L[1] + math.sin(a) * rr
            r2 = L[3] * rng.uniform(0.25, 0.4)
            ay2 = rng.uniform(0.38, 0.5)
            cy = surf(L, x, z) - ay2 * r2 * rng.uniform(0.3, 0.52)
            cy, ay2 = embed(L, x, z, cy, r2, ay2, 1.2)
            B = [x, z, cy, r2, ay2, 2, rng.uniform(0, 6.28), rng.uniform(-1, 1)]
            D.append(B)
            bumps.append(B)
    for B in bumps:
        if rng.random() < 0.4:
            continue
        k = int(rng.integers(2, 5))
        for _ in range(k):
            a = rng.uniform(0, 2 * math.pi)
            rr = B[3] * math.sqrt(rng.uniform(0.3, 0.8))
            x, z = B[0] + math.cos(a) * rr, B[1] + math.sin(a) * rr
            r3 = B[3] * rng.uniform(0.3, 0.45)
            ay3 = rng.uniform(0.5, 0.62)
            cy = surf(B, x, z) - ay3 * r3 * rng.uniform(0.3, 0.5)
            cy, ay3 = embed(B, x, z, cy, r3, ay3, 1.2)
            D.append([x, z, cy, r3, ay3, 3, rng.uniform(0, 6.28), rng.uniform(-1, 1)])
    D = np.array(D, np.float64)
    # grid (CSR): every cell lists the domes whose disc touches it; sorted by level
    nc = int(round(P / cell))
    cells = [[] for _ in range(nc * nc)]
    order = np.argsort(D[:, 5], kind='stable')
    for idx in order:
        cx, cz, r = D[idx, 0], D[idx, 1], D[idx, 3]
        i0, i1 = int(math.floor((cx - r) / cell)), int(math.floor((cx + r) / cell))
        j0, j1 = int(math.floor((cz - r) / cell)), int(math.floor((cz + r) / cell))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                qx = min(max(cx, i * cell), (i + 1) * cell)
                qz = min(max(cz, j * cell), (j + 1) * cell)
                if (qx - cx) ** 2 + (qz - cz) ** 2 > r * r:
                    continue
                cells[(i % nc) * nc + (j % nc)].append((idx, -(i // nc), -(j // nc)))
    start = np.zeros(nc * nc + 1, np.int64)
    items, offs = [], []
    for k, lst in enumerate(cells):
        start[k + 1] = start[k] + len(lst)
        for (idx, wi, wj) in lst:
            items.append(idx)
            offs.append((wi, wj))
    return D, start, np.array(items, np.int64), np.array(offs, np.float64), nc


# ----------------------------------------------------------------------------------------------- field

@njit(cache=True, fastmath=True)
def _smoothstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(cache=True, fastmath=True)
def swell(x, z, A):
    """Large non-periodic swells (breaks the tile repetition). Returns h, dh/dx, dh/dz."""
    a1 = x * 0.031 + z * 0.019 + 1.3
    a2 = -x * 0.013 + z * 0.041 + 4.1
    a3 = x * 0.071 - z * 0.047 + 2.0
    h = A * (0.55 * math.sin(a1) + 0.45 * math.sin(a2) + 0.25 * math.sin(a3))
    gx = A * (0.55 * 0.031 * math.cos(a1) - 0.45 * 0.013 * math.cos(a2) + 0.25 * 0.071 * math.cos(a3))
    gz = A * (0.55 * 0.019 * math.cos(a1) + 0.45 * 0.041 * math.cos(a2) - 0.25 * 0.047 * math.cos(a3))
    return h, gx, gz


@njit(cache=True, fastmath=True)
def height_an(x, z, lodr, t, brk, D, start, items, offs, nc, P, cell, rmax):
    """Analytic tile field (no swell). Returns h, dh/dx, dh/dz, tone."""
    best = 0.05 * math.sin(x * 0.7) * math.sin(z * 0.6)
    gx = 0.05 * 0.7 * math.cos(x * 0.7) * math.sin(z * 0.6)
    gz = 0.05 * 0.6 * math.sin(x * 0.7) * math.cos(z * 0.6)
    tone = 0.0
    lx = x - math.floor(x / P) * P
    lz = z - math.floor(z / P) * P
    ci = int(lx / cell)
    cj = int(lz / cell)
    if ci >= nc:
        ci = nc - 1
    if cj >= nc:
        cj = nc - 1
    k = ci * nc + cj
    for q in range(start[k], start[k + 1]):
        i = items[q]
        r = D[i, 3]
        if r < lodr * 0.5:
            if rmax[int(D[i, 5])] < lodr * 0.5:
                break
            continue
        dx = lx - (D[i, 0] + offs[q, 0] * P)
        dz = lz - (D[i, 1] + offs[q, 1] * P)
        d2 = dx * dx + dz * dz
        r2 = r * r
        if d2 >= r2:
            continue
        ay = D[i, 4]
        xr2 = d2 / r2
        qq = math.sqrt(max(1.0 - 0.85 * xr2, 1e-6))
        br = 0.0
        if brk > 0.0 and D[i, 5] > 0.5:
            br = brk * 0.035 * r * ay * math.sin(t * 0.6 + D[i, 6])
        fade = _smoothstep(lodr * 0.5, lodr * 1.6, r)
        h = D[i, 2] + ay * r * (qq - 0.3873) / 0.6127 + br - (1.0 - fade) * ay * r * 1.05
        if h > best:
            best = h
            g = -ay * 0.85 / (qq * 0.6127 * r)
            gx = g * dx
            gz = g * dz
            tone = D[i, 7]
    return best, gx, gz, tone


@njit(cache=True, fastmath=True, parallel=True)
def raster_height(N, e, D, start, items, offs, nc, P, cell, rmax):
    H = np.zeros((N, N), np.float32)
    for i in prange(N):
        for j in range(N):
            H[i, j] = height_an((i + 0.5) * e, (j + 0.5) * e, e * 1.2, 0.0, 0.0, D, start, items, offs, nc, P, cell, rmax)[0]
    return H


@njit(cache=True, fastmath=True)
def _bil(M, x, z, e, N, ch):
    u = x / e - 0.5
    v = z / e - 0.5
    i0 = math.floor(u)
    j0 = math.floor(v)
    fu = u - i0
    fv = v - j0
    i0 = int(i0) % N
    j0 = int(j0) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    a = M[i0, j0, ch] * (1 - fu) + M[i1, j0, ch] * fu
    b = M[i0, j1, ch] * (1 - fu) + M[i1, j1, ch] * fu
    return a * (1 - fv) + b * fv


@njit(cache=True, fastmath=True, parallel=True)
def shadow_map(H, e, Lx, Lz, tanel, hmax):
    """Soft cast shadows on the tile for a sun toward (Lx, Lz) (unit, horizontal) at elevation atan(tanel)."""
    N = H.shape[0]
    Hm = np.empty((N, N, 1), np.float32)
    Hm[:, :, 0] = H
    S = np.zeros((N, N), np.float32)
    for i in prange(N):
        for j in range(N):
            h0 = H[i, j]
            x0 = (i + 0.5) * e
            z0 = (j + 0.5) * e
            sh = 0.0
            dist = e * 1.5
            while True:
                hr = h0 + tanel * dist + 0.015
                if hr > hmax:
                    break
                hs = _bil(Hm, x0 + Lx * dist, z0 + Lz * dist, e, N, 0)
                o = (hs - hr) / (0.03 + 0.06 * dist)
                if o > 0.0:
                    v = _smoothstep(0.0, 1.0, o)
                    if v > sh:
                        sh = v
                        if sh > 0.999:
                            break
                dist += e * (1.0 + dist * 0.08)
            S[i, j] = sh
    return S


def level_rmax(D):
    return np.array([D[D[:, 5] == l][:, 3].max() for l in range(4)], np.float64)


def build_maps(D, start, items, offs, nc, P, cell, N, sun_dir_xz, tanel, levels=4):
    """Rasterise the tile: returns list of (N, N, 3) float32 maps [height, shadow, cavity] per pyramid
    level (level l is pre-filtered for a footprint of ~2^l texels)."""
    e = P / N
    H = raster_height(N, e, D, start, items, offs, nc, P, cell, level_rmax(D))
    S = shadow_map(H, e, float(sun_dir_xz[0]), float(sun_dir_xz[1]), float(tanel), float(H.max()) + 0.05)

    def wblur(img, sig):
        if sig <= 0.05:
            return img
        pad = int(sig * 3) + 2
        p = np.pad(img, pad, mode='wrap')
        p = cv2.GaussianBlur(p, (0, 0), sig)
        return p[pad:-pad, pad:-pad]
    # cavity: how far below its neighbourhood a point is (valleys, recesses between lobes)
    cav = np.clip((wblur(H, 1.0 / e) - H) / 0.45, 0, 1) * 0.6 + np.clip((wblur(H, 0.35 / e) - H) / 0.2, 0, 1) * 0.4
    S = wblur(S, 0.6)
    def grads(h):
        p = np.pad(h, 1, mode='wrap')
        gx = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * e)      # axis 0 = x
        gz = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * e)      # axis 1 = z
        return gx, gz
    maps = []
    for l in range(levels):
        if l == 0:
            h, sh, cv = H, S, cav
        else:
            sg = 0.5 * 2 ** l
            h, sh, cv = wblur(H, sg), wblur(S, sg), wblur(cav, sg)
        gx, gz = grads(h)
        maps.append(np.ascontiguousarray(np.dstack([h, sh, cv, gx, gz]).astype(np.float32)))
    return maps, e


@njit(cache=True, fastmath=True)
def _samp3(M0, M1, M2, M3, x, z, e, N, lv):
    """Trilinear sample of (h, shadow, cavity) + central-difference gradient of h."""
    l0 = int(lv)
    w = lv - l0
    if l0 >= 3:
        l0 = 3
        w = 0.0
    r0 = 0.0
    r1 = 0.0
    r2 = 0.0
    r3 = 0.0
    r4 = 0.0
    for k in range(2):
        l = l0 + k
        wk = (1.0 - w) if k == 0 else w
        if wk <= 0.0 or l > 3:
            continue
        if l == 0:
            M = M0
        elif l == 1:
            M = M1
        elif l == 2:
            M = M2
        else:
            M = M3
        u = x / e - 0.5
        v = z / e - 0.5
        i0 = math.floor(u)
        j0 = math.floor(v)
        fu = u - i0
        fv = v - j0
        i0 = int(i0) % N
        j0 = int(j0) % N
        i1 = (i0 + 1) % N
        j1 = (j0 + 1) % N
        w00 = (1 - fu) * (1 - fv) * wk
        w10 = fu * (1 - fv) * wk
        w01 = (1 - fu) * fv * wk
        w11 = fu * fv * wk
        r0 += M[i0, j0, 0] * w00 + M[i1, j0, 0] * w10 + M[i0, j1, 0] * w01 + M[i1, j1, 0] * w11
        r1 += M[i0, j0, 1] * w00 + M[i1, j0, 1] * w10 + M[i0, j1, 1] * w01 + M[i1, j1, 1] * w11
        r2 += M[i0, j0, 2] * w00 + M[i1, j0, 2] * w10 + M[i0, j1, 2] * w01 + M[i1, j1, 2] * w11
        r3 += M[i0, j0, 3] * w00 + M[i1, j0, 3] * w10 + M[i0, j1, 3] * w01 + M[i1, j1, 3] * w11
        r4 += M[i0, j0, 4] * w00 + M[i1, j0, 4] * w10 + M[i0, j1, 4] * w01 + M[i1, j1, 4] * w11
    return r0, r1, r2, r3, r4


# ----------------------------------------------------------------------------------------------- render

@njit(cache=True, fastmath=True, parallel=True)
def render_sea(Wi, Hi, f, hy, camx, camy, camz, yaw, t, D, start, items, offs, nc, P, cell, M0, M1, M2, M3, e,
               swell_a, s_an, s0, smax, kstep, pal, sun_az, light_az, sun_el, paint_el, fogd, lit_bias, out_col, out_dep,
               out_lit, rmax, hmax, cshade, rim_k, mist_h0, mist_h1, mist_k):
    """Column march. pal: (16, 3) colour table. Writes out_col (Hi, Wi, 3), out_dep (Hi, Wi) (0 = sky),
    out_lit (Hi, Wi) (0..1 direct sunlight on the surface)."""
    N = M0.shape[0]
    fx_, fz_ = math.sin(yaw), math.cos(yaw)
    rx_, rz_ = math.cos(yaw), -math.sin(yaw)
    Lx = math.sin(light_az) * math.cos(paint_el)
    Lz = math.cos(light_az) * math.cos(paint_el)
    Ly = math.sin(paint_el)
    Sx = math.sin(sun_az) * math.cos(sun_el)
    Sz = math.cos(sun_az) * math.cos(sun_el)
    Sy = math.sin(sun_el)
    for c in prange(Wi):
        u = (c + 0.5 - Wi * 0.5) / f
        dx = fx_ + u * rx_
        dz = fz_ + u * rz_
        dl = math.sqrt(dx * dx + dz * dz)
        az = (dx * math.sin(sun_az) + dz * math.cos(sun_az)) / dl
        wsun = math.exp((az - 1.0) / 0.03) * 0.6 + math.exp((az - 1.0) / 0.25) * 0.4
        hzr = pal[9, 0] + (pal[10, 0] - pal[9, 0]) * wsun
        hzg = pal[9, 1] + (pal[10, 1] - pal[9, 1]) * wsun
        hzb = pal[9, 2] + (pal[10, 2] - pal[9, 2]) * wsun
        ybuf = float(Hi)
        s = s0
        pr = 0.0
        pg = 0.0
        pb = 0.0
        pl = 0.0
        prev_vis = False
        s_prev = s0 * 0.9
        fine_left = 0
        fine_ds = 0.0
        while s < smax:
            # skip ahead while nothing at this distance can rise above the filled part of the column
            if fine_left == 0 and ybuf > hy + 1.0:
                s_vis = f * (camy - hmax) / (ybuf - hy)
                if s_vis > s:
                    s = s_vis
                    prev_vis = False
            ds = kstep * s
            px = camx + s * dx
            pz = camz + s * dz
            fp = max(ds, s / f)
            lv = math.log(max(fp * 1.5 / e, 1.0)) / math.log(2.0)
            m0, m1, m2, m3, m4 = _samp3(M0, M1, M2, M3, px, pz, e, N, lv)
            wb = _smoothstep(s_an * 0.8, s_an, s)
            if wb < 1.0:
                ha, gxa, gza, tone = height_an(px, pz, max(ds * 1.3, 2.2 * s / f), t,
                                               1.0 - _smoothstep(s_an * 0.4, s_an * 0.8, s),
                                               D, start, items, offs, nc, P, cell, rmax)
                h = ha + (m0 - ha) * wb
                gx = gxa + (m3 - gxa) * wb
                gz = gza + (m4 - gza) * wb
                tone *= 1.0 - wb
            else:
                h = m0
                gx = m3
                gz = m4
                tone = 0.0
            sw, sgx, sgz = swell(px, pz, swell_a)
            h += sw
            gx += sgx
            gz += sgz
            yp = hy + f * (camy - h) / s
            if yp < ybuf - 1.2 and fine_left == 0 and prev_vis:
                n = int((ybuf - yp) / 0.8) + 1
                if n > 48:
                    n = 48
                fine_ds = (s - s_prev) / n
                s = s_prev + fine_ds
                fine_left = n
                continue
            if yp < ybuf:
                shd = m1
                cav = m2
                nx, ny, nz = -gx, 1.0, -gz
                nl = math.sqrt(nx * nx + ny * ny + nz * nz)
                nx /= nl
                ny /= nl
                nz /= nl
                ndl = nx * Lx + ny * Ly + nz * Lz + lit_bias
                vl = math.sqrt(dx * dx + dz * dz + ((h - camy) / s) ** 2)
                vx, vy, vz = dx / vl, (h - camy) / s / vl, dz / vl
                ph = vx * Sx + vy * Sy + vz * Sz
                hv = _smoothstep(-0.2, 2.4, h - sw * 0.5)
                # --- shadow side: deep violet in the valleys -> lavender higher up; sky fill on up-faces
                wsh = _smoothstep(0.0, 1.0, hv * 1.1 - cav * 0.6)
                r_ = pal[0, 0] + (pal[1, 0] - pal[0, 0]) * wsh
                g_ = pal[0, 1] + (pal[1, 1] - pal[0, 1]) * wsh
                b_ = pal[0, 2] + (pal[1, 2] - pal[0, 2]) * wsh
                up = _smoothstep(0.55, 0.95, ny) * 0.35 * wsh
                r_ += (pal[2, 0] - r_) * up
                g_ += (pal[2, 1] - g_) * up
                b_ += (pal[2, 2] - b_) * up
                # --- direct light: N.L (painted steps) x cast shadow
                # drifting shadows of unseen high clouds: big dappled lit / unlit patches
                qx = px + t * 0.5
                cs = (math.sin(qx * 0.045 + pz * 0.021 + 0.7) + math.sin(-qx * 0.019 + pz * 0.052 + 2.3)
                      + 0.6 * math.sin(qx * 0.11 + pz * 0.083 + 4.1) + 0.35 * math.sin(qx * 0.23 - pz * 0.17 + 1.1))
                patch = _smoothstep(0.35, 1.1, cs) * cshade
                lit = (1.0 - shd) * (1.0 - _smoothstep(0.15, 0.7, cav)) * (1.0 - patch) * _smoothstep(-0.2, 1.0, h - sw)
                th = _smoothstep(-0.16, -0.02, ndl) * 0.7 * (0.35 + 0.65 * lit)
                tl = _smoothstep(-0.02, 0.06, ndl) * lit
                tc = _smoothstep(0.34, 0.44, ndl) * lit * (0.3 + 0.7 * hv)
                r_ += (pal[3, 0] - r_) * th
                g_ += (pal[3, 1] - g_) * th
                b_ += (pal[3, 2] - b_) * th
                wl = _smoothstep(0.02, 0.4, ndl) * (0.55 + 0.45 * hv)
                lr = pal[4, 0] + (pal[5, 0] - pal[4, 0]) * wl
                lg = pal[4, 1] + (pal[5, 1] - pal[4, 1]) * wl
                lb = pal[4, 2] + (pal[5, 2] - pal[4, 2]) * wl
                r_ += (lr - r_) * tl
                g_ += (lg - g_) * tl
                b_ += (lb - b_) * tl
                r_ += (pal[6, 0] - r_) * tc
                g_ += (pal[6, 1] - g_) * tc
                b_ += (pal[6, 2] - b_) * tc
                # saturated warm line where light meets shadow (terminator and cast-shadow edge)
                band = math.exp(-((ndl - 0.03) / 0.045) ** 2) * 0.4 * lit
                band = max(band, 4.0 * shd * (1.0 - shd) * 0.45 * _smoothstep(-0.02, 0.1, ndl))
                r_ += (pal[7, 0] - r_) * band
                g_ += (pal[7, 1] - g_) * band
                b_ += (pal[7, 2] - b_) * band
                tv = 1.0 + 0.02 * tone
                r_ *= tv
                g_ *= tv
                b_ *= tv
                # --- forward scattering toward the sun (the glowing path across the sea)
                fs = math.exp((ph - 1.0) / 0.006) * 0.55 + math.exp((ph - 1.0) / 0.05) * 0.22
                fs *= (0.3 + 0.7 * _smoothstep(0.2, 0.9, ny)) * (0.4 + 0.6 * lit)
                r_ += pal[8, 0] * fs
                g_ += pal[8, 1] * fs
                b_ += pal[8, 2] * fs
                # --- silver lining: surfaces turning away at grazing angles, on the sun side, glow
                ndv = -(nx * vx + ny * vy + nz * vz)
                sunf = nx * math.sin(sun_az) + nz * math.cos(sun_az) + 0.25 * ny
                rimv = _smoothstep(0.5, 0.12, ndv) * _smoothstep(0.0, 0.5, sunf) * rim_k * (0.4 + 0.6 * (1.0 - patch))
                r_ += (pal[11, 0] - r_) * min(rimv, 1.0)
                g_ += (pal[11, 1] - g_) * min(rimv, 1.0)
                b_ += (pal[11, 2] - b_) * min(rimv, 1.0)
                # --- low mist filling the valleys (lumps rise out of it)
                mist = (1.0 - _smoothstep(mist_h0, mist_h1, h - sw)) * mist_k
                r_ += (pal[12, 0] - r_) * mist
                g_ += (pal[12, 1] - g_) * mist
                b_ += (pal[12, 2] - b_) * mist
                # --- aerial perspective
                fg = 1.0 - math.exp(-(s / fogd) ** 0.9)
                r_ += (hzr - r_) * fg
                g_ += (hzg - g_) * fg
                b_ += (hzb - b_) * fg
                litv = tl * (1.0 - fg)
                # ---- fill rows (pixel centres in [yp, ybuf)), interpolate from the previous sample
                r0 = int(math.ceil(yp - 0.5))
                if r0 < 0:
                    r0 = 0
                r1 = int(math.ceil(ybuf - 0.5))
                if r1 > Hi:
                    r1 = Hi
                span = ybuf - yp
                for row in range(r0, r1):
                    w = 0.0
                    if prev_vis and span > 1e-6:
                        w = (row + 0.5 - yp) / span
                        if w < 0.0:
                            w = 0.0
                        elif w > 1.0:
                            w = 1.0
                    out_col[row, c, 0] = r_ + (pr - r_) * w
                    out_col[row, c, 1] = g_ + (pg - g_) * w
                    out_col[row, c, 2] = b_ + (pb - b_) * w
                    out_lit[row, c] = litv + (pl - litv) * w
                    out_dep[row, c] = s
                ybuf = yp
                pr, pg, pb, pl = r_, g_, b_, litv
                prev_vis = True
                if ybuf <= 0.0:
                    break
            else:
                prev_vis = False
            s_prev = s
            if fine_left > 0:
                fine_left -= 1
                s += fine_ds
            else:
                s += ds
        # far field down to the horizon
        rh = int(math.ceil(hy - 0.5))
        r1 = int(math.ceil(ybuf - 0.5))
        if r1 > Hi:
            r1 = Hi
        for row in range(max(rh, 0), r1):
            out_col[row, c, 0] = hzr
            out_col[row, c, 1] = hzg
            out_col[row, c, 2] = hzb
            out_dep[row, c] = smax


@njit(cache=True, fastmath=True, parallel=True)
def edge_maps(dep, wr, wg, ws, near):
    """Silhouette maps from the depth buffer (0 = sky), one O(H) scan per column.  For every sea pixel:
       [0] rim  - crisp line just below a crest silhouette (surface above is much farther / sky)
       [1] glow - wider translucent glow below the same crests
       [2] shadow - contact shadow on the far surface just above a nearer crest (lobe overlap)
    widths wr/wg/ws in px (scaled by near/d for close crests)."""
    Hi, Wi = dep.shape
    out = np.zeros((Hi, Wi, 3), np.float32)
    for c in prange(Wi):
        rim = 0.0
        glow = 0.0
        pd = 0.0
        for row in range(Hi):
            d = dep[row, c]
            if d <= 0.0:
                pd = 0.0
                rim = 0.0
                glow = 0.0
                continue
            sc = min(max(near / d, 0.6), 2.5)
            rim *= math.exp(-1.0 / (wr * sc))
            glow *= math.exp(-1.0 / (wg * sc))
            dj = pd if pd > 0.0 else 1e5
            if row > 0:
                ej = _smoothstep(0.03, 0.25, math.log(dj / d))
                rim = max(rim, ej)
                glow = max(glow, ej)
            out[row, c, 0] = rim
            out[row, c, 1] = glow
            pd = d
        sh = 0.0
        pd = 0.0
        for row in range(Hi - 1, -1, -1):
            d = dep[row, c]
            if d <= 0.0:
                break
            sc = min(max(near / d, 0.6), 2.5)
            sh *= math.exp(-1.0 / (ws * sc))
            if pd > 0.0:
                jump = math.log(d / pd)
                ej = _smoothstep(0.03, 0.2, jump) * _smoothstep(1.4, 0.5, jump)
                sh = max(sh, ej)
            out[row, c, 2] = sh
            pd = d
    return out


@njit(cache=True, fastmath=True, parallel=True)
def apply_edges(col, dep, lit, M, sun_x, rimcol, glowcol, shcol, rim_amt, glow_amt, sh_amt):
    Hi, Wi = dep.shape
    for row in prange(Hi):
        for c in range(Wi):
            d = dep[row, c]
            if d <= 0.0:
                continue
            dxs = (c - sun_x) / Wi
            prox = 0.3 + 0.7 * math.exp(-(dxs * dxs) / 0.05) + 0.6 * math.exp(-(dxs * dxs) / 0.004)
            far = 1.0 / (1.0 + d / 150.0)
            lt = 0.3 + 0.7 * lit[row, c]
            a = min(M[row, c, 0] * rim_amt * prox * far * lt, 1.0)
            g = min(M[row, c, 1] * glow_amt * prox * far * lt, 1.0)
            s = M[row, c, 2] * sh_amt * far
            for ch in range(3):
                v = col[row, c, ch]
                v += (shcol[ch] - v) * s
                v += (glowcol[ch] - v) * g
                v += (rimcol[ch] - v) * a
                col[row, c, ch] = v
