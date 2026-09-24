"""Numba ray tracer for the s09_classroom interior.

World: metres, y up. Room box x in [0, LX] (x = LX: front wall with the blackboard), z in [0, LZ]
(z = 0: window wall), y in [0, HC]. The sun is outside (L.z < 0) and shines in through the windows.
Geometry = the room's six planes (window aperture cut out of z = 0) + oriented boxes (yaw only),
grouped per object for a cheap AABB pre-test. Outside the window: boxes (balcony) then an RGBA tree
plane at z = TREE_Z, then a sky plate in gnomonic (u, v) = (dx, dy) / -dz.
Sun visibility through the windows is analytic (soft-edged aperture / frame bars / pillars, penumbra
grows with distance) x curtain transmission mask x box shadow ray.
"""
import math
import numpy as np
from numba import njit, prange

LX, LZ, HC = 9.0, 7.2, 3.0
AX0, AX1, AY0, AY1 = 0.5, 8.7, 0.92, 2.72
TREE_Z = -10.0

# materials
M_FLOOR, M_CEIL, M_WALL, M_WINWALL = 0, 1, 2, 3
M_PILLAR, M_FRAME, M_SILL, M_VALANCE = 10, 11, 12, 13
M_DESKTOP, M_DESKMETAL, M_CHAIRWOOD, M_CHAIRMETAL, M_BAG, M_BOOK = 20, 21, 22, 23, 24, 25
M_BOARD, M_BOARDFRAME, M_TRAY, M_PLATFORM, M_LECTERN = 30, 31, 32, 33, 34
M_LIGHT, M_CLOCK, M_PAPER, M_SPEAKER, M_TV = 40, 41, 42, 43, 44
M_PARAPET, M_RAIL, M_BALC = 50, 51, 52


@njit(inline='always', fastmath=True)
def _ss(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3.0 - 2.0 * t)


@njit(inline='always', fastmath=True)
def _clamp(x, a, b):
    return a if x < a else (b if x > b else x)


@njit(fastmath=True, cache=True)
def samp(tex, fx, fy, c):
    """bilinear sample channel c of tex (h, w, n) at pixel coords (clamped)."""
    h, w = tex.shape[0], tex.shape[1]
    fx = _clamp(fx - 0.5, 0.0, w - 1.001)
    fy = _clamp(fy - 0.5, 0.0, h - 1.001)
    x0 = int(fx)
    y0 = int(fy)
    ax = fx - x0
    ay = fy - y0
    return ((tex[y0, x0, c] * (1 - ax) + tex[y0, x0 + 1, c] * ax) * (1 - ay) +
            (tex[y0 + 1, x0, c] * (1 - ax) + tex[y0 + 1, x0 + 1, c] * ax) * ay)


@njit(fastmath=True, cache=True)
def samp2(tex, fx, fy):
    h, w = tex.shape[0], tex.shape[1]
    fx = _clamp(fx - 0.5, 0.0, w - 1.001)
    fy = _clamp(fy - 0.5, 0.0, h - 1.001)
    x0 = int(fx)
    y0 = int(fy)
    ax = fx - x0
    ay = fy - y0
    return ((tex[y0, x0] * (1 - ax) + tex[y0, x0 + 1] * ax) * (1 - ay) +
            (tex[y0 + 1, x0] * (1 - ax) + tex[y0 + 1, x0 + 1] * ax) * ay)


# ----------------------------------------------------------------------------- intersection

@njit(fastmath=True, cache=True)
def ray_box(ox, oy, oz, dx, dy, dz, B, i, tmax):
    """-> t (inf if miss), face axis (0..2), face sign (+1/-1) in local coords."""
    c, s = B[i, 6], B[i, 7]
    rx, ry, rz = ox - B[i, 0], oy - B[i, 1], oz - B[i, 2]
    lox = c * rx - s * rz
    loz = s * rx + c * rz
    loy = ry
    ldx = c * dx - s * dz
    ldz = s * dx + c * dz
    ldy = dy
    t0, t1 = 1e-4, tmax
    ax = -1
    sg = 1.0
    for a in range(3):
        if a == 0:
            o, d, h = lox, ldx, B[i, 3]
        elif a == 1:
            o, d, h = loy, ldy, B[i, 4]
        else:
            o, d, h = loz, ldz, B[i, 5]
        if abs(d) < 1e-9:
            if o < -h or o > h:
                return np.inf, 0, 1.0
            continue
        inv = 1.0 / d
        ta = (-h - o) * inv
        tb = (h - o) * inv
        sa = -1.0
        if ta > tb:
            ta, tb = tb, ta
            sa = 1.0
        if ta > t0:
            t0 = ta
            ax = a
            sg = sa
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return np.inf, 0, 1.0
    if ax < 0:
        return np.inf, 0, 1.0          # origin inside the box
    return t0, ax, sg


@njit(fastmath=True, cache=True)
def aabb_hit(ox, oy, oz, ix, iy, iz, G, g, tmax):
    t0, t1 = 0.0, tmax
    for a in range(3):
        if a == 0:
            o, iv, lo, hi = ox, ix, G[g, 0], G[g, 3]
        elif a == 1:
            o, iv, lo, hi = oy, iy, G[g, 1], G[g, 4]
        else:
            o, iv, lo, hi = oz, iz, G[g, 2], G[g, 5]
        ta = (lo - o) * iv
        tb = (hi - o) * iv
        if ta > tb:
            ta, tb = tb, ta
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return False
    return True


@njit(fastmath=True, cache=True)
def trace_boxes(ox, oy, oz, dx, dy, dz, B, G, tmax, shadow):
    ix = 1.0 / dx if abs(dx) > 1e-12 else 1e12
    iy = 1.0 / dy if abs(dy) > 1e-12 else 1e12
    iz = 1.0 / dz if abs(dz) > 1e-12 else 1e12
    best = tmax
    bi = -1
    bax = 0
    bsg = 1.0
    for g in range(G.shape[0]):
        if not aabb_hit(ox, oy, oz, ix, iy, iz, G, g, best):
            continue
        for i in range(int(G[g, 6]), int(G[g, 7])):
            if shadow and B[i, 10] > 0.5:
                continue
            t, ax, sg = ray_box(ox, oy, oz, dx, dy, dz, B, i, best)
            if t < best:
                best = t
                bi = i
                bax = ax
                bsg = sg
                if shadow:
                    return best, bi, bax, bsg
    return best, bi, bax, bsg


@njit(fastmath=True, cache=True)
def room_exit(ox, oy, oz, dx, dy, dz):
    tb = 1e9
    pl = -1
    if dy < -1e-9:
        t = -oy / dy
        if t < tb:
            tb, pl = t, 0
    elif dy > 1e-9:
        t = (HC - oy) / dy
        if t < tb:
            tb, pl = t, 1
    if dx < -1e-9:
        t = -ox / dx
        if t < tb:
            tb, pl = t, 2
    elif dx > 1e-9:
        t = (LX - ox) / dx
        if t < tb:
            tb, pl = t, 3
    if dz < -1e-9:
        t = -oz / dz
        if t < tb:
            tb, pl = t, 4
    elif dz > 1e-9:
        t = (LZ - oz) / dz
        if t < tb:
            tb, pl = t, 5
    return tb, pl


# ----------------------------------------------------------------------------- sun visibility

@njit(fastmath=True, cache=True)
def aperture(xw, yw, pen, VB, HB):
    """Soft transmission of the window wall at (xw, yw) on z = 0 (1 = open glass)."""
    e = min(xw - AX0, AX1 - xw, yw - AY0, AY1 - yw)
    a = _ss(-pen, pen, e)
    if a <= 0.0:
        return 0.0
    for k in range(VB.shape[0]):
        d = abs(xw - VB[k, 0]) - VB[k, 1]
        a *= _ss(-pen, pen, d)
        if a <= 0.0:
            return 0.0
    for k in range(HB.shape[0]):
        d = abs(yw - HB[k, 0]) - HB[k, 1]
        a *= _ss(-pen, pen, d)
    return a


@njit(fastmath=True, cache=True)
def curtain_T(xw, yw, CM, cm_ppm, cm_y0):
    """Transmission of the curtains (mask rasterised on the window plane by the Python side)."""
    v = samp2(CM, xw * cm_ppm, (cm_y0 - yw) * cm_ppm)
    return 1.0 - 0.82 * v


@njit(fastmath=True, cache=True)
def sun_vis(px, py, pz, nx, ny, nz, Lx, Ly, Lz, B, G, VB, HB, CM, cm_ppm, cm_y0, boxes):
    if pz < -0.02:
        return 1.0
    tw = -pz / Lz
    xw = px + Lx * tw
    yw = py + Ly * tw
    pen = 0.004 + tw * 0.012
    a = aperture(xw, yw, pen, VB, HB)
    if a <= 0.0:
        return 0.0
    a *= curtain_T(xw, yw, CM, cm_ppm, cm_y0)
    if boxes:
        ox, oy, oz = px + nx * 2e-3, py + ny * 2e-3, pz + nz * 2e-3
        t, bi, ax, sg = trace_boxes(ox, oy, oz, Lx, Ly, Lz, B, G, tw, True)
        if bi >= 0:
            return 0.0
    return a


# ----------------------------------------------------------------------------- environment

@njit(fastmath=True, cache=True)
def env(ox, oy, oz, dx, dy, dz, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1):
    if dz > -1e-3:
        dz = -1e-3
    u = dx / -dz
    v = dy / -dz
    fx = (u - su0) * sky_ppu
    fy = (sv1 - v) * sky_ppu
    r = samp(SKY, fx, fy, 0)
    g = samp(SKY, fx, fy, 1)
    b = samp(SKY, fx, fy, 2)
    t = (TREE_Z - oz) / dz
    qx = ox + dx * t
    qy = oy + dy * t
    tx = (qx - tx0) * tppm
    ty = (ty1 - qy) * tppm
    a = samp(TREE, tx, ty, 3)
    if a > 0.0:
        # slight aerial perspective on the trees
        tr = samp(TREE, tx, ty, 0) * 0.9 + 0.1 * 0.62
        tg = samp(TREE, tx, ty, 1) * 0.9 + 0.1 * 0.78
        tb = samp(TREE, tx, ty, 2) * 0.9 + 0.1 * 0.92
        r = r * (1 - a) + tr * a
        g = g * (1 - a) + tg * a
        b = b * (1 - a) + tb * a
    return r, g, b


# ----------------------------------------------------------------------------- materials

@njit(fastmath=True, cache=True)
def hash1(i):
    x = math.sin(i * 12.9898 + 78.233) * 43758.5453
    return x - math.floor(x)


@njit(fastmath=True, cache=True)
def line_f(x, period, width, foot):
    """Filtered periodic thin line: 1 on the line, fades to its average when the footprint is large."""
    f = x / period
    f = f - math.floor(f)
    d = min(f, 1 - f) * period
    cov = _clamp((width * 0.5 - d) / max(foot, 1e-5) + 0.5, 0.0, 1.0)
    avg = width / period
    k = _ss(0.25 * period, 1.2 * period, foot)
    return cov * (1 - k) + avg * k


@njit(fastmath=True, cache=True)
def hash2(i, j):
    x = math.sin(i * 127.1 + j * 311.7) * 43758.5453
    return x - math.floor(x)


@njit(fastmath=True, cache=True)
def vnoise(x, y):
    """smooth value noise in [0, 1]"""
    ix = math.floor(x)
    iy = math.floor(y)
    fx = x - ix
    fy = y - iy
    fx = fx * fx * (3 - 2 * fx)
    fy = fy * fy * (3 - 2 * fy)
    a = hash2(ix, iy)
    b = hash2(ix + 1, iy)
    c = hash2(ix, iy + 1)
    d = hash2(ix + 1, iy + 1)
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


@njit(fastmath=True, cache=True)
def wood(u, w, var, foot, hu, hw, warm):
    """Painted plywood veneer: u runs along the grain, w across (local metres, centred), hu/hw half
    extents. Cathedral arches + straight grain streaks + dark ring lines + broad mottling; detail fades
    to its average with the pixel footprint.  -> r, g, b (albedo)"""
    sd = var * 97.0
    if foot > 0.04:
        # far away / reflected: the averaged veneer tone only
        if warm < 0.5:
            return 0.79, 0.59, 0.37
        return 0.83, 0.62, 0.39
    fk = _clamp(1.0 - foot / 0.012, 0.0, 1.0)          # fine lines
    fk2 = _clamp(1.0 - foot / 0.04, 0.0, 1.0)          # ring bands
    n1 = vnoise(u * 3.0 + sd, w * 8.0)
    wv = w + 0.012 * math.sin(u * 4.1 + sd) + 0.008 * (n1 - 0.5) + 0.003 * math.sin(u * 17.0 + sd * 2.0)
    c0 = (hash1(sd + 1.0) - 0.5) * hw * 2.6
    u0 = (hash1(sd + 2.0) - 0.5) * hu * 1.2
    q = (u - u0) / hu
    ring = abs(wv - c0) * 30.0 + 0.8 * q * q + 1.6 * vnoise(u * 1.2 - sd, w * 2.5) + 0.02 * u * 30.0
    f = ring - math.floor(ring)
    late = _ss(0.45, 0.92, f) * (1.0 - _ss(0.93, 1.0, f))
    line = math.exp(-((1.0 - f) * 16.0) ** 2) + math.exp(-(f * 22.0) ** 2) * 0.6
    streak = vnoise(u * 2.0 + sd * 3.0, w * 150.0)
    streak = _ss(0.55, 0.9, streak)
    mott = vnoise(u * 4.0 + sd, w * 5.0 - sd)
    band = vnoise(u * 0.8 + sd, w * 55.0 + sd) * fk2 + 0.5 * (1.0 - fk2)     # long straight streak bands
    # colours: warm honey early wood, amber late wood, red-brown lines
    br, bg, bb = 0.90, 0.69, 0.44
    lr_, lg_, lb_ = 0.76, 0.52, 0.30
    kr, kg, kb = 0.52, 0.31, 0.17
    if warm < 0.5:
        br, bg, bb = 0.86, 0.66, 0.43
        lr_, lg_, lb_ = 0.70, 0.47, 0.28
        kr, kg, kb = 0.46, 0.27, 0.15
    ml = late * 0.75 * fk2 + 0.3 * (1.0 - fk2)
    r = br + (lr_ - br) * ml
    g = bg + (lg_ - bg) * ml
    b = bb + (lb_ - bb) * ml
    ln = (line * 0.5 * fk + 0.06 * (1.0 - fk)) + streak * 0.3 * fk
    r = r + (kr - r) * ln
    g = g + (kg - g) * ln
    b = b + (kb - b) * ln
    m = (0.9 + 0.2 * mott) * (0.9 + 0.2 * band)
    return r * m, g * m, b * m


@njit(fastmath=True, cache=True)
def wear_edge(e, along, var, foot):
    """0..1 worn (sanded-pale) band along an edge: e = distance to the edge, along = coordinate along it"""
    n = vnoise(along * 18.0 + var * 50.0, var * 7.0)
    n2 = vnoise(along * 70.0 - var * 30.0, 3.0)
    wdt = 0.004 + 0.016 * n * n + 0.004 * n2
    return _clamp((wdt - e) / max(foot, 1e-4) + 0.5, 0.0, 1.0) * (0.55 + 0.45 * n2)


@njit(fastmath=True, cache=True)
def scratches(u, w, var, foot):
    s = 0.0
    if foot > 0.0025:
        return 0.0
    for k in range(7):
        h1 = hash1(var * 13.0 + k * 3.1)
        h2 = hash1(var * 29.0 + k * 5.7)
        h3 = hash1(var * 41.0 + k * 1.3)
        a = (h3 - 0.5) * 1.2
        ca, sa = math.cos(a), math.sin(a)
        du, dw = u - (h1 - 0.5) * 0.4, w - (h2 - 0.5) * 0.3
        along = du * ca + dw * sa
        perp = abs(-du * sa + dw * ca)
        ln = 0.03 + 0.12 * hash1(k + var * 3.0)
        if abs(along) < ln:
            wd = 0.0007
            s = max(s, _clamp((wd - perp) / max(foot, 1e-4) + 0.5, 0.0, 1.0) * (1 - abs(along) / ln))
    return s


@njit(fastmath=True, cache=True)
def material(mat, var, px, py, pz, lx, ly, lz, ax, sg, foot, BT, bt_ppm, FAO, fao_ppm, hx, hz, hy):
    """-> albedo r, g, b, gloss (reflection weight at normal incidence), spec (sun glint), emissive"""
    r, g, b = 0.8, 0.8, 0.8
    gloss = 0.0
    spec = 0.0
    em = 0.0
    if mat == M_FLOOR:
        # narrow boards running along x with staggered end joints
        pw = 0.095
        j = math.floor(pz / pw)
        h = hash1(j)
        seg = 1.83
        off = h * seg
        k = math.floor((px + off) / seg)
        h2 = hash1(j * 31.0 + k * 7.0)
        v = 0.88 + 0.16 * h2 + 0.05 * math.sin(px * 3.1 + h * 20.0) * math.sin(px * 0.7 + h2 * 9.0)
        r, g, b = 0.56 * v, 0.36 * v, 0.215 * v
        seam = line_f(pz, pw, 0.004, foot)
        seam2 = line_f(px + off, seg, 0.004, foot)
        s = max(seam, seam2 * 0.8)
        r *= 1 - 0.45 * s
        g *= 1 - 0.47 * s
        b *= 1 - 0.47 * s
        gloss = 0.5
        spec = 1.0
    elif mat == M_CEIL:
        r, g, b = 0.88, 0.87, 0.86
        ti = math.floor(px / 0.9) * 17.0 + math.floor(pz / 0.6) * 5.0
        tv = 0.965 + 0.05 * hash1(ti)
        # soft pillowing inside each tile + fine fissure texture (band-limited)
        fx_ = px / 0.9 - math.floor(px / 0.9)
        fz_ = pz / 0.6 - math.floor(pz / 0.6)
        pil = 1.0 - 0.05 * ((2 * fx_ - 1) ** 4 + (2 * fz_ - 1) ** 4)
        fk = _clamp(1.0 - foot / 0.012, 0.0, 1.0)
        fis = 0.025 * fk * (math.sin(px * 211.0 + math.sin(pz * 97.0) * 2.0) * math.sin(pz * 173.0 + px * 31.0))
        v = tv * pil + fis
        r, g, b = r * v, g * v, b * v
        s = max(line_f(px, 0.9, 0.014, foot), line_f(pz, 0.6, 0.014, foot))
        r *= 1 - 0.22 * s
        g *= 1 - 0.22 * s
        b *= 1 - 0.19 * s
    elif mat == M_WALL or mat == M_WINWALL or mat == M_PILLAR:
        r, g, b = 0.93, 0.885, 0.79
        if mat == M_WINWALL and py < AY0:
            r, g, b = 0.86, 0.83, 0.76
        if py < 0.09 and mat != M_PILLAR:
            r, g, b = 0.32, 0.27, 0.25
            gloss = 0.1
        elif py < 0.09:
            r, g, b = 0.32, 0.27, 0.25
    elif mat == M_FRAME:
        r, g, b = 0.74, 0.76, 0.78
        gloss = 0.3
        spec = 1.5
    elif mat == M_SILL:
        r, g, b = 0.88, 0.85, 0.78
        gloss = 0.12
        spec = 0.6
    elif mat == M_VALANCE:
        r, g, b = 0.84, 0.82, 0.77
    elif mat == M_DESKTOP:
        warm = 1.0 if var > 0.35 else 0.0
        v = 0.86 + 0.2 * var
        if ax == 1:
            # top: plywood veneer, grain along the long (z) side; pale worn rim, scratches, soft grime
            r, g, b = wood(lz, lx, var, foot, hz, hx, warm)
            e = min(hx - abs(lx), hz - abs(lz))
            along = lz if hx - abs(lx) < hz - abs(lz) else lx
            wr = wear_edge(e, along, var, foot)
            # the student-side edge (local -x) is the most worn
            if lx < 0 and hx - abs(lx) < 0.03:
                wr = max(wr, _ss(0.03, 0.0, hx - abs(lx)) * 0.55 * vnoise(lz * 9.0 + var * 20.0, 1.0))
            sc = scratches(lz, lx, var, foot)
            gr = vnoise(lz * 6.0 + var * 11.0, lx * 6.0) * _ss(0.05, 0.0, abs(lx + 0.05) - 0.08)
            r, g, b = r * v * (1 - 0.1 * gr), g * v * (1 - 0.12 * gr), b * v * (1 - 0.12 * gr)
            wk = max(wr, sc * 0.8)
            r = r + (0.98 - r) * 0.55 * wk
            g = g + (0.86 - g) * 0.55 * wk
            b = b + (0.66 - b) * 0.55 * wk
            gloss = 0.3 * (1 - 0.6 * wr)
            spec = 1.5 * (1 - 0.5 * wr)
        else:
            # plywood edge: stacked laminations, a pale top veneer lip, darker lower plies
            yy = (ly + hy) / (2 * hy)
            fk = _clamp(1.0 - foot / 0.0025, 0.0, 1.0)
            ply = 0.5 + 0.5 * math.sin(yy * 6.2832 * 3.5)
            lam = 0.86 - 0.14 * ply * fk - 0.07 * (1 - fk)
            lip = _ss(0.72, 0.9, yy)
            r = (0.80 * lam + 0.14 * lip) * v
            g = (0.58 * lam + 0.14 * lip) * v
            b = (0.36 * lam + 0.12 * lip) * v
            gloss = 0.1
            spec = 0.8
    elif mat == M_DESKMETAL:
        r, g, b = 0.56, 0.59, 0.62
        gloss = 0.14
        spec = 1.6
    elif mat == M_CHAIRWOOD:
        warm = 1.0 if var > 0.5 else 0.0
        v = 0.84 + 0.2 * var
        if ax == 1:
            r, g, b = wood(lz, lx, var + 0.37, foot, hz, hx, warm)
            e = min(hx - abs(lx), hz - abs(lz))
            wr = wear_edge(e, lx + lz, var + 0.5, foot)
        elif ax == 0:
            r, g, b = wood(lz, ly, var + 0.61, foot, hz, hy, warm)
            e = min(hy - abs(ly), hz - abs(lz))
            wr = wear_edge(e, ly + lz, var + 0.2, foot)
            # the top rail of the back gets grabbed: pale scuffed band with scratches
            if ly > 0:
                sc_ = vnoise(lz * 40.0 + var * 9.0, ly * 90.0)
                wr = max(wr, _ss(0.035, 0.0, hy - ly) * _ss(0.35, 0.8, sc_) * 0.9)
        else:
            r, g, b = 0.74, 0.53, 0.32
            wr = 0.0
        r, g, b = r * v * 0.95, g * v * 0.93, b * v * 0.93
        r = r + (0.97 - r) * 0.62 * wr
        g = g + (0.88 - g) * 0.62 * wr
        b = b + (0.74 - b) * 0.62 * wr
        gloss = 0.2 * (1 - 0.6 * wr)
        spec = 1.1
    elif mat == M_CHAIRMETAL:
        r, g, b = 0.56, 0.59, 0.62
        gloss = 0.14
        spec = 1.6
    elif mat == M_BAG:
        if var < 0.5:
            r, g, b = 0.09, 0.1, 0.14
        else:
            r, g, b = 0.35, 0.22, 0.14
        gloss = 0.1
        spec = 0.6
    elif mat == M_BOOK:
        k = int(var * 4.0)
        if k == 0:
            r, g, b = 0.75, 0.3, 0.25
        elif k == 1:
            r, g, b = 0.3, 0.45, 0.7
        elif k == 2:
            r, g, b = 0.92, 0.9, 0.84
        else:
            r, g, b = 0.4, 0.6, 0.4
    elif mat == M_BOARD:
        # blackboard: local face -x; texture over (z, y)
        r, g, b = 0.085, 0.2, 0.16
        if ax == 0:
            c = samp2(BT, (pz - 0.0) * bt_ppm, (HC - py) * bt_ppm)
            r = r + (0.88 - r) * c
            g = g + (0.9 - g) * c
            b = b + (0.86 - b) * c
        gloss = 0.06
        spec = 0.3
    elif mat == M_BOARDFRAME:
        r, g, b = 0.58, 0.45, 0.3
        gloss = 0.1
    elif mat == M_TRAY:
        r, g, b = 0.62, 0.63, 0.64
        gloss = 0.2
        spec = 1.0
    elif mat == M_PLATFORM:
        r, g, b = 0.52, 0.34, 0.2
        gloss = 0.35
        spec = 1.0
    elif mat == M_LECTERN:
        r, g, b = 0.6, 0.4, 0.25
        gloss = 0.2
        spec = 0.8
    elif mat == M_LIGHT:
        r, g, b = 0.94, 0.95, 0.95
        em = 0.18
        if ax == 1 and sg < 0:
            r, g, b = 0.98, 0.99, 1.0
            em = 0.45
    elif mat == M_CLOCK:
        r, g, b = 0.22, 0.22, 0.24
        if ax == 0 and sg < 0:
            # clock face in (lz, ly)
            rr = math.sqrt(lz * lz + ly * ly)
            R = 0.15
            if rr < R * 0.86:
                r, g, b = 0.96, 0.95, 0.92
                ang = math.atan2(lz, ly)
                tick = abs(math.sin(ang * 6.0))
                if rr > R * 0.7 and tick < 0.1:
                    r, g, b = 0.15, 0.15, 0.17
                # hands: 4:25 pm
                for hnd in range(2):
                    if hnd == 0:
                        a0, ln, wd = (4.0 + 25.0 / 60.0) / 12.0 * 6.2832, R * 0.45, 0.009
                    else:
                        a0, ln, wd = 25.0 / 60.0 * 6.2832, R * 0.7, 0.006
                    hx, hy = math.sin(a0), math.cos(a0)
                    along = lz * hx + ly * hy
                    perp = abs(-lz * hy + ly * hx)
                    if along > -0.015 and along < ln and perp < wd:
                        r, g, b = 0.12, 0.12, 0.14
                gloss = 0.15
                spec = 1.0
    elif mat == M_PAPER:
        k = int(var * 5.0)
        if k == 0:
            r, g, b = 0.95, 0.94, 0.9
        elif k == 1:
            r, g, b = 0.96, 0.92, 0.72
        elif k == 2:
            r, g, b = 0.94, 0.8, 0.8
        elif k == 3:
            r, g, b = 0.78, 0.88, 0.95
        else:
            r, g, b = 0.82, 0.93, 0.8
        # printed lines
        if ax == 0:
            s = line_f(ly, 0.025, 0.006, foot) * _ss(0.02, 0.03, B_half(lz, var))
            r *= 1 - 0.3 * s
            g *= 1 - 0.3 * s
            b *= 1 - 0.3 * s
    elif mat == M_SPEAKER:
        r, g, b = 0.82, 0.8, 0.76
    elif mat == M_TV:
        r, g, b = 0.05, 0.05, 0.06
        gloss = 0.4
        spec = 1.5
    elif mat == M_PARAPET:
        r, g, b = 0.78, 0.77, 0.74
    elif mat == M_RAIL:
        r, g, b = 0.62, 0.64, 0.67
        gloss = 0.3
        spec = 2.0
    elif mat == M_BALC:
        r, g, b = 0.62, 0.62, 0.62
    return r, g, b, gloss, spec, em


@njit(inline='always')
def B_half(lz, var):
    return 0.2 - abs(lz)


# ----------------------------------------------------------------------------- shading

@njit(fastmath=True, cache=True)
def hit(ox, oy, oz, dx, dy, dz, B, G):
    """Nearest intersection. -> t, kind (0 env, 1 room plane, 2 box), index, face axis, face sign."""
    tb, bi, bax, bsg = trace_boxes(ox, oy, oz, dx, dy, dz, B, G, 1e9, False)
    tr, pl = room_exit(ox, oy, oz, dx, dy, dz)
    through = False
    if pl == 4:
        qx = ox + dx * tr
        qy = oy + dy * tr
        if qx > AX0 and qx < AX1 and qy > AY0 and qy < AY1:
            through = True
    if oz < 0.0:
        through = True
    if bi >= 0 and (tb < tr or through):
        return tb, 2, bi, bax, bsg
    if through:
        return 1e9, 0, -1, 0, 1.0
    return tr, 1, pl, 0, 1.0


@njit(fastmath=True, cache=True)
def shade(ox, oy, oz, dx, dy, dz, fpx, level, B, G, VB, HB, CM, cm_ppm, cm_y0, SKY, sky_ppu, su0, sv1,
          TREE, tppm, tx0, ty1, BT, bt_ppm, FAO, fao_ppm, P, out):
    """Shade one ray. out[0:3] total colour, out[3:6] albedo, out[6:9] direct sun (for GI),
    out[9] t, out[10] id, out[11] reflection weight, out[12:15] reflect dir, out[15:18] reflect origin.
    P: lighting params array."""
    Lx, Ly, Lz = P[0], P[1], P[2]
    sunr, sung, sunb = P[3], P[4], P[5]
    ambr, ambg, ambb = P[6], P[7], P[8]
    ext = P[9]
    t, kind, idx, ax, sg = hit(ox, oy, oz, dx, dy, dz, B, G)
    out[11] = 0.0
    if kind == 0:
        r, g, b = env(ox, oy, oz, dx, dy, dz, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1)
        out[0] = r * ext
        out[1] = g * ext
        out[2] = b * ext
        out[3] = 0.0
        out[4] = 0.0
        out[5] = 0.0
        out[6] = 0.0
        out[7] = 0.0
        out[8] = 0.0
        out[9] = 1e3
        out[10] = -1.0
        return
    px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
    lx, ly, lz = 0.0, 0.0, 0.0
    var = 0.0
    if kind == 1:
        if idx == 0:
            nx, ny, nz, mat = 0.0, 1.0, 0.0, M_FLOOR
        elif idx == 1:
            nx, ny, nz, mat = 0.0, -1.0, 0.0, M_CEIL
        elif idx == 2:
            nx, ny, nz, mat = 1.0, 0.0, 0.0, M_WALL
        elif idx == 3:
            nx, ny, nz, mat = -1.0, 0.0, 0.0, M_WALL
        elif idx == 4:
            nx, ny, nz, mat = 0.0, 0.0, 1.0, M_WINWALL
        else:
            nx, ny, nz, mat = 0.0, 0.0, -1.0, M_WALL
        oid = float(idx)
    else:
        c, s = B[idx, 6], B[idx, 7]
        rx, ry, rz = px - B[idx, 0], py - B[idx, 1], pz - B[idx, 2]
        lx = c * rx - s * rz
        lz = s * rx + c * rz
        ly = ry
        # local normal -> world
        lnx, lny, lnz = 0.0, 0.0, 0.0
        if ax == 0:
            lnx = sg
        elif ax == 1:
            lny = sg
        else:
            lnz = sg
        mt = int(B[idx, 8])
        if mt != M_PAPER and mt != M_LIGHT and mt != M_CLOCK and mt != M_BOARD and mt != M_WALL:
            # bevelled / rounded edges: bend the normal toward the nearest edges (thin metal members
            # are fully rounded, i.e. read as tubes with a specular line)
            metal = mt == M_DESKMETAL or mt == M_CHAIRMETAL or mt == M_RAIL or mt == M_FRAME
            for a in range(3):
                if a == ax:
                    continue
                if a == 0:
                    hh, cc = B[idx, 3], lx
                elif a == 1:
                    hh, cc = B[idx, 4], ly
                else:
                    hh, cc = B[idx, 5], lz
                bw = 0.0075
                if metal and hh < 0.026:
                    bw = hh
                elif mt == M_PILLAR:
                    bw = 0.02
                bw = min(bw, hh)
                e = hh - abs(cc)
                if e < bw:
                    kk = 1.0 - e / bw
                    tw = (kk * 1.35) * (1.0 if cc > 0 else -1.0)
                    if a == 0:
                        lnx += tw
                    elif a == 1:
                        lny += tw
                    else:
                        lnz += tw
            if mt == M_CHAIRWOOD and ax == 0 and B[idx, 4] > 0.05:
                lnz += lz / B[idx, 5] * 0.45          # moulded, curved chair back
            nl = math.sqrt(lnx * lnx + lny * lny + lnz * lnz)
            lnx /= nl
            lny /= nl
            lnz /= nl
        nx = c * lnx + s * lnz
        ny = lny
        nz = -s * lnx + c * lnz
        mat = int(B[idx, 8])
        var = B[idx, 9]
        oid = 10.0 + idx
    ndv = -(nx * dx + ny * dy + nz * dz)
    foot = t / fpx / max(abs(ndv), 0.08)
    if level > 0:
        foot = 1.0
    hx, hz, hy = 1.0, 1.0, 1.0
    if kind == 2:
        hx, hz, hy = B[idx, 3], B[idx, 5], B[idx, 4]
    ar, ag, ab, gloss, spec, em = material(mat, var, px, py, pz, lx, ly, lz, ax, sg, foot, BT, bt_ppm, FAO, fao_ppm,
                                           hx, hz, hy)
    cav = 1.0
    if kind == 2 and mat == M_DESKMETAL and ax == 0 and sg < 0 and B[idx, 4] > 0.03 and B[idx, 4] < 0.06:
        # open book compartment under the desk top (faces the student): a dark recess with a lit lip
        hy_ = B[idx, 4]
        ez = hz - abs(lz) - 0.012
        ey = min(hy_ - ly - 0.004, ly + hy_ - 0.012)
        inside = _clamp(min(ez, ey) / max(foot, 1e-4) + 0.5, 0.0, 1.0)
        cav = 1.0 - 0.9 * inside
        depthk = _clamp((hy_ - ly) / (2 * hy_), 0.0, 1.0)
        cav = cav + 0.22 * inside * (1.0 - 0.5 * depthk)
        ar = ar * (1 - inside) + 0.2 * inside
        ag = ag * (1 - inside) + 0.19 * inside
        ab = ab * (1 - inside) + 0.24 * inside
        gloss *= 1 - inside
        spec *= 1 - inside
    # --- direct sun
    ndl = nx * Lx + ny * Ly + nz * Lz
    sv = 0.0
    if ndl > 0.0:
        sv = sun_vis(px, py, pz, nx, ny, nz, Lx, Ly, Lz, B, G, VB, HB, CM, cm_ppm, cm_y0, True) * ndl
    # --- ambient: cool skylight entering through the windows + occlusion
    fwin = 0.5 - 0.5 * nz
    near = 0.55 + 0.75 * math.exp(-pz / 2.2)
    ao = 1.0
    if mat == M_FLOOR:
        ao = samp2(FAO, px * fao_ppm, pz * fao_ppm)
    else:
        ao = 1.0 - 0.35 * math.exp(-py / 0.12) - 0.2 * math.exp(-(HC - py) / 0.15)
        if kind == 2 and ny < -0.5:
            ao *= 0.6
    up = 0.85 + 0.15 * ny
    amb = (0.3 + 0.7 * fwin) * near * ao * up * cav
    if pz < -0.01:
        amb = 1.6 * (0.6 + 0.4 * ny)
    # ambient tint: the upper room / deep room lean cool blue, beam sides cooler still
    tr, tg, tb = 1.0, 1.0, 1.0
    if py > 2.4:
        dk = 1.0 - math.exp(-pz / 3.0)
        tr, tg, tb = 0.94 - 0.1 * dk, 0.98 - 0.03 * dk, 1.0 + 0.08 * dk
        if abs(nx) > 0.5 and mat == M_PILLAR:
            tr, tg, tb = tr * 0.9, tg * 0.95, tb * 1.04
    if mat == M_DESKTOP or mat == M_CHAIRWOOD:
        tr, tg, tb = tr * 1.42, tg * 1.12, tb * 0.8
    # warm bounce from the sunlit floor / desks onto downward-facing surfaces (ceiling, beam soffits,
    # desk undersides): strong near the windows, fading deeper into the room
    dn = max(0.0, -ny)
    bnc = P[10] * dn * (0.35 + 0.65 * math.exp(-abs(pz - 2.5) / 2.5)) * ao * cav
    if py > 2.4:
        xk = math.exp(-((px - 5.2) / 3.2) ** 2)
        bnc = dn * (0.8 * math.exp(-pz / 1.6) + 0.22 * math.exp(-((pz - 3.0) / 1.4) ** 2)) * (0.5 + 0.5 * xk)
    lr = sunr * sv * cav + ambr * amb * tr + bnc * 1.0
    lg = sung * sv * cav + ambg * amb * tg + bnc * 0.6
    lb = sunb * sv * cav + ambb * amb * tb + bnc * 0.36
    r = ar * (lr + em)
    g = ag * (lg + em)
    b = ab * (lb + em * 1.05)
    out[6] = ar * sunr * sv
    out[7] = ag * sung * sv
    out[8] = ab * sunb * sv
    # sun glint (specular): reflected view ray vs the sun direction
    if spec > 0.0 and sv > 0.0:
        rdx = dx + 2 * ndv * nx
        rdy = dy + 2 * ndv * ny
        rdz = dz + 2 * ndv * nz
        cs = rdx * Lx + rdy * Ly + rdz * Lz
        if cs > 0.9:
            sp = spec * (math.exp((cs - 1.0) * 400.0) * 3.0 + math.exp((cs - 1.0) * 40.0) * 0.35)
            r += sp * sunr * sv / max(ndl, 0.05) * 0.5
            g += sp * sung * sv / max(ndl, 0.05) * 0.5
            b += sp * sunb * sv / max(ndl, 0.05) * 0.5
    # steel: crisp reflection of the bright window panes (a hard specular streak running along each
    # tube, the sash bars read as breaks in it)
    if (mat == M_DESKMETAL or mat == M_CHAIRMETAL or mat == M_TRAY) and pz > 0.0:
        rdx = dx + 2 * ndv * nx
        rdy = dy + 2 * ndv * ny
        rdz = dz + 2 * ndv * nz
        if rdz < -0.05:
            tw = -pz / rdz
            xw = px + rdx * tw
            yw = py + rdy * tw
            if B[idx, 4] > B[idx, 3] and B[idx, 4] > B[idx, 5]:
                yw = 1.75 + (yw - 1.75) * 0.12          # vertical tube: anisotropic streak along it
            a = aperture(xw, yw, 0.01 + tw * 0.004, VB, HB)
            if a > 0.0:
                fr_ = 0.35 + 0.65 * (1 - max(ndv, 0.0)) ** 3
                # warmer / hotter toward the sun
                cs = rdx * Lx + rdy * Ly + rdz * Lz
                hot = math.exp((cs - 1.0) * 12.0)
                kk = a * fr_ * (0.9 + 2.5 * hot) * math.exp(-tw / 6.0) * 1.6
                r += kk * (0.95 + 0.3 * hot)
                g += kk * (0.97 + 0.1 * hot)
                b += kk * (1.0 - 0.2 * hot)
    out[0] = r
    out[1] = g
    out[2] = b
    out[3] = ar
    out[4] = ag
    out[5] = ab
    out[9] = t
    out[10] = oid
    if gloss > 0.0 and level == 0:
        fr = gloss * (0.25 + 0.75 * (1 - max(ndv, 0.0)) ** 4)
        out[11] = fr
        out[12] = dx + 2 * ndv * nx
        out[13] = dy + 2 * ndv * ny
        out[14] = dz + 2 * ndv * nz
        out[15] = px + nx * 2e-3
        out[16] = py + ny * 2e-3
        out[17] = pz + nz * 2e-3


@njit(parallel=True, fastmath=True, cache=True)
def render(W, H, cam, fpx, B, G, VB, HB, CM, cm_ppm, cm_y0, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1,
           BT, bt_ppm, FAO, fao_ppm, P, col, alb, direct, refl, rw, depth, oid):
    """Primary pass (1 spp) + one glossy reflection bounce."""
    ox, oy, oz = cam[0], cam[1], cam[2]
    for yy in prange(H):
        out = np.zeros(18, np.float64)
        out2 = np.zeros(18, np.float64)
        for xx in range(W):
            sx = xx + 0.5 - W * 0.5
            sy = H * 0.5 - (yy + 0.5)
            dx = cam[3] * fpx + cam[6] * sx + cam[9] * sy
            dy = cam[4] * fpx + cam[7] * sx + cam[10] * sy
            dz = cam[5] * fpx + cam[8] * sx + cam[11] * sy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n
            dy /= n
            dz /= n
            shade(ox, oy, oz, dx, dy, dz, fpx, 0, B, G, VB, HB, CM, cm_ppm, cm_y0, SKY, sky_ppu, su0, sv1,
                  TREE, tppm, tx0, ty1, BT, bt_ppm, FAO, fao_ppm, P, out)
            for c in range(3):
                col[yy, xx, c] = out[c]
                alb[yy, xx, c] = out[3 + c]
                direct[yy, xx, c] = out[6 + c]
                refl[yy, xx, c] = 0.0
            depth[yy, xx] = out[9]
            oid[yy, xx] = out[10]
            rw[yy, xx] = out[11]
            if out[11] > 0.0:
                shade(out[15], out[16], out[17], out[12], out[13], out[14], fpx, 1, B, G, VB, HB, CM, cm_ppm,
                      cm_y0, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1, BT, bt_ppm, FAO, fao_ppm, P, out2)
                for c in range(3):
                    refl[yy, xx, c] = out2[c]


@njit(parallel=True, fastmath=True, cache=True)
def render_aa(W, H, cam, fpx, B, G, VB, HB, CM, cm_ppm, cm_y0, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1,
              BT, bt_ppm, FAO, fao_ppm, P, pys, pxs, col, alb, direct, rw):
    """Re-shade edge pixels with 4 extra rotated-grid samples (averaged with the centre sample)."""
    offs = np.array([[-0.125, -0.375], [0.375, -0.125], [0.125, 0.375], [-0.375, 0.125]])
    ox, oy, oz = cam[0], cam[1], cam[2]
    for k in prange(pys.shape[0]):
        yy, xx = pys[k], pxs[k]
        out = np.zeros(18, np.float64)
        acc = np.zeros(10, np.float64)
        for c in range(3):
            acc[c] = col[yy, xx, c]
            acc[3 + c] = alb[yy, xx, c]
            acc[6 + c] = direct[yy, xx, c]
        acc[9] = rw[yy, xx]
        for s in range(4):
            sx = xx + 0.5 + offs[s, 0] - W * 0.5
            sy = H * 0.5 - (yy + 0.5 + offs[s, 1])
            dx = cam[3] * fpx + cam[6] * sx + cam[9] * sy
            dy = cam[4] * fpx + cam[7] * sx + cam[10] * sy
            dz = cam[5] * fpx + cam[8] * sx + cam[11] * sy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            shade(ox, oy, oz, dx / n, dy / n, dz / n, fpx, 0, B, G, VB, HB, CM, cm_ppm, cm_y0, SKY, sky_ppu, su0,
                  sv1, TREE, tppm, tx0, ty1, BT, bt_ppm, FAO, fao_ppm, P, out)
            for c in range(9):
                acc[c] += out[c]
            acc[9] += out[11]
        for c in range(3):
            col[yy, xx, c] = acc[c] / 5
            alb[yy, xx, c] = acc[3 + c] / 5
            direct[yy, xx, c] = acc[6 + c] / 5
        rw[yy, xx] = acc[9] / 5


@njit(parallel=True, fastmath=True, cache=True)
def volume(W, H, cam, fpx, depth_lo, Lx, Ly, Lz, VB, HB, CM, cm_ppm, cm_y0, B, G, jit, t, steps, out):
    """Low-res single-scattering march through the sunlit air. depth_lo: (H, W) ray distances."""
    ox, oy, oz = cam[0], cam[1], cam[2]
    for yy in prange(H):
        for xx in range(W):
            sx = xx + 0.5 - W * 0.5
            sy = H * 0.5 - (yy + 0.5)
            dx = cam[3] * fpx + cam[6] * sx + cam[9] * sy
            dy = cam[4] * fpx + cam[7] * sx + cam[10] * sy
            dz = cam[5] * fpx + cam[8] * sx + cam[11] * sy
            n = math.sqrt(dx * dx + dy * dy + dz * dz)
            dx /= n
            dy /= n
            dz /= n
            tm = min(depth_lo[yy, xx], 14.0)
            # only march inside the room (stop at the window plane)
            if dz < 0:
                tm = min(tm, (oz - 0.0) / -dz)
            acc = 0.0
            dt = tm / steps
            j = jit[yy % jit.shape[0], xx % jit.shape[1]]
            for i in range(steps):
                tt = (i + j) * dt
                px, py, pz = ox + dx * tt, oy + dy * tt, oz + dz * tt
                if pz <= 0.01:
                    continue
                tw = -pz / Lz
                xw = px + Lx * tw
                yw = py + Ly * tw
                a = aperture(xw, yw, 0.004 + tw * 0.012, VB, HB)
                if a <= 0.0:
                    continue
                a *= curtain_T(xw, yw, CM, cm_ppm, cm_y0)
                if py < 0.8:
                    tb, bi, bax, bsg = trace_boxes(px, py, pz, Lx, Ly, Lz, B, G, tw, True)
                    if bi >= 0:
                        continue
                # dusty air: streaks running along the beams (noise in window-plane coords) drifting slowly
                s1 = 0.5 + 0.5 * math.sin(xw * 7.0 + math.sin(yw * 2.3 + t * 0.07) * 2.2 + t * 0.09)
                s2 = 0.5 + 0.5 * math.sin(xw * 2.3 - yw * 4.1 + math.sin(xw * 1.7) * 1.5 - t * 0.05)
                dn = 0.12 + 0.88 * s1 * s1 * (0.4 + 0.6 * s2)
                dn *= 0.5 + 0.9 * _ss(0.3, 3.0, pz) * math.exp(-max(pz - 3.0, 0.0) / 3.0)
                acc += a * dn * dt
            cs = dx * Lx + dy * Ly + dz * Lz
            ph = 0.4 + (1 - 0.16) / (1 + 0.16 - 0.8 * cs) ** 1.5 * 0.3
            out[yy, xx] = acc * ph
