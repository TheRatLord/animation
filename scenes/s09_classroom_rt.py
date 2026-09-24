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
M_JACKET, M_NOTE, M_PCASE, M_ERASER, M_CHALK = 60, 61, 62, 63, 64
M_FRONTWALL = 4


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


BACK_BEND, BACK_ARCH = 0.028, 0.016


@njit(inline='always', fastmath=True)
def _rrect(a, b, ha, hb, r):
    """signed distance to a rounded rectangle (half extents ha, hb, corner radius r)"""
    qa = abs(a) - ha + r
    qb = abs(b) - hb + r
    return math.sqrt(max(qa, 0.0) ** 2 + max(qb, 0.0) ** 2) + min(max(qa, qb), 0.0) - r


@njit(inline='always', fastmath=True)
def back_outline(ly, lz, hy, hz):
    """2D outline of a chair back panel: top edge arched down toward the sides, generous rounded top
    corners, tighter bottom ones.  < 0 inside"""
    q = lz / hz
    yy = ly + BACK_ARCH * q * q
    rr = 0.045 if yy > 0.0 else 0.012
    return _rrect(yy, lz, hy, hz, rr)


@njit(fastmath=True, cache=True)
def shape_sdf(sh, lx, ly, lz, hx, hy, hz):
    if sh == 1:
        # bent plywood chair back: the panel curves forward (+x) toward its sides
        q = lz / hz
        th = hx - BACK_BEND * 0.5 - 0.002
        xc = BACK_BEND * (q * q - 0.5)
        d1 = abs(lx - xc) - th
        d2 = back_outline(ly, lz, hy, hz)
        re = 0.004
        a = d1 + re
        b = d2 + re
        return math.sqrt(max(a, 0.0) ** 2 + max(b, 0.0) ** 2) + min(max(a, b), 0.0) - re
    # 2: rounded capsule box (pencil case)
    r = min(hx, hy) * 0.95
    qx = abs(lx) - hx + r
    qy = abs(ly) - hy + r
    qz = abs(lz) - hz + r
    return math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2) + \
        min(max(qx, max(qy, qz)), 0.0) - r


@njit(fastmath=True, cache=True)
def ray_shape(ox, oy, oz, dx, dy, dz, B, i, tmax):
    """sphere-trace the SDF shape inside box i -> t (inf on miss)"""
    c, s = B[i, 6], B[i, 7]
    rx, ry, rz = ox - B[i, 0], oy - B[i, 1], oz - B[i, 2]
    lox = c * rx - s * rz
    loz = s * rx + c * rz
    loy = ry
    ldx = c * dx - s * dz
    ldz = s * dx + c * dz
    ldy = dy
    hx, hy, hz = B[i, 3], B[i, 4], B[i, 5]
    t0, t1 = 1e-4, tmax
    for a in range(3):
        if a == 0:
            o, d, h = lox, ldx, hx
        elif a == 1:
            o, d, h = loy, ldy, hy
        else:
            o, d, h = loz, ldz, hz
        if abs(d) < 1e-9:
            if o < -h or o > h:
                return np.inf
            continue
        ta = (-h - o) / d
        tb = (h - o) / d
        if ta > tb:
            ta, tb = tb, ta
        if ta > t0:
            t0 = ta
        if tb < t1:
            t1 = tb
        if t0 > t1:
            return np.inf
    sh = int(B[i, 11])
    t = t0
    for k in range(64):
        px, py, pz = lox + ldx * t, loy + ldy * t, loz + ldz * t
        dd = shape_sdf(sh, px, py, pz, hx, hy, hz)
        if dd < 2e-5 + 1e-5 * t:
            return t
        t += dd * 0.92
        if t > t1:
            return np.inf
    return np.inf


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
            if B[i, 11] > 0.5 and not shadow:
                t = ray_shape(ox, oy, oz, dx, dy, dz, B, i, best)
                ax, sg = 0, 1.0
            else:
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
    pen = 0.002 + tw * 0.005
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
def env(ox, oy, oz, dx, dy, dz, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1, tt):
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
    # leaves sway: a slow smooth warp of the crown plate (bigger up high), plus a faint flutter
    sw = _clamp((qy + 1.5) / 5.0, 0.0, 1.0)
    qx += sw * (0.045 * math.sin(tt * 1.1 + qx * 0.35 + qy * 0.2) + 0.012 * math.sin(tt * 3.3 + qx * 2.1 + qy * 1.7))
    qy += sw * 0.02 * math.sin(tt * 1.4 + qx * 0.5 + 1.3)
    tx = (qx - tx0) * tppm
    ty = (ty1 - qy) * tppm
    a = samp(TREE, tx, ty, 3)
    if a > 0.0:
        # slight aerial perspective on the trees
        tr = samp(TREE, tx, ty, 0) * 0.95 + 0.05 * 0.62
        tg = samp(TREE, tx, ty, 1) * 0.95 + 0.05 * 0.78
        tb = samp(TREE, tx, ty, 2) * 0.95 + 0.05 * 0.92
        # sun flicker on the lit leaves: slow drifting patches of extra light (coherent in time)
        lk = _ss(0.4, 0.75, tg)
        if lk > 0.0:
            fl = vnoise(qx * 1.3 + tt * 0.55, qy * 1.6 - tt * 0.35) * 0.65 +                 vnoise(qx * 3.1 - tt * 0.9, qy * 2.7 + tt * 0.6) * 0.35
            fl = 1.0 + lk * 0.4 * (_ss(0.4, 0.8, fl) - 0.3)
            tr *= fl
            tg *= fl
            tb *= fl
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
    fk = _clamp(1.0 - foot / 0.01, 0.0, 1.0)           # ring lines
    fk2 = _clamp(1.0 - foot / 0.04, 0.0, 1.0)          # ring bands
    n1 = vnoise(u * 3.0 + sd, w * 8.0)
    wv = w + 0.012 * math.sin(u * 4.1 + sd) + 0.008 * (n1 - 0.5) + 0.003 * math.sin(u * 17.0 + sd * 2.0)
    c0 = (hash1(sd + 1.0) - 0.5) * hw * 2.6
    u0 = (hash1(sd + 2.0) - 0.5) * hu * 1.2
    q = (u - u0) / hu
    ring = abs(wv - c0) * 30.0 + 0.8 * q * q + 1.6 * vnoise(u * 1.2 - sd, w * 2.5) + 0.02 * u * 30.0
    f = ring - math.floor(ring)
    late = _ss(0.45, 0.92, f) * (1.0 - _ss(0.93, 1.0, f))
    line = math.exp(-((1.0 - f) * 8.0) ** 2) + math.exp(-(f * 12.0) ** 2) * 0.5
    streak = vnoise(u * 2.0 + sd * 3.0, w * 110.0)
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
    fks = _clamp(1.0 - foot / 0.0035, 0.0, 1.0)
    ln = (line * 0.45 * fk + 0.1 * (1.0 - fk)) + streak * 0.3 * fks + 0.08 * (1.0 - fks)
    r = r + (kr - r) * ln
    g = g + (kg - g) * ln
    b = b + (kb - b) * ln
    m = (0.9 + 0.2 * mott) * (0.9 + 0.2 * band)
    return r * m, g * m, b * m


@njit(fastmath=True, cache=True)
def grain(u, w, var, foot, warm, hu, hw, nk=0):
    """Painted long-grain board: straight, slightly wavy streaks running along u (the board's length),
    a few broad colour bands, 0-2 dark knots with the grain flowing around them.
    u along the grain, w across (local metres, centred); hu / hw half extents. -> r, g, b albedo"""
    sd = var * 97.0
    br, bg, bb = 0.90, 0.68, 0.43          # honey early wood
    lr_, lg_, lb_ = 0.74, 0.50, 0.29       # amber late wood
    kr, kg, kb = 0.47, 0.28, 0.15          # dark streak / knot
    if warm < 0.5:
        br, bg, bb = 0.86, 0.66, 0.43
        lr_, lg_, lb_ = 0.69, 0.47, 0.29
        kr, kg, kb = 0.43, 0.26, 0.15
    if foot > 0.04:
        return br * 0.93, bg * 0.92, bb * 0.92
    fk = _clamp(1.0 - foot / 0.008, 0.0, 1.0)      # fine streaks
    fk2 = _clamp(1.0 - foot / 0.025, 0.0, 1.0)     # bands
    # gentle long waves in the grain + a little per-board drift
    wv = w + 0.005 * math.sin(u * 2.3 + sd) + 0.003 * math.sin(u * 7.1 + sd * 1.7) + \
        0.004 * (vnoise(u * 1.6 + sd, w * 4.0) - 0.5)
    kn = 0.0
    halo = 0.0
    for k in range(3):
        on = hash1(sd + k * 7.31) < 0.5 if nk == 0 else k < nk
        if on:
            ku = (hash1(sd + k * 3.7 + 1.0) - 0.5) * 1.6 * hu
            kw = (hash1(sd + k * 5.1 + 2.0) - 0.5) * 1.6 * hw
            rk = 0.004 + 0.005 * hash1(sd + k * 9.9)
            if nk > 0:
                rk *= 1.5
            du = (u - ku) / (rk * 3.2)
            dw = (wv - kw) / rk
            d2 = du * du + dw * dw
            bend = math.exp(-d2 * 0.12)
            sgn = 1.0 if wv > kw else -1.0
            wv = wv + sgn * rk * 1.6 * bend             # streaks part around the knot
            kn = max(kn, _ss(1.3, 0.5, math.sqrt(d2)))
            halo = max(halo, bend)
    s1 = vnoise(u * 1.3 + sd, wv * 125.0)        # fine straight streaks (~8 mm pitch)
    s1 = _ss(0.5, 0.85, s1)
    s2 = vnoise(u * 0.5 - sd, wv * 60.0)         # medium late-wood bands
    s3 = vnoise(u * 0.25 + sd * 2.0, wv * 14.0)  # broad colour drift across the board
    ml = (_ss(0.35, 0.75, s2) * 0.9 * fk2 + 0.35 * (1.0 - fk2)) * (0.75 + 0.5 * s3)
    ml = min(1.0, ml + 0.35 * halo)
    r = br + (lr_ - br) * ml
    g = bg + (lg_ - bg) * ml
    b = bb + (lb_ - bb) * ml
    s4 = vnoise(u * 0.7 + sd * 1.3, wv * 32.0)          # long dark painted streaks (low frequency)
    s4 = _ss(0.55, 0.85, s4) * fk2
    ln = s1 * 0.6 * fk + 0.1 * (1.0 - fk) + kn * 0.85 * fk2 + s4 * (0.45 if nk > 0 else 0.25)
    r = r + (kr - r) * ln
    g = g + (kg - g) * ln
    b = b + (kb - b) * ln
    m = (0.94 + 0.12 * s3) * (1.0 - 0.5 * kn * fk2 * (1.0 if nk > 0 else 0.5))
    return r * m, g * m, b * m


@njit(fastmath=True, cache=True)
def chips(e, along, cdist, var, foot):
    """small dark chips / dings along an edge, clustered near the corners. e = distance to the edge,
    along = coordinate along the edge, cdist = distance to the nearest corner -> 0..1"""
    n = vnoise(along * 90.0 + var * 31.0, var * 13.0)
    d = _ss(0.72, 0.9, n) * 0.004 + 0.0012 * _ss(0.06, 0.0, cdist) * _ss(0.4, 0.8, n)
    near = 0.35 + 0.65 * _ss(0.08, 0.0, cdist)
    return _clamp((d - e) / max(foot, 1e-4) + 0.5, 0.0, 1.0) * near


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
    for k in range(14):
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
            wd = 0.0009
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
        h3 = hash1(j * 13.0 + k * 3.0 + 0.5)
        v = 0.72 + 0.45 * h2 + 0.05 * math.sin(px * 3.1 + h * 20.0) * math.sin(px * 0.7 + h2 * 9.0)
        # long painted streaks along each board
        fk = _clamp(1.0 - foot / 0.01, 0.0, 1.0)
        st = vnoise(px * 1.2 + h * 40.0, pz * 160.0)
        v *= 1.0 + fk * 0.1 * (st - 0.5)
        r, g, b = (0.56 + 0.05 * h3) * v, (0.36 + 0.012 * h3) * v, (0.215 - 0.03 * h3) * v
        # worn paths down the aisles (between the desk columns) and across the front: varnish scuffed
        # matt and paler, broken up along the boards
        wp = 0.0
        for zc_ in (1.93, 2.98, 4.03, 5.08):
            wp = max(wp, math.exp(-((pz - zc_) / 0.16) ** 2))
        wp = max(wp, math.exp(-((px - 7.75) / 0.35) ** 2) * _ss(0.8, 1.4, pz))
        wp *= 0.55 + 0.45 * vnoise(px * 2.0 + h * 13.0, pz * 11.0)
        r = r + (0.66 - r) * 0.32 * wp
        g = g + (0.5 - g) * 0.32 * wp
        b = b + (0.36 - b) * 0.32 * wp
        seam = line_f(pz, pw, 0.003, foot)
        seam2 = line_f(px + off, seg, 0.003, foot)
        s = max(seam, seam2 * 0.9)
        r *= 1 - 0.6 * s
        g *= 1 - 0.62 * s
        b *= 1 - 0.6 * s
        # the rounded board edge next to each seam catches the light: a thin pale line
        sh_ = line_f(pz - 0.0045, pw, 0.0025, foot) * (1 - s)
        r = r + (0.9 - r) * 0.3 * sh_
        g = g + (0.7 - g) * 0.3 * sh_
        b = b + (0.5 - b) * 0.3 * sh_
        # varnish is uneven per board: reflections break into streaks along the boards
        gloss = 0.5 * (0.35 + 0.75 * h2) * (0.7 + 0.5 * st) * (1.0 - 0.6 * wp)
        spec = 1.0 - 0.5 * wp
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
    elif mat == M_WALL or mat == M_WINWALL or mat == M_PILLAR or mat == M_FRONTWALL:
        r, g, b = 0.93, 0.885, 0.79
        if mat == M_WINWALL and py < AY0:
            r, g, b = 0.86, 0.83, 0.76
        if py < 0.75 and py >= 0.09 and mat != M_PILLAR:
            # scuffs: chair backs and shoes rub the lower wall - short grey horizontal smears
            wc = pz if (mat == M_FRONTWALL or mat == M_WALL) else px
            fk = _clamp(1.0 - foot / 0.01, 0.0, 1.0)
            n1 = vnoise(wc * 9.0, py * 70.0)
            n2 = vnoise(wc * 3.0 + 7.0, py * 12.0)
            sc = _ss(0.62, 0.85, n1) * _ss(0.4, 0.75, n2) * _ss(0.75, 0.2, py)
            sc += 0.35 * _ss(0.2, 0.09, py) * vnoise(wc * 20.0, 3.0)
            sc *= 0.35 + 0.65 * fk
            r, g, b = r * (1 - 0.3 * sc), g * (1 - 0.31 * sc), b * (1 - 0.28 * sc)
        if mat == M_FRONTWALL:
            a = samp(BT, pz * bt_ppm, (HC - py) * bt_ppm, 4)
            if a > 0.0:
                r = r * (1 - a) + samp(BT, pz * bt_ppm, (HC - py) * bt_ppm, 1)
                g = g * (1 - a) + samp(BT, pz * bt_ppm, (HC - py) * bt_ppm, 2)
                b = b * (1 - a) + samp(BT, pz * bt_ppm, (HC - py) * bt_ppm, 3)
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
            r, g, b = grain(lz, lx, var, foot, warm, hz, hx, 2 + int(hash1(var * 5.0) < 0.5))
            e = min(hx - abs(lx), hz - abs(lz))
            along = lz if hx - abs(lx) < hz - abs(lz) else lx
            wr = wear_edge(e, along, var, foot)
            # the student-side edge (local -x) is the most worn
            if lx < 0 and hx - abs(lx) < 0.03:
                wr = max(wr, _ss(0.03, 0.0, hx - abs(lx)) * 0.55 * vnoise(lz * 9.0 + var * 20.0, 1.0))
            # palms / forearms rest here: two pale scuffed patches on the student side, varnish rubbed off
            hp = 0.0
            for sgz in (-1.0, 1.0):
                dz_ = (lz - sgz * (0.11 + 0.04 * hash1(var * 7.0 + sgz))) / 0.1
                dx_ = (lx + hx - 0.055) / 0.06
                hp = max(hp, math.exp(-(dz_ * dz_ + dx_ * dx_)))
            hp *= 0.55 + 0.45 * vnoise(lz * 30.0 + var * 5.0, lx * 30.0)
            wr = max(wr, _ss(0.12, 0.7, hp) * 0.9)
            sc = scratches(lz, lx, var, foot)
            gr = vnoise(lz * 6.0 + var * 11.0, lx * 6.0) * _ss(0.05, 0.0, abs(lx + 0.05) - 0.08)
            r, g, b = r * v * (1 - 0.1 * gr), g * v * (1 - 0.12 * gr), b * v * (1 - 0.12 * gr)
            wk = max(wr, sc * 0.9)
            r = r + (1.0 - r) * 0.78 * wk
            g = g + (0.9 - g) * 0.78 * wk
            b = b + (0.72 - b) * 0.78 * wk
            cdist = max(hx - abs(lx), hz - abs(lz))        # distance along the edge to the corner
            ch = chips(e, along, cdist, var, foot)
            r = r + (0.36 - r) * ch
            g = g + (0.22 - g) * ch
            b = b + (0.12 - b) * ch
            gloss = 0.12 * (1 - 0.6 * wr)
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
        # each desk its own tint: some warmer / redder, some cooler / greyer, lighter or darker
        t1 = hash1(var * 17.3 + 0.7)
        t2 = hash1(var * 29.1 + 2.3)
        lv = 0.84 + 0.3 * t1
        r, g, b = r * lv * (0.95 + 0.12 * t2), g * lv * (0.98 + 0.03 * t2), b * lv * (1.08 - 0.2 * t2)
    elif mat == M_DESKMETAL:
        r, g, b = 0.5, 0.54, 0.62
        gloss = 0.14
        spec = 1.6
        if hy > 0.03 and hx > 0.1:
            # book-box: painted sheet steel, cool blue-grey, a few scuffs
            n = vnoise(lz * 25.0 + var * 3.0, ly * 60.0 + lx * 25.0)
            r, g, b = 0.5 + 0.06 * n, 0.53 + 0.06 * n, 0.64 + 0.05 * n
            gloss = 0.08
            spec = 0.9
    elif mat == M_CHAIRWOOD:
        warm = 1.0 if var > 0.5 else 0.0
        v = 0.84 + 0.2 * var
        ch = 0.0
        if ax == 1:
            r, g, b = grain(lz, lx, var + 0.37, foot, warm, hz, hx)
            e = min(hx - abs(lx), hz - abs(lz))
            wr = wear_edge(e, lx + lz, var + 0.5, foot)
            ch = chips(e, lx + lz, max(hx - abs(lx), hz - abs(lz)), var + 0.3, foot)
        elif ax == 0:
            r, g, b = grain(lz, ly, var + 0.61, foot, warm, hz, hy)
            e = min(hy - abs(ly), hz - abs(lz))
            if hy > 0.05:
                e = -back_outline(ly, lz, hy, hz)
            wr = wear_edge(e, ly + lz, var + 0.2, foot)
            ch = chips(e, ly + lz, math.sqrt((hy - abs(ly)) ** 2 + (hz - abs(lz)) ** 2) - 0.05, var + 0.7, foot)
            # the top rail of the back gets grabbed: pale scuffed patch where the hands go, scratches
            if ly > 0:
                sc_ = vnoise(lz * 40.0 + var * 9.0, ly * 90.0)
                grab = math.exp(-((abs(lz) - 0.1) / 0.07) ** 2)
                wr = max(wr, _ss(0.06, 0.0, hy - ly) * (0.35 + 0.65 * grab) * _ss(0.3, 0.75, sc_) * 0.95)
        else:
            r, g, b = 0.74, 0.53, 0.32
            wr = 0.0
        t1 = hash1(var * 13.7 + 4.1)
        t2 = hash1(var * 23.9 + 1.9)
        lv = 0.86 + 0.28 * t1
        r, g, b = r * v * 0.95 * lv * (0.95 + 0.12 * t2), g * v * 0.93 * lv, b * v * 0.93 * lv * (1.08 - 0.2 * t2)
        r = r + (0.97 - r) * 0.62 * wr
        g = g + (0.88 - g) * 0.62 * wr
        b = b + (0.74 - b) * 0.62 * wr
        r = r + (0.34 - r) * ch
        g = g + (0.21 - g) * ch
        b = b + (0.12 - b) * ch
        gloss = 0.08 * (1 - 0.6 * wr)
        spec = 1.1
    elif mat == M_CHAIRMETAL:
        r, g, b = 0.5, 0.54, 0.62
        gloss = 0.14
        spec = 1.6
    elif mat == M_BAG:
        k = int(var * 4.0)
        if k == 0:
            r, g, b = 0.15, 0.17, 0.3          # navy nylon school bag
        elif k == 1:
            r, g, b = 0.36, 0.23, 0.15         # brown leather
        elif k == 2:
            r, g, b = 0.42, 0.14, 0.16         # maroon tote
        else:
            r, g, b = 0.2, 0.3, 0.55           # blue sports bag
        if ax == 2 and hx > 0.05:
            # flap over the front, stitched edge, a buckle / zip pull
            fl = ly - (-hy * 0.15 + 0.02 * math.sin(lx * 20.0))
            if fl > 0:
                r, g, b = r * 1.18, g * 1.18, b * 1.15
            if abs(fl) < 0.004:
                r, g, b = r * 0.5, g * 0.5, b * 0.5
            if abs(lx) < 0.018 and abs(ly + hy * 0.25) < 0.012:
                r, g, b = 0.75, 0.72, 0.62
                spec = 2.0
        if ax == 1:
            r, g, b = r * 1.1, g * 1.1, b * 1.1
        gloss = 0.1
        spec = max(spec, 0.6)
    elif mat == M_JACKET:
        # navy school blazer thrown over a chair back: soft vertical drape folds, a lighter worn crease
        fo = vnoise(lz * 22.0 + var * 9.0, ly * 1.5) * 0.6 + vnoise(lz * 55.0 - var * 3.0, ly * 3.0) * 0.4
        fo = _ss(0.25, 0.85, fo)
        r, g, b = 0.11 + 0.13 * fo, 0.13 + 0.14 * fo, 0.26 + 0.18 * fo
        # sewn seams / hem: a thin darker line near the bottom and a paler worn crease line
        if ax != 1 and abs(ly + hy * 0.82) < 0.004:
            r, g, b = r * 0.6, g * 0.6, b * 0.65
        if ax == 1:
            r, g, b = r * 1.15, g * 1.15, b * 1.12
        # a button or two on the hanging front
        for kb_ in range(2):
            bdz = lz - 0.06
            bdy = ly - (0.02 - kb_ * 0.09)
            if ax == 0 and bdz * bdz + bdy * bdy < 0.009 ** 2:
                r, g, b = 0.75, 0.62, 0.3
                spec = 1.5
        gloss = 0.03
        spec = max(spec, 0.2)
    elif mat == M_NOTE:
        k = int(var * 5.0)
        if k == 0:
            r, g, b = 0.25, 0.42, 0.75
        elif k == 1:
            r, g, b = 0.85, 0.35, 0.3
        elif k == 2:
            r, g, b = 0.35, 0.62, 0.45
        elif k == 3:
            r, g, b = 0.95, 0.78, 0.3
        else:
            r, g, b = 0.72, 0.5, 0.75
        if ax == 1 and sg > 0:
            # white name label on the cover
            if abs(lx + hx * 0.35) < hx * 0.22 and abs(lz) < hz * 0.55:
                r, g, b = 0.95, 0.94, 0.9
                if line_f(lx, 0.012, 0.0015, foot) > 0.5:
                    r, g, b = 0.6, 0.6, 0.62
            spec = 0.6
            gloss = 0.1
        elif ax != 1:
            # page block edge
            yy = (ly + hy) / (2 * hy)
            if yy > 0.12 and yy < 0.88:
                r, g, b = 0.94, 0.92, 0.86
                r -= 0.06 * line_f(ly, 0.0012, 0.0004, foot)
    elif mat == M_PCASE:
        k = int(var * 3.0)
        if k == 0:
            r, g, b = 0.92, 0.55, 0.62
        elif k == 1:
            r, g, b = 0.72, 0.68, 0.6
        else:
            r, g, b = 0.45, 0.78, 0.72
        fpp = max(foot, 1e-4)
        if ly > 0.0:
            # zip running along the top: dark tape, tiny pale teeth, a metal pull near one end
            zt = _clamp((0.003 - abs(lx + 0.004)) / fpp + 0.5, 0.0, 1.0)
            teeth = 0.5 + 0.5 * math.sin(lz * 1400.0)
            zr = 0.1 + 0.7 * teeth * _clamp(1.0 - foot / 0.0015, 0.0, 1.0) + 0.25 * _clamp(foot / 0.0015 - 0.5, 0.0, 1.0)
            r, g, b = r + (zr - r) * zt, g + (zr - g) * zt, b + (zr * 1.05 - b) * zt
            if abs(lz - hz * 0.62) < 0.007 and lx > -0.002 and lx < 0.009:
                r, g, b = 0.78, 0.76, 0.7
                spec = 3.0
            # painted specular strip along the lit shoulder of the vinyl
            stp = _ss(0.25, 0.4, lx / hx) * _ss(0.72, 0.55, lx / hx) * _ss(hy * 0.2, hy * 0.6, ly)
            r, g, b = r + (1.0 - r) * 0.45 * stp, g + (0.97 - g) * 0.45 * stp, b + (0.92 - b) * 0.45 * stp
        else:
            r, g, b = r * 0.8, g * 0.8, b * 0.82          # underside seam half, a little grubbier
        gloss = 0.22
        spec = max(spec, 2.0)
    elif mat == M_ERASER:
        if ly < -hy * 0.1:
            # felt pad, caked with chalk
            n = vnoise(lz * 120.0, lx * 120.0 + ly * 80.0)
            r, g, b = 0.62 + 0.25 * n, 0.62 + 0.25 * n, 0.63 + 0.24 * n
        else:
            r, g, b = 0.2, 0.36, 0.7
            if ax == 1:
                n = vnoise(lz * 60.0, lx * 60.0)
                c_ = _ss(0.45, 0.9, n) * 0.5
                r, g, b = r + (0.9 - r) * c_, g + (0.9 - g) * c_, b + (0.9 - b) * c_
            gloss = 0.15
            spec = 0.8
    elif mat == M_CHALK:
        k = int(var * 4.0)
        if k == 0 or k == 1:
            r, g, b = 0.96, 0.95, 0.92
        elif k == 2:
            r, g, b = 0.96, 0.86, 0.35
        else:
            r, g, b = 0.95, 0.45, 0.42
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
            c = samp(BT, (pz - 0.0) * bt_ppm, (HC - py) * bt_ppm, 0)
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
        if ax == 1 and sg > 0:
            n = vnoise(pz * 40.0, px * 90.0) * 0.6 + vnoise(pz * 150.0, px * 200.0) * 0.4
            c_ = _ss(0.3, 0.8, n) * 0.8
            r, g, b = r + (0.93 - r) * c_, g + (0.93 - g) * c_, b + (0.92 - b) * c_
        gloss = 0.2
        spec = 1.0
    elif mat == M_PLATFORM:
        r, g, b = 0.52, 0.34, 0.2
        gloss = 0.35
        spec = 1.0
    elif mat == M_LECTERN:
        if var < 0.2:
            # plinth: dark painted kick plate
            r, g, b = 0.26, 0.2, 0.18
        elif ax == 1:
            r, g, b = grain(lz, lx, var + 0.77, foot, 1.0, hz, hx, 1)
        elif hy < 0.03:
            # top board edge: laminated lip
            r, g, b = 0.82, 0.62, 0.4
        else:
            lw = lz if ax == 0 else lx
            hw = hz if ax == 0 else hx
            r, g, b = grain(ly, lw, 0.41 + var, foot, 1.0, hy, hw, 1)
            # two raised-and-fielded panels per face: frame stiles/rails, a groove with a shadowed
            # upper/left lip and a lit lower/right lip
            fw_ = 0.06
            pw_ = (hw - 1.5 * fw_) * 0.5
            pc = (fw_ * 0.25 + pw_) if lw > 0 else -(fw_ * 0.25 + pw_)
            du_ = pw_ - abs(lw - pc)
            dv_ = hy - fw_ - abs(ly)
            ew = min(du_, dv_)
            fpp = max(foot, 1e-4)
            if ew > 0.012:
                r, g, b = r * 0.9, g * 0.88, b * 0.88                  # fielded panel, a touch darker
            elif ew > -0.004:
                # groove: which side are we on?
                top_or_left = (dv_ < du_ and ly > 0) or (du_ <= dv_ and lw - pc < 0)
                if top_or_left:
                    r, g, b = r * 0.45, g * 0.42, b * 0.45
                else:
                    r, g, b = r + (1.0 - r) * 0.55, g + (0.85 - g) * 0.55, b + (0.64 - b) * 0.55
        r, g, b = r * 0.94, g * 0.9, b * 0.88
        gloss = 0.16
        spec = 1.0
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
    out[18] = 0.0
    out[19] = 0.0
    out[20] = 0.0
    if kind == 0:
        r, g, b = env(ox, oy, oz, dx, dy, dz, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1, P[11])
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
    tube = False
    tube_s = 0.0
    tube_side = 1.0
    if kind == 1:
        if idx == 0:
            nx, ny, nz, mat = 0.0, 1.0, 0.0, M_FLOOR
        elif idx == 1:
            nx, ny, nz, mat = 0.0, -1.0, 0.0, M_CEIL
        elif idx == 2:
            nx, ny, nz, mat = 1.0, 0.0, 0.0, M_WALL
        elif idx == 3:
            nx, ny, nz, mat = -1.0, 0.0, 0.0, M_FRONTWALL
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
        shp = int(B[idx, 11])
        if shp > 0:
            # SDF shape: gradient normal; ax / sg = dominant local axis for the material lookups
            e_ = 4e-4
            h0, h1, h2 = B[idx, 3], B[idx, 4], B[idx, 5]
            lnx = shape_sdf(shp, lx + e_, ly, lz, h0, h1, h2) - shape_sdf(shp, lx - e_, ly, lz, h0, h1, h2)
            lny = shape_sdf(shp, lx, ly + e_, lz, h0, h1, h2) - shape_sdf(shp, lx, ly - e_, lz, h0, h1, h2)
            lnz = shape_sdf(shp, lx, ly, lz + e_, h0, h1, h2) - shape_sdf(shp, lx, ly, lz - e_, h0, h1, h2)
            nl = math.sqrt(lnx * lnx + lny * lny + lnz * lnz) + 1e-12
            lnx /= nl
            lny /= nl
            lnz /= nl
            if abs(lnx) >= abs(lny) and abs(lnx) >= abs(lnz):
                ax, sg = 0, (1.0 if lnx > 0 else -1.0)
            elif abs(lny) >= abs(lnz):
                ax, sg = 1, (1.0 if lny > 0 else -1.0)
            else:
                ax, sg = 2, (1.0 if lnz > 0 else -1.0)
        elif mt != M_PAPER and mt != M_LIGHT and mt != M_CLOCK and mt != M_BOARD and mt != M_WALL:
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
                elif mt == M_BAG or mt == M_JACKET or mt == M_PCASE:
                    bw = 0.035
                elif mt == M_PILLAR:
                    bw = 0.02
                bw = min(bw, hh)
                e = hh - abs(cc)
                wood_ = mt == M_DESKTOP or mt == M_CHAIRWOOD
                if wood_ and a == 1 and cc < 0:
                    e = 1.0                      # painted: under-side edges stay soft / lost, no rim
                if e < bw:
                    kk = 1.0 - e / bw
                    if wood_:
                        kk = _ss(bw, bw * 0.55, e)   # a crisp flat bevel band, not a rounded gradient
                    tw = (kk * 1.35) * (1.0 if cc > 0 else -1.0)
                    if a == 0:
                        lnx += tw
                    elif a == 1:
                        lny += tw
                    else:
                        lnz += tw
            nl = math.sqrt(lnx * lnx + lny * lny + lnz * lnz)
            lnx /= nl
            lny /= nl
            lnz /= nl
        nx = c * lnx + s * lnz
        ny = lny
        nz = -s * lnx + c * lnz
        if (mt == M_DESKMETAL or mt == M_CHAIRMETAL) and shp == 0:
            # thin steel tube: replace the box normal by the true cylinder normal for this silhouette
            # position (so the rim streaks / glints sit where a painter puts them)
            h0, h1, h2 = B[idx, 3], B[idx, 4], B[idx, 5]
            la = -1
            if h0 < 0.026 and h2 < 0.026 and h1 > h0:
                la = 1
            elif h1 < 0.026 and h2 < 0.026 and h0 > h1:
                la = 0
            elif h0 < 0.026 and h1 < 0.026 and h2 > h0:
                la = 2
            if la >= 0:
                if la == 1:
                    axx, axy, axz, rr_ = 0.0, 1.0, 0.0, 0.5 * (h0 + h2)
                elif la == 0:
                    axx, axy, axz, rr_ = c, 0.0, -s, 0.5 * (h1 + h2)
                else:
                    axx, axy, axz, rr_ = s, 0.0, c, 0.5 * (h0 + h1)
                qx_, qy_, qz_ = px - B[idx, 0], py - B[idx, 1], pz - B[idx, 2]
                qa = qx_ * axx + qy_ * axy + qz_ * axz
                qx_ -= qa * axx
                qy_ -= qa * axy
                qz_ -= qa * axz
                # perp = axis x view, back = toward the camera, both perpendicular to the axis
                ppx = axy * dz - axz * dy
                ppy = axz * dx - axx * dz
                ppz = axx * dy - axy * dx
                pl_ = math.sqrt(ppx * ppx + ppy * ppy + ppz * ppz)
                if pl_ > 1e-4:
                    ppx /= pl_
                    ppy /= pl_
                    ppz /= pl_
                    da_ = dx * axx + dy * axy + dz * axz
                    bkx, bky, bkz = -(dx - da_ * axx), -(dy - da_ * axy), -(dz - da_ * axz)
                    bl_ = math.sqrt(bkx * bkx + bky * bky + bkz * bkz) + 1e-9
                    bkx /= bl_
                    bky /= bl_
                    bkz /= bl_
                    ts = _clamp((qx_ * ppx + qy_ * ppy + qz_ * ppz) / rr_, -1.0, 1.0)
                    cz_ = math.sqrt(max(0.0, 1.0 - ts * ts))
                    nx = ts * ppx + cz_ * bkx
                    ny = ts * ppy + cz_ * bky
                    nz = ts * ppz + cz_ * bkz
                    tube_s = ts
                    tube_side = 1.0 if (P[0] * ppx + P[1] * ppy + P[2] * ppz) >= 0.0 else -1.0
                    # which rim faces the room / sky light from the windows (cool reflection) -- opposite
                    tube = True
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
    rimb = 0.0
    if kind == 2 and mat == M_DESKMETAL and ax == 0 and sg < 0 and B[idx, 4] > 0.03 and B[idx, 4] < 0.06:
        # open book compartment under the desk top (faces the student): a dark recess with a lit lip
        hy_ = B[idx, 4]
        ez = hz - abs(lz) - 0.012
        ey = min(hy_ - ly - 0.004, ly + hy_ - 0.012)
        inside = _clamp(min(ez, ey) / max(foot, 1e-4) + 0.5, 0.0, 1.0)
        depthk = _clamp((hy_ - ly) / (2 * hy_), 0.0, 1.0)
        # dark cool blue-violet recess (never dead black), a touch lighter toward the lower lip
        ir, ig, ib = 0.30, 0.28, 0.52
        # painted gradient: black-violet under the desk top, lifting toward the compartment floor
        ic = 0.2 + 0.5 * depthk ** 1.8
        if depthk > 0.8:
            # the compartment floor seen through the opening: a lighter cool band
            ir, ig, ib = 0.46, 0.47, 0.62
            ic = 0.62
        # contents: some desks hold a stack of textbooks / notebooks / loose printouts
        hv = hash1(var * 57.0 + 3.0)
        if hv < 0.65:
            fill = 0.012 + 0.04 * hash1(var * 31.0)
            z0_ = -hz + 0.02 + 0.15 * hash1(var * 11.0)
            z1_ = hz - 0.02 - 0.15 * hash1(var * 13.0)
            yb = ly + hy_ - 0.012
            if yb < fill and lz > z0_ and lz < z1_:
                li = math.floor(yb / 0.0075)
                hl_ = hash1(li * 3.7 + var * 19.0)
                if hl_ < 0.45:
                    ir, ig, ib = 0.92, 0.9, 0.84             # page edges
                elif hl_ < 0.6:
                    ir, ig, ib = 0.3, 0.45, 0.8
                elif hl_ < 0.75:
                    ir, ig, ib = 0.85, 0.35, 0.3
                elif hl_ < 0.87:
                    ir, ig, ib = 0.95, 0.8, 0.35
                else:
                    ir, ig, ib = 0.4, 0.65, 0.45
                # layer seams + ragged book ends
                f_ = yb / 0.0075 - li
                if f_ < 0.12:
                    ir, ig, ib = ir * 0.45, ig * 0.45, ib * 0.5
                ic = 0.7
        cav = (1.0 - inside) + ic * inside
        ar = ar * (1 - inside) + ir * inside
        ag = ag * (1 - inside) + ig * inside
        ab = ab * (1 - inside) + ib * inside
        # warm floor bounce creeping over the lower lip into the opening
        rimb = inside * _ss(0.02, 0.0, ly + hy_ - 0.012) * 0.55
        # the opening's edge: a crisp lit line on the lower lip, a dark shadow line under the top lip
        fpp = max(foot, 1e-4)
        rimb += (1.0 - inside) * _clamp(1.6 - abs(ly + hy_ - 0.012 + 0.0015) / fpp, 0.0, 1.0) * 0.9
        topl = _clamp(1.6 - abs(ly - hy_ + 0.004 - 0.002) / fpp, 0.0, 1.0) * _ss(0.0, 0.01, ez)
        cav *= 1.0 - 0.6 * topl
        gloss *= 1 - inside
        spec *= 1 - inside
    wbn = 0.0
    if kind == 2 and mat == M_DESKMETAL and B[idx, 4] > 0.03 and B[idx, 3] > 0.1 and ax != 1 and not (ax == 0 and sg < 0):
        gy_ = (ly + hy) / (2 * hy)
        cav *= 0.62 + 0.55 * (1.0 - gy_)
        wbn = 0.3 * _ss(0.5, 0.0, gy_) * math.exp(-max(pz - 1.0, 0.0) / 3.0)
    # --- direct sun
    ndl = nx * Lx + ny * Ly + nz * Lz
    sv = 0.0
    if ndl > 0.0:
        sv = sun_vis(px, py, pz, nx, ny, nz, Lx, Ly, Lz, B, G, VB, HB, CM, cm_ppm, cm_y0, True) * ndl
        if tube:
            sv = sv / ndl * 0.45 * _ss(0.05, 0.2, ndl)      # flat painted lit band, not a CG gradient
    # --- ambient: cool skylight entering through the windows + occlusion
    fwin = 0.5 - 0.5 * nz
    # painted furniture: ambient is evaluated once per object (flat value masses per face, no
    # continuous CG gradients across a desk top / chair back)
    flat = kind == 2 and (mat == M_DESKTOP or mat == M_CHAIRWOOD or mat == M_BAG or mat == M_JACKET or
                          mat == M_NOTE or mat == M_PCASE or mat == M_BOOK or
                          (mat == M_DESKMETAL and B[idx, 4] > 0.03 and B[idx, 3] > 0.1))
    apz, apy = pz, py
    if flat:
        apz, apy = B[idx, 2], B[idx, 1]
    near = 0.44 + 0.86 * math.exp(-apz / 2.0)
    ao = 1.0
    if mat == M_FLOOR:
        ao = samp2(FAO, px * fao_ppm, pz * fao_ppm)
    else:
        ao = 1.0 - 0.35 * math.exp(-apy / 0.12) - 0.2 * math.exp(-(HC - apy) / 0.15)
        if kind == 2 and ny < -0.5:
            ao *= 0.45
    up = 0.85 + 0.15 * ny
    if kind == 2 and B[idx, 11] > 0.5 and B[idx, 11] < 1.5:
        up *= 0.72 + 0.45 * _clamp(0.5 + 0.5 * ly / hy, 0.0, 1.0)
    amb = (0.3 + 0.7 * fwin) * near * ao * up * cav
    if tube:
        # painted steel: a flat, fairly dark slate body so the thin rim streaks carry the form
        amb = 0.5 * near * (0.8 + 0.2 * ao)
    if pz < -0.01:
        amb = 1.6 * (0.6 + 0.4 * ny)
    # ambient tint: the upper room / deep room lean cool blue, beam sides cooler still
    tr, tg, tb = 1.0, 1.0, 1.0
    if py > 2.4:
        dk = 1.0 - math.exp(-pz / 3.0)
        tr, tg, tb = 0.94 - 0.1 * dk, 0.98 - 0.03 * dk, 1.0 + 0.08 * dk
        if abs(nx) > 0.5 and mat == M_PILLAR:
            tr, tg, tb = tr * 0.9, tg * 0.95, tb * 1.04
    if (mat == M_WALL or mat == M_FRONTWALL) and kind == 1:
        # counter-light: the walls facing the windows sit in cool shadow
        amb *= 0.8
        tr, tg, tb = tr * 0.9, tg * 0.97, tb * 1.1
    if mat == M_DESKTOP or mat == M_CHAIRWOOD:
        tr, tg, tb = tr * 1.42, tg * 1.12, tb * 0.8
    # warm bounce from the sunlit floor / desks onto downward-facing surfaces (ceiling, beam soffits,
    # desk undersides): strong near the windows, fading deeper into the room
    dn = max(0.0, -ny)
    bnc = P[10] * dn * (0.35 + 0.65 * math.exp(-abs(apz - 2.5) / 2.5)) * ao * cav
    if py > 2.4:
        xk = math.exp(-((px - 5.2) / 3.2) ** 2)
        bnc = dn * (0.8 * math.exp(-pz / 1.6) + 0.22 * math.exp(-((pz - 3.0) / 1.4) ** 2)) * (0.5 + 0.5 * xk)
    if kind == 2 and py < 2.0:
        # furniture undersides catch the warm light bouncing off the sunlit floor boards (ochre)
        bnc += 0.55 * dn * math.exp(-py / 1.4) * (0.5 + 0.5 * cav)
    lr = sunr * sv * cav + ambr * amb * tr + bnc * 1.0
    lg = sung * sv * cav + ambg * amb * tg + bnc * 0.6
    lb = sunb * sv * cav + ambb * amb * tb + bnc * 0.36
    if pz > 0.0 and py < 2.2 and (kind == 2 or mat == M_FLOOR):
        # cool blue-violet fill in the lower room: shadows keep a readable, saturated form
        vfk = 0.3 * (0.55 + 0.45 * cav)
        if tube:
            vfk *= 0.35
        lr += 0.34 * vfk
        lg += 0.3 * vfk
        lb += 0.62 * vfk
    r = ar * (lr + em)
    g = ag * (lg + em)
    b = ab * (lb + em * 1.05)
    if wbn > 0.0:
        r += ar * wbn * 1.0
        g += ag * wbn * 0.62
        b += ab * wbn * 0.4
    if rimb > 0.0:
        r += rimb * 0.42
        g += rimb * 0.24
        b += rimb * 0.1
    # crisp 1-2 px warm edge highlight on the lit top edges of desk tops / chair backs (painted rim)
    if kind == 2 and level == 0 and mat == M_JACKET and ax != 1:
        # soft lit rim along the top of the jacket's hanging panels (window light over the shoulders)
        hl = _clamp(1.0 - (hy - ly) / (3.0 * t / fpx), 0.0, 1.0)
        r += hl * 0.22
        g += hl * 0.22
        b += hl * 0.3
    if kind == 2 and level == 0 and mat == M_LECTERN and var > 0.2:
        # crisp lit lines along the lectern's top edges and the top board's lip
        fp = t / fpx
        e = 1.0
        if ax == 1 and sg > 0:
            e = min(hx - abs(lx), hz - abs(lz))
        elif ax != 1:
            e = hy - ly
        hl = _clamp(1.6 - e / fp, 0.0, 1.0)
        r += hl * 0.5
        g += hl * 0.38
        b += hl * 0.26
    if kind == 2 and level == 0 and (mat == M_DESKTOP or mat == M_CHAIRWOOD):
        e = 1.0
        kh = 1.0
        if mat == M_DESKTOP:
            if ax == 1 and sg > 0:
                # every top edge gets a crisp lit line; strongest on the sun / window side (local -z)
                # and the far (+x) edge, which read as a backlit rim against the room
                e = hx + lx                     # student-side (front) edge
                kh = 0.8
                if hx - lx < e:
                    e = hx - lx
                    kh = 1.0
                if hz + lz < e:
                    e = hz + lz
                    kh = 1.25
                if hz - lz < e:
                    e = hz - lz
                    kh = 0.4
            elif ax == 0 and sg < 0:
                e = hy - ly
                fp_ = t / fpx
                dl = _clamp(1.8 - (ly + hy) / fp_, 0.0, 1.0)
                r *= 1.0 - 0.55 * dl
                g *= 1.0 - 0.6 * dl
                b *= 1.0 - 0.6 * dl
        elif ax == 0 and hy > 0.05:
            e = -back_outline(ly, lz, hy, hz)     # chair back: arched top edge + rounded corners
            kh = _ss(-0.02, 0.05, ly)
        elif ax == 1 and sg > 0:
            e = min(hx - abs(lx), hz - abs(lz))
            kh = 0.4
        fp = t / fpx
        hl = _clamp(1.6 - e / fp, 0.0, 1.0)
        if hl > 0.0:
            lit = sv / ndl if (ndl > 0.0 and sv > 0.0) else 0.0
            if lit <= 0.0:
                lit = sun_vis(px, py + 0.004, pz, 0.0, 1.0, 0.0, Lx, Ly, Lz, B, G, VB, HB, CM, cm_ppm, cm_y0, True)
            k_ = hl * kh * (0.36 + 1.8 * lit)
            r += k_ * 1.0
            g += k_ * 0.8
            b += k_ * 0.52
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
    if kind == 2 and mat == M_DESKTOP and ax == 1 and sg > 0 and sv > 0.0 and level == 0:
        rdx = dx + 2 * ndv * nx
        rdy = dy + 2 * ndv * ny
        rdz = dz + 2 * ndv * nz
        cs = rdx * Lx + rdy * Ly + rdz * Lz
        shn = math.exp((cs - 1.0) * 7.0) * 0.55 * sv * (0.6 + 0.4 * vnoise(lz * 3.0 + var * 9.0, lx * 5.0))
        r += shn * 1.0
        g += shn * 0.72
        b += shn * 0.4
    # steel: crisp reflection of the bright window panes (a hard specular streak running along each
    # tube, the sash bars read as breaks in it)
    if (mat == M_DESKMETAL or mat == M_CHAIRMETAL or mat == M_TRAY or mat == M_PCASE) and pz > 0.0 and not tube:
        rdx = dx + 2 * ndv * nx
        rdy = dy + 2 * ndv * ny
        rdz = dz + 2 * ndv * nz
        if rdz < -0.05:
            tw = -pz / rdz
            xw = px + rdx * tw
            yw = py + rdy * tw
            if B[idx, 4] > B[idx, 3] and B[idx, 4] > B[idx, 5]:
                yw = 1.75 + (yw - 1.75) * 0.12          # vertical tube: anisotropic streak along it
            a = aperture(xw, yw, 0.03 + tw * 0.01, VB, HB)
            if a > 0.0:
                fr_ = 0.35 + 0.65 * (1 - max(ndv, 0.0)) ** 3
                # warmer / hotter toward the sun
                cs = rdx * Lx + rdy * Ly + rdz * Lz
                hot = math.exp((cs - 1.0) * 12.0)
                kk = a * fr_ * (0.9 + 2.5 * hot) * math.exp(-tw / 6.0) * (0.8 if tube else 1.6)
                kk *= 0.1 + 0.9 * _ss(0.0, 0.2, sv)
                if mat == M_PCASE:
                    kk = a * fr_ * (0.6 + 1.5 * hot) * 0.9
                r += kk * (0.95 + 0.3 * hot)
                g += kk * (0.97 + 0.1 * hot)
                b += kk * (1.0 - 0.2 * hot)
    if tube and level == 0:
        # painted steel: a hard thin warm streak on the sun-side rim of every tube (white-gold and hot
        # where the tube stands in a sun shaft), a cooler sky-blue streak on the other side, and a warm
        # floor bounce creeping up the bottom of the legs
        ws = tube_s * tube_side
        fpx_ = t / fpx
        rrw = max(B[idx, 3], B[idx, 5]) if B[idx, 4] > B[idx, 3] else B[idx, 4]
        soft = _clamp(fpx_ / max(rrw, 1e-4), 0.05, 0.6)          # widen the band when the tube is thin on screen
        band = _ss(0.74 - soft, 0.8, ws) * (1.0 - 0.3 * _ss(0.97, 1.0, ws))
        if band > 0.0:
            svp = sun_vis(px, py, pz, nx, ny, nz, Lx, Ly, Lz, B, G, VB, HB, CM, cm_ppm, cm_y0, True)
            kw = band * (0.32 + 2.4 * svp)
            r += kw * 1.0
            g += kw * (0.58 + 0.24 * svp)
            b += kw * (0.26 + 0.28 * svp)
            if svp > 0.05:
                core = _ss(0.72, 0.9, ws) * svp * 3.0 * (0.7 + 0.3 * vnoise(py * 9.0 + px * 5.0, pz * 4.0))
                r += core
                g += core * 0.93
                b += core * 0.78
        bandc = _ss(0.38 - soft * 0.5, 0.48, -ws) * _ss(0.7, 0.6, -ws)
        if bandc > 0.0:
            kc = bandc * 0.3 * (0.6 + 0.4 * near)
            r += kc * 0.55
            g += kc * 0.7
            b += kc * 1.0
        # warm bounce off the floor at the foot of the legs + a lighter lower-body value
        fb = math.exp(-py / 0.22) * (0.25 + 0.75 * _ss(-0.2, 0.6, -ws * 0.5 + 0.5))
        r += ar * fb * 0.5
        g += ag * fb * 0.3
        b += ab * fb * 0.14
    if mat == M_FLOOR and level == 0 and dy < 0.0:
        rdx = dx + 2 * ndv * nx
        rdy = dy + 2 * ndv * ny
        rdz = dz + 2 * ndv * nz
        if rdz < -0.05:
            tw = -pz / rdz
            xw = px + rdx * tw
            yw = py + rdy * tw
            # polished boards: the window grid mirrored, smeared a little along the view (vertical)
            a = 0.0
            for kq in range(3):
                a += aperture(xw, yw + (kq - 1) * 0.07 * tw, 0.012 + tw * 0.006, VB, HB)
            a /= 3.0
            if a > 0.0:
                fr_ = 0.05 + 0.75 * (1.0 - max(ndv, 0.0)) ** 3.5
                # is the reflected ray blocked by furniture? (cheap: only near the floor)
                tb_, bi_, bax_, bsg_ = trace_boxes(px, py + 1e-3, pz, rdx, rdy, rdz, B, G, tw, True)
                if bi_ < 0:
                    # goes to the glossy buffer (smeared along the view like the painted reflections)
                    kk = a * fr_ * 5.0 * math.exp(-tw / 10.0)
                    out[18] = kk * 1.05
                    out[19] = kk * 0.98
                    out[20] = kk * 0.92
            # the sun itself: a long, thin vertical streak on the varnish
            tws = -pz / Lz
            xs_ = px + Lx * tws
            ys_ = py + Ly * tws
            da = (xw - xs_) / tw
            de = (yw - ys_) / tw
            st = math.exp(-(da / 0.012) ** 2 - (de / 0.16) ** 2) + 0.25 * math.exp(-(da / 0.03) ** 2 - (de / 0.35) ** 2)
            if st > 0.01 and sv > 0.0:
                ks = st * 2.6 * sv / max(ndl, 0.05)
                r += ks * 1.0
                g += ks * 0.8
                b += ks * 0.55
    out[0] = r
    out[1] = g
    out[2] = b
    gk = 0.35 if tube else 1.0
    out[3] = ar * gk
    out[4] = ag * gk
    out[5] = ab * gk
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
        out = np.zeros(21, np.float64)
        out2 = np.zeros(21, np.float64)
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
                # contact reflections: fade with distance above the floor (legs smear out, then vanish)
                fd = 1.0
                if out[10] == 0.0:
                    fd = 0.15 + 0.85 * math.exp(-out2[9] / 0.5)
                for c in range(3):
                    refl[yy, xx, c] = out2[c] * fd + out[18 + c] / out[11]


@njit(parallel=True, fastmath=True, cache=True)
def render_aa(W, H, cam, fpx, B, G, VB, HB, CM, cm_ppm, cm_y0, SKY, sky_ppu, su0, sv1, TREE, tppm, tx0, ty1,
              BT, bt_ppm, FAO, fao_ppm, P, pys, pxs, col, alb, direct, rw):
    """Re-shade edge pixels with 4 extra rotated-grid samples (averaged with the centre sample)."""
    offs = np.array([[-0.125, -0.375], [0.375, -0.125], [0.125, 0.375], [-0.375, 0.125]])
    ox, oy, oz = cam[0], cam[1], cam[2]
    for k in prange(pys.shape[0]):
        yy, xx = pys[k], pxs[k]
        out = np.zeros(21, np.float64)
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
