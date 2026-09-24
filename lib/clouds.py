"""Dome-based painted cloud engine (used by sky.cumulus_plate / sky.horizon_bank).

Model
-----
A cloud is a hierarchy of 3D ellipsoidal DOMES (cauliflower lobes): big billows (~0.2 of the cloud
width) stacked into towers under a designed envelope, medium lobes (~0.07) sitting on their surfaces,
small bumps (~0.025) mostly on the silhouette. Sizes shrink upward and toward the cloud's sides.

Rendering is a hard z-buffer over all domes (supersampled): every pixel belongs to exactly ONE dome
(the frontmost surface), so where a dome overlaps the dome behind it there is a crisp edge. Each pixel
is lit with N.L where N blends the dome's own sphere normal, its parent's normal and the whole
cloud's form normal (keeps the shadow side one large coherent region while each dome still gets its
own lit cap and a soft terminator). There is no noise in the shading at all -> no blotches/islands.

Bases are clipped flat and extended with a horizontally torn wisp skirt; cumulonimbus anvils are
flat sheared plates with a smooth clipped underside and a fibrous downwind edge.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from . import core as C

# sphere table columns
SC = dict(cx=0, cy=1, cz=2, r=3, ay=4, clip=5, skd=6, cid=7, lev=8, pcx=9, pcy=10, pcz=11, pr=12, pay=13,
          fib=14, tone=15)
NS = 16
# cloud table columns
CC = dict(fx=0, fy=1, fz=2, frx=3, fry=4, frz=5, Lx=6, Ly=7, Lz=8, base=9, h=10, fog=11, back=12)
NC = 13
# output channels
OUT = dict(a=0, l=1, ny=2, v=3, fog=4, back=5, depth=6, rimf=7, fib=8, skirt=9, lf=10, tone=11)
NO = 12

# normal blend weights per level: (own, parent, form)
_WTS = np.array([[0.5, 0.0, 0.5], [0.42, 0.33, 0.25], [0.4, 0.35, 0.25], [0.4, 0.35, 0.25]], np.float64)


@njit(cache=True, fastmath=True, error_model='numpy')
def _zbuffer(Hs, Ws, S):
    n = S.shape[0]
    h1 = np.full((Hs, Ws), -1e12, np.float32)
    h2 = np.full((Hs, Ws), -1e12, np.float32)
    i1 = np.full((Hs, Ws), -1, np.int32)
    i2 = np.full((Hs, Ws), -1, np.int32)
    c1 = np.zeros((Hs, Ws), np.float32)
    cov = np.zeros((Hs, Ws), np.float32)
    sk = np.zeros((Hs, Ws), np.float32)
    ski = np.full((Hs, Ws), -1, np.int32)
    for i in range(n):
        cx, cy, cz, r, ay = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4]
        clip, skd = S[i, 5], S[i, 6]
        ry = r * ay
        mr = min(r, ry)
        x0 = max(0, int(cx - r - 2.0))
        x1 = min(Ws, int(cx + r + 3.0))
        y0 = max(0, int(cy - ry - 2.0))
        y1 = min(Hs, int(min(cy + ry, clip + skd) + 3.0))
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / ry
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                d2 = dx * dx + dy * dy
                if d2 > 1.3:
                    continue
                d = math.sqrt(d2)
                sd = (1.0 - d) * mr
                yy = y + 0.5
                if yy > clip:
                    if skd > 0.0 and sd > 0.0:
                        s = (1.0 - (yy - clip) / skd) * min(sd / (0.25 * mr + 1e-3), 1.0)
                        if s > sk[y, x]:
                            sk[y, x] = s
                            ski[y, x] = i
                    continue
                sdc = clip - yy
                if sdc < sd:
                    sd = sdc
                c = sd + 0.5
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                if c > cov[y, x]:
                    cov[y, x] = c
                h = cz + r * math.sqrt(max(1.0 - d2, 0.0))
                if h > h1[y, x]:
                    h2[y, x] = h1[y, x]
                    i2[y, x] = i1[y, x]
                    h1[y, x] = h
                    i1[y, x] = i
                    c1[y, x] = c
                elif h > h2[y, x]:
                    h2[y, x] = h
                    i2[y, x] = i
    return h1, i1, c1, i2, cov, sk, ski


@njit(cache=True, fastmath=True, error_model='numpy')
def _normal(S, Cl, W8, i, x, y):
    """Blended normal (own dome / parent dome / whole-cloud form) and surface z at (x, y) on dome i.
    Returns mx, my, mz, z, nx, ny (own normal xy)."""
    cx, cy, cz, r, ay = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4]
    cid = int(S[i, 7])
    lev = min(int(S[i, 8]), 3)
    dx = (x - cx) / r
    dy = (y - cy) / (r * ay)
    d2 = min(dx * dx + dy * dy, 1.0)
    dz = math.sqrt(1.0 - d2)
    z = cz + r * dz
    nx, ny, nz = dx, dy / ay, dz
    ln = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    pr = S[i, 12]
    if pr > 0.0:
        px = (x - S[i, 9]) / pr
        py = (y - S[i, 10]) / (pr * S[i, 13]) / S[i, 13]
        pz = (z - S[i, 11]) / pr
        lp = math.sqrt(px * px + py * py + pz * pz) + 1e-6
        px, py, pz = px / lp, py / lp, pz / lp
    else:
        px, py, pz = nx, ny, nz
    fx = (x - Cl[cid, 0]) / (Cl[cid, 3] * Cl[cid, 3])
    fy = (y - Cl[cid, 1]) / (Cl[cid, 4] * Cl[cid, 4])
    fz = (z - Cl[cid, 2]) / (Cl[cid, 5] * Cl[cid, 5])
    lf = math.sqrt(fx * fx + fy * fy + fz * fz) + 1e-12
    fx, fy, fz = fx / lf, fy / lf, fz / lf
    wa, wb, wc = W8[lev, 0], W8[lev, 1], W8[lev, 2]
    mx = wa * nx + wb * px + wc * fx
    my = wa * ny + wb * py + wc * fy
    mz = wa * nz + wb * pz + wc * fz
    # flat clipped underside faces down
    sdc = S[i, 5] - y
    bw = (0.12 + 0.3 * max(0.6 - ay, 0.0)) * r * ay * 2.0 + 1.0
    if sdc < bw:
        kk = 1.0 - max(sdc, 0.0) / bw
        kk = kk * kk
        mx = mx * (1.0 - kk)
        my = my * (1.0 - kk) + 1.0 * kk
        mz = mz * (1.0 - kk) + 0.25 * kk
    lm = math.sqrt(mx * mx + my * my + mz * mz) + 1e-6
    return mx / lm, my / lm, mz / lm, z, nx * (1.0 - dz), ny * (1.0 - dz)


@njit(cache=True, fastmath=True, error_model='numpy')
def _accum_normals(Hs, Ws, S, Cl, W8, h1, soft):
    """Soft-max blend of the normals of all domes whose surface is within ~soft of the frontmost one:
    intersection creases between lobes become smooth fillets, occluding contours stay crisp."""
    n = S.shape[0]
    N = np.zeros((Hs, Ws, 3), np.float32)
    wsum = np.zeros((Hs, Ws), np.float32)
    for i in range(n):
        cx, cy, cz, r, ay = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4]
        clip = S[i, 5]
        ry = r * ay
        x0 = max(0, int(cx - r - 2.0))
        x1 = min(Ws, int(cx + r + 3.0))
        y0 = max(0, int(cy - ry - 2.0))
        y1 = min(Hs, int(min(cy + ry, clip) + 3.0))
        for y in range(y0, y1):
            yy = y + 0.5
            if yy > clip + 0.5:
                continue
            dy = (yy - cy) / ry
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                d2 = dx * dx + dy * dy
                if d2 >= 1.0:
                    continue
                h = cz + r * math.sqrt(1.0 - d2)
                if h1[y, x] < -1e11:
                    continue
                e = min((h - h1[y, x]) / soft, 0.0)
                if e < -6.0:
                    continue
                w = math.exp(e)
                mx, my, mz, z, a_, b_ = _normal(S, Cl, W8, i, x + 0.5, yy)
                N[y, x, 0] += w * mx
                N[y, x, 1] += w * my
                N[y, x, 2] += w * mz
                wsum[y, x] += w
    return N, wsum


@njit(cache=True, fastmath=True, error_model='numpy')
def _shade1(S, Cl, W8, i, x, y, out, N, wsum, yi, xi):
    """Shade sub-sample (x, y) on dome i -> out[1:NO]."""
    cid = int(S[i, 7])
    mx, my, mz, z, rnx, rny = _normal(S, Cl, W8, i, x, y)
    ws = wsum[yi, xi]
    if ws > 1e-9:
        bx, by, bz = N[yi, xi, 0] / ws, N[yi, xi, 1] / ws, N[yi, xi, 2] / ws
        lb = math.sqrt(bx * bx + by * by + bz * bz) + 1e-6
        mx, my, mz = bx / lb, by / lb, bz / lb
    Lx, Ly, Lz = Cl[cid, 6], Cl[cid, 7], Cl[cid, 8]
    l = mx * Lx + my * Ly + mz * Lz
    fx = (x - Cl[cid, 0]) / (Cl[cid, 3] * Cl[cid, 3])
    fy = (y - Cl[cid, 1]) / (Cl[cid, 4] * Cl[cid, 4])
    fz = (z - Cl[cid, 2]) / (Cl[cid, 5] * Cl[cid, 5])
    lf = math.sqrt(fx * fx + fy * fy + fz * fz) + 1e-12
    lform = (fx * Lx + fy * Ly + fz * Lz) / lf
    base, hh = Cl[cid, 9], Cl[cid, 10]
    v = (base - y) / hh
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    l2 = math.sqrt(Lx * Lx + Ly * Ly) + 1e-6
    rimf = max((rnx * Lx + rny * Ly) / l2, 0.0)
    out[1] = l
    out[2] = my
    out[3] = v
    out[4] = Cl[cid, 11]
    out[5] = Cl[cid, 12]
    out[6] = z - Cl[cid, 2]      # depth relative to the cloud (keeps float precision for AO)
    out[7] = rimf
    out[8] = S[i, 14]
    out[10] = lform
    out[11] = S[i, 15]


@njit(cache=True, fastmath=True, parallel=True, error_model='numpy')
def _resolve(H, W, ss, i1, cov, sk, ski, S, Cl, W8, N, wsum):
    out = np.zeros((H, W, NO), np.float32)
    for Y in prange(H):
        tmp = np.zeros(NO, np.float64)
        acc = np.zeros(NO, np.float64)
        for X in range(W):
            for k in range(NO):
                acc[k] = 0.0
            asum = 0.0
            sksum = 0.0
            for sy in range(ss):
                for sx in range(ss):
                    y = Y * ss + sy
                    x = X * ss + sx
                    fx_ = x + 0.5
                    fy_ = y + 0.5
                    a = cov[y, x]
                    i = i1[y, x]
                    if i >= 0:
                        _shade1(S, Cl, W8, i, fx_, fy_, tmp, N, wsum, y, x)
                        for k in range(1, NO):
                            acc[k] += tmp[k] * a
                        asum += a
                    s = sk[y, x] * (1.0 - a)
                    if s > 0.0 and ski[y, x] >= 0:
                        _shade1(S, Cl, W8, ski[y, x], fx_, fy_, tmp, N, wsum, y, x)
                        tmp[1] = min(tmp[1], tmp[10]) - 0.25
                        tmp[2] = 1.0
                        tmp[3] = 0.0
                        tmp[7] = 0.0
                        for k in range(1, NO):
                            acc[k] += tmp[k] * s
                        asum += s
                        sksum += s
            n2 = ss * ss
            if asum > 1e-6:
                for k in range(1, NO):
                    out[Y, X, k] = acc[k] / asum
            out[Y, X, 0] = asum / n2
            out[Y, X, 9] = sksum / n2
    return out


# ============================================================================ shape generation

class Builder:
    """Accumulates domes (sphere rows) and clouds (cloud rows)."""

    def __init__(self):
        self.S = []
        self.Cl = []

    def cloud(self, row):
        self.Cl.append(row)
        return len(self.Cl) - 1

    def dome(self, cx, cy, cz, r, ay, clip, skd, cid, lev, parent=None, fib=0.0, tone=0.0):
        if parent is None:
            p = (0.0, 0.0, 0.0, 0.0, 1.0)
        else:
            p = parent
        self.S.append((cx, cy, cz, r, ay, clip, skd, cid, lev, p[0], p[1], p[2], p[3], p[4], fib, tone))


def _envelope(rng, kind):
    """Top profile top(u) in 0..1 for u in [-1, 1] (fraction of the cloud height)."""
    if kind == 'stratocumulus':
        ms = rng.uniform(-0.8, 0.8, 5)
        ss = rng.uniform(0.15, 0.35, 5)
        As = rng.uniform(0.55, 1.0, 5)
        edge = 0.2
    else:
        m0 = rng.uniform(-0.3, 0.3)
        k = int(rng.integers(2, 4))
        ms = np.concatenate([[m0], rng.uniform(-0.75, 0.75, k)])
        ss = np.concatenate([[rng.uniform(0.35, 0.5)], rng.uniform(0.16, 0.3, k)])
        As = np.concatenate([[1.0], rng.uniform(0.35, 0.72, k)])
        edge = 0.4

    def top(u):
        u = np.asarray(u, np.float64)
        t = np.zeros_like(u)
        for m, s, A in zip(ms, ss, As):
            t = np.maximum(t, A * np.exp(-((u - m) / s) ** 2))
        return np.maximum(t, 0.22) * np.clip(1 - np.abs(u) ** 3, 0, 1) ** edge
    return top


def _children(B, rng, P, cid, lev, count, ratio, front, base_y, clip, skd, height, width, cx, minr, ay_mul=1.0,
              anvil=False):
    """Cover the visible surface of each parent dome in P (tuples (cx,cy,cz,r,ay,fib)) with `count`
    child lobes of radius ~ratio*parent (blue-noise spaced directions on the front hemisphere, biased
    up/outward, none on the underside). front: (dz_min, dz_max) z range of the growth directions."""
    kids = []
    for (px, py, pz, pr, pay, pfib) in P:
        vp = np.clip((base_y - py) / max(height, 1.0), 0, 1)
        up = 1.0 - 0.3 * vp
        side = 1.0 - 0.3 * min(abs(px - cx) / (0.5 * width), 1.0)
        rr = pr * ratio * up * side
        if rr < minr:
            continue
        n = int(count + rng.random())
        m = n * 8
        dz = rng.uniform(front[0], front[1], m)
        a = -math.pi / 2 + rng.normal(0, 0.62, m)                  # mostly on the crown, some on the flanks
        a = np.clip(a, -math.pi * 1.1, math.pi * 0.1)
        if anvil:
            a = rng.uniform(-math.pi * 1.02, math.pi * 0.02, m)
        s_ = np.sqrt(np.clip(1 - dz * dz, 0, 1))
        D = np.stack([np.cos(a) * s_, np.sin(a) * s_, dz], 1)
        chosen = []
        thr = math.cos(min(1.45 * ratio + 0.04, 1.0))
        for k in range(m):
            if len(chosen) >= n:
                break
            if all(float(D[k] @ D[c]) < thr for c in chosen):
                chosen.append(k)
        for k in chosen:
            ux, uy, uz = D[k]
            r_ = rr * rng.uniform(0.65, 1.3)
            reach = pr - r_ * rng.uniform(0.1, 0.35)
            x = px + ux * reach
            y = py + uy * reach * pay
            z = pz + uz * reach
            ay = min(max(pay * ay_mul, 0.25) * rng.uniform(0.92, 1.05), 1.0)
            if y + r_ * ay * 0.3 > base_y:
                continue
            B.dome(x, y, z, r_, ay, clip, skd * (r_ / pr), cid, lev, (px, py, pz, pr, pay), fib=pfib,
                   tone=rng.uniform(-1, 1))
            kids.append((x, y, z, r_, ay, pfib))
    return kids


def gen_cloud(B, rng, spec, L, fog, back, detail=1.0, z0=0.0, wind=1.0):
    """Build one cloud into Builder B.

    spec: (cx, base_y, width, height, dist, kind, flat) in px. L: (Lx, Ly, Lz) unit light vector toward the
    sun for this cloud. fog 0..1 atmospheric fade, back 0..1 backlit amount. Returns cloud id."""
    cx, base_y, width, height, dist, kind, flat = spec
    flat = float(flat)
    wind = 1.0 if wind >= 0 else -1.0
    cid = B.cloud([cx, base_y - height * 0.45, z0, width * 0.55, height * 0.6, width * 0.45, L[0], L[1], L[2],
                   base_y, height, fog, back])
    clip = base_y
    skd = max(height * 0.06, width * 0.02) * float(np.clip(1.4 - dist / 5.0, 0, 1))   # no torn wisps far away
    lev0 = []
    minr = max(width * 0.012, 1.2)
    if kind == 'cumulonimbus':
        # ---- tower: stacked big billows under a tall, slightly leaning, top-heavy envelope
        tw = width * 0.66                   # tower width
        lean = rng.uniform(0.03, 0.08) * width * wind
        anv_bot = base_y - height * 0.8
        R0 = tw * 0.24
        cols = []
        u = -0.95
        while u <= 0.95:
            cols.append(u)
            u += rng.uniform(0.3, 0.42)
        for u in cols:
            T = height * (0.92 - 0.62 * abs(u) ** 1.3 + rng.uniform(-0.07, 0.03))
            Rb = R0 * (1 - 0.3 * abs(u)) * rng.uniform(0.9, 1.1)
            y_top = base_y - T
            yc = base_y - Rb * 0.55
            while True:
                vv = (base_y - yc) / height
                R = Rb * (1 - 0.3 * vv) * rng.uniform(0.75, 1.2)
                xc = cx + u * tw * 0.5 * (1 + 0.6 * vv ** 2.5) + lean * vv + rng.uniform(-0.04, 0.04) * tw
                zc = z0 + math.sqrt(max(1 - u * u, 0)) * tw * 0.25 + rng.uniform(-0.1, 0.1) * R
                B.dome(xc, yc, zc, R, 0.92, clip, skd, cid, 0, None, tone=rng.uniform(-1, 1))
                lev0.append((xc, yc, zc, R, 0.92, 0.0))
                if yc - R <= y_top:
                    break
                yc -= R * rng.uniform(0.8, 1.0)
        # low shoulders at the base on both sides
        for sd in (-1, 1):
            for k in range(int(rng.integers(1, 3))):
                R = tw * rng.uniform(0.12, 0.17)
                xc = cx + sd * (tw * 0.5 + (k + 0.4) * R * 1.3)
                yc = base_y - R * rng.uniform(0.5, 0.8)
                if abs(xc - cx) > width * 0.6:
                    break
                zc = z0 - tw * 0.1
                B.dome(xc, yc, zc, R, 0.9, clip, skd, cid, 0, None)
                lev0.append((xc, yc, zc, R, 0.9, 0.0))
        # ---- anvil: flat sheared plate, smooth underside, fibrous downwind edge
        anv = []
        at = base_y - height
        thick = height * 0.16
        up0, up1 = -0.45, 0.62
        if wind < 0:
            up0, up1 = -0.62, 0.45
        n = 18
        for k in range(n):
            f = k / (n - 1)
            fd = f if wind > 0 else 1 - f
            fu = 1 - fd
            xc = cx + lean + width * (up0 + (up1 - up0) * f)
            dt = max(1 - abs(xc - (cx + lean)) / (width * 0.75), 0)
            taper = 1 - 0.75 * C.smoothstep(0.35, 1.0, fd) - 0.4 * C.smoothstep(0.5, 1.0, fu)
            R = width * (0.075 + 0.08 * dt) * rng.uniform(0.9, 1.1) * (0.6 + 0.4 * taper)
            ay = min(thick * (0.45 + 0.7 * dt) * taper / R, 0.6)
            ay = max(ay, 0.12)
            # plate top sheared slightly upward downwind; underside smooth, lifting toward the tips
            yc = at + R * ay + height * 0.02 * (1 - dt) - height * 0.025 * fd ** 2
            underside = anv_bot + height * 0.05 * (1 - dt) - height * 0.06 * fd ** 2
            fib = float(C.smoothstep(0.72, 1.0, fd))
            zc = z0 + width * 0.02
            B.dome(xc, yc, zc, R, ay, underside, 0.0, cid, 0, None, fib=fib)
            anv.append((xc, yc, zc, R, ay, fib))
        k1 = _children(B, rng, anv, cid, 1, 6 * detail, 0.4, (-0.1, 0.8), base_y, anv_bot + height * 0.03, 0, height,
                       width * 1.2, cx, minr, ay_mul=1.3, anvil=True)
        _children(B, rng, k1, cid, 2, 4 * detail, 0.38, (-0.15, 0.5), base_y, anv_bot + height * 0.03, 0, height,
                  width * 1.2, cx, minr, anvil=True)
        k1 = _children(B, rng, lev0, cid, 1, 13 * detail, 0.36, (-0.2, 0.62), base_y, clip, skd, height, tw, cx, minr)
        k2 = _children(B, rng, k1, cid, 2, 12 * detail, 0.35, (-0.2, 0.35), base_y, clip, skd, height, tw, cx, minr)
        if detail > 1.1:
            _children(B, rng, k2, cid, 3, 5, 0.36, (-0.2, 0.5), base_y, clip, skd, height, tw, cx, minr)
        return cid

    top = _envelope(rng, kind)
    if kind == 'stratocumulus':
        R0 = min(width * 0.12, height * 0.9)
        ayb = 0.55 * flat
    else:
        R0 = min(width * 0.2, height * 0.55)
        ayb = 0.95 * flat
    u = -0.95
    while u <= 0.95:
        T = top(u) * height
        Rb = R0 * (1 - 0.35 * abs(u)) * rng.uniform(0.85, 1.12)
        Rb = min(Rb, T / max(ayb, 0.3) * 0.62 + 1)
        y_top = base_y - T
        yc = base_y - Rb * ayb * 0.5
        while True:
            vv = (base_y - yc) / height
            R = max(Rb * (1 - 0.4 * vv), minr * 2) * rng.uniform(0.92, 1.06)
            xc = cx + u * width * 0.5 + rng.uniform(-0.04, 0.04) * width
            zc = z0 + math.sqrt(max(1 - u * u, 0)) * width * 0.18 + rng.uniform(-0.1, 0.1) * R
            B.dome(xc, yc, zc, R, ayb, clip, skd, cid, 0, None, tone=rng.uniform(-1, 1))
            lev0.append((xc, yc, zc, R, ayb, 0.0))
            if yc - R * ayb <= y_top:
                break
            yc -= R * ayb * rng.uniform(0.7, 0.9)
        u += (Rb / (width * 0.5)) * rng.uniform(0.8, 1.15)
    if kind == 'stratocumulus':
        k1 = _children(B, rng, lev0, cid, 1, 8 * detail, 0.42, (-0.1, 0.8), base_y, clip, skd, height, width, cx,
                       minr, ay_mul=1.0)
        _children(B, rng, k1, cid, 2, 5 * detail, 0.38, (-0.15, 0.5), base_y, clip, skd, height, width, cx, minr)
        return cid
    k1 = _children(B, rng, lev0, cid, 1, 13 * detail, 0.36, (-0.2, 0.62), base_y, clip, skd, height, width, cx, minr)
    k2 = _children(B, rng, k1, cid, 2, 12 * detail, 0.35, (-0.2, 0.35), base_y, clip, skd, height, width, cx, minr)
    if detail > 1.1:
        _children(B, rng, k2, cid, 3, 5, 0.36, (-0.2, 0.5), base_y, clip, skd, height, width, cx, minr)
    return cid


def render(B, W, H, ss=2, fillet=0.006):
    """Rasterise all domes of builder B into a (H, W, NO) attribute buffer (see OUT)."""
    if not B.S:
        return np.zeros((H, W, NO), np.float32)
    S = np.array(B.S, np.float64)
    Cl = np.array(B.Cl, np.float64)
    # scale geometry to the supersampled grid
    Ss = S.copy()
    for k in ('cx', 'cy', 'cz', 'r', 'clip', 'skd', 'pcx', 'pcy', 'pcz', 'pr'):
        Ss[:, SC[k]] *= ss
    Cs = Cl.copy()
    for k in ('fx', 'fy', 'fz', 'frx', 'fry', 'frz', 'base', 'h'):
        Cs[:, CC[k]] *= ss
    h1, i1, c1, i2, cov, sk, ski = _zbuffer(H * ss, W * ss, Ss)
    N, wsum = _accum_normals(H * ss, W * ss, Ss, Cs, _WTS, h1, np.float32(max(fillet * W * ss, 0.5)))
    out = _resolve(H, W, ss, i1, cov, sk, ski, Ss, Cs, _WTS, N, wsum)
    out[..., OUT['depth']] /= ss
    return out


# ============================================================================ painting

def _fblur(img, sigma):
    if sigma <= 3.0:
        return C.blur(img, sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.55)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def streaks(W, H, seed, xcells=3.0, ycells=90.0, octaves=3):
    """Horizontally stretched value noise in 0..1 (torn base wisps / fibres)."""
    w = max(int(W / 16), 8)
    rng = np.random.default_rng(seed)
    out = np.zeros((H, w), np.float32)
    amp, tot = 1.0, 0.0
    cx, cy = xcells, ycells
    for o in range(octaves):
        g = rng.random((int(cy) + 3, int(cx) + 3)).astype(np.float32)
        out += amp * cv2.resize(g, (w, H), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
        cx *= 2
        cy *= 1.6
    out /= tot
    out = cv2.resize(out, (W, H), interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


def paint(A, pal, sky=None, seed=0, rim=1.0, halo=1.0, terminator=0.0, soft=0.22, sun_pos=None, sun_dir=None,
          texture=1.0):
    """Colour an attribute buffer A (H, W, NO) from render() into a straight-alpha RGBA plate.

    pal: dict of RGB arrays (see sky.CLOUD_PALETTES). terminator: shifts the light/shadow split
    (+ = more lit). soft: half-width of the terminator gradient in N.L units (soft on the terminator,
    while dome overlaps and the silhouette stay crisp)."""
    H, W = A.shape[:2]
    a = np.clip(A[..., 0], 0, 1)
    l, ny, v, fog, back = A[..., 1], A[..., 2], A[..., 3], np.clip(A[..., 4], 0, 1), np.clip(A[..., 5], 0, 1)
    depth, rimf, fib, skirt, lform, tone = A[..., 6], A[..., 7], A[..., 8], A[..., 9], A[..., 10], A[..., 11]
    ins = (a > 1e-3).astype(np.float32)
    # ---- ambient occlusion from the depth buffer: recesses just behind a dome's edge darken
    def wb(x, s):
        return _fblur(x * ins, s) / (_fblur(ins, s) + 1e-4)
    d0 = wb(depth, 0.0035 * W)
    d1 = wb(depth, 0.008 * W)
    d2 = wb(depth, 0.02 * W)
    occ_line = C.smoothstep(0.3, 0.6, (d0 - depth) / (0.006 * W))       # crisp recess just under a lobe
    occ = np.clip((d1 - depth) / (0.014 * W), 0, 1) * 0.5 + np.clip((d2 - depth) / (0.04 * W), 0, 1) * 0.5
    occ = np.clip(np.maximum(occ, occ_line * 0.6), 0, 1)
    # thick lower body gets less direct light than the towering tops
    vfall = (1 - C.smoothstep(0.0, 0.75, v)) * 0.22
    le = l - 0.3 * occ + terminator - vfall
    # large-scale light (form + big billows): local lobes may only deviate +-dev from it, so small lobes
    # read as half-tone lines inside the light and as lighter tops inside the shadow - never beads
    lbig = wb(l, 0.012 * W) * 0.6 + lform * 0.4 + terminator - vfall
    dev = 0.33
    le = np.clip(le, lbig - dev, lbig + dev * 0.8)
    # ---- painted value structure: shadow | half-tone | lit | cap | hot highlight
    #      terminator soft (gradient), the steps inside the light crisper -> readable dome shapes
    t_half = C.smoothstep(-0.32 - soft * 0.4, -0.32 + soft * 0.4, le)
    t_lit = C.smoothstep(0.0 - soft * 0.5, 0.0 + soft * 0.5, le)
    t_cap = C.smoothstep(0.2, 0.3, le * 0.5 + lbig * 0.5)
    t_hi = C.smoothstep(0.45, 0.7, lbig)
    if sky is None:
        skyc = np.broadcast_to(pal['fill'], (H, W, 3))
    else:
        skyc = sky
    vv = v[..., None]
    up = np.clip(-ny * 0.7 + 0.3, 0, 1)
    up = (C.smoothstep(0.38, 0.5, up) * 0.75 + up * 0.25)[..., None]   # flat painted fill tones
    down = np.clip(ny, 0, 1)[..., None]                         # faces down -> bounce from below
    shd = C.lerp(pal['shd_low'], pal['shd_high'], np.clip(up * 0.7 + vv * 0.3, 0, 1))
    shd = C.lerp(shd, pal['bounce'], np.clip(down * 0.9 * (1 - vv * 1.5), 0, 0.7))
    shd = C.lerp(shd, pal['base'], (C.smoothstep(0.05, 0.0, v) * 0.35)[..., None])
    shd = C.lerp(shd, skyc, 0.15)
    half = C.lerp(C.lerp(pal['shd_high'], pal['mid'], 0.5), shd, 0.35)
    lit0 = C.lerp(pal['lit_low'], pal['lit_high'], C.smoothstep(0.0, 0.7, v)[..., None])
    lit_dim = C.lerp(lit0, pal['mid'], 0.3) * 0.97                    # lit but turned from the sun
    lit_cap = lit0 * pal['hdr']
    col = C.lerp(shd, half, t_half[..., None] * 0.7)
    col = C.lerp(col, lit_dim, t_lit[..., None])
    col = C.lerp(col, lit_cap, t_cap[..., None])
    col = C.lerp(col, pal['hi'] * pal['hdr'] * 1.04, (t_hi * 0.8)[..., None])
    # lobe shade lines inside the light: where a lobe turns away relative to its surroundings
    t_lobe = C.smoothstep(-0.1, -0.2, le - lbig) * t_lit
    col = C.lerp(col, C.lerp(lit0, pal['shd_high'], 0.3), (t_lobe * 0.75)[..., None])
    # terminator band: a narrow saturated tone between shadow and light (pink/peach at sunset)
    band = np.exp(-((le + 0.02) / (soft * 0.45 + 1e-3)) ** 2)[..., None]
    col = C.lerp(col, pal['mid'], band * 0.3)
    t_hi = t_cap
    # tiny per-dome tonal variation (<2%) smoothed within lobes
    col = col * (1 + 0.012 * wb(tone, 0.004 * W))[..., None]
    # occlusion also cools the lit side a little (recess colour)
    col = C.lerp(col, C.lerp(pal['shd_high'], pal['mid'], 0.3), (occ_line * 0.25 * t_lit)[..., None])
    if texture:
        n = C.fbm(W, H, 14, 3, seed=seed + 5) - 0.5
        col = col * (1 + 0.025 * texture * n)[..., None]
    # ---- backlit clouds: dark body, glowing edges
    bk = back[..., None]
    col = C.lerp(col, C.lerp(pal['shd_low'], skyc, 0.3) * 0.95, bk * (1 - t_lit[..., None]) * 0.6)
    # ---- silhouette: distance to the outside, outward direction
    m8 = (a > 0.5).astype(np.uint8)
    dist_in = cv2.distanceTransform(m8, cv2.DIST_L2, 5).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5).astype(np.float32)
    cg = _fblur(a, 0.004 * W)
    ogx = -cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx ** 2 + ogy ** 2) + 1e-6
    xs, ys = C.grid(W, H)
    if sun_pos is not None:
        sx_, sy_ = sun_pos[0] - xs, sun_pos[1] - ys
        sl = np.sqrt(sx_ * sx_ + sy_ * sy_) + 1e-3
        ex, ey = sx_ / sl, sy_ / sl
        prox = 0.35 + 0.65 * np.exp(-(sl / W) / 0.3)
    else:
        d = np.asarray(sun_dir if sun_dir is not None else (0.6, -0.7), np.float32)
        d = d / (np.linalg.norm(d) + 1e-6)
        ex, ey = d[0], d[1]
        prox = np.float32(0.7)
    oface = np.clip((ogx * ex + ogy * ey) / ogl, 0, 1)
    oface = C.smoothstep(0.1, 0.8, _fblur(oface * (ogl > 1e-4), 0.002 * W))
    rimc = pal['rim'] * pal['hdr'] * 1.15
    wr = max(0.0014 * W, 1.0)
    rim_line = np.exp(-dist_in / wr) * oface * prox * rim * (0.9 + 0.8 * back) * (1 - fog * 0.6)
    col = C.lerp(col, rimc, np.clip(rim_line, 0, 1)[..., None])
    # translucent glow creeping in from sun-facing edges (thin cloud transmits light)
    thin = np.exp(-dist_in / (0.01 * W)) * oface * prox * rim * (0.15 + 0.9 * back)
    thin = thin * (1 - 0.6 * t_lit)
    col = col + (rimc - col) * np.clip(thin, 0, 0.85)[..., None]
    # ---- atmospheric perspective
    col = C.lerp(col, pal['haze'], (fog * 0.8)[..., None])
    # ---- alpha: torn base wisps and fibrous anvil edge
    st = streaks(W, H, seed + 31, xcells=max(3.0, W / 500), ycells=H / 7.0)
    body = a - skirt
    wisp = skirt * C.smoothstep(0.55, 0.75, st + skirt * 0.5 - 0.15)
    st2 = C.blur(streaks(W, H, seed + 37, xcells=max(3.0, W / 400), ycells=H / 2.2), 0.6)
    fz = C.blur(fib, 0.004 * W)
    stc = C.smoothstep(0.15, 0.85, st2 * 0.6 + st * 0.4)
    # fibrous, wind-sheared veil: the body alpha smeared downwind, thinned and combed by the streaks
    kl = max(int(0.05 * W) | 1, 3)
    ker = np.zeros((1, kl), np.float32)
    ker[0, :kl // 2 + 1] = np.linspace(0.2, 1.0, kl // 2 + 1)
    ker /= ker.sum()
    smear = cv2.filter2D(body.astype(np.float32), -1, ker, borderType=cv2.BORDER_CONSTANT)
    smear = np.maximum(smear, cv2.filter2D(body.astype(np.float32), -1, ker[:, ::-1].copy(), borderType=cv2.BORDER_CONSTANT))
    veil = np.clip(smear * 1.3, 0, 1) * (0.25 + 0.6 * stc)
    alpha = np.clip(C.lerp(body, veil, np.clip(fz * 1.2, 0, 1)) + wisp, 0, 1) * (1 - 0.1 * fog)
    col = C.lerp(col, C.lerp(col, skyc, 0.25), fz[..., None] * 0.6)
    if halo:
        of2 = C.smoothstep(0.1, 0.8, _fblur(oface, 0.008 * W))
        hal = np.exp(-dist_out / (0.01 * W)) * of2 * prox * 0.3 * halo * rim * (1 - fog)
        hal = hal * (1 - alpha)
        a2 = alpha + hal
        col = (col * alpha[..., None] + rimc * hal[..., None]) / np.maximum(a2, 1e-5)[..., None]
        alpha = a2
    return np.dstack([col, np.clip(alpha, 0, 1)]).astype(np.float32)
