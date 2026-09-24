"""Art helpers for s02_railway_crossing (round 2):

* painted foliage masses v2 - irregular clumps with domain-warped torn silhouettes (not circles), a lit warm
  top face and a cool teal underside with a crisp terminator, dark occlusion gaps between clumps, inner
  leaf-mass patches, leaf flecks along the rims and sky holes - plus tree/grove/row builders;
* painted mountain ranges - hard silhouette, lit/shadow faces and gully striations following the fall line,
  white haze toward the base, blue-lavender atmospheric tint;
* painted ballast stones tile, road lettering mask, anvil shading for the hero cumulonimbus.
"""
import math
import os

import numpy as np
import cv2
from numba import njit, prange

import s02_railway_crossing_paint as P

hx = P.hexc
sstep = P.sstep

# clump columns
(K_CX, K_CY, K_RX, K_RY, K_Z, K_GX, K_GY, K_GR, K_TONE, K_FOG, K_OX, K_OY, K_AMP, K_FLAT, K_NS,
 K_KIND) = range(16)
NK = 16


@njit(cache=True, fastmath=True)
def _samp(T, u, v):
    n = T.shape[0]
    u = u % n
    v = v % n
    i0 = int(u)
    j0 = int(v)
    fu = u - i0
    fv = v - j0
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    return ((T[j0, i0] * (1 - fu) + T[j0, i1] * fu) * (1 - fv) +
            (T[j1, i0] * (1 - fu) + T[j1, i1] * fu) * fv)


@njit(cache=True, fastmath=True)
def _zbuf2(Hs, Ws, S, ss, N1, N2, leaf):
    n = S.shape[0]
    hb = np.full((Hs, Ws), -1e12, np.float32)
    ib = np.full((Hs, Ws), -1, np.int32)
    for i in range(n):
        cx, cy = S[i, 0] * ss, S[i, 1] * ss
        rx, ry = S[i, 2] * ss, S[i, 3] * ss
        z0 = S[i, 4] * ss
        ox, oy, amp, flat, ns = S[i, 10], S[i, 11], S[i, 12], S[i, 13], S[i, 14]
        famp = 0.22 * min(rx / (6.0 * leaf * ss), 1.0)
        R = max(rx, ry) * (1.0 + 1.6 * amp + famp) + 2.0
        x0 = max(0, int(cx - R))
        x1 = min(Ws, int(cx + R) + 2)
        y0 = max(0, int(cy - R))
        y1 = min(Hs, int(cy + R) + 2)
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / ry
            if dy > 0:
                dy = dy * (1.0 + flat)
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / rx
                d = math.sqrt(dx * dx + dy * dy)
                if d > 1.0 + 1.6 * amp + famp:
                    continue
                # domain-warped edge: clump-scale lobes + leaf-scale fringe
                wu = _samp(N1, dx * ns + ox + 37.0, dy * ns + oy)
                wv = _samp(N1, dx * ns + ox, dy * ns + oy + 91.0)
                e1 = _samp(N1, dx * ns + ox + (wu - 0.5) * 2.5, dy * ns + oy + (wv - 0.5) * 2.5)
                e2 = _samp(N2, (x + 0.5) / (leaf * ss) * 4.0 + ox * 7.0, (y + 0.5) / (leaf * ss) * 4.0 + oy * 7.0)
                dd = d - amp * (e1 - 0.5) * 3.2 - famp * (e2 - 0.5) * 2.4
                if dd >= 1.0:
                    continue
                q = max(dd, 0.0)
                h = z0 + rx * math.sqrt(max(1.0 - q * q, 0.0))
                if h > hb[y, x]:
                    hb[y, x] = h
                    ib[y, x] = i
    return hb, ib


@njit(cache=True, fastmath=True)
def _shade2(hb, ib, S, ss, Lx, Ly, Lz, shift, wl, wg, N1, gap):
    Hs, Ws = hb.shape
    ndl = np.zeros((Hs, Ws), np.float32)
    sh = np.zeros((Hs, Ws), np.float32)
    gp = np.zeros((Hs, Ws), np.float32)
    inner = np.zeros((Hs, Ws), np.float32)
    rim = np.zeros((Hs, Ws), np.float32)
    ll = math.sqrt(Lx * Lx + Ly * Ly) + 1e-9
    gi = int(gap)
    gj = int(gap * 0.6)
    for y in range(Hs):
        for x in range(Ws):
            i = ib[y, x]
            if i < 0:
                continue
            cx, cy = S[i, 0] * ss, S[i, 1] * ss
            rx, ry = S[i, 2] * ss, S[i, 3] * ss
            flat = S[i, 13]
            dx = (x + 0.5 - cx) / rx
            dy = (y + 0.5 - cy) / ry
            if dy > 0:
                dy = dy * (1.0 + flat)
            dd = dx * dx + dy * dy
            if dd > 0.97:
                s = math.sqrt(0.97 / dd)
                dx *= s
                dy *= s
                dd = 0.97
            nz = math.sqrt(1.0 - dd)
            gx, gy, gr = S[i, 5] * ss, S[i, 6] * ss, S[i, 7] * ss
            ex = (x + 0.5 - gx) / gr
            ey = (y + 0.5 - gy) / gr
            ee = ex * ex + ey * ey
            if ee > 0.95:
                s2 = math.sqrt(0.95 / ee)
                ex *= s2
                ey *= s2
                ee = 0.95
            ez = math.sqrt(1.0 - ee)
            Nx = wl * dx + wg * ex
            Ny = wl * dy + wg * ey
            Nz = wl * nz + wg * ez
            nl = math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-6
            ndl[y, x] = (Nx * Lx + Ny * Ly + Nz * Lz) / nl
            ns = S[i, 14]
            ox, oy = S[i, 10], S[i, 11]
            inner[y, x] = _samp(N1, dx * ns * 2.2 + ox + 13.0, dy * ns * 2.2 + oy + 57.0)
            h0 = hb[y, x]
            # cast shadow from a nearer clump toward the light
            occ = 0.0
            for st in range(1, 7):
                d = shift * st / 6.0
                xx = int(x + Lx / ll * d)
                yy = int(y + Ly / ll * d)
                if xx < 0 or yy < 0 or xx >= Ws or yy >= Hs:
                    break
                j = ib[yy, xx]
                if j >= 0 and j != i and hb[yy, xx] > h0 + d * 0.3:
                    occ = 1.0
                    break
            sh[y, x] = occ
            # dark occlusion gap: a different, nearer clump right next to this pixel
            g = 0.0
            for k in range(1, 4):
                ay = y - gi * k
                if ay < 0:
                    break
                j2 = ib[ay, x]
                if j2 >= 0 and j2 != i and hb[ay, x] > h0 + 0.5:
                    g = 1.0 - (k - 1) * 0.3
                    break
            gp[y, x] = g
            xr = int(x + Lx / ll * 2.0 * ss)
            yr = int(y + Ly / ll * 2.0 * ss)
            if xr >= 0 and yr >= 0 and xr < Ws and yr < Hs:
                if ib[yr, xr] < 0:
                    rim[y, x] = 1.0
    return ndl, sh, gp, inner, rim


_TEX = {}


def _tex(name, seed, octaves, base):
    k = (name, seed)
    if k not in _TEX:
        _TEX[k] = P.tile_noise(256, seed=seed, octaves=octaves, base=base).astype(np.float32)
    return _TEX[k]


PAL_MID = dict(gap=hx('#0b2d34'), deep=hx('#123f48'), shd=hx('#1f5e62'), shd2=hx('#2f7466'),
               mid=hx('#4f9a45'), lit=hx('#8fc83c'), hi=hx('#cfea6c'), rim=hx('#f2f9b4'),
               warm=hx('#a6c83a'), cool=hx('#2c6b72'), haze=hx('#a9cde2'))


def foliage2(W, H, S, pal=None, light=(0.62, -0.72, 0.3), ss=2, leaf=None, shadow_shift=0.006, wl=0.45,
             wg=0.55, flecks=1.0, seed=0, holes=1.0, haze=None):
    """Render painted foliage clumps -> (H, W, 4) straight-alpha RGBA.
    S: (n, NK) clumps in 1x px: cx, cy, rx, ry, z (larger = nearer), group cx, cy, r, tone (-1..1), fog
    (0..1), noise offsets ox, oy, edge amp (0.05..0.2), bottom flatten, noise scale (edge lobes per radius),
    kind (unused)."""
    pal = PAL_MID if pal is None else pal
    L = np.asarray(light, np.float64)
    L = L / np.linalg.norm(L)
    leaf = max(0.0019 * W, 1.0) if leaf is None else leaf
    Hs, Ws = H * ss, W * ss
    S = np.ascontiguousarray(S, np.float32)
    N1 = _tex('n1', 501, 4, 8)
    N2 = _tex('n2', 502, 3, 32)
    hb, ib = _zbuf2(Hs, Ws, S, ss, N1, N2, float(leaf))
    ndl, sh, gp, inner, rim = _shade2(hb, ib, S, ss, float(L[0]), float(L[1]), float(L[2]),
                                      shadow_shift * W * ss, wl, wg, N1, max(1.0, 0.9 * leaf * ss))
    cov = (ib >= 0).astype(np.float32)
    idx = np.maximum(ib, 0)
    tone = np.where(ib >= 0, S[idx, K_TONE], 0).astype(np.float32)
    fog = np.where(ib >= 0, S[idx, K_FOG], 0).astype(np.float32)
    # leaf-scale dither at the terminator (painted broken edge, not a smooth gradient)
    rng = np.random.default_rng(seed + 7)
    N3 = _tex('n3', 503, 3, 24)
    xs = (np.arange(Ws, dtype=np.float32) / (leaf * ss * 1.3))[None, :].repeat(Hs, 0)
    ys = (np.arange(Hs, dtype=np.float32) / (leaf * ss * 1.3))[:, None].repeat(Ws, 1)
    dn = P.sample_tile(N3, xs, ys) - 0.5
    v = ndl - 0.5 * sh + 0.07 * tone + (inner - 0.5) * 0.55 + dn * 0.22
    col = pal['deep'] + (pal['shd'] - pal['deep']) * sstep(-0.55, -0.45, v)[..., None]
    col = col + (pal['shd2'] - col) * sstep(-0.12, -0.04, v)[..., None]
    col = col + (pal['mid'] - col) * sstep(0.04, 0.1, v)[..., None]
    col = col + (pal['lit'] - col) * sstep(0.36, 0.42, v)[..., None]
    col = col + (pal['hi'] - col) * (sstep(0.62, 0.68, v) * 0.85)[..., None]
    col = col + (pal['warm'] - col) * (np.clip(tone, 0, 1) * 0.35 * sstep(0.1, 0.3, v))[..., None]
    col = col + (pal['cool'] - col) * (np.clip(-tone, 0, 1) * 0.3 * (1 - sstep(0.1, 0.3, v)))[..., None]
    col = col + (pal['gap'] - col) * (gp * 0.92)[..., None]
    rimv = rim * sstep(0.05, 0.3, ndl) * (1 - gp)
    col = col + (pal['rim'] - col) * (rimv * 0.55)[..., None]
    hz_ = pal['haze'] if haze is None else haze
    col = col + (hz_ - col) * fog[..., None]
    rgba = np.dstack([col, cov]).astype(np.float32)
    # ---- sky holes inside near the silhouette + leaf flecks outside it
    if holes > 0 or flecks > 0:
        m8 = (cov > 0.5).astype(np.uint8)
        din = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
        dout = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 3)
        lr = leaf * ss
        N4 = _tex('n4', 504, 2, 48)
        hn = P.sample_tile(N4, xs * 0.8 + 11, ys * 0.8 + 5)
        if holes > 0:
            hole = (din > 0.5 * lr) & (din < 3.5 * lr) & (hn > 1 - 0.08 * holes) & (rim < 0.5)
            rgba[hole, 3] = 0.0
        if flecks > 0:
            band = (dout > 0.3 * lr) & (dout < 1.7 * lr)
            yy, xx = np.nonzero(band)
            keep = rng.random(len(yy)) < (0.0035 * flecks / max(lr * lr / 9.0, 0.3))
            yy, xx = yy[keep], xx[keep]
            # colour from the nearest foliage: normalised blur
            src = np.ascontiguousarray(rgba[..., :3] * cov[..., None])
            bl = cv2.GaussianBlur(src, (0, 0), 1.2 * lr) / np.maximum(cv2.GaussianBlur(cov, (0, 0), 1.2 * lr),
                                                                      1e-4)[..., None]
            buf = rgba
            for (y_, x_) in zip(yy, xx):
                c = bl[y_, x_]
                if not np.isfinite(c).all():
                    continue
                ax = lr * rng.uniform(0.35, 0.75)
                ay = ax * rng.uniform(0.45, 0.8)
                ang = rng.uniform(0, 180)
                cc = c * rng.uniform(0.95, 1.15)
                cv2.ellipse(buf, (int(x_ * 16), int(y_ * 16)), (max(int(ax * 16), 8), max(int(ay * 16), 8)), ang, 0,
                            360, (float(cc[0]), float(cc[1]), float(cc[2]), 1.0), -1, cv2.LINE_8, shift=4)
    a = rgba[..., 3]
    rgb = rgba[..., :3]
    pm = np.dstack([rgb * a[..., None], a])
    small = cv2.resize(pm, (W, H), interpolation=cv2.INTER_AREA)
    al = small[..., 3]
    out = np.dstack([small[..., :3] / np.maximum(al, 1e-4)[..., None], al]).astype(np.float32)
    return out


# ------------------------------------------------------------------------------------------ builders

def _row(cx, cy, rx, ry, z, g, tone, fog, rng, amp=0.13, flat=0.55, ns=None):
    ns = rng.uniform(1.6, 2.4) if ns is None else ns
    return [cx, cy, rx, ry, z, g[0], g[1], g[2], float(np.clip(tone, -1, 1)), fog, rng.uniform(0, 256),
            rng.uniform(0, 256), amp, flat, ns, 0]


def broadleaf(rng, cx, by, w, h, z0, fog, tone0=0.0, kind='round'):
    """A broadleaf crown (camphor / keyaki) as overlapping irregular leaf masses in rows; lower rows are
    nearer so their lit tops overlap the shadowed undersides of the masses behind (layered painted look)."""
    out = []
    top = by - h
    g = (cx, by - h * 0.5, max(w, h) * 0.6)
    r0 = w * (0.2 if kind != 'shrub' else 0.3) * rng.uniform(0.9, 1.1)
    nrows = max(2, int(round((h - r0) / (r0 * 0.8))))
    for rI in range(nrows):
        f = rI / max(nrows - 1, 1)                    # 0 top .. 1 bottom
        y = top + r0 * 0.75 + f * (h - r0 * 1.6)
        if kind == 'vase':
            half = w * 0.5 * (1.0 - 0.35 * f)
        elif kind == 'shrub':
            half = w * 0.5
        else:
            half = w * 0.5 * (0.35 + 0.65 * math.sin(math.pi * min(0.2 + 0.8 * f, 0.92)) ** 0.7)
        nx = max(1, int(round(2 * half / (r0 * 1.25))))
        for k in range(nx):
            u = (k + 0.5) / nx * 2 - 1
            x = cx + u * max(half - r0 * 0.7, 0) + rng.uniform(-0.25, 0.25) * r0
            r = r0 * rng.uniform(0.8, 1.2) * (0.85 + 0.3 * f)
            yy = y + rng.uniform(-0.2, 0.2) * r0
            out.append(_row(x, yy, r, r * rng.uniform(0.72, 0.85), z0 + f * r0 * 2.0 + rng.uniform(0, r0 * 0.4), g,
                            tone0 + rng.normal(0, 0.22), fog, rng, amp=rng.uniform(0.09, 0.14), flat=0.35))
    # crown-top sprigs and side sprigs breaking the silhouette
    for i in range(int(rng.integers(3, 6))):
        a = rng.uniform(-math.pi * 0.92, -math.pi * 0.08)
        r = r0 * rng.uniform(0.45, 0.65)
        x = cx + math.cos(a) * w * 0.38
        y = (top + h * 0.45) + math.sin(a) * h * 0.47
        out.append(_row(x, y, r, r * 0.7, z0 - r0 * 0.2, g, tone0 + rng.normal(0.2, 0.2), fog, rng,
                        amp=rng.uniform(0.1, 0.15), flat=0.6))
    return out


def tree_row(rng, x0, x1, ybase, height_fn, r_mean, z0, fog, tone_sd=0.35):
    """A continuous band of distant tree crowns (hedgerow / wooded hill): irregular masses packed along x,
    top edge following height_fn(x), stacked down to ybase."""
    out = []
    x = x0
    while x < x1:
        r = r_mean * rng.uniform(0.7, 1.4)
        top = height_fn(x)
        g = (x, top + r * 2.0, r * 3.5)
        y = top + r * 0.7
        lev = 0
        while y < ybase + r * 0.3:
            rr = r * (1 + 0.2 * lev) * rng.uniform(0.85, 1.15)
            out.append(_row(x + rng.normal(0, 0.3) * r, y, rr, rr * 0.72, z0 + lev * r * 0.5 + rng.uniform(0, r * 0.3),
                            g, rng.normal(0, tone_sd), fog, rng, amp=rng.uniform(0.1, 0.16)))
            y += rr * rng.uniform(0.8, 1.1)
            lev += 1
        x += r * rng.uniform(0.9, 1.4)
    return out


def draw_cedar2(cv, rng, cx, by, w, h, pal, fog=0.0, hazec=None):
    """Sugi cedar: narrow spire of drooping, jagged branch tiers. Each tier: shadowed body, lit right flank,
    bright drooping tips, dark gap under the tier above."""
    hazec = hx('#a9cde2') if hazec is None else hazec

    def fz(c):
        return np.asarray(c, np.float32) * (1 - fog) + hazec * fog
    dark, mid, lit, tip = fz(pal[0]), fz(pal[1]), fz(pal[2]), fz(pal[3])
    # trunk
    cv.poly([(cx - w * 0.05, by), (cx + w * 0.05, by), (cx + w * 0.02, by - h * 0.5), (cx - w * 0.02, by - h * 0.5)],
            fz(hx('#3a2e2a')))
    n = int(np.clip(h / max(w * 0.2, 1.0), 6, 18))
    y0 = by - h * 0.12
    for i in range(n):
        f0 = i / n
        f1 = (i + 1.6) / n
        ya = y0 - f0 * (h * 0.88)
        yb = y0 - min(f1, 1.0) * (h * 0.88)
        half = w * 0.5 * (1 - f0) ** 0.85 + w * 0.04
        jl, jr = rng.uniform(0.9, 1.15), rng.uniform(0.9, 1.15)
        L = (cx - half * jl, ya + h / n * 0.25)
        R = (cx + half * jr, ya + h / n * 0.3)
        T = (cx + rng.uniform(-0.05, 0.05) * w, yb)
        # tier body (drooping ends)
        body = [L, (cx - half * 0.55, ya - h / n * 0.1), T, (cx + half * 0.55, ya - h / n * 0.1), R,
                (cx + half * 0.4, ya + h / n * 0.05), (cx - half * 0.4, ya + h / n * 0.05)]
        cv.poly(body, dark)
        cv.poly([(cx + half * 0.05, ya - h / n * 0.05), T, (cx + half * 0.55, ya - h / n * 0.1), R,
                 (cx + half * 0.3, ya + h / n * 0.02)], mid)
        cv.poly([(cx + half * 0.35, ya - h / n * 0.12), (cx + half * 0.5, ya - h / n * 0.2), R,
                 (cx + half * 0.6, ya + h / n * 0.05)], lit)
        # bright drooping tips on the sun side
        for k in range(2):
            px = cx + half * rng.uniform(0.45, 0.95)
            py = ya - h / n * rng.uniform(-0.05, 0.25)
            s = half * 0.16
            cv.poly([(px - s, py), (px + s * 0.4, py - s * 0.6), (px + s, py + s * 0.5)], tip)
    cv.poly([(cx - w * 0.03, y0 - h * 0.86), (cx + w * 0.03, y0 - h * 0.86), (cx, by - h * 1.02)], dark)


# ------------------------------------------------------------------------------------------ mountains

def _blur1(a, s):
    if s < 0.3:
        return a.astype(np.float32)
    return cv2.GaussianBlur(a.astype(np.float32)[None, :], (0, 0), sigmaX=s, borderType=cv2.BORDER_REFLECT)[0]


def paint_range(mw, mh, top, amp, pal, seed, hz, sc, haze_amt=0.8, stri=1.0):
    """Painted mountain range below crest line `top` (px, length mw). Returns rgb (mh, mw, 3), alpha."""
    ys = np.arange(mh, dtype=np.float32)[:, None].repeat(mw, 1)
    xs = np.arange(mw, dtype=np.float32)[None, :].repeat(mh, 0)
    dd = ys - top[None, :]
    alpha = sstep(-0.7, 0.7, dd).astype(np.float32)
    # large faces: slope of the progressively smoothed crest (ridges spread out with depth)
    levels = [1.5, 4, 10, 24, 50]
    sl = [np.gradient(_blur1(top, s * sc)) for s in levels]
    tpos = np.clip(dd / (amp * 0.9), 0, 1) * (len(levels) - 1)
    i0 = np.clip(np.floor(tpos).astype(np.int32), 0, len(levels) - 2)
    fr = tpos - i0
    SL = np.stack(sl, 0)
    xi = np.broadcast_to(np.arange(mw)[None, :], dd.shape)
    s0 = SL[i0, xi]
    s1 = SL[i0 + 1, xi]
    slope = s0 * (1 - fr) + s1 * fr
    big = np.tanh(slope * 2.2)
    # gullies: ridged striations running down the fall line, fanning out from the peaks
    N = P.tile_noise(256, seed=seed + 300, octaves=4, base=6)
    Nw = P.tile_noise(256, seed=seed + 301, octaves=3, base=4)
    scx = max(amp * 0.07, 2.0)
    scy = amp * 0.7
    u = (xs - np.clip(slope, -1.5, 1.5) * dd * 0.55) / scx
    v = dd / scy
    wv = P.sample_tile(Nw, u * 0.25, v * 0.5) - 0.5
    g = P.sample_tile(N, u + wv * 3.0, v * 0.35)
    g = 1 - np.abs(g * 2 - 1)                                      # ridged
    gb = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 0.8)
    gdx = -cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3) * scx * 0.5
    deep = sstep(amp * 0.02, amp * 0.18, dd)
    fine = P.sample_tile(Nw, xs / (scx * 0.35), ys / (scx * 0.35)) - 0.5
    val = big * 0.9 + np.tanh(gdx * 1.5) * 0.55 * deep * stri + fine * 0.18
    # crisp painted value bands
    col = pal['shd'] + (pal['mid'] - pal['shd']) * sstep(-0.18, -0.1, val)[..., None]
    col = col + (pal['lit'] - col) * sstep(0.22, 0.3, val)[..., None]
    col = col + (pal['hi'] - col) * (sstep(0.62, 0.7, val) * (1 - deep * 0.6))[..., None]
    # ridge-crest highlight on the sun side
    rim = np.exp(-np.maximum(dd, 0) / (1.2 * sc + 0.4)) * sstep(-0.1, 0.3, big)
    col = col + (pal['rim'] - col) * (rim * 0.6)[..., None]
    # forest mottling on the lower slopes (dark blue-green patches, very muted)
    fm = P.sample_tile(N, xs / (scx * 2.5) + 50, ys / (scx * 1.2)) * sstep(0.35, 0.8, dd / amp)
    col = col + (pal['forest'] - col) * (sstep(0.55, 0.62, fm) * 0.35)[..., None]
    # white haze toward the base
    base0 = top[None, :] + amp * 0.25
    hzf = np.clip((ys - base0) / np.maximum(hz - base0, 1), 0, 1) ** 1.4
    col = col + (pal['haze'] - col) * (hzf * haze_amt)[..., None]
    col = col + (pal['tint'] - col) * pal.get('tint_amt', 0.2)
    return col.astype(np.float32), alpha


def profile(W, seed, freq, sharp=1.5, octs=6):
    rng = np.random.default_rng(seed)
    x = np.arange(W, dtype=np.float64) / W
    p = np.zeros(W)
    a = 1.0
    fr = freq
    for o in range(octs):
        ph = rng.uniform(0, 100)
        n = np.interp(x * fr + ph, np.arange(int(fr + ph) + 3), rng.random(int(fr + ph) + 3))
        n = cv2.GaussianBlur(n.astype(np.float32)[None, :], (0, 0), sigmaX=max(W / fr / 5, 0.5))[0]
        p += a * (1 - np.abs(n * 2 - 1)) ** sharp
        a *= 0.45
        fr *= 2.2
    return (p - p.min()) / (p.max() - p.min() + 1e-9)


# ------------------------------------------------------------------------------------------ ballast tile

def ballast_tile(n=512, seed=3, stones=2600):
    """Painted crushed-stone tile (tileable): angular stones, lit top-right faces, cool shadows, dark gaps."""
    rng = np.random.default_rng(seed)
    ss = 2
    N = n * ss
    img = np.zeros((N * 3 // 3, N, 3), np.float32)
    img[:] = hx('#4a4642')
    base_cols = [hx('#9c968c'), hx('#aaa39a'), hx('#8d8a86'), hx('#b3aca0'), hx('#948a7e'), hx('#a09a94')]
    for i in range(stones):
        cx, cy = rng.uniform(0, N), rng.uniform(0, N)
        r = rng.uniform(5, 11) * ss
        k = int(rng.integers(5, 8))
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        rad = r * rng.uniform(0.7, 1.15, k)
        pts = np.stack([np.cos(ang) * rad, np.sin(ang) * rad * rng.uniform(0.7, 1.0)], 1)
        base = base_cols[int(rng.integers(0, len(base_cols)))] * rng.uniform(0.9, 1.08)
        lit = np.minimum(base * np.array([1.22, 1.18, 1.1]), 1.0)
        shd = base * np.array([0.55, 0.6, 0.74]) + np.array([0.02, 0.03, 0.06])
        for ox in (-N, 0, N):
            for oy in (-N, 0, N):
                if not (-r * 2 < cx + ox < N + r * 2 and -r * 2 < cy + oy < N + r * 2):
                    continue
                c = np.array([cx + ox, cy + oy])
                p = np.round((pts + c) * 16).astype(np.int32)
                cv2.fillPoly(img, [p], tuple(float(v) for v in shd), cv2.LINE_8, shift=4)
                # lit facet: polygon shifted toward the light, clipped by shrinking
                p2 = np.round((pts * 0.78 + c + np.array([0.12, -0.14]) * r) * 16).astype(np.int32)
                cv2.fillPoly(img, [p2], tuple(float(v) for v in base), cv2.LINE_8, shift=4)
                p3 = np.round((pts * 0.42 + c + np.array([0.3, -0.32]) * r) * 16).astype(np.int32)
                cv2.fillPoly(img, [p3], tuple(float(v) for v in lit), cv2.LINE_8, shift=4)
    img = cv2.resize(img, (n, n), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32)


# ------------------------------------------------------------------------------------------ road text

def text_mask(text, px=256, font_names=('YuGothB.ttc', 'meiryob.ttc', 'msgothic.ttc')):
    """Render text (single line) to a float mask (h, w); returns None if no CJK font is available."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    font = None
    for fn in font_names:
        for d in (os.path.join(os.environ.get('WINDIR', 'C:/Windows'), 'Fonts'), '/usr/share/fonts'):
            p = os.path.join(d, fn)
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, px)
                    break
                except Exception:
                    pass
        if font:
            break
    if font is None:
        return None
    W = px * (len(text) + 1)
    im = Image.new('L', (W, int(px * 1.4)), 0)
    dr = ImageDraw.Draw(im)
    dr.text((px * 0.2, px * 0.1), text, fill=255, font=font)
    a = np.asarray(im, np.float32) / 255.0
    ys, xs = np.nonzero(a > 0.05)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ------------------------------------------------------------------------------------------ anvil

def shade_anvil(plate, x0, x1, y_top, y_bot, sun_xy, seed=0, sc=1.0):
    """Give the hero cumulonimbus anvil (plate region x0..x1, above y_bot) painted form: sunlit top shelf,
    lavender half-tone lobes, cool blue-lavender underside, silver rim toward the sun."""
    H, W = plate.shape[:2]
    y0 = max(int(y_top), 0)
    y1 = min(int(y_bot), H)
    x0 = max(int(x0), 0)
    x1 = min(int(x1), W)
    sub = plate[y0:y1, x0:x1]
    a = sub[..., 3]
    m = a > 0.5
    h, w = a.shape
    if not m.any():
        return plate
    ys = np.arange(h, dtype=np.float32)[:, None]
    # per-column top & bottom of the anvil body
    anyc = m.any(0)
    topc = np.where(anyc, m.argmax(0), h).astype(np.float32)
    botc = np.where(anyc, h - 1 - m[::-1].argmax(0), 0).astype(np.float32)
    topc = cv2.GaussianBlur(topc[None, :], (0, 0), 2)[0]
    botc = cv2.GaussianBlur(botc[None, :], (0, 0), 6)[0]
    th = np.maximum(botc - topc, 4)
    f = np.clip((ys - topc[None, :]) / th[None, :], 0, 1.2)
    xs = np.arange(w, dtype=np.float32)[None, :]
    N = P.tile_noise(256, seed=seed + 11, octaves=4, base=6)
    XX = np.broadcast_to(xs, (h, w)).astype(np.float32)
    YY = np.broadcast_to(ys, (h, w)).astype(np.float32)
    lob = P.sample_tile(N, XX / (0.012 * W * sc + 1) * 1.0, YY / (0.008 * W * sc + 1)) - 0.5
    # scalloped shelf lines: arcs of billows along the anvil
    period = 0.035 * W
    ph = xs / period + lob * 0.8
    arc = np.abs(np.sin(np.pi * ph)) ** 0.6
    s = f + lob * 0.35 - 0.18 * arc
    lit_top = plate[y0:y1, x0:x1, :3].copy()
    lav_half = hx('#e2def2') * 1.02
    lav_shd = hx('#aeb0dc')
    under = hx('#98a6d4')
    col = lit_top
    t1 = sstep(0.32, 0.4, s)[..., None]
    t2 = sstep(0.62, 0.7, s)[..., None]
    t3 = sstep(0.85, 1.05, f)[..., None]
    col = col + (lav_half - col) * t1 * 0.85
    col = col + (lav_shd - col) * t2 * 0.9
    col = col + (under - col) * t3 * 0.6
    # silver rim toward the sun (upper edge and right end)
    m8 = m.astype(np.uint8)
    din = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    sx, sy = sun_xy[0] - x0, sun_xy[1] - y0
    gx = xs - sx
    gy = ys - sy
    dirf = np.clip(-(gx * 0.0 + gy) / (np.sqrt(gx * gx + gy * gy) + 1) * 0.5 + 0.5 + (xs / w) * 0.4, 0, 1)
    rim = np.exp(-din / (1.6 * sc + 0.5)) * dirf * (f < 0.5)
    col = col + (np.array([1.12, 1.1, 1.04], np.float32) - col) * np.clip(rim * 1.1, 0, 1)[..., None]
    k = np.clip(a * 1.5, 0, 1)[..., None] * sstep(h * 1.0, h * 0.7, ys)[..., None]
    plate[y0:y1, x0:x1, :3] = sub[..., :3] * (1 - k) + col * k
    return plate


# ------------------------------------------------------------------------------------------ terrain (voxel)

def ridged_mf(N, seed, base=5, octaves=8, gain=0.5):
    """Ridged multifractal heightfield (N x N, ~0..1): sharp ridges, dendritic gullies."""
    from lib import core as C
    h = np.zeros((N, N), np.float32)
    w = np.ones((N, N), np.float32)
    amp = 1.0
    fr = base
    tot = 0.0
    for o in range(octaves):
        n = C.value_noise(N, N, fr, fr, seed + 17 * o)
        r = (1.0 - np.abs(2.0 * n - 1.0)) ** 2
        r = r * w
        h += r * amp
        tot += amp
        w = np.clip(r * 1.8, 0, 1)
        amp *= gain
        fr *= 2
    return h / tot


@njit(cache=True, fastmath=True)
def _voxel(Wd, Hd, hm, x0w, z0w, cell, f, px, hz, camh, zn, zf, dzk, curv, shade, hmask):
    """Comanche-style terrain raster. hm: heightmap (m) over world x = x0w + i*cell, z = z0w + j*cell.
    Returns per-pixel shade value, depth (m), height above plain (m), and coverage."""
    n_j, n_i = hm.shape
    S = np.zeros((Hd, Wd), np.float32)
    D = np.zeros((Hd, Wd), np.float32)
    E = np.zeros((Hd, Wd), np.float32)
    A = np.zeros((Hd, Wd), np.float32)
    for sx in range(Wd):
        dirx = (sx + 0.5 - px) / f
        ybuf = float(Hd)
        z = zn
        while z < zf:
            wx = dirx * z
            fi = (wx - x0w) / cell
            fj = (z - z0w) / cell
            if fi >= 0 and fj >= 0 and fi < n_i - 1 and fj < n_j - 1:
                i0 = int(fi)
                j0 = int(fj)
                a = fi - i0
                b = fj - j0
                h = ((hm[j0, i0] * (1 - a) + hm[j0, i0 + 1] * a) * (1 - b) +
                     (hm[j0 + 1, i0] * (1 - a) + hm[j0 + 1, i0 + 1] * a) * b)
                s = ((shade[j0, i0] * (1 - a) + shade[j0, i0 + 1] * a) * (1 - b) +
                     (shade[j0 + 1, i0] * (1 - a) + shade[j0 + 1, i0 + 1] * a) * b)
                sy = hz - (h - camh - z * z * curv) * f / z
                if sy < ybuf:
                    y0 = int(max(sy, 0.0))
                    yb = min(int(math.ceil(ybuf)), Hd)
                    for yy in range(y0, yb):
                        S[yy, sx] = s
                        D[yy, sx] = z
                        E[yy, sx] = h
                        # fractional coverage at the top pixel (anti-aliased silhouette)
                        if yy == y0:
                            A[yy, sx] = max(A[yy, sx], min(1.0, (y0 + 1.0) - sy))
                        else:
                            A[yy, sx] = 1.0
                    ybuf = sy
            z += max(z * dzk, 1.0)
    return S, D, E, A


def billow_form(plate, box, sun_xy, seed=0, sc=1.0, n=160, rmin=0.012, rmax=0.045, amt=0.55):
    """Paint cauliflower lobe structure into large blown-white areas of a cloud plate: random billows, each
    with a crisp lavender crescent on the side away from the sun; lower billows overlap upper ones."""
    H, W = plate.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, W), min(y1, H)
    sub = plate[y0:y1, x0:x1]
    a = sub[..., 3]
    lum = sub[..., :3].mean(-1)
    core = ((a > 0.95) & (lum > 0.86)).astype(np.uint8)
    er = int(0.006 * W) | 1
    core = cv2.erode(core, np.ones((er, er), np.uint8))
    h, w = core.shape
    ys_, xs_ = np.nonzero(core)
    if len(ys_) == 0:
        return plate
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ys_), size=min(n, len(ys_)), replace=False)
    cy = ys_[pick].astype(np.float32)
    cx = xs_[pick].astype(np.float32)
    rr = rng.uniform(rmin, rmax, len(pick)).astype(np.float32) * W
    order = np.argsort(cy)
    shade = np.zeros((h, w), np.float32)
    sx, sy = sun_xy[0] - x0, sun_xy[1] - y0
    for i in order:
        r = rr[i]
        X0, X1 = int(max(cx[i] - r - 1, 0)), int(min(cx[i] + r + 2, w))
        Y0, Y1 = int(max(cy[i] - r - 1, 0)), int(min(cy[i] + r + 2, h))
        if X1 <= X0 or Y1 <= Y0:
            continue
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        dx, dy = sx - cx[i], sy - cy[i]
        dl = math.hypot(dx, dy) + 1e-3
        ux, uy = dx / dl, dy / dl
        d = np.sqrt((xx - cx[i]) ** 2 + (yy - cy[i]) ** 2) / r
        inside = d < 1.0
        # crescent: inside the billow but outside the same disc shifted toward the sun
        sh = 0.5
        d2 = np.sqrt((xx - cx[i] - ux * r * sh) ** 2 + (yy - cy[i] - uy * r * sh) ** 2) / (r * 0.97)
        cres = inside & (d2 >= 1.0)
        rim_ = inside & (d > 0.9) & ~cres
        pt = shade[Y0:Y1, X0:X1]
        pt[inside] = 0.0
        pt[cres] = 1.0
        pt[rim_] = np.maximum(pt[rim_], 0.35)
    shade = cv2.GaussianBlur(shade, (0, 0), 0.6 * sc + 0.3)
    m = core.astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), 0.004 * W)
    lav = hx('#c4c6ea')
    k = (shade * m * amt)[..., None]
    plate[y0:y1, x0:x1, :3] = sub[..., :3] * (1 - k) + lav * k
    return plate
