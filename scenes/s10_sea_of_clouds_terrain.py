"""s10 sea of clouds as a painted HEIGHTFIELD (round 5 rebuild - no domes / spheres anywhere).

World: x right, y up, z forward (toward the sun).  The cloud deck is a periodic heightfield tile
(period P world units, rotated against the view so the tiling never lines up) built from band-limited
Gaussian random fields (FFT, so it tiles seamlessly):

  * BILLOWS: |n| of a domain-warped field, soft-capped -> broad, gently domed flat-topped plateaus of
    very different sizes separated by sharp creases (stratocumulus cells), never equal hemispheres;
  * a thresholded, domain-warped COVERAGE mask cuts the deck into masses with crisp torn edges and
    drops the gaps into deep valleys;
  * cauliflower LOBES and fine texture only on the crowns (weighted by the billow height).

Baked per texel (and per mip level): height, gradient, long cast shadow from the low sun, crease
cavity.  Rendering is a per-column 'voxel space' march at 2x supersampling (every column is a vertical
plane: no roll), shaded in the same pass with a painted ramp (deep indigo valley -> blue-violet face ->
lavender sky fill | pink terminator | peach -> gold -> cream by sun azimuth), aerial perspective, then a
rim pass paints a 2-3 px silver-gold line along every upper silhouette (strongest toward the sun) with
a soft forward-scatter glow creeping in from it.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

TAU = 2 * math.pi


def _rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


# ------------------------------------------------------------------ tile generation

def gfield(N, lam, rng, band=True):
    """Periodic Gaussian random field (N, N), correlation length ~lam texels, std 1."""
    w = rng.standard_normal((N, N)).astype(np.float32)
    F = np.fft.rfft2(w)
    ky = np.fft.fftfreq(N)[:, None]
    kx = np.fft.rfftfreq(N)[None, :]
    k = np.sqrt(kx * kx + ky * ky) * lam
    filt = np.exp(-(k * k) * 3.0)
    if band:
        filt = filt * (1 - np.exp(-(k * k) * 12.0))
    out = np.fft.irfft2(F * filt, s=(N, N)).astype(np.float32)
    return out / (out.std() + 1e-8)


def wrap_remap(img, dx, dy):
    N = img.shape[0]
    ys, xs = np.mgrid[0:N, 0:N].astype(np.float32)
    return cv2.remap(img, xs + dx, ys + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _soft_cap(x, c):
    return c * np.tanh(x / c)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def build_tile(N=3072, P=56.0, seed=3):
    """Returns dict(h, gx, gz, sh, cav) (N, N) float32; heights in world units, e = P / N."""
    rng = np.random.default_rng(seed)
    e = P / N
    U = 1.0 / e                      # texels per world unit
    # domain warp (big + medium) shared by the billows so their outlines are irregular, not round
    w1x, w1y = gfield(N, 9 * U, rng), gfield(N, 9 * U, rng)
    w2x, w2y = gfield(N, 2.2 * U, rng), gfield(N, 2.2 * U, rng)
    wx = (w1x * 1.6 + w2x * 0.35) * U
    wy = (w1y * 1.6 + w2y * 0.35) * U
    def dist(mask):
        pad = int(3.2 * U)
        m = np.pad(mask.astype(np.uint8), pad, mode='wrap')
        d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        return d, pad

    def dist_r(mask, R):
        # distance from the edge with the medial-axis ridges rounded off (no crags)
        d, pad = dist(mask)
        sg = 0.45 * R * U
        db = cv2.GaussianBlur(d, (0, 0), sg)
        d = np.minimum(d, db * 1.0)[pad:pad + N, pad:pad + N]
        return cv2.GaussianBlur(d, (0, 0), 1.0) / U

    def rounded(d, R, A=1.0):
        x = np.clip(d / R, 0, 1)
        return A * R * (1 - (1 - x) ** 2.6)
    tear = wrap_remap(gfield(N, 0.7 * U, rng), wx * 0.3, wy * 0.3)
    # --- level 1: big cloud masses (4-14 units): rounded flanks, broad gently domed tops
    n1 = wrap_remap(gfield(N, 6.0 * U, rng) * 0.75 + gfield(N, 13.0 * U, rng) * 0.6, wx, wy)
    d1 = dist_r(n1 + 0.75 + 0.1 * tear > 0, 4.0)
    p1 = rounded(d1, 4.0, 0.3) + 0.25 * np.tanh(d1 / 5.0)
    s1 = _ss(0.1, 0.9, d1)
    # --- level 2: billows grown on the masses (1-3 units), irregular heaps
    w3x, w3y = gfield(N, 1.4 * U, rng), gfield(N, 1.4 * U, rng)
    n2 = wrap_remap(gfield(N, 2.0 * U, rng) * 0.8 + gfield(N, 3.8 * U, rng) * 0.5,
                    wx * 0.3 + w3x * 0.3 * U, wy * 0.3 + w3y * 0.3 * U)
    d2 = dist_r((n2 + 0.4 + 0.1 * tear > 0) & (d1 > 0.15), 1.5)
    p2 = rounded(d2, 1.5, 0.8) + 0.3 * np.tanh(d2 / 1.5)
    s2 = _ss(0.05, 0.4, d2)
    # --- level 3: small cauliflower heads on the billows' crowns (0.3-0.7 units)
    n3 = wrap_remap(gfield(N, 0.55 * U, rng), w3x * 0.15 * U, w3y * 0.15 * U)
    d3 = dist_r((n3 - 0.05 > 0) & (d2 > 0.1), 0.35)
    p3 = rounded(d3, 0.35, 0.5)
    low = gfield(N, 3.0 * U, rng, band=False)
    h = p1 + p2 + p3 * (0.4 + 0.6 * s2)
    h = h + 0.3 * gfield(N, 24 * U, rng, band=False) * s1 - 0.8 * (1 - s1) + 0.25 * low * (1 - s1)
    h = (h - 1.3).astype(np.float32)
    # gradients (world units per world unit), periodic central differences
    def grads(a):
        return ((np.roll(a, -1, 1) - np.roll(a, 1, 1)) / (2 * e), (np.roll(a, -1, 0) - np.roll(a, 1, 0)) / (2 * e))

    def wblur(a, sg):
        pd = int(4 * sg) + 1
        return cv2.GaussianBlur(np.pad(a, pd, mode='wrap'), (0, 0), sg)[pd:pd + N, pd:pd + N]
    # shading gradient: mostly the big form (one continuous light / shadow shape per heap), a little detail
    gdx, gdz = grads(h)
    gfx, gfz = grads(wblur(h, 0.45 * U))
    gx = (0.2 * gdx + 0.8 * gfx).astype(np.float32)
    gz = (0.2 * gdz + 0.8 * gfz).astype(np.float32)
    # cavity: crease darkness (blurred minus height, wrap)
    hb = wblur(h, 0.3 * U)
    hb2 = wblur(h, 2.0 * U)
    cav = np.clip(np.maximum((hb - h) / 0.12, (hb2 - h + 0.1) / 0.35), 0, 1).astype(np.float32)
    return dict(h=h, gx=gx, gz=gz, cav=cav, e=np.float32(e))


@njit(cache=True, fastmath=True)
def _bil(a, x, y):
    N = a.shape[0]
    x0 = math.floor(x)
    y0 = math.floor(y)
    fx = x - x0
    fy = y - y0
    i0 = int(x0) % N
    j0 = int(y0) % N
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    return (a[j0, i0] * (1 - fx) + a[j0, i1] * fx) * (1 - fy) + (a[j1, i0] * (1 - fx) + a[j1, i1] * fx) * fy


@njit(cache=True, fastmath=True, parallel=True)
def bake_shadow(h, dx, dz, tanel, e, dmax):
    """Soft cast shadow toward the sun (dx, dz unit vector in tile axes). 1 = lit."""
    N = h.shape[0]
    out = np.empty_like(h)
    for j in prange(N):
        for i in range(N):
            h0 = h[j, i]
            occ = 0.0
            d = 2.0 * e
            while d < dmax:
                hs = _bil(h, i + dx * d / e, j + dz * d / e)
                o = (hs - h0 - d * tanel) / (0.05 + 0.08 * d)
                if o > occ:
                    occ = o
                    if occ >= 1.0:
                        break
                d = d * 1.035 + e
            if occ > 1.0:
                occ = 1.0
            out[j, i] = 1.0 - occ * occ * (3 - 2 * occ)
    return out


def build_pyramid(T, rot, sun_dir_xz, tanel):
    """Packs channels (h, gx, gz, sh, cav) into a mip pyramid. rot: tile rotation (rad).
    sun_dir_xz: horizontal sun direction in world (x, z) (unit)."""
    c, s = math.cos(rot), math.sin(rot)
    # world -> tile: u = c*x - s*z, v = s*x + c*z
    du = c * sun_dir_xz[0] - s * sun_dir_xz[1]
    dv = s * sun_dir_xz[0] + c * sun_dir_xz[1]
    N = T['h'].shape[0]
    pd = 64
    hs_ = cv2.GaussianBlur(np.pad(T['h'], pd, mode='wrap'), (0, 0), 0.3 / float(T['e']))[pd:pd + N, pd:pd + N]
    sh = bake_shadow(np.ascontiguousarray(hs_), np.float32(du), np.float32(dv), np.float32(tanel), np.float32(T['e']), np.float32(9.0))
    sh = cv2.GaussianBlur(np.pad(sh, 16, mode='wrap'), (0, 0), 4.0)[16:-16, 16:-16]
    base = np.dstack([T['h'], T['gx'], T['gz'], sh, T['cav']]).astype(np.float32)
    levels = [base]
    while levels[-1].shape[0] > 16 and len(levels) < 10:
        a = levels[-1]
        n = a.shape[0] // 2
        levels.append(cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA))
    sizes = np.array([l.shape[0] for l in levels], np.int64)
    offs = np.zeros(len(levels), np.int64)
    o = 0
    for i, l in enumerate(levels):
        offs[i] = o
        o += l.shape[0] * l.shape[0]
    flat = np.concatenate([l.reshape(-1, 5) for l in levels], 0)
    return dict(flat=np.ascontiguousarray(flat), sizes=sizes, offs=offs)


# ------------------------------------------------------------------ rendering

@njit(cache=True, fastmath=True, inline='always')
def _samp(flat, offs, sizes, lev, u, v, out):
    """Bilinear sample (wrap) of level lev at tile coords u, v in level-0 texels; accumulates into out."""
    n = sizes[lev]
    sc = n / sizes[0]
    x = u * sc - 0.5
    y = v * sc - 0.5
    x0 = math.floor(x)
    y0 = math.floor(y)
    fx = x - x0
    fy = y - y0
    i0 = int(x0) % n
    j0 = int(y0) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    o = offs[lev]
    a00 = o + j0 * n + i0
    a01 = o + j0 * n + i1
    a10 = o + j1 * n + i0
    a11 = o + j1 * n + i1
    w00 = (1 - fx) * (1 - fy)
    w01 = fx * (1 - fy)
    w10 = (1 - fx) * fy
    w11 = fx * fy
    for k in range(5):
        out[k] = flat[a00, k] * w00 + flat[a01, k] * w01 + flat[a10, k] * w10 + flat[a11, k] * w11


@njit(cache=True, fastmath=True)
def _sst(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3 - 2 * t)


@njit(cache=True, fastmath=True)
def _sample(flat, offs, sizes, e, c_, s_, fs, x, zw, z, smp, smp2):
    nlev = sizes.shape[0]
    u = (c_ * x - s_ * zw) / e
    v = (s_ * x + c_ * zw) / e
    lod = math.log2(max(z / (fs * e) * 1.3, 1.0))
    l0 = int(lod)
    if l0 >= nlev - 1:
        l0 = nlev - 2
        fl = 1.0
    else:
        fl = lod - l0
    _samp(flat, offs, sizes, l0, u, v, smp)
    if fl > 0.02:
        _samp(flat, offs, sizes, l0 + 1, u, v, smp2)
        for k in range(5):
            smp[k] = smp[k] * (1 - fl) + smp2[k] * fl
    # large non-periodic swells (rows of heaped ridges with broad troughs between): world units,
    # added to the height and to the (tile-axis) gradient
    sh_ = 0.0
    gxs_ = 0.0
    gzs_ = 0.0
    for i in range(3):
        if i == 0:
            a, kx, kz, ph = 1.1, 0.012, 0.085, 0.7
        elif i == 1:
            a, kx, kz, ph = 0.7, -0.03, 0.13, 2.1
        elif i == 2:
            a, kx, kz, ph = 0.6, 0.05, 0.05, 4.0
        else:
            a, kx, kz, ph = 0.35, -0.07, 0.21, 5.3
        ang = kx * x + kz * zw + ph
        sh_ += a * math.sin(ang)
        cs = a * math.cos(ang)
        gxs_ += cs * kx
        gzs_ += cs * kz
    smp[0] += sh_
    # world gradient -> tile axes (u = c*x - s*z, v = s*x + c*z); inverse of the render's conversion
    smp[1] += gxs_ * c_ - gzs_ * s_
    smp[2] += gxs_ * s_ + gzs_ * c_


@njit(cache=True, fastmath=True, parallel=True)
def render_sea(flat, offs, sizes, e, rot, Ws, Hs, fs, hys, camx, camy, camz, sunx, pal, L, light, znear, zfar,
               zmid, zfog, t, rgb, Z):
    """March + shade. rgb (Hs, Ws, 3), Z (Hs, Ws) (inf = sky) are outputs (supersampled canvas).
    Steps grow geometrically; where the surface jumps up on screen by more than a few px (a flank seen
    edge-on) the step is subdivided so neighbouring columns agree (no vertical streaks)."""
    c_, s_ = math.cos(rot), math.sin(rot)
    for col in prange(Ws):
        dx = (col + 0.5 - Ws * 0.5) / fs
        smp = np.zeros(5, np.float32)
        smp2 = np.zeros(5, np.float32)
        yb = Hs                           # first filled row
        z = znear
        ppy = -1e9
        pa = np.zeros(6, np.float32)
        ca = np.zeros(6, np.float32)
        ddx = (col + 0.5 - sunx) / Ws
        prox = math.exp(-(ddx / 0.34) ** 2)
        prox2 = math.exp(-(ddx / 0.1) ** 2)
        while z < zfar and yb > 0:
            zn = z * 1.008 + 0.002
            _sample(flat, offs, sizes, e, c_, s_, fs, camx + dx * zn, camz + zn, zn, smp, smp2)
            pyn = hys - (smp[0] - camy) * fs / zn
            nsub = 1
            if yb - pyn > 4.0 and ppy > -1e8:
                nsub = int((yb - pyn) / 2.0) + 1
                if nsub > 12:
                    nsub = 12
            for jj in range(1, nsub + 1):
                zz0 = z + (zn - z) * jj / nsub
                if jj < nsub:
                    _sample(flat, offs, sizes, e, c_, s_, fs, camx + dx * zz0, camz + zz0, zz0, smp, smp2)
                elif nsub > 1:
                    _sample(flat, offs, sizes, e, c_, s_, fs, camx + dx * zn, camz + zn, zn, smp, smp2)
                hw = smp[0]
                py = hys - (hw - camy) * fs / zz0
                gxw = smp[1] * c_ + smp[2] * s_
                gzw = -smp[1] * s_ + smp[2] * c_
                ca[0] = zz0
                ca[1] = hw
                ca[2] = gxw
                ca[3] = gzw
                ca[4] = smp[3]
                ca[5] = smp[4]
                if py < yb - 0.5:
                    r0 = int(math.ceil(py - 0.5))
                    if r0 < 0:
                        r0 = 0
                    for r in range(r0, yb):
                        tt = 0.0
                        if ppy > py:
                            tt = (r + 0.5 - py) / (ppy - py)
                            if tt > 1.0:
                                tt = 1.0
                            if tt < 0.0:
                                tt = 0.0
                        zz = ca[0] + (pa[0] - ca[0]) * tt
                        hh = ca[1] + (pa[1] - ca[1]) * tt
                        g1 = ca[2] + (pa[2] - ca[2]) * tt
                        g2 = ca[3] + (pa[3] - ca[3]) * tt
                        shd = ca[4] + (pa[4] - ca[4]) * tt
                        cv = ca[5] + (pa[5] - ca[5]) * tt
                        Z[r, col] = zz
                        _shade(rgb, r, col, zz, hh, g1, g2, shd, cv, prox, prox2, pal, L, light, zmid, zfog)
                    yb = r0
                ppy = py
                for k in range(6):
                    pa[k] = ca[k]
            z = zn
        for r in range(0, yb):
            Z[r, col] = np.inf


@njit(cache=True, fastmath=True, inline='always')
def _shade(rgb, r, col, z, h, gx, gz, sh, cav, prox, prox2, pal, L, light, zmid, zfog):
    # normal
    nx, ny, nz = -gx, 1.0, -gz
    nl = math.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nl
    ny /= nl
    nz /= nl
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    w = (ndl + 0.15) / 1.15
    # the painted terminator: one crisp-ish step, cast shadow on top
    hv0 = _sst(-1.2, 0.3, h)
    up0 = _sst(0.75, 0.97, ny)
    shf = sh + (1.0 - sh) * (up0 * hv0 * 0.25)
    lit = _sst(0.17, 0.24, w) * (0.3 + 0.7 * _sst(0.25, 0.7, shf)) * (1.0 - 0.85 * _sst(0.15, 0.7, cav))
    lit = lit * (0.85 + 0.15 * light) * _sst(-1.6, -0.9, h)
    # shadow body: deep indigo valley -> blue-violet face -> lavender sky fill on up-facing planes
    hv = _sst(-1.8, 0.4, h)
    up = _sst(0.82, 0.99, ny)
    fr = _sst(0.0, 0.6, -nz)                 # faces the camera (away from the sun)
    for k in range(3):
        sc = pal[0, k] + (pal[1, k] - pal[0, k]) * hv
        sc = sc + (pal[2, k] - sc) * (up * 0.55 * (1 - 0.5 * fr) * hv * hv * (1 - cav))
        sc = sc + (pal[0, k] - sc) * (_sst(0.1, 0.8, cav) * 0.75)
        # translucent warm glow in the shadow near the sun
        sc = sc + (pal[3, k] - sc) * (0.22 * prox2 * (1 - fr * 0.5))
        # lit colour by sun azimuth: pink (far) -> peach -> gold (near) -> cream core
        a1 = _sst(0.0, 0.55, prox)
        a2 = _sst(0.45, 1.0, prox)
        lc = pal[4, k] + (pal[5, k] - pal[4, k]) * a1
        lc = lc + (pal[6, k] - lc) * a2
        hot = _sst(0.45, 0.85, w) * prox * (0.55 + 0.45 * light)
        lc = lc + (pal[7, k] - lc) * hot
        c = sc + (lc - sc) * lit
        # narrow pink/magenta band at the terminator
        band = 4.0 * lit * (1 - lit)
        c = c + (pal[3, k] - c) * (band * 0.3)
        # aerial perspective: mid haze (bluer), far haze (pink -> gold under the sun)
        fm = (1 - math.exp(-z / zmid)) * (0.22 - 0.12 * lit)
        c = c + (pal[8, k] - c) * fm
        ff = 1 - math.exp(-z / zfog)
        ff = ff ** 1.25
        hz = pal[9, k] + (pal[10, k] - pal[9, k]) * _sst(0.0, 0.9, prox)
        hz = hz + (pal[11, k] - hz) * (prox2 * _sst(0.4, 1.0, ff))
        c = c + (hz - c) * ff
        rgb[r, col, k] = c


@njit(cache=True, fastmath=True, parallel=True)
def rim_pass(rgb, Z, sunx, pal, kmax, wrim, wglow, rim_amt, glow_amt, light):
    """Silver-gold rim along every upper silhouette: walk each column top -> bottom, remember the last
    depth discontinuity (surface starting in front of something much farther) and paint by distance."""
    Hs, Ws = Z.shape
    for c in prange(Ws):
        ddx = (c + 0.5 - sunx) / Ws
        prox = math.exp(-(ddx / 0.2) ** 2)
        prox2 = math.exp(-(ddx / 0.06) ** 2)
        e_row = -100000
        ze = 0.0
        for r in range(1, Hs):
            z0 = Z[r, c]
            if z0 == np.inf:
                continue
            za = Z[r - 1, c]
            if za > z0 * 1.1 + 0.4:
                e_row = r
                ze = z0
            d = r - e_row
            if d >= kmax:
                continue
            if z0 > ze * 1.25 + 0.5:
                continue
            fz = math.exp(-z0 / 900.0)
            rim = math.exp(-d / wrim) * rim_amt * (0.3 + 0.7 * prox) * fz
            gl = math.exp(-d / wglow) * glow_amt * (0.15 + 0.85 * prox2) * fz * (0.7 + 0.5 * light)
            if rim > 1.0:
                rim = 1.0
            for k in range(3):
                rc = pal[12, k] + (pal[13, k] - pal[12, k]) * prox
                v = rgb[r, c, k]
                v = v + (pal[14, k] - v) * gl
                v = v + (rc - v) * rim
                rgb[r, c, k] = v


@njit(cache=True, fastmath=True, parallel=True)
def resolve(rgb, Z, ss, ztw, out, alpha, front):
    """Supersampled sea -> (H, W): premultiplied average colour (divided back), coverage, and for every
    tower depth ztw[k] the fraction of subsamples in front of it."""
    H, W = alpha.shape
    nk = ztw.shape[0]
    inv = 1.0 / (ss * ss)
    for i in prange(H):
        for j in range(W):
            r = 0.0
            g = 0.0
            b = 0.0
            n = 0.0
            for k in range(nk):
                front[k, i, j] = 0.0
            for a in range(ss):
                for c in range(ss):
                    y = i * ss + a
                    x = j * ss + c
                    z = Z[y, x]
                    if z < 1e30:
                        r += rgb[y, x, 0]
                        g += rgb[y, x, 1]
                        b += rgb[y, x, 2]
                        n += 1.0
                        for k in range(nk):
                            if z < ztw[k]:
                                front[k, i, j] += inv
            alpha[i, j] = n * inv
            if n > 0:
                out[i, j, 0] = r / n
                out[i, j, 1] = g / n
                out[i, j, 2] = b / n
            else:
                out[i, j, 0] = 0.0
                out[i, j, 1] = 0.0
                out[i, j, 2] = 0.0
