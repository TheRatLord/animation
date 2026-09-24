"""3D cloud-sea heightfield for s10 (built once, rendered every frame with a numba column march).

World: x right, y up, z forward (toward the sun).  The sea of clouds is a periodic heightfield h(x, z)
(period P) built from embedded DOMES at five scales: big cloud masses, lobes growing on them, bumps
on the lobes, and cauliflower 'pebbles' on the bumps.  Each child dome is centred just below its parent's
surface, so the union (max) forms crisp creases where lobes meet - the painted overlapping-lobe lines -
while every lobe's cap is smooth.  A non-periodic macro map raises / lowers whole regions (breaks the
tiling and makes deep violet valleys and high sunlit ridges).

Baked per texel (and for every mip level): height, normal (nx, nz), cast shadow from the low sun,
cavity (crease darkness).  Rendering (render_sea):
  * voxel-space march: every screen column is a vertical plane (no roll/pitch, horizon from a lens
    shift); samples are taken front-to-back with geometrically growing steps, trilinear mip filtered by
    the pixel footprint (no shimmer), and fill the column bottom-up (y-buffer), interpolating depth
    and world position inside each filled span (smooth surfaces, sub-sample precise silhouettes).
  * shading is painted: two tones split by a crisp terminator (shadow: indigo -> violet, lit: rose ->
    peach -> gold), cast shadows, crease lines, silver-gold rims along crests seen against farther
    cloud, forward-scatter glow toward the sun, aerial perspective to the horizon haze.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

NCH = 7   # texel channels: height, detail normal (nx, nz), form normal (nx, nz), shadow, cavity


# ============================================================================================ building

@njit(cache=True)
def _raster_domes(Hm, D, e, N):
    """Union (max) of dome caps into the periodic map Hm (N x N, texel e). D rows: x, z, by, r, ay[, pw]."""
    for k in range(D.shape[0]):
        x, z, by, r, ay = D[k, 0], D[k, 1], D[k, 2], D[k, 3], D[k, 4]
        pw = D[k, 5] if D.shape[1] > 5 else 0.8
        i0 = int(math.floor((z - r) / e))
        i1 = int(math.ceil((z + r) / e))
        j0 = int(math.floor((x - r) / e))
        j1 = int(math.ceil((x + r) / e))
        r2 = r * r
        for i in range(i0, i1 + 1):
            dz = (i + 0.5) * e - z
            ii = i % N
            for j in range(j0, j1 + 1):
                dx = (j + 0.5) * e - x
                d2 = dx * dx + dz * dz
                if d2 >= r2:
                    continue
                h = by + ay * r * (1.0 - d2 / r2) ** pw
                jj = j % N
                if h > Hm[ii, jj]:
                    Hm[ii, jj] = h


@njit(cache=True)
def _sample_periodic(Hm, x, z, e, N):
    fx = x / e - 0.5
    fz = z / e - 0.5
    j0 = int(math.floor(fx))
    i0 = int(math.floor(fz))
    tx = fx - j0
    tz = fz - i0
    a = Hm[i0 % N, j0 % N]
    b = Hm[i0 % N, (j0 + 1) % N]
    c = Hm[(i0 + 1) % N, j0 % N]
    d = Hm[(i0 + 1) % N, (j0 + 1) % N]
    return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz


@njit(cache=True)
def _children(Hm, e, N, P, seed, cell, rmin, rmax, aymin, aymax, emb0, emb1, hmin, jit, gmax=100.0):
    """Jittered-grid child domes centred just below the current surface. Returns (n, 5)."""
    np.random.seed(seed)
    n = int(P / cell)
    out = np.zeros((n * n, 5), np.float64)
    k = 0
    for gi in range(n):
        for gj in range(n):
            x = (gj + 0.5 + jit * (np.random.random() - 0.5)) * cell
            z = (gi + 0.5 + jit * (np.random.random() - 0.5)) * cell
            r = rmin + (rmax - rmin) * np.random.random()
            ay = aymin + (aymax - aymin) * np.random.random()
            emb = emb0 + (emb1 - emb0) * np.random.random()
            h = _sample_periodic(Hm, x, z, e, N)
            if h < hmin:
                continue
            gx = (_sample_periodic(Hm, x + r * 0.5, z, e, N) - _sample_periodic(Hm, x - r * 0.5, z, e, N)) / r
            gz = (_sample_periodic(Hm, x, z + r * 0.5, e, N) - _sample_periodic(Hm, x, z - r * 0.5, e, N)) / r
            if gx * gx + gz * gz > gmax * gmax:
                continue
            out[k, 0] = x
            out[k, 1] = z
            out[k, 2] = h - emb * r * ay
            out[k, 3] = r
            out[k, 4] = ay
            k += 1
    return out[:k]


@njit(cache=True, parallel=True)
def _shadow(Hm, e, N, lx, lz, tanel, maxd, soft):
    """Soft cast-shadow (1 = fully shadowed) from a sun in direction (lx, lz) at elevation atan(tanel)."""
    S = np.zeros((N, N), np.float32)
    nstep = int(maxd / e)
    for i in prange(N):
        for j in range(N):
            h0 = Hm[i, j]
            occ = 0.0
            s = e
            while s < maxd:
                x = (j + 0.5) * e + lx * s
                z = (i + 0.5) * e + lz * s
                h = _sample_periodic(Hm, x, z, e, N)
                d = (h - (h0 + s * tanel)) / (soft * s + 0.02)
                if d > occ:
                    occ = d
                    if occ >= 1.0:
                        break
                s += e * (1.0 + s * 0.6)
            S[i, j] = min(occ, 1.0)
    return S


def _periodic_blur(img, sigma):
    N = img.shape[0]
    pad = int(3 * sigma) + 2
    big = np.pad(img, ((pad, pad), (pad, pad)), mode='wrap')
    return cv2.GaussianBlur(big, (0, 0), sigma)[pad:pad + N, pad:pad + N]


def _normals(Hm, e):
    gx = (np.roll(Hm, -1, 1) - np.roll(Hm, 1, 1)) / (2 * e)
    gz = (np.roll(Hm, -1, 0) - np.roll(Hm, 1, 0)) / (2 * e)
    ln = np.sqrt(gx * gx + gz * gz + 1.0)
    return (-gx / ln).astype(np.float32), (-gz / ln).astype(np.float32)


def build_field(seed=11, u=1.0, P=80.0, N=2048, sun_az=0.0, sun_el_deg=22.0, mounds=True):
    """Cloud-deck heightfield. u: unit length (all feature sizes scale with it); P: tile period (world);
    N: texels. Returns dict with the packed mip pyramid (flat float32 array + offsets) and metadata."""
    rng = np.random.default_rng(seed)
    e = P / N
    xs = (np.arange(N) + 0.5) * e
    # ---- floor: gentle periodic swell
    X, Z = np.meshgrid(xs, xs)
    fl = np.zeros((N, N), np.float64)
    for k in range(6):
        kx, kz = rng.integers(-3, 4), rng.integers(1, 4)
        ph = rng.uniform(0, 2 * math.pi)
        fl += 0.1 * u * math.cos(ph) * np.cos(2 * math.pi * (kx * X + kz * Z) / P + ph)
    Hm = (fl - 0.25 * u).astype(np.float64)
    # ---- low stratiform billows everywhere (the deck is cloud too)
    D = _children(Hm, e, N, P, seed * 7 + 1, 2.4 * u, 1.1 * u, 2.0 * u, 0.2, 0.3, 0.3, 0.6, -1e9, 0.95)
    _raster_domes(Hm, D, e, N)
    # ---- soft low heaps (rolling, never cliffs: parabolic-ish profile)
    if mounds:
        heaps = []
        cell = 5.0 * u
        n = int(P / cell)
        for gi in range(n):
            for gj in range(n):
                if rng.random() < 0.35:
                    continue
                x = (gj + 0.5 + rng.uniform(-0.45, 0.45)) * cell
                z = (gi + 0.5 + rng.uniform(-0.45, 0.45)) * cell
                r = rng.uniform(1.8, 3.4) * u
                ay = rng.uniform(0.24, 0.4)
                heaps.append((x, z, -0.2 * r * ay, r, ay, 0.9))
        _raster_domes(Hm, np.array(heaps, np.float64), e, N)
    # ---- hierarchical lobes (each level grows on the surface built so far), limited on steep slopes
    levels = [  # cell, rmin, rmax, aymin, aymax, emb0, emb1, gmax
        (1.05, 0.5, 0.9, 0.5, 0.75, 0.55, 0.8, 1.2),
        (0.5, 0.24, 0.42, 0.55, 0.8, 0.55, 0.8, 1.0),
        (0.26, 0.12, 0.2, 0.55, 0.75, 0.6, 0.85, 0.8),
    ]
    for li, (c, r0, r1, a0, a1, e0, e1, gm) in enumerate(levels):
        D = _children(Hm, e, N, P, seed * 17 + li, c * u, r0 * u, r1 * u, a0, a1, e0, e1, -1e9, 0.95, gm)
        _raster_domes(Hm, D, e, N)
    Hm = Hm.astype(np.float32)
    # ---- baked maps
    lx, lz = math.sin(sun_az), math.cos(sun_az)
    tanel = math.tan(math.radians(sun_el_deg))
    Hs_ = _periodic_blur(Hm, 0.5 * u / e)                  # only the big forms cast shadows (big, simple shapes)
    Sh = _shadow(Hs_, e, N, lx, lz, tanel, 10.0 * u, 0.15)
    Sh = _periodic_blur(Sh, 1.0)
    cav = np.clip((_periodic_blur(Hm, 2.5) - Hm) / (0.08 * u), 0, 1).astype(np.float32)
    # ---- mip pyramid (area filtered; normals recomputed from each filtered level).  The 'form' normal
    #      comes from a smoothed height (big light shapes); the detail normal adds the lobes.
    form_sig = 0.45 * u / e
    levs = []
    h, s_, c_ = Hm, Sh, cav
    hf = _periodic_blur(Hm, form_sig)
    ee = e
    while True:
        nx, nz = _normals(h, ee)
        fx, fz = _normals(hf, ee)
        levs.append(np.stack([h, nx, nz, fx, fz, s_, c_], -1).astype(np.float32))
        if h.shape[0] <= 8:
            break
        h, s_, c_, hf = [cv2.resize(a, (a.shape[1] // 2, a.shape[0] // 2), interpolation=cv2.INTER_AREA)
                         for a in (h, s_, c_, hf)]
        ee *= 2
    offs = np.zeros(len(levs) + 1, np.int64)
    for i, L in enumerate(levs):
        offs[i + 1] = offs[i] + L.size
    flat = np.concatenate([L.ravel() for L in levs]).astype(np.float32)
    return dict(flat=flat, offs=offs, nlev=len(levs), N=N, P=P, e=e, hmax=float(Hm.max()),
                hmean=float(Hm.mean()))


def build_macro(seed=3, extent=3000.0, n=768):
    """Non-periodic large-scale height offset map covering [-extent/2, extent/2] x [-50, extent-50]."""
    rng = np.random.default_rng(seed)
    m = np.zeros((n, n), np.float32)
    amp, tot = 1.0, 0.0
    for cells in (5, 11, 23, 47):
        g = rng.standard_normal((cells + 3, cells + 3)).astype(np.float32)
        m += amp * cv2.resize(g, (n, n), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.55
    m /= tot
    m = (m - m.mean()) / (m.std() + 1e-6)
    return dict(M=np.ascontiguousarray(m), x0=-extent / 2, z0=-50.0, me=extent / n)


# ============================================================================================ sampling

@njit(cache=True, fastmath=True)
def _tex(flat, offs, N, lev, ch, x, z, e):
    """Bilinear periodic fetch of channel ch at mip level lev (integer)."""
    n = N >> lev
    ee = e * (1 << lev)
    fx = x / ee - 0.5
    fz = z / ee - 0.5
    j0 = int(math.floor(fx))
    i0 = int(math.floor(fz))
    tx = fx - j0
    tz = fz - i0
    j0 %= n
    i0 %= n
    j1 = (j0 + 1) % n
    i1 = (i0 + 1) % n
    o = offs[lev]
    a = flat[o + (i0 * n + j0) * NCH + ch]
    b = flat[o + (i0 * n + j1) * NCH + ch]
    c = flat[o + (i1 * n + j0) * NCH + ch]
    d = flat[o + (i1 * n + j1) * NCH + ch]
    return (a * (1 - tx) + b * tx) * (1 - tz) + (c * (1 - tx) + d * tx) * tz


@njit(cache=True, fastmath=True)
def _tri(flat, offs, nlev, N, ch, x, z, e, lod):
    if lod <= 0.0:
        return _tex(flat, offs, N, 0, ch, x, z, e)
    l0 = int(lod)
    if l0 >= nlev - 1:
        return _tex(flat, offs, N, nlev - 1, ch, x, z, e)
    t = lod - l0
    return _tex(flat, offs, N, l0, ch, x, z, e) * (1 - t) + _tex(flat, offs, N, l0 + 1, ch, x, z, e) * t


@njit(cache=True, fastmath=True)
def _macro(M, x0, z0, me, x, z):
    n = M.shape[0]
    fx = (x - x0) / me - 0.5
    fz = (z - z0) / me - 0.5
    fx = min(max(fx, 0.0), n - 1.001)
    fz = min(max(fz, 0.0), n - 1.001)
    j0 = int(fx)
    i0 = int(fz)
    tx = fx - j0
    tz = fz - i0
    return (M[i0, j0] * (1 - tx) + M[i0, j0 + 1] * tx) * (1 - tz) + (M[i0 + 1, j0] * (1 - tx) + M[i0 + 1, j0 + 1] * tx) * tz


# ============================================================================================ render

@njit(cache=True, fastmath=True, parallel=True)
def march(Ws, Hs, fs, hys, camx, camy, camz, yaw, flat, offs, nlev, N, e, M, mx0, mz0, me, macro_amp,
          znear, zfar, kstep, lod_bias, ushift=0.0):
    """Column march. Returns per-pixel depth Z (camera forward distance; 0 = sky), world x, z and the
    mip lod used, at the (supersampled) resolution Ws x Hs."""
    Zb = np.zeros((Hs, Ws), np.float32)
    Xb = np.zeros((Hs, Ws), np.float32)
    Wb = np.zeros((Hs, Ws), np.float32)
    Lb = np.zeros((Hs, Ws), np.float32)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    for col in prange(Ws):
        u = (col + 0.5 - Ws * 0.5 - ushift) / fs
        # camera-space ray (u, *, 1) rotated by yaw about the y axis
        dx = u * cy_ + sy_
        dz = -u * sy_ + cy_
        ybuf = Hs            # next row to fill is ybuf-1
        Z = znear
        py = -1.0
        pZ = Z
        while Z < zfar and ybuf > 0:
            wx = camx + dx * Z
            wz = camz + dz * Z
            fp = Z / fs          # world size of a pixel at this depth
            lod = math.log2(max(fp / e, 1e-6)) + lod_bias
            if lod < 0.0:
                lod = 0.0
            h = _tri(flat, offs, nlev, N, 0, wx, wz, e, lod)
            h += macro_amp * _macro(M, mx0, mz0, me, wx, wz)
            y = hys - fs * (h - camy) / Z
            if py < 0.0:
                py = y
                pZ = Z
            if y < ybuf:
                # fill rows [ceil(y)?, ybuf) with depth interpolated between previous and current sample
                while ybuf > 0 and ybuf - 0.5 > y:
                    r = ybuf - 1
                    yc = r + 0.5
                    if py - y > 1e-6:
                        tt = (py - yc) / (py - y)
                    else:
                        tt = 1.0
                    if tt < 0.0:
                        tt = 0.0
                    if tt > 1.0:
                        tt = 1.0
                    Zr = pZ + (Z - pZ) * tt
                    Zb[r, col] = Zr
                    Xb[r, col] = camx + dx * Zr
                    Wb[r, col] = camz + dz * Zr
                    Lb[r, col] = math.log2(max(Zr / fs / e, 1e-6)) + lod_bias
                    ybuf -= 1
            py = y
            pZ = Z
            Z += max(Z * kstep, 0.004)
    return Zb, Xb, Wb, Lb


@njit(cache=True, fastmath=True)
def _sstep(a, b, x):
    t = (x - a) / (b - a)
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0
    return t * t * (3 - 2 * t)


@njit(cache=True, fastmath=True, parallel=True)
def shade(Zb, Xb, Wb, Lb, flat, offs, nlev, N, e, M, mx0, mz0, me, macro_amp, Lx, Ly, Lz, P,
          fs, hys, sux, suy, hmax, dw, zf):
    """Painted shading. P: palette (K, 3) array (see scene). Returns rgb (Hs, Ws, 3), alpha (Hs, Ws).
    sux, suy: sun position on the supersampled screen."""
    Hs, Ws = Zb.shape
    out = np.zeros((Hs, Ws, 3), np.float32)
    al = np.zeros((Hs, Ws), np.float32)
    for r in prange(Hs):
        for c in range(Ws):
            Z = Zb[r, c]
            if Z <= 0.0:
                continue
            al[r, c] = 1.0
            x = Xb[r, c]
            z = Wb[r, c]
            lod = Lb[r, c]
            if lod < 0.0:
                lod = 0.0
            h = _tri(flat, offs, nlev, N, 0, x, z, e, lod)
            nx = _tri(flat, offs, nlev, N, 1, x, z, e, lod)
            nz = _tri(flat, offs, nlev, N, 2, x, z, e, lod)
            fx = _tri(flat, offs, nlev, N, 3, x, z, e, lod)
            fz = _tri(flat, offs, nlev, N, 4, x, z, e, lod)
            sh = _tri(flat, offs, nlev, N, 5, x, z, e, lod)
            cv = _tri(flat, offs, nlev, N, 6, x, z, e, lod)
            # painted normal: mostly the big form, the lobes only modulate it
            nx = nx * dw + fx * (1.0 - dw)
            nz = nz * dw + fz * (1.0 - dw)
            ny = math.sqrt(max(1.0 - nx * nx - nz * nz, 0.02))
            hm = h + macro_amp * _macro(M, mx0, mz0, me, x, z)
            hv = min(max(hm / hmax, 0.0), 1.0)       # 0 = deep in the deck, 1 = the highest crowns
            ndl = nx * Lx + ny * Ly + nz * Lz
            # painted two-tone split: crisp terminator, cast shadows cut into the light
            lit = _sstep(-0.06, 0.1, ndl) * (1.0 - 0.75 * _sstep(0.25, 0.75, sh))
            # far away the lobes merge: fade the structure toward a mid value
            far = _sstep(3.0, 7.5, lod)
            # --- shadow tone: deep indigo in the valleys, violet on up-facing, lavender-pink bounce
            up = _sstep(0.55, 1.0, ny)
            t1 = hv * 0.6 + up * 0.4
            sr = P[0, 0] + (P[1, 0] - P[0, 0]) * t1
            sg = P[0, 1] + (P[1, 1] - P[0, 1]) * t1
            sb = P[0, 2] + (P[1, 2] - P[0, 2]) * t1
            # --- lit tone: rose at the terminator -> peach -> gold on the caps
            k = _sstep(0.0, 0.55, ndl)
            if k < 0.5:
                q = k * 2
                lr = P[2, 0] + (P[3, 0] - P[2, 0]) * q
                lg = P[2, 1] + (P[3, 1] - P[2, 1]) * q
                lb = P[2, 2] + (P[3, 2] - P[2, 2]) * q
            else:
                q = (k - 0.5) * 2
                lr = P[3, 0] + (P[4, 0] - P[3, 0]) * q
                lg = P[3, 1] + (P[4, 1] - P[3, 1]) * q
                lb = P[3, 2] + (P[4, 2] - P[3, 2]) * q
            # toward the sun's column the light turns gold (same hue as the glare)
            sxx = (c + 0.5 - sux) / Ws
            wgold = math.exp(-(sxx / 0.22) ** 2) * 0.5
            lr = lr + (P[6, 0] - lr) * wgold
            lg = lg + (P[6, 1] - lg) * wgold
            lb = lb + (P[6, 2] - lb) * wgold
            # lower lit surfaces a bit dimmer / pinker
            dim = 0.78 + 0.22 * hv
            lr *= dim
            lg *= dim * (0.94 + 0.06 * hv)
            lb *= dim
            # crease lines (painted separation between lobes)
            cr = _sstep(0.15, 0.7, cv) * (1.0 - far) * 0.5
            cr_r, cr_g, cr_b = P[5, 0], P[5, 1], P[5, 2]
            sr = sr + (cr_r * 0.8 - sr) * cr * 0.6
            sg = sg + (cr_g * 0.8 - sg) * cr * 0.6
            sb = sb + (cr_b * 0.8 - sb) * cr * 0.6
            lr = lr + (cr_r - lr) * cr
            lg = lg + (cr_g - lg) * cr
            lb = lb + (cr_b - lb) * cr
            rr = sr + (lr - sr) * lit
            gg = sg + (lg - sg) * lit
            bb = sb + (lb - sb) * lit
            # forward scatter toward the sun (screen space proximity): warm the whole sea there
            sx = (c + 0.5 - sux) / Ws
            sy = (r + 0.5 - suy) / Ws
            dsun = math.sqrt(sx * sx * 0.6 + sy * sy * 2.5)
            g = math.exp(-dsun / 0.1) * 0.45 + math.exp(-dsun / 0.3) * 0.12
            g = g * (0.45 + 0.55 * lit)
            # glare path: the tops between the sun and the viewer catch golden forward-scattered light
            gp = math.exp(-(sx / 0.075) ** 2) * (0.15 + 0.85 * lit) * _sstep(-0.05, 0.3, ndl) * 0.55
            g = min(g + gp, 0.9)
            wr = rr + (P[6, 0] * 1.08 - rr) * g
            wg = gg + (P[6, 1] * 1.04 - gg) * g
            wb = bb + (P[6, 2] - bb) * g
            # aerial perspective: haze toward the horizon, golden around the sun, rose-violet away from it
            fa = 1.0 - math.exp(-(Z / zf) ** 0.85)
            gx = math.exp(-abs(sx) / 0.22)
            hr = P[7, 0] + (P[8, 0] - P[7, 0]) * gx
            hg = P[7, 1] + (P[8, 1] - P[7, 1]) * gx
            hb = P[7, 2] + (P[8, 2] - P[7, 2]) * gx
            out[r, c, 0] = wr + (hr - wr) * fa
            out[r, c, 1] = wg + (hg - wg) * fa
            out[r, c, 2] = wb + (hb - wb) * fa
    return out, al


@njit(cache=True, fastmath=True, parallel=True)
def rims(Zb, fs, width, gap):
    """Crest rims: for each pixel, strength of the silhouette edge right above it (a crest seen against
    farther cloud or sky).  width: rim width in world units; gap: relative depth jump that counts."""
    Hs, Ws = Zb.shape
    R = np.zeros((Hs, Ws), np.float32)
    for c in prange(Ws):
        edge_z = 0.0
        edge_s = 0.0
        edge_r = -1
        for r in range(Hs):
            Z = Zb[r, c]
            if Z <= 0.0:
                edge_r = -1
                continue
            Za = Zb[r - 1, c] if r > 0 else 0.0
            if Za <= 0.0 or Za > Z * (1.0 + gap):
                # new silhouette starts here
                edge_r = r
                edge_z = Z
                if Za <= 0.0:
                    edge_s = 1.0
                else:
                    edge_s = min((Za / Z - 1.0 - gap) / (gap * 4.0), 1.0)
            if edge_r >= 0:
                wpx = fs * width / edge_z
                d = (r - edge_r + 0.5) / max(wpx, 0.8)
                if d < 4.0 and Z < edge_z * (1.0 + gap):
                    R[r, c] = edge_s * math.exp(-d * d)
                else:
                    edge_r = -1
    return R


@njit(cache=True, fastmath=True, parallel=True)
def head_shadows(Z, f, cx, camx, zs, x0s, ppus, feet, nfoot, ls, tanaz, soft):
    """Shadows of the cumulus heads (rows at depths zs with top-height footprints `feet`) on the deck.
    Z: deck depth per pixel (0 = none); cx: screen x of the optical axis (W/2 + yaw shift);
    ls: shadow length per unit height; tanaz: sideways lean of the shadows. Returns (H, W) 0..1."""
    H, W = Z.shape
    S = np.zeros((H, W), np.float32)
    nr = zs.shape[0]
    for r in prange(H):
        for c in range(W):
            z = Z[r, c]
            if z <= 0.0:
                continue
            X = camx + (c + 0.5 - cx) * z / f
            best = 0.0
            for k in range(nr):
                dz = zs[k] - z
                if dz < -0.4 or dz > 6.0:
                    continue
                xo = X + dz * tanaz
                fi = (xo - x0s[k]) * ppus[k]
                i = int(fi)
                if i < 0 or i >= nfoot[k]:
                    continue
                h = feet[k, i]
                if h <= 0.02:
                    continue
                L = ls * h
                s = 1.0 - min(max((dz - L + soft) / (2 * soft), 0.0), 1.0)
                s *= min(max((dz + 0.4) / 0.3, 0.0), 1.0)
                if s > best:
                    best = s
            S[r, c] = best
    return S
