"""s10 helper: the sea of clouds as a painted heightfield.

The cloud deck is a periodic heightfield built from hierarchically stacked, SMOOTH-UNIONED puffs
(big mounds -> lobes -> puffs -> cauliflower detail), so every mound is a cluster of small bumps with
filleted joins (no sphere-intersection creases).  Two tiles: a large 'deck' tile (mounds/lobes/puffs)
and a fine 'detail' tile (cauliflower bumps) that slowly drifts (boil).  Both have Gaussian mip levels
so far rows are properly filtered (no shimmer).

Rendering is an exact per-column ray march (heightfield + pinhole camera without roll => the hit depth
is monotonic up a column), shaded in the same pass with a painterly ramp:
    wrapped (half-Lambert) sun term x soft cast shadows (horizon-mapped) x cavity AO, sky term,
    brushed world-space noise along the terminator, a colour ramp indigo -> violet -> magenta -> peach
    -> cream-gold -> white, aerial perspective to the horizon glow, forward-scatter glow toward the sun.
A second pass paints backlit silver-gold rims that bleed a few px inward from every silhouette top.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

EPS0 = 0.03
DCAV = 0.0
DETK = 0.55


# ============================================================================ build: noise + puffs

def spectral_noise(N, lo, hi, seed, beta=1.0, aniso=1.0):
    """Periodic band-limited noise (N x N), zero-mean, unit std. lo/hi in cycles per tile."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((N, N)).astype(np.float32)
    F = np.fft.rfft2(w)
    ky = np.fft.fftfreq(N)[:, None] * N
    kx = np.fft.rfftfreq(N)[None, :] * N / aniso
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = 1.0
    filt = k ** (-beta) * (1.0 / (1.0 + (lo / k) ** 6)) * np.exp(-(k / hi) ** 2)
    filt[0, 0] = 0.0
    out = np.fft.irfft2(F * filt, s=(N, N)).astype(np.float32)
    return out / (out.std() + 1e-9)


@njit(cache=True)
def _raster_puffs(Hm, P, xs, zs, ys, rs, ays, ks, pw, ext=0.05):
    """Smooth-union (polynomial smax) of domes into the periodic heightmap Hm (tile size P).
    Dome profile: (1 - s^2)^pw up to s0, then continued along its tangent to zero (a finite-slope
    skirt instead of a vertical rim: no undersampled cliffs / ribbing on the heightfield)."""
    N = Hm.shape[0]
    e = P / N
    s0 = 0.82
    h0p = (1.0 - s0 * s0) ** pw
    d0p = pw * (1.0 - s0 * s0) ** (pw - 1.0) * 2.0 * s0      # -dh/ds at s0
    s1 = s0 + h0p / d0p + ext           # the skirt continues below the dome base (no cliffs)
    for i in range(xs.shape[0]):
        R = rs[i]
        k = ks[i]
        ri = int(R * s1 / e) + 2
        cx = xs[i] / e
        cz = zs[i] / e
        ix0 = int(math.floor(cx)) - ri
        iz0 = int(math.floor(cz)) - ri
        for a in range(2 * ri + 2):
            iz = iz0 + a
            dz = (iz - cz) * e / R
            if dz * dz >= s1 * s1:
                continue
            izw = iz % N
            for b in range(2 * ri + 2):
                ix = ix0 + b
                dx = (ix - cx) * e / R
                q = dx * dx + dz * dz
                if q >= s1 * s1:
                    continue
                if q < s0 * s0:
                    prof = (1.0 - q) ** pw
                else:
                    prof = h0p - d0p * (math.sqrt(q) - s0)
                v = ys[i] + R * ays[i] * prof
                ixw = ix % N
                h0 = Hm[izw, ixw]
                d = abs(h0 - v)
                if d < k:
                    hh = (k - d) / k
                    Hm[izw, ixw] = max(h0, v) + hh * hh * k * 0.25
                elif v > h0:
                    Hm[izw, ixw] = v


@njit(cache=True)
def _sample(Hm, P, x, z):
    N = Hm.shape[0]
    e = P / N
    fx = x / e - 0.5
    fz = z / e - 0.5
    ix = int(math.floor(fx))
    iz = int(math.floor(fz))
    tx = fx - ix
    tz = fz - iz
    x0 = ix % N
    z0 = iz % N
    x1 = (x0 + 1) % N
    z1 = (z0 + 1) % N
    a = Hm[z0, x0] * (1 - tx) + Hm[z0, x1] * tx
    b = Hm[z1, x0] * (1 - tx) + Hm[z1, x1] * tx
    return a * (1 - tz) + b * tz


@njit(cache=True, fastmath=True)
def _sample_cubic(Hm, P, x, z):
    """Uniform cubic B-spline sample (C2-smooth: no bilinear facets on steep walls)."""
    N = Hm.shape[0]
    e = P / N
    fx = x / e - 0.5
    fz = z / e - 0.5
    ix = int(math.floor(fx))
    iz = int(math.floor(fz))
    tx = fx - ix
    tz = fz - iz
    wx0 = (1 - tx) ** 3 / 6.0
    wx1 = (3 * tx ** 3 - 6 * tx * tx + 4) / 6.0
    wx2 = (-3 * tx ** 3 + 3 * tx * tx + 3 * tx + 1) / 6.0
    wx3 = tx ** 3 / 6.0
    wz0 = (1 - tz) ** 3 / 6.0
    wz1 = (3 * tz ** 3 - 6 * tz * tz + 4) / 6.0
    wz2 = (-3 * tz ** 3 + 3 * tz * tz + 3 * tz + 1) / 6.0
    wz3 = tz ** 3 / 6.0
    acc = 0.0
    for j in range(4):
        wz = wz0 if j == 0 else (wz1 if j == 1 else (wz2 if j == 2 else wz3))
        row = (iz - 1 + j) % N
        x0 = (ix - 1) % N
        x1 = ix % N
        x2 = (ix + 1) % N
        x3 = (ix + 2) % N
        acc += wz * (Hm[row, x0] * wx0 + Hm[row, x1] * wx1 + Hm[row, x2] * wx2 + Hm[row, x3] * wx3)
    return acc


@njit(cache=True, fastmath=True)
def _mipc(M, P, x, z, lv):
    L = M.shape[0]
    if lv <= 0.0:
        return _sample_cubic(M[0], P, x, z)
    if lv >= L - 1:
        return _sample_cubic(M[L - 1], P, x, z)
    i = int(lv)
    t = lv - i
    return _sample_cubic(M[i], P, x, z) * (1 - t) + _sample_cubic(M[i + 1], P, x, z) * t


@njit(cache=True)
def sample_many(Hm, P, xs, zs):
    out = np.empty(xs.shape[0], np.float32)
    for i in range(xs.shape[0]):
        out[i] = _sample(Hm, P, xs[i], zs[i])
    return out


def _spawn(Hm, P, rng, n, rmin, rmax, ay, embed, kf, top_bias, pw_mask=None):
    """Place n puffs sitting on the current surface (preferring high spots): returns arrays."""
    N = Hm.shape[0]
    cand = rng.uniform(0, P, (n * 4, 2))
    hs = sample_many(Hm, P, cand[:, 0], cand[:, 1])
    r = rng.uniform(rmin, rmax, n * 4)
    # the puff rests on the LOWEST point under its rim (no floating edges -> no cliffs)
    hmin = hs.copy()
    for j in range(8):
        a = j * math.pi / 4
        hj = sample_many(Hm, P, cand[:, 0] + math.cos(a) * r * 0.9, cand[:, 1] + math.sin(a) * r * 0.9)
        hmin = np.minimum(hmin, hj)
    lo, hi = np.percentile(hs, 5), np.percentile(hs, 99)
    wgt = np.clip((hs - lo) / (hi - lo + 1e-6), 0, 1) ** top_bias + 0.02
    if pw_mask is not None:
        wgt = wgt * sample_many(pw_mask, P, cand[:, 0], cand[:, 1])
    wgt = wgt / wgt.sum()
    idx = rng.choice(len(cand), n, replace=False, p=wgt)
    x = cand[idx, 0]
    z = cand[idx, 1]
    r = r[idx]
    a_ = ay * rng.uniform(0.75, 1.2, n)
    y = np.minimum(hs[idx] - embed * r * a_ * rng.uniform(0.7, 1.3, n), hmin[idx] - 0.05 * r * a_)
    return (x.astype(np.float64), z.astype(np.float64), y.astype(np.float64), r.astype(np.float64),
            a_.astype(np.float64), (kf * r).astype(np.float64))


def build_deck(seed=5, P=192.0, N=2048):
    """Deck tile: base swell + stratocumulus mounds + lobes + puffs (smooth union)."""
    rng = np.random.default_rng(seed)
    # base sea level: gentle rolling swell with 'lanes' (stratocumulus rows run left-right)
    base = spectral_noise(N, 2.0, 14.0, seed + 1, beta=1.2, aniso=0.55) * 0.75 - 1.4
    Hm = base.astype(np.float32).copy()
    # mound density: open areas (deep valleys) and clustered areas
    dens = spectral_noise(N, 2.0, 8.0, seed + 2, beta=1.0)
    dens = np.clip(0.5 + 0.55 * dens, 0.05, 1.0).astype(np.float32)
    # level 0: big mounds (wide, flattened: stratocumulus heaps), a few tall heaps
    specs = [
        # n, rmin, rmax, ay, embed, k-frac, top_bias, pw
        (120, 6.0, 12.0, 0.4, 0.2, 0.45, 0.0, 0.7),
        (700, 2.0, 4.5, 0.95, 0.35, 0.25, 1.4, 0.65),
        (2600, 0.9, 1.9, 1.0, 0.38, 0.2, 1.8, 0.5),
        (7000, 0.4, 0.85, 1.0, 0.4, 0.2, 2.0, 0.5),
    ]
    for li, (n, rmin, rmax, ay, emb, kf, tb, pw) in enumerate(specs):
        x, z, y, r, a_, k = _spawn(Hm, P, rng, n, rmin, rmax, ay, emb, kf, tb, dens if li == 0 else None)
        if li == 0:
            y = y + rng.uniform(-0.3, 0.6, n) + 0.8 * sample_many(dens, P, x, z)
        _raster_puffs(Hm, P, x, z, y, r, a_, k, pw, 0.4 if li == 0 else 0.08)
    Hm = wrap_blur(Hm, 0.5)
    lo, hi = np.percentile(Hm, 1.0), np.percentile(Hm, 99.8)
    sc = min(1.0, 5.4 / (hi - lo))
    print('deck range', lo, hi, sc)
    return (-2.2 + (Hm - lo) * sc).astype(np.float32)


def build_detail(seed=8, P=24.0, N=2048):
    """Fine cauliflower tile: densely packed small bumps (zero-mean displacement)."""
    rng = np.random.default_rng(seed)
    Hm = (spectral_noise(N, 3.0, 30.0, seed + 1, beta=1.0) * 0.03).astype(np.float32)
    for n, rmin, rmax, ay, emb, kf, tb in [(1500, 0.25, 0.55, 0.6, 0.45, 0.35, 0.0),
                                            (7000, 0.1, 0.25, 0.65, 0.45, 0.3, 0.8),
                                            (20000, 0.05, 0.11, 0.7, 0.45, 0.3, 1.0)]:
        x, z, y, r, a_, k = _spawn(Hm, P, rng, n, rmin, rmax, ay, emb, kf, tb)
        _raster_puffs(Hm, P, x, z, y, r, a_, k, 0.55)
    Hm = Hm - np.percentile(Hm, 50)
    return (Hm / (np.abs(Hm).max() + 1e-6)).astype(np.float32)


def wrap_blur(img, sigma):
    if sigma < 0.3:
        return img.copy()
    pad = int(3 * sigma) + 2
    N = img.shape[0]
    pad = min(pad, N - 1)
    big = np.pad(img, ((pad, pad), (pad, pad)) + ((0, 0),) * (img.ndim - 2), mode='wrap')
    big = cv2.GaussianBlur(big, (0, 0), sigma)
    return big[pad:pad + N, pad:pad + N].copy()


def pyramid(Hm, levels):
    return np.stack([wrap_blur(Hm, 0.0 if L == 0 else 0.85 * 2 ** (L - 1)) for L in range(levels)]).astype(np.float32)


@njit(cache=True, parallel=True)
def horizon_shadow(Hm, P, lx, lz, tanel, maxd, soft):
    """Soft cast shadows on the periodic heightmap for a sun along (lx, lz) with elevation tan."""
    N = Hm.shape[0]
    e = P / N
    out = np.empty_like(Hm)
    for iz in prange(N):
        for ix in range(N):
            x = (ix + 0.5) * e
            z = (iz + 0.5) * e
            h0 = Hm[iz, ix]
            best = -1e9
            s = e * 1.5
            while s < maxd:
                hh = _sample(Hm, P, x + lx * s, z + lz * s)
                v = (hh - h0) / s
                if v > best:
                    best = v
                s += max(e, s * 0.04)
            # lit fraction: the sun disc is 'soft' wide in elevation
            q = (tanel - best) / soft
            out[iz, ix] = min(max(0.5 + q, 0.0), 1.0)
    return out


def build_all(seed=5):
    P, N = 192.0, 2048
    Pd, Nd = 24.0, 2048
    deck = build_deck(seed, P, N)
    det = build_detail(seed + 3, Pd, Nd)
    return deck, det, P, Pd


def build_maps(deck, det, P, Pd, sun_az, sun_el, levels=6, dlevels=6):
    lx, lz = math.sin(sun_az), math.cos(sun_az)
    tanel = math.tan(sun_el)
    Dp = pyramid(deck, levels)
    Dd = pyramid(det, dlevels)
    # shadow map at half res on a slightly smoothed deck
    sh = horizon_shadow(wrap_blur(deck, 1.0), P, lx, lz, tanel, 40.0, 0.09)
    sh = wrap_blur(sh, 2.0)
    Sp = pyramid(sh, 5)
    # cavity AO for the deck (difference from a wider blur) - pre-blurred per level
    e = P / deck.shape[0]
    cav = ((deck - wrap_blur(deck, 0.35 / e)) * 0.9 + (deck - wrap_blur(deck, 1.3 / e)) * 0.45 +
           (deck - wrap_blur(deck, 5.0 / e)) * 0.2)
    Cp = pyramid(cav, 5)
    dcav = det - wrap_blur(det, 0.12 / (Pd / det.shape[0]))
    Dc = pyramid(dcav, dlevels)
    return Dp, Dd, Sp, Cp, Dc


# ============================================================================ render

@njit(cache=True, fastmath=True)
def _mip(M, P, x, z, lv):
    L = M.shape[0]
    if lv <= 0.0:
        return _sample(M[0], P, x, z)
    if lv >= L - 1:
        return _sample(M[L - 1], P, x, z)
    i = int(lv)
    t = lv - i
    return _sample(M[i], P, x, z) * (1 - t) + _sample(M[i + 1], P, x, z) * t


@njit(cache=True, fastmath=True)
def _height(Dp, Dd, P, Pd, x, z, lb, ld, damp, dox, doz):
    return _mip(Dp, P, x, z, lb) + damp * _mip(Dd, Pd, x + dox, z + doz, ld)


@njit(cache=True, fastmath=True)
def _ramp(v, rs, cs, out):
    n = rs.shape[0]
    if v <= rs[0]:
        for k in range(3):
            out[k] = cs[0, k]
        return
    for i in range(n - 1):
        if v <= rs[i + 1]:
            t = (v - rs[i]) / (rs[i + 1] - rs[i])
            t = t * t * (3 - 2 * t) * 0.5 + t * 0.5
            for k in range(3):
                out[k] = cs[i, k] * (1 - t) + cs[i + 1, k] * t
            return
    for k in range(3):
        out[k] = cs[n - 1, k]


@njit(cache=True, fastmath=True, parallel=True)
def render(Ws, Hs, f, hy, cx, camx, camy, camz, Dp, Dd, Sp, Cp, Dc, Bn, P, Pd, damp, dox, doz,
           Lx, Ly, Lz, wrap, rs, cs, hzcol, fogz, sunx, suny, scat_col, texel, dtexel, zfar, light):
    """Returns rgb (Hs, Ws, 3), depth (Hs, Ws) (0 = sky), key light (Hs, Ws)."""
    rgb = np.zeros((Hs, Ws, 3), np.float32)
    dep = np.zeros((Hs, Ws), np.float32)
    keyo = np.zeros((Hs, Ws), np.float32)
    hmax = 3.9 + damp
    ln2 = math.log(2.0)
    # view direction toward the sun (for forward scatter)
    sdx = (sunx - cx) / f
    sdy = (hy - suny) / f
    sn = math.sqrt(sdx * sdx + sdy * sdy + 1.0)
    sdx /= sn
    sdy /= sn
    sdz = 1.0 / sn
    for c in prange(Ws):
        dx = (c + 0.5 - cx) / f
        z = 0.4
        zprev = 0.4
        col = np.zeros(3, np.float32)
        done = False
        for r in range(Hs - 1, -1, -1):
            if done:
                break
            slope = (hy - (r + 0.5)) / f
            hit = False
            while True:
                fp = z / f
                ry = camy + slope * z
                if ry > hmax:
                    if slope >= 0.0:
                        break
                wx = camx + dx * z
                wz = camz + z
                lb = math.log(max(fp * 1.6 / texel, 1.0)) / ln2
                ld = math.log(max(fp * 1.6 / dtexel, 1.0)) / ln2
                if ry > hmax:
                    h = -100.0
                else:
                    h = _mip(Dp, P, wx, wz, lb)
                    if ry <= h + damp:
                        h += damp * _mip(Dd, Pd, wx + dox, wz + doz, ld)
                if ry <= h:
                    hit = True
                    break
                zprev = z
                z += max(0.012, z * 0.005)
                if z > zfar:
                    break
            if not hit:
                done = True
                break
            # refine between zprev (above) and z (below)
            a = zprev
            b = z
            for it in range(8):
                m = 0.5 * (a + b)
                fp = m / f
                lb = math.log(max(fp * 1.6 / texel, 1.0)) / ln2
                ld = math.log(max(fp * 1.6 / dtexel, 1.0)) / ln2
                h = _height(Dp, Dd, P, Pd, camx + dx * m, camz + m, lb, ld, damp, dox, doz)
                if camy + slope * m <= h:
                    b = m
                else:
                    a = m
            zh = b
            z = b
            zprev = b
            # ---------------- shade
            fp = zh / f
            lb = math.log(max(fp * 1.6 / texel, 1.0)) / ln2
            ld = math.log(max(fp * 1.6 / dtexel, 1.0)) / ln2
            wx = camx + dx * zh
            wz = camz + zh
            hh = _height(Dp, Dd, P, Pd, wx, wz, lb, ld, damp, dox, doz)
            eps = max(fp * 1.5, texel * 0.8)
            if lb < 1.5:
                bgx = (_mipc(Dp, P, wx + eps, wz, lb) - _mipc(Dp, P, wx - eps, wz, lb)) / (2 * eps)
                bgz = (_mipc(Dp, P, wx, wz + eps, lb) - _mipc(Dp, P, wx, wz - eps, lb)) / (2 * eps)
            else:
                bgx = (_mip(Dp, P, wx + eps, wz, lb) - _mip(Dp, P, wx - eps, wz, lb)) / (2 * eps)
                bgz = (_mip(Dp, P, wx, wz + eps, lb) - _mip(Dp, P, wx, wz - eps, lb)) / (2 * eps)
            eps = max(fp * 1.5, dtexel * 1.5)
            dgx = (_mip(Dd, Pd, wx + dox + eps, wz + doz, ld) - _mip(Dd, Pd, wx + dox - eps, wz + doz, ld)) / (2 * eps)
            dgz = (_mip(Dd, Pd, wx + dox, wz + doz + eps, ld) - _mip(Dd, Pd, wx + dox, wz + doz - eps, ld)) / (2 * eps)
            wdet = damp / (1.0 + (bgx * bgx + bgz * bgz) * 8.0)
            gx = bgx + wdet * dgx
            gz = bgz + wdet * dgz
            nx, ny, nz = -gx, 1.0, -gz
            nl = math.sqrt(nx * nx + ny * ny + nz * nz)
            nx /= nl
            ny /= nl
            nz /= nl
            ndl = nx * Lx + ny * Ly + nz * Lz
            dif = (ndl + wrap) / (1.0 + wrap)
            dif = min(max(dif, 0.0), 1.0)
            dif = dif * dif * (3 - 2 * dif)
            # big-form light (clean painted shapes) + a restrained share of the cauliflower detail
            bnl = math.sqrt(bgx * bgx + 1.0 + bgz * bgz)
            ndlb = (-bgx * Lx + Ly - bgz * Lz) / bnl
            difb = min(max((ndlb + wrap) / (1.0 + wrap), 0.0), 1.0)
            difb = difb * difb * (3 - 2 * difb)
            dif = difb + DETK * (dif - difb)
            dif = min(max(dif, 0.0), 1.0)
            # heightfield walls facing the camera: attributes would be constant down the wall ->
            # give them a sphere-like vertical gradient (lower on the wall = turned down, into shade)
            sl2 = bgx * bgx + bgz * bgz
            if sl2 > 0.3:
                vn2 = math.sqrt(dx * dx + 1.0)
                vdx = dx / vn2
                vdz = 1.0 / vn2
                htop = hh
                for kk in range(3):
                    dd = 0.3 * (kk + 1) * (kk + 1) * 0.5
                    hk = _mip(Dp, P, wx + vdx * dd, wz + vdz * dd, lb)
                    if hk > htop:
                        htop = hk
                u = min((htop - hh) / 1.5, 1.0) * (sl2 / (1.0 + sl2))
                dif = dif * (1.0 - 0.6 * u)
            ls = max(lb, 0.0)
            if ls < 1.0:
                sh = _mipc(Sp, P, wx, wz, ls)
            else:
                sh = _mip(Sp, P, wx, wz, ls)
            cav = _mip(Cp, P, wx, wz, lb) * 0.9 + damp * _mip(Dc, Pd, wx + dox, wz + doz, ld) * DCAV
            ao = min(max(0.72 + cav * 0.9, 0.25), 1.15)
            key = dif * (0.25 + 0.75 * sh)
            expo = min(max((hh + 1.2) / 4.0, 0.0), 1.0)
            # brushed break-up along the terminator (world-attached)
            bn = _sample(Bn, P, wx * 1.0, wz * 1.0) - 0.5
            term = 4.0 * key * (1.0 - key)
            v = key ** 0.85 * (0.8 + 0.2 * ao) + 0.08 * expo + bn * 0.1 * term + 0.08 * light * key
            v = v - (1.0 - min(ao, 1.0)) * 0.3 + max(cav, 0.0) * 0.25 * key
            _ramp(v, rs, cs, col)
            # sky fill on shadow sides that face up (cool lavender)
            skyk = (1.0 - key) * max(ny, 0.0) * 0.06
            col[0] = col[0] * (1 - skyk) + 0.55 * skyk
            col[1] = col[1] * (1 - skyk) + 0.55 * skyk
            col[2] = col[2] * (1 - skyk) + 0.95 * skyk
            # aerial perspective: distant rows go paler / warmer / hazier toward the horizon glow
            fz = 1.0 - math.exp(-(zh / fogz) ** 1.15)
            vx, vy, vz = dx, slope, 1.0
            vn = math.sqrt(vx * vx + vy * vy + 1.0)
            cosg = (vx * sdx + vy * sdy + vz * sdz) / vn
            sc = max(cosg, 0.0) ** 24
            fwd = sc * (0.35 + 0.65 * fz) * (0.8 + 0.6 * light)
            for k in range(3):
                cc = col[k] * (1.0 - fz) + hzcol[c, k] * fz
                cc = cc + scat_col[k] * fwd * (0.4 + 0.6 * key)
                rgb[r, c, k] = cc
            dep[r, c] = zh
            keyo[r, c] = key
    return rgb, dep, keyo


@njit(cache=True, fastmath=True, parallel=True)
def rims(rgb, dep, key, f, hy, width_w, sunx, W0, rim_col, glow_col, amt, gamt, jump):
    """In place: backlit rims.  Walk each column top->bottom; silhouette-top strength e is continuous
    (depth ratio to the pixel above, sky = 1).  Rim r = max(e, r_prev * decay) propagates inward over
    ~width px (world width_w at that depth); a broader warm translucency glow does the same."""
    Hs, Ws = dep.shape
    for c in prange(Ws):
        rr = 0.0
        gg = 0.0
        bl = math.exp(-((c - sunx) / (0.42 * W0)) ** 2)
        zp = 0.0
        for r in range(Hs):
            z = dep[r, c]
            if z <= 0.0:
                rr = 0.0
                gg = 0.0
                zp = 0.0
                continue
            if zp <= 0.0:
                e = 1.0
            else:
                q = zp / z
                s0 = (r + 0.5 - hy)          # this row (below horizon > 0)
                s1 = (r - 0.5 - hy)          # row above
                qp = s0 / s1 if s1 > 0.5 else 50.0     # depth ratio a flat deck would give
                qn = q / max(qp, 1.0)
                e = min(max((qn - 1.1) / (jump - 1.1), 0.0), 1.0)
                if q < 0.92:        # stepped onto something farther: rim no longer applies
                    rr = 0.0
                    gg = 0.0
            wpx = min(max(width_w * f / z, 1.0), 14.0)
            rr = max(e, rr * math.exp(-1.0 / wpx))
            gg = max(e, gg * math.exp(-1.0 / (wpx * 3.0)))
            zp = z
            if rr > 0.003 or gg > 0.003:
                fz = math.exp(-z / 400.0)
                a = amt * rr * (0.35 + 0.65 * bl) * (0.6 + 0.4 * fz)
                ga = gamt * gg * gg * (0.3 + 0.7 * bl)
                for q3 in range(3):
                    v = rgb[r, c, q3]
                    v = v + (glow_col[q3] - v) * min(ga, 1.0)
                    v = v + (rim_col[q3] - v) * min(a, 1.0)
                    rgb[r, c, q3] = v
