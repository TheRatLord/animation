"""s06_seaside hillside canopy, round 5: ONE coherent backlit leaf mass instead of a heap of
independently-lit clumps (which read as camouflage).

The canopy is a heap of spheres at three scales (big cauliflower lobes -> medium leaf clusters on
their surface -> small leaf clusters on the silhouette and scattered inside). A numba kernel keeps
the front-most sphere per pixel. Light is computed at LOBE level only: a global form normal from the
blurred silhouette (the whole tree is one mass: broad cool shadow inside, the sun side lit) mixed
with the big-lobe normal (each lobe turns its own crown to the sun). Big-lobe pixels quantise that
light per pixel; every leaf cluster takes ONE flat value from the lobe light at its centre (+ jitter),
so neighbouring clusters of equal value merge into broad masses whose borders are leaf-scalloped -
no per-cluster sphere shading, no dimples. Values: deep / shade / sky-lit shade (x2) / warm half-tone
/ lit / hot. The sun sits low to the right and slightly behind, so only right-facing crowns and
silhouettes catch warm light; a thin broken HDR rim runs along the sun-side silhouette; the down side
loses its edge into the dark hillside (soft alpha + deep value). Deterministic (seeded rng)."""
import math
import numpy as np
import cv2
from numba import njit

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


PAL_DARK = dict(deep=cc('#070d16'), shd=cc('#0d1726'), sky=cc('#18283a'), sky2=cc('#22344a'),
                half=cc('#433340'), lit=cc('#8e5634'), hot=cc('#e39052'),
                rim=np.array([1.5, 0.85, 0.45], np.float32), haze=cc('#4a4468'))
PAL = dict(deep=cc('#0c1824'), shd=cc('#16293a'), sky=cc('#27405a'), sky2=cc('#3a5270'),
           half=cc('#6a4a4e'), lit=cc('#c47a44'), hot=cc('#ffb86a'),
           rim=np.array([1.75, 1.05, 0.52], np.float32), haze=cc('#6a5a8a'))


@njit(cache=True)
def _heap(h, w, x0, y0, ss, cx, cy, cz, cr, kind, wob, wk, wph):
    """front-most sphere per (supersampled) pixel -> height, normal x/y, winner kind, winner index."""
    H = np.full((h, w), -1e9, np.float32)
    NX = np.zeros((h, w), np.float32)
    NY = np.zeros((h, w), np.float32)
    K = np.full((h, w), -1, np.int32)
    I = np.full((h, w), -1, np.int32)
    for i in range(cx.shape[0]):
        R = cr[i] * ss
        px = (cx[i] - x0) * ss
        py = (cy[i] - y0) * ss
        ja = max(int(px - R) - 1, 0)
        jb = min(int(px + R) + 2, w)
        ia = max(int(py - R) - 1, 0)
        ib = min(int(py + R) + 2, h)
        R2 = R * R
        zz = cz[i] * ss
        a = wob[i]
        k = wk[i]
        ph = wph[i]
        for y in range(ia, ib):
            dy = y + 0.5 - py
            for x in range(ja, jb):
                dx = x + 0.5 - px
                d2 = dx * dx + dy * dy
                if a > 0.0 and d2 > 0.0:
                    th = math.atan2(dy, dx)
                    s_ = 1.0 - a + a * math.sin(k * th + ph) + 0.4 * a * math.sin((2 * k + 1) * th + 2.0 * ph)
                    lim = R2 * s_ * s_
                else:
                    lim = R2
                if d2 < lim:
                    v = zz + math.sqrt(max(R2 - d2, 0.0))
                    if v > H[y, x]:
                        H[y, x] = v
                        NX[y, x] = dx / R
                        NY[y, x] = dy / R
                        K[y, x] = kind[i]
                        I[y, x] = i
    return H, NX, NY, K, I


def _norm3(x, y, z):
    n = np.sqrt(x * x + y * y + z * z) + 1e-6
    return x / n, y / n, z / n


def _quant(lam, sky, gny):
    """light term -> palette index (deep, shd, sky, sky2, half, lit, hot)."""
    idx = np.full(lam.shape, 1, np.int32)
    idx[sky < -0.12] = 0
    idx[sky > 0.38] = 2
    idx[(sky > 0.62) & (gny < -0.15)] = 3
    idx[lam > 0.16] = 4
    idx[lam > 0.3] = 5
    idx[lam > 0.5] = 6
    return idx


def canopy(cv, lobes, rng, unit=1.0, clip=None, L=(0.85, -0.2, -0.28), pal=PAL, ss=2, far_haze=0.55,
           form=None, bias=0.0):
    """Paint the hillside canopy into Canvas cv.
    lobes: list of (x, y, r, f, clipped) canvas px; f = 0 near .. 1 far (aerial perspective, detail
    scale); clipped lobes only cover pixels inside clip = (mask, x0, y0)."""
    u = max(unit, 0.3)
    lob = np.array([l_[:4] for l_ in lobes], np.float64)
    pad = 12 * u
    X0 = int(max(math.floor((lob[:, 0] - lob[:, 2]).min() - pad), 0))
    Y0 = int(max(math.floor((lob[:, 1] - lob[:, 2]).min() - pad), 0))
    X1 = int(min(math.ceil((lob[:, 0] + lob[:, 2]).max() + pad), cv.W))
    Y1 = int(min(math.ceil((lob[:, 1] + lob[:, 2]).max() + pad), cv.H))
    w, h = X1 - X0, Y1 - Y0
    if w < 3 or h < 3:
        return None
    hs, wsz = h * ss, w * ss
    sx, sy, sz, sr, sk, sc_, spx, spy = [], [], [], [], [], [], [], []

    def add(x, y, z, r, k, c=0, smp=None):
        sx.append(x); sy.append(y); sz.append(z); sr.append(r); sk.append(k); sc_.append(c)
        if smp is None:
            smp = ((x - X0) * ss, (y - Y0) * ss)
        spx.append(smp[0]); spy.append(smp[1])

    def arr(v, t=np.float32):
        return np.array(v, t)

    # ---- level 1: big lobes, each a cauliflower (core + crown of sub-lobes on its upper / sun half)
    for (x, y, r, f, c) in lobes:
        z0 = 0.55 * y
        add(x, y, z0, r * 0.7, 0, c)
        nsub = int(rng.integers(5, 9))
        for j in range(nsub):
            a = rng.uniform(-math.pi * 1.1, math.pi * 0.2)
            d = r * rng.uniform(0.35, 0.55)
            rr = r * rng.uniform(0.26, 0.42)
            add(x + math.cos(a) * d, y + math.sin(a) * d * 0.85, z0 - rr * 0.3, rr, 0, c)
    big = len(sx)
    bx, by, bz, br = arr(sx), arr(sy), arr(sz), arr(sr)
    bclip = arr(sc_, np.int32)
    Hb, NXb, NYb, Kb, Ib = _heap(hs, wsz, np.float32(X0), np.float32(Y0), float(ss), bx, by, bz, br,
                                 np.zeros(big, np.int32), np.zeros(big, np.float32), np.zeros(big, np.float32),
                                 np.zeros(big, np.float32))
    inside = Ib >= 0
    if clip is not None:
        cm, cx0, cy0 = clip
        cl_ = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            cl_[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
        cl_s = cv2.resize(cl_, (wsz, hs), interpolation=cv2.INTER_LINEAR) > 0.5
        # unclipped (skyline) lobes keep their full disc even behind a clipped one
        uc = bclip == 0
        z0_ = np.zeros(int(uc.sum()), np.float32)
        _, _, _, _, Iu = _heap(hs, wsz, np.float32(X0), np.float32(Y0), float(ss), bx[uc], by[uc], bz[uc], br[uc],
                               z0_.astype(np.int32), z0_, z0_, z0_)
        inside &= (Iu >= 0) | cl_s
    # smooth per-pixel aerial-perspective / detail-scale field from the lobes' f
    fmap = np.zeros((h, w), np.float32)
    wmap = np.zeros((h, w), np.float32)
    for (x, y, r, f, c) in lobes:
        xi, yi = int(x - X0), int(y - Y0)
        if 0 <= xi < w and 0 <= yi < h:
            fmap[yi, xi] += f
            wmap[yi, xi] += 1
    sg = max(25 * u, 4)
    bw_ = cv2.GaussianBlur(wmap, (0, 0), sg)
    fmap = np.where(bw_ > 1e-7, cv2.GaussianBlur(fmap, (0, 0), sg) / np.maximum(bw_, 1e-9), 0.5).astype(np.float32)
    fm = cv2.resize(fmap, (wsz, hs), interpolation=cv2.INTER_LINEAR)

    # ---- lobe-level light: global form normal (whole tree = one mass) + big-lobe normal
    if inside.sum() < 4:
        return None
    covb = inside.astype(np.float32)
    fs = (50 * u if form is None else form) * ss
    pb = int(2 * fs) + 1
    blur = cv2.GaussianBlur(cv2.copyMakeBorder(covb, pb, pb, pb, pb, cv2.BORDER_CONSTANT, value=0), (0, 0),
                            fs)[pb:-pb, pb:-pb]
    gy, gx = np.gradient(blur)
    gs = 1.0 / (np.percentile(np.hypot(gx, gy)[inside], 90) + 1e-6)
    gnx, gny, gnz = _norm3(-gx * gs, -gy * gs, np.full_like(gx, 0.9))
    lzb = np.sqrt(np.clip(1 - NXb * NXb - NYb * NYb, 0, 1))
    nx, ny, nz = _norm3(0.3 * gnx + 0.7 * NXb, 0.3 * gny + 0.7 * NYb, 0.3 * gnz + 0.7 * lzb)
    Lx, Ly, Lz = _norm3(*L)
    gl = gnx * Lx + gny * Ly
    nzf = C.fbm(wsz, hs, max(int(wsz / (30 * u * ss)), 2), 3, seed=int(rng.integers(0, 1 << 30)))
    lamL = nx * Lx + ny * Ly + nz * Lz + 0.3 * gl - 0.1 + bias + (nzf - 0.5) * 0.1
    skyL = nx * -0.25 + ny * -0.85 + nz * 0.45 + (nzf - 0.5) * 0.12

    # ---- level 2: medium leaf clusters sitting on the lobe surface
    ys_i, xs_i = np.nonzero(inside)
    npx = len(xs_i) / (ss * ss)
    nmed = int(np.clip(npx / (230.0 * u * u), 200, 6000))
    for p in rng.integers(0, len(xs_i), nmed):
        yy, xx = ys_i[p], xs_i[p]
        f = fm[yy, xx]
        R = (6.0 + 15.0 * rng.uniform(0, 1) ** 1.6) * u * (1 - 0.6 * f)
        add(xx / ss + X0, yy / ss + Y0, Hb[yy, xx] / ss - R * 0.5, R, 1)
    # ---- level 3: small leaf clusters: dense on the silhouette (cauliflower edge), sparse inside
    ins8 = inside.astype(np.uint8)
    eb = cv2.GaussianBlur(ins8.astype(np.float32), (0, 0), 3 * u * ss)
    egy, egx = np.gradient(eb)
    en = np.hypot(egx, egy) + 1e-6
    onx, ony = -egx / en, -egy / en
    edge = ins8 - cv2.erode(ins8, np.ones((3, 3), np.uint8))
    ey, ex = np.nonzero(edge)
    if len(ex):
        ne = int(np.clip(len(ex) / (ss * 3.0 * u), 100, 12000))
        for p in rng.integers(0, len(ex), ne):
            yy, xx = ey[p], ex[p]
            f = fm[yy, xx]
            R = (3.0 + 6.0 * rng.uniform(0, 1) ** 1.4) * u * (1 - 0.55 * f)
            ox_, oy_ = rng.normal(0, 0.5 * R, 2)
            # pushed out along the silhouette normal (leaf clusters break the outline)
            o = R * rng.uniform(0.0, 0.8) * (1.0 if onx[yy, xx] * Lx + ony[yy, xx] * Ly > -0.3 else 0.3)
            ox_ += onx[yy, xx] * o
            oy_ += ony[yy, xx] * o
            add(xx / ss + ox_ + X0, yy / ss + oy_ + Y0, Hb[yy, xx] / ss - R * 0.2, R, 2, smp=(xx, yy))
    nin = int(np.clip(npx / (70.0 * u * u), 200, 20000))
    for p in rng.integers(0, len(xs_i), nin):
        yy, xx = ys_i[p], xs_i[p]
        f = fm[yy, xx]
        R = (2.5 + 5.5 * rng.uniform(0, 1) ** 1.5) * u * (1 - 0.55 * f)
        add(xx / ss + X0, yy / ss + Y0, Hb[yy, xx] / ss - R * 0.3, R, 2)
    allx, ally, allz, allr = arr(sx), arr(sy), arr(sz), arr(sr)
    allk = arr(sk, np.int32)
    # leaf clusters get an irregular lobed outline (a few lumps), not a perfect disc
    nall = len(allx)
    wob = np.where(allk > 0, rng.uniform(0.05, 0.14, nall), 0.0).astype(np.float32)
    wk = rng.integers(3, 6, nall).astype(np.float32)
    wph = rng.uniform(0, 2 * math.pi, nall).astype(np.float32)
    Hh, NX, NY, K, I = _heap(hs, wsz, np.float32(X0), np.float32(Y0), float(ss), allx, ally, allz, allr, allk,
                             wob, wk, wph)
    kd = max(int(9 * u * ss), 1)
    ins_d = cv2.dilate(ins8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kd + 1, 2 * kd + 1))) > 0
    cov = ((I >= 0) & ins_d).astype(np.float32)
    Ic = np.maximum(I, 0)

    # ---- values: big-lobe pixels quantised per pixel; every leaf cluster takes ONE flat value
    cxs = np.clip(np.array(spx).astype(np.int32), 0, wsz - 1)
    cys = np.clip(np.array(spy).astype(np.int32), 0, hs - 1)
    jl = rng.normal(0, 0.04, len(allx)).astype(np.float32)
    js = rng.normal(0, 0.07, len(allx)).astype(np.float32)
    lam_c = lamL[cys, cxs] + jl
    sky_c = skyL[cys, cxs] + js
    idx_c = _quant(lam_c, sky_c, gny[cys, cxs])
    idx_px = _quant(lamL, skyL, gny)
    idx = np.where(K == 0, idx_px, idx_c[Ic])
    # strongly lit leaf clusters: a hot crescent on their sun side (only the brightest)
    lzc = np.sqrt(np.clip(1 - NX * NX - NY * NY, 0, 1))
    hotc = (K > 0) & (idx >= 5) & ((NX * Lx + NY * Ly + lzc * Lz) > 0.45)
    idx[hotc] = 6
    # shadow clusters: now and then a lighter cool cap on the sky-facing top (leaf detail)
    cap = (K > 0) & (idx == 1) & (NY < -0.45) & (lam_c[Ic] > -0.25) & (js[Ic] > 0.05)
    idx[cap] = 2
    names = ['deep', 'shd', 'sky', 'sky2', 'half', 'lit', 'hot']
    tab = np.stack([np.asarray(pal[n], np.float32) for n in names], 0)
    var = (1 + rng.normal(0, 0.035, len(allx))).astype(np.float32)
    rgb = tab[idx] * var[Ic][..., None]
    # rim: a thin broken warm line along the sun-facing silhouette
    Lsx, Lsy = _norm3(Lx, Ly, 0.0)[:2]
    rw = 1.6 * u * ss
    sh = np.float32([[1, 0, -Lsx * rw], [0, 1, -Lsy * rw]])
    covs = cv2.warpAffine(cov, sh, (wsz, hs), flags=cv2.INTER_LINEAR)
    rim = np.clip(cov - covs, 0, 1)
    rn = C.fbm(wsz, hs, max(int(wsz / (14 * u * ss)), 2), 2, seed=int(rng.integers(0, 1 << 30)))
    rim = rim * C.smoothstep(0.36, 0.52, rn) * C.smoothstep(-0.15, 0.3, gl) * (1 - 0.7 * fm)
    rimc = np.asarray(pal['rim'], np.float32)
    rgb = rgb + (rimc - rgb) * rim[..., None]
    # aerial perspective: far lobes toward the violet haze, lower contrast
    hz = np.asarray(pal['haze'], np.float32)
    rgb = rgb + (hz - rgb) * (far_haze * fm ** 1.4)[..., None]
    # ---- down side: lose the edge into the hillside (soft alpha where the global form faces down)
    down = (C.smoothstep(0.1, 0.6, gny) * (gny > 0)).astype(np.float32)
    soft = cv2.GaussianBlur(cov, (0, 0), 4 * u * ss)
    a = cov * (1 - down) + np.minimum(cov, soft) * down
    rgb = rgb + (np.asarray(pal['deep'], np.float32) - rgb) * (0.55 * down * (idx < 4))[..., None]
    prem = np.concatenate([rgb * a[..., None], a[..., None]], -1).astype(np.float32)
    small = cv2.resize(prem, (w, h), interpolation=cv2.INTER_AREA)
    al = small[..., 3]
    col = small[..., :3] / np.maximum(al, 1e-5)[..., None]
    cv.put((al, X0, Y0), col)
    return (al, X0, Y0)
