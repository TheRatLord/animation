"""Painted cumulonimbus for shot s01 (own renderer - lib/ untouched).

Model: a hierarchy of 3D domes (big billows -> lobes -> silhouette bumps) rasterised with a supersampled
z-buffer gives a crisp scalloped cauliflower silhouette. Lighting is designed like a painting, not a render:
* a smooth BIG-FORM light (root billow + whole-cloud pillow normal) decides where light and shadow are;
* the small-lobe light may only deviate a little from it, so lobes show up only as scallops along the
  terminator (never as a pile of shaded beads);
* ONE crisp threshold splits light / shadow; lit areas are flat warm white with a broad hot crown, shadow
  areas are flat lavender-blue with soft sky-fill / bounce gradients driven by the big form only;
* silver lining, translucent glow and halo on sun-facing edges near the (hidden) sun.
"""
import math
import numpy as np
import cv2
from numba import njit, prange


class Domes:
    def __init__(self):
        self.rows = []
        self.tid = {}        # level-0 dome index -> tower id

    def add(self, cx, cy, cz, r, ay, level, parent=-1):
        i = len(self.rows)
        root = i if parent < 0 else int(self.rows[parent][7])
        self.rows.append([cx, cy, cz, r, ay, level, parent, root])
        return i

    def arr(self):
        return np.array(self.rows, np.float64)


def grow(D, rng, parents, count, ratio, up_bias=0.9, front=(-0.1, 0.98), minr=1.0, clip_y=1e9,
         reach_k=(0.0, 0.25), max_uy=1.0, max_uz=1.0, rvar=(0.7, 1.25)):
    """Cover the visible surface of each parent dome with child lobes (blue-noise directions over the
    front hemisphere, biased toward the crown); children sit on the parent surface and protrude."""
    kids = []
    for pi in parents:
        px, py, pz, pr, pay = D.rows[pi][:5]
        lev = D.rows[pi][5]
        rr = pr * ratio
        if rr < minr:
            continue
        n = int(count + rng.random())
        m = n * 12
        dz = rng.uniform(front[0], front[1], m)
        a = -math.pi / 2 + rng.normal(0, up_bias, m)
        s_ = np.sqrt(np.clip(1 - dz * dz, 0, 1))
        V = np.stack([np.cos(a) * s_, np.sin(a) * s_, dz], 1)
        V = V[(V[:, 1] <= max_uy) & (np.abs(V[:, 2]) <= max_uz)]
        chosen = []
        thr = math.cos(min(1.25 * ratio + 0.05, 1.2))
        for k in range(len(V)):
            if len(chosen) >= n:
                break
            if all(float(V[k] @ V[c]) < thr for c in chosen):
                chosen.append(k)
        for k in chosen:
            ux, uy, uz = V[k]
            r_ = rr * rng.uniform(*rvar)
            reach = pr - r_ * rng.uniform(*reach_k)
            x = px + ux * reach
            y = py + uy * reach * pay
            z = pz + uz * reach
            if y + r_ * 0.3 > clip_y:
                continue
            ay = min(max(pay * rng.uniform(0.94, 1.04), 0.3), 1.0)
            kids.append(D.add(x, y, z, r_, ay, lev + 1, pi))
    return kids


@njit(cache=True, fastmath=True)
def _zbuf(Hs, Ws, S, clip):
    h1 = np.full((Hs, Ws), -1e12, np.float32)
    i1 = np.full((Hs, Ws), -1, np.int32)
    n = S.shape[0]
    for i in range(n):
        cx, cy, cz, r, ay = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4]
        ry = r * ay
        x0 = max(0, int(cx - r - 1))
        x1 = min(Ws, int(cx + r + 2))
        y0 = max(0, int(cy - ry - 1))
        y1 = min(Hs, int(min(cy + ry, clip) + 2))
        for y in range(y0, y1):
            yy = y + 0.5
            if yy > clip:
                continue
            dy = (yy - cy) / ry
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                d2 = dx * dx + dy * dy
                if d2 >= 1.0:
                    continue
                h = cz + r * math.sqrt(1.0 - d2)
                if h > h1[y, x]:
                    h1[y, x] = h
                    i1[y, x] = i
    return h1, i1


@njit(cache=True, fastmath=True, inline='always')
def _dn(S, j, x, y):
    cx, cy, r, ay = S[j, 0], S[j, 1], S[j, 3], S[j, 4]
    dx = (x - cx) / r
    dy = (y - cy) / (r * ay)
    d2 = dx * dx + dy * dy
    if d2 > 1.0:
        l = math.sqrt(d2)
        dx /= l
        dy /= l
        d2 = 1.0
    nz = math.sqrt(max(1.0 - d2, 0.0))
    nx, ny = dx, dy / ay
    l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    return nx / l, ny / l, nz / l


@njit(cache=True, fastmath=True, parallel=True)
def _shade(H, W, ss, i1, h1, S, FN, sunx, suny, lz_near, lz_far, sun_rad, ws, wb, out):
    """out (H, W, 6): e_small, e_big, ny_big, coverage, nz_big, crease (averaged over ss^2 subsamples)."""
    inv = 1.0 / (ss * ss)
    for Y in prange(H):
        for X in range(W):
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            a4 = 0.0
            a5 = 0.0
            cov = 0.0
            fnx = FN[Y, X, 0]
            fny = FN[Y, X, 1]
            fnz = FN[Y, X, 2]
            for sy in range(ss):
                for sx in range(ss):
                    y = Y * ss + sy
                    x = X * ss + sx
                    i = i1[y, x]
                    if i < 0:
                        continue
                    px = x + 0.5
                    py = y + 0.5
                    n0x, n0y, n0z = _dn(S, i, px, py)
                    p = int(S[i, 6])
                    if p >= 0:
                        n1x, n1y, n1z = _dn(S, p, px, py)
                    else:
                        n1x, n1y, n1z = n0x, n0y, n0z
                    rt = int(S[i, 7])
                    n2x, n2y, n2z = _dn(S, rt, px, py)
                    Nx = ws[0] * n0x + ws[1] * n1x + ws[2] * n2x + ws[3] * fnx
                    Ny = ws[0] * n0y + ws[1] * n1y + ws[2] * n2y + ws[3] * fny
                    Nz = ws[0] * n0z + ws[1] * n1z + ws[2] * n2z + ws[3] * fnz
                    l = math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-6
                    Nx /= l
                    Ny /= l
                    Nz /= l
                    Bx = wb[0] * n1x + wb[1] * n2x + wb[2] * fnx
                    By = wb[0] * n1y + wb[1] * n2y + wb[2] * fny
                    Bz = wb[0] * n1z + wb[1] * n2z + wb[2] * fnz
                    l = math.sqrt(Bx * Bx + By * By + Bz * Bz) + 1e-6
                    Bx /= l
                    By /= l
                    Bz /= l
                    ddx = sunx - px
                    ddy = suny - py
                    dd = math.sqrt(ddx * ddx + ddy * ddy) + 1e-3
                    lz = lz_far + (lz_near - lz_far) * math.exp(-dd / sun_rad)
                    c = math.sqrt(max(1.0 - lz * lz, 0.0))
                    Lx = ddx / dd * c
                    Ly = ddy / dd * c
                    a0 += Nx * Lx + Ny * Ly + Nz * lz
                    a1 += Bx * Lx + By * Ly + Bz * lz
                    a2 += By
                    a4 += Bz
                    cov += 1.0
                    cr = 0.0
                    hh = h1[y, x]
                    for k in range(4):
                        yy = y
                        xx = x
                        if k == 0:
                            yy = y - 2
                        elif k == 1:
                            xx = x - 2
                        elif k == 2:
                            xx = x + 2
                        else:
                            yy = y + 2
                        if yy < 0 or xx < 0 or yy >= H * ss or xx >= W * ss:
                            continue
                        j = i1[yy, xx]
                        if j >= 0 and j != i:
                            dh = (h1[yy, xx] - hh) / (S[i, 3] * 0.25)
                            if dh > cr:
                                cr = dh
                    a5 += min(cr, 1.0)
            if cov > 0:
                out[Y, X, 0] = a0 / cov
                out[Y, X, 1] = a1 / cov
                out[Y, X, 2] = a2 / cov
                out[Y, X, 4] = a4 / cov
                out[Y, X, 5] = a5 / cov
            out[Y, X, 3] = cov * inv


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _fblur(img, sigma, interp=cv2.INTER_LINEAR):
    if sigma <= 0.3:
        return img
    if sigma <= 3.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 2.55)
    return cv2.resize(small, (W, H), interpolation=interp)


GRADMAP = [(-0.75, (0.47, 0.50, 0.80)), (-0.4, (0.54, 0.58, 0.87)), (-0.15, (0.64, 0.68, 0.92)),
           (-0.04, (0.74, 0.76, 0.93)), (0.05, (0.83, 0.84, 0.93)), (0.18, (0.905, 0.905, 0.935)),
           (0.4, (0.965, 0.955, 0.945)), (0.7, (1.01, 0.995, 0.97)), (0.95, (1.05, 1.03, 0.99))]


def organic_warp(rgba, amp, scale, seed=0):
    """Displace a finished cloud plate by a smooth low-frequency field so dome outlines lose their
    perfect-circle CG look (slightly irregular, hand-painted lobes). amp/scale in px."""
    H, W = rgba.shape[:2]
    rng = np.random.default_rng(seed)
    q = max(int(scale / 2), 1)
    w, h = max(W // q, 4), max(H // q, 4)
    d = []
    for k in range(2):
        n = rng.standard_normal((h, w)).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 1.2)
        n = n / (n.std() + 1e-6)
        d.append(cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC) * amp)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(rgba, xs + d[0], ys + d[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


PAL = dict(
    hot=(1.03, 1.01, 0.97),          # broad sunlit crown (HDR -> bloom)
    lit=(0.96, 0.95, 0.935),          # sunlit
    lit_low=(0.95, 0.91, 0.88),     # sunlit but turning away (slightly warm cream)
    band=(0.88, 0.84, 0.95),          # thin lavender terminator
    half=(0.76, 0.79, 0.93),          # soft cool half-tone between lobes in the light
    core=(0.55, 0.59, 0.86),          # core shadow just past the terminator
    shd=(0.66, 0.71, 0.93),           # shadow facing the sky (sky fill)
    bounce=(0.74, 0.73, 0.9),         # shadow facing down (bounce)
    rim=(1.32, 1.27, 1.17),           # silver lining
    haze=(0.8, 0.9, 0.99),            # atmospheric colour near the base
)


def render_cloud(W, H, domes, sun, clip_y, ss=2, lz_near=-0.3, lz_far=0.0, sun_rad=0.3, soft=0.035,
                 pal=PAL, rim=1.0, haze_y=None, vbias=None, clip_lo=0.13, clip_hi=0.1, form_r=0.07,
                 ws=(0.38, 0.3, 0.12, 0.2), wb=(0.2, 0.4, 0.4), crease=0.2, glow=1.0, halo=1.0):
    """Rasterise + paint the dome cloud. W, H plate px, domes in plate px, sun (x, y) plate px.
    vbias = (y_top, y_base, bias_top, bias_base) shifts the terminator with height (the lower tower sits
    in the shadow of the head). Returns straight-alpha RGBA (H, W, 4)."""
    S = domes.arr()
    Ss = S.copy()
    for k in (0, 1, 2, 3):
        Ss[:, k] *= ss
    h1, i1 = _zbuf(H * ss, W * ss, Ss, clip_y * ss)
    alpha0 = cv2.resize((i1 >= 0).astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
    ins = (alpha0 > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(ins, cv2.DIST_L2, 5).astype(np.float32)
    R = form_r * W
    form = (np.sqrt(np.clip(1 - (1 - np.clip(dist / R, 0, 1)) ** 2, 0, 1)) * R).astype(np.float32)
    form = _fblur(form, 0.3 * R)
    gx = cv2.Sobel(form, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(form, cv2.CV_32F, 0, 1, ksize=3) / 8
    FN = np.dstack([-gx, -gy, np.ones_like(gx)])
    FN /= np.linalg.norm(FN, axis=-1, keepdims=True)
    out = np.zeros((H, W, 6), np.float32)
    _shade(H, W, ss, i1, h1, Ss, FN.astype(np.float32), sun[0] * ss, sun[1] * ss, lz_near, lz_far,
           sun_rad * W * ss, np.array(ws, np.float64), np.array(wb, np.float64), out)
    del h1, i1
    e0, e1, ny, alpha, nz, cr = [out[..., k] for k in range(6)]
    P = {k: np.asarray(v, np.float32) for k, v in pal.items()}
    yy = np.arange(H, dtype=np.float32)[:, None]
    vb = np.zeros((H, 1), np.float32)
    if vbias is not None:
        f = np.clip((yy - vbias[0]) / (vbias[1] - vbias[0]), 0, 1)
        vb = vbias[2] + (vbias[3] - vbias[2]) * f
    eb = e1 + vb
    e = np.clip(e0 + vb, eb - clip_lo, eb + clip_hi)
    t_lit = _ss(-soft, soft, e)
    t_hot = _ss(0.18, 0.5, eb) * t_lit
    # shadow: flat lavender, lighter where the big form faces the sky, bounce where it faces down
    up = np.clip(-ny * 1.3 + 0.1, 0, 1)
    down = np.clip(ny * 1.4, 0, 1)
    shd = P['core'] + (P['shd'] - P['core']) * _ss(-0.02, -0.4, eb)[..., None]
    shd = shd + (P['shd'] * 1.03 - shd) * (up * 0.5)[..., None]
    shd = shd + (P['bounce'] - shd) * (down * 0.6)[..., None]
    lit = P['lit_low'] + (P['lit'] - P['lit_low']) * _ss(0.0, 0.2, eb)[..., None]
    col = shd + (lit - shd) * t_lit[..., None]
    col = col + (P['hot'] - col) * (t_hot * 0.9)[..., None]
    band = np.exp(-((e + 0.01) / (soft * 1.2)) ** 2)[..., None]
    col = col + (P['band'] - col) * band * 0.3
    c = np.clip(cr, 0, 1)[..., None]
    col = col + (P['core'] * 1.25 - col) * c * crease * t_lit[..., None]
    col = col * (1 - 0.05 * c * (1 - t_lit[..., None]))
    # ---- silver lining + translucent glow near the sun
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    sx, sy = sun[0] - xs, sun[1] - ys
    sl = np.sqrt(sx * sx + sy * sy) + 1e-3
    ex, ey = sx / sl, sy / sl
    prox = np.exp(-(sl / W) / 0.18) * 0.8 + 0.2
    cg = _fblur(alpha, 0.003 * W)
    ogx = -cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx ** 2 + ogy ** 2) + 1e-6
    oface = (np.clip((ogx * ex + ogy * ey) / ogl, 0, 1) * (ogl > 1e-3)).astype(np.float32)
    oface = _ss(0.05, 0.7, _fblur(oface, 0.002 * W))
    din = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    wr = max(0.0015 * W, 1.0)
    rim_l = np.exp(-din / wr) * oface * prox * rim
    col = col + (P['rim'] - col) * np.clip(rim_l, 0, 1)[..., None]
    thin = np.exp(-din / (0.014 * W)) * oface * np.exp(-(sl / W) / 0.12) * glow
    col = col + (P['rim'] - col) * np.clip(thin, 0, 0.9)[..., None]
    if haze_y is not None:
        f = np.clip((ys - haze_y[0]) / (haze_y[1] - haze_y[0]), 0, 1) ** 1.4
        col = col + (P['haze'] - col) * (f * 0.6)[..., None]
    if halo:
        dout = cv2.distanceTransform((alpha < 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        hal = np.exp(-dout / (0.008 * W)) * _ss(0.05, 0.7, _fblur(oface, 0.006 * W)) * \
            np.exp(-(sl / W) / 0.2) * 0.45 * halo * (1 - alpha)
        a2 = alpha + hal
        col = (col * alpha[..., None] + P['rim'] * hal[..., None]) / np.maximum(a2, 1e-5)[..., None]
        alpha = a2
    return np.dstack([col, np.clip(alpha, 0, 1)]).astype(np.float32)


# =============================================================================== the s01 tower

SUN_U = 0.2          # sun x (fraction of the tower width from x0): hidden behind the tallest turret


def tower_domes(W, H, x0, base_y, top_y, width, seed=1, detail=1.0):
    """The colossal cumulonimbus: a slightly leaning column of stacked billows that swells into a broad,
    asymmetric turreted crown (tallest turret right of centre, hiding the sun), lower sister towers each
    side and wide shoulders at the base."""
    rng = np.random.default_rng(seed)
    D = Domes()
    Hc = base_y - top_y
    lev0 = []

    def prof(v):   # half-width of the main tower (fraction of width) by height v in 0..1
        return np.interp(v, [0, 0.15, 0.35, 0.55, 0.7, 0.8, 0.9],
                         [0.55, 0.45, 0.36, 0.3, 0.3, 0.3, 0.22])

    lean = 0.07 * width
    v = 0.03
    while v < 0.86:
        hw = prof(v) * width
        R = hw * rng.uniform(0.55, 0.7)
        n = max(2, int(round(2 * hw / (R * 1.1))))
        for k in range(n):
            u = -1 + (2 * k + 1) / n + rng.uniform(-0.15, 0.15) / n
            x = x0 + lean * v ** 1.5 + u * max(hw - R * 0.75, 0)
            y = base_y - v * Hc + rng.uniform(-0.08, 0.08) * R
            z = math.sqrt(max(1 - u * u, 0)) * hw * 0.5 + rng.uniform(-0.15, 0.15) * R
            rr = R * rng.uniform(0.85, 1.15)
            if y - rr * 0.95 < top_y + 0.12 * Hc:
                continue
            lev0.append(D.add(x, y, z, rr, 0.92, 0))
            D.tid[lev0[-1]] = 0
        v += R * rng.uniform(0.7, 0.9) / Hc
    # crown: turrets of different heights (u = position across the head, vv = top height, rf = size)
    for (u, vv, rf) in [(-0.75, 0.84, 0.15), (-0.42, 0.91, 0.18), (-0.1, 0.95, 0.19), (0.2, 1.0, 0.18),
                        (0.48, 0.9, 0.15), (0.72, 0.8, 0.13), (0.02, 0.86, 0.26), (0.4, 0.8, 0.2),
                        (-0.5, 0.78, 0.22)]:
        rr = width * rf * 0.6
        x = x0 + lean + u * 0.36 * width
        y = base_y - vv * Hc + rr * 0.9
        lev0.append(D.add(x, y, width * 0.14, rr, 0.92, 0))
        D.tid[lev0[-1]] = 0
    # sister towers and shoulders
    for ti, (cx, top_v, hw, zf) in enumerate([(x0 - 0.5 * width, 0.62, 0.22 * width, 0.25), (x0 + 0.58 * width, 0.5, 0.2 * width, 0.25),
                                (x0 - 0.98 * width, 0.36, 0.2 * width, 0.12), (x0 + 1.0 * width, 0.28, 0.18 * width, 0.12),
                                (x0 - 1.4 * width, 0.18, 0.16 * width, 0.05), (x0 + 1.38 * width, 0.14, 0.15 * width, 0.05)], 1):
        v = 0.03
        while v < top_v:
            f = v / top_v
            R = hw * (1 - 0.35 * f) * rng.uniform(0.9, 1.1) * 0.95
            for k in range(2):
                u = -0.5 + k
                x = cx + u * hw * 0.8 * (1 - 0.4 * f)
                y = base_y - v * Hc
                if y - R * 0.9 < base_y - top_v * Hc:
                    y = base_y - top_v * Hc + R * 0.9
                lev0.append(D.add(x, y, width * zf + rng.uniform(-0.05, 0.05) * R, R, 0.9, 0))
                D.tid[lev0[-1]] = ti
            v += R * 0.8 / Hc
    minr = max(0.0032 * W, 1.0)
    k1 = grow(D, rng, lev0, 11 * detail, 0.45, up_bias=1.2, minr=minr, clip_y=base_y, rvar=(0.6, 1.3))
    # undersides / flanks too, so no raw big spheres show (smaller, all around)
    k1 += grow(D, rng, lev0, 7 * detail, 0.36, up_bias=3.0, front=(0.1, 0.9), minr=minr, clip_y=base_y)
    k2 = grow(D, rng, k1, 9 * detail, 0.36, up_bias=0.9, max_uy=0.3, minr=minr, clip_y=base_y)
    grow(D, rng, k2, 6 * detail, 0.42, up_bias=0.9, front=(-0.2, 0.4), max_uy=0.2, minr=minr, clip_y=base_y)
    return D


# periwinkle / cerulean shadow family (summer noon), warm-white light
GRADMAP2 = [(-0.75, (0.42, 0.51, 0.79)), (-0.4, (0.5, 0.59, 0.85)), (-0.12, (0.6, 0.68, 0.9)),
            (-0.02, (0.68, 0.74, 0.93)), (0.03, (0.83, 0.85, 0.95)), (0.14, (0.885, 0.9, 0.95)),
            (0.34, (0.95, 0.95, 0.95)), (0.6, (1.01, 0.995, 0.965)), (0.95, (1.08, 1.05, 0.98))]


def tower_shading(W, H, top_y, base_y):
    Hc = base_y - top_y
    return dict(light=(0.62, -0.7, 0.3), scales=(0.003, 0.01, 0.03, 0.08, 0.16), sw=(0.2, 0.22, 0.18, 0.08, 0.06),
                vbias=(top_y, base_y, -0.02, -0.8),
                form_w=1.7, shadow=0.5, sh_len=0.03, sh_soft=0.005, sh_blur=0.0014, back_k=0.9, back_r=0.07,
                rim=1.6, gradmap=GRADMAP2, ao_line=0.9, haze_k=0.6,
                haze_y=(top_y + 0.5 * Hc, base_y + 0.02 * H))


def _grad_blur(img, sigma, mask=None):
    """Gradient of a Gaussian-blurred image, computed at reduced resolution and upsampled (smooth, no
    banding). With mask: normalised (masked) blur -> no cliffs at the silhouette."""
    H, W = img.shape
    q = max(1, int(sigma / 3))
    w, h = max(W // q, 4), max(H // q, 4)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA) if q > 1 else img
    small = cv2.GaussianBlur(small, (0, 0), sigma / q)
    if mask is not None:
        sm = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA) if q > 1 else mask
        sm = cv2.GaussianBlur(sm, (0, 0), sigma / q)
        small = small / (sm + 1e-3)
    gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3) / (8 * q)
    gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3) / (8 * q)
    if q > 1:
        gx = cv2.resize(gx, (W, H), interpolation=cv2.INTER_LINEAR)
        gy = cv2.resize(gy, (W, H), interpolation=cv2.INTER_LINEAR)
    return gx, gy


@njit(cache=True, fastmath=True, parallel=True)
def _hf_shadow(h, m, lx, ly, slope, steps, step, soft):
    """Height-field self shadowing: march toward the light (lx, ly in px/step direction); the ray
    rises by `slope` per px. Returns 0 (lit) .. 1 (shadowed) with a soft penumbra of `soft` px height."""
    H, W = h.shape
    out = np.zeros((H, W), np.float32)
    for y in prange(H):
        for x in range(W):
            if m[y, x] < 0.5:
                continue
            h0 = h[y, x]
            occ = 0.0
            for k in range(1, steps + 1):
                d = k * step
                sx = x + lx * d
                sy = y + ly * d
                ix = int(sx)
                iy = int(sy)
                if ix < 0 or iy < 0 or ix >= W - 1 or iy >= H - 1:
                    break
                fx = sx - ix
                fy = sy - iy
                hs = (h[iy, ix] * (1 - fx) * (1 - fy) + h[iy, ix + 1] * fx * (1 - fy) +
                      h[iy + 1, ix] * (1 - fx) * fy + h[iy + 1, ix + 1] * fx * fy)
                o = (hs - (h0 + d * slope)) / soft
                if o > occ:
                    occ = o
                    if occ >= 1.0:
                        break
            out[y, x] = min(occ, 1.0)
    return out


def _mblur(x, m, s):
    """Masked (normalised) blur."""
    return _fblur(x * m, s) / (_fblur(m, s) + 1e-4)


def render_soft(W, H, domes, sun, clip_y, ss=2, lz_near=-0.25, lz_far=0.2, sun_rad=0.3, pal=PAL, rim=1.0,
                haze_y=None, vbias=None, scales=(0.003, 0.01, 0.03, 0.08), sw=(0.25, 0.3, 0.3, 0.15),
                hscale=1.0, ramp=(-0.12, 0.18), ao_k=1.0, glow=1.0, halo=1.0, return_attrs=False, light=None,
                det_ramp=(-0.3, -0.03), det_k=0.6,
                shadow=0.0, sh_len=0.12, sh_soft=0.004, sh_blur=0.0012, gate=None,
                back_k=0.0, back_r=0.12, gradmap=None, ao_line=0.0, haze_k=0.6, form=None, form_w=0.0):
    """Soft painted shading from a multi-scale smoothed height field of the dome z-buffer:
    crisp silhouette, lobes merge into soft rounded forms, crevices darken (AO)."""
    S = domes.arr()
    Ss = S.copy()
    for k in (0, 1, 2, 3):
        Ss[:, k] *= ss
    h1, i1 = _zbuf(H * ss, W * ss, Ss, clip_y * ss)
    if callable(form):
        form = form(i1, ss)
    cov = (i1 >= 0).astype(np.float32)
    hs = np.where(i1 >= 0, h1, 0).astype(np.float32)
    del h1, i1
    alpha = cv2.resize(cov, (W, H), interpolation=cv2.INTER_AREA)
    h = cv2.resize(hs, (W, H), interpolation=cv2.INTER_AREA) / ss
    del hs, cov
    m = (alpha > 0.02).astype(np.float32)
    h = np.where(m > 0, h / np.maximum(alpha, 1e-3), 0).astype(np.float32)
    hmin = float(np.percentile(h[m > 0], 1)) if m.sum() else 0
    hr = np.where(m > 0, h - hmin, 0).astype(np.float32) * hscale
    G = []
    for s in scales:
        if s < 0.02:   # small scales: masked blur (lobe relief only, no silhouette cliff)
            gx, gy = _grad_blur(hr * alpha, s * W, alpha)
            gx, gy = gx * m, gy * m
        else:          # big scales: zero outside -> normals roll outward at silhouettes
            gx, gy = _grad_blur(hr * alpha, s * W)
        g = np.sqrt(gx * gx + gy * gy)
        # normalise slope per scale so every scale contributes shape, not magnitude
        k = 1.0 / (np.percentile(g[m > 0], 85) + 1e-6) if m.sum() else 1.0
        G.append((gx * k, gy * k))

    def normals(wts, nz0=0.9):
        nx = -sum(w * g[0] for w, g in zip(wts, G))
        ny = -sum(w * g[1] for w, g in zip(wts, G))
        if form is not None and form_w:
            nx = nx + form_w * form[0]
            ny = ny + form_w * form[1]
        nz = np.full((H, W), nz0, np.float32)
        nl = np.sqrt(nx * nx + ny * ny + nz * nz)
        return nx / nl, ny / nl, nz / nl
    nx, ny, nz = normals(sw)
    dw = [0.45, 0.4] + [0.15 / max(len(scales) - 2, 1)] * (len(scales) - 2)
    dnx, dny, dnz = normals(dw, 0.8)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    sx, sy = sun[0] - xs, sun[1] - ys
    sl = np.sqrt(sx * sx + sy * sy) + 1e-3
    ex, ey = sx / sl, sy / sl
    lz = lz_far + (lz_near - lz_far) * np.exp(-sl / (sun_rad * W))
    c = np.sqrt(np.clip(1 - lz * lz, 0, 1))
    if light is not None:
        L = np.asarray(light, np.float64)
        L = L / np.linalg.norm(L)
        e = nx * L[0] + ny * L[1] + nz * L[2]
        e_det = dnx * L[0] + dny * L[1] + dnz * L[2]
    else:
        e = nx * ex * c + ny * ey * c + nz * lz
        e_det = dnx * ex * c + dny * ey * c + dnz * lz
    if vbias is not None:
        f = np.clip((ys - vbias[0]) / (vbias[1] - vbias[0]), 0, 1)
        e = e + vbias[2] + (vbias[3] - vbias[2]) * f
    sh = np.zeros((H, W), np.float32)
    if shadow and light is not None:
        L = np.asarray(light, np.float64)
        L = L / np.linalg.norm(L)
        lh = math.hypot(L[0], L[1]) + 1e-6
        slope = L[2] / lh
        hsm = _fblur(hr, 0.0015 * W)
        steps = 128
        stp = sh_len * W / steps
        sh = _hf_shadow(hsm.astype(np.float32), m, L[0] / lh, L[1] / lh, slope, steps, stp, sh_soft * W)
        sh = _fblur(sh, sh_blur * W)
        e = e - shadow * sh
    if back_k:   # near the hidden sun the cloud is seen against the light: darker body, blazing rim
        e = e - back_k * np.exp(-(sl / W) / back_r)
    # ambient occlusion: below the local neighbourhood height
    ao = np.clip((_mblur(hr, m, 0.008 * W) - hr) / (0.02 * W), 0, 1) * 0.5 + \
        np.clip((_mblur(hr, m, 0.03 * W) - hr) / (0.06 * W), 0, 1) * 0.5
    ao = np.clip(ao * ao_k, 0, 1)
    P = {k: np.asarray(v, np.float32) for k, v in pal.items()}
    t_lit = _ss(ramp[0], ramp[1], e - 0.25 * ao)
    if gate is not None:   # no lit islands deep inside the big shadow side
        gx_, gy_, gz_ = normals([0, 0] + [0.5] * (len(scales) - 2), 0.9)
        Lg = np.asarray(light if light is not None else (0, -1, 0.3), np.float64)
        Lg = Lg / np.linalg.norm(Lg)
        eg = gx_ * Lg[0] + gy_ * Lg[1] + gz_ * Lg[2]
        if vbias is not None:
            eg = eg + vbias[2] + (vbias[3] - vbias[2]) * np.clip((ys - vbias[0]) / (vbias[1] - vbias[0]), 0, 1)
        t_lit = t_lit * _ss(gate[0], gate[1], eg)
    if gradmap is not None:
        # painted gradient map over the light value: shadow core -> sky-filled shadow -> half-tone ->
        # (crisp-ish) terminator -> lit -> hot; AO paints lavender crevice lines between lobes
        ee = e - ao_line * ao
        stops = gradmap
        pos = np.array([p_[0] for p_ in stops], np.float32)
        cols = np.array([p_[1] for p_ in stops], np.float32)
        ef = np.clip(ee, pos[0], pos[-1])
        idx = np.clip(np.searchsorted(pos, ef) - 1, 0, len(pos) - 2)
        f = (ef - pos[idx]) / (pos[idx + 1] - pos[idx])
        f = f * f * (3 - 2 * f)
        col = cols[idx] + (cols[idx + 1] - cols[idx]) * f[..., None]
        # sky fill on upward-facing shadow, bounce on downward-facing
        sh_w = _ss(0.0, -0.15, ee)[..., None]
        up = np.clip(-ny * 1.5, 0, 1)[..., None]
        down = np.clip(ny * 1.5, 0, 1)[..., None]
        col = col + (P['shd'] * 1.04 - col) * up * sh_w * 0.3
        col = col + (P['bounce'] - col) * down * sh_w * 0.35
        t_lit = _ss(-0.05, 0.08, ee)
    t_hot = _ss(0.3, 0.7, e)
    if gradmap is None:
        up = np.clip(-ny * 1.5, 0, 1)[..., None]
        down = np.clip(ny * 1.5, 0, 1)[..., None]
        shd = P['core'] + (P['shd'] - P['core']) * up * 0.8
        shd = shd + (P['bounce'] - shd) * down * 0.6
        shd = shd + (P['core'] * 0.92 - shd) * ao[..., None] * 0.5
        lit = P['lit_low'] + (P['lit'] - P['lit_low']) * _ss(0.0, 0.4, e)[..., None]
        lit = lit + (P['band'] - lit) * (ao * 0.6)[..., None]
        t_det = _ss(det_ramp[0], det_ramp[1], e_det - e)
        lit = lit + (P['half'] - lit) * ((1 - t_det) * det_k)[..., None]
        shd = shd + (P['shd'] * 1.08 - shd) * (np.clip(-dny * 1.4, 0, 1) * det_k * 0.8)[..., None]
        col = shd + (lit - shd) * t_lit[..., None]
        col = col + (P['hot'] - col) * (t_hot * 0.9)[..., None]
        band = np.exp(-((t_lit - 0.5) / 0.25) ** 2)[..., None]
        col = col + (P['band'] - col) * band * 0.25
    # silver lining / glow / halo
    prox = np.exp(-(sl / W) / 0.18) * 0.8 + 0.2
    cg = _fblur(alpha, 0.003 * W)
    ogx = -cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx ** 2 + ogy ** 2) + 1e-6
    oface = (np.clip((ogx * ex + ogy * ey) / ogl, 0, 1) * (ogl > 1e-3)).astype(np.float32)
    oface = _ss(0.05, 0.7, _fblur(oface, 0.002 * W))
    din = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    rim_l = np.exp(-din / max(0.0015 * W, 1.0)) * oface * prox * rim
    col = col + (P['rim'] - col) * np.clip(rim_l, 0, 1)[..., None]
    thin = np.exp(-din / (0.014 * W)) * oface * np.exp(-(sl / W) / 0.12) * glow
    col = col + (P['rim'] - col) * np.clip(thin, 0, 0.9)[..., None]
    if haze_y is not None:
        f = np.clip((ys - haze_y[0]) / (haze_y[1] - haze_y[0]), 0, 1) ** 1.4
        col = col + (P['haze'] - col) * (f * haze_k)[..., None]
    if halo:
        dout = cv2.distanceTransform((alpha < 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        hal = np.exp(-dout / (0.008 * W)) * _ss(0.05, 0.7, _fblur(oface, 0.006 * W)) * \
            np.exp(-(sl / W) / 0.2) * 0.45 * halo * (1 - alpha)
        a2 = alpha + hal
        col = (col * alpha[..., None] + P['rim'] * hal[..., None]) / np.maximum(a2, 1e-5)[..., None]
        alpha = a2
    out = np.dstack([col, np.clip(alpha, 0, 1)]).astype(np.float32)
    if return_attrs:
        return out, dict(e=e, ao=ao, t_lit=t_lit, sh=sh, alpha=alpha)
    return out


PAL2 = dict(
    hot=(1.05, 1.03, 0.98), lit=(0.97, 0.965, 0.95), lit_low=(0.95, 0.93, 0.92), band=(0.84, 0.84, 0.95),
    half=(0.78, 0.82, 0.94), core=(0.5, 0.58, 0.84), shd=(0.62, 0.7, 0.92), bounce=(0.72, 0.74, 0.9),
    rim=(1.32, 1.27, 1.17), haze=(0.74, 0.86, 0.97))

MODE = 'soft'


def render_tower(pw, ph, Dm, sun, base_y, W, H, top_y):
    if MODE == 'soft':
        return render_soft(pw, ph, Dm, sun, base_y, form=tower_form(Dm, pw, ph, base_y), **tower_shading(W, H, top_y, base_y))
    Hc = base_y - top_y
    return render_cloud(pw, ph, Dm, sun, base_y, pal=PAL2, lz_near=-0.1, lz_far=0.3, sun_rad=0.3, soft=0.035,
                        vbias=(top_y, base_y, 0.1, -0.35), rim=1.4, haze_y=(top_y + 0.5 * Hc, base_y + 0.02 * H))


def tower_form(domes, W, H, clip_y, r_frac=0.55, lean_up=0.35):
    """Per-tower big-form 'pillow' slopes. Each tower's own full silhouette -> distance transform ->
    rounded height (radius ~ r_frac of the tower's half width) -> slope; every pixel takes the slope of
    the tower that is frontmost there (crisp separation between overlapping towers).
    Returns a callable(i1, ss) -> (sx, sy) slope arrays at plate res, for render_soft(form=...)."""
    S = domes.arr()
    roots = S[:, 7].astype(np.int64)
    tid_of = np.array([domes.tid.get(int(r), 0) for r in range(len(S))], np.int64)
    tower = tid_of[roots]
    ids = sorted(set(tower.tolist()))

    def make(i1, ss):
        # frontmost tower id per plate pixel (nearest sample of the supersampled buffer)
        ii = i1[ss // 2::ss, ss // 2::ss][:H, :W]
        front = np.where(ii >= 0, tower[np.clip(ii, 0, None)], -1)
        sx = np.zeros((H, W), np.float32)
        sy = np.zeros((H, W), np.float32)
        for k in ids:
            sel = S[tower == k]
            _, ik = _zbuf(H, W, sel, clip_y)
            m = (ik >= 0).astype(np.uint8)
            del ik
            if m.sum() == 0:
                continue
            xs_ = np.nonzero(m.any(0))[0]
            hw = (xs_.max() - xs_.min()) / 2
            R = max(r_frac * hw, 4.0)
            d = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
            hgt = (np.sqrt(np.clip(1 - (1 - np.clip(d / R, 0, 1)) ** 2, 0, 1)) * R).astype(np.float32)
            hgt = _fblur(hgt, 0.25 * R).astype(np.float32)
            gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8
            gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8
            sel_px = front == k
            sx[sel_px] = -gx[sel_px]
            sy[sel_px] = -gy[sel_px] - lean_up * 0.0
        return sx, sy
    return make
