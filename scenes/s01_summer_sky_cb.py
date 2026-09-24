"""Painted cumulonimbus for s01 (round 3 rewrite: 2.5D height-field clouds with cast shadows).

Why not spheres: previous versions shaded every lobe as its own sphere -> soap foam. Here the cloud is
a 2.5D HEIGHT FIELD (orthographic view, z toward the viewer) made from a hierarchy of ellipsoids:
  * a few huge masses (columns of the tower, the anvil shelf, base shoulders),
  * medium cauliflower lobes budding off those masses, placed mostly along the silhouette and on the
    upward / sun-facing side (front faces stay broad and calm),
  * tiny lobes only on the rims of medium lobes that sit on the silhouette.
Lighting uses a normal that is mostly the SMOOTHED height field (one mass-scale form) plus a small share
of the lobe normal, so broad planes read as planes. Lobe definition comes from what a painter uses:
  * soft CAST SHADOWS marched through the height field (shadow shapes follow every lobe contour and
    the anvil throws its shadow on the crown),
  * cavity AO in the crevices between lobes,
  * a crisp lit edge on the sun side of each lobe.
Afterwards: silver lining + translucent glow on the silhouette toward the hidden sun, fibrous anvil end,
torn wispy base, atmospheric haze toward the base.
"""
import math
import os
import numpy as np
import cv2
from numba import njit, prange


def _c(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def fblur(img, sigma):
    """Gaussian blur; large sigmas done at reduced resolution (smooth, no blockiness)."""
    if sigma < 0.3:
        return img
    if sigma < 12:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    f = int(sigma // 5)
    h, w = img.shape[:2]
    sm = cv2.resize(img, (max(w // f, 2), max(h // f, 2)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sigma / f)
    return cv2.GaussianBlur(cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), f * 0.6)


def _noise(h, w, cells, seed):
    rng = np.random.default_rng(seed)
    gh, gw = max(int(h / cells) + 3, 3), max(int(w / cells) + 3, 3)
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)


# --------------------------------------------------------------------------------------------- palette
PAL = dict(
    deep=_c('#6670b0'),      # deepest core shadow (blue-violet)
    shadow=_c('#8390ca'),    # shadow side
    shadow_hi=_c('#9ea8dc'),  # shadow side facing the sky
    lav=_c('#bdb9e6'),       # lavender turn
    warm=_c('#f3e2e6'),      # warm-pink transition at the terminator
    lit_lo=_c('#f1ebf1'),    # lit plane facing away from the sun (slightly cool)
    lit=_c('#fff8ec'),       # sunlit cream-white
    hot=np.array([1.1, 1.07, 1.01], np.float32),
    sky=_c('#86aaec'),       # sky fill on up-facing shadow
    haze=_c('#d8ebf8'),      # horizon haze
)


def ramp(l, P=PAL):
    stops = [(0.0, P['deep']), (0.25, P['shadow']), (0.4, P['shadow_hi']), (0.48, P['lav']),
             (0.54, P['warm']), (0.6, P['lit_lo']), (0.78, P['lit']), (1.05, P['hot'])]
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([s[1] for s in stops], np.float32)
    out = np.empty(l.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(l, pos, cols[:, c])
    return out


# --------------------------------------------------------------------------------------------- geometry
class Cloud:
    """List of ellipsoids (cx, cy, cz, rx, ry, rz, level)."""

    def __init__(self):
        self.e = []

    def add(self, cx, cy, cz, rx, ry, rz, lev=0, clip=-1e9):
        self.e.append((cx, cy, cz, rx, ry, rz, lev, clip))
        return len(self.e) - 1

    def arr(self):
        return np.array(self.e, np.float64)


def bud(Cl, rng, parents, n, frac, min_r, up=1.2, sil=1.5, sun=(0.6, -0.8), sunw=0.6, flat=0.9,
        embed=(0.15, 0.4), lev=1, front_max=0.85, spread=None):
    """Grow child lobes on the visible surface of parent ellipsoids. Placement weights favour the
    silhouette (surface normal ~ perpendicular to the view), the upward side and the sun side, so front
    faces stay calm and the cauliflower detail lives along the rim and the terminator."""
    out = []
    for p in parents:
        cx, cy, cz, rx, ry, rz, _, clip = Cl.e[p]
        k = int(n) + (1 if rng.random() < n - int(n) else 0)
        m = 60
        th = rng.uniform(0, 2 * math.pi, m)
        nz = rng.uniform(0.0, 1.0, m) ** 1.3
        s = np.sqrt(1 - nz * nz)
        nx, ny = np.cos(th) * s, np.sin(th) * s
        w = (1 + sil * (1 - nz)) * np.exp(up * np.clip(-ny, -1, 1)) * (1 + sunw * np.clip(nx * sun[0] + ny * sun[1], 0, 1))
        w *= (nz < front_max)
        w *= (ny < 0.35)
        if spread is not None:
            w *= spread(nx, ny)
        if w.sum() <= 0:
            continue
        order = rng.choice(m, size=int((w > 0).sum()), replace=False, p=w / w.sum())
        placed = []
        for j in order:
            if len(placed) >= k:
                break
            r = min(rx, ry) * rng.uniform(*frac)
            if r < min_r:
                continue
            px, py, pz = cx + nx[j] * rx, cy + ny[j] * ry, cz + nz[j] * rz
            if any((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2 < (0.8 * (r + qr)) ** 2 for qx, qy, qz, qr in placed):
                continue
            placed.append((px, py, pz, r))
            e = rng.uniform(*embed)
            out.append(Cl.add(px - nx[j] * r * e, py - ny[j] * r * e * flat, pz - nz[j] * r * e,
                              r * rng.uniform(1.0, 1.15), r * flat * rng.uniform(0.85, 1.0), r * 0.9, lev, clip))
    return out


def tower(W, ax, base_y, top_y, seed=1):
    """Hero cumulonimbus. W = frame width (size unit). Returns (Cloud, info)."""
    rng = np.random.default_rng(seed)
    Cl = Cloud()
    Hc = base_y - top_y
    mn = max(1.2, 0.0028 * W)

    def Y(v):
        return base_y - v * Hc

    def axis(v):
        return ax + 0.04 * W * v ** 1.5

    anv = []
    # anvil: a wedge sheared to the right - thick where it leaves the crown, thinning and rising slightly
    for (du, dv, rx, ry, dz) in [(-0.12, 0.9, 0.07, 0.045, -0.05), (-0.03, 0.915, 0.1, 0.06, -0.06),
                                 (0.08, 0.925, 0.11, 0.058, -0.07), (0.18, 0.93, 0.1, 0.048, -0.08),
                                 (0.27, 0.935, 0.09, 0.036, -0.09), (0.35, 0.94, 0.075, 0.026, -0.1),
                                 (0.42, 0.945, 0.05, 0.016, -0.1)]:
        anv.append(Cl.add(axis(0.95) + du * W, Y(dv), dz * W, rx * W, ry * W, ry * W * 1.5, 0,
                          clip=Y(0.965) + (du + 0.12) * 0.03 * W))
    # cauliflower crown rising above the anvil level (the sun hides behind its right shoulder)
    dome = Cl.add(axis(0.97) + 0.03 * W, Y(0.93), 0.01 * W, 0.085 * W, 0.075 * W, 0.08 * W, 0)
    crown = [dome, Cl.add(axis(0.95) - 0.06 * W, Y(0.9), 0.0, 0.065 * W, 0.058 * W, 0.065 * W, 0),
             Cl.add(axis(0.95) + 0.115 * W, Y(0.9), 0.0, 0.06 * W, 0.052 * W, 0.06 * W, 0)]
    # one big rounded BODY (a few huge vertical ellipsoids around the axis, blended by smooth-max) with
    # TURRETS (column heads) bulging out of its silhouette and crown, and a few broad frontal bulges.
    hw = lambda v: W * float(np.interp(v, [0.0, 0.2, 0.45, 0.65, 0.85, 1.0], [0.26, 0.2, 0.15, 0.16, 0.19, 0.16]))

    def zcyl(x, v, amt=0.13):
        u = (x - axis(v)) / hw(v)
        return amt * W * math.sqrt(max(1 - u * u, 0.0))
    body = []
    for v in (0.12, 0.3, 0.48, 0.66, 0.8):
        h_ = hw(v)
        body.append(Cl.add(axis(v), Y(v), -0.02 * W, h_ * 0.9, 0.13 * Hc, 0.14 * W, -1))
    heads = []
    # silhouette turrets: (side u, v, R)
    turr = [(-0.05, 0.9, 0.08), (0.5, 0.87, 0.07), (-0.6, 0.83, 0.07), (1.0, 0.77, 0.075),
            (-1.05, 0.72, 0.06), (0.95, 0.64, 0.05), (-0.9, 0.6, 0.045), (1.15, 0.55, 0.065),
            (-1.1, 0.5, 0.07), (0.9, 0.44, 0.045), (-1.0, 0.4, 0.05), (1.1, 0.33, 0.07),
            (-1.15, 0.28, 0.08), (1.05, 0.2, 0.06), (-0.95, 0.16, 0.06), (0.2, 0.81, 0.06),
            (-0.3, 0.74, 0.055), (0.55, 0.68, 0.05), (-0.7, 0.93, 0.05), (0.75, 0.95, 0.055)]
    for (u, v, R) in turr:
        R = R * W * rng.uniform(0.9, 1.1)
        x = axis(v) + u * (hw(v) - R * 0.55)
        z = zcyl(x, v) - 0.3 * R
        heads.append(Cl.add(x, Y(v) + R * 0.35, z, R, R * 0.9, R, 0))
    # broad frontal bulges (big, calm; they give the front its large planes and a few crevices)
    for (u, v, R) in [(-0.35, 0.58, 0.085), (0.3, 0.5, 0.09), (-0.2, 0.36, 0.09), (0.35, 0.3, 0.085),
                      (0.0, 0.7, 0.08), (-0.45, 0.25, 0.08), (0.1, 0.2, 0.09)]:
        R = R * W * rng.uniform(0.95, 1.05)
        x = axis(v) + u * hw(v)
        heads.append(Cl.add(x, Y(v) + R * 0.2, zcyl(x, v) - 0.55 * R, R * 1.05, R * 0.85, R, 0))
    bodies = body
    base = []
    for (du, dv, R) in [(-0.31, 0.1, 0.085), (-0.15, 0.05, 0.11), (0.06, 0.06, 0.12), (0.26, 0.08, 0.1),
                        (-0.45, 0.04, 0.075), (0.42, 0.04, 0.08)]:
        base.append(Cl.add(axis(dv) + du * W, Y(dv), 0.14 * W, R * W * 1.3, R * W * 0.75, R * W * 0.8, 0))
    sunv = (0.55, -0.83)
    l1 = bud(Cl, rng, heads + crown, 14, (0.26, 0.42), mn, up=1.8, sil=2.0, sun=sunv, front_max=0.7)
    l1 += bud(Cl, rng, bodies, 6, (0.2, 0.3), mn, up=0.8, sil=3.0, sun=sunv, front_max=0.4)
    l1 += bud(Cl, rng, base, 6, (0.25, 0.4), mn, up=1.5, sil=1.5, sun=sunv)
    la = bud(Cl, rng, anv[:6], 5, (0.28, 0.42), mn, up=3.0, sil=1.0, sun=sunv, flat=0.65, front_max=0.55)
    l2 = bud(Cl, rng, l1, 4, (0.3, 0.45), mn, up=1.5, sil=3.0, sun=sunv, front_max=0.6, lev=2)
    bud(Cl, rng, l2, 1.2, (0.32, 0.48), mn, up=1.2, sil=3.0, sun=sunv, front_max=0.4, lev=3)
    bud(Cl, rng, la, 2, (0.3, 0.45), mn, up=2.0, sil=2.0, sun=sunv, flat=0.75, front_max=0.5, lev=2)
    info = dict(anvil=anv, dome=dome, axis_top=axis(1.0))
    return Cl, info


def cumulus(Cl, W, rng, cx, base_y, width, height, z0=0.0):
    """A secondary cumulus into an existing Cloud: flat base row + a tier or two."""
    mn = max(1.2, 0.0028 * W)
    n = max(2, int(width / (0.07 * W)))
    heads = []
    for k in range(n):
        u = (k + 0.5) / n - 0.5
        prof = math.sqrt(max(1 - (2 * u) ** 2, 0.08))
        R = width / n * rng.uniform(0.7, 0.9)
        x = cx + u * width * 0.9
        heads.append(Cl.add(x, base_y - R * 0.5, z0, R * 1.2, R * 0.6, R * 0.8, 0))
        h = height * prof
        if h > R * 1.3:
            R2 = R * rng.uniform(0.75, 0.9)
            heads.append(Cl.add(x + rng.uniform(-0.2, 0.2) * R, base_y - h + R2, z0 + 0.3 * R, R2, R2 * 0.9, R2, 0))
    sunv = (0.55, -0.83)
    l1 = bud(Cl, rng, heads, 6, (0.3, 0.45), mn, up=1.8, sil=2.0, sun=sunv)
    l2 = bud(Cl, rng, l1, 3, (0.3, 0.45), mn, up=1.2, sil=3.0, sun=sunv, front_max=0.6, lev=2)
    bud(Cl, rng, l2, 2, (0.32, 0.48), mn, up=1.0, sil=3.0, sun=sunv, front_max=0.5, lev=3)


# --------------------------------------------------------------------------------------------- raster
@njit(cache=True, fastmath=True)
def _raster(Hs, Ws, E, ks):
    hmax = np.full((Hs, Ws), -1e9, np.float32)
    cov = np.zeros((Hs, Ws), np.float32)
    nrm = np.zeros((Hs, Ws, 3), np.float32)
    for k in range(E.shape[0]):
        cx, cy, cz, rx, ry, rz = E[k, 0], E[k, 1], E[k, 2], E[k, 3], E[k, 4], E[k, 5]
        mr = min(rx, ry)
        x0 = max(int(cx - rx) - 2, 0)
        x1 = min(int(cx + rx) + 3, Ws)
        y0 = max(int(cy - ry) - 2, 0)
        y1 = min(int(cy + ry) + 3, Hs)
        y0 = max(y0, int(E[k, 7]))
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / ry
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / rx
                q2 = dx * dx + dy * dy
                sd = (math.sqrt(q2) - 1.0) * mr
                c = 0.5 - sd
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                if c > cov[y, x]:
                    cov[y, x] = c
                qq = min(q2, 0.999)
                dz = math.sqrt(1.0 - qq)
                h = cz + rz * dz
                kk = ks[k]
                a0 = hmax[y, x]
                if kk > 0.0 and a0 > -1e8:
                    hh = max(kk - abs(a0 - h), 0.0) / kk
                    hs = max(a0, h) + hh * hh * kk * 0.25
                    if hs > a0:
                        hmax[y, x] = hs
                if h > a0:
                    if not (kk > 0.0 and a0 > -1e8):
                        hmax[y, x] = h
                    nx = dx / rx
                    ny = dy / ry
                    nz = dz / rz
                    n = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
                    nrm[y, x, 0] = nx / n
                    nrm[y, x, 1] = ny / n
                    nrm[y, x, 2] = nz / n
    return hmax, cov, nrm


@njit(cache=True, fastmath=True, parallel=True)
def _shadow(Hf, A, lx, ly, lz, step, nsteps, soft):
    """Soft cast shadows through a height field. (lx, ly) unit screen dir toward the light, lz = rise of
    the light per unit screen distance. Returns visibility 0..1."""
    Hs, Ws = Hf.shape
    vis = np.ones((Hs, Ws), np.float32)
    for y in prange(Hs):
        for x in range(Ws):
            if A[y, x] < 0.05:
                continue
            h0 = Hf[y, x]
            v = 1.0
            for s in range(1, nsteps):
                t = s * step
                ix = int(x + lx * t)
                iy = int(y + ly * t)
                if ix < 0 or iy < 0 or ix >= Ws or iy >= Hs:
                    break
                if A[iy, ix] < 0.5:
                    continue
                d = (h0 + lz * t) - Hf[iy, ix]
                if d < 0:
                    o = -d / (soft * t + 1.0)
                    v = min(v, max(0.0, 1.0 - o))
                    if v <= 0.0:
                        break
            vis[y, x] = v
    return vis


@njit(cache=True, fastmath=True, parallel=True)
def _depth(A, lx, ly, step, nsteps):
    """Screen-space optical depth: amount of cloud between each pixel and the light (px)."""
    Hs, Ws = A.shape
    out = np.zeros((Hs, Ws), np.float32)
    for y in prange(Hs):
        for x in range(Ws):
            if A[y, x] < 0.05:
                continue
            acc = 0.0
            for s in range(1, nsteps):
                t = s * step
                ix = int(x + lx * t)
                iy = int(y + ly * t)
                if ix < 0 or iy < 0 or ix >= Ws or iy >= Hs:
                    break
                acc += A[iy, ix] * step
            out[y, x] = acc
    return out


def render(pw, ph, Cl, sun, W, L=(0.62, -0.62, 0.38), base_y=None, top_y=None, P=PAL, haze_amt=0.0,
           rim_amt=1.0, wisp=None, seed=0, fib=None, shadow_len=0.25, lobe_w=0.22, extra_haze=None):
    """Render a Cloud into a straight-alpha RGBA plate (ph, pw, 4).
    sun: (x, y) plate position of the (hidden) sun. L: light dir (x right, y down, z toward viewer)."""
    L = np.array(L, np.float64)
    L /= np.linalg.norm(L)
    E = Cl.arr()
    ks = np.where(E[:, 6] < 0, 0.03 * W, np.where(E[:, 6] == 0, 0.012 * W, 0.0)).astype(np.float64)
    hmax, cov, nrm = _raster(ph, pw, E.astype(np.float64), ks)
    A = cov
    wx = _noise(ph, pw, 0.02 * W, seed + 1) * 0.002 * W + _noise(ph, pw, 0.006 * W, seed + 2) * 0.0007 * W
    wy = _noise(ph, pw, 0.02 * W, seed + 3) * 0.002 * W + _noise(ph, pw, 0.006 * W, seed + 4) * 0.0007 * W
    gy, gx = np.mgrid[0:ph, 0:pw].astype(np.float32)
    mx, my = gx + wx, gy + wy
    hf = np.where(A > 0, hmax, 0).astype(np.float32)
    hf = cv2.remap(hf, mx, my, cv2.INTER_LINEAR)
    A = cv2.remap(A, mx, my, cv2.INTER_LINEAR)
    nrm = cv2.remap(nrm, mx, my, cv2.INTER_LINEAR)

    def nblur(x, s):
        return fblur(x * A, s) / (fblur(A, s) + 1e-4)
    Hs1 = nblur(hf, 0.012 * W)
    Hs2 = nblur(hf, 0.04 * W)
    Hs3 = nblur(hf, 0.09 * W)
    Hsm = 0.4 * Hs1 + 0.3 * Hs2 + 0.3 * Hs3
    hy_, hx_ = np.gradient(Hsm)
    n = np.sqrt(hx_ * hx_ + hy_ * hy_ + 1)
    Ns = np.dstack([-hx_ / n, -hy_ / n, 1 / n])
    N = Ns * (1 - lobe_w) + nrm * lobe_w
    N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-6
    ndl = N[..., 0] * L[0] + N[..., 1] * L[1] + N[..., 2] * L[2]
    ndl_lobe = nrm[..., 0] * L[0] + nrm[..., 1] * L[1] + nrm[..., 2] * L[2]
    ndl_s = Ns[..., 0] * L[0] + Ns[..., 1] * L[1] + Ns[..., 2] * L[2]
    h2, w2 = ph // 2, pw // 2
    Hh = cv2.resize(cv2.GaussianBlur(hf, (0, 0), 0.002 * W), (w2, h2), interpolation=cv2.INTER_AREA) / 2
    Ah = cv2.resize(A, (w2, h2), interpolation=cv2.INTER_AREA)
    l2 = math.hypot(L[0], L[1])
    vis = _shadow(Hh.astype(np.float32), Ah.astype(np.float32), L[0] / l2, L[1] / l2, L[2] / l2, 1.0,
                  int(shadow_len * W / 2), 0.04)
    od = _depth(cv2.resize(A, (pw // 4, ph // 4), interpolation=cv2.INTER_AREA).astype(np.float32),
                L[0] / l2, L[1] / l2, 1.0, int(0.4 * W / 4)) * 4
    od = cv2.resize(cv2.GaussianBlur(od, (0, 0), 0.006 * W / 4), (pw, ph), interpolation=cv2.INTER_CUBIC)
    mass = np.exp(-od / (0.13 * W))
    kk = max(3, int(0.009 * W / 2) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    vis = cv2.morphologyEx(vis, cv2.MORPH_CLOSE, ker)          # drop tiny isolated shadow pits
    vis = cv2.resize(cv2.GaussianBlur(vis, (0, 0), 0.002 * W), (pw, ph), interpolation=cv2.INTER_LINEAR)
    cav = np.clip((nblur(hf, 0.008 * W) - hf) / (0.01 * W), 0, 1)
    cav2 = np.clip((Hs2 - hf) / (0.05 * W), 0, 1)
    if base_y is not None and top_y is not None:
        vv = np.clip((base_y - gy) / (base_y - top_y), 0, 1)
    else:
        vv = np.full_like(gy, 0.6)
    g3y, g3x = np.gradient(Hs3)
    n3 = np.sqrt(g3x * g3x + g3y * g3y + 1)
    ndl3 = (-g3x * L[0] - g3y * L[1] + L[2]) / n3
    diff = _ss(0.0, 0.14, ndl)
    lit = diff * _ss(0.4, 0.6, vis)
    sprox = np.exp(-np.hypot(gx - sun[0], gy - sun[1]) / (0.35 * W))
    l = (0.18 + lit * (0.22 + 0.3 * np.clip(ndl, 0, 1)) + 0.1 * np.clip(ndl, -0.5, 0)
         + 0.12 * np.clip(ndl_lobe - ndl_s, -0.6, 0.6)
         + 0.2 * (ndl3 - 0.6) + 0.45 * (mass - 0.35) + 0.1 * (vv - 0.5) + 0.1 * sprox * lit
         - 0.18 * cav - 0.14 * cav2)
    if os.environ.get('CBDBG'):
        m_ = A > 0.9
        for nm, v_ in (('l', l), ('ndl', ndl), ('ndl3', ndl3), ('mass', mass), ('lit', lit), ('vis', vis)):
            print(nm, np.round(np.percentile(v_[m_], [5, 25, 50, 75, 95]), 3))
    L2 = np.array([L[0], L[1]]) / l2
    face = np.clip(nrm[..., 0] * L2[0] + nrm[..., 1] * L2[1], 0, 1)
    edge = _ss(0.55, 0.9, 1 - nrm[..., 2]) * face * vis
    l = l + 0.16 * edge
    c = ramp(l, P)
    shade = 1 - _ss(0.35, 0.55, l)
    up = np.clip(-N[..., 1], 0, 1)
    c = c + (P['sky'] - c) * ((0.15 + 0.35 * up) * shade * 0.5)[..., None]
    if haze_amt:
        hzk = haze_amt * (1 - _ss(0.03, 0.45, vv))
        c = c + (P['haze'] - c) * hzk[..., None]
    if extra_haze is not None:
        c = c + (P['haze'] - c) * extra_haze[..., None]
    col = c
    alp = np.clip(A, 0, 1)
    Ab = (alp > 0.5).astype(np.uint8)
    dU = cv2.distanceTransform(Ab, cv2.DIST_L2, 5) + (alp - 0.5) * (alp > 0.02)
    dU = np.clip(dU, 0, None)
    Ub = cv2.GaussianBlur(alp, (0, 0), max(1.0, 0.0025 * W))
    oy, ox = np.gradient(Ub)
    on = np.sqrt(ox * ox + oy * oy) + 1e-6
    ox, oy = -ox / on, -oy / on
    tx, ty = sun[0] - gx, sun[1] - gy
    ds = np.sqrt(tx * tx + ty * ty) + 1e-3
    face = np.clip((ox * tx + oy * ty) / ds, 0, 1)
    facel = np.clip(ox * L2[0] + oy * L2[1], 0, 1)
    prox = np.exp(-ds / (0.14 * W))
    prox2 = np.exp(-ds / (0.5 * W))
    rim_w = max(0.0011 * W, 0.7)
    rim = np.exp(-dU / rim_w) * (face ** 0.8 * (0.2 + 1.8 * prox) + 0.5 * facel ** 1.5 * prox2) * rim_amt
    rim *= (alp > 0.01)
    trans = np.exp(-dU / (0.01 * W)) * prox * face ** 0.5 * rim_amt
    col = col + rim[..., None] * np.array([1.25, 1.2, 1.1], np.float32)
    col = col + (np.array([1.15, 1.1, 1.0], np.float32) - col) * np.clip(trans * 0.8, 0, 0.85)[..., None]
    alp = alp * (1 - 0.4 * np.exp(-dU / (0.0025 * W)) * prox)
    if fib is not None:
        f0, f1, fy0, fy1 = fib
        st = _noise(ph, pw, 0.012 * W, seed + 11)
        st = cv2.GaussianBlur(st, (0, 0), sigmaX=0.03 * W, sigmaY=0.001 * W)
        st = st / (st.std() + 1e-6)
        k = _ss(f0, f1, gx) * _ss(fy1 + 0.02 * W, fy1 - 0.01 * W, gy)
        alp = alp * np.clip(1 - k * (0.45 + 0.4 * np.clip(st, -1, 1)), 0, 1)
        col = col + (np.array([1.05, 1.03, 1.0], np.float32) - col) * (k * 0.35)[..., None]
    if wisp is not None:
        wy0, wy1 = wisp
        n1 = _noise(ph, pw, 0.03 * W, seed + 21)
        n1 = cv2.GaussianBlur(n1, (0, 0), sigmaX=0.03 * W, sigmaY=0.002 * W)
        n2 = _noise(ph, pw, 0.008 * W, seed + 22)
        n2 = cv2.GaussianBlur(n2, (0, 0), sigmaX=0.012 * W, sigmaY=0.001 * W)
        n1 /= n1.std() + 1e-6
        n2 /= n2.std() + 1e-6
        yy2 = gy + (n1 * 0.5 + n2 * 0.25) * (wy1 - wy0) * 0.6
        k = _ss(wy1, wy0, yy2)
        alp = alp * k
        col = col + (P['haze'] - col) * ((1 - _ss(wy1, wy0 - 0.1 * W, gy)) * 0.6)[..., None]
    return np.dstack([col, np.clip(alp, 0, 1)]).astype(np.float32)
