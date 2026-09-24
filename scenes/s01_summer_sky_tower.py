"""Painted cumulonimbus for s01 (round 2 renderer).

Geometry: an explicit lobe hierarchy (big billows -> mid lobes -> small cauliflower bumps that only sit
on the outward/upper surface) of slightly irregular spheroids, rasterised in a supersampled z-buffer.
Shading is designed like a painting:
  * every lobe gets its OWN N.L gradient (lobe normal blended with its parent / root billow and a smooth
    whole-tower form normal) -> continuous graded volumes; a sharp terminator appears only where one lobe
    overlaps another (z-buffer boundary), never as cut-out patches;
  * gradient map warm white (#fffaf0) -> pale lavender -> #8a95c8 -> #6f7fb8 core;
  * crevice AO from the height field (full-res Gaussian, no down/up-sampled derivatives -> no tile seams);
  * anvil cast shadow on the column, silver lining + translucent fringe + halo near the sun, aerial haze
    and torn wisps at the base.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

# lobe row layout
CX, CY, CZ, R, AY, LEV, PAR, ROOT, P1, P2, FLAG, TID = range(12)


class Lobes:
    def __init__(self):
        self.rows = []

    def add(self, cx, cy, cz, r, ay, lev, parent=-1, flag=0, tid=0, rng=None):
        i = len(self.rows)
        root = i if parent < 0 else int(self.rows[parent][ROOT])
        p1, p2 = (rng.uniform(0, 6.283), rng.uniform(0, 6.283)) if rng is not None else (0.0, 0.0)
        self.rows.append([cx, cy, cz, r, ay, lev, parent, root, p1, p2, flag, tid])
        return i

    def arr(self):
        return np.array(self.rows, np.float64)


def grow(D, rng, parents, count, ratio, up_bias=0.9, front=(-0.1, 0.98), minr=1.0, clip_y=1e9,
         reach=(0.0, 0.3), max_uy=1.0, rvar=(0.7, 1.25), ay_k=(0.94, 1.04), up_dir=-math.pi / 2):
    """Cover the visible surface of each parent with child lobes: blue-noise directions over the front
    hemisphere biased toward the crown (up_bias = angular std), children sit on the parent surface."""
    kids = []
    for pi in parents:
        row = D.rows[pi]
        px, py, pz, pr, pay = row[CX], row[CY], row[CZ], row[R], row[AY]
        rr = pr * ratio
        if rr < minr:
            continue
        n = int(count + rng.random())
        m = n * 14
        dz = rng.uniform(front[0], front[1], m)
        a = up_dir + rng.normal(0, up_bias, m)
        s_ = np.sqrt(np.clip(1 - dz * dz, 0, 1))
        V = np.stack([np.cos(a) * s_, np.sin(a) * s_, dz], 1)
        V = V[V[:, 1] <= max_uy]
        chosen = []
        thr = math.cos(min(1.2 * ratio + 0.05, 1.2))
        for k in range(len(V)):
            if len(chosen) >= n:
                break
            if all(float(V[k] @ V[c]) < thr for c in chosen):
                chosen.append(k)
        for k in chosen:
            ux, uy, uz = V[k]
            r_ = rr * rng.uniform(*rvar)
            rc = pr - r_ * rng.uniform(*reach)
            x = px + ux * rc
            y = py + uy * rc * pay
            z = pz + uz * rc
            if y + r_ * 0.2 > clip_y:
                continue
            ay = min(max(pay * rng.uniform(*ay_k), 0.3), 1.0)
            kids.append(D.add(x, y, z, r_, ay, row[LEV] + 1, pi, row[FLAG], row[TID], rng))
    return kids


# =============================================================================== raster + shade

@njit(cache=True, fastmath=True)
def _rf(S, i, dx, dy):
    """irregular lobe radius factor (hand-painted, not perfect circles)"""
    th = math.atan2(dy, dx)
    return 1.0 + 0.045 * math.sin(3.0 * th + S[i, 8]) + 0.025 * math.sin(5.0 * th + S[i, 9]) \
        + 0.012 * math.sin(9.0 * th + S[i, 8] * 2.0)


@njit(cache=True, fastmath=True, parallel=True)
def _zbuf(Hs, Ws, S, clip, order):
    """Z-buffer of irregular spheroids. Rows split across threads by band (race free)."""
    h1 = np.full((Hs, Ws), -1e12, np.float32)
    i1 = np.full((Hs, Ws), -1, np.int32)
    nb = 64
    band = (Hs + nb - 1) // nb
    n = S.shape[0]
    for b in prange(nb):
        by0 = b * band
        by1 = min(Hs, by0 + band)
        for q in range(n):
            i = order[q]
            cx, cy, cz, r, ay = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4]
            ry = r * ay * 1.1
            y0 = max(by0, int(cy - ry - 1))
            y1 = min(by1, int(min(cy + ry, clip) + 2))
            if y1 <= y0:
                continue
            x0 = max(0, int(cx - r * 1.1 - 1))
            x1 = min(Ws, int(cx + r * 1.1 + 2))
            for y in range(y0, y1):
                yy = y + 0.5
                if yy > clip:
                    continue
                dy = (yy - cy) / (r * ay)
                for x in range(x0, x1):
                    dx = (x + 0.5 - cx) / r
                    d2 = dx * dx + dy * dy
                    if d2 >= 1.3:
                        continue
                    f = _rf(S, i, dx, dy)
                    d2f = d2 / (f * f)
                    if d2f >= 1.0:
                        continue
                    h = cz + r * f * math.sqrt(1.0 - d2f)
                    if h > h1[y, x]:
                        h1[y, x] = h
                        i1[y, x] = i
    return h1, i1


@njit(cache=True, fastmath=True, inline='always')
def _nrm(S, j, x, y):
    cx, cy, r, ay = S[j, 0], S[j, 1], S[j, 3], S[j, 4]
    dx = (x - cx) / r
    dy = (y - cy) / (r * ay)
    f = _rf(S, j, dx, dy)
    dx /= f
    dy /= f
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
def _shade(H, W, ss, i1, h1, S, FN, L, wts, out):
    """out (H, W, 7): e (N.L), e_form, ny (blended), coverage, height, level, anvil  (ss^2 averaged)."""
    inv = 1.0 / (ss * ss)
    for Y in prange(H):
        for X in range(W):
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            a4 = 0.0
            a5 = 0.0
            a6 = 0.0
            a7 = 0.0
            a8 = 0.0
            a9 = 0.0
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
                    n0x, n0y, n0z = _nrm(S, i, px, py)
                    p = int(S[i, 6])
                    if p >= 0:
                        n1x, n1y, n1z = _nrm(S, p, px, py)
                    else:
                        n1x, n1y, n1z = n0x, n0y, n0z
                    rt = int(S[i, 7])
                    n2x, n2y, n2z = _nrm(S, rt, px, py)
                    Nx = wts[0] * n0x + wts[1] * n1x + wts[2] * n2x + wts[3] * fnx
                    Ny = wts[0] * n0y + wts[1] * n1y + wts[2] * n2y + wts[3] * fny
                    Nz = wts[0] * n0z + wts[1] * n1z + wts[2] * n2z + wts[3] * fnz
                    l = math.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-6
                    a0 += (Nx * L[0] + Ny * L[1] + Nz * L[2]) / l
                    Bx = 0.5 * n2x + 0.5 * fnx
                    By = 0.5 * n2y + 0.5 * fny
                    Bz = 0.5 * n2z + 0.5 * fnz
                    lb = math.sqrt(Bx * Bx + By * By + Bz * Bz) + 1e-6
                    a1 += (Bx * L[0] + By * L[1] + Bz * L[2]) / lb
                    a2 += Ny / l
                    a4 += h1[y, x] / ss
                    a5 += S[i, 5]
                    a6 += S[i, 10]
                    a7 += n0x * L[0] + n0y * L[1] + n0z * L[2]
                    a8 += n1x * L[0] + n1y * L[1] + n1z * L[2]
                    a9 += n0y
                    cov += 1.0
            if cov > 0:
                out[Y, X, 0] = a0 / cov
                out[Y, X, 1] = a1 / cov
                out[Y, X, 2] = a2 / cov
                out[Y, X, 4] = a4 / cov
                out[Y, X, 5] = a5 / cov
                out[Y, X, 6] = a6 / cov
                out[Y, X, 7] = a7 / cov
                out[Y, X, 8] = a8 / cov
                out[Y, X, 9] = a9 / cov
            out[Y, X, 3] = cov * inv


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def gblur(img, sigma):
    """Gaussian blur; very large sigmas via a pyramid with CUBIC upsampling of the blurred result
    (smooth, never used for derivatives at low res)."""
    if sigma <= 0.3:
        return img
    if sigma <= 12:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 6.0
    w, h = max(int(W / f), 4), max(int(H / f), 4)
    small = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 6.0 * w / W)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)


def form_normals(i1, S, tid_of, ss, W, H, r_frac=0.6):
    """Smooth whole-tower 'pillow' normals, per tower (frontmost tower wins). Gradients are computed at
    a reduced resolution on a smooth field and upsampled with CUBIC interpolation (no block seams)."""
    q = 4
    w, h = W // q + 1, H // q + 1
    ii = i1[::ss * q, ::ss * q][:h, :w]
    hh, ww = ii.shape
    tid = np.where(ii >= 0, tid_of[np.clip(ii, 0, None)], -1)
    nx = np.zeros((hh, ww), np.float32)
    ny = np.zeros((hh, ww), np.float32)
    for k in np.unique(tid):
        if k < 0:
            continue
        # silhouette of tower k alone (its lobes, regardless of occlusion)
        m = np.zeros((hh, ww), np.uint8)
        for j in np.nonzero(tid_of == k)[0]:
            cx, cy, r, ay = S[j, CX] / (ss * q), S[j, CY] / (ss * q), S[j, R] / (ss * q), S[j, AY]
            cv2.ellipse(m, (int(cx * 4), int(cy * 4)), (max(int(r * 4), 1), max(int(r * ay * 4), 1)), 0, 0, 360, 1,
                        -1, cv2.LINE_8, 2)
        if m.sum() == 0:
            continue
        cols = np.nonzero(m.any(0))[0]
        hw = (cols.max() - cols.min()) / 2
        Rr = max(r_frac * hw, 3.0)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)
        hgt = (np.sqrt(np.clip(1 - (1 - np.clip(d / Rr, 0, 1)) ** 2, 0, 1)) * Rr).astype(np.float32)
        hgt = cv2.GaussianBlur(hgt, (0, 0), 0.3 * Rr)
        gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=5) / 128.0 * 4
        gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=5) / 128.0 * 4
        sel = tid == k
        nx[sel] = -gx[sel]
        ny[sel] = -gy[sel]
    nx = cv2.GaussianBlur(nx, (0, 0), 1.0)
    ny = cv2.GaussianBlur(ny, (0, 0), 1.0)
    nx = cv2.resize(nx, (W + q, H + q), interpolation=cv2.INTER_CUBIC)[:H, :W]
    ny = cv2.resize(ny, (W + q, H + q), interpolation=cv2.INTER_CUBIC)[:H, :W]
    nz = np.ones_like(nx)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.dstack([nx / l, ny / l, nz / l]).astype(np.float32)


# painted gradient map over the light value e
GRAD = [(-0.8, (0.40, 0.46, 0.72)), (-0.32, (0.435, 0.498, 0.735)), (-0.06, (0.541, 0.584, 0.8)),
        (0.05, (0.64, 0.66, 0.85)), (0.12, (0.84, 0.83, 0.9)), (0.2, (0.96, 0.945, 0.93)),
        (0.35, (1.0, 0.975, 0.935)), (0.7, (1.02, 0.99, 0.94)), (0.95, (1.06, 1.03, 0.97))]


def gradmap(e, stops=GRAD):
    pos = np.array([p[0] for p in stops], np.float32)
    cols = np.array([p[1] for p in stops], np.float32)
    ef = np.clip(e, pos[0], pos[-1])
    idx = np.clip(np.searchsorted(pos, ef) - 1, 0, len(pos) - 2)
    f = (ef - pos[idx]) / (pos[idx + 1] - pos[idx])
    f = f * f * (3 - 2 * f)
    return cols[idx] + (cols[idx + 1] - cols[idx]) * f[..., None]


def render(W, H, lobes, sun, clip_y, Wref, L=(0.72, -0.55, 0.25), ss=2, wts=(0.2, 0.17, 0.22, 0.41),
           vbias=None, haze=None, haze_col=(0.80, 0.90, 0.985), rim=1.0, sky_fill=(0.62, 0.72, 0.95),
           anvil_shadow=0.5):
    """Rasterise + paint. W, H plate size; lobes/sun/clip_y in plate px; Wref = screen width (scale).
    vbias = (y_top, y_base, b_top, b_base) height bias of the light value.
    haze = (y0, y1, k) aerial perspective toward the base. Returns (rgba, attrs)."""
    S = lobes.arr()
    Ss = S.copy()
    for k in (CX, CY, CZ, R):
        Ss[:, k] *= ss
    order = np.argsort(Ss[:, CZ] + Ss[:, R]).astype(np.int64)
    h1, i1 = _zbuf(H * ss, W * ss, Ss, clip_y * ss, order)
    tid_of = S[:, TID].astype(np.int64)
    FN = form_normals(i1, Ss, tid_of, ss, W, H)
    Lv = np.asarray(L, np.float64)
    Lv = Lv / np.linalg.norm(Lv)
    out = np.zeros((H, W, 10), np.float32)
    _shade(H, W, ss, i1, h1, Ss, FN, Lv, np.array(wts, np.float64), out)
    del h1, i1
    e, ef, ny, alpha, hgt, lev, anv, e0, e1, ny0 = [out[..., k] for k in range(10)]
    m = (alpha > 0.01).astype(np.float32)
    ys = np.arange(H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
    xs = np.arange(W, dtype=np.float32)[None, :] * np.ones((H, 1), np.float32)
    if vbias is not None:
        f = np.clip((ys - vbias[0]) / (vbias[1] - vbias[0]), 0, 1)
        bias = vbias[2] + (vbias[3] - vbias[2]) * f ** 1.2
        e = e + bias
        ef = ef + bias
    # --- break the long straight big-form terminator into a wandering, painted edge
    nrng = np.random.default_rng(17)
    nw_, nh_ = max(W // 16, 4), max(H // 16, 4)
    nz = cv2.GaussianBlur(nrng.standard_normal((nh_, nw_)).astype(np.float32), (0, 0), 0.035 * Wref / 16)
    nz = cv2.resize(nz / (nz.std() + 1e-6), (W, H), interpolation=cv2.INTER_CUBIC)
    e = e + 0.14 * nz
    ef = ef + 0.14 * nz
    # --- backlight: right under the hidden sun we look into the shadowed side of the crown
    sdx = xs - sun[0]
    sdy = ys - sun[1]
    sdist = np.sqrt(sdx * sdx + sdy * sdy) / Wref
    back = 0.55 * np.exp(-sdist / 0.12)
    e = e - back
    ef = ef - back
    # --- anvil cast shadow on the column below/left of the overhang (light from upper right)
    if anvil_shadow:
        am = (anv > 0.5).astype(np.float32) * m
        dx, dy = -Lv[0] / math.hypot(Lv[0], Lv[1]), -Lv[1] / math.hypot(Lv[0], Lv[1])
        acc = np.zeros_like(am)
        n = 10
        for k in range(1, n + 1):
            d = 0.012 * Wref * k
            M = np.float32([[1, 0, dx * d], [0, 1, dy * d]])
            acc = np.maximum(acc, cv2.warpAffine(am, M, (W, H)) * (1 - 0.06 * k))
        csh = gblur(acc, 0.006 * Wref) * (1 - am) * m
        e = e - anvil_shadow * csh
        ef = ef - anvil_shadow * csh
    # --- crevice AO from the height field (masked, full-res blur)
    hm = hgt * m
    s1 = 0.006 * Wref
    nb = gblur(hm, s1) / (gblur(m, s1) + 1e-4)
    ao = np.clip((nb - hgt) / (0.02 * Wref), 0, 1) * m
    s2 = 0.02 * Wref
    nb2 = gblur(hm, s2) / (gblur(m, s2) + 1e-4)
    ao2 = np.clip((nb2 - hgt) / (0.06 * Wref), 0, 1) * m
    ao = np.clip(ao * 0.65 + ao2 * 0.5, 0, 1)
    # lobe relief is strongest around the big-form terminator, calm on the broad lit plateau
    relief = 0.3 + 0.7 * np.exp(-((ef - 0.08) / 0.22) ** 2) + 0.25 * _ss(0.2, 0.5, ef)
    ee = ef + (e - ef) * relief - 0.35 * ao * (0.5 + 0.5 * relief)
    col = gradmap(ee)
    # shadow side: sky fill on up-facing, bounce (warmer lavender) on down-facing shadow
    shw = _ss(0.05, -0.25, ee)[..., None]
    up = np.clip(-ny * 1.6, 0, 1)[..., None]
    down = np.clip(ny * 1.4, 0, 1)[..., None]
    col = col + (np.asarray(sky_fill, np.float32) - col) * up * shw * 0.35
    col = col + (np.array([0.62, 0.62, 0.8], np.float32) - col) * down * shw * 0.25
    # a touch of cool in the crevices of the light side (lobe separation, painted)
    lw = _ss(0.0, 0.3, ee)[..., None]
    col = col + (np.array([0.78, 0.8, 0.95], np.float32) - col) * (np.clip(ao * 1.5, 0, 1)[..., None] * lw * 0.45)
    # per-lobe cauliflower modelling: each lobe's own lower-left turns lavender inside the light,
    # each lobe's upper part catches sky fill inside the shadow
    lobe_sh = (_ss(0.3, -0.35, e0) * 0.55 + _ss(0.2, -0.5, e1) * 0.3) * relief
    col = col + (np.array([0.70, 0.73, 0.89], np.float32) - col) * (np.clip(lobe_sh, 0, 1)[..., None] * lw * 0.75)
    up0 = np.clip(-ny0 * 1.3, 0, 1)[..., None]
    col = col + (np.array([0.68, 0.75, 0.95], np.float32) - col) * up0 * shw * 0.4
    # hot broad crown where the big form faces the light
    hot = _ss(0.45, 0.85, ef)[..., None] * lw
    col = col + (np.array([1.06, 1.04, 0.99], np.float32) - col) * hot * 0.5
    # --- silver lining / translucent fringe / halo near the sun
    sx, sy = sun[0] - xs, sun[1] - ys
    sl = np.sqrt(sx * sx + sy * sy) + 1e-3
    lx2, ly2 = Lv[0], Lv[1]
    l2 = math.hypot(lx2, ly2)
    prox = np.exp(-(sl / Wref) / 0.16)
    wdir = np.clip(prox * 1.4, 0, 1)
    ex = (sx / sl) * wdir + (lx2 / l2) * (1 - wdir)
    ey = (sy / sl) * wdir + (ly2 / l2) * (1 - wdir)
    el = np.sqrt(ex * ex + ey * ey) + 1e-6
    ex, ey = ex / el, ey / el
    cg = cv2.GaussianBlur(alpha, (0, 0), max(0.0015 * Wref, 0.7))
    ogx = -cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx ** 2 + ogy ** 2) + 1e-6
    face = np.clip((ogx * ex + ogy * ey) / ogl, 0, 1) * (ogl > 1e-4)
    face = _ss(0.0, 0.6, cv2.GaussianBlur(face.astype(np.float32), (0, 0), max(0.002 * Wref, 0.8)))
    din = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    rw = max(0.0018 * Wref, 1.0)
    strength = (0.45 + 1.4 * prox) * rim
    rim_l = np.clip(np.exp(-(din / rw) ** 1.5) * face * strength, 0, 1)
    rimc = np.array([1.45, 1.36, 1.15], np.float32)
    fringe = np.exp(-din / (0.014 * Wref)) * face * np.exp(-(sl / Wref) / 0.1) * 0.9
    col = col + (rimc * 0.95 - col) * np.clip(fringe, 0, 0.85)[..., None]
    col = col + (rimc - col) * rim_l[..., None]
    # aerial perspective toward the base + torn wisps
    if haze is not None:
        f = np.clip((ys - haze[0]) / (haze[1] - haze[0]), 0, 1)
        col = col + (np.asarray(haze_col, np.float32) - col) * (f ** 1.3 * haze[2])[..., None]
    # halo (glow just outside the silhouette near the sun)
    dout = cv2.distanceTransform((alpha < 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    hal = np.exp(-dout / (0.01 * Wref)) * cv2.GaussianBlur(face.astype(np.float32), (0, 0), 0.006 * Wref) * \
        np.exp(-(sl / Wref) / 0.14) * 0.55 * rim * (1 - alpha)
    a2 = alpha + hal
    col = (col * alpha[..., None] + rimc * hal[..., None]) / np.maximum(a2, 1e-5)[..., None]
    rgba = np.dstack([col, np.clip(a2, 0, 1)]).astype(np.float32)
    return rgba, dict(alpha=alpha, e=e, ao=ao, anvil=anv)


# =============================================================================== the s01 tower

def build(W, Hs, x0, base_y, top_y, seed=1, detail=1.0, scale=1.0):
    """Colossal cumulonimbus: a narrow, slightly leaning column of stacked billows with a flattened,
    right-sheared anvil + an overshooting dome, lower sister towers left/right and wide base shoulders.
    W = screen width (size scale), x0 = column x at the base, base_y/top_y plate px.
    Returns (lobes, info)."""
    rng = np.random.default_rng(seed)
    D = Lobes()
    Hc = base_y - top_y
    lev0 = []
    Wf = W
    W = W * scale
    v_col = 0.7                          # column top (the head takes over)

    def axis(v):
        return x0 + 0.07 * W * v ** 1.6

    def hw(v):
        return W * np.interp(v, [0, 0.1, 0.22, 0.35, 0.45, 0.55, 0.63, 0.7], [0.23, 0.18, 0.155, 0.175, 0.15, 0.158, 0.172, 0.18])
    v = 0.0
    while v < v_col:
        h_ = hw(v)
        Rr = h_ * rng.uniform(0.5, 0.62)
        n = max(2, int(round(2 * h_ / (Rr * 1.15))))
        for k in range(n):
            u = -1 + (2 * k + 1) / n + rng.uniform(-0.3, 0.3) / n
            x = axis(v) + u * max(h_ - Rr * 0.7, 0) + rng.uniform(-0.25, 0.25) * Rr
            y = base_y - v * Hc + rng.uniform(-0.1, 0.1) * Rr
            z = math.sqrt(max(1 - u * u, 0)) * h_ * 0.55 + rng.uniform(-0.12, 0.12) * Rr
            lev0.append(D.add(x, y, z, Rr * rng.uniform(0.88, 1.12), 0.9, 0, -1, 0, 0, rng))
        v += Rr * rng.uniform(0.62, 0.78) / Hc
    ax = axis(1.0)
    # cauliflower head: wider than the column, flattening and shearing right as it rises
    crown = []
    for (vv, hwh, rf, ay, sh) in [(0.71, 0.2, 0.085, 0.8, 0.015), (0.78, 0.225, 0.09, 0.74, 0.03),
                                  (0.85, 0.235, 0.085, 0.64, 0.05), (0.9, 0.2, 0.075, 0.55, 0.07)]:
        n = 5
        for k in range(n):
            u = -1 + (2 * k + 1) / n + rng.uniform(-0.12, 0.12) / n
            rr = rf * W * rng.uniform(0.9, 1.1)
            x = axis(vv) + sh * W + u * (hwh * W - rr * 0.8)
            y = base_y - vv * Hc + rng.uniform(-0.04, 0.04) * rr
            z = math.sqrt(max(1 - u * u, 0)) * hwh * W * 0.5 + 0.02 * W
            crown.append(D.add(x, y, z, rr, ay, 0, -1, 0, 0, rng))
    # overshooting dome (highest point, hides the sun at its right shoulder)
    rr = 0.06 * W
    crown.append(D.add(ax + 0.06 * W, base_y - 0.985 * Hc + rr * 0.75, 0.14 * W, rr, 0.75, 0, -1, 1, 0, rng))
    # flattened anvil shelf capping the head, sheared to the right and thinning
    anvil = []
    for (du, dv, rf, ay, zf) in [(-0.15, 0.93, 0.065, 0.45, 0.1), (-0.07, 0.95, 0.08, 0.42, 0.13),
                                 (0.03, 0.96, 0.09, 0.4, 0.15), (0.12, 0.96, 0.085, 0.4, 0.14),
                                 (0.2, 0.953, 0.075, 0.4, 0.12), (0.26, 0.944, 0.06, 0.42, 0.1)]:
        rr = rf * W
        y = base_y - dv * Hc + rr * ay
        anvil.append(D.add(ax + du * W, y, zf * W, rr, ay, 0, -1, 1, 0, rng))
    # sister towers and shoulders (behind, tid 1..)
    sisters = [(x0 - 0.4 * W, 0.34, 0.16 * W, -0.05), (x0 + 0.42 * W, 0.24, 0.13 * W, -0.06),
               (x0 - 0.66 * W, 0.2, 0.12 * W, -0.1), (x0 + 0.7 * W, 0.16, 0.11 * W, -0.1),
               (x0 - 0.17 * W, 0.13, 0.12 * W, 0.0), (x0 + 0.19 * W, 0.11, 0.12 * W, 0.0)]
    for ti, (cx, top_v, h_, zf) in enumerate(sisters, 2):
        v = 0.0
        while v < top_v:
            f = v / top_v
            Rr = h_ * (1 - 0.35 * f) * rng.uniform(0.85, 1.05) * 0.8
            n = 2 if h_ < 0.15 * W else 3
            for k in range(n):
                u = -1 + (2 * k + 1) / n
                x = cx + u * h_ * 0.65 * (1 - 0.35 * f) + rng.uniform(-0.1, 0.1) * Rr
                y = base_y - v * Hc
                if y - Rr * 0.9 < base_y - top_v * Hc:
                    y = base_y - top_v * Hc + Rr * 0.9
                z = zf * W + math.sqrt(max(1 - u * u, 0)) * h_ * 0.4
                lev0.append(D.add(x, y, z, Rr, 0.88, 0, -1, 0, ti, rng))
            v += Rr * 0.75 / Hc
    minr = max(0.0045 * Wf, 1.0)
    k1 = grow(D, rng, lev0 + crown, 5 * detail, 0.54, up_bias=1.2, minr=minr, clip_y=base_y, rvar=(0.7, 1.2))
    k1 += grow(D, rng, lev0, 4 * detail, 0.36, up_bias=2.4, front=(0.2, 0.9), minr=minr, clip_y=base_y)
    k2 = grow(D, rng, k1, 5 * detail, 0.42, up_bias=0.9, max_uy=0.2, front=(-0.3, 0.42), minr=minr, clip_y=base_y)
    grow(D, rng, k2, 4 * detail, 0.45, up_bias=0.9, front=(-0.3, 0.25), max_uy=0.0, minr=minr, clip_y=base_y)
    # anvil: few, flat, smooth (glaciated) bumps mostly along the upper edge + underside lumps
    a1 = grow(D, rng, anvil, 4 * detail, 0.36, up_bias=0.6, minr=minr, clip_y=base_y, ay_k=(0.75, 0.9),
              rvar=(0.8, 1.1), reach=(0.2, 0.45))
    grow(D, rng, anvil, 3 * detail, 0.33, up_bias=0.7, up_dir=math.pi / 2, front=(0.1, 0.8), minr=minr,
         clip_y=base_y, reach=(0.15, 0.4))
    grow(D, rng, a1, 3 * detail, 0.4, up_bias=0.8, max_uy=0.0, front=(-0.3, 0.3), minr=minr, clip_y=base_y,
         reach=(0.15, 0.35))
    info = dict(axis_top=ax, crown=crown, anvil=anvil)
    return D, info
