"""Helpers for s08_snow_station: pinhole camera, local polygon painting into layered canvases,
snow-laden pine painter, numba snow-flake splatter and volumetric lamp in-scattering."""
import math
import numpy as np
import cv2
from numba import njit, prange

from lib import core as C


# ----------------------------------------------------------------------------- camera

class Cam:
    """One-point perspective pinhole camera at the origin looking down +z (y up).
    Screen: u = cx + f*X/Z, v = cy - f*Y/Z (plate pixels)."""

    def __init__(self, f, cx, cy, ox=0.0):
        self.f, self.cx, self.cy, self.ox = float(f), float(cx), float(cy), float(ox)

    def p(self, X, Y, Z):
        Z = np.maximum(np.asarray(Z, np.float64), 1e-3)
        return self.cx + self.f * (np.asarray(X) - self.ox) / Z, self.cy - self.f * np.asarray(Y) / Z

    def pts(self, P):
        P = np.asarray(P, np.float64)
        u, v = self.p(P[:, 0], P[:, 1], P[:, 2])
        return np.stack([u, v], 1)


# ----------------------------------------------------------------------------- canvases

class Canvas:
    """RGBA layer + depth buffer. Painting is back-to-front 'over'; depth records the nearest
    painted surface (where coverage > 0.5)."""

    def __init__(self, W, H, rgb=None):
        self.W, self.H = W, H
        self.rgb = np.zeros((H, W, 3), np.float32) if rgb is None else rgb.astype(np.float32).copy()
        self.a = np.zeros((H, W), np.float32) if rgb is None else np.ones((H, W), np.float32)
        self.z = np.full((H, W), 1e4, np.float32)

    ztest = False

    def paint(self, x0, y0, alpha, col, z=None):
        h, w = alpha.shape
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, self.W), min(y0 + h, self.H)
        if X1 <= X0 or Y1 <= Y0:
            return
        a = alpha[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        if self.ztest and z is not None:
            zv = z[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] if isinstance(z, np.ndarray) else z
            occ = (self.z[Y0:Y1, X0:X1] < zv - (0.3 + 0.02 * zv)).astype(np.float32)
            if occ.any():
                occ = cv2.GaussianBlur(occ, (0, 0), 0.6)
                a = a * (1 - occ)
        if isinstance(col, np.ndarray) and col.ndim == 3:
            c = col[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
        else:
            c = np.asarray(col, np.float32)
        dst = self.rgb[Y0:Y1, X0:X1]
        self.rgb[Y0:Y1, X0:X1] = dst * (1 - a[..., None]) + c * a[..., None]
        self.a[Y0:Y1, X0:X1] = self.a[Y0:Y1, X0:X1] * (1 - a) + a
        if z is not None:
            zz = self.z[Y0:Y1, X0:X1]
            zv = z[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] if isinstance(z, np.ndarray) else z
            self.z[Y0:Y1, X0:X1] = np.where(a > 0.5, zv, zz)

    def paint_full(self, alpha, col, z=None):
        self.paint(0, 0, alpha, col, z)

    def rgba(self):
        return np.dstack([self.rgb, self.a]).astype(np.float32)


def poly_local(polys, ss=3, pad=2):
    """Anti-aliased union mask of polygons in a tight local box. Returns (x0, y0, mask)."""
    if isinstance(polys, np.ndarray) and polys.ndim == 2:
        polys = [polys]
    allp = np.concatenate([np.asarray(p, np.float64) for p in polys], 0)
    x0 = int(math.floor(allp[:, 0].min())) - pad
    y0 = int(math.floor(allp[:, 1].min())) - pad
    x1 = int(math.ceil(allp[:, 0].max())) + pad
    y1 = int(math.ceil(allp[:, 1].max())) + pad
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    if w * h > 6_000_000:
        raise ValueError('poly too big')
    m = np.zeros((h * ss, w * ss), np.uint8)
    for p in polys:
        q = ((np.asarray(p, np.float64) - [x0, y0]) * ss).astype(np.int32)
        cv2.fillPoly(m, [q], 255, lineType=cv2.LINE_AA)
    m = m.astype(np.float32) / 255.0
    if ss > 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return x0, y0, m


def line_local(pts, width, ss=4, pad=3):
    pts = np.asarray(pts, np.float64)
    x0 = int(math.floor(pts[:, 0].min() - width)) - pad
    y0 = int(math.floor(pts[:, 1].min() - width)) - pad
    x1 = int(math.ceil(pts[:, 0].max() + width)) + pad
    y1 = int(math.ceil(pts[:, 1].max() + width)) + pad
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    m = np.zeros((h * ss, w * ss), np.uint8)
    q = ((pts - [x0, y0]) * ss).astype(np.int32)
    cv2.polylines(m, [q], False, 255, max(1, int(round(width * ss))), cv2.LINE_AA)
    m = cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    return x0, y0, m


# ----------------------------------------------------------------------------- snow pine

def pine(cv, bu, bv, hpx, z, seed, fol, snow_top, snow_shd, warm=None, warm_side=0.0, warm_amt=0.0,
         fog=None, fog_amt=0.0, width=0.24, density=1.0, snow=1.0, ss=3, rim=None):
    """Paint a snow-laden conifer standing at screen (bu, bv) with pixel height hpx into canvas cv.
    The crown is built from many individual drooping branch clumps scattered inside a conical envelope,
    painted bottom-to-top so upper branches' dark needle fringes overlap the snow pillows below. Snow
    pillows are cel-shaded (shaded body / lit top / thin bright rim). warm: lamp colour tinting the side
    facing a light (warm_side -1 = left, +1 = right)."""
    rng = np.random.default_rng(seed)
    if hpx < 3:
        return
    hw0 = hpx * width * rng.uniform(0.85, 1.15)
    x0 = int(bu - hw0 * 1.45) - 3
    y0 = int(bv - hpx * 1.06) - 3
    w = int(hw0 * 2.9) + 6
    h = int(hpx * 1.1) + 6
    S = ss
    Hh, Ww = h * S, w * S
    col = np.zeros((Hh, Ww, 3), np.float32)
    alp = np.zeros((Hh, Ww), np.float32)
    snowm = np.zeros((Hh, Ww), np.float32)
    cx = (bu - x0) * S
    top = (bv - hpx - y0) * S
    H = hpx * S
    hw = hw0 * S
    fol = np.asarray(fol, np.float32)
    s_top = np.asarray(snow_top, np.float32)
    s_shd = np.asarray(snow_shd, np.float32)
    rimc = s_top * 1.22 if rim is None else np.asarray(rim, np.float32)

    def fill(poly, c, sn):
        p = np.round(poly).astype(np.int32)
        bx0, by0 = max(p[:, 0].min(), 0), max(p[:, 1].min(), 0)
        bx1, by1 = min(p[:, 0].max() + 1, Ww), min(p[:, 1].max() + 1, Hh)
        if bx1 <= bx0 or by1 <= by0:
            return
        m = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        cv2.fillPoly(m, [p - [bx0, by0]], 1)
        mb = m.astype(bool)
        col[by0:by1, bx0:bx1][mb] = c
        alp[by0:by1, bx0:bx1][mb] = 1.0
        snowm[by0:by1, bx0:bx1][mb] = sn

    ph1, ph2 = rng.uniform(0, 6.28, 2)

    def env(r):
        return hw * (0.05 + 0.95 * r ** 0.9) * (1 + 0.08 * math.sin(r * 11 + ph1) + 0.05 * math.sin(r * 23 + ph2))

    tw = max(hpx * 0.016, 0.5) * S
    fill(np.array([[cx - tw, top + H * 0.3], [cx + tw, top + H * 0.3], [cx + tw * 1.5, top + H],
                   [cx - tw * 1.5, top + H]]), fol * 0.75, 0.0)
    lean = rng.uniform(-0.02, 0.02)

    def branch(xs0, y, side, L, th, front=False):
        m = 28
        t = np.linspace(0, 1, m)
        xs = xs0 + side * L * t
        droop = th * rng.uniform(0.6, 1.4)
        arch = th * rng.uniform(0.0, 0.3)
        ytop = y + droop * t ** 1.6 - arch * np.sin(np.pi * t * 0.8) + rng.normal(0, th * 0.03, m)
        nt = int(rng.integers(3, 8))
        teeth = np.abs(((t * nt + rng.uniform(0, 1)) % 1.0) - 0.5) * 2
        ybot = ytop + th * (0.5 + 0.5 * (1 - t) ** 0.6) + th * 0.4 * teeth ** 1.6 * (0.3 + 0.7 * t)
        ybot[-1] = ytop[-1] + th * 0.15
        sh = rng.uniform(0.85, 1.15) * (1.1 if front else 1.0)
        fill(np.concatenate([np.stack([xs, ytop], 1), np.stack([xs[::-1], ybot[::-1]], 1)]), fol * sh, 0.0)
        fr = np.concatenate([np.stack([xs, ybot - th * 0.18], 1), np.stack([xs[::-1], ybot[::-1]], 1)])
        fill(fr, fol * (sh + 0.18) + s_shd * 0.05, 0.0)
        if snow > 0 and rng.random() < 0.96:
            t0 = rng.uniform(0.0, 0.08)
            t1 = rng.uniform(0.6, 0.97)
            sel = (t >= t0) & (t <= t1)
            if sel.sum() < 4:
                return
            u = (t - t0) / (t1 - t0)
            prof = np.clip(np.sin(np.pi * np.clip(u * 0.5 + 0.5 * (u > 0.5) * 0 + 0.5 * u, 0, 1)), 0, 1)
            prof = np.clip(np.minimum(u / 0.08 + 0.3, 1) * np.minimum((1 - u) / 0.25, 1), 0, 1) ** 0.5
            nl = int(rng.integers(1, 4))
            lump = 0.75 + 0.25 * np.abs(np.sin(np.pi * u * nl + rng.uniform(0, 6)))
            sth = th * snow * rng.uniform(0.6, 1.0) * prof
            ys_top = ytop - sth * 0.6 * lump
            ys_bot = ytop + sth * 0.5
            ys_mid = ys_top + (ys_bot - ys_top) * rng.uniform(0.4, 0.62)
            xs_s = xs[sel]
            fill(np.concatenate([np.stack([xs_s, ys_top[sel]], 1), np.stack([xs_s[::-1], ys_bot[sel][::-1]], 1)]),
                 s_shd * rng.uniform(0.92, 1.08), 1.0)
            fill(np.concatenate([np.stack([xs_s, ys_top[sel]], 1), np.stack([xs_s[::-1], ys_mid[sel][::-1]], 1)]),
                 s_top * rng.uniform(0.94, 1.04), 1.0)
            rt = ys_top + np.maximum(sth * 0.13, S * 0.7)
            fill(np.concatenate([np.stack([xs_s, ys_top[sel]], 1), np.stack([xs_s[::-1], rt[sel][::-1]], 1)]), rimc, 1.0)

    def tier(xc_, y, eL, eR, th, droop, snow_amt):
        m = 48
        t = np.linspace(-1, 1, m)
        at = np.abs(t)
        xs = xc_ + np.where(t < 0, t * eL, t * eR)
        dome = th * rng.uniform(0.15, 0.45)
        ytop = y + droop * at ** 1.8 - dome * (1 - at ** 2) + rng.normal(0, th * 0.03, m)
        nt = int(rng.integers(4, 9))
        teeth = np.abs(((at * nt + rng.uniform(0, 1)) % 1.0) - 0.5) * 2
        ybot = ytop + th * (0.45 + 0.4 * (1 - at) ** 0.7) + th * 0.42 * teeth ** 1.5 * (0.35 + 0.65 * at)
        sh = rng.uniform(0.85, 1.15)
        fill(np.concatenate([np.stack([xs, ytop], 1), np.stack([xs[::-1], ybot[::-1]], 1)]), fol * sh, 0.0)
        fr = np.concatenate([np.stack([xs, ybot - th * 0.16], 1), np.stack([xs[::-1], ybot[::-1]], 1)])
        fill(fr, fol * (sh + 0.22) + s_shd * 0.06, 0.0)
        # drooping tip twigs poking out below the tier ends
        for sd, e_ in ((-1, eL), (1, eR)):
            if rng.random() < 0.6:
                tx = xc_ + sd * e_ * rng.uniform(0.85, 1.02)
                ty = y + droop * 0.9
                ln = th * rng.uniform(0.5, 1.1)
                fill(np.array([[tx - sd * th * 0.5, ty], [tx + sd * th * 0.2, ty + ln], [tx - sd * th * 0.1, ty + ln * 0.4]]),
                     fol * sh, 0.0)
        if snow_amt <= 0:
            return
        tl, tr = rng.uniform(0.7, 0.96), rng.uniform(0.7, 0.96)
        sel = (t >= -tl) & (t <= tr)
        u = np.where(t < 0, -t / tl, t / tr)
        prof = np.clip(1 - np.clip(u, 0, 1) ** 2.2, 0, 1) ** 0.55
        nl = int(rng.integers(2, 5))
        lump = 0.8 + 0.2 * np.abs(np.sin(np.pi * t * nl + rng.uniform(0, 6)))
        sth = th * snow_amt * rng.uniform(0.75, 1.1) * prof
        ys_top = ytop - sth * 0.7 * lump
        blo = 0.75 + 0.25 * np.abs(np.sin(np.pi * t * (nl + 2) + rng.uniform(0, 6))) ** 0.5
        ys_bot = ytop + sth * 0.4 * blo
        ys_mid = ys_top + (ys_bot - ys_top) * (0.45 + 0.15 * np.sin(t * 3 + rng.uniform(0, 6)))
        xs_s = xs[sel]
        fill(np.concatenate([np.stack([xs_s, ys_top[sel]], 1), np.stack([xs_s[::-1], ys_bot[sel][::-1]], 1)]),
             s_shd * rng.uniform(0.93, 1.07), 1.0)
        fill(np.concatenate([np.stack([xs_s, ys_top[sel]], 1), np.stack([xs_s[::-1], ys_mid[sel][::-1]], 1)]),
             s_top * rng.uniform(0.95, 1.03), 1.0)
        rt = ys_top + np.maximum(sth * 0.12, S * 0.7)
        fill(np.concatenate([np.stack([xs_s, ys_top[sel]], 1), np.stack([xs_s[::-1], rt[sel][::-1]], 1)]), rimc, 1.0)

    n = int(np.clip(5 + hpx / 26, 5, 18) * rng.uniform(0.9, 1.12))
    rk = np.cumsum(rng.uniform(0.7, 1.3, n))
    rk = 0.06 + 0.84 * (rk - rk[0]) / (rk[-1] - rk[0])
    for k in range(n - 1, -1, -1):
        r = rk[k]
        e = env(r)
        th = max(H * 0.85 / n * rng.uniform(0.8, 1.15), S * 1.5)
        y = top + r * H
        xc_ = cx + lean * H * (1 - r)
        tier(xc_, y, e * rng.uniform(0.8, 1.08), e * rng.uniform(0.8, 1.08), th, th * rng.uniform(0.7, 1.4),
             snow * (1.0 if rng.random() > 0.08 else 0.4))
        # a smaller secondary clump in front on some tiers
        if r > 0.3 and rng.random() < 0.45:
            sd = rng.choice([-1.0, 1.0])
            tier(xc_ + sd * e * rng.uniform(0.25, 0.5), y + th * rng.uniform(0.2, 0.5), e * rng.uniform(0.25, 0.4),
                 e * rng.uniform(0.25, 0.4), th * 0.7, th * 0.6, snow)
    tipw = hw * 0.04
    fill(np.array([[cx - tipw, top + H * 0.07], [cx, top - H * 0.015], [cx + tipw, top + H * 0.07]]), fol, 0.0)
    fill(np.array([[cx - tipw * 1.3, top + H * 0.075], [cx - tipw * 0.3, top + H * 0.035], [cx + tipw * 0.4, top + H * 0.03],
                   [cx + tipw * 1.3, top + H * 0.075]]), s_top, 1.0)
    colL = cv2.resize(col, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(alp, (w, h), interpolation=cv2.INTER_AREA)
    sm = cv2.resize(snowm, (w, h), interpolation=cv2.INTER_AREA)
    colL = colL / np.maximum(a, 1e-4)[..., None]
    yy = np.clip((np.arange(h, dtype=np.float32)[:, None] - (bv - hpx - y0)) / max(hpx, 1), 0, 1)
    colL = colL * (1 + (0.12 - 0.28 * yy) * sm)[..., None]
    if warm is not None and warm_amt > 0:
        xx = (np.arange(w, dtype=np.float32)[None, :] - (bu - x0)) / max(hw0, 1)
        side = np.clip(0.5 + 0.6 * xx * warm_side, 0, 1) ** 1.6
        wk = (side * warm_amt)[..., None] * np.asarray(warm, np.float32)
        colL = colL + wk * (0.15 + 0.85 * sm[..., None])
    if fog is not None and fog_amt > 0:
        colL = colL * (1 - fog_amt) + np.asarray(fog, np.float32) * fog_amt
    cv.paint(x0, y0, np.clip(a, 0, 1), colL.astype(np.float32), z)


# ----------------------------------------------------------------------------- snow flakes

@njit(cache=True, fastmath=True)
def splat_flakes(img, xs, ys, vx, vy, rad, cols, alph, kind):
    """Over-composite flakes: kind 0 = soft gaussian streak (motion blurred along (vx, vy)),
    kind 1 = defocused bokeh disc (flat with brighter rim)."""
    H, W = img.shape[0], img.shape[1]
    for i in range(xs.shape[0]):
        r = rad[i]
        ext = r * (1.6 if kind[i] == 0 else 1.15) + 1.5
        hx = abs(vx[i]) * 0.5
        hy = abs(vy[i]) * 0.5
        x0 = int(xs[i] - ext - hx)
        x1 = int(xs[i] + ext + hx) + 2
        y0 = int(ys[i] - ext - hy)
        y1 = int(ys[i] + ext + hy) + 2
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        if x1 > W:
            x1 = W
        if y1 > H:
            y1 = H
        L2 = vx[i] * vx[i] + vy[i] * vy[i]
        ax = xs[i] - vx[i] * 0.5
        ay = ys[i] - vy[i] * 0.5
        inv_r = 1.0 / max(r, 0.3)
        a0 = alph[i]
        c0, c1, c2 = cols[i, 0], cols[i, 1], cols[i, 2]
        for py in range(y0, y1):
            for px in range(x0, x1):
                dx = px + 0.5 - ax
                dy = py + 0.5 - ay
                if L2 > 1e-6:
                    tt = (dx * vx[i] + dy * vy[i]) / L2
                    if tt < 0.0:
                        tt = 0.0
                    elif tt > 1.0:
                        tt = 1.0
                    dx -= tt * vx[i]
                    dy -= tt * vy[i]
                d = math.sqrt(dx * dx + dy * dy) * inv_r
                if kind[i] == 0:
                    cov = math.exp(-2.2 * d * d)
                elif kind[i] == 2:
                    e = (1.0 - d) / 0.45
                    if e <= 0.0:
                        continue
                    if e > 1.0:
                        e = 1.0
                    cov = e * e * (3.0 - 2.0 * e) * (0.85 + 0.15 * d * d)
                else:
                    e = (1.0 - d) * r * 0.9 + 0.5
                    if e <= 0.0:
                        continue
                    cov = min(e, 1.0) * (0.72 + 0.28 * min(max((d - 0.55) / 0.45, 0.0), 1.0) ** 2)
                a = cov * a0
                if a < 0.002:
                    continue
                if a > 1.0:
                    a = 1.0
                img[py, px, 0] = img[py, px, 0] * (1.0 - a) + c0 * a
                img[py, px, 1] = img[py, px, 1] * (1.0 - a) + c1 * a
                img[py, px, 2] = img[py, px, 2] * (1.0 - a) + c2 * a


# ----------------------------------------------------------------------------- volumetric lamps

@njit(cache=True, fastmath=True, parallel=True)
def inscatter(dx, dy, depth, lp, lcol, lcone, lpow, zmax, ext, nsamp):
    """Single-scattering airlight from point lamps along each view ray (camera at origin), integrated
    numerically over a window around the closest approach, with extinction exp(-d/ext) around each lamp
    (snowy air) and a downward cone for shaded lamps. dx, dy: ray slopes X/Z, Y/Z; depth: scene Z.
    lp (L,3) lamp positions (camera space); lcol (L,3) colour*intensity; lcone (L,) 0..1 cone amount;
    lpow (L,) cone sharpness. Returns (H, W, 3)."""
    H, W = dx.shape
    out = np.zeros((H, W, 3), np.float32)
    nl = lp.shape[0]
    for i in prange(H):
        for j in range(W):
            rx, ry = dx[i, j], dy[i, j]
            rn = math.sqrt(rx * rx + ry * ry + 1.0)
            ux, uy, uz = rx / rn, ry / rn, 1.0 / rn
            T = min(depth[i, j], zmax) * rn
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            for k in range(nl):
                px, py, pz = lp[k, 0], lp[k, 1], lp[k, 2]
                s0 = px * ux + py * uy + pz * uz
                qx, qy, qz = px - s0 * ux, py - s0 * uy, pz - s0 * uz
                h2 = qx * qx + qy * qy + qz * qz
                if h2 > (9.0 * ext) ** 2:
                    continue
                sa = max(s0 - 8.0 * ext, 0.0)
                sb = min(s0 + 8.0 * ext, T)
                if sb <= sa:
                    continue
                # equiangular sampling around the closest approach
                he = math.sqrt(h2) + 0.05
                ta = math.atan((sa - s0) / he)
                tb = math.atan((sb - s0) / he)
                dth = (tb - ta) / nsamp
                acc = 0.0
                for m in range(nsamp):
                    th = ta + (m + 0.5) * dth
                    ss = s0 + he * math.tan(th)
                    ds = he / (math.cos(th) ** 2) * dth
                    wx = ss * ux - px
                    wy = ss * uy - py
                    wz = ss * uz - pz
                    d2 = wx * wx + wy * wy + wz * wz
                    d = math.sqrt(d2) + 1e-4
                    v = math.exp(-d / ext) / (d2 + 0.12)
                    if lcone[k] > 0.0:
                        cd = -wy / d
                        c = min(max((cd - 0.3) / 0.55, 0.0), 1.0)
                        v *= (1.0 - lcone[k]) + lcone[k] * c ** lpow[k]
                    acc += v * ds
                a0 += acc * lcol[k, 0]
                a1 += acc * lcol[k, 1]
                a2 += acc * lcol[k, 2]
            out[i, j, 0] = a0
            out[i, j, 1] = a1
            out[i, j, 2] = a2
    return out


# ----------------------------------------------------------------------------- text sign

def sign_texture(w, h, font_dir='C:/Windows/Fonts'):
    """Japanese station name board (白雪 / しらゆき) as an RGB float texture (h, w, 3)."""
    from PIL import Image, ImageDraw, ImageFont
    S = 2
    im = Image.new('RGB', (w * S, h * S), (244, 245, 246))
    d = ImageDraw.Draw(im)

    def font(name, size):
        for n in (name, 'YuGothB.ttc', 'msgothic.ttc'):
            try:
                return ImageFont.truetype(f'{font_dir}/{n}', size)
            except OSError:
                continue
        return ImageFont.load_default()

    W, H = w * S, h * S
    # colour band (line colour) under the name
    d.rectangle([0, int(H * 0.62), W, int(H * 0.71)], fill=(22, 96, 196))
    f1 = font('YuGothB.ttc', int(H * 0.36))
    f2 = font('YuGothM.ttc', int(H * 0.13))
    f3 = font('YuGothM.ttc', int(H * 0.11))
    f4 = font('YuGothB.ttc', int(H * 0.12))

    def ctext(txt, y, f, fill, x=None, anchor='mt'):
        d.text((W / 2 if x is None else x, y), txt, font=f, fill=fill, anchor=anchor)

    ctext('しらゆき', H * 0.17, font('YuGothB.ttc', int(H * 0.32)), (12, 14, 18))
    ctext('白 雪', H * 0.47, f4, (40, 44, 52), x=W * 0.2)
    ctext('Shirayuki', H * 0.49, f2, (60, 64, 72), x=W * 0.8)
    ctext('ふゆの', H * 0.76, f3, (30, 34, 40), x=W * 0.03, anchor='lt')
    ctext('はるの', H * 0.76, f3, (30, 34, 40), x=W * 0.97, anchor='rt')
    ctext('Fuyuno', H * 0.89, font('YuGothM.ttc', int(H * 0.075)), (70, 74, 80), x=W * 0.03, anchor='lt')
    ctext('Haruno', H * 0.89, font('YuGothM.ttc', int(H * 0.075)), (70, 74, 80), x=W * 0.97, anchor='rt')
    a = np.asarray(im).astype(np.float32) / 255.0
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------------------- bare tree with snow

def bare_tree(cv, bu, bv, hpx, z, seed, bark=(0.05, 0.05, 0.08), snow_col=(0.55, 0.65, 0.9), fog=None, fog_amt=0.0,
              spread=1.0, depth=8, ss=2, rim=None, rim_side=1.0):
    """A gnarled bare deciduous tree (cherry-like: short thick trunk, wide spreading limbs, dense fine
    twigs) with snow lying along the upper side of its branches. Painted into canvas cv at screen base
    (bu, bv) with pixel height hpx."""
    rng = np.random.default_rng(seed)
    segs = []

    def grow(x, y, ang, length, width, d):
        n = 3 if d > 2 else 2
        for i in range(n):
            ang += rng.normal(0, 0.16 if d > 1 else 0.25)
            # gentle pull toward horizontal for the big limbs (cherry habit), slight droop for twigs
            if d >= depth - 3:
                ang += (math.pi / 2 - ang) * 0.0 + (0.0 if abs(math.cos(ang)) > 0.2 else rng.normal(0, 0.05))
            nx = x + math.cos(ang) * length / n
            ny = y - math.sin(ang) * length / n
            wn = width * (1 - 0.18 / n)
            segs.append((x, y, nx, ny, width, wn, d))
            x, y, width = nx, ny, wn
        if d == 0 or width < 0.25:
            return
        nch = 2 if rng.random() < 0.6 else 3
        for c in range(nch):
            da = rng.uniform(0.25, 0.75) * (1 if c % 2 == 0 else -1) + rng.normal(0, 0.1)
            if c == 2:
                da = rng.normal(0, 0.15)
            grow(x, y, ang + da, length * rng.uniform(0.62, 0.8), width * rng.uniform(0.58, 0.75), d - 1)

    trunk_w = hpx * 0.06
    # trunk splits low into 3-4 main limbs spreading wide
    tx, ty = 0.0, 0.0
    segs.append((0.0, 0.0, rng.normal(0, 0.02) * hpx, -hpx * 0.22, trunk_w, trunk_w * 0.8, depth + 1))
    tx, ty = segs[-1][2], segs[-1][3]
    nl = 4
    for k in range(nl):
        ang = math.pi / 2 + (k - (nl - 1) / 2) * 0.55 * spread + rng.normal(0, 0.12)
        grow(tx, ty, ang, hpx * rng.uniform(0.3, 0.4), trunk_w * rng.uniform(0.45, 0.6), depth)
    P = np.array([[s_[0], s_[1]] for s_ in segs] + [[s_[2], s_[3]] for s_ in segs])
    x0 = int(bu + P[:, 0].min()) - 6
    y0 = int(bv + P[:, 1].min()) - 6
    x1 = int(bu + P[:, 0].max()) + 6
    y1 = int(bv + P[:, 1].max()) + 6
    w, h = x1 - x0, y1 - y0
    S = ss
    bm = np.zeros((h * S, w * S), np.uint8)
    sm = np.zeros((h * S, w * S), np.uint8)
    for (xa, ya, xb, yb, wa, wb, d) in segs:
        ax, ay = (bu + xa - x0) * S, (bv + ya - y0) * S
        bx, by = (bu + xb - x0) * S, (bv + yb - y0) * S
        dx, dy = bx - ax, by - ay
        L = math.hypot(dx, dy) + 1e-6
        nx, ny = -dy / L, dx / L
        wa_, wb_ = max(wa * S * 0.5, 0.5), max(wb * S * 0.5, 0.5)
        quad = np.array([[ax + nx * wa_, ay + ny * wa_], [bx + nx * wb_, by + ny * wb_],
                         [bx - nx * wb_, by - ny * wb_], [ax - nx * wa_, ay - ny * wa_]])
        cv2.fillConvexPoly(bm, np.round(quad).astype(np.int32), 255, cv2.LINE_AA)
        cv2.circle(bm, (int(bx), int(by)), max(int(wb_), 0), 255, -1, cv2.LINE_AA)
        # snow on the upper side of branches that are not too steep
        slope = abs(dy) / L
        if wa * S > 1.2 and slope < 0.8 and rng.random() < 0.85:
            up = -1 if ny < 0 else 1       # choose the normal pointing up (negative y)
            ox, oy = nx * up * (-1), ny * up * (-1)
            if oy > 0:
                ox, oy = -ox, -oy
            sw = max(wa_ * rng.uniform(0.5, 0.9) * (1 - slope * 0.8), 0.6)
            q2 = np.array([[ax + ox * (wa_ * 0.6), ay + oy * (wa_ * 0.6)], [bx + ox * (wb_ * 0.6), by + oy * (wb_ * 0.6)],
                           [bx + ox * (wb_ * 0.6 + sw), by + oy * (wb_ * 0.6 + sw)],
                           [ax + ox * (wa_ * 0.6 + sw), ay + oy * (wa_ * 0.6 + sw)]])
            cv2.fillConvexPoly(sm, np.round(q2).astype(np.int32), 255, cv2.LINE_AA)
    ba = cv2.resize(bm.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA)
    sa = cv2.resize(sm.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA)
    bark = np.asarray(bark, np.float32)
    sc = np.asarray(snow_col, np.float32)
    a = np.clip(np.maximum(ba, sa), 0, 1)
    col = bark * ba[..., None] * (1 - sa[..., None]) + sc * sa[..., None]
    col = col / np.maximum(a, 1e-4)[..., None]
    if fog is not None and fog_amt > 0:
        col = col * (1 - fog_amt) + np.asarray(fog, np.float32) * fog_amt
    cv.paint(x0, y0, a, col.astype(np.float32), z)


# ----------------------------------------------------------------------------- flake lighting

@njit(cache=True, fastmath=True, parallel=True)
def flake_light(P, cam, lp, lcol, lI, lcone, lpow, g):
    """Per-flake lamp irradiance E (N,3) and forward-scattered glow fs (N,3) (Henyey-Greenstein, g).
    P (N,3) world points, cam (3,) viewer position, lamps: lp (L,3), lcol (L,3), lI, lcone, lpow (L,)."""
    n = P.shape[0]
    nl = lp.shape[0]
    E = np.zeros((n, 3), np.float32)
    FS = np.zeros((n, 3), np.float32)
    for i in prange(n):
        px, py, pz = P[i, 0], P[i, 1], P[i, 2]
        vx, vy, vz = px - cam[0], py - cam[1], pz - cam[2]
        vn = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-6
        vx /= vn
        vy /= vn
        vz /= vn
        for k in range(nl):
            dx, dy, dz = lp[k, 0] - px, lp[k, 1] - py, lp[k, 2] - pz
            d2 = dx * dx + dy * dy + dz * dz
            dist = math.sqrt(d2) + 1e-4
            e = lI[k] / (d2 + 0.35)
            if lcone[k] > 0.0:
                cd = dy / dist
                c = min(max((cd - 0.3) / 0.55, 0.0), 1.0)
                e *= (1.0 - lcone[k]) + lcone[k] * c ** lpow[k]
            cth = (vx * dx + vy * dy + vz * dz) / dist
            hg = (1 - g * g) / (1 + g * g - 2 * g * cth) ** 1.5 / 12.0
            f = lI[k] * hg / (d2 + 0.5)
            for c3 in range(3):
                E[i, c3] += e * lcol[k, c3]
                FS[i, c3] += f * lcol[k, c3]
    return E, FS


@njit(cache=True, fastmath=True, parallel=True)
def ground_maps(du, dv, f, cx, cy, tx, ty, tz, Y0):
    """Backward remap coordinates of a horizontal plane (height Y0) for a camera translated by
    (tx, ty, tz): output pixel offsets (du, dv) from the frame centre -> source plate coords."""
    H, W = du.shape
    mu = np.empty((H, W), np.float32)
    mv = np.empty((H, W), np.float32)
    for i in prange(H):
        for j in range(W):
            d = dv[i, j]
            if d <= 0.5:
                mu[i, j] = -10.0
                mv[i, j] = -10.0
                continue
            Zn = f * (ty - Y0) / d
            Z = Zn + tz
            mu[i, j] = cx + (du[i, j] * Zn / f + tx) * (f / Z)
            mv[i, j] = cy - f * Y0 / Z
    return mu, mv
