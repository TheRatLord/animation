"""Painting helpers for s02_railway_crossing: painted foliage masses (dome z-buffer, cel-lit leaf clumps
with inter-clump cast shadows), a supersampled vector canvas for per-frame hard-edged structures,
stripe clipping for hazard-striped parts, and tileable world-space noise sampling."""
import math
import numpy as np
import cv2
from numba import njit, prange


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ============================================================================ tileable noise

def tile_noise(n=256, seed=0, octaves=5, base=4):
    """Tileable fbm (0..1) of size n x n built from periodic value-noise lattices."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    amp, tot = 1.0, 0.0
    cells = base
    for o in range(octaves):
        g = rng.random((cells, cells)).astype(np.float32)
        g = np.tile(g, (3, 3))
        big = cv2.resize(g, (3 * n, 3 * n), interpolation=cv2.INTER_CUBIC)
        out += amp * big[n:2 * n, n:2 * n]
        tot += amp
        amp *= 0.5
        cells *= 2
    out /= tot
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def sample_tile(tex, u, v):
    """Bilinear, wrapping sample of a tileable texture at float texel coords (u, v) arrays."""
    n = tex.shape[0]
    return cv2.remap(tex, np.mod(u, n).astype(np.float32), np.mod(v, n).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ============================================================================ polygons / stripes

def clip_halfplane(poly, a, b, c):
    """Clip polygon (list of (x,y)) to a*x + b*y + c >= 0."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp = a * p[0] + b * p[1] + c
        dq = a * q[0] + b * q[1] + c
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            tt = dp / (dp - dq)
            out.append((p[0] + (q[0] - p[0]) * tt, p[1] + (q[1] - p[1]) * tt))
    return out


def stripes_on_quad(quad, n, phase=0.0, slant=0.35):
    """Split a quad (p0, p1, p2, p3; long axis p0->p1, width p0->p3) into n alternating stripes slanted by
    `slant` (in units of the quad length per unit width). Returns [(poly_pts, k)] with k = 0/1."""
    p0, p1, p2, p3 = [np.asarray(p, np.float64) for p in quad]

    def m(u, s):
        return tuple((p0 * (1 - u) + p1 * u) * (1 - s) + (p3 * (1 - u) + p2 * u) * s)
    res = []
    for i in range(-3, n + 4):
        u0 = (i + phase) / n
        u1 = (i + 1 + phase) / n
        sl = slant / max(n, 1) * n
        poly = [(u0, 0.0), (u1, 0.0), (u1 + slant, 1.0), (u0 + slant, 1.0)]
        poly = clip_halfplane(poly, 1.0, 0.0, 0.0)
        if len(poly) < 3:
            continue
        poly = clip_halfplane(poly, -1.0, 0.0, 1.0)
        if len(poly) < 3:
            continue
        res.append(([m(u, s) for u, s in poly], i % 2))
    return res


# ============================================================================ vector canvas

class Canvas:
    """Supersampled RGBA uint8 canvas: draw hard-edged polygons/lines in 1x pixel coordinates, then
    downsample (box filter) to get clean anti-aliased premultiplied RGB + alpha."""

    def __init__(self, W, H, ss=3):
        self.W, self.H, self.ss = W, H, ss
        self.buf = np.zeros((H * ss, W * ss, 4), np.uint8)

    def clear(self):
        self.buf[:] = 0

    @staticmethod
    def _col(rgb, a=1.0):
        a = min(max(float(a), 0.0), 1.0)
        m = 255.0 * a
        return (int(min(max(float(rgb[0]), 0.0), 1.0) * m + 0.5), int(min(max(float(rgb[1]), 0.0), 1.0) * m + 0.5),
                int(min(max(float(rgb[2]), 0.0), 1.0) * m + 0.5), int(m + 0.5))

    def poly(self, pts, rgb, a=1.0):
        p = np.round(np.asarray(pts, np.float64) * self.ss * 16).astype(np.int32)
        cv2.fillPoly(self.buf, [p], self._col(rgb, a), lineType=cv2.LINE_8, shift=4)

    def polys(self, plist, rgb, a=1.0):
        if not plist:
            return
        ps = [np.round(np.asarray(pts, np.float64) * self.ss * 16).astype(np.int32) for pts in plist]
        cv2.fillPoly(self.buf, ps, self._col(rgb, a), lineType=cv2.LINE_8, shift=4)

    def line(self, pts, width, rgb, a=1.0):
        p = np.round(np.asarray(pts, np.float64) * self.ss * 16).astype(np.int32)
        cv2.polylines(self.buf, [p], False, self._col(rgb, a), max(1, int(round(width * self.ss))),
                      lineType=cv2.LINE_8, shift=4)

    def circle(self, c, r, rgb, a=1.0):
        cv2.circle(self.buf, (int(round(c[0] * self.ss * 16)), int(round(c[1] * self.ss * 16))),
                   max(1, int(round(r * self.ss * 16))), self._col(rgb, a), -1, cv2.LINE_8, shift=4)

    def ellipse(self, c, axes, ang, *args, a=1.0):
        if len(args) == 1:
            a0, a1, rgb = 0, 360, args[0]
        else:
            a0, a1, rgb = args[:3]
        cv2.ellipse(self.buf, (int(round(c[0] * self.ss * 16)), int(round(c[1] * self.ss * 16))),
                    (max(1, int(round(axes[0] * self.ss * 16))), max(1, int(round(axes[1] * self.ss * 16)))),
                    ang, a0, a1, self._col(rgb, a), -1, cv2.LINE_8, shift=4)

    def resolve(self, y0=0, y1=None):
        """Fast path: the empty canvas is (0,0,0,0) and shapes are drawn opaque, so a box-filtered
        downsample of the raw buffer is already premultiplied. -> rgb (h, W, 3), alpha (h, W)."""
        y1 = self.H if y1 is None else y1
        s = self.ss
        sub = self.buf[y0 * s:y1 * s]
        small = cv2.resize(sub, (self.W, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) * (1 / 255.0)
        return small[..., :3], small[..., 3]

    def resolve_premult(self, y0=0, y1=None):
        """Premultiply at supersampled resolution (exact AA of edges against transparency)."""
        y1 = self.H if y1 is None else y1
        s = self.ss
        sub = self.buf[y0 * s:y1 * s].astype(np.float32)
        a = sub[..., 3:4] * (1.0 / 255.0)
        pm = np.concatenate([sub[..., :3] * (a / 255.0), a], -1)
        small = cv2.resize(pm, (self.W, y1 - y0), interpolation=cv2.INTER_AREA)
        return small[..., :3], small[..., 3]


# ============================================================================ foliage masses

# sphere columns
F_CX, F_CY, F_R, F_Z, F_GX, F_GY, F_GR, F_TONE, F_PH, F_K, F_AMP, F_FOG, F_AY = range(13)
NF = 13


@njit(cache=True, fastmath=True)
def _fol_zbuf(Hs, Ws, S, ss):
    n = S.shape[0]
    hb = np.full((Hs, Ws), -1e9, np.float32)
    ib = np.full((Hs, Ws), -1, np.int32)
    for i in range(n):
        cx, cy, r = S[i, 0] * ss, S[i, 1] * ss, S[i, 2] * ss
        z0 = S[i, 3] * ss
        ph, k, amp, ay = S[i, 8], S[i, 9], S[i, 10], S[i, 12]
        R = r * (1.0 + amp) + 2.0
        ry = R * ay
        x0 = max(0, int(cx - R))
        x1 = min(Ws, int(cx + R) + 2)
        y0 = max(0, int(cy - ry))
        y1 = min(Hs, int(cy + ry) + 2)
        for y in range(y0, y1):
            dy = (y + 0.5 - cy) / ay
            for x in range(x0, x1):
                dx = x + 0.5 - cx
                d = math.sqrt(dx * dx + dy * dy)
                if d > R:
                    continue
                th = math.atan2(dy, dx)
                re = r * (1.0 + amp * (0.6 * math.sin(k * th + ph) + 0.4 * math.sin(2.13 * k * th + 1.7 * ph)))
                re = re * (0.9 + 0.1 * abs(math.sin(1.6 * k * th + 2.9 * ph)))
                if d >= re:
                    continue
                q = d / re
                h = z0 + r * math.sqrt(max(1.0 - q * q, 0.0))
                if h > hb[y, x]:
                    hb[y, x] = h
                    ib[y, x] = i
    return hb, ib


@njit(cache=True, fastmath=True, parallel=True)
def _fol_shade(hb, ib, S, ss, Lx, Ly, Lz, shift, wl, wg):
    Hs, Ws = hb.shape
    ndl = np.zeros((Hs, Ws), np.float32)
    sh = np.zeros((Hs, Ws), np.float32)
    rim = np.zeros((Hs, Ws), np.float32)
    for y in prange(Hs):
        for x in range(Ws):
            i = ib[y, x]
            if i < 0:
                continue
            cx, cy, r = S[i, 0] * ss, S[i, 1] * ss, S[i, 2] * ss
            ay = S[i, 12]
            dx = (x + 0.5 - cx) / r
            dy = (y + 0.5 - cy) / (r * ay)
            dd = dx * dx + dy * dy
            if dd > 0.98:
                s = 0.98 / dd
                dx *= math.sqrt(s)
                dy *= math.sqrt(s)
                dd = 0.98
            nz = math.sqrt(1.0 - dd)
            gx, gy, gr = S[i, 4] * ss, S[i, 5] * ss, S[i, 6] * ss
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
            # cast shadow from a clump in front, toward the light (screen-space march)
            h0 = hb[y, x]
            occ = 0.0
            for st in range(1, 6):
                d = shift * st / 5.0
                xx = int(x + Lx * d / math.sqrt(Lx * Lx + Ly * Ly + 1e-9))
                yy = int(y + Ly * d / math.sqrt(Lx * Lx + Ly * Ly + 1e-9))
                if xx < 0 or yy < 0 or xx >= Ws or yy >= Hs:
                    break
                j = ib[yy, xx]
                if j >= 0 and j != i and hb[yy, xx] > h0 + d * 0.35:
                    occ = 1.0
                    break
            sh[y, x] = occ
            # rim: edge of silhouette facing the light
            xr = int(x + Lx * 2.5 * ss)
            yr = int(y + Ly * 2.5 * ss)
            if xr >= 0 and yr >= 0 and xr < Ws and yr < Hs:
                if ib[yr, xr] < 0:
                    rim[y, x] = 1.0
            else:
                rim[y, x] = 1.0
    return ndl, sh, rim


def foliage(W, H, S, light=(0.55, -0.7, 0.45), pal=None, ss=2, shadow_shift=0.012, wl=0.6, wg=0.4,
            rim_amt=0.6, return_parts=False):
    """Render painted foliage clumps.

    S: (n, NF) float array of clumps (1x pixel coords): cx, cy, r, z (depth bias: larger = in front),
       group cx, cy, r (for coherent group shading), tone (-1..1), phase, lobes, amp (edge scallop),
       fog (0..1 aerial perspective), ay (vertical squash).
    light: screen-space direction toward the light (x right, y DOWN, z toward viewer).
    pal: dict of colours: deep, shd, mid, lit, hi, rim, haze, (warm) - RGB arrays.
    returns (H, W, 4) straight alpha.
    """
    L = np.asarray(light, np.float64)
    L = L / np.linalg.norm(L)
    Hs, Ws = H * ss, W * ss
    S = np.ascontiguousarray(S, np.float32)
    hb, ib = _fol_zbuf(Hs, Ws, S, ss)
    ndl, sh, rim = _fol_shade(hb, ib, S, ss, float(L[0]), float(L[1]), float(L[2]), shadow_shift * W * ss, wl, wg)
    cov = (ib >= 0).astype(np.float32)
    idx = np.maximum(ib, 0)
    tone = np.where(ib >= 0, S[idx, F_TONE], 0).astype(np.float32)
    fog = np.where(ib >= 0, S[idx, F_FOG], 0).astype(np.float32)
    sh = cv2.GaussianBlur(sh, (0, 0), 0.7 * ss)
    rngd = np.random.default_rng(int(S.shape[0]) % 1000)
    dap = cv2.resize(rngd.random((max(Hs // (3 * ss), 2), max(Ws // (3 * ss), 2))).astype(np.float32), (Ws, Hs),
                     interpolation=cv2.INTER_CUBIC)
    v = ndl - 0.45 * sh + 0.08 * tone + (dap - 0.5) * 0.22
    p = pal
    col = p['deep'] + (p['shd'] - p['deep']) * sstep(-0.6, -0.1, v)[..., None]
    col = col + (p['mid'] - col) * sstep(-0.12, 0.2, v)[..., None]
    col = col + (p['lit'] - col) * sstep(0.22, 0.48, v)[..., None]
    col = col + (p['hi'] - col) * (sstep(0.62, 0.9, v) * 0.75)[..., None]
    # hue variation per clump (warmer / cooler)
    if 'warm' in p:
        col = col + (p['warm'] - col) * (np.clip(tone, 0, 1) * 0.35)[..., None] * sstep(0.1, 0.4, v)[..., None]
    if 'cool' in p:
        col = col + (p['cool'] - col) * (np.clip(-tone, 0, 1) * 0.3)[..., None]
    rimv = rim * sstep(0.0, 0.3, ndl) * rim_amt
    col = col + (p['rim'] - col) * rimv[..., None]
    col = col + (p['haze'] - col) * fog[..., None]
    rgba = np.dstack([col * cov[..., None], cov]).astype(np.float32)
    small = cv2.resize(rgba, (W, H), interpolation=cv2.INTER_AREA)
    a = small[..., 3]
    rgb = small[..., :3] / np.maximum(a, 1e-4)[..., None]
    out = np.dstack([rgb, a]).astype(np.float32)
    if return_parts:
        return out, cv2.resize(fog * cov, (W, H), interpolation=cv2.INTER_AREA)
    return out


def clump_row(rng, x0, x1, ybase, height_fn, r_mean, z0=0.0, fog=0.0, tone_sd=0.4, density=1.0, ay=0.85,
              amp=0.12, fill_rows=4, group=None, rows_down=None):
    """Pack clumps over a region whose top edge is height_fn(x) (y of canopy top) and bottom is ybase.
    Returns list of rows (NF)."""
    out = []
    x = x0
    while x < x1:
        r = r_mean * rng.uniform(0.7, 1.35)
        top = height_fn(x)
        yb = ybase if np.isscalar(ybase) else ybase(x)
        # stack from the top of the canopy down to the base
        y = top + r * 0.8
        lev = 0
        while y < yb + r * 0.5:
            rr = r * (1.0 + 0.25 * lev) * rng.uniform(0.85, 1.15)
            gx, gy, gr = (group if group is not None else (x, y + rr * 0.5, rr * 2.6))
            out.append([x + rng.normal(0, 0.25) * r, y, rr, z0 - lev * r * 0.6 + rng.uniform(0, r * 0.3),
                        gx, gy, gr, np.clip(rng.normal(0, tone_sd), -1, 1), rng.uniform(0, 6.28),
                        rng.integers(5, 9), amp * rng.uniform(0.6, 1.3), fog, ay])
            y += rr * rng.uniform(0.9, 1.3)
            lev += 1
            if rows_down is not None and lev >= rows_down:
                break
        x += r * rng.uniform(0.9, 1.3) / density
    return out


def tree_clumps(rng, cx, by, w, h, r_mean=None, z0=0.0, fog=0.0, tone_sd=0.35, amp=0.1, ay=0.9, shape='round',
                tone0=0.0):
    """Clumps for a broadleaf crown centred at cx, crown bottom at by, crown w x h px: a few big interior
    masses, a shell of medium leaf masses over the upper surface, and small leaf-bumps on the silhouette.
    All share the crown as their group normal so the crown reads as one lit volume."""
    out = []
    gx, gy, gr = cx, by - h * 0.5, max(w, h) * 0.6
    ry = h * 0.5
    rx = w * 0.5

    def add(x, y, r, zb, tsd, a_):
        out.append([x, y, r, zb, gx, gy, gr, float(np.clip(tone0 + rng.normal(0, tsd), -1, 1)),
                    rng.uniform(0, 6.28), rng.integers(7, 13), a_ * 0.6, fog, ay])
    s = min(rx, ry)
    for i in range(int(rng.integers(4, 7))):
        a = rng.uniform(0, 2 * math.pi)
        rr = math.sqrt(rng.random()) * 0.45
        add(cx + math.cos(a) * rr * rx, gy + math.sin(a) * rr * ry, s * rng.uniform(0.45, 0.6), z0, tone_sd * 0.5,
            amp)
    n1 = int(np.clip((rx + ry) / max(s * 0.2, 1) * 1.1, 10, 60))
    for i in range(n1):
        a = rng.uniform(-math.pi * 1.05, math.pi * 0.25) if rng.random() < 0.85 else rng.uniform(0, math.pi)
        rr = rng.uniform(0.55, 0.85)
        r = s * rng.uniform(0.2, 0.3)
        add(cx + math.cos(a) * rr * rx, gy + math.sin(a) * rr * ry, r, z0 + s * 0.35 + rng.uniform(0, s * 0.1),
            tone_sd, amp * 1.2)
    n2 = int(np.clip((rx + ry) / max(s * 0.1, 1) * 0.35, 6, 40))
    for i in range(n2):
        a = rng.uniform(-math.pi * 1.0, 0.0) if rng.random() < 0.8 else rng.uniform(0, math.pi)
        rr = rng.uniform(0.8, 0.95)
        r = s * rng.uniform(0.12, 0.17)
        add(cx + math.cos(a) * rr * rx, gy + math.sin(a) * rr * ry, r, z0 + s * 0.5 + rng.uniform(0, s * 0.1),
            tone_sd, amp * 1.4)
    return out


def cedar_clumps(rng, cx, by, w, h, z0=0.0, fog=0.0, ay=0.8, tone0=-0.8):
    """Conical cedar (sugi): a spire of narrow tiers, dark and cool."""
    out = []
    gx, gy, gr = cx, by - h * 0.45, max(w * 1.2, h * 0.55)
    n = max(5, int(h / max(w * 0.18, 1)))
    for i in range(n):
        f = i / (n - 1)
        y = by - f * h * 0.95
        tw = w * 0.5 * (1 - f) ** 0.9 + w * 0.05
        for j in range(3):
            x = cx + rng.uniform(-1, 1) * tw * 0.7
            r = tw * rng.uniform(0.45, 0.65) + 0.8
            out.append([x, y, r, z0 + (1 - f) * 2 + j * 0.1, gx, gy, gr, float(np.clip(tone0 + rng.normal(0, 0.15), -1, 1)),
                        rng.uniform(0, 6.28), rng.integers(6, 10), 0.12, fog, ay])
    return out


def draw_cedar(cv, rng, cx, by, w, h, dark, lit, tip):
    """Japanese cedar (sugi) silhouette: tall narrow cone with jagged drooping branch tiers;
    shadowed left body, lit right flank, a few bright branch tips."""
    n = int(np.clip(h / max(w * 0.12, 1.0), 8, 26))
    L_, R_ = [], []
    for i in range(n + 1):
        f = i / n
        y = by - f * h
        half = w * 0.5 * (1 - f) ** 0.95 + w * 0.02
        if i < n:
            jl = rng.uniform(0.95, 1.2)
            jr = rng.uniform(0.95, 1.2)
            L_.append((cx - half * jl, y))
            L_.append((cx - half * 0.72, y - h / n * 0.45))
            R_.append((cx + half * jr, y + h / n * 0.08))
            R_.append((cx + half * 0.72, y - h / n * 0.4))
        else:
            L_.append((cx, y - h * 0.02))
    body = L_ + R_[::-1]
    cv.poly(body, dark)
    mid = [(cx + w * 0.05 * (1 - i / n), by - i / n * h) for i in range(n + 1)]
    cv.poly(mid + R_[::-1], lit)
    for i in range(0, len(R_) - 2, 2):
        a, b = R_[i], R_[i + 1]
        cv.poly([a, ((a[0] + b[0]) / 2 - w * 0.03, a[1] - h / n * 0.2), b], tip)


# ============================================================================ fast compositing

@njit(cache=True, fastmath=True, parallel=True)
def over_premult(img, rgb, a):
    """In place: img = img * (1 - a) + rgb (rgb premultiplied)."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            k = a[i, j]
            if k > 0.0:
                for c in range(3):
                    img[i, j, c] = img[i, j, c] * (1.0 - k) + rgb[i, j, c]


@njit(cache=True, fastmath=True, parallel=True)
def over_color(img, col, a):
    """In place: img = img * (1 - a) + col * a (single colour, alpha map)."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            k = a[i, j]
            if k > 0.0:
                for c in range(3):
                    img[i, j, c] = img[i, j, c] * (1.0 - k) + col[c] * k


@njit(cache=True, fastmath=True, parallel=True)
def diffuse(img, gl, keep, veil, knee, hi):
    """In place: img = img * keep + gl * veil + max(gl - knee, 0) * hi."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            for c in range(3):
                g = gl[i, j, c]
                e = g - knee
                if e < 0.0:
                    e = 0.0
                img[i, j, c] = img[i, j, c] * keep + g * veil + e * hi
