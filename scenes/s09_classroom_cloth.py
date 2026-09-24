"""Curtains for s09_classroom: animated cloth meshes (analytic billowing) + a numba rasteriser with
4x coverage sub-samples, perspective-correct attributes and a z-test against the ray-traced depth.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

import s09_classroom_rt as R


class Curtain:
    """One curtain panel hanging from the rail.
    x0, x1: extent along the window wall when hanging still; gathered: deep folds (drawn-open bunch);
    billow: how strongly the wind pushes it into the room; side: +1/-1 sideways drift direction."""

    def __init__(self, x0, x1, nfold, amp, billow, side, phase, z0=0.26, top=2.78, bottom=0.78, nu=110, nv=44):
        self.x0, self.x1, self.nfold, self.amp = x0, x1, nfold, amp
        self.billow, self.side, self.phase = billow, side, phase
        self.z0, self.top, self.bottom = z0, top, bottom
        self.nu, self.nv = nu, nv
        self.u = np.linspace(0, 1, nu)[None, :]
        self.v = np.linspace(0, 1, nv)[:, None]

    def verts(self, t):
        u, v = self.u, self.v
        ph = self.phase
        # gust envelope: one slow swell over the shot + small flutter
        g = self.billow * (0.62 + 0.3 * math.sin(t * 0.75 + ph) + 0.08 * math.sin(t * 2.3 + ph * 2.1))
        vv = v ** 1.5
        width = self.x1 - self.x0
        # folds: travelling ripple, deeper toward the bottom when billowing
        # irregular folds: a main pleat train + a detuned second train + diagonal drift when billowing,
        # amplitude varying across the panel (gathered bunches and flatter stretches)
        env = 0.65 + 0.35 * np.sin(2 * math.pi * 1.3 * u + ph * 1.7) + 0.2 * np.sin(2 * math.pi * 3.1 * u + ph)
        diag = 1.6 * vv * self.billow * (0.8 + 0.2 * math.sin(t * 0.6 + ph))
        fold = self.amp * env * (np.sin(2 * math.pi * self.nfold * u + ph + diag * self.side
                                        + 0.4 * np.sin(t * 0.9 + v * 2.0 + ph))
                                 + 0.45 * np.sin(2 * math.pi * self.nfold * 2.17 * u + ph * 3.1 + 0.5 * t + 2.2 * v)
                                 + 0.2 * np.sin(2 * math.pi * self.nfold * 4.3 * u + ph * 5.0 - 0.8 * t + 3.0 * v))
        wave = np.sin(2 * math.pi * (1.2 * u - 0.28 * t) + 3.0 * v + ph) * 0.5 + 0.5
        bul = np.sin(math.pi * np.clip(u * 0.85 + 0.15 * (1 if self.side < 0 else 0), 0, 1))
        x = self.x0 + width * u + self.side * g * 0.45 * vv * (0.4 + 0.6 * u if self.side > 0 else 1.0 - 0.6 * u)
        z = self.z0 + fold * (1 + 1.2 * vv * g) + g * vv * (0.55 + 0.45 * bul) * (0.75 + 0.25 * wave) * 1.1
        y = self.top - (self.top - self.bottom) * v + g * 0.28 * v ** 2.2 * (0.6 + 0.4 * wave)
        x = np.broadcast_to(x, (self.nv, self.nu))
        return np.stack([x, np.broadcast_to(y, x.shape), np.broadcast_to(z, x.shape)], -1).astype(np.float64)


def normals(Vt):
    du = np.gradient(Vt, axis=1)
    dv = np.gradient(Vt, axis=0)
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    return n


def shadow_mask(meshes, L, CM, ppm, y0):
    """Rasterise the curtains' shadow onto the window plane (z = 0) along the sun direction."""
    CM[:] = 0
    ss = 2
    h, w = CM.shape
    m = np.zeros((h * ss, w * ss), np.uint8)
    polys = []
    for V in meshes:
        tt = -V[..., 2] / L[2]
        xw = V[..., 0] + L[0] * tt
        yw = V[..., 1] + L[1] * tt
        px = xw * ppm * ss
        py = (y0 - yw) * ppm * ss
        P = np.stack([px, py], -1)
        q = np.stack([P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]], 2).reshape(-1, 4, 2)
        polys.append(q)
    q = np.concatenate(polys, 0).astype(np.int32)
    cv2.fillPoly(m, list(q), 255)
    CM[:] = cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    CM[:] = cv2.GaussianBlur(CM, (0, 0), 1.5)


@njit(fastmath=True, cache=True)
def _shade_cloth(px, py, pz, nx, ny, nz, u, v, cx, cy, cz, P, VB, HB, out):
    """Sheer backlit curtain, painted: warm glowing translucency where the sun is behind it, cool
    lavender fold valleys, hard-ish (posterised) fold terminators, a woven vertical texture, a denser
    pleated header and double hem, window-frame shadow bars from the aperture."""
    Lx, Ly, Lz = P[0], P[1], P[2]
    vx, vy, vz = cx - px, cy - py, cz - pz
    vl = math.sqrt(vx * vx + vy * vy + vz * vz)
    vx /= vl
    vy /= vl
    vz /= vl
    ndv = nx * vx + ny * vy + nz * vz
    if ndv < 0:
        nx, ny, nz, ndv = -nx, -ny, -nz, -ndv
    ndl = nx * Lx + ny * Ly + nz * Lz            # >0: the side we see faces the sun
    ap = 0.0
    if Lz < 0 and pz > 0:
        tw = -pz / Lz
        ap = R.aperture(px + Lx * tw, py + Ly * tw, 0.004 + tw * 0.012, VB, HB)
    sunr, sung, sunb = P[3], P[4], P[5]
    # fold light: posterised terminator (painted cel step with a soft 2-tone ramp)
    fl = abs(ndl)
    step = R._ss(0.18, 0.32, fl) * 0.65 + 0.35 * fl
    trans = ap * (0.3 + 0.7 * step) * (0.55 + 0.45 * ndv) if ndl < 0 else ap * 0.35 * (1 - fl)
    dirl = ap * ndl * step if ndl > 0 else 0.0
    glow = trans * 0.62 + dirl * 0.85
    # weave / seam texture along the fabric (vertical threads, a few denser stripes)
    thr = 0.5 + 0.5 * math.sin(u * 180.0 + math.sin(v * 9.0 + u * 13.0) * 1.5)
    stripe = R._ss(0.8, 0.95, 0.5 + 0.5 * math.sin(u * 61.0 + 1.3))
    dens = 1.0 + 0.06 * thr + 0.12 * stripe
    if v < 0.035:
        dens += 0.5                                # gathered header tape
    elif v > 0.955:
        dens += 0.35                               # doubled hem
    # ambient: cool lavender, darker in the valleys that face away from the room light
    amb = (0.5 + 0.2 * ny + 0.3 * ndv) * (0.8 + 0.2 * (0.5 - 0.5 * nz))
    r = 0.80 * P[6] * amb * 1.05 + glow * sunr * 0.52 * 1.0
    g = 0.78 * P[7] * amb * 1.02 + glow * sung * 0.52 * 0.97
    b = 0.95 * P[8] * amb * 1.0 + glow * sunb * 0.52 * 0.9
    # backlit rim on grazing folds facing the sun
    back = -(Lx * vx + Ly * vy + Lz * vz)
    if back > 0.0 and ap > 0.0:
        rk = (1.0 - ndv) ** 3 * (0.4 + 0.6 * back) * ap * 2.2
        r += rk * 1.0
        g += rk * 0.62
        b += rk * 0.3
    r *= 1.0 - 0.05 * (dens - 1.0)
    g *= 1.0 - 0.07 * (dens - 1.0)
    b *= 1.0 - 0.06 * (dens - 1.0)
    # sheer: see-through at normal incidence, more layers (opaque) at grazing / in folds
    a = (0.42 + 0.5 * (1 - ndv) ** 1.5) * dens
    a = a if a < 0.97 else 0.97
    out[0] = r
    out[1] = g
    out[2] = b
    out[3] = a


@njit(parallel=True, fastmath=True, cache=True)
def raster(W, H, cam, fpx, V, N, UV, tris, zbuf_scene, P, VB, HB, sub_col, sub_a, sub_z):
    """Rasterise triangles (indices into V) with 4 sub-samples per pixel.
    V, N: (n, 3) world positions / normals; UV: (n, 2). zbuf_scene: (H, W) view-space depth of the
    ray-traced scene. Outputs sub_col (H, W, 4, 3), sub_a (H, W, 4), sub_z (H, W, 4) (initialised by caller:
    sub_z = zbuf_scene, sub_a = 0)."""
    n = V.shape[0]
    sx = np.empty(n)
    sy = np.empty(n)
    iz = np.empty(n)
    for i in range(n):
        rx, ry, rz = V[i, 0] - cam[0], V[i, 1] - cam[1], V[i, 2] - cam[2]
        zc = rx * cam[3] + ry * cam[4] + rz * cam[5]
        xc = rx * cam[6] + ry * cam[7] + rz * cam[8]
        yc = rx * cam[9] + ry * cam[10] + rz * cam[11]
        zc = max(zc, 1e-3)
        sx[i] = W * 0.5 + xc / zc * fpx
        sy[i] = H * 0.5 - yc / zc * fpx
        iz[i] = 1.0 / zc
    offs = np.array([[0.375, 0.125], [0.875, 0.375], [0.625, 0.875], [0.125, 0.625]])
    # parallel over horizontal bands to avoid write races
    nb = 64
    band = (H + nb - 1) // nb
    for bi in prange(nb):
        yb0 = bi * band
        yb1 = min(H, yb0 + band)
        out = np.zeros(4)
        for k in range(tris.shape[0]):
            a, b, c = tris[k, 0], tris[k, 1], tris[k, 2]
            x0 = min(sx[a], sx[b], sx[c])
            x1 = max(sx[a], sx[b], sx[c])
            y0 = min(sy[a], sy[b], sy[c])
            y1 = max(sy[a], sy[b], sy[c])
            if y1 < yb0 or y0 >= yb1 or x1 < 0 or x0 >= W:
                continue
            area = (sx[b] - sx[a]) * (sy[c] - sy[a]) - (sx[c] - sx[a]) * (sy[b] - sy[a])
            if abs(area) < 1e-9:
                continue
            ia = 1.0 / area
            px0 = max(int(x0), 0)
            px1 = min(int(x1) + 1, W - 1)
            py0 = max(int(y0), yb0)
            py1 = min(int(y1) + 1, yb1 - 1)
            for yy in range(py0, py1 + 1):
                for xx in range(px0, px1 + 1):
                    for s in range(4):
                        qx = xx + offs[s, 0]
                        qy = yy + offs[s, 1]
                        w0 = ((sx[b] - qx) * (sy[c] - qy) - (sx[c] - qx) * (sy[b] - qy)) * ia
                        w1 = ((sx[c] - qx) * (sy[a] - qy) - (sx[a] - qx) * (sy[c] - qy)) * ia
                        w2 = 1.0 - w0 - w1
                        if w0 < 0 or w1 < 0 or w2 < 0:
                            continue
                        pa, pb, pc = w0 * iz[a], w1 * iz[b], w2 * iz[c]
                        izs = pa + pb + pc
                        zv = 1.0 / izs
                        if zv >= sub_z[yy, xx, s]:
                            continue
                        pa *= zv
                        pb *= zv
                        pc *= zv
                        px = pa * V[a, 0] + pb * V[b, 0] + pc * V[c, 0]
                        py = pa * V[a, 1] + pb * V[b, 1] + pc * V[c, 1]
                        pz = pa * V[a, 2] + pb * V[b, 2] + pc * V[c, 2]
                        nx = pa * N[a, 0] + pb * N[b, 0] + pc * N[c, 0]
                        ny = pa * N[a, 1] + pb * N[b, 1] + pc * N[c, 1]
                        nz = pa * N[a, 2] + pb * N[b, 2] + pc * N[c, 2]
                        nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-12
                        uu = pa * UV[a, 0] + pb * UV[b, 0] + pc * UV[c, 0]
                        vv = pa * UV[a, 1] + pb * UV[b, 1] + pc * UV[c, 1]
                        _shade_cloth(px, py, pz, nx / nl, ny / nl, nz / nl, uu, vv, cam[0], cam[1], cam[2], P,
                                     VB, HB, out)
                        sub_z[yy, xx, s] = zv
                        sub_col[yy, xx, s, 0] = out[0]
                        sub_col[yy, xx, s, 1] = out[1]
                        sub_col[yy, xx, s, 2] = out[2]
                        sub_a[yy, xx, s] = out[3]


def grid_tris(nv, nu, base):
    i = np.arange(nv - 1)[:, None] * nu + np.arange(nu - 1)[None, :]
    i = i.ravel() + base
    t1 = np.stack([i, i + 1, i + nu + 1], 1)
    t2 = np.stack([i, i + nu + 1, i + nu], 1)
    return np.concatenate([t1, t2], 0)


@njit(parallel=True, fastmath=True, cache=True)
def init_bufs(zs, sub_a, sub_z):
    H, W = zs.shape
    for y in prange(H):
        for x in range(W):
            for s in range(4):
                sub_a[y, x, s] = 0.0
                sub_z[y, x, s] = zs[y, x]


@njit(parallel=True, fastmath=True, cache=True)
def resolve(img, sub_col, sub_a, sub_z, depth, zfac, cov):
    """Composite the 4 cloth sub-samples over img in place; cov = mean cloth coverage; depth updated."""
    H, W = depth.shape
    for y in prange(H):
        for x in range(W):
            ca = sub_a[y, x, 0] + sub_a[y, x, 1] + sub_a[y, x, 2] + sub_a[y, x, 3]
            if ca <= 0.0:
                cov[y, x] = 0.0
                continue
            r = 0.0
            g = 0.0
            b = 0.0
            zmin = 1e9
            n = 0
            for s in range(4):
                a = sub_a[y, x, s]
                r += img[y, x, 0] * (1 - a) + sub_col[y, x, s, 0] * a
                g += img[y, x, 1] * (1 - a) + sub_col[y, x, s, 1] * a
                b += img[y, x, 2] * (1 - a) + sub_col[y, x, s, 2] * a
                if a > 0:
                    n += 1
                    if sub_z[y, x, s] < zmin:
                        zmin = sub_z[y, x, s]
            img[y, x, 0] = r * 0.25
            img[y, x, 1] = g * 0.25
            img[y, x, 2] = b * 0.25
            cov[y, x] = ca * 0.25
            if n >= 2:
                depth[y, x] = zmin / zfac[y, x]
