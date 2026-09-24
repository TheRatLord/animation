"""Sea for s06_seaside: perspective wave field (two travelling anisotropic wave textures, mip-filtered per
row so nothing aliases toward the horizon), sky reflection with Fresnel falloff, a painted sun path of
crisp horizontal glitter dashes, and twinkling star glints."""
import math
import numpy as np
import cv2

from numba import njit, prange

from lib import core as C, fx as F


@njit(cache=True, fastmath=True)
def _bil_wrap(T, u, v):
    N = T.shape[0]
    u = u % N
    v = v % N
    i0 = int(u)
    j0 = int(v)
    fu = u - i0
    fv = v - j0
    i1 = (i0 + 1) % N
    j1 = (j0 + 1) % N
    i0 = i0 % N
    j0 = j0 % N
    return (T[j0, i0] * (1 - fu) + T[j0, i1] * fu) * (1 - fv) + (T[j1, i0] * (1 - fu) + T[j1, i1] * fu) * fv


@njit(cache=True, fastmath=True)
def _ss(e0, e1, x):
    t = (x - e0) / (e1 - e0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return t * t * (3 - 2 * t)


@njit(cache=True, fastmath=True, parallel=True)
def _sea_kernel(T1, T2, refl, out, y0, W, H, y_h, t, shift, z0, ox, oy, sunx, deep, sunc, shear):
    K = 0.9 * H
    rs = H / 1080.0
    a1 = 460.0 * rs; b1 = 94.0 * rs
    a2 = 560.0 * rs; b2 = 118.0 * rs
    capv = 3.0
    capu = 2.5
    dc1 = math.sqrt(b1 * K / capv)
    dc2 = math.sqrt(b2 * K / capv)
    du1 = a1 * K / (capu * 0.9 * W)
    du2 = a2 * K / (capu * 0.9 * W)
    ph, pw = refl.shape[0], refl.shape[1]
    for y in prange(y0, H):
        dyy = max(y - y_h, 0.35)
        f = min(max(dyy / (H - y_h), 0.0), 1.0)
        drow = shift
        dtex = drow + shear * dyy
        q = min(max(dyy / (0.3 * H), 0.06), 1.0)
        if dyy > dc1:
            v1 = b1 * K / dyy
        else:
            v1 = b1 * K / dc1 + capv * (dc1 - dyy)
        if dyy > dc2:
            v2 = b2 * K / dyy
        else:
            v2 = b2 * K / dc2 + capv * (dc2 - dyy)
        v1 = v1 - t * 7.0 * q
        v2 = v2 - t * 5.0 * q + 57.0
        k1 = a1 * K / max(dyy, du1) / (0.9 * W)
        k2 = a2 * K / max(dyy, du2) / (0.9 * W)
        fres = math.exp(-dyy / (0.07 * H))
        wr = 0.35 + 0.6 * fres
        wd = 0.65 - 0.6 * fres
        wp = 0.018 * W + 0.33 * dyy
        near_h = math.exp(-dyy / (0.012 * H))
        sheen_v = 0.4 + 0.6 * math.exp(-dyy / (0.2 * H))
        thr0 = 0.27 + 0.06 * math.exp(-dyy / (0.03 * H))
        my = (y - H / 2) / z0 + H / 2 + oy
        jy = int(my)
        fy = my - jy
        jy = min(max(jy, 0), ph - 2)
        for x in range(W):
            xc = x - W / 2 - dtex
            u1 = xc * k1 + t * 14.0 * q
            u2 = xc * k2 - t * 20.0 * q + 131.0
            hgt = 0.55 * _bil_wrap(T1, u1, v1) + 0.45 * _bil_wrap(T2, u2, v2)
            hc = hgt - 0.5
            # reflection lookup (sheared per row)
            mx = (x - W / 2) / z0 + W / 2 + ox - drow
            ix = int(mx)
            fx = mx - ix
            if ix < 0:
                ix = 0; fx = 0.0
            if ix > pw - 2:
                ix = pw - 2; fx = 1.0
            crest = _ss(0.05, 0.22, hc)
            trough = _ss(-0.05, -0.25, hc)
            dx = x - sunx
            core = math.exp(-(dx / (0.45 * wp)) ** 2)
            path = math.exp(-(dx / wp) ** 2)
            broad = math.exp(-(dx / (2.6 * wp)) ** 2)
            thr = thr0 - 0.12 * core - 0.08 * path
            sp = _ss(thr - 0.014, thr + 0.014, hc) * path
            glit = sp * (0.7 + 0.8 * core)
            side = _ss(0.3, 0.36, hc) * broad * (1 - path)
            sh = (broad * 0.22 + path * 0.25) * sheen_v
            gcol = (1.0, 0.86, 0.58)
            hcol = (1.0, 0.85, 0.6)
            scol = (1.0, 0.7, 0.62)
            ctint = (1.05, 0.95, 1.0)
            for c in range(3):
                r = (refl[jy, ix, c] * (1 - fx) + refl[jy, ix + 1, c] * fx) * (1 - fy) +                     (refl[jy + 1, ix, c] * (1 - fx) + refl[jy + 1, ix + 1, c] * fx) * fy
                v = r * wr + deep[c] * wd
                v = v * (1 + 0.28 * crest * ctint[c]) * (1 - 0.18 * trough)
                v += sh * sunc[c] + near_h * path * 0.8 * hcol[c]
                v += glit * gcol[c] * 1.5 + side * scol[c] * 0.35
                out[y, x, c] = v


def _wave_tex(N, seed, cx, cy, octaves=4):
    """Tileable anisotropic wave texture (0..1): long horizontal crests."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((N, N), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gx, gy = int(cx * 2 ** o), int(cy * 2 ** o)
        g = rng.random((gy, gx)).astype(np.float32)
        # tile by wrapping before the cubic upsample
        g = np.pad(g, 2, mode='wrap')
        big = cv2.resize(g, ((gx + 4) * N // gx, (gy + 4) * N // gy), interpolation=cv2.INTER_CUBIC)
        ox, oy = 2 * N // gx, 2 * N // gy
        acc += amp * big[oy:oy + N, ox:ox + N]
        tot += amp
        amp *= 0.55
    acc /= tot
    lo, hi = np.percentile(acc, 1), np.percentile(acc, 99)
    acc = np.clip((acc - lo) / (hi - lo), 0, 1)
    # sharpen into crest-like shapes (painted look): ridged + bias
    rid = 1 - np.abs(acc * 2 - 1)
    return (0.55 * acc + 0.45 * rid ** 1.5).astype(np.float32)


class Sea:
    def __init__(self, W, H, seed=0):
        self.W, self.H = W, H
        N = 512
        self.N = N
        t1 = _wave_tex(N, seed + 1, 6, 64, octaves=3)
        t2 = _wave_tex(N, seed + 2, 7, 72, octaves=3)
        # anisotropic mip chain: blur along v (rows) only, plus a little along u
        self.mips = []
        for lv in range(8):
            sv = 0.0 if lv == 0 else 0.7 * 2 ** (lv - 1)
            su = 0.0 if lv == 0 else 0.25 * 2 ** (lv - 1) ** 0.5
            def bl(t):
                if sv == 0:
                    return t
                k = cv2.GaussianBlur(np.tile(t, (3, 3)), (0, 0), sigmaX=max(su, 0.01), sigmaY=sv)
                return k[N:2 * N, N:2 * N].copy()
            self.mips.append((bl(t1), bl(t2)))
        rng = np.random.default_rng(seed + 9)
        n = 150
        self.g_u = rng.normal(0, 0.55, n)          # lateral position in path widths
        self.g_v = rng.random(n) ** 1.6            # 0 near horizon .. 1 near
        self.g_s = rng.uniform(0.5, 1.0, n)
        self.g_ph = rng.uniform(0, 6.28, n)
        self.g_fr = rng.uniform(1.2, 3.0, n)

    def render(self, t, y_h, x_shift_row, sunx, refl_rows, deep, sun_col, y1=None):
        """Render the sea rows [ceil(y_h), H). x_shift_row(dyy) -> per-row horizontal shift (px).
        refl_rows: (rows, W, 3) sky reflection already sampled. Returns (rows, W, 3), row0."""
        W, H = self.W, self.H
        y0 = int(math.ceil(y_h))
        ys = np.arange(y0, H, dtype=np.float32)[:, None]
        xs = np.arange(W, dtype=np.float32)[None]
        dyy = np.maximum(ys - y_h, 0.35)
        nrow = H - y0
        # plane coordinates (arbitrary units): depth ~ 1/dyy. Texture density on screen is CAPPED near the
        # horizon (painter's trick): waves compress into ever finer horizontal lines but never go sub-pixel.
        K = 0.9 * H
        N = self.N
        rs = H / 1080.0
        a1, b1 = 460.0 * rs, 94.0 * rs
        a2, b2 = 560.0 * rs, 118.0 * rs
        capv = 3.0                                 # max texels per pixel vertically
        def vcoord(b):
            dc = math.sqrt(b * K / capv)
            return np.where(dyy > dc, b * K / dyy, b * K / dc + capv * (dc - dyy))
        capu = 2.5
        du = [a * K / (capu * 0.9 * W) for a in (a1, a2)]
        xc = (xs - W / 2 - x_shift_row(dyy))
        u1 = xc * a1 * K / np.maximum(dyy, du[0]) / (0.9 * W)
        u2 = xc * a2 * K / np.maximum(dyy, du[1]) / (0.9 * W)
        q = np.clip(dyy / (0.3 * H), 0.06, 1.0)     # distant waves barely move on screen
        u1 = u1 + t * 14.0 * q
        u2 = u2 - t * 20.0 * q + 131
        v1 = vcoord(b1) - t * 7.0 * q
        v2 = vcoord(b2) - t * 5.0 * q + 57
        T1, T2 = self.mips[0]
        m1 = cv2.remap(T1, (u1 % N).astype(np.float32), np.broadcast_to(v1 % N, (nrow, W)).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        m2 = cv2.remap(T2, (u2 % N).astype(np.float32), np.broadcast_to(v2 % N, (nrow, W)).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        h = 0.55 * m1 + 0.45 * m2
        detail = np.ones((nrow, 1), np.float32)
        hc = (h - 0.5)
        # ---------------- base water colour
        fres = np.exp(-dyy / (0.07 * H))
        col = refl_rows * (0.35 + 0.6 * fres)[..., None] + deep * (0.65 - 0.6 * fres)[..., None]
        # wave shading outside the path: crests pick up the bright low sky, troughs the deep blue
        crest = C.smoothstep(0.05, 0.22, hc) * detail
        trough = C.smoothstep(-0.05, -0.25, hc) * detail
        col = col * (1 + 0.28 * crest[..., None] * np.array([1.05, 0.95, 1.0], np.float32))
        col = col * (1 - 0.18 * trough[..., None])
        # ---------------- sun path
        dx = xs - sunx
        wp = 0.018 * W + 0.33 * dyy
        core = np.exp(-(dx / (0.45 * wp)) ** 2)
        path = np.exp(-(dx / wp) ** 2)
        broad = np.exp(-(dx / (2.6 * wp)) ** 2)
        near_h = np.exp(-dyy / (0.012 * H))
        # broad warm sheen + hot band at the horizon
        col = col + (broad * 0.22 + path * 0.25)[..., None] * sun_col * (0.4 + 0.6 * np.exp(-dyy / (0.2 * H)))[..., None]
        col = col + (near_h * path * 0.8)[..., None] * np.array([1.0, 0.85, 0.6], np.float32)
        # glitter: crisp horizontal dashes where crests exceed a threshold that drops inside the path
        thr = 0.27 - 0.12 * core - 0.08 * path + 0.06 * np.exp(-dyy / (0.03 * H))
        e = 0.014
        sp = C.smoothstep(thr - e, thr + e, hc) * path
        # far rows: field is blurred -> replace with a solid bright shimmer
        sp = sp * detail + path * (1 - detail) * (0.3 + 0.35 * core)
        glit = sp * (0.7 + 0.8 * core)
        gc = np.array([1.0, 0.86, 0.58], np.float32)
        col = col + glit[..., None] * (gc * 1.15)
        # outside the path: sparse pink crest glints
        side = C.smoothstep(0.3, 0.36, hc) * broad * (1 - path) * detail
        col = col + side[..., None] * np.array([1.0, 0.7, 0.62], np.float32) * 0.35
        self.last_glit = glit
        return col.astype(np.float32), y0

    def render_fast(self, out, t, y_h, shift, z0, ox, oy, sunx, refl_plate, deep, sun_col, shear=0.0):
        """numba version: writes rows [ceil(y_h), H) of out (H, W, 3) in place."""
        y0 = int(math.ceil(y_h))
        T1, T2 = self.mips[0]
        _sea_kernel(T1, T2, refl_plate, out, y0, self.W, self.H, float(y_h), float(t), float(shift), float(z0),
                    float(ox), float(oy), float(sunx), np.asarray(deep, np.float32), np.asarray(sun_col, np.float32), float(shear))
        return y0

    def glint_params(self, t, sunx, y_h, y_max):
        W, H = self.W, self.H
        dyy = 0.004 * H + self.g_v * (y_max - y_h - 0.004 * H)
        wp = 0.018 * W + 0.33 * dyy
        xs = sunx + self.g_u * wp * 0.6
        ys = y_h + dyy
        size = 0.004 + 0.012 * self.g_s * (0.3 + 0.7 * self.g_v)
        inten = np.clip(np.sin(t * self.g_fr + self.g_ph), 0, 1) ** 4 * (0.4 + 0.9 * np.exp(-np.abs(self.g_u) * 1.5))
        return xs, ys, size, inten

    def glints(self, t, sunx, y_h, y_max):
        W, H = self.W, self.H
        dyy = 0.004 * H + self.g_v * (y_max - y_h - 0.004 * H)
        wp = 0.018 * W + 0.33 * dyy
        xs = sunx + self.g_u * wp * 0.6
        ys = y_h + dyy
        size = 0.004 + 0.012 * self.g_s * (0.3 + 0.7 * self.g_v)
        inten = np.clip(np.sin(t * self.g_fr + self.g_ph), 0, 1) ** 4 * (0.4 + 0.9 * np.exp(-np.abs(self.g_u) * 1.5))
        return F.glints(W, H, xs, ys, size=size, intensity=inten * 0.9, color=(1.0, 0.9, 0.7))
