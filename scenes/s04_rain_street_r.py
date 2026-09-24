"""s04_rain_street renderer helpers: a tiny 2.5D 'deferred' painter.

World: X right, Y up (ground = 0), Z forward (metres). Pinhole camera at (0, h, 0) looking +Z.
Everything static is painted once into depth-carrying plates (premultiplied RGB + alpha + depth);
per frame each plate is re-projected for the moving camera with a depth-driven inverse remap,
which gives exact parallax for a push-in / small translation.
"""
import math
import numpy as np
import cv2


# ----------------------------------------------------------------------------- camera

class Cam:
    def __init__(self, W, H, f_rel=0.86, h=1.35, pp=(0.55, 0.48), margin=0.04, ss=2):
        self.W, self.H = W, H
        self.mx = int(round(W * margin))
        self.my = int(round(H * margin))
        self.PW, self.PH = W + 2 * self.mx, H + 2 * self.my     # plate size (1x)
        self.ss = ss
        self.f = f_rel * W
        self.cx, self.cy = pp[0] * W, pp[1] * H                  # principal point (frame px)
        self.h = h
        self.pcx, self.pcy = self.cx + self.mx, self.cy + self.my  # principal point (plate px, 1x)

    # --- projection into the supersampled plate
    def proj(self, X, Y, Z, s=None):
        s = self.ss if s is None else s
        X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64); Z = np.asarray(Z, np.float64)
        return (self.pcx + self.f * X / Z) * s, (self.pcy - self.f * (Y - self.h) / Z) * s

    def px_per_m(self, Z, s=None):
        s = self.ss if s is None else s
        return self.f * s / Z

    def world_from_plate(self, x, y, z):
        """plate px (1x) + depth -> world X, Y."""
        X = (x - self.pcx) * z / self.f
        Y = self.h - (y - self.pcy) * z / self.f
        return X, Y

    def grid(self, s=1):
        ys, xs = np.mgrid[0:self.PH * s, 0:self.PW * s].astype(np.float32)
        return xs / s, ys / s


# ----------------------------------------------------------------------------- anti-aliased texture warp

def _warp_aa(rgba, emi, M, bw, bh, interp, max_lvl=4):
    """warpPerspective with horizontal rip-map filtering: walls seen at grazing angles compress their texture
    along u by up to ~16x; plain bilinear sampling then aliases slats / shelves / copy into streaky dither.
    Texture levels pre-filtered along u (INTER_AREA) are blended per pixel by the local u-footprint."""
    th, tw = rgba.shape[:2]
    T = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], np.float64)

    def pad(img):
        return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    def warp_level(k):
        if k == 0:
            r, e, sx = rgba, emi, 1.0
        else:
            nw = max(1, int(round(tw / 2 ** k)))
            sx = nw / tw
            r = cv2.resize(rgba, (nw, th), interpolation=cv2.INTER_AREA)
            e = None if emi is None else cv2.resize(emi, (nw, th), interpolation=cv2.INTER_AREA)
        S = np.diag([sx, 1.0, 1.0])
        Mk = M @ np.linalg.inv(S) @ np.linalg.inv(T)
        wr = cv2.warpPerspective(pad(r), Mk, (bw, bh), flags=interp, borderMode=cv2.BORDER_CONSTANT)
        we = None if e is None else cv2.warpPerspective(pad(e), Mk, (bw, bh), flags=interp,
                                                         borderMode=cv2.BORDER_CONSTANT)
        return wr, we

    # footprint along u per destination pixel (analytic Jacobian of the inverse homography)
    Mi = np.linalg.inv(M)
    step = 8
    ys, xs = np.mgrid[0:bh:step, 0:bw:step].astype(np.float64)
    s0 = Mi[0, 0] * xs + Mi[0, 1] * ys + Mi[0, 2]
    s2 = Mi[2, 0] * xs + Mi[2, 1] * ys + Mi[2, 2]
    s2 = np.where(np.abs(s2) < 1e-9, 1e-9, s2)
    dudx = (Mi[0, 0] * s2 - s0 * Mi[2, 0]) / s2 ** 2
    dudy = (Mi[0, 1] * s2 - s0 * Mi[2, 1]) / s2 ** 2
    lod = np.log2(np.maximum(np.sqrt(dudx ** 2 + dudy ** 2), 1e-6))
    lod = np.clip(lod - 0.15, 0, max_lvl).astype(np.float32)
    top = float(lod.max())
    if top < 0.25:
        return warp_level(0)
    lod = cv2.resize(lod, (bw, bh), interpolation=cv2.INTER_LINEAR)
    nl = int(np.ceil(top))
    out_r = np.zeros((bh, bw, 4), np.float32)
    out_e = None if emi is None else np.zeros((bh, bw, 3), np.float32)
    for k in range(nl + 1):
        w = np.clip(1 - np.abs(lod - k), 0, 1)
        if k == nl:
            w = np.maximum(w, (lod > k).astype(np.float32))
        if w.max() <= 0:
            continue
        wr, we = warp_level(k)
        out_r += wr * w[..., None]
        if out_e is not None:
            out_e += we * w[..., None]
    return out_r, out_e


# ----------------------------------------------------------------------------- canvas

class Canvas:
    """Supersampled G-buffer: premultiplied albedo, emissive, alpha, depth, normal, spec (gloss)."""

    def __init__(self, cam):
        self.cam = cam
        s = cam.ss
        self.H, self.W = cam.PH * s, cam.PW * s
        self.alb = np.zeros((self.H, self.W, 3), np.float32)
        self.emi = np.zeros((self.H, self.W, 3), np.float32)
        self.a = np.zeros((self.H, self.W), np.float32)
        self.z = np.zeros((self.H, self.W), np.float32)
        self.n = np.zeros((self.H, self.W, 3), np.float32)

    # generic compositing of a local patch
    def put(self, x0, y0, alb, a, z, n, emi=None):
        h, w = a.shape
        x1, y1 = x0 + w, y0 + h
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, self.W), min(y1, self.H)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        sl = (slice(cy0, cy1), slice(cx0, cx1))
        ls = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        A = a[ls]
        if A.max() <= 0:
            return
        A3 = A[..., None]
        self.alb[sl] = self.alb[sl] * (1 - A3) + alb[ls] * A3
        if emi is not None:
            self.emi[sl] = self.emi[sl] * (1 - A3) + emi[ls] * A3
        else:
            self.emi[sl] = self.emi[sl] * (1 - A3)
        zz = z[ls] if np.ndim(z) == 2 else z
        self.z[sl] = self.z[sl] * (1 - A) + zz * A
        nn = n[ls] if np.ndim(n) == 3 else np.asarray(n, np.float32)
        self.n[sl] = self.n[sl] * (1 - A3) + nn * A3
        self.a[sl] = self.a[sl] * (1 - A) + A

    def quad(self, P, alb, a=None, emi=None, normal=(0, 0, -1), interp=cv2.INTER_LINEAR):
        """Texture-map a planar world quad. P: 4x3 corners (TL, TR, BR, BL of the texture).
        alb (h,w,3) straight albedo, a (h,w) alpha, emi (h,w,3) emissive."""
        cam = self.cam
        P = np.asarray(P, np.float64)
        if np.any(P[:, 2] <= 0.05):
            return
        xs, ys = cam.proj(P[:, 0], P[:, 1], P[:, 2])
        x0, y0 = int(math.floor(xs.min())) - 1, int(math.floor(ys.min())) - 1
        x1, y1 = int(math.ceil(xs.max())) + 2, int(math.ceil(ys.max())) + 2
        x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, self.W), min(y1, self.H)
        if x1c <= x0c or y1c <= y0c:
            return
        th, tw = alb.shape[:2]
        src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
        dst = np.float32(np.stack([xs - x0c, ys - y0c], 1))
        M = cv2.getPerspectiveTransform(src, dst)
        bw, bh = x1c - x0c, y1c - y0c
        if a is None:
            a = np.ones((th, tw), np.float32)
        # pad textures by a transparent border so the quad edge is anti-aliased
        def pad(img):
            return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        T = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]], np.float64)
        M2 = M @ np.linalg.inv(T)
        rgba = np.dstack([alb * a[..., None], a]).astype(np.float32)
        emp = None if emi is None else (emi * a[..., None]).astype(np.float32)
        wr, E = _warp_aa(rgba, emp, M, bw, bh, interp)
        A = np.clip(wr[..., 3], 0, 1)
        if A.max() <= 0:
            return
        col = wr[..., :3] / np.maximum(wr[..., 3:4], 1e-6)
        if E is not None:
            E = E / np.maximum(wr[..., 3:4], 1e-6)
        # depth: 1/z is affine in screen space over a plane
        A3 = np.stack([xs[:3], ys[:3], np.ones(3)], 1)
        iz = np.linalg.solve(A3, 1.0 / P[:3, 2]) if abs(np.linalg.det(A3)) > 1e-6 else None
        if iz is None:
            A3 = np.stack([xs[[0, 2, 3]], ys[[0, 2, 3]], np.ones(3)], 1)
            iz = np.linalg.lstsq(A3, 1.0 / P[[0, 2, 3], 2], rcond=None)[0]
        yy, xx = np.mgrid[y0c:y1c, x0c:x1c].astype(np.float32)
        Z = 1.0 / np.maximum(iz[0] * xx + iz[1] * yy + iz[2], 1e-6)
        self.put(x0c, y0c, col, A, Z.astype(np.float32), np.asarray(normal, np.float32), E)

    def sprite(self, X, Y, Z, w, h, alb, a=None, emi=None, normal=(0, 0, -1), anchor='bottom'):
        """Camera-facing rectangle at depth Z: (X, Y) = bottom-centre (anchor='bottom') or centre."""
        if anchor == 'bottom':
            y0, y1 = Y, Y + h
        else:
            y0, y1 = Y - h / 2, Y + h / 2
        P = [(X - w / 2, y1, Z), (X + w / 2, y1, Z), (X + w / 2, y0, Z), (X - w / 2, y0, Z)]
        self.quad(P, alb, a, emi, normal)

    def polyline3d(self, pts, diam, color, emi=None, normal=(0, -0.3, -1), min_px=0.55, alpha=1.0):
        """Thin 3D line (wire, cable). pts (N,3). Width follows depth; sub-pixel widths fade alpha."""
        cam = self.cam
        pts = np.asarray(pts, np.float64)
        pts = pts[pts[:, 2] > 0.3]
        if len(pts) < 2:
            return
        xs, ys = cam.proj(pts[:, 0], pts[:, 1], pts[:, 2])
        wpx = diam * cam.f * cam.ss / pts[:, 2]
        x0, y0 = int(max(xs.min() - 8, 0)), int(max(ys.min() - 8, 0))
        x1, y1 = int(min(xs.max() + 8, self.W)), int(min(ys.max() + 8, self.H))
        if x1 <= x0 or y1 <= y0:
            return
        q = 4
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for i in range(len(pts) - 1):
            wv = 0.5 * (wpx[i] + wpx[i + 1])
            th = max(wv, min_px)
            cov = min(1.0, wv / min_px)
            ti = max(1, int(round(th)))
            if th < ti:
                cov *= th / ti
            p0 = (int(round((xs[i] - x0) * q)), int(round((ys[i] - y0) * q)))
            p1 = (int(round((xs[i + 1] - x0) * q)), int(round((ys[i + 1] - y0) * q)))
            cv2.line(m, p0, p1, int(round(255 * cov)), ti, cv2.LINE_AA, shift=2)
        m = m.astype(np.float32) / 255.0
        m *= alpha
        # depth by nearest segment: interpolate along x (wires are mostly horizontal-ish in screen)
        zs = pts[:, 2]
        order = np.argsort(xs)
        xx = np.arange(x0, x1, dtype=np.float64)
        if xs.max() - xs.min() > 2:
            zrow = np.interp(xx, xs[order], zs[order])
            Z = np.repeat(zrow[None, :], y1 - y0, 0).astype(np.float32)
        else:
            order = np.argsort(ys)
            yy = np.arange(y0, y1, dtype=np.float64)
            zcol = np.interp(yy, ys[order], zs[order])
            Z = np.repeat(zcol[:, None], x1 - x0, 1).astype(np.float32)
        col = np.broadcast_to(np.asarray(color, np.float32), m.shape + (3,))
        E = None if emi is None else np.broadcast_to(np.asarray(emi, np.float32), m.shape + (3,))
        self.put(x0, y0, col, m, Z, np.asarray(normal, np.float32), E)


# ----------------------------------------------------------------------------- plate baking

def pushpull_fill(val, w, levels=9):
    """Fill val where weight w ~ 0 by smooth interpolation (push-pull pyramid). val (H,W), w (H,W)."""
    H, W = val.shape
    pv = [val * w]
    pw = [w.copy()]
    for _ in range(levels):
        v = pv[-1]; ww = pw[-1]
        if min(v.shape) < 4:
            break
        pv.append(cv2.pyrDown(v)); pw.append(cv2.pyrDown(ww))
    v = pv[-1] / np.maximum(pw[-1], 1e-6)
    for i in range(len(pv) - 2, -1, -1):
        up = cv2.pyrUp(v, dstsize=(pv[i].shape[1], pv[i].shape[0]))
        cw = np.clip(pw[i] * 4, 0, 1) if i > 0 else np.clip(pw[i], 0, 1)
        cur = pv[i] / np.maximum(pw[i], 1e-6)
        v = cur * cw + up * (1 - cw)
    return v.astype(np.float32)


from numba import njit, prange


@njit(cache=True, fastmath=True, parallel=True)
def _shade_k(alb, emi, a, z, n, pcx, pcy, f, h, amb, ambg, LP, LC, LF, LD):
    H, W = a.shape
    out = np.empty((H, W, 3), np.float32)
    NL = LP.shape[0]
    for i in prange(H):
        for j in range(W):
            aa = a[i, j]
            if aa < 1e-4:
                for c in range(3):
                    out[i, j, c] = emi[i, j, c]
                continue
            zz = z[i, j] / aa
            X = (j - pcx) * zz / f
            Y = h - (i - pcy) * zz / f
            Z = zz
            nx, ny, nz = n[i, j, 0], n[i, j, 1], n[i, j, 2]
            nl = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
            nx /= nl; ny /= nl; nz /= nl
            up = max(ny, 0.0)
            dn = max(-ny, 0.0)
            r0 = amb[i, j, 0] * (0.6 + 0.4 * up) + ambg[0] * dn
            r1 = amb[i, j, 1] * (0.6 + 0.4 * up) + ambg[1] * dn
            r2 = amb[i, j, 2] * (0.6 + 0.4 * up) + ambg[2] * dn
            for k in range(NL):
                dx = LP[k, 0] - X
                dy = LP[k, 1] - Y
                dz = LP[k, 2] - Z
                d2 = dx * dx + dy * dy + dz * dz
                if LF[k, 4] > 0 and d2 > LF[k, 4] * LF[k, 4] * 9.0:
                    continue
                inv = 1.0 / math.sqrt(d2 + 1e-6)
                wrap = LF[k, 1]
                ndl = (nx * dx + ny * dy + nz * dz) * inv
                ndl = (ndl + wrap) / (1.0 + wrap)
                if ndl <= 0:
                    continue
                if ndl > 1:
                    ndl = 1.0
                r = LF[k, 2]
                att = LF[k, 0] / (d2 + r * r)
                if LF[k, 3] > 0:
                    cosang = -(dx * LD[k, 0] + dy * LD[k, 1] + dz * LD[k, 2]) * inv
                    c0 = LD[k, 3]
                    c1 = LD[k, 4]
                    q = (cosang - c0) / (c1 - c0)
                    if q <= 0:
                        continue
                    if q > 1:
                        q = 1.0
                    att *= q * math.sqrt(q)
                if LF[k, 4] > 0:
                    att *= math.exp(-d2 / (LF[k, 4] * LF[k, 4]))
                w = ndl * att
                r0 += w * LC[k, 0]
                r1 += w * LC[k, 1]
                r2 += w * LC[k, 2]
            out[i, j, 0] = alb[i, j, 0] * r0 + emi[i, j, 0]
            out[i, j, 1] = alb[i, j, 1] * r1 + emi[i, j, 1]
            out[i, j, 2] = alb[i, j, 2] * r2 + emi[i, j, 2]
    return out


def pack_lights(lights):
    n = max(len(lights), 1)
    LP = np.zeros((n, 3), np.float32); LC = np.zeros((n, 3), np.float32)
    LF = np.zeros((n, 5), np.float32); LD = np.zeros((n, 5), np.float32)
    for k, L in enumerate(lights):
        LP[k] = L['pos']; LC[k] = L['color']
        LF[k] = (L['power'], L.get('wrap', 0.25), L.get('radius', 0.5), 1.0 if 'dir' in L else 0.0, L.get('reach', 0.0))
        if 'dir' in L:
            c0, c1 = L.get('cone', (0.3, 0.8))
            LD[k] = (L['dir'][0], L['dir'][1], L['dir'][2], c0, c1)
    return LP, LC, LF, LD


def shade(cam, alb, a, z, n, emi, lights, ambient, amb_ground=(0.0, 0.0, 0.0)):
    """Deferred diffuse lighting at plate resolution (1x). alb/emi premultiplied. Returns premult RGB."""
    LP, LC, LF, LD = pack_lights(lights)
    if not lights:
        LF[:, 0] = 0
    f32 = lambda x: np.ascontiguousarray(x, dtype=np.float32)
    return _shade_k(f32(alb), f32(emi), f32(a), f32(z), f32(n), np.float32(cam.pcx), np.float32(cam.pcy),
                    np.float32(cam.f), np.float32(cam.h), f32(np.broadcast_to(np.asarray(ambient, np.float32), alb.shape)),
                    f32(amb_ground), LP, LC, LF, LD)


def dof_blur(rgb, a, depth, focus, k, max_sigma=10.0, levels=(0.0, 0.8, 1.6, 3.0, 5.0, 8.0, 12.0)):
    """Gather DOF on a premultiplied plate using a (filled, smooth) depth map. sigma = k*|1/z-1/zf|."""
    sig = np.clip(k * np.abs(1.0 / np.maximum(depth, 0.05) - 1.0 / focus), 0, max_sigma)
    lv = [l for l in levels if l <= max_sigma + 1e-6]
    stack = []
    rgba = np.dstack([rgb, a])
    for s in lv:
        stack.append(rgba if s <= 0.05 else cv2.GaussianBlur(rgba, (0, 0), s))
    out = np.zeros_like(rgba)
    lvarr = np.array(lv, np.float32)
    for i in range(len(lv)):
        lo = lvarr[i - 1] if i > 0 else -1.0
        hi = lvarr[i + 1] if i + 1 < len(lv) else 1e9
        c = lvarr[i]
        w = np.where(sig <= c, np.clip((sig - lo) / max(c - lo, 1e-6), 0, 1) if i > 0 else 1.0,
                     np.clip((hi - sig) / max(hi - c, 1e-6), 0, 1))
        out += stack[i] * w[..., None]
    return out[..., :3], out[..., 3]


MIRROR_K = 1.0


class Plate:
    """A baked, re-projectable plate: premult RGB + alpha + smooth depth (+ mirrored twin)."""

    def __init__(self, cam, rgb, a, depth, name=''):
        self.cam = cam
        self.rgba = np.ascontiguousarray(np.dstack([rgb, a]).astype(np.float32))
        self.depth = depth.astype(np.float32)
        self.name = name
        self.mirror = None

    def make_mirror(self, streak=0.0):
        """Reflection of this plate in the (horizontal) ground plane: the virtual scene below the
        ground seen from the same camera. v_r = 2 f h / z - v (v = y - cy)."""
        cam = self.cam
        H, W = self.depth.shape
        xs, ys = cam.grid(1)
        v = ys - cam.pcy
        D = self.depth
        # initial guess: mirror about the horizon
        # MIRROR_K < 1 squashes the reflected heights (Y -> k Y) about the ground contact line, like a slightly
        # convex, rain-swollen road crown: tall hanging signs then land in frame as readable inverted copies
        kq = MIRROR_K
        sy = cam.pcy - v
        for _ in range(6):
            z = cv2.remap(D, xs, sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fz = cam.f * cam.h / np.maximum(z, 0.05)
            sy = cam.pcy + ((1 + kq) * fz - v) / kq
        m = cv2.remap(self.rgba, xs, sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))
        # only below the object's own ground contact (v > f h / z) is valid reflection
        valid = (v > cam.f * cam.h / np.maximum(z, 0.05) - 1.0).astype(np.float32)
        m *= valid[..., None]
        mp = Plate(cam, m[..., :3], m[..., 3], z.astype(np.float32), self.name + '_mirror')
        self.mirror = mp
        return mp

    def warp(self, W, H, dX, dY, dZ, tilt=0.0, q=4, extra=None):
        """Sample the plate for camera offset (dX, dY, dZ) (+ tilt in radians, +up). -> (H, W, 4)."""
        mx, my = self.maps(W, H, dX, dY, dZ, tilt, q)
        out = cv2.remap(self.rgba, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        if extra is not None:
            return out, cv2.remap(extra, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return out

    def maps(self, W, H, dX, dY, dZ, tilt=0.0, q=4):
        cam = self.cam
        key = (W, H, q)
        if not hasattr(Plate, '_g') or Plate._g.get(key) is None:
            Plate._g = getattr(Plate, '_g', {})
            gx = (np.arange(W // q, dtype=np.float32) + 0.5) * (W / (W // q)) - 0.5
            gy = (np.arange(H // q, dtype=np.float32) + 0.5) * (H / (H // q)) - 0.5
            xs, ys = np.meshgrid(gx, gy)
            Plate._g[key] = (xs.astype(np.float32), ys.astype(np.float32))
        xs, ys = Plate._g[key]
        u = xs - cam.cx
        v = ys - cam.cy - tilt * cam.f
        z = cv2.remap(self.depth, xs + cam.mx, ys + cam.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        z = np.maximum(z, dZ + 0.2)
        # one refinement with depth at the source position
        for _ in range(2):
            k = (z - dZ) / z
            sx = u * k + cam.f * dX / z + cam.pcx
            sy = v * k - cam.f * dY / z + cam.pcy
            z = np.maximum(cv2.remap(self.depth, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE), dZ + 0.2)
        k = (z - dZ) / z
        sx = u * k + cam.f * dX / z + cam.pcx
        sy = v * k - cam.f * dY / z + cam.pcy
        mx = cv2.resize(sx.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        my = cv2.resize(sy.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        return mx, my


def bake(cv, lights, ambient, amb_ground=(0, 0, 0), fog=None, dof=None, name='', emissive_only=False,
         glow=None):
    """Canvas (supersampled) -> Plate at 1x: downsample, light, fog, DOF."""
    cam = cv.cam
    s = cam.ss
    size = (cam.PW, cam.PH)
    def dn(x):
        return cv2.resize(x, size, interpolation=cv2.INTER_AREA) if s > 1 else x
    alb, emi, a, z, n = dn(cv.alb), dn(cv.emi), dn(cv.a), dn(cv.z), dn(cv.n)
    rgb = shade(cam, alb, a, z, n, emi, lights, ambient, amb_ground)
    zr = z / np.maximum(a, 1e-6)
    depth = pushpull_fill(zr, np.clip(a * 4, 0, 1) ** 2)
    depth = np.where(a > 0.98, zr, depth).astype(np.float32)
    depth = cv2.GaussianBlur(depth, (0, 0), 1.0)
    if fog is not None:
        fc, zf, fmax = fog[:3]
        z0 = fog[3] if len(fog) > 3 else 0.0
        desat = fog[4] if len(fog) > 4 else 0.0
        fa = (1 - np.exp(-np.maximum(depth - z0, 0) / zf)) * fmax
        if desat > 0:
            # distant neon loses saturation and punch in the rain haze (aerial perspective)
            lum = (rgb * np.float32([0.3, 0.5, 0.2])).sum(-1, keepdims=True)
            k = (fa * desat)[..., None]
            rgb = rgb * (1 - k) + lum * k
            # HDR emissive is compressed so it cannot punch through the fog saturated
            m = rgb.max(-1, keepdims=True)
            comp = 1.0 / (1.0 + np.maximum(m - 0.6, 0) * (fa[..., None] * 1.5))
            rgb = rgb * comp
        rgb = rgb * (1 - fa[..., None]) + np.asarray(fc, np.float32) * (fa * a)[..., None]
    if glow is not None:
        # glow volumes: light scattered by rain haze around every bright source, added (alpha untouched)
        g_thr, g_amt, g_rad = glow
        br = np.maximum(rgb - g_thr * a[..., None], 0)
        sm = cv2.resize(br, (cam.PW // 4, cam.PH // 4), interpolation=cv2.INTER_AREA)
        acc = np.zeros_like(sm)
        for r, wt in zip(g_rad, (0.5, 0.35, 0.25)):
            acc += cv2.GaussianBlur(sm, (0, 0), r * cam.W / 4) * wt
        acc = cv2.resize(acc, (cam.PW, cam.PH), interpolation=cv2.INTER_LINEAR)
        if fog is not None:
            fa2 = (1 - np.exp(-np.maximum(depth - z0, 0) / zf)) * fmax
            acc = acc * (1 - 0.5 * fa2[..., None])
        rgb = rgb + acc * g_amt
    if dof is not None:
        focus, k = dof
        rgb, a = dof_blur(rgb, a, depth, focus, k)
    return Plate(cam, rgb, a, depth, name)


# ----------------------------------------------------------------------------- per-frame kernels

@njit(cache=True, fastmath=True, parallel=True)
def over_pm(dst, src):
    """In place: dst = dst * (1 - a) + rgb  (src premultiplied RGBA)."""
    H, W = dst.shape[0], dst.shape[1]
    for i in prange(H):
        for j in range(W):
            a = src[i, j, 3]
            if a < 0.0:
                a = 0.0
            if a > 1.0:
                a = 1.0
            k = 1.0 - a
            dst[i, j, 0] = dst[i, j, 0] * k + src[i, j, 0]
            dst[i, j, 1] = dst[i, j, 1] * k + src[i, j, 1]
            dst[i, j, 2] = dst[i, j, 2] * k + src[i, j, 2]


@njit(cache=True, fastmath=True, parallel=True)
def ground_combine(img, g, ex, streak, sharp, rings, glow, crowns):
    """img = sky*(1-ga) + g_rgb + ga * reflections.  ex: puddle, refl_coef, refl_puddle."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            ga = g[i, j, 3]
            k = 1.0 - ga
            pud = ex[i, j, 0]
            rc = ex[i, j, 1] * (1.0 - pud) * (0.3 + 1.1 * ex[i, j, 3])
            rp = ex[i, j, 2] * pud
            rr = rings[i, j] * (0.35 + 0.65 * pud)
            for c in range(3):
                refl = streak[i, j, c] * rc + sharp[i, j, c] * rp + rr * (0.05 + 1.5 * glow[i, j, c])
                img[i, j, c] = img[i, j, c] * k + g[i, j, c] + ga * refl + crowns[i, j] * (0.05 + 1.3 * glow[i, j, c])


@njit(cache=True, fastmath=True, parallel=True)
def rain_add(img, light, tiles, offs, shifts, gains):
    """img += sum_k tile_k[(y - off_k) % th, (x + shift_k) % W] * gain_k * light."""
    H, W = img.shape[0], img.shape[1]
    nk = tiles.shape[0]
    th = tiles.shape[1]
    for i in prange(H):
        for j in range(W):
            s = 0.0
            for k in range(nk):
                yy = (i - offs[k]) % th
                xx = (j + shifts[k]) % W
                s += tiles[k, yy, xx] * gains[k]
            if s > 0.0:
                for c in range(3):
                    img[i, j, c] += s * light[i, j, c]


@njit(cache=True, fastmath=True, parallel=True)
def grade(img, gamma, shadow_tint, sat):
    """In place: contrast (power), cool shadow lift, saturation."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            r = max(img[i, j, 0], 0.0)
            g = max(img[i, j, 1], 0.0)
            b = max(img[i, j, 2], 0.0)
            r = r ** gamma
            g = g ** gamma
            b = b ** gamma
            l = 0.2126 * r + 0.7152 * g + 0.0722 * b
            sh = 1.0 - min(l / 0.3, 1.0)
            sh = sh * sh
            r = l + (r - l) * sat + shadow_tint[0] * sh
            g = l + (g - l) * sat + shadow_tint[1] * sh
            b = l + (b - l) * sat + shadow_tint[2] * sh
            img[i, j, 0] = r
            img[i, j, 1] = g
            img[i, j, 2] = b


@njit(cache=True, fastmath=True, parallel=True)
def rain_add2(img, light, hi, far, tiles, offs, shifts, gains, modes):
    """Rain layers: mode 0 lit by the local scene light, mode 1 the same but faded toward the vanishing
    point (far mask), mode 2 = curtains/sheets, lit only by bright sources behind them (hi)."""
    H, W = img.shape[0], img.shape[1]
    nk = tiles.shape[0]
    th = tiles.shape[1]
    for i in prange(H):
        for j in range(W):
            s0 = 0.0
            s2 = 0.0
            for k in range(nk):
                v = tiles[k, (i - offs[k]) % th, (j + shifts[k]) % W] * gains[k]
                if modes[k] == 0:
                    s0 += v
                elif modes[k] == 1:
                    s0 += v * far[i, j]
                else:
                    s2 += v
            if s0 > 0.0 or s2 > 0.0:
                for c in range(3):
                    img[i, j, c] += s0 * light[i, j, c] + s2 * hi[i, j, c]


@njit(cache=True, fastmath=True, parallel=True)
def ground_combine2(img, g, ex, streak, sharp, soft, rings, glow, crowns):
    """Wet-road composite. ex: puddle, refl_streak, refl_puddle, breakup, dry, refl_crisp.
    Puddles and the wet film carry a crisp (ripple-displaced) mirror image; only the dry-ish patches between
    them keep the long vertical neon smear."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            ga = g[i, j, 3]
            k = 1.0 - ga
            pud = ex[i, j, 0]
            dry = ex[i, j, 4]
            film = 1.0 - pud
            a_st = ex[i, j, 1] * film * (0.3 + 1.1 * ex[i, j, 3]) * dry
            a_cr = ex[i, j, 5] * film * (1.0 - dry)
            rp = ex[i, j, 2] * pud
            rr = rings[i, j] * (0.35 + 0.65 * pud)
            for c in range(3):
                refl = (streak[i, j, c] * (a_st + 0.22 * a_cr) + soft[i, j, c] * a_cr + sharp[i, j, c] * rp +
                        rr * (0.05 + 1.5 * glow[i, j, c]))
                img[i, j, c] = img[i, j, c] * k + g[i, j, c] + ga * refl + crowns[i, j] * (0.05 + 1.3 * glow[i, j, c])
