"""Painting helpers for s06_seaside: a premultiplied RGBA canvas with bbox-local anti-aliased shapes,
lit foliage clumps (hard-edged leaf masses with warm rim light), grass tufts."""
import math
import numpy as np
import cv2

from lib import core as C


def col(c):
    return C.hex2rgb(c) if isinstance(c, str) else np.asarray(c, np.float32)


class Canvas:
    """Premultiplied RGBA canvas. All drawing is clipped to bounding boxes for speed."""

    def __init__(self, W, H):
        self.W, self.H = W, H
        self.rgb = np.zeros((H, W, 3), np.float32)
        self.a = np.zeros((H, W), np.float32)
        # optional depth tracking (inverse depth 1/Z, premultiplied like rgb) for the camera truck:
        # zmode = ('c', iz) constant | ('Y', height_m) point height -> iz from the row | ('F', field) array
        self.cam = None          # (horizon_y, focal_px, camera_height_m)
        self.zmode = None
        self.iz = None

    # ------------------------------------------------------------------ masks
    def poly_mask(self, pts, ss=3):
        """Anti-aliased polygon coverage restricted to its bbox -> (mask, x0, y0) or None."""
        p = np.asarray(pts, np.float64)
        x0 = max(int(math.floor(p[:, 0].min())) - 1, 0)
        y0 = max(int(math.floor(p[:, 1].min())) - 1, 0)
        x1 = min(int(math.ceil(p[:, 0].max())) + 2, self.W)
        y1 = min(int(math.ceil(p[:, 1].max())) + 2, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        w, h = x1 - x0, y1 - y0
        m = np.zeros((h * ss, w * ss), np.uint8)
        q = ((p - [x0, y0]) * ss * 16).astype(np.int32)
        cv2.fillPoly(m, [q], 255, lineType=cv2.LINE_AA, shift=4)
        m = cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
        return m, x0, y0

    def line_mask(self, pts, width, ss=4):
        p = np.asarray(pts, np.float64)
        pad = width + 3
        x0 = max(int(math.floor(p[:, 0].min() - pad)), 0)
        y0 = max(int(math.floor(p[:, 1].min() - pad)), 0)
        x1 = min(int(math.ceil(p[:, 0].max() + pad)), self.W)
        y1 = min(int(math.ceil(p[:, 1].max() + pad)), self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        w, h = x1 - x0, y1 - y0
        m = np.zeros((h * ss, w * ss), np.uint8)
        q = ((p - [x0, y0]) * ss * 16).astype(np.int32)
        th = max(1, int(round(width * ss)))
        cv2.polylines(m, [q], False, 255, th, cv2.LINE_AA, shift=4)
        m = cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
        if width * ss < 1:
            m *= width * ss
        return m, x0, y0

    # ------------------------------------------------------------------ compositing
    def put(self, mres, color, alpha=1.0):
        """Composite colour (rgb or (h,w,3) image matching the mask crop) with coverage mask."""
        if mres is None:
            return
        m, x0, y0 = mres
        h, w = m.shape
        a = m * alpha
        c = np.asarray(color, np.float32)
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))
        if c.ndim == 1:
            src = a[..., None] * c
        else:
            src = a[..., None] * c
        self.rgb[sl] = src + self.rgb[sl] * (1 - a[..., None])
        if self.zmode is not None:
            if self.iz is None:
                self.iz = np.zeros((self.H, self.W), np.float32)
            kind, v = self.zmode
            if kind == 'c':
                zs = np.float32(v)
            elif kind == 'Y':
                hz, f, hc = self.cam
                ys = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
                den = f * (hc - v)
                zs = np.clip((ys - hz) / den, 0.0, 0.5) * np.ones((1, w), np.float32)
            else:
                zs = v[sl]
            self.iz[sl] = a * zs + self.iz[sl] * (1 - a)
        self.a[sl] = a + self.a[sl] * (1 - a)

    def poly(self, pts, color, alpha=1.0, ss=3):
        r = self.poly_mask(pts, ss)
        self.put(r, color, alpha)
        return r

    def line(self, pts, width, color, alpha=1.0, ss=4):
        r = self.line_mask(pts, width, ss)
        self.put(r, color, alpha)
        return r

    def grid(self, x0, y0, w, h):
        ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
        return xs, ys

    def iz_field(self, sigma=2.0, fill=0.0):
        """straight inverse depth, smoothed and filled into empty areas (normalised convolution)."""
        if self.iz is None:
            return np.full((self.H, self.W), fill, np.float32)
        a = np.clip(self.a, 0, 1)
        num = cv2.GaussianBlur(self.iz, (0, 0), sigma)
        den = cv2.GaussianBlur(a, (0, 0), sigma)
        out = num / np.maximum(den, 1e-4)
        # far fill for empty regions: heavily blurred version
        q = 8
        small_n = cv2.resize(self.iz, (self.W // q, self.H // q), interpolation=cv2.INTER_AREA)
        small_d = cv2.resize(a, (self.W // q, self.H // q), interpolation=cv2.INTER_AREA)
        bn = cv2.GaussianBlur(small_n, (0, 0), 12) ; bd = cv2.GaussianBlur(small_d, (0, 0), 12)
        big = cv2.resize(bn / np.maximum(bd, 1e-4), (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        big = np.where(cv2.resize(bd, (self.W, self.H)) > 1e-3, big, fill)
        wgt = np.clip(den / 0.05, 0, 1)
        return (out * wgt + big * (1 - wgt)).astype(np.float32)

    def straight(self):
        a = np.clip(self.a, 0, 1)
        rgb = self.rgb / np.maximum(a, 1e-4)[..., None]
        return np.dstack([rgb, a]).astype(np.float32)


# ============================================================================ foliage

def _lobe(cv, lx, ly, lr, rng, pal, Lx, Ly, hard, rim_w, lit_amt, squash=0.9, scallop=0.2, nsc=None):
    """One leaf lobe: ellipse with a scalloped (leafy) silhouette, sphere-normal shading quantised into
    painted steps, dark crescent at the underside, warm rim on the light-facing edge."""
    R = lr * (1 + scallop) + 2
    x0 = int(max(math.floor(lx - R), 0)); x1 = int(min(math.ceil(lx + R) + 1, cv.W))
    y0 = int(max(math.floor(ly - R), 0)); y1 = int(min(math.ceil(ly + R) + 1, cv.H))
    if x1 <= x0 or y1 <= y0:
        return
    xs, ys = cv.grid(x0, y0, x1 - x0, y1 - y0)
    dx, dy = xs - lx, (ys - ly) / squash
    d = np.sqrt(dx * dx + dy * dy)
    # scallops: small circles along the rim, bigger on the top side
    cov = np.clip(lr * (1 - scallop * 0.6) - d + 0.5, 0, 1)
    if nsc is None:
        nsc = int(np.clip(lr * 0.9, 7, 28))
    ang = np.sort(rng.uniform(0, 2 * math.pi, nsc))
    for a in ang:
        top = 0.6 + 0.4 * max(-math.sin(a), 0)
        sr = lr * scallop * rng.uniform(0.7, 1.3) * (0.8 + 0.6 * top)
        cxs = lx + math.cos(a) * (lr - sr * 0.9)
        cys = ly + math.sin(a) * (lr - sr * 0.9) * squash
        dd = np.sqrt((xs - cxs) ** 2 + (ys - cys) ** 2)
        cov = np.maximum(cov, np.clip(sr - dd + 0.5, 0, 1))
    if cov.max() <= 0:
        return
    nx = np.clip(dx / lr, -1, 1)
    ny = np.clip(dy / lr, -1, 1)
    r = np.sqrt(np.clip(nx * nx + ny * ny, 0, 1.5))
    e = nx * Lx + ny * Ly                    # 2D facing toward the light
    # cel-painted lobe: flat base tone, a lighter crescent on the light side, a dark tuck on the far side
    base = pal['mid'] * rng.uniform(0.9, 1.08)
    t_lit = C.smoothstep(0.28 - hard * 2, 0.28 + hard * 2, e + 0.35 * (r - 0.6)) * lit_amt
    t_drk = C.smoothstep(-0.2 + hard * 2, -0.2 - hard * 2, e - 0.3 * (r - 0.5))
    c = base + (pal['lit'] - base) * t_lit[..., None]
    c = c + (pal['shd'] - c) * t_drk[..., None]
    # thin bright rim on the light-facing edge
    rim = C.smoothstep(1 - rim_w * 1.6, 1 - rim_w * 0.3, r) * C.smoothstep(0.45, 0.85, e) * lit_amt
    c = c + (pal['rim'] - c) * (rim * 0.9)[..., None]
    cv.put((cov, x0, y0), c)


def clump(cv, cx, cy, r, rng, pal, L=(1.0, -0.4), rim_w=0.12, n=None, flat=0.8, hard=0.05, lit_amt=1.0,
          levels=2, **_):
    """A leaf mass: a hierarchy of scalloped lobes (big lobes, then smaller lobes crowning them), each
    with its own lit cap, dark underside and rim, so the mass reads as layered foliage."""
    Lx, Ly = L
    ln = math.hypot(Lx, Ly) + 1e-9
    Lx, Ly = Lx / ln, Ly / ln
    if n is None:
        n = int(np.clip(4 + r * 0.08, 4, 9))
    lobes = []
    for i in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        rad = rng.uniform(0, 1) ** 0.7
        lx = cx + math.cos(ang) * r * rad * 0.7
        ly = cy + math.sin(ang) * r * rad * 0.7 * flat
        lr = r * rng.uniform(0.38, 0.55) * (1.1 - 0.3 * rad)
        lobes.append((ly, lx, lr))
    lobes.sort(key=lambda z: -z[0])
    def la(lx, ly):
        return lit_amt * float(np.clip(0.35 + ((lx - cx) * Lx + (ly - cy) * Ly) / r, 0.0, 1.0))
    for ly, lx, lr in lobes:
        _lobe(cv, lx, ly, lr, rng, pal, Lx, Ly, hard, rim_w, la(lx, ly))
    if levels > 1 and r > 6:
        kids = []
        for ly, lx, lr in lobes:
            for k in range(int(rng.integers(2, 5))):
                a = rng.uniform(-math.pi * 0.95, -0.05) if rng.random() < 0.8 else rng.uniform(0, math.pi)
                rr = lr * rng.uniform(0.3, 0.75)
                kr = lr * rng.uniform(0.28, 0.42)
                kids.append((ly + math.sin(a) * rr * flat, lx + math.cos(a) * rr, kr))
        kids.sort(key=lambda z: -z[0])
        for ly, lx, lr in kids:
            _lobe(cv, lx, ly, lr, rng, pal, Lx, Ly, hard, rim_w * 1.3, la(lx, ly), scallop=0.25)
        if levels > 2:
            # leaf dabs: small lobes breaking the silhouette, mostly on the light-facing side
            dabs = []
            for ly, lx, lr in kids:
                for k in range(int(rng.integers(0, 3))):
                    a = math.atan2(Ly, Lx) + rng.normal(0, 1.0)
                    rr = lr * rng.uniform(0.7, 1.0)
                    dabs.append((ly + math.sin(a) * rr * flat, lx + math.cos(a) * rr, lr * rng.uniform(0.3, 0.45)))
            dabs.sort(key=lambda z: -z[0])
            for ly, lx, lr in dabs:
                _lobe(cv, lx, ly, lr, rng, pal, Lx, Ly, hard, rim_w * 1.6, la(lx, ly), scallop=0.3)


def _put_w(wcv, res, by, hh):
    """write a sway weight (0 at the base .. 1 at the tip) for a shape into the weight canvas."""
    if wcv is None or res is None:
        return
    m, x0, y0 = res
    h, w = m.shape
    ys = np.arange(y0, y0 + h, dtype=np.float32)[:, None]
    wt = np.clip((by - ys) / max(hh, 1e-3), 0, 1) * np.ones((1, w), np.float32)
    wcv.put(res, np.repeat(wt[..., None], 3, -1))


def grass_tuft(cv, bx, by, hgt, rng, color, rimc, L=(1.0, -0.3), n=None, lean=0.0, width=1.0, wcv=None,
               rim_amt=0.7):
    """Tapered blades from a base point. Blades facing the light get a thin warm rim."""
    if n is None:
        n = int(rng.integers(5, 12))
    for i in range(n):
        hh = hgt * rng.uniform(0.5, 1.0)
        ang = rng.normal(lean, 0.35)
        bend = rng.normal(0, 0.25)
        x0 = bx + rng.normal(0, hgt * 0.08)
        tt = np.linspace(0, 1, 9)
        xs = x0 + math.sin(ang) * hh * tt + bend * hh * tt ** 2 * 0.5
        ys = by - math.cos(ang) * hh * tt
        wb = max(hgt * 0.05 * width, 0.8)
        left = np.stack([xs - wb * (1 - tt), ys], 1)
        right = np.stack([xs + wb * (1 - tt), ys], 1)[::-1]
        pts = np.concatenate([left, right])
        r = cv.poly_mask(pts, ss=3)
        if r is None:
            continue
        cv.put(r, color * rng.uniform(0.85, 1.1))
        _put_w(wcv, r, by, hgt)
        if L[0] * (ang + bend) > 0.05 and rim_amt > 0 and hh > 12:
            rl = cv.line_mask(np.stack([xs + wb * 0.6 * (1 - tt), ys], 1)[4:], max(wb * 0.22, 0.45))
            cv.put(rl, rimc, rim_amt * min(1.0, hh / 60))
            _put_w(wcv, rl, by, hgt)


def susuki(cv, bx, by, hgt, rng, stem, plume, lean=0.15, wcv=None):
    """Japanese pampas grass: a curved stalk and a drooping, fanned feathery plume that glows when
    backlit. Leaves at the base."""
    tt = np.linspace(0, 1, 16)
    bend = rng.uniform(0.1, 0.3) * (1 if rng.random() < 0.8 else -1)
    xs = bx + lean * hgt * tt + bend * hgt * tt ** 2
    ys = by - hgt * tt
    wst = max(hgt * 0.008, 0.5)
    r = cv.line_mask(np.stack([xs, ys], 1), wst)
    cv.put(r, stem)
    _put_w(wcv, r, by, hgt)
    tx, ty = xs[-1], ys[-1]
    dirx = np.sign(bend + lean + 1e-3)
    ns = int(rng.integers(7, 13))
    for k in range(ns):
        st = rng.uniform(0.72, 0.98)
        sx = np.interp(st, tt, xs); sy = np.interp(st, tt, ys)
        L = hgt * rng.uniform(0.12, 0.22)
        a0 = -math.pi / 2 + dirx * rng.uniform(0.2, 0.9)
        u = np.linspace(0, 1, 10)
        ang = a0 + dirx * u * rng.uniform(0.3, 0.9)
        px = sx + np.cumsum(np.cos(ang)) * L / 10
        py = sy + np.cumsum(np.sin(ang)) * L / 10 + (u ** 2) * L * 0.25
        w = max(hgt * 0.012 * rng.uniform(0.8, 1.4), 0.6)
        rr = cv.line_mask(np.stack([px, py], 1), w)
        cv.put(rr, plume * rng.uniform(0.85, 1.1), 0.75)
        _put_w(wcv, rr, by, hgt)
    grass_tuft(cv, bx, by, hgt * 0.45, rng, stem * 0.9, plume * 0.8, n=4, lean=0.2, wcv=wcv)
