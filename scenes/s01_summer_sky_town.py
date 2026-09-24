"""Foreground plates for shot s01: backlit Japanese residential rooftops, a concrete utility pole with
crossarms / insulators / transformer, and sagging power lines.

Everything is painted element by element into straight-alpha RGBA plates with bbox-limited supersampled
masks (crisp, anti-aliased hard-edged silhouettes). The sun is ahead of the camera, high behind the
cloud: faces toward us are in cool sky-lit shadow, top edges / ridges / wires catch a warm rim light,
tile roofs reflect the bright sky.
"""
import math
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _c(*v):
    return np.array(v, np.float32)


class Canvas:
    """Premultiplied RGBA canvas covering plate rows [y0, y0 + h) (plate coordinates)."""

    def __init__(self, W, H, y0=0):
        self.W, self.H, self.y0 = W, H, int(y0)
        self.col = np.zeros((H, W, 3), np.float32)
        self.a = np.zeros((H, W), np.float32)

    # ---------------------------------------------------------------- primitives
    def _bbox(self, pts, pad):
        x0 = int(math.floor(pts[:, 0].min() - pad))
        x1 = int(math.ceil(pts[:, 0].max() + pad)) + 1
        y0 = int(math.floor(pts[:, 1].min() - pad)) - self.y0
        y1 = int(math.ceil(pts[:, 1].max() + pad)) + 1 - self.y0
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.W), min(y1, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _mask(self, bb, draw, ss=4):
        x0, y0, x1, y1 = bb
        m = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
        draw(m, ss, x0, y0 + self.y0)
        return cv2.resize(m.astype(np.float32) / 255.0, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)

    def _put(self, bb, m, color, opacity=1.0):
        x0, y0, x1, y1 = bb
        if callable(color):
            gy, gx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            c = color(gx, gy + self.y0)
        else:
            c = np.asarray(color, np.float32)
        m = m * opacity
        r = (slice(y0, y1), slice(x0, x1))
        self.col[r] = self.col[r] * (1 - m[..., None]) + c * m[..., None]
        self.a[r] = self.a[r] + m * (1 - self.a[r])

    def poly(self, pts, color, opacity=1.0, ss=4):
        pts = np.asarray(pts, np.float64)
        bb = self._bbox(pts, 2)
        if bb is None:
            return None

        def draw(m, s, ox, oy):
            p = ((pts - [ox, oy]) * s * 16).astype(np.int32)
            cv2.fillPoly(m, [p], 255, cv2.LINE_AA, shift=4)
        m = self._mask(bb, draw, ss)
        self._put(bb, m, color, opacity)
        return bb, m

    def lines(self, polylines, width, color, opacity=1.0, ss=4):
        allp = np.concatenate([np.asarray(p, np.float64) for p in polylines])
        bb = self._bbox(allp, width + 2)
        if bb is None:
            return None

        def draw(m, s, ox, oy):
            for p in polylines:
                q = ((np.asarray(p, np.float64) - [ox, oy]) * s * 16).astype(np.int32)
                cv2.polylines(m, [q], False, 255, max(1, int(round(width * s))), cv2.LINE_AA, shift=4)
        m = self._mask(bb, draw, ss)
        self._put(bb, m, color, opacity)
        return bb, m

    def ellipse(self, cx, cy, rx, ry, color, opacity=1.0, ss=4, ang=0.0):
        pts = np.array([[cx - rx, cy - ry], [cx + rx, cy + ry]])
        bb = self._bbox(pts, 2)
        if bb is None:
            return None

        def draw(m, s, ox, oy):
            cv2.ellipse(m, (int((cx - ox) * s * 16), int((cy - oy) * s * 16)),
                        (max(1, int(rx * s * 16)), max(1, int(ry * s * 16))), ang, 0, 360, 255, -1, cv2.LINE_AA,
                        shift=4)
        m = self._mask(bb, draw, ss)
        self._put(bb, m, color, opacity)
        return bb, m

    def poly_mask(self, bb, pts, ss=4):
        pts = np.asarray(pts, np.float64)

        def draw(m, s, ox, oy):
            p = ((pts - [ox, oy]) * s * 16).astype(np.int32)
            cv2.fillPoly(m, [p], 255, cv2.LINE_AA, shift=4)
        return self._mask(bb, draw, ss)

    def put(self, bb, m, color, opacity=1.0):
        self._put(bb, m, color, opacity)

    def rgba(self):
        a = np.clip(self.a, 0, 1)
        col = self.col / np.maximum(a, 1e-4)[..., None]
        return np.dstack([col, a]).astype(np.float32)


# ============================================================================ palette (backlit summer noon)

RIM = _c(1.35, 1.14, 0.84)            # warm rim light on top edges (crisp)
RIM_COOL = _c(0.55, 0.68, 0.88)       # sky-reflection highlight
SHADOW = _c(0.03, 0.06, 0.12)         # deepest shadow (eaves)
WALL = _c(0.07, 0.12, 0.2)
WALL_WARM = _c(0.12, 0.14, 0.22)
TILE_TOP = _c(0.14, 0.22, 0.34)
TILE_BOT = _c(0.06, 0.1, 0.18)
LEAF_D = _c(0.02, 0.07, 0.1)          # foliage core
LEAF_M = _c(0.04, 0.14, 0.17)         # foliage mid-tone
LEAF_L = _c(0.16, 0.34, 0.27)         # sun-side leaf mass
LEAF_H = _c(0.75, 0.9, 0.45)          # hot leaf rims
WIRE = _c(0.04, 0.06, 0.12)
POLE_D = _c(0.07, 0.1, 0.17)
POLE_L = _c(0.3, 0.34, 0.44)


# roof palettes (near ridge, near eave, sky-sheen): all backlit near-silhouettes in dark teal-navy
ROOFS = [(_c(0.13, 0.21, 0.34), _c(0.05, 0.09, 0.17), _c(0.34, 0.5, 0.7)),
         (_c(0.16, 0.22, 0.32), _c(0.07, 0.1, 0.17), _c(0.4, 0.52, 0.68)),
         (_c(0.2, 0.15, 0.22), _c(0.09, 0.07, 0.13), _c(0.45, 0.42, 0.56)),
         (_c(0.09, 0.22, 0.28), _c(0.04, 0.1, 0.15), _c(0.3, 0.52, 0.62)),
         (_c(0.14, 0.15, 0.24), _c(0.06, 0.07, 0.13), _c(0.38, 0.44, 0.6))]
WALLS = [_c(0.08, 0.13, 0.22), _c(0.1, 0.14, 0.23), _c(0.12, 0.13, 0.21), _c(0.07, 0.12, 0.2), _c(0.13, 0.15, 0.24)]


def haze(c, k, hz):
    return c * (1 - k) + hz * k


# ============================================================================ houses

def tile_roof(cv, pts_face, x_left, x_right, y_ridge, y_eave, s, rng, hz, hk, tint=0.0, pal=None):
    """Front roof face with vertical kawara rows, course lines, sky sheen, rim along the ridge."""
    col_step = max(2.2, 0.55 * s)
    ph = rng.uniform(0, 6.28)
    pal = pal or ROOFS[0]
    top = haze(pal[0] * (1 + tint), hk, hz)
    bot = haze(pal[1] * (1 + tint), hk, hz)
    sheen = haze(pal[2], hk, hz)

    def color(gx, gy):
        v = np.clip((gy - y_ridge) / max(y_eave - y_ridge, 1), 0, 1)
        c = top + (bot - top) * v[..., None] ** 0.8
        rib = 0.5 + 0.5 * np.cos(2 * math.pi * gx / col_step + ph)
        c = c * (1 - 0.1 * rib[..., None] ** 3)
        course = (np.mod(gy - y_ridge, col_step * 1.1) < 0.9).astype(np.float32)
        c = c * (1 - 0.07 * course[..., None])
        sh = np.exp(-((v - 0.22) / 0.17) ** 2) * (0.7 + 0.3 * np.sin(gx * 0.011 + ph))
        c = c + (sheen * 1.15 - c) * (sh * 0.85)[..., None]
        return c
    cv.poly(pts_face, color)
    # ridge cap + rim light
    cv.lines([[(x_left, y_ridge), (x_right, y_ridge)]], max(1.5, 0.45 * s), haze(_c(0.16, 0.2, 0.33), hk, hz))
    cv.lines([[(x_left, y_ridge - 0.25 * s), (x_right, y_ridge - 0.25 * s)]], max(1.0, 0.18 * s),
             haze(RIM, hk * 0.6, hz))


def house(cv, x, y_eave, w, s, rng, hz, hk=0.0, kind='hip', wall_h=None, roof=None, wall=None):
    """One house seen across the rooftops. s = detail scale (px). y_eave = eave line (plate px).
    Returns the ridge / apex y."""
    roof_h = w * rng.uniform(0.2, 0.27)
    y_r = y_eave - roof_h
    wall_h = wall_h if wall_h is not None else roof_h * 3
    pal = ROOFS[roof if roof is not None else int(rng.integers(0, len(ROOFS)))]
    wc = WALLS[wall if wall is not None else int(rng.integers(0, len(WALLS)))]
    wall_c = haze(wc, hk, hz)
    xl, xr = x - w * 0.45, x + w * 0.45

    def wall_col(gx, gy):
        v = np.clip((gy - y_eave) / max(wall_h, 1), 0, 1)
        c = wall_c * (0.78 + 0.22 * _ss(0.0, 0.5, v))[..., None]
        return c * (1 - 0.05 * np.clip((gx - xl) / (xr - xl), 0, 1))[..., None]
    cv.poly([(xl, y_eave), (xr, y_eave), (xr, y_eave + wall_h), (xl, y_eave + wall_h)], wall_col)
    nwin = int(rng.integers(1, 3))
    for k in range(nwin):
        wx = x + (k - (nwin - 1) / 2) * w * 0.38 + rng.uniform(-0.05, 0.05) * w
        ww, wh = w * rng.uniform(0.14, 0.22), roof_h * rng.uniform(0.7, 1.0)
        wy = y_eave + roof_h * rng.uniform(0.35, 0.55)
        glass = haze(_c(0.05, 0.09, 0.18), hk, hz)
        refl = haze(_c(0.4, 0.55, 0.78), hk, hz)

        def gcol(gx, gy, wx=wx, wy=wy, ww=ww, wh=wh, glass=glass, refl=refl):
            f = ((gx - wx) / ww + (gy - wy) / wh * 0.7)
            k_ = np.exp(-((f + 0.1) / 0.16) ** 2) * 0.55 + np.exp(-((f - 0.45) / 0.06) ** 2) * 0.35
            c = glass + (refl - glass) * k_[..., None]
            return c * (1 - 0.35 * _ss(wy + wh * 0.35, wy, gy))[..., None]
        cv.poly([(wx - ww / 2, wy), (wx + ww / 2, wy), (wx + ww / 2, wy + wh), (wx - ww / 2, wy + wh)], gcol)
        fr = haze(_c(0.22, 0.26, 0.36), hk, hz)
        cv.lines([[(wx - ww / 2, wy), (wx + ww / 2, wy), (wx + ww / 2, wy + wh), (wx - ww / 2, wy + wh), (wx - ww / 2, wy)],
                  [(wx, wy), (wx, wy + wh)]], max(1.0, 0.22 * s), fr, 0.9)
        if rng.random() < 0.55:
            ry = wy + wh * 0.55
            bw = ww * 1.5
            cv.poly([(wx - bw / 2, ry), (wx + bw / 2, ry), (wx + bw / 2, wy + wh + 0.05 * roof_h),
                     (wx - bw / 2, wy + wh + 0.05 * roof_h)], haze(_c(0.1, 0.13, 0.22), hk, hz), 0.55)
            rails = [[(wx - bw / 2 + bw * i / 14, ry), (wx - bw / 2 + bw * i / 14, wy + wh)] for i in range(15)]
            cv.lines(rails, max(0.6, 0.12 * s), haze(_c(0.2, 0.25, 0.4), hk, hz), 0.9)
            cv.lines([[(wx - bw / 2, ry), (wx + bw / 2, ry)]], max(1.0, 0.25 * s), haze(RIM, hk * 0.5 + 0.1, hz))
            ly = ry - wh * 0.45
            cv.lines([[(wx - bw * 0.45, ly), (wx + bw * 0.45, ly)]], max(0.7, 0.14 * s), haze(_c(0.3, 0.32, 0.4), hk, hz))
            for j in range(int(rng.integers(0, 4))):
                lx = wx - bw * 0.35 + j * bw * 0.22
                cl = [_c(0.92, 0.9, 0.9), _c(0.55, 0.7, 0.9), _c(0.95, 0.8, 0.7)][j % 3] * 0.35
                cv.poly([(lx, ly), (lx + bw * 0.16, ly), (lx + bw * 0.15, ly + wh * 0.3), (lx + bw * 0.01, ly + wh * 0.3)],
                        haze(cl, hk, hz), 0.95)
        if rng.random() < 0.4:
            ax, ay = wx + ww * 0.8, wy + wh * 0.65
            aw, ah = w * 0.09, w * 0.065
            cv.poly([(ax, ay), (ax + aw, ay), (ax + aw, ay + ah), (ax, ay + ah)], haze(_c(0.2, 0.23, 0.32), hk, hz))
            cv.ellipse(ax + aw * 0.4, ay + ah * 0.5, ah * 0.33, ah * 0.33, haze(_c(0.3, 0.33, 0.45), hk, hz))
            cv.lines([[(ax, ay), (ax + aw, ay)]], max(0.8, 0.15 * s), haze(RIM, hk * 0.5, hz), 0.8)
    cv.poly([(xl, y_eave), (xr, y_eave), (xr, y_eave + roof_h * 0.2), (xl, y_eave + roof_h * 0.2)],
            haze(SHADOW, hk, hz), 0.75)
    if kind == 'hip':
        rl = w * rng.uniform(0.18, 0.3)
        face = [(x - w / 2, y_eave), (x + w / 2, y_eave), (x + w / 2 - rl, y_r), (x - w / 2 + rl, y_r)]
        tile_roof(cv, face, x - w / 2 + rl, x + w / 2 - rl, y_r, y_eave, s, rng, hz, hk, pal=pal)
        for sgn in (-1, 1):
            p0 = (x + sgn * (w / 2 - rl), y_r)
            p1 = (x + sgn * w / 2, y_eave)
            cv.lines([[p0, p1]], max(1.4, 0.45 * s), haze(pal[1] * 0.8, hk, hz))
            cv.lines([[(p0[0], p0[1] - 0.25 * s), (p1[0], p1[1] - 0.25 * s)]], max(0.8, 0.18 * s),
                     haze(RIM, hk * 0.6, hz), opacity=0.9 if sgn < 0 else 0.4)
        top_y = y_r
    elif kind == 'gable':
        face = [(x - w / 2, y_eave), (x + w / 2, y_eave), (x + w / 2 - w * 0.03, y_r), (x - w / 2 + w * 0.03, y_r)]
        tile_roof(cv, face, x - w / 2 + w * 0.03, x + w / 2 - w * 0.03, y_r, y_eave, s, rng, hz, hk, pal=pal)
        for sgn in (-1, 1):
            cv.lines([[(x + sgn * (w / 2 - w * 0.03), y_r), (x + sgn * w / 2, y_eave)]], max(1.5, 0.5 * s),
                     haze(pal[1] * 0.8, hk, hz))
        for sgn in (-1, 1):
            cv.ellipse(x + sgn * (w / 2 - w * 0.03), y_r - 0.3 * s, 0.8 * s, 0.6 * s, haze(pal[1], hk, hz))
        top_y = y_r
    else:
        apex = (x, y_r - roof_h * 0.35)
        cv.poly([(x - w * 0.47, y_eave + roof_h * 0.12), apex, (x + w * 0.47, y_eave + roof_h * 0.12)], wall_col)
        el = (x - w * 0.49, y_eave + roof_h * 0.1)
        er = (x + w * 0.49, y_eave + roof_h * 0.1)
        cv.lines([[el, apex, er]], max(2.0, 0.9 * s), haze(pal[1], hk, hz))
        cv.lines([[(el[0], el[1] - 0.45 * s), (apex[0], apex[1] - 0.45 * s)]],
                 max(0.9, 0.22 * s), haze(RIM, hk * 0.6, hz))
        cv.lines([[(apex[0], apex[1] - 0.45 * s), (er[0], er[1] - 0.45 * s)]],
                 max(0.7, 0.14 * s), haze(RIM, hk * 0.6, hz), 0.45)
        vy = apex[1] + roof_h * 0.45
        cv.poly([(x - w * 0.045, vy), (x + w * 0.045, vy), (x + w * 0.045, vy + roof_h * 0.18),
                 (x - w * 0.045, vy + roof_h * 0.18)], haze(SHADOW, hk, hz))
        cv.lines([[(x - w * 0.045, vy + roof_h * 0.18 * i / 4), (x + w * 0.045, vy + roof_h * 0.18 * i / 4)] for i in range(1, 4)],
                 max(0.6, 0.12 * s), haze(wc * 0.9, hk, hz))
        top_y = apex[1]
    cv.lines([[(x - w * 0.5, y_eave + 0.3 * s), (x + w * 0.5, y_eave + 0.3 * s)]], max(1.2, 0.5 * s),
             haze(_c(0.2, 0.24, 0.38), hk, hz))
    cv.lines([[(x - w * 0.5, y_eave), (x + w * 0.5, y_eave)]], max(0.7, 0.14 * s), haze(RIM_COOL, hk * 0.6, hz), 0.7)
    return top_y


def antenna(cv, x, y_base, h, s, rng, hz, hk):
    """TV Yagi antenna on a mast with guy wires."""
    top = y_base - h
    cv.lines([[(x, y_base), (x, top)]], max(1.0, 0.28 * s), haze(_c(0.14, 0.17, 0.28), hk, hz))
    cv.lines([[(x - 0.12 * s, y_base), (x - 0.12 * s, top)]], max(0.7, 0.1 * s), haze(RIM, hk * 0.5, hz), 0.7)
    for k, (yy, L) in enumerate([(0.0, 1.0), (0.22, 0.7)]):
        by = top + yy * h
        boom = [(x - L * h * 0.55, by), (x + L * h * 0.45, by)]
        cv.lines([boom], max(0.9, 0.2 * s), haze(_c(0.14, 0.17, 0.28), hk, hz))
        n = int(8 * L)
        els = []
        for i in range(n):
            ex = boom[0][0] + (boom[1][0] - boom[0][0]) * i / max(n - 1, 1)
            el = h * 0.12 * (1 - 0.4 * i / n)
            els.append([(ex, by - el), (ex, by + el)])
        cv.lines(els, max(0.7, 0.13 * s), haze(_c(0.14, 0.17, 0.28), hk, hz))
        cv.lines([[(boom[0][0], by - 0.12 * s), (boom[1][0], by - 0.12 * s)]], max(0.6, 0.08 * s),
                 haze(RIM, hk * 0.5, hz), 0.8)
    # guy wires
    cv.lines([[(x, top + h * 0.3), (x - h * 0.5, y_base + 0.1 * h)], [(x, top + h * 0.3), (x + h * 0.45, y_base + 0.1 * h)]],
             max(0.5, 0.06 * s), haze(_c(0.2, 0.25, 0.38), hk, hz), 0.6)


def solar_heater(cv, x, y, w, s, hz, hk):
    """Rooftop solar water heater: tilted dark-blue panel with a tank on top, glinting."""
    h = w * 0.35
    cv.poly([(x - w / 2, y), (x + w / 2, y), (x + w / 2 - w * 0.06, y - h), (x - w / 2 + w * 0.06, y - h)],
            lambda gx, gy: haze(_c(0.12, 0.2, 0.4) + _c(0.4, 0.5, 0.6) * np.exp(-((gx - x + w * 0.1) / (w * 0.15)) ** 2)[..., None] * 0.6, hk, hz))
    for i in range(1, 6):
        xx = x - w / 2 + w * i / 6
        cv.lines([[(xx, y), (xx + (x - xx) * 0.12, y - h)]], max(0.6, 0.1 * s), haze(_c(0.3, 0.42, 0.6), hk, hz), 0.6)
    cv.poly([(x - w / 2 + w * 0.05, y - h), (x + w / 2 - w * 0.05, y - h), (x + w / 2 - w * 0.05, y - h * 1.35),
             (x - w / 2 + w * 0.05, y - h * 1.35)], haze(_c(0.5, 0.55, 0.66), hk, hz))
    cv.lines([[(x - w / 2 + w * 0.05, y - h * 1.35), (x + w / 2 - w * 0.05, y - h * 1.35)]], max(0.8, 0.15 * s),
             haze(RIM, hk * 0.5, hz))


def _clump_outline(rng, cx, cy, rr, flat=0.8, m=96):
    """Irregular leaf-mass outline: a few big bulges + serrated leaf-tip edge (not a circle)."""
    ph = rng.uniform(0, 6.28, 5)
    nb = int(rng.integers(3, 6))
    t = np.linspace(0, 2 * math.pi, m, endpoint=False)
    k = (1 + 0.16 * np.sin(nb * t + ph[0]) + 0.08 * np.sin((nb + 2) * t + ph[1])
         + 0.05 * np.abs(np.sin(13 * t + ph[2])) + 0.04 * np.abs(np.sin(23 * t + ph[3]))
         + rng.uniform(-0.035, 0.035, m))
    # flatter underside (clumps sag), fuller top
    k = k * (1 - 0.12 * np.clip(np.sin(t), 0, 1))
    return np.stack([cx + np.cos(t) * rr * k, cy + np.sin(t) * rr * flat * k], 1)


def tree(cv, x, y, r, s, rng, hz, hk):
    """Backlit broadleaf tree as irregular leaf masses. Each clump: dark core, a mid-tone that follows
    the clump shape offset toward the light, a sun-side highlight crescent (clump minus a shifted copy
    of itself, so it follows every bump of the outline) and small leaf-cluster breakups at the edge."""
    rng = np.random.default_rng(int(rng.integers(1 << 30)))
    L = np.array([0.55, -0.83])                       # light direction (up-right) in screen space
    tr = haze(_c(0.03, 0.05, 0.1), hk, hz)
    cv.lines([[(x, y + r * 0.9), (x - r * 0.05, y), (x - r * 0.35, y - r * 0.45)],
              [(x - r * 0.04, y + r * 0.1), (x + r * 0.42, y - r * 0.38)],
              [(x - r * 0.02, y - r * 0.1), (x + r * 0.05, y - r * 0.6)]], max(1.0, r * 0.05), tr)
    n = int(rng.integers(6, 10))
    cl = []
    for k in range(n):
        a = -math.pi / 2 + (k / max(n - 1, 1) - 0.5) * 2.6 + rng.normal(0, 0.15)
        d = rng.uniform(0.35, 0.75) * r
        rr = r * rng.uniform(0.3, 0.45)
        cl.append((x + math.cos(a) * d * 1.1, y - r * 0.2 + math.sin(a) * d * 0.85, rr))
    cl.append((x, y - r * 0.35, r * 0.55))
    cl.sort(key=lambda c: -c[1])                       # lower clumps first, upper overlap them
    dark = haze(LEAF_D, hk, hz)
    mid = haze(LEAF_M, hk, hz)
    lit = haze(LEAF_L, hk * 0.8, hz)
    hot = haze(LEAF_H, hk * 0.6, hz)
    for (cx, cy, rr) in cl:
        pts = _clump_outline(rng, cx, cy, rr)
        res = cv.poly(pts, dark)
        if res is None:
            continue
        bb, m0 = res
        # mid-tone mass: same outline, shrunk toward the lit side
        c2 = (cx + L[0] * rr * 0.12, cy + L[1] * rr * 0.12)
        p2 = (pts - [cx, cy]) * 0.8 + c2
        m2 = cv.poly_mask(bb, p2) * m0
        cv.put(bb, m2, mid, 0.95)
        # sun-side highlight crescent: clump minus itself shifted away from the light
        sh = (pts - L * rr * 0.16)
        msh = cv.poly_mask(bb, sh)
        cres = np.clip(m0 - msh, 0, 1)
        cv.put(bb, cres, lit, 0.95)
        sh2 = (pts - L * rr * 0.06)
        cres2 = np.clip(m0 - cv.poly_mask(bb, sh2), 0, 1)
        cv.put(bb, cres2, hot, 0.8)
        # leaf-cluster breakups around the edge
        for j in range(int(rng.integers(5, 9))):
            a = rng.uniform(0, 2 * math.pi)
            ex, ey = cx + math.cos(a) * rr * 1.02, cy + math.sin(a) * rr * 0.8 * 1.02
            lr = rr * rng.uniform(0.08, 0.15)
            q = _clump_outline(rng, ex, ey, lr, flat=0.7, m=24)
            facing = math.cos(a) * L[0] + math.sin(a) * 0.8 * L[1]
            cv.poly(q, dark)
            if facing > 0.3:
                cv.poly(q + L * lr * 0.25, hot if facing > 0.7 else lit, 0.7)


# ============================================================================ layers

def town_near(W, H, pw, ph, y_top0, seed=11):
    """Nearest roofscape: three rows of overlapping houses (back -> front), rooftop props, trees.
    y_top0: plate row at screen top at t=0. Returns (RGBA (ph - y0, pw, 4), y0)."""
    rng = np.random.default_rng(seed)
    hz = _c(0.7, 0.86, 0.96)
    y0 = int(y_top0 + 0.5 * H)
    cv = Canvas(pw, ph - y0, y0)
    ox = (pw - W) / 2
    base = y_top0 + H * 1.3
    rows = [(0.815, 0.12, 0.22, 2.4), (0.86, 0.17, 0.1, 3.2), (0.94, 0.26, 0.0, 4.4)]
    for ri, (fy, fw, hk, sc) in enumerate(rows):
        s = W / 1920 * sc
        x = ox - rng.uniform(0.02, 0.1) * W
        while x < ox + 1.1 * W:
            w = fw * W * rng.uniform(0.8, 1.15)
            ye = y_top0 + (fy + rng.uniform(-0.015, 0.015)) * H
            kind = ['hip', 'gable', 'gable_end', 'hip'][int(rng.integers(0, 4))]
            yr = house(cv, x, ye, w, s, rng, hz, hk, kind, wall_h=base - ye)
            if ri < 2 and rng.random() < 0.28:
                antenna(cv, x + rng.uniform(-0.25, 0.25) * w, yr + 0.5 * s, H * rng.uniform(0.05, 0.09) * (0.7 + 0.2 * ri),
                        s, rng, hz, hk)
            if ri == 1 and rng.random() < 0.3 and kind != 'gable_end':
                solar_heater(cv, x + rng.uniform(-0.15, 0.15) * w, ye - (ye - yr) * 0.3, w * 0.28, s, hz, hk)
            x += w * rng.uniform(0.95, 1.15)
        for k in range(2 if ri < 2 else 1):
            tx = ox + rng.uniform(0.05, 0.95) * W
            tree(cv, tx, y_top0 + (fy + 0.05) * H, fw * W * rng.uniform(0.28, 0.4), s, rng, hz, hk)
    return cv.rgba(), y0


def town_far(W, H, pw, ph, y_top0, seed=5):
    """A hazier second row of rooftops, a mid-rise apartment block with water tank, distant trees."""
    rng = np.random.default_rng(seed)
    hz = _c(0.72, 0.87, 0.96)
    y0 = int(y_top0 + 0.5 * H)
    cv = Canvas(pw, ph - y0, y0)
    s = W / 1920 * 2.6
    ox = (pw - W) / 2
    hk = 0.5
    base = y_top0 + 1.2 * H
    # apartment block (flat roof, railing, water tank, stair tower)
    bx0, bx1 = ox + 0.44 * W, ox + 0.66 * W
    by = y_top0 + 0.765 * H
    wallc = haze(_c(0.1, 0.15, 0.25), hk, hz)
    cv.poly([(bx0, by), (bx1, by), (bx1, base), (bx0, base)], wallc)
    # floors: balcony bands + windows
    fh = 0.045 * H
    for k in range(6):
        yy = by + 0.02 * H + k * fh
        cv.poly([(bx0, yy), (bx1, yy), (bx1, yy + fh * 0.3), (bx0, yy + fh * 0.3)],
                haze(_c(0.18, 0.24, 0.36), hk, hz))
        for j in range(8):
            wx = bx0 + (j + 0.5) * (bx1 - bx0) / 8
            cv.poly([(wx - 0.008 * W, yy + fh * 0.35), (wx + 0.008 * W, yy + fh * 0.35), (wx + 0.008 * W, yy + fh * 0.9),
                     (wx - 0.008 * W, yy + fh * 0.9)], haze(_c(0.06, 0.1, 0.2) if (j + k) % 3 else _c(0.35, 0.5, 0.7), hk, hz))
    cv.lines([[(bx0, by), (bx1, by)]], max(1.2, 0.5 * s), haze(RIM, 0.2, hz))
    # railing on the roof
    rail = []
    for j in range(40):
        xx = bx0 + (bx1 - bx0) * j / 39
        rail.append([(xx, by), (xx, by - 0.012 * H)])
    cv.lines(rail, max(0.6, 0.12 * s), haze(_c(0.3, 0.36, 0.5), hk, hz))
    cv.lines([[(bx0, by - 0.012 * H), (bx1, by - 0.012 * H)]], max(0.8, 0.2 * s), haze(RIM, 0.25, hz))
    # water tank on legs + stair tower
    tx = bx0 + 0.72 * (bx1 - bx0)
    cv.poly([(tx - 0.03 * W, by - 0.03 * H), (tx + 0.03 * W, by - 0.03 * H), (tx + 0.03 * W, by - 0.065 * H),
             (tx - 0.03 * W, by - 0.065 * H)], haze(_c(0.42, 0.48, 0.62), hk, hz))
    cv.lines([[(tx - 0.03 * W, by - 0.065 * H), (tx + 0.03 * W, by - 0.065 * H)]], max(1, 0.3 * s), haze(RIM, 0.2, hz))
    cv.lines([[(tx - 0.025 * W, by - 0.03 * H), (tx - 0.025 * W, by)], [(tx + 0.025 * W, by - 0.03 * H), (tx + 0.025 * W, by)],
              [(tx - 0.025 * W, by), (tx + 0.025 * W, by - 0.03 * H)]], max(0.8, 0.2 * s), haze(_c(0.25, 0.3, 0.44), hk, hz))
    sx = bx0 + 0.18 * (bx1 - bx0)
    cv.poly([(sx - 0.025 * W, by), (sx + 0.025 * W, by), (sx + 0.025 * W, by - 0.05 * H), (sx - 0.025 * W, by - 0.05 * H)],
            haze(_c(0.3, 0.36, 0.52), hk, hz))
    cv.lines([[(sx - 0.025 * W, by - 0.05 * H), (sx + 0.025 * W, by - 0.05 * H)]], max(1, 0.3 * s), haze(RIM, 0.2, hz))
    # hazier rooftops row
    x = ox - 0.05 * W
    while x < ox + 1.08 * W:
        w = W * rng.uniform(0.09, 0.15)
        if bx0 - 0.02 * W < x < bx1 + 0.02 * W:
            x += w * 0.8
            continue
        ye = y_top0 + H * rng.uniform(0.79, 0.81)
        house(cv, x, ye, w, s, rng, hz, hk, ['hip', 'gable', 'gable_end'][int(rng.integers(0, 3))], wall_h=base - ye)
        if rng.random() < 0.22:
            antenna(cv, x + rng.uniform(-0.3, 0.3) * w, ye - w * 0.2, 0.05 * H * rng.uniform(0.7, 1.2), s, rng, hz, hk)
        x += w * rng.uniform(0.85, 1.05)
    for k in range(4):
        tree(cv, ox + rng.uniform(0.0, 1.0) * W, y_top0 + H * rng.uniform(0.79, 0.81), 0.03 * W, s, rng, hz, hk * 1.1)
    return cv.rgba(), y0


def pole_and_wires(W, H, pw, ph, y_top0, T, seed=7, sun_x=None):
    """Concrete utility pole on the right with crossarms, insulators and a pole transformer; power lines
    sweeping off to the left and to a smaller pole receding into the distance.
    Returns (RGBA (ph, pw, 4), list of wire polylines (plate px) for glints)."""
    rng = np.random.default_rng(seed)
    cv = Canvas(pw, ph, 0)
    ox = (pw - W) / 2
    sun_x = ox + 0.47 * W if sun_x is None else sun_x
    px = ox + 0.845 * W
    top = y_top0 - T + 0.72 * H           # pole top: ~0.66 H on screen at the end of the tilt
    bottom = ph
    wt, wb = 0.011 * W, 0.016 * W

    tex_rng = np.random.default_rng(seed + 1)
    streak = cv2.resize(tex_rng.random((1, 64)).astype(np.float32), (256, 1), interpolation=cv2.INTER_CUBIC)[0]

    def pole_col(gx, gy):
        # concrete cylinder, backlit from the upper left: crisp warm rim on the left edge, a lighter
        # flank turning into a dark core, and a faint cool sky-reflected edge on the right
        half = wt + (wb - wt) * np.clip((gy - top) / max(bottom - top, 1), 0, 1)
        f = np.clip((gx - (px - half)) / (2 * half), 0, 1)
        core = _c(0.06, 0.085, 0.15)
        flank = _c(0.2, 0.24, 0.34)
        c = core + (flank - core) * _ss(0.6, 0.12, f)[..., None]
        c = c + (_c(0.22, 0.3, 0.46) - c) * (_ss(0.78, 0.97, f) * 0.6)[..., None]
        c = c * (1 + 0.08 * (streak[(f * 255).astype(np.int32)] - 0.5))[..., None]
        rim = np.exp(-((f - 0.05) / 0.05) ** 2)
        return c + (RIM - c) * rim[..., None] * 0.9
    cv.poly([(px - wt, top), (px + wt, top), (px + wb, bottom), (px - wb, bottom)], pole_col)
    cv.ellipse(px, top, wt, wt * 0.35, _c(0.9, 0.82, 0.7))
    # step bolts (short rods with a lit top) + pole number plate
    for k in range(18):
        yy = top + 0.1 * H + k * 0.035 * H
        sgn = 1 if k % 2 else -1
        half = wt + (wb - wt) * (yy - top) / max(bottom - top, 1)
        p0, p1 = (px + sgn * half * 0.8, yy), (px + sgn * (half + 0.012 * W), yy - 0.003 * H)
        cv.lines([[p0, p1]], max(1.2, 0.0028 * W), _c(0.06, 0.08, 0.14))
        cv.lines([[(p0[0], p0[1] - 0.0012 * W), (p1[0], p1[1] - 0.0012 * W)]], max(0.6, 0.0009 * W), RIM, 0.7)
    py_ = top + 0.52 * H
    cv.poly([(px - wt * 0.7, py_), (px + wt * 0.5, py_), (px + wt * 0.5, py_ + 0.05 * H), (px - wt * 0.7, py_ + 0.05 * H)],
            _c(0.32, 0.36, 0.46))
    for j in range(3):
        cv.lines([[(px - wt * 0.5, py_ + 0.012 * H + j * 0.012 * H), (px + wt * 0.3, py_ + 0.012 * H + j * 0.012 * H)]],
                 max(0.8, 0.0012 * W), _c(0.2, 0.25, 0.42), 0.8)
    s = W / 1920
    wires = []
    arms = [(0.018, 0.12, 3), (0.06, 0.1, 3), (0.12, 0.075, 2)]
    ins_pts = []
    for (dy, half, n) in arms:
        ay = top + dy * H
        cv.poly([(px - half * W, ay - 0.003 * H), (px + half * W * 0.7, ay - 0.003 * H),
                 (px + half * W * 0.7, ay + 0.004 * H), (px - half * W, ay + 0.004 * H)], _c(0.06, 0.08, 0.14))
        cv.lines([[(px - half * W, ay - 0.003 * H), (px + half * W * 0.7, ay - 0.003 * H)]], max(1, 1.6 * s), RIM, 0.9)
        # brace
        cv.lines([[(px - half * W * 0.6, ay + 0.004 * H), (px - wt, ay + 0.03 * H)],
                  [(px + half * W * 0.45, ay + 0.004 * H), (px + wt, ay + 0.03 * H)]], max(1, 2.2 * s), _c(0.06, 0.08, 0.14))
        for i in range(n):
            ix = px - half * W + (half * W * 1.7) * (i + 0.5) / n
            # porcelain insulator: stacked discs, lit top
            for j in range(3):
                iy = ay - 0.006 * H - j * 0.006 * H
                cv.ellipse(ix, iy, 0.006 * W * (1 - 0.15 * j), 0.0028 * H, _c(0.25, 0.3, 0.42))
                cv.ellipse(ix - 0.0015 * W, iy - 0.0012 * H, 0.004 * W * (1 - 0.15 * j), 0.0012 * H, _c(1.1, 1.05, 0.95), 0.9)
            ins_pts.append((ix, ay - 0.022 * H))
    # pole transformer: two cylinders on a bracket
    for k, tx in enumerate([px - 0.03 * W, px + 0.028 * W]):
        ty = top + 0.2 * H
        tw, th = 0.019 * W, 0.075 * H

        def tcol(gx, gy, tx=tx, tw=tw):
            f = np.clip((gx - (tx - tw)) / (2 * tw), 0, 1)
            core = _c(0.07, 0.09, 0.16)
            c = core + (_c(0.28, 0.32, 0.44) - core) * np.exp(-((f - 0.18) / 0.16) ** 2)[..., None]
            c = c + (_c(0.2, 0.28, 0.42) - c) * (_ss(0.8, 0.98, f) * 0.5)[..., None]
            c = c + (RIM - c) * np.exp(-((f - 0.03) / 0.035) ** 2)[..., None] * 0.85
            ribs = (np.mod(gy, 0.008 * H) < 0.0025 * H).astype(np.float32)
            return c * (1 - 0.25 * ribs[..., None])
        cv.poly([(tx - tw, ty), (tx + tw, ty), (tx + tw, ty + th), (tx - tw, ty + th)], tcol)
        cv.ellipse(tx, ty, tw, tw * 0.35, _c(0.95, 0.86, 0.72))
        cv.lines([[(tx - tw * 0.95, ty - 0.001 * H), (tx + tw * 0.2, ty - tw * 0.3)]], max(1, 1.5 * s), RIM, 0.8)
        # bushings + drop wires
        cv.lines([[(tx, ty - 0.004 * H), (tx, ty - 0.012 * H)]], max(1, 3 * s), _c(0.3, 0.33, 0.42))
        wires.append((np.array([(tx, ty - 0.012 * H), (tx - 0.01 * W, top + 0.08 * H), (px - 0.05 * W, top + 0.06 * H)]), 1.3))
    cv.poly([(px - 0.06 * W, top + 0.19 * H), (px + 0.06 * W, top + 0.19 * H), (px + 0.06 * W, top + 0.197 * H),
             (px - 0.06 * W, top + 0.197 * H)], _c(0.06, 0.08, 0.14))
    # far pole (receding along the street to the left, smaller)
    fx, ftop = ox + 0.2 * W, top + 0.3 * H
    fwt = wt * 0.4
    cv.poly([(fx - fwt, ftop), (fx + fwt, ftop), (fx + fwt * 1.3, ph), (fx - fwt * 1.3, ph)],
            lambda gx, gy: _c(0.1, 0.14, 0.24) + (_c(1.0, 0.9, 0.75) - _c(0.1, 0.14, 0.24)) * np.exp(-((gx - fx + fwt * 0.7) / (fwt * 0.3)) ** 2)[..., None] * 0.7)
    far_ins = []
    for (dy, half, n) in arms[:2]:
        ay = ftop + dy * 0.4 * H
        cv.lines([[(fx - half * 0.4 * W, ay), (fx + half * 0.28 * W, ay)]], max(1, 2.5 * s), _c(0.1, 0.14, 0.24))
        for i in range(n):
            far_ins.append((fx - half * 0.4 * W + half * 0.68 * W * (i + 0.5) / n, ay - 0.008 * H))
    # main lines: near pole -> far pole, and near pole -> off frame left (another span), + one to the right
    spans = []
    for a, b in zip(ins_pts[:5], far_ins[:5]):
        spans.append((a, b, 0.045 * H * rng.uniform(0.85, 1.15), 2.6 * s))
    for i, a in enumerate(ins_pts):
        b = (a[0] - 1.25 * W, a[1] + 0.18 * H + i * 0.012 * H)
        spans.append((a, b, 0.14 * H * rng.uniform(0.9, 1.1), 3.2 * s))
    for i, a in enumerate(ins_pts[:3]):
        b = (a[0] + 0.5 * W, a[1] - 0.05 * H + i * 0.01 * H)
        spans.append((a, b, 0.05 * H, 3.2 * s))
    # thick communication cable lower on the pole (sags more)
    cy = top + 0.33 * H
    spans.append(((px, cy), (fx, ftop + 0.2 * H), 0.07 * H, 5.0 * s))
    spans.append(((px, cy), (px - 1.3 * W, cy + 0.3 * H), 0.2 * H, 5.5 * s))
    for (a, b, sag, wd) in spans:
        t = np.linspace(0, 1, 160)
        x = a[0] + (b[0] - a[0]) * t
        y = a[1] + (b[1] - a[1]) * t + sag * 4 * t * (1 - t)
        pts = np.stack([x, y], 1)
        wires.append((pts, wd))
    for pts, wd in wires:
        cv.lines([pts], wd, WIRE)
        # continuous thin specular along the sunward (upper) side, strongest where the wire runs
        # toward the sun, fading smoothly along its length
        def spec(gx, gy, sx=sun_x):
            k = 0.15 + 1.0 * np.exp(-((gx - sx) / (0.3 * W)) ** 2)
            return RIM[None, None, :] * 0.0 + (_c(1.3, 1.18, 0.95) * k[..., None])
        cv.lines([pts - [0, wd * 0.38]], max(0.5, wd * 0.25), spec, 0.6)
    return cv.rgba(), wires
