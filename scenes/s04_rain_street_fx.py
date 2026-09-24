"""s04_rain_street optical extras: disc bokeh (far lights beyond focus) and rain droplets on the lens."""
import math
import numpy as np
import cv2

_SPR = {}


def disc_sprite(r, rim=0.35, soft=1.0, blades=0):
    """Bokeh disc sprite (float32 2D), radius r px: flat body, brighter rim, soft 1px edge.
    blades>0 gives a slightly polygonal (aperture) outline."""
    key = (round(r * 4) / 4, rim, soft, blades)
    if key in _SPR:
        return _SPR[key]
    R = int(math.ceil(r + 2 * soft + 1))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy)
    if blades:
        ang = np.arctan2(yy, xx)
        k = math.pi / blades
        a = np.mod(ang + k, 2 * k) - k
        d = d * np.cos(a) / math.cos(k) * 0.93 + d * 0.07
    body = np.clip((r - d) / soft + 0.5, 0, 1)
    ring = np.exp(-((d - (r - 1.2 * soft)) / max(r * 0.12, 0.8)) ** 2) * body
    s = body * (0.75 + 0.25 * (d / max(r, 1)) ** 2) + rim * ring
    s = s.astype(np.float32)
    _SPR[key] = s
    return s


def add_sprite(img, spr, x, y, col):
    H, W = img.shape[:2]
    R = spr.shape[0] // 2
    xi, yi = int(round(x)), int(round(y))
    x0, y0, x1, y1 = xi - R, yi - R, xi + R + 1, yi + R + 1
    if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
        return
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1, sy1 = spr.shape[1] - max(0, x1 - W), spr.shape[0] - max(0, y1 - H)
    img[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)] += spr[sy0:sy1, sx0:sx1, None] * col


def find_lights(rgba, depth, zmin, thresh, n_max, min_dist, seed=0, point=False):
    """Bright local maxima in a premultiplied plate beyond zmin. -> list of (x, y, z, rgb)."""
    rgb = rgba[..., :3]
    lum = rgb.max(-1)
    lum = cv2.GaussianBlur(lum, (0, 0), 1.0)
    dil = cv2.dilate(lum, np.ones((5, 5), np.uint8))
    pk = (lum >= dil - 1e-6) & (lum > thresh) & (depth > zmin)
    if point:
        # isolated point-like lights only (not the body of a large sign)
        wide = cv2.blur(lum, (15, 15))
        pk &= lum > 2.2 * wide
    ys, xs = np.nonzero(pk)
    order = np.argsort(-lum[ys, xs])
    out = []
    for i in order:
        x, y = xs[i], ys[i]
        if any((x - a) ** 2 + (y - b) ** 2 < min_dist ** 2 for a, b, _, _ in out):
            continue
        c = cv2.GaussianBlur(rgb, (0, 0), 1.5)[y, x] if False else rgb[max(y - 1, 0):y + 2, max(x - 1, 0):x + 2].reshape(-1, 3).mean(0)
        out.append((float(x), float(y), float(depth[y, x]), c.astype(np.float32)))
        if len(out) >= n_max:
            break
    return out


class LensDrops:
    """Out-of-focus raindrops sitting on / sliding down the lens: faint discs lit by the scene light
    behind them, each with a bright refracted rim. Deterministic, smooth in t."""

    def __init__(self, W, H, n=16, seed=7):
        rng = np.random.default_rng(seed)
        self.W, self.H = W, H
        self.x = rng.uniform(0.0, 1.0, n) * W
        self.y = rng.uniform(-0.1, 0.95, n) * H
        self.r = (rng.uniform(0.012, 0.035, n) ** 1.0) * W
        self.v = rng.uniform(0.0, 0.025, n) * H * (rng.random(n) < 0.35)   # a few slide down
        self.a = rng.uniform(0.35, 1.0, n)
        self.t0 = rng.uniform(-3, 5, n)

    def draw(self, img, light, t):
        H, W = img.shape[:2]
        for i in range(len(self.x)):
            y = self.y[i] + self.v[i] * t
            x = self.x[i]
            r = self.r[i]
            # fade in/out over the shot for a couple of drops (appear as new drops land)
            fa = min(1.0, max(0.0, (t - self.t0[i]) / 0.6)) if self.t0[i] > 0 else 1.0
            if fa <= 0:
                continue
            xi, yi = int(min(max(x, 0), W - 1)), int(min(max(y, 0), H - 1))
            col = light[yi, xi]
            spr = disc_sprite(r, rim=0.5, soft=max(1.0, r * 0.08), blades=0)
            add_sprite(img, spr, x, y, col * 0.32 * self.a[i] * fa)


def bokeh_sprite(r, ox=0.0, oy=0.0, blades=6, soft=None, poly=0.35):
    """Large sprites are evaluated at reduced resolution and upsampled (they are soft anyway)."""
    if r > 20:
        q = r / 20.0
        sm = _bokeh_sprite(20.0, ox, oy, blades, None if soft is None else soft / q, poly)
        R = int(math.ceil(r + 2 * max(1.0, r * 0.1) + 1))
        return cv2.resize(sm, (2 * R + 1, 2 * R + 1), interpolation=cv2.INTER_LINEAR)
    return _bokeh_sprite(r, ox, oy, blades, soft, poly)


def _bokeh_sprite(r, ox=0.0, oy=0.0, blades=6, soft=None, poly=0.35):
    """Soft, fully filled bokeh disc (no bright soap-bubble rim). (ox, oy) = position in the frame relative
    to the centre in [-1, 1]: toward the edges the disc is clipped by a second, offset pupil into a
    cat-eye (lemon) shape. blades>0 blends in a faint polygonal aperture."""
    soft = max(1.0, r * 0.1) if soft is None else soft
    R = int(math.ceil(r + 2 * soft + 1))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    d = np.sqrt(xx * xx + yy * yy)
    if blades:
        ang = np.arctan2(yy, xx) + 0.3
        k = math.pi / blades
        a = np.mod(ang + k, 2 * k) - k
        dp = d * np.cos(a) / math.cos(k)
        d = d * (1 - poly) + dp * poly
    body = np.clip((r - d) / soft + 0.5, 0, 1)
    rho = math.sqrt(ox * ox + oy * oy)
    if rho > 0.25:
        k = 0.9 * (rho - 0.25) / 0.75
        sx, sy = -ox / rho * r * k, -oy / rho * r * k
        d2 = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
        body = body * np.clip((r * 1.02 - d2) / soft + 0.5, 0, 1)
    # very gentle edge lift + faint onion texture so it reads as glass, not a flat sticker
    s = body * (0.88 + 0.12 * np.clip(d / max(r, 1), 0, 1) ** 3)
    return s.astype(np.float32)


def fast_bloom(img, threshold=0.65, knee=0.4, strength=0.5, halation=0.22, radii=(0.003, 0.01, 0.035, 0.1),
               mist=0.14, mist_sigma=0.05):
    """Soft-knee multi-scale bloom + warm halation + wide atmospheric mist, all evaluated at half/quarter
    resolution and added back with a single upsample. In place on img (H, W, 3)."""
    H, W = img.shape[:2]
    small = cv2.resize(img, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    lum = small.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
    k = np.maximum(k * k, (lum > threshold + knee).astype(np.float32))
    br = small * k
    q = cv2.resize(br, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    wts = [1.0, 0.8, 0.6, 0.45][:len(radii)]
    acc_q = np.zeros_like(q)
    acc_h = cv2.GaussianBlur(br, (0, 0), max(0.5, radii[0] * W / 2)) * wts[0]
    for r, wt in zip(radii[1:], wts[1:]):
        acc_q += fast_blur(q, r * W / 4) * wt
    # wide mist from the whole (not bright-passed) image
    e = cv2.resize(small, (W // 8, H // 8), interpolation=cv2.INTER_AREA)
    e = cv2.GaussianBlur(e, (0, 0), mist_sigma * W / 8)
    acc_q += cv2.resize(e, (W // 4, H // 4), interpolation=cv2.INTER_LINEAR) * (mist * sum(wts) / strength)
    tot = (acc_h + cv2.resize(acc_q, (W // 2, H // 2), interpolation=cv2.INTER_LINEAR)) * (strength / sum(wts))
    if halation:
        hal = cv2.GaussianBlur(br, (0, 0), 0.006 * W / 2)
        tot += hal * np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5
    img += cv2.resize(tot, (W, H), interpolation=cv2.INTER_LINEAR)
    return img


def fast_blur(img, sigma):
    if sigma < 0.3:
        return img
    if sigma > 6:
        f = int(sigma // 3)
        h, w = img.shape[:2]
        s = cv2.resize(img, (max(w // f, 1), max(h // f, 1)), interpolation=cv2.INTER_AREA)
        s = cv2.GaussianBlur(s, (0, 0), sigma / f)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(img, (0, 0), sigma)


def draw_bokeh(img, r, ox, oy, blades, x, y, col):
    """Bokeh disc of (continuous) radius r at sub-pixel position (x, y): the base sprite is rendered at
    radius <= 20 px and resampled with an exact affine scale + fractional offset, so discs glide and grow
    smoothly during the camera move instead of snapping to whole pixels / integer kernel sizes."""
    H, W = img.shape[:2]
    r0 = min(float(r), 20.0)
    spr = _bokeh_sprite(r0, ox, oy, blades)
    c0 = (spr.shape[0] - 1) / 2.0
    s = float(r) / r0
    R = int(math.ceil(s * (c0 + 1))) + 2
    xi, yi = int(math.floor(x)), int(math.floor(y))
    x0, y0 = xi - R, yi - R
    n = 2 * R + 1
    if x0 + n <= 0 or y0 + n <= 0 or x0 >= W or y0 >= H:
        return
    M = np.float32([[s, 0, (x - x0) - s * c0], [0, s, (y - y0) - s * c0]])
    p = cv2.warpAffine(spr, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ax0, ay0 = max(x0, 0), max(y0, 0)
    ax1, ay1 = min(x0 + n, W), min(y0 + n, H)
    img[ay0:ay1, ax0:ax1] += p[ay0 - y0:ay1 - y0, ax0 - x0:ax1 - x0, None] * np.asarray(col, np.float32)
