"""Core rendering toolkit for the scenic anime montage.

Conventions
-----------
* Images are float32 numpy arrays, shape (H, W, 3) or (H, W, 4), RGB(A), values ~0..1
  (HDR values > 1 are allowed before `finish()` tonemaps them).
* Coordinates are pixels, origin top-left, x right, y down.
* Every scene module exposes `DURATION` (seconds) and `class Scene(W, H)` with
  `frame(t) -> float32 (H, W, 3)` where t is seconds in [0, DURATION).
  The Scene must work at any resolution (previews render at 960x540, finals at 1920x1080),
  so express sizes relative to W/H.
"""
import math
import subprocess
import numpy as np
import cv2

FPS = 24

# ----------------------------------------------------------------------------- math

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def ease_out_cubic(t):
    t = min(max(t, 0.0), 1.0)
    return 1 - (1 - t) ** 3


def ease_in_out_sine(t):
    t = min(max(t, 0.0), 1.0)
    return -(math.cos(math.pi * t) - 1) / 2


def hex2rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def grid(W, H):
    """Return (X, Y) float32 pixel coordinate grids."""
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    return xs, ys


# ----------------------------------------------------------------------------- noise

def value_noise(W, H, cells_x, cells_y, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.random((int(cells_y) + 3, int(cells_x) + 3)).astype(np.float32)
    big = cv2.resize(g, (W + 2 * max(1, W // max(1, int(cells_x))), H + 2 * max(1, H // max(1, int(cells_y)))),
                     interpolation=cv2.INTER_CUBIC)
    ox = (big.shape[1] - W) // 2
    oy = (big.shape[0] - H) // 2
    return big[oy:oy + H, ox:ox + W]


def fbm(W, H, scale=4.0, octaves=6, lacunarity=2.0, gain=0.5, seed=0, aspect=True):
    """Fractal value noise normalised to 0..1. `scale` = number of base cells across the width."""
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    cx = scale
    cy = scale * (H / W if aspect else 1.0)
    for o in range(octaves):
        out += amp * value_noise(W, H, max(1, cx), max(1, cy), seed + 101 * o)
        tot += amp
        amp *= gain
        cx *= lacunarity
        cy *= lacunarity
    out /= tot
    lo, hi = np.percentile(out, 0.5), np.percentile(out, 99.5)
    return clamp01((out - lo) / (hi - lo + 1e-9))


def ridged(W, H, scale=4.0, octaves=6, seed=0):
    n = fbm(W, H, scale, octaves, seed=seed)
    return 1.0 - np.abs(n * 2 - 1)


def warp(img, dx, dy, border=cv2.BORDER_REFLECT):
    """Displace an image by per-pixel offsets (pixels)."""
    H, W = img.shape[:2]
    xs, ys = grid(W, H)
    return cv2.remap(img, (xs + dx).astype(np.float32), (ys + dy).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=border)


# ----------------------------------------------------------------------------- painting

def vertical_gradient(W, H, stops):
    """stops: list of (pos 0..1 top->bottom, rgb or '#hex'). Returns (H, W, 3)."""
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([hex2rgb(s[1]) if isinstance(s[1], str) else s[1] for s in stops], np.float32)
    y = np.linspace(0, 1, H, dtype=np.float32)
    col = np.stack([np.interp(y, pos, cols[:, c]) for c in range(3)], -1)
    return np.repeat(col[:, None, :], W, axis=1).astype(np.float32)


def radial(W, H, cx, cy, radius, power=1.0):
    xs, ys = grid(W, H)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max(radius, 1e-6)
    return np.clip(1 - d, 0, 1) ** power


def glow(W, H, cx, cy, radius, falloff=2.0):
    """Soft inverse-power glow (for suns, lamps). Peaks at 1."""
    xs, ys = grid(W, H)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / max(radius, 1e-6)
    return 1.0 / (1.0 + d ** falloff)


def over(dst, src_rgb, alpha):
    """Alpha-composite src over dst. alpha: (H, W) or (H, W, 1)."""
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return dst * (1 - alpha) + src_rgb * alpha


def screen(a, b):
    return 1 - (1 - a) * (1 - b)


def blur(img, sigma):
    if sigma <= 0.05:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigma)


def polygon_mask(W, H, pts, aa=True, ss=2):
    """Anti-aliased filled polygon mask (H, W) from a list of (x, y) points."""
    m = np.zeros((H * ss, W * ss), np.uint8)
    p = (np.array(pts, np.float64) * ss).astype(np.int32)
    cv2.fillPoly(m, [p], 255, lineType=cv2.LINE_AA if aa else cv2.LINE_8)
    m = m.astype(np.float32) / 255.0
    if ss > 1:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    return m


def line_mask(W, H, p0, p1, width=1.0, ss=4):
    """Anti-aliased thin line (power lines etc.)."""
    m = np.zeros((H * ss, W * ss), np.uint8)
    cv2.line(m, (int(p0[0] * ss), int(p0[1] * ss)), (int(p1[0] * ss), int(p1[1] * ss)), 255,
             max(1, int(round(width * ss))), cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)


def polyline_mask(W, H, pts, width=1.0, ss=4):
    m = np.zeros((H * ss, W * ss), np.uint8)
    p = (np.array(pts, np.float64) * ss).astype(np.int32)
    cv2.polylines(m, [p], False, 255, max(1, int(round(width * ss))), cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)


def catenary(p0, p1, sag, n=64):
    """Points of a hanging wire between p0 and p1 with given sag (pixels)."""
    t = np.linspace(0, 1, n)
    x = p0[0] + (p1[0] - p0[0]) * t
    y = p0[1] + (p1[1] - p0[1]) * t + sag * 4 * t * (1 - t)
    return np.stack([x, y], 1)


def splat(img, xs, ys, radius, color, intensity=1.0, soft=True):
    """Additively splat soft round points (particles, bokeh, stars) onto img in place.
    xs, ys, radius, intensity may be arrays. color (3,) or (N,3)."""
    H, W = img.shape[:2]
    xs = np.atleast_1d(xs); ys = np.atleast_1d(ys)
    n = len(xs)
    radius = np.broadcast_to(np.atleast_1d(radius), (n,))
    intensity = np.broadcast_to(np.atleast_1d(intensity), (n,))
    color = np.asarray(color, np.float32)
    if color.ndim == 1:
        color = np.broadcast_to(color, (n, 3))
    for i in range(n):
        r = float(radius[i])
        R = int(math.ceil(r * 2.5)) + 1
        x0, y0 = int(xs[i]) - R, int(ys[i]) - R
        x1, y1 = int(xs[i]) + R + 1, int(ys[i]) + R + 1
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        d2 = ((xx - xs[i]) ** 2 + (yy - ys[i]) ** 2) / max(r * r, 1e-4)
        if soft:
            k = np.exp(-d2 * 2.0)
        else:
            k = clamp01(1.5 - np.sqrt(d2))
        img[cy0:cy1, cx0:cx1, :3] += k[..., None] * color[i] * float(intensity[i])
    return img


# ----------------------------------------------------------------------------- layers / camera

class Layer:
    """A pre-rendered RGBA plate that is positioned by a camera with parallax.

    rgba: (h, w, 4) float32, premultiplied=False. Plates are usually larger than the frame so the
    camera can pan. depth: 0 = locked to camera (UI), 1 = normal, <1 = far (moves less).
    """

    def __init__(self, rgba, depth=1.0, anchor=(0.5, 0.5)):
        self.rgba = rgba.astype(np.float32)
        self.depth = depth
        self.anchor = anchor

    def render(self, W, H, cam_x=0.0, cam_y=0.0, zoom=1.0, rot_deg=0.0):
        """Sample this plate into a (H, W, 4) frame. cam_x/cam_y in plate pixels (scaled by depth)."""
        h, w = self.rgba.shape[:2]
        z = 1 + (zoom - 1) * self.depth
        cx = w * self.anchor[0] + cam_x * self.depth
        cy = h * self.anchor[1] + cam_y * self.depth
        M = cv2.getRotationMatrix2D((cx, cy), rot_deg * self.depth, z)
        M[0, 2] += W / 2 - cx
        M[1, 2] += H / 2 - cy
        return cv2.warpAffine(self.rgba, M, (W, H), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def composite(base, layers_rgba):
    out = base.copy()
    for L in layers_rgba:
        out = over(out, L[..., :3], L[..., 3])
    return out


# ----------------------------------------------------------------------------- post FX

def bloom(img, threshold=0.75, strength=0.6, radii=(0.004, 0.012, 0.035, 0.09), tint=(1.0, 0.95, 0.9)):
    """Multi-scale bloom. radii are fractions of frame width."""
    H, W = img.shape[:2]
    lum = img[..., :3].mean(-1, keepdims=True)
    bright = img[..., :3] * smoothstep(threshold, threshold + 0.35, lum)
    small = cv2.resize(bright, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(small)
    for r in radii:
        acc += blur(small, r * W / 2)
    acc = cv2.resize(acc / len(radii), (W, H), interpolation=cv2.INTER_LINEAR)
    return img[..., :3] + acc * strength * np.asarray(tint, np.float32)


def god_rays(src, cx, cy, length=0.35, steps=28, decay=0.96, strength=0.8, tint=(1.0, 0.9, 0.7)):
    """Screen-space light shafts: radially smear `src` (bright/occlusion image, (H,W,3) or (H,W))
    toward (cx, cy). Returns an additive ray image (H, W, 3)."""
    if src.ndim == 2:
        src = np.repeat(src[..., None], 3, -1)
    H, W = src.shape[:2]
    sw, sh = W // 2, H // 2
    s = cv2.resize(src.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(s)
    wsum = 0.0
    w = 1.0
    for i in range(steps):
        k = 1.0 - length * i / steps
        M = np.array([[k, 0, (1 - k) * cx / 2], [0, k, (1 - k) * cy / 2]], np.float32)
        # sample the source at points pulled toward the light: inverse map so pixels smear outward
        acc += cv2.warpAffine(s, M, (sw, sh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT) * w
        wsum += w
        w *= decay
    acc = cv2.resize(acc / wsum, (W, H), interpolation=cv2.INTER_LINEAR)
    return acc * strength * np.asarray(tint, np.float32)


def lens_flare(W, H, lx, ly, intensity=1.0, seed=3):
    """Additive anime-style lens flare: hexagonal-ish ghosts on the axis through the frame centre,
    a soft halo ring and a horizontal anamorphic streak. Returns (H, W, 3)."""
    out = np.zeros((H, W, 3), np.float32)
    cx, cy = W / 2, H / 2
    vx, vy = cx - lx, cy - ly
    rng = np.random.default_rng(seed)
    ghosts = [(0.35, 0.020, (0.4, 0.7, 1.0)), (0.62, 0.045, (0.6, 1.0, 0.7)), (0.9, 0.012, (1.0, 0.8, 0.5)),
              (1.25, 0.07, (0.5, 0.6, 1.0)), (1.55, 0.03, (1.0, 0.6, 0.8)), (1.9, 0.11, (0.6, 0.9, 1.0))]
    xs, ys = grid(W, H)
    for k, r, col in ghosts:
        gx, gy = lx + vx * k, ly + vy * k
        R = r * W
        d = np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2) / R
        disc = smoothstep(1.0, 0.7, d) * (0.5 + 0.5 * smoothstep(0.2, 1.0, d))
        out += disc[..., None] * np.array(col, np.float32) * 0.06 * (0.6 + 0.8 * rng.random())
    # halo ring around the light
    d = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2) / (0.28 * W)
    ring = np.exp(-((d - 1.0) ** 2) / 0.004)
    hue = np.stack([0.6 + 0.4 * np.cos(d * 9), 0.7 + 0.3 * np.cos(d * 9 + 2), 1.0 + 0 * d], -1)
    out += ring[..., None] * hue * 0.05
    # anamorphic streak
    streak = np.exp(-((ys - ly) ** 2) / (2 * (0.0025 * H) ** 2)) * np.exp(-np.abs(xs - lx) / (0.35 * W))
    out += streak[..., None] * np.array([0.7, 0.85, 1.0], np.float32) * 0.5
    # core starburst
    ang = np.arctan2(ys - ly, xs - lx)
    dd = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2) / W
    rays = (np.abs(np.cos(ang * 6)) ** 40) * np.exp(-dd / 0.06)
    out += rays[..., None] * np.array([1.0, 0.95, 0.85], np.float32) * 0.35
    return out * intensity


def chromatic_aberration(img, amount=0.0015):
    H, W = img.shape[:2]
    out = img.copy()
    for c, s in ((0, 1 + amount), (2, 1 - amount)):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        out[..., c] = cv2.warpAffine(img[..., c], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def vignette(img, strength=0.35, power=2.2, color=(0.0, 0.0, 0.0)):
    H, W = img.shape[:2]
    xs, ys = grid(W, H)
    d = np.sqrt(((xs - W / 2) / (W / 2)) ** 2 + ((ys - H / 2) / (H / 2)) ** 2) / math.sqrt(2)
    v = (d ** power * strength)[..., None]
    return img * (1 - v) + np.asarray(color, np.float32) * v


def grain(img, amount=0.02, seed=0):
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((H // 2 + 1, W // 2 + 1)).astype(np.float32)
    n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
    return img + n[..., None] * amount


def tonemap(img, exposure=1.0):
    """Gentle filmic (ACES fitted) curve; keeps anime colours vivid."""
    x = img * exposure
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return clamp01((x * (a * x + b)) / (x * (c * x + d) + e))


def saturate(img, amount=1.1):
    l = (img * np.array([0.2126, 0.7152, 0.0722], np.float32)).sum(-1, keepdims=True)
    return l + (img - l) * amount


def grade(img, lift=(0, 0, 0), gamma=(1, 1, 1), gain=(1, 1, 1)):
    img = clamp01(img)
    img = img * np.asarray(gain, np.float32) + np.asarray(lift, np.float32) * (1 - img)
    return np.power(clamp01(img), 1.0 / np.asarray(gamma, np.float32))


def finish(img, t=0.0, exposure=1.0, sat=1.08, grain_amt=0.012, vig=0.3, ca=0.0012, tonemap_on=False):
    """Standard final pass. Call last in Scene.frame(). Keeps the image in 0..1."""
    x = img[..., :3]
    if tonemap_on:
        x = tonemap(x, exposure)
    else:
        x = x * exposure
    x = saturate(x, sat)
    if ca:
        x = chromatic_aberration(x, ca)
    if vig:
        x = vignette(x, vig)
    if grain_amt:
        x = grain(x, grain_amt, seed=int(t * FPS) + 7)
    return clamp01(x).astype(np.float32)


# ----------------------------------------------------------------------------- IO

def to_u8(img):
    return (clamp01(img[..., :3]) * 255 + 0.5).astype(np.uint8)


def save_png(path, img):
    cv2.imwrite(str(path), cv2.cvtColor(to_u8(img), cv2.COLOR_RGB2BGR))


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


class VideoWriter:
    """Pipe frames straight into ffmpeg (H.264, high quality)."""

    def __init__(self, path, W, H, fps=FPS, crf=15):
        self.p = subprocess.Popen(
            ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
             '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf),
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(to_u8(img).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()
