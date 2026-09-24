"""Fast post FX for s06_seaside (keep 1080p frames well under the time budget):
- Bloom computed on a quarter-res bright pass (soft knee) with warm halation.
- In-place star glints (no full-frame temporaries).
- Lens flare: translation-invariant parts (core, bloom, spikes, halo ring, streak) pre-rendered once and
  shifted per frame; chromatic ghosts along the optical axis evaluated per frame in small boxes."""
import math
import numpy as np
import cv2

from lib import core as C, fx as F


def fast_bloom(img, threshold=0.9, knee=0.35, strength=0.3, halation=0.15, radii=(0.004, 0.012, 0.035, 0.09)):
    H, W = img.shape[:2]
    q = 4
    small = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
    lum = small.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1) ** 2
    br = small * np.maximum(k, (lum > threshold + knee).astype(np.float32))
    acc = np.zeros_like(br)
    wts = (1.0, 0.8, 0.6, 0.45)
    for r, wt in zip(radii, wts):
        acc += F.fast_blur(br, max(r * W / q, 0.6)) * wt
    acc /= sum(wts[:len(radii)])
    add = acc * strength
    if halation:
        hal = F.fast_blur(br, 0.006 * W / q)
        add = add + hal * np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5
    add = cv2.resize(add, (W, H), interpolation=cv2.INTER_LINEAR)
    img += add
    return img


def add_glints(img, xs, ys, size, inten, color=(1.0, 0.95, 0.85), angle=0.0, arms=4):
    """Star glints added in place (see fx.glints)."""
    H, W = img.shape[:2]
    xs = np.atleast_1d(np.asarray(xs, np.float32))
    ys = np.atleast_1d(np.asarray(ys, np.float32))
    n = len(xs)
    size = np.broadcast_to(np.atleast_1d(size), (n,)).astype(np.float32)
    inten = np.broadcast_to(np.atleast_1d(inten), (n,)).astype(np.float32)
    col = np.asarray(color, np.float32)
    for i in range(n):
        if inten[i] < 1e-3:
            continue
        L = size[i] * W
        R = int(L) + 2
        x0, y0 = int(xs[i]) - R, int(ys[i]) - R
        x1, y1 = int(xs[i]) + R + 1, int(ys[i]) + R + 1
        if x1 <= 0 or y1 <= 0 or x0 >= W or y0 >= H:
            continue
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        dx, dy = xx - xs[i], yy - ys[i]
        k = np.zeros_like(dx)
        for a in range(arms // 2):
            th = angle + a * math.pi / (arms // 2)
            u = dx * math.cos(th) + dy * math.sin(th)
            v = -dx * math.sin(th) + dy * math.cos(th)
            wdt = 0.6 + 0.02 * L
            k += np.exp(-(v / wdt) ** 2) * np.clip(1 - np.abs(u) / L, 0, 1) ** 2.5 * (1.0 if a == 0 else 0.6)
        r2 = (dx * dx + dy * dy) / (max(L * 0.08, 0.8) ** 2)
        k += np.exp(-r2) * 1.5
        img[cy0:cy1, cx0:cx1] += k[..., None] * col * inten[i]
    return img


def _flare_plate(Wc, Hc, cx, cy, Wref, tint=(1.0, 0.93, 0.8), seed=3, rays=6, ray_len=0.055, starburst=1.0,
                 halo=1.0, streak=0.2, glow=1.0, spike_width=0.06):
    """Static part of fx.anime_flare on a (Hc, Wc) canvas with sizes relative to the FRAME width Wref."""
    s = 2
    w, h = Wc // s, Hc // s
    x0, y0 = cx / s, cy / s
    xs, ys = C.grid(w, h)
    dx, dy = xs - x0, ys - y0
    d = np.sqrt(dx * dx + dy * dy) + 1e-3
    out = np.zeros((h, w, 3), np.float32)
    tint = np.asarray(tint, np.float32)
    dW = d / (Wref / s)
    core = 1.0 / (1.0 + (dW / 0.004) ** 2.4)
    broad = 1.0 / (1.0 + (dW / 0.022) ** 2) * 0.1 + np.exp(-dW / 0.12) * 0.06 + np.exp(-dW / 0.35) * 0.03
    out += core[..., None] * np.array([1.0, 0.98, 0.94], np.float32) * 0.9 * glow
    out += broad[..., None] * tint * glow
    rng = np.random.default_rng(seed)
    ang = np.arctan2(dy, dx)
    sp = np.zeros_like(d)
    base = rng.uniform(0, 2 * math.pi)
    for i in range(rays):
        a0 = base + i * 2 * math.pi / rays + rng.normal(0, 0.08)
        da = np.angle(np.exp(1j * (ang - a0))).astype(np.float32)
        wdt = spike_width * rng.uniform(0.7, 1.3)
        L = ray_len * rng.uniform(0.55, 1.0)
        wd = wdt * (1 + 2.5 * dW / L) * 0.5 + 0.004 / (dW + 0.004)
        sp += np.exp(-(da / wd) ** 2) * np.exp(-dW / (L * 0.45)) * rng.uniform(0.6, 1.0)
    sp = sp * np.clip(dW / 0.006, 0, 1)
    out += sp[..., None] * (0.6 * tint + 0.4) * 0.07 * starburst
    R = 0.22
    rr = (dW - R) / 0.02
    ring = np.exp(-rr * rr)
    hue = np.stack([np.clip(0.55 + rr * 0.3, 0, 1), np.clip(1 - np.abs(rr) * 0.35, 0, 1),
                    np.clip(0.55 - rr * 0.3, 0, 1)], -1)
    out += ring[..., None] * hue * 0.018 * halo
    sy = np.exp(-(dy / (0.004 * (Wref / s) * 9 / 16)) ** 2)
    sx = np.exp(-np.abs(dx) / (0.22 * Wref / s)) * 0.6 + np.exp(-np.abs(dx) / (0.05 * Wref / s)) * 0.5
    out += (sy * sx)[..., None] * np.array([0.8, 0.9, 1.0], np.float32) * 0.3 * streak
    return cv2.resize(out, (Wc, Hc), interpolation=cv2.INTER_LINEAR)


class Flare:
    """Pre-rendered static flare around the light + per-frame ghosts."""

    def __init__(self, W, H, tint=(1.0, 0.72, 0.45), pad=1.0, **kw):
        self.W, self.H = W, H
        self.pw, self.ph = int(W * (1 + pad)) // 2 * 2, int(H * (1 + pad)) // 2 * 2
        self.cx, self.cy = self.pw / 2, self.ph / 2
        self.tint = tint
        self.plate = _flare_plate(self.pw, self.ph, self.cx, self.cy, W, tint=tint, **kw)
        rng = np.random.default_rng(4)
        self.spec = [(0.38, 0.014, (0.55, 0.85, 1.0), 6, False), (0.7, 0.04, (0.55, 1.0, 0.72), 6, False),
                     (1.08, 0.022, (1.0, 0.72, 0.5), 0, False), (1.42, 0.07, (0.6, 0.72, 1.0), 6, True),
                     (1.8, 0.028, (1.0, 0.6, 0.85), 0, False)]
        self.amts = [0.028 * rng.uniform(0.7, 1.2) for _ in self.spec]

    def add(self, img, lx, ly, intensity=1.0, ghosts=1.0, center=None):
        W, H = self.W, self.H
        M = np.array([[1, 0, lx - self.cx], [0, 1, ly - self.cy]], np.float32)
        fl = cv2.warpAffine(self.plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img += fl * intensity
        if not ghosts:
            return img
        cx, cy = (W / 2, H / 2) if center is None else center
        vx, vy = cx - lx, cy - ly
        for (k, r, col, sides, ring_), amt0 in zip(self.spec, self.amts):
            gx, gy = lx + vx * k, ly + vy * k
            R = r * W
            bx0, bx1 = int(max(gx - R * 1.3 - 2, 0)), int(min(gx + R * 1.3 + 3, W))
            by0, by1 = int(max(gy - R * 1.3 - 2, 0)), int(min(gy + R * 1.3 + 3, H))
            if bx1 <= bx0 or by1 <= by0:
                continue
            yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            ex, ey = xx - gx, yy - gy
            rad = np.sqrt(ex * ex + ey * ey)
            if sides:
                th = np.arctan2(ey, ex) + 0.3
                seg = 2 * math.pi / sides
                rad = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
            q = rad / R
            disc = np.stack([C.smoothstep(1.08, 0.85, q), C.smoothstep(1.03, 0.8, q), C.smoothstep(0.98, 0.75, q)], -1)
            edge = C.smoothstep(0.5, 1.0, q)[..., None]
            amt = amt0 * (0.45 + 0.55 * edge)
            if ring_:
                amt = amt * np.exp(-((q - 0.93) / 0.07) ** 2)[..., None] * 2.2
            if r > 0.06:
                amt = amt * 0.5
            img[by0:by1, bx0:bx1] += C.blur(disc.astype(np.float32), max(R * 0.04, 0.6)) * np.asarray(col, np.float32) * amt * ghosts * intensity
        return img


def birds_local(W, H, t, **kw):
    """sky.birds() evaluated only inside the flock's bounding box (same look, ~50x cheaper).
    Returns (mask, x0, y0)."""
    from lib import sky as S
    c = kw.get('center', (0.3, 0.3))
    v = kw.get('velocity', (0.04, -0.01))
    spread = kw.get('spread', 0.06)
    size = kw.get('size', 0.018)
    cx = (c[0] + v[0] * t) * W
    cy = (c[1] + v[1] * t) * H
    rx = (spread * 3.5 + size * 2) * W + 0.02 * W
    ry = (spread * 1.8 + size * 2) * W + 0.02 * W
    x0, x1 = int(max(cx - rx, 0)), int(min(cx + rx, W))
    y0, y1 = int(max(cy - ry, 0)), int(min(cy + ry, H))
    if x1 <= x0 or y1 <= y0:
        return None
    w, h = x1 - x0, y1 - y0
    # shift the flock into the local window; sizes stay relative to the full frame width via scaling
    k = dict(kw)
    k['center'] = ((cx - x0) / w - v[0] * t * W / w, (cy - y0) / h - v[1] * t * H / h)
    k['velocity'] = (v[0] * W / w, v[1] * H / h)
    k['size'] = size * W / w
    k['spread'] = spread * W / w
    m = S.birds(w, h, t, **k)
    return m, x0, y0
