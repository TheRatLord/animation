"""s05_sakura helpers: designed sun (crisp hot core, starburst, flare ghosts on the diagonal), light beams
slanting from canopy gaps onto the path, water / rail sparkles."""
import math
import numpy as np
import cv2


def _grid(w, h):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs, ys


class SunFlare:
    """Precomputes the angular starburst profile; call render() per frame with the on-screen sun position."""

    def __init__(self, W, H, seed=21):
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        N = 4096
        a = np.linspace(-math.pi, math.pi, N, endpoint=False)
        amp = np.zeros(N, np.float32)
        ln = np.zeros(N, np.float32)
        base = rng.uniform(0, 2 * math.pi)
        # 8 main spikes (two lengths), crisp and thin
        for i in range(8):
            ang = base + i * 2 * math.pi / 8 + rng.normal(0, 0.04)
            d = np.angle(np.exp(1j * (a - ang)))
            k = np.exp(-(d / 0.0065) ** 2)
            L = 1.0 if i % 2 == 0 else 0.55
            amp = np.maximum(amp, k * (1.0 if i % 2 == 0 else 0.7))
            ln = np.maximum(ln, k * L * rng.uniform(0.8, 1.0))
        # many hair-fine rays
        for i in range(56):
            ang = rng.uniform(-math.pi, math.pi)
            d = np.angle(np.exp(1j * (a - ang)))
            k = np.exp(-(d / 0.0035) ** 2)
            amp = np.maximum(amp, k * rng.uniform(0.15, 0.45))
            ln = np.maximum(ln, k * rng.uniform(0.15, 0.45))
        self.pa, self.pamp, self.pln = a, amp, ln
        # ghost spec: (position along sun->centre axis, radius frac W, colour, sides, opacity, ring)
        self.ghosts = [(0.3, 0.005, (1.0, 0.82, 0.6), 0, 0.22, False),
                       (0.42, 0.009, (0.6, 1.0, 0.82), 6, 0.075, False),
                       (0.6, 0.0035, (1.0, 0.92, 0.75), 0, 0.3, False),
                       (0.86, 0.013, (0.62, 0.78, 1.0), 6, 0.05, False),
                       (1.2, 0.03, (0.6, 0.72, 1.0), 6, 0.035, False),
                       (1.5, 0.02, (1.0, 0.6, 0.82), 0, 0.06, True),
                       (1.8, 0.008, (0.65, 1.0, 0.78), 6, 0.1, False)]

    def render(self, lx, ly, vis=1.0, rot=0.0, center=None):
        W, H = self.W, self.H
        s = 2
        w, h = W // s, H // s
        out = np.zeros((h, w, 3), np.float32)
        x0, y0 = lx / s, ly / s
        # --- starburst + core in a window around the sun
        Rw = int(0.36 * w)
        bx0, bx1 = int(max(x0 - Rw, 0)), int(min(x0 + Rw, w))
        by0, by1 = int(max(y0 - Rw, 0)), int(min(y0 + Rw, h))
        if bx1 > bx0 and by1 > by0:
            ys, xs = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
            dx, dy = xs - x0, ys - y0
            d = np.sqrt(dx * dx + dy * dy) + 1e-3
            dW = d / w
            ang = np.arctan2(dy, dx) - rot
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            idx = ((ang + math.pi) / (2 * math.pi) * len(self.pa)).astype(np.int32) % len(self.pa)
            amp = self.pamp[idx]
            L = self.pln[idx] * 0.16          # round 19: shorter rays (no long scratch lines across the sky)
            rays = amp * np.exp(-dW / np.maximum(L * 0.4, 1e-3)) * np.clip(dW / 0.008, 0, 1)
            R = 0.0105
            disc = np.clip((R - dW) / 0.0012 + 0.5, 0, 1)                   # crisp disc edge
            core = 1.0 / (1.0 + (dW / (R * 1.25)) ** 4)                       # hot corona
            glow = np.exp(-dW / 0.035) * 0.3 * getattr(self, 'g1', 1.0) + np.exp(-dW / 0.1) * 0.08 * getattr(self, 'g2', 1.0)
            ring = np.exp(-((dW - 0.13) / 0.006) ** 2) * 0.035               # faint rainbow halo
            hue = np.stack([np.clip(0.5 + (dW - 0.13) * 60, 0, 1), np.ones_like(dW) * 0.8,
                            np.clip(0.5 - (dW - 0.13) * 60, 0, 1)], -1)
            loc = disc[..., None] * np.array([2.2, 2.1, 1.95], np.float32) + \
                core[..., None] * np.array([1.0, 0.94, 0.82], np.float32) * 0.9 + \
                glow[..., None] * np.array([1.0, 0.86, 0.66], np.float32) + \
                rays[..., None] * np.array([1.0, 0.95, 0.86], np.float32) * 0.55 + \
                ring[..., None] * hue
            out[by0:by1, bx0:bx1] += loc.astype(np.float32)
        # --- ghosts on the diagonal through the optical centre
        cx, cy = (w / 2, h / 2) if center is None else (center[0] / s, center[1] / s)
        vx, vy = cx - x0, cy - y0
        for k, r, col, sides, op, ring_ in self.ghosts:
            gx, gy = x0 + vx * k, y0 + vy * k
            Rg = r * w
            gx0, gx1 = int(max(gx - Rg * 1.4 - 2, 0)), int(min(gx + Rg * 1.4 + 3, w))
            gy0, gy1 = int(max(gy - Rg * 1.4 - 2, 0)), int(min(gy + Rg * 1.4 + 3, h))
            if gx1 <= gx0 or gy1 <= gy0:
                continue
            ys, xs = np.mgrid[gy0:gy1, gx0:gx1].astype(np.float32)
            ex, ey = xs - gx, ys - gy
            rad = np.sqrt(ex * ex + ey * ey)
            if sides:
                th = np.arctan2(ey, ex) + 0.35
                seg = 2 * math.pi / sides
                rad = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
            aa = 1.2 / Rg
            if ring_:
                q = rad / Rg
                body = np.exp(-((q - 0.9) / 0.07) ** 2) * 1.6
                body = np.stack([body * np.clip(1.2 - (q - 0.9) * 8, 0, 1.2), body,
                                 body * np.clip(0.8 + (q - 0.9) * 8, 0, 1.2)], -1)
            else:
                # chromatic ghost: each channel has a slightly different size (red outside, blue inside)
                chans = []
                for sc in (1.07, 1.0, 0.93):
                    q = rad / (Rg * sc)
                    fill = np.clip((1 - q) / aa + 0.5, 0, 1)
                    prof = fill * (0.3 + 0.7 * np.clip(q, 0, 1) ** 3)          # brighter rim, soft interior
                    rim = np.exp(-((q - 0.96) / 0.04) ** 2) * 0.5
                    chans.append(prof + rim * fill)
                body = np.stack(chans, -1)
            out[gy0:gy1, gx0:gx1] += (body * np.asarray(col, np.float32) * op).astype(np.float32)
        # anamorphic streak (subtle)
        sy0, sy1 = int(max(y0 - 0.02 * h, 0)), int(min(y0 + 0.02 * h + 1, h))
        if sy1 > sy0:
            ys, xs = np.mgrid[sy0:sy1, 0:w].astype(np.float32)
            st = np.exp(-((ys - y0) / (0.0035 * h)) ** 2) * (np.exp(-np.abs(xs - x0) / (0.18 * w)) * 0.12 +
                                                             np.exp(-np.abs(xs - x0) / (0.04 * w)) * 0.25)
            out[sy0:sy1] += st[..., None] * np.array([0.75, 0.88, 1.0], np.float32)
        out = cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)
        return out * vis


def beams_toward(s, x0, y0, length=0.55, steps=24):
    """Light shafts rising from bright ground spots toward the sun: radially smears `s` (low-res (h, w)) so
    each spot extends toward (x0, y0) (low-res px; the vanishing point of parallel sun rays). Returns (h, w)."""
    h, w = s.shape
    acc = np.zeros_like(s)
    wsum = 0.0
    for i in range(1, steps + 1):
        k = 1.0 + length * i / steps          # sample further from the sun -> beam reaches toward it
        M = np.array([[k, 0, (1 - k) * x0], [0, k, (1 - k) * y0]], np.float32)
        wt = (1.0 - i / (steps + 1)) ** 1.5
        acc += cv2.warpAffine(s, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT) * wt
        wsum += wt
    acc /= wsum
    return cv2.GaussianBlur(acc, (0, 0), 0.8)


def sun_shafts(occ_q, x0, y0, strength=0.3, length=0.85, steps=18, radius=0.07):
    """Crepuscular rays from the sun through the gaps of an occluder (low-res (h, w) alpha). x0, y0 in
    low-res px. Returns (h, w) intensity (soft-compressed)."""
    h, w = occ_q.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt((xs - x0) ** 2 + (ys - y0) ** 2) / w
    src = (np.exp(-(d / radius) ** 2) * 0.7 + 0.2 / (1 + (d / 0.03) ** 2)) * (1 - occ_q) ** 1.5
    src = src.astype(np.float32)
    acc = np.zeros_like(src)
    wsum = 0.0
    for i in range(steps):
        k = 1.0 - length * (i / steps) ** 1.15
        M = np.array([[k, 0, (1 - k) * x0], [0, k, (1 - k) * y0]], np.float32)
        wt = 1.0 - 0.5 * i / steps
        acc += cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE) * wt
        wsum += wt
    acc = cv2.GaussianBlur(acc / wsum, (0, 0), 1.0) * (1 - 0.85 * occ_q)
    x = acc * strength
    return x / (1 + 0.9 * x)


def sparkles(W, H, xs, ys, size, inten, t, seed, color=(1.0, 0.96, 0.88), out=None):
    """Tiny 4-point star sparkles (+ soft core) that twinkle coherently. Returns (H, W, 3) additive."""
    n = len(xs)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.283, n)
    fr = rng.uniform(1.5, 4.0, n)
    tw = np.clip(np.sin(t * fr + ph), 0, 1) ** 3
    if out is None:
        out = np.zeros((H, W, 3), np.float32)
    col = np.asarray(color, np.float32)
    for i in range(n):
        a = inten[i] * tw[i]
        if a < 0.01:
            continue
        L = size[i]
        R = int(L) + 2
        cx, cy = xs[i], ys[i]
        x0, x1 = max(int(cx) - R, 0), min(int(cx) + R + 1, W)
        y0, y1 = max(int(cy) - R, 0), min(int(cy) + R + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xx - cx, yy - cy
        wd = 0.5 + 0.03 * L
        k = np.exp(-(dy / wd) ** 2) * np.clip(1 - np.abs(dx) / L, 0, 1) ** 3 + \
            np.exp(-(dx / wd) ** 2) * np.clip(1 - np.abs(dy) / (L * 0.8), 0, 1) ** 3
        k += np.exp(-(dx * dx + dy * dy) / max((0.12 * L) ** 2, 0.6)) * 1.2
        out[y0:y1, x0:x1] += k[..., None] * col * a
    return out


def bloom_fast(img, threshold=0.9, knee=0.3, strength=0.3, halation=0.14, W=None):
    """Cheap multi-scale bloom + warm halation computed at quarter resolution (one upsample)."""
    H, Wd = img.shape[:2]
    q = 4
    small = cv2.resize(img, (Wd // q, H // q), interpolation=cv2.INTER_AREA)
    lum = small.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1) ** 2
    br = small * k
    acc = np.zeros_like(br)
    wsum = 0.0
    for r, wt in ((0.003, 1.0), (0.01, 0.8), (0.025, 0.5), (0.06, 0.25)):
        sig = max(r * Wd / q, 0.6)
        if sig > 6:
            f = sig / 3.0
            sm = cv2.resize(br, (max(int(br.shape[1] / f), 2), max(int(br.shape[0] / f), 2)),
                            interpolation=cv2.INTER_AREA)
            sm = cv2.GaussianBlur(sm, (0, 0), 2.6)
            b = cv2.resize(sm, (br.shape[1], br.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            b = cv2.GaussianBlur(br, (0, 0), sig)
        acc += b * wt
        wsum += wt
    tot = acc * (strength / wsum)
    if halation:
        hal = cv2.GaussianBlur(br, (0, 0), max(0.006 * Wd / q, 0.6))
        tot += hal * (np.array([1.0, 0.45, 0.2], np.float32) * halation * 0.5)
    img += cv2.resize(tot, (Wd, H), interpolation=cv2.INTER_LINEAR)
    return img
