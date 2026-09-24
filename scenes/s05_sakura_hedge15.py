"""s05_sakura round 15: the land-side hedge repainted in screen space (on its parallax card).

Reviewer note: 'flat green slab with a noisy texture'.  The hedge becomes rows of cauliflower leaf clumps
painted like the blossom clusters: each clump a scalloped shape with a crisp lit crescent toward the sun
(upper left) and a cool teal underside; the top row protrudes above the old box top -> a lumpy cauliflower
silhouette with a thin warm-lime rim; lower rows overlap the undersides of the row above.  The large-scale
light of the original render (tree shade, dapple) is kept as a multiplier.  Fallen petals rest on the top.
"""
import math

import numpy as np
import cv2

import s05_sakura_r15 as R15

COLS = dict(rim=(0.8, 0.88, 0.52), lit=(0.46, 0.64, 0.3), mid=(0.24, 0.43, 0.27), shade=(0.12, 0.27, 0.28),
            deep=(0.06, 0.15, 0.19))
PETAL = (1.0, 0.82, 0.88)


def _cmap(v):
    c = [np.array(COLS[k], np.float32) for k in ('deep', 'shade', 'mid', 'lit')]
    out = np.broadcast_to(c[0], v.shape + (3,)).astype(np.float32).copy()
    for i, lv in enumerate((-0.55, -0.1, 0.35)):
        st = R15._sstep((v - lv) / 0.05 + 0.5)[..., None]
        out += st * (c[i + 1] - c[i])
    return out


def repaint(pm, r0, W, seed=15, sun=(-0.8, -0.6)):
    """pm: premultiplied (h, Wc, 4) hedge card whose row 0 is screen row r0.  Returns (pm2, r0_2)."""
    rng = np.random.default_rng(seed)
    u = W / 1920.0
    a = pm[..., 3]
    h, Wc = a.shape
    cols = np.nonzero(a.max(0) > 0.5)[0]
    if len(cols) == 0:
        return pm, r0
    top = np.full(Wc, -1.0)
    bot = np.full(Wc, -1.0)
    for x in cols:
        ys = np.nonzero(a[:, x] > 0.5)[0]
        top[x], bot[x] = ys[0], ys[-1]
    pad = int(40 * u) + 4
    H2 = h + pad
    straight = pm[..., :3] / np.maximum(a, 1e-4)[..., None]
    lum = straight.mean(-1) * (a > 0.5)
    sig = 10 * u
    Lb = cv2.GaussianBlur(lum, (0, 0), sig) / np.maximum(cv2.GaussianBlur((a > 0.5).astype(np.float32), (0, 0), sig), 1e-3)
    ref = np.percentile(Lb[a > 0.5], 80)
    mul = np.clip(Lb / max(ref, 1e-3), 0.45, 1.15)
    mul = np.concatenate([np.repeat(mul[:1], pad, 0), mul], 0)
    C = np.zeros((H2, Wc, 3), np.float32)
    A = np.zeros((H2, Wc), np.float32)
    Ls = R15._unit(sun)
    # interpolate top / height over columns
    xs_ok = cols
    x = int(xs_ok[0])
    rows = []
    while x <= xs_ok[-1]:
        if top[x] < 0:
            x += 1
            continue
        hh = max(2.0, bot[x] - top[x])
        rows.append((x, top[x] + pad, hh))
        x += int(max(2, hh * rng.uniform(0.08, 0.3)))
    # three rows of clumps: top row (silhouette), mid, low; painted top-first, lower rows overlap
    for row, (fy, fr, lift) in enumerate(((0.0, 0.17, 0.45), (0.22, 0.16, 0.0), (0.44, 0.16, 0.0), (0.66, 0.15, 0.0), (0.88, 0.14, 0.0))):
        for (x, ty, hh) in rows:
            if row > 0 and hh < 6 * u:
                continue
            if row > 0 and rng.random() < 0.3:
                continue
            r = hh * fr * rng.uniform(0.6, 1.7)
            cx = x + rng.normal(0, 0.3) * r
            cy = ty + (fy + rng.normal(0, 0.06)) * hh + (0.35 * r if row == 0 else 0) - lift * r * rng.uniform(0.0, 0.8)
            _clump(C, A, cx, cy, r, rng, Ls, u, row, (cy - ty) / max(hh, 1.0))
    # keep the hedge's footprint below the top row (no gaps at the bottom / sides)
    base = np.zeros((H2, Wc), np.float32)
    base[pad:] = (a > 0.5)
    base = cv2.erode(base, np.ones((3, 3), np.uint8))
    fill = base * (1 - A)
    C = C + fill[..., None] * np.array(COLS['deep'], np.float32) * 1.4
    A = A + fill
    # fallen petals on the clump tops (sun-lit tops only)
    gy = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)
    topmask = (gy > 0.3) & (A > 0.1)
    ys, xs = np.nonzero(topmask)
    n = int(len(ys) * 0.05)
    if n:
        pick = rng.choice(len(ys), n, replace=False)
        for i in pick:
            yy = ys[i] + rng.uniform(0, 6 * u)
            xx = xs[i] + rng.uniform(-2, 2)
            rr = u * rng.uniform(1.0, 2.2)
            col = np.array(PETAL, np.float32) * rng.uniform(0.9, 1.05)
            cv2.ellipse(C, (int(xx * 4), int(yy * 4)), (max(1, int(rr * 4)), max(1, int(rr * 2.4))),
                        rng.uniform(0, 180), 0, 360, tuple(float(q) for q in col), -1, cv2.LINE_AA, 2)
    # original light as a multiplier (only on the old body; the new bumps take the top row's light)
    C = C * mul[..., None]
    A = np.clip(A, 0, 1)
    out = np.concatenate([C, A[..., None]], -1).astype(np.float32)      # C is already premultiplied
    return out, r0 - pad


def _clump(C, A, cx, cy, r, rng, Ls, u, row, depth=0.0):
    ext = int(r * 1.5) + 3
    X0, Y0 = int(cx) - ext, int(cy) - ext
    w = h = 2 * ext
    Hc, Wc = A.shape
    xa, ya, xb, yb = max(0, X0), max(0, Y0), min(Wc, X0 + w), min(Hc, Y0 + h)
    if xb <= xa or yb <= ya:
        return
    m8 = np.zeros((h, w), np.uint8)
    for k in range(int(rng.integers(2, 4))):
        rl = r * rng.uniform(0.55, 0.85)
        ox, oy = rng.normal(0, 0.35) * r, rng.normal(0, 0.15) * r
        poly = R15._lobe_poly(rng, rl, int(np.clip(rl / (1.6 * u), 5, 12)), sq=0.8) + (cx + ox - X0, cy + oy - Y0)
        cv2.fillPoly(m8, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    m = m8.astype(np.float32) / 255.0
    d1 = max(1.2 * u, 0.22 * r)
    M = np.float32([[1, 0, Ls[0] * d1], [0, 1, Ls[1] * d1]])
    ms = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    lit = cv2.GaussianBlur(np.clip(m - ms, 0, 1), (0, 0), max(0.5, 0.1 * d1))
    d2 = 0.5 * r
    M = np.float32([[1, 0, -Ls[0] * d2 * 0.4], [0, 1, d2]])
    mu = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    und = cv2.GaussianBlur(np.clip(m - mu, 0, 1), (0, 0), max(0.8, 0.3 * r))
    yy = (np.arange(h, dtype=np.float32)[:, None] + Y0 - cy) / max(r, 1.0)
    v = 0.2 - 0.25 * yy + 0.6 * lit - 0.4 * und - 0.75 * depth + rng.normal(0, 0.06)
    col = _cmap(v)
    # small leaf flecks on the lit crescent (a step lighter, crisp)
    nf = int(r * r / (3.0 * u * u) * 0.08)
    for _ in range(nf):
        fx, fy = rng.uniform(0, w), rng.uniform(0, h)
        iy, ix = int(fy), int(fx)
        if 0 <= iy < h and 0 <= ix < w and lit[iy, ix] > 0.2 and m[iy, ix] > 0.9:
            c = tuple(float(q) for q in np.array(COLS['rim'], np.float32) * rng.uniform(0.8, 0.95))
            cv2.ellipse(col, (int(fx * 4), int(fy * 4)), (max(1, int(1.6 * u * 4)), max(1, int(0.9 * u * 4))),
                        rng.uniform(0, 180), 0, 360, c, -1, cv2.LINE_AA, 2)
    # thin warm-lime rim on the very top edge of the top row
    if row == 0:
        d0 = max(1.0, 1.2 * u)
        M = np.float32([[1, 0, 0], [0, 1, -d0]])
        m0 = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        rim = np.clip(m - m0, 0, 1) * np.clip(lit * 2, 0, 1)
        col = col + (np.array(COLS['rim'], np.float32) - col) * (rim * 0.8)[..., None]
    sl = (slice(ya, yb), slice(xa, xb))
    mm = m[ya - Y0:yb - Y0, xa - X0:xb - X0]
    cc = col[ya - Y0:yb - Y0, xa - X0:xb - X0]
    C[sl] = C[sl] * (1 - mm[..., None]) + cc * mm[..., None]
    A[sl] = A[sl] + mm * (1 - A[sl])
