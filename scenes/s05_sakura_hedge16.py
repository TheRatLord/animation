"""s05_sakura round 16: the land-side hedge repainted as LEAF-CLUSTER clumps (not clay domes).

Reviewer note: 'uniform green domes with a clay / ball look'.  Every clump is built from many small leaf dabs
(s05_sakura_r16._splat stamps) whose value is quantised into flat painted bands: sunlit yellow-green on the
up / sun-facing side of each clump and of the hedge top, mid green body, blue-green shade underneath, deep
teal in the crevices.  The dabs break the silhouette into a cauliflower leaf edge.  The large-scale light of
the original render (tree shade, dapple) is kept as a multiplier.  A few fallen petals rest on the top.
"""
import numpy as np
import cv2

from s05_sakura_r16 import _splat, _unit

BANDS = np.array([(0.05, 0.15, 0.19), (0.1, 0.28, 0.31), (0.2, 0.42, 0.3), (0.38, 0.6, 0.27),
                  (0.62, 0.78, 0.33), (0.82, 0.9, 0.45)], np.float32)
PETAL = (1.0, 0.84, 0.9)


def repaint(pm, r0, W, seed=16, sun=(-0.8, -0.6)):
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
    pad = int(30 * u) + 4
    H2 = h + pad
    straight = pm[..., :3] / np.maximum(a, 1e-4)[..., None]
    lum = straight.mean(-1) * (a > 0.5)
    sig = 10 * u
    Lb = cv2.GaussianBlur(lum, (0, 0), sig) / np.maximum(cv2.GaussianBlur((a > 0.5).astype(np.float32), (0, 0), sig), 1e-3)
    ref = np.percentile(Lb[a > 0.5], 80)
    mul = np.clip(Lb / max(ref, 1e-3), 0.5, 1.12)
    mul = np.concatenate([np.repeat(mul[:1], pad, 0), mul], 0)
    Ls = _unit(sun)
    st_rows = []
    x = int(cols[0])
    while x <= cols[-1]:
        if top[x] < 0:
            x += 1
            continue
        hh = max(3.0, bot[x] - top[x])
        r = hh * rng.uniform(0.16, 0.26)
        fys = (0.0, 0.25, 0.5, 0.75, 0.95) if hh > 10 * u else (0.0, 0.6)
        for row, fy in enumerate(fys):
            rr = r * rng.uniform(0.8, 1.2)
            cx = x + rng.normal(0, 0.25) * rr
            cy = top[x] + pad + fy * hh + (0.5 * rr if row == 0 else 0) + rng.normal(0, 0.05) * hh
            vb = 3.0 - 2.6 * fy + rng.normal(0, 0.25)
            st_rows.append((row, cx, cy, rr, vb))
        x += int(max(2, r * rng.uniform(1.0, 1.5)))
    st = []
    leaf = max(1.2, 3.2 * u)
    # lower rows first, the top row overlaps them
    for (row, cx, cy, rr, vb) in sorted(st_rows, key=lambda t: -t[0]):
        n = int(np.clip((rr / leaf) ** 2 * 1.6, 6, 400))
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = rr * np.sqrt(rng.uniform(0, 1, n))
        dx, dy = np.cos(ang) * rad, np.sin(ang) * rad * 0.85
        facing = (dx * Ls[0] + dy * Ls[1]) / rr
        v = vb + 1.5 * facing - 0.9 * np.clip(dy / rr, 0, 1) + rng.normal(0, 0.3, n)
        order = np.argsort(v)
        for i in order:
            vq = int(np.clip(np.round(v[i]), 0, 5))
            col = BANDS[vq] * rng.uniform(0.94, 1.06)
            lr = leaf * rng.uniform(0.7, 1.4)
            aa = (0.7 if vq >= 3 else 1.1) * u
            st.append((cx + dx[i], cy + dy[i], lr, rng.uniform(0, 6.3), 2.0, 0.45, max(aa, 0.6),
                       col[0], col[1], col[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
    C = np.zeros((H2, Wc, 3), np.float32)
    A = np.zeros((H2, Wc), np.float32)
    # a dark body under the leaf dabs so the hedge stays solid
    base = np.zeros((H2, Wc), np.float32)
    base[pad:] = (a > 0.5)
    base = cv2.erode(base, np.ones((3, 3), np.uint8))
    base = base * (np.arange(H2)[:, None] >= pad + top[None, :] + 0.2 * (bot - top)[None, :])
    C += base[..., None] * BANDS[1] * 0.8
    A += base
    _splat(C, A, np.array(st, np.float64))
    # fallen petals on the sun-lit tops
    gy = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)
    ys, xs = np.nonzero((gy > 0.3) & (A > 0.1))
    n = int(len(ys) * 0.04)
    pst = []
    if n:
        pick = rng.choice(len(ys), n, replace=False)
        for i in pick:
            col = np.array(PETAL, np.float32) * rng.uniform(0.9, 1.04)
            pst.append((xs[i] + rng.uniform(-2, 2), ys[i] + rng.uniform(1, 6 * u), u * rng.uniform(0.9, 1.8),
                        rng.uniform(0, 6.3), 2.0, 0.3, 0.7, col[0], col[1], col[2], 0, 0, 9.0, 0, 0, 0, 1.0))
        _splat(C, A, np.array(pst, np.float64))
    C = C * mul[..., None]
    out = np.concatenate([C, np.clip(A, 0, 1)[..., None]], -1).astype(np.float32)   # premultiplied
    return out, r0 - pad
