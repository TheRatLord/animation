"""s05_sakura round 20: land-side hedge repainted as clustered small-leaf CAULIFLOWER masses.

Reviewer (round 19): 'hedge at the right = green bubbly clay / pixel blob with no painted leaf structure -> a
clustered small-leaf cauliflower edge, a warm lit top, a cool dark underside and lost edges'.

  * the hedge is split into irregular cauliflower heads (a big head + 2-4 smaller bumps on its shoulders) in two
    overlapping tiers; each head is painted as FLAT value planes from the top-left sun (warm yellow-olive lit
    cap, sap-green mid, cool blue-green shade, deep teal core) with clean boundaries - no per-pixel noise;
  * the boundary between planes and the whole upper silhouette are built from leaf SPRIGS: 3-6 pointed leaves
    radiating from one point, lit leaf tips a notch lighter -> a crisp small-leaf edge;
  * the underside of every head is lost into a soft dark body (no outline, gradient into shade);
  * the large-scale canopy shade / dapple of the original render is kept as a multiplier.
"""
import math

import numpy as np
import cv2

from s05_sakura_r16 import _splat, _unit

BANDS = np.array([(0.07, 0.13, 0.16), (0.12, 0.22, 0.25), (0.2, 0.34, 0.27), (0.36, 0.49, 0.27),
                  (0.6, 0.66, 0.33), (0.8, 0.8, 0.48)], np.float32)
PETAL = (1.0, 0.84, 0.9)


def _sprig(st, x, y, size, ang0, col, tip, Ls, rng, nl=None):
    nl = nl or int(rng.integers(3, 7))
    for i in range(nl):
        a = ang0 + rng.normal(0, 0.9)
        L = size * rng.uniform(0.7, 1.2)
        cx, cy = x + math.cos(a) * L * 0.55, y + math.sin(a) * L * 0.55
        facing = math.cos(a) * Ls[0] + math.sin(a) * Ls[1]
        c = col + (tip - col) * max(facing, 0.0) * 0.8
        c = c * rng.uniform(0.94, 1.06)
        # elongated leaf (nb=2 lobes along its axis) with a crisp edge
        st.append((cx, cy, L * 0.5, a, 2.0, 0.62, 0.6, c[0], c[1], c[2], Ls[0], Ls[1], 0.3,
                   (tip - c)[0] * 0.5, (tip - c)[1] * 0.5, (tip - c)[2] * 0.5, 1.0))


def repaint(pm, r0, W, seed=20, sun=(-0.8, -0.6)):
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
    pad = int(max(24 * u, 0.3 * float((bot - top).max()))) + 4
    H2 = h + pad
    straight = pm[..., :3] / np.maximum(a, 1e-4)[..., None]
    lum = straight.mean(-1) * (a > 0.5)
    sig = 12 * u
    Lb = cv2.GaussianBlur(lum, (0, 0), sig) / np.maximum(cv2.GaussianBlur((a > 0.5).astype(np.float32), (0, 0), sig), 1e-3)
    ref = np.percentile(Lb[a > 0.5], 80)
    mul = np.clip(Lb / max(ref, 1e-3), 0.65, 1.12)
    mul = np.concatenate([np.repeat(mul[:1], pad, 0), mul], 0)
    Ls = _unit(sun)
    L3 = _unit([Ls[0] * 0.7, Ls[1] * 0.7 - 0.25, 0.6])
    # ---- leaf clumps: many small-leaf cauliflower clumps in 3 loose tiers, each painted ONLY with leaf dabs
    #      (value from the clump's own form + jitter -> leafy plane boundaries, no smooth cel rings)
    leaf = max(1.6, 2.6 * u)
    clumps = []
    x = float(cols[0])
    while x <= cols[-1]:
        xi = int(min(x, Wc - 1))
        if top[xi] < 0:
            x += 1
            continue
        hh = max(4.0, bot[xi] - top[xi])
        w = hh * rng.uniform(0.3, 0.6)
        for tier in range(3):
            r = hh * rng.uniform(0.2, 0.34) * (1.0 - 0.12 * tier)
            cy = top[xi] + pad + hh * (0.2 + 0.27 * tier) + rng.normal(0, 0.05) * hh
            clumps.append((tier, x + w * rng.uniform(0.2, 0.8), cy, r * rng.uniform(1.0, 1.5), r,
                           [0.35, -0.25, -0.9][tier] + rng.normal(0, 0.2)))
        x += w
    C = np.zeros((H2, Wc, 3), np.float32)
    A = np.zeros((H2, Wc), np.float32)
    # dark body: deep teal, a touch lighter toward the top
    base = np.zeros((H2, Wc), np.float32)
    base[pad:] = (a > 0.5)
    yy_all = np.arange(H2, dtype=np.float32)[:, None]
    rows_b = pad + bot[None, :]
    rows_t = pad + top[None, :]
    fy = np.clip((yy_all - rows_t) / np.maximum(rows_b - rows_t, 1.0), 0, 1)
    body = BANDS[1][None, None, :] * (1 - fy[..., None]) + BANDS[0][None, None, :] * fy[..., None]
    C += base[..., None] * body
    A += base
    order = sorted(range(len(clumps)), key=lambda i: (-clumps[i][0], rng.random()))
    st = []
    for i in order:
        tier, cx, cy, rx, ry, vo = clumps[i]
        n = int(np.clip(rx * ry * 3.1416 / (leaf * leaf * 0.8), 8, 3000))
        ang = rng.uniform(0, 2 * np.pi, n)
        rad = np.sqrt(rng.uniform(0, 1, n))
        # scalloped outline: small bumps on the clump rim
        bump = 1.0 + 0.12 * np.sin(6 * ang + cx) + 0.06 * np.sin(11 * ang + cy)
        dx, dy = np.cos(ang) * rad * bump, np.sin(ang) * rad * bump
        dz = np.sqrt(np.clip(1 - (dx * dx + dy * dy) / (bump * bump), 0, 1))
        lam = (dx * L3[0] + dy * L3[1] + dz * L3[2]) / np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-3)
        v = 1.3 + 2.8 * lam + vo - 1.1 * np.clip(dy, 0, None) + rng.normal(0, 0.4, n)
        o = np.argsort(dz + rng.normal(0, 0.15, n))          # rim leaves first, crown leaves on top
        for k in o:
            vk = int(np.clip(math.floor(v[k]), 0, 4))
            c = BANDS[vk] * rng.uniform(0.94, 1.06)
            tip = BANDS[min(vk + 1, 5)]
            la = math.atan2(dy[k], dx[k]) + rng.normal(0, 0.8)
            L = leaf * rng.uniform(0.75, 1.3)
            st.append((cx + dx[k] * rx, cy + dy[k] * ry, L, la, 2.0, 0.6, 0.6, c[0], c[1], c[2], Ls[0], Ls[1],
                       0.35, (tip - c)[0] * 0.45, (tip - c)[1] * 0.45, (tip - c)[2] * 0.45, 1.0))
    if st:
        _splat(C, A, np.array(st, np.float64))
    C = C / np.maximum(A, 1e-4)[..., None]
    # keep inside the hedge (a leafy fringe may rise a little above its top), lost darker bottom
    allow = np.zeros((H2, Wc), np.float32)
    allow[pad:] = (a > 0.3)
    up = int(max(2, 0.12 * float(np.max(bot - top))))
    allow = np.maximum(allow, cv2.dilate(allow, np.ones((up * 2 + 1, 1), np.uint8)) *
                       (yy_all < rows_t + 2))
    allow = np.where(top[None, :] >= 0, allow, 0.0).astype(np.float32)
    A = A * allow
    body_col = BANDS[0]
    k_low = np.clip((yy_all - (rows_t + 0.6 * (rows_b - rows_t))) / np.maximum(0.4 * (rows_b - rows_t), 1.0), 0, 1)
    k_low = np.where(top[None, :] >= 0, k_low, 0.0).astype(np.float32)
    C = C + (body_col - C) * (0.75 * k_low)[..., None]
    # fallen petals on the lit tops
    gy = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)
    ys, xs = np.nonzero((gy > 0.3) & (A > 0.1))
    n = int(len(ys) * 0.02)
    Cp = C * A[..., None]
    if n:
        pick = rng.choice(len(ys), n, replace=False)
        pst = []
        for i in pick:
            col = np.array(PETAL, np.float32) * rng.uniform(0.9, 1.04)
            pst.append((xs[i] + rng.uniform(-2, 2), ys[i] + rng.uniform(1, 6 * u), u * rng.uniform(0.9, 1.6),
                        rng.uniform(0, 6.3), 2.0, 0.3, 0.7, col[0], col[1], col[2], 0, 0, 9.0, 0, 0, 0, 1.0))
        Cs = C.copy()
        As = A.copy()
        _splat(Cs, As, np.array(pst, np.float64))
        Cp = Cs * As[..., None]
        A = As
    Cp = Cp * mul[..., None]
    out = np.concatenate([Cp, np.clip(A, 0, 1)[..., None]], -1).astype(np.float32)   # premultiplied
    return out, r0 - pad
