"""s05_sakura round 19: the land-side hedge repainted as painted leaf-cluster MASSES.

Reviewer note (round 18): 'repeated round bobble clumps in saturated uniform green, stylised game-asset look ->
leaf-cluster masses with a warm lit top, a cool blue-green shadow underside and a crisp silhouette fringe; tone
the saturation down so it doesn't compete with the blossoms'.

  * irregular masses of varied width (0.4-1.3x the local hedge height) in two overlapping tiers, the upper tier
    overlapping the lower one;
  * inside each mass the value is painted in 4 flat, muted bands (cool blue-green shade, grey-green mid, warm
    olive lit, pale warm tip) from a hemispherical form lit from the top-left sun; the band edges are broken by
    a leaf-scale noise so they read as crisp ragged leaf clumps, not a cel-shaded ball;
  * the silhouette (upper edge of each mass) is broken by small leaf dabs = a crisp fringe; the underside of
    each mass has no edge at all (lost into the shade body);
  * the large-scale light of the original render (canopy shade, dapple) is kept as a multiplier.
"""
import numpy as np
import cv2

from s05_sakura_r16 import _splat, _unit

BANDS = np.array([(0.14, 0.23, 0.26), (0.21, 0.34, 0.35), (0.33, 0.46, 0.33), (0.53, 0.62, 0.34),
                  (0.74, 0.76, 0.46)], np.float32)
PETAL = (1.0, 0.84, 0.9)


def _vnoise(rng, h, w, cell):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:h, :w]


def repaint(pm, r0, W, seed=19, sun=(-0.8, -0.6)):
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
    sig = 10 * u
    Lb = cv2.GaussianBlur(lum, (0, 0), sig) / np.maximum(cv2.GaussianBlur((a > 0.5).astype(np.float32), (0, 0), sig), 1e-3)
    ref = np.percentile(Lb[a > 0.5], 80)
    mul = np.clip(Lb / max(ref, 1e-3), 0.7, 1.12)
    mul = np.concatenate([np.repeat(mul[:1], pad, 0), mul], 0)
    Ls = _unit(sun)
    L3 = _unit([Ls[0] * 0.7, Ls[1] * 0.7 - 0.2, 0.62])
    leaf = max(1.2, 2.6 * u)
    N1 = _vnoise(rng, H2, Wc, leaf * 2.4)
    N2 = _vnoise(rng, H2, Wc, leaf * 5.0)
    # ---- masses
    masses = []
    x = float(cols[0])
    while x <= cols[-1]:
        xi = int(min(x, Wc - 1))
        if top[xi] < 0:
            x += 1
            continue
        hh = max(4.0, bot[xi] - top[xi])
        w = hh * float(np.clip(np.exp(rng.normal(-0.3, 0.35)), 0.35, 1.05))
        t0 = top[xi] + pad
        # upper tier: a gently scalloped line of masses, each with 1-3 smaller cauliflower heads on top
        cy0 = t0 + hh * rng.uniform(0.2, 0.32)
        ry0 = hh * rng.uniform(0.24, 0.32)
        masses.append((0, x + w * 0.5, cy0, w * 0.62, ry0, rng.uniform(-0.3, 0.2)))
        for _ in range(int(rng.integers(1, 4))):
            rs = rng.uniform(0.3, 0.55)
            masses.append((0, x + w * rng.uniform(0.15, 0.85), cy0 - ry0 * rng.uniform(0.3, 0.7), w * 0.62 * rs,
                           ry0 * rs * 1.3, rng.uniform(0.0, 0.35)))
        # lower tier (scattered), shaded by the upper
        for _ in range(2):
            masses.append((1, x + w * rng.uniform(-0.2, 1.2), t0 + hh * rng.uniform(0.5, 0.95),
                           w * rng.uniform(0.45, 0.8), hh * rng.uniform(0.18, 0.3), rng.uniform(-1.9, -1.3)))
        x += w * rng.uniform(0.7, 1.0)
    C = np.zeros((H2, Wc, 3), np.float32)
    A = np.zeros((H2, Wc), np.float32)
    # dark shade body so the hedge stays solid
    base = np.zeros((H2, Wc), np.float32)
    base[pad:] = (a > 0.5)
    base = cv2.erode(base, np.ones((3, 3), np.uint8))
    base = base * (np.arange(H2)[:, None] >= pad + top[None, :] + 0.25 * (bot - top)[None, :])
    C += base[..., None] * (BANDS[0] * 0.6 + BANDS[1] * 0.4)
    A += base
    fringe = []
    order = sorted(range(len(masses)), key=lambda i: (-masses[i][0], rng.random()))
    for i in order:
        tier, cx, cy, rx, ry, vo = masses[i]
        x0, x1 = int(max(0, cx - rx * 1.2 - 2)), int(min(Wc, cx + rx * 1.2 + 3))
        y0, y1 = int(max(0, cy - ry * 1.3 - 2)), int(min(H2, cy + ry * 1.3 + 3))
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        nx = (xx + 0.5 - cx) / rx
        ny = (yy + 0.5 - cy) / ry
        # leaf-ragged boundary: radius modulated by noise, flat-ish bottom lost into the body
        rn = np.sqrt(nx * nx + ny * ny)
        rag = 1.0 + 0.22 * (N1[y0:y1, x0:x1] - 0.5) + 0.12 * (N2[y0:y1, x0:x1] - 0.5)
        inside = np.clip((rag - rn) * min(rx, ry) / 0.8 + 0.5, 0, 1)
        inside *= np.clip((1.25 - ny) * 3.0, 0, 1)                 # the lower edge is lost (no outline)
        nz = np.sqrt(np.clip(1 - np.minimum(rn, 1) ** 2, 0, 1))
        lam = (nx * L3[0] + ny * L3[1] + nz * L3[2]) / np.maximum(np.sqrt(nx * nx + ny * ny + nz * nz), 1e-3)
        v = 1.3 + 2.5 * lam + vo - 0.8 * np.clip(ny, 0, None) + 0.55 * (N1[y0:y1, x0:x1] - 0.5) \
            + 0.4 * (N2[y0:y1, x0:x1] - 0.5)
        vq = np.clip(np.floor(v), 0, 4).astype(int)
        col = BANDS[vq]
        sub_c = C[y0:y1, x0:x1]
        sub_a = A[y0:y1, x0:x1]
        C[y0:y1, x0:x1] = sub_c * (1 - inside[..., None]) + col * inside[..., None]
        A[y0:y1, x0:x1] = sub_a + inside * (1 - sub_a)
        # leaf dabs over the painted mass (leaf texture inside the value bands)
        nd = int(np.clip(rx * ry * 3.1416 / (leaf * leaf) * 0.7, 4, 4000))
        ang = rng.uniform(0, 2 * np.pi, nd)
        rad = np.sqrt(rng.uniform(0, 1, nd)) * 1.02
        dx, dy = np.cos(ang) * rad, np.sin(ang) * rad
        keepd = dy < 0.95
        dx, dy = dx[keepd], dy[keepd]
        dz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
        lamd = (dx * L3[0] + dy * L3[1] + dz * L3[2])
        vd = 1.3 + 2.5 * lamd + vo - 0.8 * np.clip(dy, 0, None) + rng.normal(0, 0.45, len(dx))
        od = np.argsort(vd)
        dst = []
        for k in od:
            vk = int(np.clip(np.floor(vd[k]), 0, 4))
            cc = BANDS[vk] * rng.uniform(0.95, 1.05)
            dst.append((cx + dx[k] * rx, cy + dy[k] * ry, leaf * rng.uniform(0.6, 1.25), rng.uniform(0, 6.3), 2.0,
                        0.45, 0.7, cc[0], cc[1], cc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
        if dst:
            _splat(C, A, np.array(dst, np.float64))
        # crisp leaf fringe on the upper silhouette of the upper tier
        if tier == 0:
            nfr = int(np.clip(rx * 2.2 / leaf, 4, 240))
            ang = rng.uniform(-np.pi * 0.98, -0.02 * np.pi, nfr)
            rad = rng.uniform(0.92, 1.1, nfr)
            for k in range(nfr):
                fx = cx + np.cos(ang[k]) * rx * rad[k]
                fy = cy + np.sin(ang[k]) * ry * rad[k]
                lamk = (np.cos(ang[k]) * L3[0] + np.sin(ang[k]) * L3[1]) + 0.25
                vk = int(np.clip(np.floor(1.4 + 2.6 * lamk + vo + rng.normal(0, 0.4)), 1, 4))
                cc = BANDS[vk] * rng.uniform(0.95, 1.05)
                fringe.append((fx, fy, leaf * rng.uniform(0.7, 1.3), rng.uniform(0, 6.3), 2.0, 0.5, 0.7,
                               cc[0], cc[1], cc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
    if fringe:
        _splat(C, A, np.array(fringe, np.float64))
    # fallen petals on the lit tops
    gy = cv2.Sobel(A, cv2.CV_32F, 0, 1, ksize=3)
    ys, xs = np.nonzero((gy > 0.3) & (A > 0.1))
    n = int(len(ys) * 0.025)
    pst = []
    if n:
        pick = rng.choice(len(ys), n, replace=False)
        for i in pick:
            col = np.array(PETAL, np.float32) * rng.uniform(0.9, 1.04)
            pst.append((xs[i] + rng.uniform(-2, 2), ys[i] + rng.uniform(1, 6 * u), u * rng.uniform(0.9, 1.6),
                        rng.uniform(0, 6.3), 2.0, 0.3, 0.7, col[0], col[1], col[2], 0, 0, 9.0, 0, 0, 0, 1.0))
        _splat(C, A, np.array(pst, np.float64))
    C = C * mul[..., None]
    out = np.concatenate([C, np.clip(A, 0, 1)[..., None]], -1).astype(np.float32)   # premultiplied
    return out, r0 - pad
