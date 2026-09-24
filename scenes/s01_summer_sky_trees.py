"""s01 helper: Shinkai-style broadleaf trees for the rooftop band.

A canopy is a handful of leaf CLUMPS; every clump is painted from many small leaf-cluster dabs (not
one outlined blob): cool blue-green undersides first, mid greens, then sunlit yellow-green dabs on the
upper / sunward side and a few hot near-white-yellow tips. The outline is broken by loose dabs and
stray leaf tufts, and small sky holes are punched through the upper canopy. No dark outline."""
import math
import numpy as np
import cv2


def _h(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


BANDS = [_h('#173a48'), _h('#23564f'), _h('#3a7a4c'), _h('#7aae48'), _h('#c8e070')]
HOT = _h('#f0f8b0')
TRUNK = _h('#2a2e3c')


def tree(cv, x, y, r, s, rng, hk, hz, lean=0.0):
    """Paint a tree into canvas cv (plate coords): (x, y) = trunk top / canopy base centre, r = radius."""
    rng = np.random.default_rng(int(rng.integers(1 << 30)))
    r = r * 0.88
    y = y + 0.12 * r
    L = np.array([0.35, -0.8, 0.5], np.float32)
    L /= np.linalg.norm(L)
    pts = np.array([[x - 1.5 * r, y - 1.45 * r], [x + 1.5 * r, y + 1.3 * r]])
    bb = cv._bbox(pts, 3)
    if bb is None:
        return
    bx0, by0, bx1, by1 = bb
    oy = by0 + cv.y0
    ss = 3
    Wb, Hb = (bx1 - bx0) * ss, (by1 - by0) * ss
    rgb = np.zeros((Hb, Wb, 3), np.float32)
    al = np.zeros((Hb, Wb), np.float32)
    cols = [hz(c, hk) for c in BANDS]
    hot = hz(HOT, hk * 0.6)

    def P(px, py):
        return (int(round((px - bx0) * ss * 16)), int(round((py - oy) * ss * 16)))

    def disc(px, py, rad, col, a=1.0, asp=1.0, ang=0.0):
        c = tuple(float(v) for v in col)
        ax = (max(1, int(rad * ss * 16)), max(1, int(rad * asp * ss * 16)))
        cv2.ellipse(rgb, P(px, py), ax, ang, 0, 360, c, -1, cv2.LINE_AA, shift=4)
        cv2.ellipse(al, P(px, py), ax, ang, 0, 360, a, -1, cv2.LINE_AA, shift=4)

    # trunk + a couple of limbs (mostly hidden)
    tk = tuple(float(v) for v in hz(TRUNK, hk))
    for seg in (((x, y + r * 1.25), (x + lean * r, y - r * 0.1)), ((x + lean * r, y + 0.1 * r), (x - 0.35 * r, y - 0.45 * r)),
                ((x + lean * r, y), (x + 0.4 * r, y - 0.5 * r))):
        w_ = max(1, int(r * 0.07 * ss))
        cv2.line(rgb, P(*seg[0]), P(*seg[1]), tk, w_, cv2.LINE_AA, shift=4)
        cv2.line(al, P(*seg[0]), P(*seg[1]), 1.0, w_, cv2.LINE_AA, shift=4)

    # clumps: a crown of 5-8 masses, back (upper) to front (lower)
    n = int(rng.integers(5, 9))
    clumps = []
    for k in range(n):
        a = -math.pi / 2 + (k / max(n - 1, 1) - 0.5) * 2.5 + rng.normal(0, 0.18)
        d = rng.uniform(0.35, 0.72) * r
        rr = r * rng.uniform(0.3, 0.46)
        clumps.append((x + lean * r + math.cos(a) * d * 1.1, y - r * 0.3 + math.sin(a) * d * 0.85, rr))
    clumps.append((x + lean * r, y - r * 0.45, r * 0.52))
    clumps.sort(key=lambda c: c[1] - c[2] * 0.3)
    ytop = min(c[1] - c[2] for c in clumps)
    ybot = max(c[1] + c[2] for c in clumps)
    for ci, (cx, cy, rr) in enumerate(clumps):
        low = np.clip((cy - ytop) / max(ybot - ytop, 1), 0, 1)       # lower clumps sit in more shade
        m = int(110 + 200 * (rr / r))
        th = rng.uniform(0, 2 * math.pi, m)
        rad = np.sqrt(rng.uniform(0, 1, m)) ** 0.8
        dx = np.cos(th) * rad
        dy = np.sin(th) * rad * 0.8
        # clump-local form normal (dome) -> lambert vs. the sun, + dab jitter, posterised into bands
        nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0.05, 1))
        lam = (dx * L[0] + dy * L[1] + nz * L[2]) / np.sqrt(dx * dx + dy * dy + nz * nz)
        val = lam * 0.85 + 0.12 - 0.5 * low + rng.normal(0, 0.09, m)
        order = np.argsort(val)
        for i in order:
            px = cx + dx[i] * rr
            py = cy + dy[i] * rr
            dr = rr * rng.uniform(0.09, 0.17) * (1.0 if rad[i] < 0.85 else 0.75)
            b = int(np.clip(np.floor((val[i] + 0.2) / 0.3), 0, 4))
            asp, ang = rng.uniform(0.55, 0.85), rng.uniform(-40, 40)
            disc(px, py, dr, cols[b], asp=asp, ang=ang)
            if b >= 4:
                # lit lip on the sunward side of the dab (a painted leaf-cluster highlight)
                disc(px + L[0] * dr * 0.45, py + L[1] * dr * 0.45 * asp, dr * 0.5, hot * 0.5 + cols[4] * 0.5,
                     asp=asp, ang=ang)
        # stray leaf tufts breaking the outline (mostly on the top / sides)
        for _ in range(int(6 + 8 * rng.random())):
            a_ = rng.uniform(-math.pi * 0.95, -math.pi * 0.05) if rng.random() < 0.8 else rng.uniform(0, math.pi)
            d_ = rr * rng.uniform(0.95, 1.12)
            px, py = cx + math.cos(a_) * d_, cy + math.sin(a_) * d_ * 0.8
            up = -math.sin(a_)
            b = 3 if (up > 0.3 and low < 0.6) else 2
            disc(px, py, rr * rng.uniform(0.05, 0.09), cols[b])
    # sky holes through the upper canopy (never in the lowest, densest part)
    for _ in range(int(rng.integers(1, 4))):
        hx_ = x + lean * r + rng.uniform(-0.6, 0.6) * r
        hy_ = y - r * rng.uniform(0.35, 0.95)
        hr_ = r * rng.uniform(0.025, 0.05)
        cv2.ellipse(al, P(hx_, hy_), (max(1, int(hr_ * ss * 16)), max(1, int(hr_ * 0.7 * ss * 16))),
                    rng.uniform(0, 180), 0, 360, 0.0, -1, cv2.LINE_AA, shift=4)
    w, h = bx1 - bx0, by1 - by0
    rgb_s = cv2.resize(rgb * al[..., None], (w, h), interpolation=cv2.INTER_AREA)
    al_s = cv2.resize(al, (w, h), interpolation=cv2.INTER_AREA)
    col = rgb_s / np.maximum(al_s, 1e-4)[..., None]
    cv._put(bb, np.clip(al_s, 0, 1), lambda gx, gy: col)
