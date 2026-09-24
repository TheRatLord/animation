"""Round-3 art helpers for s02_railway_crossing: crisp carved cumulonimbus base lobes with a lavender shadow
core and silver rims + torn scud at the base; irregular sugi cedars with sun-rim highlights; lens-flare
(anamorphic streak + coloured ghost discs); light wrap / halation behind silhouettes."""
import math

import numpy as np
import cv2

import s02_railway_crossing_paint as P

hx = P.hexc
sstep = P.sstep


# ------------------------------------------------------------------------------------------ clouds

def carve_cloud_base(plate, box, sun_xy, seed=0, sc=1.0, n=80, rmin=0.016, rmax=0.042, wisps=True,
                     wisp_band=None):
    """Paint crisp cauliflower lobes over the lower body of a cumulonimbus plate (RGBA straight, in place).

    box = (x0, y0, x1, y1) plate px region of the lower tower. Lobes are placed where the cloud already is,
    painted top-to-bottom so lower bulges overlap upper ones; each lobe is a cel-lit sphere: warm-white lit
    face toward the sun, pale lavender half-tone, a cool lavender shadow side deepening toward the base
    (shadow core), crisp edges and a silver rim where the lobe faces the sun."""
    H, W = plate.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    rng = np.random.default_rng(seed)
    a = plate[..., 3]
    cand = []
    tries = 0
    while len(cand) < n and tries < n * 60:
        tries += 1
        cx = rng.uniform(x0, x1)
        cy = rng.uniform(y0, y1)
        if a[int(cy), int(cx)] < 0.85:
            continue
        fy = (cy - y0) / max(y1 - y0, 1)
        r = rng.uniform(rmin, rmax) * W * (1.15 - 0.4 * fy)
        cand.append((cx, cy, r, fy))
    cand.sort(key=lambda c: c[1] + c[2] * 0.6)
    Ls = np.array([0.7, -0.5, 0.42])
    Ls = Ls / np.linalg.norm(Ls)
    c_lit = hx('#fcfaf4')
    c_hi = hx('#eceaf6')
    c_half = hx('#d0cfee')
    c_shd = hx('#aeb0dc')
    c_core = hx('#959dd0')
    rgb = plate[..., :3]
    for (cx, cy, r, fy) in cand:
        X0, X1 = int(max(cx - r - 2, 0)), int(min(cx + r + 3, W))
        Y0, Y1 = int(max(cy - r - 2, 0)), int(min(cy + r + 3, H))
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        # slightly lumpy outline (cauliflower sub-bumps)
        ang = np.arctan2(yy - cy, xx - cx)
        ph = rng.uniform(0, 6.28)
        rr = r * (1 + 0.06 * np.sin(ang * 5 + ph) + 0.035 * np.sin(ang * 9 + ph * 1.7))
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        m = np.clip(rr - d + 0.5, 0, 1)
        if m.max() <= 0:
            continue
        nx = (xx - cx) / rr
        ny = (yy - cy) / rr
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        ndl = nx * Ls[0] + ny * Ls[1] + nz * Ls[2]
        # lobes away from the sun (left) and low in the tower sit in the body's shadow
        fxp = (cx - x0) / max(x1 - x0, 1)
        ndl = ndl - 0.14 * (1 - fxp) - 0.14 * fy
        col = c_core + (c_shd - c_core) * (1 - fy * 0.8)
        col = np.broadcast_to(col, ndl.shape + (3,)).astype(np.float32)
        col = col + (c_half - col) * sstep(-0.12, -0.04, ndl)[..., None]
        col = col + (c_hi - col) * sstep(0.12, 0.2, ndl)[..., None]
        col = col + (c_lit - col) * sstep(0.4, 0.48, ndl)[..., None]
        # silver rim on the sun-facing limb
        rdir = (nx * Ls[0] + ny * Ls[1]) / (np.sqrt(nx * nx + ny * ny) + 1e-4)
        rim = sstep(0.86, 0.97, d / rr) * sstep(0.2, 0.6, rdir)
        col = col + (np.array([1.12, 1.1, 1.06], np.float32) - col) * (rim * 0.9)[..., None]
        # dark crease just inside the lower-left limb (separates from the lobe behind)
        crease = sstep(0.9, 1.0, d / rr) * sstep(0.0, -0.5, rdir)
        col = col * (1 - 0.1 * crease)[..., None]
        sub = rgb[Y0:Y1, X0:X1]
        sa = a[Y0:Y1, X0:X1]
        # follow the painted body's large-scale shading (keeps the tower's lavender gradient)
        inm = (m > 0.5) & (sa > 0.5)
        if inm.any():
            tint = sub[inm].mean(0) / 0.97
            col = col * np.clip(tint, 0.75, 1.04)[None, None, :] ** 0.45
        # lobes that stick out of the silhouette keep the local cloud colour behind them on their far side
        k = m[..., None]
        rgb[Y0:Y1, X0:X1] = sub * (1 - k) + col * k
        a[Y0:Y1, X0:X1] = np.maximum(sa, m * np.clip(sa.max() * 1.2, 0, 1))
    if wisps and wisp_band is not None:
        wy0, wy1, wx0, wx1 = wisp_band
        for i in range(14):
            cx = rng.uniform(wx0, wx1)
            cy = rng.uniform(wy0, wy1)
            rx = rng.uniform(0.01, 0.035) * W
            ry = rx * rng.uniform(0.1, 0.22)
            X0, X1 = int(max(cx - rx * 1.4, 0)), int(min(cx + rx * 1.4, W))
            Y0, Y1 = int(max(cy - ry * 2.5, 0)), int(min(cy + ry * 2.5, H))
            if X1 <= X0 or Y1 <= Y0:
                continue
            yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
            u = (xx - cx) / rx
            v = (yy - cy) / ry
            # torn: ragged top edge from a sum of sines, flat-ish bottom, feathered ends
            top = -1.0 + 0.35 * np.sin(u * 7 + i) + 0.2 * np.sin(u * 17 + i * 3)
            m = sstep(top - 0.05, top + 0.25, v) * sstep(1.0, 0.6, v) * sstep(1.0, 0.55, np.abs(u))
            m = m * rng.uniform(0.35, 0.65)
            col = hx('#b8badf') + (hx('#e4e4f4') - hx('#b8badf')) * sstep(0.4, -0.6, v)[..., None]
            sub = rgb[Y0:Y1, X0:X1]
            sa = a[Y0:Y1, X0:X1]
            rgb[Y0:Y1, X0:X1] = sub * (1 - m[..., None] * (1 - sa[..., None] * 0.0)) + col * m[..., None]
            a[Y0:Y1, X0:X1] = sa + (1 - sa) * m
    plate[..., :3] = rgb
    plate[..., 3] = a
    return plate


# ------------------------------------------------------------------------------------------ cedars

def draw_cedar3(cv, rng, cx, by, w, h, pal, fog=0.0, hazec=None, rim=None):
    """Sugi cedar with an irregular, slightly leaning spire of drooping tiers, a lit right flank, bright
    drooping tips and a crisp warm sun-rim along the right silhouette of each tier."""
    hazec = hx('#a9cde2') if hazec is None else hazec
    rim = hx('#e8f6a8') if rim is None else rim

    def fz(c):
        return np.asarray(c, np.float32) * (1 - fog) + hazec * fog
    dark, mid, lit, tip, rc = fz(pal[0]), fz(pal[1]), fz(pal[2]), fz(pal[3]), fz(rim)
    lean = rng.normal(0, 0.03) * h
    cv.poly([(cx - w * 0.05, by), (cx + w * 0.05, by), (cx + w * 0.02 + lean * 0.4, by - h * 0.5),
             (cx - w * 0.02 + lean * 0.4, by - h * 0.5)], fz(hx('#3a2e2a')))
    n = int(np.clip(h / max(w * 0.22, 1.0), 6, 16) + rng.integers(-1, 2))
    y0 = by - h * rng.uniform(0.08, 0.16)
    for i in range(n):
        f0 = i / n
        f1 = (i + 1.7) / n
        ya = y0 - f0 * (h * 0.88)
        yb = y0 - min(f1, 1.0) * (h * 0.88)
        xc = cx + lean * f0
        half = w * 0.5 * (1 - f0) ** rng.uniform(0.7, 1.0) + w * 0.04
        half *= rng.uniform(0.8, 1.15)
        jl, jr = rng.uniform(0.85, 1.2), rng.uniform(0.85, 1.2)
        L = (xc - half * jl, ya + h / n * rng.uniform(0.15, 0.35))
        R = (xc + half * jr, ya + h / n * rng.uniform(0.2, 0.4))
        T = (xc + lean * (f1 - f0) + rng.uniform(-0.05, 0.05) * w, yb)
        body = [L, (xc - half * 0.55, ya - h / n * 0.1), T, (xc + half * 0.55, ya - h / n * 0.1), R,
                (xc + half * 0.4, ya + h / n * 0.05), (xc - half * 0.4, ya + h / n * 0.05)]
        cv.poly(body, dark)
        cv.poly([(xc + half * 0.05, ya - h / n * 0.05), T, (xc + half * 0.55, ya - h / n * 0.1), R,
                 (xc + half * 0.3, ya + h / n * 0.02)], mid)
        cv.poly([(xc + half * 0.35, ya - h / n * 0.12), (xc + half * 0.5, ya - h / n * 0.2), R,
                 (xc + half * 0.6, ya + h / n * 0.05)], lit)
        # sun-rim: thin bright sliver along the upper-right silhouette edge of the tier
        e = max(half * 0.08, 0.35)
        cv.poly([T, (xc + half * 0.55, ya - h / n * 0.1), R, (R[0] - e, R[1] - e * 0.2),
                 (xc + half * 0.55 - e, ya - h / n * 0.1 + e * 0.6), (T[0] - e * 0.3, T[1] + e * 1.2)], rc)
        for k in range(2):
            px = xc + half * rng.uniform(0.45, 0.95)
            py = ya - h / n * rng.uniform(-0.05, 0.25)
            s = half * 0.16
            cv.poly([(px - s, py), (px + s * 0.4, py - s * 0.6), (px + s, py + s * 0.5)], tip)
    cv.poly([(cx + lean - w * 0.03, y0 - h * 0.86), (cx + lean + w * 0.03, y0 - h * 0.86), (cx + lean, by - h * 1.02)],
            dark)
    cv.line([(cx + lean + w * 0.015, y0 - h * 0.86), (cx + lean, by - h * 1.02)], max(w * 0.02, 0.5), rc)


# ------------------------------------------------------------------------------------------ lens / light

def anamorphic_flare(W, H, lx, ly, seed=0):
    """Anime photographic flare for a sun at (lx, ly): a long thin horizontal anamorphic streak with a blue
    fringe, and 4 soft coloured ghost discs (with brighter rims) + a small hex along the sun->centre axis."""
    s = 2
    w, h = W // s, H // s
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xs = xs * s + s * 0.5
    ys = ys * s + s * 0.5
    out = np.zeros((h, w, 3), np.float32)
    dy = (ys - ly) / H
    dx = (xs - lx) / W
    core = np.exp(-(dy / 0.0022) ** 2) * np.exp(-np.abs(dx) / 0.32)
    wide = np.exp(-(dy / 0.009) ** 2) * np.exp(-np.abs(dx) / 0.5)
    out += core[..., None] * np.array([1.0, 0.97, 0.92], np.float32) * 0.3
    out += wide[..., None] * np.array([0.35, 0.6, 1.0], np.float32) * 0.1
    cx, cy = W * 0.5, H * 0.5
    vx, vy = cx - lx, cy - ly
    ghosts = [(0.55, 0.022, (1.0, 0.72, 0.42), 0.22), (0.95, 0.055, (0.45, 1.0, 0.7), 0.11),
              (1.28, 0.034, (0.55, 0.7, 1.0), 0.18), (1.62, 0.09, (0.9, 0.55, 1.0), 0.09),
              (1.95, 0.016, (1.0, 0.9, 0.6), 0.2)]
    for (k, r, col, amp) in ghosts:
        gx, gy = lx + vx * k, ly + vy * k
        d = np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2) / (r * W)
        disc = sstep(1.0, 0.86, d) * (0.45 + 0.55 * sstep(0.2, 0.95, d))
        ring = np.exp(-((d - 0.97) / 0.05) ** 2) * 0.6
        out += (disc + ring)[..., None] * np.array(col, np.float32) * amp
    # small hexagonal aperture ghost
    gx, gy = lx + vx * 0.78, ly + vy * 0.78
    ang = np.arctan2(ys - gy, xs - gx)
    dd = np.sqrt((xs - gx) ** 2 + (ys - gy) ** 2) * (np.cos(np.pi / 6) / np.cos(((ang + np.pi / 6) % (np.pi / 3)) - np.pi / 6))
    out += sstep(1.0, 0.9, dd / (0.03 * W))[..., None] * np.array([0.7, 0.85, 1.0], np.float32) * 0.07
    return cv2.resize(out, (W, H), interpolation=cv2.INTER_LINEAR)


def light_wrap(img, bg, alpha, lx, ly, W, H, sigma, strength=0.6):
    """Halation: bright background light bleeding over the edges of foreground silhouettes (strongest toward
    the sun). img: composited frame, bg: frame before the silhouettes were drawn, alpha: silhouette coverage."""
    q = 4
    small = cv2.resize(bg, (W // q, H // q), interpolation=cv2.INTER_AREA)
    lum = small.max(-1, keepdims=True)
    br = small * sstep(0.7, 1.0, lum)
    bl = cv2.GaussianBlur(br, (0, 0), sigma / q)
    bl = cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR)
    return bl
