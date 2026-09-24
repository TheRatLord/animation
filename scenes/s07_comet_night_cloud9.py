"""Round-11 night clouds for s07_comet_night (yn_05 moonlit cut-out clouds).

Each cloud is painted like a cel background: a few big cauliflower lobes (plus two generations of smaller
lobes budding off their upper rims) laid in painter's order, every lobe one flat mid-blue body value with a
crisp pale-cyan crescent on its light-facing side (flat colour, soft only on the crescent's INNER edge - no
gradients, no normal shading, no emboss).  The underside dissolves into a soft, torn, lost edge.  Trailing
fragments (small lobed scraps, thinning out) break the masses up.  Everything is rasterised at 3x and area-
downsampled, so edges are antialiased, never pixel-stepped.
"""
import math

import numpy as np
import cv2

from lib import core as C


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _lobes(rng, w, h, levels=2, flat_top=0.0):
    """Circles (x, y, r, level) in local coords; x in [-w/2, w/2], base at y = 0, up negative."""
    out = []
    n0 = int(np.clip(round(w / (0.55 * h)), 2, 12))
    xs = np.sort(rng.uniform(-0.5, 0.5, n0)) * 0.86 * w
    xs[0], xs[-1] = -0.36 * w, 0.36 * w if n0 > 1 else 0.0
    for x in xs:
        e = max(1 - (2 * x / w) ** 2, 0.05) ** 0.65
        e = e * (1 - flat_top) + flat_top * 0.8
        r = h * rng.uniform(0.22, 0.5) * (0.5 + 0.6 * e) * (1.25 if rng.random() < 0.3 else 1.0)
        top = -h * e * rng.uniform(0.7, 1.0)
        out.append((x, top + r, r, 0))
    # secondary big lobes filling between / in front (lower), gives depth layering
    for i in range(max(1, n0 - 1)):
        x = rng.uniform(-0.42, 0.42) * w
        e = max(1 - (2 * x / w) ** 2, 0.05) ** 0.65
        r = h * rng.uniform(0.22, 0.34)
        out.append((x, -h * e * rng.uniform(0.35, 0.6) + r, r, 0))
    if levels >= 1:
        base = [o for o in out]
        for (x, y, r, _) in base:
            for k in range(rng.integers(2, 5)):
                a = math.radians(rng.uniform(-165, -15))
                rr = r * rng.uniform(0.22, 0.45)
                d = r * rng.uniform(0.7, 0.95)
                out.append((x + d * math.cos(a), y + d * math.sin(a), rr, 1))
    if levels >= 2:
        l1 = [o for o in out if o[3] == 1]
        for (x, y, r, _) in l1:
            for k in range(rng.integers(0, 3)):
                a = math.radians(rng.uniform(-160, -20))
                rr = r * rng.uniform(0.3, 0.5)
                d = r * rng.uniform(0.75, 0.95)
                out.append((x + d * math.cos(a), y + d * math.sin(a), rr, 2))
    return out


def _disc(canvas, cx, cy, rs, val, rng, mode='max'):
    Hs, Ws = canvas.shape
    bx0, bx1 = int(max(cx - rs * 1.15 - 2, 0)), int(min(cx + rs * 1.15 + 3, Ws))
    by0, by1 = int(max(cy - rs * 1.15 - 2, 0)), int(min(cy + rs * 1.15 + 3, Hs))
    if bx1 <= bx0 or by1 <= by0:
        return None
    yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    th = np.arctan2(dy, dx)
    ph = rng.uniform(0, 6.28, 3)
    wob = 0.04 * np.sin(2 * th + ph[0]) + 0.025 * np.sin(3 * th + ph[1]) + 0.012 * np.sin(5 * th + ph[2])
    dist = np.sqrt(dx * dx + dy * dy) / (1 + wob)
    return (slice(by0, by1), slice(bx0, bx1)), dx, dy, dist, wob


def _paint(lobes, w, h, light, pal, rng, ss, s, warm=0.0, bottom=0.18, amt=1.0, fringe=True):
    """Rasterise one cloud in a local canvas (supersampled by ss). Returns (rgb, alpha, x0, y0) at 1x,
    x0 / y0 = local coords of the canvas origin."""
    pad = 0.15 * h + 6
    xs_ = [l[0] - l[2] for l in lobes] + [l[0] + l[2] for l in lobes] + [-w / 2, w / 2]
    ys_ = [l[1] - l[2] for l in lobes]
    x0, x1 = min(xs_) - pad, max(xs_) + pad
    y0, y1 = min(ys_) - pad, bottom * h + pad + 0.25 * h
    x0, y0 = math.floor(x0), math.floor(y0)
    Wc, Hc = int(math.ceil(x1 - x0)) + 1, int(math.ceil(y1 - y0)) + 1
    Ws, Hs = Wc * ss, Hc * ss
    lit_c = np.array(pal['lit'], np.float32)
    body_c = np.array(pal['body'], np.float32)
    sh_c = np.array(pal['sh'], np.float32)
    warm_c = np.array(pal['warm'], np.float32)
    lx, ly = light

    U = np.zeros((Hs, Ws), np.float32)          # silhouette
    T = np.zeros((Hs, Ws), np.float32)          # tone: 0 body .. 1 lit (internal lobe crescents)
    lob0 = [l for l in lobes if l[3] == 0]
    pts = [((min(l[0] - 0.6 * l[2] for l in lob0) - x0) * ss, (0 - y0) * ss)]
    pts += [((x - x0) * ss, (y - y0) * ss) for (x, y, r, _) in sorted(lob0)]
    pts += [((max(l[0] + 0.6 * l[2] for l in lob0) - x0) * ss, (0 - y0) * ss)]
    m = np.zeros((Hs, Ws), np.uint8)
    quads = [np.array(pts, np.int32)]
    cv2.fillPoly(m, quads, 255, lineType=cv2.LINE_AA)
    U = m.astype(np.float32) / 255
    lvl0 = sorted([i for i in range(len(lobes)) if lobes[i][3] == 0], key=lambda i: lobes[i][1])
    rest = sorted([i for i in range(len(lobes)) if lobes[i][3] > 0], key=lambda i: (lobes[i][3], lobes[i][1]))
    for i in rest + lvl0:
        x, y, r, lev = lobes[i]
        cx, cy, rs = (x - x0) * ss, (y - y0) * ss, r * ss
        res = _disc(U, cx, cy, rs, 1.0, rng)
        if res is None:
            continue
        sl, dx, dy, dist, wob = res
        disc = np.clip(rs - dist + 0.5, 0, 1)
        off = 0.22 * rs + 1.5 * ss
        d2 = np.sqrt((dx + lx * off) ** 2 + (dy + ly * off) ** 2) / (1 + wob)
        soft = 0.1 * rs + ss
        cres = _ss(rs - soft, rs + 0.3 * soft, d2)
        # internal lobe edges are only a half-step lighter than the body (painted, not modelled)
        tone = cres * (0.5 if lev == 0 else 0.35)
        T[sl] = T[sl] * (1 - disc) + tone * disc
        U[sl] = np.maximum(U[sl], disc)

    # break the circle geometry: two-scale displacement of silhouette + tone (cauliflower ragged edge)
    seed_ = int(rng.integers(1 << 30))
    f1 = max(Ws / (0.25 * h * ss), 2.0)
    f2 = max(Ws / (0.08 * h * ss), 3.0)
    dxm = (C.fbm(Ws, Hs, f1, 3, seed=seed_) - 0.5) * 0.09 * h * ss + (C.fbm(Ws, Hs, f2, 2, seed=seed_ + 1) - 0.5) * 0.014 * h * ss
    dym = (C.fbm(Ws, Hs, f1, 3, seed=seed_ + 2) - 0.5) * 0.09 * h * ss + (C.fbm(Ws, Hs, f2, 2, seed=seed_ + 3) - 0.5) * 0.014 * h * ss
    gx, gy = np.meshgrid(np.arange(Ws, dtype=np.float32), np.arange(Hs, dtype=np.float32))
    mx, my = gx + dxm, gy + dym
    U = cv2.remap(U, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    T = cv2.remap(T, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    U = np.clip((U - 0.5) * 1.6 + 0.5, 0, 1)     # re-sharpen the resampled edge
    # torn, fractus edges: erode holes / bays near the silhouette (bottom and sides more than the top),
    # and add detached wispy scraps just outside the mass
    Ub = cv2.GaussianBlur(U, (0, 0), 0.07 * h * ss + 1)
    Uw = cv2.GaussianBlur(U, (0, 0), 0.25 * h * ss + 1)
    ne = C.fbm(Ws, Hs, max(Ws / (0.07 * h * ss), 3.0), 4, seed=seed_ + 21)
    yy_ = (np.arange(Hs, dtype=np.float32)[:, None] / ss + y0)
    lowr = _ss(-0.6 * h, 0.1 * h, yy_)                # stronger erosion toward the base
    ero = _ss(-0.06, 0.06, ne - 0.5 + (2.6 - 1.7 * lowr) * (Ub - 0.5) + 0.28)
    U = U * ero
    nw = C.fbm(Ws, Hs, max(Ws / (0.12 * h * ss), 3.0), 4, seed=seed_ + 22)
    wisp = _ss(0.66, 0.72, nw + 0.2 * Uw) * _ss(0.12, 0.35, Uw) * (1 - U) * 0.7 * lowr * _ss(-0.35 * h, -0.1 * h, yy_)
    U = np.clip(U + wisp, 0, 1)

    # outline lit band: where the silhouette, looked at from a step toward the light, is empty
    k = 0.09 * h * ss + 2 * ss
    Mk = np.array([[1, 0, lx * k], [0, 1, ly * k]], np.float32)
    Us = cv2.warpAffine(U, Mk, (Ws, Hs), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
    band = np.clip(U - Us, 0, 1)
    nb = C.fbm(Ws, Hs, max(Ws / (0.06 * h * ss), 3.0), 3, seed=seed_ + 5)
    band = _ss(0.25, 0.55, band + 0.35 * (nb - 0.5))
    T = np.maximum(T, band)
    # painted shadow patches in the lower part of the mass
    yy = (np.arange(Hs, dtype=np.float32)[:, None] / ss + y0)
    q = (yy + 0.55 * h) / (0.6 * h)                       # 0 high in the cloud, 1 at the base
    ns = C.fbm(Ws, Hs, max(Ws / (0.3 * h * ss), 2.0), 4, seed=seed_ + 6)
    shade = _ss(0.42, 0.68, ns * 0.8 + 0.45 * q - 0.02) * (1 - T) * 0.75
    col = body_c * (1 - T[..., None]) + lit_c * T[..., None]
    col = col * (1 - 0.8 * shade[..., None]) + sh_c * (0.8 * shade[..., None])
    if warm > 0:
        wl = _ss(0.3, 1.1, q)[..., None] * warm * (1 - T[..., None])
        col = col * (1 - 0.5 * wl) + warm_c * 0.5 * wl
    # lost, torn underside: alpha fades below the base through broken noise
    nb2 = C.fbm(Ws, Hs, max(Ws / (0.2 * h * ss), 2.0), 4, seed=seed_ + 7)
    qb = (yy - (-0.1 * h)) / (0.35 * h)
    fade = _ss(1.1, 0.0, qb + 0.9 * (nb2 - 0.5))
    A = U * fade
    if warm > 0:
        # low clouds sit in the thicker, warmer air near the horizon: dimmer, lower contrast
        col = col * (1 - 0.22 * warm) + np.array([0.5, 0.42, 0.55], np.float32) * (0.12 * warm)
    # never let anything touch the canvas border (no clipped straight edges)
    bw = max(int(0.06 * h * ss), 3)
    wy_ = np.minimum(np.arange(Hs), np.arange(Hs)[::-1]).astype(np.float32)
    wx_ = np.minimum(np.arange(Ws), np.arange(Ws)[::-1]).astype(np.float32)
    A = A * _ss(0, bw, wy_)[:, None] * _ss(0, bw, wx_)[None, :]
    rgb = cv2.resize(col, (Wc, Hc), interpolation=cv2.INTER_AREA)
    a = cv2.resize(A, (Wc, Hc), interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    return rgb, np.clip(a * amt, 0, 1), x0, y0


def _fragments(rng, cx, cy, w, h, ang, n, size0):
    """Trailing scraps: small lobed clusters strung along a direction, shrinking and thinning."""
    out = []
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    q = 0.0
    for k in range(n):
        q += rng.uniform(0.6, 1.3) / n
        taper = 1 - 0.75 * q
        fw = size0 * rng.uniform(0.5, 1.2) * taper
        off = rng.normal(0, 0.25) * h
        x = cx + q * w * ca - off * sa
        y = cy + q * w * sa + off * ca
        out.append((x, y, fw, fw * rng.uniform(0.35, 0.55), rng.uniform(0.75, 1.0) * (0.5 + 0.5 * taper)))
    return out


def night_clouds9(Wp, Hp, specs, s, light_xy, seed=17, pal=None, ss=3):
    """specs: (cx, base_y, w, h, amt, warm, levels, trail) with trail = None or (angle_deg, length/w, n).
    Returns straight RGBA (Hp, Wp, 4)."""
    if pal is None:
        pal = dict(lit=(0.79, 0.9, 0.98), body=(0.48, 0.62, 0.8), sh=(0.34, 0.46, 0.67), warm=(0.95, 0.55, 0.45))
    rgb = np.zeros((Hp, Wp, 3), np.float32)
    A = np.zeros((Hp, Wp), np.float32)
    items = []
    for i, sp in enumerate(specs):
        cx, by, w, h, amt, warm, lev, trail = sp
        rng = np.random.default_rng(seed * 1000 + i * 13)
        items.append((cx, by, w, h, amt, warm, lev, rng.integers(1 << 30)))
        if trail is not None:
            ang, ln, n = trail
            for (fx, fy, fw, fh, fa) in _fragments(rng, cx + math.cos(math.radians(ang)) * 0.45 * w,
                                                   by - 0.25 * h + math.sin(math.radians(ang)) * 0.45 * w,
                                                   ln * w, h, ang, n, 0.28 * w):
                items.append((fx, fy, fw, fh, amt * fa, warm, 1, rng.integers(1 << 30)))
    # far/high first, lower (nearer) clouds last
    for (cx, by, w, h, amt, warm, lev, sd) in items:
        rng = np.random.default_rng(sd)
        lobes = _lobes(rng, w, h, levels=lev)
        # light from the comet, mostly from above
        vx, vy = light_xy[0] - cx, light_xy[1] - (by - h)
        n_ = math.hypot(vx, vy) + 1e-6
        lx, ly = 0.55 * vx / n_, 0.55 * vy / n_ - 0.8
        n_ = math.hypot(lx, ly)
        lx, ly = lx / n_, ly / n_
        crgb, ca, x0, y0 = _paint(lobes, w, h, (lx, ly), pal, rng, ss, s, warm=warm, amt=amt)
        gx0, gy0 = int(math.floor(cx + x0)), int(math.floor(by + y0))
        hh, ww = ca.shape
        sx0, sy0 = max(gx0, 0), max(gy0, 0)
        sx1, sy1 = min(gx0 + ww, Wp), min(gy0 + hh, Hp)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        cr = crgb[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
        a = ca[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
        sl = (slice(sy0, sy1), slice(sx0, sx1))
        # accumulate premultiplied
        rgb[sl] = rgb[sl] * (1 - a[..., None]) + cr * a[..., None]
        A[sl] = a + A[sl] * (1 - a)
    rgb = rgb / np.maximum(A, 1e-5)[..., None] * (A > 1e-5)[..., None]
    return np.dstack([rgb, A]).astype(np.float32)
