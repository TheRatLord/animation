"""Round-10 painting helpers for s07_comet_night (Your Name comet night).

- night_clouds8: painted cauliflower night clouds (yn_05). Each cloud is a cluster of lobes (big base lobes
  resting on a flat, lost base, smaller child lobes piled on their upper sides, 2 levels). Lobes are painted
  top -> bottom, each with its own lit cap, so every lower lobe's pale cap overlaps the shadowed underside of
  the lobes above it: flat painted value planes (lit / mid / shadow) with crisp breaks, a crisp lit upper
  silhouette, and a soft underside lost into the sky. No outline or rim halo anywhere. Plus torn cirrus
  wisps (streaky, tapered) and small scraps.
"""
import math

import numpy as np
import cv2

from lib import core as C


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _lobes(rng, cx, base, w, h, levels=2):
    """List of (x, y, r, level). Big lobes rest on the flat base; children pile on their upper sides."""
    out = []
    n = max(int(round(w / (0.26 * h))), 2)
    xs = ((np.arange(n) + rng.uniform(0.15, 0.85, n)) / n - 0.5) * w * 0.86 + cx
    for x in xs:
        q = (x - cx) / (0.5 * w)
        dome = math.sqrt(max(1 - q * q, 0.05))                     # taller in the middle
        r = h * rng.uniform(0.18, 0.32) * (0.5 + 0.5 * dome)
        y = base - r * rng.uniform(0.55, 0.9) - h * 0.25 * dome * rng.uniform(0.3, 1.0)
        out.append((x, y, r, 0))
    frontier = list(out)
    for lev in range(1, levels + 1):
        nxt = []
        for (x, y, r, _) in frontier:
            for _k in range(int(rng.integers(2, 6 - lev))):
                a = rng.uniform(-2.9, -0.25)                          # upper half (y down)
                rr = r * rng.uniform(0.3, 0.52)
                dd = r * rng.uniform(0.7, 0.98)
                c = (x + math.cos(a) * dd, y + math.sin(a) * dd, rr, lev)
                nxt.append(c)
        out += nxt
        frontier = nxt
    return out


def _paint_cluster(rgb, A, x0, y0, lobes, base, h, light, pal, seed, s, warm=0.0, lost=0.75, detail=1.0,
                   amt=1.0, asp=1.3, flat=0.0):
    """Paint one lobe cluster into the crop (rgb, A are the crop arrays; crop origin x0, y0)."""
    hh, ww = A.shape
    ys, xs = np.mgrid[0:hh, 0:ww].astype(np.float32)
    xs += x0
    ys += y0
    # domain warp: breaks the circles into frilly, torn, cauliflower edges (two scales)
    sc1 = max(ww / (0.25 * h), 2)
    wx = (C.fbm(ww, hh, sc1, 4, seed=seed + 1) - 0.5) * 0.14 * h
    wy = (C.fbm(ww, hh, sc1, 4, seed=seed + 2) - 0.5) * 0.12 * h
    sc2 = max(ww / (0.08 * h), 4)
    fx = (C.fbm(ww, hh, sc2, 3, seed=seed + 3) - 0.5) * 0.035 * h * detail
    fy = (C.fbm(ww, hh, sc2, 3, seed=seed + 4) - 0.5) * 0.035 * h * detail
    X = xs + wx + fx
    Y = ys + wy + fy
    tex = C.fbm(ww, hh, max(ww / (0.12 * h), 3), 4, seed=seed + 5)
    lx, ly = light
    ln = math.hypot(lx, ly)
    lx, ly = lx / ln, ly / ln
    c_lit, c_mid, c_sh, c_sky = (np.asarray(pal[k], np.float32) for k in ('lit', 'mid', 'sh', 'sky'))
    c_sh = c_sh + (c_mid - c_sh) * flat
    warm_c = np.asarray(pal.get('warm', (1.0, 0.55, 0.4)), np.float32)
    # paint the lobes top -> bottom so lower (nearer) lobes overlap the shadowed bellies above them
    order = sorted(lobes, key=lambda l: l[1] + 0.3 * l[2])
    for (lx0, ly0, r, lev) in order:
        bx0 = int(max(lx0 - 1.5 * r - x0, 0)); bx1 = int(min(lx0 + 1.5 * r - x0 + 1, ww))
        by0 = int(max(ly0 - 1.5 * r - y0, 0)); by1 = int(min(ly0 + 1.5 * r - y0 + 1, hh))
        if bx1 <= bx0 or by1 <= by0:
            continue
        sl = (slice(by0, by1), slice(bx0, bx1))
        dx = (X[sl] - lx0) / (r * asp)
        dy = (Y[sl] - ly0) / r
        d = np.sqrt(dx * dx + dy * dy)
        aa = 1.0 / max(r, 1.0) * 1.2
        m = _ss(1.0 + aa, 1.0 - aa, d)
        if not m.any():
            continue
        # the lit cap: facing the light, broken by texture -> flat planes with painterly boundaries
        nd = (dx * lx + dy * ly) / np.maximum(d, 1e-3)
        f = nd * np.clip(d, 0, 1) ** 0.6 + 0.35 * (tex[sl] - 0.5)
        lit = _ss(0.02, 0.24, f)
        midp = _ss(-0.55, -0.2, f)
        col = c_sh + (c_mid - c_sh) * midp[..., None]
        col = col + (c_lit - col) * lit[..., None]
        # small child lobes are further up the cluster -> slightly brighter overall
        col = col * (1 + 0.05 * lev)
        # underside: toward the base the shadow falls to the sky colour and picks up the warm horizon
        u = _ss(base - 0.55 * h, base + 0.05 * h, ys[sl] + 0.25 * h * (tex[sl] - 0.5))
        col = col + (c_sky - col) * (u * lost * (1 - lit))[..., None]
        if warm > 0:
            col = col + warm_c * (warm * u * (1 - 0.7 * lit))[..., None]
        if warm > 0:                                          # low clouds sit in the twilight: warm cast
            col = col * (1 - 0.3 * warm) + warm_c * (0.3 * warm) * (0.6 + 0.4 * (1 - lit))[..., None]
        col = col * (0.965 + 0.07 * tex[sl])[..., None]
        a = m
        rgb[sl] = rgb[sl] * (1 - a[..., None]) + col * a[..., None]
        A[sl] = A[sl] + a * (1 - A[sl])
    # lost base: alpha fades out below the base (soft, broken)
    yb = _ss(base + 0.1 * h, base - 0.3 * h, ys + 0.3 * h * (tex - 0.5))
    k = ((1 - lost) + lost * yb) * amt
    A *= k
    rgb *= k[..., None]
    return rgb, A


def torn_specs(rng, cx, cy, w, h, ang, amt=0.9, warm=0.0, n=None):
    """A torn cloud streamer (yn_05 top-left) as a chain of small cauliflower clumps strung along a
    wobbling path: bigger, denser clumps at the head, shrinking, thinning scraps with gaps toward the
    tail. Returns cluster specs for night_clouds8."""
    a_ = math.radians(ang)
    ca, sa = math.cos(a_), math.sin(a_)
    n = n or max(int(w / (0.9 * h)), 3)
    out = []
    ph = rng.uniform(0, 6.28)
    q = -0.5
    while q < 0.5:
        taper = 1.0 - 0.75 * (q + 0.5) ** 1.3                          # head at q=-0.5
        ww_ = h * rng.uniform(1.4, 2.8) * taper
        hh_ = h * rng.uniform(0.4, 0.75) * taper
        off = 0.45 * h * math.sin(q * 5.0 + ph) + rng.normal(0, 0.18 * h)
        x = cx + q * w * ca - off * sa
        y = cy + q * w * sa + off * ca
        out.append((x, y, ww_, hh_, amt * rng.uniform(0.7, 1.0) * (0.55 + 0.45 * taper), warm,
                    1, 1.0))
        q += (ww_ * rng.uniform(0.12, 0.3)) / w
    return out


def night_clouds8(Wp, Hp, specs, wisps, s, light_xy, seed=17, pal=None):
    """specs: list of (cx, cy_base, width, height, amt, warm, levels); wisps: list of torn scraps (cx, cy, w, h, angle_deg, amt).
    Returns straight RGBA (Hp, Wp, 4)."""
    if pal is None:
        pal = dict(lit=(0.64, 0.78, 0.93), mid=(0.43, 0.55, 0.78), sh=(0.3, 0.38, 0.62), sky=(0.1, 0.14, 0.36),
                   warm=(0.9, 0.42, 0.3))
    rgb = np.zeros((Hp, Wp, 3), np.float32)
    A = np.zeros((Hp, Wp), np.float32)
    rng = np.random.default_rng(seed)
    specs = list(specs)
    for i, (cx, cy, w, h, ang, am) in enumerate(wisps):
        specs = torn_specs(np.random.default_rng(seed + 700 + i), cx, cy, w, h, ang, am) + specs
    for i, sp_ in enumerate(specs):
        cx, base, w, h, amt, warm, lev = sp_[:7]
        flat = sp_[7] if len(sp_) > 7 else 0.0
        lobes = _lobes(np.random.default_rng(seed * 1000 + i * 7 + int(cx) % 97), cx, base, w, h, levels=lev)
        pad = 0.4 * h
        xs_ = [l[0] - l[2] for l in lobes] + [l[0] + l[2] for l in lobes]
        ys_ = [l[1] - l[2] for l in lobes]
        cx0, cx1 = int(max(min(xs_) - pad, 0)), int(min(max(xs_) + pad, Wp))
        cy0, cy1 = int(max(min(ys_) - pad, 0)), int(min(base + 0.5 * h, Hp))
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        lx = float(np.clip((light_xy[0] - cx) / (0.5 * Wp), -0.7, 0.7))
        crgb = np.zeros((cy1 - cy0, cx1 - cx0, 3), np.float32)
        ca = np.zeros((cy1 - cy0, cx1 - cx0), np.float32)
        crgb, ca = _paint_cluster(crgb, ca, cx0, cy0, lobes, base, h, (lx, -1.0), pal, seed + 37 * i, s,
                                  warm=warm, amt=amt, detail=1.0, flat=flat)
        sl = (slice(cy0, cy1), slice(cx0, cx1))
        rgb[sl] = rgb[sl] * (1 - ca[..., None]) + crgb
        A[sl] = ca + A[sl] * (1 - ca)
    rgb = rgb / np.maximum(A, 1e-5)[..., None] * (A > 1e-5)[..., None]
    # dry-brush breakup: side / under edges dissolve into speckled, feathered strokes; lit tops stay crisp
    rr = np.nonzero(A.max(1) > 0.002)[0]
    cc = np.nonzero(A.max(0) > 0.002)[0]
    if len(rr):
        y0, y1 = max(rr.min() - 4, 0), min(rr.max() + 5, Hp)
        x0, x1 = max(cc.min() - 4, 0), min(cc.max() + 5, Wp)
        a = A[y0:y1, x0:x1]
        hh, ww = a.shape
        ab = cv2.GaussianBlur(a, (0, 0), 3.0 * s + 0.5)
        d = max(int(3 * s), 1)
        gy = np.zeros_like(ab)
        gy[d:-d] = ab[2 * d:] - ab[:-2 * d]
        top = _ss(0.02, 0.12, gy)
        hf = C.fbm(ww, hh, ww / (3.5 * s + 0.5), 3, seed=seed + 900)
        hf2 = C.fbm(ww, hh, ww / (14.0 * s + 1.0), 3, seed=seed + 901)
        n = 0.45 * hf + 0.55 * hf2
        keep = _ss(-0.05, 0.05, n - 0.5 + 2.6 * (ab - 0.66))
        k = 1 - (1 - keep) * (1 - 0.85 * top)
        A[y0:y1, x0:x1] = a * k
    return np.dstack([rgb, A]).astype(np.float32)


def coloured_stars(Wp, Hp, n, s, seed=0, dens=None):
    """yn_05 point stars: small soft round dots in saturated magenta / cyan / amber / white, no spikes."""
    rng = np.random.default_rng(seed)
    out = np.zeros((Hp, Wp, 3), np.float32)
    pal = np.array([[1.0, 0.35, 0.85], [0.35, 0.85, 1.0], [1.0, 0.7, 0.3], [0.75, 0.55, 1.0], [0.9, 0.95, 1.0]],
                   np.float32)
    p = np.array([0.3, 0.26, 0.2, 0.12, 0.12])
    x = rng.random(n) * Wp
    y = rng.random(n) ** 1.15 * Hp
    ci = rng.choice(len(pal), n, p=p)
    m = rng.uniform(0.35, 1.0, n) ** 1.5
    r = rng.uniform(0.9, 1.9, n) * s + 0.3
    for i in range(n):
        C.splat(out, x[i:i + 1], y[i:i + 1], r[i], pal[ci[i]], 0.9 * m[i])
        C.splat(out, x[i:i + 1], y[i:i + 1], r[i] * 2.6, pal[ci[i]], 0.06 * m[i])
    return out
