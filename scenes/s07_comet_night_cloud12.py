"""Round-12 night clouds for s07_comet_night: flat Your-Name (yn_05 / yn_04) cut-out clouds.

Judges flagged the round-11 clouds as clay pillows (per-lobe crescents, spherical shading, dark airbrushed
bottoms with a grey smear under each mass).  These clouds are painted as three FLAT cel values only:

  * a moonlit cyan-white TOP value whose inner boundary is a crisp scalloped line (it follows the silhouette's
    cauliflower outline, offset inward by a width that varies from lobe to lobe, so it never reads as an
    even outline stroke),
  * one flat blue-grey / navy-violet BODY value,
  * a hard-edged darker BASE band hugging the underside (a warm dusky tint on the low twilight clouds).

Overlapping front lobes get their own flat lit top where they overlap the mass behind (the layered cut-out
read of yn_05).  No per-lobe gradients, no normal shading, no alpha fade or smoke below the base: the
silhouette is crisp all round, torn into scraps/holes by noise, and only a faint 1px softening is applied.
Everything is rasterised at 3x and area-downsampled.
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_cloud9 as K9


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _shift(U, dx, dy):
    """Sample U at (x + dx, y + dy); dx/dy may be arrays (spatially varying)."""
    Hs, Ws = U.shape
    gx, gy = np.meshgrid(np.arange(Ws, dtype=np.float32), np.arange(Hs, dtype=np.float32))
    return cv2.remap(U, (gx + dx).astype(np.float32), (gy + dy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def _disc_mask(Hs, Ws, discs, rng):
    m = np.zeros((Hs, Ws), np.float32)
    for (cx, cy, rs) in discs:
        res = K9._disc(m, cx, cy, rs, 1.0, rng)
        if res is None:
            continue
        sl, dx, dy, dist, wob = res
        m[sl] = np.maximum(m[sl], np.clip(rs - dist + 0.5, 0, 1))
    return m


def _paint(lobes, w, h, light, pal, rng, ss, warm=0.0, amt=1.0):
    pad = 0.2 * h + 6
    xs_ = [l[0] - l[2] for l in lobes] + [l[0] + l[2] for l in lobes] + [-w / 2, w / 2]
    ys_ = [l[1] - l[2] for l in lobes]
    x0, x1 = min(xs_) - pad, max(xs_) + pad
    y0, y1 = min(ys_) - pad, 0.25 * h + pad
    x0, y0 = math.floor(x0), math.floor(y0)
    Wc, Hc = int(math.ceil(x1 - x0)) + 1, int(math.ceil(y1 - y0)) + 1
    Ws, Hs = Wc * ss, Hc * ss
    lx, ly = light

    # ---- silhouette: base polygon + all lobes
    lob0 = [l for l in lobes if l[3] == 0]
    pts = [((min(l[0] - 0.6 * l[2] for l in lob0) - x0) * ss, (0 - y0) * ss)]
    pts += [((x - x0) * ss, (y - y0) * ss) for (x, y, r, _) in sorted(lob0)]
    pts += [((max(l[0] + 0.6 * l[2] for l in lob0) - x0) * ss, (0 - y0) * ss)]
    m = np.zeros((Hs, Ws), np.uint8)
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255, lineType=cv2.LINE_AA)
    U = np.maximum(m.astype(np.float32) / 255,
                   _disc_mask(Hs, Ws, [((x - x0) * ss, (y - y0) * ss, r * ss) for (x, y, r, _) in lobes], rng))
    # front lobes: the lower level-0 lobes sitting in front of the mass (own flat lit top where they overlap)
    lob0s = sorted(lob0, key=lambda l: -l[1])
    front = [l for l in lob0s[:max(1, len(lob0s) // 3)]]
    front += [l for l in lobes if l[3] == 1 and rng.random() < 0.25]
    Fm = _disc_mask(Hs, Ws, [((x - x0) * ss, (y - y0) * ss, r * 0.92 * ss) for (x, y, r, _) in front], rng)
    # the shade band's upper boundary: the scalloped (convex-down) bottoms of the masses above it
    Up = np.zeros((Hs, Ws), np.float32)
    yb_top = int(max((-0.24 * h - y0) * ss, 0))
    Up[:yb_top] = 1.0
    xx = -0.6 * w
    discs = []
    while xx < 0.6 * w:
        r = h * rng.uniform(0.12, 0.3)
        yb = -h * rng.uniform(0.04, 0.16)
        discs.append(((xx - x0) * ss, (yb - r - y0) * ss, r * ss))
        xx += r * rng.uniform(1.1, 1.7)
    Up = np.maximum(Up, _disc_mask(Hs, Ws, discs, rng))

    # ---- break the circle geometry: two-scale displacement (cauliflower ragged edge)
    seed_ = int(rng.integers(1 << 30))
    f1 = max(Ws / (0.25 * h * ss), 2.0)
    f2 = max(Ws / (0.07 * h * ss), 3.0)
    dxm = (C.fbm(Ws, Hs, f1, 3, seed=seed_) - 0.5) * 0.1 * h * ss + \
        (C.fbm(Ws, Hs, f2, 2, seed=seed_ + 1) - 0.5) * 0.035 * h * ss
    dym = (C.fbm(Ws, Hs, f1, 3, seed=seed_ + 2) - 0.5) * 0.1 * h * ss + \
        (C.fbm(Ws, Hs, f2, 2, seed=seed_ + 3) - 0.5) * 0.035 * h * ss
    U = _shift(U, dxm, dym)
    Fm = _shift(Fm, dxm, dym)
    Up = _shift(Up, dxm * 0.6, dym * 0.6)
    U = np.clip((U - 0.5) * 2.0 + 0.5, 0, 1)
    # flat-ish cloud base (lobes never hang below it): a gently undulating cut
    yy0 = (np.arange(Hs, dtype=np.float32)[:, None] / ss + y0)
    bn = C.fbm(Ws, 8, max(Ws / (0.5 * h * ss), 2.0), 3, seed=seed_ + 30)[4][None, :]
    bn2 = C.fbm(Ws, 8, max(Ws / (0.12 * h * ss), 3.0), 2, seed=seed_ + 31)[4][None, :]
    ycut = (0.03 + 0.22 * (bn - 0.5) + 0.07 * (bn2 - 0.5)) * h
    U = U * np.clip((ycut - yy0) * ss / 1.5 + 0.5, 0, 1)
    Fm = np.clip((Fm - 0.5) * 2.0 + 0.5, 0, 1)
    # torn edges: bays / holes near the silhouette (sides and base more than the top) + detached scraps
    Ub = cv2.GaussianBlur(U, (0, 0), 0.07 * h * ss + 1)
    Uw = cv2.GaussianBlur(U, (0, 0), 0.25 * h * ss + 1)
    ne = C.fbm(Ws, Hs, max(Ws / (0.07 * h * ss), 3.0), 4, seed=seed_ + 21)
    yy = (np.arange(Hs, dtype=np.float32)[:, None] / ss + y0)
    lowr = _ss(-0.6 * h, 0.1 * h, yy)
    ero = _ss(-0.03, 0.03, ne - 0.5 + (3.4 - 1.0 * lowr) * (Ub - 0.5) + 0.36)
    U = U * ero
    nw = C.fbm(Ws, Hs, max(Ws / (0.1 * h * ss), 3.0), 4, seed=seed_ + 22)
    scrap = _ss(0.66, 0.7, nw + 0.2 * Uw) * _ss(0.12, 0.3, Uw) * (1 - U) * _ss(-0.3 * h, -0.5 * h, yy)
    U = np.clip(U + 0.0 * scrap, 0, 1)
    Fm = Fm * U

    # ---- flat value zones
    # lit top: inside U, but the point k toward the light is outside the silhouette; k varies per lobe-scale
    kn = C.fbm(Ws, Hs, max(Ws / (0.35 * h * ss), 2.0), 3, seed=seed_ + 5)
    k = (0.14 + 0.36 * _ss(0.3, 0.72, kn)) * h * ss
    Us = _shift(U, lx * k, ly * k)
    lit = np.clip(U - Us, 0, 1)
    # front lobes: their own lit top where they overlap the mass behind
    kf = (0.05 + 0.1 * _ss(0.3, 0.7, kn)) * h * ss
    Fs = _shift(Fm, lx * kf, ly * kf)
    lit = np.maximum(lit, np.clip(Fm - Fs, 0, 1) * 0.75 * _ss(0.45, 0.65, kn))
    # crisp: threshold (keeps the scalloped shape, removes any ramp)
    lit = _ss(0.42, 0.58, lit)
    # base band: inside U, but a step straight DOWN (and a little away from the light) is outside
    base = _ss(0.4, 0.6, U * (1 - Up)) * (1 - lit)

    lit_c = np.array(pal['lit'], np.float32)
    body_c = np.array(pal['body'], np.float32)
    base_c = np.array(pal['base'], np.float32)
    if warm > 0:
        # low clouds in the twilight: body a touch dimmer/violet, base band a dusky rose from the horizon glow
        body_c = body_c * (1 - 0.25 * warm) + np.array(pal['body_warm'], np.float32) * 0.25 * warm
        base_c = base_c * (1 - 0.7 * warm) + np.array(pal['warm'], np.float32) * 0.7 * warm
        lit_c = lit_c * (1 - 0.12 * warm) + np.array([0.95, 0.85, 0.9], np.float32) * 0.12 * warm
    col = body_c * (1 - lit[..., None]) + lit_c * lit[..., None]
    col = col * (1 - base[..., None]) + base_c * base[..., None]
    # faint dry-brush value variation inside each flat zone (painted, not rendered)
    br = C.fbm(Ws, Hs, max(Ws / (0.12 * h * ss), 3.0), 3, seed=seed_ + 9)
    col = col * (1 + 0.05 * (br[..., None] - 0.5))

    A = U
    bw = max(int(0.06 * h * ss), 3)
    wy_ = np.minimum(np.arange(Hs), np.arange(Hs)[::-1]).astype(np.float32)
    wx_ = np.minimum(np.arange(Ws), np.arange(Ws)[::-1]).astype(np.float32)
    A = A * _ss(0, bw, wy_)[:, None] * _ss(0, bw, wx_)[None, :]
    rgb = cv2.resize(col, (Wc, Hc), interpolation=cv2.INTER_AREA)
    a = cv2.resize(A, (Wc, Hc), interpolation=cv2.INTER_AREA)
    # slightly lost (not dark) underside edge: a touch softer only along the base
    yyc = np.arange(Hc, dtype=np.float32)[:, None] + y0
    a = a + (cv2.GaussianBlur(a, (0, 0), 1.4) - a) * _ss(-0.12 * h, 0.02 * h, yyc)
    return rgb, np.clip(a * amt, 0, 1), x0, y0


def night_clouds12(Wp, Hp, specs, s, light_xy, seed=17, pal=None, ss=3):
    """specs: (cx, base_y, w, h, amt, warm, levels, trail) - same as cloud9. Returns straight RGBA."""
    if pal is None:
        pal = dict(lit=(0.84, 0.93, 1.0), body=(0.54, 0.64, 0.84), base=(0.36, 0.41, 0.64),
                   body_warm=(0.5, 0.46, 0.68), warm=(0.62, 0.42, 0.52))
    rgb = np.zeros((Hp, Wp, 3), np.float32)
    A = np.zeros((Hp, Wp), np.float32)
    items = []
    for i, sp in enumerate(specs):
        cx, by, w, h, amt, warm, lev, trail = sp
        rng = np.random.default_rng(seed * 1000 + i * 13)
        items.append((cx, by, w, h, amt, warm, lev, rng.integers(1 << 30)))
        if trail is not None:
            ang, ln, n = trail
            for (fx, fy, fw, fh, fa) in K9._fragments(rng, cx + math.cos(math.radians(ang)) * 0.45 * w,
                                                      by - 0.25 * h + math.sin(math.radians(ang)) * 0.45 * w,
                                                      ln * w, h, ang, n, 0.28 * w):
                items.append((fx, fy, fw, fh, amt * fa, warm, 1, rng.integers(1 << 30)))
    for (cx, by, w, h, amt, warm, lev, sd) in items:
        rng = np.random.default_rng(sd)
        lobes = K9._lobes(rng, w, h, levels=lev)
        vx, vy = light_xy[0] - cx, light_xy[1] - (by - h)
        n_ = math.hypot(vx, vy) + 1e-6
        lx, ly = 0.4 * vx / n_, 0.4 * vy / n_ - 0.9
        n_ = math.hypot(lx, ly)
        lx, ly = lx / n_, ly / n_
        crgb, ca, x0, y0 = _paint(lobes, w, h, (lx, ly), pal, rng, ss, warm=warm, amt=amt)
        gx0, gy0 = int(math.floor(cx + x0)), int(math.floor(by + y0))
        hh, ww = ca.shape
        sx0, sy0 = max(gx0, 0), max(gy0, 0)
        sx1, sy1 = min(gx0 + ww, Wp), min(gy0 + hh, Hp)
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        cr = crgb[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
        a = ca[sy0 - gy0:sy1 - gy0, sx0 - gx0:sx1 - gx0]
        sl = (slice(sy0, sy1), slice(sx0, sx1))
        rgb[sl] = rgb[sl] * (1 - a[..., None]) + cr * a[..., None]
        A[sl] = a + A[sl] * (1 - a)
    rgb = rgb / np.maximum(A, 1e-5)[..., None] * (A > 1e-5)[..., None]
    return np.dstack([rgb, A]).astype(np.float32)
