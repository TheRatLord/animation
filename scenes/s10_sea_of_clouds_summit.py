"""s10 helpers - the summit foreground, distant peaks and sky for the sea-of-clouds finale.

Everything is painted procedurally into straight-alpha RGBA plates (float32, plate px), sized relative
to the frame width so the layout is the same at 960 and 1920.

  sky_plate          dawn gradient: navy zenith -> blue -> lilac -> rose -> gold -> white-hot horizon
  peaks_plate        distant mountains standing in the cloud sea (fractal ridges, painted lit / shade
                     flanks with gully striations, gold ridge rim on the sun side, aerial perspective)
  ridge_plate        a nearer forested ridge (conifer fringe) rising out of the clouds
  summit_plates      the foreground: rocky summit with grass, a weathered vermilion torii with a
                     shimenawa + shide papers, a wooden summit signpost with real Japanese text and a
                     lone Japanese black pine; backlit (cool sky fill on up-facing planes, near-silhouette
                     faces, hot gold rim light on every sun-facing contour, translucent glowing grass).
                     Returns (static RGBA, sway RGBA, sway weight) so grass / needles / papers can move.
"""
import math
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

F32 = np.float32


# ----------------------------------------------------------------------------------------------- utils
def ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-12), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ss_py(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def fbm(w, h, cells, seed, octaves=4, stretch=1.0, angle=0.0):
    from lib import clouds3 as K
    return K._noise(w, h, cells, seed, octaves, stretch, angle)


def frac1d(n, seed, octaves=6, base=6, gain=0.55):
    """1D fractal noise in ~-1..1, length n."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    out = np.zeros(n)
    amp, tot = 1.0, 0.0
    k = base
    for _ in range(octaves):
        pts = rng.uniform(-1, 1, k + 2)
        xs = np.linspace(0, 1, k + 2)
        # cosine interpolation
        u = x * (k + 1)
        i = np.clip(u.astype(int), 0, k)
        fr = u - i
        fr = (1 - np.cos(fr * math.pi)) / 2
        out += amp * (pts[i] * (1 - fr) + pts[i + 1] * fr)
        tot += amp
        amp *= gain
        k *= 2
    return out / tot


def poly_mask(w, h, pts, ss_=4):
    """Anti-aliased polygon mask (h, w) float32."""
    m = np.zeros((h * ss_ // ss_, w), np.uint8)
    p = np.round(np.asarray(pts, np.float64) * 16).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [p], 255, lineType=cv2.LINE_AA, shift=4)
    return m.astype(F32) / 255.0


def over(dst, rgb, a):
    """Straight-alpha over onto an RGBA canvas (in place)."""
    a = np.clip(a, 0, 1)
    A0 = dst[..., 3]
    Ao = a + A0 * (1 - a)
    num = rgb * a[..., None] + dst[..., :3] * (A0 * (1 - a))[..., None]
    dst[..., :3] = num / np.maximum(Ao, 1e-6)[..., None]
    dst[..., 3] = Ao
    return dst


def shift_toward(img, sx, sy, dist):
    """Sample img at p + dist * unit(sun - p) (per pixel direction toward the sun)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    dx, dy = sx - xx, sy - yy
    n = np.sqrt(dx * dx + dy * dy) + 1e-6
    return cv2.remap(img, xx + dx / n * dist, yy + dy / n * dist, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rim_light(alpha, sx, sy, px, soft=0.6, macro=0.0):
    """Rim mask: pixels inside `alpha` whose neighbour toward the sun (px away) is empty.
    macro > 0: test against a blurred (macro-scale) silhouette so fine fringes (needles, grass) only glow
    where the whole mass faces the sun."""
    am = alpha if not macro else np.clip(cv2.GaussianBlur(alpha, (0, 0), macro) * 1.6, 0, 1)
    a1 = shift_toward(am, sx, sy, px)
    a2 = shift_toward(am, sx, sy, px * 2.6)
    r = alpha * (1 - a1) + 0.35 * alpha * (1 - a2)
    if soft:
        r = cv2.GaussianBlur(r, (0, 0), soft)
    return np.clip(r, 0, 1.2) * alpha


def rim_normal(alpha, sx, sy, px, soft=0.4, macro=1.5, power=1.5, bias=0.15, up=0.0, depth=None):
    """Backlight rim from the silhouette NORMAL: an edge band (exp falloff over `px` inward from the
    contour) weighted by max(0, n . s)^power, n = outward normal of the macro-blurred silhouette and
    s = unit direction toward the sun (per pixel; `up` blends it toward straight up). Edges turned away
    from the sun (undersides, far sides) get nothing - no outline all the way round."""
    h, w = alpha.shape
    am = cv2.GaussianBlur(alpha, (0, 0), macro) if macro else alpha
    gx = cv2.Sobel(am, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(am, cv2.CV_32F, 0, 1, ksize=3)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gn, -gy / gn
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    dx, dy = sx - xx, sy - yy
    dn = np.sqrt(dx * dx + dy * dy) + 1e-6
    dx, dy = dx / dn, dy / dn
    if up:
        dx, dy = dx * (1 - up), dy * (1 - up) - up
        dn = np.sqrt(dx * dx + dy * dy) + 1e-6
        dx, dy = dx / dn, dy / dn
    f = np.clip((nx * dx + ny * dy - bias) / (1 - bias), 0, 1) ** power
    # gradient strength gate: only where there IS a contour nearby
    f = f * np.clip(gn * (macro + 1.0) * 1.5, 0, 1)
    dist = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    band = np.exp(-np.maximum(dist - 1.0, 0.0) / max(px, 0.3)) * np.clip(alpha, 0, 1)
    r = band * f
    if soft:
        r = cv2.GaussianBlur(r.astype(F32), (0, 0), soft)
    return (np.clip(r, 0, 1.2) * alpha).astype(F32)


def pad_facing(alpha, sc, macro=5.0):
    """(allowed, down) masks from the macro silhouette normal of a pine pad: 'allowed' = upper edges and
    edges turned to the lower-left sun; 'down' = undersides (normal pointing down / away)."""
    am = cv2.GaussianBlur(alpha.astype(F32), (0, 0), macro * sc)
    gx = cv2.Sobel(am, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(am, cv2.CV_32F, 0, 1, ksize=3)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gn, -gy / gn
    up = ss(0.25, -0.15, ny)
    sunf = ss(0.5, 0.85, -nx) * ss(0.75, 0.45, ny)
    allowed = np.maximum(up, sunf)
    down = ss(0.2, 0.6, ny) * (1 - sunf)
    return allowed.astype(F32), down.astype(F32)


def bleed(rgba, sigma):
    from lib import clouds3 as K
    rgba[..., :3] = K._bleed(rgba[..., :3], rgba[..., 3], sigma)
    return rgba


# ------------------------------------------------------------------------------------------------ sky
SKY_STOPS = [(0.0, (1.00, 0.92, 0.72)), (0.008, (1.00, 0.76, 0.40)), (0.035, (1.00, 0.56, 0.26)),
             (0.09, (0.96, 0.44, 0.34)), (0.18, (0.80, 0.44, 0.52)), (0.3, (0.56, 0.44, 0.70)),
             (0.46, (0.32, 0.40, 0.74)), (0.7, (0.16, 0.28, 0.62)), (1.0, (0.06, 0.14, 0.40))]


def sky_plate(w, h, hy, sun, seed=3):
    """RGB sky (h, w, 3). u = height above the horizon in units of hy."""
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    u = np.clip((hy - yy) / hy, 0, 1)
    us = np.array([s[0] for s in SKY_STOPS], F32)
    cs = np.array([s[1] for s in SKY_STOPS], F32)
    col = np.stack([np.interp(u, us, cs[:, k]) for k in range(3)], -1).astype(F32)
    # warm sun side: the lower sky is hotter / more golden toward the sun, cooler rose away from it
    dxs = (xx - sun[0]) / w
    near = np.exp(-(dxs / 0.28) ** 2)[..., None]
    low = np.exp(-u / 0.22)[..., None]
    col = col * (1 - 0.25 * near * low) + np.array([1.15, 0.8, 0.5], F32) * 0.25 * near * low
    away = (1 - near) * low
    col = col * (1 - 0.12 * away) + np.array([0.75, 0.55, 0.78], F32) * 0.12 * away
    # subtle painted variation (large soft value shifts, no noise speckle)
    n = fbm(w, h, 3.0, seed, 3, stretch=3.0)
    col = col * (1 + 0.025 * n[..., None])
    return col.astype(F32)


# ------------------------------------------------------------------------------------------ mountains
def _ridge_profile(n, seed, peaks, rough=0.3, expo=1.5, round_=0.0):
    """peaks: list of (x0..1, height0..1, half_width) -> profile height 0..1 over n samples."""
    x = np.linspace(0, 1, n)
    prof = np.zeros(n)
    rng = np.random.default_rng(seed)
    for (px, ph, hw) in peaks:
        d = (x - px) / hw
        d = np.where(d > 0, d * rng.uniform(0.8, 1.25), -d * rng.uniform(0.8, 1.25))
        # concave flanks (steeper near the summit) with shoulders
        if round_:
            d = np.sqrt(d * d + round_ * round_) - round_
        p = ph * np.clip(1 - d, 0, 1) ** expo
        prof = np.maximum(prof, p)
    r1 = frac1d(n, seed, 7, 7, 0.62)
    r2 = frac1d(n, seed + 1, 6, 50, 0.55)
    prof = prof * (1 + rough * r1) + 0.3 * rough * r2 * (0.2 + prof)
    return np.clip(prof, 0, None)


def peak_layer(w, h, x0, x1, base_y, height, peaks, seed, sun, lit_col, shade_col, haze_col, haze, rim_col,
               rim_px, snow=0.0, lit_amt=1.0):
    """One mountain group as RGBA plate. x0..x1 plate px span, base_y the (hidden) foot, height px.
    Screen-space painted relief: a dome from the silhouette distance + vertical spurs (ridged noise
    along the fall line); faces turned toward the sun catch warm light, the rest is cool violet shade,
    stepped into flat painted planes."""
    n = int(x1 - x0)
    prof = _ridge_profile(n, seed, peaks)
    top = base_y - prof * height
    xs = np.arange(n) + x0
    pts = [(x0, base_y + 4)] + list(zip(xs, top)) + [(x1, base_y + 4)]
    m = poly_mask(w, h, pts)
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    ttop = np.interp(xx[0], xs, top).astype(F32)[None, :]
    depth_in = np.clip((yy - ttop) / max(height, 1), 0, 1)
    # relief field: the profile extruded downward (so each summit sheds ridges) + spurs
    # the profile spreads (blurs) downward like a cone and its spine wanders (x warp grows with depth)
    prw = np.interp(np.arange(w, dtype=F32), xs, prof, left=0, right=0).astype(F32)
    lv = [0.0, 0.006, 0.015, 0.03, 0.05, 0.08]
    stack = [prw if s_ == 0 else cv2.GaussianBlur(prw[None], (0, 0), s_ * w)[0] for s_ in lv]
    wx = fbm(w, h, max(w / (0.08 * w), 4), seed + 9, 3) * 0.035 * w * depth_in
    xw = np.clip(xx + wx, 0, w - 1)
    li = depth_in * (len(lv) - 1)
    i0 = np.clip(li.astype(int), 0, len(lv) - 2)
    fr = li - i0
    pr = np.zeros((h, w), F32)
    for k in range(len(lv) - 1):
        a_ = np.interp(xw, np.arange(w), stack[k])
        b_ = np.interp(xw, np.arange(w), stack[k + 1])
        pr += np.where(i0 == k, a_ * (1 - fr) + b_ * fr, 0)
    sp_cells = max(w / (0.045 * w), 6)
    s1 = fbm(w, h, sp_cells, seed + 3, 5, stretch=0.25)          # vertical streaks (fall lines)
    s2 = fbm(w, h, sp_cells * 2.5, seed + 4, 4, stretch=0.35)
    ridged = 1 - np.abs(s1)
    R = pr * height + 0.045 * height * ridged * (0.3 + depth_in) + 0.006 * height * s2
    R = cv2.GaussianBlur(R.astype(F32), (0, 0), 1.5 * w / 2227)
    gx = np.gradient(R, axis=1)
    sdir = 1.0 if sun[0] > (x0 + x1) / 2 else -1.0
    # facing the sun: surface falls toward the sun's side
    light = np.clip(-gx * sdir * 1.2, -1, 1) * 0.5 + 0.5
    light = light - 0.25 * depth_in + 0.1 * s2
    p1 = ss(0.5, 0.56, light)
    p2 = ss(0.72, 0.78, light)
    shade_col = np.asarray(shade_col, F32)
    lit_col = np.asarray(lit_col, F32)
    col = shade_col * (1 + 0.1 * s2[..., None])
    col = col + (shade_col * 0.35 + lit_col * 0.65 - col) * (p1 * lit_amt)[..., None]
    col = col + (lit_col - col) * (p2 * lit_amt)[..., None]
    if snow:
        sn = ss(0.2, 0.5, ridged - depth_in * 3) * snow
        col = col + (np.array([0.85, 0.8, 0.95], F32) * (0.7 + 0.5 * p1[..., None]) - col) * sn[..., None]
    # aerial perspective: lower slopes melt into the mist of the cloud sea
    mist = ss(0.3, 0.95, depth_in)[..., None] * 0.7
    hz = np.clip(haze + mist * (1 - haze), 0, 1)
    col = col * (1 - hz) + np.asarray(haze_col, F32) * hz
    rim = rim_light(m, sun[0], sun[1], rim_px, soft=0.5)
    col = col + np.asarray(rim_col, F32) * rim[..., None] * (1 - haze * 0.7)
    out = np.dstack([col, m]).astype(F32)
    return bleed(out, 3.0)


# ---------------------------------------------------------------------------------------------- ridge
def _cedar(rng, x, yb, hh, ww):
    """Cedar (sugi) silhouette polygon: a narrow spire of 6-10 drooping, notched tiers (clear spire tip)."""
    k = int(rng.integers(6, 11))
    left, right = [], []
    for i in range(1, k + 1):
        fy = i / k
        y = yb - hh + hh * fy
        for side, lst in ((-1, left), (1, right)):
            wd = ww * (fy ** 0.85) * rng.uniform(0.8, 1.15)
            lst.append((x + side * wd, y + hh * 0.02))                                   # drooping tier tip
            lst.append((x + side * wd * rng.uniform(0.35, 0.6), y + hh / k * rng.uniform(0.2, 0.4)))   # notch
    tip = [(x + rng.uniform(-0.04, 0.04) * ww, yb - hh * 1.05)]
    bot = [(x + ww * 0.3, yb + hh * 0.3), (x - ww * 0.3, yb + hh * 0.3)]
    return tip + right + bot + left[::-1]


def ridge_plate(w, h, pts_x, pts_y, base_y, seed, sun, sc, shade=(0.075, 0.07, 0.19), lit=(0.13, 0.1, 0.25),
                crest=(0.06, 0.055, 0.15), haze_col=(0.46, 0.34, 0.6), rim_col=(1.9, 1.15, 0.55), fade=(0.55, 0.95)):
    """Forested mountain shoulder rising out of the cloud sea.
    - silhouette: a ridge line crowned by a readable cedar treeline (distinct spires in stands of different
      height, a few gaps and broadleaf crowns), trees shrinking with distance along the crest;
    - inside: a few painted value planes (spurs / gullies falling from the crest; the faces turned toward
      the sun a little lighter and warmer), darker directly under the crest treeline;
    - a thin hot gold rim only where the silhouette faces the sun (plus a faint sky rim on top);
    - aerial perspective: the lower slopes melt into lavender haze, and the base tears into the clouds."""
    rng = np.random.default_rng(seed)
    xs = np.arange(w, dtype=F32)
    top = np.interp(xs, pts_x, pts_y) + 4 * sc * frac1d(w, seed, 5, 10, 0.5)
    S2 = 2
    M = np.zeros((h * S2, w * S2), np.uint8)
    poly = [(0, base_y)] + [(float(x), float(y) + 6 * sc) for x, y in zip(xs[::2], top[::2])] + [(w - 1, base_y)]
    cv2.fillPoly(M, [np.round(np.asarray(poly) * S2 * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    clus = frac1d(w, seed + 3, 5, 9, 0.6)
    clus = (clus - clus.min()) / (clus.max() - clus.min() + 1e-6)
    x = 0.0
    trees = []
    while x < w:
        xi = int(np.clip(x, 0, w - 1))
        c = float(clus[xi])
        if c < 0.18 and rng.random() < 0.7:           # clearing: bare ridge line
            x += 6 * sc
            continue
        hh = sc * rng.uniform(18, 30) * (0.55 + 1.0 * c ** 1.2) * (1 + 0.5 * rng.random() ** 4)
        if top[xi] < base_y - 4 * sc:
            yb = top[xi] + hh * rng.uniform(0.3, 0.55) + 2 * sc
            trees.append((x, yb, hh, hh * rng.uniform(0.22, 0.3), rng.random() < 0.15))
        x += hh * rng.uniform(0.1, 0.24)
    for (x, yb, hh, ww, broad) in trees:
        if broad:
            for _ in range(int(rng.integers(3, 6))):
                cx_ = x + rng.uniform(-0.9, 0.9) * ww
                cy_ = yb - hh * rng.uniform(0.3, 0.6)
                rr = hh * rng.uniform(0.16, 0.26)
                cv2.circle(M, (int(cx_ * S2 * 16), int(cy_ * S2 * 16)), int(rr * S2 * 16), 255, -1, cv2.LINE_AA, shift=4)
        else:
            q = np.asarray(_cedar(rng, x, yb, hh, ww)) * S2
            cv2.fillPoly(M, [np.round(q * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    a = cv2.resize(M.astype(F32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
    yy = np.arange(h, dtype=F32)[:, None]
    din = np.clip((yy - top[None, :]) / (base_y - top[None, :] + 1e-3), 0, 1)
    # painted treatment: one near-uniform cool blue-violet silhouette with only two broad value planes
    # (a big soft diagonal plane: the shoulder turned toward the sun a touch lighter), darker right under
    # the crest treeline, and aerial haze growing toward the base where it sinks into the cloud
    sdir = 1.0 if sun[0] > float(np.mean(pts_x)) else -1.0
    xn = (xs[None, :] - float(np.mean(pts_x))) / w
    pl = ss(-0.05, 0.12, -sdir * xn + 0.25 * (din - 0.3))           # broad plane, away from the sun side
    col = np.asarray(lit, F32) + (np.asarray(shade, F32) - np.asarray(lit, F32)) * pl[..., None]
    col = col * np.ones((h, 1, 1), F32)
    crown = 1 - ss(0.0, 0.1, din)
    col = col + (np.asarray(crest, F32) - col) * (crown * 0.7)[..., None]
    hz = ss(0.2, 1.0, din) ** 1.3 * 0.75
    col = col * (1 - hz[..., None]) + np.asarray(haze_col, F32) * hz[..., None]
    # torn base
    nb = fbm(w, h, max(w / (60 * sc), 4), seed + 5, 4, stretch=5.0)
    a = a * (1 - ss(fade[0], fade[1], din + 0.22 * nb))
    env = np.clip((cv2.GaussianBlur(a, (0, 0), 3.0 * sc) - 0.35) * 4, 0, 1)
    rim_s = rim_light(env, sun[0], sun[1], 1.6 * sc, soft=0.6 * sc) * a +         0.35 * rim_light(a, sun[0], sun[1], 1.0 * sc, soft=0.4 * sc)
    rim_u = rim_light(env, sun[0], sun[1] - 3 * h, 1.2 * sc, soft=0.5 * sc) * a
    dx = (xs[None, :] - sun[0]) / w
    prox = np.exp(-(dx / 0.25) ** 2)
    r = (rim_s * (0.45 + 0.7 * prox) + 0.15 * rim_u) * ss(0.3, 0.0, din)
    col = col + np.asarray(rim_col, F32) * r[..., None]
    # sun-side glow just inside the lit crest (sky light scattering through the treeline)
    gl = rim_light(a, sun[0], sun[1], 7 * sc, soft=4 * sc, macro=4 * sc) * ss(0.25, 0.0, din) * prox
    col = col + np.array([0.5, 0.25, 0.12], F32) * gl[..., None] * 0.4
    out = np.dstack([col, a]).astype(F32)
    return bleed(out, 3.0)


def wisp_plate(w, h, x0, x1, yc, thick, seed, sun, sc, lit=(1.2, 0.8, 0.62), body=(0.5, 0.42, 0.68),
               shade=(0.24, 0.2, 0.44), amount=0.9):
    """Soft cloud tongues lying across the foot of the ridge (overlapping it): stretched noise masses with
    a warm lit upper edge, lavender body and cool violet underside; edges lost at the ends."""
    xx = np.arange(w, dtype=F32)[None, :]
    yy = np.arange(h, dtype=F32)[:, None]
    n = fbm(w, h, max(w / (140 * sc), 3), seed, 5, stretch=7.0, angle=-2.0)
    n2 = fbm(w, h, max(w / (35 * sc), 3), seed + 1, 4, stretch=4.0, angle=-2.0)
    v = (yy - yc) / thick
    win = np.exp(-v ** 2 * 1.2) * ss(x0, x0 + 0.12 * w, xx) * (1 - ss(x1 - 0.12 * w, x1, xx))
    d = n + 0.35 * n2 + 0.9 * win - 0.9
    a = ss(-0.05, 0.2, d) * win * amount
    a = cv2.GaussianBlur(a.astype(F32), (0, 0), 1.2 * sc)
    ab = cv2.GaussianBlur(a, (0, 0), 3 * sc)
    up = np.clip(ab - np.roll(ab, int(6 * sc) + 1, axis=0), 0, 1)      # upper edge (empty above)
    dn = np.clip(ab - np.roll(ab, -int(10 * sc) - 1, axis=0), 0, 1)
    col = np.asarray(body, F32) * np.ones((h, w, 1), F32)
    col = col + (np.asarray(shade, F32) - col) * ss(0.0, 0.25, dn)[..., None]
    col = col + (np.asarray(lit, F32) - col) * ss(0.02, 0.2, up)[..., None]
    out = np.dstack([col, a]).astype(F32)
    return bleed(out, 3.0)


# -------------------------------------------------------------------------------------------- summit
def _text_img(txt, px, font='C:/Windows/Fonts/YuGothB.ttc', vertical=True):
    f = ImageFont.truetype(font, max(int(px), 6))
    if vertical:
        wd = int(px * 1.2)
        ht = int(px * 1.12 * len(txt)) + 4
        im = Image.new('L', (wd, ht), 0)
        d = ImageDraw.Draw(im)
        for i, ch in enumerate(txt):
            d.text((wd / 2, i * px * 1.12 + 2), ch, fill=255, font=f, anchor='mt')
    else:
        bb = f.getbbox(txt)
        im = Image.new('L', (bb[2] + 4, bb[3] + 4), 0)
        ImageDraw.Draw(im).text((2, 2), txt, fill=255, font=f)
    return np.asarray(im, F32) / 255.0


def _stamp(dst, src, x, y):
    """Max-stamp small (h, w) mask into dst at top-left (x, y)."""
    h, w = src.shape
    x, y = int(round(x)), int(round(y))
    H, W = dst.shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    dst[y0:y1, x0:x1] = np.maximum(dst[y0:y1, x0:x1], src[y0 - y:y1 - y, x0 - x:x1 - x])


# backlit silhouette palette: everything in front of the sun is a deep plum / indigo value mass
SIL_D = np.array([0.030, 0.026, 0.048], F32)
SIL_M = np.array([0.055, 0.045, 0.075], F32)
SIL_L = np.array([0.085, 0.072, 0.115], F32)      # cool sky fill on up-facing planes (very subdued)


class Summit:
    """Builds the foreground plates. All layout in frame fractions (fx, fy) mapped to plate px."""

    def __init__(self, PW, PH, W, H, ox, oy, sun, seed=11):
        self.PW, self.PH, self.W, self.H, self.ox, self.oy = PW, PH, W, H, ox, oy
        self.sc = W / 1920.0
        self.sun = sun            # plate px (at the frame the plate was built for)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def P(self, fx, fy):
        return (self.ox + fx * self.W, self.oy + fy * self.H)

    # ----------------------------------------------------------------------------------- ground line
    def ground(self):
        PW, PH, sc = self.PW, self.PH, self.sc
        xs = np.arange(PW, dtype=np.float64)
        fx = (xs - self.ox) / self.W
        kx = [-0.2, 0.0, 0.12, 0.3, 0.42, 0.52, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
        ky = [0.735, 0.74, 0.75, 0.765, 0.795, 0.845, 0.89, 0.905, 0.875, 0.835, 0.81, 0.795]
        base = np.interp(fx, kx, ky)
        g = self.oy + base * self.H + 7 * sc * frac1d(PW, self.seed + 1, 6, 10, 0.55) + \
            2.5 * sc * frac1d(PW, self.seed + 2, 4, 160, 0.5)
        self.gy = g.astype(F32)
        return self.gy

    def mound_line(self):
        """Top of the nearer foreground mound (a step in front of the crest plane)."""
        PW, sc = self.PW, self.sc
        xs = np.arange(PW, dtype=np.float64)
        fx = (xs - self.ox) / self.W
        kx = [-0.3, -0.05, 0.1, 0.22, 0.34, 0.46, 0.58, 0.7, 0.8, 0.9, 1.0, 1.3]
        ky = [0.875, 0.88, 0.895, 0.915, 0.94, 0.955, 0.965, 0.97, 0.96, 0.945, 0.925, 0.91]
        base = np.interp(fx, kx, ky)
        g = self.oy + base * self.H + 6 * sc * frac1d(PW, self.seed + 5, 6, 8, 0.55) + \
            2.0 * sc * frac1d(PW, self.seed + 6, 4, 120, 0.5)
        return np.maximum(g, self.gy + 0.03 * self.H).astype(F32)

    # ------------------------------------------------------------------------------------------ rocks
    def rock_shapes(self):
        R = []
        spec = [(0.03, 0.05, 0.034), (0.155, 0.022, 0.018), (0.46, 0.036, 0.024), (0.885, 0.04, 0.028),
                (0.6, 0.03, 0.022)]
        for i, (fx, rx, ry) in enumerate(spec):
            x = self.ox + fx * self.W
            gi = int(np.clip(x, 0, self.PW - 1))
            y = self.gy[gi] + ry * self.H * 0.55
            R.append((x, y, rx * self.W, ry * self.H * 1.7, self.seed * 7 + i))
        return R

    def rock_mask(self, cx, cy, rx, ry, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(26, 34))
        ang = np.sort(np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(-0.06, 0.06, n))
        # weathered boulder: a broad rounded mass with small chips (no low-poly facets)
        r = 1 + 0.07 * np.sin(ang * 2 + rng.uniform(0, 6)) + 0.05 * rng.uniform(-1, 1, n)
        pts = []
        for a, rr in zip(ang, r):
            y = math.sin(a)
            yf = 0.75 if y < 0 else 1.0
            pts.append((cx + math.cos(a) * rx * rr, cy - (-y) * ry * rr * yf))
        # chipped crown: a couple of small notches on the upper outline
        return pts

    # ------------------------------------------------------------------------------------------ build
    def build(self):
        PW, PH, sc = self.PW, self.PH, self.sc
        sx, sy = self.sun
        yy, xx = np.mgrid[0:PH, 0:PW].astype(F32)
        gy = self.ground()
        S = np.zeros((PH, PW, 4), F32)
        gm = (yy >= gy[None, :]).astype(F32)
        gm = cv2.GaussianBlur(gm, (0, 0), 0.6)
        din = np.clip((yy - gy[None, :]) / (0.25 * self.H), 0, 1)
        # a clean painted silhouette in TWO planes (a readable value step, no texture mush):
        #  * the crest plane behind: near-black violet, with a narrow cool sky-lift hanging under its edge
        #  * a nearer mound in front, a step darker, with its own crisp edge and grass fringe
        n1 = fbm(PW, PH, max(PW / (200 * sc), 3), 31, 3, stretch=3.0, angle=-4)
        # back ridge: a readable lifted cool value (sky light on the far slope), clearly above the near mound
        # (crushed: a near-black violet backlit silhouette, value < 20 once graded)
        back_top = np.array([0.04, 0.03, 0.058], F32)
        back_low = np.array([0.014, 0.011, 0.024], F32)
        k = (0.6 * np.exp(-din / 0.06) + 0.4 * np.exp(-din / 0.3))[..., None]
        col = back_low + (back_top - back_low) * k
        col = col * (1 + 0.06 * n1[..., None])
        # warm backlit crest: the grass along the ridge line glows with the low sun (translucent turf),
        # strongest toward the sun column, a lost soft falloff down the slope
        dxs_ = (xx - sx) / self.W
        sunw_ = 0.3 + 0.7 * np.exp(-(dxs_ / 0.35) ** 2)
        crest = ((0.75 * np.exp(-din / 0.012) + 0.25 * np.exp(-din / 0.045)) * (0.75 + 0.25 * ss(-0.3, 0.3, n1)) * sunw_)[..., None]
        col = col + (np.array([0.26, 0.11, 0.06], F32) - col) * (0.55 * crest)
        # painterly brush strokes laid along the slope (value + hue breaks, not noise mush)
        stk = fbm(PW, PH, max(PW / (70 * sc), 3), 38, 3, stretch=7.0, angle=-6)
        stk2 = fbm(PW, PH, max(PW / (22 * sc), 3), 39, 2, stretch=5.0, angle=-8)
        sv = ss(0.05, 0.25, stk + 0.3 * stk2) - 0.6 * ss(0.1, 0.3, -stk - 0.3 * stk2)
        col = col * (1 + 0.1 * sv[..., None])
        # broad painted value masses: sky-lit slope planes (cool bounce) with crisp brushed borders, and
        # darker hollows between them, fading with depth down the slope
        n_lo = fbm(PW, PH, max(PW / (380 * sc), 3), 35, 3, stretch=3.0, angle=-5)
        n_md = fbm(PW, PH, max(PW / (90 * sc), 3), 36, 3, stretch=2.2, angle=-5)
        pl_ = ss(0.02, 0.07, n_lo + 0.22 * n_md) * ss(0.004, 0.03, din / 4.0)
        hol = ss(0.08, 0.14, -n_lo + 0.2 * n_md)
        bounce = np.array([0.05, 0.042, 0.085], F32)        # cool sky bounce on the slope planes
        col = col + (bounce - col) * (pl_ * (0.3 - 0.2 * din))[..., None]
        col = col * (1 - 0.3 * hol * ss(0.0, 0.1, din))[..., None]
        # painted turf: thousands of small grass-tuft strokes (sky-lit cool tips / dark roots) laid into
        # the whole slope, growing toward the camera, warm-tinted near the backlit crest
        self.turf_l, self.turf_d = self._turf(gy)
        tl, td = self.turf_l, self.turf_d
        warm_t = (np.exp(-din / 0.08) * sunw_)[..., None]
        lcol = np.array([0.13, 0.12, 0.23], F32) * (1 - warm_t) + np.array([0.34, 0.16, 0.09], F32) * warm_t
        # (only near the crest: down the slope the silhouette stays one clean near-black mass, no speckle)
        col = col + (lcol - col) * (tl * 0.4 * np.exp(-din / 0.05))[..., None]
        col = col * (1 - 0.2 * td)[..., None]
        # broad painted value shapes down the slope (the lower ~150 px of the frame): a few flat,
        # half-buried rock slabs whose up-facing top planes catch a faint cool sky fill (hard ragged upper
        # border, lost below), and grass tufts whose tips pick up the same cool fill - so the foot of the
        # frame is not a dead black slab
        rgm = np.random.default_rng(self.seed + 313)
        Rk = np.zeros((PH, PW), F32)
        Rt = np.zeros((PH, PW), F32)
        for i in range(9):
            cxr = self.ox + (i + rgm.uniform(0.1, 0.9)) / 9 * 1.1 * self.W - 0.05 * self.W
            ci = int(np.clip(cxr, 0, PW - 1))
            cyr = float(gy[ci]) + rgm.uniform(0.07, 0.2) * self.H
            rxr = rgm.uniform(0.035, 0.085) * self.W
            ryr = rxr * rgm.uniform(0.2, 0.3)
            angs = np.linspace(math.pi, 2 * math.pi, 11)
            pts = [(cxr + math.cos(a_) * rxr * rgm.uniform(0.9, 1.05),
                    cyr + math.sin(a_) * ryr * rgm.uniform(0.75, 1.05) * (0.55 if abs(math.sin(a_)) > 0.5 else 1.0))
                   for a_ in angs]
            pts = pts + [(cxr + rxr, cyr + ryr * 2.5), (cxr - rxr, cyr + ryr * 2.5)]
            mk = poly_mask(PW, PH, pts)
            Rk = np.maximum(Rk, mk)
            top_y = cyr - ryr * 0.45
            wob_ = fbm(PW, PH, max(PW / (40 * sc), 3), 320 + i, 2)
            tp_ = ss(top_y + 0.35 * ryr + 0.25 * ryr * wob_, top_y + 0.2 * ryr + 0.25 * ryr * wob_, yy) * mk
            Rt = np.maximum(Rt, tp_)
        cool_fill = np.array([0.085, 0.08, 0.15], F32)
        low = (ss(0.12, 0.3, din) * (1 - 0.5 * ss(0.8, 1.2, din)))[..., None]
        col = col * (1 - 0.3 * (Rk - Rt)[..., None] * low) + cool_fill * (Rt * 0.36)[..., None] * low
        tlf = cv2.GaussianBlur(self.turf_l, (0, 0), 0.5 * sc) * (1 - Rk)
        col = col + cool_fill * (tlf * 0.5)[..., None] * low
        # faint bluish aerial haze on the far slope only
        col = col + np.array([0.004, 0.005, 0.012], F32) * np.exp(-din / 0.1)[..., None]
        over(S, col.astype(F32), gm)
        self.gy2 = self.mound_line()
        mm = cv2.GaussianBlur((yy >= self.gy2[None, :]).astype(F32), (0, 0), 0.7 * sc)
        dm = np.clip((yy - self.gy2[None, :]) / (0.2 * self.H), 0, 1)
        nmm = fbm(PW, PH, max(PW / (150 * sc), 3), 37, 3, stretch=2.5)
        mcol = np.array([0.013, 0.01, 0.022], F32) * (1 + 0.8 * np.exp(-dm / 0.04) * (0.8 + 0.4 * ss(-0.3, 0.3, nmm)))[..., None] * (1 - 0.3 * dm[..., None])
        mcol = mcol + np.array([0.008, 0.007, 0.016], F32) * (ss(0.05, 0.12, nmm) * np.exp(-dm / 0.25))[..., None]
        mcol = mcol * (1 + 0.06 * sv[..., None])
        mcol = mcol + cool_fill * (tlf * 0.5)[..., None] * ss(0.03, 0.15, dm)[..., None]
        self.mound = mm
        self.ground_up = gm * 0
        self.extra_rim = np.zeros((PH, PW, 3), F32)
        self.own_rim_mask = np.zeros((PH, PW), F32)
        self.rocks = self.rock_shapes()
        self.rock_all = np.zeros((PH, PW), F32)
        for (cx, cy, rx, ry, sd) in self.rocks:
            self._paint_rock(S, cx, cy, rx, ry, sd)
        self.stone_mask = np.zeros((PH, PW), F32)
        self._torii(S)
        self._signpost(S)
        D = np.zeros((PH, PW, 4), F32)
        Wt = np.zeros((PH, PW), F32)
        self._pine(D, Wt)
        self._shimenawa(S, D, Wt)
        self._fringe(D, Wt, gy, S, mcol, mm)
        self._susuki(D, Wt, gy)
        # rim light on the union silhouette: thin hot gold on edges facing the sun, plus the sky-facing
        # (upper) edges; kept as separate additive plates so the light ramp can swell it over the shot
        Au = np.clip(S[..., 3] + D[..., 3] * (1 - S[..., 3]), 0, 1)
        # normal-based: only contours whose outward normal points at the sun (or, faintly, up at the sky)
        # catch light; undersides and far sides stay pure silhouette
        r_s = rim_normal(Au, sx, sy, 1.1 * sc, soft=0.35 * sc, macro=1.2 * sc, power=1.6, bias=0.2, up=0.35)
        r_u = rim_normal(Au, sx, sy, 1.0 * sc, soft=0.35 * sc, macro=1.2 * sc, power=2.0, bias=0.3, up=0.75)
        dd = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / self.W
        prox = np.exp(-(dd / 0.35) ** 2)
        g_s = (0.55 + 0.8 * prox)
        pa = np.clip(self._pine_A + D[..., 3] * self.pine_pads, 0, 1)
        pp = np.maximum(self.pine_pads, cv2.dilate(pa, np.ones((int(9 * sc) | 1, int(9 * sc) | 1), np.uint8)))
        tm = np.maximum(self.own_rim_mask, self.pine_own)   # parts that paint their own rim (torii, signpost, rocks, pine)
        up_w = 0.35 * (1 - pp) * (1 - tm)
        # pine pads: the lining only on the edges truly facing the sun (no sky-up bias)
        r_p = rim_normal(Au, sx - 0.4 * self.W, sy - 0.9 * self.H, 1.1 * sc, soft=0.35 * sc, macro=2.5 * sc, power=1.8, bias=0.4)
        # pads near the sun: a hot rim on the edges turned to the real sun direction as well
        # (sun direction blended toward straight up: the pads above the sun must not get an underside rim)
        r_p2 = rim_normal(Au, sx, sy, 1.3 * sc, soft=0.35 * sc, macro=2.0 * sc, power=1.4, bias=0.3, up=0.6)
        near_p = np.exp(-(dd / 0.3) ** 2)
        r_p = np.maximum(r_p, r_p2 * near_p * 1.3)
        # warm lining only on the upper edges and the edges turned to the (lower-left) sun: never underneath
        r_p = r_p * self.pad_allow
        r_s = r_s * (1 - pp) + r_p * pp
        rim = np.clip(r_s * g_s * (1 - tm) + r_u * up_w, 0, 1.6)
        rc = np.array([1.75, 1.08, 0.5], F32)
        add = rim[..., None] * rc
        add = add + self.extra_rim + self.paper_rim
        aD = D[..., 3]
        aS = S[..., 3] * (1 - aD)
        Rd = np.dstack([add * aD[..., None], aD]).astype(F32)
        Rs = np.dstack([add * aS[..., None], aS]).astype(F32)
        bleed(S, 2.0)
        bleed(D, 2.0)
        Wt = cv2.GaussianBlur(Wt, (0, 0), 3 * sc)
        return S, D, Wt, Rs, Rd

    # ----------------------------------------------------------------------------------------- painters
    def _turf(self, gy):
        """Grass-tuft brush texture for the ground planes: returns (light strokes, dark strokes) in 0..1."""
        PW, PH, sc, H = self.PW, self.PH, self.sc, self.H
        rng = np.random.default_rng(self.seed + 901)
        ssf = 2
        L = np.zeros((PH * ssf, PW * ssf), np.uint8)
        Dk = np.zeros((PH * ssf, PW * ssf), np.uint8)
        gmin = float(gy.min())
        n = int(26000 * (PW * PH) / (2342 * 1318))
        xs = rng.uniform(0, PW, n)
        ys = rng.uniform(gmin, PH + 10 * sc, n)
        cl = fbm(PW, PH, max(PW / (60 * sc), 3), self.seed + 902, 3)
        for x, y in zip(xs, ys):
            xi = int(min(max(x, 0), PW - 1))
            yi = int(min(max(y, 0), PH - 1))
            g0 = float(gy[xi])
            if y < g0 + 4 * sc:
                continue
            d = (y - g0) / (0.3 * H)
            if cl[yi, xi] < -0.45 + 0.6 * rng.random():
                continue                      # sparser patches between the clumps
            hgt = (3 + 20 * min(d, 1.3) ** 1.1) * sc * rng.uniform(0.5, 1.4)
            nb = int(rng.integers(2, 6))
            lean = -0.2 + rng.normal(0, 0.12)
            img = L if rng.random() < 0.55 else Dk
            val = int(rng.uniform(60, 220))
            wd = max(1, int(round((0.6 + 1.2 * min(d, 1.0)) * sc * ssf)))
            for j in range(nb):
                u = (j + rng.uniform(0.2, 0.8)) / nb - 0.5
                a = -math.pi / 2 + lean + u * rng.uniform(0.6, 1.6)
                ln = hgt * rng.uniform(0.5, 1.0) * (1 - 0.5 * abs(u))
                p0 = (x + u * hgt * 0.2, y)
                p1 = (p0[0] + math.cos(a) * ln, p0[1] + math.sin(a) * ln)
                cv2.line(img, (int(p0[0] * ssf * 16), int(p0[1] * ssf * 16)), (int(p1[0] * ssf * 16), int(p1[1] * ssf * 16)),
                         val, wd, cv2.LINE_AA, 4)
        L = cv2.resize(L.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        Dk = cv2.resize(Dk.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        return L, Dk

    def _paint_rock(self, S, cx, cy, rx, ry, sd):
        """Backlit boulder, near-black: a broken angular outline, TWO facet values (a slightly lifted cool
        top plane vs the near-black face turned away from the sun, split along a ragged ridge line), dark
        cracks with a hairline lit lip, sparse lichen specks, and a crisp rim only on the top edges turned
        toward the sun. The base is buried later by grass tufts that overlap it."""
        sc = self.sc
        sx, sy = self.sun
        pts = self.rock_mask(cx, cy, rx, ry, sd)
        x0, x1 = int(max(cx - rx * 1.5, 0)), int(min(cx + rx * 1.5, self.PW))
        y0, y1 = int(max(cy - ry * 1.5, 0)), int(min(cy + ry * 1.5, self.PH))
        if x1 <= x0 or y1 <= y0:
            return
        w, h = x1 - x0, y1 - y0
        rng = np.random.default_rng(sd)
        # angular outline: chip the polygon with a few straight cuts (fractured, not a blob)
        P_ = [(p[0] - x0, p[1] - y0) for p in pts]
        P2 = []
        for i in range(len(P_)):
            a_, b_ = np.array(P_[i]), np.array(P_[(i + 1) % len(P_)])
            P2.append(tuple(a_))
            if rng.random() < 0.55:
                mid = a_ + (b_ - a_) * rng.uniform(0.35, 0.65)
                nrm = np.array([-(b_ - a_)[1], (b_ - a_)[0]])
                nrm = nrm / (np.linalg.norm(nrm) + 1e-6)
                P2.append(tuple(mid + nrm * rng.uniform(-0.06, 0.03) * rx))
        m = poly_mask(w, h, P2)
        ly, lx_ = np.mgrid[0:h, 0:w].astype(F32)
        ccx, ccy = cx - x0, cy - y0
        # facet split: a ridge line running from the crown down toward the sun-far side
        ang = rng.uniform(-0.35, 0.35)
        ridge = (lx_ - ccx) * math.sin(ang) - (ly - (ccy - 0.15 * ry)) * math.cos(ang)
        wob = fbm(w, h, max(w / (30 * sc), 3), sd + 5, 3) * 0.12 * ry
        top_f = ss(-4.0 * sc, 4.0 * sc, ridge + wob)          # 1 on the upper plane
        # the top plane splits in two: the half turned to the sun takes warm light, the far half the cool
        # sky; the face below the ridge line stays near-black, with a dim warm flank on the sun side
        sdir = 1.0 if sx > cx else -1.0
        cut2 = sdir * (lx_ - ccx) - rng.uniform(-0.1, 0.25) * rx + (ly - ccy) * rng.uniform(-0.4, 0.4)
        bev = ss(-1.5 * sc, 1.5 * sc, cut2 + wob) * top_f
        flank = ss(-1.5 * sc, 1.5 * sc, cut2 - 0.35 * rx + wob) * (1 - top_f)
        # a near-black silhouette (the same family as the crest it sits in) read by two FLAT plane breaks:
        # a barely lifted cool top plane (sky), and a darker face turned away from the sun; no bevel, no
        # outline - the only light is a rim on the sun-facing top edge (added below)
        dark = np.array([0.02, 0.016, 0.032], F32)
        topv = np.array([0.05, 0.042, 0.078], F32)            # sky-lit top plane
        awayv = np.array([0.013, 0.01, 0.022], F32)
        cut_a = -sdir * (lx_ - ccx) - rng.uniform(0.05, 0.3) * rx + (ly - ccy) * rng.uniform(-0.6, -0.2) + wob
        away_f = ss(-1.2 * sc, 1.2 * sc, cut_a) * (1 - top_f)
        col = dark + (topv - dark) * top_f[..., None]
        col = col + (awayv - col) * (away_f * 0.9)[..., None]
        # the flank turned to the sun: a warm backlit plane (reflected dawn light)
        col = col + (np.array([0.16, 0.075, 0.06], F32) - col) * (flank * 0.25 + bev * 0.2)[..., None]
        col = np.broadcast_to(col, (h, w, 3)).astype(F32)
        # cracks: dark fissures, with a hairline lit lip on their upper edge on the top plane
        Cm = np.zeros((h, w), F32)
        Lp = np.zeros((h, w), F32)
        for _ in range(int(rng.integers(0, 2))):
            px_ = ccx + rng.uniform(-0.7, 0.7) * rx
            py_ = ccy - ry * rng.uniform(0.2, 0.7)
            a_ = math.pi / 2 + rng.uniform(-0.9, 0.9)
            for _k in range(int(rng.integers(3, 8))):
                a_ += rng.uniform(-0.6, 0.6)
                ln = rng.uniform(0.06, 0.16) * ry * 2
                qx, qy = px_ + math.cos(a_) * ln, py_ + math.sin(a_) * ln
                wd = max(1, int(round(rng.uniform(1.0, 2.0) * sc)))
                cv2.line(Cm, (int(px_ * 16), int(py_ * 16)), (int(qx * 16), int(qy * 16)), 1.0, wd, cv2.LINE_AA, 4)
                cv2.line(Lp, (int((px_ + 1.0 * sc) * 16), int((py_ - 1.2 * sc) * 16)),
                         (int((qx + 1.0 * sc) * 16), int((qy - 1.2 * sc) * 16)), 1.0, 1, cv2.LINE_AA, 4)
                px_, py_ = qx, qy
        Cm *= m
        col = col * (1 - 0.5 * Cm[..., None])
        # lichen: sparse pale grey-green / ochre specks clustered on the upper plane
        Lc = np.zeros((h, w), F32)
        lk = fbm(w, h, max(w / (40 * sc), 3), sd + 9, 3)
        for _ in range(0):
            qx, qy = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
            if m[int(qy), int(qx)] < 0.9 or lk[int(qy), int(qx)] < 0.0:
                continue
            r_ = max(rng.uniform(0.6, 1.8) * sc, 0.6)
            cv2.ellipse(Lc, (int(qx * 16), int(qy * 16)), (int(r_ * 16 * 1.6), int(r_ * 16)), rng.uniform(0, 180),
                        0, 360, rng.uniform(0.5, 1.0), -1, cv2.LINE_AA, 4)
        Lc *= m
        lc_col = np.array([0.1, 0.105, 0.085], F32) * (0.6 + 0.4 * top_f[..., None])
        col = col + (lc_col - col) * (Lc * 0.7)[..., None]
        sub = S[y0:y1, x0:x1]
        # the lower part of the boulder sinks into the turf (no floating flat-bottomed lozenge)
        m = m * ss(ccy + 0.7 * ry + 3.0 * sc * wob / (0.12 * ry + 1e-6), ccy + 0.2 * ry, ly)
        over(sub, col.astype(F32), m)
        self.rock_all[y0:y1, x0:x1] = np.maximum(self.rock_all[y0:y1, x0:x1], m)
        self.own_rim_mask[y0:y1, x0:x1] = np.maximum(self.own_rim_mask[y0:y1, x0:x1], m)
        # rim: only the TOP edges turned toward the sun (normal . sun > 0 and normal pointing up)
        rs = rim_normal(m, sx - x0, sy - y0, 1.2 * sc, soft=0.3 * sc, macro=1.0 * sc, power=1.4, bias=0.2)
        mb_ = cv2.GaussianBlur(m, (0, 0), 2.0 * sc)
        gyn = cv2.Sobel(mb_, cv2.CV_32F, 0, 1, ksize=3)
        gxn = cv2.Sobel(mb_, cv2.CV_32F, 1, 0, ksize=3)
        nyn = gyn / (np.sqrt(gxn * gxn + gyn * gyn) + 1e-6)
        topedge = ss(0.55, 0.85, nyn) * ss(0.0, 0.05, gyn)     # only edges whose normal points UP
        rs = rs * topedge
        # lower silhouette rows are buried in grass anyway; fade the rim toward the base
        rs = rs * ss(ccy + 0.3 * ry, ccy - 0.2 * ry, ly)
        dsun = math.hypot(cx - sx, cy - sy) / self.W
        g = 0.6 + 0.8 * math.exp(-(dsun / 0.35) ** 2)
        self.extra_rim[y0:y1, x0:x1] += (rs * g)[..., None] * np.array([1.7, 1.02, 0.46], F32)

    def _steps(self, S):
        """Worn stepping stones leading up through the grass to the torii (perspective: small far, big near)."""
        sc, W, H = self.sc, self.W, self.H
        rng = np.random.default_rng(self.seed + 3)
        cx = self.ox + 0.255 * W
        g0 = float(self.gy[int(cx)])
        n = 7
        yy = np.arange(self.PH, dtype=F32)[:, None]
        self.stone_mask = np.zeros((self.PH, self.PW), F32)
        for i in range(n):
            k = i / (n - 1)
            y = g0 + 0.012 * H + (k ** 1.6) * 0.3 * H
            x = cx + (0.02 + 0.07 * k ** 1.3) * W + rng.uniform(-0.01, 0.01) * W
            rw = (0.018 + 0.07 * k ** 1.4) * W
            rh = rw * (0.13 + 0.12 * k)
            th = rh * 0.55
            nv = 9
            ang = np.linspace(0, 2 * math.pi, nv, endpoint=False) + rng.uniform(0, 0.5)
            rr = 1 + 0.12 * rng.uniform(-1, 1, nv)
            top = [(x + math.cos(a) * rw * r_, y + math.sin(a) * rh * r_) for a, r_ in zip(ang, rr)]
            side = [(p[0], p[1] + th) for p in top]
            x0b, x1b = int(max(x - rw * 1.4, 0)), int(min(x + rw * 1.4, self.PW))
            y0b, y1b = int(max(y - rh * 1.6, 0)), int(min(y + rh * 1.6 + th, self.PH))
            sub = S[y0b:y1b, x0b:x1b]
            ms = poly_mask(x1b - x0b, y1b - y0b, [(p[0] - x0b, p[1] - y0b) for p in side])
            over(sub, np.broadcast_to(SIL_D, sub[..., :3].shape).astype(F32), ms)
            mt = poly_mask(x1b - x0b, y1b - y0b, [(p[0] - x0b, p[1] - y0b) for p in top])
            self.stone_mask[y0b:y1b, x0b:x1b] = np.maximum(self.stone_mask[y0b:y1b, x0b:x1b], mt)
            nz = fbm(x1b - x0b, y1b - y0b, max((x1b - x0b) / (12 * sc), 3), 300 + i, 3)
            lit = SIL_L * (1.15 - 0.3 * k)
            ly = yy[y0b:y1b]
            col = lit * (0.85 + 0.2 * nz[..., None]) * (1 - 0.35 * ss(y - rh, y + rh, ly))[..., None]
            over(sub, col.astype(F32), mt)

    def _torii(self, S):
        """Weathered torii, hard against the sun: a near-black warm-maroon silhouette. Wood grain / rain
        streaks and worn patches barely lift the dark faces; the kasagi throws its shadow on the shimaki
        below it; a crisp 1-2 px rim only on the TOP faces (sky) and the SUN-FACING (right) faces of the
        kasagi, the nuki and the right pillar."""
        sc, W, H = self.sc, self.W, self.H
        sx, sy = self.sun
        cx = self.ox + 0.255 * W
        gyp = float(self.gy[int(cx)]) + 3 * sc
        Ht = 0.36 * H
        span = 0.62 * Ht
        pw = 0.075 * Ht
        top = gyp - Ht
        red_d = np.array([0.04, 0.012, 0.018], F32)
        red_m = np.array([0.07, 0.02, 0.025], F32)           # backlit vermilion lacquer (slightly desaturated front)
        black = np.array([0.016, 0.011, 0.022], F32)
        parts = []
        for s in (-1, 1):
            xb = cx + s * span / 2
            xt = xb - s * 0.035 * Ht
            y_t = top + 0.1 * Ht
            q = [(xb - pw / 2, gyp), (xt - pw * 0.45, y_t), (xt + pw * 0.45, y_t), (xb + pw / 2, gyp)]
            parts.append((q, red_m, 'v', 'pillarR' if s > 0 else 'pillarL'))
            q2 = [(xb - pw * 0.56, gyp), (xb - pw * 0.55, gyp - 0.08 * Ht), (xb + pw * 0.55, gyp - 0.08 * Ht),
                  (xb + pw * 0.56, gyp)]
            parts.append((q2, black, 'n', 'nemaki'))
            q3 = [(xb - pw * 0.85, gyp + 0.012 * Ht), (xb - pw * 0.75, gyp - 0.02 * Ht),
                  (xb + pw * 0.75, gyp - 0.02 * Ht), (xb + pw * 0.85, gyp + 0.012 * Ht)]
            parts.append((q3, SIL_D, 's', 'base'))
        yn = top + 0.3 * Ht
        tn = 0.055 * Ht
        ext = span / 2 + 0.16 * Ht
        parts.append(([(cx - ext, yn), (cx + ext, yn), (cx + ext, yn + tn), (cx - ext, yn + tn)], red_m, 'h', 'nuki'))
        parts.append(([(cx - 0.03 * Ht, top + 0.12 * Ht), (cx + 0.03 * Ht, top + 0.12 * Ht),
                       (cx + 0.03 * Ht, yn), (cx - 0.03 * Ht, yn)], red_d, 'v', 'gaku'))
        ys_ = top + 0.075 * Ht
        ts = 0.05 * Ht
        es = span / 2 + 0.2 * Ht
        parts.append(([(cx - es, ys_), (cx + es, ys_), (cx + es, ys_ + ts), (cx - es, ys_ + ts)], red_m, 'h',
                      'shimaki'))
        ek = span / 2 + 0.3 * Ht
        n = 48
        xs = np.linspace(-ek, ek, n)
        up = 0.07 * Ht * (np.abs(xs) / ek) ** 2.6
        yk = top + 0.01 * Ht - up
        tk = 0.07 * Ht
        upper = [(cx + x, y) for x, y in zip(xs, yk)]
        lower = [(cx + x, y + tk * (0.9 - 0.25 * (abs(x) / ek) ** 3)) for x, y in zip(xs[::-1], yk[::-1])]
        parts.append((upper + lower, red_m, 'k', 'kasagi'))
        cap = upper + [(cx + x, y + tk * 0.32) for x, y in zip(xs[::-1], yk[::-1])]
        parts.append((cap, black, 'c', 'kasagi'))
        kb = yk + tk * (0.9 - 0.25 * (np.abs(xs) / ek) ** 3)
        kas_bot = np.interp(np.arange(self.PW, dtype=F32), cx + xs, kb, left=-1e5, right=-1e5).astype(F32)[None, :]
        # wood grain: long streaks along each member (horizontal on beams, vertical on posts), rain streaks
        gh = fbm(self.PW, self.PH, max(self.PW / (5 * sc), 4), 74, 4, stretch=9.0)
        gv = fbm(self.PW, self.PH, max(self.PW / (5 * sc), 4), 71, 4, stretch=0.1)
        wr = fbm(self.PW, self.PH, max(self.PW / (3 * sc), 4), 72, 4, stretch=0.08)
        wp = fbm(self.PW, self.PH, max(self.PW / (14 * sc), 4), 73, 4)
        wood = np.array([0.05, 0.036, 0.05], F32)        # silvered bare wood where the lacquer is gone
        yy = np.arange(self.PH, dtype=F32)[:, None]
        TM = np.zeros((self.PH, self.PW), F32)
        masks = {}
        for q, c, kind, name in parts:
            m = poly_mask(self.PW, self.PH, q)
            ys = [p[1] for p in q]
            y0_, y1_ = min(ys), max(ys)
            g = gh if kind in ('h', 'k', 'c') else gv
            colr = c * (0.82 + 0.3 * ss(-0.4, 0.6, g)[..., None]) * (1 + 0.35 * ss(0.3, 0.9, wr)[..., None])
            if kind in ('v', 'h', 'k'):
                worn = ss(0.4, 0.62, wp + 0.3 * wr)[..., None]
                colr = colr + (wood - colr) * worn * 0.3
            if kind in ('v', 'n'):
                colr = colr * (1 - 0.4 * ss(y0_, y1_, yy))[..., None]
            elif kind == 'k':
                colr = colr * (1 - 0.55 * ss(y0_ + (y1_ - y0_) * 0.35, y1_, yy))[..., None]
            else:
                colr = colr * (1 - 0.45 * ss(y0_ + (y1_ - y0_) * 0.4, y1_, yy))[..., None]
            if name in ('shimaki', 'gaku'):
                # cast shadow of the kasagi: the member just below it is darker at the top
                csh = ss(kas_bot + 0.03 * Ht, kas_bot, yy)
                colr = colr * (1 - 0.5 * csh)[..., None]
            over(S, np.broadcast_to(colr, S[..., :3].shape).astype(F32), m)
            TM = np.maximum(TM, m)
            masks[name] = np.maximum(masks.get(name, 0), m)
        pwid, ph = 0.075 * Ht, 0.1 * Ht
        pq = [(cx - pwid / 2, top + 0.13 * Ht), (cx + pwid / 2, top + 0.13 * Ht),
              (cx + pwid / 2, top + 0.13 * Ht + ph), (cx - pwid / 2, top + 0.13 * Ht + ph)]
        pm = poly_mask(self.PW, self.PH, pq)
        over(S, np.broadcast_to(black, S[..., :3].shape), pm)
        tm = _text_img('雲見', ph * 0.36)
        T = np.zeros((self.PH, self.PW), F32)
        _stamp(T, tm, cx - tm.shape[1] / 2, top + 0.13 * Ht + ph * 0.08)
        over(S, np.broadcast_to(np.array([0.2, 0.15, 0.09], F32), S[..., :3].shape), T * pm * 0.8)
        TM = np.maximum(TM, pm)
        self.torii_mask = TM
        self.own_rim_mask = np.maximum(self.own_rim_mask, TM)
        # --- rims, per member: sun-facing faces + near-horizontal top faces (sky), crisp 1-2 px
        rc = np.array([1.8, 1.1, 0.5], F32)
        xx_t = np.arange(self.PW, dtype=F32)[None, :]
        xsun = 0.12 + 0.88 * ss(cx - 0.35 * span, cx + 0.75 * span, xx_t)
        R = np.zeros((self.PH, self.PW), F32)
        gain = dict(kasagi=1.0, nuki=1.0, pillarR=1.0, pillarL=0.9, nemaki=0.35, shimaki=0.8)
        for name, m in masks.items():
            gk = gain.get(name, 0.0)
            if gk <= 0:
                continue
            r_sun = rim_normal(m, sx, sy, 2.8 * sc, soft=0.4 * sc, macro=0.8 * sc, power=0.9, bias=0.25)
            r_sky = rim_normal(m, sx, sy, 0.8 * sc, soft=0.3 * sc, macro=0.8 * sc, power=2.0, bias=0.55, up=1.0)
            # the sky-lit top faces only catch real light toward the sun (right) end of each member
            R = np.maximum(R, gk * np.maximum(1.3 * r_sun, 0.6 * r_sky * xsun))
        # nothing lit in the kasagi's shadow (just below its underside)
        R = R * (1 - ss(kas_bot - 0.004 * H, kas_bot + 0.002 * H, yy) * ss(kas_bot + 0.035 * H, kas_bot + 0.02 * H, yy))
        self.extra_rim += (R * 1.7)[..., None] * np.array([2.0, 1.0, 0.38], F32)
        # a hot orange-vermilion glow band inside the sun-side edges and the top faces (lacquer lit through)
        V = np.zeros((self.PH, self.PW), F32)
        for name, m in masks.items():
            gk = gain.get(name, 0.0)
            if gk <= 0:
                continue
            v_sun = rim_normal(m, sx, sy, 9.0 * sc, soft=1.5 * sc, macro=1.5 * sc, power=0.8, bias=0.2)
            v_top = rim_normal(m, sx, sy, 4.0 * sc, soft=1.0 * sc, macro=1.5 * sc, power=1.5, bias=0.4, up=1.0)
            V = np.maximum(V, gk * np.maximum(v_sun, 0.35 * v_top * xsun))
        V = V * (1 - ss(kas_bot - 0.004 * H, kas_bot + 0.002 * H, yy) * ss(kas_bot + 0.035 * H, kas_bot + 0.02 * H, yy))
        self.extra_rim += (V * 0.9)[..., None] * np.array([1.1, 0.3, 0.08], F32)
        # warm bounce from the lit cloud sea on the down-facing faces (kasagi / shimaki / nuki undersides)
        B = np.zeros((self.PH, self.PW), F32)
        for name in ('kasagi', 'shimaki', 'nuki'):
            if name in masks:
                B = np.maximum(B, rim_normal(masks[name], sx, sy + 40.0 * H, 5.0 * sc, soft=1.2 * sc,
                                             macro=1.2 * sc, power=1.0, bias=0.3))
        self.extra_rim += (B * 0.05)[..., None] * np.array([0.9, 0.45, 0.3], F32)
        # faint sheen on the lacquer: grazing light on the upper faces (a hair lighter, warm)
        sheen = rim_normal(TM, sx, sy, 4.0 * sc, soft=1.5 * sc, macro=2.0 * sc, power=1.5, bias=0.3, up=0.6)
        self.extra_rim += (sheen * xsun * 0.03)[..., None] * np.array([0.9, 0.35, 0.2], F32)
        self.torii = dict(cx=cx, top=top, gy=gyp, span=span, pw=pw, Ht=Ht, yn=yn, tn=tn)

    def _shimenawa(self, S, D, Wt):
        """Twisted straw rope slung under the nuki + four zigzag shide papers. The rope is a dark body
        (twist grooves) with a warm specular only along its top; the papers are dim translucent paper."""
        T = self.torii
        sc = self.sc
        sx, sy = self.sun
        cx, Ht, span, pw = T['cx'], T['Ht'], T['span'], T['pw']
        y0 = T['yn'] + T['tn'] + 0.012 * Ht
        xl, xr = cx - span / 2 + pw * 0.5, cx + span / 2 - pw * 0.5
        sag = 0.05 * Ht
        th = 0.034 * Ht
        n = 60
        xs = np.linspace(xl, xr, n)
        ys = y0 + sag * (1 - ((xs - cx) / ((xr - xl) / 2)) ** 2)
        R = np.zeros((self.PH, self.PW), F32)
        Vt = np.zeros((self.PH, self.PW), F32)
        for i in range(n - 1):
            w_ = th * (0.75 + 0.25 * (1 - abs((xs[i] - cx) / ((xr - xl) / 2)) ** 2))
            cv2.line(R, (int(xs[i] * 16), int(ys[i] * 16)), (int(xs[i + 1] * 16), int(ys[i + 1] * 16)), 1.0,
                     max(1, int(w_)), cv2.LINE_AA, shift=4)
        for i in range(0, n - 1):
            x_, y_ = xs[i], ys[i]
            cv2.line(Vt, (int((x_ - th * 0.3) * 16), int((y_ - th * 0.45) * 16)),
                     (int((x_ + th * 0.3) * 16), int((y_ + th * 0.45) * 16)), 1.0, max(1, int(1.2 * sc)),
                     cv2.LINE_AA, shift=4)
        self.rope = R.copy()
        straw = np.array([0.05, 0.038, 0.03], F32)
        col = straw * (1 - 0.5 * Vt[..., None])
        over(S, np.broadcast_to(col, S[..., :3].shape).astype(F32), R)
        self.own_rim_mask = np.maximum(self.own_rim_mask, R)
        # warm specular on the top of the rope only (broken by the twist grooves)
        sp = rim_normal(R, sx, sy, 1.2 * sc, soft=0.3 * sc, macro=0.8 * sc, power=1.5, bias=0.3, up=0.6)
        sp = sp * (1 - 0.8 * Vt)
        self.extra_rim += (sp * 0.8)[..., None] * np.array([1.6, 1.0, 0.5], F32)
        P = np.zeros((self.PH, self.PW), F32)
        for k in range(4):
            fx = (k + 0.5) / 4
            x = xl + (xr - xl) * fx
            y = y0 + sag * (1 - ((x - cx) / ((xr - xl) / 2)) ** 2) + th * 0.3
            wz = 0.012 * Ht
            hz = 0.021 * Ht
            pts = []
            xo = x
            for j in range(4):
                sl = (1 if j % 2 == 0 else -1) * wz * 0.45
                yt, yb = y + j * hz * 0.92, y + (j + 1) * hz * 0.92 + 1
                pts.append([(xo - wz / 2, yt), (xo + wz / 2, yt), (xo + wz / 2 + sl, yb), (xo - wz / 2 + sl, yb)])
                xo = xo + sl + (1 if j % 2 == 0 else -1) * wz * 0.25
            for q in pts:
                cv2.fillPoly(P, [np.round(np.asarray(q) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
            cv2.circle(Wt, (int(x), int(y + 2 * hz)), int(3 * hz), 1.0, -1)
        # backlit paper: dim warm translucency, a lit edge on the sun side of each fold
        paper = np.array([0.2, 0.14, 0.12], F32)
        over(D, np.broadcast_to(paper, D[..., :3].shape).astype(F32), P)
        pe = rim_normal(P, sx, sy, 0.8 * sc, soft=0.3 * sc, macro=0.6 * sc, power=1.2, bias=0.3)
        self.paper_rim = (pe * 0.9)[..., None] * np.array([1.6, 1.05, 0.55], F32)

    def _signpost(self, S):
        sc, W, H = self.sc, self.W, self.H
        sx, sy = self.sun
        x = self.ox + 0.105 * W
        gyp = float(self.gy[int(x)]) + 4 * sc
        hp = 0.2 * H
        wp = 0.026 * W
        top = gyp - hp
        face = [(x - wp / 2, gyp), (x - wp / 2, top + wp * 0.3), (x, top), (x + wp / 2, top + wp * 0.3),
                (x + wp / 2, gyp)]
        side = [(x + wp / 2, gyp), (x + wp / 2, top + wp * 0.3), (x + wp * 0.72, top + wp * 0.36),
                (x + wp * 0.72, gyp - 2)]
        yy = np.arange(self.PH, dtype=F32)[:, None]
        wood = np.array([0.042, 0.034, 0.04], F32)
        n = fbm(self.PW, self.PH, max(self.PW / (3 * sc), 4), 91, 3, stretch=0.1)
        n2 = fbm(self.PW, self.PH, max(self.PW / (12 * sc), 4), 92, 3, stretch=0.2)
        col = wood * (0.8 + 0.3 * ss(-0.4, 0.6, n)[..., None]) * (1 + 0.25 * ss(0.3, 0.8, n2)[..., None])
        col = col * (1 - 0.4 * ss(top, gyp, yy))[..., None]
        m = poly_mask(self.PW, self.PH, face)
        over(S, col.astype(F32), m)
        ms = poly_mask(self.PW, self.PH, side)
        # the side plane faces the sun: a little warmer / lighter than the face
        over(S, (col * np.array([1.9, 1.5, 1.35], F32)).astype(F32), ms)
        # engraved inscription (real Japanese): the carved strokes hold a little sky light
        tm = _text_img('雲見岳山頂', wp * 0.62)
        T = np.zeros((self.PH, self.PW), F32)
        _stamp(T, tm, x - tm.shape[1] / 2, top + wp * 0.75)
        over(S, np.broadcast_to(np.array([0.1, 0.085, 0.11], F32), S[..., :3].shape), T * m * 0.9)
        t2 = _text_img('標高二四六八米', wp * 0.26)
        T2 = np.zeros((self.PH, self.PW), F32)
        _stamp(T2, t2, x + wp * 0.12, top + wp * 0.75 + tm.shape[0] + wp * 0.1)
        over(S, np.broadcast_to(np.array([0.085, 0.072, 0.09], F32), S[..., :3].shape), T2 * m * 0.8)
        U = np.clip(m + ms, 0, 1)
        self.own_rim_mask = np.maximum(self.own_rim_mask, U)
        r_sun = rim_normal(U, sx, sy, 1.0 * sc, soft=0.3 * sc, macro=0.8 * sc, power=1.2, bias=0.35)
        r_sky = rim_normal(U, sx, sy, 0.8 * sc, soft=0.3 * sc, macro=0.8 * sc, power=2.0, bias=0.5, up=1.0)
        self.extra_rim += (np.maximum(r_sun, 0.7 * r_sky) * 0.75)[..., None] * np.array([1.8, 1.1, 0.5], F32)

    def _pine(self, D, Wt):
        """Japanese black pine on the right knoll. Needle pads are cauliflower-edged silhouettes built from
        many small needle clumps (fans of needles around a dense core), with a scalloped top, a flatter
        underside, sky holes and a few stray tufts; one dark value mass, a faint cool lift on the crowns and
        a lost (soft) underside. Rim light comes from the global rim pass."""
        sc, W, H = self.sc, self.W, self.H
        rng = np.random.default_rng(self.seed + 55)
        PW, PH = self.PW, self.PH
        A = np.zeros((PH, PW), F32)
        sw = np.zeros((PH, PW), F32)
        bx = self.ox + 0.95 * W
        by = float(self.gy[min(int(bx), PW - 1)]) + 20 * sc

        def line(img, p0, p1, wd, val=1.0):
            cv2.line(img, tuple(np.round(np.asarray(p0) * 16).astype(int)), tuple(np.round(np.asarray(p1) * 16).astype(int)),
                     val, max(1, int(round(wd))), cv2.LINE_AA, shift=4)

        pos = np.array([bx, by], np.float64)
        ang = -math.pi / 2 - 0.42
        tp = [pos.copy()]
        n = 26
        seg = 1.0 * H / n
        for i in range(n):
            ang += rng.uniform(-0.14, 0.14) + (0.022 if i > 8 else 0.0) + 0.09 * math.sin(i * 0.9 + 1.3)
            pos = pos + np.array([math.cos(ang), math.sin(ang)]) * seg
            tp.append(pos.copy())
        tp = np.asarray(tp)
        for i in range(len(tp) - 1):
            u_t = i / len(tp)
            wdt = (30 + 52 * (1 - u_t) ** 1.8 + 30 * max(0.0, 1 - u_t * 6) ** 2) * sc
            line(A, tp[i], tp[i + 1], wdt)
            cv2.circle(A, (int(tp[i + 1][0] * 16), int(tp[i + 1][1] * 16)), int(wdt * 8), 1.0, -1, cv2.LINE_AA, 4)
        pads = []
        brs = []
        spec = [(9, -1, 0.17, 0.0), (12, -1, 0.21, -0.1), (15, -1, 0.15, -0.2), (17, 1, 0.07, 0.0),
                (19, -1, 0.19, -0.15), (22, -1, 0.12, -0.25), (24, 1, 0.08, -0.2), (6, -1, 0.07, 0.25),
                (14, -1, 0.11, 0.15), (21, -1, 0.15, 0.05)]
        sx, sy = self.sun
        spec2 = []
        for (i, side, blen, lift) in spec:
            # branches reaching toward the sun would lose their pads (kept clear of the sun): shorten them
            # so no bare stick points into the glare
            if side < 0 and tp[i][1] + 0.02 * W > sy - 0.12 * H:
                room = (tp[i][0] - (sx + 0.1 * W) - 0.06 * W) / W
                blen = float(np.clip(room, 0.05, blen))
            spec2.append((i, side, blen, lift))
        spec = spec2
        for (i, side, blen, lift) in spec:
            p = tp[i].copy()
            ang = (math.pi if side < 0 else 0.0) + lift * side * -1
            pts = [p.copy()]
            nseg = 8
            for j in range(nseg):
                ang += rng.uniform(-0.3, 0.3)
                p = p + np.array([math.cos(ang), math.sin(ang) * 0.7]) * blen * W / nseg
                pts.append(p.copy())
                if j in (3, 5) and rng.random() < 0.75:
                    pads.append((p.copy() + np.array([0, -6 * sc]), rng.uniform(0.04, 0.058) * W, len(brs)))
            pads.append((p.copy(), rng.uniform(0.062, 0.085) * W * (0.7 + 0.3 * blen / 0.2), len(brs)))
            # layered: a second, smaller pad tier stacked above / behind the end pad
            if rng.random() < 0.6:
                pads.append((p.copy() + np.array([rng.uniform(-0.3, 0.3) * 0.05 * W, -0.018 * W]),
                             rng.uniform(0.03, 0.042) * W, len(brs)))
            brs.append(np.asarray(pts))
            for j in range(len(pts) - 1):
                # thick where it leaves the trunk, tapering out toward the pad
                wdt = (3.0 + 24.0 * (1 - j / (len(pts) - 1)) ** 1.4) * sc * (0.55 + blen * 2.2)
                line(A, pts[j], pts[j + 1], wdt)
                # twigs forking up into the pad
                if j >= nseg - 4 and rng.random() < 0.8:
                    rng.uniform(-0.9, 0.9), rng.uniform(10, 26)      # (keeps the random stream; no bare twigs)
        sx, sy = self.sun
        pads = [(c, r, bi) for (c, r, bi) in pads if (c[0] - r) > sx + 0.1 * W or c[1] + r * 0.3 < sy - 0.12 * H]
        # ---- needle pads (2x supersampled): each pad is a HARD silhouette - a solid, flat-bottomed cloud of
        # overlapping lobes (the pruned 'cloud' pad of a black pine) whose upper / outer contour is bristled
        # with tidy fans of stiff needles (a needle-brush edge, not fur), a clean slightly ragged underside
        # with a few short drooping tufts, and one flat dark value inside.
        S2 = 2
        G = np.zeros((PH * S2, PW * S2), np.uint8)
        Tf = np.zeros((PH * S2, PW * S2), np.uint8)      # stray-tuft marker (more sway)
        TP = np.zeros((PH * S2, PW * S2), np.uint8)      # needle tips (backlit translucency)
        PM = np.zeros((PH, PW), np.uint8)
        NI = np.zeros((PH * S2, PW * S2), np.uint8)      # inner needle texture

        def nl(img, p0, p1, wd, val=255):
            cv2.line(img, tuple(np.round(np.asarray(p0) * S2 * 16).astype(int)),
                     tuple(np.round(np.asarray(p1) * S2 * 16).astype(int)), val, max(1, int(round(wd * S2))),
                     cv2.LINE_AA, shift=4)

        def fan(bx_, by_, a0, rt, nn, spread, img=G):
            """A fan of stiff needles from one base point around direction a0; near-equal lengths."""
            for q in range(nn):
                u_ = (q + rng.uniform(0.3, 0.7)) / nn - 0.5
                a = a0 + u_ * 2 * spread + rng.uniform(-0.05, 0.05)
                la = rt * rng.uniform(0.85, 1.08) * (1 - 0.35 * (2 * abs(u_)) ** 2)
                tip = (bx_ + math.cos(a) * la, by_ + math.sin(a) * la)
                # a stiff tapered needle (crisp spike, not a fuzzy line)
                wb = (0.8 * sc + 0.3) * 0.5
                px_, py_ = -math.sin(a) * wb, math.cos(a) * wb
                poly = np.array([(bx_ + px_, by_ + py_), tip, (bx_ - px_, by_ - py_)]) * S2
                cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
                if img is G:
                    nl(TP, (bx_ + math.cos(a) * la * 0.55, by_ + math.sin(a) * la * 0.55), tip, 0.7 * sc + 0.25)

        from lib import clouds3 as K_
        for (c, r, bi) in pads:
            # ---- a FLAT, WIDE, THIN slab (height : width ~ 1 : 5) with a hard, clean underside. Only the
            # top edge is modelled: 3-6 distinct cauliflower clumps (each crowned by a few rounded sub-lobes
            # of varied size) with a fine, crisp needle serration along the up-facing contour - no fur, no
            # halo, no drooping fringe.
            hw = r * rng.uniform(1.05, 1.2)
            th = 2.0 * hw / rng.uniform(4.6, 5.6)
            flat_y = c[1] + r * 0.08
            bx0, bx1 = int((c[0] - hw - 16 * sc) * S2), int((c[0] + hw + 16 * sc) * S2)
            by0, by1 = int((flat_y - th * 1.5 - 16 * sc) * S2), int((flat_y + 6 * sc) * S2)
            bx0, by0 = max(bx0, 0), max(by0, 0)
            bx1, by1 = min(bx1, PW * S2), min(by1, PH * S2)
            bw, bh = bx1 - bx0, by1 - by0
            if bw <= 8 or bh <= 8:
                continue
            Lc = np.zeros((bh, bw), np.uint8)

            def blob(img, x, y, ax, ay, ang=0.0):
                cv2.ellipse(img, (int(round((x * S2 - bx0) * 16)), int(round((y * S2 - by0) * 16))),
                            (max(int(ax * S2 * 16), 16), max(int(ay * S2 * 16), 16)), ang, 0, 360, 255, -1,
                            cv2.LINE_AA, 4)
            ncl = int(np.clip(round(2 * hw / (60 * sc)) + rng.integers(-1, 2), 3, 6))
            wts = rng.uniform(0.7, 1.3, ncl)
            edges_ = np.concatenate([[0.0], np.cumsum(wts)]) / wts.sum() * 2.0 - 1.0
            # shared flat base slab (the clumps sit on it)
            blob(Lc, c[0], flat_y - th * 0.2, hw, th * 0.32)
            crowns = []
            for ci in range(ncl):
                a_, b_ = edges_[ci], edges_[ci + 1]
                mid_ = 1.0 - abs((a_ + b_) / 2)            # middle clumps stand taller, the ends taper
                cxc = c[0] + (a_ + b_) / 2 * hw * 0.94
                hwc = (b_ - a_) / 2 * hw * rng.uniform(0.86, 1.0)
                hc_ = th * rng.uniform(0.7, 1.05) * (0.5 + 0.5 * mid_ ** 0.6)
                yc_ = flat_y - th * 0.3
                ay_ = max(hc_ - th * 0.3, th * 0.2)
                blob(Lc, cxc, yc_, hwc, ay_)
                # crown: 2-4 rounded sub-lobes of varied size along the dome's upper arc (cauliflower)
                nb_ = int(rng.integers(2, 5))
                for k_ in range(nb_):
                    th_ = math.pi * (0.15 + 0.7 * (k_ + rng.uniform(0.3, 0.7)) / nb_)
                    rb_ = hwc * rng.uniform(0.3, 0.46)
                    px_ = cxc + math.cos(th_) * hwc * 0.66
                    py_ = yc_ - math.sin(th_) * ay_ * 0.72
                    blob(Lc, px_, py_, rb_, rb_ * rng.uniform(0.72, 0.9))
                    crowns.append((px_, py_, rb_, rb_ * 0.8))
            # needle fans: every crown lobe's upper arc is bristled with short stiff needles radiating
            # from the lobe centre (a half-starburst per lobe), so each clump reads as a cauliflower tuft
            # of needles - grouped fans with notches between them, never a continuous fur
            Tl = np.zeros((bh, bw), np.uint8)
            for (qx_, qy_, qa_, qb_) in crowns:
                arc = math.pi * 0.5 * (qa_ + qb_)
                nbm = max(int(arc / (2.3 * sc)), 6)
                ln0 = rng.uniform(0.28, 0.4) * qa_ + 2.0 * sc
                for k_ in range(nbm):
                    t_ = math.pi * (0.06 + 0.88 * (k_ + rng.uniform(0.2, 0.8)) / nbm)
                    ex_, ey_ = qx_ + math.cos(t_) * qa_ * 0.8, qy_ - math.sin(t_) * qb_ * 0.8
                    a = -t_ + rng.uniform(-0.12, 0.12)
                    la = ln0 * rng.uniform(0.75, 1.1) * (0.55 + 0.45 * math.sin(t_))
                    wb = 0.8 * sc
                    pxn, pyn = -math.sin(a) * wb, math.cos(a) * wb
                    poly = np.array([(ex_ + pxn, ey_ + pyn), (ex_ + math.cos(a) * la, ey_ + math.sin(a) * la),
                                     (ex_ - pxn, ey_ - pyn)]) * S2 - np.array([bx0, by0])
                    cv2.fillPoly(Tl, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            Lf = cv2.GaussianBlur(Lc.astype(F32) / 255.0, (0, 0), 0.8 * S2 * sc)
            # up-facing contour weight (macro normal)
            Lm = cv2.GaussianBlur(Lf, (0, 0), 5.0 * S2 * sc)
            gyl = cv2.Sobel(Lm, cv2.CV_32F, 0, 1, ksize=3)
            gxl = cv2.Sobel(Lm, cv2.CV_32F, 1, 0, ksize=3)
            nyl = gyl / (np.sqrt(gxl * gxl + gyl * gyl) + 1e-6)          # +1 = edge faces up
            w_up = ss(0.0, 0.6, nyl)
            # fine needle serration: vertically elongated noise, a few px deep, only on the upper contour
            nz = K_._noise(bw, bh, max(bw / (2.6 * S2 * sc), 4), int(rng.integers(0, 1 << 20)), 2, stretch=0.22)
            nz2 = K_._noise(bw, bh, max(bw / (9.0 * S2 * sc), 4), int(rng.integers(0, 1 << 20)), 2, stretch=0.5)
            v = Lf + (0.06 * nz + 0.03 * nz2) * w_up
            L = np.maximum(ss(0.45, 0.55, v), Tl.astype(F32) / 255.0)
            # hard, clean underside: a near-straight cut (1 px waver), the ends tucking up a little
            xsl = (np.arange(bw, dtype=np.float64) + bx0 + 0.5) / S2
            uu_ = (xsl - c[0]) / hw
            fb = (flat_y + 0.8 * sc * frac1d(bw, int(rng.integers(0, 1 << 30)), 3, 5, 0.5)
                  - th * 0.5 * np.clip((np.abs(uu_) - 0.8) / 0.2, 0, 1) ** 2)
            yl = (np.arange(bh, dtype=np.float64)[:, None] + by0 + 0.5) / S2
            cut = np.clip((fb[None, :] - yl) * S2 + 0.5, 0, 1).astype(F32)
            L = (np.clip(L * cut, 0, 1) * 255).astype(np.uint8)
            G[by0:by1, bx0:bx1] = np.maximum(G[by0:by1, bx0:bx1], L)
            cv2.ellipse(PM, (int(c[0]), int(flat_y - th * 0.5)), (int(hw * 1.1), int(th * 0.9)), 0, 0, 360, 255, -1)
            cv2.ellipse(sw, (int(c[0]), int(flat_y - th * 0.5)), (int(hw * 1.1), int(th * 1.2)), 0, 0, 360, 1.0, -1)
        self.pine_pads = cv2.GaussianBlur(PM.astype(F32) / 255.0, (0, 0), 6 * sc)
        self._pine_A = A
        N = cv2.resize(G.astype(F32) / 255.0, (PW, PH), interpolation=cv2.INTER_AREA)
        Tn = cv2.resize(Tf.astype(F32) / 255.0, (PW, PH), interpolation=cv2.INTER_AREA)
        # ---- values: flat painted silhouettes. Pads: one flat near-black mass with a clean flat underside.
        # Trunk / limbs: a flat dark value with a couple of painted bark bands. Light: a hot orange rim only
        # on the edges facing the sun (left / upper-left), 2-4 px hot, melting into a warm glow over ~20 px.
        Nc = np.clip(N, 0, 1)
        Ab = np.clip(A, 0, 1)
        nc_ = np.broadcast_to(np.array([0.016, 0.014, 0.032], F32), (PH, PW, 3)).astype(F32)
        # a barely lifted cool top plane on the flat crowns (sky light), hard-edged, never on the underside
        upl = rim_normal(Nc, sx, sy, 7.0 * sc, soft=0.0, macro=6.0 * sc, power=1.0, bias=0.4, up=1.0)
        nc_ = nc_ + np.array([0.012, 0.012, 0.028], F32) * ss(0.3, 0.45, upl)[..., None]
        # bark: flat dark silhouette + a few broad, slightly lighter / darker bark bands (painted, no stipple)
        bark = np.array([0.03, 0.024, 0.036], F32)
        bnd = fbm(PW, PH, max(PW / (60 * sc), 4), 60, 2, stretch=6.0, angle=-12)
        bnd2 = fbm(PW, PH, max(PW / (25 * sc), 4), 61, 2, stretch=4.0, angle=-12)
        bv = ss(0.12, 0.2, bnd + 0.25 * bnd2) * 0.35 - ss(0.12, 0.2, -bnd - 0.25 * bnd2) * 0.3
        bc = bark * (1 + bv[..., None])
        # black-pine bark: plates broken by dark cross fissures and a few long splits (so the trunk reads as
        # painted bark, not a smooth rendered tube); each plate's upper lip catches a little warm light
        rb = np.random.default_rng(self.seed + 77)
        # black-pine bark, painted: long vertical plate strokes running along the trunk (a hair lighter or
        # darker than the body) and only a few short horizontal fissures - no tiled cross-hatch
        Bk = np.zeros((PH, PW), F32)
        Bp = np.zeros((PH, PW), F32)
        for i in range(len(tp) - 1):
            u_t = i / len(tp)
            wdt = (30 + 52 * (1 - u_t) ** 1.8 + 30 * max(0.0, 1 - u_t * 6) ** 2) * sc
            d_ = tp[i + 1] - tp[i]
            L_ = float(np.hypot(*d_)) + 1e-6
            t_ = d_ / L_
            nr_ = np.array([-t_[1], t_[0]])
            for k_ in range(int(rb.integers(3, 6))):
                off = rb.uniform(-0.42, 0.42) * wdt
                c0 = tp[i] + d_ * rb.uniform(-0.3, 1.0)
                ln_ = rb.uniform(0.6, 1.6) * L_
                wv = rb.uniform(1.4, 3.6) * sc
                img_ = Bp if rb.random() < 0.55 else Bk
                line(img_, c0 + nr_ * off, c0 + nr_ * (off + rb.uniform(-2, 2) * sc) + t_ * ln_, wv,
                     rb.uniform(0.5, 1.0))
            if rb.random() < 0.45:
                off = rb.uniform(-0.35, 0.3) * wdt
                c0 = tp[i] + d_ * rb.uniform(0, 1)
                half = rb.uniform(0.08, 0.2) * wdt
                line(Bk, c0 + nr_ * (off - half), c0 + nr_ * (off + half) + t_ * rb.uniform(-2, 2) * sc,
                     rb.uniform(1.2, 2.0) * sc)
        Bk = cv2.GaussianBlur(np.clip(Bk, 0, 1), (0, 0), 0.6 * sc)
        Bp = cv2.GaussianBlur(np.clip(Bp, 0, 1), (0, 0), 0.6 * sc) * (1 - Bk)
        bc = bark * (1 + 0.5 * bv[..., None])
        bc = bc * (1 - 0.45 * Bk[..., None]) + (Bp * 0.5)[..., None] * np.array([0.03, 0.022, 0.03], F32)
        # ---- rim (sun-facing edges only): outward normal from the blurred silhouette, light direction =
        # toward the sun, bent upward (pads above the sun light on their left / upper-left, never below)
        yy_, xx_ = np.mgrid[0:PH, 0:PW].astype(F32)
        tx_, ty_ = sx - xx_, sy - yy_
        tn_ = np.sqrt(tx_ * tx_ + ty_ * ty_) + 1e-3
        lx_, ly_ = tx_ / tn_ - 0.5, ty_ / tn_ - 0.75
        ln_ = np.sqrt(lx_ * lx_ + ly_ * ly_) + 1e-6
        lx_, ly_ = lx_ / ln_, ly_ / ln_
        dsun = tn_ / W
        prox = 0.55 + 0.45 * np.exp(-(dsun / 0.35) ** 2)

        def facing(M, sig):
            Mb = cv2.GaussianBlur(M, (0, 0), sig)
            gx = cv2.Sobel(Mb, cv2.CV_32F, 1, 0, ksize=3)
            gy_ = cv2.Sobel(Mb, cv2.CV_32F, 0, 1, ksize=3)
            gn = np.sqrt(gx * gx + gy_ * gy_) + 1e-6
            nx, ny = -gx / gn, -gy_ / gn
            f_ = np.clip(nx * lx_ + ny * ly_, 0, 1)
            return f_ * ss(0.35, -0.05, ny) * ss(0.0, 0.02, gn)

        # pads: normals at two scales (hot hairline follows the needle fringe, the warm glow the pad mass)
        Npad = np.clip(Nc, 0, 1)
        dpad = cv2.distanceTransform((Npad > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
        f_hot = facing(Npad, 2.0 * sc)
        f_warm = facing(Npad, 10.0 * sc)
        # hot 2-4 px line on the sun-facing (upper-left) contour, melting into the near-black body over
        # ~10-20 px (a warm mid-value glow that decays; no halo outside the silhouette)
        hotp = ss(0.25, 0.7, f_hot) * ss(0.2, 0.6, f_warm) * np.exp(-dpad / (1.3 * sc))
        warmp = ss(0.35, 0.85, f_warm) * np.exp(-dpad / (6.5 * sc)) * ss(0.0, 2.0 * sc, dpad)
        # trunk / limbs: a single warm lit edge on the sun (left) side: 3-6 px hot, falling off over ~25 px
        dtr = cv2.distanceTransform((Ab > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
        f_tr = facing(Ab, 3.0 * sc)
        f_trm = facing(Ab, 12.0 * sc)
        hott = ss(0.3, 0.75, f_tr) * np.exp(-dtr / (2.2 * sc))
        warmt = ss(0.2, 0.8, f_trm) * np.exp(-dtr / (9.0 * sc)) * ss(0.0, 2.0 * sc, dtr)
        # the lit bark plates near the lit edge catch a little of the warm light (painted plate strokes)
        warmt = warmt * (0.75 + 0.9 * Bp) * (1 - 0.5 * Bk)
        # the pads cover the limbs: only the visible (uncovered) part of each keeps its light
        vis_t = Ab * (1 - Nc)
        rim_hot = (hotp * Nc + hott * vis_t) * prox
        rim_warm = (warmp * Nc + 0.8 * warmt * vis_t) * prox
        self.pine_rim = (rim_hot[..., None] * np.array([1.9, 0.92, 0.34], F32) * 1.25 +
                         rim_warm[..., None] * np.array([0.34, 0.12, 0.04], F32)).astype(F32)
        self.extra_rim += self.pine_rim
        self.pine_own = cv2.dilate(np.clip(Nc + Ab, 0, 1), np.ones((3, 3), np.uint8))
        self.pad_allow = np.ones((PH, PW), F32)
        self.pine_fringe = np.zeros((PH, PW), F32)
        over(D, bc.astype(F32), Ab)
        over(D, nc_.astype(F32), Nc)
        yy = np.arange(PH, dtype=F32)[:, None]
        hfac = ss(by, by - 0.8 * H, yy)
        Wt[:] = np.maximum(Wt, cv2.GaussianBlur(sw, (0, 0), 8 * sc) * 1.25 * hfac + np.clip(A, 0, 1) * hfac * 0.2)
        Wt[:] = np.maximum(Wt, np.clip(Tn * 1.4, 0, 1) * hfac * 1.3)

    def _pebbles(self, S):
        """Small stones scattered in the turf: dark, a cool sky lift on the top, and a tiny warm specular
        glint on the sun-facing top edge (additive, swells with the light)."""
        sc, W, H, PW, PH = self.sc, self.W, self.H, self.PW, self.PH
        rng = np.random.default_rng(self.seed + 21)
        Mk = np.zeros((PH, PW), F32)
        Tp = np.zeros((PH, PW), F32)
        G = np.zeros((PH, PW), F32)
        for _ in range(30):
            x = rng.uniform(0, PW - 1)
            g0 = float(self.gy[int(x)])
            k = rng.uniform(0, 1) ** 1.3
            y = g0 + 0.015 * H + k * (PH - g0 - 0.015 * H)
            if y >= PH - 2:
                continue
            r = (2.5 + 9 * k) * sc * rng.uniform(0.7, 1.4)
            ry = r * rng.uniform(0.45, 0.65)
            ang = rng.uniform(-15, 15)
            cv2.ellipse(Mk, (int(x * 16), int(y * 16)), (int(r * 16), int(ry * 16)), ang, 0, 360, 1.0, -1,
                        cv2.LINE_AA, 4)
            cv2.ellipse(Tp, (int(x * 16), int((y - ry * 0.35) * 16)), (int(r * 0.8 * 16), int(ry * 0.55 * 16)),
                        ang, 0, 360, 1.0, -1, cv2.LINE_AA, 4)
            if rng.random() < 0.15:
                gx = x + r * 0.45
                gy_ = y - ry * 0.7
                cv2.circle(G, (int(gx * 16), int(gy_ * 16)), max(int(1.1 * sc * 16 * (0.7 + k)), 8), 1.0, -1,
                           cv2.LINE_AA, 4)
        Tp *= Mk
        col = SIL_D + (SIL_L * 1.35 - SIL_D) * (Tp * 0.8)[..., None]
        over(S, col.astype(F32), Mk)
        G = cv2.GaussianBlur(G, (0, 0), 0.6 * sc) * Mk
        gl = G[..., None] * np.array([1.6, 1.15, 0.7], F32) + \
            cv2.GaussianBlur(G, (0, 0), 2.5 * sc)[..., None] * np.array([0.5, 0.3, 0.15], F32)
        self.extra_rim += gl.astype(F32)

    def _grass(self, D, Wt, gy):
        """Grass as READABLE CLUMPS painted in depth bands (back -> front). Along the crest a dense row of
        tufts with a few tall blades crossing the cloud sea; down the slope, distinct fan-shaped clumps
        that grow with nearness (perspective) with dark ground planes showing between them. Each clump is a
        near-black silhouette with a crisp backlit tip on the sun side; dew glints sparkle along the far
        edge; the nearest band is slightly defocused (depth of field)."""
        sc = self.sc
        PW, PH, H = self.PW, self.PH, self.H
        sx, sy = self.sun
        rng = np.random.default_rng(self.seed + 77)
        ssf = 2
        w2, h2 = PW * ssf, PH * ssf
        edges = [0.0, 0.05, 0.16, 0.34, 0.6, 10.0]
        nb_ = len(edges) - 1
        Gs = [np.zeros((h2, w2), np.uint8) for _ in range(nb_)]
        Vs = [np.zeros((h2, w2), np.uint8) for _ in range(nb_)]
        DW = np.zeros((h2, w2), np.uint8)             # dew glint points
        clump = frac1d(PW, self.seed + 9, 5, 24, 0.6)
        roots = []                                    # (y, x, L, n, spread, width, tall)
        # crest row: dense small tufts + a few tall blades against the clouds
        x = 0.0
        while x < PW - 1:
            x += rng.uniform(4, 11) * sc
            xi = int(min(x, PW - 1))
            if self.stone_mask[int(min(gy[xi] + 3 * sc, PH - 1)), xi] > 0.3:
                continue
            cl = 0.5 + 0.5 * clump[xi]
            tall = rng.random() < 0.07
            L = rng.uniform(55, 115) * sc if tall else rng.uniform(12, 34) * sc * (0.6 + 0.8 * cl)
            roots.append((gy[xi] + rng.uniform(0, 5) * sc, x, L, int(rng.integers(2, 5)) if tall else
                          int(rng.integers(6, 12)), 0.35 if tall else 0.8, 1.3 * sc, tall))
        # slope: rows of distinct clumps, spacing and size growing toward the camera
        dn = 0.035
        while True:
            yrow_off = dn * 0.25 * H
            if np.min(gy) + yrow_off > PH + 60 * sc:
                break
            spx = (24 + 120 * dn) * sc
            x = rng.uniform(0, spx)
            while x < PW - 1:
                xi = int(min(x, PW - 1))
                y = gy[xi] + yrow_off + rng.uniform(-0.3, 0.3) * (8 + 50 * dn) * sc
                if rng.random() < 0.9 and y < PH + 80 * sc:
                    if not (self.stone_mask[int(np.clip(y, 0, PH - 1)), xi] > 0.3 and rng.random() < 0.95):
                        L = (18 + 120 * min(dn, 1.2)) * sc * rng.uniform(0.7, 1.3)
                        roots.append((y, x, L, int(rng.integers(14, 26)), rng.uniform(0.35, 0.6),
                                      (1.8 + 4.5 * min(dn, 1.2)) * sc, False))
                x += spx * rng.uniform(0.6, 1.4)
            dn += (12 + 55 * dn) * sc / (0.25 * H)
        # tufts that bury the rock bases (they overlap the stone)
        for (cx, cy, rx, ry, sd) in self.rocks:
            nn_ = int(10 * rx / (0.04 * self.W)) + 4
            for k in range(nn_):
                x = cx + (-1.05 + 2.1 * (k + rng.uniform(0.2, 0.8)) / nn_) * rx
                ys_ = cy + ry * rng.uniform(0.25, 0.6) * (1 - 0.4 * ((x - cx) / rx) ** 2)
                if 0 <= x < PW - 1 and 0 <= ys_ < PH - 1:
                    roots.append((ys_, x, ry * rng.uniform(0.35, 0.7), int(rng.integers(8, 14)), 0.8, 1.6 * sc, False))
        roots.sort(key=lambda r_: r_[0])
        lean0 = -0.12                                   # the wind leans everything a touch to the left
        for (y, x, L0, nb, spr, wd0, tall) in roots:
            xi = int(np.clip(x, 0, PW - 1))
            dn = (y - gy[xi]) / (0.25 * H)
            bi = int(np.searchsorted(edges, dn, side='right') - 1)
            bi = min(max(bi, 0), nb_ - 1)
            G, V = Gs[bi], Vs[bi]
            lean = lean0 + rng.normal(0, 0.08)
            tone = rng.uniform(0.7, 1.0)
            for j in range(nb):
                L = L0 * rng.uniform(0.55, 1.1) * (1 - 0.35 * abs(j / max(nb - 1, 1) - 0.5) * 2)
                spread = (j / max(nb - 1, 1) - 0.5) * 2 * spr + rng.normal(0, 0.06)
                a = -math.pi / 2 + lean + spread * 0.75
                curv = spread * rng.uniform(0.3, 0.8) + rng.normal(0.0, 0.07) + lean * 0.5
                k = 6
                p = np.array([x + rng.uniform(-1.5, 1.5) * sc + spread * wd0, y])
                pts = [p.copy()]
                for i in range(k):
                    a += curv / k
                    p = p + np.array([math.cos(a), math.sin(a)]) * L / k
                    pts.append(p.copy())
                wd = wd0 * rng.uniform(0.8, 1.25) * ssf
                for i in range(k):
                    p0 = np.round(pts[i] * ssf * 16).astype(int)
                    p1 = np.round(pts[i + 1] * ssf * 16).astype(int)
                    th = max(1, int(round(wd * (1 - i / k * 0.8))))
                    cv2.line(G, tuple(p0), tuple(p1), 255, th, cv2.LINE_AA, shift=4)
                    cv2.line(V, tuple(p0), tuple(p1), int(255 * tone * (i + 1) / k), th, cv2.LINE_8, shift=4)
                if bi == 0 and rng.random() < 0.035:
                    q = np.round(pts[-2] * ssf + (pts[-1] - pts[-2]) * ssf * 0.6).astype(int)
                    cv2.circle(DW, (int(q[0]), int(q[1])), max(1, int(round(0.9 * sc * ssf))), 255, -1, cv2.LINE_AA)
        yy = np.arange(PH, dtype=F32)[:, None]
        din = np.clip((yy - gy[None, :]) / (0.25 * self.H), 0, 1)
        dx = (np.arange(PW, dtype=F32)[None, :] - sx) / self.W
        sunw = np.exp(-(dx / 0.3) ** 2)
        base = np.array([0.022, 0.02, 0.034], F32)
        tipc = np.array([0.075, 0.068, 0.11], F32)          # cool sky fill on the upper blade
        rimc = np.array([1.5, 0.9, 0.38], F32)
        Wacc = np.zeros((PH, PW), F32)
        for bi in range(nb_):
            a = cv2.resize(Gs[bi].astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
            if a.max() <= 0:
                continue
            v = cv2.resize(Vs[bi].astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
            v = np.clip(v / np.maximum(a, 1e-3), 0, 1)
            col = base + (tipc - base) * (v ** 1.5)[..., None] * (0.6 + 0.4 * np.exp(-din / 0.3))[..., None]
            col = col * (1 - 0.3 * din[..., None])
            # translucent tips along the crest (light through the blades)
            tr = v ** 2 * np.exp(-din / 0.05) * (0.25 + 0.75 * sunw)
            col = col + np.array([0.7, 0.45, 0.14], F32) * tr[..., None] * 0.35
            # crisp backlit tip on the sun-facing side of each blade
            r = rim_normal(a, sx, sy, 1.0 * sc * (1 + bi * 0.4), soft=0.3 * sc, macro=0.7 * sc, power=1.2,
                           bias=0.1)
            amt = (0.35 + 0.65 * sunw) * (ss(0.35, 0.85, v) if bi < 2 else 0.5 * ss(0.6, 0.95, v)) * (1.0 - 0.35 * din)
            rim_add = (r * amt)[..., None] * rimc
            if bi == nb_ - 1:
                # nearest band: slight defocus (depth of field) - soften alpha, colour and its rim together
                sg = 2.2 * sc
                a = cv2.GaussianBlur(a, (0, 0), sg)
                col = cv2.GaussianBlur(col.astype(F32), (0, 0), sg)
                rim_add = cv2.GaussianBlur(rim_add.astype(F32), (0, 0), sg) * 0.7
            elif bi == nb_ - 2:
                sg = 0.8 * sc
                a = cv2.GaussianBlur(a, (0, 0), sg)
                rim_add = cv2.GaussianBlur(rim_add.astype(F32), (0, 0), sg)
            self.extra_rim *= (1 - a[..., None])
            self.extra_rim += rim_add
            over(D, col.astype(F32), a)
            Wacc = np.maximum(Wacc, v * a)
        # dew glints: tiny hot points near the tips along the far edge, a soft sparkle halo
        dw = cv2.resize(DW.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        dw = dw * (0.35 + 0.65 * sunw)
        self.extra_rim += dw[..., None] * np.array([2.2, 1.8, 1.3], F32) + \
            cv2.GaussianBlur(dw, (0, 0), 2.0 * sc)[..., None] * np.array([1.2, 0.8, 0.45], F32)
        # flowers: small clusters of pale lilac / cream dots in the turf, a few catching a warm glint
        Fm = np.zeros((PH * ssf, PW * ssf), np.uint8)
        Fw = np.zeros((PH * ssf, PW * ssf), np.uint8)
        for _ in range(int(18 * sc) + 8):
            x = rng.uniform(0, PW - 1)
            k = rng.uniform(0, 0.7) ** 1.4
            y0 = gy[int(x)] + 4 * sc + k * (PH - gy[int(x)])
            n = int(rng.integers(2, 7))
            warm = rng.random() < 0.3
            for _j in range(n):
                fx = x + rng.normal(0, 8 * sc * (1 + 2 * k))
                fy = y0 + rng.normal(0, 3 * sc * (1 + 2 * k))
                r = max(1, int(round(rng.uniform(0.8, 1.5) * sc * ssf * (1 + 1.0 * k))))
                cv2.circle(Fw if warm else Fm, (int(fx * ssf), int(fy * ssf)), r, 255, -1, cv2.LINE_AA)
        fm = cv2.resize(Fm.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        fw = cv2.resize(Fw.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        over(D, np.broadcast_to(np.array([0.16, 0.13, 0.22], F32), D[..., :3].shape), fm * 0.9)
        over(D, np.broadcast_to(np.array([0.2, 0.16, 0.13], F32), D[..., :3].shape), fw * 0.9)
        fa = np.clip(fm + fw, 0, 1)
        self.extra_rim *= (1 - fa[..., None])
        fr = rim_normal(fa, sx, sy, 1.0 * sc, soft=0.3 * sc, macro=0.5 * sc, power=1.0, bias=0.1)
        self.extra_rim += (fr * (0.1 + 0.25 * sunw))[..., None] * np.array([1.3, 0.9, 0.5], F32)
        Wt[:] = np.maximum(Wt, Wacc * 1.0)

    def _fringe(self, D, Wt, gy, S, mcol, mm):
        """Grass as a painted silhouette fringe: hand-placed clumps of tapered blades along the crest line
        and along the nearer mound, in runs with bare gaps between them (the ridge line reads cleanly in
        the gaps). Clumps are the same near-black as the plane they grow from; only the tips facing the
        sun glow (translucency) and the global rim pass lines their sun-facing edges."""
        sc, W, H, PW, PH = self.sc, self.W, self.H, self.PW, self.PH
        sx, sy = self.sun
        rng = np.random.default_rng(self.seed + 177)
        ssf = 3
        GB = np.zeros((PH * ssf, PW * ssf), np.uint8)     # crest clumps
        GF = np.zeros((PH * ssf, PW * ssf), np.uint8)     # mound clumps
        TB = np.zeros((PH * ssf, PW * ssf), np.uint8)     # blade tips (translucency)
        GR = np.zeros((PH * ssf, PW * ssf), np.uint8)     # tufts burying the rock bases
        WW = np.zeros((PH, PW), F32)

        def blade(img, x, y, L, ang, curv, w0, tip_img=None):
            k = 7
            pts = []
            a = ang
            p = np.array([x, y], np.float64)
            for i in range(k + 1):
                pts.append(p.copy())
                a += curv / k
                p = p + np.array([math.cos(a), math.sin(a)]) * L / k
            pts = np.asarray(pts)
            tang = np.gradient(pts, axis=0)
            tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
            nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
            u = np.linspace(0, 1, k + 1)[:, None]
            wd = w0 * (1 - u) ** 0.9 + 0.15
            left = pts + nrm * wd / 2
            right = pts - nrm * wd / 2
            poly = np.concatenate([left, right[::-1]], 0) * ssf
            cv2.fillPoly(img, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            if tip_img is not None:
                j = int(k * 0.55)
                pl = np.concatenate([left[j:], right[j:][::-1]], 0) * ssf
                cv2.fillPoly(tip_img, [np.round(pl * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
            for i in range(k):
                cv2.line(WW, (int(pts[i][0]), int(pts[i][1])), (int(pts[i + 1][0]), int(pts[i + 1][1])),
                         float((i + 1) / k) ** 1.5, max(1, int(2 * sc)))

        def clump(img, x, y, hgt, nb, tip_img=None, spread=0.55):
            lean = -0.18 + rng.normal(0, 0.08)            # wind leans everything a touch to the left
            for j in range(nb):
                u = (j + rng.uniform(0.2, 0.8)) / nb - 0.5
                L = hgt * rng.uniform(0.55, 1.0) * (1 - 0.55 * abs(u) * 2) ** 0.7
                ang = -math.pi / 2 + lean + u * 2 * spread + rng.normal(0, 0.05)
                curv = u * rng.uniform(0.4, 1.0) + lean * 0.8 + rng.normal(0, 0.1)
                w0 = max(hgt * rng.uniform(0.045, 0.075), 1.3 * sc)
                blade(img, x + u * hgt * 0.25, y + rng.uniform(0, 0.05) * hgt, L, ang, curv, w0, tip_img)

        def row(img, line, hmin, hmax, dens, tip_img=None):
            x = rng.uniform(-40, 0) * sc
            while x < PW + 40 * sc:
                # a run of clumps, then a bare gap
                run = int(rng.integers(8, 22))
                for _ in range(run):
                    xi = int(np.clip(x, 0, PW - 1))
                    hg = rng.uniform(hmin, hmax) * (1.6 if rng.random() < 0.08 else 1.0)
                    clump(img, x, float(line[xi]) + hg * 0.08, hg, int(rng.integers(5, 13)), tip_img)
                    x += rng.uniform(0.18, 0.5) * hmax * dens
                x += rng.uniform(0.05, 0.6) * hmax

        # crest plane fringe (smaller: it is further away), mound fringe (bigger, nearer)
        tor = self.torii
        row(GB, gy, 12 * sc, 48 * sc, 0.8, TB)
        # tufts burying the torii / rock bases
        for bx in (tor['cx'] - tor['span'] / 2, tor['cx'] + tor['span'] / 2):
            for _ in range(3):
                x = bx + rng.uniform(-0.03, 0.03) * W
                clump(GB, x, float(gy[int(np.clip(x, 0, PW - 1))]) + 3 * sc, rng.uniform(18, 34) * sc, 9, TB)
        for (cx, cy, rx, ry, sd) in self.rocks:
            for _ in range(int(3 + rx / (0.012 * W))):
                x = cx + rng.uniform(-1.0, 1.0) * rx
                yb = cy + ry * rng.uniform(0.2, 0.5) * (1 - 0.5 * ((x - cx) / rx) ** 2)
                clump(GR, x, yb, rng.uniform(0.3, 0.6) * ry, int(rng.integers(6, 11)))
        # the nearer mound: its own row of silhouette clumps (a step in front of the crest plane)
        TF = np.zeros_like(TB)
        row(GF, self.gy2, 16 * sc, 44 * sc, 1.0, TF)
        ab = cv2.resize(GB.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        af = cv2.resize(GF.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        tb = cv2.resize(TB.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        yy = np.arange(PH, dtype=F32)[:, None]
        dx = (np.arange(PW, dtype=F32)[None, :] - sx) / W
        sunw = np.exp(-(dx / 0.25) ** 2)
        # crest clumps: the crest's lifted silhouette value; tips glow (light through the blades) near the sun
        cb = np.broadcast_to(np.array([0.1, 0.07, 0.12], F32), (PH, PW, 3)).copy()
        cb = cb * (1 - 0.25 * ss(gy[None, :] - 10 * sc, gy[None, :] + 30 * sc, yy))[..., None]
        # the turf blades near the sun column catch the warm backlight through their bodies
        cb = cb + (np.array([0.24, 0.11, 0.07], F32) - cb) * (0.45 * np.exp(-(dx / 0.35) ** 2))[..., None]
        cb = cb + np.array([0.42, 0.24, 0.07], F32) * (tb * (0.1 + 0.9 * sunw) * 0.55)[..., None]
        over(D, cb.astype(F32), ab)
        ar = cv2.resize(GR.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        over(D, np.broadcast_to(np.array([0.03, 0.025, 0.042], F32), D[..., :3].shape).astype(F32), ar)
        # the mound itself (static) and its clumps (sway) share the near-black value
        over(S, np.broadcast_to(mcol, S[..., :3].shape).astype(F32), mm)
        self.mound_all = np.clip(mm + af, 0, 1)
        tf = cv2.resize(TF.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        mc2 = mcol + np.array([0.05, 0.045, 0.09], F32) * tf[..., None] * 0.2     # sky-lit blade tips (subdued)
        over(D, mc2.astype(F32), af)
        # a cool sky line on the mound's top edge (it lies in the crest's shadow: no warm rim)
        mr = rim_normal(np.clip(mm + af, 0, 1), sx, sy - 2.0 * H, 1.0 * sc, soft=0.3 * sc, macro=1.5 * sc,
                        power=2.0, bias=0.5)
        self.extra_rim += (mr * 0.2)[..., None] * np.array([0.35, 0.33, 0.6], F32)
        Wt[:] = np.maximum(Wt, cv2.GaussianBlur(WW, (0, 0), 1.5 * sc) * 1.1)

    def _susuki(self, D, Wt, gy):
        """Tall susuki (pampas) stalks rising from the crest and breaking its silhouette against the cloud
        sea: thin arching stems, a drooping feathery plume that glows when backlit. Strong sway weight."""
        sc, W, H, PW, PH = self.sc, self.W, self.H, self.PW, self.PH
        sx, sy = self.sun
        rng = np.random.default_rng(self.seed + 91)
        ssf = 2
        Sm = np.zeros((PH * ssf, PW * ssf), np.uint8)
        Pm = np.zeros((PH * ssf, PW * ssf), np.uint8)
        Wm = np.zeros((PH, PW), F32)

        def ln(img, p0, p1, wd, val=255):
            cv2.line(img, tuple(np.round(np.asarray(p0) * ssf * 16).astype(int)),
                     tuple(np.round(np.asarray(p1) * ssf * 16).astype(int)), val, max(1, int(round(wd * ssf))),
                     cv2.LINE_AA, shift=4)

        # hand-placed, uneven: a thick stand, a lone stalk, a pair, a gap ...
        groups = [(0.395, 4, 0.016), (0.452, 1, 0.0), (0.64, 3, 0.008), (0.705, 1, 0.0), (0.915, 2, 0.006),
                  (0.968, 1, 0.0), (0.27, 1, 0.0), (0.06, 2, 0.01), (0.19, 2, 0.008), (0.55, 2, 0.01),
                  (0.8, 3, 0.012), (0.86, 1, 0.0)]
        for (gx, n, spr) in groups:
            for _ in range(n):
                x = self.ox + (gx + rng.normal(0, spr)) * W
                xi = int(np.clip(x, 0, PW - 1))
                y = float(gy[xi]) + rng.uniform(2, 10) * sc
                hS = rng.uniform(0.06, 0.16) * H * rng.choice([0.6, 1.0, 1.0])
                lean = rng.uniform(-0.35, -0.05)          # the wind leans them left, away from the sun
                K_ = 14
                us = np.linspace(0, 1, K_ + 1)
                pts = np.stack([x + hS * lean * us ** 2 * 0.6, y - hS * us * (1 - 0.1 * abs(lean) * us)], -1)
                for i in range(K_):
                    ln(Sm, pts[i], pts[i + 1], (1.6 - 0.8 * i / K_) * sc)
                    cv2.line(Wm, tuple(int(v) for v in np.round(pts[i])), tuple(int(v) for v in np.round(pts[i + 1])),
                             float(0.4 + 2.6 * (i / K_) ** 1.5), max(1, int(3 * sc)))
                # plume (panicle): 7-11 silky branches from the top of the stem, each rising along the stem,
                # then arcing out downwind (left) and drooping, fringed with fine hairs
                ph_ = hS * rng.uniform(0.24, 0.34)
                nbr = int(rng.integers(7, 12))
                for _k in range(nbr):
                    u = _k / max(nbr - 1, 1)
                    i0 = int(round(K_ * (1 - 0.28 * u)))
                    b0 = pts[max(min(i0, K_), 0)].copy()
                    ang = -math.pi / 2 + lean * 0.8 - rng.uniform(0.05, 0.45)
                    Lb = ph_ * rng.uniform(0.55, 1.0) * (1 - 0.35 * u)
                    p_ = b0.copy()
                    for st in range(8):
                        q_ = p_ + np.array([math.cos(ang), math.sin(ang)]) * Lb / 8
                        ln(Pm, p_, q_, 0.8 * sc)
                        for sgn in (-1, 1):
                            ha = ang + sgn * rng.uniform(0.3, 0.8) + 0.4
                            hl = rng.uniform(2.5, 5.5) * sc * (1 - 0.4 * st / 8)
                            ln(Pm, q_, q_ + np.array([math.cos(ha), math.sin(ha)]) * hl, 0.55 * sc)
                        cv2.line(Wm, tuple(int(v) for v in np.round(p_)), tuple(int(v) for v in np.round(q_)),
                                 3.0, max(1, int(5 * sc)))
                        p_ = q_
                        ang += -0.2 * (1 if lean < -0.1 else 0.6) + 0.0
                        ang += 0.18 * (st > 3)                      # droop at the end
                ln(Pm, pts[-1], pts[-1] + np.array([lean * 6 * sc, -3 * sc]), 1.0 * sc)
        sm = cv2.resize(Sm.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        pm = cv2.resize(Pm.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        dx = (np.arange(PW, dtype=F32)[None, :] - sx) / W
        sunw = np.exp(-(dx / 0.3) ** 2)
        over(D, np.broadcast_to(np.array([0.045, 0.038, 0.06], F32), D[..., :3].shape), sm)
        over(D, np.broadcast_to(np.array([0.32, 0.22, 0.2], F32), D[..., :3].shape), pm * 0.9)
        self.extra_rim *= (1 - np.clip(sm + pm, 0, 1)[..., None])
        # backlit plumes glow through (translucent), strongest toward the sun
        glow = cv2.GaussianBlur(pm, (0, 0), 0.8 * sc) * (0.45 + 0.55 * sunw)
        self.extra_rim += glow[..., None] * np.array([1.1, 0.72, 0.38], F32) * 0.7
        self.extra_rim += (rim_normal(sm, sx, sy, 1.0 * sc, soft=0.3 * sc, macro=0.6 * sc, power=1.0, bias=0.1) * (0.5 + 0.5 * sunw))[..., None] * \
            np.array([1.4, 0.85, 0.4], F32)
        Wt[:] = np.maximum(Wt, cv2.GaussianBlur(Wm, (0, 0), 2 * sc))


# ------------------------------------------------------------------------------------------ near cloud band
NB_RAMP = dict(stops=[(0.0, (0.03, 0.022, 0.12)), (0.3, (0.07, 0.05, 0.2)), (0.5, (0.14, 0.09, 0.3)),
                      (0.7, (0.3, 0.17, 0.4)), (0.85, (0.62, 0.34, 0.46)), (1.0, (0.95, 0.6, 0.5))],
               warm=[(0.0, (0.07, 0.04, 0.2)), (0.4, (0.24, 0.12, 0.36)), (0.7, (0.66, 0.34, 0.44)),
                     (1.0, (1.1, 0.7, 0.5))],
               rim=(2.0, 1.25, 0.62), haze=(0.4, 0.26, 0.58), glow=(1.4, 0.75, 0.45))


def near_band_plate(w, h, rows, sun, sc, seed=5, x_range=None, crown_col=(0.3, 0.18, 0.36), crown_amt=0.55,
                    lower_col=(0.1, 0.075, 0.26)):
    """The nearest cloud banks (just below the summit lip): rows of cumulus heads with cauliflower
    silhouettes (clouds3.HeadSet), backlit by a low sun beyond them.
    - rim: each head is lit from ITS OWN screen direction to the sun, restricted to up-facing contours, so
      the hot lining sits only on the tops and sun-facing flanks and fades to nothing on the far side and
      the underside (no outline all round);
    - value: 2-3 flat painted planes - a warm pink-mauve upper plane hanging under the lit crest, a cool
      violet lower plane, deep indigo toward the base;
    - bases: wide lost edges, dissolving into the indigo valley below instead of outlined scallops.
    rows: [(base_y, radius, haze)] back -> front, plate px."""
    from lib import clouds3 as K
    rng = np.random.default_rng(seed)
    hs = K.HeadSet()
    x0_, x1_ = x_range if x_range is not None else (-0.05 * w, 1.05 * w)
    for (yb, r0, hz) in rows:
        x = x0_ - r0 * rng.uniform(0.5, 1.5)
        while x < x1_ + r0:
            r = r0 * rng.uniform(0.55, 1.45)
            cx = x + r
            yb_ = yb + r0 * rng.uniform(-0.25, 0.25)
            dxs, dys = sun[0] - cx, sun[1] - (yb_ - r * 0.25)
            dn = math.hypot(dxs, dys) + 1e-6
            ang = math.atan2(dys, dxs)
            # per-head light: screen direction to the sun, strongly backlit (-z)
            lv = np.array([dxs / dn, dys / dn, -0.35])
            lv = lv / np.linalg.norm(lv)
            hs.add(cx, yb_ - r * 0.25, r * rng.uniform(1.3, 1.9), r * rng.uniform(0.55, 0.85), rng,
                   bump=(0.07, 0.22), arc=(-195, 15), sub=0.6, min_px=1.2 * sc, sun_ang=ang, sun_bias=0.6,
                   clump=0.6, base_y=yb_ + r * 0.35, base_soft=30 * sc, haze=hz, rim=1.1, rim_px=1.0 * sc,
                   glow=0.45, soft=24 * sc, aa=0.7, warm=1.0, light=tuple(lv), beta=0.08)
            x += r * rng.uniform(1.5, 2.3)
    K._DEBUG = {}
    try:
        P = K.paint_heads(w, h, hs, sun_dir=(0.1, -1.0), sun_z=-0.3, ramp=NB_RAMP, lost_up=0.0, floor=0.25,
                          poster=0.85, step_soft=0.02, seed=seed, rim_up=0.55, rim_sil=False, crust_mix=0.0)
        Fd = K._DEBUG.get('fields')
    finally:
        K._DEBUG = None
    # close tiny alpha pockets between cauliflower bumps (they read as dark bead chains)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(9 * sc) | 1, int(9 * sc) | 1))
    a0 = P[..., 3].copy()
    ac = cv2.morphologyEx(a0, cv2.MORPH_CLOSE, ker)
    gain = np.clip(ac - a0, 0, 1)
    if gain.max() > 0:
        cc_ = cv2.morphologyEx(np.ascontiguousarray(P[..., :3]), cv2.MORPH_CLOSE, ker)
        P[..., :3] = P[..., :3] * (1 - gain[..., None]) + cc_ * 0.85 * gain[..., None]
        P[..., 3] = ac
    if Fd is None or not crown_amt:
        return P
    # value planes: a warm pink-mauve UPPER plane hanging under each head's lit crest (the lining field
    # smeared downward ~30 px, then cut with a hard, wobbly painted border), cool violet LOWER plane
    fx0, fy0 = Fd['x0'], Fd['y0']
    A_ = cv2.morphologyEx(Fd['A'].astype(F32), cv2.MORPH_CLOSE, ker)
    gyA = cv2.Sobel(cv2.GaussianBlur(A_, (0, 0), 2.0 * sc), cv2.CV_32F, 0, 1, ksize=3)
    gxA = cv2.Sobel(cv2.GaussianBlur(A_, (0, 0), 2.0 * sc), cv2.CV_32F, 1, 0, ksize=3)
    upness = np.clip(gyA / (np.sqrt(gxA * gxA + gyA * gyA) + 1e-4), 0, 1)       # 1 on a flat top edge
    Rm = (np.clip(Fd['RIM'], 0, 1) * ss(0.5, 0.85, upness)).astype(F32)
    hh, ww = Rm.shape
    n = max(int(24 * sc), 4)
    acc = np.zeros_like(Rm)
    for k in range(0, n, 2):
        sh = np.zeros_like(Rm)
        sh[k:] = Rm[:hh - k]
        acc = np.maximum(acc, sh * (1 - 0.7 * k / n))
    acc = cv2.GaussianBlur(acc, (0, 0), sigmaX=7 * sc, sigmaY=2.5 * sc) * 1.4
    wob = fbm(ww, hh, max(ww / (40 * sc), 3), seed + 3, 3, stretch=2.0) * 0.08 +         fbm(ww, hh, max(ww / (160 * sc), 2), seed + 4, 2, stretch=1.5) * 0.22
    plane = ss(0.14, 0.26, acc + wob) * np.clip(A_, 0, 1)
    xs = (np.arange(fx0, fx0 + ww, dtype=F32)[None, :] - sun[0]) / w
    focus = np.exp(-(xs / 0.28) ** 2)
    cc = np.asarray(crown_col, F32) * (1 - focus[..., None]) + np.array([0.62, 0.34, 0.42], F32) * focus[..., None]
    sub = P[fy0:fy0 + hh, fx0:fx0 + ww]
    # fill the little dark pits between the cauliflower bumps inside the lit plane (no bead chains)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(7 * sc) | 1, int(7 * sc) | 1))
    closed = cv2.morphologyEx(np.ascontiguousarray(sub[..., :3]), cv2.MORPH_CLOSE, ker)
    pz = (plane > 0.05)[..., None] | (acc > 0.1)[..., None]
    sub[..., :3] = np.where(pz, np.maximum(sub[..., :3], closed * 0.9), sub[..., :3])
    prot = ss(0.8, 1.3, sub[..., :3].max(-1))[..., None]          # keep the painted linings
    k_ = (plane * crown_amt)[..., None] * (1 - prot)
    lo = ((1 - plane) * np.clip(A_, 0, 1) * 0.35)[..., None] * (1 - prot)
    sub[..., :3] = sub[..., :3] * (1 - lo) + np.asarray(lower_col, F32) * lo
    sub[..., :3] = sub[..., :3] * (1 - k_) + cc * k_
    return P


# ------------------------------------------------------------------------------------ far peak silhouettes
def far_peak_plate(w, h, x0, x1, base_y, height, peaks, seed, sun, sc, col=(0.3, 0.28, 0.55),
                   lit=(0.42, 0.34, 0.58), haze_col=(0.72, 0.52, 0.66), rim_col=(1.9, 1.15, 0.6), rim_amt=1.0,
                   fade=(0.35, 0.9), rough=0.18, aerial=0.0, face_blur=2.0, face_fall=(0.2, 0.9), crest=None):
    """A distant mountain standing in the cloud sea, painted as a hazy blue-violet SILHOUETTE (yn_01):
    one flat value, only two broad planes (the flank turned toward the sun a touch lighter / warmer,
    split along the main ridge line), a thin warm rim on the sun-facing ridgelines, and a soft lost base
    that melts into the cloud-top haze. No facets, no snow, no noise texture."""
    n = int(x1 - x0)
    prof = _ridge_profile(n, seed, peaks, rough=rough, expo=1.15, round_=0.08)
    top = base_y - prof * height
    xs = np.arange(n) + x0
    if crest is not None:
        # cloud-capped crest: cauliflower scallops along the ridge (rounded bumps, sharp notches)
        rng_ = np.random.default_rng(seed + 500)
        ph_, u_ = 0.0, np.zeros(n)
        per = rng_.uniform(24, 46) * sc
        amp = np.zeros(n)
        a_ = rng_.uniform(3.0, 7.0) * sc
        for i in range(n):
            ph_ += math.pi / per
            if ph_ >= math.pi:
                ph_ -= math.pi
                per = rng_.uniform(22, 48) * sc
                a_ = rng_.uniform(3.0, 8.0) * sc
            u_[i] = ph_
            amp[i] = a_
        # 2-3 big lobe groupings (broad cumulus domes with deep notches between them); the small
        # scallops grow on the dome crowns and shrink in the notches (no even sawtooth)
        ii = np.arange(n, dtype=np.float64)
        ng = int(rng_.integers(3, 5))
        cen = np.sort(rng_.uniform(0.12, 0.92, ng)) * n
        dome = np.zeros(n)
        for gc in cen:
            wd = n * rng_.uniform(0.13, 0.22)
            dome = np.maximum(dome, rng_.uniform(0.6, 1.0) * np.sqrt(np.clip(1 - ((ii - gc) / wd) ** 2, 0, 1)))
        dome = dome * (0.3 + 0.7 * np.clip(prof / max(prof.max(), 1e-6) * 1.6, 0, 1))
        top = top - dome * 0.2 * height
        # medium cumulus lobes (50-130 px, rounded with sharp notches) carry the small scallops
        ph3, per3, a3 = 0.0, rng_.uniform(50, 130) * sc, rng_.uniform(8, 20) * sc
        lob = np.zeros(n)
        for i in range(n):
            ph3 += math.pi / per3
            if ph3 >= math.pi:
                ph3 -= math.pi
                per3 = rng_.uniform(50, 130) * sc
                a3 = rng_.uniform(8, 20) * sc
            lob[i] = a3 * math.sin(ph3) ** 0.6
        top = top - lob * (0.5 + 0.5 * dome)
        top = top - amp * (0.3 + 0.5 * dome) * np.sqrt(np.abs(np.sin(u_)))
    pts = [(x0, base_y + 4)] + list(zip(xs, top)) + [(x1, base_y + 4)]
    m = poly_mask(w, h, pts)
    yy, xx = np.mgrid[0:h, 0:w].astype(F32)
    ttop = np.interp(xx[0], xs, top, left=base_y, right=base_y).astype(F32)[None, :]
    # depth below the ridge measured from a SMOOTHED ridge line (the ridge's small jags must not print
    # down the slope as vertical streaks in the haze / plane gradients)
    ttop_s = cv2.GaussianBlur(ttop, (0, 0), 0.02 * w)
    ttop = np.minimum(ttop, ttop_s) * 0.0 + ttop_s
    din = np.clip((yy - ttop) / max(height, 1), 0, 1.5)
    # the lit flank: the smoothed profile's slope facing the sun, carried down the fall line
    prw = np.interp(np.arange(w, dtype=F32), xs, prof, left=0, right=0).astype(F32)
    pb = cv2.GaussianBlur(prw[None], (0, 0), 0.012 * w)[0]
    g = np.gradient(pb)
    sdir = 1.0 if sun[0] > (x0 + x1) / 2 else -1.0
    face = ss(0.0, 0.0015, -g * sdir)[None, :] * np.ones((h, 1), F32)
    # the plane border wanders a little (painted, not ruled)
    face = cv2.GaussianBlur(face.astype(F32), (0, 0), face_blur * sc)
    base = np.asarray(col, F32)
    c = base + (np.asarray(lit, F32) - base) * (face * 0.8 * (1 - ss(face_fall[0], face_fall[1], din)))[..., None]
    # aerial perspective: lower slopes vanish into the cloud-top haze
    hz = ss(fade[0], fade[1], din)[..., None]
    c = c * (1 - hz * 0.85) + np.asarray(haze_col, F32) * hz * 0.85
    if aerial:
        # aerial perspective: the whole silhouette sinks toward the horizon haze, most at its base
        ak = (aerial * (0.55 + 0.45 * ss(0.0, 0.8, din)))[..., None]
        c = c * (1 - ak) + np.asarray(haze_col, F32) * ak
    if crest is not None:
        # a crisp lit crest plane hanging under the scalloped ridge (a flat warm-lilac cap with a hard,
        # scalloped terminator) + a hot hairline on the crest; light from up / toward the sun
        traw = np.interp(xx[0], xs, top, left=base_y, right=base_y).astype(F32)[None, :]
        dc = yy - traw
        tv0 = np.gradient(cv2.GaussianBlur(traw, (0, 0), 3.0 * sc)[0])[None, :]
        nz = fbm(w, h, max(w / (60 * sc), 3), seed + 501, 3, stretch=2.0)
        ph2 = np.interp(xx[0], xs, np.abs(np.sin(u_)), left=0, right=0).astype(F32)[None, :]
        bw = (crest.get('band', 16.0) * sc) * (0.55 + 0.45 * ph2 + 0.35 * nz) * (0.5 + 0.5 / np.sqrt(1.0 + 4.0 * tv0 * tv0))
        tv = np.gradient(cv2.GaussianBlur(traw, (0, 0), 3.0 * sc)[0])[None, :]
        # up- and sun-facing: flat tops and the slope turned to the sun (not the far flank)
        fac = np.clip(1.0 / np.sqrt(1.0 + tv * tv) * 0.7 + (-tv * sdir) * 0.6, 0, 1)
        plane_ = ss(bw + 1.2 * sc, bw - 0.8 * sc, dc) * ss(-1.0, 0.5, dc) * ss(0.25, 0.7, fac)
        c = c + (np.asarray(crest.get('lit', (0.76, 0.54, 0.62)), F32) - c) * (plane_ * 0.0)[..., None]
        # a WARM-WHITE lit crest band 6-15 px wide (wider on the lobe crowns), hot at the edge and melting
        # softly down into the body (light, not a hairline). Distance to the silhouette (2D, no column
        # drips) and the macro normal decide where it lands: up / sun-facing contours only.
        dist = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
        mb = cv2.GaussianBlur(m, (0, 0), 5.0 * sc)
        gx_ = cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3)
        gy_ = cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3)
        gn_ = np.sqrt(gx_ * gx_ + gy_ * gy_) + 1e-6
        lx_, ly_ = 0.55 * sdir, -1.0
        ln_ = math.hypot(lx_, ly_)
        fcn = np.clip((-gx_ * lx_ - gy_ * ly_) / gn_ / ln_, 0, 1)
        fcn = cv2.GaussianBlur(fcn * ss(0.0, 0.01, gn_), (0, 0), 3.0 * sc)
        fcn = np.clip(fcn / max(float(fcn.max()), 1e-3) * 1.6, 0, 1)
        nz2 = fbm(w, h, max(w / (14 * sc), 3), seed + 502, 2)
        bwh = (6.0 + 9.0 * np.clip(0.5 + 0.5 * nz, 0, 1)) * sc
        kb_ = ss(bwh * 1.1, bwh * 0.3, dist + 1.5 * sc * nz2) * ss(0.25, 0.7, fcn)
        kh_ = np.exp(-np.maximum(dist - 0.5, 0) / (2.0 * sc)) * ss(0.4, 0.85, fcn)
        c = c + (np.asarray(crest.get('white', (1.05, 0.86, 0.8)), F32) - c) * (kb_ * 0.6)[..., None]
        c = c + (np.asarray(crest.get('hot', (1.35, 1.1, 0.9)), F32) - c) * (kh_ * 0.55)[..., None]
    a = m * (1 - ss(fade[0] + 0.25, fade[1] + 0.35, din))
    rim = rim_light(m, sun[0], sun[1], 1.4 * sc, soft=0.45 * sc)
    rim = rim * (0.35 + 0.65 * face) * (1 - ss(0.0, 0.5, din))
    c = c + np.asarray(rim_col, F32) * (rim * rim_amt)[..., None]
    out = np.dstack([c, a]).astype(F32)
    return bleed(out, 3.0)


def torn_wisps_plate(w, h, specs, sun, sc, seed=0, lit=(1.15, 0.66, 0.52), body=(0.2, 0.12, 0.38),
                     shade=(0.07, 0.04, 0.22)):
    """A few small torn cloud fragments (stretched ragged noise blobs) drifting in front of the near cloud
    banks: crisp warm-lit upper edge, violet body, lower edge lost. specs: [(cx, cy, len, thick)] plate px."""
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w, 4), F32)
    for k, (cx, cy, ln, th) in enumerate(specs):
        x0, x1 = int(max(cx - ln, 0)), int(min(cx + ln, w))
        y0, y1 = int(max(cy - th * 3, 0)), int(min(cy + th * 3, h))
        if x1 <= x0 or y1 <= y0:
            continue
        bw, bh = x1 - x0, y1 - y0
        yy, xx = np.mgrid[0:bh, 0:bw].astype(F32)
        u = (xx + x0 - cx) / ln
        v = (yy + y0 - cy) / th
        n1 = fbm(bw, bh, max(bw / (40 * sc), 3), seed + 11 * k, 5, stretch=4.0, angle=-3)
        n2 = fbm(bw, bh, max(bw / (12 * sc), 3), seed + 11 * k + 5, 4, stretch=2.5, angle=-3)
        # a lens-shaped core (flat underside, humped top) torn up by noise
        env = 1 - u * u - (np.where(v < 0, v * v * 0.8, v * v * 2.2)) - 0.18 * u * v
        d = env + 1.0 * n1 + 0.45 * n2
        a = ss(0.1, 0.35, d) * ss(-1.2, -0.3, -np.abs(u))
        a = cv2.GaussianBlur(a.astype(F32), (0, 0), 0.8 * sc)
        ab = cv2.GaussianBlur(a, (0, 0), 3 * sc)
        a1 = cv2.GaussianBlur(a, (0, 0), 1.0 * sc)
        up = np.clip(a1 - np.roll(a1, int(2.5 * sc) + 1, axis=0), 0, 1)
        dn = np.clip(ab - np.roll(ab, -int(6 * sc) - 1, axis=0), 0, 1)
        col = np.ones((bh, bw, 1), F32) * np.asarray(body, F32)
        col = col + (np.asarray(shade, F32) - col) * ss(0.0, 0.3, dn + 0.5 * ss(-0.3, 0.8, v))[..., None]
        col = col + (np.asarray(lit, F32) - col) * ss(0.1, 0.4, up)[..., None]
        # soft lost underside
        a = a * (1 - 0.6 * ss(0.2, 1.0, v))
        sub = out[y0:y1, x0:x1]
        over(sub, col.astype(F32), a * 0.95)
    return bleed(out, 3.0)
