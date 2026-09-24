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


def bleed(rgba, sigma):
    from lib import clouds3 as K
    rgba[..., :3] = K._bleed(rgba[..., :3], rgba[..., 3], sigma)
    return rgba


# ------------------------------------------------------------------------------------------------ sky
SKY_STOPS = [(0.0, (1.00, 0.95, 0.80)), (0.025, (1.00, 0.83, 0.55)), (0.07, (0.98, 0.66, 0.50)),
             (0.15, (0.88, 0.56, 0.60)), (0.27, (0.62, 0.50, 0.72)), (0.45, (0.36, 0.40, 0.72)),
             (0.7, (0.18, 0.26, 0.60)), (1.0, (0.07, 0.12, 0.38))]


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
    col = col * (1 - 0.35 * near * low) + np.array([1.25, 0.9, 0.55], F32) * 0.35 * near * low
    away = (1 - near) * low
    col = col * (1 - 0.12 * away) + np.array([0.75, 0.55, 0.78], F32) * 0.12 * away
    # subtle painted variation (large soft value shifts, no noise speckle)
    n = fbm(w, h, 3.0, seed, 3, stretch=3.0)
    col = col * (1 + 0.025 * n[..., None])
    return col.astype(F32)


# ------------------------------------------------------------------------------------------ mountains
def _ridge_profile(n, seed, peaks, rough=0.3):
    """peaks: list of (x0..1, height0..1, half_width) -> profile height 0..1 over n samples."""
    x = np.linspace(0, 1, n)
    prof = np.zeros(n)
    rng = np.random.default_rng(seed)
    for (px, ph, hw) in peaks:
        d = (x - px) / hw
        d = np.where(d > 0, d * rng.uniform(0.8, 1.25), -d * rng.uniform(0.8, 1.25))
        # concave flanks (steeper near the summit) with shoulders
        p = ph * np.clip(1 - d, 0, 1) ** 1.5
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
    R = pr * height + 0.02 * height * ridged * (0.3 + depth_in) + 0.004 * height * s2
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
def ridge_plate(w, h, pts_x, pts_y, base_y, seed, sun, sc, body, top_col, rim_col, haze_col, haze):
    """Forested ridge: smooth ridge line (interpolated control points) crowned by conifer spires."""
    rng = np.random.default_rng(seed)
    xs = np.arange(w, dtype=F32)
    top = np.interp(xs, pts_x, pts_y) + 6 * sc * frac1d(w, seed, 5, 12, 0.5)
    m = np.zeros((h, w), np.uint8)
    poly = [(0, base_y)] + [(float(x), float(y)) for x, y in zip(xs[::2], top[::2])] + [(w - 1, base_y)]
    p = np.round(np.asarray(poly) * 16).astype(np.int32)
    cv2.fillPoly(m, [p], 255, cv2.LINE_AA, shift=4)
    # conifer spires along the crest (+ a few below it)
    x = 0.0
    while x < w:
        x += rng.uniform(1.2, 3.5) * sc
        yi = int(np.clip(x, 0, w - 1))
        if top[yi] >= base_y - 2:
            continue
        hh = rng.uniform(5, 13) * sc * (0.6 + 0.9 * rng.random() ** 2)
        ww = hh * rng.uniform(0.3, 0.45)
        yb = top[yi] + rng.uniform(2, 10) * sc
        tri = [(x - ww, yb), (x, yb - hh), (x + ww, yb)]
        # ragged tiers
        k = int(rng.integers(3, 6))
        spine = []
        for i in range(k + 1):
            fy = i / k
            spine.append((x - ww * fy * rng.uniform(0.85, 1.1), yb - hh + hh * fy))
        pts_ = [(x, yb - hh * 1.05)] + spine[1:] + [(x + ww * rng.uniform(0.85, 1.1), yb)] + \
               [(x + ww * (i / k) * rng.uniform(0.85, 1.1), yb - hh + hh * i / k) for i in range(k, 0, -1)]
        pp = np.round(np.asarray(pts_) * 16).astype(np.int32)
        cv2.fillPoly(m, [pp], 255, cv2.LINE_AA, shift=4)
    a = m.astype(F32) / 255.0
    yy = np.arange(h, dtype=F32)[:, None]
    tt = np.interp(xs, xs, top)[None, :]
    din = np.clip((yy - tt) / (base_y - tt + 1e-3), 0, 1)
    n = fbm(w, h, max(w / (18 * sc), 4), seed + 2, 4)
    col = np.asarray(top_col, F32) * (1 - din[..., None]) + np.asarray(body, F32) * din[..., None]
    col = col * (1 + 0.12 * n[..., None])
    # mist rising from the clouds on the lower slopes
    mist = ss(0.25, 0.9, din + 0.2 * n)[..., None]
    hz = np.clip(haze + 0.75 * mist, 0, 1)
    col = col * (1 - hz) + np.asarray(haze_col, F32) * hz
    rim = rim_light(a, sun[0], sun[1], 1.6 * sc, soft=0.5)
    col = col + np.asarray(rim_col, F32) * rim[..., None]
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
        # summit plateau: high on the left (torii), falling toward centre-right, a rock knoll on the right
        kx = [-0.2, 0.0, 0.12, 0.3, 0.42, 0.52, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
        ky = [0.70, 0.705, 0.715, 0.73, 0.76, 0.81, 0.855, 0.875, 0.845, 0.80, 0.775, 0.76]
        base = np.interp(fx, kx, ky)
        g = self.oy + base * self.H + 7 * sc * frac1d(PW, self.seed + 1, 6, 10, 0.55) + \
            2.5 * sc * frac1d(PW, self.seed + 2, 4, 160, 0.5)
        self.gy = g.astype(F32)
        return self.gy

    # ------------------------------------------------------------------------------------------ rocks
    def rock_shapes(self):
        """(cx, cy, rx, ry, seed) boulders sitting on the ground line."""
        R = []
        spec = [(0.035, 0.05, 0.045), (0.14, 0.03, 0.03), (0.37, 0.028, 0.026), (0.47, 0.04, 0.032),
                (0.87, 0.045, 0.045), (0.975, 0.06, 0.055)]
        for i, (fx, rx, ry) in enumerate(spec):
            x = self.ox + fx * self.W
            gi = int(np.clip(x, 0, self.PW - 1))
            y = self.gy[gi] + ry * self.H * 0.25
            R.append((x, y, rx * self.W, ry * self.H * 2.2, self.seed * 7 + i))
        return R

    def rock_mask(self, cx, cy, rx, ry, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(9, 13))
        ang = np.sort(np.linspace(0, 2 * math.pi, n, endpoint=False) + rng.uniform(-0.2, 0.2, n))
        r = 1 + 0.16 * rng.uniform(-1, 1, n)
        # faceted: sparse vertices, straight-ish edges, flatter top
        pts = []
        for a, rr in zip(ang, r):
            y = math.sin(a)
            yf = 0.75 if y < 0 else 1.0
            pts.append((cx + math.cos(a) * rx * rr, cy - (-y) * ry * rr * yf))
        return pts

    # ------------------------------------------------------------------------------------------ build
    def build(self):
        PW, PH, sc = self.PW, self.PH, self.sc
        rng = self.rng
        sx, sy = self.sun
        yy, xx = np.mgrid[0:PH, 0:PW].astype(F32)
        gy = self.ground()
        S = np.zeros((PH, PW, 4), F32)      # static layer
        # ground mass: dark, cool, soft top-lit gradient near the crest; broad painted patches of soil / turf
        gm = (yy >= gy[None, :]).astype(F32)
        gm = cv2.GaussianBlur(gm, (0, 0), 0.6)
        din = np.clip((yy - gy[None, :]) / (0.25 * self.H), 0, 1)
        n1 = fbm(PW, PH, max(PW / (140 * sc), 3), 31, 4, stretch=3.0, angle=-6)
        n2 = fbm(PW, PH, max(PW / (40 * sc), 3), 32, 4, stretch=2.0, angle=-10)
        turf = ss(-0.1, 0.15, n1 + 0.4 * n2)[..., None]
        soil = np.array([0.13, 0.10, 0.11], F32)
        grass = np.array([0.07, 0.095, 0.09], F32)
        col = soil + (grass - soil) * turf
        top_fill = np.exp(-din / 0.08)[..., None]
        col = col + np.array([0.16, 0.13, 0.24], F32) * top_fill * 0.5      # sky fill on the crest
        col = col * (1 - 0.7 * din[..., None])
        col = col * (1 + 0.1 * n2[..., None])
        over(S, col, gm)
        # boulders
        self.rocks = self.rock_shapes()
        for (cx, cy, rx, ry, sd) in self.rocks:
            self._paint_rock(S, cx, cy, rx, ry, sd, xx, yy)
        # torii + signpost
        self._torii(S)
        self._signpost(S)
        # sway layer: pine, grass, shide papers
        D = np.zeros((PH, PW, 4), F32)
        Wt = np.zeros((PH, PW), F32)
        self._pine(D, Wt)
        self._shimenawa(S, D, Wt)
        self._grass(D, Wt, gy)
        # rim light on the union silhouette (baked into both layers)
        Au = np.clip(S[..., 3] + D[..., 3] * (1 - S[..., 3]), 0, 1)
        rim = rim_light(Au, sx, sy, 2.2 * sc, soft=0.6 * sc, macro=2.0 * sc)
        rim_w = rim_light(Au, sx, sy, 6.0 * sc, soft=2.5 * sc, macro=3.0 * sc) * 0.2
        rc = np.array([1.55, 0.98, 0.52], F32)
        # stronger rim close to the sun direction (grazing)
        dd = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2) / self.W
        gain = (0.4 + 0.9 * np.exp(-(dd / 0.3) ** 2))[..., None]
        add = (rim[..., None] * rc + rim_w[..., None] * np.array([1.0, 0.55, 0.35], F32)) * gain
        dS = S[..., 3] / np.maximum(Au, 1e-4)
        S[..., :3] += add * (S[..., 3] > 0.01)[..., None] * 1.0
        D[..., :3] += add * (D[..., 3] > 0.01)[..., None]
        bleed(S, 2.0)
        bleed(D, 2.0)
        Wt = cv2.GaussianBlur(Wt, (0, 0), 3 * sc)
        return S, D, Wt

    # ----------------------------------------------------------------------------------------- painters
    def _paint_rock(self, S, cx, cy, rx, ry, sd, xx, yy):
        sc = self.sc
        pts = self.rock_mask(cx, cy, rx, ry, sd)
        x0, x1 = int(max(cx - rx * 1.4, 0)), int(min(cx + rx * 1.4, self.PW))
        y0, y1 = int(max(cy - ry * 1.4, 0)), int(min(cy + ry * 1.4, self.PH))
        if x1 <= x0 or y1 <= y0:
            return
        w, h = x1 - x0, y1 - y0
        m = poly_mask(w, h, [(p[0] - x0, p[1] - y0) for p in pts])
        # shape-from-silhouette height -> normals
        dt = cv2.distanceTransform((m > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
        hgt = np.sqrt(np.clip(dt / (0.5 * ry + 1e-3), 0, 1)) * ry * 0.6
        hgt = cv2.GaussianBlur(hgt, (0, 0), max(ry * 0.08, 1))
        rng = np.random.default_rng(sd)
        # facet planes: add a few planar tilts cut by lines
        for _ in range(3):
            a = rng.uniform(0, math.pi)
            nx, ny = math.cos(a), math.sin(a)
            off = rng.uniform(-0.3, 0.3) * rx
            ly, lx_ = np.mgrid[0:h, 0:w].astype(F32)
            dsg = (lx_ - (cx - x0)) * nx + (ly - (cy - y0)) * ny - off
            hgt = hgt + np.where(dsg > 0, dsg * 0.18, 0)
        gy_, gx_ = np.gradient(hgt)
        nz = np.ones_like(hgt) * 1.2
        nn = np.sqrt(gx_ ** 2 + gy_ ** 2 + nz ** 2)
        nxn, nyn, nzn = -gx_ / nn, -gy_ / nn, nz / nn
        up = np.clip(-nyn * 0.9 + nzn * 0.25, 0, 1)                       # sky fill from above
        sdx, sdy = self.sun[0] - cx, self.sun[1] - cy
        ln = math.hypot(sdx, sdy) + 1e-6
        Ls = np.array([sdx / ln * 0.8, sdy / ln * 0.8, -0.55])            # sun behind the rock
        sunl = np.clip(nxn * Ls[0] + nyn * Ls[1] + nzn * Ls[2], 0, 1)
        n = fbm(w, h, max(w / (10 * sc), 3), sd, 3)
        v = 0.55 * up + 0.12 * n
        # painted planes
        pl = ss(0.28, 0.34, v) * 0.55 + ss(0.5, 0.56, v) * 0.45
        dark = np.array([0.07, 0.065, 0.10], F32)
        mid = np.array([0.13, 0.12, 0.18], F32)
        lite = np.array([0.25, 0.22, 0.33], F32)
        col = dark + (mid - dark) * ss(0.0, 0.55, pl)[..., None]
        col = col + (lite - col) * ss(0.55, 1.0, pl)[..., None]
        col = col + np.array([1.3, 0.75, 0.4], F32) * (ss(0.35, 0.6, sunl) * 0.35)[..., None]
        # cracks: a few dark hairline walks
        cr = np.zeros((h, w), F32)
        for _ in range(int(3 + rx / (20 * sc))):
            p = np.array([rng.uniform(0.2, 0.8) * w, rng.uniform(0.2, 0.7) * h])
            ang = rng.uniform(0, math.pi)
            pl_ = [p.copy()]
            for k in range(int(rng.integers(3, 7))):
                ang += rng.uniform(-0.6, 0.6)
                p = p + np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(4, 12) * sc
                pl_.append(p.copy())
            cv2.polylines(cr, [np.round(np.asarray(pl_) * 16).astype(np.int32)], False, 1.0,
                          max(1, int(1.0 * sc + 0.5)), cv2.LINE_AA, shift=4)
        col = col * (1 - 0.55 * cr[..., None] * m[..., None])
        # lichen specks / weathering
        sp = (rng.random((h, w)) > 0.994).astype(F32)
        sp = cv2.GaussianBlur(sp, (0, 0), 0.7 * sc) * 3
        col = col + np.array([0.25, 0.25, 0.2], F32) * (np.clip(sp, 0, 1) * up)[..., None]
        # base dissolves into the grass/soil (darker toward the ground)
        ly = np.arange(h, dtype=F32)[:, None] + y0
        col = col * (1 - 0.35 * ss(cy - ry * 0.2, cy + ry * 0.6, ly))[..., None]
        sub = S[y0:y1, x0:x1]
        over(sub, col, m)

    def _box(self, S, pts, col):
        m = poly_mask(self.PW, self.PH, pts)
        over(S, np.broadcast_to(np.asarray(col, F32), S[..., :3].shape), m)
        return m

    def _torii(self, S):
        sc, W, H = self.sc, self.W, self.H
        cxf, gyf = 0.235, None
        cx = self.ox + cxf * W
        gyp = float(self.gy[int(cx)]) + 3 * sc
        Ht = 0.36 * H
        span = 0.62 * Ht                          # pillar spacing (centres)
        pw = 0.075 * Ht                           # pillar width
        top = gyp - Ht
        red_d = np.array([0.19, 0.05, 0.05], F32)
        red_m = np.array([0.31, 0.075, 0.065], F32)
        black = np.array([0.06, 0.05, 0.07], F32)
        L = np.zeros((self.PH, self.PW), F32)
        parts = []
        # pillars (slight inward lean), black-lacquered base ring (nemaki), stone footing
        for s in (-1, 1):
            xb = cx + s * span / 2
            xt = xb - s * 0.035 * Ht
            y_t = top + 0.1 * Ht
            q = [(xb - pw / 2, gyp), (xt - pw * 0.45, y_t), (xt + pw * 0.45, y_t), (xb + pw / 2, gyp)]
            parts.append((q, red_m, 'v'))
            q2 = [(xb - pw * 0.56, gyp), (xb - pw * 0.55, gyp - 0.08 * Ht), (xb + pw * 0.55, gyp - 0.08 * Ht),
                  (xb + pw * 0.56, gyp)]
            parts.append((q2, black, 'n'))
            q3 = [(xb - pw * 0.85, gyp + 0.012 * Ht), (xb - pw * 0.75, gyp - 0.02 * Ht),
                  (xb + pw * 0.75, gyp - 0.02 * Ht), (xb + pw * 0.85, gyp + 0.012 * Ht)]
            parts.append((q3, np.array([0.2, 0.19, 0.25], F32), 's'))
        # nuki (tie beam) passing through the pillars
        yn = top + 0.3 * Ht
        tn = 0.055 * Ht
        ext = span / 2 + 0.16 * Ht
        parts.append(([(cx - ext, yn), (cx + ext, yn), (cx + ext, yn + tn), (cx - ext, yn + tn)], red_m, 'h'))
        # gakuzuka (centre strut) + plaque
        parts.append(([(cx - 0.03 * Ht, top + 0.12 * Ht), (cx + 0.03 * Ht, top + 0.12 * Ht),
                       (cx + 0.03 * Ht, yn), (cx - 0.03 * Ht, yn)], red_d, 'v'))
        # shimaki (lower lintel)
        ys_ = top + 0.075 * Ht
        ts = 0.05 * Ht
        es = span / 2 + 0.2 * Ht
        parts.append(([(cx - es, ys_), (cx + es, ys_), (cx + es, ys_ + ts), (cx - es, ys_ + ts)], red_m, 'h'))
        # kasagi: black-capped top lintel with upswept ends (sorimashi)
        ek = span / 2 + 0.3 * Ht
        n = 40
        xs = np.linspace(-ek, ek, n)
        up = 0.07 * Ht * (np.abs(xs) / ek) ** 2.6
        yk = top + 0.01 * Ht - up
        tk = 0.07 * Ht
        upper = [(cx + x, y) for x, y in zip(xs, yk)]
        lower = [(cx + x, y + tk * (0.9 - 0.25 * (abs(x) / ek) ** 3)) for x, y in zip(xs[::-1], yk[::-1])]
        kas = upper + lower
        parts.append((kas, red_m, 'k'))
        cap = upper + [(cx + x, y + tk * 0.32) for x, y in zip(xs[::-1], yk[::-1])]
        parts.append((cap, black, 'c'))
        # paint the parts: backlit lacquer - dark face, sky-lit top edges, weathering streaks
        n1 = fbm(self.PW, self.PH, max(self.PW / (6 * sc), 4), 71, 3, stretch=0.15)
        for q, c, kind in parts:
            m = poly_mask(self.PW, self.PH, q)
            ys = [p[1] for p in q]
            y0_, y1_ = min(ys), max(ys)
            yy = np.arange(self.PH, dtype=F32)[:, None]
            xx_ = np.arange(self.PW, dtype=F32)[None, :]
            if kind in ('h', 'k', 'c'):
                # top face catches the cool sky (thin light band), front face in shadow, dark underside line
                g = ss(y0_, y0_ + (y1_ - y0_) * 0.3, yy)
                bot = ss(y1_ - (y1_ - y0_) * 0.25, y1_, yy)
                colr = c * (1.0 - 0.3 * bot[..., None]) + np.array([0.16, 0.12, 0.24], F32) * (1 - g[..., None])
                colr = colr * (0.93 + 0.1 * n1[..., None])
            else:
                # pillars: backlit cylinders - dark core, lighter sun-side edge, weathering streaks,
                # darker toward the ground
                xs_ = [p[0] for p in q]
                xc = (min(xs_) + max(xs_)) / 2
                hw = (max(xs_) - min(xs_)) / 2 + 1e-3
                u = np.clip((xx_ - xc) / hw, -1, 1)
                cyl = 0.78 + 0.22 * ss(0.2, 0.95, u) + 0.12 * ss(-0.5, -1.0, u)
                colr = c * cyl[..., None] * (0.92 + 0.14 * n1[..., None]) * (1 - 0.3 * ss(y0_, y1_, yy))[..., None]
            over(S, np.broadcast_to(colr, S[..., :3].shape).astype(F32), m)
            L = np.maximum(L, m)
        # plaque with the shrine name (small, dark board with pale characters)
        pwid, ph = 0.075 * Ht, 0.1 * Ht
        pq = [(cx - pwid / 2, top + 0.13 * Ht), (cx + pwid / 2, top + 0.13 * Ht),
              (cx + pwid / 2, top + 0.13 * Ht + ph), (cx - pwid / 2, top + 0.13 * Ht + ph)]
        pm = poly_mask(self.PW, self.PH, pq)
        over(S, np.broadcast_to(np.array([0.07, 0.06, 0.09], F32), S[..., :3].shape), pm)
        tm = _text_img('雲見', ph * 0.36)
        T = np.zeros((self.PH, self.PW), F32)
        _stamp(T, tm, cx - tm.shape[1] / 2, top + 0.13 * Ht + ph * 0.08)
        over(S, np.broadcast_to(np.array([0.62, 0.5, 0.36], F32), S[..., :3].shape), T * pm * 0.85)
        self.torii = dict(cx=cx, top=top, gy=gyp, span=span, pw=pw, Ht=Ht, yn=yn, tn=tn)

    def _shimenawa(self, S, D, Wt):
        """Twisted straw rope slung under the nuki + four zigzag shide papers (translucent, they sway)."""
        T = self.torii
        sc = self.sc
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
        # twist: diagonal strand lines
        for i in range(0, n - 1):
            x_ = xs[i]
            y_ = ys[i]
            cv2.line(Vt, (int((x_ - th * 0.3) * 16), int((y_ - th * 0.45) * 16)),
                     (int((x_ + th * 0.3) * 16), int((y_ + th * 0.45) * 16)), 1.0, max(1, int(1.2 * sc)),
                     cv2.LINE_AA, shift=4)
        straw = np.array([0.30, 0.25, 0.17], F32)
        yy = np.arange(self.PH, dtype=F32)[:, None]
        col = straw * (1 - 0.35 * Vt[..., None])
        over(S, np.broadcast_to(col, S[..., :3].shape).astype(F32), R)
        # shide: zigzag strips hanging from the rope
        rng = np.random.default_rng(5)
        P = np.zeros((self.PH, self.PW), F32)
        for k in range(4):
            fx = (k + 0.5) / 4
            x = xl + (xr - xl) * fx
            y = y0 + sag * (1 - ((x - cx) / ((xr - xl) / 2)) ** 2) + th * 0.3
            wz = 0.016 * Ht
            hz = 0.022 * Ht
            pts = []
            for j in range(4):
                ox_ = (j % 2) * wz * 0.6
                pts.append([(x + ox_ - wz / 2, y + j * hz), (x + ox_ + wz / 2, y + j * hz),
                            (x + ox_ + wz / 2, y + (j + 1) * hz + 1), (x + ox_ - wz / 2, y + (j + 1) * hz + 1)])
            for q in pts:
                cv2.fillPoly(P, [np.round(np.asarray(q) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
            cv2.circle(Wt, (int(x), int(y + 2 * hz)), int(3 * hz), 1.0, -1)
        paper = np.array([0.82, 0.70, 0.60], F32)          # backlit translucent paper glows warm
        over(D, np.broadcast_to(paper, D[..., :3].shape).astype(F32), P)

    def _signpost(self, S):
        sc, W, H = self.sc, self.W, self.H
        x = self.ox + 0.075 * W
        gyp = float(self.gy[int(x)]) + 4 * sc
        hp = 0.2 * H
        wp = 0.026 * W
        top = gyp - hp
        # square post with a pointed top, front face + narrow sun-side face
        face = [(x - wp / 2, gyp), (x - wp / 2, top + wp * 0.3), (x, top), (x + wp / 2, top + wp * 0.3),
                (x + wp / 2, gyp)]
        side = [(x + wp / 2, gyp), (x + wp / 2, top + wp * 0.3), (x + wp * 0.72, top + wp * 0.36),
                (x + wp * 0.72, gyp - 2)]
        yy = np.arange(self.PH, dtype=F32)[:, None]
        wood = np.array([0.20, 0.16, 0.16], F32)
        n = fbm(self.PW, self.PH, max(self.PW / (3 * sc), 4), 91, 3, stretch=0.1)
        col = wood * (0.9 + 0.15 * n[..., None]) * (1 - 0.35 * ss(top, gyp, yy))[..., None]
        m = poly_mask(self.PW, self.PH, face)
        over(S, col.astype(F32), m)
        ms = poly_mask(self.PW, self.PH, side)
        over(S, np.broadcast_to(np.array([0.62, 0.38, 0.26], F32), S[..., :3].shape), ms)
        # engraved vertical inscription (real Japanese): 雲見岳山頂 + elevation
        tm = _text_img('雲見岳山頂', wp * 0.62)
        T = np.zeros((self.PH, self.PW), F32)
        _stamp(T, tm, x - tm.shape[1] / 2, top + wp * 0.75)
        over(S, np.broadcast_to(np.array([0.04, 0.03, 0.04], F32), S[..., :3].shape), T * m * 0.95)
        t2 = _text_img('標高二四六八米', wp * 0.26)
        T2 = np.zeros((self.PH, self.PW), F32)
        _stamp(T2, t2, x + wp * 0.12, top + wp * 0.75 + tm.shape[0] + wp * 0.1)
        over(S, np.broadcast_to(np.array([0.08, 0.06, 0.08], F32), S[..., :3].shape), T2 * m * 0.8)

    def _pine(self, D, Wt):
        """Japanese black pine on the right knoll: gnarled trunk leaning in from the right edge, layered
        cloud-pruned needle pads (clustered domes with needle-fringed edges), kept clear of the sun."""
        sc, W, H = self.sc, self.W, self.H
        rng = np.random.default_rng(self.seed + 55)
        PW, PH = self.PW, self.PH
        A = np.zeros((PH, PW), F32)          # trunk / branch
        N = np.zeros((PH, PW), F32)          # needles
        Tp = np.zeros((PH, PW), F32)         # pad top-ness (lighter tops)
        sw = np.zeros((PH, PW), F32)
        bx = self.ox + 1.02 * W
        by = float(self.gy[min(int(bx), PW - 1)]) + 20 * sc

        def line(img, p0, p1, wd, val=1.0):
            cv2.line(img, tuple(np.round(np.asarray(p0) * 16).astype(int)), tuple(np.round(np.asarray(p1) * 16).astype(int)),
                     val, max(1, int(round(wd))), cv2.LINE_AA, shift=4)

        # trunk: leans left, kinks, then rises off the top of the frame
        pos = np.array([bx, by], np.float64)
        ang = -math.pi / 2 - 0.42
        tp = [pos.copy()]
        n = 26
        seg = 1.0 * H / n
        for i in range(n):
            ang += rng.uniform(-0.14, 0.14) + (0.035 if i > 8 else 0.0)
            pos = pos + np.array([math.cos(ang), math.sin(ang)]) * seg
            tp.append(pos.copy())
        tp = np.asarray(tp)
        for i in range(len(tp) - 1):
            wdt = (46 - 26 * i / len(tp)) * sc
            line(A, tp[i], tp[i + 1], wdt)
            # bark plates: a few knobbly bumps along the trunk edge
            if rng.random() < 0.0:
                q = tp[i] + rng.normal(0, 1, 2) * wdt * 0.25
                cv2.circle(A, (int(q[0]), int(q[1])), max(1, int(wdt * 0.55)), 1.0, -1, cv2.LINE_AA)
        pads = []
        spec = [(9, -1, 0.17, 0.0), (12, -1, 0.21, -0.1), (15, -1, 0.15, -0.2), (17, 1, 0.07, 0.0),
                (19, -1, 0.19, -0.15), (22, -1, 0.12, -0.25), (24, 1, 0.08, -0.2), (6, -1, 0.07, 0.25)]
        for (i, side, blen, lift) in spec:
            p = tp[i].copy()
            ang = (math.pi if side < 0 else 0.0) + lift * side * -1
            pts = [p.copy()]
            nseg = 8
            for j in range(nseg):
                ang += rng.uniform(-0.3, 0.3)
                p = p + np.array([math.cos(ang), math.sin(ang) * 0.7]) * blen * W / nseg
                pts.append(p.copy())
                if j in (3, 5) and rng.random() < 0.7:
                    pads.append((p.copy() + np.array([0, -6 * sc]), rng.uniform(0.035, 0.05) * W))
            pads.append((p.copy(), rng.uniform(0.05, 0.07) * W * (0.7 + 0.3 * blen / 0.2)))
            for j in range(len(pts) - 1):
                wdt = (16 - 12 * j / len(pts)) * sc * (0.6 + blen * 2)
                line(A, pts[j], pts[j + 1], wdt)
        # keep pads clear of the sun
        sx, sy = self.sun
        pads = [(c, r) for (c, r) in pads if (c[0] - r) > sx + 0.1 * W or c[1] + r * 0.3 < sy - 0.12 * H]
        V = np.zeros((PH, PW), F32)           # painted value of the needles (0 dark core -> 1 sky-lit tips)
        for (c, r) in pads:
            ry = r * 0.34
            # tufts (brush-like fans of needles) packed into a flattened dome; drawn top -> bottom so the
            # lower, nearer tufts overlap the ones behind (pads seen from slightly below)
            nt = int(r * ry / (7.5 * sc) ** 2 * 1.6) + 10
            tufts = []
            for _ in range(nt * 3):
                u = rng.uniform(-1, 1)
                v = rng.uniform(-1, 1)
                prof = 1 - u * u
                if v < -prof ** 0.6 or v > 0.55:
                    continue
                tufts.append((c[0] + u * r, c[1] + v * ry, 0.5 - 0.5 * v))
                if len(tufts) >= nt:
                    break
            tufts.sort(key=lambda q: q[1])
            for (tx, ty, lvl) in tufts:
                rt = rng.uniform(11, 18) * sc
                base = np.array([tx, ty + rt * 0.35])
                cv2.ellipse(N, (int(tx * 4), int((ty + rt * 0.1) * 4)), (int(rt * 0.75 * 4), int(rt * 0.42 * 4)), 0, 0, 360,
                            1.0, -1, cv2.LINE_AA, shift=2)
                cv2.ellipse(V, (int(tx * 4), int((ty + rt * 0.1) * 4)), (int(rt * 0.75 * 4), int(rt * 0.42 * 4)), 0, 0, 360,
                            0.25 * lvl, -1, cv2.LINE_AA, shift=2)
                for q in range(int(rng.integers(12, 20))):
                    a = -math.pi / 2 + rng.uniform(-1.45, 1.45)
                    la = rt * rng.uniform(0.7, 1.15)
                    tip = base + np.array([math.cos(a), math.sin(a) * 0.9]) * la
                    line(N, base, tip, 0.9 * sc + 0.5)
                    line(V, base + (tip - base) * 0.45, tip, 0.9 * sc + 0.5, float(np.clip(0.35 + 0.65 * lvl, 0, 1)
                                                                                   * rng.uniform(0.6, 1.0)))
            yy = np.arange(PH, dtype=F32)[:, None]
            cv2.ellipse(sw, (int(c[0]), int(c[1])), (int(r * 1.2), int(ry * 2.0)), 0, 0, 360, 1.0, -1)
        Tp = V
        N = np.clip(N, 0, 1)
        bark = np.array([0.085, 0.06, 0.07], F32)
        nz = fbm(PW, PH, max(PW / (4 * sc), 4), 58, 4, stretch=0.25)
        bc = bark * (0.75 + 0.5 * ss(-0.3, 0.5, nz)[..., None])
        over(D, bc.astype(F32), np.clip(A, 0, 1))
        needle_d = np.array([0.035, 0.05, 0.06], F32)
        needle_m = np.array([0.07, 0.09, 0.10], F32)
        needle_t = np.array([0.15, 0.165, 0.2], F32)
        # painted light from the MASS, not per needle: crowns (mask empty just above) catch the sky,
        # a second smaller step marks the top of each clump; interior stays a flat dark plane
        Nb = cv2.GaussianBlur(N, (0, 0), 3.0 * sc)
        yy_, xx_ = self._pg if hasattr(self, '_pg') else np.mgrid[0:PH, 0:PW].astype(F32)
        up1 = np.clip(Nb - cv2.remap(Nb, xx_, yy_ - 9 * sc, cv2.INTER_LINEAR), 0, 1)
        up2 = np.clip(Nb - cv2.remap(Nb, xx_, yy_ - 22 * sc, cv2.INTER_LINEAR), 0, 1)
        nz2 = fbm(PW, PH, max(PW / (30 * sc), 4), 61, 3)
        crown = ss(0.25, 0.45, up2 + 0.15 * nz2)
        tipb = ss(0.3, 0.5, up1 + 0.1 * nz2)
        nc = needle_d + (needle_m - needle_d) * crown[..., None]
        nc = nc + (needle_t - nc) * (tipb * 0.8)[..., None]
        # a little needle texture only near the crowns
        nc = nc * (1 + 0.25 * (np.clip(Tp, 0, 1) - 0.3) * crown)[..., None]
        over(D, nc.astype(F32), N)
        yy = np.arange(PH, dtype=F32)[:, None]
        hfac = ss(by, by - 0.8 * H, yy)
        Wt[:] = np.maximum(Wt, cv2.GaussianBlur(sw, (0, 0), 8 * sc) * 0.7 * hfac + np.clip(A, 0, 1) * hfac * 0.2)

    def _grass(self, D, Wt, gy):
        """Grass blades: silhouetted along the crest, dense tufts in front; tips glow when backlit."""
        sc = self.sc
        PW, PH = self.PW, self.PH
        rng = np.random.default_rng(self.seed + 77)
        ssf = 2
        w2, h2 = PW * ssf, PH * ssf
        G = np.zeros((h2, w2), np.uint8)
        V = np.zeros((h2, w2), np.uint8)       # value along the blade (0 base -> 255 tip)
        clump = frac1d(PW, self.seed + 9, 5, 24, 0.6)
        tor = self.torii
        rocks = self.rocks
        n_blades = int(9000 * sc)
        for b in range(n_blades):
            x = rng.uniform(0, PW - 1)
            xi = int(x)
            # most roots near the crest (silhouette), some scattered further down
            if rng.random() < 0.62:
                y = gy[xi] + rng.uniform(-1, 6) * sc
            else:
                y = gy[xi] + rng.uniform(0, 1) ** 1.7 * (PH - gy[xi])
            depth = np.clip((y - gy[xi]) / (0.25 * self.H), 0, 1)
            cl = 0.5 + 0.5 * clump[xi]
            L = rng.uniform(6, 30) * sc * (0.4 + 1.1 * cl) * (1 + 1.4 * depth)
            if rng.random() < 0.08:
                L *= 1.8
            a = -math.pi / 2 + rng.normal(0.18, 0.28)
            curv = rng.normal(0.25, 0.25)
            k = 5
            p = np.array([x, y])
            pts = [p.copy()]
            for i in range(k):
                a += curv / k
                p = p + np.array([math.cos(a), math.sin(a)]) * L / k
                pts.append(p.copy())
            wd = max(1, int(round(rng.uniform(1.0, 2.6) * sc * ssf * (1 + depth))))
            for i in range(k):
                p0 = np.round(pts[i] * ssf * 16).astype(int)
                p1 = np.round(pts[i + 1] * ssf * 16).astype(int)
                th = max(1, int(round(wd * (1 - i / k * 0.75))))
                cv2.line(G, tuple(p0), tuple(p1), 255, th, cv2.LINE_AA, shift=4)
                cv2.line(V, tuple(p0), tuple(p1), int(255 * (i + 1) / k), th, cv2.LINE_8, shift=4)
        # alpine flowers (tiny pink / violet dots near the crest)
        Fm = np.zeros((h2, w2), np.uint8)
        for _ in range(int(260 * sc)):
            x = rng.uniform(0, PW - 1)
            y = gy[int(x)] + rng.uniform(-6, 40) * sc
            r = max(1, int(rng.uniform(1.2, 2.6) * sc * ssf))
            cv2.circle(Fm, (int(x * ssf), int(y * ssf)), r, 255, -1, cv2.LINE_AA)
        a = cv2.resize(G.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        v = cv2.resize(V.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        fm = cv2.resize(Fm.astype(F32) / 255, (PW, PH), interpolation=cv2.INTER_AREA)
        v = v / np.maximum(a, 1e-3)
        yy = np.arange(PH, dtype=F32)[:, None]
        din = np.clip((yy - gy[None, :]) / (0.25 * self.H), 0, 1)
        base = np.array([0.05, 0.07, 0.07], F32)
        tip = np.array([0.20, 0.23, 0.16], F32)
        col = (base + (tip - base) * np.clip(v, 0, 1)[..., None] * (1 - 0.5 * din[..., None])) * (1 - 0.55 * din[..., None])
        # translucent glow: blades near the crest and toward the sun let light through
        dx = (np.arange(PW, dtype=F32)[None, :] - self.sun[0]) / self.W
        sunw = np.exp(-(dx / 0.3) ** 2)
        tr = np.clip(v, 0, 1) * (1 - din) ** 2 * (0.35 + 0.65 * sunw)
        col = col + np.array([0.75, 0.62, 0.2], F32) * tr[..., None] * 0.45
        over(D, col.astype(F32), a)
        over(D, np.broadcast_to(np.array([0.55, 0.28, 0.45], F32), D[..., :3].shape), fm * 0.8)
        Wt[:] = np.maximum(Wt, np.clip(v, 0, 1) * a * 1.0)
