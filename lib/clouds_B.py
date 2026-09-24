"""Technique B: layered cut-paper cloud painting (Shinkai-style cumulus).

A cloud is painted like a background artist would: back to front, as a handful (3-8) of big FLAT
shapes. Each shape is
  * a cauliflower silhouette: a smooth dome envelope whose contour is grown fractally by unions of
    boundary circles (big heads on top, small florets on the flanks, none on the base) - so all the
    detail lives on the EDGE and the interior is one flat paper cut-out;
  * filled with a smooth 2-tone painted gradient (lit warm-white -> cool lavender) whose direction
    follows the sun. The light/shadow split is a *soft offset of the silhouette itself* (a pixel is lit
    if the shape does not continue for D px toward the sun), so the terminator echoes the outline the
    way painters draw it - no per-lobe sphere shading, no ambient occlusion;
  * composited over what is already painted with a crisp edge on the sun side and a soft, lost edge
    on the shadow side;
  * given a thin bright rim stroke on the sun-facing contour (silver/gold lining; wider and hotter
    where the sun is right behind the edge).
Final passes: watercolour-like pigment pooling on the edges, faint paper-scale value variation, and
torn wispy base fringes (noise-warped, alpha-faded streaks).

API (all colours linear-ish display RGB 0..1, plates are straight-alpha RGBA float32)
--------------------------------------------------------------------------------------
PALETTES                          'summer', 'summer_far', 'sunrise', 'sunrise_far', 'sunrise_tower'
                                  keys: hi lit mid shade deep rim haze edge_dark;  mix_palette(a, b, t)
envelope(cx, base_y, w, h, rng, ...)            dome polygon (N, 2) for one mass
anvil_poly(xc, top_y, left, right, thick, rng)  flat-topped spreading anvil polygon
billow_strip(x0, x1, base_y, bump_w, bump_h, rng) long scalloped strip (one row segment of a cloud sea)
cauliflower(poly, rng, size, levels, ...)       -> (mask float32 (h, w), x0, y0) boundary-circle florets
Painter(w, h, sun, pal, unit, seed)             back-to-front painter on one plate (unit = W/1920)
    .shape(poly, rng, size, group, pal, ...)    paint one flat mass (lit_depth, two_tone, firm, rim_px,
                                                lost, cast, echo, haze, backlit, fray, band_var ...)
    .wisps(x0, x1, base_y, depth, ...)          torn base edge + faint wisps
    .haze_band(y0, y1, color, amount)           aerial haze toward rows y0 -> y1
    .rgba()                                     -> (h, w, 4) float32 straight alpha (use sky.drift to
                                                sample with drift / parallax, fx.over_rgba to composite)
cumulonimbus(P, cx, base_y, Hc, seed, anvil, anvil_dir, width, pal, haze, backlit)
cumulus(P, cx, base_y, w, h, seed, n, pal, haze, ...)
bank(P, y, x0, x1, height, seed, pal, haze, rows)   distant horizon bank
cloud_sea(P_or_fn, horizon_y, bottom_y, seed, rows, pal_near, pal_far, lit_depth, valley,
          row_callback)                         sea of clouds from above (perspective rows)
Cost: ~5 s (summer tower scene) / ~12 s (sea of clouds, 3 plates) at 1080p; numba kernel _paint_px.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from . import core as C

# ----------------------------------------------------------------------------------------- palettes


def _c(h):
    return C.hex2rgb(h) if isinstance(h, str) else np.asarray(h, np.float32)


PALETTES = {
    # lit: sunlit face   hi: hottest light (HDR ok)   mid: terminator band   shade: shadow face
    # deep: core shadow / base   rim: silver lining (HDR)   haze: aerial perspective target
    'summer': dict(hi=(1.02, 1.0, 0.96), lit='#f5f3f1', mid='#dfd6e2', shade='#a2acdc', deep='#7885c6',
                   rim=(1.08, 1.07, 1.03), haze='#bfe0f5', edge_dark=0.07),
    'summer_far': dict(hi=(0.97, 0.98, 0.98), lit='#eef2f6', mid='#d6e0ef', shade='#b3c6e6', deep='#9cb2de',
                       rim=(1.2, 1.18, 1.12), haze='#cdeaf7', edge_dark=0.04),
    'sunrise': dict(hi=(1.02, 0.78, 0.54), lit='#ee8f86', mid='#d8708e', shade='#8c6cb4', deep='#2c2566',
                    rim=(1.12, 0.86, 0.55), haze='#e89cb0', edge_dark=0.08),
    'sunrise_far': dict(hi=(1.02, 0.8, 0.62), lit='#f8b294', mid='#e896aa', shade='#a484c0', deep='#7c64aa',
                        rim=(1.1, 0.9, 0.7), haze='#f0b0b0', edge_dark=0.04),
    'sunrise_tower': dict(hi=(1.0, 0.8, 0.56), lit='#f6aa7c', mid='#ea8a9c', shade='#8c74c0', deep='#40377e',
                          rim=(1.12, 0.92, 0.66), haze='#e0a0bc', edge_dark=0.08),
}


def palette(p):
    if isinstance(p, str):
        p = PALETTES[p]
    out = {}
    for k, v in p.items():
        out[k] = float(v) if k == 'edge_dark' else np.asarray(_c(v), np.float32)
    out.setdefault('edge_dark', 0.06)
    return out


def mix_palette(a, b, t):
    a, b = palette(a), palette(b)
    t = float(t)
    return {k: (float(a[k] * (1 - t) + b[k] * t) if k == 'edge_dark' else
                (a[k] * (1 - t) + b[k] * t).astype(np.float32)) for k in a}


# ----------------------------------------------------------------------------------------- geometry


def envelope(cx, base_y, width, height, rng, lean=0.0, n=72, power=2.4, lump=0.06, base_round=0.07,
             base_wave=0.02, skew=0.0, top_flat=0.0):
    """Dome-shaped mass outline (closed polygon, (n+m, 2) px).

    width/height - size (px); lean - horizontal shift of the top relative to the base (fraction of
    height, + = right); power - superellipse exponent (2 = ellipse, 3+ = boxier shoulders);
    lump - amplitude of low-frequency irregularity; base_round - how far below base_y the underside
    bulges (fraction of height, small = flat-ish base); base_wave - undulation of the base;
    skew - move the peak sideways (-1..1); top_flat - flatten the top (anvils: 0.5-0.9)."""
    t = np.linspace(0.0, math.pi, n)
    c, s = np.cos(t), np.sin(t)
    ex = np.sign(c) * np.abs(c) ** (2.0 / power)
    ey = np.abs(s) ** (2.0 / power)
    if top_flat:
        ey = np.minimum(ey, 1.0 - top_flat * 0.0) ** (1.0 - 0.8 * top_flat)
    # low-frequency irregularity (harmonics 2..5)
    m = np.ones_like(t)
    for k in range(2, 6):
        m += lump / (k - 1) * rng.uniform(-1, 1) * np.sin(k * t + rng.uniform(0, 6.28))
    # skew the peak
    ex = ex + skew * 0.25 * ey * (1 - ey)
    x = cx + ex * width / 2 * m
    y = base_y - ey * height * m
    x = x + lean * (base_y - y)
    # underside: from left back to right, slightly rounded and wavy
    tb = np.linspace(math.pi, 2 * math.pi, max(n // 2, 8))[1:-1]
    cb, sb = np.cos(tb), np.sin(tb)
    bx = cx + np.sign(cb) * np.abs(cb) ** (2.0 / 4.0) * width / 2 * 0.98
    by = base_y + np.abs(sb) ** 0.5 * height * base_round
    ph = rng.uniform(0, 6.28)
    by = by + base_wave * height * np.sin((bx - cx) / max(width, 1) * 9 + ph)
    return np.concatenate([np.stack([x, y], 1), np.stack([bx, by], 1)], 0).astype(np.float32)


# level = (rmin, rmax, prob[, space_min, space_max[, bulge_min, bulge_max]]), radii fractions of `size`.
# First level = a few big convective heads (widely spaced, bulging far out); later levels = florets.
def anvil_poly(xc, top_y, left, right, thick, rng, n=90, dome=0.35, droop=0.25, tip=0.18):
    """Cumulonimbus anvil outline: flat top that bulges up over the tower (x = xc), spreading `left` px
    upwind and `right` px downwind; thickest over the tower (`thick` px) and tapering to thin tips with
    a concave underside. dome/droop/tip - fractions of `thick`: dome over the tower, sag of the upper
    edge toward the tips, thickness left at the tips."""
    x = np.concatenate([np.linspace(xc - left, xc, n // 3, endpoint=False), np.linspace(xc, xc + right, n - n // 3)])
    uu = np.where(x < xc, (x - xc) / max(left, 1), (x - xc) / max(right, 1))
    a = np.clip(np.abs(uu), 0, 1)
    up = top_y - thick * dome * np.exp(-(uu / 0.22) ** 2) + thick * droop * a ** 2
    up = up + thick * 0.06 * np.sin(a * 7 + rng.uniform(0, 6.28))
    th = thick * (tip + (1 - tip) * (1 - a) ** 1.6)
    lo = up + th
    lo = lo + thick * 0.05 * np.sin(a * 11 + rng.uniform(0, 6.28))
    upper = np.stack([x, up], 1)
    lower = np.stack([x[::-1], lo[::-1]], 1)
    return np.concatenate([upper, lower], 0).astype(np.float32)


DEFAULT_LEVELS = ((0.16, 0.34, 0.9, 1.5, 2.6, 0.05, 0.4), (0.06, 0.13, 0.85, 1.1, 2.2),
                  (0.025, 0.05, 0.8, 1.0, 2.0), (0.01, 0.02, 0.7, 0.9, 1.8))


def cauliflower(poly, rng, size, levels=DEFAULT_LEVELS, down_cut=0.3, side_scale=0.5, ss=2, bulge=(0.3, 0.62),
                top_bias=1.0, max_px=None, clump=0.6, sun_dir=None, sun_bias=0.0):
    """Grow a cauliflower silhouette from an envelope polygon.

    For each level (rmin, rmax, prob[, smin, smax[, kmin, kmax]]) (radii as fractions of `size`), walk the
    current outline and union circles centred just inside it (they bulge outward by (1-k)*r). Bumps are
    biggest on upward-facing contour (convective heads), shrink by `side_scale` on vertical flanks, and
    are skipped where the outline faces down more than `down_cut` (flat-ish base). Radii are power-law
    distributed, spacing is irregular (smin..smax x r) and a slow random 'clump' modulation along the
    contour leaves some stretches smooth and others knobbly, so the edge reads as fractal florets, never
    a row of equal balloons. sun_dir/sun_bias: more florets on the sun-facing contour.
    returns (mask (h, w) float32 0..1 anti-aliased, x0, y0) - mask origin in the polygon's pixel space."""
    pts = np.asarray(poly, np.float32)
    pad = size * 0.45 + 4
    x0 = int(math.floor(pts[:, 0].min() - pad))
    y0 = int(math.floor(pts[:, 1].min() - pad))
    x1 = int(math.ceil(pts[:, 0].max() + pad))
    y1 = int(math.ceil(pts[:, 1].max() + pad))
    w, h = x1 - x0, y1 - y0
    if max_px and max(w, h) * ss > max_px:
        ss = max(1, int(max_px // max(w, h)))
    M = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(M, [np.round((pts - [x0, y0]) * ss).astype(np.int32)], 255, lineType=cv2.LINE_8)
    for li, lev in enumerate(levels):
        rmin, rmax, prob = lev[:3]
        smin, smax = (lev[3], lev[4]) if len(lev) >= 5 else (0.8, 1.7)
        kmin, kmax = (lev[5], lev[6]) if len(lev) >= 7 else bulge
        Rmax = rmax * size * ss
        if Rmax < 1.0:
            continue
        sig = max(Rmax * 0.5, 1.5)
        g = _blur(M.astype(np.float32), sig)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        cnts, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        circles = []
        for cnt in cnts:
            P = cnt[:, 0, :]
            nP = len(P)
            if nP < 12:
                continue
            ph1, ph2 = rng.uniform(0, 6.28, 2)
            f1, f2 = rng.uniform(1.5, 3.5), rng.uniform(4, 8)
            i = int(rng.integers(0, max(int(Rmax), 1)))
            while i < nP:
                px, py = int(P[i, 0]), int(P[i, 1])
                nx, ny = -gx[py, px], -gy[py, px]
                ln = math.hypot(nx, ny) + 1e-9
                nx, ny = nx / ln, ny / ln
                up = -ny
                if up < -down_cut:
                    i += max(int(Rmax * 0.5), 2)
                    continue
                f = side_scale + (1 - side_scale) * max(min(up * top_bias, 1.0), 0.0)
                r = (rmin + (rmax - rmin) * rng.random() ** 1.7) * size * ss * f
                if up < 0:
                    r *= 1 + up / max(down_cut, 1e-3) * 0.6
                s_ = i / nP * 2 * math.pi
                cl = 1.0 - clump * (0.5 + 0.25 * math.sin(f1 * s_ + ph1) + 0.25 * math.sin(f2 * s_ + ph2))
                pr = prob * cl
                if sun_dir is not None and sun_bias:
                    pr *= 1 + sun_bias * (nx * sun_dir[0] + ny * sun_dir[1])
                if r >= 0.8 and rng.random() < pr:
                    k = rng.uniform(kmin, kmax)
                    circles.append((px - nx * r * k, py - ny * r * k, r))
                i += max(int(r * rng.uniform(smin, smax)), 2)
        for (cx, cy, r) in circles:
            cv2.circle(M, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1, lineType=cv2.LINE_8)
    m = M.astype(np.float32) / 255.0
    if ss > 1:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    return m, x0, y0


# ----------------------------------------------------------------------------------------- helpers


def _shift(m, dx, dy):
    """dst(p) = m(p + (dx, dy)); outside = 0."""
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _blur(img, sigma):
    if sigma <= 0.3:
        return img
    if sigma <= 6.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    s = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 3.0 * 0.9)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


def _shift_blur(m, dx, dy, sigma):
    """blur(shift(m, dx, dy), sigma), computed at half resolution when the blur is wide (smooth result
    anyway) - the dominant cost of painting big shapes."""
    if sigma < 5.0 or min(m.shape) < 16:
        return _blur(_shift(m, dx, dy), sigma)
    h, w = m.shape
    q = 3 if sigma >= 12.0 else 2
    s = cv2.resize(m, (max(w // q, 2), max(h // q, 2)), interpolation=cv2.INTER_AREA)
    s = _blur(_shift(s, dx / q, dy / q), sigma / q)
    return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _noise(w, h, cells, seed, octaves=3, stretch=1.0):
    """Smooth value noise in 0..1 at plate size; `stretch` > 1 elongates horizontally."""
    q = 4
    wq, hq = max(w // q, 4), max(h // q, 4)
    n = C.fbm(wq, max(int(hq * stretch), 4), cells, octaves, seed=seed)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    return n.astype(np.float32)


# ----------------------------------------------------------------------------------------- painter


@njit(cache=True, fastmath=True, parallel=True)
def _paint_px(prem, oy, ox, v, dk, lw, tm, hotk, tz, hz, e, e2, rimk, glow, a, tx, ck, pal, ed):
    """Colour recipe of one flat mass + premultiplied 'over' into the plate window at (oy, ox).
    pal rows: 0 lit_lo, 1 hi, 2 mid, 3 shade, 4 deep, 5 haze, 6 rim, 7 lit."""
    H, W = v.shape
    for i in prange(H):
        for j in range(W):
            vv = v[i, j]
            aa = a[i, j]
            ckk = ck[i, j] * (1.0 - aa)
            if ckk > 0.6:
                ckk = 0.6
            dst_a = prem[oy + i, ox + j, 3]
            for c in range(3):
                sh = pal[3, c] * (1.0 - dk[i, j]) + pal[4, c] * dk[i, j]
                lc = pal[0, c] * (1.0 - lw[i, j]) + pal[1, c] * lw[i, j]
                col = sh * (1.0 - vv) + lc * vv
                col = col * (1.0 - tm[i, j]) + pal[2, c] * tm[i, j]
                col += (pal[1, c] - pal[7, c]) * hotk[i, j]
                k = tz[i, j]
                col = col * (1.0 - k) + (pal[3, c] * 1.08 + 0.03) * k
                col = col * (1.0 - hz[i, j]) + pal[5, c] * hz[i, j]
                cool = 0.7 if c == 0 else (0.85 if c == 1 else 1.0)
                col = col * (1.0 - 0.5 * ed * e2[i, j] * (1.0 - vv)) + 0.5 * ed * e[i, j] * (1.0 - vv) * cool
                col += 0.03 * e[i, j] * vv
                r = rimk[i, j]
                col = col * (1.0 - r) + pal[6, c] * r + pal[6, c] * 0.35 * glow[i, j]
                col *= 1.0 + tx[i, j]
                d = prem[oy + i, ox + j, c]
                if ckk > 0.0:
                    d = d * (1.0 - ckk) + pal[3, c] * 0.9 * dst_a * ckk
                prem[oy + i, ox + j, c] = col * aa + d * (1.0 - aa)
            prem[oy + i, ox + j, 3] = aa + dst_a * (1.0 - aa)


class Painter:
    """Back-to-front cut-paper cloud painter on one RGBA plate.

    w, h  - plate size (px); sun - (x, y) sun position in plate px (may be off-plate);
    palette - name/dict (see PALETTES), per-shape overrides allowed; unit - px per 1080p-pixel
    (W/1920), scales rim widths/softness; seed - texture seed."""

    def __init__(self, w, h, sun, pal='summer', unit=1.0, seed=0):
        self.w, self.h = int(w), int(h)
        self.sun = np.array(sun, np.float32)
        self.pal = palette(pal)
        self.u = float(unit)
        self.prem = np.zeros((self.h, self.w, 4), np.float32)   # premultiplied RGBA
        self.seed = seed
        # paper-scale value variation (very subtle), shared by all shapes -> coherent
        tp = self.tp = int(0.3 * max(self.w, self.h))          # texture padding (shapes may overhang)
        TW, TH = self.w + 2 * tp, self.h + 2 * tp
        self.tex = (_noise(TW, TH, 7.0 * TW / self.w, seed + 5, 3) - 0.5).astype(np.float32)
        self.tex2 = (_noise(TW, TH, 30.0 * TW / self.w, seed + 9, 2) - 0.5).astype(np.float32)
        # anisotropic brush-stroke noise (slightly tilted, elongated): perturbs terminators
        bn = _noise(TW, TH, 16.0 * TW / self.w, seed + 13, 4, stretch=3.5) - 0.5
        M = cv2.getRotationMatrix2D((TW / 2, TH / 2), -12, 1.0)
        self.brush = cv2.warpAffine(bn.astype(np.float32), M, (TW, TH), borderMode=cv2.BORDER_REFLECT)
        self.brush *= 1.0 / (np.abs(self.brush).max() + 1e-6)
        sn = _noise(TW, TH, 10.0 * TW / self.w, seed + 17, 5, stretch=8.0)
        M = cv2.getRotationMatrix2D((TW / 2, TH / 2), -4, 1.0)
        self.streak = cv2.warpAffine(sn.astype(np.float32), M, (TW, TH), borderMode=cv2.BORDER_REFLECT)

    def _tex(self, T, x0, y0, w, h, xs=None, ys=None):
        """Window (h, w) of a padded plate texture at plate origin (x0, y0) (remap fallback if outside)."""
        tp = self.tp
        a, b = y0 + tp, x0 + tp
        if a >= 0 and b >= 0 and a + h <= T.shape[0] and b + w <= T.shape[1]:
            return T[a:a + h, b:b + w]
        return cv2.remap(T, xs + tp, ys + tp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ------------------------------------------------------------------ one flat shape
    def shape(self, poly, rng=None, size=None, levels=DEFAULT_LEVELS, group=None, pal=None, light=1.0,
              haze=0.0, haze_grad=0.0, lit_depth=0.4, lit_soft=0.08, band_w=0.4, rim=1.0, rim_px=2.6, lost=0.15, hot=1.0,
              base_dark=0.5, flat_shade=0.0, down_cut=0.3, side_scale=0.5, mask=None, sun=None, alpha=1.0,
              two_tone=(0.4, 0.6), cast=0.0, backlit=0.0, bulge=(0.3, 0.62), top_bias=1.0, sun_bias=0.5, clump=0.6, brush=0.07, fray=None, band_var=0.0, firm=0.68,
              valley_dark=0.3, lit_cool=0.15, valley_span=(0.15, 0.95), echo=0.0, echo_amt=0.5):
        """Paint one flat cauliflower mass.

        poly - envelope polygon (plate px) (or pass mask=(m, x0, y0) directly);
        size - reference size for the bump radii (default: envelope height);
        group - (cx, cy, R): frame of the whole cloud; the linear part of the light gradient is measured
                in it so neighbouring masses of one cloud agree in their shadow areas (they merge there);
        pal - palette override; light - 0..1 how much this mass receives sun (0 = in the tower's shade);
        haze/haze_grad - aerial perspective: constant + extra toward the bottom of the mass;
        lit_depth - distance (fraction of size) the lit face extends in from the sun-facing contour;
        lit_soft - softness of that terminator; two_tone - value thresholds of the 2-tone split;
        rim/rim_px - silver-lining stroke strength / width (1080p px); hot - strength of the hot light
                on the sunward contour; base_dark - darkening toward the mass's base;
        lost - softness of the shadow-side edge (lost edges into the sky);
        cast - darken what is already painted where this mass shadows it (away from the sun);
        backlit - 0..1 sun is behind this mass: glowing, translucent thin edges near the sun."""
        u = self.u
        pal = self.pal if pal is None else palette(pal)
        # plain Python floats (numpy float64 scalars would promote every array op to float64)
        light, haze, haze_grad, lit_depth, lit_soft, band_w = (float(light), float(haze), float(haze_grad),
                                                               float(lit_depth), float(lit_soft), float(band_w))
        base_dark, lost, rim, hot, backlit, cast = (float(base_dark), float(lost), float(rim), float(hot),
                                                    float(backlit), float(cast))
        if rng is None:
            rng = np.random.default_rng(0)
        if mask is None:
            poly = np.asarray(poly, np.float32)
            if size is None:
                size = float(poly[:, 1].max() - poly[:, 1].min())
            sp = (self.sun if sun is None else np.array(sun, np.float32)) - poly.mean(0)
            sp = sp / (float(np.hypot(*sp)) + 1e-6)
            # child generator: the floret walk consumes a resolution-dependent number of draws, so it
            # must not advance the caller's stream (keeps the layout identical at every resolution)
            crng = np.random.default_rng(int(rng.integers(1 << 31)))
            m, x0, y0 = cauliflower(poly, crng, size, levels, down_cut=down_cut, side_scale=side_scale,
                                    bulge=bulge, top_bias=top_bias, sun_dir=sp, sun_bias=sun_bias,
                                    clump=clump)
        else:
            m, x0, y0 = mask
            if size is None:
                size = m.shape[0]
        # expand canvas for soft edges / casts
        pad = int(max(12 * u, size * 0.06))
        m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0 -= pad
        y0 -= pad
        h, w = m.shape
        # clip against plate
        px0, py0 = max(x0, 0), max(y0, 0)
        px1, py1 = min(x0 + w, self.w), min(y0 + h, self.h)
        if px1 <= px0 or py1 <= py0:
            return None
        mom = cv2.moments(m)
        area = mom['m00'] + 1e-6
        mcx = float(mom['m10'] / area) + x0
        mcy = float(mom['m01'] / area) + y0
        ytop = float(y0 + pad)
        yb = float(y0 + h - pad)
        # crop to the plate (+ a margin for the sunward offsets / blurs) - big savings for strips
        mg = int(max(lit_depth * size * (2.7 if band_var else 1.1), 0.08 * size, 16 * u)) + pad
        cx0, cy0 = max(x0, -mg), max(y0, -mg)
        cx1, cy1 = min(x0 + w, self.w + mg), min(y0 + h, self.h + mg)
        if (cx0, cy0, cx1, cy1) != (x0, y0, x0 + w, y0 + h):
            m = np.ascontiguousarray(m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0])
            x0, y0 = cx0, cy0
            h, w = m.shape
        ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
        sun = self.sun if sun is None else np.array(sun, np.float32)
        sd = sun - np.array([mcx, mcy], np.float32)
        sdl = float(np.hypot(*sd)) + 1e-6
        sx, sy = sd / sdl
        if group is None:
            group = (mcx, mcy, size)
        gcx, gcy, gR = group
        # sun direction at the group level (smooth across masses)
        gd = sun - np.array([gcx, gcy], np.float32)
        gsx, gsy = gd / (float(np.hypot(*gd)) + 1e-6)

        # --- value structure --------------------------------------------------------------
        D = lit_depth * size
        far = _shift_blur(m, sx * D, sy * D, max(lit_soft * size, 1.0))
        if band_var:
            # vary the depth of the lit face along the contour (painted, not a uniform offset)
            far2 = _shift_blur(m, sx * D * 1.9, sy * D * 1.9, max(lit_soft * size * 1.5, 1.0))
            nv = self._tex(self.tex, x0, y0, w, h, xs, ys)
            k = _ss(-0.15, 0.2, nv)
            far = far * (1 - k * band_var) + far2 * k * band_var
        band = m * (1 - far)                                                  # lit face (echoes outline)
        Ds = 0.045 * size
        hotb = m * (1 - _shift_blur(m, sx * Ds, sy * Ds, max(0.02 * size, 1.0)))
        lin = ((xs - gcx) * gsx + (ys - gcy) * gsy) / max(gR, 1.0)            # -1..1 across the cloud
        lin = _ss(-0.75, 0.75, lin)
        vpos = np.clip((ys - ytop) / max(yb - ytop, 1.0), 0, 1)               # 0 top .. 1 base
        raw = (band_w * band + (1 - band_w) * lin) * light - base_dark * _ss(0.35, 1.0, vpos) * (1 - 0.5 * band)
        raw = raw + flat_shade + 0.12 * (1 - vpos) - 0.06
        # painterly irregularity of the terminator: anisotropic 'brush' noise sampled in plate space
        br = self._tex(self.brush, x0, y0, w, h, xs, ys)
        raw = raw + brush * br
        t0, t1 = two_tone
        c0 = 0.5 * (t0 + t1)
        # 2-tone split: a firm painted edge plus a soft falloff around it
        v = firm * _ss(c0 - 0.04, c0 + 0.04, raw) + (1 - firm) * _ss(t0 - 0.08, t1 + 0.08, raw)
        v2 = _ss(-0.25, t0 + 0.05, raw)                                        # shade -> deep inside shadow

        lit, mid, shade, deep, hi = pal['lit'], pal['mid'], pal['shade'], pal['deep'], pal['hi']
        # (all maps below are single-channel; the 3-channel colour mixing + compositing runs in one
        #  numba kernel `_paint_px` - see there for the colour recipe)
        # shadow face: lighter (skylight) high up, sinking to the deep colour toward the base / valley
        dk = np.clip((1 - v2) * 0.6 + _ss(valley_span[0], valley_span[1], vpos) * valley_dark, 0, 1)
        # lit face: hottest right at the sunward contour, cooling toward the terminator and away from
        # the sun (a painted gradient inside the flat lit shape)
        lw = np.clip(_ss(0.55, 1.0, band) * 0.65 + _ss(0.3, 1.0, lin) * 0.35, 0, 1)
        lit_lo = (lit * np.array([0.965, 0.97, 0.99], np.float32)) * (1 - lit_cool) + mid * lit_cool
        # terminator gets the saturated mid colour
        tm = np.exp(-((v - 0.45) / 0.28) ** 2) * 0.55
        hotk = hotb * (hot * light) * _ss(0.3, 0.9, v)
        # echo: a faint painted second contour inside the shadow (upper florets catch skylight)
        if echo:
            tz = m * (1 - _blur(_shift(m, 0, -echo * size), 1.5 * u)) * (1 - v) * echo_amt
        else:
            tz = np.zeros_like(m)
        # aerial perspective
        hz = np.clip(haze + haze_grad * _ss(0.2, 1.0, vpos), 0, 1)
        # watercolour edge: faint pigment line inside the shadow-side edge + cool skylight lift
        e = np.clip(m - _blur(m, 2.5 * u), 0, 1) * 2.0
        e2 = np.clip(m - _blur(m, 9.0 * u), 0, 1)
        # rim (silver lining): thin stroke on the sunward contour, hotter near a backlighting sun
        near = None
        if rim:
            r = rim_px * u
            rimm = np.clip(m * (1 - _shift(m, sx * r, sy * r)) * 1.4, 0, 1)
            rw = rim * (0.15 + 0.85 * v ** 1.5)
            if backlit:
                near = np.exp(-np.hypot(xs - sun[0], ys - sun[1]) / max(size, 1) / 0.6)
                rw = rw * (1 + 1.5 * backlit * near)
            rimk = np.clip(rimm * rw, 0, 1)
        else:
            rimk = np.zeros_like(m)
        if backlit:
            r2 = 10 * u
            glow = m * (1 - _blur(_shift(m, sx * r2, sy * r2), 4 * u)) * backlit * near
        else:
            glow = np.zeros_like(m)

        # --- alpha: crisp on the lit side, lost on the shadow side --------------------------
        soft = _blur(m, max(lost * 0.02 * size, 1.0)) if lost else m
        crisp_w = _ss(0.15, 0.65, _blur(v, 3 * u))
        a = m * crisp_w + np.minimum(soft, 1.0) * (1 - crisp_w)
        if backlit:
            # thin edges near the sun are translucent (light passes through)
            thin = np.clip(m - _blur(m, 6 * u), 0, 1) * 2
            a = a * (1 - 0.35 * backlit * thin * np.exp(-np.hypot(xs - sun[0], ys - sun[1]) / max(size, 1) / 0.5))
        if fray is not None:
            # fibrous, torn edge (anvil tips): erode alpha with streaky noise over x in [fx0 -> fx1]
            fx0, fx1, fs = fray
            fr = _ss(fx0, fx1, xs)
            fn = self._tex(self.streak, x0, y0, w, h, xs, ys)
            dist = np.clip(_blur(m, 0.04 * size), 0, 1)             # 0.5 at the edge -> 1 deep inside
            thr = fr * fs
            a = a * _ss(thr - 0.15, thr + 0.1, dist * 0.55 + fn * 0.6)
            a = a * (1 - 0.35 * fr)
        a = np.clip(a * alpha, 0, 1)

        # --- paste into plate -------------------------------------------------------------
        sl = (slice(py0 - y0, py1 - y0), slice(px0 - x0, px1 - x0))
        tp = self.tp
        tx = 0.035 * self.tex[py0 + tp:py1 + tp, px0 + tp:px1 + tp] + 0.015 * self.tex2[py0 + tp:py1 + tp, px0 + tp:px1 + tp]
        if cast:
            # this mass shades the paint behind it, displaced away from the sun
            cd = 0.06 * size
            ck = _shift_blur(m, sx * cd, sy * cd, 0.03 * size)[sl] * cast
        else:
            ck = np.zeros((py1 - py0, px1 - px0), np.float32)
        palm = np.stack([lit_lo, hi, mid, shade, deep, pal['haze'], pal['rim'], lit]).astype(np.float32)
        f32 = lambda q: np.ascontiguousarray(q[sl], dtype=np.float32)
        _paint_px(self.prem, py0, px0, f32(v), f32(dk), f32(lw), f32(tm), f32(hotk), f32(tz), f32(hz), f32(e),
                  f32(e2), f32(rimk), f32(glow), f32(a), np.ascontiguousarray(tx, dtype=np.float32),
                  np.ascontiguousarray(ck, dtype=np.float32), palm, np.float32(pal['edge_dark']))
        return dict(mask=m, x0=x0, y0=y0, cx=mcx, cy=mcy)

    # ------------------------------------------------------------------ fringes / wisps
    def wisps(self, x0, x1, base_y, depth, rng=None, pal=None, amount=0.5, streak=2.5, color_t=0.35,
              erode=0.6, seed=0, angle=0.0):
        """Torn wispy base: erode the painted alpha near the base with warped noise and add thin
        horizontal alpha-faded streaks below it.
        x0, x1 - horizontal extent (px); base_y - cloud base (px); depth - vertical extent of the
        fringe (px); amount - wisp opacity; streak - horizontal elongation; color_t - wisp colour between
        deep (0) and shade (1)...(haze beyond); erode - how much the base is eaten away."""
        pal = self.pal if pal is None else palette(pal)
        u = self.u
        X0 = int(max(x0 - depth, 0))
        X1 = int(min(x1 + depth, self.w))
        Y0 = int(max(base_y - depth * 1.2, 0))
        Y1 = int(min(base_y + depth * 1.4, self.h))
        if X1 <= X0 or Y1 <= Y0:
            return
        w, h = X1 - X0, Y1 - Y0
        ys, xs = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        n1 = _noise(w, h, max(w / (depth * 1.5), 2.0), seed + 1, 4, stretch=streak)
        n2 = _noise(w, h, max(w / (depth * 0.5), 3.0), seed + 2, 3, stretch=streak * 1.5)
        dy = (ys - base_y) / depth                       # 0 at base, + below
        region = self.prem[Y0:Y1, X0:X1]
        a0 = region[..., 3].copy()
        # torn base edge: a ragged 1D profile (no holes), soft fade below it
        n1d = _noise(w, 8, max(w / (depth * 1.6), 2.0), seed + 4, 4)[4]
        n1d2 = _noise(w, 8, max(w / (depth * 0.5), 3.0), seed + 5, 3)[4]
        edge = base_y - depth * erode * (0.1 + 0.7 * n1d + 0.3 * n1d2)
        soft = max(depth * 0.18, 1.5)
        keep = _ss(edge[None, :] + soft, edge[None, :] - soft * 0.3, ys)
        region *= keep[..., None]
        # wisps hanging below: thin torn horizontal strands under the parts of the base that exist
        band = (dy > -0.6) & (dy < 0.05)
        prof = (a0 * band).max(0) if band.any() else np.zeros(w, np.float32)
        prof = cv2.GaussianBlur(prof[None, :].astype(np.float32), (0, 0), sigmaX=max(depth * 0.4, 1), sigmaY=0.1)[0]
        n3 = _noise(w, h, max(w / (depth * 0.8), 3.0), seed + 7, 3, stretch=streak * 2.5)
        st = _ss(0.5, 0.85, n3 * 0.55 + n2 * 0.25 + n1 * 0.2)
        fade = _ss(1.0, 0.1, dy) * _ss(-0.3, 0.15, dy)
        wa = np.clip(prof[None, :] * st * fade * amount, 0, 1)
        colw = pal['deep'] * (1 - color_t) + pal['shade'] * color_t
        colw = colw * 0.55 + pal['haze'] * 0.45
        region[..., :3] = region[..., :3] + colw * wa[..., None] * (1 - region[..., 3:4])
        region[..., 3] = region[..., 3] + wa * (1 - region[..., 3])

    def haze_band(self, y0, y1, color, amount):
        """Blend the painted plate toward `color` between rows y0..y1 (atmospheric haze toward the base)."""
        ys = np.arange(self.h, dtype=np.float32)
        k = (_ss(y0, y1, ys) * amount)[:, None, None]
        a = self.prem[..., 3:4]
        self.prem[..., :3] = self.prem[..., :3] * (1 - k) + np.asarray(color, np.float32) * a * k

    def rgba(self):
        a = self.prem[..., 3:4]
        rgb = self.prem[..., :3] / np.maximum(a, 1e-5)
        return np.concatenate([rgb, np.clip(a, 0, 1)], -1).astype(np.float32)


# ----------------------------------------------------------------------------------------- clouds


def cumulonimbus(P, cx, base_y, Hc, seed=0, anvil=True, pal=None, anvil_dir=1.0, sun=None, haze=0.0,
                 backlit=0.0, width=1.0):
    """Paint a towering summer cumulonimbus into Painter P (back to front, ~9 flat masses).

    cx, base_y - foot centre (px); Hc - tower height to the top of the anvil (px); anvil_dir - +1 anvil
    spreads right, -1 left; width - width multiplier; haze - extra aerial haze; backlit - sun behind."""
    rng = np.random.default_rng(seed)
    wd = width
    g = (cx + 0.05 * Hc, base_y - 0.5 * Hc, 0.7 * Hc)
    top = base_y - Hc
    L = []   # (envelope kwargs, shape kwargs), back to front
    if anvil:
        # anvil: flat-topped, spreading downwind, thin tips; florets only near the tower top
        ap = anvil_poly(cx + 0.08 * Hc, top + 0.1 * Hc, (0.36 if anvil_dir > 0 else 0.75) * Hc * wd,
                        (0.75 if anvil_dir > 0 else 0.36) * Hc * wd, 0.3 * Hc, rng, dome=0.5, droop=0.35, tip=0.12)
        L.append((ap, dict(levels=((0.07, 0.16, 0.85, 1.3, 2.4, 0.15, 0.5), (0.025, 0.05, 0.8), (0.01, 0.02, 0.7)),
                           size=0.32 * Hc, lit_depth=0.4, lost=0.4, haze=0.08, base_dark=0.45, side_scale=0.55,
                           down_cut=0.0, band_w=0.6, clump=0.7,
                           fray=(cx + anvil_dir * 0.25 * Hc, cx + anvil_dir * 0.85 * Hc, 0.9))))
    # crown: the tallest heads, punching into the anvil
    L.append((dict(cx=cx + 0.07 * Hc, base_y=base_y - 0.56 * Hc, width=0.4 * Hc * wd, height=0.42 * Hc,
                   power=2.4, lump=0.1, lean=0.1), dict(size=0.34 * Hc)))
    L.append((dict(cx=cx + 0.22 * Hc * wd, base_y=base_y - 0.46 * Hc, width=0.3 * Hc * wd, height=0.28 * Hc,
                   power=2.2, lump=0.1, lean=0.05), dict(size=0.26 * Hc, cast=0.3)))
    # column
    L.append((dict(cx=cx - 0.02 * Hc, base_y=base_y - 0.24 * Hc, width=0.52 * Hc * wd, height=0.52 * Hc,
                   power=3.0, lump=0.07, lean=0.08, skew=0.2), dict(size=0.36 * Hc, cast=0.4)))
    L.append((dict(cx=cx - 0.24 * Hc * wd, base_y=base_y - 0.14 * Hc, width=0.36 * Hc * wd, height=0.36 * Hc,
                   power=2.4, lump=0.1, lean=0.04), dict(size=0.28 * Hc, cast=0.35)))
    L.append((dict(cx=cx + 0.3 * Hc * wd, base_y=base_y - 0.12 * Hc, width=0.36 * Hc * wd, height=0.3 * Hc,
                   power=2.3, lump=0.1, lean=0.06), dict(size=0.26 * Hc, cast=0.35)))
    # broad body and foot
    L.append((dict(cx=cx + 0.04 * Hc, base_y=base_y - 0.02 * Hc, width=0.95 * Hc * wd, height=0.26 * Hc,
                   power=2.8, lump=0.06, skew=0.3), dict(size=0.24 * Hc, cast=0.4, haze_grad=0.3, light=0.8,
                                                         haze=0.06)))
    L.append((dict(cx=cx - 0.3 * Hc * wd, base_y=base_y + 0.01 * Hc, width=0.42 * Hc * wd, height=0.15 * Hc,
                   power=2.4, lump=0.1), dict(size=0.16 * Hc, cast=0.35, haze_grad=0.35, light=0.8, haze=0.1)))
    L.append((dict(cx=cx + 0.36 * Hc * wd, base_y=base_y + 0.015 * Hc, width=0.34 * Hc * wd, height=0.13 * Hc,
                   power=2.3, lump=0.1), dict(size=0.14 * Hc, cast=0.35, haze_grad=0.35, light=0.85, haze=0.1)))
    info = []
    for ek, sk in L:
        poly = ek if isinstance(ek, np.ndarray) else envelope(rng=rng, **ek)
        sk = dict(sk)
        sk.setdefault('group', g)
        sk.setdefault('pal', pal)
        sk.setdefault('backlit', backlit)
        sk.setdefault('echo', 0.12)
        sk['haze'] = sk.get('haze', 0.0) + haze
        info.append(P.shape(poly, rng=rng, sun=sun, **sk))
    P.wisps(cx - 0.55 * Hc * wd, cx + 0.55 * Hc * wd, base_y + 0.02 * Hc, 0.05 * Hc, rng=rng, pal=pal,
            seed=seed + 11, amount=0.4)
    return info


def cumulus(P, cx, base_y, w, h, seed=0, pal=None, n=3, haze=0.0, sun=None, lit_depth=0.4, wisps=True,
            levels=None, lost=0.3, backlit=0.0):
    """Fair-weather cumulus: n overlapping flat masses (biggest in the middle/back)."""
    rng = np.random.default_rng(seed)
    g = (cx, base_y - 0.5 * h, max(w, h) * 0.6)
    offs = sorted([(rng.uniform(-0.3, 0.3), rng.uniform(0.55, 1.0)) for _ in range(n)], key=lambda q: -q[1])
    for i, (ox, sc) in enumerate(offs):
        ww = w * (0.45 + 0.4 * sc)
        hh = h * sc
        by = base_y + (i / max(n - 1, 1)) * 0.06 * h
        poly = envelope(cx + ox * w, by, ww, hh, rng, power=rng.uniform(2.1, 2.7), lump=0.08,
                        lean=rng.uniform(-0.1, 0.15), skew=rng.uniform(-0.4, 0.4))
        kw = {}
        if levels is not None:
            kw['levels'] = levels
        P.shape(poly, rng=rng, size=hh, group=g, pal=pal, haze=haze, haze_grad=0.2, lit_depth=lit_depth,
                cast=0.3 if i else 0.0, sun=sun, lost=lost, backlit=backlit, **kw)
    if wisps:
        P.wisps(cx - w * 0.55, cx + w * 0.55, base_y + 0.04 * h, 0.14 * h, rng=rng, pal=pal, seed=seed + 3,
                amount=0.3)


def bank(P, y, x0, x1, height, seed=0, pal=None, haze=0.3, rows=2, sun=None, levels=None, lost=0.6):
    """Distant cumulus bank along the horizon seen from the side: `rows` sub-rows of flattened, heavily
    overlapping billows of power-law widths (back rows taller and hazier). y = bank base (px), height =
    typical billow height (px)."""
    rng = np.random.default_rng(seed)
    lv = levels or ((0.1, 0.22, 0.9, 1.4, 2.4, 0.1, 0.45), (0.04, 0.08, 0.85), (0.015, 0.03, 0.7))
    for r in range(rows):
        f = r / max(rows - 1, 1)                  # 0 back .. 1 front
        x = x0 - rng.uniform(0, 1) * height
        while x < x1 + height:
            hh = height * (1.25 - 0.45 * f) * (0.35 + 1.0 * rng.random() ** 2.2)
            ww = hh * rng.uniform(1.8, 3.6)
            poly = envelope(x + ww / 2, y + height * 0.25 * f, ww, hh, rng, power=rng.uniform(2.0, 2.6),
                            lump=0.1, skew=rng.uniform(-0.6, 0.6), base_round=0.0)
            P.shape(poly, rng=rng, size=hh * 1.1, pal=pal, haze=haze * (1 - 0.35 * f), haze_grad=0.35,
                    levels=lv, lost=lost, cast=0.3, sun=sun, side_scale=0.6, rim_px=2.0)
            x += ww * rng.uniform(0.35, 0.75)


def billow_strip(x0, x1, base_y, bump_w, bump_h, rng, depth=None, n=None, power=2.2, taper=0.25):
    """Polygon of a long strip of merged billows (one row segment of a sea of clouds): a scalloped top
    made of domes of random width/height (the upper envelope of their union), a flat-ish bottom
    `depth` px below base_y (hidden by the next row), ends tapering down over `taper` of the length."""
    L = x1 - x0
    depth = bump_h * 0.8 if depth is None else depth
    n = n or max(int(L / max(bump_w, 1) * 24), 48)
    xs = np.linspace(x0, x1, n)
    top = np.full(n, base_y + depth * 0.0, np.float32)
    x = x0 - bump_w * rng.uniform(0.0, 0.5)
    while x < x1 + bump_w * 0.5:
        bw = bump_w * rng.uniform(0.55, 1.5)
        bh = bump_h * rng.uniform(0.45, 1.25) * (bw / bump_w) ** 0.5
        d = np.clip(1 - ((xs - x) / (bw / 2)) ** 2, 0, 1)
        top = np.minimum(top, base_y - bh * d ** (1.0 / power))
        x += bw * rng.uniform(0.45, 0.8)
    # taper the ends down to the base
    e = np.minimum((xs - x0) / (L * taper + 1e-6), (x1 - xs) / (L * taper + 1e-6))
    e = np.clip(e, 0, 1)
    e = e * e * (3 - 2 * e)
    mid = base_y + depth * 0.5
    re = np.sqrt(np.clip(1 - (1 - e) ** 2, 0, 1))        # round cap
    top = mid - (mid - top) * re
    upper = np.stack([xs, top], 1)
    bot = mid + depth * 0.5 * re
    lower = np.stack([xs[::-1], bot[::-1]], 1)
    return np.concatenate([upper, lower], 0).astype(np.float32)


def cloud_sea(P, horizon_y, bottom_y, seed=0, rows=14, pal_near='sunrise', pal_far=None, sun=None,
              x0=0.0, x1=None, persp=2.2, near_w=0.3, far_w=0.03, aspect_near=0.42, aspect_far=0.2,
              haze_far=0.75, row_callback=None, valley=0.6, lit_depth=0.12, seg_len=(2.5, 5.0), rim_near=3.0):
    """Sea of clouds seen from above: `rows` rows from the horizon (small, flat, hazy) to the bottom of
    the plate (big, tall, detailed), painted back to front. Row spacing follows a perspective curve so
    the sea flattens toward the horizon. Each row is a few long overlapping 'billow strips' (scalloped
    merged domes + cauliflower florets) instead of individual puffs - one continuous undulating
    surface, not rows of marshmallows.

    P may be a Painter or a function row_index -> Painter (to split rows into parallax plates).
    near_w/far_w - billow (bump) width as a fraction of plate width at the front / horizon;
    aspect_* - bump height / width; lit_depth - how far the lit face reaches down from each sunward
    contour (fraction of the bump height: small = low sun, thin gold tops); valley - darkness of the
    lower part of each strip (indigo-violet valleys); row_callback(i, y) after each row (insert towers)."""
    rng = np.random.default_rng(seed)
    get = P if callable(P) else (lambda i: P)
    Pw = get(0).w
    x1 = Pw if x1 is None else x1
    pal_far = pal_near if pal_far is None else pal_far
    for i in range(rows):
        P = get(i)
        f = i / max(rows - 1, 1)                      # 0 far .. 1 near
        z = f ** persp
        y = horizon_y + (bottom_y - horizon_y) * z
        bw_px = (far_w + (near_w - far_w) * z) * Pw
        asp = aspect_far + (aspect_near - aspect_far) * f
        bh = bw_px * asp
        pal = mix_palette(pal_far, pal_near, _ss(0.05, 0.75, f))
        hz = haze_far * (1 - _ss(0.0, 1.0, f)) ** 0.9
        if f < 0.3:
            lv = ((0.05, 0.12, 0.8, 1.2, 2.2), (0.02, 0.04, 0.75))
        elif f < 0.65:
            lv = ((0.08, 0.18, 0.85, 1.3, 2.4, 0.1, 0.45), (0.03, 0.07, 0.8), (0.012, 0.025, 0.7))
        else:
            lv = DEFAULT_LEVELS
        # row segments
        x = x0 - bw_px * rng.uniform(0.5, 1.5)
        segs = []
        while x < x1 + bw_px:
            L = bw_px * rng.uniform(*seg_len) * (1 + 0.8 * (1 - f))
            segs.append((x, x + L, y + rng.uniform(-0.35, 0.35) * bh))
            x += L * rng.uniform(0.55, 0.8)
        rng.shuffle(segs)
        for (sx0, sx1, by) in segs:
            poly = billow_strip(sx0, sx1, by, bw_px, bh, rng, depth=bh * 1.2)
            if poly[:, 1].min() - bh * 0.4 > P.h:
                continue                                  # entirely below the plate
            cxs = 0.5 * (sx0 + sx1)
            P.shape(poly, rng=rng, size=bh * 1.4, pal=pal, haze=hz, haze_grad=0.0, group=(cxs, by, (sx1 - sx0)),
                    lit_depth=lit_depth * rng.uniform(0.7, 1.5), lit_soft=0.03, band_w=0.85, firm=0.85,
                    valley_dark=0.85, lit_cool=0.4, valley_span=(0.0, 0.5), echo=0.22, echo_amt=0.45,
                    light=rng.uniform(0.85, 1.1), levels=lv, base_dark=valley,
                    lost=0.2 + 0.6 * (1 - f), rim=1.0, rim_px=1.2 + (rim_near - 1.2) * f, sun=sun,
                    side_scale=0.7, down_cut=0.1, two_tone=(0.36, 0.56), cast=0.3, brush=0.1, band_var=0.55)
        if row_callback is not None:
            row_callback(i, y)
