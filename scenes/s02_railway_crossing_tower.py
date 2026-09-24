"""s02 helper: the hero cumulonimbus, painted with lib/clouds2 (Painter, mass by mass).

Summer noon, sun high up-right of the tower. Layout (left of the vanishing point, behind the groves):
a broad hazy base that tears into wisps over the treeline, a stout column of big overlapping masses
stepping up and slightly right, a crown of cauliflower heads, and a sheared anvil that grows OUT of the
crown (no separate cap) and streams downwind to the right, thinning into fibres.

Light: one key high up-right (so the terminator runs under every bulge - curved, form-following - rather
than straight down the tower); warm cream lit tops and sun-side flanks with near-white peaks, a firm
scalloped terminator, cool lavender-blue shadow that is deepest under the crown heads and lifts to a
bluish reflected light low down. Lit lobes wrap round onto the shadow side (lit-on-shade overlaps), and
the sun-facing silhouette carries a crisp 2-4 px silver/gold lining with a soft halation. The lower
third dissolves into the horizon haze (aerial perspective)."""
import numpy as np
import cv2

from lib import clouds2 as K

# lit: warm cream -> near-white peak; mid lavender-grey terminator band; shadow: cool lavender-blue,
# lighter bluish reflected light low down
PAL = dict(hi=(0.99, 0.975, 0.945), lit='#f4e8d6', lit_lo='#dcd8e2', mid='#c0b6cf', shade='#8191c9',
           deep='#6f80bd', refl='#a0aedb', bounce='#cbc7d6', rim=(1.6, 1.5, 1.2), haze='#cfe6f7',
           edge_dark=0.0)
PAL_BASE = dict(PAL, lit='#ece6e4', lit_lo='#d4d6e2', hi=(0.96, 0.955, 0.945), shade='#95a2d0',
                deep='#8594c6', refl='#adbadf', bounce='#cdcbd8')
PAL_ANVIL = dict(PAL, lit='#f1ebe2', lit_lo='#dfe0ea', mid='#cdcbdc', shade='#98a6d8', deep='#8797cf',
                 refl='#b2bfe6')
LEV_BIG = ((0.14, 0.4, 0.95, 1.2, 2.4, 0.12, 0.45), (0.05, 0.13, 0.85, 1.0, 2.2, 0.3, 0.6),
           (0.02, 0.045, 0.6, 1.2, 3.0), (0.012, 0.02, 0.2, 1.6, 4.0))


def _sun_rim(rgba, key_dir, W, top_y, base_y, strength=1.0):
    """Silver/gold lining where the outer silhouette faces the key light (upper, sun-facing flank and the
    crown / anvil top). Lower and shadow-side edges meet the sky softly."""
    u = W / 1920.0
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    Ab = cv2.GaussianBlur(A, (0, 0), 4 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gl, -gy / gl                                   # outward normal
    h, w = A.shape
    ys = np.arange(h, dtype=np.float32)[:, None]
    Hc = base_y - top_y
    kd = np.asarray(key_dir, np.float32)
    kd = kd / float(np.hypot(*kd))
    face = np.clip(nx * kd[0] + ny * kd[1], 0, 1) ** 1.3
    vz = 1 - K._ss(top_y + 0.45 * Hc, top_y + 0.8 * Hc, ys)      # fades out toward the hazy base
    k = np.clip(face * vz, 0, 1)
    rb = max(int(round(1.5 * u)), 1)
    core = np.clip(A - cv2.erode(A, np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)), 0, 1)
    rb2 = max(int(round(3.2 * u)), 1)
    wide = np.clip(A - cv2.erode(A, np.ones((2 * rb2 + 1, 2 * rb2 + 1), np.uint8)), 0, 1)
    wide = cv2.GaussianBlur(wide, (0, 0), 1.0 * u)
    ring = np.maximum(core, wide * 0.45)
    kk = np.clip(ring * k * strength, 0, 1)
    col = np.array([1.5, 1.42, 1.2], np.float32)
    rgb = rgba[..., :3] * (1 - kk[..., None]) + col * kk[..., None]
    hk = cv2.GaussianBlur(kk, (0, 0), 2.5 * u) * 0.7 + cv2.GaussianBlur(kk, (0, 0), 9 * u) * 0.45
    hk = np.clip(hk * (1 - A), 0, 1)
    a = np.clip(A + hk * 0.4, 0, 1)
    halo_col = np.array([1.1, 1.06, 0.98], np.float32)
    rgb = (rgb * A[..., None] + halo_col * hk[..., None] * 0.4) / np.maximum(a, 1e-4)[..., None]
    rgb = np.where(a[..., None] > 1e-4, rgb, rgba[..., :3])
    return np.concatenate([rgb, a[..., None]], -1).astype(np.float32), kk


def tower(pw, ph, cx, base_y, Hc, W, seed=12, haze_col='#d2e9f7', horizon_y=None):
    """Paint the hero tower on a (ph, pw) straight-alpha plate. cx / base_y: foot centre (plate px);
    Hc: height to the anvil top (px). Returns (rgba, rim_mask)."""
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    # key light: high and to the right (the real sun sits far up-right, slightly in front)
    key = (cx + 1.3 * Hc, base_y - 1.75 * Hc)
    P = K.Painter(pw, ph, key, PAL, u, seed=seed + 1, sun_z=0.26)
    by = base_y
    g = (cx + 0.06 * Hc, by - 0.5 * Hc, 0.5 * Hc, 0.62 * Hc)
    common = dict(group=g, form=0.42, term_w=0.09, wrap=0.35, brush=0.0, lit_step=0.32, term=0.5, hot=1.0,
                  sky=0.14, rim=0.0, rim_interior=0.0, split=0.3, scallop=1.15, term_noise=0.14, levels=LEV_BIG,
                  concave=0.7, clump=0.5)

    def mass(env, **kw):
        k = dict(common)
        k.update(kw)
        poly = env if isinstance(env, np.ndarray) else K.envelope(rng=rng, **env)
        return P.shape(poly, rng=rng, **k)

    shade_low = dict(pal=PAL_BASE, light=0.5, lit_bias=-0.1, form=0.4, cast=0.45, wrap=0.2, sky=0.12,
                     inner=1, inner_val=0.45, inner_size=(0.35, 0.6), shade_top=0.12, valley_dark=0.1,
                     refl=0.55, bounce=0.35)
    top = by - Hc
    # ---- anvil: grows out of the crown, blunt short upwind (left), long wedge downwind (right) that
    # tears into fibres; lit flat top, cool shaded underside
    xa = cx + 0.1 * Hc
    long_, short_ = 0.62 * Hc, 0.2 * Hc
    ap = K.anvil_poly(xa, top + 0.03 * Hc, short_, long_, 0.15 * Hc, rng, dome=0.45, rise=0.35, tip=0.08,
                      sag=0.55)
    n_up = len(ap) // 2
    xx = ap[n_up:, 0]
    ap[n_up:, 1] += 0.014 * Hc * np.sin((xx - xa) / (0.08 * Hc) + rng.uniform(0, 6.28)) * \
        np.clip(np.abs(xx - xa) / (0.2 * Hc), 0, 1)
    mass(ap, pal=PAL_ANVIL, levels=((0.05, 0.11, 0.5, 2.2, 4.5, 0.25, 0.5), (0.02, 0.04, 0.4, 1.4, 3.0)),
         size=0.2 * Hc, firm=0.7, soften=0.5, base_dark=0.8, side_scale=0.35, down_cut=0.05, clump=0.7,
         top_bias=1.8, inner=0, fray=(xa + 0.3 * long_, xa + 0.95 * long_, 1.25, 0.4 * long_), refl=0.5,
         wrap=0.0, bounce=0.2, shade_top=0.0, form=0.2, mass_r=0.45, split=0.12, scallop=0.6, sky=0.0,
         lit_bias=0.12, cast=0.0, group=(xa + 0.25 * long_, top + 0.12 * Hc, 0.7 * long_, 0.16 * Hc))
    # ---- crown: broad cauliflower cap merging up into the anvil's underside
    mass(dict(cx=cx + 0.08 * Hc, base_y=by - 0.66 * Hc, width=0.62 * Hc, height=0.26 * Hc, power=2.3, lump=0.1,
              base_round=0.4, base_wave=0.06, skew=0.15), size=0.26 * Hc, cast=0.35, side_scale=0.62,
         down_cut=0.2, lit_bias=0.06, group=(cx + 0.1 * Hc, by - 0.78 * Hc, 0.34 * Hc, 0.2 * Hc), form=0.45)
    # crown heads punching up under the anvil (left one in half shade, right one sunlit)
    mass(dict(cx=cx - 0.08 * Hc, base_y=by - 0.76 * Hc, width=0.3 * Hc, height=0.15 * Hc, power=2.2, lump=0.12,
              base_round=0.45, base_wave=0.06), size=0.17 * Hc, cast=0.4, side_scale=0.6, form=0.4)
    mass(dict(cx=cx + 0.25 * Hc, base_y=by - 0.73 * Hc, width=0.26 * Hc, height=0.15 * Hc, power=2.2, lump=0.12,
              base_round=0.45, base_wave=0.06, skew=0.2), size=0.16 * Hc, cast=0.4, side_scale=0.6, form=0.4,
         lit_bias=0.08)
    # ---- main column: stout, stepping right
    mass(dict(cx=cx + 0.02 * Hc, base_y=by - 0.3 * Hc, width=0.54 * Hc, height=0.44 * Hc, power=2.6, lump=0.09,
              lean=0.08, base_round=0.3, base_wave=0.05), size=0.32 * Hc, cast=0.4, side_scale=0.72, inner=1)
    # shadow-side (left) lobes: their tops catch light (curved terminator wrapping under each bulge)
    mass(dict(cx=cx - 0.2 * Hc, base_y=by - 0.46 * Hc, width=0.3 * Hc, height=0.18 * Hc, power=2.3, lump=0.1,
              skew=0.2, base_round=0.35, base_wave=0.08), size=0.18 * Hc, cast=0.45, side_scale=0.6, form=0.35,
         light=0.9, inner=1, inner_val=0.5, wrap=0.8, wrap_px=0.14, lit_bias=0.1)
    mass(dict(cx=cx - 0.3 * Hc, base_y=by - 0.26 * Hc, width=0.34 * Hc, height=0.22 * Hc, power=2.3, lump=0.12,
              lean=-0.04, skew=0.1), size=0.22 * Hc, cast=0.45, side_scale=0.7, form=0.35, light=0.85, inner=1,
         inner_val=0.5, wrap=0.8, wrap_px=0.13, lit_bias=0.08, group=(cx - 0.25 * Hc, by - 0.34 * Hc, 0.3 * Hc, 0.3 * Hc))
    # lit-on-lit lobes on the sunward (right) face: crisp overlapping-mass edges inside the light
    mass(dict(cx=cx + 0.24 * Hc, base_y=by - 0.46 * Hc, width=0.3 * Hc, height=0.2 * Hc, power=2.3, lump=0.1,
              skew=-0.2, base_round=0.35, base_wave=0.08), size=0.19 * Hc, cast=0.3, side_scale=0.6, form=0.4,
         lit_bias=0.1)
    mass(dict(cx=cx + 0.09 * Hc, base_y=by - 0.52 * Hc, width=0.24 * Hc, height=0.14 * Hc, power=2.2, lump=0.1,
              skew=0.1, base_round=0.3, base_wave=0.08), size=0.14 * Hc, cast=0.3, side_scale=0.6, form=0.35,
         lit_bias=0.06)
    mass(dict(cx=cx + 0.33 * Hc, base_y=by - 0.25 * Hc, width=0.3 * Hc, height=0.22 * Hc, power=2.3, lump=0.1,
              skew=-0.25, base_round=0.4, base_wave=0.08), size=0.2 * Hc, cast=0.35, side_scale=0.65, form=0.4,
         lit_bias=0.04, inner=1, inner_val=0.5)
    # ---- low left shoulder, tearing into wisps on its far side
    mass(dict(cx=cx - 0.52 * Hc, base_y=by - 0.07 * Hc, width=0.36 * Hc, height=0.2 * Hc, power=2.3, lump=0.12,
              lean=-0.05, skew=-0.2), size=0.2 * Hc, cast=0.45, side_scale=0.7, form=0.4, inner=1, light=0.9,
         fray=(cx - 0.5 * Hc, cx - 0.72 * Hc, 0.9, 0.1 * Hc), group=(cx - 0.52 * Hc, by - 0.16 * Hc, 0.22 * Hc, 0.2 * Hc))
    for (fx_, fy_, fw_, fh_) in ((-0.8, -0.14, 0.12, 0.045), (-0.74, -0.24, 0.08, 0.03)):
        mass(dict(cx=cx + fx_ * Hc, base_y=by + fy_ * Hc, width=fw_ * Hc, height=fh_ * Hc, power=2.2, lump=0.15),
             size=0.06 * Hc, form=0.3, light=0.85, alpha=0.75,
             fray=(cx + (fx_ + 0.03) * Hc, cx + (fx_ - 0.06) * Hc, 1.0, 0.05 * Hc),
             group=(cx + fx_ * Hc, by + (fy_ - 0.02) * Hc, 0.08 * Hc, 0.05 * Hc), inner=0, wrap=0.0)
    # ---- lower body: a few big overlapping masses of varied size, mostly in the tower's own shade
    mass(dict(cx=cx - 0.06 * Hc, base_y=by - 0.04 * Hc, width=0.6 * Hc, height=0.26 * Hc, power=2.4, lump=0.1,
              skew=-0.2), size=0.28 * Hc, **shade_low)
    mass(dict(cx=cx + 0.38 * Hc, base_y=by + 0.0 * Hc, width=0.38 * Hc, height=0.18 * Hc, power=2.3, lump=0.1,
              skew=0.2), size=0.2 * Hc, **dict(shade_low, light=0.75, lit_bias=0.0))
    mass(dict(cx=cx - 0.32 * Hc, base_y=by + 0.03 * Hc, width=0.4 * Hc, height=0.13 * Hc, power=2.3, lump=0.1,
              skew=-0.2), size=0.14 * Hc, **dict(shade_low, inner=0, light=0.6))
    # torn wispy underside
    P.wisps(cx - 0.8 * Hc, cx + 0.62 * Hc, by + 0.03 * Hc, 0.11 * Hc, rng=rng, pal=PAL_BASE, seed=seed + 11,
            amount=0.55, erode=1.3, base_haze=0.5, haze_color=haze_col)
    # sheared fibres beyond the anvil tip (thin, fading into the blue)
    tipx = xa + long_
    P.fibres(tipx - 0.35 * long_, tipx + 0.45 * long_, top + 0.1 * Hc, 0.03 * Hc, pal=PAL_ANVIL, seed=seed + 21,
             amount=0.5, lit=0.85)
    P.fibres(tipx - 0.2 * long_, tipx + 0.3 * long_, top + 0.15 * Hc, 0.018 * Hc, pal=PAL_ANVIL, seed=seed + 23,
             amount=0.32, lit=0.6)
    rgba = P.rgba()
    # de-stair the thresholded value edges (keeps edges crisp) and remove the faint interior mottle
    rgb = np.ascontiguousarray(rgba[..., :3])
    k_ = 5 if u > 0.75 else 3
    med = cv2.medianBlur(cv2.medianBlur(rgb, k_), k_)
    keep = ((rgb.max(-1) > 1.0) | (med.max(-1) > 1.0) | (rgba[..., 3] < 0.98))[..., None]
    rgb = np.where(keep, rgb, med)
    rgb = cv2.bilateralFilter(rgb, 0, 0.035, 3.0 * u + 0.5)
    rgba = np.concatenate([rgb, rgba[..., 3:4]], -1)
    rgba, rimk = _sun_rim(rgba, (0.75, -0.66), W, top, by)
    # aerial perspective: the lowest part of the tower dissolves toward the pale horizon haze
    ys = np.arange(ph, dtype=np.float32)[:, None]
    k = (K._ss(by - 0.45 * Hc, by + 0.02 * Hc, ys) ** 1.2)[..., None]
    rgb = rgba[..., :3]
    lum = rgb.mean(-1, keepdims=True)
    rgb = rgb * (1 - 0.3 * k) + lum * 0.3 * k
    hc = K._c(haze_col)
    rgb = rgb * (1 - 0.6 * k) + hc * (0.6 * k)
    al = rgba[..., 3:4] * (1 - 0.25 * k * K._ss(by - 0.2 * Hc, by + 0.05 * Hc, ys)[..., None])
    rgba = np.concatenate([rgb, al], -1).astype(np.float32)
    return rgba, rimk


def haze_gradient(rgba, y0, y1, color, amount, x_range=None):
    """Aerial perspective on a cloud plate: rows y0 -> y1 blend (smoothly) toward `color` by up to
    `amount`, with a slight contrast loss (optionally only inside x_range=(x0, x1, feather))."""
    h, w = rgba.shape[:2]
    ys = np.arange(h, dtype=np.float32)[:, None]
    k = K._ss(y0, y1, ys) * amount
    if x_range is not None:
        xs = np.arange(w, dtype=np.float32)[None, :]
        x0, x1, fe = x_range
        k = k * K._ss(x0 - fe, x0, xs) * (1 - K._ss(x1, x1 + fe, xs))
    k = k[..., None].astype(np.float32)
    rgb = rgba[..., :3]
    lum = rgb.mean(-1, keepdims=True)
    rgb = rgb * (1 - 0.35 * k) + lum * 0.35 * k
    rgb = rgb * (1 - k) + K._c(color) * k
    return np.concatenate([rgb, rgba[..., 3:4]], -1).astype(np.float32)
