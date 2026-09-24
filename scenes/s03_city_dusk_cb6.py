"""Hero cumulonimbus + far bank for s03_city_dusk (round 7), painted with the shared lib.clouds2 module.

Round-7 art direction (fixing round 6's 'flat vector cut-out' read):
  * 4 big painted value masses (hot pink-gold lit face toward the afterglow at centre-left, magenta mid,
    cool violet shadow, a deeper indigo core low down) + 2 small forward knots. Interiors stay large and
    smooth; the few crisp overlapping-mass edges come from the masses themselves (no inner lobes, no
    skylight steps, no shadow-side wrap -> no stacked scallops inside the violet);
  * fine, irregular cauliflower breakup on the SILHOUETTE only (5 floret levels, low clump);
  * lit face pushed to a hot pink-gold that goes near-white on the sun-facing crowns; a firm lit-on-lit
    step instead of a soft seam between lit and magenta;
  * a crisp 2-4 px hot-gold lining ONLY on the sun-facing (left / lower-left) silhouette, no glow halo;
    the shadow side (right) keeps at most a dim cool sky-bounce edge;
  * a torn, hazy base: the lowest ~20 % is eaten along wind-torn streaks and fades into the sky colour
    behind the skyline (no solid violet block behind the towers).
"""
import numpy as np
import cv2

from lib import core as C, clouds2 as K2

PAL = dict(hi=(1.0, 0.9, 0.74), lit=(1.0, 0.6, 0.5), lit_lo='#e8567e', mid='#c2367a', shade='#6c5abb',
           deep='#3b3187', refl='#7c6cc8', bounce='#b25c9c', rim=(1.7, 0.86, 0.36), haze='#9a84cc',
           edge_dark=0.0)
PAL_FAR = dict(hi=(1.0, 0.78, 0.7), lit='#f0909e', lit_lo='#d97aa6', mid='#b862a4', shade='#7466b8',
               deep='#564a9e', refl='#8274c2', bounce='#a868a6', rim=(1.3, 0.9, 0.7), haze='#9488cc',
               edge_dark=0.0)

# silhouette cauliflower: big heads, florets, fine florets, micro breakup (edge only)
LEVELS = ((0.14, 0.3, 0.9, 1.4, 2.6, 0.05, 0.4), (0.05, 0.11, 0.95, 0.9, 1.9),
          (0.022, 0.045, 0.95, 0.9, 1.9, 0.25, 0.55), (0.011, 0.02, 0.85, 1.0, 2.2, 0.2, 0.5),
          (0.005, 0.01, 0.6, 1.1, 2.6, 0.15, 0.45))


def _tower(P, X, Y, H, seed, form=0.4, sun_z=0.5):
    rng = np.random.default_rng(seed)
    cx, base, Hc = X(0.84), Y(0.575), 0.48 * H
    g = (cx - 0.02 * Hc, base - 0.5 * Hc, 0.42 * Hc, 0.56 * Hc)
    common = dict(group=g, form=form, sun_z=sun_z, wrap=0.0, term_w=0.07, rim_interior=0.0, side_scale=0.85,
                  term_noise=0.25, inner=0, sky=0.08, halo=0.0, rim_px=2.6, hot=0.85, lit_step=0.45,
                  scallop=1.1, levels=LEVELS, clump=0.22, firm=1.0, soften=0.05, rim_shadow=0.08,
                  shade_top=0.2, valley_dark=0.25, refl=0.35, bounce=0.2, sun_bias=0.25, concave=0.4)
    masses = [
        # A  back crown: the tallest head, leaning left (updraft sheared by the wind)
        (dict(cx=cx + 0.05 * Hc, base_y=base - 0.5 * Hc, width=0.38 * Hc, height=0.5 * Hc, power=2.3,
              lump=0.13, lean=-0.07, base_round=0.3), dict(size=0.32 * Hc, cast=0.0)),
        # B  shadow-side shoulder on the right (violet, cauliflower edge against the sky)
        (dict(cx=cx + 0.3 * Hc, base_y=base - 0.14 * Hc, width=0.4 * Hc, height=0.42 * Hc, power=2.2,
              lump=0.14, lean=0.05, base_round=0.3), dict(size=0.28 * Hc, cast=0.0, light=0.85)),
        # C  sunward shoulder: lower-left head catching the full afterglow (crisp lit-over-shadow edge)
        (dict(cx=cx - 0.2 * Hc, base_y=base - 0.22 * Hc, width=0.36 * Hc, height=0.42 * Hc, power=2.3,
              lump=0.12, lean=-0.05, base_round=0.3), dict(size=0.28 * Hc, cast=0.35)),
        # D  main body in front, wide
        (dict(cx=cx + 0.06 * Hc, base_y=base - 0.02 * Hc, width=0.62 * Hc, height=0.44 * Hc, power=2.6,
              lump=0.1, skew=0.15, base_round=0.15), dict(size=0.32 * Hc, cast=0.4)),
        # small forward knots low on each flank
        (dict(cx=cx - 0.36 * Hc, base_y=base + 0.03 * Hc, width=0.3 * Hc, height=0.17 * Hc, power=2.4,
              lump=0.1, base_round=0.12), dict(size=0.15 * Hc, cast=0.0, haze=0.05)),
        (dict(cx=cx + 0.42 * Hc, base_y=base + 0.04 * Hc, width=0.28 * Hc, height=0.16 * Hc, power=2.3,
              lump=0.12, base_round=0.12), dict(size=0.14 * Hc, cast=0.0, haze=0.06, light=0.8)),
    ]
    for ek, sk in masses:
        poly = K2.envelope(rng=rng, **ek)
        kw = dict(common)
        kw.update(sk)
        P.shape(poly, rng=rng, pal=PAL, **kw)
    return cx, base, Hc


def _torn_base(tower, X, Y, H, sky, seed):
    """Tear the lowest part of the tower along wind-streaks and dissolve it into the sky colour."""
    ph, pw = tower.shape[:2]
    ys = np.arange(ph, dtype=np.float32)[:, None]
    rng_n = K2._noise(pw, ph, 18.0, seed + 31, 4, stretch=6.0)
    rng_f = K2._noise(pw, ph, 60.0, seed + 37, 2, stretch=8.0)
    prof = K2._noise(pw, 8, 9.0, seed + 41, 3)[4][None, :]
    # the cut line wanders along x: lobes of base hang lower in places, deep tears elsewhere
    y_cut = Y(0.47) + (prof - 0.5) * 0.09 * H
    d = (ys - y_cut) / (0.07 * H)
    n = 0.65 * rng_n + 0.35 * rng_f
    keep = C.smoothstep(0.55, -0.35, d + 0.9 * (n - 0.5) * 2.0 * C.smoothstep(-1.2, 0.4, d))
    keep = np.clip(keep, 0, 1).astype(np.float32)
    a = tower[..., 3] * keep
    # fading part takes the sky's colour (aerial haze at the horizon, the sky shows through)
    fz = C.smoothstep(Y(0.36), Y(0.56), ys)[..., None] * 0.6
    if sky is not None:
        hz = cv2.GaussianBlur(sky, (0, 0), 0.01 * H)[..., :3] * 0.8 + np.array([0.55, 0.42, 0.78], np.float32) * 0.2
    else:
        hz = np.array([0.62, 0.46, 0.8], np.float32)
    tower[..., :3] = tower[..., :3] * (1 - fz) + hz * fz
    tower[..., 3] = a
    return tower


def paint(pw, ph, W, H, ox, sun, seed=5, sky=None, form=0.4, sun_z=0.5, sun_y=0.33):
    """Straight-alpha RGBA plates (ph, pw, 4): (tower, bank). Frame fractions offset by ox px."""
    X = lambda fx: fx * W + ox
    Y = lambda fy: fy * H
    u = W / 1920.0
    sun_l = (float(sun[0]), Y(sun_y))
    P = K2.Painter(pw, ph, sun_l, PAL, u, seed=seed + 1, sun_z=sun_z)
    cx, base, Hc = _tower(P, X, Y, H, seed, form=form, sun_z=sun_z)
    top = base - Hc
    # thin sheared anvil shelf streaming downwind (right) from the crown, fraying into fibres
    P.fibres(cx + 0.05 * Hc, X(1.02), top + 0.07 * H, 0.018 * H, direction=1.0, pal=PAL, amount=0.6,
             seed=seed + 21, lit=0.7)
    P.fibres(cx + 0.1 * Hc, X(0.98), top + 0.1 * H, 0.01 * H, direction=1.0, pal=PAL, amount=0.35,
             seed=seed + 23, lit=0.5)
    # hot-gold lining on the sun-facing silhouette only (no halo glow)
    P.lining(1.6, rim_px=2.6, halo=0.0)
    tower = P.rgba()
    tower = _torn_base(tower, X, Y, H, sky, seed)
    # ------------------------------------------------------------------ low far bank behind the skyline
    Pb = K2.Painter(pw, ph, (float(sun[0]), float(sun[1])), PAL_FAR, u, seed=seed + 5, sun_z=0.05)
    for i, (fx, fy, cw, ch) in enumerate(((0.64, 0.585, 0.16, 0.07), (0.98, 0.59, 0.3, 0.14),
                                         (1.1, 0.59, 0.18, 0.09), (0.76, 0.59, 0.14, 0.06))):
        K2.cumulus(Pb, X(fx), Y(fy), cw * W, ch * H, seed=seed * 10 + i, pal=PAL_FAR, n=3, haze=0.3)
    Pb.lining(0.8, rim_px=2.2, halo=0.0)
    Pb.haze_band(Y(0.47), Y(0.6), np.array([0.62, 0.5, 0.84], np.float32), 0.55)
    bank = Pb.rgba()
    return tower.astype(np.float32), bank.astype(np.float32)


def bloom_src(img, cloud_a, k=0.6):
    """Bloom SOURCE with the painted cloud interior held down (the lit face must not bloom into an
    airbrushed glow outline); the outer ~5 px (the HDR gold lining) keeps its full value."""
    a = np.ascontiguousarray(cloud_a, dtype=np.float32)
    a = cv2.GaussianBlur(cv2.erode(a, np.ones((11, 11), np.uint8)), (0, 0), 2.0)[..., None]
    return img * (1 - k * np.clip(a, 0, 1))
