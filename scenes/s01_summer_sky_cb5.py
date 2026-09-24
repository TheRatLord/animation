"""s01 helper (round 4): the hero cumulonimbus. Geometry (silhouette, masses) painted with lib/clouds2
exactly as in s01_summer_sky_cb3; the VALUES are then repainted as a background artist would:
a few big flat masses with high contrast - brilliant warm-white lit faces, a cool mid step and deep
blue-grey shadow planes - each shadow plane with ONE crisp edge (where a lit lobe in front overlaps it)
and one soft turning edge (toward the light); a crisp 2-4 px silver-gold lining on the sun-facing
silhouette, a lit anvil top with a cool underside and torn wisps; a torn, hazy base."""
import numpy as np
import cv2

from lib import clouds2 as K
from s01_summer_sky_cb3 import PAL, PAL_BASE, PAL_ANVIL, LEV_BIG, top_edge  # noqa: F401


def raw_tower(pw, ph, cx, base_y, Hc, W, seed=12, haze_col='#d4ebf8'):
    """Paint the tower on a (ph, pw) straight-alpha plate; cx / base_y: foot centre, Hc: height to the
    anvil top. Returns (rgba, sun) with the sun seated on the anvil's upper-left crest."""
    u = W / 1920.0
    WF = 0.82                       # width factor: a tall, narrow rising column
    rng = np.random.default_rng(seed)
    by = base_y
    top = by - Hc
    # key light: high up-left, a little in front of the tower (the real sun is behind the crest)
    key = (cx - 0.9 * Hc, by - 1.9 * Hc)
    P = K.Painter(pw, ph, key, PAL, u, seed=seed + 1, sun_z=0.4)
    ids = np.full((ph, pw), -1, np.int16)      # id of the front-most mass at each pixel (overlap edges)
    vmap = np.zeros((ph, pw), np.float32)      # the painter's lit factor of that front-most mass
    ids_n = []
    masks = {}
    orig = K._paint_px

    def rec(prem, oy, ox, v, dk, lw, tm, hotk, tz, hz, e, e2, rimk, glow, a, *rest):
        hh, ww = a.shape
        m = a > 0.5
        ids[oy:oy + hh, ox:ox + ww][m] = len(ids_n) - 1
        k_ = len(ids_n) - 1
        if k_ not in masks:
            masks[k_] = np.zeros((ph, pw), bool)
        masks[k_][oy:oy + hh, ox:ox + ww] |= m
        vmap[oy:oy + hh, ox:ox + ww][m] = v[m]
        return orig(prem, oy, ox, v, dk, lw, tm, hotk, tz, hz, e, e2, rimk, glow, a, *rest)
    K._paint_px = rec
    g = (cx + 0.02 * Hc, by - 0.5 * Hc, 0.42 * Hc, 0.62 * Hc)
    common = dict(group=g, form=0.25, term_w=0.08, wrap=0.0, brush=0.0, lit_step=0.3, term=0.45, hot=1.0,
                  sky=0.16, rim=0.0, rim_interior=0.0, split=0.38, scallop=1.15, term_noise=0.14,
                  levels=LEV_BIG, concave=0.7, clump=0.5, lit_bias=0.0)

    def mass(env, **kw):
        k = dict(common)
        k.update(kw)
        if not isinstance(env, np.ndarray):
            env = dict(env)
            env['cx'] = cx + (env['cx'] - cx) * WF
            env['width'] = env['width'] * WF
        poly = env if isinstance(env, np.ndarray) else K.envelope(rng=rng, **env)
        ids_n.append(1)
        return P.shape(poly, rng=rng, **k)

    shade_low = dict(pal=PAL_BASE, light=0.6, lit_bias=-0.02, form=0.4, cast=0.35, wrap=0.3, sky=0.14,
                     inner=1, inner_val=0.55, inner_size=(0.35, 0.6), shade_top=0.08, valley_dark=0.06,
                     refl=0.6, bounce=0.4)
    # ---- anvil: grows out of the crown; blunt, rounded upwind end on the LEFT (the sun sits behind
    # its crest), long wedge downwind to the right that tears into fibres
    xa = cx + 0.02 * Hc
    long_, short_ = 0.6 * Hc, 0.24 * Hc
    ap = K.anvil_poly(xa, top + 0.03 * Hc, short_, long_, 0.16 * Hc, rng, dome=0.5, rise=0.35, tip=0.08,
                      sag=0.55)
    n_up = len(ap) // 2
    xx = ap[n_up:, 0]
    ap[n_up:, 1] += 0.014 * Hc * np.sin((xx - xa) / (0.08 * Hc) + rng.uniform(0, 6.28)) * \
        np.clip(np.abs(xx - xa) / (0.2 * Hc), 0, 1)
    mass(ap, pal=PAL_ANVIL, levels=((0.06, 0.13, 0.6, 2.0, 4.0, 0.25, 0.5), (0.02, 0.045, 0.45, 1.4, 3.0)),
         size=0.2 * Hc, firm=0.7, soften=0.5, base_dark=0.6, side_scale=0.45, down_cut=0.05, clump=0.6,
         top_bias=1.8, inner=0, fray=(xa + 0.3 * long_, xa + 0.95 * long_, 1.25, 0.4 * long_), refl=0.55,
         wrap=0.0, bounce=0.25, shade_top=0.0, form=0.25, mass_r=0.45, split=0.1, scallop=0.6, sky=0.0,
         lit_bias=0.16, cast=0.0, group=(xa + 0.1 * long_, top + 0.1 * Hc, 0.6 * long_, 0.16 * Hc))
    # ---- crown heads punching up into the anvil's underside (staggered heights, no flat cap)
    mass(dict(cx=cx + 0.2 * Hc, base_y=by - 0.7 * Hc, width=0.3 * Hc, height=0.16 * Hc, power=2.2, lump=0.12,
              base_round=0.5, base_wave=0.06, skew=0.2), size=0.17 * Hc, cast=0.3, side_scale=0.6, form=0.4,
         lit_bias=0.02)
    mass(dict(cx=cx - 0.12 * Hc, base_y=by - 0.72 * Hc, width=0.36 * Hc, height=0.17 * Hc, power=2.2,
              lump=0.12, base_round=0.5, base_wave=0.06, skew=-0.2), size=0.19 * Hc, cast=0.3,
         side_scale=0.62, form=0.4, lit_bias=0.12)
    # ---- the rising column: one stout body from the base up into the crown, leaning right
    mass(dict(cx=cx + 0.03 * Hc, base_y=by - 0.22 * Hc, width=0.56 * Hc, height=0.58 * Hc, power=2.5,
              lump=0.08, lean=0.1, base_round=0.3, base_wave=0.05), size=0.34 * Hc, cast=0.3, side_scale=0.72,
         inner=1, inner_val=0.55)
    # ---- right flank: 2-3 big lobes bulging out on the shadow side, staggered down the column
    mass(dict(cx=cx + 0.3 * Hc, base_y=by - 0.52 * Hc, width=0.32 * Hc, height=0.2 * Hc, power=2.3, lump=0.1,
              skew=0.2, base_round=0.35, base_wave=0.08), size=0.2 * Hc, cast=0.3, side_scale=0.62, form=0.4,
         wrap=0.0, lit_bias=0.14)
    mass(dict(cx=cx + 0.42 * Hc, base_y=by - 0.3 * Hc, width=0.34 * Hc, height=0.24 * Hc, power=2.3, lump=0.1,
              lean=0.04, skew=0.25, base_round=0.35, base_wave=0.08), size=0.22 * Hc, cast=0.35,
         side_scale=0.68, form=0.4, light=0.95, wrap=0.0, inner=1, inner_val=0.55, lit_bias=0.2)
    # ---- sun-side (left) lobes, lit-on-lit: crisp overlapping-mass edges inside the light
    mass(dict(cx=cx - 0.2 * Hc, base_y=by - 0.52 * Hc, width=0.3 * Hc, height=0.19 * Hc, power=2.3, lump=0.1,
              skew=-0.2, base_round=0.35, base_wave=0.08), size=0.19 * Hc, cast=0.25, side_scale=0.6,
         form=0.4, lit_bias=0.06)
    mass(dict(cx=cx - 0.02 * Hc, base_y=by - 0.46 * Hc, width=0.26 * Hc, height=0.15 * Hc, power=2.2, lump=0.1,
              skew=-0.1, base_round=0.3, base_wave=0.08), size=0.15 * Hc, cast=0.25, side_scale=0.6, form=0.35,
         lit_bias=0.04)
    mass(dict(cx=cx - 0.27 * Hc, base_y=by - 0.3 * Hc, width=0.32 * Hc, height=0.21 * Hc, power=2.3, lump=0.1,
              lean=-0.03, skew=-0.25, base_round=0.4, base_wave=0.08), size=0.2 * Hc, cast=0.3,
         side_scale=0.65, form=0.4, lit_bias=0.05)
    mass(dict(cx=cx + 0.14 * Hc, base_y=by - 0.28 * Hc, width=0.28 * Hc, height=0.17 * Hc, power=2.3, lump=0.1,
              skew=0.1, base_round=0.35, base_wave=0.08), size=0.17 * Hc, cast=0.3, side_scale=0.62, form=0.4,
         lit_bias=0.16)
    # ---- low left shoulder, tearing into wisps on its far side
    mass(dict(cx=cx - 0.44 * Hc, base_y=by - 0.04 * Hc, width=0.3 * Hc, height=0.15 * Hc, power=2.3, lump=0.12,
              lean=-0.05, skew=-0.2), size=0.2 * Hc, cast=0.35, side_scale=0.7, form=0.4, inner=1, light=0.95,
         lit_bias=0.1, fray=(cx - 0.45 * Hc, cx - 0.66 * Hc, 0.9, 0.1 * Hc),
         group=(cx - 0.46 * Hc, by - 0.16 * Hc, 0.22 * Hc, 0.2 * Hc))
    # ---- lower body: big overlapping masses of varied size, uneven heights
    mass(dict(cx=cx - 0.08 * Hc, base_y=by - 0.02 * Hc, width=0.6 * Hc, height=0.24 * Hc, power=2.4, lump=0.1,
              skew=0.2), size=0.26 * Hc, **shade_low)
    mass(dict(cx=cx + 0.36 * Hc, base_y=by + 0.02 * Hc, width=0.42 * Hc, height=0.17 * Hc, power=2.3, lump=0.1,
              skew=-0.2), size=0.18 * Hc, **dict(shade_low, light=0.5, lit_bias=-0.04))
    mass(dict(cx=cx - 0.3 * Hc, base_y=by + 0.04 * Hc, width=0.38 * Hc, height=0.12 * Hc, power=2.3, lump=0.1,
              skew=0.2), size=0.13 * Hc, **dict(shade_low, inner=0, light=0.7))
    K._paint_px = orig
    rgba = P.rgba()
    # de-stair the thresholded value edges and remove the faint interior mottle
    rgb = np.ascontiguousarray(rgba[..., :3])
    k_ = 5 if u > 0.75 else 3
    med = cv2.medianBlur(cv2.medianBlur(rgb, k_), k_)
    keep = ((rgb.max(-1) > 1.0) | (med.max(-1) > 1.0) | (rgba[..., 3] < 0.98))[..., None]
    rgb = np.where(keep, rgb, med)
    rgb = cv2.bilateralFilter(rgb, 0, 0.035, 3.0 * u + 0.5)
    rgba = np.concatenate([rgb, rgba[..., 3:4]], -1)
    # seat the sun on the anvil's upper-left crest: the highest point of the blunt upwind end
    best = None
    for x in np.linspace(xa - 0.3 * Hc, xa - 0.08 * Hc, 40):
        ey = top_edge(rgba[..., 3], x)
        if ey is not None and (best is None or ey < best[1] - 0.004 * Hc):
            best = (float(x), ey)
    sun = (best[0], best[1] + 0.004 * Hc) if best else (xa - 0.2 * Hc, top)
    return rgba, sun, dict(xa=xa, long_=long_, top=top, by=by, ids=ids, vmap=vmap, masks=masks)


