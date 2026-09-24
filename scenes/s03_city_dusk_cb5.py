"""Hero cumulonimbus + far bank for s03_city_dusk (round 6), painted with the shared lib.clouds2 module.

Round-6 art direction (fixing the 'two paper cut-outs glued along a vertical seam' look of round 5):
  * 4 big overlapping masses (back crown, sunward shoulder, main body, low foot) + a few secondary masses,
    each placed so the overlaps give crisp mass-over-mass edges, instead of one symmetric column;
  * the light comes from the afterglow low at centre-left but a little IN FRONT of the cloud (sun_z > 0)
    and a mass-dome weighted form (low `form`), so the terminator wraps round every mass as a curved,
    scalloped line - never one straight vertical split;
  * lit face pushed from flat peach to a hot pink-gold that goes near-white at the sun-facing left crown;
    magenta terminator band; cool violet shadow with skylit up-facing tops (shade-on-shade edges);
  * cauliflower detail kept on the WHOLE silhouette incl. the shadow-side right edge (side_scale high);
  * crisp 2-4 px gold lining on the sunward silhouette with very little halo (no soft glow outline);
  * the base is torn into wisps and sinks into the violet haze behind the towers.
"""
import numpy as np
import cv2

from lib import core as C, clouds2 as K2

PAL = dict(hi=(1.0, 0.86, 0.7), lit=(1.0, 0.5, 0.42), lit_lo='#e04a7c', mid='#c2427e', shade='#7560bc',
           deep='#3e3488', refl='#8c7cd0', bounce='#c46aa0', rim=(1.7, 1.25, 0.66), haze='#9c86cc',
           edge_dark=0.05)
PAL_FAR = dict(hi=(1.0, 0.78, 0.7), lit='#f0909e', lit_lo='#d97aa6', mid='#b862a4', shade='#7466b8',
               deep='#564a9e', refl='#8274c2', bounce='#a868a6', rim=(1.3, 0.9, 0.7), haze='#9488cc',
               edge_dark=0.03)


def _tower(P, X, Y, H, seed, form=0.38, sun_z=0.55):
    """4 big masses + 3 secondary ones, back to front."""
    rng = np.random.default_rng(seed)
    cx, base, Hc = X(0.84), Y(0.575), 0.48 * H
    g = (cx - 0.02 * Hc, base - 0.5 * Hc, 0.42 * Hc, 0.56 * Hc)
    common = dict(group=g, form=form, sun_z=sun_z, wrap=0.45, term_w=0.09, rim_interior=0.0, side_scale=0.9,
                  term_noise=0.3, inner=1, inner_val=0.55, sky=0.3, halo=0.15, rim_px=3.0, hot=0.6,
                  lit_step=0.5, scallop=1.2)
    masses = [
        # back crown: tallest head, slightly right of centre, leaning left (updraft sheared by the wind)
        (dict(cx=cx + 0.06 * Hc, base_y=base - 0.52 * Hc, width=0.36 * Hc, height=0.48 * Hc, power=2.3,
              lump=0.12, lean=-0.06, base_round=0.3), dict(size=0.3 * Hc, cast=0.25)),
        # right shadow-side shoulder (mostly in shade: violet, cauliflower edge against the sky)
        (dict(cx=cx + 0.27 * Hc, base_y=base - 0.2 * Hc, width=0.34 * Hc, height=0.42 * Hc, power=2.2,
              lump=0.13, lean=0.04, base_round=0.3), dict(size=0.26 * Hc, cast=0.3)),
        # sunward shoulder: lower-left head catching the full afterglow
        (dict(cx=cx - 0.2 * Hc, base_y=base - 0.28 * Hc, width=0.34 * Hc, height=0.4 * Hc, power=2.3,
              lump=0.12, lean=-0.04, base_round=0.3), dict(size=0.27 * Hc, cast=0.3)),
        # main body in front of the crown
        (dict(cx=cx + 0.02 * Hc, base_y=base - 0.12 * Hc, width=0.52 * Hc, height=0.46 * Hc, power=2.6,
              lump=0.09, skew=0.15, base_round=0.2), dict(size=0.32 * Hc, cast=0.4)),
        # low foot, wide and flat-bottomed (torn later)
        (dict(cx=cx - 0.02 * Hc, base_y=base, width=0.86 * Hc, height=0.22 * Hc, power=2.2, lump=0.08,
              skew=0.25, base_round=0.1, base_wave=0.05),
         dict(size=0.2 * Hc, cast=0.35, haze_grad=0.3, light=0.85, haze=0.06)),
        # small forward knots on the sunward flank and on the right (crisp overlapping edges)
        (dict(cx=cx - 0.33 * Hc, base_y=base + 0.01 * Hc, width=0.3 * Hc, height=0.17 * Hc, power=2.4,
              lump=0.1, base_round=0.12), dict(size=0.15 * Hc, cast=0.3, haze=0.05)),
        (dict(cx=cx + 0.38 * Hc, base_y=base + 0.015 * Hc, width=0.3 * Hc, height=0.2 * Hc, power=2.3,
              lump=0.12, base_round=0.12), dict(size=0.16 * Hc, cast=0.3, haze=0.06)),
    ]
    for ek, sk in masses:
        poly = K2.envelope(rng=rng, **ek)
        kw = dict(common)
        kw.update(sk)
        P.shape(poly, rng=rng, pal=PAL, **kw)
    P.wisps(cx - 0.55 * Hc, cx + 0.6 * Hc, base + 0.02 * Hc, 0.09 * Hc, rng=rng, pal=PAL, seed=seed + 11,
            amount=0.5, erode=1.3)
    return cx, base, Hc


def paint(pw, ph, W, H, ox, sun, seed=5, sky=None, form=0.33, sun_z=0.5, sun_y=0.3):
    """Straight-alpha RGBA plates (ph, pw, 4): (tower, bank). Frame fractions offset by ox px."""
    X = lambda fx: fx * W + ox
    Y = lambda fy: fy * H
    u = W / 1920.0
    # light: the afterglow at centre-left; the upper tower still catches direct light
    sun_l = (float(sun[0]), Y(sun_y))
    P = K2.Painter(pw, ph, sun_l, PAL, u, seed=seed + 1, sun_z=sun_z)
    cx, base, Hc = _tower(P, X, Y, H, seed, form=form, sun_z=sun_z)
    top = base - Hc
    # thin sheared anvil shelf streaming downwind (right) from the crown, fraying into fibres
    P.fibres(cx + 0.05 * Hc, X(1.02), top + 0.07 * H, 0.02 * H, direction=1.0, pal=PAL, amount=0.7,
             seed=seed + 21, lit=0.8)
    P.fibres(cx + 0.1 * Hc, X(0.98), top + 0.1 * H, 0.011 * H, direction=1.0, pal=PAL, amount=0.4,
             seed=seed + 23, lit=0.6)
    P.lining(1.4, rim_px=3.0, halo=0.25)
    tower = P.rgba()
    ys = np.arange(ph, dtype=np.float32)[:, None]
    # the lowest part sinks into the violet haze behind the skyline (earth's shadow + dust)
    hz = np.array([0.58, 0.44, 0.8], np.float32)
    fz = C.smoothstep(Y(0.4), Y(0.58), ys)[..., None]
    tower[..., :3] = tower[..., :3] * (1 - 0.55 * fz) + hz * 0.55 * fz
    # ------------------------------------------------------------------ low far bank behind the skyline
    Pb = K2.Painter(pw, ph, (float(sun[0]), float(sun[1])), PAL_FAR, u, seed=seed + 5, sun_z=0.05)
    for i, (fx, fy, cw, ch) in enumerate(((0.64, 0.585, 0.16, 0.07), (0.98, 0.59, 0.3, 0.14),
                                         (1.1, 0.59, 0.18, 0.09), (0.76, 0.59, 0.14, 0.06))):
        K2.cumulus(Pb, X(fx), Y(fy), cw * W, ch * H, seed=seed * 10 + i, pal=PAL_FAR, n=3, haze=0.3)
    Pb.lining(0.8, rim_px=2.2, halo=0.3)
    Pb.haze_band(Y(0.47), Y(0.6), np.array([0.62, 0.5, 0.84], np.float32), 0.5)
    bank = Pb.rgba()
    return tower.astype(np.float32), bank.astype(np.float32)
