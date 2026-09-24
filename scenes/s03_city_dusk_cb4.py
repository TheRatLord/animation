"""Hero cumulonimbus + far bank for s03_city_dusk (round 5): painted with the shared lib.clouds2 module.

The tower is built from a handful of big FLAT painted masses (hot pink-gold lit face toward the afterglow at
centre-left, magenta terminator band, cool violet shadow, indigo core low down) with cauliflower detail only
on the silhouette, a crisp 2-4 px gold lining on the sun-facing edge, a sheared anvil streaming right
(downwind) and a torn base that dissolves into the dusk haze behind the skyline. The earth's shadow climbs
the tower: below a tilted line everything grades into cool violet so only the upper sun-facing masses keep
the hot light. A low hazy cumulus bank sits on the horizon behind the skyline at the right.
"""
import numpy as np
import cv2

from lib import core as C, clouds2 as K2

# dusk palette: lit = hot pink-gold, mid = magenta, shade = cool violet
PAL = dict(hi=(0.9, 0.6, 0.46), lit='#e66e78', lit_lo='#cc5288', mid='#a8327e', shade='#7a64c0',
           deep='#40368c', refl='#8a78cc', bounce='#c0629e', rim=(1.55, 1.08, 0.62), haze='#9c86cc',
           edge_dark=0.05)
PAL_FAR = dict(hi=(1.0, 0.78, 0.7), lit='#f0909e', lit_lo='#d97aa6', mid='#b862a4', shade='#7466b8',
               deep='#564a9e', refl='#8274c2', bounce='#a868a6', rim=(1.3, 0.9, 0.7), haze='#9488cc',
               edge_dark=0.03)


def paint(pw, ph, W, H, ox, sun, seed=7, sky=None):
    """Straight-alpha RGBA plates (ph, pw, 4): (tower, bank). Frame fractions offset by ox px."""
    X = lambda fx: fx * W + ox
    Y = lambda fy: fy * H
    u = W / 1920.0
    # light: the afterglow at centre-left, a touch above the horizon (the sun has only just set; the
    # upper tower is still in direct light from the left)
    sun_l = (float(sun[0]), Y(0.3))
    P = K2.Painter(pw, ph, sun_l, PAL, u, seed=seed + 1, sun_z=0.38)
    K2.cumulonimbus(P, X(0.84), Y(0.57), 0.48 * H, seed=seed, anvil=False, width=1.0, form=0.62, backlit=0.15, wrap=0.35, term_w=0.12, side=0.7, inner=1)
    # thin sheared anvil shelf streaming downwind (right) from the crown, fraying into fibres
    top = Y(0.57) - 0.48 * H
    P.fibres(X(0.84), X(1.0), top + 0.07 * H, 0.022 * H, direction=1.0, pal=PAL, amount=0.75, seed=seed + 21,
             lit=0.8)
    P.fibres(X(0.86), X(0.97), top + 0.1 * H, 0.012 * H, direction=1.0, pal=PAL, amount=0.45, seed=seed + 23,
             lit=0.6)
    P.lining(1.25, rim_px=3.0, halo=0.8)
    tower = P.rgba()
    ys = np.arange(ph, dtype=np.float32)[:, None]
    xs = np.arange(pw, dtype=np.float32)[None, :]
    # earth's shadow climbing the tower (a little higher away from the sun), painted as a soft value step
    line = Y(0.43) + (xs - X(0.8)) * 0.1
    esh = C.smoothstep(line - 0.03 * H, line + 0.13 * H, ys)[..., None]
    col = tower[..., :3]
    shadow = np.array([0.46, 0.38, 0.78], np.float32)
    lum = col.max(-1, keepdims=True)
    tower[..., :3] = col * (1 - 0.55 * esh) + shadow * np.clip(lum, 0, 1.0) * 0.55 * esh
    # the lowest part sinks into the violet haze behind the skyline
    hz = np.array([0.6, 0.46, 0.8], np.float32)
    fz = C.smoothstep(Y(0.44), Y(0.58), ys)[..., None]
    tower[..., :3] = tower[..., :3] * (1 - 0.5 * fz) + hz * 0.5 * fz
    # ------------------------------------------------------------------ low far bank behind the skyline
    Pb = K2.Painter(pw, ph, (float(sun[0]), float(sun[1])), PAL_FAR, u, seed=seed + 5, sun_z=0.05)
    for i, (fx, fy, cw, ch) in enumerate(((0.64, 0.585, 0.16, 0.07), (0.98, 0.59, 0.3, 0.14),
                                         (1.1, 0.59, 0.18, 0.09), (0.76, 0.59, 0.14, 0.06))):
        K2.cumulus(Pb, X(fx), Y(fy), cw * W, ch * H, seed=seed * 10 + i, pal=PAL_FAR, n=3, haze=0.3)
    Pb.lining(0.8, rim_px=2.2, halo=0.5)
    Pb.haze_band(Y(0.47), Y(0.6), np.array([0.62, 0.5, 0.84], np.float32), 0.5)
    bank = Pb.rgba()
    return tower.astype(np.float32), bank.astype(np.float32)
