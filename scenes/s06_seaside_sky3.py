"""s06_seaside sky (round 4): the two sunset cumulus heaps flanking the low sun, painted with
lib/clouds2's cumulonimbus recipe (no anvil): a few big flat masses sharing ONE tower ellipsoid for their
light (high `form`), so each heap reads as 2-3 large smooth lit planes (warm white-gold top, peach mid)
with one scalloped terminator into cool lavender shade, crisp overlapping-mass edges, cauliflower detail
concentrated on the silhouette, a 2-4 px gold lining on the sun-facing edges and torn wispy bases.
The right heap is painted mirrored (sun mirrored too) and flipped back, so both heaps turn their
stepped, lit flank toward the sun."""
import numpy as np
import cv2

from lib import core as C, clouds2 as K

PAL = dict(K.PRESETS['sunset'])
PAL.update(hi=(1.0, 0.9, 0.72), lit=(0.98, 0.74, 0.5), lit_lo=(0.93, 0.56, 0.5), mid='#e27a8e',
           shade='#8676c2', deep='#3e3486', refl='#a08ed0', bounce='#ee9a88', rim=(1.55, 1.1, 0.6),
           haze='#e4a2bc', edge_dark=0.05)

PAL_DK = dict(PAL, shade='#6c5cae', deep='#342a78', refl='#8474c0', mid='#b86a9a')

KEY_UP = 0.55
SUN_Z = 0.45


LEV = ((0.13, 0.36, 0.95, 1.2, 2.4, 0.12, 0.45), (0.055, 0.13, 0.9, 1.0, 2.2, 0.3, 0.6),
       (0.026, 0.055, 0.85, 1.1, 2.6), (0.012, 0.024, 0.45, 1.4, 3.2))


def heap(P, cx, by, Hc, seed, wd=1.0):
    """a cumulus congestus heap, sun to the RIGHT: broad base, a stout body widening into a crown of
    big cauliflower heads, stepped lobes on the sunward flank. Every mass shares the heap ellipsoid for
    its light (form high) -> one continuous terminator, big flat lit planes."""
    rng = np.random.default_rng(seed)
    g = (cx + 0.06 * Hc * wd, by - 0.46 * Hc, 0.5 * Hc * wd, 0.58 * Hc)
    base = dict(group=g, form=0.78, scallop=1.5, levels=LEV, wrap=0.3, term_w=0.07, rim_interior=0.0, side_scale=0.72,
                split=0.3, lit_step=0.55, term=0.6, hot=0.85, term_noise=0.2, concave=0.55, clump=0.5,
                inner=0, sky=0.16, bounce_group=0.25, cast=0.35, shade_top=0.45, valley_dark=0.25)

    def m(dx, dy, w, h, size, env=None, **kw):
        k = dict(base)
        k.update(kw)
        e = dict(power=2.4, lump=0.1)
        e.update(env or {})
        poly = K.envelope(cx + dx * Hc * wd, by + dy * Hc, w * Hc * wd, h * Hc, rng, **e)
        P.shape(poly, rng=rng, size=size * Hc, **k)

    # broad far body in the heap's own shade (seen past the sunward towers)
    m(-0.22, 0.0, 0.8, 0.5, 0.3, env=dict(skew=-0.2, base_round=0.2), light=0.7, side_scale=0.95, down_cut=0.15)
    m(-0.3, -0.36, 0.4, 0.36, 0.26, env=dict(power=2.2, lump=0.13, skew=-0.2), side_scale=0.95)
    # a cooler, darker plane turned away from the light inside the shade side (crisp overlap edge,
    # skylight catching its top)
    m(-0.3, -0.08, 0.36, 0.26, 0.2, env=dict(power=2.3, lump=0.12, skew=-0.3, base_round=0.3), pal=PAL_DK,
      light=0.0, side_scale=0.9, sky=0.22, cast=0.2)
    m(-0.52, 0.0, 0.3, 0.2, 0.16, env=dict(power=2.3, lump=0.12, base_round=0.3), pal=PAL_DK, light=0.0,
      side_scale=0.95, sky=0.2, cast=0.2, haze=0.05)
    # crown heads
    m(0.02, -0.62, 0.62, 0.34, 0.26, env=dict(power=2.0, lump=0.15, base_round=0.5, skew=0.1), down_cut=0.2)
    m(0.2, -0.74, 0.3, 0.2, 0.18, env=dict(power=2.2, lump=0.12, base_round=0.45))
    m(-0.16, -0.72, 0.26, 0.18, 0.17, env=dict(power=2.2, lump=0.12, base_round=0.45, skew=-0.2))
    # main body
    m(0.04, -0.22, 0.6, 0.5, 0.34, env=dict(power=2.6, lump=0.09, lean=0.05, base_round=0.3))
    # stepped lit lobes on the sunward flank
    m(0.3, -0.5, 0.3, 0.26, 0.19, env=dict(power=2.0, lump=0.14, skew=0.1, base_round=0.4))
    m(0.38, -0.24, 0.3, 0.28, 0.2, env=dict(power=2.0, lump=0.14, skew=0.1, base_round=0.45))
    m(0.44, -0.02, 0.3, 0.2, 0.17, env=dict(power=2.1, lump=0.14, base_round=0.35))
    # low body: its own shadow underneath, warm bounce
    m(-0.02, 0.02, 0.9, 0.24, 0.24, env=dict(power=2.2, lump=0.08, skew=0.3, base_round=0.1, base_wave=0.05),
      light=0.85, haze=0.04, haze_grad=0.3, inner=1, inner_val=0.5)
    m(-0.42, 0.03, 0.44, 0.16, 0.16, env=dict(power=2.4, base_round=0.12, base_wave=0.05), light=0.8,
      haze=0.06, haze_grad=0.35)


def _heap(pw, ph, sun, x0, by, Hc, seed, u, width=1.0, mirror=False):
    if mirror:
        sun = (pw - 1 - sun[0], sun[1])
        x0 = pw - 1 - x0
    key = (sun[0], sun[1] - KEY_UP * ph)
    P = K.Painter(pw, ph, key, PAL, u, seed=seed + 1, sun_z=SUN_Z)
    heap(P, x0, by, Hc, seed, width)
    rng = np.random.default_rng(seed + 5)
    P.wisps(x0 - 0.62 * Hc * width, x0 + 0.62 * Hc * width, by + 0.02 * Hc, 0.12 * Hc, rng=rng, pal=PAL,
            seed=seed + 11, amount=0.6, erode=1.2, base_haze=0.45, haze_color='#eaa0b0')
    P.lining(1.1, rim_px=2.8, halo=0.8, backlit=0.35)
    rgba = P.rgba()
    if mirror:
        rgba = np.ascontiguousarray(rgba[:, ::-1])
    return rgba


def cumulus(pw, ph, sun, W, H, ox, oy, seed=61):
    """two sunset heaps flanking the low sun -> straight RGBA plate (plate px)."""
    Hs = H / 1080.0
    u = W / 1920.0
    a = _heap(pw, ph, sun, 0.205 * W + ox, 0.39 * H + oy, 390 * Hs, 71, u, width=1.35)
    b = _heap(pw, ph, sun, 0.9 * W + ox, 0.405 * H + oy, 370 * Hs, 77, u, width=1.3, mirror=True)
    return _over(a, b)


def _over(a, b):
    ab, aa = b[..., 3:4], a[..., 3:4]
    al = ab + aa * (1 - ab)
    rgb = (b[..., :3] * ab + a[..., :3] * aa * (1 - ab)) / np.maximum(al, 1e-5)
    rgb = np.where(al > 1e-5, rgb, a[..., :3] * (aa > 0) + b[..., :3] * (aa <= 0))
    return np.concatenate([rgb, al], -1).astype(np.float32)
