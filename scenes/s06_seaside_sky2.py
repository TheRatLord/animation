"""s06_seaside sky (round 3): sunset cumulus painted with lib/clouds2 at hero scale.

Two cumulus heaps flank the low sun: built with clouds2.Painter mass by mass - a broad hazy back body,
a main tower with big cauliflower heads along the top and the sun-side flank, and lit lower lobes toward
the sun; sunset palette (warm-white/gold lit face -> peach/rose terminator -> lavender/violet shade),
crisp 2-4 px gold lining on the sun-facing silhouette, torn wispy bases dissolving into the haze.
All positions in plate px; sizes scale with the frame height."""
import math
import numpy as np
import cv2

from lib import core as C, clouds2 as K

PAL = dict(K.PRESETS['sunset'])
PAL.update(hi=(1.02, 0.9, 0.7), lit=(1.0, 0.74, 0.5), lit_lo=(0.96, 0.58, 0.52), mid='#e07890',
           shade='#8a78c4', deep='#4a3e90', refl='#a592d2', bounce='#eb9a8a', rim=(1.5, 1.08, 0.62),
           haze='#e2a0bc', edge_dark=0.05)


LEV_BIG = ((0.14, 0.4, 0.95, 1.2, 2.4, 0.12, 0.45), (0.06, 0.15, 0.95, 1.0, 2.1, 0.3, 0.6),
           (0.028, 0.06, 0.85, 1.1, 2.6), (0.014, 0.026, 0.4, 1.4, 3.2))
PAL_SH = dict(PAL, lit=(0.95, 0.64, 0.52), lit_lo=(0.9, 0.55, 0.56), hi=(0.98, 0.76, 0.6), shade='#8676c4',
              deep='#56489a', refl='#a495d4')


def _heap(P, cx, by, Hc, side, rng, seed, sun, masses):
    """one cumulus heap: side=+1 sun to the right of it, -1 to the left. masses = rows of
    (dx, dbase, width, height, size, lit_bias, kind, skew) in units of Hc (dx toward the sun); kind:
    0 = shaded back body, 1 = lit mass, 2 = low shade mass."""
    s = side
    g = (cx + 0.08 * s * Hc, by - 0.45 * Hc, 0.75 * Hc, 0.55 * Hc)
    common = dict(group=g, sun=sun, form=0.3, term_w=0.1, wrap=0.25, brush=0.0, lit_step=0.3, term=0.6,
                  hot=0.9, sky=0.14, rim=0.0, rim_interior=0.0, split=0.28, scallop=1.1, term_noise=0.12,
                  levels=LEV_BIG, concave=0.7, clump=0.5, side_scale=0.66, cast=0.35, lost=0.03)
    shade = dict(pal=PAL_SH, light=0.5, lit_bias=-0.12, form=0.4, cast=0.45, wrap=0.1, sky=0.12, inner=1,
                 inner_val=0.45, inner_size=(0.35, 0.6), shade_top=0.2, valley_dark=0.12, refl=0.5, bounce=0.4)
    for (dx, db, w, h, sz, lb, kind, sk) in masses:
        k = dict(common)
        if kind == 0:
            k.update(shade, inner=0, haze=0.1, haze_grad=0.3)
        elif kind == 2:
            k.update(shade)
        k['lit_bias'] = k.get('lit_bias', 0.0) + lb
        poly = K.envelope(cx + dx * s * Hc, by + db * Hc, w * Hc, h * Hc, rng, power=2.3, lump=0.1,
                          skew=sk * s, base_round=0.35, base_wave=0.06, lean=0.02 * s)
        P.shape(poly, rng=rng, size=sz * Hc, **k)
    P.wisps(cx - 0.75 * Hc, cx + 0.85 * Hc, by + 0.03 * Hc, 0.1 * Hc, rng=rng, pal=PAL_SH, seed=seed + 11,
            amount=0.55, erode=1.1, base_haze=0.4, haze_color='#e8a0b0')


# (dx, dbase, width, height, size, lit_bias, kind, skew): stacked tiers, stepping out toward the sun
LEFT = [(-0.2, 0.02, 1.4, 0.42, 0.3, 0.0, 0, -0.2),        # broad shaded base body
        (-0.22, -0.62, 0.5, 0.3, 0.22, 0.05, 1, 0.1),      # crown head
        (0.02, -0.55, 0.36, 0.22, 0.18, 0.1, 1, 0.3),      # second crown head (sunward)
        (-0.12, -0.36, 0.85, 0.36, 0.3, 0.05, 1, 0.2),     # upper tier
        (0.3, -0.3, 0.42, 0.26, 0.2, 0.15, 1, 0.3),        # sunward upper lobe
        (0.05, -0.14, 1.0, 0.34, 0.3, 0.05, 1, 0.1),       # middle tier
        (0.52, -0.1, 0.42, 0.24, 0.2, 0.2, 1, 0.3),        # sunward mid lobe
        (-0.35, 0.02, 0.7, 0.24, 0.2, 0.0, 2, -0.2),       # low far shade
        (0.25, 0.03, 0.8, 0.22, 0.2, 0.1, 1, 0.1),         # low sunward tier
        (0.68, 0.02, 0.3, 0.16, 0.14, 0.25, 1, 0.3)]       # leading lobe
RIGHT = [(-0.2, 0.02, 1.3, 0.45, 0.3, 0.0, 0, -0.2),
         (-0.05, -0.66, 0.46, 0.3, 0.22, 0.05, 1, 0.2),
         (-0.1, -0.4, 0.8, 0.36, 0.3, 0.05, 1, 0.1),
         (0.24, -0.34, 0.4, 0.26, 0.2, 0.15, 1, 0.3),
         (0.08, -0.16, 0.95, 0.32, 0.28, 0.05, 1, 0.1),
         (0.48, -0.1, 0.38, 0.22, 0.18, 0.2, 1, 0.3),
         (-0.3, 0.03, 0.7, 0.24, 0.2, 0.0, 2, -0.2),
         (0.25, 0.03, 0.7, 0.2, 0.18, 0.12, 1, 0.1)]


def _destair(rgba, u):
    rgb = np.ascontiguousarray(rgba[..., :3])
    k_ = 5 if u > 0.75 else 3
    med = cv2.medianBlur(cv2.medianBlur(rgb, k_), k_)
    keep = ((rgb.max(-1) > 1.0) | (med.max(-1) > 1.0) | (rgba[..., 3] < 0.98))[..., None]
    rgb = np.where(keep, rgb, med)
    rgb = cv2.bilateralFilter(rgb, 0, 0.03, 2.0 * u + 0.5)
    return np.concatenate([rgb, rgba[..., 3:4]], -1).astype(np.float32)


def _rim(rgba, key_dir, sun, W, top_y, base_y, strength=1.0):
    """gold lining only where the outer silhouette faces the light (sunward flank + crown), hotter
    toward the sun; shadow-side edges meet the sky softly."""
    u = W / 1920.0
    A0 = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    A = C.smoothstep(0.35, 0.75, A0).astype(np.float32)
    h, w = A.shape
    Ab = cv2.GaussianBlur(A, (0, 0), 5 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gl, -gy / gl
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    kd = np.asarray(key_dir, np.float32)
    kd = kd / float(np.hypot(*kd))
    face = C.smoothstep(0.2, 0.7, nx * kd[0] + ny * kd[1])
    d = np.hypot(xs - sun[0], ys - sun[1]) / W
    near = np.exp(-(d / 0.35) ** 2)
    k = face * (0.3 + 0.7 * near)
    rb = max(int(round(2.0 * u)), 1)
    core = np.clip(A - cv2.erode(A, np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)), 0, 1)
    rb2 = max(int(round(3.5 * u)), 1)
    wide = cv2.GaussianBlur(np.clip(A - cv2.erode(A, np.ones((2 * rb2 + 1, 2 * rb2 + 1), np.uint8)), 0, 1),
                            (0, 0), 1.0 * u)
    ring = np.maximum(core, wide * 0.6)
    kk = np.clip(ring * k * strength, 0, 1)
    col = np.array([1.9, 1.35, 0.75], np.float32)
    rgb = rgba[..., :3] * (1 - kk[..., None]) + col * kk[..., None]
    hk = cv2.GaussianBlur(kk, (0, 0), 2.5 * u) * 0.6 + cv2.GaussianBlur(kk, (0, 0), 9 * u) * 0.45
    hk = np.clip(hk * (1 - A) * (0.4 + 0.8 * near), 0, 1)
    a = np.clip(A0 + hk * 0.45, 0, 1)
    A = A0
    hcol = np.array([1.3, 0.85, 0.55], np.float32)
    rgb = (rgb * A[..., None] + hcol * hk[..., None]) / np.maximum(a, 1e-4)[..., None]
    rgb = np.where(a[..., None] > 1e-4, rgb, rgba[..., :3])
    return np.concatenate([rgb, a[..., None]], -1).astype(np.float32)


def tower(P, x0, base_y, Hc, s, seed, sq=1.0):
    """one sunset cumulus heap (after the s01 hero recipe, mirrored by s = +1 sun to the right / -1 left):
    broad shaded base, a stout column widening into a cauliflower crown, lit lobes on the sunward face,
    a lower shoulder tearing into wisps on the far side, lower body in the heap's own shade.
    sq squashes the heights (a lower, broader heap)."""
    rng = np.random.default_rng(seed)
    cx, by = x0, base_y
    Hv = Hc * sq
    g = (cx + 0.02 * s * Hc, by - 0.52 * Hv, 0.5 * Hc, 0.62 * Hv)
    common = dict(group=g, form=0.55, term_w=0.1, wrap=0.25, brush=0.03, halo=0.0, lit_step=0.3, term=0.65, hot=0.9,
                  sky=0.12, rim=0.0, rim_interior=0.0, split=0.26, scallop=1.1, term_noise=0.12, levels=LEV_BIG,
                  concave=0.3, clump=0.5)

    def mass(dx, dy, w, h, size, env=None, group=None, **kw):
        k = dict(common)
        k.update(kw)
        e = dict(power=2.4, lump=0.1)
        e.update(env or {})
        if 'lean' in e:
            e['lean'] *= s
        if 'skew' in e:
            e['skew'] *= s
        poly = K.envelope(cx + dx * s * Hc, by + dy * Hv, w * Hc, h * Hv, rng, **e)
        if group is not None:
            k['group'] = (cx + group[0] * s * Hc, by + group[1] * Hv, group[2] * Hc, group[3] * Hv)
        P.shape(poly, rng=rng, size=size * Hc, **k)

    shade_low = dict(pal=PAL_SH, light=0.45, lit_bias=-0.12, form=0.4, cast=0.5, wrap=0.15, sky=0.1,
                     inner=0, inner_val=0.42, inner_size=(0.35, 0.6), shade_top=0.15, valley_dark=0.12,
                     refl=0.5, bounce=0.4)
    mass(0.06, 0.02, 1.15, 0.42, 0.3, **dict(shade_low, inner=0))
    # secondary tower on the far (shadow) side, shorter
    mass(-0.36, -0.2, 0.36, 0.42, 0.3, env=dict(power=2.6, lean=-0.06, skew=-0.2), cast=0.35, side_scale=0.72,
         light=0.45, pal=PAL_SH, inner=0, group=(-0.36, -0.42, 0.2, 0.26))
    # crown
    mass(0.04, -0.68, 0.82, 0.3, 0.28, env=dict(power=2.0, lump=0.16, base_round=0.5, base_wave=0.06, skew=0.1), cast=0.3,
         side_scale=0.62, down_cut=0.2, lit_bias=0.1, group=(0.05, -0.8, 0.36, 0.22), form=0.5)
    mass(0.17, -0.79, 0.36, 0.19, 0.2, env=dict(power=2.2, lump=0.12, base_round=0.45, base_wave=0.06), cast=0.45,
         side_scale=0.6, form=0.45)
    mass(-0.2, -0.82, 0.26, 0.16, 0.17, env=dict(power=2.2, lump=0.12, base_round=0.45, base_wave=0.06,
                                                   skew=-0.2), cast=0.45, side_scale=0.6, form=0.45)
    # main column
    mass(0.02, -0.26, 0.52, 0.5, 0.34, env=dict(power=2.6, lump=0.09, lean=0.05, base_round=0.3, base_wave=0.05),
         cast=0.4, side_scale=0.72, inner=0, lit_bias=0.08)
    # lit lobes on the sunward face
    mass(0.15, -0.52, 0.32, 0.18, 0.18, env=dict(power=2.3, skew=0.2, base_round=0.3, base_wave=0.08), cast=0.3,
         side_scale=0.6, form=0.4, lit_bias=0.12)
    mass(0.3, -0.3, 0.3, 0.2, 0.18, env=dict(power=2.3, skew=0.2, base_round=0.4, base_wave=0.08), cast=0.3,
         side_scale=0.6, form=0.4, lit_bias=0.15, group=(0.3, -0.4, 0.18, 0.16))
    # far-side mid lobe in shade
    mass(-0.2, -0.4, 0.26, 0.2, 0.2, env=dict(power=2.4, skew=0.2, base_round=0.5, base_wave=0.08), cast=0.55,
         side_scale=0.7, form=0.4, light=0.8, inner=0)
    # sunward low shoulder
    mass(0.42, -0.08, 0.38, 0.26, 0.24, env=dict(power=2.3, lump=0.12, lean=0.05, skew=0.2), cast=0.45,
         side_scale=0.7, form=0.45, lit_bias=0.1, group=(0.42, -0.2, 0.22, 0.22))
    # far-side shoulder tearing into wisps
    fx0 = cx - 0.45 * s * Hc
    mass(-0.42, -0.08, 0.34, 0.24, 0.22, env=dict(power=2.3, lump=0.12, skew=-0.2), cast=0.45, side_scale=0.7,
         form=0.45, light=0.35, pal=PAL_SH, lit_bias=-0.15, fray=(fx0, fx0 - 0.17 * s * Hc, 0.9, 0.1 * Hc), group=(-0.42, -0.2, 0.2, 0.2))
    # lower body in the heap's own shade
    mass(-0.06, -0.1, 0.56, 0.28, 0.3, env=dict(skew=-0.2), **shade_low)
    mass(0.26, -0.03, 0.4, 0.2, 0.22, env=dict(power=2.3, skew=0.3), **dict(shade_low, light=0.65, lit_bias=0.0))
    mass(-0.3, 0.02, 0.4, 0.16, 0.15, env=dict(power=2.3, skew=-0.2), **dict(shade_low, inner=0, light=0.6))
    P.wisps(cx - 0.68 * Hc, cx + 0.66 * Hc, by + 0.02 * Hv, 0.1 * Hv, rng=rng, pal=PAL_SH, seed=seed + 11,
            amount=0.55, erode=1.2, base_haze=0.45, haze_color='#e8a0b0')


KEY_UP = 0.2
SUN_Z = 0.3


def cumulus(pw, ph, sun, W, H, ox, oy, seed=61):
    """two sunset heaps flanking the low sun -> straight RGBA plate (plate px)."""
    Hs = H / 1080.0
    u = W / 1920.0
    out = None
    for (fx, fy, hc, s, sd, sq) in ((0.2, 0.37, 420, 1, 71, 0.85), (0.885, 0.395, 400, -1, 73, 0.85)):
        x0, by, Hc = fx * W + ox, fy * H + oy, hc * Hs
        key = (sun[0], sun[1] - KEY_UP * H)
        P = K.Painter(pw, ph, key, PAL, u, seed=sd + 1, sun_z=SUN_Z)
        tower(P, x0, by, Hc, s, sd, sq)
        rgba = _destair(P.rgba(), u)
        rgba = _rim(rgba, (s * 0.8, -0.6), sun, W, by - Hc * sq, by)
        out = rgba if out is None else _over(out, rgba)
    return out


def _over(a, b):
    ab, aa = b[..., 3:4], a[..., 3:4]
    al = ab + aa * (1 - ab)
    rgb = (b[..., :3] * ab + a[..., :3] * aa * (1 - ab)) / np.maximum(al, 1e-5)
    rgb = np.where(al > 1e-5, rgb, a[..., :3] * (aa > 0) + b[..., :3] * (aa <= 0))
    return np.concatenate([rgb, al], -1).astype(np.float32)
