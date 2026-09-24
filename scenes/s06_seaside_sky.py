"""s06_seaside sky: painted sunset cumulus (lib/clouds2) + brushed horizon stratus streaks.

Everything is built in plate px (plate = frame + margin ox/oy); sizes scale with the frame height so
the layout is resolution independent.
  * two hero cumulus painted mass by mass (clouds2.Painter): warm-gold lit side toward the low sun, a
    pink-orange terminator band, cool lavender shadow away from it, overlapping masses with crisp
    internal edges, torn wispy bases fading into the haze, gold lining only on sun-facing edges;
  * thin horizon stratus as soft horizontal brush streaks with torn ends and backlit gold undersides
    (no billow scallops, no rim sparkle).
"""
import math

import numpy as np
import cv2

from lib import core as C, clouds2 as K

# sunset cumulus palette: gold lit face, rose-orange terminator, violet shadow
PAL = dict(hi=(1.04, 0.9, 0.66), lit=(1.0, 0.72, 0.45), lit_lo=(0.97, 0.56, 0.46), mid='#e27488', shade='#8a72c0',
           deep='#4c3f8e', refl='#a48ed0', bounce='#e6908c', rim=(1.45, 1.05, 0.6), haze='#dc9cc0',
           edge_dark=0.04)
PAL_SH = dict(PAL, lit=(0.95, 0.6, 0.48), lit_lo=(0.9, 0.52, 0.52), hi=(0.98, 0.72, 0.56), shade='#8676c4',
              deep='#56489a', refl='#a495d4')
LEV = ((0.14, 0.38, 0.95, 1.2, 2.4, 0.12, 0.45), (0.05, 0.13, 0.85, 1.0, 2.2, 0.3, 0.6),
       (0.02, 0.045, 0.6, 1.2, 3.0), (0.012, 0.02, 0.2, 1.6, 4.0))


def _destair(rgba, u):
    rgb = np.ascontiguousarray(rgba[..., :3])
    k_ = 5 if u > 0.75 else 3
    med = cv2.medianBlur(cv2.medianBlur(rgb, k_), k_)
    keep = ((rgb.max(-1) > 1.0) | (med.max(-1) > 1.0) | (rgba[..., 3] < 0.98))[..., None]
    rgb = np.where(keep, rgb, med)
    rgb = cv2.bilateralFilter(rgb, 0, 0.03, 2.0 * u + 0.5)
    return np.concatenate([rgb, rgba[..., 3:4]], -1).astype(np.float32)


def _cloud(P, cx, by, Hc, rng, side, seed):
    """One sunset cumulus heap. side = +1 when the sun lies to the right of the cloud, -1 to the left.
    Masses back to front; the sunward lower flank is lit gold, the far upper flank in violet shade."""
    s = side
    g = (cx + 0.05 * s * Hc, by - 0.45 * Hc, 0.75 * Hc, 0.55 * Hc)
    # key light for the shading: the low sun lies ahead and BELOW the cloud (it lights the sunward flank and
    # the underside), so the terminator runs diagonally instead of as a vertical split
    d = 6.0 * Hc
    sun_c = (cx + s * 0.72 * d, by - 0.45 * Hc + 0.7 * d)
    common = dict(group=g, form=0.55, term_w=0.12, wrap=0.2, brush=0.03, lit_step=0.35, term=0.7, hot=0.6,
                  sky=0.1, rim=0.9, rim_px=2.6, rim_interior=0.0, split=0.06, lit_bias=0.0, scallop=1.0, term_noise=0.07, clean=0.45,
                  levels=LEV, concave=0.7, clump=0.5, pal=PAL, halo=0.4, rim_shadow=0.25, sun=sun_c)

    def mass(env, **kw):
        k = dict(common)
        k.update(kw)
        poly = K.envelope(rng=rng, **env)
        return P.shape(poly, rng=rng, **k)

    shade = dict(pal=PAL_SH, light=0.55, lit_bias=-0.12, form=0.45, cast=0.4, wrap=0.0, inner=0, inner_val=0.45,
                 inner_size=(0.3, 0.55), shade_top=0.25, valley_dark=0.1, refl=0.45, bounce=0.35)
    # back: broad low body, far side (away from the sun) - mostly in its own shade
    mass(dict(cx=cx - 0.25 * s * Hc, base_y=by + 0.02 * Hc, width=1.5 * Hc, height=0.5 * Hc, power=2.4, lump=0.1,
              skew=-0.3 * s), size=0.36 * Hc, **dict(shade, inner=0))
    # far-side shoulder tower (upper, away from the sun): cool lavender with a lit crest
    mass(dict(cx=cx - 0.38 * s * Hc, base_y=by - 0.22 * Hc, width=0.62 * Hc, height=0.62 * Hc, power=2.5,
              lump=0.1, lean=-0.05 * s, skew=-0.2 * s), size=0.34 * Hc, cast=0.35, light=0.5, inner=0, wrap=0.0, pal=PAL_SH,
         inner_val=0.5, group=(cx - 0.38 * s * Hc, by - 0.5 * Hc, 0.34 * Hc, 0.36 * Hc))
    # main heap
    mass(dict(cx=cx + 0.02 * s * Hc, base_y=by - 0.12 * Hc, width=1.05 * Hc, height=0.88 * Hc, power=2.8,
              lump=0.08, lean=0.04 * s, base_round=0.2), size=0.42 * Hc, cast=0.45, inner=0, lit_bias=0.1)
    # top head
    mass(dict(cx=cx - 0.08 * s * Hc, base_y=by - 0.72 * Hc, width=0.44 * Hc, height=0.3 * Hc, power=2.3, lump=0.12,
              base_round=0.45, skew=0.2 * s), size=0.2 * Hc, cast=0.45, side_scale=0.6, form=0.45)
    # sunward lower flank: lit-on-lit lobes (crisp overlapping-mass edges in the light)
    mass(dict(cx=cx + 0.36 * s * Hc, base_y=by - 0.06 * Hc, width=0.5 * Hc, height=0.42 * Hc, power=2.0,
              lump=0.14, skew=0.2 * s, base_round=0.35), size=0.24 * Hc, cast=0.2, form=0.4, lit_bias=0.45,
         group=(cx + 0.36 * s * Hc, by - 0.25 * Hc, 0.28 * Hc, 0.26 * Hc))
    # bulging sunward heads on the flank (break the sunward silhouette into 2-3 big cauliflower bulges)
    mass(dict(cx=cx + 0.44 * s * Hc, base_y=by - 0.42 * Hc, width=0.32 * Hc, height=0.26 * Hc, power=2.1,
              lump=0.12, base_round=0.5, skew=0.3 * s), size=0.16 * Hc, cast=0.15, form=0.4, lit_bias=0.5,
         group=(cx + 0.44 * s * Hc, by - 0.55 * Hc, 0.18 * Hc, 0.16 * Hc))
    mass(dict(cx=cx + 0.5 * s * Hc, base_y=by + 0.02 * Hc, width=0.3 * Hc, height=0.2 * Hc, power=2.1,
              lump=0.12, base_round=0.4), size=0.14 * Hc, cast=0.15, form=0.4, lit_bias=0.5,
         group=(cx + 0.5 * s * Hc, by - 0.08 * Hc, 0.16 * Hc, 0.12 * Hc))
    mass(dict(cx=cx + 0.2 * s * Hc, base_y=by - 0.36 * Hc, width=0.46 * Hc, height=0.3 * Hc, power=2.5,
              lump=0.1, base_round=0.5), size=0.2 * Hc, cast=0.2, form=0.4, lit_bias=0.45)
    # low body in shade across the base
    mass(dict(cx=cx - 0.12 * s * Hc, base_y=by + 0.04 * Hc, width=0.8 * Hc, height=0.24 * Hc, power=2.3,
              lump=0.1), size=0.2 * Hc, **dict(shade, inner=0, light=0.7))
    P.wisps(cx - 0.95 * Hc, cx + 0.8 * Hc, by + 0.03 * Hc, 0.1 * Hc, rng=rng, pal=PAL_SH, seed=seed + 11,
            amount=0.55, erode=1.1, base_haze=0.4, haze_color='#e8a0b0')


def cumulus(pw, ph, sun, clouds, u, seed=61, sun_z=0.35):
    """clouds = [(cx, base_y, Hc, side, seed), ...] (plate px) -> straight RGBA plate."""
    P = K.Painter(pw, ph, sun, PAL, u, seed=seed, sun_z=sun_z)
    for (cx, by, Hc, side, sd) in clouds:
        _cloud(P, cx, by, Hc, np.random.default_rng(sd), side, sd)
    P.lining(0.55, rim_px=2.4, halo=0.5, backlit=0.0)
    return _destair(P.rgba(), u)


def streaks(pw, ph, sun, specs, u, seed=7):
    """Thin horizon stratus: soft horizontal brush streaks with torn ends, violet-rose bodies and
    backlit gold undersides (hotter toward the sun). specs = [(x0, x1, y, thick, seed), ...] plate px."""
    out = np.zeros((ph, pw, 4), np.float32)
    top_c = C.hex2rgb('#d49ac4')
    mid_c = C.hex2rgb('#ffb4a0')
    gold = np.array([1.35, 0.95, 0.56], np.float32)
    pink = C.hex2rgb('#f4a6a0')
    for (x0, x1, y, th, sd) in specs:
        pad = int(4 * th + 8)
        X0, X1 = int(max(x0 - pad, 0)), int(min(x1 + pad, pw))
        Y0, Y1 = int(max(y - 3 * th - 4, 0)), int(min(y + 3 * th + 4, ph))
        if X1 - X0 < 8 or Y1 - Y0 < 4:
            continue
        w, h = X1 - X0, Y1 - Y0
        ys, xs = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        rng = np.random.default_rng(sd)
        L = max(x1 - x0, 1.0)
        tq = (xs - x0) / L
        # 1-D profiles along the streak: thickness swells, centre line waves gently
        n1 = K._noise(w, 4, max(w / (L * 0.18), 2.0), sd + 1, 3)[2]
        n2 = K._noise(w, 4, max(w / (L * 0.06), 2.0), sd + 2, 2)[2]
        thick = th * (0.45 + 0.9 * n1 + 0.3 * (n2 - 0.5))[None, :]
        yc = y + th * 0.6 * np.sin(tq * rng.uniform(2, 4) + rng.uniform(0, 6)) + th * 0.35 * (n2[None, :] - 0.5)
        dy = (ys - yc) / np.maximum(thick, 0.5)
        # flat-ish lit base, soft brushed top
        body = np.where(dy > 0, np.exp(-(dy / 0.55) ** 4), np.exp(-(dy / 1.15) ** 2))
        # torn ends: the end position varies per row (brush drag) -> ragged, frayed tips
        sn = K._noise(w, h, max(w / (th * 9.0), 3.0), sd + 3, 4, stretch=9.0)
        sn2 = K._noise(w, h, max(w / (th * 3.0), 3.0), sd + 4, 3, stretch=6.0)
        endn = (sn - 0.5) * 0.35
        taper = C.smoothstep(0.0, 0.22, tq + endn) * C.smoothstep(1.0, 0.7, tq - endn)
        a = body * taper
        a = K._ss(0.18, 0.55, a * (0.55 + 0.75 * sn2) + 0.25 * a)
        a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), sigmaX=2.5 * u, sigmaY=0.9 * u) * 0.85
        # colour: violet top -> rose -> gold underside; gold hotter toward the sun
        prox = np.exp(-((xs - sun[0]) / (0.22 * pw)) ** 2)
        lit = K._ss(-0.9, 0.3, dy)
        col = top_c * (1 - lit[..., None]) + mid_c * lit[..., None]
        under = K._ss(-0.1, 0.55, dy) * (0.55 + 0.45 * prox)
        uc = pink * (1 - prox[..., None]) + gold * prox[..., None]
        col = col * (1 - under[..., None]) + uc * under[..., None]
        # the part nearest the sun glows through (thin cloud, backlit)
        col = col * (1 + 0.25 * prox[..., None] * (1 - a[..., None]))
        reg = out[Y0:Y1, X0:X1]
        aa = a[..., None]
        prev = reg[..., 3:4]
        al = aa + prev * (1 - aa)
        reg[..., :3] = (col * aa + reg[..., :3] * prev * (1 - aa)) / np.maximum(al, 1e-5)
        reg[..., 3:4] = al
    # bleed colour into transparent margin
    a = out[..., 3:4]
    pb = cv2.GaussianBlur(out * np.concatenate([a, a, a, np.ones_like(a)], -1), (0, 0), 3.0)
    fill = pb[..., :3] / np.maximum(pb[..., 3:4], 1e-5)
    wgt = np.clip(a * 6, 0, 1)
    out[..., :3] = out[..., :3] * wgt + fill * (1 - wgt)
    return out
