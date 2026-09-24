"""Painted night snow-cloud masses for s08_snow_station (built with the shared lib/clouds2 painter).

A low, heavy snow sky is painted as a few big flat value masses: slate-navy bodies whose undersides
catch a faint glow from the lit town below (light comes from under the deck, so the 'sun' of the
painter sits below the frame), crisp overlapping-mass edges, torn soft bases. Toward the horizon a
lighter, hazier band sits behind the far tree line, and a faint warm town glow rises behind the houses
on the right. Returns an additive-free straight-alpha RGBA plate plus a glow/haze RGB plate."""
import math

import numpy as np
import cv2

from lib import clouds2 as K

NIGHT = dict(hi=(0.16, 0.19, 0.38), lit=(0.13, 0.16, 0.34), lit_lo=(0.11, 0.135, 0.3), mid=(0.085, 0.11, 0.26),
             shade=(0.05, 0.068, 0.18), deep=(0.038, 0.052, 0.15), refl=(0.07, 0.09, 0.22),
             bounce=(0.17, 0.15, 0.27), rim=(0.3, 0.32, 0.5), haze=(0.12, 0.17, 0.38), edge_dark=0.02)
NIGHT_FAR = dict(hi=(0.2, 0.26, 0.47), lit=(0.18, 0.24, 0.45), lit_lo=(0.16, 0.22, 0.43), mid=(0.14, 0.2, 0.41),
                 shade=(0.11, 0.16, 0.36), deep=(0.1, 0.15, 0.34), refl=(0.12, 0.17, 0.38),
                 bounce=(0.2, 0.22, 0.4), rim=(0.28, 0.33, 0.52), haze=(0.16, 0.22, 0.45), edge_dark=0.01)


def night_clouds(PW, PH, horizon_v, s, seed=11):
    """RGBA plate (PH, PW, 4) of painted snow-cloud masses above the horizon row horizon_v (plate px).

    Heavy nimbostratus seen from below at night: a few big, flattened, softly lumpy masses in three
    depth rows (high/dark, mid, low/hazy), each painted with the clouds2 painter (flat values, light from
    BELOW - the town and station glow - so the undersides catch a faint warm lavender-amber and the tops
    stay slate-navy), then torn apart along the base (clouds2 wisps: ragged tears, dissolving fractus
    strands) so the undersides read feathered, never as a ruled horizontal band."""
    u = s
    rng = np.random.default_rng(seed + 7)
    sun = (0.45 * PW, horizon_v + 0.55 * PH)            # glow from below (town + station), a little left
    P = K.Painter(PW, PH, sun=sun, pal=NIGHT, unit=u, seed=seed, sun_z=0.05)
    soft_lv = ((0.14, 0.3, 0.8, 1.4, 2.6, 0.08, 0.45), (0.05, 0.1, 0.6, 1.1, 2.4))
    WARMLIT = dict(NIGHT, lit=(0.13, 0.13, 0.28), lit_lo=(0.115, 0.125, 0.27), hi=(0.15, 0.14, 0.28), bounce=(0.16, 0.13, 0.24),
                   rim=(0.24, 0.24, 0.4))
    rows = [  # (y of base as fraction of horizon_v, height frac of PH, width frac of PW, count, dist 0 near..1 far)
        (0.2, 0.13, (0.55, 0.85), 3, 0.0),
        (0.46, 0.1, (0.4, 0.7), 4, 0.45),
        (0.72, 0.07, (0.3, 0.5), 5, 0.85)]
    for (by, hh, (w0, w1), cnt, dist) in rows:
        pal = K.mix_palette(NIGHT, NIGHT_FAR, dist * 0.8)
        pal = K.mix_palette(pal, WARMLIT, 0.25 + 0.6 * dist)       # warmer undersides nearer the town glow
        xs = (np.arange(cnt) + rng.uniform(0.1, 0.9, cnt)) / cnt * 1.3 - 0.15
        for x in xs[rng.permutation(cnt)]:
            cw = rng.uniform(w0, w1) * PW
            ch = hh * PH * rng.uniform(0.7, 1.25)
            yb = by * horizon_v + rng.uniform(-0.05, 0.05) * horizon_v
            cx = x * PW
            # a flattened, lumpy long mass (two or three overlapping sub-masses), painted on its own
            # layer so the torn base below never eats into the clouds already painted (no cut rectangles)
            under = P.prem
            P.prem = np.zeros_like(under)
            nsub = int(rng.integers(2, 4))
            for k in range(nsub):
                ww = cw * rng.uniform(0.45, 0.8)
                ox = rng.uniform(-0.3, 0.3) * cw
                h2 = ch * rng.uniform(0.55, 1.0)
                poly = K.envelope(cx + ox, yb + rng.uniform(-0.1, 0.15) * ch, ww, h2, rng, power=rng.uniform(1.6, 2.1),
                                  lump=0.16, skew=rng.uniform(-0.7, 0.7), base_round=0.12, base_wave=0.12,
                                  top_flat=rng.uniform(0.0, 0.4))
                P.shape(poly, rng=rng, size=h2 * 1.2, pal=pal, levels=soft_lv, haze=0.1 + 0.45 * dist, haze_grad=0.3,
                        group=(cx, yb - 0.4 * ch, 0.6 * cw, 0.9 * ch), form=0.5, split=rng.uniform(0.15, 0.3),
                        lost=0.3, firm=0.35, soften=0.8, scallop=0.5, cast=0.2, rim=0.5, rim_px=2.0, hot=0.2,
                        bounce=0.45, refl=0.35, shade_top=0.4, sky=0.1, brush=0.08, halo=0.25)
            P.wisps(cx - cw * 0.6, cx + cw * 0.6, yb + 0.02 * ch, 0.55 * ch, rng=rng, pal=pal, seed=int(rng.integers(1 << 20)),
                    amount=0.45, erode=1.25, streak=3.5, base_haze=0.35, color_t=0.7)
            top = P.prem
            P.prem = top + under * (1 - np.clip(top[..., 3:4], 0, 1))
        if dist < 0.8:
            P.haze_band(0.0, horizon_v, NIGHT_FAR['haze'], 0.12)
    P.lining(0.3, rim_px=2.0, halo=0.35)
    pl = P.rgba()
    pl[..., :3] *= 1.35                                 # masses read clearly against the night sky
    return feather_undersides(pl, horizon_v, s, seed)


def feather_undersides(pl, horizon_v, s, seed=0):
    """Soft, torn undersides: alpha fades out over a ragged band above every lower edge (distance to the
    sky below, measured straight down), and that band catches a faint warm lavender-amber glow from the
    station and town lights - stronger on the low clouds near the horizon."""
    PH, PW = pl.shape[:2]
    q = 2
    A = cv2.resize(pl[..., 3], (PW // q, PH // q), interpolation=cv2.INTER_AREA)
    h, w = A.shape
    d = np.zeros_like(A)
    run = np.zeros(w, np.float32)
    for y in range(h - 1, -1, -1):
        run = np.where(A[y] > 0.35, run + q, 0.0)
        d[y] = run
    d = cv2.GaussianBlur(d, (0, 0), 1.5)
    rng = np.random.default_rng(seed + 99)
    n = cv2.resize(rng.random((h // 20 + 2, w // 6 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    n2 = cv2.resize(rng.random((h // 8 + 2, w // 12 + 2)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    band = (26.0 * s) * (0.5 + 1.1 * n) + 10.0 * s * (n2 - 0.5)
    fe = np.clip(d / np.maximum(band, 3.0), 0, 1)
    fe = fe * fe * (3 - 2 * fe)
    ys = (np.arange(h, dtype=np.float32)[:, None] * q)
    near = np.clip(ys / max(horizon_v, 1.0), 0, 1) ** 1.5
    glow = np.exp(-d / (40.0 * s)) * (A > 0.05)
    fe = cv2.resize(fe, (PW, PH), interpolation=cv2.INTER_LINEAR)
    glow = cv2.resize((glow * (0.25 + 0.75 * near)).astype(np.float32), (PW, PH), interpolation=cv2.INTER_LINEAR)
    out = pl.copy()
    out[..., 3] = pl[..., 3] * (0.25 + 0.75 * fe)
    warm = np.array([0.15, 0.095, 0.07], np.float32)
    out[..., :3] = pl[..., :3] + glow[..., None] * warm
    return out.astype(np.float32)


def night_glow(PW, PH, horizon_v, f, s):
    """Additive plate: lighter hazy band just above the far tree line + faint warm town glow on the right."""
    us = np.arange(PW, dtype=np.float32)[None, :]
    vs = np.arange(PH, dtype=np.float32)[:, None]
    el = (horizon_v - vs) / f
    band = np.exp(-np.clip(el - 0.05, 0, None) / 0.07) * (el > -0.05)
    band = band * (0.8 + 0.2 * np.sin(us / PW * 5.0 + 0.7))
    haze = band[..., None] * np.array([0.05, 0.075, 0.11], np.float32)
    gx = np.exp(-((us / PW - 0.8) / 0.16) ** 2)
    gy = np.exp(-np.clip(el, 0, None) / 0.09) * (el > -0.08)
    town = (gx * gy)[..., None] * np.array([0.1, 0.07, 0.045], np.float32)
    return (haze + town).astype(np.float32)
