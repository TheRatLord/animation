"""s10 helpers (round 8): the sea of clouds as a few BROAD HORIZONTAL BANKS in 6 depth layers, plus two
backlit towering cumulus, all painted with lib/clouds2.Painter.

Plates are painted in plate px = (frame px + margin) * s  (s = per-layer supersampling so the dolly never
softens the near banks). Sampling reuses s10_sea_of_clouds_c2.sample (zoom about the vanishing point).

Sea (sun dead ahead, low): every bank is a long flattened strip with a cauliflower TOP silhouette only (no
florets on the base), painted in backlit 'crest' mode - the camera-facing body is a cool lavender /
blue-grey shadow, and only a thin gold/peach band runs along the sunward top edge; near the sun axis the
up-facing tops keep a broad smooth gold fill (surfaces angled toward the sun). Far layers: thinner,
flatter, paler, lavender-haze streaks melting into the horizon glow band. Deep indigo valleys only
between the near banks (base_dark + a valley wash painted under each near row). Bases are torn / wispy
(Painter.wisps) and dissolve into the bank below.
Towers: clouds2.cumulonimbus (no anvil) with a side/back light: a warm lit sun-facing flank, a mid
peach-rose band, a cool lavender shadow; crisp overlapping-mass edges (inner lobes); a silver-white
backlit lining on the sunward silhouette and crown; aerial haze + wisps at their bases so they stand IN
the sea.
"""
import math

import numpy as np
import cv2

from lib import core as C, clouds2 as K

M = 0.05          # plate margin (fraction of W / H per side) - must match s10_sea_of_clouds_c2.M

SEA = dict(hi=(1.0, 0.88, 0.62), lit=(1.0, 0.72, 0.46), lit_lo=(0.98, 0.64, 0.52), mid='#e07a98',
           shade='#7a70c0', deep='#262766', refl='#8e88d0', bounce='#c68aa8', rim=(1.45, 1.1, 0.7),
           haze='#e9b4c0', edge_dark=0.05)
SEA_FAR = dict(hi=(1.0, 0.9, 0.78), lit=(1.0, 0.8, 0.66), lit_lo=(0.96, 0.7, 0.72), mid='#e2a0b8',
               shade='#aa98cc', deep='#8a78b8', refl='#b4a6d6', bounce='#dcaabb', rim=(1.2, 1.0, 0.8),
               haze='#f2c2c4', edge_dark=0.02)
TOWER = dict(hi=(1.0, 0.93, 0.8), lit=(1.0, 0.8, 0.6), lit_lo='#f2a08e', mid='#dc7c98', shade='#7f76c4',
             deep='#4a4596', refl='#9d96d8', bounce='#e89c9a', rim=(1.55, 1.42, 1.22), haze='#e6b0c4',
             edge_dark=0.05)


class Plate:
    """A Painter covering the frame + margin at supersampling s. xy() maps frame px -> plate px."""

    def __init__(self, W, H, s, sun, seed, pal=SEA, sun_z=-0.3):
        self.W, self.H, self.s = W, H, s
        self.pw, self.ph = int(round(W * (1 + 2 * M) * s)), int(round(H * (1 + 2 * M) * s))
        self.P = K.Painter(self.pw, self.ph, sun=self.xy(sun), pal=pal, unit=W / 1920.0 * s, seed=seed, sun_z=sun_z)

    def xy(self, p):
        return ((p[0] + M * self.W) * self.s, (p[1] + M * self.H) * self.s)

    def X(self, x):
        return (x + M * self.W) * self.s

    def Y(self, y):
        return (y + M * self.H) * self.s

    def rgba(self):
        return self.P.rgba()


def _levels(f):
    """Floret levels by nearness f (0 far .. 1 near): far = small busy tops, near = a few big heads."""
    if f < 0.25:
        return ((0.05, 0.12, 0.8, 1.1, 2.0), (0.02, 0.04, 0.6))
    if f < 0.6:
        return ((0.08, 0.2, 0.85, 1.3, 2.6, 0.1, 0.45), (0.03, 0.07, 0.6, 1.3, 2.8), (0.012, 0.025, 0.3, 1.5, 3.5))
    return ((0.12, 0.28, 0.9, 1.3, 2.6, 0.1, 0.45), (0.045, 0.09, 0.55, 1.2, 2.6), (0.018, 0.03, 0.15, 1.4, 3.2))


# (layer, y below horizon (fraction of H), bump width (frac W), bump height (frac H), bank length (frac W))
ROWS = [
    (0, 0.002, 0.030, 0.0035, (0.5, 0.9)),
    (0, 0.009, 0.040, 0.0050, (0.5, 0.9)),
    (0, 0.018, 0.050, 0.0070, (0.45, 0.8)),
    (1, 0.030, 0.065, 0.0100, (0.45, 0.8)),
    (1, 0.046, 0.085, 0.0140, (0.4, 0.75)),
    (2, 0.068, 0.11, 0.020, (0.4, 0.7)),
    (2, 0.095, 0.14, 0.027, (0.4, 0.7)),
    (3, 0.135, 0.18, 0.036, (0.45, 0.75)),
    (3, 0.185, 0.22, 0.046, (0.45, 0.75)),
    (4, 0.26, 0.28, 0.06, (0.5, 0.8)),
    (5, 0.36, 0.36, 0.08, (0.55, 0.9)),
    (5, 0.50, 0.46, 0.10, (0.6, 0.95)),
]
NL = 6


def sea(W, H, hy, sun, scales=(1.0, 1.0, 1.0, 1.15, 1.35, 1.6), seed=7, tower_cb=None):
    """Paint the sea into NL plates. Returns list of (Plate, layer). tower_cb(layer, y, plate) is called
    after the first row of each layer (a tower rising out of that layer's banks)."""
    rng = np.random.default_rng(seed)
    plates = [Plate(W, H, scales[k], sun, seed + 10 + k) for k in range(NL)]
    sunx = sun[0]
    done_cb = set()
    floored = set()
    nrows = len(ROWS)
    for ri, (k, dyf, bwf, bhf, Lr) in enumerate(ROWS):
        pl = plates[k]
        P = pl.P
        s = pl.s
        f = ri / (nrows - 1)                       # 0 far .. 1 near
        y = hy + dyf * H
        pal = K.mix_palette(SEA_FAR, SEA, K._ss(0.1, 0.7, f))
        hz = 0.7 * (1 - K._ss(0.0, 0.6, f)) ** 1.3
        bw, bh = bwf * W, bhf * H
        lv = _levels(f)
        fgk = K._ss(0.55, 1.0, f)
        if k not in floored:
            floored.add(k)
            floor(pl, y + 0.5 * bh, pal, hz, f, seed + k)
        x = -0.12 * W - rng.uniform(0, 0.2) * W
        segs = []
        while x < 1.12 * W:
            L = rng.uniform(*Lr) * W
            segs.append((x, x + L, y + rng.uniform(-0.35, 0.25) * bh, rng.uniform(0.75, 1.25)))
            x += L * rng.uniform(0.45, 0.75)
        rng.shuffle(segs)
        for (x0, x1, by, var) in segs:
            cxs = 0.5 * (x0 + x1)
            prox = math.exp(-((cxs - sunx) / (0.22 * W)) ** 2)
            proxw = math.exp(-((cxs - sunx) / (0.5 * W)) ** 2)
            # heaped: a bank is broad and low, with 1-2 taller heaped groups (not a row of equal domes)
            poly = K.billow_strip(pl.X(x0), pl.X(x1), pl.Y(by), bw * s * var, bh * s * var, rng,
                                  depth=bh * s * 2.2, var=0.9 + 0.5 * f, power=2.0, taper=0.3)
            # flatten the profile: compress the top toward the base where the strip is long and low
            ux = 0.3 * float(np.clip((sunx - cxs) / (0.5 * W), -1, 1))
            sun_s = (pl.X(cxs) + ux * 1e5, pl.Y(by) - 1e5)
            bhs = bh * s * var
            crest_px = P.u * (2.0 + 2.0 * f) * (0.7 + 0.6 * proxw) + bhs * 0.3 * prox ** 2 * (1 - 0.6 * f)
            info = P.shape(poly, rng=rng, size=bhs * 1.5, pal=pal, haze=hz, haze_grad=0.0, levels=lv,
                    group=(pl.X(cxs), pl.Y(by) + bhs * 0.3, (pl.X(x1) - pl.X(x0)) * 0.55, bhs * 1.8),
                    form=0.25, mass_r=0.5, smooth=0.04, sun_z=-0.35 + 0.3 * prox,
                    split=0.3 + 0.12 * (1 - prox), lw_base=0.0, firm=0.95, soften=0.15, scallop=0.7, term=0.1,
                    valley_dark=0.35 + 0.15 * (1 - prox) + 0.35 * fgk, lit_cool=0.5 * (1 - prox),
                    valley_span=(0.1, 0.7), light=0.7 + 0.35 * prox, base_dark=0.5 + 0.5 * fgk,
                    lost=0.03 + 0.1 * (1 - f), rim=(0.3 + 0.9 * proxw) if bhs > 6 * P.u else 0.0,
                    rim_px=(1.0 + 1.6 * min(f, 0.7)) * (0.7 + 0.5 * prox), sun=sun_s, side_scale=0.7,
                    down_cut=0.05, cast=0.3, brush=0.05, sun_bias=0.2, refl=0.06, bounce=0.06, shade_top=0.0,
                    inner=0, hot=0.2 + 0.8 * prox, rim_near=1.5, halo=0.5 * proxw, rim_shadow=0.0,
                    rim_interior=0.7 * proxw, sky=0.3, term_w=0.05, wrap=0.0, term_noise=0.05,
                    crest=crest_px, crest_fill=0.85 * prox ** 1.5, crest_amt=0.45 + 0.55 * proxw,
                    concave=0.8, top_bias=1.4)
            if info is not None:
                sheen(P, info, bhs * (0.25 + 0.5 * prox), 0.1 + 0.3 * proxw + 0.35 * prox, pal)
        if k not in done_cb:
            done_cb.add(k)
            if tower_cb is not None:
                tower_cb(k, y, pl)
        # each far row recedes into the lavender/peach haze behind the next one
        if f < 0.5:
            P.haze_band(pl.Y(y - 3 * bh), pl.Y(y + 2 * bh), K._c('#f0c0c4'), 0.25 * (1 - f))
    return plates


def sheen(P, info, D, amt, pal):
    """Backlit translucency under a bank's top silhouette: a soft warm glow that is strongest right under
    the sunward (top) edge and fades down into the lavender body over ~D px (light scattering through the
    thin cloud edge) - turns the crisp crest into a painted gradient instead of an outline."""
    m, x0, y0 = info['mask'], info['x0'], info['y0']
    h, w = m.shape
    if D < 1.5 or amt <= 0:
        return
    u = P.u
    g = np.zeros_like(m)
    for k, wt in ((0.25, 0.5), (0.6, 0.3), (1.0, 0.2)):
        g += wt * m * (1 - K._shift_blur(m, 0.0, -D * k, max(0.25 * D * k, u)))
    g = K._blur(g, max(0.12 * D, u)) * m
    px0, py0 = max(x0, 0), max(y0, 0)
    px1, py1 = min(x0 + w, P.w), min(y0 + h, P.h)
    if px1 <= px0 or py1 <= py0:
        return
    g = g[py0 - y0:py1 - y0, px0 - x0:px1 - x0]
    reg = P.prem[py0:py1, px0:px1]
    col = pal['lit'] * 0.6 + pal['lit_lo'] * 0.4
    k = np.clip(g * amt, 0, 0.8)[..., None] * reg[..., 3:4]
    reg[..., :3] = reg[..., :3] * (1 - k) + col * k


def floor(pl, y_top, pal, hz, f, seed):
    """The layer's own cloud 'surface' behind its banks: an opaque body from a softly undulating line
    y_top (frame px) down to the plate bottom - lavender under the crest line deepening to indigo in the
    valleys, hazed with distance. Never seen directly except as the deep valleys between banks and when
    the dolly pulls a nearer layer out of frame."""
    P = pl.P
    ys = np.arange(P.h, dtype=np.float32)[:, None]
    xs = np.arange(P.w, dtype=np.float32)[None, :]
    rng = np.random.default_rng(seed)
    wav = sum(np.sin(xs[0] / P.w * fq * 6.283 + rng.uniform(0, 6.28)) * a
              for fq, a in ((1.3, 1.0), (3.1, 0.5), (7.3, 0.25)))
    yt = pl.Y(y_top) + wav[None, :] * 0.006 * pl.H * pl.s * (0.3 + f)
    d = (ys - yt) / (0.08 * pl.H * pl.s)
    a = K._ss(-0.15, 0.1, d)
    g = K._ss(0.0, 1.0, d)[..., None]
    col = pal['shade'] * 0.75 * (1 - g) + pal['deep'] * g + pal['deep'] * 0.25 * (1 - g)
    col = col * (1 - hz) + pal['haze'] * hz
    P.prem[..., :3] = P.prem[..., :3] * (1 - a[..., None]) + col * a[..., None]
    P.prem[..., 3] = P.prem[..., 3] * (1 - a) + a


def tower(pl, cx, base_y, Hc, seed=31, width=0.85, haze=0.0, side=1):
    """A towering cumulus rising out of the banks on plate `pl` (frame coords in, painted on pl.P):
    warm lit sun-facing flank, peach-rose mid band, cool lavender shadow, inner lobes (crisp overlapping
    masses), silver-white backlit lining, hazed base with torn wisps."""
    P = pl.P
    s = pl.s
    K.cumulonimbus(P, pl.X(cx), pl.Y(base_y), Hc * s, seed=seed, anvil=False, pal=TOWER, haze=haze,
                   backlit=0.25, width=width, inner=0, wisps=False, sun_z=-0.08, form=0.75, bounce_group=0.45,
                   wrap=0.08, rim_interior=0.2, overshoot=False, term_w=0.12, side=0.8)
    P.lining(1.1, rim_px=2.6, halo=0.8, backlit=0.35, pal=TOWER, y_max=pl.Y(base_y - 0.05 * Hc))
    # aerial haze toward the base: the foot melts into the sea's glow
    ys = np.arange(P.h, dtype=np.float32)
    k = (K._ss(pl.Y(base_y - 0.5 * Hc), pl.Y(base_y), ys) * (1 - K._ss(pl.Y(base_y), pl.Y(base_y + 0.08 * Hc), ys))
         * (0.5 + haze))[:, None, None]
    xs = np.arange(P.w, dtype=np.float32)
    kx = np.exp(-((xs - pl.X(cx)) / (0.7 * Hc * s * width)) ** 2)[None, :, None]
    k = k * kx
    a = P.prem[..., 3:4]
    P.prem[..., :3] = P.prem[..., :3] * (1 - k) + K._c('#eab8c8') * a * k
