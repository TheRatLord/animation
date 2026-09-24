"""Shared sky & atmosphere library (Shinkai-style painted skies).

Everything returns float32 arrays, RGB (H, W, 3) or straight-alpha RGBA (H, W, 4), values ~0..1
(lit cloud tops / sun may exceed 1 slightly - HDR - so bloom picks them up).

Main entry points
-----------------
sky_gradient(W, H, preset, horizon=0.78, sun=None)            -> (H, W, 3) sky backdrop
cumulus_plate(W, H, seed, sun_dir, palette, coverage, kind)    -> (H, W, 4) painted clouds
    (kind: 'cumulus' | 'cumulonimbus' | 'stratocumulus'; `layer=(dmin, dmax)` renders only the clouds
     in that distance range of the SAME layout -> split one sky into far/near plates for parallax;
     `clouds=[...]` places clouds explicitly)
cloud_specs(...)                                               -> the automatic layout, to edit & pass back
horizon_bank(W, H, seed, palette, horizon)                     -> (H, W, 4) receding rows of distant clouds
cirrus_plate(W, H, seed, ...)                                  -> (H, W, 4) hooked mare's-tail cirrus
contrail(W, H, p0, p1, ...)                                    -> (H, W, 4)
starfield(W, H, seed, ...) / milky_way(W, H, seed, ...)        -> (H, W, 3) additive night plates
sun(W, H, x, y, radius, ...) / moon(...)                       -> (H, W, 3) additive disc + glow
birds(W, H, t, ...)                                            -> (H, W) alpha of a small flock
bird_color(sky_rgb)                                            -> deep indigo silhouette colour for birds
drift(plate, W, H, t, speed, cam, zoom, depth, billow)         -> sample a plate with wind + parallax
                                                                  + slow coherent billowing of the lobes
CloudDrift([...])                                              -> layered drifting plates

Cloud model (lib/clouds.py)
---------------------------
Clouds are hierarchies of 3D DOMES: big billows (~0.2 of the cloud width) stacked under a designed
envelope, medium lobes (~0.07) crowning them, small bumps (~0.025) on the silhouettes; sizes shrink
upward and toward the sides. A supersampled hard z-buffer gives crisp edges wherever a dome overlaps
the dome behind it; each pixel is lit by N.L of a normal blended from its dome, the parent dome and the
whole cloud's form, so every dome has its own lit cap and soft terminator while the shadow side stays
one coherent crescent-shaped region. No noise is used in the shading (no blotches). Painted value steps:
shadow (sky-fill tones up, warm bounce down) | half-tone | lit | cap | hot core; a narrow saturated
terminator band; 2-3 px silver lining on sun-facing silhouettes; translucent glow on thin sun-facing
edges (strong when backlit); flat bases with horizontally torn wisps; anvils = flat sheared plates.

Typical use in a scene::

    from lib import core as C, sky as S, fx as F
    class Scene:
        def __init__(self, W, H):
            hz = 0.9
            pw, ph = int(W * 1.2), int(H * 1.15)            # plates larger than the frame (camera moves)
            ox, oy = (pw - W) / 2, (ph - H) / 2
            self.sun = (0.7 * W, 0.15 * H)
            sun_p = (self.sun[0] + ox, self.sun[1] + oy)
            hzp = (hz * H + oy) / ph
            sky_p = S.sky_gradient(pw, ph, 'summer_noon', horizon=hzp, sun=sun_p)
            kw = dict(seed=3, sun_pos=sun_p, palette='summer_noon', coverage=0.5, horizon=hzp, sky=sky_p)
            far  = S.cumulus_plate(pw, ph, layer=(4, 99), **kw)      # distant clouds
            near = S.cumulus_plate(pw, ph, layer=(0, 4), **kw)       # near clouds (same layout)
            bank = S.horizon_bank(pw, ph, seed=5, palette='summer_noon', horizon=hzp, sky=sky_p, sun_pos=sun_p)
            self.sky = sky_p
            self.clouds = S.CloudDrift([(bank, 0.001, 0.1), (far, 0.003, 0.25, 0.0008),
                                        (near, 0.006, 0.5, 0.0012)])   # (plate, speed, depth, billow)
        def frame(self, t):
            u = C.ease_in_out_sine(t / DURATION)
            bg = S.drift(self.sky, W, H, t, 0.0, cam=(0, cam_y), zoom=zoom, depth=0.05)
            img = self.clouds.render(bg, t, cam_x=..., cam_y=..., zoom=1 + 0.05 * u)   # push-in + parallax
            ...
"""
import math
import numpy as np
import cv2

from . import core as C
from . import clouds as K
from .fx import fast_blur as _fblur, over_rgba as _over_rgba

# =============================================================================== palettes

def _c(h):
    return C.hex2rgb(h) if isinstance(h, str) else np.asarray(h, np.float32)


SKY_PRESETS = {
    # stops: (position relative to horizon: 0 = top of frame, 1 = horizon line), colour.
    # band: the luminous, slightly desaturated band just above the horizon (colour, strength)
    'summer_noon': dict(stops=[(0.0, '#0a3aa8'), (0.2, '#1253c4'), (0.42, '#2a78dc'), (0.62, '#55a2ea'),
                               (0.8, '#8ccaf2'), (0.92, '#b4e2f6'), (1.0, '#cdeef8')],
                        sun_glow='#fff8e6', sun_glow_amt=0.2, below='#d4f0f8', band=('#eaf8fb', 0.55)),
    'magic_hour': dict(stops=[(0.0, '#1a3290'), (0.22, '#2c52b4'), (0.44, '#5674cc'), (0.6, '#9884cf'),
                              (0.73, '#e184b4'), (0.84, '#ff9c88'), (0.93, '#ffb870'), (1.0, '#ffd684')],
                       sun_glow='#ffb05a', sun_glow_amt=0.18, below='#ffd488', band=('#ffe6b0', 0.3)),
    'sunset': dict(stops=[(0.0, '#121856'), (0.25, '#272e8a'), (0.45, '#5c3c9c'), (0.6, '#ad4798'),
                          (0.72, '#ec5484'), (0.83, '#ff7a55'), (0.92, '#ffa048'), (1.0, '#ffcc66')],
                   sun_glow='#ffb050', sun_glow_amt=0.3, below='#ffc862', band=('#ffe3a0', 0.4)),
    'dawn': dict(stops=[(0.0, '#1f3a82'), (0.3, '#4a6cc0'), (0.55, '#9a94d6'), (0.72, '#e2a0c6'),
                        (0.86, '#ffbcae'), (0.95, '#ffd6b8'), (1.0, '#ffe6c8')],
                 sun_glow='#ffe4c4', sun_glow_amt=0.3, below='#fbdcc4', band=('#fff0e2', 0.4)),
    'night': dict(stops=[(0.0, '#010414'), (0.35, '#051033'), (0.7, '#0e2156'), (0.9, '#1d3a78'),
                         (1.0, '#2e5391')],
                  sun_glow='#8fb0ff', sun_glow_amt=0.12, below='#2e5391', band=('#4a70b0', 0.3)),
}

# Cloud palettes.
#   lit_low / lit_high : sunlit side near the base / at the tops      hi   : hot core of the light
#   mid                : terminator band + lobe lines (saturated: pink at sunset)
#   shd_high / shd_low : shadow side lit by the sky above / deep core   bounce: warm light from below
#   base               : flat underside      rim : silver lining / translucent glow
#   haze               : atmospheric perspective colour for distant clouds      fill: sky colour fallback
CLOUD_PALETTES = {
    'summer_noon': dict(lit_low='#f6f1ea', lit_high='#ffffff', hi='#fffdf8', mid='#b6c0ea',
                        shd_high='#98abe6', shd_low='#6070c4', bounce='#b8bde6', base='#8f95cf',
                        rim='#fffcf2', haze='#c4e4f6', fill='#8ec0f0', hdr=1.0),
    'magic_hour': dict(lit_low='#ff9468', lit_high='#ffc294', hi='#ffe2b4', mid='#f07890',
                       shd_high='#a48ad0', shd_low='#62539e', bounce='#e08aa8', base='#80649e',
                       rim='#fff0c4', haze='#f2b2a6', fill='#8f8fd0', hdr=1.0),
    'sunset': dict(lit_low='#ff6c40', lit_high='#ffa262', hi='#ffcc8c', mid='#e8588a',
                   shd_high='#8a64a6', shd_low='#46307a', bounce='#c8628a', base='#5e3c78',
                   rim='#ffdc9c', haze='#f7a07c', fill='#7a6fb8', hdr=1.0),
    'dawn': dict(lit_low='#ffcab6', lit_high='#fff3e8', hi='#fffaf2', mid='#f0b2c0',
                 shd_high='#b6b0dc', shd_low='#8580b8', bounce='#d8b0c4', base='#9a92c0',
                 rim='#fff2e2', haze='#ecc6cc', fill='#b4b8e6', hdr=1.0),
    'night': dict(lit_low='#5a6c9c', lit_high='#8ea2cf', hi='#a8b8e0', mid='#44538a',
                  shd_high='#2b3a66', shd_low='#1a2448', bounce='#26325c', base='#141d3c',
                  rim='#b8c8f0', haze='#263e78', fill='#23376a', hdr=1.0),
}


def _palette(p):
    """Palette name or dict -> dict of float32 RGB arrays (missing keys derived)."""
    if isinstance(p, str):
        p = CLOUD_PALETTES[p]
    p = dict(p)
    p.setdefault('hi', p.get('lit_high'))
    p.setdefault('bounce', p.get('shd_high'))
    if 'mid' not in p:
        p['mid'] = (_c(p['lit_low']) + _c(p['shd_high'])) * 0.5
    p.setdefault('hdr', 1.0)
    return {k: (_c(v) if k not in ('hdr', 'lobe') else float(v)) for k, v in p.items()}


# =============================================================================== sky gradient

def sky_gradient(W, H, preset='summer_noon', horizon=0.78, sun=None, sun_radius=0.55, variation=0.01, seed=0,
                 horizon_glow=1.0):
    """Painted sky backdrop.

    params: W, H - size; preset - key of SKY_PRESETS or a dict of the same shape;
            horizon - horizon line as fraction of H (gradient reaches its horizon colour there);
            sun - optional (x, y) pixel position of the sun: adds a broad warm brightening around it
                  (the sky near the sun is paler/warmer);
            sun_radius - size of that brightening as fraction of W;
            variation - amplitude of very soft large-scale tonal variation (avoids flat CG gradients);
            horizon_glow - strength multiplier of the luminous, slightly desaturated band just above
                  the horizon (atmospheric scattering).
    returns: (H, W, 3) float32. The gradient is smoothed (no linear-blend kinks between stops).
    """
    P = SKY_PRESETS[preset] if isinstance(preset, str) else preset
    hy = max(horizon * H, 1.0)
    stops = [(p * hy / H, c) for p, c in P['stops']] + [(1.0, P['below'])]
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([_c(s[1]) for s in stops], np.float32)
    # build the 1D profile with margin, smooth it (C1-continuous), then broadcast
    pad = H // 4
    y = (np.arange(-pad, H + pad, dtype=np.float32) + 0.5) / H
    prof = np.stack([np.interp(y, pos, cols[:, c]) for c in range(3)], -1).astype(np.float32)
    prof = cv2.GaussianBlur(prof[:, None, :], (1, 0), sigmaX=0.1, sigmaY=0.035 * H)[:, 0, :]
    prof = prof[pad:pad + H]
    ys1 = np.arange(H, dtype=np.float32)
    if horizon_glow and 'band' in P:
        bc, bs = P['band']
        d = (ys1 - hy) / H
        band = np.where(d < 0, np.exp(-(d / 0.09) ** 2), np.exp(-(d / 0.03) ** 2)) * bs * horizon_glow
        prof = C.lerp(prof, _c(bc), band[:, None])
    img = np.repeat(prof[:, None, :], W, axis=1)
    if sun is not None:
        xs, ys = C.grid(W, H)
        d = np.sqrt((xs - sun[0]) ** 2 + (ys - sun[1]) ** 2) / W
        g = np.exp(-d / (0.07 * sun_radius / 0.55)) * 0.6 + np.exp(-d / (0.3 * sun_radius / 0.55)) * 0.25
        band = np.exp(-((ys - hy) / (0.22 * H)) ** 2) * np.exp(-((xs - sun[0]) / (0.6 * W)) ** 2)
        a = (P['sun_glow_amt'] * (g + 0.3 * band))[..., None]
        img = C.screen(img, _c(P['sun_glow']) * np.clip(a, 0, 0.5))
    if variation:
        n = C.fbm(max(W // 8, 8), max(H // 8, 8), 2.0, 3, seed=seed + 71) - 0.5
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        img = img * (1 + variation * n[..., None] * np.array([0.6, 0.8, 1.0], np.float32) * 2)
    return img.astype(np.float32)


# =============================================================================== cloud layout

def cloud_specs(W, H, seed=0, kind='cumulus', coverage=0.5, horizon=0.78, hero_x=None, fragments=1.0,
                strato=0.5):
    """Automatic cloud layout with strong scale variety (one hero, a few mid clouds, many small fragments,
    stratocumulus streaks low near the horizon). Returns a list of specs
    (cx, base_y, width, height, dist, kind, flat) in px that cumulus_plate(clouds=...) accepts - edit it
    (e.g. drop a cloud that covers your subject) and pass it back.

    params: kind - 'cumulus' | 'cumulonimbus' (adds a hero tower + anvil) | 'stratocumulus';
            coverage 0..1; horizon - fraction of H; hero_x - hero centre as fraction of W (None = random);
            fragments - multiplier for small scattered fragments; strato - amount of low streaks.
    """
    rng = np.random.default_rng(seed + 7)
    hy = horizon * H
    specs = []
    placed = []

    def free(x, w, d):
        for (px, pw, pd) in placed:
            if abs(pd - d) < 0.35 * max(pd, d) and abs(px - x) < 0.45 * (pw + w):
                return False
        return True

    if kind == 'cumulonimbus':
        hx = hero_x * W if hero_x is not None else rng.uniform(0.35, 0.65) * W
        wdt = W * rng.uniform(0.36, 0.44)
        specs.append((hx, hy - H * 0.02, wdt, min(hy * 0.84, wdt * 2.0), 2.0, 'cumulonimbus', 1.0))
        placed.append((hx, wdt, 2.0))
    elif kind == 'cumulus' and coverage > 0.2:
        # a hero cumulus bank
        hx = hero_x * W if hero_x is not None else rng.uniform(0.25, 0.75) * W
        wdt = W * rng.uniform(0.3, 0.42)
        by = hy - (hy - 0.25 * H) * rng.uniform(0.25, 0.45)
        specs.append((hx, by, wdt, wdt * rng.uniform(0.45, 0.6), 2.2, 'cumulus', 1.0))
        placed.append((hx, wdt, 2.2))
    # mid-size clouds
    n_mid = int(2 + coverage * 6)
    tries = 0
    while n_mid > 0 and tries < 300:
        tries += 1
        f = rng.random() ** 0.8
        base = H * 0.22 + f * (hy - H * 0.25)
        d = float(np.clip((hy - H * 0.08) / max(hy - base, 1e-3) * 0.9, 1.2, 12.0))
        wdt = W * rng.uniform(0.12, 0.26) / d ** 0.7
        if kind == 'stratocumulus':
            hgt, k, flat = wdt * rng.uniform(0.18, 0.28), 'stratocumulus', float(np.clip(0.8 - d * 0.03, 0.4, 0.75))
        else:
            hgt, k, flat = wdt * rng.uniform(0.35, 0.65), 'cumulus', float(np.clip(1.05 - d * 0.04, 0.6, 1.0))
        x = rng.uniform(-0.05, 1.05) * W
        if not free(x, wdt, d):
            continue
        placed.append((x, wdt, d))
        specs.append((x, base, wdt, hgt, d, k, flat))
        n_mid -= 1
    # small scattered fragments
    n_frag = int((4 + coverage * 14) * fragments)
    tries = 0
    while n_frag > 0 and tries < 500:
        tries += 1
        f = rng.random() ** 0.6
        base = H * 0.12 + f * (hy - H * 0.14)
        d = float(np.clip((hy - H * 0.06) / max(hy - base, 1e-3) * 1.0, 1.5, 14.0))
        wdt = W * rng.uniform(0.025, 0.07) / d ** 0.4
        x = rng.uniform(-0.03, 1.03) * W
        if not free(x, wdt, d):
            continue
        placed.append((x, wdt, d))
        specs.append((x, base, wdt, wdt * rng.uniform(0.25, 0.5), d, 'cumulus' if rng.random() < 0.6 else 'stratocumulus',
                      float(np.clip(0.9 - d * 0.03, 0.5, 0.9))))
        n_frag -= 1
    # stratocumulus streaks low near the horizon
    n_st = int(strato * (3 + coverage * 6))
    for _ in range(n_st):
        base = hy - H * rng.uniform(0.02, 0.12)
        d = float(np.clip((hy - H * 0.06) / max(hy - base, 1e-3), 6.0, 14.0))
        wdt = W * rng.uniform(0.1, 0.25)
        specs.append((rng.uniform(0, 1) * W, base, wdt, wdt * rng.uniform(0.05, 0.09), d, 'stratocumulus', 0.35))
    return specs


def _light_for(cx, cy, sun_dir, sun_z, sun_pos, backlit, backlit_radius, W):
    """Per-cloud light vector (toward the sun) + backlit amount."""
    if sun_pos is not None:
        dx, dy = sun_pos[0] - cx, sun_pos[1] - cy
        dist = math.hypot(dx, dy) + 1e-3
        dx, dy = dx / dist, dy / dist
        lz = sun_z - backlit * math.exp(-dist / (backlit_radius * W))
    else:
        d = np.asarray(sun_dir, np.float64)
        d = d / (np.linalg.norm(d) + 1e-9)
        dx, dy = float(d[0]), float(d[1])
        lz = sun_z
    lz = float(np.clip(lz, -0.95, 0.95))
    c = math.sqrt(1 - lz * lz)
    return (dx * c, dy * c, lz), float(np.clip(-lz * 1.6, 0, 1))


# =============================================================================== cumulus plate

def cumulus_plate(W, H, seed=0, sun_dir=(0.6, -0.7), sun_z=0.3, palette='summer_noon', coverage=0.5,
                  kind='cumulus', horizon=0.78, hero_x=None, sun_pos=None, detail=1.0, rim=1.0,
                  clouds=None, texture=1.0, haze=None, ss=2, sky=None, lit_bias=0.0, layer=None, halo=1.0,
                  depth_fog=1.0, backlit=0.0, backlit_radius=0.25, wind=1.0, soft=0.22, return_attrs=False,
                  **_ignored):
    """Painted Shinkai-style clouds as a straight-alpha RGBA plate (see module docstring for the model).

    params:
      W, H      plate size in px (make it larger than the frame if you want to pan/drift it)
      seed      layout / shape seed (deterministic)
      sun_dir   2D screen direction pointing TOWARD the sun (x right, y down), e.g. (0.6,-0.7) = up-right
      sun_z     sun's z component: >0 sun behind the viewer (front-lit, more of each cloud lit), <0 sun
                behind the clouds (backlit: dark bodies with blazing edges). 0.2-0.4 typical.
      sun_pos   optional (x, y) sun position in PLATE px: each cloud is lit from the direction of the sun
                as seen from that cloud (use for sun-in-frame shots; overrides sun_dir); silver linings
                and halation are strongest on edges near the sun.
      backlit   0..1: clouds close to sun_pos become backlit silhouettes (sun_z reduced by up to this
                amount within ~backlit_radius*W of the sun) - dark violet bodies with gold rims at sunset.
      palette   key of CLOUD_PALETTES ('summer_noon','magic_hour','sunset','dawn','night') or dict
      coverage  0..1 amount of cloud (automatic layout)
      kind      'cumulus' | 'cumulonimbus' (hero tower + anvil + smaller clouds) | 'stratocumulus'
      horizon   horizon line (fraction of H); clouds shrink/flatten/haze towards it
      hero_x    x of the hero cloud as fraction of W (None = random)
      detail    lobe density multiplier (1 = normal, 1.2+ adds a 4th level of tiny bumps)
      rim       silver-lining / translucency strength multiplier
      clouds    optional explicit list of (cx, base_y, width, height, dist, kind, flat) (px; dist ~1 near
                .. 14 far; flat<1 squashes lobes for perspective) - see cloud_specs()
      texture   very subtle large-scale tonal texture (0 = none)
      haze      optional RGB colour for atmospheric fade (defaults to the palette's haze)
      ss        supersampling of the raster (2 = crisp anti-aliased edges)
      sky       optional (H, W, 3) sky image behind the plate: ~15% of it is mixed into shadow sides
      lit_bias  shifts the terminator: +0.1 = more of each cloud lit, -0.1 = less
      layer     optional (dmin, dmax): keep only clouds whose distance is in [dmin, dmax). Same seed +
                args => same layout, so layer=(0,4) and layer=(4,99) split one sky into near/far plates.
      halo      strength of the soft halation just outside sun-facing edges (0 disables)
      depth_fog atmospheric-perspective multiplier (distant clouds lighter/bluer/lower contrast)
      wind      +1 / -1: direction the cumulonimbus anvil is sheared toward
      soft      terminator softness (N.L units); dome overlaps and silhouettes stay crisp regardless
    returns: (H, W, 4) float32 straight alpha RGBA (and the attribute buffer if return_attrs).
             ~2-5 s for a 2300x1250 plate.
    """
    pal = _palette(palette)
    if haze is not None:
        pal['haze'] = _c(haze)
    rng = np.random.default_rng(seed)
    specs = clouds if clouds is not None else cloud_specs(W, H, seed, kind, coverage, horizon, hero_x)
    B = K.Builder()
    # far clouds first (painter's order is handled by z anyway)
    order = sorted(range(len(specs)), key=lambda i: -specs[i][4])
    for i in order:
        cx, by, wd, ht, d, k, flat = specs[i]
        crng = np.random.default_rng(seed * 1000 + i)
        if layer is not None and not (layer[0] <= d < layer[1]):
            continue
        fog = float(np.clip((d - 1.5) / 9.0, 0, 1)) ** 0.75 * 0.85 * depth_fog
        L, back = _light_for(cx, by - ht * 0.5, sun_dir, sun_z, sun_pos, backlit, backlit_radius, W)
        z0 = -d * W * 3.0
        K.gen_cloud(B, crng, (cx, by, wd, ht, d, k, flat), L, fog, back,
                    detail=detail * (1.0 if d < 4 else 0.8), z0=z0, wind=wind)
    if not B.S:
        z = np.zeros((H, W, 4), np.float32)
        return (z, None) if return_attrs else z
    A = K.render(B, W, H, ss=ss)
    if sky is None:
        sky_img = None
    else:
        sky_img = np.asarray(sky, np.float32)[..., :3]
        if sky_img.shape[:2] != (H, W):
            sky_img = cv2.resize(sky_img, (W, H), interpolation=cv2.INTER_LINEAR)
    P = K.paint(A, pal, sky=sky_img, seed=seed, rim=rim, halo=halo, terminator=lit_bias, soft=soft,
                sun_pos=sun_pos, sun_dir=sun_dir, texture=texture)
    return (P, A) if return_attrs else P


def horizon_bank(W, H, seed=0, palette='summer_noon', horizon=0.8, height=0.09, density=1.0, sun_dir=(0.5, -0.8),
                 sun_z=0.4, sun_pos=None, sky=None, haze=None, rows=3, fade=0.5, backlit=0.0, **kw):
    """Receding rows of distant cumulus sitting on the horizon: flat bases, each row further away is
    smaller, flatter, lower contrast and bluer (atmospheric perspective); the lowest parts melt into the
    horizon haze.

    params: horizon - horizon line (fraction of H); height - tallest cloud height in the nearest row
            (fraction of H); density - clouds multiplier; rows - number of receding rows;
            fade - how much the bank melts into haze near the horizon; other params as cumulus_plate.
    returns: (H, W, 4) straight alpha.
    """
    rng = np.random.default_rng(seed + 1000)
    hy = horizon * H
    specs = []
    for r in range(rows):
        f = r / max(rows - 1, 1)                       # 0 nearest row .. 1 farthest
        d = 5.0 + 5.0 * f
        scale = 1.0 - 0.55 * f
        by = hy - H * (0.03 - 0.028 * f) * (1 if r < rows - 1 else 0.2)
        x = -0.06 * W + rng.uniform(0, 0.05) * W
        while x < 1.06 * W:
            wdt = W * rng.uniform(0.03, 0.14) ** 1.0 * scale * (1.8 if rng.random() < 0.12 else 1.0)
            hgt = min(wdt * rng.uniform(0.2, 0.6), height * H * scale * rng.uniform(0.5, 1.2))
            if rng.random() < 0.25:
                x += wdt * rng.uniform(0.5, 1.5)           # gaps between groups
            specs.append((x, by + rng.uniform(-0.004, 0.004) * H, wdt, hgt, d, 'cumulus',
                          float(0.7 - 0.25 * f)))
            x += wdt * rng.uniform(0.45, 0.9) / max(density, 0.2)
    p = cumulus_plate(W, H, seed=seed, sun_dir=sun_dir, sun_z=sun_z, sun_pos=sun_pos, palette=palette,
                      clouds=specs, sky=sky, haze=haze, texture=0.5, detail=0.9, halo=0.4, rim=0.7,
                      backlit=backlit, **kw)
    ys = np.arange(H, dtype=np.float32)[:, None]
    f = C.smoothstep(hy + H * 0.005, hy - height * H * 0.8, ys)    # 1 high up, 0 at the horizon
    hz = _palette(palette)['haze'] if haze is None else _c(haze)
    if sky is not None:
        hz = C.lerp(hz, np.asarray(sky, np.float32)[..., :3], 0.6)
    p[..., :3] = C.lerp(hz, p[..., :3], (1 - fade + fade * f)[..., None])
    return p.astype(np.float32)


# =============================================================================== cirrus / contrails

def _brush_noise(W, H, seed, angle_deg=-12.0, stretch=5.0, cells=40):
    """Directional 'brush stroke' noise: fbm stretched along angle_deg. 0..1."""
    D = int(math.hypot(W, H)) + 4
    q = 2
    n = C.fbm(max(int(D / q / stretch), 8), D // q, cells / stretch, 3, seed=seed, aspect=False)
    n = cv2.resize(n, (D, D), interpolation=cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((D / 2, D / 2), 90 - angle_deg, 1.0)
    M[0, 2] += W / 2 - D / 2
    M[1, 2] += H / 2 - D / 2
    return cv2.warpAffine(n, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)




def _stroke(canvas, pts, widths, vals, ss):
    """Draw a tapered polyline (per-segment width/value) into canvas (max-blend)."""
    for i in range(len(pts) - 1):
        th = max(1, int(round(widths[i] * ss)))
        p0 = (int(pts[i][0] * ss), int(pts[i][1] * ss))
        p1 = (int(pts[i + 1][0] * ss), int(pts[i + 1][1] * ss))
        tmp_v = float(vals[i])
        if tmp_v <= 0.003:
            continue
        cv2.line(canvas, p0, p1, tmp_v, th, cv2.LINE_AA)


def cirrus_plate(W, H, seed=0, color=(1.0, 0.98, 0.95), angle=-8.0, density=0.5, region=(0.0, 0.55),
                 streak=0.012, opacity=0.65, curl=1.0, width_scale=1.0, clusters=None, shadow=None,
                 under=None, veil=0.35):
    """High cirrus painted as hooked mare's tails: each tail has a dense comma-shaped head (hook) from
    which a fan of fibrous filaments trails downwind, curving and thinning; widths and opacities vary
    along each filament and the ends fray into fibres. A faint cirrostratus veil sits under the tails.

    params: angle - main trailing direction in degrees (0 = horizontal, negative = rising to the right);
            density 0..1 - tails per cluster; region - (top, bottom) vertical band (fractions of H);
            streak - directional smear (fraction of W) that silkens strokes; opacity - max alpha;
            curl - hook / fall-streak curvature (0 = straight); width_scale - filament width mult;
            clusters - number of tail clusters (default 2-3); color - head colour (light side);
            under - optional RGB for the undersides / trailing fibres (e.g. pink/gold at sunset: they
            catch the low sun) - default = color; shadow - optional RGB mixed into the densest cores;
            veil - strength of the soft cirrostratus veil.
    returns: (H, W, 4) straight alpha. ~0.5-1 s at 1080p.
    """
    rng = np.random.default_rng(seed)
    q = 2
    w, h = W // q, H // q
    ss = 2
    acc = np.zeros((h * ss, w * ss), np.float32)
    und = np.zeros((h * ss, w * ss), np.float32)
    ncl = clusters if clusters is not None else 2 + int(rng.random() < density)
    th0 = math.radians(angle)
    for ci in range(ncl):
        cx = (0.1 + 0.8 * (ci + rng.uniform(0.15, 0.85)) / ncl) * w
        cy = rng.uniform(region[0] + 0.06, region[1] - 0.04) * h
        ntail = int(3 + density * 7)
        for ti in range(ntail):
            hx = cx + rng.normal(0, 0.12) * w
            hy = cy + rng.normal(0, 0.05) * h
            L = w * rng.uniform(0.12, 0.3) * (1.0 if ti else 1.4)
            head_w = h * rng.uniform(0.014, 0.03) * width_scale
            hook = curl * rng.uniform(0.6, 1.4) * rng.choice([-1, 1], p=[0.3, 0.7])
            nf = int(rng.integers(8, 16))
            for fi in range(nf):
                npt = 36
                tt = np.linspace(0, 1, npt)
                fan = rng.normal(0, 1) * 0.12                       # filaments fan out with distance
                a = th0 + math.pi + fan * tt ** 0.8                  # trail back (upwind side)
                a = a + hook * 0.9 * (1 - tt) ** 3 * (1 if fi % 2 else 0.8)   # hook at the head
                a = a + curl * 0.25 * tt ** 2 * rng.uniform(-1, 1)
                lf = L * rng.uniform(0.45, 1.0)
                step = lf / (npt - 1)
                off = rng.normal(0, 1) * head_w * 0.45
                px_ = hx + off * -math.sin(th0) + np.concatenate([[0], np.cumsum(np.cos(a[:-1]) * step)])
                py_ = hy + off * math.cos(th0) + np.concatenate([[0], np.cumsum(np.sin(a[:-1]) * step)])
                # fall streaks droop slightly
                py_ = py_ + curl * h * 0.012 * tt ** 2 * rng.uniform(0.2, 1.0)
                wid = head_w * rng.uniform(0.35, 1.0) * (1 - tt) ** 0.9 * (0.6 + 0.4 * np.sin(tt * rng.uniform(5, 12) + rng.uniform(0, 6)) ** 2) + 0.35
                op = rng.uniform(0.2, 0.6) * (1 - tt) ** 0.6 * (0.55 + 0.45 * np.sin(tt * rng.uniform(4, 9) + rng.uniform(0, 6)) ** 2)
                op = op * C.smoothstep(0.0, 0.08, tt)
                pts = np.stack([px_, py_], 1)
                _stroke(acc, pts, wid, op, ss)
                _stroke(und, pts + [0, 0.4 + head_w * 0.35], wid, op * C.smoothstep(0.05, 0.4, tt), ss)
    a = cv2.resize(acc, (w, h), interpolation=cv2.INTER_AREA)
    u = cv2.resize(und, (w, h), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
    u = cv2.resize(u, (W, H), interpolation=cv2.INTER_CUBIC)
    a = C.blur(np.clip(a, 0, 1), 1.6 * max(W / 1920, 0.5))
    # silken along the stroke direction
    Lk = max(3, int(streak * W)) | 1
    ker = np.zeros((Lk, Lk), np.float32)
    ker[Lk // 2, :] = np.hanning(Lk) + 1e-3
    M = cv2.getRotationMatrix2D((Lk / 2 - 0.5, Lk / 2 - 0.5), -angle, 1.0)
    ker = cv2.warpAffine(ker, M, (Lk, Lk))
    ker /= ker.sum() + 1e-9
    a = 0.5 * cv2.filter2D(a, -1, ker) + 0.5 * a
    # fibres combed along the direction
    comb = _brush_noise(W, H, seed + 5, angle_deg=angle, stretch=12.0, cells=220 * max(W / 1920, 0.5))
    a = a * (0.65 + 0.6 * comb)
    if veil:
        vn = _brush_noise(W, H, seed + 9, angle_deg=angle, stretch=6.0, cells=10)
        a = np.maximum(a, C.smoothstep(0.45, 0.9, vn) * 0.18 * veil)
    a = 1 - np.exp(-a * 1.5)
    ys = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    band = C.smoothstep(region[0] - 0.1, region[0] + 0.04, ys) * (1 - C.smoothstep(region[1] - 0.04, region[1] + 0.12, ys))
    a = np.clip(a * band, 0, 1) * opacity
    col = np.broadcast_to(np.asarray(color, np.float32), (H, W, 3)).copy()
    if under is not None:
        uu = np.clip(C.blur(u, 0.8), 0, 1)
        k = np.clip(uu * 1.4, 0, 1) * 0.55 + 0.2
        col = C.lerp(col, np.asarray(under, np.float32), k[..., None])
    if shadow is not None:
        cc = C.smoothstep(0.35, 0.8, a / max(opacity, 1e-3))[..., None]
        col = C.lerp(col, np.asarray(shadow, np.float32), cc * 0.35)
    return np.dstack([col, a]).astype(np.float32)


def contrail(W, H, p0, p1, width=0.003, spread=4.0, color=(1.0, 1.0, 1.0), opacity=0.8, seed=0, curve=0.0):
    """Aircraft contrail from p0 (old end, dispersed) to p1 (the plane - sharp, thin).

    params: p0, p1 - (x, y) pixels; width - core width at the plane end, fraction of W; spread - how
            much wider the old end gets; curve - sideways bow in px; color/opacity; seed.
    returns: (H, W, 4) straight alpha. Animate by moving p1 over time (cheap to regenerate at 1/2 res,
            or pre-render long and reveal with a mask).
    """
    n = 160
    tt = np.linspace(0, 1, n)
    px = p0[0] + (p1[0] - p0[0]) * tt
    py = p0[1] + (p1[1] - p0[1]) * tt
    nx, ny = -(p1[1] - p0[1]), (p1[0] - p0[0])
    nl = math.hypot(nx, ny) + 1e-6
    bow = curve * 4 * tt * (1 - tt)
    px += nx / nl * bow
    py += ny / nl * bow
    a = np.zeros((H, W), np.float32)
    w0 = width * W
    ss = 2
    for layer, (off, wmul, amp) in enumerate([(0, 1.0, 1.0), (1, 1.0, 1.0)]):
        m = np.zeros((H * ss, W * ss), np.float32)
        side = (off - 0.5) * 1.6 * w0
        for i in range(n - 1):
            age = 1 - tt[i]
            w = w0 * (1 + spread * age ** 1.3) * wmul
            sx = nx / nl * side * (1 - age * 0.7)
            sy = ny / nl * side * (1 - age * 0.7)
            inten = (1 - age) ** 0.3 * (1 - 0.6 * age) * amp
            cv2.line(m, (int((px[i] + sx) * ss), int((py[i] + sy) * ss)),
                     (int((px[i + 1] + sx) * ss), int((py[i + 1] + sy) * ss)), float(inten),
                     max(1, int(w * ss)), cv2.LINE_AA)
        a = np.maximum(a, cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA))
    # dispersal: break the older part with noise + blur more
    nz = C.fbm(W, H, 40, 3, seed=seed)
    a_soft = C.blur(a, 0.002 * W)
    xs, ys = C.grid(W, H)
    L2 = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + 1e-6
    tpos = np.clip(((xs - p0[0]) * (p1[0] - p0[0]) + (ys - p0[1]) * (p1[1] - p0[1])) / L2, 0, 1)
    age = 1 - tpos
    a = C.lerp(a, a_soft * (0.5 + nz), np.clip(age * 1.2, 0, 1))
    a = np.clip(a, 0, 1) * opacity
    col = np.broadcast_to(np.asarray(color, np.float32), (H, W, 3))
    return np.dstack([col, a]).astype(np.float32)


# =============================================================================== night

def starfield(W, H, seed=0, density=1.0, horizon=0.8, twinkle_t=None, brightness=1.0):
    """Additive star plate (H, W, 3): power-law magnitudes, subtle colour temperatures, a few bright
    stars with tiny cross glints; fades toward the horizon.

    params: density - star count multiplier; horizon - fraction of H where stars fade out;
            twinkle_t - pass the scene time to get gentle, temporally coherent twinkle (each star has its
            own slow phase) - costs a re-splat (~0.2 s at 1080p), otherwise static;
            brightness - overall multiplier.
    """
    rng = np.random.default_rng(seed)
    n = int(2600 * density * W * H / (1920 * 1080))
    xs = rng.random(n) * W
    ys = (rng.random(n) ** 1.25) * H * horizon
    mag = rng.pareto(2.0, n) * 0.1 + 0.03
    mag = np.minimum(mag, 1.4)
    temp = rng.random(n)
    cols = np.where(temp[:, None] < 0.2, [1.0, 0.82, 0.7], np.where(temp[:, None] > 0.75, [0.75, 0.85, 1.0], [1, 1, 1]))
    if twinkle_t is not None:
        ph = rng.random(n) * 6.28
        fr = rng.uniform(0.6, 2.0, n)
        mag = mag * (0.75 + 0.25 * np.sin(twinkle_t * fr + ph))
    img = np.zeros((H, W, 3), np.float32)
    r = np.clip(0.55 + mag * 1.2, 0.5, 2.2) * W / 1920
    C.splat(img, xs, ys, np.maximum(r, 0.45), cols.astype(np.float32), mag * 4.0 * brightness)
    # a few bright ones get glints
    big = np.argsort(mag)[-max(3, n // 250):]
    gl = np.zeros_like(img)
    for i in big:
        C.splat(gl, xs[i:i + 1], ys[i:i + 1], 2.5 * W / 1920, cols[i].astype(np.float32), mag[i] * 1.2)
    kx = cv2.getGaussianKernel(max(3, int(0.012 * W) | 1), 0.004 * W).T
    gx = cv2.filter2D(gl, -1, kx / kx.max()) * 0.15
    gy = cv2.filter2D(gl, -1, kx.T / kx.max()) * 0.15
    img = img + gl + gx + gy
    fade = 1 - C.smoothstep(horizon * 0.6, horizon, np.linspace(0, 1, H, dtype=np.float32))
    return (img * fade[:, None, None]).astype(np.float32)


def _rotated_fbm(W, H, angle_deg, stretch, scale, octaves, seed):
    """fbm whose features are stretched `stretch`x along direction angle_deg (for bands/lanes)."""
    D = int(math.hypot(W, H)) + 4
    n = C.fbm(max(D // int(max(stretch, 1)), 8), D, scale / max(stretch, 1), octaves, seed=seed, aspect=False)
    n = cv2.resize(n, (D, D), interpolation=cv2.INTER_CUBIC)
    M = cv2.getRotationMatrix2D((D / 2, D / 2), -angle_deg, 1.0)
    M[0, 2] += W / 2 - D / 2
    M[1, 2] += H / 2 - D / 2
    return np.clip(cv2.warpAffine(n, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), 0, 1)


def milky_way(W, H, seed=0, angle=-35.0, center=(0.55, 0.35), width=0.16, strength=1.0):
    """Additive Milky Way (H, W, 3): luminous band with a warm core, dark dust lanes running along the
    band, and a dense field of tiny unresolved stars concentrated in it.

    params: angle - band tilt in degrees; center - band centre (fractions of W, H); width - band
            half-thickness as a fraction of H; strength - brightness multiplier.
    """
    xs, ys = C.grid(W, H)
    a = math.radians(angle)
    cx, cy = center[0] * W, center[1] * H
    u = (xs - cx) * math.cos(a) + (ys - cy) * math.sin(a)
    vv = -(xs - cx) * math.sin(a) + (ys - cy) * math.cos(a)
    wv = width * H
    wob = (_rotated_fbm(W, H, angle, 6, 3, 3, seed + 1) - 0.5) * wv * 0.9
    vb = (vv + wob) / wv
    band = np.exp(-vb * vb * 1.6)
    core = np.exp(-((u + 0.15 * W) / (0.4 * W)) ** 2)
    cloud = _rotated_fbm(W, H, angle, 4, 10, 6, seed + 2)
    lanes = _rotated_fbm(W, H, angle, 8, 7, 5, seed + 3)
    lane_center = np.exp(-((vb + 0.15) / 0.35) ** 2)
    dark = 1 - np.clip(C.smoothstep(0.45, 0.75, lanes) * (0.35 + 0.6 * lane_center), 0, 0.85)
    lum = band * (0.25 + 0.75 * cloud ** 1.6) * (0.45 + 0.55 * core) * dark
    hue = _rotated_fbm(W, H, angle, 3, 4, 3, seed + 4)
    col = np.stack([0.7 + 0.4 * core + 0.25 * hue, 0.66 + 0.14 * core + 0.1 * (1 - hue), 0.98 - 0.2 * core], -1)
    img = lum[..., None] * col * 0.7 * strength
    # dense tiny stars, concentrated in the band
    rng = np.random.default_rng(seed + 9)
    n = int(26000 * W * H / (1920 * 1080))
    sx = rng.random(n) * W
    sy = rng.random(n) * H
    ix = np.clip(sx.astype(int), 0, W - 1)
    iy = np.clip(sy.astype(int), 0, H - 1)
    keep = rng.random(n) < (0.15 + 0.85 * band[iy, ix] * dark[iy, ix])
    pts = np.zeros((H, W), np.float32)
    np.add.at(pts, (iy[keep], ix[keep]), (rng.random(keep.sum()) ** 2).astype(np.float32) * 0.9)
    pts = C.blur(pts, 0.45 * max(W / 1920, 0.5))
    img += pts[..., None] * np.array([0.92, 0.94, 1.0], np.float32) * strength * 1.6
    return img.astype(np.float32)


# =============================================================================== sun / moon

def sun(W, H, x, y, radius=0.012, color=(1.0, 0.93, 0.8), intensity=1.0, glow_size=1.0, disc=True, core=1.0):
    """Additive sun: compact hot disc + tight saturated halo + broad soft warm glow (HDR, core > 1 for bloom).

    params: x, y - pixel position (may be off-frame); radius - disc radius fraction of W;
            color - halo tint (use a saturated gold (1.0, 0.75, 0.4) at sunset: the halo keeps its colour
            instead of washing to white); intensity - multiplier; glow_size - glow radius multiplier;
            disc - draw the hard disc (False when it is behind clouds / you only want the glow);
            core - brightness of the white core.
    returns: (H, W, 3) float32 to ADD to the frame (before bloom). Peak ~1.6, broad glow <= ~0.25.
    """
    col = np.asarray(color, np.float32)
    R = radius * W
    q = 4
    w, h = max(W // q, 2), max(H // q, 2)
    xs, ys = C.grid(w, h)
    d = np.sqrt((xs * q - x) ** 2 + (ys * q - y) ** 2)
    g2 = np.exp(-d / (R * 4 * glow_size))
    g3 = np.exp(-d / (0.2 * W * glow_size))
    low = g2[..., None] * col * 0.3 + g3[..., None] * col * 0.07
    out = cv2.resize(low.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    B = int(R * 10 * glow_size) + 4
    bx0, bx1 = int(max(x - B, 0)), int(min(x + B, W))
    by0, by1 = int(max(y - B, 0)), int(min(y + B, H))
    if bx1 > bx0 and by1 > by0:
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        dd = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        g1 = 1.0 / (1.0 + (dd / (R * 1.1 * glow_size)) ** 3)
        g1 = g1 * C.smoothstep(B, B * 0.6, dd)
        white = np.array([1.0, 0.97, 0.92], np.float32)
        loc = g1[..., None] * C.lerp(col, white, 0.5) * 0.7
        if disc:
            cr = C.smoothstep(R * 1.04, R * 0.9, dd)
            loc = loc + cr[..., None] * np.array([1.5, 1.45, 1.35], np.float32) * core
        out[by0:by1, bx0:bx1] += loc
    return (out * intensity).astype(np.float32)


def moon(W, H, x, y, radius=0.02, phase=0.0, color=(0.92, 0.95, 1.0), intensity=1.0):
    """Additive moon disc with maria texture and soft halo. phase 0 = full, 0.5 = half (lit from right).
    returns (H, W, 3)."""
    R = radius * W
    xs, ys = C.grid(W, H)
    d = np.sqrt((xs - x) ** 2 + (ys - y) ** 2)
    disc = C.smoothstep(R + 1, R - 1, d)
    tex = 0.72 + 0.28 * C.fbm(W, H, W / (R * 0.7), 4, seed=5)
    if phase > 0:
        sh = C.smoothstep(R * (1 - 2 * phase) - 2, R * (1 - 2 * phase) + 2, -(xs - x) + 0 * ys)
        disc = disc * (1 - sh * 0.97)
    halo = 1.0 / (1.0 + (d / (R * 2.2)) ** 2) * 0.35 + np.exp(-d / (0.15 * W)) * 0.1
    out = disc[..., None] * tex[..., None] * np.asarray(color, np.float32) * 1.0 + halo[..., None] * np.asarray(color, np.float32)
    return (out * intensity).astype(np.float32)


# =============================================================================== birds

def bird_color(sky_rgb, darkness=0.62):
    """Silhouette colour for birds / distant silhouettes against a sky colour: a deep indigo sky-shadow
    tone that keeps some of the sky's hue (never black). sky_rgb: (3,) colour sampled near the flock.
    darkness 0..1 (0.5-0.7 reads well)."""
    s = np.asarray(sky_rgb, np.float32)
    base = np.array([0.14, 0.12, 0.3], np.float32)
    return (base * darkness + s * (1 - darkness) * 0.6).astype(np.float32)


def birds(W, H, t, seed=0, n=5, center=(0.3, 0.3), velocity=(0.04, -0.01), size=0.018, spread=0.06, flap=3.2,
          heading=1.0, size_var=0.45):
    """A small flock of bird silhouettes (body + swept wings), temporally coherent.

    Each bird has its own wingbeat phase/rate (wing pose changes every frame: up-stroke, spread glide,
    down-stroke), its own size (size_var), uneven spacing, a lazy positional wander and occasional glides,
    so the flock never moves in lockstep.
    params: t - time (s); center - flock centre at t=0 (fraction of W, H); velocity - flock drift
            (fraction of W per second); size - mean wingspan fraction of W (0.012-0.025 at 1080p);
            spread - flock spread fraction of W; flap - mean wingbeat Hz; heading - +1 right / -1 left;
            size_var - relative size variation (0.45 = 0.55x..1.45x; far birds smaller).
    returns: (H, W) alpha mask - composite with bird_color(sky) (deep indigo, not black).
    """
    rng = np.random.default_rng(seed)
    offs = rng.normal(0, 1, (n, 2)) * spread * W * np.array([1.0, 0.45])
    offs[:, 0] += np.cumsum(rng.uniform(0.3, 1.6, n)) * spread * W * 0.25 * heading - spread * W * 0.2 * n * 0.25
    ph = rng.random(n) * 6.28
    fr = flap * rng.uniform(0.75, 1.3, n)
    sc = 1 + size_var * rng.uniform(-1, 1, n)
    wob = rng.random((n, 3)) * 6.28
    glide_ph = rng.random(n) * 6.28
    ss = 4
    m = np.zeros((H * ss, W * ss), np.uint8)
    for i in range(n):
        cx = (center[0] + velocity[0] * t) * W + offs[i, 0] + math.sin(t * 0.7 + wob[i, 0]) * 0.004 * W
        cy = (center[1] + velocity[1] * t) * H + offs[i, 1] + math.sin(t * 0.9 + wob[i, 1]) * 0.003 * W
        s = size * W * sc[i]
        g = 0.5 + 0.5 * math.sin(t * 0.8 + glide_ph[i])
        amp = 0.25 + 0.75 * C.smoothstep(0.25, 0.6, g)
        f = math.sin(t * fr[i] * 2 * math.pi + ph[i]) * amp
        hd = heading
        tip_y = -0.34 * f
        elb_y = -0.06 - 0.14 * f
        span = 0.5 * (1 - 0.18 * abs(f))            # wings fold a little at the extremes of the beat
        R_up = [(0.04, -0.03), (0.2, elb_y - 0.02), (0.36 * span / 0.5, (elb_y + tip_y) * 0.5 - 0.02), (span, tip_y)]
        R_lo = [(span, tip_y + 0.01), (0.4 * span / 0.5, (elb_y + tip_y) * 0.5 + 0.06), (0.24, elb_y + 0.12), (0.06, 0.08)]
        pts_l = np.array([(0, 0.0)] + [(-0.04, -0.03)] + [(-x, y) for x, y in R_up[1:]] +
                         [(-x, y) for x, y in R_lo[1:]], np.float64)
        pts_r = np.array([(0, 0.0)] + R_up + R_lo[1:], np.float64)
        for pts in (pts_l, pts_r):
            p = ((pts * [s, s] + [cx, cy]) * ss).astype(np.int32)
            cv2.fillPoly(m, [p], 255, cv2.LINE_AA)
        bl, bh = 0.26 * s, 0.065 * s
        cv2.ellipse(m, (int(cx * ss), int((cy + 0.02 * s) * ss)), (max(1, int(bl * 0.5 * ss)), max(1, int(bh * ss))),
                    0, 0, 360, 255, -1, cv2.LINE_AA)
        cv2.circle(m, (int((cx + hd * 0.13 * s) * ss), int(cy * ss)), max(1, int(0.045 * s * ss)), 255, -1,
                   cv2.LINE_AA)
        tail = np.array([(-hd * 0.08, 0.02), (-hd * 0.18, -0.01), (-hd * 0.18, 0.06)], np.float64)
        cv2.fillPoly(m, [((tail * s + [cx, cy]) * ss).astype(np.int32)], 255, cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255.0, (W, H), interpolation=cv2.INTER_AREA)


# =============================================================================== drift / animation

_BILLOW_CACHE = {}


def _billow_fields(h, w, seed, scale):
    key = (h, w, seed, round(scale, 4))
    if key not in _BILLOW_CACHE:
        q = 4
        hq, wq = max(h // q, 8), max(w // q, 8)
        f = [C.fbm(wq, hq, 1.0 / max(scale, 1e-3), 3, seed=seed + 17 * k) * 2 - 1 for k in range(4)]
        f2 = [C.fbm(wq, hq, 0.4 / max(scale, 1e-3), 2, seed=seed + 91 + 13 * k) * 2 - 1 for k in range(4)]
        _BILLOW_CACHE[key] = (np.stack(f, -1).astype(np.float32), np.stack(f2, -1).astype(np.float32), q)
    return _BILLOW_CACHE[key]


_GRID_CACHE = {}


def drift(plate, W, H, t, speed=(0.004, 0.0), cam=(0.0, 0.0), zoom=1.0, depth=1.0, offset=(0.0, 0.0),
          billow=0.0, billow_scale=0.03, billow_rate=0.5, seed=0):
    """Sample a (larger) plate into a W x H frame with slow wind drift + camera parallax + billowing.

    params: plate - (h, w, 3|4) plate, ideally >= frame size (wider by speed*DURATION*W + camera travel);
            t - time (s); speed - drift (fraction of W per second, scalar or (sx, sy); 0.002-0.008
            reads as calm Shinkai clouds, far layers slower); cam - camera pan (px), multiplied by
            depth for parallax; zoom - camera zoom (1 = none; push-ins: 1 -> 1.06 over the shot, scaled
            by depth); depth - parallax factor (sky far = 0.05-0.3, near clouds 0.3-0.6);
            offset - static offset (px);
            billow - amplitude (fraction of W, e.g. 0.001-0.002) of a slow, smooth, time-evolving
            displacement attached to the plate: lobes swell and churn gently instead of sliding as rigid
            sprites. 0 = off (plain affine sample, fastest);
            billow_scale - feature size of the churn (fraction of plate width, ~lobe size);
            billow_rate - evolution speed (rad/s of the internal phase; 0.2-0.5 = calm);
            seed - billow field seed.
    returns: (H, W, C) float32 sampled with bilinear sub-pixel precision (no jitter).
    The plate is centred on the frame at t=0, cam=0. Clouds move toward +x for positive speed.
    """
    if not isinstance(speed, (tuple, list, np.ndarray)):
        speed = (float(speed), 0.0)
    h, w = plate.shape[:2]
    z = 1 + (zoom - 1) * depth
    ox = offset[0] - speed[0] * W * t + cam[0] * depth
    oy = offset[1] - speed[1] * W * t + cam[1] * depth
    cx, cy = w / 2 + ox, h / 2 + oy
    bv = (0, 0, 0, 0) if plate.shape[2] == 4 else (0, 0, 0)
    if not billow:
        M = np.array([[z, 0, W / 2 - cx * z], [0, z, H / 2 - cy * z]], np.float32)
        return cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=bv)
    key = (W, H)
    if key not in _GRID_CACHE:
        _GRID_CACHE[key] = C.grid(W, H)
    X, Y = _GRID_CACHE[key]
    px = (X - W / 2) / z + cx
    py = (Y - H / 2) / z + cy
    f1, f2, q = _billow_fields(h, w, seed, billow_scale)
    a = t * billow_rate
    # two slowly rotating phases per scale -> smooth, non-periodic-looking evolution
    wts = np.array([math.cos(a), math.sin(a), math.cos(a * 0.61 + 1.3), math.sin(a * 0.61 + 1.3)], np.float32)
    wts2 = np.array([math.cos(a * 0.5), math.sin(a * 0.5), math.cos(a * 0.37 + 2.1), math.sin(a * 0.37 + 2.1)],
                    np.float32)
    dxq = (f1[..., 0] * wts[0] + f1[..., 1] * wts[1]) + 0.7 * (f2[..., 0] * wts2[0] + f2[..., 1] * wts2[1])
    dyq = (f1[..., 2] * wts[2] + f1[..., 3] * wts[3]) + 0.7 * (f2[..., 2] * wts2[2] + f2[..., 3] * wts2[3])
    D = np.dstack([dxq, dyq]).astype(np.float32)
    d = cv2.remap(D, (px / q).astype(np.float32), (py / q).astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT)
    A = billow * W
    return cv2.remap(plate, (px + d[..., 0] * A).astype(np.float32), (py + d[..., 1] * A).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=bv)


class CloudDrift:
    """Layered cloud plates with independent wind speeds, parallax depths and billowing.

    layers: list of (rgba_plate, speed[, depth[, billow]]) far -> near. speed = fraction of W per second
            (scalar or (sx, sy)); depth default 0.2 + 0.2*i; billow default 0 (see drift()).
    render(bg, t, cam_x=0, cam_y=0, zoom=1) composites all layers over bg (H, W, 3) and returns it.
    render_layers(W, H, t, ...) returns the list of sampled (H, W, 4) layers (e.g. to build occluders).
    """

    def __init__(self, layers):
        self.layers = []
        for i, L in enumerate(layers):
            dp = L[2] if len(L) > 2 else 0.2 + 0.2 * i
            bl = L[3] if len(L) > 3 else 0.0
            self.layers.append((L[0], L[1], dp, bl, i))

    def render_layers(self, W, H, t, cam_x=0.0, cam_y=0.0, zoom=1.0):
        return [drift(p, W, H, t, sp, (cam_x, cam_y), zoom, dp, billow=bl, seed=31 * i)
                for p, sp, dp, bl, i in self.layers]

    def render(self, bg, t, cam_x=0.0, cam_y=0.0, zoom=1.0):
        H, W = bg.shape[:2]
        out = bg
        for L in self.render_layers(W, H, t, cam_x, cam_y, zoom):
            out = _over_rgba(out, L)
        return out
