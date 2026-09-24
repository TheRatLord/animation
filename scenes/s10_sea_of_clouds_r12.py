"""s10 helpers (round 12): painted sea of broken cumulus tops, towers, foreground fragments and cirrus,
all painted with lib.clouds2 (P.shape driven directly so every knob of the painted look is set here).

Cloud tops (backlit sunrise, sun low ahead):
  * crown = 2-3 flat value masses: hot warm-white core facing the sun -> gold -> peach falling into the
    shadow (firm lit_lo -> lit -> hi steps, a strong `hot` peak), a crisp 2-4 px gold lining only on the
    sun-facing top silhouette. The terminator band is a narrow, desaturated dusk-mauve (no pink outline).
  * body = cool blue-grey shadow with a deeper core shadow low in each mass (valley_dark), lighter
    bounce/sky-lit up-facing steps and reflected light, crisp cast crescents where a top overlaps the one
    behind it; torn bases sink into the deep blue gaps.
  * aerial perspective: the palette slides from CROWN (near) through a hazier mid to FAR (pale, warm, low
    contrast) over the first ~quarter of the frame below the horizon; linings fade with distance.
"""
import math

import numpy as np
import cv2

from lib import clouds2 as K
from lib import core as C

# ----------------------------------------------------------------------------------------- palettes
CROWN = dict(hi=(1.2, 1.13, 1.0), lit='#ffc47a', lit_lo='#f39a74', mid='#b08aae', shade='#5064aa',
             deep='#172360', refl='#9cb0e6', bounce='#a894be', rim=(2.0, 1.72, 1.15), haze='#f4c8ae',
             edge_dark=0.05)
# off the sun axis: pinker, cooler lit faces, same deep shadows
ROSE = dict(hi=(1.1, 1.02, 0.96), lit='#f9bc94', lit_lo='#e39a98', mid='#a086b4', shade='#5064aa',
            deep='#182462', refl='#9aaee4', bounce='#a492c0', rim=(1.7, 1.45, 1.1), haze='#f0c4bc',
            edge_dark=0.05)
# toward the horizon: pale, warm, low contrast
FAR = dict(hi=(1.04, 0.95, 0.84), lit='#fdd8b4', lit_lo='#f5c4ac', mid='#e8bcbc', shade='#c2b8d6',
           deep='#b0aad2', refl='#cac6e2', bounce='#f0c2b4', rim=(1.25, 1.06, 0.86), haze='#fad8b8',
           edge_dark=0.02)
# towers (sunrise, side/back light): warm-white lit face, gold mid, lavender shadow
TOWER = dict(hi=(1.2, 1.14, 1.02), lit='#ffe6c2', lit_lo='#f6b486', mid='#d48ea6', shade='#7a7ac0',
             deep='#3c3e8e', refl='#a0a6de', bounce='#eeb0a0', rim=(2.1, 1.8, 1.2), haze='#f6d2bc',
             edge_dark=0.05)
TOWER_FAR = dict(hi=(1.16, 1.1, 1.0), lit='#ffe2c0', lit_lo='#f5ba92', mid='#d898ae', shade='#8684c6',
                 deep='#4c4e9c', refl='#aaaee2', bounce='#eeb8a8', rim=(2.0, 1.72, 1.2), haze='#f6d6c4',
                 edge_dark=0.04)

TOP_LEVELS = ((0.15, 0.32, 0.9, 1.5, 2.8, 0.05, 0.4), (0.06, 0.12, 0.85, 1.1, 2.4), (0.026, 0.05, 0.6, 1.3, 2.8))
SMALL_LEVELS = ((0.08, 0.16, 0.85, 1.1, 2.0), (0.03, 0.06, 0.5))
TOWER_LEVELS = ((0.15, 0.34, 0.9, 1.5, 2.8, 0.05, 0.4), (0.06, 0.13, 0.9, 1.1, 2.4), (0.026, 0.05, 0.65, 1.2, 2.8))


def _ss(e0, e1, x):
    return K._ss(e0, e1, x)


def _blur_smooth(img, sigma):
    """clouds2._blur, smoothed for wide blurs: the linear upsample leaves a piecewise-linear
    field whose gradients (the form normals of big masses) are blocky, so the painted lit-steps of large
    near tops came out as 5-6 px stair steps; a final small blur removes the facets. Same result otherwise."""
    if sigma <= 0.3 or img.size == 0:
        return img
    if sigma <= 6.0 or min(img.shape[:2]) < 8:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    s = K._resize(img, w, h, cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 3.0 * 0.9)
    s = K._resize(s, W, H, cv2.INTER_LINEAR)
    return cv2.GaussianBlur(s, (0, 0), 0.6 * f)       # removes the piecewise-linear facets


class smooth_forms:
    """Context: paint with the smooth wide blur (restores clouds2._blur afterwards, other shots unaffected)."""

    def __enter__(self):
        self._orig = K._blur
        K._blur = _blur_smooth
        return self

    def __exit__(self, *exc):
        K._blur = self._orig
        return False


def floor_color(dy):
    """Colour of the deep under-floor seen in the gaps, by depth below the horizon (fraction of H):
    pale gold haze at the horizon -> periwinkle -> deep ultramarine near the camera."""
    k = np.clip(dy / 0.5, 0, 1) ** 0.6
    far = np.array(K._c('#eec8b8'), np.float32)
    mid = np.array(K._c('#6e7cbc'), np.float32)
    near = np.array(K._c('#1c2b72'), np.float32)
    a = np.clip(k / 0.35, 0, 1)[..., None]
    b = np.clip((k - 0.35) / 0.65, 0, 1)[..., None]
    return (far * (1 - a) + mid * a) * (1 - b) + near * b


# ----------------------------------------------------------------------------------------- field layout
def layout(W, H, hy, sx, seed=3, dy_min=0.006, dy_max=0.8, x_margin=0.16, gap=0.3, S=0.7):
    """Scatter cumulus tops on the ground plane (screen size ~ distance below the horizon), sorted
    far -> near. Near rows are sparser (deep blue gaps between clouds), far rows merge into a sheet."""
    rng = np.random.default_rng(seed)
    out = []
    dy = dy_min
    k = 0
    while dy < dy_max:
        step = dy * rng.uniform(0.22, 0.36) + 0.002
        cw0 = S * W * dy
        x = -x_margin * W - rng.uniform(0, 1) * cw0
        while x < (1 + x_margin) * W + cw0:
            r = float(np.clip(math.exp(rng.normal(0.0, 0.42)), 0.45, 1.9))
            cw = cw0 * r
            if rng.random() > gap * min(dy / 0.07, 1.0) + 0.2 * min(max(dy - 0.15, 0.0) / 0.2, 1.0):
                ddy = dy + rng.uniform(-0.55, 0.55) * step
                asp = rng.uniform(0.34, 0.58)
                cx = x + cw * 0.5
                prox = math.exp(-((cx - sx) / (0.22 * W + 0.6 * W * ddy)) ** 2)
                if rng.random() < 0.06 and 0.08 < ddy < 0.3 and prox < 0.3:
                    asp *= rng.uniform(1.2, 1.5)          # an occasional heaped turret (never under the sun)
                if ddy < 0.25:
                    asp *= 1.0 - 0.4 * prox
                asp *= 0.72 + 0.28 * min(ddy / 0.1, 1.0)
                asp *= 1.0 - 0.3 * min(max(ddy - 0.15, 0.0) / 0.3, 1.0)
                out.append(dict(cx=cx, by=hy + ddy * H, w=cw, h=cw * asp, dy=ddy, prox=prox, seed=int(k)))
                k += 1
            x += cw * rng.uniform(0.5, 0.95) * (1.0 + 0.25 * min(max(dy - 0.15, 0.0) / 0.2, 1.0))
        dy += step
    out.sort(key=lambda c: c['by'])
    return out


def _pal_for(c):
    """Palette of one top by depth (aerial perspective) and sun proximity."""
    f = float(np.clip((c['dy'] - 0.004) / 0.2, 0, 1))      # 0 horizon .. 1 near (y ~ 0.68 H and closer)
    near = float(np.clip((c['dy'] - 0.15) / 0.35, 0, 1))
    pal = K.mix_palette(ROSE, CROWN, 0.2 + 0.8 * c['prox'])
    pal = K.mix_palette(FAR, pal, float(K._ss(0.0, 1.0, f)) ** 1.1)
    pal = dict(pal)
    # torn bases sink into the valley colour: pale gold haze far away, deep blue near the camera
    kd = float(K._ss(0.25, 1.0, f))
    pal['haze'] = (pal['haze'] * (1 - kd) + np.asarray(K._c('#1d2a6c'), np.float32) * kd).astype(np.float32)
    if near > 0:
        dv = np.asarray(K._c('#3d4f9c'), np.float32)
        pal['shade'] = (pal['shade'] * (1 - 0.45 * near) + dv * 0.45 * near).astype(np.float32)
    return pal, f, near


def top(P, c, x0, y0, u, sun_z):
    """One broken cumulus top: n flat masses (biggest at the back) sharing one ellipsoid."""
    pal, f, near = _pal_for(c)
    cx, by, w, h = c['cx'] - x0, c['by'] - y0, c['w'], c['h']
    rng = np.random.default_rng(1000 + c['seed'])
    n = 2 if w < 40 * u else (3 if w < 260 * u else 4)
    g = (cx, by - 0.45 * h, 0.58 * w, 0.62 * h)
    d = P.sun - np.array([cx, by - 0.5 * h], np.float32)
    dist = float(np.hypot(*d)) + 1e-6
    ang = math.atan2(d[1], d[0]) + math.radians(rng.uniform(-1, 1) * 14.0)
    sun_c = np.array([cx + math.cos(ang) * dist, by - 0.5 * h + math.sin(ang) * dist], np.float32)
    sz = sun_z - 0.18 * near + rng.uniform(-0.1, 0.1)
    offs = sorted([(rng.uniform(-0.3, 0.3), rng.uniform(0.5, 1.0)) for _ in range(n)], key=lambda q: -q[1])
    lv = SMALL_LEVELS if w < 60 * u else TOP_LEVELS
    haze = 0.6 * (1 - f) ** 1.6
    con = 0.35 + 0.65 * f                         # value contrast grows toward the camera
    n2 = float(np.clip((c['dy'] - 0.1) / 0.25, 0, 1))   # near tops: the lit crown is only the top of the dome
    for i, (ox, sc) in enumerate(offs):
        ww = w * (0.45 + 0.4 * sc)
        hh = h * sc
        yb = by + (i / max(n - 1, 1)) * 0.07 * h
        poly = K.envelope(cx + ox * w, yb, ww, hh, rng, power=rng.uniform(2.1, 2.7), lump=0.08,
                          lean=rng.uniform(-0.1, 0.15), skew=rng.uniform(-0.4, 0.4), base_round=0.1, base_wave=0.05)
        P.shape(poly, rng=rng, size=hh, group=g, pal=pal, haze=haze, haze_grad=0.3 + 0.4 * f, sun=sun_c, sun_z=sz,
                form=rng.uniform(0.4, 0.6), split=rng.uniform(0.26, 0.38) + 0.24 * n2, lit_bias=0.17, light=0.8 + 0.2 * c['prox'],
                cast=(0.5 * con if i else 0.0), lost=0.03 + 0.1 * (1 - f), levels=lv,
                term=0.16, term_w=0.05, hot=1.0 * con, lit_step=0.42, lw_base=0.05,
                valley_dark=0.75 * con, valley_span=(0.15, 0.95), shade_top=0.08, refl=0.4, bounce=0.18,
                sky=0.35 * con, rim=0.9, rim_px=2.0 + 1.2 * f, halo=0.0, rim_shadow=0.1,
                inner=(1 if (i == 0 and w > 200 * u) else 0), inner_val=0.45, clean=0.3)
    if w > 30 * u:
        P.wisps(cx - w * 0.55, cx + w * 0.55, by + 0.05 * h, 0.13 * h, rng=rng, pal=pal, seed=c['seed'] + 3,
                amount=0.08 * (1 - f), base_haze=0.55)


def paint_band(W, H, hy, sun, tops, dy0, dy1, pad=0.12, sun_z=-0.02, lining=1.0, seed=0):
    """Paint the tops whose base depth is in [dy0, dy1) on one plate. Returns a plate dict(rgba, ox, oy)."""
    sel = [c for c in tops if dy0 <= c['dy'] < dy1]
    if not sel:
        return None
    u = W / 1920.0
    x0 = -pad * W
    y_top = min(c['by'] - c['h'] * 1.6 for c in sel)
    y_bot = max(c['by'] + 0.2 * c['h'] for c in sel)
    y0 = max(int(y_top - 8 * u), int(-pad * H))
    y1 = min(int(y_bot + 8 * u), int(H * (1 + pad)))
    pw, ph = int((1 + 2 * pad) * W), max(int(y1 - y0), 8)
    sp = (sun[0] - x0, sun[1] - y0)
    P = K.Painter(pw, ph, sun=sp, pal=CROWN, unit=u, seed=seed + 5, sun_z=sun_z)
    with smooth_forms():
        for c in sel:
            top(P, c, x0, y0, u, sun_z)
    if lining:
        fmid = float(np.clip((0.5 * (dy0 + dy1)) / 0.2, 0, 1))
        P.lining(lining, rim_px=2.0 + 1.2 * fmid, halo=0.0, pal=CROWN if fmid > 0.5 else FAR)
    rgba = P.rgba()
    return dict(rgba=rgba, ox=x0, oy=y0, d=None)


# ----------------------------------------------------------------------------------------- towers
def _tower_mass(P, poly, rng, g, pal, levels, haze, sk):
    kw = dict(group=g, pal=pal, levels=levels, form=0.5, split=0.14, lit_bias=0.06, term=0.25, term_w=0.04,
              term_noise=0.07,
              hot=1.0, lit_step=0.25, lw_base=0.1, valley_dark=0.35, shade_top=0.15, refl=0.45, bounce=0.35,
              bounce_group=0.4, sky=0.28, inner=1, rim=1.6, rim_px=3.0, halo=0.3, rim_shadow=0.25, wrap=0.25,
              inner_val=0.55, side_scale=0.75, haze=haze, lost=0.02, scallop=0.8, firm=0.95)
    kw.update(sk)
    P.shape(poly, rng=rng, **kw)


def tower(W, H, sun, cx, base_y, Hc, seed, pal, width=1.0, haze=0.0, levels=TOWER_LEVELS, rim=2.2):
    """Towering cumulus (no anvil) as 3-4 big painted value masses + a few crisp overlapping lobes.
    Returns plate dict(rgba, ox, oy). The silhouette alpha is anti-aliased (no stair steps)."""
    u = W / 1920.0
    top_y = base_y - Hc * 1.32
    x0 = int(cx - 0.8 * Hc)
    y0 = int(top_y)
    pw, ph = int(1.6 * Hc), int(base_y + 0.14 * Hc - top_y)
    sp = (sun[0] - x0, sun[1] - y0)
    P = K.Painter(pw, ph, sun=sp, pal=pal, unit=u, seed=seed, sun_z=0.1)
    rng = np.random.default_rng(seed)
    cx, by = cx - x0, base_y - y0
    wd = width
    g = (cx + 0.02 * Hc * wd, by - 0.5 * Hc, 0.45 * Hc * wd, 0.62 * Hc)
    # (envelope kw, shape kw): crown, upper body, main mass, sunward flank, low shoulders
    L = [
        (dict(cx=cx - 0.02 * Hc, base_y=by - 0.66 * Hc, width=0.34 * Hc * wd, height=0.4 * Hc, power=2.3,
              lump=0.1, lean=0.08), dict(size=0.3 * Hc, cast=0.3)),
        (dict(cx=cx + 0.08 * Hc, base_y=by - 0.4 * Hc, width=0.44 * Hc * wd, height=0.42 * Hc, power=2.4,
              lump=0.09, lean=0.06), dict(size=0.32 * Hc, cast=0.4, inner=1)),
        (dict(cx=cx - 0.06 * Hc, base_y=by - 0.16 * Hc, width=0.56 * Hc * wd, height=0.44 * Hc, power=2.6,
              lump=0.07, lean=0.05, skew=0.2), dict(size=0.34 * Hc, cast=0.45, inner=2)),
        (dict(cx=cx + 0.22 * Hc * wd, base_y=by - 0.08 * Hc, width=0.36 * Hc * wd, height=0.34 * Hc, power=2.3,
              lump=0.1, lean=0.05), dict(size=0.26 * Hc, cast=0.45)),
        (dict(cx=cx - 0.26 * Hc * wd, base_y=by + 0.0 * Hc, width=0.4 * Hc * wd, height=0.2 * Hc, power=2.4,
              lump=0.1, base_round=0.12), dict(size=0.18 * Hc, cast=0.4, haze_grad=0.4)),
        (dict(cx=cx + 0.05 * Hc, base_y=by + 0.02 * Hc, width=0.9 * Hc * wd, height=0.2 * Hc, power=2.2,
              lump=0.08, skew=0.3, base_round=0.1, base_wave=0.05), dict(size=0.2 * Hc, cast=0.4, haze_grad=0.45)),
    ]
    for ek, sk in L:
        poly = K.envelope(rng=rng, **ek)
        with smooth_forms():
            _tower_mass(P, poly, rng, g, pal, levels, haze, sk)
    P.lining(rim, rim_px=3.0, halo=0.5, backlit=0.2)
    P.wisps(cx - 0.6 * Hc * wd, cx + 0.6 * Hc * wd, by + 0.02 * Hc, 0.1 * Hc, rng=rng, pal=pal, seed=seed + 11,
            amount=0.3, erode=1.1, base_haze=0.5)
    P.haze_band(by - 0.3 * Hc, by + 0.02 * Hc, pal['haze'] if isinstance(pal['haze'], np.ndarray) else pal['haze'], 0.5)
    rgba = P.rgba()
    # anti-alias the silhouette: the tiniest florets leave 1-px stair steps once zoomed
    a = rgba[..., 3]
    rgba[..., 3] = np.maximum(cv2.GaussianBlur(a, (0, 0), 0.7 * max(u, 0.5)) * (a < 0.999), a * (a >= 0.999))
    return dict(rgba=rgba, ox=x0, oy=y0, d=None)


# ----------------------------------------------------------------------------------------- foreground fragments
def fg_fragment(W, H, sun_frame, w, h, seed, flip=False):
    """A torn foreground cloud fragment close to the camera: crisp cauliflower top edge catching a gold
    lining, cool shadowed body, torn streaky underside. Returns RGBA (h+, w+) and its anchor offset."""
    u = W / 1920.0
    pw, ph = int(w * 1.35), int(h * 2.1)
    P = K.Painter(pw, ph, sun=sun_frame, pal=CROWN, unit=u, seed=seed, sun_z=-0.15)
    rng = np.random.default_rng(seed)
    pal = dict(K.palette(CROWN))
    pal['shade'] = np.asarray(K._c('#36468e'), np.float32)
    pal['deep'] = np.asarray(K._c('#16205a'), np.float32)
    pal['refl'] = np.asarray(K._c('#6e84c8'), np.float32)
    pal['haze'] = np.asarray(K._c('#1c2766'), np.float32)
    cx, by = pw * 0.5, ph * 0.7
    forms = smooth_forms().__enter__()
    for i, (ox, sc) in enumerate([(-0.18, 1.0), (0.2, 0.8), (0.0, 0.62)]):
        poly = K.envelope(cx + ox * w, by + 0.05 * h * i, w * (0.55 + 0.3 * sc), h * sc, rng, power=2.3, lump=0.08,
                          lean=rng.uniform(-0.1, 0.1), skew=rng.uniform(-0.3, 0.3), base_round=0.1, base_wave=0.06)
        P.shape(poly, rng=rng, size=h * sc, group=(cx, by - 0.45 * h, 0.6 * w, 0.62 * h), pal=pal,
                split=0.3, lit_bias=0.1, cast=0.5 if i else 0.0, term=0.15, term_w=0.04, hot=0.9, lit_step=0.35, valley_dark=0.7,
                shade_top=0.05, refl=0.3, sky=0.3, rim=1.2, rim_px=3.5, halo=0.0, levels=TOP_LEVELS, haze_grad=0.5,
                form=0.45, inner=1)
    forms.__exit__()
    P.wisps(cx - 0.62 * w, cx + 0.62 * w, by + 0.1 * h, 0.22 * h, rng=rng, pal=pal, seed=seed + 5, amount=0.35,
            erode=1.4, streak=4.0, base_haze=0.4)
    P.lining(1.5, rim_px=3.5, halo=0.0)
    rgba = P.rgba()
    if flip:
        rgba = np.ascontiguousarray(rgba[:, ::-1])
    return rgba


# ----------------------------------------------------------------------------------------- cirrus
def cirrus(W, H, sun, seed=12, y1=0.45):
    """High cirrus painted as a few distinct feathery wisps, each a bundle of fine filaments that fan out
    from a denser head and dissolve downwind (combed opacity along each filament, tapered both ends),
    over a very soft veil. Clusters differ in length, width, curvature, filament count and opacity (one
    bold hooked tail, a couple of long faint strands, small tufts). Cool lavender-white high up, gold
    under-light toward the sun. Returns RGBA (int(y1*H), W, 4)."""
    rng = np.random.default_rng(seed)
    h = int(y1 * H)
    ss = 2
    u = W / 1920.0
    acc = np.zeros((h * ss, W * ss), np.float32)
    # (x0, y0, length, angle_deg, bend, width, opacity, n_filaments, spread, hook)  fractions of W / H
    clusters = [(0.04, 0.26, 0.4, -12, -0.05, 0.010, 0.75, 16, 0.007, 0.05),
                (0.36, 0.08, 0.3, 4, 0.03, 0.005, 0.45, 9, 0.006, 0.0),
                (0.6, 0.2, 0.32, -10, -0.04, 0.008, 0.6, 13, 0.01, -0.04),
                (0.17, 0.36, 0.13, -4, 0.01, 0.004, 0.35, 6, 0.004, 0.0),
                (0.8, 0.07, 0.2, 6, -0.02, 0.006, 0.4, 8, 0.008, 0.03),
                (0.47, 0.3, 0.1, -6, 0.0, 0.003, 0.3, 5, 0.003, 0.0),
                (0.72, 0.33, 0.16, -3, 0.015, 0.003, 0.28, 5, 0.004, 0.0)]
    for (x0, y0, L, ang, bend, wd, op, nf, spread, hook) in clusters:
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        for fi in range(nf):
            n = 90
            s0 = rng.uniform(0.0, 0.25)
            lf = L * rng.uniform(0.55, 1.0)
            tt = np.linspace(0, 1, n)
            dist = (s0 + lf * tt) * W
            off = rng.normal(0, 1) * spread * W * (0.35 + 0.9 * tt)            # filaments fan out downwind
            curve = bend * H * np.sin(tt * math.pi) + hook * H * (1 - tt) ** 3 * (1 if fi % 3 else 0.6)
            px = x0 * W + dist * ca - (off + curve) * sa
            py = y0 * H + dist * sa + (off + curve) * ca
            wprof = wd * W * rng.uniform(0.25, 1.0) * np.clip(tt / 0.12, 0, 1) ** 0.5 * (1 - tt) ** 1.2
            comb = 0.55 + 0.45 * np.sin(tt * rng.uniform(6, 14) + rng.uniform(0, 6)) ** 2
            oprof = op * rng.uniform(0.3, 1.0) * np.clip(tt / 0.08, 0, 1) * (1 - tt) ** 0.8 * comb
            for j in range(n - 1):
                if oprof[j] < 0.01:
                    continue
                p0 = (int(px[j] * ss * 4), int(py[j] * ss * 4))
                p1 = (int(px[j + 1] * ss * 4), int(py[j + 1] * ss * 4))
                th = max(int(round(wprof[j] * ss)), 1)
                _seg_max(acc, p0, p1, float(oprof[j]), th)      # max: filaments build density, no seams
    acc = cv2.GaussianBlur(acc, (0, 0), max(1.2 * ss * u, 0.8))
    veil = cv2.GaussianBlur(acc, (0, 0), 14 * ss * u) * 0.5
    a = np.clip(np.maximum(acc, veil), 0, 1)
    a = cv2.resize(a, (W, h), interpolation=cv2.INTER_AREA)
    ys = np.arange(h, dtype=np.float32)[:, None] / H
    xs = np.arange(W, dtype=np.float32)[None, :] / W
    g = np.exp(-((xs - sun[0] / W) / 0.3) ** 2) * K._ss(0.1, 0.4, ys)
    white = np.array([1.0, 0.97, 0.94], np.float32)
    gold = np.array([1.0, 0.82, 0.6], np.float32)
    lav = np.array([0.8, 0.82, 0.98], np.float32)
    hi = K._ss(0.02, 0.28, ys)[..., None]
    col = lav * (1 - hi) + white * hi
    col = col * (1 - 0.7 * g[..., None]) + gold * 0.7 * g[..., None]
    col = np.broadcast_to(col, (h, W, 3))
    return np.concatenate([col, a[..., None]], -1).astype(np.float32)


def _seg_max(acc, p0, p1, val, th):
    """Max-composite one anti-aliased line segment (coordinates in 1/4 px) into acc."""
    x0, y0 = min(p0[0], p1[0]) // 4 - th - 2, min(p0[1], p1[1]) // 4 - th - 2
    x1, y1 = max(p0[0], p1[0]) // 4 + th + 3, max(p0[1], p1[1]) // 4 + th + 3
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, acc.shape[1]), min(y1, acc.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    tmp = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.line(tmp, (p0[0] - x0 * 4, p0[1] - y0 * 4), (p1[0] - x0 * 4, p1[1] - y0 * 4), val, th, cv2.LINE_AA, shift=2)
    np.maximum(acc[y0:y1, x0:x1], tmp, out=acc[y0:y1, x0:x1])


# ----------------------------------------------------------------------------------------- per-frame helpers
from numba import njit, prange  # noqa: E402


@njit(cache=True, fastmath=True, parallel=True)
def _sheen(img, path, cloud_a, near_a, gain, sheen, fgshade):
    """In place: specular glow path under the sun. Lit crowns (warm, bright) get a strong hot sheen,
    shadowed tops in the path a soft warm glaze; stronger on the nearest plane. + foreground darkening."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
            x = min(max((r - 0.62) / 0.35, 0.0), 1.0)
            lit = x * x * (3 - 2 * x)
            # lit crowns: multiplicative sheen (keeps the painted value steps), hottest on the brightest
            # paint; shadowed tops: a soft additive warm glaze
            y = min(max((g - 0.6) / 0.4, 0.0), 1.0)
            k = lit * (0.55 + 0.45 * near_a[i, j]) * gain * (0.5 + 0.5 * y)
            k2 = sheen * cloud_a[i, j] * (1.0 - lit)
            img[i, j, 0] = (r * (1.0 + path[i, j, 0] * k) + path[i, j, 0] * k2) * fgshade[i, 0, 0]
            img[i, j, 1] = (g * (1.0 + path[i, j, 1] * k) + path[i, j, 1] * k2) * fgshade[i, 0, 1]
            img[i, j, 2] = (b * (1.0 + path[i, j, 2] * k) + path[i, j, 2] * k2) * fgshade[i, 0, 2]


def sheen(img, path, cloud_a, near_a, gain, amt, fgshade):
    _sheen(img, path, cloud_a, near_a, np.float32(gain), np.float32(amt), fgshade)
    return img


class Billow:
    """Slow, coherent churning of a plate's cauliflower edges: the plate is resampled through a smooth
    low-frequency displacement field that rotates slowly between two noise fields (no flicker)."""

    def __init__(self, plate, amp_px, cells, seed):
        h, w = plate['rgba'].shape[:2]
        q = 4
        hq, wq = max(h // q, 4), max(w // q, 4)
        self.f = []
        for k in range(4):
            n = C.fbm(wq, hq, cells, 3, seed=seed + k) - 0.5
            n = cv2.resize(n.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
            self.f.append(n * (2.0 * amp_px))
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        self.xs, self.ys = xs, ys
        self.plate = plate

    def at(self, t, rate=0.35):
        c, s = math.cos(rate * t), math.sin(rate * t)
        mx = self.xs + self.f[0] * c + self.f[1] * s
        my = self.ys + self.f[2] * c + self.f[3] * s
        rgba = cv2.remap(self.plate['rgba'], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0, 0))
        return dict(self.plate, rgba=rgba)


@njit(cache=True, fastmath=True, parallel=True)
def _finish(x, paper, vig, exposure, sat, s0, w0, w1):
    """Tone map: soft shoulder from s0 (hue preserving) + highlight desaturation from w0..w1 so HDR hot
    cores / linings read as luminous warm white-gold instead of saturated orange; static paper texture,
    vignette, saturation."""
    H, W = x.shape[0], x.shape[1]
    out = np.empty_like(x)
    k = 1.0 - s0
    for i in prange(H):
        for j in range(W):
            r = max(x[i, j, 0] * exposure, 0.0)
            g = max(x[i, j, 1] * exposure, 0.0)
            b = max(x[i, j, 2] * exposure, 0.0)
            m = max(r, max(g, b))
            mc = m
            if m > s0:
                o = m - s0
                mc = s0 + k * o / (o + k)
            f = mc / max(m, 1e-6)
            t = min(max((m - w0) / (w1 - w0), 0.0), 1.0)
            wd = t * t * (3 - 2 * t) * 0.8
            pp = paper[i, j, 0]
            r = (r * f * (1 - wd) + mc * wd) * pp
            g = (g * f * (1 - wd) + mc * 0.97 * wd) * pp
            b = (b * f * (1 - wd) + mc * 0.86 * wd) * pp
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            v = vig[i, j, 0]
            out[i, j, 0] = min(max((lum + (r - lum) * sat) * v, 0.0), 1.0)
            out[i, j, 1] = min(max((lum + (g - lum) * sat) * v, 0.0), 1.0)
            out[i, j, 2] = min(max((lum + (b - lum) * sat) * v, 0.0), 1.0)
    return out


def finish(x, paper, vig, exposure, sat, s0=0.86, w0=1.7, w1=4.5):
    return _finish(np.ascontiguousarray(x, dtype=np.float32), paper, vig, np.float32(exposure), np.float32(sat),
                   np.float32(s0), np.float32(w0), np.float32(w1))
