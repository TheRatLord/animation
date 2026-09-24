"""clouds2 - painted Shinkai-style clouds (shared module; evolved from technique B, 'cut-paper').

A cloud is painted the way a background artist paints one: back to front, as a handful (3-10) of big
FLAT masses. Each mass is
  * a cauliflower silhouette: a smooth dome envelope whose contour is grown fractally by unions of
    boundary circles - a few big heads, smaller florets riding on the heads, smooth concave gaps between
    them (florets are suppressed in concavities), none on the base. All detail lives on the EDGE;
  * lit from ONE coherent form: every mass of a cloud shares the cloud's ellipsoid normal, blended with
    the mass's own dome (distance-field height, florets smoothed out), dotted with a 3D sun direction
    (screen direction to the sun + `sun_z`). So each tower has one continuous, readable terminator:
    sun-side tops and flanks lit, shadow collecting on the underside and on the side away from the sun;
    overlapping masses add crisp secondary edges (lit-on-lit, shade-on-shade);
  * painted in large flat values: the terminator is a FIRM, SCALLOPED edge (lit cauliflower heads bulge
    into the shadow; specks and holes removed) that only softens in a few passages, with a saturated
    terminator band (pink at sunset); the lit face steps crisply lit_lo (cool mid / pink) -> lit (cream /
    gold) -> hi (near-white peak toward the sun); the shadow is cool (deeper at the top, bluish
    reflected light low down, warm bounce underneath) and the up-facing tops of shadowed masses catch a
    firm step of skylight. No per-lobe sphere shading, no ambient-occlusion dimples;
  * edged with a thin (2-4 px at 1080p) HDR silver/gold rim - a crisp core line with a soft falloff
    inside - on the sun-facing silhouette ONLY where it meets open sky (interior overlap boundaries get
    no rim, no edge accent, no cool band: they stay soft painted edges, never cut-paper outlines), a
    slightly cooler band just inside it and a soft halation spilling into the sky; backlit edges near
    the sun glow;
  * backlit sea of clouds: dusky violet-blue bodies with a thin gold/peach CREST band along each sunward
    top edge (wider and hotter on the sun axis, a dim pink edge far from it) instead of lit caps;
  * composited with a crisp edge all round (value, not blur, makes depth).
Final passes: torn, soft wispy bases that dissolve into the horizon haze, flat-topped anvil caps with a
shaded underside and a wedge tail frayed into wind-sheared fibres, aerial haze, faint paper-scale value
variation. Plates bleed their colours into the transparent margin so bilinear drift sampling never
produces a dark fringe.

All colours are display RGB ~0..1 (rim / hi may exceed 1: HDR for bloom). Plates are straight-alpha
RGBA float32 (h, w, 4). Everything is deterministic for a given seed; layouts are specified in pixels
of the plate (scale all sizes with W so scenes stay resolution independent).

API
===
Palettes / presets
    PRESETS                       'noon', 'magic_hour', 'sunset', 'sunrise'  (+ '<name>_far' distant
                                  variants; 'summer' = 'noon').  keys: hi lit lit_lo mid shade deep refl
                                  bounce rim haze edge_dark
    palette(p)                    name | dict -> dict of float32 arrays
    preset(name, dist=0.0)        preset mixed toward its '_far' variant (dist 0 near .. 1 far)
    mix_palette(a, b, t)
Sun: every plate / Painter takes sun=(x, y) in plate px (may be off-plate) OR sun_dir=(dx, dy) (screen
direction, y down), plus sun_z = elevation of the light toward the viewer (+0.3 = sun a little in front:
faces lit; 0 = side light; < 0 = low sun ahead / backlit: only crests lit).
Plates (one call -> finished RGBA plate)
    cumulonimbus_plate(w, h, cx, base_y, height, sun|sun_dir, preset='noon', seed, sun_z=0.3,
                       anvil=True, anvil_dir=1, anvil_len=0.46, width=1, backlit=0, haze=0, form=0.7,
                       bounce_group=0, haze_band=None, lining=1.0, wrap=0.5, rim_interior=0.35,
                       overshoot=True, term_w=0.1, side=0.7)
    cumulus_plate(w, h, clouds=[(cx, base_y, cw, ch, seed[, dist]), ...], sun|sun_dir, preset, sun_z=0.3,
                  haze=0, n=3, sun_jitter=25, lining=0.8)
    horizon_bank_plate(w, h, y, height, sun|sun_dir, preset, seed, rows=3, haze=0.3, haze_color=None,
                       shrink=0.55, layer_haze=0.22)   (heights vary along each row; every row recedes
                       into the haze behind the next)
    sea_of_clouds_plate(w, h, horizon_y, sun|sun_dir, preset='sunrise', seed, rows=15, splits=None,
                        towers=None, sun_z=-0.35, horizon_haze=None, **cloud_sea kw (near_w, far_w,
                        aspect_near, aspect_far, persp, haze_far, valley, gaps, big, lit, sun_focus,
                        sun_lean, crest, crest_near, fill_near, fg_deep))  -> plate, or list of plates when
                        `splits` is given (parallax); towers take bounce_group (warm sea bounce)
    cirrus_plate(w, h, preset='noon', seed, region=(0.05, 0.4), angle=-8, density=0.5, opacity=0.6)
    haze_plate(w, h, y0, y1, color, amount)  -> RGBA horizon haze wash (put between depth layers)
Painter (fine control: several clouds / layers on one plate)
    P = Painter(w, h, sun=(x, y) | sun_dir=(dx, dy), pal='noon', unit=W/1920, seed, sun_z=0.3)
    P.shape(poly, rng, size, group=(cx, cy, rx, ry), ...)  one flat mass (see its docstring: form, split,
        scallop, term, term_w, term_noise, wrap, merge, hot, sky, lit_step, rim, rim_px, halo, backlit,
        cast, inner, fray, tip_fade, rim_interior, crest, crest_fill, crest_amt, bounce_group, ...)
    P.wisps(...) torn base;  P.fibres(...) sheared anvil fibres;  P.haze_band(...)
    P.lining(strength, rim_px=2.5, halo, backlit)  plate-level silver/gold lining on the outer silhouette
    P.rgba()
    cumulonimbus(P, ...), cumulus(P, ...), bank(P, ...), cloud_sea(P | fn(row) -> P, ...)
Geometry
    envelope(...), anvil_poly(...), billow_strip(...), cauliflower(...)
Drift / compositing helpers
    drift(plate, W, H, t, speed, cam, zoom, depth, billow=0, billow_scale, billow_rate, seed)
        (= lib.sky.drift, empty-safe; billow ~0.001 gives slow, coherent churning of the lobes)
    screen_pos(p, W, H, cam, zoom, depth)  where a plate-space point lands on screen (sun, anchors)
    occluder(*layers)             combined alpha (0..1) of sampled layers (light shafts / sun visibility)
    composite(img, *layers)       straight-alpha 'over' of sampled layers onto an RGB image
Cost at 1080p: ~12 s setup (tower scene) / ~30 s (sea of clouds, 3 plates + 2 towers); per frame the
plates are only sampled: ~1.2 s/frame for the full test scenes incl. billow, flare, bloom and finish.
Robustness: every resize goes through _resize (empty / zero-size crops never reach cv2.resize).
Scene tips: keep the final bloom threshold >= ~1.1 and the shoulder desaturation low (e.g.
F.shoulder(img, 0.92, 0.03)) so the cream lit faces keep their warmth and only the HDR linings bloom;
drift speeds ~0.003 (far) .. 0.012 (near) of W/s with billow 0.0015-0.002 read as visible, calm motion.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from . import core as C
from . import sky as _S
from . import fx as _F

# ----------------------------------------------------------------------------------------- palettes


def _c(h):
    return C.hex2rgb(h) if isinstance(h, str) else np.asarray(h, np.float32)


# lit_lo: lit face far from the sun / near the terminator   lit: main lit face   hi: hottest light toward
# the sun (capped ~0.97 so highlights never clip flat)   mid: terminator band   shade: shadow (top)
# deep: core / valley shadow   refl: reflected light low in the shadow   bounce: warm underside
# rim: silver/gold lining (HDR)   haze: aerial-perspective target colour   edge_dark: pigment edge line
PRESETS = {
    'noon': dict(hi=(0.975, 0.95, 0.9), lit=(0.945, 0.915, 0.86), lit_lo='#dcd5d2', mid='#cfc6c4', shade='#a3b2de',
                 deep='#8497cf', refl='#b8c7eb', bounce='#dccfd0', rim=(1.55, 1.45, 1.15), haze='#c3e1f6',
                 edge_dark=0.05),
    'noon_far': dict(hi=(0.97, 0.965, 0.945), lit='#eff1f2', lit_lo='#e2e8f0', mid='#d5dcec', shade='#b4c4e8',
                     deep='#a2b4e2', refl='#c6d6f2', bounce='#d8dcee', rim=(1.2, 1.18, 1.1), haze='#cdeaf8',
                     edge_dark=0.03),
    'magic_hour': dict(hi=(1.0, 0.95, 0.84), lit='#ffe3b4', lit_lo='#f5c6aa', mid='#e0aab8', shade='#8f93d0',
                       deep='#6a6cb4', refl='#a8addf', bounce='#f0c0a4', rim=(1.34, 1.14, 0.84),
                       haze='#f0d6cc', edge_dark=0.06),
    'magic_hour_far': dict(hi=(0.99, 0.93, 0.85), lit='#f8dcc4', lit_lo='#efc9bc', mid='#dfb8c6',
                           shade='#aeaad6', deep='#9a97cc', refl='#bbb9e2', bounce='#ecc8bc',
                           rim=(1.2, 1.08, 0.9), haze='#f2dcd4', edge_dark=0.03),
    'sunset': dict(hi=(1.0, 0.86, 0.66), lit=(0.97, 0.7, 0.45), lit_lo=(0.93, 0.56, 0.55), mid='#d8789a', shade='#8a7ac6',
                   deep='#3a3478', refl='#9b8ed0', bounce='#e89a8e', rim=(1.34, 1.04, 0.66), haze='#e8a2b6',
                   edge_dark=0.07),
    'sunset_far': dict(hi=(1.0, 0.86, 0.72), lit='#f8b89c', lit_lo='#eea0a8', mid='#dc94b0', shade='#a288c4',
                       deep='#7d68ae', refl='#ad98cc', bounce='#e8a4a4', rim=(1.16, 0.96, 0.76),
                       haze='#eeb0b8', edge_dark=0.03),
    'sunrise': dict(hi=(1.0, 0.87, 0.7), lit=(0.97, 0.69, 0.5), lit_lo=(0.93, 0.56, 0.6), mid='#da7c9e', shade='#8c7ec8',
                    deep='#2e2a6c', refl='#9d92d2', bounce='#f0a292', rim=(1.3, 1.06, 0.74), haze='#eaa8ba',
                    edge_dark=0.07),
    'sunrise_far': dict(hi=(1.0, 0.88, 0.76), lit='#f9bea4', lit_lo='#efa6ac', mid='#df98b4', shade='#a48cc6',
                        deep='#7c66ac', refl='#ae9cce', bounce='#eaa8a6', rim=(1.14, 0.96, 0.8),
                        haze='#f0b4b8', edge_dark=0.03),
}
PRESETS['summer'] = PRESETS['noon']
PRESETS['summer_far'] = PRESETS['noon_far']
PALETTES = PRESETS


def palette(p):
    """Preset name or dict -> dict of float32 RGB arrays (+ float edge_dark). Missing keys are derived."""
    if isinstance(p, str):
        p = PRESETS[p]
    out = {}
    for k, v in p.items():
        out[k] = float(v) if k == 'edge_dark' else np.asarray(_c(v), np.float32)
    out.setdefault('edge_dark', 0.06)
    out.setdefault('lit_lo', out['lit'] * 0.8 + out['mid'] * 0.2)
    out.setdefault('refl', out['shade'] * 0.6 + out['haze'] * 0.4)
    out.setdefault('bounce', out['shade'] * 0.6 + out['lit'] * 0.4)
    return out


def mix_palette(a, b, t):
    a, b = palette(a), palette(b)
    t = float(t)
    return {k: (float(a[k] * (1 - t) + b[k] * t) if k == 'edge_dark' else
                (a[k] * (1 - t) + b[k] * t).astype(np.float32)) for k in a}


def preset(name, dist=0.0):
    """Preset `name` mixed toward its distant variant: dist 0 = near, 1 = '<name>_far'."""
    if not isinstance(name, str):
        return palette(name)
    far = name + '_far'
    if dist <= 0 or far not in PRESETS:
        return palette(name)
    return mix_palette(name, far, min(float(dist), 1.0))


# ----------------------------------------------------------------------------------------- geometry


def envelope(cx, base_y, width, height, rng, lean=0.0, n=72, power=2.4, lump=0.06, base_round=0.07,
             base_wave=0.02, skew=0.0, top_flat=0.0):
    """Dome-shaped mass outline (closed polygon, (N, 2) px).

    width/height - size (px); lean - horizontal shift of the top relative to the base (fraction of
    height, + = right); power - superellipse exponent (2 = ellipse, 3+ = boxier shoulders);
    lump - amplitude of low-frequency irregularity; base_round - how far below base_y the underside
    bulges (fraction of height, small = flat-ish base); base_wave - undulation of the base;
    skew - move the peak sideways (-1..1); top_flat - flatten the top (0..0.9)."""
    t = np.linspace(0.0, math.pi, n)
    c, s = np.cos(t), np.sin(t)
    ex = np.sign(c) * np.abs(c) ** (2.0 / power)
    ey = np.abs(s) ** (2.0 / power)
    if top_flat:
        ey = ey ** (1.0 - 0.8 * top_flat)
    m = np.ones_like(t)
    for k in range(2, 6):
        m += lump / (k - 1) * rng.uniform(-1, 1) * np.sin(k * t + rng.uniform(0, 6.28))
    ex = ex + skew * 0.25 * ey * (1 - ey)
    x = cx + ex * width / 2 * m
    y = base_y - ey * height * m
    x = x + lean * (base_y - y)
    tb = np.linspace(math.pi, 2 * math.pi, max(n // 2, 8))[1:-1]
    cb, sb = np.cos(tb), np.sin(tb)
    bx = cx + np.sign(cb) * np.abs(cb) ** (2.0 / 4.0) * width / 2 * 0.98
    by = base_y + np.abs(sb) ** 0.5 * height * base_round
    ph = rng.uniform(0, 6.28)
    by = by + base_wave * height * np.sin((bx - cx) / max(width, 1) * 9 + ph)
    return np.concatenate([np.stack([x, y], 1), np.stack([bx, by], 1)], 0).astype(np.float32)


def anvil_poly(xc, top_y, left, right, thick, rng, n=120, dome=0.25, rise=0.12, tip=0.04, sag=0.15):
    """Cumulonimbus anvil seen from the side: a flat cap with a nearly flat top (slight dome over the
    tower at x = xc) and a flat, crisp underside; the upwind end is blunt and rounded, the downwind end
    tapers to a thin blade (top droops, underside rises) that is later frayed into fibres.
    left/right - extent (px) upwind / downwind (pass the long side as `left` for a left anvil);
    thick - thickness over the tower (px); dome - bulge over the tower; rise - how much the underside
    rises toward the downwind tip; tip - thickness left at the tip; sag - droop of the top toward the
    tip (all fractions of `thick`)."""
    x = np.concatenate([np.linspace(xc - left, xc, n // 3, endpoint=False), np.linspace(xc, xc + right, n - n // 3)])
    uu = np.where(x < xc, (x - xc) / max(left, 1), (x - xc) / max(right, 1))
    down = uu >= 0 if right >= left else uu <= 0
    a = np.clip(np.abs(uu), 0, 1)
    bulge = thick * dome * np.exp(-(uu / 0.3) ** 2)
    # downwind: thickness tapers (top droops more than the underside rises) -> a long wedge
    taper = (1 - tip) * a ** 1.4
    up_dn = top_y - bulge + thick * taper * (1 - rise / max(rise + sag, 1e-3) * 0.4)
    lo_dn = top_y + thick - thick * taper * (0.4 * rise / max(rise + sag, 1e-3))
    # upwind: blunt rounded end
    r_ = 1 - np.sqrt(np.clip(1 - a ** 2, 0, 1))
    up_up = top_y - bulge + thick * 0.45 * r_
    lo_up = top_y + thick - thick * 0.5 * r_
    up = np.where(down, up_dn, up_up)
    lo = np.where(down, lo_dn, lo_up)
    up = up + thick * 0.03 * np.sin(a * 11 + rng.uniform(0, 6.28))
    lo = np.maximum(lo, up + thick * tip * 0.6)
    upper = np.stack([x, up], 1)
    lower = np.stack([x[::-1], lo[::-1]], 1)
    return np.concatenate([upper, lower], 0).astype(np.float32)


# level = (rmin, rmax, prob[, space_min, space_max[, bulge_min, bulge_max]]), radii fractions of `size`.
# First level = a few big convective heads; later levels = florets that ride on the heads (they are
# suppressed in concave gaps, so the gaps between heads stay smooth).
DEFAULT_LEVELS = ((0.15, 0.34, 0.9, 1.5, 2.8, 0.05, 0.4), (0.06, 0.13, 0.9, 1.1, 2.4),
                  (0.024, 0.05, 0.7, 1.2, 2.8), (0.014, 0.022, 0.18, 1.6, 4.0))
# scallops of the lit/shadow terminator (lit heads bulging into the shadow): big heads + a few florets
TERM_LEVELS = ((0.06, 0.15, 0.9, 1.3, 2.5, 0.2, 0.55), (0.03, 0.055, 0.3, 1.6, 3.4))


def _grow(M, rng, size, levels, pref=(0.0, -1.0), down_cut=0.3, side_scale=0.5, bulge=(0.3, 0.62),
          top_bias=1.0, clump=0.6, sun_dir=None, sun_bias=0.0, concave=0.6, fill=True):
    """Fractal circle-union growth on a uint8 mask M (in place, M at its own pixel scale; `size` in M px).

    For each level (rmin, rmax, prob[, smin, smax[, kmin, kmax]]) walk the outer contours and union
    circles centred just inside them. `pref` is the direction the bumps prefer to face (outward normal;
    (0, -1) = up for a silhouette, 'away from the sun' for a lit/shadow terminator): biggest where the
    contour faces `pref`, shrunk by `side_scale` on perpendicular stretches, skipped where it faces away
    more than `down_cut`. `concave` suppresses florets (levels >= 1) in the concave gaps between big
    heads; `clump` leaves some stretches smooth; sun_dir/sun_bias: more florets on the sun-facing side."""
    px_, py_ = float(pref[0]), float(pref[1])
    for li, lev in enumerate(levels):
        rmin, rmax, prob = lev[:3]
        smin, smax = (lev[3], lev[4]) if len(lev) >= 5 else (0.8, 1.7)
        kmin, kmax = (lev[5], lev[6]) if len(lev) >= 7 else bulge
        Rmax = rmax * size
        if Rmax < 1.0:
            continue
        sig = max(Rmax * 0.5, 1.5)
        g = _blur(M.astype(np.float32), sig)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        gconc = None
        if li >= 1 and concave:
            # convexity at the scale of the big heads: blurred mask > 0.5 at the contour = concave gap
            gconc = _blur(M.astype(np.float32) / 255.0, max(0.07 * size, 1.5))
        cnts, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        circles = []
        for cnt in cnts:
            P = cnt[:, 0, :]
            nP = len(P)
            if nP < 12:
                continue
            ph1, ph2 = rng.uniform(0, 6.28, 2)
            f1, f2 = rng.uniform(1.5, 3.5), rng.uniform(4, 8)
            i = int(rng.integers(0, max(int(Rmax), 1)))
            while i < nP:
                px, py = int(P[i, 0]), int(P[i, 1])
                nx, ny = -gx[py, px], -gy[py, px]
                ln = math.hypot(nx, ny) + 1e-9
                nx, ny = nx / ln, ny / ln
                up = nx * px_ + ny * py_
                if up < -down_cut:
                    i += max(int(Rmax * 0.5), 2)
                    continue
                f = side_scale + (1 - side_scale) * max(min(up * top_bias, 1.0), 0.0)
                r = (rmin + (rmax - rmin) * rng.random() ** 1.7) * size * f
                if up < 0:
                    r *= 1 + up / max(down_cut, 1e-3) * 0.6
                s_ = i / nP * 2 * math.pi
                cl = 1.0 - clump * (0.5 + 0.25 * math.sin(f1 * s_ + ph1) + 0.25 * math.sin(f2 * s_ + ph2))
                pr = prob * cl
                if gconc is not None:
                    cv_ = float(gconc[py, px])
                    t_ = min(max((cv_ - 0.5) / 0.14, 0.0), 1.0)
                    pr *= 1.0 - concave * t_ * t_ * (3 - 2 * t_)
                if sun_dir is not None and sun_bias and (li >= 2 or sun_bias > 0):
                    pr *= 1 + sun_bias * (nx * sun_dir[0] + ny * sun_dir[1])
                if r >= 0.8 and rng.random() < pr:
                    k = rng.uniform(kmin, kmax)
                    circles.append((px - nx * r * k, py - ny * r * k, r))
                i += max(int(r * rng.uniform(smin, smax)), 2)
        for (cx, cy, r) in circles:
            cv2.circle(M, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1, lineType=cv2.LINE_8)
    if fill:
        # fill holes (tiny enclosed pockets read as dark specks once edge-lined)
        cnts, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            cv2.drawContours(M, cnts, -1, 255, thickness=-1)
    return M


def cauliflower(poly, rng, size, levels=DEFAULT_LEVELS, down_cut=0.3, side_scale=0.5, ss=2, bulge=(0.3, 0.62),
                top_bias=1.0, max_px=None, clump=0.6, sun_dir=None, sun_bias=0.0, concave=0.6):
    """Grow a cauliflower silhouette from an envelope polygon (see _grow): a few big heads, florets
    riding on them, smooth concave gaps, none on the base (outline facing down more than `down_cut`).
    returns (mask (h, w) float32 0..1 anti-aliased, holes filled, x0, y0) in the polygon's pixel space."""
    pts = np.asarray(poly, np.float32)
    pad = size * 0.45 + 4
    x0 = int(math.floor(pts[:, 0].min() - pad))
    y0 = int(math.floor(pts[:, 1].min() - pad))
    x1 = int(math.ceil(pts[:, 0].max() + pad))
    y1 = int(math.ceil(pts[:, 1].max() + pad))
    w, h = max(x1 - x0, 2), max(y1 - y0, 2)
    if max_px and max(w, h) * ss > max_px:
        ss = max(1, int(max_px // max(w, h)))
    M = np.zeros((h * ss, w * ss), np.uint8)
    cv2.fillPoly(M, [np.round((pts - [x0, y0]) * ss).astype(np.int32)], 255, lineType=cv2.LINE_8)
    _grow(M, rng, size * ss, levels, (0.0, -1.0), down_cut, side_scale, bulge, top_bias, clump, sun_dir, sun_bias,
          concave)
    m = M.astype(np.float32) / 255.0
    if ss > 1:
        m = _resize(m, w, h, cv2.INTER_AREA)
    return m, x0, y0


# ----------------------------------------------------------------------------------------- helpers


def _shift(m, dx, dy):
    """dst(p) = m(p + (dx, dy)); outside = 0."""
    if m.size == 0:
        return m
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _resize(img, w, h, interp=cv2.INTER_LINEAR):
    """cv2.resize that never sees an empty source/target (guards the '!ssize.empty()' assertion)."""
    w, h = max(int(w), 1), max(int(h), 1)
    if img.size == 0 or img.shape[0] < 1 or img.shape[1] < 1:
        return np.zeros((h, w) + img.shape[2:], np.float32)
    return cv2.resize(img, (w, h), interpolation=interp)


def _blur(img, sigma):
    if sigma <= 0.3 or img.size == 0:
        return img
    if sigma <= 6.0 or min(img.shape[:2]) < 8:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    s = _resize(img, w, h, cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 3.0 * 0.9)
    return _resize(s, W, H, cv2.INTER_LINEAR)


def _shift_blur(m, dx, dy, sigma):
    """blur(shift(m, dx, dy), sigma), computed at reduced resolution when the blur is wide."""
    if sigma < 5.0 or min(m.shape) < 16:
        return _blur(_shift(m, dx, dy), sigma)
    h, w = m.shape
    q = 3 if sigma >= 12.0 else 2
    s = _resize(m, w // q, h // q, cv2.INTER_AREA)
    s = _blur(_shift(s, dx / q, dy / q), sigma / q)
    return _resize(s, w, h, cv2.INTER_LINEAR)


def _clean(M, inside, min_area):
    """In place on a uint8 0/255 mask M: drop lit islands and fill shadow holes (within `inside`) whose
    area is below min_area px."""
    if min_area < 2:
        return M
    for val, sel in ((0, M > 127), (255, (M <= 127) & inside)):
        n, lab, st, _ = cv2.connectedComponentsWithStats(sel.astype(np.uint8), connectivity=8)
        if n <= 1:
            continue
        small = st[:, cv2.CC_STAT_AREA] < min_area
        small[0] = False
        if small.any():
            M[small[lab]] = val
    return M


def _step(x, t, u=1.0):
    """Crisp, anti-aliased step of a smooth field at threshold t (a painted edge, not a gradient)."""
    return _blur((x > t).astype(np.float32), 0.75 * max(u, 0.5))


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _noise(w, h, cells, seed, octaves=3, stretch=1.0):
    """Smooth value noise in 0..1 at size (w, h); `stretch` > 1 elongates horizontally."""
    w, h = max(int(w), 1), max(int(h), 1)
    q = 4
    wq, hq = max(w // q, 4), max(h // q, 4)
    n = C.fbm(wq, max(int(hq * stretch), 4), cells, octaves, seed=seed)
    n = _resize(n.astype(np.float32), w, h, cv2.INTER_CUBIC)
    return n.astype(np.float32)


def _sun_from_dir(w, h, sun_dir):
    """Virtual sun far away along sun_dir (screen direction, y down) from the plate centre."""
    d = np.asarray(sun_dir, np.float32)
    d = d / (float(np.hypot(*d)) + 1e-6)
    return (w / 2 + d[0] * 40 * max(w, h), h / 2 + d[1] * 40 * max(w, h))


# ----------------------------------------------------------------------------------------- painter


@njit(cache=True, fastmath=True, parallel=True)
def _paint_px(prem, oy, ox, v, dk, lw, tm, hotk, tz, hz, e, e2, rimk, glow, a, tx, ck, rf, bk, pal, ed):
    """Colour recipe of one flat mass + premultiplied 'over' into the plate window at (oy, ox).
    pal rows: 0 lit_lo, 1 hi, 2 mid, 3 shade, 4 deep, 5 haze, 6 rim, 7 lit, 8 refl, 9 bounce."""
    H, W = v.shape
    for i in prange(H):
        for j in range(W):
            vv = v[i, j]
            aa = a[i, j]
            if aa <= 0.0:
                continue
            ckk = ck[i, j] * (1.0 - aa)
            if ckk > 0.6:
                ckk = 0.6
            dst_a = prem[oy + i, ox + j, 3]
            l = lw[i, j]
            rr = rimk[i, j]
            for c in range(3):
                sh = pal[3, c] * (1.0 - dk[i, j]) + pal[4, c] * dk[i, j]
                sh = sh * (1.0 - rf[i, j]) + pal[8, c] * rf[i, j]
                sh = sh * (1.0 - bk[i, j]) + pal[9, c] * bk[i, j]
                if l < 0.75:
                    lc = pal[0, c] * (1.0 - l / 0.75) + pal[7, c] * (l / 0.75)
                else:
                    lc = pal[7, c] * (1.0 - (l - 0.75) * 1.2) + pal[1, c] * ((l - 0.75) * 1.2)
                col = sh * (1.0 - vv) + lc * vv
                col = col * (1.0 - tm[i, j]) + pal[2, c] * tm[i, j]
                col = col * (1.0 - hotk[i, j]) + pal[1, c] * hotk[i, j]
                k = tz[i, j]
                col = col * (1.0 - k) + pal[8, c] * k
                col = col * (1.0 - hz[i, j]) + pal[5, c] * hz[i, j]
                cool = 0.7 if c == 0 else (0.85 if c == 1 else 1.0)
                col = col * (1.0 - 0.5 * ed * e2[i, j] * (1.0 - vv)) + 0.5 * ed * e[i, j] * (1.0 - vv) * cool
                r = rr
                col = col * (1.0 - r) + pal[6, c] * r + pal[6, c] * 0.35 * glow[i, j]
                col *= 1.0 + tx[i, j]
                d = prem[oy + i, ox + j, c]
                if ckk > 0.0:
                    d = d * (1.0 - ckk) + pal[3, c] * 0.9 * dst_a * ckk
                prem[oy + i, ox + j, c] = col * aa + d * (1.0 - aa)
            prem[oy + i, ox + j, 3] = aa + dst_a * (1.0 - aa)


class Painter:
    """Back-to-front cloud painter on one RGBA plate.

    w, h  - plate size (px); sun - (x, y) sun position in plate px (may be off-plate), or
    sun_dir - (dx, dy) screen direction toward a distant sun (y down) when no position is given;
    pal - preset name / dict (see PRESETS), per-shape overrides allowed; unit - W/1920 (scales rim
    widths / edge softness); seed - texture seed."""

    def __init__(self, w, h, sun=None, pal='noon', unit=1.0, seed=0, sun_dir=None, sun_z=0.3):
        self.w, self.h = int(w), int(h)
        self.sun_z = float(sun_z)
        if sun is None:
            sun = _sun_from_dir(self.w, self.h, sun_dir if sun_dir is not None else (0.5, -0.8))
        self.sun = np.array(sun, np.float32)
        self.pal = palette(pal)
        self.u = float(unit)
        self.prem = np.zeros((self.h, self.w, 4), np.float32)   # premultiplied RGBA
        self.seed = seed
        tp = self.tp = int(0.3 * max(self.w, self.h))            # texture padding (shapes may overhang)
        TW, TH = self.w + 2 * tp, self.h + 2 * tp
        self.tex = (_noise(TW, TH, 7.0 * TW / self.w, seed + 5, 3) - 0.5).astype(np.float32)
        self.tex2 = (_noise(TW, TH, 30.0 * TW / self.w, seed + 9, 2) - 0.5).astype(np.float32)
        bn = _noise(TW, TH, 16.0 * TW / self.w, seed + 13, 4, stretch=3.5) - 0.5
        M = cv2.getRotationMatrix2D((TW / 2, TH / 2), -12, 1.0)
        self.brush = cv2.warpAffine(bn.astype(np.float32), M, (TW, TH), borderMode=cv2.BORDER_REFLECT)
        self.brush *= 1.0 / (np.abs(self.brush).max() + 1e-6)
        sn = _noise(TW, TH, 14.0 * TW / self.w, seed + 17, 5, stretch=10.0)
        M = cv2.getRotationMatrix2D((TW / 2, TH / 2), -3, 1.0)
        self.streak = cv2.warpAffine(sn.astype(np.float32), M, (TW, TH), borderMode=cv2.BORDER_REFLECT)
        # plate-level proximity to the sun (rims hottest there)
        self.sun_R = 0.45 * math.hypot(self.w, self.h)

    def _tex(self, T, x0, y0, w, h, xs=None, ys=None):
        tp = self.tp
        a, b = y0 + tp, x0 + tp
        if a >= 0 and b >= 0 and a + h <= T.shape[0] and b + w <= T.shape[1]:
            return T[a:a + h, b:b + w]
        return cv2.remap(T, xs + tp, ys + tp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # ------------------------------------------------------------------ one flat shape
    def shape(self, poly, rng=None, size=None, levels=DEFAULT_LEVELS, group=None, pal=None, light=1.0,
              haze=0.0, haze_grad=0.0, sun=None, sun_z=None, form=0.55, mass_r=0.3, smooth=0.055, split=0.3, lit_bias=0.0,
              firm=0.92, soften=0.4, scallop=1.0, term=0.55, hot=0.8, rim=1.0, rim_px=3.0, rim_near=1.2,
              backlit=0.0, lost=0.04, base_dark=0.25, cast=0.0, down_cut=0.3, side_scale=0.68, bulge=(0.3, 0.62),
              top_bias=1.0, sun_bias=-0.15, clump=0.45, concave=0.6, brush=0.05, inner=0, inner_val=0.6,
              inner_size=(0.3, 0.55), refl=0.45, bounce=0.25, shade_top=0.3, valley_dark=0.2,
              valley_span=(0.15, 0.95), lit_cool=0.0, lw_base=0.1, sky=0.18, clean=0.3, lit_step=0.3, rim_shadow=0.35, halo=0.5, fray=None, alpha=1.0,
              mask=None, rim_interior=0.0, crest=0.0, crest_fill=1.0, crest_amt=1.0, bounce_group=0.0, merge=True,
              term_w=0.1, wrap=0.0, wrap_px=0.05, tip_fade=None, term_noise=0.15):
        """Paint one flat cauliflower mass.

        Form: the light/shadow split is driven by a coherent 3D normal field aimed at the sun - a blend
        (`form`) of the whole cloud's ellipsoid normal (group) and this mass's own dome (distance-field
        height, radius `mass_r` * size, florets smoothed out) - so every tower has one continuous,
        readable terminator: sun-side tops and flanks lit, shadow collecting on the underside and on the
        side away from the sun; overlapping masses give crisp secondary edges. The terminator is then
        painted as a firm SCALLOPED edge (lit cauliflower heads bulging into the shadow) that only
        softens in a few noise-driven passages.

        poly - envelope polygon (plate px) (or pass mask=(m, x0, y0) directly);
        size - reference size for the bump radii (default: envelope height);
        group - (cx, cy, R) or (cx, cy, rx, ry): ellipsoid of the whole cloud (all its masses share it);
        sun / sun_z - sun position (plate px) and its elevation toward the viewer (+ = in front of the
                cloud, lights the faces; 0 = side light; - = backlit, only sunward edges lit);
        pal - palette override; light - 0..1 how much sun this mass receives (0 = in the tower's shade);
        smooth - blur (fraction of size) of the mass dome: florets never get their own shading;
        split - lambert value of the terminator (higher = less lit); lit_bias - offset of the lighting;
        firm/soften - share of the terminator painted crisp / how much it softens in places;
        scallop - size multiplier of the terminator scallops (0 = smooth terminator);
        term - saturated terminator band (mid colour) on the lit side of the split;
        sky - skylight on the up-facing tops of shadowed masses (posterised: shade-on-shade edges);
        halo - soft glow of the lining spilling into the sky outside the silhouette;
        rim_shadow - cooler band just inside the sunward contour, under the rim (makes the lining read);
        lit_step - where (0..1 across the lit face) the firm lit_lo -> lit step sits;
        clean - lit specks / shadow holes smaller than (clean * size)^2 are removed;
        hot - near-white luminous peak on the most sun-facing parts;
        inner - number of half-lit secondary lobes painted inside big shadows (crisp overlapping-mass
                edges); inner_val - their value; inner_size - their size range (fraction of size);
        shade_top / refl / bounce - shadow deeper at the top / bluish reflected light low down / warm
                bounce on the underside; valley_dark/valley_span - darkening low in the mass;
        rim/rim_px - silver-lining strength / width (1080p px, 2-4 px); rim_near - extra width/heat near
                the sun; backlit - 0..1 sun behind/near: glowing rim + bloom on every edge near the sun;
        lost - softness of the whole silhouette (keep small: crisp edges everywhere);
        cast - darken what is already painted where this mass shadows it (away from the sun);
        fray - (x_start, x_end, strength): fibrous torn edge over that x range (anvil tails);
        concave - suppress florets in concave gaps (smooth gaps between the big heads);
        rim_interior - 0 = lining / edge accents only where the mass meets open sky (interior overlap
                boundaries stay soft painted edges, no cut-paper outlines); 1 = also over darker paint
                behind (sea-of-clouds crests over the row behind);
        crest - backlit crest mode (px): the lit face is a band of this depth along the sunward contour
                (a pixel is lit when the mass ends within `crest` px toward the sun); crest_fill - how much
                of the ordinary lambert-lit fill is kept on top of the band (0 = crest only: dusky body);
                crest_amt - strength of the crest band (< 1: a dimmer, pinker edge far from the sun);
        bounce_group - warm bounce on the lower part of the whole cloud (light reflected up from a sea
                of clouds / the ground), on top of each mass's own underside bounce;
        merge - the mass's form (dome) is merged with the paint already behind it, so interior overlap
                boundaries get no shading falloff of their own (no dark cell seams inside lit faces);
        term_w - width of the painted mid-value terminator band (fraction of size, capped 80 px @1080p);
        term_noise - low-frequency wander of the terminator (never a ruler-straight light/shadow line);
        wrap - light wrapping round the silhouette: up/sunward-facing lobe tops on the shadow side catch a
                firm band of light (width wrap_px * size);
        fray = (x_start, x_end, strength[, extend_px]) - with extend_px the tail is first smeared downwind
                and then torn into long fibres (sheared anvil tail streaming into cirrus);
        tip_fade = (x0, x1) - alpha fades out from x0 to x1."""
        u = self.u
        pal = self.pal if pal is None else palette(pal)
        light, haze, haze_grad = float(light), float(haze), float(haze_grad)
        base_dark, lost, rim, hot, backlit, cast = (float(base_dark), float(lost), float(rim), float(hot),
                                                    float(backlit), float(cast))
        sun_z = self.sun_z if sun_z is None else float(sun_z)
        if rng is None:
            rng = np.random.default_rng(0)
        # child generators: floret walks consume a resolution-dependent number of draws; they must not
        # advance the caller's stream (keeps the layout identical at every resolution)
        crng = np.random.default_rng(int(rng.integers(1 << 31)))
        irng = np.random.default_rng(int(rng.integers(1 << 31)))
        trng = np.random.default_rng(int(rng.integers(1 << 31)))
        sun = self.sun if sun is None else np.array(sun, np.float32)
        if mask is None:
            poly = np.asarray(poly, np.float32)
            if size is None:
                size = float(poly[:, 1].max() - poly[:, 1].min())
            sp = sun - poly.mean(0)
            sp = sp / (float(np.hypot(*sp)) + 1e-6)
            m, x0, y0 = cauliflower(poly, crng, size, levels, down_cut=down_cut, side_scale=side_scale,
                                    bulge=bulge, top_bias=top_bias, sun_dir=sp, sun_bias=sun_bias,
                                    clump=clump, concave=concave)
        else:
            m, x0, y0 = mask
            if size is None:
                size = m.shape[0]
        size = max(float(size), 2.0)
        if m.size == 0 or m.max() <= 0:
            return None
        pad = int(max(12 * u, size * (0.14 if cast else 0.06)))
        m = cv2.copyMakeBorder(m, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0 -= pad
        y0 -= pad
        fext = float(fray[3]) if (fray is not None and len(fray) > 3) else 0.0
        if fext > 0:
            # wind-sheared tail: smear the tail downwind by `fext` px (thinning, fading) before it is
            # torn into fibres below, so the tip streams out into cirrus strands instead of ending
            dirx = 1.0 if fray[1] >= fray[0] else -1.0
            pe = int(fext) + 2
            m = cv2.copyMakeBorder(m, 0, 0, pe if dirx < 0 else 0, pe if dirx > 0 else 0, cv2.BORDER_CONSTANT,
                                   value=0)
            if dirx < 0:
                x0 -= pe
            xs_ = (np.arange(m.shape[1], dtype=np.float32) + x0)[None, :]
            src = m * _ss(fray[0], fray[1], xs_)
            acc = m.copy()
            nk = 20
            for k in range(1, nk + 1):
                acc = np.maximum(acc, _shift(src, -dirx * fext * k / nk, 0.0) * (1 - k / (nk + 1)) ** 0.6)
            m = acc
        h, w = m.shape
        px0, py0 = max(x0, 0), max(y0, 0)
        px1, py1 = min(x0 + w, self.w), min(y0 + h, self.h)
        if px1 <= px0 or py1 <= py0:
            return None
        mom = cv2.moments(m)
        area = mom['m00'] + 1e-6
        mcx = float(mom['m10'] / area) + x0
        mcy = float(mom['m01'] / area) + y0
        ytop = float(y0 + pad)
        yb = float(y0 + h - pad)
        # crop to the plate (+ a margin so offsets near the plate border stay right)
        mg = int(max(0.35 * size, 16 * u)) + pad
        cx0, cy0 = max(x0, -mg), max(y0, -mg)
        cx1, cy1 = min(x0 + w, self.w + mg), min(y0 + h, self.h + mg)
        if cx1 - cx0 < 4 or cy1 - cy0 < 4:
            return None
        if (cx0, cy0, cx1, cy1) != (x0, y0, x0 + w, y0 + h):
            m = np.ascontiguousarray(m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0])
            x0, y0 = cx0, cy0
            h, w = m.shape
        ys, xs = np.mgrid[y0:y0 + h, x0:x0 + w].astype(np.float32)
        # what is already painted under this mass's window (interior overlaps vs open sky)
        sl = (slice(py0 - y0, py1 - y0), slice(px0 - x0, px1 - x0))
        pa = np.zeros_like(m)
        plum = np.zeros_like(m)
        reg0 = self.prem[py0:py1, px0:px1]
        pa[sl] = np.clip(reg0[..., 3], 0, 1)
        plum[sl] = _ss(0.7, 0.85, reg0[..., :3].max(-1) / np.maximum(reg0[..., 3], 1e-5)) * pa[sl]
        sky_open = 1.0 - _blur(pa, 3.0 * u)            # ~1 where the edge of this mass meets open sky
        if group is None:
            group = (mcx, mcy, size)
        if len(group) == 3:
            gcx, gcy, grx = group
            gry = grx
        else:
            gcx, gcy, grx, gry = group
        gd = sun - np.array([gcx, gcy], np.float32)
        gdl = float(np.hypot(*gd)) + 1e-6
        sx, sy = float(gd[0] / gdl), float(gd[1] / gdl)           # screen direction to the sun (cloud-wide)
        sdl = float(np.hypot(*(sun - np.array([mcx, mcy], np.float32)))) + 1e-6

        # --- form: coherent normal field ------------------------------------------------------
        qx = (xs - gcx) / max(grx, 1.0)
        qy = (ys - gcy) / max(gry, 1.0)
        nz = np.sqrt(np.clip(1.0 - qx * qx - qy * qy, 0.0, 1.0))
        gl = np.sqrt(qx * qx + qy * qy + nz * nz) + 1e-6
        Rm = max(mass_r * size, 2.0)
        # dome of this mass MERGED with the paint already behind it: interior overlap boundaries get no
        # falloff of their own (no darker seam / 'stained glass' cell lines inside lit faces); only the
        # silhouette against open sky turns the form
        mu = (m > 0.5) | ((pa > 0.5) & (_blur(m, 0.08 * size) > 0.02)) if merge else (m > 0.5)
        dist = cv2.distanceTransform(mu.astype(np.uint8), cv2.DIST_L2, 5)
        tt = np.clip(dist / Rm, 0.0, 1.0)
        hg = _blur((Rm * np.sqrt(np.clip(1.0 - (1.0 - tt) ** 2, 0.0, 1.0))).astype(np.float32),
                   max(smooth * size, 1.0))                                       # florets smoothed out
        hx = cv2.Sobel(hg, cv2.CV_32F, 1, 0, ksize=3) * 0.125
        hy = cv2.Sobel(hg, cv2.CV_32F, 0, 1, ksize=3) * 0.125
        ml = np.sqrt(hx * hx + hy * hy + 1.0)
        fg = float(form)
        Nx = fg * qx / gl - (1 - fg) * hx / ml
        Ny = fg * qy / gl - (1 - fg) * hy / ml
        Nz = fg * nz / gl + (1 - fg) / ml
        nl = np.sqrt(Nx * Nx + Ny * Ny + Nz * Nz) + 1e-6
        ll = math.sqrt(sx * sx + sy * sy + sun_z * sun_z) + 1e-6
        lam = (Nx * sx + Ny * sy + Nz * sun_z) / (nl * ll)          # -1..1

        vpos = np.clip((ys - ytop) / max(yb - ytop, 1.0), 0, 1)               # 0 top .. 1 base (mass)
        gv = np.clip((ys - (gcy - gry)) / max(2 * gry, 1.0), 0, 1)            # 0 top .. 1 base (cloud)
        br = self._tex(self.brush, x0, y0, w, h, xs, ys)
        raw = (lam * light + lit_bias + brush * br - base_dark * _ss(0.4, 1.0, vpos) * 0.5).astype(np.float32)
        if term_noise:
            # low-frequency wander of the terminator (never a ruler-straight light/shadow line)
            raw = raw + term_noise * (0.6 * self._tex(self.tex, x0, y0, w, h, xs, ys) +
                                      0.4 * self._tex(self.tex2, x0, y0, w, h, xs, ys))

        # --- the terminator: firm, scalloped (lit heads bulge into the shadow), soft in places -----
        q = 2
        rawU = _resize(raw, w * q, h * q, cv2.INTER_LINEAR)
        mU = _resize(m, w * q, h * q, cv2.INTER_LINEAR) > 0.5
        LM = (((rawU > split) & mU) * 255).astype(np.uint8)
        _clean(LM, mU, (clean * size * q) ** 2)                 # no lit specks / shadow holes
        if scallop and LM.any():
            _grow(LM, trng, size * q * scallop, TERM_LEVELS, pref=(-sx, -sy), down_cut=0.45, side_scale=0.4,
                  top_bias=1.2, clump=0.5, concave=0.5, fill=False)
            _clean(LM, mU, (0.5 * clean * size * q) ** 2)
        vc = _resize(LM.astype(np.float32) / 255.0, w, h, cv2.INTER_AREA)
        # extend the lit value over the anti-aliased silhouette ring (else the AA pixels take the shadow
        # colour and every lit mass gets a dark/blue hairline where it overlaps lit paint)
        kd = 2 * int(math.ceil(max(u, 1.0))) + 1
        vc = cv2.dilate(vc, np.ones((kd, kd), np.uint8)) * (m > 0.0)
        if crest:
            # backlit crest: lit where the mass ends within `crest` px toward the sun (follows the outline)
            Dc = float(crest)
            band = m * (1 - _shift_blur(m, sx * Dc, sy * Dc, max(0.12 * Dc, 0.6 * u)))
            vcr = _ss(0.3, 0.6, band) * float(crest_amt)
            vc = np.maximum(vc * crest_fill, vcr)
        if wrap:
            # light wraps round the silhouette: up/sunward-facing lobe tops on the SHADOW side, where they
            # meet open sky, catch a firm band of light (lit lobes breaking across the shadow side)
            wd_ = np.array([0.45 * sx, -1.0 + 0.3 * min(sy, 0.0)], np.float32)
            wd_ /= float(np.hypot(*wd_))
            Dw = max(wrap_px * size, 2 * u)
            bw_ = m * (1 - _shift_blur(m, wd_[0] * Dw, wd_[1] * Dw, max(0.1 * Dw, 0.7 * u)))
            bw_ = bw_ * _shift(sky_open, wd_[0] * Dw * 0.5, wd_[1] * Dw * 0.5)
            vc = np.maximum(vc, _ss(0.35, 0.65, bw_) * wrap)
        vs = _ss(split - 0.06, split + 0.06, raw)
        tn = self._tex(self.tex, x0, y0, w, h, xs, ys)
        fm = np.clip(firm - soften * _ss(0.1, 0.32, tn), 0, 1)
        v = np.clip(fm * vc + (1 - fm) * np.maximum(vs, vc * 0.6), 0, 1)
        v2 = _ss(-0.7, split, raw)                                            # shadow depth (form)

        # --- secondary half-lit lobes inside the shadow (crisp overlapping-mass edges) ---------
        dk_extra = None
        if inner:
            inner_n = int(inner)
            hv = np.zeros_like(m)
            cs = np.zeros_like(m)
            got = 0
            core_ = cv2.erode((m > 0.99).astype(np.uint8), np.ones((3, 3), np.uint8),
                              iterations=max(int(0.04 * size), 1))
            for _ in range(24):                       # fixed number of draws (layout stable)
                fx_, fy_ = irng.uniform(0.15, 0.85), irng.uniform(0.2, 0.8)
                rs = irng.uniform(*inner_size)
                wf, hf, lean_, pw_ = irng.uniform(1.1, 1.8), irng.uniform(0.6, 0.9), irng.uniform(-0.1, 0.1), \
                    irng.uniform(2.1, 2.7)
                seed_ = int(irng.integers(1 << 30))
                sv_ = irng.uniform(0.75, 1.0)
                if got >= inner_n:
                    continue
                qx_ = int(pad + fx_ * (w - 2 * pad)) if w > 2 * pad else w // 2
                qy_ = int(pad + fy_ * (h - 2 * pad)) if h > 2 * pad else h // 2
                qx_, qy_ = min(max(qx_, 0), w - 1), min(max(qy_, 0), h - 1)
                if not core_[qy_, qx_] or v[qy_, qx_] > 0.3:
                    continue
                # lobe envelope: its lit crest sits at the sample point, body away from the sun
                rr = rs * size
                bx_ = x0 + qx_ - sx * rr * 0.35
                by_ = y0 + qy_ - sy * rr * 0.35 + rr * hf * 0.5
                ip = envelope(bx_, by_, rr * wf, rr * hf, np.random.default_rng(seed_), power=pw_, lump=0.08,
                              lean=lean_, base_round=0.15)
                lm, lx0, ly0 = cauliflower(ip, np.random.default_rng(seed_ + 1), rr * hf,
                                           ((0.1, 0.24, 0.9, 1.3, 2.6, 0.1, 0.45), (0.04, 0.08, 0.6, 1.2, 2.8)),
                                           down_cut=0.2, side_scale=0.5, sun_dir=(sx, sy), sun_bias=0.6,
                                           concave=0.7)
                sub = np.zeros_like(m)
                ax0, ay0 = lx0 - x0, ly0 - y0
                bx0, by0 = max(ax0, 0), max(ay0, 0)
                bx1, by1 = min(ax0 + lm.shape[1], w), min(ay0 + lm.shape[0], h)
                if bx1 <= bx0 or by1 <= by0:
                    continue
                sub[by0:by1, bx0:bx1] = lm[by0 - ay0:by1 - ay0, bx0 - ax0:bx1 - ax0]
                sub = sub * m
                Dl = 0.3 * rr
                lit_s = sub * (1 - _shift_blur(sub, sx * Dl, sy * Dl, max(0.03 * rr, 0.8 * u)))
                lit_s = _ss(0.35, 0.6, lit_s) * (1 - _ss(0.25, 0.5, v))
                hv = np.maximum(hv, lit_s * sv_)
                # the lobe darkens the mass just behind it (away from the sun): overlap shadow
                cd = 0.06 * rr
                cs = np.maximum(cs, _shift_blur(sub, -sx * cd, -sy * cd, 1.2 * u) * (1 - sub) * m)
                got += 1
            v = np.maximum(v, hv * inner_val)
            dk_extra = cs * 0.35

        dsun = np.hypot(xs - sun[0], ys - sun[1])
        nearP = np.exp(-dsun / self.sun_R)                      # plate-scale proximity to the sun
        near = np.exp(-dsun / max(size, 1) / 0.6)               # shape-scale proximity
        lit, mid, shade, deep, hi = pal['lit'], pal['mid'], pal['shade'], pal['deep'], pal['hi']
        # shadow: deeper/cooler high up and where the form turns fully away; reflected light below
        dk = (1 - v2) * 0.35 + shade_top * (1 - _ss(0.15, 0.75, gv)) + \
            _ss(valley_span[0], valley_span[1], vpos) * valley_dark
        if dk_extra is not None:
            dk = dk + dk_extra
        dk = np.clip(dk, 0, 1)
        rf = refl * (0.55 * _ss(0.4, 1.0, vpos) + 0.45 * _ss(0.35, 1.0, gv))
        if sky:
            # skylight: up-facing tops of the masses in shadow catch the blue sky -> a firm lighter step
            upf = hy / ml
            skm = _blur((upf + 0.1 * tn > 0.22).astype(np.float32), 2.5 * u)
            rf = np.maximum(rf, sky * skm)
        rf = np.clip(rf * (1 - v), 0, 1)
        bk = np.clip((bounce * _ss(0.72, 1.0, vpos) + bounce_group * _ss(0.55, 1.0, gv)) * (1 - v) * m, 0, 1)
        # lit face 3-stop ramp: lit_lo (turning away / near the terminator) -> lit -> hi (facing the sun)
        # half smooth, half a firm step (lit-on-lit: a brighter sun-facing mass with a crisp edge)
        lstep = split + 0.5 * (1 - split) * lit_step
        lw = np.clip(lw_base + (1 - lw_base) * (0.25 * _ss(split, min(split + 0.55, 0.98), raw) +
                                                 0.75 * (fm * _step(raw, lstep, u) +
                                                         (1 - fm) * _ss(lstep - 0.1, lstep + 0.1, raw))), 0, 1)
        lw = np.clip(lw * (1 - lit_cool), 0, 1)
        if rim and rim_shadow:
            # a slightly cooler band just inside the sunward contour, under the rim -> the lining pops
            eb = np.clip(m - _blur(_shift(m, sx * 8 * u, sy * 8 * u), 3.5 * u), 0, 1)
            eb = eb * _shift(sky_open, sx * 8 * u, sy * 8 * u)
            lw = lw * (1 - rim_shadow * eb)
            hotk_cut = eb
        else:
            hotk_cut = None
        # terminator band: lit pixels whose neighbour away from the sun is in shadow (inside the mass)
        # a painted mid-value band 30-80 px wide (1080p) with a firm inner edge that softens in places
        Dt = float(np.clip(term_w * size, 1.5 * u, 80 * u))
        tb0 = np.clip(vc - _shift(vc, -sx * Dt, -sy * Dt), 0, 1) * _shift(m, -sx * Dt, -sy * Dt)
        tb = fm * _blur(tb0, 1.2 * u) + (1 - fm) * _blur(tb0, max(0.3 * Dt, 0.7))
        # the band is strongest right at the terminator and steps down to half across its width
        tb = tb * (0.55 + 0.45 * _blur(np.clip(vc - _shift(vc, -sx * Dt * 0.45, -sy * Dt * 0.45), 0, 1), 1.2 * u))
        tm = np.clip(term * tb + np.exp(-((v - 0.45) / 0.2) ** 2) * 0.5 * (1 - fm), 0, 0.9)
        if hotk_cut is not None:
            # the cool band under the lining takes a touch of the terminator colour (lining reads as light)
            tm = np.maximum(tm, 0.35 * rim_shadow * hotk_cut * v)
        hotk = np.clip(hot * light * (0.4 * _ss(0.6, 0.85, raw) + 0.6 * fm * _step(raw, 0.66, u))
                       * v * (0.5 + 0.5 * nearP), 0, 1)
        if hotk_cut is not None:
            hotk = hotk * (1 - hotk_cut)
        tz = np.zeros_like(m)
        hz = np.clip(haze + haze_grad * _ss(0.2, 1.0, vpos), 0, 1)
        e = np.clip(m - _blur(m, 2.5 * u), 0, 1) * 2.0 * sky_open
        e2 = np.clip(m - _blur(m, 9.0 * u), 0, 1) * sky_open
        # rim (silver lining): thin crisp stroke on the sunward contour; wider/hotter near the sun
        if rim:
            r = rim_px * u * (1 + rim_near * float(np.exp(-sdl / self.sun_R)) ** 2)
            # crisp bright core line + softer falloff inside it (a painted lining, not a stroke)
            core_ = m * (1 - _shift(m, sx * r * 0.6, sy * r * 0.6))
            soft_ = m * (1 - _shift_blur(m, sx * r * 1.7, sy * r * 1.7, 0.5 * r))
            rimm = np.clip(core_ * 1.5 + 0.4 * soft_, 0, 1)
            rw = rim * (0.5 + 0.5 * nearP + 0.8 * rim_near * nearP ** 3) * (0.2 + 0.8 * v + backlit * near)
            if backlit:
                # backlit: every edge near the sun gets a rim (not only the sun-facing direction)
                rb = max(int(round(rim_px * u * 1.2)), 1)
                ring = np.clip(m - cv2.erode(m, np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)), 0, 1)
                rimm = np.maximum(rimm, ring * backlit * near)
            rimk = np.clip(rimm * rw, 0, 1)
        else:
            rimk = np.zeros_like(m)
        if backlit:
            r2 = 12 * u
            glow = m * (1 - _blur(_shift(m, sx * r2, sy * r2), 5 * u)) * backlit * near *                 _shift(sky_open, sx * r2, sy * r2)
        else:
            glow = np.zeros_like(m)

        # --- alpha: crisp all round ---------------------------------------------------------
        ls = lost * 0.02 * size
        a = _blur(m, ls) if ls > 0.6 * u else m
        if fray is not None:
            # fibrous, torn edge (anvil tails): strands of streak noise eat the mass toward the tip
            fx0, fx1, fs = fray[:3]
            fr = _ss(fx0, fx1 + (fext if fx1 >= fx0 else -fext), xs)
            fn = self._tex(self.streak, x0, y0, w, h, xs, ys)
            dst_ = np.clip(_blur(m, 0.05 * size), 0, 1)
            fnn = np.clip((fn - 0.5) * 2.6 + 0.5, 0, 1)            # high-contrast wind streaks
            thr = fr * fs * 0.75
            # the threshold sweeps up slowly toward the tip: only the strongest streaks survive -> the
            # mass tears into long horizontal fibres instead of ending in a cut
            a = a * _ss(thr - 0.12, thr + 0.12, fnn * 0.8 + dst_ * 0.2 + 0.1 * (1 - fr))
            a = a * (1 - 0.25 * fr)
        if tip_fade is not None:
            a = a * (1 - _ss(tip_fade[0], tip_fade[1], xs))
        a = np.clip(a * alpha, 0, 1)

        # --- paste into plate -------------------------------------------------------------
        tp = self.tp
        tx = 0.03 * self.tex[py0 + tp:py1 + tp, px0 + tp:px1 + tp] + \
            0.012 * self.tex2[py0 + tp:py1 + tp, px0 + tp:px1 + tp]
        if cast:
            # crisp crescent on what lies just behind this mass, away from the sun (overlap edge)
            cd = 0.04 * size
            ck = _shift_blur(m, sx * cd, sy * cd, max(0.01 * size, 0.8 * u))[sl] * cast
            ck = ck * (1 - 0.85 * plum[sl])            # never a dark hairline over lit paint
        else:
            ck = np.zeros((py1 - py0, px1 - px0), np.float32)
        palm = np.stack([pal['lit_lo'], hi, mid, shade, deep, pal['haze'], pal['rim'], lit, pal['refl'],
                         pal['bounce']]).astype(np.float32)
        f32 = lambda q_: np.ascontiguousarray(q_[sl], dtype=np.float32)
        if rim:
            # lining only on the silhouette against open sky (rim_interior > 0: also over darker paint);
            # never over lit paint -> no outline around interior masses
            q_any = _shift(pa, sx * 2.5 * r, sy * 2.5 * r)
            q_lit = _shift(plum, sx * 2.5 * r, sy * 2.5 * r)
            rimk = rimk * (1 - np.maximum(0.9 * q_lit, (1 - rim_interior) * q_any))
        rimk = f32(rimk)
        _paint_px(self.prem, py0, px0, f32(v), f32(dk), f32(lw), f32(tm), f32(hotk), f32(tz), f32(hz), f32(e),
                  f32(e2), rimk, f32(glow), f32(a), np.ascontiguousarray(tx, dtype=np.float32),
                  np.ascontiguousarray(ck, dtype=np.float32), f32(rf), f32(bk), palm,
                  np.float32(pal['edge_dark']))
        if rim and halo:
            # halation of the lining into the sky just outside the silhouette (glow wrap), strongest near
            # the sun; only over sky / thin paint so it never haloes interior overlaps
            aa_ = f32(a)
            hr = _blur(rimk * aa_, 2.2 * u) * (1 - aa_) * halo * (0.35 + 0.65 * f32(nearP))
            reg = self.prem[py0:py1, px0:px1]
            hr = np.clip(hr * (1 - reg[..., 3]), 0, 1)[..., None]
            reg[..., :3] += np.asarray(pal['rim'], np.float32) * hr
            reg[..., 3:4] += hr
        return dict(mask=m, x0=x0, y0=y0, cx=mcx, cy=mcy)

    # ------------------------------------------------------------------ fringes / wisps
    def wisps(self, x0, x1, base_y, depth, rng=None, pal=None, amount=0.5, streak=2.5, color_t=0.35,
              erode=0.8, seed=0, base_haze=0.45, haze_color=None):
        """Torn wispy base that fades into the haze: the painted base is tinted toward the haze colour,
        eaten away along a ragged, torn profile (deep tears + fine tatters) and trailed by thin
        alpha-faded horizontal strands.
        x0, x1 - horizontal extent (px); base_y - cloud base (px); depth - vertical extent of the
        fringe (px); amount - strand opacity; streak - horizontal elongation; color_t - strand colour
        between deep (0) and shade (1); erode - how far the base is eaten; base_haze - how strongly the
        lowest part of the cloud fades into haze_color (default: palette haze)."""
        pal = self.pal if pal is None else palette(pal)
        hc = pal['haze'] if haze_color is None else np.asarray(_c(haze_color), np.float32)
        X0 = int(max(x0 - depth, 0))
        X1 = int(min(x1 + depth, self.w))
        Y0 = int(max(base_y - depth * 2.5, 0))
        Y1 = int(min(base_y + depth * 1.6, self.h))
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            return
        w, h = X1 - X0, Y1 - Y0
        ys, xs = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        dy = (ys - base_y) / depth
        region = self.prem[Y0:Y1, X0:X1]
        a0 = region[..., 3].copy()
        # 1) the lowest part of the cloud fades toward the haze colour (aerial perspective at the base)
        hk = (_ss(-2.4, 0.3, dy) * base_haze)[..., None]
        region[..., :3] = region[..., :3] * (1 - hk) + hc * region[..., 3:4] * hk
        # 2) torn base: a ragged cut whose profile varies slowly along x (a few deep tears, lobes of
        #    base hanging lower), with a SOFT dissolve below it - no fine horizontal ripples
        prof1 = _noise(w, 8, max(w / (depth * 4.0), 2.0), seed + 4, 3)[4]
        prof2 = _noise(w, 8, max(w / (depth * 1.3), 3.0), seed + 6, 2)[4]
        tear = _blur(_ss(0.4, 0.9, prof2)[None, :], max(depth * 0.5, 1.0))[0]
        n1 = _noise(w, h, max(w / (depth * 1.6), 2.0), seed + 1, 3, stretch=min(streak, 1.6))
        n2 = _noise(w, h, max(w / (depth * 0.7), 3.0), seed + 2, 2, stretch=2.5)
        cut = dy + erode * (1.3 * (prof1[None, :] - 0.5) + 0.6 * tear[None, :] + 0.7 * (n1 - 0.5)
                            + 0.35 * (n2 - 0.5)) + 0.2 * erode
        # torn edge: firm (painted) with a short dissolve, hanging fragments where the noise dips
        keep = 0.8 * _ss(0.22, -0.22, cut) + 0.2 * _ss(0.9, -0.6, cut)
        region *= keep[..., None]
        # 3) a few thin, soft trailing strands under the base (fractus), coloured toward the haze
        band = (dy > -0.8) & (dy < 0.05)
        prof = (a0 * band).max(0) if band.any() else np.zeros(w, np.float32)
        prof = cv2.GaussianBlur(prof[None, :].astype(np.float32), (0, 0), sigmaX=max(depth * 0.6, 1), sigmaY=0.1)[0]
        n3 = _noise(w, h, max(w / (depth * 2.0), 2.0), seed + 7, 3, stretch=streak)
        st = _ss(0.55, 0.85, n3 * 0.7 + n1 * 0.3)
        fade = _ss(1.2, 0.2, dy) * _ss(-0.3, 0.2, dy)
        wa = np.clip(_blur(prof[None, :] * st * fade * amount, 1.0), 0, 1)
        colw = pal['shade'] * (1 - color_t) + pal['refl'] * color_t
        colw = colw * 0.35 + hc * 0.65
        region[..., :3] = region[..., :3] + colw * wa[..., None] * (1 - region[..., 3:4])
        region[..., 3] = region[..., 3] + wa * (1 - region[..., 3])

    def fibres(self, x0, x1, y_mid, thick, direction=1.0, pal=None, amount=0.7, seed=0, lit=0.7):
        """Wind-sheared fibrous streaks trailing from an anvil tip: thin horizontal strands between
        x0 -> x1 (fading out toward x1) around y_mid, thickness `thick` px, coloured between shadow and
        lit (`lit` 0..1)."""
        pal = self.pal if pal is None else palette(pal)
        xa, xb = min(x0, x1), max(x0, x1)
        X0, X1 = int(max(xa - thick, 0)), int(min(xb + thick, self.w))
        Y0, Y1 = int(max(y_mid - thick * 1.6, 0)), int(min(y_mid + thick * 1.6, self.h))
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            return
        w, h = X1 - X0, Y1 - Y0
        ys, xs = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        s = self._tex(self.streak, X0, Y0, w, h, xs, ys)
        s2 = _noise(w, h, max(w / (thick * 4), 2.0), seed + 3, 3, stretch=12.0)
        along = (xs - x0) / (x1 - x0 + 1e-6)
        along = np.clip(along, 0, 1)
        prof = np.exp(-((ys - y_mid - thick * 0.35 * along) / (thick * (0.5 + 0.4 * along))) ** 2)
        wa = _ss(0.52, 0.7, s * 0.6 + s2 * 0.4) * prof * (1 - _ss(0.35, 1.0, along)) * amount
        wa = np.clip(wa, 0, 1)
        col = pal['lit_lo'] * lit + pal['refl'] * (1 - lit)
        region = self.prem[Y0:Y1, X0:X1]
        region[..., :3] = region[..., :3] + col * wa[..., None] * (1 - region[..., 3:4])
        region[..., 3] = region[..., 3] + wa * (1 - region[..., 3])

    def haze_band(self, y0, y1, color, amount):
        """Blend the painted plate toward `color` between rows y0..y1 (atmospheric haze toward y1)."""
        ys = np.arange(self.h, dtype=np.float32)
        k = (_ss(y0, y1, ys) * amount)[:, None, None]
        a = self.prem[..., 3:4]
        self.prem[..., :3] = self.prem[..., :3] * (1 - k) + np.asarray(_c(color), np.float32) * a * k

    def lining(self, strength=1.0, rim_px=2.5, halo=0.7, backlit=0.0, pal=None, y_max=None):
        """Plate-level silver/gold lining on the OUTER silhouette of everything painted so far: a crisp
        2-4 px (1080p) HDR line where the painted union faces the sun (per-pixel direction to the sun),
        strongest near the sun and on lit paint, with a soft falloff inside and a halation into the sky.
        Interior overlaps never get it (only the union's edge against open sky). backlit (0..1) adds the
        lining on every edge near the sun; y_max - no lining below this row (e.g. under a sea surface)."""
        pal = self.pal if pal is None else palette(pal)
        u = self.u
        A = np.ascontiguousarray(np.clip(self.prem[..., 3], 0, 1))
        if A.max() <= 0:
            return
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        dx, dy = self.sun[0] - xs, self.sun[1] - ys
        dl = np.sqrt(dx * dx + dy * dy) + 1e-3
        ux, uy = dx / dl, dy / dl
        prox = np.exp(-dl / self.sun_R)

        def toward(img, r):
            return cv2.remap(img, xs + ux * r, ys + uy * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=0)
        r = rim_px * u
        core = A * (1 - toward(A, 0.9 * r))
        soft = A * (1 - _blur(toward(A, 2.2 * r), 0.8 * r))
        rgb = self.prem[..., :3] / np.maximum(A, 1e-5)[..., None]
        litn = _ss(0.55, 0.85, rgb.max(-1))
        k = np.clip(core * 1.2 + 0.3 * soft, 0, 1) * (0.12 + 0.88 * litn) * (0.45 + 0.55 * prox)
        if backlit:
            rb = max(int(round(1.2 * r)), 1)
            ring = np.clip(A - cv2.erode(A, np.ones((2 * rb + 1, 2 * rb + 1), np.uint8)), 0, 1)
            k = np.maximum(k, ring * backlit * prox ** 2)
        k = np.clip(k * strength, 0, 1)
        if y_max is not None:
            k = k * (1 - _ss(y_max - 4 * r, y_max, ys))
        rim = np.asarray(pal['rim'], np.float32)
        self.prem[..., :3] = self.prem[..., :3] * (1 - k[..., None]) + rim * (A * k)[..., None]
        if halo:
            hk = _blur(k * A, 2.5 * u) + 0.5 * _blur(k * A, 8 * u)
            hk = np.clip(hk * (1 - A) * halo * (0.3 + 0.7 * prox), 0, 1)[..., None]
            self.prem[..., :3] += rim * hk
            self.prem[..., 3:4] += hk * 0.6

    def rgba(self):
        """Straight-alpha RGBA. Colours are bled a few px outward into the transparent area, so bilinear
        sampling (drift / zoom) never mixes the edge with black (no dark fringe / stroke outline)."""
        a = self.prem[..., 3:4]
        rgb = self.prem[..., :3] / np.maximum(a, 1e-5)
        pb = cv2.GaussianBlur(self.prem, (0, 0), max(2.5 * self.u, 1.0))
        fill = pb[..., :3] / np.maximum(pb[..., 3:4], 1e-5)
        wgt = np.clip(a * 6.0, 0, 1)
        rgb = rgb * wgt + fill * (1 - wgt)
        return np.concatenate([rgb, np.clip(a, 0, 1)], -1).astype(np.float32)


# ----------------------------------------------------------------------------------------- clouds


def cumulonimbus(P, cx, base_y, Hc, seed=0, anvil=True, pal=None, anvil_dir=1.0, sun=None, haze=0.0,
                 backlit=0.0, width=1.0, inner=0, wisps=True, sun_z=None, form=0.7, anvil_len=0.46,
                 bounce_group=0.0, wrap=0.5, rim_interior=0.35, overshoot=True, term_w=0.1, side=0.7):
    """Paint a towering cumulonimbus into Painter P (back to front, ~9 flat masses + anvil).

    All masses share one ellipsoid (the tower) for their light, so the tower has one readable
    terminator; each mass adds its own dome (crisp overlapping-mass edges, lit-on-lit / shade-on-shade).
    cx, base_y - foot centre (px); Hc - height to the top of the anvil (px); anvil_dir +1 = anvil
    spreads right (downwind), -1 left; width - width multiplier; haze - extra aerial haze; backlit - sun
    behind/near (rim + glow on edges near the sun); inner - half-lit lobes per big shadow mass;
    sun_z - light elevation toward the viewer (None = the painter's); form - weight of the tower-wide
    ellipsoid in the light (vs. each mass's own dome): high = one clean terminator; anvil_len - downwind
    reach of the anvil (fraction of Hc; keep it well short of the sun); bounce_group - warm light bounced
    up onto the lower third of the tower (from a sea of clouds below)."""
    rng = np.random.default_rng(seed)
    wd = width
    g = (cx + 0.04 * Hc * wd, base_y - 0.44 * Hc, 0.46 * Hc * wd, 0.56 * Hc)
    top = base_y - Hc
    ad = 1.0 if anvil_dir >= 0 else -1.0
    L = []
    if anvil:
        # wedge anvil: short rounded upwind, long tapering downwind with a torn fibrous tail
        long_, short_ = anvil_len * Hc * wd, 0.26 * Hc * wd
        xa = cx + ad * 0.05 * Hc
        ap = anvil_poly(xa, top + 0.04 * Hc, short_ if ad > 0 else long_, long_ if ad > 0 else short_,
                        0.14 * Hc, rng, dome=0.3, rise=0.3, tip=0.1, sag=0.7)
        # underside: gently undulating, not ruler-straight
        n_up = len(ap) // 2
        xx = ap[n_up:, 0]
        ap[n_up:, 1] += 0.012 * Hc * np.sin((xx - xa) / (0.09 * Hc) + rng.uniform(0, 6.28)) * \
            np.clip(np.abs(xx - xa) / (0.2 * Hc), 0, 1)
        fx0, fx1 = xa + ad * 0.35 * long_, xa + ad * 0.95 * long_
        # smooth, low florets only (an anvil is flat and fibrous, not cauliflower); flat shaded underside
        L.append((ap, dict(levels=((0.05, 0.1, 0.4, 2.5, 5.0, 0.3, 0.55),),
                           firm=0.6, soften=0.5,
                           size=0.2 * Hc, haze=0.04, base_dark=0.3, side_scale=0.3, down_cut=0.1,
                           clump=0.8, top_bias=1.6, inner=0, rim_px=2.2, fray=(fx0, fx1, 1.2, 0.45 * long_), refl=0.55, wrap=0.0,
                           bounce=0.35, shade_top=0.05, form=0.15, mass_r=0.5, split=0.2, scallop=0.5,
                           sun_z=0.3, sun_bias=-0.5, lit_bias=0.05, sky=0.0,
                           group=(xa + ad * 0.3 * long_, top + 0.2 * Hc, 0.75 * long_, 0.35 * Hc))))
        if overshoot:
            # overshooting top: a cauliflower dome punching up through the anvil over the updraft
            L.append((dict(cx=xa - ad * 0.04 * Hc, base_y=top + 0.09 * Hc, width=0.22 * Hc * wd, height=0.12 * Hc,
                           power=2.2, lump=0.12, base_round=0.05),
                      dict(size=0.14 * Hc, cast=0.3, side_scale=0.6, form=0.45, wrap=0.3)))
    L.append((dict(cx=cx + 0.06 * Hc, base_y=base_y - 0.58 * Hc, width=0.38 * Hc * wd, height=0.36 * Hc,
                   power=2.4, lump=0.1, lean=0.1), dict(size=0.32 * Hc, cast=0.3, inner=min(inner, 1))))
    L.append((dict(cx=cx + 0.22 * Hc * wd, base_y=base_y - 0.3 * Hc, width=0.3 * Hc * wd, height=0.44 * Hc,
                   power=2.2, lump=0.1, lean=0.05, base_round=0.2), dict(size=0.26 * Hc, cast=0.3, side_scale=0.7)))
    L.append((dict(cx=cx - 0.03 * Hc, base_y=base_y - 0.25 * Hc, width=0.5 * Hc * wd, height=0.5 * Hc,
                   power=2.7, lump=0.07, lean=0.08, skew=0.2), dict(size=0.36 * Hc, cast=0.4, inner=inner,
                                                                    side_scale=0.65)))
    L.append((dict(cx=cx - 0.25 * Hc * wd, base_y=base_y - 0.13 * Hc, width=0.34 * Hc * wd, height=0.34 * Hc,
                   power=2.4, lump=0.1, lean=0.04), dict(size=0.28 * Hc, cast=0.35, inner=inner)))
    L.append((dict(cx=cx + 0.3 * Hc * wd, base_y=base_y - 0.12 * Hc, width=0.36 * Hc * wd, height=0.3 * Hc,
                   power=2.3, lump=0.1, lean=0.06), dict(size=0.26 * Hc, cast=0.35, inner=min(inner, 1))))
    L.append((dict(cx=cx + 0.03 * Hc, base_y=base_y - 0.02 * Hc, width=0.9 * Hc * wd, height=0.25 * Hc,
                   power=2.2, lump=0.08, skew=0.3, base_round=0.1, base_wave=0.05),
              dict(size=0.24 * Hc, side_scale=0.75, cast=0.4, haze_grad=0.3, light=0.9, haze=0.05, inner=inner)))
    L.append((dict(cx=cx - 0.31 * Hc * wd, base_y=base_y + 0.01 * Hc, width=0.4 * Hc * wd, height=0.15 * Hc,
                   power=2.4, lump=0.1, base_round=0.12, base_wave=0.05),
              dict(size=0.16 * Hc, cast=0.35, haze_grad=0.35, light=0.9, haze=0.08)))
    L.append((dict(cx=cx + 0.37 * Hc * wd, base_y=base_y + 0.015 * Hc, width=0.32 * Hc * wd, height=0.13 * Hc,
                   power=2.3, lump=0.1, base_round=0.12, base_wave=0.05),
              dict(size=0.14 * Hc, cast=0.35, haze_grad=0.35, light=0.9, haze=0.08)))
    info = []
    for ek, sk in L:
        poly = ek if isinstance(ek, np.ndarray) else envelope(rng=rng, **ek)
        sk = dict(sk)
        sk.setdefault('group', g)
        sk.setdefault('pal', pal)
        sk.setdefault('backlit', backlit)
        sk.setdefault('sun_z', sun_z)
        sk.setdefault('form', form)
        sk.setdefault('bounce_group', bounce_group)
        sk.setdefault('wrap', wrap)
        sk.setdefault('term_w', term_w)
        sk['side_scale'] = max(sk.get('side_scale', 0.68), side)
        sk.setdefault('rim_interior', rim_interior)
        sk['haze'] = sk.get('haze', 0.0) + haze
        info.append(P.shape(poly, rng=rng, sun=sun, **sk))
    if anvil:
        # sheared fibres beyond the anvil tip
        tipx = xa + ad * long_
        P.fibres(tipx - ad * 0.45 * long_, tipx + ad * 0.4 * long_, top + 0.1 * Hc,
                 0.035 * Hc, pal=pal, seed=seed + 21, amount=0.55, lit=0.85)
        P.fibres(tipx - ad * 0.3 * long_, tipx + ad * 0.25 * long_, top + 0.15 * Hc,
                 0.02 * Hc, pal=pal, seed=seed + 23, amount=0.35, lit=0.6)
    if wisps:
        P.wisps(cx - 0.6 * Hc * wd, cx + 0.6 * Hc * wd, base_y + 0.02 * Hc, 0.1 * Hc, rng=rng, pal=pal,
                seed=seed + 11, amount=0.4, erode=1.2)
    return info


def cumulus(P, cx, base_y, w, h, seed=0, pal=None, n=3, haze=0.0, sun=None, wisps=True, levels=None,
            lost=0.04, backlit=0.0, sun_jitter=25.0, inner=0, sun_z=None):
    """Fair-weather cumulus: n overlapping flat masses (biggest at the back) sharing one ellipsoid,
    with a per-cloud jitter of the light direction (`sun_jitter` degrees) and of its elevation so
    neighbouring clouds never share one shading template."""
    rng = np.random.default_rng(seed)
    g = (cx, base_y - 0.45 * h, 0.58 * w, 0.62 * h)
    sun0 = P.sun if sun is None else np.asarray(sun, np.float32)
    d = sun0 - np.array([cx, base_y - 0.5 * h], np.float32)
    dist = float(np.hypot(*d)) + 1e-6
    ang = math.atan2(d[1], d[0]) + math.radians(rng.uniform(-1, 1) * sun_jitter)
    sun_c = np.array([cx + math.cos(ang) * dist, base_y - 0.5 * h + math.sin(ang) * dist], np.float32)
    sz = (P.sun_z if sun_z is None else sun_z) + rng.uniform(-0.15, 0.15)
    offs = sorted([(rng.uniform(-0.3, 0.3), rng.uniform(0.5, 1.0)) for _ in range(n)], key=lambda q: -q[1])
    for i, (ox, sc) in enumerate(offs):
        ww = w * (0.45 + 0.4 * sc)
        hh = h * sc
        by = base_y + (i / max(n - 1, 1)) * 0.06 * h
        poly = envelope(cx + ox * w, by, ww, hh, rng, power=rng.uniform(2.1, 2.7), lump=0.08,
                        lean=rng.uniform(-0.1, 0.15), skew=rng.uniform(-0.4, 0.4), base_round=0.1, base_wave=0.05)
        kw = {}
        if levels is not None:
            kw['levels'] = levels
        P.shape(poly, rng=rng, size=hh, group=g, pal=pal, haze=haze, haze_grad=0.25, sun_z=sz,
                form=rng.uniform(0.4, 0.65), split=rng.uniform(0.2, 0.36), cast=0.3 if i else 0.0, sun=sun_c,
                lost=lost, backlit=backlit, inner=inner if i == 0 else 0, **kw)
    if wisps:
        P.wisps(cx - w * 0.55, cx + w * 0.55, base_y + 0.05 * h, 0.13 * h, rng=rng, pal=pal, seed=seed + 3,
                amount=0.14)


def bank(P, y, x0, x1, height, seed=0, pal=None, haze=0.3, rows=2, sun=None, levels=None, lost=0.1,
         shrink=0.55, layer_haze=0.22):
    """Distant cumulus bank along the horizon seen from the side: `rows` sub-rows of flattened,
    overlapping billows of power-law widths, back rows smaller, busier and hazier (aerial perspective).
    y = bank base (px), height = typical billow height (px) of the front row; heights vary along the row
    (a few heaped groups, long low stretches); layer_haze washes each row toward the haze colour before
    the next row is painted in front of it."""
    rng = np.random.default_rng(seed)
    lv = levels or ((0.1, 0.22, 0.9, 1.4, 2.6, 0.1, 0.45), (0.04, 0.08, 0.7, 1.2, 2.6), (0.015, 0.03, 0.4))
    span = max(x1 - x0, 1.0)
    for r in range(rows):
        f = r / max(rows - 1, 1)                  # 0 back .. 1 front
        hr = height * (shrink + (1 - shrink) * f)
        x = x0 - rng.uniform(0, 1) * hr
        # slowly varying height envelope along the row: a few taller heaped groups, long low stretches
        ph = rng.uniform(0, 6.28, 3)
        fq = rng.uniform(1.5, 3.5, 3)
        while x < x1 + hr:
            q = (x - x0) / span
            env = 0.55 + 0.45 * (0.5 * math.sin(fq[0] * 6.28 * q + ph[0]) + 0.3 * math.sin(fq[1] * 9.4 * q + ph[1])
                                 + 0.2 * math.sin(fq[2] * 15.7 * q + ph[2]))
            hh = hr * max(env, 0.2) * (0.35 + 1.25 * rng.random() ** 2.4)
            ww = hh * rng.uniform(1.8, 3.6)
            by = y - hr * 0.3 * (1 - f)
            poly = envelope(x + ww / 2, by, ww, hh, rng, power=rng.uniform(2.0, 2.6),
                            lump=0.1, skew=rng.uniform(-0.6, 0.6), base_round=0.0)
            P.shape(poly, rng=rng, size=hh * 1.1, pal=pal, haze=min(haze * (1.3 - 0.6 * f), 0.95),
                    haze_grad=0.45, levels=lv, lost=lost, cast=0.25, sun=sun, side_scale=0.6, rim_px=1.6,
                    group=(x + ww / 2, by - 0.4 * hh, 0.6 * ww, 0.7 * hh), form=0.5, split=rng.uniform(0.2, 0.4),
                    refl=0.3, bounce=0.0, shade_top=0.1, scallop=0.7)
            x += ww * rng.uniform(0.35, 0.75)
        if r < rows - 1 and layer_haze:
            # each row recedes into the haze behind the next one (layered aerial perspective)
            P.haze_band(y - 1.6 * height, y + 0.1 * height, palette(pal if pal is not None else P.pal)['haze'],
                        layer_haze)


def billow_strip(x0, x1, base_y, bump_w, bump_h, rng, depth=None, n=None, power=2.2, taper=0.25, var=1.0):
    """Polygon of a long strip of merged billows (one row segment of a sea of clouds): a scalloped top
    made of domes of random width/height (`var` scales the size spread), a flat-ish bottom `depth` px
    below base_y (hidden by the next row), ends tapering down over `taper` of the length."""
    L = max(x1 - x0, 1.0)
    depth = bump_h * 0.8 if depth is None else depth
    n = n or max(int(L / max(bump_w, 1) * 24), 48)
    xs = np.linspace(x0, x1, n)
    top = np.full(n, base_y, np.float32)
    x = x0 - bump_w * rng.uniform(0.0, 0.5)
    while x < x1 + bump_w * 0.5:
        bw = bump_w * math.exp(rng.normal(0.0, 0.45 * var))
        bh = bump_h * rng.uniform(0.45, 1.25) * (bw / bump_w) ** 0.6
        d = np.clip(1 - ((xs - x) / (bw / 2)) ** 2, 0, 1)
        top = np.minimum(top, base_y - bh * d ** (1.0 / power))
        x += bw * rng.uniform(0.35, 0.9)
    e = np.minimum((xs - x0) / (L * taper + 1e-6), (x1 - xs) / (L * taper + 1e-6))
    e = np.clip(e, 0, 1)
    e = e * e * (3 - 2 * e)
    mid = base_y + depth * 0.5
    re = np.sqrt(np.clip(1 - (1 - e) ** 2, 0, 1))
    top = mid - (mid - top) * re
    upper = np.stack([xs, top], 1)
    bot = mid + depth * 0.5 * re
    lower = np.stack([xs[::-1], bot[::-1]], 1)
    return np.concatenate([upper, lower], 0).astype(np.float32)


def cloud_sea(P, horizon_y, bottom_y, seed=0, rows=15, pal_near='sunrise', pal_far=None, sun=None,
              x0=0.0, x1=None, persp=2.2, near_w=0.4, far_w=0.016, aspect_near=0.36, aspect_far=0.16,
              haze_far=0.75, row_callback=None, valley=0.6, seg_len=(1.5, 4.5), rim_near=2.0,
              sun_focus=0.25, sun_z=None, gaps=0.15, big=0.25, lit=0.0, sun_lean=0.3, crest=0.05, crest_near=0.2,
              fill_near=0.5, fg_deep='#5a58a6'):
    """Sea of clouds seen from above: `rows` rows from the horizon (small, busy, hazy, merging into the
    glow) to the bottom of the plate (few, big, calm), painted back to front on a perspective curve.
    Each row is a few long overlapping billow strips (merged domes of varied size + florets): one
    undulating surface, not sausages. Lit crests cluster toward the sun (`sun_focus` = width of that
    zone as a fraction of plate width): near the sun lit tops are broader and rims hotter; away from it
    the strips sink into calm violet valleys with thin or no rims. `gaps` = chance a strip is left out
    (reveals the rows behind), `big` = chance a near strip is a large foreground mound; `lit` shifts the
    lighting (+ = more lit faces); sun_lean - how much the light on the sea leans toward the sun's
    screen x (0 = straight up toward the horizon, 1 = toward the sun point).
    Backlit sea (low sun ahead): the billow bodies stay dusky violet-blue and only a crest band along
    each sunward top edge is lit (`crest` = its depth as a fraction of the billow height away from the
    sun, `crest_near` on the sun axis); the lambert-lit fill (orange caps) is kept only near the sun
    axis (`fill_near` = its weight there, 0 = crest only everywhere).

    P may be a Painter or a function row_index -> Painter (to split rows into parallax plates).
    near_w/far_w - billow width (fraction of plate width) at the front / horizon; aspect_* - billow
    height / width; valley - darkness of the lower part of each strip; row_callback(i, y) after each
    row (insert towers)."""
    rng = np.random.default_rng(seed)
    get = P if callable(P) else (lambda i: P)
    P0 = get(0)
    Pw = P0.w
    x1 = Pw if x1 is None else x1
    pal_far = pal_near if pal_far is None else pal_far
    sunp = P0.sun if sun is None else np.asarray(sun, np.float32)
    sz0 = P0.sun_z if sun_z is None else sun_z
    for i in range(rows):
        P = get(i)
        f = i / max(rows - 1, 1)
        z = f ** persp
        y = horizon_y + (bottom_y - horizon_y) * z
        bw_px = (far_w + (near_w - far_w) * z) * Pw
        asp = aspect_far + (aspect_near - aspect_far) * f
        bh = bw_px * asp
        pal = mix_palette(pal_far, pal_near, _ss(0.05, 0.75, f))
        hz = haze_far * (1 - _ss(0.0, 0.75, f)) ** 1.4
        # lobe scale by depth: far rows small and busy, near rows a few big heads with sparse florets
        if f < 0.3:
            lv = ((0.05, 0.12, 0.85, 1.1, 2.0), (0.02, 0.04, 0.7))
        elif f < 0.65:
            lv = ((0.08, 0.2, 0.85, 1.3, 2.6, 0.1, 0.45), (0.03, 0.07, 0.6, 1.3, 2.8), (0.012, 0.025, 0.3, 1.5, 3.5))
        else:
            lv = ((0.1, 0.26, 0.9, 1.3, 2.6, 0.1, 0.45), (0.035, 0.08, 0.75, 1.2, 2.6), (0.014, 0.028, 0.4, 1.4, 3.2))
        # foreground sinks into a deeper blue-violet with more value range (deep valleys, sky-lit tops)
        fgk = _ss(0.45, 1.0, f)
        if fg_deep and fgk > 0:
            pal = dict(pal)
            dv = np.asarray(_c(fg_deep), np.float32)
            pal['shade'] = (pal['shade'] * (1 - 0.55 * fgk) + dv * 0.55 * fgk).astype(np.float32)
            pal['deep'] = (pal['deep'] * (1 - 0.3 * fgk) + dv * 0.5 * 0.3 * fgk).astype(np.float32)
        x = x0 - bw_px * rng.uniform(0.5, 1.5)
        segs = []
        while x < x1 + bw_px:
            L = bw_px * rng.uniform(*seg_len) * (1 + 0.8 * (1 - f))
            sc = 1.0
            if f > 0.55 and rng.random() < big:
                sc = rng.uniform(1.4, 1.9)             # a large foreground mound breaks the banding
            skip = rng.random() < gaps * f
            segs.append((x, x + L * sc, y + rng.uniform(-0.45, 0.35) * bh, rng.uniform(0.7, 1.3), rng.uniform(0.6, 1.5),
                         sc, skip, rng.uniform(-0.12, 0.12)))
            x += L * sc * rng.uniform(0.5, 0.85)
        rng.shuffle(segs)
        for (sx0, sx1, by, ldv, rmv, sc, skip, jz) in segs:
            poly = billow_strip(sx0, sx1, by, bw_px * sc ** 0.7, bh * sc, rng, depth=bh * 1.2 * sc,
                                var=0.6 + 0.8 * f)
            if skip or poly[:, 1].min() - bh * 0.4 > P.h:
                continue
            cxs = 0.5 * (sx0 + sx1)
            prox = math.exp(-((cxs - float(sunp[0])) / (sun_focus * Pw)) ** 2)
            # rim / crest only on the sun-facing tops: fade away from the sun and toward the foreground
            proxw = math.exp(-((cxs - float(sunp[0])) / (1.8 * sun_focus * Pw)) ** 2)
            fgf = 1.0 - 0.85 * _ss(0.45, 1.0, f)
            # light from a low sun far ahead: on the sea surface it rakes mostly 'up' (toward the horizon),
            # only a little toward the sun's screen x -> lit crests, not lit vertical flanks
            ux = sun_lean * float(np.clip((float(sunp[0]) - cxs) / (0.5 * Pw), -1, 1))
            sun_s = (cxs + ux * 1e5, by - 1e5)
            P.shape(poly, rng=rng, size=bh * 1.4 * sc, pal=pal, haze=hz, haze_grad=0.0,
                    group=(cxs, by + bh * 0.2, (sx1 - sx0) * 0.55, bh * 1.6 * sc), form=0.25, mass_r=0.45, smooth=0.03,
                    sun_z=sz0 + 0.5 * jz + 0.25 * prox, split=0.25 + 0.1 * (1 - prox) - lit - 0.1 * ldv * prox, lw_base=0.0,
                    firm=0.95, soften=0.15, scallop=0.8, term=0.2, valley_dark=0.2 + 0.2 * (1 - prox) + 0.25 * fgk,
                    lit_cool=0.55 * (1 - prox), valley_span=(0.2, 0.9), light=0.75 + 0.35 * prox, levels=lv,
                    base_dark=valley * (1 + 0.6 * fgk), lost=0.04 + 0.12 * (1 - f),
                    rim=(proxw ** 1.5 * fgf) if bh > 9 * P.u else 0.0,
                    rim_px=(0.8 + (rim_near - 0.8) * min(f, 0.6)) * rmv * (0.6 + 0.6 * prox), sun=sun_s,
                    side_scale=0.7, down_cut=0.1, cast=0.3, brush=0.06, sun_bias=0.2, refl=0.2, bounce=0.1, shade_top=0.0,
                    inner=0, hot=0.2 + 0.8 * prox, rim_near=1.5, halo=0.6 * prox * fgf, rim_shadow=0.0,
                    rim_interior=0.6 * proxw, sky=0.14 - 0.06 * fgk, term_w=0.05, wrap=0.0, term_noise=0.05,
                    crest=bh * sc * (crest * (0.5 + 0.5 * f) * proxw + (crest_near - crest) * prox) * rmv * ldv
                    if crest else 0.0,
                    crest_fill=fill_near * prox ** 1.5, crest_amt=(0.15 + 0.85 * proxw) * fgf)
        if row_callback is not None:
            row_callback(i, y)


# ----------------------------------------------------------------------------------------- plates


def cumulonimbus_plate(w, h, cx, base_y, height, sun=None, sun_dir=None, preset='noon', seed=0, unit=None,
                       haze_band=None, sun_z=0.3, lining=1.0, **kw):
    """One towering cumulonimbus on its own (w, h) RGBA plate. cx/base_y/height in plate px; sun =
    plate-px position or sun_dir = direction; sun_z = light elevation toward the viewer; kw ->
    cumulonimbus() (anvil, anvil_dir, width, backlit, haze, inner). haze_band = (y0, y1, color,
    amount) optional aerial haze toward the base."""
    u = w / 1920.0 if unit is None else unit
    P = Painter(w, h, sun, preset, u, seed=seed + 1, sun_dir=sun_dir, sun_z=sun_z)
    cumulonimbus(P, cx, base_y, height, seed=seed, **kw)
    if lining:
        P.lining(lining, backlit=kw.get('backlit', 0.0))
    if haze_band:
        P.haze_band(*haze_band)
    return P.rgba()


def cumulus_plate(w, h, clouds, sun=None, sun_dir=None, preset='noon', unit=None, haze=0.0, seed=0, sun_z=0.3,
                  lining=0.8, **kw):
    """Several fair-weather cumulus on one plate. clouds = [(cx, base_y, cw, ch, seed[, dist]), ...] in
    plate px; dist (0..1) mixes the palette toward the '_far' variant and adds haze (aerial perspective)."""
    u = w / 1920.0 if unit is None else unit
    P = Painter(w, h, sun, preset, u, seed=seed + 2, sun_dir=sun_dir, sun_z=sun_z)
    for c in clouds:
        dist = c[5] if len(c) > 5 else 0.0
        cumulus(P, c[0], c[1], c[2], c[3], seed=c[4], pal=globals()['preset'](preset, dist),
                haze=haze + 0.35 * dist, **kw)
    if lining:
        P.lining(lining, backlit=kw.get('backlit', 0.0))
    return P.rgba()


def horizon_bank_plate(w, h, y, height, sun=None, sun_dir=None, preset='noon', seed=0, rows=3, haze=0.3,
                       haze_color=None, unit=None, x0=None, x1=None, sun_z=0.3, shrink=0.55, layer_haze=0.22):
    """Distant horizon cumulus bank (far palette, hazy, small busy back rows) whose bases dissolve into
    a haze band (haze_color, default the palette haze). y = bank base (plate px)."""
    u = w / 1920.0 if unit is None else unit
    pal = globals()['preset'](preset, 1.0)
    P = Painter(w, h, sun, pal, u, seed=seed + 3, sun_dir=sun_dir, sun_z=sun_z)
    bank(P, y, -0.02 * w if x0 is None else x0, 1.02 * w if x1 is None else x1, height, seed=seed, pal=pal,
         haze=haze, rows=rows, shrink=shrink, layer_haze=layer_haze)
    hc = pal['haze'] if haze_color is None else _c(haze_color)
    P.haze_band(y - 0.9 * height, y + 0.1 * height, hc, 0.55)
    return P.rgba()


def sea_of_clouds_plate(w, h, horizon_y, sun=None, sun_dir=None, preset='sunrise', seed=0, rows=15,
                        splits=None, towers=None, unit=None, bottom_y=None, horizon_haze=None, sun_z=-0.35, **kw):
    """Sea of clouds seen from above. horizon_y (plate px); splits = row indices where a new parallax
    plate starts (e.g. (6, 10) -> 3 plates far/mid/near, returned as a list); towers = list of
    dict(row=i, plate=k, cx=, base_off=, height=, seed=, width=, preset=, backlit=, sun_z=, form=, anvil=)
    cumulonimbus rising out of the sea after row i; horizon_haze = (color, amount) wash over the
    farthest rows; sun_z = light elevation (< 0: low sun ahead, only crests lit). kw -> cloud_sea()."""
    u = w / 1920.0 if unit is None else unit
    npl = 1 + (len(splits) if splits else 0)
    Ps = [Painter(w, h, sun, preset, u, seed=seed + 10 + k, sun_dir=sun_dir, sun_z=sun_z) for k in range(npl)]

    def pick(i):
        k = 0
        for s in (splits or ()):
            if i >= s:
                k += 1
        return Ps[k]

    def cb(i, y):
        if horizon_haze and i == max(rows // 4, 1):
            Ps[0].haze_band(y + 0.06 * h, horizon_y - 0.002 * h, horizon_haze[0], horizon_haze[1])
        for tw in towers or ():
            if tw['row'] == i:
                Pt = Ps[tw.get('plate', 0)]
                cumulonimbus(Pt, tw['cx'], y + tw.get('base_off', 0.0), tw['height'], seed=tw.get('seed', 0),
                             anvil=tw.get('anvil', False), pal=tw.get('preset', preset), haze=tw.get('haze', 0.0),
                             width=tw.get('width', 0.85), backlit=tw.get('backlit', 0.0), wisps=False,
                             inner=tw.get('inner', 0), sun_z=tw.get('sun_z', None), form=tw.get('form', 0.8),
                             bounce_group=tw.get('bounce_group', 0.0), wrap=tw.get('wrap', 0.0),
                             term_w=tw.get('term_w', 0.06), side=tw.get('side', 0.85))
                if tw.get('wisps', True):
                    yb = y + tw.get('base_off', 0.0)
                    Pt.wisps(tw['cx'] - 0.5 * tw['height'], tw['cx'] + 0.5 * tw['height'], yb, 0.08 * tw['height'],
                             rng=np.random.default_rng(tw.get('seed', 0) + 5), pal=tw.get('preset', preset),
                             seed=tw.get('seed', 0) + 7, amount=0.3, erode=1.0, base_haze=0.3)
                if tw.get('base_haze'):
                    Pt.haze_band(y + tw.get('base_off', 0.0) - 0.3 * tw['height'], y + tw.get('base_off', 0.0),
                                 tw['base_haze'][0], tw['base_haze'][1])

    far_pal = preset + '_far' if isinstance(preset, str) and preset + '_far' in PRESETS else preset
    cloud_sea(pick, horizon_y, (h + 0.1 * h) if bottom_y is None else bottom_y, seed=seed, rows=rows,
              pal_near=preset, pal_far=far_pal, row_callback=cb, **kw)
    out = [P.rgba() for P in Ps]
    return out if splits else out[0]


def cirrus_plate(w, h, preset='noon', seed=0, region=(0.05, 0.4), angle=-8.0, density=0.5, opacity=0.6,
                 **kw):
    """High cirrus (hooked mare's tails with fibrous trails) tinted by the preset: heads in the lit
    colour, trailing fibres in lit_lo (peach/pink at sunset). Wraps lib.sky.cirrus_plate."""
    p = palette(preset)
    col = np.clip(p['lit'] * 0.7 + p['hi'] * 0.3, 0, 1)
    return _S.cirrus_plate(w, h, seed=seed, color=tuple(col), under=tuple(np.clip(p['lit_lo'], 0, 1)), angle=angle, density=density, region=region,
                           opacity=opacity, **kw)


def haze_plate(w, h, y0, y1, color, amount=0.6, y2=None):
    """Horizontal aerial-haze wash as an RGBA plate: alpha ramps 0 at y0 -> `amount` at y1 (and back to 0
    at y2 if given). Composite it between depth layers so cloud bases dissolve into a pale horizon."""
    ys = np.arange(h, dtype=np.float32)
    a = _ss(y0, y1, ys) * amount
    if y2 is not None:
        a = a * (1 - _ss(y1, y2, ys))
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = _c(color)
    out[..., 3] = a[:, None]
    return out


# ----------------------------------------------------------------------------------------- drift helpers


def drift(plate, W, H, t, speed=(0.003, 0.0), cam=(0.0, 0.0), zoom=1.0, depth=1.0, **kw):
    """Sample a plate into the W x H frame with wind drift + camera parallax (lib.sky.drift); safe on
    empty plates (returns a transparent frame)."""
    if plate is None or plate.size == 0:
        return np.zeros((H, W, 4), np.float32)
    return _S.drift(plate, W, H, t, speed=speed, cam=cam, zoom=zoom, depth=depth, **kw)


def screen_pos(p, W, H, plate_size=None, cam=(0.0, 0.0), zoom=1.0, depth=1.0, t=0.0, speed=(0.0, 0.0)):
    """Screen position of plate-space point p for a plate sampled with drift(..., cam, zoom, depth).
    plate_size = (pw, ph) (default: frame size, i.e. p is already in frame coords)."""
    if not isinstance(speed, (tuple, list, np.ndarray)):
        speed = (float(speed), 0.0)
    pw, ph = plate_size if plate_size is not None else (W, H)
    z = 1 + (zoom - 1) * depth
    cx = pw / 2 - speed[0] * W * t + cam[0] * depth
    cy = ph / 2 - speed[1] * W * t + cam[1] * depth
    return (W / 2 + (p[0] - cx) * z, H / 2 + (p[1] - cy) * z)


def occluder(*layers):
    """Combined alpha (H, W) of sampled RGBA layers."""
    a = None
    for L in layers:
        la = np.clip(L[..., 3], 0, 1)
        a = la if a is None else a + la * (1 - a)
    return np.clip(a, 0.0, 1.0)


def composite(img, *layers):
    """Straight-alpha 'over' of sampled RGBA layers (far -> near) onto an RGB image."""
    for L in layers:
        img = _F.over_rgba(img, L)
    return img
