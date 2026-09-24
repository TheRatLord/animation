"""s01 helper (round 3): the hero cumulonimbus. Geometry (silhouette, masses) painted with lib/clouds2
exactly as in s01_summer_sky_cb3; the VALUES are then repainted as a background artist would:
a few big flat masses with high contrast - brilliant warm-white lit faces, a cool mid step and deep
blue-grey shadow planes - each shadow plane with ONE crisp edge (where a lit lobe in front overlaps it)
and one soft turning edge (toward the light); a crisp 2-4 px silver-gold lining on the sun-facing
silhouette, a lit anvil top with a cool underside and torn wisps; a torn, hazy base."""
import numpy as np
import cv2

from lib import clouds2 as K
from s01_summer_sky_cb3 import PAL, PAL_BASE, PAL_ANVIL, LEV_BIG, top_edge  # noqa: F401


def raw_tower(pw, ph, cx, base_y, Hc, W, seed=12, haze_col='#d4ebf8'):
    """Paint the tower on a (ph, pw) straight-alpha plate; cx / base_y: foot centre, Hc: height to the
    anvil top. Returns (rgba, sun) with the sun seated on the anvil's upper-left crest."""
    u = W / 1920.0
    WF = 0.82                       # width factor: a tall, narrow rising column
    rng = np.random.default_rng(seed)
    by = base_y
    top = by - Hc
    # key light: high up-left, a little in front of the tower (the real sun is behind the crest)
    key = (cx - 0.9 * Hc, by - 1.9 * Hc)
    P = K.Painter(pw, ph, key, PAL, u, seed=seed + 1, sun_z=0.4)
    ids = np.full((ph, pw), -1, np.int16)      # id of the front-most mass at each pixel (overlap edges)
    vmap = np.zeros((ph, pw), np.float32)      # the painter's lit factor of that front-most mass
    ids_n = []
    orig = K._paint_px

    def rec(prem, oy, ox, v, dk, lw, tm, hotk, tz, hz, e, e2, rimk, glow, a, *rest):
        hh, ww = a.shape
        m = a > 0.5
        ids[oy:oy + hh, ox:ox + ww][m] = len(ids_n) - 1
        vmap[oy:oy + hh, ox:ox + ww][m] = v[m]
        return orig(prem, oy, ox, v, dk, lw, tm, hotk, tz, hz, e, e2, rimk, glow, a, *rest)
    K._paint_px = rec
    g = (cx + 0.02 * Hc, by - 0.5 * Hc, 0.42 * Hc, 0.62 * Hc)
    common = dict(group=g, form=0.25, term_w=0.08, wrap=0.0, brush=0.0, lit_step=0.3, term=0.45, hot=1.0,
                  sky=0.16, rim=0.0, rim_interior=0.0, split=0.38, scallop=1.15, term_noise=0.14,
                  levels=LEV_BIG, concave=0.7, clump=0.5, lit_bias=0.0)

    def mass(env, **kw):
        k = dict(common)
        k.update(kw)
        if not isinstance(env, np.ndarray):
            env = dict(env)
            env['cx'] = cx + (env['cx'] - cx) * WF
            env['width'] = env['width'] * WF
        poly = env if isinstance(env, np.ndarray) else K.envelope(rng=rng, **env)
        ids_n.append(1)
        return P.shape(poly, rng=rng, **k)

    shade_low = dict(pal=PAL_BASE, light=0.6, lit_bias=-0.02, form=0.4, cast=0.35, wrap=0.3, sky=0.14,
                     inner=1, inner_val=0.55, inner_size=(0.35, 0.6), shade_top=0.08, valley_dark=0.06,
                     refl=0.6, bounce=0.4)
    # ---- anvil: grows out of the crown; blunt, rounded upwind end on the LEFT (the sun sits behind
    # its crest), long wedge downwind to the right that tears into fibres
    xa = cx + 0.02 * Hc
    long_, short_ = 0.6 * Hc, 0.24 * Hc
    ap = K.anvil_poly(xa, top + 0.03 * Hc, short_, long_, 0.16 * Hc, rng, dome=0.5, rise=0.35, tip=0.08,
                      sag=0.55)
    n_up = len(ap) // 2
    xx = ap[n_up:, 0]
    ap[n_up:, 1] += 0.014 * Hc * np.sin((xx - xa) / (0.08 * Hc) + rng.uniform(0, 6.28)) * \
        np.clip(np.abs(xx - xa) / (0.2 * Hc), 0, 1)
    mass(ap, pal=PAL_ANVIL, levels=((0.06, 0.13, 0.6, 2.0, 4.0, 0.25, 0.5), (0.02, 0.045, 0.45, 1.4, 3.0)),
         size=0.2 * Hc, firm=0.7, soften=0.5, base_dark=0.6, side_scale=0.45, down_cut=0.05, clump=0.6,
         top_bias=1.8, inner=0, fray=(xa + 0.3 * long_, xa + 0.95 * long_, 1.25, 0.4 * long_), refl=0.55,
         wrap=0.0, bounce=0.25, shade_top=0.0, form=0.25, mass_r=0.45, split=0.1, scallop=0.6, sky=0.0,
         lit_bias=0.16, cast=0.0, group=(xa + 0.1 * long_, top + 0.1 * Hc, 0.6 * long_, 0.16 * Hc))
    # ---- crown heads punching up into the anvil's underside (staggered heights, no flat cap)
    mass(dict(cx=cx + 0.2 * Hc, base_y=by - 0.7 * Hc, width=0.3 * Hc, height=0.16 * Hc, power=2.2, lump=0.12,
              base_round=0.5, base_wave=0.06, skew=0.2), size=0.17 * Hc, cast=0.3, side_scale=0.6, form=0.4,
         lit_bias=0.02)
    mass(dict(cx=cx - 0.12 * Hc, base_y=by - 0.72 * Hc, width=0.36 * Hc, height=0.17 * Hc, power=2.2,
              lump=0.12, base_round=0.5, base_wave=0.06, skew=-0.2), size=0.19 * Hc, cast=0.3,
         side_scale=0.62, form=0.4, lit_bias=0.12)
    # ---- the rising column: one stout body from the base up into the crown, leaning right
    mass(dict(cx=cx + 0.03 * Hc, base_y=by - 0.22 * Hc, width=0.56 * Hc, height=0.58 * Hc, power=2.5,
              lump=0.08, lean=0.1, base_round=0.3, base_wave=0.05), size=0.34 * Hc, cast=0.3, side_scale=0.72,
         inner=1, inner_val=0.55)
    # ---- right flank: 2-3 big lobes bulging out on the shadow side, staggered down the column
    mass(dict(cx=cx + 0.3 * Hc, base_y=by - 0.52 * Hc, width=0.32 * Hc, height=0.2 * Hc, power=2.3, lump=0.1,
              skew=0.2, base_round=0.35, base_wave=0.08), size=0.2 * Hc, cast=0.3, side_scale=0.62, form=0.4,
         wrap=0.0, lit_bias=0.14)
    mass(dict(cx=cx + 0.42 * Hc, base_y=by - 0.3 * Hc, width=0.34 * Hc, height=0.24 * Hc, power=2.3, lump=0.1,
              lean=0.04, skew=0.25, base_round=0.35, base_wave=0.08), size=0.22 * Hc, cast=0.35,
         side_scale=0.68, form=0.4, light=0.95, wrap=0.0, inner=1, inner_val=0.55, lit_bias=0.2)
    # ---- sun-side (left) lobes, lit-on-lit: crisp overlapping-mass edges inside the light
    mass(dict(cx=cx - 0.2 * Hc, base_y=by - 0.52 * Hc, width=0.3 * Hc, height=0.19 * Hc, power=2.3, lump=0.1,
              skew=-0.2, base_round=0.35, base_wave=0.08), size=0.19 * Hc, cast=0.25, side_scale=0.6,
         form=0.4, lit_bias=0.06)
    mass(dict(cx=cx - 0.02 * Hc, base_y=by - 0.46 * Hc, width=0.26 * Hc, height=0.15 * Hc, power=2.2, lump=0.1,
              skew=-0.1, base_round=0.3, base_wave=0.08), size=0.15 * Hc, cast=0.25, side_scale=0.6, form=0.35,
         lit_bias=0.04)
    mass(dict(cx=cx - 0.27 * Hc, base_y=by - 0.3 * Hc, width=0.32 * Hc, height=0.21 * Hc, power=2.3, lump=0.1,
              lean=-0.03, skew=-0.25, base_round=0.4, base_wave=0.08), size=0.2 * Hc, cast=0.3,
         side_scale=0.65, form=0.4, lit_bias=0.05)
    mass(dict(cx=cx + 0.14 * Hc, base_y=by - 0.28 * Hc, width=0.28 * Hc, height=0.17 * Hc, power=2.3, lump=0.1,
              skew=0.1, base_round=0.35, base_wave=0.08), size=0.17 * Hc, cast=0.3, side_scale=0.62, form=0.4,
         lit_bias=0.16)
    # ---- low left shoulder, tearing into wisps on its far side
    mass(dict(cx=cx - 0.44 * Hc, base_y=by - 0.04 * Hc, width=0.3 * Hc, height=0.15 * Hc, power=2.3, lump=0.12,
              lean=-0.05, skew=-0.2), size=0.2 * Hc, cast=0.35, side_scale=0.7, form=0.4, inner=1, light=0.95,
         lit_bias=0.1, fray=(cx - 0.45 * Hc, cx - 0.66 * Hc, 0.9, 0.1 * Hc),
         group=(cx - 0.46 * Hc, by - 0.16 * Hc, 0.22 * Hc, 0.2 * Hc))
    for (fx_, fy_, fw_, fh_) in ((-0.6, -0.2, 0.08, 0.03),):
        mass(dict(cx=cx + fx_ * Hc, base_y=by + fy_ * Hc, width=fw_ * Hc, height=fh_ * Hc, power=2.2, lump=0.15),
             size=0.06 * Hc, form=0.3, light=0.85, alpha=0.7,
             fray=(cx + (fx_ + 0.03 * np.sign(fx_)) * Hc, cx + (fx_ + 0.07 * np.sign(fx_)) * Hc, 1.0, 0.05 * Hc),
             group=(cx + fx_ * Hc, by + (fy_ - 0.02) * Hc, 0.08 * Hc, 0.05 * Hc), inner=0, wrap=0.0)
    # ---- lower body: big overlapping masses of varied size, uneven heights
    mass(dict(cx=cx - 0.08 * Hc, base_y=by - 0.02 * Hc, width=0.6 * Hc, height=0.24 * Hc, power=2.4, lump=0.1,
              skew=0.2), size=0.26 * Hc, **shade_low)
    mass(dict(cx=cx + 0.36 * Hc, base_y=by + 0.02 * Hc, width=0.42 * Hc, height=0.17 * Hc, power=2.3, lump=0.1,
              skew=-0.2), size=0.18 * Hc, **dict(shade_low, light=0.5, lit_bias=-0.04))
    mass(dict(cx=cx - 0.3 * Hc, base_y=by + 0.04 * Hc, width=0.38 * Hc, height=0.12 * Hc, power=2.3, lump=0.1,
              skew=0.2), size=0.13 * Hc, **dict(shade_low, inner=0, light=0.7))
    # torn wispy underside, melting into the horizon haze
    P.wisps(cx - 0.66 * Hc, cx + 0.56 * Hc, by + 0.03 * Hc, 0.13 * Hc, rng=rng, pal=PAL_BASE, seed=seed + 11,
            amount=0.45, erode=1.5, base_haze=0.65, haze_color=haze_col)
    # the base is torn apart again just above the rooftops' horizon line so the tower dissolves into the
    # haze there instead of standing on a clean lavender foot
    P.wisps(cx - 0.62 * Hc, cx + 0.56 * Hc, by - 0.19 * Hc, 0.07 * Hc, rng=rng, pal=PAL_BASE, seed=seed + 13,
            amount=0.5, erode=1.1, base_haze=0.55, haze_color=haze_col)
    tipx = xa + long_
    P.fibres(tipx - 0.35 * long_, tipx + 0.45 * long_, top + 0.11 * Hc, 0.03 * Hc, pal=PAL_ANVIL, seed=seed + 21,
             amount=0.5, lit=0.85)
    P.fibres(tipx - 0.2 * long_, tipx + 0.3 * long_, top + 0.16 * Hc, 0.018 * Hc, pal=PAL_ANVIL, seed=seed + 23,
             amount=0.32, lit=0.6)
    K._paint_px = orig
    rgba = P.rgba()
    # de-stair the thresholded value edges and remove the faint interior mottle
    rgb = np.ascontiguousarray(rgba[..., :3])
    k_ = 5 if u > 0.75 else 3
    med = cv2.medianBlur(cv2.medianBlur(rgb, k_), k_)
    keep = ((rgb.max(-1) > 1.0) | (med.max(-1) > 1.0) | (rgba[..., 3] < 0.98))[..., None]
    rgb = np.where(keep, rgb, med)
    rgb = cv2.bilateralFilter(rgb, 0, 0.035, 3.0 * u + 0.5)
    rgba = np.concatenate([rgb, rgba[..., 3:4]], -1)
    # seat the sun on the anvil's upper-left crest: the highest point of the blunt upwind end
    best = None
    for x in np.linspace(xa - 0.3 * Hc, xa - 0.08 * Hc, 40):
        ey = top_edge(rgba[..., 3], x)
        if ey is not None and (best is None or ey < best[1] - 0.004 * Hc):
            best = (float(x), ey)
    sun = (best[0], best[1] + 0.004 * Hc) if best else (xa - 0.2 * Hc, top)
    return rgba, sun, dict(xa=xa, long_=long_, top=top, by=by, ids=ids, vmap=vmap)


# ------------------------------------------------------------------------------------------ repaint
def _c(h):
    return K._c(h)


LIT_HOT = np.array([1.1, 1.085, 1.03], np.float32)     # nearest the sun (slight warm tint)
LIT = np.array([1.06, 1.05, 1.02], np.float32)      # brilliant near-white
LIT_FAR = np.array([1.0, 1.0, 0.99], np.float32)   # far from the sun / low: a touch cooler
MID = _c('#a9b4de')                                    # cool mid step (half-light planes)
MID_LOW = _c('#a5b1dc')
SH_TOP = _c('#6a7aae')                                 # core shadow high up under the anvil
SH = _c('#7888bd')                                     # shadow plane
SH_REFL = _c('#9aa9da')                                # reflected sky light low in the shadow
BOUNCE = _c('#b4b3cf')                                 # warm bounce from the ground haze underneath


def _normal(m, sigma):
    mb = cv2.GaussianBlur(m, (0, 0), sigma)
    gx = cv2.Sobel(mb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mb, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    return gx / gl, gy / gl, gl


def _one_sided(L, kd, u, soft_sigma, hard_sigma):
    """Soften the value boundaries of level map L only where the brighter side faces the sun (the
    terminator: the same mass turning away from the light); boundaries whose brighter side faces away
    (a lit lobe in FRONT of a shadow plane) stay crisp."""
    nx, ny, gl = _normal(L, 4 * u)
    cosk = nx * kd[0] + ny * kd[1]
    soft = K._ss(0.15, 0.65, cosk) * K._ss(0.002, 0.02, gl)
    soft = cv2.GaussianBlur(soft, (0, 0), 3 * u)
    Lsoft = cv2.GaussianBlur(L, (0, 0), soft_sigma)
    Lhard = cv2.GaussianBlur(L, (0, 0), hard_sigma)
    return Lhard * (1 - soft) + Lsoft * soft, soft


def _despeck(m, min_area):
    """Remove islands and fill holes smaller than min_area px (painted masses, no specks)."""
    m8 = (m > 0.5).astype(np.uint8)
    for val in (1, 0):
        src = m8 if val else 1 - m8
        n, lab, st, _ = cv2.connectedComponentsWithStats(src, connectivity=4)
        small = np.zeros(n, bool)
        small[1:] = st[1:, cv2.CC_STAT_AREA] < min_area
        kill = small[lab]
        m8[kill] = 1 - val
    return m8.astype(np.float32)


def _levels(Vs, u, min_area=0):
    lit = (Vs > 0.915).astype(np.float32)
    nsh = (Vs > 0.8).astype(np.float32)
    if min_area:
        lit = _despeck(lit, min_area)
        nsh = _despeck(nsh, min_area)
    k = np.ones((7, 7), np.uint8)
    lit = cv2.morphologyEx(cv2.morphologyEx(lit, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    nsh = cv2.morphologyEx(cv2.morphologyEx(nsh, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    # thin shadow slivers (the painter's terminator band) are not planes: open them away
    kk = int(round(19 * u)) | 1
    ell = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    sh = cv2.morphologyEx((1 - nsh).astype(np.uint8), cv2.MORPH_OPEN, ell).astype(np.float32)
    nsh = 1 - sh
    lit = (cv2.GaussianBlur(lit, (0, 0), 5 * u) > 0.5).astype(np.float32)
    nsh = (cv2.GaussianBlur(nsh, (0, 0), 5 * u) > 0.5).astype(np.float32)
    return lit, np.maximum(nsh, lit)


def _anvil_levels(ids, L, A, top, Hc, xa0, u):
    """Anvil (mass 0): brilliant lit top, a firm scalloped terminator ~55 % of the way down its local
    thickness and a cool shadowed underside (continuous with the shadow recess under it)."""
    m = (ids == 0) & (A > 0.5)
    if not m.any():
        return L
    h, w = m.shape
    rows = np.arange(h, dtype=np.float32)[:, None]
    anyc = m.any(0)
    yt = np.where(anyc, np.argmax(m, 0), 0).astype(np.float32)
    yb = np.where(anyc, h - 1 - np.argmax(m[::-1], 0), 0).astype(np.float32)
    # smooth the per-column profile (no stair-steps from lobes) but keep a scalloped wave on it
    yt = cv2.GaussianBlur(yt[None], (0, 0), 25 * u)[0]
    yb = cv2.GaussianBlur(yb[None], (0, 0), 25 * u)[0]
    xs = np.arange(w, dtype=np.float32)
    wave = 0.012 * Hc * (np.abs(np.sin(xs / (0.035 * Hc) + 1.3)) - 0.5)
    th = np.maximum(yb - yt, 1)
    wx = K._ss(xa0 + 0.1 * Hc, xa0 + 0.28 * Hc, xs)
    ty = yt + (np.clip(0.5 * th, 0.03 * Hc, 0.09 * Hc) + wave) * wx + (th + 0.2 * Hc) * (1 - wx)
    lit = (rows < ty[None]).astype(np.float32)
    La = np.minimum(L, 0.12 + 0.88 * lit)
    out = L.copy()
    out[m] = La[m]
    return out


def finish(rgba, sun, info, W, haze_col='#d4ebf8'):
    u = W / 1920.0
    xa, long_, top, by, ids = info['xa'], info['long_'], info['top'], info['by'], info['ids']
    Hc = by - top
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    rgb = np.ascontiguousarray(rgba[..., :3])
    h, w = A.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = np.clip((ys - top) / Hc, 0, 1)
    dsun = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    kd = np.array([-0.72, -0.7], np.float32)
    kd /= float(np.hypot(*kd))
    # ---- value structure: 3 flat levels from the painted masses
    V = rgb.mean(-1)
    Vi = V * A + 0.96 * (1 - A)
    Vs = cv2.GaussianBlur(cv2.medianBlur(Vi, 5), (0, 0), 2.0 * u)
    lit, nsh = _levels(Vs, u, int((0.04 * Hc) ** 2))
    L = 0.5 * nsh + 0.5 * lit                          # 0 shadow, 0.5 mid, 1 lit
    L = _anvil_levels(ids, L, A, top, Hc, xa, u)
    # ---- overlap structure from the mass ids: a later mass is IN FRONT. Pixels of a back mass near the
    # edge of a front mass get a painted cast / occlusion step - crisp at the front lobe's edge, fading
    # softly away from it (one crisp edge, one soft edge); strongest where the front lobe sits between
    # the pixel and the sun
    idv = ids.astype(np.int32)
    O = np.zeros((h, w), np.float32)
    Ob = np.zeros((h, w), np.float32)
    n_ids = int(idv.max()) + 1
    nz = K._noise(w, h, 9.0, 41, 3)
    nz = (nz - nz.mean()) / (nz.std() + 1e-6) * 0.5
    for i in range(n_ids):
        Ri = idv == i
        if not Ri.any():
            continue
        front = (idv > i).astype(np.uint8)
        if not front.any():
            continue
        d = cv2.distanceTransform(1 - front, cv2.DIST_L2, 5)
        gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=5)
        gl = np.sqrt(gx * gx + gy * gy) + 1e-6
        # -grad d points toward the front mass; toward the sun => shadowed by it
        tow = cv2.GaussianBlur(K._ss(-0.2, 0.7, (-gx * kd[0] - gy * kd[1]) / gl), (0, 0), 14 * u)
        wd = 0.07 * Hc * (0.5 + 0.5 * tow) * (0.75 + 0.5 * nz)
        # a flat painted step: crisp at the front lobe's edge, a short soft fall-off at its far side
        k = K._ss(wd, wd * 0.85, d) * K._ss(0.25, 0.75, tow)
        O = np.where(Ri, np.maximum(O, k), O)
        wb = 0.05 * Hc * (0.75 + 0.5 * nz)
        kb = K._ss(wb, 0.2 * wb, d)
        Ob = np.where(Ri, np.maximum(Ob, kb), Ob)
    Ob = Ob * K._ss(0.5 * u, 1.5 * u, cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5))
    O = O * K._ss(0.5 * u, 1.5 * u, cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5))
    # ---- edges: crisp at mass overlaps, firm-soft (terminator) inside one mass
    bnd = np.zeros((h, w), np.uint8)
    bnd[:, 1:] |= (idv[:, 1:] != idv[:, :-1])
    bnd[1:, :] |= (idv[1:, :] != idv[:-1, :])
    crisp = cv2.GaussianBlur(cv2.dilate(bnd, np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3 * u)
    Lh = cv2.GaussianBlur(L, (0, 0), 0.8 * u)
    nx, ny, gl = _normal(L, 4 * u)
    term = K._ss(0.0, 0.6, nx * kd[0] + ny * kd[1])      # brighter side toward the sun: the terminator
    Ls = cv2.GaussianBlur(L, (0, 0), 1.2 * u) * (1 - term) + cv2.GaussianBlur(L, (0, 0), 3.0 * u) * term
    Lf = Ls * (1 - crisp) + Lh * crisp
    # ---- colour fields
    warm = np.exp(-dsun / 0.5)[..., None]
    low = K._ss(0.35, 0.95, yn)[..., None]
    lit_c = LIT * (1 - low) + LIT_FAR * low
    lit_c = lit_c * (1 - warm) + LIT_HOT * warm
    mid_c = MID * (1 - low) + MID_LOW * low
    shm = 1 - nsh
    rv = np.clip((Vs - 0.52) / 0.28, 0, 1)
    rv = cv2.GaussianBlur(rv * shm, (0, 0), 22 * u) / (cv2.GaussianBlur(shm, (0, 0), 22 * u) + 1e-3)
    g = K._ss(0.15, 0.85, yn)[..., None]
    sh_c = SH_TOP * (1 - g) + SH * g
    sh_c = sh_c * (1 - 0.5 * rv[..., None]) + SH_REFL * (0.5 * rv[..., None])
    bo = (0.45 * K._ss(0.65, 1.0, yn) * K._ss(-0.1, 0.3, (xs - (xa)) / Hc))[..., None]
    sh_c = sh_c * (1 - bo) + BOUNCE * bo
    # per-mass shadow value: masses further forward (painted later) catch a little more reflected
    # light, the ones behind sit deeper -> overlapping shadow masses separate with crisp value steps
    tone = np.zeros((h, w), np.float32)
    for i in range(n_ids):
        tone[idv == i] = ((i * 7) % 5) / 4.0 - 0.5 + 0.6 * (i / max(n_ids - 1, 1) - 0.5)
    tone = cv2.GaussianBlur(tone, (0, 0), 0.8 * u)
    sh_c = sh_c * (1 + 0.07 * tone[..., None])
    # reflected / bounce lip: shadowed edges whose outward normal faces down-right pick up light
    # bounced from the haze and the lower cloud (a lighter band just inside the silhouette)
    Ab_ = cv2.GaussianBlur(A, (0, 0), 10 * u)
    ogx = -cv2.Sobel(Ab_, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(Ab_, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx * ogx + ogy * ogy) + 1e-6
    fdn = K._ss(0.1, 0.7, (ogx * 0.55 + ogy * 0.83) / ogl) * (ogl > 1e-4)
    fdn = cv2.GaussianBlur(fdn.astype(np.float32), (0, 0), 12 * u)
    dIn = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    lip = (K._ss(0.045 * Hc, 0.012 * Hc, dIn) * np.clip(fdn * 2, 0, 1) * K._ss(0.3, 0.6, yn))[..., None]
    sh_c = sh_c * (1 - 0.5 * lip) + _c('#a9b3da') * (0.5 * lip)
    # ---- compose by level, then the overlap steps
    a_ = np.clip(Lf * 2, 0, 1)[..., None]
    b_ = np.clip(Lf * 2 - 1, 0, 1)[..., None]
    col = sh_c * (1 - a_) + mid_c * a_
    col = col * (1 - b_) + lit_c * b_
    Oc = O[..., None]
    litness = np.clip(Lf, 0, 1)[..., None]
    col = col * (1 - 0.8 * Oc * litness) + mid_c * (0.8 * Oc * litness)
    # lit-on-lit: the back mass is a notch lower/cooler right behind a front lobe's edge
    Obc = (Ob * (1 - O))[..., None]
    lo_c = (_c('#e3e6f2') * (1 - low) + _c('#d9def0') * low) * 1.04
    col = col * (1 - 0.75 * Obc * litness) + lo_c * (0.75 * Obc * litness)
    col = col * (1 - 0.08 * Oc * (1 - litness))
    return _post(col, A, sun, info, W, haze_col, ys, xs, yn, dsun, kd)


def _post(col, A, sun, info, W, haze_col, ys, xs, yn, dsun, kd):
    """Silver-gold lining, sunlit glow at the crest, torn anvil tip, torn hazy base."""
    u = W / 1920.0
    xa, long_, top, by = info['xa'], info['long_'], info['top'], info['by']
    Hc = by - top
    h, w = A.shape
    # ---- anvil tip: the solid wedge tapers and tears into wind-sheared fibres (no blunt cut-off)
    st = K._noise(w, h, 5.0, 77, 4, stretch=9.0)
    st = (st - st.mean()) / (st.std() + 1e-6)
    tipk = K._ss(xa + 0.3 * long_, xa + 1.4 * long_, xs) * (1 - K._ss(top + 0.25 * Hc, top + 0.32 * Hc, ys))
    thr = -1.4 + 2.2 * tipk
    keep = K._ss(thr - 0.35, thr + 0.35, st)
    A = A * (1 - tipk + tipk * keep)
    # ---- the lining: a crisp 2-4 px silver/gold line on the sun-facing silhouette only (upper-left
    # flank, the crest and the anvil top), hottest nearest the sun, fading down the left flank
    Ab = cv2.GaussianBlur(A, (0, 0), 9 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gl, -gy / gl
    face = K._ss(-0.1, 0.55, nx * kd[0] + ny * kd[1])
    near = np.exp(-(dsun / 0.22) ** 2)
    up = K._ss(-0.2, 0.6, -ny)
    vz = 1 - K._ss(0.25, 0.6, yn)
    far_fade = 0.3 + 0.7 * np.exp(-dsun / 0.4)
    k = np.clip(np.maximum(face * vz * far_fade, near * np.maximum(up, 0.4 * face)), 0, 1)
    k *= 1 - 0.8 * K._ss(xa + 0.45 * long_, xa + 0.9 * long_, xs)      # not on the fibres
    dI = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    width = (2.0 + 1.6 * near) * u
    ring = K._ss(0.0, 0.8 * u, dI) * K._ss(width + 0.9 * u, width - 0.3 * u, dI)
    kk = np.clip(ring * k, 0, 1)[..., None]
    gold = np.array([1.9, 1.45, 0.9], np.float32)
    silver = np.array([1.5, 1.42, 1.2], np.float32)
    gw = np.clip(near * 1.4, 0, 1)[..., None]
    rim_c = silver * (1 - gw) + gold * gw
    # a narrow band just inside the lining is held a notch down (cooler) so the line reads as light
    # catching the edge, not as an outline
    band = (K._ss(width + 0.3 * u, width + 2.5 * u, dI) * np.exp(-dI / (22 * u)) * k)[..., None]
    col = col * (1 - 0.07 * band * np.array([1.0, 0.99, 0.95], np.float32))
    # backlit crest: near the sun the body just inside the lining is seen against the light - a warm
    # pearl-grey step, so the lining glows as light spilling round the edge
    nearb = np.exp(-(dsun / 0.26) ** 2) * np.clip(0.3 + k, 0, 1)
    kb = cv2.GaussianBlur(k, (0, 0), 6 * u)
    reach = (0.02 + 0.05 * nearb) * Hc
    bl = K._ss(width, width + 2 * u, dI) * K._ss(reach, 0.3 * reach, dI) * np.maximum(nearb, 0.55 * kb)
    bl = cv2.GaussianBlur(bl, (0, 0), 2.0 * u)[..., None]
    col = col * (1 - 0.85 * bl) + np.array([0.8, 0.8, 0.86], np.float32) * (0.85 * bl)
    col = col * (1 - kk) + rim_c * kk
    # a tight halation just outside the lining (light spilling off the edge into the sky)
    hk = cv2.GaussianBlur(kk[..., 0], (0, 0), 1.8 * u) * 0.8 + cv2.GaussianBlur(kk[..., 0], (0, 0), 5 * u) * 0.5
    hk = np.clip(hk * (1 - A) * (0.4 + 0.6 * np.exp(-dsun / 0.3)), 0, 1)
    A_new = np.clip(A + hk * 0.45, 0, 1)
    col = (col * A[..., None] + rim_c * 0.75 * (hk * 0.45)[..., None]) / np.maximum(A_new, 1e-4)[..., None]
    col = np.where(A_new[..., None] > 1e-4, col, rim_c * 0.75)
    A = A_new
    # light through the thin crest nearest the sun: a warm glow just inside the silhouette
    gk = (np.exp(-dI / (0.025 * Hc)) * np.exp(-dsun / 0.14) * (A > 0.5))[..., None]
    col = col + np.array([0.05, 0.04, 0.02], np.float32) * gk
    # ---- base: aerial perspective + torn horizontal fragments melting into the (warm, pale) haze.
    # The colour of the lower body is simplified into one smooth vertical gradient (cool mid -> pale
    # warm haze) so the torn shapes carry the drawing, not smeared streaks.
    hz = K._c(haze_col)
    warmhz = hz * 0.55 + np.array([0.97, 0.955, 0.93], np.float32) * 0.45
    ka = (K._ss(0.52, 0.88, yn) ** 1.1)[..., None]
    grad_t = K._ss(0.55, 0.9, yn)[..., None]
    base_c = _c('#aab6dc') * (1 - grad_t) + warmhz * grad_t
    lum = col.mean(-1, keepdims=True)
    # keep a hint of the lit / shadow split high in the zone, none low down
    keepv = (1 - K._ss(0.55, 0.8, yn))[..., None]
    c2 = base_c + (lum - 0.75) * 0.5 * keepv
    col = col * (1 - ka) + c2 * ka
    fr = K._noise(w, h, 4.0, 91, 4, stretch=10.0)
    fr2 = K._noise(w, h, 16.0, 93, 3, stretch=6.0)
    fr = (fr - fr.mean()) / (fr.std() + 1e-6) * 0.8 + (fr2 - fr2.mean()) / (fr2.std() + 1e-6) * 0.3
    nxo = K._noise(w, 8, 6.0, 97, 3)[:1]
    nxo = (nxo - nxo.mean()) / (nxo.std() + 1e-6)
    tb = K._ss(0.6, 0.96, yn + 0.035 * nxo)
    thr_b = -1.8 + 2.5 * tb
    keep_b = K._ss(thr_b - 0.1, thr_b + 0.15, fr)
    A = A * (1 - tb + tb * keep_b) * (1 - 0.3 * K._ss(0.75, 0.97, yn))
    return np.concatenate([col, A[..., None]], -1).astype(np.float32)


def tower(pw, ph, cx, base_y, Hc, W, seed=12, haze_col='#d4ebf8'):
    rgba, sun, info = raw_tower(pw, ph, cx, base_y, Hc, W, seed=seed, haze_col=haze_col)
    return finish(rgba, sun, info, W, haze_col=haze_col), sun
