"""s01 helper (round 7): the hero cumulonimbus repainted with lib/clouds2 Painter.shape.

Every mass shares one tower-wide light (group ellipsoid) blended with its own dome, so value follows
form: each lobe steps warm white -> pale warm grey -> lavender across itself; ivory lobes overlap into
the shadow side and lavender turning-shadows wrap under the lit lobes. Interior overlaps carry no rim /
outline (rim_interior=0); a single plate-level silver lining runs only on the sun-facing crests."""
import numpy as np
import cv2
from lib import clouds2 as K

HAZE = np.asarray(K._c('#d3e9f7'), np.float32)
# fine cauliflower: big heads, dense mid florets, and two small floret sizes along the whole edge
LEV = ((0.13, 0.32, 0.9, 1.3, 2.5, 0.06, 0.4), (0.055, 0.11, 0.95, 0.95, 1.9, 0.2, 0.55),
       (0.026, 0.045, 0.9, 1.0, 2.1), (0.012, 0.02, 0.65, 1.1, 2.4))
PAL = dict(K.PRESETS['noon'])
PAL.update(hi=(0.955, 0.945, 0.915), lit=(0.93, 0.915, 0.88), lit_lo='#d6d2d8', mid='#c9c0c8',
           shade='#a4b0dc', deep='#8290cc', refl='#bac6ea', bounce='#d9cfd2')
# shadow masses: a deeper cool core behind, a mid body, a lighter bounce-lit front/low right
PAL_BACK = dict(PAL, shade='#94a0d6', deep='#7482c6', refl='#a9b5e2')
PAL_MID = dict(PAL, shade='#a6b1dd', deep='#8b98d0', refl='#bcc6ea')
PAL_FRONT = dict(PAL, shade='#b5bde0', deep='#9ea8d6', refl='#cfd0e6', bounce='#e2d4d0')


def tower(pw, ph, cx, base_y, Hc, W, hz_y, seed=12):
    top = base_y - Hc
    sun = np.array([cx - 0.36 * Hc, top - 0.02 * Hc], np.float32)
    u = W / 1920.0
    P = K.Painter(pw, ph, sun, PAL, u, seed=seed + 1, sun_z=0.3)
    rng = np.random.default_rng(seed)
    g = (cx + 0.06 * Hc, base_y - 0.46 * Hc, 0.44 * Hc, 0.56 * Hc)

    def M(dx, h0, w, hh, size=None, lean=0.0, power=2.4, lump=0.1, skew=0.0, base_round=0.15, **kw):
        poly = K.envelope(cx + dx * Hc, base_y - h0 * Hc, w * Hc, hh * Hc, rng, lean=lean, power=power,
                          lump=lump, skew=skew, base_round=base_round)
        kw.setdefault('group', g)
        kw.setdefault('form', 0.55)
        kw.setdefault('cast', 0.3)
        kw.setdefault('wrap', 0.0)
        kw.setdefault('rim_interior', 0.0)
        kw.setdefault('side_scale', 0.8)
        kw.setdefault('levels', LEV)
        kw.setdefault('term_w', 0.08)
        kw.setdefault('rim', 0.0)
        kw.setdefault('bounce_group', 0.35)
        P.shape(poly, rng=rng, size=(size or hh) * Hc, sun=sun, **kw)

    LL = dict(form=0.28, cast=0.45, merge=False, mass_r=0.35, term_w=0.12, term=0.7)
    SH = dict(sky=0.42, cast=0.6, form=0.5, lit_bias=-0.05)
    # ---- back: the body (shadow side dominated)
    M(0.12, 0.0, 0.92, 0.44, size=0.28, base_round=0.1)                    # base body
    M(0.36, 0.12, 0.44, 0.42, size=0.26, lean=0.05, pal=PAL_BACK)         # right tower
    M(0.08, 0.46, 0.46, 0.4, size=0.28, lean=0.03, pal=PAL_BACK)                        # upper column
    # ---- anvil: blunt upwind, long wedge downwind with a feathered fibrous tail
    long_, short_ = 0.95 * Hc, 0.3 * Hc
    xa = cx + 0.05 * Hc
    ap = K.anvil_poly(xa, top + 0.05 * Hc, short_, long_, 0.12 * Hc, rng, dome=0.35, rise=0.3, tip=0.08,
                      sag=0.6)
    n_up = len(ap) // 2
    xx = ap[n_up:, 0]
    ap[n_up:, 1] += 0.014 * Hc * np.sin((xx - xa) / (0.08 * Hc) + 1.3) * np.clip(np.abs(xx - xa) / (0.2 * Hc), 0, 1)
    anv = P.shape(ap, rng=rng, size=0.2 * Hc, sun=sun, levels=((0.02, 0.04, 0.45, 1.4, 3.0, 0.3, 0.5),
                                                         (0.01, 0.02, 0.4, 1.4, 3.0)),
            firm=0.55, soften=0.6, haze=0.03, base_dark=0.45, side_scale=0.3, down_cut=0.08, clump=0.8,
            top_bias=1.6, rim=0.0, fray=(xa + 0.3 * long_, xa + 0.95 * long_, 1.25, 0.5 * long_), refl=0.5,
            wrap=0.0, bounce=0.2, shade_top=0.0, form=0.1, mass_r=0.55, split=0.25, scallop=0.4, sun_z=0.35,
            sun_bias=-0.5, lit_bias=0.06, sky=0.0, rim_interior=0.0, term_w=0.25,
            group=(xa + 0.35 * long_, top + 0.12 * Hc, 0.8 * long_, 0.2 * Hc))
    # ---- the anvil overhang shades the column top: a deep cool blue-violet core right under it
    # (crisp where the anvil's underside cuts it, fading softly downward)
    if anv is not None:
        am, ax0, ay0 = anv['mask'], anv['x0'], anv['y0']
        cols = np.arange(am.shape[1])
        has = (am > 0.5).any(0)
        bot = np.where(has, am.shape[0] - 1 - np.argmax((am > 0.5)[::-1], 0), -1).astype(np.float32) + ay0
        bot = cv2.GaussianBlur(bot[None, :], (0, 0), sigmaX=6 * u + 1, sigmaY=0.1)[0]
        X0, X1 = max(int(ax0), 0), min(int(ax0 + am.shape[1]), P.w)
        Y0, Y1 = max(int(ay0), 0), min(int(ay0 + am.shape[0] + 0.3 * Hc), P.h)
        yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
        bb = bot[X0 - ax0:X1 - ax0][None, :]
        xx = np.arange(X0, X1, dtype=np.float32)[None, :]
        dd = yy - bb
        k = np.exp(-np.clip(dd, 0, None) / (0.1 * Hc)) * (dd > -1) * (bb > 0)
        k = k * K._ss(cx - 0.12 * Hc, cx + 0.12 * Hc, xx) * K._ss(cx + 0.6 * Hc, cx + 0.35 * Hc, xx) * 0.55
        reg = P.prem[Y0:Y1, X0:X1]
        amask = np.zeros((Y1 - Y0, X1 - X0), np.float32)
        sub = am[Y0 - ay0:min(Y1 - ay0, am.shape[0]), X0 - ax0:X1 - ax0]
        amask[:sub.shape[0]] = sub
        k = (k * (1 - amask))[..., None]
        deep = np.asarray(K._c('#7684c6'), np.float32)
        reg[..., :3] = reg[..., :3] * (1 - k) + deep * reg[..., 3:4] * k
    # ---- crown: overshooting top punching up through the anvil next to the sun
    M(-0.04, 0.83, 0.36, 0.18, size=0.15, power=2.2, base_round=0.05, form=0.45, cast=0.35)
    M(0.13, 0.8, 0.22, 0.11, size=0.1, power=2.2, base_round=0.1, form=0.45)
    # ---- shadow side masses (right), in front of the column
    M(0.27, 0.44, 0.32, 0.26, size=0.2, skew=0.2, pal=PAL_MID, **SH)        # right shoulder
    M(0.26, 0.26, 0.4, 0.28, size=0.22, skew=0.15, pal=PAL_MID, **SH)
    M(0.4, 0.08, 0.34, 0.26, size=0.2, pal=PAL_FRONT, **SH)
    # ---- lit lobes crossing the seam into the shadow side at mid height (own dome dominates)
    M(0.13, 0.57, 0.3, 0.19, size=0.16, lit_bias=0.1, skew=-0.3, **LL)
    # ---- the lit face: big ivory lobes stacked down the sunward flank, each with a crisp top edge and
    # a lavender turning shadow wrapped under it
    M(-0.1, 0.6, 0.42, 0.25, size=0.22, skew=-0.25, **LL)             # L1 upper
    M(-0.12, 0.38, 0.42, 0.28, size=0.24, skew=-0.2, **LL)            # L2 middle
    M(-0.14, 0.15, 0.44, 0.28, size=0.25, skew=-0.3, **LL)            # L3 lower
    M(0.06, 0.1, 0.36, 0.2, size=0.18, pal=PAL_FRONT, **SH)            # low centre (shadow, bounce-lit)
    M(-0.33, 0.02, 0.28, 0.16, size=0.15, form=0.5, haze_grad=0.3)                   # left foot
    M(0.46, -0.01, 0.34, 0.15, size=0.14, form=0.5, haze_grad=0.3)                  # right foot
    P.fibres(xa + 0.6 * long_, xa + 1.5 * long_, top + 0.13 * Hc, 0.035 * Hc, seed=seed + 21, amount=0.55, lit=0.85)
    P.fibres(xa + 0.8 * long_, xa + 1.7 * long_, top + 0.17 * Hc, 0.02 * Hc, seed=seed + 23, amount=0.35, lit=0.6)
    P.fibres(xa - 0.8 * short_, xa - 1.4 * short_, top + 0.1 * Hc, 0.02 * Hc, seed=seed + 25, amount=0.35, lit=0.9)
    P.wisps(cx - 0.62 * Hc, cx + 0.7 * Hc, hz_y - 0.01 * Hc, 0.07 * Hc, rng=np.random.default_rng(seed + 3),
            seed=seed + 11, amount=0.5, erode=0.8, base_haze=0.5, haze_color=HAZE)
    P.haze_band(hz_y - 0.22 * Hc, hz_y + 0.02 * Hc, HAZE, 0.55)
    yy = np.arange(P.h, dtype=np.float32)[:, None, None]
    P.prem *= K._ss(hz_y + 0.06 * Hc, hz_y + 0.01 * Hc, yy)
    rgba = P.rgba()
    rgba = _rim(rgba, sun, Hc, top, u)
    return rgba, sun


def _rim(rgba, sun, Hc, top, u):
    """Continuous 2-4 px silver lining ONLY on the outer silhouette where it faces the sun, on lit
    paint and on the upper part of the tower (crown + upper-left lobes); warmer right beside the sun.
    Edges turned away from the sun and all interior overlaps get nothing."""
    _ss = K._ss
    A = np.ascontiguousarray(np.clip(rgba[..., 3], 0, 1))
    h, w = A.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = sun[0] - xs, sun[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl
    rb = max(int(round(2.6 * u)), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rb + 1, 2 * rb + 1))
    core = K._blur(np.clip(A - cv2.erode(A, ker), 0, 1), 0.45 * u)
    Ab = K._blur(A, 4 * u)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    face = _ss(0.2, 0.65, np.clip(-(gx * ux + gy * uy) / gl, 0, 1))
    prox = np.exp(-dl / (0.8 * Hc))
    upper = 1 - _ss(top + 0.5 * Hc, top + 0.8 * Hc, ys)
    litn = _ss(0.7, 0.82, rgba[..., :3].min(-1))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(7 * u) + 1,) * 2)
    solid = K._blur(cv2.dilate(cv2.erode((A > 0.5).astype(np.uint8), ko), ko, iterations=2).astype(np.float32), 1.5 * u)
    k = np.clip(core * 1.25, 0, 1) * face * (0.4 + 0.6 * prox) * upper * litn * solid
    k = np.clip(k, 0, 1)[..., None]
    near = np.exp(-dl / (0.25 * Hc))[..., None]
    col = np.array([1.4, 1.4, 1.38], np.float32) * (1 - near) + np.array([1.5, 1.38, 1.12], np.float32) * near
    # a slightly cooler band just inside the lining (the lit face turning under the edge): the lining pops
    er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(9 * u) + 1,) * 2)
    band = np.clip(A - K._blur(cv2.erode(A, er), 2.5 * u), 0, 1)
    kb = (band * face * (0.4 + 0.6 * prox) * upper * litn * solid)[..., None]
    out = rgba.copy()
    out[..., :3] = rgba[..., :3] * (1 - 0.08 * kb * np.array([1.0, 0.95, 0.8], np.float32))
    out[..., :3] = out[..., :3] * (1 - k) + col * k
    # faint halation just outside the lining
    hk = (K._blur(k[..., 0] * A, 1.8 * u) * (1 - A) * 0.35)[..., None]
    a2 = np.clip(A[..., None] + hk * 0.5, 0, 1)
    out[..., :3] = np.where(a2 > 1e-4, (out[..., :3] * A[..., None] + col * hk * 0.5) / np.maximum(a2, 1e-4),
                            rgba[..., :3])
    out[..., 3:4] = a2
    return out
