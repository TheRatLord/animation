"""s06_seaside sky (round 4): the two sunset cumulus heaps flanking the low sun, painted with
lib/clouds2's cumulonimbus recipe (no anvil): a few big flat masses sharing ONE tower ellipsoid for their
light (high `form`), so each heap reads as 2-3 large smooth lit planes (warm white-gold top, peach mid)
with one scalloped terminator into cool lavender shade, crisp overlapping-mass edges, cauliflower detail
concentrated on the silhouette, a 2-4 px gold lining on the sun-facing edges and torn wispy bases.
The right heap is painted mirrored (sun mirrored too) and flipped back, so both heaps turn their
stepped, lit flank toward the sun."""
import math
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
    base = dict(group=g, form=0.6, scallop=1.5, levels=LEV, wrap=0.3, term_w=0.07, rim_interior=0.0, side_scale=0.72,
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
    # varied-scale heads riding on the crown and the sunward flank: a few big crowns carry small sub-heads,
    # so the silhouette steps big / small / medium instead of a row of equal bumps
    for (dx, dy, w_, h_, sz) in ((0.1, -0.86, 0.2, 0.13, 0.1), (0.3, -0.7, 0.13, 0.09, 0.07),
                                 (-0.08, -0.84, 0.11, 0.08, 0.06), (0.46, -0.44, 0.17, 0.15, 0.1),
                                 (0.55, -0.24, 0.1, 0.09, 0.06), (0.24, -0.84, 0.07, 0.05, 0.045),
                                 (0.56, -0.08, 0.13, 0.1, 0.08)):
        m(dx, dy, w_, h_, sz, env=dict(power=2.0, lump=0.18, base_round=0.5))
    # low body: its own shadow underneath, warm bounce
    m(-0.02, 0.02, 0.9, 0.24, 0.24, env=dict(power=2.2, lump=0.08, skew=0.3, base_round=0.1, base_wave=0.05),
      light=0.85, haze=0.04, haze_grad=0.3, inner=1, inner_val=0.5)
    m(-0.42, 0.03, 0.44, 0.16, 0.16, env=dict(power=2.4, base_round=0.12, base_wave=0.05), light=0.8,
      haze=0.06, haze_grad=0.35)


def _heap(pw, ph, sun, x0, by, Hc, seed, u, width=1.0, mirror=False, ph_key=None):
    if mirror:
        sun = (pw - 1 - sun[0], sun[1])
        x0 = pw - 1 - x0
    key = (sun[0], sun[1] - KEY_UP * (ph if ph_key is None else ph_key))
    P = K.Painter(pw, ph, key, PAL, u, seed=seed + 1, sun_z=SUN_Z)
    heap(P, x0, by, Hc, seed, width)
    rng = np.random.default_rng(seed + 5)
    P.wisps(x0 - 0.62 * Hc * width, x0 + 0.62 * Hc * width, by + 0.02 * Hc, 0.12 * Hc, rng=rng, pal=PAL,
            seed=seed + 11, amount=0.6, erode=1.2, base_haze=0.45, haze_color='#eaa0b0')
    P.lining(1.0, rim_px=2.2, halo=0.25, backlit=0.08)
    rgba = _soft_shade_edges(_soften_planes(P.rgba(), u), u)
    if mirror:
        rgba = np.ascontiguousarray(rgba[:, ::-1])
    return rgba


def _soften_planes(rgba, u):
    """internal plane boundaries painted wet-into-wet: inside the silhouette the flat cel planes are blended
    toward an alpha-normalised blur of themselves (no sky bleeds in), strongly in the mid / shade values and
    lightly in the bright lit planes, so only the lit top edges and the sun-side silhouette stay crisp."""
    A = np.clip(rgba[..., 3], 0, 1)
    rgb = rgba[..., :3]
    pm = rgb * A[..., None]
    out = rgba.copy()
    acc = np.zeros_like(rgb)
    for sg, wt in ((4.0, 0.55), (11.0, 0.45)):
        num = cv2.GaussianBlur(pm, (0, 0), sg * u)
        den = cv2.GaussianBlur(A, (0, 0), sg * u)
        acc += wt * num / np.maximum(den, 1e-4)[..., None]
    lum = rgb.max(-1)
    lit = C.smoothstep(0.78, 0.95, lum) * C.smoothstep(0.1, 0.3, rgb[..., 0] - rgb[..., 2])
    inner = C.smoothstep(0.5, 0.98, cv2.GaussianBlur(A, (0, 0), 3.0 * u))
    k = inner * (0.68 - 0.42 * lit)
    out[..., :3] = rgb + (acc - rgb) * k[..., None]
    return out


def _soft_shade_edges(rgba, u):
    """lost edges on the shade side: where the cloud paint is cool / unlit (and on down-facing edges) the
    silhouette dissolves over a few px into the sky; lit (warm, bright) edges stay crisp."""
    A = rgba[..., 3]
    rgb = rgba[..., :3]
    lit = C.smoothstep(0.62, 0.86, rgb.max(-1)) * C.smoothstep(0.0, 0.25, rgb[..., 0] - rgb[..., 2])
    litb = cv2.GaussianBlur(lit * A, (0, 0), 6 * u) / np.maximum(cv2.GaussianBlur(A, (0, 0), 6 * u), 1e-4)
    Ab = 0.6 * cv2.GaussianBlur(A, (0, 0), 3 * u) + 0.4 * cv2.GaussianBlur(A, (0, 0), 7 * u)
    gy, gx = np.gradient(cv2.GaussianBlur(A, (0, 0), 8 * u))
    gm = np.hypot(gx, gy) + 1e-6
    down = C.smoothstep(0.2, 0.9, gy / gm)
    w = np.clip((1 - litb) * 1.0 + down * 0.7, 0, 1)
    soft = np.minimum(A, np.clip(Ab * 1.15 - 0.15, 0, 1) ** 1.3)
    out = rgba.copy()
    out[..., 3] = A * (1 - w) + soft * w
    return out


def _grade(rgba, sun, u, seed=0):
    """paint-over of one heap: warm lit top -> lavender / violet base (vertical grade over the heap), the
    salmon shade plane broken into 2-3 soft values (wet-into-wet, low-frequency value drift), the underside
    dissolved into the sky (lost edges), and a thin incandescent near-white gold rim ONLY on the silhouette
    that faces the sun."""
    A = np.clip(rgba[..., 3], 0, 1)
    ys, xs = np.nonzero(A > 0.05)
    if not len(ys):
        return rgba
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    pad = int(40 * u) + 4
    Y0, Y1 = max(y0 - pad, 0), min(y1 + pad, rgba.shape[0])
    X0, X1 = max(x0 - pad, 0), min(x1 + pad, rgba.shape[1])
    sub = rgba[Y0:Y1, X0:X1].copy()
    A = np.clip(sub[..., 3], 0, 1)
    rgb = sub[..., :3]
    h, w = A.shape
    yy = (np.arange(Y0, Y1, dtype=np.float32)[:, None] - y0) / max(y1 - y0, 1)
    tv = np.broadcast_to(yy, (h, w))
    lum = rgb.max(-1)
    warmth = rgb[..., 0] - rgb[..., 2]
    lit = C.smoothstep(0.84, 0.97, lum) * C.smoothstep(0.15, 0.35, warmth)
    # 1) shade plane broken into soft values: alpha-normalised wide blur + low-frequency value drift
    pm = rgb * A[..., None]
    wide = cv2.GaussianBlur(pm, (0, 0), 22 * u) / np.maximum(cv2.GaussianBlur(A, (0, 0), 22 * u), 1e-4)[..., None]
    nz = C.fbm(w, h, max(w / (160 * u), 2), 3, seed=900 + seed) - 0.5
    shd = (1 - lit) * C.smoothstep(0.2, 0.9, cv2.GaussianBlur(A, (0, 0), 6 * u))
    rgb = rgb + (wide - rgb) * (0.35 * shd)[..., None]
    rgb = rgb * (1 + 0.16 * nz * shd)[..., None]
    # 2) vertical grade: warm lit top, lavender -> violet base
    lav = C.hex2rgb('#b08ccc')
    vio = C.hex2rgb('#7a68b8')
    base_c = lav + (vio - lav) * C.smoothstep(0.7, 1.0, tv)[..., None]
    wb = C.smoothstep(0.35, 1.0, tv) * (1 - 0.75 * lit) * 0.6
    rgb = rgb + (base_c * (0.55 + 0.5 * lum)[..., None] - rgb) * wb[..., None]
    top = (1 - C.smoothstep(0.0, 0.45, tv)) * (0.4 + 0.6 * lit)
    rgb = rgb * (1 + np.array([0.06, 0.05, 0.0], np.float32) * top[..., None])
    # 3) underside lost into the sky
    Ab = cv2.GaussianBlur(A, (0, 0), 9 * u)
    und = C.smoothstep(0.62, 1.0, tv) * (1 - lit)
    A2 = A * (1 - und) + np.minimum(A, np.clip(Ab * 1.2 - 0.1, 0, 1) ** 1.6) * und
    # 4) thin incandescent rim on the sun-facing silhouette
    sx, sy = sun[0] - X0, sun[1] - Y0
    gx_ = np.arange(w, dtype=np.float32)[None] - sx
    gy_ = np.arange(h, dtype=np.float32)[:, None] - sy
    dn = np.sqrt(gx_ ** 2 + gy_ ** 2) + 1e-3
    dxs, dys = -gx_ / dn, -gy_ / dn                              # unit vector toward the sun
    Abl = cv2.GaussianBlur(A, (0, 0), 4 * u)
    gyA, gxA = np.gradient(Abl)
    gmA = np.hypot(gxA, gyA) + 1e-6
    facing = C.smoothstep(0.2, 0.7, -(gxA * dxs + gyA * dys) / gmA)
    d = 2.0 * u
    mx = (np.arange(w, dtype=np.float32)[None] + dxs * d).astype(np.float32)
    my = (np.arange(h, dtype=np.float32)[:, None] + dys * d).astype(np.float32)
    Ash = cv2.remap(A, np.broadcast_to(mx, (h, w)).astype(np.float32), np.broadcast_to(my, (h, w)).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    edge = np.clip(A - Ash, 0, 1) * facing * (1 - 0.25 * C.smoothstep(0.6, 1.0, tv))
    edge = np.clip(cv2.GaussianBlur(edge, (0, 0), 0.6 * u) * 1.3, 0, 1)
    rimc = np.array([1.75, 1.5, 1.1], np.float32)
    rgb = rgb + (rimc - rgb) * (0.85 * edge)[..., None]
    sub[..., :3] = rgb
    sub[..., 3] = A2
    out = rgba.copy()
    out[Y0:Y1, X0:X1] = sub
    return out


def cumulus(pw, ph, sun, W, H, ox, oy, seed=61):
    """two sunset heaps flanking the low sun -> straight RGBA plate (plate px)."""
    Hs = H / 1080.0
    u = W / 1920.0
    pk = int(H * 1.08)       # key-light offset relative to the nominal plate height (plate may be taller)
    a = _heap(pw, ph, sun, 0.205 * W + ox, 0.39 * H + oy, 390 * Hs, 71, u, width=1.35, ph_key=pk)
    b = _heap(pw, ph, sun, 0.9 * W + ox, 0.405 * H + oy, 370 * Hs, 77, u, width=1.3, mirror=True, ph_key=pk)
    a = _grade(a, sun, u, 1)
    b = _grade(b, sun, u, 2)
    if REPAINT:
        a = _repaint(a, sun, u, BILLOWS_L, 3)
        b = _repaint(b, sun, u, BILLOWS_R, 4)
    return _over(a, b)


REPAINT = True
# value ramp (luma of the painted cloud -> colour): cool blue-violet base / lavender shade -> rose-lavender
# terminator -> peach -> gold -> hot white-gold on the sunward edge (Shinkai warm / cool split)
RAMP = ((0.30, (0.30, 0.29, 0.58)), (0.42, (0.40, 0.38, 0.70)), (0.52, (0.53, 0.49, 0.80)),
        (0.61, (0.68, 0.57, 0.83)), (0.69, (0.86, 0.60, 0.70)), (0.76, (0.98, 0.66, 0.54)),
        (0.83, (1.0, 0.76, 0.54)), (0.9, (1.03, 0.88, 0.64)), (0.97, (1.1, 1.0, 0.8)), (1.06, (1.25, 1.18, 1.0)))
# interior sub-billow planes: (s = 0 far side .. 1 sun side, v = 0 top .. 1 bottom, rx, ry in bbox-width units)
BILLOWS_L = ((0.74, 0.3, 0.13, 0.085), (0.6, 0.5, 0.15, 0.09), (0.83, 0.6, 0.11, 0.075))
BILLOWS_R = ((0.72, 0.28, 0.12, 0.08), (0.58, 0.48, 0.14, 0.085), (0.84, 0.62, 0.1, 0.07))


def _ramp(t):
    xs = np.array([r[0] for r in RAMP], np.float32)
    cs = np.array([r[1] for r in RAMP], np.float32)
    out = np.empty(t.shape + (3,), np.float32)
    for k in range(3):
        out[..., k] = np.interp(t, xs, cs[:, k])
    return out


def _repaint(rgba, sun, u, billows, seed):
    """paint-over #2: re-colour the heap through a warm -> cool value ramp (the salmon terminator plane goes
    cool rose-lavender, the flat cream lit fill becomes a gradation from a hot white-gold sunward edge through
    gold / peach), add 2-3 interior sub-billow planes (crisp scalloped lit tops, lost soft bottoms, a cool
    shadow band above each top) and a thin glowing rim + faint bloom on the sun-facing silhouette."""
    A = np.clip(rgba[..., 3], 0, 1)
    ys, xs = np.nonzero(A > 0.05)
    if not len(ys):
        return rgba
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    pad = int(40 * u) + 4
    Y0, Y1 = max(y0 - pad, 0), min(y1 + pad, rgba.shape[0])
    X0, X1 = max(x0 - pad, 0), min(x1 + pad, rgba.shape[1])
    sub = rgba[Y0:Y1, X0:X1].copy()
    A = np.clip(sub[..., 3], 0, 1)
    rgb = sub[..., :3]
    h, w = A.shape
    bw = float(x1 - x0)
    sun_right = sun[0] > 0.5 * (x0 + x1)
    Y = rgb @ np.array([0.3, 0.59, 0.11], np.float32)
    # proximity to the sun-facing silhouette (hot edge) and the vertical position in the heap
    sx, sy = sun[0] - X0, sun[1] - Y0
    gx_ = np.arange(w, dtype=np.float32)[None] - sx
    gy_ = np.arange(h, dtype=np.float32)[:, None] - sy
    dn = np.sqrt(gx_ ** 2 + gy_ ** 2) + 1e-3
    dxs, dys = -gx_ / dn, -gy_ / dn
    Abl = cv2.GaussianBlur(A, (0, 0), 4 * u)
    gyA, gxA = np.gradient(Abl)
    gmA = np.hypot(gxA, gyA) + 1e-6
    facing = C.smoothstep(0.1, 0.6, -(gxA * dxs + gyA * dys) / gmA)
    edge = C.smoothstep(0.02, 0.2, gmA * 12 * u) * facing
    prox = np.clip(cv2.GaussianBlur(edge, (0, 0), 26 * u) * 5.0, 0, 1)
    prox2 = np.clip(cv2.GaussianBlur(edge, (0, 0), 70 * u) * 6.0, 0, 1)
    tv = (np.arange(Y0, Y1, dtype=np.float32)[:, None] - y0) / max(y1 - y0, 1)
    litw = C.smoothstep(0.72, 0.84, Y)
    t = Y + litw * (0.09 * prox + 0.06 * prox2 - 0.1) - 0.04 * C.smoothstep(0.5, 1.0, tv) * litw
    # ---- sub-billow planes
    r = np.random.default_rng(seed)
    X = np.arange(X0, X1, dtype=np.float32)[None]
    Yg = np.arange(Y0, Y1, dtype=np.float32)[:, None]
    for (sv, vv, rx, ry) in billows:
        cx = x0 + (sv if sun_right else 1 - sv) * bw
        cy = y0 + vv * (y1 - y0)
        rxp, ryp = rx * bw, ry * bw
        dx, dy = (X - cx) / rxp, (Yg - cy) / ryp
        th = np.arctan2(dy, dx)
        e = np.sqrt(dx * dx + dy * dy) * (1 + 0.03 * np.sin(3 * th + r.uniform(0, 6)))
        sd = ryp * (1 - e)                                   # signed distance (px, + inside)
        # cauliflower heads along the upper arc of the billow (varied size)
        nk = int(r.integers(4, 7))
        for ang in np.sort(r.uniform(-2.75, -0.4, nk)):
            rk = ryp * r.uniform(0.28, 0.5)
            ccx = cx + math.cos(ang) * (rxp - rk * 0.75)
            ccy = cy + math.sin(ang) * (ryp - rk * 0.75)
            sd = np.maximum(sd, rk - np.sqrt((X - ccx) ** 2 + (Yg - ccy) ** 2))
        crisp = np.clip(sd / (1.2 * u) + 0.5, 0, 1)
        topw = C.smoothstep(0.55, -0.35, dy)                       # lit top, dissolving toward the bottom
        side = C.smoothstep(-0.6, 0.6, dx if sun_right else -dx)   # brighter on its sunward shoulder
        pl = crisp * topw * A
        t = t + pl * (0.07 + 0.08 * side)
        band = np.clip(-sd / (0.22 * ryp), 0, 1)
        band = (1 - band) * (sd < 0) * C.smoothstep(0.2, -0.5, dy) * A
        t = t - 0.11 * cv2.GaussianBlur(band.astype(np.float32), (0, 0), 1.5 * u)
    col = _ramp(t)
    # keep a little of the original hue variation, mostly the new ramp
    k = 0.78 * C.smoothstep(0.1, 0.6, A)
    rgb = rgb + (col - rgb) * k[..., None]
    # ---- thin glowing rim on the sun-facing silhouette + faint bloom outside it
    d = 1.6 * u
    mx = (np.arange(w, dtype=np.float32)[None] + dxs * d).astype(np.float32)
    my = (np.arange(h, dtype=np.float32)[:, None] + dys * d).astype(np.float32)
    Ash = cv2.remap(A, np.broadcast_to(mx, (h, w)).astype(np.float32), np.broadcast_to(my, (h, w)).astype(np.float32),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fac2 = C.smoothstep(0.25, 0.75, -(gxA * dxs + gyA * dys) / gmA) * (1 - 0.6 * C.smoothstep(0.7, 1.0, tv))
    rim = np.clip((A - Ash) * 1.4, 0, 1) * fac2
    rim = np.clip(cv2.GaussianBlur(rim, (0, 0), 0.5 * u) * 1.2, 0, 1)
    rgb = rgb + (np.array([1.8, 1.55, 1.15], np.float32) - rgb) * (0.9 * rim)[..., None]
    glow = cv2.GaussianBlur(rim * A, (0, 0), 6 * u) * 1.6 + cv2.GaussianBlur(rim * A, (0, 0), 16 * u) * 1.2
    ga = np.clip(glow * 0.5, 0, 0.35) * (1 - A)
    gc = np.array([1.0, 0.8, 0.55], np.float32)
    A2 = A + ga
    rgb = (rgb * A[..., None] + gc * ga[..., None]) / np.maximum(A2, 1e-5)[..., None]
    sub[..., :3] = rgb
    sub[..., 3] = A2
    out = rgba.copy()
    out[Y0:Y1, X0:X1] = sub
    return out


def _over(a, b):
    ab, aa = b[..., 3:4], a[..., 3:4]
    al = ab + aa * (1 - ab)
    rgb = (b[..., :3] * ab + a[..., :3] * aa * (1 - ab)) / np.maximum(al, 1e-5)
    rgb = np.where(al > 1e-5, rgb, a[..., :3] * (aa > 0) + b[..., :3] * (aa <= 0))
    return np.concatenate([rgb, al], -1).astype(np.float32)
