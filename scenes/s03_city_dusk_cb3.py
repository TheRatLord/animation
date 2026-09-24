"""Hero cumulonimbus for s03_city_dusk (round 4): built on the shared dome engine (lib.clouds).

* Shape: a designed silhouette -- a wide, lumpy base that narrows into a billowing column and flares
  into a crown, from which a thin sheared anvil shelf streams downwind (right) and frays into fibres
  before the frame edge.
* Light: the sun has just set at lower centre-left. The tower is lit from the left, but the earth's
  shadow climbs it: the N.L attribute is lowered with height-below-the-shadow-line BEFORE painting,
  so the terminator follows the lobes (soft painted steps, no outlines) and only the upper sun-facing
  lobes keep hot pink / gold. Everything else grades to cool lavender / indigo.
* A lower, farther cumulus bank behind the skyline (right): same engine, top-lit, bluer and lighter
  toward the horizon (depth fog), a faint magenta rim on its sun side.
"""
import math
import numpy as np
import cv2

from lib import core as C, sky as S, clouds as K
import s03_city_dusk_sky as SK

PAL = dict(lit_low='#f0508a', lit_high='#ff6e80', hi='#ffc49a', mid='#d0508e',
           shd_high='#7e6cc4', shd_low='#3c3488', bounce='#b0609c', base='#584290',
           rim='#ffd8a8', haze='#9a88cc', fill='#8c80c8', hdr=1.0)
PAL_BANK = dict(lit_low='#d86890', lit_high='#f08aa0', hi='#ffb0b0', mid='#c05896',
                shd_high='#6c62b8', shd_low='#3a3890', bounce='#9c5aa0', base='#4c4494',
                rim='#ff9cc0', haze='#9c90d8', fill='#7a78c8', hdr=1.0)


def _tower(B, rng, cx, base_y, width, height, z0, L, detail=1.3):
    """Designed cumulonimbus tower + anvil into builder B. Returns cloud id."""
    cid = B.cloud([cx, base_y - height * 0.45, z0, width * 0.5, height * 0.55, width * 0.4, L[0], L[1], L[2],
                   base_y, height, 0.0, 0.0])
    clip = base_y
    skd = height * 0.05
    minr = max(width * 0.012, 1.2)

    # half-width profile over v = height fraction (0 base .. 1 top of the crown)
    def hw(v):
        base = 0.2 * math.exp(-(v / 0.25) ** 2) + 0.04 * math.exp(-((v - 0.35) / 0.1) ** 2)             # wide skirt at the bottom
        col = 0.2 + 0.02 * math.sin(v * 6.0 + 1.0) - 0.04 * v              # billowing column
        crown = 0.04 * math.exp(-((v - 0.9) / 0.08) ** 2)  # flare below the anvil
        return (max(base, 0.0) + col + crown) * width

    def xc_at(v):                                          # column leans slightly downwind, crown back
        return cx + width * (0.06 * v - 0.05 * v * v)

    lev0 = []
    v = 0.04
    while v < 0.97:
        w = hw(v)
        R = width * (0.15 - 0.045 * v) * rng.uniform(0.85, 1.15)
        n = max(1, int(round(2 * w / (1.25 * R))))
        yc = base_y - v * height
        for k in range(n):
            u = (k + 0.5) / n * 2 - 1
            x = xc_at(v) + u * (w - R * 0.6) + rng.uniform(-0.15, 0.15) * R
            z = z0 + math.sqrt(max(1 - u * u, 0)) * width * 0.22 + rng.uniform(-0.1, 0.1) * R
            y = yc + rng.uniform(-0.2, 0.2) * R
            if y + R * 0.3 > base_y:
                y = base_y - R * 0.3
            B.dome(x, y, z, R, 0.9, clip, skd, cid, 0, None, tone=rng.uniform(-1, 1))
            lev0.append((x, y, z, R, 0.9, 0.0))
        v += (R / height) * rng.uniform(0.75, 0.95)
    # crown head: a cluster of big rounded masses
    yt = base_y - height
    xh = xc_at(0.95)
    for k in range(5):
        a = -math.pi / 2 + (k - 2) * 0.55 + rng.uniform(-0.15, 0.15)
        R = width * rng.uniform(0.085, 0.11)
        x = xh + math.cos(a) * width * 0.16
        y = yt + R * 0.9 + (1 + math.sin(a)) * height * 0.03
        z = z0 + width * 0.12 + rng.uniform(-0.05, 0.05) * width
        B.dome(x, y, z, R, 0.88, clip, skd, cid, 0, None, tone=rng.uniform(-1, 1))
        lev0.append((x, y, z, R, 0.88, 0.0))
    k1 = K._children(B, rng, lev0, cid, 1, 13 * detail, 0.36, (-0.2, 0.62), base_y, clip, skd, height, width, cx,
                     minr)
    k2 = K._children(B, rng, k1, cid, 2, 7 * detail, 0.4, (-0.2, 0.35), base_y, clip, skd, height, width, cx,
                     minr)
    if detail > 1.1:
        K._children(B, rng, k2, cid, 3, 4, 0.36, (-0.2, 0.5), base_y, clip, skd, height, width, cx, minr)
    return cid


def _light(sdx, sdy, sz):
    n = math.hypot(sdx, sdy) + 1e-9
    c = math.sqrt(1 - sz * sz)
    return (sdx / n * c, sdy / n * c, sz)


def paint(pw, ph, W, H, ox, sun, seed=7, sky=None, ss=2):
    """Straight-alpha RGBA plates (ph, pw, 4): (tower, bank). Frame fractions offset by ox px."""
    X = lambda fx: fx * W + ox
    Y = lambda fy: fy * H
    rng = np.random.default_rng(seed)
    # ------------------------------------------------------------------ hero tower
    B = K.Builder()
    L = _light(-0.93, -0.3, 0.2)
    _tower(B, rng, X(0.83), Y(0.555), 0.3 * W, 0.44 * H, -2.0 * W * 3.0, L, detail=0.75)
    wts0 = K._WTS.copy()
    try:        # lobe-dominated normals -> the terminator follows the cauliflower lobes, not the form ellipse
        K._WTS[:] = np.array([[0.8, 0.0, 0.2], [0.75, 0.17, 0.08], [0.72, 0.2, 0.08], [0.72, 0.2, 0.08]])
        A = K.render(B, pw, ph, ss=ss)
    finally:
        K._WTS[:] = wts0
    ys = np.arange(ph, dtype=np.float32)[:, None]
    xs = np.arange(pw, dtype=np.float32)[None, :]
    # earth's shadow line climbing the tower (slightly tilted: higher away from the sun)
    line = Y(0.30) + (xs - X(0.8)) * 0.12
    esh = C.smoothstep(line - 0.02 * H, line + 0.16 * H, ys)          # 0 lit zone .. 1 deep shadow
    A = A.copy()
    # soften the depth buffer -> no dark crease lines between lobes (soft value steps only)
    dpt = A[..., 6].copy()
    dpt_raw = dpt.copy()
    ins = (A[..., 0] > 1e-3).astype(np.float32)
    sg = 0.006 * W
    A[..., 6] = dpt * 0.75 + 0.25 * (cv2.GaussianBlur(dpt * ins, (0, 0), sg) /
                                     np.maximum(cv2.GaussianBlur(ins, (0, 0), sg), 1e-4))
    # damp the small beads' own shading (they read as pits / craters otherwise): lobes stay readable as
    # soft value steps through the big billows
    lch = A[..., 1]
    lb = cv2.GaussianBlur(lch * ins, (0, 0), 0.005 * W) / np.maximum(cv2.GaussianBlur(ins, (0, 0), 0.005 * W), 1e-4)
    ys_ = np.arange(ph, dtype=np.float32)[:, None]
    xs_ = np.arange(pw, dtype=np.float32)[None, :]
    lz_ = 1 - C.smoothstep(Y(0.3) - 0.04 * H, Y(0.3) + 0.08 * H, ys_ - (xs_ - X(0.8)) * -0.12)
    kk = 0.3 + 0.65 * lz_           # lit zone keeps more of the lobes' own crisp shading
    A[..., 1] = lch * kk + lb * (1 - kk)
    nyc = A[..., 2]
    A[..., 2] = nyc * 0.4 + 0.6 * cv2.GaussianBlur(nyc * ins, (0, 0), 0.005 * W) / np.maximum(cv2.GaussianBlur(ins, (0, 0), 0.005 * W), 1e-4)
    A[..., 1] -= 0.95 * esh
    A[..., 10] -= 0.95 * esh
    skyp = None if sky is None else cv2.resize(sky, (pw, ph))
    pal = S._palette(PAL)
    kw = dict(sky=skyp, seed=seed, soft=0.1, sun_dir=(-0.97, 0.05), texture=0.6)
    P1 = K.paint(A, pal, rim=1.3, halo=1.0, **kw)
    P0 = K.paint(A, pal, rim=0.25, halo=0.2, **kw)
    wr = (1 - C.smoothstep(line - 0.04 * H, line + 0.12 * H, ys))[..., None]
    tower = P0 * (1 - wr) + P1 * wr
    # painted lobe structure: where a lobe sits in front of another, the one behind gets a soft shade
    # step just below/right of the front lobe's edge (away from the light), and the front lobe's
    # sun-facing edge a thin warm rim -> cauliflower read without dark outlines
    insm = A[..., 0] > 0.5
    insf = insm.astype(np.float32)
    lxv, lyv = -0.8, -0.6
    dfill = cv2.GaussianBlur(dpt_raw * insf, (0, 0), 0.003 * W) / np.maximum(cv2.GaussianBlur(insf, (0, 0), 0.003 * W), 1e-4)
    dsm = np.where(insm, dpt_raw, dfill).astype(np.float32)
    shade = np.zeros((ph, pw), np.float32)
    for k, wgt in ((2, 1.0), (5, 0.9), (9, 0.6), (14, 0.35)):
        o = k * H / 1080.0
        M = np.float32([[1, 0, -lxv * o], [0, 1, -lyv * o]])     # neighbour toward the light
        nb = cv2.warpAffine(dsm, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rise = (nb - dsm) / (o + 1.0)                             # depth climbs toward the light = a lobe in front
        shade = np.maximum(shade, C.smoothstep(0.07, 0.35, rise) * wgt)
    shade = cv2.GaussianBlur(shade * insf, (0, 0), 1.0 * H / 1080)
    o = 2.0 * H / 1080.0
    M = np.float32([[1, 0, lxv * o], [0, 1, lyv * o]])            # neighbour away from the light
    nb = cv2.warpAffine(dsm, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    edge = C.smoothstep(0.25, 0.6, (dsm - nb) / (o + 1.0)) * insf * 0
    edge = cv2.GaussianBlur(edge, (0, 0), 0.6 * H / 1080)
    litz = wr[..., 0]
    col = tower[..., :3]
    shade_c = C.lerp(np.array([0.86, 0.4, 0.56], np.float32), np.array([0.5, 0.42, 0.78], np.float32),
                     (1 - litz)[..., None])
    col[:] = C.lerp(col, shade_c, (shade * (0.4 * litz + 0.28 * (1 - litz)))[..., None])
    col[:] = C.lerp(col, np.array([1.25, 0.9, 0.7], np.float32), (edge * 0.6 * litz)[..., None])
    # re-grade the lit zone through a painted pink -> hot pink -> gold ramp with stretched values
    lum = col[..., 0] * 0.3 + col[..., 1] * 0.5 + col[..., 2] * 0.2
    sel = (litz > 0.6) & (tower[..., 3] > 0.9)
    if sel.any():
        lo, hi_ = np.percentile(lum[sel], 4), np.percentile(lum[sel], 97)
        vv = np.clip((lum - lo) / max(hi_ - lo, 1e-4), 0, 1)
        stops = np.array([[0.62, 0.32, 0.62], [0.9, 0.36, 0.58], [1.0, 0.5, 0.6], [1.05, 0.66, 0.6],
                          [1.12, 0.84, 0.66]], np.float32)
        pos = np.linspace(0, 1, len(stops))
        ramp = np.stack([np.interp(vv, pos, stops[:, c]) for c in range(3)], -1).astype(np.float32)
        col[:] = C.lerp(col, ramp, (litz * 0.85)[..., None])
    # crisp 2-3 px silver-gold lining on the silhouette edges that face the set sun (lit zone only)
    al = tower[..., 3]
    rim = np.zeros_like(al)
    for (dx, dy) in ((-1.0, 0.0), (-0.8, -0.6), (-0.9, 0.45)):
        o = 2.6 * H / 1080.0
        M = np.float32([[1, 0, -dx * o], [0, 1, -dy * o]])
        sh_ = cv2.warpAffine(al, M, (pw, ph), flags=cv2.INTER_LINEAR, borderValue=0)
        rim = np.maximum(rim, np.clip(al - sh_, 0, 1))
    rim = rim * (0.25 + 0.75 * litz) * C.smoothstep(0.3, 0.8, al)
    col[:] = C.lerp(col, np.array([1.45, 1.0, 0.72], np.float32), np.clip(rim * 1.1, 0, 1)[..., None])
    # keep the lit pinks just under the bloom threshold (the frame's bloom would otherwise wash them)
    tower[..., :3] *= (1 - 0.12 * wr)
    # warm glow of the hidden sun on the lowest sun-facing flank (bounce from the horizon)
    # + a cool indigo floor near the base (thick, far from any light)
    col = tower[..., :3]
    base_c = C.smoothstep(Y(0.42), Y(0.55), ys)[..., None]
    col[:] = C.lerp(col, col * np.array([0.82, 0.8, 1.0], np.float32), base_c * 0.5)
    # ------------------------------------------------------------------ anvil: torn fibrous shelf
    # painted with the same streak painter as the cirrus (lit pink undersides), streaming from the
    # crown to the right, tapering and fraying out before the frame edge; lies behind the crown.
    specs = [dict(cx=X(0.95), cy=Y(0.1), L=0.36 * W, T=0.036 * H, tier=1.0, tilt=-0.03, sub=2, wave=1.0),
             dict(cx=X(0.86), cy=Y(0.11), L=0.2 * W, T=0.04 * H, tier=1.0, tilt=-0.01, sub=1, wave=0.8)]
    anvil = SK.paint(pw, ph, skyp if skyp is not None else np.zeros((ph, pw, 3), np.float32), specs, sun,
                     seed=seed + 40)
    # fade the downwind end (it thins into the sky before the frame edge)
    fade = 1 - C.smoothstep(X(0.97), X(1.07), xs)
    anvil[..., 3] = np.clip(anvil[..., 3] * 1.25, 0, 1) * fade
    # it is the highest part of the storm: still in direct light -> warm pink body, gold toward the crown
    av = anvil[..., :3]
    pink = C.lerp(np.array([1.0, 0.5, 0.62], np.float32), np.array([1.1, 0.72, 0.6], np.float32),
                  (1 - C.smoothstep(X(0.8), X(1.0), xs))[..., None])
    anvil[..., :3] = C.lerp(av, pink, 0.3) * 1.08
    tower = _over(anvil, tower)
    # ------------------------------------------------------------------ lower bank behind the skyline
    B2 = K.Builder()
    L2 = _light(-0.9, -0.35, 0.05)
    specs = [(X(0.66), Y(0.575), 0.2 * W, 0.075 * H, 4.5, 'cumulus', 0.85),
             (X(0.97), Y(0.585), 0.34 * W, 0.16 * H, 3.2, 'cumulus', 0.95),
             (X(1.08), Y(0.585), 0.2 * W, 0.1 * H, 3.8, 'cumulus', 0.9),
             (X(0.76), Y(0.585), 0.16 * W, 0.08 * H, 5.0, 'stratocumulus', 0.6)]
    for i, sp in enumerate(specs):
        crng = np.random.default_rng(seed * 100 + i)
        fog = float(np.clip((sp[4] - 1.5) / 9.0, 0, 1)) ** 0.75 * 0.85
        K.gen_cloud(B2, crng, sp, L2, fog, 0.0, detail=1.1, z0=-sp[4] * W * 3.0)
    A2 = K.render(B2, pw, ph, ss=ss)
    bank = K.paint(A2, S._palette(PAL_BANK), sky=skyp, seed=seed + 3, rim=0.9, halo=0.6, soft=0.3,
                   sun_dir=(-0.95, -0.2), texture=0.5, terminator=-0.05)
    # aerial perspective: bluer and lighter toward the horizon
    hz = np.array([0.62, 0.5, 0.86], np.float32)
    fz = C.smoothstep(Y(0.46), Y(0.585), ys)[..., None]
    bank[..., :3] = C.lerp(bank[..., :3], hz, fz * 0.45)
    return tower.astype(np.float32), bank.astype(np.float32)


def _over(dst, src):
    """straight-alpha RGBA over straight-alpha RGBA"""
    a = src[..., 3:4] + dst[..., 3:4] * (1 - src[..., 3:4])
    c = (src[..., :3] * src[..., 3:4] + dst[..., :3] * dst[..., 3:4] * (1 - src[..., 3:4])) / np.maximum(a, 1e-5)
    return np.concatenate([c, a], -1).astype(np.float32)
