"""s10 round 15 - the sea of clouds painted as continuous, overlapping BANKS of masses (not cards).

Every mass is one lib.clouds2 cauliflower silhouette (edge-only florets). Its value planes are painted
the way a background painter blocks them in: the light plane is the silhouette itself offset AWAY from
the sun (so the terminator is a scalloped echo of the big forms, wide where a big head bulges toward the
sun, thin or absent elsewhere), a second, softer offset gives the mid plane, the rest is cool shadow that
deepens inside the body and picks up a little reflected light low down, then dissolves into a torn,
streaky, feathered base (no bottom edge ever reads). Light direction is per pixel: toward the sun and up
(masses left of the sun axis are lit on their upper-right, right of it on their upper-left).
Aerial perspective: palette + haze by distance (far rows melt into the pale peach horizon with no rims,
low contrast, small scale; near rows larger, cooler, deeper). Rims: a continuous line only on edges that
face the sun (offset test of the actual floret mask), 2-4 px hot gold on the sun axis for near rows,
thinner silver-pink away from it, none on far rows or on edges turned away.
Plates are dicts {rgba (straight alpha), ox, oy} in frame px (see s10_sea_of_clouds_r11.place_fast).
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import core as C, clouds2 as K  # noqa: E402
import s10_sea_of_clouds_r14 as N  # noqa: E402

_ss = K._ss


def _c(h):
    return np.asarray(K._c(h), np.float32)


def _mix(a, b, t):
    return a * (1 - t) + b * t


def _blur(img, s):
    if s < 0.3:
        return img
    return K._blur(img, s)


# ---------------------------------------------------------------------------------------- palettes
# per-pixel palette = mix(OFF, SUN, axis) then mix(.., NEAR, near)
PK = ('hot', 'lit', 'lit_lo', 'mid', 'shade', 'deep', 'refl')
P_SUN = dict(hot=(1.0, 0.95, 0.86), lit=(1.0, 0.87, 0.72), lit_lo=(0.95, 0.72, 0.7), mid=(0.8, 0.7, 0.82),
             shade=(0.6, 0.6, 0.84), deep=(0.5, 0.51, 0.78), refl=(0.66, 0.64, 0.84))
P_OFF = dict(hot=(0.98, 0.9, 0.9), lit=(0.93, 0.82, 0.86), lit_lo=(0.84, 0.74, 0.86), mid=(0.7, 0.68, 0.86),
             shade=(0.53, 0.56, 0.84), deep=(0.43, 0.47, 0.78), refl=(0.58, 0.6, 0.84))
P_NEAR = dict(hot=(1.0, 0.9, 0.8), lit=(0.95, 0.8, 0.76), lit_lo=(0.8, 0.66, 0.8), mid=(0.58, 0.57, 0.84),
              shade=(0.4, 0.43, 0.78), deep=(0.29, 0.33, 0.66), refl=(0.46, 0.49, 0.79))
TW = dict(hot=(1.1, 1.05, 0.97), lit=(1.07, 0.99, 0.9), lit_lo=(1.0, 0.86, 0.82), mid=(0.8, 0.75, 0.88),
          shade=(0.62, 0.63, 0.87), deep=(0.52, 0.55, 0.82), refl=(0.68, 0.68, 0.88))
HAZE_SUN = np.asarray((1.0, 0.88, 0.74), np.float32)
HAZE_OFF = np.asarray((0.91, 0.84, 0.89), np.float32)
RIM_SUN = np.asarray((1.5, 1.12, 0.66), np.float32)
RIM_OFF = np.asarray((1.02, 0.9, 0.98), np.float32)


def _P(d):
    return {k: np.asarray(d[k], np.float32) for k in PK}


def pal_field(ax, near):
    """Per-pixel palette: ax (h, w) sun-axis proximity 0..1, near (scalar or (h, w)) 0..1."""
    A, B, Cn = _P(P_OFF), _P(P_SUN), _P(P_NEAR)
    a = ax[..., None]
    n = np.asarray(near, np.float32)
    n = n[..., None] if n.ndim else n
    out = {}
    for k in PK:
        c = A[k] * (1 - a) + B[k] * a
        out[k] = (c * (1 - n) + (Cn[k] * (0.75 + 0.25 * a) + B[k] * 0.25 * a) * n).astype(np.float32)
    return out


def haze_amount(dy):
    """Aerial-perspective mix toward the horizon haze (dy = fraction of H below the horizon)."""
    return np.clip(0.95 * np.exp(-np.maximum(dy, 0) / 0.05) + 0.03, 0, 0.96)


def haze_col(X, W, sx):
    p = np.exp(-((X - sx) / (0.33 * W)) ** 2)[..., None]
    return (HAZE_OFF * (1 - p) + HAZE_SUN * p).astype(np.float32)


# ---------------------------------------------------------------------------------------- floor
def floor(W, H, hy, sun, seed=3):
    """The hazy lower deck seen through the gaps: pale gold at the horizon -> soft lavender -> calm
    blue-violet near the camera, warmer on the sun axis, with a soft, perspective-mapped volumetric
    texture (big soft lumps near the camera, fine far away) - never a flat slab or a dark void."""
    sx = sun[0]
    xs, ys = C.grid(W, H)
    dy = np.clip((ys - hy) / H, 0, None)
    col = N.floor_color(dy, xs, W, sx)
    # perspective texture: world coords (x / depth, 1 / depth)
    n0 = C.fbm(512, 512, 5, 4, seed=seed).astype(np.float32)
    n0 = cv2.GaussianBlur(n0, (0, 0), 3)
    dd = np.maximum(dy, 0.004)
    wx = ((xs - sx) / (dd * H)) * 22.0
    wz = (0.12 / dd) * 60.0
    def tri(v):          # reflected (continuous) coordinates: no wrap seam
        v = np.mod(v + 7.0, 1020.0)
        return np.where(v > 510.0, 1020.0 - v, v).astype(np.float32)
    tex = cv2.remap(n0, tri(wx), tri(wz), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) - 0.5
    amp = _ss(0.015, 0.12, dy) * 0.16
    ax = np.exp(-((xs - sx) / (0.3 * W)) ** 2)
    lump = np.clip(tex * 2.2, -1, 1)
    lit = np.array([1.0, 0.86, 0.78], np.float32)
    col = col * (1 + (amp * lump * 0.6)[..., None]) + (amp * np.clip(lump, 0, 1) * (0.4 + 0.6 * ax))[..., None] * lit * 0.5
    return col.astype(np.float32)


# ---------------------------------------------------------------------------------------- painting
LV_BIG = ((0.16, 0.36, 0.9, 1.3, 2.8, 0.05, 0.42), (0.06, 0.13, 0.85, 1.1, 2.6), (0.026, 0.05, 0.65, 1.2, 3.0),
          (0.013, 0.022, 0.2, 1.6, 4.0))
LV_MID = ((0.16, 0.36, 0.9, 1.3, 2.8, 0.05, 0.42), (0.06, 0.13, 0.8, 1.1, 2.6), (0.028, 0.05, 0.4, 1.4, 3.4))
LV_FAR = ((0.2, 0.42, 0.9, 1.2, 2.6, 0.05, 0.4), (0.07, 0.14, 0.6, 1.3, 3.0))


def _sun_dir(X, Y, sun, W, H, lean=0.9, up=1.0):
    """Per-pixel unit screen direction toward the light: toward the sun axis and up."""
    ux = np.clip((sun[0] - X) / (0.55 * W), -1, 1) * lean
    uy = -up * np.ones_like(ux)
    ln = np.sqrt(ux * ux + uy * uy) + 1e-6
    return (ux / ln).astype(np.float32), (uy / ln).astype(np.float32)


def _sample(m, xs, ys, dx, dy, D):
    return cv2.remap(m, (xs + dx * D).astype(np.float32), (ys + dy * D).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _rot(img, ang, center, out_wh, inverse=False):
    M = cv2.getRotationMatrix2D(center, ang, 1.0)
    if inverse:
        M = cv2.invertAffineTransform(M)
    return M


def terminator(m, L, size, rng, tau, noise_seed=0, scal=(0.07, 0.22), scal_k=0.55, soft_frac=0.3,
               u=1.0, mid_len=0.55):
    """A painted light/shadow split for one mass (mask m, local px) lit from screen direction L (unit, y
    down). Rotate so the light comes from straight above; per column, the terminator sits tau*size below
    the (smoothed) sunward silhouette and is made of lit heads bulging AWAY from the light (scallops of
    varied size) - an independent line, not an offset of the outline. Returns (lit, mid) in m's frame:
    lit = crisp (a few softer passages), mid = soft band under the terminator."""
    h, w = m.shape
    ang = math.degrees(math.atan2(L[0], -L[1]))       # rotate the light direction to 'up'
    D = int(math.ceil(math.hypot(w, h))) + 4
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    M[0, 2] += (D - w) / 2.0
    M[1, 2] += (D - h) / 2.0
    R = cv2.warpAffine(m, M, (D, D), flags=cv2.INTER_LINEAR, borderValue=0)
    Rs = _blur(R, 0.04 * size)
    inside = Rs > 0.5
    has = inside.any(0)
    top = np.where(has, inside.argmax(0), D).astype(np.float32)
    # smooth the sunward silhouette (normalised convolution over the columns that have paint)
    sg = max(0.2 * size, 1.0)
    wgt = has.astype(np.float32)
    k = cv2.getGaussianKernel(int(6 * sg) | 1, sg).ravel().astype(np.float32)
    num = np.convolve(np.where(has, top, 0) * wgt, k, 'same')
    den = np.convolve(wgt, k, 'same')
    top_s = num / np.maximum(den, 1e-4)
    # stay close to the local outline where it dips (smooth, so no per-column streaks)
    lim = np.convolve(np.where(has, top + 0.08 * size, top_s.max()), cv2.getGaussianKernel(int(0.3 * size) | 1, max(0.05 * size, 1.0)).ravel().astype(np.float32), 'same')
    top_s = np.minimum(top_s, np.where(den > 0.3, lim, top_s))
    xs = np.arange(D, dtype=np.float32)
    nz = np.interp(xs, np.linspace(0, D, 8), rng.uniform(-1, 1, 8))
    yt = top_s + size * tau * (1 + 0.45 * nz)
    # scallops: lit heads bulging down into the shadow, varied sizes
    x = -rng.uniform(0, scal[1]) * size
    bul = np.zeros(D, np.float32)
    while x < D:
        r = size * (scal[0] + (scal[1] - scal[0]) * rng.random() ** 1.5)
        d = np.clip(1 - ((xs - x - r) / r) ** 2, 0, None)
        bul = np.maximum(bul, scal_k * r * np.sqrt(d))
        x += 2 * r * rng.uniform(0.75, 1.05)
    yt0 = yt.copy()
    yt = yt + bul
    Y = np.arange(D, dtype=np.float32)[:, None]
    # crisp edge with a few softer passages
    sp = np.interp(xs, np.linspace(0, D, 6), rng.uniform(0, 1, 6))
    e = 0.8 * u + soft_frac * size * 0.2 * _ss(0.65, 1.0, sp)
    lit = 1 - _ss(yt[None, :] - e[None, :], yt[None, :] + e[None, :], Y)
    ytm = yt0 + scal_k * size * 0.5 * (scal[0] + scal[1]) * 0.7       # smooth line (no scallop kinks)
    mid = 1 - _ss(ytm[None, :] - 0.05 * size, ytm[None, :] + mid_len * size, Y)
    tp = np.clip((Y - top_s[None, :]) / np.maximum(ytm - top_s, 1.0)[None, :], 0, 1.5)
    Mi = cv2.invertAffineTransform(M)
    tp = cv2.warpAffine(tp.astype(np.float32), Mi, (w, h), flags=cv2.INTER_LINEAR, borderValue=1)
    lit = cv2.warpAffine(lit.astype(np.float32), Mi, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mid = cv2.warpAffine(mid.astype(np.float32), Mi, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    return lit, mid, tp


def paint(cv, m, x0, y0, size, base_y, hgt, sun, W, H, hy, rng, near=None, pal=None, lit_k=1.0, tear=1.0,
          rim=1.0, rim_px=None, haze=None, form_blur=0.035, up=1.0, lean=0.9, dissolve=None, mid_k=1.0,
          term_soft=0.04, lit_d=0.1, lit_amt=1.0):
    """Paint one mass (mask m, local box at canvas px (x0, y0)) as flat value planes into canvas cv.
    base_y/hgt in frame px (base line and height of the mass) drive the torn feathered base."""
    h, w = m.shape
    if h < 2 or w < 2 or m.max() < 0.05:
        return
    u = cv.u
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    X, Y = xs + x0 + cv.ox, ys + y0 + cv.oy
    dy_f = (Y - hy) / H
    dxs, dys = _sun_dir(X, Y, sun, W, H, lean, up)
    ax = np.exp(-((X - sun[0]) / (0.27 * W)) ** 2).astype(np.float32)
    if near is None:
        near = _ss(0.06, 0.5, dy_f)
    P = pal_field(ax, near) if pal is None else {k: np.broadcast_to(np.asarray(pal[k], np.float32), (h, w, 3)) for k in PK}
    # cut the canvas noise slices
    r = cv.sl(x0, y0, w, h)
    if r is None:
        return
    if not hasattr(cv, 'n_hs'):
        # horizontally streaked noise for torn, wind-combed bases (cells ~6x wider than tall)
        wq, hq = max(int(cv.w / 18), 4), max(int(cv.h / 3), 4)
        nn = C.fbm(wq, hq, max(cv.w / (260 * u), 2), 4, seed=cv.seed + 17, aspect=True).astype(np.float32)
        cv.n_hs = cv2.resize(nn, (cv.w, cv.h), interpolation=cv2.INTER_CUBIC) - 0.5
    nz = {}
    for k in ('n_lo', 'n_md', 'n_st', 'n_fn', 'n_hs'):
        a = np.zeros((h, w), np.float32)
        (py, px), (ly, lx) = r
        a[ly, lx] = getattr(cv, k)[py, px]
        nz[k] = a
    form = _blur(m, form_blur * size)
    # light plane: an independent scalloped terminator under the sunward silhouette
    cx_ = float((X * m).sum() / max(m.sum(), 1e-3))
    Lx = float(np.clip((sun[0] - cx_) / (0.55 * W), -1, 1)) * lean
    Ln = math.hypot(Lx, up)
    tau = lit_d * lit_k * (0.7 + 0.6 * float(np.exp(-((cx_ - sun[0]) / (0.27 * W)) ** 2)))
    lit, midt, tp = terminator(m, (Lx / Ln, -up / Ln), size, rng, tau, u=u, mid_len=0.55 * mid_k)
    lit = lit * lit_amt
    s0 = _sample(form, xs, ys, dxs, dys, 0.05 * size)
    hot = _blur(1 - _ss(0.3, 0.7, s0), 0.015 * size) * (0.3 + 0.7 * ax)
    # big-form values: upper body mid (sky + scattered light), lower body cool shadow, deepest just
    # above the base, a little reflected light at the base; all soft painted gradients
    vpos = np.clip((Y - (base_y - hgt)) / max(hgt, 1.0), -0.2, 1.8)
    vn = vpos + 0.18 * nz['n_lo'] + 0.08 * nz['n_md']
    midw = np.maximum(midt * 0.8, (1 - _ss(0.0, 0.5 * mid_k, vn)) * 0.45)
    col = _mix(P['shade'], P['deep'], (_ss(0.45, 0.85, vn) * (1 - _ss(0.95, 1.3, vn)))[..., None])
    col = _mix(col, P['refl'], (0.5 * _ss(0.95, 1.35, vn))[..., None])
    col = col * (1 + 0.03 * nz['n_md'][..., None])
    col = _mix(col, P['mid'], midw[..., None])
    # lit plane: hot cream toward the sunward crest -> warm lit -> rose lit_lo along the terminator
    tq = tp + 0.15 * nz['n_md']
    litc = _mix(P['lit'], P['lit_lo'], (0.7 * _ss(0.75, 1.05, tq))[..., None])
    litc = _mix(litc, P['hot'], (np.maximum(hot * 0.7, 1 - _ss(0.05, 0.3, tq)) * (0.3 + 0.7 * ax))[..., None])
    col = _mix(col, litc, lit[..., None])
    # torn, streaky, feathered base that dissolves into the deck below
    tb = (Y - base_y) / max(0.45 * hgt, 1.0) + tear * (1.3 * nz['n_hs'] + 0.2 * nz['n_md'])
    fade = 1 - _ss(-0.35, 0.85, tb)
    a = m * fade
    if dissolve is not None:
        col = _mix(col, dissolve, (0.55 * _ss(-0.2, 0.9, tb))[..., None])
    # rim: continuous line only where the actual silhouette faces the light
    if rim > 0:
        rp = (rim_px if rim_px is not None else 2.6) * u * (0.55 + 0.45 * ax)
        e1 = _sample(m, xs, ys, dxs, dys, rp)
        core = np.clip(m * (1 - e1) * 1.25, 0, 1)
        k = np.clip(core * rim * (0.2 + 0.8 * ax) * (0.3 + 0.7 * lit), 0, 1)
        k *= 0.6 + 0.6 * np.clip(nz['n_md'] + 0.5, 0, 1)
        rc = RIM_OFF * (1 - ax[..., None]) + RIM_SUN * ax[..., None]
        col = _mix(col, rc, np.clip(k, 0, 1)[..., None])
    # aerial perspective
    hz = haze_amount(dy_f) if haze is None else haze
    hz = np.broadcast_to(np.asarray(hz, np.float32), (h, w))
    col = _mix(col, haze_col(X, W, sun[0]), hz[..., None])
    cv.over(col.astype(np.float32), np.clip(a, 0, 1).astype(np.float32), x0, y0)


# ---------------------------------------------------------------------------------------- the sea
def rows_layout(H, hy, dy0=0.007, dy1=0.66, ratio=1.24):
    out = []
    dy = dy0
    while dy < dy1:
        out.append(dy)
        dy *= ratio
    return out


def _masses(rng, W, H, y, w0, asp, margin):
    """Masses of one row: widths lognormal around w0, heavy overlap, a few low valleys."""
    ms = []
    x = -margin * W - rng.uniform(0, 0.5) * w0
    while x < W * (1 + margin) + w0 * 0.5:
        w = w0 * math.exp(rng.normal(0.0, 0.5))
        w = float(np.clip(w, 0.35 * w0, 1.7 * w0))
        hh = w * asp * rng.uniform(0.5, 0.95) * (w / w0) ** -0.15
        by = y + rng.uniform(-0.12, 0.3) * hh
        if rng.random() < 0.1:
            by += 0.35 * hh           # a sunken mass -> a valley in the row
        ms.append([(x + w * 0.5, by, w, hh, rng.uniform(0.65, 1.0))])
        x += w * rng.uniform(0.4, 0.8)
    ms.sort(key=lambda g: g[0][1])
    out = []
    for g in ms:
        cx, by, w, hh, la0 = g[0]
        out.append(g[0])
        if hh > 60 * W / 1920.0:
            # lower heads heaped in front of the body: crisp overlapping-mass edges inside it
            for j in range(int(rng.integers(1, 4))):
                cw = w * rng.uniform(0.3, 0.6)
                ch = hh * rng.uniform(0.5, 0.8)
                out.append((cx + rng.uniform(-0.4, 0.4) * w, by + rng.uniform(0.1, 0.35) * hh, cw, ch,
                            rng.uniform(0.45, 0.9)))
    return out


def sea_plates(W, H, hy, sun, bands, seed=7, margin=0.14):
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    dys = rows_layout(H, hy)
    plates = []
    for k, (a0, a1) in enumerate(bands):
        rs = [d for d in dys if a0 <= d < (a1 * 1.7 if k < len(bands) - 1 else a1)]
        if not rs:
            plates.append(None)
            continue
        top = hy + rs[0] * H - 1.4 * (1.2 * rs[0] * H) * 0.6
        ox, oy = -margin * W, max(hy - 0.03 * H, top - 0.04 * H)
        cv = N.Canvas(W * (1 + 2 * margin), H * 1.3 - oy, ox, oy, u, seed + 31 * k)
        for dy in rs:
            y = hy + dy * H
            w0 = 0.6 * dy * H
            near = float(_ss(0.06, 0.5, dy))
            asp = 0.36 + 0.3 * near
            r2 = np.random.default_rng(int(rng.integers(1 << 30)))
            if w0 < 40 * u:
                # far rows: one continuous billow strip per row
                poly = K.billow_strip(-margin * W, W * (1 + margin), y, w0, w0 * asp, r2, depth=w0 * asp * 1.2, var=1.2)
                poly = poly - np.array([cv.ox, cv.oy], np.float32)
                size = w0 * asp * 1.4
                m, mx0, my0 = K.cauliflower(poly, r2, size, LV_FAR, down_cut=0.2, side_scale=0.6, concave=0.6)
                paint(cv, m, mx0, my0, size, y, w0 * asp, sun, W, H, hy, r2, rim=0.0 if dy < 0.03 else 0.5,
                      rim_px=1.2, tear=0.6)
                continue
            for (cx, by, w, hh, la) in _masses(r2, W, H, y, w0, asp, margin):
                env = K.envelope(cx - cv.ox, by - cv.oy, w, hh, r2, power=r2.uniform(2.0, 2.8), lump=0.08,
                                 base_round=0.35, skew=r2.uniform(-0.5, 0.5), top_flat=r2.uniform(0.0, 0.4),
                                 lean=r2.uniform(-0.1, 0.1))
                size = max(hh, 0.45 * w)
                lv = LV_BIG if size > 140 * u else LV_MID
                sd = (float(np.clip((sun[0] - cx) / (0.55 * W), -1, 1)) * 0.9, -1.0)
                m, mx0, my0 = K.cauliflower(env, r2, size, lv, down_cut=0.25, side_scale=0.55, concave=0.65,
                                            clump=0.55, sun_dir=sd, sun_bias=0.3)
                rimk = float(_ss(0.025, 0.08, dy))
                axm = math.exp(-((cx - sun[0]) / (0.36 * W)) ** 2)
                la2 = la * (0.5 + 0.5 * axm)
                paint(cv, m, mx0, my0, size, by, hh, sun, W, H, hy, r2, rim=rimk * la,
                      rim_px=2.0 + 2.0 * near, tear=1.0, lit_amt=la2, mid_k=1.0 - 0.45 * near, lit_d=0.04 + 0.36 * r2.random() ** 1.6)
        plates.append(cv.plate())
    return plates


# ---------------------------------------------------------------------------------------- towers
TOWER_LV = ((0.1, 0.3, 0.9, 1.2, 2.8, 0.05, 0.45), (0.045, 0.1, 0.8, 1.1, 2.6), (0.02, 0.04, 0.55, 1.3, 3.2),
            (0.01, 0.017, 0.2, 1.8, 4.0))


def tower(W, H, sun, cx, base_y, Hc, seed, side=1.0, width=1.0, haze=0.0, margin=0.1):
    """Towering cumulus painted as ONE coherent form: the union of 7 overlapping cauliflower masses is lit
    as a whole (broad warm-white sunward face, scalloped terminator hugging the big heads, rose-lavender
    mid, cool lavender shadow side deepening low down); front masses add crisp overlapping-mass edges (a
    lit crescent where they bulge out of the shadow, a thin cast-shade edge where they overlap lit paint);
    florets only on the edge; a thin warm rim only on the sunward outer silhouette; hazy torn base."""
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    wd = 0.95 * Hc * width
    x0 = int(cx - 0.95 * wd - margin * W)
    y0 = int(base_y - 1.25 * Hc)
    cw = int(1.9 * wd + 2 * margin * W)
    ch = int(1.25 * Hc + 0.3 * Hc)
    cv = N.Canvas(cw, ch, x0, y0, u, seed)
    lx, ly = cx - x0, base_y - y0
    P = _P(TW)
    masses = [
        (-0.1, 0.0, 0.95, 0.66),     # main body (back)
        (-0.14, 0.42, 0.62, 0.58),   # upper head, set back from the sun
        (0.08, 0.62, 0.46, 0.4),     # crown
        (0.26, 0.3, 0.5, 0.4),       # sunward shoulder (in front of the body)
        (-0.3, 0.08, 0.5, 0.42),     # shadow-side lower mass (front)
        (0.2, 0.02, 0.62, 0.32),     # sunward lower mass (front)
        (-0.04, -0.06, 0.95, 0.2),   # base billows
    ]
    Ms = []
    U = np.zeros((ch, cw), np.float32)
    for i, (rx, rb, rw, rh) in enumerate(masses):
        mcx = lx + side * rx * wd + rng.uniform(-0.03, 0.03) * wd
        mby = ly - rb * Hc
        mw, mh = rw * wd * rng.uniform(0.92, 1.08), rh * Hc * rng.uniform(0.92, 1.08)
        env = K.envelope(mcx, mby, mw, mh, rng, lean=0.1 * side * rng.uniform(-1, 1), power=2.1, lump=0.1,
                         base_round=0.12, skew=side * rng.uniform(-0.2, 0.4))
        size = max(mh, 0.6 * mw)
        m, mx0, my0 = K.cauliflower(env, rng, size, TOWER_LV, down_cut=0.3, side_scale=0.8,
                                    sun_dir=(0.9 * side, -0.45), sun_bias=0.25, concave=0.7, clump=0.5)
        full = np.zeros((ch, cw), np.float32)
        r = cv.sl(mx0, my0, m.shape[1], m.shape[0])
        if r is None:
            continue
        (py, px), (ly_, lx_) = r
        full[py, px] = m[ly_, lx_]
        Ms.append((full, size, i))
        U = np.maximum(U, full)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    L = np.array([0.9 * side, -0.42], np.float32)
    L /= np.linalg.norm(L)
    dx, dy = np.full_like(xx, L[0]), np.full_like(xx, L[1])
    nlo, nmd, nfn = cv.n_lo, cv.n_md, cv.n_fn
    # ---- one coherent form: broad lit face, independent scalloped terminator
    lit, mid, tp = terminator(U, (float(L[0]), float(L[1])), wd, rng, 0.3, scal=(0.04, 0.12), scal_k=0.7,
                              soft_frac=0.25, u=u, mid_len=0.3)
    form = _blur(U, 0.02 * wd)
    s0 = _sample(form, xx, yy, dx, dy, 0.03 * wd)
    hot = _blur(1 - _ss(0.3, 0.7, s0), 0.012 * wd)
    # ---- front masses: crisp crescents out of the shadow, a slight value break shade-on-shade
    flit = np.zeros_like(U)
    fin = np.zeros_like(U)
    edge_sh = np.zeros_like(U)
    for (full, size, i) in Ms:
        if i < 3:
            continue
        li, _, _ = terminator(full, (float(L[0]), float(L[1])), size, rng, 0.2, scal=(0.06, 0.16), scal_k=0.6,
                              soft_frac=0.1, u=u)
        own = _ss(0.4, 0.6, full)
        flit = flit * (1 - own) + li * own
        fin = np.maximum(fin, own)
    # front lobes: inside the union's light plane their own away-side turns a step down (crisp shade
    # crescents = overlapping-mass edges on the lit face); out of the union's shadow they bulge into light
    lit = lit * (1 - fin * 0.5 * (1 - flit)) + (1 - lit) * 0.4 * flit
    lit = _blur(lit, 0.7 * u)
    vpos = np.clip((ly - yy) / Hc, 0, 1.3)                 # 0 at the base .. 1 at the top
    away = np.clip(-side * (xx - lx) / (0.6 * wd), -1, 1)   # + on the shadow side
    col = _mix(P['shade'], P['deep'], (_ss(0.0, 0.6, away + 0.3 * (0.6 - vpos)) * 0.8)[..., None])
    col = _mix(col, P['refl'], (0.45 * _ss(0.25, 0.0, vpos) * (1 - 0.5 * _ss(0, 1, away)))[..., None])
    col = col * (1 + 0.025 * nmd[..., None] + 0.015 * nfn[..., None])
    col = _mix(col, P['mid'], (mid * 0.85)[..., None])
    col = _mix(col, P['refl'], (0.18 * fin)[..., None])
    tq = tp + 0.15 * nmd
    litc = _mix(P['lit'], P['lit_lo'], (0.8 * _ss(0.78, 1.05, tq))[..., None])
    litc = _mix(litc, P['hot'], np.maximum(0.8 * hot, 1 - _ss(0.1, 0.45, tq))[..., None])
    col = _mix(col, litc, lit[..., None])
    # haze toward the base, torn feathered base
    hc = _mix(HAZE_OFF, HAZE_SUN, 0.35)
    hk = (_ss(0.3, -0.03, vpos) * 0.4 + haze)[..., None]
    col = _mix(col, hc, np.clip(hk, 0, 1))
    tb = (yy - ly) / (0.08 * Hc) + 1.2 * cv.n_st + 0.4 * nfn
    A = U * (1 - _ss(-0.6, 0.9, tb))
    # thin warm rim only on the sunward outer silhouette
    rp = 2.0 * u
    e1 = _sample(A, xx, yy, dx, dy, rp)
    k = np.clip(A * (1 - e1) * 1.5, 0, 1) * (0.7 + 0.8 * np.clip(nmd + 0.5, 0, 1)) * (1 - _ss(0.3, 0.0, vpos))
    rc = np.asarray((1.28, 1.18, 1.0), np.float32)          # warm-white silver lining, not an orange stroke
    k = k * (0.35 + 0.65 * _ss(-0.3, 0.3, nlo))             # breaks up along the edge
    col = _mix(col, rc, np.clip(0.85 * k, 0, 1)[..., None])
    cv.over(col.astype(np.float32), np.clip(A, 0, 1).astype(np.float32), 0, 0)
    return cv.plate()
