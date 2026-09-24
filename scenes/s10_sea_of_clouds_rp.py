"""s10 sea-of-clouds ROW PLATES (round 5).

Each row of the cloud sea is a painted plate standing at a world depth z (anime multiplane).  Geometry
comes from the shared dome engine (lib.clouds: cauliflower lobe hierarchy, crisp z-buffered overlaps,
blended form normals) with the child lobes grown on the crowns / silhouettes (rows.crown_lobes) so the
camera-facing fronts stay large coherent billows.  The painting is our own BACKLIT painter:

  * the sun is behind the sea: faces toward the camera are cool indigo / blue-violet shadow, graded
    lighter toward the crowns (sky fill) and deep toward the valley;
  * warm cream / peach light only where a surface turns up / toward the sun (the crowns), strongest near
    the sun's azimuth; a narrow pink band at the terminator;
  * a thin (1.5-3 px) silver-gold rim along every sun-facing silhouette + a forward-scatter glow that
    creeps a few px into the cloud (translucent edges), strongest near the sun;
  * a soft deck band below the tops that falls off smoothly into the indigo valley.
All widths are in screen pixels (plates are shown at 0.6-1x their resolution).
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from lib import core as C, clouds as K
import s10_sea_of_clouds_rows as R


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _rgb(c):
    if isinstance(c, str):
        h = c.lstrip('#')
        return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    return np.asarray(c, np.float32)


def _fblur(img, sigma):
    if sigma <= 0.3:
        return img
    if sigma <= 3.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    s = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 2.55)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


def paint_backlit(A, P, sun_p, px_scale, prox_w, seed=0, rim=1.0, glow=1.0, lit_k=1.0, lit_bias=0.0):
    """A: attribute buffer from K.render. P: palette dict (rgb arrays). sun_p: sun in plate px.
    px_scale: plate px per screen px at 1080p-equivalent (widths). prox_w: sun proximity width (plate px).
    Returns straight-alpha RGBA."""
    H, W = A.shape[:2]
    a = np.clip(A[..., 0], 0, 1)
    l, ny, v = A[..., 1], A[..., 2], A[..., 3]
    depth, lform, skirt = A[..., 6], A[..., 10], A[..., 9]
    ins = (a > 1e-3).astype(np.float32)

    def wb(x, s):
        return _fblur(x * ins, s) / (_fblur(ins, s) + 1e-4)
    # crevice occlusion just behind a lobe edge (crisp recess line + broader)
    u = 1.0 * px_scale
    d0 = wb(depth, 3.0 * u)
    d2 = wb(depth, 16.0 * u)
    occ_line = _ss(0.25, 0.6, (d0 - depth) / (5.0 * u))
    occ = np.clip(np.maximum(np.clip((d2 - depth) / (30.0 * u), 0, 1), occ_line * 0.7), 0, 1)
    # large-scale light keeps the shadow side one coherent mass; lobes deviate only a little
    lbig = wb(l, 10.0 * u) * 0.55 + lform * 0.45
    le = np.clip(l, lbig - 0.28, lbig + 0.22) - 0.25 * occ + lit_bias
    # screen direction / proximity to the sun
    xs, ys = C.grid(W, H)
    dxs, dys = sun_p[0] - xs, sun_p[1] - ys
    dl = np.sqrt(dxs * dxs + dys * dys) + 1e-3
    prox = np.exp(-((xs - sun_p[0]) / prox_w) ** 2)
    up = np.clip(-ny, 0, 1)                                   # faces up (y down in the plate)
    # ---- shadow body: deep valley -> blue face -> lighter lavender upward-facing planes
    vv = v[..., None]
    shd = P['deep'] + (P['face'] - P['deep']) * np.clip(0.35 + 0.9 * vv, 0, 1)
    shd = shd + (P['fill'] - shd) * (_ss(0.25, 0.9, up) * 0.32)[..., None]
    shd = shd + (P['deep'] - shd) * (occ * 0.45)[..., None]
    # ---- light: crowns only (up-facing & toward the sun), gold near the sun, peach-pink away
    lit = _ss(-0.02, 0.14, le) * _ss(0.05, 0.45, up + 0.15 * prox) * lit_k
    band = np.exp(-((le + 0.04) / 0.07) ** 2) * (1 - lit) * _ss(0.0, 0.3, up)
    warm = P['pink'] + (P['warm'] - P['pink']) * np.clip(0.1 + 1.0 * prox, 0, 1)[..., None]
    col = shd + (P['pink'] - shd) * (band * 0.35 * (0.4 + 0.6 * prox))[..., None]
    col = col + (warm - col) * (lit * (0.22 + 0.78 * prox))[..., None]
    col = col + (P['hot'] - col) * (_ss(0.25, 0.5, le) * lit * prox * 0.6)[..., None]
    # ---- silhouette rim + forward-scatter glow on sun-facing edges
    m8 = (a > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(m8, cv2.DIST_L2, 5).astype(np.float32)
    cg = _fblur(a, 2.0 * u)
    gx = -cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    gy = -cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    face = np.clip((gx * dxs + gy * dys) / (gl * dl), 0, 1)
    face = _ss(0.1, 0.75, _fblur(face * (gl > 1e-4), 1.5 * u))
    pr2 = 0.3 + 0.7 * prox
    wr = 1.4 * u
    rim_line = np.exp(-din / wr) * face * pr2 * rim
    # inner lobe edges (dome overlaps): crisp rim on the upper edge of a lobe over the one behind
    ov = np.clip((d0 - depth) / (4.0 * u), 0, 1)
    lobe_edge = np.exp(-np.clip(-(depth - d0) / (2.0 * u), 0, 50)) * 0.0
    glow_in = np.exp(-din / (7.0 * u)) * face * prox * 0.5 * glow
    col = col + (P['glow'] - col) * np.clip(glow_in, 0, 0.8)[..., None]
    col = col + (P['rim'] - col) * np.clip(rim_line, 0, 1)[..., None]
    # ---- torn skirt wisps at the flat bases
    st = K.streaks(W, H, seed + 31, xcells=max(3.0, W / 500), ycells=H / 7.0)
    body = a - skirt
    wisp = skirt * _ss(0.55, 0.75, st + skirt * 0.5 - 0.15)
    alpha = np.clip(body + wisp, 0, 1)
    del ov, lobe_edge
    return np.dstack([col, alpha]).astype(np.float32)


def build_row_plate(seed, z_ref, ppu, x0u, x1u, hmax, f, W, hy, camx, camy, sun_scr, P, haze_fn, fog,
                    big=0.18, strato=0.6, wscale=1.0, gap=(0.3, 0.55), detail=1.0, sun_z=-0.3, backlit=0.6,
                    rim=1.0, glow=1.0, lit_k=1.0, deck_col=None, under=0.7, ss=2, mid_col=None, fog_mid=0.0, deck=True):
    """One row plate. ppu: plate px per world unit. x0u..x1u world x extent. hmax: tallest cloud (world).
    sun_scr: sun (x, y) on screen; mapped into the plate through depth z_ref. haze_fn(xs_screen) ->
    (n, 3) haze colour; fog 0..1 aerial perspective (applied to the whole row).
    Returns dict(rgba uint16, ppu, x0u, top, hmax)."""
    rng = np.random.default_rng(seed)
    top = max(0.35, 0.35 * hmax) * ppu
    base_py = top + hmax * ppu
    pw = int(math.ceil((x1u - x0u) * ppu))
    ph = int(math.ceil(base_py + under * ppu))
    specs = []
    for (cxu, bo, wu, hu, kind, flat) in R.row_specs(rng, x0u, x1u, hmax, big, strato, wscale, gap):
        specs.append(((cxu - x0u) * ppu, base_py + bo * ppu, wu * ppu, hu * ppu, 2.0, kind, flat))
    # sun on screen -> plate coordinates at z_ref
    Xs = camx + (sun_scr[0] - W / 2) * z_ref / f
    hs = camy - (sun_scr[1] - hy) * z_ref / f
    sun_p = ((Xs - x0u) * ppu, top + (hmax - hs) * ppu)
    a_scr = f / (z_ref * ppu)                    # screen px per plate px
    px_scale = (W / 1920.0) / a_scr              # plate px per 1080p screen px
    phc = min(ph, int(math.ceil(base_py + 0.3 * ppu)))
    B = K.Builder()
    with R.crown_lobes():
        for i, (cx, by, wd, ht, d, k, fl) in enumerate(specs):
            crng = np.random.default_rng(seed * 1000 + i)
            L, back = _light_for(cx, by - ht * 0.5, sun_p, sun_z, backlit, 0.5 * W / a_scr)
            K.gen_cloud(B, crng, (cx, by, wd, ht, d, k, fl), L, 0.0, back, detail=detail, z0=-d * pw * 0.0)
    if B.S:
        Aat = K.render(B, pw, phc, ss=ss, fillet=0.006 * 1920.0 / pw * px_scale)
        clc = paint_backlit(Aat, P, sun_p, px_scale, 0.36 * W / a_scr, seed=seed, rim=rim, glow=glow, lit_k=lit_k)
    else:
        clc = np.zeros((phc, pw, 4), np.float32)
    cl = np.zeros((ph, pw, 4), np.float32)
    cl[:phc] = clc
    # deck band under the tops: soft torn upper edge, falls off into the indigo valley
    ys = np.arange(ph, dtype=np.float32)[:, None]
    xs = np.arange(pw, dtype=np.float32)[None, :]
    edge = base_py - 0.2 * hmax * ppu + 0.06 * ppu * np.sin(xs / (0.9 * ppu) + seed) + \
        0.04 * ppu * np.sin(xs / (0.31 * ppu) + 2 * seed)
    da = _ss(edge - 0.08 * ppu, edge + 0.2 * ppu, ys) * (1.0 if deck else 0.0)
    g = _ss(base_py - 0.3 * hmax * ppu, base_py + 0.45 * ppu, ys)[..., None]
    dtop = P['face'] * 0.85 + P['deep'] * 0.15 if deck_col is None else deck_col
    dcol = dtop + (P['deep'] - dtop) * g
    dcol = np.broadcast_to(dcol, (ph, pw, 3))
    ca = cl[..., 3:4]
    alpha = ca + da[..., None] * (1 - ca)
    rgb = (cl[..., :3] * ca + dcol * da[..., None] * (1 - ca)) / np.maximum(alpha, 1e-5)
    # aerial perspective toward the haze colour seen on screen
    xs_scr = W / 2 + (f / z_ref) * (x0u + (np.arange(pw) + 0.5) / ppu - camx)
    hz = haze_fn(xs_scr)[None, :, :]
    if fog_mid > 0 and mid_col is not None:
        rgb = rgb + (np.asarray(mid_col, np.float32) - rgb) * fog_mid
    if fog > 0:
        rgb = rgb + (hz - rgb) * fog
    rgba = np.dstack([rgb, alpha[..., 0]])
    q = np.clip(rgba, 0, 2.0) * 32767.0
    return dict(rgba=np.ascontiguousarray(q.astype(np.uint16)), ppu=ppu, x0u=x0u, top=top, hmax=hmax)


def _light_for(cx, cy, sun_p, sun_z, backlit, radius):
    dx, dy = sun_p[0] - cx, sun_p[1] - cy
    dist = math.hypot(dx, dy) + 1e-3
    dx, dy = dx / dist, dy / dist
    lz = sun_z - backlit * math.exp(-dist / radius)
    lz = float(np.clip(lz, -0.95, 0.95))
    c = math.sqrt(1 - lz * lz)
    return (dx * c, dy * c, lz), float(np.clip(-lz * 1.6, 0, 1))


def place(Rw, z, f, W, hy, camx, camy):
    """Affine plate->screen for a row at depth z. Returns (a, bx, by)."""
    a = f / (z * Rw['ppu'])
    bx = W / 2 + (f / z) * (Rw['x0u'] - camx)
    by = hy + (f / z) * (camy - Rw['hmax']) - a * Rw['top']
    return a, bx, by


def draw(img, occ, Rw, z, f, W, H, hy, camx, camy, gain=1.0):
    """Composite the row into img in place (rows below by only). Also max its alpha into occ."""
    a, bx, by = place(Rw, z, f, W, hy, camx, camy)
    y0 = int(max(math.floor(by), 0))
    if y0 >= H:
        return
    y1 = int(min(math.ceil(by + a * Rw['rgba'].shape[0]) + 1, H))
    if y1 <= y0:
        return
    M = np.array([[a, 0, bx], [0, a, by - y0]], np.float32)
    t = cv2.warpAffine(Rw['rgba'], M, (W, y1 - y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    _over_u16(img, occ, t, y0, np.float32(gain))


@njit(cache=True, fastmath=True, parallel=True)
def _over_u16(img, occ, t, y0, gain):
    h, W = t.shape[0], t.shape[1]
    k = np.float32(1.0 / 32767.0)
    for i in prange(h):
        r = y0 + i
        for c in range(W):
            a = t[i, c, 3] * k
            if a <= 0.0:
                continue
            if a > 1.0:
                a = 1.0
            for ch in range(3):
                img[r, c, ch] = img[r, c, ch] * (1.0 - a) + t[i, c, ch] * k * gain * a
            if a > occ[r, c]:
                occ[r, c] = a


@njit(cache=True, fastmath=True, parallel=True)
def over_col(img, col, a, ka, kc):
    """In place: img = over(img, col * kc, alpha = a * ka)."""
    H, W = a.shape
    for r in prange(H):
        for c in range(W):
            al = a[r, c] * ka
            if al <= 0.0:
                continue
            if al > 1.0:
                al = 1.0
            for k in range(3):
                img[r, c, k] = img[r, c, k] * (1.0 - al) + col[r, c, k] * kc * al


@njit(cache=True, fastmath=True, parallel=True)
def add_tint(img, m, tint):
    """In place: img += m * tint."""
    H, W = m.shape
    for r in prange(H):
        for c in range(W):
            v = m[r, c]
            if v != 0.0:
                for k in range(3):
                    img[r, c, k] += v * tint[k]
