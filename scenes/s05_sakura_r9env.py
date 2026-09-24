"""s05_sakura round 9: painted passes over the supersampled environment render (before it is split into plates).

* weather9: grime / soot streaks running down from every eave, parapet and sill (streak noise anchored to the top
  of each wall, fading downward), damp darker plinths, per-panel value / temperature variation of the wall
  cladding (concrete panel seams on the blocks) - applied to the far-bank town AND the land-side houses;
* ink9: crisp dark accent lines where surfaces meet, for the land-side houses too;
* far_haze: stronger aerial perspective on the distant apartment blocks (value compressed toward the sky haze);
* paint_hedge: the hedge repainted as clumped foliage - warm lit top plane, cool blue-green shadowed face,
  cel-banded leaf clumps (few big value shapes, crisp edges) instead of a procedural leaf-noise tile.
"""
import numpy as np
import cv2

import s05_sakura_tree2d as T2


def _vn(u, v, seed):
    u = np.ascontiguousarray(np.atleast_1d(u), dtype=np.float32)
    v = np.ascontiguousarray(np.atleast_1d(v), dtype=np.float32)
    return T2.vnoise_at(u[None], v[None], seed)[0]


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def weather9(out, mat, bid, face, uo, Yo, so, layer, Zo, B, amt=1.0):
    rgb = out[..., :3]
    typ = np.where(mat > 0, B[np.maximum(bid, 0), 6], -1)
    walls = (mat > 0) & ((typ == 0) | (typ == 5)) & (np.abs(face) != 2) & ((layer == 0) | (layer == 5))
    if not walls.any():
        return
    ys, xs = np.nonzero(walls)
    b = bid[ys, xs]
    Y = Yo[ys, xs]
    Z = Zo[ys, xs]
    fa = np.abs(face[ys, xs])
    hc = np.where(fa == 1, so[ys, xs], uo[ys, xs]).astype(np.float32)
    top = (B[b, 3] - Y).astype(np.float32)                 # metres below the top of the wall
    base = (Y - B[b, 2]).astype(np.float32)
    near = np.clip(1.0 - (Z - 20.0) / 220.0, 0, 1).astype(np.float32)
    sd = (B[b, 10] % 97).astype(np.float32)
    # streaks: 1-D noise across the wall (stretched vertically), soot under the eave / parapet
    st = _vn(hc * 3.2 + sd * 7.1, Y * 0.12 + sd, 71)
    st2 = _vn(hc * 9.0 + sd * 3.3, Y * 0.3, 73)
    streak = _ss(0.5, 0.75, 0.7 * st + 0.3 * st2)
    run = np.exp(-top / (0.8 + 1.6 * st))                  # streak length varies
    soot = np.exp(-top / 0.35) * 0.6
    # floor-line stains (every storey the slab edge drips a little)
    fy = np.mod(Y - B[b, 2] + 0.3, 2.8) / 2.8
    slab = streak * (1 - fy) ** 3 * 0.5
    g = np.clip(0.07 * streak * run + 0.3 * soot + 0.03 * slab, 0, 0.4)     # round 19: fewer stripes, deeper eave shadow
    # damp plinth
    g += 0.12 * (1 - _ss(0.2, 0.7, base))
    # panel variation: per-panel (1.8 m x storey) value shift + seams on tall blocks
    pi_ = np.floor(hc / 1.8)
    pj = np.floor((Y - B[b, 2]) / 2.8)
    h = np.sin(pi_ * 12.9898 + pj * 78.233 + sd * 3.7) * 43758.5453
    h = h - np.floor(h)
    pv = (h - 0.5) * 0.07
    tall = (B[b, 3] - B[b, 2]) > 8.5
    fpx = Z / 1500.0
    seam_h = (1 - _ss(0.0, 0.03 + fpx, np.abs(hc / 1.8 - np.round(hc / 1.8)) * 1.8)) * tall * 0.07
    mott = _vn(hc * 0.45 + sd, Y * 0.35, 79) - 0.5
    k = (g * amt * near)[:, None]
    c = rgb[ys, xs]
    c = c * (1 - k * np.array([0.95, 1.0, 0.8], np.float32))
    c = c * (1 + (pv + 0.08 * mott)[:, None] * near[:, None] * np.array([1.0, 0.95, 0.8], np.float32))
    c = c * (1 - (seam_h * near)[:, None])
    rgb[ys, xs] = c


def ink9(out, mat, bid, face, Zo, layer, ss, wscale, lay=5, zmax=120.0, strength=0.9):
    """dark accent lines where surfaces meet (land-side houses)"""
    rgb = out[..., :3]
    bmask = (layer == lay) & (mat > 0)
    if not bmask.any():
        return
    key = np.where(bmask, bid.astype(np.int64) * 16 + face.astype(np.int64) + 8, -1)
    Z = np.where(bmask, Zo, 1e5).astype(np.float32)
    e = np.zeros(mat.shape, np.float32)
    dxk = key[:, 1:] != key[:, :-1]
    dyk = key[1:, :] != key[:-1, :]
    zr = Z[:, 1:] < Z[:, :-1]
    zd = Z[1:, :] < Z[:-1, :]
    e[:, 1:] = np.maximum(e[:, 1:], (dxk & zr).astype(np.float32))
    e[:, :-1] = np.maximum(e[:, :-1], (dxk & ~zr).astype(np.float32))
    e[1:, :] = np.maximum(e[1:, :], (dyk & zd).astype(np.float32))
    e[:-1, :] = np.maximum(e[:-1, :], (dyk & ~zd).astype(np.float32))
    kk = max(1, int(round(0.8 * ss * wscale)))
    if kk > 1:
        e = cv2.dilate(e, np.ones((kk, kk), np.uint8))
    e = cv2.GaussianBlur(e, (0, 0), 0.5 * ss * wscale)
    near = np.clip(1.0 - (Z - 8.0) / (zmax - 8.0), 0, 1)
    e = e * bmask * near * strength
    acc = np.array([0.55, 0.52, 0.66], np.float32)
    rgb *= (1 - 0.6 * e[..., None]) + acc * (0.6 * e[..., None])


def far_haze(out, mat, layer, Zo, haze=(0.8, 0.87, 0.97), z0=90.0, z1=420.0, amt=0.45):
    """extra aerial perspective on distant blocks: value compressed toward the pale sky haze"""
    m = (layer == 0) & (mat > 0)
    k = (np.clip((Zo - z0) / (z1 - z0), 0, 1) ** 0.8 * amt * m).astype(np.float32)[..., None]
    rgb = out[..., :3]
    h = np.array(haze, np.float32)
    rgb[:] = rgb * (1 - k) + h * k


def paint_hedge(out, mat, bid, face, uo, Yo, so, B, ss, Zo=None, haze=(0.8, 0.87, 0.96), haze_k=260.0, haze_max=0.85):
    rgb = out[..., :3]
    typ = np.where(mat > 0, B[np.maximum(bid, 0), 6], -1)
    hm = (mat > 0) & (typ == 4)
    if not hm.any():
        return
    ys, xs = np.nonzero(hm)
    old = rgb[ys, xs].copy()
    # big-scale light already in the render (tree shade, dapple, top vs face) -> keep, but as flat values
    hm_f = hm.astype(np.float32)
    sig = 6.0 * ss
    lum = rgb.mean(-1) * hm_f
    lb = cv2.GaussianBlur(lum, (0, 0), sig) / np.maximum(cv2.GaussianBlur(hm_f, (0, 0), sig), 1e-3)
    L = lb[ys, xs]
    fa = face[ys, xs]
    top = np.abs(fa) == 2
    s = so[ys, xs].astype(np.float32)
    u = uo[ys, xs].astype(np.float32)
    Y = Yo[ys, xs].astype(np.float32)
    b = bid[ys, xs]
    Ytop = B[b, 3].astype(np.float32)
    dtop = np.clip(Ytop - Y, 0, 5)
    # clump pattern: big leaf clumps (0.3-0.45 m) + a mid scale, cel-banded (crisp value steps)
    cu = np.where(top, u, Y)
    n1 = _vn(s / 0.9, cu / 0.36, 91)
    n2 = _vn(s / 0.3, cu / 0.14, 93)
    n = 0.72 * n1 + 0.28 * n2
    # side face: lit tops of clumps (upper edge of each blob), cool shadow below
    nb = _vn(s / 0.9, (cu + 0.09) / 0.36, 91) * 0.72 + 0.28 * _vn(s / 0.3, (cu + 0.05) / 0.14, 93)
    capf = np.clip((n - nb) * 6.0, 0, 1)                   # clump top facing up (value rising toward top)
    band = np.where(n > 0.6, 2, np.where(n > 0.42, 1, 0)).astype(np.float32)
    # palettes
    top_cols = np.array([[0.34, 0.5, 0.2], [0.52, 0.66, 0.26], [0.72, 0.8, 0.36]], np.float32)
    face_cols = np.array([[0.1, 0.2, 0.22], [0.16, 0.3, 0.27], [0.28, 0.44, 0.3]], np.float32)
    bi = band.astype(np.int32)
    ct = top_cols[bi]
    cf = face_cols[bi]
    # face: lighter band just under the top edge (catches sky), darker toward the ground
    upper = np.exp(-dtop / 0.18)
    cf = cf * (0.8 + 0.35 * upper[:, None]) + np.array([0.18, 0.25, 0.1], np.float32) * (capf * 0.6 + 0.4 * upper)[:, None] * (band >= 1)[:, None]
    cf = cf * (1 - 0.35 * _ss(0.3, 1.2, dtop))[:, None]
    col = np.where(top[:, None], ct, cf)
    # tiny leaf flecks (sparse, crisp) on lit clumps
    fl = _vn(s / 0.045, cu / 0.04, 95)
    fk = (fl > 0.78) & (band >= 1)
    col = np.where(fk[:, None], col * 1.18 + 0.03, col)
    # keep the large-scale light (tree shadows / dapple) of the original render as a multiplier
    ref = np.where(top, 0.62, 0.3)
    mul = np.clip(L / ref, 0.35, 1.5)[:, None]
    col = col * mul
    if Zo is not None:
        hz = (haze_max * (1 - np.exp(-Zo[ys, xs] / haze_k)))[:, None].astype(np.float32)
        col = col * (1 - hz) + np.array(haze, np.float32) * hz
    rgb[ys, xs] = old * 0.1 + col * 0.9


def cumulus9(K2, P, cx, base_y, w, h, seed=0, pal=None, n=3, haze=0.0, levels=None, lost=0.04, sun_jitter=8.0,
             **shape_kw):
    """lib.clouds2.cumulus with the terminator painted firm (2-3 flat masses, crisp overlapping edges, no
    airbrushed lavender interior)"""
    import math
    rng = np.random.default_rng(seed)
    g = (cx, base_y - 0.45 * h, 0.58 * w, 0.62 * h)
    sun0 = P.sun
    d = sun0 - np.array([cx, base_y - 0.5 * h], np.float32)
    dist = float(np.hypot(*d)) + 1e-6
    ang = math.atan2(d[1], d[0]) + math.radians(rng.uniform(-1, 1) * sun_jitter)
    sun_c = np.array([cx + math.cos(ang) * dist, base_y - 0.5 * h + math.sin(ang) * dist], np.float32)
    sz = P.sun_z + rng.uniform(-0.15, 0.15)
    offs = sorted([(rng.uniform(-0.3, 0.3), rng.uniform(0.5, 1.0)) for _ in range(n)], key=lambda q: -q[1])
    for i, (ox, sc) in enumerate(offs):
        ww = w * (0.45 + 0.4 * sc)
        hh = h * sc
        by = base_y + (i / max(n - 1, 1)) * 0.06 * h
        poly = K2.envelope(cx + ox * w, by, ww, hh, rng, power=rng.uniform(2.1, 2.7), lump=0.08,
                           lean=rng.uniform(-0.1, 0.15), skew=rng.uniform(-0.4, 0.4), base_round=0.1, base_wave=0.05)
        kw = dict(shape_kw)
        if levels is not None:
            kw['levels'] = levels
        P.shape(poly, rng=rng, size=hh, group=g, pal=pal, haze=haze, haze_grad=0.25, sun_z=sz,
                form=rng.uniform(0.4, 0.65), split=rng.uniform(0.2, 0.36), cast=0.35 if i else 0.0, sun=sun_c,
                lost=lost, **kw)
    P.wisps(cx - w * 0.55, cx + w * 0.55, base_y + 0.05 * h, 0.13 * h, rng=rng, pal=pal, seed=seed + 3,
            amount=0.14)
