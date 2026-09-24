"""s01 helper (round 11): the hero cumulonimbus rebuilt on the clouds3 VOLUMETRIC lobe engine, with a
custom Shinkai paint pass.

Geometry (clouds3 head_lobes + hand-placed cores, camera space, units of the cloud height Hc):
  * a TAPERED tower: a broad heavy lower mass (~1.3x the mid column), a column that leans downwind and
    narrows, and a cauliflower crown that billows back out over the neck;
  * lobe radii vary ~4:1 (power-law head sizes + clouds3.surface_grow cauliflower at 3 finer scales),
    small secondary turrets break the sun-side silhouette, lobes grow out of lobes asymmetrically;
  * the base is not a ruler cut: the cut plane is jittered per lobe and the foot is later dissolved
    into torn, soft scud that sinks into the horizon haze.
Light: clouds3.vol_render (sun march + multiple scattering octaves + sky ambient occlusion).
Paint (this module):
  * the value from the renderer is mapped to painted planes: near-white warm cream lit planes
    (~95 % luma), a pale cool turn, periwinkle shade and a saturated cobalt / ultramarine core deep in
    the far-side and lower lobes;
  * stepped value masses with brush-warped borders, Kuwahara flattening, dry-brush strokes;
  * edges: crisp where lit, lost soft edges on the shadow / down side and on the base;
  * backlight at the crown: a hot silver-gold rim (2-6 px) along every crown lobe facing the sun, glowing
    into bloom (values > 1), a transmitted glow in thin edge lobes, the crown interior darkened a little
    against the rim;
  * torn fibrous veils / wisps at the crown and the flanks.
Deterministic; resolution independent (sizes scale with Hc / plate width).
"""
import math

import numpy as np
import cv2

from lib import clouds3 as K3
from lib import core as C

F32 = np.float32


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _c(v):
    return np.asarray(v, F32)


def _noise(w, h, px, seed, octaves=3, stretch=1.0, angle=0.0):
    """Value-ish fbm in -1..1 with features ~px across."""
    return K3._noise(w, h, max(w / max(px, 1.0), 2.0), seed, octaves, stretch=stretch, angle=angle)


# -------------------------------------------------------------------------------------------- geometry
# half-width profile (units of Hc) of a TOP-HEAVY tower (wwy_01 / cm5_04): a broad foot, a waist that
# swells with height, and a wide cauliflower crown that billows out over the neck (a young anvil / crown)
PROF = ((0.0, 0.25), (0.1, 0.27), (0.25, 0.26), (0.4, 0.28), (0.55, 0.31), (0.68, 0.35), (0.8, 0.4),
        (0.88, 0.42), (0.94, 0.38), (0.98, 0.26), (1.0, 0.12))


def _prof(s):
    return float(np.interp(s, [p[0] for p in PROF], [p[1] for p in PROF]))


def tower_E(rng, cx, base_y, Hc, eye, dist, ppx, L, lean=0.1, **_):
    """Lobe array (N, 13) for the hero tower in camera space: a top-heavy stack - broad foot, a stepped
    left shoulder tower, turrets stepping up the shaded right flank, and a wide crown leaning downwind
    (right) that overhangs the column, with a flattened shelf starting to spread on the downwind side."""
    X, Yb, Z = cx - ppx, base_y - eye, dist
    ph = rng.uniform(0, 6.28, 3)

    def ax(s):
        return X + Hc * (lean * s * s + 0.01 * math.sin(2 * math.pi * 1.3 * s + ph[0]))

    extra = []

    def core(s, xo, r, flat=0.8, zo=0.0, sx=1.1):
        # s = height of the head's TOP (fraction of Hc); xo = x offset of the centre (Hc)
        y = Yb - s * Hc + r * flat * Hc
        y = min(y, Yb - 0.02 * Hc)
        extra.append((np.array([ax(s) + xo * Hc, y, Z + zo * Hc]),
                      np.array([r * sx * Hc, r * flat * Hc, r * 0.85 * Hc])))

    # --- foot: flat wide masses (flat undersides sit on the haze)
    core(0.12, -0.07, 0.16, 0.55, -0.06, 1.05)
    core(0.14, 0.1, 0.17, 0.5, -0.04, 1.15)
    core(0.1, 0.27, 0.1, 0.55, -0.02, 1.2)
    # --- left shoulder: a lower sister tower stepping up into the main column (staircase silhouette)
    core(0.3, -0.3, 0.12, 0.85, -0.08, 1.1)
    core(0.42, -0.24, 0.1, 0.9, -0.06, 1.05)
    core(0.22, -0.3, 0.09, 0.8, -0.06, 1.05)
    # --- big tiers: masses growing with height (top-heavy), alternating small offsets
    for s, xo, r, zo in ((0.3, 0.0, 0.18, -0.04), (0.45, 0.03, 0.19, 0.0), (0.59, -0.02, 0.21, 0.02),
                         (0.72, 0.04, 0.23, 0.04), (0.84, 0.0, 0.25, 0.05)):
        core(s, xo, r, 0.78, zo, 1.1)
    # --- turrets stepping up the shaded right flank (varied size, bulging past the profile)
    for s, r in ((0.22, 0.08), (0.36, 0.11), (0.5, 0.075), (0.62, 0.12), (0.76, 0.09)):
        core(s, (_prof(s - 0.04) - 0.8 * r), r, 0.95, -0.01 + rng.uniform(-0.02, 0.02), 1.05)
    for s, r in ((0.55, 0.09), (0.7, 0.1), (0.8, 0.08)):
        core(s, -(_prof(s - 0.04) - 0.85 * r), r, 0.95, -0.03 + rng.uniform(-0.02, 0.02), 1.05)
    # --- crown: a wide boiling cauliflower dome of big heads at varied heights (a rounded crown, heads
    # rising out of heads), overhanging the neck on both sides and leaning downwind
    core(0.97, 0.02, 0.2, 0.85, 0.05, 1.15)
    for xo, top, rr, zo in ((-0.22, 0.92, 0.12, 0.0), (0.2, 0.96, 0.13, 0.0), (-0.09, 1.02, 0.1, -0.02),
                            (0.08, 1.0, 0.1, -0.02), (-0.02, 1.06, 0.065, -0.05), (-0.32, 0.84, 0.1, 0.0),
                            (0.33, 0.9, 0.11, 0.0), (0.42, 0.82, 0.08, 0.02), (-0.38, 0.76, 0.08, 0.0),
                            (0.14, 1.04, 0.06, -0.04), (0.27, 0.99, 0.07, -0.03), (-0.2, 0.99, 0.07, -0.03)):
        core(top, xo, rr, 0.88, zo, 1.1)
    E = K3.head_lobes(rng, X, Yb, Z, 2 * 0.22 * Hc, Hc, lean=0.0, sun3=L, n_heads=95,
                      prof=lambda s_: _prof(s_) / 0.22 * 0.82, r_rng=(0.22, 0.5), recede=0.25,
                      sizes=(0.05 / 0.22, 0.025 / 0.22, 0.012 / 0.22, 0.006 / 0.22),
                      dens=2.4, front=0.88, hier=0.8, clump=0.55, flat=0.82, wobble=0.0, extra=extra, shrink=0.0)
    # ragged base: the cut plane is jittered per lobe (steps + slopes, not a ruler line)
    E[:, 7] = Yb + Hc * (0.006 * rng.standard_normal(len(E)) + 0.006 * np.sin(E[:, 0] / (0.07 * Hc) + ph[1]))
    return E

# -------------------------------------------------------------------------------------------- raster
def _norm(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)


def _wblur(n, m, sig):
    w = cv2.GaussianBlur(m, (0, 0), sig) + 1e-4
    return np.stack([cv2.GaussianBlur(np.ascontiguousarray(n[..., k] * m), (0, 0), sig) / w
                     for k in range(n.shape[-1])], -1)


def raster(E, w, h, f, ppx, ppy, L, Hc, march=(22, 0.03, 1.12, 0.05), scale=0.3):
    """Analytic ray cast (clouds3._raycast) + screen-space sun march (clouds3._march), cropped to the
    cloud's bbox. Returns dict of fields."""
    E = np.ascontiguousarray(E, np.float64).copy()
    E[:, 8] = scale * Hc
    bb = K3._bboxes(E, f, ppx, ppy, w, h)
    pad = int(0.03 * Hc) + 4
    x0, y0 = max(int(bb[:, 0].min()) - pad, 0), max(int(bb[:, 1].min()) - pad, 0)
    x1, y1 = min(int(bb[:, 2].max()) + pad, w), min(int(bb[:, 3].max()) + pad, h)
    W2, H2 = x1 - x0, y1 - y0
    px2, py2 = ppx - x0, ppy - y0
    bb = K3._bboxes(E, f, px2, py2, W2, H2)
    t0, t1, nx, ny, nz, sid = K3._raycast(E, bb, W2, H2, f, px2, py2, 0.0, 1.0, 0.0)
    hit = np.isfinite(t0)
    q = 2
    t0q, t1q, sidq = (np.ascontiguousarray(a_[::q, ::q]) for a_ in (t0, t1, sid))
    Hq, Wq = t0q.shape
    od = K3._march(t0q, t1q, sidq, E[:, 8].astype(np.float64), Wq, Hq, f / q, px2 / q, py2 / q,
                   float(L[0]), float(L[1]), float(L[2]), int(march[0]), march[1], march[2], march[3])
    od = cv2.resize(od, (W2, H2), interpolation=cv2.INTER_LINEAR)
    return dict(t0=t0, t1=t1, N=np.stack([nx, ny, nz], -1), sid=sid, hit=hit, od=od, box=(x0, y0, x1, y1),
                E=E, f=f, ppx=px2, ppy=py2)


# -------------------------------------------------------------------------------------------- paint
RAMP = [(0.0, (0.25, 0.36, 0.8)), (0.16, (0.34, 0.47, 0.87)), (0.34, (0.54, 0.65, 0.93)),
        (0.52, (0.76, 0.83, 0.96)), (0.68, (0.92, 0.94, 0.965)), (0.84, (0.985, 0.98, 0.965)),
        (1.0, (1.03, 1.015, 0.975))]


LIT = [(0.0, (0.7, 0.78, 0.94)), (0.3, (0.84, 0.88, 0.96)), (0.55, (0.94, 0.95, 0.965)),
       (0.8, (1.0, 0.978, 0.94)), (1.1, (1.05, 1.02, 0.95))]
SHADE = [(0.0, (0.2, 0.36, 0.8)), (0.25, (0.33, 0.49, 0.86)), (0.55, (0.53, 0.66, 0.91)),
         (0.8, (0.68, 0.78, 0.95)), (1.0, (0.78, 0.85, 0.965))]


def _ramp(stops, v):
    xs = np.array([s[0] for s in stops], F32)
    cs = np.array([s[1] for s in stops], F32)
    out = np.empty(v.shape + (3,), F32)
    for k in range(3):
        out[..., k] = np.interp(v, xs, cs[:, k])
    return out


def light(R, Hc, L, density=1.6, wrap=0.45, w_head=0.5, w_big=0.35, w_micro=0.15, dbg=None):
    """Value fields from the raster."""
    hit = R['hit']
    hitf = hit.astype(F32)
    E = R['E']
    H2, W2 = hit.shape
    sidc = np.where(hit, R['sid'], 0)
    f, px2, py2 = R['f'], R['ppx'], R['ppy']
    # head normal at each pixel's 3D point (normal of the head the floret grew from)
    hid = E[:, 11].astype(np.int64)[sidc]
    xs_ = (np.arange(W2, dtype=F32)[None, :] + 0.5 - px2) / f
    ys_ = (np.arange(H2, dtype=F32)[:, None] + 0.5 - py2) / f
    inv_ = 1.0 / np.sqrt(xs_ * xs_ + ys_ * ys_ + 1.0)
    tq = np.where(hit, R['t0'], 0).astype(F32) * inv_
    Pq = np.stack([tq * xs_, tq * ys_, tq], -1)
    Nhd = _norm((Pq - E[:, 0:3].astype(F32)[hid]) / (E[:, 3:6].astype(F32)[hid] ** 2))
    del Pq
    Nh = _norm(_wblur(Nhd, hitf, 0.012 * Hc) * 0.7 + Nhd * 0.3)
    Ns = R['N'].astype(F32)
    # big form: the silhouette inflated into a dome
    Rr = 0.22 * Hc
    D = np.minimum(cv2.distanceTransform(hit.astype(np.uint8), cv2.DIST_L2, 5), Rr).astype(F32)
    hgt = np.sqrt(np.maximum(2 * Rr * D - D * D, 0))
    hgt = cv2.GaussianBlur(hgt, (0, 0), 0.03 * Hc)
    gy_, gx_ = np.gradient(hgt)
    Nb = _norm(np.stack([gx_, gy_, -np.ones_like(gx_) * 0.8], -1))
    Nb = _norm(Nb * 0.7 + _wblur(Ns, hitf, 0.06 * Hc) * 0.3)
    N = _norm(Nh * w_head + Nb * w_big + Ns * w_micro)
    Lf = np.asarray(L, F32)
    ndl = N @ Lf
    lam = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
    vis_raw = np.exp(-R['od'] * density)
    ssig = 0.006 * Hc
    vis = cv2.GaussianBlur(vis_raw * hitf, (0, 0), ssig) / (cv2.GaussianBlur(hitf, (0, 0), ssig) + 1e-4)
    vis = np.clip(vis, 0, 1)
    flatb = (hit & (Ns[..., 1] > 0.97)).astype(F32)
    up = np.clip(-N[..., 1], 0, 1)
    direct = lam * vis * (1 - 0.8 * flatb)
    amb = 0.35 + 0.35 * up
    if dbg is not None:
        dbg.update(lam=lam, vis=vis, direct=direct, D=D / Rr)
    return dict(direct=direct, amb=amb, vis=vis, lam=lam, N=N, Ns=Ns, D=D, hitf=hitf)


def light_vol(R, Hc, R2=None, dbg=None):
    A = R['A']
    ia = 1.0 / np.maximum(A, 1e-4)
    S = R['sun'] * ia
    Am = R['amb'] * ia
    m = A > 0.5
    S = S / (np.percentile(S[m], 75) + 1e-6)
    Am = Am / (np.percentile(Am[m], 92) + 1e-6)
    lam = np.clip((R['lam'] + 0.35) / 1.35, 0, 1)
    direct = np.clip(S, 0, 1.2) * (0.35 + 0.65 * lam)
    if dbg is not None:
        dbg.update(lam=lam, vis=np.clip(S, 0, 1), direct=direct)
    amb = 0.15 + 0.85 * np.clip(Am, 0, 1.2) ** 1.5
    if R2 is not None:
        # sky light from above: lobe tops pale, undersides / lobes under an overhang deep (shade modelling)
        S2 = R2['sun'] / np.maximum(R2['A'], 1e-4)
        S2 = np.clip(S2 / (np.percentile(S2[m], 85) + 1e-6), 0, 1.2)
        l2 = np.clip((R2['lam'] + 0.3) / 1.3, 0, 1)
        sky = S2 * (0.3 + 0.7 * l2)
        amb = 0.1 + 0.9 * sky * (0.5 + 0.5 * np.clip(Am, 0, 1))
        if dbg is not None:
            dbg['sky'] = sky
    return dict(direct=direct, amb=amb, hitf=np.clip(A, 0, 1), A=A)


def _to_box(field, box_src, box_dst, fill=0.0):
    x0, y0, x1, y1 = box_dst
    X0, Y0, X1, Y1 = box_src
    out = np.full((y1 - y0, x1 - x0), fill, F32)
    ix0, iy0 = max(x0, X0), max(y0, Y0)
    ix1, iy1 = min(x1, X1), min(y1, Y1)
    if ix1 > ix0 and iy1 > iy0:
        out[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = field[iy0 - Y0:iy1 - Y0, ix0 - X0:ix1 - X0]
    return out


def light_hybrid(R, Rv, Hc, L, wrap=0.5, w_head=0.45, w_big=0.4, w_micro=0.15, dbg=None):
    """Crisp analytic geometry (silhouette, head normals, head ids) lit with the volumetric sun /
    ambient transport of the same lobe cluster."""
    hit = R['hit']
    hitf = hit.astype(F32)
    E = R['E']
    H2, W2 = hit.shape
    sidc = np.where(hit, R['sid'], 0)
    f, px2, py2 = R['f'], R['ppx'], R['ppy']
    hid = E[:, 11].astype(np.int64)[sidc]
    xs_ = (np.arange(W2, dtype=F32)[None, :] + 0.5 - px2) / f
    ys_ = (np.arange(H2, dtype=F32)[:, None] + 0.5 - py2) / f
    inv_ = 1.0 / np.sqrt(xs_ * xs_ + ys_ * ys_ + 1.0)
    tq = np.where(hit, R['t0'], 0).astype(F32) * inv_
    Pq = np.stack([tq * xs_, tq * ys_, tq], -1)
    Nhd = _norm((Pq - E[:, 0:3].astype(F32)[hid]) / (E[:, 3:6].astype(F32)[hid] ** 2))
    del Pq
    Nh = _norm(_wblur(Nhd, hitf, 0.01 * Hc) * 0.7 + Nhd * 0.3)
    Ns = R['N'].astype(F32)
    Rr = 0.2 * Hc
    D = np.minimum(cv2.distanceTransform(hit.astype(np.uint8), cv2.DIST_L2, 5), Rr).astype(F32)
    hgt = np.sqrt(np.maximum(2 * Rr * D - D * D, 0))
    hgt = cv2.GaussianBlur(hgt, (0, 0), 0.03 * Hc)
    gy_, gx_ = np.gradient(hgt)
    Nb = _norm(np.stack([gx_, gy_, -np.ones_like(gx_) * 0.8], -1))
    N = _norm(Nh * w_head + Nb * w_big + Ns * w_micro)
    Lf = np.asarray(L, F32)
    lam = np.clip((N @ Lf + wrap) / (1 + wrap), 0, 1)
    # volumetric transport resampled into the raster window
    Av = Rv['A']
    ia = 1.0 / np.maximum(Av, 1e-4)
    S = Rv['sun'] * ia
    Am = Rv['amb'] * ia
    mv = Av > 0.5
    S = np.clip(S / (np.percentile(S[mv], 75) + 1e-6), 0, 1.3)
    Am = np.clip(Am / (np.percentile(Am[mv], 90) + 1e-6), 0, 1.2)
    # fill outside the vol silhouette by dilation so raster edge pixels get a value
    S = K3._bleed(np.dstack([S, S, S]), mv.astype(F32), 6.0)[..., 0]
    Am = K3._bleed(np.dstack([Am, Am, Am]), mv.astype(F32), 6.0)[..., 0]
    S = _to_box(S, Rv['box'], R['box'], 1.0)
    Am = _to_box(Am, Rv['box'], R['box'], 1.0)
    up = np.clip(-N[..., 1], 0, 1)
    upl = np.clip(-Nh[..., 1] * 0.7 - Nh[..., 0] * 0.2 - Nh[..., 2] * 0.3, 0, 1)
    direct = lam * S
    amb = (0.25 + 0.75 * upl) * (0.4 + 0.6 * Am)
    flatb = (hit & (Ns[..., 1] > 0.97)).astype(F32)
    direct *= 1 - 0.8 * flatb
    if dbg is not None:
        dbg.update(lam=lam, vis=np.clip(S, 0, 1), direct=direct, amb=amb)
    return dict(direct=direct, amb=amb, hitf=hitf, Nh=Nh, N=N, hid=np.where(hit, hid, -1), D=D)


def paint(R, Lt, w, h, Hc, sun, base_y, seed=0, expo_k=0.4, lit_t=0.2, lit_w=0.05, kuwa=0, strokes=0.02, dbg=None):
    x0, y0, x1, y1 = R['box']
    hitf = Lt['hitf']
    hit = hitf > 0.5
    hh, ww = hitf.shape
    sc = w / 1920.0
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    # painter's big form: across-the-mass position per row (-1 sun flank .. +1 far flank)
    has = hit.any(1)
    xl = np.where(has, hit.argmax(1), 0).astype(F32)
    xr_ = np.where(has, ww - 1 - hit[:, ::-1].argmax(1), 0).astype(F32)
    idx = np.nonzero(has)[0]
    xl = np.interp(np.arange(hh), idx, xl[idx]).astype(F32)
    xr_ = np.interp(np.arange(hh), idx, xr_[idx]).astype(F32)
    xl = cv2.GaussianBlur(xl[None, :], (0, 0), 0.05 * Hc)[0]
    xr_ = cv2.GaussianBlur(xr_[None, :], (0, 0), 0.05 * Hc)[0]
    mid = 0.5 * (xl + xr_)[:, None]
    half = np.maximum(0.5 * (xr_ - xl), 1.0)[:, None]
    u = np.clip((np.arange(ww, dtype=F32)[None, :] - mid) / half, -1.3, 1.3)
    sh = np.clip((base_y - ys) / Hc, 0, 1.1)
    expo = np.clip(0.55 - 0.32 * u + 0.2 * (sh - 0.5) - 0.2 * _ss(0.3, 0.0, sh), 0, 1)
    mk = hitf > 0.5
    wn = _noise(ww, hh, 50 * sc, seed + 4, 3) * 0.07 + _noise(ww, hh, 12 * sc, seed + 5, 2, stretch=2.5, angle=-30) * 0.03
    # ---- lit plane
    lv = Lt['direct'] * (1 - expo_k + expo_k * 2 * expo)
    lv = lv / (np.percentile(lv[mk], 92) + 1e-6)
    m_l = _ss(lit_t - lit_w, lit_t + lit_w, lv + wn)
    lq = 0.5 * _ss(0.55 - 0.04, 0.55 + 0.04, lv + wn * 0.7) + 0.5 * _ss(0.85 - 0.05, 0.85 + 0.05, lv + wn * 0.5)
    lvv = np.clip(0.55 * lv + 0.45 * lq, 0, 1.1)
    lit_col = _ramp(LIT, lvv)
    # ---- shade plane: sky light models the shaded lobes (pale tops, deep undersides)
    sk = Lt['amb']
    sk = np.clip((sk - np.percentile(sk[mk], 4)) / (np.percentile(sk[mk], 95) - np.percentile(sk[mk], 4) + 1e-6), 0, 1)
    sk = cv2.GaussianBlur(sk.astype(F32), (0, 0), 0.006 * Hc)
    sk = sk * (0.75 + 0.25 * np.clip(expo * 1.4, 0, 1))
    ins = cv2.distanceTransform(mk.astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    core = _ss(-0.2, 0.8, u) * _ss(0.015 * Hc, 0.1 * Hc, ins) * (0.6 + 0.4 * _ss(0.6, 0.1, sh))
    sv = sk * (1 - 0.55 * core)
    sq = 0.5 * _ss(0.3 - 0.05, 0.3 + 0.05, sv + wn) + 0.5 * _ss(0.62 - 0.05, 0.62 + 0.05, sv + wn)
    sv = 0.68 * sv + 0.32 * sq
    sh_col = _ramp(SHADE, np.clip(sv, 0, 1))
    col = sh_col * (1 - m_l[..., None]) + lit_col * m_l[..., None]
    v = m_l * (0.5 + 0.5 * lvv) + (1 - m_l) * 0.45 * sv
    if dbg is not None:
        dbg['v0'] = v.copy()
        dbg['core'] = core
    if kuwa:
        col = K3.kuwahara(col, max(1, int(round(kuwa * sc))), q=6.0)
    tn = _noise(ww, hh, 180 * sc, seed + 31, 3)
    col = col * (1 + 0.025 * tn[..., None] * _c((1.0, 0.3, -0.8)))
    # brush: noise-displaced paint (wobbly dab borders), then dry-brush stroke texture
    gx1, gy1 = np.meshgrid(np.arange(ww, dtype=F32), np.arange(hh, dtype=F32))
    amp = 3.5 * sc + 0.5
    wx = _noise(ww, hh, 10 * sc, seed + 51, 3) * amp + _noise(ww, hh, 40 * sc, seed + 53, 2) * amp * 1.5
    wy = _noise(ww, hh, 10 * sc, seed + 52, 3) * amp + _noise(ww, hh, 40 * sc, seed + 54, 2) * amp * 1.5
    col = K3._bleed(np.ascontiguousarray(col, F32), _ss(0.3, 0.7, Lt['A'] if 'A' in Lt else hitf), 4.0)
    col = cv2.remap(col, gx1 + wx, gy1 + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if strokes:
        st = _noise(ww, hh, 9 * sc, seed + 41, 2, stretch=5.0, angle=-35)
        st2 = _noise(ww, hh, 4 * sc, seed + 42, 2, stretch=6.0, angle=-20)
        col = col * (1 + strokes * st[..., None] + 0.012 * st2[..., None])
    # alpha: crisp where lit, lost where shaded
    A = Lt['A'] if 'A' in Lt else cv2.GaussianBlur(hitf, (0, 0), 0.7)
    A = _ss(0.15, 0.6, A)
    lit = _ss(0.35, 0.7, v)
    soft = cv2.GaussianBlur(hitf, (0, 0), 0.012 * Hc)
    A = A * lit + np.minimum(A, _ss(0.2, 0.8, soft)) * (1 - lit)
    out = np.zeros((h, w, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = A
    out[..., :3] = K3._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    return out


# painted planes (display RGB): lit = warm cream white; the lavender lives ONLY in the thin half-tone turn;
# shade = clear sky-lit cobalt, core = saturated ultramarine, lobe undersides deepening toward the base
C_HI = (1.03, 1.01, 0.975)
C_LIT = (0.995, 0.985, 0.955)
C_LIT2 = (0.915, 0.92, 0.96)
C_TURN = (0.84, 0.85, 0.95)
C_SHADE = (0.6, 0.72, 0.94)
C_CORE = (0.37, 0.52, 0.88)
C_DEEP = (0.28, 0.42, 0.82)


def paint2(R, Lt, w, h, Hc, sun, base_y, L, seed=0, kuwa=3, dbg=None):
    """Broad painted value masses: big-form light (the silhouette inflated into one mass) modulated by the
    lobe light only at the lobe scale, then snapped into 3-4 painted planes with brush-warped borders."""
    x0, y0, x1, y1 = R['box']
    hitf = Lt['hitf']
    mk = hitf > 0.5
    hh, ww = hitf.shape
    sc = w / 1920.0
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    sh = np.clip((base_y - ys) / Hc, 0, 1.1)                  # 0 at the base .. 1 at the crown
    # --- big form: the silhouette inflated into a soft dome, lit from the sun (upper left, a little behind)
    mkf = mk.astype(F32)
    hgt = (cv2.GaussianBlur(mkf, (0, 0), 0.03 * Hc) * 0.25 + cv2.GaussianBlur(mkf, (0, 0), 0.07 * Hc) * 0.45 +
           cv2.GaussianBlur(mkf, (0, 0), 0.14 * Hc) * 0.3) * (0.35 * Hc)
    gy_, gx_ = np.gradient(hgt)
    Nb = _norm(np.stack([-gx_, -gy_, -np.ones_like(gx_) * 0.55], -1))
    Lb = _norm(np.asarray((L[0] * 0.9, L[1] * 1.25, -0.7), F32))
    big = np.clip((Nb @ Lb + 0.1) / 1.1, 0, 1) ** 1.3
    # --- lobe light (volumetric sun, lobe self-shadow), normalised + lightly smoothed
    dl = Lt['direct'].astype(F32)
    dl = dl / (np.percentile(dl[mk], 90) + 1e-6)
    dl = cv2.GaussianBlur(np.clip(dl, 0, 1.2), (0, 0), 0.004 * Hc)
    sky = Lt['amb'].astype(F32)
    sky = np.clip((sky - np.percentile(sky[mk], 5)) / (np.percentile(sky[mk], 95) - np.percentile(sky[mk], 5) + 1e-6), 0, 1)
    sky = cv2.GaussianBlur(sky, (0, 0), 0.018 * Hc)
    wn = _noise(ww, hh, 70 * sc, seed + 4, 3) * 0.085 + _noise(ww, hh, 16 * sc, seed + 5, 2, stretch=2.5, angle=-30) * 0.025
    # lit-ness: mostly the big form; the lobe term only carves the undersides of heads and overhangs
    lv = big * (0.5 + 0.5 * np.clip(dl, 0, 1)) + 0.3 * big * big
    lv = lv + 0.12 * _ss(0.55, 1.0, sh) * big                     # crown catches more light
    if dbg is not None:
        dbg['big'] = big
        dbg['dl'] = dl
        dbg['lv'] = lv
        dbg['sky'] = sky
    lvw = lv + wn + 0.35 * (dl - cv2.GaussianBlur(dl, (0, 0), 0.03 * Hc))
    hp_pre = sky - cv2.GaussianBlur(sky, (0, 0), 0.035 * Hc)
    m_lit = _ss(0.36, 0.42, lvw)                                    # lit plane
    m_hi = _ss(0.7, 0.76, lvw + 0.5 * wn) * (1 - _ss(0.05, 0.12, -hp_pre))                          # hot top plane
    m_turn = _ss(0.27, 0.32, lvw)                                   # thin half-tone turn
    # shade value: sky light models the shaded heads (pale tops, deeper bellies), darker toward the base
    # and deeper in the core (far from the lit flank)
    ins = cv2.distanceTransform(mk.astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    sv = 0.5 * sky + 0.5 * big / 0.44
    sv = sv * (0.8 + 0.2 * _ss(0.0, 0.5, sh)) - 0.25 * _ss(0.02 * Hc, 0.12 * Hc, ins) * (1 - big / 0.44)
    sv = np.clip(sv, 0, 1)
    svw = sv + wn
    q_sh = _ss(0.42, 0.48, svw)
    q_dp = _ss(0.2, 0.26, svw)
    shade = _c(C_DEEP) + (_c(C_CORE) - _c(C_DEEP)) * q_dp[..., None]
    shade = shade + (_c(C_SHADE) - shade) * q_sh[..., None]
    shade = shade * (1 + 0.06 * (sv - 0.5))[..., None]
    # on the lit flank a shaded belly is only a pale lavender half-tone, never the cobalt of the core
    has = mk.any(1)
    idx = np.nonzero(has)[0]
    xl = np.where(has, mk.argmax(1), 0).astype(F32)
    xr_ = np.where(has, ww - 1 - mk[:, ::-1].argmax(1), 0).astype(F32)
    xl = cv2.GaussianBlur(np.interp(np.arange(hh), idx, xl[idx]).astype(F32)[None, :], (0, 0), 0.06 * Hc)[0]
    xr_ = cv2.GaussianBlur(np.interp(np.arange(hh), idx, xr_[idx]).astype(F32)[None, :], (0, 0), 0.06 * Hc)[0]
    u = (np.arange(ww, dtype=F32)[None, :] - 0.5 * (xl + xr_)[:, None]) / np.maximum(0.5 * (xr_ - xl), 1.0)[:, None]
    flank = _ss(0.25, -0.45, u + 2 * wn)
    shade = shade + (_c(C_TURN) * 0.96 - shade) * np.maximum(_ss(0.3, 0.75, big + wn), 0.85 * flank)[..., None]
    # lobe-scale modelling (high-pass of the sky / sun light): bellies under each head, pale head tops
    hp = sky - cv2.GaussianBlur(sky, (0, 0), 0.05 * Hc)
    hpd = dl - cv2.GaussianBlur(dl, (0, 0), 0.035 * Hc)
    belly = _ss(0.05, 0.16, -(0.6 * hp + 0.4 * hpd) + 0.5 * wn)
    ptop = _ss(0.04, 0.14, hp + 0.5 * wn)
    shade = shade + (_c(C_SHADE) * 1.08 - shade) * (0.15 * ptop)[..., None]
    shade = shade + (_c(C_DEEP) - shade) * (0.45 * belly)[..., None]
    col = shade + (_c(C_TURN) - shade) * m_turn[..., None]
    lm = np.clip(cv2.GaussianBlur(dl, (0, 0), 0.006 * Hc) * 1.25 + 1.6 * hpd + 0.12 * _ss(0.6, 0.95, sh) + 0.3 * _ss(0.8, 0.9, sh), 0, 1)
    lit_c = _c(C_LIT2) + (_c(C_LIT) - _c(C_LIT2)) * _ss(0.2, 0.75, lm + 0.5 * wn)[..., None]
    lit_c = lit_c + (_c(C_TURN) - lit_c) * (0.85 * belly * (1 - 0.75 * _ss(0.78, 0.92, sh)))[..., None]
    col = col + (lit_c - col) * m_lit[..., None]
    col = col + (_c(C_HI) - col) * m_hi[..., None]
    # flat base: a darker blue-grey underside band (lit flank a little lighter), deepening to the base line
    bnd = _ss(0.075, 0.015, sh + 0.012 * wn / 0.06)
    base_c = _c((0.5, 0.58, 0.82)) + _c((0.12, 0.1, 0.06)) * (0.5 * m_lit)[..., None]
    col = col + (base_c - col) * (0.8 * bnd)[..., None]
    # warm bounce on the lit flank near the base, faint cool fill on the lit tops of shaded heads
    col = col + _c((0.02, 0.0, -0.03)) * (m_lit * _ss(0.4, 0.0, sh))[..., None]
    v = m_lit * 0.8 + m_hi * 0.2 + (1 - m_lit) * 0.4 * sv
    if dbg is not None:
        dbg['v0'] = v.copy()
    if kuwa:
        col = K3.kuwahara(col, max(1, int(round(kuwa * sc))), q=6.0)
    tn = _noise(ww, hh, 180 * sc, seed + 31, 3)
    col = col * (1 + 0.02 * tn[..., None] * _c((1.0, 0.3, -0.8)))
    gx1, gy1 = np.meshgrid(np.arange(ww, dtype=F32), np.arange(hh, dtype=F32))
    amp = 2.5 * sc + 0.5
    wx = _noise(ww, hh, 10 * sc, seed + 51, 3) * amp + _noise(ww, hh, 40 * sc, seed + 53, 2) * amp * 1.5
    wy = _noise(ww, hh, 10 * sc, seed + 52, 3) * amp + _noise(ww, hh, 40 * sc, seed + 54, 2) * amp * 1.5
    col = K3._bleed(np.ascontiguousarray(col, F32), _ss(0.3, 0.7, Lt['A'] if 'A' in Lt else hitf), 4.0)
    col = cv2.remap(col, gx1 + wx, gy1 + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    st = _noise(ww, hh, 9 * sc, seed + 41, 2, stretch=5.0, angle=-35)
    col = col * (1 + 0.012 * st[..., None])
    # alpha: crisp where lit, lost soft edges on the shaded / down side
    A = Lt['A'] if 'A' in Lt else cv2.GaussianBlur(hitf, (0, 0), 0.7)
    A = _ss(0.15, 0.6, A)
    lit = _ss(0.3, 0.7, v)
    soft = cv2.GaussianBlur(hitf, (0, 0), 0.005 * Hc)
    A = A * lit + np.minimum(A, _ss(0.2, 0.8, soft)) * (1 - lit)
    out = np.zeros((h, w, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = A
    out[..., :3] = K3._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    return out


# painted planes for paint3 (display RGB). Lit: warm cream white. Shade: DESATURATED blue-grey (wwy_01),
# four steps from the sky-lit lobe tops down to the deep bellies; a warm lavender-grey bounce in the
# undersides of the low shaded lobes (light reflected up from the sunlit haze / town).
P3_HI = (1.04, 1.02, 0.98)
P3_LIT = (0.99, 0.975, 0.945)
P3_LIT2 = (0.9, 0.91, 0.94)
P3_TURN = (0.77, 0.82, 0.91)
P3_S1 = (0.66, 0.74, 0.88)
P3_S2 = (0.56, 0.65, 0.82)
P3_S3 = (0.48, 0.57, 0.76)
P3_S4 = (0.41, 0.5, 0.7)
P3_WARM = (0.8, 0.74, 0.76)


def _lit_islands(m, min_area):
    """Mask of small isolated connected components of a binary mask (stray specks)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    small = np.zeros(n, bool)
    small[1:] = st[1:, cv2.CC_STAT_AREA] < min_area
    return small[lab]


def paint3(R, R2, w, h, Hc, base_y, Rr=None, seed=0, Lp=(-0.55, -0.7, -0.45), kuwa=3, lobe_k=0.42, lit_med=0.4, dbg=None):
    """Lobe-wise painted light. Per-lobe normals come from the volume's depth map at two scales (small
    lobes, heads) blended with the silhouette dome (big form); so the terminator wraps around each lobe
    instead of running as one vertical line. The lit side snaps into cream planes with crisp lobe tops;
    the shade side is four desaturated blue-grey steps modelled by sky light (pale tops, deeper bellies),
    with warm bounce in the undersides of the low lobes and lost soft edges against the sky."""
    x0, y0, x1, y1 = R['box']
    A = R['A'].astype(F32)
    mk = A > 0.5
    mkf = mk.astype(F32)
    hh, ww = A.shape
    sc = w / 1920.0
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    sh = np.clip((base_y - ys) / Hc, 0, 1.1)
    # ---- normals from depth (lobe + head scale) and the silhouette dome (big form)
    d = R['depth'].astype(F32)
    dm = np.where(mk, d, 0).astype(F32)
    d = K3._bleed(np.dstack([dm] * 3), mkf, 0.01 * Hc)[..., 0]

    def dnorm(sig, k):
        db = cv2.GaussianBlur(d, (0, 0), sig)
        gy, gx = np.gradient(db)
        return _norm(np.stack([gx, gy, -np.full_like(gx, k)], -1))
    N1 = dnorm(0.004 * Hc, 0.5)
    N2 = dnorm(0.014 * Hc, 0.45)
    hgt = (cv2.GaussianBlur(mkf, (0, 0), 0.03 * Hc) * 0.35 + cv2.GaussianBlur(mkf, (0, 0), 0.08 * Hc) * 0.65) * (0.3 * Hc)
    gy_, gx_ = np.gradient(hgt)
    Nb = _norm(np.stack([-gx_, -gy_, -np.ones_like(gx_) * 0.5], -1))
    N = _norm(0.4 * N1 + 0.3 * N2 + 0.3 * Nb)
    Lv = _norm(np.asarray(Lp, F32))
    # lobe light: each visible lobe of the analytic raster lit as its own head (lit top, shaded belly, a
    # crisp step where a nearer lobe's top overlaps the belly of the one behind); median-filtered so the
    # smallest florets do not read as bubbles, then high-passed against the big form
    if Rr is not None:
        hr = Rr['hit'].astype(F32)
        lr = np.clip((Rr['N'].astype(F32) @ Lv + 0.3) / 1.3, 0, 1) * hr
        lr = K3._bleed(np.dstack([lr] * 3), hr, 0.01 * Hc)[..., 0]
        lamL = _to_box(lr, Rr['box'], R['box'], 0.5)
        k_ = int(0.006 * Hc) | 1
        lamL = cv2.medianBlur((np.clip(lamL, 0, 1) * 255).astype(np.uint8), min(k_, 5)).astype(F32) / 255
        if k_ > 5:
            lamL = cv2.medianBlur((lamL * 255).astype(np.uint8), 5).astype(F32) / 255
            lamL = cv2.medianBlur((lamL * 255).astype(np.uint8), 5).astype(F32) / 255
    else:
        lamL = np.clip((_norm(0.55 * N1 + 0.45 * N2) @ Lv + 0.3) / 1.3, 0, 1)
    sgm = 0.03 * Hc
    hpL = lamL - cv2.GaussianBlur(lamL * mkf, (0, 0), sgm) / (cv2.GaussianBlur(mkf, (0, 0), sgm) + 1e-4)
    hpL = hpL * mkf
    hpP = cv2.GaussianBlur(np.maximum(hpL, 0), (0, 0), 0.0025 * Hc)
    hpN = cv2.GaussianBlur(np.minimum(hpL, 0), (0, 0), 0.006 * Hc)
    hpL = (hpP + 0.65 * hpN) * lobe_k
    lam = np.clip((N @ Lv + 0.25) / 1.25, 0, 1)
    # volumetric self-shadow (the mass shades its own far flank / lobes under overhangs)
    ia = 1.0 / np.maximum(A, 1e-4)
    S = R['sun'] * ia
    S = np.clip(S / (np.percentile(S[mk], 80) + 1e-6), 0, 1.2)
    S = cv2.GaussianBlur(S.astype(F32), (0, 0), 0.006 * Hc)
    big = np.clip((Nb @ Lv + 0.2) / 1.2, 0, 1)
    big = (big / (np.percentile(big[mk], 95) + 1e-6)) ** 1.8
    wl = 1 - 0.6 * _ss(0.45, 0.9, big)
    hpL = np.where(hpL > 0, hpL, hpL * wl)
    v = big * (0.55 + 0.45 * np.clip(S, 0, 1)) + hpL + 0.12 * _ss(0.75, 0.98, sh)
    v = (v - np.percentile(v[mk], 50)) + lit_med
    wn = _noise(ww, hh, 60 * sc, seed + 4, 3) * 0.05 + _noise(ww, hh, 14 * sc, seed + 5, 2, stretch=2.5, angle=-30) * 0.02
    vw = v + wn
    # ---- sky light (shade modelling): pale lobe tops, deeper bellies
    S2 = R2['sun'] / np.maximum(R2['A'], 1e-4)
    S2 = np.clip(S2 / (np.percentile(S2[mk], 85) + 1e-6), 0, 1.2)
    up = np.clip(-N[..., 1] * 0.8 - N[..., 2] * 0.3, 0, 1)
    sky = 0.55 * np.clip(S2, 0, 1) + 0.45 * up
    sky = cv2.GaussianBlur(sky.astype(F32), (0, 0), 0.003 * Hc)
    sky = (sky - np.percentile(sky[mk], 5)) / (np.percentile(sky[mk], 95) - np.percentile(sky[mk], 5) + 1e-6)
    sky = np.clip(sky * (0.8 + 0.2 * _ss(0.0, 0.6, sh)), 0, 1)
    svw = sky + wn
    # four stepped shade values, soft transitions (gradual value shift, no hard vertical step)
    shade = _c(P3_S4) + (_c(P3_S3) - _c(P3_S4)) * _ss(0.2, 0.3, svw)[..., None]
    shade = shade + (_c(P3_S2) - shade) * _ss(0.42, 0.52, svw)[..., None]
    shade = shade + (_c(P3_S1) - shade) * _ss(0.66, 0.76, svw)[..., None]
    # warm reflected light in the undersides of the low shaded lobes
    down = np.clip(N2[..., 1] * 1.2 + 0.1, 0, 1)
    warm = down * _ss(0.55, 0.1, sh) * _ss(0.55, 0.2, svw)
    shade = shade + (_c(P3_WARM) - shade) * (0.55 * warm)[..., None]
    # ---- lit side: terminator as a two-step gradual turn, then cream planes; crisp lobe tops
    m_turn = _ss(0.2, 0.3, vw)
    m_lit = _ss(0.34, 0.4, vw)
    lit_is = _lit_islands(m_lit > 0.5, (0.035 * Hc) ** 2)
    m_lit = m_lit * (1 - 0.85 * lit_is)
    m_hi = _ss(0.66, 0.7, vw + 0.4 * wn)
    lit_c = _c(P3_LIT2) + (_c(P3_LIT) - _c(P3_LIT2)) * _ss(0.45, 0.58, vw)[..., None]
    col = shade + (_c(P3_TURN) - shade) * m_turn[..., None]
    col = col + (lit_c - col) * m_lit[..., None]
    col = col + (_c(P3_HI) - col) * m_hi[..., None]
    # flat base: a darker blue-grey underside band deepening to the base line
    bnd = _ss(0.07, 0.012, sh + 0.012 * wn / 0.05)
    base_c = _c((0.56, 0.62, 0.78)) + _c((0.1, 0.08, 0.05)) * (0.5 * m_lit)[..., None]
    col = col + (base_c - col) * (0.7 * bnd)[..., None]
    vv = m_lit * 0.8 + m_hi * 0.2 + (1 - m_lit) * 0.4 * sky
    if dbg is not None:
        dbg.update(v3=v, sky3=sky, lam3=lam, S3=S, hpL=hpL + 0.5)
    if kuwa:
        col = K3.kuwahara(col, max(1, int(round(kuwa * sc))), q=6.0)
    tn = _noise(ww, hh, 180 * sc, seed + 31, 3)
    col = col * (1 + 0.02 * tn[..., None] * _c((1.0, 0.3, -0.8)))
    gx1, gy1 = np.meshgrid(np.arange(ww, dtype=F32), np.arange(hh, dtype=F32))
    amp = 2.0 * sc + 0.5
    wx = _noise(ww, hh, 10 * sc, seed + 51, 3) * amp + _noise(ww, hh, 40 * sc, seed + 53, 2) * amp * 1.5
    wy = _noise(ww, hh, 10 * sc, seed + 52, 3) * amp + _noise(ww, hh, 40 * sc, seed + 54, 2) * amp * 1.5
    col = K3._bleed(np.ascontiguousarray(col, F32), _ss(0.3, 0.7, A), 4.0)
    col = cv2.remap(col, gx1 + wx, gy1 + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    st = _noise(ww, hh, 9 * sc, seed + 41, 2, stretch=5.0, angle=-35)
    col = col * (1 + 0.012 * st[..., None])
    # ---- alpha: crisp on lit lobes and on the tops of shaded lobes, lost + soft on the shaded down side
    Ac = _ss(0.15, 0.6, A)
    crisp = np.clip(_ss(0.3, 0.7, vv) + 0.6 * _ss(0.5, 0.8, sky) * _ss(0.2, 0.6, up), 0, 1)
    soft = cv2.GaussianBlur(mkf, (0, 0), 0.01 * Hc)
    Al = np.minimum(Ac, _ss(0.1, 0.9, soft)) * (0.9 + 0.1 * soft)
    Aout = Ac * crisp + Al * (1 - crisp)
    out = np.zeros((h, w, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = Aout
    out[..., :3] = K3._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    return out


def finish(P, Hc, sun, base_y, seed=0, sun_dir=(-0.62, -0.78), rim=1.0, rim_reach=0.42, glow=1.0,
           wisps=0.35, base_fade=0.1, fl=1.0):
    """Crown backlight (hot rim + translucent glow + darkened interior), silhouette florets, wisps,
    and a soft torn base sinking into the haze."""
    h, w = P.shape[:2]
    sc = w / 1920.0
    if fl:
        K3.edge_florets(P, sun_dir=sun_dir, r=(1.5, 10.0), density=1.4 * fl, up_min=0.15, sun_w=0.7, seed=seed,
                        lift=0.03)
        K3.edge_florets(P, sun_dir=sun_dir, r=(1.0, 4.0), density=1.2 * fl, up_min=0.1, sun_w=0.7, seed=seed + 1,
                        lift=0.03)
    A = P[..., 3]
    nzr = np.nonzero(A.max(1) > 0.01)[0]
    nzc = np.nonzero(A.max(0) > 0.01)[0]
    pad = int(0.08 * Hc)
    y0, y1 = max(nzr.min() - pad, 0), min(nzr.max() + pad, h)
    x0, x1 = max(nzc.min() - pad, 0), min(nzc.max() + pad, w)
    c = P[y0:y1, x0:x1, :3].copy()
    a = P[y0:y1, x0:x1, 3].copy()
    hh, ww = a.shape
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    dsun = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    m = (a > 0.5).astype(np.uint8)
    ins = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(F32)
    outd = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5).astype(F32)
    Ab = cv2.GaussianBlur(a, (0, 0), 2.5 * sc + 1)
    gx = cv2.Sobel(Ab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Ab, cv2.CV_32F, 0, 1, ksize=3)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    nx, ny = -gx / gl, -gy / gl                     # outward normal
    to_x, to_y = sun[0] - xs, sun[1] - ys
    tl = np.sqrt(to_x ** 2 + to_y ** 2) + 1e-6
    face = np.clip((nx * to_x + ny * to_y) / tl, 0, 1)
    near = np.exp(-(dsun / rim_reach) ** 2)
    # --- backlit crown: interior a little darker / cooler against the rim, hot rim, transmitted glow
    if rim:
        # backlight: the sun sits just behind the crown, so EVERY edge near it glows (no facing test):
        # a blown-out 3-8 px rim fading with distance from the sun, and just inside it a thin cooler
        # half-tone turn so the rim reads against the lit crown
        rn = _noise(ww, hh, 30 * sc, seed + 71, 2)
        nearr = np.exp(-(dsun / (rim_reach * 0.55)) ** 2)
        rw_ = (2.0 + 6.0 * nearr * (0.7 + 0.3 * rn)) * sc + 0.8          # 3-8 px at 1080p near the sun
        band = 1 - _ss(rw_ * 0.6, rw_ * 1.25, ins)
        turn = _ss(rw_ * 0.8, rw_ * 1.6, ins) * _ss(0.05 * Hc, 0.015 * Hc, ins) * nearr
        c = c + (_c((0.82, 0.84, 0.94)) - c) * (0.7 * rim * turn)[..., None]
        k = band * np.clip(nearr * (0.75 + 0.25 * face), 0, 1) * rim
        rimc = _c((1.9, 1.75, 1.4))
        c = c + (rimc - c) * np.clip(k, 0, 1)[..., None]
        # silver-lined lobe edges down the shaded flank (cm5_05): thin, on up / outward facing lobe tops
        lum = c @ _c((0.3, 0.55, 0.15))
        thin = 1 - _ss(1.2 * sc + 0.6, 3.0 * sc + 1.0, ins)
        k2 = thin * _ss(0.2, 0.6, -ny * 0.7 + nx * 0.5) * _ss(0.8, 0.62, lum) * np.exp(-dsun / 0.5) * (m > 0) * (rn > -0.3)
        c = c + (_c((1.3, 1.28, 1.25)) - c) * np.clip(0.85 * k2 * rim, 0, 1)[..., None]
        gl_ = np.exp(-ins / (0.012 * Hc)) * nearr * glow
        c = c + _c((0.35, 0.3, 0.2)) * gl_[..., None]
    # --- wisps: torn fibrous veils hanging off the crown and the flanks (outside the silhouette)
    if wisps:
        n1 = _noise(ww, hh, 0.05 * Hc, seed + 81, 4, stretch=4.0, angle=-8)
        n2 = _noise(ww, hh, 0.012 * Hc, seed + 82, 3, stretch=6.0, angle=-8)
        fib = _ss(0.15, 0.7, 0.6 * n1 + 0.4 * n2)
        band_o = np.exp(-outd / (0.03 * Hc)) * (outd > 0)
        sh_ = np.clip((base_y - ys) / Hc, 0, 1.2)
        where = _ss(0.25, 0.55, sh_) * (0.5 + 0.5 * _ss(0.6, 0.95, sh_))
        wa = np.clip(band_o * fib * 1.2, 0, 1) * where * 0.55 * wisps
        wcol = _c((0.93, 0.95, 0.99)) + (_c((1.35, 1.25, 1.05)) - _c((0.93, 0.95, 0.99))) * (near * 0.8)[..., None]
        # also thin veil over the silhouette edge (soft torn fringe)
        c = c * (1 - wa[..., None]) * (a[..., None]) / np.maximum(np.maximum(a, wa)[..., None], 1e-4) +             wcol * (wa[..., None]) / np.maximum(np.maximum(a, wa)[..., None], 1e-4) * (1 - a[..., None]) +             c * 0.0
        a = np.maximum(a, wa)
    # --- base: torn, soft, darker flat shelf that sinks into the haze
    if base_fade:
        fh = base_fade * Hc
        n = _noise(ww, hh, 0.08 * Hc, seed + 7, 4, stretch=5.0)
        edge = base_y - fh * (0.55 + 0.45 * n)
        f = _ss(edge - fh * 0.4, edge + fh * 0.9, ys)
        c = c + (_c((0.6, 0.68, 0.88)) - c) * (0.5 * _ss(base_y - 2.2 * fh, base_y - 0.3 * fh, ys))[..., None] * (1 - near[..., None])
        a = a * (1 - 0.92 * f)
    P[y0:y1, x0:x1, :3] = c
    P[y0:y1, x0:x1, 3] = a
    P[..., :3] = K3._bleed(P[..., :3], (P[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * w))
    return P


def _drop_specks(P, min_area):
    """Remove detached little blobs (stray lobes floating off the mass)."""
    m = (P[..., 3] > 0.3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 2:
        return P
    big = np.zeros(n, bool)
    big[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
    keep = big[lab]
    keep = cv2.dilate(keep.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    P[..., 3] *= keep
    return P


def hero(pw, ph, cx, base_y, Hc, W, hz_y, seed=7, sun_dir=(-0.62, -0.78), sun_z=-0.1, sun_dx=-0.03, engine='vol', max_dim=380, dbg=None):
    """Returns (rgba plate (ph, pw, 4), sun position in plate px)."""
    rng = np.random.default_rng(seed)
    eye = base_y + 0.012 * ph
    dist = 4.0 * Hc
    cam = K3._side_camera(pw, ph, eye, dist)
    L = K3._sun_vec(None, sun_dir, sun_z, None)
    E = tower_E(rng, cx, base_y, Hc, eye, dist, cam['ppx'], L)
    if engine == 'hybrid':
        R = raster(E, pw, ph, cam['f'], cam['ppx'], cam['ppy'], L, Hc)
        Rv = K3.vol_render(E, pw, ph, cam['f'], cam['ppx'], cam['ppy'], L, max_dim=max_dim, fine_amp=0.5,
                           erode=(0.8, 8.0, 3, 0.8), seed=seed)
        Lt = light_hybrid(R, Rv, Hc, L, dbg=dbg)
    elif engine == 'vol':
        R = K3.vol_render(E, pw, ph, cam['f'], cam['ppx'], cam['ppy'], L, max_dim=max_dim, fine_amp=0.7,
                          erode=(1.0, 6.0, 3, 0.8), seed=seed)
        Lsky = K3._unit((0.2, -0.75, -0.65))
        R2 = K3.vol_render(E, pw, ph, cam['f'], cam['ppx'], cam['ppy'], Lsky, max_dim=max_dim, fine_amp=0.7,
                           erode=(1.0, 6.0, 3, 0.8), seed=seed)
        Lt = light_vol(R, Hc, R2=R2, dbg=dbg)
        rr_ = E[:, 3:5].max(1)
        Er = E[(rr_ > 0.03 * Hc) & (rr_ < 0.11 * Hc)]
        Rr = raster(Er, pw, ph, cam['f'], cam['ppx'], cam['ppy'], L, Hc, march=(4, 0.03, 1.12, 0.05))
        if dbg is not None:
            dbg['R2'] = R2
            dbg['Rr'] = Rr
    else:
        R = raster(E, pw, ph, cam['f'], cam['ppx'], cam['ppy'], L, Hc)
        Lt = light(R, Hc, L, dbg=dbg)
    sun = np.array([cx - 0.02 * Hc, base_y - 1.0 * Hc], np.float32)
    if engine == 'vol':
        P = paint3(R, R2, pw, ph, Hc, base_y, Rr=Rr, seed=seed, dbg=dbg)
    else:
        P = paint2(R, Lt, pw, ph, Hc, sun, base_y, L, seed=seed, dbg=dbg)
    _drop_specks(P, (0.02 * Hc) ** 2)
    if dbg is not None:
        dbg['R'] = R
    # sun just behind the crown: on the column through the crown's highest point, slightly inside
    a_ = P[..., 3] > 0.5
    rows = np.nonzero(a_.any(1))[0]
    ytop = rows[0]
    xs_top = np.nonzero(a_[ytop + int(0.01 * Hc)])[0]
    sun = np.array([xs_top.mean() + sun_dx * Hc, ytop + 0.03 * Hc], np.float32)
    P = finish(P, Hc, sun, base_y, seed=seed, sun_dir=sun_dir)
    return P, sun


if __name__ == '__main__':
    import sys
    import os
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib import sky as S
    W, H = (1920, 1080) if 'full' in sys.argv else (960, 540)
    pw, ph = int(W * 1.12), int(H * 1.22)
    t0 = time.time()
    mg = 0.03 * H
    base_y = mg + 0.845 * H
    top_y = mg + 0.15 * H
    Hc = (base_y - top_y) / 1.01
    dbg = {}
    P, sun = hero(pw, ph, (pw - W) / 2 + 0.38 * W, base_y, Hc, W, base_y, seed=4, dbg=dbg)
    print('hero', time.time() - t0, sun)
    sky = S.sky_gradient(pw, ph, dict(stops=[(0.0, '#0a45b8'), (0.16, '#155dcd'), (0.34, '#2e80de'), (0.5, '#58a2e9'),
                                             (0.64, '#8ec4f0'), (0.78, '#b8dcf6'), (0.9, '#d6ecf9'), (1.0, '#ebf8fc')],
                                      sun_glow='#fff6e0', sun_glow_amt=0.3, below='#eef9fc'), horizon=0.95)[..., :3]
    img = sky * (1 - P[..., 3:4]) + P[..., :3] * P[..., 3:4]
    ox = (pw - W) // 2
    img = img[int(mg):int(mg) + H, ox:ox + W]
    tag = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'full' else 'a'
    C.save_png(os.path.join('out', 'compare', 'nb_%s.png' % tag), np.clip(img, 0, 1))
    for k_ in ('big', 'dl', 'lv', 'sky'):
        if k_ in dbg:
            C.save_png(os.path.join('out', 'compare', 'nb_%s.png' % k_), np.clip(dbg[k_], 0, 1)[..., None].repeat(3, -1))
    if 'v0' in dbg:
        C.save_png(os.path.join('out', 'compare', 'nb_v0.png'), np.clip(dbg['v0'], 0, 1)[..., None].repeat(3, -1))
