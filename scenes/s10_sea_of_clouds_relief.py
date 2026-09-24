"""Relief-painted towering cumulus for s10.

The dome LAYOUT comes from lib.clouds (gen_cloud: billows, lobes, anvil), but the rendering is our own:
the domes are rasterised into a camera-facing RELIEF map (front-surface depth, hard max -> crisp creases
where lobes overlap), and shading uses a blend of the detailed relief normal and a heavily smoothed
'form' normal - the big light shapes of the tower stay simple and painterly while the lobes only
modulate them (no bead-like spheres).  Silhouette normals turn outward (rim-light / translucency on the
sun-facing edges), crease occlusion draws the lobe separations, and the palette is applied as painted
value steps (shadow | terminator band | lit | hot cap).
"""
import math
import numpy as np
import cv2
from numba import njit

from lib import core as C, clouds as K, sky as S


@njit(cache=True)
def _raster(Hs, Ws, D, ss, ks):
    """D rows: cx, cy, cz, r, ay, clip (plate px). Returns the front depth as a SOFT max (log-sum-exp
    with softness ks px: big billows merge into smooth masses, small lobes stay distinct), -1e9 outside."""
    Z = np.full((Hs, Ws), -1e9, np.float32)
    Sm = np.zeros((Hs, Ws), np.float32)
    I = np.zeros((Hs, Ws), np.int32)
    for k in range(D.shape[0]):
        cx, cy, cz, r, ay, clip = D[k, 0] * ss, D[k, 1] * ss, D[k, 2] * ss, D[k, 3] * ss, D[k, 4], D[k, 5] * ss
        ry = r * ay
        x0 = max(0, int(cx - r) - 1)
        x1 = min(Ws, int(cx + r) + 2)
        y0 = max(0, int(cy - ry) - 1)
        y1 = min(Hs, int(min(cy + ry, clip)) + 2)
        for y in range(y0, y1):
            yy = y + 0.5
            if yy > clip:
                continue
            dy = (yy - cy) / ry
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                d2 = dx * dx + dy * dy
                if d2 >= 1.0:
                    continue
                z = cz + r * math.sqrt(1.0 - d2)
                m = Z[y, x]
                if z > m:
                    Sm[y, x] = Sm[y, x] * math.exp((m - z) / ks) + 1.0
                    Z[y, x] = z
                    I[y, x] = int(D[k, 6])
                else:
                    Sm[y, x] += math.exp((z - m) / ks)
    for y in range(Hs):
        for x in range(Ws):
            if Sm[y, x] > 0.0:
                Z[y, x] += ks * math.log(Sm[y, x])
    return Z, I


def _nblur(x, m, s):
    return cv2.GaussianBlur(x * m, (0, 0), s) / (cv2.GaussianBlur(m, (0, 0), s) + 1e-4)


def _normals(z, m, s_edge, ogx, ogy, dist):
    gx = cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    nx, ny, nz = -gx, -gy, np.ones_like(z)
    ln = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / ln, ny / ln, nz / ln
    # silhouettes: the surface turns away from the viewer -> outward normal
    e = np.exp(-dist / s_edge)[..., None]
    n = np.stack([nx, ny, nz], -1)
    o = np.stack([ogx, ogy, np.full_like(ogx, 0.15)], -1)
    n = n * (1 - e) + o * e
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6)


def relief_clouds(pw, ph, specs, pal, lights, seed=0, detail=1.2, ss=2, dw=0.5, form=0.035, rim=1.0,
                  sky=None, wind=-1.0, sun_pos=None, soft_k=0.003, crease=0.25):
    """specs: list of (cx, base_y, w, h, dist, kind, flat) in plate px (as sky.cumulus_plate).
    lights: per cloud unit vector toward the sun (x right, y down, z toward the viewer).
    Returns straight-alpha RGBA (ph, pw, 4)."""
    P = S._palette(pal)
    B = K.Builder()
    for i, sp in enumerate(specs):
        rng = np.random.default_rng(seed * 1000 + i)
        K.gen_cloud(B, rng, sp, lights[i], 0.0, 0.0, detail=detail, z0=-sp[4] * pw * 3.0, wind=wind)
    A = np.array(B.S, np.float64)
    cid = A[:, 7].astype(int)
    D = np.ascontiguousarray(A[:, [0, 1, 2, 3, 4, 5, 7]])
    Hs, Ws = ph * ss, pw * ss
    Z, own = _raster(Hs, Ws, D, ss, np.float32(soft_k * Ws))
    own = cv2.dilate(own.astype(np.float32), np.ones((5, 5), np.uint8)).astype(np.int32)
    m = (Z > -1e8).astype(np.float32)
    Z = np.where(m > 0, Z - float(Z[m > 0].mean()), Z).astype(np.float32)   # small magnitudes: blur precision
    # owner cloud per pixel (for per-cloud light): nearest cloud centre in x among the specs
    z = np.where(m > 0, Z, 0).astype(np.float32)
    zf_ = _nblur(z, m, 1.0)
    z = np.where(m > 0, z, zf_ - 2.0 * ss).astype(np.float32)
    dist = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    ab = cv2.GaussianBlur(m, (0, 0), 0.004 * Ws)
    ogx = -cv2.Sobel(ab, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(ab, cv2.CV_32F, 0, 1, ksize=3)
    ol = np.sqrt(ogx ** 2 + ogy ** 2)
    ol = ol + 0.08 * float(ol.max())          # no arbitrary directions far from the silhouette
    ogx, ogy = ogx / ol, ogy / ol
    n_det = _normals(z, m, 0.006 * Ws, ogx, ogy, dist)
    zform = _nblur(z, m, form * Ws)
    zform = np.where(m > 0, zform, zform - 2.0 * ss).astype(np.float32)
    n_form = _normals(zform, m, 0.008 * Ws, ogx, ogy, dist)
    n = n_det * dw + n_form * (1 - dw)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    # light per pixel: from the cloud that owns the front surface there
    Ls = np.array(lights, np.float32)[own]                 # (Hs, Ws, 3)
    ndl = (n * Ls).sum(-1)
    # small lobes may only nudge the big light shape (never punch dark holes into the lit side)
    nf = (n_form * Ls).sum(-1)
    ndl = nf + np.clip(ndl - nf, -0.03, 0.3)
    # base-height fraction per pixel (0 base .. 1 top) from the owner spec
    bys = np.array([sp[1] for sp in specs], np.float32)[own] * ss
    hts = np.array([sp[3] for sp in specs], np.float32)[own] * ss
    ys = (np.arange(Hs, dtype=np.float32) + 0.5)[:, None]
    v = np.clip((bys - ys) / hts, 0, 1)
    # crease occlusion (relief valleys) - crisp lobe separations
    occ = np.clip((_nblur(z, m, 0.006 * Ws) - z) / (0.004 * Ws), 0, 1) * m
    occ2 = np.clip((_nblur(z, m, 0.02 * Ws) - z) / (0.02 * Ws), 0, 1) * m
    le = ndl - 0.06 * occ2
    lit = C.smoothstep(-0.04, 0.06, le)
    k = C.smoothstep(0.02, 0.42, le)[..., None]
    up = np.clip(-n[..., 1], 0, 1)[..., None]
    vv = v[..., None]
    shd = C.lerp(P['shd_low'], P['shd_high'], np.clip(0.35 * up + 0.65 * vv, 0, 1))
    shd = C.lerp(shd, P['bounce'], np.clip(n[..., 1:2], 0, 1) * 0.5 * (1 - vv))
    if sky is not None:
        skyr = cv2.resize(sky, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
        shd = C.lerp(shd, skyr, 0.12)
    lit_c = C.lerp(P['lit_low'], P['lit_high'], np.clip(k * 1.6, 0, 1))
    lit_c = C.lerp(lit_c, P['hi'], np.clip(k * 2 - 1, 0, 1) * (0.4 + 0.6 * vv))
    col = C.lerp(shd, lit_c, lit[..., None])
    band = np.exp(-((le - 0.01) / 0.05) ** 2)[..., None]
    col = C.lerp(col, P['mid'], band * 0.45)
    col = C.lerp(col, col * np.array([0.78, 0.74, 0.92], np.float32), (occ * crease)[..., None])
    # rims + translucency on sun-facing silhouettes
    Lxy = Ls[..., :2]
    face = np.clip(ogx * Lxy[..., 0] + ogy * Lxy[..., 1], 0, 1)
    face = C.smoothstep(0.1, 0.8, face)
    wr = max(0.0016 * Ws, 1.0)
    rline = np.exp(-dist / wr) * face * rim
    glowt = np.exp(-dist / (0.012 * Ws)) * face * rim * 0.45
    rimc = P['rim'] * 1.12
    col = C.lerp(col, rimc, np.clip(glowt, 0, 0.8)[..., None])
    col = C.lerp(col, rimc, np.clip(rline, 0, 1)[..., None])
    rgba = np.dstack([col * m[..., None], m])
    out = cv2.resize(rgba, (pw, ph), interpolation=cv2.INTER_AREA)
    a = out[..., 3:4]
    rgb = out[..., :3] / np.maximum(a, 1e-5)
    # soft halation just outside the sun-facing edges
    return np.dstack([rgb, a[..., 0]]).astype(np.float32)
