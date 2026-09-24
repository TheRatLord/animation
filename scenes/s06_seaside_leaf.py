"""s06_seaside foliage, round 4: clustered leaf masses painted with supersampled flat values.

One clump = a heap of irregular leaf clusters (blobs whose outline carries a few lumps and some pointed
leaf-tip flicks, sizes varied), painted back to front on a 4x supersampled label grid and box-filtered
down, so every silhouette and value edge is anti-aliased (no stair-steps). Each cluster is one flat
value chosen from the clump's form (ellipsoid normal . light): a cool blue-green core in shadow, a
teal mid tone, a warm ochre lit face on the sun side; lit clusters carry a smaller offset blob of the
lit / hot value toward the sun (crisp, irregular value steps inside the mass - never a gradient). The
sun-facing silhouette gets a broken warm rim: flicks of varied width, interrupted by noise.
Everything is bbox-local and deterministic (rng passed in)."""
import math
import numpy as np
import cv2

from lib import core as C

SS = 4


def _blob(px, py, R, rng, flat=0.9, tips=0.0, ntip=None, bottom=0.25):
    """closed polygon of one irregular leaf cluster (float px)."""
    n = int(np.clip(R * 2.2, 24, 160))
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    k1, k2 = rng.integers(2, 4), rng.integers(4, 7)
    rr = 1 + 0.13 * np.sin(k1 * th + rng.uniform(0, 6.3)) + 0.08 * np.sin(k2 * th + rng.uniform(0, 6.3))
    if tips > 0:
        k3 = ntip or int(np.clip(R / 3.0, 6, 16) * rng.uniform(0.8, 1.2))
        ph = th * k3 / (2 * math.pi) + 0.3 * np.sin(th * 2 + rng.uniform(0, 6.3))
        saw = np.abs(2 * (ph - np.floor(ph)) - 1)            # 1 at tips .. 0 at notches
        dep = rng.uniform(0.5, 1.2, k3)[np.floor(ph).astype(int) % k3]
        up = np.clip(-np.sin(th) + 0.3, 0, 1)                # tips mostly on the upper / side edges
        rr = rr * (1 - tips * 0.35 + tips * 0.7 * dep * saw ** 2 * up)
    down = np.clip(np.sin(th), 0, 1)
    rr = rr * (1 - bottom * down ** 2)
    return np.stack([px + R * rr * np.cos(th), py + R * rr * np.sin(th) * flat], 1)


def _fill(lab, poly, x0, y0, val):
    q = np.round((np.asarray(poly, np.float64) - [x0, y0]) * SS * 16).astype(np.int32)
    cv2.fillPoly(lab, [q], float(val), lineType=cv2.LINE_8, shift=4)


def _shift(m, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def clump(cv, cx, cy, r, rng, pal, L=(1.0, -0.35), flat=0.78, clip=None, lit_bias=0.0, rim=1.0, rim_px=2.2,
          unit=1.0, alpha=1.0, holes=None, density=1.0, **_):
    """Paint one foliage clump centred (cx, cy), radius r (canvas px) into Canvas cv.
    pal: deep, shd, sky, mid, lit, hot, rim (rgb; rim HDR). L: screen direction toward the light."""
    Lx, Ly = L
    nL = math.hypot(Lx, Ly) + 1e-9
    Lx, Ly = Lx / nL, Ly / nL
    u = max(unit, 0.3)
    if 'mid' not in pal:
        pal = dict(pal, mid=np.asarray(pal['shd']) * 0.55 + np.asarray(pal['lit']) * 0.45)
    ext = r * 1.45 + 6
    X0 = int(max(math.floor(cx - ext), 0)); Y0 = int(max(math.floor(cy - ext * flat - 4), 0))
    X1 = int(min(math.ceil(cx + ext), cv.W)); Y1 = int(min(math.ceil(cy + ext * flat + 4), cv.H))
    if X1 - X0 < 3 or Y1 - Y0 < 3:
        return
    w, h = X1 - X0, Y1 - Y0
    names = ['deep', 'shd', 'sky', 'mid', 'lit', 'hot']
    table = [np.zeros(3, np.float32)]                       # label 0 = empty
    lab = np.zeros((h * SS, w * SS), np.float32)

    def paint(poly, name, var=0.04):
        c = np.asarray(pal[name], np.float32) * rng.uniform(1 - var, 1 + var)
        table.append(c)
        _fill(lab, poly, X0, Y0, len(table) - 1)

    ybot = cy + 0.38 * r * flat
    # ---- clusters: big ones inside, smaller ones toward the rim, leaf-tip flicks on the outer edge
    cl = []
    nbig = int(np.clip(5 + r / (10.0 * u), 5, 22) * density)
    for _i in range(nbig):
        a = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 1)) * 0.72
        px = cx + math.cos(a) * rad * r
        py = min(cy + math.sin(a) * rad * r * flat, ybot - 0.12 * r)
        R = r * (0.12 + 0.36 * rng.uniform(0, 1) ** 1.6) * (1.2 - 0.45 * rad)
        cl.append((px, py, R, rng.uniform(0.1, 0.4)))
    nedge = int(np.clip(6 + r / (5.0 * u), 6, 40) * density)
    for _i in range(nedge):
        a = rng.uniform(-math.pi * 1.05, math.pi * 0.05)          # upper half + a bit of the flanks
        rad = rng.uniform(0.72, 0.95)
        px = cx + math.cos(a) * rad * r
        py = min(cy + math.sin(a) * rad * r * flat, ybot - 0.05 * r)
        R = r * (0.06 + 0.2 * rng.uniform(0, 1) ** 1.8)
        cl.append((px, py, R, rng.uniform(0.15, 0.4)))
    cl.append((cx, cy - 0.05 * r * flat, r * 0.62, 0.0))
    # value of each cluster from the clump's form
    vals = []
    for (px, py, R, tp) in cl:
        nx, ny = (px - cx) / r, (py - cy) / (r * flat)
        lam = 0.8 * (nx * Lx + ny * Ly) - 0.32 * ny + lit_bias + rng.normal(0, 0.1)
        vals.append(lam)
    # back to front: the core first, then darker / far-side clusters, lit ones on top (bulge into shade)
    order = sorted(range(len(cl)), key=lambda i: (cl[i][2] < r * 0.6, vals[i] + 0.25 * (cl[i][1] - cy) / r))
    lit_parts = []
    for i in order:
        px, py, R, tp = cl[i]
        v = vals[i]
        ny = (py - cy) / (r * flat)
        poly = _blob(px, py, R, rng, tips=tp, bottom=0.3)
        if v < -0.2:
            base = 'deep' if ny > 0.15 else 'shd'
        elif v < -0.02:
            base = 'shd' if ny > 0.1 else 'sky'
        elif v < 0.12:
            base = 'olv' if 'olv' in pal else 'mid'
        else:
            base = 'mid'
        paint(poly, base)
        # lit crescent: the same cluster shape displaced away from the light, subtracted -> lit side
        if v > 0.12:
            f = min((v - 0.12) / 0.6, 1.0)
            D = R * (0.25 + 0.45 * f)
            lit_parts.append((poly, px, py, D, f))
    # lit / hot steps (flat, irregular): painted as the cluster minus a displaced copy of itself
    for (poly, px, py, D, f) in lit_parts:
        sub = np.zeros_like(lab)
        _fill(sub, poly, X0, Y0, 1)
        steps = [('lit', D, 1.04)]
        if f > 0.65:
            steps.append(('hot', D * (0.3 + 0.4 * (f - 0.65)), 1.02))
        for (name, d, sc) in steps:
            s_ = sub.copy()
            rel = (poly - [px, py]) * sc
            th = np.arctan2(rel[:, 1], rel[:, 0])
            k = int(rng.integers(4, 14))
            ph = th * k / (2 * math.pi) + rng.uniform(0, 1) + 0.25 * np.sin(th * 3 + rng.uniform(0, 6.3))
            saw = np.abs(2 * (ph - np.floor(ph)) - 1)
            dep = rng.uniform(0.4, 1.2, k)[np.floor(ph).astype(int) % k]
            nd = np.hypot(rel[:, 0], rel[:, 1]).max() + 1e-6
            rel = rel * (1 - rng.uniform(0.08, 0.24) * dep * saw ** 1.5)[:, None] if nd > 3 else rel
            q = rel + [px - Lx * d, py - Ly * d + 0.2 * d]
            _fill(s_, q, X0, Y0, 0)
            sel = (s_ > 0) & (lab > 0)
            table.append(np.asarray(pal[name], np.float32) * rng.uniform(0.95, 1.05))
            lab[sel] = len(table) - 1
    # ---- a few sky holes in the upper part of big clumps
    nh = int(r / (32.0 * u)) if holes is None else holes
    for _ in range(min(nh, 6)):
        hx = cx + rng.uniform(-0.45, 0.45) * r
        hy = cy - rng.uniform(0.0, 0.6) * r * flat
        hr = r * rng.uniform(0.02, 0.07)
        _fill(lab, _blob(hx, hy, hr, rng, flat=rng.uniform(0.6, 0.85), bottom=0.2), X0, Y0, 0)
    # ---- colours at 4x, then box-filter down (premultiplied) -> anti-aliased edges
    tab = np.stack(table, 0).astype(np.float32)
    li = lab.astype(np.int32)
    covs = (li > 0).astype(np.float32)
    rgbs = tab[li]
    # broken warm rim on the sun-facing silhouette (at 4x: varied width, interrupted by noise)
    if rim > 0:
        rw = rim_px * u * SS
        nz = C.fbm(w * SS, h * SS, max(int(w / (6.0 * u)), 2), 2, seed=int(rng.integers(0, 1 << 30))) \
            if hasattr(C, 'fbm') else np.full_like(covs, 0.5)
        width = rw * (0.5 + 2.0 * np.clip(nz - 0.3, 0, 1))
        near = _shift(covs, -Lx * rw * 0.6, -Ly * rw * 0.6)
        far = _shift(covs, -Lx * rw * 2.4, -Ly * rw * 2.4)
        thin = np.clip(covs - near, 0, 1)
        thick = np.clip(covs - far, 0, 1)
        wmix = np.clip((width - rw * 0.5) / (rw * 1.9), 0, 1)
        rm = thin * (1 - wmix) + thick * wmix
        rm = rm * C.smoothstep(0.35, 0.6, nz) * (0.5 + 0.5 * (rgbs.max(-1) > np.asarray(pal['shd']).max() + 0.05))
        rimc = np.asarray(pal['rim'], np.float32)
        rgbs = rgbs + (rimc - rgbs) * np.clip(rm * rim, 0, 1)[..., None]
    prem = np.concatenate([rgbs * covs[..., None], covs[..., None]], -1)
    small = cv2.resize(prem, (w, h), interpolation=cv2.INTER_AREA)
    cov = small[..., 3]
    col = small[..., :3] / np.maximum(cov, 1e-5)[..., None]
    if clip is not None:
        cm, cx0, cy0 = clip
        cl_ = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            cl_[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
        cov = cov * cl_
    if cov.max() <= 1e-3:
        return
    cv.put((cov * alpha, X0, Y0), col)
