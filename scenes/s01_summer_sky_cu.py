"""s01 cumulonimbus, round 5: a smooth-union SDF cloud, ray-marched once into a painted plate.

Geometry (units of the tower height Hc; u right (+ = sun side), v up from the flat base, w toward camera)
  * 5-7 big MASSES (ellipsoids) stacked into a stepped tower (flat foot, body, column, crown, turrets),
    blended with a LARGE smooth-min -> their interiors merge into broad continuous planes.
  * MEDIUM billows (power-law radii) on the upper / outer surface, blended with a medium smooth-min.
  * SMALL cauliflower bumps ONLY on the silhouette and on the sunlit crowns, blended with a small
    smooth-min so the outline stays crisp and knobbly.
  * flat base (plane clip), torn afterwards.
Rendering (numba, orthographic, straight down -z): sphere-traced front surface, SDF normal, soft
shadow toward the sun (one big coherent shadow plane), SDF ambient occlusion (only the deep creases
where masses meet go dark), anti-aliased alpha from the closest approach of each ray.
Painting (numpy): warm-white crowns -> luminous peach mid-tone -> cool lavender / blue-violet shade,
a lighter bluer bounce inside the shadow sides, aerial haze toward the base, a silver-gold backlit rim
and translucent thin edges near the hidden sun, torn transparent wisps at the base.
"""
import hashlib
import math
import os

import numpy as np
import cv2
from numba import njit, prange

STATS = {}
LIGHT = (0.62, -0.72, 0.14)       # key light (x right, y down, z toward camera)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def _h(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def fblur(img, sigma):
    if sigma <= 0.3:
        return img
    if sigma <= 4.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    H, W = img.shape[:2]
    f = sigma / 3.0
    w, h = max(int(W / f), 2), max(int(H / f), 2)
    s = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    s = cv2.GaussianBlur(s, (0, 0), 3.0)
    return cv2.resize(s, (W, H), interpolation=cv2.INTER_LINEAR)


# ============================================================================================ SDF core
@njit(cache=True, fastmath=True, inline='always')
def _sdf(px, py, pz, P, KL, lst, n, base, margin, lvmax=2):
    """Primitive rows: cx, cy, cz, rx, ry, rz, level. Levels are blended with exp smooth-min (per
    level k), then the levels are unioned with a polynomial smooth-min."""
    m0 = 1e9
    a0 = 0.0
    m1 = 1e9
    a1 = 0.0
    m2 = 1e9
    a2 = 0.0
    k0 = KL[0]
    k1 = KL[1]
    k2 = KL[2]
    for j in range(n):
        i = lst[j]
        lv = int(P[i, 6])
        if lv > lvmax:
            continue
        qx = (px - P[i, 0]) / P[i, 3]
        qy = (py - P[i, 1]) / P[i, 4]
        qz = (pz - P[i, 2]) / P[i, 5]
        rm = min(P[i, 3], min(P[i, 4], P[i, 5]))
        d = (math.sqrt(qx * qx + qy * qy + qz * qz) - 1.0) * rm
        if lv == 0:
            hh = max(k0 - abs(m0 - d), 0.0) / k0
            m0 = min(m0, d) - hh * hh * k0 * 0.25
            a0 = 1.0
        elif lv == 1:
            hh = max(k1 - abs(m1 - d), 0.0) / k1
            m1 = min(m1, d) - hh * hh * k1 * 0.25
            a1 = 1.0
        else:
            hh = max(k2 - abs(m2 - d), 0.0) / k2
            m2 = min(m2, d) - hh * hh * k2 * 0.25
            a2 = 1.0
    D = 1e9
    if a0 > 0:
        D = m0
    if a1 > 0:
        d1 = m1
        h = max(k1 * 1.5 - abs(D - d1), 0.0) / (k1 * 1.5)
        D = min(D, d1) - h * h * k1 * 1.5 * 0.25
    if a2 > 0:
        d2 = m2
        h = max(k2 * 1.5 - abs(D - d2), 0.0) / (k2 * 1.5)
        D = min(D, d2) - h * h * k2 * 1.5 * 0.25
    D = max(D, py - base)                       # flat base
    return min(D, margin)


@njit(cache=True, fastmath=True, inline='always')
def _sdf_at(px, py, pz, P, KL, T0, TL, ts, tw, th, base, margin, lvmax=2):
    tx = int(px / ts)
    ty = int(py / ts)
    if tx < 0 or ty < 0 or tx >= tw or ty >= th:
        return margin
    k = ty * tw + tx
    return _sdf(px, py, pz, P, KL, TL[k], T0[k], base, margin, lvmax)


@njit(cache=True, parallel=True, fastmath=True)
def _march(Hb, Wb, P, KL, T0, TL, ts, tw, th, base, margin, zmin, zmax, pix, order):
    """Primary rays. Returns out (Hb, Wb, 7): alpha, nx, ny, nz, z, dmin, owner billow id."""
    out = np.zeros((Hb, Wb, 7), np.float32)
    for yi in prange(Hb):
        y = order[yi]
        py = y + 0.5
        for x in range(Wb):
            px = x + 0.5
            k = int(py / ts) * tw + int(px / ts)
            n = T0[k]
            if n == 0:
                continue
            lst = TL[k]
            z = zmax
            dmin = 1e9
            zb = z
            hit = False
            for it in range(200):
                d = _sdf(px, py, z, P, KL, lst, n, base, margin)
                if d < dmin:
                    dmin = d
                    zb = z
                if d < 0.2 * pix:
                    hit = True
                    break
                z -= max(d * 0.9, 0.3 * pix)
                if z < zmin:
                    break
            if hit:
                out[y, x, 0] = 1.0
            else:
                a = 0.5 - dmin / (1.2 * pix)
                if a <= 0:
                    continue
                out[y, x, 0] = a
                z = zb
            for it in range(3):
                d = _sdf(px, py, z, P, KL, lst, n, base, margin)
                z -= d
            e = 0.6 * pix
            gx = _sdf(px + e, py, z, P, KL, lst, n, base, margin) - _sdf(px - e, py, z, P, KL, lst, n, base, margin)
            gy = _sdf(px, py + e, z, P, KL, lst, n, base, margin) - _sdf(px, py - e, z, P, KL, lst, n, base, margin)
            gz = _sdf(px, py, z + e, P, KL, lst, n, base, margin) - _sdf(px, py, z - e, P, KL, lst, n, base, margin)
            gl = math.sqrt(gx * gx + gy * gy + gz * gz) + 1e-9
            nx, ny, nz = gx / gl, gy / gl, gz / gl
            if nz < 0.0:
                nz = 0.0
                l2 = math.sqrt(nx * nx + ny * ny) + 1e-9
                nx /= l2
                ny /= l2
            out[y, x, 1] = nx
            out[y, x, 2] = ny
            out[y, x, 3] = nz
            out[y, x, 4] = z
            out[y, x, 5] = dmin
            best = 1e9
            bi = -1
            for j in range(n):
                i = lst[j]
                if int(P[i, 6]) != 1:
                    continue
                qx = (px - P[i, 0]) / P[i, 3]
                qy = (py - P[i, 1]) / P[i, 4]
                qz = (z - P[i, 2]) / P[i, 5]
                d = (math.sqrt(qx * qx + qy * qy + qz * qz) - 1.0) * min(P[i, 3], min(P[i, 4], P[i, 5]))
                if d < best:
                    best = d
                    bi = i
            if best > 3.0 * KL[1]:
                bi = -1
            out[y, x, 6] = bi
    return out


@njit(cache=True, parallel=True, fastmath=True)
def _light(G, q, P, KL, T0, TL, ts, tw, th, base, margin, pix, Lx, Ly, Lz, ao_r, sh_k, sh_max, order, Wb, Hb):
    """Secondary terms on a coarse grid (every q px) of the primary buffer G: ao, shadow, thickness."""
    hq, wq = (G.shape[0] + q - 1) // q, (G.shape[1] + q - 1) // q
    out = np.zeros((hq, wq, 3), np.float32)
    for yi in prange(hq):
        yq = order[yi]
        y = min(yq * q + q // 2, G.shape[0] - 1)
        py = y + 0.5
        for xq in range(wq):
            x = min(xq * q + q // 2, G.shape[1] - 1)
            if G[y, x, 0] <= 0.0:
                out[yq, xq, 0] = 1.0
                out[yq, xq, 1] = 1.0
                continue
            px = x + 0.5
            nx, ny, nz, z = G[y, x, 1], G[y, x, 2], G[y, x, 3], G[y, x, 4]
            occ = 0.0
            wsum = 0.0
            for j in range(1, 6):
                hh = ao_r * j / 5.0
                dd = _sdf_at(px + nx * hh, py + ny * hh, z + nz * hh, P, KL, T0, TL, ts, tw, th, base, margin, 1)
                w = 1.0 / j
                occ += w * max(hh - dd, 0.0) / hh
                wsum += w
            out[yq, xq, 0] = max(1.0 - occ / wsum, 0.0)
            res = 1.0
            t = sh_max * 0.03
            ox = px + nx * sh_max * 0.02
            oy = py + ny * sh_max * 0.02
            oz = z + nz * sh_max * 0.02
            for it in range(80):
                sx = ox + Lx * t
                sy = oy + Ly * t
                sz = oz + Lz * t
                if sy < 0 or sx < 0 or sx >= Wb or sy >= Hb:
                    break
                h = _sdf_at(sx, sy, sz, P, KL, T0, TL, ts, tw, th, base, margin, 0)
                r = 0.5 + sh_k * h / t
                if r <= 0.0:
                    res = 0.0
                    break
                if r < res:
                    res = r
                t += min(max(h, 0.7 * pix), margin)
                if t > sh_max:
                    break
            out[yq, xq, 1] = max(res, 0.0)
            k = int(py / ts) * tw + int(px / ts)
            zz = z
            tk = 0.0
            for it in range(20):
                zz -= 2.0 * pix * (1 + it * 0.6)
                if zz < -1e8:
                    break
                dd = _sdf(px, py, zz, P, KL, TL[k], T0[k], base, margin)
                if dd < 0:
                    tk = z - zz
                else:
                    break
            out[yq, xq, 2] = tk
    return out


def _tiles(P, KL, Wb, Hb, ts, margin):
    tw, th = int(math.ceil(Wb / ts)), int(math.ceil(Hb / ts))
    lists = [[] for _ in range(tw * th)]
    for i in range(P.shape[0]):
        cx, cy, rx, ry = P[i, 0], P[i, 1], P[i, 3], P[i, 4]
        mg = margin + 4 * KL[int(P[i, 6])]
        x0, x1 = int(max((cx - rx - mg) // ts, 0)), int(min((cx + rx + mg) // ts, tw - 1))
        y0, y1 = int(max((cy - ry - mg) // ts, 0)), int(min((cy + ry + mg) // ts, th - 1))
        for ty in range(y0, y1 + 1):
            for tx in range(x0, x1 + 1):
                lists[ty * tw + tx].append(i)
    mx = max(max(len(l) for l in lists), 1)
    TL = np.zeros((tw * th, mx), np.int32)
    T0 = np.zeros(tw * th, np.int32)
    for k, l in enumerate(lists):
        T0[k] = len(l)
        TL[k, :len(l)] = l
    return T0, TL, tw, th


# ============================================================================================ geometry
def _ell_sdf(p, P):
    """numpy: approximate union sdf of ellipsoids (hard min) at points p (n,3)."""
    q = (p[:, None, :] - P[None, :, :3]) / P[None, :, 3:6]
    rm = P[:, 3:6].min(1)
    return ((np.linalg.norm(q, axis=-1) - 1) * rm[None]).min(1)


def build(Hc, seed=3, detail=1.0):
    """Returns primitive table in tower units (u right, v up, w toward camera) scaled by Hc later."""
    rng = np.random.default_rng(seed)
    # ---- masses: (u, v, w, ru, rv, rw)
    M = [(-0.2, 0.07, 0.02, 0.25, 0.08, 0.22),    # foot, left
         (0.16, 0.065, 0.05, 0.24, 0.075, 0.22), # foot, right
         (0.4, 0.055, -0.08, 0.11, 0.06, 0.11),  # low trailing foot (sun side)
         (-0.14, 0.2, 0.0, 0.17, 0.15, 0.2),     # body, left head
         (0.1, 0.22, 0.02, 0.17, 0.16, 0.2),     # body, right head
         (-0.27, 0.3, 0.03, 0.1, 0.13, 0.13),    # left secondary turret
         (0.24, 0.3, -0.03, 0.1, 0.13, 0.12),    # right shoulder (sun side)
         (0.02, 0.46, -0.03, 0.17, 0.2, 0.18),   # column
         (0.05, 0.64, -0.06, 0.15, 0.17, 0.16),  # column, upper
         (0.08, 0.8, -0.09, 0.14, 0.13, 0.15),   # crown
         (0.12, 0.91, -0.12, 0.1, 0.08, 0.1),    # overshooting top
         ]
    rows = [[u, -v, w, ru, rv, rw, 0] for (u, v, w, ru, rv, rw) in M]
    A = np.array(rows, np.float64)

    def conv(a):   # (u,-v,w) internal -> keep y downwards (v negative up)
        return a

    # ---- medium billows: power-law radii, on the upper / outer surface of each mass
    med = []
    cand = []
    for mi, m in enumerate(A):
        cu, cv, cw, ru, rv, rw = m[:6]
        area = ru * rv + rv * rw + ru * rw
        n = int(900 * area * detail) + 40
        for _ in range(n * 6):
            th_ = rng.uniform(-math.pi - 1.0, 1.0)            # domes on the upper half + flanks
            ph_ = math.asin(rng.uniform(-0.15, 1.0))           # uniform over the visible hemisphere
            dx = math.cos(th_) * math.cos(ph_)
            dy = math.sin(th_) * math.cos(ph_)
            dz = math.sin(ph_)
            if dy > 0.25 and dz < 0.55:                         # lower sides only where they face us
                continue
            cand.append((mi, dx, dy, dz))
    rng.shuffle(cand)
    # three passes of decreasing size; a candidate is kept only where the mass surface is still bare
    for (rlo, rhi, lim) in ((0.12, 0.2, 45), (0.07, 0.12, 120), (0.04, 0.07, 300)):
        for (mi, dx, dy, dz) in cand:
            cu, cv, cw, ru, rv, rw = A[mi, :6]
            sp = np.array([cu + dx * ru, cv + dy * rv, cw + dz * rw])
            if _ell_sdf(sp[None], A[:, :6])[0] < -0.01:
                continue
            v_ = -sp[1]
            r = rng.uniform(rlo, rhi) * (1.15 - 0.4 * v_)
            r = min(r, 0.85 * min(ru, rv) + 0.02)
            if v_ < 0.12:
                r *= 0.75
            nrm = np.array([dx / ru, dy / rv, dz / rw])
            nrm /= np.linalg.norm(nrm)
            if rhi < 0.05 and -nrm[1] < 0.35:                   # finest pass: crowns only
                continue
            c = sp - nrm * r * rng.uniform(0.35, 0.55)
            if c[1] + r * 0.2 > 0:
                continue
            if med:
                B = np.array([[m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2]] for m in med])
                if _ell_sdf(sp[None], B)[0] < -0.12 * r:           # already covered by a billow
                    continue
                if _ell_sdf(c[None], B)[0] < -0.5 * r:
                    continue
            ax_ = (r * rng.uniform(1.0, 1.2), r * rng.uniform(0.78, 0.92), r * rng.uniform(0.9, 1.0))
            med.append((c, r, ax_))
            rows.append([c[0], c[1], c[2], ax_[0], ax_[1], ax_[2], 1])
            if len(med) > lim * detail:
                break
    STATS['med'] = len(med)
    # ---- sub-billows: smaller domes on the upper / outer faces of the medium billows (cauliflower)
    A1 = np.array(rows, np.float64)
    nmed = len(rows)
    sub = []
    cand = []
    for i in range(len(A), nmed):
        for _ in range(int(90 * detail)):
            th_ = rng.uniform(-math.pi - 0.5, 0.5)
            ph_ = math.asin(rng.uniform(-0.1, 1.0))
            cand.append((i, math.cos(th_) * math.cos(ph_), math.sin(th_) * math.cos(ph_), math.sin(ph_)))
    rng.shuffle(cand)
    for (i, dx, dy, dz) in cand:
        cu, cv, cw, ru, rv, rw = A1[i, :6]
        sp = np.array([cu + dx * ru, cv + dy * rv, cw + dz * rw])
        if _ell_sdf(sp[None], A1[:, :6])[0] < -0.004:
            continue
        v_ = -sp[1]
        if v_ < 0.08:
            continue
        r = rw * rng.uniform(0.28, 0.5)
        nrm = np.array([dx / ru, dy / rv, dz / rw])
        nrm /= np.linalg.norm(nrm)
        if not (-nrm[1] > 0.45 or abs(nrm[2]) < 0.45):          # only crowns and the silhouette
            continue
        c = sp - nrm * r * rng.uniform(0.3, 0.55)
        ok = True
        for (mc, mr) in sub:
            if np.linalg.norm(mc - c) < 0.75 * (mr + r):
                ok = False
                break
        if not ok:
            continue
        sub.append((c, r))
        rows.append([c[0], c[1], c[2], r * rng.uniform(1.0, 1.15), r * rng.uniform(0.82, 0.95), r, 1])
        if len(sub) > 650 * detail:
            break
    A2 = np.array(rows, np.float64)
    # ---- small cauliflower bumps: only where the surface faces up (sunlit crowns) or on the silhouette
    small = []
    cand = []
    for i in range(len(A2)):
        cu, cv, cw, ru, rv, rw = A2[i, :6]
        lvl = A2[i, 6]
        n = int((900 if lvl == 1 else 120) * ru * detail) + 4
        for _ in range(n * 4):
            th_ = rng.uniform(-math.pi, 0.0)
            ph_ = rng.uniform(-0.1, 0.9) * math.pi / 2
            cand.append((i, math.cos(th_) * math.cos(ph_), math.sin(th_) * math.cos(ph_), math.sin(ph_)))
    rng.shuffle(cand)
    for (i, dx, dy, dz) in cand:
        cu, cv, cw, ru, rv, rw = A2[i, :6]
        sp = np.array([cu + dx * ru, cv + dy * rv, cw + dz * rw])
        if _ell_sdf(sp[None], A2[:, :6])[0] < -0.004:
            continue
        v_ = -sp[1]
        nrm = np.array([dx / ru, dy / rv, dz / rw])
        nrm /= np.linalg.norm(nrm)
        nxy = nrm[:2] / (np.linalg.norm(nrm[:2]) + 1e-9)
        crown = False
        if not crown:
            if abs(nrm[2]) > 0.75:
                continue
            q = sp[:2] + nxy * 0.025
            cov = (((q[0] - A2[:, 0]) / A2[:, 3]) ** 2 + ((q[1] - A2[:, 1]) / A2[:, 4]) ** 2) < 1.0
            if cov.any():                                       # not on the outer silhouette
                continue
        if v_ < 0.1:
            continue
        r = 0.016 * (1 - rng.random()) ** (-1 / 1.8)
        r = min(r, 0.05) * (1.05 - 0.3 * v_)
        c = sp - nrm * r * rng.uniform(0.15, 0.35)
        ok = True
        for (mc, mr) in small:
            if np.linalg.norm(mc - c) < 0.75 * (mr + r):
                ok = False
                break
        if not ok:
            continue
        small.append((c, r))
        rows.append([c[0], c[1], c[2], r * rng.uniform(1.0, 1.25), r * rng.uniform(0.85, 1.0), r, 2])
        if len(small) > 500 * detail:
            break
    return np.array(rows, np.float64)


# ============================================================================================ render
PAL = dict(
    hot=_h('#fffaf2'), cap=_h('#fff6ee'), peach=_h('#fbe4dc'), rose=_h('#f2c9d2'), lav_l=_h('#d8d3f0'),
    lav=_h('#a6b2ea'), shd=_h('#7b8cd6'), deep=_h('#5b66b5'), bounce=_h('#aac8f4'), haze=_h('#c4e2f6'),
    rim=_h('#fff0cc'), backlit=_h('#b4c0ec'),
)


def _cache_path(key):
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out', 'cache')
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, 's01cu_' + hashlib.md5(key.encode()).hexdigest()[:16] + '.npz')


def raymarch(ax, base_y, Hc, pw, ph, sun, seed=3, detail=1.0):
    """March the cloud into a crop box of the plate. Returns (fields dict, box)."""
    T = build(Hc, seed, detail)
    P = T.copy()
    P[:, 0] = ax + T[:, 0] * Hc
    P[:, 1] = base_y + T[:, 1] * Hc
    P[:, 2] = T[:, 2] * Hc
    P[:, 3:6] = T[:, 3:6] * Hc
    x0 = int(max(np.min(P[:, 0] - P[:, 3]) - 0.05 * Hc, 0))
    x1 = int(min(np.max(P[:, 0] + P[:, 3]) + 0.05 * Hc, pw))
    y0 = int(max(np.min(P[:, 1] - P[:, 4]) - 0.05 * Hc, 0))
    y1 = int(min(base_y + 0.02 * Hc, ph))
    Pb = P.copy()
    Pb[:, 0] -= x0
    Pb[:, 1] -= y0
    Wb, Hb = x1 - x0, y1 - y0
    KL = np.array([0.08, 0.012, 0.004], np.float64) * Hc
    margin = 0.06 * Hc
    ts = max(int(0.03 * Hc), 8)
    key = f'{Hc:.2f}|{pw}|{ph}|{ax:.2f}|{base_y:.2f}|{sun}|{seed}|{detail}|v11|{KL.tolist()}|{LIGHT}|' + hashlib.md5(P.tobytes()).hexdigest()
    cp = _cache_path(key)
    if os.path.exists(cp):
        F = np.load(cp)['F']
    else:
        T0, TL, tw, th = _tiles(Pb, KL, Wb, Hb, ts, margin)
        zmax = float(np.max(Pb[:, 2] + Pb[:, 5]) + 0.02 * Hc)
        zmin = float(np.min(Pb[:, 2] - Pb[:, 5]) - 0.02 * Hc)
        L = np.array(LIGHT)
        L /= np.linalg.norm(L)
        rng = np.random.default_rng(0)
        import time as _t
        _t0 = _t.time()
        G = _march(Hb, Wb, Pb, KL, T0, TL, float(ts), tw, th, float(base_y - y0), margin, zmin, zmax, 1.0,
                   rng.permutation(Hb).astype(np.int64))
        print('march', _t.time() - _t0, 'maxlist', TL.shape, flush=True)
        _t0 = _t.time()
        q = 2
        hq = (Hb + q - 1) // q
        Lq = _light(G, q, Pb, KL, T0, TL, float(ts), tw, th, float(base_y - y0), margin, 1.0, L[0], L[1], L[2],
                    0.05 * Hc, 3.5, 0.8 * Hc, rng.permutation(hq).astype(np.int64), Wb, Hb)
        print('light', _t.time() - _t0, flush=True)
        Lf = cv2.resize(Lq, (Wb, Hb), interpolation=cv2.INTER_LINEAR)
        F = np.concatenate([G[..., :5], Lf[..., 0:2], G[..., 5:6], Lf[..., 2:3], G[..., 6:7]], -1)
        np.savez_compressed(cp, F=F)
    return F, (x0, y0, x1, y1), P


def render_tower(ax, base_y, Hc, pw, ph, sun, W, seed=3, P=PAL, detail=1.0):
    F, (x0, y0, x1, y1), prims = raymarch(ax, base_y, Hc, pw, ph, sun, seed, detail)
    bh, bw = F.shape[:2]
    alpha = F[..., 0].copy()
    nx, ny, nz = F[..., 1], F[..., 2], F[..., 3]
    ao = F[..., 5]
    shd = F[..., 6]
    thick = F[..., 8]
    ins = (alpha > 0).astype(np.float32)
    xs1, ys1 = np.meshgrid(np.arange(bw, dtype=np.float32) + x0, np.arange(bh, dtype=np.float32) + y0)
    v = np.clip((base_y - ys1) / Hc, 0, 1.2)
    # ---- lighting terms
    L0 = np.array(LIGHT, np.float32)
    L0 /= np.linalg.norm(L0)
    # big-form normal (masked blur of the surface normal) -> broad light / shadow planes
    sgf = 0.045 * Hc
    insb = fblur((alpha > 0.5).astype(np.float32), sgf) + 1e-4
    fnx, fny, fnz = [fblur(c * (alpha > 0.5), sgf) / insb for c in (nx, ny, nz)]
    fl = np.sqrt(fnx ** 2 + fny ** 2 + fnz ** 2) + 1e-6
    ndf = (fnx * L0[0] + fny * L0[1] + fnz * L0[2]) / fl
    ndl_loc = nx * L0[0] + ny * L0[1] + nz * L0[2]
    shs = fblur(shd, 0.03 * Hc)
    # depth steps: billow in front of the surface behind it
    zf = np.where(alpha > 0.5, F[..., 4], -1e6).astype(np.float32)
    kr = max(int(0.004 * W) | 1, 3)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))
    zmx = cv2.dilate(zf, ker)
    zmn = cv2.erode(np.where(alpha > 0.5, zf, 1e6).astype(np.float32), ker)
    inm = (alpha > 0.5).astype(np.float32)
    far_e = cv2.GaussianBlur(_ss(0.02 * Hc, 0.05 * Hc, zmx - zf) * inm, (0, 0), max(0.0012 * W, 0.6))
    near_e = cv2.GaussianBlur(_ss(0.004 * Hc, 0.015 * Hc, zf - zmn) * inm, (0, 0), max(0.0007 * W, 0.5))
    far_w = cv2.GaussianBlur(_ss(0.02 * Hc, 0.06 * Hc, cv2.dilate(zf, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3 * kr, 3 * kr))) - zf) * inm, (0, 0), 0.004 * W)
    # value: big form + local billow modelling, cast shade from the masses, contact shade behind billows
    # ---- big planes: form light (blurred normal) x cast shade of the masses -> 3 painted tones
    Vf = np.clip((ndf + 0.62 * (ndl_loc - ndf) + 0.4) / 1.4, 0, 1) * (0.78 + 0.22 * shs)
    Vf = Vf + 0.08 * _ss(0.15, 0.9, v) - 0.06
    Vf = cv2.GaussianBlur(Vf.astype(np.float32), (0, 0), max(0.002 * W, 0.6))
    # painterly irregularity: the tone boundaries wobble like brush edges (static, seeded)
    rngb = np.random.default_rng(seed + 21)
    bn = np.zeros_like(Vf)
    for cells, amp in ((0.02, 1.0), (0.008, 0.5)):
        gh, gw = max(int(bh * cells * 1000 / Hc) + 3, 4), max(int(bw * cells * 1000 / Hc * 0.6) + 3, 4)
        bn += amp * cv2.resize(rngb.standard_normal((gh, gw)).astype(np.float32), (bw, bh), interpolation=cv2.INTER_CUBIC)
    Vf = Vf + 0.022 * bn
    # remove small isolated islands of each tone (they read as holes / spots, not as form)
    amin = int((0.05 * Hc) ** 2)
    dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(int(0.004 * W) | 1, 3),) * 2)
    for thr in (0.32, 0.52, 0.7):
        for inv in (False, True):
            m = ((Vf < thr) if not inv else (Vf >= thr)) & (alpha > 0.5)
            ncc, lab, st_, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
            small = np.zeros(ncc, np.float32)
            small[1:] = (st_[1:, cv2.CC_STAT_AREA] < amin).astype(np.float32)
            isl = cv2.GaussianBlur(cv2.dilate(small[lab], dk), (0, 0), max(0.0015 * W, 0.6))
            Vf = Vf + ((thr + 0.05) - Vf if not inv else (thr - 0.05) - Vf) * np.clip(isl, 0, 1) * (alpha > 0.5)
    # per-billow tone: every billow takes (mostly) the mean light of its own area, so the terminator
    # steps along billow outlines (scalloped, painted) instead of cutting across them in a straight line
    bid = F[..., 9].astype(np.int64)
    okb = (bid >= 0) & (alpha > 0.5)
    nb = int(bid.max()) + 2
    cnt = np.bincount(bid[okb] + 1, minlength=nb).astype(np.float32)
    sm_ = np.bincount(bid[okb] + 1, weights=Vf[okb], minlength=nb).astype(np.float32)
    mean_b = sm_ / np.maximum(cnt, 1)
    Vb = np.where(okb, mean_b[np.clip(bid + 1, 0, nb - 1)], Vf)
    Vb = cv2.GaussianBlur(Vb.astype(np.float32), (0, 0), max(0.0008 * W, 0.5))
    Vf = 0.2 * Vb + 0.8 * Vf
    Vq = 0.18 + 0.3 * _ss(0.28, 0.37, Vf) + 0.3 * _ss(0.47, 0.57, Vf) + 0.12 * _ss(0.66, 0.74, Vf)
    V = 0.7 * Vq + 0.3 * Vf
    # ---- billow articulation inside the planes (soft painted marks, not a 3D render)
    nys = cv2.GaussianBlur(ny.astype(np.float32), (0, 0), max(0.0015 * W, 0.6))
    under = _ss(0.05, 0.75, nys)                                  # billow undersides
    crown = _ss(-0.15, -0.75, nys)                                # billow tops
    sidel = np.clip(ndl_loc - ndf, 0, 1)                          # local faces turned to the sun
    V = V - 0.3 * under * (0.4 + 0.6 * V) + 0.12 * crown + 0.1 * sidel
    V = V - 0.22 * far_w - 0.15 * far_e
    V = V * (0.9 + 0.1 * ao)
    V = np.clip(V, 0, 1)
    lit = V
    k_cap = _ss(0.66, 0.8, V)
    k_half = _ss(0.4, 0.55, V)
    stops = [(0.0, P['deep']), (0.22, P['shd']), (0.4, P['lav']), (0.55, P['lav_l']), (0.68, P['peach']),
             (0.8, P['cap']), (0.95, P['hot'] * 1.06)]
    col = np.empty(V.shape + (3,), np.float32)
    col[:] = stops[0][1]
    for (a0, c0), (a1, c1) in zip(stops[:-1], stops[1:]):
        kk = _ss(a0, a1, V)[..., None]
        col = col + (c1 - col) * kk * (V[..., None] >= a0)
    # bounce: faces turned down / sideways inside the shade pick up the bright sky and sunlit ground
    bnc = np.clip(ny * 0.9 + (1 - nz) * 0.4 - 0.25, 0, 1) * (1 - k_half)
    col = col + (P['bounce'] - col) * (bnc * 0.45)[..., None]
    # deep crevices only where the big masses meet
    crease = np.clip((0.7 - ao) / 0.45, 0, 1) * _ss(0.5, 0.2, V)
    col = col + (P['deep'] * 0.9 - col) * (crease * 0.4)[..., None]
    # the front billow's edge catches light (painted separation line) where it overlaps another
    col = col + (P['hot'] * 1.05 - col) * (near_e * (0.25 + 0.4 * k_half))[..., None]
    # ---- aerial perspective: lower third bluer, lower contrast
    hzk = _ss(0.42, 0.0, v) * 0.5 + 0.08
    colh = col + (P['haze'] - col) * hzk[..., None]
    col = colh
    # ---- backlight near the hidden sun: bodies facing us dim; thin edges glow; silver-gold rim
    m8 = (alpha > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(m8, cv2.DIST_L2, 5).astype(np.float32)
    dsun = np.sqrt((xs1 - sun[0]) ** 2 + (ys1 - sun[1]) ** 2) / W
    prox = np.exp(-dsun / 0.14)
    prox2 = np.exp(-dsun / 0.4)
    # silhouette facing the sun (screen-space outward normal = (nx, ny) at the rim)
    snl = np.sqrt(nx * nx + ny * ny) + 1e-6
    tsx, tsy = (sun[0] - xs1), (sun[1] - ys1)
    tsl = np.sqrt(tsx ** 2 + tsy ** 2) + 1e-3
    face = np.clip((nx * tsx + ny * tsy) / (snl * tsl), 0, 1)
    face = fblur(face * ins, 0.002 * W) / (fblur(ins, 0.002 * W) + 1e-4)
    front = np.clip(nz, 0, 1)
    dim = np.exp(-dsun / 0.22) * (0.3 + 0.7 * front) * _ss(0.003 * W, 0.025 * W, din) * 0.7
    col = col * (1 - dim[..., None]) + P['backlit'] * dim[..., None]
    # translucency: thin cloud (little thickness behind) near the sun lights up warm
    thin = np.exp(-thick / (0.05 * Hc)) * np.exp(-din / (0.018 * W))
    trans = np.clip(thin * prox * 1.1 + np.exp(-din / (0.008 * W)) * face * prox * 0.6, 0, 1)
    col = col + (P['rim'] * 1.25 - col) * (trans * 0.85)[..., None]
    rw = max(0.0022 * W, 1.0)
    rim = np.exp(-din / rw) * face * np.clip(np.exp(-dsun / 0.2) * 1.6 + prox2 * 0.4, 0, 1.2)
    col = col + (P['rim'] * 2.0 - col) * np.clip(rim, 0, 1)[..., None]
    # lit silhouette of upward-facing crowns elsewhere gets a thin bright edge
    upe = np.exp(-din / rw) * np.clip(-ny, 0, 1) * k_cap * 0.35
    col = col + (P['hot'] * 1.15 - col) * np.clip(upe, 0, 1)[..., None]
    # ---- torn transparent base: horizontal streaks eat the underside; flat hazy skirt spills out
    rng = np.random.default_rng(seed + 9)
    st = cv2.resize(rng.random((bh // 12 + 3, bw // 90 + 3)).astype(np.float32), (bw, bh), interpolation=cv2.INTER_CUBIC)
    st2 = cv2.resize(rng.random((bh // 5 + 3, bw // 40 + 3)).astype(np.float32), (bw, bh), interpolation=cv2.INTER_CUBIC)
    stn = np.clip(st * 0.6 + st2 * 0.4, 0, 1)
    db = (base_y - ys1) / Hc
    tear = _ss(-0.005, 0.07, db + (stn - 0.5) * 0.08)
    alpha = alpha * (0.25 + 0.75 * tear)
    alpha = alpha * _ss(-0.01, 0.02, db + (stn - 0.5) * 0.05)
    # skirt: wispy semi-transparent streaks extending sideways at the base
    m_ext = fblur(cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.3 * Hc) | 1, 3))).astype(np.float32),
                  0.04 * Hc)
    sk = _ss(0.09, 0.03, db) * _ss(-0.01, 0.015, db) * _ss(0.15, 0.6, m_ext) * _ss(0.4, 0.75, stn) * 0.55
    skc = P['haze'] * 0.94 + P['shd'] * 0.06
    a2 = alpha + sk * (1 - alpha)
    col = (col * alpha[..., None] + skc * (sk * (1 - alpha))[..., None]) / np.maximum(a2, 1e-4)[..., None]
    alpha = a2
    # faint halation just outside the sun-facing silhouette
    dout = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5).astype(np.float32)
    fo = fblur(face * ins, 0.004 * W) / (fblur(ins, 0.004 * W) + 1e-4)
    halo = np.exp(-dout / (0.005 * W)) * np.clip(fo * 1.5, 0, 1) * prox * 0.5 * (1 - alpha)
    a3 = alpha + halo
    col = (col * alpha[..., None] + P['rim'] * 1.3 * halo[..., None]) / np.maximum(a3, 1e-4)[..., None]
    alpha = np.clip(a3, 0, 1)
    out = np.zeros((ph, pw, 4), np.float32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = alpha
    litf = np.zeros((ph, pw), np.float32)
    litf[y0:y1, x0:x1] = k_cap
    return out, dict(lit=litf, box=(x0, y0, x1, y1))
