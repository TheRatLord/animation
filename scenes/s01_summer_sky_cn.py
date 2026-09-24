"""s01 cumulonimbus, round 4: a designed hierarchy of ellipsoid lobes, painted per lobe.

Geometry
  * PRIMARY masses (few, big) stacked in stepped tiers: a wide flat-based foot, a narrower body with
    shoulders, an upper tower and an overshooting crown.
  * MEDIUM lobes grown on the upper / outer surface of every primary (never on the underside).
  * SMALL cauliflower bumps only along the lit upper rim (silhouette facing up / toward the sun).
Rendering
  * hard z-buffer of all ellipsoids (supersampled): each pixel belongs to exactly one lobe, so lobe
    overlaps are crisp painted edges.
  * shading normal = own lobe normal blended with its parent's and the whole-cloud form normal, lit from
    above/behind (the sun hides behind the crown): every lobe gets a lit cap and a terminator that
    follows its rounded underside.
  * short-range contact shadow marched along the light direction in the height field: the shadow is
    tucked directly under each lobe, continuous with its own terminator - never a floating blob.
  * painted colour ramp: warm cream/peach caps, cool lavender half tone, saturated blue-grey deep
    shadow, sky-blue bounce on undersides, lighter atmospheric shadows higher up, silver-gold rim near
    the sun, torn wispy underside.
"""
import math
import numpy as np
import cv2
from numba import njit


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


# ============================================================================================ raster
@njit(cache=True)
def _zbuf(Hs, Ws, S):
    """S rows: cx, cy, cz, r, ay, clip. Returns front depth Z and lobe index I."""
    Z = np.full((Hs, Ws), -1e9, np.float32)
    I = np.full((Hs, Ws), -1, np.int32)
    for i in range(S.shape[0]):
        cx, cy, cz, r, ay, clip = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4], S[i, 5]
        ry = r * ay
        x0 = max(int(cx - r) - 1, 0)
        x1 = min(int(cx + r) + 2, Ws)
        y0 = max(int(cy - ry) - 1, 0)
        y1 = min(int(min(cy + ry, clip)) + 2, Hs)
        for y in range(y0, y1):
            yc = y + 0.5
            if yc > clip:
                break
            dy = (yc - cy) / ry
            dy2 = dy * dy
            if dy2 >= 1.0:
                continue
            for x in range(x0, x1):
                dx = (x + 0.5 - cx) / r
                q = dx * dx + dy2
                if q < 1.0:
                    z = cz + r * math.sqrt(1.0 - q)
                    if z > Z[y, x]:
                        Z[y, x] = z
                        I[y, x] = i
    return Z, I


@njit(cache=True)
def _march(Z, inside, lx, ly, lz, step, n, soft):
    """Contact shadow: walk from each pixel toward the light (screen dir lx,ly; the ray rises lz per px
    of travel) and find how deeply the height field blocks it. Returns 0..1 shadow."""
    H, W = Z.shape
    out = np.zeros((H, W), np.float32)
    for y in range(H):
        for x in range(W):
            if not inside[y, x]:
                continue
            z0 = Z[y, x]
            occ = 0.0
            for k in range(1, n + 1):
                d = k * step
                sx = int(x + lx * d)
                sy = int(y + ly * d)
                if sx < 0 or sy < 0 or sx >= W or sy >= H:
                    break
                if not inside[sy, sx]:
                    continue
                h = Z[sy, sx] - (z0 + lz * d)
                if h > 0:
                    o = min(h / soft, 1.0) * (1.0 - 0.6 * k / n)
                    if o > occ:
                        occ = o
            out[y, x] = occ
    return out


# ============================================================================================ geometry
class Tower:
    """Builds the lobe table. Units: pixels of the target plate. Row: cx, cy, cz, r, ay, clip, parent,
    level, tone."""

    def __init__(self, ax, base_y, Hc, seed=1, sun_u=0.12):
        self.ax, self.base, self.Hc = ax, base_y, Hc
        self.rng = np.random.default_rng(seed)
        self.rows = []
        self.sun_u = sun_u
        self.sx, self.sw = 1.3, 1.12
        self.dome_uz = 0.3

    def P(self, u, v):
        return self.ax + u * self.Hc, self.base - v * self.Hc

    def add(self, x, y, z, r, ay, parent=-1, lev=0, clip=None):
        clip = self.base if clip is None else clip
        self.rows.append([x, y, z, r, ay, clip, parent, lev, self.rng.uniform(-1, 1)])
        return len(self.rows) - 1

    def finalize(self):
        for r in self.rows:
            if len(r) < 10:
                r.append(0.0)

    def primary(self, u, v, w, z=0.0, ay=0.86):
        u, w = u * self.sx, w * self.sw
        x, y = self.P(u, v)
        return self.add(x, y, z * self.Hc, w * self.Hc, ay)

    def grow(self, parents, count, ratio, lev, dzr=(0.05, 0.75), spread=0.75, sun_bias=0.0, rim_only=False,
             jitter=(0.75, 1.25), min_r=1.5, reach=(0.15, 0.4), side_lim=1.9, sil_only=False):
        """Children on the upper/outer surface of each parent (blue-noise directions)."""
        rng = self.rng
        kids = []
        if sil_only:
            A = np.array([r[:5] for r in self.rows], np.float64)
        for pi in parents:
            px, py, pz, pr, pay = self.rows[pi][:5]
            if sil_only:
                # only lobes whose crown lies on the cloud silhouette (no other lobe covers it)
                ok = False
                for ang in (-1.0, -0.5, 0.0, 0.5, 1.0):
                    qx = px + math.sin(ang) * pr * 1.02
                    qy = py - math.cos(ang) * pr * pay * 1.02
                    d = ((qx - A[:, 0]) / A[:, 3]) ** 2 + ((qy - A[:, 1]) / (A[:, 3] * A[:, 4])) ** 2
                    d[pi] = 9.0
                    if not (d < 1.0).any():
                        ok = True
                        break
                if not ok:
                    continue
            v = (self.base - py) / self.Hc
            rr = pr * ratio * (1 - 0.25 * v)
            if rr < min_r:
                continue
            n = int(count + rng.random())
            m = n * 10
            dz = rng.uniform(dzr[0], dzr[1], m)
            a = -math.pi / 2 + rng.normal(sun_bias, spread, m)
            a = np.clip(a, -math.pi / 2 - side_lim, -math.pi / 2 + side_lim)
            s_ = np.sqrt(np.clip(1 - dz * dz, 0, 1))
            D = np.stack([np.cos(a) * s_, np.sin(a) * s_, dz], 1)
            chosen = []
            thr = math.cos(min(1.15 * ratio + 0.06, 1.2))
            for k in range(m):
                if len(chosen) >= n:
                    break
                if all(float(D[k] @ D[c]) < thr for c in chosen):
                    chosen.append(k)
            for k in chosen:
                ux, uy, uz = D[k]
                r_ = rr * rng.uniform(*jitter)
                rch = pr - r_ * rng.uniform(*reach)
                x = px + ux * rch
                y = py + uy * rch * pay
                z = pz + uz * rch
                if y + r_ * 0.4 > self.base:
                    continue
                ay = min(pay * rng.uniform(0.95, 1.08), 1.0)
                # front-facing lobes are DOMES: only the upper part shows, the lower part merges into the
                # parent (no full circular outline -> no bubbles)
                i_ = self.add(x, y, z, r_, ay, pi, lev)
                self.rows[i_].append(float(_ss(self.dome_uz, self.dome_uz + 0.3, uz)))
                kids.append(i_)
        return kids

    def table(self):
        self.finalize()
        return np.array(self.rows, np.float64)


def build_tower(ax, base_y, Hc, seed=1, detail=1.0):
    """The designed cumulonimbus. u: horizontal offset from the axis (units of Hc, + = right = sun side),
    v: height above the base (units of Hc)."""
    T = Tower(ax, base_y, Hc, seed)
    prim = []
    # ---- foot: wide, flat-based, heavy masses (wider than the tower above), flattened
    for (u, v, w, z, ay) in [(-0.47, 0.06, 0.095, -0.14, 0.8), (-0.33, 0.08, 0.15, -0.02, 0.72),
                             (-0.12, 0.08, 0.19, 0.1, 0.7), (0.13, 0.08, 0.19, 0.12, 0.7),
                             (0.34, 0.08, 0.15, 0.02, 0.72), (0.49, 0.07, 0.09, -0.12, 0.8),
                             (0.58, 0.05, 0.06, -0.16, 0.8)]:
        prim.append(T.primary(u, v, w, z, ay=ay))
    # ---- lower body (tier 2): steps in from the foot
    for (u, v, w, z) in [(-0.26, 0.23, 0.15, 0.0), (-0.04, 0.26, 0.18, 0.08), (0.2, 0.24, 0.15, 0.04)]:
        prim.append(T.primary(u, v, w, z, ay=0.8))
    # ---- secondary turret on the left (lower, older, rounded top) -> a clear step in the silhouette
    for (u, v, w, z) in [(-0.3, 0.38, 0.11, -0.06), (-0.27, 0.48, 0.085, -0.08)]:
        prim.append(T.primary(u, v, w, z, ay=0.85))
    # ---- main column (tier 3), near-vertical sides, leaning slightly toward the sun
    for (u, v, w, z) in [(-0.08, 0.42, 0.15, 0.04), (0.13, 0.41, 0.13, 0.02), (-0.03, 0.57, 0.14, 0.02),
                         (0.14, 0.56, 0.11, 0.0), (0.02, 0.7, 0.13, 0.0), (-0.1, 0.66, 0.09, -0.04)]:
        prim.append(T.primary(u, v, w, z, ay=0.88))
    # ---- crown: overshooting dome and its shoulders
    for (u, v, w, z) in [(0.05, 0.83, 0.12, -0.02), (0.13, 0.785, 0.075, -0.04), (0.07, 0.93, 0.085, -0.06),
                         (-0.06, 0.8, 0.08, -0.05)]:
        prim.append(T.primary(u, v, w, z, ay=0.92))
    # ---- right sun-side turret (lower), gives the stepped silhouette
    for (u, v, w, z) in [(0.36, 0.22, 0.1, -0.05), (0.33, 0.32, 0.075, -0.07)]:
        prim.append(T.primary(u, v, w, z, ay=0.88))
    # medium lobes mostly on the upper silhouette of each primary (few on the front face)
    l1 = T.grow(prim, 18 * detail, 0.32, 1, dzr=(-0.1, 0.85), spread=0.9, sun_bias=0.15, reach=(0.15, 0.4),
                jitter=(0.6, 1.35))
    # small lobes: along the upper rims of the mediums
    l2 = T.grow(l1, 3 * detail, 0.42, 2, dzr=(-0.4, 0.1), spread=0.65, sun_bias=0.2, reach=(0.25, 0.45),
                side_lim=1.4, sil_only=True)
    # tiny cauliflower bumps only along the lit rim of the upper half
    upper = [i for i in l2 if (base_y - T.rows[i][1]) / Hc > 0.4]
    T.grow(upper, 2 * detail, 0.45, 3, dzr=(-0.25, 0.15), spread=0.5, sun_bias=0.35, reach=(0.3, 0.5),
           side_lim=1.2, sil_only=True)
    return T


# ============================================================================================ painting
PAL = dict(
    cap=_h('#fff2e0'), hot=_h('#fffbf4'), peach=_h('#ffe2cc'), lit_low=_h('#e4e0f0'),
    half=_h('#c3c4ea'), shd_hi=_h('#a0ace6'), shd=_h('#7080c8'), deep=_h('#4858a6'),
    bounce=_h('#9cc2f0'), base=_h('#6f7cc0'), rim=_h('#ffe6b4'), haze=_h('#cfe8f8'),
)


def render_tower(T, pw, ph, sun, W, ss=2, L=(0.72, -0.68, 0.06), P=PAL, sky=None, seed=1):
    """Rasterise + paint the tower into a straight-alpha RGBA plate (ph, pw, 4).
    Also returns a dict with 'edge' (sun-facing silhouette rim, 0..1) for light shafts."""
    S = T.table()
    Hc, base = T.Hc, T.base
    # crop box
    x0 = int(max(np.min(S[:, 0] - S[:, 3]) - 0.08 * Hc, 0))
    x1 = int(min(np.max(S[:, 0] + S[:, 3]) + 0.08 * Hc, pw))
    y0 = int(max(np.min(S[:, 1] - S[:, 3]) - 0.05 * Hc, 0))
    y1 = int(min(base + 0.08 * Hc, ph))
    bw, bh = x1 - x0, y1 - y0
    Ss = S[:, :6].copy()
    Ss[:, 0] = (Ss[:, 0] - x0) * ss
    Ss[:, 1] = (Ss[:, 1] - y0) * ss
    Ss[:, 2] *= ss
    Ss[:, 3] *= ss
    Ss[:, 5] = (Ss[:, 5] - y0) * ss
    Hs, Ws = bh * ss, bw * ss
    Z, I = _zbuf(Hs, Ws, Ss)
    ins = I >= 0
    insf = ins.astype(np.float32)
    Ii = np.where(ins, I, 0)
    ys, xs = np.mgrid[0:Hs, 0:Ws].astype(np.float32)
    xs += 0.5
    ys += 0.5

    def lobe_normal(idx):
        cx, cy, r, ay = Ss[idx, 0], Ss[idx, 1], Ss[idx, 3], Ss[idx, 4]
        dx = (xs - cx) / r
        dy = (ys - cy) / (r * ay)
        q = dx * dx + dy * dy
        k = np.where(q > 0.985, np.sqrt(0.985 / np.maximum(q, 1e-6)), 1.0)
        dx, dy = dx * k, dy * k
        nz = np.sqrt(np.clip(1 - dx * dx - dy * dy, 0, 1))
        n = np.stack([dx, dy / ay, nz], -1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    # form normal: blurred silhouette as a height field (big cylinder-ish shape)
    sig = 0.07 * Hc * ss
    fq = max(int(sig / 3), 1)
    sw_, sh_ = max(Ws // fq, 4), max(Hs // fq, 4)
    a_s = cv2.resize(insf, (sw_, sh_), interpolation=cv2.INTER_AREA)
    a_s = cv2.GaussianBlur(a_s, (0, 0), sig / fq)
    gx = cv2.Sobel(a_s, cv2.CV_32F, 1, 0, ksize=3) / 8.0 / fq
    gy = cv2.Sobel(a_s, cv2.CV_32F, 0, 1, ksize=3) / 8.0 / fq
    gx = cv2.resize(gx, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    gy = cv2.resize(gy, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    kf = sig * 2.2
    n_f = np.stack([-gx * kf, -gy * kf, np.ones_like(gx) * 0.9], -1)
    n_f /= np.linalg.norm(n_f, axis=-1, keepdims=True)
    n_f_keep = n_f
    par = S[:, 6].astype(np.int64)
    levs = S[:, 7].astype(np.int64)
    lev = levs[Ii]
    chain = [Ii]
    for _ in range(3):
        pc = par[chain[-1]]
        chain.append(np.where(pc >= 0, pc, chain[-1]))
    root = np.choose(np.clip(lev, 0, 3), chain)
    N = 0.3 * lobe_normal(root).astype(np.float32) + 0.7 * n_f
    # dome weight of the owning lobe: its lower half takes the parent's shading (only the cap reads)
    cyo, ro, ayo = Ss[Ii, 1], Ss[Ii, 3], Ss[Ii, 4]
    dyl = (ys - cyo) / (ro * ayo)
    wd = S[:, 9][Ii] * _ss(-0.15, 0.7, dyl)
    vpix = np.clip((base - (ys / ss + y0)) / Hc, 0, 1.2)
    wlev = 0.42 * (0.45 + 0.55 * _ss(0.08, 0.4, vpix))       # the shadowed foot is less articulated
    for j in (2, 1, 0):
        m = (lev > j)
        if not m.any():
            continue
        nj = lobe_normal(chain[j]).astype(np.float32)
        w = np.where(m, wlev, 0.0)
        if j == 0:
            w = w * (1 - wd)
        N = N * (1 - w[..., None]) + nj * w[..., None]
        N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-6
    del chain
    Lv = np.array(L, np.float32)
    Lv /= np.linalg.norm(Lv)
    ndl = N @ Lv
    ndl_f = n_f_keep @ Lv
    ny = N[..., 1]
    nzz = N[..., 2]
    del N
    v = np.clip((base - (ys / ss + y0)) / Hc, 0, 1.2)       # height fraction
    # ---- contact shadow (short, directional -> tucked under each lobe)
    q = 2
    Zs = cv2.resize(np.where(ins, Z, 0).astype(np.float32), (Ws // q, Hs // q), interpolation=cv2.INTER_AREA)
    insq = cv2.resize(insf, (Ws // q, Hs // q), interpolation=cv2.INTER_AREA) > 0.5
    l2 = math.hypot(L[0], L[1])
    lx, ly = L[0] / l2, L[1] / l2
    step = max(0.004 * Hc * ss / q, 1.0)
    shq = _march(Zs, insq, lx, ly, 0.22, step, 14, 0.012 * Hc * ss / q)
    sh = cv2.resize(shq, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
    sh = cv2.GaussianBlur(sh, (0, 0), 0.006 * Hc * ss) * insf
    # ---- crevice occlusion (recess just behind a lobe edge)
    Zin = np.where(ins, Z, 0).astype(np.float32)

    def wb(x, s):
        return fblur(x * insf, s) / (fblur(insf, s) + 1e-4)
    d1 = wb(Zin, 0.012 * Hc * ss)
    d2 = wb(Zin, 0.04 * Hc * ss)
    crev = np.clip((d1 - Zin) / (0.02 * Hc * ss), 0, 1) * 0.6 + np.clip((d2 - Zin) / (0.07 * Hc * ss), 0, 1) * 0.4
    crev = crev * insf
    # ---- light term: terminator threshold drops with height (tops fully lit, lower body in shade)
    thr = 0.3 - 0.36 * np.clip(v, 0, 1)
    le = ndl + 0.5 * (ndl_f - 0.15) - thr - 0.18 * sh - 0.1 * crev
    del n_f_keep
    # painted value bands: narrow transitions -> flat shapes instead of airbrushed balls
    lit = _ss(-0.045, 0.045, le)                       # crisp painted terminator
    capk = _ss(0.14, 0.22, le)                          # brightest cap
    hotk = _ss(0.34, 0.39, le) * _ss(0.45, 0.8, v)
    half = _ss(-0.15, -0.11, le) * (1 - lit)           # half-tone just below the terminator
    # small ISOLATED shadow islands inside lit areas read as holes -> lift them to lit/half-tone
    shm = ((lit < 0.5) & ins).astype(np.uint8)
    ncc, lab, st_, _ = cv2.connectedComponentsWithStats(shm, connectivity=8)
    amin = (0.035 * Hc * ss) ** 2
    small = np.zeros(ncc, np.float32)
    small[1:] = (st_[1:, cv2.CC_STAT_AREA] < amin).astype(np.float32)
    isl = cv2.GaussianBlur(small[lab], (0, 0), 0.003 * Hc * ss)
    lit = np.maximum(lit, isl * 0.75)
    half = half * (1 - isl)
    band = np.exp(-((le - 0.02) / 0.05) ** 2) * lit    # warm terminator band
    # ---- colours
    vv = v[..., None]
    nyb = cv2.GaussianBlur(ny.astype(np.float32), (0, 0), 0.01 * Hc * ss)
    upk = np.clip(-nyb * 0.8 + 0.2, 0, 1)[..., None]
    downk = (_ss(0.3, 0.4, ny) * 0.8)[..., None]
    shd = P['deep'] + (P['shd'] - P['deep']) * np.clip(0.1 + 0.95 * vv, 0, 1)       # higher = lighter
    shd = shd + (P['shd_hi'] - shd) * np.clip(upk * 0.6 + (vv - 0.5) * 0.6, 0, 0.8)
    shd = shd + (P['bounce'] - shd) * (downk * 0.55 * np.clip(1.2 - vv * 1.2, 0, 1))  # sky bounce underneath
    shd = shd + (P['deep'] * 0.9 - shd) * (crev * 0.3 + sh * 0.15)[..., None] * (1 - 0.4 * vv)
    shd = shd + (P['base'] - shd) * (_ss(0.12, 0.0, v) * 0.5)[..., None]             # flat dark base
    hal = P['half'] + (P['shd_hi'] - P['half']) * 0.3
    litc = P['lit_low'] + (P['cap'] - P['lit_low']) * np.clip(vv * 1.3, 0, 1)
    col = shd + (hal - shd) * (half * 0.75)[..., None]
    col = col + (litc - col) * lit[..., None]
    col = col + (P['cap'] * 1.03 - col) * (capk * 0.8)[..., None]
    col = col + (P['hot'] * 1.08 - col) * (hotk * 0.7)[..., None]
    col = col + (P['peach'] - col) * (band * 0.6)[..., None]
    # lit side facing us but turned slightly away -> faint lavender tint in the light (lobe separation)
    col = col + (P['half'] - col) * (lit * crev * 0.45)[..., None]
    col = col * (1 + 0.012 * wb(S[:, 8][Ii].astype(np.float32), 0.004 * Hc * ss))[..., None]
    del ndl, le
    # ---- downsample to plate resolution
    colp = cv2.resize(col * insf[..., None], (bw, bh), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(insf, (bw, bh), interpolation=cv2.INTER_AREA)
    colp = colp / np.maximum(alpha, 1e-4)[..., None]
    litp = cv2.resize(lit * insf, (bw, bh), interpolation=cv2.INTER_AREA)
    del col
    # ---- plate-res passes: rim / translucency near the sun, torn base, atmospheric haze
    xs1, ys1 = np.meshgrid(np.arange(bw, dtype=np.float32) + x0, np.arange(bh, dtype=np.float32) + y0)
    m8 = (alpha > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(m8, cv2.DIST_L2, 5).astype(np.float32)
    dout = cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5).astype(np.float32)
    ab = fblur(alpha, 0.004 * W)
    ogx = -cv2.Sobel(ab, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(ab, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx ** 2 + ogy ** 2) + 1e-6
    sx_, sy_ = sun[0] - xs1, sun[1] - ys1
    sl = np.sqrt(sx_ ** 2 + sy_ ** 2) + 1e-3
    face = np.clip((ogx * sx_ + ogy * sy_) / (ogl * sl), 0, 1)
    face = _ss(0.05, 0.7, fblur(face * (ogl > 1e-3), 0.0015 * W))
    upf = np.clip(-ogy / ogl, 0, 1)                                            # upward-facing edges
    prox = np.exp(-(sl / W) / 0.09)
    prox2 = np.exp(-(sl / W) / 0.35)
    vpl = np.clip((base - ys1) / Hc, 0, 1.2)
    edge_k = np.clip(face * (0.35 + 0.65 * prox) + upf * 0.35 * prox2 * _ss(0.3, 0.8, vpl), 0, 1.3)
    # backlight: near the hidden sun the cloud faces away from the light -> the body dims to a cool
    # translucent half-tone so the silver-gold rim can blaze against it
    bkl = np.exp(-(sl / W) / 0.16) * _ss(0.004 * W, 0.03 * W, din)
    colp = colp + (P['half'] * np.float32(1.02) - colp) * (bkl * 0.55)[..., None]
    wr = max(0.0022 * W, 1.0)
    rim_line = np.exp(-din / wr) * edge_k * (0.45 + 1.2 * prox) * np.clip(prox2 * 1.6, 0, 1)
    rimc = P['rim'] * np.float32(1.4)
    colp = colp + (rimc - colp) * np.clip(rim_line, 0, 1)[..., None]
    glow_in = np.exp(-din / (0.006 * W)) * face * prox * 0.5
    colp = colp + (rimc * 1.05 - colp) * np.clip(glow_in, 0, 0.85)[..., None]
    # torn, wispy underside: horizontal streak noise eats the bottom, a flat dark skirt spills sideways
    rng = np.random.default_rng(seed + 9)
    stw = max(bw // 12, 8)
    st = rng.random((bh // 14 + 3, stw // 3 + 3)).astype(np.float32)
    st = cv2.resize(st, (bw, bh), interpolation=cv2.INTER_CUBIC)
    st2 = cv2.resize(rng.random((bh // 30 + 3, stw // 8 + 3)).astype(np.float32), (bw, bh),
                     interpolation=cv2.INTER_CUBIC)
    stn = np.clip(st * 0.55 + st2 * 0.45, 0, 1)
    dist_base = (base - ys1) / Hc                                               # 0 at the base
    tear = _ss(0.0, 0.06, dist_base + (stn - 0.5) * 0.09)
    alpha = alpha * tear
    # skirt: wide, flat, dark layer at the base, torn into streaks
    skirt_top = base - 0.1 * Hc
    sk = _ss(skirt_top, skirt_top + 0.04 * Hc, ys1) * _ss(base + 0.03 * Hc, base - 0.02 * Hc, ys1)
    ext = fblur(cv2.dilate(m8, cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.16 * Hc) | 1, 3))).astype(np.float32),
                0.03 * Hc)
    sk = sk * _ss(0.2, 0.7, ext) * _ss(0.35, 0.65, stn + 0.1)
    skc = P['base'] + (P['bounce'] - P['base']) * 0.35
    a2 = alpha + sk * 0.85 * (1 - alpha)
    colp = (colp * alpha[..., None] + skc * (sk * 0.85 * (1 - alpha))[..., None]) / np.maximum(a2, 1e-4)[..., None]
    alpha = a2
    # atmospheric haze toward the base (distance), and a faint halation just outside sun-facing edges
    hz = _ss(0.12, 0.0, dist_base) * 0.25
    colp = colp + (P['haze'] - colp) * hz[..., None]
    halo = np.exp(-dout / (0.006 * W)) * face * prox * 0.55 * (1 - alpha)
    a3 = alpha + halo
    colp = (colp * alpha[..., None] + rimc * halo[..., None]) / np.maximum(a3, 1e-4)[..., None]
    alpha = np.clip(a3, 0, 1)
    out = np.zeros((ph, pw, 4), np.float32)
    out[y0:y1, x0:x1, :3] = colp
    out[y0:y1, x0:x1, 3] = alpha
    edge = np.zeros((ph, pw), np.float32)
    edge[y0:y1, x0:x1] = np.clip(rim_line, 0, 1)
    litf = np.zeros((ph, pw), np.float32)
    litf[y0:y1, x0:x1] = litp
    return out, dict(edge=edge, lit=litf, box=(x0, y0, x1, y1))
