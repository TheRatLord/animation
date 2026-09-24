"""Hero cumulonimbus for s02_railway_crossing (round 4): painted two-plane cloud.

Geometry: a lobe hierarchy - a few BIG billow masses stacked into a leaning column, medium lobes on their
surfaces, and clusters of small cauliflower bumps concentrated on the sun-facing (upper-right) surfaces and
silhouette. Rasterised as irregular spheroids in a supersampled z-buffer (numba).

Painting (the Shinkai look, not a 3D render):
  * light value e = own-lobe N.L + whole-tower form N.L + height bias, then a HARD terminator splits every
    pixel into one of two planes: a warm near-white sunlit plane (#fff8ec) and a cool blue-grey shadow
    plane (#a7a6d6 -> #8e9cc8, deepened ~30% in the lower core and underside). Because the own-lobe normal
    changes discontinuously at lobe overlaps, the terminator is a crisp scalloped line that follows the
    lobes;
  * soft internal gradients inside each plane (warmer crown, cooler toward the terminator; sky fill on the
    upward faces of shadowed lobes), crevice separation lines from the height field;
  * a 2-3 px silver rim only on the sun-facing silhouette;
  * the anvil is a separate sheared plate: lit top, cool blue-grey underside band, wind-sheared fibrous
    cirrus texture fading out downwind, irregular (non-periodic) underside edge; it casts a shadow on the
    column top.
"""
import math
import numpy as np
import cv2
from numba import njit, prange


def hx(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def ss_(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def gblur(a, s):
    if s < 0.3:
        return a
    return cv2.GaussianBlur(a, (0, 0), s)


# lobe columns: cx, cy, cz, r, ay, clip_y, p1, p2, parent, level, wob
NL = 11
DK = 0.35


@njit(cache=True, parallel=True, fastmath=True)
def _zbuf(Hs, Ws, S, dk):
    n = S.shape[0]
    h1 = np.full((Hs, Ws), -1e12, np.float32)
    i1 = np.full((Hs, Ws), -1, np.int32)
    nx = np.zeros((Hs, Ws), np.float32)
    ny = np.zeros((Hs, Ws), np.float32)
    nz = np.zeros((Hs, Ws), np.float32)
    for y in prange(Hs):
        yf = y + 0.5
        for i in range(n):
            cx, cy, cz, r, ay, clip = S[i, 0], S[i, 1], S[i, 2], S[i, 3], S[i, 4], S[i, 5]
            ry = r * ay * 1.12
            if yf < cy - ry or yf > cy + ry or yf > clip:
                continue
            rx = r * 1.12
            x0 = max(0, int(cx - rx))
            x1 = min(Ws, int(cx + rx) + 1)
            p1, p2, wob = S[i, 6], S[i, 7], S[i, 10]
            for x in range(x0, x1):
                xf = x + 0.5
                u = (xf - cx) / r
                v = (yf - cy) / (r * ay)
                th = math.atan2(v, u)
                rr = 1.0 + wob * (0.6 * math.sin(3.0 * th + p1) + 0.4 * math.sin(5.0 * th + p2))
                d2 = (u * u + v * v) / (rr * rr)
                if d2 >= 1.0:
                    continue
                s = math.sqrt(1.0 - d2)
                z = cz + r * s * dk
                if z > h1[y, x]:
                    h1[y, x] = z
                    i1[y, x] = i
                    nx[y, x] = u / rr
                    ny[y, x] = -v / rr
                    nz[y, x] = s
    return h1, i1, nx, ny, nz


class Lobes:
    def __init__(self, rng):
        self.rows = []
        self.rng = rng

    def add(self, cx, cy, cz, r, ay=0.92, clip=1e9, parent=-1, level=0, wob=0.05):
        self.rows.append([cx, cy, cz, r, ay, clip, self.rng.uniform(0, 6.28), self.rng.uniform(0, 6.28),
                          parent, level, wob])
        return len(self.rows) - 1

    def arr(self):
        return np.array(self.rows, np.float64)


def _union_depth(D, shape, q=4):
    """Inside-distance (px, plate res) to the silhouette of the current union of lobes."""
    ph, pw = shape
    m = np.zeros((ph // q + 1, pw // q + 1), np.uint8)
    for r_ in D.rows:
        cv2.ellipse(m, (int(r_[0] / q), int(r_[1] / q)), (max(int(r_[3] / q), 1), max(int(r_[3] * r_[4] / q), 1)),
                    0, 0, 360, 255, -1)
    d = cv2.distanceTransform((m > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) * q
    return d, q


def _children(D, rng, parents, count, ratio, sun_ang, bias_sun=0.6, bias_up=0.4, clip=1e9, level=1, spread=1.0,
              front=0.35, only_facing=None, edge=None, interior=0.0):
    """Place child lobes on the visible surface of each parent (angles biased toward the sun and up).
    edge=(dist_map, q, k): keep a child only if its centre lies within k*rr of the union silhouette
    (cauliflower bumps live on the outline), except for a fraction `interior`."""
    rows = D.rows
    out = []
    for pi in parents:
        cx, cy, cz, r = rows[pi][0], rows[pi][1], rows[pi][2], rows[pi][3]
        n = int(count if not isinstance(count, tuple) else rng.integers(count[0], count[1] + 1))
        for k in range(n):
            # direction: mixture of uniform and biased to sun / up
            a = rng.uniform(-math.pi, math.pi)
            if rng.random() < bias_sun:
                a = sun_ang + rng.normal(0, 0.9 * spread)
            elif rng.random() < bias_up:
                a = -math.pi / 2 + rng.normal(0, 0.8 * spread)
            if only_facing is not None:
                # keep only children whose direction faces the sun (dot > only_facing)
                if math.cos(a - sun_ang) < only_facing:
                    continue
            rr = r * rng.uniform(ratio[0], ratio[1])
            dist = r * rng.uniform(0.72, 0.95)
            x = cx + math.cos(a) * dist
            y = cy + math.sin(a) * dist * 0.92
            if y > clip - rr * 0.3:
                continue
            if edge is not None:
                dm, q, kk = edge
                iy, ix = int(np.clip(y / q, 0, dm.shape[0] - 1)), int(np.clip(x / q, 0, dm.shape[1] - 1))
                if dm[iy, ix] > kk * rr and rng.random() > interior:
                    continue
            # z: sits on the parent's surface (dk-scaled sphere depth) and pops in front of it
            dd = min(dist / r, 0.99)
            z = cz + r * DK * math.sqrt(1 - dd * dd) + rr * DK * rng.uniform(0.1, 0.5) * front / 0.35
            out.append(D.add(x, y, z, rr, rng.uniform(0.84, 0.98), clip, pi, level, rng.uniform(0.03, 0.08)))
    return out


def build_tower(pw, ph, x0, base_y, col_top, anvil, seed=5, sun_xy=None):
    """Lobe hierarchy for the column (+ the overshooting dome, returned separately)."""
    rng = np.random.default_rng(seed)
    D = Lobes(rng)
    Hc = base_y - col_top
    sx, sy = sun_xy
    lean = 0.035 * pw

    def axis(v):
        return x0 + lean * v ** 1.3

    def hw(v):
        return pw * np.interp(v, [0, 0.12, 0.3, 0.5, 0.7, 0.86, 1.0],
                              [0.2, 0.18, 0.14, 0.125, 0.135, 0.15, 0.13])
    big = []
    v = 0.0
    row = 0
    # a few very big masses, rows of big billows between them (sizes vary a lot)
    while v < 1.0:
        h_ = hw(v)
        R = pw * rng.choice([0.03, 0.04, 0.05, 0.065, 0.08], p=[0.2, 0.3, 0.25, 0.15, 0.1])
        R = min(R, h_ * 0.75)
        n = max(1, int(round(2 * h_ / (R * 1.35))))
        for k in range(n):
            u = -1 + (2 * k + 1) / n + rng.uniform(-0.35, 0.35) / n
            Rk = R * rng.uniform(0.7, 1.25)
            x = axis(v) + u * max(h_ - Rk * 0.6, 0) + rng.uniform(-0.2, 0.2) * Rk
            y = base_y - v * Hc + rng.uniform(-0.15, 0.15) * Rk
            z = math.sqrt(max(1 - u * u, 0)) * h_ * 0.6 + rng.uniform(-0.1, 0.1) * Rk
            big.append(D.add(x, y, z, Rk, rng.uniform(0.82, 0.95), base_y, -1, 0, rng.uniform(0.02, 0.05)))
        v += R * rng.uniform(0.55, 0.75) / Hc
        row += 1
    # lateral shoulder billows at the base (wide, low)
    for (du, dv, rf) in [(-0.2, 0.05, 0.06), (0.21, 0.04, 0.055), (-0.16, 0.16, 0.05), (0.19, 0.2, 0.045),
                         (-0.27, 0.02, 0.04), (0.27, 0.0, 0.04)]:
        big.append(D.add(axis(dv) + du * pw, base_y - dv * Hc, rng.uniform(0.0, 0.03) * pw, rf * pw,
                         rng.uniform(0.8, 0.9), base_y, -1, 0, 0.04))
    # sun direction angle (screen, y down) from the tower centre
    sun_ang = math.atan2(sy - (base_y - 0.5 * Hc), sx - axis(0.5))
    dm = _union_depth(D, (ph, pw))
    mid = _children(D, rng, big, (6, 10), (0.3, 0.5), sun_ang, bias_sun=0.45, bias_up=0.5, clip=base_y, level=1,
                    edge=(dm[0], dm[1], 1.2), interior=0.18)
    dm = _union_depth(D, (ph, pw))
    small = _children(D, rng, mid, (4, 8), (0.25, 0.42), sun_ang, bias_sun=0.7, bias_up=0.5, clip=base_y,
                      level=2, spread=0.8, only_facing=-0.2, edge=(dm[0], dm[1], 0.9), interior=0.03)
    dm = _union_depth(D, (ph, pw))
    _children(D, rng, small, (2, 4), (0.3, 0.5), sun_ang, bias_sun=0.8, bias_up=0.5, clip=base_y, level=3,
              spread=0.7, only_facing=0.0, edge=(dm[0], dm[1], 0.8), interior=0.0)
    # overshooting dome poking above the anvil (separate set)
    O = Lobes(rng)
    ax_ = anvil['dome_x']
    oy = anvil['top'] + 0.01 * ph
    obig = [O.add(ax_, oy, 0.1 * pw, 0.05 * pw, 0.72, oy + 0.02 * ph, -1, 0, 0.03),
            O.add(ax_ - 0.045 * pw, oy + 0.012 * ph, 0.08 * pw, 0.036 * pw, 0.7, oy + 0.02 * ph, -1, 0, 0.03),
            O.add(ax_ + 0.04 * pw, oy + 0.014 * ph, 0.08 * pw, 0.032 * pw, 0.7, oy + 0.02 * ph, -1, 0, 0.03)]
    om = _children(O, rng, obig, (5, 8), (0.28, 0.45), sun_ang, bias_sun=0.5, bias_up=0.7, clip=oy + 0.02 * ph,
                   level=1, front=0.2)
    _children(O, rng, om, (2, 4), (0.28, 0.45), sun_ang, bias_sun=0.7, bias_up=0.6, clip=oy + 0.02 * ph, level=2,
              only_facing=0.0, front=0.2)
    return D, O, axis


def render_lobes(D, pw, ph, L, sun_xy, Wref, ss=2, form_sigma=0.05, form_k=1.0, vbias=None, noise_seed=3,
                 pal=None, rim=1.0, core_box=None, extra_shadow=None, dk=DK, w_own=0.32, w_form=0.8):
    """Rasterise + paint a lobe set. Returns straight-alpha RGBA (ph, pw, 4) and the light value map."""
    S = D.arr().copy()
    for k in (0, 1, 2, 3, 5):
        S[:, k] *= ss
    Hs, Ws = ph * ss, pw * ss
    h1, i1, nx, ny, nz = _zbuf(Hs, Ws, S, dk)
    cov = (i1 >= 0).astype(np.float32)
    # downsample to plate res (area average of normals / heights over covered subsamples)
    def dn(a):
        return cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)
    alpha = dn(cov)
    m = np.maximum(alpha, 1e-4)
    Nx = dn(nx * cov) / m
    Ny = dn(ny * cov) / m
    Nz = dn(nz * cov) / m
    hh = np.where(cov > 0, h1, 0).astype(np.float32)
    Hh = dn(hh) / m / ss
    lev = np.where(cov > 0, S[np.maximum(i1, 0), 9], 0).astype(np.float32)
    Lev = dn(lev) / m
    del h1, i1, nx, ny, nz
    Lv = np.asarray(L, np.float64)
    Lv = Lv / np.linalg.norm(Lv)
    ys = np.arange(ph, dtype=np.float32)[:, None] * np.ones((1, pw), np.float32)
    xs = np.arange(pw, dtype=np.float32)[None, :] * np.ones((ph, 1), np.float32)
    # ---- form normal from a heavily blurred silhouette (the whole tower as one volume)
    sgm = form_sigma * Wref
    B = gblur(alpha, sgm)
    gx = cv2.Sobel(B, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(B, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    kf = form_k * sgm * 2.2
    fx, fy, fz = -gx * kf, gy * kf, np.ones_like(gx)
    fl = np.sqrt(fx * fx + fy * fy + fz * fz)
    fx, fy, fz = fx / fl, fy / fl, fz / fl
    e_own = (Nx * Lv[0] + Ny * Lv[1] + Nz * Lv[2]) * (1.0 - 0.18 * Lev)
    e_form = fx * Lv[0] + fy * Lv[1] + fz * Lv[2]
    e = w_own * e_own + w_form * e_form
    if vbias is not None:
        f = np.clip((ys - vbias[0]) / (vbias[1] - vbias[0]), 0, 1)
        e = e + vbias[2] + (vbias[3] - vbias[2]) * f ** 1.3
    # low-frequency wander of the terminator (painted, not mechanical)
    nrng = np.random.default_rng(noise_seed)
    nw_, nh_ = max(pw // 24, 4), max(ph // 24, 4)
    nzs = gblur(nrng.standard_normal((nh_, nw_)).astype(np.float32), 0.02 * Wref / 24)
    nzs = cv2.resize(nzs / (nzs.std() + 1e-6), (pw, ph), interpolation=cv2.INTER_CUBIC)
    e = e + 0.05 * nzs
    if extra_shadow is not None:
        e = e - extra_shadow
    # crevices from the height field (lobes overlapped by a nearer lobe)
    hm = Hh * alpha
    s1 = 0.004 * Wref
    nb = gblur(hm, s1) / (gblur(alpha, s1) + 1e-4)
    ao = np.clip((nb - Hh) / (0.012 * Wref), 0, 1) * (alpha > 0.01)
    s2 = 0.015 * Wref
    nb2 = gblur(hm, s2) / (gblur(alpha, s2) + 1e-4)
    ao2 = np.clip((nb2 - Hh) / (0.05 * Wref), 0, 1) * (alpha > 0.01)
    # ---- the two planes, with a hard terminator (1-2 px anti-aliased)
    thr = 0.0
    lit = ss_(thr - 0.012, thr + 0.012, e - 0.25 * ao)
    P = pal
    # sunlit plane: warm near-white crown -> slightly cooler/greyer toward the terminator
    litc = P['lit'] + (P['lit_lo'] - P['lit']) * (ss_(0.3, 0.0, e) * 0.55)[..., None]
    litc = litc + (P['lit_hot'] - litc) * ss_(0.45, 0.8, e)[..., None]
    # soft separation of lit lobes: a cool lavender line where a lobe tucks under its neighbour
    litc = litc + (P['lit_crev'] - litc) * np.clip(ao * 1.8 + ao2 * 0.5, 0, 1)[..., None] * 0.9
    # shadow plane: sky fill on upward faces (lighter blue-grey), deeper toward the lower core/underside
    up = np.clip(Ny * 1.3, 0, 1)[..., None]
    down = np.clip(-Ny * 1.3, 0, 1)[..., None]
    shc = P['sh'] + (P['sh_up'] - P['sh']) * up * 0.8
    shc = shc + (P['sh_dn'] - shc) * down * 0.55
    if core_box is not None:
        cxc, cyc, rxc, ryc, amt = core_box
        core = np.exp(-(((xs - cxc) / rxc) ** 2 + ((ys - cyc) / ryc) ** 2))
        shc = shc + (P['sh_core'] - shc) * (core * amt)[..., None]
    shc = shc + (P['sh_crev'] - shc) * np.clip(ao * 1.4 + ao2 * 0.6, 0, 1)[..., None] * 0.7
    # value-deep shadow very close under the terminator (form shadow core), then reflected light
    near_t = ss_(-0.02, -0.12, e) * ss_(-0.45, -0.18, e)
    shc = shc * (1 - 0.06 * near_t)[..., None]
    col = shc + (litc - shc) * lit[..., None]
    # ---- silver rim on the sun-facing silhouette only (2-3 px at 1080p)
    sx, sy = sun_xy
    dxs, dys = sx - xs, sy - ys
    dl = np.sqrt(dxs * dxs + dys * dys) + 1e-3
    ex, ey = dxs / dl, dys / dl
    cg = gblur(alpha, max(0.0008 * Wref, 0.6))
    ogx = -cv2.Sobel(cg, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(cg, cv2.CV_32F, 0, 1, ksize=3)
    ogl = np.sqrt(ogx ** 2 + ogy ** 2) + 1e-6
    face = np.clip((ogx * ex + ogy * ey) / ogl, 0, 1) * (ogl > 1e-3)
    face = ss_(0.15, 0.7, gblur(face.astype(np.float32), max(0.0015 * Wref, 0.7)))
    din = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    rw = 0.0012 * Wref
    prox = np.exp(-(dl / Wref) / 0.35)
    rim_l = np.clip(ss_(rw * 2.2, rw * 0.6, din) * face * (0.55 + 0.6 * prox) * rim, 0, 1)
    col = col + (P['rim'] - col) * rim_l[..., None]
    return np.dstack([col, alpha]).astype(np.float32), dict(e=e, lit=lit, alpha=alpha, ao=ao)


def streak_noise(pw, ph, seed, cx=6.0, cy=110.0, octaves=3, angle=0.0):
    """Fibrous (strongly anisotropic) noise 0..1: long streaks along x."""
    rng = np.random.default_rng(seed)
    out = np.zeros((ph, pw), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        gx_, gy_ = int(cx * 2 ** o) + 2, int(cy * 2 ** o) + 2
        g = rng.random((gy_, gx_)).astype(np.float32)
        out += amp * cv2.resize(g, (pw, ph), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.55
    out /= tot
    if angle:
        M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (pw, ph), borderMode=cv2.BORDER_REFLECT)
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


def _prof_noise(n, seed, freqs=((3, 1.0), (7, 0.5), (17, 0.3), (41, 0.15))):
    """Non-periodic 1D profile noise in -1..1 over n samples (sum of random-phase smooth value noise)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    out = np.zeros(n)
    tot = 0
    for (f, a) in freqs:
        k = int(f * rng.uniform(0.8, 1.3)) + 3
        v = rng.uniform(-1, 1, k)
        out += a * np.interp(x * (k - 3) + rng.uniform(0, 1), np.arange(k), v)
        tot += a
    return (out / tot).astype(np.float32)


def paint_anvil(pw, ph, an, sun_xy, Wref, pal, seed=11, ss=2):
    """Anvil shelf: flat sheared plate; lit top + cool underside band + fibrous downwind tail."""
    x0, x1 = an['x0'], an['x1']
    top0, bot0 = an['top'], an['bot']
    dome_x = an['dome_x']
    Ws, Hs = pw * ss, ph * ss
    xs = (np.arange(Ws, dtype=np.float32) + 0.5) / ss
    u = (xs - x0) / (x1 - x0)                                    # 0 upwind .. 1 downwind
    # top edge: low dome over the column, sinking gently downwind, with irregular soft bumps upwind
    tn = _prof_noise(Ws, seed, ((5, 1.0), (13, 0.6), (31, 0.35), (70, 0.2)))
    top = (top0 + 0.03 * ph * (1 - np.exp(-((xs - dome_x) / (0.16 * pw)) ** 2)) * (1 - ss_(0.4, 0.9, u))
           + 0.012 * ph * ss_(0.4, 0.9, u) + tn * 0.008 * ph * (1.0 - 0.6 * ss_(0.4, 0.8, u)))
    # underside: rises toward the tail (shear), irregular non-periodic sag (no scallop rhythm)
    bn = _prof_noise(Ws, seed + 1, ((3, 1.0), (9, 0.7), (23, 0.45), (57, 0.25), (130, 0.12)))
    bot = (bot0 - 0.03 * ph * ss_(0.3, 1.0, u) + bn * 0.035 * ph + 0.012 * ph * np.exp(-((xs - dome_x) / (0.08 * pw)) ** 2))
    # irregular sagging bulges of the underside where the updraft spreads out (varied sizes, no rhythm)
    brng = np.random.default_rng(seed + 2)
    for _ in range(9):
        bxc = dome_x + brng.normal(0, 0.13) * pw
        bw_ = brng.uniform(0.012, 0.05) * pw
        bd_ = brng.uniform(0.006, 0.022) * ph * (1 - ss_(0.4, 0.8, (bxc - x0) / (x1 - x0)))
        bot = bot + bd_ * np.sqrt(np.clip(1 - ((xs - bxc) / bw_) ** 2, 0, 1))
    # upwind rounded cap
    cap = ss_(0.0, 0.06, u)
    mid = 0.5 * (top + bot)
    half = 0.5 * (bot - top) * np.sqrt(np.clip(cap * (2 - cap), 0, 1))
    top = mid - half
    bot = mid + half
    # thin out downwind
    thin = ss_(0.55, 1.0, u)
    top = top + (mid - top) * thin * 0.1
    bot = bot - (bot - mid) * thin * 0.2
    yy = (np.arange(Hs, dtype=np.float32)[:, None] + 0.5) / ss
    inside = (yy >= top[None, :]) & (yy <= bot[None, :]) & (u[None, :] >= 0) & (u[None, :] <= 1.0)
    alpha = cv2.resize(inside.astype(np.float32), (pw, ph), interpolation=cv2.INTER_AREA)
    X = np.arange(pw, dtype=np.float32)
    uu = ((X - x0) / (x1 - x0))[None, :] * np.ones((ph, 1), np.float32)
    topd = cv2.resize(top[None, :].astype(np.float32), (pw, 1), interpolation=cv2.INTER_AREA)[0]
    botd = cv2.resize(bot[None, :].astype(np.float32), (pw, 1), interpolation=cv2.INTER_AREA)[0]
    Y = np.arange(ph, dtype=np.float32)[:, None]
    v = np.clip((Y - topd[None, :]) / np.maximum(botd - topd, 1.0)[None, :], 0, 1)   # 0 top .. 1 underside
    # fibrous texture (wind-sheared cirrus), slight upward tilt downwind
    fib = streak_noise(pw, ph, seed + 5, cx=5.0, cy=70.0, octaves=4, angle=2.0)
    fib2 = streak_noise(pw, ph, seed + 6, cx=12.0, cy=160.0, octaves=3, angle=1.0)
    # fibrous dissolve toward the tail and at the soft underside edge downwind
    tail = ss_(0.45, 1.0, uu)
    a_f = np.clip((fib * 0.65 + fib2 * 0.35) - tail * 0.95 + 0.35, 0, 1)
    a_f = ss_(0.0, 0.45, a_f)
    alpha = alpha * (1 - tail + tail * a_f)
    # wispy streaks trailing beyond the tail
    over = ss_(0.95, 1.25, uu) * ss_(1.5, 1.0, uu)
    band = np.exp(-((Y - (topd[None, :] + botd[None, :]) * 0.5) / (0.02 * ph)) ** 2)
    xt = np.clip(np.round(x1 - x0 + x0).astype(int), 0, pw - 1)
    tailc = np.exp(-((Y - float(0.5 * (top[min(int(x1 * ss), Ws - 1)] + bot[min(int(x1 * ss), Ws - 1)]))) / (0.02 * ph)) ** 2)
    wisp = np.clip(fib2 * 1.4 - 0.55, 0, 1) * over * tailc * 0.7
    alpha = np.maximum(alpha, wisp)
    alpha = alpha * (uu > -0.02)
    # ---- shading: lit top band (wavy boundary), cool underside band with internal gradient
    lbn = cv2.resize(_prof_noise(Ws, seed + 4, ((4, 1.0), (11, 0.6), (29, 0.3)))[None, :], (pw, 1),
                     interpolation=cv2.INTER_AREA)[0][None, :]
    lb = 0.42 + 0.2 * lbn + 0.06 * (fib - 0.5) + 0.1 * ss_(0.3, 1.0, uu)
    litk = ss_(lb + 0.03, lb - 0.03, v)
    litc = pal['lit'] + (pal['lit_lo'] - pal['lit']) * (v / np.maximum(lb, 0.1))[..., None] * 0.6
    litc = litc + (pal['lit_hot'] - litc) * (ss_(0.35, 0.0, v) * ss_(0.7, 0.2, uu))[..., None] * 0.6
    und = pal['sh_up'] + (pal['sh_dn'] - pal['sh_up']) * ss_(lb, 1.0, v)[..., None]
    # underside is darkest right over the column (thickest ice), lighter and bluer out toward the tail
    ovc = np.exp(-((X[None, :] - dome_x) / (0.14 * pw)) ** 2) * ss_(lb, 1.0, v)
    und = und + (pal['sh_core'] - und) * (ovc * 0.55)[..., None]
    und = und + (pal['tail'] - und) * (ss_(0.4, 0.9, uu) * 0.3)[..., None]
    # reflected light along the very underside edge (cooler, lighter)
    und = und + (pal['sh_refl'] - und) * ss_(0.85, 1.0, v)[..., None] * 0.35
    col = und + (litc - und) * litk[..., None]
    # fibres modulate value along the streaks (stronger downwind)
    fm = (fib - 0.5) * (0.05 + 0.12 * tail) + (fib2 - 0.5) * 0.06 * tail
    col = col * (1 + fm)[..., None]
    # tail: translucent, mixes with the sky blue (thin ice)
    col = col + (pal['tail'] - col) * (tail * 0.35)[..., None]
    # silver edge on the sun-facing top
    din = cv2.distanceTransform((alpha > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    topedge = ss_(0.0012 * Wref * 2.2, 0.0012 * Wref * 0.5, din) * ss_(0.5, 0.2, v) * ss_(0.9, 0.5, uu)
    col = col + (pal['rim'] - col) * (topedge * 0.8)[..., None]
    return np.dstack([col, np.clip(alpha, 0, 1)]).astype(np.float32), dict(top=topd, bot=botd, alpha=alpha)


PAL = dict(
    lit=hx('#fffaf2'), lit_hot=np.array([1.04, 1.0, 0.95], np.float32), lit_lo=hx('#ece8f2'),
    lit_crev=hx('#c2c8e8'),
    sh=hx('#96a2d2'), sh_up=hx('#a6b2e0'), sh_dn=hx('#8090c2'), sh_core=hx('#6474ac'), sh_crev=hx('#7282b6'),
    sh_refl=hx('#b8c4e8'),
    rim=np.array([1.12, 1.1, 1.04], np.float32), tail=hx('#c8dcf4'),
)


def over(bot, top):
    ta = top[..., 3:4]
    ba = bot[..., 3:4] * (1 - ta)
    a = ta + ba
    rgb = (top[..., :3] * ta + bot[..., :3] * ba) / np.maximum(a, 1e-5)
    return np.concatenate([rgb, a], -1).astype(np.float32)


def hero_cumulonimbus(pw, ph, Wref, x0, base_y, anvil, sun_xy, seed=5, sky=None, horizon_y=None):
    """Full hero: column (+ anvil cast shadow) -> anvil -> overshooting dome. Returns RGBA plate."""
    col_top = anvil['bot'] + 0.01 * ph
    D, O, axis = build_tower(pw, ph, x0, base_y, col_top, anvil, seed=seed, sun_xy=sun_xy)
    L = (0.72, 0.62, 0.12)
    # anvil first (its alpha is needed for the cast shadow on the column)
    arg, ainf = paint_anvil(pw, ph, anvil, sun_xy, Wref, PAL, seed=seed + 7)
    ys = np.arange(ph, dtype=np.float32)[:, None]
    # cast shadow of the anvil on the column: band just below the underside, shifted down-left
    am = (ainf['alpha'] > 0.4).astype(np.float32)
    acc = np.zeros_like(am)
    for k in range(1, 9):
        d = 0.006 * Wref * k
        M = np.float32([[1, 0, -0.8 * d], [0, 1, 0.6 * d]])
        acc = np.maximum(acc, cv2.warpAffine(am, M, (pw, ph)) * (1 - 0.08 * k))
    csh = gblur(acc * (1 - am), 0.004 * Wref) * 0.55
    Hc = base_y - col_top
    rgba, inf = render_lobes(D, pw, ph, L, sun_xy, Wref, vbias=(col_top, base_y, 0.2, -0.5),
                             pal=PAL, core_box=(axis(0.3) - 0.04 * pw, base_y - 0.3 * Hc, 0.14 * pw, 0.2 * Hc, 0.85),
                             extra_shadow=csh)
    # billowy upwind top of the anvil: flattened lobes of strongly varied size, clipped inside the plate
    rng2 = np.random.default_rng(seed + 21)
    AL = Lobes(rng2)
    X_ = np.arange(pw, dtype=np.float32)
    x = anvil['x0'] + 0.03 * pw
    xend = anvil['x0'] + 0.62 * (anvil['x1'] - anvil['x0'])
    while x < xend:
        uu_ = (x - anvil['x0']) / (xend - anvil['x0'])
        r = pw * rng2.choice([0.03, 0.045, 0.065, 0.09], p=[0.25, 0.3, 0.3, 0.15]) * (1.0 - 0.45 * uu_)
        ix = int(np.clip(x, 0, pw - 1))
        tp, bt = float(ainf['top'][ix]), float(ainf['bot'][ix])
        ay = rng2.uniform(0.2, 0.3)
        cy = tp + r * ay * rng2.uniform(0.7, 0.95)
        clip = tp + (bt - tp) * rng2.uniform(0.3, 0.45)
        AL.add(x, cy, rng2.uniform(0, 0.02) * pw, r, ay, clip, -1, 0, rng2.uniform(0.02, 0.06))
        x += r * rng2.uniform(0.9, 1.6)
    sun_ang = math.atan2(sun_xy[1] - anvil['top'], sun_xy[0] - anvil['dome_x'])
    dmA = _union_depth(AL, (ph, pw))
    if False:
      _children(AL, rng2, list(range(len(AL.rows))), (1, 3), (0.25, 0.4), sun_ang, bias_sun=0.3, bias_up=0.8, only_facing=0.3,
              level=1, edge=(dmA[0], dmA[1], 1.0), front=0.2)
    lrgba, _ = render_lobes(AL, pw, ph, L, sun_xy, Wref, vbias=(anvil['top'] - 0.02 * ph, anvil['bot'], 0.3, -0.3),
                            pal=PAL, form_sigma=0.015, noise_seed=13, w_own=0.6, w_form=0.5)
    arg = over(arg, lrgba)
    orgba, _ = render_lobes(O, pw, ph, L, sun_xy, Wref, vbias=(anvil['top'] - 0.05 * ph, anvil['top'] + 0.03 * ph,
                                                                0.25, 0.0),
                            pal=PAL, form_sigma=0.02, noise_seed=9)
    out = over(rgba, arg)
    if False:
        out = over(out, orgba)
    # aerial haze toward the base (lower tower melts into the horizon haze)
    if sky is not None and horizon_y is not None:
        f = ss_(base_y - 0.3 * Hc, base_y + 0.02 * ph, ys)[..., None] * 0.3
        out[..., :3] = out[..., :3] + (sky[..., :3] - out[..., :3]) * f
    # torn wisps along the flat base
    rng = np.random.default_rng(seed + 3)
    fib = streak_noise(pw, ph, seed + 9, cx=8.0, cy=90.0, octaves=3)
    band = ss_(base_y - 0.01 * ph, base_y, ys) * ss_(base_y + 0.035 * ph, base_y + 0.005 * ph, ys)
    X = np.arange(pw, dtype=np.float32)[None, :]
    span = ss_(x0 - 0.3 * pw, x0 - 0.2 * pw, X) * ss_(x0 + 0.34 * pw, x0 + 0.24 * pw, X)
    wa = np.clip((fib - 0.45) * 2.2, 0, 1) * band * span
    wcol = PAL['sh_dn'] * 0.5 + PAL['sh_refl'] * 0.5
    wl = np.dstack([np.broadcast_to(wcol, (ph, pw, 3)), wa]).astype(np.float32)
    out = over(wl, out) if False else over(out, wl * np.array([1, 1, 1, 1], np.float32))
    return out, dict(axis=axis, col_top=col_top)
