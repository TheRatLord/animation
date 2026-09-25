"""Round-20 clouds for s07_comet_night: PAINTED yn_05 night cumulus + yn_02 flat-based twilight rows.

Earlier rounds failed two ways: posterized popcorn sprites (stepped value rings, pixel edges) and airbrushed
cotton puffs (one blob, thick white top rim, smeared gradient underside).  yn_05 clouds are neither: they are
broken CLUSTERS of rounded lobes with star-filled gaps and satellite scraps, the mass is mostly a light
blue-white, the lit plane is broad (not a rim), the shadow is a slightly deeper blue-grey painted in soft dabs
low in the mass, and the underside frays into dry-brush wisps.

  night: clumps of Gaussian lobes (smooth union) -> domain warp -> billow noise on the sky side (cauliflower)
         and wind-stretched tearing on the underside -> full-res anti-aliased silhouette from a signed distance
         (crisp ~1 px on top, feathered 5-12 px on the underside).  Paint: continuous base gradient + 2 painted
         accents (broad moonlit plane facing up / toward the comet, soft shadow dabs low in each clump), brush
         texture streaks along the wind.  Tiny crumbs are dropped; real satellites are placed on purpose.
  twilight rows (yn_02): lobes sitting on a flat base line, crisp cauliflower tops in cool violet shadow, the
         underside lit by the afterglow with a WIDE graded rim (hot peach where it faces the glow, pink, then
         fading into the violet body) whose strength falls off along the row away from the glow.
Returns two straight-RGBA plates (near night clouds, far twilight/cirrus) so they can drift at different rates.
"""
import math

import numpy as np
import cv2

from s07_comet_night_cloud19 import _ss, _noise, _fbm, _rot_noise, _gauss, _col_dist


def _billow(w, h, T, rng, s):
    return (0.5 * np.abs(_noise(w, h, 0.16 * T + 3, 0.14 * T + 3, rng)) +
            0.3 * np.abs(_noise(w, h, 0.07 * T + 2, 0.065 * T + 2, rng)) +
            0.2 * np.abs(_noise(w, h, 0.03 * T + 1.5 * s + 1, 0.028 * T + 1.5 * s + 1, rng)))


def _night_lobes(cl, rng):
    """Clumps of lobes along the cloud axis + satellite scraps.  Returns list of (x, y, rx, ry, ang, w)."""
    T, L = cl['T'], cl['L']
    a = math.radians(cl['ang'])
    c, sn = math.cos(a), math.sin(a)
    X, Y = cl['x'], cl['y']

    def wpos(u_, v_):
        return X + u_ * c - v_ * sn, Y + u_ * sn + v_ * c
    lobes = []
    nclump = cl.get('clumps', 3)
    us = np.sort(rng.uniform(-0.36, 0.36, nclump)) * L if nclump > 1 else np.array([0.0])
    for ci, u0 in enumerate(us):
        q = abs(u0) / (0.5 * L + 1)
        Rc = T * rng.uniform(0.34, 0.5) * (1 - 0.45 * q ** 1.3)
        v0 = rng.normal(0, 0.1 * T)
        nl = int(rng.integers(7, 12))
        for k in range(nl):
            du = rng.normal(0, 0.85) * Rc
            dv = rng.normal(0, 0.4) * Rc - 0.1 * Rc
            r = Rc * rng.uniform(0.3, 0.58) * (1 - 0.3 * min(abs(du) / (1.5 * Rc), 1))
            wx, wy = wpos(u0 + du, v0 + dv)
            lobes.append((wx, wy, r * rng.uniform(1.1, 1.5), r, a, 0.62))
        # a few florets on the top of the clump (cauliflower silhouette)
        for k in range(int(rng.integers(2, 5))):
            th = math.radians(-90 + rng.uniform(-70, 70))
            rf = Rc * rng.uniform(0.22, 0.38)
            wx, wy = wpos(u0 + math.cos(th) * Rc * 0.9, v0 + math.sin(th) * Rc * 0.55)
            lobes.append((wx, wy, rf * 1.15, rf, 0.0, 0.62))
    # satellites: small broken scraps around the periphery, mostly downwind and below
    for k in range(cl.get('sats', 5)):
        side = 1 if rng.random() < 0.7 else -1
        u_ = side * rng.uniform(0.42, 0.72) * L * 0.5 * 2 * 0.5 + side * 0.15 * L
        v_ = rng.uniform(-0.2, 0.6) * T
        r = T * rng.uniform(0.07, 0.15)
        cx, cy = wpos(u_, v_)
        for j in range(int(rng.integers(1, 4))):
            lobes.append((cx + rng.normal(0, 0.8) * r, cy + rng.normal(0, 0.4) * r, r * rng.uniform(1.0, 1.6),
                          r * rng.uniform(0.6, 1.0), a, 0.62))
    return lobes


def _tw_lobes(cl, rng):
    """Flat-based stratocumulus row: lobes sitting on a base line + cauliflower heads."""
    T, L = cl['T'], cl['L']
    a = math.radians(cl['ang'])
    c, sn = math.cos(a), math.sin(a)
    X, Y = cl['x'], cl['y']

    def wpos(u_, v_):
        return X + u_ * c - v_ * sn, Y + u_ * sn + v_ * c
    lobes = []
    N = max(int(L / (0.35 * T + 2)), 4)
    uu = np.linspace(-0.5, 0.5, N) * L + rng.normal(0, 0.12 * L / N, N)
    env = np.clip(1 - np.abs(uu / (0.5 * L)) ** 2.2, 0.05, 1) ** 0.6
    walk = rng.normal(0, 1, N).cumsum()
    walk = (walk - walk.mean()) / (walk.std() + 1e-6)
    for i in range(N):
        r = T * (0.3 + 0.12 * walk[i] + rng.uniform(-0.05, 0.08)) * env[i]
        r = max(r, 0.08 * T)
        wx, wy = wpos(uu[i], -0.55 * r)       # sits on the base (v = 0)
        lobes.append((wx, wy, r * rng.uniform(1.3, 1.9), r, a, 0.62))
        if rng.random() < 0.6 * env[i]:
            rf = r * rng.uniform(0.35, 0.6)
            th = math.radians(-90 + rng.uniform(-50, 50))
            wx, wy = wpos(uu[i] + math.cos(th) * r * 1.0, -0.55 * r + math.sin(th) * r * 0.8)
            lobes.append((wx, wy, rf * 1.2, rf, 0.0, 0.62))
    # thin base skirt so the row reads as one flat-based bank
    lobes.append((X, Y - 0.12 * T, 0.5 * L, 0.12 * T, a, 0.55))
    return lobes


def _field(lobes, pad):
    bx0 = int(min(p[0] - 1.9 * max(p[2], p[3]) for p in lobes) - pad)
    bx1 = int(max(p[0] + 1.9 * max(p[2], p[3]) for p in lobes) + pad)
    by0 = int(min(p[1] - 1.9 * max(p[2], p[3]) for p in lobes) - pad)
    by1 = int(max(p[1] + 1.9 * max(p[2], p[3]) for p in lobes) + pad)
    F = np.zeros((by1 - by0, bx1 - bx0), np.float32)
    for (x, y, rx, ry, a, wg) in lobes:
        _gauss(F, x - bx0, y - by0, rx, ry, a, wg, 'sum')
    return F, bx0, by0


def _sdf_alpha(F, thr, soft):
    gx = cv2.Sobel(F, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(F, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gn = np.sqrt(gx * gx + gy * gy) + 1e-5
    sd = (F - thr) / gn
    return _ss(-soft, soft, np.clip(sd, -30, 30)), gx / gn, gy / gn


def _scallop(F, thr, T, rng, face, scales=((0.11, 0.8), (0.055, 1.0), (0.028, 0.9)), s=1.0):
    """Cauliflower silhouette: stamp small round florets centred on the contour (where `face` allows)."""
    h, w = F.shape
    for (rf_, dens_) in scales:
        rf = max(rf_ * T, 1.6 * s + 1.0)
        edge = (np.abs(F - thr) < 0.035) & (face > 0.3)
        ys_, xs_ = np.nonzero(edge)
        if len(xs_) == 0:
            continue
        n = int(len(xs_) / (rf * 6.0) * dens_) + 1
        idx = rng.choice(len(xs_), size=min(n, len(xs_)), replace=False)
        G = np.zeros_like(F)
        for i in idx:
            r = rf * rng.uniform(0.7, 1.3)
            _gauss(G, float(xs_[i]), float(ys_[i]) + 0.25 * r, r * rng.uniform(1.0, 1.3), r, 0.0, 0.62, 'max')
        F = np.maximum(F, G)
    return F


def _drop_crumbs(F, thr, min_area):
    ins = (F > thr).astype(np.uint8)
    n_, lab, stt, _ = cv2.connectedComponentsWithStats(ins, connectivity=8)
    if n_ > 2:
        keep = stt[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = True
        drop = ((~keep[lab]) & (ins > 0)).astype(np.float32)
        if drop.any():
            drop = cv2.GaussianBlur(cv2.dilate(drop, np.ones((3, 3), np.uint8), iterations=2), (0, 0), 2.0)
            F = F - 0.6 * np.clip(drop * 3, 0, 1)
    return F


def _night(cl, rng, s, light_xy):
    T = cl['T']
    lobes = _night_lobes(cl, rng)
    F, bx0, by0 = _field(lobes, int(0.3 * T + 30))
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = math.radians(cl['ang'])
    # large-scale domain warp: irregular, wind-sheared forms (stronger horizontally)
    dxl = _noise(w, h, 0.5 * T, 0.4 * T, rng) * 0.1 * T + (yy - h / 2) * cl.get('shear', 0.12) * 0.5
    dyl = _noise(w, h, 0.5 * T, 0.4 * T, rng) * 0.05 * T
    F = cv2.remap(F, xx + dxl, yy + dyl, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    near = _ss(0.1, 0.35, F)                         # noise only acts near the mass (no stray specks/streaks)
    # star-filled gaps breaking the mass into clustered pieces
    hole = _fbm(w, h, 0.3 * T + 6, 0.22 * T + 6, 2, rng)
    F = F - 0.14 * _ss(0.8, 1.6, hole)
    Fb = cv2.GaussianBlur(F, (0, 0), 0.1 * T + 2)
    gyb = cv2.Sobel(Fb, cv2.CV_32F, 0, 1, ksize=3)
    gxb = cv2.Sobel(Fb, cv2.CV_32F, 1, 0, ksize=3)
    gnb = np.sqrt(gxb * gxb + gyb * gyb) + 1e-7
    upf = _ss(-0.2, 0.6, gyb / gnb)              # edge faces the sky above
    dnf = _ss(-0.1, 0.7, -gyb / gnb)             # edge faces down
    bil = _billow(w, h, T, rng, s)
    F = F + 0.16 * (bil - 0.72) * (0.5 + 0.5 * upf) * near
    F = _scallop(F, 0.5, T, rng, 0.4 + upf, s=s)
    # underside: wind-stretched dry-brush tearing (long thin wisps trailing along the wind)
    tear = _rot_noise(w, h, 0.5 * T + 10, 0.045 * T + 2, ang, 3, rng)
    tear2 = _rot_noise(w, h, 0.18 * T + 6, 0.025 * T + 1.5, ang, 2, rng)
    F = F + (0.05 * tear + 0.04 * tear2 - 0.02) * dnf * near
    thr = 0.5
    F = _drop_crumbs(F, thr, max((0.06 * T) ** 2, 70 * s * s))
    # silhouette: crisp on the lit top, feathered/lost on the underside
    A_top, nx, ny = _sdf_alpha(F, thr, 0.75)
    A_soft, _, _ = _sdf_alpha(F, thr - 0.01, 1.4 * s + 0.8)
    A = A_top * (1 - dnf) + A_soft * dnf
    # thin, translucent parts near the silhouette on the underside let the stars through
    dens = _ss(thr, thr + 0.35, F)
    ins = (F > thr - 0.02).astype(np.uint8)
    d_up = cv2.GaussianBlur(_col_dist(ins, True), (0, 0), sigmaX=0.04 * T + 2, sigmaY=0.012 * T + 1)
    d_dn = cv2.GaussianBlur(_col_dist(ins, False), (0, 0), sigmaX=0.04 * T + 2, sigmaY=0.012 * T + 1)
    vpos = d_up / (d_up + d_dn + 1.0)
    A = A * (1 - 0.18 * _ss(0.6, 1.0, vpos) * (1 - dens))
    # ---- paint
    lx = light_xy[0] - cl['x']
    ld = np.array([0.35 * np.sign(lx), -1.0], np.float32)
    ld /= np.linalg.norm(ld)
    off = 0.14 * T + 3
    M = np.float32([[1, 0, ld[0] * off], [0, 1, ld[1] * off]])
    Fh = cv2.GaussianBlur(np.clip(F, 0, 1.3), (0, 0), 0.05 * T + 1.5)
    Fsh = cv2.warpAffine(Fh, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT)
    facing = (Fh - Fsh)                               # >0 where the surface turns toward the light
    nz = _fbm(w, h, 0.2 * T + 4, 0.14 * T + 4, 3, rng)
    nzf = _fbm(w, h, 0.06 * T + 2, 0.05 * T + 2, 2, rng)
    lit_field = 2.2 * facing + 0.3 * (1 - _ss(0.0, 0.5 * T, d_up)) + 0.14 * nz + 0.06 * nzf
    # the light/shadow boundary is itself a painted cauliflower edge (lit florets overlapping the shade)
    bil2 = _billow(w, h, 0.7 * T, rng, s)
    lit = _ss(0.36, 0.44, lit_field + 0.22 * (bil2 - 0.72))
    shd_field = _ss(0.4, 0.95, vpos) + 0.25 * nz - 1.2 * np.clip(facing, 0, None) + 0.2 * (bil2 - 0.72)
    shd = _ss(0.48, 0.62, shd_field) * 0.6 + _ss(0.3, 0.9, shd_field) * 0.2   # painted dabs low in each clump
    tone = cl.get('tone', 1.0)
    LIT = np.array([0.8, 0.89, 1.0], np.float32) * tone
    MID = np.array([0.6, 0.69, 0.88], np.float32) * tone
    SHD = np.array([0.41, 0.49, 0.74], np.float32) * tone
    # base: continuous gentle gradient top -> bottom (no steps)
    g = np.clip(vpos, 0, 1)[..., None]
    col = MID * (1.04 - 0.1 * g)
    col = col * (1 - shd[..., None]) + SHD * shd[..., None]
    # the lit plane is not flat: faint lighter/darker sub-florets painted into it
    bil3 = cv2.GaussianBlur(_billow(w, h, 0.9 * T, rng, s), (0, 0), 1.5 * s + 0.5)
    litc = LIT[None, None, :] * (1.0 - 0.035 * np.clip(bil3 - 0.6, -0.6, 0.8)[..., None] / 0.3) -       0.05 * _ss(0.2, 0.9, vpos)[..., None] * np.array([0.3, 0.25, 0.1], np.float32)
    col = col * (1 - lit[..., None]) + litc * lit[..., None]
    # brush texture along the wind + faint cool translucency toward the thin underside
    brush = _rot_noise(w, h, 0.25 * T + 8, 0.04 * T + 2, ang, 2, rng)
    col = col * (1 + 0.02 * brush * (0.4 + 0.6 * (1 - lit)))[..., None]
    col = col * (1 - (1 - dens) * 0.06 * (1 - lit))[..., None] + \
        ((1 - dens) * 0.06 * (1 - lit))[..., None] * np.array([0.3, 0.4, 0.75], np.float32)
    return col, A, bx0, by0


def _twilight(cl, rng, s, sun_x, Wp):
    T, L = cl['T'], cl['L']
    lobes = _tw_lobes(cl, rng)
    F, bx0, by0 = _field(lobes, int(0.4 * T + 30))
    h, w = F.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = math.radians(cl['ang'])
    dxl = _noise(w, h, 0.5 * T, 0.4 * T, rng) * 0.08 * T
    dyl = _noise(w, h, 0.5 * T, 0.4 * T, rng) * 0.04 * T
    F = cv2.remap(F, xx + dxl, yy + dyl, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    # flat base: cut along the base line (slightly ragged, wind-stretched)
    by = cl['y'] - by0 + (xx + bx0 - cl['x']) * math.tan(ang)
    rag = _rot_noise(w, h, 0.6 * T + 10, 0.05 * T + 2, ang, 2, rng)
    base_cut = _ss(by + 0.05 * T + 0.05 * T * rag, by - 0.12 * T, yy)
    Fb = cv2.GaussianBlur(F, (0, 0), 0.1 * T + 2)
    gyb = cv2.Sobel(Fb, cv2.CV_32F, 0, 1, ksize=3)
    gxb = cv2.Sobel(Fb, cv2.CV_32F, 1, 0, ksize=3)
    gnb = np.sqrt(gxb * gxb + gyb * gyb) + 1e-7
    upf = _ss(-0.2, 0.6, gyb / gnb)
    bil = _billow(w, h, T, rng, s)
    F = F + 0.22 * (bil - 0.72) * (0.3 + 0.7 * upf)
    F = F * (0.35 + 0.65 * base_cut) + 0.0
    thr = 0.5
    F = _drop_crumbs(F, thr, max((0.08 * T) ** 2, 60 * s * s))
    A, nx, ny = _sdf_alpha(F, thr, 0.8)
    A2, _, _ = _sdf_alpha(F, thr, 2.2 * s + 1.0)
    dnf = _ss(-0.1, 0.7, -gyb / gnb)
    A = A * (1 - dnf) + A2 * dnf                     # base edge a touch softer than the crisp tops
    ins = (F > thr).astype(np.uint8)
    d_up = cv2.GaussianBlur(_col_dist(ins, True), (0, 0), sigmaX=0.05 * T + 2, sigmaY=0.02 * T + 1)
    d_dn = cv2.GaussianBlur(_col_dist(ins, False), (0, 0), sigmaX=0.05 * T + 2, sigmaY=0.02 * T + 1)
    vpos = d_up / (d_up + d_dn + 1.0)
    nz = _fbm(w, h, 0.25 * T + 4, 0.1 * T + 3, 3, rng)
    # horizontal falloff: hot where the row faces the glow, fading to nothing toward the far end
    X = xx + bx0
    u = (X - cl['x']) / (0.5 * L)
    sunk = np.exp(-((X - sun_x) / (0.38 * Wp)) ** 2).astype(np.float32)
    endf = np.clip(1 - np.abs(u) ** 2.5, 0, 1)
    warm = cl['warm'] * (0.42 + 0.58 * sunk) * (0.25 + 0.75 * endf)
    # wide, graded rim from the base upward (width varies along the row)
    Rw = 0.2 * T * np.clip(1.0 + 0.35 * nz, 0.45, 1.7) * (0.6 + 0.6 * sunk)
    q = d_dn / Rw
    hot = np.exp(-q * 1.6) * warm
    pinkw = (1 - _ss(0.2, 2.6, q + 0.4 * nz)) * warm
    VTOP = np.array([0.3, 0.27, 0.48], np.float32)
    VBODY = np.array([0.43, 0.31, 0.52], np.float32)
    PINK = np.array([0.95, 0.52, 0.62], np.float32)
    PEACH = np.array([1.45, 0.78, 0.45], np.float32)
    col = VTOP * (1 - vpos[..., None]) + VBODY * vpos[..., None]
    # faint cool light reading the top lobes (no outline)
    ltw = (1 - _ss(0.0, 0.2 * T, d_up)) * _ss(0.1, 0.7, upf) * 0.22
    col = col * (1 - ltw[..., None]) + np.array([0.5, 0.5, 0.74], np.float32) * ltw[..., None]
    # pink-orange afterglow on the lower lobes (painted, broken by the lobe noise)
    lowl = _ss(0.5, 0.95, vpos + 0.15 * nz) * warm * 0.4
    pk = np.clip(np.maximum(pinkw * 0.8, lowl), 0, 1)[..., None]
    col = col * (1 - pk) + PINK * pk
    hk = np.clip(hot, 0, 1)[..., None]
    hc = PEACH * (0.7 + 0.3 * sunk[..., None]) + np.array([-0.1, -0.1, 0.12], np.float32) * (1 - sunk[..., None])
    col = col * (1 - hk) + hc * hk
    return col, A, bx0, by0


def clouds20(Wp, Hp, clusters, H, light_xy, sky, tail_glow=None, seed=20, sun_x=None):
    s = H / 1080.0
    plates = [np.zeros((Hp, Wp, 4), np.float32), np.zeros((Hp, Wp, 4), np.float32)]
    tg = None
    if tail_glow is not None:
        q = 8
        sm = cv2.resize(tail_glow, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        tg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 0.03 * H / q), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    if sun_x is None:
        sun_x = 0.3 * Wp
    order = sorted(range(len(clusters)), key=lambda i: (-clusters[i].get('haze', 0.0), clusters[i]['y']))
    for ic in order:
        cl = clusters[ic]
        rng = np.random.default_rng(seed * 1000 + ic * 11 + 5)
        if cl['kind'] == 'tw':
            col, A, bx0, by0 = _twilight(cl, rng, s, sun_x, Wp)
        else:
            col, A, bx0, by0 = _night(cl, rng, s, light_xy)
        h, w = A.shape
        xa, xb = max(bx0, 0), min(bx0 + w, Wp)
        ya, yb = max(by0, 0), min(by0 + h, Hp)
        if xb <= xa or yb <= ya:
            continue
        sl = (slice(ya - by0, yb - by0), slice(xa - bx0, xb - bx0))
        cc = col[sl]
        haze = float(cl.get('haze', 0.0))
        if haze > 0:
            skc = sky[ya:yb, xa:xb]
            cc = cc * (1 - haze * 0.4) + (skc * 1.05 + 0.02) * (haze * 0.4)
        if tg is not None:
            tpk = np.clip(tg[ya:yb, xa:xb], 0, 0.5)
            cc = cc * (1 - 0.25 * np.clip(tpk * 3, 0, 1)) + tpk * 0.7
        aa = (A[sl] * cl.get('amt', 1.0))[..., None]
        out = plates[cl.get('grp', 0)]
        dst = out[ya:yb, xa:xb]
        a0 = dst[..., 3:4]
        na = aa + a0 * (1 - aa)
        dst[..., :3] = (cc * aa + dst[..., :3] * a0 * (1 - aa)) / np.maximum(na, 1e-5)
        dst[..., 3:4] = na
    for out in plates:
        a = out[..., 3]
        q = 4
        pm = cv2.resize(out[..., :3] * a[..., None], (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        am = cv2.resize(a, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        pm = cv2.GaussianBlur(pm, (0, 0), 3)
        am = cv2.GaussianBlur(am, (0, 0), 3)[..., None]
        bl = cv2.resize(pm / np.maximum(am, 1e-4), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        wgt = _ss(0.0, 0.2, a)[..., None]
        out[..., :3] = out[..., :3] * wgt + bl * (1 - wgt)
    return plates


def layout20(Wp, W, H, my):
    """Frame fractions at t=0 (the camera tilts down ~0.5 H during the shot)."""
    def fp(fx, fy):
        return Wp / 2 + (fx - 0.5) * W, my + fy * H
    out = []

    def add(fx, fy, L, T, ang, kind='cu', grp=0, **kw):
        x, y = fp(fx, fy)
        d = dict(x=x, y=y, L=L * W, T=T * H, ang=ang, kind=kind, grp=grp)
        d.update(kw)
        out.append(d)
    # near night clouds (yn_05): big mass partly off-frame top-left streaming toward the comet
    add(0.08, 0.05, 0.3, 0.22, 8, clumps=4, sats=6, shear=0.15)
    add(0.3, 0.2, 0.2, 0.11, 22, clumps=3, sats=6, shear=0.2, tone=0.97)
    add(0.44, 0.31, 0.1, 0.06, 20, clumps=2, sats=4, shear=0.2, tone=0.95)
    # the clump crossing the comet tail
    add(0.53, 0.43, 0.13, 0.085, 4, clumps=4, sats=6, shear=0.12, tone=0.9)
    # upper right: a big broken bank partly off the top, scraps below it
    add(0.9, 0.07, 0.26, 0.17, 4, clumps=3, sats=6, shear=0.12)
    add(0.77, 0.29, 0.12, 0.06, 14, clumps=2, sats=4, shear=0.18, tone=0.94)
    # small scattered wisps
    add(0.16, 0.44, 0.07, 0.04, 10, clumps=2, sats=3, shear=0.2, tone=0.9, amt=0.9)
    add(0.66, 0.2, 0.06, 0.035, 12, clumps=1, sats=2, shear=0.2, tone=0.92, amt=0.9)
    # far twilight rows (yn_02), slower drift
    add(0.22, 0.815, 0.34, 0.075, 0, 'tw', 1, warm=1.0, haze=0.3)
    add(0.89, 0.83, 0.24, 0.085, 0, 'tw', 1, warm=0.95, haze=0.35)
    add(0.63, 0.875, 0.14, 0.035, 0, 'tw', 1, warm=0.85, haze=0.45)
    add(0.43, 0.9, 0.12, 0.028, 0, 'tw', 1, warm=0.9, haze=0.5)
    add(0.06, 0.9, 0.12, 0.03, 0, 'tw', 1, warm=0.9, haze=0.5)
    return out
