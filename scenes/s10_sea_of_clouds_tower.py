"""Painted towering cumulus / cumulonimbus for s10 (sunrise, sun low on the horizon BETWEEN the towers;
adapted from the s03 lobe painter: backlit, inner faces rim-lit, outer faces in cool shadow).

Shape: a cluster of cauliflower turrets built from a hierarchy of 3D lobes of very different sizes
(big masses -> medium bumps on their surfaces -> small beads on the sunward silhouette). The pixel
coordinates are domain-warped before the lobes are rasterised, so no lobe outline is a clean circle.
A hard z-buffer picks the front-most lobe per sample -> crisp internal contours where one lobe sits
in front of another.

Paint: flat-ish lavender shadow planes with a HARD terminator (no airbrushed per-lobe radial
gradient), sunlit tops only above the earth's-shadow line (decided per lobe, so the line follows the
lobes), a narrow saturated coral band on the terminator, and a crisp 2-3 px hot gold/white rim on
every edge (outer or internal) that faces the sun. Torn horizontal wisps at the base, a sheared
fibrous anvil blowing to the right.
"""
import math
import numpy as np
import cv2
from numba import njit, prange

from lib import core as C


# lobe columns
LX, LY, LZ, LR, LAY, LLEV, LPAR, LLIT = range(8)
NL = 8


@njit(cache=True, parallel=True, fastmath=True)
def _raster(L, bb, Hs, Ws, ss, x_off, y_off, wx, wy):
    """Front-most lobe per sample. wx/wy: warp offsets (Hs, Ws) in frame px.
    returns idx (Hs, Ws) int32 (-1 = empty), depth (Hs, Ws)"""
    idx = np.full((Hs, Ws), -1, np.int32)
    dep = np.full((Hs, Ws), -1e9, np.float32)
    n = L.shape[0]
    for py in prange(Hs):
        for px in range(Ws):
            x = x_off + (px + 0.5) / ss + wx[py, px]
            y = y_off + (py + 0.5) / ss + wy[py, px]
            best = -1e9
            bi = -1
            for i in range(n):
                if x < bb[i, 0] or x > bb[i, 2] or y < bb[i, 1] or y > bb[i, 3]:
                    continue
                r = L[i, LR]
                dx = (x - L[i, LX]) / r
                dy = (y - L[i, LY]) / (r * L[i, LAY])
                d2 = dx * dx + dy * dy
                if d2 >= 1.0:
                    continue
                z = L[i, LZ] + r * math.sqrt(1.0 - d2)
                if z > best:
                    best = z
                    bi = i
            idx[py, px] = bi
            dep[py, px] = best
    return idx, dep


def _turret(rng, lobes, xc, yb, yt, hw, zc, lean=0.0, big=(0.05, 0.075), W=1920):
    """Stack of big masses from base yb up to top yt (px, yt < yb), half-width hw at the base."""
    y = yb - big[0] * W * 0.6
    first = len(lobes)
    while y > yt + big[0] * W * 0.35:
        u = (yb - y) / max(yb - yt, 1)
        w = hw * (1.0 - 0.45 * u ** 1.6)
        r = rng.uniform(*big) * W * (1.0 - 0.35 * u)
        n = max(1, int(round(2 * w / (1.3 * r))))
        for k in range(n):
            if n > 1:
                x = xc + lean * (yb - y) - w + r * 0.8 + (2 * w - 1.6 * r) * (k + rng.uniform(0.3, 0.7)) / n
            else:
                x = xc + lean * (yb - y) + rng.uniform(-0.2, 0.2) * w
            lobes.append([x, y + rng.uniform(-0.25, 0.25) * r, zc + rng.uniform(-0.3, 0.3) * r, r,
                          rng.uniform(0.82, 0.95), 0, -1, 0])
        y -= r * rng.uniform(0.55, 0.8)
    # cauliflower head: a crown of 3-5 big lobes
    rh = big[1] * W * 0.62
    xh = xc + lean * (yb - yt)
    for k in range(rng.integers(3, 6)):
        a = -math.pi / 2 + rng.uniform(-1.2, 1.2)
        lobes.append([xh + math.cos(a) * rh * 0.8, yt + rh + math.sin(a) * rh * 0.55 + rh * 0.2,
                      zc + rng.uniform(-0.2, 0.4) * rh, rh * rng.uniform(0.7, 1.05), rng.uniform(0.85, 0.95), 0, -1, 0])
    return first


def _children(rng, lobes, parents, count, rr, level, sun_dir, bias_up=0.6, bias_sun=0.5, front=0.5):
    """Put smaller lobes on the surfaces of `parents` (list of lobe indices)."""
    sdx, sdy = sun_dir
    for pi in parents:
        p = lobes[pi]
        for _ in range(count):
            # random direction on the visible hemisphere, biased up / toward the sun / to the silhouette
            for _try in range(6):
                th = rng.uniform(0, 2 * math.pi)
                ex, ey = math.cos(th), math.sin(th)
                score = bias_up * (-ey) + bias_sun * (ex * sdx + ey * sdy)
                if rng.random() < 0.35 + 0.5 * score:
                    break
            ez = rng.uniform(0.0, 1.0) * front                 # 0 = silhouette, 1 = facing camera
            s = math.sqrt(max(1 - ez * ez, 0))
            r = p[LR] * rng.uniform(*rr)
            d = p[LR] - r * rng.uniform(0.25, 0.55)
            lobes.append([p[LX] + ex * s * d, p[LY] + ey * s * d * p[LAY], p[LZ] + ez * d, r,
                          rng.uniform(0.8, 0.96), level, pi, 0])


def streak_noise(W, H, cw, ch, seed, octaves=3):
    """0..1 noise with cells cw x ch px (long horizontal fibres when cw >> ch)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((H, W), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        nx, ny = max(int(W / cw) + 3, 3), max(int(H / ch) + 3, 3)
        g = rng.random((ny, nx)).astype(np.float32)
        out += amp * cv2.resize(g, (W, H), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
        cw, ch = cw / 2, ch / 2
    out /= tot
    lo, hi = np.percentile(out, 1), np.percentile(out, 99)
    return np.clip((out - lo) / (hi - lo + 1e-6), 0, 1)


def build_lobes(U, turrets, sun_dir, seed=7, anvil=None, detail=1.0):
    """turrets: list of (xc, base_y, top_y, half_w, z, lean, (rmin, rmax)) in plate px (radii as fraction
    of U). anvil: optional (x0, y0, length_px (signed), thick_px). Returns the lobe table."""
    rng = np.random.default_rng(seed)
    lobes = []
    for (xc, yb, yt, hw, zc, lean, big) in turrets:
        _turret(rng, lobes, xc, yb, yt, hw, zc, lean, big=big, W=U)
    n0 = len(lobes)
    _children(rng, lobes, list(range(n0)), int(6 * detail), (0.25, 0.5), 1, sun_dir, front=0.55)
    n1 = len(lobes)
    _children(rng, lobes, list(range(n0, n1)), int(6 * detail), (0.22, 0.45), 2, sun_dir, bias_up=0.9,
              bias_sun=0.9, front=0.25)
    n2 = len(lobes)
    # tiny beads on the upper sunward silhouettes only (cauliflower crispness)
    med = float(np.median([lb[LY] for lb in lobes]))
    ups = [i for i in range(n1, n2) if lobes[i][LY] < med]
    _children(rng, lobes, ups, 2, (0.3, 0.5), 3, sun_dir, bias_up=1.0, bias_sun=1.0, front=0.1)
    if anvil is not None:
        ax0, ay0, alen, ath = anvil
        for k in range(110):
            u = rng.uniform(0, 1) ** 0.9
            thick = ath * (1.0 - 0.7 * u)
            x = ax0 + u * alen + rng.uniform(-0.01, 0.01) * U
            y = ay0 - 0.2 * ath * math.sin(u * 2.5) + 0.25 * ath * u + rng.uniform(-0.6, 0.35) * thick
            r = (0.03 - 0.016 * u) * U * rng.uniform(0.6, 1.2)
            lobes.append([x, y, -0.03 * U + rng.uniform(-0.02, 0.02) * U, r, rng.uniform(0.35, 0.55), 4, -1, 0])
    return np.array(lobes, np.float64)


# sunrise palette (linear-ish HDR; final grade does the rest)
TPAL = dict(
    shd_hi=(0.47, 0.42, 0.80), shd_lo=(0.27, 0.22, 0.56), bounce=(0.72, 0.42, 0.62),
    lit_far=(1.0, 0.56, 0.58), lit_near=(1.12, 0.70, 0.40), hot=(1.2, 0.92, 0.66), coral=(1.0, 0.46, 0.52),
    rim=(1.9, 1.5, 1.0), wisp=(0.62, 0.46, 0.70), anvil_lo=(0.62, 0.46, 0.78), anvil_hi=(1.2, 0.72, 0.62))


def paint_tower(W, H, U, sky, sun, turrets, base_y, seed=7, ss=2, anvil=None, lz=-0.12, pal=None,
                top_y=None, detail=1.0):
    """Straight-alpha RGBA plate (H, W, 4) of one towering cloud.
    U: size unit (frame width); sun: (x, y) sun in plate px; base_y: y of the torn base (px)."""
    P = dict(TPAL)
    if pal:
        P.update(pal)
    P = {k: np.array(v, np.float32) for k, v in P.items()}
    cxm = np.mean([tt[0] for tt in turrets])
    cym = np.mean([0.5 * (tt[1] + tt[2]) for tt in turrets])
    sdx, sdy = sun[0] - cxm, sun[1] - cym
    sn = math.hypot(sdx, sdy) + 1e-6
    L = build_lobes(U, turrets, (sdx / sn, sdy / sn), seed, anvil, detail)
    if top_y is None:
        top_y = float(np.min(L[:, LY] - L[:, LR] * L[:, LAY]))
    ext_x = L[:, LR]
    ext_y = L[:, LR] * L[:, LAY]
    x0 = int(max(np.min(L[:, LX] - ext_x) - 0.02 * U, 0))
    x1 = int(min(np.max(L[:, LX] + ext_x) + 0.02 * U, W))
    y0 = int(max(np.min(L[:, LY] - ext_y) - 0.02 * U, 0))
    y1 = int(min(np.max(L[:, LY] + ext_y) + 0.04 * U, H))
    w, h = x1 - x0, y1 - y0
    Ws, Hs = w * ss, h * ss
    n1 = C.fbm(Ws // 4, Hs // 4, scale=w / (0.035 * U), octaves=3, seed=seed + 1)
    n2 = C.fbm(Ws // 4, Hs // 4, scale=w / (0.035 * U), octaves=3, seed=seed + 2)
    m1 = C.fbm(Ws // 2, Hs // 2, scale=w / (0.012 * U), octaves=2, seed=seed + 3)
    m2 = C.fbm(Ws // 2, Hs // 2, scale=w / (0.012 * U), octaves=2, seed=seed + 4)
    up = lambda a: cv2.resize(a, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    wx = (up(n1) - 0.5) * 0.01 * U + (up(m1) - 0.5) * 0.0025 * U
    wy = (up(n2) - 0.5) * 0.01 * U + (up(m2) - 0.5) * 0.0025 * U
    bb = np.stack([L[:, LX] - ext_x, L[:, LY] - ext_y, L[:, LX] + ext_x, L[:, LY] + ext_y], 1)
    idx, dep = _raster(L, bb, Hs, Ws, float(ss), float(x0), float(y0), wx.astype(np.float32), wy.astype(np.float32))
    cov = (idx >= 0).astype(np.float32)
    ii = np.maximum(idx, 0)
    xs = (np.arange(Ws, dtype=np.float32)[None, :] + 0.5) / ss + x0 + wx
    ys = (np.arange(Hs, dtype=np.float32)[:, None] + 0.5) / ss + y0 + wy
    r = L[ii, LR].astype(np.float32)
    nx = (xs - L[ii, LX].astype(np.float32)) / r
    ny = (ys - L[ii, LY].astype(np.float32)) / (r * L[ii, LAY].astype(np.float32))
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    lev = L[ii, LLEV]
    anv = (lev == 4).astype(np.float32)
    lx_, ly_ = sun[0] - xs, sun[1] - ys
    ln = np.sqrt(lx_ ** 2 + ly_ ** 2) + 1e-3
    # the sun is at infinity on the horizon: ONE horizontal light direction for the whole tower
    # (toward the sun's side, slightly behind the cloud), never an under-light toward the sun's screen point
    s_ = math.sqrt(1 - lz * lz - 0.08 ** 2)
    Lx = np.float32(math.copysign(s_, sdx))
    Ly = np.float32(-0.08)
    # big-form normal (coherent shadow planes): macro depth + puffy dome from the silhouette distance
    sg = 0.01 * U * ss
    dmin = float(dep[cov > 0].min()) if (cov > 0).any() else 0.0
    dd = np.where(cov > 0, dep - dmin, 0).astype(np.float32)
    cb_ = cv2.GaussianBlur(cov, (0, 0), sg)
    macro = cv2.GaussianBlur(dd, (0, 0), sg) / np.maximum(cb_, 1e-3)
    din = cv2.distanceTransform((cov > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    R = 0.03 * U * ss
    puff = np.sqrt(np.clip(din, 0, R) * R * 2 - np.clip(din, 0, R) ** 2)
    hf = cv2.GaussianBlur(puff + 0.35 * macro, (0, 0), 2.0 * ss)
    gx_ = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy_ = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    fn = np.sqrt(gx_ ** 2 + gy_ ** 2 + 1.0)
    fx_, fy_, fz_ = -gx_ / fn, -gy_ / fn, 1.0 / fn
    wo = 0.32
    nx2, ny2, nz2 = wo * nx + (1 - wo) * fx_, wo * ny + (1 - wo) * fy_, wo * nz + (1 - wo) * fz_
    nl = np.sqrt(nx2 ** 2 + ny2 ** 2 + nz2 ** 2) + 1e-6
    ndl = (nx2 * Lx + ny2 * Ly + nz2 * lz) / nl
    ndl_own = nx * Lx + ny * Ly + nz * lz
    nyn = ny2 / nl
    ndl = cv2.GaussianBlur(ndl * cov, (0, 0), 0.5 * ss) / np.maximum(cv2.GaussianBlur(cov, (0, 0), 0.5 * ss), 1e-3)
    # low parts of the tower sit in the shade of the cloud sea / morning haze
    sunlit = C.smoothstep(base_y + 0.01 * U, base_y - 0.12 * U, ys).astype(np.float32)
    ndl_mix = 0.8 * ndl + 0.2 * ndl_own
    # cel ramp: shadow | coral terminator | lit | hot, soft gradient only inside each band
    t_lit = C.smoothstep(-0.02, 0.05, ndl) * sunlit
    t_full = C.smoothstep(0.05, 0.4, ndl_mix) * sunlit
    t_hot = C.smoothstep(0.45, 0.75, ndl_mix) * sunlit
    t_band = np.exp(-((ndl + 0.015) / 0.022) ** 2) * sunlit
    prox = np.exp(-np.sqrt(((xs - sun[0]) / U) ** 2 + ((ys - sun[1]) / U) ** 2) / 0.35)
    hgt = np.clip((base_y - ys) / max(base_y - top_y, 1.0), 0, 1)
    shd = C.lerp(P['shd_lo'], P['shd_hi'], C.smoothstep(0.0, 0.8, hgt)[..., None])
    # shadow side: soft internal gradient, lighter toward the terminator
    sgrad = C.smoothstep(-0.6, 0.0, ndl)[..., None]
    shd = shd * (0.86 + 0.2 * sgrad)
    upf = C.smoothstep(-0.2, -0.5, nyn)[..., None]
    dnf = C.smoothstep(0.35, 0.65, nyn)[..., None]
    shd = shd * (1 + 0.1 * upf) + (P['bounce'] - shd) * dnf * 0.4 * (1 - hgt[..., None] * 0.6)
    lit = C.lerp(P['lit_far'], P['lit_near'], np.clip(prox * 1.8, 0, 1)[..., None])
    col = shd
    col = C.lerp(col, P['coral'], (t_band * 0.55)[..., None])
    lit_dim = C.lerp(lit, P['coral'], 0.35) * 0.9
    col = C.lerp(col, lit_dim, t_lit[..., None])
    col = C.lerp(col, lit, t_full[..., None])
    col = C.lerp(col, P['hot'], (t_hot * 0.7)[..., None])
    t_lit = np.maximum(t_lit, t_full)
    anv_col = C.lerp(P['anvil_lo'], P['anvil_hi'], C.smoothstep(-0.1, 0.5, nyn)[..., None])
    col = C.lerp(col, anv_col, anv[..., None] * 0.8)
    # recess seams where a lobe sits in front of another (thin, never crater-like)
    dmax = cv2.dilate(dep, np.ones((3, 3), np.uint8), iterations=max(ss, 1))
    seam = C.smoothstep(0.003 * U, 0.012 * U, dmax - dep) * cov
    seam = cv2.GaussianBlur(seam, (0, 0), 0.5 * ss)
    col = C.lerp(col, C.lerp(lit, P['coral'], 0.6) * 0.85, (seam * t_lit * 0.5)[..., None])
    col = col * (1 - 0.05 * seam[..., None] * (1 - t_lit[..., None]))
    # crisp single rim on edges facing the sun (neighbour toward the sun is empty / much farther)
    off = 2.5 * ss
    ux, uy = np.float32(math.copysign(0.9, sdx)), np.float32(-0.44)
    gx, gy = np.meshgrid(np.arange(Ws, dtype=np.float32), np.arange(Hs, dtype=np.float32))
    nd = cv2.remap(dep, gx + ux * off, gy + uy * off, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                   borderValue=-1e9)
    edge = ((dep - nd) > 0.01 * U).astype(np.float32) * cov
    outer = (nd < -1e8).astype(np.float32) * cov
    rim = np.clip(edge * (0.5 + 0.5 * outer), 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), 0.45 * ss)
    rim_w = (0.4 + 0.6 * np.clip(prox * 1.6, 0, 1)) * (0.3 + 0.7 * np.maximum(sunlit, 0.3 * outer))
    col = C.lerp(col, P['rim'], np.clip(rim * rim_w * 1.4, 0, 1)[..., None])
    glow = cv2.GaussianBlur(rim, (0, 0), 3.0 * ss) * rim_w
    col = col + np.array([0.6, 0.32, 0.2], np.float32) * np.clip(glow * 1.2, 0, 0.5)[..., None]
    tex = up(C.fbm(Ws // 2, Hs // 2, scale=w / (0.02 * U), octaves=3, seed=seed + 9)) - 0.5
    col = col * (1 + 0.03 * tex[..., None])
    # ---- alpha: torn wispy base dissolving into the cloud sea + fibrous anvil edge
    alpha = cov.copy()
    fib = streak_noise(Ws, Hs, 0.06 * U * ss, 0.006 * U * ss, seed + 11)
    torn = C.smoothstep(base_y - 0.06 * U, base_y + 0.005 * U, ys)
    alpha = alpha * (1 - torn * C.smoothstep(0.3, 0.7, fib + (torn - 0.5) * 0.7))
    if anvil is not None:
        ax0, ay0, alen, ath = anvil
        ue = (xs - ax0) / alen
        ar = C.smoothstep(0.6, 1.05, ue) * anv
        fib2 = streak_noise(Ws, Hs, 0.1 * U * ss, 0.004 * U * ss, seed + 13)
        alpha = alpha * (1 - ar * C.smoothstep(0.3, 0.7, fib2 + ar * 0.35))
        alpha = alpha * (1 - anv * 0.08)
    wisp = C.smoothstep(0.5, 0.8, fib) * np.exp(-((ys - base_y + 0.01 * U) / (0.02 * U)) ** 2)
    wcol = P['wisp']
    na = np.clip(alpha + wisp * 0.55 * (1 - alpha), 0, 1)
    col = (col * alpha[..., None] + wcol * (wisp * 0.55 * (1 - alpha))[..., None]) / np.maximum(na, 1e-4)[..., None]
    alpha = na
    pm = np.dstack([col * alpha[..., None], alpha]).astype(np.float32)
    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 4), np.float32)
    a = pm[..., 3]
    out[y0:y1, x0:x1, :3] = pm[..., :3] / np.maximum(a, 1e-4)[..., None]
    out[y0:y1, x0:x1, 3] = a
    return out
