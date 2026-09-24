"""Hand-built painted cumulonimbus for s03_city_dusk (sunset, sun just below the horizon to the lower left).

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


def build_lobes(W, H, sun, seed=7):
    rng = np.random.default_rng(seed)
    lobes = []
    # turrets: (xc, base, top, half width, z, lean)
    base = 0.56 * H
    tur = [
        (0.80 * W, base, 0.12 * H, 0.085 * W, 0.0, -0.05, (0.036, 0.056)),   # main tower (leans toward the sun)
        (0.685 * W, base, 0.35 * H, 0.05 * W, 0.08 * W, 0.03, (0.026, 0.04)),  # low turret, sunward
        (0.915 * W, base, 0.25 * H, 0.07 * W, -0.04 * W, 0.03, (0.03, 0.048)),  # secondary tower right
        (1.03 * W, base, 0.33 * H, 0.06 * W, -0.08 * W, 0.0, (0.03, 0.045)),   # far right shoulder
    ]
    for (xc, yb, yt, hw, zc, lean, big) in tur:
        _turret(rng, lobes, xc, yb, yt, hw, zc, lean, big=big, W=W)
    n0 = len(lobes)
    # level 1: medium bumps on big masses (varied 0.25-0.5 of parent)
    sdx, sdy = -0.8, 0.6
    _children(rng, lobes, list(range(n0)), 6, (0.25, 0.5), 1, (sdx, sdy), front=0.75)
    n1 = len(lobes)
    # level 2: small beads, mostly on the silhouette / up / sunward
    _children(rng, lobes, list(range(n0, n1)), 7, (0.2, 0.42), 2, (sdx, sdy), bias_up=0.9, bias_sun=0.8,
              front=0.3)
    n2 = len(lobes)
    # level 3: tiny sparkle beads on the sunward top edges only
    # anvil: flattened plates sheared to the right at the top of the main tower
    ax0, ay0 = 0.74 * W, 0.1 * H
    for k in range(90):
        u = rng.uniform(0, 1) ** 0.9
        thick = (0.055 - 0.04 * u) * H          # wedge: thick over the tower, thin downwind
        x = ax0 + u * 0.36 * W + rng.uniform(-0.01, 0.01) * W
        y = ay0 - 0.012 * H * math.sin(u * 2.5) + 0.012 * H * u + rng.uniform(-0.6, 0.35) * thick
        r = (0.03 - 0.016 * u) * W * rng.uniform(0.6, 1.2)
        lobes.append([x, y, -0.03 * W + rng.uniform(-0.02, 0.02) * W, r, rng.uniform(0.35, 0.55), 4, -1, 0])
    L = np.array(lobes, np.float64)
    return L


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


def _earth_shadow(L, H, W, sun):
    """per-lobe sunlit weight: tops above the shadow line still see the sun"""
    y = L[:, LY] - L[:, LR] * L[:, LAY] * 0.3
    return C.smoothstep(0.43 * H, 0.27 * H, y).astype(np.float64)


def paint_cumulonimbus(W, H, sky, sun, seed=7, ss=2):
    """Returns a straight-alpha RGBA plate (H, W, 4). sky: (H, W, 3) sky plate behind (for fill tones).
    sun: (x, y) sun position in plate px (below/at the horizon, lower-left of the cloud)."""
    L = build_lobes(W, H, sun, seed)
    lit_w = _earth_shadow(L, H, W, sun)
    # region
    ext_x = L[:, LR]
    ext_y = L[:, LR] * L[:, LAY]
    x0 = int(max(np.min(L[:, LX] - ext_x) - 0.02 * W, 0))
    x1 = int(min(np.max(L[:, LX] + ext_x) + 0.02 * W, W))
    y0 = int(max(np.min(L[:, LY] - ext_y) - 0.02 * H, 0))
    y1 = int(min(np.max(L[:, LY] + ext_y) + 0.04 * H, H))
    w, h = x1 - x0, y1 - y0
    Ws, Hs = w * ss, h * ss
    # domain warp (two scales): wiggles every lobe outline into a lumpy cauliflower edge
    n1 = C.fbm(Ws // 4, Hs // 4, scale=w / (0.035 * W), octaves=3, seed=seed + 1)
    n2 = C.fbm(Ws // 4, Hs // 4, scale=w / (0.035 * W), octaves=3, seed=seed + 2)
    m1 = C.fbm(Ws // 2, Hs // 2, scale=w / (0.012 * W), octaves=2, seed=seed + 3)
    m2 = C.fbm(Ws // 2, Hs // 2, scale=w / (0.012 * W), octaves=2, seed=seed + 4)
    up = lambda a: cv2.resize(a, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    wx = (up(n1) - 0.5) * 0.01 * W + (up(m1) - 0.5) * 0.002 * W
    wy = (up(n2) - 0.5) * 0.01 * W + (up(m2) - 0.5) * 0.002 * W
    bb = np.stack([L[:, LX] - ext_x, L[:, LY] - ext_y, L[:, LX] + ext_x, L[:, LY] + ext_y], 1)
    idx, dep = _raster(L, bb, Hs, Ws, float(ss), float(x0), float(y0), wx.astype(np.float32), wy.astype(np.float32))
    cov = (idx >= 0).astype(np.float32)
    ii = np.maximum(idx, 0)
    # ---- per-sample geometry
    xs = (np.arange(Ws, dtype=np.float32)[None, :] + 0.5) / ss + x0 + wx
    ys = (np.arange(Hs, dtype=np.float32)[:, None] + 0.5) / ss + y0 + wy
    r = L[ii, LR].astype(np.float32)
    nx = (xs - L[ii, LX].astype(np.float32)) / r
    ny = (ys - L[ii, LY].astype(np.float32)) / (r * L[ii, LAY].astype(np.float32))
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    lev = L[ii, LLEV]
    anv = (lev == 4).astype(np.float32)
    # light: toward the sun (screen direction per pixel) and slightly behind the cloud
    lx_, ly_ = sun[0] - xs, sun[1] - ys
    ln = np.sqrt(lx_ ** 2 + ly_ ** 2) + 1e-3
    lz = 0.22
    s = math.sqrt(1 - lz * lz)
    Lx, Ly = lx_ / ln * s, ly_ / ln * s
    # big-form normal: macro depth + a puffy dome from the distance to the silhouette. The lobe's own
    # normal only perturbs it, so shadow planes stay large and coherent (no speckled beads)
    sg = 0.01 * W * ss
    dmin = float(dep[cov > 0].min()) if (cov > 0).any() else 0.0
    dd = np.where(cov > 0, dep - dmin, 0).astype(np.float32)
    cb_ = cv2.GaussianBlur(cov, (0, 0), sg)
    macro = cv2.GaussianBlur(dd, (0, 0), sg) / np.maximum(cb_, 1e-3)
    din = cv2.distanceTransform((cov > 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    R = 0.03 * W * ss
    puff = np.sqrt(np.clip(din, 0, R) * R * 2 - np.clip(din, 0, R) ** 2)
    hf = cv2.GaussianBlur(puff + 0.35 * macro / ss * ss, (0, 0), 2.0 * ss)
    gx_ = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy_ = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    fn = np.sqrt(gx_ ** 2 + gy_ ** 2 + 1.0)
    fx_, fy_, fz_ = -gx_ / fn, -gy_ / fn, 1.0 / fn
    wo = 0.55
    nx2, ny2, nz2 = wo * nx + (1 - wo) * fx_, wo * ny + (1 - wo) * fy_, wo * nz + (1 - wo) * fz_
    nl = np.sqrt(nx2 ** 2 + ny2 ** 2 + nz2 ** 2) + 1e-6
    ndl = (nx2 * Lx + ny2 * Ly + nz2 * lz) / nl
    ndl_own = nx * Lx + ny * Ly + nz * lz
    ny = ny2 / nl
    ndl = cv2.GaussianBlur(ndl * cov, (0, 0), 0.5 * ss) / np.maximum(cv2.GaussianBlur(cov, (0, 0), 0.5 * ss), 1e-3)
    sunlit = lit_w[ii].astype(np.float32)
    sunlit = cv2.GaussianBlur(sunlit, (0, 0), 1.2 * ss)
    # hard terminator (2 px wide at the frame scale)
    # posterised painted value steps: shadow | lit-dim | lit | hot, each boundary ~1 px wide
    t_lit = C.smoothstep(-0.015, 0.06, ndl) * sunlit
    ndl_mix = 0.6 * ndl + 0.4 * ndl_own
    t_full = C.smoothstep(0.02, 0.5, ndl_mix) * sunlit
    t_hot = C.smoothstep(0.5, 0.8, ndl_mix) * sunlit
    t_band = np.exp(-((ndl + 0.012) / 0.02) ** 2) * sunlit
    # ---- colours (linear-ish HDR, final grade does the rest)
    prox = np.exp(-np.sqrt(((xs - sun[0]) / W) ** 2 + ((ys - sun[1]) / W) ** 2) / 0.45)
    hgt = np.clip((0.56 * H - ys) / (0.46 * H), 0, 1)
    # shadow: flat lavender planes; upward-facing a touch lighter/bluer (sky fill), downward pinker (city glow)
    shd_hi = np.array([0.50, 0.42, 0.80], np.float32)
    shd_lo = np.array([0.34, 0.27, 0.62], np.float32)
    bounce = np.array([0.66, 0.40, 0.62], np.float32)
    shd = C.lerp(shd_lo, shd_hi, C.smoothstep(0.1, 0.7, hgt)[..., None])
    upf = C.smoothstep(-0.2, -0.5, ny)[..., None]            # quantised (flat) plane tones
    dnf = C.smoothstep(0.35, 0.65, ny)[..., None]
    shd = shd * (1 + 0.1 * upf) + (bounce - shd) * dnf * 0.45 * (1 - hgt[..., None] * 0.6)
    # lit: gold near the sun, peach-pink further up
    lit = C.lerp(np.array([0.98, 0.44, 0.46], np.float32), np.array([1.08, 0.6, 0.34], np.float32),
                 np.clip(prox * 2.0, 0, 1)[..., None])
    lit_hot = np.array([1.15, 0.78, 0.5], np.float32)
    col = shd
    coral = np.array([1.0, 0.42, 0.52], np.float32)
    col = C.lerp(col, coral, (t_band * 0.6)[..., None])
    lit_dim = C.lerp(lit, coral, 0.3) * 0.88
    col = C.lerp(col, lit_dim, t_lit[..., None])
    col = C.lerp(col, lit, t_full[..., None])
    col = C.lerp(col, lit_hot, (t_hot * 0.75)[..., None])
    t_lit = np.maximum(t_lit, t_full)
    # anvil: smoother, pinker, lit from below
    anv_col = C.lerp(np.array([0.72, 0.5, 0.78], np.float32), np.array([1.2, 0.66, 0.66], np.float32),
                     C.smoothstep(-0.1, 0.4, ny)[..., None])
    col = C.lerp(col, anv_col, anv[..., None] * 0.85)
    # ---- recess lines: where a lobe sits in front of another, a thin darker seam just behind its edge
    dmax = cv2.dilate(dep, np.ones((3, 3), np.uint8), iterations=max(ss, 1))
    seam = C.smoothstep(0.003 * W, 0.012 * W, dmax - dep) * cov
    seam = cv2.GaussianBlur(seam, (0, 0), 0.5 * ss)
    # half-tone lobe lines: in the light a thin coral line, in the shadow a slightly deeper violet
    col = C.lerp(col, C.lerp(lit, coral, 0.6) * 0.85, (seam * t_lit * 0.55)[..., None])
    col = col * (1 - 0.12 * seam[..., None] * (1 - t_lit[..., None]))
    # ---- crisp rim on edges facing the sun: neighbour toward the sun is empty or much farther away
    off = 3.0 * ss
    ux, uy = lx_ / ln, ly_ / ln
    gx, gy = np.meshgrid(np.arange(Ws, dtype=np.float32), np.arange(Hs, dtype=np.float32))
    nd = cv2.remap(dep, gx + ux * off, gy + uy * off, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                   borderValue=-1e9)
    edge = ((dep - nd) > 0.01 * W).astype(np.float32) * cov
    # outer silhouette edges get the full rim, internal ones a softer one; strength by proximity & height
    outer = (nd < -1e8).astype(np.float32) * cov
    rim = np.clip(edge * (0.55 + 0.45 * outer), 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), 0.5 * ss)
    rim_w = (0.45 + 0.55 * np.clip(prox * 1.6, 0, 1)) * (0.35 + 0.65 * np.maximum(sunlit, 0.25 + 0.5 * outer))
    rimc = np.array([1.9, 1.45, 0.95], np.float32)
    col = C.lerp(col, rimc, np.clip(rim * rim_w * 1.4, 0, 1)[..., None])
    # thin translucent glow inside sunward edges
    glow = cv2.GaussianBlur(rim, (0, 0), 3.0 * ss) * rim_w
    col = col + np.array([0.6, 0.3, 0.2], np.float32) * np.clip(glow * 1.2, 0, 0.5)[..., None]
    # ---- subtle painted texture (brush), not noise-looking: very low amplitude
    tex = up(C.fbm(Ws // 2, Hs // 2, scale=w / (0.02 * W), octaves=3, seed=seed + 9)) - 0.5
    col = col * (1 + 0.035 * tex[..., None])
    # ---- alpha: torn wispy base + fibrous anvil edge
    alpha = cov.copy()
    yy = ys
    base_y = 0.56 * H
    fib = streak_noise(Ws, Hs, 0.06 * W * ss, 0.006 * H * ss, seed + 11)
    torn = C.smoothstep(base_y - 0.07 * H, base_y - 0.0 * H, yy)
    alpha = alpha * (1 - torn * C.smoothstep(0.35, 0.65, fib + (torn - 0.5) * 0.6))
    # anvil right end: comb into fibres
    ar = C.smoothstep(0.86 * W, 1.08 * W, xs) * anv
    fib2 = streak_noise(Ws, Hs, 0.1 * W * ss, 0.004 * H * ss, seed + 13)
    alpha = alpha * (1 - ar * C.smoothstep(0.3, 0.7, fib2 + ar * 0.35))
    alpha = alpha * (1 - anv * 0.1)
    # wisps below the base: horizontal streaks, dim violet
    wisp = C.smoothstep(0.55, 0.8, fib) * np.exp(-((yy - base_y - 0.0 * H) / (0.025 * H)) ** 2) * \
        C.smoothstep(0.58 * W, 0.7 * W, xs)
    wcol = np.array([0.55, 0.38, 0.62], np.float32)
    na = np.clip(alpha + wisp * 0.6 * (1 - alpha), 0, 1)
    col = (col * alpha[..., None] + wcol * (wisp * 0.6 * (1 - alpha))[..., None]) / np.maximum(na, 1e-4)[..., None]
    alpha = na
    # ---- downsample (premultiplied) and paste into the plate
    pm = np.dstack([col * alpha[..., None], alpha]).astype(np.float32)
    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros((H, W, 4), np.float32)
    a = pm[..., 3]
    out[y0:y1, x0:x1, :3] = pm[..., :3] / np.maximum(a, 1e-4)[..., None]
    out[y0:y1, x0:x1, 3] = a
    # soft halation just outside the sunward silhouette
    return out
