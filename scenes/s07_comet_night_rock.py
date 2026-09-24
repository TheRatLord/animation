"""Round-2 painting helpers for s07_comet_night: faceted rock planes, comet dust tail, torii."""
import math
import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def spur_relief(top, Hl, spacing, seed, base_y, k=(0.55, 1.2), lean=1.4, sub=2, wfrac=(0.45, 0.95)):
    """Piecewise-planar relief field for a mountain face: the max of many 'tent' ridges (spurs) that
    descend from crest points along the fall line, with side branches. max() of planar tents gives hard
    creases -> crisp painted rock planes (lit face / shadow face per spur, sharp gullies between).
    top: per-column crest y (len W); base_y: y where the face meets the valley. Returns R (Hl, W)."""
    W = len(top)
    rng = np.random.default_rng(seed)
    R = np.zeros((Hl, W), np.float32)
    ts = cv2.GaussianBlur(top.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=max(spacing * 2.5, 1.0))[0]
    g = np.gradient(ts)

    def tent(x0, y0, dx, dy, L, w, kk, m, off):
        pad = w + 2
        xa, xb = min(x0, x0 + dx * L) - pad, max(x0, x0 + dx * L) + pad
        ya, yb = y0 - pad, y0 + dy * L + pad
        xi0, xi1 = int(max(xa, 0)), int(min(xb + 1, W))
        yi0, yi1 = int(max(ya, 0)), int(min(yb + 1, Hl))
        if xi1 <= xi0 or yi1 <= yi0:
            return
        yy, xx = np.mgrid[yi0:yi1, xi0:xi1].astype(np.float32)
        px, py = xx - x0, yy - y0
        a = np.clip(px * dx + py * dy, 0, L)
        perp = np.sqrt((px - a * dx) ** 2 + (py - a * dy) ** 2)
        v = kk * (w - perp) - m * a + off
        sub_ = R[yi0:yi1, xi0:xi1]
        np.maximum(sub_, v, out=sub_)

    x = rng.uniform(0, spacing)
    while x < W - 1:
        xi = int(x)
        y0 = float(top[xi])
        by = float(base_y if np.isscalar(base_y) else base_y[xi])
        if y0 < by - 3:
            dx = float(np.clip(g[xi] * lean + rng.normal(0, 0.3), -1.8, 1.8))
            n = math.hypot(dx, 1.0)
            dx, dy = dx / n, 1.0 / n
            L = (by - y0) * rng.uniform(0.35, 1.2) / dy
            w = spacing * rng.uniform(*wfrac)
            kk = rng.uniform(*k)
            m = kk * w / max(L, 1) * rng.uniform(0.3, 0.7)
            tent(x, y0 + 0.5, dx, dy, L, w, kk, m, 0.0)
            for j in range(sub):
                a_s = L * rng.uniform(0.18, 0.65)
                side = 1 if rng.random() < 0.5 else -1
                ang = math.radians(rng.uniform(22, 48)) * side
                ex = dx * math.cos(ang) - dy * math.sin(ang)
                ey = dx * math.sin(ang) + dy * math.cos(ang)
                if ey < 0.35:
                    continue
                ws = w * rng.uniform(0.35, 0.65)
                ks = kk * rng.uniform(0.8, 1.25)
                Ls = L * rng.uniform(0.25, 0.55)
                ms = ks * ws / max(Ls, 1) * 0.5
                base_h = kk * w - m * a_s
                tent(x + dx * a_s, y0 + dy * a_s, ex, ey, Ls, ws, ks, ms, base_h - ks * ws)
        x += spacing * rng.uniform(0.4, 1.6)
    return R


def faceted_layer(Wp, Hl, top, ss, s, seed, spacing, comet, shadow, lit, haze, y_base, haze_px,
                  rim_col=(0.45, 0.8, 1.0), rim_amt=0.6, rim_px=1.4, contrast=1.0, k=(0.55, 1.2), lean=1.4,
                  glob=0.35, haze_amt=0.75, var_amt=0.05, crest_col=None, crest_px=0.0, sub=2):
    """RGBA mountain layer painted as hard-edged rock planes lit by the comet + cyan crest rim light.
    top: supersampled (len Wp*ss) crest line."""
    ysl = np.arange(Hl, dtype=np.float32)[:, None]
    msk = np.clip(ysl - top[None, :] + 0.5, 0, 1)
    msk = cv2.resize(msk, (Wp, Hl), interpolation=cv2.INTER_AREA)
    top1 = top.reshape(Wp, ss).mean(1)
    R = spur_relief(top1, Hl, spacing, seed, y_base, k=k, lean=lean, sub=sub)
    R += 0.55 * spur_relief(top1, Hl, spacing * 0.42, seed + 77, y_base, k=k, lean=lean * 0.8, sub=1,
                            wfrac=(0.35, 0.9))
    R += glob * np.clip(ysl - top1[None, :], 0, None)
    R = C.blur(R, 0.5 * s + 0.35)
    Rx = np.gradient(R, axis=1)
    Ry = np.gradient(R, axis=0)
    nz = 1.0 / np.sqrt(Rx * Rx + Ry * Ry + 1)
    nx, ny = -Rx * nz, -Ry * nz
    xs, ys = C.grid(Wp, Hl)
    lx, ly = comet[0] - xs, comet[1] - ys
    ln = np.sqrt(lx * lx + ly * ly) + 1e-3
    lx, ly = lx / ln, ly / ln
    d = nx * lx * 0.8 + ny * ly * 0.8 + nz * 0.6
    inside = msk > 0.5
    med = float(np.median(d[inside])) if inside.any() else 0.0
    v = np.clip(0.5 + (d - med) * 1.8 * contrast, 0, 1)
    shadow, lit, haze = [np.asarray(c, np.float32) for c in (shadow, lit, haze)]
    img = shadow + (lit - shadow) * v[..., None]
    if var_amt:
        vn = P.fbm_lowres(Wp, Hl, 3, 3, seed + 900, q=8) - 0.5
        img = img * (1 + var_amt * vn[..., None])
    hz = C.smoothstep(y_base - haze_px, y_base + 0.2 * haze_px, ys) * haze_amt
    img = img * (1 - hz[..., None]) + haze * hz[..., None]
    dtop = np.clip(ys - top1[None, :], 0, None)
    if crest_col is not None and crest_px:
        cb = np.exp(-dtop / crest_px)
        img = img + cb[..., None] * np.asarray(crest_col, np.float32)
    m8 = (msk > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3).astype(np.float32)
    gm = C.blur(msk, 2.0 * s + 0.5)
    gx, gy = np.gradient(gm, axis=1), np.gradient(gm, axis=0)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    facing = np.clip((-gx * lx - gy * ly) / gn, 0, 1)
    prox = np.exp(-np.sqrt((xs - comet[0]) ** 2 + (ys - comet[1]) ** 2) / (0.55 * Wp))
    rim = np.exp(-dist / (rim_px * s + 0.3)) * (0.2 + 0.8 * facing) * (0.25 + 0.75 * prox)
    img = img + rim[..., None] * np.asarray(rim_col, np.float32) * rim_amt
    return np.dstack([img, msk]).astype(np.float32)


def dust_tail(W, H, head, direction, length, width0, width1, bend, seed, col=(1.0, 0.9, 0.72),
              strength=1.0, stria=0.4):
    """Broad, curved, fanning cometary dust tail with a sharper leading edge and faint striae."""
    rng = np.random.default_rng(seed)
    xs, ys = C.grid(W, H)
    dx, dy = direction
    n = math.hypot(dx, dy)
    dx, dy = dx / n, dy / n
    px, py = xs - head[0], ys - head[1]
    u = (px * dx + py * dy) / length
    across = -px * dy + py * dx
    across = across - bend * length * np.clip(u, 0, None) ** 2
    wu = (width0 + (width1 - width0) * np.clip(u, 0, 1) ** 0.85) * W
    vn = across / wu
    fall = C.smoothstep(-0.02, 0.06, u) * np.clip(1 - u, 0, 1) ** 1.4 * (0.35 + 0.65 * np.exp(-u / 0.28))
    lead = -1.0 if bend < 0 else 1.0
    vv = vn * lead
    vv = np.clip(vv, -8, 8)
    prof = np.where(vv > 0, np.exp(-(vv / 0.8) ** 3), np.exp(-(vv / 1.1) ** 2))
    p1, p2 = rng.uniform(0, 6.28, 2)
    st = 0.5 + 0.5 * (0.6 * np.sin(u * 48 + vn * 2.2 + p1) + 0.4 * np.sin(u * 21 - vn * 1.4 + p2))
    sa = stria * C.smoothstep(0.08, 0.35, u)
    b = prof * fall * (1 - sa + sa * st) * strength
    cc = np.asarray(col, np.float32)
    far = np.array([0.9, 0.85, 1.0], np.float32)
    uu = np.clip(u, 0, 1)[..., None]
    colm = cc * (1 - uu) + far * uu
    return (b[..., None] * colm).astype(np.float32)


def torii_polys(x, base, h, w):
    """Polygons for a torii gate silhouette (x centre, base y, height, width)."""
    pw = 0.085 * w
    polys = []
    for sx in (-1, 1):
        cx = x + sx * 0.33 * w
        polys.append([(cx - pw * 0.6, base), (cx - pw * 0.5, base - h * 0.9), (cx + pw * 0.5, base - h * 0.9),
                      (cx + pw * 0.6, base)])
    t0 = base - h
    kt = 0.12 * h
    pts_top, pts_bot = [], []
    for i in range(13):
        q = i / 12 * 2 - 1
        xx = x + q * 0.55 * w
        up = 0.09 * h * abs(q) ** 3
        pts_top.append((xx, t0 - up))
        pts_bot.append((xx, t0 + kt - up * 0.7))
    polys.append(pts_top + pts_bot[::-1])
    polys.append([(x - 0.46 * w, t0 + kt), (x + 0.46 * w, t0 + kt), (x + 0.46 * w, t0 + kt * 1.8),
                  (x - 0.46 * w, t0 + kt * 1.8)])
    ny = base - 0.68 * h
    polys.append([(x - 0.45 * w, ny), (x + 0.45 * w, ny), (x + 0.45 * w, ny + 0.07 * h), (x - 0.45 * w, ny + 0.07 * h)])
    polys.append([(x - 0.03 * w, t0 + kt), (x + 0.03 * w, t0 + kt), (x + 0.03 * w, ny), (x - 0.03 * w, ny)])
    return polys


def milky_way2(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0):
    """Painterly Milky Way with crisp domain-warped dust filaments and warm core knots.
    Returns (glow RGB additive, dust (H,W), density (H,W))."""
    s = W / 1920.0
    xs, ys = C.grid(W, H)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    ang = math.degrees(math.atan2(dy, dx))
    u = ((xs - p0[0]) * ux + (ys - p0[1]) * uy) / L
    v = (-(xs - p0[0]) * uy + (ys - p0[1]) * ux)
    v = v - bend * L * ((u - 0.5) ** 2 * 4 - 1)
    wob = (P.rot_fbm(W, H, ang, 6, 1.5, 3, seed + 1) - 0.5) * width * 0.9
    vb = (v + wob) / width
    w_u = 1.0 + 0.5 * np.clip(1 - u, 0, 1) ** 2
    band = np.exp(-(vb / w_u) ** 2 * 1.2)
    inner = np.exp(-(vb / (w_u * 0.5)) ** 2)
    halo = np.exp(-(vb / (w_u * 2.4)) ** 2)
    core = np.exp(-((u - 0.14) / 0.28) ** 2)
    big = P.rot_fbm(W, H, ang, 2.0, 3.5, 5, seed + 2, warp_amt=0.05)
    fine = P.rot_fbm(W, H, ang, 1.8, 22, 4, seed + 7, warp_amt=0.02)
    mid = P.rot_fbm(W, H, ang, 2.5, 8, 4, seed + 17, warp_amt=0.03)
    tex = np.clip(big * 0.5 + mid * 0.35 + fine * 0.3 - 0.15, 0, 1)
    lum = band * (0.15 + 1.15 * tex ** 1.6) * (0.7 + 0.5 * core) + inner * 0.3 * tex + halo * 0.12
    # dust: main rift (thresholded warped noise, crisp edge) + thin ridged filaments
    lanes = P.rot_fbm(W, H, ang, 5, 4, 6, seed + 3, warp_amt=0.09)
    fn = P.rot_fbm(W, H, ang, 3.0, 9, 5, seed + 8, warp_amt=0.08)
    fil = 1 - np.abs(fn * 2 - 1)
    fn2 = P.rot_fbm(W, H, ang + 10, 2.0, 20, 4, seed + 9, warp_amt=0.06)
    fil2 = 1 - np.abs(fn2 * 2 - 1)
    lc = np.exp(-((vb + 0.1) / 0.75) ** 2)
    main = C.smoothstep(0.64, 0.69, lanes) * lc * (0.75 + 0.25 * C.smoothstep(0.63, 0.75, lanes))
    fils = C.smoothstep(0.86, 0.91, fil) * np.exp(-(vb / 1.15) ** 2)
    fils2 = C.smoothstep(0.9, 0.94, fil2) * np.exp(-(vb / 0.9) ** 2) * (1 - main)
    dust = np.clip(main * 0.45 + fils * 0.3 + fils2 * 0.22, 0, 0.6)
    dust = C.blur(dust.astype(np.float32), 0.9 * s + 0.3)
    # brighter luminous rim along the lane edges (painted look)
    edge = np.clip(C.blur(dust, 2.5 * s + 0.5) - dust, 0, 1)
    lum = lum * (1 - dust) + edge * band * 0.5
    hue = P.rot_fbm(W, H, ang, 2, 2.5, 3, seed + 4)
    body = np.stack([0.74 + 0.16 * hue, 0.8 + 0.04 * hue, 1.0 - 0.06 * hue], -1)
    warm = np.array([1.0, 0.84, 0.66], np.float32)
    cw = (core * 0.7)[..., None]
    col = body * (1 - cw) + warm * cw
    img = lum[..., None] * col * 0.85 * strength
    # warm-white core knots (star clouds)
    kn = P.rot_fbm(W, H, ang, 1.4, 12, 4, seed + 21, warp_amt=0.04)
    knots = C.smoothstep(0.72, 0.86, kn) * np.exp(-(vb / 0.75) ** 2) * (0.35 + 0.65 * core) * (1 - dust)
    kg = C.blur(knots.astype(np.float32), 0.006 * W)
    img += (knots * 0.3 + kg * 0.45)[..., None] * np.array([1.0, 0.93, 0.8], np.float32) * strength
    # emission nebulae: pink/magenta knots
    neb = P.rot_fbm(W, H, ang, 1.5, 8, 5, seed + 11, warp_amt=0.05)
    nebm = C.smoothstep(0.74, 0.9, neb) * np.exp(-(vb / 1.3) ** 2) * (1 - dust * 0.7)
    nebg = C.blur(nebm.astype(np.float32), 0.012 * W)
    img += (nebm * 0.6 + nebg * 0.45)[..., None] * np.array([1.0, 0.3, 0.6], np.float32) * 0.22 * strength
    neb2 = P.rot_fbm(W, H, ang, 3, 3, 4, seed + 13, warp_amt=0.04)
    cy = C.smoothstep(0.45, 0.85, neb2) * np.exp(-((vb - 1.0) / 0.9) ** 2)
    img += cy[..., None] * np.array([0.15, 0.6, 0.9], np.float32) * 0.12 * strength
    vi = np.exp(-((vb + 1.4) / 1.0) ** 2) * (0.5 + neb2)
    img += vi[..., None] * np.array([0.4, 0.18, 0.62], np.float32) * 0.07 * strength
    density = np.clip(band * (1 - dust) * (0.3 + 0.9 * tex) + knots * 0.6, 0, 1)
    return img.astype(np.float32), dust.astype(np.float32), density.astype(np.float32)


def star_clusters(W, H, seed, density, n):
    """Extra dense fine stars packed along the Milky Way core (clusters)."""
    rng = np.random.default_rng(seed)
    s = W / 1920.0
    x = rng.random(n) * W
    y = rng.random(n) * H
    ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    keep = rng.random(n) < density[iy, ix] ** 2.4
    x, y = x[keep], y[keep]
    m = rng.random(len(x)) ** 3 * 0.8 + 0.15
    warm = rng.random(len(x)) < 0.4
    acc = np.zeros((H, W, 3), np.float32)
    col = np.where(warm[:, None], np.array([1.0, 0.9, 0.78]), np.array([0.85, 0.92, 1.0])).astype(np.float32)
    for c in range(3):
        np.add.at(acc[..., c], (np.clip(y.astype(int), 0, H - 1), np.clip(x.astype(int), 0, W - 1)), m * col[:, c])
    return C.blur(acc, 0.45 * max(s, 0.5)) * 3.0


def big_glints(W, H, seed, n, density=None):
    """A handful of large 4-point glint stars with soft halos (star hierarchy top tier)."""
    rng = np.random.default_rng(seed)
    s = W / 1920.0
    out = np.zeros((H, W, 3), np.float32)
    pal = np.array([[1, 1, 1], [0.8, 0.9, 1.0], [1.0, 0.92, 0.8], [0.75, 0.88, 1.0]], np.float32)
    for i in range(n):
        x, y = rng.random() * W, rng.random() ** 1.2 * H * 0.9
        col = pal[rng.integers(0, len(pal))]
        m = rng.uniform(0.7, 1.3)
        L = rng.uniform(0.012, 0.024) * W
        R = int(L) + 2
        x0, y0 = int(x) - R, int(y) - R
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        dx_, dy_ = xx - x, yy - y
        wdt = 0.55 * s + 0.35
        k = np.exp(-(dy_ / wdt) ** 2) * np.clip(1 - np.abs(dx_) / L, 0, 1) ** 3
        k += np.exp(-(dx_ / wdt) ** 2) * np.clip(1 - np.abs(dy_) / L, 0, 1) ** 3
        # faint diagonal secondary spikes
        a1, a2 = (dx_ + dy_) * 0.7071, (dx_ - dy_) * 0.7071
        k += 0.3 * np.exp(-(a2 / wdt) ** 2) * np.clip(1 - np.abs(a1) / (0.4 * L), 0, 1) ** 3
        k += 0.3 * np.exp(-(a1 / wdt) ** 2) * np.clip(1 - np.abs(a2) / (0.4 * L), 0, 1) ** 3
        r2 = dx_ * dx_ + dy_ * dy_
        k += np.exp(-r2 / (1.1 * s + 0.5) ** 2) * 2.5 + np.exp(-r2 / (0.009 * W) ** 2) * 0.12
        out[cy0:cy1, cx0:cx1] += k[..., None] * col * m
    return out
