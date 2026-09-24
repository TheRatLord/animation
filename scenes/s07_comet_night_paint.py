"""Painting helpers for s07_comet_night (night sky, Milky Way, comet, mountains, town, water)."""
import math
import numpy as np
import cv2

from lib import core as C


# ----------------------------------------------------------------------------- 1D noise

def noise1d(x, freq, seed):
    """Smooth 1D value noise in -1..1 at positions x (array), `freq` cells per unit."""
    rng = np.random.default_rng(seed)
    xf = np.asarray(x, np.float64) * freq
    i0 = np.floor(xf).astype(np.int64)
    n = int(i0.max() - i0.min()) + 3
    g = rng.uniform(-1, 1, n)
    base = i0.min()
    f = xf - i0
    f = f * f * (3 - 2 * f)
    a = g[i0 - base]
    b = g[i0 - base + 1]
    return (a + (b - a) * f).astype(np.float32)


def fbm1d(x, freq, octaves, seed, gain=0.5, ridged=False):
    out = np.zeros_like(np.asarray(x, np.float32))
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        n = noise1d(x, freq * 2 ** o, seed + 31 * o)
        if ridged:
            n = 1 - 2 * np.abs(n)
        out += amp * n
        tot += amp
        amp *= gain
    return out / tot


def tile_noise(n, seed, kx, ky, power=2.0):
    """Tileable noise (n x n) via filtered white noise in Fourier domain. kx, ky = characteristic
    frequencies (cycles per tile) in x and y. Returns -1..1ish (std-normalised)."""
    rng = np.random.default_rng(seed)
    wn = rng.standard_normal((n, n))
    F = np.fft.fft2(wn)
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.fft.fftfreq(n)[None, :] * n
    r = np.sqrt((fx / kx) ** 2 + (fy / ky) ** 2)
    filt = np.exp(-r ** power)
    filt[0, 0] = 0
    out = np.real(np.fft.ifft2(F * filt)).astype(np.float32)
    return out / (out.std() + 1e-9)


def rot_fbm(W, H, angle_deg, stretch, scale, octaves, seed, warp_amt=0.0):
    """fbm whose features are elongated `stretch`x along direction angle_deg (image coords, y down).
    `scale` = number of base cells across the frame width in the cross direction."""
    D = int(math.hypot(W, H)) + 4
    n = C.fbm(D, D, scale * D / W, octaves, seed=seed, aspect=False)
    if warp_amt:
        w1 = C.fbm(D // 4, D // 4, 3, 3, seed=seed + 5) - 0.5
        w2 = C.fbm(D // 4, D // 4, 3, 3, seed=seed + 6) - 0.5
        w1 = cv2.resize(w1, (D, D)) * warp_amt * D
        w2 = cv2.resize(w2, (D, D)) * warp_amt * D
        n = C.warp(n, w1, w2)
    xs, ys = C.grid(W, H)
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    u = (xs - W / 2) * ca + (ys - H / 2) * sa
    v = -(xs - W / 2) * sa + (ys - H / 2) * ca
    mx = (u / max(stretch, 1e-3) + D / 2).astype(np.float32)
    my = (v + D / 2).astype(np.float32)
    return np.clip(cv2.remap(n, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT), 0, 1)


def fbm_lowres(W, H, scale, octaves, seed, q=4):
    n = C.fbm(max(W // q, 8), max(H // q, 8), scale, octaves, seed=seed)
    return cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)


# ----------------------------------------------------------------------------- sky

def sky_backdrop(W, H, y_h, stops, seed=0):
    """Vertical gradient; stops positions are fractions of y_h (0 top, 1 = horizon), beyond = last."""
    y = np.arange(H, dtype=np.float32) / max(y_h, 1)
    pos = np.array([s[0] for s in stops], np.float32)
    cols = np.array([C.hex2rgb(s[1]) for s in stops], np.float32)
    prof = np.stack([np.interp(y, pos, cols[:, c]) for c in range(3)], -1).astype(np.float32)
    prof = cv2.GaussianBlur(prof[:, None, :], (1, 0), sigmaX=0.1, sigmaY=0.02 * H)[:, 0, :]
    img = np.repeat(prof[:, None, :], W, axis=1)
    n = fbm_lowres(W, H, 2.5, 3, seed + 3, q=8) - 0.5
    img = img * (1 + 0.08 * n[..., None])
    return img.astype(np.float32)


def milky_way(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0):
    """Painterly Milky Way along the line p0 -> p1 (px). width = half-thickness px.
    Returns (glow RGB additive, dust multiplier (H,W), density (H,W) for star placement)."""
    xs, ys = C.grid(W, H)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    ang = math.degrees(math.atan2(dy, dx))
    u = ((xs - p0[0]) * ux + (ys - p0[1]) * uy) / L          # 0..1 along
    v = (-(xs - p0[0]) * uy + (ys - p0[1]) * ux)             # px across
    v = v - bend * L * ((u - 0.5) ** 2 * 4 - 1)
    wob = (rot_fbm(W, H, ang, 6, 1.5, 3, seed + 1) - 0.5) * width * 0.9
    vb = (v + wob) / width
    w_u = 1.0 + 0.5 * np.clip(1 - u, 0, 1) ** 2
    band = np.exp(-(vb / w_u) ** 2 * 1.2)
    inner = np.exp(-(vb / (w_u * 0.5)) ** 2)
    halo = np.exp(-(vb / (w_u * 2.4)) ** 2)
    core = np.exp(-((u - 0.02) / 0.3) ** 2)
    big = rot_fbm(W, H, ang, 2.0, 3.5, 5, seed + 2, warp_amt=0.05)
    fine = rot_fbm(W, H, ang, 1.8, 22, 4, seed + 7, warp_amt=0.02)
    mid = rot_fbm(W, H, ang, 2.5, 8, 4, seed + 17, warp_amt=0.03)
    tex = np.clip(big * 0.5 + mid * 0.35 + fine * 0.3 - 0.15, 0, 1)
    lum = band * (0.15 + 1.1 * tex ** 1.7) * (0.7 + 0.4 * core) + inner * 0.25 * tex + halo * 0.12
    # dust lanes: thin dark filaments following the band (ridges of warped noise), plus a main rift
    lanes = rot_fbm(W, H, ang, 6, 5, 6, seed + 3, warp_amt=0.03)
    lanes2 = rot_fbm(W, H, ang, 3, 16, 5, seed + 8, warp_amt=0.02)
    lc = np.exp(-((vb + 0.15) / 0.65) ** 2)
    dust = C.smoothstep(0.56, 0.74, lanes) * lc * 0.75 * (0.6 + 0.4 * lanes2) +         C.smoothstep(0.62, 0.82, lanes2) * band * 0.3
    dust = np.clip(dust, 0, 0.85)
    dust = C.blur(dust.astype(np.float32), 0.7 * W / 1920 + 0.3)
    lum = lum * (1 - dust)
    hue = rot_fbm(W, H, ang, 2, 2.5, 3, seed + 4)
    body = np.stack([0.74 + 0.16 * hue, 0.8 + 0.04 * hue, 1.0 - 0.06 * hue], -1)
    warm = np.array([1.0, 0.84, 0.66], np.float32)
    cw = (core * 0.75)[..., None]
    col = body * (1 - cw) + warm * cw
    img = lum[..., None] * col * 0.85 * strength
    # emission nebulae: small bright pink/magenta knots with soft glow, inside the band
    neb = rot_fbm(W, H, ang, 1.5, 8, 5, seed + 11, warp_amt=0.05)
    nebm = C.smoothstep(0.74, 0.93, neb) * np.exp(-(vb / 1.3) ** 2) * (1 - dust * 0.7)
    nebg = C.blur(nebm.astype(np.float32), 0.012 * W)
    img += (nebm * 0.7 + nebg * 0.35)[..., None] * np.array([1.0, 0.3, 0.6], np.float32) * 0.2 * strength
    # cyan-teal glow on the upper flank, violet on the lower flank
    neb2 = rot_fbm(W, H, ang, 3, 3, 4, seed + 13, warp_amt=0.04)
    cy = C.smoothstep(0.45, 0.85, neb2) * np.exp(-((vb - 1.0) / 0.9) ** 2)
    img += cy[..., None] * np.array([0.15, 0.6, 0.9], np.float32) * 0.1 * strength
    vi = np.exp(-((vb + 1.4) / 1.0) ** 2) * (0.5 + neb2)
    img += vi[..., None] * np.array([0.35, 0.18, 0.6], np.float32) * 0.06 * strength
    density = np.clip(band * (1 - dust) * (0.3 + 0.9 * tex), 0, 1)
    return img.astype(np.float32), dust.astype(np.float32), density.astype(np.float32)


def stars(W, H, seed, density_map, n_base, mask_top=None, scale=1.0):
    """Returns (stars RGB, phase map). Many tiny + medium + a few bright glinting stars."""
    rng = np.random.default_rng(seed)
    s = W / 1920
    img = np.zeros((H, W, 3), np.float32)
    phase = np.zeros((H, W), np.float32)
    pal = np.array([[1, 1, 1], [0.78, 0.87, 1.0], [0.7, 0.82, 1.0], [1.0, 0.9, 0.78], [1.0, 0.8, 0.85],
                    [0.85, 0.95, 1.0]], np.float32)
    # 1. dust of tiny stars (point accumulation), density following the milky way
    n = int(n_base * 40 * W * H / (1920 * 1080))
    x = rng.random(n) * W
    y = rng.random(n) * H
    ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    keep = rng.random(n) < (0.05 + 0.95 * density_map[iy, ix] ** 1.3)
    x, y = x[keep], y[keep]
    m = (rng.random(len(x)) ** 4) * 0.9 + 0.08
    ci = rng.integers(0, len(pal), len(x))
    acc = np.zeros((H, W, 3), np.float32)
    for c in range(3):
        np.add.at(acc[..., c], (np.clip(y.astype(int), 0, H - 1), np.clip(x.astype(int), 0, W - 1)),
                  m * pal[ci, c])
    img += C.blur(acc, 0.45 * max(s, 0.5)) * 3.0
    # 2. medium stars (soft round), a bit of twinkle
    n2 = int(n_base * W * H / (1920 * 1080))
    x = rng.random(n2) * W
    y = rng.random(n2) * H
    m = np.minimum(rng.pareto(2.2, n2) * 0.18 + 0.12, 1.6)
    ci = rng.integers(0, len(pal), n2)
    r = np.clip(0.6 + m * 0.7, 0.6, 1.6) * max(s, 0.5)
    ph = rng.uniform(0.3, 1.0, n2)
    for i in range(n2):
        C.splat(img, x[i:i + 1], y[i:i + 1], r[i], pal[ci[i]], m[i] * 1.4)
        xi, yi = int(x[i]), int(y[i])
        R = 2
        phase[max(yi - R, 0):yi + R + 1, max(xi - R, 0):xi + R + 1] = ph[i]
    # 3. bright stars with soft halo + thin 4-point glints
    n3 = int(34 * W * H / (1920 * 1080) * scale) + 6
    x = rng.random(n3) * W
    y = rng.random(n3) ** 1.1 * H
    m = rng.uniform(0.6, 1.4, n3)
    ci = rng.integers(0, len(pal), n3)
    gl = np.zeros((H, W, 3), np.float32)
    for i in range(n3):
        col = pal[ci[i]]
        C.splat(img, x[i:i + 1], y[i:i + 1], 1.3 * max(s, 0.5), col, m[i] * 2.2)
        C.splat(img, x[i:i + 1], y[i:i + 1], 5.0 * s, col * np.array([0.8, 0.9, 1.0]), m[i] * 0.12)
        C.splat(gl, x[i:i + 1], y[i:i + 1], 1.2 * max(s, 0.5), col, m[i] * 1.2)
        xi, yi = int(x[i]), int(y[i])
        R = int(10 * s) + 2
        phase[max(yi - R, 0):yi + R + 1, max(xi - R, 0):xi + R + 1] = rng.uniform(0.4, 1.0)
    L = max(int(0.02 * W) | 1, 5)
    k = np.exp(-np.abs(np.linspace(-1, 1, L)) * 5).astype(np.float32)[None, :]
    gx = cv2.filter2D(gl, -1, k)
    gy = cv2.filter2D(gl, -1, k.T)
    img += (gx + gy) * 0.35
    return img, phase


def comet(W, H, head, direction, length, seed=0, width0=0.004, width1=0.075, strength=1.0,
          tail_col=((0.9, 0.97, 1.0), (0.5, 0.72, 1.0), (0.45, 0.45, 1.0)), coma=1.0, core=1.0, bend=0.0,
          stream_amp=0.35):
    """Comet: coma + tail along `direction` (unit vec pointing away from the sun, i.e. tail direction).
    Returns (rgb additive static, u, vn, body) - u/vn maps let the scene animate streamers per frame."""
    xs, ys = C.grid(W, H)
    dx, dy = direction
    n = math.hypot(dx, dy)
    dx, dy = dx / n, dy / n
    px, py = xs - head[0], ys - head[1]
    along = px * dx + py * dy
    u = along / length
    across = -px * dy + py * dx
    across = across - bend * length * np.clip(u, 0, None) ** 2        # gently curved (dust-tail) bend
    wu = (width0 + (width1 - width0) * np.clip(u, 0, 1) ** 0.8) * W
    vn = across / wu
    fall = C.smoothstep(-0.01, 0.04, u) * np.clip(1 - u, 0, 1) ** 1.6 * (0.35 + 0.65 * np.exp(-u / 0.25))
    body = np.exp(-vn * vn * 1.6) * fall
    edge = np.exp(-((np.abs(vn) - 0.75) / 0.25) ** 2) * fall * 0.35   # brighter limb (hollow cone look)
    ion = np.exp(-(across / (0.0018 * W + 0.004 * W * np.clip(u, 0, 1))) ** 2) * fall * \
        (C.smoothstep(-0.005, 0.02, u)) * 0.9
    # static streamers (per frame the scene adds flowing ones)
    st = np.zeros_like(u)
    rng = np.random.default_rng(seed)
    for f, a in ((5.0, 0.4), (13.0, 0.35), (29.0, 0.25)):
        ph = rng.uniform(0, 6.28)
        st += a * np.sin(vn * f + ph + 2.0 * u)
    st = 0.5 + 0.5 * st
    b = (body * (1 - stream_amp + stream_amp * st) + edge) * strength
    # colour along the tail
    c0, c1, c2 = [np.asarray(c, np.float32) for c in tail_col]
    uu = np.clip(u, 0, 1)[..., None]
    col = np.where(uu < 0.3, c0 + (c1 - c0) * (uu / 0.3), c1 + (c2 - c1) * ((uu - 0.3) / 0.7))
    img = b[..., None] * col * 0.9 + ion[..., None] * np.array([0.75, 0.9, 1.0], np.float32) * 0.8 * strength
    # coma
    d = np.sqrt(px * px + py * py) / W
    cm = 1.0 / (1.0 + (d / 0.005) ** 2) * 0.6 + np.exp(-d / 0.02) * 0.2 + np.exp(-d / 0.08) * 0.05
    img += cm[..., None] * np.array([0.7, 0.9, 1.0], np.float32) * coma * strength
    hot = 1.0 / (1.0 + (d / 0.0016) ** 2.5)
    img += hot[..., None] * np.array([1.0, 1.0, 1.0], np.float32) * 2.2 * core * strength
    return img.astype(np.float32), u.astype(np.float32), vn.astype(np.float32), (body * strength).astype(np.float32)


# ----------------------------------------------------------------------------- land

def ridge_mask(W, H, top, ss=2):
    """Anti-aliased mask of everything below the per-column top line `top` (length W*ss, px in H)."""
    Hs = H
    ys = np.arange(Hs, dtype=np.float32)[:, None]
    m = np.clip(ys - top[None, :] + 0.5, 0, 1)
    if ss > 1:
        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    return m.astype(np.float32)


def tree_bumps(xs, x0, x1, spacing, height, seed, jitter=0.6, width_ratio=0.45, dens=None):
    """Add conifer silhouettes along a ridge: returns bump height per column."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(xs, dtype=np.float32)
    x = x0
    while x < x1:
        hgt = height * rng.uniform(0.55, 1.25)
        if dens is not None and rng.random() > dens(x):
            x += spacing * rng.uniform(0.5, 1.5)
            continue
        w = hgt * width_ratio * rng.uniform(0.8, 1.2)
        i0 = np.searchsorted(xs, x - w)
        i1 = np.searchsorted(xs, x + w)
        seg = xs[i0:i1]
        prof = hgt * np.clip(1 - np.abs(seg - x) / w, 0, 1) ** 1.15
        out[i0:i1] = np.maximum(out[i0:i1], prof)
        x += spacing * rng.uniform(1 - jitter, 1 + jitter)
    return out


def layer_shade(W, H, top_line, m, col_top, col_base, fade_px, rim_col=None, rim_px=2.0, rim_amt=0.0,
                tex=None, tex_amt=0.0):
    """Colour for a mountain layer: gradient from ridge (col_top) down to col_base over fade_px, + rim."""
    ys = np.arange(H, dtype=np.float32)[:, None]
    d = np.clip((ys - top_line[None, :]) / fade_px, 0, 1)
    ct, cb = np.asarray(col_top, np.float32), np.asarray(col_base, np.float32)
    img = ct + (cb - ct) * (d[..., None] ** 0.8)
    if tex is not None:
        img = img * (1 + tex_amt * (tex[..., None] - 0.5))
    if rim_amt:
        dr = np.clip(ys - top_line[None, :], 0, None)
        r = np.exp(-dr / rim_px)
        img = img + r[..., None] * np.asarray(rim_col, np.float32) * rim_amt
    return np.dstack([img, m]).astype(np.float32)


def box_mask(W, H, rects, ss=3):
    """AA mask of many axis-aligned rectangles (x0,y0,x1,y1) px, supersampled."""
    m = np.zeros((H * ss, W * ss), np.uint8)
    for (x0, y0, x1, y1) in rects:
        cv2.rectangle(m, (int(round(x0 * ss)), int(round(y0 * ss))), (int(round(x1 * ss)) - 1,
                      int(round(y1 * ss)) - 1), 255, -1)
    return cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)


def polys_mask(W, H, polys, ss=3):
    m = np.zeros((H * ss, W * ss), np.uint8)
    pts = [(np.asarray(p, np.float64) * ss).astype(np.int32) for p in polys]
    if pts:
        cv2.fillPoly(m, pts, 255, lineType=cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)


def lines_mask(W, H, polylines, width, ss=3):
    m = np.zeros((H * ss, W * ss), np.uint8)
    for pl in polylines:
        p = (np.asarray(pl, np.float64) * ss).astype(np.int32)
        cv2.polylines(m, [p], False, 255, max(1, int(round(width * ss))), cv2.LINE_AA)
    return cv2.resize(m.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_AREA)


def relief(mask, H, light=(0.55, -0.55, 0.62), seed=0, cap=0.08, gully=0.35, angle=90.0, scale=14.0):
    """Painterly slope shading for a mountain silhouette mask: a height field from the distance to the
    ridge plus vertical ridged 'spur/gully' noise, lit from `light` (x right, y down, z toward viewer).
    Returns shade in 0..1 (0.5 = neutral)."""
    h, w = mask.shape
    m8 = (mask > 0.5).astype(np.uint8)
    m8[-1, :] = 1
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5).astype(np.float32)
    hf = np.clip(dist / (cap * H), 0, 1) ** 0.6
    g = rot_fbm(w, h, angle, 4, scale, 5, seed, warp_amt=0.03)
    g = 1 - np.abs(g * 2 - 1)
    g2 = rot_fbm(w, h, angle + 25, 3, scale * 2.2, 4, seed + 1, warp_amt=0.03)
    g2 = 1 - np.abs(g2 * 2 - 1)
    hf = hf + (gully * g + gully * 0.5 * g2) * C.smoothstep(0.0, 0.5, hf)
    hf = C.blur(hf.astype(np.float32), 0.004 * H + 0.5)
    k = 0.02 * H
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3) * k
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3) * k
    nz = 1.0 / np.sqrt(gx * gx + gy * gy + 1)
    nx, ny = -gx * nz, -gy * nz
    L = np.asarray(light, np.float32)
    L = L / np.linalg.norm(L)
    d = nx * L[0] + ny * L[1] + nz * L[2]
    d = (d - np.median(d[m8 > 0])) if m8.any() else d
    return np.clip(0.5 + d * 0.9, 0, 1).astype(np.float32)
