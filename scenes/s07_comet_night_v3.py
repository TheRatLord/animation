"""Round-3 painting helpers for s07_comet_night: Your-Name comet (curved main tail + split secondary tail
with fragment sparkle), a Milky Way painted as granular star clouds with soft feathered dark rifts,
snow-and-rock mountain ranges with irregular couloirs and moonlit crest rims, and stamped conifer forests."""
import math
import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


# ----------------------------------------------------------------------------- comet

def tail_field(W, H, start, direction, length, bend):
    """Curved tail coordinates: u (0 at start .. 1 at the end, along), across (px, signed; curve removed)."""
    xs, ys = C.grid(W, H)
    dx, dy = direction
    n = math.hypot(dx, dy)
    dx, dy = dx / n, dy / n
    px, py = xs - start[0], ys - start[1]
    u = (px * dx + py * dy) / length
    across = -px * dy + py * dx
    across = across - bend * length * np.clip(u, 0, None) ** 2
    return u.astype(np.float32), across.astype(np.float32)


def tail_paint(u, across, W, w0, w1, strength, cols, seed, stria=0.22, core_w=0.0016, core_amt=0.9,
               decay=0.32, power=1.5):
    """Luminous tapering tail: narrow near the nucleus, widening + fading along its length.
    Soft body, brighter limb, a thin ion core near the head and faint lengthwise striae."""
    uc = np.clip(u, 0, 1)
    wu = (w0 + (w1 - w0) * uc ** 0.75) * W
    vn = across / wu
    fall = C.smoothstep(-0.004, 0.02, u) * np.clip(1 - u, 0, 1) ** power * \
        (0.28 + 0.72 * np.exp(-uc / decay))
    body = np.exp(-vn * vn * 1.7)
    limb = np.exp(-((np.abs(vn) - 0.62) / 0.22) ** 2) * 0.22
    rng = np.random.default_rng(seed)
    st = np.zeros_like(u)
    for f, a in ((7.0, 0.45), (15.0, 0.35), (31.0, 0.2)):
        st += a * np.sin(vn * f + rng.uniform(0, 6.28) + 1.5 * uc)
    st = 0.5 + 0.5 * st
    sa = stria * C.smoothstep(0.02, 0.2, uc)
    b = (body * (1 - sa + sa * st) + limb) * fall * strength
    ion = np.exp(-(across / ((core_w + 0.004 * uc) * W)) ** 2) * fall * np.exp(-uc / 0.45) * core_amt * strength
    c0, c1, c2 = [np.asarray(c, np.float32) for c in cols]
    uu = uc[..., None]
    col = np.where(uu < 0.3, c0 + (c1 - c0) * (uu / 0.3), c1 + (c2 - c1) * ((uu - 0.3) / 0.7))
    img = b[..., None] * col + ion[..., None] * np.array([0.8, 0.95, 1.0], np.float32)
    return img.astype(np.float32), (body * fall * strength).astype(np.float32), vn.astype(np.float32)


def comet_head(W, H, head, s, strength=1.0, col=(0.7, 0.9, 1.0)):
    xs, ys = C.grid(W, H)
    d = np.sqrt((xs - head[0]) ** 2 + (ys - head[1]) ** 2) / W
    cm = 1.0 / (1.0 + (d / 0.0045) ** 2) * 0.55 + np.exp(-d / 0.018) * 0.22 + np.exp(-d / 0.07) * 0.06
    img = cm[..., None] * np.asarray(col, np.float32) * strength
    hot = 1.0 / (1.0 + (d / 0.0014) ** 2.5)
    img += hot[..., None] * np.array([1.0, 1.0, 1.0], np.float32) * 2.4 * strength
    return img.astype(np.float32)


def sparkles(W, H, pts, s):
    """Tiny glinting fragment points (x, y, intensity, colour)."""
    out = np.zeros((H, W, 3), np.float32)
    for (x, y, it, col) in pts:
        col = np.asarray(col, np.float32)
        C.splat(out, x, y, 0.9 * s + 0.35, col, 2.2 * it)
        C.splat(out, x, y, 4.0 * s + 0.5, col, 0.12 * it)
        L = (0.006 + 0.006 * it) * W
        R = int(L) + 2
        x0, y0 = int(x) - R, int(y) - R
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + 2 * R + 1, W), min(y0 + 2 * R + 1, H)
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        yy, xx = np.mgrid[cy0:cy1, cx0:cx1].astype(np.float32)
        dx_, dy_ = xx - x, yy - y
        wd = 0.45 * s + 0.3
        k = np.exp(-(dy_ / wd) ** 2) * np.clip(1 - np.abs(dx_) / L, 0, 1) ** 3
        k += np.exp(-(dx_ / wd) ** 2) * np.clip(1 - np.abs(dy_) / L, 0, 1) ** 3
        out[cy0:cy1, cx0:cx1] += k[..., None] * col * 0.6 * it
    return out


# ----------------------------------------------------------------------------- Milky Way

def milky_way3(W, H, p0, p1, width, seed=0, strength=1.0, bend=0.0, core_u=0.1):
    """Milky Way painted as lumpy, granular star clouds with soft feathered dark rifts (warm-edged dust)
    and cool/warm nebular colour. Returns (glow RGB additive, dust (H,W), density (H,W))."""
    s = W / 1920.0
    xs, ys = C.grid(W, H)
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    ang = math.degrees(math.atan2(dy, dx))
    u = ((xs - p0[0]) * ux + (ys - p0[1]) * uy) / L
    v = (-(xs - p0[0]) * uy + (ys - p0[1]) * ux)
    v = v - bend * L * ((u - 0.5) ** 2 * 4 - 1)
    wob = (P.rot_fbm(W, H, ang, 6, 1.5, 3, seed + 1) - 0.5) * width * 0.7
    vb = (v + wob) / width
    w_u = 1.0 + 0.45 * np.clip(1 - u, 0, 1) ** 2
    band = np.exp(-(vb / w_u) ** 2 * 1.1)
    halo = np.exp(-(vb / (w_u * 2.6)) ** 2)
    core = np.exp(-((u - core_u) / 0.3) ** 2)
    # star clouds: lumpy mid-scale brightness + fine granular texture
    big = P.rot_fbm(W, H, ang, 2.2, 4.5, 5, seed + 2, warp_amt=0.06)
    mid = P.rot_fbm(W, H, ang, 1.6, 13, 4, seed + 17, warp_amt=0.05)
    sc = np.clip(C.smoothstep(0.38, 0.78, big) * 0.75 + C.smoothstep(0.45, 0.85, mid) * 0.5, 0, 1)
    rng = np.random.default_rng(seed + 5)
    g = rng.random((H // 2 + 2, W // 2 + 2)).astype(np.float32)
    g = cv2.resize(C.blur(g, 0.7), (W, H), interpolation=cv2.INTER_LINEAR)[:H, :W]
    grain = np.clip((g - 0.5) * 3.0 + 0.5, 0, 1)
    lum = band * (0.25 + 1.5 * sc ** 1.4) * (0.72 + 0.28 * core) * (0.75 + 0.5 * grain) + halo * 0.1
    # dark rifts: a main (great) rift slightly off-centre that breaks into segments + scattered patches,
    # all feathered (multi-scale blur) with fine torn fingers
    rc = 0.15 + 0.7 * (P.rot_fbm(W, H, ang, 4, 2.2, 3, seed + 31) - 0.5)
    zone = np.exp(-((vb - rc) / 0.95) ** 2)
    dn = P.rot_fbm(W, H, ang, 1.8, 12, 6, seed + 34, warp_amt=0.2)
    dn2 = P.rot_fbm(W, H, ang + 15, 1.2, 20, 4, seed + 35, warp_amt=0.15)
    fld = dn * 0.62 + dn2 * 0.38 + zone * 0.2 - 0.1
    dust = C.smoothstep(0.5, 0.68, fld) * np.exp(-(vb / 1.6) ** 2) * (0.55 + 0.45 * zone)
    dust = np.clip(dust, 0, 0.95).astype(np.float32)
    dust = 0.45 * C.blur(dust, 0.8 * s + 0.4) + 0.35 * C.blur(dust, 3 * s + 0.5) + 0.2 * C.blur(dust, 9 * s + 1)
    dust = np.clip(dust * 0.85, 0, 0.72).astype(np.float32)
    glow = lum * (1 - dust)
    # colour: cool blue-white body, warm cream core / bright clouds, warm-lit dust edges, cyan + violet flanks
    hue = P.rot_fbm(W, H, ang, 2, 2.5, 3, seed + 4)
    cool = np.stack([0.66 + 0.1 * hue, 0.78 + 0.05 * hue, 1.0 + 0 * hue], -1)
    warm = np.array([1.0, 0.86, 0.66], np.float32)
    cw = np.clip(core * 0.75 + sc * 0.25, 0, 1)[..., None]
    col = cool * (1 - cw) + warm * cw
    img = glow[..., None] * col * 0.8 * strength
    edge = np.clip(C.blur(dust, 5 * s + 0.5) - dust * 0.8, 0, 1) * band
    img += edge[..., None] * np.array([1.0, 0.62, 0.42], np.float32) * 0.3 * strength
    img += (dust * band)[..., None] * np.array([0.1, 0.05, 0.06], np.float32) * strength
    neb = P.rot_fbm(W, H, ang, 1.5, 8, 5, seed + 11, warp_amt=0.05)
    nebm = C.smoothstep(0.76, 0.92, neb) * np.exp(-(vb / 1.2) ** 2) * (1 - dust)
    nebg = C.blur(nebm.astype(np.float32), 0.01 * W)
    img += (nebm * 0.4 + nebg * 0.5)[..., None] * np.array([1.0, 0.36, 0.62], np.float32) * 0.2 * strength
    neb2 = P.rot_fbm(W, H, ang, 3, 3, 4, seed + 13, warp_amt=0.04)
    cy = C.smoothstep(0.45, 0.85, neb2) * np.exp(-((vb - 1.1) / 0.9) ** 2)
    img += cy[..., None] * np.array([0.12, 0.55, 0.85], np.float32) * 0.1 * strength
    vi = np.exp(-((vb + 1.4) / 1.0) ** 2) * (0.5 + neb2)
    img += vi[..., None] * np.array([0.42, 0.2, 0.62], np.float32) * 0.06 * strength
    density = np.clip(band * (1 - dust) ** 1.5 * (0.25 + 1.0 * sc), 0, 1)
    return img.astype(np.float32), dust, density.astype(np.float32)


def fine_stars(W, H, seed, density, n, bright=1.0):
    """Very many faint point stars distributed by density (granular star clouds)."""
    rng = np.random.default_rng(seed)
    s = W / 1920.0
    x = rng.random(n) * W
    y = rng.random(n) * H
    ix, iy = np.clip(x.astype(int), 0, W - 1), np.clip(y.astype(int), 0, H - 1)
    keep = rng.random(n) < np.minimum(density[iy, ix], 0.75) ** 1.6
    x, y = x[keep], y[keep]
    m = rng.random(len(x)) ** 3 * 0.55 + 0.1
    warm = rng.random(len(x)) < 0.45
    col = np.where(warm[:, None], np.array([1.0, 0.88, 0.74]), np.array([0.82, 0.9, 1.0])).astype(np.float32)
    acc = np.zeros((H, W, 3), np.float32)
    yi = np.clip(y.astype(int), 0, H - 1)
    xi = np.clip(x.astype(int), 0, W - 1)
    for c in range(3):
        np.add.at(acc[..., c], (yi, xi), m * col[:, c])
    return C.blur(acc, 0.42 * max(s, 0.5)) * 2.6 * bright


# ----------------------------------------------------------------------------- mountains

def _smooth1d(a, sigma):
    return cv2.GaussianBlur(a.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=max(sigma, 0.5))[0]


def paint_range(Wp, Hl, top, ss, s, seed, light, pal, y_base, haze_px, gully_px, snow_px, lean=1.1,
                haze_amt=0.6, rim_col=(0.6, 0.85, 1.0), rim_amt=0.9, rim_px=1.3, snow_amt=1.0,
                tree_px=None, relief=1.0, H=1080):
    """Mountain range painted as rock planes + snow couloirs.

    top: supersampled crest line (len Wp*ss). light: (x, y) of the comet (plate px).
    pal: dict rock_sh, rock_lit, snow_sh, snow_lit, haze, forest.
    Gullies follow the local fall line (lean along the crest slope), snow fills the upper face and runs
    down the couloirs in irregular tongues; exposed rock ribs are left dark. Crest gets a crisp rim toward
    the comet. Returns RGBA plate and the forest mask (for tree stamping)."""
    ysl = np.arange(Hl, dtype=np.float32)[:, None]
    msk = np.clip(ysl - top[None, :] + 0.5, 0, 1)
    msk = cv2.resize(msk, (Wp, Hl), interpolation=cv2.INTER_AREA)
    top1 = top.reshape(Wp, ss).mean(1)
    xs, ys = C.grid(Wp, Hl)
    d = np.clip(ys - top1[None, :], 0, None)
    ts = _smooth1d(top1, 0.012 * Wp)
    g = np.clip(np.gradient(ts), -2.5, 2.5)
    xc = (xs - lean * g[None, :] * d).astype(np.float32)
    # noise textures in fall-line coordinates (elongated down the slope)
    pad = int(0.3 * Wp)
    Tw = Wp + 2 * pad

    def tex(stretch, cells, octv, sd, warp):
        t = P.rot_fbm(Tw, Hl, 90.0, stretch, Tw / cells, octv, sd, warp_amt=warp)
        return cv2.remap(t, xc + pad, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    n1 = tex(3.2, gully_px, 5, seed, 0.1)
    n2 = tex(2.4, gully_px * 0.42, 4, seed + 7, 0.08)
    n3 = tex(1.4, gully_px * 0.12, 3, seed + 13, 0.05)
    r1 = 1 - np.abs(n1 * 2 - 1)          # ribs (sharp crest lines) / broad gullies
    r2 = 1 - np.abs(n2 * 2 - 1)
    ramp = C.smoothstep(0, 0.02 * H, d)
    hgt = d * 0.35 + (r1 * gully_px * 0.5 + r2 * gully_px * 0.22) * ramp * relief
    hgt = C.blur(hgt.astype(np.float32), 0.5 * s + 0.3)
    hx = np.gradient(hgt, axis=1)
    hy = np.gradient(hgt, axis=0)
    lx, ly = light[0] - xs, light[1] - ys
    ln = np.sqrt(lx * lx + ly * ly) + 1e-3
    lx, ly = lx / ln, ly / ln
    nz = 1.0 / np.sqrt(hx * hx + hy * hy + 1)
    dot = (-hx * lx * 0.85 - hy * ly * 0.5 + 0.35) * nz
    inside = msk > 0.5
    med = float(np.median(dot[inside])) if inside.any() else 0.0
    v = np.clip(0.5 + (dot - med) * 2.2, 0, 1)
    # painterly: 3 flat value steps with narrow soft transitions
    vq = (C.smoothstep(0.34, 0.4, v) + C.smoothstep(0.62, 0.68, v)) * 0.5
    vp = 0.72 * vq + 0.28 * v
    prox = np.exp(-np.sqrt((xs - light[0]) ** 2 + (ys - light[1]) ** 2) / (0.7 * Wp))
    pal = {k: np.asarray(c, np.float32) for k, c in pal.items()}
    rock = pal['rock_sh'] + (pal['rock_lit'] - pal['rock_sh']) * vp[..., None]
    rock = rock * (0.85 + 0.3 * prox[..., None])
    # painted value variation inside the rock planes: broad patches, gullies that deepen toward the valley
    patch = cv2.GaussianBlur(n1, (0, 0), 0.01 * Wp) - 0.5
    gdeep = np.clip(1 - r1, 0, 1) ** 2 * C.smoothstep(0.01 * H, 0.16 * H, d)
    rock = rock * (1 + 0.35 * patch[..., None]) * (1 - 0.42 * gdeep[..., None]) *         (1 - 0.22 * C.smoothstep(0.0, 0.22 * H, d)[..., None])
    rock = rock + (n3 - 0.5)[..., None] * 0.035 * (pal['rock_lit'] - pal['rock_sh'])
    # snow: upper face + tongues down the gullies (low r1), torn edge from fine noise
    sd_x = snow_px * np.clip(0.2 + 1.1 * (P.fbm1d(xs[0] / Wp, 5, 3, seed + 51) * 0.5 + 0.5), 0.1, 1.4)
    gl = np.clip(1 - r1, 0, 1) ** 1.3
    reach = sd_x[None, :] * (0.1 + 2.2 * gl ** 1.5 + 0.4 * (n2 - 0.5)) + (n3 - 0.5) * 0.01 * H
    reach = reach * (0.7 + 0.6 * v)
    snow = C.smoothstep(-1.0, 1.0, reach - d)
    # snow caught on ledges / gentler lit planes: short horizontal-ish streaks lower on the face
    led = cv2.remap(P.rot_fbm(Tw, Hl, 8.0, 5.0, Tw / (gully_px * 0.6), 4, seed + 71, warp_amt=0.06),
                    xc + pad, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    ledge = C.smoothstep(0.66, 0.72, led) * C.smoothstep(0.45, 0.6, v) *         (1 - C.smoothstep(sd_x[None, :] * 2.5, sd_x[None, :] * 3.5, d))
    snow = np.maximum(snow, ledge * 0.9) * snow_amt
    snow = snow * C.smoothstep(0.1, 0.32, (1 - r2) * 0.45 + gl * 0.8)       # ribs stay bare
    vs = np.clip(0.25 + 0.75 * v, 0, 1) ** 1.2
    snowc = pal['snow_sh'] + (pal['snow_lit'] - pal['snow_sh']) * vs[..., None]
    snowc = snowc * (0.8 + 0.4 * prox[..., None])
    # snow: brighter crust near the crest, cooler/darker down the couloirs, soft wind-sculpted variation
    crest = np.exp(-d / (0.03 * H))
    snowc = snowc * (0.84 + 0.2 * crest[..., None] + 0.16 * (n2 - 0.5)[..., None]) *         (1 - 0.25 * C.smoothstep(0.02 * H, 0.2 * H, d)[..., None])
    snowc = snowc + (0.5 - n3)[..., None] * 0.05 * (pal['snow_lit'] - pal['snow_sh'])
    img = rock * (1 - snow[..., None]) + snowc * snow[..., None]
    forest = np.zeros((Hl, Wp), np.float32)
    if tree_px is not None:
        tl = tree_px * (0.75 + 0.5 * P.fbm1d(xs[0] / Wp, 11, 3, seed + 61))
        tb = (y_base - tl)[None, :] + (n2 - 0.5) * 0.02 * H + (r1 - 0.5) * 0.015 * H
        forest = C.smoothstep(-1.0, 1.0, ys - tb) * inside
        img = img * (1 - forest[..., None]) + pal['forest'] * forest[..., None]
    hz = C.smoothstep(y_base - haze_px, y_base + 0.15 * haze_px, ys) * haze_amt
    img = img * (1 - hz[..., None]) + pal['haze'] * hz[..., None]
    # crest rim toward the comet
    m8 = inside.astype(np.uint8)
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3).astype(np.float32)
    gm = C.blur(msk, 2.0 * s + 0.5)
    gx, gy = np.gradient(gm, axis=1), np.gradient(gm, axis=0)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    facing = np.clip((-gx * lx - gy * ly) / gn, 0, 1)
    rim = np.exp(-dist / (rim_px * s + 0.3)) * (0.15 + 0.85 * facing) * (0.3 + 0.7 * prox)
    rim *= (0.55 + 0.45 * snow + 0.0 * forest)
    img = img + rim[..., None] * np.asarray(rim_col, np.float32) * rim_amt
    return np.dstack([img, msk]).astype(np.float32), forest.astype(np.float32)


def stamp_forest(rgba, region, seed, s, light_x, size_fn, col_dark, col_lit, spacing=1.0, ss=3,
                 x_range=None, rim_col=None, density_fn=None):
    """Stamp conifer silhouettes (two-tone: lit half toward the light) inside `region` (mask), back to
    front. size_fn(y) -> tree height px. Composites onto rgba (straight alpha) in place."""
    Hh, Ww = region.shape
    rng = np.random.default_rng(seed)
    ys_, xs_ = np.nonzero(region > 0.5)
    if len(ys_) == 0:
        return rgba
    y0, y1 = ys_.min(), ys_.max()
    xa, xb = (0, Ww) if x_range is None else x_range
    trees = []
    y = float(y0)
    while y <= y1 + 2:
        h = size_fn(y)
        stepx = max(h * 0.3 * spacing, 1.2)
        x = xa + rng.uniform(0, stepx)
        while x < xb:
            xi, yi = int(x), int(min(y, Hh - 1))
            if 0 <= xi < Ww and region[yi, xi] > 0.5 and (density_fn is None or rng.random() < density_fn(x, y)):
                hh = h * rng.uniform(0.65, 1.35)
                trees.append((y + rng.uniform(-0.3, 0.3) * h * 0.3, x + rng.uniform(-0.3, 0.3) * stepx, hh,
                              rng.uniform(-0.12, 0.12)))
            x += stepx * rng.uniform(0.7, 1.3)
        y += max(h * 0.28 * spacing, 1.0)
    trees.sort(key=lambda a: a[0])
    cv_ = np.zeros((Hh * ss, Ww * ss, 3), np.float32)
    al = np.zeros((Hh * ss, Ww * ss), np.uint8)
    cd, cl = np.asarray(col_dark, np.float32), np.asarray(col_lit, np.float32)
    for (ty, tx, th, jit) in trees:
        w = th * 0.36
        ap = (tx * ss, (ty - th) * ss)
        bl = ((tx - w / 2) * ss, ty * ss)
        br = ((tx + w / 2) * ss, ty * ss)
        bm = (tx * ss, ty * ss)
        # a couple of branch tiers for a serrated conifer outline
        tierL = ((tx - w * 0.36) * ss, (ty - th * 0.45) * ss)
        tierR = ((tx + w * 0.36) * ss, (ty - th * 0.45) * ss)
        inL = ((tx - w * 0.2) * ss, (ty - th * 0.5) * ss)
        inR = ((tx + w * 0.2) * ss, (ty - th * 0.5) * ss)
        left = np.array([ap, inL, tierL, bl, bm], np.float64).astype(np.int32)
        right = np.array([ap, inR, tierR, br, bm], np.float64).astype(np.int32)
        lit_right = light_x > tx
        f = 1.0 + jit
        cdk = tuple(float(c) for c in cd * f)
        clt = tuple(float(c) for c in cl * f)
        cv2.fillPoly(cv_, [left], clt if not lit_right else cdk)
        cv2.fillPoly(cv_, [right], clt if lit_right else cdk)
        cv2.fillPoly(al, [left, right], 255)
    a = cv2.resize(al.astype(np.float32) / 255, (Ww, Hh), interpolation=cv2.INTER_AREA)
    c = cv2.resize(cv_, (Ww, Hh), interpolation=cv2.INTER_AREA)
    rgb = rgba[..., :3] * (1 - a[..., None]) + c
    if rim_col is not None:
        # thin highlight on tree tops facing the light (top edge of the canopy silhouette)
        sh = np.zeros_like(a)
        dd = max(int(round(1.2 * s)), 1)
        sh[dd:] = a[:-dd]
        rim = np.clip(a - sh, 0, 1)
        rgb = rgb + rim[..., None] * np.asarray(rim_col, np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.maximum(rgba[..., 3], a)
    return rgba
