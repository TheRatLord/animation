"""Technique A - distance-field cloud painter (Shinkai-style painted cumulus).

Idea
----
A cloud is a short list of MASS GROUPS (back -> front). Each group is a few huge ellipses ("the
masses") whose outline is then roughened hierarchically: circles of radius r, r/2, r/4 ... are placed
ONLY on the current boundary (sampled along the contour, mostly on up/side-facing normals, almost
never on the base), so cauliflower detail lives on the silhouette while the interior stays one big
smooth shape. The masks are drawn supersampled and domain-warped (large + mid + fine) for irregularity.

Shading is NOT per lobe. Each group gets a large-scale "pillow" heightfield computed from the distance
transform of its whole mask (sqrt-dome profile clipped to a plateau, then heavily blurred, computed at
low resolution); its gradient gives a normal and N.L one broad soft terminator per big mass. Near the
silhouette the normal is bent by a small-scale edge normal (edge bumps catch light / turn into shadow)
but that influence dies within a few % of the cloud size, so interiors are large smooth value masses.

Groups are composited painter-style: where a front mass overlaps a back mass there is a crisp edge
whose crispness is modulated along its length (it fades out in places), with a thin lighter rim on
the front mass's sun-facing edge and a slight shadow on the back mass beside it. The union gets a
crisp 2-4 px silver/gold rim on sun-facing edges, translucent glow on edges near the sun, torn wispy
bases, fibrous anvil edges, aerial haze and low-contrast anisotropic brush-stroke texture.

Public API (sizes in plate pixels; everything float32)
------------------------------------------------------
    PALETTES / palette(name_or_dict)
    stroke_noise(W, H, angle, length, width, seed)              -> (H, W) ~N(0,1) anisotropic noise
    build_masks(W, H, groups, seed, ss=2)                       -> [ (H, W) alpha per group ]
    paint(W, H, groups, masks, light, pal, sun=None, ...)       -> (H, W, 4) RGBA, straight alpha
    render_cloud(W, H, groups, light, pal, sun, seed, **paint_kw)    -> RGBA (build_masks + paint)
    render_clouds(W, H, [groups, ...], ...)                     -> RGBA of several clouds (back->front)
    cumulonimbus_groups(cx, base_y, width, height, seed, wind)  -> groups for a CB tower + anvil
    cumulus_groups(cx, base_y, width, height, seed, towers)     -> groups for a (towering) cumulus
    cloud_sea(W, H, horizon, sun, seed, ...)                    -> dict of RGBA layer plates (far->near)

Group spec (dict):
    ell   : [(cx, cy, rx, ry[, angle_deg]), ...]  the huge masses (px)
    S     : size reference (px) for bump radii / softness
    levels: bump levels (default 5), r0: first bump radius as a fraction of S (0.15)
    prob  : callable(u, x, y) -> probability of a bump at a boundary point whose outward normal has
            up-ness u (1 up, -1 down) at plate position x, y
    base  : y (px) where the group is torn off (wispy base), tear: ragged band height (fraction of S)
    fib   : dict(wind=-1|1, x0=px, len=0.3, amt=0.8) fibrous downwind edge (anvils)
    haze  : aerial perspective 0..1, shade: extra darkening 0..1, pillow: plateau size (0.55),
    warp  : large domain-warp amplitude (fraction of S), crisp: overlap-edge crispness multiplier,
    edge  : edge-detail multiplier, base_dark: underside darkening.
"""
import math
import numpy as np
import cv2

from . import core as C
from .fx import fast_blur


# ============================================================================ palettes

def _c(h):
    return C.hex2rgb(h) if isinstance(h, str) else np.asarray(h, np.float32)


PALETTES = {
    # ramp along N.L: deep -> shadow -> mid -> lit -> hot. base: underside tint, rim: silver lining,
    # glow: translucent edge glow near the sun, haze: aerial perspective colour
    'summer': dict(deep='#6f7cc4', shadow='#96a1dc', mid='#c8c8ec', lit='#f8f5f3', hot='#fffcf4',
                   base='#7c83c0', rim='#fffdf2', glow='#fff4dc', haze='#b9dcf4'),
    'sunrise': dict(deep='#302a70', shadow='#62509e', mid='#c07aa4', lit='#ffb487', hot='#ffe2a6',
                    base='#3a2f70', rim='#ffe6aa', glow='#ffc27e', haze='#d99cb6', lit2='#f59aa6'),
}


def palette(p):
    if isinstance(p, str):
        p = PALETTES[p]
    return {k: _c(v) for k, v in p.items()}


# ============================================================================ noise helpers

def _lnoise(W, H, cells, octaves, seed, q=8):
    """Signed (-1..1) smooth fbm generated at 1/q res and upsampled (fast, for warps/modulation)."""
    w, h = max(int(W // q), 8), max(int(H // q), 8)
    n = C.fbm(w, h, max(cells, 1.0), octaves, seed=seed) * 2 - 1
    return cv2.resize(n.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)


def stroke_noise(W, H, angle=-8.0, length=40.0, width=4.0, seed=0, octaves=2):
    """Anisotropic brush-stroke noise (~zero mean, unit std): elongated blobs `length` x `width` px
    oriented at `angle` degrees. Use as a low-contrast multiplier for painted texture."""
    D = int(math.hypot(W, H)) + 8
    rng = np.random.default_rng(seed)
    out = np.zeros((D, D), np.float32)
    amp = 1.0
    for o in range(octaves):
        l, w = length / 2 ** o, width / 2 ** o
        g = rng.standard_normal((max(int(D / w), 2), max(int(D / l), 2))).astype(np.float32)
        out += amp * cv2.resize(g, (D, D), interpolation=cv2.INTER_CUBIC)
        amp *= 0.55
    M = cv2.getRotationMatrix2D((D / 2, D / 2), -angle, 1.0)
    M[0, 2] -= (D - W) / 2
    M[1, 2] -= (D - H) / 2
    out = cv2.warpAffine(out, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return (out - out.mean()) / (out.std() + 1e-6)


def _warp_u8(m, amp, cells, seed, octaves=3):
    """Domain-warp a uint8 mask by smooth noise displacement of `amp` px, re-binarised."""
    H, W = m.shape
    dx = _lnoise(W, H, cells, octaves, seed, q=16) * amp
    dy = _lnoise(W, H, cells, octaves, seed + 17, q=16) * amp
    dx += np.arange(W, dtype=np.float32)[None, :]
    dy += np.arange(H, dtype=np.float32)[:, None]
    w = cv2.remap(m, dx, dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return np.where(w > 127, 255, 0).astype(np.uint8)


# ============================================================================ silhouette

def _default_prob(u, x, y):
    # u = up-ness of the outward normal (1 = facing up, -1 = facing down (base))
    return float(np.interp(u, [-1.0, -0.55, -0.15, 0.25, 1.0], [0.0, 0.03, 0.45, 0.85, 0.95]))


def _add_bumps(m, r, rng, prob, ss, spacing=1.7, depth=(0.1, 0.6), rjit=(0.55, 1.35), squash=1.0, concave=0.15):
    """Place circles of radius ~r (ss px) along the boundary of mask m (uint8, modified in place)."""
    H, W = m.shape
    q = 4
    small = cv2.resize(m, (max(W // q, 2), max(H // q, 2)), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small = cv2.GaussianBlur(small, (0, 0), max(1.0, r / q * 0.6))
    gy, gx = np.gradient(small)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    circles = []
    for c in cnts:
        P = c[:, 0, :].astype(np.float32)
        if len(P) < 12:
            continue
        d = np.sqrt(((np.roll(P, -1, 0) - P) ** 2).sum(1))
        s = np.concatenate([[0.0], np.cumsum(d)[:-1]])
        L = s[-1] + d[-1]
        if L < 3 * r:
            continue
        pos = rng.uniform(0, r * spacing)
        while pos < L:
            i = min(int(np.searchsorted(s, pos)), len(P) - 1)
            x, y = P[i]
            sx, sy = min(int(x / q), small.shape[1] - 1), min(int(y / q), small.shape[0] - 1)
            nx, ny = -gx[sy, sx], -gy[sy, sx]
            nn = math.hypot(nx, ny)
            step = r * spacing * rng.uniform(0.6, 1.4)
            if nn > 1e-7:
                nx, ny = nx / nn, ny / nn
                # keep the V-notches between lobes: few bumps in concave spots
                ox = min(max(int((x + nx * r * 0.7) / q), 0), small.shape[1] - 1)
                oy = min(max(int((y + ny * r * 0.7) / q), 0), small.shape[0] - 1)
                cv = concave if small[oy, ox] > 0.3 else 1.0
                if rng.random() < prob(-ny, x / ss, y / ss) * cv:
                    rr = r * rng.uniform(*rjit)
                    dd = rr * rng.uniform(*depth)
                    circles.append((x - nx * dd, y - ny * dd, rr))
                    step = rr * spacing * rng.uniform(0.8, 1.3)
            pos += step
    for (x, y, rr) in circles:
        cv2.ellipse(m, (int(x * 4), int(y * 4)), (int(rr * 4 * squash), int(rr * 4)),
                    0, 0, 360, 255, -1, cv2.LINE_8, 2)
    return m


def build_masks(W, H, groups, seed=0, ss=2):
    """Rasterise each group's silhouette (supersampled x ss). Returns [ (H, W) float32 AA alpha ]."""
    out = []
    Ws, Hs = W * ss, H * ss
    for gi, g in enumerate(groups):
        rng = np.random.default_rng(seed * 1000 + gi * 37 + 5)
        S = float(g['S']) * ss
        ex = [(e[0] * ss, e[1] * ss, max(e[2], e[3]) * ss) for e in g['ell']]
        ex += [((r[0] + r[2]) / 2 * ss, (r[1] + r[3]) / 2 * ss, max(r[2] - r[0], r[3] - r[1]) / 2 * ss)
               for r in g.get('rects', [])]
        mg = S * 0.4 + 8
        fibl = g['fib'].get('len', 0.3) * S * 1.5 if g.get('fib') else 0
        x0 = int(max(min(e[0] - e[2] for e in ex) - mg - fibl, 0))
        x1 = int(min(max(e[0] + e[2] for e in ex) + mg + fibl, Ws))
        y0 = int(max(min(e[1] - e[2] for e in ex) - mg, 0))
        y1 = int(min(max(e[1] + e[2] for e in ex) + mg, Hs))
        x0, y0 = x0 - x0 % ss, y0 - y0 % ss
        x1, y1 = x1 - x1 % ss, y1 - y1 % ss
        if x1 <= x0 + 4 * ss or y1 <= y0 + 4 * ss:
            out.append(np.zeros((H, W), np.float32))
            continue
        cw, ch = x1 - x0, y1 - y0
        m = np.zeros((ch, cw), np.uint8)
        for e in g['ell']:
            cx, cy, rx, ry = (e[0] * ss - x0), (e[1] * ss - y0), e[2] * ss, e[3] * ss
            ang = e[4] if len(e) > 4 else 0.0
            cv2.ellipse(m, (int(cx * 4), int(cy * 4)), (int(rx * 4), int(ry * 4)), ang, 0, 360, 255, -1,
                        cv2.LINE_8, 2)
        for r in g.get('rects', []):
            cv2.rectangle(m, (int(r[0] * ss - x0), int(r[1] * ss - y0)), (int(r[2] * ss - x0), int(r[3] * ss - y0)), 255, -1)
        # large irregularity of the masses
        m = _warp_u8(m, g.get('warp', 0.035) * S, max(cw / S * 1.2, 2.0), seed + 11 * gi)
        prob = g.get('prob') or _default_prob

        lvl_p = g.get('lvl_p', (1.0, 0.9, 0.7, 0.45, 0.3))
        cur = [1.0]

        def pr(u, x, y, prob=prob):
            return prob(u, x + x0 / ss, y + y0 / ss) * cur[0]

        r = g.get('r0', 0.2) * S
        for lev in range(g.get('levels', 4)):
            cur[0] = lvl_p[min(lev, len(lvl_p) - 1)]
            _add_bumps(m, r, rng, pr, ss, spacing=g.get('spacing', 1.7), squash=g.get('squash', 1.0),
                       depth=(0.0, 0.45) if lev == 0 else (0.1, 0.55))
            if lev in (0, 2):
                # re-warp so bumps are not perfect circles
                m = _warp_u8(m, r * 0.3, max(cw / (r * 2.5), 2.0), seed + 101 * gi + lev, octaves=2)
            r *= 0.5
            if r < 1.8 * ss:
                break
        # fine edge irregularity
        m = _warp_u8(m, 1.3 * ss, max(cw / (7.0 * ss), 2.0), seed + 7 * gi + 3, octaves=2)
        A = m.astype(np.float32) / 255.0
        # torn, uneven base
        if g.get('base') is not None:
            by = g['base'] * ss - y0
            yy = np.arange(ch, dtype=np.float32)[:, None]
            xx = np.arange(cw, dtype=np.float32)[None, :]
            wav = _lnoise(cw, 8, cw / (S * 0.6), 3, seed + 9 * gi, q=4)[4] * S * 0.035
            tilt = g.get('base_tilt', 0.0) * (xx - cw / 2)
            srng = np.random.default_rng(seed + 3 * gi)
            streak = cv2.resize(srng.random((max(int(ch / (S * 0.025)), 2), max(int(cw / (S * 0.12)), 2))).astype(np.float32),
                                (cw, ch), interpolation=cv2.INTER_CUBIC)
            band = S * g.get('tear', 0.06)
            cut = by + wav[None, :] + tilt - band * C.smoothstep(0.2, 0.9, streak)
            A = A * C.smoothstep(cut + band * 0.35, cut - band * 0.15, yy)
        # fibrous downwind edge (anvils)
        if g.get('fib'):
            f = g['fib']
            wind = f.get('wind', -1)
            k = max(S * f.get('len', 0.3), 3)
            sm = cv2.GaussianBlur(A, (0, 0), sigmaX=k * 0.45, sigmaY=max(ss * 1.2, 1))
            M = np.float32([[1, 0, wind * k * 0.55], [0, 1, 0]])
            sm = cv2.warpAffine(sm, M, (cw, ch))
            frng = np.random.default_rng(seed + 77 + gi)
            st = cv2.resize(frng.random((max(ch // int(7 * ss), 2), max(int(cw / k * 1.5), 2))).astype(np.float32),
                            (cw, ch), interpolation=cv2.INTER_CUBIC) * 0.65 +                 cv2.resize(frng.random((max(ch // int(2.5 * ss), 2), max(int(cw / k * 3), 2))).astype(np.float32),
                           (cw, ch), interpolation=cv2.INTER_CUBIC) * 0.35
            fx0 = f.get('x0', 0) * ss - x0
            xx = np.arange(cw, dtype=np.float32)[None, :]
            side = C.smoothstep(-S * 0.1, S * 0.3, (fx0 - xx) * (-wind))
            fibm = np.clip(sm * 1.5 - 0.2, 0, 1) * C.smoothstep(0.3, 0.85, st) * side
            A = np.maximum(A * (1 - side * 0.35 * (1 - C.smoothstep(0.2, 0.7, st))), fibm * f.get('amt', 0.8))
        a1 = cv2.resize(A, (cw // ss, ch // ss), interpolation=cv2.INTER_AREA)
        full = np.zeros((H, W), np.float32)
        full[y0 // ss:y0 // ss + a1.shape[0], x0 // ss:x0 // ss + a1.shape[1]] = a1
        out.append(full)
    return out


# ============================================================================ shading

def _sobel(h):
    gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy


def _ramp(x, stops, cols):
    """Piecewise-linear colour ramp of scalar field x."""
    out = np.empty(x.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(x, stops, [col[c] for col in cols])
    return out


def _dt(mask_bool):
    return cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)


def _pillow_grad(A, pillow=0.55):
    """Large-scale pillow heightfield of mask A -> smooth full-res gradient (gx, gy) (dimensionless).
    h = D*sqrt(1-(1-min(DT/D,1))^2) (round shoulder, flat plateau), blurred by 0.4 D. Computed at low
    resolution and upsampled bicubically (no blocky derivatives)."""
    H, W = A.shape
    rows = np.where((A > 0.5).any(1))[0]
    cols = np.where((A > 0.5).any(0))[0]
    if len(rows) < 2 or len(cols) < 2:
        z = np.zeros((H, W), np.float32)
        return z, z
    ext = min(rows[-1] - rows[0], cols[-1] - cols[0])
    q = int(np.clip(ext / 120, 1, 8))
    w, h = max(W // q, 4), max(H // q, 4)
    a = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    DT = _dt(a > 0.5)
    D = max(float(DT.max()) * pillow, 2.0)
    x = np.clip(DT / D, 0, 1)
    hf = D * np.sqrt(1 - (1 - x) ** 2)
    hf = C.blur(hf, D * 0.4)
    gx, gy = _sobel(hf)
    gx = cv2.resize(gx, (W, H), interpolation=cv2.INTER_CUBIC)
    gy = cv2.resize(gy, (W, H), interpolation=cv2.INTER_CUBIC)
    return gx, gy


def _light_march(A, lx, ly, dist, crisp=None, steps=24):
    """Fraction of light reaching each pixel of mask A from direction (lx, ly) (screen, toward the
    light), marching `dist` px through the mask. 1 = lit (near a sun-facing edge), 0 = deep shadow.
    Computed at reduced resolution; slightly softened (softness varies with `crisp`)."""
    H, W = A.shape
    n = math.hypot(lx, ly)
    if n < 1e-4 or dist < 2:
        return np.ones_like(A)
    lx, ly = lx / n, ly / n
    q = int(np.clip(dist / 40, 1, 6))
    w, h = max(W // q, 2), max(H // q, 2)
    a = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
    acc = np.zeros_like(a)
    tot = 0.0
    for k in range(1, steps + 1):
        d = dist / q * k / steps
        M = np.float32([[1, 0, lx * d], [0, 1, ly * d]])
        # sample a at p + d*L  (inverse map)
        sh = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT)
        wt = 1.0 - 0.5 * k / steps
        acc += sh * wt
        tot += wt
    occ = acc / tot
    occ = C.blur(occ, max(dist / q * 0.035, 0.6))
    lit = 1.0 - C.smoothstep(0.1, 0.65, occ)
    lit = cv2.resize(lit, (W, H), interpolation=cv2.INTER_CUBIC)
    soft = C.blur(lit, max(dist * 0.015, 1.0))
    if crisp is not None:
        lit = lit * crisp + soft * (1 - crisp)
    else:
        lit = soft
    return np.clip(lit, 0, 1).astype(np.float32)


def _bbox(A, margin, thr=0.004):
    H, W = A.shape
    rows = np.where((A > thr).any(1))[0]
    cols = np.where((A > thr).any(0))[0]
    if len(rows) == 0:
        return None
    m = int(margin)
    return max(rows[0] - m, 0), min(rows[-1] + m + 1, H), max(cols[0] - m, 0), min(cols[-1] + m + 1, W)


def paint(W, H, groups, masks, light=(0.55, -0.6, 0.35), pal='summer', sun=None, seed=0,
          rim=1.0, rim_px=None, glow=1.0, edge_detail=1.0, overlap=1.0, texture=1.0, backlit=0.0,
          light_fn=None, sky=None, ambient=0.18, terminator=0.0, soft=1.0, stops=None, haze_col=None,
          march=0.55, march_len=0.3, bands=3, band_mix=0.7):
    """Paint the groups of ONE cloud into an RGBA plate (H, W, 4), straight alpha.

    light: direction TOWARD the light (x right, y down, z toward the viewer), or light_fn(xs, ys) ->
           (lx, ly, lz) normalised arrays for a per-pixel light (e.g. a low sun inside the frame).
    pal: palette name/dict. sun: (x, y) px of the sun (rim/glow strength grows near it).
    rim: silver-lining strength; rim_px: rim width (default 2.8 px at 1920 wide); glow: translucent
    edge glow near the sun; edge_detail: how much edge bumps bend the normal; overlap: strength of the
    internal overlap edges; texture: brush-stroke texture amount; backlit: 0..1 darkening of the body
    near the sun (contre-jour); sky: optional (H, W, 3) sky, the shadow side picks a little up;
    terminator: shifts the lit/shadow boundary (+ = more shadow); soft: terminator wobble;
    stops: 6 N.L positions for the ramp (deep, shadow, mid, lit, lit, hot); haze_col overrides pal haze;
    march: 0..1 weight of the 2D light-march self-shadow (lit caps shaped like the sun-facing edge);
    march_len: march distance as a fraction of the cloud's largest group S;
    bands: number of painted value steps (0 = continuous), band_mix: how strongly values snap to them.
    """
    P = palette(pal)
    if haze_col is not None:
        P['haze'] = _c(haze_col)
    out = np.zeros((H, W, 4), np.float32)
    uni = np.zeros((H, W), np.float32)
    for A in masks:
        np.maximum(uni, A, out=uni)
    bb = _bbox(uni, 0.02 * W + 8)
    if bb is None:
        return out
    Y0, Y1, X0, X1 = bb
    cw, ch = X1 - X0, Y1 - Y0
    sc = W / 1920.0
    rim_px = rim_px or 2.8 * max(sc, 0.5)
    xs, ys = C.grid(cw, ch)
    xs += X0
    ys += Y0
    if light_fn is None:
        L = np.asarray(light, np.float32)
        L = L / np.linalg.norm(L)
        Lx, Ly, Lz = float(L[0]), float(L[1]), float(L[2])
    else:
        Lx, Ly, Lz = light_fn(xs, ys)
    stops = np.array(stops if stops is not None else [-0.25, 0.04, 0.15, 0.27, 0.55, 0.95], np.float32)
    cols = [P['deep'], P['shadow'], P['mid'], P['lit'], P['lit'], P['hot']]
    wob = _lnoise(cw, ch, 5.0 * cw / W + 1, 3, seed + 5)
    crisp = C.smoothstep(-0.2, 0.3, _lnoise(cw, ch, 4.0 * cw / W + 1, 2, seed + 9))
    rgb = np.zeros((ch, cw, 3), np.float32)
    union = np.zeros((ch, cw), np.float32)
    skyc = None if sky is None else sky[Y0:Y1, X0:X1]
    first = True
    S_ref = max(float(g['S']) for g in groups)
    for gi, (g, Af) in enumerate(zip(groups, masks)):
        Ac = Af[Y0:Y1, X0:X1]
        gS = float(g['S'])
        gb = _bbox(Ac, gS * max(0.12, g.get('occ_w', 0.015) * 4.5) + 6)
        if gb is None:
            continue
        y0, y1, x0, x1 = gb
        sl = (slice(y0, y1), slice(x0, x1))
        A = np.ascontiguousarray(Ac[sl])
        h, w = A.shape
        gL = (Lx, Ly, Lz) if light_fn is None else (Lx[sl], Ly[sl], Lz[sl])
        Mb = A > 0.5
        DT = _dt(Mb)
        gx, gy = _pillow_grad(A, g.get('pillow', 0.75))
        # small-scale edge normal (edge bumps), fading into the interior
        es = max(gS * 0.012, 1.0)
        q = 2 if es > 4 else 1
        e = cv2.resize(A, (max(w // q, 2), max(h // q, 2)), interpolation=cv2.INTER_AREA) if q > 1 else A
        e = C.blur(e, es / q)
        ex, ey = _sobel(e)
        if q > 1:
            ex = cv2.resize(ex, (w, h), interpolation=cv2.INTER_CUBIC) / q
            ey = cv2.resize(ey, (w, h), interpolation=cv2.INTER_CUBIC) / q
        en = np.sqrt(ex * ex + ey * ey) + 1e-6
        # edge normal magnitude ~1 right at an edge, ->0 in flat interior (no noise amplification)
        ek = es * 2.5 * np.exp(-DT / max(gS * 0.03, 2.0)) * (edge_detail * g.get('edge', 1.0))
        nx = -gx * 1.1 - ex * ek * 0.8
        ny = -gy * 1.1 - ey * ek * 0.8
        nz = 0.9
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        ndl = (nx * gL[0] + ny * gL[1] + nz * gL[2]) / nn
        if march:
            # 2D light march: optical depth toward the light through this mass -> lit caps whose
            # terminator echoes the sun-facing cauliflower silhouette (painted self-shadow shapes)
            lx2, ly2 = (float(np.mean(gL[0])), float(np.mean(gL[1])))
            mdist = g.get('march_px') or S_ref * march_len * g.get('march', 1.0)
            lm = _light_march(A, lx2, ly2, mdist, crisp[sl])
            ndl = ndl * (1 - march) + (lm * 1.1 - 0.35) * march
        ndl = ndl + wob[sl] * (0.06 * soft) - terminator - g.get('shade', 0.0) * 0.35
        if bands:
            # painted value steps: soft-edged posterisation (edge width varies along the terminator)
            lo, hi = -0.35, 1.05
            u = (ndl - lo) / (hi - lo) * bands
            fl = np.floor(u)
            fr = u - fl
            bw = 0.22 + 0.26 * (1 - crisp[sl])
            q = (fl + C.smoothstep(0.5 - bw, 0.5 + bw, fr)) / bands * (hi - lo) + lo
            bm = band_mix * C.smoothstep(0.62, 0.4, ndl)
            ndl = ndl * (1 - bm) + q * bm
        col = _ramp(ndl, stops, cols)
        if 'lit2' in P and sun is not None:
            # lit colour drifts from gold (near the sun) to pink (away from it)
            sdd = np.sqrt((sun[0] - xs[sl]) ** 2 + (sun[1] - ys[sl]) ** 2) / W
            far_k = C.smoothstep(0.08, 0.5, sdd) * C.smoothstep(stops[2], stops[4], ndl)
            col += (P['lit2'] - P['lit']) * far_k[..., None]
        # lower part of each mass cooler & darker (underside)
        rows = np.where(Mb.any(1))[0]
        yy = np.arange(h, dtype=np.float32)[:, None]
        yt, yb = (rows[0], rows[-1]) if len(rows) else (0, h)
        rel = (yy - yt) / max(yb - yt, 1)
        bd = C.smoothstep(0.5, 1.05, rel) * g.get('base_dark', 0.5)
        kk = bd * (0.25 + C.smoothstep(0.55, 0.1, ndl))
        col += (P['base'] - col) * np.clip(kk, 0, 1)[..., None]
        if skyc is not None and ambient:
            col += (skyc[sl] - col) * (ambient * 0.5 * C.smoothstep(0.5, -0.2, ndl))[..., None]
        if g.get('haze', 0.0):
            hz = g['haze'] * (0.6 + 0.4 * C.smoothstep(0.0, 1.0, rel))
            col += (P['haze'] - col) * np.clip(hz, 0, 1)[..., None]
        R = rgb[sl]
        U = union[sl]
        if first:
            R[...] = col
            U[...] = A
            first = False
            continue
        # ---- composite over the masses behind: overlap edge that fades in/out along its length
        back = U > 0.5
        a_soft = fast_blur(A, max(gS * 0.012, 1.2))
        a_vsoft = fast_blur(A, max(gS * 0.1, 2.0))
        # upper outlines of a front mass stay (mostly) crisp; its lower outline dissolves into the mass behind
        upn = C.smoothstep(-0.3, 0.35, -ey / en) if not g.get('solid') else np.ones_like(A)
        k = np.clip(0.4 + 0.6 * crisp[sl] * g.get('crisp', 1.0), 0, 1) * upn
        a_soft = a_soft * upn + np.minimum(a_vsoft, A) * (1 - upn)
        a_eff = np.clip(np.where(back, A * k + a_soft * (1 - k), A), 0, 1)
        if overlap:
            dout = _dt(~Mb)
            sh = np.exp(-dout / max(gS * g.get('occ_w', 0.015), 2.0)) * (1 - A) * back *                 (overlap * g.get('occ_amt', 0.12)) * (0.3 + 0.7 * crisp[sl])
            R += (P['deep'] - R) * sh[..., None]
            fac = np.clip((-ex / en) * gL[0] + (-ey / en) * gL[1] + 0.1, 0, 1) * C.smoothstep(0.1, 0.4, ndl)
            band = C.smoothstep(rim_px * 1.8, rim_px * 0.6, DT) * Mb * back * fac * crisp[sl] * overlap
            col += (P['hot'] * 1.03 - col) * (band * 0.75)[..., None]
        R += (col - R) * a_eff[..., None]
        U[...] = 1 - (1 - U) * (1 - A)
    if first:
        return out
    # ---- silhouette rim light / translucency on the union
    Mu = union > 0.5
    DTu = _dt(Mu)
    eu = C.blur(union, 1.2 * max(sc, 0.5))
    ux, uy = _sobel(eu)
    un = np.sqrt(ux * ux + uy * uy) + 1e-6
    onx, ony = -ux / un, -uy / un  # outward normal
    if sun is not None:
        sdx, sdy = sun[0] - xs, sun[1] - ys
        sd = np.sqrt(sdx * sdx + sdy * sdy) + 1e-3
        facing = np.clip(onx * sdx / sd + ony * sdy / sd, 0, 1)
        prox = np.exp(-sd / (0.16 * W))
        prox2 = np.exp(-sd / (0.05 * W))
    else:
        ll = np.sqrt(Lx * Lx + Ly * Ly) + 1e-3
        facing = np.clip((onx * Lx + ony * Ly) / ll, 0, 1)
        prox = np.zeros_like(xs)
        prox2 = prox
    # shadow-side edges read slightly lighter (thin, sky-lit, translucent) - painted soft edge light
    el = np.exp(-DTu / max(2.0 * sc, 1.0)) * (1 - facing) * 0.25
    rgb += (P['mid'] - rgb) * np.clip(el, 0, 1)[..., None] * (rgb.mean(-1, keepdims=True) < P['mid'].mean())
    if backlit and sun is not None:
        dk = prox * backlit * C.smoothstep(0, 0.06 * W, DTu)
        rgb += (P['shadow'] - rgb) * (dk * 0.5)[..., None]
    wpx = rim_px * (1 + 2.5 * prox2)
    rband = C.smoothstep(wpx + 1.0, wpx - 0.8, DTu) * Mu
    rim_amt = rband * (facing ** 0.8) * (0.55 + 1.2 * prox) * rim
    rgb += (P['rim'] * 1.15 - rgb) * np.clip(rim_amt, 0, 1)[..., None]
    rgb += P['rim'] * (np.clip(rim_amt - 1, 0, None) * 0.3)[..., None]
    if glow and sun is not None:
        gl = np.exp(-DTu / (0.006 * W)) * prox2 * glow * (0.2 + 0.8 * facing)
        rgb += P['glow'] * (gl * 0.55)[..., None]
    # ---- brush texture: low-contrast anisotropic strokes (two directions)
    if texture:
        n1 = stroke_noise(cw, ch, -12, 60 * sc + 6, 6 * sc + 1, seed + 1)
        n2 = stroke_noise(cw, ch, 28, 40 * sc + 4, 5 * sc + 1, seed + 2)
        n = (n1 * 0.65 + n2 * 0.35) * (0.009 * texture)
        rgb *= 1 + n[..., None] * np.array([1.0, 1.0, 1.1], np.float32)
    out[Y0:Y1, X0:X1, :3] = rgb
    out[Y0:Y1, X0:X1, 3] = np.clip(union, 0, 1)
    return out


def render_cloud(W, H, groups, light=(0.55, -0.6, 0.35), pal='summer', sun=None, seed=0, ss=2, **kw):
    """One cloud (list of groups, back -> front) -> RGBA plate (H, W, 4)."""
    masks = build_masks(W, H, groups, seed=seed, ss=ss)
    return paint(W, H, groups, masks, light=light, pal=pal, sun=sun, seed=seed, **kw)


def over_plate(dst, src):
    """Straight-alpha RGBA over RGBA (in place on dst, returns dst)."""
    a = src[..., 3:4]
    da = dst[..., 3:4]
    oa = a + da * (1 - a)
    dst[..., :3] = (src[..., :3] * a + dst[..., :3] * da * (1 - a)) / np.maximum(oa, 1e-5)
    dst[..., 3:4] = oa
    return dst


def render_clouds(W, H, clouds, light=(0.55, -0.6, 0.35), pal='summer', sun=None, seed=0, **kw):
    """Several independent clouds (list of group lists, back -> front) -> one RGBA plate."""
    acc = np.zeros((H, W, 4), np.float32)
    for i, groups in enumerate(clouds):
        p = render_cloud(W, H, groups, light=light, pal=pal, sun=sun, seed=seed + 31 * i, **kw)
        over_plate(acc, p)
    return acc


# ============================================================================ shape designers

def _up_prob(lo=0.0, side=0.5, top=0.95):
    def f(u, x, y):
        return float(np.interp(u, [-1.0, -0.5, -0.1, 0.35, 1.0], [0.0, lo, side, top * 0.95, top]))
    return f


def cumulonimbus_groups(cx, base_y, width, height, seed=0, wind=-1, anvil=1.0, stack=6, lean=0.0):
    """Group specs for a towering cumulonimbus, back -> front: anvil, two cauliflower heads, then a
    stack of broad overlapping masses from the top of the tower down to the base (each lower mass sits
    in front of the one above, so its sun-lit upper outline crosses the tower as a crisp internal edge
    while its lower outline dissolves), and two low flank masses.
    cx, base_y: base centre (px); width: width of the lower cloud mass (px); height: base -> anvil top
    (px); wind: -1 = anvil streams left, +1 = right; anvil: anvil length (0 = no anvil); stack: number
    of stacked masses; lean: horizontal drift of the tower top (fraction of width)."""
    rng = np.random.default_rng(seed)
    w, h = width, height
    S = w * 0.8
    Y = lambda f: base_y - h * f
    X = lambda f, dx=0.0: cx + w * (dx + lean * f)
    ay = Y(0.9)
    tw = wind
    up = _up_prob(0.02, 0.6, 0.95)

    def anvil_prob(u, x, y):
        near = math.exp(-((x - X(0.9)) / (w * 0.3)) ** 2)
        return float(np.interp(u, [-1, 0.0, 0.5, 1.0], [0.0, 0.0, 0.2, 0.5])) * near

    G = []
    if anvil > 0:
        ax = X(0.9)
        G.append(dict(name='anvil', S=S * 0.6, levels=3, r0=0.1, prob=anvil_prob, warp=0.03, pillow=0.9,
                      ell=[(ax + tw * w * 0.1, ay + h * 0.02, w * 0.4, h * 0.055, 0),
                           (ax + tw * w * 0.55, ay + h * 0.008, w * 0.5 * anvil, h * 0.04, -1.5 * tw),
                           (ax + tw * w * 1.0, ay - h * 0.004, w * 0.42 * anvil, h * 0.024, -2 * tw),
                           (ax - tw * w * 0.26, ay + h * 0.03, w * 0.2, h * 0.022, 4 * tw),
                           (ax - tw * w * 0.42, ay + h * 0.036, w * 0.12, h * 0.01, 5 * tw)],
                      fib=dict(wind=tw, x0=ax + tw * w * 0.7, len=0.5, amt=0.7), edge=0.4, base_dark=0.45,
                      haze=0.04, crisp=0.8, spacing=1.5, march=0.7))
    # cauliflower heads (behind the stack top)
    for k, (dx, f, rw, rh) in enumerate([(-0.07, 0.86, 0.13, 0.07), (0.07, 0.83, 0.12, 0.065)]):
        G.append(dict(name='head%d' % k, S=S * 0.4, levels=4, r0=0.2, prob=up, haze=0.02, base_dark=0.1,
                      ell=[(X(f, dx), Y(f), w * rw, h * rh, rng.uniform(-8, 8)),
                           (X(f, dx + rng.uniform(-0.05, 0.05)), Y(f - 0.05), w * rw * 1.2, h * rh, 0)]))
    # stacked masses, top (back) -> bottom (front)
    side = rng.choice([-1, 1])
    for k in range(stack):
        t = k / max(stack - 1, 1)
        f = 0.76 - 0.62 * t                      # height of the mass top-ish
        mw = w * (0.18 + 0.2 * t) * rng.uniform(0.75, 1.2)
        mh = h * rng.uniform(0.08, 0.15)
        dx = side * rng.uniform(0.02, 0.14) * (1 - 0.3 * t)
        side = -side
        G.append(dict(name='m%d' % k, S=S * (0.45 + 0.3 * t), levels=4, r0=0.18, prob=up, haze=0.02 + 0.03 * t,
                      base_dark=0.2, crisp=rng.uniform(0.6, 1.0),
                      ell=[(X(f, dx), Y(f - 0.06), mw, mh, rng.uniform(-6, 6)),
                           (X(f, dx + rng.uniform(-0.12, 0.12)), Y(f - 0.1), mw * 0.8, mh * 1.2, 0),
                           (X(f, dx * 0.3), Y(f - 0.16), mw * 1.05, mh * 1.3, 0)]))
    G.append(dict(name='flankL', S=S * 0.55, levels=4, r0=0.18, prob=up,
                  ell=[(cx - w * 0.32, Y(0.11), w * 0.22, h * 0.11, 4), (cx - w * 0.24, Y(0.2), w * 0.12, h * 0.08, 0)],
                  base=base_y, base_dark=0.45, haze=0.05))
    G.append(dict(name='front', S=S * 0.5, levels=4, r0=0.18, prob=up,
                  ell=[(cx + w * 0.3, Y(0.06), w * 0.22, h * 0.07, 0), (cx + w * 0.22, Y(0.12), w * 0.11, h * 0.06, 0),
                       (cx + w * 0.42, Y(0.1), w * 0.08, h * 0.05, 0)],
                  base=base_y + h * 0.01, base_dark=0.5, haze=0.02))
    return G


def towering_cumulus_groups(cx, base_y, width, height, seed=0, haze=0.0):
    """Group specs for a towering cumulus (cumulus congestus) rising out of a cloud layer: two
    cauliflower heads BEHIND the top of the trunk (so the trunk's lit top edge overlaps them), the
    trunk, and a low shoulder mass in front. cx, base_y: base centre (px); width/height in px."""
    rng = np.random.default_rng(seed)
    w, h = width, height
    Y = lambda f: base_y - h * f
    S = w * 0.8
    up = _up_prob(0.02, 0.6, 0.95)
    s1 = rng.choice([-1, 1])
    return [
        dict(name='headB', S=S * 0.5, levels=4, r0=0.2, prob=up, haze=haze, base_dark=0.1,
             ell=[(cx + s1 * w * 0.12, Y(0.86), w * 0.15, h * 0.1, 0), (cx + s1 * w * 0.2, Y(0.78), w * 0.12, h * 0.08, 0)]),
        dict(name='headA', S=S * 0.55, levels=4, r0=0.2, prob=up, haze=haze, base_dark=0.1,
             ell=[(cx - s1 * w * 0.08, Y(0.93), w * 0.14, h * 0.08, 0), (cx - s1 * w * 0.02, Y(0.86), w * 0.16, h * 0.09, 0)]),
        dict(name='trunk', S=S, levels=4, r0=0.16, prob=up, haze=haze, base_dark=0.2, warp=0.04,
             ell=[(cx, Y(0.3), w * 0.32, h * 0.3, 0), (cx + s1 * w * 0.03, Y(0.55), w * 0.24, h * 0.22, 3),
                  (cx - s1 * w * 0.02, Y(0.72), w * 0.18, h * 0.12, 0)]),
        dict(name='front', S=S * 0.5, levels=4, r0=0.2, prob=up, haze=haze, base_dark=0.2,
             ell=[(cx - s1 * w * 0.25, Y(0.15), w * 0.2, h * 0.12, 0), (cx - s1 * w * 0.18, Y(0.25), w * 0.1, h * 0.07, 0)]),
    ]


def cumulus_groups(cx, base_y, width, height, seed=0, towers=3, haze=0.0, base=True, r0=0.17):
    """Group specs for a fair-weather / towering cumulus: `towers` domed masses on a common base."""
    rng = np.random.default_rng(seed)
    groups = []
    S = width * 0.6
    xs = np.sort(rng.uniform(-0.3, 0.3, towers)) * width
    order = np.argsort(-np.abs(xs) + rng.uniform(0, 0.1, towers) * width)  # tallest (central) at the back
    for k in order[::-1][::-1]:
        x = cx + xs[k]
        hh = height * rng.uniform(0.6, 1.0) * (1 - abs(xs[k]) / width * 1.2)
        ww = width * rng.uniform(0.25, 0.36)
        groups.append(dict(S=S * rng.uniform(0.7, 1.0), levels=5, r0=r0, pillow=0.55, prob=_up_prob(0.02, 0.5, 0.95),
                           ell=[(x, base_y - hh * 0.5, ww * 0.55, hh * 0.5, rng.uniform(-8, 8)),
                                (x + ww * rng.uniform(-0.3, 0.3), base_y - hh * 0.22, ww * 0.75, hh * 0.25, 0)],
                           base=base_y if base else None, haze=haze, base_dark=0.5))
    return groups


# ============================================================================ sea of clouds

def sun_light_fn(sun, lz=0.3, min_len=1.0):
    """Per-pixel light direction toward a sun at screen position `sun` (for low suns in frame):
    returns f(xs, ys) -> (lx, ly, lz) normalised."""
    def f(xs, ys):
        dx, dy = sun[0] - xs, sun[1] - ys
        d = np.sqrt(dx * dx + dy * dy) + min_len
        k = math.sqrt(1 - lz * lz)
        return (dx / d * k).astype(np.float32), (dy / d * k).astype(np.float32), np.full_like(xs, lz)
    return f


def sea_row_groups(W, H, horizon, z, seed, K, amp, feat, x_range=None, haze=0.0, bottom=None, flat=1.0):
    """One row of cloud-sea billows at depth z: rounded mounds on a common body that extends to the
    bottom of the plate. Screen base line y = horizon + K / z; mound height ~ amp / z; mound width ~ feat / z."""
    rng = np.random.default_rng(seed)
    yb = horizon + K / z
    a, f = amp / z * flat, feat / z
    x0, x1 = x_range or (-0.05 * W, 1.05 * W)
    ell = []
    x = x0 + rng.uniform(-0.5, 0) * f
    while x < x1 + f:
        wd = f * rng.uniform(0.35, 1.5)
        ht = a * rng.uniform(0.2, 1.0) * (0.6 + 0.4 * rng.random())
        if rng.random() < 0.12:     # an occasional big billow rising above the rest
            wd *= 1.4
            ht *= 1.8
        ht = min(ht, wd * 0.55)
        ell.append((x, yb, wd * 0.62, ht, rng.uniform(-6, 6)))
        if rng.random() < 0.45:     # secondary dome on the shoulder
            ell.append((x + wd * rng.uniform(-0.3, 0.3), yb - ht * 0.45, wd * 0.3, ht * 0.75, 0))
        x += wd * rng.uniform(0.55, 0.95)
    S = max(f * 1.1, 6.0)

    def prob(u, xx, yy):
        return float(np.interp(u, [-1, 0.1, 0.45, 1.0], [0.0, 0.0, 0.7, 0.95]))

    lev = int(np.clip(np.log2(max(S * 0.18, 1) / 1.6) + 1, 0, 4))
    bot = H + 4 if bottom is None else min(bottom, H + 4)
    return dict(S=S, ell=ell, rects=[(x0 - f, yb, x1 + f, bot)], march_px=max(a * 0.7, 3.0), levels=lev, r0=0.18, prob=prob,
                warp=0.03, pillow=0.6, haze=haze, base_dark=0.0, edge=0.8, crisp=1.0,
                occ_w=0.18, occ_amt=0.55, z=z, spacing=1.5, solid=True)


def cloud_sea(W, H, horizon, sun, seed=0, pal='sunrise', n_rows=16, z_near=1.0, z_far=16.0, layer_splits=(3.5, 1.8),
              amp=None, feat=None, towers=(), sky=None, lz=0.28, haze_col=None, rim=1.2, **paint_kw):
    """Sea of clouds seen from above, flattening toward the horizon, lit by a low sun.

    W, H: plate size; horizon: horizon y (px); sun: (x, y) px of the sun (usually just above the horizon);
    n_rows: billow rows between z_near (bottom of the plate) and z_far (horizon); layer_splits: depths
    at which rows are split into separate parallax layers (far -> near); amp / feat: mound height / width
    at z = 1 (px; defaults relative to H / W); towers: list of dict(x=px, z=depth, height=px, width=px,
    seed=int) towering cumulus rising out of the sea at that depth (heights in px at their depth);
    sky: optional sky image (ambient); pal: palette.
    returns: list of (rgba_plate, z_mean) from far to near. Each plate is (H, W, 4)."""
    K = (H - horizon) * z_near
    amp = amp if amp is not None else 0.2 * (H - horizon)
    feat = feat if feat is not None else 0.2 * W
    zs = z_far * (z_near / z_far) ** (np.arange(n_rows) / max(n_rows - 1, 1))   # far -> near
    hz = C.smoothstep(z_far * 0.12, z_far * 0.9, zs) ** 0.7
    groups = []
    for i, z in enumerate(zs):
        bottom = None if i == len(zs) - 1 else horizon + K / zs[i + 1] + 0.6 * amp / zs[i + 1] + 0.06 * feat / zs[i + 1]
        flat = 1.0 - 0.7 * float(C.smoothstep(z_far * 0.2, z_far, z))
        groups.append(sea_row_groups(W, H, horizon, float(z), seed + 13 * i, K, amp, feat, haze=0.92 * float(hz[i]),
                                     bottom=bottom, flat=flat))
    for t in towers:
        z = t['z']
        yb = horizon + K / z
        cg = cumulonimbus_groups(t['x'], yb + 0.12 * t['height'], t['width'], t['height'], seed=t.get('seed', 5),
                                 anvil=0, stack=t.get('stack', 5), lean=t.get('lean', 0.0))
        cg = [g for g in cg if g['name'] not in ('flankL', 'front')]
        thz = 0.6 * float(C.smoothstep(z_far * 0.15, z_far, z) ** 0.8)
        for g in cg:
            g['z'] = z + 1e-3
            g['base_dark'] = 0.2
            g['haze'] = thz
            g['base'] = None
            g['solid'] = True
        groups += cg
    groups.sort(key=lambda g: -g['z'])
    splits = sorted(layer_splits, reverse=True)
    layers = []
    bounds = [np.inf] + list(splits) + [-np.inf]
    lf = sun_light_fn(sun, lz)
    for li in range(len(bounds) - 1):
        gs = [g for g in groups if bounds[li] >= g['z'] > bounds[li + 1]]
        if not gs:
            continue
        masks = build_masks(W, H, gs, seed=seed + 101 * li, ss=1 if min(g['z'] for g in gs) > 4.0 else 2)
        for g in gs:
            g.setdefault('march_px', max(g['S'] * 0.9, 4))
        rgba = paint(W, H, gs, masks, light_fn=lf, pal=pal, sun=sun, seed=seed + 7 * li, sky=sky,
                     haze_col=haze_col, rim=rim, **paint_kw)
        layers.append((rgba, float(np.mean([g['z'] for g in gs]))))
    return layers
