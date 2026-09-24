"""Painted cumulonimbus for s03_city_dusk (round 3 rewrite: flat 2D painted lobes, not shaded spheres).

How a background painter builds a sunset cumulonimbus, reproduced literally:
  * A macro silhouette (tower, shoulders, a flat sheared anvil) decides the BIG light: the sunward
    (left, upper) part is lit, the rest -- and everything below the earth's-shadow line -- is one flat
    cool lavender shadow plane.
  * The silhouette and the lit zone are built from many small cauliflower lobes (small on the lit top
    edges, larger lower down). Lobes are FLAT discs painted in painter's order; each gets a posterized
    tone: lit (peach -> warm white toward the sun) with a crescent of shadow on its far side bounded by
    an ARC (the crisp painted terminator), a thin saturated coral band on that arc. Lobes deep in the
    shadow zone are the same flat lavender as the plane, so they merge into it.
  * A 2-3 px gold/white silver lining on every silhouette edge facing the (set) sun, a soft halo outside
    it, torn wisps at the base, cirrus fibres streaming off the anvil.
Everything is supersampled and box-filtered -> clean, crisp edges without jaggies.
"""
import math
import numpy as np
import cv2
from numba import njit

from lib import core as C


@njit(cache=True)
def _paint_ids(L, Hs, Ws, ss, x0, y0):
    """painter's order: each lobe (cx, cy, r, ay) overwrites the id buffer inside its ellipse"""
    ids = np.full((Hs, Ws), -1, np.int32)
    for i in range(L.shape[0]):
        cx = (L[i, 0] - x0) * ss
        cy = (L[i, 1] - y0) * ss
        r = L[i, 2] * ss
        ry = r * L[i, 3]
        xa = max(int(cx - r) - 1, 0)
        xb = min(int(cx + r) + 2, Ws)
        ya = max(int(cy - ry) - 1, 0)
        yb = min(int(cy + ry) + 2, Hs)
        for y in range(ya, yb):
            dy = (y + 0.5 - cy) / ry
            for x in range(xa, xb):
                dx = (x + 0.5 - cx) / r
                if dx * dx + dy * dy < 1.0:
                    ids[y, x] = i
    return ids


@njit(cache=True)
def _paint_ids_w(L, Hs, Ws, ss, x0, y0, wx, wy):
    """like _paint_ids, but the sample position is displaced by (wx, wy) (frame px) -> organic outlines"""
    ids = np.full((Hs, Ws), -1, np.int32)
    pad = 0.01 * (Ws / ss)
    for i in range(L.shape[0]):
        cx = (L[i, 0] - x0) * ss
        cy = (L[i, 1] - y0) * ss
        r = L[i, 2] * ss
        ry = r * L[i, 3]
        m = 0.006 * 1920 * ss
        xa = max(int(cx - r - m) - 1, 0)
        xb = min(int(cx + r + m) + 2, Ws)
        ya = max(int(cy - ry - m) - 1, 0)
        yb = min(int(cy + ry + m) + 2, Hs)
        for y in range(ya, yb):
            for x in range(xa, xb):
                dx = (x + 0.5 + wx[y, x] * ss - cx) / r
                dy = (y + 0.5 + wy[y, x] * ss - cy) / ry
                if dx * dx + dy * dy < 1.0:
                    ids[y, x] = i
    return ids


def _fibres(w, h, n, rng, xspan, yspan, lmin, lmax, wmin, wmax, amp=(0.4, 1.0)):
    """painterly horizontal brush fibres: n thin tapered strokes, returns (h, w) float32 0..1"""
    out = np.zeros((h, w), np.float32)
    for _ in range(n):
        L = rng.uniform(lmin, lmax)
        x = rng.uniform(*xspan)
        y = rng.uniform(*yspan)
        th = rng.uniform(wmin, wmax)
        a = rng.uniform(*amp)
        k = 24
        pts = []
        for j in range(k + 1):
            u = j / k
            pts.append((x + u * L, y + math.sin(u * 3.0 + x * 0.01) * th * 0.6 + u * L * rng.uniform(-0.004, 0.004)))
        for j in range(k):
            u = (j + 0.5) / k
            taper = math.sin(math.pi * u) ** 0.7
            wd = max(int(round(th * taper)), 1)
            cv2.line(out, (int(pts[j][0]), int(pts[j][1])), (int(pts[j + 1][0]), int(pts[j + 1][1])),
                     float(a * (0.35 + 0.65 * taper)), wd, cv2.LINE_AA)
    return out


def paint(pw, ph, W, H, ox, sun, seed=7, ss=3):
    """Returns straight-alpha RGBA plate (ph, pw, 4). Frame coords (fractions of W/H) are offset by ox px.
    sun: (x, y) of the set sun in plate px."""
    rng = np.random.default_rng(seed)
    X = lambda fx: fx * W + ox
    Y = lambda fy: fy * H
    # ------------------------------------------------------------------ macro silhouette (low res)
    q = 4                                        # macro mask resolution divider
    mw, mh = pw // q + 1, ph // q + 1
    macro = np.zeros((mh, mw), np.uint8)

    def ell(cx, cy, rx, ry, m=macro):
        cv2.ellipse(m, (int(X(cx) / q), int(Y(cy) / q)), (max(int(rx * W / q), 1), max(int(ry * H / q), 1)),
                    0, 0, 360, 1, -1)

    base_y = 0.52
    # main tower: stacked masses, slightly leaning left toward the top
    for k, (fy, rx) in enumerate([(0.45, 0.078), (0.385, 0.07), (0.32, 0.062), (0.26, 0.056), (0.205, 0.05),
                                  (0.16, 0.044), (0.125, 0.038)]):
        ell(0.80 - 0.009 * k, fy, rx, 0.06)
    # sunward low shoulder / turret
    ell(0.70, 0.45, 0.045, 0.06)
    ell(0.69, 0.40, 0.024, 0.045)
    ell(0.735, 0.43, 0.035, 0.055)
    # right secondary tower + far shoulder
    ell(0.925, 0.40, 0.055, 0.1)
    ell(0.92, 0.31, 0.036, 0.06)
    ell(1.02, 0.42, 0.06, 0.1)
    ell(1.04, 0.35, 0.03, 0.05)
    # flat base
    macro[int(Y(base_y) / q):] = 0
    macro_f = macro.astype(np.float32)
    mdist = cv2.distanceTransform(macro, cv2.DIST_L2, 5).astype(np.float32) * q
    # ------------------------------------------------------------------ lobes
    lobes = []           # cx, cy, r, ay, kind(0 body, 1 interior, 2 edge, 3 bead)
    # body fill: large lobes on a jittered grid covering the macro shape
    rb = 0.034 * W
    for yy in np.arange(Y(0.1), Y(base_y) + rb, rb * 0.9):
        for xx in np.arange(X(0.6), X(1.12), rb * 1.1):
            x_ = xx + rng.uniform(-0.3, 0.3) * rb
            y_ = yy + rng.uniform(-0.3, 0.3) * rb
            mx_, my_ = int(x_ / q), int(y_ / q)
            if 0 <= mx_ < mw and 0 <= my_ < mh and macro[my_, mx_]:
                rr = min(rb * rng.uniform(0.9, 1.25), mdist[my_, mx_] * 0.95)
                if rr > 0.006 * W:
                    lobes.append([x_, y_, rr, rng.uniform(0.85, 1.0), 0])
    # contour of the macro silhouette -> edge lobes (smaller up high, larger low)
    cs, _ = cv2.findContours(macro, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_pts = []
    for c in cs:
        c = c[:, 0, :].astype(np.float32) * q
        edge_pts.append(c)
    edge_pts = np.concatenate(edge_pts, 0) if edge_pts else np.zeros((0, 2), np.float32)

    def size_at(y):
        u = np.clip((y - Y(0.08)) / (Y(base_y) - Y(0.08)), 0, 1)
        return (0.013 + 0.02 * u ** 1.3) * W

    def place_on(pts, count_scale, kind, rmul, inset):
        placed = []
        if len(pts) == 0:
            return placed
        acc = 0.0
        prev = pts[0]
        for p in pts[1:]:
            acc += float(np.hypot(*(p - prev)))
            prev = p
            r = size_at(p[1]) * rmul
            if acc >= r * count_scale and p[1] < Y(base_y) - 0.012 * H:
                acc = 0.0
                rr = r * rng.uniform(0.75, 1.25)
                # push toward the inside by `inset` radii along the macro normal
                mx_, my_ = int(p[0] / q), int(p[1] / q)
                gx = macro_f[my_, min(mx_ + 1, mw - 1)] - macro_f[my_, max(mx_ - 1, 0)]
                gy = macro_f[min(my_ + 1, mh - 1), mx_] - macro_f[max(my_ - 1, 0), mx_]
                gl = math.hypot(gx, gy) + 1e-6
                placed.append([p[0] + gx / gl * rr * inset, p[1] + gy / gl * rr * inset, rr,
                               rng.uniform(0.8, 0.97), kind])
        return placed

    # sort contour by arclength already (findContours order)
    edge1 = place_on(edge_pts, 1.1, 2, 1.0, 0.3)
    m2 = macro.copy()
    for l in edge1:
        cv2.ellipse(m2, (int(l[0] / q), int(l[1] / q)), (max(int(l[2] / q), 1), max(int(l[2] * l[3] / q), 1)), 0, 0, 360, 1, -1)
    cs2, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts2 = np.concatenate([c[:, 0, :].astype(np.float32) * q for c in cs2], 0)
    edge2 = [l for l in place_on(pts2, 1.6, 2, 0.55, 0.15) if l[1] < Y(0.40) and rng.random() < 0.6 and (l[0] < X(0.79) - (Y(0.3) - l[1]) * 0.2 or l[1] < Y(0.15))]
    edge1 = edge1 + edge2
    # interior lit-zone lobes: rows of cauliflower bulges in the upper-left of the tower (tiers)
    inter = []
    # tiers: rows of overlapping bulges (each tier's lit tops sit against the shaded underside of the tier
    # above) -- the stacked cauliflower read of a towering cumulus
    ty = Y(0.13)
    while ty < Y(0.46):
        r = size_at(ty) * 1.35
        row = []
        xx = X(0.6)
        while xx < X(1.1):
            mx_, my_ = int(xx / q), int(ty / q)
            if 0 <= mx_ < mw and 0 <= my_ < mh and mdist[my_, mx_] > r * 0.9:
                row.append([xx + rng.uniform(-0.15, 0.15) * r, ty + rng.uniform(-0.25, 0.25) * r,
                            r * rng.uniform(0.85, 1.2), rng.uniform(0.75, 0.9), 1])
            xx += r * rng.uniform(1.1, 1.5)
        inter += row
        ty += r * rng.uniform(1.1, 1.4)
    # painter's order: body, interior (top first -> lower tiers overlap the ones above), edge lobes, beads
    inter.sort(key=lambda l: l[1])
    edge1.sort(key=lambda l: l[1])
    lobes = lobes + inter + edge1
    # beads: tiny lobes on the sunward/upper silhouette of the edge lobes
    beads = []
    for l in edge1:
        if l[1] > Y(0.36):
            continue
        for _ in range(int(rng.integers(1, 4))):
            th = rng.uniform(math.pi * 0.75, math.pi * 1.75)   # upper-left hemisphere (y down)
            r = l[2] * rng.uniform(0.3, 0.5)
            d = l[2] - r * rng.uniform(0.2, 0.6)
            beads.append([l[0] + math.cos(th) * d, l[1] + math.sin(th) * d * l[3], r, rng.uniform(0.8, 0.95), 3])
    lobes = lobes + beads
    # ------------------------------------------------------------------ anvil (flat sheared plate)
    anv = np.zeros((mh, mw), np.uint8)
    ax0, ax1 = X(0.705), X(1.12)
    top0 = Y(0.03)
    AU = lambda x: min(max((x - X(0.72)) / (X(1.12) - X(0.72)), 0.0), 1.0)
    atop = lambda u: top0 - 0.018 * H * u                  # flat, slightly rising sheared top
    abot = lambda u: Y(0.15) - (0.15 - 0.045) * H * u ** 0.8  # thick over the tower, thin downwind
    poly = [(X(0.74), Y(0.13)), (X(0.735), Y(0.08)), (X(0.745), Y(0.05)), (X(0.765), top0 + 0.002 * H)]
    for x_ in np.linspace(X(0.78), ax1, 12):
        poly.append((x_, atop(AU(x_))))
    for x_ in np.linspace(ax1, X(0.75), 14):
        poly.append((x_, abot(AU(x_))))
    cv2.fillPoly(anv, [np.array([(int(p[0] / q), int(p[1] / q)) for p in poly], np.int32)], 1)
    # underside scallops (small hanging lobes along the base of the shelf)
    anv_lobes = []
    for xx in np.arange(X(0.75), X(1.02), 0.012 * W):
        u = AU(xx)
        anv_lobes.append([xx + rng.uniform(-0.3, 0.3) * 0.012 * W, abot(u) - rng.uniform(0.004, 0.012) * H,
                          (0.015 - 0.007 * u) * W * rng.uniform(0.8, 1.2), rng.uniform(0.45, 0.6), 4])
    # plate lobes for the anvil body (flat ellipses filling the polygon so it gets the id machinery)
    for xx in np.arange(X(0.745), X(1.13), 0.015 * W):
        u = AU(xx)
        yt, yb = atop(u), abot(u)
        ry = max((yb - yt) * 0.6, 0.004 * H)
        anv_lobes.append([xx, 0.5 * (yt + yb), 0.03 * W, ry / (0.03 * W), 5])
    anv_lobes.sort(key=lambda l: -l[4])        # plates (5) first, scallops (4) after
    lobes = anv_lobes + lobes                   # anvil lies behind the tower crown
    L = np.array(lobes, np.float64)
    # ------------------------------------------------------------------ rasterise ids in a bbox
    ext = L[:, 2]
    x0 = int(max(np.min(L[:, 0] - ext) - 0.03 * W, 0))
    x1 = int(min(np.max(L[:, 0] + ext) + 0.02 * W, pw))
    y0 = int(max(np.min(L[:, 1] - ext) - 0.03 * H, 0))
    y1 = int(min(Y(base_y) + 0.08 * H, ph))
    w, h = x1 - x0, y1 - y0
    Ws, Hs = w * ss, h * ss
    # domain warp: lumpy, organic lobe outlines instead of perfect circles
    wn1 = C.fbm(max(Ws // 6, 8), max(Hs // 6, 8), scale=w / (0.02 * W), octaves=3, seed=seed + 1)
    wn2 = C.fbm(max(Ws // 6, 8), max(Hs // 6, 8), scale=w / (0.02 * W), octaves=3, seed=seed + 2)
    upw = lambda a: cv2.resize(a, (Ws, Hs), interpolation=cv2.INTER_CUBIC)
    wx = ((upw(wn1) - 0.5) * 0.006 * W).astype(np.float32)
    wy = ((upw(wn2) - 0.5) * 0.006 * W).astype(np.float32)
    ids = _paint_ids_w(L, Hs, Ws, float(ss), float(x0), float(y0), wx, wy)
    # the anvil plate is clipped by the polygon (flat straight top edge)
    anv_hi = np.zeros((Hs, Ws), np.uint8)
    cv2.fillPoly(anv_hi, [np.array([((p[0] - x0) * ss, (p[1] - y0) * ss) for p in poly], np.int32)], 1)
    kind_px = np.where(ids >= 0, L[np.maximum(ids, 0), 4], -1)
    plate_px = (kind_px == 5)
    ids = np.where(plate_px & (anv_hi == 0), -1, ids)
    # base fill: the macro silhouette itself under all lobes (no holes between lobes)
    mup = cv2.resize(macro_f[int(y0 / q):int(y0 / q) + (h // q) + 2, int(x0 / q):int(x0 / q) + (w // q) + 2],
                     None, fx=q * ss, fy=q * ss, interpolation=cv2.INTER_LINEAR)[:Hs, :Ws]
    mup = np.pad(mup, ((0, max(Hs - mup.shape[0], 0)), (0, max(Ws - mup.shape[1], 0))))
    kpre = np.where(ids >= 0, L[np.maximum(ids, 0), 4], -1)
    basefill = (mup > 0.5) & ((ids < 0) | (kpre >= 4))
    ids = np.where(basefill, 0, ids)
    cov = (ids >= 0).astype(np.float32)
    ii = np.maximum(ids, 0)
    kind_px = np.where(ids >= 0, L[ii, 4], -1)
    kind_px = np.where(basefill, 0, kind_px)
    # ------------------------------------------------------------------ macro light field
    xs = (np.arange(Ws, dtype=np.float32)[None, :] + 0.5) / ss + x0
    ys = (np.arange(Hs, dtype=np.float32)[:, None] + 0.5) / ss + y0
    # form normal from the heavily blurred coverage
    small = cv2.resize(cov, (max(w // 4, 8), max(h // 4, 8)), interpolation=cv2.INTER_AREA)
    sb = cv2.GaussianBlur(small, (0, 0), 0.02 * W / 4)
    gx = cv2.Sobel(sb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sb, cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.resize(gx, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
    gy = cv2.resize(gy, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
    gl = np.sqrt(gx * gx + gy * gy) + 1e-6
    # outward normal = -gradient (coverage increases inward)
    onx, ony = -gx / gl, -gy / gl
    strength = np.clip(gl / (np.percentile(gl, 97) + 1e-6), 0, 1)
    sdir = np.array([-0.96, -0.1], np.float32)
    sdir /= np.linalg.norm(sdir)
    face = (onx * sdir[0] + ony * sdir[1]) * strength
    # earth's shadow: sunlit above a slightly tilted line; the anvil is high -> fully lit
    esh = C.smoothstep(Y(0.40) + (xs - X(0.8)) * 0.1, Y(0.32) + (xs - X(0.8)) * 0.1, ys)
    # right side of the tower (away from the sun) goes into shadow in a broad posterized sweep
    side = C.smoothstep(X(0.825), X(0.785), xs + (ys - Y(0.25)) * 0.2)
    mfield = np.clip(0.35 + 1.1 * face, 0, 1) * 0.55 + side * 0.6
    mfield = np.clip(mfield, 0, 1) * esh
    # per-lobe light factor sampled at the lobe centre
    lx_ = np.clip(((L[:, 0] - x0) * ss).astype(int), 0, Ws - 1)
    ly_ = np.clip(((L[:, 1] - y0) * ss).astype(int), 0, Hs - 1)
    m_l = mfield[ly_, lx_].astype(np.float32)
    kind = L[:, 4]
    # anvil: top lit, underside scallops lit on their left only
    m_l = np.where(kind == 5, 0.95, m_l)
    m_l = np.where(kind == 4, np.clip(m_l * 0.4 + 0.35, 0, 1), m_l)
    # quantize to a few painted values
    m_l = np.round(m_l * 5) / 5
    m = m_l[ii]
    # ------------------------------------------------------------------ per-pixel lobe shading
    # value = macro light + a small share of the lobe's own (sphere-like) shading, then POSTERIZED into
    # flat painted steps. Only near the macro terminator do lobes flip into shadow, so the terminator
    # follows lobe outlines (cauliflower), while inside the light / shadow planes lobes merge.
    cxp, cyp, rp, ayp = L[ii, 0].astype(np.float32), L[ii, 1].astype(np.float32), L[ii, 2].astype(np.float32),         L[ii, 3].astype(np.float32)
    qx = np.clip((xs + wx - cxp) / rp, -1, 1)
    qy = np.clip((ys + wy - cyp) / (rp * ayp), -1, 1)
    qz = np.sqrt(np.clip(1 - qx * qx - qy * qy, 0, 1))
    l3 = np.array([sdir[0] * 0.8, sdir[1] * 0.8 - 0.25, 0.45], np.float32)
    l3 /= np.linalg.norm(l3)
    loc = qx * l3[0] + qy * l3[1] + qz * l3[2]                      # -1..1
    loc = np.where(basefill, 0.35, loc)
    mpx = cv2.GaussianBlur(mfield, (0, 0), 0.004 * W * ss)          # smooth macro light
    val = mpx * 1.15 - 0.35 + 0.46 * (loc - 0.35) * C.smoothstep(0.2, 0.55, mpx)
    val = np.where(kind_px == 5, 0.9, val)
    aa = 0.012
    shadow = C.smoothstep(0.02 + aa, 0.02 - aa, val)                 # crisp terminator
    band = np.exp(-((val - 0.02) / 0.05) ** 2)
    din = cv2.distanceTransform((ids >= 0).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32) / ss
    near_edge = C.smoothstep(0.03 * W, 0.012 * W, din)
    t_hot = C.smoothstep(0.62 - aa, 0.62 + aa, val + 0.4 * (loc - 0.6) + 0.25 * near_edge - 0.1) * (kind_px <= 3) * (kind_px != 1) * (0.4 + 0.6 * near_edge)
    t_mid = C.smoothstep(0.2 - aa, 0.2 + aa, val)
    m = mpx
    # ------------------------------------------------------------------ colours
    hgt = np.clip((Y(base_y) - ys) / (Y(base_y) - Y(0.05)), 0, 1)
    prox = np.clip(1 - (xs - X(0.7)) / (0.16 * W), 0, 1)
    # flat shadow plane: one lavender, tiny shift up high (sky fill) and at the base (pink city glow)
    shd = C.lerp(np.array([0.36, 0.30, 0.62], np.float32), np.array([0.43, 0.37, 0.72], np.float32),
                 C.smoothstep(0.2, 0.9, hgt)[..., None])
    shd = C.lerp(shd, np.array([0.56, 0.36, 0.6], np.float32), (C.smoothstep(0.25, 0.0, hgt) * 0.55)[..., None])
    lit_lo = C.lerp(np.array([0.74, 0.32, 0.44], np.float32), np.array([0.82, 0.38, 0.36], np.float32),
                    prox[..., None])                                    # coral-pink -> orange-peach
    lit = C.lerp(np.array([0.86, 0.46, 0.46], np.float32), np.array([0.92, 0.54, 0.4], np.float32),
                 prox[..., None])                                       # peach -> gold-peach
    litw = np.array([0.98, 0.74, 0.6], np.float32)                     # warm white cap
    coral = np.array([0.9, 0.34, 0.46], np.float32)
    col = C.lerp(lit_lo, lit, t_mid[..., None])
    col = C.lerp(col, litw, (t_hot * 0.85)[..., None])
    # warm-white wash along the sunward silhouette (flat painted band, crisp inner edge)
    sunward = C.smoothstep(0.05, 0.35, face)
    wash = C.smoothstep(0.022 * W, 0.016 * W, din) * sunward * C.smoothstep(0.3, 0.6, mpx)
    col = C.lerp(col, C.lerp(lit, litw, 0.6), (wash * 0.75)[..., None])
    col = C.lerp(col, shd, shadow[..., None])
    col = C.lerp(col, coral, (band * 0.5 * (1 - shadow))[..., None])
    # faint sky-fill edges on shadowed lobes' upper side (lobe structure barely readable in the plane)
    upfill = C.smoothstep(0.6, 0.95, -qy) * shadow * C.smoothstep(0.1, 0.5, hgt)
    col = C.lerp(col, shd * 1.08, (upfill * 0.5)[..., None])
    shadow_px = shadow
    # anvil plates: gradient from lit gold-pink top to lavender underside, smooth (painted flat wash)
    apl = (kind_px == 5)
    if apl.any():
        uu_ = np.clip((xs - X(0.72)) / (X(1.12) - X(0.72)), 0, 1)
        yt = top0 - 0.018 * H * uu_
        yb_ = Y(0.15) - (0.15 - 0.045) * H * uu_ ** 0.8
        v = np.clip((ys - yt) / np.maximum(yb_ - yt, 1.0), 0, 1)
        acol = C.lerp(C.lerp(np.array([1.1, 0.66, 0.6], np.float32), np.array([1.0, 0.5, 0.62], np.float32),
                             C.smoothstep(0.3, 1.0, uu_)[..., None]),
                      np.array([0.5, 0.4, 0.74], np.float32), C.smoothstep(0.3, 0.55, v)[..., None])
        fa_ = _fibres(Ws, Hs, 160, rng, ((X(0.7) - x0) * ss, (X(1.12) - x0) * ss), ((Y(0.02) - y0) * ss,
                      (Y(0.15) - y0) * ss), 0.04 * W * ss, 0.15 * W * ss, 1.0 * ss, 4.0 * ss)
        fa_ = cv2.GaussianBlur(fa_, (0, 0), 0.7 * ss)
        acol = acol * (0.9 + 0.2 * np.clip(fa_, 0, 1))[..., None]
        # hot lit top edge of the anvil (highest part of the cloud, still in direct sun)
        acol = C.lerp(acol, np.array([1.35, 0.92, 0.72], np.float32), C.smoothstep(0.1, 0.0, v)[..., None] * 0.7)
        col = np.where(apl[..., None], acol, col)
    # ------------------------------------------------------------------ silver lining (2-3 px) + glow
    rdir = np.array([sun[0] - X(0.82), sun[1] - Y(0.3)], np.float32)
    rdir /= np.linalg.norm(rdir)
    # combine: left (toward sun) and bottom edges
    rim = np.zeros_like(cov)
    for (dx, dy, wgt) in ((-1.0, 0.0, 1.0), (rdir[0], rdir[1], 1.0), (0.0, 1.0, 0.6)):
        off = 2.6 * ss * H / 1080
        M = np.float32([[1, 0, -dx * off], [0, 1, -dy * off]])
        sh = cv2.warpAffine(cov, M, (Ws, Hs), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rim = np.maximum(rim, np.clip(cov - sh, 0, 1) * wgt)
    rim_w = (0.45 + 0.55 * prox) * (0.55 + 0.45 * esh) * (0.5 + 0.5 * C.smoothstep(0.0, 0.5, mpx + 0.3 * prox))
    rimc = np.array([1.5, 1.15, 0.75], np.float32)
    col = C.lerp(col, rimc, np.clip(rim * rim_w * 1.2, 0, 1)[..., None])
    # internal sunward lobe edges (a lobe edge whose sun-side neighbour is another, farther lobe)
    off = 2.0 * ss * H / 1080
    M = np.float32([[1, 0, off], [0, 1, 0]])
    nb = cv2.warpAffine(ids.astype(np.float32), M, (Ws, Hs), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=-1)
    inner = ((nb != ids) & (nb >= 0) & (ids >= 0) & (nb < ids)).astype(np.float32)
    inner = inner * C.smoothstep(0.35, 0.7, m) * (1 - shadow) * (kind_px < 4)
    col = C.lerp(col, litw, (inner * 0.3)[..., None])
    # translucent glow creeping in from the sunward silhouette
    glow = cv2.GaussianBlur(rim, (0, 0), 4.0 * ss * H / 1080) * rim_w
    col = col + np.array([0.7, 0.36, 0.22], np.float32) * np.clip(glow * 0.8, 0, 0.22)[..., None]
    # ------------------------------------------------------------------ alpha: torn base, fibrous anvil end
    alpha = cov.copy()
    fibA = _fibres(Ws, Hs, 260, rng, (0, Ws), ((Y(base_y) - 0.1 * H - y0) * ss, (Y(base_y) + 0.02 * H - y0) * ss),
                   0.04 * W * ss, 0.16 * W * ss, 1.5 * ss, 5 * ss)
    fibA = cv2.GaussianBlur(fibA, (0, 0), 0.6 * ss)
    torn = C.smoothstep(Y(base_y) - 0.07 * H, Y(base_y) - 0.005 * H, ys)
    alpha = alpha * (1 - torn * (1 - np.clip(fibA * 1.1, 0, 1) * (1 - torn * 0.6)))
    # anvil downwind: combed into fibres beyond x ~ 0.97
    fibB = _fibres(Ws, Hs, 220, rng, ((X(0.85) - x0) * ss, (X(1.12) - x0) * ss),
                   ((Y(0.02) - y0) * ss, (Y(0.1) - y0) * ss), 0.05 * W * ss, 0.2 * W * ss, 1.0 * ss, 3.5 * ss)
    fibB = cv2.GaussianBlur(fibB, (0, 0), 0.5 * ss)
    comb = C.smoothstep(X(0.95), X(1.1), xs) * (kind_px >= 4)
    alpha = alpha * (1 - comb * (1 - np.clip(fibB * 1.3, 0, 1)))
    # cirrus fibres streaming off the anvil top edge to the right (beyond the solid plate)
    stream = fibB * C.smoothstep(X(0.9), X(1.0), xs) * (1 - alpha) * 0.7
    scol = np.array([1.05, 0.62, 0.66], np.float32)
    # base wisps below the solid body
    wisp = fibA * C.smoothstep(Y(base_y) - 0.03 * H, Y(base_y) + 0.0 * H, ys) * \
        C.smoothstep(X(0.62), X(0.7), xs) * (1 - alpha) * 0.3
    wcol = np.array([0.55, 0.38, 0.62], np.float32)
    na = np.clip(alpha + stream + wisp, 0, 1)
    col = (col * alpha[..., None] + scol * stream[..., None] + wcol * wisp[..., None]) / np.maximum(na, 1e-4)[..., None]
    alpha = na
    # ------------------------------------------------------------------ downsample + halo outside
    pm = np.dstack([col * alpha[..., None], alpha]).astype(np.float32)
    pm = cv2.resize(pm, (w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros((ph, pw, 4), np.float32)
    a = pm[..., 3]
    rgb = pm[..., :3] / np.maximum(a, 1e-4)[..., None]
    # soft warm halo just outside the sunward silhouette (backlit air)
    rim_s = cv2.resize(rim * rim_w, (w, h), interpolation=cv2.INTER_AREA)
    halo = cv2.GaussianBlur(rim_s, (0, 0), 0.004 * W) * 0.6 * (1 - a)
    halo = np.clip(halo, 0, 0.35)
    a2 = a + halo
    rgb = (rgb * a[..., None] + np.array([1.4, 0.85, 0.6], np.float32) * halo[..., None]) / np.maximum(a2, 1e-4)[..., None]
    out[y0:y1, x0:x1, :3] = rgb
    out[y0:y1, x0:x1, 3] = np.clip(a2, 0, 1)
    return out
