"""Painted foliage for s06_seaside: clustered leaf masses (no per-lobe sphere shading).

A mass is painted the way a background artist blocks in a tree / bush:
  * silhouette = a few big envelope blobs whose EDGE is broken into small leaf-cluster scallops
    (florets live only on the contour, flatter and sparser on the underside);
  * interior = a power diagram of leaf clusters; every cluster takes ONE flat value chosen from the
    mass's coherent form normal (ellipsoid + blurred-silhouette normal) dotted with the light, so
    neighbouring clusters of the same value merge into large flat masses whose borders are scalloped
    (lit clusters bulging into the shadow) - deep / shadow / sky-lit / lit / hot values only;
  * a thin warm HDR rim along the sun-facing outer silhouette, a hot edge on the sun-facing borders of
    the lit clusters, a few sky holes on big masses.
Everything is bbox-local and deterministic (rng passed in)."""
import math
import numpy as np
import cv2
from numba import njit

from lib import core as C


def _ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _shift(m, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


@njit(cache=True)
def _circle_field(qx, qy, cx, cy, rad, pad):
    """max over circles of (rad - |q - c|), evaluated only within rad + pad of each circle (else -1e3).
    qx, qy: warped pixel coordinates (h, w) on the integer grid x0.., y0.. (warp < pad)."""
    h, w = qx.shape
    out = np.full((h, w), -1e3, np.float32)
    x0 = qx[0, 0] - (qx[0, 0] - np.floor(qx[0, 0]))
    for i in range(cx.shape[0]):
        R = rad[i] + pad
        # bbox in the (unwarped) grid; warp is small so pad covers it
        ja = int(np.floor(cx[i] - R - qx[0, 0])); jb = int(np.ceil(cx[i] + R - qx[0, 0])) + 1
        ia = int(np.floor(cy[i] - R - qy[0, 0])); ib = int(np.ceil(cy[i] + R - qy[0, 0])) + 1
        if ja < 0:
            ja = 0
        if ia < 0:
            ia = 0
        if jb > w:
            jb = w
        if ib > h:
            ib = h
        for y in range(ia, ib):
            for x in range(ja, jb):
                dx = qx[y, x] - cx[i]
                dy = qy[y, x] - cy[i]
                v = rad[i] - np.sqrt(dx * dx + dy * dy)
                if v > out[y, x]:
                    out[y, x] = v
    return out


def _cells(xs, ys, pts, rad, k=6):
    pts = np.asarray(pts, np.float32)
    pad = 3.0 + 0.5 * float(np.max(rad)) if len(rad) else 3.0
    f = _circle_field(xs.astype(np.float32), ys.astype(np.float32), pts[:, 0].copy(), pts[:, 1].copy(),
                      np.asarray(rad, np.float32), np.float32(pad))
    return None, f


def mass(cv, cx, cy, r, rng, pal, L=(1.0, -0.3), lz=-0.15, flat=0.8, leaf=0.1, rim_px=2.2, rim=1.0,
         clip=None, env=None, lit_bias=0.0, holes=0, alpha=1.0, bottom=0.55, unit=1.0, lit_edge=0.7,
         hot_amt=1.0, lean=0.0):
    """Paint one leaf mass centred (cx, cy), radius r (px) into Canvas cv.

    pal: dict deep, shd, sky, lit, hot, rim (rgb, rim may be HDR). L: screen direction toward the light
    (y down); lz: light z toward the viewer (< 0 = backlit: only flanks facing the sun catch light).
    clip: optional (mask, x0, y0) coverage the mass is multiplied with (in canvas px).
    env: optional list of (x, y, rad) envelope blobs (canvas px) instead of the random ones."""
    Lx, Ly = L
    n = math.hypot(Lx, Ly) + 1e-9
    Lx, Ly = Lx / n, Ly / n
    L3 = np.array([Lx * math.sqrt(max(1 - lz * lz, 0)), Ly * math.sqrt(max(1 - lz * lz, 0)), lz], np.float32)
    if env is None:
        env = []
        ne = int(rng.integers(4, 8))
        for i in range(ne):
            a = rng.uniform(-math.pi, 0) if rng.random() < 0.75 else rng.uniform(0, math.pi)
            rr = rng.uniform(0.0, 0.55) * r
            er = r * rng.uniform(0.38, 0.6)
            env.append((cx + math.cos(a) * rr + lean * r * (0.5 - rr / r), cy + math.sin(a) * rr * flat, er))
        env.append((cx, cy, r * 0.55))
    env = np.asarray(env, np.float64)
    rl = leaf * r
    ext = (env[:, 2] + (env[:, :2] - [cx, cy]).__abs__().max(1)).max() + 2 * rl + 3
    x0 = int(math.floor(cx - ext)); y0 = int(math.floor(cy - ext))
    x1 = int(math.ceil(cx + ext)); y1 = int(math.ceil(cy + ext))
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x1, cv.W), min(y1, cv.H)
    if X1 - X0 < 2 or Y1 - Y0 < 2:
        return
    w, h = X1 - X0, Y1 - Y0
    ys, xs = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
    # ---- envelope signed distance (positive inside)
    D = np.full((h, w), -1e9, np.float32)
    for ex, ey, er in env:
        D = np.maximum(D, er - np.sqrt((xs - ex) ** 2 + ((ys - ey) / 0.92) ** 2))
    ybot = cy + r * flat * bottom
    wav = rl * 0.6 * (np.sin(xs / (rl * 1.7) + rng.uniform(0, 6)) + 0.6 * np.sin(xs / (rl * 0.77) + rng.uniform(0, 6)))
    D = np.minimum(D, (ybot + wav - ys) * 1.5 + rl * 0.8)
    # small domain warp so leaf clusters are irregular, not compass circles
    wn = max(rl * 1.6, 2.0)
    wx = cv2.resize(rng.standard_normal((int(h / wn) + 2, int(w / wn) + 2)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC) * rl * 0.12
    wy = cv2.resize(rng.standard_normal((int(h / wn) + 2, int(w / wn) + 2)).astype(np.float32), (w, h),
                    interpolation=cv2.INTER_CUBIC) * rl * 0.12
    qx, qy = xs + wx, ys + wy
    # ---- leaf-cluster florets on the contour
    step = max(rl * 0.75, 1.2)
    gx, gy = np.meshgrid(np.arange(X0, X1, step), np.arange(Y0, Y1, step))
    gx = gx.ravel() + rng.uniform(-0.5, 0.5, gx.size) * step
    gy = gy.ravel() + rng.uniform(-0.5, 0.5, gy.size) * step
    Dg = cv2.remap(D, (gx - X0).astype(np.float32), (gy - Y0).astype(np.float32), cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE).ravel()
    sel = (Dg > -0.35 * rl) & (Dg < 1.3 * rl)
    sel &= (Dg > 0.1 * rl) | (gy < cy)
    under = np.clip((gy - (cy + r * flat * 0.1)) / (r * flat * 0.5 + 1e-6), 0, 1)
    keep = (rng.random(gx.size) > under * 0.7) & (gy < ybot - 0.3 * rl)
    fx, fy = gx[sel & keep], gy[sel & keep]
    fr_ = rl * rng.uniform(0.65, 1.3, fx.size) * (1 - 0.45 * under[sel & keep])
    F = D - rl * 0.75
    if fx.size:
        _, best = _cells(qx, qy, np.stack([fx, fy], 1), fr_, k=5)
        F = np.maximum(F, best)
    cov = np.clip(F + 0.5, 0, 1)
    if holes and r > 40:
        for _ in range(holes):
            hx = cx + rng.uniform(-0.5, 0.5) * r
            hy = cy + rng.uniform(-0.55, 0.0) * r * flat
            hr = rl * rng.uniform(0.5, 1.0)
            dh = np.sqrt((xs - hx) ** 2 + (ys - hy) ** 2)
            cov = cov * np.clip(dh - hr + 0.5, 0, 1)
    if clip is not None:
        cm, cx0, cy0 = clip
        cl = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            cl[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
        cov = cov * cl
    if cov.max() <= 1e-3:
        return
    # ---- coherent form normal: ellipsoid of the mass + blurred silhouette gradient
    ex = (xs - cx) / r
    ey = (ys - (cy - 0.1 * r * flat)) / (r * flat)
    sm = cv2.GaussianBlur(np.clip(D / (0.5 * r), -1, 1).astype(np.float32), (0, 0), max(0.12 * r, 1.0))
    gyy, gxx = np.gradient(sm)
    gn = np.sqrt(gxx ** 2 + gyy ** 2) + 1e-6
    edge = np.clip(1 - sm, 0, 1)
    nx = 0.55 * ex + 0.45 * (-gxx / gn) * edge
    ny = 0.55 * ey + 0.45 * (-gyy / gn) * edge
    nn = np.sqrt(nx * nx + ny * ny)
    s = np.minimum(nn, 0.97) / (nn + 1e-6)
    nx, ny = nx * s, ny * s
    nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
    lam = nx * L3[0] + ny * L3[1] + nz * L3[2] + lit_bias
    # ---- interior leaf clusters: flat values painted back to front as unions of cluster circles, so
    # every value border is scalloped (lit clusters bulge into the shadow)
    cstep = max(rl * 1.25, 1.5)
    px, py = np.meshgrid(np.arange(X0 - cstep, X1 + cstep, cstep), np.arange(Y0 - cstep, Y1 + cstep, cstep))
    px = px.ravel() + rng.uniform(-0.45, 0.45, px.size) * cstep
    py = py.ravel() + rng.uniform(-0.45, 0.45, py.size) * cstep
    prad = cstep * rng.uniform(0.75, 1.15, px.size)
    mx_ = np.clip(px - X0, 0, w - 1).astype(np.float32)
    my_ = np.clip(py - Y0, 0, h - 1).astype(np.float32)
    lam_c = cv2.remap(lam.astype(np.float32), mx_, my_, cv2.INTER_LINEAR).ravel()
    ny_c = cv2.remap(ny.astype(np.float32), mx_, my_, cv2.INTER_LINEAR).ravel()
    jit = rng.normal(0, 0.07, px.size)
    v = lam_c + jit
    col = np.empty((h, w, 3), np.float32)
    col[:] = pal['shd']
    litm = np.zeros((h, w), np.float32)
    levels = [('deep', ny_c > 0.42 + 0.25 * jit, 1.0),
              ('sky', (ny_c < -0.3 + jit) & (v <= 0.2), 1.0),
              ('lit', v > 0.2, 1.0),
              ('hot', (v > 0.46) if hot_amt > 0 else np.zeros(px.size, bool), hot_amt)]
    for name, sel_, amt in levels:
        if not sel_.any():
            continue
        _, fld = _cells(qx, qy, np.stack([px[sel_], py[sel_]], 1), prad[sel_], k=4)
        m = np.clip(fld + 0.5, 0, 1) * amt
        col = col + (pal[name] * rng.uniform(0.97, 1.03) - col) * m[..., None]
        if name == 'lit':
            litm = m
    litm = cv2.GaussianBlur(litm, (0, 0), 0.5)
    # hot edge on the sun-facing borders of lit clusters
    if lit_edge > 0:
        sh = _shift(litm, -Lx * 1.6 * unit, -Ly * 1.6 * unit)
        le = np.clip(litm - sh, 0, 1)
        col = col + (pal['rim'] * 0.8 - col) * (le * lit_edge * 0.6)[..., None]
    # ---- rim on the sun-facing silhouette
    if rim > 0:
        rp = rim_px * unit
        sh = _shift(cov, -Lx * rp, -Ly * rp)
        rm = np.clip(cov - sh, 0, 1)
        sh2 = _shift(cov, -Lx * rp * 2.5, -Ly * rp * 2.5)
        glow = np.clip(cov - sh2, 0, 1) - rm
        face = np.clip(0.4 + 0.8 * lam, 0, 1)
        col = col + (pal['rim'] - col) * (rm * rim * face)[..., None]
        col = col + (pal['lit'] * 1.1 - col) * (np.clip(glow, 0, 1) * 0.35 * rim * face)[..., None]
    cv.put((cov * alpha, X0, Y0), col)


# ============================================================================================ clumps
# Round 3: irregular leaf CLUSTERS (varied size, serrated leaf-tip silhouettes) painted back to front,
# each lit as a crescent on its sun-facing side (shadow -> mid transition -> lit -> hot -> rim), so the
# lit, toothed edge of a front cluster is cut against the shadowed cluster behind it. Rim width varies
# (thick on edges facing the sun, none on the shadow side); a few sky holes and loose leaf sprays at the
# silhouette.

def _serrated(ccx, ccy, R, rng, tooth, flat=0.85, bottom=0.35, n=None, sharp=1.0, depth=0.6, cap=0.3):
    """closed polygon of one leaf cluster: low-frequency lumpy outline + pointed leaf tips with rounded
    notches between them (tips smaller / fewer on the underside)."""
    k = int(np.clip(2 * math.pi * R / max(tooth, 1.0), 7, 60))
    n = n or max(8 * k, 48)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    lump = 1 + 0.14 * np.sin(2 * th + rng.uniform(0, 6.3)) + 0.09 * np.sin(3 * th + rng.uniform(0, 6.3)) \
        + 0.05 * np.sin(5 * th + rng.uniform(0, 6.3))
    # tooth phase jitter: irregular spacing
    ph = th * k / (2 * math.pi) + 0.25 * np.sin(th * rng.uniform(2, 4) + rng.uniform(0, 6.3))
    saw = ph - np.floor(ph)
    x = np.abs(2 * saw - 1)                                  # 1 at the notch, 0 at the tip
    tid = np.floor(ph).astype(int) % k
    dep = rng.uniform(0.5, 1.3, k)[tid]                      # tooth depth varies tip to tip
    down = np.clip(np.sin(th), 0, 1)                         # screen y down: underside
    d = np.minimum((tooth / R) * depth * dep * (1 - 0.6 * down), cap)
    prof = sharp * (1 - (1 - x) ** 1.6) + (1 - sharp) * x ** 2
    rr = R * lump * (1 - d * prof)
    rr = rr * (1 - bottom * down ** 2)
    return np.stack([ccx + rr * np.cos(th), ccy + rr * np.sin(th) * flat], 1)


def _fill(poly, x0, y0, w, h, ss=2):
    m = np.zeros((h * ss, w * ss), np.uint8)
    q = ((np.asarray(poly, np.float64) - [x0, y0]) * ss * 16).astype(np.int32)
    cv2.fillPoly(m, [q], 255, lineType=cv2.LINE_AA, shift=4)
    return cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)


def _crescent(m, dx, dy):
    """part of mask m whose neighbour at (+dx, +dy) (toward the light) is outside m."""
    return np.clip(m - _shift(m, -dx, -dy), 0, 1)


def _tips(cov, col, rng, tooth, Lx, Ly, pal, rim, u, density=0.8, lit_bias=0.0):
    """Jagged leaf-tip silhouette: small pointed leaves along the outer edge of the clump, pointing
    outward, coloured like the paint just inside; tips facing the light turn hot.
    Returns (cov, col) updated (bbox local)."""
    h, w = cov.shape
    ring = (cov > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(ring, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return cov, col
    sm = cv2.GaussianBlur(cov, (0, 0), max(tooth * 0.8, 1.0))
    gy, gx = np.gradient(sm)
    ss = 2
    polys, lit_polys = [], []
    for c in cnts:
        c = c[:, 0, :].astype(np.float64)
        per = len(c)
        nt = int(per / max(tooth, 1.0) * density)
        if nt < 2:
            continue
        idx = np.sort(rng.choice(per, nt, replace=False))
        for i in idx:
            x, y = c[i]
            xi, yi = int(min(max(x, 0), w - 1)), int(min(max(y, 0), h - 1))
            nx, ny = -gx[yi, xi], -gy[yi, xi]
            g = math.hypot(nx, ny)
            if g < 1e-6:
                continue
            nx, ny = nx / g, ny / g
            if ny > 0.55:                   # few tips hanging off the underside
                continue
            ang = math.atan2(ny, nx) + rng.uniform(-0.6, 0.6)
            tx, ty = math.cos(ang), math.sin(ang)
            ln = tooth * rng.uniform(0.7, 1.6)
            wd = ln * rng.uniform(0.28, 0.42)
            bx, by = x - tx * ln * 0.35, y - ty * ln * 0.35
            q = [(bx - ty * wd, by + tx * wd),
                 (bx + tx * ln * 0.45 - ty * wd * 0.9, by + ty * ln * 0.45 + tx * wd * 0.9),
                 (bx + tx * ln, by + ty * ln),
                 (bx + tx * ln * 0.45 + ty * wd * 0.9, by + ty * ln * 0.45 - tx * wd * 0.9),
                 (bx + ty * wd, by - tx * wd)]
            (lit_polys if (nx * Lx + ny * Ly + lit_bias) > 0.35 else polys).append(q)
    out = []
    for group in (polys, lit_polys):
        m = np.zeros((h * ss, w * ss), np.uint8)
        if group:
            qq = [(np.asarray(q, np.float64) * ss * 16).astype(np.int32) for q in group]
            cv2.fillPoly(m, qq, 255, lineType=cv2.LINE_AA, shift=4)
        out.append(cv2.resize(m.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA))
    tm, tl = out
    # colour just inside the edge, extended outward (normalised blur)
    k = max(tooth * 0.7, 1.0)
    pb = cv2.GaussianBlur(col * cov[..., None], (0, 0), k)
    ab = cv2.GaussianBlur(cov, (0, 0), k)
    tcol = pb / np.maximum(ab, 1e-4)[..., None]
    newm = np.clip(tm + tl, 0, 1) * (1 - cov)
    if rim > 0:
        tcol = tcol + (pal['hot'] - tcol) * (tl * 0.6)[..., None]
    col = col * cov[..., None] + tcol * newm[..., None]
    cov = cov + newm
    col = col / np.maximum(cov, 1e-4)[..., None]
    return np.clip(cov, 0, 1), col


def clump(cv, cx, cy, r, rng, pal, L=(1.0, -0.35), lz=-0.3, flat=0.75, leaf=0.1, rim_px=2.2, rim=1.0,
          clip=None, lit_bias=0.0, holes=None, sprays=None, alpha=1.0, unit=1.0, n=None, lean=0.0, **_):
    """Paint one foliage clump (centre cx, cy, radius r px) into Canvas cv as clustered leaf masses:
    lobed leaf clusters of varied size painted back to front, each lit as a scalloped crescent on its
    sun-facing side (shadow -> mid -> lit -> hot), a jagged leaf-tip silhouette, a rim whose width
    follows how squarely the edge faces the light (none on the shadow side), a few sky holes.
    pal: deep, shd, sky, lit, hot, rim (+ optional mid). L: screen direction to the light (y down)."""
    Lx, Ly = L
    nL = math.hypot(Lx, Ly) + 1e-9
    Lx, Ly = Lx / nL, Ly / nL
    u = max(unit, 0.3)
    tooth = max(leaf * r * 0.45, 2.6 * u)
    ext = r * 1.35 + 3 * tooth + 6
    X0, Y0 = int(max(math.floor(cx - ext), 0)), int(max(math.floor(cy - ext * flat - tooth * 2), 0))
    X1, Y1 = int(min(math.ceil(cx + ext), cv.W)), int(min(math.ceil(cy + ext * flat + tooth * 2), cv.H))
    if X1 - X0 < 3 or Y1 - Y0 < 3:
        return
    w, h = X1 - X0, Y1 - Y0
    mid = pal.get('mid', pal['shd'] * 0.55 + pal['lit'] * 0.45)
    col = np.zeros((h, w, 3), np.float32)
    cov = np.zeros((h, w), np.float32)
    # ---- cluster layout: a few big clusters + smaller ones, heaped (upper half fuller)
    nc = n or int(np.clip(4 + r / (12.0 * u), 4, 16))
    cl = []
    for i in range(nc):
        a = rng.uniform(0, 2 * math.pi)
        rad = math.sqrt(rng.uniform(0, 1)) * 0.7
        px = cx + math.cos(a) * rad * r + lean * r * (0.5 - rad)
        py = cy + math.sin(a) * rad * r * flat
        if py > cy + 0.35 * r * flat:
            py = cy + 0.35 * r * flat - rng.uniform(0, 0.1) * r
        R = r * (0.24 + 0.34 * rng.uniform(0, 1) ** 1.5) * (1.15 - 0.4 * rad)
        cl.append((px, py, R, rng.uniform(-0.2, 0.2)))
    cl.append((cx, cy, r * 0.55, 0.0))                   # core (painted first)
    # back to front: upper / far-side clusters first; lower & sun-side ones in front
    cl.sort(key=lambda c: -9.0 if c[2] >= r * 0.55 else
            (c[1] - cy) / (r * flat) + 0.35 * ((c[0] - cx) * Lx) / r + c[3])
    lobe = tooth * 2.4
    for (px, py, R, _j) in cl:
        poly = _serrated(px, py, R, rng, lobe, flat=0.88, sharp=0.35, depth=0.45, cap=0.14)
        m = _fill(poly, X0, Y0, w, h)
        if m.max() <= 0:
            continue
        # clump-level form: position on the clump's ellipsoid toward the light
        nx, ny = (px - cx) / r, (py - cy) / (r * flat)
        lam = 1.0 * (nx * Lx + ny * Ly) - 0.12 * ny + lit_bias + lz * 0.2 + rng.normal(0, 0.1)
        f = float(np.clip(0.42 + lam, 0.0, 1.0))           # 0 = fully in shade .. 1 = lit face
        Dm = R * (0.1 + 0.7 * f) if f > 0.28 else 0.0
        Dl = R * (0.05 + 0.55 * (f - 0.4)) if f > 0.45 else 0.0
        Dh = R * 0.35 * (f - 0.65) if f > 0.7 else 0.0
        c_ = np.empty((h, w, 3), np.float32)
        c_[:] = pal['shd']
        dd = _crescent(m, 0.0, R * 0.4)                   # underside in deep shadow
        c_ += (pal['deep'] - c_) * (dd * 0.75)[..., None]
        if f < 0.55:                                      # skylight on up-facing edges of shaded clusters
            sk = _crescent(m, 0.0, -R * 0.15) * (0.55 - f) * 1.3
            c_ += (pal['sky'] - c_) * np.clip(sk, 0, 1)[..., None]
        # value boundaries: the shadow shape is a second lobed outline displaced away from the light
        for (D, key, amt, tf, up) in ((Dm, None, 1.0, 1.3, 0.3), (Dl, 'lit', 1.0, 1.0, 0.0),
                                      (Dh, 'hot', 0.9, 0.8, -0.1)):
            if D <= 0.8:
                continue
            sp = _serrated(px - Lx * D, py - Ly * D + up * D, R * rng.uniform(0.98, 1.06), rng, lobe * tf,
                           flat=0.88, bottom=0.0, sharp=0.3, depth=0.5, cap=0.16)
            cm_ = np.clip(m - _fill(sp, X0, Y0, w, h), 0, 1)
            c_ += ((mid if key is None else pal[key]) - c_) * (cm_ * amt)[..., None]
        col = c_ * m[..., None] + col * (1 - m[..., None])
        cov = m + cov * (1 - m)
    # ---- jagged leaf-tip silhouette
    cov, col = _tips(cov, col, rng, tooth, Lx, Ly, pal, rim, u, lit_bias=lit_bias)
    # ---- rim on the outer silhouette: width follows how squarely the edge faces the light
    if rim > 0:
        sm = cv2.GaussianBlur(cov, (0, 0), max(tooth, 1.5))
        gy, gx = np.gradient(sm)
        g = np.sqrt(gx * gx + gy * gy) + 1e-6
        face = np.clip((-gx * Lx - gy * Ly) / g, 0, 1)            # outward normal . light
        rw_lo, rw_hi = rim_px * u * 0.8, rim_px * u * 2.6
        r_thin = _crescent(cov, Lx * rw_lo, Ly * rw_lo)
        r_thick = _crescent(cov, Lx * rw_hi, Ly * rw_hi)
        rm = r_thin * _ss(0.2, 0.5, face) + (r_thick - r_thin) * _ss(0.55, 0.9, face)
        col = col + (pal['rim'] - col) * np.clip(rm * rim, 0, 1)[..., None]
    # ---- a few sky holes in the upper part of big clumps
    nh = (int(r / (55.0 * u)) if holes is None else holes)
    if nh and r > 30 * u:
        inner = cv2.erode((cov > 0.99).astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=int(tooth * 2) + 1)
        pts = np.argwhere(inner[: int(h * 0.5)] > 0)
        for _ in range(min(nh, 3)):
            if not len(pts):
                break
            hy, hx = pts[rng.integers(0, len(pts))]
            hr = tooth * rng.uniform(1.0, 1.8)
            hp = _serrated(hx + X0, hy + Y0, hr, rng, tooth * 0.7, flat=rng.uniform(0.5, 0.75), bottom=0.3)
            cov = cov * (1 - _fill(hp, X0, Y0, w, h))
    if clip is not None:
        cm, cx0, cy0 = clip
        cl_ = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            cl_[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
        cov = cov * cl_
    if cov.max() <= 1e-3:
        return
    cv.put((cov * alpha, X0, Y0), col)
