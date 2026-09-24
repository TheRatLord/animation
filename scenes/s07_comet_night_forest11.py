"""Round-11 forest painter for s07_comet_night (replaces the stacked conifer-top bands of v7).

Individual conifers placed by a clumped point process instead of repeated rows:
  * a low-frequency grove field decides where trees stand -> clumps with ragged edges and snow gaps,
  * tree size varies (random, taller in the heart of a grove, smaller up-slope = farther away),
  * every tree is a tiered spruce silhouette (jagged tiers, slight lean jitter) with a darker shadow half,
    a lighter comet-lit half and a pale snow-dusted crown tip on the lit side,
  * aerial perspective: trees up the slope (farther) are smaller, paler and bluer, low contrast,
  * drawn back to front at 2x and area-downsampled (antialiased silhouettes).
Signature matches v7.conifer_bands7 so it can be swapped in.
"""
import numpy as np
import cv2

from lib import core as C


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _tree_poly(x, y, h, w, tiers, rng, ss):
    """Tiered spruce: returns (full outline, lit half outline) in supersampled px. (x, y) = base centre."""
    pts_l, pts_r = [], []
    top = y - h
    for k in range(tiers):
        f0 = k / tiers
        f1 = (k + 1) / tiers
        yb = top + h * f1
        yt = top + h * f0
        wb = w * (0.25 + 0.75 * f1) * rng.uniform(0.85, 1.15)
        wi = wb * 0.45
        pts_l += [(x - wi, yt + (yb - yt) * 0.35), (x - wb, yb)]
        pts_r += [(x + wi, yt + (yb - yt) * 0.35), (x + wb, yb)]
    trunk = [(x - w * 0.12, y), (x + w * 0.12, y)]
    outline = [(x, top)] + pts_r + [trunk[1], trunk[0]] + pts_l[::-1]
    return outline


def conifer_bands7(rgb, fmask, top_ss, depth, ss, s, Wp, th, light_x, seed, pal, haze_col, haze=0.35,
                   crown_col=(0.56, 0.72, 1.0)):
    rng = np.random.default_rng(seed)
    Hl = rgb.shape[0]
    t1 = top_ss.reshape(Wp, ss).mean(1)
    dep = np.asarray(depth, np.float32)
    dep1 = dep[0] if dep.ndim == 2 else dep
    rows = np.nonzero(fmask.max(1) > 0.3)[0]
    if len(rows) == 0:
        return rgb
    y0, y1 = int(max(rows.min() - 3 * th, 0)), int(min(rows.max() + 2, Hl))
    fs, fl, fb = pal['forest_sh'], pal['forest_lit'], pal['forest_base']
    crown = np.asarray(crown_col, np.float32)
    haze_col = np.asarray(haze_col, np.float32)
    # grove field (clumps) at the canopy scale
    Hr = y1 - y0
    grove = C.fbm(Wp, Hr, Wp / (th * 6.0), 4, seed=seed + 3).astype(np.float32)
    fine = C.fbm(Wp, Hr, Wp / (th * 1.8), 2, seed=seed + 4).astype(np.float32)
    fm = fmask[y0:y1]
    # candidate points on a jittered grid (dense), accepted by grove membership
    sp = max(th * 0.42, 1.5)
    gx = np.arange(0, Wp, sp)
    gy = np.arange(y0, y1, sp * 0.8)
    X, Y = np.meshgrid(gx, gy)
    X = (X + rng.uniform(-0.5, 0.5, X.shape) * sp).ravel()
    Y = (Y + rng.uniform(-0.5, 0.5, Y.shape) * sp * 0.8).ravel()
    ok = (X >= 0) & (X < Wp - 1) & (Y >= y0) & (Y < y1 - 1)
    X, Y = X[ok], Y[ok]
    xi, yi = X.astype(int), (Y - y0).astype(int)
    fmv = fm[yi, xi]
    gv = grove[yi, xi]
    fv = fine[yi, xi]
    memb = fmv * _ss(0.38, 0.58, gv + 0.25 * (fv - 0.5))
    keep = rng.random(len(X)) < memb * 0.95
    X, Y, gv = X[keep], Y[keep], gv[keep]
    xi = X.astype(int)
    q = np.clip((Y - t1[xi]) / np.maximum(dep1[xi], 1.0), 0, 1)          # 0 up-slope (far) .. 1 low (near)
    size = th * (0.5 + 0.9 * q) * rng.uniform(0.65, 1.35, len(X)) * (0.8 + 0.5 * _ss(0.45, 0.75, gv))
    order = np.argsort(Y)
    near = np.exp(-np.abs(X - light_x) / (0.45 * Wp))
    lit_side = np.sign(light_x - X)
    # canvas at ss
    Ws, Hs = Wp * ss, Hr * ss
    can = np.zeros((Hs, Ws, 4), np.float32)
    for i in order:
        h = size[i] * 1.25
        w = h * rng.uniform(0.26, 0.36)
        tiers = int(rng.integers(3, 6))
        x, y = X[i] * ss, (Y[i] - y0) * ss
        lean = rng.normal(0, 0.04) * h
        poly = _tree_poly(x, y, h * ss, w * ss, tiers, rng, ss)
        poly = [(px + lean * ss * (y - py) / (h * ss), py) for (px, py) in poly]
        # colours: aerial haze by distance up the slope, grove heart darker
        hz = haze * (1 - q[i]) ** 1.2 + 0.08
        v = rng.uniform(0.8, 1.15)
        body = (fb * 0.4 + fs * 0.6) * v
        body = body * (1 - hz) + haze_col * hz
        litc = (fl * 1.25 + crown * 0.12 * near[i]) * v
        litc = litc * (1 - hz) + haze_col * hz
        pa = np.array(poly, np.int32)
        cv2.fillPoly(can, [pa], (float(body[0]), float(body[1]), float(body[2]), 1.0))
        # lit half: the side of the outline toward the comet, clipped at the trunk line
        n = len(poly)
        half = [p for p in poly if (p[0] - x) * lit_side[i] >= 0] + [(x, y), (x, y - h * ss)]
        if len(half) >= 3:
            hp = np.array(sorted(half, key=lambda p: p[1]), np.float32)
            # rebuild a proper polygon: apex, outer lit points top->bottom, base centre
            outer = [p for p in poly[1:n] if (p[0] - x) * lit_side[i] > 0.5]
            if outer:
                hp = np.array([(x, y - h * ss)] + outer + [(x, y)], np.int32)
                cv2.fillPoly(can, [hp], (float(litc[0]), float(litc[1]), float(litc[2]), 1.0))
        # snow-dusted crown tip (lit side) on the taller, nearer-the-light trees
        if rng.random() < 0.55:
            tc = crown * (0.55 + 0.45 * near[i])
            tc = tc * (1 - hz) + haze_col * hz
            tip = np.array([(x, y - h * ss), (x + lit_side[i] * w * ss * 0.35, y - h * ss * 0.7),
                            (x, y - h * ss * 0.78)], np.int32)
            cv2.fillPoly(can, [tip], (float(tc[0]), float(tc[1]), float(tc[2]), 1.0))
    small = cv2.resize(can, (Wp, Hr), interpolation=cv2.INTER_AREA)
    a = small[..., 3:4]
    rgb[y0:y1] = rgb[y0:y1] * (1 - a) + small[..., :3]
    return rgb
