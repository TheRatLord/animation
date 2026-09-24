"""Painted towering cumulus for s10 (round 5 rebuild: no per-ball shading).

Shape: a hierarchy of lobes (a few big masses stacked into a tower -> medium lobes grown on the outer
contour -> small cauliflower beads along the sunward / upper silhouette only), evaluated on a
domain-warped grid so no outline is a clean circle.  The base dissolves into torn horizontal wisps.

Paint (the key difference from a 3D sphere render):
  * ONE form normal from the heavily blurred silhouette ('pillow' of the whole tower) carries the light:
    one continuous shadow plane runs down the side away from the sun; lobes only perturb it slightly
    (25%), so the terminator wobbles along the lobe contours instead of every lobe getting a crescent.
  * two-to-three painted tones: cool lavender shadow (deeper and bluer low down, a violet bounce from the
    sea at the very bottom) | narrow pink terminator band | peach -> warm cream light, plus a few flat
    hot highlight shapes where the form faces the sun most.  Higher parts catch more light (sunrise).
  * thin crease lines only where a big lobe overlaps the one behind it (soft, low contrast).
  * a crisp 2-3 px silver-gold rim along the sun-facing silhouette, with a soft translucent glow inside.
"""
import math
import numpy as np
import cv2

from lib import core as C


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _rgb(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def _blur(a, s):
    if s < 0.3:
        return a
    if s < 4:
        return cv2.GaussianBlur(a, (0, 0), s)
    k = s / 4.0
    h, w = a.shape[:2]
    sm = cv2.resize(a, (max(int(w / k), 2), max(int(h / k), 2)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 3.87)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_CUBIC)


def tower_lobes(rng, Hc, ax, base_y, profile, n_levels=9, lean=0.0):
    """Big masses: list of (cx, cy, r, z0). profile(u) -> half width (fraction of Hc) at height u (0..1)."""
    L = []
    for i in range(n_levels):
        u = (i + 0.35) / n_levels
        hw = profile(u) * Hc
        cy = base_y - u * Hc * 0.92
        cxc = ax + lean * u * u * Hc
        k = max(1, int(round(hw / (0.1 * Hc))))
        for j in range(k):
            fx = (j + 0.5) / k * 2 - 1
            r = hw * rng.uniform(0.45, 0.9) / math.sqrt(k) * 1.15
            r = min(max(r, 0.06 * Hc), 0.13 * Hc)
            cx = cxc + fx * (hw - r * 0.8) + rng.normal(0, 0.02) * Hc
            L.append((cx, cy + rng.normal(0, 0.045) * Hc, r, rng.uniform(0.0, 0.15) * Hc))
    return L


_MARGIN = [0.0]


def _disc_union(W, H, xs, ys, lobes, depth=None, ids=None, base_id=0):
    """Rasterise lobes into mask/depth (z-buffer of sphere front surfaces)."""
    m = np.zeros((H, W), np.float32) if depth is None else None
    for n, (cx, cy, r, z0) in enumerate(lobes):
        mg = r + 2 + _MARGIN[0]
        x0, x1 = int(max(cx - mg, 0)), int(min(cx + mg + 1, W))
        y0, y1 = int(max(cy - mg, 0)), int(min(cy + mg + 1, H))
        if x1 <= x0 or y1 <= y0:
            continue
        dx = xs[y0:y1, x0:x1] - cx
        dy = ys[y0:y1, x0:x1] - cy
        d2 = dx * dx + dy * dy
        ins = d2 < r * r
        zf = z0 + np.sqrt(np.clip(r * r - d2, 0, None))
        if depth is not None:
            sub = depth[y0:y1, x0:x1]
            upd = ins & (zf > sub)
            sub[upd] = zf[upd]
            if ids is not None:
                ids[y0:y1, x0:x1][upd] = base_id + n
        else:
            m[y0:y1, x0:x1] = np.maximum(m[y0:y1, x0:x1], ins.astype(np.float32))
    return m


def contour_lobes(rng, mask, Hc, n, rmin, rmax, sun2, bias_up=0.6, sun_bias=0.4, inset=0.55, zfun=None):
    """Grow lobes on the contour of mask (upper / sunward parts favoured)."""
    m8 = (mask > 0.5).astype(np.uint8)
    cs, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    pts = np.concatenate([c[:, 0, :] for c in cs], 0).astype(np.float32)
    g = _blur(mask, 0.03 * Hc)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    pi = np.clip(pts[:, 1].astype(int), 0, mask.shape[0] - 1)
    pj = np.clip(pts[:, 0].astype(int), 0, mask.shape[1] - 1)
    nx, ny = -gx[pi, pj], -gy[pi, pj]
    nl = np.sqrt(nx * nx + ny * ny) + 1e-6
    nx, ny = nx / nl, ny / nl
    wgt = 0.15 + bias_up * np.clip(-ny, 0, 1) + sun_bias * np.clip(nx * sun2[0] + ny * sun2[1], 0, 1)
    wgt = wgt * (ny < 0.55)          # never on the flat underside
    wgt = wgt / wgt.sum()
    idx = rng.choice(len(pts), size=n, p=wgt)
    out = []
    for i in idx:
        r = rng.uniform(rmin, rmax) * Hc
        cx = pts[i, 0] - nx[i] * r * inset
        cy = pts[i, 1] - ny[i] * r * inset
        z0 = zfun(cx, cy) if zfun else 0.0
        out.append((cx, cy, r, z0))
    return out


def paint_tower(W, H, Hc, ax, base_y, seed, P, sun2, L, profile, lean=0.0, n_levels=9, detail=1.0,
                px=1.0, haze=0.0, haze_col=None, light_k=1.0):
    """Returns straight-alpha RGBA (H, W, 4) float32.
    sun2: 2D unit vector toward the sun in plate coords (x right, y down). L: 3D light (x, y down, z to
    viewer). px: plate px per 1080p screen px (for line widths)."""
    rng = np.random.default_rng(seed)
    xs, ys = C.grid(W, H)
    # domain warp: irregular outlines, lobes never clean circles
    wn = 0.022 * Hc
    wx = (C.fbm(W, H, 5.0, 4, seed=seed + 1) - 0.5) * 2 * wn + (C.fbm(W, H, 18.0, 3, seed=seed + 2) - 0.5) * 0.5 * wn
    wy = (C.fbm(W, H, 5.0, 4, seed=seed + 3) - 0.5) * 2 * wn + (C.fbm(W, H, 18.0, 3, seed=seed + 4) - 0.5) * 0.5 * wn
    gxs, gys = xs + wx, ys + wy
    _MARGIN[0] = float(max(np.abs(wx).max(), np.abs(wy).max())) + 2
    big = tower_lobes(rng, Hc, ax, base_y, profile, n_levels, lean)
    m0 = _disc_union(W, H, gxs, gys, big)

    def zform(cx, cy):
        return 0.0
    med = contour_lobes(rng, m0, Hc, int(90 * detail), 0.03, 0.07, sun2, 0.8, 0.6, 0.3)
    m1 = np.maximum(m0, _disc_union(W, H, gxs, gys, med))
    small = contour_lobes(rng, m1, Hc, int(260 * detail), 0.01, 0.022, sun2, 1.0, 1.4, 0.1)
    mask = np.maximum(m1, _disc_union(W, H, gxs, gys, small))
    alpha = _blur(mask, 0.7 * px)
    # ---- form: pillow of the whole silhouette (normal + depth)
    q = 4
    ms = cv2.resize(mask, (W // q, H // q), interpolation=cv2.INTER_AREA)
    B = cv2.GaussianBlur(ms, (0, 0), 0.09 * Hc / q)
    B2 = cv2.GaussianBlur(ms, (0, 0), 0.03 * Hc / q)
    Pm = np.sqrt(np.clip(B, 0, 1)) * 0.75 + np.sqrt(np.clip(B2, 0, 1)) * 0.25
    fgx = cv2.Sobel(Pm, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    fgy = cv2.Sobel(Pm, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    k = 0.16 * Hc / q
    fnx, fny, fnz = -fgx * k, -fgy * k, np.ones_like(fgx)
    fl = np.sqrt(fnx ** 2 + fny ** 2 + fnz ** 2)
    up = lambda a: cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
    fnx, fny, fnz = up(fnx / fl), up(fny / fl), up(fnz / fl)
    Df = up(Pm) * 0.16 * Hc
    # ---- detail lobes embedded in the form: contour lobes + cauliflower heads inside the upper tower
    iy, ix = np.nonzero(mask[::4, ::4] > 0.5)
    uu = (base_y - iy * 4) / Hc
    wgt = _ss(0.2, 0.9, uu) + 0.05
    wgt = wgt / wgt.sum()
    pick = rng.choice(len(iy), size=int(16 * detail), p=wgt)
    inner = [(ix[i] * 4.0, iy[i] * 4.0, rng.uniform(0.06, 0.12) * Hc, 0.0) for i in pick]
    # painter's order: lower lobes first, upper lobes drawn over them -> every lobe boundary is the
    # crisp lower ARC of the lobe above (the cauliflower read), never a straight sphere intersection
    allb = [(cx, cy, r) for (cx, cy, r, _) in big + inner + [m_ for m_ in med if m_[2] > 0.055 * Hc]]
    allb.sort(key=lambda q: -(q[1] + 0.3 * q[2]))
    ids = np.full((H, W), -1, np.int32)
    mg = _MARGIN[0]
    for n, (cx, cy, r) in enumerate(allb):
        x0, x1 = int(max(cx - r - mg, 0)), int(min(cx + r + mg + 1, W))
        y0, y1 = int(max(cy - r - mg, 0)), int(min(cy + r + mg + 1, H))
        if x1 <= x0 or y1 <= y0:
            continue
        dx = gxs[y0:y1, x0:x1] - cx
        dy = gys[y0:y1, x0:x1] - cy
        ins = dx * dx + dy * dy < r * r
        ids[y0:y1, x0:x1][ins] = n
    ids[mask < 0.5] = -1
    cxa = np.array([q[0] for q in allb] + [0], np.float32)
    cya = np.array([q[1] for q in allb] + [0], np.float32)
    ra = np.array([q[2] for q in allb] + [1], np.float32)
    ii = np.where(ids >= 0, ids, len(allb))
    lx = np.clip((gxs - cxa[ii]) / ra[ii], -1, 1)
    ly = np.clip((gys - cya[ii]) / ra[ii], -1, 1)
    lz = np.sqrt(np.clip(1 - lx * lx - ly * ly, 0.02, 1))
    lnx, lny, lnz = _blur(lx.astype(np.float32), 1.0 * px), _blur(ly.astype(np.float32), 1.0 * px), lz
    # lobe boundaries: pixel belongs to a lower lobe and the lobe just above it is a different one
    edge = np.zeros((H, W), np.float32)
    kw = max(1, int(round(2.5 * px)))
    for sft in range(1, kw * 3 + 1):
        upi = np.empty_like(ids)
        upi[sft:] = ids[:-sft]
        upi[:sft] = -1
        e = ((upi != ids) & (upi >= 0) & (ids >= 0)).astype(np.float32) * math.exp(-(sft - 1) / kw)
        edge = np.maximum(edge, e)
    dfill = np.zeros((H, W), np.float32)
    wl = 0.08
    nx, ny, nz = fnx * (1 - wl) + lnx * wl, fny * (1 - wl) + lny * wl, fnz * (1 - wl) + lnz * wl
    nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
    nx, ny, nz = nx / nl, ny / nl, nz / nl
    ndl = nx * L[0] + ny * L[1] + nz * L[2]
    # height: sunrise light catches the tops first; lower parts sink into shadow
    u = np.clip((base_y - ys) / Hc, 0, 1.2)
    lift = 0.3 * _ss(0.25, 0.95, u) - 0.12
    v = ndl + lift
    # painted tone split (soft-edged but decisive)
    # per-lobe modelling INSIDE each tone (never crossing the terminator): lobe caps facing up / sunward
    lcap = np.clip(-lny * 0.85 + lnx * np.sign(L[0]) * 0.45, -1, 1)
    v = v + 0.05 * lcap
    lit = _ss(-0.015, 0.03, v) * light_k
    hot = _ss(0.25, 0.5, lcap) * _ss(0.35, 0.9, u) * _ss(0.04, 0.12, v)
    band = np.exp(-((v - 0.005) / 0.022) ** 2) * (1 - lit * 0.5)
    # crease lines where a lobe overlaps the one behind (depth step)
    crease = _blur(edge, 0.5 * px) * mask
    # ---- colours
    shd_hi, shd, deep, bounce = P['shd_hi'], P['shd'], P['deep'], P['bounce']
    sh_col = shd_hi[None, None] + (shd - shd_hi)[None, None] * _ss(0.8, 0.15, u)[..., None]
    sh_col = sh_col + (deep - sh_col) * (_ss(0.4, 0.0, u) * 0.6)[..., None]
    sh_col = sh_col + (bounce - sh_col) * (_ss(0.12, 0.0, u) * 0.5 * np.clip(fny, 0, 1) * 2)[..., None]
    # sky fill on the upward faces of the lobes in shadow (soft internal gradients)
    sh_col = sh_col + (P['fill'] - sh_col) * (_ss(-0.1, 0.8, lcap) * 0.6)[..., None]
    sh_col = sh_col + (deep - sh_col) * (_ss(-0.2, -0.85, lcap) * 0.3)[..., None]
    lc = P['peach'] + (P['cream'] - P['peach']) * (0.3 + 0.7 * _ss(-0.6, 0.5, lcap))[..., None]
    lc = lc + (P['pink'] - lc) * (_ss(-0.4, -0.9, lcap) * 0.45)[..., None]
    col = sh_col + (lc - sh_col) * lit[..., None]
    col = col + (P['pink'] - col) * (band * 0.35)[..., None]
    col = col + (P['hot'] - col) * (hot * 0.85)[..., None]
    # crease lines: warm darker in light, deeper blue in shadow
    cdark = col * np.array([0.84, 0.76, 0.86], np.float32)
    col = col + (cdark - col) * (crease * (0.12 + 0.3 * lit))[..., None]
    # ---- rim: sun-facing silhouette + inner translucency
    din = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
    ga = _blur(mask, 2.0 * px)
    ogx = -cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3)
    ogy = -cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
    ol = np.sqrt(ogx * ogx + ogy * ogy) + 1e-6
    face = np.clip((ogx * sun2[0] + ogy * sun2[1]) / ol, 0, 1) * (ol > 1e-3)
    face = _blur(face.astype(np.float32), 2.0 * px)
    face = _ss(0.15, 0.7, face) * _ss(0.05, 0.35, u)
    rim = np.exp(-din / (1.3 * px)) * face
    glow = np.exp(-din / (4.0 * px)) * face * 0.35
    col = col + (P['glow'] - col) * np.clip(glow, 0, 1)[..., None]
    col = col + (P['rim'] - col) * np.clip(rim, 0, 1)[..., None]
    # ---- torn wispy base
    st = C.fbm(W, H, 3.0, 3, seed=seed + 9)
    stx = cv2.resize(C.fbm(max(W // 8, 4), H, 1.0, 4, seed=seed + 10, aspect=False), (W, H))
    bu = (base_y - ys) / Hc
    tear = _ss(0.0, 0.2, bu + 0.12 * (stx - 0.5) + 0.06 * (st - 0.5))
    streak = _ss(0.35, 0.6, stx)
    alpha = alpha * np.maximum(tear, streak * _ss(-0.06, 0.04, bu) * 0.8)
    # the base sinks into the sea: fade toward the sea-haze colour and thin out
    if 'base_haze' in P:
        bh = _ss(0.3, 0.0, bu)[..., None]
        col = col + (P['base_haze'] - col) * bh * 0.6
    if haze and haze_col is not None:
        col = col + (haze_col - col) * haze
    return np.dstack([col, alpha]).astype(np.float32)
