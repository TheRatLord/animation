"""Sky FX for s01: painted cirrus hooks, occlusion-driven crepuscular rays and a lens-flare ghost chain."""
import math
import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------------------------------------ cirrus
def cirrus_strokes(pw, ph, W, strokes, seed=0, sun=None):
    """Deliberate, crisp cirrus: each stroke is a bundle of fine fibres along a curved centre line that
    ends in a hook (uncinus), tapering at both ends, with a soft veil under it. Fibres near the sun are
    brighter. strokes: list of (x0, y0, length, angle_deg, bend, hook, width) in plate px.
    Returns straight-alpha RGBA (ph, pw, 4)."""
    rng = np.random.default_rng(seed)
    ss = 2
    fib = np.zeros((ph * ss, pw * ss), np.float32)
    veil = np.zeros((ph, pw), np.float32)
    for (x0, y0, ln, ang, bend, hook, wd) in strokes:
        a = math.radians(ang)
        d = np.array([math.cos(a), math.sin(a)])
        nrm = np.array([-d[1], d[0]])
        t = np.linspace(0, 1, 220)
        # centre line: gentle bend + hook curling up at the end
        off = bend * np.sin(np.pi * t) * ln + hook * ln * np.clip((t - 0.78) / 0.22, 0, 1) ** 2.2
        cen = np.array([x0, y0]) + np.outer(t * ln, d) + np.outer(off, nrm)
        prof = np.sin(np.pi * np.clip(t, 0, 1)) ** 0.45 * (1 - 0.55 * np.clip((t - 0.8) / 0.2, 0, 1))
        nf = int(rng.integers(14, 24))
        for k in range(nf):
            o = rng.uniform(-1, 1)
            t0, t1 = sorted(rng.uniform(0.0, 1.0, 2))
            if t1 - t0 < 0.25:
                t1 = min(1.0, t0 + 0.25 + rng.uniform(0, 0.3))
            sel = (t >= t0) & (t <= t1)
            if sel.sum() < 3:
                continue
            wav = 0.12 * np.sin(t * rng.uniform(4, 9) + rng.uniform(0, 6))
            pts = cen + np.outer((o + wav) * prof * wd, nrm)
            pts = pts[sel]
            op = rng.uniform(0.35, 1.0)
            lw = max(1, int(round(rng.uniform(0.6, 1.6) * W / 1920 * ss)))
            m = np.zeros_like(fib)
            cv2.polylines(m, [(pts * ss * 16).astype(np.int32)], False, 1.0, lw, cv2.LINE_AA, shift=4)
            # fade fibre ends
            fib = np.maximum(fib, m * op)
        # veil under the bundle
        vm = np.zeros((ph, pw), np.float32)
        poly = np.concatenate([cen + np.outer(prof * wd * 1.1, nrm), (cen - np.outer(prof * wd * 1.1, nrm))[::-1]])
        cv2.fillPoly(vm, [(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
        veil = np.maximum(veil, vm)
    fib = cv2.resize(fib, (pw, ph), interpolation=cv2.INTER_AREA)
    fib = cv2.GaussianBlur(fib, (0, 0), max(0.5, 0.0006 * W))
    fibs = cv2.GaussianBlur(fib, (0, 0), 0.004 * W)
    veil = cv2.GaussianBlur(veil, (0, 0), 0.01 * W)
    a = np.clip(fib * 0.42 + fibs * 0.3 + veil * 0.1, 0, 1)
    col = np.empty((ph, pw, 3), np.float32)
    col[:] = np.array([1.0, 1.0, 1.02], np.float32)
    if sun is not None:
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        pr = np.exp(-np.hypot(xx - sun[0], yy - sun[1]) / (0.4 * W))
        col = col * (1 + 0.25 * pr)[..., None]
    # slightly cooler, lavender veil
    col = col * (1 - 0.1 * (veil * (1 - fib))[..., None] * np.array([0.3, 0.2, 0.0], np.float32))
    return np.dstack([col, a]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ rays
def crepuscular(w, h, lx, ly, occ, strength=0.5, length=0.85, radius=0.07, t=0.0, tint=(1.0, 0.95, 0.82)):
    """Rays fanning out from a sun hidden behind a cloud edge. The bright sky around the sun, masked by
    the cloud's occlusion, is radially smeared AWAY from the sun: every notch/bump of the cloud edge near
    the sun becomes a lit shaft or a shadow wedge. Work at (w, h) (already reduced) resolution.
    Returns additive (h, w, 3)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot(xs - lx, ys - ly) / w
    src = (np.exp(-(d / radius) ** 2) * 1.0 + 0.35 / (1 + (d / 0.02) ** 2)) * (1 - occ) ** 2
    # slight angular modulation so the shafts have varied widths (slowly breathing)
    ang = np.arctan2(ys - ly, xs - lx)
    mod = 0.6 + 0.4 * np.sin(ang * 9 + 0.15 * t) * np.sin(ang * 5 - 0.1 * t + 1.3) * np.sin(ang * 23 + 0.7)
    src = (src * mod).astype(np.float32)
    acc = np.zeros_like(src)
    wsum = 0.0
    n = 40
    for i in range(n):
        k = 1.0 - length * (i / n)
        M = np.float32([[k, 0, (1 - k) * lx], [0, k, (1 - k) * ly]])
        wt = 1.0 - 0.6 * i / n
        acc += cv2.warpAffine(src, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT) * wt
        wsum += wt
    acc /= wsum
    acc = cv2.GaussianBlur(acc, (0, 0), 1.0)
    # keep shafts in the air mostly; a little veil over the cloud
    acc = acc * (1 - 0.93 * occ)
    x = acc * strength * 6.0
    x = x / (1 + 0.6 * x)
    return x[..., None] * np.asarray(tint, np.float32)


# ------------------------------------------------------------------------------------------------ flare
GHOSTS = [  # (position along sun->centre axis, radius (frac W), tint, sides (0 = disc), style)
    (0.22, 0.012, (1.0, 0.85, 0.55), 6, 'fill'),
    (0.45, 0.028, (0.55, 0.9, 1.0), 6, 'fill'),
    (0.62, 0.009, (0.8, 1.0, 0.6), 0, 'fill'),
    (0.8, 0.045, (0.6, 1.0, 0.75), 6, 'ring'),
    (1.02, 0.016, (1.0, 0.6, 0.8), 6, 'fill'),
    (1.2, 0.06, (0.55, 0.65, 1.0), 6, 'fill'),
    (1.38, 0.022, (1.0, 0.8, 0.45), 0, 'fill'),
    (1.6, 0.09, (0.7, 0.85, 1.0), 6, 'ring'),
    (1.85, 0.035, (0.9, 0.55, 1.0), 6, 'fill'),
]


def ghosts(w, h, lx, ly, cx, cy, amt=1.0, scale=1.0):
    """Hexagonal lens ghosts in varied tints along the axis from the light through the optical centre
    (they slide as the camera tilts). Work at reduced (w, h). Returns additive (h, w, 3)."""
    out = np.zeros((h, w, 3), np.float32)
    vx, vy = cx - lx, cy - ly
    for k, r, col, sides, style in GHOSTS:
        gx, gy = lx + vx * k, ly + vy * k
        R = r * w * scale
        bx0, bx1 = int(max(gx - R * 1.4 - 2, 0)), int(min(gx + R * 1.4 + 3, w))
        by0, by1 = int(max(gy - R * 1.4 - 2, 0)), int(min(gy + R * 1.4 + 3, h))
        if bx1 <= bx0 or by1 <= by0:
            continue
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        ex, ey = xx - gx, yy - gy
        rad = np.sqrt(ex * ex + ey * ey)
        if sides:
            th = np.arctan2(ey, ex) + 0.26
            seg = 2 * math.pi / sides
            rad = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
        q = rad / max(R, 1e-3)
        c = np.asarray(col, np.float32)
        if style == 'ring':
            m = np.stack([np.exp(-((q - 0.97) / 0.06) ** 2), np.exp(-((q - 0.93) / 0.06) ** 2),
                          np.exp(-((q - 0.89) / 0.06) ** 2)], -1) * 0.9 + _ss(1.0, 0.6, q)[..., None] * 0.12
            a = 0.08
        else:
            m = np.stack([_ss(1.04, 0.9, q), _ss(1.0, 0.86, q), _ss(0.96, 0.82, q)], -1)
            m = m * (0.55 + 0.45 * _ss(0.3, 0.95, q))[..., None]
            a = 0.11 if r < 0.03 else 0.07
        out[by0:by1, bx0:bx1] += m * c * a * amt
    return cv2.GaussianBlur(out, (0, 0), 0.8)


# ------------------------------------------------------------------------------------------------ round 4
def _fibre_tex(seed, n=512, m=64):
    """Streaky noise texture (m rows x n cols), strongly stretched along x. Tileable enough for remap."""
    rng = np.random.default_rng(seed)
    out = np.zeros((m, n), np.float32)
    amp, tot = 1.0, 0.0
    for o, (cx, cy) in enumerate([(6, 10), (12, 22), (24, 44), (48, 64)]):
        g = rng.random((cy + 3, cx + 3)).astype(np.float32)
        out += amp * cv2.resize(g, (n, m), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.6
    out /= tot
    lo, hi = np.percentile(out, 2), np.percentile(out, 98)
    return np.clip((out - lo) / (hi - lo), 0, 1)


def cirrus_soft(pw, ph, W, streams, seed=0, sun=None):
    """Soft, fibrous, wind-swept cirrus. Each stream: (x0, y0, length, angle_deg, bend, width, opacity).
    Density = across-profile * streaky fibre noise (stretched along the stream) * patchiness, with
    feathered ends; varying opacity. Returns straight-alpha RGBA (ph, pw, 4)."""
    alpha = np.zeros((ph, pw), np.float32)
    for i, (x0, y0, ln, ang, bend, wd, op) in enumerate(streams):
        tex = _fibre_tex(seed * 31 + i)
        tex2 = _fibre_tex(seed * 31 + i + 7, 256, 24)
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        corners = np.array([[0, -3 * wd], [ln, -3 * wd], [0, 3 * wd], [ln, 3 * wd]], np.float64)
        pts = []
        for s_, q_ in corners:
            for bb in (0, bend * ln):
                pts.append((x0 + s_ * ca - (q_ + bb) * sa, y0 + s_ * sa + (q_ + bb) * ca))
        pts = np.array(pts)
        bx0, by0 = int(max(pts[:, 0].min() - 2, 0)), int(max(pts[:, 1].min() - 2, 0))
        bx1, by1 = int(min(pts[:, 0].max() + 3, pw)), int(min(pts[:, 1].max() + 3, ph))
        if bx1 <= bx0 or by1 <= by0:
            continue
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        dx, dy = xx - x0, yy - y0
        s = (dx * ca + dy * sa) / ln                            # 0..1 along
        q = -dx * sa + dy * ca
        q = q - bend * ln * np.sin(np.pi * np.clip(s, 0, 1))    # gentle bow
        q = q + 0.35 * wd * np.sin(s * 7.0 + i)                  # slight waviness
        wprof = wd * (0.35 + 0.65 * np.clip(np.sin(np.pi * np.clip(s, 0, 1)), 0, 1) ** 0.6)
        qq = q / (wprof + 1e-3)
        prof = np.exp(-qq ** 2 * 1.4)
        ends = _ss(0.0, 0.22, s) * _ss(1.0, 0.7, s)
        # fibres: texture sampled with x along the stream (slow) and y across (fast)
        u = (s * 3.0 * tex.shape[1] / 3.0) % tex.shape[1]
        v = np.clip((qq * 0.5 + 0.5) * (tex.shape[0] - 1), 0, tex.shape[0] - 1)
        fib = cv2.remap(tex, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        u2 = (s * tex2.shape[1] * 0.8) % tex2.shape[1]
        v2 = np.clip((qq * 0.35 + 0.5) * (tex2.shape[0] - 1), 0, tex2.shape[0] - 1)
        patch = cv2.remap(tex2, u2.astype(np.float32), v2.astype(np.float32), cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)
        d = prof * ends * (0.25 + 0.75 * _ss(0.3, 0.85, fib)) * (0.35 + 0.65 * _ss(0.2, 0.8, patch))
        alpha[by0:by1, bx0:bx1] = np.maximum(alpha[by0:by1, bx0:bx1], np.clip(d * op, 0, 1))
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(0.6, 0.0008 * W))
    col = np.empty((ph, pw, 3), np.float32)
    col[:] = np.array([0.97, 0.99, 1.03], np.float32)
    if sun is not None:
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        pr = np.exp(-np.hypot(xx - sun[0], yy - sun[1]) / (0.3 * W))
        col = col + (np.array([1.12, 1.08, 1.0], np.float32) - col) * pr[..., None]
    return np.dstack([col, alpha]).astype(np.float32)


GHOSTS2 = [  # (pos along sun->centre axis, radius frac w, tint, hex?, strength)
    (0.3, 0.010, (1.0, 0.8, 0.5), False, 0.16),
    (0.52, 0.026, (0.5, 0.85, 1.0), True, 0.07),
    (0.7, 0.008, (0.7, 1.0, 0.6), False, 0.14),
    (0.9, 0.05, (0.45, 0.75, 1.0), False, 0.045),
    (1.1, 0.016, (1.0, 0.55, 0.75), True, 0.09),
    (1.32, 0.07, (0.55, 0.95, 0.8), False, 0.035),
    (1.5, 0.022, (1.0, 0.8, 0.45), True, 0.07),
    (1.8, 0.1, (0.6, 0.7, 1.0), False, 0.028),
]


def ghosts_soft(w, h, lx, ly, cx, cy, amt=1.0):
    """Soft filled lens ghosts (discs and rounded hexes) with chromatic fringes along the axis from the
    light through the optical centre. Returns additive (h, w, 3) at (w, h)."""
    out = np.zeros((h, w, 3), np.float32)
    vx, vy = cx - lx, cy - ly
    for k, r, col, hexa, st in GHOSTS2:
        gx, gy = lx + vx * k, ly + vy * k
        R = r * w
        bx0, bx1 = int(max(gx - R * 1.3 - 2, 0)), int(min(gx + R * 1.3 + 3, w))
        by0, by1 = int(max(gy - R * 1.3 - 2, 0)), int(min(gy + R * 1.3 + 3, h))
        if bx1 <= bx0 or by1 <= by0:
            continue
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        ex, ey = xx - gx, yy - gy
        rad = np.sqrt(ex * ex + ey * ey)
        if hexa:
            th = np.arctan2(ey, ex) + 0.3
            seg = math.pi / 3
            hx = rad * np.cos((th % seg) - seg / 2) / math.cos(seg / 2)
            rad = rad * 0.35 + hx * 0.65                          # rounded hexagon
        q = rad / max(R, 1e-3)
        c = np.asarray(col, np.float32)
        # per-channel radius shift -> chromatic fringe; soft body brighter toward the rim
        m = np.stack([_ss(1.06, 0.8, q), _ss(1.0, 0.74, q), _ss(0.94, 0.68, q)], -1)
        m = m * (0.55 + 0.45 * _ss(0.2, 0.95, q))[..., None]
        out[by0:by1, bx0:bx1] += m * c * st * amt
    return cv2.GaussianBlur(out, (0, 0), max(0.8, 0.002 * w))
