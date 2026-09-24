"""s01 helper (round 12): the hero cumulonimbus as a hand-PAINTED lobe hierarchy (no volume render).

Why: the volumetric render + paint pass gave evenly sized bevelled blobs separated by dark crevices
(crumpled paper / clay).  A Shinkai background painter builds the tower the other way round:

  * one broad value design first (big form): warm cream lit flank toward the sun (upper left), a broad
    desaturated blue-grey shade mass on the down-sun third, a darker flat base; 3-4 soft value steps;
  * then lobes at 2-3 scales painted OVER that body: big masses (150-250 px), medium heads (50-90 px),
    small cauliflower (15-30 px) only along the lit silhouette.  Every lobe is a crisp lit cap whose
    lower part is TRANSPARENT (a lost underside fading into the body value underneath) - never a dark
    groove / outline.  Lobes are painted back to front (upper first, lower in front), so a lower
    head's crisp top sits over the soft belly of the one above;
  * each lobe is lit on its own (value = big form at the lobe + the lobe's own facing term), so the
    terminator wraps around individual lobes instead of running as one vertical line;
  * warm reflected light in the low shaded lobes, a flattened top-heavy anvil / crown with cirrus shear
    on the up-sun side, lost soft silhouette on the shade side against the sky, a flat hazed base that
    is continuous with the shade and fades into the horizon haze.
Deterministic, resolution independent (all sizes in units of the tower height Hc).
"""
import math

import numpy as np
import cv2

from lib import clouds3 as K3

F32 = np.float32


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise(w, h, px, seed, octaves=3, stretch=1.0, angle=0.0):
    return K3._noise(w, h, max(w / max(px, 1.0), 2.0), seed, octaves, stretch=stretch, angle=angle)


def _c(v):
    return np.asarray(v, F32)


# half-widths (units of Hc) left / right of the axis vs height s (0 base .. 1 top): broad foot, a waist,
# swelling into a top-heavy spreading crown / young anvil (~1.45x the mid-body width), flattened top
PROF_S = [0.0, 0.08, 0.16, 0.26, 0.36, 0.46, 0.56, 0.66, 0.74, 0.8, 0.86, 0.9, 0.94, 0.98, 1.0]
PROF_L = [0.42, 0.45, 0.43, 0.36, 0.33, 0.27, 0.26, 0.23, 0.27, 0.37, 0.45, 0.49, 0.46, 0.34, 0.16]
PROF_R = [0.34, 0.39, 0.33, 0.37, 0.3, 0.33, 0.25, 0.31, 0.31, 0.37, 0.42, 0.45, 0.42, 0.3, 0.14]


def _wl(s):
    return float(np.interp(s, PROF_S, PROF_L))


def _wr(s):
    return float(np.interp(s, PROF_S, PROF_R))


class Lobe:
    __slots__ = ('x', 'y', 'rx', 'ry', 'lvl', 'key', 'warm', 'jit', 'rot')

    def __init__(self, x, y, rx, ry, lvl, key, warm=0.0, jit=0.0, rot=0.0):
        self.x, self.y, self.rx, self.ry = float(x), float(y), float(rx), float(ry)
        self.lvl, self.key, self.warm, self.jit, self.rot = lvl, key, warm, jit, rot


def _union(lobes, w, h, x0, y0, q=4):
    """Low-res coverage mask of the lobes (for silhouette contour sampling)."""
    m = np.zeros((h // q + 1, w // q + 1), np.uint8)
    for lb in lobes:
        cv2.ellipse(m, (int((lb.x - x0) / q * 16), int((lb.y - y0) / q * 16)),
                    (max(int(lb.rx / q * 16), 16), max(int(lb.ry / q * 16), 16)), lb.rot, 0, 360, 255, -1,
                    cv2.LINE_AA, 4)
    return m


def _boundary(m, q, x0, y0, step_px, sig):
    """Points on the outer contour of mask m (low-res), spaced ~step_px (full-res px), with outward normals."""
    cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return []
    c = max(cs, key=len)[:, 0, :].astype(F32)
    mb = cv2.GaussianBlur(m.astype(F32) / 255, (0, 0), max(sig / q, 1.0))
    gy, gx = np.gradient(mb)
    out = []
    # arc-length resample
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(c, axis=0).T))]
    L = d[-1]
    n = max(int(L * q / step_px), 8)
    for t in np.linspace(0, L, n, endpoint=False):
        k = int(np.searchsorted(d, t))
        k = min(k, len(c) - 1)
        px, py = c[k]
        ix, iy = int(px), int(py)
        nx, ny = -gx[iy, ix], -gy[iy, ix]
        nl = math.hypot(nx, ny) + 1e-9
        out.append((px * q + x0, py * q + y0, nx / nl, ny / nl))
    return out


def build_lobes(rng, cx, base_y, Hc, Ld):
    """Hierarchical lobe list: big masses (tiers following the top-heavy profile), medium heads on the
    silhouette + on the upper rims of the big masses, small cauliflower only on the lit edges."""
    lobes = []
    lean = 0.16

    def ax(s):
        return cx + Hc * (lean * s + 0.012 * math.sin(6.0 * s + 1.0))

    key = 0
    # ---------------- big masses (radius 0.09-0.16 Hc -> ~150-250 px across at 1080p)
    tiers = [0.05, 0.14, 0.24, 0.34, 0.44, 0.54, 0.63, 0.72, 0.8, 0.87, 0.93]
    pn = rng.standard_normal(len(tiers) * 2)
    for ti, s in enumerate(tiers):
        l_, r_ = _wl(s) * (1 + 0.06 * pn[2 * ti]), _wr(s) * (1 + 0.06 * pn[2 * ti + 1])
        wid = (l_ + r_) * Hc
        flat = 0.62 if s < 0.1 else (0.66 if s > 0.84 else 0.85)
        ry0 = Hc * (0.085 if s < 0.1 else (0.1 + 0.04 * s) * (0.8 if s > 0.84 else 1.0))
        rx0 = ry0 / flat
        n = max(2, int(round(wid / (1.3 * rx0))))
        xs = np.linspace(ax(s) - l_ * Hc + 0.8 * rx0, ax(s) + r_ * Hc - 0.8 * rx0, n)
        order = rng.permutation(n)
        for j in order:
            sc = rng.uniform(0.8, 1.2)
            rx, ry = rx0 * sc, ry0 * sc * rng.uniform(0.9, 1.08)
            x = xs[j] + rng.uniform(-0.25, 0.25) * rx0
            yc = base_y - s * Hc + rng.uniform(-0.025, 0.025) * Hc
            if j in (0, n - 1) and 0.1 < s < 0.84:
                # outer masses bulge out by varied amounts: a stepped, lumpy silhouette, never a ruler edge
                x += (-1 if j == 0 else 1) * rng.uniform(-0.02, 0.06) * Hc
                yc += rng.uniform(-0.03, 0.03) * Hc
            lobes.append(Lobe(x, yc, rx, ry, 0, (ti, 0, key), jit=rng.uniform(-1, 1), rot=rng.uniform(-12, 12)))
            key += 1
    # anvil: the crown spreads into thin flattened tips (a lens, not a block), overhanging downwind
    for xo, s_, rxk, ryk in ((-0.38, 0.925, 0.13, 0.045), (0.4, 0.915, 0.12, 0.042),
                             (-0.22, 0.965, 0.12, 0.05), (-0.05, 1.01, 0.13, 0.06), (0.1, 1.03, 0.1, 0.055), (0.22, 0.99, 0.12, 0.05),
                             (0.36, 0.95, 0.1, 0.04)):
        lobes.append(Lobe(ax(s_) + xo * Hc, base_y - s_ * Hc + ryk * Hc, rxk * Hc, ryk * Hc, 0, (len(tiers), 0, key),
                          jit=rng.uniform(-1, 1), rot=rng.uniform(-4, 4)))
        key += 1
    bigs = list(lobes)
    # ---------------- medium heads on the upper rims of the big masses (interior lobe definition)
    for b in bigs:
        s = (base_y - b.y) / Hc
        n = int(rng.integers(2, 5))
        for k in range(n):
            th = math.radians(rng.uniform(-165, -15))
            rr = rng.uniform(0.035, 0.06) * Hc * (0.8 if s > 0.9 else 1.0)
            rim = rng.uniform(0.45, 0.8)
            x = b.x + math.cos(th) * b.rx * rim
            y = b.y + math.sin(th) * b.ry * rim + 0.4 * rr
            lobes.append(Lobe(x, y, rr * rng.uniform(1.0, 1.25), rr, 1, (b.key[0], b.key[1] + 1 + k, b.key[2]),
                              jit=rng.uniform(-1, 1), rot=rng.uniform(-20, 20)))
    # ---------------- medium heads bulging out of the silhouette (varied size; big on the shade side)
    x0, y0 = int(cx - 1.2 * Hc), int(base_y - 1.2 * Hc)
    w, h = int(2.4 * Hc), int(1.3 * Hc)
    q = 4
    m = _union(lobes, w, h, x0, y0, q)
    for (px, py, nx, ny) in _boundary(m, q, x0, y0, 0.05 * Hc, 0.03 * Hc):
        s = (base_y - py) / Hc
        if ny > 0.55 or s < 0.03:
            continue                                  # no bulges on the flat underside
        lit = nx * Ld[0] + ny * Ld[1]
        rr = rng.uniform(0.032, 0.055) * Hc * (rng.uniform(0.9, 1.6) if lit < -0.2 else 1.0)
        if s > 0.84:
            rr *= 0.55                                # anvil: thin flattened tips, no round knobs
        dd = rr * rng.uniform(0.35, 0.65)
        # tier key: attach to the big mass tier at this height so the lower tiers overlap it
        ti = int(np.clip(np.searchsorted(tiers, s), 0, len(tiers)))
        lobes.append(Lobe(px - nx * dd, py - ny * dd + 0.2 * rr, rr * rng.uniform(1.0, 1.2), rr, 1,
                          (ti, 50, int(rng.integers(1 << 20))), jit=rng.uniform(-1, 1), rot=rng.uniform(-20, 20)))
    # ---------------- small cauliflower only on the lit (sun-facing) silhouette
    m = _union(lobes, w, h, x0, y0, q)
    for (px, py, nx, ny) in _boundary(m, q, x0, y0, 0.018 * Hc, 0.012 * Hc):
        s = (base_y - py) / Hc
        lit = nx * Ld[0] + ny * Ld[1]
        if lit < 0.12 or s < 0.08 or ny > 0.3:
            continue
        if rng.random() < 0.25:
            continue
        rr = rng.uniform(0.01, 0.02) * Hc
        dd = rr * rng.uniform(0.2, 0.55)
        ti = int(np.clip(np.searchsorted(tiers, s), 0, len(tiers)))
        lobes.append(Lobe(px - nx * dd, py - ny * dd, rr * rng.uniform(1.0, 1.3), rr, 2,
                          (ti, 90, int(rng.integers(1 << 20))), jit=rng.uniform(-1, 1), rot=rng.uniform(-25, 25)))
    # small florets on the tops of the lit interior heads, upper half of the tower only
    meds = [lb for lb in lobes if lb.lvl == 1]
    for lb in meds:
        s = (base_y - lb.y) / Hc
        if s < 0.35 or rng.random() < 0.4:
            continue
        for k in range(int(rng.integers(1, 4))):
            th = math.radians(rng.uniform(-160, -70))
            rr = rng.uniform(0.009, 0.016) * Hc
            lobes.append(Lobe(lb.x + math.cos(th) * lb.rx * 0.85, lb.y + math.sin(th) * lb.ry * 0.85 + 0.3 * rr,
                              rr * 1.15, rr, 2, (lb.key[0], lb.key[1], lb.key[2] + 1), jit=rng.uniform(-1, 1)))
    # painting order: upper tiers first (behind); inside a tier big -> medium -> small
    lobes.sort(key=lambda lb: (lb.key[0], lb.lvl, lb.key[1], lb.y))
    return lobes, ax


def paint(pw, ph, cx, base_y, Hc, W, seed=4, Ld=(-0.55, -0.83), dbg=None):
    rng = np.random.default_rng(seed)
    Ld = np.asarray(Ld, F32) / np.linalg.norm(Ld)
    lobes, ax = build_lobes(rng, cx, base_y, Hc, Ld)
    # working window
    x0 = max(int(cx - 0.95 * Hc), 0)
    x1 = min(int(cx + 1.0 * Hc), pw)
    y0 = max(int(base_y - 1.12 * Hc), 0)
    y1 = min(int(base_y + 0.06 * Hc), ph)
    ww, hh = x1 - x0, y1 - y0
    sc = W / 1920.0
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    # painterly domain warp: lobe edges wobble like a loaded brush, not a compass circle
    amp = 0.004 * Hc
    wx = _noise(ww, hh, 0.02 * Hc, seed + 1, 3) * amp + _noise(ww, hh, 0.006 * Hc, seed + 2, 2) * amp * 0.45
    wy = _noise(ww, hh, 0.02 * Hc, seed + 3, 3) * amp + _noise(ww, hh, 0.006 * Hc, seed + 4, 2) * amp * 0.45
    X = xs + wx
    Y = ys + wy
    # ---------------- coverage (silhouette) first: union of all lobes
    cov = np.zeros((hh, ww), F32)
    for lb in lobes:
        bx0, bx1 = int(max(lb.x - lb.rx * 1.3 - x0, 0)), int(min(lb.x + lb.rx * 1.3 - x0 + 1, ww))
        by0, by1 = int(max(lb.y - lb.ry * 1.3 - y0, 0)), int(min(lb.y + lb.ry * 1.3 - y0 + 1, hh))
        if bx1 <= bx0 or by1 <= by0:
            continue
        cr, sr = math.cos(math.radians(lb.rot)), math.sin(math.radians(lb.rot))
        dx = X[by0:by1, bx0:bx1] - lb.x
        dy = Y[by0:by1, bx0:bx1] - lb.y
        u = (dx * cr + dy * sr) / lb.rx
        v = (-dx * sr + dy * cr) / lb.ry
        q = np.sqrt(u * u + v * v)
        a = np.clip((1 - q) * min(lb.rx, lb.ry) + 0.5, 0, 1)
        np.maximum(cov[by0:by1, bx0:bx1], a, out=cov[by0:by1, bx0:bx1])
    # flat base: a near-level cut, gently stepped
    sh = (base_y - ys) / Hc
    bn = _noise(ww, 1, 0.12 * Hc, seed + 9, 3)[0][None, :]
    cut = base_y - Hc * (0.004 + 0.006 * bn)
    cov = cov * _ss(cut + 0.004 * Hc, cut - 0.006 * Hc, ys)
    mk = cov > 0.5
    mkf = mk.astype(F32)
    # ---------------- big form value G (smooth): dome of the silhouette lit from the upper left,
    # across-the-mass position (down-sun third in shade), crown brighter, base darker
    hgt = (cv2.GaussianBlur(mkf, (0, 0), 0.04 * Hc) * 0.4 + cv2.GaussianBlur(mkf, (0, 0), 0.1 * Hc) * 0.6)
    gy_, gx_ = np.gradient(hgt * 0.3 * Hc)
    nz = np.sqrt(gx_ ** 2 + gy_ ** 2 + 0.45 ** 2)
    dome = (-gx_ * Ld[0] - gy_ * Ld[1]) / nz                    # -1..1: faces the sun (+)
    has = mk.any(1)
    idx = np.nonzero(has)[0]
    xl = np.where(has, mk.argmax(1), 0).astype(F32)
    xr = np.where(has, ww - 1 - mk[:, ::-1].argmax(1), 0).astype(F32)
    xl = cv2.GaussianBlur(np.interp(np.arange(hh), idx, xl[idx]).astype(F32)[None, :], (0, 0), 0.05 * Hc)[0]
    xr = cv2.GaussianBlur(np.interp(np.arange(hh), idx, xr[idx]).astype(F32)[None, :], (0, 0), 0.05 * Hc)[0]
    u = ((np.arange(ww, dtype=F32)[None, :] - 0.5 * (xl + xr)[:, None]) / np.maximum(0.5 * (xr - xl), 1.0)[:, None])
    u = np.clip(u, -1.3, 1.3)
    gn = _noise(ww, hh, 0.12 * Hc, seed + 11, 3) * 0.06
    G = (0.42 - 0.45 * u + 0.5 * dome + 0.16 * (np.clip(sh, 0, 1) - 0.5) - 0.22 * _ss(0.14, 0.0, sh) + gn)
    G = G.astype(F32)
    # ---------------- paint: body first (value G minus a touch), then every lobe as a crisp lit cap
    # with a transparent (lost) underside, back to front
    G = np.where(G > 0.58, 0.58 + (G - 0.58) * 0.4, G).astype(F32)
    V = (G - 0.16).astype(F32)
    Wm = np.zeros((hh, ww), F32)
    for lb in lobes:
        pad = 1.25
        bx0, bx1 = int(max(lb.x - lb.rx * pad - x0, 0)), int(min(lb.x + lb.rx * pad - x0 + 1, ww))
        by0, by1 = int(max(lb.y - lb.ry * pad - y0, 0)), int(min(lb.y + lb.ry * pad - y0 + 1, hh))
        if bx1 <= bx0 or by1 <= by0:
            continue
        sl = (slice(by0, by1), slice(bx0, bx1))
        cr, sr = math.cos(math.radians(lb.rot)), math.sin(math.radians(lb.rot))
        dx = X[sl] - lb.x
        dy = Y[sl] - lb.y
        uu = (dx * cr + dy * sr) / lb.rx
        vv = (-dx * sr + dy * cr) / lb.ry
        q = np.sqrt(uu * uu + vv * vv)
        a = np.clip((1 - q) * min(lb.rx, lb.ry) + 0.5, 0, 1)
        # the lobe's own light: facing term toward the sun (upper left) + a crisp bright cap on the rim
        # Ld points TOWARD the sun (up-left): f > 0 on the sun-facing part of the lobe
        f = uu * Ld[0] + vv * Ld[1]
        cx_, cy_ = int(np.clip(lb.x - x0, 0, ww - 1)), int(np.clip(lb.y - y0, 0, hh - 1))
        s_l = (base_y - lb.y) / Hc
        if lb.lvl < 2 and s_l < 0.42 and G[cy_, cx_] < 0.4 and (lb.jit > -0.2):
            lb.warm = 0.6 + 0.4 * abs(lb.jit)          # reflected warm light from the sunlit haze / town
        ks = (0.26, 0.26, 0.2)[lb.lvl]
        Vl = 0.55 * G[sl] + 0.45 * G[cy_, cx_] + 0.03 * lb.jit + 1.2 * ks * f - 0.04
        Vl = Vl + 0.12 * _ss(0.35, 0.85, f) * _ss(0.55, 0.95, q)        # crisp bright cap along the lit rim
        # lost underside: the lower / down-sun part of the lobe is transparent (fades into the body)
        up = -vv * 0.85 + f * 0.35
        fade = _ss(-0.3, 0.45, up)
        la = a * fade
        V[sl] = V[sl] * (1 - la) + Vl * la
        if lb.warm:
            Wm[sl] = Wm[sl] * (1 - la) + lb.warm * la
    # ---------------- warm reflected light: low shaded lobes pick up bounce from the sunlit haze/town
    Wm = Wm * _ss(0.46, 0.3, V)                     # only where the lobe sits in shade
    # ---------------- value -> painted planes: 3-4 soft steps in the shade, cream lit planes
    wn = _noise(ww, hh, 0.05 * Hc, seed + 5, 3) * 0.025 + _noise(ww, hh, 0.012 * Hc, seed + 6, 2, stretch=3, angle=-25) * 0.012
    Vw = V + wn
    steps = [(0.0, (0.57, 0.63, 0.75)), (0.14, (0.615, 0.675, 0.79)), (0.26, (0.665, 0.72, 0.82)),
             (0.36, (0.72, 0.765, 0.85)), (0.44, (0.8, 0.83, 0.89)), (0.52, (0.88, 0.89, 0.92)),
             (0.62, (0.955, 0.945, 0.915)), (0.76, (0.998, 0.975, 0.925)), (0.9, (1.035, 1.01, 0.94))]
    # gentle stepping: part continuous, part snapped to the nearest plane (soft brush-warped borders)
    lv = np.array([s[0] for s in steps], F32)
    Vs = np.zeros_like(Vw)
    for i in range(1, len(lv)):
        mid = 0.5 * (lv[i - 1] + lv[i])
        Vs = Vs + (lv[i] - lv[i - 1]) * _ss(mid - 0.02, mid + 0.02, Vw)
    Vs = Vs + lv[0]
    Vf = np.clip(0.45 * Vw + 0.55 * Vs, 0, 1)
    cs = np.array([s[1] for s in steps], F32)
    col = np.stack([np.interp(Vf, lv, cs[:, k]) for k in range(3)], -1).astype(F32)
    col = col + (_c((0.82, 0.78, 0.8)) - col) * (0.38 * np.clip(Wm, 0, 1))[..., None]
    # flat base band: darker blue-grey underside, continuous with the shade, hazing toward the horizon
    bb = _ss(0.075, 0.01, sh + 0.01 * bn)
    col = col + (_c((0.6, 0.65, 0.76)) - col) * (0.65 * bb)[..., None]
    # aerial perspective: the lower tower sits deeper in the summer haze
    hz = _ss(0.45, 0.0, sh) * 0.22
    col = col + (_c((0.8, 0.87, 0.96)) - col) * hz[..., None]
    # subtle paint grain / dry-brush strokes
    st = _noise(ww, hh, 0.012 * Hc, seed + 41, 2, stretch=5.0, angle=-35)
    col = col * (1 + 0.01 * st[..., None]) * (1 + 0.012 * _noise(ww, hh, 0.2 * Hc, seed + 42, 3)[..., None] *
                                              _c((1.0, 0.4, -0.6)))
    # ---------------- alpha: crisp on the lit silhouette, lost + soft on the shade side / base
    litness = _ss(0.36, 0.56, cv2.GaussianBlur(V, (0, 0), 0.01 * Hc))
    Asoft = cv2.GaussianBlur(cov, (0, 0), 0.012 * Hc)
    Asoft = np.minimum(np.clip(Asoft * 1.25, 0, 1), cv2.dilate(cov, np.ones((3, 3), np.uint8)))
    Asoft = np.maximum(Asoft * 0.92, cov * _ss(0.45, 0.9, Asoft))
    litx = cv2.dilate(litness, np.ones((9, 9), np.uint8))
    A = cov * litx + Asoft * (1 - litx)
    # base: soft lower edge sinking into the haze
    A = A * (1 - 0.55 * _ss(0.05, -0.005, sh + 0.006 * bn))
    # ---------------- cirrus shear off the up-sun side of the anvil (fibrous, fanning left)
    n1 = _noise(ww, hh, 0.06 * Hc, seed + 81, 4, stretch=6.0, angle=-6)
    n2 = _noise(ww, hh, 0.015 * Hc, seed + 82, 3, stretch=8.0, angle=-6)
    fib = _ss(0.05, 0.6, 0.65 * n1 + 0.35 * n2)
    outd = cv2.distanceTransform((1 - mk).astype(np.uint8), cv2.DIST_L2, 5).astype(F32)
    band = np.exp(-outd / (0.1 * Hc)) * (outd > 0)
    xrel = (xs - ax(0.95)) / Hc
    where = _ss(0.8, 0.9, sh) * _ss(1.1, 0.98, sh) * _ss(-0.1, -0.3, xrel)
    wa = np.clip(band * fib * where * 0.8, 0, 1)
    wcol = np.broadcast_to(_c((0.97, 0.97, 0.98)), col.shape)
    Ao = A + wa * (1 - A)
    col = (col * A[..., None] + wcol * (wa * (1 - A))[..., None]) / np.maximum(Ao, 1e-4)[..., None]
    A = Ao
    out = np.zeros((ph, pw, 4), F32)
    out[y0:y1, x0:x1, :3] = col
    out[y0:y1, x0:x1, 3] = A
    out[..., :3] = K3._bleed(out[..., :3], (out[..., 3] > 0.02).astype(F32), max(3.0, 0.004 * W))
    if dbg is not None:
        dbg.update(V=V, G=G, cov=cov, box=(x0, y0, x1, y1))
    return out


def rim_light(P, Hc, sun, W, reach=0.35):
    """Backlit crown: the silhouette nearest the sun blows out into a thin warm rim + glow."""
    h, w = P.shape[:2]
    sc = W / 1920.0
    r = int(reach * 2.2 * Hc)
    x0, x1 = max(int(sun[0]) - r, 0), min(int(sun[0]) + r, w)
    y0, y1 = max(int(sun[1]) - r, 0), min(int(sun[1]) + r, h)
    c = P[y0:y1, x0:x1, :3]
    a = P[y0:y1, x0:x1, 3]
    ys = np.arange(y0, y1, dtype=F32)[:, None]
    xs = np.arange(x0, x1, dtype=F32)[None, :]
    d = np.hypot(xs - sun[0], ys - sun[1]) / Hc
    near = np.exp(-(d / reach) ** 2)
    m = (a > 0.5).astype(np.uint8)
    ins = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(F32)
    rw = (1.5 + 5.0 * near) * sc + 0.7
    band = (1 - _ss(rw * 0.5, rw * 1.3, ins)) * (m > 0)
    k = np.clip(band * near, 0, 1)
    c += (_c((1.7, 1.58, 1.3)) - c) * k[..., None]
    c += _c((0.18, 0.15, 0.08)) * (np.exp(-ins / (0.02 * Hc)) * near * (m > 0))[..., None]
    P[y0:y1, x0:x1, :3] = c
    return P


def hero(pw, ph, cx, base_y, Hc, W, hz_y=None, seed=4, dbg=None):
    """Returns (rgba plate (ph, pw, 4), sun position in plate px). The sun sits just behind the crown's
    up-sun shoulder."""
    P = paint(pw, ph, cx, base_y, Hc, W, seed=seed, dbg=dbg)
    a_ = P[..., 3] > 0.5
    rows = np.nonzero(a_.any(1))[0]
    ytop = rows[0]
    xs_top = np.nonzero(a_[ytop + int(0.012 * Hc)])[0]
    sun = np.array([xs_top.mean() - 0.1 * Hc, ytop + 0.035 * Hc], np.float32)
    # make sure the sun sits just inside the silhouette (behind the cloud edge)
    col = a_[:, int(sun[0])]
    yy = np.nonzero(col)[0]
    if len(yy):
        sun[1] = yy[0] + 0.02 * Hc
    P = rim_light(P, Hc, sun, W)
    return P, sun


if __name__ == '__main__':
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib import core as C, sky as S
    W, H = (1920, 1080) if 'full' in sys.argv else (960, 540)
    pw, ph = int(W * 1.12), int(H * 1.26)
    mg = 0.03 * H
    base_y = mg + 0.845 * H
    top_y = mg + 0.15 * H
    Hc = (base_y - top_y) / 1.01
    t0 = time.time()
    dbg = {}
    P, sun = hero(pw, ph, (pw - W) / 2 + 0.38 * W, base_y, Hc, W, seed=4, dbg=dbg)
    print('hero', time.time() - t0, sun)
    sky = S.sky_gradient(pw, ph, dict(stops=[(0.0, '#0a45b8'), (0.16, '#155dcd'), (0.34, '#2e80de'), (0.5, '#58a2e9'),
                                             (0.64, '#8ec4f0'), (0.78, '#b8dcf6'), (0.9, '#d6ecf9'), (1.0, '#ebf8fc')],
                                      sun_glow='#fff6e0', sun_glow_amt=0.3, below='#eef9fc'), horizon=0.8)[..., :3]
    img = sky * (1 - P[..., 3:4]) + P[..., :3] * P[..., 3:4]
    ox = (pw - W) // 2
    img = img[int(mg):int(mg) + H, ox:ox + W]
    tag = [a for a in sys.argv[1:] if a != 'full']
    C.save_png(os.path.join('out', 'compare', 'p6_%s.png' % (tag[0] if tag else 'a')), np.clip(img, 0, 1))
