"""s10 round 14 - painted sea of clouds + framing towers (value-plane painting on lib.clouds2 silhouettes).

Every cloud mass is ONE cauliflower silhouette grown by lib.clouds2 (edge-only florets, smooth concave
gaps) and painted as a few large value planes from ONE coherent form (the whole cloud's ellipsoid blended
with the mass's own smoothed dome, florets never shaded individually):
  warm lit top plane (hot gold-white toward the sun, peach, rose terminator band) with a firm scalloped
  terminator -> mid lavender -> cool blue shadow collecting on the underside,
  aerial perspective (value + saturation compress toward a pale peach horizon),
  a torn, wispy lower edge that dissolves into a hazy lower deck (no dark voids),
  a continuous 2-4 px gold rim ONLY on sun-facing lit crests (hot on the sun axis, thinner and
  silver-pink away from it, none on crests turned away).
Plates are dicts {rgba (straight alpha), ox, oy} in frame px (see s10_sea_of_clouds_r11.place_fast).
"""
import math
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib import core as C, clouds2 as K  # noqa: E402

_ss = K._ss


def _c(h):
    return np.asarray(K._c(h), np.float32)


def _mix(a, b, t):
    return a * (1 - t) + b * t


def _blur(img, s):
    if s < 0.3:
        return img
    return cv2.GaussianBlur(img, (0, 0), s)


# ----------------------------------------------------------------------------------------- palettes
# keys: hot lit lit_lo term mid shade deep
SEA_SUN = dict(hot=(1.0, 0.95, 0.84), lit=(1.0, 0.82, 0.64), lit_lo=(0.95, 0.66, 0.64), term=(0.9, 0.58, 0.68),
               mid=(0.72, 0.64, 0.84), shade=(0.54, 0.54, 0.8), deep=(0.44, 0.45, 0.74))
SEA_OFF = dict(hot=(0.99, 0.92, 0.9), lit=(0.95, 0.82, 0.84), lit_lo=(0.85, 0.72, 0.84), term=(0.78, 0.62, 0.82),
               mid=(0.64, 0.62, 0.84), shade=(0.5, 0.52, 0.8), deep=(0.4, 0.43, 0.72))
SEA_NEAR = dict(hot=(1.0, 0.9, 0.78), lit=(0.96, 0.78, 0.7), lit_lo=(0.86, 0.68, 0.78), term=(0.78, 0.56, 0.74),
                mid=(0.47, 0.47, 0.8), shade=(0.32, 0.35, 0.7), deep=(0.24, 0.27, 0.58))
TOWER = dict(hot=(1.0, 0.97, 0.91), lit=(1.0, 0.91, 0.8), lit_lo=(0.97, 0.82, 0.76), term=(0.93, 0.68, 0.72),
             mid=(0.72, 0.68, 0.86), shade=(0.56, 0.58, 0.84), deep=(0.45, 0.49, 0.78))
PAL_KEYS = ('hot', 'lit', 'lit_lo', 'term', 'mid', 'shade', 'deep')

HAZE_SUN = np.asarray((1.0, 0.87, 0.72), np.float32)
HAZE_OFF = np.asarray((0.9, 0.82, 0.88), np.float32)
RIM_SUN = np.asarray((1.55, 1.24, 0.82), np.float32)
RIM_OFF = np.asarray((1.08, 1.0, 1.04), np.float32)


def pal(d):
    return {k: np.asarray(v, np.float32) for k, v in d.items()}


def mixpal(a, b, t):
    return {k: (a[k] * (1 - t) + b[k] * t).astype(np.float32) for k in PAL_KEYS}


def haze_color(x, W, sx):
    p = math.exp(-((x - sx) / (0.33 * W)) ** 2)
    return (HAZE_OFF * (1 - p) + HAZE_SUN * p).astype(np.float32)


def floor_color(dy, xs=None, W=1920, sx=960):
    """Hazy lower deck seen through the gaps (never a dark void): pale gold at the horizon -> soft
    lavender -> a calm mid blue-violet near the camera; a touch warmer on the sun axis."""
    k = np.clip(dy, 0, None)
    far = HAZE_SUN
    mid = _c('#b3aad2')
    near = _c('#6f76b8')
    a = _ss(0.0, 0.06, k)[..., None]
    b = _ss(0.04, 0.45, k)[..., None]
    col = (far * (1 - a) + mid * a) * (1 - b) + near * b
    if xs is not None:
        sp = np.exp(-((xs - sx) / (0.3 * W)) ** 2)[..., None]
        col = col + sp * (1 - b) * 0.08 * np.array([1.0, 0.5, -0.2], np.float32)
    return col.astype(np.float32)


# ----------------------------------------------------------------------------------------- canvas
class Canvas:
    """Premultiplied RGBA plate at frame-px origin (ox, oy) + shared paint-texture noise."""

    def __init__(self, w, h, ox, oy, u, seed):
        self.w, self.h, self.ox, self.oy, self.u = int(w), int(h), float(ox), float(oy), u
        self.prem = np.zeros((self.h, self.w, 4), np.float32)
        rng = np.random.default_rng(seed)
        self.seed = int(rng.integers(1 << 30))
        s = self.seed
        self.n_lo = K._noise(self.w, self.h, max(self.w / (420 * u), 2), s + 1, 3) - 0.5
        self.n_md = K._noise(self.w, self.h, max(self.w / (120 * u), 2), s + 2, 3) - 0.5
        self.n_st = K._noise(self.w, self.h, max(self.w / (90 * u), 2), s + 3, 3, stretch=4.0) - 0.5
        self.n_fn = K._noise(self.w, self.h, max(self.w / (30 * u), 2), s + 4, 2, stretch=2.5) - 0.5

    def sl(self, x0, y0, w, h):
        """Clip a local box (plate px) to the canvas: returns (plate slice, local slice) or None."""
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, self.w), min(y0 + h, self.h)
        if X1 <= X0 or Y1 <= Y0:
            return None
        return (slice(Y0, Y1), slice(X0, X1)), (slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0))

    def over(self, col, a, x0, y0):
        r = self.sl(x0, y0, a.shape[1], a.shape[0])
        if r is None:
            return
        (py, px), (ly, lx) = r
        aa = a[ly, lx][..., None]
        d = self.prem[py, px]
        d[..., :3] = d[..., :3] * (1 - aa) + col[ly, lx] * aa
        d[..., 3:4] = d[..., 3:4] * (1 - aa) + aa

    def plate(self):
        a = self.prem[..., 3:4]
        rgb = self.prem[..., :3] / np.maximum(a, 1e-5)
        pb = cv2.GaussianBlur(self.prem, (0, 0), max(2.5 * self.u, 1.0))
        fill = pb[..., :3] / np.maximum(pb[..., 3:4], 1e-5)
        wgt = np.clip(a * 6.0, 0, 1)
        rgb = rgb * wgt + fill * (1 - wgt)
        rgba = np.concatenate([rgb, np.clip(a, 0, 1)], -1).astype(np.float32)
        return dict(rgba=np.ascontiguousarray(rgba), ox=self.ox, oy=self.oy)


# ----------------------------------------------------------------------------------------- one mass
TERM_LV = ((0.07, 0.16, 0.9, 1.2, 2.4, 0.2, 0.55), (0.03, 0.06, 0.45, 1.5, 3.2))


def paint_mass(cv, m, x0, y0, L, P, size, group=None, rng=None, dome=0.12, dv=0.0, t_term=0.12, t_hot=0.6,
               term_px=None, scallop=1.0, wander=0.07, under=0.35, haze=0.0, hazec=None, base=None, tear=0.0,
               rim=0.0, rim_px=2.5, rimc=None, rim_dir=None, alpha=1.0, glow=0.0, sky=0.35, refl=0.0, strip=None, hot_w=0.03, step=0.55, lit_span=0.35):
    """Paint one cauliflower mask m (local, float 0..1, top-left at plate px x0, y0) onto Canvas cv.

    Light comes from ONE big form: the whole cloud's ellipsoid `group` (cx, cy, rx, ry) (plate px), with a
    small, heavily smoothed contribution (`dome`) of the mass's own shape - florets never get their own
    shading. The lit plane is a region of that form grown into the shadow with cauliflower scallops
    (lib.clouds2 growth, lit heads bulging away from the sun): a FIRM painted terminator, a rose band
    just inside it, peach -> warm gold -> hot plane toward the sun; the shadow is mid lavender -> cool
    blue, deepest on the underside; up-facing shadowed parts catch a firm skylight step.
    L - 3D light direction (screen x right, y down, z toward the viewer); dv - value offset of this mass
    (overlapping masses of one cloud differ slightly -> crisp overlap edges); base - (base_y, depth) torn
    wispy lower edge; rim - continuous gold rim on the sun-facing LIT contour only (rim_dir = 2D screen
    direction toward the sun)."""
    u = cv.u
    h, w = m.shape
    if h < 3 or w < 3:
        return
    r = cv.sl(x0, y0, w, h)
    if r is None:
        return
    rng = np.random.default_rng(0) if rng is None else rng
    xs = (np.arange(w, dtype=np.float32) + x0)[None, :]
    ys = (np.arange(h, dtype=np.float32) + y0)[:, None]
    ins = (m > 0.5).astype(np.uint8)
    if group is None:
        group = (x0 + w / 2, y0 + h * 0.65, w * 0.55, h * 0.65)
    (py, px), (ly, lx) = r
    nlo = np.zeros((h, w), np.float32)
    nmd = np.zeros((h, w), np.float32)
    nst = np.zeros((h, w), np.float32)
    nfn = np.zeros((h, w), np.float32)
    nlo[ly, lx], nmd[ly, lx], nst[ly, lx], nfn[ly, lx] = cv.n_lo[py, px], cv.n_md[py, px], cv.n_st[py, px], cv.n_fn[py, px]
    if strip is not None:
        # sea-of-clouds billow strip: the form is read from the strip's own (head-scale smoothed) top
        # profile: t = 0 at the top of each head .. 1 at the strip's base line; lit where the head top
        # faces the light (heads leaning toward the sun lit wider, flanks turned away narrower)
        basel, hs = strip
        colm = ins.any(0)
        top = np.where(colm, ins.argmax(0), h).astype(np.float32)
        top = np.minimum(top, basel)
        tsig = max(hs, 1.0)
        top_s = cv2.GaussianBlur(top[None, :], (0, 0), sigmaX=tsig, sigmaY=0.1)[0]
        top_s = np.minimum(top_s, cv2.GaussianBlur(top[None, :], (0, 0), sigmaX=0.35 * tsig, sigmaY=0.1)[0])
        hh = np.maximum(basel - top_s, 0.25 * size)
        t = (np.arange(h, dtype=np.float32)[:, None] - top_s[None, :]) / hh[None, :]
        g = np.gradient(top_s)
        l2 = np.array([L[0], L[1]], np.float32)
        l2 = l2 / (np.linalg.norm(l2) + 1e-6)
        lamb = (g * l2[0] - l2[1]) / np.sqrt(1 + g * g)
        s = 0.55 * lamb[None, :] - t + dv + wander * (2 * nlo + 0.5 * nmd)
        ny = np.clip(1.6 * t - 0.6, -1, 1).astype(np.float32)
    else:
        gcx, gcy, grx, gry = group
        ex = (xs - gcx) / grx
        ey = (ys - gcy) / gry
        ez = np.sqrt(np.clip(1 - ex * ex - ey * ey, 0.02, 1))
        ex, ey = ex + 0 * ey, ey + 0 * ex
        if dome > 0:
            d = cv2.distanceTransform(ins, cv2.DIST_L2, 5)
            R = max(0.35 * size, 2.0)
            q = np.clip(d / R, 0, 1)
            dm = _blur(np.sqrt(q * (2 - q)).astype(np.float32), max(0.1 * size, 1.0))
            gy, gx = np.gradient(dm * R * 0.8)
            ex = ex - dome * gx
            ey = ey - dome * gy
        ln = np.sqrt(ex * ex + ey * ey + ez * ez) + 1e-6
        nx, ny, nz = ex / ln, ey / ln, ez / ln
        s = nx * L[0] + ny * L[1] + nz * L[2] + dv + wander * (2 * nlo + 0.5 * nmd)
    # --- lit plane: firm terminator grown into the shadow with cauliflower scallops
    ss_ = 2 if size < 260 * u else 1
    lit0 = (s > t_term).astype(np.uint8) * 255
    lit0 = lit0 * ins
    if ss_ > 1:
        lit0 = cv2.resize(lit0, (w * ss_, h * ss_), interpolation=cv2.INTER_NEAREST)
    if scallop > 0 and lit0.max() > 0 and lit0.min() == 0:
        rd = rim_dir if rim_dir is not None else (L[0], L[1])
        away = (-float(rd[0]), -float(rd[1]))
        nrm = math.hypot(*away) + 1e-6
        K._grow(lit0, rng, size * ss_ * scallop, TERM_LV, (away[0] / nrm, away[1] / nrm), 0.45, 0.6, (0.3, 0.6),
                1.0, 0.5, None, 0.0, 0.6, fill=False)
    litm = lit0.astype(np.float32) / 255.0
    if ss_ > 1:
        litm = cv2.resize(litm, (w, h), interpolation=cv2.INTER_AREA)
    # firm painted terminator that softens in a few passages
    soft_k = _ss(0.05, 0.25, nlo + 0.3 * nmd)
    lh = _blur(litm, 0.6 * u)
    ls = _blur(litm, max(0.03 * size, 1.5 * u))
    litm = (lh * (1 - soft_k) + ls * soft_k) * np.clip(m * 1.5, 0, 1) * _ss(t_term - 0.45, t_term - 0.2, s)
    lit_t = litm[..., None]
    lt = _ss(t_term - 0.05, t_term + lit_span, s + 0.04 * nmd)[..., None]
    litc = _mix(P['lit_lo'], P['lit'], lt)
    hot_t = _ss(t_hot - hot_w, t_hot + hot_w, s + 0.03 * nmd)[..., None]
    litc = _mix(litc, P['hot'], hot_t)
    # rose band just inside the terminator (follows the scallops)
    tp = (term_px if term_px is not None else 0.05 * size)
    shd = ((litm < 0.5) & (ins > 0)).astype(np.uint8)       # shadow inside the mass (not the outside)
    dt = cv2.distanceTransform(1 - shd, cv2.DIST_L2, 3) if shd.any() else np.full_like(litm, 1e4)
    tb = (1 - _ss(0.3 * tp, tp, dt))[..., None] * (1 - hot_t) * _ss(0.2, 0.6, m)[..., None]
    litc = _mix(litc, P['term'], 0.8 * tb)
    # --- shadow plane
    sh = (step * _ss(t_term - 0.3, t_term - 0.34, s + 0.04 * nmd) + (1 - step) * _ss(t_term - 0.05, t_term - 0.6, s))[..., None]
    shc = _mix(P['mid'], P['shade'], sh)
    dn = _ss(0.1, 0.75, ny)[..., None]
    shc = _mix(shc, P['deep'], np.clip(under * dn + 0.3 * _ss(-0.3, -0.8, s)[..., None], 0, 0.85))
    if sky:
        upk = _ss(-0.4, -0.65, ny + 0.12 * nmd)[..., None]
        shc = _mix(shc, P['mid'], sky * upk)
    if refl:
        shc = shc + refl * dn * np.array([0.05, 0.03, 0.0], np.float32)
    col = _mix(shc, litc, lit_t)
    col = col * (1 + 0.022 * nmd[..., None] + 0.01 * nfn[..., None])
    a = np.clip(m, 0, 1)
    if base is not None:
        by, dep = base
        v = (ys - by) / max(dep, 1.0)
        cut = v + tear * (1.3 * nst + 0.5 * nmd + 0.35 * nfn)
        keep = 0.78 * _ss(0.25, -0.2, cut) + 0.22 * _ss(1.0, -0.5, cut)
        a = a * keep
        bh = _ss(-1.6, 0.4, v)[..., None]
        hz2 = hazec if hazec is not None else P['mid']
        col = _mix(col, _mix(P['shade'], hz2, 0.55), 0.4 * bh * (1 - lit_t))
    if haze:
        hc = hazec if hazec is not None else HAZE_OFF
        col = _mix(col, hc, haze)
    # --- rim: continuous line along the sun-facing LIT contour only
    if rim > 0:
        rd = rim_dir if rim_dir is not None else (0.0, -1.0)
        rr = rim_px * u
        gxs, gys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        mm = np.clip(m, 0, 1).astype(np.float32)
        sh1 = cv2.remap(mm, gxs + rd[0] * rr, gys + rd[1] * rr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        sh2 = cv2.remap(mm, gxs + rd[0] * 2.4 * rr, gys + rd[1] * 2.4 * rr, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        core = mm * (1 - sh1)
        soft = mm * (1 - _blur(sh2, 0.8 * rr))
        gate = _blur(litm, 1.5 * rr)
        k = np.clip((1.2 * core + 0.12 * soft) * gate * rim, 0, 1)[..., None]
        rc = rimc if rimc is not None else RIM_SUN
        col = _mix(col, rc, k)
        if glow:
            gk = _blur(k[..., 0], 3 * rr)[..., None] * glow
            col = col + rc * 0.12 * gk
    cv.over(col.astype(np.float32), (a * alpha).astype(np.float32), x0, y0)


def mask_from_poly(poly, rng, size, levels, **kw):
    return K.cauliflower(poly, rng, size, levels, **kw)


# ----------------------------------------------------------------------------------------- sea
LV_FAR = ((0.06, 0.14, 0.85, 1.1, 2.0), (0.025, 0.05, 0.6))
LV_MID = ((0.1, 0.22, 0.85, 1.3, 2.6, 0.1, 0.45), (0.035, 0.075, 0.65, 1.3, 2.8), (0.014, 0.026, 0.35, 1.5, 3.5))
LV_NEAR = ((0.12, 0.26, 0.9, 1.3, 2.6, 0.1, 0.45), (0.04, 0.085, 0.75, 1.2, 2.6), (0.016, 0.03, 0.45, 1.4, 3.2),
           (0.008, 0.014, 0.25, 1.6, 3.6))


def sea_rows(W, H, hy, seed=7, rows=17, persp=1.85, bottom=1.12):
    """Row layout: [(f, y_base, bump_w, bump_h)] from the horizon (f=0) to below the frame (f=1)."""
    out = []
    for i in range(rows):
        f = i / (rows - 1)
        z = f ** persp
        y = hy + 0.006 * H + (bottom * H - hy) * z
        bw = (0.018 + (0.3 - 0.018) * z ** 0.9) * W
        asp = 0.2 + 0.2 * f
        out.append((f, y, bw, bw * asp))
    return out


def paint_row(cv, W, H, hy, sun, f, y, bw, bh, rng, u):
    """One row of the sea: overlapping billow strips painted back to front (plate px = frame px - origin)."""
    sx, sy = sun
    ox, oy = cv.ox, cv.oy
    lv = LV_FAR if f < 0.25 else (LV_MID if f < 0.6 else LV_NEAR)
    near = _ss(0.45, 1.0, f)
    x = -0.12 * W - bw * rng.uniform(0.3, 1.0)
    segs = []
    while x < 1.12 * W:
        Ls = bw * rng.uniform(1.6, 3.6) * (1 + 0.6 * (1 - f))
        sc = rng.uniform(1.25, 1.6) if (f > 0.55 and rng.random() < 0.3) else 1.0
        segs.append((x, x + Ls * sc, y + rng.uniform(-0.35, 0.3) * bh, sc, rng.uniform(0.75, 1.25)))
        x += Ls * sc * rng.uniform(0.45, 0.8)
    if f >= DOME_F:
        return paint_row_domes(cv, W, H, hy, sun, f, y, bw, bh, rng, u)
    rng.shuffle(segs)
    for (a0, a1, by, sc, var) in segs:
        cx = 0.5 * (a0 + a1)
        prox = math.exp(-((cx - sx) / (0.24 * W)) ** 2)          # on the sun axis
        proxw = math.exp(-((cx - sx) / (0.5 * W)) ** 2)
        hgt = bh * sc * var
        poly = K.billow_strip(a0 - ox, a1 - ox, by - oy, bw * sc ** 0.7, hgt, rng, depth=hgt * 1.1,
                              var=0.6 + 0.8 * f)
        if poly[:, 1].min() > cv.h + 4 or poly[:, 1].max() < -4:
            continue
        size = hgt * 1.4
        m, mx0, my0 = mask_from_poly(poly, rng, size, lv, down_cut=0.1, side_scale=0.7, sun_dir=(0.0, -1.0),
                                     sun_bias=0.2, concave=0.6, clump=0.45)
        if m.size == 0 or m.max() <= 0:
            continue
        # palette: warm on the sun axis, cooler toward the frame edges, deeper & cooler near the camera
        Pw = mixpal(pal(SEA_OFF), pal(SEA_SUN), prox ** 0.8)
        Pm = mixpal(Pw, pal(SEA_NEAR), 0.75 * near * (1 - 0.4 * prox))
        hz = 0.82 * (1 - _ss(0.0, 0.62, f)) ** 1.5
        hc = haze_color(cx, W, sx)
        # light: low sun ahead -> rakes up over the tops, leaning toward the sun's x
        lean = 0.45 * float(np.clip((sx - cx) / (0.5 * W), -1, 1))
        Lv = np.array([lean, -0.75, -0.55 + 0.15 * prox], np.float32)
        Lv = Lv / np.linalg.norm(Lv)
        t_term = 0.27 - 0.17 * prox + 0.2 * near * (1 - 0.5 * prox) + rng.uniform(-0.05, 0.05)
        rim_k = (0.35 + 0.65 * prox) * (1 - 0.5 * near) if hgt > 6 * u else 0.0
        rim_px = (1.5 + 1.6 * prox) * (0.8 + 0.4 * min(f * 2, 1))
        rimc = _mix(RIM_OFF, RIM_SUN, prox)
        rd = np.array([0.5 * lean, -1.0], np.float32)
        rd = rd / np.linalg.norm(rd)
        paint_mass(cv, m, mx0, my0, Lv, Pm, size, group=(cx - ox, by - oy + 0.3 * hgt, (a1 - a0) * 0.6, hgt * 1.3),
                   rng=rng, dv=rng.uniform(-0.05, 0.05), t_term=t_term, t_hot=0.46 + 0.3 * (1 - prox), sky=0.0,
                   strip=(by - oy + 0.3 * hgt - my0, 0.3 * bw * sc ** 0.7),
                   scallop=0.8 + 0.3 * near, wander=0.06, under=0.45 + 0.2 * near, haze=hz, hazec=hc,
                   base=(by - oy + 0.25 * hgt, 0.55 * hgt), tear=0.55 + 0.25 * near, rim=rim_k, rim_px=rim_px,
                   rimc=rimc, rim_dir=rd, glow=0.6 * prox)


DOME_F = 0.3


def paint_row_domes(cv, W, H, hy, sun, f, y, bw, bh, rng, u):
    """Mid / near rows: individual cumulus tops (a main dome + 1-2 lower heads in front sharing its form),
    each lit on its sunward top with a scalloped terminator, lavender faces, cool undersides, torn base."""
    sx, sy = sun
    ox, oy = cv.ox, cv.oy
    lv = LV_MID if f < 0.6 else LV_NEAR
    near = _ss(0.45, 1.0, f)
    cl = []
    x = -0.1 * W - bw * rng.uniform(0.0, 0.8)
    while x < 1.1 * W:
        wc = bw * float(np.clip(math.exp(rng.normal(0.1, 0.38)), 0.5, 2.2 - 0.8 * near))
        if rng.random() > 0.12 * f:
            cl.append((x, y + rng.uniform(-0.45, 0.45) * bh, wc, bh * rng.uniform(0.6, 1.35) * (wc / bw) ** 0.6))
        x += wc * rng.uniform(0.45, 1.0)
    rng.shuffle(cl)
    for (cx, by, wc, hc) in cl:
        prox = math.exp(-((cx - sx) / (0.24 * W)) ** 2)
        Pw = mixpal(pal(SEA_OFF), pal(SEA_SUN), prox ** 0.8)
        Pm = mixpal(Pw, pal(SEA_NEAR), 0.75 * near * (1 - 0.4 * prox))
        hz = 0.82 * (1 - _ss(0.0, 0.62, f)) ** 1.5
        hc_ = haze_color(cx, W, sx)
        lean = 0.5 * float(np.clip((sx - cx) / (0.5 * W), -1, 1))
        Lv = np.array([lean, -0.8, -0.3 + 0.15 * prox], np.float32)
        Lv = Lv / np.linalg.norm(Lv)
        rd = np.array([0.6 * lean, -1.0], np.float32)
        rd = rd / np.linalg.norm(rd)
        t_term = 0.3 + 0.2 * (1 - prox) + 0.3 * near + rng.uniform(-0.05, 0.05)
        rim_k = min((0.3 + 1.0 * prox) * (1 - 0.4 * near), 1.0)
        rim_px = (1.4 + 1.6 * prox) * (0.8 + 0.4 * min(f * 2, 1))
        rimc = _mix(RIM_OFF, RIM_SUN, prox)
        grp = (cx - ox, by - oy + 0.1 * hc, 0.62 * wc, 1.15 * hc)
        heads = [(0.0, 0.0, 1.0, 1.0)]
        nh = int(rng.integers(1, 3 + int(round(2 * near))))
        for k in range(nh):
            sgn = -1 if rng.random() < 0.5 else 1
            dyk = rng.uniform(0.15, 0.3) + 0.12 * k * near
            heads.append((sgn * rng.uniform(0.12, 0.42), dyk, rng.uniform(0.35, 0.6), rng.uniform(0.4, 0.65)))
        for (dx, dy, sw, sh_) in heads:
            hw, hh = wc * sw, hc * sh_
            env = K.envelope(cx - ox + dx * wc, by - oy + dy * hc, hw, hh, rng, lean=0.1 * lean, power=2.2,
                             lump=0.08, base_round=0.12, skew=rng.uniform(-0.3, 0.3))
            size = max(hh, 0.45 * hw)
            m, mx0, my0 = K.cauliflower(env, rng, size, lv, down_cut=0.2, side_scale=0.65, sun_dir=tuple(rd),
                                        sun_bias=0.25, concave=0.6, clump=0.45)
            if m.size == 0 or m.max() <= 0:
                continue
            paint_mass(cv, m, mx0, my0, Lv, Pm, size, group=grp, rng=rng, dome=0.1,
                       dv=rng.uniform(-0.05, 0.05) - 0.12 * dy, t_term=t_term, t_hot=t_term + 0.2 + 0.2 * (1 - prox),
                       hot_w=0.1, lit_span=0.25, step=0.35, sky=0.25, scallop=0.6, wander=0.06, under=0.5 + 0.2 * near, haze=hz,
                       hazec=hc_, base=(by - oy + dy * hc - 0.15 * hh, 0.45 * hh), tear=0.6 + 0.3 * near,
                       rim=rim_k, rim_px=rim_px, rimc=rimc, rim_dir=rd, glow=0.6 * prox)


def under_deck(cv, W, H, hy, sun, y_top, seed):
    """Opaque hazy lower cloud deck from y_top (frame px, wavy) down: fills every gap behind the rows."""
    ox, oy = cv.ox, cv.oy
    xs = np.arange(cv.w, dtype=np.float32) + ox
    ys = (np.arange(cv.h, dtype=np.float32) + oy)[:, None]
    wav = K._noise(cv.w, 8, max(cv.w / (160 * cv.u), 2), seed, 3)[4] - 0.5
    yt = y_top + wav * 0.03 * H
    a = _ss(yt[None, :] - 0.01 * H, yt[None, :] + 0.02 * H, ys)
    col = floor_color((ys - hy) / H, xs[None, :], W, sun[0])
    # soft lumps of the lower deck (very low contrast, hazy)
    lump = cv.n_md * 0.7 + cv.n_lo * 0.6
    col = col * (1 + 0.07 * lump[..., None]) + 0.04 * _ss(0.1, 0.3, cv.n_md)[..., None] * np.array([1.0, 0.85, 0.8], np.float32)
    cv.over(col.astype(np.float32), a.astype(np.float32), 0, 0)


def sea_plates(W, H, hy, sun, bands, seed=7, margin=0.14):
    """Paint the sea into depth plates. bands = [(dy0, dy1), ...] (fraction of H below the horizon) ->
    list of plate dicts (far -> near). Every plate but the farthest carries its own hazy under deck."""
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    rows = sea_rows(W, H, hy, seed)
    plates = []
    for k, (a, b) in enumerate(bands):
        rs = [r for r in rows if a <= (r[1] - hy) / H < b]
        if not rs:
            plates.append(None)
            continue
        top = min(r[1] - 3.2 * r[3] for r in rs)
        ox, oy = -margin * W, max(hy - 0.02 * H, top - 0.18 * H)
        cv = Canvas(W * (1 + 2 * margin), H * 1.25 - oy, ox, oy, u, seed + 31 * k)
        if k > 0:
            under_deck(cv, W, H, hy, sun, rs[0][1] - 0.2 * rs[0][3], seed + 5 * k)
        for (f, y, bw, bh) in rs:
            paint_row(cv, W, H, hy, sun, f, y, bw, bh, np.random.default_rng(int(rng.integers(1 << 30))), u)
        plates.append(cv.plate())
    return plates


# ----------------------------------------------------------------------------------------- towers
TOWER_LV = ((0.13, 0.26, 0.9, 1.4, 2.6, 0.08, 0.42), (0.05, 0.1, 0.85, 1.1, 2.3), (0.022, 0.04, 0.7, 1.2, 2.6),
            (0.011, 0.018, 0.3, 1.6, 3.6))


def tower(W, H, sun, cx, base_y, Hc, seed, side=1.0, width=1.0, haze=0.0, margin=0.1):
    """A towering cumulus congestus: 6-8 overlapping masses (back to front), one shared ellipsoid form,
    lit warm-white on the sunward side (side=+1: sun to the right), cool lavender shadow away from it,
    crisp overlapping-mass edges, edge-only florets, gold rim on the sunward silhouette only, hazy base."""
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    wd = 0.95 * Hc * width
    x0 = int(cx - 0.9 * wd - margin * W)
    y0 = int(base_y - 1.25 * Hc)
    cw = int(1.8 * wd + 2 * margin * W)
    ch = int(1.25 * Hc + 0.25 * Hc)
    cv = Canvas(cw, ch, x0, y0, u, seed)
    lx, ly = cx - x0, base_y - y0              # local base centre
    sx, sy = sun[0] - x0, sun[1] - y0
    Lv = np.array([0.8 * side, -0.5, -0.1], np.float32)
    Lv /= np.linalg.norm(Lv)
    group = (lx + 0.02 * side * wd, ly - 0.5 * Hc, 0.62 * wd, 0.62 * Hc)
    P = pal(TOWER)
    # masses (rel x (sunward +), rel base (0 = cloud base, 1 = top), width, height): back -> front
    masses = [
        (-0.08, 0.02, 0.95, 0.72),     # main column (back)
        (-0.18, 0.5, 0.52, 0.5),        # upper back head (away from sun)
        (0.05, 0.62, 0.46, 0.38),       # crown
        (0.2, 0.4, 0.5, 0.36),          # sunward shoulder
        (-0.24, 0.1, 0.5, 0.46),        # shadow-side lower mass (front)
        (0.22, 0.05, 0.6, 0.38),        # sunward lower mass (front)
        (-0.02, -0.02, 0.8, 0.24),      # front base billows
    ]
    for i, (rx, rb, rw, rh) in enumerate(masses):
        mcx = lx + side * rx * wd + rng.uniform(-0.03, 0.03) * wd
        mby = ly - rb * Hc
        mw, mh = rw * wd * rng.uniform(0.92, 1.08), rh * Hc * rng.uniform(0.92, 1.08)
        env = K.envelope(mcx, mby, mw, mh, rng, lean=0.08 * side * rng.uniform(-1, 1), power=2.0, lump=0.08,
                         base_round=0.05, skew=rng.uniform(-0.3, 0.3))
        size = max(mh, 0.6 * mw)
        sp = np.array([sx - mcx, sy - (mby - 0.5 * mh)], np.float32)
        sp /= np.linalg.norm(sp) + 1e-6
        m, mx0, my0 = K.cauliflower(env, rng, size, TOWER_LV, down_cut=0.3, side_scale=0.85, sun_dir=tuple(sp),
                                    sun_bias=0.1, concave=0.65, clump=0.3)
        front = i >= 4
        paint_mass(cv, m, mx0, my0, Lv, P, size, group=group, rng=rng, dome=0.12, dv=(0.0 if i == 0 else rng.uniform(-0.12, 0.12)),
                   t_term=0.05, t_hot=0.45, hot_w=0.2, step=0.0, sky=0.2, scallop=1.0, wander=0.05, refl=1.0,
                   under=0.4, haze=haze + (0.1 if front else 0.0) * (1 - 0.3 * i / 6),
                   hazec=HAZE_OFF, rim=0.0)
    # haze toward the base (the tower rises out of the sea's haze)
    ys = np.arange(ch, dtype=np.float32)[:, None]
    hk = (_ss(ly - 0.55 * Hc, ly + 0.02 * Hc, ys) * 0.55)[..., None]
    hc = _mix(HAZE_OFF, HAZE_SUN, 0.35)
    a = cv.prem[..., 3:4]
    cv.prem[..., :3] = cv.prem[..., :3] * (1 - hk) + hc * a * hk
    lining(cv, (sx, sy), 2.2, strength=1.0)
    return cv.plate()


def lining(cv, sun, rim_px, strength=1.0, color=None):
    """Silver/gold lining on the outer silhouette of the canvas union where it faces the sun and the
    paint there is lit: a crisp 2-4 px HDR line + soft falloff inside + a faint halation outside."""
    u = cv.u
    A = np.ascontiguousarray(np.clip(cv.prem[..., 3], 0, 1))
    ys, xs = np.mgrid[0:cv.h, 0:cv.w].astype(np.float32)
    dx, dy = sun[0] - xs, sun[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl
    r = rim_px * u
    s1 = cv2.remap(A, xs + ux * r, ys + uy * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    s2 = cv2.remap(A, xs + ux * 2.4 * r, ys + uy * 2.4 * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    core = A * (1 - s1)
    soft = A * (1 - _blur(s2, 0.8 * r))
    rgb = cv.prem[..., :3] / np.maximum(A, 1e-5)[..., None]
    litn = _ss(0.72, 0.9, rgb.max(-1))
    k = np.clip((1.2 * core + 0.3 * soft) * (0.1 + 0.9 * litn) * strength, 0, 1)
    col = RIM_SUN if color is None else color
    cv.prem[..., :3] = cv.prem[..., :3] * (1 - k[..., None]) + col * (A * k)[..., None]
    hk = np.clip((_blur(k * A, 2.5 * u) + 0.4 * _blur(k * A, 9 * u)) * (1 - A) * 0.5, 0, 1)[..., None]
    cv.prem[..., :3] += col * hk
    cv.prem[..., 3:4] += hk * 0.5


# ----------------------------------------------------------------------------------------- fg wisps
def fg_wisp(W, H, sun, cx, cy, w, h, seed):
    """A torn, semi-transparent cloud wisp very near the camera (defocused), lit gold along its top."""
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    pw, ph = int(w * 1.3), int(h * 2.2)
    x0, y0 = int(cx - pw / 2), int(cy - ph / 2)
    cv = Canvas(pw, ph, x0, y0, u, seed)
    poly = K.billow_strip(0.15 * pw, 0.85 * pw, 0.55 * ph, 0.22 * w, 0.28 * h, rng, depth=0.5 * h, var=1.0)
    m, mx0, my0 = K.cauliflower(poly, rng, 0.45 * h, LV_NEAR, down_cut=0.1, side_scale=0.7, concave=0.6)
    P = mixpal(pal(SEA_NEAR), pal(SEA_OFF), 0.3)
    Lv = np.array([0.2, -1.0, 0.0], np.float32)
    Lv /= np.linalg.norm(Lv)
    paint_mass(cv, m, mx0, my0, Lv, P, 0.45 * h, group=(pw / 2, 0.6 * ph, 0.4 * pw, 0.4 * h), rng=rng,
               t_term=0.35, t_hot=0.75, under=0.5, base=(0.55 * ph, 0.35 * h), tear=1.1, rim=0.5, rim_px=2.0,
               rimc=_mix(RIM_OFF, RIM_SUN, 0.5), rim_dir=(0.0, -1.0))
    pl = cv.plate()
    rgba = pl['rgba']
    # fibrous, see-through: streaky alpha, soft defocus
    st = K._noise(pw, ph, max(pw / (60 * u), 2), seed + 9, 3, stretch=5.0)
    rgba[..., 3] *= (0.45 + 0.55 * _ss(0.3, 0.7, st)) * 0.8
    pl['rgba'] = cv2.GaussianBlur(rgba, (0, 0), 3.5 * u)
    return pl
