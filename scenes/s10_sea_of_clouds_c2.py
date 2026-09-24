"""s10 helpers (round 7): sea of clouds + towering cumulus painted with lib/clouds2.Painter.

Plates are painted in SCREEN coordinates (1 unit = 1 frame px) and stored at `scale` x resolution
(near plates are supersampled so the forward dolly never softens them). A plate covers the frame plus a
margin M on every side.

Sea: rows from the horizon to below the frame on a perspective curve. Far rows are long, flat, pale
streaks dissolving into a peach-white haze; mid rows are banks of varied billows; near rows are a few
huge cauliflower mounds. Each billow is lit from above (low sun ahead, light raking over the tops): a
top plane stepping gold -> peach -> pink toward a scalloped terminator, then deep indigo valleys.
Gold and hotter on the sun axis, pinker away from it.
Towers: laid out mass by mass (a leaning column, a crown dome, shoulders, a shadowed base) with a key
light above and toward the sun so the inner flank AND the crown catch one broad warm-lit plane; the
far side is a cool lavender shadow with crisp overlapping sub-masses; one plate-level silver/gold
lining on the outer silhouette.
"""
import math

import numpy as np
import cv2

from numba import njit, prange

from lib import core as C, sky as S, clouds2 as K

M = 0.05                     # plate margin (fraction of W / H per side)

# --------------------------------------------------------------------------------------- palettes
SEA_GOLD = dict(hi=(1.0, 0.9, 0.62), lit=(1.0, 0.74, 0.46), lit_lo=(0.97, 0.56, 0.5), mid='#e0668c',
                shade='#3f3c92', deep='#2a2c6e', refl='#4a4a9e', bounce='#5e4a92', rim=(1.5, 1.2, 0.78),
                haze='#ffd8bc', edge_dark=0.05)
SEA_PINK = dict(hi=(1.0, 0.8, 0.66), lit=(0.98, 0.64, 0.58), lit_lo=(0.88, 0.5, 0.66), mid='#b4589c',
                shade='#3b3a8a', deep='#2a2c6e', refl='#46489a', bounce='#54488e', rim=(1.25, 1.0, 0.88),
                haze='#f4c2c6', edge_dark=0.05)
SEA_FAR = dict(hi=(1.0, 0.92, 0.8), lit=(1.0, 0.83, 0.7), lit_lo=(0.96, 0.72, 0.72), mid='#e2a0b6',
               shade='#9c88c4', deep='#7c6cb2', refl='#a898cc', bounce='#c4a0bc', rim=(1.2, 1.05, 0.9),
               haze='#ffdcc4', edge_dark=0.02)

TOWER = dict(hi=(1.0, 0.92, 0.7), lit=(1.0, 0.8, 0.56), lit_lo='#fbb48c', mid='#e27a92', shade='#6c64b8',
             deep='#4a4596', refl='#8a86cc', bounce='#e0949c', rim=(1.55, 1.3, 0.9), haze='#f2bcc0',
             edge_dark=0.04)
TOWER_BASE = dict(TOWER, lit=(0.98, 0.72, 0.58), lit_lo='#f0a090', hi=(1.0, 0.82, 0.66), shade='#5c56aa',
                  deep='#3e3a88')


def mixp(a, b, t):
    return K.mix_palette(a, b, float(np.clip(t, 0, 1)))


# --------------------------------------------------------------------------------------- plate frame
class Plate:
    """A Painter covering the frame + margin at `scale` x resolution; draw in frame px via .P / .tp."""

    def __init__(self, W, H, scale, sun, pal, seed, sun_z=0.2):
        self.W, self.H, self.s = W, H, float(scale)
        self.ox, self.oy = M * W, M * H
        self.pw = int(round((W + 2 * self.ox) * self.s))
        self.ph = int(round((H + 2 * self.oy) * self.s))
        self.P = K.Painter(self.pw, self.ph, self.tp(sun), pal, self.s * W / 1920.0, seed=seed, sun_z=sun_z)

    def tp(self, p):
        p = np.asarray(p, np.float32)
        return (p + np.array([self.ox, self.oy], np.float32)) * self.s

    def rgba(self):
        return self.P.rgba()


def sample(plate, W, H, s, z, focus, shift=(0.0, 0.0), billow=0.0, t=0.0, seed=0, _cache={}):
    """Frame = focus + (plate_px / s - margin - focus) * z + shift  (zoom about the vanishing point)."""
    ox, oy = M * W, M * H
    fx, fy = focus
    a = z / s
    bx = fx + (-ox - fx) * z + shift[0]
    by = fy + (-oy - fy) * z + shift[1]
    if not billow:
        A = np.array([[a, 0, bx], [0, a, by]], np.float32)
        return cv2.warpAffine(plate, A, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(0, 0, 0, 0))
    key = (W, H)
    if key not in _cache:
        _cache[key] = C.grid(W, H)
    X, Y = _cache[key]
    px = (X - bx) / a
    py = (Y - by) / a
    h, w = plate.shape[:2]
    f1, f2, q = S._billow_fields(h, w, seed, 0.03)
    ph = t * 0.25
    wt = (math.cos(ph), math.sin(ph), math.cos(0.61 * ph + 1.3), math.sin(0.61 * ph + 1.3))
    dx = f1[..., 0] * wt[0] + f1[..., 1] * wt[1] + 0.7 * f2[..., 0]
    dy = f1[..., 2] * wt[2] + f1[..., 3] * wt[3] + 0.7 * f2[..., 2]
    D = np.dstack([dx, dy]).astype(np.float32)
    d = cv2.remap(D, (px / q).astype(np.float32), (py / q).astype(np.float32), cv2.INTER_LINEAR,
                  borderMode=cv2.BORDER_REFLECT)
    A_ = billow * W * s
    return cv2.remap(plate, (px + d[..., 0] * A_).astype(np.float32), (py + d[..., 1] * A_).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# --------------------------------------------------------------------------------------- the sea
def sea(W, H, hy, sun, rows=19, splits=(6, 10, 14, 17), scales=(1.0, 1.0, 1.08, 1.2, 1.35), seed=7,
        persp=2.15, tower_cb=None):
    """Paint the sea of clouds into len(splits)+1 plates (far .. near). Returns the ordered layer list
    [(Plate, depth_index, kind)]: sea plates, with tower plates (from tower_cb(i, y, k) -> Plate | None,
    called after each row) inserted right after the sea plate of their row. Towers should sit on the
    last row of a sea plate so nearer rows of that plate are not drawn behind them."""
    npl = len(splits) + 1
    plates = [Plate(W, H, scales[k], sun, SEA_GOLD, seed + 10 + k, sun_z=0.15) for k in range(npl)]
    rng = np.random.default_rng(seed)
    bottom = 1.16 * H
    sunx = float(sun[0])

    def pick(i):
        return sum(1 for s_ in splits if i >= s_)

    towers_ = []

    # under-layer: below the horizon the sea is continuous (never sky through the gaps): a hazy
    # peach -> lavender -> indigo wash painted first on the far plate
    P0 = plates[0]
    ysf = (np.arange(P0.ph, dtype=np.float32) / P0.s - P0.oy)[:, None]
    kk = np.clip((ysf - hy) / (H - hy), 0, 1)
    c0, c1, c2 = K._c('#f6c4b4'), K._c('#8a6cb4'), K._c('#34307a')
    t1 = np.clip(kk / 0.25, 0, 1)
    t2 = np.clip((kk - 0.25) / 0.5, 0, 1)
    col = np.where(kk < 0.25, c0 + (c1 - c0) * t1, c1 + (c2 - c1) * t2)          # (ph, 3)
    a = K._ss(hy - 0.002 * H, hy + 0.016 * H, ysf)                                 # (ph, 1)
    P0.P.prem[..., :3] = (col * a)[:, None, :]
    P0.P.prem[..., 3] = a

    for i in range(rows):
        k = pick(i)
        pl = plates[k]
        P, s = pl.P, pl.s
        f = i / (rows - 1)
        zf = f ** persp
        y = hy + (bottom - hy) * zf
        bw = (0.014 + (0.26 - 0.014) * zf) * W                 # typical billow width (frame px)
        asp = 0.07 + (0.27 - 0.07) * f ** 1.3                    # billow height / width: flat far, domed near
        bh = bw * asp
        hz = 0.9 * (1 - K._ss(0.0, 0.5, f)) ** 1.25              # haze: far rows melt into the glow
        farm = 1 - K._ss(0.08, 0.45, f)                          # far palette weight
        # lobe hierarchy by depth: horizon rows smooth streaks, mid rows busy, near rows big heads
        if f < 0.2:
            lv = ((0.06, 0.14, 0.5, 1.6, 3.2),)
        elif f < 0.45:
            lv = ((0.07, 0.17, 0.85, 1.2, 2.4, 0.1, 0.45), (0.025, 0.05, 0.6, 1.3, 2.8))
        elif f < 0.75:
            lv = ((0.09, 0.22, 0.9, 1.3, 2.6, 0.1, 0.45), (0.035, 0.07, 0.7, 1.2, 2.6), (0.013, 0.024, 0.35, 1.5, 3.4))
        else:
            lv = ((0.12, 0.3, 0.9, 1.3, 2.5, 0.08, 0.42), (0.045, 0.09, 0.75, 1.2, 2.4),
                  (0.016, 0.03, 0.45, 1.4, 3.0))
        segs = []
        x = -M * W - bw * rng.uniform(0.3, 1.2)
        while x < W * (1 + M) + bw:
            L = bw * rng.uniform(1.4, 4.0 - 1.6 * f) * (1 + 1.6 * (1 - f) ** 2)
            sc = math.exp(rng.normal(0, 0.3 + 0.1 * f))            # billow size varies strip to strip
            if f > 0.6 and rng.random() < 0.25:
                sc *= rng.uniform(1.25, 1.5)                          # a big foreground mound
            sc = min(max(sc, 0.6), 1.7)
            segs.append((x, x + L, y + rng.uniform(-0.5, 0.35) * bh, sc, float(np.clip(rng.normal(0, 1), -1.3, 1.3)), rng.random()))
            x += L * rng.uniform(0.45, 0.85)
        rng.shuffle(segs)
        for (sx0, sx1, by, sc, jl, skip) in segs:
            tr = (math.sin(sx0 * 12.9898 + i * 78.233) * 43758.5453) % 1.0      # per-strip paint variety
            tr2 = (math.sin(sx1 * 4.1414 + i * 17.17) * 24634.6345) % 1.0
            if skip < 0.1 * f:
                continue
            bws, bhs = bw * sc ** 0.8, bh * sc
            poly = K.billow_strip(sx0, sx1, by, bws, bhs, rng, depth=bhs * 1.5, var=0.7 + 0.9 * f, power=2.0 + f)
            poly = pl.tp(poly)
            cxs = 0.5 * (sx0 + sx1)
            prox = math.exp(-((cxs - sunx) / (0.2 * W)) ** 2)            # on the sun axis
            proxw = math.exp(-((cxs - sunx) / (0.42 * W)) ** 2)
            pal = mixp(SEA_PINK, SEA_GOLD, 0.25 + 0.75 * proxw)
            pal = mixp(pal, SEA_FAR, farm)
            # sun-path haze tint: hotter and paler right under the sun
            sz = bhs * 1.35 * s
            ux = 0.35 * float(np.clip((sunx - cxs) / (0.5 * W), -1, 1))
            sun_s = pl.tp((cxs + ux * 1e5, by - 1e5))
            near = K._ss(0.55, 1.0, f)
            kw = dict(size=sz, pal=pal, haze=hz, sun=sun_s,
                      group=(float(poly[:, 0].mean()), float(pl.tp((0, by + bhs * 0.3))[1]),
                             (sx1 - sx0) * 0.55 * s, bhs * 1.5 * s),
                      form=0.18, mass_r=0.55, smooth=0.03, sun_z=0.1 - 0.12 * f + 0.1 * prox + 0.04 * jl,
                      split=0.08 + 0.4 * f + 0.1 * near + 0.08 * (1 - proxw) - 0.04 * prox - 0.05 * jl,
                      lit_step=0.3 + 0.3 * tr, term=0.15 + 0.5 * tr2, term_w=0.03 + 0.07 * tr,
                      firm=0.95, soften=0.2, scallop=1.3, levels=lv, light=0.9 + 0.2 * prox,
                      valley_dark=0.55 + 0.25 * near, valley_span=(0.1, 0.75), base_dark=0.9 + 0.4 * near,
                      lost=0.03 + 0.1 * (1 - f), cast=0.35, brush=0.03, sun_bias=0.25, refl=0.08, bounce=0.06,
                      shade_top=0.0, sky=0.1, hot=0.35 + 0.65 * prox, lit_cool=0.1 * (1 - proxw),
                      rim=(0.35 + 0.9 * prox) * (1 - 0.5 * near) if bhs > 6 else 0.0, rim_px=1.6 + 1.2 * prox,
                      rim_near=1.0, halo=0.3 * prox, rim_shadow=0.0, rim_interior=0.4 * proxw, inner=0,
                      down_cut=0.1, side_scale=0.7, wrap=0.0, term_noise=0.06, lw_base=0.0)
            if f < 0.55:
                P.shape(poly, rng=rng, **kw)
                continue
            # near rows: a shaded body strip, then overlapping cauliflower mounds heaped on it, each with
            # its own form -> lit-on-lit crisp edges and shadowed clefts between the heads
            kb = dict(kw)
            kb['split'] = kw['split'] + 0.12
            P.shape(poly, rng=rng, **kb)
            mr = np.random.default_rng(int(1000 * i + 7 * sx0) & 0x7fffffff)
            nm = max(int((sx1 - sx0) / (bws * 1.1)), 1)
            mounds = []
            for q in range(nm):
                mx = sx0 + (sx1 - sx0) * (q + 0.5 + mr.uniform(-0.3, 0.3)) / nm
                edge = min(mx - sx0, sx1 - mx) / max(sx1 - sx0, 1)
                mw = bws * mr.uniform(0.9, 1.7)
                mh = bhs * mr.uniform(0.8, 1.35) * (0.6 + 0.8 * min(edge * 3, 0.5))
                mb = by + bhs * mr.uniform(0.15, 0.55)
                mounds.append((mb, mx, mw, mh))
            for (mb, mx, mw, mh) in sorted(mounds):
                env = K.envelope(mx, mb, mw, mh, mr, power=mr.uniform(2.1, 2.6), lump=0.1,
                                 skew=mr.uniform(-0.4, 0.4), base_round=0.15, base_wave=0.04)
                km = dict(kw)
                km.update(size=mh * 1.1 * s, form=0.45, mass_r=0.4,
                          group=tuple(float(v) for v in pl.tp((mx, mb - 0.45 * mh))) + (0.6 * mw * s, 0.7 * mh * s),
                          split=kw['split'] + mr.uniform(-0.06, 0.06), cast=0.45, lit_step=0.4 + 0.3 * mr.random())
                P.shape(pl.tp(env), rng=mr, **km)
        if tower_cb is not None:
            tp_ = tower_cb(i, y, k)
            if tp_ is not None:
                towers_.append((tp_, k))
    # plate-level silver/gold lining on the outer silhouettes (tower rims, backlit crests near the sun);
    # not on the far plate (its under-layer would line the horizon itself)
    for pl in plates[1:]:
        pl.P.lining(0.55, rim_px=2.0, halo=0.45, backlit=0.25)
    out = []
    for k, pl in enumerate(plates):
        out.append((pl, k, 'sea'))
        for tp_, kk in towers_:
            if kk == k:
                out.append((tp_, k, 'tower'))
    return out


# --------------------------------------------------------------------------------------- towers
def tower(pl, cx, by, Hc, side=1.0, seed=12, pal=TOWER, pal_base=TOWER_BASE, haze=0.0):
    """Towering cumulus on Plate pl (frame px). side=+1: sun-facing flank on the right."""
    P, s = pl.P, pl.s
    rng = np.random.default_rng(seed)
    sd = float(side)
    Hs = Hc * s
    cxs, bys = [float(v) for v in pl.tp((cx, by))]
    # key light: high and toward the sun, a little in front -> inner flank + crown share one lit plane
    key = (cxs + sd * 1.1 * Hs, bys - 1.35 * Hs)
    g = (cxs, bys - 0.45 * Hs, 0.48 * Hs, 0.62 * Hs)
    common = dict(group=g, form=0.58, term_w=0.08, wrap=0.3, brush=0.0, lit_step=0.55, term=0.35, hot=1.0,
                  sky=0.12, rim_interior=0.0, split=0.27, scallop=1.4, term_noise=0.07, sun=key, sun_z=0.22,
                  pal=pal, haze=haze, rim=0.9, rim_px=2.4, halo=0.4, rim_shadow=0.25, shade_top=0.25,
                  valley_dark=0.25, refl=0.3, bounce=0.25, lost=0.02)

    def X(dx):
        return cxs + sd * dx * Hs

    def mass(env, **kw):
        k = dict(common)
        k.update(kw)
        if isinstance(env, dict):
            env = dict(env)
            env['cx'] = X(env['cx'])
            env['base_y'] = bys - env['base_y'] * Hs
            env['width'] *= Hs
            env['height'] *= Hs
            env['lean'] = sd * env.get('lean', 0.0)
            env['skew'] = sd * env.get('skew', 0.0)
            env = K.envelope(rng=rng, **env)
        if 'size' in k:
            k['size'] = k['size'] * Hs
        if 'group' in kw and kw['group'] is not None:
            gx, gy, grx, gry = kw['group']
            k['group'] = (X(gx), bys - gy * Hs, grx * Hs, gry * Hs)
        return P.shape(env, rng=rng, **k)

    # back to front. 1) far-side (shadow) sub-masses: big overlapping cauliflower heads
    mass(dict(cx=-0.3, base_y=0.3, width=0.38, height=0.4, power=2.4, lump=0.1, lean=-0.04, skew=-0.3,
              base_round=0.35), size=0.24, cast=0.5, side_scale=0.7)
    mass(dict(cx=-0.34, base_y=0.04, width=0.44, height=0.34, power=2.4, lump=0.1, skew=-0.2, base_round=0.3),
         size=0.22, cast=0.5, side_scale=0.7)
    # 2) crown: cauliflower domes behind the body's top, leaning toward the sun (the most gold)
    mass(dict(cx=-0.12, base_y=0.6, width=0.26, height=0.3, power=2.3, lump=0.12, skew=-0.2, base_round=0.5),
         size=0.16, cast=0.5, side_scale=0.62)
    mass(dict(cx=0.07, base_y=0.52, width=0.46, height=0.48, power=2.4, lump=0.1, lean=0.08, skew=0.2,
              base_round=0.5), size=0.26, cast=0.45, side_scale=0.66)
    # 3) the tower body: one broad mass carrying the main lit plane
    mass(dict(cx=0.0, base_y=0.0, width=0.76, height=0.62, power=2.7, lump=0.06, lean=0.06, skew=0.1),
         size=0.3, cast=0.4, side_scale=0.75)
    # 4) sun-side front lobe (lit-on-lit crisp edge), rounded underside
    mass(dict(cx=0.25, base_y=0.2, width=0.32, height=0.26, power=2.4, lump=0.1, lean=0.05, skew=0.3,
              base_round=0.45), size=0.18, cast=0.4, side_scale=0.7)
    # 5) base: broad low masses sinking into the sea (mid values, warm bounce from the sea)
    low = dict(pal=pal_base, light=0.85, cast=0.5, rim=0.7, rim_interior=0.5, shade_top=0.3,
               valley_dark=0.35, sky=0.16, wrap=0.35, wrap_px=0.05, bounce=0.4)
    mass(dict(cx=0.12, base_y=0.02, width=0.52, height=0.24, power=2.5, lump=0.1, skew=0.2), size=0.2, **low)
    mass(dict(cx=-0.12, base_y=-0.02, width=0.5, height=0.16, power=2.3, lump=0.1, skew=-0.2), size=0.14, **low)
    P.wisps(cxs - 0.55 * Hs, cxs + 0.55 * Hs, bys + 0.02 * Hs, 0.07 * Hs, rng=rng, pal=pal_base, seed=seed + 11,
            amount=0.35, erode=1.0, base_haze=0.2)
    P.lining(0.8, rim_px=2.4, halo=0.5, backlit=0.2)
    return pl


# --------------------------------------------------------------------------------------- cirrus
def cirrus(W, H, sun, seed=12):
    """A few deliberate, crisp cirrus wisps high in the sky catching pink / gold underlight: long tapered
    brush strokes (combed fibres inside), RGBA (H, W, 4) in frame px (plate = frame + margin)."""
    rng = np.random.default_rng(seed)
    pw, ph = int(W * (1 + 2 * M)), int(H * (1 + 2 * M))
    ss = 2
    acc = np.zeros((ph * ss, pw * ss), np.float32)
    col = np.zeros((ph, pw, 3), np.float32)
    streaks = [(0.08, 0.17, 0.46, -0.05, 0.010), (0.3, 0.08, 0.36, 0.03, 0.007), (0.62, 0.12, 0.34, -0.04, 0.009),
               (0.78, 0.25, 0.26, 0.05, 0.006), (0.18, 0.3, 0.24, 0.02, 0.005)]
    fib = K._noise(pw * ss, ph * ss, 24, seed + 3, 3, stretch=10.0)
    fib = cv2.warpAffine(fib, cv2.getRotationMatrix2D((pw * ss / 2, ph * ss / 2), -4, 1.0), (pw * ss, ph * ss),
                         borderMode=cv2.BORDER_REFLECT)
    for (x0, y0, L, bend, wd) in streaks:
        n = 80
        tt = np.linspace(0, 1, n)
        xs = (x0 + L * tt) * W
        ys = (y0 + bend * np.sin(tt * math.pi) - 0.04 * tt) * H + 0.004 * H * np.sin(tt * 9 + rng.uniform(0, 6))
        wprof = wd * W * np.sin(np.clip(tt, 0, 1) * math.pi) ** 0.7 * (0.6 + 0.4 * np.sin(tt * 5 + rng.uniform(0, 6)))
        m = np.zeros_like(acc)
        for j in range(n - 1):
            p0 = (int((xs[j] + M * W) * ss), int((ys[j] + M * H) * ss))
            p1 = (int((xs[j + 1] + M * W) * ss), int((ys[j + 1] + M * H) * ss))
            cv2.line(m, p0, p1, 1.0, max(int(wprof[j] * ss), 1), cv2.LINE_AA)
        m = cv2.GaussianBlur(m, (0, 0), 3.0 * ss * W / 1920)
        acc = np.maximum(acc, m)
    a = acc * (0.45 + 0.55 * K._ss(0.3, 0.7, fib))
    a = cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)
    ys = np.arange(ph, dtype=np.float32)[:, None] / ph
    xs = np.arange(pw, dtype=np.float32)[None, :] / pw
    # underlight: gold near the sun side, pink further, cooler lavender high up
    g = np.exp(-((xs - (sun[0] / W + M) / (1 + 2 * M)) / 0.35) ** 2)
    pink = np.array([1.0, 0.72, 0.8], np.float32)
    gold = np.array([1.0, 0.84, 0.62], np.float32)
    lav = np.array([0.8, 0.76, 0.95], np.float32)
    c = pink * (1 - g[..., None]) + gold * g[..., None]
    c = c * K._ss(0.05, 0.35, ys)[..., None] + lav * (1 - K._ss(0.05, 0.35, ys)[..., None])
    col = np.broadcast_to(c, (ph, pw, 3))
    return np.concatenate([col, np.clip(a * 0.7, 0, 1)[..., None]], -1).astype(np.float32)


# --------------------------------------------------------------------------------------- birds
def _bird_pts(cx, cy, s, ph):
    """Distant gull as an M-shaped stroke: tips, wrists, body; wings beat with phase ph."""
    f = math.sin(ph)
    tip = -0.34 * f + 0.06
    wr = -0.16 * f - 0.06
    span = 0.5 * (1 - 0.18 * abs(f))
    pts = [(-span, tip), (-0.22, wr - 0.05), (-0.05, 0.02), (0.0, 0.05), (0.05, 0.02), (0.22, wr - 0.05), (span, tip)]
    return [(cx + x * s, cy + y * s) for x, y in pts]


def birds(img, W, H, t, seed=4):
    """A loose flock of gulls: varied sizes (near / far), independent wingbeats (some gliding),
    slow drift; colour pulled toward the sky behind them by distance (aerial perspective)."""
    rng = np.random.default_rng(seed)
    n = 7
    base = np.array([[0.0, 0.0], [0.04, 0.016], [0.075, -0.008], [0.11, 0.022], [0.06, 0.036], [0.145, 0.006],
                     [0.19, 0.03]]) + rng.normal(0, 0.004, (n, 2))
    dist = np.array([0.1, 0.5, 0.3, 0.8, 0.6, 0.9, 1.0])       # 0 near .. 1 far
    sz = (1.25 - 0.7 * dist) * 0.014 * W
    fr = rng.uniform(1.8, 2.8, n)
    ph0 = rng.uniform(0, 6.28, n)
    glide = rng.uniform(0, 6.28, n)
    ss = 4
    out = img.copy()
    for i in range(n):
        cx = (0.62 + base[i, 0] + (0.012 - 0.006 * dist[i]) * t) * W + 0.003 * W * math.sin(0.5 * t + i)
        cy = (0.19 + base[i, 1] - 0.002 * t) * H + 0.0025 * W * math.sin(0.7 * t + 2 * i)
        # wingbeat, with glide phases (wings held slightly raised, small motion)
        g = 0.5 + 0.5 * math.sin(0.9 * t + glide[i])
        amp = 0.25 + 0.75 * float(C.smoothstep(0.25, 0.45, g))
        ph = ph0[i] + 2 * math.pi * fr[i] * t
        s = sz[i]
        pad = int(s * 0.8) + 3
        x0, y0 = int(cx - pad), int(cy - pad)
        x1, y1 = x0 + 2 * pad, y0 + 2 * pad
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        m = np.zeros((2 * pad * ss, 2 * pad * ss), np.uint8)
        pts = _bird_pts((cx - x0) * ss, (cy - y0) * ss, s * ss, math.asin(amp * math.sin(ph)))
        P = np.array(pts).astype(np.int32)
        th = max(int(round(s * 0.055 * ss)), 1)
        for j in range(len(P) - 1):
            cv2.line(m, tuple(P[j]), tuple(P[j + 1]), 255, th + (th // 2 if 1 <= j <= 4 else 0), cv2.LINE_AA)
        cv2.ellipse(m, tuple(P[3]), (max(int(s * 0.07 * ss), 1), max(int(s * 0.04 * ss), 1)), 0, 0, 360, 255, -1,
                    cv2.LINE_AA)
        a = cv2.resize(m.astype(np.float32) / 255.0, (2 * pad, 2 * pad), interpolation=cv2.INTER_AREA)[..., None]
        reg = out[y0:y1, x0:x1]
        skyc = reg.reshape(-1, 3).mean(0)
        col = np.array([0.18, 0.18, 0.4], np.float32)
        col = col * (1 - 0.55 * dist[i]) + skyc * 0.55 * dist[i]
        out[y0:y1, x0:x1] = reg * (1 - a * 0.92) + col * a * 0.92
    return out


# --------------------------------------------------------------------------------------- fast post
@njit(cache=True, fastmath=True, parallel=True)
def _light_pass(img, path, gp, wr, wg, wb, gw, y0, y1, hza, hzc):
    """In place: horizon haze strip (rows y0..y1), sun path on the lit cloud tops and a warm swell of
    the lit faces (lit = smoothstep(0.55, 0.95, max channel))."""
    H, W = img.shape[0], img.shape[1]
    for i in prange(H):
        for j in range(W):
            r, g, b = img[i, j, 0], img[i, j, 1], img[i, j, 2]
            if i >= y0 and i < y1:
                a = hza[i - y0, j]
                r = r * (1 - a) + hzc[i - y0, j, 0] * a
                g = g * (1 - a) + hzc[i - y0, j, 1] * a
                b = b * (1 - a) + hzc[i - y0, j, 2] * a
            m = max(r, max(g, b))
            x = min(max((m - 0.55) / 0.4, 0.0), 1.0)
            lt = x * x * (3 - 2 * x)
            r = (r + path[i, j, 0] * lt * gp) * (1 + lt * wr * gw)
            g = (g + path[i, j, 1] * lt * gp) * (1 + lt * wg * gw)
            b = (b + path[i, j, 2] * lt * gp) * (1 + lt * wb * gw)
            img[i, j, 0], img[i, j, 1], img[i, j, 2] = r, g, b
    return img


def light_pass(img, path, gp, warm, gw, y0, y1, hza, hzc):
    return _light_pass(np.ascontiguousarray(img, dtype=np.float32), path, np.float32(gp), np.float32(warm[0]),
                       np.float32(warm[1]), np.float32(warm[2]), np.float32(gw), int(y0), int(y1), hza, hzc)


def bloom(img, threshold=1.05, knee=0.25, strength=0.35, halation=0.12, radii=(0.004, 0.012, 0.035, 0.09)):
    """Quarter-resolution multi-scale bloom + warm halation (same look as fx.bloom_soft, ~4x cheaper)."""
    H, W = img.shape[:2]
    q = 4
    sm = cv2.resize(img, (W // q, H // q), interpolation=cv2.INTER_AREA)
    lum = sm.max(-1, keepdims=True)
    k = np.clip((lum - threshold + knee) / (2 * knee), 0, 1)
    br = sm * (k * k)
    acc = np.zeros_like(br)
    wts = (1.0, 0.8, 0.6, 0.45)
    hq, wq = br.shape[:2]
    br4 = cv2.resize(br, (wq // 4, hq // 4), interpolation=cv2.INTER_AREA)
    acc4 = np.zeros_like(br4)
    for r, wt in zip(radii, wts):
        sg = max(r * W / q, 0.6)
        if sg > 6:
            acc4 += cv2.GaussianBlur(br4, (0, 0), sg / 4) * wt
        else:
            acc += cv2.GaussianBlur(br, (0, 0), sg) * wt
    acc += cv2.resize(acc4, (wq, hq), interpolation=cv2.INTER_LINEAR)
    acc *= strength / sum(wts[:len(radii)])
    if halation:
        acc += cv2.GaussianBlur(br, (0, 0), max(0.006 * W / q, 0.6)) * (np.array([1.0, 0.45, 0.2], np.float32) *
                                                                        halation * 0.5)
    return img + cv2.resize(acc, (W, H), interpolation=cv2.INTER_LINEAR)


def paper(W, H, seed=3):
    """Static paper / paint texture (multiplicative, ~+-0.8%): a soft blotchy wash + faint tooth."""
    rng = np.random.default_rng(seed)
    n1 = C.fbm(W, H, W / 6.0, 3, seed=seed) - 0.5
    n2 = C.fbm(max(W // 4, 8), max(H // 4, 8), 6.0, 3, seed=seed + 1) - 0.5
    n2 = cv2.resize(n2.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    fib = rng.standard_normal((H, W)).astype(np.float32)
    fib = cv2.GaussianBlur(fib, (0, 0), sigmaX=max(W / 1920 * 3.0, 0.8), sigmaY=max(W / 1920 * 0.6, 0.4))
    fib = fib / (np.abs(fib).std() * 3 + 1e-6)
    tex = 1 + 0.007 * n1 + 0.016 * n2 + 0.003 * fib
    return tex[..., None].astype(np.float32)
