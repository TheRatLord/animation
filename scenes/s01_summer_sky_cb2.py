"""s01 helper: the hero cumulonimbus, painted as a handful of BIG overlapping value masses.

Silhouettes come from lib/clouds2 (envelope domes grown into cauliflower edges; big billows on the crown,
mid florets on the shoulders, small fractal bumps low down). The interior is painted like a background
artist would: every mass is one smooth value mass (warm white where it faces the sun, lavender mid,
cool blue-grey shadow core) whose tone comes from ONE tower-wide form plus a gentle turn of its own
dome - no per-lobe shading. Masses are laid back to front with crisp alpha edges, so the only hard edges
inside the cloud are where one mass overlaps another; the mass behind receives a soft contact shade.
Shadow masses get a warmer bounce lip underneath and a cyan skylight tint on their up-facing parts.
Finally the clouds2 Painter adds the silver/gold lining on the sun-facing silhouette, the torn wispy
base, and we add a warm translucent glow just inside the edge nearest the sun."""
import math
import numpy as np
import cv2

from lib import clouds2 as K

_ss = K._ss

# value ramp (t: 0 = deepest shadow core .. 1 = hottest light toward the sun)
RAMP = [(0.00, '#6f80b6'), (0.18, '#8494c6'), (0.34, '#9ea9d4'), (0.46, '#b7b8da'), (0.53, '#cfc8dd'),
        (0.60, '#e3d8dc'), (0.70, '#f0e4da'), (0.84, '#f7ece0'), (1.00, '#fcf4e8')]
_RT = np.array([r[0] for r in RAMP], np.float32)
_RC = np.stack([K._c(r[1]) for r in RAMP]).astype(np.float32)

BOUNCE = K._c('#d6c6c9')        # warm light bounced up from the sunlit ground haze
SKYREF = K._c('#a3c3ee')        # reflected cyan from the open sky on up-facing shadow parts
GLOW = np.array([1.0, 0.86, 0.66], np.float32)

# cauliflower levels (rmin, rmax, prob, spacing min/max, bulge min/max) as fractions of the mass size
LEV_CROWN = ((0.16, 0.34, 0.95, 1.3, 2.3, 0.1, 0.42), (0.06, 0.12, 0.85, 1.0, 2.2, 0.25, 0.55),
             (0.022, 0.045, 0.55, 1.2, 3.0))
LEV_MID = ((0.1, 0.22, 0.9, 1.2, 2.4, 0.12, 0.45), (0.04, 0.08, 0.8, 1.0, 2.4, 0.25, 0.55),
           (0.018, 0.032, 0.5, 1.2, 3.2))
LEV_LOW = ((0.06, 0.13, 0.85, 1.1, 2.4, 0.15, 0.5), (0.025, 0.05, 0.75, 1.0, 2.6),
           (0.012, 0.022, 0.5, 1.2, 3.4))


def _ramp(t):
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, _RT, _RC[:, c]) for c in range(3)], -1).astype(np.float32)


def _gblur(img, s):
    return K._blur(img.astype(np.float32), float(s))


def _shift(m, dx, dy):
    return K._shift(m, dx, dy)


def ellipse(cx, cy, rx, ry, rng, lump=0.08, flat=0.6, lean=0.0, n=96):
    """Lumpy ellipse outline (a rounded billow mass); `flat` squashes the underside (0..1), `lean` shifts
    the top sideways (fraction of rx)."""
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    m = np.ones_like(t)
    for k in range(2, 6):
        m += lump / (k - 1) * rng.uniform(-1, 1) * np.sin(k * t + rng.uniform(0, 6.28))
    x = np.cos(t) * rx * m
    y = np.sin(t) * ry * m
    y = np.where(y > 0, y * (1 - flat), y)
    x = x + lean * rx * np.clip(-y / ry, 0, 1)
    return np.stack([cx + x, cy + y], -1).astype(np.float32)


def _normal(hf, k):
    gx = cv2.Sobel(hf, cv2.CV_32F, 1, 0, ksize=3) * 0.125 * k
    gy = cv2.Sobel(hf, cv2.CV_32F, 0, 1, ksize=3) * 0.125 * k
    nl = np.sqrt(gx * gx + gy * gy + 1.0)
    return -gx / nl, -gy / nl, 1.0 / nl


# mass table (all in Hc units, relative to (cx, base)); drawn back to front.
# env: envelope kwargs; lev: floret levels; tone: value offset (+ lit / - shade); dome: weight of the
# mass's own form; size: floret reference size
MASSES = [
    # painted top -> bottom: the crown sits at the back and every lower mass overlaps the one above with
    # its lit cauliflower top (the crisp overlap edges run across the shaded underside above).
    # ell = (dx, dy, rx, ry, flat, lean) in Hc units (lumpy ellipse); otherwise a clouds2 envelope.
    # --- crown heads: a big one up-left billowing toward the sun, a smaller one on the shadowed right
    dict(env=dict(ell=(0.16, -0.86, 0.15, 0.11, 0.2, 0.1), lump=0.1),
         lev=LEV_CROWN, size=0.2, tone=-0.07, dome=0.45, sd=(0.9, 0.6)),
    dict(env=dict(ell=(-0.2, -0.88, 0.19, 0.13, 0.2, -0.15), lump=0.1),
         lev=LEV_CROWN, size=0.22, tone=-0.02, dome=0.45, sd=(0.9, 0.6)),
    # --- crown cap: broad cauliflower billowing up and out (widest toward the sun, upper left)
    dict(env=dict(ell=(-0.04, -0.74, 0.37, 0.17, 0.2, -0.12), lump=0.12),
         lev=LEV_CROWN, size=0.3, tone=0.03, dome=0.45, sd=(0.9, 0.55)),
    # --- right mid-level shoulder (stepped), in shade
    dict(env=dict(ell=(0.25, -0.52, 0.17, 0.14, 0.25, 0.1), lump=0.1),
         lev=LEV_MID, size=0.24, tone=-0.08, dome=0.3, sd=(0.9, 0.6)),
    # --- rising column, narrower than the crown, its top hidden in the crown
    dict(env=dict(dx=0.04, by=-0.18, w=0.5, h=0.52, power=2.3, lump=0.07, lean=-0.05, base_round=0.1),
         lev=LEV_MID, size=0.28, tone=-0.01, dome=0.25, sd=(0.3, 0.72)),
    # --- sun-side lit mass under the crown's left overhang (a step in the left wall)
    dict(env=dict(ell=(-0.13, -0.5, 0.18, 0.15, 0.25, -0.1), lump=0.1),
         lev=LEV_MID, size=0.22, tone=0.04, dome=0.3, sd=(0.9, 0.6)),
    # --- shadow-side mid mass (right front), bounce lip underneath
    dict(env=dict(ell=(0.2, -0.33, 0.18, 0.13, 0.25, 0.1), lump=0.1),
         lev=LEV_MID, size=0.2, tone=-0.06, dome=0.3, sd=(0.9, 0.6)),
    # --- mid lit front mass
    dict(env=dict(ell=(-0.08, -0.3, 0.2, 0.13, 0.25, -0.05), lump=0.1),
         lev=LEV_MID, size=0.2, tone=0.02, dome=0.3, sd=(0.9, 0.6)),
    # --- lower right step
    dict(env=dict(dx=0.38, by=-0.06, w=0.44, h=0.3, power=2.3, lump=0.1, skew=0.2),
         lev=LEV_LOW, size=0.2, tone=-0.1, dome=0.3, sd=(0.35, 0.75)),
    # --- low front mass and the wide base skirt (small fractal bumps low down)
    dict(env=dict(ell=(0.05, -0.14, 0.25, 0.11, 0.3, 0.0), lump=0.1),
         lev=LEV_LOW, size=0.16, tone=-0.02, dome=0.3, sd=(0.9, 0.65)),
    dict(env=dict(dx=0.03, by=0.03, w=1.2, h=0.2, power=2.6, lump=0.08, skew=0.2, base_round=0.05),
         lev=LEV_LOW, size=0.16, tone=-0.08, dome=0.2, sd=(0.35, 0.8)),
    dict(env=dict(dx=-0.3, by=0.0, w=0.44, h=0.2, power=2.6, lump=0.1, skew=-0.3, lean=-0.05),
         lev=LEV_LOW, size=0.16, tone=0.0, dome=0.3, sd=(0.3, 0.75)),
    dict(env=dict(dx=0.18, by=0.02, w=0.5, h=0.15, power=2.6, lump=0.1, skew=0.25),
         lev=LEV_LOW, size=0.13, tone=-0.05, dome=0.3, sd=(0.3, 0.75)),
]


def tower(pw, ph, cx, base, Hc, W, sun, seed=12):
    """Paint the tower. Returns straight-alpha RGBA (ph, pw, 4). sun = (x, y) plate px."""
    u = W / 1920.0
    rng = np.random.default_rng(seed)
    sun = np.asarray(sun, np.float32)
    # light: screen direction from the tower's upper body toward the sun, a little in front (faces lit)
    gc = np.array([cx, base - 0.55 * Hc], np.float32)
    sd = sun - gc
    sd = sd / float(np.hypot(*sd))
    L = np.array([sd[0] * 0.85, sd[1] * 0.85, 0.3], np.float32)
    L /= float(np.linalg.norm(L))

    ys, xs = np.mgrid[0:ph, 0:pw].astype(np.float32)
    masks, envs = [], []
    for M in MASSES:
        e = M['env']
        if 'ell' in e:
            dx_, dy_, rx_, ry_, fl_, ln_ = e['ell']
            poly = ellipse(cx + dx_ * Hc, base + dy_ * Hc, rx_ * Hc, ry_ * Hc, rng, lump=e.get('lump', 0.08),
                           flat=fl_, lean=ln_)
        else:
            ek = {k: v for k, v in e.items() if k not in ('dx', 'by', 'w', 'h')}
            poly = K.envelope(cx + e['dx'] * Hc, base + e['by'] * Hc, e['w'] * Hc, e['h'] * Hc, rng, **ek)
        sp = sun - poly.mean(0)
        sp = sp / (float(np.hypot(*sp)) + 1e-6)
        m, x0, y0 = K.cauliflower(poly, rng, M['size'] * Hc, M['lev'], down_cut=M['sd'][0],
                                  side_scale=M['sd'][1], concave=0.75, clump=0.5, sun_dir=sp, sun_bias=0.25)
        full = np.zeros((ph, pw), np.float32)
        a0, b0 = max(y0, 0), max(x0, 0)
        a1, b1 = min(y0 + m.shape[0], ph), min(x0 + m.shape[1], pw)
        if a1 > a0 and b1 > b0:
            full[a0:a1, b0:b1] = m[a0 - y0:a1 - y0, b0 - x0:b1 - x0]
        masks.append(full)
        env = np.zeros((ph, pw), np.float32)
        cv2.fillPoly(env, [np.round(poly * 4).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=2)
        envs.append(env)
    U = np.clip(np.maximum.reduce(masks), 0, 1)

    # shading is computed at quarter resolution (smooth fields) and upsampled with a cubic filter
    q = 4
    qw, qh = pw // q, ph // q

    def dn(a):
        return cv2.resize(a, (qw, qh), interpolation=cv2.INTER_AREA)

    def up(a):
        return cv2.resize(a.astype(np.float32), (pw, ph), interpolation=cv2.INTER_CUBIC)

    Uq = dn(U)
    Hg = 0.65 * cv2.GaussianBlur(Uq, (0, 0), 0.16 * Hc / q) + 0.35 * cv2.GaussianBlur(Uq, (0, 0), 0.05 * Hc / q)
    gnx, gny, gnz = _normal(Hg, 2.2 * Hc / q)
    ysq, xsq = np.mgrid[0:qh, 0:qw].astype(np.float32) * q
    top_y = base - Hc
    yv = np.clip((ysq - top_y) / Hc, 0, 1.2)
    height_light = 0.07 * (1 - _ss(0.15, 0.8, yv)) - 0.1 * _ss(0.5, 1.0, yv)
    br = K._noise(qw, qh, 5.0, seed + 3, 3) - 0.5
    dsq = np.hypot(xsq - sun[0], ysq - sun[1]) / Hc
    nearQ = np.exp(-dsq / 0.35)
    br2 = K._noise(pw, ph, 14.0, seed + 5, 2) - 0.5
    dsun = np.hypot(xs - sun[0], ys - sun[1]) / Hc

    rgb = np.zeros((ph, pw, 3), np.float32)
    A = np.zeros((ph, pw), np.float32)
    for i, (M, m, env) in enumerate(zip(MASSES, masks, envs)):
        if m.max() <= 0:
            continue
        size = M['size'] * Hc / q
        mq, eq = dn(m), dn(env)
        hm = cv2.GaussianBlur(np.maximum(eq, cv2.GaussianBlur(mq, (0, 0), 0.04 * size + 0.3)), (0, 0), 0.22 * size)
        mnx, mny, mnz = _normal(hm, 0.9 * size)
        wd = M['dome']
        nx = (1 - wd) * gnx + wd * mnx
        ny = (1 - wd) * gny + wd * mny
        nz = (1 - wd) * gnz + wd * mnz
        nl = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        lam = (nx * L[0] + ny * L[1] + nz * L[2]) / nl
        t = 0.5 + 0.75 * (lam - 0.55) + M['tone'] + height_light + 0.06 * br + 0.04 * nearQ
        t = t + 0.06 * np.tanh((t - 0.5) / 0.04)
        shade = 1 - _ss(0.42, 0.58, t)
        # bounce lip: a broad warm lift along the underside of every mass in shade
        d_lip = 0.12 * size
        lip = np.clip(mq - cv2.GaussianBlur(K._shift(mq, 0, d_lip), (0, 0), 0.5 * d_lip + 0.3), 0, 1)
        lip = cv2.GaussianBlur(lip, (0, 0), 0.5 * d_lip + 0.3) * shade * 0.45
        upf = np.clip(-ny / nl * 1.6, 0, 1) * shade * 0.28
        t, lip, upf = up(t), up(lip)[..., None], up(upf)[..., None]
        col = _ramp(t)
        col = col * (1 - lip) + BOUNCE * lip
        col = col * (1 - upf) + SKYREF * upf
        # soft contact shade on the paint behind, just outside this mass away from the sun
        if A.max() > 0:
            dc = 0.05 * size
            cast = up(cv2.GaussianBlur(K._shift(mq, L[0] * dc, L[1] * dc), (0, 0), 0.6 * dc + 0.3)) * (1 - m) * A
            rgb *= (1 - 0.1 * np.clip(cast, 0, 1)[..., None] * np.array([1.05, 1.0, 0.9], np.float32))
        rgb = rgb * (1 - m[..., None]) + col * m[..., None]
        A = A * (1 - m) + m

    # warm translucent glow just inside the edge nearest the sun (light through the thin crown edge)
    dT = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    band = np.exp(-dT / (0.028 * Hc)) * A
    gk = band * np.exp(-dsun / 0.2)
    rgb = rgb + GLOW * (0.3 * gk)[..., None]
    rgb = rgb * (1 + 0.012 * br2[..., None])                        # faint paint-scale variation

    # hand the plate to the clouds2 Painter for the silver/gold lining and the torn wispy base
    pal = dict(K.PRESETS['noon'])
    pal.update(rim=(1.7, 1.52, 1.2), haze='#cfe5f5', shade='#9aa6d2', deep='#8594c6', refl='#b0bfe6',
               lit_lo='#e6dcdf')
    P = K.Painter(pw, ph, sun=tuple(sun), pal=pal, unit=u, seed=seed + 1, sun_z=0.3)
    P.prem[..., :3] = rgb * A[..., None]
    P.prem[..., 3] = A
    P.sun_R = 0.5 * Hc
    P.lining(strength=1.5, rim_px=3.0, halo=0.9, backlit=0.0, y_max=base - 0.3 * Hc)
    # aerial perspective + horizontal streaks: the lower body dissolves into the horizon haze
    yk = _ss(base - 0.55 * Hc, base - 0.1 * Hc, ys)
    st = K._noise(pw, ph, 9.0, seed + 21, 4, stretch=6.0)
    st2 = K._noise(pw, ph, 22.0, seed + 23, 3, stretch=9.0)
    lum = P.prem[..., :3].mean(-1, keepdims=True)
    kk = (0.72 * yk ** 1.3)[..., None]
    P.prem[..., :3] = P.prem[..., :3] * (1 - kk) + (lum * 0.3 + K._c('#cfe4f5') * P.prem[..., 3:4] * 0.7) * kk
    tear = 0.85 * _ss(0.5, 0.95, yk) * (1 - _ss(0.35, 0.62, 0.6 * st + 0.4 * st2 + 0.25 * (1 - yk)))
    P.prem *= (1 - np.clip(tear * 0.9, 0, 1))[..., None]
    P.wisps(cx - 0.72 * Hc, cx + 0.78 * Hc, base + 0.02 * Hc, 0.09 * Hc, rng=rng, seed=seed + 11,
            amount=0.5, streak=5.0, erode=1.0, base_haze=0.2)
    return P.rgba()
