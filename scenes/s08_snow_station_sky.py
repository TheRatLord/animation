"""Low, heavy snow-cloud ceiling for s08_snow_station.

A deck of painted cloud lumps on a horizontal plane above the camera, projected in perspective (big
lumps overhead, thin flattened rows toward the horizon). Each lump is a union of rounded lobes whose
undersides catch the warm orange-magenta glow of the town below while their bodies fall off into deep
navy-violet; lobes are shaded individually so the ceiling has painted structure instead of fbm fuzz."""
import math
import numpy as np
import cv2

from lib import core as C


def _lobe_poly(cx, cy, a, b, rng, n=40):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ph = rng.uniform(0, 6.28, 3)
    r = 1 + 0.06 * np.sin(th * 3 + ph[0]) + 0.04 * np.sin(th * 7 + ph[1]) + rng.normal(0, 0.012, n)
    ys = np.sin(th)
    # flatter base (underside), rounder top
    yy = np.where(ys > 0, ys * 0.55, ys)
    return np.stack([cx + a * r * np.cos(th), cy + b * r * yy], 1)


def overcast(PW, PH, f, cx, cy, seed=5, glow_x=0.25):
    """Returns (rgb (PH,PW,3), cloud alpha (PH,PW)). f, cx, cy: camera focal/principal point (plate px)."""
    rng = np.random.default_rng(seed)
    us, vs = C.grid(PW, PH)
    elev = (cy - vs) / f                     # tan(elevation)
    # ---- base sky behind the deck: deep blue-violet, warmer glow toward the horizon
    e = np.clip(elev, -0.2, 1.0)
    stops_e = np.array([-0.2, 0.0, 0.04, 0.1, 0.2, 0.35, 0.6], np.float32)
    stops_c = np.array([[0.26, 0.2, 0.34], [0.26, 0.2, 0.34], [0.2, 0.16, 0.33], [0.13, 0.12, 0.29], [0.08, 0.09, 0.23],
                        [0.055, 0.065, 0.18], [0.035, 0.045, 0.13]], np.float32)
    base = np.stack([np.interp(e, stops_e, stops_c[:, c]) for c in range(3)], -1).astype(np.float32)
    # town glow (warm, low, off to one side) and a fainter second glow
    gx = np.exp(-((us / PW - glow_x) / 0.32) ** 2)
    gx2 = np.exp(-((us / PW - 0.78) / 0.2) ** 2) * 0.5
    low = np.exp(-np.clip(elev, 0, None) / 0.08)
    town = (gx + gx2)[..., None] * low[..., None] * np.array([0.12, 0.06, 0.11], np.float32)
    base = base + town

    # ---- cloud deck
    Hc = 420.0
    col = np.zeros((PH, PW, 3), np.float32)
    alp = np.zeros((PH, PW), np.float32)
    lumps = []
    # sample lumps by depth: projected rows from the top of the frame down to the horizon
    zmin = Hc / max((cy + 0.05 * PH) / f, 0.05)
    z = zmin
    while z < 12000:
        row_h = z * 0.26
        xext = (PW * 0.62) * z / f
        X = -xext + rng.uniform(0, 1) * 200
        while X < xext:
            wdt = rng.uniform(260, 800) * (1 + z / 5000)
            if rng.random() < 0.07:
                X += wdt * rng.uniform(0.3, 0.9)       # gap
                continue
            lumps.append((X + wdt / 2, z + rng.uniform(-0.3, 0.3) * row_h, wdt, rng.uniform(80, 200), int(rng.integers(1e9))))
            X += wdt * rng.uniform(0.55, 0.85)
        z += row_h
    lumps.sort(key=lambda L: -L[1])
    for (X, Z, wdt, thick, sd) in lumps:
        r2 = np.random.default_rng(sd)
        u0 = cx + f * X / Z
        vb = cy - f * Hc / Z                       # base line on screen
        a_px = 0.5 * f * wdt / Z
        # the deck is seen from below: thickness projects small, flattening toward the horizon
        el = Hc / Z
        b_px = f * thick / Z * (0.25 + 1.2 * el)
        if a_px < 2 or vb < -b_px * 3 or vb > cy + 4:
            continue
        # glow colour for this lump's underside
        ux = u0 / PW
        g = math.exp(-((ux - glow_x) / 0.35) ** 2) + 0.5 * math.exp(-((ux - 0.78) / 0.22) ** 2)
        warm = math.exp(-el / 0.09)
        und = (np.array([0.24, 0.2, 0.42], np.float32) * (0.55 + 0.45 * warm)
               + np.array([0.22, 0.1, 0.17], np.float32) * g * (0.3 + 0.7 * warm)
               + np.array([0.03, 0.05, 0.1], np.float32) * (1 - warm))
        body = np.array([0.06, 0.07, 0.19], np.float32) * (1 - warm) + und * 0.62 * warm
        nlob = int(r2.integers(4, 9))
        lobes = []
        for k in range(nlob):
            lx = u0 + r2.uniform(-0.8, 0.8) * a_px
            la = a_px * r2.uniform(0.25, 0.5) * (1 - 0.5 * abs(lx - u0) / max(a_px, 1))
            lb = b_px * r2.uniform(0.5, 1.0) * (la / max(a_px * 0.4, 1)) ** 0.5
            ly = vb - lb * r2.uniform(0.4, 1.2)
            lobes.append((ly, lx, la, lb))
        # bottom skirt lobes along the base
        for k in range(int(r2.integers(2, 4))):
            lx = u0 + r2.uniform(-0.7, 0.7) * a_px
            la = a_px * r2.uniform(0.3, 0.55)
            lb = b_px * r2.uniform(0.25, 0.45)
            lobes.append((vb - lb * 0.3, lx, la, lb))
        lobes.sort(key=lambda q: q[0])            # upper (farther) lobes first, lower ones overlap
        for (ly, lx, la, lb) in lobes:
            if la < 1.0:
                continue
            poly = _lobe_poly(lx, ly, la, max(lb, 0.8), r2)
            x0 = int(poly[:, 0].min()) - 2
            y0 = int(poly[:, 1].min()) - 2
            x1 = int(poly[:, 0].max()) + 3
            y1 = int(poly[:, 1].max()) + 3
            X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, PW), min(y1, PH)
            if X1 <= X0 or Y1 <= Y0:
                continue
            ss = 2
            m = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
            cv2.fillPoly(m, [((poly - [x0, y0]) * ss).astype(np.int32)], 255, cv2.LINE_AA)
            m = cv2.resize(m.astype(np.float32) / 255, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
            m = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
            yy = np.arange(Y0, Y1, dtype=np.float32)[:, None]
            xx = np.arange(X0, X1, dtype=np.float32)[None, :]
            # vertical position inside the lobe: 0 top .. 1 base
            tv = np.clip((yy - (ly - lb)) / max(lb * 1.55, 1e-3), 0, 1)
            lit = C.smoothstep(0.1, 1.0, tv) ** 1.8
            # rounded-form shading across the lobe
            dx = np.clip(np.abs(xx - lx) / max(la, 1), 0, 1)
            lit = lit * (1 - 0.35 * dx ** 2)
            c = body * (1 - lit[..., None]) + und * lit[..., None]
            # darker top edge where it tucks behind the lobe in front / the rim of the underside is brighter
            rim = C.smoothstep(0.82, 0.97, tv) * (1 - dx ** 3)
            c = c + und * 0.35 * rim[..., None]
            a = m
            dst = col[Y0:Y1, X0:X1]
            col[Y0:Y1, X0:X1] = dst * (1 - a[..., None]) + c * a[..., None]
            alp[Y0:Y1, X0:X1] = alp[Y0:Y1, X0:X1] * (1 - a) + a
    # soft torn edges: tiny displacement + slight softening; a mist veil toward the horizon
    dx = (C.fbm(PW // 2, PH // 2, 18, 3, seed=seed + 3) - 0.5)
    dy = (C.fbm(PW // 2, PH // 2, 18, 3, seed=seed + 4) - 0.5)
    dx = cv2.resize(dx, (PW, PH)) * PW * 0.006
    dy = cv2.resize(dy, (PW, PH)) * PW * 0.003
    col = C.warp(col, dx, dy)
    alp = C.warp(alp, dx, dy)
    col = cv2.GaussianBlur(col, (0, 0), 1.0)
    alp = cv2.GaussianBlur(alp, (0, 0), 1.0)
    colu = col / np.maximum(alp, 1e-4)[..., None]
    img = base * (1 - alp[..., None]) + colu * alp[..., None]
    # atmospheric haze at the horizon: the deck melts into the glow
    haze = np.exp(-np.clip(elev, 0, None) / 0.05)[..., None]
    img = img * (1 - 0.6 * haze) + (base + town * 0.3) * 0.6 * haze
    return img.astype(np.float32), alp.astype(np.float32)


def snow_night(PW, PH, f, cx, cy, seed=5, glow_x=0.22):
    """Snowing overcast night: an almost uniform deep ultramarine-to-indigo dome (hue ~228deg) with a
    barely-there cloud mass (<4% luminance variation, very soft), falling-snow haze lightening it toward
    the horizon and only a thin warm light-pollution band (amber with a faint rose edge) right on the
    horizon behind the trees. Returns rgb (PH, PW, 3)."""
    us, vs = C.grid(PW, PH)
    elev = (cy - vs) / f
    e = np.clip(elev, -0.2, 1.2)
    stops_e = np.array([-0.2, 0.0, 0.03, 0.08, 0.16, 0.3, 0.5, 0.8], np.float32)
    stops_c = np.array([[0.20, 0.27, 0.46], [0.20, 0.27, 0.46], [0.17, 0.24, 0.45], [0.12, 0.18, 0.39],
                        [0.08, 0.125, 0.31], [0.052, 0.085, 0.235], [0.036, 0.058, 0.175], [0.026, 0.04, 0.13]],
                       np.float32)
    base = np.stack([np.interp(e, stops_e, stops_c[:, c]) for c in range(3)], -1).astype(np.float32)
    # a faint lighter veil where the snow cloud is thinner (huge, very soft forms, low contrast)
    n1 = C.fbm(PW // 4, PH // 4, 2.2, 4, seed=seed + 11)
    n2 = C.fbm(PW // 4, PH // 4, 5.0, 3, seed=seed + 12)
    n = cv2.resize(0.7 * n1 + 0.3 * n2, (PW, PH), interpolation=cv2.INTER_CUBIC)
    n = cv2.GaussianBlur(n, (0, 0), PW * 0.01)
    n = (n - n.mean()) / (n.std() + 1e-6)
    cloud = np.clip(n, -2.2, 2.2) * 0.022 * C.smoothstep(0.02, 0.2, elev)
    base = base * (1 + cloud[..., None]) + cloud[..., None].clip(0, None) * np.array([0.02, 0.03, 0.06], np.float32)
    # town glow scattered in the snow: broad and blue-lavender higher up, a thin amber band at the horizon
    gx = np.exp(-((us / PW - glow_x) / 0.3) ** 2) + 0.55 * np.exp(-((us / PW - 0.8) / 0.22) ** 2)
    ep = np.clip(elev, 0, None)
    broad = np.exp(-ep / 0.14)[..., None] * np.array([0.03, 0.04, 0.08], np.float32)
    band = np.exp(-ep / 0.028)[..., None] * (gx[..., None] * np.array([0.2, 0.11, 0.05], np.float32)
                                              + np.array([0.035, 0.02, 0.03], np.float32))
    rose = (np.exp(-ep / 0.06) - np.exp(-ep / 0.028)).clip(0, None)[..., None] * gx[..., None] * \
        np.array([0.05, 0.022, 0.04], np.float32)
    img = base + broad + band + rose
    return img.astype(np.float32)


def snow_clouds(PW, PH, f, cx, cy, seed=5, glow_x=0.22, strength=1.0):
    """Low snow-cloud ceiling seen from below: soft but defined cloud masses on a plane above the
    camera (perspective-correct, flattening toward the horizon), faintly lit from below by the town
    glow (brighter lower rims, darker tops), gaps showing deeper navy. Returns an additive/multiplicative
    pair applied onto the base night sky: (rgb (PH,PW,3), density (PH,PW))."""
    us, vs = C.grid(PW, PH)
    e = (cy - vs) / f
    rx = (us - cx) / f
    k = 190.0
    TW, TH = 3400, 2000
    n1 = C.fbm(TW // 2, TH // 2, 15.0, 5, seed=seed + 31)
    w1 = C.fbm(TW // 4, TH // 4, 5.0, 3, seed=seed + 32)
    w2 = C.fbm(TW // 4, TH // 4, 5.0, 3, seed=seed + 33)
    n1 = cv2.resize(n1, (TW, TH), interpolation=cv2.INTER_CUBIC)
    w1 = cv2.resize(w1, (TW, TH), interpolation=cv2.INTER_CUBIC)
    w2 = cv2.resize(w2, (TW, TH), interpolation=cv2.INTER_CUBIC)
    n1 = C.warp(n1, (w1 - 0.5) * 260, (w2 - 0.5) * 160)
    n2 = C.fbm(TW // 2, TH // 2, 30.0, 3, seed=seed + 34)
    n2 = cv2.resize(n2, (TW, TH), interpolation=cv2.INTER_CUBIC)
    n = 0.8 * n1 + 0.2 * n2
    n = (n - n.mean()) / (n.std() + 1e-6)
    # mip for the far rows (avoid aliasing where the plane recedes)
    nb = cv2.GaussianBlur(n, (0, 0), 6.0)
    ec = np.maximum(e, 0.02)
    Zc = 1.0 / ec
    mu = (rx * Zc * k + TW / 2).astype(np.float32)
    mv = (Zc * k - 1.0 * k).astype(np.float32)
    sa = cv2.remap(n.astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sb = cv2.remap(nb.astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    far = C.smoothstep(3.0, 7.0, Zc)
    d = sa * (1 - far) + sb * far
    # soft-edged masses with a crisper lower (lit) edge
    dens = C.smoothstep(-0.55, 0.6, d)
    # lower rim: density decreasing downward on screen
    sh = max(int(PH * 0.006), 2)
    below = np.empty_like(dens)
    below[:-sh] = dens[sh:]
    below[-sh:] = dens[-sh:]
    rim = np.clip(dens - below, 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), PH * 0.004)
    above = np.empty_like(dens)
    above[sh * 2:] = dens[:-sh * 2]
    above[:sh * 2] = dens[:sh * 2]
    topdark = np.clip(dens - above, 0, 1)
    fade = C.smoothstep(0.03, 0.14, e) * strength
    gx = np.exp(-((us / PW - glow_x) / 0.35) ** 2) + 0.6 * np.exp(-((us / PW - 0.8) / 0.25) ** 2)
    lowglow = np.exp(-np.clip(e, 0, None) / 0.35)
    body = np.array([0.028, 0.036, 0.075], np.float32)
    rimc = np.array([0.07, 0.06, 0.1], np.float32) + np.array([0.08, 0.045, 0.05], np.float32) * gx[..., None]
    add = dens[..., None] * body * (0.6 + 0.8 * lowglow[..., None]) + (rim * lowglow * 3.0)[..., None] * rimc \
        - (topdark * 0.5)[..., None] * np.array([0.02, 0.025, 0.05], np.float32)
    add = add * fade[..., None]
    return add.astype(np.float32), (dens * fade).astype(np.float32)


def town_ceiling(PW, PH, f, cx, cy, seed=5, glow_x=0.22):
    """The low, heavy snow-cloud ceiling of the 5cm snow night, lit orange-pink from below by the town:
    thick cloud masses (perspective plane above the camera) whose undersides glow salmon/rose near the
    horizon and fade to violet-navy overhead, darker lumpy tops, and deep indigo gaps.
    Returns an additive rgb (PH, PW, 3) and a multiplicative darkening (PH, PW)."""
    us, vs = C.grid(PW, PH)
    e = (cy - vs) / f
    rx = (us - cx) / f
    k = 120.0
    TW, TH = 2400, 1600
    n1 = C.fbm(TW // 4, TH // 4, 7.0, 5, seed=seed + 61)
    w1 = C.fbm(TW // 8, TH // 8, 3.0, 3, seed=seed + 62)
    w2 = C.fbm(TW // 8, TH // 8, 3.0, 3, seed=seed + 63)
    n1 = cv2.resize(n1, (TW, TH), interpolation=cv2.INTER_CUBIC)
    w1 = cv2.resize(w1, (TW, TH), interpolation=cv2.INTER_CUBIC)
    w2 = cv2.resize(w2, (TW, TH), interpolation=cv2.INTER_CUBIC)
    n = C.warp(n1, (w1 - 0.5) * 300, (w2 - 0.5) * 120)
    n = (n - n.mean()) / (n.std() + 1e-6)
    nb = cv2.GaussianBlur(n, (0, 0), 5.0)
    ec = np.maximum(e, 0.015)
    Zc = 1.0 / ec
    mu = (rx * Zc * k + TW / 2).astype(np.float32)
    mv = (Zc * k * 0.5).astype(np.float32)
    sa = cv2.remap(n.astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    sb = cv2.remap(nb.astype(np.float32), mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    far = C.smoothstep(4.0, 10.0, Zc)
    d = sa * (1 - far) + sb * far
    dens = C.smoothstep(-0.35, 0.35, d)
    # lit undersides: density falling off downward on screen = lower edge of a mass
    sh = max(int(PH * 0.008), 2)
    below = np.empty_like(dens)
    below[:-sh] = dens[sh:]
    below[-sh:] = dens[-sh:]
    under = np.clip(dens - below, 0, 1)
    under = cv2.GaussianBlur(under, (0, 0), PH * 0.006)
    body_grad = cv2.GaussianBlur(dens, (0, 0), PH * 0.02)
    inner = np.clip(dens - body_grad, -1, 1)          # >0 near the bottom-lit bulges
    ep = np.clip(e, 0, None)
    town = np.exp(-((us / PW - glow_x) / 0.32) ** 2) + 0.7 * np.exp(-((us / PW - 0.78) / 0.25) ** 2) + 0.25
    low = np.exp(-ep / 0.2)                           # town light reaches the low cloud most
    fade_in = C.smoothstep(0.012, 0.05, e)
    salmon = np.array([0.5, 0.22, 0.11], np.float32)
    rose = np.array([0.26, 0.1, 0.13], np.float32)
    violet = np.array([0.06, 0.05, 0.12], np.float32)
    lit_c = salmon[None, None] * low[..., None] + rose[None, None] * (np.exp(-ep / 0.35) - low).clip(0)[..., None] \
        + violet[None, None] * (1 - np.exp(-ep / 0.35))[..., None]
    add = (dens * 0.55 + under * 1.6 + np.clip(inner, 0, 1) * 0.6)[..., None] * lit_c * (town * fade_in)[..., None]
    dark = 1 - 0.25 * (1 - dens) * C.smoothstep(0.05, 0.4, e)   # gaps: deeper indigo
    return add.astype(np.float32), dark.astype(np.float32)
