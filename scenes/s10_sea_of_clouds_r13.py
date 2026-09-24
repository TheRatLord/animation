"""s10 helpers, round 13: a hand-painted sea of cumulus tops + painted towers (silhouettes grown with
lib.clouds2's cauliflower machinery; the value design is painted here as flat masses).

Every cloud top is ONE silhouette built from 2-6 broad dome heads on a flat-bottomed body, with
cauliflower florets grown only on the up / sun-facing edge (clouds2._grow, down_cut). It is painted
back to front as flat value masses:
  * crown: the sun-facing part of each head, bounded by the head's dome shifted away from the sun, so the
    lit cap is thick on top of a head and vanishes toward its flanks and in the notches between heads
    (never a band of constant width along the outline). 3 crisp steps: peach-gold -> gold -> hot
    white-gold toward the sun; the terminator is scalloped (lit heads bulging into the shadow) and carries
    a thin rose band;
  * body: cool blue-grey, a firm lighter skylit step on the up-facing shoulders, a deep saturated blue
    lower core; the base is torn into horizontal wisps that dissolve into the deep-blue valley between
    the tops;
  * a 2-3 px gold lining only on the sun-facing top silhouette.
Aerial perspective: the tops shrink, flatten, pale and lose contrast toward a hazy gold horizon.
Towers are 4 big masses each (warm-white lit face, gold/rose terminator, lavender shadow, deep lavender
underside, a skylit top step), crisp overlaps, a 2-4 px gold rim on the sunward silhouette, torn hazy feet.
"""
import math

import numpy as np
import cv2

from lib import core as C, clouds2 as K


def _c(h):
    return np.asarray(K._c(h), np.float32)


def _ss(e0, e1, x):
    return K._ss(e0, e1, x)


# ----------------------------------------------------------------------------------------- palettes
SEA = dict(hot=(1.3, 1.16, 0.9), crown=(1.0, 0.86, 0.6), crown_lo=(0.98, 0.7, 0.5), term=(0.82, 0.5, 0.6),
           sky=(0.6, 0.64, 0.87), body=(0.37, 0.44, 0.74), deep=(0.14, 0.2, 0.55), rim=(1.75, 1.36, 0.9))
SEA = {k: np.asarray(v, np.float32) for k, v in SEA.items()}
HAZE_SUN = np.asarray((1.0, 0.86, 0.66), np.float32)
HAZE_OFF = np.asarray((0.93, 0.78, 0.8), np.float32)

FLORETS = ((0.07, 0.14, 0.9, 1.1, 2.3), (0.028, 0.055, 0.75, 1.2, 2.6), (0.014, 0.024, 0.25, 1.6, 4.0))
TERM = ((0.08, 0.2, 0.95, 1.1, 2.2, 0.15, 0.5), (0.035, 0.07, 0.6, 1.3, 2.8), (0.015, 0.03, 0.3, 1.6, 3.4))


def floor_color(dy):
    """Valley / gap colour by depth below the horizon (fraction of H): hazy gold at the horizon ->
    lavender -> deep saturated blue near the camera."""
    k = np.clip(dy, 0, None)
    far = HAZE_SUN
    mid = _c('#8a8ccc')
    near = _c('#1f2d78')
    a = _ss(0.0, 0.05, k)[..., None]
    b = _ss(0.02, 0.16, k)[..., None]
    return ((far * (1 - a) + mid * a) * (1 - b) + near * b).astype(np.float32)


def haze_color(x, W, sx):
    p = math.exp(-((x - sx) / (0.35 * W)) ** 2)
    return HAZE_OFF * (1 - p) + HAZE_SUN * p


# ----------------------------------------------------------------------------------------- layout
def layout(W, H, hy, sx, seed=7, dy_min=0.004, dy_max=0.75, margin=0.14):
    """Scatter cumulus tops on the ground plane (screen size ~ depth below the horizon). Sorted far->near."""
    rng = np.random.default_rng(seed)
    out = []
    dy = dy_min
    k = 0
    while dy < dy_max:
        step = dy * rng.uniform(0.2, 0.3) + 0.0025
        cw0 = min(1.0 * W * dy, 0.26 * W) + 5 * W / 1920.0
        x = -margin * W - rng.uniform(0, 1) * cw0
        while x < (1 + margin) * W + cw0:
            r = float(np.clip(math.exp(rng.normal(0.0, 0.33)), 0.55, 1.8))
            cw = cw0 * r
            ddy = dy + rng.uniform(-0.5, 0.5) * step
            if ddy > 0.035 and rng.random() < 0.34 * _ss(0.035, 0.1, ddy) * (1 - _ss(0.18, 0.32, ddy)):
                x += cw * rng.uniform(0.6, 1.0)               # an open valley: deep blue floor shows
                continue
            cx = x + 0.5 * cw
            ang = math.degrees(math.atan2(abs(cx - sx), max(ddy * H + 0.05 * H, 1.0)))
            prox = math.exp(-(ang / 38.0) ** 2)
            asp = rng.uniform(0.3, 0.5)
            asp *= 0.55 + 0.45 * _ss(0.0, 0.12, ddy)             # far tops flatten into the haze sheet
            if 0.03 < ddy < 0.35 and rng.random() < 0.12 and prox < 0.4:
                asp *= rng.uniform(1.3, 1.6)                      # an occasional heaped turret
            asp *= 1.0 - 0.3 * _ss(0.2, 0.6, ddy)                # near tops are seen more from above
            if ddy < 0.2:
                asp *= 1.0 - 0.35 * prox                          # keep the sun's horizon open
            out.append(dict(cx=cx, by=hy + ddy * H, w=cw, h=cw * asp, dy=ddy, prox=prox, seed=k))
            k += 1
            x += cw * rng.uniform(0.5, 0.85)
        dy += step
    out.sort(key=lambda c: c['by'])
    return out


# ----------------------------------------------------------------------------------------- one top
def _to_f(M, w, h):
    m = M.astype(np.float32) / 255.0
    return K._resize(m, w, h, cv2.INTER_AREA) if M.shape[1] != w else m


def _heads(rng, cx, by, w, h, tier):
    """Dome heads (hx, hcy, rx, ry) of one tier. tier 0 = back (tall, lit crowns), 1 = front lobes."""
    if tier == 0:
        n = int(np.clip(round(w / (0.75 * h)), 1, 7))
        span = 0.4
    else:
        n = int(np.clip(round(w / (1.3 * h)), 0, 4))
        span = 0.36
    if n == 0:
        return []
    if n == 1:
        xs_ = np.array([rng.uniform(-0.08, 0.08)])
    else:
        xs_ = np.linspace(-span, span, n) + rng.uniform(-0.35, 0.35, n) * (2 * span / n)
    skew = rng.uniform(-0.25, 0.25)
    out = []
    for xi in xs_:
        prof = max(1.0 - 1.5 * (xi - skew) ** 2, 0.3)
        if tier == 0:
            r = h * rng.uniform(0.22, 0.5) * (0.7 + 0.4 * prof)
            top = by - h * (0.42 + 0.58 * prof * rng.uniform(0.6, 1.08))
        else:
            r = h * rng.uniform(0.22, 0.34)
            top = by - h * rng.uniform(0.3, 0.48)
        r = min(r, 0.5 * (by + 0.4 * h - top))
        rx = r * rng.uniform(1.05, 1.4) * min(1.0, 0.9 * w / max(n, 1) / (2 * r) + 0.45)
        out.append((cx + xi * w, top + r, rx, r))
    return out


def _draw_heads(M, heads, P, ss, bottom, dx=0.0, dy=0.0, cols=True, grow=1.0, gx=1.0):
    for (hx, hcy, rx, ry) in heads:
        px, py = P(hx + dx, hcy + dy)
        cv2.ellipse(M, (int(round(px)), int(round(py))), (max(int(rx * grow * gx * ss), 1), max(int(ry * grow * ss), 1)),
                    0, 0, 360, 255, -1, lineType=cv2.LINE_8)
        if cols:
            # the bulging column under the head reaches down to the base (rounded, tucked in at the bottom)
            pb = P(hx + dx, bottom + dy)
            ry2 = max(pb[1] - py, 1)
            cv2.ellipse(M, (int(round(px)), int(round(py))), (max(int(rx * 1.18 * grow * gx * ss), 1), int(ry2)),
                        0, 0, 180, 255, -1, lineType=cv2.LINE_8)


def paint_top(c, sun, W, H, u, pal=SEA, near_boost=0.0):
    """Paint one cumulus top: a back tier of tall heads (gold crowns) and a front tier of lower lobes
    (skylit rose-lavender tops) overlapping it with crisp edges; cool bodies, torn base.
    Returns (rgba straight float32, x0, y0)."""
    rng = np.random.default_rng(1000 + c['seed'])
    cx, by, w, h = c['cx'], c['by'], c['w'], max(c['h'], 1.5)
    dy, prox = c['dy'], c['prox']
    far = float(1 - _ss(0.0, 0.2, dy))                    # 1 at the horizon .. 0 from mid-distance on
    near = float(_ss(0.15, 0.5, dy))
    tiers = [_heads(rng, cx, by, w, h, 0)]
    if h > 10 * u:
        tiers.append(_heads(rng, cx, by, w, h, 1))
    allh = [hd for t in tiers for hd in t]
    pad = 0.22 * h + 4
    x0 = int(math.floor(min(hx - rx * 1.3 for hx, _, rx, _ in allh) - pad - 0.12 * w))
    x1 = int(math.ceil(max(hx + rx * 1.3 for hx, _, rx, _ in allh) + pad + 0.12 * w))
    y0 = int(math.floor(min(hcy - ry for _, hcy, _, ry in allh) - pad))
    bottom = by + 0.45 * h
    y1 = int(math.ceil(bottom + 3))
    bw, bh = x1 - x0, y1 - y0
    ss = 3 if h < 30 else 2
    SW, SH = bw * ss, bh * ss

    def P(x, y):
        return ((x - x0) * ss, (y - y0) * ss)

    sdx, sdy = sun[0] - cx, sun[1] - (by - 0.6 * h)
    ln = math.hypot(sdx, sdy) + 1e-6
    sdx, sdy = sdx / ln, min(sdy / ln, -0.6)
    ln = math.hypot(sdx, sdy)
    sdx, sdy = sdx / ln, sdy / ln
    uu = max(u, 0.3)
    ys = (np.arange(bh, dtype=np.float32)[:, None] + y0)
    acc = np.zeros((bh, bw, 4), np.float32)
    kc = (0.1 + 0.3 * prox ** 1.3 + 0.06 * near) * rng.uniform(0.6, 1.3)
    nlow = K._noise(bw, bh, max(3, bw / max(140 * uu, 1)), 50 + c['seed'], octaves=1, stretch=2.5)
    hz = haze_color(cx, W, sun[0])
    fa = far ** 1.2 * 0.9
    for ti, heads in enumerate(tiers):
        if not heads:
            continue
        M = np.zeros((SH, SW), np.uint8)
        _draw_heads(M, heads, P, ss, bottom)
        # a broad base joining the columns (no slits between the heads)
        xa = min(hx - 0.6 * rx for hx, _, rx, _ in heads)
        xb = max(hx + 0.6 * rx for hx, _, rx, _ in heads)
        if xb > xa:
            pc = P(0.5 * (xa + xb), by + 0.12 * h)
            cv2.ellipse(M, (int(pc[0]), int(pc[1])), (int(0.62 * (xb - xa) * ss), int(0.42 * h * ss)), 0, 0, 360, 255, -1)
        if h * ss > 6:
            K._grow(M, rng, h * ss, FLORETS if ti == 0 else FLORETS[:2], pref=(0.0, -1.0), down_cut=0.15,
                    side_scale=0.35, bulge=(0.3, 0.6), clump=0.55, sun_dir=(sdx, sdy), sun_bias=0.4, concave=0.7)

        def lit_mask(frac, scallop, kh=None):
            S_ = np.zeros_like(M)
            for hi_, (hx, hcy, rx, ry) in enumerate(heads):
                k_ = 1.0 if kh is None else kh[hi_]
                t = ry * frac * k_
                # weak heads: the shadow dome is grown so it also swallows their florets (only the lining
                # stays) - no lit fringe of constant width along every top
                g_ = 1.1 + 0.35 * max(0.0, 1.0 - k_)
                _draw_heads(S_, [(hx, hcy, rx, ry * 1.02)], P, ss, bottom + h, dx=-0.5 * sdx * t, dy=-sdy * t, gx=1.12,
                            grow=g_)
            L = cv2.bitwise_and(M, cv2.bitwise_not(S_))
            if scallop and h * ss > 14 and L.any():
                K._grow(L, rng, h * ss * 0.8, TERM, pref=(-sdx, -sdy), down_cut=0.3, side_scale=0.5, clump=0.5,
                        concave=0.5, fill=False)
                L = cv2.bitwise_and(L, M)
            return L

        m = _to_f(M, bw, bh)
        if ti == 0:
            # per-head crown size varies: some heads catch a big gold cap, some only a lining
            tops_ = np.array([hcy - ry for (_, hcy, _, ry) in heads])
            kh = rng.uniform(0.2, 0.7, len(heads)) + 0.3 * prox ** 1.5
            kh[int(np.argmin(tops_))] = rng.uniform(1.1, 1.6)       # the tallest head catches the big crown
            heads_k = [(hx, hcy, rx, ry) for (hx, hcy, rx, ry) in heads]
            l1 = _to_f(lit_mask(kc, True, kh), bw, bh)
            l3 = _to_f(lit_mask(kc * 0.3, False, kh), bw, bh) * (0.25 + 0.75 * prox)
        else:
            l1 = _to_f(lit_mask(0.45, True), bw, bh)
            l3 = np.zeros_like(l1)
        # values
        lowk = K._blur(((ys + (nlow - 0.5) * 0.25 * h) > by - 0.1 * h).astype(np.float32)
                       * np.ones((1, bw), np.float32), 0.8 * uu)[..., None]
        # body value by depth: lavender far away -> cool blue-grey -> deeper blue near the camera
        kb = float(_ss(0.0, 0.12, dy))
        bodyc = _c('#9a96d0') * (1 - kb) + pal['body'] * kb
        bodyc = bodyc * (1 - 0.5 * near) + _c('#34449a') * 0.5 * near
        if ti == 1:
            bodyc = bodyc * 0.9 + pal['deep'] * 0.1
        deep = pal['deep'] * (1 - 0.3 * near_boost) + np.array([0.08, 0.1, 0.36], np.float32) * 0.3 * near_boost
        body = bodyc * (1 - lowk) + deep * lowk
        img = np.broadcast_to(body, (bh, bw, 3)).astype(np.float32).copy()
        if ti == 0:
            # gold toward the sun axis, a rose-lilac skylit crown away from it
            kp = float(_ss(0.08, 0.7, prox))
            rose = np.asarray((0.96, 0.74, 0.72), np.float32) * (1 - 0.3 * near) + bodyc * 0.3 * near
            cr = rose * (1 - kp) + pal['crown'] * kp
            hot = cr * (1 - kp) + pal['hot'] * kp
            bandc = pal['term'] * kp + bodyc * (1 - kp)
        else:
            # front lobes sit in the back tier's shadow: skylit rose-lavender tops
            cr = bodyc * 0.45 + (pal['sky'] * 0.8 + pal['term'] * 0.2) * 0.55
            hot = cr
            bandc = bodyc
        tb_px = max(1.0, min(0.03 * h, 5 * uu))
        band = np.clip(K._blur(l1, tb_px) * 1.8 - l1, 0, 1) * (1 - l1) * m
        img = img * (1 - band[..., None] * 0.75) + bandc * band[..., None] * 0.75
        img = img * (1 - l1[..., None]) + cr * l1[..., None]
        img = img * (1 - l3[..., None]) + hot * l3[..., None]
        # gold lining along the sun-facing top silhouette of the back tier
        if ti == 0:
            rp = (1.2 + 1.6 * (1 - far)) * uu
            rim = np.clip(m - K._shift(m, sdx * rp, sdy * rp), 0, 1)
            rimc = pal['rim'] * (0.4 + 0.6 * kp) + np.asarray((1.0, 0.8, 0.78), np.float32) * (0.6 - 0.6 * kp)
            ra = 0.15 + 0.85 * kp
            img = img * (1 - rim[..., None] * ra) + rimc * (rim[..., None] * ra)
        # aerial perspective
        if fa > 0:
            img = img * (1 - fa) + hz * fa
            img += (l1[..., None] * np.array([0.07, 0.05, 0.01], np.float32)
                    - (1 - l1[..., None]) * np.array([0.06, 0.06, 0.0], np.float32)) * fa
        a = m.copy()
        _over_pm(acc, np.concatenate([img, a[..., None]], -1), 0, 0)
    # torn wispy base: horizontal streaks dissolving into the valley
    nt = K._noise(bw, bh, max(3, bw / max(110 * uu, 1)), 70 + c['seed'], octaves=4, stretch=5.0)
    yf = by + 0.08 * h + (nt - 0.5) * 0.6 * h
    fade = 1 - _ss(yf - 0.12 * h, yf + 0.1 * h, ys)
    acc *= fade[..., None]
    a = acc[..., 3:4]
    rgb = acc[..., :3] / np.maximum(a, 1e-5)
    return np.concatenate([rgb, np.clip(a, 0, 1)], -1).astype(np.float32), x0, y0


def _over_pm(acc, rgba, x0, y0):
    """Premultiplied 'over' of a straight-alpha patch into acc (h, w, 4 premultiplied), clipped."""
    h, w = rgba.shape[:2]
    H, W = acc.shape[:2]
    X0, Y0 = max(x0, 0), max(y0, 0)
    X1, Y1 = min(x0 + w, W), min(y0 + h, H)
    if X1 <= X0 or Y1 <= Y0:
        return
    s = rgba[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0]
    a = s[..., 3:4]
    d = acc[Y0:Y1, X0:X1]
    d *= (1 - a)
    d[..., :3] += s[..., :3] * a
    d[..., 3:4] += a


def _unpremult(acc):
    a = acc[..., 3:4]
    rgb = acc[..., :3] / np.maximum(a, 1e-5)
    # bleed colour into the transparent margin so bilinear resampling never darkens edges
    k = (a[..., 0] > 0.02).astype(np.float32)
    bl = cv2.GaussianBlur(acc[..., :3] * k[..., None], (0, 0), 3.0)
    bk = cv2.GaussianBlur(k * a[..., 0], (0, 0), 3.0)[..., None]
    fill = bl / np.maximum(bk, 1e-5)
    rgb = np.where(a > 0.02, rgb, fill)
    return np.concatenate([rgb, np.clip(a, 0, 1)], -1).astype(np.float32)


def paint_band(W, H, hy, sun, tops, dy0, dy1, pad=0.14, near_boost=0.0):
    sel = [c for c in tops if dy0 <= c['dy'] < dy1]
    if not sel:
        return None
    u = W / 1920.0
    ox = int(-pad * W)
    y_top = min(c['by'] - 1.4 * c['h'] for c in sel) - 6
    y_bot = max(c['by'] + 0.6 * c['h'] for c in sel) + 6
    oy = int(max(y_top, -pad * H))
    y1 = int(min(y_bot, H * (1 + pad)))
    pw, ph = int((1 + 2 * pad) * W), max(y1 - oy, 8)
    acc = np.zeros((ph, pw, 4), np.float32)
    for c in sel:
        nb = near_boost * float(_ss(0.3, 0.7, c['dy']))
        rgba, x0, y0 = paint_top(c, sun, W, H, u, near_boost=nb)
        _over_pm(acc, rgba, x0 - ox, y0 - oy)
    return dict(rgba=_unpremult(acc), ox=ox, oy=oy)


# ----------------------------------------------------------------------------------------- towers
TOWER_PAL = dict(hi=(1.0, 0.95, 0.86), lit=(0.99, 0.9, 0.76), lit_lo=(0.97, 0.76, 0.62), mid='#e0a0b0',
                 shade='#8c90d0', deep='#5a64b0', refl='#a4a8e0', bounce='#f0b8a8', rim=(1.7, 1.3, 0.8),
                 haze='#f4d0c4', edge_dark=0.04)


def _mix(a, b, t):
    return a * (1 - t) + b * t


def tower(W, H, sun, cx, base_y, Hc, seed, width=1.0, haze=0.0, pal=TOWER_PAL):
    """A towering cumulus painted with lib.clouds2 (big flat masses sharing one terminator: warm-white
    lit face toward the sun, rose terminator, lavender shadow, crisp overlapping masses) + an explicit
    2-4 px gold rim on the sunward silhouette and a foot that melts into the sea haze. Plate dict."""
    u = W / 1920.0
    pw, ph = int(1.05 * Hc * width + 0.2 * W), int(Hc * 1.25)
    ox, oy = int(cx - pw / 2), int(base_y - Hc * 1.12)
    by = base_y - oy
    sp = (sun[0] - ox, sun[1] - oy)
    rgba = K.cumulonimbus_plate(pw, ph, pw / 2, by, Hc, sun=sp, preset=pal, seed=seed, unit=u, anvil=False,
                                width=width * 1.25, sun_z=0.45, backlit=0.5, form=0.8, side=0.9, lining=1.6,
                                bounce_group=0.5, inner=1, term_w=0.06)
    A = rgba[..., 3]
    ys = np.arange(ph, dtype=np.float32)[:, None]
    sdx, sdy = sp[0] - pw / 2, sp[1] - (by - 0.5 * Hc)
    ln = math.hypot(sdx, sdy) + 1e-6
    sdx, sdy = sdx / ln, sdy / ln - 0.35
    ln = math.hypot(sdx, sdy)
    sdx, sdy = sdx / ln, sdy / ln
    rp = 3.0 * u
    rim = np.clip(A - K._shift(A, sdx * rp, sdy * rp), 0, 1)
    rim = np.clip(rim * 1.5, 0, 1) * _ss(by - 0.04 * Hc, by - 0.2 * Hc, ys)
    rgba[..., :3] = _mix(rgba[..., :3], np.asarray((1.85, 1.42, 0.86), np.float32), rim[..., None] * 0.9)
    if haze:
        rgba[..., :3] = _mix(rgba[..., :3], np.asarray((0.98, 0.84, 0.78), np.float32), haze)
    # the foot melts into the horizon haze
    fk = _ss(by - 0.3 * Hc, by + 0.02 * Hc, ys)[..., None]
    rgba[..., :3] = _mix(rgba[..., :3], np.asarray((1.0, 0.84, 0.72), np.float32), fk * 0.5)
    return dict(rgba=rgba, ox=ox, oy=oy)


# ----------------------------------------------------------------------------------------- cirrus
def cirrus(W, H, sun, seed=12, y1=0.45):
    """A few tapered cirrus wisps: each a soft ribbon swelling in the middle and tapering to hair-thin
    ends, with 1-3 fine companion strands; varied length, curvature, width and opacity."""
    rng = np.random.default_rng(seed)
    h = int(y1 * H)
    acc = np.zeros((h, W), np.float32)
    # (x0, y0, length, angle, bend, width, opacity, strands)  fractions of W / H
    wisps = [(0.2, 0.3, 0.34, -16, -0.05, 0.012, 0.5, 3), (0.55, 0.09, 0.26, 3, 0.03, 0.007, 0.28, 2),
             (0.64, 0.24, 0.3, -9, -0.035, 0.01, 0.42, 2), (0.05, 0.12, 0.18, -4, 0.02, 0.005, 0.22, 1),
             (0.83, 0.13, 0.16, 5, -0.015, 0.006, 0.3, 1), (0.4, 0.36, 0.12, -7, 0.01, 0.004, 0.2, 1)]
    for (x0, y0, L, ang, bend, wd, op, ns) in wisps:
        a = math.radians(ang)
        ca, sa = math.cos(a), math.sin(a)
        for si in range(ns + 1):
            main = si == 0
            n = 80
            tt = np.linspace(0, 1, n)
            s0 = 0.0 if main else rng.uniform(0.05, 0.35)
            lf = L * (1.0 if main else rng.uniform(0.35, 0.7))
            dist = (s0 * L + lf * tt) * W
            off = 0.0 if main else rng.normal(0, 1) * wd * W * 1.6
            curve = bend * H * np.sin(tt * math.pi) + off * (0.4 + tt)
            px = x0 * W + dist * ca - curve * sa
            py = y0 * H + dist * sa + curve * ca
            wmax = wd * W * (1.0 if main else rng.uniform(0.2, 0.4))
            prof = np.sin(np.clip(tt, 0, 1) * math.pi) ** 1.4 * (1 - 0.5 * tt)
            wv = wmax * prof
            nx, ny = -sa, ca
            left = np.stack([px + nx * wv * 0.5, py + ny * wv * 0.5], 1)
            right = np.stack([px - nx * wv * 0.5, py - ny * wv * 0.5], 1)
            poly = np.concatenate([left, right[::-1]], 0)
            tmp = np.zeros_like(acc)
            cv2.fillPoly(tmp, [np.round(poly * 4).astype(np.int32)], 1.0, lineType=cv2.LINE_AA, shift=2)
            o = op * (1.0 if main else rng.uniform(0.3, 0.6))
            np.maximum(acc, tmp * o, out=acc)
    u = W / 1920.0
    acc = cv2.GaussianBlur(acc, (0, 0), max(1.6 * u, 0.6))
    # comb the ribbons along their length (streaky, not scratchy): low-contrast stretched noise
    nz = K._noise(W, h, 6, seed + 3, octaves=3, stretch=0.12)
    acc = acc * (0.7 + 0.3 * nz)
    veil = cv2.GaussianBlur(acc, (0, 0), 10 * u) * 0.35
    al = np.clip(np.maximum(acc, veil), 0, 1)
    ys = np.arange(h, dtype=np.float32)[:, None] / H
    xs = np.arange(W, dtype=np.float32)[None, :] / W
    g = np.exp(-((xs - sun[0] / W) / 0.3) ** 2) * _ss(0.1, 0.4, ys)
    white = np.array([1.0, 0.97, 0.95], np.float32)
    gold = np.array([1.0, 0.84, 0.64], np.float32)
    lav = np.array([0.84, 0.86, 1.0], np.float32)
    hi = _ss(0.02, 0.3, ys)[..., None]
    col = lav * (1 - hi) + white * hi
    col = col * (1 - 0.6 * g[..., None]) + gold * 0.6 * g[..., None]
    col = np.broadcast_to(col, (h, W, 3))
    return np.concatenate([col, al[..., None]], -1).astype(np.float32)


# ----------------------------------------------------------------------------------------- foreground
def fg_fragment(W, H, sun, cx, by, w, h, seed):
    """A big torn cloud top near the camera (plate in its own coordinates, straight alpha)."""
    c = dict(cx=cx, by=by, w=w, h=h, dy=0.8, prox=0.15, seed=seed)
    rgba, x0, y0 = paint_top(c, sun, W, H, W / 1920.0, near_boost=0.6)
    return dict(rgba=rgba, ox=x0, oy=y0)
