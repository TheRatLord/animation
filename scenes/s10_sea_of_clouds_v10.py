"""s10 helpers (round 10): painted sea of clouds as depth-sorted SPRITES + backlit towers (lib/clouds2).

Sea of clouds
  * every bank is its own premultiplied RGBA sprite, anchored at a depth dz = (top - horizon) / H.
    Per frame each sprite is placed with a true forward-travel perspective about the vanishing point
    (scale 1 / (1 - G * p * dz): near banks swell and slide under the camera much faster than far
    ones) plus a lateral truck proportional to dz -> continuous multi-plane parallax;
  * silhouettes: a few BIG heads (broad, low domes of varied width) on a long mass with rounded ends,
    cauliflower florets grown only on the up-facing edge (clouds2._grow) - detail lives on the edge;
  * values painted as large masses: a broad sunlit TOP PLANE (hot gold / peach on the sun axis, pale
    pink-peach away from it) whose lower boundary is a soft, big-scalloped edge, a narrow rose
    terminator, then a cool lavender body deepening to indigo in the pocket below (soft feathered
    bottom - valleys are rounded shadow pockets, never cut edges); a 1.5-3 px crest line only on the
    up-facing silhouette, strongest near the sun axis;
  * aerial perspective: far banks get flat, low, wide, low-contrast and melt into a smooth pale haze
    sheet at the horizon (a separate full-width band sprite, no striping).
Towers: clouds2 Painter + cumulonimbus masses (no anvil) lit from BEHIND (sun_z < 0): cool blue-violet
bodies with a few large painted planes, the silver-gold lining only on the sun-facing silhouette, torn
wispy feet dissolving into the sea.
"""
import math

import numpy as np
import cv2

from lib import core as C, clouds2 as K

_ss = K._ss


def _lerp(a, b, t):
    return a + (b - a) * t


def _v(*c):
    return np.array(c, np.float32)


# ---------------------------------------------------------------------------------------------- palette
CROWN_ON = _v(1.16, 0.92, 0.62)      # hot gold crown on the sun axis (HDR -> blooms a touch)
CROWN_OFF = _v(1.0, 0.8, 0.74)       # pale pink-peach crown away from the sun
LOW_ON = _v(1.0, 0.66, 0.46)         # peach lower part of the lit plane (sun axis)
LOW_OFF = _v(0.92, 0.6, 0.7)         # pink lower part of the lit plane (off axis)
TERM = _v(0.8, 0.5, 0.74)            # rose-mauve terminator
LAV = _v(0.62, 0.55, 0.84)
BODY = _v(0.44, 0.39, 0.74)
DEEP = _v(0.2, 0.17, 0.48)
DEEPEST = _v(0.1, 0.09, 0.32)
HAZE_ON = _v(1.0, 0.86, 0.72)
HAZE_OFF = _v(0.94, 0.76, 0.84)
RIM_ON = _v(1.55, 1.22, 0.8)
RIM_OFF = _v(1.12, 0.84, 0.86)


def floor_col(dz, prox):
    """Colour of the deep valley floor at depth dz (0 horizon .. 0.5 bottom of frame)."""
    hz = _haze_amt(dz)
    base = _lerp(DEEP, DEEPEST, _ss(0.15, 0.5, dz))
    hc = _lerp(HAZE_OFF, HAZE_ON, prox)
    return _lerp(base, hc, hz * 0.95)


def _haze_amt(dz):
    return 0.9 * (1 - _ss(0.02, 0.26, dz)) ** 1.3


# ---------------------------------------------------------------------------------------------- utils
def _noise1(n, cells, seed):
    rng = np.random.default_rng(seed)
    k = max(int(cells), 2) + 3
    v = rng.random(k).astype(np.float32)
    x = np.linspace(0, cells, n, dtype=np.float32)
    i = np.floor(x).astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    return v[i] * (1 - f) + v[i + 1] * f


def _col_top(m):
    b = m > 0.5
    valid = b.any(0)
    top = np.where(valid, b.argmax(0), m.shape[0]).astype(np.float32)
    return top, valid


def _smooth_top(top, valid, sigma):
    wv = valid.astype(np.float32)
    num = cv2.GaussianBlur((top * wv)[None, :], (0, 0), sigmaX=max(sigma, 0.5), sigmaY=0.1)[0]
    den = cv2.GaussianBlur(wv[None, :], (0, 0), sigmaX=max(sigma, 0.5), sigmaY=0.1)[0]
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), top)


class Sprite:
    """Premultiplied RGBA canvas (h, w, 4) at supersampling ss, whose pixel (0, 0) sits at frame px
    (x0, y0) at rest; dz = depth anchor ((anchor_y - horizon) / H)."""

    def __init__(self, prem, x0, y0, ss, dz, kind='bank'):
        self.prem = np.ascontiguousarray(np.nan_to_num(prem, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32))
        self.x0, self.y0, self.ss, self.dz, self.kind = float(x0), float(y0), float(ss), float(dz), kind


class Camera:
    """Forward travel over the sea (vanishing point at (fx, hy)) + lateral truck."""

    def __init__(self, W, H, hy, fx, G=0.62, truck=0.07):
        self.W, self.H, self.hy, self.fx, self.G, self.truck = W, H, hy, fx, G, truck

    def xform(self, sp, p, t):
        """(a, bx, by): screen = a * sprite_px + (bx, by)."""
        dz = max(sp.dz, 0.0)
        s = 1.0 / (1.0 - self.G * p * min(dz, 0.9))
        shx = -self.truck * self.W * p * (dz / 0.5) - 0.004 * self.W * t * (0.3 + dz / 0.5)
        a = s / sp.ss
        bx = self.fx + (sp.x0 - self.fx) * s + shx
        by = self.hy + (sp.y0 - self.hy) * s
        return a, bx, by


def place(img, sp, a, bx, by, occ=None):
    """Composite sprite (premultiplied) onto img (H, W, 3) in place with screen = a * px + b."""
    H, W = img.shape[:2]
    h, w = sp.prem.shape[:2]
    X0, Y0 = int(math.floor(bx)) - 1, int(math.floor(by)) - 1
    X1, Y1 = int(math.ceil(bx + a * w)) + 1, int(math.ceil(by + a * h)) + 1
    X0c, Y0c, X1c, Y1c = max(X0, 0), max(Y0, 0), min(X1, W), min(Y1, H)
    if X1c <= X0c or Y1c <= Y0c:
        return
    M = np.array([[a, 0, bx - X0c], [0, a, by - Y0c]], np.float32)
    out = cv2.warpAffine(sp.prem, M, (X1c - X0c, Y1c - Y0c), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    al = np.clip(out[..., 3:4], 0, 1)
    reg = img[Y0c:Y1c, X0c:X1c]
    reg *= (1 - al)
    reg += out[..., :3]
    if occ is not None:
        o = occ[Y0c:Y1c, X0c:X1c]
        o += al[..., 0] * (1 - o)


# ---------------------------------------------------------------------------------------------- banks
def _bank_poly(x0, x1, top, hh, bottom, rng, head_w=(1.6, 4.2), head_h=(0.35, 1.0), p_end=2.6, n_per=0.2):
    """Top profile of a long mass: a few broad heads of varied width on a gently undulating line, ends
    rounded down to `bottom`. Returns closed polygon (canvas px)."""
    L = max(x1 - x0, 4.0)
    n = int(max(L / max(hh * n_per, 1.0), 80))
    xs = np.linspace(x0, x1, n)
    u = (xs - 0.5 * (x0 + x1)) / (0.5 * L)
    e = np.clip(1 - np.abs(u) ** p_end, 0, 1) ** (1.0 / p_end)
    prof = np.zeros(n, np.float64)
    x = x0 - rng.uniform(0.2, 1.0) * hh
    while x < x1 + hh:
        bw = hh * rng.uniform(*head_w)
        bh = hh * rng.uniform(*head_h) * min((bw / (2.6 * hh)) ** 0.5, 1.3)
        d = np.clip(1 - ((xs - x) / (bw / 2)) ** 2, 0, 1)
        prof = np.maximum(prof, bh * d ** (1 / 2.3))
        x += bw * rng.uniform(0.45, 0.85)
    und = _noise1(n, max(L / (6 * hh), 2), int(rng.integers(1 << 30))) - 0.5
    topl = top - prof - 0.35 * hh * und
    ytop = bottom - (bottom - topl) * e
    upper = np.stack([xs, ytop], 1)
    lower = np.stack([xs[::-1], np.full(n, bottom)], 1)
    return np.concatenate([upper, lower], 0).astype(np.float32)


def _levels(f):
    """Floret levels (fractions of size) by nearness f (0 far .. 1 near)."""
    if f < 0.2:
        return ((0.08, 0.16, 0.6, 1.4, 2.8, 0.2, 0.5),)
    if f < 0.55:
        return ((0.08, 0.18, 0.85, 1.2, 2.4, 0.15, 0.45), (0.03, 0.06, 0.5, 1.4, 3.0))
    return ((0.1, 0.22, 0.85, 1.2, 2.4, 0.15, 0.45), (0.04, 0.08, 0.6, 1.3, 2.8), (0.016, 0.03, 0.25, 1.6, 3.6))


def paint_bank(cv, parts, sun_x, dz, u, seed, lit_k=1.0):
    """Paint one or more overlapping masses (back to front) into canvas cv = dict(prem, x0, y0, ss).
    parts = [(x0, x1, top, hh, bottom)] in canvas px. sun_x in canvas px."""
    prem = cv['prem']
    Hc, Wc = prem.shape[:2]
    f = float(_ss(0.02, 0.45, dz))                   # nearness
    hz = float(_haze_amt(dz))
    rng = np.random.default_rng(seed)
    for pi, (x0, x1, top, hh, bottom) in enumerate(parts):
        crng = np.random.default_rng(int(rng.integers(1 << 31)))
        poly = _bank_poly(x0, x1, top, hh, bottom, rng,
                          head_w=(1.6, 4.0) if f > 0.3 else (2.5, 7.0),
                          head_h=(0.4, 1.4) if f > 0.3 else (0.2, 0.6))
        q = 2
        bx0 = int(max(math.floor(poly[:, 0].min() - hh), 0))
        by0 = int(max(math.floor(poly[:, 1].min() - hh), 0))
        bx1 = int(min(math.ceil(poly[:, 0].max() + hh), Wc))
        by1 = int(min(math.ceil(poly[:, 1].max() + 2), Hc))
        w, h = bx1 - bx0, by1 - by0
        if w < 4 or h < 4:
            continue
        Mk = np.zeros((h * q, w * q), np.uint8)
        cv2.fillPoly(Mk, [np.round((poly - [bx0, by0]) * q).astype(np.int32)], 255, lineType=cv2.LINE_8)
        size = hh * 1.3 * q
        if hh * f > 2.5 * u:
            K._grow(Mk, crng, size, _levels(f), pref=(0.0, -1.0), down_cut=0.05, side_scale=0.55, top_bias=1.4,
                    clump=0.5, concave=0.8)
        m = cv2.resize(Mk.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_AREA)
        if f < 0.12:
            m = K._blur(m, 0.8 * u)
        X = np.arange(w, dtype=np.float32) + bx0
        Y = np.arange(h, dtype=np.float32)[:, None] + by0
        prox_w = (0.07 + 0.35 * dz) * cv['W'] * cv['ss']
        prox = np.exp(-((X - sun_x) / prox_w) ** 2)[None, :]
        prox2 = np.exp(-((X - sun_x) / (1.8 * prox_w)) ** 2)[None, :]
        top_c, valid = _col_top(m)
        top_c = top_c + by0
        ts = _smooth_top(top_c, valid, 0.5 * hh)                  # follows the big heads, not florets
        tb = _smooth_top(top_c, valid, 2.5 * hh)
        nseed = int(rng.integers(1 << 30))
        wander = _noise1(w, max(w / (2.2 * hh), 2), nseed)
        softn = _ss(0.55, 0.85, _noise1(w, max(w / (3.0 * hh), 2), nseed + 1))
        pockn = 0.45 + 0.55 * _noise1(w, max(w / (2.5 * hh), 2), nseed + 2)
        # sunlit top plane: the lit cap of each big head is a crescent (deep under the crown, thin in
        # the saddles between heads); broad and gold on the sun axis, narrower and pinker off it
        hgt = np.clip(tb - ts, 0, None)
        Dl = hh * lit_k * (0.2 + 0.28 * prox2[0]) * (0.6 + 0.8 * wander) + (0.25 + 0.15 * prox2[0]) * hgt
        Dl = Dl * (1.1 - 0.2 * f)
        # form of the big heads: the flank turned toward the sun a touch lighter, the far flank darker
        slope = np.gradient(cv2.GaussianBlur(ts[None, :].astype(np.float32), (0, 0), sigmaX=max(0.6 * hh, 1),
                                             sigmaY=0.1)[0])
        tsun = np.tanh((sun_x - X) / (0.08 * cv['W'] * cv['ss']))
        form = np.clip(-slope * tsun * 1.5, -1, 1)[None, :]       # + = flank facing the sun
        dd = Y - ts[None, :]                                      # px below the smoothed top
        # lit -> lavender: a firm painted edge in places, a broad soft blend elsewhere
        bwid = np.maximum(hh * (0.02 + 0.12 * softn), 1.2 * u)[None, :]
        lit = _ss(Dl[None, :] + bwid, Dl[None, :] - bwid, dd)
        qd = np.clip((dd + 0.3 * hh) / np.maximum(Dl[None, :] + 0.3 * hh, 1.0), 0, 1)   # 0 crown .. 1 lower edge
        crown = _lerp(CROWN_OFF, CROWN_ON, (prox ** 0.8)[..., None])
        low = _lerp(LOW_OFF, LOW_ON, prox2[..., None])
        litc = _lerp(crown, low, _ss(0.2, 1.0, qd)[..., None])
        # body: rose terminator -> lavender -> blue-violet; rounded indigo pockets low down (vary along x)
        db = np.clip((dd - Dl[None, :]) / hh, 0, 4)
        deep = _lerp(DEEP, DEEPEST, f)
        s1 = (0.3 + 0.45 * _noise1(w, max(w / (2.0 * hh), 2), nseed + 4))[None, :]
        e1 = np.maximum(0.03 * hh, 1.2 * u) / hh
        body = _lerp(LAV, BODY, (0.75 * _ss(0.0, 0.8, db) + 0.25 * _ss(s1 - e1, s1 + e1, db))[..., None])
        pk = _ss(0.5 + 0.6 * (1 - pockn[None, :]), 1.8, db) * (0.55 + 0.45 * pockn[None, :]) * (0.35 + 0.65 * f)
        body = _lerp(body, deep, pk[..., None])
        tband = _ss(0.0, 0.05, db) * _ss(0.3, 0.05, db) * (0.25 + 0.35 * prox2) * (1 - 0.6 * softn[None, :])
        body = _lerp(body, TERM, tband[..., None])
        col = _lerp(body, litc, lit[..., None])
        # crest line: thin hot line on the up-facing silhouette only (sun axis strongest)
        r = u * (1.2 + 1.6 * prox[0]) * (0.7 + 0.5 * f)
        rim_w = float(np.clip(r.max(), 1.0, 6.0))
        up = K._shift(m, 0.0, -rim_w)
        rimm = _ss(0.25, 0.75, m * (1 - up)) * (0.2 + 0.8 * prox2)
        rimc = _lerp(RIM_OFF, RIM_ON, prox[..., None])
        col = _lerp(col, rimc, (rimm * (0.45 + 0.55 * f))[..., None])
        # big soft paint variation (static)
        bt = K._noise(w, h, max(w / (4 * hh), 2), nseed + 3, 3) - 0.5
        col = col * (1 + 0.06 * bt[..., None])
        col = col * (1 + (0.12 * form * (0.4 + 0.6 * (1 - lit)))[..., None])
        # aerial perspective
        hc = _lerp(HAZE_OFF, HAZE_ON, np.clip(prox2 * 1.1, 0, 1)[..., None])
        col = _lerp(col, hc, hz)
        # soft feathered bottom + the body melts away under the rounded ends (no cut walls)
        a = m * _ss(bottom, bottom - 0.45 * (bottom - ts[None, :]), Y)
        xe = np.minimum(X - x0, x1 - X)[None, :]
        endf = _ss(0.0, 2.0 * hh, xe)
        a = a * (lit + (1 - lit) * endf)
        a = np.clip(a, 0, 1).astype(np.float32)
        reg = prem[by0:by1, bx0:bx1]
        reg[..., :3] = reg[..., :3] * (1 - a[..., None]) + col * a[..., None]
        reg[..., 3] = reg[..., 3] * (1 - a) + a


def make_sea(W, H, hy, sun_x, seed=11):
    """Depth-sorted list of bank sprites (far -> near)."""
    rng = np.random.default_rng(seed)
    sprites = []
    u0 = W / 1920.0
    N = 9
    d0, d1 = 0.075, 0.6
    dzs = [d0 * (d1 / d0) ** (i / (N - 1)) for i in range(N)]
    ratio = (d1 / d0) ** (1 / (N - 1))
    for i, dz0 in enumerate(dzs):
        spacing = dz0 * (ratio - 1)
        f0 = float(_ss(0.02, 0.45, dz0))
        x = -0.3 * W - rng.uniform(0, 0.3) * W
        while x < 1.25 * W:
            L = W * (rng.uniform(0.9, 1.7) if f0 < 0.3 else rng.uniform(0.5, 1.15))
            dz = dz0 + rng.uniform(-0.5, 0.4) * spacing
            f = float(_ss(0.02, 0.45, dz))
            if rng.random() < 0.12 and i > 1:
                x += L * rng.uniform(0.3, 0.6)
                continue
            k = rng.uniform(0.6, 1.8) ** 1.2
            hh_f = (0.003 + 0.17 * dz) * H * k
            top_f = hy + dz * H
            bottom_f = top_f + max(3.2 * hh_f, 2.0 * spacing * H)
            ss = 1.0 + 0.6 * float(_ss(0.12, 0.4, dz))
            u = u0 * ss
            parts_f = [(x, x + L, top_f, hh_f, bottom_f)]
            # near masses: a lower, smaller heap in front (crisp overlapping-mass edge, own lit plane)
            if f > 0.5 and rng.random() < 0.6:
                Ls = L * rng.uniform(0.35, 0.6)
                xs0 = x + rng.uniform(0.05, 0.95) * (L - Ls)
                hs = hh_f * rng.uniform(0.55, 0.8)
                ts_ = top_f + hh_f * rng.uniform(1.0, 1.5)
                parts_f.append((xs0, xs0 + Ls, ts_, hs, bottom_f + 0.3 * hh_f))
            X0 = min(pp[0] for pp in parts_f) - 1.5 * hh_f
            X1 = max(pp[1] for pp in parts_f) + 1.5 * hh_f
            Y0 = top_f - 2.8 * hh_f
            Y1 = max(pp[4] for pp in parts_f) + 2
            cw, ch = int((X1 - X0) * ss) + 2, int((Y1 - Y0) * ss) + 2
            cv = dict(prem=np.zeros((ch, cw, 4), np.float32), W=W, ss=ss)
            parts = [((a0 - X0) * ss, (a1 - X0) * ss, (tp - Y0) * ss, hh * ss, (bt - Y0) * ss)
                     for (a0, a1, tp, hh, bt) in parts_f]
            paint_bank(cv, parts, (sun_x - X0) * ss, dz, u, int(rng.integers(1 << 30)))
            sprites.append(Sprite(cv['prem'], X0, Y0, ss, dz))
            x += L * rng.uniform(0.5, 0.85)
    sprites.sort(key=lambda s: s.dz)
    return sprites


def skirt(W, H, hy, sun_x, cx, width, dz, seed):
    """A bank sprite passing in front of a tower's foot (the tower rises out of the sea)."""
    rng = np.random.default_rng(seed)
    u0 = W / 1920.0
    hh_f = (0.003 + 0.17 * dz) * H * 1.6
    top_f = hy + dz * H
    bottom_f = top_f + 3.5 * hh_f
    x0, x1 = cx - 0.5 * width, cx + 0.5 * width
    X0, X1 = x0 - 1.5 * hh_f, x1 + 1.5 * hh_f
    Y0, Y1 = top_f - 2.8 * hh_f, bottom_f + 2
    ss = 1.0
    cv = dict(prem=np.zeros((int(Y1 - Y0) + 2, int(X1 - X0) + 2, 4), np.float32), W=W, ss=ss)
    paint_bank(cv, [(x0 - X0, x1 - X0, top_f - Y0, hh_f, bottom_f - Y0)], sun_x - X0, dz, u0,
               int(rng.integers(1 << 30)))
    return Sprite(cv['prem'], X0, Y0, ss, dz)


def haze_band(W, H, hy, sun_x, seed=5):
    """Far sea: smooth pale luminous sheet from the horizon down to ~0.08 H, faint long low ribbons of
    value (compressed far banks), no silhouettes but a softly bumpy horizon line. Returns a Sprite."""
    x0 = -0.3 * W
    w = int(1.6 * W)
    y0 = hy - 0.02 * H
    h = int(0.16 * H)
    X = np.arange(w, dtype=np.float32)[None, :] + x0
    Y = np.arange(h, dtype=np.float32)[:, None] + y0
    dz = np.clip((Y - hy) / H, 0, None)
    prox = np.exp(-((X - sun_x) / (0.22 * W)) ** 2)
    prox2 = np.exp(-((X - sun_x) / (0.45 * W)) ** 2)
    # horizon line with very soft low bumps
    bump = (K._noise(w, 8, w / (0.05 * W), seed, 3)[4] - 0.5) * 0.004 * H
    top = hy + bump[None, :]
    a = _ss(top - 0.6 * W / 1920, top + 1.2 * W / 1920, Y)
    a = a * (1 - _ss(hy + 0.1 * H, hy + 0.14 * H, Y))
    # ribbons: horizontal value streaks compressed toward the horizon (perspective)
    zz = np.log(np.maximum(dz, 1e-4) * H / (0.004 * H) + 1.0)
    rib = K._noise(w, h, 3.0, seed + 3, 3) - 0.5
    lines = 0.5 + 0.5 * np.sin(zz * 9.0 + 3.0 * rib)
    lines = _ss(0.55, 0.95, lines)
    base = _lerp(HAZE_OFF, HAZE_ON, prox2[..., None])
    base = _lerp(base, _v(1.08, 0.94, 0.8), (prox * _ss(0.03, 0.0, dz))[..., None])
    under = _lerp(_v(0.74, 0.6, 0.8), _v(0.9, 0.7, 0.72), prox2[..., None])
    kd = _ss(0.0, 0.11, dz)
    col = _lerp(base, under, (kd * (0.35 + 0.35 * lines) * (1 - 0.5 * prox))[..., None])
    col = col + (_v(0.12, 0.08, 0.04) * (lines * (1 - kd) * 0.6)[..., None]) * (0.4 + 0.6 * prox)[..., None]
    prem = np.dstack([col * a[..., None], a]).astype(np.float32)
    return Sprite(prem, x0, y0, 1.0, 0.0, kind='band')


# ---------------------------------------------------------------------------------------------- towers
TOWER_PAL = dict(hi=(1.0, 0.9, 0.7), lit=(0.98, 0.74, 0.58), lit_lo='#e8a0a4', mid='#b87aa8', shade='#6a66b6',
                 deep='#48449a', refl='#8480c8', bounce='#c896b0', rim=(1.7, 1.42, 1.0), haze='#e6b8c8',
                 edge_dark=0.04)


def tower(W, H, hy, cx, base_y, Hc, sun, seed=33, width=0.9, haze=0.0, ss=1.0, sun_z=-0.35, lean_pal=None,
          lining=1.4):
    """Backlit towering cumulus sprite. Returns Sprite (dz from its base)."""
    u = W / 1920.0 * ss
    X0 = cx - 0.95 * Hc * width
    X1 = cx + 0.95 * Hc * width
    Y0 = base_y - 1.1 * Hc
    Y1 = base_y + 0.2 * Hc
    pw, ph = int((X1 - X0) * ss), int((Y1 - Y0) * ss)
    sun_l = ((sun[0] - X0) * ss, (sun[1] - Y0) * ss)
    pal = TOWER_PAL if lean_pal is None else lean_pal
    P = K.Painter(pw, ph, sun=sun_l, pal=pal, unit=u, seed=seed, sun_z=sun_z)
    cxl, byl, Hl = (cx - X0) * ss, (base_y - Y0) * ss, Hc * ss
    K.cumulonimbus(P, cxl, byl, Hl, seed=seed, anvil=False, pal=pal, backlit=0.35, width=width, sun_z=sun_z,
                   wisps=False, form=0.75, bounce_group=0.25, wrap=0.0, term_w=0.05, side=0.85, inner=0,
                   rim_interior=0.0)
    P.wisps(cxl - 0.6 * Hl * width, cxl + 0.6 * Hl * width, byl - 0.02 * Hl, 0.12 * Hl,
            rng=np.random.default_rng(seed + 5), pal=pal, seed=seed + 7, amount=0.2, erode=0.9, base_haze=0.12,
            haze_color='#c8a4cc')
    # a few large painted value planes: light scattering through the thin sunward half of the tower
    # (a broad lighter, warmer plane inside the lining), the crown a touch lighter from the open sky,
    # the foot cooler and deeper
    A = np.clip(P.prem[..., 3], 0, 1)
    rgb = P.prem[..., :3] / np.maximum(A, 1e-5)[..., None]
    ys, xs = np.mgrid[0:ph, 0:pw].astype(np.float32)
    dx, dy = sun_l[0] - xs, sun_l[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl

    def toward(img, r):
        return cv2.remap(img, xs + ux * r, ys + uy * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    gw = 0.07 * Hl
    g2 = K._ss(0.3, 0.7, A * (1 - K._blur(toward(A, gw), 0.25 * gw)))            # narrow sunward band
    g3 = K._ss(0.2, 0.6, A * (1 - K._blur(toward(A, 3.0 * gw), 1.0 * gw)))       # broad sunward plane
    vv = np.clip((ys - (byl - Hl)) / Hl, 0, 1)
    rgb = rgb * (1 + 0.08 * (1 - K._ss(0.0, 0.6, vv)))[..., None]
    rgb = _lerp(rgb, rgb * _v(0.86, 0.88, 1.0), K._ss(0.55, 1.0, vv)[..., None] * 0.6)
    rgb = _lerp(rgb, _v(0.7, 0.6, 0.86), (g3 * 0.22)[..., None])
    rgb = _lerp(rgb, _v(0.95, 0.68, 0.72), (g2 * 0.25)[..., None])
    P.prem[..., :3] = rgb * A[..., None]
    P.lining(lining * 1.8, rim_px=2.4, halo=0.3, backlit=0.35, y_max=byl - 0.05 * Hl,
             pal=dict(pal, rim=(1.9, 1.62, 1.15)))
    if haze:
        P.haze_band(byl - Hl, byl, K._c('#e8c0c8'), haze)
    dz = (base_y - hy) / H - 0.03
    return Sprite(P.prem.copy(), X0, Y0, ss, dz, kind='tower')
