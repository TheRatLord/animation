"""s10 helpers (round 9): painted sea of clouds + backlit towers on lib/clouds2 geometry.

Silhouettes come from clouds2 (billow_strip / envelope -> cauliflower: detail on the EDGE only); the
values are painted here as a few LARGE value masses per bank instead of a crest outline:
  * sunlit TOP MASS: the upper part of every bank facing the sun is a broad gold / peach / pink mass
    whose lower boundary is a soft, big-scalloped edge (it follows the heads, deeper under the taller
    heads and wandering along the bank - never a constant-width band); a hotter core near the top on
    the sun axis; a thin 2-4 px hot line only on the very crest, thinner / cooler away from the sun;
  * cool lavender body mid-tones below, deepening to indigo low in each bank, so the visible bottom of
    every bank (just above the crest of the bank in front) is a dark trough -> depth;
  * a per-layer deep indigo valley floor behind the banks (darkest in the foreground);
  * aerial perspective: far banks lose contrast and edge detail and melt into a pale luminous
    peach-white haze sheet, with a static sun glitter on the crests under the sun.
Each plate also carries a 5th channel E = 'extra gold': the zone that turns gold as the light swells at
the end of the shot (the scene lerps toward gold by E * ramp(t)).
Towers: flat masses (clouds2 envelope + cauliflower) painted as cool lavender-blue bodies, crisp
overlapping-mass edges (skylit tops of each mass + a soft cast shade on what is behind), a translucent
gold glow along the SUNWARD silhouette of the whole union with a silver-gold lining (Painter.lining),
torn wispy feet (Painter.wisps) that sink into the banks.
"""
import math

import numpy as np
import cv2

from lib import core as C, clouds2 as K

M = 0.05          # plate margin (fraction of W / H per side) - must match s10_sea_of_clouds_c2.M

_c = K._c
GOLD_HOT = np.array([1.22, 0.98, 0.66], np.float32)
GOLD = np.array([1.0, 0.76, 0.44], np.float32)
PEACH = np.array([0.99, 0.64, 0.5], np.float32)
PINK = np.array([0.9, 0.52, 0.66], np.float32)
LAV = np.array([0.6, 0.53, 0.82], np.float32)
BODY = np.array([0.42, 0.38, 0.72], np.float32)
DEEP = np.array([0.2, 0.17, 0.46], np.float32)
DEEPEST = np.array([0.1, 0.09, 0.3], np.float32)
HAZE_SUN = np.array([1.0, 0.87, 0.74], np.float32)
HAZE_OFF = np.array([0.92, 0.74, 0.84], np.float32)


class Plate:
    """RGBA(+E) premultiplied canvas covering frame + margin at supersampling s (frame px in)."""

    def __init__(self, W, H, s, sun, seed):
        self.W, self.H, self.s = W, H, s
        self.pw, self.ph = int(round(W * (1 + 2 * M) * s)), int(round(H * (1 + 2 * M) * s))
        self.u = W / 1920.0 * s
        self.prem = np.zeros((self.ph, self.pw, 5), np.float32)
        self.sun = self.xy(sun)
        self.seed = seed

    def xy(self, p):
        return ((p[0] + M * self.W) * self.s, (p[1] + M * self.H) * self.s)

    def X(self, x):
        return (x + M * self.W) * self.s

    def Y(self, y):
        return (y + M * self.H) * self.s

    def over(self, x0, y0, rgb, a, e=None):
        """Composite local straight rgb (h, w, 3) with alpha a (h, w) at plate px (x0, y0), clipped."""
        h, w = a.shape
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, self.pw), min(y0 + h, self.ph)
        if X1 <= X0 or Y1 <= Y0:
            return
        sl = (slice(Y0 - y0, Y1 - y0), slice(X0 - x0, X1 - x0))
        aa = a[sl][..., None]
        reg = self.prem[Y0:Y1, X0:X1]
        reg[..., :3] = reg[..., :3] * (1 - aa) + rgb[sl] * aa
        reg[..., 3:4] = reg[..., 3:4] * (1 - aa) + aa
        ee = 0.0 if e is None else e[sl][..., None]
        reg[..., 4:5] = reg[..., 4:5] * (1 - aa) + ee * aa

    def rgba(self):
        """Straight RGBA + E (5 ch); colours bled into the transparent margin (no dark fringes)."""
        a = self.prem[..., 3:4]
        rgb = self.prem[..., :3] / np.maximum(a, 1e-5)
        e = self.prem[..., 4:5] / np.maximum(a, 1e-5)
        pb = cv2.GaussianBlur(self.prem[..., :4], (0, 0), max(2.5 * self.u, 1.0))
        fill = pb[..., :3] / np.maximum(pb[..., 3:4], 1e-5)
        wgt = np.clip(a * 6.0, 0, 1)
        rgb = rgb * wgt + fill * (1 - wgt)
        return np.concatenate([rgb, np.clip(a, 0, 1), np.clip(e, 0, 1)], -1).astype(np.float32)


def _lerp(a, b, t):
    return a + (b - a) * t


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
    """Row index of the first covered pixel per column (h where empty) + validity."""
    b = m > 0.5
    valid = b.any(0)
    top = np.where(valid, b.argmax(0), m.shape[0]).astype(np.float32)
    return top, valid


def _smooth_top(top, valid, sigma):
    wv = valid.astype(np.float32)
    num = cv2.GaussianBlur((top * wv)[None, :], (0, 0), sigmaX=max(sigma, 0.5), sigmaY=0.1)[0]
    den = cv2.GaussianBlur(wv[None, :], (0, 0), sigmaX=max(sigma, 0.5), sigmaY=0.1)[0]
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), top)


def _drop(m, D, ox=0.0):
    """m sampled D[x] px ABOVE (and ox px sunward) each pixel: 1 - it = 'within D of the top edge'."""
    h, w = m.shape
    xs = np.arange(w, dtype=np.float32)[None, :] + np.float32(ox)
    ys = np.arange(h, dtype=np.float32)[:, None] - D[None, :].astype(np.float32)
    return cv2.remap(m, np.broadcast_to(xs, (h, w)).astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ------------------------------------------------------------------------------------------ banks
def bank(pl, poly, size, levels, rng, f, prox_w, hz, lit_depth, deep_col, glitter=0.0, rim=1.0, dark=1.0,
         soft=0.0, bump_w=0.0):
    """Paint one cloud-sea bank on plate pl. poly in plate px; f = nearness 0 far .. 1 near;
    prox_w = sun-proximity width (plate px); hz = aerial haze 0..1; lit_depth = depth of the sunlit
    top mass (plate px, on the sun axis); deep_col = colour of the bank's lowest part."""
    u = pl.u
    crng = np.random.default_rng(int(rng.integers(1 << 31)))
    nseed = int(rng.integers(1 << 30))
    sp = np.array(pl.sun, np.float32) - poly.mean(0)
    sp = sp / (float(np.hypot(*sp)) + 1e-6)
    m, x0, y0 = K.cauliflower(poly, crng, size, levels, down_cut=0.05, side_scale=0.7, top_bias=1.4,
                              concave=0.8, sun_dir=sp, sun_bias=0.2, clump=0.45)
    if m.size == 0 or m.max() <= 0:
        return
    pad = int(lit_depth * 2.5 + 8 * u)
    m = cv2.copyMakeBorder(m, pad, 0, 4, 4, cv2.BORDER_CONSTANT, value=0)
    x0 -= 4
    y0 -= pad
    h, w = m.shape
    X = np.arange(w, dtype=np.float32) + x0
    sx, sy = pl.sun
    prox = np.exp(-((X - sx) / prox_w) ** 2)                      # (w,) sun axis
    prox2 = np.exp(-((X - sx) / (1.7 * prox_w)) ** 2)
    top, valid = _col_top(m)
    bw = max(size, 4.0)
    tops = _smooth_top(top, valid, 1.2 * bw)
    # big heads (band-pass of the top profile at the bump scale; florets ignored): the lit cap of
    # each head is a crescent - deep at the head's crown, vanishing in the valleys between heads
    bwid = max(bump_w, 4 * u)
    t_small = _smooth_top(top, valid, 0.1 * bwid)
    t_big = _smooth_top(top, valid, 0.7 * bwid)
    heads = np.clip((t_big - t_small) / (0.5 * lit_depth + 1e-3), 0, 1.5)
    wander = _noise1(w, max(w / (1.5 * bwid), 2), nseed)
    Dl = lit_depth * (0.3 + 0.7 * prox2) * (0.7 + 0.6 * wander) * (0.12 + 1.1 * heads) + 1.5 * u
    ox = float(np.clip((sx - (x0 + w / 2)) / (6 * prox_w), -1, 1)) * 0.3 * lit_depth
    sig = lambda k: max(k, 0.6)
    def within(k, sg):
        return m * (1 - K._blur(_drop(m, k * Dl, k * ox), sig(sg * lit_depth)))
    w1, w2, w3 = within(0.35, 0.05), within(0.7, 0.08), within(1.0, 0.1)
    w4 = within(1.4, 0.15)
    st = lambda x: K._ss(0.3, 0.7, x)
    cap = st(w3)                                                    # the lit cap (firm painted edge)
    lit = cap
    hot = st(w1) * (prox ** 1.2)[None, :]
    widez = np.clip(0.8 * cap + 0.2 * st(w4), 0, 1)                     # warms / brightens as the light swells
    # body: lavender -> blue-violet -> indigo low in the bank (the trough above the next bank's crest)
    Y = np.arange(h, dtype=np.float32)[:, None]
    dd = (Y - tops[None, :]) / max(1.8 * bw, 1.0)
    body = _lerp(LAV, BODY, K._ss(0.0, 0.55, dd)[..., None])
    body = _lerp(body, deep_col, (K._ss(0.35, 1.15, dd) * dark)[..., None])
    col = body
    P2 = prox2[None, :, None]
    P1 = (prox ** 0.8)[None, :, None]
    amt = (0.6 + 0.4 * prox2)[None, :, None]
    # saturated rose terminator just under the cap, then the cap: peach at its lower edge -> luminous
    # cream-gold at the crown (on the sun axis) / pale pink-cream away from it
    rose = _lerp(np.array([0.8, 0.5, 0.74], np.float32), np.array([0.94, 0.55, 0.62], np.float32), P2)
    col = _lerp(col, rose, np.clip(0.5 * w4 + 0.5 * st(w4) - cap, 0, 1)[..., None] * amt * 0.5)
    low = _lerp(np.array([0.93, 0.64, 0.76], np.float32), np.array([1.0, 0.7, 0.56], np.float32), P2)
    crown = _lerp(np.array([1.0, 0.84, 0.88], np.float32), np.array([1.05, 0.88, 0.72], np.float32), P2)
    crown = _lerp(crown, np.array([1.14, 0.98, 0.78], np.float32), P1)
    capc = _lerp(low, crown, K._ss(0.1, 0.9, 0.6 * w1 + 0.4 * w2)[..., None])
    col = _lerp(col, capc, cap[..., None] * amt)
    col = _lerp(col, GOLD_HOT, (hot * 0.5)[..., None])
    # crest line: thin hot rim toward the sun, only on the top silhouette; thinner / cooler off-axis
    r = u * (1.2 + 1.8 * prox2) * (0.8 + 0.6 * f)
    rimm = m * (1 - _drop(m, r, 0.25 * ox / max(lit_depth, 1) * float(r.mean())))
    rimm = K._ss(0.25, 0.75, rimm) * (0.2 + 0.8 * prox2)[None, :] * rim
    rimc = _lerp(np.array([1.1, 0.8, 0.8], np.float32), np.array([1.5, 1.2, 0.8], np.float32),
                 prox[None, :, None])
    col = _lerp(col, rimc, rimm[..., None])
    if glitter > 0:
        g = K._noise(w, h, max(w / (0.25 * bw), 3), nseed + 5, 2)
        gl = K._ss(0.62, 0.75, g) * rimm * (prox ** 2)[None, :] * glitter
        col = col + np.array([1.6, 1.3, 0.9], np.float32) * gl[..., None]
    # brush texture (big soft value variation, static)
    bt = K._noise(w, h, max(w / (3 * bw), 2), nseed + 9, 3) - 0.5
    col = col * (1 + 0.06 * bt[..., None])
    # aerial perspective: toward a pale luminous peach-white on the sun axis, lavender-pink elsewhere
    hc = _lerp(HAZE_OFF, HAZE_SUN, np.clip(prox2 * 1.2, 0, 1)[None, :, None])
    col = _lerp(col, hc, hz)
    a = m
    if soft > 0:
        a = K._blur(m, soft) * 0.7 + m * 0.3
    e = widez * (0.2 + 0.8 * prox2)[None, :] * (1 - 0.7 * hz)
    pl.over(x0, y0, col.astype(np.float32), a.astype(np.float32), e.astype(np.float32))


def floor(pl, y_top, deep_top, deep_bot, hz, seed, amp):
    """Valley floor of a layer: opaque from an undulating line y_top (frame px) to the plate bottom."""
    ys = np.arange(pl.ph, dtype=np.float32)[:, None]
    xs = np.arange(pl.pw, dtype=np.float32)
    rng = np.random.default_rng(seed)
    wav = sum(np.sin(xs / pl.pw * fq * 6.283 + rng.uniform(0, 6.28)) * a
              for fq, a in ((1.3, 1.0), (3.1, 0.5), (7.3, 0.25)))
    yt = pl.Y(y_top) + wav[None, :] * amp * pl.s
    d = (ys - yt) / (0.12 * pl.H * pl.s)
    a = K._ss(-0.25, 0.25, d)
    col = _lerp(deep_top, deep_bot, K._ss(0.0, 1.0, d)[..., None])
    sx = pl.sun[0]
    prox = np.exp(-((xs - sx) / (0.35 * pl.W * pl.s)) ** 2)[None, :, None]
    hc = _lerp(HAZE_OFF, HAZE_SUN, prox)
    col = _lerp(col, hc, hz)
    pl.over(0, 0, col.astype(np.float32), np.broadcast_to(a, (pl.ph, pl.pw)).astype(np.float32))


# rows: y below horizon (fraction of H), geometric from the horizon to the bottom of the frame
def rows(n=17, d0=0.0035, d1=0.62):
    return [d0 * (d1 / d0) ** (i / (n - 1)) for i in range(n)]


LAYER_EDGES = (0.012, 0.03, 0.07, 0.14, 0.28)     # row dy thresholds -> layer index 0..5
NL = 6


def layer_of(dy):
    for k, e in enumerate(LAYER_EDGES):
        if dy < e:
            return k
    return NL - 1


def _levels(f):
    if f < 0.25:
        return ((0.05, 0.12, 0.8, 1.1, 2.0), (0.02, 0.04, 0.6))
    if f < 0.6:
        return ((0.08, 0.2, 0.85, 1.3, 2.6, 0.1, 0.45), (0.03, 0.07, 0.6, 1.3, 2.8), (0.012, 0.025, 0.3, 1.5, 3.5))
    return ((0.1, 0.24, 0.9, 1.3, 2.6, 0.1, 0.45), (0.04, 0.085, 0.55, 1.2, 2.6), (0.016, 0.03, 0.25, 1.4, 3.2))


def sea(W, H, hy, sun, scales=(1.0, 1.0, 1.0, 1.15, 1.35, 1.7), seed=7, layer_cb=None):
    """Paint the sea into NL plates (far .. near). layer_cb(k, y, plate) is called after the floor of
    each layer is painted and before its banks (towers standing in that layer)."""
    rng = np.random.default_rng(seed)
    plates = [Plate(W, H, scales[k], sun, seed + 10 + k) for k in range(NL)]
    R = rows()
    n = len(R)
    started = set()
    for ri, dyf in enumerate(R):
        k = layer_of(dyf)
        pl = plates[k]
        s = pl.s
        f = ri / (n - 1)
        y = hy + dyf * H
        bw = (0.028 + 0.8 * dyf) * W
        bh = (0.0028 + 0.17 * dyf) * H
        hz = 0.82 * (1 - K._ss(0.0, 0.55, f)) ** 1.4
        deep = _lerp(DEEP, DEEPEST, float(K._ss(0.5, 1.0, f)))
        if k not in started:
            started.add(k)
            dt = _lerp(BODY, DEEP, 0.5 + 0.5 * f)
            floor(pl, y + 0.4 * bh, dt, _lerp(DEEP, DEEPEST, f), hz * 0.9, seed + k, 0.003 * H * (0.3 + f))
            if layer_cb is not None:
                layer_cb(k, y, pl)
        x = -0.1 * W - rng.uniform(0, 0.15) * W
        segs = []
        while x < 1.1 * W:
            L = rng.uniform(0.35, 0.8) * W * (0.8 + 0.4 * f)
            if rng.random() > 0.06 or f < 0.3:
                segs.append((x, x + L, y + rng.uniform(-0.4, 0.3) * bh, rng.uniform(0.75, 1.3)))
            x += L * rng.uniform(0.5, 0.8)
        rng.shuffle(segs)
        for (x0, x1, by, var) in segs:
            bws, bhs = bw * s * var, bh * s * var
            dep = bhs * 2.6 if f < 0.7 else max(bhs * 2.6, pl.ph - pl.Y(by))
            poly = K.billow_strip(pl.X(x0), pl.X(x1), pl.Y(by), bws, bhs, rng, depth=dep,
                                  var=0.9 + 0.5 * f, power=2.0, taper=0.14)
            lit_depth = bhs * (1.1 - 0.3 * f)
            bank(pl, poly, bhs * 1.5, _levels(f), rng, f, prox_w=(0.13 + 0.12 * f) * W * s, hz=hz,
                 lit_depth=lit_depth, deep_col=deep, glitter=(1.0 - f) * 1.2 if f < 0.5 else 0.0,
                 rim=K._ss(0.05, 0.4, f) * 0.9 + 0.1, dark=0.6 + 0.4 * f,
                 soft=(1 - K._ss(0.0, 0.3, f)) * 1.2 * pl.u, bump_w=bws)
            # nearer banks are heaped: 1-2 lower, smaller heaps in front of the main bank (crisp
            # overlapping-mass edges, each with its own sunlit top) instead of one big flat face
            nsub = 0 if f < 0.35 else (1 if f < 0.65 else 2)
            for j in range(nsub):
                Ls = (x1 - x0) * rng.uniform(0.25, 0.55)
                xs0 = rng.uniform(x0, x1 - Ls)
                sub_by = by + bh * var * rng.uniform(0.7, 1.0) * (1 + 0.7 * j)
                sv = rng.uniform(0.5, 0.75)
                bws2, bhs2 = bws * sv, bhs * sv
                poly = K.billow_strip(pl.X(xs0), pl.X(xs0 + Ls), pl.Y(sub_by), bws2, bhs2, rng,
                                      depth=max(bhs2 * 2.4, dep - (pl.Y(sub_by) - pl.Y(by))), var=1.0,
                                      power=2.0, taper=0.2)
                bank(pl, poly, bhs2 * 1.5, _levels(f), rng, f, prox_w=(0.13 + 0.12 * f) * W * s, hz=hz,
                     lit_depth=bhs2 * (1.1 - 0.3 * f), deep_col=deep, rim=K._ss(0.05, 0.4, f) * 0.9 + 0.1,
                     dark=0.6 + 0.4 * f, bump_w=bws2)
        # far rows recede into the haze behind the next row
        if f < 0.55:
            ys = np.arange(pl.ph, dtype=np.float32)
            kk = (K._ss(pl.Y(y - 3 * bh), pl.Y(y + 1.5 * bh), ys) * 0.22 * (1 - f))[:, None, None]
            a = pl.prem[..., 3:4]
            pl.prem[..., :3] = pl.prem[..., :3] * (1 - kk) + HAZE_SUN * 0.6 * a * kk + HAZE_OFF * 0.4 * a * kk
    return plates


# ------------------------------------------------------------------------------------------ towers
def tower(pl, cx, base_y, Hc, seed=31, width=1.0, haze=0.0, lean=0.0, glow=1.0, kind='tower'):
    """A backlit towering cumulus on plate pl (frame px in). kind='tower' | 'mass' (low, broad)."""
    u = pl.u
    s = pl.s
    rng = np.random.default_rng(seed)
    X, Y = pl.X(cx), pl.Y(base_y)
    Hs = Hc * s
    wd = width
    if kind == 'tower':
        L = [(0.06, 0.62, 0.36, 0.34, 0.1), (0.2, 0.33, 0.3, 0.42, 0.05), (-0.04, 0.27, 0.5, 0.5, 0.08),
             (-0.24, 0.14, 0.34, 0.33, 0.04), (0.3, 0.12, 0.34, 0.3, 0.06), (0.02, 0.02, 0.9, 0.26, 0.0),
             (-0.33, 0.0, 0.42, 0.15, 0.0), (0.4, 0.0, 0.34, 0.14, 0.0)]
    else:
        L = [(-0.3, 0.05, 0.9, 0.55, 0.0), (0.35, 0.02, 1.0, 0.7, 0.0), (0.05, 0.0, 1.3, 1.0, 0.0),
             (-0.8, 0.0, 0.8, 0.45, 0.0), (0.95, 0.0, 0.7, 0.4, 0.0)]
    pad = int(0.45 * Hs * max(wd, 1)) + 16
    bx0 = int(X - 1.3 * Hs * wd) - pad
    by0 = int(Y - 1.1 * Hs) - pad
    bw_, bh_ = int(2.6 * Hs * wd) + 2 * pad, int(1.3 * Hs) + 2 * pad
    P = K.Painter(bw_, bh_, sun=(pl.sun[0] - bx0, pl.sun[1] - by0), pal='sunrise', unit=u, seed=seed)
    sunl = np.array([pl.sun[0] - bx0, pl.sun[1] - by0], np.float32)
    top_y = Y - by0 - Hs
    base_l = Y - by0
    for i, (ox, oy, ww, hh, ln) in enumerate(L):
        crng = np.random.default_rng(int(rng.integers(1 << 31)))
        poly = K.envelope(X - bx0 + ox * Hs * wd, base_l - oy * Hs, ww * Hs * wd, hh * Hs, rng,
                          power=rng.uniform(2.2, 2.7), lump=0.1, lean=ln + lean, base_round=0.12, base_wave=0.05)
        size = hh * Hs * 0.8
        sp = sunl - poly.mean(0)
        sp = sp / (float(np.hypot(*sp)) + 1e-6)
        m, x0, y0 = K.cauliflower(poly, crng, size, K.DEFAULT_LEVELS, down_cut=0.3, side_scale=0.75,
                                  sun_dir=sp, sun_bias=0.15, clump=0.45, concave=0.6)
        if m.size == 0:
            continue
        pd = int(0.12 * size) + 4
        m = cv2.copyMakeBorder(m, pd, pd, pd, pd, cv2.BORDER_CONSTANT, value=0)
        x0 -= pd
        y0 -= pd
        h, w = m.shape
        yy = (np.arange(h, dtype=np.float32)[:, None] + y0)
        v = np.clip((yy - top_y) / max(Hs, 1), 0, 1)                   # 0 crown .. 1 foot
        body = _lerp(np.array([0.7, 0.66, 0.92], np.float32), np.array([0.5, 0.47, 0.82], np.float32),
                     K._ss(0.0, 0.7, v)[..., None])
        body = _lerp(body, np.array([0.4, 0.36, 0.7], np.float32), K._ss(0.6, 1.0, v)[..., None])
        # warm bounce from the lit sea on the lowest part
        body = body + np.array([0.12, 0.04, 0.0], np.float32) * K._ss(0.7, 1.0, v)[..., None]
        # skylit top of every mass: a crisp lighter cap (overlapping-mass edges read without outlines)
        mt, mv = _col_top(m)
        mts = _smooth_top(mt, mv, 0.4 * size)
        dm = (np.arange(h, dtype=np.float32)[:, None] - mts[None, :]) / max(hh * Hs, 1)
        col = _lerp(body * 1.1 + 0.03, body * 0.86, K._ss(0.05, 0.9, dm)[..., None])
        # the mass's side away from the sun a touch deeper (form), no sphere shading
        xx = np.arange(w, dtype=np.float32)[None, :] + x0
        cxm = float(poly[:, 0].mean())
        side = np.clip((xx - cxm) * np.sign(sunl[0] - cxm) / (0.5 * ww * Hs * wd + 1), -1, 1)
        col = col * (1 - 0.08 * K._ss(0.2, -0.8, side))[..., None]
        # cast shade onto what is already painted just outside the mass (crisp overlap edge)
        if i:
            sh = K._blur(m, max(0.05 * size, 1)) * (1 - m)
            X0, Y0 = max(x0, 0), max(y0, 0)
            X1, Y1 = min(x0 + w, P.w), min(y0 + h, P.h)
            if X1 > X0 and Y1 > Y0:
                reg = P.prem[Y0:Y1, X0:X1]
                reg[..., :3] *= (1 - 0.18 * sh[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0])[..., None]
        X0, Y0 = max(x0, 0), max(y0, 0)
        X1, Y1 = min(x0 + w, P.w), min(y0 + h, P.h)
        if X1 <= X0 or Y1 <= Y0:
            continue
        mm = m[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0][..., None]
        reg = P.prem[Y0:Y1, X0:X1]
        reg[..., :3] = reg[..., :3] * (1 - mm) + col[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] * mm
        reg[..., 3:4] = reg[..., 3:4] * (1 - mm) + mm
    # translucent glow along the SUNWARD silhouette of the whole union (light through thin edges)
    A = np.ascontiguousarray(np.clip(P.prem[..., 3], 0, 1))
    ys, xs = np.mgrid[0:P.h, 0:P.w].astype(np.float32)
    dx, dy = sunl[0] - xs, sunl[1] - ys
    dl = np.sqrt(dx * dx + dy * dy) + 1e-3
    ux, uy = dx / dl, dy / dl
    prox = np.exp(-dl / (0.9 * Hs))

    def toward(img, r):
        return cv2.remap(img, xs + ux * r, ys + uy * r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=0)
    gw = 0.06 * Hs
    g1 = A * (1 - K._blur(toward(A, 0.3 * gw), 0.08 * gw))
    g2 = A * (1 - K._blur(toward(A, gw), 0.4 * gw))
    g3 = A * (1 - K._blur(toward(A, 2.5 * gw), 1.2 * gw))
    g1 = K._ss(0.3, 0.7, g1)
    rgb = P.prem[..., :3] / np.maximum(A, 1e-5)[..., None]
    # cool shadow side: the whole tower body away from the sun edge a touch deeper / bluer
    rgb = _lerp(rgb, rgb * np.array([0.88, 0.9, 1.0], np.float32), (1 - g3)[..., None] * 0.5)
    k3 = (g3 * 0.18 * glow * prox)[..., None]
    rgb = _lerp(rgb, PINK * 0.6 + LAV * 0.4, k3)
    k2 = (K._ss(0.3, 0.7, g2) * 0.35 * glow * (0.2 + 0.8 * prox))[..., None]
    rgb = _lerp(rgb, PEACH * 0.8 + PINK * 0.2, k2)
    k1 = (g1 * 0.95 * glow * (0.35 + 0.65 * prox))[..., None]
    rgb = _lerp(rgb, _lerp(np.array([1.05, 0.86, 0.72], np.float32), np.array([1.4, 1.22, 0.95], np.float32),
                           prox[..., None]), k1)
    # aerial haze (distant towers melt toward the sky / horizon glow), stronger toward the foot
    vv = np.clip((ys - top_y) / max(Hs, 1), 0, 1)[..., None]
    hk = np.clip(haze + 0.25 * K._ss(0.5, 1.0, vv) * (0.4 + haze), 0, 0.9)
    rgb = _lerp(rgb, np.array([0.96, 0.8, 0.84], np.float32), hk)
    P.prem[..., :3] = rgb * A[..., None]
    P.lining(1.8 * glow, rim_px=3.0, halo=0.8, backlit=0.6, pal=dict(K.PRESETS['sunrise'], rim=(1.9, 1.7, 1.35)),
             y_max=base_l - 0.08 * Hs)
    P.wisps(X - bx0 - 0.6 * Hs * wd, X - bx0 + 0.6 * Hs * wd, base_l + 0.02 * Hs, 0.1 * Hs, rng=rng,
            pal=dict(K.PRESETS['sunrise'], haze='#d8b0cc', shade='#8a80c8', refl='#a8a0d8'), seed=seed + 11,
            amount=0.5, erode=1.3, base_haze=0.35, haze_color='#d4b4d0')
    a = np.clip(P.prem[..., 3], 0, 1)
    rgb = P.prem[..., :3] / np.maximum(a, 1e-5)[..., None]
    pl.over(bx0, by0, rgb.astype(np.float32), a.astype(np.float32),
            (g3 * 0.6 * (0.3 + 0.7 * prox) * glow).astype(np.float32))
    return A


# ------------------------------------------------------------------------------------------ cirrus
def cirrus(W, H, sun, seed=12):
    """Painted cirrus: tapered brush strokes of varying width (a few split into 2-3 strands) with warm
    underlighting from the low sun along their lower edges. RGBA (frame + margin)."""
    rng = np.random.default_rng(seed)
    pw, ph = int(W * (1 + 2 * M)), int(H * (1 + 2 * M))
    ss = 2
    fib = K._noise(pw * ss, ph * ss, 26, seed + 3, 3, stretch=10.0)
    fib = cv2.warpAffine(fib, cv2.getRotationMatrix2D((pw * ss / 2, ph * ss / 2), -4, 1.0), (pw * ss, ph * ss),
                         borderMode=cv2.BORDER_REFLECT)
    acc = np.zeros((ph * ss, pw * ss), np.float32)
    streaks = [(0.06, 0.15, 0.42, -0.05, 0.016, 3), (0.34, 0.08, 0.3, 0.03, 0.009, 2),
               (0.62, 0.11, 0.33, -0.04, 0.014, 3), (0.76, 0.26, 0.24, 0.04, 0.008, 1),
               (0.2, 0.3, 0.2, 0.02, 0.006, 1)]
    for (x0, y0, L, bend, wd, ns) in streaks:
        for j in range(ns):
            n = 90
            tt = np.linspace(0, 1, n)
            off = (j - (ns - 1) / 2) * wd * 1.4 * H / W
            xs = (x0 + L * tt + rng.uniform(-0.02, 0.02) * j) * W
            ys = (y0 + off + bend * np.sin(tt * math.pi) - 0.04 * tt) * H + 0.004 * H * np.sin(tt * 7 + rng.uniform(0, 6))
            wprof = wd * W * (1 - 0.35 * j / max(ns, 1)) * np.sin(np.clip((tt - 0.05 * j) / (1 - 0.1 * j + 1e-3), 0, 1) * math.pi) ** 0.6 \
                * (0.45 + 0.55 * (0.5 + 0.5 * np.sin(tt * rng.uniform(4, 8) + rng.uniform(0, 6))))
            m = np.zeros_like(acc)
            for q in range(n - 1):
                p0 = (int((xs[q] + M * W) * ss), int((ys[q] + M * H) * ss))
                p1 = (int((xs[q + 1] + M * W) * ss), int((ys[q + 1] + M * H) * ss))
                if wprof[q] * ss >= 1:
                    cv2.line(m, p0, p1, 1.0, max(int(wprof[q] * ss), 1), cv2.LINE_AA)
            m = cv2.GaussianBlur(m, (0, 0), 2.5 * ss * W / 1920)
            acc = np.maximum(acc, m * (0.7 + 0.3 * rng.random()))
    a = acc * (0.35 + 0.65 * K._ss(0.3, 0.7, fib))
    a = cv2.resize(a, (pw, ph), interpolation=cv2.INTER_AREA)
    # underlight: lower edge of every stroke catches the sun
    r = max(2.0 * W / 1920, 1.0)
    under = np.clip(a - K._shift(a, 0, 2.5 * r), 0, 1)
    under = K._blur(under, r) * 2.0
    ys = np.arange(ph, dtype=np.float32)[:, None] / ph
    xs = np.arange(pw, dtype=np.float32)[None, :] / pw
    sxn = (sun[0] / W + M) / (1 + 2 * M)
    g = np.exp(-((xs - sxn) / 0.4) ** 2)
    base = _lerp(np.array([0.82, 0.8, 0.98], np.float32), np.array([0.98, 0.8, 0.88], np.float32),
                 K._ss(0.05, 0.4, ys)[..., None])
    warm = _lerp(np.array([1.0, 0.72, 0.72], np.float32), np.array([1.1, 0.86, 0.6], np.float32), g[..., None])
    col = _lerp(base, warm, np.clip(under * (0.4 + 0.6 * g), 0, 1)[..., None])
    return np.concatenate([col, np.clip(a * 0.72, 0, 1)[..., None]], -1).astype(np.float32)


# ------------------------------------------------------------------------------------------ near vapour
def vapour(W, H, seed=40):
    """Near-plane vapour: thin torn mist wisps just below the horizon line of sight; with the dolly's
    zoom (about the vanishing point) they stream down and out past the camera. Lit warm on top."""
    pw, ph = int(W * (1 + 2 * M)), int(H * (1 + 2 * M))
    n1 = K._noise(pw, ph, 6.0, seed, 4, stretch=7.0)
    n2 = K._noise(pw, ph, 18.0, seed + 1, 3, stretch=5.0)
    ys = np.arange(ph, dtype=np.float32)[:, None] / ph
    xs = np.arange(pw, dtype=np.float32)[None, :] / pw
    band = K._ss(0.56, 0.62, ys) * (1 - K._ss(0.74, 0.8, ys))
    side = 1 - 0.8 * np.exp(-((xs - 0.55) / 0.12) ** 2)           # keep the sun path clear
    a = K._ss(0.58, 0.76, 0.7 * n1 + 0.3 * n2) * band * side
    a = K._blur(a, 2 * W / 1920)
    top = np.clip(a - K._shift(a, 0, -8 * W / 1920), 0, 1)
    col = _lerp(np.array([0.7, 0.62, 0.88], np.float32), np.array([1.05, 0.84, 0.72], np.float32),
                np.clip(0.3 + top * 3, 0, 1)[..., None])
    return np.concatenate([col, np.clip(a * 0.6, 0, 1)[..., None]], -1).astype(np.float32)
