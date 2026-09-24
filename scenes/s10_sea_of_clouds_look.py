"""s10 helpers: clouds2 sea + towers layout, light swell, paper texture, birds."""
import math

import numpy as np
import cv2
from numba import njit, prange

from lib import core as C, sky as S, clouds2 as K

ROWS = 17
SPLITS = (6, 10, 13)            # -> 4 sea plates far .. near
DEPTHS = (0.12, 0.35, 0.7, 1.25)
SPEEDS = (0.0008, 0.0018, 0.0032, 0.005)
BILLOW = (0.0, 0.0, 0.0012, 0.0016)
PERSP = 2.1


# scene palettes: the clouds2 sunrise preset with deeper indigo valleys / shadows
K.PRESETS['s10dawn'] = dict(K.PRESETS['sunrise'], shade='#7468b8', deep='#241f5e', refl='#8a7fc8', mid='#d8749c')
K.PRESETS['s10tower'] = dict(K.PRESETS['sunrise'], hi=(1.0, 0.93, 0.8), lit=(1.0, 0.8, 0.58), lit_lo=(0.96, 0.62, 0.62),
                             shade='#7a70bc', deep='#3a3480', rim=(1.4, 1.2, 0.9))
K.PRESETS['s10dawn_far'] = dict(K.PRESETS['sunrise_far'], shade='#9c86c2', deep='#6a58a4')


def row_y(i, hy, bottom):
    f = i / (ROWS - 1)
    return hy + (bottom - hy) * f ** PERSP


def sea(PW, PH, hy, sun_p, u):
    return K.sea_of_clouds_plate(PW, PH, hy, sun=sun_p, preset='s10dawn', seed=7, rows=ROWS, splits=SPLITS,
                                 horizon_haze=(K._c('#ffd0a8'), 0.7), near_w=0.42, far_w=0.02,
                                 aspect_near=0.2, aspect_far=0.12, haze_far=0.8, persp=PERSP, valley=0.95,
                                 unit=u, sun_z=-0.3, gaps=0.12, big=0.0, crest=0.07, crest_near=0.26,
                                 fill_near=0.65, lit=0.06, fg_deep='#34327e', sun_focus=0.3)


def towers(PW, PH, hy, sun_p, u):
    bottom = 1.1 * PH
    out = []
    # big tower left (mid distance), rising behind plate 1's front rows (drawn after plate 1)
    yb = row_y(SPLITS[1] + 1, hy, bottom)
    Hc = 0.62 * PH
    pl = K.cumulonimbus_plate(PW, PH, 0.17 * PW, yb, Hc, sun=sun_p, preset='s10tower', seed=33, sun_z=-0.15,
                              anvil=False, width=0.95, backlit=0.8, wisps=False, form=0.8, bounce_group=0.6, lining=1.0,
                              haze=0.04, haze_band=(yb - 0.25 * Hc, yb, K._c('#8a80c4'), 0.25))
    out.append(dict(plate=pl, after=1, depth=0.5, speed=0.0022))
    # slender hazier tower right, farther (drawn after plate 0)
    yb2 = row_y(SPLITS[0] + 1, hy, bottom)
    Hc2 = 0.33 * PH
    pl2 = K.cumulonimbus_plate(PW, PH, 0.85 * PW, yb2, Hc2, sun=sun_p, preset='s10tower', seed=31, sun_z=-0.15,
                               anvil=False, width=0.75, backlit=0.8, wisps=False, form=0.8, bounce_group=0.5, lining=1.0,
                               haze=0.3, haze_band=(yb2 - 0.35 * Hc2, yb2, K._c('#e6b4c0'), 0.5))
    out.append(dict(plate=pl2, after=0, depth=0.24, speed=0.0012))
    return out


def light(p):
    """Light swell 0..1 over the shot (eased, strongest at the end)."""
    return float(C.smoothstep(0.1, 1.0, p))


_CACHE = {}


def sky_warm(W, H, sun, hz):
    key = ('sw', W, H)
    if key not in _CACHE:
        xs, ys = C.grid(W, H)
        d = np.sqrt(((xs - sun[0]) / W) ** 2 + ((ys - sun[1]) / H * 0.6) ** 2)
        g = np.exp(-(d / 0.35) ** 2)
        _CACHE[key] = (g[..., None] * np.array([1.0, 0.7, 0.35], np.float32)).astype(np.float32)
    return _CACHE[key]


def fg_shade(H, L):
    key = ('fg', H)
    if key not in _CACHE:
        yy = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        _CACHE[key] = (K._ss(0.62, 1.0, yy) * np.array([1.0, 0.85, 0.6], np.float32)).astype(np.float32)
    return 1 - (0.2 - 0.08 * L) * _CACHE[key]


def horizon_glow(W, H, hz, sx):
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    d = (ys - hz * H) / H
    g = np.exp(-(d / 0.03) ** 2) * (0.3 + 0.7 * np.exp(-((xs - sx) / (0.3 * W)) ** 2))
    return (g[..., None] * np.array([1.0, 0.72, 0.45], np.float32) * 0.12).astype(np.float32)


def paper(W, H, seed=3):
    """Static paper / paint texture (multiplicative, ~+-1.5%)."""
    rng = np.random.default_rng(seed)
    n1 = C.fbm(W, H, W / 6.0, 4, seed=seed) - 0.5                 # fine tooth
    n2 = C.fbm(max(W // 4, 8), max(H // 4, 8), 6.0, 3, seed=seed + 1) - 0.5   # blotchy wash
    n2 = cv2.resize(n2.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    fib = rng.standard_normal((H, W)).astype(np.float32)
    fib = cv2.GaussianBlur(fib, (0, 0), sigmaX=max(W / 1920 * 3.0, 0.8), sigmaY=max(W / 1920 * 0.6, 0.4))
    fib = fib / (np.abs(fib).std() * 3 + 1e-6)
    tex = 1 + 0.016 * n1 + 0.02 * n2 + 0.006 * fib
    return tex[..., None].astype(np.float32)


def _bird_pts(cx, cy, s, ph, hd):
    """Distant bird as a gull 'M' stroke: tips, wrists, body; wings beat with phase ph."""
    f = math.sin(ph)
    tip = -0.32 * f + 0.07                  # tip height (neg = up); tips droop in the glide
    wr = -0.14 * f - 0.07                   # wrist stays raised: arched gull 'm'
    span = 0.5 * (1 - 0.15 * abs(f))
    pts = [(-span, tip), (-0.22, wr - 0.05), (-0.05, 0.02), (0.0, 0.05), (0.05, 0.02), (0.22, wr - 0.05), (span, tip)]
    return [(cx + x * s * hd, cy + y * s) for x, y in pts]


def birds_over(img, W, H, t, seed=4):
    """A small, loose flock of distant gulls: tapered 'M' strokes with an independent wingbeat each,
    slow drift, deep indigo against the sky."""
    rng = np.random.default_rng(seed)
    n = 6
    base = np.array([[0.0, 0.0], [0.035, 0.012], [0.07, -0.006], [0.1, 0.018], [0.055, 0.03], [0.13, 0.004]])
    base = base + rng.normal(0, 0.004, base.shape)
    sz = rng.uniform(0.75, 1.2, n) * 0.013 * W
    fr = rng.uniform(2.2, 3.0, n)
    ph0 = rng.uniform(0, 6.28, n)
    ss = 4
    cxs = [(0.6 + base[i, 0] + 0.009 * t) * W + 0.003 * W * math.sin(0.6 * t + i) for i in range(n)]
    cys = [(0.2 + base[i, 1] - 0.0015 * t) * H + 0.002 * W * math.sin(0.8 * t + 2 * i) for i in range(n)]
    pad = 0.02 * W
    x0 = int(max(min(cxs) - pad, 0)); x1 = int(min(max(cxs) + pad, W))
    y0 = int(max(min(cys) - pad, 0)); y1 = int(min(max(cys) + pad, H))
    if x1 <= x0 or y1 <= y0:
        return img
    bw, bh = x1 - x0, y1 - y0
    m = np.zeros((bh * ss, bw * ss), np.uint8)
    for i in range(n):
        cx, cy = cxs[i] - x0, cys[i] - y0
        pts = _bird_pts(cx, cy, sz[i], ph0[i] + 2 * math.pi * fr[i] * t, 1.0)
        P = (np.array(pts) * ss).astype(np.int32)
        th = max(int(round(sz[i] * 0.05 * ss)), 1)
        for j in range(len(P) - 1):
            w = th + (th // 2 if 1 <= j <= 4 else 0)
            cv2.line(m, tuple(P[j]), tuple(P[j + 1]), 255, w, cv2.LINE_AA)
        cv2.ellipse(m, tuple(P[3]), (max(int(sz[i] * 0.07 * ss), 1), max(int(sz[i] * 0.045 * ss), 1)), 0, 0, 360, 255, -1,
                    cv2.LINE_AA)
    a = cv2.resize(m.astype(np.float32) / 255.0, (bw, bh), interpolation=cv2.INTER_AREA)[..., None] * 0.9
    col = np.array([0.2, 0.22, 0.42], np.float32)
    img = img.copy()
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + col * a
    return img


@njit(cache=True, fastmath=True, parallel=True)
def _tone(x, paper, s, w0, w1):
    H, W = x.shape[0], x.shape[1]
    out = np.empty((H, W, 3), np.float32)
    k = 1.0 - s
    for i in prange(H):
        for j in range(W):
            r = max(x[i, j, 0], 0.0)
            g = max(x[i, j, 1], 0.0)
            b = max(x[i, j, 2], 0.0)
            m = max(r, max(g, b))
            mc = m
            if m > s:
                o = m - s
                mc = s + k * o / (o + k)
            f = mc / max(m, 1e-6)
            wd = min(max((m - w0) / (w1 - w0), 0.0), 1.0) * 0.9
            pp = paper[i, j]
            out[i, j, 0] = (r * f * (1 - wd) + mc * wd) * pp
            out[i, j, 1] = (g * f * (1 - wd) + mc * wd) * pp
            out[i, j, 2] = (b * f * (1 - wd) + mc * wd) * pp
    return out


def tone(img, s=0.78, white0=1.3, white1=4.0, paper=None):
    """Hue-preserving highlight roll-off: the max channel is compressed smoothly above `s` toward 1
    (ratios kept, so gold glows stay gold instead of washing to per-channel white); only really hot
    values (sun core, > white0) drift toward white. Optionally multiplies the static paper texture."""
    x = np.ascontiguousarray(img[..., :3], dtype=np.float32)
    pp = np.ones(x.shape[:2], np.float32) if paper is None else np.ascontiguousarray(paper.reshape(x.shape[:2]))
    return _tone(x, pp, np.float32(s), np.float32(white0), np.float32(white1))


def sun_glow(W, H, sun):
    """Creamy-gold halo around the sun (additive; hue kept by tone())."""
    xs, ys = C.grid(W, H)
    d = np.sqrt((xs - sun[0]) ** 2 + ((ys - sun[1]) * 1.15) ** 2) / W
    g = 0.42 * np.exp(-d / 0.022) + 0.22 * np.exp(-(d / 0.09) ** 2) + 0.08 * np.exp(-(d / 0.25) ** 2)
    col = np.array([1.0, 0.72, 0.42], np.float32)
    inner = np.exp(-d / 0.012)[..., None] * np.array([0.0, 0.18, 0.22], np.float32)   # whiter near the core
    return (g[..., None] * col + inner * 0.8).astype(np.float32)
