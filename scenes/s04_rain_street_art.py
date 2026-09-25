"""s04_rain_street texture painters (all procedural). Textures are returned as dicts
{'alb': (h,w,3) albedo, 'a': (h,w) alpha, 'emi': (h,w,3) emissive (HDR)} with straight colours."""
import math
import numpy as np
import cv2
import s04_rain_street_text as JT


def hexc(h):
    h = h.lstrip('#')
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


def tex(w, h, col=(0, 0, 0), a=1.0):
    w, h = max(int(w), 2), max(int(h), 2)
    return {'alb': np.ones((h, w, 3), np.float32) * np.asarray(col, np.float32),
            'a': np.full((h, w), a, np.float32),
            'emi': np.zeros((h, w, 3), np.float32)}


def rect(img, x0, y0, x1, y1, col, blend=1.0):
    """Fill rect (float coords, anti-aliased edges) on an (h,w,C) or (h,w) image."""
    h, w = img.shape[:2]
    xi0, yi0 = max(int(math.floor(x0)), 0), max(int(math.floor(y0)), 0)
    xi1, yi1 = min(int(math.ceil(x1)), w), min(int(math.ceil(y1)), h)
    if xi1 <= xi0 or yi1 <= yi0:
        return
    xs = np.arange(xi0, xi1, dtype=np.float32)
    ys = np.arange(yi0, yi1, dtype=np.float32)
    cx = np.clip(np.minimum(xs + 1, x1) - np.maximum(xs, x0), 0, 1)
    cy = np.clip(np.minimum(ys + 1, y1) - np.maximum(ys, y0), 0, 1)
    cov = (cy[:, None] * cx[None, :]) * blend
    sl = img[yi0:yi1, xi0:xi1]
    col = np.asarray(col, np.float32)
    if img.ndim == 3:
        sl[:] = sl * (1 - cov[..., None]) + col * cov[..., None]
    else:
        sl[:] = sl * (1 - cov) + col * cov


def vgrad(h, w, c0, c1, power=1.0):
    t = (np.linspace(0, 1, h, dtype=np.float32) ** power)[:, None, None]
    return np.broadcast_to(np.asarray(c0, np.float32) * (1 - t) + np.asarray(c1, np.float32) * t, (h, w, 3)).copy()


def noise(w, h, cells, seed, octaves=3):
    rng = np.random.default_rng(seed)
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    c = cells
    for o in range(octaves):
        cx, cy = max(int(c), 1), max(int(c * h / max(w, 1)), 1)
        g = rng.random((cy + 2, cx + 2)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= 0.5
        c *= 2
    return out / tot


# ----------------------------------------------------------------------------- glyphs

def _component(m, rng, x0, y0, x1, y1, lw):
    """Draw a kanji-like component into mask m within box (pixel coords)."""
    W, H = x1 - x0, y1 - y0
    kind = rng.choice(['lines', 'box', 'grid', 'person', 'dots', 'tree', 'sun', 'hook', 'roof', 'lines'])
    L = lambda a, b, c, d: cv2.line(m, (int(x0 + a * W), int(y0 + b * H)), (int(x0 + c * W), int(y0 + d * H)),
                                    255, lw, cv2.LINE_AA)
    if kind == 'lines':
        n = rng.integers(2, 5)
        for i in range(n):
            yy = 0.12 + 0.76 * i / max(n - 1, 1)
            L(0.1 + 0.1 * rng.random(), yy, 0.9 - 0.1 * rng.random(), yy)
        L(0.5, 0.05, 0.5, 0.95)
    elif kind == 'box':
        L(0.15, 0.15, 0.85, 0.15); L(0.15, 0.15, 0.15, 0.85); L(0.85, 0.15, 0.85, 0.85); L(0.15, 0.85, 0.85, 0.85)
        if rng.random() < 0.6:
            L(0.15, 0.5, 0.85, 0.5)
    elif kind == 'grid':
        L(0.12, 0.12, 0.88, 0.12); L(0.12, 0.88, 0.88, 0.88); L(0.12, 0.12, 0.12, 0.88); L(0.88, 0.12, 0.88, 0.88)
        L(0.5, 0.12, 0.5, 0.88); L(0.12, 0.5, 0.88, 0.5)
    elif kind == 'person':
        L(0.5, 0.08, 0.15, 0.9); L(0.5, 0.35, 0.88, 0.9)
    elif kind == 'dots':
        L(0.2, 0.12, 0.35, 0.25); L(0.15, 0.42, 0.3, 0.55); L(0.12, 0.92, 0.35, 0.65)
        L(0.5, 0.2, 0.9, 0.2); L(0.55, 0.55, 0.9, 0.55); L(0.7, 0.2, 0.7, 0.9)
    elif kind == 'tree':
        L(0.1, 0.3, 0.9, 0.3); L(0.5, 0.05, 0.5, 0.95); L(0.5, 0.32, 0.12, 0.8); L(0.5, 0.32, 0.88, 0.8)
    elif kind == 'sun':
        L(0.25, 0.1, 0.75, 0.1); L(0.25, 0.1, 0.25, 0.9); L(0.75, 0.1, 0.75, 0.9); L(0.25, 0.9, 0.75, 0.9)
        L(0.25, 0.5, 0.75, 0.5)
    elif kind == 'hook':
        L(0.15, 0.2, 0.85, 0.2); L(0.85, 0.2, 0.75, 0.85); L(0.75, 0.85, 0.55, 0.75); L(0.4, 0.2, 0.2, 0.9)
    elif kind == 'roof':
        L(0.5, 0.02, 0.5, 0.15); L(0.1, 0.18, 0.9, 0.18); L(0.1, 0.18, 0.1, 0.35); L(0.9, 0.18, 0.9, 0.35)
        L(0.25, 0.5, 0.75, 0.5); L(0.3, 0.7, 0.7, 0.7); L(0.5, 0.5, 0.5, 0.95); L(0.2, 0.95, 0.8, 0.95)


def glyph(rng, n=64, weight=0.1, pool='noren'):
    """Single real character mask (n x n, float 0..1), rendered with a Japanese font."""
    return JT.single(rng, n, pool)


def glyph_column(rng, w, h, count, weight=0.11, margin=0.12, text=None, pool='shop'):
    """Vertical run of real Japanese text filling a (h, w) mask."""
    return JT.column(rng, int(w), int(h), count, margin, text=text, pool=pool)


def glyph_row(rng, w, h, count, weight=0.11, margin=0.12, text=None, pool='shop'):
    return JT.row(rng, int(w), int(h), count, margin, text=text, pool=pool)


def _row(rng, w, h, count, weight=0.11, margin=0.12, text=None, pool='shop'):
    return JT.row(rng, int(w), int(h), count, margin, text=text, pool=pool)


# ----------------------------------------------------------------------------- signs

SIGN_STYLES = [
    # (panel bg, glyph colour, emissive gain of the panel, glyph emissive)
    dict(bg='#fff6e0', fg='#c8102e', pe=1.05, ge=0.0),    # white lightbox, red glyphs
    dict(bg='#ffe14a', fg='#1a1a1a', pe=1.0, ge=0.0),    # yellow lightbox, black glyphs
    dict(bg='#e8203a', fg='#fff7ea', pe=1.2, ge=1.2),    # red box, white glyphs
    dict(bg='#101018', fg='#ff4fa8', pe=0.0, ge=4.0),    # black + pink neon
    dict(bg='#0b1420', fg='#40e8ff', pe=0.0, ge=4.0),    # black + cyan neon
    dict(bg='#f2f8ff', fg='#1d4fd8', pe=1.0, ge=0.0),    # white box, blue glyphs
    dict(bg='#1a2a6a', fg='#fff4d0', pe=0.7, ge=2.4),    # navy box, cream glyphs
    dict(bg='#2a0f3a', fg='#ffb13d', pe=0.2, ge=3.6),    # purple + amber neon
]


def sign_panel(rng, w, h, style=None, vertical=True, count=None, frame=0.06, bright=1.0, side_strip=None, text=None):
    """Lightbox / neon sign face. w,h px."""
    st = style if isinstance(style, dict) else SIGN_STYLES[style if style is not None else rng.integers(len(SIGN_STYLES))]
    bg, fg = hexc(st['bg']), hexc(st['fg'])
    T = tex(w, h, bg)
    h, w = T['a'].shape
    fw = max(1.0, frame * min(w, h))
    # casing frame (dark metal)
    casing = hexc('#1c1d24')
    inner = np.zeros((h, w), np.float32)
    rect(inner, fw, fw, w - fw, h - fw, 1.0)
    T['alb'] = T['alb'] * inner[..., None] * (0.35 if st['pe'] > 0.5 else 1.0) + casing * (1 - inner[..., None])
    if count is None:
        count = int(np.clip(round((h / max(w, 1)) if vertical else (w / max(h, 1))), 1, 7))
    if vertical:
        gm = glyph_column(rng, w, h, max(count, len(text) if text else 0), weight=0.12, margin=0.16, text=text)
    else:
        gm = _row(rng, w, h, count, 0.12, 0.16)
    gm *= inner
    T['alb'] = T['alb'] * (1 - gm[..., None]) + fg * gm[..., None]
    # emissive: lit panel with slight hot-spot falloff (tubes behind acrylic)
    yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    hot = (1.0 - 0.18 * yy ** 2 - 0.25 * xx ** 2)
    pe = st['pe'] * bright
    E = bg * pe * (inner * hot)[..., None] * (1 - gm[..., None])
    if st['ge'] > 0:
        gl = cv2.GaussianBlur(gm, (0, 0), max(0.8, 0.02 * min(w, h)))
        E = E + fg * (gm * st['ge'] + gl * st['ge'] * 0.6)[..., None] * bright
        if pe > 0:
            E = E + fg * (gm * pe * 0.5)[..., None] * bright * (fg.max() > 0.9)
    T['emi'] = E.astype(np.float32)
    if min(w, h) >= 24:
        from s04_rain_street_art2 import sign_detail
        sign_detail(T, rng, fg, bg, lit=(pe > 0 or st['ge'] > 0))
    return T


def neon_outline(T, color, gain=3.0, inset=0.1, width=0.03):
    """Add a neon tube border inside a texture."""
    h, w = T['a'].shape
    m = np.zeros((h, w), np.float32)
    lw = max(1, int(width * min(w, h)))
    i = int(inset * min(w, h))
    cv2.rectangle(m, (i, i), (w - 1 - i, h - 1 - i), 1.0, lw, cv2.LINE_AA)
    g = cv2.GaussianBlur(m, (0, 0), max(1.0, lw * 1.5))
    c = np.asarray(color, np.float32)
    T['emi'] = T['emi'] + c * (m * gain + g * gain * 0.5)[..., None]
    T['alb'] = T['alb'] * (1 - m[..., None]) + c * m[..., None]
    return T


# ----------------------------------------------------------------------------- windows & facades

def window_tex(rng, w, h, kind, frame_col=(0.25, 0.26, 0.3)):
    """Window pane texture. kind: 'warm', 'cool', 'dark', 'curtain', 'blind'."""
    T = tex(w, h, frame_col)
    h, w = T['a'].shape
    f = max(1.0, 0.07 * min(w, h))
    g = np.zeros((h, w), np.float32)
    rect(g, f, f, w - f, h - f, 1.0)
    # mullion
    if w > h * 0.9:
        rect(g, w / 2 - f * 0.4, 0, w / 2 + f * 0.4, h, 0.0)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    if kind == 'dark':
        # dark pane mirroring the city-lit rain sky: pink-violet glow at the top fading to deep navy,
        # a crisp diagonal sheen band and a thin lit sill edge
        xx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        sky = hexc('#5a6e70') * (1 - yy) ** 1.6 + hexc('#16222a') * (0.6 + 0.4 * yy)
        glass = sky * rng.uniform(0.75, 1.1)
        glass = glass + hexc('#c8d8d8') * 0.18 * np.exp(-((xx - 0.25 - 0.45 * yy) / 0.06) ** 2)
        glass = glass + hexc('#b8ccd0') * 0.08 * np.exp(-((xx - 0.62 - 0.45 * yy) / 0.025) ** 2)
        E = glass * 0.3
    else:
        if kind == 'warm':
            c0, c1 = hexc('#ffcf7a'), hexc('#ff9a4a')
        elif kind == 'cool':
            c0, c1 = hexc('#e8f4ff'), hexc('#a8d4ff')
        elif kind == 'curtain':
            c0, c1 = hexc('#ffb86a'), hexc('#e0703a')
        else:
            c0, c1 = hexc('#fff0c8'), hexc('#ffd08a')
        glass = c0 * (1 - yy) + c1 * yy
        E = glass * rng.uniform(0.7, 1.4)
        if kind == 'curtain':
            xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
            folds = 0.75 + 0.25 * np.sin(xx * rng.uniform(14, 24) + rng.uniform(0, 6)) ** 2
            E = E * folds[..., None]
            gap = rng.uniform(0.3, 0.7)
            E = E * (1 + 0.8 * np.exp(-((xx - gap) / 0.05) ** 2))[..., None]
        if kind == 'blind':
            yl = np.arange(h, dtype=np.float32)[:, None]
            per = max(2.0, h / 14)
            sl = 0.7 + 0.3 * (np.sin(yl / per * 2 * np.pi) > -0.3)
            E = E * sl[..., None]
        if kind in ('warm', 'cool') and rng.random() < 0.5:
            # interior silhouette (shelf / plant / person)
            m = np.zeros((h, w), np.float32)
            x0 = rng.uniform(0.1, 0.6) * w
            rect(m, x0, h * rng.uniform(0.35, 0.6), x0 + w * rng.uniform(0.15, 0.35), h, 1.0)
            E = E * (1 - 0.7 * m[..., None])
    T['alb'] = T['alb'] * (1 - g[..., None]) + glass * g[..., None] * 0.6
    T['emi'] = (E * g[..., None]).astype(np.float32)
    return T


def facade(rng, wm, hm, ppm, style, floors=None, ground='shutter', wall=None, lit_p=0.45, cool_p=0.25,
           max_px=4096, near=False, spandrel=False):
    """Facade texture for a wall wm (along the street) x hm metres at ppm px/m."""
    ppm = min(ppm, max_px / max(wm, hm))
    w, h = int(max(wm * ppm, 4)), int(max(hm * ppm, 4))
    if wall is None:
        wall = hexc(rng.choice(['#5a5560', '#6a6258', '#4c5260', '#70645c', '#3e4454', '#5e5a66', '#7a6e66']))
    # night rain: lift every wall toward a rich deep blue-violet instead of neutral grey mud
    wall = np.asarray(wall, np.float32) * 0.72 + hexc('#3c4c4e') * 0.28
    T = tex(w, h, wall)
    A = T['alb']
    E = T['emi']
    P = lambda m: m * ppm
    stains = np.zeros((h, w), np.float32)     # rain-dark run-off below sills / AC units
    # subtle tile texture + grime
    n = noise(w, h, max(4, w // 60), rng.integers(1e9), 3)
    A *= (0.85 + 0.3 * n)[..., None]
    # rain streaks (vertical stains)
    st = noise(max(w // 3, 4), 8, max(4, w // 12), rng.integers(1e9), 2)
    st = cv2.resize(st, (w, h), interpolation=cv2.INTER_CUBIC)
    A *= (0.95 + 0.08 * st)[..., None]
    fh = 3.0 if style != 'office' else 3.6
    if floors is None:
        floors = max(1, int(hm // fh))
    gh = 3.2   # ground floor height
    # ground floor ------------------------------------------------------------
    gy0 = h - P(gh)
    from s04_rain_street_shop import shopfront
    shopfront(T, rng, ground, 0, gy0, w, h, ppm)
    # slab line above ground floor
    rect(A, 0, gy0 - P(0.18), w, gy0, A[int(max(gy0 - P(0.5), 0)), w // 2] * 1.25)
    # spandrel between the shop fascia and the first-floor windows: upstairs tenant lightbox, outdoor AC units
    # on the ledge, a gutter line and rain run-off (so the band never reads as a flat dark plane)
    if spandrel and style != 'office':
        sy0, sy1 = gy0 - P(0.18) - P(0.78), gy0 - P(0.3)
        sw = w * rng.uniform(0.3, 0.5)
        sx = rng.uniform(0.05, 0.95) * (w - sw)
        if sy1 - sy0 > 4 and sw > 12:
            sp = sign_panel(rng, sw, sy1 - sy0, style=int(rng.choice([0, 1, 2, 5, 6])), vertical=False,
                            count=int(np.clip(sw / max(sy1 - sy0, 1) * 0.8, 2, 6)), bright=0.75)
            _paste(T, sp, sx, sy0)
        rect(A, 0, gy0 - P(0.26), w, gy0 - P(0.2), hexc('#8a8c9a'))
        rect(E, 0, gy0 - P(0.26), w, gy0 - P(0.245), hexc('#ff9ad8') * 0.25)
        for k in range(int(rng.integers(1, 4))):
            ax = rng.uniform(0.0, 0.92) * w
            if sx - P(0.8) < ax < sx + sw:
                continue
            ay1 = gy0 - P(0.26)
            ay0 = ay1 - P(0.6)
            rect(A, ax, ay0, ax + P(0.78), ay1, hexc('#c4c8d0'))
            rect(A, ax, ay0, ax + P(0.78), ay0 + max(P(0.03), 1), hexc('#eef0f6'))
            if P(0.78) > 10:
                c = (int(ax + P(0.3)), int((ay0 + ay1) / 2))
                cv2.circle(A, c, max(1, int(P(0.22))), tuple(float(x) for x in hexc('#3a3e48')), -1, cv2.LINE_AA)
                for rr in (0.07, 0.14, 0.2):
                    cv2.circle(A, c, max(1, int(P(rr))), tuple(float(x) for x in hexc('#8a8e98')), 1, cv2.LINE_AA)
            rect(stains, ax, ay1, ax + P(0.78), gy0, 0.5)
    # upper floors -------------------------------------------------------------
    y = gy0 - P(0.18)
    fl = 0
    while y - P(fh) > P(0.4) and fl < floors:
        ytop = y - P(fh)
        # floor band
        rect(A, 0, ytop, w, ytop + P(0.22), A[int(ytop + P(0.3)), w // 2] * 1.2)
        if style == 'office':
            wx0 = P(0.3)
            nw = max(1, int((wm - 0.6) / 1.6))
            ww = (w - 2 * wx0) / nw
            lit_row = rng.random() < 0.6
            for i in range(nw):
                kind = 'dark'
                if lit_row and rng.random() < 0.8:
                    kind = 'cool'
                elif rng.random() < lit_p * 0.4:
                    kind = 'blind'
                wt = window_tex(rng, max(ww - P(0.12), 3), P(fh - 0.9), kind)
                _paste(T, wt, wx0 + i * ww + P(0.06), ytop + P(0.45))
        else:
            # apartment / mixed: 2-3 windows per floor, AC units, balconies
            nwin = max(1, int(wm / (rng.uniform(1.7, 2.1) if near else rng.uniform(2.4, 3.4))))
            slot = w / nwin
            for i in range(nwin):
                ww = P(rng.uniform(1.0, 1.6))
                hh = P(rng.uniform(1.1, 1.6))
                wx = i * slot + (slot - ww) / 2
                wy = ytop + P(0.6)
                r = rng.random()
                kind = 'warm' if r < lit_p * 0.5 else ('curtain' if r < lit_p else ('cool' if r < lit_p + cool_p * 0.3 else 'dark'))
                # outer frame (aluminium sash) + lintel shadow
                rect(A, wx - P(0.06), wy - P(0.06), wx + ww + P(0.06), wy + hh + P(0.04), hexc('#2a2c36'))
                wt = window_tex(rng, ww, hh, kind)
                _paste(T, wt, wx, wy)
                # sill catching the street neon + rain run-off stain beneath it
                rect(A, wx - P(0.1), wy + hh, wx + ww + P(0.1), wy + hh + P(0.07), hexc('#a8a8b8'))
                rect(E, wx - P(0.1), wy + hh, wx + ww + P(0.1), wy + hh + P(0.025),
                     hexc(rng.choice(['#ff9ad8', '#9ae0ff', '#ffd0a0'])) * 0.35)
                rect(stains, wx, wy + hh + P(0.07), wx + ww, wy + hh + P(rng.uniform(0.8, 1.6)), 1.0)
                # half the lit windows show a hanging curtain edge / a person-shaped shadow
                if kind in ('warm', 'curtain') and rng.random() < 0.25 and ww > 12:
                    m = np.zeros((h, w), np.float32)
                    cx_, hb = wx + ww * rng.uniform(0.3, 0.7), hh * 0.55
                    cv2.ellipse(m, (int(cx_), int(wy + hh - hb - hb * 0.18)), (max(1, int(hb * 0.12)), max(1, int(hb * 0.15))),
                                0, 0, 360, 1.0, -1, cv2.LINE_AA)
                    cv2.ellipse(m, (int(cx_), int(wy + hh)), (max(1, int(hb * 0.3)), max(1, int(hb * 0.62))), 0, 180, 360,
                                1.0, -1, cv2.LINE_AA)
                    m[:int(wy)] = 0
                    m[int(wy + hh):] = 0
                    E *= (1 - 0.75 * m[..., None])
                # AC unit under/next to window: casing, fan grille, side louvres, drain hose
                if rng.random() < 0.6:
                    ax = wx + ww + P(0.15) if rng.random() < 0.5 else wx - P(0.85)
                    ay0, ay1 = wy + hh - P(0.55), wy + hh
                    rect(A, ax, ay0, ax + P(0.7), ay1, hexc('#c8ccd2'))
                    rect(A, ax, ay0, ax + P(0.7), ay0 + max(P(0.03), 1), hexc('#eef0f4'))
                    if P(0.7) > 10:
                        c = (int(ax + P(0.26)), int((ay0 + ay1) / 2))
                        cv2.circle(A, c, max(1, int(P(0.2))), tuple(float(x) for x in hexc('#3a3e48')), -1, cv2.LINE_AA)
                        for rr in (0.06, 0.12, 0.18):
                            cv2.circle(A, c, max(1, int(P(rr))), tuple(float(x) for x in hexc('#8a8e98')), 1, cv2.LINE_AA)
                        for q in range(5):
                            yq = ay0 + P(0.08 + 0.09 * q)
                            rect(A, ax + P(0.52), yq, ax + P(0.65), yq + max(P(0.02), 1), hexc('#7a7e88'))
                    else:
                        rect(A, ax + P(0.08), ay0 + P(0.07), ax + P(0.45), ay1 - P(0.07), hexc('#6a7078'))
                    rect(A, ax + P(0.6), ay1, ax + P(0.6) + max(P(0.025), 1), ay1 + P(0.9), hexc('#d0d2d8'))
                    rect(stains, ax, ay1, ax + P(0.7), ay1 + P(1.4), 0.8)
            if style == 'balcony':
                # balcony slab + railing band across the floor
                rect(A, 0, y - P(1.15), w, y - P(1.05), hexc('#b8bcc4') * 0.7)
                rail = np.zeros((h, w), np.float32)
                rect(rail, 0, y - P(1.05), w, y - P(0.05), 0.55)
                A[:] = A * (1 - rail[..., None]) + hexc('#9aa0aa') * 0.55 * rail[..., None]
                # railing bars
                step = max(P(0.12), 2)
                xb = 0.0
                while xb < w:
                    rect(A, xb, y - P(1.05), xb + max(P(0.025), 0.6), y - P(0.05), hexc('#6a707c'))
                    xb += step
        if style == 'apt' and rng.random() < 0.3:
            # a slim balcony rail across part of the floor
            bx0 = rng.uniform(0.0, 0.4) * w
            bx1 = bx0 + rng.uniform(0.3, 0.6) * w
            rect(A, bx0, y - P(1.0), bx1, y - P(0.94), hexc('#b8bcc8'))
            xb = bx0
            while xb < bx1:
                rect(A, xb, y - P(1.0), xb + max(P(0.022), 0.6), y - P(0.05), hexc('#5a5e6c'))
                xb += max(P(0.11), 2)
            rect(E, bx0, y - P(1.0), bx1, y - P(0.975), hexc('#ffb0e0') * 0.25)
        y = ytop
        fl += 1
    # drain pipes: pipe + neon glint line + wet run-off fan at the foot
    for _ in range(1 + int(rng.random() < 0.5)):
        px = rng.uniform(0.05, 0.95) * w
        pw_ = max(P(0.1), 1)
        rect(A, px, 0, px + pw_, gy0 - P(0.2), hexc('#8a8ea0'))
        rect(A, px + pw_ * 0.65, 0, px + pw_, gy0 - P(0.2), hexc('#4a4e5c'))
        rect(E, px + pw_ * 0.15, 0, px + pw_ * 0.35, gy0 - P(0.2),
             hexc(rng.choice(['#ff8ad0', '#8ae0ff'])) * 0.45)
        for yb in np.arange(P(1.0), gy0, P(1.6)):
            rect(A, px - P(0.03), yb, px + pw_ + P(0.03), yb + max(P(0.04), 1), hexc('#3a3c46'))
        rect(stains, px - P(0.25), 0, px + pw_ + P(0.25), gy0 - P(0.2), 0.6)
    # rain-dark streaking: vertical run-off below sills, units and pipes, broken by fine noise
    if stains.max() > 0:
        sn = noise(max(w // 2, 4), 6, max(4, w // 5), int(rng.integers(1e9)), 2)
        sn = cv2.resize(sn, (w, h), interpolation=cv2.INTER_CUBIC)
        st2 = stains * np.clip((sn - 0.3) * 2.2, 0, 1)
        st2 = cv2.GaussianBlur(st2, (0, 0), max(0.6, P(0.03)), max(0.6, P(0.2)))
        A *= (1 - 0.32 * st2)[..., None]
        # the wet streaks also carry a faint sheen of the neon
        E += hexc('#6a8a8c') * (st2 * 0.05)[..., None]
    # ambient neon bounce from the street canyon: keeps the darks a luminous blue-violet
    yy_ = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    E += A * (hexc('#4e6c70') * (0.10 + 0.08 * yy_))
    # parapet top
    rect(A, 0, 0, w, P(0.35), A[int(P(0.6)), w // 2] * 1.25)
    T['alb'] = np.clip(A, 0, 1)
    return T


def _paste(T, S, x, y):
    """Paste texture S into T at (x, y) px (integer-aligned, with alpha)."""
    h, w = S['a'].shape
    x, y = int(round(x)), int(round(y))
    H, W = T['a'].shape
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    ls = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    sl = (slice(y0, y1), slice(x0, x1))
    a = S['a'][ls][..., None]
    T['alb'][sl] = T['alb'][sl] * (1 - a) + S['alb'][ls] * a
    T['emi'][sl] = T['emi'][sl] * (1 - a) + S['emi'][ls] * a
    T['a'][sl] = np.maximum(T['a'][sl], S['a'][ls])


# ----------------------------------------------------------------------------- vending machine

def vending_front(rng, w, h, brand=None):
    """Front face of a Japanese drink vending machine (w x h px, ~1.0 x 1.83 m)."""
    brand = brand if brand is not None else hexc(rng.choice(['#e4002b', '#0050c8', '#f2f2f2', '#00873e']))
    T = tex(w, h, brand)
    h, w = T['a'].shape
    A, E = T['alb'], T['emi']
    white = hexc('#f4fbff')
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    A[:] = brand * (1.05 - 0.2 * yy)
    # header strip (lit, brand glyphs)
    rect(A, 0.05 * w, 0.015 * h, 0.95 * w, 0.055 * h, white)
    rect(E, 0.05 * w, 0.015 * h, 0.95 * w, 0.055 * h, white * 1.1)
    g = _row(rng, int(0.9 * w), max(3, int(0.04 * h)), 5, 0.14, 0.1)
    gh, gw = g.shape
    y0 = int(0.015 * h)
    E[y0:y0 + gh, int(0.05 * w):int(0.05 * w) + gw] *= (1 - 0.9 * g[..., None])
    A[y0:y0 + gh, int(0.05 * w):int(0.05 * w) + gw] = A[y0:y0 + gh, int(0.05 * w):int(0.05 * w) + gw] * (1 - g[..., None]) + brand * g[..., None]
    # display window
    x0, x1 = 0.07 * w, 0.93 * w
    y0, y1 = 0.07 * h, 0.58 * h
    rect(A, x0, y0, x1, y1, hexc('#20242c'))
    rect(E, x0, y0, x1, y1, hexc('#dcecff') * 0.7)
    rows = 3
    for r in range(rows):
        ry0 = y0 + (y1 - y0) * (r / rows) + 0.01 * h
        ry1 = y0 + (y1 - y0) * ((r + 1) / rows) - 0.04 * h
        n = 7
        cw = (x1 - x0) / n
        for i in range(n):
            col = hexc(rng.choice(['#e8303a', '#1f6fe0', '#f2c230', '#24a860', '#f07a28', '#b04ad8', '#e8e8e8',
                                   '#20b8c8', '#7a4a30', '#101010']))
            bx0 = x0 + i * cw + cw * 0.16
            bx1 = x0 + (i + 1) * cw - cw * 0.16
            bottle = rng.random() < 0.5
            tall = 1.0 if bottle else rng.uniform(0.62, 0.78)
            by0 = ry1 - (ry1 - ry0) * tall
            if bottle:  # neck + cap
                nx0, nx1 = bx0 + (bx1 - bx0) * 0.3, bx1 - (bx1 - bx0) * 0.3
                rect(A, nx0, by0, nx1, by0 + (ry1 - by0) * 0.25, col * 0.8)
                rect(E, nx0, by0, nx1, by0 + (ry1 - by0) * 0.25, col * 0.5)
                rect(A, nx0, by0, nx1, by0 + (ry1 - by0) * 0.06, hexc('#303030'))
                rect(E, nx0, by0, nx1, by0 + (ry1 - by0) * 0.06, (0.05, 0.05, 0.05))
                by0 = by0 + (ry1 - by0) * 0.25
            rect(A, bx0, by0, bx1, ry1, col * 0.2)
            rect(E, bx0, by0, bx1, ry1, col * 1.1 + 0.05)
            # label band + specular stripe
            lb0 = by0 + (ry1 - by0) * 0.35
            lb1 = ry1 - (ry1 - by0) * 0.2
            rect(A, bx0, lb0, bx1, lb1, white * 0.2)
            rect(E, bx0, lb0, bx1, lb1, white * 0.9 * (0.6 + 0.4 * rng.random()) + col * 0.2)
            rect(E, bx0 + (bx1 - bx0) * 0.2, by0 + 1, bx0 + (bx1 - bx0) * 0.32, ry1 - 1, white * 0.6)
            # price tag + button
            rect(A, bx0, ry1 + 0.006 * h, bx1, ry1 + 0.018 * h, white)
            rect(E, bx0, ry1 + 0.006 * h, bx1, ry1 + 0.018 * h, white * 0.5)
            bc = hexc('#40ff80') if rng.random() < 0.85 else hexc('#ff3040')
            rect(E, (bx0 + bx1) / 2 - cw * 0.1, ry1 + 0.022 * h, (bx0 + bx1) / 2 + cw * 0.1, ry1 + 0.03 * h, bc * 2.5)
        rect(A, x0, ry1 + 0.034 * h, x1, ry1 + 0.04 * h, hexc('#20242e'))
        rect(E, x0, ry1 + 0.034 * h, x1, ry1 + 0.04 * h, (0.02, 0.02, 0.03))
    # frame around the window
    rect(A, x0 - 0.01 * w, y0 - 0.006 * h, x1 + 0.01 * w, y0, hexc('#c8ccd4'))
    # lower panel
    rect(A, x0, 0.61 * h, 0.64 * w, 0.84 * h, brand * 1.1)
    rect(E, x0, 0.61 * h, 0.64 * w, 0.84 * h, brand * 0.35)
    rect(A, x0, 0.66 * h, 0.64 * w, 0.685 * h, white)
    rect(E, x0, 0.66 * h, 0.64 * w, 0.685 * h, white * 0.8)
    rect(A, 0.68 * w, 0.61 * h, 0.93 * w, 0.84 * h, hexc('#1c1f28'))
    rect(E, 0.72 * w, 0.63 * h, 0.89 * w, 0.66 * h, hexc('#50ff9a') * 1.4)
    rect(A, 0.76 * w, 0.7 * h, 0.85 * w, 0.72 * h, hexc('#9aa0aa'))
    rect(A, 0.15 * w, 0.87 * h, 0.85 * w, 0.96 * h, hexc('#0c0e14'))
    rect(A, 0.15 * w, 0.87 * h, 0.85 * w, 0.885 * h, hexc('#5a5e68'))
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :, None]
    A *= 1 - 0.3 * xx ** 8
    T['alb'] = np.clip(A, 0, 1)
    T['emi'] = E.astype(np.float32)
    return T


def vending_side(rng, w, h, brand):
    """Side panel of a vending machine: a painted product advert (bottle, glyph copy, gradient), a lit
    front edge (spill from the display), wet sheen and drips."""
    T = tex(w, h, brand * 0.85)
    h, w = T['a'].shape
    A, E = T['alb'], T['emi']
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    white = hexc('#ffffff')
    lum = float(brand.mean())
    dark = brand * 0.45 if lum < 0.6 else hexc('#b8c4d4')
    A[:] = brand * (1.0 - 0.25 * yy) * (1 - yy ** 3 * 0.3) + 0 * xx
    # diagonal colour swoosh
    sw = np.clip(1 - np.abs((yy * 1.0 - xx * 0.5) - 0.62) / 0.08, 0, 1)
    acc = hexc('#ffffff') if lum < 0.6 else hexc('#1a64d0')
    A[:] = A * (1 - 0.85 * sw) + acc * 0.85 * sw
    # bottle silhouette (rounded body, shoulder, neck, cap)
    m = np.zeros((h * 2, w * 2), np.uint8)
    cxp, bw = 0.5 * w * 2, 0.32 * w * 2
    top, bot = 0.2 * h * 2, 0.62 * h * 2
    pts = [(cxp - bw, bot), (cxp - bw, top + 0.09 * h * 2), (cxp - bw * 0.35, top), (cxp - bw * 0.35, top - 0.06 * h * 2),
           (cxp + bw * 0.35, top - 0.06 * h * 2), (cxp + bw * 0.35, top), (cxp + bw, top + 0.09 * h * 2), (cxp + bw, bot)]
    cv2.fillPoly(m, [np.array(pts, np.int32)], 255, cv2.LINE_AA)
    cv2.ellipse(m, (int(cxp), int(bot)), (int(bw), int(0.02 * h * 2)), 0, 0, 180, 255, -1, cv2.LINE_AA)
    bm = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    shade = np.clip(0.55 + 0.9 * np.cos((xx - 0.5) / 0.32 * 1.4), 0.3, 1.3)
    bcol = np.clip(brand * 1.25 + 0.12, 0, 1) * shade
    A[:] = A * (1 - bm) + bcol * bm
    # cap
    rect(A, 0.43 * w, 0.125 * h, 0.57 * w, 0.145 * h, hexc('#e8e8ec'))
    # label band with glyphs
    rect(A, 0.18 * w, 0.36 * h, 0.82 * w, 0.47 * h, white * 0.95)
    g = _row(rng, int(0.6 * w), max(3, int(0.08 * h)), 3, 0.16, 0.1)
    gh, gw = g.shape
    y0, x0 = int(0.375 * h), int(0.2 * w)
    A[y0:y0 + gh, x0:x0 + gw] = A[y0:y0 + gh, x0:x0 + gw] * (1 - g[..., None]) + brand * 0.8 * g[..., None]
    # bottle highlight stripe
    rect(A, 0.26 * w, 0.24 * h, 0.3 * w, 0.6 * h, np.clip(bcol.mean((0, 1)) * 0 + 1.0, 0, 1) * 0.95)
    # vertical copy (glyph column) near the front edge
    gc = glyph_column(rng, max(3, int(0.12 * w)), int(0.5 * h), 4, weight=0.14, margin=0.05)
    gh, gw = gc.shape
    y0, x0 = int(0.15 * h), int(0.82 * w)
    x0 = min(x0, w - gw)
    A[y0:y0 + gh, x0:x0 + gw] = A[y0:y0 + gh, x0:x0 + gw] * (1 - gc[..., None]) + white * gc[..., None]
    # lower dark kick panel + vents
    rect(A, 0, 0.9 * h, w, h, hexc('#16181e'))
    for k in range(4):
        rect(A, 0.2 * w, (0.915 + 0.018 * k) * h, 0.8 * w, (0.922 + 0.018 * k) * h, hexc('#3a3e48'))
    # wet sheen: vertical drip streaks and a soft top-to-bottom glossy band
    n = noise(max(w // 2, 4), 6, max(4, w // 6), int(rng.integers(1e9)), 2)
    n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
    A[:] = A * (0.85 + 0.25 * n[..., None])
    # emissive: advert is backlit faintly + strong spill along the front edge (right, toward the display)
    E[:] = A * 0.28 + brand * 0.05
    edge = np.clip((xx - 0.86) / 0.14, 0, 1) ** 2
    E[:] += hexc('#e8f4ff') * edge * 0.9 * (0.4 + 0.6 * (1 - yy))
    # metal trims
    rect(A, 0, 0, 0.03 * w, h, hexc('#c8ccd4'))
    rect(A, 0.97 * w, 0, w, h, hexc('#e8ecf2'))
    rect(E, 0.97 * w, 0, w, h, hexc('#e8f4ff') * 1.2)
    rect(A, 0, 0, w, 0.012 * h, hexc('#c8ccd4'))
    T['alb'] = np.clip(A, 0, 1)
    T['emi'] = E.astype(np.float32)
    return T


# ----------------------------------------------------------------------------- utility pole

def pole_tex(rng, wpx, hpx, hm, sleeve=True, plate=True):
    """Concrete utility pole as a camera-facing sprite texture (cylinder shading)."""
    T = tex(wpx, hpx, hexc('#8c8a86'))
    h, w = T['a'].shape
    xx = np.linspace(-1, 1, w, dtype=np.float32)
    # cylinder: lit from the street side (right) by the city glow, rim on the left
    shade = 0.45 + 0.5 * np.clip(0.3 + 0.7 * xx, 0, 1) ** 0.8
    shade = shade * (1 - 0.35 * np.abs(xx) ** 6)
    A = T['alb'] * shade[None, :, None]
    # slight taper: alpha narrower at the top
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    half = 0.78 + 0.22 * yy
    a = np.clip((half - np.abs(xx)[None, :]) * w * 0.5, 0, 1)
    ppm = h / hm
    # stains / bands
    n = noise(4, h, 3, rng.integers(1e9), 3)
    A *= (0.85 + 0.25 * cv2.resize(n, (w, h)))[..., None]
    if sleeve:  # yellow/black guard sleeve at the bottom
        y0 = h - 1.8 * ppm
        y1 = h - 0.2 * ppm
        yl = np.arange(h, dtype=np.float32)[:, None]
        stripe = ((np.floor((yl + np.linspace(0, 1, w)[None, :] * 0.3 * ppm) / (0.18 * ppm)) % 2) == 0)
        m = ((yl >= y0) & (yl <= y1)).astype(np.float32)
        col = np.where(stripe[..., None], hexc('#f2c200'), hexc('#1a1a1a'))
        A = A * (1 - m[..., None]) + col * shade[None, :, None] * m[..., None]
    if plate:   # blue address plate with glyphs
        py0 = h - 3.2 * ppm
        py1 = h - 2.2 * ppm
        pm = np.zeros((h, w), np.float32)
        rect(pm, w * 0.04, py0, w * 0.96, py1, 1.0)
        A = A * (1 - pm[..., None]) + hexc('#1f4fb0') * pm[..., None]
        g = glyph_column(rng, max(int(w * 0.9), 3), max(int(py1 - py0), 3), 3, 0.12, 0.1, pool='addr')
        gp = np.zeros((h, w), np.float32)
        gh, gw = g.shape
        gp[int(py0):int(py0) + gh, int(w * 0.05):int(w * 0.05) + gw] = g[:max(0, min(gh, h - int(py0))), :max(0, min(gw, w - int(w * 0.05)))]
        A = A * (1 - gp[..., None]) + hexc('#f2f4f8') * gp[..., None]
    T['alb'] = np.clip(A, 0, 1).astype(np.float32)
    T['a'] = a.astype(np.float32)
    return T


def lantern_tex(w, h, col='#e02820', glow=2.0, rng=None):
    """Red paper lantern (chochin) sprite."""
    T = tex(w, h, hexc(col))
    h, w = T['a'].shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + 0.5) / w * 2 - 1
    v = (yy + 0.5) / h * 2 - 1
    body = np.clip((1 - (u ** 2 + (v * 0.95) ** 2 * 0.0) - np.abs(v) ** 6 * 0.0), 0, 1)
    prof = np.sqrt(np.clip(1 - v ** 2 * 0.55, 0, 1))  # barrel profile
    a = np.clip((prof - np.abs(u)) * w * 0.5, 0, 1)
    caps = (np.abs(v) > 0.86).astype(np.float32)
    ribs = 0.8 + 0.2 * (np.sin(v * 40) > 0.6)
    c = hexc(col)
    shade = (1 - 0.5 * u ** 2)
    E = c * (glow * shade * ribs)[..., None] * (1 - caps)[..., None]
    E += hexc('#ffd080') * (glow * 0.6 * np.exp(-(u ** 2 + v ** 2) * 2.5))[..., None] * (1 - caps)[..., None]
    A = c * shade[..., None]
    A = A * (1 - caps[..., None]) + hexc('#151515') * caps[..., None]
    # a black glyph in the middle
    if rng is not None:
        gs = max(int(min(w, h) * 0.55), 2)
        g = glyph(rng, gs, 0.12, pool='lantern')
        gm = np.zeros((h, w), np.float32)
        y0, x0 = (h - gs) // 2, (w - gs) // 2
        gm[y0:y0 + gs, x0:x0 + gs] = g
        E *= (1 - 0.85 * gm[..., None])
        A = A * (1 - gm[..., None]) + 0.05 * gm[..., None]
    T['alb'], T['emi'], T['a'] = A.astype(np.float32), E.astype(np.float32), a.astype(np.float32)
    return T
