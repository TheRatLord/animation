"""s04_rain_street detail painters (round 2): vending machine fronts with real product silhouettes,
bookshop windows, weathered wooden lattice, sign wear/bolts/secondary copy, droplets & wet sheen."""
import math
import numpy as np
import cv2

from s04_rain_street_art import hexc, rect, tex, glyph, noise, _row, glyph_column

Q = 4          # sub-pixel shift bits for cv2 drawing (1 << 2)


def _pts(pts):
    return np.array([[int(round(x * Q)), int(round(y * Q))] for x, y in pts], np.int32)


def fill(mask, pts, val=1.0):
    cv2.fillPoly(mask, [_pts(pts)], float(val), cv2.LINE_AA, shift=2)


def ellipse(mask, cx, cy, rx, ry, val=1.0, a0=0, a1=360, thick=-1):
    cv2.ellipse(mask, (int(round(cx * Q)), int(round(cy * Q))), (max(1, int(round(rx * Q))), max(1, int(round(ry * Q)))),
                0, a0, a1, float(val), thick, cv2.LINE_AA, shift=2)


def paint(img, mask, col, gain=1.0):
    m = mask[..., None] * gain
    img[:] = img * (1 - m) + np.asarray(col, np.float32) * m


def add(img, mask, col, gain=1.0):
    img[:] = img + np.asarray(col, np.float32) * (mask * gain)[..., None]


# ----------------------------------------------------------------------------- droplets / wet film

def droplets(A, E, rng, x0, y0, x1, y1, n, rmin, rmax, light=(1.0, 1.0, 1.0), trails=0.3, gain=1.0):
    """Water droplets on a vertical surface (glass or painted metal): darker refracting body with a bright
    specular pip (top-left) and a lit lower meniscus; some leave a thin wet trail above them."""
    h, w = A.shape[:2]
    light = np.asarray(light, np.float32)
    for _ in range(int(n)):
        r = rng.uniform(rmin, rmax) * (1.0 if rng.random() < 0.8 else 1.8)
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        ry = r * rng.uniform(1.0, 1.35)
        L = rng.uniform(4, 18) * r if rng.random() < trails else 0.0
        px0, py0 = int(max(x - r - 3, 0)), int(max(y - ry - L - 3, 0))
        px1, py1 = int(min(x + r + 4, w)), int(min(y + ry + 4, h))
        if px1 <= px0 or py1 <= py0:
            continue
        lx, ly = x - px0, y - py0
        ph, pw = py1 - py0, px1 - px0
        body = np.zeros((ph, pw), np.float32)
        hi = np.zeros((ph, pw), np.float32)
        low = np.zeros((ph, pw), np.float32)
        tr = np.zeros((ph, pw), np.float32)
        ellipse(body, lx, ly, r, ry, 1.0)
        ellipse(hi, lx - r * 0.35, ly - ry * 0.4, max(0.35, r * 0.28), max(0.35, r * 0.24), 1.0)
        ellipse(low, lx, ly + ry * 0.35, r * 0.7, ry * 0.35, 1.0)
        if L > 0:
            cv2.line(tr, (int(lx * Q), int((ly - L) * Q)), (int(lx * Q), int(ly * Q)), 1.0, max(1, int(r * 0.6)),
                     cv2.LINE_AA, shift=2)
        Av, Ev = A[py0:py1, px0:px1], E[py0:py1, px0:px1]
        Av[:] = Av * (1 - 0.25 * body[..., None])
        Ev[:] = Ev * (1 - 0.35 * body[..., None]) + light * (hi * 1.6 * gain + low * 0.35 * gain +
                                                              tr * 0.12 * gain)[..., None]


def wet_sheen(A, E, rng, strength=0.25, light=(0.85, 0.9, 1.0), stripe_cells=None):
    """Vertical run-off streaks + soft glossy vertical highlight bands."""
    h, w = A.shape[:2]
    cells = stripe_cells or max(6, w // 5)
    st = noise(max(w // 2, 4), 6, cells, int(rng.integers(1e9)), 3)
    st = cv2.resize(st, (w, h), interpolation=cv2.INTER_CUBIC)
    lo = noise(max(w // 4, 4), 3, max(3, cells // 4), int(rng.integers(1e9)), 2)
    lo = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    A[:] = A * (0.82 + 0.3 * st)[..., None]
    sheen = np.clip((st - 0.66) * 6, 0, 1) ** 2 * np.clip((lo - 0.5) * 3, 0, 1) * (0.4 + 0.6 * yy)
    E[:] = E + np.asarray(light, np.float32) * (sheen * strength)[..., None]


# ----------------------------------------------------------------------------- products

def _can(A, E, rng, x0, y0, x1, y1, col, panel):
    """Drink can silhouette (upright) with cylinder shading, label band and specular strip."""
    w, h = x1 - x0, y1 - y0
    m = np.zeros(A.shape[:2], np.float32)
    rr = max(0.5, h * 0.035)
    fill(m, [(x0 + w * 0.08, y0 + rr), (x1 - w * 0.08, y0 + rr), (x1, y0 + rr * 2.5), (x1, y1 - rr),
             (x0, y1 - rr), (x0, y0 + rr * 2.5)])
    ellipse(m, (x0 + x1) / 2, y1 - rr, w / 2, rr, 1.0)
    ellipse(m, (x0 + x1) / 2, y0 + rr, w / 2 * 0.84, rr, 1.0)
    _shade_product(A, E, m, x0, x1, y0, y1, col, panel, rng, label=(0.3, 0.72))
    # silver lid and bottom rims
    lid = np.zeros_like(m)
    rect(lid, x0 + w * 0.08, y0, x1 - w * 0.08, y0 + rr * 2.4, 1.0)
    lid *= m
    paint(A, lid, hexc('#c8ccd4'))
    add(E, lid, panel * 0.55)


def _bottle(A, E, rng, x0, y0, x1, y1, col, panel, cap):
    """PET bottle: cap, neck ring, curved shoulder, label, ridged base."""
    w, h = x1 - x0, y1 - y0
    cx = (x0 + x1) / 2
    m = np.zeros(A.shape[:2], np.float32)
    nw = w * 0.2
    pts = [(cx - nw, y0 + h * 0.07), (cx + nw, y0 + h * 0.07), (cx + nw, y0 + h * 0.16)]
    for k in range(1, 9):
        a = k / 8 * math.pi / 2
        pts.append((cx + nw + (w / 2 - nw) * math.sin(a), y0 + h * 0.16 + h * 0.16 * (1 - math.cos(a))))
    pts += [(x1, y1 - h * 0.03), (x1 - w * 0.08, y1), (x0 + w * 0.08, y1), (x0, y1 - h * 0.03)]
    for k in range(8, 0, -1):
        a = k / 8 * math.pi / 2
        pts.append((cx - nw - (w / 2 - nw) * math.sin(a), y0 + h * 0.16 + h * 0.16 * (1 - math.cos(a))))
    pts += [(cx - nw, y0 + h * 0.16)]
    fill(m, pts)
    _shade_product(A, E, m, x0, x1, y0, y1, col, panel, rng, label=(0.42, 0.78), liquid=True)
    c = np.zeros_like(m)
    rect(c, cx - nw * 1.15, y0, cx + nw * 1.15, y0 + h * 0.075, 1.0)
    paint(A, c, cap)
    add(E, c, cap * panel * 0.9)
    ring = np.zeros_like(m)
    rect(ring, cx - nw * 1.3, y0 + h * 0.075, cx + nw * 1.3, y0 + h * 0.09, 1.0)
    add(E, ring, panel * 0.5)
    # base ridges
    for k in range(3):
        yk = y1 - h * (0.03 + 0.035 * k)
        rg = np.zeros_like(m)
        rect(rg, x0 + w * 0.06, yk - max(0.5, h * 0.006), x1 - w * 0.06, yk, 1.0)
        E[:] = E - (rg * 0.12)[..., None] * E


def _shade_product(A, E, m, x0, x1, y0, y1, col, panel, rng, label=(0.3, 0.7), liquid=False):
    h, w = A.shape[:2]
    xi0, xi1 = max(int(x0) - 1, 0), min(int(x1) + 2, w)
    yi0, yi1 = max(int(y0) - 1, 0), min(int(y1) + 2, h)
    if xi1 <= xi0 or yi1 <= yi0:
        return
    sl = (slice(yi0, yi1), slice(xi0, xi1))
    mm = m[sl]
    xx = (np.arange(xi0, xi1, dtype=np.float32)[None, :] - x0) / max(x1 - x0, 1) * 2 - 1
    yy = (np.arange(yi0, yi1, dtype=np.float32)[:, None] - y0) / max(y1 - y0, 1)
    cyl = np.clip(np.sqrt(np.clip(1 - xx ** 2, 0, 1)), 0, 1)
    shade = (0.35 + 0.75 * cyl) * (1 - 0.15 * yy)
    col = np.asarray(col, np.float32)
    lab = ((yy > label[0]) & (yy < label[1])).astype(np.float32)
    lcol = np.asarray(hexc(rng.choice(['#ffffff', '#f2e6c8', '#202020', '#e8f0ff'])), np.float32)
    base = col[None, None, :] * (1 - lab[..., None]) + (lcol * 0.8 + col * 0.2)[None, None, :] * lab[..., None]
    # tiny logo mark on the label
    logo = np.exp(-(xx / 0.25) ** 2 - ((yy - (label[0] + label[1]) / 2) / 0.05) ** 2) * lab
    base = base * (1 - logo[..., None] * 0.9) + col[None, None, :] * logo[..., None] * 0.9
    shaded = base * shade[..., None]
    em = shaded * panel * (1.1 if not liquid else 1.25)
    if liquid:  # backlit liquid glows through the edges
        em = em + col * panel * 0.5 * (1 - lab[..., None]) * (1 - cyl[..., None] * 0.5)
    spec = np.exp(-((xx + 0.48) / 0.09) ** 2) * 1.1 + np.exp(-((xx - 0.62) / 0.05) ** 2) * 0.35
    em = em + panel * spec[..., None] * 0.9
    k = mm[..., None]
    A[sl] = A[sl] * (1 - k) + shaded * 0.35 * k
    E[sl] = E[sl] * (1 - k) + em * k


# ----------------------------------------------------------------------------- vending machine

def vending_front(rng, w, h, brand):
    """Front face of a Japanese drink vending machine (~1.0 x 1.83 m), painted at w x h px."""
    brand = np.asarray(brand, np.float32)
    T = tex(w, h, brand)
    h, w = T['a'].shape
    A, E = T['alb'], T['emi']
    white = hexc('#f4fbff')
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    A[:] = brand * (1.05 - 0.25 * yy) * (1 - 0.25 * np.abs(xx - 0.5) ** 3)
    E[:] = brand * 0.06
    # header lightbox with brand copy
    hx0, hx1, hy0, hy1 = 0.05 * w, 0.95 * w, 0.012 * h, 0.06 * h
    rect(A, hx0, hy0, hx1, hy1, white)
    hg = np.clip(1 - ((xx - 0.5) / 0.6) ** 2, 0, 1)
    E[int(hy0):int(hy1), int(hx0):int(hx1)] = (white * 1.3 * (0.75 + 0.25 * hg))[0, int(hx0):int(hx1)][None]
    g = _row(rng, int(hx1 - hx0), max(3, int((hy1 - hy0) * 0.8)), 6, 0.15, 0.08, pool='vend_head')
    gh, gw = g.shape
    y0 = int(hy0 + (hy1 - hy0 - gh) / 2)
    sl = (slice(y0, y0 + gh), slice(int(hx0), int(hx0) + gw))
    A[sl] = A[sl] * (1 - g[..., None]) + brand * g[..., None]
    E[sl] = E[sl] * (1 - 0.9 * g[..., None]) + brand * 0.4 * g[..., None]
    # display window: glowing panel behind the product rows
    x0, x1 = 0.065 * w, 0.935 * w
    y0, y1 = 0.075 * h, 0.6 * h
    rect(A, x0 - 0.012 * w, y0 - 0.008 * h, x1 + 0.012 * w, y1 + 0.008 * h, hexc('#c8ccd4'))
    rect(E, x0 - 0.012 * w, y0 - 0.008 * h, x1 + 0.012 * w, y1 + 0.008 * h, hexc('#dfefff') * 0.35)
    rows = 3
    rh = (y1 - y0) / rows
    ncol = 8 if rng.random() < 0.5 else 9
    cw = (x1 - x0) / ncol
    palette = ['#e8303a', '#1f6fe0', '#f2c230', '#24a860', '#f07a28', '#b04ad8', '#f2f2f2', '#20b8c8', '#8a5a30',
               '#151515', '#e86aa0', '#c8e84a']
    for r in range(rows):
        ry0, ry1 = y0 + r * rh, y0 + (r + 1) * rh
        sy1 = ry1 - rh * 0.28          # product base line
        # backlight: fluorescent tube at the top of each bay, falling off downward
        by = (np.arange(int(ry0), int(ry1), dtype=np.float32) - ry0) / rh
        bl = hexc('#eaf4ff') * (1.7 * np.exp(-by / 0.35) + 0.45)[:, None]
        rect(A, x0, ry0, x1, ry1, hexc('#20242c'))
        E[int(ry0):int(ry1), int(x0):int(x1)] = bl[:, None, :] * (0.9 + 0.1 * np.cos(
            (np.arange(int(x0), int(x1)) - x0) / (x1 - x0) * np.pi * 2))[None, :, None]
        rect(E, x0, ry0, x1, ry0 + rh * 0.04, white * 3.0)
        panel = hexc('#f0f6ff') * 1.2
        for i in range(ncol):
            col = hexc(rng.choice(palette))
            bx0 = x0 + i * cw + cw * 0.08
            bx1 = x0 + (i + 1) * cw - cw * 0.08
            kind = rng.random()
            ox, oy = int(bx0) - 2, int(ry0)
            Av = A[oy:int(sy1) + 3, ox:int(bx1) + 3]
            Ev = E[oy:int(sy1) + 3, ox:int(bx1) + 3]
            cx = (bx0 + bx1) / 2
            if kind < 0.5:
                ph = rh * rng.uniform(0.56, 0.62)
                cw2 = min(ph / 2.7, (bx1 - bx0))
                _bottle(Av, Ev, rng, cx - cw2 / 2 - ox, sy1 - ph - oy, cx + cw2 / 2 - ox, sy1 - oy, col, panel,
                        hexc(rng.choice(['#f2f2f2', '#e8303a', '#1f6fe0', '#24a860', '#f2c230'])))
            else:
                ph = rh * rng.uniform(0.36, 0.42) * (1.35 if rng.random() < 0.3 else 1.0)
                cw2 = min(ph / (1.85 if ph < rh * 0.45 else 2.5), (bx1 - bx0))
                _can(Av, Ev, rng, cx - cw2 / 2 - ox, sy1 - ph - oy, cx + cw2 / 2 - ox, sy1 - oy, col, panel)
            # shelf lip, price label and push button
            lx0, lx1 = x0 + i * cw + cw * 0.12, x0 + (i + 1) * cw - cw * 0.12
            rect(A, lx0, sy1 + rh * 0.03, lx1, sy1 + rh * 0.1, white)
            rect(E, lx0, sy1 + rh * 0.03, lx1, sy1 + rh * 0.1, white * 0.75)
            for d in range(3):   # price digits
                dx = lx0 + (lx1 - lx0) * (0.22 + 0.2 * d)
                rect(E, dx, sy1 + rh * 0.045, dx + (lx1 - lx0) * 0.12, sy1 + rh * 0.085, -white * 0.6)
                rect(A, dx, sy1 + rh * 0.045, dx + (lx1 - lx0) * 0.12, sy1 + rh * 0.085, hexc('#202020'))
            sold = rng.random() < 0.12
            bc = hexc('#ff3a3a') if sold else hexc(rng.choice(['#40c8ff', '#40ff90']))
            ox, oy = int(lx0) - 2, int(sy1)
            Av = A[oy:int(sy1 + rh * 0.25), ox:int(lx1) + 3]
            Ev = E[oy:int(sy1 + rh * 0.25), ox:int(lx1) + 3]
            bm = np.zeros(Av.shape[:2], np.float32)
            bxc, byc = (lx0 + lx1) / 2 - ox, sy1 + rh * 0.155 - oy
            ellipse(bm, bxc, byc, (lx1 - lx0) * 0.3, rh * 0.03, 1.0)
            paint(Av, bm, bc * 0.6)
            add(Ev, bm, bc, 2.2)
            bm[:] = 0
            ellipse(bm, bxc - (lx1 - lx0) * 0.08, byc - rh * 0.01, (lx1 - lx0) * 0.08, rh * 0.008, 1.0)
            add(Ev, bm, white, 0.6)
        # shelf rail shadow line
        rect(A, x0, ry1 - rh * 0.045, x1, ry1, hexc('#14161c'))
        rect(E, x0, ry1 - rh * 0.045, x1, ry1, -hexc('#f0f0f0') * 0.0)
        E[int(ry1 - rh * 0.045):int(ry1), int(x0):int(x1)] *= 0.1
    # glass: two soft diagonal reflection bands + fine vertical run-off
    gx = np.zeros((h, w), np.float32)
    gy_ = np.arange(h, dtype=np.float32)[:, None]
    gx_ = np.arange(w, dtype=np.float32)[None, :]
    d = (gx_ - x0) / w + (gy_ - y0) / h * 0.55
    refl = 0.14 * np.exp(-((d - 0.3) / 0.035) ** 2) + 0.07 * np.exp(-((d - 0.42) / 0.015) ** 2)
    win = np.zeros((h, w), np.float32)
    rect(win, x0, y0, x1, y1, 1.0)
    E[:] = E + hexc('#e8f0ff') * (refl * win)[..., None]
    # lower half: coin/bill panel, keypad, big dispensing slot
    rect(A, 0.065 * w, 0.625 * h, 0.64 * w, 0.84 * h, brand * 1.1)
    rect(E, 0.065 * w, 0.625 * h, 0.64 * w, 0.84 * h, brand * 0.28)
    gg = _row(rng, int(0.5 * w), max(3, int(0.05 * h)), 5, 0.15, 0.08, pool='vend_panel')
    gh, gw = gg.shape
    sl = (slice(int(0.68 * h), int(0.68 * h) + gh), slice(int(0.1 * w), int(0.1 * w) + gw))
    A[sl] = A[sl] * (1 - gg[..., None]) + white * gg[..., None]
    E[sl] = E[sl] + white * 0.9 * gg[..., None]
    rect(A, 0.67 * w, 0.625 * h, 0.935 * w, 0.84 * h, hexc('#1c1f28'))
    rect(E, 0.7 * w, 0.64 * h, 0.9 * w, 0.675 * h, hexc('#50ff9a') * 1.6)          # LCD
    for k in range(3):
        rect(E, 0.73 * w + k * 0.055 * w, 0.655 * h, 0.76 * w + k * 0.055 * w, 0.662 * h, -hexc('#50ff9a') * 1.2)
    rect(A, 0.74 * w, 0.7 * h, 0.87 * w, 0.715 * h, hexc('#9aa0aa'))              # coin slot
    rect(A, 0.72 * w, 0.74 * h, 0.9 * w, 0.765 * h, hexc('#0c0c10'))              # bill slot
    rect(E, 0.72 * w, 0.765 * h, 0.9 * w, 0.769 * h, hexc('#ffb040') * 1.2)
    rect(A, 0.78 * w, 0.79 * h, 0.84 * w, 0.82 * h, hexc('#c8ccd4'))              # return lever
    # dispensing slot with a lit flap edge
    rect(A, 0.12 * w, 0.865 * h, 0.88 * w, 0.955 * h, hexc('#07080c'))
    rect(A, 0.12 * w, 0.865 * h, 0.88 * w, 0.878 * h, hexc('#6a6e78'))
    rect(E, 0.12 * w, 0.865 * h, 0.88 * w, 0.869 * h, hexc('#dfefff') * 0.7)
    rect(A, 0, 0.97 * h, w, h, hexc('#0a0a0e'))
    # bevelled side edges catch the street neon
    rect(E, 0, 0, 0.012 * w, 0.97 * h, hexc('#ff7ac8') * 0.35)
    rect(E, 0.988 * w, 0, w, 0.97 * h, hexc('#8ad8ff') * 0.35)
    # wet sheen and droplets all over the front (bigger on the glass)
    wet_sheen(A, E, rng, strength=0.18)
    s = h / 1000.0
    droplets(A, E, rng, x0, y0, x1, y1, 260 * (w * h) / 2.5e6 + 60, 1.2 * s, 3.2 * s, light=hexc('#f4f8ff'),
             trails=0.35, gain=0.9)
    droplets(A, E, rng, 0, y1, w, h * 0.96, 120 * (w * h) / 2.5e6 + 25, 1.0 * s, 2.6 * s, light=hexc('#f4f8ff'),
             trails=0.4, gain=0.6)
    T['alb'] = np.clip(A, 0, 1).astype(np.float32)
    T['emi'] = np.maximum(E, 0).astype(np.float32)
    return T


def vending_side(rng, w, h, brand):
    """Side panel: printed product advert (big can), glyph copy, spill of the display light along the front
    edge, wet sheen and droplets. No stray swooshes."""
    brand = np.asarray(brand, np.float32)
    T = tex(w, h, brand * 0.85)
    h, w = T['a'].shape
    A, E = T['alb'], T['emi']
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
    white = hexc('#ffffff')
    lum = float(brand.mean())
    A[:] = brand * (1.05 - 0.3 * yy)
    # advert: large can with cylinder shading, crisp edges
    m = np.zeros((h, w), np.float32)
    cx, cw2 = 0.5 * w, 0.3 * w
    top, bot = 0.2 * h, 0.66 * h
    fill(m, [(cx - cw2 * 0.85, top), (cx + cw2 * 0.85, top), (cx + cw2, top + 0.03 * h), (cx + cw2, bot),
             (cx - cw2, bot), (cx - cw2, top + 0.03 * h)])
    ellipse(m, cx, bot, cw2, 0.015 * h, 1.0)
    acc = hexc('#e8ecf4') if lum < 0.55 else hexc('#1a64d0')
    xc = np.clip((xx[..., 0] - (cx - cw2) / w) / (2 * cw2 / w) * 2 - 1, -1, 1)
    cyl = np.sqrt(np.clip(1 - xc ** 2, 0, 1))
    body = acc * (0.25 + 0.8 * cyl[..., None] ** 1.5)
    body = body * (1 - 0.35 * np.exp(-((xc - 0.55) / 0.2) ** 2))[..., None] + brand * 0.15
    body = body + white * (np.exp(-((xc + 0.5) / 0.06) ** 2) * 0.7 + np.exp(-((xc - 0.75) / 0.04) ** 2) * 0.3)[..., None]
    # condensation beads printed on the can (it is an advert for a cold drink)
    bead = noise(w, h, max(8, w // 6), int(rng.integers(1e9)), 2)
    body = body + white * (np.clip((bead - 0.72) * 6, 0, 1) * 0.35)[..., None]
    A[:] = A * (1 - m[..., None]) + np.clip(body, 0, 1) * m[..., None]
    band = np.zeros((h, w), np.float32)
    rect(band, cx - cw2, 0.38 * h, cx + cw2, 0.48 * h, 1.0)
    band *= m
    A[:] = A * (1 - band[..., None]) + brand * (0.6 + 0.5 * cyl[..., None]) * band[..., None]
    g = _row(rng, int(1.6 * cw2), max(3, int(0.07 * h)), 4, 0.16, 0.1, pool='vend_can')
    gh, gw = g.shape
    sl = (slice(int(0.395 * h), int(0.395 * h) + gh), slice(int(cx - 0.8 * cw2), int(cx - 0.8 * cw2) + gw))
    A[sl] = A[sl] * (1 - g[..., None]) + white * g[..., None]
    lid = np.zeros((h, w), np.float32)
    rect(lid, cx - cw2 * 0.85, top - 0.012 * h, cx + cw2 * 0.85, top + 0.01 * h, 1.0)
    A[:] = A * (1 - lid[..., None]) + hexc('#d8dce4') * lid[..., None]
    # vertical copy
    gc = glyph_column(rng, max(3, int(0.12 * w)), int(0.5 * h), 5, weight=0.14, margin=0.05, pool='vend_v')
    gh, gw = gc.shape
    y0, x0 = int(0.14 * h), min(int(0.84 * w), w - gw)
    A[y0:y0 + gh, x0:x0 + gw] = A[y0:y0 + gh, x0:x0 + gw] * (1 - gc[..., None]) + white * gc[..., None]
    # kick panel + vents
    rect(A, 0, 0.9 * h, w, h, hexc('#16181e'))
    for k in range(4):
        rect(A, 0.2 * w, (0.915 + 0.018 * k) * h, 0.8 * w, (0.922 + 0.018 * k) * h, hexc('#3a3e48'))
    E[:] = A * 0.22 + (m * 0.25)[..., None] * A
    edge = np.clip((xx - 0.82) / 0.18, 0, 1) ** 2
    E[:] += hexc('#e8f4ff') * edge * 0.8 * (0.4 + 0.6 * (1 - yy))
    rect(A, 0, 0, 0.03 * w, h, hexc('#c8ccd4'))
    rect(A, 0.97 * w, 0, w, h, hexc('#e8ecf2'))
    rect(E, 0.97 * w, 0, w, h, hexc('#e8f4ff') * 1.2)
    rect(A, 0, 0, w, 0.012 * h, hexc('#c8ccd4'))
    wet_sheen(A, E, rng, strength=0.12)
    s = h / 1000.0
    droplets(A, E, rng, 0, 0.02 * h, w, 0.9 * h, 80 * (w * h) / 1.5e6 + 20, 1.0 * s, 2.6 * s,
             light=hexc('#f4f8ff'), trails=0.4, gain=0.5)
    T['alb'] = np.clip(A, 0, 1).astype(np.float32)
    T['emi'] = np.maximum(E, 0).astype(np.float32)
    return T


# ----------------------------------------------------------------------------- bookshop window

def bookshelves(As, Es, rng, gx0, gy0, gx1, gy1, P):
    """Interior of a lit bookshop seen through glass: shelves of books with varied heights, widths and
    tilts, lit from ceiling lamps with falloff; glass with street reflections and condensation."""
    hh, ww = As.shape[:2]
    inter = np.zeros((hh, ww, 3), np.float32)
    xs = np.arange(ww, dtype=np.float32)[None, :]
    ys = np.arange(hh, dtype=np.float32)[:, None]
    # lamp falloff: ceiling spots across the width
    lamps = np.zeros((hh, ww), np.float32)
    nl = max(2, int((gx1 - gx0) / P(1.3)))
    for k in range(nl):
        lx = gx0 + (gx1 - gx0) * (k + 0.5) / nl
        lamps += np.exp(-((xs - lx) / P(0.75)) ** 2 - ((ys - gy0) / P(1.4)) ** 2)
    light = 0.35 + 1.1 * np.clip(lamps, 0, 1.3)
    wall = hexc('#e8d4b0')
    inter[:] = wall * (light * 0.55)[..., None]
    # ceiling lamps themselves
    for k in range(nl):
        lx = gx0 + (gx1 - gx0) * (k + 0.5) / nl
        rect(inter, lx - P(0.18), gy0 + P(0.05), lx + P(0.18), gy0 + P(0.1), hexc('#fff8e8') * 3.0)
    sh_top = gy0 + P(0.45)
    shelf_h = P(0.36)
    book_cols = ['#c83a3a', '#3a6ac8', '#e8c040', '#3a9a5a', '#e87a30', '#8a4ac8', '#f0ece0', '#2a2a30',
                 '#d86a8a', '#40a8b8', '#8a6a40', '#f2d8a8', '#5a2a2a', '#1a3a6a']
    # each bay has its own dominant palette (manga sets, bunko white spines, art books...)
    themes = [book_cols, ['#f4f0e4', '#e8e0cc', '#f8f4ec', '#d8d0bc', '#c83a3a'], ['#1a3a6a', '#3a6ac8', '#40a8b8', '#f0ece0'],
              ['#e87a30', '#e8c040', '#c83a3a', '#f2d8a8'], ['#8a4ac8', '#d86a8a', '#f0ece0', '#2a2a30'],
              ['#3a9a5a', '#c8e84a', '#f0ece0', '#1a1a1a']]
    run_n, run_col = 0, None
    r = 0
    while True:
        sy = sh_top + r * (shelf_h + P(0.04))
        if sy + shelf_h > gy1 - P(0.05):
            break
        base = sy + shelf_h
        # shelf back shadow
        sh = np.exp(-((ys - base) / P(0.12)) ** 2) * (ys < base)
        inter *= (1 - 0.35 * sh)[..., None]
        x = gx0 + P(rng.uniform(0.02, 0.1))
        while x < gx1 - P(0.05):
            if rng.random() < 0.06:          # gap
                x += P(rng.uniform(0.05, 0.2))
                continue
            bay = int((x - gx0) / max(P(1.0), 1)) + 7 * r
            theme = themes[(bay * 7 + 3) % len(themes)]
            if run_n <= 0:
                if rng.random() < 0.3:          # a numbered set of the same series
                    run_n = int(rng.integers(4, 12))
                    run_col = hexc(rng.choice(theme)) * rng.uniform(0.8, 1.05)
                    run_h = shelf_h * rng.uniform(0.62, 0.8)
                    run_w = P(rng.uniform(0.02, 0.03))
                else:
                    run_col = None
            if run_col is not None:
                run_n -= 1
                bw, bh = run_w, run_h
                col = run_col * rng.uniform(0.93, 1.03)
            else:
                run_n = 0
                bw = P(rng.uniform(0.022, 0.05))
                bh = shelf_h * (rng.uniform(0.45, 0.95) if rng.random() < 0.8 else rng.uniform(0.3, 0.5))
                col = hexc(rng.choice(theme)) * rng.uniform(0.75, 1.05)
            tilt = 0.0
            if rng.random() < 0.16:
                tilt = rng.uniform(0.12, 0.45) * (1 if rng.random() < 0.5 else -1)
            dx = bh * tilt
            ox, oy = int(max(min(x, x + dx) - 2, 0)), int(max(base - bh - 2, 0))
            ex_, ey_ = int(min(max(x + bw, x + bw + dx) + 3, ww)), int(min(base + 2, hh))
            if ex_ - ox < 2 or ey_ - oy < 2:
                x += bw
                continue
            iv = inter[oy:ey_, ox:ex_]
            m = np.zeros(iv.shape[:2], np.float32)
            pts = [(x - ox, base - oy), (x + bw - ox, base - oy), (x + bw + dx - ox, base - bh - oy),
                   (x + dx - ox, base - bh - oy)]
            fill(m, pts)
            loc = light[oy:ey_, ox:ex_] * (0.8 + 0.2 * rng.random())
            spine = col * loc[..., None] * 0.85
            iv[:] = iv * (1 - m[..., None]) + spine * m[..., None]
            # title marks and bands on the spine
            if bw > 2.5 and tilt == 0:
                bm = np.zeros(iv.shape[:2], np.float32)
                yb = base - bh * rng.uniform(0.75, 0.88) - oy
                rect(bm, x + bw * 0.15 - ox, yb, x + bw * 0.85 - ox, yb + max(1.0, bh * 0.03), 1.0)
                for q in range(int(rng.integers(1, 4))):
                    yq = base - bh * (0.2 + 0.15 * q) - bh * 0.05 - oy
                    rect(bm, x + bw * 0.3 - ox, yq, x + bw * 0.7 - ox, yq + max(1.0, bh * 0.06), 1.0)
                bm *= m
                tc = hexc('#f4e8c8') if col.mean() < 0.5 else hexc('#1a1a1a')
                iv[:] = iv * (1 - bm[..., None]) + tc * loc[..., None] * 0.8 * bm[..., None]
            x += bw + (dx if tilt > 0 else 0) + P(rng.uniform(0.0, 0.008))
            if tilt != 0:
                x += P(0.02)
        # lying stacks (books piled flat) and hand-written POP cards standing on the shelf edge
        for _ in range(int(rng.integers(1, 3))):
            sx = rng.uniform(gx0, gx1 - P(0.3))
            yb = base
            for q in range(int(rng.integers(2, 6))):
                th = P(rng.uniform(0.025, 0.045))
                wl = P(rng.uniform(0.18, 0.26))
                ox = P(rng.uniform(-0.02, 0.02))
                rect(inter, sx + ox, yb - th, sx + ox + wl, yb, hexc(rng.choice(book_cols)) * light[
                    int(min(max(yb - 1, 0), hh - 1)), int(min(max(sx, 0), ww - 1))] * 0.8)
                rect(inter, sx + ox, yb - th, sx + ox + wl, yb - th + max(1.0, th * 0.2), hexc('#f4ecd8') * 0.7)
                yb -= th
        for _ in range(int(rng.integers(1, 4))):
            cx_ = rng.uniform(gx0, gx1 - P(0.15))
            cw_, ch_ = P(rng.uniform(0.09, 0.16)), P(rng.uniform(0.1, 0.16))
            cc = hexc(rng.choice(['#fff27a', '#ffffff', '#ffb0b8', '#b8f0ff']))
            y1_ = base + P(0.02)
            rect(inter, cx_, y1_ - ch_, cx_ + cw_, y1_, cc * 1.2)
            for q in range(3):
                yq = y1_ - ch_ * (0.8 - 0.25 * q)
                rect(inter, cx_ + cw_ * 0.15, yq, cx_ + cw_ * (0.85 - 0.2 * (q % 2)), yq + max(1.0, ch_ * 0.07),
                     hexc(rng.choice(['#e01020', '#202020', '#1a50c0'])))
        # shelf board
        rect(inter, gx0, base, gx1, base + P(0.035), hexc('#f2e2c4') * 0.9)
        rect(inter, gx0, base + P(0.035), gx1, base + P(0.045), hexc('#503828') * 0.6)
        r += 1
    # interior falloff toward the bottom / sides
    fall = np.clip(1 - ((ys - gy0) / max(gy1 - gy0, 1)) ** 2 * 0.45, 0.4, 1)
    inter *= fall[..., None]
    # a customer browsing, back-lit by the shelves (only on the wide, near shop windows)
    if False:     # (removed: read as a flat cut-out)
        m = np.zeros((hh, ww), np.float32)
        px = gx0 + (gx1 - gx0) * rng.uniform(0.35, 0.6)
        base = gy1 - P(0.02)
        top = base - P(1.62)
        fill(m, [(px - P(0.2), base), (px - P(0.24), top + P(0.62)), (px - P(0.17), top + P(0.32)),
                 (px + P(0.17), top + P(0.32)), (px + P(0.25), top + P(0.62)), (px + P(0.21), base)])
        ellipse(m, px + P(0.02), top + P(0.16), P(0.105), P(0.13))
        fill(m, [(px + P(0.18), top + P(0.4)), (px + P(0.44), top + P(0.5)), (px + P(0.46), top + P(0.56)),
                 (px + P(0.2), top + P(0.55))])       # arm reaching to the shelf
        ellipse(m, px + P(0.5), top + P(0.5), P(0.07), P(0.1))   # open book in hand
        rim = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1)
        inter = inter * (1 - m[..., None]) + hexc('#241c2a') * m[..., None] + hexc('#ffe0b0') * (rim * 0.9)[..., None]
    # glass: reflected street (vertical neon bands), diagonal sheen, condensation at the bottom
    refl = np.zeros((hh, ww, 3), np.float32)
    for k in range(int(rng.integers(3, 6))):
        cx = rng.uniform(gx0, gx1)
        c = hexc(rng.choice(['#ff4fa8', '#40e8ff', '#ffb13d', '#9ad8d0']))
        refl += c * (0.26 * np.exp(-((xs - cx) / P(rng.uniform(0.05, 0.2))) ** 2))[..., None] * \
            (0.4 + 0.6 * np.clip((ys - gy0) / max(gy1 - gy0, 1), 0, 1))[..., None]
    d = (xs - gx0) / max(gx1 - gx0, 1) + (ys - gy0) / max(gy1 - gy0, 1) * 0.35
    refl += hexc('#dfe8ff') * (0.07 * np.exp(-((d - 0.35) / 0.05) ** 2) +
                               0.04 * np.exp(-((d - 0.8) / 0.02) ** 2))[..., None]
    cond_n = noise(ww, hh, max(6, ww // 40), int(rng.integers(1e9)), 4)
    cond = np.clip((ys - (gy1 - P(0.9))) / P(0.9), 0, 1) ** 1.3 * (0.55 + 0.6 * cond_n)
    cond = np.clip(cond, 0, 1)
    inter = inter * (1 - 0.55 * cond[..., None]) + hexc('#f4e8d8') * (0.45 * cond * light)[..., None]
    # clear drip channels through the condensation
    tr = np.zeros((hh, ww), np.float32)
    for k in range(int((gx1 - gx0) / P(0.12))):
        x = rng.uniform(gx0, gx1)
        y0 = rng.uniform(gy1 - P(0.9), gy1 - P(0.3))
        cv2.line(tr, (int(x * Q), int(y0 * Q)), (int(x * Q), int(gy1 * Q)), 1.0, max(1, int(P(0.008))),
                 cv2.LINE_AA, shift=2)
    inter = inter * (1 - 0.3 * tr[..., None]) + hexc('#fff4e0') * (0.25 * tr)[..., None]
    return inter + refl


# ----------------------------------------------------------------------------- weathered wood lattice

def weather_wood(As, Es, rng, x0, y0, x1, y1, P, glow_col):
    """Grime, water-stain gradients, base AO and wet specular streaks on a vertical-slat wooden front."""
    hh, ww = As.shape[:2]
    ys = np.arange(hh, dtype=np.float32)[:, None]
    xs = np.arange(ww, dtype=np.float32)[None, :]
    reg = np.zeros((hh, ww), np.float32)
    rect(reg, x0, y0, x1, y1, 1.0)
    # per-slat tone variation
    per = max(P(0.07), 2.0)
    slat = np.floor(xs / per)
    tone = 0.85 + 0.3 * np.modf(np.sin(slat * 12.9898) * 43758.5453)[0].__abs__()
    grime = noise(ww, hh, max(6, ww // 30), int(rng.integers(1e9)), 4)
    stain = noise(max(ww // 3, 4), 6, max(6, ww // 10), int(rng.integers(1e9)), 3)
    stain = cv2.resize(stain, (ww, hh), interpolation=cv2.INTER_CUBIC)
    v = np.clip((ys - y0) / max(y1 - y0, 1), 0, 1)
    top_stain = np.clip(1 - v / 0.25, 0, 1) * stain            # water runs from the fascia
    ao = np.clip((v - 0.72) / 0.28, 0, 1) ** 1.5                 # dark splash-back at the base
    streaks = noise(max(ww // 2, 4), 5, max(8, ww // 6), int(rng.integers(1e9)), 3)
    streaks = cv2.resize(streaks, (ww, hh), interpolation=cv2.INTER_CUBIC)
    run = np.clip((streaks - 0.5) * 2.5, 0, 1) * np.clip(1 - v * 1.2, 0, 1)   # dark water runs from the top
    # clean, flat painted values: per-slat tone, a soft fascia stain and base AO only (no noisy streaks;
    # rain lives in its own layer)
    tone = 0.92 + 0.16 * (tone - 0.85) / 0.3
    mul = tone * (1 - 0.25 * np.clip(1 - v / 0.25, 0, 1)) * (1 - 0.6 * ao)
    k = reg[..., None]
    As[:] = As * (1 - k) + As * mul[..., None] * k
    Es[:] = Es * (1 - k) + Es * (mul * (1 - 0.4 * ao))[..., None] * k
    # wet specular: thin bright glints along slat edges, broken by noise, stronger low down
    edge = np.clip(1 - np.abs(np.mod(xs, per) - per * 0.1) / max(per * 0.08, 0.6), 0, 1)
    spec = edge * (0.15 + 0.35 * v) * reg
    Es[:] = Es + (np.asarray(glow_col, np.float32) * 0.5 + 0.35) * (spec * 1.1)[..., None]
    # puddle-splash darkening line + a wet sheen band near the bottom
    band = np.exp(-((v - 0.9) / 0.05) ** 2) * reg * 0.6
    Es[:] = Es + hexc('#c8d8ff') * (band * 0.12)[..., None]


# ----------------------------------------------------------------------------- sign wear / detail

def sign_detail(T, rng, fg, bg, lit):
    """Secondary copy line, corner bolts, edge wear and dirt on a sign face texture (in place)."""
    A, E = T['alb'], T['emi']
    h, w = T['a'].shape
    fw = max(1.0, 0.06 * min(w, h))
    fg = np.asarray(fg, np.float32)
    # secondary copy: a band of small glyphs at the bottom (and sometimes a thin rule above it)
    band_h = max(3, int(min(w, h) * 0.2))
    if h > w:
        by0 = int(h - fw - band_h * 1.25)
        g = _row(rng, int(w - 2 * fw - 2), band_h, max(2, int((w - 2 * fw) / band_h)), 0.12, 0.12)
    else:
        band_h = max(3, int(h * 0.16))
        by0 = int(h - fw - band_h * 1.1)
        g = _row(rng, int(w - 2 * fw - 2), band_h, max(3, int((w - 2 * fw) / band_h * 0.9)), 0.12, 0.12)
    gh, gw = g.shape
    x0 = int(fw + 1)
    if by0 > 0 and gh > 2 and gw > 2:
        sl = (slice(by0, by0 + gh), slice(x0, x0 + gw))
        sub = np.zeros((h, w), np.float32)
        sub[sl] = g
        rule = np.zeros((h, w), np.float32)
        rect(rule, fw * 1.5, by0 - max(1.0, band_h * 0.12), w - fw * 1.5, by0 - max(1.0, band_h * 0.05), 1.0)
        m = np.clip(sub * 0.85 + rule * 0.7, 0, 1)
        A[:] = A * (1 - m[..., None]) + fg * m[..., None]
        if lit:
            E[:] = E * (1 - 0.8 * m[..., None]) + fg * m[..., None] * (0.25 if bg.mean() > 0.4 else 1.6)
    # bolts on the casing corners
    br = max(0.6, fw * 0.28)
    for (bx, by) in ((fw * 0.5, fw * 0.5), (w - fw * 0.5, fw * 0.5), (fw * 0.5, h - fw * 0.5),
                     (w - fw * 0.5, h - fw * 0.5), (fw * 0.5, h / 2), (w - fw * 0.5, h / 2)):
        m = np.zeros((h, w), np.float32)
        ellipse(m, bx, by, br, br, 1.0)
        A[:] = A * (1 - m[..., None]) + hexc('#8a8e98') * m[..., None]
        E[:] = E * (1 - m[..., None]) + hexc('#ffffff') * 0.05 * m[..., None]
    # edge wear + dirt: darken near the casing and in blotches; slight panel non-uniformity
    yy = np.linspace(-1, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    n = noise(w, h, max(4, w // 12), int(rng.integers(1e9)), 4)
    edge = np.clip(np.maximum(np.abs(xx) - 0.75, np.abs(yy) - 0.85) * 5, 0, 1)
    dirt = np.clip(edge * (0.5 + n) * 0.6 + np.clip(n - 0.62, 0, 1) * 1.2, 0, 0.7)
    runs = noise(max(w // 2, 3), 4, max(3, w // 8), int(rng.integers(1e9)), 2)
    runs = cv2.resize(runs, (w, h), interpolation=cv2.INTER_CUBIC)
    dirt = dirt * 0.5
    A[:] = A * (1 - 0.4 * dirt[..., None])
    E[:] = E * (1 - 0.45 * dirt[..., None])
    # scuffed bright wear on the casing edge
    wear = np.clip((n - 0.7) * 5, 0, 1) * (edge > 0.9)
    A[:] = A * (1 - wear[..., None]) + hexc('#a0a4ac') * wear[..., None]
    return T
