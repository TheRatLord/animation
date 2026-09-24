"""s04_rain_street: ground-floor shop front painters (izakaya lattice + noren, lit shop, lobby, shutter)."""
import numpy as np
import cv2

from s04_rain_street_art import hexc, rect, tex, glyph, noise, _row


def _mask(h, w):
    return np.zeros((h, w), np.float32)


def _paste_arr(A, E, S, x, y):
    h, w = S['a'].shape
    x, y = int(round(x)), int(round(y))
    H, W = A.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    ls = (slice(y0 - y, y1 - y), slice(x0 - x, x1 - x))
    sl = (slice(y0, y1), slice(x0, x1))
    a = S['a'][ls][..., None]
    A[sl] = A[sl] * (1 - a) + S['alb'][ls] * a
    E[sl] = E[sl] * (1 - a) + S['emi'][ls] * a


def _signboard(rng, w, h, bg, fg, gain):
    sb = tex(w, h, bg)
    hh, ww = sb['a'].shape
    g = _row(rng, ww, hh, max(2, int(ww / max(hh * 1.1, 1))), 0.13, 0.12)
    sb['alb'] = sb['alb'] * (1 - g[..., None]) + fg * g[..., None]
    if np.mean(bg) > 0.4:
        sb['emi'] = bg * (gain * (1 - g))[..., None]
    else:
        sb['emi'] = bg * (0.3 * gain * (1 - g))[..., None] + fg * (gain * g)[..., None]
    return sb


def shopfront(T, rng, kind, x0, y0, x1, y1, ppm):
    """Paint a ground-floor front into texture T within [x0,x1]x[y0,y1] (px)."""
    A, E = T['alb'], T['emi']
    h, w = T['a'].shape
    P = lambda m: m * ppm
    x0, y0, x1, y1 = int(x0), int(max(y0, 0)), int(min(x1, w)), int(min(y1, h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    sub = (slice(y0, y1), slice(x0, x1))
    hh, ww = y1 - y0, x1 - x0
    As, Es = A[sub], E[sub]
    yy = np.linspace(0, 1, hh, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, ww, dtype=np.float32)[None, :]
    fas = P(0.6)       # fascia height
    if kind == 'bar':
        As[:] = hexc('#3a2418')
        rect(As, 0, 0, ww, fas, hexc('#1a120e'))
        bx0, bx1 = P(0.4), ww - P(0.4)
        if bx1 - bx0 > 8 and fas > 3:
            sb = _signboard(rng, bx1 - bx0, fas * 0.7, hexc('#f4e6c8'), hexc('#3a0606'), 0.32)
            _paste_arr(As, Es, sb, bx0, fas * 0.15)
        ly0 = fas + P(0.05)
        glow = hexc('#ffa050') * (0.5 + 0.5 * yy)[..., None]
        slat_p = P(0.07)
        xs = np.arange(ww, dtype=np.float32)[None, :]
        if slat_p > 3:
            gaps = ((xs % slat_p) > slat_p * 0.55).astype(np.float32)
        else:
            gaps = np.full((1, ww), 0.42, np.float32)
        m = _mask(hh, ww)
        rect(m, 0, ly0, ww, hh - P(0.25), 1.0)
        dx0 = ww * rng.uniform(0.45, 0.62)
        dx1 = dx0 + P(1.3)
        door = _mask(hh, ww)
        rect(door, dx0, ly0, dx1, hh, 1.0)
        latt = m * (1 - door) * gaps
        Es[:] = Es + glow * (latt * 0.8)[..., None]
        As[:] = As * (1 - (m * (1 - door) * (1 - gaps))[..., None] * 0.2)
        Es[:] = Es + hexc('#ffb868') * (door * 1.1 * (0.6 + 0.4 * yy))[..., None]
        As[:] = As * (1 - door[..., None]) + hexc('#402818') * door[..., None]
        ncol = hexc(rng.choice(['#1c2a5a', '#5a1a1a', '#202020']))
        nor = _mask(hh, ww)
        rect(nor, dx0 - P(0.1), ly0, dx1 + P(0.1), ly0 + P(0.85), 1.0)
        for k in range(1, 4):
            xk = dx0 - P(0.1) + (dx1 - dx0 + P(0.2)) * k / 4
            rect(nor, xk - P(0.015), ly0 + P(0.15), xk + P(0.015), ly0 + P(0.85), 0.0)
        As[:] = As * (1 - nor[..., None]) + ncol * nor[..., None]
        Es[:] = Es * (1 - nor[..., None]) + ncol * 0.45 * nor[..., None]
        gs = int(min(P(0.45), dx1 - dx0))
        if gs > 6:
            g = cv2.resize(glyph(rng, 96, 0.12), (gs, gs), interpolation=cv2.INTER_AREA)
            gm = _mask(hh, ww)
            gx, gy = int((dx0 + dx1) / 2 - gs / 2), int(ly0 + P(0.2))
            hh2, ww2 = max(0, min(gs, hh - gy)), max(0, min(gs, ww - gx))
            gm[gy:gy + hh2, gx:gx + ww2] = g[:hh2, :ww2]
            As[:] = As * (1 - gm[..., None]) + hexc('#f0ece0') * gm[..., None]
            Es[:] = Es + hexc('#fff0e0') * (gm * 0.6)[..., None]
        if P(0.07) > 2.5:
            from s04_rain_street_art2 import weather_wood
            weather_wood(As, Es, rng, 0, ly0, ww, hh - P(0.25), P, hexc('#ffa050'))
        rect(As, 0, hh - P(0.25), ww, hh, hexc('#2a2420'))
        mx0 = dx0 - P(0.9)
        if mx0 > 0:
            rect(As, mx0, hh - P(1.5), mx0 + P(0.6), hh - P(0.6), hexc('#1a1a1a'))
            rect(Es, mx0 + P(0.05), hh - P(1.45), mx0 + P(0.55), hh - P(0.65), hexc('#fff2d0') * 0.8)
    elif kind == 'drug':
        from s04_rain_street_v5 import drugstore
        drugstore(As, Es, rng, ww, hh, fas, P)
    elif kind == 'shop':
        As[:] = hexc('#2a2c34')
        band = hexc(rng.choice(['#1d3a6a', '#6a1d2a', '#1d5a3a', '#e8e0d0', '#303030']))
        bx0, bx1 = P(0.3), ww - P(0.3)
        rect(As, 0, 0, ww, fas, band * 0.8)
        if bx1 - bx0 > 8 and fas > 3:
            fg = hexc('#fff6e0') if band.mean() < 0.5 else hexc('#202020')
            sb = _signboard(rng, bx1 - bx0, fas * 0.8, band, fg, 1.0)
            _paste_arr(As, Es, sb, bx0, fas * 0.1)
        gy0 = fas + P(0.05)
        glass = _mask(hh, ww)
        rect(glass, P(0.2), gy0, ww - P(0.2), hh - P(0.35), 1.0)
        warm = hexc(rng.choice(['#ffd8a0', '#ffe8c8', '#ffc890']))
        inter = np.broadcast_to(warm * (0.35 + 0.3 * yy)[..., None], (hh, ww, 3)).copy()
        if P(0.03) >= 1.5:
            from s04_rain_street_art2 import bookshelves
            inter = bookshelves(As, Es, rng, P(0.2), gy0, ww - P(0.2), hh - P(0.35), P)
        else:
            # distant: vectorised spines (column colours) on shelves under a warm wash
            cols = np.array([hexc(c) for c in ['#c83a3a', '#3a6ac8', '#e8c040', '#3a9a5a', '#e87a30', '#f0ece0',
                                               '#2a2a30', '#d86a8a']], np.float32)
            ci = rng.integers(0, len(cols), ww)
            spines = cols[ci][None, :, :] * rng.uniform(0.5, 0.9, ww).astype(np.float32)[None, :, None]
            shelf = ((np.arange(hh)[:, None] - gy0) % max(P(0.4), 2)) > max(P(0.4), 2) * 0.25
            inter = inter * (1 - 0.6 * shelf[..., None]) + spines * 0.8 * shelf[..., None]
        nm = max(1, int(ww / P(1.5)))
        mul = _mask(hh, ww)
        for i in range(nm + 1):
            xm = P(0.2) + (ww - P(0.4)) * i / nm
            rect(mul, xm - P(0.035), gy0, xm + P(0.035), hh - P(0.35), 1.0)
        inter = inter * (1 - mul[..., None])
        Es[:] = Es + inter * glass[..., None]
        As[:] = As * (1 - glass[..., None]) + inter * 0.1 * glass[..., None]
        rect(As, 0, hh - P(0.35), ww, hh, hexc('#1e2026'))
    elif kind == 'entrance':
        As[:] = hexc('#4a4a52')
        tl = max(P(0.3), 3)
        xs = (np.arange(ww)[None, :] % tl) < 1
        ys = (np.arange(hh)[:, None] % tl) < 1
        As[:] = As * (1 - 0.25 * (xs | ys)[..., None])
        dx0 = ww * rng.uniform(0.2, 0.5)
        dx1 = dx0 + P(1.8)
        d = _mask(hh, ww)
        rect(d, dx0, P(0.5), dx1, hh, 1.0)
        lob = np.broadcast_to(hexc('#e8f0ff') * (0.7 + 0.4 * yy)[..., None], (hh, ww, 3)).copy()
        for i in range(4):
            for j in range(3):
                rect(lob, dx0 + P(0.2 + 0.3 * i), P(0.9 + 0.25 * j), dx0 + P(0.45 + 0.3 * i), P(1.1 + 0.25 * j),
                     hexc('#6a6e78'))
        rect(lob, (dx0 + dx1) / 2 - P(0.03), P(0.5), (dx0 + dx1) / 2 + P(0.03), hh, hexc('#40444c'))
        Es[:] = Es + lob * d[..., None]
        As[:] = As * (1 - d[..., None]) + 0.2 * d[..., None]
        rect(As, (dx0 + dx1) / 2 - P(0.2), P(0.25), (dx0 + dx1) / 2 + P(0.2), P(0.38), hexc('#e8e8e0'))
        rect(Es, (dx0 + dx1) / 2 - P(0.2), P(0.25), (dx0 + dx1) / 2 + P(0.2), P(0.38), hexc('#fff4e0') * 3.0)
    elif kind in ('shutter', 'shutter_plain'):
        As[:] = hexc('#34363e')
        sx0, sx1 = P(0.35), ww - P(0.35)
        sh = _mask(hh, ww)
        rect(sh, sx0, fas, sx1, hh, 1.0)
        yl = np.arange(hh, dtype=np.float32)[:, None]
        per = P(0.07)
        rib = (0.78 + 0.22 * (np.sin(yl / per * 2 * np.pi) > 0)) if per > 2.5 else np.full_like(yl, 0.89)
        col = hexc('#5a5e68') * rib[..., None] * (0.9 + 0.2 * xx[..., None])
        As[:] = As * (1 - sh[..., None]) + col * sh[..., None]
        rect(As, sx0 - P(0.1), fas - P(0.3), sx1 + P(0.1), fas, hexc('#4a4e58'))
        if rng.random() < 0.7 and fas > 3 and kind == 'shutter':
            c = hexc(rng.choice(['#ffe0a0', '#ffffff', '#ffd0d8']))
            bw = min(P(2.2), (sx1 - sx0) * 0.6)
            sb = _signboard(rng, bw, fas * 0.55, c, hexc('#202028'), 1.1)
            _paste_arr(As, Es, sb, (sx0 + sx1) / 2 - bw / 2, fas * 0.1)
        st = noise(max(ww // 4, 4), 4, max(3, ww // 16), int(rng.integers(1e6)), 2)
        st = cv2.resize(st, (ww, hh), interpolation=cv2.INTER_CUBIC)
        As[:] = As * (0.85 + 0.25 * st)[..., None]
        if P(0.05) > 1.2 and rng.random() < 0.75:
            from s04_rain_street_v5 import graffiti
            graffiti(As, Es, rng, sx0, fas + P(0.3), sx1, hh - P(0.15), P)
    elif kind == 'wall':
        As[:] = hexc('#3e3a40')
        tl = max(P(0.25), 3)
        xs = (np.arange(ww)[None, :] % tl) < 1
        ys = (np.arange(hh)[:, None] % (tl * 0.5)) < 1
        As[:] = As * (1 - 0.2 * (xs | ys)[..., None])
        for k in range(int(rng.integers(2, 5))):
            px = ww * rng.uniform(0.05, 0.85)
            py = hh * rng.uniform(0.3, 0.55)
            c = hexc(rng.choice(['#e8d8b0', '#c83a3a', '#3a6ac8', '#f0f0f0', '#e8a030']))
            rect(As, px, py, px + P(0.45), py + P(0.65), c * 0.8)
        dx = ww * rng.uniform(0.6, 0.8)
        rect(As, dx, hh - P(2.1), dx + P(0.9), hh, hexc('#4a4038'))
        rect(Es, dx + P(0.35), hh - P(2.35), dx + P(0.55), hh - P(2.2), hexc('#ffe8c0') * 2.0)
    elif kind == 'conbini':
        # handled separately in the scene (bright store front); paint a lit glass base here
        As[:] = hexc('#d8dde4')
        g = _mask(hh, ww)
        rect(g, P(0.2), fas, ww - P(0.2), hh - P(0.1), 1.0)
        Es[:] = Es + hexc('#f2f8ff') * (1.8 * g)[..., None]
        rect(Es, 0, 0, ww, fas * 0.4, hexc('#ffffff') * 1.6)
        for i, c in enumerate(['#1aa24a', '#f08a1c', '#1a64d0']):
            rect(Es, 0, fas * (0.4 + 0.2 * i), ww, fas * (0.6 + 0.2 * i), hexc(c) * 1.5)
