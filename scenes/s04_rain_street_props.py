"""s04_rain_street alley clutter (wwy_12): recycling bin and beer crates past the vending machines, traffic
cones at the kerbs and a mamachari parked against the right-hand wall. Every prop is painted into its own
canvas and baked as a single-surface card (clean parallax during the push-in, no depth-warp folding)."""
import numpy as np
import cv2

import s04_rain_street_art as A
import s04_rain_street_text as JT

hexc = A.hexc


def _ppm(cv, z):
    cam = cv.cam
    return min(cam.px_per_m(z), 700 * cam.ss * cam.W / 1920)


def _box(cv, x0, x1, y0, y1, z0, z1, Tf, Ts, Tt=None):
    """Box against the left wall: street-facing side (x = x1), top, camera-facing front (z = z0)."""
    cv.quad([(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)], Ts['alb'], Ts['a'], Ts.get('emi'),
            normal=(1, 0, 0))
    if Tt is not None:
        cv.quad([(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)], Tt['alb'], Tt['a'], Tt.get('emi'),
                normal=(0, 1, 0))
    cv.quad([(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)], Tf['alb'], Tf['a'], Tf.get('emi'),
            normal=(0, 0, -1))


def _shade_tex(rng, w, h, col, dirt=0.25):
    T = A.tex(w, h, col)
    hh, ww = T['a'].shape
    n = cv2.GaussianBlur(A.noise(ww, hh, max(3, ww // 10), int(rng.integers(1e9)), 3), (0, 0), max(1.0, ww / 60))
    yy = np.linspace(0, 1, hh, dtype=np.float32)[:, None]
    T['alb'] *= ((1 - dirt + dirt * 2 * n) * (1.05 - 0.25 * yy))[..., None]
    return T


def prop_bin(sc, cv, z0=9.35):
    rng = np.random.default_rng(909)
    col = hexc('#3a5474')
    p = _ppm(cv, z0)
    x0, x1, h = -3.4, -3.02, 0.82
    Tf = _shade_tex(rng, (x1 - x0) * p, h * p, col, dirt=0.12)
    hh, ww = Tf['a'].shape
    A.rect(Tf['alb'], 0, 0, ww, hh * 0.08, col * 0.6)
    lab = np.zeros((hh, ww), np.float32)
    A.rect(lab, ww * 0.18, hh * 0.3, ww * 0.82, hh * 0.52, 1.0)
    Tf['alb'] = Tf['alb'] * (1 - lab[..., None]) + hexc('#e4e8e8') * lab[..., None]
    g = JT.text_line('かん', max(3, int(hh * 0.16)))
    gh, gw = g.shape
    ys_, xs_ = int(hh * 0.33), int((ww - gw) / 2)
    if gw < ww and gh < hh:
        sub = Tf['alb'][ys_:ys_ + gh, xs_:xs_ + gw]
        sub[:] = sub * (1 - g[..., None]) + col * 0.8 * g[..., None]
    Ts = _shade_tex(rng, 0.5 * p, h * p, col * 0.85, dirt=0.12)
    A.rect(Ts['alb'], 0, 0, Ts['a'].shape[1], Ts['a'].shape[0] * 0.08, col * 0.5)
    Tt = A.tex((x1 - x0) * p, 0.5 * p, col * 0.5)
    th_, tw_ = Tt['a'].shape
    cv2.ellipse(Tt['alb'], (tw_ // 2, th_ // 2), (max(1, int(tw_ * 0.18)), max(1, int(th_ * 0.18))), 0, 0, 360,
                (0.02, 0.02, 0.03), -1, cv2.LINE_AA)
    _box(cv, x0, x1, 0.0, h, z0, z0 + 0.5, Tf, Ts, Tt)


def prop_crates(sc, cv, zc=10.0):
    rng = np.random.default_rng(910)
    for k, (col, dx) in enumerate(((hexc('#a8882a'), 0.0), (hexc('#8a3a30'), 0.03))):
        p = _ppm(cv, zc)
        x0, x1 = -3.4 + dx, -2.96 + dx
        y0, y1 = k * 0.3, k * 0.3 + 0.29
        Tf = _shade_tex(rng, (x1 - x0) * p, 0.29 * p, col, dirt=0.3)
        hh, ww = Tf['a'].shape
        for q in range(4):
            A.rect(Tf['alb'], ww * (0.1 + 0.22 * q), hh * 0.25, ww * (0.1 + 0.22 * q) + ww * 0.12, hh * 0.75,
                   col * 0.25)
        A.rect(Tf['alb'], 0, 0, ww, hh * 0.1, col * 1.15)
        Ts = _shade_tex(rng, 0.36 * p, 0.29 * p, col * 0.8, dirt=0.3)
        hs, ws = Ts['a'].shape
        for q in range(3):
            A.rect(Ts['alb'], ws * (0.1 + 0.3 * q), hs * 0.25, ws * (0.1 + 0.3 * q) + ws * 0.18, hs * 0.75,
                   col * 0.22)
        Tt = A.tex((x1 - x0) * p, 0.36 * p, col * 0.35) if k == 1 else None
        _box(cv, x0, x1, y0, y1, zc, zc + 0.36, Tf, Ts, Tt)


def prop_cone(sc, cv, X, Z):
    p = _ppm(cv, Z)
    w, h = 0.36, 0.72
    T = A.tex(w * p, h * p, hexc('#e0401e'))
    hh, ww = T['a'].shape
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    u = yy / hh
    half = (0.1 + 0.32 * np.clip((u - 0.08) / 0.84, 0, 1)) * ww
    body = np.clip(half - np.abs(xx - ww / 2) + 0.5, 0, 1) * (u < 0.92)
    base = ((u >= 0.9) & (np.abs(xx - ww / 2) < ww * 0.5)).astype(np.float32)
    T['a'] = np.clip(body + base, 0, 1)
    shade = 0.55 + 0.6 * np.clip(1 - ((xx - ww * 0.42) / np.maximum(half, 1)) ** 2, 0, 1)
    T['alb'] *= shade[..., None]
    band = ((u > 0.3) & (u < 0.42)) | ((u > 0.56) & (u < 0.66))
    T['alb'][band] = hexc('#e8ecec') * 0.9
    T['emi'][band] = hexc('#dfe8ff') * 0.25
    T['alb'][u >= 0.9] = hexc('#301410')
    cv.sprite(X, 0.0, Z, w, h, T['alb'], T['a'], T['emi'])


def prop_bike(sc, cv):
    Xb, zc_, r = 3.42, 14.6, 0.33
    fr, wh = hexc('#8a9098'), hexc('#1a1c20')
    th = np.linspace(0, 2 * np.pi, 48)
    for zw in (zc_ - 0.52, zc_ + 0.52):
        ring = np.stack([np.full_like(th, Xb), r + r * np.sin(th), zw + r * np.cos(th)], 1)
        cv.polyline3d(ring, 0.04, wh, min_px=0.9)
        ring2 = np.stack([np.full_like(th, Xb - 0.005), r + r * 0.92 * np.sin(th), zw + r * 0.92 * np.cos(th)], 1)
        cv.polyline3d(ring2, 0.012, hexc('#b8c0c8'), emi=hexc('#9ab8c8') * 0.25, min_px=0.6)
    hub_r, hub_f = (Xb, r, zc_ + 0.52), (Xb, r, zc_ - 0.52)
    crank = (Xb, 0.3, zc_ + 0.05)
    seat = (Xb, 0.95, zc_ + 0.3)
    head = (Xb, 0.92, zc_ - 0.38)
    for a_, b_ in ((hub_r, crank), (crank, seat), (seat, hub_r), (crank, head), (head, hub_f), (seat, head)):
        cv.polyline3d(np.array([a_, b_]), 0.035, fr, emi=hexc('#b0c8d0') * 0.08, min_px=0.9)
    cv.polyline3d(np.array([(Xb, 0.98, zc_ + 0.22), (Xb, 1.0, zc_ + 0.4)]), 0.07, hexc('#15161a'))
    cv.polyline3d(np.array([(Xb + 0.25, 1.06, zc_ - 0.42), (Xb, 1.03, zc_ - 0.4), (Xb - 0.25, 1.06, zc_ - 0.42)]),
                  0.03, fr, min_px=0.9)
    p = _ppm(cv, zc_ - 0.8)
    Tb = A.tex(0.36 * p, 0.26 * p, hexc('#7a8088'))
    hb, wb = Tb['a'].shape
    mesh = np.zeros((hb, wb), np.float32)
    for q in range(1, 8):
        A.rect(mesh, wb * q / 8, 0, wb * q / 8 + max(1.0, wb * 0.015), hb, 1.0)
    for q in range(1, 5):
        A.rect(mesh, 0, hb * q / 5, wb, hb * q / 5 + max(1.0, hb * 0.02), 1.0)
    A.rect(mesh, 0, 0, wb, max(1.0, hb * 0.06), 1.0)
    Tb['a'] = np.clip(mesh + 0.15, 0, 1)
    cv.quad([(Xb - 0.18, 1.0, zc_ - 0.66), (Xb + 0.18, 1.0, zc_ - 0.66), (Xb + 0.18, 0.74, zc_ - 0.66),
             (Xb - 0.18, 0.74, zc_ - 0.66)], Tb['alb'], Tb['a'], None, normal=(0, 0, -1))
    cv.polyline3d(np.array([(Xb, 0.78, zc_ + 0.35), (Xb, 0.8, zc_ + 0.85)]), 0.05, hexc('#50565e'), min_px=0.9)


# (painter, zkey)
PROPS = [(lambda sc, cv: prop_bin(sc, cv), 9.35),
         (lambda sc, cv: prop_crates(sc, cv), 10.0),
         (lambda sc, cv: prop_cone(sc, cv, 3.05, 11.6), 11.6),
         (lambda sc, cv: prop_cone(sc, cv, 3.2, 12.35), 12.35),
         (lambda sc, cv: prop_cone(sc, cv, -2.85, 19.3), 19.3),
         (lambda sc, cv: prop_bike(sc, cv), 14.0)]
