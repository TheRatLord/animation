"""s04_rain_street extra architectural detail for mid/far buildings: downpipes with brackets, roof-edge
gutters/parapet caps, TV (Yagi) antennas and satellite dishes on the roofs, laundry poles with hanging
washing, electric meter boxes. All deterministic per building seed; painted into the building canvases."""
import numpy as np
import cv2

from s04_rain_street_art import hexc, tex, rect


def _line(cv, pts, diam, col, emi=None):
    cv.polyline3d(np.asarray(pts, np.float64), diam, hexc(col) if isinstance(col, str) else col, emi=emi)


def downpipes(cv, b, rng):
    s, z0, z1, xf, hh = b['side'], b['z0'], b['z1'], b['x'], b['h']
    for zc in ([z0 + 0.22] + ([z1 - 0.25] if rng.random() < 0.5 else [])):
        if zc < 1.2:
            continue
        x = xf - s * 0.13
        _line(cv, [(xf, hh - 0.15, zc), (x, hh - 0.35, zc), (x, 0.15, zc)], 0.085, '#6c707a')
        # a thin highlight line on the camera side
        _line(cv, [(x - s * 0.02, hh - 0.4, zc - 0.035), (x - s * 0.02, 0.2, zc - 0.035)], 0.018, '#b8bcc8')
        for y in np.arange(1.5, hh - 0.5, 1.8):
            _line(cv, [(xf, y, zc), (x, y, zc)], 0.05, '#3a3c44')


def roof_edge(cv, b, rng):
    s, z0, z1, xf, hh = b['side'], b['z0'], b['z1'], b['x'], b['h']
    za = max(z0, 1.0)
    if z1 - za < 0.5:
        return
    x = xf - s * 0.06
    # parapet cap: light concrete edge catching the sky glow, plus a darker gutter just below
    _line(cv, [(x, hh + 0.05, za), (x, hh + 0.05, z1)], 0.14, '#9a9aa8')
    _line(cv, [(x - s * 0.08, hh - 0.25, za), (x - s * 0.08, hh - 0.25, z1)], 0.1, '#3c3e48')
    if rng.random() < 0.5:   # guard railing on the roof
        xr = xf + s * 0.3
        _line(cv, [(xr, hh + 1.1, za), (xr, hh + 1.1, z1)], 0.05, '#5a5e6a')
        for zz in np.arange(za + 0.2, z1, 1.2):
            _line(cv, [(xr, hh, zz), (xr, hh + 1.1, zz)], 0.04, '#4a4e5a')


def antennas(cv, b, rng):
    s, z0, z1, xf, hh = b['side'], b['z0'], b['z1'], b['x'], b['h']
    n = 1 + int(rng.random() < 0.45)
    for k in range(n):
        zm = rng.uniform(z0 + 1.0, max(z1 - 1.0, z0 + 1.1))
        xm = xf + s * rng.uniform(0.8, 3.5)
        mh = rng.uniform(1.8, 3.2)
        col = '#2a2c36'
        _line(cv, [(xm, hh, zm), (xm, hh + mh, zm)], 0.05, col)
        # guy wires
        for (dx, dz) in ((0.9, 0.6), (-0.9, 0.6), (0.0, -1.0)):
            _line(cv, [(xm, hh + mh * 0.75, zm), (xm + dx, hh, zm + dz)], 0.012, '#3a3c48')
        # Yagi booms (one or two) with elements
        for j in range(1 + int(rng.random() < 0.6)):
            yb = hh + mh - 0.15 - j * 0.6
            L = rng.uniform(1.0, 1.8)
            ang = rng.uniform(-0.6, 0.6)
            dx, dz = np.sin(ang) * L, np.cos(ang) * L
            bz0 = zm - dz * 0.35
            bx0 = xm - dx * 0.35
            _line(cv, [(bx0, yb, bz0), (bx0 + dx, yb, bz0 + dz)], 0.03, col)
            ne = int(rng.integers(5, 9))
            for e in range(ne):
                u = e / (ne - 1)
                ex, ez = bx0 + dx * u, bz0 + dz * u
                half = 0.28 * (1.0 - 0.45 * u)
                px, pz = np.cos(ang) * half, -np.sin(ang) * half
                _line(cv, [(ex - px, yb, ez - pz), (ex + px, yb, ez + pz)], 0.018, col)
    if rng.random() < 0.35:
        # satellite dish: a small elliptical sprite on a stub
        zc = rng.uniform(z0 + 0.8, max(z1 - 0.8, z0 + 0.9))
        xd = xf + s * rng.uniform(0.6, 2.0)
        T = tex(64, 64, hexc('#b8bcc6'))
        yy, xx = np.mgrid[0:64, 0:64].astype(np.float32)
        d = ((xx - 32) / 30) ** 2 + ((yy - 30) / 26) ** 2
        T['a'] = np.clip((1 - d) * 12, 0, 1).astype(np.float32)
        T['alb'] *= (0.75 + 0.35 * (1 - yy / 64))[..., None]
        cv.sprite(xd, hh + 0.5, zc, 0.65, 0.6, T['alb'], T['a'], None)
        _line(cv, [(xd, hh, zc), (xd, hh + 0.55, zc)], 0.05, '#3a3c48')


_CLOTH = ['#e8e4d8', '#7aa0d8', '#e89aa8', '#f4f4f0', '#a8c890', '#303848', '#d8c070', '#b0b8c8', '#c86a58']


def laundry(cv, b, rng, ppm):
    """Laundry poles (monohoshi) under some windows: a pole parallel to the wall with hanging washing."""
    s, z0, z1, xf, hh = b['side'], b['z0'], b['z1'], b['x'], b['h']
    fh = 3.0
    floors = int((hh - 3.4) // fh)
    for f in range(floors):
        if rng.random() > 0.45:
            continue
        L = rng.uniform(1.4, 2.4)
        zc = rng.uniform(z0 + L / 2 + 0.3, max(z1 - L / 2 - 0.3, z0 + L / 2 + 0.31))
        if zc - L / 2 < 1.5:
            continue
        yb = 3.4 + f * fh + rng.uniform(1.7, 2.1)
        d = 0.45
        xo = xf - s * d
        # brackets + pole
        for zz in (zc - L / 2, zc + L / 2):
            _line(cv, [(xf, yb, zz), (xo, yb, zz)], 0.035, '#8a8e98')
        _line(cv, [(xo, yb, zc - L / 2 - 0.1), (xo, yb, zc + L / 2 + 0.1)], 0.035, '#c0c4cc')
        # washing (texture on a quad hanging from the pole, parallel to the wall)
        hpx = max(8, int(1.0 * ppm))
        wpx = max(8, int(L * ppm))
        T = tex(wpx, hpx, (0, 0, 0), a=0.0)
        x = 0.03 * wpx
        while x < wpx * 0.95:
            cw = wpx * rng.uniform(0.07, 0.16)
            ch = hpx * rng.uniform(0.35, 0.85)
            col = hexc(rng.choice(_CLOTH))
            m = np.zeros((hpx, wpx), np.float32)
            kind = rng.random()
            if kind < 0.4:     # shirt: body + sleeves
                rect(m, x, 0, x + cw, ch, 1.0)
                rect(m, x - cw * 0.25, 0, x + cw * 1.25, ch * 0.3, 1.0)
            elif kind < 0.7:   # towel
                rect(m, x, 0, x + cw * 0.8, ch * 0.8, 1.0)
            else:              # trousers
                rect(m, x, 0, x + cw * 0.42, ch, 1.0)
                rect(m, x + cw * 0.55, 0, x + cw, ch, 1.0)
                rect(m, x, 0, x + cw, ch * 0.25, 1.0)
            shade = np.linspace(1.0, 0.72, hpx, dtype=np.float32)[:, None, None]
            T['alb'] = T['alb'] * (1 - m[..., None]) + col * shade * m[..., None]
            T['a'] = np.maximum(T['a'], m)
            x += cw * rng.uniform(1.15, 1.5)
        zz0, zz1 = zc - L / 2, zc + L / 2
        yt, yb2 = yb, yb - 1.0
        if s < 0:
            P = [(xo, yt, zz0), (xo, yt, zz1), (xo, yb2, zz1), (xo, yb2, zz0)]
        else:
            P = [(xo, yt, zz1), (xo, yt, zz0), (xo, yb2, zz0), (xo, yb2, zz1)]
        cv.quad(P, T['alb'], T['a'], None, normal=(-s, 0, 0))


def meters(cv, b, rng, ppm):
    """Electric/gas meter boxes and a conduit at the ground floor edge."""
    s, z0, z1, xf = b['side'], b['z0'], b['z1'], b['x']
    if b.get('ground') in ('shop', 'conbini', 'bar'):
        return
    zc = z0 + rng.uniform(0.3, 0.5)
    if zc < 2.0:
        return
    n = int(rng.integers(1, 4))
    for k in range(n):
        y = 1.3 + k * 0.45
        wpx, hpx = max(6, int(0.3 * ppm)), max(6, int(0.38 * ppm))
        T = tex(wpx, hpx, hexc('#b8bcc2'))
        rect(T['alb'], wpx * 0.2, hpx * 0.15, wpx * 0.8, hpx * 0.5, hexc('#e8ecf0'))
        rect(T['alb'], wpx * 0.25, hpx * 0.6, wpx * 0.75, hpx * 0.68, hexc('#2a2c34'))
        z = zc + (k % 2) * 0.33
        xo = xf - s * 0.12
        if s < 0:
            P = [(xo, y + 0.38, z - 0.15), (xo, y + 0.38, z + 0.15), (xo, y, z + 0.15), (xo, y, z - 0.15)]
        else:
            P = [(xo, y + 0.38, z + 0.15), (xo, y + 0.38, z - 0.15), (xo, y, z - 0.15), (xo, y, z + 0.15)]
        cv.quad(P, T['alb'], T['a'], None, normal=(-s, 0, 0))
    _line(cv, [(xf - s * 0.06, 1.2, zc - 0.25), (xf - s * 0.06, 3.1, zc - 0.25)], 0.04, '#5a5e68')


def add(cv, b, cam):
    rng = np.random.default_rng(b['seed'] + 7777)
    z0 = b['z0']
    ppm = min(cam.px_per_m(max(z0, 2.0)), 300 * cam.ss * cam.W / 1920)
    if z0 < 120:
        downpipes(cv, b, rng)
        roof_edge(cv, b, rng)
    if z0 < 70:
        meters(cv, b, rng, ppm)
        if b['style'] in ('apt', 'bar') and z0 > 12.0:
            laundry(cv, b, rng, ppm)
    if 12.0 < z0 < 170 and rng.random() < 0.8:
        antennas(cv, b, rng)
