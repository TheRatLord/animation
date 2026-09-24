"""Architectural detail for the village houses of s08_snow_station: the clutter that makes a Japanese
rural street read as lived-in - rain gutters with downspouts, outdoor AC units on the wall (fan grille,
louvres, snow cap), TV aerials on the ridge with snow on the crossbars, an electric meter box, a laundry
pole under the eave with towels forgotten in the snow, and on one house a lit shop sign (real Japanese:
佐藤商店) with a red たばこ tobacco sign at the corner. All painted in world space through the scene
camera, lit by the scene's lamps + night ambient and fogged like the house itself."""
import math

import numpy as np
import cv2

import s08_snow_station_lib as L
import s08_snow_station_paint as PT


def _ac_texture(tw, th):
    """Outdoor AC unit front: off-white casing, round fan grille (dark with rings + hub), louvres on the
    right, a thin shadow line under the lid. Reflectance RGB (th, tw, 3)."""
    y = np.linspace(0, 1, th, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, tw, dtype=np.float32)[None, :]
    col = np.ones((th, tw, 3), np.float32) * np.array([0.78, 0.78, 0.74], np.float32)
    col = col * (0.92 + 0.08 * (1 - y))[..., None]
    asp = tw / max(th, 1)
    cx, cy, r = 0.36, 0.52, 0.36
    d = np.sqrt(((x - cx) * asp) ** 2 + (y - cy) ** 2) / r
    grille = d < 1.0
    ring = (np.abs(np.mod(d * 4.0, 1.0) - 0.5) > 0.38) & grille
    g = np.where(grille, 0.16, 1.0) * np.where(ring, 1.9, 1.0) * np.where(d < 0.18, 2.6, 1.0)
    col = col * np.where(grille[..., None], g[..., None], 1.0)
    lou = (x > 0.74) & (x < 0.94) & (y > 0.2) & (y < 0.85) & (np.mod(y * 14, 1.0) < 0.35)
    col = np.where(lou[..., None], col * 0.45, col)
    col[: max(int(th * 0.06), 1)] *= 0.7
    return col


def _meter_texture(tw, th):
    col = np.ones((th, tw, 3), np.float32) * np.array([0.6, 0.62, 0.62], np.float32)
    y = np.linspace(0, 1, th, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, tw, dtype=np.float32)[None, :]
    win = (np.abs(x - 0.5) < 0.3) & (np.abs(y - 0.4) < 0.22)
    col = np.where(win[..., None], np.array([0.25, 0.3, 0.38], np.float32), col)
    return col


def house_details(sc, cv, X, Z, w, d, h, seed, fa, fogc, wins, gable_front=False, amb=None, Y0=-2.6,
                  win_col=None, snow_col=None):
    """Add detail to a house whose front wall is the plane z = Z, x in [X, X + w], y in [Y0, Y0 + h]."""
    cam, s = sc.cam, sc.s
    rng = np.random.default_rng(abs(int(seed)) + 4242)
    amb = np.asarray(amb, np.float32)
    snow = np.asarray(snow_col, np.float32)
    px_per_m = cam.f / Z
    if px_per_m < 8:
        return

    def E_at(x, y, z, N=(0, 0, -1)):
        return sc.light_at(np.array([[x, y, z]], np.float32), np.array(N, np.float32))[0]

    def line(P, width_m, col, z, alpha=1.0):
        x0, y0, m = L.line_local(cam.pts(np.asarray(P, np.float64)), max(cam.f * width_m / z, 0.55 * s))
        cv.paint(x0, y0, m * alpha, col, z)

    def box_front(xa, xb, ya, yb, z, tex):
        pts = cam.pts(np.array([(xa, ya, z), (xb, ya, z), (xb, yb, z), (xa, yb, z)]))
        PT.paste_quad(cv, pts, tex, z)

    def snow_cap(xa, xb, y, z, dep, hgt):
        n = 14
        xs = np.linspace(xa - 0.03, xb + 0.03, n)
        top = np.stack([xs, y + hgt * np.sin(np.linspace(0, np.pi, n)) ** 0.5 + 0.01, np.full(n, z - 0.01)], 1)
        bot = np.stack([xs, np.full(n, y - 0.015), np.full(n, z - 0.01)], 1)
        x0, y0, m = L.poly_local(cam.pts(np.concatenate([top, bot[::-1]])), ss=4)
        E = E_at((xa + xb) / 2, y + 0.1, z, (0, 1, 0))
        cv.paint(x0, y0, m, fogc(snow * 1.05 + E * 0.5), z - 0.01)

    dark = np.array([0.055, 0.055, 0.07], np.float32)
    # ---- downspouts at the front corners (dark pipe with a cool lit edge and brackets)
    for xs_ in ((X + 0.18,) if gable_front else (X + 0.18, X + w - 0.18)):
        zz = Z - 0.12
        line([(xs_, Y0 + 0.05, zz), (xs_, Y0 + h - 0.05, zz)], 0.09, fogc(dark), zz)
        line([(xs_ - 0.03, Y0 + 0.05, zz - 0.01), (xs_ - 0.03, Y0 + h - 0.05, zz - 0.01)], 0.022,
             fogc(amb * 0.55 + 0.03), zz - 0.01, 0.8)
        for yb in np.arange(Y0 + 0.9, Y0 + h - 0.3, 1.3):
            line([(xs_ - 0.07, yb, zz - 0.01), (xs_ + 0.07, yb, zz - 0.01)], 0.035, fogc(dark * 0.8), zz - 0.01)
        # elbow into the eave gutter
        line([(xs_, Y0 + h - 0.05, zz), (xs_ + (0.25 if xs_ < X + w / 2 else -0.25), Y0 + h + 0.05, zz - 0.3)],
             0.08, fogc(dark), zz - 0.02)

    # ---- outdoor AC units: on the ground against the wall and (two-storey) on a bracket upstairs
    occupied = [(wx - ww / 2 - 0.2, wx + ww / 2 + 0.2, Y0 + fl - 0.2, Y0 + fl + wh + 0.2) for (wx, fl, ww, wh) in wins]

    def free(xa, xb, ya, yb):
        for (a0, a1, b0, b1) in occupied:
            if xa < a1 and xb > a0 and ya < b1 and yb > b0:
                return False
        return xa > X + 0.35 and xb < X + w - 0.35

    spots = []
    for fl_y in ([0.05] + ([2.35] if h >= 4.2 else [])):
        for _ in range(12):
            xa = X + rng.uniform(0.4, w - 1.3)
            if free(xa, xa + 0.85, Y0 + fl_y, Y0 + fl_y + 0.65):
                spots.append((xa, fl_y))
                break
    ac_tex = _ac_texture(max(int(px_per_m * 0.85 * 2), 8), max(int(px_per_m * 0.62 * 2), 6))
    for (xa, fl_y) in spots[:2]:
        z = Z - 0.32
        ya = Y0 + fl_y
        E = E_at(xa + 0.42, ya + 0.3, z - 0.05)
        lit = amb * 0.42 + E * 0.9 + 0.012
        box_front(xa, xa + 0.85, ya, ya + 0.62, z, fogc(ac_tex * lit))
        # side face (toward the camera axis)
        sx = xa if X > 0 else xa + 0.85
        sc.quad(cv, [(sx, ya, z), (sx, ya, Z - 0.02), (sx, ya + 0.62, Z - 0.02), (sx, ya + 0.62, z)],
                fogc(np.array([0.6, 0.6, 0.57], np.float32) * (amb * 0.3 + E * 0.5)), z + 0.1)
        # contact shadow on the wall + snow on top + refrigerant pipe up the wall
        line([(xa + 0.8, ya + 0.45, Z - 0.03), (xa + 0.8, ya + 1.4 + rng.uniform(0, 0.6), Z - 0.03)], 0.05,
             fogc(np.array([0.5, 0.48, 0.44], np.float32) * (amb * 0.4 + E * 0.6)), Z - 0.03)
        snow_cap(xa, xa + 0.85, ya + 0.62, z, 0.3, 0.1)
        if fl_y > 1:
            for bx in (xa + 0.15, xa + 0.7):
                line([(bx, ya - 0.02, z + 0.02), (bx, ya - 0.02, Z - 0.02)], 0.035, fogc(dark), z + 0.05)
                line([(bx, ya - 0.02, z + 0.05), (bx, ya - 0.4, Z - 0.02)], 0.03, fogc(dark), z + 0.05)

    # ---- electric meter by the door
    if rng.random() < 0.8:
        xm = X + rng.uniform(0.3, 0.8) if rng.random() < 0.5 else X + w - rng.uniform(0.6, 1.0)
        if free(xm, xm + 0.25, Y0 + 1.3, Y0 + 1.7):
            E = E_at(xm, Y0 + 1.5, Z - 0.1)
            box_front(xm, xm + 0.25, Y0 + 1.3, Y0 + 1.68, Z - 0.08,
                      fogc(_meter_texture(max(int(px_per_m * 0.5), 4), max(int(px_per_m * 0.76), 5)) * (amb * 0.45 + E)))

    # ---- laundry pole under the upstairs eave with a couple of towels forgotten in the snow
    if h >= 4.2 and rng.random() < 0.75 and not gable_front:
        z = Z - 0.45
        yp = Y0 + h - 0.45
        xa, xb = X + w * 0.12, X + w * 0.55
        for bx in (xa, xb):
            line([(bx, yp + 0.3, Z - 0.02), (bx, yp - 0.05, z)], 0.03, fogc(dark), z + 0.1)
        line([(xa - 0.2, yp, z), (xb + 0.2, yp, z)], 0.035, fogc(np.array([0.5, 0.52, 0.56], np.float32) * amb * 0.9 + 0.02), z)
        cols = [np.array([0.85, 0.5, 0.5], np.float32), np.array([0.6, 0.75, 0.9], np.float32),
                np.array([0.9, 0.88, 0.8], np.float32)]
        for k in range(int(rng.integers(2, 4))):
            tx = xa + (k + 0.4) * (xb - xa) / 3.2
            tw_, tl = rng.uniform(0.3, 0.45), rng.uniform(0.45, 0.7)
            E = E_at(tx, yp - 0.3, z - 0.05)
            sc.quad(cv, [(tx, yp - tl, z - 0.01), (tx + tw_, yp - tl, z - 0.01), (tx + tw_, yp, z - 0.01), (tx, yp, z - 0.01)],
                    fogc(cols[k % 3] * (amb * 0.45 + E * 0.8)), z - 0.01)
            snow_cap(tx, tx + tw_, yp, z - 0.02, 0.1, 0.035)

    # ---- TV aerial on the ridge (mast, stay wire, Yagi elements with snow on top)
    if rng.random() < 0.85:
        if gable_front:
            rh = w * 0.42
            ax, ay, az = X + w / 2 + 0.3, Y0 + h + rh + 0.3, Z + d * 0.4
        else:
            ax, ay, az = X + w * rng.uniform(0.6, 0.8), Y0 + h + h * 0.45 + 0.45, Z + d / 2
        mast = rng.uniform(1.6, 2.4)
        mc = fogc(dark * 1.1)
        line([(ax, ay, az), (ax, ay + mast, az)], 0.045, mc, az)
        line([(ax, ay + mast * 0.8, az), (ax - 1.3, ay, az + 0.5)], 0.012, mc, az, 0.7)
        for k, (yy, ln) in enumerate(((mast * 0.98, 1.1), (mast * 0.88, 0.8), (mast * 0.66, 1.4))):
            line([(ax - ln / 2, ay + yy, az), (ax + ln / 2, ay + yy, az)], 0.028, mc, az)
            line([(ax - ln / 2 + 0.05, ay + yy + 0.035, az - 0.01), (ax + ln / 2 - 0.05, ay + yy + 0.035, az - 0.01)],
                 0.02, fogc(snow * 1.1), az - 0.01, 0.8)
            ne = 5 if k < 2 else 3
            for e in range(ne):
                ex = ax - ln / 2 + (e + 0.5) * ln / ne
                line([(ex, ay + yy - 0.12, az), (ex, ay + yy + 0.12, az)], 0.016, mc, az, 0.9)


def shop_signs(sc, cv, X, Z, w, h, fa, fogc, amb, Y0=-2.6):
    """A lit horizontal shop sign over the ground floor (佐藤商店) and a red vertical たばこ sign at the
    corner, plus a small warm lamp over the entrance lighting them."""
    cam, s = sc.cam, sc.s
    ppm = cam.f / Z
    # horizontal fascia sign (back-lit acrylic, cream with dark blue lettering)
    xa, xb = X + 0.8, X + 4.4
    ya, yb = Y0 + 2.05, Y0 + 2.6
    tw, th = max(int(ppm * (xb - xa) * 3), 24), max(int(ppm * (yb - ya) * 3), 8)
    tex = PT.text_board(tw, th, [('佐藤商店', 0.54, 0.62, (24, 40, 110), True)], bg=(246, 240, 222), border=(40, 60, 120))
    glow = np.array([1.35, 1.2, 0.95], np.float32)
    pts = cam.pts(np.array([(xa, ya, Z - 0.1), (xb, ya, Z - 0.1), (xb, yb, Z - 0.1), (xa, yb, Z - 0.1)]))
    PT.paste_quad(cv, pts, tex * glow * (1 - fa * 0.5) + np.asarray(sc.FOG_C, np.float32) * fa * 0.4, Z - 0.1)
    # vertical tobacco sign (red enamel, white kana), hanging off the corner
    xs_ = X + w - 0.1
    va, vb = Y0 + 1.6, Y0 + 3.2
    tw, th = max(int(ppm * 0.45 * 3), 8), max(int(ppm * (vb - va) * 3), 20)
    from PIL import Image, ImageDraw
    S = 2
    im = Image.new('RGB', (tw * S, th * S), (196, 28, 30))
    dd = ImageDraw.Draw(im)
    f = PT._font(tw * S * 0.72)
    for i, ch in enumerate('たばこ'):
        dd.text((tw * S / 2, th * S * (0.2 + i * 0.3)), ch, font=f, fill=(250, 246, 240), anchor='mm')
    vt = cv2.resize(np.asarray(im).astype(np.float32) / 255, (tw, th), interpolation=cv2.INTER_AREA)
    E = sc.light_at(np.array([[xs_, Y0 + 2.4, Z - 0.5]], np.float32), np.array([0, 0, -1], np.float32))[0]
    vlit = vt * (amb * 0.5 + E + np.array([0.35, 0.22, 0.12], np.float32))
    pts = cam.pts(np.array([(xs_, va, Z - 0.45), (xs_ + 0.45, va, Z - 0.45), (xs_ + 0.45, vb, Z - 0.45), (xs_, vb, Z - 0.45)]))
    PT.paste_quad(cv, pts, vlit * (1 - fa) + np.asarray(sc.FOG_C, np.float32) * fa, Z - 0.45)
    x0, y0, m = L.line_local(cam.pts(np.array([(xs_ - 0.2, vb + 0.02, Z - 0.3), (xs_ + 0.5, vb + 0.02, Z - 0.45)])),
                             max(cam.f * 0.03 / Z, 0.6 * s))
    cv.paint(x0, y0, m, fogc(np.array([0.05, 0.05, 0.06], np.float32)), Z - 0.46)
