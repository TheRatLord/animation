"""s01 helper: extra Shinkai-style rooftop clutter painted onto the town plates (s01_summer_sky_city):
solar panel arrays with sky reflections, rooftop laundry decks (monohoshi-dai) with hanging washing,
BS satellite dishes, vent stacks, rooftop AC condensers / cable trays on flat roofs, balcony laundry
and wall AC units on the apartment blocks. All shapes are hard-edged supersampled polygons in the
same Canvas; everything is deterministic (own rng, so the base town layout is unchanged)."""
import math
import numpy as np

import s01_summer_sky_city as T

_h, _c, hz, RIM = T._h, T._c, T.hz, T.RIM
GL = T.GLINTS


def _ss(e0, e1, x):
    return T._ss(e0, e1, x)


def solar_array(cv, xc, y0, y1, half_w, s, hk, rng, nr=2, nc=4, slope=0.0):
    """Array of PV panels lying on a roof face that faces us: navy cells reflecting the sky (lighter
    toward the ridge), silver frames, a diagonal sheen and a glint."""
    gap = 0.012 * half_w
    ph = (y1 - y0 - gap * (nr - 1)) / nr
    pw = (2 * half_w - gap * (nc - 1)) / nc
    base = hz(_h('#1b2b5c'), hk)
    refl = hz(_h('#8fb4e6'), hk)
    for r in range(nr):
        for c in range(nc):
            x0 = xc - half_w + c * (pw + gap)
            yy0 = y0 + r * (ph + gap)

            def col(gx, gy, x0=x0, yy0=yy0):
                v = np.clip((gy - y0) / max(y1 - y0, 1), 0, 1)
                u = np.clip((gx - (xc - half_w)) / (2 * half_w), 0, 1)
                k = 0.55 * (1 - v) ** 1.5 + 0.35 * np.exp(-(((u - v * 0.5) - 0.35) / 0.08) ** 2)
                return base + (refl - base) * np.clip(k, 0, 1)[..., None]
            cv.poly([(x0, yy0), (x0 + pw, yy0), (x0 + pw, yy0 + ph), (x0, yy0 + ph)], col)
            # cell grid
            ln = []
            for i in range(1, 6):
                xx = x0 + pw * i / 6
                ln.append([(xx, yy0), (xx, yy0 + ph)])
            for j in range(1, 3):
                y_ = yy0 + ph * j / 3
                ln.append([(x0, y_), (x0 + pw, y_)])
            cv.lines(ln, max(0.5, 0.35 * s), hz(_h('#3a5288'), hk), 0.8)
            cv.lines([[(x0, yy0), (x0 + pw, yy0), (x0 + pw, yy0 + ph), (x0, yy0 + ph), (x0, yy0)]],
                     max(0.6, 0.6 * s), hz(_h('#c8d0de'), hk), 0.9)
            cv.lines([[(x0, yy0 - 0.3 * s), (x0 + pw, yy0 - 0.3 * s)]], max(0.5, 0.4 * s), hz(RIM, hk * 0.5), 0.8)
    GL.append((xc - half_w * 0.3, y0 + (y1 - y0) * 0.2, 0.9, (xc * 0.13) % 6.28))


def laundry_deck(cv, xc, y_deck, w, s, hk, rng):
    """Rooftop laundry deck (monohoshi-dai): a slatted platform on short legs straddling the roof, a
    railing, two T-posts carrying a pole with washing (towels, shirts, a futon) hanging in the sun."""
    x0, x1 = xc - w / 2, xc + w / 2
    lw = max(0.9, 1.3 * s)
    dark = hz(_h('#3a4058'), hk)
    metal = hz(_h('#b8bfcc'), hk)
    # legs down onto the roof
    legs = [[(x0 + w * f, y_deck), (x0 + w * f, y_deck + w * 0.12)] for f in (0.04, 0.35, 0.65, 0.96)]
    cv.lines(legs, lw * 1.2, dark)
    # deck slab (slats)
    th = w * 0.035
    cv.poly([(x0, y_deck - th), (x1, y_deck - th), (x1, y_deck), (x0, y_deck)], hz(_h('#8a8a96'), hk))
    cv.lines([[(x0, y_deck - th), (x1, y_deck - th)]], lw * 0.8, hz(RIM, hk * 0.5), 0.9)
    # posts + pole
    ph_ = w * 0.34
    ptop = y_deck - th - ph_
    for f in (0.1, 0.9):
        px = x0 + w * f
        cv.lines([[(px, y_deck - th), (px, ptop)]], lw * 1.3, metal)
        cv.lines([[(px - w * 0.05, ptop + w * 0.02), (px + w * 0.05, ptop + w * 0.02)]], lw, metal)
    poles = [ptop + w * 0.02]
    cv.lines([[(x0 + w * 0.05, poles[0]), (x1 - w * 0.05, poles[0])]], lw * 0.9, metal)
    cv.lines([[(x0 + w * 0.05, poles[0] - 0.4 * s), (x1 - w * 0.05, poles[0] - 0.4 * s)]], max(0.5, 0.4 * s),
             hz(RIM, hk * 0.4), 0.9)
    # washing
    cols = [_h('#f6f4f0'), _h('#9cc4ea'), _h('#f4cbb6'), _h('#ecebf6'), _h('#b6d6bc'), _h('#f6e4a8'), _h('#e8a8a8')]
    xx = x0 + w * 0.1
    while xx < x1 - w * 0.14:
        kind = rng.random()
        c = cols[int(rng.integers(0, len(cols)))]
        if kind < 0.2:                                   # futon folded over the pole
            cw, ch = w * rng.uniform(0.2, 0.26), ph_ * rng.uniform(0.5, 0.6)
            c = _h('#f2ece0') if rng.random() < 0.5 else _h('#c8d8f0')
        elif kind < 0.6:                                 # towel
            cw, ch = w * rng.uniform(0.07, 0.1), ph_ * rng.uniform(0.3, 0.4)
        else:                                            # shirt on a hanger
            cw, ch = w * rng.uniform(0.1, 0.12), ph_ * rng.uniform(0.38, 0.46)
        yt = poles[0]

        def ccol(gx, gy, xx=xx, cw=cw, c=c, yt=yt, ch=ch):
            u = np.clip((gx - xx) / cw, 0, 1)
            v = np.clip((gy - yt) / ch, 0, 1)
            # flat cloth in cool sky light, a soft fold shadow and a warm sun-lit top edge (backlit)
            k = 0.86 - 0.1 * np.exp(-((u - 0.62) / 0.12) ** 2) - 0.08 * v + 0.12 * _ss(0.12, 0.0, v)
            return hz(c * k[..., None], hk)
        if kind >= 0.6:
            sh = cw * 0.3
            pts = [(xx + cw * 0.35, yt), (xx + cw * 0.65, yt), (xx + cw, yt + sh), (xx + cw * 0.85, yt + sh * 1.3),
                   (xx + cw * 0.82, yt + ch), (xx + cw * 0.18, yt + ch), (xx + cw * 0.15, yt + sh * 1.3), (xx, yt + sh)]
        else:
            pts = [(xx, yt), (xx + cw, yt), (xx + cw * 0.98, yt + ch), (xx + cw * 0.02, yt + ch)]
        cv.poly(pts, ccol)
        cv.lines([[(xx, yt + 0.3 * s), (xx + cw, yt + 0.3 * s)]], max(0.5, 0.5 * s), hz(RIM, hk * 0.5), 0.6)
        xx += cw + w * rng.uniform(0.015, 0.04)
    # railing on the deck front
    ry = y_deck - th - w * 0.14
    cv.lines([[(x0, ry), (x1, ry)]], lw, metal)
    cv.lines([[(x0 + w * i / 16, ry), (x0 + w * i / 16, y_deck - th)] for i in range(17)], max(0.5, 0.5 * s), metal, 0.85)
    cv.lines([[(x0, ry - 0.4 * s), (x1, ry - 0.4 * s)]], max(0.5, 0.4 * s), hz(RIM, hk * 0.4), 0.9)


def bs_dish(cv, x, y, r, s, hk, facing=-1):
    """BS satellite dish on a short arm: pale disc, dark rim shadow, feed arm + LNB."""
    cv.lines([[(x, y), (x, y + r * 1.4)]], max(0.7, 0.5 * s), hz(_h('#3a4058'), hk))
    cv.ellipse(x, y, r * 0.55, r, hz(_h('#9aa2b4'), hk), ang=0.0)
    cv.ellipse(x + facing * r * 0.08, y, r * 0.45, r * 0.9, hz(_h('#e4e6ec'), hk))
    cv.ellipse(x + facing * r * 0.16, y - r * 0.25, r * 0.18, r * 0.35, hz(_h('#fafaf6'), hk * 0.5), 0.8)
    lx, ly = x + facing * r * 1.1, y + r * 0.35
    cv.lines([[(x + facing * r * 0.2, y + r * 0.8), (lx, ly)]], max(0.6, 0.4 * s), hz(_h('#5a6278'), hk))
    cv.ellipse(lx, ly, r * 0.14, r * 0.12, hz(_h('#3a4058'), hk))


def vent_stack(cv, x, y, h, s, hk):
    w = h * 0.3
    cv.poly([(x - w / 2, y), (x + w / 2, y), (x + w / 2, y - h), (x - w / 2, y - h)],
            lambda gx, gy: hz(_h('#5a6078') + (_h('#c4c8d4') - _h('#5a6078')) *
                              np.clip(1 - (gx - x + w / 2) / w, 0, 1)[..., None] ** 1.5, hk))
    cv.poly([(x - w * 0.8, y - h), (x + w * 0.8, y - h), (x + w * 0.6, y - h - w * 0.5), (x - w * 0.6, y - h - w * 0.5)],
            hz(_h('#3a4058'), hk))
    cv.lines([[(x - w * 0.6, y - h - w * 0.5), (x + w * 0.6, y - h - w * 0.5)]], max(0.5, 0.5 * s), hz(RIM, hk * 0.4), 0.9)


def roof_clutter(cv, x, w, y_r, y_e, kind, s, hk, rng, big=1.0):
    """Clutter for one house roof that faces the camera (hip / gable faces): pick a few items."""
    if kind == 'gable_end':
        if rng.random() < 0.5:
            bs_dish(cv, x + w * 0.22, y_e + (y_e - y_r) * 0.2, w * 0.035 * big, s, hk)
        return
    rh = y_e - y_r
    items = rng.random(4)
    used = []
    if items[0] < 0.32:
        hw = w * rng.uniform(0.18, 0.24)
        xc = x + rng.uniform(-0.12, 0.12) * w
        solar_array(cv, xc, y_r + rh * 0.18, y_r + rh * 0.78, hw, s, hk, rng, nr=3, nc=int(rng.integers(5, 8)))
        used.append((xc - hw, xc + hw))
    if items[1] < 0.3 and big >= 1.0:
        dw = w * rng.uniform(0.32, 0.4)
        xc = x + (0.22 if not used or used[0][0] < x else -0.22) * w
        if not any(a - dw / 2 < xc < b + dw / 2 for a, b in used):
            laundry_deck(cv, xc, y_r + rh * 0.62, dw, s, hk, rng)
            used.append((xc - dw / 2, xc + dw / 2))
    if items[2] < 0.45:
        vx = x + rng.uniform(-0.35, 0.35) * w
        if not any(a < vx < b for a, b in used):
            vent_stack(cv, vx, y_r + rh * 0.35, rh * 0.22, s, hk)
    if items[3] < 0.35:
        bx = x + rng.uniform(-0.3, 0.3) * w
        if not any(a < bx < b for a, b in used):
            bs_dish(cv, bx, y_r + rh * 0.3, w * 0.03 * big, s, hk, facing=-1 if bx > x else 1)


def flat_roof_kit(cv, x0, x1, by, s, hk, rng, fh):
    """Rooftop AC condenser rows, cable trays and a railing on a flat roof (row across x0..x1)."""
    n = max(int((x1 - x0) / (fh * 0.9)), 2)
    for i in range(n):
        if rng.random() < 0.6:
            ax = x0 + (x1 - x0) * (i + rng.uniform(0.1, 0.3)) / n
            T.ac_unit(cv, ax, by - fh * 0.36, fh * 0.5, s, hk)
    cv.lines([[(x0 + fh * 0.2, by - fh * 0.06), (x1 - fh * 0.2, by - fh * 0.06)]], max(0.6, 0.7 * s),
             hz(_h('#6a7288'), hk), 0.8)


def facade_life(cv, x0, x1, by, base, s, hk, rng, fh, ncol):
    """Apartment facade: washing hung on balconies and AC units next to windows."""
    nfl = int((base - by) / fh) + 1
    cols = [_h('#f6f4f0'), _h('#9cc4ea'), _h('#f4cbb6'), _h('#e8e6f2'), _h('#b6d6bc'), _h('#f6e4a8')]
    for k in range(nfl):
        yy = by + 0.35 * fh + k * fh
        for j in range(ncol):
            wx0 = x0 + j * (x1 - x0) / ncol
            cw = (x1 - x0) / ncol
            r = rng.random()
            if r < 0.3:
                # washing across the balcony front
                xx = wx0 + cw * 0.15
                while xx < wx0 + cw * 0.8:
                    ww = cw * rng.uniform(0.08, 0.14)
                    c = cols[int(rng.integers(0, len(cols)))]
                    cv.poly([(xx, yy + fh * 0.12), (xx + ww, yy + fh * 0.12), (xx + ww, yy + fh * 0.48),
                             (xx, yy + fh * 0.48)], hz(c * 0.9, hk))
                    xx += ww + cw * 0.03
            elif r < 0.55:
                T.ac_unit(cv, wx0 + cw * 0.72, yy + fh * 0.3, cw * 0.18, s, hk)
