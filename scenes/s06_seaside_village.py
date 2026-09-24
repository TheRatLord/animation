"""Tiny fishing village at the foot of the headland for s06_seaside (far layer, ~350 m).

Backlit hazy-violet houses climbing the slope in rows (gabled roofs with a warm rim on the ridge line,
walls split into a shaded and a faintly sunlit face), windows either lit warm (dusk interiors) or
catching the pink sky, tiny AC units, a seawall with a harbour breakwater and its red beacon, moored
fishing boats, a few poles with wires. Drawn into the headland canvas before it is finalised, so the
village also appears in the headland's sea reflection."""
import math
import numpy as np

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


def village(far, cv, wl, x0f=0.235, x1f=0.39):
    sc = far.sc
    W, H = sc.W, sc.H
    u = W / 1920.0
    rng = np.random.default_rng(314)
    X0, _ = far.fr(x0f, 0)
    X1, _ = far.fr(x1f, 0)
    wall_s = cc('#6a5a92')
    wall_l = cc('#a488ac')
    roof = cc('#3a2e62')
    roof2 = cc('#4a3a70')
    rimc = cc('#ffb48a')
    win_lit = np.array([1.15, 0.8, 0.46], np.float32)
    win_sky = cc('#e6a6c0')
    # seawall / quay along the waterline
    qy = wl - 0.004 * H
    cv.poly([(X0 - 0.01 * W, wl + 0.001 * H), (X0 - 0.01 * W, qy), (X1 + 0.012 * W, qy), (X1 + 0.012 * W, wl + 0.001 * H)],
            cc('#8a78a8'))
    cv.line([(X0 - 0.01 * W, qy), (X1 + 0.012 * W, qy)], max(0.8 * u, 0.5), cc('#f0b8a0'), 0.8)
    # breakwater with a red beacon at its tip
    bx0 = X1 - 0.02 * W
    bx1 = X1 + 0.03 * W
    cv.poly([(bx0, wl + 0.0015 * H), (bx0, wl - 0.0022 * H), (bx1, wl + 0.002 * H), (bx1, wl + 0.004 * H)],
            cc('#7a6898'))
    cv.line([(bx0, wl - 0.0022 * H), (bx1, wl + 0.002 * H)], max(0.7 * u, 0.5), cc('#f4bca4'), 0.8)
    bh = 0.012 * H
    cv.poly([(bx1 - 2.2 * u, wl + 0.002 * H), (bx1 - 2.2 * u, wl + 0.002 * H - bh), (bx1 + 2.2 * u, wl + 0.002 * H - bh),
             (bx1 + 2.2 * u, wl + 0.002 * H)], cc('#c84a5a'))
    cv.poly([(bx1 - 2.2 * u, wl + 0.002 * H - bh * 0.55), (bx1 + 2.2 * u, wl + 0.002 * H - bh * 0.55),
             (bx1 + 2.2 * u, wl + 0.002 * H - bh * 0.4), (bx1 - 2.2 * u, wl + 0.002 * H - bh * 0.4)], cc('#f0e0e0'))
    far.beacon = (bx1, wl + 0.002 * H - bh - 1.5 * u)
    # moored boats inside the breakwater
    for k in range(3):
        x = X1 - 0.045 * W + k * 0.011 * W + rng.uniform(-2, 2) * u
        L = rng.uniform(0.006, 0.009) * W
        y = wl + 0.0012 * H
        cv.poly([(x - L / 2, y - 0.0025 * H), (x + L / 2, y - 0.003 * H), (x + L * 0.4, y), (x - L * 0.42, y)],
                cc('#e8d8e8'))
        cv.poly([(x - L * 0.15, y - 0.0028 * H), (x - L * 0.15, y - 0.006 * H), (x + L * 0.12, y - 0.006 * H),
                 (x + L * 0.12, y - 0.0029 * H)], cc('#b7a2c8'))
        cv.line([(x + L * 0.05, y - 0.006 * H), (x + L * 0.05, y - 0.013 * H)], max(0.6 * u, 0.5), cc('#5a4a78'))
        cv.line([(x - L / 2, y - 0.0025 * H), (x + L / 2, y - 0.003 * H)], max(0.6 * u, 0.5), cc('#ffc8a8'), 0.8)
    # houses: rows climbing the slope (back rows first)
    rows = [(0.038, 0.68, 0.32), (0.03, 0.76, 0.45), (0.022, 0.84, 0.6), (0.014, 0.92, 0.78), (0.006, 1.0, 1.0)]
    roofs = [cc('#3a2e62'), cc('#4a3a70'), cc('#7a3c4c'), cc('#2c4a78'), cc('#2e5a62'), cc('#5e4638'), cc('#46406e')]
    houses = []
    for (dy, sc_, haze) in rows:
        x = X0 + rng.uniform(0, 0.006) * W
        while x < X1:
            w = rng.uniform(0.0075, 0.014) * W * sc_
            h = rng.uniform(0.007, 0.011) * H * sc_ * (1.6 if rng.random() < 0.15 else 1.0)
            base = qy - dy * H + rng.uniform(-0.003, 0.003) * H
            houses.append((x, w, h, base, haze))
            x += w + rng.uniform(0.001, 0.006) * W + (rng.uniform(0.006, 0.016) * W if rng.random() < 0.3 else 0.0)
    for (x, w, h, base, haze) in houses:
        top = base - h
        split = x + w * rng.uniform(0.6, 0.75)
        hazec = cc('#b8a0c8')
        ws = C.lerp(hazec, wall_s, haze)
        wl_ = C.lerp(hazec, wall_l, haze)
        cv.poly([(x, base), (x, top), (x + w, top), (x + w, base)], ws)
        cv.poly([(split, base), (split, top), (x + w, top), (x + w, base)], wl_, 0.85)
        # roof: gable or flat
        rh = h * rng.uniform(0.35, 0.6)
        ov = w * 0.08
        if rng.random() < 0.8:
            rc = roofs[int(rng.integers(0, len(roofs)))]
            cv.poly([(x - ov, top + 0.3 * u), (x + w * 0.25, top - rh), (x + w * 0.75, top - rh), (x + w + ov, top + 0.3 * u)],
                    C.lerp(hazec, rc, haze))
            # sunlit roof plane (the sun is to the right): a lighter, warmer tint on the right slope
            cv.poly([(x + w * 0.75, top - rh), (x + w + ov, top + 0.3 * u), (x + w * 0.62, top + 0.3 * u),
                     (x + w * 0.5, top - rh)], C.lerp(hazec, rc * 1.35 + cc('#402010'), haze), 0.7)
            # gutter line under the eave
            cv.line([(x - ov, top + 0.6 * u), (x + w + ov, top + 0.6 * u)], max(0.5 * u, 0.4),
                    C.lerp(hazec, cc('#2a2248'), haze), 0.8)
            cv.line([(x + w * 0.25, top - rh), (x + w * 0.75, top - rh), (x + w + ov, top + 0.3 * u)],
                    max(0.7 * u, 0.5), rimc, 0.55 * haze)
        else:
            cv.poly([(x - 0.5 * u, top), (x - 0.5 * u, top - 1.5 * u), (x + w + 0.5 * u, top - 1.5 * u),
                     (x + w + 0.5 * u, top)], C.lerp(hazec, roof2, haze))
            # rooftop water tank / antenna
            if rng.random() < 0.6:
                tx = x + w * rng.uniform(0.3, 0.7)
                cv.line([(tx, top - 1.5 * u), (tx, top - h * 0.8)], max(0.5 * u, 0.4), cc('#4a3a70'))
                cv.line([(tx - 2 * u, top - h * 0.7), (tx + 2 * u, top - h * 0.7)], max(0.5 * u, 0.4), cc('#4a3a70'))
        # windows: a row (or two) of tiny panes, some lit, some reflecting the sky
        nr = 2 if h > 0.009 * H else 1
        for rr in range(nr):
            wy = top + h * (0.28 + 0.36 * rr)
            nw = max(1, int(w / (5.5 * u)))
            for k in range(nw):
                if rng.random() < 0.3:
                    continue
                wx = x + (k + 0.5) * w / nw
                ww_, wh_ = 1.6 * u, 1.5 * u
                lit = rng.random() < 0.4
                c = win_lit * rng.uniform(0.9, 1.15) if lit else win_sky
                if lit:        # warm glow spilling round the lit pane
                    g_ = 2.6 * u
                    cv.poly([(wx - ww_ - g_, wy - wh_ - g_), (wx + ww_ + g_, wy - wh_ - g_), (wx + ww_ + g_, wy + wh_ + g_),
                             (wx - ww_ - g_, wy + wh_ + g_)], np.array([1.0, 0.62, 0.36], np.float32), 0.18 * (0.4 + 0.6 * haze))
                cv.poly([(wx - ww_, wy - wh_), (wx + ww_, wy - wh_), (wx + ww_, wy + wh_), (wx - ww_, wy + wh_)],
                        C.lerp(hazec, c, haze) if not lit else c, 0.95)
                # window frame / sash bar
                cv.line([(wx, wy - wh_), (wx, wy + wh_)], max(0.35 * u, 0.3), C.lerp(hazec, cc('#3a2e5a'), haze), 0.6)
        # AC unit on the side wall
        if rng.random() < 0.5 and w > 0.008 * W:
            ax = x + w + 0.2 * u
            ay = base - h * 0.35
            cv.poly([(ax, ay), (ax + 2.6 * u, ay), (ax + 2.6 * u, ay - 2 * u), (ax, ay - 2 * u)], cc('#bcb0d0'))
    # utility poles + wires along the waterfront road
    pts = []
    for k, x in enumerate(np.linspace(X0 + 0.004 * W, X1 - 0.004 * W, 5)):
        pb = qy - 0.0015 * H
        pt = pb - 0.02 * H
        cv.line([(x, pb), (x, pt)], max(0.7 * u, 0.5), cc('#4a3a70'))
        cv.line([(x - 2.2 * u, pt + 1.2 * u), (x + 2.2 * u, pt + 1.2 * u)], max(0.6 * u, 0.4), cc('#4a3a70'))
        pts.append((x, pt + 1.2 * u))
    for a, b in zip(pts[:-1], pts[1:]):
        tt = np.linspace(0, 1, 20)
        xs = a[0] + (b[0] - a[0]) * tt
        ys = a[1] + (b[1] - a[1]) * tt + 0.0025 * H * 4 * tt * (1 - tt)
        cv.line(np.stack([xs, ys], 1), max(0.45 * u, 0.4), cc('#4a3a70'), 0.8)
