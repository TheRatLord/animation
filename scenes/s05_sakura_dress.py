"""s05_sakura round 6: extra set dressing for the far-bank town (denser mid-ground architecture detail).

Adds boxes to the environment box list (same 16-column layout as s05_sakura_env):
  * rain gutters under every roof edge / eave and downpipes running down the river-facing facades;
  * wall-mounted AC outdoor units (with a pipe duct) on the plain facades, more on the houses;
  * TV antennas (mast + Yagi cross bars) on the pitched house roofs, satellite-dish boxes on the blocks;
  * block-wall / aluminium fences in front of the riverside houses, with gate gaps;
  * laundry poles with washing on more of the house balconies, futons over the rails;
  * electric meter boxes / small signs at street level.
"""
import numpy as np

import s05_sakura_env as E

GREY = (0.62, 0.64, 0.7)
DARK = (0.4, 0.42, 0.48)
PIPE = (0.78, 0.78, 0.8)
AC = (0.93, 0.93, 0.94)
WALL = (0.84, 0.82, 0.78)


def dress(B, seed=61):
    rng = np.random.default_rng(seed)
    rows = []

    def add(u0, u1, Y0, Y1, s0, s1, typ, col, layer=0):
        rows.append([u0, u1, Y0, Y1, s0, s1, typ, col[0], col[1], col[2], rng.integers(0, 1000), layer,
                     0, 0, 0, 0])

    n0 = B.shape[0]
    for i in range(n0):
        typ = int(B[i, 6])
        lay = int(B[i, 11])
        u0, u1, Y0, Y1, s0, s1 = B[i, :6]
        if lay != 0:
            continue
        if typ == 0:
            if u1 < -60 or s0 > 320:
                continue
            w = s1 - s0
            hB = Y1 - Y0
            low = hB < 8.5
            # downpipes on the river face (one or two), with a lit edge from the rail shading
            for sp in ([s0 + 0.25] if rng.random() < 0.5 else [s0 + 0.25, s1 - 0.4]):
                add(u1, u1 + 0.14, Y0, Y1 - 0.1, sp, sp + 0.13, 1, PIPE)
            if not low:
                # coping gutter line under the parapet
                add(u1, u1 + 0.18, Y1 - 0.32, Y1 - 0.18, s0, s1, 1, (0.7, 0.72, 0.78))
                # wall AC units on plain storeys + a duct running up to them
                for _ in range(int(rng.integers(1, 4))):
                    fl = int(rng.integers(1, max(2, int(hB / 2.8))))
                    y = Y0 + fl * 2.8 + 0.4
                    if y + 0.7 > Y1 - 0.5:
                        continue
                    a = s0 + rng.uniform(0.6, max(0.8, w - 1.4))
                    add(u1, u1 + 0.32, y, y + 0.62, a, a + 0.82, 8, AC)
                    add(u1, u1 + 0.08, y + 0.3, y + 1.6, a + 0.86, a + 0.95, 1, PIPE)
                # satellite dish / aerial on some roofs
                if rng.random() < 0.45:
                    a = s0 + rng.uniform(0.5, max(0.6, w - 1.5))
                    b = u0 + rng.uniform(0.5, 3.0)
                    add(b, b + 0.08, Y1, Y1 + rng.uniform(2.0, 3.5), a, a + 0.08, 1, DARK)
                    add(b - 0.5, b + 0.6, Y1 + 1.4, Y1 + 1.46, a, a + 0.05, 1, DARK)
            else:
                # houses: wall AC unit by the ground-floor window, meter box
                for _ in range(int(rng.integers(1, 3))):
                    y = Y0 + rng.choice([0.05, 2.9])
                    a = s0 + rng.uniform(0.4, max(0.5, w - 1.2))
                    add(u1, u1 + 0.32, y, y + 0.62, a, a + 0.82, 8, AC)
                if rng.random() < 0.8:
                    a = s0 + rng.uniform(0.3, max(0.4, w - 0.6))
                    add(u1, u1 + 0.1, Y0 + 1.2, Y0 + 1.6, a, a + 0.3, 8, (0.86, 0.87, 0.9))
                # fence: block wall with an aluminium top rail, broken by a gate
                if u1 > -42 and rng.random() < 0.75:
                    fu = u1 + rng.uniform(1.3, 2.0)
                    g0 = s0 + rng.uniform(0.25, 0.6) * w
                    g1 = g0 + rng.uniform(1.0, 1.6)
                    for (a, b) in ((s0 - 0.3, g0), (g1, s1 + 0.3)):
                        if b - a < 0.4:
                            continue
                        add(fu, fu + 0.15, Y0, Y0 + 0.55, a, b, 9, WALL)
                        add(fu + 0.05, fu + 0.1, Y0 + 0.55, Y0 + 1.2, a, b, 10, (0.8, 0.84, 0.9))
                    for gp in (g0, g1 - 0.15):
                        add(fu, fu + 0.18, Y0, Y0 + 1.25, gp, gp + 0.15, 9, WALL)
                # washing pole on a second-floor balcony
                if rng.random() < 0.6 and hB > 5.0:
                    a0 = s0 + rng.uniform(0.3, max(0.4, 0.5 * w))
                    a1 = min(s1 - 0.3, a0 + rng.uniform(2.0, 3.5))
                    yp = Y0 + 2.8 + 1.7
                    add(u1 + 0.4, u1 + 0.45, yp, yp + 0.04, a0, a1, 1, (0.85, 0.85, 0.88))
                    xx = a0 + 0.15
                    while xx < a1 - 0.4:
                        lw = rng.uniform(0.3, 0.6)
                        add(u1 + 0.4, u1 + 0.44, yp - rng.uniform(0.6, 1.1), yp, xx, xx + lw, 14,
                            E.LAUNDRY[int(rng.integers(len(E.LAUNDRY)))])
                        xx += lw + rng.uniform(0.08, 0.35)
        elif typ in (12, 13):
            if u1 < -60 or s0 > 320:
                continue
            # gutter along the eaves on the river side
            add(u1 - 0.05, u1 + 0.12, Y0 - 0.12, Y0 + 0.02, s0, s1, 1, (0.72, 0.72, 0.76))
            # TV antenna on the ridge: mast + three Yagi cross bars
            if rng.random() < 0.7:
                uc = 0.5 * (u0 + u1) + rng.uniform(-0.8, 0.8)
                sc = s0 + rng.uniform(0.25, 0.75) * (s1 - s0)
                hm = rng.uniform(1.6, 2.6)
                add(uc, uc + 0.07, Y1 - 0.3, Y1 + hm, sc, sc + 0.07, 1, DARK)
                for k in range(3):
                    yb = Y1 + hm - 0.15 - 0.35 * k
                    ln = 0.9 - 0.2 * k
                    add(uc, uc + 0.05, yb, yb + 0.05, sc - ln / 2, sc + ln / 2, 1, DARK)
                add(uc - 0.7, uc + 0.7, Y1 + hm - 0.9, Y1 + hm - 0.86, sc, sc + 0.05, 1, DARK)
    if not rows:
        return B
    return np.concatenate([B, np.array(rows, np.float64)], 0)
