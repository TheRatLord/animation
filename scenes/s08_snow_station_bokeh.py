"""Near-camera snow bokeh for s08_snow_station (round 3).

Big flakes a hand's width in front of the lens, drawn the way a real defocused highlight behaves:
small-to-medium FILLED soft discs (brighter core, a faint hexagonal aperture edge), screen-blended at
low opacity (0.15-0.3). They are not spread over the whole frame: each flake falls slowly down its own
column that is placed where bokeh reads naturally - inside / beside the amber lamp cones and over the
dark foreground masses (the pine on the left, the trees on the right, the platform foreground) - and a
smooth screen-space weight keeps them out of the empty sky (especially the upper-left), so no disc ever
sits alone in the sky reading as a ghost moon. Discs take the colour of the light behind them: warm
amber near the lamps, cool blue-white elsewhere. Deterministic, smooth in time (no popping)."""
import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def screen_discs(img, xs, ys, rad, soft, cols, alph, rot):
    """Screen-blend filled soft discs with a faint hexagonal edge into img (H, W, 3) in place.
    soft - feather width as a fraction of the radius."""
    H, W = img.shape[0], img.shape[1]
    c30 = math.cos(math.pi / 6)
    for i in range(xs.shape[0]):
        a0 = alph[i]
        if a0 <= 0.002:
            continue
        r = rad[i]
        sf = max(soft[i], 0.06)
        ext = r * (1.0 + sf) + 2.0
        x0 = max(int(xs[i] - ext), 0)
        x1 = min(int(xs[i] + ext) + 2, W)
        y0 = max(int(ys[i] - ext), 0)
        y1 = min(int(ys[i] + ext) + 2, H)
        if x1 <= x0 or y1 <= y0:
            continue
        c0, c1, c2 = cols[i, 0], cols[i, 1], cols[i, 2]
        ro = rot[i]
        for py in range(y0, y1):
            for px in range(x0, x1):
                dx = (px + 0.5 - xs[i]) / r
                dy = (py + 0.5 - ys[i]) / r
                dc = math.sqrt(dx * dx + dy * dy)
                # hexagonal aperture distance (rotated), blended 35% with the circle
                dh = 0.0
                for k in range(3):
                    an = ro + k * math.pi / 3
                    v = abs(dx * math.cos(an) + dy * math.sin(an)) / c30
                    if v > dh:
                        dh = v
                d = 0.8 * dc + 0.2 * dh
                e = (1.0 + sf * 0.5 - d) / sf
                if e <= 0.0:
                    continue
                if e > 1.0:
                    e = 1.0
                e = e * e * (3.0 - 2.0 * e)
                # filled disc: even body, brighter soft core, a barely-there rim
                core = math.exp(-(dc / 0.45) ** 2)
                ring = math.exp(-((d - 0.92) / 0.08) ** 2)
                cov = e * (0.55 + 0.6 * core + 0.1 * ring)
                k = cov * a0
                for c in range(3):
                    cc = c0 if c == 0 else (c1 if c == 1 else c2)
                    v = img[py, px, c]
                    x = cc * k
                    if v < 1.0:
                        img[py, px, c] = v + x * (1.0 - v)
                    else:
                        img[py, px, c] = v + x * 0.05


def _sst(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Bokeh:
    """Fixed set of drifting near-lens flakes; frame(img, t, tx, light, f) adds them."""

    # columns: (u0, u1, v_min, v_max, count, r_min, r_max)  (u, v in frame fractions, r in px @1080p)
    COLUMNS = [
        (0.015, 0.17, 0.34, 1.05, 5, 10, 26),     # over the dark pine (below the open sky)
        (0.555, 0.665, 0.10, 0.62, 5, 8, 22),     # in and beside the foreground lamp's cone
        (0.17, 0.235, 0.36, 0.62, 2, 8, 16),      # under the left platform lamp
        (0.875, 0.99, 0.20, 0.95, 3, 10, 24),     # over the snowy trees on the right
        (0.30, 0.52, 0.74, 1.05, 3, 14, 30),      # over the platform foreground
    ]
    # round 4: large defocused flakes right at the lens (40-120 px across), slow drift
    BIG = [
        (0.02, 0.2, 0.45, 1.05, 3, 24, 58),       # over the pine
        (0.56, 0.7, 0.12, 0.7, 3, 20, 44),        # through the main lamp cone (warm)
        (0.3, 0.55, 0.7, 1.08, 2, 30, 60),        # platform foreground
        (0.8, 0.99, 0.3, 1.0, 2, 24, 52),         # right trees
        (0.17, 0.26, 0.34, 0.6, 1, 20, 34),       # by the left lamp (warm)
    ]

    def __init__(self, W, H, seed=4041):
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        u0, v0, r, vy, vx, ph, vlo, vhi, ulo, uhi, soft, a, rot = ([] for _ in range(13))
        big = []
        for ci, (ua, ub, va, vb, n, ra, rb) in enumerate(self.COLUMNS + self.BIG):
            isbig = ci >= len(self.COLUMNS)
            for i in range(n):
                big.append(isbig)
                u0.append(ua + (ub - ua) * (i + rng.uniform(0.15, 0.85)) / n)
                v0.append(rng.uniform(0, 1))
                r.append(rng.uniform(ra, rb))
                vlo.append(va)
                vhi.append(vb)
                ulo.append(ua)
                uhi.append(ub)
                soft.append(rng.uniform(0.55, 0.9) if isbig else rng.uniform(0.3, 0.55))
                a.append(rng.uniform(0.32, 0.45) if isbig else rng.uniform(0.2, 0.3))
                rot.append(rng.uniform(0, math.pi / 3))
                ph.append(rng.uniform(0, 6.28))
        self.big = np.array(big, bool)
        self.u0 = np.array(u0, np.float32)
        self.v0 = np.array(v0, np.float32)
        self.r = np.array(r, np.float32)
        n = len(u0)
        # slow fall (frame heights / s) and gentle sideways sway; bigger = nearer = faster
        self.vy = (0.035 + 0.0012 * np.minimum(self.r, 30) + rng.uniform(-0.006, 0.006, n)).astype(np.float32)
        self.vx = rng.uniform(0.002, 0.008, n).astype(np.float32)
        self.ph = np.array(ph, np.float32)
        self.vlo, self.vhi = np.array(vlo, np.float32), np.array(vhi, np.float32)
        self.ulo, self.uhi = np.array(ulo, np.float32), np.array(uhi, np.float32)
        self.soft = np.array(soft, np.float32)
        self.a = np.array(a, np.float32)
        self.rot = np.array(rot, np.float32)
        self.par = (0.8 + 0.01 * self.r).astype(np.float32)

    def frame(self, img, t, tx, light, f):
        W, H = self.W, self.H
        s = W / 1920.0
        par = tx * f / 2.2 / W * self.par
        u = self.u0 + self.vx * t + 0.006 * np.sin(0.45 * t + self.ph) - par
        # each flake falls down its column and wraps (its weight is ~0 at the wrap point)
        span = (self.vhi - self.vlo) + 0.16
        v = self.vlo - 0.08 + np.mod(self.v0 * span + self.vy * t, span)
        R = self.r * s
        uu, vv = u * W, v * H
        # smooth screen-space presence: fade in/out at the ends of the column, never in the open sky
        wv = _sst(self.vlo - 0.06, self.vlo + 0.04, v) * _sst(self.vhi + 0.06, self.vhi - 0.04, v)
        sky = _sst(0.42, 0.3, v) * _sst(0.5, 0.25, np.abs(u - 0.28))       # upper-left / top sky
        lampcol = _sst(0.07, 0.03, np.abs(u - 0.61)) * _sst(0.66, 0.6, v)   # the lamp cone column is allowed
        w = wv * np.clip(1 - sky + lampcol, 0, 1)
        # light behind each disc (scattered lamp light + halos), averaged over its area
        wk = np.zeros(len(u), np.float32)
        if light is not None:
            for i in range(len(u)):
                x0, x1 = int(max(uu[i] - R[i], 0)), int(min(uu[i] + R[i], W))
                y0, y1 = int(max(vv[i] - R[i], 0)), int(min(vv[i] + R[i], H))
                if x1 > x0 and y1 > y0:
                    wk[i] = float(light[y0:y1:2, x0:x1:2].max(-1).mean())
        wk = np.clip(wk * 3.5, 0, 1)[:, None]
        cool = np.array([1.05, 1.15, 1.45], np.float32)
        amber = np.array([1.7, 1.15, 0.55], np.float32)
        col = cool * (1 - wk) + amber * wk
        a = (self.a * w * (1 + 0.25 * wk[:, 0])).astype(np.float32)
        a = np.clip(a, 0, 0.45).astype(np.float32)
        screen_discs(img, uu.astype(np.float32), vv.astype(np.float32), R.astype(np.float32),
                     self.soft.astype(np.float32), col.astype(np.float32), a, self.rot)
        return img
