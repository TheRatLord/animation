"""s05_sakura round 12: cherry trees repainted for the reviewer's 'stacked UFO discs / chopstick stubs' notes.

Tree12 (model, metres, x right / y up, trunk foot at the origin)
  * trunk: tapered cubic with a root flare and a random lean / curve style;
  * RECURSIVE limbs: every limb is a gently curving cubic whose radius tapers continuously; at its end it
    forks into 2-3 children whose radii follow Leonardo's rule (area preserved) so there is never a step,
    a kink or a truncated stub; occasional lateral side limbs;
  * the terminal limbs are grouped into 5-8 clusters -> one blossom clump per cluster, placed so the limb
    ends sit INSIDE the clump's lower half, and 2-4 fine twigs fan from each limb end up into the clump;
  * clumps are unions of irregular rounded lobes (tilted, some drooping, none flat-bottomed), spaced so
    real sky gaps stay open; loose twigs with sparse blossom bunches cross the gaps.

Painter12 (card painter: paint(tree, rng) -> premultiplied ss-card, ox, oy)
  * each clump is shaded as ONE form: lobe normal + clump normal against a top-left light, mapped to
    lit / mid / mauve value planes with soft anti-aliased transitions (no posterized steps), scalloped at
    blossom scale; the mauve wraps around under each lobe; front lobes cut crisp lit edges across the lobes
    behind them;
  * 5-petal florets only on the silhouette (dense on the sun side), twig sprays, a thin warm rim on the
    sun-facing edges; fine twigs visibly entering the clump undersides;
  * wood: dark tapered silhouettes, a subtle lighter sun-side bark plane, knots and darker crotches, a
    1-3 px warm gold rim on the sun side only.
"""
import math

import numpy as np
import cv2


# ============================================================================ helpers

def _cbez(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def _qbez(p0, p1, p2, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.asarray(p, np.float64) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def _rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _sample(a, dx, dy):
    """dst(p) = a(p + (dx, dy)) (zero outside)"""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)


def _vnoise(h, w, cell, rng):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:h, :w]


def _bumps(h, w, r, rng):
    """blossom-scale smooth bump field (~N(0,1)): scallops terminators without stair steps"""
    n = rng.standard_normal((h, w)).astype(np.float32)
    b = cv2.GaussianBlur(n, (0, 0), max(0.8, r))
    return b / (b.std() + 1e-6)


def _floret_poly(x, y, r, ph, sc):
    th = np.linspace(0, 2 * math.pi, 30, endpoint=False)
    if r > 2.5:
        rr = r * (0.6 + 0.4 * np.abs(np.cos(2.5 * (th + ph))) ** 0.55)
    else:
        rr = np.full_like(th, r)
    return np.round(np.stack([x + rr * np.cos(th), y + rr * np.sin(th)], 1) * sc).astype(np.int32)


# ============================================================================ model

class Tree12:
    def __init__(self, rng, lean=-1.0, scale=1.0, big=True, spread=1.0, hmul=1.0, csize=1.0, bunches=True,
                 n_clumps=None, depth=None):
        s = scale
        self.s = s
        self.lean = lean
        sg = 1.0 if lean >= 0 else -1.0
        self.sg = sg
        self.wood = []           # (P (n,2), r (n,), kind) kind 0 trunk, 1 limb, 2 fine twig, 4 loose twig
        self.forks = []          # (point, radius) crotches
        self.tips = []
        self.clumps = []
        self.loose = []          # (x, y, radius, n) sparse blossom bunches on loose twigs
        self.s = s
        # ---- trunk
        hf = rng.uniform(1.25, 1.8) * s * hmul
        la = rng.uniform(0.08, 0.5)
        style = int(rng.integers(0, 3))
        Fx = sg * la * hf
        if style == 0:
            c1x, c2x = sg * 0.02 * hf, Fx * 0.5
        elif style == 1:
            c1x, c2x = -sg * 0.1 * hf, Fx * 1.1
        else:
            c1x, c2x = Fx * 0.33, Fx * 0.7
        T = _cbez([0.0, -0.12 * s], [c1x, 0.36 * hf], [c2x, 0.7 * hf], [Fx, hf], 40)
        f = np.linspace(0, 1, len(T))
        r0 = rng.uniform(0.15, 0.2) * s
        yb = T[:, 1] + 0.12 * s
        rt = r0 * (1.0 - 0.3 * f) * (1.0 + 0.4 * np.exp(-np.maximum(yb, 0) / (0.2 * s)))
        self.wood.append((T, rt, 0))
        F = T[-1].copy()
        self.F = F
        self.hf = hf
        tdir = _unit(T[-1] - T[-6])
        rF = float(rt[-1])
        self.forks.append((F, rF))
        self.maxd = depth if depth is not None else (3 if big else 2)
        # ---- scaffold limbs (Leonardo's rule: sum r^2 = rF^2)
        nsc = 3 if (big or rng.random() < 0.5) else 2
        if nsc == 3:
            angs = [rng.uniform(-42, -22), rng.uniform(-6, 14), rng.uniform(34, 58)]
        else:
            angs = [rng.uniform(-34, -12), rng.uniform(22, 48)]
        w = rng.uniform(0.6, 1.0, nsc)
        w = w / w.sum()
        for j, ang in enumerate(angs):
            d = _rot(tdir, -sg * ang)
            if d[1] < 0.2:
                d = _unit([d[0], 0.2])
            Lm = rng.uniform(0.95, 1.35) * s * spread * (1.15 if abs(ang) > 35 else 1.0)
            self._limb(rng, F + rng.normal(0, 0.02, 2) * s, d, Lm, rF * math.sqrt(w[j]) * 1.04, 1, spread)
        # ---- clumps from the terminal limbs
        self._make_clumps(rng, s, csize, n_clumps or (int(rng.integers(6, 9)) if big else int(rng.integers(3, 7))))
        # ---- loose twigs with sparse blossoms crossing the sky gaps
        if bunches:
            self._loose_twigs(rng, s, csize)

    # ------------------------------------------------------------------ limbs
    def _limb(self, rng, p0, d, L, r0, depth, spread):
        s = self.s
        perp = np.array([-d[1], d[0]])
        b1, b2 = rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12)
        # gravitropism: inner limbs curve up, outer limbs (near-horizontal) sag a little at their end
        up = np.array([0.0, 1.0])
        g1 = rng.uniform(0.02, 0.1)
        g2 = rng.uniform(-0.01, 0.12) if depth >= 2 else rng.uniform(0.02, 0.14)
        p1 = p0 + d * L / 3 + perp * L * b1 + up * L * g1
        p2 = p0 + d * 2 * L / 3 + perp * L * b2 + up * L * (g1 + g2) * 0.8
        p3 = p0 + d * L + up * L * g2
        n = max(10, int(18 * L / s))
        P = _cbez(p0, p1, p2, p3, n)
        tt = np.linspace(0, 1, n)
        taper = rng.uniform(0.62, 0.74)
        r1 = max(0.012 * s, r0 * taper)
        r = r0 + (r1 - r0) * tt
        self.wood.append((P, r, 1))
        dend = _unit(P[-1] - P[-3])
        # occasional lateral side limb from the middle
        if depth < self.maxd and rng.random() < 0.35:
            i = int(rng.uniform(0.4, 0.7) * (n - 1))
            sd = _rot(_unit(P[i + 1] - P[i]), rng.choice([-1, 1]) * rng.uniform(35, 55))
            if sd[1] < 0.1:
                sd = _unit([sd[0], 0.25])
            self._limb(rng, P[i], sd, L * rng.uniform(0.45, 0.6), r[i] * 0.55, depth + 1, spread)
        if depth >= self.maxd:
            self.tips.append(dict(p=P[-1].copy(), d=dend, r=r1, L=L))
            return
        self.forks.append((P[-1].copy(), r1))
        nch = 3 if rng.random() < 0.3 else 2
        sgn = rng.choice([-1, 1])
        angs = [sgn * rng.uniform(6, 18), -sgn * rng.uniform(24, 42)]
        if nch == 3:
            angs.append(sgn * rng.uniform(34, 50))
        w = np.array([1.0] + [rng.uniform(0.45, 0.8) for _ in angs[1:]])
        w = w / w.sum()
        for a_, w_ in zip(angs, w):
            cd = _rot(dend, a_)
            if cd[1] < -0.05:
                cd = _unit([cd[0], -0.05])
            lf = rng.uniform(0.38, 0.55) if depth + 1 >= 3 else rng.uniform(0.62, 0.82)
            self._limb(rng, P[-1], cd, L * lf, r1 * math.sqrt(w_) * 1.05, depth + 1, spread)

    # ------------------------------------------------------------------ clumps
    def _make_clumps(self, rng, s, csize, ncl):
        tips = self.tips
        if not tips:
            return
        pts = np.array([t['p'] for t in tips])
        # agglomerative merge of the nearest tip groups down to ncl clusters
        groups = [[i] for i in range(len(tips))]
        cen = [pts[i].copy() for i in range(len(tips))]
        while len(groups) > ncl:
            best, bi, bj = 1e9, 0, 1
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    dd = np.linalg.norm(cen[i] - cen[j]) * (1 + 0.15 * (len(groups[i]) + len(groups[j])))
                    if dd < best:
                        best, bi, bj = dd, i, j
            groups[bi] += groups[bj]
            cen[bi] = pts[groups[bi]].mean(0)
            del groups[bj], cen[bj]
        cl = []
        for g in groups:
            ptg = pts[g]
            dg = _unit(np.mean([tips[i]['d'] for i in g], 0))
            R = csize * s * rng.uniform(0.55, 0.72) * (1.0 + 0.2 * (len(g) - 1) ** 0.7)
            spanx = np.ptp(ptg[:, 0]) if len(g) > 1 else 0.0
            c = ptg.mean(0) + dg * 0.15 * R + np.array([0.0, 0.3 * R])
            cl.append(dict(c=c, R=R, spanx=spanx, tips=g))
        # open real sky gaps: about half the neighbour pairs must not touch
        for it in range(40):
            moved = False
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    a, b = cl[i], cl[j]
                    dx = (a['c'][0] - b['c'][0]) / 1.25
                    dy = (a['c'][1] - b['c'][1]) / 0.85
                    dd = math.hypot(dx, dy)
                    need = 0.95 if ((i * 7 + j * 3) % 2 == 0) else 0.72
                    if dd < need * (a['R'] + b['R']):
                        k = max(0.8, dd / (need * (a['R'] + b['R'])))
                        sm = a if a['R'] < b['R'] else b
                        sm['R'] *= max(0.9, k)
                        moved = True
            if not moved:
                break
        for q in cl:
            q['R'] = max(q['R'], 0.24 * csize * s)
            self.clumps.append(self._clump_shape(rng, q))
        # fine twigs: from each limb end fan up into its clump (their ends hidden inside the blossom)
        for q, cdef in zip(cl, self.clumps):
            for ti in q['tips']:
                tp = tips[ti]
                cdef['entries'].append(tp['p'].copy())
                for _ in range(int(rng.integers(2, 5))):
                    tgt = cdef['c'] + np.array([rng.uniform(-0.75, 0.75) * cdef['rx'],
                                                rng.uniform(-0.1, 0.55) * cdef['ry']])
                    v = tgt - tp['p']
                    ln = np.linalg.norm(v)
                    if ln < 0.05 * s:
                        continue
                    ctrl = tp['p'] + tp['d'] * ln * 0.4 + v * 0.3
                    Q = _qbez(tp['p'], ctrl, tgt, 12)
                    rr = np.linspace(tp['r'] * 0.62, 0.008 * s, 12)
                    cdef['twigs'].append((Q, rr))

    def _clump_shape(self, rng, q):
        """irregular rounded lobes: a body + 3-6 lobes spread round its top and sides, some drooping ones
        hanging from the outer underside; lobes are tilted and never flat-bottomed"""
        R = q['R']
        c = q['c']
        sg = self.sg
        rx = R * rng.uniform(0.95, 1.2) + 0.3 * q['spanx']
        ry = R * rng.uniform(0.72, 0.92)
        lobes = [dict(x=c[0], y=c[1], rx=rx, ry=ry, rot=rng.uniform(-14, 14), order=0.0)]
        nl = int(rng.integers(4, 8))
        base = rng.uniform(0, 2 * math.pi)
        for i in range(nl):
            th = base + 2 * math.pi * i / nl + rng.uniform(-0.3, 0.3)
            # screen-model: y up; bias lobes upward
            ux, uy = math.cos(th), math.sin(th)
            uy = 0.35 + 0.65 * uy if uy < 0 else uy
            rr = R * rng.uniform(0.38, 0.62) * (1.1 if uy > 0.4 else 0.85)
            lx = c[0] + ux * rx * rng.uniform(0.45, 0.72)
            ly = c[1] + uy * ry * rng.uniform(0.45, 0.85)
            lobes.append(dict(x=lx, y=ly, rx=rr * rng.uniform(1.0, 1.35), ry=rr * rng.uniform(0.75, 1.0),
                              rot=rng.uniform(-25, 25), order=uy + rng.normal(0, 0.2)))
        # drooping lobes (weight of blossom on the outer end)
        for _ in range(int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.2]))):
            side = sg if rng.random() < 0.7 else -sg
            rr = R * rng.uniform(0.28, 0.42)
            lobes.append(dict(x=c[0] + side * rx * rng.uniform(0.35, 0.75), y=c[1] - ry * rng.uniform(0.55, 0.85),
                              rx=rr * rng.uniform(0.8, 1.05), ry=rr * rng.uniform(1.05, 1.4),
                              rot=side * rng.uniform(5, 25), order=-1.0 + rng.normal(0, 0.1)))
        # paint order: highest (farthest back) first, lower lobes overlap in front
        lobes.sort(key=lambda l: -l['order'])
        for l in lobes:
            l['k1'] = int(rng.integers(3, 6))
            l['ph1'] = rng.uniform(0, 6.3)
            l['ph2'] = rng.uniform(0, 6.3)
        return dict(c=np.asarray(c, np.float64), R=R, rx=rx, ry=ry, lobes=lobes, entries=[], twigs=[],
                    back=False, seed=int(rng.integers(1 << 30)))

    def _loose_twigs(self, rng, s, csize):
        limbs = [(P, r) for (P, r, k) in self.wood if k == 1]
        if not limbs:
            return
        n = int(rng.integers(2, 5))
        for _ in range(n):
            P, r = limbs[int(rng.integers(len(limbs)))]
            i = int(rng.uniform(0.35, 0.85) * (len(P) - 1))
            d = _unit(P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)])
            sd = _rot(d, rng.choice([-1, 1]) * rng.uniform(35, 70))
            if sd[1] < -0.2:
                sd = _unit([sd[0], -0.2])
            L = rng.uniform(0.45, 0.85) * s
            perp = np.array([-sd[1], sd[0]])
            e = P[i] + sd * L
            Q = _qbez(P[i], P[i] + sd * L * 0.5 + perp * L * rng.uniform(-0.15, 0.15), e, 14)
            rr = np.linspace(min(r[i] * 0.5, 0.03 * s), 0.007 * s, 14)
            self.wood.append((Q, rr, 4))
            for _ in range(int(rng.integers(2, 5))):
                u = rng.uniform(0.35, 1.0)
                j = int(u * 13)
                self.loose.append((Q[j, 0], Q[j, 1] + 0.02 * s, csize * s * rng.uniform(0.06, 0.12),
                                   int(rng.integers(4, 10))))

    def extent(self):
        xs, ys = [], []
        for P, r, _ in self.wood:
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for c in self.clumps:
            for l in c['lobes']:
                m = max(l['rx'], l['ry']) * 1.25 + 0.2 * c['R']
                xs += [l['x'] - m, l['x'] + m]
                ys += [l['y'] - m, l['y'] + m]
        for (x, y, rr, n) in self.loose:
            xs += [x - 2 * rr, x + 2 * rr]
            ys += [y - 2 * rr, y + 2 * rr]
        return min(xs), max(xs), min(ys), max(ys)


# ============================================================================ painter

PAL12 = dict(hot=(0.98, 0.83, 0.87), lit=(0.95, 0.74, 0.83), mid=(0.92, 0.6, 0.76), shade=(0.72, 0.53, 0.77),
             center=(0.9, 0.46, 0.62),
             deep=(0.55, 0.44, 0.69), bounce=(0.92, 0.68, 0.76), rim=(1.24, 1.08, 0.96), cast=(0.42, 0.33, 0.58),
             wood=(0.07, 0.048, 0.065), wood_lit=(0.2, 0.13, 0.12), wood_rim=(1.45, 1.0, 0.55),
             wood_cool=(0.18, 0.18, 0.3), twig=(0.2, 0.13, 0.16), twig_sh=(0.3, 0.2, 0.32))


class Painter12:
    def __init__(self, k, sun_dir=(-0.85, -0.5), fr_m=0.045, fr_min=2.2, rim_px=3.0, wood_rim_px=4.0, pal=None,
                 detail=1.0, ss=2):
        self.k = float(k)
        L = np.asarray(sun_dir, np.float64)
        self.L = L / (np.linalg.norm(L) + 1e-9)
        Lp = self.L + np.array([0.0, -0.9])            # painting light tipped toward 'from above'
        Lp = Lp / np.linalg.norm(Lp)
        self.L3 = np.array([Lp[0] * 0.78, Lp[1] * 0.78, 0.62])
        self.L3 /= np.linalg.norm(self.L3)
        self.fr = max(fr_min, fr_m * k)
        self.rim_px = rim_px
        self.wood_rim_px = wood_rim_px
        self.ss = ss
        p = dict(PAL12)
        if pal:
            p.update(pal)
        self.pal = {kk: np.array(v, np.float32) for kk, v in p.items()}
        self.detail = detail

    # ------------------------------------------------------------------ wood
    def _wood_mask(self, h, w, items, to_px, rmap=None):
        m = np.zeros((h, w), np.uint8)
        SH = 3
        sc = 1 << SH
        k = self.k
        items = sorted(items, key=lambda it: -it[2])
        for P, r, kind in items:
            pp = to_px(P)
            rr = np.maximum(r * k, 0.55)
            for i in range(len(pp) - 1):
                a, b = pp[i], pp[i + 1]
                d = b - a
                ln = math.hypot(*d) + 1e-9
                n = np.array([-d[1], d[0]]) / ln
                quad = np.array([a + n * rr[i], b + n * rr[i + 1], b - n * rr[i + 1], a - n * rr[i]])
                cv2.fillConvexPoly(m, np.round(quad * sc).astype(np.int32), 255, cv2.LINE_AA, SH)
                if rmap is not None:
                    cv2.fillConvexPoly(rmap, np.round(quad).astype(np.int32), float(0.5 * (rr[i] + rr[i + 1])))
            for i in range(0 if kind != 0 else 4, len(pp)):
                if rr[i] * sc < 2:
                    continue
                cv2.circle(m, tuple(np.round(pp[i] * sc).astype(np.int32)), int(round(rr[i] * sc)), 255, -1,
                           cv2.LINE_AA, SH)
        return m.astype(np.float32) / 255.0

    def _paint_wood(self, rgb_full, a_full, tree, to_px_full, rng, canopy_y, kinds, extra=()):
        pal = self.pal
        k = self.k
        items = [(P, r, kd) for (P, r, kd) in list(tree.wood) + list(extra) if kd in kinds]
        if not items:
            return
        # work in the bounding box of the wood only
        allp = np.concatenate([to_px_full(P) for P, r, kd in items], 0)
        rmax = max(float(r.max()) for P, r, kd in items) * k + 8
        H0, W0 = a_full.shape
        bx0 = int(max(0, allp[:, 0].min() - rmax)); bx1 = int(min(W0, allp[:, 0].max() + rmax))
        by0 = int(max(0, allp[:, 1].min() - rmax)); by1 = int(min(H0, allp[:, 1].max() + rmax))
        rgb = rgb_full[by0:by1, bx0:bx1]
        a = a_full[by0:by1, bx0:bx1]
        h, w = a.shape
        canopy_y = canopy_y - by0

        def to_px(P):
            return to_px_full(P) - np.array([bx0, by0])
        if not items:
            return
        rmap = np.zeros((h, w), np.float32)
        m = self._wood_mask(h, w, items, to_px, rmap)
        if m.max() <= 0:
            return
        lx, ly = self.L
        mb = (m > 0.5).astype(np.uint8)
        DT = cv2.distanceTransform(mb, cv2.DIST_L2, 5).astype(np.float32)
        DTs = cv2.GaussianBlur(DT, (0, 0), 1.5)
        gx = cv2.Sobel(DTs, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(DTs, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        side = -(gx * lx + gy * ly) / gn
        rl = np.maximum(cv2.dilate(rmap, np.ones((3, 3), np.uint8)), 1.0)
        frac = DT / rl
        yy = np.arange(h, dtype=np.float32)[:, None]
        # the wood up inside the canopy is in blossom shade
        hk = 0.35 + 0.65 * np.clip((yy - canopy_y) / max(0.08 * h, 4.0), 0.0, 1.0)
        var = _vnoise(h, w, max(8.0, 0.35 * k), rng) - 0.5
        var2 = _vnoise(h, w, max(4.0, 0.08 * k), rng) - 0.5
        col = pal['wood'][None, None, :] * (1.0 + 0.35 * var[..., None] + 0.2 * var2[..., None])
        # subtle lighter bark plane on the sun side (value break ~ a third of the width in)
        pl = _ss(0.05, 0.35, side) * (1.0 - _ss(0.3, 0.45, frac)) * np.clip((rl - 2.0) / 3.0, 0, 1)
        col = col + (pal['wood_lit'] * (1.0 + 0.3 * var[..., None]) - col) * (0.8 * pl * hk)[..., None]
        # cool sky bounce on the shadow edge of the thick wood
        cb = _ss(0.2, 0.6, -side) * (1.0 - _ss(0.12, 0.3, frac)) * np.clip((rl - 4.0) / 4.0, 0, 1)
        col = col + (pal['wood_cool'] - col) * (0.4 * cb)[..., None]
        # bark value breaks: darker crotches, knots, horizontal lenticels
        dk = np.zeros((h, w), np.float32)
        lk = np.zeros((h, w), np.float32)
        for (p, r) in tree.forks:
            q = to_px(np.array([p]))[0]
            rr = r * k
            if rr < 2:
                continue
            cv2.ellipse(dk, (int(q[0]), int(q[1] - 0.2 * rr)), (int(rr * 0.9), int(rr * 1.3)), 0, 0, 360, 1.0, -1,
                        cv2.LINE_AA)
        dk = cv2.GaussianBlur(dk, (0, 0), max(1.0, 0.03 * k)) * 0.55
        for P, r, kd in items:
            if kd not in (0, 1):
                continue
            pp = to_px(P)
            nkn = int((5 if kd == 0 else 1.2 * len(P) / 18) * self.detail + rng.random())
            for _ in range(nkn):
                j = rng.uniform(0.08, 0.85) * (len(pp) - 1)
                i0 = int(j)
                p = pp[i0] + (pp[i0 + 1] - pp[i0]) * (j - i0)
                rr = r[i0] * k
                if rr < 4:
                    continue
                t_ = _unit(pp[i0 + 1] - pp[i0])
                nrm = np.array([-t_[1], t_[0]])
                off = rng.uniform(-0.45, 0.45) * rr
                c = p + nrm * off
                ax = (max(1, int(rr * rng.uniform(0.12, 0.22))), max(1, int(rr * rng.uniform(0.2, 0.35))))
                ang = math.degrees(math.atan2(t_[1], t_[0]))
                cv2.ellipse(dk, (int(c[0]), int(c[1])), ax, ang, 0, 360, 0.75, -1, cv2.LINE_AA)
                # a lighter lip on the lower rim of the knot
                cv2.ellipse(lk, (int(c[0] + 0.4 * ax[0] * lx), int(c[1] + ax[1] * 0.5)), ax, ang, 20, 160, 0.6,
                            max(1, int(rr * 0.05)), cv2.LINE_AA)
            if kd == 0 or r.max() * k > 8:
                nl = int((26 if kd == 0 else 6) * self.detail)
                for _ in range(nl):
                    j = rng.uniform(0, len(pp) - 1.001)
                    i0 = int(j)
                    p = pp[i0] + (pp[i0 + 1] - pp[i0]) * (j - i0)
                    rr = r[i0] * k
                    if rr < 4:
                        continue
                    t_ = _unit(pp[i0 + 1] - pp[i0])
                    nrm = np.array([-t_[1], t_[0]])
                    c = p + nrm * rng.uniform(-0.8, 0.8) * rr
                    ln = rr * rng.uniform(0.12, 0.35)
                    th = max(1, int(round(rr * 0.05)))
                    e0 = c - nrm * ln / 2
                    e1 = c + nrm * ln / 2
                    cv2.line(dk, (int(e0[0]), int(e0[1])), (int(e1[0]), int(e1[1])), 0.35, th, cv2.LINE_AA)
        dk = np.clip(dk, 0, 1) * m
        col = col * (1.0 - 0.55 * dk[..., None])
        col = col + (pal['wood_lit'] * 1.3 - col) * (np.clip(lk, 0, 1) * m * hk * np.clip(side + 0.3, 0, 1))[..., None]
        # rim: pixels whose neighbour TOWARD the sun is outside the wood -> sun-facing edges only
        rp = self.wood_rim_px
        out_sun = 1.0 - _sample(m, lx * rp, ly * rp)
        out_sun2 = 1.0 - _sample(m, lx * rp * 0.5, ly * rp * 0.5)
        rk = np.clip(0.6 * out_sun + 0.6 * out_sun2, 0, 1) * m * _ss(-0.05, 0.3, side)
        rk = rk * np.clip(rl / 2.0, 0.35, 1.0) * hk
        col = col + (pal['wood_rim'] - col) * np.clip(1.1 * rk, 0, 1)[..., None]
        rgb[:] = rgb * (1 - m[..., None]) + col * m[..., None]
        a[:] = a * (1 - m) + m

    # ------------------------------------------------------------------ one clump
    def _lobe_poly(self, l, to_px):
        n = 160
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        r = 1.0 + 0.07 * np.sin(l['k1'] * th + l['ph1']) + 0.035 * np.sin(2.7 * l['k1'] * th + l['ph2'])
        X = l['rx'] * r * np.cos(th)
        Y = l['ry'] * r * np.sin(th)
        a = math.radians(l['rot'])
        ca, sa = math.cos(a), math.sin(a)
        P = np.stack([l['x'] + ca * X - sa * Y, l['y'] + sa * X + ca * Y], 1)
        return to_px(P)

    def _ell_normal(self, l, to_px, x0, y0, h, w, soft=1.0):
        """analytic 'painted' normal of an (un-rotated in px) tilted ellipse lobe: (nx, ny, nz) in screen px frame"""
        k = self.k
        c = to_px(np.array([[l['x'], l['y']]]))[0] - np.array([x0, y0])
        a = -math.radians(l['rot'])                     # y flips in px space
        ca, sa = math.cos(a), math.sin(a)
        xs = np.arange(w, dtype=np.float32)[None, :] - c[0]
        ys = np.arange(h, dtype=np.float32)[:, None] - c[1]
        u = (ca * xs + sa * ys) / (l['rx'] * k * soft)
        v = (-sa * xs + ca * ys) / (l['ry'] * k * soft)
        q2 = np.clip(u * u + v * v, 0, 0.999)
        nz = np.sqrt(1 - q2)
        nx = ca * u - sa * v
        ny = sa * u + ca * v
        return nx, ny, nz

    def _clump(self, rgb, a, c, to_px, tone=0.0):
        k = self.k
        pal = self.pal
        rng = np.random.default_rng(c['seed'])
        lx, ly = self.L
        L3 = self.L3
        fr = self.fr
        polys = [self._lobe_poly(l, to_px) for l in c['lobes']]
        allp = np.concatenate(polys, 0)
        ext = 3 * fr + 0.35 * c['R'] * k + 6
        x0 = int(max(0, math.floor(allp[:, 0].min() - ext)))
        x1 = int(min(a.shape[1], math.ceil(allp[:, 0].max() + ext)))
        y0 = int(max(0, math.floor(allp[:, 1].min() - ext)))
        y1 = int(min(a.shape[0], math.ceil(allp[:, 1].max() + ext)))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return
        h, w = y1 - y0, x1 - x0
        off = np.array([x0, y0])
        bump = _bumps(h, w, 0.9 * fr, rng)
        big = _vnoise(h, w, max(6.0, 0.5 * c['R'] * k), rng) - 0.5
        # clump-scale form (one coherent lit top-left / mauve underside for the whole clump)
        cx_, cy_ = c['c']
        body = dict(x=cx_, y=cy_ + 0.12 * c['ry'], rx=c['rx'] * 1.25, ry=c['ry'] * 1.55, rot=0.0)
        cnx, cny, cnz = self._ell_normal(body, to_px, x0, y0, h, w)
        s_cl = cnx * L3[0] + cny * L3[1] + cnz * L3[2]
        acc = np.zeros((h, w, 3), np.float32)
        U = np.zeros((h, w), np.float32)
        SH = 3
        prevU = []
        edges_over = []
        for i, (l, poly) in enumerate(zip(c['lobes'], polys)):
            pl = poly - off
            pad = 4
            lx0 = int(max(0, math.floor(pl[:, 0].min()) - pad)); lx1 = int(min(w, math.ceil(pl[:, 0].max()) + pad))
            ly0 = int(max(0, math.floor(pl[:, 1].min()) - pad)); ly1 = int(min(h, math.ceil(pl[:, 1].max()) + pad))
            if lx1 <= lx0 + 1 or ly1 <= ly0 + 1:
                continue
            hh, ww = ly1 - ly0, lx1 - lx0
            m8 = np.zeros((hh, ww), np.uint8)
            cv2.fillPoly(m8, [np.round((pl - np.array([lx0, ly0])) * (1 << SH)).astype(np.int32)], 255, cv2.LINE_AA, SH)
            Mi = m8.astype(np.float32) / 255.0
            nx, ny, nz = self._ell_normal(l, to_px, x0 + lx0, y0 + ly0, hh, ww, soft=1.08)
            s_lo = nx * L3[0] + ny * L3[1] + nz * L3[2]
            win = (slice(ly0, ly1), slice(lx0, lx1))
            sv = 0.5 * s_lo + 0.5 * s_cl[win] + 0.028 * bump[win] + 0.03 * big[win] + tone
            # soft painted planes (AA transitions ~2-3 px), terminators scalloped at blossom scale
            wdt = 0.011
            t_hot = _ss(0.64, 0.86, 0.5 * s_lo + 0.5 * s_cl[win] + tone)
            t_lit = _ss(0.5 - wdt, 0.5 + wdt, sv)
            t_mid = _ss(0.2 - wdt * 1.5, 0.2 + wdt * 1.5, sv)
            t_sh = _ss(-0.2, 0.05, sv)
            col = pal['deep'] + (pal['shade'] - pal['deep']) * t_sh[..., None]
            col = col + (pal['mid'] - col) * t_mid[..., None]
            col = col + (pal['lit'] - col) * t_lit[..., None]
            col = col + (pal['hot'] - col) * (0.8 * t_hot)[..., None]
            # warm bounce light along the very bottom of each lobe's underside
            bot = _ss(0.55, 0.95, ny) * (1 - t_mid) * _ss(0.0, 0.5, 1 - nz)
            col = col + (pal['bounce'] - col) * (0.35 * bot)[..., None]
            Uw = U[win]
            accw = acc[win]
            if Uw.max() > 0:
                # a lobe overlapping the paint behind it: its edge crossing the earlier lobes gets florets
                # (in the lobe's own colour, so the overlap edge turns lacy)
                eo = np.clip(Mi - cv2.erode(Mi, np.ones((5, 5), np.uint8)), 0, 1) * (Uw > 0.6) * (ny < -0.25)
                yo_, xo_ = np.nonzero(eo > 0.3)
                if len(yo_):
                    edges_over.append((xo_ + lx0, yo_ + ly0, col[yo_, xo_]))
            accw[:] = accw * (1 - Mi[..., None]) + col * Mi[..., None]
            Uw[:] = Uw + Mi * (1 - Uw)
        if U.max() <= 0:
            return
        acc = acc / np.maximum(U, 1e-4)[..., None]          # straight colour (lobes were composited premultiplied)
        # very faint floret texture inside (no noise mush: a few % value)
        acc = acc * (1.0 + 0.015 * np.clip(bump, -2, 2)[..., None])
        # ---------------- silhouette detail
        Ub = (U > 0.5).astype(np.uint8)
        din = cv2.distanceTransform(Ub, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(1 - Ub, cv2.DIST_L2, 5)
        sd = np.where(Ub > 0, -din, dout).astype(np.float32)
        Bl = cv2.GaussianBlur(U, (0, 0), max(2.0, 2.0 * fr))
        gx = cv2.Sobel(Bl, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Bl, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-9
        nx, ny = -gx / gn, -gy / gn
        facing = nx * lx + ny * ly
        fid = np.zeros((h, w), np.uint16)
        fcov = np.zeros((h, w), np.uint8)
        cols = [np.zeros(3, np.float32)]
        sc = 1 << SH

        def floret(x, y, r, colr):
            if len(cols) >= 65000:
                return
            cols.append(np.asarray(colr, np.float32))
            poly = _floret_poly(x, y, r, rng.uniform(0, 6.3), sc)
            cv2.fillPoly(fcov, [poly], 255, cv2.LINE_AA, SH)
            cv2.fillPoly(fid, [poly], len(cols) - 1, cv2.LINE_8, SH)

        # paint colour extended beyond the silhouette (normalized convolution) for sampling edge florets
        sgf = max(2.0, 1.5 * fr)
        Uc = (U > 0.98).astype(np.float32)
        acc = acc.astype(np.float32)
        acc_f = cv2.GaussianBlur(acc * Uc[..., None], (0, 0), sgf) /             np.maximum(cv2.GaussianBlur(Uc, (0, 0), sgf), 1e-4)[..., None]
        acc_f = np.where((Uc > 0)[..., None], acc, acc_f)

        def base_col(x, y, inward):
            yi0 = int(np.clip(y, 0, h - 1))
            xi0 = int(np.clip(x, 0, w - 1))
            xi = int(np.clip(x - nx[yi0, xi0] * inward, 0, w - 1))
            yi = int(np.clip(y - ny[yi0, xi0] * inward, 0, h - 1))
            return acc_f[yi, xi]

        band = (sd > -1.5) & (sd < 1.5)
        ys_, xs_ = np.nonzero(band)
        centres = []
        if len(xs_):
            fw = np.clip(facing[ys_, xs_], -1, 1)
            dens = (0.45 + 0.55 * np.clip((fw + 0.3) / 1.0, 0, 1)) * self.detail
            nf = int(len(xs_) / (0.85 * fr) * 0.6)
            p = dens / dens.sum()
            sel = rng.choice(len(xs_), min(nf, len(xs_)), replace=False, p=p)
            for j in sel:
                x, y = xs_[j] + 0.5, ys_[j] + 0.5
                r = fr * float(np.clip(rng.lognormal(0.0, 0.35), 0.5, 1.9))
                o = r * rng.uniform(-0.25, 0.8)
                x += nx[ys_[j], xs_[j]] * o
                y += ny[ys_[j], xs_[j]] * o
                bc = base_col(x, y, 2.0 * r)
                colr = bc * (1.0 + rng.normal(0, 0.06))
                if fw[j] > 0.2:
                    colr = colr + (pal['lit'] - colr) * 0.3 * min(1.0, fw[j])
                floret(x, y, r, colr)
                if r > 4.0 and fw[j] > -0.2 and rng.random() < 0.22:
                    centres.append((x, y, r))
        # little bunches (3-6 florets) bulging out of the silhouette: a lacy, blossom-built outline
        if len(xs_):
            nb = int(len(xs_) / (6.0 * fr) * self.detail)
            wb = np.clip(facing[ys_, xs_] + 0.5, 0.05, None)
            for j in rng.choice(len(xs_), min(nb, len(xs_)), replace=False, p=wb / wb.sum()):
                n_ = np.array([nx[ys_[j], xs_[j]], ny[ys_[j], xs_[j]]])
                cc = np.array([xs_[j] + 0.5, ys_[j] + 0.5]) + n_ * fr * rng.uniform(0.3, 1.0)
                bc = base_col(cc[0], cc[1], 3.0 * fr)
                if facing[ys_[j], xs_[j]] > 0.2:
                    bc = bc + (pal['lit'] - bc) * 0.3
                for q in range(int(rng.integers(3, 7))):
                    pq = cc + rng.normal(0, 0.8 * fr, 2)
                    floret(pq[0], pq[1], fr * rng.uniform(0.6, 1.1), bc * (1.0 + rng.normal(0, 0.035)))
        # florets along the edges where a front lobe crosses the lobes behind it (lacy overlap edges)
        for (xo, yo, co) in edges_over:
            no = int(len(xo) / (2.5 * fr) * 0.45 * self.detail)
            for j in rng.choice(len(xo), min(no, len(xo)), replace=False):
                x, y = xo[j] + 0.5, yo[j] + 0.5
                if U[int(y), int(x)] < 0.9:
                    continue
                floret(x, y, fr * rng.uniform(0.55, 1.0), co[j] * (1.0 + rng.normal(0, 0.04)))
        # sprays on thin twigs poking out beyond the silhouette + drooping sprays below
        twig = np.zeros((h, w), np.uint8)
        nsp = int((c['R'] * k / (5.0 * fr)) * self.detail) + int(rng.integers(0, 2))
        if len(xs_) and nsp > 0:
            wsp = np.clip(facing[ys_, xs_] + 0.6, 0.05, None) + 0.8 * np.clip(-ny[ys_, xs_] - 0.7, 0, None) + \
                0.6 * np.clip(ny[ys_, xs_] - 0.7, 0, None)
            sel = rng.choice(len(xs_), min(nsp, len(xs_)), replace=False, p=wsp / wsp.sum())
            for j in sel:
                x, y = xs_[j] + 0.5, ys_[j] + 0.5
                n_ = np.array([nx[ys_[j], xs_[j]], ny[ys_[j], xs_[j]]])
                hang = n_[1] > 0.55
                d = _rot(n_, rng.uniform(-30, 30))
                if hang:
                    d = _unit([d[0] * 0.4, 1.0])
                ln = fr * rng.uniform(2.0, 4.2)
                bend = np.array([-d[1], d[0]]) * ln * rng.uniform(-0.25, 0.25)
                st = np.array([x, y]) - d * fr * 0.8
                en = st + d * ln
                Q = _qbez(st, (st + en) / 2 + bend, en, 8)
                tw = max(1, int(round(fr * 0.13)))
                cv2.polylines(twig, [np.round(Q * sc).astype(np.int32)], False, 255, tw, cv2.LINE_AA, SH)
                fwj = float(facing[ys_[j], xs_[j]])
                bc = base_col(x, y, 3.0 * fr)
                if fwj > 0.1:
                    bc = bc + (pal['lit'] - bc) * 0.4
                for q in range(int(rng.integers(3, 7))):
                    u = rng.uniform(0.5, 1.0)
                    pq = st + (en - st) * u + bend * 4 * u * (1 - u) + rng.normal(0, 0.55 * fr, 2)
                    floret(pq[0], pq[1], fr * rng.uniform(0.55, 0.95), bc * (1.0 + rng.normal(0, 0.04)))
        # a few loose single blossoms just off the sun-side edge
        out_b = (sd > 1.0 * fr) & (sd < 2.4 * fr) & (facing > 0.1)
        yo, xo = np.nonzero(out_b)
        if len(xo):
            no = int(len(xo) / (fr * fr) * 0.02 * self.detail)
            for j in rng.choice(len(xo), min(no, len(xo)), replace=False):
                floret(xo[j] + 0.5, yo[j] + 0.5, fr * rng.uniform(0.5, 0.8), pal['lit'] * (1 + rng.normal(0, 0.03)))
        # the solid paint stops just inside the silhouette: the outer edge is built by the florets alone
        # (small notches of sky between the blossoms instead of a smooth blob outline)
        U = U * _ss(0.15 * fr, 0.85 * fr, -sd + 0.5)
        tw_ = twig.astype(np.float32) / 255.0
        acc = acc * (1 - tw_[..., None]) + pal['twig'] * tw_[..., None]
        U = U + tw_ * (1 - U)
        if len(cols) > 1:
            fc = fcov.astype(np.float32) / 255.0
            fidd = cv2.dilate(fid, np.ones((5, 5), np.uint8))
            fid2 = np.where(fid > 0, fid, fidd)
            C = np.stack(cols, 0)[fid2]
            C = np.where((fid2 > 0)[..., None], C, acc)
            acc = acc * (1 - fc[..., None]) + C * fc[..., None]
            U = U + fc * (1 - U)
            if centres:
                cm = np.zeros((h, w), np.uint8)
                for (x, y, r) in centres:
                    cv2.circle(cm, (int(x * sc), int(y * sc)), max(1, int(0.24 * r * sc)), 255, -1, cv2.LINE_AA, SH)
                cmf = cm.astype(np.float32) / 255.0 * fc * 0.4
                acc = acc + (pal['center'] - acc) * cmf[..., None]
        # fine twigs visibly entering the clump underside (fade out as they go up into the blossom)
        tv = np.zeros((h, w), np.float32)
        for (Q, rr) in c['twigs']:
            pp = to_px(Q) - off
            n = len(pp)
            for i in range(n - 1):
                fade = max(0.0, 1.0 - i / (0.45 * n))
                if fade <= 0:
                    break
                wpx = max(1, int(round(rr[i] * k * 1.6)))
                cv2.line(tv, (int(pp[i][0] * 8), int(pp[i][1] * 8)), (int(pp[i + 1][0] * 8), int(pp[i + 1][1] * 8)),
                         float(fade), wpx, cv2.LINE_AA, 3)
        U = U.astype(np.float32)
        acc = acc.astype(np.float32)
        Bl2 = cv2.GaussianBlur(U, (0, 0), max(3.0, 0.3 * c["R"] * k))
        gy2 = cv2.Sobel(Bl2, cv2.CV_32F, 0, 1, ksize=3)
        gx2 = cv2.Sobel(Bl2, cv2.CV_32F, 1, 0, ksize=3)
        ny2 = -gy2 / (np.sqrt(gx2 * gx2 + gy2 * gy2) + 1e-9)
        tv = np.clip(tv, 0, 1) * U * np.clip(1.0 + sd / (0.3 * c['R'] * k), 0, 1) * _ss(0.2, 0.7, ny2)
        # only on the shadowed underside
        lum = acc.mean(-1)
        tv = tv * np.clip((0.86 - lum) / 0.12, 0, 1)
        acc = acc + (pal['twig_sh'] - acc) * (0.85 * tv)[..., None]
        # thin warm rim on the sun-facing silhouette (lit paint only)
        rp = self.rim_px
        rim = np.clip(U - _sample(U, lx * rp, ly * rp), 0, 1)
        lum = acc.mean(-1)
        rim = rim * np.clip((lum - 0.76) / 0.1, 0, 1)
        acc = acc + (pal['rim'] - acc) * np.clip(rim * 0.7, 0, 1)[..., None]
        if c['back']:
            acc = acc * 0.85 + pal['shade'] * 0.15
        # crisp cast shadow of the clump onto what is already on the card (down-right, away from the sun)
        so = max(2.0, 0.14 * c['R'] * k)
        cs = _sample(U, lx * so, ly * so - 0.3 * so)
        cs = cv2.GaussianBlur(cs, (0, 0), max(0.7, 0.08 * so))
        sa = a[y0:y1, x0:x1]
        sub = rgb[y0:y1, x0:x1]
        cast = np.clip(cs - U, 0, 1) * 0.55
        st = sub / np.maximum(sa, 1e-4)[..., None]           # (card rgb is premultiplied)
        tgt = pal['cast'] * np.maximum(st.mean(-1, keepdims=True) * 1.4, 0.25) * sa[..., None]
        sub[:] = sub + (tgt - sub) * cast[..., None]
        sub[:] = sub * (1 - U[..., None]) + acc * U[..., None]
        sa[:] = sa * (1 - U) + U

    def _loose(self, rgb, a, tree, to_px, rng):
        if not tree.loose:
            return
        pal = self.pal
        h, w = a.shape
        k = self.k
        fr = self.fr
        SH = 3
        sc = 1 << SH
        fcov = np.zeros((h, w), np.uint8)
        fid = np.zeros((h, w), np.uint16)
        cols = [np.zeros(3, np.float32)]
        lx, ly = self.L
        for (x, y, rr, n) in tree.loose:
            q = to_px(np.array([[x, y]]))[0]
            R = rr * k
            for _ in range(n):
                ang = rng.uniform(0, 2 * math.pi)
                d = R * math.sqrt(rng.random())
                px, py = q[0] + d * math.cos(ang), q[1] + d * math.sin(ang) * 0.8
                facing = -(math.cos(ang) * lx + math.sin(ang) * ly)
                v = 0.5 + 0.5 * np.clip(-facing, -1, 1)
                colr = pal['mid'] + (pal['lit'] - pal['mid']) * v + rng.normal(0, 0.03, 3)
                if py > q[1] + 0.3 * R:
                    colr = colr * 0.85 + pal['shade'] * 0.15
                cols.append(colr.astype(np.float32))
                poly = _floret_poly(px, py, fr * rng.uniform(0.6, 1.05), rng.uniform(0, 6.3), sc)
                cv2.fillPoly(fcov, [poly], 255, cv2.LINE_AA, SH)
                cv2.fillPoly(fid, [poly], len(cols) - 1, cv2.LINE_8, SH)
        fc = fcov.astype(np.float32) / 255.0
        fid2 = np.where(fid > 0, fid, cv2.dilate(fid, np.ones((5, 5), np.uint8)))
        C = np.stack(cols, 0)[fid2]
        C = np.where((fid2 > 0)[..., None], C, pal['mid'])
        rgb[:] = rgb * (1 - fc[..., None]) + C * fc[..., None]
        a[:] = a * (1 - fc) + fc

    # ------------------------------------------------------------------ tree
    def paint(self, tree, rng, margin=8):
        k = self.k
        margin = int(margin + 4 * self.fr)
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * margin
        ox = -x0 * k + margin
        oy = y1 * k + margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        rgb = np.zeros((H, W, 3), np.float32)
        a = np.zeros((H, W), np.float32)
        cl = list(tree.clumps)
        # one high clump sits behind the limbs (wood crossing in front of blossom = depth)
        if len(cl) >= 4 and rng.random() < 0.6:
            hi = int(np.argmax([c['c'][1] for c in cl]))
            cl[hi]['back'] = True
        backs = [c for c in cl if c['back']]
        fronts = sorted([c for c in cl if not c['back']], key=lambda c: -c['c'][1])
        canopy_y = to_px(np.array([[0.0, tree.hf + 0.15 * tree.s]]))[0][1]
        for c in backs:
            self._clump(rgb, a, c, to_px, tone=-0.1)
        extra = [(Q, rr, 2) for c in fronts for (Q, rr) in c['twigs']]
        self._paint_wood(rgb, a, tree, to_px, rng, canopy_y, kinds=(0, 1, 2, 4), extra=extra)
        self._loose(rgb, a, tree, to_px, rng)
        for c in fronts:
            self._clump(rgb, a, c, to_px)
        card = np.concatenate([rgb, a[..., None]], -1).astype(np.float32)     # rgb is already premultiplied
        return card, ox, oy
