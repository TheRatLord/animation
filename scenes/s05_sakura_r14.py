"""s05_sakura round 14: hierarchical cherry crowns (5 Centimeters per Second study).

Reviewer notes addressed: 'pink popcorn / pixel mosaic' (one dab size, spread evenly), noisy per-dab
shading, straight stick trunks with floating dark dashes, no lit-top / lavender-underside split.

Tree14 (model in metres, x right / y up, trunk foot at the origin)
  * a gently S-curved trunk with a root flare that forks into 3-5 scaffold limbs; every limb is a random-walk
    polyline (small kinks at nodes, arching out then drooping) that tapers at every level and forks 3 levels
    deep (laterals along the limb + a terminal fork);
  * each scaffold owns ONE big blossom MASS (3-5 per tree); the mass is built from 15-40 sub-clusters strung
    along its depth-2/3 branches, each sub-cluster from 3-7 bunches, each bunch from a lumpy body dab, 5-petal
    florets and a few single blossoms -> three dab sizes, uneven spacing, real sky holes between the
    sub-clusters and a lacy fringe (sparser bunches, single blossoms, pendant sprays) on the outer / lower edge;
  * every limb has a 'front visible' prefix: the wood shows from the trunk up to that point and then goes
    into the blossom (drawn under the front sub-clusters), so every visible limb is connected to the trunk;
    all other wood sits behind the blossom and only shows through the sky holes.

Painter14
  * value is painted per MASS, not per dab: a smooth form field (blurred mass coverage as a height field,
    lit from the sun tipped to 'above', plus the height within the mass and cast shadow from the masses
    between it and the sun) is sampled at each dab and quantised into broad bands (pink-white top plane,
    light pink, pink, lavender, violet) -> flat painted value masses whose terminator steps along the
    blossom shapes; thin sun-side edges in shade glow warm salmon (transmitted light);
  * crisp warm rim on the sun-facing silhouette, soft lost edges on the shadow side.
"""
import math

import numpy as np
import cv2

import s05_sakura_r13 as R13

SC8 = R13.SC8


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _cbez(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


# ============================================================================ model

class Tree14:
    def __init__(self, rng, scale=1.0, lean=-1.0, spread=1.0, hmul=1.0, big=True, density=1.0, trunk=1.0,
                 nsc=None, cl_r=1.0):
        s = self.s = float(scale)
        sg = 1.0 if lean >= 0 else -1.0
        self.sg = sg
        self.big = big
        self.density = density
        self.cl_r = cl_r
        self.wood = []        # dict(P, r, z, d, vis, m)
        self.clusters = []    # dict(x, y, R, z, m, front)
        self.maxd = 3
        self.mz = {}
        # ---- trunk: S-curve, root flare
        ht = rng.uniform(1.0, 1.4) * s * hmul
        r0 = rng.uniform(0.21, 0.27) * s * trunk
        lx = sg * rng.uniform(0.15, 0.4) * ht
        b1 = rng.uniform(0.08, 0.16) * ht * rng.choice([-1, 1])
        T = _cbez([0.0, -0.1 * s], [b1, 0.35 * ht], [lx * 0.4 - b1 * 0.8, 0.72 * ht], [lx, ht], 34)
        f = np.linspace(0, 1, len(T))
        rt = r0 * (1.0 - 0.3 * f) * (1.0 + 0.6 * np.exp(-np.maximum(T[:, 1] + 0.1 * s, 0) / (0.16 * s)))
        # slight bumps along the trunk (not a machined taper)
        rt = rt * (1.0 + 0.05 * np.sin(f * rng.uniform(9, 15) + rng.uniform(0, 6)))
        self.wood.append(dict(P=T, r=rt, z=0.0, d=0, vis=1.0, m=-1))
        self.ht = ht
        F = T[-1]
        tdir = _unit(T[-1] - T[-4])
        rF = float(rt[-1])
        n = nsc or int(rng.choice([3, 4, 4, 5] if big else [3, 3, 4]))
        base = np.linspace(-60, 58, n) + rng.uniform(-8, 8, n) + sg * 8.0
        w = rng.uniform(0.6, 1.0, n)
        w = w / w.sum()
        order = rng.permutation(n)
        for j in range(n):
            ang = float(base[j])
            d = _unit(0.8 * _rot(np.array([0.0, 1.0]), -ang) + 0.2 * tdir)
            side = 1.0 if d[0] * sg > 0 else 0.0
            L = rng.uniform(1.5, 2.0) * s * spread * (1.0 + 0.25 * side) * (0.85 + 0.3 * abs(math.sin(math.radians(ang))))
            # mass depth: spread so masses overlap in a clear order
            z = (order[j] / max(n - 1, 1) - 0.5) * 1.6 + rng.uniform(-0.1, 0.1)
            self.mz[j] = z
            p0 = F + np.array([rng.uniform(-0.4, 0.4) * rF, rng.uniform(-0.8, 0.0) * rF])
            self._branch(rng, p0, d, L, rF * math.sqrt(w[j]) * 1.1, 1, z, j, True)
        self._finish(rng)

    # ------------------------------------------------------------------ branches
    def _branch(self, rng, p0, d, L, r0, depth, z, m, parent_vis, sm=None):
        s = self.s
        if depth <= 2 or sm is None:
            self._sm = getattr(self, '_sm', -1) + 1
            sm = self._sm
        nn = max(8, int(26 * L / s))
        # random-walk polyline: gentle arch, kinks at a few nodes, outer part droops (more for thin wood)
        P = [np.asarray(p0, np.float64)]
        dd = _unit(d)
        step = L / (nn - 1)
        g = (0.004 if depth == 1 else 0.009 * depth) * rng.uniform(0.5, 1.3)
        curl = rng.normal(0, 1.2)
        kinks = set(rng.choice(np.arange(2, nn - 1), size=min(nn - 3, int(rng.integers(1, 4))), replace=False).tolist())
        for i in range(1, nn):
            a = rng.normal(0, 2.0) + curl
            if i in kinks:
                a += rng.choice([-1, 1]) * rng.uniform(4, 10)
            dd = _rot(dd, a)
            u = i / (nn - 1)
            dd = _unit(dd + np.array([0.0, -g * u * 1.5]) + (np.array([0.0, 0.025]) if u < 0.4 else 0))
            if dd[1] < -0.3:
                dd = _unit([dd[0], -0.3])
            P.append(P[-1] + dd * step)
        P = np.array(P)
        tt = np.linspace(0, 1, nn)
        rmin = 0.005 * s
        r1 = max(rmin, r0 * (0.35 if depth < self.maxd else 0.2))
        r = r0 + (r1 - r0) * tt ** 0.8
        # front-visible prefix (connected to the trunk)
        if depth == 1:
            vis = rng.uniform(0.5, 0.85)
        elif parent_vis and depth == 2:
            vis = rng.uniform(0.25, 0.7) if rng.random() < 0.8 else 0.0
        else:
            vis = 0.0
        self.wood.append(dict(P=P, r=r, z=z, d=depth, vis=vis, m=m, sm=sm))
        if depth < self.maxd:
            nl = {1: int(rng.integers(3, 5)), 2: int(rng.integers(2, 4))}.get(depth, 1)
            ts = np.sort(rng.uniform(0.2, 0.8, nl))
            side = rng.choice([-1, 1])
            for tl in ts:
                i = int(tl * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                cd = _rot(td, side * rng.uniform(28, 52))
                side = -side
                if cd[1] < -0.4:
                    cd = _unit([cd[0], -0.4])
                Lc = L * rng.uniform(0.42, 0.62) * (1.0 - 0.35 * tl)
                if Lc < 0.15 * s:
                    continue
                self._branch(rng, P[i], cd, Lc, max(rmin, r[i] * rng.uniform(0.5, 0.68)), depth + 1,
                             z + rng.uniform(-0.3, 0.3), m, vis > tl + 0.45 * s / L, sm)
            # terminal fork (Leonardo: children share the parent's cross-section)
            td = _unit(P[-1] - P[-3])
            share = rng.uniform(0.35, 0.65)
            for sgn, sh in ((-1, share), (1, 1 - share)):
                cd = _rot(td, sgn * rng.uniform(15, 32))
                self._branch(rng, P[-1], cd, L * rng.uniform(0.45, 0.62), max(rmin, r[-1] * math.sqrt(sh) * 1.05),
                             depth + 1, z + rng.uniform(-0.3, 0.3), m, vis > 0.97, sm)
        # ---- blossom sub-clusters along the outer part of depth>=2 wood (uneven spacing -> sky holes)
        if depth >= 2 or not self.big:
            Rb = rng.uniform(0.3, 0.4) * s * self.cl_r
            t = (0.25 if depth >= 3 else 0.45) + rng.uniform(0, 0.15)
            while t <= 1.0:
                i = int(t * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                nrm = np.array([-td[1], td[0]])
                if nrm[1] < 0:
                    nrm = -nrm
                R = Rb * rng.uniform(0.7, 1.25)
                c = P[i] + nrm * R * rng.uniform(0.05, 0.5) + rng.normal(0, 0.12, 2) * R
                if rng.random() < 0.95 * min(1.0, self.density):
                    self.clusters.append(dict(x=float(c[0]), y=float(c[1]), R=float(R), m=m, sm=sm,
                                              z=z + rng.uniform(-0.25, 0.25), front=bool(rng.random() < 0.3)))
                t += (R / L) * rng.uniform(0.6, 1.3) / max(self.density, 0.4)
            if depth == self.maxd:
                # twig-end spray: a thin drooping twig with a small bunch (lacy silhouette)
                tip = P[-1]
                td = _unit(P[-1] - P[-3])
                for _ in range(int(rng.integers(1, 3))):
                    dd = _unit(td + np.array([rng.normal(0, 0.4), -rng.uniform(0.2, 0.8)]))
                    Lt = rng.uniform(0.1, 0.25) * s
                    q = _cbez(tip, tip + dd * Lt * 0.4, tip + dd * Lt * 0.75, tip + dd * Lt, 6)
                    self.wood.append(dict(P=q, r=np.full(6, rmin * 0.8), z=z, d=depth + 1, vis=0.0, m=m))
                    self.clusters.append(dict(x=float(q[-1, 0]), y=float(q[-1, 1]), R=float(Rb * rng.uniform(0.3, 0.5)), sm=sm,
                                              m=m, z=z + 0.1, front=True, spray=True))

    def _finish(self, rng):
        self.clusters = [c for c in self.clusters if c['y'] > 0.62 * self.ht]
        cl = self.clusters
        X = np.array([c['x'] for c in cl])
        Y = np.array([c['y'] for c in cl])
        R = np.array([c['R'] for c in cl])
        mids = np.array([c['m'] for c in cl])
        self.masses = {}
        for m in np.unique(mids):
            sel = mids == m
            w_ = R[sel] ** 2
            mx, my = np.average(X[sel], weights=w_), np.average(Y[sel], weights=w_)
            dd = np.hypot(X[sel] - mx, (Y[sel] - my) * 1.3) + R[sel]
            mr = max(float(np.percentile(dd, 85)), float(R[sel].max()) * 1.3)
            self.masses[int(m)] = (float(mx), float(my), mr)
        sms = np.array([c['sm'] for c in cl])
        self.submasses = {}
        for m in np.unique(sms):
            sel = sms == m
            w_ = R[sel] ** 2
            mx, my = np.average(X[sel], weights=w_), np.average(Y[sel], weights=w_)
            dd = np.hypot(X[sel] - mx, (Y[sel] - my) * 1.3) + R[sel]
            self.submasses[int(m)] = (float(mx), float(my), max(float(np.percentile(dd, 85)), float(R[sel].max()) * 1.3))
        for c in cl:
            mx, my, mr = self.masses[int(c['m'])]
            dn = math.hypot(c['x'] - mx, (c['y'] - my) * 1.3) / mr
            below = max(0.0, (my - c['y']) / mr)
            c['fr'] = float(np.clip((dn - 0.5) / 0.45 + 0.6 * below, 0, 1))

    def extent(self):
        xs, ys = [], []
        for w in self.wood:
            P, r = w['P'], w['r']
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for c in self.clusters:
            xs += [c['x'] - 1.5 * c['R'], c['x'] + 1.5 * c['R']]
            ys += [c['y'] - 1.9 * c['R'], c['y'] + 1.5 * c['R']]
        return min(xs), max(xs), min(ys), max(ys)


# ============================================================================ painter

PAL14 = dict(
    hot=(1.03, 0.93, 0.94), lit=(1.0, 0.77, 0.84), mid=(0.95, 0.58, 0.72), shade=(0.83, 0.5, 0.68),
    deep=(0.6, 0.4, 0.64), trans=(1.04, 0.72, 0.62), core=(0.9, 0.52, 0.68), rim=(1.16, 1.02, 0.92),
    wood=(0.17, 0.13, 0.2), wood_lit=(0.38, 0.28, 0.28), wood_rim=(1.0, 0.74, 0.54), wood_cool=(0.3, 0.3, 0.48),
    lent=(0.3, 0.23, 0.28))


# far bank: softer aerial pinks, lifted shadows (less value range, cooler)
PAL14_FAR = dict(hot=(1.02, 0.93, 0.95), lit=(0.99, 0.8, 0.87), mid=(0.95, 0.66, 0.8), shade=(0.84, 0.62, 0.82),
                 deep=(0.72, 0.56, 0.8), rim=(1.08, 0.98, 0.94))


def _floret_shape(ph, kind, n):
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    if kind == 0:          # 5-petal blossom with notched petal tips
        c = np.abs(np.cos(2.5 * (th + ph)))
        rr = (0.5 + 0.5 * c ** 0.55) * (1.0 - 0.1 * np.exp(-((th * 5 / (2 * math.pi) + ph * 5 / (2 * math.pi)) % 1.0 - 0.5) ** 2 / 0.002))
    elif kind == 1:        # lumpy bunch body
        rr = (0.8 + 0.1 * np.cos(3 * th + ph) + 0.07 * np.cos(5 * th + 2 * ph) + 0.05 * np.cos(7 * th + 3 * ph))
    else:
        rr = (0.9 + 0.1 * np.cos(2 * th + ph))
    return np.stack([rr * np.cos(th), rr * np.sin(th) * 0.92], 1)


_TPL = {}
NPH = 24


def _floret(x, y, r, ph, kind):
    n = 10 if r < 2.5 else (25 if r < 8 else 35)
    i = int(ph * NPH / 6.2832) % NPH
    key = (kind, n, i)
    t = _TPL.get(key)
    if t is None:
        t = _TPL[key] = _floret_shape(i * 6.2832 / NPH, kind, n)
    return t * r + (x, y)


class Painter14:
    def __init__(self, k, sun_dir=(-0.7, -0.7), fl_m=0.05, dab_min=2.4, pal=None, detail=1.0, rim_px=2.0,
                 wood_rim_px=2.0, far=False, soft_px=1.5):
        self.k = float(k)
        L = _unit(sun_dir)
        self.Ls = L
        Lt = _unit(L + np.array([0.0, -1.1]))          # painting light tipped toward 'from above'
        self.L3 = _unit(np.array([Lt[0] * 0.8, Lt[1] * 0.8, 0.55]))
        self.fl = max(dab_min, fl_m * k)
        p = dict(PAL14)
        if pal:
            p.update(pal)
        self.pal = {kk: np.array(v, np.float32) for kk, v in p.items()}
        self.detail = detail
        self.rim_px = rim_px
        self.far = far
        self.soft_px = soft_px
        self.p13 = R13.Painter13(k, sun_dir=tuple(L), wood_rim_px=wood_rim_px,
                                 pal={kk: p[kk] for kk in ('wood', 'wood_lit', 'wood_rim', 'wood_cool', 'lent')})
        # wood behind the blossom (seen through the holes): cooler, lighter, lost in the crown's shade
        self.p13b = R13.Painter13(k, sun_dir=tuple(L), wood_rim_px=wood_rim_px * 0.6,
                                  pal=dict(wood=(0.3, 0.21, 0.32), wood_lit=(0.44, 0.31, 0.38), wood_rim=(0.8, 0.6, 0.58),
                                           wood_cool=(0.4, 0.34, 0.52), lent=(0.34, 0.25, 0.34)))

    @staticmethod
    def _col(c):
        return (min(255.0, max(0.0, float(c[0]) * SC8)), min(255.0, max(0.0, float(c[1]) * SC8)),
                min(255.0, max(0.0, float(c[2]) * SC8)))

    def _fill(self, rgb, a, pts, col):
        p = [np.round(np.asarray(pts) * 16).astype(np.int32)]
        cv2.fillPoly(rgb, p, self._col(col), cv2.LINE_AA, 4)
        cv2.fillPoly(a, p, 255, cv2.LINE_AA, 4)

    def _cap(self, rgb, a, w, to_px, pw_):
        """round joint at the end of a limb (where its thinner fork children continue): no blunt cut ends"""
        rr = float(w['r'][-1]) * self.k
        if rr < 1.2:
            return
        e = to_px(w['P'][-1:])[0]
        c = pw_.pal['wood']
        cv2.circle(rgb, (int(round(e[0] * 16)), int(round(e[1] * 16))), int(round(rr * 16)), self._col(c), -1,
                   cv2.LINE_AA, 4)
        cv2.circle(a, (int(round(e[0] * 16)), int(round(e[1] * 16))), int(round(rr * 16)), 255, -1, cv2.LINE_AA, 4)

    # ------------------------------------------------------------------ mass value fields
    def _fields(self, tree, to_px, W, H):
        q = 4
        Wl, Hl = W // q + 2, H // q + 2
        k = self.k
        Ls = self.Ls
        L3 = self.L3
        ys, xs = np.mgrid[0:Hl, 0:Wl].astype(np.float32)

        def covs(key):
            out = {}
            for cl in tree.clusters:
                c = out.get(cl[key])
                if c is None:
                    c = out[cl[key]] = np.zeros((Hl, Wl), np.float32)
                x, y = to_px(np.array([[cl['x'], cl['y']]]))[0] / q
                cv2.circle(c, (int(round(x * 4)), int(round(y * 4))), max(1, int(cl['R'] * k / q * 1.2 * 4)), 1.0, -1,
                           cv2.LINE_AA, 2)
            return {kk: np.clip(v, 0, 1) for kk, v in out.items()}

        def field(cov, cen, mrp, occ=None):
            sig = max(1.0, 0.3 * mrp)
            h = cv2.GaussianBlur(cov, (0, 0), sig)
            h = h / max(h.max(), 1e-3)
            gx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            hs = mrp * 0.9
            nx, ny = -gx * hs, -gy * hs
            nn = np.sqrt(nx * nx + ny * ny + 1.0)
            dif = (nx * L3[0] + ny * L3[1] + L3[2]) / nn
            pos = (cen[1] - ys) / mrp                   # +1 at the top of the mass
            v = 1.4 * (dif - 0.55) + 0.45 * pos
            if occ is not None and occ.any():
                d = 0.22 * mrp
                M = np.float32([[1, 0, -Ls[0] * d], [0, 1, -Ls[1] * d]])
                occ = cv2.warpAffine(occ, M, (Wl, Hl), borderMode=cv2.BORDER_CONSTANT)
                occ = cv2.GaussianBlur(occ, (0, 0), max(0.7, 0.06 * mrp))
                v = v - 0.75 * np.clip(occ, 0, 1)
            gn = np.sqrt(gx * gx + gy * gy) + 1e-6
            face = (-(gx * Ls[0] + gy * Ls[1]) / gn)
            tr = np.clip(face * 1.3, 0, 1) * np.clip((0.85 - h) / 0.5, 0, 1)
            self._htot = np.maximum(self._htot, h)
            return v.astype(np.float32), tr.astype(np.float32)

        V, TR = {}, {}
        self._htot = np.zeros((Hl, Wl), np.float32)
        cm = covs('m')
        cen = {m: to_px(np.array([[tree.masses[m][0], tree.masses[m][1]]]))[0] / q for m in cm}
        rank = {m: float(cen[m] @ Ls) for m in cm}
        for m in cm:
            occ = np.zeros((Hl, Wl), np.float32)
            for o in cm:
                if o != m and rank[o] > rank[m]:
                    occ = np.maximum(occ, cm[o])
            V[('m', m)], TR[('m', m)] = field(cm[m], cen[m], tree.masses[m][2] * k / q, occ)
        cs = covs('sm')
        for m in cs:
            c_ = to_px(np.array([[tree.submasses[m][0], tree.submasses[m][1]]]))[0] / q
            V[('s', m)], TR[('s', m)] = field(cs[m], c_, tree.submasses[m][2] * k / q)
        self._q = q
        return V, TR

    @staticmethod
    def _samp(F, x, y):
        if isinstance(F, tuple):
            A, B = F
            h, w = A.shape
            xi = min(max(int(x + 0.5), 0), w - 1)
            yi = min(max(int(y + 0.5), 0), h - 1)
            return 0.55 * float(A[yi, xi]) + 0.6 * float(B[yi, xi]) + 0.05
        h, w = F.shape
        xi = min(max(int(x + 0.5), 0), w - 1)
        yi = min(max(int(y + 0.5), 0), h - 1)
        return float(F[yi, xi])

    def _band(self, v):
        if v > 0.62:
            return 4
        if v > 0.12:
            return 3
        if v > -0.22:
            return 2
        if v > -0.62:
            return 1
        return 0

    def _color(self, b, tr, rng):
        pal = self.pal
        cols = (pal['deep'], pal['shade'], pal['mid'], pal['lit'], pal['hot'])
        c = cols[b]
        # warm transmitted light where the sun shines through the thin sun-side edges
        if b <= 2 and tr > 0.25:
            c = c * 0.35 + pal['trans'] * 0.65
        elif b <= 2 and tr > 0.1:
            c = c * 0.65 + pal['trans'] * 0.35
        elif b == 3 and tr > 0.3:
            c = c * 0.7 + pal['trans'] * 0.3
        return c * (1.0 + rng.normal(0, 0.012))

    # ------------------------------------------------------------------ one sub-cluster
    def _cluster(self, rgb, a, c, to_px, rng, V, TR):
        k = self.k
        q = self._q
        cx, cy = to_px(np.array([[c['x'], c['y']]]))[0]
        R = c['R'] * k
        Vm_, Vs_ = V[('m', c['m'])], V[('s', c['sm'])]
        Tm_ = TR[('m', c['m'])]
        Vm = (Vm_, Vs_)
        Tm = Tm_
        fr = c.get('fr', 0.0)
        fl = self.fl * rng.uniform(0.9, 1.1)
        if R < 1.6 * fl:
            fl = max(1.2, R * 0.6)
        nb = int(np.clip(round((R / (0.42 * R + 1e-6)) ** 2 * 0.9 * rng.uniform(0.8, 1.2)), 3, 7))
        if c.get('spray'):
            nb = 1
        bunches = []
        for _ in range(nb):
            rad = R * math.sqrt(rng.uniform(0, 1)) * 0.62
            an = rng.uniform(0, 2 * math.pi)
            rb = R * rng.uniform(0.32, 0.5)
            bx, by = cx + rad * math.cos(an), cy + rad * math.sin(an) * 0.8
            if rng.random() < 0.3 * fr:
                continue                              # lacy outer edge: missing bunches -> small holes
            bunches.append((bx, by, rb))
        if c.get('cover'):
            bunches.append((cx, cy, R * 0.8))       # always over the point where the limb goes in
        bunches.sort(key=lambda b: -b[1])            # bottom first: lit tops overlap the undersides
        pal = self.pal
        for bx, by, rb in bunches:
            v0 = self._samp(Vm, bx / q, by / q)
            t0 = self._samp(Tm, bx / q, by / q)
            loose = fr > 0.55 and rng.random() < 0.5 * fr
            # 1) bunch body (largest dab), a step darker than the florets on it
            if rb > 2.2 * fl and not loose:
                b = self._band(v0 - 0.22)
                self._fill(rgb, a, _floret(bx, by + 0.1 * rb, rb * 0.78, rng.uniform(0, 6), 1),
                           self._color(b, t0, rng))
            # 2) florets (medium), painted bottom-up
            nf = int(np.clip((rb / fl) ** 2 * (1.1 if not loose else 0.45) * self.detail, 2, 26))
            rr = rb * np.sqrt(rng.uniform(0.15, 1.0, nf))
            aa = rng.uniform(0, 2 * math.pi, nf)
            fx = bx + rr * np.cos(aa)
            fy = by + rr * np.sin(aa) * 0.85 - 0.1 * rb
            order = np.argsort(-fy)
            for i in order:
                rf = fl * rng.uniform(0.8, 1.2)
                local = (by - fy[i]) / max(rb, 1.0)           # up within the bunch
                v = self._samp(Vm, fx[i] / q, fy[i] / q) + 0.1 * local + rng.normal(0, 0.015)
                b = self._band(v)
                col = self._color(b, self._samp(Tm, fx[i] / q, fy[i] / q), rng)
                self._fill(rgb, a, _floret(fx[i], fy[i], rf, rng.uniform(0, 6), 0 if rf > 2.5 else 2), col)
                if rf > 9 and b in (2, 3) and rng.random() < 0.3:
                    cc = col * 0.75 + pal['core'] * 0.25
                    self._fill(rgb, a, _floret(fx[i], fy[i], rf * 0.28, 0.0, 2), cc)
            # 3) single blossoms around the bunch rim (small) - more on the fringe
            ns = int(rng.integers(1, 3) + round(4 * fr))
            if self.far:
                ns = int(ns * 0.5)
            for _ in range(ns):
                an = rng.uniform(0, 2 * math.pi)
                if fr > 0.3 and rng.random() < 0.6:
                    an = rng.uniform(0.2, 0.8) * math.pi       # hang below (screen y down)
                d = rb * rng.uniform(0.85, 1.35)
                sx, sy = bx + d * math.cos(an), by + d * math.sin(an)
                rf = fl * rng.uniform(0.5, 0.75)
                v = self._samp(Vm, sx / q, sy / q) + rng.normal(0, 0.03)
                b = self._band(v)
                self._fill(rgb, a, _floret(sx, sy, rf, rng.uniform(0, 6), 0 if rf > 2.5 else 2),
                           self._color(b, self._samp(Tm, sx / q, sy / q), rng))

    # ------------------------------------------------------------------ tree
    def paint(self, tree, rng, margin=12):
        k = self.k
        margin = int(margin + 3 * self.fl)
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * margin
        ox = -x0 * k + margin
        oy = y1 * k + margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        V, TR = self._fields(tree, to_px, W, H)
        rgb = np.zeros((H, W, 3), np.uint8)
        a = np.zeros((H, W), np.uint8)
        wm = np.zeros((H, W), np.uint8)
        # 1) all wood behind the blossom (seen only through the sky holes), thick first
        for w in sorted(tree.wood, key=lambda w: -w['d']):
            if w['d'] == 0:
                continue
            pw_ = self.p13 if w['d'] <= 2 else self.p13b
            pw_._wood(rgb, a, w, to_px, rng)
            self._cap(rgb, a, w, to_px, pw_)
        back_a = a
        a = np.zeros_like(back_a)            # coverage of everything painted after the back wood
        # 2) back sub-clusters, then the visible wood prefixes + trunk, then the front sub-clusters
        items = []
        for i, c in enumerate(tree.clusters):
            zc = tree.mz.get(c['m'], 0.0) * 3.0 + (0.8 if c['front'] else -0.8) + 0.1 * c['z']
            items.append((zc, 1, i))
        for i, w in enumerate(tree.wood):
            if w['d'] == 0:
                items.append((-0.3, 0, i))
            elif w['vis'] > 0.02:
                # visible limb prefixes over the blossom (connected to the trunk), thick first ...
                items.append((50.0 - w['d'], 0, i))     # thin first: child bases tuck under the parent
                # ... and a blossom clump painted over the point where each limb goes into the crown
                n = len(w['P'])
                j = max(1, int(math.ceil(w['vis'] * (n - 1))))
                Rm = np.median([c['R'] for c in tree.clusters if c['m'] == w['m']] or [0.25 * tree.s])
                if w['vis'] < 0.999 or w['d'] == tree.maxd:
                    seg = np.linalg.norm(np.diff(w['P'][:j + 1], axis=0), axis=1)
                    jb = max(0, j - int(np.searchsorted(np.cumsum(seg[::-1]), 0.25 * Rm)))
                    cov_ = dict(x=float(w['P'][jb, 0]), y=float(w['P'][jb, 1]), R=float(Rm * rng.uniform(0.85, 1.05)),
                                m=w['m'], sm=w.get('sm', 0), z=0.0, front=True, fr=0.2, cover=True)
                    if w['sm'] not in tree.submasses:
                        cov_['sm'] = min(tree.submasses, key=lambda q_: abs(q_ - w['sm']))
                    items.append((60.0 + rng.uniform(0, 1), 2, cov_))
        items.sort(key=lambda q_: (q_[0], q_[1] if q_[1] < 2 else 2))
        for zc, kind, i in items:
            if kind == 0:
                w = tree.wood[i]
                if w['d'] > 0:
                    n = len(w['P'])
                    jc = max(1, int(math.ceil(w['vis'] * (n - 1))))
                    j = max(2, jc + 1)
                    rr_ = w['r'][:j].copy()
                    nt = max(2, int(j * 0.3))
                    rr_[-nt:] *= np.linspace(1.0, 0.4, nt) ** 0.8     # taper into the blossom, no butt end
                    w = dict(w, P=w['P'][:j], r=rr_)
                self.p13._wood(rgb, a, w, to_px, rng)
                P = to_px(w['P'])
                rr = max(1, int(round(2 * float(np.mean(w['r'])) * k)))
                cv2.polylines(wm, [np.round(P * 16).astype(np.int32)], False, 255, rr, cv2.LINE_8, 4)
            else:
                c = tree.clusters[i] if kind == 1 else i
                if ('s', c['sm']) not in V:
                    continue
                self._cluster(rgb, a, c, to_px, rng, V, TR)
                cx, cy = to_px(np.array([[c['x'], c['y']]]))[0]
                cv2.circle(wm, (int(cx), int(cy)), int(c['R'] * k * 0.9), 0, -1)
        a = np.maximum(a, back_a)
        A = a.astype(np.float32) / 255.0
        C = rgb.astype(np.float32) / SC8
        # crisp warm rim on the sun-facing silhouette
        rp = max(1.0, self.rim_px)
        sx, sy = self.Ls
        M = np.float32([[1, 0, sx * rp], [0, 1, sy * rp]])
        As = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        edge = cv2.GaussianBlur(np.clip(A - As, 0, 1), (0, 0), max(0.5, rp * 0.35))
        blos = 1.0 - (wm > 0).astype(np.float32) * 0.7
        kr = np.clip(edge * 1.3, 0, 1) * blos * 0.6
        C = C + (self.pal['rim'][None, None, :] * A[..., None] - C) * kr[..., None]
        # lost edges on the shadow side: soften the silhouette that faces away from the sun
        sp = max(1.0, self.soft_px)
        M = np.float32([[1, 0, -sx * sp * 2.5], [0, 1, -sy * sp * 2.5]])
        As = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        ks = cv2.GaussianBlur(np.clip(A - As, 0, 1), (0, 0), sp) * blos
        card = np.concatenate([C, A[..., None]], -1).astype(np.float32)
        soft = cv2.GaussianBlur(card, (0, 0), sp)
        ks = np.clip(ks * 1.2, 0, 0.85)[..., None]
        card = card + (soft - card) * ks
        return card, ox, oy
