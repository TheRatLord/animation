"""s05_sakura round 15: painted cherry crowns + gnarled trunks (5 Centimeters per Second study).

Reviewer notes addressed: 'hexagon/polygon flake mosaic' (every dab the same size, individually outlined),
jagged stair-stepped fringe, few sky holes, straight bevelled plank trunks with a clean Y fork, weak transmitted
light, magenta undersides.

Tree15 (model in metres, x right / y up, trunk foot at the origin)
  * gnarled trunk: random-walk centre line with slow bends + one or two elbows, root flare, burls; scaffolds
    leave the trunk at DIFFERENT heights (lower laterals + a top fork) instead of one Y point, and wander
    with real elbows, taper and fork 3 levels deep;
  * blossom sub-clusters strung along the depth-2/3 wood (uneven spacing -> sky holes), grouped per scaffold
    into masses; twig-tip sprays carry small floating clumps (lacy fringe).

Painter15 - value is PAINTED, not per-dab:
  (a) broad value masses: a smooth per-mass form field (blurred coverage lit from the sun, height in the mass,
      cast shadow from the masses nearer the sun), mapped through soft value bands (pink-white top plane,
      light pink, mid pink, cool lavender-blue, blue-violet core);
  (b) every sub-cluster is ONE scalloped shape (a union of 3-8 cauliflower lobes, 20-60 px at 1080p) with
      its own crisp lit crescent on the sun side and a soft, lost underside; weighted by how exposed the
      cluster is, so interior clusters melt into the mass gradient;
  (c) only on the lit silhouette / lit tops: sparse small 5-petal flower dabs and white-pink specks, a few
      floating just outside the edge.
  Then: warm peach transmitted light on thin sun-side edges, a crisp warm rim + soft outer bloom on the
  sun-facing silhouette, pink halation into the sky holes, lost edges on the shadow side.
  Wood: a numba capsule rasteriser gives every wood pixel (across, along) coordinates -> cylinder shading
  in painted bands (blue-violet shadow side, warm mid, 1-2 px gold rim on the sun side, cool sky bounce),
  horizontal lenticel dashes (sakura bark banding) and mottled patches.
"""
import math

import numpy as np
import cv2
from numba import njit


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def _rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _smooth(x, n):
    k = np.ones(n) / n
    return np.convolve(np.pad(x, (n // 2, n - 1 - n // 2), mode='edge'), k, mode='valid')


# ============================================================================ model

class Tree15:
    def __init__(self, rng, scale=1.0, lean=-1.0, spread=1.0, hmul=1.0, big=True, density=1.0, trunk=1.0,
                 nsc=None, cl_r=1.0, gnarl=1.0, fork_lo=0.55):
        s = self.s = float(scale)
        sg = 1.0 if lean >= 0 else -1.0
        self.sg = sg
        self.big = big
        self.density = density
        self.cl_r = cl_r
        self.wood = []
        self.clusters = []
        self.maxd = 3
        self.mz = {}
        # ---- gnarled trunk: random-walk centre line
        ht = rng.uniform(1.0, 1.4) * s * hmul
        r0 = rng.uniform(0.2, 0.26) * s * trunk
        n = 48
        lean_ang = sg * rng.uniform(8, 22)
        a = np.zeros(n)
        # slow bends (two sines) + 1-2 elbows
        f = np.linspace(0, 1, n)
        a += gnarl * rng.uniform(8, 16) * np.sin(2 * math.pi * f * rng.uniform(0.6, 1.3) + rng.uniform(0, 6))
        a += gnarl * rng.uniform(4, 8) * np.sin(2 * math.pi * f * rng.uniform(2.0, 3.2) + rng.uniform(0, 6))
        for _ in range(int(rng.integers(1, 3))):
            i = int(rng.integers(n // 5, n - 4))
            a[i:] += gnarl * rng.choice([-1, 1]) * rng.uniform(8, 16)
        a = _smooth(a, 3)
        a = a - a[0] * 0.3
        P = [np.array([0.0, -0.1 * s])]
        step = (ht + 0.1 * s) / (n - 1)
        for i in range(1, n):
            ang = math.radians(lean_ang * (0.4 + 0.6 * f[i]) + a[i])
            P.append(P[-1] + step * np.array([math.sin(ang), math.cos(ang)]))
        P = np.array(P)
        # radius: taper, root flare, burls, slight bumps
        rt = r0 * (1.0 - 0.32 * f) * (1.0 + 0.5 * np.exp(-np.maximum(P[:, 1] + 0.1 * s, 0) / (0.22 * s)))
        for _ in range(int(rng.integers(1, 3))):
            c = rng.uniform(0.25, 0.85)
            rt = rt * (1.0 + rng.uniform(0.08, 0.16) * np.exp(-((f - c) / 0.05) ** 2))
        rt = rt * (1.0 + 0.04 * np.sin(f * rng.uniform(14, 22) + rng.uniform(0, 6)))
        rt = rt * (1.0 - 0.4 * np.clip((f - 0.88) / 0.12, 0, 1) ** 1.5)     # tucks under the scaffold forks
        self.ht = ht
        # ---- scaffolds: lower laterals leave the trunk at different heights, the rest fork at the top
        nsc_ = nsc or int(rng.choice([3, 4, 4, 5] if big else [3, 3, 4]))
        nlow = int(rng.integers(1, 3)) if nsc_ >= 3 else 0
        base = np.linspace(-62, 60, nsc_) + rng.uniform(-8, 8, nsc_) + sg * 8.0
        order = rng.permutation(nsc_)
        low_idx = set(np.argsort(np.abs(base))[::-1][:nlow].tolist())       # the most sideways ones go low
        w = rng.uniform(0.6, 1.0, nsc_)
        w = w / w.sum()
        itop = n - 1
        self.wood.append(dict(P=P, r=rt, z=0.0, d=0, vis=1.0, m=-1))
        F = P[itop]
        rF = float(rt[itop])
        for j in range(nsc_):
            ang = float(base[j])
            z = (order[j] / max(nsc_ - 1, 1) - 0.5) * 1.6 + rng.uniform(-0.1, 0.1)
            self.mz[j] = z
            if j in low_idx:
                fi = rng.uniform(max(fork_lo, 0.62), 0.85)
                i = int(fi * (n - 1))
                tdir = _unit(P[min(i + 2, n - 1)] - P[max(i - 2, 0)])
                side = 1.0 if ang > 0 else -1.0
                d = _unit(_rot(tdir, -side * rng.uniform(28, 45)))
                p0 = P[i]
                rr = float(rt[i]) * rng.uniform(0.5, 0.62)
                L = rng.uniform(1.6, 2.1) * s * spread
            else:
                tdir = _unit(P[-1] - P[-4])
                d = _unit(0.8 * _rot(np.array([0.0, 1.0]), -ang) + 0.2 * tdir)
                p0 = F + np.array([rng.uniform(-0.3, 0.3) * rF, rng.uniform(-0.6, 0.0) * rF])
                rr = rF * math.sqrt(w[j]) * 1.15
                side = 1.0 if d[0] * sg > 0 else 0.0
                L = rng.uniform(1.4, 1.9) * s * spread * (1.0 + 0.25 * side) * \
                    (0.85 + 0.3 * abs(math.sin(math.radians(ang))))
            self._branch(rng, p0, d, L, rr, 1, z, j, True)
        self._finish(rng)

    def _branch(self, rng, p0, d, L, r0, depth, z, m, parent_vis, sm=None):
        s = self.s
        if depth <= 2 or sm is None:
            self._sm = getattr(self, '_sm', -1) + 1
            sm = self._sm
        nn = max(8, int(30 * L / s))
        P = [np.asarray(p0, np.float64)]
        dd = _unit(d)
        step = L / (nn - 1)
        g = (0.004 if depth == 1 else 0.009 * depth) * rng.uniform(0.5, 1.3)
        curl = rng.normal(0, 1.4)
        nk = int(rng.integers(2, 5)) if depth <= 2 else int(rng.integers(1, 3))
        kinks = set(rng.choice(np.arange(2, nn - 1), size=min(nn - 3, nk), replace=False).tolist())
        for i in range(1, nn):
            a = rng.normal(0, 2.2) + curl
            if i in kinks:
                a += rng.choice([-1, 1]) * rng.uniform(8, 20)      # elbows: sakura limbs zig-zag
            dd = _rot(dd, a)
            u = i / (nn - 1)
            dd = _unit(dd + np.array([0.0, -g * u * 1.5]) + (np.array([0.0, 0.025]) if u < 0.4 else 0))
            if dd[1] < -0.35:
                dd = _unit([dd[0], -0.35])
            P.append(P[-1] + dd * step)
        P = np.array(P)
        tt = np.linspace(0, 1, nn)
        rmin = 0.005 * s
        r1 = max(rmin, r0 * (0.35 if depth < self.maxd else 0.2))
        r = r0 + (r1 - r0) * tt ** 0.8
        r = r * (1.0 + 0.06 * np.sin(tt * rng.uniform(10, 18) + rng.uniform(0, 6)))
        if depth == 1:
            vis = rng.uniform(0.5, 0.85)
        elif parent_vis and depth == 2:
            vis = rng.uniform(0.25, 0.7) if rng.random() < 0.8 else 0.0
        else:
            vis = 0.0
        self.wood.append(dict(P=P, r=r, z=z, d=depth, vis=vis, m=m, sm=sm))
        if depth < self.maxd:
            nl = {1: int(rng.integers(3, 6)), 2: int(rng.integers(2, 4))}.get(depth, 1)
            ts = np.sort(rng.uniform(0.18, 0.82, nl))
            side = rng.choice([-1, 1])
            for tl in ts:
                i = int(tl * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                cd = _rot(td, side * rng.uniform(28, 55))
                side = -side
                if cd[1] < -0.45:
                    cd = _unit([cd[0], -0.45])
                Lc = L * rng.uniform(0.42, 0.62) * (1.0 - 0.35 * tl)
                if Lc < 0.15 * s:
                    continue
                self._branch(rng, P[i], cd, Lc, max(rmin, r[i] * rng.uniform(0.5, 0.68)), depth + 1,
                             z + rng.uniform(-0.3, 0.3), m, vis > tl + 0.45 * s / L, sm)
            td = _unit(P[-1] - P[-3])
            share = rng.uniform(0.35, 0.65)
            for sgn, sh in ((-1, share), (1, 1 - share)):
                cd = _rot(td, sgn * rng.uniform(15, 34))
                self._branch(rng, P[-1], cd, L * rng.uniform(0.45, 0.62), max(rmin, r[-1] * math.sqrt(sh) * 1.05),
                             depth + 1, z + rng.uniform(-0.3, 0.3), m, vis > 0.97, sm)
        # blossom sub-clusters along the outer part of depth >= 2 wood
        if depth >= 2 or not self.big:
            Rb = rng.uniform(0.28, 0.4) * s * self.cl_r
            t = (0.25 if depth >= 3 else 0.45) + rng.uniform(0, 0.15)
            while t <= 1.0:
                i = int(t * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                nrm = np.array([-td[1], td[0]])
                if nrm[1] < 0:
                    nrm = -nrm
                R = Rb * rng.uniform(0.6, 1.25)
                c = P[i] + nrm * R * rng.uniform(0.05, 0.5) + rng.normal(0, 0.12, 2) * R
                if rng.random() < 0.93 * min(1.0, self.density):
                    self.clusters.append(dict(x=float(c[0]), y=float(c[1]), R=float(R), m=m, sm=sm,
                                              z=z + rng.uniform(-0.25, 0.25), front=bool(rng.random() < 0.3)))
                t += (R / L) * rng.uniform(0.75, 1.5) / max(self.density, 0.4)
            if depth == self.maxd:
                tip = P[-1]
                td = _unit(P[-1] - P[-3])
                for _ in range(int(rng.integers(1, 4))):
                    dd = _unit(td + np.array([rng.normal(0, 0.5), -rng.uniform(0.1, 0.8)]))
                    Lt = rng.uniform(0.03, 0.08) * s
                    q = np.array([tip + dd * Lt * u_ + np.array([0, -0.15 * Lt * u_ * u_]) for u_ in np.linspace(0, 1, 6)])
                    self.wood.append(dict(P=q, r=np.full(6, rmin * 0.8), z=z, d=depth + 1, vis=0.0, m=m))
                    self.clusters.append(dict(x=float(q[-1, 0]), y=float(q[-1, 1]), R=float(Rb * rng.uniform(0.16, 0.3)),
                                              sm=sm, m=m, z=z + 0.1, front=True, spray=True))

    def _finish(self, rng):
        self.clusters = [c for c in self.clusters if c['y'] > 0.45 * self.ht]
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
            xs += [c['x'] - 1.6 * c['R'], c['x'] + 1.6 * c['R']]
            ys += [c['y'] - 1.8 * c['R'], c['y'] + 1.6 * c['R']]
        return min(xs), max(xs), min(ys), max(ys)


# ============================================================================ wood raster

@njit(cache=True)
def _raster_wood(H, W, segs, cov, S, V, RR, PRI, NX, NY, RID, DEP):
    n = segs.shape[0]
    for i in range(n):
        x0, y0, x1, y1 = segs[i, 0], segs[i, 1], segs[i, 2], segs[i, 3]
        r0, r1, v0, v1 = segs[i, 4], segs[i, 5], segs[i, 6], segs[i, 7]
        pri = segs[i, 8]
        rid = segs[i, 9]
        n0x, n0y, n1x, n1y = segs[i, 10], segs[i, 11], segs[i, 12], segs[i, 13]
        rm = max(r0, r1) + 1.5
        xa = max(0, int(math.floor(min(x0, x1) - rm)))
        xb = min(W, int(math.ceil(max(x0, x1) + rm)) + 1)
        ya = max(0, int(math.floor(min(y0, y1) - rm)))
        yb = min(H, int(math.ceil(max(y0, y1) + rm)) + 1)
        dx = x1 - x0
        dy = y1 - y0
        L2 = dx * dx + dy * dy + 1e-9
        Ln = math.sqrt(L2)
        px_ = -dy / Ln
        py_ = dx / Ln
        for y in range(ya, yb):
            for x in range(xa, xb):
                qx = x + 0.5
                qy = y + 0.5
                t = ((qx - x0) * dx + (qy - y0) * dy) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                cx = x0 + t * dx
                cy = y0 + t * dy
                ex = qx - cx
                ey = qy - cy
                d = math.sqrt(ex * ex + ey * ey)
                rr = r0 + (r1 - r0) * t
                c = rr - d + 0.5
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                side = ex * px_ + ey * py_
                s = side / max(rr, 0.3)
                if s > 1.0:
                    s = 1.0
                elif s < -1.0:
                    s = -1.0
                better = False
                if cov[y, x] <= 0.0:
                    better = True
                elif pri < PRI[y, x]:
                    better = True
                elif pri == PRI[y, x] and (rr - d) > DEP[y, x]:
                    better = True
                if better:
                    DEP[y, x] = rr - d
                    S[y, x] = s
                    V[y, x] = v0 + (v1 - v0) * t
                    RR[y, x] = rr
                    PRI[y, x] = pri
                    nx_ = n0x + (n1x - n0x) * t
                    ny_ = n0y + (n1y - n0y) * t
                    nl_ = math.sqrt(nx_ * nx_ + ny_ * ny_) + 1e-9
                    NX[y, x] = nx_ / nl_
                    NY[y, x] = ny_ / nl_
                    RID[y, x] = rid
                if c > cov[y, x]:
                    cov[y, x] = c


def _hash(a):
    a = (a.astype(np.int64) * 374761393 + 668265263) & 0xFFFFFFFF
    a = ((a ^ (a >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((a ^ (a >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


# ============================================================================ painter

PAL15 = dict(
    hot=(1.0, 0.9, 0.92), lit=(0.98, 0.74, 0.84), mid=(0.93, 0.56, 0.73), shade=(0.8, 0.58, 0.8),
    deep=(0.56, 0.5, 0.8), trans=(1.06, 0.8, 0.68), rim=(1.2, 1.06, 0.94), glow=(1.0, 0.86, 0.9),
    spark=(1.08, 1.0, 1.0),
    # wood: blue-violet shadow side, warm mid, gold rim, cool sky bounce, lenticels
    w_sh=(0.14, 0.1, 0.17), w_mid=(0.33, 0.23, 0.24), w_lit=(0.5, 0.36, 0.32), w_rim=(1.05, 0.76, 0.52),
    w_cool=(0.3, 0.3, 0.46), w_len=(0.42, 0.36, 0.42), w_len_d=(0.07, 0.05, 0.09),
    # wood behind the blossom (seen through the holes): in the crown's shade, cooler
    b_sh=(0.34, 0.28, 0.4), b_mid=(0.44, 0.34, 0.44), b_rim=(0.85, 0.66, 0.64),
)

PAL15_FAR = dict(hot=(0.97, 0.88, 0.93), lit=(0.95, 0.78, 0.87), mid=(0.9, 0.66, 0.8), shade=(0.78, 0.66, 0.86),
                 deep=(0.66, 0.6, 0.84), rim=(1.04, 0.96, 0.95),
                 w_sh=(0.24, 0.2, 0.3), w_mid=(0.38, 0.3, 0.36), w_lit=(0.5, 0.4, 0.42), w_rim=(0.95, 0.78, 0.66),
                 w_cool=(0.36, 0.36, 0.5), b_sh=(0.34, 0.28, 0.4), b_mid=(0.42, 0.34, 0.44))

BANDS = (-0.85, -0.4, 0.04, 0.5)


def _sstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _petal_poly(n=30):
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c = np.abs(np.cos(2.5 * th))
    rr = 0.45 + 0.55 * c ** 0.6
    notch = 1.0 - 0.18 * np.exp(-(((th * 5 / (2 * math.pi)) % 1.0) - 0.5) ** 2 / 0.003)
    rr = rr * notch
    return np.stack([rr * np.cos(th), rr * np.sin(th)], 1)


_PETAL = _petal_poly()


def _lobe_poly(rng, r, nsc, sq=0.85):
    """scalloped cauliflower lobe outline: nsc bumps with inward cusps, a little irregular"""
    n = max(24, nsc * 6)
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    ph = rng.uniform(0, 6.3)
    b = np.abs(np.sin(nsc * 0.5 * th + ph)) ** 0.55
    amp = rng.uniform(0.1, 0.17)
    rr = r * (1.0 - amp + amp * b) * (1.0 + 0.07 * np.sin(2 * th + rng.uniform(0, 6)) + 0.05 * np.sin(3 * th + rng.uniform(0, 6)))
    return np.stack([rr * np.cos(th), rr * np.sin(th) * sq], 1)


class Painter15:
    def __init__(self, k, sun_dir=(-0.7, -0.7), pal=None, px=2.0, detail=1.0, far=False, glow=1.0, lace=1.0):
        self.k = float(k)
        L = _unit(sun_dir)
        self.Ls = L
        Lt = _unit(L + np.array([0.0, -1.1]))
        self.L3 = _unit(np.array([Lt[0] * 0.8, Lt[1] * 0.8, 0.55]))
        p = dict(PAL15)
        if pal:
            p.update(pal)
        self.pal = {kk: np.array(v, np.float32) for kk, v in p.items()}
        self.px = float(px)          # one 1080p pixel in card px
        self.detail = detail
        self.far = far
        self.glow = glow
        self.lace = lace

    # ------------------------------------------------------------------ mass value fields (quarter res)
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
            pos = (cen[1] - ys) / mrp
            v = 1.4 * (dif - 0.55) + 0.5 * pos
            if occ is not None and occ.any():
                d = 0.22 * mrp
                M = np.float32([[1, 0, -Ls[0] * d], [0, 1, -Ls[1] * d]])
                occ = cv2.warpAffine(occ, M, (Wl, Hl), borderMode=cv2.BORDER_CONSTANT)
                occ = cv2.GaussianBlur(occ, (0, 0), max(0.7, 0.06 * mrp))
                v = v - 0.7 * np.clip(occ, 0, 1)
            gn = np.sqrt(gx * gx + gy * gy) + 1e-6
            face = (-(gx * Ls[0] + gy * Ls[1]) / gn)
            tr = np.clip(face * 1.3, 0, 1) * np.clip((0.85 - h) / 0.5, 0, 1)
            return v.astype(np.float32), tr.astype(np.float32)

        V, TR = {}, {}
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

    def _nz_roi(self, X0, Y0, w, h):
        N = self._nz
        Hn, Wn = N.shape
        xa, ya = X0 + self._nzo, Y0 + self._nzo
        if xa < 0 or ya < 0 or xa + w > Wn or ya + h > Hn:
            M = np.float32([[1, 0, xa], [0, 1, ya]])
            return cv2.warpAffine(N, M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_WRAP)
        return N[ya:ya + h, xa:xa + w]

    def _make_noise(self, W, H, rng):
        """organic blob field: overlapping round dabs (flower-head sized) of random value, lightly blurred -
        no grid, no axis-aligned steps"""
        px = self.px
        pad = int(60 * px)
        Hn, Wn = H + 2 * pad, W + 2 * pad
        N = np.full((Hn, Wn), 0.5, np.float32)
        r0 = 2.6 * px
        n = int(Hn * Wn / (r0 * r0) * 0.55)
        xs = rng.uniform(0, Wn, n)
        ys = rng.uniform(0, Hn, n)
        rs = r0 * rng.uniform(0.6, 1.5, n)
        vs = rng.uniform(0, 1, n)
        for x, y, r, v in zip((xs * 4).astype(np.int32), (ys * 4).astype(np.int32), (rs * 4).astype(np.int32), vs):
            cv2.circle(N, (int(x), int(y)), int(r), float(v), -1, cv2.LINE_AA, 2)
        N = cv2.GaussianBlur(N, (0, 0), 0.5 * px)
        N = (N - N.mean()) / (N.std() * 3.5) + 0.5
        self._nz = np.clip(N, 0, 1).astype(np.float32)
        self._nzo = pad

    def _up(self, F, X0, Y0, w, h):
        q = self._q
        M = np.float32([[1.0 / q, 0, (X0 + 0.5) / q - 0.5], [0, 1.0 / q, (Y0 + 0.5) / q - 0.5]])
        return cv2.warpAffine(F, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)

    # ------------------------------------------------------------------ colour
    def cmap(self, v, tr):
        pal = self.pal
        cols = [pal['deep'], pal['shade'], pal['mid'], pal['lit'], pal['hot']]
        wd = 0.035
        out = np.broadcast_to(cols[0], v.shape + (3,)).astype(np.float32).copy()
        for i, lv in enumerate(BANDS):
            st = _sstep((v - lv) / wd + 0.5)[..., None]
            out += st * (cols[i + 1] - cols[i])
        # warm transmitted light in the shadow / mid bands of thin sun-side edges
        low = np.clip((0.45 - v) / 0.5, 0, 1)
        kt = (np.clip(tr, 0, 1) * (0.35 + 0.5 * low))[..., None]
        out = out + (pal['trans'] - out) * kt * 0.8
        return out

    # ------------------------------------------------------------------ wood layer
    def _wood_layer(self, H, W, woods, to_px, back):
        pal = self.pal
        segs = []
        for wi, w in enumerate(woods):
            P = to_px(w['P'])
            r = np.asarray(w['r'], np.float64) * self.k
            v = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(w['P'], axis=0), axis=1))])
            # smooth (3-tap, ends pinned) + densify to ~3 px segments: no scalloped capsule silhouettes
            if len(P) > 3:
                for _ in range(2):
                    P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
                    r[1:-1] = 0.25 * r[:-2] + 0.5 * r[1:-1] + 0.25 * r[2:]
            sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            nd = int(max(len(P), sl[-1] / (1.5 * self.px)))
            if nd > len(P):
                u = np.linspace(0, sl[-1], nd)
                P = np.stack([np.interp(u, sl, P[:, 0]), np.interp(u, sl, P[:, 1])], 1)
                r = np.interp(u, sl, r)
                v = np.interp(u, sl, v)
            T = np.zeros_like(P)
            T[1:-1] = P[2:] - P[:-2]
            T[0] = P[1] - P[0]
            T[-1] = P[-1] - P[-2]
            T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
            Nn = np.stack([-T[:, 1], T[:, 0]], 1)
            for i in range(len(P) - 1):
                segs.append((P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1], max(r[i], 0.45 * self.px),
                             max(r[i + 1], 0.45 * self.px), v[i], v[i + 1], float(w['d']), float(wi),
                             Nn[i, 0], Nn[i, 1], Nn[i + 1, 0], Nn[i + 1, 1]))
        cov = np.zeros((H, W), np.float32)
        if not segs:
            return np.zeros((H, W, 3), np.float32), cov
        S = np.zeros((H, W), np.float32)
        Vv = np.zeros((H, W), np.float32)
        RR = np.zeros((H, W), np.float32)
        PRI = np.full((H, W), 99.0, np.float32)
        NX = np.zeros((H, W), np.float32)
        NY = np.zeros((H, W), np.float32)
        RID = np.zeros((H, W), np.float32)
        _raster_wood(H, W, np.array(segs, np.float64), cov, S, Vv, RR, PRI, NX, NY, RID, np.full((H, W), -1e9, np.float32))
        Ls = self.Ls
        facing = S * (NX * Ls[0] + NY * Ls[1])              # +1 on the sun-side edge
        a_s = np.abs(S)
        if back:
            sh, mid, rim = pal['b_sh'], pal['b_mid'], pal['b_rim']
        else:
            sh, mid, rim = pal['w_sh'], pal['w_mid'], pal['w_rim']
        col = np.broadcast_to(sh, (H, W, 3)).astype(np.float32).copy()
        # painted mid band on the sun side (crisp step, not a gradient bevel)
        m1 = _sstep((facing - 0.32) / 0.12 + 0.5)
        col += (mid - sh) * m1[..., None]
        if not back:
            m2 = _sstep((facing - 0.62) / 0.1 + 0.5) * np.clip((RR - 2.5 * self.px) / (3 * self.px), 0, 1)
            col += (pal['w_lit'] - mid) * (m2 * 0.7)[..., None]
        # thin gold rim on the sun side (1-2 px at 1080p), strongest on thick wood
        rw = np.clip((1.4 * self.px) / np.maximum(RR, 0.5), 0.05, 0.7)
        rimk = _sstep((a_s - (1.0 - rw)) / (0.5 * rw) + 0.5) * _sstep((facing - 0.35) / 0.25 + 0.5)
        col += (rim - col) * (rimk * (0.9 if not back else 0.55))[..., None]
        # cool sky bounce on the shadow side edge
        cb = _sstep((-facing - 0.7) / 0.15 + 0.5) * _sstep((a_s - 0.75) / 0.2 + 0.5) * np.clip((RR - 2 * self.px) / (3 * self.px), 0, 1)
        col += (pal['w_cool'] - col) * (cb * 0.45)[..., None]
        # bark: horizontal lenticel dashes + mottled patches (only on wood thick enough to read)
        thick = np.clip((RR - 2.5 * self.px) / (4 * self.px), 0, 1)
        if thick.max() > 0:
            rm = RR / self.k                                   # radius in metres
            v_eff = Vv - 0.08 * rm * np.sqrt(np.clip(1 - S * S, 0, 1))       # bands curve round the cylinder
            sp = 0.045 * (0.6 + rm / 0.25)
            j = np.floor(v_eff / sp)
            fr = v_eff / sp - j
            hid = (j.astype(np.int64) * 131 + RID.astype(np.int64) * 7919)
            h1, h2, h3 = _hash(hid), _hash(hid + 17), _hash(hid + 91)
            a0 = -1.0 + 0.8 * h1
            a1 = a0 + 0.9 + 0.8 * h2
            thk = np.clip(1.0 * self.px / (sp * self.k), 0.04, 0.3)
            dash = (fr < thk) & (S > a0) & (S < a1) & (h3 < 0.6)
            dk = dash.astype(np.float32) * thick
            lit_side = np.clip(facing * 2, 0, 1)
            lc = pal['w_len'] * (1 - lit_side[..., None]) + (pal['w_lit'] * 1.35) * lit_side[..., None]
            col += (lc - col) * (dk * 0.75)[..., None]
            # darker ring just below each dash (painted lenticel shadow)
            dash2 = (fr > thk) & (fr < 2 * thk) & (S > a0) & (S < a1) & (h3 < 0.6)
            col += (pal['w_len_d'] - col) * (dash2.astype(np.float32) * thick * 0.4)[..., None]
            # mottled patches (lichen / old bark) along the trunk
            mot = np.sin(Vv / (0.11 * (0.6 + rm / 0.25)) + 2.3 * S + RID) * np.sin(Vv / 0.07 - 1.7 * S + 0.5 * RID)
            col *= (1.0 + 0.1 * np.clip(mot, -1, 1) * thick)[..., None]
        return col.astype(np.float32), cov

    # ------------------------------------------------------------------ one cluster
    def _cluster(self, c, to_px, rng, V, TR, acc):
        k = self.k
        cx, cy = to_px(np.array([[c['x'], c['y']]]))[0]
        R = c['R'] * k
        fr = c.get('fr', 0.0)
        px = self.px
        spray = c.get('spray', False)
        # ---- lobes (20-60 px at 1080p), a union of scalloped cauliflower heads
        lobes = []
        if spray or R < 5 * px:
            nl = 1 if spray else 2
        else:
            nl = int(np.clip(round(2 + (R / (14 * px)) * rng.uniform(0.8, 1.3)), 3, 8))
        for i in range(nl):
            if i == 0:
                bx, by = cx + rng.normal(0, 0.1) * R, cy - 0.15 * R
                rl = R * rng.uniform(0.45, 0.6)
            else:
                rad = R * math.sqrt(rng.uniform(0.05, 1.0)) * 0.62
                an = rng.uniform(0, 2 * math.pi)
                bx, by = cx + rad * math.cos(an), cy + rad * math.sin(an) * 0.7 + 0.08 * R
                rl = R * rng.uniform(0.3, 0.5)
            if spray:
                rl = R
            lobes.append((bx, by, rl))
        # lacy fringe: detached small flower clumps around exposed clusters
        nd = int(round(self.lace * (1 + 5 * fr) * rng.uniform(0.3, 1.0))) if not spray else 0
        for _ in range(nd):
            an = rng.uniform(0, 2 * math.pi)
            if rng.random() < 0.5:
                an = rng.uniform(0.15, 0.85) * math.pi
            d = R * rng.uniform(0.85, 1.35)
            lobes.append((cx + d * math.cos(an), cy + d * math.sin(an) * 0.85, R * rng.uniform(0.1, 0.2)))
        Rm = max(l[2] for l in lobes)
        ext = max(math.hypot(l[0] - cx, l[1] - cy) + l[2] for l in lobes) + 4
        X0, Y0 = int(math.floor(cx - ext)), int(math.floor(cy - ext))
        X1, Y1 = int(math.ceil(cx + ext)), int(math.ceil(cy + ext))
        Hc, Wc = acc['A'].shape
        xa, ya, xb, yb = max(0, X0), max(0, Y0), min(Wc, X1), min(Hc, Y1)
        if xb <= xa or yb <= ya:
            return
        w, h = X1 - X0, Y1 - Y0
        m8 = np.zeros((h, w), np.uint8)
        for (bx, by, rl) in lobes:
            nsc = int(np.clip(rl / (3.2 * px), 5, 14))
            poly = _lobe_poly(rng, rl, nsc) + (bx - X0, by - Y0)
            cv2.fillPoly(m8, [np.round(poly * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        m = m8.astype(np.float32) / 255.0
        # blossom-scale breakup: the edge crumbles into rounded flower dabs (blob noise, not pixel noise)
        nzr = self._nz_roi(X0, Y0, w, h)
        mb = cv2.GaussianBlur(m, (0, 0), 2.2 * px)
        if Rm > 7 * px:
            m = np.clip((mb - 0.5 + 0.5 * (nzr - 0.5)) * 6.0 + 0.5, 0, 1) * np.clip(mb * 3, 0, 1)
            m = np.maximum(m, np.clip((mb - 0.75) * 8, 0, 1))
        Ls = self.Ls
        # crisp lit crescent on the sun side (follows the scalloped edge)
        d1 = max(1.5 * px, 0.26 * Rm)
        Ms = np.float32([[1, 0, Ls[0] * d1], [0, 1, Ls[1] * d1]])
        m_s = cv2.warpAffine(m, Ms, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        lit = np.clip(m - m_s, 0, 1)
        lit = cv2.GaussianBlur(lit, (0, 0), max(0.6, 0.12 * d1))
        # soft, lost underside
        d2 = 0.55 * R
        Mu = np.float32([[1, 0, -Ls[0] * d2], [0, 1, -Ls[1] * d2 * 0.6 + d2 * 0.4]])
        m_u = cv2.warpAffine(m, Mu, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        und = cv2.GaussianBlur(np.clip(m - m_u, 0, 1), (0, 0), max(1.0, 0.3 * R))
        yy = (np.arange(h, dtype=np.float32)[:, None] + Y0 - cy) / max(R, 1.0)
        pos = np.clip(-yy, -1, 1)
        Vm = self._up(V[('m', c['m'])], X0, Y0, w, h)
        Vs = self._up(V[('s', c['sm'])], X0, Y0, w, h)
        Tm = self._up(TR[('m', c['m'])], X0, Y0, w, h)
        base = 0.55 * Vm + 0.55 * Vs + 0.06
        # exposed clusters (fringe, lit side) keep their own form; buried ones melt into the mass gradient
        wl = 0.12 + 0.6 * max(fr, float(np.clip(base[h // 2, w // 2] + 0.2, 0, 1)))
        if spray:
            wl = 1.0
        v = base + wl * (0.6 * lit - 0.22 * und + 0.1 * pos) + 0.22 * (nzr - 0.5) + self._up(self._wash, X0, Y0, w, h)
        # thin detached fringe clumps glow (transmitted light) more than the body
        tr = Tm * (0.7 + 0.6 * fr)
        col = self.cmap(v, tr)
        # composite into the ROI
        sx0, sy0 = xa - X0, ya - Y0
        sl = (slice(ya, yb), slice(xa, xb))
        mm = m[sy0:sy0 + (yb - ya), sx0:sx0 + (xb - xa)]
        cc = col[sy0:sy0 + (yb - ya), sx0:sx0 + (xb - xa)]
        vv = v[sy0:sy0 + (yb - ya), sx0:sx0 + (xb - xa)]
        li = lit[sy0:sy0 + (yb - ya), sx0:sx0 + (xb - xa)]
        A = acc['A'][sl]
        acc['C'][sl] = acc['C'][sl] * (1 - mm[..., None]) + cc * mm[..., None]
        acc['V'][sl] = acc['V'][sl] * (1 - mm) + vv * mm
        acc['LIT'][sl] = acc['LIT'][sl] * (1 - mm) + li * wl * mm
        acc['A'][sl] = A + mm * (1 - A)
        acc['B'][sl] = acc['B'][sl] * (1 - mm) + mm              # blossom (vs wood) coverage on top

    # ------------------------------------------------------------------ tree
    def paint(self, tree, rng, margin=12):
        k = self.k
        px = self.px
        margin = int(margin + 14 * px)
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * margin
        ox = -x0 * k + margin
        oy = y1 * k + margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        V, TR = self._fields(tree, to_px, W, H)
        self._make_noise(W, H, rng)
        # low-frequency painted wash variation (quarter res, 20-60 px blotches)
        q = self._q
        g = rng.standard_normal((H // q + 2, W // q + 2)).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), max(1.0, 7 * px / q))
        self._wash = (g / (g.std() + 1e-6) * 0.09).astype(np.float32)
        acc = dict(C=np.zeros((H, W, 3), np.float32), A=np.zeros((H, W), np.float32),
                   V=np.zeros((H, W), np.float32), LIT=np.zeros((H, W), np.float32), B=np.zeros((H, W), np.float32))

        def over_layer(col, cov):
            acc['C'] = acc['C'] * (1 - cov[..., None]) + col * cov[..., None]
            acc['A'] = acc['A'] + cov * (1 - acc['A'])
            acc['B'] = acc['B'] * (1 - cov)
            acc['LIT'] = acc['LIT'] * (1 - cov)

        # 1) all wood behind the blossom (seen through the sky holes)
        back = [w for w in tree.wood if w['d'] > 0]
        col, cov = self._wood_layer(H, W, back, to_px, True)
        over_layer(col, cov)
        thin = [w for w in back if w['d'] >= 2]
        back_col, back_cov = self._wood_layer(H, W, thin, to_px, True)
        # 2) back clusters, 3) trunk + visible limb prefixes, 4) front clusters + cover clumps
        cl = tree.clusters
        order = sorted(range(len(cl)), key=lambda i: (tree.mz.get(cl[i]['m'], 0.0) * 3.0
                                                      + (0.8 if cl[i]['front'] else -0.8) + 0.1 * cl[i]['z'],
                                                      -cl[i]['y']))
        backc = [i for i in order if not cl[i]['front']]
        frontc = [i for i in order if cl[i]['front']]
        for i in backc:
            if ('s', cl[i]['sm']) in V:
                self._cluster(cl[i], to_px, rng, V, TR, acc)
        fw, covers = [], []
        for w in tree.wood:
            if w['d'] == 0:
                fw.append(w)
            elif w['vis'] > 0.02:
                n = len(w['P'])
                j = max(2, int(math.ceil(w['vis'] * (n - 1))) + 1)
                # end the visible prefix where the limb first enters a blossom cluster of its mass
                mc = [c for c in cl if c['m'] == w['m']]
                if mc:
                    CX = np.array([c['x'] for c in mc])
                    CY = np.array([c['y'] for c in mc])
                    CR = np.array([c['R'] for c in mc])
                    dn = np.min(np.hypot(w['P'][:, 0:1] - CX[None], w['P'][:, 1:2] - CY[None]) / CR[None], axis=1)
                    inside = np.nonzero(dn < 0.35)[0]
                    if len(inside):
                        j = max(3, min(j, int(inside[0]) + 2))
                j = min(j, n)
                if w['d'] >= 2 and (j < 7 or j < 0.3 * n):
                    continue                     # a stub poking out of its parent: leave it inside the crown
                rr_ = np.array(w['r'][:j], np.float64).copy()
                nt = max(2, int(j * 0.6))
                rr_[-nt:] *= np.linspace(1.0, 0.2, nt) ** 0.9
                fw.append(dict(w, P=w['P'][:j], r=rr_))
                Rm = np.median([c['R'] for c in mc] or [0.25 * tree.s])
                jb = max(0, j - 1)
                cov_ = dict(x=float(w['P'][jb, 0]), y=float(w['P'][jb, 1] + 0.1 * Rm), R=float(Rm * rng.uniform(1.3, 1.6)),
                            m=w['m'], sm=w.get('sm', 0), z=0.0, front=True, fr=0.2)
                # smaller clumps overlapping the last stretch of the limb: it goes INTO the blossom
                for q_ in range(int(rng.integers(3, 6))):
                    jj = max(0, j - 1 - int(rng.uniform(0.05, 0.45) * j))
                    td_ = _unit(w['P'][min(jj + 1, n - 1)] - w['P'][max(jj - 1, 0)])
                    nr_ = np.array([-td_[1], td_[0]]) * rng.choice([-1, 1])
                    Rq = Rm * rng.uniform(0.45, 0.8)
                    pq = w['P'][jj] + nr_ * Rq * rng.uniform(0.2, 0.8)
                    cq = dict(x=float(pq[0]), y=float(pq[1]), R=float(Rq), m=w['m'], sm=w.get('sm', 0), z=0.0,
                              front=True, fr=0.3)
                    if cq['sm'] not in tree.submasses:
                        cq['sm'] = min(tree.submasses, key=lambda q2: abs(q2 - cq['sm']))
                    if cq['m'] in tree.masses:
                        covers.append(cq)
                if cov_['sm'] not in tree.submasses:
                    cov_['sm'] = min(tree.submasses, key=lambda q_: abs(q_ - cov_['sm']))
                if cov_['m'] in tree.masses:
                    covers.append(cov_)
        cw = []
        for w in tree.wood:
            if w["d"] == 2 and w["vis"] <= 0.02 and len(w["P"]) > 6 and rng.random() < 0.35:
                n = len(w['P'])
                a_ = int(rng.uniform(0.3, 0.5) * n)
                b_ = min(n, a_ + max(4, int(rng.uniform(0.5, 0.8) * n)))
                rr_ = np.array(w['r'][a_:b_], np.float64) * 0.6
                nt = max(2, (b_ - a_) // 4)
                rr_[:nt] *= np.linspace(0.5, 1.0, nt)
                rr_[-nt:] *= np.linspace(1.0, 0.4, nt)
                cw.append(dict(w, P=w['P'][a_:b_], r=rr_))
        if cw:
            col, cov = self._wood_layer(H, W, cw, to_px, True)
            bl = cv2.GaussianBlur(acc['B'], (0, 0), 3 * self.px)
            over_layer(col, cov * 0.9 * np.clip((bl - 0.6) / 0.3, 0, 1))
        col, cov = self._wood_layer(H, W, fw, to_px, False)
        over_layer(col, cov)
        wood_front = cov
        for i in frontc:
            if ('s', cl[i]['sm']) in V:
                self._cluster(cl[i], to_px, rng, V, TR, acc)
        for c in covers:
            if ('s', c['sm']) in V:
                self._cluster(c, to_px, rng, V, TR, acc)
        # bare back wood far outside the blossom reads as scratches in the sky: keep only twig ends that stay
        # close to a blossom clump (a short bare stretch at most), fading out
        far_ = cv2.distanceTransform((acc['B'] < 0.5).astype(np.uint8), cv2.DIST_L2, 5)
        keepw = np.clip((10.0 * self.px - far_) / (4.0 * self.px), 0, 1)
        bare = np.clip(acc['A'] - acc['B'] - wood_front, 0, 1)
        kill = bare * (1 - keepw)
        acc['C'] *= (1 - kill)[..., None]
        acc['A'] = acc['A'] * (1 - kill)
        self._holes(acc, back_col, back_cov, wood_front, rng)
        C, A, Bm = acc['C'], acc['A'], acc['B']
        pal = self.pal
        Ls = self.Ls
        # ---- blossom-only post: sun-facing rim, transmitted rim bloom, lost shadow-side edges
        rp = max(1.0, 1.6 * px)
        M = np.float32([[1, 0, Ls[0] * rp], [0, 1, Ls[1] * rp]])
        As = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        edge = cv2.GaussianBlur(np.clip(A - As, 0, 1), (0, 0), max(0.5, rp * 0.3))
        kr = np.clip(edge * 1.4, 0, 1) * Bm * 0.75
        C = C + (pal['rim'] - C) * kr[..., None]
        # warm band just inside the sun-facing silhouette (translucent petals, 3-6 px)
        rp2 = 5.0 * px
        M2 = np.float32([[1, 0, Ls[0] * rp2], [0, 1, Ls[1] * rp2]])
        As2 = cv2.warpAffine(A, M2, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        band = cv2.GaussianBlur(np.clip(A - As2, 0, 1), (0, 0), 1.5 * px) * Bm
        C = C + (pal['trans'] * 1.05 - C) * (np.clip(band, 0, 1) * 0.35)[..., None]
        # ---- flower dabs and specks on the lit silhouette / lit tops (sparse)
        self._florets(C, A, acc['V'], acc['LIT'], edge, Bm, rng)
        A = acc['A']
        # ---- lost edges on the shadow side of the silhouette
        sp = max(1.0, 1.3 * px)
        M = np.float32([[1, 0, -Ls[0] * sp * 2.5], [0, 1, -Ls[1] * sp * 2.5]])
        As = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
        ks = cv2.GaussianBlur(np.clip(A - As, 0, 1), (0, 0), sp) * cv2.GaussianBlur(Bm, (0, 0), 1.0)
        card = np.concatenate([C * A[..., None], A[..., None]], -1).astype(np.float32)
        soft = cv2.GaussianBlur(card, (0, 0), sp)
        ks = np.clip(ks * 1.2, 0, 0.8)[..., None]
        card = card + (soft - card) * ks
        # ---- halation: a soft pink-white glow bleeding from the blossom into the sky holes / around the crown,
        #      stronger on the sun side
        if self.glow > 0:
            g1 = cv2.GaussianBlur(A * Bm, (0, 0), 5 * px)
            g2 = cv2.GaussianBlur(A * Bm, (0, 0), 16 * px)
            Mg = np.float32([[1, 0, Ls[0] * 6 * px], [0, 1, Ls[1] * 6 * px]])
            gs = cv2.warpAffine(g1, Mg, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
            gl = (0.22 * g1 + 0.12 * g2 + 0.1 * np.clip(g1 - gs, 0, 1)) * self.glow
            ga = np.clip(gl, 0, 0.35) * (1 - card[..., 3])
            card[..., :3] += pal['glow'] * ga[..., None]
            card[..., 3] += ga
        # the trunk foot sinks into the grass: nothing below the foot line
        cut = int(oy + 0.03 * k)
        if cut < H:
            fade = np.clip((cut - np.arange(H, dtype=np.float32)) / (2.0 * px) + 1.0, 0, 1)
            card *= fade[:, None, None]
        self.wood_front = wood_front
        return card, ox, oy

    def _holes(self, acc, bcol, bcov, wfront, rng):
        """sky holes punched through the blossom mass (scalloped, varied size), showing the wood behind"""
        px = self.px
        A, Bm = acc['A'], acc['B']
        H, W = A.shape
        inner = cv2.erode((A * Bm > 0.98).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(9 * px) | 1,) * 2))
        wf = cv2.dilate((wfront > 0.05).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(40 * px) | 1,) * 2))
        inner = inner & (wf == 0)
        ys, xs = np.nonzero(inner[::4, ::4])
        if len(ys) == 0:
            return
        area = len(ys) * 16.0
        nh = int(area / (70 * px) ** 2 * 0.5 * self.lace)
        if nh <= 0:
            return
        pick = rng.choice(len(ys), min(nh, len(ys)), replace=False)
        m8 = np.zeros((H, W), np.uint8)
        for i in pick:
            x, y = xs[i] * 4 + 2, ys[i] * 4 + 2
            # fewer holes deep in shadow, more in the upper / lit crown
            if acc['V'][y, x] < -0.3 and rng.random() < 0.7:
                continue
            r = px * rng.uniform(6, 15) * (1.0 if rng.random() < 0.75 else 1.8)
            ang = rng.uniform(0, math.pi)
            for _ in range(int(rng.integers(2, 5))):
                q = _lobe_poly(rng, r * rng.uniform(0.35, 0.8), int(np.clip(r / (2.5 * px), 5, 10)), sq=rng.uniform(0.55, 1.0))
                t_ = rng.normal(0, 0.9) * r
                q = q + (x + t_ * math.cos(ang) + rng.normal(0, 0.2) * r, y + t_ * math.sin(ang) * 0.6 + rng.normal(0, 0.2) * r)
                cv2.fillPoly(m8, [np.round(q * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        h = m8.astype(np.float32) / 255.0
        h = h * np.clip(Bm, 0, 1)
        # reveal: back wood where it is, sky elsewhere
        keep = 1 - h
        acc['C'] = acc['C'] * keep[..., None] + bcol * (h * bcov)[..., None]
        acc['A'] = A * keep + h * bcov
        acc['B'] = Bm * keep
        acc['LIT'] = acc['LIT'] * keep

    def _florets(self, C, A, V, LIT, edge, Bm, rng):
        """(c) small 5-petal flowers / white-pink specks only on the lit silhouette and the lit tops"""
        px = self.px
        H, W = A.shape
        pal = self.pal
        step = max(2, int(3 * px))
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        ys = ys.ravel()
        xs = xs.ravel()
        a = A[ys, xs]
        bm = Bm[ys, xs]
        vv = V[ys, xs]
        lt = LIT[ys, xs]
        eg = edge[ys, xs]
        p = bm * (a > 0.5) * (np.clip(eg * 3, 0, 1) * 0.07 + np.clip(lt * 1.5, 0, 1) * 0.03 * (vv > 0.1)
                              + np.clip((vv - 0.35) / 0.4, 0, 1) * 0.012 + 0.008) * self.detail
        sel = rng.random(len(p)) < p
        idx = np.nonzero(sel)[0]
        F8 = np.zeros((H, W, 3), np.uint8)             # colour / 200 (headroom above 1.0)
        FA = np.zeros((H, W), np.uint8)
        Ls = self.Ls
        for i in idx:
            x, y = xs[i] + rng.uniform(0, step), ys[i] + rng.uniform(0, step)
            r = px * rng.uniform(1.6, 3.4)
            if eg[i] > 0.3 and rng.random() < 0.35:
                x += Ls[0] * r * rng.uniform(0.8, 2.0)
                y += Ls[1] * r * rng.uniform(0.8, 2.0)
            lvl = rng.uniform(0, 1)
            if vv[i] > 0.4 or eg[i] > 0.3:
                col = pal['hot'] * (1 - 0.4 * lvl) + pal['spark'] * 0.4 * lvl
            else:
                col = pal['lit'] * (1 - 0.3 * lvl) + pal['hot'] * 0.3 * lvl
            if r < 2.2 * px:
                pts = np.array([[x - r * 0.6, y], [x, y - r * 0.6], [x + r * 0.6, y], [x, y + r * 0.6]])
            else:
                an = rng.uniform(0, 6.3)
                c_, s_ = math.cos(an), math.sin(an)
                P = _PETAL @ np.array([[c_, s_], [-s_, c_]]) * r
                P[:, 1] *= rng.uniform(0.75, 1.0)
                pts = P + (x, y)
            pp = [np.round(pts * 16).astype(np.int32)]
            cv2.fillPoly(F8, pp, tuple(float(min(255, max(0, q * 200))) for q in col), cv2.LINE_AA, 4)
            cv2.fillPoly(FA, pp, 255, cv2.LINE_AA, 4)
        fa = FA.astype(np.float32) / 255.0
        nz = fa > 0
        fc = F8[nz].astype(np.float32) / 200.0 / np.maximum(fa[nz], 1e-3)[:, None]
        C[nz] = C[nz] * (1 - fa[nz, None]) + np.minimum(fc, 1.3) * fa[nz, None]
        A[nz] = A[nz] + fa[nz] * (1 - A[nz])
        Bm[nz] = np.maximum(Bm[nz], fa[nz])
