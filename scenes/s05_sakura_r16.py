"""s05_sakura round 16: cherry trees built as a BRANCH SKELETON carrying blossom clusters (cm5_01 study).

Reviewer notes addressed: 'cotton-candy blob canopy', 'lollipop / square-cut limbs + floating fragments',
'flat saturated violet underside', 'balloon-on-string blossoms', 'clay trunks with lenticel dashes',
'white speckle noise'.

Tree16 (model in metres, x right / y up, trunk foot at the origin)
  * trunk: centre line of one of several shapes (straight / C / S / elbow), varied lean, root flare + roots;
  * 3-5 scaffold limbs (one maybe leaving lower on the trunk) that fork recursively (laterals + a terminal
    fork, Leonardo radii) down to thin twigs - every piece of wood is connected and tapers continuously;
  * blossom clusters strung along the depth>=2 wood (and twig tips), each one a small bunch of 3-7 flower
    heads + 5-petal florets on its rim; half the clusters sit BEHIND the wood, half in FRONT, so limbs weave
    through the blossom; low-frequency gating leaves real sky holes, larger toward the crown edge.

Painter16
  * value is a smooth crown field (canopy thickness at two scales -> form normal, lit from the sun, height
    in the crown, shadow cast by the blossom toward the sun) quantised into flat painted bands per head:
    cool lavender-grey core, lavender-pink shade, pink, light pink, pink-white top;
  * warm peach transmitted light on thin sun-side clusters; lit heads crisp with a next-band crescent on
    the sun side, shadow heads with soft, lost edges;
  * wood painted grey-violet: cool shadow side, warm lit rim on the sun side, vertical bark fissures, a few
    knots, lichen / moss patches low on the shade side; wood inside the crown darker and cooler.
"""
import math

import numpy as np
import cv2
from numba import njit

from s05_sakura_r15 import _raster_wood, _unit, _rot, _smooth, _sstep


# ============================================================================ model

class Tree16:
    def __init__(self, rng, scale=1.0, lean=-1.0, spread=1.0, hmul=1.0, density=1.0, trunk=1.0, cl=1.0,
                 maxd=4, big=True, fork=1.0):
        s = self.s = float(scale)
        sg = 1.0 if lean >= 0 else -1.0
        self.sg = sg
        self.density = float(density)
        self.cl = float(cl)
        self.maxd = maxd
        self.wood = []
        self.clusters = []
        self.big = big
        self.rmin = 0.006 * s
        # ---- trunk
        ht = rng.uniform(1.25, 1.7) * s * hmul * fork
        r0 = rng.uniform(0.15, 0.2) * s * trunk
        n = 40
        f = np.linspace(0, 1, n)
        lean_ang = sg * rng.uniform(3, 24)
        kind = rng.choice(['straight', 'C', 'S', 'elbow'], p=[0.2, 0.3, 0.25, 0.25])
        a = np.zeros(n)
        if kind == 'C':
            a += rng.choice([-1, 1]) * rng.uniform(8, 16) * (f - 0.5) * 2
        elif kind == 'S':
            a += rng.uniform(7, 13) * np.sin(2 * math.pi * f + rng.uniform(-0.5, 0.5)) * rng.choice([-1, 1])
        elif kind == 'elbow':
            i = int(rng.integers(n // 4, 3 * n // 4))
            a[i:] += rng.choice([-1, 1]) * rng.uniform(12, 22)
            a = _smooth(a, 5)
        a += rng.normal(0, 1.5, n).cumsum() * 0.5
        a = _smooth(a, 3)
        P = [np.array([0.0, -0.3 * s])]
        step = (ht + 0.3 * s) / (n - 1)
        for i in range(1, n):
            ang = math.radians(lean_ang * (0.5 + 0.5 * f[i]) + a[i])
            P.append(P[-1] + step * np.array([math.sin(ang), math.cos(ang)]))
        P = np.array(P)
        rt = r0 * (1.0 - 0.25 * f) * (1.0 + 0.6 * np.exp(-np.maximum(P[:, 1], 0) / (0.16 * s)))
        for _ in range(int(rng.integers(1, 3))):
            c = rng.uniform(0.2, 0.8)
            rt = rt * (1.0 + rng.uniform(0.05, 0.12) * np.exp(-((f - c) / 0.06) ** 2))
        self.ht = ht
        self.wood.append(dict(P=P, r=rt, d=0, m=-1, rid=0))
        # roots: short buttresses spreading at the foot
        for sd in (-1, 1):
            if rng.random() < 0.0:            # round 16: roots read as knobs -> flare only
                q0 = np.array([sd * r0 * 0.3, 0.16 * s])
                dd = _unit([sd * 1.0, -rng.uniform(0.55, 0.8)])
                L = rng.uniform(0.3, 0.4) * s
                Q = np.array([q0 + dd * L * u for u in np.linspace(0, 1, 8)])
                rr = r0 * 0.75 * (1 - 0.55 * np.linspace(0, 1, 8))
                self.wood.append(dict(P=Q, r=rr, d=0, m=-1, rid=len(self.wood)))
        # ---- scaffolds
        nsc = int(rng.choice([3, 4, 4, 5] if big else [3, 3, 4]))
        base = np.linspace(-72, 68, nsc) + rng.uniform(-10, 10, nsc) + sg * 10.0
        w = rng.uniform(0.6, 1.0, nsc)
        w = w / w.sum()
        low = int(np.argmax(np.abs(base))) if rng.random() < 0.7 else -1
        F = P[-1]
        rF = float(rt[-1])
        self._mid = 0
        for j in range(nsc):
            ang = float(base[j])
            if j == low:
                i = int(rng.uniform(0.55, 0.8) * (n - 1))
                p0 = P[i]
                rr = float(rt[i]) * 0.62
                d = _unit(_rot(np.array([0.0, 1.0]), -float(np.clip(ang, -62, 62))))
                L = rng.uniform(2.2, 2.9) * s * spread
            else:
                p0 = F + np.array([rng.uniform(-0.3, 0.3) * rF, rng.uniform(-0.5, 0.0) * rF])
                rr = rF * math.sqrt(w[j]) * 1.2
                d = _unit(_rot(np.array([0.0, 1.0]), -ang))
                out = 1.0 + 0.25 * (1.0 if d[0] * sg > 0 else 0.0)
                L = rng.uniform(2.0, 2.8) * s * spread * out * (0.8 + 0.3 * abs(math.sin(math.radians(ang))))
            self._limb(rng, p0, d, L, rr, 1, j)
        self._gate(rng)

    def _limb(self, rng, p0, d, L, r0, depth, m):
        s = self.s
        nn = max(6, int(L / (0.04 * s)))
        dd = _unit(d)
        step = L / (nn - 1)
        g = (0.0 if depth == 1 else 0.006 * depth) * rng.uniform(0.6, 1.4)     # droop increases outward
        ymin = {1: 0.12, 2: -0.2}.get(depth, -0.4)
        curl = rng.normal(0, 1.0)
        nk = int(rng.integers(1, 4))
        kinks = set(rng.choice(np.arange(2, nn - 1), size=min(nn - 3, nk), replace=False).tolist())
        P = [np.asarray(p0, np.float64)]
        for i in range(1, nn):
            a = rng.normal(0, 2.4) + curl
            if i in kinks:
                a += rng.choice([-1, 1]) * rng.uniform(6, 16)
            dd = _rot(dd, a)
            u = i / (nn - 1)
            dd = _unit(dd + np.array([0.0, -g * u * 2.0]))
            if dd[1] < ymin:
                dd = _unit([dd[0] * math.sqrt(max(1 - ymin * ymin, 1e-3)) / max(abs(dd[0]), 1e-3), ymin])
            P.append(P[-1] + dd * step)
        P = np.array(P)
        tt = np.linspace(0, 1, nn)
        last = depth >= self.maxd
        r1 = self.rmin if last else max(self.rmin, r0 * 0.5)
        r = r0 + (r1 - r0) * tt ** (0.9 if not last else 0.7)
        self.wood.append(dict(P=P, r=r, d=depth, m=m, rid=len(self.wood)))
        if not last:
            nl = {1: int(rng.integers(3, 6)), 2: int(rng.integers(3, 5)), 3: int(rng.integers(2, 4))}.get(depth, 2)
            ts = np.sort(rng.uniform(0.15, 0.88, nl))
            side = rng.choice([-1, 1])
            for tl in ts:
                i = int(tl * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                cd = _rot(td, side * rng.uniform(25, 60))
                side = -side
                cd = _unit(cd + np.array([0.0, 0.25]))                 # laterals reach up toward light
                Lc = L * rng.uniform(0.4, 0.62) * (1.0 - 0.3 * tl)
                if Lc < 0.08 * s:
                    continue
                self._limb(rng, P[i], cd, Lc, max(self.rmin, r[i] * rng.uniform(0.45, 0.65)), depth + 1, m)
            td = _unit(P[-1] - P[-3])
            share = rng.uniform(0.35, 0.65)
            for sgn, sh in ((-1, share), (1, 1 - share)):
                cd = _rot(td, sgn * rng.uniform(14, 36))
                self._limb(rng, P[-1], cd, L * rng.uniform(0.42, 0.6), max(self.rmin, r[-1] * math.sqrt(sh) * 1.05),
                           depth + 1, m)
        # ---- blossom clusters along the wood (not on the inner part of the scaffolds)
        if depth >= 2 or (depth == 1 and not self.big):
            Rb = 0.1 * s * self.cl
            t = (0.55 if depth == 1 else 0.25 if depth == 2 else 0.05) + rng.uniform(0, 0.12)
            while t <= 1.0:
                i = int(t * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                nrm = np.array([-td[1], td[0]])
                R = Rb * rng.uniform(0.65, 1.35) * (1.2 if depth <= 2 else 1.0)
                off = rng.normal(0, 0.45) * R
                c = P[i] + nrm * off + np.array([0.0, -0.25 * R])           # blossom hangs off the twig
                self.clusters.append(dict(x=float(c[0]), y=float(c[1]), R=float(R), m=m, d=depth,
                                          tx=float(td[0]), ty=float(td[1]),
                                          front=bool(rng.random() < 0.5), seed=int(rng.integers(1 << 30))))
                t += (R / L) * rng.uniform(1.1, 2.0) / max(self.density, 0.4)
            if last:
                tip = P[-1]
                R = Rb * rng.uniform(0.6, 1.0)
                self.clusters.append(dict(x=float(tip[0]), y=float(tip[1] - 0.2 * R), R=float(R), m=m, d=depth,
                                          tx=float(dd[0]), ty=float(dd[1]), front=True,
                                          seed=int(rng.integers(1 << 30))))

    def _gate(self, rng):
        """Low-frequency gating: drop clusters in patches -> real sky holes, larger toward the crown edge;
        and no blossom low on the trunk."""
        cl = [c for c in self.clusters if c['y'] > 0.75 * self.ht]
        if not cl:
            self.clusters = cl
            return
        X = np.array([c['x'] for c in cl])
        Y = np.array([c['y'] for c in cl])
        cx, cy = np.median(X), np.median(Y)
        sx_, sy_ = X.std() + 1e-6, Y.std() + 1e-6
        # random blobs of 'no blossom'
        nb = 7
        bx = rng.uniform(X.min(), X.max(), nb)
        by = rng.uniform(Y.min(), Y.max(), nb)
        br = rng.uniform(0.25, 0.55, nb) * self.s
        keep = []
        for c in cl:
            dn = math.hypot((c['x'] - cx) / sx_, (c['y'] - cy) / sy_)
            p = 0.93 - 0.18 * max(0.0, dn - 1.0)
            hole = np.exp(-((c['x'] - bx) ** 2 + (c['y'] - by) ** 2) / (br * br)).max()
            p *= 1.0 - 0.85 * hole
            if rng.random() < p:
                keep.append(c)
        self.clusters = keep

    def extent(self):
        xs, ys = [], []
        for w in self.wood:
            P, r = w['P'], w['r']
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for c in self.clusters:
            xs += [c['x'] - 1.5 * c['R'], c['x'] + 1.5 * c['R']]
            ys += [c['y'] - 1.5 * c['R'], c['y'] + 1.5 * c['R']]
        return min(xs), max(xs), min(ys), max(ys)


# ============================================================================ stamp rasteriser

@njit(cache=True)
def _splat(C, A, st):
    """st rows: x, y, r, rot, nb, amp, aa, cr, cg, cb, lx, ly, th, ar, ag, ab, alpha
    polar-scalloped shape (nb bumps); a crisp crescent (colour +a) where the offset along (lx, ly) > th."""
    H, W = A.shape
    for i in range(st.shape[0]):
        x, y, r, rot = st[i, 0], st[i, 1], st[i, 2], st[i, 3]
        nb, amp, aa = st[i, 4], st[i, 5], st[i, 6]
        lx, ly, th = st[i, 10], st[i, 11], st[i, 12]
        al = st[i, 16]
        rm = r + aa + 1.0
        xa = max(0, int(x - rm))
        xb = min(W, int(x + rm) + 2)
        ya = max(0, int(y - rm))
        yb = min(H, int(y + rm) + 2)
        for yy in range(ya, yb):
            for xx in range(xa, xb):
                dx = xx + 0.5 - x
                dy = yy + 0.5 - y
                d = math.sqrt(dx * dx + dy * dy)
                if d > rm:
                    continue
                th_ = math.atan2(dy, dx) - rot
                b = abs(math.cos(0.5 * nb * th_))
                re = r * (1.0 - amp + amp * math.sqrt(b))
                c = (re - d) / aa + 0.5
                if c <= 0.0:
                    continue
                if c > 1.0:
                    c = 1.0
                c *= al
                cr, cg, cb = st[i, 7], st[i, 8], st[i, 9]
                if st[i, 13] != 0.0 or st[i, 14] != 0.0 or st[i, 15] != 0.0:
                    tt = (dx * lx + dy * ly) / max(r, 1e-3)
                    k = (tt - th) * r / 0.9 + 0.5
                    if k > 0.0:
                        if k > 1.0:
                            k = 1.0
                        cr += st[i, 13] * k
                        cg += st[i, 14] * k
                        cb += st[i, 15] * k
                C[yy, xx, 0] = C[yy, xx, 0] * (1.0 - c) + cr * c
                C[yy, xx, 1] = C[yy, xx, 1] * (1.0 - c) + cg * c
                C[yy, xx, 2] = C[yy, xx, 2] * (1.0 - c) + cb * c
                A[yy, xx] = A[yy, xx] * (1.0 - c) + c


# ============================================================================ painter

BANDS16 = np.array([
    (0.6, 0.55, 0.74),     # 0 deep interior: cool lavender-grey
    (0.8, 0.56, 0.76),     # 1 shade: mauve rose
    (0.96, 0.56, 0.73),    # 2 mid: rose pink
    (1.0, 0.72, 0.83),     # 3 light pink
    (1.03, 0.86, 0.9),     # 4 lit top: pink-white
    (1.05, 0.95, 0.94),    # 5 hot
], np.float32)
BANDS16_FAR = (BANDS16 * 0.75 + np.array([0.9, 0.86, 0.95], np.float32) * 0.25).astype(np.float32)
TRANS = np.array([1.02, 0.77, 0.68], np.float32)
WOOD16 = dict(sh=(0.16, 0.14, 0.2), mid=(0.3, 0.27, 0.31), lit=(0.47, 0.41, 0.4), rim=(1.0, 0.8, 0.6),
              cool=(0.34, 0.37, 0.5), crack=(0.08, 0.07, 0.1), moss=(0.42, 0.5, 0.33), lichen=(0.56, 0.58, 0.5),
              inner=(0.26, 0.23, 0.32))
WOOD16_FAR = dict(sh=(0.28, 0.26, 0.34), mid=(0.38, 0.35, 0.42), lit=(0.5, 0.45, 0.48), rim=(0.92, 0.8, 0.7),
                  cool=(0.4, 0.42, 0.54), crack=(0.2, 0.19, 0.25), moss=(0.46, 0.52, 0.42),
                  lichen=(0.56, 0.58, 0.54), inner=(0.4, 0.37, 0.46))


def _wrap_blur(g, sx, sy):
    h, w = g.shape
    t = np.tile(g, (3, 3))
    t = cv2.GaussianBlur(t, (0, 0), sigmaX=sx, sigmaY=sy)
    return np.ascontiguousarray(t[h:2 * h, w:2 * w])


def _band_col(bands, v):
    """v in [0, 5] -> piecewise-linear band colour (caller quantises)"""
    v = np.clip(v, 0, len(bands) - 1.001)
    i = np.floor(v).astype(int)
    fr = (v - i)[..., None]
    return bands[i] * (1 - fr) + bands[i + 1] * fr


class Painter16:
    def __init__(self, k, sun_dir=(-0.7, -0.7), px=2.0, far=False, detail=1.0):
        self.k = float(k)
        self.Ls = _unit(sun_dir)                 # screen direction toward the sun (y down)
        self.px = float(px)                      # one 1080p pixel in card px
        self.far = far
        self.detail = detail
        self.bands = BANDS16_FAR if far else BANDS16
        self.wp = {kk: np.array(v, np.float32) for kk, v in (WOOD16_FAR if far else WOOD16).items()}

    # ------------------------------------------------------------------ crown field
    def _field(self, cl, H, W):
        q = 4
        h, w = H // q + 1, W // q + 1
        D = np.zeros((h, w), np.float32)
        for c in cl:
            cv2.circle(D, (int(c['px'] / q), int(c['py'] / q)), max(1, int(c['pr'] * 1.1 / q)), 1.0, -1)
        k = self.k
        s1 = max(1.0, 0.28 * k / q)
        s2 = max(1.0, 0.8 * k / q)
        D1 = cv2.GaussianBlur(D, (0, 0), s1)
        D2 = cv2.GaussianBlur(D, (0, 0), s2)
        # form normal of the 'thickness' height field
        hgt = D1 * 0.6 + D2 * 0.8
        gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        sc = 0.55 * k / q
        nx, ny, nz = -gx * sc, -gy * sc, np.ones_like(gx)
        nl = np.sqrt(nx * nx + ny * ny + nz * nz)
        Ls = self.Ls
        L3 = _unit([Ls[0] * 0.75, Ls[1] * 0.75 - 0.35, 0.55])
        lam = (nx * L3[0] + ny * L3[1] + nz * L3[2]) / nl
        # blossom toward the sun shades this spot (cast shadow inside the crown)
        sh = max(1, int(0.45 * k / q))
        M = np.float32([[1, 0, -Ls[0] * sh], [0, 1, -Ls[1] * sh]])
        Dsun = cv2.warpAffine(D2, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
        # outward (silhouette) normal toward the sun: thin edges facing the sun transmit light
        edge_face = -(gx * Ls[0] + gy * Ls[1]) * sc
        self._q = q
        return dict(D1=D1, D2=D2, lam=lam, Dsun=Dsun, face=edge_face)

    def _at(self, F, x, y):
        q = self._q
        h, w = F.shape
        xi = np.clip((np.asarray(x) / q).astype(int), 0, w - 1)
        yi = np.clip((np.asarray(y) / q).astype(int), 0, h - 1)
        return F[yi, xi]

    # ------------------------------------------------------------------ wood
    def _wood(self, tree, to_px, H, W, fld, rng):
        segs = []
        px = self.px
        for w in tree.wood:
            P = to_px(w['P'])
            r = np.asarray(w['r'], np.float64) * self.k
            v = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(w['P'], axis=0), axis=1))])
            if len(P) > 3:
                for _ in range(2):
                    P[1:-1] = 0.25 * P[:-2] + 0.5 * P[1:-1] + 0.25 * P[2:]
            sl = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
            nd = int(max(len(P), sl[-1] / (1.5 * px)))
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
            rmin = 0.35 * px
            for i in range(len(P) - 1):
                segs.append((P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1], max(r[i], rmin), max(r[i + 1], rmin),
                             v[i], v[i + 1], float(w['d']), float(w['rid']), Nn[i, 0], Nn[i, 1], Nn[i + 1, 0],
                             Nn[i + 1, 1]))
        cov = np.zeros((H, W), np.float32)
        S = np.zeros((H, W), np.float32)
        Vv = np.zeros((H, W), np.float32)
        RR = np.zeros((H, W), np.float32)
        PRI = np.full((H, W), 99.0, np.float32)
        NX = np.zeros((H, W), np.float32)
        NY = np.zeros((H, W), np.float32)
        RID = np.zeros((H, W), np.float32)
        _raster_wood(H, W, np.array(segs, np.float64), cov, S, Vv, RR, PRI, NX, NY, RID,
                     np.full((H, W), -1e9, np.float32))
        # thin wood: minimum coverage fades instead of drawing sub-pixel lines at full strength
        thin = np.clip(RR / (0.5 * px), 0.35, 1.0)
        cov = cov * np.where(RR > 0, thin, 1.0).astype(np.float32)
        wp = self.wp
        Ls = self.Ls
        facing = S * (NX * Ls[0] + NY * Ls[1])            # +1 on the sun-side edge
        a_s = np.abs(S)
        col = np.broadcast_to(wp['sh'], (H, W, 3)).astype(np.float32).copy()
        # painted planes: shadow / mid / lit (soft steps), cool bounce on the shadow edge
        m1 = _sstep((facing - 0.15) / 0.3 + 0.5)
        col += (wp['mid'] - wp['sh']) * m1[..., None]
        m2 = _sstep((facing - 0.5) / 0.2 + 0.5) * np.clip((RR - 2.0 * px) / (3 * px), 0, 1)
        col += (wp['lit'] - wp['mid']) * m2[..., None]
        cb = _sstep((-facing - 0.65) / 0.2 + 0.5) * np.clip((RR - 3 * px) / (4 * px), 0, 1)
        col += (wp['cool'] - col) * (cb * 0.55)[..., None]
        # bark texture (thick wood): vertical fissures in (across, along) space, knots, lichen / moss
        thick = np.clip((RR - 3.0 * px) / (6 * px), 0, 1)
        if thick.max() > 0:
            ang = np.arcsin(np.clip(S, -1, 1))                    # angle round the cylinder
            rm = np.maximum(RR / self.k, 1e-3)
            arc = ang * rm                                          # metres round the bark
            tex = self._bark_tex(rng)
            th_, tw_ = tex.shape
            # texture coords: 0.03 m per texel across, 0.12 m along -> long vertical fissures
            tu = ((arc / 0.018 + RID * 37.0) % tw_).astype(np.float32)
            tv = ((Vv / 0.05 + RID * 91.0) % th_).astype(np.float32)
            n = cv2.remap(tex, tu, tv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            crack = _sstep((np.abs(n) - 0.09) / -0.05 + 0.5) * (1 - 0.6 * np.clip(facing, 0, 1))
            # fissures foreshorten and vanish toward the silhouette
            crack *= np.clip(1.2 - a_s, 0, 1)
            col += (wp['crack'] - col) * (crack * thick * 0.9)[..., None]
            # ridges between the fissures catch a little light on the sun side
            ridge = _sstep((n - 0.25) / 0.15 + 0.5) * np.clip(facing + 0.2, 0, 1)
            col += (wp['lit'] - col) * (ridge * thick * 0.25)[..., None]
            # lichen / moss: low-frequency patches, more on the shade side and low on the trunk
            tv2 = ((Vv / 0.35 + RID * 13.0) % th_).astype(np.float32)
            tu2 = ((arc / 0.09 + RID * 5.0) % tw_).astype(np.float32)
            mn = cv2.remap(self._blob_tex, tu2, tv2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            low = np.clip(1.4 - Vv / 1.4, 0.2, 1.0)
            moss = _sstep((mn - 0.55) / 0.12 + 0.5) * np.clip(0.4 - facing, 0, 1) * low
            col += (wp['moss'] - col) * (moss * thick * 0.6)[..., None]
            lich = _sstep((mn + 0.62) / -0.1 + 0.5) * np.clip(facing + 0.6, 0, 1)
            col += (wp['lichen'] - col) * (lich * thick * 0.35)[..., None]
            # a few sparse, irregular horizontal lenticels (faint)
            sp = 0.09
            j = np.floor(Vv / sp + 0.3 * np.sin(arc * 9 + RID))
            fr = Vv / sp + 0.3 * np.sin(arc * 9 + RID) - j
            hsh = np.sin(j * 12.9898 + RID * 78.233) * 43758.5453
            hsh = hsh - np.floor(hsh)
            dash = (fr < 0.08) & (hsh < 0.3) & (np.abs(S - (hsh * 3 - 0.5)) < 0.35)
            col += (wp['crack'] - col) * (dash.astype(np.float32) * thick * 0.35)[..., None]
        # warm lit rim on the sun side (1-2 px at 1080p)
        rw = np.clip((1.5 * px) / np.maximum(RR, 0.5), 0.05, 0.6)
        rimk = _sstep((a_s - (1.0 - rw)) / (0.5 * rw) + 0.5) * _sstep((facing - 0.3) / 0.25 + 0.5)
        # wood in the crown: darker, cooler (in the blossom's shade), rim weaker
        inside = np.clip(cv2.resize(fld['D2'], (W, H), interpolation=cv2.INTER_LINEAR) * 1.6, 0, 1)
        col += (wp['rim'] - col) * (rimk * (0.9 - 0.5 * inside))[..., None]
        col += (wp['inner'] - col) * (inside * 0.55 * (1 - thick))[..., None]
        return col.astype(np.float32), cov

    def _bark_tex(self, rng):
        if getattr(self, '_btex', None) is not None:
            return self._btex
        th, tw = 256, 256
        g = rng.standard_normal((th, tw)).astype(np.float32)
        # anisotropic: long along (rows), short across (cols)
        g = _wrap_blur(g, 1.2, 5.0)
        g = g / (g.std() + 1e-6) * 0.35
        self._btex = g
        b = rng.standard_normal((64, 64)).astype(np.float32)
        b = _wrap_blur(b, 2.5, 2.5)
        b = b / (b.std() + 1e-6) * 0.35
        self._blob_tex = cv2.resize(b, (tw, th), interpolation=cv2.INTER_CUBIC)
        return g

    # ------------------------------------------------------------------ blossom
    def _cluster_stamps(self, c, v, trans, rng):
        """stamps for one cluster: 3-7 flower heads + florets on the rim.  v: band value (0..5)."""
        px = self.px
        Ls = self.Ls
        bands = self.bands
        R = c['pr']
        nh = int(rng.integers(3, 8))
        st = []
        # heads spread along the twig direction (bunches hang along the wood)
        tx, ty = c['tx'], -c['ty']
        heads = []
        for h in range(nh):
            u = rng.uniform(-1, 1)
            w_ = rng.normal(0, 0.35)
            hx = c['px'] + (u * tx - w_ * ty) * R * 0.6
            hy = c['py'] + (u * ty + w_ * tx) * R * 0.6
            hr = R * rng.uniform(0.38, 0.62)
            heads.append((hx, hy, hr))
        # heads lower in the bunch drawn first (the upper ones overlap them - lit tops read on top)
        heads.sort(key=lambda t: -(t[0] * Ls[0] + t[1] * Ls[1]))
        for (hx, hy, hr) in heads:
            # local light: the sun side of the bunch is a notch brighter
            off = ((hx - c['px']) * Ls[0] + (hy - c['py']) * Ls[1]) / max(R, 1e-3)
            vh = v + 0.55 * off + rng.normal(0, 0.18)
            vq = float(np.clip(np.round(vh), 0, 4))
            col = bands[int(vq)].copy()
            # rose saturation variety between bunches
            col = col + (col - col.mean()) * rng.uniform(-0.1, 0.25)
            if trans > 0:
                col = col + (TRANS - col) * trans * 0.8
            lit = vq >= 2
            aa = (0.8 if lit else 1.2 + 0.4 * (2 - vq)) * px
            cres = (bands[int(vq) + 1] - bands[int(vq)]) * (0.9 if lit else 0.5)
            if trans > 0:
                cres = cres + (TRANS * 1.08 - col) * trans * 0.4
            st.append((hx, hy, hr, rng.uniform(0, 6.3), float(rng.integers(5, 8)), 0.24, aa,
                       col[0], col[1], col[2], Ls[0], Ls[1], 0.35 if lit else 0.55,
                       cres[0], cres[1], cres[2], 1.0))
            # a few deeper-rose blossoms inside the head (painted texture, same hue family, no speckle)
            if hr > 3.5 * px:
                for _ in range(int(rng.integers(1, 4))):
                    a_ = rng.uniform(0, 6.3)
                    rr_ = hr * rng.uniform(0.1, 0.55)
                    fc = _band_col(bands, np.array(max(vq - 0.45, 0.0))).astype(np.float32)
                    fc = fc + (fc - fc.mean()) * 0.25
                    st.append((hx + math.cos(a_) * rr_, hy + math.sin(a_) * rr_, hr * rng.uniform(0.25, 0.38),
                               rng.uniform(0, 6.3), 5.0, 0.4, 0.8 * px, fc[0], fc[1], fc[2], 0.0, 0.0, 9.0,
                               0.0, 0.0, 0.0, 0.8))
            # florets on the head rim (mostly on the outer / sun side), same band a notch lighter
            nf = int(rng.integers(2, 7) * self.detail + 0.5)
            fr_ = max(0.9 * px, hr * rng.uniform(0.28, 0.4))
            for _ in range(nf):
                a = math.atan2(Ls[1], Ls[0]) + rng.normal(0, 1.3)
                fx = hx + math.cos(a) * hr * rng.uniform(0.75, 1.0)
                fy = hy + math.sin(a) * hr * rng.uniform(0.75, 1.0)
                ft = min(vq + 0.6, 4.6) if math.cos(a - math.atan2(Ls[1], Ls[0])) > 0 else vq
                fc = _band_col(bands, np.array(ft)).astype(np.float32)
                if trans > 0:
                    fc = fc + (TRANS * 1.05 - fc) * trans * 0.7
                st.append((fx, fy, fr_ * rng.uniform(0.8, 1.2), rng.uniform(0, 6.3), 5.0, 0.42,
                           (0.7 if lit else 1.2) * px, fc[0], fc[1], fc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
        return st

    def paint(self, tree, rng, margin=10):
        k = self.k
        px = self.px
        margin = int(margin + 12 * px)
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * margin
        ox = -x0 * k + margin
        oy = y1 * k + margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        cl = tree.clusters
        for c in cl:
            c['px'] = ox + c['x'] * k
            c['py'] = oy - c['y'] * k
            c['pr'] = max(c['R'] * k, 2.5 * px)
        fld = self._field(cl, H, W)
        # ---- per-cluster value
        Ls = self.Ls
        X = np.array([c['px'] for c in cl])
        Y = np.array([c['py'] for c in cl])
        lam = self._at(fld['lam'], X, Y)
        D1 = self._at(fld['D1'], X, Y)
        D2 = self._at(fld['D2'], X, Y)
        Dsun = self._at(fld['Dsun'], X, Y)
        face = self._at(fld['face'], X, Y)
        ytop, ybot = Y.min(), Y.max()
        hgt = 1.0 - (Y - ytop) / max(ybot - ytop, 1.0)                    # 1 at the top of the crown
        v = 2.2 + 2.6 * (lam - 0.48) + 1.7 * (hgt - 0.35) - 1.5 * (Dsun - 0.67) - 0.8 * (D2 - 0.69)
        mb = np.random.default_rng(int(self.k * 7) % 1000).normal(0, 0.3, 64)
        v = v + np.array([mb[int(c['m']) % 64] for c in cl])
        # local form of each scaffold mass: its top catches the light, its underside falls into lavender
        mids = np.array([int(c['m']) for c in cl])
        loc = np.zeros(len(cl))
        for m in np.unique(mids):
            sel = mids == m
            yy = Y[sel]
            lo, hi = np.percentile(yy, 8), np.percentile(yy, 92)
            loc[sel] = np.clip((yy - lo) / max(hi - lo, 1.0), 0, 1)          # 0 top .. 1 bottom
        v = v + 1.0 * (0.4 - loc) - 1.4 * np.clip(loc - 0.55, 0, 1)
        front = np.array([c['front'] for c in cl])
        v = v + np.where(front, 0.3, -0.75)
        # thin, sun-facing edge clusters: warm transmitted light
        thin = np.clip((0.97 - D1) / 0.45, 0, 1)
        # also the crown side that faces the sun (screen offset from the crown centre toward the sun)
        cx_, cy_ = np.median(X), np.median(Y)
        sw = np.clip(((X - cx_) * Ls[0] + (Y - cy_) * Ls[1]) / max(np.percentile(np.abs(X - cx_), 90), 1.0), 0, 1)
        trans = np.clip((thin + 0.5 * sw) * np.clip(face * 3.0 + 0.4 * sw, 0, 1), 0, 1) * (v < 3.8) * 0.85
        self.dbg = dict(trans=trans, v=v)
        stb, stf = [], []
        order = np.argsort(v)
        for i in order:
            c = cl[i]
            r2 = np.random.default_rng(c['seed'])
            st = self._cluster_stamps(c, float(v[i]), float(trans[i]), r2)
            (stf if c['front'] else stb).extend(st)
        C = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W), np.float32)
        if stb:
            _splat(C, A, np.array(stb, np.float64))
        wcol, wcov = self._wood(tree, to_px, H, W, fld, rng)
        C = C * (1 - wcov[..., None]) + wcol * wcov[..., None]
        A = A + wcov * (1 - A)
        if stf:
            _splat(C, A, np.array(stf, np.float64))
        # ---- sun-facing silhouette: thin warm rim + soft outer glow; shadow side: lost edge
        sh = 1.6 * px
        M = np.float32([[1, 0, -Ls[0] * sh], [0, 1, -Ls[1] * sh]])
        As = cv2.warpAffine(A, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        rim = np.clip(A - As, 0, 1)
        C = C / np.maximum(A, 1e-4)[..., None]                   # premultiplied -> straight
        C += (np.array([1.08, 0.92, 0.84], np.float32) - C) * (rim * 0.55)[..., None]
        M2 = np.float32([[1, 0, Ls[0] * sh], [0, 1, Ls[1] * sh]])
        Ad = cv2.warpAffine(A, M2, (W, H), borderMode=cv2.BORDER_CONSTANT)
        lost = np.clip(A - Ad, 0, 1)            # edges on the side away from the sun
        lost = cv2.GaussianBlur(lost, (0, 0), 1.5 * px)
        # the trunk sinks into the ground: flat soft cut just below the foot (no rounded capsule end)
        yy = np.arange(H, dtype=np.float32)[:, None]
        A = A * np.clip((oy + 0.02 * k - yy) / (1.5 * px) + 0.5, 0, 1)
        pm = np.dstack([C * A[..., None], A])
        pmb = cv2.GaussianBlur(pm, (0, 0), 1.3 * px)
        lk = np.clip(lost * 1.0, 0, 0.6)[..., None] * (1 - wcov[..., None])
        pm = pm * (1 - lk) + pmb * lk
        self.stats = dict(n=len(cl))
        return pm.astype(np.float32), ox, oy
