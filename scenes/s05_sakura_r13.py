"""s05_sakura round 13: cherry trees rebuilt after the 5 Centimeters per Second sakura frames.

Reviewer notes addressed: 'lumpy pink broccoli / cotton-candy blobs', 'noisy speckled procedural edges',
'lollipop stick trunks'.

Tree13 (model in metres, x right / y up, trunk foot at the origin)
  * a STOUT trunk with a root flare that splits low into 3-5 wide-spreading scaffold limbs (cherries are
    wider than tall); every limb is a gently arching cubic that carries alternating LATERAL branches along
    its length (not just forks at the end), recursively, continuously tapering down to fine twigs whose
    outer ends droop;
  * blossom is born ON the wood: small clusters (0.12-0.3 m) are strung along every fine branch and
    gathered at the twig ends -> larger masses emerge from many small clusters, sky shows between the
    branches, and the limbs visibly run into and between the masses;
  * every element has a depth z (per branch) -> painter's order interleaves back blossom, wood and
    front blossom.

Painter13 (card painter: paint(tree, rng) -> premultiplied ss-card, ox, oy)
  * each cluster = a shadow under-body + 10-60 opaque blossom dabs painted bottom-up, so lit dabs on the
    top overlap the shaded ones below (painted, not rendered); dab value = local cluster form + crown form
    + occlusion by the clusters above, quantised into 5 painted value bands (pink-white / light pink / pink
    / lavender / deep mauve); sun-side fringe dabs in shade take a warm transmitted peach-pink;
  * dabs are 5-petal floret shapes (near) or irregular round dabs (far), drawn anti-aliased (8-bit AA
    rasteriser, premultiplied), never single-pixel speckle;
  * wood: cel-shaded tapered polygons - cool plum shadow body, warm lit plane on the sun side, 1-2 px
    warm rim, cool sky bounce on the shade edge, horizontal lenticel bands on the trunk / big limbs;
  * a warm rim on the sun-facing blossom silhouette.
"""
import math

import numpy as np
import cv2

SC8 = 200.0          # 8-bit paint buffers store value * SC8 (headroom above 1.0 for hot lights)


def _cbez(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def _rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


# ============================================================================ model

class Tree13:
    def __init__(self, rng, scale=1.0, lean=-1.0, spread=1.0, hmul=1.0, big=True, density=1.0, nsc=None,
                 droop=1.0, trunk=1.0):
        s = self.s = float(scale)
        self.lean = lean
        sg = 1.0 if lean >= 0 else -1.0
        self.sg = sg
        self.big = big
        self.density = density
        self.droop = droop
        self.wood = []        # dict(P, r, z, d)
        self.clusters = []    # dict(x, y, R, z, d)
        self.maxd = 4 if big else 3
        # ---- trunk: stout, root flare, slight lean / curve
        ht = rng.uniform(1.15, 1.6) * s * hmul
        r0 = rng.uniform(0.2, 0.26) * s * trunk
        lx = sg * rng.uniform(0.12, 0.35) * ht
        c1 = rng.uniform(-0.06, 0.06) * ht
        T = _cbez([0.0, -0.1 * s], [c1, 0.35 * ht], [lx * 0.55, 0.7 * ht], [lx, ht], 30)
        f = np.linspace(0, 1, len(T))
        rt = r0 * (1.0 - 0.28 * f) * (1.0 + 0.55 * np.exp(-np.maximum(T[:, 1] + 0.1 * s, 0) / (0.18 * s)))
        self.wood.append(dict(P=T, r=rt, z=0.0, d=0))
        self.ht = ht
        F = T[-1]
        tdir = _unit(T[-1] - T[-4])
        rF = float(rt[-1])
        # ---- scaffold limbs from the crotch (plus a lower one on some trees)
        n = nsc or int(rng.choice([3, 4, 4, 5] if big else [2, 3, 3, 4]))
        base = np.linspace(-62, 58, n) + rng.uniform(-9, 9, n)
        base = base + sg * 10.0
        w = rng.uniform(0.6, 1.0, n)
        w = w / w.sum()
        for j in range(n):
            ang = float(base[j])
            d = _rot(np.array([0.0, 1.0]), -ang)                    # +ang -> toward +x
            d = _unit(0.75 * d + 0.25 * tdir)
            side = 1.0 if d[0] * sg > 0 else 0.0
            L = rng.uniform(1.9, 2.6) * s * spread * (1.0 + 0.3 * side) * (0.85 + 0.3 * abs(math.sin(math.radians(ang))))
            z = rng.uniform(-0.8, 0.8)
            p0 = F + np.array([rng.uniform(-0.5, 0.5) * rF, rng.uniform(-0.6, 0.0) * rF])
            self._branch(rng, p0, d, L, rF * math.sqrt(w[j]) * 1.08, 1, z)
        if big and rng.random() < 0.7:
            i = int(rng.uniform(0.55, 0.75) * (len(T) - 1))
            d = _unit([sg * 0.9, rng.uniform(0.25, 0.5)])
            self._branch(rng, T[i], d, rng.uniform(1.4, 2.0) * s * spread, rt[i] * 0.5, 1, rng.uniform(0.2, 0.9))
        self._crown_light()

    # ------------------------------------------------------------------ branches
    def _branch(self, rng, p0, d, L, r0, depth, z, mid=None):
        s = self.s
        # blossom mass id: every branch up to depth 2 starts its own mass, finer branches join their parent's
        if depth <= 2 or mid is None:
            self._mid = getattr(self, '_mid', -1) + 1
            mid = self._mid
        perp = np.array([-d[1], d[0]])
        up = np.array([0.0, 1.0])
        # arching: rises then the outer part levels / droops (more for thin outer branches)
        rise = rng.uniform(0.05, 0.16) if depth <= 2 else rng.uniform(0.0, 0.08)
        drop = (rng.uniform(0.0, 0.14) if depth <= 1 else rng.uniform(0.06, 0.26)) * self.droop
        b1, b2 = rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1)
        p1 = p0 + d * L / 3 + perp * L * b1 + up * L * rise
        p2 = p0 + d * 2 * L / 3 + perp * L * b2 + up * L * rise * 0.7
        p3 = p0 + d * L - up * L * drop * (1.0 if d[1] < 0.75 else 0.3)
        nn = max(8, int(22 * L / s))
        P = _cbez(p0, p1, p2, p3, nn)
        tt = np.linspace(0, 1, nn)
        rmin = 0.006 * s
        r1 = max(rmin, r0 * (0.12 if depth < self.maxd else 0.3))
        r = r0 + (r1 - r0) * tt ** 0.65
        self.wood.append(dict(P=P, r=r, z=z, d=depth))
        # ---- laterals along the branch (alternate sides)
        if depth < self.maxd:
            nl = {1: rng.integers(3, 6), 2: rng.integers(3, 5), 3: rng.integers(2, 4)}.get(depth, 1)
            ts = np.sort(rng.uniform(0.18, 0.92, nl))
            side = rng.choice([-1, 1])
            for tl in ts:
                i = int(tl * (nn - 1))
                tdir = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                a = side * rng.uniform(28, 55)
                side = -side
                cd = _rot(tdir, a)
                if cd[1] < -0.35:
                    cd = _unit([cd[0], -0.35])
                Lc = L * rng.uniform(0.38, 0.6) * (1.0 - 0.45 * tl)
                if Lc < 0.18 * s:
                    continue
                zc = z + rng.uniform(-0.45, 0.45)
                self._branch(rng, P[i], cd, Lc, max(rmin, r[i] * rng.uniform(0.5, 0.7)), depth + 1, zc, mid)
        # ---- blossom clusters strung along the (outer) branch
        dens = self.density
        if depth >= 2 or (depth == 1 and not self.big):
            t0 = 0.3 if depth >= 3 else 0.45
        else:
            t0 = 0.55
        Rb = rng.uniform(0.17, 0.27) * s
        t = t0 + rng.uniform(0, 0.1)
        while t <= 1.0:
            i = int(t * (nn - 1))
            tdir = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
            nrm = np.array([-tdir[1], tdir[0]])
            if nrm[1] < 0:
                nrm = -nrm
            R = Rb * rng.uniform(0.75, 1.2) * (0.8 + 0.35 * t)
            off = nrm * R * rng.uniform(0.0, 0.45) + np.array([rng.normal(0, 0.12), rng.normal(0, 0.1)]) * R
            c = P[i] + off
            if rng.random() < 0.93 * min(1.0, dens):
                self.clusters.append(dict(x=float(c[0]), y=float(c[1]), R=float(R), z=z + rng.uniform(-0.15, 0.2), m=mid,
                                          d=depth))
            t += (R / L) * rng.uniform(0.85, 1.35) / max(dens, 0.4)
        # the twig end: a drooping spray of 1-3 tiny clusters (lacy outer silhouette)
        if depth >= 2:
            tip = P[-1]
            tdir = _unit(P[-1] - P[-3])
            for k in range(int(rng.integers(1, 3))):
                dd = _unit(tdir + np.array([rng.normal(0, 0.35), -rng.uniform(0.1, 0.6)]))
                Lt = rng.uniform(0.12, 0.3) * s
                q = _cbez(tip, tip + dd * Lt * 0.4, tip + dd * Lt * 0.75 - np.array([0, 0.1 * Lt]), tip + dd * Lt, 6)
                self.wood.append(dict(P=q, r=np.full(6, rmin * 0.9), z=z + 0.05, d=depth + 1))
                Rt = Rb * rng.uniform(0.35, 0.6)
                self.clusters.append(dict(x=float(q[-1, 0]), y=float(q[-1, 1] + 0.3 * Rt), R=float(Rt), z=z + 0.1, m=mid,
                                          d=depth + 1))

    # ------------------------------------------------------------------ crown-scale light
    def _crown_light(self):
        cl = self.clusters
        if not cl:
            return
        X = np.array([c['x'] for c in cl])
        Y = np.array([c['y'] for c in cl])
        R = np.array([c['R'] for c in cl])
        cx, cy = np.average(X, weights=R * R), np.average(Y, weights=R * R)
        rx = max(1e-3, np.percentile(np.abs(X - cx), 90))
        ry = max(1e-3, np.percentile(np.abs(Y - cy), 90))
        self.crown = (cx, cy, rx, ry)
        # occlusion by the clusters above (sun is high): count cover in a cone above each cluster
        occ = np.zeros(len(cl))
        for i in range(len(cl)):
            dy = Y - Y[i]
            dx = np.abs(X - X[i] - 0.25 * dy)                 # light comes slightly from the left
            m = (dy > 0.2 * R[i]) & (dx < 0.9 * (R + R[i]) + 0.25 * dy)
            occ[i] = np.sum(np.exp(-dy[m] / (0.9 * self.s)) * (R[m] / R[i]).clip(0.3, 2.0))
        # blossom masses: centre / radius of every mass (for one coherent lit-top / shaded-underside form)
        mids = np.array([c['m'] for c in cl])
        self.masses = {}
        for m in np.unique(mids):
            sel = mids == m
            w_ = R[sel] ** 2
            mx, my = np.average(X[sel], weights=w_), np.average(Y[sel], weights=w_)
            dd = np.hypot(X[sel] - mx, Y[sel] - my) + R[sel]
            mr = max(float(np.percentile(dd, 80)), float(R[sel].max()) * 1.2)
            self.masses[int(m)] = (float(mx), float(my), mr)
        for i, c in enumerate(cl):
            c['nx'] = float((X[i] - cx) / rx)
            c['ny'] = float((Y[i] - cy) / ry)
            c['occ'] = float(occ[i])
            c['back'] = False
            c['mx'], c['my'], c['mr'] = self.masses[int(c['m'])]
        # shadow body: behind every covered cluster a larger, lower back cluster fills the crown interior with
        # the lavender underside mass (the lit clusters in front read as many small heads on one big mass)
        rng = np.random.default_rng(int(1000 * abs(cx) + 7))
        extra = []
        for i, c in enumerate(cl):
            if occ[i] < 0.6 or rng.random() < 0.25:
                continue
            e = dict(c)
            e['x'] = c['x'] + rng.normal(0, 0.25) * c['R']
            e['y'] = c['y'] - rng.uniform(0.25, 0.6) * c['R']
            e['R'] = c['R'] * rng.uniform(1.05, 1.35)
            e['z'] = -2.0 + 0.1 * c['z']
            e['ny'] = float((e['y'] - cy) / ry)
            e['occ'] = float(occ[i]) + 0.8
            e['back'] = True
            extra.append(e)
        self.clusters = cl + extra

    def extent(self):
        xs, ys = [], []
        for w in self.wood:
            P, r = w['P'], w['r']
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for c in self.clusters:
            xs += [c['x'] - 1.3 * c['R'], c['x'] + 1.3 * c['R']]
            ys += [c['y'] - 1.3 * c['R'], c['y'] + 1.3 * c['R']]
        return min(xs), max(xs), min(ys), max(ys)


# ============================================================================ painter

PAL13 = dict(
    hot=(1.03, 0.94, 0.91), lit=(1.0, 0.8, 0.81), mid=(0.96, 0.6, 0.68), shade=(0.78, 0.5, 0.68),
    deep=(0.5, 0.38, 0.64), trans=(1.03, 0.76, 0.7), core=(0.88, 0.4, 0.55), under=(0.6, 0.44, 0.68),
    rim=(1.2, 1.02, 0.9),
    wood=(0.12, 0.1, 0.15), wood_lit=(0.28, 0.23, 0.28), wood_rim=(0.85, 0.68, 0.55), wood_cool=(0.26, 0.27, 0.42),
    lent=(0.28, 0.22, 0.28))


def _floret(x, y, r, ph, kind, rng):
    """closed polygon of a blossom dab in px (float)"""
    th = np.linspace(0, 2 * math.pi, 20 if r < 6 else 30, endpoint=False)
    if kind == 0:          # 5-petal floret silhouette
        rr = r * (0.72 + 0.28 * np.abs(np.cos(2.5 * (th + ph))) ** 0.6)
    elif kind == 1:        # bunch of 2-3 florets: lumpy round
        rr = r * (0.82 + 0.1 * np.cos(3 * th + ph) + 0.08 * np.cos(5 * th + 2 * ph))
    else:                  # small round dab
        rr = r * (0.9 + 0.1 * np.cos(2 * th + ph))
    return np.stack([x + rr * np.cos(th), y + rr * np.sin(th)], 1)


class Painter13:
    def __init__(self, k, sun_dir=(-0.85, -0.5), dab_m=0.055, dab_min=2.4, pal=None, detail=1.0, rim_px=2.0,
                 wood_rim_px=2.0, far=False):
        self.k = float(k)
        L = np.asarray(sun_dir, np.float64)
        L = L / (np.linalg.norm(L) + 1e-9)
        # model coords (y up); the painting light is tipped toward 'from above'
        Lm = np.array([L[0], -L[1]])
        Lm = _unit(Lm + np.array([0.0, 0.8]))
        self.L2 = Lm
        self.L3 = _unit(np.array([Lm[0] * 0.8, Lm[1] * 0.8, 0.6]))
        self.Ls = L                                   # screen-space sun direction (y down)
        self.dab = max(dab_min, dab_m * k)
        p = dict(PAL13)
        if pal:
            p.update(pal)
        self.pal = {kk: np.array(v, np.float32) for kk, v in p.items()}
        self.detail = detail
        self.rim_px = rim_px
        self.wood_rim_px = wood_rim_px
        self.far = far

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _col(c):
        c = np.clip(np.asarray(c, np.float64) * SC8, 0, 255)
        return (float(c[0]), float(c[1]), float(c[2]))

    def _fill(self, rgb, a, pts, col, lt=cv2.LINE_AA):
        p = [np.round(np.asarray(pts) * 16).astype(np.int32)]
        cv2.fillPoly(rgb, p, self._col(col), lt, 4)
        cv2.fillPoly(a, p, 255, lt, 4)

    # ------------------------------------------------------------------ wood
    def _wood(self, rgb, a, w, to_px, rng):
        P = to_px(w['P'])
        r = w['r'] * self.k
        pal = self.pal
        n = len(P)
        tng = np.gradient(P, axis=0)
        tng = tng / (np.linalg.norm(tng, axis=1, keepdims=True) + 1e-9)
        nrm = np.stack([-tng[:, 1], tng[:, 0]], 1)
        # which side faces the sun (screen space)
        sd = float(np.mean(nrm @ self.Ls))
        ns = nrm if sd >= 0 else -nrm
        rr = np.maximum(r, 0.55)
        if rr.max() < 1.3:
            p = [np.round(P * 16).astype(np.int32)]
            th = max(1, int(round(2 * rr.mean())))
            cv2.polylines(rgb, p, False, self._col(pal['wood'] * 1.15), th, cv2.LINE_AA, 4)
            cv2.polylines(a, p, False, 255, th, cv2.LINE_AA, 4)
            return
        L_ = P + ns * rr[:, None]
        R_ = P - ns * rr[:, None]
        self._fill(rgb, a, np.concatenate([L_, R_[::-1]]), pal['wood'])
        # lit plane on the sun side
        if rr.max() > 2.0:
            m_ = P + ns * rr[:, None] * rng.uniform(0.25, 0.45)
            lit = pal['wood_lit'] * (1.0 if w['d'] <= 1 else 0.85)
            self._fill(rgb, a, np.concatenate([L_, m_[::-1]]), lit)
            # cool sky bounce on the shade edge
            b_ = P - ns * rr[:, None] * 0.72
            self._fill(rgb, a, np.concatenate([b_, R_[::-1]]), pal['wood'] * 0.5 + pal['wood_cool'] * 0.5)
            # lenticel bands (short strokes across the axis) on the trunk / scaffolds
            if rr.max() > 5.0 and w['d'] <= 1:
                arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
                sp = max(4.0, rr.mean() * 0.8)
                s_ = rng.uniform(0, sp)
                while s_ < arc[-1]:
                    i = int(np.searchsorted(arc, s_))
                    i = min(i, n - 1)
                    f0 = rng.uniform(-0.3, 0.6)
                    f1 = f0 + rng.uniform(0.12, 0.35)
                    q0 = P[i] + ns[i] * rr[i] * f0
                    q1 = P[i] + ns[i] * rr[i] * f1
                    th = max(1, int(round(rr[i] * rng.uniform(0.05, 0.1))))
                    if rng.random() < 0.5:
                        s_ += sp * rng.uniform(0.6, 1.6)
                        continue
                    c = pal['lent']
                    cv2.line(rgb, tuple(np.round(q0 * 16).astype(int)), tuple(np.round(q1 * 16).astype(int)),
                             self._col(c), th, cv2.LINE_AA, 4)
                    s_ += sp * rng.uniform(0.6, 1.6)
        # thin warm rim on the sun edge
        rp = max(0.8, min(self.wood_rim_px, rr.mean() * 0.3))
        if rr.max() > 1.5:
            q = [np.round((P + ns * (rr[:, None] - rp * 0.5)) * 16).astype(np.int32)]
            cv2.polylines(rgb, q, False, self._col(pal['wood_rim'] * 0.8), max(1, int(round(rp))), cv2.LINE_AA, 4)

    # ------------------------------------------------------------------ blossom cluster
    def _cluster(self, rgb, a, c, to_px, rng):
        pal = self.pal
        k = self.k
        cx, cy = to_px(np.array([[c['x'], c['y']]]))[0]
        R = c['R'] * k
        L3 = self.L3
        # cluster-level tone: crown form + occlusion by clusters above + depth
        crown = c['nx'] * self.L2[0] + c['ny'] * self.L2[1]
        tone = 0.12 + 0.3 * crown - 0.2 * min(c['occ'], 3.0) + 0.1 * np.clip(c['z'], -1, 1) + rng.normal(0, 0.03)
        if c.get('back'):
            tone -= 0.3
        bands = [pal['deep'], pal['shade'], pal['mid'], pal['lit'], pal['hot']]
        # mass form: every dab is lit as a point on its blossom MASS (one coherent lit top / crisp terminator /
        # lavender underside across many small clusters) plus a little of its own cluster's form
        mcx, mcy = to_px(np.array([[c['mx'], c['my']]]))[0]
        mr = c['mr'] * k

        def band(v):
            return int(np.clip(np.floor((v + 0.95) / 0.42), 0, 4))

        # under-body: a darker rounded mass low in the cluster (solid core, sky only at the fringe)
        ub = band(tone - 0.35)
        body = _floret(cx + rng.normal(0, 0.05) * R, cy + 0.14 * R, 0.72 * R, rng.uniform(0, 6), 1, rng)
        self._fill(rgb, a, body, bands[max(ub, 0)] * 0.96 + pal['under'] * 0.04)
        dab = self.dab * rng.uniform(0.9, 1.1)
        if R < 1.3 * dab:
            dab = max(1.2, R * 0.75)
        nd = int(np.clip(2.1 * (R / dab) ** 2 * self.detail, 4, 70))
        rad = R * np.sqrt(rng.uniform(0, 1, nd)) * 0.93
        ang = rng.uniform(0, 2 * math.pi, nd)
        dx = rad * np.cos(ang)
        dy = rad * np.sin(ang) * 0.82 - 0.06 * R
        rd = dab * rng.uniform(0.7, 1.3, nd)
        order = np.argsort(dy)[::-1]                       # bottom (large screen y) first
        for i in order:
            nx, ny = dx[i] / R, -dy[i] / R
            nz = math.sqrt(max(0.0, 1 - min(1.0, nx * nx + ny * ny)))
            loc = nx * L3[0] + ny * L3[1] + nz * L3[2]
            mx_, my_ = (cx + dx[i] - mcx) / mr, -(cy + dy[i] - mcy) / mr
            mq = mx_ * mx_ + my_ * my_
            if mq > 1.0:
                mx_, my_ = mx_ / math.sqrt(mq), my_ / math.sqrt(mq)
            mz = math.sqrt(max(0.0, 1 - min(1.0, mq)))
            mloc = mx_ * L3[0] + my_ * L3[1] + mz * L3[2]
            v = 0.95 * mloc + 0.3 * loc - 0.32 + tone + rng.normal(0, 0.03)
            b = band(v)
            col = bands[b]
            rim = math.hypot(nx, ny)
            # warm transmitted light: thin sun-side fringe dabs in shade glow peach-pink
            sunside = -(nx * self.L2[0] + ny * self.L2[1]) < -0.2
            if b <= 2 and rim > 0.7 and sunside:
                col = col * 0.45 + pal['trans'] * 0.55
            if b == 2 and rng.random() < 0.18:
                col = col * 0.7 + pal['core'] * 0.3               # pink blossom centres
            kind = 0 if (rd[i] > 4.5 and rng.random() < 0.7) else (1 if rd[i] > 3 else 2)
            self._fill(rgb, a, _floret(cx + dx[i], cy + dy[i], rd[i], rng.uniform(0, 6), kind, rng), col)

    # ------------------------------------------------------------------ tree
    def paint(self, tree, rng, margin=10):
        k = self.k
        margin = int(margin + 2 * self.dab)
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * margin
        ox = -x0 * k + margin
        oy = y1 * k + margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        rgb = np.zeros((H, W, 3), np.uint8)
        a = np.zeros((H, W), np.uint8)
        wm = np.zeros((H, W), np.uint8)
        # fine wood (depth >= 3) sits deep inside the blossom: it only shows at the lacy silhouette and in the sky
        # gaps, never as chopped dark dashes across the lit masses
        items = [(w['z'] - (1.2 if w['d'] >= 2 else 0.02), 0, i) for i, w in enumerate(tree.wood)]
        items += [(c['z'], 1, i) for i, c in enumerate(tree.clusters)]
        items.sort(key=lambda q: (q[0], q[1]))
        for z, kind, i in items:
            if kind == 0:
                self._wood(rgb, a, tree.wood[i], to_px, rng)
                # wood coverage (for the rims) - drawn crudely, cleared by later blossom
                P = to_px(tree.wood[i]['P'])
                r = max(1, int(round(2 * float(np.mean(tree.wood[i]['r'])) * k)))
                cv2.polylines(wm, [np.round(P * 16).astype(np.int32)], False, 255, r, cv2.LINE_8, 4)
            else:
                c = tree.clusters[i]
                self._cluster(rgb, a, c, to_px, rng)
                cx, cy = to_px(np.array([[c['x'], c['y']]]))[0]
                cv2.circle(wm, (int(cx), int(cy)), int(c['R'] * k * 0.9), 0, -1)
        A = a.astype(np.float32) / 255.0
        C = rgb.astype(np.float32) / SC8               # premultiplied (AA over empty = premultiplied)
        # warm rim on the sun-facing silhouette (sky on the sun side)
        rp = max(1.0, self.rim_px)
        sx, sy = self.Ls
        M = np.float32([[1, 0, sx * rp], [0, 1, sy * rp]])
        As = cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT)
        edge = np.clip(A - As, 0, 1)
        edge = cv2.GaussianBlur(edge, (0, 0), max(0.5, rp * 0.35))
        blos = 1.0 - (wm > 0).astype(np.float32) * 0.6
        kr = np.clip(edge * 1.3, 0, 1) * blos * 0.65
        C = C + (self.pal['rim'][None, None, :] * A[..., None] - C) * kr[..., None]
        card = np.concatenate([C, A[..., None]], -1).astype(np.float32)
        return card, ox, oy
