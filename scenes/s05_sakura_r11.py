"""s05_sakura round 11: cherry trees repainted for the final-panel 'cotton-ball canopy' notes.

Tree11 (model, metres, x right / y up, trunk foot at the origin)
  * trunk: tapered cubic with a root flare, three curve styles (C-bow, S, straight lean) and a random lean
    amount, so neighbouring trunks do not repeat the same banana curve;
  * 2-3 scaffold limbs fork from the crown base at different angles, each forks again once or twice; the
    blossom clumps sit on the limb ends, so every clump is visibly fed by wood and the limbs cross the sky
    gaps between clumps;
  * 5-8 clumps per tree, each built from 2-3 FLATTENED horizontal tiers (wide, low ellipses with a flat,
    tasselled underside and a sagging outer end) -> irregular drooping masses, not balls;
  * small blossom bunches sit directly on the limbs in the gaps.

Painter11 (card painter: paint(tree, rng) -> premultiplied ss-card, ox, oy)
  * each tier is painted in flat value PLANES: the distance from the sun-facing silhouette, measured along
    the light direction, decides the plane -> a warm pink-white lit top-left plane whose terminator is a
    crisp copy of the tier's own top edge, broken into blossom-sized steps, a mid-pink body and a cool
    mauve underside (no radial airbrush gradient);
  * the tiers overlap back-to-front so each lower tier's lit top cuts cleanly across the mauve underside of
    the tier behind it;
  * silhouette detail: 5-petal florets of varied size (dense on the sun side, sparse on the shadow side),
    sprays of blossoms on thin twigs poking out beyond the silhouette, drooping sprays on the underside,
    a thin warm rim on sun-facing edges, a crisp cast shadow down-right;
  * wood: dark plum silhouettes with a crisp lighter warm sun-side plane (bark value break), a 2-3 px
    warm rim on the sun-facing edge, cool bounce on the far edge, lenticel dashes and fork knots.
"""
import math

import numpy as np
import cv2


# ============================================================================ helpers

def _qbez(p0, p1, p2, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = (np.asarray(p, np.float64) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def _cbez(p0, p1, p2, p3, n):
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, np.float64) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3


def _rot(v, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


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


def dir_dist(mask, L, maxd, step):
    """distance (px) from each pixel to where a ray toward L first leaves the (soft) mask"""
    n = max(2, int(math.ceil(maxd / step)))
    inside = mask.copy()
    acc = np.zeros_like(mask)
    for i in range(1, n + 1):
        sm = _sample(mask, L[0] * i * step, L[1] * i * step)
        np.minimum(inside, sm, out=inside)
        acc += inside
    return acc * step


def _floret_poly(x, y, r, ph, sc):
    th = np.linspace(0, 2 * math.pi, 25, endpoint=False)
    if r > 3.0:
        rr = r * (0.58 + 0.42 * np.abs(np.cos(2.5 * (th + ph))) ** 0.6)
    else:
        rr = np.full_like(th, r)
    return np.round(np.stack([x + rr * np.cos(th), y + rr * np.sin(th)], 1) * sc).astype(np.int32)


# ============================================================================ model

class Tree11:
    def __init__(self, rng, lean=-1.0, scale=1.0, big=True, n_target=None, spread=1.0, hmul=1.0, csize=1.0,
                 bunches=True):
        s = scale
        self.s = s
        self.lean = lean
        self.wood = []           # (P (n,2), r (n,), kind) kind 0 trunk, 1 limb, 2 twig/sub-limb, 3 back limb
        self.clumps = []
        sg = 1.0 if lean >= 0 else -1.0
        hf = rng.uniform(1.45, 2.15) * s * hmul
        la = rng.uniform(0.1, 0.62)
        style = int(rng.integers(0, 3))
        Fx = sg * la * hf
        if style == 0:                     # C-bow: rises nearly upright, then bends over
            c1x, c2x = sg * 0.02 * hf, Fx * 0.55
        elif style == 1:                   # S: kicks back first
            c1x, c2x = -sg * 0.14 * hf, Fx * 1.15
        else:                              # straight lean
            c1x, c2x = Fx * 0.33, Fx * 0.67
        T = _cbez([0.0, -0.12 * s], [c1x, 0.35 * hf], [c2x, 0.7 * hf], [Fx, hf], 32)
        f = np.linspace(0, 1, len(T))
        r0 = rng.uniform(0.16, 0.23) * s
        yb = T[:, 1] + 0.12 * s
        rt = r0 * (1.0 - 0.36 * f) * (1.0 + 0.35 * np.exp(-np.maximum(yb, 0) / (0.22 * s)))
        self.wood.append((T, rt, 0))
        F = T[-1].copy()
        self.F = F
        self.hf = hf
        tdir = T[-1] - T[-5]
        tdir = tdir / (np.linalg.norm(tdir) + 1e-9)
        # ---- scaffold limbs
        nsc = 3 if big else int(rng.choice([2, 3], p=[0.4, 0.6]))
        if nsc == 3:
            angs = [rng.uniform(-38, -18), rng.uniform(4, 22), rng.uniform(42, 64)]
        else:
            angs = [rng.uniform(-28, -6), rng.uniform(30, 58)]
        rF = float(rt[-1])
        terms = []                         # (point, limb length, kind)
        self._limbs = []
        for j, ang in enumerate(angs):
            d = _rot(tdir, -sg * ang)      # positive angle -> toward the lean side
            # keep limbs from pointing downward
            if d[1] < 0.15:
                d = np.array([d[0], 0.15])
                d = d / np.linalg.norm(d)
            Lm = rng.uniform(1.35, 2.2) * s * spread * (1.2 if abs(ang) > 40 else 1.0)
            p0 = F + rng.normal(0, 0.03, 2) * s
            end = p0 + d * Lm
            perp = np.array([-d[1], d[0]])
            ctrl = p0 + d * Lm * 0.5 + perp * Lm * rng.uniform(-0.1, 0.1) + np.array([0, 0.12 * Lm])
            P = _qbez(p0, ctrl, end, 26)
            tt = np.linspace(0, 1, len(P))
            rs = rF * rng.uniform(0.7, 0.86)
            r = rs + (0.045 * s - rs) * tt ** 0.85
            back = (j == 0 and rng.random() < 0.5)
            self.wood.append((P, r, 3 if back else 1))
            self._limbs.append((P, r))
            terms.append((end, Lm, back))
            # sub-forks
            for q in range(int(rng.integers(1, 3))):
                tp = rng.uniform(0.45, 0.72)
                i = int(tp * (len(P) - 1))
                fp_, fr_ = P[i], r[i]
                sd = _rot(d, rng.choice([-1, 1]) * rng.uniform(16, 32))
                if sd[1] < 0.1:
                    sd = _rot(d, -np.sign(sd[0] - d[0] + 1e-9) * 30)
                Ls = Lm * rng.uniform(0.42, 0.66)
                e2 = fp_ + sd * Ls
                c2 = fp_ + (sd * 0.6 + d * 0.4) * Ls * 0.5 + np.array([0, 0.1 * Ls])
                Q = _qbez(fp_, c2, e2, 16)
                r2 = fr_ * 0.75 + (0.03 * s - fr_ * 0.75) * np.linspace(0, 1, len(Q)) ** 0.8
                self.wood.append((Q, r2, 3 if back else 2))
                self._limbs.append((Q, r2))
                terms.append((e2, Ls, back))
        # ---- clumps on the limb ends (with a few rejected / shrunk so real sky gaps stay open)
        n_t = n_target or (int(rng.integers(6, 9)) if big else int(rng.integers(5, 8)))
        order = sorted(range(len(terms)), key=lambda i: -terms[i][1])
        cen = []
        for oi in order:
            p, Lm, back = terms[oi]
            R = rng.uniform(0.75, 1.05) * s * csize
            c = p + np.array([rng.normal(0, 0.08) * R, rng.uniform(0.12, 0.25) * R])
            # too close to an existing clump: shrink; if it would merge, the limb end is already hidden in it
            dmin = min([math.hypot((c[0] - q[0]) / 1.35, (c[1] - q[1]) / 0.65) / (R + Rq) for q, Rq in cen] or [9])
            if dmin < 0.45:
                # merged into a neighbour: the limb end still finishes in a blossom bunch, never a bare stub
                self.clumps.append(self._make_clump(rng, p + np.array([0, 0.05 * R]), 0.3 * R, back, small=True))
                continue
            if dmin < 0.75:
                R *= max(0.55, dmin / 0.75)
            cen.append((c, R))
            self.clumps.append(self._make_clump(rng, c, R, back, small=False))
            if len(cen) >= n_t:
                break
        # fill up along the limbs if too few clumps survived
        tries = 0
        while len(cen) < min(n_t, 5) and tries < 60:
            tries += 1
            P, r = self._limbs[int(rng.integers(len(self._limbs)))]
            p = P[int(rng.uniform(0.55, 0.95) * (len(P) - 1))]
            R = rng.uniform(0.6, 0.85) * s * csize
            c = p + np.array([0, 0.2 * R])
            if all(math.hypot((c[0] - q[0]) / 1.35, (c[1] - q[1]) / 0.6) > 0.85 * (R + Rq) for q, Rq in cen):
                cen.append((c, R))
                self.clumps.append(self._make_clump(rng, c, R, False, small=False))
        # back layer: the highest / furthest-back third
        if len(self.clumps) >= 4:
            sc_ = [c['cy'] - sg * 0.3 * (c['cx'] - F[0]) + rng.normal(0, 0.3 * s) for c in self.clumps]
            for i in np.argsort(sc_)[::-1][:max(1, len(self.clumps) // 4)]:
                self.clumps[i]['back'] = True
        # ---- small blossom bunches directly on the limbs (read in the sky gaps)
        if bunches:
            for P, r in self._limbs:
                for _ in range(int(rng.integers(1, 3))):
                    i = int(rng.uniform(0.3, 0.9) * (len(P) - 1))
                    p = P[i] + np.array([0, r[i] * 1.2])
                    Rb = rng.uniform(0.14, 0.24) * s * csize
                    self.clumps.append(self._make_clump(rng, p, Rb, False, small=True))

    def _make_clump(self, rng, c, R, back, small):
        sg = 1.0 if self.lean >= 0 else -1.0
        cx, cy = float(c[0]), float(c[1])
        if small:
            tiers = [(cx, cy, 1.25 * R, 0.62 * R, 0.1)]
        else:
            tiers = []
            if rng.random() < 0.6:
                tiers.append((cx + rng.uniform(-0.35, 0.25) * R * sg, cy + rng.uniform(0.38, 0.55) * R,
                              rng.uniform(0.8, 1.05) * R, rng.uniform(0.36, 0.46) * R, 0.1))
            tiers.append((cx, cy, rng.uniform(1.25, 1.5) * R, rng.uniform(0.46, 0.58) * R, 0.25))
            if rng.random() < 0.65:
                tiers.append((cx + sg * rng.uniform(0.15, 0.55) * R, cy - rng.uniform(0.34, 0.48) * R,
                              rng.uniform(0.8, 1.1) * R, rng.uniform(0.32, 0.42) * R, 0.45))
        return dict(cx=cx, cy=cy, R=R, tiers=tiers, back=back, small=small, seed=int(rng.integers(1 << 30)))

    def extent(self):
        xs, ys = [], []
        for P, r, _ in self.wood:
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for c in self.clumps:
            for (x, y, rx, ry, dr) in c['tiers']:
                m = 0.45 * c['R']
                xs += [x - rx * 1.2 - m, x + rx * 1.2 + m]
                ys += [y - ry * (1.4 + dr) - m, y + ry * 1.2 + m]
        return min(xs), max(xs), min(ys), max(ys)


# ============================================================================ painter

PAL11 = dict(hot=(1.05, 0.95, 0.94), lit=(1.0, 0.84, 0.88), mid=(0.95, 0.67, 0.8), shade=(0.7, 0.55, 0.78),
             deep=(0.5, 0.4, 0.66), bounce=(0.93, 0.68, 0.76), rim=(1.22, 1.08, 0.98), cast=(0.42, 0.33, 0.58),
             center=(0.9, 0.5, 0.62),
             wood=(0.075, 0.05, 0.07), wood_lit=(0.3, 0.2, 0.17), wood_rim=(1.25, 0.9, 0.62),
             wood_cool=(0.2, 0.2, 0.32), twig=(0.2, 0.13, 0.15))


class Painter11:
    def __init__(self, k, sun_dir=(-0.85, -0.5), fr_m=0.045, fr_min=2.2, rim_px=3.0, wood_rim_px=4.0, pal=None,
                 detail=1.0):
        self.k = float(k)
        L = np.asarray(sun_dir, np.float64)
        self.L = L / (np.linalg.norm(L) + 1e-9)
        # painting light: the sun direction tipped toward 'from above' (lit tops, shaded undersides)
        Lp = self.L + np.array([0.0, -0.75])
        self.Lp = Lp / np.linalg.norm(Lp)
        self.fr = max(fr_min, fr_m * k)
        self.rim_px = rim_px
        self.wood_rim_px = wood_rim_px
        p = dict(PAL11)
        if pal:
            p.update(pal)
        self.pal = {kk: np.array(v, np.float32) for kk, v in p.items()}
        self.detail = detail

    # ------------------------------------------------------------------ wood
    def _wood_mask(self, h, w, wood, to_px, kinds, rmap=None):
        m = np.zeros((h, w), np.uint8)
        SH = 3
        sc = 1 << SH
        k = self.k
        items = [(P, r, kind) for (P, r, kind) in wood if kind in kinds]
        items.sort(key=lambda it: -it[2])          # twigs first, trunk last (radius map: thick wins)
        for P, r, kind in items:
            pp = to_px(P)
            rr = np.maximum(r * k, 0.6)
            for i in range(len(pp) - 1):
                a, b = pp[i], pp[i + 1]
                d = b - a
                ln = math.hypot(*d) + 1e-9
                n = np.array([-d[1], d[0]]) / ln
                quad = np.array([a + n * rr[i], b + n * rr[i + 1], b - n * rr[i + 1], a - n * rr[i]])
                cv2.fillConvexPoly(m, np.round(quad * sc).astype(np.int32), 255, cv2.LINE_AA, SH)
                if rmap is not None:
                    cv2.fillConvexPoly(rmap, np.round(quad).astype(np.int32), float(0.5 * (rr[i] + rr[i + 1])))
            for i in range(4 if kind == 0 else 0, len(pp)):
                cv2.circle(m, tuple(np.round(pp[i] * sc).astype(np.int32)), int(round(rr[i] * sc)), 255, -1,
                           cv2.LINE_AA, SH)
                if rmap is not None and i > 0 and i < len(pp) - 1:
                    cv2.circle(rmap, tuple(np.round(pp[i]).astype(np.int32)), int(round(rr[i])), float(rr[i]), -1)
        return m.astype(np.float32) / 255.0

    def _paint_wood(self, rgb, a, tree, to_px, rng, canopy_y, kinds):
        h, w = a.shape
        pal = self.pal
        rmap = np.zeros((h, w), np.float32)
        m = self._wood_mask(h, w, tree.wood, to_px, kinds, rmap)
        if m.max() <= 0:
            return
        lx, ly = self.L
        mb = (m > 0.5).astype(np.uint8)
        DT = cv2.distanceTransform(mb, cv2.DIST_L2, 5).astype(np.float32)
        DTs = cv2.GaussianBlur(DT, (0, 0), 1.2)
        gx = cv2.Sobel(DTs, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(DTs, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        side = -(gx * lx + gy * ly) / gn                 # outward normal . L
        rl = np.maximum(cv2.dilate(rmap, np.ones((3, 3), np.uint8)), 1.0)
        frac = DT / rl
        yy = np.arange(h, dtype=np.float32)[:, None]
        hk = 0.55 + 0.45 * np.clip((yy - canopy_y) / max(0.1 * h, 4.0), 0.0, 1.0)
        # value variation along the wood: large and soft
        var = _vnoise(h, w, max(8.0, 0.5 * self.k), rng) - 0.5
        col = pal['wood'][None, None, :] * (1.0 + 0.3 * var[..., None])
        # crisp sun-side bark plane (value break at ~35% of the width)
        pl = _ss(0.0, 0.2, side) * (1.0 - _ss(0.62, 0.72, frac)) * (rl > 1.6)
        col = col + (pal['wood_lit'] * (1.0 + 0.25 * var[..., None]) - col) * (pl * hk)[..., None]
        # cool bounce on the far edge
        cb = _ss(0.1, 0.4, -side) * (1.0 - _ss(0.18, 0.35, frac)) * (rl > 2.0)
        col = col + (pal['wood_cool'] - col) * (0.45 * cb)[..., None]
        # lenticels + fissure strokes on the trunk and the thick limbs
        if self.k > 50:
            mk = np.zeros((h, w), np.float32)
            for P, r, kind in tree.wood:
                if kind not in kinds or kind == 2:
                    continue
                pp = to_px(P)
                nlen = int((30 if kind == 0 else 8) * self.detail)
                for _ in range(nlen):
                    j = rng.uniform(0, len(pp) - 1.001)
                    i0 = int(j)
                    p = pp[i0] + (pp[i0 + 1] - pp[i0]) * (j - i0)
                    rr = r[i0] * self.k
                    if rr < 3:
                        continue
                    x0 = p[0] + rng.uniform(-0.75, 0.75) * rr
                    ln = rr * rng.uniform(0.15, 0.45)
                    th = max(1, int(round(rr * rng.uniform(0.03, 0.06))))
                    cv2.line(mk, (int(x0 - ln / 2), int(p[1])), (int(x0 + ln / 2), int(p[1] + rng.normal(0, 0.6))),
                             float(rng.uniform(0.5, 1.0)), th, cv2.LINE_AA)
            col = col * (1.0 - 0.4 * (mk * m)[..., None])
        # rim last: thin, warm, crisp, sun-facing edge only
        rp = self.wood_rim_px
        rk = _ss(0.1, 0.35, side) * (1.0 - _ss(rp - 0.8, rp + 0.6, DT)) * np.clip(rl / 2.5, 0.25, 1.0)
        rk = np.clip(rk * m * hk, 0, 1) * 0.95
        col = col * (1 - rk[..., None]) + pal['wood_rim'][None, None, :] * rk[..., None]
        rgb[:] = rgb * (1 - m[..., None]) + col * m[..., None]
        a[:] = a * (1 - m) + m

    # ------------------------------------------------------------------ one tier
    def _tier_poly(self, t, to_px, rng, small):
        x, y, rx, ry, droop = t
        k = self.k
        sg = 1.0 if self._lean >= 0 else -1.0
        n = 150
        th = np.linspace(0, 2 * math.pi, n, endpoint=False)
        k1 = int(rng.integers(3, 6))
        k2 = int(rng.integers(11, 19))
        r = 1.0 + 0.08 * np.sin(k1 * th + rng.uniform(0, 6.3)) + 0.04 * np.sin(2.3 * k1 * th + rng.uniform(0, 6.3))
        # bunch scallops (outward bumps, inward cusps) - strongest on the top edge
        sc = np.abs(np.sin(0.5 * k2 * th + rng.uniform(0, 6.3))) - 0.64
        up = np.sin(th)
        r = r + (0.07 + 0.04 * np.clip(up, 0, 1)) * sc
        X = rx * r * np.cos(th)
        Y = ry * r * np.where(up > 0, up, 0.7 * up)          # flatter underside
        # tassels hanging from the underside
        tas = np.zeros_like(th)
        for _ in range(int(rng.integers(2, 5))):
            c = rng.uniform(math.pi * 1.15, math.pi * 1.85)
            wdt = rng.uniform(0.08, 0.18)
            tas += rng.uniform(0.1, 0.25) * np.exp(-((th - c) / wdt) ** 2)
        Y = Y - ry * tas * (up < 0)
        # the outer end sags (weighted by blossom)
        u = np.clip(sg * X / rx, 0, None)
        Y = Y - droop * ry * u ** 1.7
        P = np.stack([x + X, y + Y], 1)
        return to_px(P)

    def _cells(self, h, w, fr, rng, dens=1.3):
        """floret-scale piecewise-constant value cells (blossom-sized steps on the terminator)"""
        cm = np.full((h, w), 128, np.uint8)
        n = int(h * w / (math.pi * (1.3 * fr) ** 2) * dens) + 1
        xs = rng.random(n) * w
        ys = rng.random(n) * h
        rr = fr * rng.uniform(0.9, 1.7, n)
        vv = rng.integers(0, 256, n)
        for x, y, r, v in zip(xs, ys, rr, vv):
            cv2.circle(cm, (int(x), int(y)), max(1, int(r)), int(v), -1, cv2.LINE_8)
        return cm.astype(np.float32) / 255.0

    def _paint_tier(self, poly, x0, y0, h, w, tone, rng, cells, low, clump_g, f_c=None, wc=0.55):
        pal = self.pal
        SH = 3
        m8 = np.zeros((h, w), np.uint8)
        pp = np.round((poly - np.array([x0, y0])) * (1 << SH)).astype(np.int32)
        cv2.fillPoly(m8, [pp], 255, cv2.LINE_AA, SH)
        M = m8.astype(np.float32) / 255.0
        if M.max() <= 0:
            return M, None
        # chord of the tier along the painting light
        bx0, by0 = poly.min(0)
        bx1, by1 = poly.max(0)
        rx, ry = 0.5 * (bx1 - bx0), 0.5 * (by1 - by0)
        Lp = self.Lp
        chord = 2.0 / math.sqrt((Lp[0] / max(rx, 1)) ** 2 + (Lp[1] / max(ry, 1)) ** 2)
        step = max(0.75, chord / 30.0)
        dL = dir_dist(M, Lp, chord * 1.05, step)
        f = dL / chord
        if f_c is not None:
            f = wc * f_c + (1.0 - wc) * f
        f = f + 0.14 * (cells - 0.5) + 0.1 * (low - 0.5) - tone + clump_g
        aa = 1.2 / chord
        t_hot = 1.0 - _ss(0.08 - aa, 0.08 + aa, f)
        t_lit = 1.0 - _ss(0.34 - aa, 0.34 + aa, f)
        t_mid = 1.0 - _ss(0.54 - aa, 0.54 + aa, f)
        t_deep = _ss(0.85, 1.25, f)
        col = pal['shade'] + (pal['deep'] - pal['shade']) * t_deep[..., None]
        col = col + (pal['mid'] - col) * t_mid[..., None]
        col = col + (pal['lit'] - col) * t_lit[..., None]
        col = col + (pal['hot'] - col) * (0.85 * t_hot)[..., None]
        # warm bounce along the very bottom edge of the underside
        b = max(1.5, 0.12 * ry)
        bot = np.clip(M - _sample(M, 0.0, b), 0, 1) * (1.0 - t_mid)
        col = col + (pal['bounce'] - col) * (0.4 * bot)[..., None]
        return M, col

    def _clump(self, rgb, a, c, to_px, tone=0.0):
        k = self.k
        pal = self.pal
        rng = np.random.default_rng(c['seed'])
        lx, ly = self.L
        fr = self.fr * (0.75 if c['small'] else 1.0)
        polys = [self._tier_poly(t, to_px, rng, c['small']) for t in c['tiers']]
        allp = np.concatenate(polys, 0)
        ext = 0.5 * c['R'] * k + 3 * fr + 4
        x0 = int(max(0, math.floor(allp[:, 0].min() - ext)))
        x1 = int(min(a.shape[1], math.ceil(allp[:, 0].max() + ext)))
        y0 = int(max(0, math.floor(allp[:, 1].min() - ext)))
        y1 = int(min(a.shape[0], math.ceil(allp[:, 1].max() + ext)))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return
        h, w = y1 - y0, x1 - x0
        cells = self._cells(h, w, fr, rng)
        low = _vnoise(h, w, max(4.0, 0.45 * c['R'] * k), rng)
        acc = np.zeros((h, w, 3), np.float32)
        U = np.zeros((h, w), np.float32)
        tt = [0.04, 0.0, -0.06] if len(polys) == 3 else ([0.03, -0.03] if len(polys) == 2 else [0.0])
        cyp = to_px(np.array([[c['cx'], c['cy']]]))[0][1] - y0
        Yg = np.arange(h, dtype=np.float32)[:, None] - cyp
        clump_g = np.broadcast_to(0.08 * Yg / (c['R'] * k), (h, w))
        # clump-scale light (one coherent lit top-left / mauve underside for the whole clump)
        f_c = None
        if len(polys) > 1:
            m8 = np.zeros((h, w), np.uint8)
            for poly in polys:
                cv2.fillPoly(m8, [np.round((poly - np.array([x0, y0])) * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
            Uc = m8.astype(np.float32) / 255.0
            rx, ry = 0.5 * np.ptp(allp[:, 0]), 0.5 * np.ptp(allp[:, 1])
            Lp = self.Lp
            chc = 2.0 / math.sqrt((Lp[0] / max(rx, 1)) ** 2 + (Lp[1] / max(ry, 1)) ** 2)
            f_c = dir_dist(Uc, Lp, chc * 1.05, max(0.75, chc / 36.0)) / chc
        for i, poly in enumerate(polys):
            Mt, col = self._paint_tier(poly, x0, y0, h, w, tone + tt[min(i, len(tt) - 1)], rng, cells, low,
                                       clump_g, f_c)
            if col is None:
                continue
            acc = acc * (1 - Mt[..., None]) + col * Mt[..., None]
            U = U + Mt * (1 - U)
        if U.max() <= 0:
            return
        # ---------------- silhouette detail
        Ub = (U > 0.5).astype(np.uint8)
        din = cv2.distanceTransform(Ub, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(1 - Ub, cv2.DIST_L2, 5)
        sd = np.where(Ub > 0, -din, dout).astype(np.float32)
        Bl = cv2.GaussianBlur(U, (0, 0), max(2.0, 2.0 * fr))
        gx = cv2.Sobel(Bl, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Bl, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-9
        nx, ny = -gx / gn, -gy / gn                            # outward normal
        facing = nx * lx + ny * ly
        fid = np.zeros((h, w), np.uint16)
        fcov = np.zeros((h, w), np.uint8)
        cols = [np.zeros(3, np.float32)]
        SH = 3
        sc = 1 << SH

        def floret(x, y, r, colr):
            cols.append(np.asarray(colr, np.float32))
            poly = _floret_poly(x, y, r, rng.uniform(0, 6.3), sc)
            cv2.fillPoly(fcov, [poly], 255, cv2.LINE_AA, SH)
            cv2.fillPoly(fid, [poly], len(cols) - 1, cv2.LINE_8, SH)

        def base_col(x, y, inward):
            xi = int(np.clip(x - nx[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] * inward, 0, w - 1))
            yi = int(np.clip(y - ny[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))] * inward, 0, h - 1))
            if U[yi, xi] < 0.5:
                return pal['mid']
            return acc[yi, xi]

        band = np.abs(sd) < 1.2
        ys_, xs_ = np.nonzero(band)
        if len(xs_):
            fw = np.clip(facing[ys_, xs_], -1, 1)
            dens = (0.28 + 0.72 * np.clip((fw + 0.3) / 1.0, 0, 1)) * self.detail
            nf = int(len(xs_) / (0.9 * fr) * 0.8)
            p = dens / dens.sum()
            sel = rng.choice(len(xs_), min(nf, len(xs_)), replace=False, p=p)
            for j in sel:
                x, y = xs_[j] + 0.5, ys_[j] + 0.5
                r = fr * float(np.clip(rng.lognormal(0.0, 0.33), 0.45, 1.9))
                o = r * rng.uniform(-0.3, 0.6)
                x += nx[ys_[j], xs_[j]] * o
                y += ny[ys_[j], xs_[j]] * o
                bc = base_col(x, y, 2.0 * r)
                fwj = fw[j]
                colr = bc * (1.0 + rng.normal(0, 0.035))
                if fwj > 0.2:
                    colr = colr + (pal['lit'] - colr) * 0.35 * min(1.0, fwj)
                floret(x, y, r, colr)
        # sprays on thin twigs poking out beyond the silhouette + drooping sprays below
        twig = np.zeros((h, w), np.uint8)
        nsp = int((c['R'] * k / (6.0 * fr)) * (0.35 if c['small'] else 1.0) * self.detail) + int(rng.integers(0, 2))
        if len(xs_) and nsp > 0:
            wsp = np.clip(facing[ys_, xs_] + 0.6, 0.05, None) + 0.8 * np.clip(-ny[ys_, xs_] - 0.7, 0, None) + \
                0.5 * np.clip(ny[ys_, xs_] - 0.8, 0, None)
            sel = rng.choice(len(xs_), min(nsp, len(xs_)), replace=False, p=wsp / wsp.sum())
            for j in sel:
                x, y = xs_[j] + 0.5, ys_[j] + 0.5
                n_ = np.array([nx[ys_[j], xs_[j]], ny[ys_[j], xs_[j]]])
                hang = n_[1] > 0.6
                d = _rot(n_, rng.uniform(-28, 28))
                if hang:
                    d = np.array([d[0] * 0.4, 1.0])
                    d = d / np.linalg.norm(d)
                ln = fr * rng.uniform(2.2, 4.5)
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
                    u = rng.uniform(0.55, 1.0)
                    pq = st + (en - st) * u + bend * 4 * u * (1 - u) + rng.normal(0, 0.55 * fr, 2)
                    floret(pq[0], pq[1], fr * rng.uniform(0.55, 0.95), bc * (1.0 + rng.normal(0, 0.05)))
        # a few loose single blossoms just off the sun-side edge
        out_b = (sd > 1.0 * fr) & (sd < 2.2 * fr) & (facing > 0.1)
        yo, xo = np.nonzero(out_b)
        if len(xo):
            no = int(len(xo) / (fr * fr) * 0.02 * self.detail)
            for j in rng.choice(len(xo), min(no, len(xo)), replace=False):
                floret(xo[j] + 0.5, yo[j] + 0.5, fr * rng.uniform(0.5, 0.8), pal['lit'] * (1 + rng.normal(0, 0.03)))
        tw_ = twig.astype(np.float32) / 255.0
        acc = acc * (1 - tw_[..., None]) + pal['twig'] * tw_[..., None]
        U = U + tw_ * (1 - U)
        if len(cols) > 1:
            fc = fcov.astype(np.float32) / 255.0
            fidd = cv2.dilate(fid, np.ones((3, 3), np.uint8))
            fid2 = np.where(fid > 0, fid, fidd)
            C = np.stack(cols, 0)[fid2]
            acc = acc * (1 - fc[..., None]) + C * fc[..., None]
            U = U + fc * (1 - U)
            # small pink centres on the larger lit-side florets are left to the hero branch (keeps the canopy
            # edge calm); only a faint value step between neighbouring florets comes from the colour jitter
        # thin warm rim on the sun-facing silhouette (lit paint only)
        rp = self.rim_px
        rim = np.clip(U - _sample(U, lx * rp, ly * rp), 0, 1)
        lum = acc.mean(-1)
        rim = rim * np.clip((lum - 0.78) / 0.1, 0, 1)
        acc = acc + (pal['rim'] - acc) * np.clip(rim * 0.75, 0, 1)[..., None]
        if c['back']:
            acc = acc * 0.86 + pal['shade'] * 0.14
        # crisp cast shadow of the clump onto what is already on the card (down-right, away from the sun)
        so = max(2.0, 0.12 * c['R'] * k)
        cs = _sample(U, lx * so, ly * so - 0.3 * so)
        cs = cv2.GaussianBlur(cs, (0, 0), max(0.7, 0.08 * so))
        sa = a[y0:y1, x0:x1]
        sub = rgb[y0:y1, x0:x1]
        cast = np.clip(cs - U, 0, 1) * sa * 0.5
        sub[:] = sub + (pal['cast'] * np.maximum(sub.mean(-1, keepdims=True) * 1.4, 0.3) - sub) * cast[..., None]
        sub[:] = sub * (1 - U[..., None]) + acc * U[..., None]
        sa[:] = sa * (1 - U) + U

    # ------------------------------------------------------------------ tree
    def paint(self, tree, rng, margin=8):
        k = self.k
        self._lean = tree.lean
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
        big = [c for c in tree.clumps if not c['small']]
        small = [c for c in tree.clumps if c['small']]
        backs = sorted([c for c in big if c['back']], key=lambda c: -c['cy'])
        fronts = sorted([c for c in big if not c['back']], key=lambda c: -c['cy'])
        canopy_y = to_px(np.array([[0.0, tree.hf + 0.2 * tree.s]]))[0][1]
        self._paint_wood(rgb, a, tree, to_px, rng, canopy_y, kinds=(3,))
        for c in backs:
            self._clump(rgb, a, c, to_px, tone=-0.12)
        self._paint_wood(rgb, a, tree, to_px, rng, canopy_y, kinds=(0, 1, 2))
        for c in small:
            self._clump(rgb, a, c, to_px, tone=0.02)
        for c in fronts:
            self._clump(rgb, a, c, to_px)
        card = np.concatenate([rgb * a[..., None], a[..., None]], -1).astype(np.float32)
        return card, ox, oy
