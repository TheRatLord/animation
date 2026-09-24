"""s05_sakura round 10: near-row cherry trees repainted from scratch for the final-panel notes.

Notes addressed: 'the right-hand canopy is procedural blotch mush / uniform confetti with no clustered masses and
no light direction; trunks are bent brown tubes with CG bark and broken angular kinks'.

Tree10 (model, metres, x right / y up, trunk foot at the origin)
  * trunk: one smooth tapered cubic curve with a root flare - no noise kinks;
  * 5-8 big blossom CLUMPS per tree laid out over a leaning crown envelope with real gaps between them
    (sky holes), a few flagged as the shadowed back layer;
  * limbs: gently arcing tapered quadratic curves that grow as a tree (every clump is fed by a limb that
    forks off the trunk or a lower limb) and vanish into the clump undersides; thin twigs reach out into
    the gaps and end in small flower bunches.

Painter10 (card painter, same contract as Painter5/9: paint(tree, rng) -> premultiplied ss-card, ox, oy)
  * each clump = 3-5 overlapping lobes painted back (upper) to front (lower): every lobe boundary is a crisp
    overlapping-mass edge (lit top of the front lobe against the cool underside of the one behind it);
  * value comes from ONE light direction (sun at the top-left): a clump-scale ramp + a lobe-scale ramp +
    the silhouette normal, painted in 4 values (hot pink-white / lit pink-white / mid pink / cool mauve)
    whose boundaries are scalloped by floret-sized bumps (reads as flowers, not noise);
  * crisp small-petal detail ONLY on the silhouettes: the clump edge is built from 5-petal florets (a
    lacy edge with a few detached blossoms), interiors stay as big smooth value masses;
  * crisp cast shadow of each clump on what is behind it (down-right, away from the sun) and a thin warm
    pink-white rim on the sun-facing silhouette;
  * wood: dark cool plum-brown silhouettes (not mid-brown texture) with a faint warmer sun-side plane,
    a thin 1.5-2.5 px warm rim on the sun side (fading up into the canopy shade), faint cool bounce on
    the shadow side and sparse horizontal lenticel marks on the trunk.
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


def _sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _shift(a, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(a, M, (a.shape[1], a.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _vnoise(h, w, cell, rng):
    gh, gw = int(h / cell) + 3, int(w / cell) + 3
    g = rng.random((gh, gw)).astype(np.float32)
    return cv2.resize(g, (int(gw * cell), int(gh * cell)), interpolation=cv2.INTER_CUBIC)[:h, :w]


# ============================================================================ model

class Tree10:
    def __init__(self, rng, lean=-1.0, scale=1.0, big=True, n_clumps=None, lift=2.55, clear=1.15, reach=None,
                 rscale=1.0, u0=-0.12):
        s = scale
        self.s = s
        self.lean = lean
        self.wood = []          # (P (n,2) metres, r (n,) metres, kind) kind 0 trunk, 1 limb, 2 twig
        self.clumps = []        # dict(cx, cy, R, lobes=[(x, y, rx, ry)], back, holes)
        hf = rng.uniform(1.55, 2.0) * s
        F = np.array([lean * rng.uniform(0.25, 0.45) * s, hf])
        # trunk: gentle S, never kinked
        c1 = np.array([-lean * rng.uniform(0.02, 0.1) * s, 0.35 * hf])
        c2 = np.array([F[0] * rng.uniform(0.35, 0.6), 0.72 * hf])
        T = _cbez([0.0, -0.12 * s], c1, c2, F, 28)
        f = np.linspace(0, 1, len(T))
        r0 = rng.uniform(0.2, 0.24) * s
        yb = T[:, 1] + 0.12 * s
        rt = r0 * (1.0 - 0.32 * f) * (1.0 + 0.45 * np.exp(-np.maximum(yb, 0) / (0.2 * s)))
        self.wood.append((T, rt, 0))
        self.F = F
        # ---- crown: clumps along an arch that rises over the trunk and droops out over the path (lean side);
        #      the region right above the fork stays open so the limbs read
        reach = (reach or (4.6 if big else 4.3)) * s
        n = n_clumps or int(rng.integers(6, 9) if big else rng.integers(5, 8))
        cen, Rs = [], []
        tries = 0
        while len(cen) < n and tries < 6000:
            tries += 1
            u = rng.uniform(u0, 1.0)
            x = F[0] + lean * (0.2 + u * reach)
            yc = hf + (lift - 0.9 * max(u, 0.0) ** 2) * s
            y = yc + rng.uniform(-0.75, 0.95) * s
            R = rng.uniform(0.7, 1.1) * s * (1.05 if big else 1.0) * rscale
            if len(cen) < 2:
                R *= 1.12
            # keep the fork / scaffold limbs visible
            if abs(x - F[0]) < 1.2 * s and y - 0.8 * R < hf + clear * s:
                continue
            ok = True
            for (cx, cy), Rr in zip(cen, Rs):
                dd = math.hypot((x - cx) / 1.2, (y - cy) / 0.85)
                if dd < 0.8 * (R + Rr):
                    ok = False
                    break
            if ok:
                cen.append((x, y))
                Rs.append(R)
        # back layer: ~1/3 of the clumps, preferring upper / central ones
        order = np.argsort([-c[1] + rng.normal(0, 0.4 * s) for c in cen])
        nb = max(1, len(cen) // 3)
        back = set(order[:nb].tolist())
        for i, ((cx, cy), R) in enumerate(zip(cen, Rs)):
            rx, ry = 1.18 * R, 0.8 * R
            lobes = [(cx, cy, rx, ry)]
            nl = int(rng.integers(2, 5))
            angs = np.sort(rng.uniform(0.15, math.pi - 0.15, nl))
            for a in angs:
                lr = R * rng.uniform(0.45, 0.66)
                dd = R * rng.uniform(0.5, 0.72)
                lobes.append((cx + math.cos(a) * dd * 1.2, cy + math.sin(a) * dd * 0.8, lr * 1.12, lr * 0.86))
            # a lower front lobe hanging off one side (the clump's own crisp inner edge)
            if rng.random() < 0.7:
                sx = rng.choice([-1, 1])
                lr = R * rng.uniform(0.42, 0.58)
                lobes.append((cx + sx * R * rng.uniform(0.45, 0.75), cy - R * rng.uniform(0.3, 0.45), lr * 1.15,
                              lr * 0.8))
            holes = []
            if R > 0.9 * s and rng.random() < 0.45:
                for _ in range(int(rng.integers(1, 3))):
                    a = rng.uniform(0.3, math.pi - 0.3)
                    holes.append((cx + math.cos(a) * R * 0.45 * rng.uniform(0.6, 1.1),
                                  cy + math.sin(a) * R * 0.35 * rng.uniform(0.5, 1.0) + 0.1 * R,
                                  R * rng.uniform(0.07, 0.11)))
            self.clumps.append(dict(cx=cx, cy=cy, R=R, lobes=lobes, back=i in back, holes=holes, small=False,
                                    seed=int(rng.integers(1 << 30))))
        # ---- limbs: 2-3 scaffold limbs rise steeply from the fork, every other clump is fed by a limb that
        #      forks off the nearest lower node (chains out through the neighbouring clumps, hidden in them)
        def limb(p0, E, rs, re, kind):
            d = E - p0
            ax = abs(d[0])
            # rise steeply out of the fork, then arc over toward the clump (reaching it from above-inside)
            c1 = p0 + np.array([d[0] * rng.uniform(0.08, 0.2), max(0.45 * d[1], 0.0) + rng.uniform(0.35, 0.5) * ax])
            c2 = E + np.array([-d[0] * rng.uniform(0.3, 0.45), rng.uniform(0.12, 0.22) * ax])
            P = _cbez(p0, c1, c2, E, 22)
            tt = np.linspace(0, 1, len(P))
            r = rs + (re - rs) * tt ** 0.8
            self.wood.append((P, r, kind))
            return P, r
        nodes = []
        rF = float(rt[-1])
        big_c = [c for c in self.clumps]
        xs_order = sorted(range(len(big_c)), key=lambda i: lean * (big_c[i]['cx'] - F[0]))
        nsc = 3 if (big and len(big_c) >= 6) else 2
        groups = np.array_split(np.array(xs_order), nsc)
        done = set()
        for g in groups:
            if len(g) == 0:
                continue
            # target: the lowest clump of the group (nearest the fork)
            j = min(g, key=lambda i: math.hypot(big_c[i]['cx'] - F[0], (big_c[i]['cy'] - F[1]) * 0.7))
            c = big_c[j]
            E = np.array([c['cx'], c['cy'] - 0.1 * c['R']])
            P, r = limb(F + rng.normal(0, 0.03, 2) * s, E, rF * rng.uniform(0.62, 0.75), 0.035 * s,
                        3 if c['back'] else 1)
            for p_, r_ in zip(P[3:-2:2], r[3:-2:2]):
                nodes.append((p_, float(r_)))
            done.add(j)
            c['entry'] = E
        idx = np.argsort([math.hypot(c['cx'] - F[0], c['cy'] - F[1]) for c in self.clumps])
        for i in idx:
            c = self.clumps[i]
            if i in done:
                continue
            E = np.array([c['cx'] + rng.normal(0, 0.1) * c['R'], c['cy'] - 0.1 * c['R']])
            best, bs = None, 1e9
            for (p, r) in nodes:
                if p[1] > E[1] + 0.15 * s:
                    continue
                sc = np.linalg.norm(E - p) - 0.8 * r / s
                if sc < bs:
                    bs, best = sc, (p, r)
            if best is None:
                best = (F, rF)
            p0, rp = best
            P, r = limb(p0, E, min(rp * 0.75, 0.09 * s), 0.028 * s, 3 if c['back'] else 1)
            for p_, r_ in zip(P[3:-2:3], r[3:-2:3]):
                nodes.append((p_, float(r_)))
            c['entry'] = E
            # twigs reaching out of the clump into the gaps, ending in small bunches
            for _ in range(int(rng.integers(0, 3) if big else rng.integers(0, 2))):
                a = rng.uniform(-0.3, math.pi + 0.3) if rng.random() < 0.8 else rng.uniform(math.pi, 2 * math.pi)
                st = np.array([c['cx'], c['cy']]) + rng.normal(0, 0.15, 2) * c['R']
                ln = c['R'] * rng.uniform(1.25, 1.7)
                en = st + np.array([math.cos(a) * 1.2, math.sin(a) * 0.8]) * ln
                mid = (st + en) / 2 + np.array([0, -0.12 * ln]) + rng.normal(0, 0.06 * ln, 2)
                Q = _qbez(st, mid, en, 12)
                self.wood.append((Q, np.linspace(0.028 * s, 0.009 * s, len(Q)), 4 if c['back'] else 2))
                self.clumps.append(dict(cx=float(en[0]), cy=float(en[1]), R=c['R'] * rng.uniform(0.2, 0.3),
                                        lobes=None, back=False, holes=[], small=True, seed=int(rng.integers(1 << 30))))
        # small-bunch lobes
        for c in self.clumps:
            if c['lobes'] is None:
                R = c['R']
                lob = [(c['cx'], c['cy'], R * 1.15, R * 0.8)]
                for a in rng.uniform(0.2, math.pi - 0.2, 2):
                    lob.append((c['cx'] + math.cos(a) * R * 0.55, c['cy'] + math.sin(a) * R * 0.4, R * 0.65, R * 0.5))
                c['lobes'] = lob

    def extent(self):
        xs, ys = [], []
        for P, r, _ in self.wood:
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for c in self.clumps:
            for (x, y, rx, ry) in c['lobes']:
                xs += [x - rx * 1.1, x + rx * 1.1]
                ys += [y - ry * 1.1, y + ry * 1.1]
        return min(xs), max(xs), min(ys), max(ys)


# ============================================================================ painter

PAL10 = dict(hot=(1.04, 0.94, 0.94), lit=(1.0, 0.83, 0.88), mid=(0.96, 0.67, 0.79), shade=(0.74, 0.58, 0.78),
             deep=(0.5, 0.38, 0.62), bounce=(0.9, 0.66, 0.74), rim=(1.18, 1.05, 0.95), cast=(0.46, 0.35, 0.6),
             wood=(0.085, 0.06, 0.08), wood_lit=(0.24, 0.15, 0.14), wood_rim=(1.1, 0.78, 0.55),
             wood_cool=(0.2, 0.2, 0.3))


class Painter10:
    def __init__(self, k, sun_dir=(-0.8, -0.6), fr_m=0.07, fr_min=2.2, rim_px=3.0, wood_rim_px=3.5, pal=None,
                 detail=1.0, lz=0.5):
        self.lz = lz
        self.k = float(k)
        L = np.asarray(sun_dir, np.float64)
        self.L = L / (np.linalg.norm(L) + 1e-9)
        self.fr = max(fr_min, fr_m * k)
        self.rim_px = rim_px
        self.wood_rim_px = wood_rim_px
        p = dict(PAL10)
        if pal:
            p.update(pal)
        self.pal = {kk: np.array(v, np.float32) for kk, v in p.items()}
        self.detail = detail

    # ------------------------------------------------------------------ wood
    def _wood_mask(self, h, w, wood, to_px, kinds):
        m = np.zeros((h, w), np.uint8)
        SH = 4
        sc = 1 << SH
        k = self.k
        for P, r, kind in wood:
            if kind not in kinds:
                continue
            pp = to_px(P)
            rr = np.maximum(r * k, 0.6)
            for i in range(len(pp) - 1):
                a, b = pp[i], pp[i + 1]
                d = b - a
                ln = math.hypot(*d) + 1e-9
                n = np.array([-d[1], d[0]]) / ln
                quad = np.array([a + n * rr[i], b + n * rr[i + 1], b - n * rr[i + 1], a - n * rr[i]])
                cv2.fillConvexPoly(m, np.round(quad * sc).astype(np.int32), 255, cv2.LINE_AA, SH)
            for i in range(4 if kind == 0 else 0, len(pp)):
                cv2.circle(m, tuple(np.round(pp[i] * sc).astype(np.int32)), int(round(rr[i] * sc)), 255, -1,
                           cv2.LINE_AA, SH)
        return m.astype(np.float32) / 255.0

    def _paint_wood(self, rgb, a, tree, to_px, rng, canopy_y, kinds=(0, 1, 2)):
        h, w = a.shape
        pal = self.pal
        m = self._wood_mask(h, w, tree.wood, to_px, kinds)
        if m.max() <= 0:
            return
        lx, ly = self.L
        # sun-facing edge band (rim) and the broader sun-side plane
        rp = self.wood_rim_px
        rim = np.clip(m - _shift(m, -lx * rp, -ly * rp), 0, 1)
        plane = np.clip(m - _shift(m, -lx * rp * 2.2, -ly * rp * 2.2), 0, 1)
        plane = cv2.GaussianBlur(plane, (0, 0), rp * 0.9) * m
        cool = np.clip(m - _shift(m, lx * rp * 1.5, ly * rp * 1.5), 0, 1)
        cool = cv2.GaussianBlur(cool, (0, 0), rp * 0.6) * m
        # height factor: the rim fades up into the canopy shade (screen rows above canopy_y)
        yy = np.arange(h, dtype=np.float32)[:, None]
        hk = np.clip((yy - canopy_y) / max(0.12 * h, 4.0), 0.0, 1.0)
        hk = 0.35 + 0.65 * hk
        # subtle value variation along the wood (large, soft) + sparse horizontal lenticels
        var = _vnoise(h, w, max(6.0, 0.35 * self.k), rng) - 0.5
        col = pal['wood'][None, None, :] * (1.0 + 0.25 * var[..., None])
        col = col + (pal['wood_lit'] - pal['wood'])[None, None, :] * (0.4 * plane * hk)[..., None]
        col = col + (pal['wood_cool'] - pal['wood'])[None, None, :] * (0.45 * cool)[..., None]
        # lenticels (trunk only): short dark / light horizontal dashes
        tm = self._wood_mask(h, w, tree.wood, to_px, (0,))
        if self.k > 60 and 0 in kinds:
            len_m = np.zeros((h, w), np.float32)
            T, rT, _ = tree.wood[0]
            pp = to_px(T)
            for _ in range(int(40 * self.detail)):
                j = rng.uniform(0, len(pp) - 1.001)
                i0 = int(j)
                p = pp[i0] + (pp[i0 + 1] - pp[i0]) * (j - i0)
                rr = rT[i0] * self.k
                x0 = p[0] + rng.uniform(-0.8, 0.8) * rr
                ln = rr * rng.uniform(0.15, 0.4)
                th = max(1.0, rr * rng.uniform(0.025, 0.05))
                cv2.line(len_m, (int(x0 - ln / 2), int(p[1])), (int(x0 + ln / 2), int(p[1] + rng.normal(0, 0.5))),
                         float(rng.uniform(0.5, 1.0)), int(round(th)), cv2.LINE_AA)
            len_m = len_m * tm
            col = col * (1.0 - 0.35 * len_m[..., None])
        # rim last (crisp, warm)
        rk = np.clip(rim * 1.3, 0, 1) * hk * 0.95
        col = col * (1 - rk[..., None]) + pal['wood_rim'][None, None, :] * rk[..., None]
        rgb[:] = rgb * (1 - m[..., None]) + col * m[..., None]
        a[:] = a * (1 - m) + m

    # ------------------------------------------------------------------ one clump
    def _florets(self, mask, pts, rad, rng, jit=None):
        """5-petal florets (AA polygons) into an existing uint8 mask; optionally a per-floret random value
        into `jit` (uint8, last floret wins) so neighbouring blossoms separate by a small value step"""
        SH = 3
        sc = 1 << SH
        th = np.linspace(0, 2 * math.pi, 30, endpoint=False)
        for (x, y), r in zip(pts, rad):
            ph = rng.uniform(0, 2 * math.pi)
            if r > 3.5:
                rr = r * (0.55 + 0.45 * np.abs(np.cos(2.5 * (th + ph))) ** 0.5)
            else:
                rr = np.full_like(th, r)
            poly = np.round(np.stack([x + rr * np.cos(th), y + rr * np.sin(th)], 1) * sc).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255, cv2.LINE_AA, SH)
            if jit is not None:
                cv2.fillPoly(jit, [poly], int(rng.integers(40, 255)), cv2.LINE_8, SH)


    def _edge_florets(self, d, fr, rng, dens=0.55, detached=0.035, jit=None):
        """lacy floret edge along the zero set of the signed distance d (px): returns coverage 0..1"""
        h, w = d.shape
        fm = np.zeros((h, w), np.uint8)
        band = (d > -1.5 * fr) & (d < 0.45 * fr)
        ys_, xs_ = np.nonzero(band)
        if len(xs_):
            nf = int(len(xs_) / (fr * fr) * dens * self.detail) + 1
            sel = rng.choice(len(xs_), min(nf, len(xs_)), replace=False)
            self._florets(fm, np.stack([xs_[sel] + rng.random(len(sel)), ys_[sel] + rng.random(len(sel))], 1),
                          fr * rng.uniform(0.6, 1.05, len(sel)), rng, jit)
        if detached > 0:
            outer = (d > 0.7 * fr) & (d < 1.9 * fr)
            yo, xo = np.nonzero(outer)
            if len(xo):
                no = max(1, int(len(xo) / (fr * fr) * detached * self.detail))
                so = rng.choice(len(xo), min(no, len(xo)), replace=False)
                self._florets(fm, np.stack([xo[so] + 0.5, yo[so] + 0.5], 1),
                              fr * rng.uniform(0.55, 0.85, len(so)), rng, jit)
        return fm.astype(np.float32) / 255.0

    def _bunch_edge(self, d0, rl, fr, rng):
        """re-grow a lobe silhouette from flower BUNCHES (cauliflower-like bumps ~0.15 lobe radius) -> signed
        distance (px) of the bunched outline"""
        Rb = max(2.2 * fr, 0.17 * rl)
        if Rb < 3.0:
            return d0
        h, w = d0.shape
        bm = (d0 < -0.3 * Rb).astype(np.uint8) * 255
        band = (d0 > -0.55 * Rb) & (d0 < 0.05 * Rb)
        ys_, xs_ = np.nonzero(band)
        if len(xs_):
            nb = int(len(xs_) / (Rb * Rb) * 0.9) + 1
            sel = rng.choice(len(xs_), min(nb, len(xs_)), replace=False)
            for x_, y_ in zip(xs_[sel], ys_[sel]):
                cv2.circle(bm, (int(x_ * 4), int(y_ * 4)), int(Rb * rng.uniform(0.5, 0.95) * 4), 255, -1, cv2.LINE_AA, 2)
        b = (bm > 127).astype(np.uint8)
        din = cv2.distanceTransform(b, cv2.DIST_L2, 5)
        dout = cv2.distanceTransform(1 - b, cv2.DIST_L2, 5)
        return np.where(b > 0, -din + 0.5, dout - 0.5).astype(np.float32)

    def _clump(self, rgb, a, c, to_px, tone=0.0):
        k = self.k
        pal = self.pal
        rng = np.random.default_rng(c['seed'])
        lx, ly = self.L
        L3 = np.array([lx, -ly, self.lz])            # normals use y up
        L3 = L3 / np.linalg.norm(L3)
        fr = self.fr * (0.8 if c['small'] else 1.0)
        lobes = []
        for (x, y, rx, ry) in c['lobes']:
            p = to_px(np.array([[x, y]]))[0]
            lobes.append((p[0], p[1], rx * k, ry * k))
        cxy = to_px(np.array([[c['cx'], c['cy']]]))[0]
        Rp = c['R'] * k
        ext = 2.5 * fr + 4
        x0 = int(max(0, math.floor(min(l[0] - l[2] for l in lobes) - ext)))
        x1 = int(min(a.shape[1], math.ceil(max(l[0] + l[2] for l in lobes) + ext)))
        y0 = int(max(0, math.floor(min(l[1] - l[3] for l in lobes) - ext)))
        y1 = int(min(a.shape[0], math.ceil(max(l[1] + l[3] for l in lobes) + ext)))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            return
        h, w = y1 - y0, x1 - x0
        X = np.arange(w, dtype=np.float32)[None, :] + x0 + 0.5
        Y = np.arange(h, dtype=np.float32)[:, None] + y0 + 0.5
        # clump-scale form: a dome grown from the blurred silhouette of all lobes (the terminator follows
        # the clump's own outline, one coherent form) blended with a broad ellipsoid
        un = np.zeros((h, w), np.float32)
        for (lxp, lyp, rx, ry) in lobes:
            un = np.maximum(un, (((X - lxp) / rx) ** 2 + ((Y - lyp) / ry) ** 2 < 1.0).astype(np.float32))
        sgm = max(1.5, 0.3 * Rp)
        Bd = cv2.GaussianBlur(un, (0, 0), sgm)
        hd = np.sqrt(np.clip(Bd, 0, 1))
        gxd = cv2.Sobel(hd, cv2.CV_32F, 1, 0, ksize=3) / 8.0 * sgm * 2.2
        gyd = cv2.Sobel(hd, cv2.CV_32F, 0, 1, ksize=3) / 8.0 * sgm * 2.2
        ex_ = np.broadcast_to((X - cxy[0]) / (1.6 * Rp), (h, w))
        ey_ = np.broadcast_to(-(Y - cxy[1]) / (1.2 * Rp), (h, w))
        nC = np.stack([-gxd + 0.5 * ex_, gyd + 0.5 * ey_, np.ones((h, w), np.float32)], 0)
        nC = nC / (np.sqrt((nC * nC).sum(0)) + 1e-6)
        # mid-scale clustering + floret-scale texture (only near value boundaries)
        cl = _vnoise(h, w, max(4.0, 0.5 * Rp), rng) - 0.5
        tex = np.zeros((h, w), np.uint8)
        nt = int(h * w / (fr * fr) * 0.3) + 1
        self._florets(tex, np.stack([rng.random(nt) * w, rng.random(nt) * h], 1), fr * rng.uniform(0.7, 1.1, nt), rng)
        tex = tex.astype(np.float32) / 255.0 - 0.5
        # sky holes carved at the end
        hole = np.zeros((h, w), np.float32)
        for (hx, hy, hr) in c['holes']:
            p = to_px(np.array([[hx, hy]]))[0]
            r = hr * k
            hm = np.zeros((h, w), np.uint8)
            for j in range(4):
                ox_, oy_ = rng.normal(0, 0.7 * r, 2)
                cv2.circle(hm, (int((p[0] + ox_ - x0) * 8), int((p[1] + oy_ - y0) * 8)),
                           int(r * rng.uniform(0.45, 0.9) * 8), 255, -1, cv2.LINE_AA, 3)
            hole = np.maximum(hole, hm.astype(np.float32) / 255.0)
        # lobes painted back (upper) to front (lower): each overlap is a crisp mass edge
        lorder = sorted(range(len(lobes)), key=lambda i: lobes[i][1] - 0.25 * lobes[i][3])
        acc = np.zeros((h, w, 3), np.float32)
        cov = np.zeros((h, w), np.float32)
        sp = pal['shade']
        for n_i, i in enumerate(lorder):
            lxp, lyp, rx, ry = lobes[i]
            ux = (X - lxp) / rx
            uy = (Y - lyp) / ry
            q = np.sqrt(ux * ux + uy * uy)
            d = (q - 1.0) * min(rx, ry)
            d = self._bunch_edge(d, math.sqrt(rx * ry), fr, rng)
            core = np.clip(0.5 - (d + 0.5 * fr), 0, 1)
            jit = np.full((h, w), 128, np.uint8)
            fl = self._edge_florets(d, fr, rng, detached=0.035 if (n_i == 0 or rng.random() < 0.5) else 0.0, jit=jit)
            jv = (jit.astype(np.float32) - 128.0) / 255.0
            ci = np.maximum(core, fl).astype(np.float32)
            # lobe dome normal (flat-radial outside the ellipse)
            nz = np.sqrt(np.clip(1.0 - q * q, 0.0, 1.0))
            s_ = np.minimum(1.0, 1.0 / np.maximum(q, 1e-4))
            nL = np.stack([ux * s_, -uy * s_, nz], 0)
            n = 0.6 * nC + 0.4 * nL
            n = n / (np.sqrt((n * n).sum(0)) + 1e-6)
            ndl = L3[0] * n[0] + L3[1] * n[1] + L3[2] * n[2]
            v = 0.5 + 0.5 * ndl + tone
            v = v + 0.04 * cl + 0.07 * tex * np.clip(1.0 - np.abs(v - 0.5) * 2.5, 0.3, 1.0)
            v = v + 0.05 * np.clip(fl - core, 0, 1)          # edge blossoms catch a little more light
            v = v + 0.09 * jv * fl                            # each edge blossom its own small value step
            # 4 painted values with crisp, floret-scalloped boundaries
            t_s = _sstep(0.47, 0.51, v)
            t_l = _sstep(0.7, 0.74, v)
            t_h = _sstep(0.93, 0.97, v)
            sh = sp + (pal['deep'] - sp) * np.clip((0.47 - v) / 0.3, 0, 1)[..., None]
            col = sh + (pal['mid'] - sh) * t_s[..., None]
            col = col + (pal['lit'] - col) * t_l[..., None]
            col = col + (pal['hot'] - col) * (0.9 * t_h)[..., None]
            col = col * (0.93 + 0.1 * np.clip(v, 0, 1))[..., None]
            # warm bounce on the lower rim of the underside (light off the path)
            bot = np.clip((uy - 0.35) / 0.5, 0, 1) * np.clip((0.4 - v) / 0.2, 0, 1)
            col = col + (pal['bounce'] - col) * (0.35 * bot)[..., None]
            # crisp cast shadow of this lobe on the lobes already painted behind it
            if n_i > 0:
                so = max(1.5, 0.07 * Rp)
                cs = _shift(ci, -lx * so, -ly * so + 0.5 * so)
                cst = np.clip(cs - ci, 0, 1) * cov * 0.45
                acc = acc + (pal['cast'] - acc) * cst[..., None]
            acc = acc * (1 - ci[..., None]) + col * ci[..., None]
            cov = cov + ci * (1 - cov)
        if hole.max() > 0.5:
            hb = (hole > 0.5).astype(np.uint8)
            din = cv2.distanceTransform(hb, cv2.DIST_L2, 5)
            dout = cv2.distanceTransform(1 - hb, cv2.DIST_L2, 5)
            dh = np.where(hb > 0, -din, dout).astype(np.float32)
            flh = self._edge_florets(-dh - 0.6 * fr, fr * 0.9, rng, dens=0.5, detached=0.0)
            hole = hole * (1 - flh)
        cov = (cov * (1 - hole)).astype(np.float32)
        # thin warm rim on the sun-facing silhouette of the whole clump
        rp = self.rim_px
        rim = np.clip(cov - _shift(cov, -lx * rp, -ly * rp), 0, 1)
        vv = acc.mean(-1)
        rim = rim * np.clip((vv - 0.72) / 0.12, 0, 1)
        acc = acc + (pal['rim'] - acc) * np.clip(rim * 0.8, 0, 1)[..., None]
        if c['back']:
            acc = acc * 0.88 + pal['shade'] * 0.12
        # cast shadow of the clump onto what is already on the card (down-right, away from the sun)
        so = max(2.0, 0.1 * Rp)
        cs = _shift(cov, -lx * so, -ly * so + 0.4 * so)
        cs = cv2.GaussianBlur(cs, (0, 0), max(0.7, 0.1 * so))
        cast = np.clip(cs - cov, 0, 1) * a[y0:y1, x0:x1] * 0.5
        sub = rgb[y0:y1, x0:x1]
        sub[:] = sub + (pal['cast'] - sub) * cast[..., None]
        sa = a[y0:y1, x0:x1]
        sub[:] = sub * (1 - cov[..., None]) + acc * cov[..., None]
        sa[:] = sa * (1 - cov) + cov


    # ------------------------------------------------------------------ tree
    def paint(self, tree, rng, margin=8):
        k = self.k
        margin = int(margin + 3 * self.fr)
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
        canopy_y = to_px(np.array([[0.0, tree.F[1] + 0.3 * tree.s]]))[0][1]
        self._paint_wood(rgb, a, tree, to_px, rng, canopy_y, kinds=(3, 4))
        for c in backs:
            self._clump(rgb, a, c, to_px, tone=-0.1)
        self._paint_wood(rgb, a, tree, to_px, rng, canopy_y, kinds=(0, 1, 2))
        for c in small:
            self._clump(rgb, a, c, to_px, tone=0.02)
        for c in fronts:
            self._clump(rgb, a, c, to_px)
        card = np.concatenate([rgb * a[..., None], a[..., None]], -1).astype(np.float32)
        return card, ox, oy
