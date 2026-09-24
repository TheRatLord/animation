"""s05_sakura round 6: cherry crowns painted as BRANCH STRUCTURE + a few big blossom masses sitting on it.

Tree3 (geometry, metres, y up, origin at the trunk foot):
  trunk -> 2-4 kinked primary limbs, each rising into the underside of ONE blossom mass, where it forks into a
  fan of twigs (which fork again) radiating to the mass outline; some twig tips poke out past the blossom.
  The silhouette of each mass is the union of lobes placed ALONG those twigs, so the blossom follows the wood.
Painter3 (card, premultiplied, supersampled):
  * dark warm-brown bark behind the blossom (visible below the masses, in the sky gaps between masses, through
    the sky pinholes and as calligraphic twigs at the edges) + limb stretches drawn OVER the cool lavender
    undersides, broken by flower clusters (the wood weaves in and out of the blossom);
  * per mass: a mass-level light progression (lit pink-white top -> mid pink -> cool lavender underside) with
    a smaller lobe-level term so each lobe carries its own lit cap (scalloped, cumulus-like value shapes, no
    camouflage blotches); soft painted band transitions broken by florets; contact shade where a lower lobe
    overlaps an upper one; warm peach transmitted light along the undersides / sun-facing edges; 2-3 px rim;
  * lacy edge: clusters of anti-aliased 5-petal florets with strongly varied sizes (few big, many small) and
    real negative space between them; sky pinholes with twigs crossing them.
"""
import math
import numpy as np
import cv2

import s05_sakura_tree2d as T2
from s05_sakura_tree2d import _noise, _ss, _shift, stamp_florets


def _bez(p0, p1, p2, n):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2 = (np.asarray(p, np.float64) for p in (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t * t * p2


def _kink(P, amp, rng):
    """zig-zag an old cherry limb: alternate lateral offsets at the nodes"""
    if len(P) < 4:
        return P
    d = P[-1] - P[0]
    n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
    sgn = np.where(np.arange(len(P)) % 2 == 0, 1.0, -1.0)
    off = amp * sgn * rng.uniform(0.4, 1.0, len(P))
    off[0] = 0
    off[-1] *= 0.3
    return P + n[None, :] * off[:, None]


class Tree3:
    def __init__(self, rng, lean=-1.0, scale=1.0, n_mass=(3, 4), crown=(4.4, 2.2), fork=(1.5, 2.0), r_trunk=0.2,
                 mass_r=(1.1, 1.5), crown_lift=0.9, sep=1.08, twigs=(4, 6), trunk_lean=0.35):
        s = scale
        self.s = s
        self.limbs = []           # (P (n,2), r (n,)) bark, trunk first
        self.masses = []
        self.tips = []            # twig tips (x, y, mass index)  -> little clusters
        hf = rng.uniform(*fork) * s
        top = np.array([lean * rng.uniform(0.5, 1.0) * trunk_lean * s, hf])
        c1 = np.array([-lean * rng.uniform(0.0, 0.12) * s, hf * 0.45])
        n_t = 12
        pts = _bez([0.0, -0.2], c1, top, n_t)
        pts[1:-1, 0] += rng.normal(0, 0.02 * s, n_t - 2)
        f = np.linspace(0, 1, n_t)
        rad = r_trunk * s * (1.0 - 0.25 * f + 0.5 * np.exp(-f / 0.07))
        self.limbs.append((pts, rad))
        self.r_trunk = r_trunk * s
        # ---- masses spread over the crown, biased to the lean side, no two stacked
        nm = int(rng.integers(n_mass[0], n_mass[1] + 1))
        W_, H_ = crown[0] * s, crown[1] * s
        cen, Rs = [], []
        tries = 0
        while len(cen) < nm and tries < 600:
            tries += 1
            u = rng.uniform(-1, 1)
            x = top[0] + lean * (0.25 * W_ + 0.75 * W_ * u * 0.8)
            y = hf + crown_lift * s + H_ * rng.uniform(0.0, 1.0) * (1 - 0.3 * u * u)
            R = rng.uniform(*mass_r) * s
            if all(math.hypot(x - c[0], (y - c[1]) * 1.15) > sep * (R + r) for c, r in zip(cen, Rs)):
                cen.append((x, y))
                Rs.append(R)
        order = np.argsort([-c[1] for c in cen])          # upper masses first (painted behind)
        cen = [cen[i] for i in order]
        Rs = [Rs[i] for i in order]
        for mi, ((cx, cy), R) in enumerate(zip(cen, Rs)):
            rx, ry = R * rng.uniform(1.1, 1.25), R * rng.uniform(0.78, 0.9)
            # primary limb: trunk top -> the underside of this mass
            j = int(rng.integers(n_t - 3, n_t))
            p0 = pts[j]
            r0 = rad[j] * rng.uniform(0.55, 0.72)
            ent = np.array([cx + rng.normal(0, 0.12 * rx), cy - 0.5 * ry])
            d = ent - p0
            ctrl = p0 + np.array([d[0] * rng.uniform(0.25, 0.45), d[1] * rng.uniform(0.55, 0.8)])
            P = _kink(_bez(p0, ctrl, ent, 9), 0.05 * s, rng)
            ll = np.linspace(0, 1, len(P))
            r = r0 * (1 - 0.42 * ll)
            self.limbs.append((P, r))
            re = float(r[-1])
            core = np.array([cx + rng.normal(0, 0.1 * rx), cy - 0.05 * ry])
            Pc = _kink(_bez(ent, (ent + core) / 2 + [rng.normal(0, 0.1 * s), 0], core, 5), 0.03 * s, rng)
            self.limbs.append((Pc, np.linspace(re, re * 0.75, len(Pc))))
            # fan of twigs from the entry / core to the mass outline; lobes sit along them
            nt = int(rng.integers(twigs[0], twigs[1] + 1))
            angs = np.sort(rng.uniform(-0.25, math.pi + 0.25, nt))
            angs = np.linspace(-0.2, math.pi + 0.2, nt) + rng.normal(0, 0.18, nt)
            blobs = [(float(cx), float(cy - 0.02 * ry), 0.62 * R)]
            for a in angs:
                ex = cx + math.cos(a) * rx * rng.uniform(0.92, 1.12)
                ey = cy + math.sin(a) * ry * rng.uniform(0.9, 1.12) - 0.1 * ry
                st = core if math.sin(a) > -0.05 else (ent + core) / 2
                if rng.random() < 0.3:
                    st = (ent * 0.35 + core * 0.65)
                dd = np.array([ex, ey]) - st
                mid = st + dd * 0.5 + np.array([-dd[1], dd[0]]) * rng.normal(0, 0.12) + [0, 0.08 * s]
                Q = _kink(_bez(st, mid, (ex, ey), 7), 0.025 * s, rng)
                rq = np.linspace(re * rng.uniform(0.45, 0.6), 0.014 * s, len(Q))
                self.limbs.append((Q, rq))
                self.tips.append((float(ex), float(ey), mi))
                # the lobe of blossom carried by this twig
                t_ = rng.uniform(0.55, 0.72)
                bx, by = st + dd * t_
                blobs.append((float(bx), float(by), float(np.linalg.norm(dd) * rng.uniform(0.42, 0.55))))
                # a side twig forking off it
                if rng.random() < 0.8:
                    kq = int(rng.integers(2, 4))
                    q0 = Q[kq]
                    sa = a + rng.choice([-1, 1]) * rng.uniform(0.45, 0.8)
                    Lq = np.linalg.norm(dd) * rng.uniform(0.4, 0.6)
                    q1 = q0 + np.array([math.cos(sa), math.sin(sa)]) * Lq
                    Qs = _kink(_bez(q0, (q0 + q1) / 2 + [0, 0.05 * s], q1, 5), 0.015 * s, rng)
                    self.limbs.append((Qs, np.linspace(rq[kq] * 0.6, 0.01 * s, len(Qs))))
                    self.tips.append((float(q1[0]), float(q1[1]), mi))
            # a flatter under-lobe pair so the underside is a broad lavender shelf, not beads
            for sx_ in (-0.4, 0.4):
                blobs.append((float(cx + sx_ * rx + rng.normal(0, 0.08 * rx)), float(cy - 0.32 * ry),
                              0.45 * R * rng.uniform(0.9, 1.1)))
            self.masses.append(dict(cx=float(cx), cy=float(cy), rx=float(rx), ry=float(ry), R=float(R),
                                    blobs=blobs, seed=int(rng.integers(1 << 30))))

    def extent(self):
        xs, ys = [], []
        for P, r in self.limbs:
            xs += [P[:, 0].min() - r.max(), P[:, 0].max() + r.max()]
            ys += [P[:, 1].min() - r.max(), P[:, 1].max() + r.max()]
        for m in self.masses:
            for (bx, by, br) in m['blobs']:
                xs += [bx - br * 1.3, bx + br * 1.3]
                ys += [by - br * 1.3, by + br * 1.3]
        return min(xs), max(xs), min(ys), max(ys)


PAL = dict(hot=np.array([1.0, 0.79, 0.88], np.float32), lit=np.array([1.0, 0.72, 0.85], np.float32),
           mid=np.array([0.94, 0.6, 0.78], np.float32), lav=np.array([0.7, 0.57, 0.82], np.float32),
           deep=np.array([0.55, 0.45, 0.72], np.float32), trans=np.array([1.02, 0.78, 0.72], np.float32),
           glow=np.array([1.08, 0.84, 0.66], np.float32), rim=np.array([1.14, 1.03, 0.9], np.float32),
           bounce=np.array([0.9, 0.66, 0.7], np.float32))


class _Grid:
    """spatial hash for greedy non-overlap placement"""

    def __init__(self, cell):
        self.c = cell
        self.d = {}

    def ok(self, x, y, r, f):
        ci, cj = int(x // self.c), int(y // self.c)
        for i in (ci - 1, ci, ci + 1):
            for j in (cj - 1, cj, cj + 1):
                for (px, py, pr) in self.d.get((i, j), ()):
                    if math.hypot(px - x, py - y) < f * (r + pr):
                        return False
        return True

    def add(self, x, y, r):
        self.d.setdefault((int(x // self.c), int(y // self.c)), []).append((x, y, r))


class Painter3:
    def __init__(self, k, sun_dir=(-0.8, -0.6), fr_m=0.032, fr_min=3.0, pal=None, detail=1.0, holes=(6, 10),
                 hole_m=(0.12, 0.45), rim_px=5.0, bark=((0.12, 0.075, 0.07), (0.4, 0.26, 0.2)), front_bark=1.0,
                 lace=1.0, glow=1.0, tip_clusters=True):
        self.k = k
        L = np.array(sun_dir, np.float64)
        self.L = L / (np.linalg.norm(L) + 1e-9)       # screen direction toward the sun (y down)
        self.fr = max(fr_min, fr_m * k)               # floret radius (card px)
        self.pal = pal or PAL
        self.detail = detail
        self.holes = holes
        self.hole_m = hole_m
        self.rim_px = rim_px
        self.front_bark = front_bark
        self.lace = lace
        self.glow = glow
        self.tip_clusters = tip_clusters
        self._bp = T2.Painter(k, sun_dir=tuple(self.L), bark=bark, bounce=(0.34, 0.3, 0.44), rim=(1.0, 0.72, 0.5))

    # ------------------------------------------------------------------ one mass
    def _mass(self, m, to_px, k, rgbC, aC, vC, sdC, H, W):
        fr = self.fr
        mr = np.random.default_rng(m['seed'])
        Lx, Ly = float(self.L[0]), float(self.L[1])
        bl = []
        for (bx, by, br) in m['blobs']:
            p = to_px(np.array([[bx, by]]))[0]
            bl.append((float(p[0]), float(p[1]), float(br * k)))
        R = m['R'] * k
        x0 = min(b[0] - b[2] for b in bl) - 0.3 * R - 6 * fr
        x1 = max(b[0] + b[2] for b in bl) + 0.3 * R + 6 * fr
        y0 = min(b[1] - b[2] for b in bl) - 0.3 * R - 6 * fr
        y1 = max(b[1] + b[2] for b in bl) + 0.3 * R + 6 * fr
        X0, X1 = int(max(0, x0)), int(min(W, x1))
        Y0, Y1 = int(max(0, y0)), int(min(H, y1))
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            return
        h, w = Y1 - Y0, X1 - X0
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        xx += 0.5
        yy += 0.5
        nz_lo = _noise(h, w, max(4.0, 0.4 * R), mr) - 0.5
        nz_mid = _noise(h, w, max(3.0, 0.1 * R), mr) - 0.5
        # ---- lobes: smooth union SDF + front-most lobe ownership (lower lobes in front)
        kk = 0.06 * R
        acc = np.zeros((h, w), np.float64)
        order = np.argsort([b[1] + 0.2 * b[2] for b in bl])
        own = np.full((h, w), -1, np.int16)
        dl = []
        for i, (bx, by, br) in enumerate(bl):
            d = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2) - br
            dl.append(d)
            acc += np.exp(-np.clip(d, -40 * kk, 40 * kk) / kk)
        sd = (-kk * np.log(np.maximum(acc, 1e-30))).astype(np.float32)
        sd = sd + nz_lo * 0.2 * R + nz_mid * 0.06 * R
        dsh = nz_mid * 0.06 * R
        for i in order:
            own[(dl[i] + dsh) < 0] = i
        body = sd < 0
        own[~body & (own < 0)] = -1
        # lobe-local terms
        hb = np.zeros((h, w), np.float32)
        nlb = np.zeros((h, w), np.float32)
        cs = np.zeros((h, w), np.float32)
        rank = np.empty(len(bl), np.int32)
        rank[order] = np.arange(len(bl))
        for i, (bx, by, br) in enumerate(bl):
            mi = own == i
            if mi.any():
                hb[mi] = ((yy[mi] - (by - br)) / (2 * br))
                nlb[mi] = ((xx[mi] - bx) * Lx + (yy[mi] - by) * Ly) / br
            # contact shade on the lobes BEHIND this one, just outside its edge
            wsh = 0.1 * br
            d = dl[i] + dsh
            behind = (own >= 0) & (rank[np.maximum(own, 0)] < rank[i]) & (d > 0) & (d < wsh)
            if behind.any():
                cs[behind] = np.maximum(cs[behind], (1 - d[behind] / wsh) ** 1.5)
        cs *= body
        # mass-level light: height in the mass + dome normal of the blurred silhouette
        ytop = min(b[1] - b[2] for b in bl)
        ybot = max(b[1] + b[2] for b in bl)
        hf = np.clip((yy - ytop) / max(ybot - ytop, 1.0), 0, 1)
        sig = max(1.0, 0.3 * R)
        fq = max(1.0, sig / 4.0)
        lw, lh = max(4, int(w / fq)), max(4, int(h / fq))
        sm = cv2.resize(_ss(0.0, -1.0, sd).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), sig / fq)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / (8 * fq)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / (8 * fq)
        ndl = np.clip(-(gx * Lx + gy * Ly) * sig * 2.5, -1, 1)
        nup = np.clip(gy * sig * 2.5, -1, 1)                # +1 where the surface faces up
        ndl = cv2.resize(ndl, (w, h), interpolation=cv2.INTER_CUBIC)
        nup = cv2.resize(nup, (w, h), interpolation=cv2.INTER_CUBIC)
        v = (0.5 - 0.62 * (hf - 0.45) + 0.2 * ndl + 0.12 * nup
             + 0.2 * (0.42 - np.clip(hb, 0, 1)) + 0.06 * nlb + 0.16 * nz_lo + 0.03 * nz_mid)
        v = np.where(own >= 0, v, 0.5 - 0.62 * (hf - 0.45) + 0.2 * ndl)
        v = v - 0.06 * cs
        v = v.astype(np.float32)
        pal = self.pal
        e = 0.007
        wl = _ss(0.6 - e, 0.6 + e, v)[..., None]
        wm = _ss(0.33 - e, 0.33 + e, v)[..., None]
        wh = _ss(0.95 - e, 0.95 + e, v)[..., None]
        col = pal['lav'] * (1 - wm) + (pal['mid'] * (1 - wl) + pal['lit'] * wl) * wm
        col = col * (1 - wh) + pal['hot'] * wh
        col = col * (0.95 + 0.08 * np.clip(v, 0, 1.2))[..., None]
        col = col * (1 - 0.06 * cs[..., None]) + pal['deep'] * (0.06 * cs)[..., None]
        # warm bounce from the sunlit path in the lowest lavender
        lav_w = 1 - wm[..., 0]
        wb = lav_w * _ss(0.7, 1.0, hf) * 0.28
        col = col * (1 - wb[..., None]) + pal['bounce'] * wb[..., None]
        # warm transmitted light: backlit petals along the thin under-edges and the sun-facing side
        bw = max(2.2 * fr, 0.06 * R) * self.lace
        sdi = np.minimum(sd, -0.4 * bw)
        edge = np.clip(1 + sdi / (0.14 * R + 2 * bw), 0, 1) ** 1.5
        sside = _ss(-0.2, 0.7, ndl)
        tr = edge * np.clip(0.55 * _ss(0.4, 1.0, hf) + 0.8 * sside, 0, 1) * (1 - 0.6 * wl[..., 0]) * self.glow
        col = col * (1 - 0.55 * tr[..., None]) + pal['trans'] * (0.55 * tr)[..., None]
        gl = np.clip(1 + sdi / (0.1 * R + bw), 0, 1) ** 1.2 * _ss(0.1, 0.8, ndl) * self.glow
        col = col * (1 - 0.45 * gl[..., None]) + pal['glow'] * (0.45 * gl)[..., None]
        col = col.astype(np.float32)
        # ---- coverage: a solid core set in from the outline (irregular) + lace clusters
        nzc = _noise(h, w, max(3.0, 2.2 * fr), mr)
        nzb = _noise(h, w, max(4.0, 9.0 * fr), mr)
        core_in = bw * (0.1 + 0.8 * nzc + 1.4 * _ss(0.35, 0.8, nzb))
        cov = _ss(0.8, -0.8, sd + core_in)
        fl = []
        grid = _Grid(max(4.0, 3 * fr))

        def cluster(px, py, cr, tone, pet=0.3):
            # body dab (2 overlapping strokes) + smaller dabs on its rim; only a few real 5-petal florets poke
            # out of the outer ones
            for j_ in range(2):
                fl.append([px + mr.uniform(-0.35, 0.35) * cr, py + mr.uniform(-0.15, 0.15) * cr,
                           cr * mr.uniform(0.4, 0.58), 0.0, tone, 1.0])
            nf = 2 + int(1.6 * cr / fr + mr.integers(0, 2))
            for q in range(nf):
                a_ = mr.uniform(0, 6.283)
                d_ = cr * math.sqrt(mr.uniform(0.2, 1.0)) * 0.8
                isp = mr.random() < pet
                fl.append([px + math.cos(a_) * d_, py + math.sin(a_) * d_,
                           fr * (mr.uniform(0.8, 1.12) if isp else mr.uniform(0.7, 1.2)),
                           mr.uniform(0, 6.283), tone * mr.uniform(0.99, 1.02), 0.0 if isp else 1.0])

        cell = 1.2 * fr
        gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
        jx = gx_ + mr.uniform(0, cell, gx_.shape)
        jy = gy_ + mr.uniform(0, cell, gy_.shape)
        ix = np.clip(jx.astype(int), 0, w - 1)
        iy = np.clip(jy.astype(int), 0, h - 1)
        s_ = sd[iy, ix]
        band = (s_ < 0.25 * bw) & (s_ > -1.3 * bw - core_in[iy, ix])
        sel = np.nonzero(band)
        pr = mr.permutation(len(sel[0]))
        for q in pr:
            a0, b0 = sel[0][q], sel[1][q]
            px, py = float(jx[a0, b0]), float(jy[a0, b0])
            ss_ = float(s_[a0, b0])
            dep = np.clip((ss_ + bw) / (1.25 * bw), 0, 1)       # 0 inside .. 1 outside
            if mr.random() > 0.92 - 0.55 * dep:
                continue
            u_ = mr.random()
            cr = fr * (0.9 + 3.2 * u_ ** 2.6) * (1.0 - 0.3 * dep)
            if not grid.ok(px, py, cr, 0.78 + 0.2 * dep):
                continue
            grid.add(px, py, cr)
            cluster(px, py, cr, 1.0, pet=0.45 * dep)
        # ---- band edges painted with florets: the lighter band spills over the darker one
        if self.detail > 0 and fr >= 2.5:
            b = ((v > 0.33).astype(np.uint8) + (v > 0.6)).astype(np.uint8)
            rad = max(1, int(round(1.4 * fr)))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
            bmax = cv2.dilate(b, ker)
            edge_ = (bmax > b) & (sd < -1.5 * bw)
            colL = {}
            sgc = max(1.0, 2.0 * fr)
            for L_ in (1, 2):
                mL = (b == L_).astype(np.float32)
                if mL.any():
                    colL[L_] = cv2.GaussianBlur(col * mL[..., None], (0, 0), sgc) /                         np.maximum(cv2.GaussianBlur(mL, (0, 0), sgc), 1e-3)[..., None]
            cell = 1.1 * fr
            gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
            jx = gx_ + mr.uniform(0, cell, gx_.shape)
            jy = gy_ + mr.uniform(0, cell, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            ok = np.nonzero(edge_[iy, ix])
            g2 = _Grid(max(4.0, 3 * fr))
            for q in mr.permutation(len(ok[0])):
                a0, b0 = ok[0][q], ok[1][q]
                px, py = float(jx[a0, b0]), float(jy[a0, b0])
                qx, qy = int(ix[a0, b0]), int(iy[a0, b0])
                L_ = int(bmax[qy, qx])
                if L_ not in colL or mr.random() > 0.85 * self.detail:
                    continue
                cr = fr * (0.8 + 2.4 * mr.random() ** 2.5)
                if mr.random() > 0.55 or not g2.ok(px, py, cr, 0.9):
                    continue
                g2.add(px, py, cr)
                c_ = colL[L_][qy, qx] * mr.uniform(0.995, 1.02)
                # a flat, horizontally drawn dab (2-3 overlapping strokes) + a few small bits on its rim
                for j_ in range(int(mr.integers(2, 4))):
                    fl.append([px + mr.uniform(-0.7, 0.7) * cr, py + mr.uniform(-0.15, 0.15) * cr,
                               cr * mr.uniform(0.32, 0.5), 0.0, 0.0, 1.0, c_])
                for _ in range(1 + int(0.8 * cr / fr)):
                    a_ = mr.uniform(0, 6.283)
                    d_ = cr * mr.uniform(0.5, 0.95)
                    fl.append([px + math.cos(a_) * d_ * 1.2, py + math.sin(a_) * d_ * 0.45,
                               fr * mr.uniform(0.6, 1.0), mr.uniform(0, 6.283), 0.0, 1.0, c_])
            # sparse, low-contrast floret texture inside the lit caps and the lavender (body reads as flowers)
            cell2 = 3.4 * fr
            gy_, gx_ = np.mgrid[0:h:cell2, 0:w:cell2]
            jx = gx_ + mr.uniform(0, cell2, gx_.shape)
            jy = gy_ + mr.uniform(0, cell2, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            gate = _ss(0.45, 0.6, _noise(h, w, max(4.0, 6 * fr), mr))
            ok = (sd[iy, ix] < -2 * bw) & (mr.random(ix.shape) < 0.22 * gate[iy, ix] * self.detail)
            for px, py in zip(jx[ok], jy[ok]):
                vv = v[int(py), int(px)]
                tone = 1.025 if vv > 0.6 else (1.05 if vv < 0.33 else 1.03)
                for _ in range(int(mr.integers(2, 5))):
                    fl.append([px + mr.normal(0, 1.1 * fr), py + mr.normal(0, 1.1 * fr), fr * mr.uniform(0.7, 1.0),
                               mr.uniform(0, 6.283), tone, 0.0])
        # ---- assemble into the local buffers
        rgb = np.ascontiguousarray(col)
        a = np.ascontiguousarray(cov.astype(np.float32))
        if fl:
            rows = []
            for f_ in fl:
                if len(f_) == 7:
                    x_, y_, r_, ph_, _, bl_, c = f_
                    rows.append([x_, y_, r_, ph_, c[0], c[1], c[2], 0.0 if bl_ > 0.5 else 0.03, 0.0, bl_])
                else:
                    x_, y_, r_, ph_, tone, bl_ = f_
                    ix_ = int(np.clip(x_, 0, w - 1))
                    iy_ = int(np.clip(y_, 0, h - 1))
                    c = col[iy_, ix_] * tone
                    eye = 0.0 if bl_ > 0.5 else 0.05
                    rows.append([x_, y_, r_, ph_, c[0], c[1], c[2], eye, 0.0, bl_])
            F = np.clip(np.array(rows, np.float64), -1e6, 1e6)
            F[:, 4:7] = np.clip(F[:, 4:7], 0, 1.15)
            order_ = np.lexsort((-F[:, 1], -F[:, 9]))           # bodies first, then florets top-last
            stamp_florets(rgb, a, np.ascontiguousarray(F[order_]))
        # ---- over into the tree buffers (straight colour rgbC + coverage aC), record value / depth maps
        sub_a = aC[Y0:Y1, X0:X1]
        sub_c = rgbC[Y0:Y1, X0:X1]
        # soft cast shadow of this (front) mass onto masses already painted behind it
        off = 0.08 * R
        shd = _shift(a, -Lx * off, -Ly * off + 0.04 * R)
        shd = cv2.GaussianBlur(shd, (0, 0), max(1.0, 0.02 * R)) * (1 - a) * sub_a
        sub_c[:] = sub_c * (1 - 0.3 * shd[..., None]) + pal['deep'] * (0.3 * shd)[..., None]
        na = a + sub_a * (1 - a)
        sub_c[:] = (rgb * a[..., None] + sub_c * (sub_a * (1 - a))[..., None]) / np.maximum(na, 1e-4)[..., None]
        sub_a[:] = na
        take = a > 0.5
        vC[Y0:Y1, X0:X1][take] = v[take]
        sdC[Y0:Y1, X0:X1][take] = (sd / max(bw, 1.0))[take]

    # ------------------------------------------------------------------ whole tree
    def paint(self, tree, rng, margin=6):
        k = self.k
        fr = self.fr
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * margin
        ox = -x0 * k + margin
        oy = y1 * k + margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        rgbC = np.zeros((H, W, 3), np.float32)
        aC = np.zeros((H, W), np.float32)
        vC = np.full((H, W), 0.5, np.float32)
        sdC = np.full((H, W), 9.0, np.float32)
        for m in tree.masses:
            self._mass(m, to_px, k, rgbC, aC, vC, sdC, H, W)
        pal = self.pal
        Lx, Ly = float(self.L[0]), float(self.L[1])
        # ---- twig-tip clusters: little flower bunches on the wood that pokes out of the silhouette
        if self.tip_clusters and fr >= 2.0:
            mr = np.random.default_rng(int(rng.integers(1 << 30)))
            fl = []
            for (tx_, ty_, mi) in tree.tips:
                p = to_px(np.array([[tx_, ty_]]))[0]
                ix_, iy_ = int(np.clip(p[0], 0, W - 1)), int(np.clip(p[1], 0, H - 1))
                if aC[iy_, ix_] > 0.6:
                    continue
                for j_ in range(int(mr.integers(1, 3))):
                    cr = fr * mr.uniform(1.2, 2.6)
                    px, py = p[0] + mr.normal(0, 0.8 * fr), p[1] + mr.normal(0, 0.8 * fr)
                    # colour: lit pink-white on top of the crown, mid pink / lavender lower down
                    m = tree.masses[mi]
                    hfm = np.clip((m['cy'] + m['ry'] - ty_) / (2 * m['ry']), 0, 1)
                    c = pal['lit'] * (1 - hfm) + pal['mid'] * hfm
                    if hfm > 0.8:
                        c = pal['lav'] * 0.6 + pal['trans'] * 0.4
                    fl.append([px, py, 0.5 * cr, 0.0, c[0], c[1], c[2], 0.0, 0.0, 1.0])
                    for q in range(3 + int(cr / fr)):
                        a_ = mr.uniform(0, 6.283)
                        d_ = cr * math.sqrt(mr.uniform(0.2, 1.0)) * 0.8
                        cc = c * mr.uniform(0.98, 1.04)
                        fl.append([px + math.cos(a_) * d_, py + math.sin(a_) * d_, fr * mr.uniform(0.85, 1.1),
                                   mr.uniform(0, 6.283), cc[0], cc[1], cc[2], 0.14, 0.0, 0.0])
            if fl:
                F = np.array(fl, np.float64)
                F = F[np.lexsort((-F[:, 1], -F[:, 9]))]
                tip_rgb = np.zeros((H, W, 3), np.float32)
                tip_a = np.zeros((H, W), np.float32)
                stamp_florets(tip_rgb, tip_a, np.ascontiguousarray(F))
            else:
                tip_a = None
        else:
            tip_a = None
        # ---- sky pinholes through the crown (twigs cross some of them)
        cut = np.zeros((H, W), np.float32)
        if self.holes and self.holes[1] > 0 and fr >= 2.0:
            solid = (aC > 0.97).astype(np.uint8)
            dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
            ys = np.arange(H, dtype=np.float32)[:, None]
            ms = tree.masses
            ytop = to_px(np.array([[0.0, max(m['cy'] + m['ry'] for m in ms)]]))[0, 1]
            ybot = to_px(np.array([[0.0, min(m['cy'] - m['ry'] for m in ms)]]))[0, 1]
            hfc = (ys - ytop) / max(ybot - ytop, 1.0)
            okm = (solid > 0) & (hfc < 0.75) & (hfc > 0.0)
            if getattr(self, 'vis_top', None) is not None:
                okm &= ys > to_px(np.array([[0.0, self.vis_top]]))[0, 1] + 4 * fr
            # candidates: half on the twigs (the wood shows across the hole), half anywhere
            cand = list(map(tuple, np.argwhere(okm)[rng.permutation(int(okm.sum()))[:3000]]))
            twc = []
            rmax = max(1.2 * fr, 0.3 * 0.5 * k * self.hole_m[0])
            for P, r in tree.limbs[1:]:
                pp = to_px(P)
                for q in range(len(pp)):
                    if r[q] * k > rmax:
                        continue           # thin twigs only: a twig crossing sky, not a dark hole of wood
                    # centre the hole beside the twig so sky shows on both sides of it
                    iy_, ix_ = int(pp[q, 1] + rng.normal(0, 0.8 * fr)), int(pp[q, 0] + rng.normal(0, 0.8 * fr))
                    if 0 <= iy_ < H and 0 <= ix_ < W and okm[iy_, ix_]:
                        twc.append((iy_, ix_))
            twc = [twc[i] for i in rng.permutation(len(twc))]
            mix = []
            for i in range(max(len(cand), len(twc))):
                if i < len(twc):
                    mix.append(twc[i])
                if i < len(cand):
                    mix.append(cand[i])
            nh = int(rng.integers(self.holes[0], self.holes[1] + 1))
            placed = []
            fl = []
            lo, hi = self.hole_m
            for (py, px) in mix:
                if len(placed) >= nh:
                    break
                hr = 0.5 * k * (lo + (hi - lo) * rng.uniform(0, 1) ** 1.4)
                hr = max(hr, 2.5 * fr)
                if dist[py, px] < hr + 2.5 * fr:
                    continue
                if any(math.hypot(px - q[0], py - q[1]) < 1.8 * (hr + q[2]) + 4 * fr for q in placed):
                    continue
                placed.append((px, py, hr))
                pts = [(px + 0.5, py + 0.5, hr)]
                for _ in range(int(rng.integers(2, 5))):
                    a_ = rng.uniform(0, 6.283)
                    d_ = hr * rng.uniform(0.5, 1.0)
                    pts.append((px + 0.5 + math.cos(a_) * d_, py + 0.5 + math.sin(a_) * d_ * 0.75,
                                hr * rng.uniform(0.35, 0.7)))
                ext = int(2.2 * hr + 3 * fr + 4)
                xa, xb = int(max(0, px - ext)), int(min(W, px + ext + 1))
                ya, yb = int(max(0, py - ext)), int(min(H, py + ext + 1))
                gy_, gx_ = np.mgrid[ya:yb, xa:xb].astype(np.float32)
                fld = np.full(gx_.shape, -1e9, np.float32)
                for (hx, hy, rr_) in pts:
                    np.maximum(fld, rr_ - np.sqrt((gx_ + 0.5 - hx) ** 2 + (gy_ + 0.5 - hy) ** 2), out=fld)
                hh_, ww_ = fld.shape
                fld += (_noise(hh_, ww_, max(2.0, 0.5 * hr), rng) - 0.5) * 0.8 * hr
                np.maximum(cut[ya:yb, xa:xb], np.clip(fld + 0.5, 0, 1), out=cut[ya:yb, xa:xb])
                for (hx, hy, rr_) in pts:
                    for _ in range(int(max(3, 2 * math.pi * rr_ / (2.4 * fr)))):
                        a_ = rng.uniform(0, 6.283)
                        d_ = rr_ + fr * rng.uniform(0.2, 0.9)
                        fl.append((hx + math.cos(a_) * d_, hy + math.sin(a_) * d_, fr * rng.uniform(0.8, 1.15),
                                   rng.uniform(0, 6.283), math.cos(a_), math.sin(a_)))
            if placed:
                # backlit ring of petals around every hole
                ring = np.clip(cv2.GaussianBlur(cut, (0, 0), max(1.5, 1.5 * fr)) * 2.0, 0, 1) * (1 - cut)
                rgbC = rgbC * (1 - 0.3 * ring[..., None]) + pal['trans'] * (0.3 * ring)[..., None]
                aC = aC * (1 - cut)
                if fl:
                    Fh = np.array(fl, np.float64)
                    sx_ = np.clip((Fh[:, 0] + Fh[:, 4] * 2.5 * fr).astype(int), 0, W - 1)
                    sy_ = np.clip((Fh[:, 1] + Fh[:, 5] * 2.5 * fr).astype(int), 0, H - 1)
                    c = rgbC[sy_, sx_].astype(np.float64) * rng.uniform(1.0, 1.05, (len(Fh), 1))
                    Fs = np.hstack([Fh[:, :4], np.clip(c, 0, 1.12), np.full((len(Fh), 1), 0.1),
                                    np.zeros((len(Fh), 2))])
                    rgbC = np.ascontiguousarray(rgbC.astype(np.float32))
                    aC = np.ascontiguousarray(aC.astype(np.float32))
                    stamp_florets(rgbC, aC, np.ascontiguousarray(Fs))
        # ---- bark: dark warm brown, darker / cooler where it runs inside the crown shade
        crown = cv2.GaussianBlur(aC, (0, 0), max(2.0, 0.08 * k))
        shade = (0.55 * np.clip(crown * 1.3, 0, 1)).astype(np.float32)
        col_b, av = self._bp.bark_layer(H, W, tree.limbs, to_px, rng, shade)
        # wood in the crown shade: keep it a warm dark brown (not grey-violet)
        col_b = col_b * (1 - 0.4 * shade[..., None]) + np.array([0.2, 0.12, 0.11], np.float32) * (0.4 * shade)[..., None]
        # limbs only (no trunk) for the 'over the underside' pass
        colL, avL = self._bp.bark_layer(H, W, tree.limbs[1:], to_px, rng, shade)
        colL = colL * 0.55 + np.array([0.17, 0.1, 0.1], np.float32) * 0.45
        # ---- compose: bark behind the blossom
        out_rgb = rgbC * aC[..., None] + col_b * (av * (1 - aC))[..., None]
        out_a = aC + av * (1 - aC)
        if tip_a is not None:
            # tip clusters sit on the twigs: in front of the bark, behind the main blossom
            ta = tip_a * (1 - aC)
            out_rgb = out_rgb * (1 - ta[..., None]) + tip_rgb * ta[..., None]
            out_a = out_a + ta * (1 - out_a)
        # ---- limbs showing OVER the lavender underside, woven: broken by flower clusters
        if self.front_bark > 0:
            gate_n = _noise(H, W, max(4.0, 0.9 * k), rng)
            gate = _ss(0.4, 0.3, vC) * _ss(-1.2, -2.2, sdC) * _ss(0.2, 0.4, gate_n) * self.front_bark
            ga = (avL * gate * 0.92).astype(np.float32)
            out_rgb = out_rgb * (1 - ga[..., None]) + colL * ga[..., None]
            # a few clusters over the wood so it dives in and out of the blossom
            if fr >= 2.0:
                mr = np.random.default_rng(int(rng.integers(1 << 30)))
                ys_, xs_ = np.nonzero(ga > 0.5)
                if len(ys_):
                    n = max(1, len(ys_) // int(max(40, 90 * fr)))
                    pick = mr.choice(len(ys_), min(n, len(ys_)), replace=False)
                    fl = []
                    srgb = out_rgb / np.maximum(out_a, 1e-4)[..., None]
                    for q in pick:
                        py, px = ys_[q], xs_[q]
                        c = rgbC[py, px] * 1.02
                        cr = fr * mr.uniform(1.0, 2.0)
                        fl.append([px + 0.5, py + 0.5, 0.55 * cr, 0.0, c[0], c[1], c[2], 0.0, 0.0, 1.0])
                        for _ in range(3):
                            a_ = mr.uniform(0, 6.283)
                            fl.append([px + 0.5 + math.cos(a_) * 0.7 * cr, py + 0.5 + math.sin(a_) * 0.7 * cr,
                                       fr * mr.uniform(0.85, 1.1), mr.uniform(0, 6.283), c[0], c[1], c[2], 0.12,
                                       0.0, 0.0])
                    F = np.array(fl, np.float64)
                    F[:, 4:7] = np.clip(F[:, 4:7], 0, 1.1)
                    srgb = np.ascontiguousarray(srgb.astype(np.float32))
                    sa = np.ascontiguousarray(out_a.astype(np.float32))
                    stamp_florets(srgb, sa, np.ascontiguousarray(F))
                    out_a = sa
                    out_rgb = srgb * sa[..., None]
        # ---- sun-side rim on the blossom silhouette (2-3 px at 1080p) + a soft warm halo just inside
        rw = self.rim_px
        ab = np.clip(aC + (0 if tip_a is None else tip_a), 0, 1)
        sh = _shift(ab, -Lx * rw, -Ly * rw)          # content moved away from the sun
        rim = np.clip(ab - sh, 0, 1)
        # only the edges that really face the sun (the left flank of the crown), not the whole top
        gxa = cv2.Sobel(cv2.GaussianBlur(ab, (0, 0), 2 * rw), cv2.CV_32F, 1, 0, ksize=3)
        gya = cv2.Sobel(cv2.GaussianBlur(ab, (0, 0), 2 * rw), cv2.CV_32F, 0, 1, ksize=3)
        gna = np.sqrt(gxa * gxa + gya * gya) + 1e-6
        facing = -(gxa * Lx + gya * Ly) / gna
        rim = rim * _ss(0.35, 0.8, facing)
        rim = cv2.GaussianBlur(rim, (0, 0), max(0.5, 0.25 * rw)) * ab
        rim = np.clip(rim * 1.3, 0, 1) * (1 - _ss(0.2, 0.6, 1 - vC) * 0.5)
        out_rgb = out_rgb * (1 - 0.75 * rim[..., None]) + pal['rim'] * (0.75 * rim * out_a)[..., None]
        # the trunk foot enters the ground: nothing below the ground line
        cutg = _ss(oy + 0.03 * k + 1.0, oy + 0.03 * k - 1.0, np.arange(H, dtype=np.float32))[:, None]
        card = np.zeros((H, W, 4), np.float32)
        card[..., :3] = out_rgb * cutg[..., None]
        card[..., 3] = out_a * cutg
        return card, ox, oy
