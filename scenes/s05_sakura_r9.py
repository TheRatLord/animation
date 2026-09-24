"""s05_sakura round 9: Painter9 - cherry crowns and trunks repainted for the art-director notes.

Built on Painter5 (many small AA flower clusters grouped into value-carrying clumps) with these changes:
  * petal-scale silhouettes: every cluster head's outline is a ring of rounded petals separated by V notches
    (not a soft sinusoidal scallop), the crown silhouette gets a fringe of 5-petal florets poking out past the
    edge and small bites cut into it -> no cotton-ball / popcorn outline;
  * clump sizes vary a lot (a few big masses carrying many small bunches) - set from the scene via clump_r;
  * WOOD IS NEVER CHOPPED: the limbs are Chaikin-smoothed tapering curves (no low-poly facets) that run from the
    trunk up through the shadowed back layer of the crown and fork into thin twigs that vanish behind the lit
    clusters; the back layer sits BEHIND the wood, so limbs visibly enter the canopy underside and twigs show as
    dark lines in the sky gaps (no square stubs where a hidden stretch used to be cut away);
  * bark: dark horizontal lenticel bands, vertical fissures, painted plate mottling, a cool blue-grey shadow
    side with sky bounce, and a crisp 2-3 px (screen) warm rim on the sun-facing edge.
"""
import math
import os
import numpy as np
import cv2
from numba import njit

import s05_sakura_crown5 as P5
import s05_sakura_canopy3 as _C3
import s05_sakura_tree2d as T2
from s05_sakura_tree2d import _noise, _ss, _shift, limb_raster


# ============================================================================ petal-edged clusters

@njit(cache=True, inline='always')
def _sdf9(px, py, cx, cy, rc, Hd, h0, h1):
    dx = px - cx
    dy = py - cy
    sy = 0.82 if dy > 0 else 1.0
    d = math.sqrt((dx / 1.08) ** 2 + (dy / sy) ** 2) - 0.6 * rc
    best = -1
    for j in range(h0, h1):
        ex = px - Hd[j, 0]
        ey = py - Hd[j, 1]
        r = math.sqrt(ex * ex + ey * ey)
        hr = Hd[j, 2]
        e = r - hr
        if e < d + 6.0 and hr > 3.0:
            # a ring of rounded petals with V notches between them (petal-scale, up to ~5 px deep)
            th = math.atan2(ey, ex)
            amp = min(0.24 * hr, 5.0)
            s = abs(math.sin(0.5 * Hd[j, 3] * th + Hd[j, 4]))
            e += amp * (1.0 - s) ** 1.6
        if e < d:
            d = e
            best = j
    return d, best


@njit(cache=True)
def paint_clusters9(rgb, a, vl, C, Hd, Lx, Ly, P, cap_amt, und_amt, sh_amt, shc):
    """same contract as crown5.paint_clusters, with petal-notched heads"""
    H, W = a.shape
    for i in range(C.shape[0]):
        cx, cy, rc, Vc = C[i, 0], C[i, 1], C[i, 2], C[i, 3]
        tc, tu = C[i, 4], C[i, 5]
        h0, h1 = int(C[i, 6]), int(C[i, 7])
        ox = -Lx * 0.3 * rc
        oy = -Ly * 0.3 * rc + 0.14 * rc
        ext = 1.25 * rc + abs(ox) + abs(oy) + 3
        x0 = max(0, int(cx - ext))
        x1 = min(W, int(cx + ext) + 1)
        y0 = max(0, int(cy - ext))
        y1 = min(H, int(cy + ext) + 1)
        for y in range(y0, y1):
            for x in range(x0, x1):
                px = x + 0.5
                py = y + 0.5
                d, hb = _sdf9(px, py, cx, cy, rc, Hd, h0, h1)
                cov = min(max(0.5 - d, 0.0), 1.0)
                if cov < 1.0 and a[y, x] > 0.0 and sh_amt > 0:
                    ds, _ = _sdf9(px - ox, py - oy, cx, cy, rc, Hd, h0, h1)
                    s = min(max(0.5 - ds / 1.5, 0.0), 1.0) * sh_amt * (1 - cov)
                    if s > 0:
                        for c in range(3):
                            rgb[y, x, c] = rgb[y, x, c] * (1 - s) + shc[c] * s
                        vl[y, x] -= 0.2 * s
                if cov <= 0.0:
                    continue
                lnc = ((px - cx) * Lx + (py - cy) * Ly) / rc
                upc = -(py - cy) / rc
                loc = 0.55 * lnc + 0.45 * upc
                if hb >= 0:
                    hr = Hd[hb, 2]
                    lnh = ((px - Hd[hb, 0]) * Lx + (py - Hd[hb, 1]) * Ly) / hr
                    uph = -(py - Hd[hb, 1]) / hr
                    loc = 0.6 * loc + 0.4 * (0.55 * lnh + 0.45 * uph)
                cap = min(max((loc - tc) / 0.06, 0.0), 1.0)
                und = min(max((tu - loc) / 0.08, 0.0), 1.0)
                # crisp darker lower hem of every cluster (clusters read as separate painted clumps)
                hem = min(max((-loc - 0.32) / 0.05, 0.0), 1.0)
                v = Vc + cap_amt * cap - und_amt * und - 0.1 * hem
                col = P5._band(v, P)
                old = a[y, x]
                na = cov + old * (1 - cov)
                for c in range(3):
                    rgb[y, x, c] = (col[c] * cov + rgb[y, x, c] * old * (1 - cov)) / na
                a[y, x] = na
                if cov > 0.5:
                    vl[y, x] = v


def _chaikin(P, r, it=3):
    """corner-cutting smoothing of a limb polyline (endpoints kept) -> no low-poly facets at the kinks"""
    P = np.asarray(P, np.float64)
    r = np.asarray(r, np.float64)
    for _ in range(it):
        if len(P) < 3:
            break
        Q = 0.75 * P[:-1] + 0.25 * P[1:]
        R_ = 0.25 * P[:-1] + 0.75 * P[1:]
        rq = 0.75 * r[:-1] + 0.25 * r[1:]
        rr = 0.25 * r[:-1] + 0.75 * r[1:]
        NP = np.empty((2 * len(Q), 2))
        NP[0::2] = Q
        NP[1::2] = R_
        NR = np.empty(2 * len(Q))
        NR[0::2] = rq
        NR[1::2] = rr
        P = np.vstack([P[:1], NP[1:-1], P[-1:]])
        r = np.concatenate([r[:1], NR[1:-1], r[-1:]])
    return P, r


def _tail(P, r, s_, rng):
    """continue a limb past its blunt end as a tapering, gently up-curving twig (never a square stub)"""
    P = np.asarray(P, np.float64)
    r = np.asarray(r, np.float64)
    re = float(r[-1])
    tw = 0.011 * s_
    if re <= 1.6 * tw:
        return P, r
    d = P[-1] - P[-2]
    d = d / (np.linalg.norm(d) + 1e-9)
    Lt = float(np.clip(7.0 * re, 0.2 * s_, 0.6 * s_))
    bend = np.array([-d[1], d[0]]) * rng.uniform(-0.25, 0.25) + np.array([0.0, 0.3])
    n = 6
    tt = np.linspace(0, 1, n + 1)[1:, None]
    Q = P[-1] + d * Lt * tt + bend * Lt * 0.5 * tt * tt
    rq = re + (tw - re) * np.linspace(0, 1, n + 1)[1:] ** 0.7
    return np.vstack([P, Q]), np.concatenate([r, rq])


class Painter9(P5.Painter5):
    def __init__(self, *a, edge_florets=1.0, notch=1.0, wood_over_fill=True, twig_min=0.012, **kw):
        super().__init__(*a, **kw)
        self.edge_florets = edge_florets
        self.notch = notch
        self.wood_over_fill = wood_over_fill
        self.twig_min = twig_min

    def _mass(self, m, to_px, k, rgbC, aC, vC, sdC, H, W):
        mr = np.random.default_rng(m['seed'])
        Lx, Ly = float(self.L[0]), float(self.L[1])
        pal = self.pal
        bl = []
        for (bx, by, br) in m['blobs']:
            p = to_px(np.array([[bx, by]]))[0]
            bl.append((float(p[0]), float(p[1]), float(br * k)))
        R = m['R'] * k
        c0, c1 = self.clump_m
        rmin = max(2.2, c0 * k)
        rmax = max(rmin * 1.3, c1 * k)
        pad = 0.2 * R + max(2.2 * rmin, self.clump_r[1] * k) + 1.5 * rmax + 4
        x0 = min(b[0] - b[2] for b in bl) - pad
        x1 = max(b[0] + b[2] for b in bl) + pad
        y0 = min(b[1] - b[2] for b in bl) - pad
        y1 = max(b[1] + b[2] for b in bl) + pad
        X0, X1 = int(max(0, x0)), int(min(W, x1))
        Y0, Y1 = int(max(0, y0)), int(min(H, y1))
        if X1 - X0 < 4 or Y1 - Y0 < 4:
            return
        h, w = Y1 - Y0, X1 - X0
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(np.float32)
        xx += 0.5
        yy += 0.5
        nz_lo = _noise(h, w, max(4.0, 0.45 * R), mr) - 0.5
        # ---- mass envelope (smooth union of the lobes, softly irregular)
        kk = 0.08 * R
        acc = np.zeros((h, w), np.float64)
        for (bx, by, br) in bl:
            d = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2) - br
            acc += np.exp(-np.clip(d, -40 * kk, 40 * kk) / kk)
        sd = (-kk * np.log(np.maximum(acc, 1e-30))).astype(np.float32) + nz_lo * 0.18 * R
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
        ndl = cv2.resize(np.clip(-(gx * Lx + gy * Ly) * sig * 2.5, -1, 1), (w, h), interpolation=cv2.INTER_CUBIC)
        # big light pattern of the mass (a few large value shapes; mid-scale noise breaks the band edges into
        # whole clusters)
        nz_m = _noise(h, w, max(3.0, 2.5 * rmax), mr) - 0.5
        vm = (0.55 - 0.8 * (hf - 0.42) + 0.3 * ndl + 0.12 * nz_lo + 0.12 * nz_m).astype(np.float32)

        # ---- CLUMPS (0.3-0.55 m flower bunches, the value-carrying shapes) placed inside the envelope with
        #      sky gaps between them in the upper crown; each clump is then built from many small clusters
        C0, C1 = self.clump_r
        Rmin = max(2.2 * rmin, C0 * k)
        Rmax = max(Rmin * 1.2, C1 * k)
        grid = _C3._Grid(max(4.0, 2.2 * Rmax))
        clumps = []
        n_try = int(np.clip(40 * (w * h) / (Rmax * Rmax), 200, 20000))
        px = mr.uniform(0, w, n_try)
        py = mr.uniform(0, h, n_try)
        rr = Rmin + (Rmax - Rmin) * mr.random(n_try) ** 1.3
        ix = np.clip(px.astype(int), 0, w - 1)
        iy = np.clip(py.astype(int), 0, h - 1)
        sdv = sd[iy, ix]
        hfv = hf[iy, ix]
        spacing = 0.62 + 0.26 * (1 - hfv) ** 2 * self.open_top
        ok = np.nonzero(sdv < -0.25 * rr)[0]
        ok = ok[np.argsort(-rr[ok])]
        for i in ok:
            if grid.ok(float(px[i]), float(py[i]), float(rr[i]), float(spacing[i])):
                grid.add(float(px[i]), float(py[i]), float(rr[i]))
                clumps.append((float(px[i]), float(py[i]), float(rr[i])))
        ey, ex = np.nonzero((sd > -0.4 * Rmin) & (sd < 0.1 * Rmin) & (hf < 0.85))
        if len(ey):
            for q in mr.permutation(len(ey))[:600]:
                cx_, cy_ = float(ex[q]) + 0.5, float(ey[q]) + 0.5
                rc_ = Rmin + (Rmax - Rmin) * 0.5 * mr.random()
                if grid.ok(cx_, cy_, rc_, 0.9):
                    grid.add(cx_, cy_, rc_)
                    clumps.append((cx_, cy_, rc_))
        if not clumps:
            return
        clumps.sort(key=lambda c: c[1] + 0.3 * c[2] + mr.uniform(-0.4, 0.4) * c[2])

        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        vloc = np.full((h, w), 0.25, np.float32)
        Pp = P5._pal_arr(pal)
        rows = []
        heads = []

        def add_cluster(cx_, cy_, rc_, Vc, tc):
            nh = int(mr.integers(3, 8)) if rc_ > 4 else 2
            h0 = len(heads)
            for j in range(nh):
                an = mr.uniform(-math.pi * 1.15, math.pi * 0.15) if mr.random() < 0.75 else mr.uniform(0, math.pi)
                hr = rc_ * mr.uniform(0.3, 0.55)
                dd = (rc_ - hr) * mr.uniform(0.75, 1.15)
                heads.append((cx_ + math.cos(an) * dd * 1.1, cy_ + math.sin(an) * dd * 0.9, hr,
                              float(mr.integers(5, 8)), mr.uniform(0, 6.283)))
            rows.append((cx_, cy_, rc_, Vc, tc, -9.0, h0, len(heads)))

        # back layer: dense dark lavender clusters filling the lower core (no smooth fill blob)
        uf = self.under_fill
        cand = np.nonzero((sd.ravel() < -0.6 * Rmin) & (hf.ravel() > 0.35 - 0.25 * uf))[0]
        if len(cand):
            nb = int(min(len(cand), 1.6 * uf * len(cand) / (math.pi * (0.8 * rmax) ** 2)))
            for q in mr.choice(cand, max(nb, 0), replace=False):
                cy_, cx_ = divmod(int(q), w)
                rc_ = rmin + (rmax - rmin) * mr.random()
                v_ = float(np.clip(vm[cy_, cx_] - 0.38, 0.04, 0.3))
                add_cluster(cx_ + 0.5, cy_ + 0.5, rc_, v_, 9.0)
        # clumps: clusters packed in each clump disc; shadow side first, lit side painted over it
        for (Cx, Cy, Rc) in clumps:
            n = int(np.clip(2.6 * (Rc / (0.5 * (rmin + rmax))) ** 2, 3, 110))
            pts = []
            for _ in range(n):
                r_ = Rc * math.sqrt(mr.random()) * 0.85
                an = mr.uniform(0, 2 * math.pi)
                qx = Cx + math.cos(an) * r_ * 1.1
                qy = Cy + math.sin(an) * r_ * 0.8
                rc_ = rmin + (rmax - rmin) * mr.random() ** 1.4
                lx_ = (qx - Cx) / Rc
                ly_ = (qy - Cy) / Rc
                loc = 0.6 * (lx_ * Lx + ly_ * Ly) - 0.5 * ly_
                ic = (int(np.clip(qy, 0, h - 1)), int(np.clip(qx, 0, w - 1)))
                Vc = float(vm[ic]) + self.clump_amt * loc + mr.normal(0, 0.06)
                pts.append((loc, qx, qy, rc_, Vc))
            pts.sort(key=lambda p_: p_[0] + mr.normal(0, 0.12))
            for (loc, qx, qy, rc_, Vc) in pts:
                add_cluster(qx, qy, rc_, Vc, 0.25 + mr.normal(0, 0.08))
        C = np.array(rows, np.float64)
        Hd = np.array(heads, np.float64)
        shc = (pal['lav'] * 0.6 + pal['deep'] * 0.4).astype(np.float64)
        paint_clusters9(rgb, a, vloc, C, Hd, Lx, Ly, Pp, self.cap_amt, self.und_amt, 0.4, shc)

        # ---- flower texture: sparse small 5-petal florets (same local colour, darker eye) on the clusters
        if self.florets and self.fr >= 3.0:
            fr = self.fr
            cell = 2.4 * fr
            gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
            jx = (gx_ + mr.uniform(0, cell, gx_.shape)).ravel()
            jy = (gy_ + mr.uniform(0, cell, gy_.shape)).ravel()
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            sel = (a[iy, ix] > 0.95) & (mr.random(len(ix)) < 0.3)
            if sel.any():
                cc = rgb[iy[sel], ix[sel]].astype(np.float64) * mr.uniform(1.0, 1.04, (int(sel.sum()), 1))
                n_ = int(sel.sum())
                Fm = np.column_stack([jx[sel], jy[sel], fr * mr.uniform(0.6, 0.95, n_), mr.uniform(0, 6.283, n_),
                                      np.clip(cc, 0, 1.12), np.full(n_, 0.22), np.full(n_, 0.1), np.zeros(n_)])
                T2.stamp_florets(rgb, a, np.ascontiguousarray(Fm[np.argsort(Fm[:, 1])]))

        # ---- warm transmitted light on thin under-edges and the sun-facing flank of the mass
        solid = (a > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
        bw = max(1.5, 0.35 * rmin)
        edge = _ss(1.6 * bw, 1.2 * bw, dist)
        sside = _ss(-0.1, 0.7, ndl)
        tr = edge * np.clip(0.55 * _ss(0.5, 1.0, hf) + 0.55 * sside, 0, 1) * self.glow * (vloc < 0.6)
        rgb = rgb * (1 - 0.5 * tr[..., None]) + pal['trans'] * (0.5 * tr)[..., None]
        wb = _ss(0.72, 1.0, hf) * (vloc < 0.3) * 0.25
        rgb = rgb * (1 - wb[..., None]) + pal['bounce'] * wb[..., None]

        # ---- over into the tree buffers (a soft shadow of this mass on the masses behind it)
        sub_a = aC[Y0:Y1, X0:X1]
        sub_c = rgbC[Y0:Y1, X0:X1]
        off = 0.08 * R
        shd = _shift(a, -Lx * off, -Ly * off + 0.04 * R)
        shd = cv2.GaussianBlur(shd, (0, 0), 0.7) * (1 - a) * sub_a
        sub_c[:] = sub_c * (1 - 0.3 * shd[..., None]) + pal['deep'] * (0.3 * shd)[..., None]
        na = a + sub_a * (1 - a)
        sub_c[:] = (rgb * a[..., None] + sub_c * (sub_a * (1 - a))[..., None]) / np.maximum(na, 1e-4)[..., None]
        sub_a[:] = na
        take = a > 0.5
        vC[Y0:Y1, X0:X1][take] = vloc[take]
        sdC[Y0:Y1, X0:X1][take] = -(dist / max(bw, 1.0))[take]


    # ------------------------------------------------------------------ bark
    def _bark(self, h, w, limbs, to_px, rng, shade):
        k = self.k
        segs = []
        axes = []
        self._axes = axes
        trng = np.random.default_rng(71)
        s_ = float(getattr(self, '_tree_s', 1.0))
        for li, (P, r) in enumerate(limbs):
            P, r = _chaikin(P, r)
            if li == 0:
                # trunk top: a short rounded taper that merges into the primary limbs (no spike, no flat top)
                d = P[-1] - P[-2]
                d = d / (np.linalg.norm(d) + 1e-9)
                re = float(r[-1])
                tt = np.linspace(0, 1, 5)[1:]
                P = np.vstack([P, P[-1] + d[None] * (1.2 * re) * tt[:, None]])
                r = np.concatenate([r, re * (1 - 0.55 * tt ** 1.5)])
            else:
                P, r = _tail(P, r, s_, trng)
            pp = to_px(P)
            rr = np.maximum(r * k, self.twig_min * k * 0.5)
            axes.append((pp, rr))
            l0 = 0.0
            for i in range(len(pp) - 1):
                segs.append((pp[i, 0], pp[i, 1], rr[i], pp[i + 1, 0], pp[i + 1, 1], rr[i + 1], l0))
                l0 += float(np.hypot(*(pp[i + 1] - pp[i])))
        segs = np.array(segs, np.float64)
        q = np.full((h, w), 1e9, np.float32)
        nx = np.zeros((h, w), np.float32)
        ny = np.zeros((h, w), np.float32)
        sv = np.zeros((h, w), np.float32)
        lv = np.zeros((h, w), np.float32)
        rv = np.ones((h, w), np.float32)
        av = np.zeros((h, w), np.float32)
        limb_raster(segs, q, nx, ny, sv, lv, rv, av)
        col = np.zeros((h, w, 3), np.float32)
        iy, ix = np.nonzero(av > 0)
        self._rv = rv
        if len(iy) == 0:
            return col, av
        Lx, Ly = float(self.L[0]), float(self.L[1])
        Dm = cv2.distanceTransform((av > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        Ds = cv2.GaussianBlur(Dm, (0, 0), 2.0)
        gxd = cv2.Sobel(Ds, cv2.CV_32F, 1, 0, ksize=3)
        gyd = cv2.Sobel(Ds, cv2.CV_32F, 0, 1, ksize=3)
        gnd = np.sqrt(gxd * gxd + gyd * gyd) + 1e-6
        rvs = cv2.GaussianBlur(np.where(av > 0, rv, 0).astype(np.float32), (0, 0), 2.0) / \
            np.maximum(cv2.GaussianBlur((av > 0).astype(np.float32), (0, 0), 2.0), 1e-3)
        rs = np.maximum(rvs[iy, ix], 1.0)
        dm = Dm[iy, ix]
        qq = np.clip(1 - dm / rs, 0, 1)                    # 0 axis .. 1 edge
        fac = -(gxd[iy, ix] * Lx + gyd[iy, ix] * Ly) / gnd[iy, ix]
        # painted cylinder: two flat value planes (lit / shadow) with a soft terminator, not a smooth gradient
        lit = np.clip(fac * np.sqrt(qq) * 1.4, -1, 1)
        plane = _ss(-0.25, 0.15, lit)                      # 0 shadow plane .. 1 lit plane
        big = _ss(2.0, 5.0, rs)
        u_ = np.sign(fac) * qq
        l_ = lv[iy, ix]
        # dark horizontal lenticel bands (across the limb), dashed
        per = max(3.0, 0.065 * k)
        ph = l_ / per + 0.3 * T2.vnoise_at((u_ * 2.0).astype(np.float32)[None],
                                           (l_ / per * 0.3).astype(np.float32)[None], 3)[0]
        bw_n = T2.vnoise_at((u_ * 1.5).astype(np.float32)[None], (l_ / per * 0.7).astype(np.float32)[None], 9)[0]
        band = np.clip(1 - np.abs(ph - np.round(ph)) / (0.05 + 0.1 * bw_n), 0, 1)
        brk = T2.vnoise_at((u_ * 3.0).astype(np.float32)[None], (l_ / per).astype(np.float32)[None], 5)[0]
        lent = band * _ss(0.5, 0.62, brk) * big
        # vertical fissures along the axis
        fu = u_ * np.clip(rs / 3.0, 1.5, 8.0)
        fv = l_ / np.maximum(rs * 2.5, 4.0)
        nf = T2.vnoise_at((fu * 1.4).astype(np.float32)[None], (fv * 0.4).astype(np.float32)[None], 7)[0]
        fis = _ss(0.66, 0.8, nf) * big
        nl = T2.vnoise_at((ix / max(5.0, 0.07 * k)).astype(np.float32)[None],
                          (iy / max(5.0, 0.07 * k)).astype(np.float32)[None], 11)[0]
        lich = _ss(0.66, 0.74, nl - 0.12 * lit) * big * 0.4
        # colour: warm red-brown lit plane, cool blue-grey shadow plane
        c_sh = np.array([0.16, 0.13, 0.17], np.float32)
        c_lt = np.array([0.46, 0.3, 0.23], np.float32)
        c = c_sh + (c_lt - c_sh) * plane[:, None]
        # plate mottling (painted, big shapes)
        nm = T2.vnoise_at((ix / max(3.0, 0.03 * k)).astype(np.float32)[None],
                          (iy / max(3.0, 0.08 * k)).astype(np.float32)[None], 17)[0]
        c = c * (1 + 0.22 * (nm - 0.5) * big)[:, None]
        c = c * (1 - 0.5 * lich[:, None]) + np.array([0.42, 0.44, 0.4], np.float32) * (0.5 * lich)[:, None] * \
            (0.6 + 0.5 * plane[:, None])
        c = c * (1 - 0.55 * lent[:, None])
        c = c * (1 - 0.5 * fis[:, None])
        # cool sky bounce on the shadow side (blue-grey), warm bounce low on the sun side
        bnc = (1 - plane) * _ss(0.45, 0.95, qq)
        c = c * (1 - 0.45 * bnc[:, None]) + np.array([0.3, 0.33, 0.48], np.float32) * (0.45 * bnc)[:, None]
        c = c + np.array([0.12, 0.07, 0.03], np.float32) * plane[:, None] * 0.6
        # crisp warm rim: a fixed-width band (rim_px card px) along the sun-facing silhouette
        rw = max(2.0, 0.9 * self.rim_px)
        rim = _ss(rw + 0.8, rw - 0.6, dm) * _ss(0.15, 0.55, fac) * _ss(1.5, 3.0, rs)
        c = c * (1 - 0.85 * rim[:, None]) + np.array([1.05, 0.78, 0.55], np.float32) * (0.85 * rim)[:, None]
        n1 = rng.random(len(iy)).astype(np.float32)
        c = c * (0.97 + 0.06 * n1)[:, None]
        if shade is not None:
            st = shade[iy, ix][:, None]
            c = c * (1 - st) + np.array([0.2, 0.15, 0.24], np.float32) * st
        col[iy, ix] = c
        return col, av

    # ------------------------------------------------------------------ edge breakup
    def _edge_breakup(self, rgb, a, rng):
        """florets poking out past the blossom silhouette + small bites cut into it (petal-scale edge)"""
        fr = self.fr
        if fr < 2.5:
            return rgb, a
        h, w = a.shape
        m = (a > 0.5).astype(np.uint8)
        er = cv2.erode(m, np.ones((3, 3), np.uint8))
        edge = (m > 0) & (er == 0)
        ey, ex = np.nonzero(edge)
        if len(ey) == 0:
            return rgb, a
        ab = cv2.GaussianBlur(a, (0, 0), max(1.5, 0.8 * fr))
        gx = cv2.Sobel(ab, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ab, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        nxo, nyo = -gx / gn, -gy / gn                   # outward normal
        # thin the edge pixels to ~1 per cell
        cell = 1.5 * fr
        key = (ey // cell).astype(np.int64) * 100000 + (ex // cell).astype(np.int64)
        perm = rng.permutation(len(ey))
        _, first = np.unique(key[perm], return_index=True)
        sel = perm[first]
        ey, ex = ey[sel], ex[sel]
        r = rng.random(len(ey))
        Lx, Ly = float(self.L[0]), float(self.L[1])
        # ---- bites (carve small round notches, reveals what is behind)
        nb = r < 0.1 * self.notch
        if nb.any():
            yy, xx = ey[nb], ex[nb]
            rad = fr * rng.uniform(0.4, 0.75, len(yy))
            cxs = xx + 0.5 + nxo[yy, xx] * rad * 0.35
            cys = yy + 0.5 + nyo[yy, xx] * rad * 0.35
            mask = np.ones((h, w), np.float32)
            ssf = 4
            big = np.zeros((h * 1, w * 1), np.uint8)
            for x_, y_, r_ in zip(cxs, cys, rad):
                cv2.circle(big, (int(x_ * ssf), int(y_ * ssf)), int(r_ * ssf), 255, -1, cv2.LINE_AA, shift=2)
            mask = 1 - big.astype(np.float32) / 255.0
            a = a * mask
        # ---- florets beyond the edge (5-petal, AA), coloured from the paint just inside
        nf = (r >= 0.1 * self.notch) & (r < 0.1 * self.notch + 0.55 * self.edge_florets)
        if nf.any():
            yy, xx = ey[nf], ex[nf]
            R = fr * rng.uniform(0.75, 1.25, len(yy))
            off = rng.uniform(0.15, 0.6, len(yy))
            cx = xx + 0.5 + nxo[yy, xx] * R * off
            cy = yy + 0.5 + nyo[yy, xx] * R * off
            ix = np.clip((xx - nxo[yy, xx] * 1.5 * fr).astype(int), 0, w - 1)
            iy = np.clip((yy - nyo[yy, xx] * 1.5 * fr).astype(int), 0, h - 1)
            inside = a[iy, ix] > 0.9
            ix = np.where(inside, ix, xx)
            iy = np.where(inside, iy, yy)
            cc = rgb[iy, ix].astype(np.float64)
            okc = (a[iy, ix] > 0.5) & (cc.sum(1) > 0.6)
            yy, xx, R, off, cx, cy, cc = yy[okc], xx[okc], R[okc], off[okc], cx[okc], cy[okc], cc[okc]
            facing = nxo[yy, xx] * Lx + nyo[yy, xx] * Ly
            cc = cc * (1.0 + 0.05 * np.clip(facing, 0, 1))[:, None]
            n_ = len(yy)
            Fm = np.column_stack([cx, cy, R, rng.uniform(0, 6.283, n_), np.clip(cc, 0, 1.12),
                                  np.full(n_, 0.18), np.full(n_, 0.0), np.zeros(n_)])
            rgb = np.ascontiguousarray(rgb.astype(np.float32))
            a = np.ascontiguousarray(a.astype(np.float32))
            T2.stamp_florets(rgb, a, np.ascontiguousarray(Fm[np.argsort(Fm[:, 1])]))
        return rgb, a

    # ------------------------------------------------------------------ whole tree
    def paint(self, tree, rng, margin=6):
        k = self.k
        margin = int(margin + 0.35 * k)
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
        c0, c1 = self.clump_m
        rmin, rmax = max(2.2, c0 * k), max(3.0, c1 * k)
        # ---- blossom clusters at the twig tips that end in open air (no bare stick ends)
        rows_, heads = [], []
        for P, r in tree.limbs[1:]:
            pp = to_px(P[-2:])
            d_ = pp[1] - pp[0]
            d_ = d_ / (np.hypot(*d_) + 1e-6)
            e = pp[1] + d_ * 0.02 * k
            ex_, ey_ = int(np.clip(e[0], 0, W - 1)), int(np.clip(e[1], 0, H - 1))
            if aC[ey_, ex_] > 0.9:
                continue
            Rc = max(1.4 * rmax, rng.uniform(0.18, 0.3) * k)
            pts = []
            for _ in range(int(np.clip(3.0 * (Rc / (0.5 * (rmin + rmax))) ** 2, 5, 50))):
                rr_ = Rc * rng.random() * 0.85
                an = rng.uniform(0, 2 * math.pi)
                qx, qy = e[0] + math.cos(an) * rr_ * 1.15, e[1] + math.sin(an) * rr_ * 0.8
                loc = 0.6 * ((qx - e[0]) * Lx + (qy - e[1]) * Ly) / Rc - 0.5 * (qy - e[1]) / Rc
                pts.append((loc + rng.normal(0, 0.12), qx, qy, rmin + (rmax - rmin) * rng.random() ** 1.4,
                            0.45 + 0.35 * loc + rng.normal(0, 0.04)))
            pts.sort(key=lambda p_: p_[0])
            for (_, qx, qy, rc_, Vc) in pts:
                h0 = len(heads)
                for j in range(int(rng.integers(3, 7))):
                    an = rng.uniform(-math.pi * 1.15, math.pi * 0.15)
                    hr = rc_ * rng.uniform(0.3, 0.55)
                    dd = (rc_ - hr) * rng.uniform(0.75, 1.15)
                    heads.append((qx + math.cos(an) * dd * 1.1, qy + math.sin(an) * dd * 0.9, hr,
                                  float(rng.integers(5, 8)), rng.uniform(0, 6.283)))
                rows_.append((qx, qy, rc_, Vc, 0.25, -9.0, h0, len(heads)))
        if rows_:
            paint_clusters9(rgbC, aC, vC, np.array(rows_, np.float64), np.array(heads, np.float64), Lx, Ly,
                            P5._pal_arr(pal), self.cap_amt, 0.0, 0.4,
                            (pal['lav'] * 0.6 + pal['deep'] * 0.4).astype(np.float64))
        # ---- petal-scale silhouette
        rgbC, aC = self._edge_breakup(rgbC, aC, rng)
        self._tree_s = float(getattr(tree, 's', 1.0))
        win = np.zeros((H, W), np.float32)
        # ---- shadowed back layer of the lower crown (behind the wood)
        ab8 = (aC > 0.5).astype(np.uint8)
        closed = P5._enclosed(ab8, max(4, int(0.6 * k))).astype(np.float32)
        rows = np.nonzero(ab8.max(1))[0]
        rgbF = np.zeros((H, W, 3), np.float32)
        aF = np.zeros((H, W), np.float32)
        hfy = np.zeros((H, 1), np.float32)
        if len(rows):
            ct, cb = rows.min(), rows.max()
            hfy = ((np.arange(H, dtype=np.float32) - ct) / max(cb - ct, 1))[:, None]
            nzf = _noise(H, W, max(4.0, 0.3 * k), rng)
            lowz = _ss(0.3, 0.5, hfy + 0.25 * (nzf - 0.5))
            fillm = np.maximum(closed * lowz, win) * (1 - aC)
            fillm = cv2.GaussianBlur(fillm.astype(np.float32), (0, 0), 0.7)
            fc = pal['lav'] * 0.55 + pal['deep'] * 0.45
            rgbF[:] = fc
            aF = np.ascontiguousarray(fillm.astype(np.float32))
            vF = np.full((H, W), 0.2, np.float32)
            cell = 0.95 * rmin
            gy_, gx_ = np.mgrid[0:H:cell, 0:W:cell]
            jx = (gx_ + rng.uniform(0, cell, gx_.shape)).ravel()
            jy = (gy_ + rng.uniform(0, cell, gy_.shape)).ravel()
            ix = np.clip(jx.astype(int), 0, W - 1)
            iy = np.clip(jy.astype(int), 0, H - 1)
            sel = np.nonzero((fillm[iy, ix] > 0.3) & (aC[iy, ix] < 0.9) & (rng.random(len(ix)) < 0.95))[0]
            if len(sel):
                sel = sel[np.argsort(jy[sel])]
                rr2, hh2 = [], []
                for q_ in sel:
                    rc_ = rmin + (rmax - rmin) * 0.6 * rng.random()
                    h0 = len(hh2)
                    for j in range(int(rng.integers(3, 6))):
                        an = rng.uniform(0, 2 * math.pi)
                        hr = rc_ * rng.uniform(0.3, 0.5)
                        dd = (rc_ - hr) * rng.uniform(0.75, 1.1)
                        hh2.append((jx[q_] + math.cos(an) * dd, jy[q_] + math.sin(an) * dd * 0.85, hr,
                                    float(rng.integers(5, 8)), rng.uniform(0, 6.283)))
                    rr2.append((jx[q_], jy[q_], rc_, rng.choice([0.08, 0.2, 0.36, 0.5], p=[0.25, 0.35, 0.28, 0.12]) + rng.normal(0, 0.03), 0.3, -9.0, h0, len(hh2)))
                paint_clusters9(rgbF, aF, vF, np.array(rr2, np.float64), np.array(hh2, np.float64), Lx, Ly,
                                P5._pal_arr(pal), 0.14, 0.0, 0.45,
                                (pal['deep'] * 0.8 + pal['lav'] * 0.2).astype(np.float64))
            aF = aF * (1 - np.clip(aC, 0, 1)) * np.maximum(closed, win)
        # ---- bark (shaded by the crown above it)
        crown = cv2.GaussianBlur(np.maximum(aC, aF), (0, 0), max(2.0, 0.06 * k))
        shade = (0.45 * np.clip(crown * 1.3, 0, 1)).astype(np.float32)
        self._tree_s = float(getattr(tree, 's', 1.0))
        col_b, av = self._bark(H, W, tree.limbs, to_px, rng, shade)
        # limbs run on INTO the shadowed underside of the crown for a short way and are lost behind flower
        # clusters (cluster-sized noise), instead of ending at the crown's lower edge
        fw = np.zeros((H, W), np.float32)
        if len(rows):
            # thick limbs stay visible in front of the lower crown (they carry it), broken by flower clusters
            # (cluster-sized noise) and lost behind blossom where they taper toward the twigs
            nzw = _noise(H, W, max(3.0, 2.2 * rmax), rng)
            nzw2 = _noise(H, W, max(2.0, 0.45 * rmax), rng)
            n_ = 0.7 * nzw + 0.3 * nzw2
            thick = np.clip((self._rv / k - 0.024) / 0.022, 0, 1)
            low = _ss(0.3, 0.6, hfy)
            score = 0.75 * n_ + 0.42 * thick + 0.2 * low
            fw = np.clip((score - 0.8) / 0.015, 0, 1) * (aC > 0.02)
            kc_ = max(3, int(0.05 * k) | 1)
            fw = cv2.morphologyEx(fw.astype(np.float32), cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc_, kc_)))
            fw = (fw * av).astype(np.float32)
            # no ragged half-width slivers: a visible stretch keeps the full limb width or goes
            ko_ = max(3, int(0.05 * k) | 1)
            fwo = cv2.morphologyEx((fw > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko_, ko_)))
            fwo = cv2.dilate(fwo, np.ones((3, 3), np.uint8)).astype(np.float32)
            fw = (fwo * av).astype(np.float32)
            inside = (aC > 0.5).astype(np.uint8)
            dep = cv2.distanceTransform(inside, cv2.DIST_L2, 5).astype(np.float32)
            Ld = 0.4 * k
        av_full = av.copy()
        # only wood connected to the trunk shows (no floating fragments / stray sticks)
        vis = np.maximum(av * (aC < 0.15), fw)
        lab_n, lab, stats, _ = cv2.connectedComponentsWithStats((vis > 0.5).astype(np.uint8), connectivity=4)
        if lab_n > 1:
            foot = to_px(np.array([[0.0, 0.3]]))[0]
            fx_, fy_ = int(np.clip(foot[0], 0, W - 1)), int(np.clip(foot[1], 0, H - 1))
            keep = np.zeros(lab_n, bool)
            tl = lab[max(0, fy_ - 3):fy_ + 4, max(0, fx_ - 3):fx_ + 4]
            for v_ in np.unique(tl):
                if v_ > 0:
                    keep[v_] = True
            # long THIN twig runs crossing a sky gap are kept (they read as twigs, not stubs)
            rvl = np.where(lab > 0, self._rv, 0)
            for j in range(1, lab_n):
                bw_, bh_ = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
                if max(bw_, bh_) > 0.5 * k and stats[j, cv2.CC_STAT_AREA] < 0.012 * k * max(bw_, bh_):
                    keep[j] = True
            drop = (~keep[lab]) & (lab > 0)
            dm = cv2.dilate(drop.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
            av = av * (1 - dm * (1 - aC))
            fw = fw * (1 - dm)
        if os.environ.get('R9_DUMP'):
            np.savez_compressed(os.environ['R9_DUMP'], av=av, aC=aC, aF=aF, col_b=col_b, fw=fw, rv=self._rv)
        # ---- composite: back layer -> wood -> blossom clusters
        rgb = rgbF * aF[..., None]
        acc = aF.copy()
        rgb = rgb * (1 - av[..., None]) + col_b * av[..., None]
        acc = acc * (1 - av) + av
        rgb = rgb * (1 - aC[..., None]) + rgbC * aC[..., None]
        out_a = acc * (1 - aC) + aC
        if fw.any():
            dk = (0.35 + 0.4 * np.clip(dep / Ld, 0, 1))[..., None]
            wc = col_b * (1 - 0.5 * dk) + np.array([0.2, 0.15, 0.28], np.float32) * (0.5 * dk)
            rgb = rgb * (1 - fw[..., None]) + wc * (fw * out_a)[..., None]
        # ---- a flower clump over every place where visible wood goes behind the blossom (the limb is lost
        #      behind a bunch of flowers, never cut by a clean line)
        vw = ((av * (1 - aC) + fw) > 0.5).astype(np.uint8)
        kd = max(3, int(0.012 * k) | 1)
        hidw = ((vw == 0) & (av_full > 0.5)).astype(np.uint8)
        hidw = cv2.morphologyEx(hidw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        endm = (cv2.dilate(vw, np.ones((kd, kd), np.uint8)) > 0) & (hidw > 0) & (self._rv > 0.012 * k)
        ends = []
        for pp_, rr_ in self._axes:
            n_ = len(pp_)
            ix_ = np.clip(pp_[:, 0].astype(int), 0, W - 1)
            iy_ = np.clip(pp_[:, 1].astype(int), 0, H - 1)
            on = vw[iy_, ix_] > 0
            thick_ = rr_ > 0.014 * k
            for q_ in range(1, n_):
                if on[q_ - 1] != on[q_] and thick_[q_]:
                    ends.append(pp_[q_] if on[q_ - 1] else pp_[q_ - 1])
            if on[-1] and rr_[-1] > 0.02 * k:
                ends.append(pp_[-1])
        # merge ends closer than a clump
        merged = []
        for e_ in ends:
            if all(np.hypot(*(e_ - m_)) > 1.2 * rmax for m_ in merged):
                merged.append(e_)
        rows_, heads = [], []
        for e_ in merged:
            ex_, ey_ = float(e_[0]), float(e_[1])
            iy_, ix_ = int(np.clip(ey_, 0, H - 1)), int(np.clip(ex_, 0, W - 1))
            lr_ = float(self._rv[iy_, ix_])
            Rc = max(1.5 * rmax, 2.6 * lr_)
            for _ in range(int(np.clip(3.0 * (Rc / (0.5 * (rmin + rmax))) ** 2, 6, 45))):
                rr_ = Rc * math.sqrt(rng.random()) * 0.8
                an = rng.uniform(0, 2 * math.pi)
                qx = ex_ + math.cos(an) * rr_ * 1.15
                qy = ey_ + math.sin(an) * rr_ * 0.85
                rc_ = rmin + (rmax - rmin) * rng.random() ** 1.3
                loc = 0.6 * ((qx - ex_) * Lx + (qy - ey_) * Ly) / Rc - 0.5 * (qy - ey_) / Rc
                v0 = float(vC[int(np.clip(qy, 0, H - 1)), int(np.clip(qx, 0, W - 1))])
                h0 = len(heads)
                for jj in range(int(rng.integers(3, 7))):
                    a2 = rng.uniform(-math.pi * 1.15, math.pi * 0.15)
                    hr = rc_ * rng.uniform(0.3, 0.55)
                    dd = (rc_ - hr) * rng.uniform(0.75, 1.15)
                    heads.append((qx + math.cos(a2) * dd * 1.1, qy + math.sin(a2) * dd * 0.9, hr,
                                  float(rng.integers(5, 8)), rng.uniform(0, 6.283)))
                rows_.append((qx, qy, rc_, float(np.clip(v0, 0.2, 0.8)) + 0.2 * loc + rng.normal(0, 0.06), 0.25,
                              -9.0, h0, len(heads)))
        if rows_:
            order = np.argsort([r_[1] for r_ in rows_])
            rows_ = [rows_[i] for i in order]
            rgbE = np.zeros((H, W, 3), np.float32)
            aE = np.zeros((H, W), np.float32)
            vE = np.zeros((H, W), np.float32)
            paint_clusters9(rgbE, aE, vE, np.array(rows_, np.float64), np.array(heads, np.float64), Lx, Ly,
                            P5._pal_arr(pal), self.cap_amt, 0.0, 0.0,
                            (pal['lav'] * 0.6 + pal['deep'] * 0.4).astype(np.float64))
            rgb = rgb * (1 - aE[..., None]) + rgbE * aE[..., None]
            out_a = out_a * (1 - aE) + aE
        out_rgb = rgb                                   # premultiplied
        # ---- sky gaps: warm pink-gold backlit edge on the clusters around them
        rw = self.rim_px
        ab = (aC > 0.5).astype(np.uint8)
        if self.gap_rim > 0:
            kc = max(3, int(0.45 * k) | 1)
            closed2 = cv2.morphologyEx(ab, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
            inner = cv2.erode(closed2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
            gap = ((inner > 0) & (ab == 0) & (out_a < 0.5)).astype(np.uint8)
            ng, lg, stg, _ = cv2.connectedComponentsWithStats(gap, connectivity=8)
            big_ = stg[:, cv2.CC_STAT_AREA] > (0.08 * k) ** 2
            big_[0] = False
            gap = big_[lg].astype(np.uint8)
            if len(rows):
                gap = gap * (hfy < 0.7).astype(np.uint8)
            if gap.any():
                r_ = max(1, int(round(0.8 * rw)))
                gd = cv2.dilate(gap, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1, 2 * r_ + 1)))
                ring = (gd > 0).astype(np.float32) * aC
                ring = cv2.GaussianBlur(ring, (0, 0), 0.6) * aC * (vC < 0.75)
                ring = np.clip(ring, 0, 1) * self.gap_rim
                gold = np.array([1.12, 0.88, 0.76], np.float32)
                out_rgb = out_rgb * (1 - 0.55 * ring[..., None]) + gold * (0.55 * ring * out_a)[..., None]
        # ---- sun-side rim on the crown flank facing the sun
        abf = aC
        sh = _shift(abf, -Lx * rw, -Ly * rw)
        rim = np.clip(abf - sh, 0, 1)
        sgc = max(2.0, 0.35 * k)
        bigb = T2._blur(abf, sgc)
        gxa = cv2.Sobel(bigb, cv2.CV_32F, 1, 0, ksize=3)
        gya = cv2.Sobel(bigb, cv2.CV_32F, 0, 1, ksize=3)
        gna = np.sqrt(gxa * gxa + gya * gya) + 1e-6
        facing = -(gxa * Lx + gya * Ly) / gna
        rim = rim * _ss(0.1, 0.6, facing) * _ss(0.02, 0.1, gna * sgc)
        rim = cv2.GaussianBlur(rim, (0, 0), 0.5) * abf
        rim = np.clip(rim * 1.3, 0, 1)
        out_rgb = out_rgb * (1 - 0.8 * rim[..., None]) + pal['rim'] * (0.8 * rim * out_a)[..., None]
        cutg = _ss(oy + 0.03 * k + 1.0, oy + 0.03 * k - 1.0, np.arange(H, dtype=np.float32))[:, None]
        card = np.zeros((H, W, 4), np.float32)
        card[..., :3] = out_rgb * cutg[..., None]
        card[..., 3] = out_a * cutg
        return card, ox, oy
