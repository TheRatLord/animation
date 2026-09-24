"""s05_sakura round 8: cherry crowns painted as MANY SMALL BLOSSOM CLUSTERS (Painter5).

Keeps the Tree3 geometry (trunk, kinked limbs, 3-5 blossom masses per crown).  Every mass is re-painted the way a
background artist builds a sakura crown:
  * the mass is a pile of 40-400 small flower clusters (0.09-0.22 m), each an anti-aliased union of a flat-
    bottomed core and 5-9 round floret heads (a lacy petal-cluster outline, no noise-stepped edges);
  * value comes from the CROWN (height in the crown + facing of the blurred mass toward the sun) sampled at the
    cluster centre -> the big light pattern (warm-white lit top, clean mid pink body, cool lavender underside)
    is carried by whole clusters, so value edges follow cluster outlines; each cluster adds a crisp lit cap on
    its own sun side (per head -> scalloped cap edge) with per-cluster jittered thresholds (no stamped shape);
  * clusters painted back to front, each casting a crisp lavender contact shadow onto what is behind it;
  * lower core backed by a dim lavender under-fill; the upper crown stays open, so real sky gaps show between
    the clusters and get a warm pink-gold backlit rim;
  * bark: only coherent wood shows (visible bark fragments not connected to the trunk are removed -> no floating
    stubs / random sticks); cherry bark with horizontal lenticel bands, dark fissures, grey-green lichen
    patches, warm rim on the sun side and cool shadow on the far side.
"""
import math
import numpy as np
import cv2
from numba import njit

import s05_sakura_canopy3 as C3
import s05_sakura_tree2d as T2
from s05_sakura_tree2d import _noise, _ss, _shift, limb_raster


PAL5 = dict(hot=np.array([1.03, 0.94, 0.93], np.float32), lit=np.array([1.0, 0.84, 0.88], np.float32),
            mid=np.array([0.96, 0.66, 0.8], np.float32), lav=np.array([0.76, 0.6, 0.83], np.float32),
            deep=np.array([0.52, 0.42, 0.7], np.float32), trans=np.array([1.04, 0.8, 0.74], np.float32),
            glow=np.array([1.1, 0.86, 0.66], np.float32), rim=np.array([1.16, 1.02, 0.86], np.float32),
            bounce=np.array([0.92, 0.66, 0.68], np.float32))


@njit(cache=True, inline='always')
def _band(v, P):
    """cel bands (AA): deep -> lavender -> mid pink -> lit pink-white -> hot. P rows: deep, lav, mid, lit, hot"""
    e = 0.018
    w1 = min(max((v - (0.3 - e)) / (2 * e), 0.0), 1.0)
    w2 = min(max((v - (0.58 - e)) / (2 * e), 0.0), 1.0)
    w3 = min(max((v - (0.88 - e)) / (2 * e), 0.0), 1.0)
    wd = min(max((0.14 - v) / 0.14, 0.0), 1.0) * 0.55
    r = np.empty(3)
    for c in range(3):
        x = P[1, c] * (1 - w1) + P[2, c] * w1
        x = x * (1 - w2) + P[3, c] * w2
        x = x * (1 - w3) + P[4, c] * w3
        x = x * (1 - wd) + P[0, c] * wd
        r[c] = x
    return r


@njit(cache=True, inline='always')
def _sdf(px, py, cx, cy, rc, Hd, h0, h1):
    dx = px - cx
    dy = py - cy
    sy = 0.82 if dy > 0 else 1.0
    d = math.sqrt((dx / 1.08) ** 2 + (dy / sy) ** 2) - 0.64 * rc
    best = -1
    for j in range(h0, h1):
        ex = px - Hd[j, 0]
        ey = py - Hd[j, 1]
        r = math.sqrt(ex * ex + ey * ey)
        hr = Hd[j, 2]
        e = r - hr
        if e < d + 3.0 and hr > 4.0:
            # petal-scale scallops on each floret head (5-7 soft notches, <= 2.5 px deep)
            th = math.atan2(ey, ex)
            amp = min(0.12 * hr, 2.5)
            e += amp * (0.5 - 0.5 * math.cos(Hd[j, 3] * th + Hd[j, 4]))
        if e < d:
            d = e
            best = j
    return d, best


@njit(cache=True)
def paint_clusters(rgb, a, vl, C, Hd, Lx, Ly, P, cap_amt, und_amt, sh_amt, shc):
    """C rows: cx, cy, rc, Vc, thr_cap, thr_und, h0, h1 (local px).  Straight rgb + coverage a, in place."""
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
                d, hb = _sdf(px, py, cx, cy, rc, Hd, h0, h1)
                cov = min(max(0.5 - d, 0.0), 1.0)
                # contact shadow on what is already painted behind (outside this cluster)
                if cov < 1.0 and a[y, x] > 0.0 and sh_amt > 0:
                    ds, _ = _sdf(px - ox, py - oy, cx, cy, rc, Hd, h0, h1)
                    s = min(max(0.5 - ds / 1.5, 0.0), 1.0) * sh_amt * (1 - cov)
                    if s > 0:
                        for c in range(3):
                            rgb[y, x, c] = rgb[y, x, c] * (1 - s) + shc[c] * s
                        vl[y, x] -= 0.2 * s
                if cov <= 0.0:
                    continue
                # value: crown light at the cluster centre + a crisp lit cap on the sun side of each head
                lnc = ((px - cx) * Lx + (py - cy) * Ly) / rc
                upc = -(py - cy) / rc
                loc = 0.55 * lnc + 0.45 * upc
                if hb >= 0:
                    hr = Hd[hb, 2]
                    lnh = ((px - Hd[hb, 0]) * Lx + (py - Hd[hb, 1]) * Ly) / hr
                    uph = -(py - Hd[hb, 1]) / hr
                    loc = 0.55 * loc + 0.45 * (0.55 * lnh + 0.45 * uph)
                cap = min(max((loc - tc) / 0.08, 0.0), 1.0)
                und = min(max((tu - loc) / 0.08, 0.0), 1.0)
                v = Vc + cap_amt * cap - und_amt * und
                col = _band(v, P)
                old = a[y, x]
                na = cov + old * (1 - cov)
                for c in range(3):
                    rgb[y, x, c] = (col[c] * cov + rgb[y, x, c] * old * (1 - cov)) / na
                a[y, x] = na
                if cov > 0.5:
                    vl[y, x] = v


def _enclosed(m, L, need=7):
    """pixels of the empty part of mask m (uint8) that see mask within L px in >= need of 8 directions"""
    q = max(1, int(L / 40))
    h, w = m.shape
    ms = cv2.resize(m.astype(np.float32), (max(1, w // q), max(1, h // q)), interpolation=cv2.INTER_AREA)
    ms = (ms > 0.5).astype(np.uint8)
    Lq = max(2, int(L / q))
    cnt = np.zeros(ms.shape, np.int32)
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        ker = np.zeros((2 * Lq + 1, 2 * Lq + 1), np.uint8)
        for t in range(1, Lq + 1):
            ker[Lq + dy * t if abs(dx) + abs(dy) == 1 else Lq + dy * int(t / 1.41),
                Lq + dx * t if abs(dx) + abs(dy) == 1 else Lq + dx * int(t / 1.41)] = 1
        # dilate looks at src(x + k - anchor): a hit in direction (dx, dy)
        hit = cv2.dilate(ms, ker, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        cnt += hit > 0
    enc = ((cnt >= need) | (ms > 0)).astype(np.float32)
    enc = cv2.resize(enc, (w, h), interpolation=cv2.INTER_LINEAR)
    if q > 1:
        enc = cv2.GaussianBlur(enc, (0, 0), 0.6 * q)
    return ((enc > 0.5) | (m > 0)).astype(np.uint8)


def _pal_arr(pal):
    return np.stack([pal['deep'], pal['lav'], pal['mid'], pal['lit'], pal['hot']]).astype(np.float64)


class Painter5(C3.Painter3):
    def __init__(self, *a, clump_m=(0.07, 0.14), clump_r=(0.28, 0.55), under_fill=0.6, cap_amt=0.12,
                 und_amt=0.0, clump_amt=0.4, open_top=1.0, gap_rim=1.0, bark_keep=0.05, florets=False, **kw):
        kw.setdefault('pal', PAL5)
        super().__init__(*a, **kw)
        self.clump_m = clump_m
        self.under_fill = under_fill
        self.cap_amt = cap_amt
        self.clump_r = clump_r
        self.clump_amt = clump_amt
        self.open_top = open_top
        self.und_amt = und_amt
        self.gap_rim = gap_rim
        self.bark_keep = bark_keep
        self.florets = florets

    # ------------------------------------------------------------------ one mass
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
        grid = C3._Grid(max(4.0, 2.2 * Rmax))
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
        Pp = _pal_arr(pal)
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
                Vc = float(vm[ic]) + self.clump_amt * loc + mr.normal(0, 0.035)
                pts.append((loc, qx, qy, rc_, Vc))
            pts.sort(key=lambda p_: p_[0] + mr.normal(0, 0.12))
            for (loc, qx, qy, rc_, Vc) in pts:
                add_cluster(qx, qy, rc_, Vc, 0.25 + mr.normal(0, 0.08))
        C = np.array(rows, np.float64)
        Hd = np.array(heads, np.float64)
        shc = (pal['lav'] * 0.6 + pal['deep'] * 0.4).astype(np.float64)
        paint_clusters(rgb, a, vloc, C, Hd, Lx, Ly, Pp, self.cap_amt, self.und_amt, 0.4, shc)

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
        edge = np.clip(1 - dist / (2.0 * bw), 0, 1) ** 1.2
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
        shd = cv2.GaussianBlur(shd, (0, 0), max(1.0, 0.012 * R)) * (1 - a) * sub_a
        sub_c[:] = sub_c * (1 - 0.35 * shd[..., None]) + pal['deep'] * (0.35 * shd)[..., None]
        na = a + sub_a * (1 - a)
        sub_c[:] = (rgb * a[..., None] + sub_c * (sub_a * (1 - a))[..., None]) / np.maximum(na, 1e-4)[..., None]
        sub_a[:] = na
        take = a > 0.5
        vC[Y0:Y1, X0:X1][take] = vloc[take]
        sdC[Y0:Y1, X0:X1][take] = -(dist / max(bw, 1.0))[take]

    # ------------------------------------------------------------------ bark
    def _bark(self, h, w, limbs, to_px, rng, shade):
        """cherry bark: dark red-brown, horizontal lenticel bands, fissures, lichen, warm sun rim, cool shade"""
        k = self.k
        segs = []
        for P, r in limbs:
            pp = to_px(P)
            rr = r * k
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
        # smooth cylinder shading from the distance field of the whole wood silhouette (no facets at kinks)
        Dm = cv2.distanceTransform((av > 0.5).astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)
        Ds = cv2.GaussianBlur(Dm, (0, 0), 1.5)
        gxd = cv2.Sobel(Ds, cv2.CV_32F, 1, 0, ksize=3)
        gyd = cv2.Sobel(Ds, cv2.CV_32F, 0, 1, ksize=3)
        gnd = np.sqrt(gxd * gxd + gyd * gyd) + 1e-6
        rvs = cv2.GaussianBlur(np.where(av > 0, rv, 0).astype(np.float32), (0, 0), 2.0) /             np.maximum(cv2.GaussianBlur((av > 0).astype(np.float32), (0, 0), 2.0), 1e-3)
        rs = np.maximum(rvs[iy, ix], 1.0)
        qq = np.clip(1 - Dm[iy, ix] / rs, 0, 1)                  # 0 axis .. 1 edge
        fac = -(gxd[iy, ix] * Lx + gyd[iy, ix] * Ly) / gnd[iy, ix]
        lit = np.clip(fac * np.sqrt(qq) * 1.2, -1, 1)              # -1 shadow side .. +1 sun side
        big = _ss(2.0, 5.0, rs)
        u_ = np.sign(fac) * qq                  # across (-1..1)
        l_ = lv[iy, ix]                     # along (px)
        # horizontal lenticel bands (across the limb), broken into dashes
        per = max(3.0, 0.05 * k)
        ph = l_ / per + 0.25 * T2.vnoise_at((u_ * 2.0).astype(np.float32)[None], (l_ / per * 0.3).astype(np.float32)[None], 3)[0]
        band = np.clip(1 - np.abs(ph - np.round(ph)) / 0.07, 0, 1)
        brk = T2.vnoise_at((u_ * 3.0).astype(np.float32)[None], (l_ / per).astype(np.float32)[None], 5)[0]
        lent = band * _ss(0.45, 0.6, brk) * big
        # fissures along the axis
        fu = u_ * np.clip(rs / 3.0, 1.5, 8.0)
        fv = l_ / np.maximum(rs * 2.5, 4.0)
        nf = T2.vnoise_at((fu * 1.4).astype(np.float32)[None], (fv * 0.4).astype(np.float32)[None], 7)[0]
        fis = _ss(0.64, 0.78, nf) * big
        # lichen patches (grey-green), mostly on the shade side
        nl = T2.vnoise_at((ix / max(5.0, 0.07 * k)).astype(np.float32)[None],
                          (iy / max(5.0, 0.07 * k)).astype(np.float32)[None], 11)[0]
        nl2 = T2.vnoise_at((ix / max(3.0, 0.035 * k)).astype(np.float32)[None],
                           (iy / max(3.0, 0.035 * k)).astype(np.float32)[None], 13)[0]
        lich = _ss(0.64, 0.72, nl + 0.35 * (nl2 - 0.5) - 0.1 * lit) * big * 0.5
        base_dk = np.array([0.13, 0.09, 0.095], np.float32)
        base_lt = np.array([0.4, 0.27, 0.22], np.float32)
        tone = np.clip(0.35 + 0.45 * lit, 0, 1)[:, None]
        c = base_dk + (base_lt - base_dk) * tone
        c = c * (1 - 0.55 * lich[:, None]) + np.array([0.4, 0.4, 0.36], np.float32) * (0.55 * lich)[:, None] * \
            (0.7 + 0.5 * tone)
        c = c * (1 - 0.5 * lent[:, None]) + np.array([0.55, 0.48, 0.44], np.float32) * (0.5 * lent)[:, None] * \
            (0.6 + 0.6 * tone)
        c = c * (1 - 0.6 * fis[:, None])
        # rough painted mottling (bark plates)
        nm = T2.vnoise_at((ix / max(3.0, 0.025 * k)).astype(np.float32)[None],
                          (iy / max(3.0, 0.06 * k)).astype(np.float32)[None], 17)[0]
        c = c * (0.85 + 0.3 * nm)[:, None] * big[:, None] + c * (1 - big[:, None])
        # cool sky bounce on the shade side, warm sun side + crisp rim
        bnc = _ss(0.1, 0.8, -lit) * _ss(0.4, 0.95, qq)
        c = c * (1 - 0.35 * bnc[:, None]) + np.array([0.26, 0.26, 0.42], np.float32) * (0.35 * bnc)[:, None]
        warm = np.array([1.0, 0.72, 0.5], np.float32)
        c = c + warm * (0.1 * _ss(0.2, 0.9, lit))[:, None]
        rim = _ss(0.74, 0.96, qq) * _ss(0.25, 0.7, lit)
        c = c * (1 - 0.8 * rim[:, None]) + np.array([1.0, 0.74, 0.52], np.float32) * (0.8 * rim)[:, None]
        n1 = rng.random(len(iy)).astype(np.float32)
        c = c * (0.96 + 0.08 * n1)[:, None]
        if shade is not None:
            st = shade[iy, ix][:, None]
            c = c * (1 - st) + np.array([0.24, 0.16, 0.22], np.float32) * st
        col[iy, ix] = c
        return col, av

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
        # ---- bark behind the blossom
        crown = cv2.GaussianBlur(aC, (0, 0), max(2.0, 0.06 * k))
        shade = (0.5 * np.clip(crown * 1.3, 0, 1)).astype(np.float32)
        col_b, av = self._bark(H, W, tree.limbs, to_px, rng, shade)
        # only coherent wood shows: drop visible bark fragments that are cut off from the trunk (no floating
        # stubs in the gaps, no stray sticks poking out of the silhouette) unless they are a long limb stretch
        # lower crown: close the gaps between masses with a shadowed back layer of small dark clusters (sky only
        # shows through the upper, thinner crown); wood inside it is hidden, thick limbs never poke up through
        # the upper gaps either
        ab8 = (aC > 0.5).astype(np.uint8)
        closed = _enclosed(ab8, max(4, int(0.6 * k))).astype(np.float32)
        rows = np.nonzero(ab8.max(1))[0]
        if len(rows):
            ct, cb = rows.min(), rows.max()
            hfy = ((np.arange(H, dtype=np.float32) - ct) / max(cb - ct, 1))[:, None]
            nzf = _noise(H, W, max(4.0, 0.3 * k), rng)
            lowz = _ss(0.3, 0.5, hfy + 0.25 * (nzf - 0.5))
            thick = _ss(0.035 * k, 0.06 * k, self._rv)
            hid = av * closed * np.maximum(lowz, thick)
            av = av - hid
            fillm = np.maximum(closed * lowz * (1 - aC), hid * lowz)
            fillm = cv2.GaussianBlur(fillm.astype(np.float32), (0, 0), 0.7)
            fc = pal['lav'] * 0.55 + pal['deep'] * 0.45
            rgbF = np.ascontiguousarray(np.broadcast_to(fc, (H, W, 3)).astype(np.float32))
            aF = np.ascontiguousarray(fillm.astype(np.float32))
            vF = np.full((H, W), 0.2, np.float32)
            c0, c1 = self.clump_m
            rmin, rmax = max(2.2, c0 * k), max(3.0, c1 * k)
            cell = 1.1 * rmin
            gy_, gx_ = np.mgrid[0:H:cell, 0:W:cell]
            jx = (gx_ + rng.uniform(0, cell, gx_.shape)).ravel()
            jy = (gy_ + rng.uniform(0, cell, gy_.shape)).ravel()
            ix = np.clip(jx.astype(int), 0, W - 1)
            iy = np.clip(jy.astype(int), 0, H - 1)
            sel = np.nonzero((fillm[iy, ix] > 0.6) & (aC[iy, ix] < 0.9) & (rng.random(len(ix)) < 0.7))[0]
            if len(sel):
                sel = sel[np.argsort(jy[sel])]
                rows_, heads = [], []
                for q in sel:
                    rc_ = rmin + (rmax - rmin) * 0.6 * rng.random()
                    h0 = len(heads)
                    for j in range(int(rng.integers(3, 6))):
                        an = rng.uniform(0, 2 * math.pi)
                        hr = rc_ * rng.uniform(0.3, 0.5)
                        dd = (rc_ - hr) * rng.uniform(0.75, 1.1)
                        heads.append((jx[q] + math.cos(an) * dd, jy[q] + math.sin(an) * dd * 0.85, hr,
                                      float(rng.integers(5, 8)), rng.uniform(0, 6.283)))
                    rows_.append((jx[q], jy[q], rc_, rng.uniform(0.08, 0.34), 0.3, -9.0, h0, len(heads)))
                paint_clusters(rgbF, aF, vF, np.array(rows_, np.float64), np.array(heads, np.float64), Lx, Ly,
                               _pal_arr(pal), 0.1, 0.0, 0.35,
                               (pal['deep'] * 0.8 + pal['lav'] * 0.2).astype(np.float64))
            aF = aF * (1 - np.clip(aC, 0, 1)) * closed
            rgbC = rgbC * aC[..., None] + rgbF * aF[..., None]
            aC = aC + aF
            rgbC = (rgbC / np.maximum(aC, 1e-4)[..., None]).astype(np.float32)
            aC = aC.astype(np.float32)
        # blossom clumps where limbs end in open air (no chopped / blunt wood ends)
        c0, c1 = self.clump_m
        rmin, rmax = max(2.2, c0 * k), max(3.0, c1 * k)
        rows_, heads = [], []
        ends = []
        for P, r in tree.limbs[1:]:
            pp = to_px(P[-2:])
            d_ = pp[1] - pp[0]
            d_ = d_ / (np.hypot(*d_) + 1e-6)
            ends.append(pp[1] + d_ * 0.03 * k)
        if len(rows):
            # places where visible wood was cut (hidden inside the crown) and the cut is not covered
            kk_ = max(3, int(0.03 * k) | 1)
            cut = (cv2.dilate(((hid > 0.3) | (aC > 0.5)).astype(np.uint8), np.ones((kk_, kk_), np.uint8)) > 0) &                 (av > 0.5) & (aC < 0.5) & (self._rv > 0.02 * k)
            nc, lc, stc, cen = cv2.connectedComponentsWithStats(cut.astype(np.uint8), connectivity=8)
            for j in range(1, nc):
                if stc[j, cv2.CC_STAT_AREA] >= 4:
                    ends.append(cen[j])
        for e in ends:
            ex_, ey_ = int(np.clip(e[0], 0, W - 1)), int(np.clip(e[1], 0, H - 1))
            rw_ = max(4, int(0.05 * k))
            win = av[max(0, ey_ - rw_):ey_ + rw_ + 1, max(0, ex_ - rw_):ex_ + rw_ + 1]
            if win.size == 0 or win.max() < 0.3 or aC[ey_, ex_] > 0.95:
                continue
            lr_ = float(self._rv[ey_, ex_]) if av[ey_, ex_] > 0 else 0.05 * k
            Rc = max(1.6 * rmax, 2.2 * lr_, rng.uniform(0.28, 0.4) * k)
            pts = []
            for _ in range(int(np.clip(3.2 * (Rc / (0.5 * (rmin + rmax))) ** 2, 6, 70))):
                rr_ = Rc * rng.random() * 0.85
                an = rng.uniform(0, 2 * math.pi)
                qx, qy = e[0] + math.cos(an) * rr_ * 1.15, e[1] + math.sin(an) * rr_ * 0.8
                loc = 0.6 * ((qx - e[0]) * Lx + (qy - e[1]) * Ly) / Rc - 0.5 * (qy - e[1]) / Rc
                pts.append((loc + rng.normal(0, 0.12), qx, qy, rmin + (rmax - rmin) * rng.random() ** 1.4,
                            0.42 + 0.35 * loc + rng.normal(0, 0.04)))
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
            rgbC = np.ascontiguousarray(rgbC.astype(np.float32))
            aC = np.ascontiguousarray(aC.astype(np.float32))
            paint_clusters(rgbC, aC, vC, np.array(rows_, np.float64), np.array(heads, np.float64), Lx, Ly,
                           _pal_arr(pal), self.cap_amt, 0.0, 0.4,
                           (pal['lav'] * 0.6 + pal['deep'] * 0.4).astype(np.float64))
        vis = av * (1 - aC)
        lab_n, lab, stats, _ = cv2.connectedComponentsWithStats((vis > 0.35).astype(np.uint8), connectivity=8)
        if lab_n > 1:
            foot = to_px(np.array([[0.0, 0.3]]))[0]
            fx_, fy_ = int(np.clip(foot[0], 0, W - 1)), int(np.clip(foot[1], 0, H - 1))
            keep = np.zeros(lab_n, bool)
            tl = lab[max(0, fy_ - 3):fy_ + 4, max(0, fx_ - 3):fx_ + 4]
            for v_ in np.unique(tl):
                if v_ > 0:
                    keep[v_] = True
            amin = self.bark_keep * k * k
            for j in range(1, lab_n):
                bw_, bh_ = stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT]
                if stats[j, cv2.CC_STAT_AREA] > amin and max(bw_, bh_) > 0.45 * k and self.bark_keep > 1:
                    keep[j] = True
            drop = (~keep[lab]) & (lab > 0)
            # feather the cut a little so it never leaves hard fragments of edge pixels
            dm = cv2.dilate(drop.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
            av = av * (1 - dm * (1 - aC))
        out_rgb = rgbC * aC[..., None] + col_b * (av * (1 - aC))[..., None]
        out_a = aC + av * (1 - aC)
        # ---- sky gaps inside the crown: warm pink-gold backlit edge on the clusters around them (thin, only
        #      on the gap side, never an outline around the whole silhouette)
        rw = self.rim_px
        ab = (aC > 0.5).astype(np.uint8)
        if self.gap_rim > 0:
            kc = max(3, int(0.45 * k) | 1)
            closed = cv2.morphologyEx(ab, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
            inner = cv2.erode(closed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
            gap = ((inner > 0) & (ab == 0) & (av < 0.5)).astype(np.uint8)
            # only real sky openings (not slivers) in the upper two thirds of the crown
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
        # ---- sun-side rim: only the crown flank that faces the sun (facing from the whole-crown silhouette)
        abf = aC
        sh = _shift(abf, -Lx * rw, -Ly * rw)
        rim = np.clip(abf - sh, 0, 1)
        sgc = max(2.0, 0.35 * k)
        big = T2._blur(abf, sgc)
        gxa = cv2.Sobel(big, cv2.CV_32F, 1, 0, ksize=3)
        gya = cv2.Sobel(big, cv2.CV_32F, 0, 1, ksize=3)
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
