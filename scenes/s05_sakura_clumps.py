"""s05_sakura round 7: cherry crowns painted as CLUSTERS OF SMALL BLOSSOM CLUMPS (Painter4).

Keeps the Tree3 geometry (trunk, limbs, 3-5 masses per crown) and Painter3's bark / sky-hole / rim passes, but
replaces the per-mass soft value field with a painter's build-up of many small clumps:
  * each mass is a loose pile of 8-30 clumps (0.2-0.4 x the mass radius), placed by a greedy non-overlap pass
    inside the mass envelope; clumps near the envelope top poke out, so the crown silhouette is a crisp,
    lumpy, scalloped edge made of clump caps (never a smooth blob);
  * a clump is a flat-bottomed dome whose top is a union of 3-6 scallop heads (a floret-cluster outline) with
    a lacy petal-scale break-up of the contour, anti-aliased (card is supersampled);
  * value per clump is FLAT and cel-like: the clump inherits the crown's big light pattern (lit pink-white top
    toward the sun -> clean pink body -> cool lavender underside) and adds a crisp lit CAP on its sun-facing
    top plus a lavender-pink underside band; band edges are broken by petal-scale noise, not gradients;
  * clumps painted back to front (upper / far first): every newer clump casts a crisp lavender contact shadow
    on the clumps behind it (offset away from the sun) so the overlaps read as separate flower bunches;
  * a dim lavender under-fill sits behind the clumps in the lower core only - in the upper crown the gaps
    between clumps stay open and the sky shows through (backlit rim from Painter3's hole pass);
  * warm transmitted glow on thin under-edges and 5-petal florets strewn along the exposed silhouette.
"""
import math
import numpy as np
import cv2

import s05_sakura_canopy3 as C3
from s05_sakura_tree2d import _noise, _ss, _shift, stamp_florets


PAL4 = dict(hot=np.array([1.0, 0.84, 0.9], np.float32), lit=np.array([0.98, 0.73, 0.85], np.float32),
            mid=np.array([0.95, 0.63, 0.8], np.float32), lav=np.array([0.72, 0.58, 0.84], np.float32),
            deep=np.array([0.5, 0.41, 0.7], np.float32), trans=np.array([1.02, 0.78, 0.74], np.float32),
            glow=np.array([1.08, 0.84, 0.66], np.float32), rim=np.array([1.14, 1.04, 0.92], np.float32),
            bounce=np.array([0.9, 0.66, 0.7], np.float32))


class Painter4(C3.Painter3):
    def __init__(self, *a, clump=(0.2, 0.36), under_fill=0.62, cap=1.0, **kw):
        kw.setdefault('pal', PAL4)
        super().__init__(*a, **kw)
        self.clump = clump
        self.under_fill = under_fill
        self.cap = cap

    def _mass(self, m, to_px, k, rgbC, aC, vC, sdC, H, W):
        fr = self.fr
        mr = np.random.default_rng(m['seed'])
        Lx, Ly = float(self.L[0]), float(self.L[1])
        pal = self.pal
        bl = []
        for (bx, by, br) in m['blobs']:
            p = to_px(np.array([[bx, by]]))[0]
            bl.append((float(p[0]), float(p[1]), float(br * k)))
        R = m['R'] * k
        pad = 0.45 * R + 6 * fr
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
        # ---- mass envelope (smooth union of the lobes)
        kk = 0.08 * R
        acc = np.zeros((h, w), np.float64)
        for (bx, by, br) in bl:
            d = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2) - br
            acc += np.exp(-np.clip(d, -40 * kk, 40 * kk) / kk)
        sd = (-kk * np.log(np.maximum(acc, 1e-30))).astype(np.float32) + nz_lo * 0.16 * R
        ytop = min(b[1] - b[2] for b in bl)
        ybot = max(b[1] + b[2] for b in bl)
        hf = np.clip((yy - ytop) / max(ybot - ytop, 1.0), 0, 1)
        # dome normal of the blurred envelope (big light pattern of the crown)
        sig = max(1.0, 0.35 * R)
        fq = max(1.0, sig / 4.0)
        lw, lh = max(4, int(w / fq)), max(4, int(h / fq))
        sm = cv2.resize(_ss(0.0, -1.0, sd).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), sig / fq)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / (8 * fq)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / (8 * fq)
        ndl = cv2.resize(np.clip(-(gx * Lx + gy * Ly) * sig * 2.5, -1, 1), (w, h), interpolation=cv2.INTER_CUBIC)
        vm = (0.52 - 0.7 * (hf - 0.45) + 0.24 * ndl + 0.1 * nz_lo).astype(np.float32)

        # ---- clump placement: greedy, big first, inside the envelope; slight overlap allowed
        c0, c1 = self.clump
        rmin = max(2.6 * fr, 0.5 * c0 * R)
        cands = []
        n_try = int(np.clip(60 * (w * h) / max(R * R, 1.0), 300, 3000))
        px = mr.uniform(0, w, n_try)
        py = mr.uniform(0, h, n_try)
        rr = np.maximum(R * (c0 + (c1 - c0) * mr.random(n_try) ** 0.7), rmin)
        ix = np.clip(px.astype(int), 0, w - 1)
        iy = np.clip(py.astype(int), 0, h - 1)
        sdv = sd[iy, ix]
        ok = sdv < -0.25 * rr
        order = np.argsort(-rr[ok])
        pxo, pyo, rro = px[ok][order], py[ok][order], rr[ok][order]
        grid = C3._Grid(max(4.0, 2 * c1 * R))
        clumps = []
        for i in range(len(pxo)):
            cx, cy, rc = float(pxo[i]), float(pyo[i]), float(rro[i])
            if not grid.ok(cx, cy, rc, 0.74):
                # try a smaller clump here (fills gaps with little bunches)
                rc2 = max(rmin, 0.6 * rc)
                if rc2 >= rc or not grid.ok(cx, cy, rc2, 0.74):
                    continue
                rc = rc2
            grid.add(cx, cy, rc)
            clumps.append((cx, cy, rc))
        # edge clumps: bunches riding on the envelope outline (the crown's scalloped silhouette), top/sides only
        ey, ex = np.nonzero((sd > -0.12 * R) & (sd < 0.02 * R) & (hf < 0.8))
        if len(ey):
            pick = mr.permutation(len(ey))[:400]
            for q in pick:
                cx, cy = float(ex[q]) + 0.5, float(ey[q]) + 0.5
                rc = max(rmin, R * mr.uniform(0.6 * c0, c0 + 0.3 * (c1 - c0)))
                if grid.ok(cx, cy, rc, 0.8):
                    grid.add(cx, cy, rc)
                    clumps.append((cx, cy, rc))
        # paint order: far/upper first; lower (and slightly sunward) clumps in front
        clumps.sort(key=lambda c: c[1] + 0.35 * c[2] - 0.15 * (c[0] * Lx) + mr.uniform(-0.1, 0.1) * c[2])

        # ---- under-fill (lower core of the mass): dim lavender / mid pink, holes stay open higher up
        rgb = np.zeros((h, w, 3), np.float32)
        a = np.zeros((h, w), np.float32)
        vloc = np.full((h, w), 0.3, np.float32)
        mbar = float(np.mean([c[2] for c in clumps])) if clumps else 0.3 * R
        nzh = _noise(h, w, max(3.0, 0.9 * mbar), mr)
        fill_ok = (_ss(0.25, 0.6, hf) * self.under_fill + (1 - self.under_fill)) > nzh * 0.9 + 0.05
        fa = np.clip(-(sd + 1.0 * mbar) / 1.5, 0, 1) * fill_ok
        fa = cv2.GaussianBlur(fa.astype(np.float32), (0, 0), max(0.8, 0.12 * fr))
        vf = np.where(vm - 0.22 > 0.33, 0.45, 0.2).astype(np.float32)
        bands = self._bands(vf)
        rgb[:] = bands
        a[:] = fa
        vloc[:] = vf

        # ---- clumps
        nzf = _noise(h, w, max(4.0, 3.0 * fr), mr) - 0.5           # petal-scale break-up of edges / bands
        nzm = _noise(h, w, max(3.0, 0.35 * mbar), mr) - 0.5
        ext_all = []
        for (cx, cy, rc) in clumps:
            ext = int(1.75 * rc + 3 * fr + 3)
            xa, xb = int(max(0, cx - ext)), int(min(w, cx + ext + 1))
            ya, yb = int(max(0, cy - ext)), int(min(h, cy + ext + 1))
            if xb - xa < 2 or yb - ya < 2:
                continue
            gx_ = xx[ya:yb, xa:xb] - X0
            gy_ = yy[ya:yb, xa:xb] - Y0
            dx = gx_ - cx
            dy = gy_ - cy
            # flat-bottomed core
            sy_ = np.where(dy > 0, 0.8, 1.0)
            d = np.sqrt((dx / 1.1) ** 2 + (dy / sy_) ** 2) - 0.8 * rc
            # scallop heads on the upper rim
            nh = int(mr.integers(3, 7))
            angs = np.linspace(-math.pi * 1.1, math.pi * 0.1, nh) + mr.normal(0, 0.25, nh)
            for an in angs:
                hr = rc * mr.uniform(0.34, 0.5)
                hx = cx + math.cos(an) * (rc - hr) * 1.12
                hy = cy + math.sin(an) * (rc - hr) * 0.95
                d = np.minimum(d, np.sqrt((gx_ - hx) ** 2 + (gy_ - hy) ** 2) - hr)
            d = d + nzf[ya:yb, xa:xb] * (0.9 * fr if self.detail > 0 else 0.5 * fr)
            cov = np.clip(0.5 - d, 0, 1)
            if cov.max() <= 0:
                continue
            # value: the crown's light at the clump centre + a crisp lit cap on the sun side / top
            ic = (int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1)))
            vc = float(vm[ic])
            ln = (dx * Lx + dy * Ly) / rc                       # + toward the sun
            up = -dy / rc
            loc = 0.55 * ln + 0.45 * up
            cap = _ss(0.12, 0.2, loc + 0.22 * nzf[ya:yb, xa:xb] + 0.15 * nzm[ya:yb, xa:xb])
            und = _ss(-0.3, -0.38, loc + 0.22 * nzf[ya:yb, xa:xb])
            v = vc + 0.32 * self.cap * cap - 0.22 * und + 0.04 * mr.standard_normal()
            v = v.astype(np.float32)
            col = self._bands(v)
            # contact shadow cast by this clump on what is already painted behind it
            off = 0.22 * rc
            sa = _shift(cov, -Lx * off, -Ly * off + 0.1 * rc)
            sh = np.clip(sa - cov, 0, 1) * a[ya:yb, xa:xb]
            sub = rgb[ya:yb, xa:xb]
            sub[:] = sub * (1 - 0.5 * sh[..., None]) + (pal['lav'] * 0.7 + pal['deep'] * 0.3) * (0.5 * sh)[..., None]
            vloc[ya:yb, xa:xb] -= 0.15 * sh
            # over
            sa_ = a[ya:yb, xa:xb]
            na = cov + sa_ * (1 - cov)
            sub[:] = (col * cov[..., None] + sub * (sa_ * (1 - cov))[..., None]) / np.maximum(na, 1e-4)[..., None]
            sa_[:] = na
            vloc[ya:yb, xa:xb] = np.where(cov > 0.5, v, vloc[ya:yb, xa:xb])
            ext_all.append((cx, cy, rc))

        # ---- warm transmitted light along thin under-edges and the sun-facing side of the crown
        solid = (a > 0.5).astype(np.uint8)
        dist = cv2.distanceTransform(solid, cv2.DIST_L2, 3)
        bw = max(2.0 * fr, 0.05 * R)
        edge = np.clip(1 - dist / (2.2 * bw), 0, 1) ** 1.3
        sside = _ss(-0.1, 0.7, ndl)
        tr = edge * np.clip(0.6 * _ss(0.45, 1.0, hf) + 0.6 * sside, 0, 1) * self.glow
        tr *= (vloc < 0.62)
        rgb = rgb * (1 - 0.45 * tr[..., None]) + pal['trans'] * (0.45 * tr)[..., None]
        # warm bounce from the sunlit path into the lowest lavender
        wb = _ss(0.72, 1.0, hf) * (vloc < 0.33) * 0.25
        rgb = rgb * (1 - wb[..., None]) + pal['bounce'] * wb[..., None]

        # ---- 5-petal florets strewn along the exposed silhouette (lace) + a few inside on clump rims
        fl = []
        if fr >= 1.6 and self.lace > 0:
            er = max(1, int(round(0.8 * fr)))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1))
            outer = (cv2.dilate(solid, ker) > 0) & (solid == 0)
            cell = 1.6 * fr
            gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
            jx = gx_ + mr.uniform(0, cell, gx_.shape)
            jy = gy_ + mr.uniform(0, cell, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            sel = outer[iy, ix] & (mr.random(ix.shape) < 0.55 * self.lace)
            for qx, qy, px_, py_ in zip(ix[sel], iy[sel], jx[sel], jy[sel]):
                # colour of the blossom just inside
                sx_ = int(np.clip(qx - Lx * 0 + 0, 0, w - 1))
                win = a[max(0, qy - 2 * er):qy + 2 * er + 1, max(0, sx_ - 2 * er):sx_ + 2 * er + 1]
                if win.size == 0:
                    continue
                j = np.unravel_index(np.argmax(win), win.shape)
                c = rgb[max(0, qy - 2 * er) + j[0], max(0, sx_ - 2 * er) + j[1]] * mr.uniform(1.0, 1.04)
                fl.append([px_, py_, fr * mr.uniform(0.75, 1.15), mr.uniform(0, 6.283), c[0], c[1], c[2],
                           0.08, 0.0, 0.0])
            # sparse florets inside the lit caps (the body reads as flowers, not paint)
            if self.detail > 0:
                cell2 = 3.0 * fr
                gy_, gx_ = np.mgrid[0:h:cell2, 0:w:cell2]
                jx = gx_ + mr.uniform(0, cell2, gx_.shape)
                jy = gy_ + mr.uniform(0, cell2, gy_.shape)
                ix = np.clip(jx.astype(int), 0, w - 1)
                iy = np.clip(jy.astype(int), 0, h - 1)
                sel = (dist[iy, ix] > 1.5 * fr) & (mr.random(ix.shape) < 0.16 * self.detail)
                for qx, qy, px_, py_ in zip(ix[sel], iy[sel], jx[sel], jy[sel]):
                    c = rgb[qy, qx] * (1.03 if vloc[qy, qx] > 0.6 else 1.05)
                    fl.append([px_, py_, fr * mr.uniform(0.7, 1.0), mr.uniform(0, 6.283), c[0], c[1], c[2], 0.1,
                               0.0, 0.0])
        rgb = np.ascontiguousarray(rgb.astype(np.float32))
        a = np.ascontiguousarray(a.astype(np.float32))
        if fl:
            Fm = np.array(fl, np.float64)
            Fm[:, 4:7] = np.clip(Fm[:, 4:7], 0, 1.15)
            Fm = Fm[np.argsort(Fm[:, 1])]
            stamp_florets(rgb, a, np.ascontiguousarray(Fm))

        # ---- over into the tree buffers
        sub_a = aC[Y0:Y1, X0:X1]
        sub_c = rgbC[Y0:Y1, X0:X1]
        off = 0.08 * R
        shd = _shift(a, -Lx * off, -Ly * off + 0.04 * R)
        shd = cv2.GaussianBlur(shd, (0, 0), max(1.0, 0.015 * R)) * (1 - a) * sub_a
        sub_c[:] = sub_c * (1 - 0.35 * shd[..., None]) + pal['deep'] * (0.35 * shd)[..., None]
        na = a + sub_a * (1 - a)
        sub_c[:] = (rgb * a[..., None] + sub_c * (sub_a * (1 - a))[..., None]) / np.maximum(na, 1e-4)[..., None]
        sub_a[:] = na
        take = a > 0.5
        vC[Y0:Y1, X0:X1][take] = vloc[take]
        sdC[Y0:Y1, X0:X1][take] = -(dist / max(bw, 1.0))[take]

    def _bands(self, v):
        """flat cel bands: lavender underside -> mid pink -> lit pink-white -> hot cap"""
        pal = self.pal
        e = 0.01
        wm = _ss(0.33 - e, 0.33 + e, v)[..., None]
        wl = _ss(0.6 - e, 0.6 + e, v)[..., None]
        wh = _ss(0.97 - e, 0.97 + e, v)[..., None]
        col = pal['lav'] * (1 - wm) + (pal['mid'] * (1 - wl) + pal['lit'] * wl) * wm
        col = col * (1 - wh) + pal['hot'] * wh
        # the deepest undersides lean toward the cool deep lavender
        wd = _ss(0.12, 0.0, v)[..., None]
        col = col * (1 - 0.5 * wd) + pal['deep'] * (0.5 * wd)
        return (col * (0.97 + 0.05 * np.clip(v, 0, 1.2))[..., None]).astype(np.float32)
