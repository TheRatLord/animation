"""s05_sakura round 5: blossom crowns painted as a FEW BIG VALUE MASSES (replaces the floret-wallpaper look).

Per mass: 3-5 big overlapping sub-masses (upper ones behind) painted as flat value shapes - warm pink-white lit
crown, mid pink body, clearly darker cool lavender underside band with a warm bounce at the very bottom - with
crisp overlap edges and a thin contact shade.  5-petal floret detail lives only on the silhouette (lace) and on the
boundaries between value bands / sub-masses; the interiors keep a sparse, low-contrast floret texture (~30%).
Real sky holes (10-60 px at 1080p, irregular, floret rims) through the upper and middle crown; warm peach/gold
transmitted glow on the sun-facing edges plus a 2-3 px bright rim.  Every tree gets a low lavender 'skirt' mass
around the fork so the limbs vanish into shade instead of poking up as bare Y-forks.
"""
import math
import os
import numpy as np
import cv2

import s05_sakura_tree2d as T2
from s05_sakura_tree2d import _noise, _ss, _shift, stamp_florets


class Painter(T2.Painter):
    def __init__(self, k, clumps=(3, 5), hole_m=(0.07, 0.4), n_holes=(0, 0), interior=0.3, glow=1.0,
                 skirt=True, rim_px=2.5, tree_holes=(8, 15), **kw):
        super().__init__(k, **kw)
        self.nclumps = clumps
        self.hole_m = hole_m
        self.n_holes = n_holes
        self.interior = interior
        self.glow = glow
        self.skirt = skirt
        self.rim_px = rim_px
        self.skirt_lift = 1.0
        self.tree_holes = tree_holes
        self.vis_top = None

    # ------------------------------------------------------------------ one mass
    def mass(self, card, m, to_px, rng, kind='mass', occl=None, cutmap=None):
        k = self.k
        H, W = card.shape[:2]
        cx, cy = to_px(np.array([[m['cx'], m['cy']]]))[0]
        rx, ry = m['rx'] * k, m['ry'] * k
        R = 0.5 * (rx + ry)
        fr = self.fr
        pad = int(0.45 * R + 6 * fr + 4)
        X0, X1 = int(max(0, cx - rx - pad)), int(min(W, cx + rx + pad))
        Y0, Y1 = int(max(0, cy - ry - pad)), int(min(H, cy + ry + pad))
        if X1 - X0 < 3 or Y1 - Y0 < 3:
            return
        h, w = Y1 - Y0, X1 - X0
        mr = np.random.default_rng(m['seed'])
        big = kind == 'mass'
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx += X0 - cx + 0.5
        yy += Y0 - cy + 0.5
        Lx, Ly = float(self.L[0]), float(self.L[1])
        # ---- silhouette SDF (px): union of sub-lobes, flatter base, noisy
        nsub = int(mr.integers(5, 8)) if big else int(mr.integers(3, 5))
        sd = np.full((h, w), 1e9, np.float32)
        subs = [(0.0, 0.1, 0.8, 0.72)]
        for i in range(nsub):
            a = math.pi * (0.02 + 0.96 * (i + mr.uniform(-0.25, 0.25)) / max(nsub - 1, 1))
            dd = mr.uniform(0.42, 0.6)
            subs.append((math.cos(a) * dd, -math.sin(a) * dd * 0.85 + 0.05, mr.uniform(0.36, 0.52),
                         mr.uniform(0.36, 0.5)))
        for j in range(2):
            subs.append((mr.uniform(-0.4, 0.4), mr.uniform(0.32, 0.42), mr.uniform(0.45, 0.6), 0.3))
        for (ox, oy, sx, sy) in subs:
            ex, ey = sx * rx, sy * ry
            e = np.sqrt(((xx - ox * rx) / ex) ** 2 + ((yy - oy * ry) / ey) ** 2)
            sd = np.minimum(sd, (e - 1.0) * min(ex, ey))
        nz_lo = _noise(h, w, max(4.0, 0.45 * R), mr) - 0.5
        nz_mid = _noise(h, w, max(3.0, 0.12 * R), mr) - 0.5
        sd = sd + nz_lo * 0.18 * R + nz_mid * 0.06 * R
        bw = max(2.4 * fr, 0.09 * R) * self.lace
        if not big:
            bw = min(bw, 0.16 * R)          # small clumps keep a solid core (they must hide the wood)
        bw_l = bw * (0.35 + 1.2 * _noise(h, w, max(3.0, 2.0 * bw), mr))
        body = sd < 0
        # ---- mass-level light: dome normal from a blurred silhouette
        sig = max(1.0, 0.33 * R)
        fq = max(1.0, sig / 4.0)
        lw, lh = max(4, int(w / fq)), max(4, int(h / fq))
        sm = cv2.resize(_ss(0.0, -1.0, sd).astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), sig / fq)
        gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3) / (8 * fq)
        gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3) / (8 * fq)
        ndl = np.clip(-(gx * Lx + gy * Ly) * sig * 2.5, -1, 1)
        ndl = cv2.resize(ndl, (w, h), interpolation=cv2.INTER_CUBIC)
        hf = (yy + ry) / (2 * ry)
        vg = 0.9 - 1.05 * hf + 0.45 * ndl + 0.3 * nz_lo + 0.08 * nz_mid
        # ---- clumps
        own = np.full((h, w), -1, np.int16)
        vc = np.zeros((h, w), np.float32)
        csh = np.zeros((h, w), np.float32)
        cin = np.full((h, w), 1e9, np.float32)          # distance inside the owning clump's edge
        cfac = np.zeros((h, w), np.float32)             # how much the owning clump's local edge faces up/sun
        capf = np.zeros((h, w), np.float32)             # lit-cap weight of the owning clump
        ux_, uy_ = 0.55 * Lx, 0.55 * Ly - 0.45
        ul = math.hypot(ux_, uy_) + 1e-6
        Ux, Uy = ux_ / ul, uy_ / ul
        clumps = []
        if big:
            nct = int(mr.integers(self.nclumps[0], self.nclumps[1] + 1))
        else:
            nct = int(mr.integers(2, 4))
        cand = np.argwhere(sd < -0.18 * R)
        if len(cand):
            cand = cand[mr.permutation(len(cand))[:600]]
            for (py, px) in cand:
                # strongly varied sizes: a few big heads, several mid, some small
                u_ = mr.random()
                rc = R * (0.55 if u_ < 0.3 else (0.44 if u_ < 0.7 else 0.34)) * mr.uniform(0.9, 1.1)
                if not big:
                    rc = R * mr.uniform(0.45, 0.65)
                qx, qy = px + 0.5 + X0 - cx, py + 0.5 + Y0 - cy
                if all(math.hypot(qx - c[0], qy - c[1]) > 0.62 * (rc + c[2]) for c in clumps):
                    clumps.append((qx, qy, rc))
                if len(clumps) >= nct:
                    break
        clumps.sort(key=lambda c: c[1] + mr.normal(0, 0.15 * c[2]))
        for j, (qx, qy, rc) in enumerate(clumps):
            ex, ey = rc * mr.uniform(1.0, 1.15), rc * mr.uniform(0.85, 1.0)
            bx0 = int(max(0, qx - ex * 1.4 - 3 * fr - (X0 - cx)))
            bx1 = int(min(w, qx + ex * 1.4 + 3 * fr - (X0 - cx) + 1))
            by0 = int(max(0, qy - ey * 1.4 - 3 * fr - (Y0 - cy)))
            by1 = int(min(h, qy + ey * 1.4 + 3 * fr - (Y0 - cy) + 1))
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            hb, wb = by1 - by0, bx1 - bx0
            lx = xx[by0:by1, bx0:bx1] - qx
            ly = yy[by0:by1, bx0:bx1] - qy
            e = np.sqrt((lx / ex) ** 2 + (ly / ey) ** 2)
            sdc = (e - 1.0) * min(ex, ey)
            sdc += (_noise(hb, wb, max(3.0, 0.45 * rc), mr) - 0.5) * 0.35 * rc
            sdc += (_noise(hb, wb, max(1.5, 2.6 * fr), mr) - 0.5) * 1.4 * fr     # floret-scale scallop
            ob = own[by0:by1, bx0:bx1]
            inj = (sdc < 0) & body[by0:by1, bx0:bx1]
            # contact shade on the clump behind, just outside this clump's edge (strongest away from the sun)
            wsh = max(1.5, 0.07 * rc)
            nl = np.maximum(np.sqrt(lx * lx + ly * ly), 1e-3)
            away = np.clip(-(lx * Lx + ly * Ly) / nl, 0, 1)
            band = (~inj) & (ob >= 0) & (sdc < wsh)
            cs = csh[by0:by1, bx0:bx1]
            cs[band] = np.maximum(cs[band], (_ss(wsh, 0.8 * wsh, sdc) * (0.35 + 0.65 * away))[band])
            cs[inj] = 0
            ob[inj] = j
            # clump-local cap light: toward the sun and toward the top
            nx_, ny_ = lx / ex, ly / ey
            # lit cap = the band along the clump's sun/up-facing outline (follows the lacy edge -> crescent)
            dcap = mr.uniform(0.28, 0.42) * rc
            mf = inj.astype(np.float32)
            shf = _shift(mf, -Ux * dcap, -Uy * dcap)
            capm = mf * (1 - cv2.GaussianBlur(shf, (0, 0), max(0.7, 0.05 * rc)))
            capf[by0:by1, bx0:bx1][inj] = capm[inj]
            hs_ = np.clip((ly / ey + 1.0) * 0.5, 0, 1)
            vloc = 1.02 - 1.2 * hs_ + 0.22 * (nx_ * Lx + ny_ * Ly) + 0.12 * capm
            vc[by0:by1, bx0:bx1][inj] = vloc[inj]
            cin[by0:by1, bx0:bx1][inj] = -sdc[inj]
            cfac[by0:by1, bx0:bx1][inj] = np.clip((-ny_ * 0.8 + (nx_ * Lx + ny_ * Ly) * 0.5) / np.maximum(e, 0.3),
                                                  -1, 1)[inj]
        nz_f = _noise(h, w, max(1.5, 1.3 * fr), mr) - 0.5
        v = vg + np.where(own >= 0, 0.36 * (vc - 0.45), 0.0) - 0.28 * csh + 0.07 * nz_f
        if kind == 'hem':
            v = v - 0.1
        elif kind == 'entry':
            v = v - 0.16
        if m.get('skirt'):
            v = v - 0.22
        pal = self.pal
        wl = _ss(0.585, 0.615, v)[..., None]
        wm = _ss(0.175, 0.205, v)[..., None]
        wh = _ss(0.92, 0.95, v)[..., None]
        col = pal['lav'] * (1 - wm) + (pal['mid'] * (1 - wl) + pal['lit'] * wl) * wm
        col = col * (1 - wh) + pal['hot'] * wh
        # a faint gradient inside each band + deepest lavender in the contact shade
        col = col * (0.95 + 0.08 * np.clip(v, 0, 1))[..., None]
        col = col * (1 - 0.3 * csh[..., None]) + pal['deep'] * (0.3 * csh)[..., None]
        # warm transmitted light in the thin edges (lower hem / shadow side: backlit petals glow)
        edge = np.clip(1 + sd / (2.5 * bw), 0, 1) ** 1.4
        tr = edge * (0.35 + 0.65 * _ss(0.35, 1.0, hf)) * (1 - 0.7 * wl[..., 0])
        if not big:
            tr = np.maximum(tr, 0.3 * (1 - wl[..., 0]))
        col = col * (1 - 0.4 * tr[..., None]) + pal['trans'] * (0.4 * tr)[..., None]
        # warm bounce in the lowest part of the lavender underside (light reflected up from the sunlit path)
        lav_w = 1 - wm[..., 0]
        wb = lav_w * _ss(0.62, 1.0, hf) * 0.3
        col = col * (1 - wb[..., None]) + pal['bounce'] * wb[..., None]
        # warm peach/gold transmitted glow in the sun-facing edge band (backlit petals, nearest the sun)
        gw = max(3.0 * fr, 0.16 * R)
        sedge = np.clip(1 + sd / gw, 0, 1) ** 1.3
        sside = _ss(-0.1, 0.6, ndl + 0.35 * (0.5 - hf))
        glow = sedge * sside * self.glow
        col = col * (1 - 0.7 * glow[..., None]) + pal['glow'] * (0.7 * glow)[..., None]
        col = col.astype(np.float32)
        core = _ss(0.9, -0.9, sd + bw_l)
        cov = core.copy()
        fl = []

        def cluster(px, py, crad, bodyc=True, nf=None, noline=0.0, tone=1.0):
            if bodyc:
                fl.append([px, py, 0.62 * crad, 0.0, 1.0, noline, tone])
            nf_ = nf if nf is not None else int(mr.integers(3, 7))
            for _ in range(nf_):
                a_ = mr.uniform(0, 6.283)
                d_ = crad * (0.45 + 0.5 * math.sqrt(mr.uniform(0, 1)))
                fl.append([px + math.cos(a_) * d_, py + math.sin(a_) * d_, fr * mr.uniform(0.75, 1.1),
                           mr.uniform(0, 6.283), 0.0, noline, tone])

        # ---- real sky holes: irregular negative shapes (10-60 px at 1080p) through the upper / middle crown,
        #      rims re-covered by single florets (jagged 5-petal edge, no outline)
        hole = np.zeros((h, w), np.float32)
        if big and fr >= 1.0 and self.holes > 0:
            nh = int(round(self.holes * mr.integers(self.n_holes[0], self.n_holes[1] + 1)))
            cand = np.argwhere((sd < -0.6 * bw) & (sd > -0.42 * R) & (hf < 0.72) & (hf > 0.08))
            placed = []
            if nh > 0 and len(cand):
                cand = cand[mr.permutation(len(cand))]
                for (py, px) in cand:
                    if len(placed) >= nh:
                        break
                    hr = max(1.5, mr.uniform(0.0, 1.0) ** 1.6 * (self.hole_m[1] - self.hole_m[0]) * k +
                             self.hole_m[0] * k) * 0.5
                    if -sd[py, px] < 0.7 * hr + bw * 0.5:
                        continue
                    if any(math.hypot(px - q[0], py - q[1]) < 2.2 * (hr + q[2]) for q in placed):
                        continue
                    placed.append((px, py, hr))
                    pts = [(px + 0.5, py + 0.5, hr)]
                    for _ in range(int(mr.integers(2, 5))):
                        a_ = mr.uniform(0, 6.283)
                        d_ = hr * mr.uniform(0.5, 1.0)
                        pts.append((px + 0.5 + math.cos(a_) * d_, py + 0.5 + math.sin(a_) * d_ * 0.8,
                                    hr * mr.uniform(0.35, 0.7)))
                    ext = int(2.2 * hr + 3 * fr + 4)
                    x0_, x1_ = int(max(0, px - ext)), int(min(w, px + ext + 1))
                    y0_, y1_ = int(max(0, py - ext)), int(min(h, py + ext + 1))
                    gy_, gx_ = np.mgrid[y0_:y1_, x0_:x1_].astype(np.float32)
                    hl = hole[y0_:y1_, x0_:x1_]
                    for (hx, hy, rr_) in pts:
                        dd = np.sqrt((gx_ + 0.5 - hx) ** 2 + (gy_ + 0.5 - hy) ** 2)
                        np.maximum(hl, np.clip(rr_ - dd + 0.5, 0, 1), out=hl)
                    # rim florets: petals bite into the hole so its edge is lacy, not a clean disc
                    per = 2 * math.pi * hr
                    for _ in range(int(max(4, per / (1.6 * fr)))):
                        a_ = mr.uniform(0, 6.283)
                        hx, hy, rr_ = pts[int(mr.integers(len(pts)))]
                        d_ = rr_ + fr * mr.uniform(-0.4, 0.5)
                        fl.append([hx + math.cos(a_) * d_, hy + math.sin(a_) * d_, fr * mr.uniform(0.8, 1.1),
                                   mr.uniform(0, 6.283), 0.0, 1.0, 1.0])
                cov *= (1 - hole)
        # ---- lace clusters along the outline band
        if fr >= 0.7:
            cell = max(2.0, 2.8 * fr)
            gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
            jx = gx_ + mr.uniform(0, cell, gx_.shape)
            jy = gy_ + mr.uniform(0, cell, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            s_ = sd[iy, ix]
            bl = bw_l[iy, ix]
            depth = np.clip((s_ + bl) / (bl + 0.2 * bw + 1e-3), 0, 1)
            keep = (s_ < 0.2 * bw) & (s_ > -bl - 0.8 * fr) & (mr.random(s_.shape) < 0.95 - 0.6 * depth)
            for px, py, dp in zip(jx[keep], jy[keep], depth[keep]):
                cluster(px, py, fr * mr.uniform(1.1, 2.0) * (1.5 - 0.6 * dp))
        # ---- lit clump tops broken by floret clusters (the clump edge reads as flowers)
        if fr >= 1.4 and big:
            cell = 2.3 * fr
            gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
            jx = gx_ + mr.uniform(0, cell, gx_.shape)
            jy = gy_ + mr.uniform(0, cell, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            ok = (own[iy, ix] >= 0) & (cin[iy, ix] < 1.6 * fr) & (cfac[iy, ix] > 0.1) & (sd[iy, ix] < -bw) & \
                (mr.random(ix.shape) < 0.55)
            for px, py in zip(jx[ok], jy[ok]):
                cluster(px, py, fr * mr.uniform(1.0, 1.6), bodyc=False, nf=int(mr.integers(2, 5)), tone=1.02)
        # ---- floret texture inside: clustered groups of lit florets in the top third, faint ones in the shade
        if self.detail > 0 and fr >= 2.0 and big:
            gate = _noise(h, w, max(4.0, 5.0 * fr), mr)
            cell2 = 3.2 * fr
            gy_, gx_ = np.mgrid[0:h:cell2, 0:w:cell2]
            jx = gx_ + mr.uniform(0, cell2, gx_.shape)
            jy = gy_ + mr.uniform(0, cell2, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            vv = v[iy, ix]
            g_ = 0.25 + 0.75 * _ss(0.4, 0.65, gate[iy, ix])
            p = np.where(vv > 0.6, 0.75, np.where(vv > 0.19, 0.5, 0.42)) * g_ * self.detail
            p = np.maximum(p, 0.6 * (capf[iy, ix] > 0.5) * self.detail) * self.interior
            ok = (sd[iy, ix] < -bl_mean(bw_l)) & (mr.random(ix.shape) < p)
            for px, py, v_ in zip(jx[ok], jy[ok], vv[ok]):
                tone = 1.03 if v_ > 0.6 else (1.07 if v_ < 0.19 else 1.04)
                for _ in range(int(mr.integers(2, 5))):
                    fl.append([px + mr.normal(0, 1.0 * fr), py + mr.normal(0, 1.0 * fr), fr * mr.uniform(0.7, 1.0),
                               mr.uniform(0, 6.283), 0.0, 0.0, tone])
        # ---- value-band edges painted as florets: the lighter band spills over the darker one in 5-petal
        #      shapes, so no band boundary is a smooth line (no stripes, no 'shelves')
        flc = []
        if fr >= 1.4:
            b = ((v > 0.19).astype(np.uint8) + (v > 0.6) + (v > 0.935)).astype(np.uint8)
            b = np.where(body, b, 0).astype(np.uint8)
            rad = max(1, int(round(1.3 * fr)))
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
            bmax = cv2.dilate(b, ker)
            edge_ = (bmax > b) & body & (sd < -0.5 * bw)
            sgc = max(1.0, 2.0 * fr)
            colL = {}
            for L_ in (1, 2, 3):
                mL = (b == L_).astype(np.float32)
                if mL.any():
                    colL[L_] = _blur3(col * mL[..., None], sgc) / np.maximum(_blur3(mL, sgc), 1e-3)[..., None]
            cell = 1.5 * fr
            gy_, gx_ = np.mgrid[0:h:cell, 0:w:cell]
            jx = gx_ + mr.uniform(0, cell, gx_.shape)
            jy = gy_ + mr.uniform(0, cell, gy_.shape)
            ix = np.clip(jx.astype(int), 0, w - 1)
            iy = np.clip(jy.astype(int), 0, h - 1)
            ok = edge_[iy, ix] & (mr.random(ix.shape) < 0.8) & (hole[iy, ix] < 0.3)
            for px, py, qx, qy in zip(jx[ok], jy[ok], ix[ok], iy[ok]):
                L_ = int(bmax[qy, qx])
                if L_ not in colL:
                    continue
                c_ = colL[L_][qy, qx] * mr.uniform(0.99, 1.04)
                flc.append([px, py, fr * mr.uniform(0.8, 1.15), mr.uniform(0, 6.283), c_[0], c_[1], c_[2], 0.16,
                            0.05, 0.0])
        # ---- assemble
        rgb = np.ascontiguousarray(col)
        a = np.ascontiguousarray(cov.astype(np.float32))
        if fl:
            F = np.array(fl, np.float64)
            ix = np.clip(F[:, 0].astype(int), 0, w - 1)
            iy = np.clip(F[:, 1].astype(int), 0, h - 1)
            okh = (hole[iy, ix] < 0.3) | (F[:, 5] > 0.5)          # keep the pinholes open
            F, ix, iy = F[okh], ix[okh], iy[okh]
            c = col[iy, ix].astype(np.float64)
            isb = F[:, 4:5] > 0.5
            vv = np.where(isb, 1.0, mr.uniform(0.97, 1.04, (len(F), 1)) * F[:, 6:7])
            c = np.clip(c * vv, 0, 1.1)
            nol = F[:, 5:6] > 0.5
            eye = np.where(isb, 0.0, np.where(nol, 0.12, 0.22))
            ol = np.where(isb | nol, 0.0, 0.08)
            F = np.hstack([F[:, :4], c, eye, ol, F[:, 4:5]])
            order = np.lexsort((-F[:, 1], -F[:, 9]))
            stamp_florets(rgb, a, np.ascontiguousarray(F[order]))
        if flc and not os.environ.get("S05_NOFLC"):
            Fc = np.array(flc, np.float64)
            Fc = Fc[np.argsort(-Fc[:, 1])]
            stamp_florets(rgb, a, np.ascontiguousarray(Fc))
        # ---- 2 px sun-side rim on the silhouette
        rimw = self.rim_px
        sh = _shift(a, Lx * rimw, Ly * rimw)
        rim = np.clip(a - sh, 0, 1) * _ss(-0.25, 0.35, ndl + 0.4 * (1 - hf))
        if not big:
            rim *= 0.6
        rgb = rgb * (1 - 0.85 * rim[..., None]) + pal['rim'] * (0.85 * rim)[..., None]
        sub = card[Y0:Y1, X0:X1]
        if big:
            off = 0.1 * R
            shd = _shift(a, -Lx * off, -Ly * off * 1.2)
            shd = _ss(0.35, 0.65, shd) * (1 - a)
            ex_ = occl[Y0:Y1, X0:X1] if occl is not None else sub[..., 3]
            dk = (shd * ex_ * 0.5)[..., None]
            sub[..., :3] = sub[..., :3] * (1 - dk) + pal['deep'] * sub[..., 3:4] * dk * 0.9
        cut = (hole * (1 - a))[..., None]
        keep_ = (1 - a[..., None]) * (1 - cut)
        sub[..., :3] = rgb * a[..., None] + sub[..., :3] * keep_
        sub[..., 3] = a + sub[..., 3] * keep_[..., 0]
        if occl is not None:
            occl[Y0:Y1, X0:X1] = np.maximum(occl[Y0:Y1, X0:X1] * (1 - cut[..., 0]), a)
        if cutmap is not None:
            cm = cutmap[Y0:Y1, X0:X1]
            cm *= (1 - a)
            np.maximum(cm, cut[..., 0], out=cm)

    # ------------------------------------------------------------------ sky holes through the whole crown
    def _tree_holes(self, mcard, occl, cutmap, tree, to_px, rng):
        k, fr = self.k, self.fr
        H, W = occl.shape
        a = mcard[..., 3]
        solid = (a > 0.95).astype(np.uint8)
        dist = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
        ms = [m for m in tree.masses if not m.get('skirt')]
        ytop = to_px(np.array([[0.0, max(m['cy'] + m['ry'] for m in ms)]]))[0, 1]
        ybot = to_px(np.array([[0.0, min(m['cy'] - m['ry'] for m in ms)]]))[0, 1]
        ys = np.arange(H, dtype=np.float32)[:, None]
        hfc = (ys - ytop) / max(ybot - ytop, 1.0)
        okm = (solid > 0) & (hfc < 0.62) & (hfc > 0.02)
        if self.vis_top is not None:
            yv = to_px(np.array([[0.0, self.vis_top]]))[0, 1]
            okm &= ys > yv + 3 * fr
            okm &= hfc < 0.75
        cand = np.argwhere(okm)
        if not len(cand):
            return
        cand = cand[rng.permutation(len(cand))[:4000]]
        nh = int(rng.integers(self.tree_holes[0], self.tree_holes[1] + 1))
        hole = np.zeros((H, W), np.float32)
        placed = []
        fl = []
        lo, hi = self.hole_m
        for (py, px) in cand:
            if len(placed) >= nh:
                break
            hr = 0.5 * k * (lo + (hi - lo) * rng.uniform(0, 1) ** 1.1)
            hr = max(hr, 2.2 * fr)
            if dist[py, px] < hr + 2.5 * fr:
                continue
            if any(math.hypot(px - q[0], py - q[1]) < 1.8 * (hr + q[2]) + 4 * fr for q in placed):
                continue
            placed.append((px, py, hr))
            pts = [(px + 0.5, py + 0.5, hr)]
            for _ in range(int(rng.integers(2, 5))):
                a_ = rng.uniform(0, 6.283)
                d_ = hr * rng.uniform(0.45, 0.95)
                pts.append((px + 0.5 + math.cos(a_) * d_, py + 0.5 + math.sin(a_) * d_ * 0.75,
                            hr * rng.uniform(0.35, 0.7)))
            ext = int(2.2 * hr + 3 * fr + 4)
            x0_, x1_ = int(max(0, px - ext)), int(min(W, px + ext + 1))
            y0_, y1_ = int(max(0, py - ext)), int(min(H, py + ext + 1))
            gy_, gx_ = np.mgrid[y0_:y1_, x0_:x1_].astype(np.float32)
            hl = hole[y0_:y1_, x0_:x1_]
            fld = np.full(gx_.shape, -1e9, np.float32)
            for (hx, hy, rr_) in pts:
                dd = np.sqrt((gx_ + 0.5 - hx) ** 2 + (gy_ + 0.5 - hy) ** 2)
                np.maximum(fld, rr_ - dd, out=fld)
            hh_, ww_ = fld.shape
            fld += (_noise(hh_, ww_, max(2.0, 0.5 * hr), rng) - 0.5) * 0.9 * hr
            fld += (_noise(hh_, ww_, max(1.5, 1.2 * fr), rng) - 0.5) * 1.4 * fr
            np.maximum(hl, np.clip(fld + 0.5, 0, 1), out=hl)
            for (hx, hy, rr_) in pts:
                per = 2 * math.pi * rr_
                for _ in range(int(max(3, per / (2.6 * fr)))):
                    a_ = rng.uniform(0, 6.283)
                    d_ = rr_ + fr * rng.uniform(0.35, 1.0)
                    fl.append((hx + math.cos(a_) * d_, hy + math.sin(a_) * d_, fr * rng.uniform(0.8, 1.15),
                               rng.uniform(0, 6.283), math.cos(a_), math.sin(a_)))
        if os.environ.get('S05_DBG'):
            print('holes', nh, len(placed), [round(q[2], 1) for q in placed], 'fr', round(fr, 2), 'cand', len(cand))
        if not placed:
            return
        # union of discs -> keep only the parts that do not merge into a later floret rim
        rgb = mcard[..., :3] / np.maximum(a, 1e-4)[..., None]
        # petals around a sky hole are backlit: a warm transmitted ring
        ring = np.clip(_blur3(hole, max(1.5, 1.6 * fr)) * 2.0, 0, 1) * (1 - hole)
        rgb = rgb * (1 - 0.3 * ring[..., None]) + self.pal['glow'] * (0.3 * ring)[..., None]
        na = a * (1 - hole)
        rgb = np.ascontiguousarray(rgb.astype(np.float32))
        na = np.ascontiguousarray(na.astype(np.float32))
        if fl:
            F = np.array(fl, np.float64)
            # colour sampled just outside the hole (the surrounding blossom), a touch lighter / warmer
            sx_ = np.clip((F[:, 0] + F[:, 4] * 2.5 * fr).astype(int), 0, W - 1)
            sy_ = np.clip((F[:, 1] + F[:, 5] * 2.5 * fr).astype(int), 0, H - 1)
            c = rgb[sy_, sx_].astype(np.float64) * rng.uniform(1.0, 1.06, (len(F), 1))
            Fs = np.hstack([F[:, :4], np.clip(c, 0, 1.1), np.full((len(F), 1), 0.12), np.zeros((len(F), 1)),
                            np.zeros((len(F), 1))])
            stamp_florets(rgb, na, np.ascontiguousarray(Fs))
        mcard[..., :3] = rgb * na[..., None]
        mcard[..., 3] = na
        cut = np.clip(a - na, 0, 1)
        occl *= (1 - cut)
        np.maximum(cutmap, cut, out=cutmap)

    # ------------------------------------------------------------------ full tree
    def paint(self, tree, rng, ss_margin=4):
        # 0) a low lavender 'skirt' mass around the fork (painted first = behind): the limbs vanish into
        #    shaded blossom right above the fork instead of showing bare Y-forks
        if self.skirt and not getattr(tree, '_skirted', False):
            Pt_, rt_ = tree.limbs[0]
            tx_, ty_ = float(Pt_[-1, 0]), float(Pt_[-1, 1])
            xs_ = [m['cx'] for m in tree.masses]
            cxs = 0.4 * tx_ + 0.6 * float(np.mean(xs_))
            span = max(max(m['cx'] + m['rx'] for m in tree.masses) - min(m['cx'] - m['rx'] for m in tree.masses), 1.0)
            ry_s = max(0.55, 0.28 * span)
            bot = min(m['cy'] - m['ry'] for m in tree.masses)
            yb_ = ty_ + self.skirt_lift * float(rt_[-4]) * 1.6
            yb_ = min(yb_, bot + 0.4)
            tree.masses.insert(0, dict(cx=cxs, cy=yb_ + ry_s, rx=0.5 * span, ry=ry_s,
                                       seed=int(rng.integers(1 << 30)), skirt=True))
            tree._skirted = True
        k = self.k
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * ss_margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * ss_margin
        ox = -x0 * k + ss_margin
        oy = y1 * k + ss_margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        # 1) blossom masses on their own card
        mcard = np.zeros((H, W, 4), np.float32)
        occl = np.zeros((H, W), np.float32)
        cutmap = np.zeros((H, W), np.float32)
        for m in tree.masses:
            self.mass(mcard, m, to_px, rng, 'mass', occl, cutmap)
        if self.tree_holes is not None and self.fr >= 1.0:
            self._tree_holes(mcard, occl, cutmap, tree, to_px, rng)
        # 2) limbs truncated where they first dive into a mass; limbs that never reach one are dropped
        Pt, rt = tree.limbs[0]
        rt = rt.copy()
        rt[-3:] *= np.array([0.9, 0.7, 0.45])           # the trunk top thins into the fork (no flat cut end)
        limbs = [(Pt, rt)]
        entry = []
        solid = cv2.distanceTransform((occl > 0.92).astype(np.uint8), cv2.DIST_L2, 5)
        for P, r in tree.limbs[1:]:
            pp = to_px(P)
            n = len(pp)
            tt = np.linspace(0, 1, 6 * n)
            xs = np.interp(tt, np.linspace(0, 1, n), pp[:, 0])
            ys = np.interp(tt, np.linspace(0, 1, n), pp[:, 1])
            rs = np.interp(tt, np.linspace(0, 1, n), r) * k
            ix = np.clip(xs.astype(int), 0, W - 1)
            iy = np.clip(ys.astype(int), 0, H - 1)
            inside = occl[iy, ix] > 0.6
            if not inside.any():
                continue
            j = int(np.argmax(inside))
            if j == 0:
                continue
            # go a little under the blossom edge so the wood never ends visibly
            # run on until the wood is well inside SOLID blossom (lace / pinholes never show a cut end)
            deep = np.nonzero((solid[iy, ix] > 1.5 * rs + 2.0) & (np.arange(len(tt)) >= j))[0]
            j2 = int(deep[0]) if len(deep) else len(tt) - 1
            keep = np.arange(len(tt)) <= j2
            rk = rs[keep].copy()
            nk = len(rk)
            # re-taper over the kept length: the wood thins visibly before it dives into the blossom
            ll = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs[keep]), np.diff(ys[keep])))])
            rk = rk[0] * (1 - 0.72 * (ll / max(ll[-1], 1e-3)) ** 0.85)
            # small blossom clusters sitting on the last visible stretch of the limb
            lj = ll[min(j, nk - 1)]
            for f_ in (0.62, 0.8):
                jj = int(np.searchsorted(ll, lj * f_))
                if jj <= 0 or jj >= nk or rk[jj] * 2.2 < 2.5 * self.fr:
                    continue
                if rng.random() < 0.8:
                    Rc = max(2.2 * rk[jj], 3.0 * self.fr) / k
                    entry.append(dict(cx=float((xs[jj] - ox) / k), cy=float((oy - ys[jj]) / k + 0.3 * Rc),
                                      rx=Rc * 1.3, ry=Rc * 0.9, seed=int(rng.integers(1 << 30))))
            Pm = np.stack([(xs[keep] - ox) / k, (oy - ys[keep]) / k], 1)
            limbs.append((Pm, rk / k))
            R_ = max(5.0 * rs[j] / k, 0.42)
            # centred a little back along the limb so the clump swallows the point where the wood disappears
            back = np.nonzero(np.hypot(xs[:j + 1] - xs[j], ys[:j + 1] - ys[j]) <= 0.45 * R_ * k)[0]
            jb = int(back[0]) if len(back) else j
            entry.append(dict(cx=float((xs[jb] - ox) / k), cy=float((oy - ys[jb]) / k), rx=R_ * 1.3,
                              ry=R_ * 0.95, seed=int(rng.integers(1 << 30))))
        # the trunk top (where the limbs fork) sits in a blossom clump as well
        tp_ = to_px(Pt[-1:])[0]
        if False:
            R_ = max(2.6 * rt[-4], 0.4)
            entry.append(dict(cx=float(Pt[-1, 0]), cy=float(Pt[-1, 1] + 0.15 * R_), rx=R_ * 1.35, ry=R_ * 0.9,
                              seed=int(rng.integers(1 << 30))))
        ys_ = np.arange(H, dtype=np.float32)[:, None]
        cb = min(m['cy'] - m['ry'] for m in tree.masses)
        yb = oy - cb * k
        shade = 0.8 * _ss(yb + 0.7 * k, yb - 0.2 * k, ys_) * np.ones((1, W), np.float32)
        col, av = self.bark_layer(H, W, limbs, to_px, rng, shade.astype(np.float32))
        if os.environ.get("S05_NOBARK"):
            av = av * 0
        av = av * (1 - cutmap)
        card = np.zeros((H, W, 4), np.float32)
        card[..., :3] = mcard[..., :3] + col * (av * (1 - mcard[..., 3]))[..., None]
        card[..., 3] = mcard[..., 3] + av * (1 - mcard[..., 3])
        # 3) hem clumps: only those that sit on wood or against a mass (no floating puffs)
        support = ((occl > 0.5) | (av > 0.5)).astype(np.uint8)
        dist = cv2.distanceTransform(1 - support, cv2.DIST_L2, 5)
        hem = []
        for m in tree.hem:
            px, py = to_px(np.array([[m['cx'], m['cy']]]))[0]
            ix, iy = int(np.clip(px, 0, W - 1)), int(np.clip(py, 0, H - 1))
            if dist[iy, ix] < 0.35 * m['rx'] * k:
                hem.append(m)
        ccard = np.zeros((H, W, 4), np.float32)
        for m in entry:
            self.mass(ccard, m, to_px, rng, 'entry', None, None)
        for m in hem:
            self.mass(ccard, m, to_px, rng, 'hem', None, None)
        ca_ = ccard[..., 3:4]
        card = ccard + card * (1 - ca_)
        # 4) no free-floating wood: walk every limb from its root; after a hidden stretch longer than gap_max
        #    the rest of the limb is disconnected from the tree -> repaint any of it that still shows as blossom
        barkw = av * (1 - mcard[..., 3]) * (1 - ca_[..., 0])
        vis = barkw > 0.35
        gap_max = max(0.4 * k, 8.0)
        rem = np.zeros((H, W), np.uint8)
        for li, (P_, r_) in enumerate(limbs):
            if li == 0:
                continue
            pp = to_px(P_)
            seg = np.hypot(np.diff(pp[:, 0]), np.diff(pp[:, 1]))
            ll = np.concatenate([[0], np.cumsum(seg)])
            if ll[-1] < 2:
                continue
            n = int(ll[-1]) + 1
            tt = np.linspace(0, ll[-1], n)
            xs = np.interp(tt, ll, pp[:, 0])
            ys = np.interp(tt, ll, pp[:, 1])
            rr = np.interp(tt, ll, r_ * k)
            v_ = vis[np.clip(ys.astype(int), 0, H - 1), np.clip(xs.astype(int), 0, W - 1)]
            gap = 0.0
            cut_i = None
            for i in range(n):
                if v_[i]:
                    gap = 0.0
                else:
                    gap += 1.0
                    if gap > gap_max:
                        cut_i = i
                        break
            if cut_i is None:
                continue
            pts = np.stack([xs[cut_i:], ys[cut_i:]], 1)
            for a_, b_, r0 in zip(pts[:-1], pts[1:], rr[cut_i:-1]):
                cv2.line(rem, (int(round(a_[0])), int(round(a_[1]))), (int(round(b_[0])), int(round(b_[1]))), 1,
                         int(2 * r0 + 4))
        remf = (rem > 0) & (barkw > 0.02)
        if remf.any():
            # blossom-only layer (premultiplied) -> fill colour from the surrounding flowers
            bl_ = ccard + mcard * (1 - ca_)
            sgf = max(3.0, 0.12 * k)
            fa = _blur3(bl_[..., 3], sgf)
            fc = _blur3(bl_[..., :3], sgf) / np.maximum(fa, 1e-3)[..., None]
            wgt = (barkw * remf)[..., None]
            inside = np.clip((fa - 0.35) / 0.3, 0, 1)[..., None]
            # inside the canopy: the wood becomes shaded blossom; out in the open it is simply removed
            # (round 5) inside the crown it stays as a thin branch silhouette in the blossoms' shade (flat,
            # no blurred fill); out in the open it is removed
            card[..., :3] -= col * wgt * (1 - inside)
            card[..., :3] += (np.float32(0.45) * self.pal['deep'] - 0.45 * col) * wgt * inside
            card[..., 3] -= (barkw * remf) * (1 - inside[..., 0])
        cut = _ss(oy + 0.03 * k + 1.0, oy + 0.03 * k - 1.0, np.arange(H, dtype=np.float32))[:, None]
        card *= cut[..., None]
        return card, ox, oy


def bl_mean(bw_l):
    return float(np.mean(bw_l))


def _blur3(a, sig):
    return cv2.GaussianBlur(a, (0, 0), sig)
