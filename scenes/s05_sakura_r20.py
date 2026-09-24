"""s05_sakura round 20: Painter20 = LACY floret crowns (cm5_01 study).

Reviewer (round 19 FAIL): 'dabs are opaque roundish 15-30 px polygon blobs -> popcorn / clay; mass far too dense
and solid; change the primitive to small 5-petal florets (6-12 px) grouped in clusters of 5-20, leave 25-35 % of
the crown as lacy see-through sky gaps, feather the silhouette into loose sprays of florets along twig tips';
'no rim glow on the sun-facing crowns'; 'violet shadow core hard-edged everywhere -> the down side loses its edges
into a soft violet-magenta haze'; 'bark = tiled diagonal hatch -> vertical painted bark value bands, warm lit
left edge, cool violet right side, irregular dark lenticel marks'; 'blossom clusters overlapping in front of the
trunk and the main limb so the branches sit inside the mass'.

  * no solid mass underpaint and no dark core blob per head: every cluster is 5-20 distinct five-petal florets
    (radius 3-6 px at 1080p) on a small umbel layout with spacing so each flower reads, value from the per-mass
    field in three tiers; a few darker florets behind give the inner clump separation;
  * lacy crown: sky between florets; a faint pink veil (alpha <= 0.2) keeps the gaps atmospheric, not holes;
  * feathered silhouette: many twig sprays with 2-6 single florets past the crown edge;
  * down side: a soft violet-magenta haze under the lower clusters (lost edge), sun side: near-white cream rim
    + warm halation;
  * dobuki: small clusters sprouting from the upper trunk and along the scaffold limbs, drawn IN FRONT of the wood;
  * bark: vertical painted value bands (long streaks along the wood), warm lit sun-side edge, cool violet
    shade side, sparse irregular dark lenticel marks; no ring bands.
"""
import math

import numpy as np
import cv2

from s05_sakura_r16 import _splat, _band_col, TRANS
from s05_sakura_r15 import _raster_wood, _unit, _rot, _sstep
from s05_sakura_r17 import PEACH, HALO, _kmeans
from s05_sakura_r19 import Tree18, Painter19, BANDS18, _tier

BANDS20 = np.array([
    (0.46, 0.28, 0.6),     # 0 violet core
    (0.8, 0.38, 0.62),     # 1 magenta-mauve shade
    (0.96, 0.54, 0.72),    # 2 rose body
    (1.0, 0.7, 0.8),       # 3 light pink
    (1.03, 0.84, 0.88),    # 4 pale pink tip
    (1.1, 1.0, 0.96),      # 5 hot cream (sun-exposed tips only)
], np.float32)
BANDS20_FAR = (BANDS20 * 0.72 + np.array([0.9, 0.86, 0.95], np.float32) * 0.28).astype(np.float32)
HAZE_V = np.array([0.72, 0.5, 0.78], np.float32)       # soft violet-magenta under-haze
VEIL = np.array([0.98, 0.78, 0.86], np.float32)


class Tree20(Tree18):
    def _gate(self, rng):
        """round 20: lacier crowns - more / bigger low-frequency sky holes, thinner crown edge"""
        cl = [c for c in self.clusters if c['y'] > 0.75 * self.ht]
        if not cl:
            self.clusters = cl
            return
        X = np.array([c['x'] for c in cl])
        Y = np.array([c['y'] for c in cl])
        cx, cy = np.median(X), np.median(Y)
        sx_, sy_ = X.std() + 1e-6, Y.std() + 1e-6
        nb = 12
        bx = rng.uniform(X.min(), X.max(), nb)
        by = rng.uniform(Y.min(), Y.max(), nb)
        br = rng.uniform(0.22, 0.5, nb) * self.s
        keep = []
        for c in cl:
            dn = math.hypot((c['x'] - cx) / sx_, (c['y'] - cy) / sy_)
            p = 0.93 - 0.22 * max(0.0, dn - 0.95)
            hole = np.exp(-((c['x'] - bx) ** 2 + (c['y'] - by) ** 2) / (br * br)).max()
            p *= 1.0 - 0.8 * hole
            if rng.random() < p:
                keep.append(c)
        self.clusters = keep

    """Tree18 + dobuki: small blossom clusters sprouting directly from the upper trunk and along the scaffold
    limbs (drawn in front of the wood), so the thick wood sits inside the blossom."""

    def __init__(self, rng, *a, dobuki=1.0, **kw):
        super().__init__(rng, *a, **kw)
        if dobuki <= 0:
            return
        s = self.s
        Rb = 0.1 * s * self.cl
        add = []
        for w in self.wood:
            if w['d'] > 1:
                continue
            P = w['P']
            n = len(P)
            if w['d'] == 0:
                t0, gap = 0.62, 2.8          # upper trunk only
            else:
                t0, gap = 0.2, 2.2
            L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
            t = t0 + rng.uniform(0, 0.1)
            while t < 0.98:
                i = int(t * (n - 1))
                td = _unit(P[min(i + 1, n - 1)] - P[max(i - 1, 0)])
                nrm = np.array([-td[1], td[0]])
                rr = float(w['r'][i])
                R = Rb * rng.uniform(0.5, 0.95)
                side = rng.choice([-1.0, 1.0]) * rng.uniform(0.0, 0.9)
                c = P[i] + nrm * side * rr + np.array([0.0, -0.3 * R])
                if rng.random() < 0.8 * dobuki:
                    add.append(dict(x=float(c[0]), y=float(c[1]), R=float(R), m=w['m'], d=max(w['d'], 1),
                                    tx=float(td[0]), ty=float(td[1]), front=True, force_front=True,
                                    seed=int(rng.integers(1 << 30))))
                t += (R / max(L, 1e-3)) * rng.uniform(gap, gap * 2.2)
        self.clusters = self.clusters + add


class Painter20(Painter19):
    FRONT = 0.4

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bands = BANDS20_FAR if self.far else BANDS20

    # ------------------------------------------------------------------ florets
    def _floret_r(self, R, fs=1.0):
        px = self.px
        return float(np.clip(0.25 * R * fs, 1.1 * px, 6.2 * px))

    def _cluster_stamps(self, c, v, trans, rng):
        """5-20 separate five-petal florets on 1-4 small umbels hanging off the twig."""
        px = self.px
        Ls = self.Ls
        bands = self.bands
        R = c['pr']
        fr0 = self._floret_r(R, c.get('fs', 1.0))
        ex = c.get('ex', 0.0)
        mv = c.get('mv', None)
        tx, ty = c['tx'], -c['ty']
        # number of florets: 5-20, fewer on small clusters
        nfl = int(np.clip(round((R / max(fr0, 1e-3)) ** 2 * rng.uniform(0.55, 0.85)), 5, 20))
        nu = int(np.clip(round(nfl / rng.uniform(4.0, 6.0)), 1, 4))
        umb = []
        for _ in range(nu):
            u = rng.uniform(-1, 1)
            w_ = rng.normal(0, 0.3)
            ux = c['px'] + (u * tx - w_ * ty) * R * 0.62
            uy = c['py'] + (u * ty + w_ * tx) * R * 0.62
            umb.append((ux, uy, R * rng.uniform(0.38, 0.6)))
        fl = []
        placed = []
        tries = 0
        while len(fl) < nfl and tries < nfl * 6:
            tries += 1
            ux, uy, ur = umb[int(rng.integers(nu))]
            a = rng.uniform(0, 2 * math.pi)
            rr = ur * math.sqrt(rng.uniform(0.05, 1.0))
            fx, fy = ux + math.cos(a) * rr, uy + math.sin(a) * rr + 0.15 * ur
            fr = fr0 * rng.uniform(0.8, 1.2)
            ok = True
            for (qx, qy, qr) in placed:
                if (fx - qx) ** 2 + (fy - qy) ** 2 < (0.95 * (fr + qr) * 0.62) ** 2:
                    ok = False
                    break
            if not ok:
                continue
            placed.append((fx, fy, fr))
            s_ = ((fx - c['px']) * Ls[0] + (fy - c['py']) * Ls[1]) / max(R, 1e-3)      # +1 on the sun side
            fl.append((s_, fx, fy, fr, rr / max(ur, 1e-3)))
        fl.sort(key=lambda q: q[0])                  # far side first, sun-side florets on top
        st = []
        for (s_, fx, fy, fr, rn) in fl:
            vh = v + 0.35 * s_ + rng.normal(0, 0.12)
            if mv is not None:
                vh += -0.8 * (fy - c['py']) / max(mv[2], 1.0)
            vq = _tier(vh)
            ft = vq + 0.4 * s_ + 0.25 * rn * max(s_, 0.0) + rng.normal(0, 0.18)
            ft += ex * 1.3 * max(s_ + 0.2, 0.0) * (vq > 2.0)
            ft = float(np.clip(ft, 0.0, 5.0))
            fc = _band_col(bands, np.array(ft)).astype(np.float32)
            fc = fc + (fc - fc.mean()) * rng.uniform(-0.05, 0.2)
            if trans > 0:
                fc = fc + (TRANS * 1.04 - fc) * trans * 0.6 * max(s_ + 0.3, 0.0)
            # a darker floret just behind (away from the sun): the inner clump separation, not a blob
            if rng.random() < 0.35:
                dc = _band_col(bands, np.array(max(ft - 1.6, 0.0))).astype(np.float32)
                st.append((fx - Ls[0] * fr * 0.7, fy - Ls[1] * fr * 0.7 + 0.3 * fr, fr * 0.95, rng.uniform(0, 6.3),
                           5.0, 0.5, 0.75 * px, dc[0], dc[1], dc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
            cres = (_band_col(bands, np.array(min(ft + 1.1, 5.0))) - fc) * 0.7
            # deeper petal notches -> distinct five-petal blossoms
            st.append((fx, fy, fr, rng.uniform(0, 6.3), 5.0, 0.5, 0.75 * px,
                       fc[0], fc[1], fc[2], Ls[0], Ls[1], 0.25, cres[0], cres[1], cres[2], 1.0))
            # a small darker eye (stamens) on the bigger, lit florets
            if fr > 3.2 * px and ft > 2.5 and rng.random() < 0.5:
                ec = _band_col(bands, np.array(max(ft - 1.8, 0.0))).astype(np.float32)
                st.append((fx, fy, fr * 0.22, 0.0, 5.0, 0.2, 0.7 * px, ec[0], ec[1], ec[2], 0.0, 0.0, 9.0,
                           0.0, 0.0, 0.0, 0.7))
        return st

    def paint(self, tree, rng, margin=10):
        hr = np.random.default_rng(len(tree.clusters) * 7 + 3)
        for c in tree.clusters:
            c['front'] = bool(c.get('force_front', False) or hr.random() < self.FRONT)
        return self._paint20(tree, rng, margin)

    # ------------------------------------------------------------------ wood: vertical bark value bands
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
            nP = len(P)
            for i in range(nP - 1):
                pri = float(w['d'])
                if w['d'] == 0 and i > 0.88 * nP:
                    pri = 1.5                     # the scaffolds grow OUT of the trunk top (no capsule cap)
                segs.append((P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1], max(r[i], rmin), max(r[i + 1], rmin),
                             v[i], v[i + 1], pri, float(w['rid']), Nn[i, 0], Nn[i, 1], Nn[i + 1, 0],
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
        thin = np.clip(RR / (0.5 * px), 0.35, 1.0)
        cov = cov * np.where(RR > 0, thin, 1.0).astype(np.float32)
        wp = self.wp
        Ls = self.Ls
        facing = S * (NX * Ls[0] + NY * Ls[1])          # +1 on the sun-side edge
        a_s = np.abs(S)
        self._bark_tex(rng)
        bt = self._blob_tex
        th_, tw_ = bt.shape
        rm = np.maximum(RR / self.k, 1e-3)
        ang = np.arcsin(np.clip(S, -1, 1))
        arc = ang * rm
        # low-frequency wobble so the terminator is painted, not ruled
        wob = cv2.remap(bt, ((arc / 0.2 + RID * 3.0) % tw_).astype(np.float32),
                        ((Vv / 0.5 + RID * 17.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)
        fct = facing + 0.2 * wob
        sh = np.array([0.2, 0.15, 0.3], np.float32)          # cool violet shade side
        mid = np.array([0.34, 0.26, 0.28], np.float32)       # warm grey-brown body
        lit = np.array([0.58, 0.44, 0.36], np.float32)       # warm lit sun-side band
        if self.far:
            sh, mid, lit = sh * 0.6 + 0.4 * np.array([0.4, 0.4, 0.52]), mid * 0.6 + 0.4 * np.array([0.46, 0.44, 0.52]), \
                lit * 0.7 + 0.3 * np.array([0.6, 0.56, 0.6])
            sh, mid, lit = [np.asarray(q, np.float32) for q in (sh, mid, lit)]
        col = np.broadcast_to(sh, (H, W, 3)).astype(np.float32).copy()
        hard = getattr(self, 'WOOD_HARD', False)
        if hard:
            # round 21: painted bark - one flat dark shadow side, a hard terminator, a hard lit edge
            fct = facing + 0.08 * wob
            if not self.far:
                sh = np.array([0.16, 0.11, 0.2], np.float32)
                mid = np.array([0.44, 0.33, 0.31], np.float32)
                lit = np.array([0.7, 0.53, 0.42], np.float32)
        m1 = _sstep((fct + (0.22 if hard else 0.4)) / (0.035 if hard else 0.2) + 0.5)
        col += (mid - sh) * m1[..., None]
        big = np.clip((RR - 2.0 * px) / (3 * px), 0, 1)
        m2 = _sstep((fct - (0.28 if hard else 0.2)) / (0.03 if hard else 0.14) + 0.5) * big
        col += (lit - mid) * m2[..., None]
        # reflected cool sky light on the far shade edge
        cb = _sstep((-fct - 0.75) / 0.12 + 0.5) * np.clip((RR - 3 * px) / (4 * px), 0, 1)
        col += (np.array([0.38, 0.38, 0.6], np.float32) - col) * (cb * (0.0 if hard else 0.55))[..., None]
        thick = np.clip((RR - 3.0 * px) / (6 * px), 0, 1)
        if thick.max() > 0:
            # vertical painted value bands: long streaks ALONG the wood (varying across it), broken slowly
            bn = cv2.remap(bt, ((arc / 0.009 + RID * 7.0) % tw_).astype(np.float32),
                           ((Vv / 0.9 + RID * 29.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            bn2 = cv2.remap(bt, ((arc / 0.0045 + RID * 11.0) % tw_).astype(np.float32),
                            ((Vv / 0.35 + RID * 5.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            vb = 0.65 * bn + 0.35 * bn2
            lo = _sstep((-vb - 0.0) / 0.05 + 0.5)          # darker streaks
            hi = _sstep((vb - 0.07) / 0.05 + 0.5)           # lighter ridges (more on the lit side)
            fade = np.clip(1.2 - a_s, 0, 1) * thick
            if hard:
                fade = fade * (0.25 + 0.75 * m1)          # the shadow side stays one flat value
            col = col * (1 - (0.38 * lo * fade)[..., None])
            col += (col * 1.5 + np.array([0.03, 0.02, 0.03], np.float32) - col) * (hi * fade * 0.6)[..., None]
            # thin dark vertical fissures inside the streaks (painted with a fine brush)
            fz = cv2.remap(bt, ((arc / 0.003 + RID * 23.0) % tw_).astype(np.float32),
                           ((Vv / 0.6 + RID * 41.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            fis = _sstep((0.035 - np.abs(fz)) / 0.02 + 0.5) * _sstep((bn2 + 0.05) / 0.1 + 0.5)
            col += (np.array([0.12, 0.08, 0.13], np.float32) - col) * (fis * fade * 0.55)[..., None]
            # sparse irregular dark lenticel marks (varied length / thickness, not a lattice)
            sp = 0.085
            q = Vv / sp + 0.6 * bn
            jj = np.floor(q)
            fq = q - jj
            ca = arc / 0.06 + np.sin(jj * 3.1 + RID) * 9.0
            ci = np.floor(ca)
            fa = ca - ci
            h2 = np.sin(jj * 12.9898 + ci * 4.1414 + RID * 78.233) * 43758.5453
            h2 = h2 - np.floor(h2)
            h3 = np.sin(jj * 7.13 + ci * 19.7 + RID * 3.3) * 2451.77
            h3 = h3 - np.floor(h3)
            wq = 0.06 + 0.1 * h3
            lx0, lx1 = 0.1 + 0.3 * h3, 0.55 + 0.4 * (1 - h3)
            dash = _sstep((wq - np.abs(fq - 0.5)) / 0.03 + 0.5) * (h2 < 0.34) * \
                _sstep((fa - lx0) / 0.05 + 0.5) * _sstep((lx1 - fa) / 0.05 + 0.5)
            dash *= np.clip(1.1 - a_s, 0, 1)
            col += (np.array([0.1, 0.07, 0.11], np.float32) - col) * (dash * thick * 0.8)[..., None]
            # moss low on the shade side
            mn = cv2.remap(bt, ((arc / 0.09 + RID * 5.0) % tw_).astype(np.float32),
                           ((Vv / 0.35 + RID * 13.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            low = np.clip(1.4 - Vv / 1.4, 0.2, 1.0)
            moss = _sstep((mn - 0.55) / 0.1 + 0.5) * np.clip(0.2 - fct, 0, 1) * low
            col += (wp['moss'] - col) * (moss * thick * 0.5)[..., None]
        # warm lit rim on the sun side
        rw = np.clip((1.8 * px) / np.maximum(RR, 0.5), 0.05, 0.6)
        rimk = _sstep((a_s - (1.0 - rw)) / (0.4 * rw) + 0.5) * _sstep((facing - 0.3) / 0.2 + 0.5)
        rimk *= np.clip((RR - 1.4 * px) / (1.6 * px), 0, 1)          # thin twigs stay dark (no pale threads)
        inside = np.clip(cv2.resize(fld['D2'], (W, H), interpolation=cv2.INTER_LINEAR) * 1.6, 0, 1)
        col += (wp['rim'] - col) * (rimk * (0.95 - 0.45 * inside))[..., None]
        tn = 1.0 - np.clip((RR - 1.0 * px) / (2.0 * px), 0, 1)
        col += (np.array([0.22, 0.15, 0.2], np.float32) - col) * (tn * 0.7)[..., None]
        col += (wp['inner'] - col) * (inside * 0.45 * (1 - thick))[..., None]
        return col.astype(np.float32), cov

    # ------------------------------------------------------------------ paint
    def _paint20(self, tree, rng, margin=10):
        k = self.k
        px = self.px
        Ls = self.Ls
        margin = int(margin + 30 * px)
        x0, x1, y0, y1 = tree.extent()
        W = int(math.ceil((x1 - x0) * k)) + 2 * margin
        H = int(math.ceil((y1 - y0) * k)) + 2 * margin
        ox = -x0 * k + margin
        oy = y1 * k + margin

        def to_px(P):
            P = np.asarray(P, np.float64)
            return np.stack([ox + P[:, 0] * k, oy - P[:, 1] * k], 1)

        cl = tree.clusters
        mr = np.random.default_rng(int(k * 13) % 100003 + len(cl))
        for c in cl:
            c['px'] = ox + c['x'] * k
            c['py'] = oy - c['y'] * k
            c['pr'] = max(c['R'] * k, 2.5 * px)
        n = len(cl)
        X = np.array([c['px'] for c in cl])
        Y = np.array([c['py'] for c in cl])
        # ---- masses
        nm_t = self.masses or int(np.clip(round(n / 45), 3, 6))
        q4 = 4
        m4 = np.zeros((H // q4 + 1, W // q4 + 1), np.uint8)
        medr0 = float(np.median([c['pr'] for c in cl])) if n else 4 * px
        for c in cl:
            cv2.circle(m4, (int(c['px'] / q4), int(c['py'] / q4)), max(1, int(c['pr'] * 0.9 / q4)), 255, -1)
        kr4 = max(3, int(1.6 * medr0 / q4)) | 1
        m4 = cv2.morphologyEx(m4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr4, kr4)))
        ncc, cc = cv2.connectedComponents(m4)
        comp = cc[np.clip((Y / q4).astype(int), 0, cc.shape[0] - 1), np.clip((X / q4).astype(int), 0, cc.shape[1] - 1)]
        lab = np.zeros(n, int)
        nxt = 0
        per = max(n / nm_t, 20)
        for cid in np.unique(comp):
            sel = np.nonzero(comp == cid)[0]
            kk_ = int(np.clip(round(len(sel) / per), 1, 6))
            if kk_ > 1:
                l2, _ = _kmeans(np.stack([X[sel], Y[sel] * 1.6], 1), kk_, mr)
                for t_ in np.unique(l2):
                    lab[sel[l2 == t_]] = nxt
                    nxt += 1
            else:
                lab[sel] = nxt
                nxt += 1
        nm = nxt
        info = []
        for j in range(nm):
            sel = lab == j
            if not sel.any():
                info.append(None)
                continue
            xs, ys = X[sel], Y[sel]
            info.append(dict(cx=float(xs.mean()), cy=float(ys.mean()),
                             rx=float(max(np.percentile(np.abs(xs - xs.mean()), 90), 1.5 * medr0)),
                             ry=float(max(np.percentile(np.abs(ys - ys.mean()), 90), 1.5 * medr0)),
                             base=float(mr.normal(0, 0.25))))
        rr = np.zeros(n)
        for i, c in enumerate(cl):
            m = info[lab[i]]
            dn = math.hypot((c['px'] - m['cx']) / m['rx'], (c['py'] - m['cy']) / m['ry'])
            f = float(np.clip(math.exp(mr.normal(0.0, 0.3)) * (1.35 - 0.5 * min(dn, 1.3)), 0.55, 1.65))
            c['pr'] = max(c['pr'] * f, 2.5 * px)
            rr[i] = dn
        ytop, ybot = Y.min(), Y.max()
        for m in info:
            if m is not None:
                m['hg'] = 1.0 - (m['cy'] - ytop) / max(ybot - ytop, 1.0)
        fld = self._field(cl, H, W)
        Dsun = self._at(fld['Dsun'], X, Y)
        D1 = self._at(fld['D1'], X, Y)
        face = self._at(fld['face'], X, Y)
        v = np.zeros(n)
        for i, c in enumerate(cl):
            m = info[lab[i]]
            tv = (c['py'] - m['cy']) / m['ry']
            tl = ((c['px'] - m['cx']) * Ls[0] + (c['py'] - m['cy']) * Ls[1]) / max(m['rx'], m['ry'])
            v[i] = 2.8 - 1.4 * tv + 0.8 * tl - 1.4 * max(tv - 0.1, 0.0) + 0.9 * (m['hg'] - 0.5) + m['base']
            c['mv'] = (m['cx'], m['cy'], m['ry'])
        v = v - 0.9 * np.clip(Dsun - 0.6, 0, 1)
        ymid = 0.5 * (ytop + ybot)
        v = v - 0.6 * np.clip((Y - ymid) / max(ybot - ymid, 1.0), 0, 1)
        exs = np.clip((0.5 - Dsun) / 0.35, 0, 1)
        v = v + 0.9 * exs
        for i, c in enumerate(cl):
            c['ex'] = float(exs[i])
            c['fs'] = float(np.clip(math.exp(mr.normal(0.0, 0.25)) * (1.3 - 0.45 * min(rr[i], 1.2)), 0.6, 1.6))
        front = np.array([c['front'] for c in cl])
        v = v + np.where(front, 0.15, -0.4)
        forced = np.array([bool(c.get('force_front', False)) for c in cl])
        v = v - 0.5 * forced                     # dobuki sit in the crown's shade (mid / violet tiers)
        # sky holes in the upper masses (clusters inside removed)
        order_m = sorted([j for j in range(nm) if info[j] is not None], key=lambda j: info[j]['cy'])
        holes = []
        medr = float(np.median([c['pr'] for c in cl])) if n else 4 * px
        for j in order_m:
            m = info[j]
            if m['hg'] < 0.3:
                continue
            for _ in range(int(mr.integers(2, 5))):
                holes.append((m['cx'] + mr.uniform(-0.7, 0.7) * m['rx'], m['cy'] + mr.uniform(-0.8, 0.3) * m['ry'],
                              medr * mr.uniform(0.8, 1.7)))
        keep = np.ones(n, bool)
        for (hx, hy, hr_) in holes:
            keep &= ~(np.hypot(X - hx, Y - hy) < hr_ * 0.9)
        keep |= forced
        thin = np.clip((0.97 - D1) / 0.45, 0, 1)
        cx_, cy_ = np.median(X), np.median(Y)
        sw = np.clip(((X - cx_) * Ls[0] + (Y - cy_) * Ls[1]) / max(np.percentile(np.abs(X - cx_), 90), 1.0), 0, 1)
        trans = np.clip((thin + 0.5 * sw) * np.clip(face * 3.0 + 0.4 * sw, 0, 1), 0, 1) * (v < 3.6) * 0.7
        self.dbg = dict(trans=trans, v=v, lab=lab)
        # ---- paint: back florets (per mass, top -> bottom), wood, twigs, front florets, sprays
        C = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W), np.float32)
        stf_all = []
        stb = []
        for j in order_m:
            idx = [i for i in np.nonzero((lab == j) & keep)[0]]
            idx.sort(key=lambda i: v[i])
            for i in idx:
                c = cl[i]
                r2 = np.random.default_rng(c['seed'])
                st = self._cluster_stamps(c, float(v[i]), float(trans[i]), r2)
                (stf_all if c['front'] else stb).extend(st)
        if stb:
            _splat(C, A, np.array(stb, np.float64))
        Ab_ = A.copy()
        wcol, wcov = self._wood(tree, to_px, H, W, fld, rng)
        C = C * (1 - wcov[..., None]) + wcol * wcov[..., None]
        A = A + wcov * (1 - A)
        # ---- sprays: thin twigs carrying 2-6 single florets past the crown silhouette (feathered edge)
        spr = []
        tw = np.zeros((H, W), np.float32)
        cand = [i for i in range(n) if keep[i] and rr[i] > 0.6 and not forced[i]]
        nsp = int(min(len(cand), (70 if not self.far else 18) * self.detail + 6))
        if cand:
            pick = mr.choice(len(cand), size=nsp, replace=False)
            for pi in pick:
                c = cl[cand[pi]]
                m = info[lab[cand[pi]]]
                d = _unit([c['px'] - m['cx'], (c['py'] - m['cy']) * 1.2 - 0.2 * m['ry']])
                d = _rot(d, mr.uniform(-35, 35))
                L = c['pr'] * mr.uniform(1.3, 2.8)
                p0 = np.array([c['px'], c['py']])
                p1 = p0 + d * L
                bend = _rot(d, mr.choice([-1, 1]) * 90) * L * 0.12 + np.array([0.0, 0.08 * L])
                pts = np.array([p0 + (p1 - p0) * u + bend * 4 * u * (1 - u) for u in np.linspace(0, 1, 8)])
                cv2.polylines(tw, [np.round(pts * 4).astype(np.int32)], False, 1.0,
                              max(1, int(round(0.8 * px * 4))), cv2.LINE_AA, shift=2)
                lit_side = ((p1 - np.array([m['cx'], m['cy']])) @ Ls) > 0
                upper = p1[1] < m['cy']
                nb = int(mr.integers(2, 7))
                for b_ in range(nb):
                    u = mr.uniform(0.3, 1.05)
                    bp = p0 + (p1 - p0) * u + bend * 4 * u * (1 - u) + mr.normal(0, 0.1, 2) * c['pr']
                    br = self._floret_r(c['pr']) * mr.uniform(0.75, 1.1)
                    vq = 4.0 if (upper and lit_side) else (3.2 if (upper or lit_side) else 2.2)
                    vq += mr.normal(0, 0.25)
                    fc = _band_col(self.bands, np.array(float(np.clip(vq, 0, 5)))).astype(np.float32)
                    if lit_side:
                        fc = fc + (PEACH - fc) * 0.3
                    cres = (self.bands[5] - fc) * 0.8
                    spr.append((bp[0], bp[1], max(br, 1.0 * px), mr.uniform(0, 6.3), 5.0, 0.5, 0.7 * px,
                                fc[0], fc[1], fc[2], Ls[0], Ls[1], 0.25, cres[0], cres[1], cres[2], 1.0))
        tw = np.clip(tw * 0.85, 0, 1)
        C = C * (1 - tw[..., None]) + self.wp['inner'] * tw[..., None]
        A = A + tw * (1 - A)
        if stf_all:
            _splat(C, A, np.array(stf_all, np.float64))
        if spr:
            _splat(C, A, np.array(spr, np.float64))
        C = C / np.maximum(A, 1e-4)[..., None]
        blos = np.clip(A - wcov - tw, 0, 1)
        if getattr(self, 'BLOS_FULL', False):
            # round 21: wood crossing the blossom is not a silhouette (no rim / peach lines along every branch)
            blos = np.clip(A, 0, 1) * (1 - np.clip(wcov * (1 - Ab_) * 1.0, 0, 1))
        # ---- soft pink veil inside the crown gaps (lacy, but atmospheric, like cm5_01)
        hull = cv2.GaussianBlur(blos, (0, 0), 3.0 * px)
        veil = np.clip((hull - 0.25) / 0.4, 0, 1) * 0.05 * (1 - A)
        # ---- down side: soft violet-magenta haze below the lower clusters (lost edge)
        yyr = np.arange(H, dtype=np.float32)[:, None]
        low = np.clip((yyr - (ytop + 0.5 * (ybot - ytop))) / max(0.45 * (ybot - ytop), 1.0), 0, 1)
        sd = 4.0 * px
        Mdn = np.float32([[1, 0, -Ls[0] * sd * 0.6], [0, 1, sd * 1.4]])
        hz = cv2.warpAffine(cv2.GaussianBlur(blos, (0, 0), 4.5 * px), Mdn, (W, H), borderMode=cv2.BORDER_CONSTANT)
        hz = np.clip(hz * 1.2, 0, 1) * low * 0.3 * (1 - A)
        # ---- sun-facing silhouette: near-white cream rim + peach transmitted band
        sh = 2.6 * px
        M = np.float32([[1, 0, -Ls[0] * sh], [0, 1, -Ls[1] * sh]])
        As = cv2.warpAffine(blos, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        rim = np.clip(blos - As, 0, 1)
        # rim only on the crown's outer sun-facing silhouette (not on every floret inside the mass)
        outer = cv2.warpAffine(cv2.GaussianBlur(blos, (0, 0), 5.0 * px), np.float32([[1, 0, -Ls[0] * 9 * px],
                                                                                    [0, 1, -Ls[1] * 9 * px]]),
                               (W, H), borderMode=cv2.BORDER_CONSTANT)
        rim = rim * (0.35 + 0.65 * np.clip((0.6 - outer) / 0.4, 0, 1))
        rim = np.clip(rim * 1.4, 0, 1) * np.clip(0.55 + 0.8 * (1 - yyr / max(H, 1)), 0.55, 1.2)
        C += (np.array([1.18, 1.07, 0.94], np.float32) - C) * np.clip(rim * 0.9, 0, 0.95)[..., None]
        band = np.clip(blos - cv2.warpAffine(cv2.GaussianBlur(blos, (0, 0), 2.5 * px),
                                             np.float32([[1, 0, -Ls[0] * 6 * px], [0, 1, -Ls[1] * 6 * px]]),
                                             (W, H), borderMode=cv2.BORDER_CONSTANT), 0, 1)
        band = cv2.GaussianBlur(band, (0, 0), 2.0 * px) * blos
        lumC = C.mean(-1)
        C += (PEACH - C) * (band * 0.35 * np.clip((1.02 - lumC) / 0.25, 0.2, 1))[..., None]
        # lower florets a touch softer (lost edge) on the down side
        A = A * np.clip((oy + 0.02 * k - yyr) / (1.5 * px) + 0.5, 0, 1)
        pm = np.dstack([C * A[..., None], A])
        pmb = cv2.GaussianBlur(pm, (0, 0), 1.3 * px)
        lk = (low * 0.45 * (1 - wcov))[..., None]
        pm = pm * (1 - lk) + pmb * lk
        # add veil + under-haze behind (premultiplied 'under' compositing)
        for colr, al in ((VEIL if not self.far else VEIL * 0.9 + 0.1, veil), (HAZE_V, hz)):
            pa = pm[..., 3]
            add = al * (1 - pa)
            pm[..., :3] += np.asarray(colr, np.float32) * add[..., None]
            pm[..., 3] = pa + add
        # halation: warm glow bleeding into the sky around the sun-facing crown
        Ab = cv2.GaussianBlur(blos, (0, 0), 10.0 * px)
        sh4 = 7.0 * px
        M = np.float32([[1, 0, Ls[0] * sh4], [0, 1, Ls[1] * sh4]])
        Ab = cv2.warpAffine(Ab, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        g = np.clip(Ab - cv2.GaussianBlur(blos, (0, 0), 3 * px), 0, 1) * (0.42 if not self.far else 0.18)
        pa = pm[..., 3]
        add_a = g * (1 - pa)
        pm[..., :3] += HALO * add_a[..., None]
        pm[..., 3] = pa + add_a
        self.stats = dict(n=n, masses=nm, holes=len(holes), sprigs=len(spr), dobuki=int(forced.sum()))
        return pm.astype(np.float32), ox, oy
