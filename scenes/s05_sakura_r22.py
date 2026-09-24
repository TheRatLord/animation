"""s05_sakura round 22: Painter22 (reviewer FAIL on round 21: 'whole canopy one saturated lilac-magenta, clay puffs;
white confetti tips everywhere; stitched rivet dots on the bark; no sky breakup').

  * two-value clusters: exposed (upper / sun-side) clusters carry big crisp pale near-white pink caps (#F6D9E4
    range); covered / lower clusters drop to ONE cooler, darker magenta-violet value with soft lost edges;
  * a desaturated, higher-key palette (pale pinks, not lilac) with violet reserved for the shadows;
  * lighter petal tips only on the sun-facing crown silhouette, grouped into 3-5 petal sprays - none inside;
  * more / bigger sky holes through the canopy (cm5_01);
  * bark: horizontal lenticel dashes + a few irregular cracks in one flat darker value, flat cool violet-brown
    shadow side (no gradient, no dot rows).
"""
import math

import numpy as np
import cv2

from s05_sakura_r16 import _splat, _band_col, TRANS
from s05_sakura_r15 import _raster_wood, _unit, _rot, _sstep
from s05_sakura_r17 import PEACH, HALO, _kmeans
from s05_sakura_r19 import _tier
from s05_sakura_r20 import VEIL, HAZE_V
from s05_sakura_r21 import Tree21, Painter21


# ---- copied from Painter20 (modified)
class _P(Painter21):
    def _wood_base(self, tree, to_px, H, W, fld, rng):
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
                sh = np.array([0.22, 0.16, 0.25], np.float32)      # flat cool violet-brown shadow side
                mid = np.array([0.5, 0.38, 0.34], np.float32)
                lit = np.array([0.78, 0.6, 0.46], np.float32)
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
            # round 22: long vertical value streaks only on the lit side; the shadow side stays ONE flat value
            bn = cv2.remap(bt, ((arc / 0.009 + RID * 7.0) % tw_).astype(np.float32),
                           ((Vv / 0.9 + RID * 29.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            bn2 = cv2.remap(bt, ((arc / 0.0045 + RID * 11.0) % tw_).astype(np.float32),
                            ((Vv / 0.35 + RID * 5.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            vb = 0.65 * bn + 0.35 * bn2
            lo = _sstep((-vb - 0.02) / 0.04 + 0.5)
            fade = np.clip(1.2 - a_s, 0, 1) * thick * m1
            col = col * (1 - (0.22 * lo * fade)[..., None])
            # horizontal lenticel dashes (sakura bark): thin bands ACROSS the trunk, varied length, staggered
            # rows, one flat darker value (no dot columns)
            sp = 0.11
            q = Vv / sp + 0.35 * bn
            jj = np.floor(q)
            fq = q - jj
            circ = 2.0 * np.pi * rm
            ca = (arc + np.sin(jj * 5.31 + RID) * 3.0) / np.maximum(0.55 * circ, 0.05)
            ci = np.floor(ca)
            fa = ca - ci
            h2 = np.sin(jj * 12.9898 + ci * 4.1414 + RID * 78.233) * 43758.5453
            h2 = h2 - np.floor(h2)
            h3 = np.sin(jj * 7.13 + ci * 19.7 + RID * 3.3) * 2451.77
            h3 = h3 - np.floor(h3)
            wq = (0.045 + 0.05 * h3)
            lx0, lx1 = 0.05 + 0.25 * h3, 0.45 + 0.5 * (1 - h3)
            dash = _sstep((wq - np.abs(fq - 0.5)) / 0.025 + 0.5) * (h2 < 0.6) *                 _sstep((fa - lx0) / 0.03 + 0.5) * _sstep((lx1 - fa) / 0.03 + 0.5)
            dash *= _sstep((0.97 - a_s) / 0.05 + 0.5)
            # a few long irregular bark cracks (wandering vertical lines, low frequency)
            fz = cv2.remap(bt, ((arc / 0.004 + RID * 23.0) % tw_).astype(np.float32),
                           ((Vv / 1.6 + RID * 41.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            crk = _sstep((0.022 - np.abs(fz)) / 0.012 + 0.5) * _sstep((bn - 0.05) / 0.08 + 0.5)
            crk *= _sstep((0.9 - a_s) / 0.1 + 0.5)
            mark = np.clip(np.maximum(dash, crk) * thick, 0, 1)
            dk_lit = np.array([0.3, 0.2, 0.22], np.float32)
            dk_sh = sh * 0.72
            dk = dk_sh + (dk_lit - dk_sh) * m1[..., None]
            col += (dk - col) * (mark * 0.9)[..., None]
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
            if m['hg'] < 0.12:
                continue
            for _ in range(int(mr.integers(*self.HOLES))):
                holes.append((m['cx'] + mr.uniform(-0.75, 0.75) * m['rx'], m['cy'] + mr.uniform(-0.8, 0.5) * m['ry'],
                              medr * mr.uniform(*self.HOLE_R)))
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
                    vq = 4.0 if (upper and lit_side) else (2.8 if (upper or lit_side) else SHADE_T)
                    vq += mr.normal(0, 0.1)
                    fc = _band_col(self.bands, np.array(float(np.clip(vq, 0, 5)))).astype(np.float32)
                    if lit_side:
                        fc = fc + (PEACH - fc) * 0.15
                    cres = (self.bands[4] - fc) * (0.5 if (upper and lit_side) else 0.0)
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
        rim = rim * np.clip((0.55 - outer) / 0.4, 0, 1)
        rim = np.clip(rim * 1.4, 0, 1) * np.clip(0.55 + 0.8 * (1 - yyr / max(H, 1)), 0.55, 1.2)
        C += (np.array([1.14, 1.04, 0.95], np.float32) - C) * np.clip(rim * 0.8, 0, 0.85)[..., None]
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


# (0 deep violet, 1 cool magenta-violet shade, 2 rose, 3 light pink, 4 pale near-white pink, 5 near white)
BANDS22 = np.array([
    (0.32, 0.2, 0.46),
    (0.5, 0.3, 0.57),
    (0.84, 0.52, 0.7),
    (0.97, 0.66, 0.79),
    (1.0, 0.77, 0.86),
    (1.03, 0.86, 0.91),
], np.float32)
BANDS22_FAR = (BANDS22 * 0.76 + np.array([0.9, 0.87, 0.95], np.float32) * 0.24).astype(np.float32)
SHADE_T = 1.05          # the one underside value (band index)


class Tree22(Tree21):
    pass


class Painter22(_P):
    FRONT = 0.4
    BLOS_FULL = True
    WOOD_HARD = True
    HOLES = (4, 8)
    HOLE_R = (0.9, 2.2)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bands = BANDS22_FAR if self.far else BANDS22

    def _wood(self, tree, to_px, H, W, fld, rng):
        col, cov = self._wood_base(tree, to_px, H, W, fld, rng)
        inside = np.clip(cv2.resize(fld['D2'], (W, H), interpolation=cv2.INTER_LINEAR) * 2.0, 0, 1)
        dark = np.array([0.26, 0.17, 0.25], np.float32) if not self.far else np.array([0.44, 0.36, 0.48], np.float32)
        lum = col.mean(-1)
        k = inside * np.clip((lum - 0.3) / 0.3, 0, 1) * 0.85
        col = col + (dark - col) * k[..., None]
        return col, cov

    def _cluster_stamps(self, c, v, trans, rng):
        """two-value clump: one soft-edged cool shade value + crisp pale caps on exposed clumps only"""
        px = self.px
        Ls = self.Ls
        bands = self.bands
        R = c['pr'] * 1.12
        tx, ty = c['tx'], -c['ty']
        mv = c.get('mv', None)
        ex = c.get('ex', 0.0)
        lx, ly = 0.55 * Ls[0], 0.55 * Ls[1] - 0.8
        ln = math.hypot(lx, ly) + 1e-9
        lx, ly = lx / ln, ly / ln
        st = []
        nh = int(np.clip(round(R / (3.2 * px)), 2, 6)) + int(rng.integers(0, 2))
        heads = []
        for _ in range(nh):
            u = rng.uniform(-1, 1)
            w_ = rng.normal(0, 0.35)
            hx = c['px'] + (u * tx - w_ * ty) * R * 0.55
            hy = c['py'] + (u * ty + w_ * tx) * R * 0.55 + 0.1 * R
            hr = R * rng.uniform(0.42, 0.62)
            heads.append((hx, hy, hr))
        heads.sort(key=lambda q: -q[1])
        vbase = v
        if mv is not None:
            vbase = v - 0.9 * (c['py'] - mv[1]) / max(mv[2], 1.0)
        # coherent value groups per MASS: position along the (up-biased) light direction inside the mass decides
        # lit top group / mid band / one cool underside value, nudged by the cast-shadow value field
        if mv is not None:
            tl = ((c['px'] - mv[0]) * lx + (c['py'] - mv[1]) * ly) / max(mv[2], 1.0)
        else:
            tl = 0.0
        score = 1.2 * tl + 0.22 * (v - 2.3) + 0.5 * ex
        tier = 2 if score > 0.12 else (1 if score > -0.6 else 0)
        # dark separation head behind (down side)
        bx = c['px'] - lx * R * 0.35
        by = c['py'] - ly * R * 0.35
        if tier < 2:
            dc = _band_col(bands, np.array(0.6)).astype(np.float32)
            st.append((bx, by, R * 0.55, rng.uniform(0, 6.3), 7.0, 0.22, 2.5 * px, dc[0], dc[1], dc[2],
                       0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
        # each value group is ONE flat plane: heads of the same tier fuse (same colour), so the terminator lives at
        # the group boundary (scalloped by the head outlines), not as a crescent on every puff
        if tier == 2:
            lt = 3.8 + 0.35 * ex + rng.normal(0, 0.05)
        elif tier == 1:
            lt = 2.75 + rng.normal(0, 0.05)
        else:
            lt = SHADE_T + rng.normal(0, 0.04)
        gc = _band_col(bands, np.array(float(lt))).astype(np.float32)
        if trans > 0 and tier > 0:
            gc = gc + (TRANS * 1.04 - gc) * trans * 0.3
        aa = (0.7 if tier == 2 else (1.1 if tier == 1 else 2.2)) * px
        for (hx, hy, hr) in heads:
            cc = gc * rng.uniform(0.99, 1.01)
            st.append((hx, hy, hr, rng.uniform(0, 6.3), float(rng.integers(6, 10)), 0.2, aa,
                       cc[0], cc[1], cc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
        # lighter petal tips: ONLY exposed, sun-facing clumps, as one grouped spray of 3-5 petals on the lit edge
        if tier == 2 and ex > 0.3:
            fr0 = float(np.clip(0.22 * R, 2.2 * px, 5.5 * px))
            a0 = math.atan2(ly, lx) + rng.uniform(-0.7, 0.7)
            ng = int(rng.integers(3, 6))
            for _ in range(ng):
                a = a0 + rng.normal(0, 0.28)
                rr = R * rng.uniform(0.85, 1.1)
                fx, fy = c['px'] + math.cos(a) * rr, c['py'] + math.sin(a) * rr + 0.1 * R
                fc = _band_col(bands, np.array(float(4.2 + rng.normal(0, 0.1)))).astype(np.float32)
                cr_ = (bands[5] - fc) * 0.5
                st.append((fx, fy, fr0 * rng.uniform(0.75, 1.15), rng.uniform(0, 6.3), 5.0, 0.5, 0.7 * px,
                           fc[0], fc[1], fc[2], Ls[0], Ls[1], 0.3, cr_[0], cr_[1], cr_[2], 1.0))
        return st
