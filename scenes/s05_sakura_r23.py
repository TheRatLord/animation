"""s05_sakura round 23: Painter23 (reviewer FAIL on round 22: 'two-tone popcorn - pale cotton balls over a saturated
lavender underside, hard uniform clump outlines, no flower-level breakup, floating clumps, vector-ribbon trunks').

  * every cluster is built from 5-petal FLOWER dabs 3-8 px across (cm5_01), not smooth heads: sky and twigs show
    through the gaps inside the masses; only shaded clusters get a small solid core so the mass still reads;
  * per-flower value from its position inside the mass along the (up-biased) light: pale pink lit top group, a
    thin mid-pink transition band, a cooler rose-magenta underside (hue ~330, lower saturation, higher value than
    the old lavender) with softer (lost) flower edges; flowers of a mass are drawn shade -> lit;
  * lighter 1-2 px petal-tip crescents only on sun-facing flowers of exposed clusters;
  * under-haze / veil moved from violet to rose;
  * Tree23: limbs taper properly (Leonardo taper to a fine twig), so no constant-width ribbons;
  * bark: hard warm lit edge on the sun side, ONE flat cool grey-violet shadow side (~40% of the width),
    horizontal lenticel dashes, cracks and a few dark knots.
"""
import math

import numpy as np
import cv2

from s05_sakura_r16 import _splat, _band_col, TRANS
from s05_sakura_r15 import _raster_wood, _unit, _rot, _sstep
from s05_sakura_r17 import PEACH, HALO, _kmeans
from s05_sakura_r19 import _tier
VEIL = np.array([0.99, 0.8, 0.86], np.float32)
HAZE_V = np.array([0.8, 0.56, 0.68], np.float32)     # rose-magenta under-haze (was violet)
from s05_sakura_r21 import Tree21, Painter21
from s05_sakura_r16 import _unit as _u16
import s05_sakura_r16 as _R16


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
                for _ in range(2 if w['d'] == 0 else getattr(self, 'SMOOTH', 2)):
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
            # round 23: across-coordinate toward the sun (independent of how steeply the limb faces it) -> the
            # shadow side is a constant ~40% of the width, the lit band ~22%
            _up = getattr(self, 'WOOD_UP', 0.0)
            nl_ = (NX * Ls[0] + NY * (Ls[1] - _up)) / math.hypot(Ls[0], Ls[1] - _up)
            fn = S * np.sign(nl_) * np.clip(np.abs(nl_) * 4.0, 0.0, 1.0)
            fct = fn + 0.06 * wob
            if not self.far:
                sh = np.array([0.27, 0.25, 0.36], np.float32)      # flat cool grey-violet shadow side
                mid = np.array([0.4, 0.35, 0.37], np.float32)      # grey-brown bark body
                lit = np.array([0.78, 0.62, 0.48], np.float32)     # hard warm lit edge
        m1 = _sstep((fct + (0.2 if hard else 0.4)) / (getattr(self, 'TERM_W', 0.03) if hard else 0.2) + 0.5)
        col += (mid - sh) * m1[..., None]
        big = np.clip((RR - 2.0 * px) / (3 * px), 0, 1)
        m2 = _sstep((fct - (getattr(self, 'LIT_T', 0.55) if hard else 0.2)) / (0.025 if hard else 0.14) + 0.5) * big
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
            sp = 0.075
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
            dash = _sstep((wq - np.abs(fq - 0.5)) / 0.025 + 0.5) * (h2 < 0.75 * getattr(self, 'LENT', 1.0)) *                 _sstep((fa - lx0) / 0.03 + 0.5) * _sstep((lx1 - fa) / 0.03 + 0.5)
            dash *= _sstep((0.97 - a_s) / 0.05 + 0.5)
            # a few long irregular bark cracks (wandering vertical lines, low frequency)
            fz = cv2.remap(bt, ((arc / 0.004 + RID * 23.0) % tw_).astype(np.float32),
                           ((Vv / 1.6 + RID * 41.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            crk = _sstep((0.022 - np.abs(fz)) / 0.012 + 0.5) * _sstep((bn - 0.05) / 0.08 + 0.5)
            crk *= _sstep((0.9 - a_s) / 0.1 + 0.5)
            mark = np.clip(np.maximum(dash, crk) * thick, 0, 1)
            # round 23: a few dark bark knots / branch scars (ellipses, dark ring + a lighter lip on the lit side)
            ksp = 0.55
            kq = Vv / ksp
            kj = np.floor(kq)
            hk = np.sin(kj * 91.7 + RID * 13.1) * 9173.31
            hk = hk - np.floor(hk)
            hk2 = np.sin(kj * 27.3 + RID * 5.9) * 3571.7
            hk2 = hk2 - np.floor(hk2)
            kcx = (hk2 - 0.5) * 1.1                         # across position (S units)
            kdv = (kq - kj - 0.5) * ksp / 0.05
            kds = (S - kcx) / 0.16
            kd = np.sqrt(kdv * kdv + kds * kds)
            knot = _sstep((1.0 - kd) / 0.15 + 0.5) * (hk < 0.45 * getattr(self, 'KNOT', 1.0)) * _sstep((0.8 - a_s) / 0.1 + 0.5)
            mark = np.clip(np.maximum(mark, knot * thick), 0, 1)
            dk_lit = np.array([0.22, 0.16, 0.19], np.float32)
            dk_sh = sh * 0.72
            dk = dk_sh + (dk_lit - dk_sh) * m1[..., None]
            col += (dk - col) * (mark * getattr(self, 'MARK_A', 0.9))[..., None]
            # round 23: painted bark patches - pale grey-green lichen blotches and darker bark scabs on the
            # lit / mid side (the shadow side stays one flat value)
            pa = cv2.remap(bt, ((arc / 0.05 + RID * 13.0) % tw_).astype(np.float32),
                           ((Vv / 0.12 + RID * 31.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            pa = (pa - float(bt.mean())) / (float(bt.std()) + 1e-6)
            lich = _sstep((pa - 0.9) / 0.25 + 0.5) * m1 * thick
            scab = _sstep((-pa - 1.1) / 0.25 + 0.5) * m1 * thick
            lc_ = np.array([0.56, 0.55, 0.5], np.float32) * (0.75 + 0.45 * m2)[..., None]
            col += (lc_ - col) * (0.45 * lich)[..., None]
            col += (col * 0.7 - col) * (0.5 * scab)[..., None]
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
        stb_m, stf_m = [], []
        for j in order_m:
            idx = [i for i in np.nonzero((lab == j) & keep)[0]]
            idx.sort(key=lambda i: v[i])
            for i in idx:
                c = cl[i]
                r2 = np.random.default_rng(c['seed'])
                st = self._cluster_stamps(c, float(v[i]), float(trans[i]), r2)
                (stf_m if c['front'] else stb_m).extend(st)
            # round 23: inside one mass draw shade flowers first, lit flowers on top (sort key = column 17)
            stb_m.sort(key=lambda q: q[17] if len(q) > 17 else 0.0)
            stf_m.sort(key=lambda q: q[17] if len(q) > 17 else 0.0)
            stb.extend(stb_m)
            stf_all.extend(stf_m)
            stb_m, stf_m = [], []
        if stb:
            _splat(C, A, np.array([q[:17] for q in stb], np.float64))
        Ab_ = A.copy()
        wcol, wcov = self._wood(tree, to_px, H, W, fld, rng)
        C = C * (1 - wcov[..., None]) + wcol * wcov[..., None]
        A = A + wcov * (1 - A)
        # ---- sprays: thin twigs carrying 2-6 single florets past the crown silhouette (feathered edge)
        spr = []
        tw = np.zeros((H, W), np.float32)
        cand = [i for i in range(n) if keep[i] and rr[i] > 0.6 and not forced[i]]
        nsp = int(min(len(cand), (40 if not self.far else 12) * self.detail + 4))
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
                    br = min(self._floret_r(c['pr']), 3.4 * px) * mr.uniform(0.75, 1.1)
                    vq = 3.5 if (upper and lit_side) else (2.5 if (upper or lit_side) else SHADE_T)
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
        A_pre = A.copy()
        if stf_all:
            _splat(C, A, np.array([q[:17] for q in stf_all], np.float64))
        Af_ = np.clip((A - A_pre) / np.maximum(1 - A_pre, 1e-3), 0, 1) if stf_all else np.zeros_like(A)
        if spr:
            _splat(C, A, np.array(spr, np.float64))
        C = C / np.maximum(A, 1e-4)[..., None]
        blos = np.clip(A - wcov - tw, 0, 1)
        if getattr(self, 'BLOS_FULL', False):
            # round 21: wood crossing the blossom is not a silhouette (no rim / peach lines along every branch)
            blos = np.clip(A, 0, 1) * (1 - np.clip(wcov * (1 - Ab_) * 1.0, 0, 1))
            if getattr(self, 'BLOS_NOWOOD', False):
                # round 26: wood lying over back blossom is not blossom either (no cream rim / peach band on the
                # limb -> no pale 'joint' segments where a limb crosses a clump)
                blos = blos * (1 - np.clip(wcov * (1 - Af_), 0, 1))
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
        sh = 1.6 * px
        M = np.float32([[1, 0, -Ls[0] * sh], [0, 1, -Ls[1] * sh]])
        As = cv2.warpAffine(blos, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        rim = np.clip(blos - As, 0, 1)
        # rim only on the crown's outer sun-facing silhouette (not on every floret inside the mass)
        outer = cv2.warpAffine(cv2.GaussianBlur(blos, (0, 0), 5.0 * px), np.float32([[1, 0, -Ls[0] * 9 * px],
                                                                                    [0, 1, -Ls[1] * 9 * px]]),
                               (W, H), borderMode=cv2.BORDER_CONSTANT)
        rim = rim * np.clip((0.55 - outer) / 0.4, 0, 1)
        rim = np.clip(rim * 1.4, 0, 1) * np.clip(0.55 + 0.8 * (1 - yyr / max(H, 1)), 0.55, 1.2)
        C += (np.array([1.14, 1.04, 0.95], np.float32) - C) * np.clip(rim * 0.5, 0, 0.55)[..., None]
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
        lk = (low * 0.2 * (1 - wcov))[..., None]
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


# (0 deep rose-magenta, 1 cool rose-magenta shade (hue ~330), 2 mid rose-pink transition, 3 light pink,
#  4 pale lit pink, 5 petal-tip near white)
BANDS23 = np.array([
    (0.68, 0.33, 0.53),
    (0.84, 0.5, 0.68),
    (0.93, 0.63, 0.77),
    (0.97, 0.73, 0.83),
    (0.99, 0.81, 0.88),
    (1.03, 0.93, 0.94),
], np.float32)
BANDS23_FAR = (BANDS23 * 0.78 + np.array([0.9, 0.87, 0.95], np.float32) * 0.22).astype(np.float32)
SHADE_T = 1.1           # the one underside value (band index)


class Tree23(Tree21):
    """Tree21 with a real Leonardo taper: every limb thins to ~30% of its base radius (twigs to the minimum),
    children leave from the tapered radius, so no limb reads as a constant-width ribbon."""
    TAPER = 0.3

    def _gate(self, rng):
        """fewer / smaller patch holes than Tree21: the flower-level gaps already let the sky through"""
        cl = [c for c in self.clusters if c['y'] > 0.75 * self.ht]
        if not cl:
            self.clusters = cl
            return
        X = np.array([c['x'] for c in cl])
        Y = np.array([c['y'] for c in cl])
        cx, cy = np.median(X), np.median(Y)
        sx_, sy_ = X.std() + 1e-6, Y.std() + 1e-6
        nb = 11
        bx = rng.uniform(X.min(), X.max(), nb)
        by = rng.uniform(Y.min(), Y.max(), nb)
        br = rng.uniform(0.2, 0.45, nb) * self.s
        keep = []
        for c in cl:
            dn = math.hypot((c['x'] - cx) / sx_, (c['y'] - cy) / sy_)
            p = 0.95 - 0.25 * max(0.0, dn - 0.95)
            hole = np.exp(-((c['x'] - bx) ** 2 + (c['y'] - by) ** 2) / (br * br)).max()
            p *= 1.0 - 0.85 * hole
            if rng.random() < p:
                keep.append(c)
        self.clusters = keep

    def _limb(self, rng, p0, d, L, r0, depth, m):
        s = self.s
        nn = max(6, int(L / (0.04 * s)))
        dd = _unit(d)
        step = L / (nn - 1)
        g = (0.0 if depth == 1 else 0.006 * depth) * rng.uniform(0.6, 1.4)
        ymin = {1: 0.12, 2: -0.2}.get(depth, -0.4)
        curl = rng.normal(0, 1.0)
        nk = int(rng.integers(1, 4))
        kinks = set(rng.choice(np.arange(2, nn - 1), size=min(nn - 3, nk), replace=False).tolist())
        P = [np.asarray(p0, np.float64)]
        for i in range(1, nn):
            a = rng.normal(0, 2.4) + curl
            if i in kinks:
                a += rng.choice([-1, 1]) * rng.uniform(6, 16) * getattr(self, 'KINK', 1.0)
            dd = _rot(dd, a)
            u = i / (nn - 1)
            dd = _unit(dd + np.array([0.0, -g * u * 2.0]))
            if dd[1] < ymin:
                dd = _unit([dd[0] * math.sqrt(max(1 - ymin * ymin, 1e-3)) / max(abs(dd[0]), 1e-3), ymin])
            P.append(P[-1] + dd * step)
        P = np.array(P)
        tt = np.linspace(0, 1, nn)
        last = depth >= self.maxd
        r1 = self.rmin if last else max(self.rmin, r0 * self.TAPER)
        r = r0 + (r1 - r0) * tt ** 0.8
        self.wood.append(dict(P=P, r=r, d=depth, m=m, rid=len(self.wood)))
        if not last:
            nl = {1: int(rng.integers(3, 6)), 2: int(rng.integers(3, 5)), 3: int(rng.integers(2, 4))}.get(depth, 2)
            ts = np.sort(rng.uniform(0.15, 0.88, nl))
            side = rng.choice([-1, 1])
            for tl in ts:
                i = int(tl * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                cd = _rot(td, side * rng.uniform(25, 60))
                side = -side
                cd = _unit(cd + np.array([0.0, 0.25]))
                Lc = L * rng.uniform(0.4, 0.62) * (1.0 - 0.3 * tl)
                if Lc < 0.08 * s:
                    continue
                self._limb(rng, P[i], cd, Lc, max(self.rmin, r[i] * rng.uniform(0.5, 0.7)), depth + 1, m)
            td = _unit(P[-1] - P[-3])
            share = rng.uniform(0.35, 0.65)
            for sgn, sh in ((-1, share), (1, 1 - share)):
                cd = _rot(td, sgn * rng.uniform(14, 36))
                self._limb(rng, P[-1], cd, L * rng.uniform(0.42, 0.6), max(self.rmin, r[-1] * math.sqrt(sh) * 1.05),
                           depth + 1, m)
        if depth >= 2 or (depth == 1 and not self.big):
            Rb = 0.1 * s * self.cl
            t = (0.55 if depth == 1 else 0.25 if depth == 2 else 0.05) + rng.uniform(0, 0.12)
            while t <= 1.0:
                i = int(t * (nn - 1))
                td = _unit(P[min(i + 1, nn - 1)] - P[max(i - 1, 0)])
                nrm = np.array([-td[1], td[0]])
                R = Rb * rng.uniform(0.65, 1.35) * (1.2 if depth <= 2 else 1.0)
                off = rng.normal(0, 0.3) * R                    # round 23: blossom sits ON the twig
                c = P[i] + nrm * off + np.array([0.0, -0.15 * R])
                self.clusters.append(dict(x=float(c[0]), y=float(c[1]), R=float(R), m=m, d=depth,
                                          tx=float(td[0]), ty=float(td[1]),
                                          front=bool(rng.random() < 0.5), seed=int(rng.integers(1 << 30))))
                t += (R / L) * rng.uniform(1.1, 2.0) / max(self.density, 0.4)
            if last:
                tip = P[-1]
                R = Rb * rng.uniform(0.6, 1.0)
                self.clusters.append(dict(x=float(tip[0]), y=float(tip[1] - 0.1 * R), R=float(R), m=m, d=depth,
                                          tx=float(dd[0]), ty=float(dd[1]), front=True,
                                          seed=int(rng.integers(1 << 30))))


class Painter23(_P):
    FRONT = 0.4
    BLOS_FULL = True
    WOOD_HARD = True
    HOLES = (2, 5)
    HOLE_R = (0.9, 2.2)
    FILL = 0.95

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bands = BANDS23_FAR if self.far else BANDS23

    def _wood(self, tree, to_px, H, W, fld, rng):
        col, cov = self._wood_base(tree, to_px, H, W, fld, rng)
        inside = np.clip(cv2.resize(fld['D2'], (W, H), interpolation=cv2.INTER_LINEAR) * 2.0, 0, 1)
        dark = np.array([0.27, 0.2, 0.26], np.float32) if not self.far else np.array([0.46, 0.4, 0.5], np.float32)
        lum = col.mean(-1)
        k = inside * np.clip((lum - 0.3) / 0.3, 0, 1) * 0.85
        col = col + (dark - col) * k[..., None]
        return col, cov

    def _cluster_stamps(self, c, v, trans, rng):
        """a cluster = 4-70 five-petal flower dabs (3-8 px) with gaps; value per flower from its place in the mass"""
        px = self.px
        Ls = self.Ls
        bands = self.bands
        R = c['pr'] * 1.15
        mv = c.get('mv', None)
        ex = c.get('ex', 0.0)
        lx, ly = 0.55 * Ls[0], 0.55 * Ls[1] - 0.8
        ln = math.hypot(lx, ly) + 1e-9
        lx, ly = lx / ln, ly / ln
        if mv is not None:
            tl = ((c['px'] - mv[0]) * lx + (c['py'] - mv[1]) * ly) / max(mv[2], 1.0)
            msc = max(mv[2], 1.0)
        else:
            tl, msc = 0.0, R * 3
        score = 1.2 * tl + 0.22 * (v - 2.3) + 0.5 * ex
        fmin, fmax = (1.5 * px, 3.9 * px) if not self.far else (1.1 * px, 2.6 * px)
        fr0 = float(np.clip(0.22 * R, fmin, fmax))
        st = []
        # a small solid core on shaded clusters only (the mass still reads, gaps stay at flower scale)
        if score < 0.1 and R > 3 * fr0:
            dc = _band_col(bands, np.array(SHADE_T - 0.25)).astype(np.float32)
            for _ in range(int(rng.integers(1, 3))):
                st.append((c['px'] + rng.normal(0, 0.15) * R, c['py'] + rng.normal(0, 0.15) * R + 0.1 * R,
                           R * rng.uniform(0.3, 0.42), rng.uniform(0, 6.3), 7.0, 0.3, 1.6 * px, dc[0], dc[1],
                           dc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0, -9.0))
        n = int(np.clip(round((R / fr0) ** 2 * self.FILL * rng.uniform(0.8, 1.1)), 4, 70))
        placed = []
        tries = 0
        # flowers bunch in 1-3 umbels along the twig: denser centres, ragged outline, gaps between the umbels
        tx, ty = c['tx'], -c['ty']
        nu = int(rng.integers(1, 4))
        umb = []
        for _ in range(nu):
            u = rng.uniform(-0.6, 0.6)
            w_ = rng.normal(0, 0.25)
            umb.append((c['px'] + (u * tx - w_ * ty) * R * 0.5, c['py'] + (u * ty + w_ * tx) * R * 0.5 + 0.08 * R,
                        R * rng.uniform(0.55, 0.8)))
        while len(placed) < n and tries < 5 * n:
            tries += 1
            ux, uy, ur = umb[int(rng.integers(nu))]
            a = rng.uniform(0, 2 * math.pi)
            rr = ur * rng.uniform(0.0, 1.0) ** 0.6
            fx, fy = ux + math.cos(a) * rr, uy + math.sin(a) * rr * 0.9
            fr = fr0 * math.exp(rng.normal(0, 0.18))
            ok = True
            for (qx, qy, qr) in placed:
                if (fx - qx) ** 2 + (fy - qy) ** 2 < (0.78 * (fr + qr)) ** 2:
                    ok = False
                    break
            if ok:
                placed.append((fx, fy, fr))
        lit_hi = 3.55 + 0.3 * ex
        _H = getattr(Painter23, 'HIST', None)
        for (fx, fy, fr) in placed:
            off = ((fx - c['px']) * lx + (fy - c['py']) * ly)
            sf = score + 0.6 * off / msc + 0.25 * off / max(R, 1e-3) + rng.normal(0, 0.07)
            if sf > 0.95:
                lt = lit_hi + rng.normal(0, 0.06)
                aa = 0.7 * px
            elif sf > 0.55:
                lt = 2.55 + rng.normal(0, 0.1)                 # thin mid-pink transition band
                aa = 0.85 * px
            elif sf > -1.0:
                lt = SHADE_T + rng.normal(0, 0.06)
                aa = 1.3 * px                                  # softer (lost) flower edges in the shade
            else:
                lt = SHADE_T - 0.45 + rng.normal(0, 0.05)
                aa = 1.5 * px
            if _H is not None:
                _H.append(sf)
            fc = _band_col(bands, np.array(float(lt))).astype(np.float32)
            if trans > 0 and lt > 2.0:
                fc = fc + (TRANS * 1.04 - fc) * trans * 0.25
            cres = np.zeros(3, np.float32)
            th = 9.0
            out = off / max(R, 1e-3)
            if lt > 3.0 and ex > 0.25 and out > 0.35:
                # lighter petal tip: a 1-2 px crescent on the sun-facing side of the flower only
                cres = (bands[5] - fc) * 0.4
                th = float(np.clip(1.0 - 1.8 * px / max(fr, 1e-3), 0.15, 0.8))
            st.append((fx, fy, fr, rng.uniform(0, 6.3), 5.0, rng.uniform(0.42, 0.55), aa,
                       fc[0], fc[1], fc[2], lx, ly, th, cres[0], cres[1], cres[2], 1.0, lt))
            if fr > 2.4 * px and lt > 2.0 and rng.random() < 0.35:
                ec = _band_col(bands, np.array(max(lt - 1.7, 0.2))).astype(np.float32)
                st.append((fx, fy, fr * 0.2, 0.0, 5.0, 0.2, 0.6 * px, ec[0], ec[1], ec[2], 0.0, 0.0, 9.0,
                           0.0, 0.0, 0.0, 0.6, lt + 0.001))
        return st
