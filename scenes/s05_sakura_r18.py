"""s05_sakura round 18: blossom crowns built from clustered five-petal dabs (cm5_01 study).

Final-panel notes: 'pink cotton-candy / clay blobs on bare stick trunks (lollipops)', 'low-frequency pastel
blobs with dithered noise edges', 'build them as clustered blossom dabs: a few hundred small 5-petal clumps per
crown, lit tips pale near-white-pink, violet-magenta shadow core with hard-edged inner clump separations, sky
holes, branches that go into the crown and taper, dark branches crossing in front'.

Painter18 (on top of Painter17's masses / wood / rim / lost edges)
  * every flower head is no longer one scalloped disc: it is a clump of 4-30 small AA five-petal blossoms
    (deep-notched petals) scattered over the head with gaps, drawn back -> front along the sun direction so
    the sun-side blossoms overlap (lit tips pale pink-white, the far side of the clump a band darker);
  * under each clump a smaller, darker rose / violet core disc -> the gaps between blossoms read as a hard-
    edged shadow separation instead of a soft blob;
  * more saturated palette: violet core, magenta-rose shade, rose body, light pink, pink-white tips;
  * fewer clusters in front of the wood (35%) so limbs and twigs cross the blossom and run into the crown;
  * mass underpaint (Painter17 bodies) shrunk so more sky reads through the crown (lacy, not solid).
"""
import math

import numpy as np
import cv2

from s05_sakura_r16 import _splat, _band_col, TRANS
from s05_sakura_r15 import _unit, _rot
from s05_sakura_r17 import Tree17, Painter17, PEACH, HALO, _kmeans

BANDS18 = np.array([
    (0.47, 0.3, 0.6),      # 0 violet core
    (0.76, 0.36, 0.6),     # 1 magenta-rose shade
    (0.94, 0.57, 0.73),    # 2 rose body
    (0.99, 0.74, 0.83),    # 3 light pink
    (1.02, 0.86, 0.9),     # 4 pale tip
    (1.05, 0.96, 0.95),    # 5 hot
], np.float32)
BANDS18_FAR = (BANDS18 * 0.74 + np.array([0.9, 0.86, 0.95], np.float32) * 0.26).astype(np.float32)


class Tree18(Tree17):
    def _gate(self, rng):
        """round 18: more / larger low-frequency sky holes than Tree16 (a lacy crown, cm5_01)"""
        cl = [c for c in self.clusters if c['y'] > 0.75 * self.ht]
        if not cl:
            self.clusters = cl
            return
        X = np.array([c['x'] for c in cl])
        Y = np.array([c['y'] for c in cl])
        cx, cy = np.median(X), np.median(Y)
        sx_, sy_ = X.std() + 1e-6, Y.std() + 1e-6
        nb = 10
        bx = rng.uniform(X.min(), X.max(), nb)
        by = rng.uniform(Y.min(), Y.max(), nb)
        br = rng.uniform(0.25, 0.5, nb) * self.s
        keep = []
        for c in cl:
            dn = math.hypot((c['x'] - cx) / sx_, (c['y'] - cy) / sy_)
            p = 0.93 - 0.18 * max(0.0, dn - 1.0)
            hole = np.exp(-((c['x'] - bx) ** 2 + (c['y'] - by) ** 2) / (br * br)).max()
            p *= 1.0 - 0.85 * hole
            if rng.random() < p:
                keep.append(c)
        self.clusters = keep


class Painter18(Painter17):
    FRONT = 0.35

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bands = BANDS18_FAR if self.far else BANDS18

    def _floret_r(self, R):
        px = self.px
        return float(np.clip(0.3 * R, 2.2 * px, 5.5 * px))

    def _cluster_stamps(self, c, v, trans, rng):
        px = self.px
        Ls = self.Ls
        bands = self.bands
        R = c['pr']
        nh = int(rng.integers(3, 7))
        st = []
        tx, ty = c['tx'], -c['ty']
        heads = []
        for h in range(nh):
            u = rng.uniform(-1, 1)
            w_ = rng.normal(0, 0.35)
            hx = c['px'] + (u * tx - w_ * ty) * R * 0.6
            hy = c['py'] + (u * ty + w_ * tx) * R * 0.6
            hr = R * rng.uniform(0.4, 0.66)
            heads.append((hx, hy, hr))
        heads.sort(key=lambda t: -(t[0] * Ls[0] + t[1] * Ls[1]))
        mv = c.get('mv', None)
        fr0 = self._floret_r(R)
        for (hx, hy, hr) in heads:
            off = ((hx - c['px']) * Ls[0] + (hy - c['py']) * Ls[1]) / max(R, 1e-3)
            vh = v + 0.3 * off + rng.normal(0, 0.1)
            if mv is not None:
                vh += -0.9 * (hy - c['py']) / max(mv[2], 1.0)
            vq = float(np.clip(np.round(vh), 0, 4))
            # hard-edged dark core under the clump (the gaps between blossoms read as shadow separation)
            cc = _band_col(bands, np.array(max(vq - 1.6, 0.0))).astype(np.float32)
            cc = cc + (cc - cc.mean()) * 0.15
            st.append((hx - Ls[0] * hr * 0.12, hy - Ls[1] * hr * 0.12 + 0.1 * hr, hr * 0.62, rng.uniform(0, 6.3),
                       float(rng.integers(5, 8)), 0.2, 0.9 * px, cc[0], cc[1], cc[2], 0.0, 0.0, 9.0,
                       0.0, 0.0, 0.0, 1.0))
            # the blossoms of the clump
            fr = fr0 * rng.uniform(0.85, 1.15)
            nf = int(np.clip((hr / fr) ** 2 * 1.25 * (0.6 + 0.4 * self.detail), 4, 34))
            fl = []
            for _ in range(nf):
                a = rng.uniform(0, 2 * math.pi)
                rr = hr * math.sqrt(rng.uniform(0, 1)) * 0.95
                fx, fy = hx + math.cos(a) * rr, hy + math.sin(a) * rr
                s_ = ((fx - hx) * Ls[0] + (fy - hy) * Ls[1]) / max(hr, 1e-3)      # +1 on the sun side
                fl.append((s_, fx, fy, rr / max(hr, 1e-3)))
            fl.sort(key=lambda q: q[0])                    # far side first, sun-side blossoms on top
            for (s_, fx, fy, rn) in fl:
                ft = vq + 0.75 * s_ + 0.35 * rn * max(s_, 0.0) + rng.normal(0, 0.22)
                ft = float(np.clip(ft, 0.0, 4.7))
                fc = _band_col(bands, np.array(ft)).astype(np.float32)
                fc = fc + (fc - fc.mean()) * rng.uniform(-0.05, 0.2)
                if trans > 0:
                    fc = fc + (TRANS * 1.04 - fc) * trans * 0.6 * max(s_ + 0.3, 0.0)
                # petal crescent: the sun-side petal edge a notch lighter
                cres = (_band_col(bands, np.array(min(ft + 1.0, 5.0))) - fc) * 0.6
                st.append((fx, fy, fr * rng.uniform(0.8, 1.2), rng.uniform(0, 6.3), 5.0, 0.5, 0.7 * px,
                           fc[0], fc[1], fc[2], Ls[0], Ls[1], 0.3, cres[0], cres[1], cres[2], 1.0))
        return st

    def paint(self, tree, rng, margin=10):
        # fewer clusters in front of the wood: limbs and twigs run into / across the crown
        hr = np.random.default_rng(len(tree.clusters) * 7 + 3)
        for c in tree.clusters:
            c['front'] = bool(hr.random() < self.FRONT)
        return self._paint17(tree, rng, margin)

    def _paint17(self, tree, rng, margin=10):
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
        # ---- masses (wide tiers)
        # connected blossom bodies first (an isolated clump is its own little mass, lit on its own), then the
        # big bodies split into wide tiers by k-means
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
                             ytop=float(np.percentile(ys, 5)), ybot=float(np.percentile(ys, 95)),
                             base=float(mr.normal(0, 0.25))))
        # ---- 3:1 size variety: big bunches in the body of a mass, small ones on its rim
        rr = np.zeros(n)
        for i, c in enumerate(cl):
            m = info[lab[i]]
            dn = math.hypot((c['px'] - m['cx']) / m['rx'], (c['py'] - m['cy']) / m['ry'])
            f = math.exp(mr.normal(0.0, 0.3)) * (1.35 - 0.5 * min(dn, 1.3))
            f = float(np.clip(f, 0.55, 1.65))
            c['pr'] = max(c['pr'] * f, 2.5 * px)
            rr[i] = dn
        # masses brighter toward the top of the crown
        ytop, ybot = Y.min(), Y.max()
        for m in info:
            if m is not None:
                m['hg'] = 1.0 - (m['cy'] - ytop) / max(ybot - ytop, 1.0)
        fld = self._field(cl, H, W)
        Dsun = self._at(fld['Dsun'], X, Y)
        D1 = self._at(fld['D1'], X, Y)
        face = self._at(fld['face'], X, Y)
        # ---- per-mass value: position in the mass (top -> bottom), light direction, crown height
        v = np.zeros(n)
        for i, c in enumerate(cl):
            m = info[lab[i]]
            tv = (c['py'] - m['cy']) / m['ry']                       # -1 top .. +1 bottom
            tl = ((c['px'] - m['cx']) * Ls[0] + (c['py'] - m['cy']) * Ls[1]) / max(m['rx'], m['ry'])
            vv = 2.5 - 1.25 * tv + 0.55 * tl - 1.1 * max(tv - 0.25, 0.0) + 0.7 * (m['hg'] - 0.5) + m['base']
            v[i] = vv
            c['mv'] = (m['cx'], m['cy'], m['ry'])
        v = v - 0.6 * np.clip(Dsun - 0.75, 0, 1)
        front = np.array([c['front'] for c in cl])
        v = v + np.where(front, 0.15, -0.45)
        # ---- mass bodies (underpaint) at card res
        q = 2
        hq, wq = H // q + 1, W // q + 1
        bodies = []
        pockets = []
        medr = float(np.median([c['pr'] for c in cl])) if n else 4 * px
        for j, m in enumerate(info):
            if m is None:
                bodies.append(None)
                continue
            mk = np.zeros((hq, wq), np.uint8)
            for i in np.nonzero(lab == j)[0]:
                c = cl[i]
                cv2.circle(mk, (int(c['px'] / q), int(c['py'] / q)), max(1, int(c['pr'] * 0.8 / q)), 255, -1,
                           cv2.LINE_AA)
            kr = max(3, int(1.3 * medr / q)) | 1
            mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr)))
            ke = max(3, int(2.0 * medr / q)) | 1          # round 18: underpaint well inside the blossom
            mk = cv2.erode(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
            bodies.append(mk)
        # sky holes in the upper part of the upper masses (scalloped), clusters inside removed
        holes = []
        order_m = sorted([j for j in range(nm) if info[j] is not None], key=lambda j: info[j]['cy'])
        nh_ = 0
        for j in order_m:
            m = info[j]
            if m['hg'] < 0.35:
                continue
            for _ in range(int(mr.integers(1, 4))):
                hx = m['cx'] + mr.uniform(-0.6, 0.6) * m['rx']
                hy = m['cy'] + mr.uniform(-0.75, 0.05) * m['ry']
                hr = medr * mr.uniform(0.6, 1.3)
                holes.append((hx, hy, hr))
                nh_ += 1
        keep = np.ones(n, bool)
        for (hx, hy, hr) in holes:
            d = np.hypot(X - hx, Y - hy)
            keep &= ~(d < hr * 0.9)
        # ---- shadow pockets where the limbs enter a mass
        allb = np.zeros((hq, wq), np.uint8)
        for b in bodies:
            if b is not None:
                allb = np.maximum(allb, b)
        for w in tree.wood:
            if w['d'] > 2:
                continue
            P = to_px(w['P'])
            ins = allb[np.clip((P[:, 1] / q).astype(int), 0, hq - 1), np.clip((P[:, 0] / q).astype(int), 0, wq - 1)]
            ins = ins > 128
            for i in range(3, len(P)):
                if ins[i] and not ins[i - 1] and not ins[i - 2] and not ins[i - 3]:
                    # a bit inside the mass - only where the mass is big and solid around the entry
                    ii = min(i + 2, len(P) - 1)
                    rq = int(3.5 * medr / q)
                    cx_q, cy_q = int(P[ii, 0] / q), int(P[ii, 1] / q)
                    win = allb[max(0, cy_q - rq):cy_q + rq + 1, max(0, cx_q - rq):cx_q + rq + 1]
                    if win.size and (win > 128).mean() > 0.62:
                        pockets.append((P[ii, 0], P[ii, 1], medr * mr.uniform(1.1, 1.7)))
                    break
        if len(pockets) > 8:
            sel_ = mr.choice(len(pockets), 8, replace=False)
            pockets = [pockets[i] for i in sorted(sel_)]
        for (px_, py_, pr_) in pockets:
            d = np.hypot(X - px_, Y - py_)
            v = v - 1.8 * np.exp(-(d / pr_) ** 2)
        # thin, sun-facing edge clusters: warm transmitted light
        thin = np.clip((0.97 - D1) / 0.45, 0, 1)
        cx_, cy_ = np.median(X), np.median(Y)
        sw = np.clip(((X - cx_) * Ls[0] + (Y - cy_) * Ls[1]) / max(np.percentile(np.abs(X - cx_), 90), 1.0), 0, 1)
        trans = np.clip((thin + 0.5 * sw) * np.clip(face * 3.0 + 0.4 * sw, 0, 1), 0, 1) * (v < 3.6) * 0.7
        self.dbg = dict(trans=trans, v=v, lab=lab)
        # ---- paint: per mass (top -> bottom): body, back clusters; then wood; then front clusters
        C = np.zeros((H, W, 3), np.float32)
        A = np.zeros((H, W), np.float32)
        yy, xx = np.mgrid[0:hq, 0:wq].astype(np.float32)
        yy *= q
        xx *= q
        hole_m = np.zeros((hq, wq), np.float32)
        for (hx, hy, hr) in holes:
            cv2.circle(hole_m, (int(hx / q), int(hy / q)), max(1, int(hr / q)), 1.0, -1, cv2.LINE_AA)
            for _ in range(4):           # scalloped rim
                a_ = mr.uniform(0, 6.3)
                cv2.circle(hole_m, (int((hx + math.cos(a_) * hr * 0.8) / q), int((hy + math.sin(a_) * hr * 0.8) / q)),
                           max(1, int(hr * mr.uniform(0.35, 0.55) / q)), 1.0, -1, cv2.LINE_AA)
        pk = np.zeros((hq, wq), np.float32)
        for (px_, py_, pr_) in pockets:
            pk = np.maximum(pk, np.exp(-(((xx - px_) ** 2 + (yy - py_) ** 2) / (pr_ * 0.9) ** 2)))
        stf_all = []
        for j in order_m:
            m = info[j]
            b = bodies[j].astype(np.float32) / 255.0 * (1 - hole_m)
            tv = (yy - m['cy']) / m['ry']
            tl = ((xx - m['cx']) * Ls[0] + (yy - m['cy']) * Ls[1]) / max(m['rx'], m['ry'])
            vb = 0.7 - 1.0 * tv + 0.4 * tl - 0.9 * np.clip(tv - 0.25, 0, None) + 0.6 * (m['hg'] - 0.5) - 2.0 * pk
            vb = np.clip(vb, 0, 1.6)
            vb = np.round(vb * 2) / 2
            bc = _band_col(self.bands, vb).astype(np.float32)
            bc = bc * (1 - 0.3 * (pk > 0.45)[..., None])
            bcf = cv2.resize(bc, (W, H), interpolation=cv2.INTER_LINEAR)[:H, :W]
            bf = cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR)[:H, :W]
            C = C * (1 - bf[..., None]) + bcf * bf[..., None]
            A = A + bf * (1 - A)
            idx = [i for i in np.nonzero((lab == j) & keep)[0]]
            idx.sort(key=lambda i: v[i])
            stb = []
            for i in idx:
                c = cl[i]
                r2 = np.random.default_rng(c['seed'])
                st = self._cluster_stamps(c, float(v[i]), float(trans[i]), r2)
                (stf_all if c['front'] else stb).extend(st)
            if stb:
                _splat(C, A, np.array(stb, np.float64))
        wcol, wcov = self._wood(tree, to_px, H, W, fld, rng)
        C = C * (1 - wcov[..., None]) + wcol * wcov[..., None]
        A = A + wcov * (1 - A)
        # ---- sprigs: thin twigs with 2-5 blossoms past the lit silhouette
        spr = []
        tw = np.zeros((H, W), np.float32)
        cand = [i for i in range(n) if keep[i] and rr[i] > 0.75 and
                ((cl[i]['py'] - info[lab[i]]['cy']) < 0.2 * info[lab[i]]['ry'] or
                 ((cl[i]['px'] - info[lab[i]]['cx']) * Ls[0] > 0))]
        nsp = int(min(len(cand), (26 if not self.far else 8) * self.detail + 4))
        if cand:
            pick = mr.choice(len(cand), size=nsp, replace=False)
            for pi in pick:
                c = cl[cand[pi]]
                m = info[lab[cand[pi]]]
                d = _unit([c['px'] - m['cx'], (c['py'] - m['cy']) * 1.3 - 0.4 * m['ry']])
                d = _rot(d, mr.uniform(-30, 30))
                L = c['pr'] * mr.uniform(1.4, 2.6)
                p0 = np.array([c['px'], c['py']])
                p1 = p0 + d * L
                bend = _rot(d, mr.choice([-1, 1]) * 90) * L * 0.12
                pts = np.array([p0 + (p1 - p0) * u + bend * 4 * u * (1 - u) for u in np.linspace(0, 1, 8)])
                cv2.polylines(tw, [np.round(pts * 4).astype(np.int32)], False, 1.0,
                              max(1, int(round(0.9 * px * 4))), cv2.LINE_AA, shift=2)
                nb = int(mr.integers(2, 6))
                for b_ in range(nb):
                    u = mr.uniform(0.35, 1.0)
                    bp = p0 + (p1 - p0) * u + bend * 4 * u * (1 - u) + mr.normal(0, 0.12, 2) * c['pr']
                    br = self._floret_r(c['pr']) * mr.uniform(0.9, 1.25)
                    lit = bp[1] - m['cy'] < 0
                    vq = 4.0 if lit else 3.0
                    fc = _band_col(self.bands, np.array(vq)).astype(np.float32)
                    if (bp - np.array([m['cx'], m['cy']])) @ Ls > 0:
                        fc = fc + (PEACH - fc) * 0.35
                    cres = (self.bands[5] - fc) * 0.8
                    spr.append((bp[0], bp[1], max(br, 1.1 * px), mr.uniform(0, 6.3), 5.0, 0.5, 0.7 * px,
                                fc[0], fc[1], fc[2], Ls[0], Ls[1], 0.3, cres[0], cres[1], cres[2], 1.0))
        tw = np.clip(tw * 0.85, 0, 1)
        C = C * (1 - tw[..., None]) + self.wp['inner'] * tw[..., None]
        A = A + tw * (1 - A)
        if stf_all:
            _splat(C, A, np.array(stf_all, np.float64))
        if spr:
            _splat(C, A, np.array(spr, np.float64))
        # ---- sun-facing silhouette: warm rim, peach transmitted band, halation; shadow side: lost edge
        C = C / np.maximum(A, 1e-4)[..., None]
        sh = 1.6 * px
        M = np.float32([[1, 0, -Ls[0] * sh], [0, 1, -Ls[1] * sh]])
        As = cv2.warpAffine(A, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        rim = np.clip(A - As, 0, 1) * (1 - wcov)
        C += (np.array([1.1, 0.95, 0.86], np.float32) - C) * (rim * 0.6)[..., None]
        sh2 = 7.0 * px
        M = np.float32([[1, 0, -Ls[0] * sh2], [0, 1, -Ls[1] * sh2]])
        As2 = cv2.warpAffine(cv2.GaussianBlur(A, (0, 0), 2.5 * px), M, (W, H), borderMode=cv2.BORDER_CONSTANT)
        band = np.clip(A - As2, 0, 1)
        band = cv2.GaussianBlur(band, (0, 0), 2.0 * px) * A * (1 - wcov)
        lumC = C.mean(-1)
        C += (PEACH - C) * (band * 0.4 * np.clip((1.02 - lumC) / 0.25, 0.2, 1))[..., None]
        # lost edges on the side away from the sun and on the undersides
        sh3 = 2.5 * px
        M2 = np.float32([[1, 0, Ls[0] * sh3], [0, 1, Ls[1] * sh3 + 0.8 * sh3]])
        Ad = cv2.warpAffine(A, M2, (W, H), borderMode=cv2.BORDER_CONSTANT)
        lost = np.clip(A - Ad, 0, 1)
        lost = cv2.GaussianBlur(lost, (0, 0), 2.5 * px)
        yyf = np.arange(H, dtype=np.float32)[:, None]
        A = A * np.clip((oy + 0.02 * k - yyf) / (1.5 * px) + 0.5, 0, 1)
        pm = np.dstack([C * A[..., None], A])
        pmb = cv2.GaussianBlur(pm, (0, 0), 1.5 * px)
        lk = np.clip(lost * 1.1, 0, 0.6)[..., None] * (1 - wcov[..., None])
        pm = pm * (1 - lk) + pmb * lk
        # halation: warm glow bleeding into the sky around the sun-facing crown
        if not self.far or True:
            Ab = cv2.GaussianBlur(A * (1 - wcov), (0, 0), 9.0 * px)
            sh4 = 6.0 * px
            M = np.float32([[1, 0, Ls[0] * sh4], [0, 1, Ls[1] * sh4]])
            Ab = cv2.warpAffine(Ab, M, (W, H), borderMode=cv2.BORDER_CONSTANT)
            g = np.clip(Ab - A, 0, 1) * (0.2 if not self.far else 0.12)
            # round 18: no halation filling the sky holes inside the crown (read as smooth pink blobs)
            Ain = cv2.GaussianBlur(A, (0, 0), 4.0 * px)
            g = g * np.clip((0.35 - Ain) / 0.25, 0, 1)
            pa = pm[..., 3]
            add_a = g * (1 - pa)
            pm[..., :3] += HALO * add_a[..., None]
            pm[..., 3] = pa + add_a
        self.stats = dict(n=n, masses=nm, holes=len(holes), pockets=len(pockets), sprigs=len(spr))
        return pm.astype(np.float32), ox, oy
