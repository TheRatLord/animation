"""s05_sakura round 17: cherry crowns shaded PER MASS (cm5_03 / cm5_01 study).

Reviewer notes addressed: 'evenly sized, evenly coloured popcorn dabs', 'no big painted value masses',
'weak lavender undersides / no shadow pockets where limbs enter', 'no warm transmitted light / halation',
'uniform scalloped silhouette', 'smooth flat-brown trunk tubes'.

Tree17 = Tree16 skeleton + a stronger root flare.

Painter17 (on top of Painter16's cluster stamps / wood raster)
  * the clusters of a crown are grouped (k-means on position, squashed vertically so the groups are wide
    tiers) into 3-6 big MASSES; each mass is painted as one form: a solid underpaint body (closing of the
    cluster discs, darker violet-magenta, hue ~290-310) with a few scalloped sky holes in its upper part,
    then its clusters on top, valued by their position IN THE MASS (pink-white top plane, rose body,
    mauve / violet underside ~30% darker) - per-dab jitter kept small so the shading reads per mass;
  * masses are layered top -> bottom so the lit top of a lower mass cuts crisply across the dark underside of
    the mass above it;
  * cluster sizes vary over ~3:1 (big bunches inside a mass, small ones on its rim);
  * deep violet shadow pockets where the limbs enter a mass (clusters around them darkened too);
  * isolated sprigs: thin twigs with 2-5 blossoms poking out past the lit silhouette;
  * warm peach transmitted band inside the sun-facing silhouette + a soft warm halation bleeding into the
    sky; shadow-side (lower) silhouettes softened into lost edges;
  * bark: crisp cel planes (cool violet shadow side, warm lit side, gold rim), dark horizontal bark bands and
    lenticel dashes, stronger root flare.
"""
import math

import numpy as np
import cv2

from s05_sakura_r15 import _raster_wood, _unit, _rot, _sstep
from s05_sakura_r16 import Tree16, Painter16, _splat, _band_col, TRANS

BANDS17 = np.array([
    (0.56, 0.4, 0.68),     # 0 deep interior: violet (hue ~285)
    (0.8, 0.46, 0.66),     # 1 shade: rose-mauve (hue ~325)
    (0.96, 0.6, 0.7),      # 2 body: warm rose pink
    (1.0, 0.74, 0.79),     # 3 light pink
    (1.03, 0.88, 0.88),    # 4 lit top: pink-white
    (1.05, 0.96, 0.92),    # 5 hot
], np.float32)
BANDS17_FAR = (BANDS17 * 0.72 + np.array([0.9, 0.86, 0.95], np.float32) * 0.28).astype(np.float32)
WOOD17 = dict(sh=(0.17, 0.13, 0.24), mid=(0.33, 0.26, 0.29), lit=(0.52, 0.4, 0.35), rim=(1.1, 0.84, 0.58),
              cool=(0.36, 0.36, 0.56), crack=(0.07, 0.05, 0.09), band=(0.12, 0.09, 0.14),
              lent=(0.62, 0.55, 0.5), moss=(0.42, 0.5, 0.33), lichen=(0.58, 0.6, 0.52), inner=(0.24, 0.19, 0.32))
WOOD17_FAR = dict(sh=(0.28, 0.25, 0.36), mid=(0.38, 0.34, 0.42), lit=(0.52, 0.45, 0.46), rim=(0.95, 0.8, 0.68),
                  cool=(0.4, 0.42, 0.56), crack=(0.2, 0.18, 0.25), band=(0.24, 0.21, 0.3),
                  lent=(0.56, 0.53, 0.54), moss=(0.46, 0.52, 0.42), lichen=(0.56, 0.58, 0.54),
                  inner=(0.38, 0.34, 0.46))
PEACH = np.array([1.06, 0.84, 0.7], np.float32)
HALO = np.array([1.0, 0.86, 0.78], np.float32)


class Tree17(Tree16):
    def __init__(self, rng, *a, **kw):
        super().__init__(rng, *a, **kw)
        # stronger root flare: the trunk widens into the grass
        w = self.wood[0]
        P, r = w['P'], w['r']
        y = np.maximum(P[:, 1], 0.0)
        w['r'] = r * (1.0 + 0.45 * np.exp(-y / (0.3 * self.s))) / (1.0 + 0.6 * np.exp(-y / (0.16 * self.s)))


def _kmeans(X, k, rng, it=12):
    n = len(X)
    k = max(1, min(k, n))
    # farthest-point init (deterministic given rng)
    c = [X[int(rng.integers(n))]]
    for _ in range(1, k):
        d = np.min([((X - q) ** 2).sum(1) for q in c], 0)
        c.append(X[int(np.argmax(d))])
    C = np.array(c, np.float64)
    lab = np.zeros(n, int)
    for _ in range(it):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = np.argmin(d, 1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    return lab, C


class Painter17(Painter16):
    def __init__(self, k, sun_dir=(-0.7, -0.7), px=2.0, far=False, detail=1.0, masses=None, hero=False):
        super().__init__(k, sun_dir=sun_dir, px=px, far=far, detail=detail)
        self.bands = BANDS17_FAR if far else BANDS17
        self.wp = {kk: np.array(v, np.float32) for kk, v in (WOOD17_FAR if far else WOOD17).items()}
        self.masses = masses
        self.hero = hero

    # ------------------------------------------------------------------ blossom stamps (per-mass value)
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
            hx = c['px'] + (u * tx - w_ * ty) * R * 0.55
            hy = c['py'] + (u * ty + w_ * tx) * R * 0.55
            hr = R * rng.uniform(0.42, 0.66)
            heads.append((hx, hy, hr))
        heads.sort(key=lambda t: -(t[0] * Ls[0] + t[1] * Ls[1]))
        mv = c.get('mv', None)          # (mass centre x, y, ry) -> heads valued by their place in the mass
        for (hx, hy, hr) in heads:
            off = ((hx - c['px']) * Ls[0] + (hy - c['py']) * Ls[1]) / max(R, 1e-3)
            vh = v + 0.28 * off + rng.normal(0, 0.08)
            if mv is not None:
                vh += -0.9 * (hy - c['py']) / max(mv[2], 1.0)
            vq = float(np.clip(np.round(vh), 0, 4))
            col = bands[int(vq)].copy()
            col = col + (col - col.mean()) * rng.uniform(-0.05, 0.12)
            if trans > 0:
                col = col + (TRANS - col) * trans * 0.75
            lit = vq >= 2
            aa = (0.8 if lit else 1.0 + 0.2 * (2 - vq)) * px
            cres = (bands[min(int(vq) + 1, 5)] - bands[int(vq)]) * (0.9 if lit else 0.45)
            if trans > 0:
                cres = cres + (TRANS * 1.08 - col) * trans * 0.4
            st.append((hx, hy, hr, rng.uniform(0, 6.3), float(rng.integers(5, 8)), 0.24, aa,
                       col[0], col[1], col[2], Ls[0], Ls[1], 0.35 if lit else 0.6,
                       cres[0], cres[1], cres[2], 1.0))
            if hr > 3.5 * px:
                for _ in range(int(rng.integers(1, 3))):
                    a_ = rng.uniform(0, 6.3)
                    rr_ = hr * rng.uniform(0.1, 0.55)
                    fc = _band_col(bands, np.array(max(vq - 0.4, 0.0))).astype(np.float32)
                    st.append((hx + math.cos(a_) * rr_, hy + math.sin(a_) * rr_, hr * rng.uniform(0.25, 0.36),
                               rng.uniform(0, 6.3), 5.0, 0.4, 0.8 * px, fc[0], fc[1], fc[2], 0.0, 0.0, 9.0,
                               0.0, 0.0, 0.0, 0.7))
            # florets only on the lit rim of lit heads (lace on the light side, quiet shadow side)
            if vq >= 2:
                nf = int(rng.integers(1, 5) * self.detail + 0.5)
                fr_ = max(0.9 * px, hr * rng.uniform(0.26, 0.38))
                for _ in range(nf):
                    a = math.atan2(Ls[1], Ls[0]) + rng.normal(0, 1.1)
                    fx = hx + math.cos(a) * hr * rng.uniform(0.75, 1.0)
                    fy = hy + math.sin(a) * hr * rng.uniform(0.75, 1.0)
                    ft = min(vq + 0.6, 4.6)
                    fc = _band_col(bands, np.array(ft)).astype(np.float32)
                    if trans > 0:
                        fc = fc + (TRANS * 1.05 - fc) * trans * 0.7
                    st.append((fx, fy, fr_ * rng.uniform(0.8, 1.2), rng.uniform(0, 6.3), 5.0, 0.42,
                               0.7 * px, fc[0], fc[1], fc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
        return st

    # ------------------------------------------------------------------ wood (cel bark)
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
            for i in range(len(P) - 1):
                segs.append((P[i, 0], P[i, 1], P[i + 1, 0], P[i + 1, 1], max(r[i], rmin), max(r[i + 1], rmin),
                             v[i], v[i + 1], float(w['d']), float(w['rid']), Nn[i, 0], Nn[i, 1], Nn[i + 1, 0],
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
        facing = S * (NX * Ls[0] + NY * Ls[1])
        a_s = np.abs(S)
        tex = self._bark_tex(rng)
        th_, tw_ = tex.shape
        # a little low-frequency wobble so the cel terminator is painted, not a ruler line
        rm = np.maximum(RR / self.k, 1e-3)
        ang = np.arcsin(np.clip(S, -1, 1))
        arc = ang * rm
        tvw = ((Vv / 0.25 + RID * 17.0) % th_).astype(np.float32)
        tuw = ((arc / 0.2 + RID * 3.0) % tw_).astype(np.float32)
        wob = cv2.remap(self._blob_tex, tuw, tvw, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fct = facing + 0.25 * wob
        col = np.broadcast_to(wp['sh'], (H, W, 3)).astype(np.float32).copy()
        # cel planes: violet shadow | brown mid | warm lit (crisp steps)
        m1 = _sstep((fct - 0.05) / 0.08 + 0.5)
        col += (wp['mid'] - wp['sh']) * m1[..., None]
        m2 = _sstep((fct - 0.55) / 0.08 + 0.5) * np.clip((RR - 2.0 * px) / (3 * px), 0, 1)
        col += (wp['lit'] - wp['mid']) * m2[..., None]
        cb = _sstep((-fct - 0.7) / 0.12 + 0.5) * np.clip((RR - 3 * px) / (4 * px), 0, 1)
        col += (wp['cool'] - col) * (cb * 0.6)[..., None]
        thick = np.clip((RR - 3.0 * px) / (6 * px), 0, 1)
        if thick.max() > 0:
            # horizontal bark bands (cherry bark rings): dark cel bands of varied width / spacing
            bn = cv2.remap(self._blob_tex, ((arc / 0.35 + RID * 7.0) % tw_).astype(np.float32),
                           ((Vv / 0.06 + RID * 29.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_WRAP)
            ph = Vv / 0.13 + 0.8 * bn + RID * 0.37
            fr = ph - np.floor(ph)
            j = np.floor(ph)
            hs = np.sin(j * 12.9898 + RID * 78.233) * 43758.5453
            hs = hs - np.floor(hs)
            bw = 0.1 + 0.22 * hs
            part = cv2.remap(self._blob_tex, ((arc / 0.05 + j * 3.3) % tw_).astype(np.float32),
                             ((j * 5.7 + RID * 3.0) % th_).astype(np.float32), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_WRAP)
            bw = 0.06 + 0.14 * hs
            band = _sstep((bw - np.abs(fr - 0.5) * 2 * 0.5) / 0.03 + 0.5) * (hs > 0.4) * _sstep((part + 0.1) / 0.1 + 0.5)
            band *= np.clip(1.25 - a_s, 0, 1)
            col += (wp['band'] - col) * (band * thick * (0.75 - 0.3 * np.clip(fct, 0, 1)))[..., None]
            # vertical fissures (fainter than round 16)
            tu = ((arc / 0.018 + RID * 37.0) % tw_).astype(np.float32)
            tv = ((Vv / 0.05 + RID * 91.0) % th_).astype(np.float32)
            n = cv2.remap(tex, tu, tv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            crack = _sstep((np.abs(n) - 0.07) / -0.04 + 0.5) * (1 - 0.6 * np.clip(fct, 0, 1))
            crack *= np.clip(1.2 - a_s, 0, 1)
            col += (wp['crack'] - col) * (crack * thick * 0.5)[..., None]
            # horizontal lenticel dashes: dark dash + pale lip on the lit side
            sp = 0.05
            q = Vv / sp + 0.25 * np.sin(arc * 11 + RID)
            jj = np.floor(q)
            fq = q - jj
            ca = arc / 0.04 + np.sin(jj * 3.1) * 7.0
            ci = np.floor(ca)
            fa = ca - ci
            h2 = np.sin(jj * 12.9898 + ci * 4.1414 + RID * 78.233) * 43758.5453
            h2 = h2 - np.floor(h2)
            dash = ((fq < 0.16) & (h2 < 0.45) & (fa > 0.15) & (fa < 0.85)).astype(np.float32)
            dash *= np.clip(1.1 - a_s, 0, 1)
            col += (wp['crack'] - col) * (dash * thick * 0.7)[..., None]
            lip = ((fq >= 0.16) & (fq < 0.28) & (h2 < 0.45) & (fa > 0.15) & (fa < 0.85)).astype(np.float32)
            col += (wp['lent'] - col) * (lip * thick * 0.35 * np.clip(fct + 0.3, 0, 1))[..., None]
            # moss / lichen low on the shade side
            tv2 = ((Vv / 0.35 + RID * 13.0) % th_).astype(np.float32)
            tu2 = ((arc / 0.09 + RID * 5.0) % tw_).astype(np.float32)
            mn = cv2.remap(self._blob_tex, tu2, tv2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            low = np.clip(1.4 - Vv / 1.4, 0.2, 1.0)
            moss = _sstep((mn - 0.55) / 0.1 + 0.5) * np.clip(0.3 - fct, 0, 1) * low
            col += (wp['moss'] - col) * (moss * thick * 0.55)[..., None]
        # warm lit rim on the sun side
        rw = np.clip((1.8 * px) / np.maximum(RR, 0.5), 0.05, 0.6)
        rimk = _sstep((a_s - (1.0 - rw)) / (0.4 * rw) + 0.5) * _sstep((facing - 0.3) / 0.2 + 0.5)
        inside = np.clip(cv2.resize(fld['D2'], (W, H), interpolation=cv2.INTER_LINEAR) * 1.6, 0, 1)
        col += (wp['rim'] - col) * (rimk * (0.95 - 0.5 * inside))[..., None]
        col += (wp['inner'] - col) * (inside * 0.55 * (1 - thick))[..., None]
        return col.astype(np.float32), cov

    # ------------------------------------------------------------------ paint
    def paint(self, tree, rng, margin=10):
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
            mk = cv2.erode(mk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
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
            vb = 1.2 - 1.0 * tv + 0.4 * tl - 0.9 * np.clip(tv - 0.25, 0, None) + 0.6 * (m['hg'] - 0.5) - 2.0 * pk
            vb = np.clip(vb, 0, 3.2)
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
                    br = c['pr'] * mr.uniform(0.2, 0.34)
                    lit = bp[1] - m['cy'] < 0
                    vq = 4.0 if lit else 3.0
                    fc = _band_col(self.bands, np.array(vq)).astype(np.float32)
                    if (bp - np.array([m['cx'], m['cy']])) @ Ls > 0:
                        fc = fc + (PEACH - fc) * 0.35
                    cres = (self.bands[5] - fc) * 0.8
                    spr.append((bp[0], bp[1], max(br, 1.1 * px), mr.uniform(0, 6.3), 5.0, 0.42, 0.7 * px,
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
            g = np.clip(Ab - A, 0, 1) * (0.32 if not self.far else 0.18)
            pa = pm[..., 3]
            add_a = g * (1 - pa)
            pm[..., :3] += HALO * add_a[..., None]
            pm[..., 3] = pa + add_a
        self.stats = dict(n=n, masses=nm, holes=len(holes), pockets=len(pockets), sprigs=len(spr))
        return pm.astype(np.float32), ox, oy
