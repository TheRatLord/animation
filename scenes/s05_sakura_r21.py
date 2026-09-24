"""s05_sakura round 21: Painter21 = CLUSTER-MASS crowns (cm5_01 study, panel: 'fine-grained pink noise and speckle,
no clustered mass structure').

Each blossom cluster is no longer 5-20 separate florets but one painted clump:
  * 3-7 overlapping scalloped flower HEADS (bumpy round dabs, big relative to the florets) fused into one mass, so
    the inside of a clump is solid paint, not speckle;
  * every head carries a crisp two-value split from the sun direction: a lit pale-pink top crescent over a cooler
    magenta-violet underside (per-mass value field decides how far up the terminator sits);
  * florets (lighter five-petal tips) only on the clump silhouette, mostly the upper / sun side;
  * a darker violet-magenta head tucked behind on the down side gives clump-to-clump separation.
"""
import math

import numpy as np

from s05_sakura_r16 import _band_col, TRANS
from s05_sakura_r19 import _tier
from s05_sakura_r20 import Tree20, Painter20

# (0 deep violet, 1 magenta-mauve shade, 2 rose, 3 light pink, 4 pale pink, 5 hot cream)
BANDS21 = np.array([
    (0.5, 0.3, 0.6),
    (0.74, 0.42, 0.7),
    (0.92, 0.56, 0.76),
    (1.0, 0.78, 0.86),
    (1.02, 0.9, 0.93),
    (1.06, 0.99, 0.96),
], np.float32)
BANDS21_FAR = (BANDS21 * 0.74 + np.array([0.9, 0.86, 0.95], np.float32) * 0.26).astype(np.float32)


class Tree21(Tree20):
    def _gate(self, rng):
        """round 21: clumped crown with real sky holes between the cluster groups (cm5_01)"""
        cl = [c for c in self.clusters if c['y'] > 0.75 * self.ht]
        if not cl:
            self.clusters = cl
            return
        X = np.array([c['x'] for c in cl])
        Y = np.array([c['y'] for c in cl])
        cx, cy = np.median(X), np.median(Y)
        sx_, sy_ = X.std() + 1e-6, Y.std() + 1e-6
        nb = 20
        bx = rng.uniform(X.min(), X.max(), nb)
        by = rng.uniform(Y.min(), Y.max(), nb)
        br = rng.uniform(0.25, 0.55, nb) * self.s
        keep = []
        for c in cl:
            dn = math.hypot((c['x'] - cx) / sx_, (c['y'] - cy) / sy_)
            p = 0.9 - 0.3 * max(0.0, dn - 0.9)
            hole = np.exp(-((c['x'] - bx) ** 2 + (c['y'] - by) ** 2) / (br * br)).max()
            p *= 1.0 - 0.92 * hole
            if rng.random() < p:
                keep.append(c)
        self.clusters = keep


class Painter21(Painter20):
    FRONT = 0.4
    BLOS_FULL = True
    WOOD_HARD = True

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bands = BANDS21_FAR if self.far else BANDS21

    def _cluster_stamps(self, c, v, trans, rng):
        px = self.px
        Ls = self.Ls
        bands = self.bands
        R = c['pr'] * 1.12
        tx, ty = c['tx'], -c['ty']
        mv = c.get('mv', None)
        ex = c.get('ex', 0.0)
        # light direction for the crescent: the sun, biased upward (sky light from above)
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
        heads.sort(key=lambda q: -q[1])           # lower heads first -> upper heads overlap them
        vbase = v
        if mv is not None:
            vbase = v - 0.9 * (c['py'] - mv[1]) / max(mv[2], 1.0)
        vq = _tier(vbase)
        # separation head behind (down side, darker)
        bx = c['px'] - lx * R * 0.35
        by = c['py'] - ly * R * 0.35
        dc = _band_col(bands, np.array(max(vq - 1.9, 0.0))).astype(np.float32)
        st.append((bx, by, R * 0.6, rng.uniform(0, 6.3), 7.0, 0.22, 0.9 * px, dc[0], dc[1], dc[2],
                   0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
        for (hx, hy, hr) in heads:
            s_ = ((hx - c['px']) * Ls[0] + (hy - c['py']) * Ls[1]) / max(R, 1e-3)
            hv = vq + 0.3 * s_ - 0.35 * (hy - c['py']) / max(R, 1e-3) + rng.normal(0, 0.12)
            shade_t = float(np.clip(hv - 2.0, 0.45, 2.0))
            lit_t = float(np.clip(hv + 0.6 + 0.8 * ex, 2.4, 5.0))
            sc = _band_col(bands, np.array(shade_t)).astype(np.float32)
            lc = _band_col(bands, np.array(lit_t)).astype(np.float32)
            if trans > 0:
                lc = lc + (TRANS * 1.04 - lc) * trans * 0.45
            # terminator: higher on shaded clumps (small lit cap), lower on exposed ones (big lit face)
            th = float(np.clip(0.4 - 0.25 * (hv - 2.5), -0.3, 0.75))
            # shade dab, then a smaller lit dab shifted toward the light: curved, scalloped terminator
            st.append((hx, hy, hr, rng.uniform(0, 6.3), float(rng.integers(6, 10)), 0.2, 0.8 * px,
                       sc[0], sc[1], sc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
            cov_ = float(np.clip(0.72 - 0.5 * th, 0.4, 0.9))          # lit fraction of the head
            # the lit cap is 2-3 notched flower shapes along the upper edge (not one smooth disc)
            nlc = int(rng.integers(2, 4))
            for j in range(nlc):
                ang = math.atan2(ly, lx) + rng.uniform(-0.9, 0.9)
                fr_ = hr * rng.uniform(0.42, 0.55) * (0.8 + 0.4 * cov_)
                dd = hr - fr_ * rng.uniform(0.75, 1.0) - hr * (cov_ - 0.5) * 0.5
                cc = lc * rng.uniform(0.97, 1.02)
                st.append((hx + math.cos(ang) * dd, hy + math.sin(ang) * dd, fr_, rng.uniform(0, 6.3), 5.0,
                           0.3, 0.8 * px, cc[0], cc[1], cc[2], 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0))
        # florets (lighter petal tips) only on the clump silhouette, mostly on the upper / sun side
        fr0 = float(np.clip(0.24 * R, 2.4 * px, 6.0 * px))
        nfl = int(np.clip(round(R / fr0 * 0.9), 2, 8))
        for _ in range(nfl):
            a = rng.uniform(0, 2 * math.pi)
            dx, dy = math.cos(a), math.sin(a)
            up = -(dx * lx + dy * ly)
            up = dx * lx + dy * ly                  # +1 toward the light
            if up < -0.2 and rng.random() < 0.75:
                continue
            rr = R * rng.uniform(0.85, 1.12)
            fx, fy = c['px'] + dx * rr, c['py'] + dy * rr + 0.1 * R
            fr = fr0 * rng.uniform(0.75, 1.2)
            ft = float(np.clip(vq + 0.9 + 0.8 * max(up, 0) + 0.8 * ex + rng.normal(0, 0.2), 1.0, 5.0))
            ft = min(ft, 3.6)
            if up < 0:
                ft = float(np.clip(vq - 0.6, 0.8, 3.0))
            fc = _band_col(bands, np.array(ft)).astype(np.float32)
            cr_ = (_band_col(bands, np.array(min(ft + 1.0, 5.0))) - fc) * 0.8
            st.append((fx, fy, fr, rng.uniform(0, 6.3), 5.0, 0.5, 0.75 * px,
                       fc[0], fc[1], fc[2], Ls[0], Ls[1], 0.2, cr_[0], cr_[1], cr_[2], 1.0))
        return st

    def _wood(self, tree, to_px, H, W, fld, rng):
        """thin wood inside the crown stays a dark plum silhouette (no pale rim scratches across the blossom)"""
        import cv2
        col, cov = super()._wood(tree, to_px, H, W, fld, rng)
        inside = np.clip(cv2.resize(fld['D2'], (W, H), interpolation=cv2.INTER_LINEAR) * 2.0, 0, 1)
        dark = np.array([0.24, 0.15, 0.22], np.float32) if not self.far else np.array([0.42, 0.34, 0.46], np.float32)
        lum = col.mean(-1)
        k = inside * np.clip((lum - 0.3) / 0.3, 0, 1) * 0.85
        col = col + (dark - col) * k[..., None]
        return col, cov
