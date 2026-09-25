"""s05_sakura round 26 (reviewer FAIL on round 25: 'hairy fan of thin twigs across the sky gaps, jointed / banded limb
segments with kinks, salty white stipple on the clump tops, weak violet shade under the clumps').

  * Tree26: limbs grow without the random kinks (same random stream -> same layout), and the wood is PRUNED to the
    blossom: every limb of depth >= 2 is cut back to the last point that carries a clump or a surviving child limb
    (+ a short tapering end buried in that clump); bare limbs that carry nothing are removed -> the twig length
    outside the blossom drops sharply and the rest of the fine wood lives inside the masses;
  * Painter26: limbs painted as one continuous tapered stroke - no lenticel ring rows / knots on the limbs (sparse,
    faint lenticels only on the trunk), a narrower single lit edge and a softer terminator on the shade side;
  * regrade26: regrade25 + (a) the lit tops merged into broad pale-cream shapes (flower-scale stipple and pinholes
    closed inside the lit masses, florets kept only on the silhouette), (b) a clearer violet-grey shade value under
    and inside every clump (per-clump underside from the clump seeds), then the R25 twig trim.
"""
import math

import numpy as np
import cv2

import s05_sakura_r23 as R23
import s05_sakura_r24 as R24
import s05_sakura_r25 as R25

_ss = R24._ss
clumps_px = R25.clumps_px


# ============================================================================ tree
class Tree26(R25.Tree25):
    KINK = 1.0           # layout kept; kinks rounded off by the painter's SMOOTH pass
    KEEP_TAIL = 1          # points kept past the last blossom-carrying point
    KEEP_HEAD = 1          # points kept before the first blossom-carrying point of a hidden-start twig
    DROP_DOBUKI = False
    START_D = 3            # twigs this deep (carrying no children) start at their blossom

    def __init__(self, rng, *a, **kw):
        super().__init__(rng, *a, **kw)
        if self.DROP_DOBUKI:
            # the small forced-front tufts on the trunk / scaffolds melted into a translucent pale film over the
            # limbs (read as pale 'joint' bands) -> dropped; the limbs run clean into the clumps
            self.clusters = [c for c in self.clusters if not c.get('force_front')]
        self._prune()

    def _prune(self):
        wood = self.wood
        n = len(wood)
        if n < 2 or not self.clusters:
            return
        CX = np.array([c['x'] for c in self.clusters])
        CY = np.array([c['y'] for c in self.clusters])
        CR = np.array([c['R'] for c in self.clusters])
        # parent / attach index of every limb (a child starts exactly on a point of an earlier limb)
        parent = [-1] * n
        att = [-1] * n
        for j in range(1, n):
            p0 = wood[j]['P'][0]
            dj = wood[j]['d']
            for i in range(j - 1, -1, -1):
                if wood[i]['d'] != dj - 1:
                    continue
                dd = np.hypot(wood[i]['P'][:, 0] - p0[0], wood[i]['P'][:, 1] - p0[1])
                q = int(np.argmin(dd))
                if dd[q] < 1e-9:
                    parent[j], att[j] = i, q
                    break
        need = [-1] * n
        first = [-1] * n
        holds = [False] * n
        for j in range(n - 1, -1, -1):
            w = wood[j]
            P = w['P']
            dx = P[:, 0][:, None] - CX[None]
            dy = P[:, 1][:, None] - CY[None]
            dmin = (np.hypot(dx, dy) - 1.05 * CR[None]).min(1)
            hit = np.nonzero(dmin < 0.0)[0]
            e = int(hit[-1]) if len(hit) else -1
            need[j] = max(need[j], e)
            first[j] = int(hit[0]) if len(hit) else -1
            if w['d'] >= 2 and need[j] < 0:
                continue                         # bare limb carrying nothing: removed, does not hold its parent
            if parent[j] >= 0:
                need[parent[j]] = max(need[parent[j]], att[j])
                holds[parent[j]] = True
        keep = []
        for j, w in enumerate(wood):
            if w['d'] < 1:
                keep.append(w)
                continue
            e = need[j]
            if w['d'] >= 2 and e < 0:
                continue
            nP = len(w['P'])
            if w['d'] == 1:
                e = max(e, int(0.55 * (nP - 1)))
            e = min(nP - 1, e + self.KEEP_TAIL)
            if e < 1:
                continue
            b0 = 0
            if w['d'] >= self.START_D and not holds[j]:
                # fine twig that carries no child: its bare run from the parent limb across the sky is hidden,
                # the twig starts (tapered in) just before its first flower cluster
                b0 = max(0, first[j] - self.KEEP_HEAD) if first[j] >= 0 else 0
            if e < nP - 1 or b0 > 0:
                P = w['P'][b0:e + 1].copy()
                r = np.asarray(w['r'][b0:e + 1], np.float64).copy()
                m = min(len(r), 4)
                if e < nP - 1:
                    m2 = min(len(r), 6)
                    r[-m2:] *= np.linspace(1.0, 0.2, m2) ** 0.8    # the cut end tapers to a point in its clump
                if b0 > 0:
                    r[:m] *= np.linspace(0.5, 1.0, m)
                r = np.maximum(r, self.rmin)
                if len(P) < 2:
                    continue
                w = dict(w, P=P, r=r)
            keep.append(w)
        self.wood = keep


# ============================================================================ painter
class Painter26(R25.Painter25):
    LENT = 0.25          # sparse lenticels, trunk only (see MARK_A / thickness)
    KNOT = 0.0
    MARK_A = 0.45
    TERM_W = 0.16        # softer terminator (shade side)
    LIT_T = 0.72         # narrower single lit edge
    BLOS_NOWOOD = True
    WOOD_UP = 0.8        # wood lit from above-left: every limb gets its lit edge on the upper side
    SMOOTH = 8           # limb centre lines smoothed into continuous curves (no kinks)

    def _wood(self, tree, to_px, H, W, fld, rng):
        """R23 wood, but the 'inside the crown' darkening follows a wide, smooth field (no hard value seam
        across a limb where it enters the blossom -> the limb reads as one continuous stroke)"""
        col, cov = self._wood_base(tree, to_px, H, W, fld, rng)
        d2 = cv2.resize(fld['D2'], (W, H), interpolation=cv2.INTER_LINEAR)
        sg = max(4.0, 14.0 * self.px)
        d2 = cv2.GaussianBlur(d2, (0, 0), sg)
        inside = _ss(np.clip(d2 * 1.6, 0, 1))
        dark = np.array([0.27, 0.2, 0.26], np.float32) if not self.far else np.array([0.46, 0.4, 0.5], np.float32)
        lum = col.mean(-1)
        k = inside * np.clip((lum - 0.3) / 0.3, 0, 1) * 0.6
        col = col + (dark - col) * k[..., None]
        return col, cov


# ============================================================================ value pass
P_SHADE = np.array([0.66, 0.56, 0.74], np.float32)     # violet-grey underside
P_CREAM = np.array([1.0, 0.95, 0.92], np.float32)      # pale cream lit shape


def regrade26(pm, sun_dir, px, far=False, trim=True, clump_px=None, clumps=None):
    out = R25.regrade25(pm, sun_dir, px, far=far, trim=False, clump_px=clump_px, clumps=clumps)
    H, W = out.shape[:2]
    A = out[..., 3]
    C = out[..., :3] / np.maximum(A, 1e-4)[..., None]
    r, g, b = C[..., 0], C[..., 1], C[..., 2]
    lum = C.mean(-1)
    pink = np.clip((b - g + 0.02) / 0.05, 0, 1) * np.clip((r - g - 0.03) / 0.05, 0, 1) * \
        np.clip((lum - 0.4) / 0.1, 0, 1)
    Bw = pink * np.clip((A - 0.25) / 0.5, 0, 1)
    if (Bw > 0.5).sum() < 50:
        return R25._trim25(out, Bw, ((A > 0.3) & (pink < 0.3) & (lum < 0.55)).astype(np.float32), px) if trim else out
    wood = ((A > 0.3) & (pink < 0.3) & (lum < 0.55)).astype(np.float32)
    # ---- (b) per-clump violet-grey underside / inner shade
    if clumps:
        q = 4
        hq, wq = max(1, H // q), max(1, W // q)
        lx, ly = float(sun_dir[0]), float(sun_dir[1]) - 0.7
        ln = math.hypot(lx, ly) + 1e-9
        lx, ly = lx / ln, ly / ln
        Sh = np.zeros((hq, wq), np.float32)
        yq, xq = np.mgrid[0:hq, 0:wq].astype(np.float32)
        for (cx, cy, rc, p1, p2, nl) in clumps:
            cx, cy, rc = cx / q, cy / q, rc / q
            if rc < 1.0:
                continue
            x0, x1 = int(max(cx - 1.3 * rc, 0)), int(min(cx + 1.3 * rc + 1, wq))
            y0, y1 = int(max(cy - 1.3 * rc, 0)), int(min(cy + 1.3 * rc + 1, hq))
            if x1 <= x0 or y1 <= y0:
                continue
            dx = xq[y0:y1, x0:x1] - cx
            dy = yq[y0:y1, x0:x1] - cy
            th = np.arctan2(-dy, dx)
            rr = rc * (1.0 + 0.24 * np.cos(nl * th + p1) + 0.14 * np.cos((2 * nl + 1) * th + p2))
            d = np.hypot(dx, dy * 1.12) / rr
            inside = np.clip((1.05 - d) / 0.25, 0, 1)
            # away from the sun: the lower / right part of each clump (terminator broken by the lobes)
            t = -(dx * lx + dy * ly) / rr + 0.25 * np.cos(nl * th + p2)
            s_ = _ss((t - 0.05) / 0.35) * inside
            Sh[y0:y1, x0:x1] = np.maximum(Sh[y0:y1, x0:x1], s_)
        Sh = cv2.GaussianBlur(Sh, (0, 0), 0.8)
        Sh = cv2.resize(Sh, (W, H), interpolation=cv2.INTER_LINEAR)
        k = Sh * Bw * (0.5 if far else 0.62)
        # keep the value order: only darken (never lift a pixel that is already shade)
        tgt = np.minimum(C, P_SHADE[None, None] * (0.9 + 0.1 * lum[..., None]))
        C = C + (tgt - C) * k[..., None]
    # ---- (a) lit tops: merge the flower stipple into broad pale-cream shapes
    lum2 = C.mean(-1)
    Ab = (A > 0.5).astype(np.uint8)
    kc = max(3, int(round(4.0 * px)) | 1)
    Acl = cv2.morphologyEx(Ab, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    ke = max(3, int(round(4.0 * px)) | 1)
    inner = cv2.erode(Acl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke))).astype(np.float32)
    inner = cv2.GaussianBlur(inner, (0, 0), 1.0 * px)
    litm = _ss((cv2.GaussianBlur(lum2 * Bw, (0, 0), 2.5 * px) / np.maximum(cv2.GaussianBlur(Bw, (0, 0), 2.5 * px), 1e-3)
                - 0.8) / 0.08)
    zl = inner * litm * (1 - np.clip(wood * 2, 0, 1))
    zl = cv2.GaussianBlur(zl, (0, 0), 0.6 * px)
    sg = 1.8 * px
    Cp = C * A[..., None]
    Cbl = cv2.GaussianBlur(Cp, (0, 0), sg) / np.maximum(cv2.GaussianBlur(A, (0, 0), sg), 1e-3)[..., None]
    # broad painted shape: the local mean, nudged to cream where it is the brightest
    Cbl = Cbl + (P_CREAM - Cbl) * (0.35 * _ss((Cbl.mean(-1) - 0.86) / 0.06))[..., None]
    C = C + (Cbl - C) * (0.85 * zl)[..., None]
    A2 = A + (np.maximum(A, cv2.GaussianBlur(Acl.astype(np.float32), (0, 0), 0.6 * px)) - A) * zl
    out = np.dstack([C * A2[..., None], A2]).astype(np.float32)
    if trim:
        # faint sub-pixel twigs (alpha < 0.3) count as wood too, so they are trimmed with the rest
        woodT = ((A > 0.04) & (pink < 0.3) & (lum < 0.6)).astype(np.float32)
        out = R25._trim25(out, Bw, woodT, px, bridge_keep=0.15, bridge_len=3.5, limb_len=4.0,
                             term_keep=0.6, term_ramp=2.5)
    return out
