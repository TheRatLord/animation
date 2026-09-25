"""s06_seaside foliage, round 14: leaf-dab painter for the big backlit hillside tree.

A background painter's order of work, in code (no per-clump spheres anywhere):

  * silhouette: every clump is a core plus a ring of cauliflower lobes of mixed size, each big lobe carrying
    smaller lobes on its outer arc (flatter, sparser along the underside) -> a scalloped outline whose bumps
    run from ~8 to ~60 px at 1080p. Rasterised supersampled (no pixel stepping).
  * value is NOT shaded per clump. It is read off the whole crown's shape with the light coming from the
    sun (to the right of the tree): "how much foliage lies between this point and the sun" (a march along
    the light direction through the crown mask). Close to the sun-facing silhouette -> backlit warm; deep
    inside / on the far side -> one merged cool shade mass. A smaller-scale version of the same march on
    each clump's own shape adds lobe-on-lobe structure inside the lit crown only. The underside (march
    downward) sinks into violet navy, the up-facing tops away from the sun pick up a cool sky tint.
  * the paint is laid on as thousands of short directional leaf dabs (3-8 px) grouped in leaf clusters that
    share a value offset: dappled texture, dab-broken terminator, a leafy broken silhouette.
  * edges: a crisp hot rim (2-6 px) only where the silhouette faces the sun, melting into warm mid-value over
    ~30 px; the shade / underside silhouette is lost and soft. A few sky holes near the edge.
  * limbs sit BEHIND the leaves (seen in the gaps under the crown).

Writes a per-clump sway map (weight, clump id) like canopy10. Deterministic."""
import math
import numpy as np
import cv2
from scipy.spatial import cKDTree

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


SH_DEEP = cc('#131a33')      # underside (violet navy)
SH_MID = cc('#1e3244')       # shade body
SH_TOP = cc('#34525c')       # up-facing shade, sky-lit (cool teal)
HALF = cc('#3f5042')         # half tone next to the terminator
LIT_LO = cc('#687440')       # lit crown, far from the sun edge (olive, transmitted light)
LIT_MID = cc('#b58644')      # warm mid
LIT_HI = cc('#f2a255')       # hot near the sun-facing silhouette
RIM = np.array([1.5, 1.0, 0.58], np.float32)
BRANCH, BRANCH_LIT = cc('#121827'), cc('#6e4632')
LIMBS = ((0.088, 0.55, -1.35, 22.0, 7), (0.13, 0.54, -1.05, 16.0, 6), (0.165, 0.52, -1.9, 12.0, 6))


def _sm(a, b, x):
    return C.smoothstep(a, b, x)


def _shift(M, sx, sy):
    """dst(x, y) = M(x + sx, y + sy), zero outside."""
    h, w = M.shape
    return cv2.warpAffine(M, np.float32([[1, 0, sx], [0, 1, sy]]), (w, h),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=0)


def _march(M, dx, dy, n, step=1.0):
    """amount of mask met walking from each pixel along (dx, dy) (in px of M)."""
    acc = np.zeros_like(M)
    for k in range(1, n + 1):
        acc += _shift(M, dx * k * step, dy * k * step)
    return acc * step


def _lobes(rng, cx, cy, r, u, L):
    """cauliflower outline of one clump: core + ring lobes + sub-lobes (circles, 1x px)."""
    out = [(cx, cy + 0.06 * r, 0.8 * r)]
    n = max(int(2 * math.pi * r / (0.42 * r)), 7)
    th0 = rng.uniform(0, 2 * math.pi)
    for k in range(n):
        a = th0 + 2 * math.pi * (k + rng.uniform(-0.35, 0.35)) / n
        ca, sa = math.cos(a), math.sin(a)
        down = sa > 0.45
        if down and rng.random() < 0.35:
            continue
        rl = r * rng.uniform(0.16, 0.36) * (0.7 if down else 1.0)
        rl = max(rl, 4.5 * u)
        d = r - rl * rng.uniform(0.35, 0.95) - (0.08 * r if down else 0.0)
        lx, ly = cx + ca * d, cy + sa * d * (0.9 if down else 1.0)
        out.append((lx, ly, rl))
        if rl > 10 * u and not down:
            for j in range(int(rng.integers(2, 5))):
                b = a + rng.uniform(-1.0, 1.0)
                rs = max(rl * rng.uniform(0.22, 0.48), 4.0 * u)
                ds = rl - rs * rng.uniform(0.2, 0.75)
                out.append((lx + math.cos(b) * ds, ly + math.sin(b) * ds, rs))
    return out


def canopy11(cv, clumps, rng, unit=1.0, clip=None, L=(0.985, -0.1), ss=2, far_haze=0.2, branches=(), wcv=None,
             anchor=None, n_holes=9, glow=0.7, **_):
    """clumps: (x, y, r, f, clipped) canvas px, back to front; f = 0 near .. 1 far."""
    u = max(unit, 0.3)
    cl = np.array([c[:4] for c in clumps], np.float64)
    pad = 40 * u
    X0 = int(max(math.floor((cl[:, 0] - cl[:, 2] * 1.3).min() - pad), 0))
    Y0 = int(max(math.floor((cl[:, 1] - cl[:, 2] * 1.3).min() - pad), 0))
    X1 = int(min(math.ceil((cl[:, 0] + cl[:, 2] * 1.3).max() + pad), cv.W))
    Y1 = int(min(math.ceil((cl[:, 1] + cl[:, 2] * 1.3).max() + pad), cv.H))
    for br in branches:
        p = np.array(br[0])
        X0 = int(max(min(X0, p[:, 0].min() - pad), 0)); Y0 = int(max(min(Y0, p[:, 1].min() - pad), 0))
        X1 = int(min(max(X1, p[:, 0].max() + pad), cv.W)); Y1 = int(min(max(Y1, p[:, 1].max() + pad), cv.H))
    w, h = X1 - X0, Y1 - Y0
    if w < 3 or h < 3:
        return None
    S = int(ss)
    hs, ws = h * S, w * S
    Lx, Ly = L
    ln = math.hypot(Lx, Ly)
    Lx, Ly = Lx / ln, Ly / ln
    clip1 = None
    if clip is not None:
        cm, cx0, cy0 = clip
        clip1 = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            clip1[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]

    def clipv(x, y):
        if clip1 is None:
            return 0.0
        return float(clip1[int(np.clip(y, 0, h - 1)), int(np.clip(x, 0, w - 1))])

    # ---------------------------------------------------------------- A. silhouette (ss) + clump id (1x)
    MS = np.zeros((hs, ws), np.uint8)
    ID = np.full((h, w), -1, np.int32)
    kept = []                 # (index, cx, cy, r, f, sway, bbox, localmask 1x)
    SH = 4                    # cv2 subpixel shift bits
    fs = S * (1 << SH)
    for ci, (x, y, r, f, clipped) in enumerate(clumps):
        cx, cy = x - X0, y - Y0
        if clipped and clip1 is not None and clipv(cx, cy) < 0.5:
            continue
        lobes = _lobes(rng, cx, cy, r, u, (Lx, Ly))
        if clipped and clip1 is not None:
            lobes = [lb for lb in lobes if clipv(lb[0], lb[1]) > 0.1] or lobes[:1]
        la = np.array(lobes)
        bx0 = int(max(math.floor((la[:, 0] - la[:, 2]).min()) - 2, 0))
        by0 = int(max(math.floor((la[:, 1] - la[:, 2]).min()) - 2, 0))
        bx1 = int(min(math.ceil((la[:, 0] + la[:, 2]).max()) + 3, w))
        by1 = int(min(math.ceil((la[:, 1] + la[:, 2]).max()) + 3, h))
        if bx1 - bx0 < 2 or by1 - by0 < 2:
            continue
        lm = np.zeros(((by1 - by0) * S, (bx1 - bx0) * S), np.uint8)
        for (lx, ly, lr) in lobes:
            cv2.circle(lm, (int(round((lx - bx0) * fs)), int(round((ly - by0) * fs))), int(round(lr * fs)), 255,
                       -1, cv2.LINE_8, SH)
        sl = (slice(by0 * S, by1 * S), slice(bx0 * S, bx1 * S))
        np.maximum(MS[sl], lm, out=MS[sl])
        lm1 = cv2.resize(lm.astype(np.float32) / 255, (bx1 - bx0, by1 - by0), interpolation=cv2.INTER_AREA)
        ID[by0:by1, bx0:bx1][lm1 > 0.5] = ci
        kept.append((ci, cx, cy, r, f, rng.uniform(0, 1), (bx0, by0, bx1, by1), lm1))
    if not kept:
        return None
    A0 = cv2.resize(MS.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA)

    # ---------------------------------------------------------------- B. light fields (from the shape)
    q = 4
    hq, wq = max(h // q, 2), max(w // q, 2)
    Mq = cv2.resize(A0, (wq, hq), interpolation=cv2.INTER_AREA)
    dg = cv2.resize(_march(Mq, Lx, Ly, int(70 * u)), (w, h)) * q          # foliage toward the sun (px)
    du = cv2.resize(_march(Mq, 0.0, -1.0, int(25 * u)), (w, h)) * q       # foliage above
    dd = cv2.resize(_march(Mq, 0.0, 1.0, int(30 * u)), (w, h)) * q        # foliage below
    Eg = np.exp(-dg / (120 * u))              # big lit crown on the sun side of the whole tree
    Tsky = np.exp(-du / (30 * u))            # up-facing tops
    Und = np.exp(-dd / (70 * u))             # underside
    # near-edge warm band: fall-off over ~30 px from the sun-facing silhouette (1x march of the real shape)
    de = _march(A0, Lx, Ly, int(45 * u), 1.0)
    Ew = np.exp(-de / (22 * u))
    # edges facing AWAY from the sun (left / upper-left): cool and dark, even on thin sticking-out lobes
    db = _march(A0, -Lx, -Ly, int(40 * u), 1.0)
    Eb = np.exp(-db / (18 * u))
    # lobe-on-lobe structure inside the crown: each clump's own sun-facing flank (visible part only)
    El = np.zeros((h, w), np.float32)
    TL = np.zeros((h, w), np.float32)
    UL = np.zeros((h, w), np.float32)
    FH = np.zeros((h, w), np.float32)
    SW = np.zeros((h, w), np.float32)
    for (ci, cx, cy, r, f, swc, (bx0, by0, bx1, by1), lm1) in kept:
        vis = ID[by0:by1, bx0:bx1] == ci
        if not vis.any():
            continue
        dl = _march(lm1, Lx, Ly, int(min(0.6 * r, 70 * u) / 2) + 1, 2.0)
        el = np.exp(-dl / (0.28 * r))
        El[by0:by1, bx0:bx1][vis] = el[vis]
        nq = int(min(0.5 * r, 50 * u) / 2) + 1
        tl = np.exp(-_march(lm1, 0.3 * Lx, -1.0, nq, 2.0) / (0.2 * r))
        ul = np.exp(-_march(lm1, 0.0, 1.0, nq, 2.0) / (0.16 * r))
        TL[by0:by1, bx0:bx1][vis] = tl[vis]
        UL[by0:by1, bx0:bx1][vis] = ul[vis]
        FH[by0:by1, bx0:bx1][vis] = f
        SW[by0:by1, bx0:bx1][vis] = swc
    lowf = C.fbm(w, h, max(w / (160.0 * u), 1.5), 3, seed=1107) - 0.5
    lowf = cv2.GaussianBlur(np.asarray(lowf, np.float32), (0, 0), 6 * u)
    # lit amount: the whole crown's sun flank dominates; lobes only modulate inside it
    # clump-scale structure inside the shade mass: sky-lit tops / dark bottoms, fading out low in the crown
    stw = np.clip(1 - 0.75 * Und, 0, 1) * _sm(0.95, 0.2, Eg)
    V = 0.78 * Eg + 0.42 * Ew + 0.3 * (El - 0.45) * _sm(0.18, 0.55, Eg) - 0.22 * Und * (1 - Ew) - 0.35 * Eb * (1 - Ew) + 0.18 * lowf
    V = V.astype(np.float32)
    THR = 0.5

    def colour(v, tsky, und, fh, tl=None, ul=None, stw_=None):
        """value -> paint (vectorised over dabs). Two broad masses cut at THR."""
        if tl is not None:
            tsky = np.maximum(0.8 * tsky, 0.95 * _sm(0.42, 0.5, tl) * stw_)
            und = np.maximum(und, 0.5 * _sm(0.35, 0.6, ul) * stw_)
        lit = _sm(THR - 0.005, THR + 0.005, v)[..., None]
        # shade: body -> sky-lit teal where it faces up -> navy underside
        sh = SH_MID + (SH_TOP - SH_MID) * (0.75 * np.clip(tsky, 0, 1))[..., None]
        sh = sh + (SH_DEEP - sh) * (0.85 * np.clip(und, 0, 1))[..., None]
        half = _sm(THR - 0.14, THR - 0.03, v)[..., None]
        sh = sh + (HALF - sh) * (0.5 * half)
        t = np.clip((v - THR) / 0.5, 0, 1)
        lc = LIT_LO + (LIT_MID - LIT_LO) * _sm(0.05, 0.45, t)[..., None]
        lc = lc + (LIT_HI - lc) * _sm(0.45, 0.95, t)[..., None]
        col = sh * (1 - lit) + lc * lit
        col = col + (cc('#8a7ab8') - col) * (far_haze * fh)[..., None]
        return col

    # ---------------------------------------------------------------- C. leaf-dab painting (ss)
    base = colour(V, Tsky, Und, FH, TL, UL, stw).astype(np.float32)
    BASE = cv2.resize(base, (ws, hs), interpolation=cv2.INTER_LINEAR)
    CAN = BASE.copy()
    DA = np.zeros((hs, ws), np.uint8)           # dab coverage (leafy silhouette)
    r2 = np.random.default_rng(4411)
    ys_, xs_ = np.nonzero(A0 > 0.5)
    if len(ys_):
        area = float(len(ys_))
        # leaf clusters (8-20 px) sharing a value offset: mid-scale dappling
        nclu = int(area / (13 * u) ** 2 * 1.6)
        ic = r2.integers(0, len(ys_), nclu)
        cxy = np.stack([xs_[ic] + r2.uniform(0, 1, nclu), ys_[ic] + r2.uniform(0, 1, nclu)], 1)
        coff = r2.normal(0, 1, nclu).astype(np.float32)
        kd = cKDTree(cxy)
        nd = int(area / (3.9 * u) ** 2 * 1.25)
        idx = r2.integers(0, len(ys_), nd)
        px = xs_[idx] + r2.uniform(0, 1, nd)
        py = ys_[idx] + r2.uniform(0, 1, nd)
        _, near = kd.query(np.stack([px, py], 1), workers=-1)
        ix, iy = np.clip(px.astype(int), 0, w - 1), np.clip(py.astype(int), 0, h - 1)
        v = V[iy, ix] + 0.07 * coff[near] + r2.normal(0, 0.025, nd)
        # terminator dabs break the boundary a little more
        col = colour(v, Tsky[iy, ix] + 0.18 * coff[near], Und[iy, ix], FH[iy, ix], TL[iy, ix] + 0.05 * coff[near],
                     UL[iy, ix] + 0.05 * coff[near], stw[iy, ix])
        lum = 1 + 0.11 * coff[near] + r2.normal(0, 0.045, nd)
        col = col * lum[:, None]
        # a few warm flecks in the shade next to the terminator (light through gaps)
        fl = (v < THR) & (v > THR - 0.12) & (r2.random(nd) < 0.05)
        col[fl] = col[fl] * 0.4 + LIT_LO * 0.75
        rad = r2.uniform(1.8, 4.6, nd) * u
        big = r2.random(nd) < 0.18
        rad[big] *= 1.7
        el = r2.uniform(0.45, 0.8, nd)
        ang = r2.normal(20, 40, nd)
        order = np.argsort(v, kind='stable')                       # light dabs overlap the shade
        k4 = S * 4
        for i in order:
            c_ = (float(col[i, 0]), float(col[i, 1]), float(col[i, 2]))
            ctr = (int(px[i] * k4), int(py[i] * k4))
            ax = (max(int(rad[i] * k4), 2), max(int(rad[i] * el[i] * k4), 2))
            cv2.ellipse(CAN, ctr, ax, float(ang[i]), 0, 360, c_, -1, cv2.LINE_8, 2)
            cv2.ellipse(DA, ctr, ax, float(ang[i]), 0, 360, 255, -1, cv2.LINE_8, 2)
    # silhouette = slightly eroded lobe mask + the leaf dabs sticking out of it
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * S + 1, 2 * S + 1))
    AS = np.maximum(cv2.erode(MS, ke), DA).astype(np.float32) / 255
    AS *= (MS > 0) | (cv2.dilate(MS, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8 * S + 1, 8 * S + 1))) > 0)
    # sky holes near the silhouette (small leaf-edged clusters)
    Dh = cv2.distanceTransform((A0 > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    okh = (Dh > 10 * u) & (Dh < 45 * u) & (dd > 60 * u)
    if clip1 is not None:
        okh &= clip1 < 0.02
    yh, xh = np.nonzero(okh[::3, ::3])
    made, tries = [], 0
    HM = np.zeros((hs, ws), np.uint8)
    while len(made) < n_holes and tries < 4000 and len(yh):
        tries += 1
        k = int(r2.integers(0, len(yh)))
        hx_, hy_ = float(xh[k] * 3), float(yh[k] * 3)
        if any(math.hypot(hx_ - a_, hy_ - b_) < 70 * u for a_, b_ in made):
            continue
        made.append((hx_, hy_))
        hr = r2.uniform(4.0, 9.0) * u
        for _k in range(int(r2.integers(4, 8))):
            qx, qy = hx_ + r2.normal(0, hr * 0.8), hy_ + r2.normal(0, hr * 0.5)
            rr_ = hr * r2.uniform(0.25, 0.6)
            cv2.ellipse(HM, (int(qx * S * 4), int(qy * S * 4)), (int(rr_ * S * 4), int(rr_ * 0.7 * S * 4)),
                        float(r2.uniform(0, 180)), 0, 360, 255, -1, cv2.LINE_8, 2)
    AS *= 1 - HM.astype(np.float32) / 255

    # ---------------------------------------------------------------- D. down to 1x, rim + lost edges
    col1 = cv2.resize(CAN * AS[..., None], (w, h), interpolation=cv2.INTER_AREA)
    A1 = cv2.resize(AS, (w, h), interpolation=cv2.INTER_AREA)
    col1 = col1 / np.maximum(A1, 1e-5)[..., None]
    # crisp hot rim on the sun-facing edge of the real (leafy) silhouette: 2-6 px, then warm
    dr = _march(A1, Lx, Ly, 10, 0.75)
    hot = np.exp(-dr / (2.4 * u)) * A1
    facing = _sm(0.35, 0.65, Eg + 0.5 * Ew)                # only on the sun side of the crown
    hot *= facing * _sm(0.1, 0.3, V)
    col1 += (RIM - col1) * np.clip(0.9 * hot, 0, 1)[..., None]
    # lost edges on the shade / underside silhouette (away from the sun)
    Ab = cv2.GaussianBlur(A1, (0, 0), 3.0 * u)
    Ag = cv2.GaussianBlur(A1, (0, 0), 6 * u)
    gy1, gx1 = np.gradient(Ag)
    gm = np.hypot(gx1, gy1) + 1e-6
    away = _sm(0.0, 0.7, (gx1 * Lx + gy1 * Ly) / gm)            # gradient points inward: +L means edge faces away
    down = _sm(0.1, 0.8, -gy1 / gm)
    lost = np.clip(np.maximum(away, down) * _sm(0.55, 0.25, Ew), 0, 1)
    A2 = A1 * (1 - lost) + np.minimum(A1, Ab * Ab * 1.2) * lost
    col1 += (cc('#34406a') - col1) * (0.18 * lost * _sm(0.95, 0.4, Ab))[..., None]

    # ---------------------------------------------------------------- E. limbs behind the leaves
    BM = np.zeros((hs, ws), np.uint8)

    def limb(pts_w):
        for i in range(len(pts_w) - 1):
            (xa, ya, wa), (xb, yb, wb) = pts_w[i], pts_w[i + 1]
            cv2.line(BM, (int(xa * S * 4), int(ya * S * 4)), (int(xb * S * 4), int(yb * S * 4)), 255,
                     max(int(round(0.5 * (wa + wb) * S)), 1), cv2.LINE_AA, 2)

    for pts, wd in branches:
        p = np.array(pts, np.float64) - [X0, Y0]
        n = len(p)
        limb([(p[i, 0], p[i, 1], max(wd * (1 - 0.7 * i / max(n - 1, 1)), 1.0)) for i in range(n)])
    r3 = np.random.default_rng(907)
    if anchor is not None:
        aox, aoy, aW, aH = anchor
        for (fx, fy, ang0, wd0, nseg) in LIMBS:
            x_, y_ = aox + fx * aW - X0, aoy + fy * aH - Y0
            ang, wd = ang0, wd0 * u
            pts = [(x_, y_, wd)]
            for k in range(nseg):
                ang += r3.normal(0, 0.13)
                ln_ = r3.uniform(20, 30) * u
                wd *= 0.78
                pts.append((pts[-1][0] + math.cos(ang) * ln_, pts[-1][1] + math.sin(ang) * ln_, wd))
            limb(pts)
    bm = cv2.resize(BM.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA)
    bedge = np.clip(bm - _shift(bm, 1.5 * Lx * u, 1.5 * Ly * u), 0, 1)
    BR = BRANCH[None, None] + (BRANCH_LIT - BRANCH)[None, None] * (0.8 * bedge)[..., None]

    al = A2 + bm * (1 - A2)
    colr = (col1 * A2[..., None] + BR * (bm * (1 - A2))[..., None]) / np.maximum(al, 1e-5)[..., None]
    als = al
    if glow > 0:
        g = cv2.GaussianBlur(hot, (0, 0), 4 * u) * 0.8 + cv2.GaussianBlur(hot, (0, 0), 12 * u) * 1.0
        ga = np.clip(g * glow, 0, 0.4) * (1 - als)
        gc = np.array([1.0, 0.72, 0.42], np.float32)
        al2 = als + ga
        colr = (colr * als[..., None] + gc * ga[..., None]) / np.maximum(al2, 1e-5)[..., None]
        als = al2
    colr = np.clip(colr, 0, None).astype(np.float32)
    cv.put((als.astype(np.float32), X0, Y0), colr)
    if wcv is not None:
        num = cv2.GaussianBlur((SW * A1).astype(np.float32), (0, 0), 7 * u)
        den = cv2.GaussianBlur(A1.astype(np.float32), (0, 0), 7 * u)
        swm = np.clip(num / np.maximum(den, 1e-4), 0, 1)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        basew = 0.45 + 0.55 * C.smoothstep(1.0, 0.35, yy) * np.ones((1, w), np.float32)
        wt = np.clip(cv2.GaussianBlur(als.astype(np.float32), (0, 0), 3 * u) * 2.0, 0, 1) * basew
        wcv.put((wt.astype(np.float32), X0, Y0), np.dstack([swm] * 3))
    return (als, X0, Y0)
