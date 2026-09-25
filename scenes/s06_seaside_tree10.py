"""s06_seaside foliage, round 13: broad-mass painter for the big backlit hillside tree.

How a background artist blocks a backlit summer tree, in code:

  * silhouette: every clump is a few big heads, each head carries smaller heads on its upper / sun arc, and
    every head outline is built from explicit leaf-dab scallops of mixed size (dense and small on the lit arc,
    sparse and soft on the shade arc) -> irregular cauliflower outline, analytic SDF, supersampled (no pixel
    stepping).
  * value: TWO broad masses only. A continuous light term (head sphere blended with the whole crown's flank
    facing the sun, a small bulge per leaf-dab scallop) is cut ONCE by a crisp, dab-broken terminator:
    warm backlit mass (olive -> amber, hot toward the sun edge) vs cool shade mass (teal top -> violet
    underside). The shade colour is taken from a heavily blurred field, so it reads as one flat painted mass
    (no per-lobe rings, no dark contour lines).
  * texture: small leaf-dab strokes (stamped ellipses) break the terminator and add dappled low-contrast
    value variation inside both masses.
  * edges: a thin hot rim only on sun-facing edges of the lit mass, a warm glow just outside it; lost soft
    edges on the shade / underside silhouette. Limbs sit BEHIND the leaves (seen only through gaps and holes).
  * sprigs: a few twigs leave the silhouette carrying chains of overlapping leaf tufts (attached, no sprites).

Rendered at ss x and area-downsampled. Writes a per-clump sway map like canopy9. Deterministic."""
import math
import numpy as np
import cv2
from scipy.spatial import cKDTree

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


SH_DEEP = cc('#141d36')      # shade core / underside (violet navy)
SH_MID = cc('#1c3044')       # shade body
SH_TOP = cc('#2c4a58')       # sky-lit shade tops (teal)
HALF = cc('#4a5f45')
SKY2 = cc('#3a5670')          # sky-lit plane on shade-head tops         # half tone at the terminator (dabs)
LIT_LO = cc('#6c7842')       # lit mass, away from the sun edge (olive)
LIT_MID = cc('#bf8c48')      # warm
LIT_HI = cc('#f8a656')       # hot near the sun-facing edge
RIM = np.array([1.55, 1.02, 0.56], np.float32)
BRANCH, BRANCH_LIT = cc('#131a28'), cc('#7a4e36')
LIMBS = ((0.088, 0.55, -1.35, 22.0, 7), (0.13, 0.54, -1.05, 16.0, 6), (0.165, 0.52, -1.9, 12.0, 6))


def _sm(a, b, x):
    return C.smoothstep(a, b, x)


def canopy10(cv, clumps, rng, unit=1.0, clip=None, L=(0.88, -0.47), ss=3, far_haze=0.2, branches=(), wcv=None,
             anchor=None, holes=None, expo=1.0, ex_floor=0.2, n_sprigs=4, glow=0.85, **_):
    """clumps: (x, y, r, f, clipped) canvas px, back to front; f = 0 near .. 1 far."""
    u = max(unit, 0.3)
    cl = np.array([c[:4] for c in clumps], np.float64)
    pad = 50 * u
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
    S = float(ss)
    hs, ws = h * ss, w * ss
    Lx, Ly = L
    ln = math.hypot(Lx, Ly)
    Lx, Ly = Lx / ln, Ly / ln
    clipS = None
    if clip is not None:
        cm, cx0, cy0 = clip
        c_ = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            c_[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
        clipS = cv2.resize(c_, (ws, hs), interpolation=cv2.INTER_LINEAR)
    uS = u * S

    def clipv(x, y):
        if clipS is None:
            return 0.0
        return float(clipS[int(np.clip(y, 0, hs - 1)), int(np.clip(x, 0, ws - 1))])

    # ---- tree-level form (low res): which flank of the whole crown faces the sun, and its underside
    q = 4
    hq, wq = hs // q + 1, ws // q + 1
    G = np.zeros((hq, wq), np.float32)
    for (x, y, r, f, _c) in clumps:
        cv2.circle(G, (int((x - X0) * S / q), int((y - Y0) * S / q)), int(r * S / q), 1.0, -1, cv2.LINE_AA)
    gsig = max(float(np.median(cl[:, 2])) * S / q * 1.1, 3)
    pb = int(2 * gsig) + 1
    Gb = cv2.GaussianBlur(cv2.copyMakeBorder(G, pb, pb, pb, pb, cv2.BORDER_CONSTANT, value=0), (0, 0), gsig)[pb:-pb, pb:-pb]
    ggy, ggx = np.gradient(Gb)
    gn = np.hypot(ggx, ggy)
    gsc = 1.0 / (np.percentile(gn[G > 0.5], 85) + 1e-6) if (G > 0.5).any() else 1.0
    Tg_q = np.clip(-(ggx * Lx + ggy * Ly) * gsc, -1, 1)
    Td_q = np.clip(ggy * gsc, 0, 1)

    def tree_at(px, py):
        iy, ix = int(np.clip(py / q, 0, hq - 1)), int(np.clip(px / q, 0, wq - 1))
        return float(Tg_q[iy, ix]), float(Td_q[iy, ix])

    Tg = cv2.resize(Tg_q, (ws, hs), interpolation=cv2.INTER_LINEAR)[:hs, :ws]

    # ---- paint buffers (ss res)
    A = np.zeros((hs, ws), np.float32)        # coverage
    LS = np.zeros((hs, ws), np.float32)       # signed light (terminator at 0)
    LW = np.zeros((hs, ws), np.float32)       # light strength inside the lit mass (0..1)
    TOP = np.zeros((hs, ws), np.float32)      # sky-facing top of the head
    UND = np.zeros((hs, ws), np.float32)      # underside
    SW = np.zeros((hs, ws), np.float32)       # sway id
    FH = np.zeros((hs, ws), np.float32)       # far haze
    EXB = np.zeros((hs, ws), np.float32)      # exposure
    SOFT = np.zeros((hs, ws), np.float32)     # shade-side soft outline weight

    def paint_head(cx, cy, rad, ex, swc, f, leaf, parent=None, wc=0.35, arc_lo=-math.pi * 1.1, arc_hi=0.5,
                   dens=1.0, kbulge=0.16):
        """one head: core disc + explicit leaf-dab scallops round the outline (dense on the lit / upper arc)."""
        # outline dabs
        dabs = []
        per = 2 * math.pi * rad
        n = max(int(per / (leaf * 1.1) * dens), 6)
        th0 = rng.uniform(0, 2 * math.pi)
        for k in range(n):
            th = th0 + 2 * math.pi * (k + rng.uniform(-0.3, 0.3)) / n
            a = math.atan2(math.sin(th), math.cos(th))
            litarc = math.cos(a) * Lx + math.sin(a) * Ly          # +1 facing the sun
            upper = -math.sin(a)
            face = max(litarc, 0.7 * upper)
            if face < -0.25 and rng.random() < 0.2:
                continue                                         # a little sparser on the shade / underside arc
            rd = leaf * rng.uniform(0.55, 1.25) * (1.0 + 0.5 * rng.random() ** 3)
            if face > 0.2:
                rd *= rng.uniform(0.7, 1.05)
            push = rd * rng.uniform(0.15, 0.6)
            dabs.append((cx + math.cos(a) * (rad - rd + push), cy + math.sin(a) * (rad - rd + push), rd, face))
        pdg = rad + leaf * 2.2 + 3 * S
        x0 = int(max(math.floor(cx - pdg), 0)); x1 = int(min(math.ceil(cx + pdg) + 1, ws))
        y0 = int(max(math.floor(cy - pdg), 0)); y1 = int(min(math.ceil(cy + pdg) + 1, hs))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx, dy = xs - cx, ys - cy
        d0 = np.sqrt(dx * dx + dy * dy)
        core_r = rad * 0.9
        sd = core_r - d0
        # local leaf-dab bulge normal (each scallop is its own tiny clump)
        bx = np.zeros_like(sd); by = np.zeros_like(sd); bw = np.zeros_like(sd)
        for (qx, qy, rd, face) in dabs:
            a0_ = int(max(qx - rd - 2 - x0, 0)); a1_ = int(min(qx + rd + 3 - x0, x1 - x0))
            b0_ = int(max(qy - rd - 2 - y0, 0)); b1_ = int(min(qy + rd + 3 - y0, y1 - y0))
            if a1_ <= a0_ or b1_ <= b0_:
                continue
            ddx = xs[b0_:b1_, a0_:a1_] - qx
            ddy = ys[b0_:b1_, a0_:a1_] - qy
            di = np.sqrt(ddx * ddx + ddy * ddy)
            sdi = rd - di
            sl = sd[b0_:b1_, a0_:a1_]
            m = sdi > sl
            np.maximum(sl, sdi, out=sl)
            ins = m & (sdi > -1.0)
            bx[b0_:b1_, a0_:a1_][ins] = (ddx / rd)[ins]
            by[b0_:b1_, a0_:a1_][ins] = (ddy / rd)[ins]
            bw[b0_:b1_, a0_:a1_][ins] = 1.0
        cov = np.clip(sd / S + 0.5, 0, 1).astype(np.float32)
        if cov.max() <= 0:
            return
        # normal: parent head curvature + this head + small scallop bulge
        nx, ny = dx / rad, dy / rad
        dd = np.sqrt(nx * nx + ny * ny)
        s_ = np.maximum(dd, 1.0)
        nx, ny = nx / s_, ny / s_
        nz = np.sqrt(np.clip(1 - nx * nx - ny * ny, 0, 1))
        if parent is not None:
            ppx, ppy, prr = parent
            px_, py_ = (xs - ppx) / prr, (ys - ppy) / prr
            pd = np.maximum(np.sqrt(px_ * px_ + py_ * py_), 1.0)
            px_, py_ = px_ / pd, py_ / pd
            pz_ = np.sqrt(np.clip(1 - px_ * px_ - py_ * py_, 0, 1))
            nx, ny, nz = px_ + wc * nx, py_ + wc * ny, pz_ + wc * nz
        bb = np.clip(bx * bx + by * by, 0, 1)
        bz = np.sqrt(1 - bb)
        kb = kbulge * bw
        nx, ny, nz = nx + kb * bx, ny + kb * by, nz + kb * bz
        nn = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        lam = nx * Lx * 0.93 + ny * Ly * 0.93 + nz * 0.2
        # terminator position set by exposure: exposed heads lit over most of their sun half, buried heads
        # keep only a thin upper-right crescent (or none)
        thr = 0.82 - 0.92 * ex
        ls = lam - thr
        lw = np.clip((lam - thr) / (1.0 - thr + 1e-3), 0, 1)
        top = np.clip(-ny, 0, 1)
        und = _sm(0.15, 0.9, ny)
        sl = (slice(y0, y1), slice(x0, x1))
        for B, v in ((LS, ls), (LW, lw), (TOP, top), (UND, und), (SW, swc), (FH, f), (EXB, ex)):
            B[sl] = B[sl] * (1 - cov) + v * cov
        away = np.clip((dx * Lx + dy * Ly) / (d0 + 1e-3), -1, 1)
        soft = _sm(0.0, 0.7, np.maximum(-away, dy / (d0 + 1e-3))) * _sm(0.0, -0.15, ls)
        SOFT[sl] = SOFT[sl] * (1 - cov) + soft * cov
        A[sl] = A[sl] + cov * (1 - A[sl])

    def exposure(px, py, pc):
        tg, td = tree_at(px, py)
        e = float(np.clip(0.4 + 0.75 * tg + 0.35 * pc - 0.3 * td, 0, 1)) * expo
        return ex_floor + (1 - ex_floor) * e

    leaf0 = 8.0 * uS
    for ci, (x, y, r, f, clipped) in enumerate(clumps):
        R = r * S
        cx, cy = (x - X0) * S, (y - Y0) * S
        swc = rng.uniform(0.0, 1.0)
        cull = clipped and clipS is not None
        if cull and clipv(cx, cy) < 0.5:
            continue
        leaf = leaf0 * (1 - 0.3 * f)
        crowns = [(cx + rng.uniform(-0.06, 0.06) * R, cy + 0.1 * R, R * rng.uniform(0.7, 0.8))]
        nh = int(rng.integers(2, 5)) if R > 40 * uS else int(rng.integers(1, 3))
        for j in range(nh):
            a = -math.pi / 2 + rng.uniform(-1.5, 1.3)
            d = R * rng.uniform(0.3, 0.55)
            rr = R * rng.uniform(0.3, 0.5)
            crowns.append((cx + math.cos(a) * d * 1.1, cy + math.sin(a) * d * 0.85, rr))
        crowns.sort(key=lambda c: c[1] + 0.3 * c[2] + 0.5 * ((c[0] - cx) * Lx))
        for (hx, hy, hr) in crowns:
            if cull and clipv(hx, hy) < 0.1:
                continue
            pc = ((hx - cx) * Lx + (hy - cy) * Ly) / R
            ex = float(np.clip(exposure(hx, hy, pc) + rng.normal(0, 0.04), 0, 1))
            lf = leaf * float(np.clip(hr / (60 * uS), 0.85, 1.5))
            paint_head(hx, hy, hr, ex, swc, f, lf, parent=(cx, cy, R * 1.05), wc=0.55)
            # smaller heads on the upper / sun arc (cauliflower heads on heads)
            nsub = int(rng.integers(3, 7)) if hr > 30 * uS else int(rng.integers(0, 3))
            angs = np.sort(rng.uniform(-math.pi * 1.0, 0.45, nsub))
            for a in angs:
                sr = hr * rng.uniform(0.2, 0.36)
                dd = hr - sr * rng.uniform(0.35, 0.8)
                sx_, sy_ = hx + math.cos(a) * dd, hy + math.sin(a) * dd
                if cull and clipv(sx_, sy_) < 0.1:
                    continue
                pc2 = math.cos(a) * Lx + math.sin(a) * Ly
                ex2 = float(np.clip(ex + 0.05 * pc2 + rng.normal(0, 0.03), 0, 1))
                paint_head(sx_, sy_, sr, ex2, swc, f, leaf * rng.uniform(0.75, 1.0), parent=(hx, hy, hr * 1.02),
                           wc=0.18)

    # ---- sprigs: a twig leaves the upper / left silhouette carrying a chain of overlapping leaf tufts
    TW = np.zeros((hs, ws), np.uint8)
    Ab = (A > 0.5).astype(np.uint8)
    Abl = cv2.GaussianBlur(A, (0, 0), 10 * uS)
    gy_, gx_ = np.gradient(Abl)
    cnts, _ = cv2.findContours(Ab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cands = []
    for c in cnts:
        c = c[:, 0, :]
        if len(c) < 200:
            continue
        for i in range(0, len(c), max(int(12 * uS), 1)):
            px, py = int(c[i][0]), int(c[i][1])
            if not (4 < px < ws - 5 and 4 < py < hs - 5):
                continue
            g = np.array([gx_[py, px], gy_[py, px]])
            nrm = -g / (np.linalg.norm(g) + 1e-9)
            if nrm[1] > -0.25 or clipv(px, py) > 0.2:
                continue
            cands.append((px, py, nrm))
    sprig_n = 0
    used = []
    order = rng.permutation(len(cands)) if cands else []
    for k in order:
        if sprig_n >= n_sprigs:
            break
        px, py, nrm = cands[k]
        if any(math.hypot(px - a, py - b) < 90 * uS for a, b in used):
            continue
        used.append((px, py))
        sprig_n += 1
        ang = math.atan2(nrm[1], nrm[0]) + rng.normal(0, 0.3)
        ln_ = rng.uniform(16, 26) * uS
        bx0, by0 = px - math.cos(ang) * 10 * uS, py - math.sin(ang) * 10 * uS
        ex_, ey_ = px + math.cos(ang) * ln_, py + math.sin(ang) * ln_
        tx1, ty1 = px + math.cos(ang) * ln_ * 1.2, py + math.sin(ang) * ln_ * 1.2
        # tapered twig (drawn on the branch layer, behind the tufts)
        nseg = 6
        for s in range(nseg):
            t0, t1 = s / nseg, (s + 1) / nseg
            wd = max((2.4 - 1.7 * t0) * uS, 1.0)
            cv2.line(TW, (int((bx0 + (tx1 - bx0) * t0) * 4), int((by0 + (ty1 - by0) * t0) * 4)),
                     (int((bx0 + (tx1 - bx0) * t1) * 4), int((by0 + (ty1 - by0) * t1) * 4)), 255,
                     int(round(wd)), cv2.LINE_AA, 2)
        tg, _td = tree_at(px, py)
        ex = float(np.clip(0.6 + 0.5 * tg, 0.35, 1))
        swc = float(SW[py, px])
        # tufts: overlapping, shrinking toward the tip, the first one sits on the silhouette (attached)
        # leaves alternate along the twig: small tufts, overlapping the silhouette at the base
        nt = int(rng.integers(4, 7))
        mx_, my_ = px + (ex_ - px) * 0.5, py + (ey_ - py) * 0.5
        for j in range(nt):
            tt = 0.9 * j / max(nt - 1, 1)
            rr = (6.0 - 2.6 * tt) * uS * rng.uniform(0.85, 1.15)
            side = (1 if j % 2 else -1) * rng.uniform(1.5, 3.5) * uS * (0.4 + tt)
            tx_ = px + (ex_ - px) * tt - math.sin(ang) * side
            ty_ = py + (ey_ - py) * tt + math.cos(ang) * side
            paint_head(tx_, ty_, rr, 1.5, swc, 0.0, 2.6 * uS, parent=(mx_, my_, ln_ * 1.2), wc=0.2,
                       dens=1.3, kbulge=0.08)

    # ---- small leaf-dab strokes (stamped ellipses) -> dappled texture + dab-broken terminator
    def dab_field(n_per, rmin, rmax, seed, elong=(0.5, 0.85)):
        r_ = np.random.default_rng(seed)
        Dm = np.zeros((hs, ws), np.float32)
        ys_, xs_ = np.nonzero(A[::8, ::8] > 0.3)
        if len(ys_) == 0:
            return Dm
        area = len(ys_) * 64.0
        n = int(area / ((rmin + rmax) * 0.5 * uS) ** 2 * n_per)
        idx = r_.integers(0, len(ys_), n)
        cxs = xs_[idx] * 8 + r_.uniform(0, 8, n)
        cys = ys_[idx] * 8 + r_.uniform(0, 8, n)
        vals = r_.uniform(-1, 1, n).astype(np.float32)
        rad = r_.uniform(rmin, rmax, n) * uS
        el = r_.uniform(elong[0], elong[1], n)
        an = r_.normal(15, 35, n)
        for i in range(n):
            cv2.ellipse(Dm, (int(cxs[i] * 4), int(cys[i] * 4)), (int(rad[i] * 4), int(rad[i] * el[i] * 4)),
                        float(an[i]), 0, 360, float(vals[i]), -1, cv2.LINE_8, 2)
        return Dm

    Dsmall = dab_field(1.4, 3.0, 6.0, 811)
    Dbig = dab_field(1.1, 8.0, 15.0, 812, elong=(0.55, 0.9))

    # ---- value: one crisp dab-broken terminator -> two broad masses
    LSp = LS + 0.07 * Dbig + 0.03 * Dsmall
    lit = _sm(-0.018, 0.018, LSp)
    # close thin shade slivers inside the lit mass (no dark contour lines between lit heads)
    kc = max(int(3.0 * uS), 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kc + 1, 2 * kc + 1))
    lit = np.maximum(lit, cv2.erode(cv2.dilate(lit, ker), ker) * _sm(0.3, 0.6, EXB))
    # ... and drop thin lit slivers of the heads behind (no hairline lit strips in the shade)
    ko = max(int(3.2 * uS), 1)
    ker2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ko + 1, 2 * ko + 1))
    lit = np.minimum(lit, cv2.GaussianBlur(cv2.dilate(cv2.erode(lit, ker2), ker2), (0, 0), 0.6 * uS))
    # half-tone dabs hugging the terminator on the shade side (a few, painted)
    half = _sm(-0.16, -0.05, LS + 0.1 * Dsmall) * (1 - lit) * _sm(0.2, 0.6, Dsmall + 0.4 * Dbig)
    # shade colour from broad (alpha-normalised, blurred) fields: one painted mass, no per-lobe rings
    def nblur(F_, sig):
        return cv2.GaussianBlur(F_ * A, (0, 0), sig) / np.maximum(cv2.GaussianBlur(A, (0, 0), sig), 1e-4)
    topb = nblur(TOP, 14 * uS)
    undb = nblur(UND, 22 * uS)
    tgb = Tg
    shade = SH_MID[None, None] + (SH_TOP - SH_MID)[None, None] * (np.clip(0.9 * topb + 0.35 * tgb, 0, 1) * 0.8)[..., None]
    shade += (SH_DEEP - shade) * (np.clip(0.8 * undb - 0.3 * tgb, 0, 1) * 0.85)[..., None]
    shade *= (1 + 0.05 * Dsmall + 0.035 * Dbig)[..., None]            # dappled leaf strokes (low contrast)
    shade += (HALF - shade) * (0.55 * half)[..., None]
    # sky-lit tops of the shade heads: ONE crisp, dab-broken lighter plane (cauliflower structure in the shade)
    skyq = _sm(-0.03, 0.03, TOP - 0.6 + 0.16 * Dbig + 0.06 * Dsmall) * (1 - lit) * _sm(0.9, 0.3, undb)
    shade += (SKY2 - shade) * (0.26 * skyq)[..., None]
    # sun-flecks: small warm leaf clusters catching light through gaps, near the terminator only
    fleck = _sm(0.55, 0.75, Dsmall) * _sm(-0.3, -0.08, LS + 0.08 * Dbig) * (1 - lit) * _sm(0.35, 0.6, EXB)
    # lit mass: olive -> warm -> hot toward the sun edge
    lwb = nblur(LW, 3 * uS)
    lv = np.clip(1.1 * lwb + 0.25 * np.clip(tgb, 0, 1) + 0.06 * Dbig + 0.04 * Dsmall - 0.05, 0, 1)
    litc = LIT_LO[None, None] + (LIT_MID - LIT_LO)[None, None] * _sm(0.1, 0.55, lv)[..., None]
    litc += (LIT_HI - litc) * _sm(0.55, 0.95, lv)[..., None]
    litc *= (1 + 0.06 * Dsmall + 0.05 * Dbig)[..., None]
    col = shade * (1 - lit[..., None]) + litc * lit[..., None]
    col += (LIT_LO * 0.9 - col) * (0.8 * fleck)[..., None]
    # ---- thin hot rim on the sun-facing edges of the lit mass (silhouette + fronts over darker clumps)
    litA = lit * A
    sh = np.float32([[1, 0, -Lx * 2.4 * uS], [0, 1, -Ly * 2.4 * uS]])
    # neighbour toward the sun (sample at +L): pixels whose sun-side neighbour is not lit
    nb = cv2.warpAffine(litA, np.float32([[1, 0, Lx * 2.4 * uS], [0, 1, Ly * 2.4 * uS]]), (ws, hs),
                        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    rim = np.clip(litA - nb, 0, 1) * _sm(0.3, 0.7, EXB)
    rim = cv2.GaussianBlur(rim, (0, 0), 0.5 * uS)
    col += (RIM - col) * np.clip(0.9 * rim, 0, 1)[..., None]
    del sh
    # far clumps: aerial haze
    col += (cc('#8a7ab8') - col) * (far_haze * FH)[..., None]

    # ---- lost edges on the shade / underside silhouette
    softA = cv2.GaussianBlur(A, (0, 0), 3.2 * uS)
    Ag = cv2.GaussianBlur(A, (0, 0), 5 * uS)
    gy1, gx1 = np.gradient(Ag)
    gm = np.hypot(gx1, gy1) + 1e-6
    awayg = _sm(0.2, 0.8, (gx1 * Lx + gy1 * Ly) / gm)
    dn = np.maximum(_sm(0.3, 0.9, gy1 / gm), 0.8 * awayg) * (1 - lit)
    dn = np.maximum(dn, 0.6 * nblur(SOFT, 2 * uS) * (1 - lit) * _sm(0.95, 0.5, Ag))
    A2 = A * (1 - dn) + np.minimum(A, softA * softA * 1.25) * dn
    # the soft outer fringe takes a little sky colour (air between the leaves)
    col += (cc('#3a4470') - col) * (0.35 * dn * _sm(0.9, 0.3, softA))[..., None]

    # ---- branches / limbs BEHIND the leaves (visible only in gaps and sky holes), tapered
    BR = np.zeros((hs, ws, 3), np.float32)
    BA = np.zeros((hs, ws), np.float32)

    def limb(pts_w, into):
        for i in range(len(pts_w) - 1):
            (xa, ya, wa), (xb, yb, wb) = pts_w[i], pts_w[i + 1]
            cv2.line(into, (int(xa * 4), int(ya * 4)), (int(xb * 4), int(yb * 4)), 255,
                     max(int(round(0.5 * (wa + wb))), 1), cv2.LINE_AA, 2)

    BM = np.zeros((hs, ws), np.uint8)
    for pts, wd in branches:
        p = (np.array(pts, np.float64) - [X0, Y0]) * S
        n = len(p)
        limb([(p[i, 0], p[i, 1], max(wd * S * (1 - 0.7 * i / max(n - 1, 1)), 1.0)) for i in range(n)], BM)
    r2 = np.random.default_rng(907)
    if anchor is not None:
        aox, aoy, aW, aH = anchor
        for (fx, fy, ang0, wd0, nseg) in LIMBS:
            x_, y_ = (aox + fx * aW - X0) * S, (aoy + fy * aH - Y0) * S
            ang, wd = ang0, wd0 * uS
            pts = [(x_, y_, wd)]
            for k in range(nseg):
                ang += r2.normal(0, 0.13)
                ln_ = r2.uniform(20, 30) * uS
                wd *= 0.78
                pts.append((pts[-1][0] + math.cos(ang) * ln_, pts[-1][1] + math.sin(ang) * ln_, wd))
            limb(pts, BM)
    bm = np.maximum(BM.astype(np.float32) / 255, TW.astype(np.float32) / 255)
    bedge = np.clip(bm - cv2.warpAffine(bm, np.float32([[1, 0, -Lx * 1.5 * uS], [0, 1, -Ly * 1.5 * uS]]), (ws, hs)), 0, 1)
    BR[:] = BRANCH[None, None] + (BRANCH_LIT - BRANCH)[None, None] * (0.8 * bedge)[..., None]
    BA[:] = bm

    # ---- a few sky holes through the shade near the edges (only where sky is behind), a twig crossing
    Dh = cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    okh = (Dh > 16 * uS) & (Dh < 80 * uS) & (lit < 0.05)
    if clipS is not None:
        okh &= clipS < 0.02
    yh, xh = np.nonzero(okh[::4, ::4])
    HM = np.zeros((hs, ws), np.float32)
    nh_ = 6 if holes is None else holes
    made, tries = [], 0
    while len(made) < nh_ and tries < 3000 and len(yh):
        tries += 1
        k = int(r2.integers(0, len(yh)))
        hx_, hy_ = float(xh[k] * 4), float(yh[k] * 4)
        if any(math.hypot(hx_ - a_, hy_ - b_) < 80 * uS for a_, b_ in made):
            continue
        made.append((hx_, hy_))
        hr = r2.uniform(7.0, 13.0) * uS
        for _k in range(int(r2.integers(3, 6))):
            cx_, cy_ = hx_ + r2.normal(0, hr * 0.7), hy_ + r2.normal(0, hr * 0.45)
            rr_ = hr * r2.uniform(0.35, 0.75)
            cv2.circle(HM, (int(cx_ * 4), int(cy_ * 4)), int(rr_ * 4), 1.0, -1, cv2.LINE_8, 2)
        if r2.random() < 0.7:
            a_ = r2.uniform(-2.4, -0.7)
            l_ = hr * 2.4
            m = np.zeros((hs, ws), np.uint8)
            cv2.line(m, (int((hx_ - math.cos(a_) * l_) * 4), int((hy_ - math.sin(a_) * l_) * 4)),
                     (int((hx_ + math.cos(a_) * l_) * 4), int((hy_ + math.sin(a_) * l_) * 4)), 255,
                     max(int(round(1.5 * uS)), 1), cv2.LINE_AA, 2)
            mf = m.astype(np.float32) / 255
            BR[:] = BR * (1 - mf[..., None]) + BRANCH[None, None] * mf[..., None]
            BA[:] = np.maximum(BA, mf)
    A2 = A2 * (1 - HM)

    # ---- composite over the branches, downsample (area) -> canvas
    al = A2 + BA * (1 - A2)
    prem = col * A2[..., None] + BR * (BA * (1 - A2))[..., None]
    small = cv2.resize(np.dstack([prem, al]).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    als = small[..., 3]
    colr = small[..., :3] / np.maximum(als, 1e-5)[..., None]
    if glow > 0:
        rs = cv2.resize(np.ascontiguousarray(rim * A), (w, h), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(rs, (0, 0), 4 * u) * 0.7 + cv2.GaussianBlur(rs, (0, 0), 12 * u) * 0.9
        ga = np.clip(g * glow, 0, 0.45) * (1 - als)
        gc = np.array([1.0, 0.72, 0.42], np.float32)
        al2 = als + ga
        colr = (colr * als[..., None] + gc * ga[..., None]) / np.maximum(al2, 1e-5)[..., None]
        als = al2
    cv.put((als, X0, Y0), colr)
    if wcv is not None:
        a0 = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
        sws = cv2.resize(SW * A, (w, h), interpolation=cv2.INTER_AREA)
        num = cv2.GaussianBlur(sws.astype(np.float32), (0, 0), 7 * u)
        den = cv2.GaussianBlur(a0.astype(np.float32), (0, 0), 7 * u)
        swm = np.clip(num / np.maximum(den, 1e-4), 0, 1)
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        base = 0.45 + 0.55 * C.smoothstep(1.0, 0.35, yy) * np.ones((1, w), np.float32)
        wt = np.clip(cv2.GaussianBlur(als.astype(np.float32), (0, 0), 3 * u) * 2.0, 0, 1) * base
        wcv.put((wt, X0, Y0), np.dstack([swm] * 3))
    return (als, X0, Y0)
