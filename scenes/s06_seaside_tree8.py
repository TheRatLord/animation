"""s06_seaside foliage, round 11: analytic cauliflower-lobe painter for the big hillside tree.

Every clump is built as a hierarchy of anti-aliased lobes, like a background artist blocks in a tree:

  clump -> 3-6 crown lobes (big heads) -> 4-9 sub-lobes riding on each crown's upper / sun arc
        -> serrated leaf-tuft scallops on every lobe outline (analytic radius r(theta), no pixel stepping).

Each lobe is shaded as a painted sphere lit by the low sun on the right / upper right: a sun-facing lambert
term blended with the lobe's place in the whole crown (tree-level flank), plus a sky-lit top. The continuous
value is broken up by a leaf-cluster dome field (small + medium Worley domes) and then softly quantised into
a few painted value planes, so every terminator follows the leaf clusters (dappled, scalloped) rather than a
vector arc. Colour ramp: saturated cool violet core -> teal-olive half tone -> olive body -> ochre ->
amber / orange-gold rim; the undersides sink to violet with lost (soft) edges on the shade side, the
sun-facing edges get a thin warm glow. Lobes cast a soft contact shadow on what is behind them (depth).
Sky holes are punched near the shade-side edges and a few leaf sprigs break the outline. Everything is
rendered at 2x and area-downsampled. Also writes a per-clump sway map (weight, clump id). Deterministic."""
import math
import numpy as np
import cv2
from scipy.spatial import cKDTree

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


# painted value planes (index -> colour); the continuous value is quantised onto these with soft steps
PLANES = ['#241e5c', '#312e78', '#3c4482', '#3d5a70', '#58684a', '#80783e', '#b88a40', '#eaa654', '#ffcc88']
STEPS = [0.14, 0.26, 0.37, 0.47, 0.57, 0.67, 0.77, 0.88]
SHD_LOW = cc('#2a1f66')        # violet core toward the underside
SKY_LIT = cc('#5156a0')        # sky-lit blue-violet on the upper shade tops
RIM = np.array([1.5, 0.98, 0.52], np.float32)
BRANCH, BRANCH_LIT = cc('#1c1d3a'), cc('#8a5a3a')


def _planes():
    return np.array([cc(h) for h in PLANES], np.float32)


def _leaf_field(ws, hs, cs, seed):
    """Worley dome field: one rounded dome per leaf cluster (cell size cs px) -> 0..1 (1 at the dome tops)."""
    r = np.random.default_rng(seed)
    gx = np.arange(-1, ws / cs + 2)
    gy = np.arange(-1, hs / cs + 2)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([(X + r.uniform(0.1, 0.9, X.shape)).ravel() * cs,
                    (Y + r.uniform(0.1, 0.9, Y.shape)).ravel() * cs], 1)
    rad = r.uniform(0.75, 1.15, len(pts)) * cs * 0.72
    tr = cKDTree(pts)
    ys, xs = np.mgrid[0:hs, 0:ws].astype(np.float32)
    q = np.stack([xs.ravel(), ys.ravel()], 1)
    d, i = tr.query(q, k=1, workers=-1)
    dd = (d / rad[i]).reshape(hs, ws).astype(np.float32)
    return np.sqrt(np.clip(1 - dd * dd, 0, 1))


def _lobe_geom(xs, ys, cx, cy, rad, leaf, phi, dep, rng_amp, ph2):
    """distance to a serrated lobe outline (positive inside, in px) + local unit coords."""
    dx, dy = xs - cx, ys - cy
    dist = np.sqrt(dx * dx + dy * dy)
    th = np.arctan2(dy, dx)
    n = max(int(round(2 * math.pi * rad / leaf)), 5)
    sc = np.abs(np.cos(n * th * 0.5 + phi)) ** 0.55
    amp = dep * (1 + rng_amp * np.sin(3 * th + ph2))
    rr = rad + amp * (sc - 0.75)
    return rr - dist, dx / rad, dy / rad, rr


def canopy8(cv, clumps, rng, unit=1.0, clip=None, L=(0.88, -0.47), ss=2, far_haze=0.2, branches=(), wcv=None,
            **_):
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
    # ---- tree-level form: blurred union of the clumps -> which flank of the whole crown faces the sun
    G = np.zeros((hs, ws), np.float32)
    for (x, y, r, f, _c) in clumps:
        cv2.ellipse(G, (int((x - X0) * S * 4), int((y - Y0) * S * 4)), (int(r * S * 1.0 * 4), int(r * S * 0.9 * 4)),
                    0, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
    gsig = max(float(np.median(cl[:, 2])) * S * 1.2, 4)
    pb = int(2 * gsig) + 1
    Gb = cv2.GaussianBlur(cv2.copyMakeBorder(G, pb, pb, pb, pb, cv2.BORDER_CONSTANT, value=0), (0, 0), gsig)[pb:-pb, pb:-pb]
    ggy, ggx = np.gradient(Gb)
    gn = np.hypot(ggx, ggy)
    gsc = 1.0 / (np.percentile(gn[G > 0.5], 85) + 1e-6) if (G > 0.5).any() else 1.0
    Tg = np.clip(-(ggx * Lx + ggy * Ly) * gsc, -1, 1)          # +1: crown flank facing the sun
    Tdown = np.clip(ggy * gsc, 0, 1)                            # underside of the whole crown

    # ---- branches (dark, a warm lit edge on the sun side); they show through gaps / sky holes
    BR = np.zeros((hs, ws, 3), np.float32)
    BA = np.zeros((hs, ws), np.float32)
    for pts, wd in branches:
        p = (np.array(pts, np.float64) - [X0, Y0]) * S
        n = len(p)
        m = np.zeros((hs, ws), np.uint8)
        for i in range(n - 1):
            th = max(wd * (1 - 0.75 * i / max(n - 1, 1)) * S, 1.0)
            cv2.line(m, (int(p[i, 0] * 4), int(p[i, 1] * 4)), (int(p[i + 1, 0] * 4), int(p[i + 1, 1] * 4)), 255,
                     int(round(th)), cv2.LINE_AA, 2)
        mf = m.astype(np.float32) / 255
        sh = np.float32([[1, 0, -Lx * 1.5 * uS], [0, 1, -Ly * 1.5 * uS]])
        edge = np.clip(mf - cv2.warpAffine(mf, sh, (ws, hs)), 0, 1) * 0.7
        col = BRANCH[None, None] + (BRANCH_LIT - BRANCH)[None, None] * edge[..., None]
        BR[:] = BR * (1 - mf[..., None]) + col * mf[..., None]
        BA[:] = np.maximum(BA, mf)

    # ---- paint buffers
    V = np.zeros((hs, ws), np.float32)        # painted value (continuous, 0 shade .. 1 sun)
    A = np.zeros((hs, ws), np.float32)        # coverage
    U = np.zeros((hs, ws), np.float32)        # underside / shade-core weight
    SW = np.zeros((hs, ws), np.float32)       # sway id
    FH = np.zeros((hs, ws), np.float32)       # far haze weight
    TP = np.zeros((hs, ws), np.float32)       # sky-facing top of each lobe (shade lobes read by it)
    EX = np.zeros((hs, ws), np.float32)       # exposure of the lobe (for rim / leaf breakup strength)

    def paint_lobe(cx, cy, rad, ex, swc, f, leaf, sky=0.0, depth_sh=0.3, lit_gain=1.0, parent=None, wc=0.6):
        pdg = rad * 1.35 + 4 * uS
        x0 = int(max(math.floor(cx - pdg), 0)); x1 = int(min(math.ceil(cx + pdg) + 1, ws))
        y0 = int(max(math.floor(cy - pdg), 0)); y1 = int(min(math.ceil(cy + pdg) + 1, hs))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dep = min(max(leaf * 0.42, 1.2 * uS), rad * 0.3)
        sd, nx, ny, rr = _lobe_geom(xs, ys, cx, cy, rad, leaf, rng.uniform(0, 6.28), dep, 0.5, rng.uniform(0, 6.28))
        cov = np.clip(sd / 1.1 + 0.5, 0, 1).astype(np.float32)
        # ---- contact shadow cast on what is already painted behind (down / away from the sun)
        if depth_sh > 0:
            ox_, oy_ = -Lx * 0.16 * rad, (-Ly * 0.16 + 0.12) * rad
            dxs, dys = xs - cx - ox_, ys - cy - oy_
            dsh = np.sqrt(dxs * dxs + dys * dys)
            shd = np.clip((rad * 1.12 - dsh) / (0.3 * rad + 1e-3), 0, 1) * (1 - cov)
            sl = V[y0:y1, x0:x1]
            sl *= 1 - depth_sh * shd
            U[y0:y1, x0:x1] = np.maximum(U[y0:y1, x0:x1], 0.6 * shd * A[y0:y1, x0:x1])
        # ---- sphere shading
        d2 = np.clip(nx * nx + ny * ny, 0, 1)
        nz = np.sqrt(1 - d2)
        if parent is not None:
            # cauliflower normal: the parent head's curvature + this lobe's own bulge
            ppx, ppy, prr = parent
            qx, qy = (xs - ppx) / prr, (ys - ppy) / prr
            qd = np.sqrt(qx * qx + qy * qy)
            qs = np.maximum(qd, 1.0)
            qx, qy = qx / qs, qy / qs
            qz = np.sqrt(np.clip(1 - qx * qx - qy * qy, 0, 1))
            mx_, my_, mz_ = qx + wc * nx, qy + wc * ny, qz + wc * nz
            mn = np.sqrt(mx_ * mx_ + my_ * my_ + mz_ * mz_) + 1e-6
            nx, ny, nz = mx_ / mn, my_ / mn, mz_ / mn
        # sun low on the right, a little above: mostly side / back light (small z) -> lit crescent on the right
        lam = nx * Lx * 0.92 + ny * Ly * 0.92 + nz * 0.22
        lam = np.clip(lam, -1, 1)
        top = np.clip(-ny, 0, 1) * nz                                # sky-lit top of every lobe
        # value: exposure sets how much of the sun crescent this lobe gets
        v = (0.13 + 0.12 * ex + 0.07 * sky * top
             + (0.08 + 0.72 * ex) * lit_gain * C.smoothstep(0.0, 0.85, lam) + 0.05 * top)
        und = C.smoothstep(0.1, 0.95, ny) * (1 - 0.5 * ex)
        v = v - 0.07 * und
        sl = (slice(y0, y1), slice(x0, x1))
        V[sl] = V[sl] * (1 - cov) + v * cov
        U[sl] = U[sl] * (1 - cov) + und * cov
        TP[sl] = TP[sl] * (1 - cov) + top * cov
        SW[sl] = SW[sl] * (1 - cov) + swc * cov
        FH[sl] = FH[sl] * (1 - cov) + f * cov
        EX[sl] = EX[sl] * (1 - cov) + ex * cov
        A[sl] = A[sl] + cov * (1 - A[sl])

    def exposure(px, py, pc):
        iy, ix = int(np.clip(py, 0, hs - 1)), int(np.clip(px, 0, ws - 1))
        tg, td = float(Tg[iy, ix]), float(Tdown[iy, ix])
        return float(np.clip(0.42 + 0.75 * tg + 0.35 * pc - 0.3 * td, 0, 1)), iy, ix

    leaf0 = 9.0 * uS
    for ci, (x, y, r, f, clipped) in enumerate(clumps):
        R = r * S
        cx, cy = (x - X0) * S, (y - Y0) * S
        swc = rng.uniform(0.0, 1.0)
        cull = clipped and clipS is not None
        if cull and clipS[int(np.clip(cy, 0, hs - 1)), int(np.clip(cx, 0, ws - 1))] < 0.5:
            continue
        leaf = leaf0 * (1 - 0.35 * f)
        # crown lobes: a big core + heads on its upper / sun arc (varied scale)
        crowns = [(cx + rng.uniform(-0.06, 0.06) * R, cy + 0.1 * R, R * rng.uniform(0.68, 0.78))]
        nh = int(rng.integers(2, 5)) if R > 40 * uS else int(rng.integers(1, 3))
        for j in range(nh):
            a = -math.pi / 2 + rng.uniform(-1.5, 1.3)
            d = R * rng.uniform(0.3, 0.55)
            rr = R * rng.uniform(0.3, 0.52)
            crowns.append((cx + math.cos(a) * d * 1.1, cy + math.sin(a) * d * 0.85, rr))
        crowns.sort(key=lambda c: c[1] + 0.3 * c[2] + 0.5 * ((c[0] - cx) * Lx))
        for (hx, hy, hr) in crowns:
            if cull and clipS[int(np.clip(hy, 0, hs - 1)), int(np.clip(hx, 0, ws - 1))] < 0.1:
                continue
            pc = ((hx - cx) * Lx + (hy - cy) * Ly) / R
            ex, iy, ix = exposure(hx, hy, pc)
            ex = float(np.clip(ex + rng.normal(0, 0.05), 0, 1))
            paint_lobe(hx, hy, hr, ex, swc, f, leaf * 1.3, sky=1.0, depth_sh=0.4, parent=(cx, cy, R * 1.05),
                       wc=0.9)
            # sub-lobes on the upper / sun arc of the crown (cauliflower heads on heads, varied scale)
            nsub = int(rng.integers(4, 9)) if hr > 25 * uS else int(rng.integers(1, 4))
            angs = np.sort(rng.uniform(-math.pi * 1.05, 0.55, nsub))
            for a in angs:
                sr = hr * rng.uniform(0.14, 0.3) * (1.25 if rng.random() < 0.25 else 1.0)
                dd = hr - sr * rng.uniform(0.55, 1.0)
                sx_, sy_ = hx + math.cos(a) * dd, hy + math.sin(a) * dd
                if cull and clipS[int(np.clip(sy_, 0, hs - 1)), int(np.clip(sx_, 0, ws - 1))] < 0.1:
                    continue
                pc2 = math.cos(a) * Lx + math.sin(a) * Ly
                ex2 = float(np.clip(ex + 0.06 * pc2 + rng.normal(0, 0.04), 0, 1))
                paint_lobe(sx_, sy_, sr, ex2, swc, f, leaf, sky=0.6, depth_sh=0.12, parent=(hx, hy, hr), wc=0.55)
                # a few tiny heads on the sun-facing sub-lobes
                if sr > 14 * uS and pc2 > 0.1 and rng.random() < 0.6:
                    for _k in range(int(rng.integers(1, 3))):
                        a2 = a + rng.normal(0, 0.5)
                        tr = sr * rng.uniform(0.3, 0.45)
                        paint_lobe(sx_ + math.cos(a2) * (sr - tr * 0.6), sy_ + math.sin(a2) * (sr - tr * 0.6), tr,
                                   float(np.clip(ex2 + 0.08, 0, 1)), swc, f, leaf * 0.8, sky=0.4, depth_sh=0.08,
                                   parent=(sx_, sy_, sr), wc=0.5)

    # ---- leaf sprigs breaking the outline (short twig + a few small leaf tufts), mostly on top / sun side
    Ab = (A > 0.5).astype(np.uint8)
    Abl = cv2.GaussianBlur(A, (0, 0), 8 * uS)
    gy_, gx_ = np.gradient(Abl)
    cnts, _ = cv2.findContours(Ab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tw = np.zeros((hs, ws), np.uint8)
    sprigs = []
    for c in cnts:
        c = c[:, 0, :]
        if len(c) < 60:
            continue
        i = int(rng.integers(0, 40))
        while i < len(c):
            px, py = int(c[i][0]), int(c[i][1])
            g = np.array([gx_[py, px], gy_[py, px]])
            nrm = -g / (np.linalg.norm(g) + 1e-9)
            ok = nrm[1] < 0.2 and 2 < px < ws - 3 and 2 < py < hs - 3
            if ok and clipS is not None and clipS[py, px] > 0.3 and nrm[1] > -0.5:
                ok = False
            if ok and rng.random() < 0.55:
                ln_ = rng.uniform(8, 18) * uS
                ang = math.atan2(nrm[1], nrm[0]) + rng.normal(0, 0.35)
                ex_ = px + math.cos(ang) * ln_, py + math.sin(ang) * ln_
                sprigs.append((px, py, ex_[0], ex_[1], ang, float(V[py, px]), float(SW[py, px]),
                               float(FH[py, px]), float(EX[py, px])))
            i += int(rng.uniform(70, 150) * uS)
    for (px, py, qx, qy, ang, vv, swc, f, ex) in sprigs:
        # twig
        bx, by = px - math.cos(ang) * 8 * uS, py - math.sin(ang) * 8 * uS
        cv2.line(tw, (int(bx * 4), int(by * 4)), (int(qx * 4), int(qy * 4)), 255, max(int(1.3 * uS), 1),
                 cv2.LINE_AA, 2)
        nl = int(rng.integers(4, 8))
        for k in range(nl):
            tt = rng.uniform(0.2, 1.0)
            lx_ = px + (qx - px) * tt + rng.normal(0, 3 * uS)
            ly_ = py + (qy - py) * tt + rng.normal(0, 3 * uS)
            exk = float(np.clip(ex + 0.1, 0, 1))
            paint_lobe(lx_, ly_, rng.uniform(2.5, 5.0) * uS, exk, swc, f, 4.5 * uS, sky=0.5, depth_sh=0.0)
    twf = tw.astype(np.float32) / 255
    BR[:] = BR * (1 - twf[..., None]) + BRANCH[None, None] * twf[..., None]
    BA[:] = np.maximum(BA, twf)

    # ---- sky holes near the shade-side / upper edges (only where sky is behind the crown)
    k1, k2 = max(int(14 * uS), 2), max(int(90 * uS), 4)
    Ab = (A > 0.6).astype(np.uint8)
    ring = cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1,) * 2)) - \
        cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1,) * 2))
    ys_, xs_ = np.nonzero(ring > 0)
    HM = np.zeros((hs, ws), np.float32)
    made, tries = 0, 0
    while made < 22 and tries < 2000 and len(ys_):
        tries += 1
        k = int(rng.integers(0, len(ys_)))
        hx_, hy_ = xs_[k], ys_[k]
        if V[hy_, hx_] > 0.45:
            continue
        if clipS is not None and clipS[hy_, hx_] > 0.02:
            continue
        hr = rng.uniform(3.0, 7.0) * uS
        for _k in range(int(rng.integers(2, 5))):
            cx_, cy_ = hx_ + rng.normal(0, hr * 0.7), hy_ + rng.normal(0, hr * 0.7)
            x0 = int(max(cx_ - hr - 3, 0)); x1 = int(min(cx_ + hr + 4, ws))
            y0 = int(max(cy_ - hr - 3, 0)); y1 = int(min(cy_ + hr + 4, hs))
            if x1 <= x0 or y1 <= y0:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            sd, _a, _b, _c = _lobe_geom(xx, yy, cx_, cy_, hr * rng.uniform(0.5, 1.0), 3.0 * uS, rng.uniform(0, 6),
                                        0.8 * uS, 0.3, rng.uniform(0, 6))
            HM[y0:y1, x0:x1] = np.maximum(HM[y0:y1, x0:x1], np.clip(sd / 1.1 + 0.5, 0, 1))
        made += 1

    # ---- value -> painted planes. Leaf-cluster domes perturb the value before quantising, so the
    # terminators follow leaf clusters (scalloped / dappled) rather than smooth arcs.
    lf = 0.45 * _leaf_field(ws, hs, 11.0 * uS, 811) + 0.55 * _leaf_field(ws, hs, 26.0 * uS, 812)
    Vp = V + (lf - 0.55) * (0.018 + 0.16 * np.exp(-((V - 0.52) / 0.16) ** 2))
    # low-frequency brush variation (painted, no pixel noise)
    nz = C.fbm(ws, hs, max(int(ws / (70 * uS)), 2), 3, seed=611) - 0.5
    Vp = Vp + nz * 0.06
    pl = _planes()
    q = np.zeros_like(Vp)
    wq = 0.012
    for s_ in STEPS:
        q += C.smoothstep(s_ - wq, s_ + wq, Vp)
    qi = np.clip(q, 0, len(PLANES) - 1)
    i0 = np.floor(qi).astype(np.int32)
    i1 = np.minimum(i0 + 1, len(PLANES) - 1)
    fr = (qi - i0)[..., None]
    col = pl[i0] * (1 - fr) + pl[i1] * fr
    # a little continuous gradation inside each plane (painted, not flat vector)
    cont_ = np.clip((Vp - 0.5) * 0.12, -0.06, 0.06)[..., None]
    col = col * (1 + cont_)
    # shade core / underside: saturated violet
    shade_w = C.smoothstep(0.42, 0.15, Vp)
    Us = cv2.GaussianBlur(U, (0, 0), 6 * uS)
    col += (SHD_LOW - col) * (0.55 * Us * shade_w)[..., None]
    # sky-lit tops of shade lobes: a blue-violet half step
    skq = C.smoothstep(0.3, 0.36, TP + (lf - 0.55) * 0.25) * 0.6 + 0.4 * C.smoothstep(0.55, 0.61, TP + (lf - 0.55) * 0.2)
    col += (SKY_LIT - col) * (0.5 * skq * C.smoothstep(0.6, 0.0, Us) * shade_w)[..., None]
    # ---- thin warm rim on the sun-facing silhouette (and on lobe edges over darker lobes)
    shx = np.float32([[1, 0, -Lx * 2.2 * uS], [0, 1, -Ly * 2.2 * uS]])
    edge = np.clip(A - cv2.warpAffine(A, shx, (ws, hs)), 0, 1)
    Ag = cv2.GaussianBlur(A, (0, 0), 5 * uS)
    gy1, gx1 = np.gradient(Ag)
    gm = np.hypot(gx1, gy1) + 1e-6
    facing = C.smoothstep(0.1, 0.7, -(gx1 * Lx + gy1 * Ly) / gm)
    rim = edge * facing * C.smoothstep(0.25, 0.7, EX)
    Vsh = cv2.warpAffine(V, shx, (ws, hs))
    inner = np.clip((V - Vsh) * 3.0, 0, 1) * C.smoothstep(0.55, 0.85, V) * C.smoothstep(0.4, 0.8, EX)
    rim = np.clip(np.maximum(rim, 0.6 * inner), 0, 1)
    col += (RIM - col) * (0.85 * rim)[..., None]
    # far clumps: a little aerial haze
    col += (cc('#8a7ab8') - col) * (far_haze * FH)[..., None]
    # ---- lost edges on the down / away-from-sun silhouette (soft alpha, in shade)
    softA = cv2.GaussianBlur(A, (0, 0), 2.6 * uS)
    away = C.smoothstep(0.3, 0.85, (gx1 * Lx + gy1 * Ly) / gm)
    dn = np.maximum(C.smoothstep(0.4, 0.9, gy1 / gm), 0.8 * away) * C.smoothstep(0.45, 0.2, V)
    A2 = A * (1 - dn) + np.minimum(A, softA * softA * 1.3) * dn
    A2 = A2 * (1 - HM)
    # ---- composite over the branches, downsample (area) -> canvas
    al = A2 + BA * (1 - A2)
    prem = col * A2[..., None] + BR * (BA * (1 - A2))[..., None]
    small = cv2.resize(np.dstack([prem, al]).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    als = small[..., 3]
    colr = small[..., :3] / np.maximum(als, 1e-5)[..., None]
    cv.put((als, X0, Y0), colr)
    if wcv is not None:
        a0 = cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA)
        sws = cv2.resize(SW * A, (w, h), interpolation=cv2.INTER_AREA)
        num = cv2.GaussianBlur(sws.astype(np.float32), (0, 0), 7 * u)
        den = cv2.GaussianBlur(a0.astype(np.float32), (0, 0), 7 * u)
        swm = np.clip(num / np.maximum(den, 1e-4), 0, 1)
        # weight grows toward the outer (upper / right) crown: the base stays anchored on the slope
        yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        base = 0.45 + 0.55 * C.smoothstep(1.0, 0.35, yy) * np.ones((1, w), np.float32)
        wt = np.clip(cv2.GaussianBlur(als.astype(np.float32), (0, 0), 3 * u) * 2.0, 0, 1) * base
        wcv.put((wt, X0, Y0), np.dstack([swm] * 3))
    return (als, X0, Y0)
