"""s06_seaside foliage, round 8: clustered leaf-mass painter for the big hillside tree.

Replaces the noise-thresholded canopy (which read as posterised vector camouflage). The crown is built the way a
background painter builds it, as a hierarchy of rounded leaf masses:
  clump (big mass) -> lobes (cauliflower heads, lower ones overlap the ones above) -> florets (small bumps on
  the upper arc of every lobe).
Every lobe is painted as one flat shape whose value comes from where it sits (its own sun-facing side, its
place in the clump, the clump's place in the tree), quantised into 4 painted steps
  cool blue-green shadow (one broad value, violet toward the underside) -> olive half-tone -> gold-green lit
  -> pale gold top dabs,
with soft 1-2 px transitions whose borders are broken only by small leaf-cluster dabs (dab-tipped edges, no
large-scale noise). A lobe's top edge is crisp, its underside is lost into the mass behind. Fine leaf breakup
lives only on the outer silhouette (sun / top side); the underside of the whole mass dissolves softly. A thin
warm rim follows the sun-facing silhouette. A few small sky holes are cut near the shade-side edge (alpha
holes: the sky shows through, never black).
Also returns a smooth per-clump sway weight map (secondary wind motion). Deterministic."""
import math
import numpy as np
import cv2

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


PAL4 = dict(shd=cc('#1c3942'), shd_sky=cc('#2b4f5a'), shd_low=cc('#2b2b58'), half=cc('#546a40'),
            lit=cc('#b09540'), top=cc('#f6d47c'), rim=np.array([1.6, 1.2, 0.7], np.float32),
            haze=cc('#7a6a9a'), branch=cc('#1a2230'), branch_lit=cc('#8a5a3a'))


def _cluster(m, x, y, r, ang, val=255, nleaf=4, rng=None):
    """a leaf cluster: nleaf pointed leaves fanned around (x, y), drawn into uint8 mask m (shift 2, AA)."""
    k = nleaf
    for i in range(k):
        a = ang + (i - (k - 1) / 2) * (2.2 / max(k - 1, 1)) + (rng.normal(0, 0.2) if rng is not None else 0)
        cx, cy = x + math.cos(a) * r * 0.5, y + math.sin(a) * r * 0.5
        cv2.ellipse(m, (int(cx * 4), int(cy * 4)), (max(int(r * 0.58 * 4), 2), max(int(r * 0.3 * 4), 1)),
                    math.degrees(a), 0, 360, val, -1, cv2.LINE_AA, 2)


def _ell(m, x, y, rx, ry, ang=0.0, val=255):
    cv2.ellipse(m, (int(x * 4), int(y * 4)), (max(int(rx * 4), 1), max(int(ry * 4), 1)), ang, 0, 360, val, -1,
                cv2.LINE_AA, 2)


def canopy4(cv, clumps, rng, unit=1.0, clip=None, L=(0.84, -0.54), pal=PAL4, ss=2, far_haze=0.2,
            branches=(), lit_bias=0.0, wcv=None):
    """clumps: (x, y, r, f, clipped) canvas px, back to front; f = 0 near .. 1 far.
    branches: [(polyline, base width px), ...] painted first. wcv: optional canvas for sway weights."""
    u = max(unit, 0.3)
    cl = np.array([c[:4] for c in clumps], np.float64)
    pad = 20 * u
    X0 = int(max(math.floor((cl[:, 0] - cl[:, 2] * 1.4).min() - pad), 0))
    Y0 = int(max(math.floor((cl[:, 1] - cl[:, 2] * 1.4).min() - pad), 0))
    X1 = int(min(math.ceil((cl[:, 0] + cl[:, 2] * 1.4).max() + pad), cv.W))
    Y1 = int(min(math.ceil((cl[:, 1] + cl[:, 2] * 1.4).max() + pad), cv.H))
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
    Pc = {k: np.asarray(v, np.float32) for k, v in pal.items()}
    clipS = None
    if clip is not None:
        cm, cx0, cy0 = clip
        c_ = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            c_[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
        clipS = cv2.resize(c_, (ws, hs), interpolation=cv2.INTER_LINEAR)
    RGB = np.zeros((hs, ws, 3), np.float32)
    A = np.zeros((hs, ws), np.float32)
    SW = np.zeros((hs, ws), np.float32)          # sway weight (painted per clump)
    Vf = np.full((hs, ws), -1.0, np.float32)     # painted light value (for the silhouette pass)
    uS = u * S
    # ---- branches (tapering limbs, a warm lit edge on the sun side)
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
        col = Pc['branch'][None, None] + (Pc['branch_lit'] - Pc['branch'])[None, None] * edge[..., None]
        RGB[:] = RGB * (1 - mf[..., None]) + col * mf[..., None]
        A[:] = np.maximum(A, mf)
    # ---- tree-level form: blurred union of the clumps -> which side of the whole crown faces the sun
    G = np.zeros((hs, ws), np.float32)
    for (x, y, r, f, _c) in clumps:
        cv2.ellipse(G, (int((x - X0) * S * 4), int((y - Y0) * S * 4)), (int(r * S * 1.05 * 4), int(r * S * 0.85 * 4)),
                    0, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
    gsig = max(float(np.median(cl[:, 2])) * S * 1.1, 4)
    pb = int(2 * gsig) + 1
    Gb = cv2.GaussianBlur(cv2.copyMakeBorder(G, pb, pb, pb, pb, cv2.BORDER_CONSTANT, value=0), (0, 0), gsig)[pb:-pb, pb:-pb]
    ggy, ggx = np.gradient(Gb)
    gn = np.hypot(ggx, ggy)
    gsc = 1.0 / (np.percentile(gn[G > 0.5], 85) + 1e-6) if (G > 0.5).any() else 1.0
    Tg = np.clip(-(ggx * Lx + ggy * Ly) * gsc, -1, 1)          # +1: crown flank facing the sun
    Tdown = np.clip(ggy * gsc, 0, 1)                            # underside of the whole crown
    # ---- dab field: small leaf-cluster stamps that break the value borders (dab-tipped, not noise)
    Dm = np.zeros((hs, ws), np.uint8)
    Dn = np.zeros((hs, ws), np.uint8)
    nd = int(hs * ws / (170 * uS * uS))
    for k in range(nd):
        x_, y_ = rng.uniform(0, ws), rng.uniform(0, hs)
        rr = rng.uniform(2.6, 5.5) * uS
        _cluster(Dm, x_, y_, rr, rng.uniform(-2.8, -0.4), nleaf=int(rng.integers(3, 6)), rng=rng)
    D = cv2.GaussianBlur((Dm.astype(np.float32) - Dn.astype(np.float32)) / 255, (0, 0), 0.5 * uS)
    # colour of a painted value v (4 steps, soft 1-2 px transitions)
    t1, t2, t3 = 0.02, 0.3, 0.74
    bw = 0.035

    def paint_col(v, shd):
        c = shd.copy()
        c += (Pc['half'] - c) * C.smoothstep(t1 - bw, t1 + bw, v)[..., None]
        c += (Pc['lit'] - c) * C.smoothstep(t2 - bw, t2 + bw, v)[..., None]
        c += (Pc['top'] - c) * C.smoothstep(t3 - bw, t3 + bw, v)[..., None]
        return c
    # ---- clumps, back to front; lobes inside a clump top to bottom (lower heads overlap the upper ones)
    for ci, (x, y, r, f, clipped) in enumerate(clumps):
        R = r * S
        cx, cy = (x - X0) * S, (y - Y0) * S
        swc = rng.uniform(0.45, 1.0)
        lobes = []
        # core: one or two broad heads carrying most of the shadow mass
        lobes.append((cx + rng.uniform(-0.1, 0.1) * R, cy + 0.2 * R, R * 0.78, R * 0.6, 0))
        # mid ring of cauliflower heads on the upper arc, larger toward the sun
        for j in range(int(rng.integers(5, 8))):
            a = -math.pi / 2 + rng.uniform(-1.75, 1.75)
            d = R * rng.uniform(0.3, 0.58)
            sunf = max(math.cos(a) * Lx + math.sin(a) * Ly, 0)
            rr = R * rng.uniform(0.26, 0.42) * (1 + 0.3 * sunf)
            lobes.append((cx + math.cos(a) * d * 1.1, cy + math.sin(a) * d * 0.85, rr, rr * rng.uniform(0.78, 0.92), 1))
        # small outer heads on the rim (skyline cauliflower)
        for j in range(int(rng.integers(6, 11))):
            a = -math.pi / 2 + rng.uniform(-1.8, 1.8)
            d = R * rng.uniform(0.68, 0.86)
            rr = R * rng.uniform(0.11, 0.2)
            lobes.append((cx + math.cos(a) * d * 1.1, cy + math.sin(a) * d * 0.8 + 0.04 * R, rr, rr * 0.86, 2))
        lobes.sort(key=lambda l: (l[1] - l[3] * 0.3) if l[4] else -1e9)
        bb = int(R * 1.5 + 14 * uS)
        ya, yb = max(int(cy - bb), 0), min(int(cy + bb), hs)
        xa, xb = max(int(cx - bb), 0), min(int(cx + bb), ws)
        if yb <= ya or xb <= xa:
            continue
        hh, ww = yb - ya, xb - xa
        sl = (slice(ya, yb), slice(xa, xb))
        yy, xx = np.mgrid[ya:yb, xa:xb].astype(np.float32)
        Dl = D[sl]
        Tgl = Tg[sl]
        cull = clipped and clipS is not None
        if cull and clipS[int(np.clip(cy, 0, hs - 1)), int(np.clip(cx, 0, ws - 1))] < 0.5:
            continue
        for (lx, ly, rx, ry, lev) in lobes:
            # the hill clip culls whole heads (by their centre), so the silhouette is always made of heads
            if cull:
                iy_c, ix_c = int(np.clip(ly, 0, hs - 1)), int(np.clip(lx, 0, ws - 1))
                if clipS[iy_c, ix_c] < 0.1 and lev > 0:
                    continue
            # lobe silhouette: the head + florets on its upper arc (third cauliflower level)
            m = np.zeros((hh, ww), np.uint8)
            _ell(m, lx - xa, ly - ya, rx, ry, rng.uniform(-15, 15))
            if rx > 8 * uS:
                a2 = rng.uniform(-2.6, -0.5)
                _ell(m, lx - xa + math.cos(a2) * rx * 0.35, ly - ya + math.sin(a2) * ry * 0.3, rx * rng.uniform(0.55, 0.75),
                     ry * rng.uniform(0.5, 0.7), rng.uniform(-30, 30))
            nfl = int(rng.integers(4, 8)) if rx > 6 * uS else 0
            for q in range(nfl):
                a = -math.pi / 2 + rng.uniform(-1.5, 1.5)
                fr_ = rx * rng.uniform(0.2, 0.34)
                _ell(m, lx - xa + math.cos(a) * rx * 0.86, ly - ya + math.sin(a) * ry * 0.86, fr_, fr_ * 0.85,
                     rng.uniform(-20, 20))
            # leaf-cluster scallops along the lobe's upper arc (leafy lobe tops inside the mass)
            if rx > 5 * uS:
                nlf = int(rx / (2.6 * uS)) + 4
                for q in range(nlf):
                    a = -math.pi / 2 + rng.uniform(-1.7, 1.7)
                    rr = rng.uniform(2.8, 6.5) * uS
                    ex_, ey_ = lx - xa + math.cos(a) * rx * 0.95, ly - ya + math.sin(a) * ry * 0.95
                    _cluster(m, ex_, ey_, rr, a + rng.normal(0, 0.4), nleaf=int(rng.integers(3, 6)), rng=rng)
            mf = m.astype(np.float32) / 255
            if mf.max() < 0.3:
                continue
            # light value: lobe's own sun side + its place in the clump + the clump's place in the tree
            dx, dy = (xx - lx) / rx, (yy - ly) / ry
            # lit cap = the part of the head outside a copy of itself shifted away from the sun: a curved
            # crescent terminator (flat painted, not a gradient across the ball)
            kk = rng.uniform(0.4, 0.6)
            ddx, ddy = dx + Lx * kk, dy + Ly * kk
            v_l = np.clip((np.sqrt(ddx * ddx + ddy * ddy) - 0.9) * 2.2, -1.0, 1.0)
            pc = ((lx - cx) / R) * Lx + ((ly - cy) / R) * Ly
            iy_, ix_ = int(np.clip(ly - ya, 0, hh - 1)), int(np.clip(lx - xa, 0, ww - 1))
            tg = float(Tgl[iy_, ix_])
            lb = 0.35 * pc + 0.45 * tg + rng.normal(0, 0.05) + lit_bias - 0.25 * (lev == 0)
            ex_ = float(np.clip(0.2 + 1.2 * lb, 0.0, 1.0))    # how much of the sun this head gets
            # body one value, lit cap one step (or two) above it
            vb = -0.25 + 0.7 * ex_
            v = vb + (0.15 + 0.4 * ex_) * C.smoothstep(-0.15, 0.15, v_l) + 0.15 * Dl * min(ex_ * 3, 1.0)
            v = v - 0.2 * C.smoothstep(0.2, 0.9, dy)           # a lobe's underside always sinks into shade
            # shadow colour: sky-lit teal on upper shade flanks, violet toward the crown underside
            up = C.smoothstep(0.0, -0.8, dy)
            shd = np.broadcast_to(Pc['shd'], (hh, ww, 3)).copy()
            crest = C.smoothstep(-0.2, -0.45, dy + 0.25 * Dl) * C.smoothstep(0.6, 0.0, dx * Lx)
            shd += (Pc['shd_sky'] - shd) * (0.8 * crest)[..., None]
            shd += (Pc['shd_low'] - shd) * (0.75 * np.maximum(Tdown[sl], C.smoothstep(0.2, 1.0, dy) * 0.5))[..., None]
            col = paint_col(v, shd)
            col += (Pc['haze'] - col) * (far_haze * f ** 1.3)
            # crisp top edge, lost underside (soft alpha only where the lobe faces down)
            soft = cv2.GaussianBlur(mf, (0, 0), max(0.12 * ry, 1.5 * uS))
            down = C.smoothstep(0.15, 0.85, dy)
            ab_ = A[sl]
            a = mf * (1 - down) + (np.minimum(mf, soft) * ab_ + mf * (1 - ab_)) * down
            RGB[sl] = RGB[sl] * (1 - a[..., None]) + col * a[..., None]
            A[sl] = A[sl] * (1 - a) + a
            Vf[sl] = Vf[sl] * (1 - a) + v * a
            SW[sl] = SW[sl] * (1 - a) + swc * a
    # ---- leaf texture inside the shadow mass: sparse drooping clusters a half-step lighter (sky-lit leaves),
    # low contrast so the shadow stays one broad value
    LT = np.zeros((hs, ws), np.uint8)
    nlt = int(hs * ws / (150 * uS * uS))
    for _ in range(nlt):
        x_, y_ = rng.uniform(0, ws), rng.uniform(0, hs)
        iy, ix = int(y_), int(x_)
        if A[iy, ix] < 0.9 or Vf[iy, ix] > t1 - 0.05:
            continue
        _cluster(LT, x_, y_, rng.uniform(2.4, 4.5) * uS, math.pi / 2 + rng.normal(0, 0.6), nleaf=int(rng.integers(3, 5)),
                 rng=rng)
    lt = LT.astype(np.float32) / 255 * 0.45 * C.smoothstep(t1 + 0.02, t1 - 0.1, Vf)
    RGB[:] += (Pc['shd_sky'] * A[..., None] - RGB) * lt[..., None]
    # ---- outer silhouette: fine leaf breakup on the top / sun side only, colour picked from the mass
    Ab = (A > 0.5).astype(np.uint8)
    Abl = cv2.GaussianBlur(A, (0, 0), 6 * uS)
    gy_, gx_ = np.gradient(Abl)
    cnts, _ = cv2.findContours(Ab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    FM = np.zeros((hs, ws), np.uint8)
    FV = np.zeros((hs, ws), np.float32)
    FC = np.zeros((hs, ws, 3), np.float32)
    for c in cnts:
        c = c[:, 0, :]
        if len(c) < 20:
            continue
        i = 0
        while i < len(c):
            px, py = int(c[i][0]), int(c[i][1])
            g = np.array([gx_[py, px], gy_[py, px]])
            nrm = -g / (np.linalg.norm(g) + 1e-9)
            down = nrm[1]
            rr = rng.uniform(2.8, 6.5) * uS
            droop = down > 0.35
            if droop and rng.random() < 0.4:
                i += int(rr * 1.2) + 1
                continue
            iy, ix = int(np.clip(py - nrm[1] * rr, 0, hs - 1)), int(np.clip(px - nrm[0] * rr, 0, ws - 1))
            o = rr * rng.uniform(0.2, 1.0)
            e = int(rr * 2.2) + 3
            wy0, wy1, wx0, wx1 = max(py - e, 0), min(py + e, hs), max(px - e, 0), min(px + e, ws)
            loc = np.zeros((wy1 - wy0, wx1 - wx0), np.uint8)
            ang_ = math.atan2(nrm[1], nrm[0]) + rng.normal(0, 0.5)
            if droop:
                ang_ = 0.5 * ang_ + 0.5 * math.pi / 2          # hanging leaf clusters on the underside
                o *= 0.6
            _cluster(loc, px + nrm[0] * o - wx0, py + nrm[1] * o - wy0, rr * (0.85 if droop else 1.0),
                     ang_, nleaf=int(rng.integers(3, 6)), rng=rng)
            vin = float(Vf[iy, ix])
            sunf = nrm[0] * Lx + nrm[1] * Ly
            nw = loc > FM[wy0:wy1, wx0:wx1]
            FV[wy0:wy1, wx0:wx1][nw] = vin + 0.14 * max(sunf, 0) + 0.04
            FC[wy0:wy1, wx0:wx1][nw] = RGB[iy, ix]
            FM[wy0:wy1, wx0:wx1] = np.maximum(FM[wy0:wy1, wx0:wx1], loc)
            i += max(int(rr * rng.uniform(0.7, 1.2)), 1)
    fm = FM.astype(np.float32) / 255 * (1 - A)
    fcol = paint_col(FV, FC)
    RGB[:] = RGB * (1 - fm[..., None]) + fcol * fm[..., None]
    A[:] = A + fm * (1 - A)
    # ---- warm rim on the sun-facing silhouette (thin, broken, fading down the crown)
    sh = np.float32([[1, 0, -Lx * 1.6 * uS], [0, 1, -Ly * 1.6 * uS]])
    edge = np.clip(A - cv2.warpAffine(A, sh, (ws, hs)), 0, 1)
    Abl = cv2.GaussianBlur(A, (0, 0), 5 * uS)
    gy_, gx_ = np.gradient(Abl)
    gm = np.hypot(gx_, gy_) + 1e-6
    facing = C.smoothstep(0.2, 0.7, -(gx_ * Lx + gy_ * Ly) / gm)
    brk = C.smoothstep(-0.1, 0.25, D + 0.1)
    rim = edge * facing * brk * C.smoothstep(-0.1, 0.4, Vf)
    RGB[:] += (Pc['rim'] - RGB) * (0.85 * rim)[..., None]
    # ---- lost underside of the whole crown
    softA = cv2.GaussianBlur(A, (0, 0), 2.0 * uS)
    dn = C.smoothstep(0.45, 0.9, gy_ / gm) * (Vf < t2)
    A = A * (1 - dn) + np.minimum(A, softA * softA * 1.3) * dn
    # ---- small sky holes near the shade-side edge (alpha holes, the sky shows through)
    k1, k2 = max(int(7 * uS), 2), max(int(34 * uS), 4)
    Ab = (A > 0.6).astype(np.uint8)
    ring = cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1,) * 2)) - \
        cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1,) * 2))
    ys_, xs_ = np.nonzero(ring > 0)
    HM = np.zeros((hs, ws), np.uint8)
    nh = int(len(ys_) / (2600 * uS * uS))
    for _ in range(nh):
        k = int(rng.integers(0, len(ys_)))
        hx_, hy_ = xs_[k], ys_[k]
        if Vf[hy_, hx_] > t1 or Tg[hy_, hx_] > 0.0:
            continue
        hr = rng.uniform(2.0, 4.0) * uS
        for _k in range(int(rng.integers(1, 3))):
            _cluster(HM, hx_ + rng.normal(0, hr * 0.5), hy_ + rng.normal(0, hr * 0.5), hr, rng.uniform(0, 6.28),
                     nleaf=int(rng.integers(3, 5)), rng=rng)
    A = A * (1 - HM.astype(np.float32) / 255)
    prem = np.concatenate([RGB * A[..., None], A[..., None]], -1)
    small = cv2.resize(prem, (w, h), interpolation=cv2.INTER_AREA)
    al = small[..., 3]
    colr = small[..., :3] / np.maximum(al, 1e-5)[..., None]
    cv.put((al, X0, Y0), colr)
    if wcv is not None:
        swm = cv2.GaussianBlur(SW, (0, 0), 10 * uS)
        swm = cv2.resize(swm, (w, h), interpolation=cv2.INTER_AREA)
        wcv.put((al, X0, Y0), np.dstack([swm] * 3))
    return (al, X0, Y0)
