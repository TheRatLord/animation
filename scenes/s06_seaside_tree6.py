"""s06_seaside foliage, round 10: leaf-MASS painter for the big hillside tree.

Round 9 filled every lobe with thousands of tiny leaf dabs, each with its own value jitter: at full
resolution that read as a cut-out-filter speckle full of dark pits. This painter works the way a background
artist blocks a tree in, mass first:

  clump -> 2-5 cauliflower heads (a core ellipse + a ring of bump circles along its upper / sun arc).

Each head is filled with FLAT painted values, never per-pixel noise:
  base   : its place in the whole crown (tree-level sun-facing flank) -> one broad cool shadow value on the
           shade side, a mid green on the lit flank;
  lit cap: head minus itself shifted away from the sun -> a crescent on the sun side whose inner border is
           the scalloped head outline itself (a cauliflower terminator, not a vector cut);
  hot top: a thinner crescent, only on the heads facing the sun.
Heads are laid back to front (lower heads overlap the ones above), so every lit cap reads against the shade
of the head behind. Shadow-flank heads get only a faint sky-lit half-step cap: the shadow mass stays one
broad value with a few lobe shapes inside. The value buffer is softened a little (soft transitions between
the painted steps, the alpha stays crisp) and mapped through one colour ramp. A low-frequency brush
variation (40-px scale) keeps the flat values from looking digital. Leaf-cluster tufts (5-9 px) are laid
only along the sun-facing outer silhouette and dangle a little from the lit caps into the shade (dab-tipped
edges). A few small sky holes (alpha, filled by the sky behind) sit near the shade-side edge.
Also returns a per-clump sway map (weight, clump id) for the per-cluster wind sway. Deterministic."""
import math
import numpy as np
import cv2

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


RAMP = [(0.00, '#243c56'), (0.16, '#2b4a5c'), (0.32, '#3a5f52'), (0.48, '#5f8a3c'), (0.62, '#94ab32'),
        (0.76, '#d0c03a'), (0.88, '#f6cf62'), (1.00, '#ffe0aa')]
SHD_LOW = cc('#2e3266')        # violet toward the underside of the crown
SHD_SKY = cc('#33607e')        # sky-lit teal on the upper shade flanks
RIM = np.array([1.7, 1.02, 0.62], np.float32)
BRANCH, BRANCH_LIT = cc('#1a2230'), cc('#8a5a3a')


def ramp(v):
    v = np.clip(v, 0.0, 1.0)
    xs = np.array([p for p, _ in RAMP], np.float32)
    cs = np.array([cc(h) for _, h in RAMP], np.float32)
    out = np.empty(v.shape + (3,), np.float32)
    for k in range(3):
        out[..., k] = np.interp(v, xs, cs[:, k])
    return out


def _leaf(img, x, y, r, ang, col, nleaf, rng, S4=4):
    """leaf-cluster dab: nleaf pointed leaves fanned around (x, y)."""
    k = nleaf
    col = tuple(float(c) for c in col) if isinstance(col, tuple) else float(col)
    for i in range(k):
        a = ang + (i - (k - 1) / 2) * (2.1 / max(k - 1, 1)) + rng.normal(0, 0.2)
        cx, cy = x + math.cos(a) * r * 0.55, y + math.sin(a) * r * 0.55
        cv2.ellipse(img, (int(cx * S4), int(cy * S4)), (max(int(r * 0.62 * S4), 2), max(int(r * 0.3 * S4), 1)),
                    math.degrees(a), 0, 360, col, -1, cv2.LINE_8, 2)


def _head(ws, hs, cx, cy, rx, ry, bumps, S4=4):
    """binary cauliflower head mask in a local window: returns (mask uint8, x0, y0)."""
    pad = 2
    xs = [cx - rx] + [b[0] - b[2] for b in bumps]
    xe = [cx + rx] + [b[0] + b[2] for b in bumps]
    ys = [cy - ry] + [b[1] - b[2] for b in bumps]
    ye = [cy + ry] + [b[1] + b[2] for b in bumps]
    x0 = int(max(math.floor(min(xs)) - pad, 0)); x1 = int(min(math.ceil(max(xe)) + pad, ws))
    y0 = int(max(math.floor(min(ys)) - pad, 0)); y1 = int(min(math.ceil(max(ye)) + pad, hs))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None, x0, y0
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.ellipse(m, (int((cx - x0) * S4), int((cy - y0) * S4)), (int(rx * S4), int(ry * S4)), 0, 0, 360, 1, -1,
                cv2.LINE_8, 2)
    for (bx, by, br) in bumps:
        cv2.circle(m, (int((bx - x0) * S4), int((by - y0) * S4)), int(br * S4), 1, -1, cv2.LINE_8, 2)
    return m, x0, y0


def _shift(m, dx, dy):
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])       # out(p) = m(p + d)
    return cv2.warpAffine(m, M, (m.shape[1], m.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)


def canopy6(cv, clumps, rng, unit=1.0, clip=None, L=(0.92, -0.4), ss=2, far_haze=0.2, branches=(), wcv=None,
            lit_bias=0.0):
    """clumps: (x, y, r, f, clipped) canvas px, back to front; f = 0 near .. 1 far."""
    u = max(unit, 0.3)
    cl = np.array([c[:4] for c in clumps], np.float64)
    pad = 30 * u
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

    # paint buffer: (value, underside, sway id, coverage)
    B = np.zeros((hs, ws, 4), np.float32)
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

    S4 = 4
    for ci, (x, y, r, f, clipped) in enumerate(clumps):
        R = r * S
        cx, cy = (x - X0) * S, (y - Y0) * S
        swc = rng.uniform(0.0, 1.0)
        cull = clipped and clipS is not None
        if cull and clipS[int(np.clip(cy, 0, hs - 1)), int(np.clip(cx, 0, ws - 1))] < 0.5:
            continue
        # heads: one big core head + 1-4 smaller heads on its upper / sun arc, drawn top (behind) to bottom
        heads = [(cx + rng.uniform(-0.08, 0.08) * R, cy + 0.12 * R, R * 0.86, R * 0.7)]
        nh = int(rng.integers(1, 5)) if R > 30 * uS else int(rng.integers(0, 2))
        for j in range(nh):
            a = -math.pi / 2 + rng.uniform(-1.4, 1.2)
            d = R * rng.uniform(0.35, 0.6)
            rr = R * rng.uniform(0.34, 0.55)
            heads.append((cx + math.cos(a) * d * 1.05, cy + math.sin(a) * d * 0.9, rr, rr * rng.uniform(0.78, 0.92)))
        heads.sort(key=lambda hd: hd[1] + 0.3 * hd[3])
        for (hx, hy, rx, ry) in heads:
            if cull:
                if clipS[int(np.clip(hy, 0, hs - 1)), int(np.clip(hx, 0, ws - 1))] < 0.1:
                    continue
            # cauliflower bumps along the upper / sun-side arc: medium heads + small heads on heads
            mr = min(rx, ry)
            bumps = []
            for (nb, s0, s1, d0, d1) in ((int(rng.integers(7, 11)), 0.17, 0.32, 0.3, 0.7),
                                         (int(rng.integers(10, 16)), 0.07, 0.13, 0.05, 0.45)):
                angs = rng.uniform(-math.pi * 1.12, 0.45, nb)
                for a in angs:
                    br_ = mr * rng.uniform(s0, s1) * (1 - 0.3 * f)
                    dd = 1.0 - br_ / max(mr, 1e-3) * rng.uniform(d0, d1)
                    bumps.append((hx + math.cos(a) * rx * dd, hy + math.sin(a) * ry * dd, br_))
            m, mx0, my0 = _head(ws, hs, hx, hy, rx, ry, bumps, S4)
            if m is None:
                continue
            hh_, ww_ = m.shape
            iy, ix = int(np.clip(hy, 0, hs - 1)), int(np.clip(hx, 0, ws - 1))
            tg = float(Tg[iy, ix])
            td = float(Tdown[iy, ix])
            # exposure of this head: tree-level flank + its place on its clump
            pc = ((hx - cx) * Lx + (hy - cy) * Ly) / R
            ex = float(np.clip(0.66 + 1.0 * (0.75 * tg + 0.35 * pc - 0.35 * td + lit_bias) + rng.normal(0, 0.05),
                               0, 1))
            v_base = 0.08 + 0.3 * C.smoothstep(0.45, 0.95, ex)
            v_lit = 0.19 + 0.6 * C.smoothstep(0.3, 0.75, ex)
            v_hot = 0.94
            dl = mr * (0.26 + 0.5 * ex)
            lit = m & (1 - _shift(m, Lx * dl, Ly * dl))
            dh = mr * 0.12
            hot = m & (1 - _shift(m, Lx * dh, Ly * dh - 0.4 * dh)) if ex > 0.66 else np.zeros_like(m)
            yy = (np.arange(my0, my0 + hh_, dtype=np.float32)[:, None] - hy) / ry
            xx = (np.arange(mx0, mx0 + ww_, dtype=np.float32)[None, :] - hx) / rx
            pl = C.smoothstep(-0.9, 0.9, xx * Lx + yy * Ly)          # soft internal gradient toward the sun
            v = np.where(lit > 0, v_lit + 0.05 * pl, v_base + (0.035 + 0.05 * ex) * pl)
            v = np.where(hot > 0, v_hot, v).astype(np.float32)
            v = v * (1 - 0.25 * f) + 0.06 * f
            v = np.broadcast_to(v, (hh_, ww_))
            und = np.clip(0.45 * C.smoothstep(0.3, 1.0, yy) + 0.6 * td, 0, 1) * np.ones((1, ww_), np.float32)
            sl = B[my0:my0 + hh_, mx0:mx0 + ww_]
            mm = m.astype(bool)
            sl[..., 0][mm] = v[mm]
            sl[..., 1][mm] = und[mm]
            sl[..., 2][mm] = swc
            sl[..., 3][mm] = 1.0
            vl = float(v_lit * (1 - 0.25 * f) + 0.06 * f)
            # leafy tips along the head's upper / sun-side outline (its own value): the head's edge over the
            # head behind breaks into small leaf clusters instead of a clean vector arc
            mb = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 2.5 * uS)
            gyh, gxh = np.gradient(mb)
            ed = m & (1 - cv2.erode(m, np.ones((3, 3), np.uint8)))
            eyy, exx = np.nonzero(ed)
            if len(eyy):
                gnh = np.hypot(gxh[eyy, exx], gyh[eyy, exx]) + 1e-6
                nx_, ny_ = -gxh[eyy, exx] / gnh, -gyh[eyy, exx] / gnh
                ok = (ny_ < -0.15) | ((nx_ * Lx + ny_ * Ly) > 0.3)
                idx = np.nonzero(ok)[0]
                if len(idx):
                    sp_ = 5.5 * uS * (1 - 0.3 * f)
                    nk = int(min(len(idx) / sp_ * 1.3, 400))
                    for k in rng.choice(idx, min(nk, len(idx)), replace=False):
                        ey_, ex_ = eyy[k], exx[k]
                        ix_ = int(np.clip(ex_ - nx_[k] * 3 * uS, 0, ww_ - 1))
                        iy_ = int(np.clip(ey_ - ny_[k] * 3 * uS, 0, hh_ - 1))
                        vv = float(v[iy_, ix_])
                        tr = rng.uniform(3.0, 5.5) * uS * (1 - 0.3 * f)
                        ta = math.atan2(ny_[k], nx_[k]) + rng.normal(0, 0.5)
                        _leaf(B, ex_ + mx0 + nx_[k] * tr * 0.2, ey_ + my0 + ny_[k] * tr * 0.2, tr, ta,
                              (vv, float(und[iy_, ix_]), swc, 1.0), int(rng.integers(3, 5)), rng, S4)
            # lit leaf clusters dangling off the inner border of the lit cap (dab-tipped light into the shade)
            if ex > 0.4:
                inner = lit & _shift(m, Lx * dl * 1.25, Ly * dl * 1.25) & (1 - _shift(lit, Lx * 3 * uS, Ly * 3 * uS))
                yy_, xx_ = np.nonzero(inner)
                if len(yy_):
                    nt = int(min(len(yy_) / (5 * uS), 40))
                    for k in rng.choice(len(yy_), nt, replace=False):
                        tr = rng.uniform(4.0, 8.0) * uS * (1 - 0.3 * f)
                        ta = math.atan2(-Ly, -Lx) + rng.normal(0.35, 0.6)
                        tx, ty = xx_[k] + mx0, yy_[k] + my0
                        _leaf(B, tx + math.cos(ta) * tr * 0.4, ty + math.sin(ta) * tr * 0.4, tr, ta,
                              (vl, float(B[int(min(ty, hs - 1)), int(min(tx, ws - 1)), 1]), swc, 1.0),
                              int(rng.integers(3, 6)), rng, S4)
                # a few mid-value leaf clusters inside the lit cap (painted texture, never dark pits)
                yy_, xx_ = np.nonzero(lit & (1 - hot))
                if len(yy_):
                    nt = int(min(len(yy_) / (420 * uS * uS), 18))
                    vm = vl + 0.04
                    for k in rng.choice(len(yy_), nt, replace=False):
                        tr = rng.uniform(6.0, 10.0) * uS * (1 - 0.3 * f)
                        _leaf(B, xx_[k] + mx0, yy_[k] + my0, tr, math.pi / 2 + rng.normal(0, 0.8),
                              (vm, 0.0, swc, 1.0), int(rng.integers(3, 5)), rng, S4)
            else:
                # shade heads: a faint sky-lit half-step on the very top of the head (lobe shape inside the mass)
                cap = m & (1 - _shift(m, 0.0, -mr * 0.14))
                sl[..., 0][cap > 0] = np.maximum(sl[..., 0][cap > 0], v_base + 0.05)
    # ---- dab-tipped leaf breakup along the sun-facing outer silhouette only
    A = B[..., 3]
    Abl = cv2.GaussianBlur(A, (0, 0), 6 * uS)
    gy_, gx_ = np.gradient(Abl)
    Ab = (A > 0.5).astype(np.uint8)
    cnts, _ = cv2.findContours(Ab, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        c = c[:, 0, :]
        if len(c) < 20:
            continue
        i = 0
        while i < len(c):
            px, py = int(c[i][0]), int(c[i][1])
            g = np.array([gx_[py, px], gy_[py, px]])
            nrm = -g / (np.linalg.norm(g) + 1e-9)
            sunf = nrm[0] * Lx + nrm[1] * Ly
            rr = rng.uniform(5.0, 9.0) * uS
            if sunf > 0.1 and nrm[1] < 0.35:
                iy, ix = int(np.clip(py - nrm[1] * rr * 1.5, 0, hs - 1)), int(np.clip(px - nrm[0] * rr * 1.5, 0, ws - 1))
                vin = float(B[iy, ix, 0])
                o = rr * rng.uniform(0.0, 0.7)
                vv = min(vin + 0.04 * sunf, 1.0)
                _leaf(B, px + nrm[0] * o, py + nrm[1] * o, rr, math.atan2(nrm[1], nrm[0]) + rng.normal(0, 0.45),
                      (vv, float(B[iy, ix, 1]), float(B[iy, ix, 2]), 1.0), int(rng.integers(3, 6)), rng, S4)
            i += max(int(rr * rng.uniform(0.9, 1.6)), 1)
    # ---- a few small sky holes near the shade-side outer edge, only where the sky is behind the crown
    A = B[..., 3].copy()
    k1, k2 = max(int(14 * uS), 2), max(int(45 * uS), 4)
    Ab = (A > 0.6).astype(np.uint8)
    ring = cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1,) * 2)) - \
        cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1,) * 2))
    ys_, xs_ = np.nonzero(ring > 0)
    HM = np.zeros((hs, ws), np.float32)
    made, tries = 0, 0
    while made < 9 and tries < 400 and len(ys_):
        tries += 1
        k = int(rng.integers(0, len(ys_)))
        hx_, hy_ = xs_[k], ys_[k]
        if B[hy_, hx_, 0] > 0.3 or Tg[hy_, hx_] > 0.3:
            continue
        if clipS is not None and clipS[hy_, hx_] > 0.05:
            continue
        hr = rng.uniform(3.0, 5.0) * uS
        for _k in range(int(rng.integers(1, 3))):
            _leaf(HM, hx_ + rng.normal(0, hr * 0.5), hy_ + rng.normal(0, hr * 0.5), hr, rng.uniform(0, 6.28), 1.0,
                  int(rng.integers(3, 5)), rng, S4)
        made += 1
    # ---- to colour
    small = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)
    A1 = small[..., 3]
    inv = 1.0 / np.maximum(A1, 1e-5)
    V = small[..., 0] * inv
    UND = small[..., 1] * inv
    SWm = small[..., 2] * inv
    # soft transitions between the painted steps (premultiplied blur so the edge never darkens)
    num = cv2.GaussianBlur(V * A1, (0, 0), 1.6 * u)
    den = cv2.GaussianBlur(A1, (0, 0), 1.6 * u)
    Vs = num / np.maximum(den, 1e-4)
    V = np.where(A1 > 1e-3, 0.45 * V + 0.55 * Vs, V)
    UND = cv2.GaussianBlur(UND * A1, (0, 0), 8 * u) / np.maximum(cv2.GaussianBlur(A1, (0, 0), 8 * u), 1e-4)
    # low-frequency brush variation (no pixel noise): +/- a little value in the lit, almost none in shade
    nz = C.fbm(w, h, max(int(w / (40 * u)), 2), 3, seed=611) - 0.5
    V = V + nz * (0.02 + 0.08 * C.smoothstep(0.3, 0.8, V))
    col = ramp(V)
    shadow_w = C.smoothstep(0.4, 0.12, V)[..., None]
    Tg1 = cv2.resize(Tg, (w, h), interpolation=cv2.INTER_AREA)
    up = np.clip(1 - UND, 0, 1) * C.smoothstep(-0.6, 0.2, Tg1)
    col += (SHD_SKY - col) * (0.35 * up)[..., None] * shadow_w
    col += (SHD_LOW - col) * (0.65 * UND)[..., None] * shadow_w
    bsm = cv2.resize(np.dstack([BR * BA[..., None], BA]), (w, h), interpolation=cv2.INTER_AREA)
    ba = bsm[..., 3]
    # ---- warm rim on the sun-facing silhouette (1-2 px, only on lit edges)
    Au = A1.copy()
    shx = np.float32([[1, 0, -Lx * 1.3 * u], [0, 1, -Ly * 1.3 * u]])
    edge = np.clip(Au - cv2.warpAffine(Au, shx, (w, h)), 0, 1)
    Abl1 = cv2.GaussianBlur(Au, (0, 0), 5 * u)
    gy1, gx1 = np.gradient(Abl1)
    gm = np.hypot(gx1, gy1) + 1e-6
    facing = C.smoothstep(0.25, 0.75, -(gx1 * Lx + gy1 * Ly) / gm)
    rim = edge * facing * C.smoothstep(0.35, 0.65, V)
    col += (RIM - col) * np.clip(0.55 * cv2.GaussianBlur(rim, (0, 0), 0.8 * u), 0, 1)[..., None]
    # ---- lost underside of the whole crown (soft alpha where the silhouette faces down, in shade)
    softA = cv2.GaussianBlur(A1, (0, 0), 2.5 * u)
    dn = C.smoothstep(0.45, 0.9, gy1 / gm) * C.smoothstep(0.4, 0.2, V)
    A1 = A1 * (1 - dn) + np.minimum(A1, softA * softA * 1.25) * dn
    hm_small = cv2.resize(np.clip(HM, 0, 1), (w, h), interpolation=cv2.INTER_AREA)
    A1 = A1 * (1 - hm_small)
    al = A1 + ba * (1 - A1)
    colr = (col * A1[..., None] + bsm[..., :3] * (1 - A1[..., None])) / np.maximum(al, 1e-5)[..., None]
    cv.put((al, X0, Y0), colr)
    if wcv is not None:
        a0 = small[..., 3]
        num = cv2.GaussianBlur((SWm * a0).astype(np.float32), (0, 0), 6 * u)
        den = cv2.GaussianBlur(a0.astype(np.float32), (0, 0), 6 * u)
        swm = np.clip(num / np.maximum(den, 1e-4), 0, 1)
        wt = np.clip(cv2.GaussianBlur(al.astype(np.float32), (0, 0), 3 * u) * 2.0, 0, 1)
        wcv.put((wt, X0, Y0), np.dstack([swm] * 3))
    return (al, X0, Y0)
