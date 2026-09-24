"""s06_seaside foliage, round 9: leaf-dab painter for the big hillside tree.

The earlier painters built the crown from flat, quantised lobe shapes (4 hard value steps + a uniform
speckle field): it read as posterised vector camouflage. This one paints the way a background artist does,
with a leaf brush:

  clump (big mass) -> lobes (cauliflower heads, lower ones overlap the ones above) -> leaf-cluster dabs.

Every lobe is filled with small leaf-cluster dabs (3-5 pointed leaves each). A dab carries a painted light
VALUE (not a colour) taken from a smooth light field: the lobe's own sun-facing cap, its place in the clump
and the clump's place in the whole crown. Dabs of a lobe are laid dark first, lit last, so the lit cap's
border is dab-tipped (leaves of light over the shade) and never a vector cut. The value buffer is then
mapped through one continuous colour ramp
    cool blue-green shadow (violet toward the underside, teal on sky-lit upper flanks)
    -> cool green half-tone -> warm green -> saturated gold-green lit -> pale warm gold top,
so the shadow side stays one broad value while the lit cap turns through two intermediate values softly.
In shadow the dab value jitter is tiny (leaf texture barely reads); it grows toward the light.
Lit dabs spill out past the outer silhouette only on the sun-facing (upper-right) side; the crown's
underside is lost softly. A thin hot peach rim follows the sun-facing silhouette. Small sky holes are cut
near the shade-side edge where the sky is behind (alpha holes, never dark).
Also returns a per-clump sway weight map (secondary wind motion). Deterministic."""
import math
import numpy as np
import cv2

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


# value ramp (v in 0..1)
RAMP = [(0.00, '#1f3a52'), (0.22, '#2a4f58'), (0.42, '#3f6c48'), (0.60, '#7fa034'), (0.76, '#cfbe34'),
        (0.90, '#fccc5e'), (1.00, '#ffdca4')]
SHD_LOW = cc('#2c2c64')        # violet underside of the crown
SHD_SKY = cc('#34587a')        # sky-lit teal on the upper shade flanks
HAZE = cc('#7a6c9e')
RIM = np.array([1.8, 1.0, 0.6], np.float32)
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
    """leaf-cluster dab: nleaf pointed leaves fanned around (x, y), drawn with colour tuple col."""
    k = nleaf
    col = tuple(float(c) for c in col) if isinstance(col, tuple) else float(col)
    for i in range(k):
        a = ang + (i - (k - 1) / 2) * (2.3 / max(k - 1, 1)) + rng.normal(0, 0.22)
        cx, cy = x + math.cos(a) * r * 0.5, y + math.sin(a) * r * 0.5
        cv2.ellipse(img, (int(cx * S4), int(cy * S4)), (max(int(r * 0.6 * S4), 2), max(int(r * 0.3 * S4), 1)),
                    math.degrees(a), 0, 360, col, -1, cv2.LINE_8, 2)



def canopy5(cv, clumps, rng, unit=1.0, clip=None, L=(0.84, -0.54), ss=3, far_haze=0.2, branches=(), wcv=None,
            lit_bias=0.0):
    """clumps: (x, y, r, f, clipped) canvas px, back to front; f = 0 near .. 1 far."""
    u = max(unit, 0.3)
    cl = np.array([c[:4] for c in clumps], np.float64)
    pad = 24 * u
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

    # paint buffer: channels (value, underside, sway, coverage); hard dabs at ss, AA by area downsample
    B = np.zeros((hs, ws, 4), np.float32)
    # ---- branches (tapering limbs) into a separate colour layer (painted first, under the leaves)
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
    # ---- clumps back to front; lobes top to bottom inside a clump
    for ci, (x, y, r, f, clipped) in enumerate(clumps):
        R = r * S
        cx, cy = (x - X0) * S, (y - Y0) * S
        swc = rng.uniform(0.0, 1.0)
        cull = clipped and clipS is not None
        if cull and clipS[int(np.clip(cy, 0, hs - 1)), int(np.clip(cx, 0, ws - 1))] < 0.5:
            continue
        lobes = [(cx + rng.uniform(-0.1, 0.1) * R, cy + 0.18 * R, R * 0.8, R * 0.62, 0)]
        for j in range(int(rng.integers(4, 7))):
            a = -math.pi / 2 + rng.uniform(-1.8, 1.8)
            d = R * rng.uniform(0.28, 0.55)
            sunf = max(math.cos(a) * Lx + math.sin(a) * Ly, 0)
            rr = R * rng.uniform(0.28, 0.44) * (1 + 0.3 * sunf)
            lobes.append((cx + math.cos(a) * d * 1.1, cy + math.sin(a) * d * 0.85, rr, rr * rng.uniform(0.76, 0.9), 1))
        for j in range(int(rng.integers(5, 9))):
            a = -math.pi / 2 + rng.uniform(-1.8, 1.8)
            d = R * rng.uniform(0.66, 0.84)
            rr = R * rng.uniform(0.12, 0.22)
            lobes.append((cx + math.cos(a) * d * 1.1, cy + math.sin(a) * d * 0.8 + 0.04 * R, rr, rr * 0.86, 2))
        lobes.sort(key=lambda l: (l[1] - l[3] * 0.3) if l[4] else -1e9)
        # how exposed the clump is (tree-level flank + place in crown)
        for (lx, ly, rx, ry, lev) in lobes:
            if cull:
                iy_c, ix_c = int(np.clip(ly, 0, hs - 1)), int(np.clip(lx, 0, ws - 1))
                if clipS[iy_c, ix_c] < 0.1 and lev > 0:
                    continue
            pc = ((lx - cx) / R) * Lx + ((ly - cy) / R) * Ly
            tg = float(Tg[int(np.clip(ly, 0, hs - 1)), int(np.clip(lx, 0, ws - 1))])
            td = float(Tdown[int(np.clip(ly, 0, hs - 1)), int(np.clip(lx, 0, ws - 1))])
            lb = 0.3 * pc + 0.5 * tg + rng.normal(0, 0.05) + lit_bias - 0.22 * (lev == 0) - 0.3 * td
            ex_ = float(np.clip(0.5 + 1.2 * lb, 0.0, 1.0))
            # cauliflower outline: radial bumps on the upper arc
            nb = int(rng.integers(5, 9))
            bph = rng.uniform(0, 6.28, 3)
            # dab size scales with the lobe but stays leaf-sized; far clumps finer
            dr = float(np.clip(rx * 0.1, 3.4 * uS, 8.5 * uS)) * (1 - 0.35 * f)
            sp = dr * 0.95
            gx = np.arange(lx - rx * 1.1, lx + rx * 1.1, sp)
            gy = np.arange(ly - ry * 1.15, ly + ry * 1.1, sp * 0.9)
            if len(gx) == 0 or len(gy) == 0:
                continue
            PX, PY = np.meshgrid(gx, gy)
            PX = PX.ravel() + rng.uniform(-0.5, 0.5, PX.size) * sp
            PY = PY.ravel() + rng.uniform(-0.5, 0.5, PY.size) * sp
            dx, dy = (PX - lx) / rx, (PY - ly) / ry
            ang = np.arctan2(dy, dx)
            rad = np.hypot(dx, dy)
            upper = np.clip(-np.sin(ang), 0, 1)
            bump = 1 + (0.07 * np.sin(ang * nb + bph[0]) + 0.05 * np.sin(ang * (nb * 2 + 1) + bph[1])) * (0.3 + upper)
            keep = rad < bump * 0.93
            if not keep.any():
                continue
            PX, PY, dx, dy, rad = PX[keep], PY[keep], dx[keep], dy[keep], rad[keep]
            nz = np.sqrt(np.clip(1 - np.minimum(rad, 1) ** 2, 0, 1))
            s = (dx * Lx + dy * Ly) / np.maximum(rad, 1e-3) * np.minimum(rad, 1) + 0.25 * nz - 0.12
            jit = rng.normal(0, 1, s.size)
            sl = C.smoothstep(-0.4, 0.5, s + 0.09 * jit)
            v = ex_ * (0.2 + 0.8 * sl) * (0.35 + 0.65 * ex_)
            v = v - 0.18 * C.smoothstep(0.25, 0.95, dy)
            # sky light: every head's upper arc catches a half-step of cool sky light, so the heads still read
            # inside the broad shadow mass (cool half-tone cap, soft terminator)
            sky_up = C.smoothstep(-0.05, -0.6, dy + 0.12 * jit) * C.smoothstep(1.02, 0.6, rad)
            v = np.maximum(v, 0.05 + 0.15 * sky_up * (1 - 0.5 * f))
            # value jitter: tiny in shadow (one broad value), growing toward the light
            v = v + jit * (0.012 + 0.05 * np.clip(v, 0, 1))
            v = v * (1 - 0.25 * f) + 0.08 * f
            und = np.clip(0.55 * C.smoothstep(0.1, 1.0, dy) + 0.7 * td, 0, 1)
            # solid under-fill of the lobe core in its shade value (no gaps through to the hill behind)
            vfill = float(np.percentile(v, 12))
            cv2.ellipse(B, (int(lx * S4), int(ly * S4)), (int(rx * 0.8 * S4), int(ry * 0.78 * S4)), 0, 0, 360,
                        (vfill, float(np.mean(und)), float(swc), 1.0), -1, cv2.LINE_8, 2)
            order = np.argsort(v, kind='stable')
            for k in order:
                vv = float(v[k])
                col = (vv, float(und[k]), swc, 1.0)
                rr_ = dr * rng.uniform(0.75, 1.25)
                a0 = math.atan2(dy[k], dx[k]) if rad[k] > 0.75 else rng.uniform(-2.9, -0.3)
                if rad[k] > 0.75 and dy[k] > 0.3:
                    a0 = math.pi / 2 + rng.normal(0, 0.5)       # drooping leaves on the underside
                _leaf(B, PX[k], PY[k], rr_, a0, col, int(rng.integers(3, 6)), rng, S4)
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
            rr = rng.uniform(3.0, 6.0) * uS
            if sunf > 0.05 and nrm[1] < 0.3:
                iy, ix = int(np.clip(py - nrm[1] * rr * 1.5, 0, hs - 1)), int(np.clip(px - nrm[0] * rr * 1.5, 0, ws - 1))
                vin = float(B[iy, ix, 0])
                o = rr * rng.uniform(0.1, 0.8)
                vv = min(vin + 0.08 + 0.12 * sunf + rng.normal(0, 0.03), 1.0)
                _leaf(B, px + nrm[0] * o, py + nrm[1] * o, rr, math.atan2(nrm[1], nrm[0]) + rng.normal(0, 0.5),
                      (vv, float(B[iy, ix, 1]), float(B[iy, ix, 2]), 1.0), int(rng.integers(3, 6)), rng, S4)
            i += max(int(rr * rng.uniform(0.6, 1.1)), 1)
    # ---- small sky holes near the shade-side outer edge, only where the sky is behind the crown
    A = B[..., 3].copy()
    k1, k2 = max(int(8 * uS), 2), max(int(40 * uS), 4)
    Ab = (A > 0.6).astype(np.uint8)
    ring = cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1,) * 2)) - \
        cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1,) * 2))
    ys_, xs_ = np.nonzero(ring > 0)
    HM = np.zeros((hs, ws), np.float32)
    nh = int(len(ys_) / (1800 * uS * uS))
    for _ in range(nh):
        if len(ys_) == 0:
            break
        k = int(rng.integers(0, len(ys_)))
        hx_, hy_ = xs_[k], ys_[k]
        if B[hy_, hx_, 0] > 0.55 or Tg[hy_, hx_] > 0.6:
            continue
        if clipS is not None and clipS[hy_, hx_] > 0.05:
            continue
        hr = rng.uniform(2.2, 4.2) * uS
        for _k in range(int(rng.integers(1, 3))):
            _leaf(HM, hx_ + rng.normal(0, hr * 0.5), hy_ + rng.normal(0, hr * 0.5), hr, rng.uniform(0, 6.28), 1.0,
                  int(rng.integers(3, 5)), rng, S4)
    # ---- to colour: downsample premultiplied value channels, then one continuous ramp
    small = cv2.resize(B, (w, h), interpolation=cv2.INTER_AREA)
    A1 = small[..., 3]
    inv = 1.0 / np.maximum(A1, 1e-5)
    V = small[..., 0] * inv
    UND = small[..., 1] * inv
    SWm = small[..., 2] * inv
    V = cv2.GaussianBlur(V, (0, 0), 0.45 * u)       # a touch of paint softness, no pixel noise
    col = ramp(V)
    # shadow hue: teal on the sky-lit upper flanks, violet toward the underside; fades out in the light
    shadow_w = C.smoothstep(0.45, 0.1, V)[..., None]
    Tg1 = cv2.resize(Tg, (w, h), interpolation=cv2.INTER_AREA)
    up = np.clip(1 - UND, 0, 1) * C.smoothstep(-0.6, 0.2, Tg1)
    col += (SHD_SKY - col) * (0.45 * up)[..., None] * shadow_w
    col += (SHD_LOW - col) * (0.8 * UND)[..., None] * shadow_w
    # aerial haze on the far (lower-right) clumps: from the clump f stored implicitly via position -> use
    # the tree-level Tdown / distance along the crown ridge is not tracked, so tint by the lit/unlit level
    # ---- branch layer under the leaves
    bsm = cv2.resize(np.dstack([BR * BA[..., None], BA]), (w, h), interpolation=cv2.INTER_AREA)
    ba = bsm[..., 3]
    # ---- warm rim on the sun-facing silhouette (1-2 px, broken, only on lit edges)
    Au = A1.copy()
    shx = np.float32([[1, 0, -Lx * 2.2 * u], [0, 1, -Ly * 2.2 * u]])
    edge = np.clip(Au - cv2.warpAffine(Au, shx, (w, h)), 0, 1)
    Abl1 = cv2.GaussianBlur(Au, (0, 0), 5 * u)
    gy1, gx1 = np.gradient(Abl1)
    gm = np.hypot(gx1, gy1) + 1e-6
    facing = C.smoothstep(0.25, 0.75, -(gx1 * Lx + gy1 * Ly) / gm)
    rim = edge * facing * C.smoothstep(0.3, 0.6, V)
    col += (RIM - col) * np.clip(1.1 * rim, 0, 1)[..., None]
    # ---- lost underside of the whole crown (soft alpha where the silhouette faces down, in shade)
    softA = cv2.GaussianBlur(A1, (0, 0), 2.2 * u)
    dn = C.smoothstep(0.45, 0.9, gy1 / gm) * C.smoothstep(0.45, 0.25, V)
    A1 = A1 * (1 - dn) + np.minimum(A1, softA * softA * 1.25) * dn
    # sky holes
    hm_small = cv2.resize(np.clip(HM, 0, 1), (w, h), interpolation=cv2.INTER_AREA)
    A1 = A1 * (1 - hm_small)
    # composite leaves over branches
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
