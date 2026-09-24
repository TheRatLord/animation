"""s06_seaside foliage, round 7: painted leaf-dab canopy (replaces the sphere-shaded, hard-thresholded clump
painter for the big hillside tree).

The mass is painted like a background artist would, in layers, all anti-aliased at 3x supersampling:
  1. silhouette = union of clump cores + crown lobes + a leaf-cluster fringe whose dab size varies along the
     outline (big fringe clusters on the sun-facing crowns, fine ones between them, sparse and soft on the
     down / shade side); a few sky holes are punched near the outer edge;
  2. 2-3 BROAD cool value masses inside (deep blue-green core low down, cool green body, a sky-lit teal cap
     on the upward shade-side flanks), blended through a blurred leaf-cluster field so the borders are
     lumpy but never speckled;
  3. sparse directional leaf-cluster strokes inside the shadow (drooping, slightly lighter / darker);
  4. warm light built up as stamped leaf clusters where the form faces the sun: half-tone -> lit -> hot,
     each layer smaller and rarer (cauliflower crowns with scalloped lit edges);
  5. a thin broken HDR rim on the sun-facing silhouette only; the lower / shade-side edge is lost (soft alpha).
Deterministic."""
import math
import numpy as np
import cv2

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


PAL3 = dict(deep=cc('#0e2026'), shd=cc('#163234'), mid=cc('#1f4541'), sky=cc('#35616c'), half=cc('#7a5a48'),
            lit=cc('#cf7a40'), hot=cc('#ffbd6c'), rim=np.array([1.75, 1.08, 0.56], np.float32),
            haze=cc('#6a5a8a'), leafd=cc('#0c1e26'), leafl=cc('#34645c'))


def _n3(x, y, z):
    n = np.sqrt(x * x + y * y + z * z) + 1e-6
    return x / n, y / n, z / n


def _cluster(m, x, y, r, ang, val=255, nleaf=None, rng=None):
    """a leaf cluster: 3-5 pointed leaves fanned around (x, y), drawn into uint8 mask m (shift 2, AA)."""
    k = nleaf if nleaf is not None else 4
    for i in range(k):
        a = ang + (i - (k - 1) / 2) * (2.2 / max(k - 1, 1)) + (rng.normal(0, 0.2) if rng is not None else 0)
        cx, cy = x + math.cos(a) * r * 0.5, y + math.sin(a) * r * 0.5
        cv2.ellipse(m, (int(cx * 4), int(cy * 4)), (max(int(r * 0.58 * 4), 2), max(int(r * 0.27 * 4), 1)),
                    math.degrees(a), 0, 360, val, -1, cv2.LINE_AA, 2)


def canopy3(cv, clumps, rng, unit=1.0, clip=None, L=(0.88, -0.3, -0.3), pal=PAL3, ss=3, far_haze=0.22,
            branches=(), rim_amt=1.0, holes=1.0, leaf=1.0):
    """clumps: (x, y, r, f, clipped) canvas px (back to front); f = 0 near .. 1 far.
    branches: [(polyline, base width px), ...] painted first (they read in the gaps / sky holes)."""
    u = max(unit, 0.3)
    cl = np.array([c[:4] for c in clumps], np.float64)
    pad = 18 * u
    X0 = int(max(math.floor((cl[:, 0] - cl[:, 2] * 1.35).min() - pad), 0))
    Y0 = int(max(math.floor((cl[:, 1] - cl[:, 2] * 1.35).min() - pad), 0))
    X1 = int(min(math.ceil((cl[:, 0] + cl[:, 2] * 1.35).max() + pad), cv.W))
    Y1 = int(min(math.ceil((cl[:, 1] + cl[:, 2] * 1.35).max() + pad), cv.H))
    for br in branches:
        p = np.array(br[0])
        X0 = int(max(min(X0, p[:, 0].min() - pad), 0)); Y0 = int(max(min(Y0, p[:, 1].min() - pad), 0))
        X1 = int(min(max(X1, p[:, 0].max() + pad), cv.W)); Y1 = int(min(max(Y1, p[:, 1].max() + pad), cv.H))
    w, h = X1 - X0, Y1 - Y0
    if w < 3 or h < 3:
        return None
    S = float(ss)
    hs, ws = h * ss, w * ss
    Lx, Ly, Lz = _n3(*L)
    Lxy = np.array([Lx, Ly]) / (math.hypot(Lx, Ly) + 1e-9)
    P = {k: np.asarray(v, np.float32) for k, v in pal.items()}
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
    # ---- branches
    for pts, wd in branches:
        p = np.array(pts, np.float64)
        n = len(p)
        m = np.zeros((hs, ws), np.uint8)
        for i in range(n - 1):
            t = i / max(n - 1, 1)
            th = max(wd * (1 - 0.7 * t) * S, 1.0)
            a_, b_ = (p[i] - [X0, Y0]) * S, (p[i + 1] - [X0, Y0]) * S
            cv2.line(m, (int(a_[0] * 4), int(a_[1] * 4)), (int(b_[0] * 4), int(b_[1] * 4)), 255, int(round(th)),
                     cv2.LINE_AA, 2)
        mf = m.astype(np.float32) / 255
        RGB[:] = RGB * (1 - mf[..., None]) + P['leafd'] * mf[..., None]
        A[:] = np.maximum(A, mf)
    # low-frequency fields shared by the whole mass
    seed = int(rng.integers(0, 1 << 30))
    q = 4
    nlo = cv2.resize(C.fbm(max(ws // q, 8), max(hs // q, 8), max(ws / (140 * u * S), 2), 3, seed=seed),
                     (ws, hs), interpolation=cv2.INTER_CUBIC)
    nmid = cv2.resize(C.fbm(max(ws // q, 8), max(hs // q, 8), max(ws / (40 * u * S), 2), 3, seed=seed + 1),
                      (ws, hs), interpolation=cv2.INTER_CUBIC)
    # global form: blurred union of the clump ellipses
    G = np.zeros((hs, ws), np.float32)
    for (x, y, r, f, clipped) in clumps:
        cv2.ellipse(G, (int((x - X0) * S * 4), int((y - Y0) * S * 4)), (int(r * S * 1.1 * 4), int(r * S * 0.85 * 4)),
                    0, 0, 360, 1.0, -1, cv2.LINE_AA, 2)
    gsig = max(float(np.median(cl[:, 2])) * S * 0.9, 4)
    pb = int(2 * gsig) + 1
    Gb = cv2.GaussianBlur(cv2.copyMakeBorder(G, pb, pb, pb, pb, cv2.BORDER_CONSTANT, value=0), (0, 0), gsig)[pb:-pb, pb:-pb]
    ggy, ggx = np.gradient(Gb)
    gg = np.hypot(ggx, ggy)
    gsc_ = 1.0 / (np.percentile(gg[G > 0.5], 90) + 1e-6) if (G > 0.5).any() else 1.0
    Gnx, Gny, Gnz = _n3(-ggx * gsc_, -ggy * gsc_, np.full_like(ggx, 0.75))
    for ci, (x, y, r, f, clipped) in enumerate(clumps):
        R = r * S
        cx, cy = (x - X0) * S, (y - Y0) * S
        bb = int(R * 1.7 + 24 * u * S)
        ya, yb = max(int(cy - bb), 0), min(int(cy + bb), hs)
        xa, xb = max(int(cx - bb), 0), min(int(cx + bb), ws)
        if yb <= ya or xb <= xa:
            continue
        hh, ww = yb - ya, xb - xa
        lx, ly = cx - xa, cy - ya
        # ---- silhouette: core + crown lobes of varied size (bigger toward the sun / top)
        M = np.zeros((hh, ww), np.uint8)
        cv2.ellipse(M, (int(lx * 4), int((ly + 0.12 * R) * 4)), (int(R * 0.95 * 4), int(R * 0.62 * 4)), 0, 0, 360,
                    255, -1, cv2.LINE_AA, 2)
        lobes = []
        for j in range(int(rng.integers(5, 9))):
            a = -math.pi / 2 + rng.uniform(-1.65, 1.65)
            d = R * rng.uniform(0.35, 0.7)
            sunf = max(math.cos(a) * Lxy[0] + math.sin(a) * Lxy[1], 0)
            rr = R * rng.uniform(0.2, 0.42) * (1 + 0.3 * sunf)
            lobes.append((lx + math.cos(a) * d * 1.15, ly + math.sin(a) * d * 0.85, rr))
        for j in range(int(rng.integers(6, 11))):
            a = -math.pi / 2 + rng.uniform(-1.7, 1.7)
            d = R * rng.uniform(0.72, 0.92)
            lobes.append((lx + math.cos(a) * d * 1.12, ly + math.sin(a) * d * 0.78 + 0.05 * R,
                          R * rng.uniform(0.09, 0.19)))
        lobes.sort(key=lambda l: l[1])
        CR = np.zeros((hh, ww), np.uint8)     # sun-side crescents of every lobe (cauliflower lit caps)
        for (sx_, sy_, sr_) in lobes:
            an_ = float(rng.uniform(-20, 20))
            cv2.ellipse(M, (int(sx_ * 4), int(sy_ * 4)), (int(sr_ * 4), int(sr_ * 0.86 * 4)),
                        an_, 0, 360, 255, -1, cv2.LINE_AA, 2)
            e1 = np.zeros((hh, ww), np.uint8)
            cv2.ellipse(e1, (int(sx_ * 4), int(sy_ * 4)), (int(sr_ * 4), int(sr_ * 0.86 * 4)),
                        an_, 0, 360, 255, -1, cv2.LINE_AA, 2)
            d_ = sr_ * rng.uniform(0.55, 0.85)
            e2 = np.zeros((hh, ww), np.uint8)
            cv2.ellipse(e2, (int((sx_ - Lxy[0] * d_) * 4), int((sy_ - Lxy[1] * d_ + 0.25 * d_) * 4)),
                        (int(sr_ * 1.02 * 4), int(sr_ * 0.9 * 4)), an_, 0, 360, 255, -1, cv2.LINE_AA, 2)
            CR = np.maximum(np.minimum(CR, 255 - e1), np.minimum(e1, 255 - e2))
        # leaf-cluster fringe along the outline: size varies along the contour (big florets / fine leaves)
        Mb = cv2.GaussianBlur(M.astype(np.float32), (0, 0), max(0.1 * R, 2))
        gy_, gx_ = np.gradient(Mb)
        cnts, _ = cv2.findContours((M > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        base = 3.2 * u * S * leaf * (1 - 0.45 * f)
        for c in cnts:
            c = c[:, 0, :]
            nC = len(c)
            ph = rng.uniform(0, 6.28)
            i = 0
            while i < nC:
                px, py = c[i]
                g = np.array([gx_[py, px], gy_[py, px]])
                nrm = -g / (np.linalg.norm(g) + 1e-9)
                sunf = nrm[0] * Lxy[0] + nrm[1] * Lxy[1]
                down = nrm[1]
                # floret scale modulation along the contour
                mod = 0.55 + 0.45 * math.sin(i * 2 * math.pi / max(nC / rng.uniform(5, 9), 8) + ph)
                rr = base * (0.7 + 1.9 * mod * rng.uniform(0.6, 1.2)) * (1.0 + 0.35 * max(sunf, 0))
                if down > 0.45 and rng.random() < 0.55:
                    i += max(int(rr * 0.8), 1)
                    continue
                o = rr * rng.uniform(-0.2, 0.7)
                ang = math.atan2(nrm[1], nrm[0]) + rng.normal(0, 0.45)
                if down > 0.3:
                    ang = ang * 0.5 + math.pi / 2 * 0.5      # drooping leaves on the underside
                _cluster(M, px + nrm[0] * o, py + nrm[1] * o, rr, ang, nleaf=int(rng.integers(3, 6)), rng=rng)
                i += max(int(rr * rng.uniform(0.6, 1.1)), 1)
        # sky holes near the outer edge (a few, leaf-cluster shaped, not on the lit crown)
        if False:
            Md = M.astype(np.float32) / 255
            inner = cv2.erode(Md, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.12 * R) * 2 + 1,) * 2))
            ring = np.clip(Md - cv2.erode(Md, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.3 * R) * 2 + 1,) * 2)), 0, 1) * inner
            ys_, xs_ = np.nonzero(ring > 0.5)
            if len(ys_):
                nh = int(rng.integers(1, 4) * holes)
                for _ in range(nh):
                    k = int(rng.integers(0, len(ys_)))
                    hx_, hy_ = xs_[k], ys_[k]
                    g = np.array([gx_[hy_, hx_], gy_[hy_, hx_]])
                    nrm = -g / (np.linalg.norm(g) + 1e-9)
                    if nrm[0] * Lxy[0] + nrm[1] * Lxy[1] > 0.3:
                        continue
                    hr = R * rng.uniform(0.05, 0.1)
                    hm_ = np.zeros_like(M)
                    for _k in range(int(rng.integers(2, 4))):
                        _cluster(hm_, hx_ + rng.normal(0, hr * 0.6), hy_ + rng.normal(0, hr * 0.6), hr,
                                 rng.uniform(0, 6.28), nleaf=int(rng.integers(3, 5)), rng=rng)
                    M = np.minimum(M, 255 - hm_)
        Mf = M.astype(np.float32) / 255.0
        if clipped and clipS is not None:
            Mf = Mf * clipS[ya:yb, xa:xb]
        if not (Mf > 0.5).any():
            continue
        # ---- form: clump-level normal (blurred silhouette) + global mass normal
        Bf = cv2.GaussianBlur(Mf, (0, 0), max(0.4 * R, 2))
        fy, fx = np.gradient(Bf)
        gsc = 1.0 / (np.percentile(np.hypot(fx, fy)[Mf > 0.5], 90) + 1e-6)
        fnx, fny, fnz = _n3(-fx * gsc, -fy * gsc, np.full_like(fx, 0.9))
        gl_ = (slice(ya, yb), slice(xa, xb))
        nx, ny, nz = _n3(0.5 * Gnx[gl_] + 0.5 * fnx, 0.5 * Gny[gl_] + 0.5 * fny, 0.5 * Gnz[gl_] + 0.5 * fnz)
        # lumpy (leaf-cluster scale) but smooth modulation for the value boundaries
        lump = nmid[gl_] - 0.5
        crf = cv2.GaussianBlur(CR.astype(np.float32) / 255, (0, 0), max(1.2 * u * S, 1))
        lam = -0.08 + (nx * Lx + ny * Ly + nz * Lz) * 1.5 + 0.3 * lump + 0.26 * crf - 0.1 * (1 - crf) * (lump + 0.5)
        # ---- broad cool masses: deep core low down, cool green body, sky-lit teal cap on upward flanks
        col = np.broadcast_to(P['mid'], (hh, ww, 3)).copy()
        lowf = C.smoothstep(0.0, 0.75, 0.3 * fny + 0.7 * Gny[gl_] + 0.55 * lump)
        col += (P['shd'] - col) * C.smoothstep(0.35, 0.65, nlo[gl_] + 0.3 * fny)[..., None] * 0.7
        col += (P['deep'] - col) * (0.9 * lowf)[..., None]
        capf = C.smoothstep(0.15, 0.55, -0.35 * fny - 0.45 * Gny[gl_] - 0.3 * nx * Lxy[0] + 0.5 * lump) * C.smoothstep(0.3, 0.0, lam)
        col += (P['sky'] - col) * (0.75 * capf)[..., None]
        # ---- sparse directional leaf strokes inside the shadow (drooping clusters, lighter and darker)
        nstk = int(hh * ww / (110 * (u * S) ** 2) * leaf)
        if nstk > 0:
            mL = np.zeros((hh, ww), np.uint8)
            mD = np.zeros((hh, ww), np.uint8)
            for _ in range(nstk):
                px_, py_ = rng.uniform(0, ww), rng.uniform(0, hh)
                iy, ix = int(py_), int(px_)
                if Mf[iy, ix] < 0.9 or lam[iy, ix] > 0.25:
                    continue
                rr = base * rng.uniform(0.7, 1.5)
                tgt = mL if (capf[iy, ix] > 0.2 or rng.random() < 0.35) and lowf[iy, ix] < 0.6 else mD
                _cluster(tgt, px_, py_, rr, math.pi / 2 + rng.normal(0, 0.5), nleaf=int(rng.integers(3, 5)), rng=rng)
            kL = mL.astype(np.float32) / 255 * 0.55
            kD = mD.astype(np.float32) / 255 * 0.5
            col += (P['leafl'] * 0.9 + col * 0.1 - col) * kL[..., None]
            col += (P['leafd'] - col) * kD[..., None]
        # ---- warm light: stamped leaf clusters where the form faces the sun (half -> lit -> hot)
        ncl = int(hh * ww / (40 * (u * S) ** 2) * leaf)
        ptsx = rng.uniform(0, ww, ncl)
        ptsy = rng.uniform(0, hh, ncl)
        ix, iy = ptsx.astype(int), ptsy.astype(int)
        ok = Mf[iy, ix] > 0.5
        lv = lam[iy, ix]
        prev = 'mid'
        for thr, key, sz, amt in ((0.14, 'half', 1.3, 0.9), (0.32, 'lit', 1.1, 1.0), (0.66, 'hot', 0.8, 1.0)):
            mm = np.zeros((hh, ww), np.uint8)
            sel = np.nonzero(ok & (lv > thr - 0.07) & (lv < thr + 0.12))[0]
            for k in sel:
                rr = base * sz * rng.uniform(0.6, 1.4)
                _cluster(mm, ptsx[k], ptsy[k], rr, rng.uniform(-2.6, -0.5), nleaf=int(rng.integers(3, 6)), rng=rng)
            z = np.maximum(mm.astype(np.float32) / 255, C.smoothstep(thr + 0.06, thr + 0.09, lam)) * amt
            # leaf texture inside the lit zone: sparse clusters of the value below (shadowed leaves)
            mi = np.zeros((hh, ww), np.uint8)
            sel = np.nonzero(ok & (lv > thr + 0.14))[0]
            sel = sel[rng.random(len(sel)) < 0.22]
            for k in sel:
                rr = base * sz * rng.uniform(0.5, 1.1)
                _cluster(mi, ptsx[k], ptsy[k], rr, rng.uniform(0.5, 2.6), nleaf=int(rng.integers(2, 4)), rng=rng)
            z = z * (1 - 0.8 * mi.astype(np.float32) / 255)
            col += (P[key] - col) * z[..., None]
            prev = key
        # ---- rim on the sun-facing silhouette only (sub-pixel shift -> anti-aliased), broken by leaf lumps
        sh = np.float32([[1, 0, -Lxy[0] * 1.9 * u * S], [0, 1, -Lxy[1] * 1.9 * u * S]])
        edge = np.clip(Mf - cv2.warpAffine(Mf, sh, (ww, hh), flags=cv2.INTER_LINEAR), 0, 1)
        facing = C.smoothstep(0.05, 0.5, fnx * Lxy[0] + fny * Lxy[1])
        rim = edge * facing * C.smoothstep(0.4, 0.55, nmid[gl_]) * rim_amt * (1 - 0.6 * f)
        col += (P['rim'] - col) * rim[..., None]
        col += (P['haze'] - col) * (far_haze * f ** 1.3)
        # ---- lost lower / shade-side edge
        soft = cv2.GaussianBlur(Mf, (0, 0), max(0.045 * R, 2.5 * u * S))
        down = C.smoothstep(0.1, 0.6, fny - 0.5 * (fnx * Lxy[0])) * 0.75
        a = Mf * (1 - down) + np.minimum(Mf, soft * soft) * down
        sl = (slice(ya, yb), slice(xa, xb))
        # soft cast shadow of this clump onto the mass behind (clump hierarchy, away from the sun)
        dsh = 0.1 * R
        cs_ = cv2.warpAffine(Mf, np.float32([[1, 0, -Lxy[0] * dsh], [0, 1, -Lxy[1] * dsh + 0.03 * R]]), (ww, hh))
        cs_ = cv2.GaussianBlur(cs_, (0, 0), 0.06 * R) * (1 - a)
        RGB[sl] *= (1 - 0.45 * cs_)[..., None]
        RGB[sl] = RGB[sl] * (1 - a[..., None]) + col * a[..., None]
        A[sl] = A[sl] * (1 - a) + a
    # ---- sky holes near the outer silhouette of the whole mass (shade / upper side, not the lit crown)
    if holes > 0:
        Ab = (A > 0.5).astype(np.uint8)
        k1 = max(int(10 * u * S), 2)
        k2 = max(int(45 * u * S), 4)
        inner = cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k1 + 1,) * 2))
        ring = inner - cv2.erode(Ab, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1,) * 2))
        Abl = cv2.GaussianBlur(Ab.astype(np.float32), (0, 0), 12 * u * S)
        gy_, gx_ = np.gradient(Abl)
        ys_, xs_ = np.nonzero(ring > 0)
        hm_ = np.zeros((hs, ws), np.uint8)
        nh = int(len(ys_) / (900 * (u * S) ** 2) * holes)
        for _ in range(nh):
            k = int(rng.integers(0, len(ys_)))
            hx_, hy_ = xs_[k], ys_[k]
            g = np.array([gx_[hy_, hx_], gy_[hy_, hx_]])
            nrm = -g / (np.linalg.norm(g) + 1e-9)
            if nrm[0] * Lxy[0] + nrm[1] * Lxy[1] > -0.15:
                continue
            hr = rng.uniform(2.5, 6.0) * u * S
            for _k in range(int(rng.integers(1, 4))):
                _cluster(hm_, hx_ + rng.normal(0, hr * 0.7), hy_ + rng.normal(0, hr * 0.7), hr,
                         rng.uniform(0, 6.28), nleaf=int(rng.integers(3, 5)), rng=rng)
        A = A * (1 - hm_.astype(np.float32) / 255)
    prem = np.concatenate([RGB * A[..., None], A[..., None]], -1)
    small = cv2.resize(prem, (w, h), interpolation=cv2.INTER_AREA)
    al = small[..., 3]
    colr = small[..., :3] / np.maximum(al, 1e-5)[..., None]
    cv.put((al, X0, Y0), colr)
    return (al, X0, Y0)
