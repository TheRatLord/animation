"""s06_seaside foliage, round 6: clump-hierarchy painter (replaces the per-leaf-cluster value mosaic).

A foliage mass is painted as a short list of cauliflower CLUMPS, back to front. Each clump:
  - silhouette = core + 5-8 crown sub-lobes (upper / sun half) + leaf-dab scallops along the outline
    (dense on the top / sun side, sparse on the down side);
  - ONE broad shadow value for the body, a deep underside, a cool sky-lit cap on top, and a warm lit crown on
    the sun side (half / lit / hot steps). Value boundaries are scalloped by a shared leaf-dab field, so the
    interior stays broad and only the boundaries carry leaf texture;
  - a thin broken HDR rim on the sun-facing silhouette only;
  - a lost (soft alpha, deep) lower edge into the clump below.
Optional dark branches are painted first so they read in the gaps under the clumps. Deterministic."""
import math
import numpy as np
import cv2

from lib import core as C


def cc(h):
    return C.hex2rgb(h)


PAL2 = dict(deep=cc('#0b1820'), shd=cc('#173038'), sky=cc('#2a4c5c'), half=cc('#5e4450'), lit=cc('#c07442'),
            hot=cc('#ffb466'), rim=np.array([1.8, 1.08, 0.55], np.float32), haze=cc('#6a5a8a'))
PAL2_DARK = dict(deep=cc('#070d16'), shd=cc('#0e1826'), sky=cc('#1a2a3c'), half=cc('#3e3040'), lit=cc('#8a5434'),
                 hot=cc('#dc8a50'), rim=np.array([1.55, 0.88, 0.46], np.float32), haze=cc('#4a4468'))


def _norm3(x, y, z):
    n = np.sqrt(x * x + y * y + z * z) + 1e-6
    return x / n, y / n, z / n


def _dab_field(h, w, rng, rmin, rmax, n):
    """piecewise-constant field of overlapping small leaf-shaped dabs with random values in [-1, 1]."""
    f = np.zeros((h, w), np.float32)
    xs = rng.uniform(0, w, n)
    ys = rng.uniform(0, h, n)
    rs = rmin + (rmax - rmin) * rng.uniform(0, 1, n) ** 1.5
    vs = rng.uniform(-1, 1, n)
    an = rng.uniform(0, 180, n)
    for x, y, r, v, a in zip(xs, ys, rs, vs, an):
        cv2.ellipse(f, (int(x * 4), int(y * 4)), (max(int(r * 4), 1), max(int(r * 2.8), 1)),
                    float(a), 0, 360, float(v), -1, cv2.LINE_8, 2)
    return f


def _leaf_dab(m, x, y, r, ang, val=255):
    """one leaf cluster: 3 small pointed ellipses fanned around (x, y) (uint8 mask)."""
    for k in range(3):
        a = ang + (k - 1) * 0.9
        cx, cy = x + math.cos(a) * r * 0.45, y + math.sin(a) * r * 0.45
        cv2.ellipse(m, (int(cx * 4), int(cy * 4)), (max(int(r * 0.62 * 4), 1), max(int(r * 0.34 * 4), 1)),
                    math.degrees(a), 0, 360, val, -1, cv2.LINE_AA, 2)


def canopy2(cv, clumps, rng, unit=1.0, clip=None, L=(0.88, -0.28, -0.3), pal=PAL2, ss=2, far_haze=0.5,
            leaf=1.0, lit_bias=0.0, branches=(), rim_amt=1.0, sky_amt=1.0, soft_down=1.0, hang=0.8):
    """clumps: (x, y, r, f, clipped) canvas px, painted in the given order (back to front); f = 0 near .. 1 far.
    branches: [(polyline, base width px), ...] painted before the clumps."""
    u = max(unit, 0.3)
    cl = np.array([c[:4] for c in clumps], np.float64)
    pad = 16 * u
    X0 = int(max(math.floor((cl[:, 0] - cl[:, 2] * 1.25).min() - pad), 0))
    Y0 = int(max(math.floor((cl[:, 1] - cl[:, 2] * 1.25).min() - pad), 0))
    X1 = int(min(math.ceil((cl[:, 0] + cl[:, 2] * 1.25).max() + pad), cv.W))
    Y1 = int(min(math.ceil((cl[:, 1] + cl[:, 2] * 1.25).max() + pad), cv.H))
    for br in branches:
        p = np.array(br[0])
        X0 = int(max(min(X0, p[:, 0].min() - pad), 0)); Y0 = int(max(min(Y0, p[:, 1].min() - pad), 0))
        X1 = int(min(max(X1, p[:, 0].max() + pad), cv.W)); Y1 = int(min(max(Y1, p[:, 1].max() + pad), cv.H))
    w, h = X1 - X0, Y1 - Y0
    if w < 3 or h < 3:
        return None
    hs, ws = h * ss, w * ss
    S = float(ss)
    Lx, Ly, Lz = _norm3(*L)
    Lxy = np.array([Lx, Ly]) / (math.hypot(Lx, Ly) + 1e-9)
    RGB = np.zeros((hs, ws, 3), np.float32)
    A = np.zeros((hs, ws), np.float32)
    clipS = None
    if clip is not None:
        cm, cx0, cy0 = clip
        c_ = np.zeros((h, w), np.float32)
        a0, b0 = max(Y0, cy0), max(X0, cx0)
        a1, b1 = min(Y1, cy0 + cm.shape[0]), min(X1, cx0 + cm.shape[1])
        if a1 > a0 and b1 > b0:
            c_[a0 - Y0:a1 - Y0, b0 - X0:b1 - X0] = cm[a0 - cy0:a1 - cy0, b0 - cx0:b1 - cx0]
        clipS = cv2.resize(c_, (ws, hs), interpolation=cv2.INTER_LINEAR)
    dr = 5.0 * u * S * leaf
    dab = _dab_field(hs, ws, rng, 0.5 * dr, 1.4 * dr, int(hs * ws / (dr * dr * 1.6)))
    rn = C.fbm(ws, hs, max(int(ws / (16 * u * S)), 2), 2, seed=int(rng.integers(0, 1 << 30)))
    deep, shd, sky, half, lit, hot = [np.asarray(pal[k], np.float32) for k in ('deep', 'shd', 'sky', 'half', 'lit', 'hot')]
    rimc = np.asarray(pal['rim'], np.float32)
    hz = np.asarray(pal['haze'], np.float32)
    # ---- branches (dark, tapered, a thin warm thread on the sun side)
    for pts, wd in branches:
        p = np.array(pts, np.float64)
        n = len(p)
        m = np.zeros((hs, ws), np.uint8)
        m2 = np.zeros((hs, ws), np.uint8)
        for i in range(n - 1):
            t = i / max(n - 1, 1)
            th = max(wd * (1 - 0.75 * t) * S, 1.0)
            a_, b_ = (p[i] - [X0, Y0]) * S, (p[i + 1] - [X0, Y0]) * S
            cv2.line(m, (int(a_[0] * 4), int(a_[1] * 4)), (int(b_[0] * 4), int(b_[1] * 4)), 255, int(round(th)),
                     cv2.LINE_AA, 2)
            o = th * 0.32
            cv2.line(m2, (int((a_[0] + o) * 4), int(a_[1] * 4)), (int((b_[0] + o) * 4), int(b_[1] * 4)), 255,
                     max(int(th * 0.22), 1), cv2.LINE_AA, 2)
        mf = m.astype(np.float32) / 255
        RGB[:] = RGB * (1 - mf[..., None]) + deep * mf[..., None]
        A[:] = A * (1 - mf) + mf
        mf2 = (m2.astype(np.float32) / 255) * mf * 0.6
        RGB[:] = RGB * (1 - mf2[..., None]) + half * mf2[..., None]
    # global form of the whole mass (interior clumps stay in shadow, the outer sun-side edge catches light)
    G = np.zeros((hs, ws), np.float32)
    for (x, y, r, f, clipped) in clumps:
        cv2.ellipse(G, (int((x - X0) * S * 4), int((y - Y0) * S * 4)), (int(r * S * 1.1 * 4), int(r * S * 0.85 * 4)),
                    0, 0, 360, 1.0, -1, cv2.LINE_8, 2)
    if clipS is not None:
        G = np.maximum(G * 0, G)
    gsig = max(float(np.median(cl[:, 2])) * S * 0.9, 4)
    pb = int(2 * gsig) + 1
    Gb = cv2.GaussianBlur(cv2.copyMakeBorder(G, pb, pb, pb, pb, cv2.BORDER_CONSTANT, value=0), (0, 0), gsig)[pb:-pb, pb:-pb]
    ggy, ggx = np.gradient(Gb)
    gg = np.hypot(ggx, ggy)
    gsc_ = 1.0 / (np.percentile(gg[G > 0.5], 90) + 1e-6) if (G > 0.5).any() else 1.0
    Gnx, Gny, Gnz = _norm3(-ggx * gsc_, -ggy * gsc_, np.full_like(ggx, 0.7))
    for (x, y, r, f, clipped) in clumps:
        R = r * S
        cx, cy = (x - X0) * S, (y - Y0) * S
        bb = int(R * 1.6 + 20 * u * S)
        ya, yb = max(int(cy - bb), 0), min(int(cy + bb), hs)
        xa, xb = max(int(cx - bb), 0), min(int(cx + bb), ws)
        if yb <= ya or xb <= xa:
            continue
        hh, ww = yb - ya, xb - xa
        lx, ly = cx - xa, cy - ya
        # core + cauliflower sub-lobes (crown on the upper / sun half)
        subs = [(lx - 0.25 * R, ly + 0.12 * R, 0.6 * R), (lx + 0.25 * R, ly + 0.1 * R, 0.6 * R)]
        nsub = int(rng.integers(4, 7))
        for j in range(nsub):
            a = -math.pi / 2 + rng.uniform(-1.5, 1.5) + 0.15
            d = R * rng.uniform(0.45, 0.7)
            rr = R * rng.uniform(0.22, 0.46) * (1.12 if math.cos(a) * Lxy[0] > 0.3 else 1.0)
            subs.append((lx + math.cos(a) * d * 1.2, ly + math.sin(a) * d * 0.8, rr))
        # mid-scale bumps on the upper outline (cauliflower florets)
        for j in range(int(rng.integers(6, 11))):
            a = -math.pi / 2 + rng.uniform(-1.7, 1.7)
            d = R * rng.uniform(0.75, 0.95)
            rr = R * rng.uniform(0.1, 0.2)
            subs.append((lx + math.cos(a) * d * 1.15, ly + math.sin(a) * d * 0.78 + 0.05 * R, rr))
        subs.append((lx - 0.7 * R, ly + 0.25 * R, 0.36 * R))
        subs.append((lx + 0.65 * R, ly + 0.28 * R, 0.34 * R))
        M = np.zeros((hh, ww), np.uint8)
        sid = np.full((hh, ww), -1, np.int32)
        order = [0, 1] + sorted(range(2, len(subs)), key=lambda k: subs[k][1])
        for k in order:
            sx_, sy_, sr_ = subs[k]
            mm = np.zeros((hh, ww), np.uint8)
            cv2.circle(mm, (int(sx_ * 4), int(sy_ * 4)), int(sr_ * 4), 255, -1, cv2.LINE_8, 2)
            sid[mm > 0] = k
            M = np.maximum(M, mm)
        # leaf-dab scallops along the silhouette
        cnts, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        Mb = cv2.GaussianBlur(M.astype(np.float32), (0, 0), max(0.12 * R, 2))
        gy_, gx_ = np.gradient(Mb)
        step = max(2.4 * u * S * leaf * (1 - 0.4 * f), 1.5)
        for c in cnts:
            c = c[:, 0, :]
            for i in range(0, len(c), max(int(step), 1)):
                px, py = c[i]
                g = np.array([gx_[py, px], gy_[py, px]])
                nrm = -g / (np.linalg.norm(g) + 1e-9)
                if -nrm[1] < -0.35 and rng.random() < hang:
                    continue
                rr = (2.5 + 4.5 * rng.uniform() ** 1.3) * u * S * leaf * (1 - 0.45 * f)
                o = rr * rng.uniform(-0.3, 0.75)
                _leaf_dab(M, px + nrm[0] * o, py + nrm[1] * o, rr, math.atan2(nrm[1], nrm[0]) + rng.normal(0, 0.5))
        Mf = M.astype(np.float32) / 255.0
        if clipped and clipS is not None:
            Mf = Mf * clipS[ya:yb, xa:xb]
        if Mf.max() < 0.01 or not (Mf > 0.5).any():
            continue
        # form normal: blurred clump silhouette mixed with the sub-lobe spheres
        Bf = cv2.GaussianBlur(Mf, (0, 0), max(0.35 * R, 2))
        fy, fx = np.gradient(Bf)
        gsc = 1.0 / (np.percentile(np.hypot(fx, fy)[Mf > 0.5], 90) + 1e-6)
        fnx, fny, fnz = _norm3(-fx * gsc, -fy * gsc, np.full_like(fx, 0.8))
        sxs = np.array([s_[0] for s_ in subs], np.float32)
        sys_ = np.array([s_[1] for s_ in subs], np.float32)
        srs = np.array([s_[2] for s_ in subs], np.float32)
        sidc = np.maximum(sid, 0)
        Y_, X_ = np.mgrid[0:hh, 0:ww].astype(np.float32)
        snx = np.clip((X_ - sxs[sidc]) / srs[sidc], -1, 1)
        sny = np.clip((Y_ - sys_[sidc]) / srs[sidc], -1, 1)
        snz = np.sqrt(np.clip(1 - snx ** 2 - sny ** 2, 0, 1))
        snx = np.where(sid >= 0, snx, fnx); sny = np.where(sid >= 0, sny, fny); snz = np.where(sid >= 0, snz, fnz)
        gl_ = (slice(ya, yb), slice(xa, xb))
        nx, ny, nz = _norm3(0.45 * Gnx[gl_] + 0.33 * fnx + 0.22 * snx, 0.45 * Gny[gl_] + 0.33 * fny + 0.22 * sny,
                            0.45 * Gnz[gl_] + 0.33 * fnz + 0.22 * snz)
        dsub = dab[ya:yb, xa:xb]
        lam = 0.2 + (nx * Lx + ny * Ly + nz * Lz - 0.08 + lit_bias) * 1.7 + 0.1 * dsub
        skyv = -ny * 0.85 - nx * 0.25 + 0.14 * dsub
        col = np.broadcast_to(shd, (hh, ww, 3)).copy()
        low = C.smoothstep(0.3, 0.75, 0.5 * fny + 0.5 * Gny[gl_] + 0.12 * dsub)
        col = col + (deep - col) * (0.8 * low)[..., None]
        # cool sky-lit band along the upper silhouette of each clump (separates it from the mass behind)
        th_ = max(0.055 * R, 2.5 * u * S)
        band = np.clip(Mf - cv2.warpAffine(Mf, np.float32([[1, 0, 0.25 * th_], [0, 1, th_]]), (ww, hh)), 0, 1)
        band = cv2.GaussianBlur(band, (0, 0), max(0.25 * th_, 1))
        capm = ((band * 1.6 + 0.35 * dsub > 0.55) & (lam < 0.2) & (fny < 0.1)).astype(np.float32) * sky_amt * (
            C.smoothstep(0.4, 0.5, rn[ya:yb, xa:xb]) * 0.75)
        capm = np.maximum(capm, ((skyv > 0.62) & (lam < 0.2) & (fny < -0.4)).astype(np.float32) * sky_amt)
        col = col + (sky - col) * (capm * 0.85)[..., None]
        for thr, cc_ in ((0.2, half), (0.36, lit), (0.56, hot)):
            z = (lam > thr).astype(np.float32)
            col = col + (cc_ - col) * z[..., None]
        # rim on the sun-facing silhouette only
        sh = np.float32([[1, 0, -Lxy[0] * 1.7 * u * S], [0, 1, -Lxy[1] * 1.7 * u * S]])
        edge = np.clip(Mf - cv2.warpAffine(Mf, sh, (ww, hh)), 0, 1)
        facing = C.smoothstep(0.0, 0.45, fnx * Lxy[0] + fny * Lxy[1])
        rim = edge * facing * C.smoothstep(0.35, 0.5, rn[ya:yb, xa:xb]) * rim_amt * (1 - 0.6 * f)
        col = col + (rimc - col) * rim[..., None]
        col = col + (hz - col) * (far_haze * f ** 1.3)
        # lost lower edge
        soft = cv2.GaussianBlur(Mf, (0, 0), max(0.035 * R, 2 * u * S))
        down = C.smoothstep(0.15, 0.6, fny) * soft_down
        a = Mf * (1 - down) + np.minimum(Mf, soft * soft) * down
        sl = (slice(ya, yb), slice(xa, xb))
        RGB[sl] = RGB[sl] * (1 - a[..., None]) + col * a[..., None]
        A[sl] = A[sl] * (1 - a) + a
    prem = np.concatenate([RGB * A[..., None], A[..., None]], -1)
    small = cv2.resize(prem, (w, h), interpolation=cv2.INTER_AREA)
    al = small[..., 3]
    colr = small[..., :3] / np.maximum(al, 1e-5)[..., None]
    cv.put((al, X0, Y0), colr)
    return (al, X0, Y0)
