"""Round-17 night clouds for s07_comet_night: FLAT painted yn_05 cloud shapes.

Reviewer notes on round 16: the upper cloudlets were soft 3D cotton/clay puffs (airbrushed interior, pale
rounded top band that followed every bump like an extrusion, fuzzy halo rim); the diagonal streaks were bead
chains of small round puffs; the horizon clouds had a mushy translucent grey top with a stepped outline.

This painter works like a background painter with a flat brush instead of building circles:
  * silhouette = one smooth 'mass' field made of a FEW big anisotropic lobes (sizes varied, ends tapered),
    torn by multi-octave value noise.  Round cauliflower bumps are added on the sky-facing (upper) side
    only; the underside is torn along the wind by stretched streak noise.  No circle unions -> no beads.
  * 2-3 hard-edged FLAT values: a pale moonlit top plane (light side, upper-left) whose lower boundary is a
    crisp, brush-jagged terminator that follows the big masses (not every bump: on the lee side the top edge
    falls straight into the mid value), one flat mid blue-grey body, and a darker blue-violet base.
    No Gaussian interior gradient, no rim/halo: the silhouette is a 1px anti-aliased edge in the plane's
    own value.
  * lost edges underneath: the down-facing edge fades over a long distance and the base turns translucent
    (stars show through).
  * streaks: long torn bands with 1-3 larger lobes riding on them, tapering to wisps at both ends.
  * horizon afterglow clouds: crisp, slightly lighter violet top edge, a flat saturated orange underside,
    soft lost edge at the bottom.
Returns straight RGBA (Hp, Wp, 4) in plate px.
"""
import math

import numpy as np
import cv2


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _vn(w, h, cx, cy, rng):
    """Smooth value noise 0..1 with cell size cx, cy px (cubic-interpolated random lattice)."""
    cx, cy = max(cx, 1.5), max(cy, 1.5)
    gw, gh = int(w / cx) + 4, int(h / cy) + 4
    g = rng.random((gh, gw)).astype(np.float32)
    bw, bh = int(round(gw * cx)), int(round(gh * cy))
    big = cv2.resize(g, (bw, bh), interpolation=cv2.INTER_CUBIC)
    ox, oy = int(cx), int(cy)
    out = big[oy:oy + h, ox:ox + w]
    return 0.5 + 0.25 * (out - out.mean()) / (out.std() + 1e-6)


def _fbm(w, h, cx, cy, octaves, rng, gain=0.5):
    out = np.zeros((h, w), np.float32)
    a, tot = 1.0, 0.0
    for o in range(octaves):
        out += a * _vn(w, h, cx / 2 ** o, cy / 2 ** o, rng)
        tot += a
        a *= gain
    out /= tot
    if out.size > 16:
        out = 0.5 + 0.25 * (out - out.mean()) / (out.std() + 1e-6)   # unit-ish contrast (std 0.25)
    return out


def _depth_top(inside):
    h, w = inside.shape
    idx = np.arange(h, dtype=np.float32)[:, None]
    last_out = np.maximum.accumulate(np.where(inside, -1.0, idx), axis=0)
    return np.where(inside, idx - last_out - 0.5, 0.0).astype(np.float32)


# ----------------------------------------------------------------------------- local-frame shape fields
def _mass_cu(rng, L, T, w, h, u, v):
    """Compact moonlit cloud: 2-4 big lobes, tallest near the middle, flat torn base.  u, v local px
    (u along the axis, v down).  Returns mass field ~0..1+."""
    n = int(rng.integers(3, 6))
    us = np.sort(rng.uniform(-0.4, 0.4, n)) * L
    M = np.zeros((h, w), np.float32)
    for i, cu in enumerate(us):
        q = abs(cu) / (0.5 * L + 1e-3)
        big = rng.uniform(0.6, 1.0) * (1 - 0.45 * q)
        ru = (0.2 + 0.14 * rng.random()) * L * (0.6 + 0.4 * big)
        rv = T * (0.3 + 0.3 * big * rng.uniform(0.6, 1.0))
        cv = 0.12 * T - 0.55 * rv          # lobes sit on the base line (v=0.12T) and rise from it
        g = np.exp(-(((u - cu) / ru) ** 2 + ((v - cv) / rv) ** 2) * 1.2)
        M = np.maximum(M, g) + 0.35 * np.minimum(M, g)
    # a few secondary humps on the upper side (lobe-on-lobe, fewer and larger than cauliflower bumps)
    for _ in range(int(rng.integers(2, 5))):
        cu = rng.uniform(-0.35, 0.35) * L
        r = T * rng.uniform(0.16, 0.3)
        col = np.argmin(np.abs(u[0] - cu))
        colm = M[:, col]
        top = np.nonzero(colm > 0.5)[0]
        if len(top) == 0:
            continue
        cv = v[top[0], 0] + 0.25 * r
        g = np.exp(-(((u - cu) / (1.3 * r)) ** 2 + ((v - cv) / r) ** 2) * 1.3)
        M = np.maximum(M, g * 0.95)
    # flat base, soft and torn (handled by noise later)
    M = M * _ss(0.3 * T, 0.0, v)
    return M


def _mass_st(rng, L, T, w, h, u, v, heads=(1, 3)):
    """Torn streak: a long band whose thickness varies along its length, 1-3 larger lobes riding on it,
    both ends tapering to wisps."""
    uu = u[0]
    s = np.clip(uu / (0.5 * L) , -1.2, 1.2)
    env = np.clip(1 - np.abs(s) ** 1.6, 0, 1) ** 0.6                  # taper at both ends
    gap = _ss(0.3, 0.5, _fbm(w, 1, 0.12 * L + 4, 1, 2, rng)[0])            # torn apart into patches
    th = np.float32(T) * 0.46 * (0.3 + 1.0 * _fbm(w, 1, 0.25 * L, 1, 2, rng)[0]) * env * (0.35 + 0.65 * gap)
    cen = T * 0.45 * (_fbm(w, 1, 0.3 * L, 1, 2, rng)[0] - 0.5)       # gentle meander
    M = np.exp(-((v - cen[None, :]) / np.maximum(th[None, :], 0.5)) ** 2 * 1.1)
    # larger lobes (heads): fewer, varied size, mostly on the upper side of the band
    k = int(rng.integers(heads[0], heads[1] + 1))
    for j in range(k):
        cu = rng.uniform(-0.42, 0.25) * L
        e = float(np.interp(cu, uu, env))
        r = T * rng.uniform(0.2, 0.42) * (0.4 + 0.6 * e)
        ru = r * rng.uniform(1.4, 2.4)
        cv = float(np.interp(cu, uu, cen)) - 0.45 * r
        g = np.exp(-(((u - cu) / ru) ** 2 + ((v - cv) / r) ** 2) * 1.2)
        M = np.maximum(M, g) + 0.2 * np.minimum(M, g)
    return M


def _field(kind, rng, cl, T, L):
    """Paint the cloud's shape field in its local frame (axis along +u) and return it plus local grid."""
    pad = int(0.45 * T + 16)
    w = int(L + 2 * pad + (0.6 * T if kind == 'cu' else 0.2 * T))
    h = int(T * (1.9 if kind == 'cu' else 1.4) + 2 * pad)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ox, oy = w / 2.0, (0.62 if kind == 'cu' else 0.5) * h
    # low-frequency domain warp: irregular lobe outlines
    wa = 0.12 * T
    wx = (_fbm(w, h, 0.5 * T, 0.5 * T, 2, rng) - 0.5) * 2 * wa
    wy = (_fbm(w, h, 0.5 * T, 0.5 * T, 2, rng) - 0.5) * 2 * wa
    u, v = xx - ox + wx, yy - oy + wy
    if kind == 'cu':
        M = _mass_cu(rng, L, T, w, h, u, v)
    else:
        M = _mass_st(rng, L, T, w, h, u, v, cl.get('heads', (1, 3)))
    M = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 0.02 * T + 0.5)
    gy = cv2.Sobel(cv2.GaussianBlur(M, (0, 0), 0.08 * T + 1), cv2.CV_32F, 0, 1, ksize=3)
    gx = cv2.Sobel(cv2.GaussianBlur(M, (0, 0), 0.08 * T + 1), cv2.CV_32F, 1, 0, ksize=3)
    gn = np.sqrt(gx * gx + gy * gy) + 1e-6
    upf = np.clip(gy / gn, 0, 1)          # M increases downward -> the edge faces the sky
    dnf = np.clip(-gy / gn, 0, 1)
    # cauliflower bumps (rounded, sky side only): smooth noise at two sizes, pushed toward 'bumpy' by a
    # power so bumps are round-topped and the valleys between them are sharp
    b1 = _vn(w, h, 0.16 * T, 0.14 * T, rng)
    b2 = _vn(w, h, 0.07 * T, 0.06 * T, rng)
    b3 = _vn(w, h, 0.035 * T + 1.5, 0.03 * T + 1.5, rng)
    bump = 0.26 * (b1 - 0.5) + 0.18 * (b2 - 0.5) + 0.1 * (b3 - 0.5)
    # torn edges everywhere (dry-brush fringe), stronger on the lee/underside and on streaks
    t1 = _fbm(w, h, 0.05 * T + 2, 0.05 * T + 2, 3, rng) - 0.5
    # wind-stretched streak noise for the underside
    sn = _fbm(w, h, 0.35 * T + 6, 0.07 * T + 2, 3, rng) - 0.5
    st = kind == 'st'
    t2 = _fbm(w, h, 0.18 * T + 3, 0.12 * T + 3, 2, rng) - 0.5
    t0 = _vn(w, h, 0.02 * T + 1.2, 0.018 * T + 1.2, rng) - 0.5        # dry-brush fringe
    F = (M + bump * upf * (0.8 if st else 1.0)
         + (0.3 if st else 0.2) * t1 + (0.5 if st else 0.3) * t2 + 0.1 * t0
         + (0.2 if st else 0.14) * sn * (0.3 + dnf))
    # streaks: fibrous along the wind everywhere
    if st:
        sn2 = _fbm(w, h, 0.5 * T + 8, 0.03 * T + 1.2, 3, rng) - 0.5
        F = F + 0.16 * sn2
    return F.astype(np.float32), (ox, oy)


def night_clouds17(Wp, Hp, clusters, H, light_xy, tail_glow=None, mw_glow=None, seed=17, pal=None):
    if pal is None:
        pal = dict(top=(0.68, 0.85, 0.97), lit2=(0.6, 0.75, 0.88), mid=(0.53, 0.67, 0.82),
                   base=(0.45, 0.57, 0.78), warm=(1.0, 0.53, 0.29),
                   vtop=(0.7, 0.62, 0.82), vmid=(0.47, 0.42, 0.64))
    cP = {k: np.array(v, np.float32) for k, v in pal.items()}
    out = np.zeros((Hp, Wp, 4), np.float32)
    tg = mg = None
    if tail_glow is not None:
        q = 8
        sm = cv2.resize(tail_glow, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        tg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 0.03 * H / q), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    if mw_glow is not None:
        q = 8
        sm = cv2.resize(mw_glow.max(-1), (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        mg = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 0.06 * H / q), (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        mg = mg / (mg.max() + 1e-6)

    order = sorted(range(len(clusters)), key=lambda i: (-clusters[i].get('far', 0.0), clusters[i]['y']))
    for ic in order:
        cl = clusters[ic]
        rng = np.random.default_rng(seed * 1000 + ic * 7 + 3)
        T, L = cl['T'], cl['L']
        kind = cl['kind']
        st = kind == 'st'
        haze = float(cl.get('haze', 0.0))
        wm = float(cl.get('warm', 0.0))
        F, (ox, oy) = _field(kind, rng, cl, T, L)
        h0, w0 = F.shape
        # rotate the local painting into the plate (bounding box of the rotated patch)
        ang = cl['ang']
        R = cv2.getRotationMatrix2D((ox, oy), -ang, 1.0)
        corners = np.array([[0, 0, 1], [w0, 0, 1], [0, h0, 1], [w0, h0, 1]], np.float32) @ R.T
        bx0 = int(math.floor(cl['x'] + corners[:, 0].min() - ox)) - 2
        by0 = int(math.floor(cl['y'] + corners[:, 1].min() - oy)) - 2
        bx1 = int(math.ceil(cl['x'] + corners[:, 0].max() - ox)) + 2
        by1 = int(math.ceil(cl['y'] + corners[:, 1].max() - oy)) + 2
        w, h = bx1 - bx0, by1 - by0
        R2 = R.copy()
        R2[0, 2] += cl['x'] - ox - bx0
        R2[1, 2] += cl['y'] - oy - by0
        Fw = cv2.warpAffine(F, R2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xg = xx + bx0
        thr = 0.5
        # signed distance (px) from the gradient of the field
        gx = cv2.Sobel(Fw, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(Fw, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        gn = np.sqrt(gx * gx + gy * gy) + 1e-5
        sd = (Fw - thr) / gn
        sd = np.clip(sd, -50, 50)
        # facing of the silhouette in world space (big-form scale)
        Gs = cv2.GaussianBlur(np.clip(Fw, 0, 1.2), (0, 0), 0.07 * T + 1)
        sx = cv2.Sobel(Gs, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(Gs, cv2.CV_32F, 0, 1, ksize=3)
        sn_ = np.sqrt(sx * sx + sy * sy) + 1e-6
        down = np.clip(-sy / sn_, 0, 1) ** 1.3
        inside = Fw > thr
        # remove stray crumbs smaller than a brush dab
        n_, lab, stt, _ = cv2.connectedComponentsWithStats(inside.astype(np.uint8), connectivity=8)
        if n_ > 2:
            keep = stt[:, cv2.CC_STAT_AREA] >= (0.05 * T) ** 2 + 5
            keep[0] = False
            inside = keep[lab] & inside
            kl = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        else:
            kl = np.ones((h, w), bool)
        # alpha: 1px AA crisp edge; long lost edge where the silhouette faces down
        # fill small interior holes (they read as punched stamps, not sky gaps)
        outm = (~inside).astype(np.uint8)
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats(outm, connectivity=4)
        if n2 > 1:
            brd = np.zeros(n2, bool)
            brd[np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))] = True
            small = (st2[:, cv2.CC_STAT_AREA] < (0.14 * T) ** 2) & ~brd
            small[0] = small[0] and not brd[0]
            hole = small[lab2] & (outm > 0)
            hole = cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            sd = np.where(hole, np.maximum(sd, 1.5), sd)
            inside = inside | hole
        # alpha: 1px AA crisp edge; long smooth lost edge where the silhouette faces down (measured with a
        # true distance to the edge, so the fade is a clean painted blend, not streaky)
        soft = 1.0 + ((0.18 if wm > 0 else 0.1) if st else (0.16 if wm > 0 else 0.08)) * T * down
        dist = cv2.distanceTransform((sd > 0).astype(np.uint8), cv2.DIST_L2, 5)
        dist = cv2.GaussianBlur(dist, (0, 0), 0.025 * T + 1)
        a0 = 0.35 if st else 0.3           # the torn underside stays faintly legible while it dissolves
        A = _ss(-0.6, 0.9, sd + 0.3) * (a0 + (1 - a0) * _ss(0.0, soft, dist + 1.0)) * kl
        # ---- value structure measured per column from the silhouette
        m_ = sd > -0.5
        dt = _depth_top(m_)
        db = _depth_top(m_[::-1])[::-1]
        # smoothed top line + thickness (the big masses, not every bump)
        sgx, sgy = 0.18 * T + 1, 0.06 * T + 1
        mf = m_.astype(np.float32)
        nrm = np.maximum(cv2.GaussianBlur(mf, (0, 0), sigmaX=sgx, sigmaY=sgy), 1e-3)
        thick = np.maximum(cv2.GaussianBlur((dt + db) * mf, (0, 0), sigmaX=sgx, sigmaY=sgy) / nrm, 2.0)
        dts = cv2.GaussianBlur(dt * mf, (0, 0), sigmaX=0.06 * T + 1, sigmaY=0.03 * T + 1) / nrm
        dep = dts / thick                                    # 0 at the top, ~1 at the base
        hf = db / np.maximum(dt + db, 1.0)                    # 1 at the top ... 0 at the underside
        hf = cv2.GaussianBlur(hf.astype(np.float32), (0, 0), 0.05 * T + 1)
        # light side (moon high, pulled toward the comet)
        vx = light_xy[0] - cl['x']
        lx = 1.0 if vx >= 0 else -1.0
        side = np.clip((xg - cl['x']) * lx / (0.5 * L + 0.5 * T), -1, 1)
        big = _vn(w, h, 0.45 * T + 4, 0.45 * T + 4, rng) - 0.5                 # plane variation
        jag = (_fbm(w, h, 0.045 * T + 2, 0.03 * T + 2, 2, rng) - 0.5)         # brush-jagged terminator
        cap = (0.5 if not st else 0.56) + 0.16 * side + 0.34 * big
        if wm > 0:
            cap = 0.2 + 0.06 * side + 0.12 * big          # afterglow clouds: a thin crisp lit violet top
        lit_s = cap - dep + 0.22 * jag
        lit = _ss(-0.02, 0.02, lit_s * thick / (0.3 * T + 4))                 # crisp painted boundary
        # secondary lit planes lower in the body: broad dabs of a lighter mid (lobe-on-lobe faces)
        p2 = _fbm(w, h, 0.4 * T + 3, 0.25 * T + 3, 2, rng) - 0.5
        lit2 = _ss(0.09, 0.125, p2 + 0.1 * side - 0.25 * np.clip(dep - 0.75, 0, 1)) * (1 - lit)
        # base plane (darker, cooler) where the column nears the underside
        bj = _fbm(w, h, 0.08 * T + 2, 0.04 * T + 2, 2, rng) - 0.5
        basep = 0.3 * _ss(0.02, -0.02, hf - 0.3 + 0.25 * bj) * (1 - lit)
        # ---- colours
        top, lt2, mid, basec = cP['top'].copy(), cP['lit2'].copy(), cP['mid'].copy(), cP['base'].copy()
        if haze > 0:
            top = top * (1 - haze) + cP['vtop'] * haze
            lt2 = lt2 * (1 - haze) + (cP['vtop'] * 0.6 + cP['vmid'] * 0.4) * haze
            mid = mid * (1 - haze) + cP['vmid'] * haze
            basec = basec * (1 - haze) + cP['vmid'] * 0.85 * haze
        # faint brush grain inside each flat plane (painted, not airbrushed)
        grain = (_fbm(w, h, 0.06 * T + 2, 0.04 * T + 2, 2, rng) - 0.5)[..., None] * 0.03
        col = mid[None, None, :] * np.ones((h, w, 1), np.float32)
        col = col * (1 - lit2[..., None]) + lt2 * lit2[..., None]
        col = col * (1 - basep[..., None]) + basec * basep[..., None]
        # faint painted planes inside the lit area (a slightly greyer lobe face under a lighter one)
        p3 = _fbm(w, h, 0.25 * T + 3, 0.18 * T + 3, 2, rng) - 0.5
        ls = _ss(0.11, 0.145, p3 + 0.2 * np.clip(dep, 0, 1)) * 0.45
        topc = top * (1 - ls[..., None]) + lt2 * ls[..., None]
        col = col * (1 - lit[..., None]) + topc * lit[..., None]
        col = col * (1 + grain)
        # warm afterglow underside: flat saturated orange plane under a crisp violet top
        if wm > 0:
            wj = _fbm(w, h, 0.1 * T + 3, 0.05 * T + 2, 2, rng) - 0.5
            wk = _ss(0.6, 0.54, hf + 0.2 * wj) * (1 - 0.9 * lit)
            wk = np.clip(wk * min(1.0, 1.3 * wm), 0, 1)[..., None]
            col = col * (1 - wk) + cP['warm'] * wk
        # ---- translucency: opaque lit top, lighter base (stars show through)
        tr = 1 - (0.5 if not st else 0.58) * _ss(0.02, -0.02, hf - 0.3 + 0.25 * bj) * (1 - lit)
        if wm > 0:
            tr = 1 - 0.3 * _ss(0.3, 0.0, hf)
        if st:
            tr = tr * (0.82 + 0.18 * lit)
        A = A * tr
        # ---- composite
        xa, xb = max(bx0, 0), min(bx1, Wp)
        ya, yb = max(by0, 0), min(by1, Hp)
        if xb <= xa or yb <= ya:
            continue
        sl = (slice(ya - by0, yb - by0), slice(xa - bx0, xb - bx0))
        cc = col[sl]
        if tg is not None:
            tpk = tg[ya:yb, xa:xb]
            cc = cc + np.clip(tpk, 0, 0.35) * (0.3 + 0.7 * lit[sl])[..., None] * 0.6
        if mg is not None:
            cc = cc * (1 + mg[ya:yb, xa:xb][..., None] * np.array([0.07, 0.02, -0.04], np.float32))
        aa = (A[sl] * cl['amt'])[..., None]
        dst = out[ya:yb, xa:xb]
        a0 = dst[..., 3:4]
        na = aa + a0 * (1 - aa)
        dst[..., :3] = (cc * aa + dst[..., :3] * a0 * (1 - aa)) / np.maximum(na, 1e-5)
        dst[..., 3:4] = na
    return out
