"""Round-16 night clouds for s07_comet_night: yn_05 moonlit cloudlets, rebuilt from scratch.

Reviewer notes on round 15: every cloud was one opaque pale periwinkle mass with mottled dough texture and
melted, eroded (crumpled-paper) edges plus stray crumbs; no lit-top / dark-body value structure; no
translucent base.  This painter builds each cloud from explicit geometry instead of eroded noise:

  * 'cu' compact puffs: a flat-based row of big lobes, a tier of child lobes on their upper arcs and a
    tier of small cauliflower bumps on top (union of circles -> clean clustered round bumps on the sky
    side, flattened base),
  * 'st' torn wisps: a few long thin strands along the wind (union of stretched ellipses) that tear into
    streaks at their ends and undersides, with one or two small puffy heads riding on them,
  * value: an occlusion march toward the light (moon high, pulled toward the comet).  Exposed upper faces
    of every lobe come out lit -> a crisp near-white cyan-grey top (~20% of the height) with a hard
    terminator where lit faces meet shadow, the body falls to ONE broad darker blue-violet value
    (~50% luminance), the base cools further.  No internal noise mottling: only 2-3 large planes,
  * edges: 1px crisp AA edge on the lit/sky-facing side; the underside/lee side is torn into long
    horizontal streaks and softened (lost edge), and the lower body turns translucent (alpha 0.35-0.6)
    so stars and the Milky Way show through,
  * local light: near-edge cyan/magenta pickup from the comet tail glow, slight warmth near the Milky
    Way, warm afterglow undersides on the horizon clouds, whose tops are dimmed into violet-grey haze.
Returns straight RGBA (Hp, Wp, 4).
"""
import math

import numpy as np
import cv2

import s07_comet_night_paint as P


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _aniso(w, h, along, across, ang, seed, octaves=3):
    """fbm (0..1) stretched along direction `ang` (deg); along/across = feature size in px."""
    N = 256
    base = P.fbm_lowres(N, N, 8, octaves, seed, q=1)
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    cell = N / 8.0
    M = np.array([[ca * cell / along, sa * cell / along, 17.0],
                  [-sa * cell / across, ca * cell / across, 23.0]], np.float32)
    return cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_WRAP)


# ----------------------------------------------------------------------------- geometry
def _cumulus(rng, cl):
    """Circles (x, y, r) in plate px + base line y for a compact moonlit puff."""
    L, T, ang = cl['L'], cl['T'], math.radians(cl['ang'])
    ca, sa = math.cos(ang), math.sin(ang)
    x0, y0 = cl['x'], cl['y']
    circ = []
    n = max(2, int(round(L / (0.55 * T))) + 1)
    us = np.linspace(-0.5, 0.5, n) * L + rng.normal(0, 0.06 * T, n)
    for i, u in enumerate(us):
        q = abs(u) / (0.5 * L + 1e-3)
        dome = 1.0 - 0.55 * q ** 1.6                 # taller in the middle, low at the ends
        r = T * rng.uniform(0.34, 0.5) * dome
        cx = x0 + u * ca
        cy = y0 + u * sa - 0.55 * r
        circ.append((cx, cy, r, 0))
        # child lobes on the upper arc (sky side)
        for _ in range(rng.integers(2, 4)):
            a = math.radians(rng.uniform(-165, -15))
            rr = r * rng.uniform(0.38, 0.72)
            d = r * rng.uniform(0.55, 0.95)
            ccx, ccy = cx + math.cos(a) * d, cy + math.sin(a) * d
            circ.append((ccx, ccy, rr, 1))
            # small cauliflower bumps on the child's upper arc
            for _ in range(rng.integers(3, 6)):
                a2 = math.radians(rng.uniform(-165, -15))
                r2 = rr * rng.uniform(0.3, 0.48)
                d2 = rr * rng.uniform(0.8, 0.95)
                circ.append((ccx + math.cos(a2) * d2, ccy + math.sin(a2) * d2, r2, 2))
    # flat base: slightly tilted with the cloud axis
    return circ, (ca, sa)


def _strands(rng, cl):
    """Torn wisps: thin connecting ellipses (cx, cy, a, b, ang) + chains of small puffs (circles) riding
    along them, with gaps; returns (ellipses, circles)."""
    L, T, ang = cl['L'], cl['T'], cl['ang']
    x0, y0 = cl['x'], cl['y']
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    ell, circ = [], []
    n = int(rng.integers(2, 4))
    nh = cl.get('heads', (3, 7))
    for i in range(n):
        u0 = rng.uniform(-0.3, 0.3) * L
        v0 = rng.normal(0, 0.2 * T)
        a = L * rng.uniform(0.2, 0.42)
        b = T * rng.uniform(0.07, 0.15)
        ea = ang + rng.normal(0, 3)
        c_, s_ = math.cos(math.radians(ea)), math.sin(math.radians(ea))
        ex, ey = x0 + u0 * ca - v0 * sa, y0 + u0 * sa + v0 * ca
        # the strand is broken into 2-4 segments of varied thickness with gaps (torn, not a tube)
        nseg = int(rng.integers(2, 5))
        cuts = np.sort(rng.uniform(-1, 1, nseg - 1)) if nseg > 1 else np.array([])
        edges_ = np.concatenate([[-1.0], cuts, [1.0]])
        for j in range(nseg):
            f0, f1 = edges_[j], edges_[j + 1]
            gap = rng.uniform(0.0, 0.12)
            fm, hl = 0.5 * (f0 + f1), max(0.5 * (f1 - f0) - gap, 0.05)
            bj = b * rng.uniform(0.5, 1.5)
            ell.append((ex + fm * a * c_ + rng.normal(0, 0.1 * T) * -s_, ey + fm * a * s_ + rng.normal(0, 0.1 * T) * c_,
                        hl * a, bj, ea + rng.normal(0, 4)))
        # puffs along the strand: bigger near one end (the 'head'), tapering toward the tail, with gaps
        k = int(rng.integers(nh[0], nh[1] + 1))
        for j in range(k):
            f = rng.uniform(-0.9, 0.4)
            if rng.random() < 0.25:
                continue
            taper = 0.55 + 0.45 * (1 - (f + 0.9) / 1.8) ** 0.8
            r = T * rng.uniform(0.16, 0.32) * taper
            px, py = ex + f * a * c_, ey + f * a * s_ - 0.4 * r
            circ.append((px, py, r, 1))
            for _ in range(int(rng.integers(1, 3))):
                a2 = math.radians(rng.uniform(-160, -20))
                r2 = r * rng.uniform(0.35, 0.6)
                circ.append((px + math.cos(a2) * 0.8 * r, py + math.sin(a2) * 0.8 * r, r2, 2))
    return ell, circ


def _dabs(D, rng, T, spacing, rmin, rmax):
    """Painterly edge: stamp many small round dabs along the up-facing contour (cauliflower bumps) and
    bite small dabs out of the down-facing contour (torn underside).  Local-window updates."""
    h, w = D.shape
    gy, gx = np.gradient(D)
    ys, xs = np.nonzero(np.abs(D) < 0.75)
    if len(xs) == 0:
        return D
    n = int(len(xs) / max(spacing, 1.0))
    if n <= 0:
        return D
    pick = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    out = D.copy()
    for k in pick:
        x, y = xs[k], ys[k]
        nx, ny = gx[y, x], gy[y, x]
        nn = math.hypot(nx, ny) + 1e-6
        nx, ny = nx / nn, ny / nn                     # outward normal
        up = -ny                                       # +1 facing the sky
        if up > -0.25:
            r = rng.uniform(rmin, rmax) * (0.6 + 0.4 * max(up, 0))
            cx, cy = x + nx * r * rng.uniform(-0.3, 0.3), y + ny * r * rng.uniform(-0.3, 0.3)
            sgn = 1
        else:
            continue
            r = rng.uniform(rmin, rmax) * 0.8
            cx, cy = x - nx * r * rng.uniform(0.2, 0.6), y - ny * r * rng.uniform(0.2, 0.6)
            sgn = -1
        x0_, x1_ = max(int(cx - r - 2), 0), min(int(cx + r + 3), w)
        y0_, y1_ = max(int(cy - r - 2), 0), min(int(cy + r + 3), h)
        if x1_ <= x0_ or y1_ <= y0_:
            continue
        yy, xx = np.mgrid[y0_:y1_, x0_:x1_].astype(np.float32)
        dd = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r
        win = out[y0_:y1_, x0_:x1_]
        if sgn > 0:
            out[y0_:y1_, x0_:x1_] = np.minimum(win, dd)
        else:
            out[y0_:y1_, x0_:x1_] = np.maximum(win, -dd)
    return out


def _depth_top(M):
    """Per column: distance down from the nearest outside pixel above (0 outside)."""
    h, w = M.shape
    inside = M > 0.5
    idx = np.arange(h, dtype=np.float32)[:, None]
    last_out = np.maximum.accumulate(np.where(inside, -1.0, idx), axis=0)
    d = np.where(inside, idx - last_out - 0.5, 0.0)
    return d.astype(np.float32)


def _circ_sdf(circ, xg, yg):
    d = np.full(xg.shape, 1e6, np.float32)
    for (cx, cy, r, _) in circ:
        d = np.minimum(d, np.sqrt((xg - cx) ** 2 + (yg - cy) ** 2) - r)
    return d


def _ell_sdf(ell, xg, yg):
    d = np.full(xg.shape, 1e6, np.float32)
    for (cx, cy, a, b, ang) in ell:
        c, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        u = (xg - cx) * c + (yg - cy) * s_
        v = -(xg - cx) * s_ + (yg - cy) * c
        q = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        d = np.minimum(d, (q - 1.0) * b)
    return d


# ----------------------------------------------------------------------------- painter
def night_clouds16(Wp, Hp, clusters, H, light_xy, tail_glow=None, mw_glow=None, seed=17, pal=None):
    """tail_glow / mw_glow: optional (Hp, Wp, 3) plate-space light maps used for local tints."""
    if pal is None:
        pal = dict(top=(0.78, 0.92, 1.0), mid=(0.5, 0.61, 0.76), body=(0.37, 0.46, 0.64),
                   base=(0.27, 0.33, 0.55), warm=(1.0, 0.6, 0.42), haze_top=(0.56, 0.5, 0.68))
    cP = {k: np.array(v, np.float32) for k, v in pal.items()}
    out = np.zeros((Hp, Wp, 4), np.float32)
    # wide soft light maps for the local tints
    tg = None
    if tail_glow is not None:
        q = 8
        sm = cv2.resize(tail_glow, (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 0.03 * H / q)
        tg = cv2.resize(sm, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
    mg = None
    if mw_glow is not None:
        q = 8
        sm = cv2.resize(mw_glow.max(-1), (Wp // q, Hp // q), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 0.06 * H / q)
        mg = cv2.resize(sm, (Wp, Hp), interpolation=cv2.INTER_LINEAR)
        mg = mg / (mg.max() + 1e-6)

    order = sorted(range(len(clusters)), key=lambda i: (-clusters[i].get('far', 0.0), clusters[i]['y']))
    for ic in order:
        cl = clusters[ic]
        rng = np.random.default_rng(seed * 1000 + ic * 7 + 1)
        T = cl['T']
        st = cl['kind'] == 'st'
        haze = float(cl.get('haze', 0.0))
        if st:
            ell, circ = _strands(rng, cl)
            ext = [(e[0], e[1], max(e[2], e[3])) for e in ell] + [(c[0], c[1], c[2]) for c in circ]
        else:
            circ, (ca, sa) = _cumulus(rng, cl)
            ext = [(c[0], c[1], c[2]) for c in circ]
        pad = int(0.35 * T + 12)
        bx0 = int(math.floor(min(e[0] - e[2] for e in ext))) - pad
        bx1 = int(math.ceil(max(e[0] + e[2] for e in ext))) + pad
        by0 = int(math.floor(min(e[1] - e[2] for e in ext))) - pad
        by1 = int(math.ceil(max(e[1] + e[2] for e in ext))) + pad
        w, h = bx1 - bx0, by1 - by0
        if w < 8 or h < 8:
            continue
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xg, yg = xx + bx0, yy + by0
        # low-amplitude painterly warp (hand-drawn irregularity of the round bumps, not erosion)
        wa = 0.03 * T + 0.6
        wx = (P.fbm_lowres(w, h, w / (0.15 * T + 4), 2, seed + ic * 11 + 1, q=2) - 0.5) * 2 * wa
        wy = (P.fbm_lowres(w, h, w / (0.15 * T + 4), 2, seed + ic * 11 + 2, q=2) - 0.5) * 2 * wa
        if st:
            De = _ell_sdf(ell, xg + wx * 3, yg + wy * 2)
            Dc = _circ_sdf(circ, xg + wx, yg + wy) if circ else np.full((h, w), 1e6, np.float32)
            D = np.minimum(De, Dc)
            edom = _ss(-0.02 * T, 0.06 * T, Dc - De).astype(np.float32)   # 1 where the thin strand dominates
        else:
            D = _circ_sdf(circ, xg + wx, yg + wy)
            # soft flattening of the base along the cloud axis (wavy, not a ruled line)
            vb = -(xg - cl['x']) * sa + (yg - cl['y']) * ca
            base = vb - 0.1 * T - (0.14 * T) * (P.fbm1d(xx[0] / (0.35 * T + 1), 1, 3, seed + ic) - 0.5)[None, :]
            D = np.maximum(D, base)
        D = D.astype(np.float32)
        # painterly contour: small round dabs on the sky side, torn bites underneath
        sp_ = 2.0 if st else 1.0
        D = _dabs(D, rng, T, spacing=(0.08 * T + 4) * sp_, rmin=0.04 * T + 1.5, rmax=0.1 * T + 2)
        D = _dabs(D, rng, T, spacing=(0.12 * T + 6) * sp_, rmin=0.015 * T + 1, rmax=0.035 * T + 1.2)
        # close narrow interior slits between lobes (they read as dark creases)
        ins = (D < 0).astype(np.uint8)
        kk = int(0.05 * T + 3) | 1
        clo = cv2.morphologyEx(ins, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
        D = np.where((clo > 0) & (ins == 0), np.minimum(D, -0.6), D).astype(np.float32)
        D = cv2.GaussianBlur(D, (0, 0), 0.6)
        # ---- facing of the silhouette
        G = cv2.GaussianBlur(np.clip(0.5 - D, 0, 1), (0, 0), 0.06 * T + 1)
        gx = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        down = np.clip(-gy / gn, 0, 1) ** 1.5                   # 1 where the edge faces down
        # lost edge underneath: long soft wind-streaked fade
        streak = _aniso(w, h, 0.6 * T + 6, 0.05 * T + 1.5, cl['ang'], seed + ic * 17 + 3)
        soft = 0.7 + (0.14 if st else 0.08) * T * down * (0.5 + streak)
        if st:
            soft = np.maximum(soft, 0.7 + 0.1 * T * edom * (0.4 + streak))   # veil strands: lost edges
        A = np.clip(0.5 - D / soft, 0, 1)
        # stray crumbs off: keep only components that belong to a sizeable mass
        m8 = (A > 0.3).astype(np.uint8)
        n_, lab, stt, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
        if n_ > 2:
            amin = (0.2 * T) ** 2
            keep = stt[:, cv2.CC_STAT_AREA] >= amin
            keep[0] = True
            kl = cv2.dilate(keep[lab].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            A = A * kl
        # ---- light: moon high, pulled toward the comet
        vx, vy = light_xy[0] - cl['x'], light_xy[1] - cl['y']
        nv = math.hypot(vx, vy) + 1e-3
        lx = 0.3 * vx / nv
        # segment-wise depth under the top silhouette (dt) and height above the underside (db)
        m_ = (D < 1.5).astype(np.float32)
        dt = _depth_top(m_)
        db = _depth_top(m_[::-1])[::-1]
        # painted terminator measured from a SMOOTHED top line: small cauliflower bumps sit wholly in the
        # light and the lit/shadow boundary follows the big lobe forms (a plane, not an outline band)
        tp = (yy - dt) * m_
        sgx, sgy = 0.2 * T + 1, 0.05 * T + 1
        nrm = np.maximum(cv2.GaussianBlur(m_, (0, 0), sigmaX=sgx, sigmaY=sgy), 1e-3)
        tps = cv2.GaussianBlur(tp, (0, 0), sigmaX=sgx, sigmaY=sgy) / nrm
        thick = cv2.GaussianBlur((dt + db) * m_, (0, 0), sigmaX=sgx, sigmaY=sgy) / nrm
        thick = np.maximum(thick, 2.0)
        dep = (yy - tps) / thick                                   # 0 at the smoothed top, ~1 at the base
        tn = P.fbm_lowres(w, h, w / (0.5 * T + 4), 2, seed + ic * 19 + 7, q=2) - 0.5
        side = np.clip((xg - cl['x']) * np.sign(lx) / (0.5 * cl['L'] + T), -1, 1)
        cap = (0.4 if not st else 0.48) + 0.08 * side + 0.24 * tn
        tw_ = (0.025 * T + 1.2) / thick                             # terminator: crisp-ish painted edge
        lit = _ss(cap + tw_, cap - tw_, dep)
        # inside the lit plane: brightest right under the sky edge
        glowt = _ss(cap, -0.05, dep)
        hf = np.clip(dt / np.maximum(dt + db, 1.0), 0, 1)
        hf = cv2.GaussianBlur(hf.astype(np.float32), (0, 0), 0.06 * T + 1)
        # ---- colour: top -> (terminator) -> mid -> body -> base
        top = cP['top'].copy()
        mid = cP['mid'].copy()
        body = cP['body'].copy()
        basec = cP['base'].copy()
        if haze > 0:
            top = top * (1 - haze) + cP['haze_top'] * haze
            mid = mid * (1 - 0.7 * haze) + cP['haze_top'] * 0.82 * 0.7 * haze
            body = body * (1 - 0.4 * haze) + cP['haze_top'] * 0.68 * 0.4 * haze
        bd = _ss(cap + 0.02, cap + 0.4, dep)                      # soft mid -> body under the terminator
        bb = _ss(0.45, 1.0, hf)[..., None]
        shade = mid * (1 - bd[..., None]) + body * bd[..., None]
        shade = shade * (1 - bb) + basec * bb
        # lobe-on-lobe: a few front lobes low in the body catch their own lit upper face (hard terminator)
        sub = np.zeros((h, w), np.float32)
        cand = [c for c in circ if c[3] <= 1 and c[2] > 0.14 * T]
        rng.shuffle(cand)
        nsub = 0
        for (cx_, cy_, r_, _) in cand:
            iy, ix = int(cy_ - by0), int(cx_ - bx0)
            if not (0 <= iy < h and 0 <= ix < w) or dep[iy, ix] < cap[iy, ix] + 0.1:
                continue
            u = xg - cx_
            v = yg - cy_
            th_ = np.arctan2(v, u)
            ph = rng.uniform(0, 6.28, 3)
            rb = r_ * (1 + 0.05 * np.sin(7 * th_ + ph[0]) + 0.035 * np.sin(13 * th_ + ph[1]) + 0.025 * np.sin(23 * th_ + ph[2]))
            dd = np.sqrt(u * u + v * v)
            inner = _ss(-0.6, 0.6, rb - dd)                           # crisp arc edge of the front lobe
            v2 = v - 0.5 * r_
            below = _ss(-0.35 * r_, 0.25 * r_, r_ * 1.05 - np.sqrt(u * u + v2 * v2))   # soft fall into shade
            k_ = inner * (1 - below)
            sub = np.maximum(sub, k_)
            nsub += 1
            if nsub >= 4:
                break
        sub = sub * (1 - lit) * (D < 0)
        topv = top * (0.86 + 0.14 * glowt[..., None])
        col = shade * (1 - lit[..., None]) + topv * lit[..., None]
        subc = mid * 0.45 + top * 0.55
        col = col * (1 - 0.8 * sub[..., None]) + subc * 0.8 * sub[..., None]
        if st:
            # thin veil strands are lit through: pale, low-contrast moonlit grey-cyan
            vc = mid * 0.45 + top * 0.55
            ev = (0.75 * edom * (1 - lit))[..., None]
            col = col * (1 - ev) + vc * ev
        # warm afterglow on the undersides of the low clouds
        wm = float(cl.get('warm', 0.0))
        if wm > 0:
            wk = np.clip(wm * 1.3 * _ss(0.12, 0.8, hf) * (1 - 0.8 * lit), 0, 0.9)[..., None]
            col = col * (1 - wk) + cP['warm'] * wk
        # ---- alpha: opaque lit top, translucent lower body (stars show through), lighter wisps
        tr = 1 - (0.62 * (1 - 0.4 * edom) if st else 0.55) * (1 - 0.5 * haze) * _ss(0.35, 1.0, hf) * (1 - lit)
        if haze > 0:
            tr = tr * (1 - 0.3 * haze * _ss(0.2, 1.0, hf))
        A = A * tr
        if st:
            A = A * (0.8 + 0.2 * lit)
        # ---- composite
        xa, xb = max(bx0, 0), min(bx1, Wp)
        ya, yb = max(by0, 0), min(by1, Hp)
        if xb <= xa or yb <= ya:
            continue
        sl = (slice(ya - by0, yb - by0), slice(xa - bx0, xb - bx0))
        cc = col[sl]
        # local light pickups
        if tg is not None:
            tpk = tg[ya:yb, xa:xb]
            nearw = (0.3 + 0.7 * lit[sl])[..., None]
            cc = cc + np.clip(tpk, 0, 0.6) * nearw * 1.1
        if mg is not None:
            wmw = mg[ya:yb, xa:xb][..., None]
            cc = cc * (1 + wmw * np.array([0.08, 0.02, -0.05], np.float32))
        aa = (A[sl] * cl['amt'])[..., None]
        dst = out[ya:yb, xa:xb]
        a0 = dst[..., 3:4]
        na = aa + a0 * (1 - aa)
        dst[..., :3] = (cc * aa + dst[..., :3] * a0 * (1 - aa)) / np.maximum(na, 1e-5)
        dst[..., 3:4] = na
    return out
