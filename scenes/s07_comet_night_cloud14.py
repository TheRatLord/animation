"""Round-14 night clouds for s07_comet_night: yn_05 flat cut-outs.

Round 13 read as melted marshmallow (blurred, noise-eroded, near-white).  Here every cloud is a hard-edged
cut-out built as a union of circles at three scales (big body lobes, mid buds, fine scallops), anti-aliased to
~1px, never blurred and never eroded by noise:

  * silhouette = union of circles (cumulus: with a gently wobbling flat base line; streamers: a string of broken
    hard-edged fragments, each a short run of scallops with its own flat base),
  * values are FLAT paint cels, no per-lobe spherical falloff:
      - moonlit cyan-white top band (the top contour copied down by ~20-25% of the cloud height),
      - one flat desaturated blue-grey body (~#7890bc, 55-65% value),
      - a hard-edged darker navy-violet base band along the underside,
      - 2-3 inner lobe separations per large mass: a front lobe with a lighter crescent cap and a slightly
        darker blue-grey sickle under it,
  * small detached scrap clusters around the larger masses (yn_05 x600-850 y240-420),
  * no blur, no haze skirt, no smoke under the bases; opacity is flat per cloud (far wisps partly transparent).
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


class _Piece:
    __slots__ = ('circ', 'clip', 'kr')

    def __init__(self, clip=None, kr=4.0):
        self.kr = kr
        self.circ = []      # (cx, cy, r, tag)   tag: 0 body, 1 front lobe
        self.clip = clip    # (px, py, nx, ny): inside where (p - p0).n <= 0, plus wobble amplitude


def _rot(u, v, ca, sa):
    return u * ca - v * sa, u * sa + v * ca


def _buds(rng, piece, parents, gens, up=(0.0, -1.0), spread=(-165, -15)):
    """Add scallop buds on the light-facing arcs of the parent circles, `gens` generations deep."""
    cur = parents
    base_a = math.degrees(math.atan2(up[1], up[0]))
    for g, (nlo, nhi, rlo, rhi, dlo, dhi) in enumerate(gens):
        nxt = []
        for (cx, cy, r) in cur:
            for _ in range(int(rng.integers(nlo, nhi + 1))):
                a = math.radians(base_a + rng.uniform(spread[0] + 90, spread[1] + 90))
                rr = r * rng.uniform(rlo, rhi)
                d = r * rng.uniform(dlo, dhi)
                bx, by = cx + d * math.cos(a), cy + d * math.sin(a)
                piece.circ.append((bx, by, rr, 0))
                nxt.append((bx, by, rr))
        cur = nxt


def _envelope_circles(rng, ells, T, box, n_target, rmin, rmax, top_bias=0.0, gam=1.3):
    """Scatter circles inside a soft envelope (union of ellipses); big circles in the core, small ones at the
    periphery, so the union silhouette is cauliflower-scalloped at several scales with bays and detached bits.
    ells: (cx, cy, ru, rv, ang_deg).  Returns list of (cx, cy, r, E)."""
    x0, y0, x1, y1 = box

    def env(px, py):
        e = np.zeros_like(px)
        for (cx, cy, ru, rv, a) in ells:
            c_, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
            du = (px - cx) * c_ + (py - cy) * s_
            dv = -(px - cx) * s_ + (py - cy) * c_
            e = np.maximum(e, 1 - (du / ru) ** 2 - (dv / rv) ** 2)
        return np.clip(e, 0, 1)
    m = int(n_target * 6)
    px = rng.uniform(x0, x1, m)
    py = rng.uniform(y0, y1, m)
    E = env(px, py)
    keep = rng.random(m) < E ** 1.6
    px, py, E = px[keep][:n_target], py[keep][:n_target], E[keep][:n_target]
    out = []
    for X, Y, e in zip(px, py, E):
        r = (rmin + (rmax - rmin) * e ** gam) * rng.uniform(0.7, 1.15)
        out.append((float(X), float(Y - top_bias * r), float(r), float(e)))
    # draw order irrelevant (union); sort big first for front-lobe picking
    out.sort(key=lambda c: -c[2])
    return out


def _cumulus(rng, cl):
    x, y, L, T, ang = cl['x'], cl['y'], cl['L'], cl['T'], cl['ang']
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    yb = y + 0.32 * T                                 # flat base line (tilted with the spine)
    pc = _Piece(clip=(x, yb, -sa * 0.35, 1.0), kr=0.12 * T)
    # a row of big domes of uneven height (crown higher in the middle) ...
    n = max(2, int(round(L / (0.5 * T))))
    base = []
    for i in range(n):
        q = (i + 0.5) / n - 0.5 + rng.normal(0, 0.04)
        e = max(1 - abs(2 * q) ** 3, 0.08) ** 0.5 * rng.uniform(0.6, 1.1)
        r = T * rng.uniform(0.3, 0.46) * (0.65 + 0.4 * e)
        cx = x + q * L * ca
        cy = yb - r * rng.uniform(0.6, 1.0) - 0.3 * T * e * rng.uniform(0.3, 1.0)
        pc.circ.append((cx, cy, r, 0))
        base.append((cx, cy, r))
    # ... filled underneath so the body is solid down to the flat base
    for i in range(2 * n):
        q = (i + 0.5) / (2 * n) - 0.5
        r = T * 0.22 * (1 - abs(2 * q) ** 2.5) ** 0.5 + 0.02 * T
        pc.circ.append((x + q * L * 0.85 * ca, yb - 0.1 * T, r, 0))
    # ... carrying mid buds, then fine scallops (cauliflower crown at three scales)
    _buds(rng, pc, base, [(2, 4, 0.36, 0.55, 0.62, 0.88), (1, 3, 0.32, 0.5, 0.8, 0.97),
                          (0, 2, 0.35, 0.5, 0.85, 1.0)], spread=(-185, 5))
    # stepped-down, rounded ends: a couple of ever smaller lobes sitting on the base line beyond each end
    for sgn in (-1, 1):
        cx, cy, r = base[0] if sgn < 0 else base[-1]
        for j in range(int(rng.integers(1, 3))):
            r = r * rng.uniform(0.65, 0.85)
            cx = cx + sgn * r * rng.uniform(0.7, 1.1)
            cy = yb - r * rng.uniform(0.35, 0.8)
            pc.circ.append((cx, cy, r, 0))
            if rng.random() < 0.6:
                pc.circ.append((cx + rng.normal(0, 0.3) * r, cy - 0.8 * r, r * rng.uniform(0.35, 0.55), 0))
    # front lobes (inner lobe separations): lobes standing in front of the mass, low in it
    nf = (1 if L > 1.1 * T else 0) + (1 if L > 1.8 * T else 0) + (1 if L > 2.6 * T else 0)
    for k in range(nf):
        q = (k + 0.5) / nf - 0.5 + rng.normal(0, 0.06)
        R = T * rng.uniform(0.3, 0.42)
        cx = x + q * L * 0.8 * ca
        cy = yb - R * rng.uniform(0.75, 1.0)
        tmp = _Piece()
        # a lumpy dome: 3-5 overlapping lobes of different size, crown buds on the upper ones
        m = int(rng.integers(3, 6))
        dome = []
        for j in range(m):
            u = (j / (m - 1) - 0.5) * 1.4 * R
            rr = R * rng.uniform(0.38, 0.6) * (1 - 0.35 * abs(u) / R)
            dome.append((cx + u, cy - 0.35 * R * (1 - (u / R) ** 2) + rr * 0.2, rr))
            tmp.circ.append((dome[-1][0], dome[-1][1], rr, 1))
        _buds(rng, tmp, dome, [(1, 2, 0.35, 0.55, 0.7, 0.92)], spread=(-150, -30))
        pc.circ.extend([(c[0], c[1], c[2], 1) for c in tmp.circ])
    pieces = [pc]
    # detached scraps around the mass (hard-edged, each its own tiny flat-based cut-out)
    for k in range(int(rng.integers(2, 5)) if cl['amt'] >= 0.9 else 0):
        side = -1 if rng.random() < 0.5 else 1
        qq = side * (0.5 + rng.uniform(0.1, 0.35))
        sx = x + qq * L * ca + rng.normal(0, 0.15) * T
        sy = y + qq * L * sa + rng.uniform(-0.6, 0.1) * T
        pieces.append(_scrap(rng, sx, sy, T * rng.uniform(0.07, 0.13), ang))
    return pieces


def _scrap(rng, sx, sy, r0, ang):
    pc = _Piece(clip=(sx, sy + 0.5 * r0, -math.sin(math.radians(ang)) * 0.5, 1.0), kr=0.4 * r0)
    L = r0 * rng.uniform(2.5, 5.0)
    ells = [(sx, sy, 0.5 * L, 1.1 * r0, ang)]
    box = (sx - L, sy - 2 * r0, sx + L, sy + 2 * r0)
    for (cx, cy, r, e) in _envelope_circles(rng, ells, r0, box, int(rng.integers(8, 22)), 0.25 * r0, 0.8 * r0):
        pc.circ.append((cx, cy, r, 0))
    return pc


def _streamer(rng, cl):
    """Torn stratocumulus streamer: a long thin envelope along the comet's direction, broken into pieces by a
    1-D noise along the spine; scattered circles give a scalloped top and frayed, broken ends."""
    x, y, L, T, ang = cl['x'], cl['y'], cl['L'], cl['T'], cl['ang']
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    pieces = []
    T0 = cl['H'] * 0.012
    q = -0.5
    while q < 0.5:
        taper = 1 - 0.6 * (q + 0.5) if cl.get('taper', True) else 1.0
        fl = T * rng.uniform(3.5, 7.0) * taper
        th = T * rng.uniform(0.3, 0.45) * (0.55 + 0.45 * taper)
        off = rng.normal(0, 0.3) * T
        fx = x + (q * L + 0.5 * fl) * ca - off * sa
        fy = y + (q * L + 0.5 * fl) * sa + off * ca
        a2 = ang + rng.normal(0, 5)
        c2, s2 = math.cos(math.radians(a2)), math.sin(math.radians(a2))
        pc = _Piece(clip=None)
        ells = [(fx, fy, 0.5 * fl, th, a2), (fx, fy + 0.25 * th, 0.42 * fl, 0.6 * th, a2)]
        # a puffier head somewhere along the fragment
        if rng.random() < 0.6:
            u = rng.uniform(-0.3, 0.3) * fl
            ells.append((fx + u * c2, fy + u * s2 - 0.5 * th, 0.18 * fl, 1.1 * th, a2))
        box = (fx - 0.6 * fl - th, fy - 2.2 * th, fx + 0.6 * fl + th, fy + 1.5 * th)
        nt = int(np.clip(fl * th / (th * th) * 16, 16, 260))
        for (cx, cy, r, e) in _envelope_circles(rng, ells, th, box, nt, max(0.13 * th, 0.35 * T0), 0.45 * th, gam=1.5):
            pc.circ.append((cx, cy, r, 0))
        pieces.append(pc)
        q += (fl * rng.uniform(0.75, 1.05) + T * rng.uniform(-0.2, 0.8) * (0.6 + 0.6 * (q + 0.5))) / L
    return pieces


def _raster(pieces, bx0, by0, w, h, wob, WX, WY, mg=3):
    """Signed distance (px, negative inside) of the union of the pieces over a local box; plus the front-lobe
    distance field (union of tag-1 circles)."""
    big = 1e4
    D = np.full((h, w), big, np.float32)
    Df = np.full((h, w), big, np.float32)
    Bh = np.full((h, w), big, np.float32)     # height above the owning piece's flat base line
    for pc in pieces:
        Dp = np.full((h, w), big, np.float32)
        x0 = y0 = 10 ** 9
        x1 = y1 = -10 ** 9
        for (cx, cy, r, tag) in pc.circ:
            a0 = int(max(math.floor(cx - r - mg - bx0), 0))
            a1 = int(min(math.ceil(cx + r + mg - bx0), w))
            b0 = int(max(math.floor(cy - r - mg - by0), 0))
            b1 = int(min(math.ceil(cy + r + mg - by0), h))
            if a1 <= a0 or b1 <= b0:
                continue
            yy, xx = np.mgrid[b0:b1, a0:a1].astype(np.float32)
            wx_, wy_ = WX[b0:b1, a0:a1], WY[b0:b1, a0:a1]
            d = np.sqrt((xx + wx_ + bx0 - cx) ** 2 + (yy + wy_ + by0 - cy) ** 2) - r
            Dp[b0:b1, a0:a1] = np.minimum(Dp[b0:b1, a0:a1], d)
            if tag == 1:
                Df[b0:b1, a0:a1] = np.minimum(Df[b0:b1, a0:a1], d)
            x0, x1, y0, y1 = min(x0, a0), max(x1, a1), min(y0, b0), max(y1, b1)
        if x1 <= x0:
            continue
        if pc.clip is not None:
            px, py, nx, ny = pc.clip
            nn = math.hypot(nx, ny)
            nx, ny = nx / nn, ny / nn
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            hp = (xx + bx0 - px) * nx + (yy + by0 - py) * ny
            hp = hp - wob[y0:y1, x0:x1]
            # rounded union with the base plane (smooth max): no sharp prow where a lobe meets the base
            a_, b_ = Dp[y0:y1, x0:x1], hp
            kr = pc.kr
            hh = np.clip(0.5 + 0.5 * (a_ - b_) / kr, 0, 1)
            Dp[y0:y1, x0:x1] = a_ * hh + b_ * (1 - hh) + kr * hh * (1 - hh)
            sub = Bh[y0:y1, x0:x1]
            own = Dp[y0:y1, x0:x1] < 0.7
            Bh[y0:y1, x0:x1] = np.where(own, np.minimum(sub, -hp), sub)
        D = np.minimum(D, Dp)
    return D, Df, Bh


def _depth_top(M):
    """Per column: pixels below the nearest outside pixel above (distance down from the top silhouette)."""
    h, w = M.shape
    inside = (M > 0.5).astype(np.uint8)
    inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    idx = np.arange(h, dtype=np.float32)[:, None]
    last_out = np.maximum.accumulate(np.where(inside, -1.0, idx), axis=0)
    last_out = np.where(last_out < 0, -1.0, last_out)
    d = idx - last_out - 0.5
    return cv2.GaussianBlur(d.astype(np.float32), (0, 0), 0.8)


def _cov(d):
    return np.clip(0.5 - d, 0.0, 1.0)


def _shiftmap(A, dx, dy):
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)     # out(p) = A(p + (dx, dy))
    return cv2.warpAffine(A, M, (A.shape[1], A.shape[0]), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)


def night_clouds14(Wp, Hp, clusters, s, H, light_xy, seed=17, pal=None):
    """clusters: dicts with x, y (plate px), L, T (px), ang (deg), kind ('cu' | 'st'), amt, warm.
    Returns straight RGBA (Hp, Wp, 4)."""
    if pal is None:
        pal = dict(lit=(0.8, 0.9, 0.99), cap=(0.62, 0.74, 0.9), body=(0.43, 0.52, 0.71),
                   sep=(0.40, 0.47, 0.67), base=(0.27, 0.29, 0.49),
                   w_lit=(1.0, 0.86, 0.82), w_body=(0.56, 0.47, 0.64), w_base=(0.36, 0.26, 0.44))
    out = np.zeros((Hp, Wp, 4), np.float32)
    # very low-frequency dry-brush value drift (painted, not rendered; no dimples)
    drift = P.fbm_lowres(Wp, Hp, Wp / (0.25 * H), 2, seed + 5, q=8)
    # small domain warp at two scales: breaks the perfect circle arcs into lumpy painted scallops while
    # the edge stays a crisp 1px anti-aliased cut (the warp moves the edge, it never blurs it)
    GWX = (P.fbm_lowres(Wp, Hp, Wp / (0.035 * H), 2, seed + 21, q=2) - 0.5) * 0.008 * H +           (P.fbm_lowres(Wp, Hp, Wp / (0.012 * H), 2, seed + 23, q=2) - 0.5) * 0.004 * H
    GWY = (P.fbm_lowres(Wp, Hp, Wp / (0.035 * H), 2, seed + 22, q=2) - 0.5) * 0.008 * H +           (P.fbm_lowres(Wp, Hp, Wp / (0.012 * H), 2, seed + 24, q=2) - 0.5) * 0.004 * H
    order = sorted(range(len(clusters)), key=lambda i: (clusters[i]['amt'] >= 0.9, clusters[i]['y']))
    for ic in order:
        cl = clusters[ic]
        rng = np.random.default_rng(seed * 1000 + ic * 7 + 1)
        pieces = _cumulus(rng, cl) if cl['kind'] == 'cu' else _streamer(rng, cl)
        allc = [c for pc in pieces for c in pc.circ]
        if not allc:
            continue
        T = cl['T']
        pad = int(0.5 * T + 6)
        bx0 = int(math.floor(min(c[0] - c[2] for c in allc))) - pad
        bx1 = int(math.ceil(max(c[0] + c[2] for c in allc))) + pad
        by0 = int(math.floor(min(c[1] - c[2] for c in allc))) - pad
        by1 = int(math.ceil(max(c[1] + c[2] for c in allc))) + pad
        w, h = bx1 - bx0, by1 - by0
        if w <= 4 or h <= 4:
            continue
        # base-line wobble (hard edge, gently irregular)
        xl = (np.arange(w, dtype=np.float32) + bx0) / H
        wob1 = (P.fbm1d(xl * 1.2, 9, 3, seed + ic * 13) - 0.5) * 0.14 * T
        wob = np.broadcast_to(wob1[None, :], (h, w)).astype(np.float32)
        sl0 = (slice(max(by0, 0), min(by1, Hp)), slice(max(bx0, 0), min(bx1, Wp)))
        WX = np.zeros((h, w), np.float32)
        WY = np.zeros((h, w), np.float32)
        WX[sl0[0].start - by0:sl0[0].stop - by0, sl0[1].start - bx0:sl0[1].stop - bx0] = GWX[sl0]
        WY[sl0[0].start - by0:sl0[0].stop - by0, sl0[1].start - bx0:sl0[1].stop - bx0] = GWY[sl0]
        D, Df, Bh = _raster(pieces, bx0, by0, w, h, wob, WX, WY, mg=int(3 + 0.008 * H))
        A = _cov(D)
        # light direction: moon overhead with a pull toward the comet head
        vx, vy = light_xy[0] - cl['x'], light_xy[1] - cl['y']
        nv = math.hypot(vx, vy) + 1e-3
        lx, ly = 0.35 * vx / nv, 0.35 * vy / nv - 1.0
        nl = math.hypot(lx, ly)
        lx, ly = lx / nl, ly / nl
        kt = T * (0.19 if cl['kind'] == 'cu' else 0.22) * rng.uniform(0.85, 1.15)
        kb = T * (0.12 if cl['kind'] == 'cu' else 0.1) * rng.uniform(0.85, 1.15)
        # moonlit top: a flat cel reaching down a VARYING depth from the top silhouette (broad lit masses whose
        # terminator wanders across lobes - never a parallel bevel copy of the outline)
        yl = (np.arange(h, dtype=np.float32) + by0)[:, None]
        nl_ = P.fbm_lowres(w, h, w / (0.55 * T), 3, seed + ic * 31 + 5, q=2) if min(w, h) > 16 else             np.full((h, w), 0.5, np.float32)
        thr = kt * np.clip(1.0 + 3.0 * (nl_ - 0.5), 0.35, 1.5)
        dt = _depth_top(A)
        # smooth the per-column depth sideways so the terminator curves across a tower's flank instead of
        # dropping as a vertical line where the top silhouette steps
        dts = cv2.GaussianBlur(np.minimum(dt, 3 * kt), (0, 0), sigmaX=0.1 * T, sigmaY=0.02 * T)
        lit = A * _cov(np.maximum(dts, 0.6 * dt) - thr)
        # never on the anti-aliased underside pixels (their 'depth from top' is zero)
        lit *= 1 - ((A < 0.98) & (_shiftmap(A, 0, -1.0) > 0.5))
        # hard base band: a flat darker cel along the flat base line, its upper edge gently irregular
        kbx = kb * (1 + 1.0 * (P.fbm1d(xl * 2.0, 11, 3, seed + ic * 17 + 9) - 0.5))
        base = A * _cov(_depth_top(A[::-1])[::-1] - kbx[None, :])
        # front lobes: lighter crescent cap on top of each, darker sickle just under it
        Af = _cov(Df) * A
        kc = 0.14 * T
        crest = Af * (1 - _shiftmap(_cov(Df), lx * kc, ly * kc))
        # the crevice: the mass BEHIND a front lobe is in its own shade just above the lobe's lit cap
        ks = 0.045 * T
        sick = A * (1 - _cov(Df)) * _shiftmap(_cov(Df), -lx * ks, -ly * ks)
        # compose flat cels: body < separation < base < front-lobe cap < top band
        wm = cl.get('warm', 0.0)

        def pc_(k):
            return np.array(pal[k], np.float32)
        body = pc_('body') * (1 - wm) + pc_('w_body') * wm
        basec = pc_('base') * (1 - wm) + pc_('w_base') * wm
        litc = pc_('lit') * (1 - 0.7 * wm) + pc_('w_lit') * 0.7 * wm
        capc = pc_('cap') * (1 - 0.6 * wm) + (pc_('w_lit') * 0.8) * 0.6 * wm
        sepc = pc_('sep') * (1 - wm) + (pc_('w_base') * 1.2) * wm
        col = np.broadcast_to(body, (h, w, 3)).copy()
        for m_, c_ in ((np.clip(sick, 0, 1), sepc), (np.clip(base, 0, 1), basec), (crest, capc),
                       (np.clip(lit, 0, 1), litc)):
            m3 = m_[..., None]
            col = col * (1 - m3) + c_ * m3
        # clip to plate
        ox0, oy0 = max(bx0, 0), max(by0, 0)
        ox1, oy1 = min(bx1, Wp), min(by1, Hp)
        if ox1 <= ox0 or oy1 <= oy0:
            continue
        sl = (slice(oy0 - by0, oy1 - by0), slice(ox0 - bx0, ox1 - bx0))
        dr = drift[oy0:oy1, ox0:ox1, None]
        cc = col[sl] * (0.95 + 0.1 * dr)
        aa = (A[sl] * cl['amt'])[..., None]
        dst = out[oy0:oy1, ox0:ox1]
        a0 = dst[..., 3:4]
        na = aa + a0 * (1 - aa)
        dst[..., :3] = (cc * aa + dst[..., :3] * a0 * (1 - aa)) / np.maximum(na, 1e-5)
        dst[..., 3:4] = na
    return out
