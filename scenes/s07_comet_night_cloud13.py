"""Round-13 night clouds for s07_comet_night (yn_05 painted night clouds).

Round 12 read as low-poly crumpled-paper chips (angular value facets, an extruded dark base lip) and every cloud
was the same evenly-spaced lozenge.  Here the clouds are painted from ONE continuous density field:

  * clusters of irregular cumulus masses and torn diagonal stratocumulus STREAMERS that follow the comet's
    direction (yn_05 upper-left), grouped into a few broken banks + scattered scraps, never tiled,
  * the silhouette is the zero-crossing of (envelope + cauliflower noise), so edges are ragged and rounded at
    two scales, with bays, holes and detached scraps; the TOP edge is crisp,
  * lighting is a smooth, feathered ramp: one broad moonlit cyan-white upper mass per cloud (large-scale
    density falloff toward the light) with a faint small-scale lift on the budding tops - no facets, no steps,
  * one flat navy-violet body value below it,
  * the underside is LOST: alpha thins out toward the lower edge so stars and sky show through; no base band,
    no drop-shadow lip, no dark smear,
  * per-cluster opacity: far / high thin clouds are partly transparent.
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P


def _ss(a, b, x):
    return C.smoothstep(a, b, x)


def _shift(U, dx, dy):
    Hs, Ws = U.shape
    gx, gy = np.meshgrid(np.arange(Ws, dtype=np.float32), np.arange(Hs, dtype=np.float32))
    return cv2.remap(U, (gx + dx).astype(np.float32), (gy + dy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _cluster_blobs(rng, cl, H):
    """Ellipses (cx, cy, ru, rv, ang_deg, weight) for one cluster."""
    x, y, L, T, ang, kind = cl['x'], cl['y'], cl['L'], cl['T'], cl['ang'], cl['kind']
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    out = []
    if kind == 'cu':
        # irregular cumulus mass: big blobs along a short spine (heights vary -> uneven crown), then two
        # generations of rounded buds of very different sizes on the light-facing upper rim (cauliflower domes)
        n = max(2, int(round(L / (0.8 * T))))
        base = []
        for i in range(n):
            q = (i + 0.5) / n - 0.5 + rng.normal(0, 0.07)
            e = max(1 - (2 * q) ** 2, 0.1) ** 0.5
            r = T * rng.uniform(0.35, 0.75) * (0.55 + 0.55 * e)
            cx = x + q * L * ca
            cy = y + q * L * sa - r * rng.uniform(0.1, 0.7) * e
            base.append((cx, cy, r))
            out.append((cx, cy, r * rng.uniform(0.95, 1.25), r * rng.uniform(0.8, 1.0), ang * 0.3, 1.0))
        gen = base
        for g in range(2):
            nxt = []
            for (cx, cy, r) in gen:
                for k in range(rng.integers(1, 4)):
                    a = math.radians(rng.uniform(-155, -25))
                    rr = r * rng.uniform(0.3, 0.6)
                    d = r * rng.uniform(0.6, 0.9)
                    bx, by = cx + d * math.cos(a) * 1.15, cy + d * math.sin(a) * 0.8
                    nxt.append((bx, by, rr))
                    out.append((bx, by, rr, rr * 0.9, 0.0, 0.95))
            gen = nxt
    else:
        # torn streamer: many elongated scraps strung along the spine with gaps, thinning toward the tail end
        n = max(3, int(round(L / (0.7 * T))))
        q = -0.5
        while q < 0.5:
            taper = 1 - 0.55 * (q + 0.5) if cl.get('taper', True) else 1.0
            ru = T * rng.uniform(0.9, 2.2) * taper
            rv = T * rng.uniform(0.28, 0.55) * taper
            off = rng.normal(0, 0.35) * T
            cx = x + q * L * ca - off * sa
            cy = y + q * L * sa + off * ca
            a2 = ang + rng.normal(0, 7)
            out.append((cx, cy, ru, rv, a2, rng.uniform(0.75, 1.0)))
            if rng.random() < 0.45:     # a small puffy bud riding on the streamer
                out.append((cx + rng.normal(0, 0.3) * ru, cy - rv * 0.6, rv * 1.1, rv * 0.9, 0.0, 0.85))
            q += rng.uniform(0.35, 0.9) * ru / L * 1.6
    return out


def night_clouds13(Wp, Hp, clusters, s, H, light_xy, seed=17, pal=None):
    """clusters: dicts with x, y (plate px), L, T (px), ang (deg), kind ('cu' | 'st'), amt, warm.
    Returns straight RGBA (Hp, Wp, 4)."""
    if pal is None:
        pal = dict(lit=(0.82, 0.93, 1.0), mid=(0.55, 0.67, 0.84), body=(0.38, 0.47, 0.68),
                   warm=(0.66, 0.46, 0.6), warm_lit=(1.0, 0.86, 0.86))
    # shared noise fields (plate space): billowy (|2n-1|) noise gives round outward bumps with sharp inward
    # cusps between them = the scalloped cauliflower edge; domain warp breaks any lozenge regularity
    q = 2
    w1 = (P.fbm_lowres(Wp, Hp, Wp / (0.08 * H), 3, seed + 11, q=4) - 0.5) * 0.05 * H
    w2 = (P.fbm_lowres(Wp, Hp, Wp / (0.08 * H), 3, seed + 12, q=4) - 0.5) * 0.05 * H

    def bil(scale_px, octs, sd, qq=1):
        n = P.fbm_lowres(Wp, Hp, Wp / scale_px, octs, sd, q=qq) if qq > 1 else C.fbm(Wp, Hp, Wp / scale_px, octs, seed=sd)
        return np.sqrt((2 * n - 1) ** 2 + 0.03)       # creases slightly rounded (no crack lines)
    n1 = C.warp(bil(0.05 * H, 3, seed + 1, 2), w1, w2)                      # big lobes
    nm = C.warp(bil(0.02 * H, 3, seed + 4, 2), w1, w2)                      # mid lobes
    n2 = C.warp(bil(0.013 * H, 2, seed + 2), w1 * 0.5, w2 * 0.5)            # fine scallops
    nt = C.fbm(Wp, Hp, Wp / (0.012 * H), 3, seed=seed + 7)                  # translucency mottling
    na = P.rot_fbm(Wp // q, Hp // q, 28.0, 4.0, Wp / (0.03 * H), 4, seed + 3, warp_amt=0.03)
    na = cv2.resize(na, (Wp, Hp), interpolation=cv2.INTER_CUBIC)            # streaky, along the comet
    D = np.full((Hp, Wp), -1.0, np.float32)
    AMT = np.zeros((Hp, Wp), np.float32)
    WARM = np.zeros((Hp, Wp), np.float32)
    rng0 = np.random.default_rng(seed)
    for ic, cl in enumerate(clusters):
        rng = np.random.default_rng(seed * 1000 + ic * 7 + 1)
        blobs = _cluster_blobs(rng, cl, H)
        pad = 3.0
        bx0 = int(max(min(b[0] - pad * b[2] for b in blobs), 0))
        bx1 = int(min(max(b[0] + pad * b[2] for b in blobs), Wp))
        by0 = int(max(min(b[1] - pad * b[2] for b in blobs), 0))
        by1 = int(min(max(b[1] + pad * b[2] for b in blobs), Hp))
        if bx1 <= bx0 or by1 <= by0:
            continue
        yy, xx = np.mgrid[by0:by1, bx0:bx1].astype(np.float32)
        E = np.zeros_like(xx)
        for (cx, cy, ru, rv, a, w) in blobs:
            c_, s_ = math.cos(math.radians(a)), math.sin(math.radians(a))
            du = (xx - cx) * c_ + (yy - cy) * s_
            dv = -(xx - cx) * s_ + (yy - cy) * c_
            e = np.exp(-1.1 * ((du / ru) ** 2 + (dv / rv) ** 2)) * w
            E = 1 - (1 - E) * (1 - e)
        sl = (slice(by0, by1), slice(bx0, bx1))
        if cl['kind'] == 'cu':
            # ragged but bounded underside: nothing hangs far below the cluster's base line
            E = E * _ss(cl['y'] + 0.7 * cl['T'], cl['y'] - 0.05 * cl['T'], yy + 0.25 * cl['T'] * (n1[sl] - 0.5))
            d = E - 0.4 + 0.5 * (n1[sl] - 0.5) + 0.3 * (nm[sl] - 0.5) + 0.26 * (n2[sl] - 0.5)
        else:
            d = E - 0.52 + 0.7 * (na[sl] - 0.5) + 0.4 * (nm[sl] - 0.5) + 0.18 * (n2[sl] - 0.5)
        d = d * cl.get('dens', 1.0)
        upd = d > D[sl]
        D[sl] = np.where(upd, d, D[sl])
        AMT[sl] = np.where(upd & (d > -0.2), cl['amt'], AMT[sl])
        WARM[sl] = np.where(upd & (d > -0.2), cl.get('warm', 0.0), WARM[sl])
    del rng0
    # spread the per-cluster attributes a little so they never step inside a cloud
    # (normalised blur: no fade toward the silhouette, which must stay crisp)
    msk = (D > -0.2).astype(np.float32)
    mb = cv2.GaussianBlur(msk, (0, 0), 0.01 * H) + 1e-4
    AMT = np.where(msk > 0, cv2.GaussianBlur(AMT * msk, (0, 0), 0.01 * H) / mb, 0)
    WARM = np.where(msk > 0, cv2.GaussianBlur(WARM * msk, (0, 0), 0.01 * H) / mb, 0)

    # ---- thickness + lighting (continuous)
    Th = _ss(0.0, 0.3, cv2.GaussianBlur(D, (0, 0), 0.003 * H))
    h1 = cv2.GaussianBlur(Th, (0, 0), 0.009 * H)
    h2 = cv2.GaussianBlur(Th, (0, 0), 0.03 * H)
    # light direction per pixel: mostly overhead (moon) with a pull toward the comet head
    xs, ys = C.grid(Wp, Hp)
    vx, vy = light_xy[0] - xs, light_xy[1] - ys
    n_ = np.sqrt(vx * vx + vy * vy) + 1e-3
    lx = 0.4 * vx / n_
    ly = 0.4 * vy / n_ - 1.0
    n_ = np.sqrt(lx * lx + ly * ly)
    lx, ly = lx / n_, ly / n_
    del xs, ys, vx, vy
    k1, k2 = 0.022 * H, 0.05 * H
    g1 = h1 - _shift(h1, lx * k1, ly * k1)
    g2 = h2 - _shift(h2, lx * k2, ly * k2)
    h0 = cv2.GaussianBlur(_ss(0.0, 0.25, D), (0, 0), 0.004 * H)
    k0 = 0.009 * H
    g0 = h0 - _shift(h0, lx * k0, ly * k0)
    self_ = globals().get('_TUNE', {})
    lit = 0.3 + self_.get('a2', 0.55) * g2 + self_.get('a1', 1.4) * g1 + self_.get('a0', 0.3) * g0 + self_.get('an', 0.22) * (nm - 0.5) +         0.08 * (nt - 0.5)
    # painted caps: each lit patch has a CRISP upper (light-side) edge and fades out softly downward
    # (a hard brush stroke dragged down), plus a softer mid-value underlayer
    lit = cv2.GaussianBlur(lit, (0, 0), 0.0015 * H + 0.3)
    if self_.get('dbg'):
        m_ = D > 0.05
        print('lit pct', np.percentile(lit[m_], [20, 40, 50, 60, 70, 80]))
    gx_ = cv2.Sobel(lit, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy_ = cv2.Sobel(lit, cv2.CV_32F, 0, 1, ksize=3) / 8
    gn_ = np.sqrt(gx_ * gx_ + gy_ * gy_) + 1e-5
    # 'down' = this iso-edge is the patch's lower (away-from-light) side -> lost edge; upper side stays crisp
    down = _ss(-0.1, 0.7, -(gx_ * lx + gy_ * ly) / gn_ * -1.0)
    wpx = 1.0 + self_.get('soft', 0.012) * H * down
    cap = _ss(-0.5, 0.5, (lit - self_.get('t1', 0.68)) / (gn_ * wpx))
    mid_ = _ss(-0.5, 0.5, (lit - self_.get('t0', 0.42)) / (gn_ * (1.0 + 0.02 * H * down)))
    lit = np.maximum(cap, 0.45 * mid_)
    lit = cv2.GaussianBlur(lit, (0, 0), 0.0008 * H + 0.3)
    # lower edge: thins out (lost) where density drops going down
    kb = 0.03 * H
    hb = cv2.GaussianBlur(Th, (0, 0), 0.015 * H)
    bot = np.clip((hb - _shift(hb, 0, kb)) * 2.2, 0, 1)
    bot = _ss(0.05, 0.7, bot) * (1 - lit)
    # silhouette crisp (about 1px); opacity follows thickness: thin parts are translucent, mottled
    gm = np.sqrt(cv2.Sobel(D, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(D, cv2.CV_32F, 0, 1, ksize=3) ** 2) / 8 + 1e-4
    wdt = gm * (1.2 + 5.0 * bot)
    A = _ss(-0.5, 0.5, D / wdt)
    thick = np.clip(_ss(0.0, 0.1, D) * (0.85 + 0.3 * nt), 0, 1)
    A = A * (0.6 + 0.4 * thick) * (1 - 0.55 * bot)
    A = np.clip(A * AMT, 0, 1)

    lit_c = np.array(pal['lit'], np.float32)
    mid_c = np.array(pal['mid'], np.float32)
    body_c = np.array(pal['body'], np.float32)
    l3 = lit[..., None]
    col = np.where(l3 < 0.5, body_c + (mid_c - body_c) * (l3 / 0.5),
                   mid_c + (lit_c - mid_c) * ((l3 - 0.5) / 0.5))
    wm = WARM[..., None]
    # low twilight clouds: dusky rose on the lower body, faintly warm tops
    col = col + wm * (1 - l3) * (np.array(pal['warm'], np.float32) - body_c) * 0.8
    col = col + wm * l3 * (np.array(pal['warm_lit'], np.float32) - lit_c) * 0.35
    # faint dry-brush variation (painted, not rendered)
    col = col * (1 + 0.06 * (nt[..., None] - 0.5))
    return np.dstack([col, A]).astype(np.float32)
