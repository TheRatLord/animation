"""Round-15 night clouds for s07_comet_night: flat moonlit painted shapes (yn_05), no cut-out / emboss look.

Round 14 read as thick white-edged cut-out slabs with a drop-shadow: a narrow lit band copied down from the
top contour (= bright outline), a hard dark base band (= drop shadow), beaded circle chains on the streamers.
Here the silhouettes still come from the round-14 union-of-circles builders (s07_comet_night_cloud14), but:

  * the signed distance is smoothed before thresholding, so bead chains merge into continuous scalloped
    edges (small bumps survive, dotted chains do not),
  * edge quality depends on the facing: crisp 1px anti-aliased edge where the contour faces up / toward the
    moon, a widening lost edge (and a little dry-brush fray) on the underside,
  * value comes from BROAD painted planes, not from the distance to the outline: a moonlit top value on the
    upward / light-facing planes of each big lobe (terminator wanders with low-frequency noise), a darker
    blue-violet body, a bluer base that turns translucent so the stars show through,
  * front lobes get a soft lighter plane (no crescent line), faint dry-brush streaks along the wind,
  * clusters flagged `warm` pick up the orange afterglow on their undersides.
"""
import math

import numpy as np
import cv2

from lib import core as C
import s07_comet_night_paint as P
import s07_comet_night_cloud14 as K14




def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _aniso(w, h, along, across, ang, seed, octaves=4):
    """fbm (0..1) stretched along direction `ang` (deg): brush-stroke-like streaks; along/across in px."""
    N = 256
    base = P.fbm_lowres(N, N, 8, octaves, seed, q=1)
    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
    cell = N / 8.0
    # output (x, y) -> noise (u, v): u along the stroke, v across it
    M = np.array([[ca * cell / along, sa * cell / along, 17.0],
                  [-sa * cell / across, ca * cell / across, 23.0]], np.float32)
    return cv2.warpAffine(base, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_WRAP)


def night_clouds15(Wp, Hp, clusters, s, H, light_xy, seed=17, pal=None):
    """Same contract as K14.night_clouds14: returns straight RGBA (Hp, Wp, 4)."""
    if pal is None:
        pal = dict(top=(0.60, 0.78, 0.90), mid=(0.44, 0.61, 0.80), body=(0.33, 0.46, 0.70),
                   base=(0.26, 0.33, 0.60), warm=(0.98, 0.62, 0.45), warm_top=(0.95, 0.80, 0.80))
    out = np.zeros((Hp, Wp, 4), np.float32)
    drift = P.fbm_lowres(Wp, Hp, Wp / (0.25 * H), 2, seed + 5, q=8)
    GWX = (P.fbm_lowres(Wp, Hp, Wp / (0.035 * H), 2, seed + 21, q=2) - 0.5) * 0.009 * H + \
          (P.fbm_lowres(Wp, Hp, Wp / (0.010 * H), 2, seed + 23, q=2) - 0.5) * 0.004 * H
    GWY = (P.fbm_lowres(Wp, Hp, Wp / (0.035 * H), 2, seed + 22, q=2) - 0.5) * 0.009 * H + \
          (P.fbm_lowres(Wp, Hp, Wp / (0.010 * H), 2, seed + 24, q=2) - 0.5) * 0.004 * H
    order = sorted(range(len(clusters)), key=lambda i: (clusters[i]['amt'] >= 0.9, clusters[i]['y']))
    for ic in order:
        cl = clusters[ic]
        rng = np.random.default_rng(seed * 1000 + ic * 7 + 1)
        pieces = K14._cumulus(rng, cl) if cl['kind'] == 'cu' else K14._streamer(rng, cl)
        allc = [c for pc in pieces for c in pc.circ]
        if not allc:
            continue
        T = cl['T']
        pad = int(0.6 * T + 8)
        bx0 = int(math.floor(min(c[0] - c[2] for c in allc))) - pad
        bx1 = int(math.ceil(max(c[0] + c[2] for c in allc))) + pad
        by0 = int(math.floor(min(c[1] - c[2] for c in allc))) - pad
        by1 = int(math.ceil(max(c[1] + c[2] for c in allc))) + pad
        w, h = bx1 - bx0, by1 - by0
        if w <= 4 or h <= 4:
            continue
        xl = (np.arange(w, dtype=np.float32) + bx0) / H
        wob1 = (P.fbm1d(xl * 1.2, 9, 3, seed + ic * 13) - 0.5) * 0.14 * T
        wob = np.broadcast_to(wob1[None, :], (h, w)).astype(np.float32)
        WX = np.zeros((h, w), np.float32)
        WY = np.zeros((h, w), np.float32)
        ya, yb_ = max(by0, 0), min(by1, Hp)
        xa, xb_ = max(bx0, 0), min(bx1, Wp)
        WX[ya - by0:yb_ - by0, xa - bx0:xb_ - bx0] = GWX[ya:yb_, xa:xb_]
        WY[ya - by0:yb_ - by0, xa - bx0:xb_ - bx0] = GWY[ya:yb_, xa:xb_]
        D, Df, _ = K14._raster(pieces, bx0, by0, w, h, wob, WX, WY, mg=int(3 + 0.008 * H))
        st = cl['kind'] == 'st'
        fine = (P.fbm_lowres(w, h, w / (0.018 * H + 1), 3, seed + ic * 43 + 1, q=1) - 0.5) if min(w, h) > 16 \
            else np.zeros((h, w), np.float32)
        fine2 = (P.fbm_lowres(w, h, w / (0.007 * H + 1), 2, seed + ic * 47 + 2, q=1) - 0.5) if min(w, h) > 16 \
            else np.zeros((h, w), np.float32)
        # ---- merge bead chains: smooth the (clamped) distance field before thresholding; fine painted
        # breakup of the contour (small bumps / dry-brush nicks, never a vector-smooth arc)
        Dc = np.clip(D, -T, T)
        Ds = cv2.GaussianBlur(Dc, (0, 0), (0.035 if st else 0.018) * T + 0.6)
        Ds = Ds + (fine * 0.006 + fine2 * 0.005) * H
        # torn, broken shapes (yn_05): noise erosion that bites bays, holes and detached fragments into the
        # thin outer parts of each cloud while its thick core stays whole; the cut stays a crisp edge
        if min(w, h) > 16:
            hn = 0.65 * (_aniso(w, h, 0.38 * T + 4, 0.14 * T + 2, cl['ang'], seed + ic * 53 + 7, 3) - 0.5) + \
                (0.0 if st else 0.35) * (P.fbm_lowres(w, h, w / (0.2 * T + 2), 3, seed + ic * 57 + 9, q=2) - 0.5) * 1.6
            if not st:
                # rounded, tapering ends instead of the vertical cut sides of a flat-based slab
                ca_, sa_ = math.cos(math.radians(cl['ang'])), math.sin(math.radians(cl['ang']))
                xg = np.arange(w, dtype=np.float32)[None, :] + bx0 - cl['x']
                yg = np.arange(h, dtype=np.float32)[:, None] + by0 - cl['y']
                qq = np.abs(xg * ca_ + yg * sa_) / (0.5 * cl['L'] + 0.5 * T)
                up_ = np.clip((cl['y'] + 0.32 * T - (yg + cl['y'])) / T, 0, 1.5)
                Ds = Ds + 0.75 * T * _ss(0.4, 1.0, qq) ** 1.3 * (0.2 + up_)
            core = _ss(0.02 * T, (0.25 if st else 0.12) * T, -Ds)
            Ds = Ds + (hn * (0.42 if st else 0.24) * T + (0.02 if st else 0.0) * T) * (1 - core)
        # ---- facing: broad normal of the silhouette
        # fill small enclosed holes (dark 'moth holes' read as blotches, not sky gaps); gaps that open to
        # the outside stay
        outside = (Ds > 0).astype(np.uint8)
        n_, lab, stt, _ = cv2.connectedComponentsWithStats(outside, connectivity=4)
        if n_ > 2:
            amax = (0.6 * T) ** 2
            bord = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
            fill = np.zeros(n_, bool)
            for k_ in range(1, n_):
                if k_ not in bord and stt[k_, cv2.CC_STAT_AREA] < amax:
                    fill[k_] = True
            fm = fill[lab]
            if fm.any():
                fm = cv2.dilate(fm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
                Ds = np.where(fm, np.minimum(Ds, -1.5), Ds).astype(np.float32)
        A0 = np.clip(0.5 - Ds, 0, 1)
        G = cv2.GaussianBlur(A0, (0, 0), 0.10 * T + 1)
        gx = cv2.Sobel(G, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(G, cv2.CV_32F, 0, 1, ksize=3)
        gn = np.sqrt(gx * gx + gy * gy) + 1e-6
        ny = -gy / gn
        edge_w = np.clip(gn / (gn.max() + 1e-6) * 4, 0, 1)
        down = np.clip(ny + 0.2, 0, 1) / 1.2 * edge_w       # underside-facing
        # ---- light direction: moon overhead, pulled toward the comet
        vx, vy = light_xy[0] - cl['x'], light_xy[1] - cl['y']
        nv = math.hypot(vx, vy) + 1e-3
        lx, ly = 0.35 * vx / nv, 0.35 * vy / nv - 1.0
        nl = math.hypot(lx, ly)
        lx, ly = lx / nl, ly / nl
        # ---- alpha: crisp lit edge (1px AA), lost soft frayed edge on the underside
        fray = (P.fbm_lowres(w, h, w / (0.10 * T + 2), 3, seed + ic * 41 + 3, q=2) - 0.5) if min(w, h) > 16 \
            else np.zeros((h, w), np.float32)
        soft = 0.8 + (0.14 if st else 0.10) * T * down ** 1.5
        Dd = Ds + fray * 0.12 * T * down
        A = np.clip(0.5 - Dd / soft, 0, 1)
        # dry-brush: the outer margin of the paint is thin in places (stars show through), mostly underneath
        inner = _ss(0.0, 0.07 * T + 2, -Ds)
        thin = np.clip(0.6 + 1.4 * fine2 + 1.2 * fine, 0, 1)
        A = A * (inner + (1 - inner) * (1 - (0.25 + 0.55 * down) * (1 - thin)))
        # ---- value: a soft moonlit gradient, not cels
        dt = K14._depth_top(A0)                                   # depth under the top silhouette
        db = K14._depth_top(A0[::-1])[::-1]                       # height above the underside
        thick = dt + db + 1.0
        hf = np.clip(dt, 0, None) / np.maximum(thick, 1.0)        # 0 at top, 1 at base
        m_ = (A0 > 0.5).astype(np.float32)
        sgx, sgy = 0.35 * T + 1, 0.1 * T + 1
        hf = cv2.GaussianBlur(hf * m_, (0, 0), sigmaX=sgx, sigmaY=sgy) / \
            np.maximum(cv2.GaussianBlur(m_, (0, 0), sigmaX=sgx, sigmaY=sgy), 1e-3)
        hf = np.where(m_ > 0, hf, 1.0).astype(np.float32)
        # broad light-facing planes from a heavily smoothed shape
        G2 = cv2.GaussianBlur(A0, (0, 0), 0.2 * T + 1)
        g2x = cv2.Sobel(G2, cv2.CV_32F, 1, 0, ksize=3)
        g2y = cv2.Sobel(G2, cv2.CV_32F, 0, 1, ksize=3)
        g2n = np.sqrt(g2x ** 2 + g2y ** 2) + 1e-6
        face = (-(g2x * lx + g2y * ly) / g2n) * np.clip(g2n / (np.percentile(g2n, 95) + 1e-6), 0, 1)
        # sub-lobes: a soft lit upper side and a soft shade under each front lobe (lobe-on-lobe structure
        # read through value only, no line)
        # per-puff soft planes: the biggest lobes each get a soft lighter light-side and a soft shade on the
        # far side (painted puffs, low weight, wide soft transitions)
        big = sorted([c for pc in pieces for c in pc.circ if c[2] > (0.4 if st else 0.16) * T], key=lambda c: -c[2])
        # pick well separated puffs
        pk = []
        for c in big:
            if all(math.hypot(c[0] - q[0], c[1] - q[1]) > 0.8 * max(c[2], q[2]) for q in pk):
                pk.append(c)
            if len(pk) >= 9:
                break
        lpos = np.zeros((h, w), np.float32)
        lneg = np.zeros((h, w), np.float32)
        yy_, xx_ = np.mgrid[0:h, 0:w].astype(np.float32)
        for (cx_, cy_, r_, _) in pk:
            u = xx_ + WX * 2.0 + bx0 - cx_
            v = yy_ + WY * 2.0 + by0 - cy_
            dd = np.sqrt(u * u + v * v) / r_
            disc = _ss(1.15, 0.75, dd)
            proj = (u * lx + v * ly) / r_            # +1 toward the light
            lpos = np.maximum(lpos, disc * _ss(-0.1, 0.8, proj))
            lneg = np.maximum(lneg, disc * _ss(0.1, -0.9, proj) * _ss(0.55, 1.0, dd))
        lobe = (lpos - 0.7 * lneg) * m_
        # billow noise: soft internal painted variation (puffs), scaled to the cloud
        nz = P.fbm_lowres(w, h, w / (0.28 * T + 2), 3, seed + ic * 31 + 5, q=2) if min(w, h) > 16 \
            else np.full((h, w), 0.5, np.float32)
        light = 0.2 + 0.5 * (1 - hf) ** 1.3 + 0.22 * face + 0.15 * lobe + 0.34 * (nz - 0.5) + 0.05 * fine
        brush = _aniso(w, h, 0.3 * T + 3, 0.05 * T + 1.5, cl['ang'] * 0.6, seed + ic * 59 + 11, 3) - 0.5
        light = light + 0.12 * brush
        # small painted dabs: lighter flecks in the lit upper body, darker ones low (brush texture, not noise)
        if min(w, h) > 16:
            dab = P.fbm_lowres(w, h, w / (0.006 * H + 1), 2, seed + ic * 61 + 13, q=1)
            fl = _ss(0.6, 0.68, dab)
            light = light + 0.1 * fl * _ss(0.35, 0.6, light) - 0.07 * _ss(0.4, 0.32, dab) * _ss(0.5, 0.25, light)
        light = np.clip(light, 0, 1.2)
        wm = float(cl.get('warm', 0.0))
        cP = {k: np.array(v, np.float32) for k, v in pal.items()}
        top = cP['top'] * (1 - 0.35 * wm) + cP['warm_top'] * 0.35 * wm
        # smooth ramp base -> body -> mid -> top (soft painted transitions)
        stops = [(0.0, cP['base']), (0.3, cP['body']), (0.6, cP['mid']), (0.95, top)]
        col = np.zeros((h, w, 3), np.float32)
        for c in range(3):
            col[..., c] = np.interp(light, [q for q, _ in stops], [v[c] for _, v in stops])
        # warm afterglow on the undersides of the low clouds
        if wm > 0:
            wk = (wm * _ss(0.2, 0.85, hf) * (0.4 + 0.6 * _ss(0.9, 0.3, light)))[..., None]
            wk = np.clip(wk * 1.4, 0, 0.85)
            col = col * (1 - wk) + cP['warm'] * wk
        # translucent lower body: stars show through the base
        tr = 1 - (0.5 if st else 0.4) * _ss(0.45, 1.0, hf) * _ss(0.6, 0.2, light)
        A = A * tr
        # ---- composite into the plate
        ox0, oy0, ox1, oy1 = xa, ya, xb_, yb_
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
