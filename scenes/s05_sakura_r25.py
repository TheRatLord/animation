"""s05_sakura round 25 (reviewer FAIL on round 24: 'toothpaste ribbons of even width following every branch, flat
saturated bubble-gum mid band, violet shade built from identical 5-petal stamps, twig spikes past the silhouette,
flat pink smear on the mid-distance row').

  * Tree25: the blossom is regrouped into separate, rounded, uneven CLUMPS hanging off the branches (cm5_01):
    clump seeds are Poisson-spaced along the limbs with log-normal sizes, every cluster is kept only inside the
    lobed (cauliflower) outline of its clump, the clumps are filled out to a round hanging shape, and the stretches
    of limb between clumps stay bare so sky shows between them;
  * Painter25: shade-side flowers are varied (size, rotation, petal count, softer petals) and merge into a soft
    mass; fewer patch holes (the clump gaps already let the sky through);
  * regrade25: value painted continuously per clump from the sun (top-left): near-white lit tips -> pale greyish
    sakura pink body -> violet-grey underside, deeper rose only in small pockets; the shade interior is closed into
    a soft lost-edge mass (crisp florets only on the silhouette); twig stubs past the silhouette cut shorter and
    tapered.
"""
import math

import numpy as np
import cv2

import s05_sakura_r23 as R23
import s05_sakura_r24 as R24

_ss = R24._ss
_shift = R24._shift


# ============================================================================ clumped tree
class Tree25(R23.Tree23):
    CLUMP_R = (0.95, 0.35)        # log-normal clump radius (x median cluster radius): mean, sigma
    GAP = (0.85, 1.15)            # seed spacing factor (x sum of radii)
    FILL = 1.1
    seeds = ()
    rc_med = None

    def __init__(self, rng, *a, clump=1.0, **kw):
        super().__init__(rng, *a, **kw)
        if clump > 0:
            self._clump(np.random.default_rng(int(rng.integers(1 << 30))))

    def _clump(self, rng):
        cl = self.clusters
        forced = [c for c in cl if c.get('force_front')]
        free = [c for c in cl if not c.get('force_front')]
        n = len(free)
        if n < 6:
            return
        X = np.array([c['x'] for c in free])
        Y = np.array([c['y'] for c in free])
        Rm = float(np.median([c['R'] for c in free]))
        mu, sg = self.CLUMP_R
        seeds = []
        order = rng.permutation(n)
        for i in order:
            rc = Rm * float(np.clip(math.exp(rng.normal(mu, sg)), 1.3, 5.0))
            ok = True
            for (sx, sy, sr, _, _, _) in seeds:
                if math.hypot(X[i] - sx, Y[i] - sy) < (rc + sr) * rng.uniform(*self.GAP):
                    ok = False
                    break
            if ok:
                seeds.append((float(X[i]), float(Y[i]) - 0.25 * rc, rc, rng.uniform(0, 6.3), rng.uniform(0, 6.3),
                              int(rng.integers(3, 6))))
        if not seeds:
            return
        S = np.array([[s[0], s[1], s[2]] for s in seeds])

        def lobed(dx, dy, s):
            th = np.arctan2(dy, dx)
            rr = s[2] * (1.0 + 0.24 * np.cos(s[5] * th + s[3]) + 0.14 * np.cos((2 * s[5] + 1) * th + s[4]))
            # hanging: a bit wider than tall, rounder underneath
            return np.hypot(dx, dy * 1.12) / rr
        D = np.stack([lobed(X - s[0], Y - s[1], s) for s in seeds], 1)
        j = np.argmin(D, 1)
        dmin = D[np.arange(n), j]
        keep = dmin < 1.0
        out = [c for c, k in zip(free, keep) if k]
        # fill each clump out to a round hanging cauliflower shape
        for js, s in enumerate(seeds):
            mem = np.nonzero(keep & (j == js))[0]
            target = int(self.FILL * (s[2] / Rm) ** 2 * 0.9) + 1
            add = max(0, target - len(mem))
            if len(mem) == 0:
                continue
            for _ in range(add):
                for _t in range(8):
                    a = rng.uniform(0, 2 * math.pi)
                    r = s[2] * math.sqrt(rng.uniform(0.0, 1.0)) * 0.85
                    px_, py_ = s[0] + r * math.cos(a), s[1] + r * math.sin(a) / 1.12
                    if lobed(np.array([px_ - s[0]]), np.array([py_ - s[1]]), s)[0] < 0.9:
                        break
                q = free[mem[int(np.argmin(np.hypot(X[mem] - px_, Y[mem] - py_)))]]
                out.append(dict(x=float(px_), y=float(py_), R=float(Rm * rng.uniform(0.75, 1.2)), m=q['m'], d=q['d'],
                                tx=q['tx'], ty=q['ty'], front=bool(rng.random() < 0.45),
                                seed=int(rng.integers(1 << 30))))
        self.clusters = out + forced
        self.seeds = seeds
        self.rc_med = float(np.median([q[2] for q in seeds]))


def clumps_px(tree, k, ox, oy):
    """clump seeds of a Tree25 in card pixels"""
    return [(ox + c[0] * k, oy - c[1] * k, c[2] * k, c[3], c[4], c[5]) for c in getattr(tree, 'seeds', [])]


# ============================================================================ painter
class Painter25(R23.Painter23):
    HOLES = (0, 2)
    HOLE_R = (0.7, 1.4)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.detail = self.detail * 0.5          # half the floret sprays on thin twigs past the silhouette

    def _cluster_stamps(self, c, v, trans, rng):
        st = super()._cluster_stamps(c, v, trans, rng)
        px = self.px
        out = []
        for q in st:
            lt = q[17] if len(q) > 17 else 0.0
            if lt < 2.0 and q[4] == 5.0 and q[2] > 1.2 * px:
                # shade-side flower: varied size / rotation / petal count, soft petals -> merges into a mass
                q = list(q)
                q[2] = q[2] * math.exp(rng.normal(0.15, 0.3))
                q[3] = rng.uniform(0, 6.3)
                q[4] = float(rng.choice([4.0, 5.0, 6.0, 7.0]))
                q[5] = rng.uniform(0.15, 0.42)
                q[6] = q[6] * rng.uniform(0.9, 1.25)
                q = tuple(q)
            out.append(q)
        return out


# ============================================================================ value pass
P_DEEP = np.array([0.55, 0.43, 0.62], np.float32)     # violet-grey core
P_SHADE = np.array([0.7, 0.55, 0.74], np.float32)     # violet-grey underside
P_SHM = np.array([0.89, 0.62, 0.74], np.float32)      # shade -> body transition (greyed mauve pink)
P_MID = np.array([0.98, 0.72, 0.78], np.float32)      # pale greyish sakura pink body
P_LIT2 = np.array([1.0, 0.84, 0.86], np.float32)     # light pink
P_LIT = np.array([1.0, 0.93, 0.93], np.float32)       # near-white lit tips
P_ROSE = np.array([0.9, 0.48, 0.62], np.float32)      # deeper rose accent (small pockets only)
STOPS = ((0.0, P_DEEP), (0.08, P_SHADE), (0.24, P_SHADE), (0.36, P_SHM), (0.5, P_MID), (0.72, P_LIT2), (0.9, P_LIT))


def _ramp(u):
    col = np.zeros(u.shape + (3,), np.float32)
    col[:] = STOPS[0][1]
    for (u0, c0), (u1, c1) in zip(STOPS[:-1], STOPS[1:]):
        k = _ss((u - u0) / max(u1 - u0, 1e-4))
        col += (c1 - c0)[None, None] * k[..., None]
    return col


def _dab_map(sel, px, rng):
    """random round dabs (2-6 screen px across) with a random value each, later dabs overwrite earlier ones"""
    H, W = sel.shape
    dm = np.full((H, W), 0.5, np.float32)
    ys, xs = np.nonzero(sel[::2, ::2])
    if len(ys) == 0:
        return dm
    r_mean = 2.2 * px
    n = int(len(ys) * 4 / (math.pi * r_mean * r_mean) * 2.2)
    pick = rng.integers(0, len(ys), n)
    rad = np.clip(r_mean * np.exp(rng.normal(0, 0.35, n)), 1.0 * px, 4.5 * px)
    val = rng.random(n)
    cx = xs[pick] * 2 + rng.integers(0, 2, n)
    cy = ys[pick] * 2 + rng.integers(0, 2, n)
    for i in range(n):
        cv2.circle(dm, (int(cx[i]), int(cy[i])), int(round(rad[i])), float(val[i]), -1, cv2.LINE_8)
    return dm


def regrade25(pm, sun_dir, px, far=False, trim=True, clump_px=None, clumps=None):
    pm = pm.astype(np.float32)
    H, W = pm.shape[:2]
    A = pm[..., 3]
    C = pm[..., :3] / np.maximum(A, 1e-4)[..., None]
    r, g, b = C[..., 0], C[..., 1], C[..., 2]
    lum = C.mean(-1)
    pink = np.clip((b - g + 0.02) / 0.05, 0, 1) * np.clip((r - g - 0.03) / 0.05, 0, 1) * \
        np.clip((lum - 0.4) / 0.1, 0, 1)
    Bw = pink * np.clip((A - 0.25) / 0.5, 0, 1)
    wood = ((A > 0.3) & (pink < 0.3) & (lum < 0.55)).astype(np.float32)
    sel = Bw > 0.5
    if sel.sum() < 50:
        return pm
    q = 4
    hq, wq = max(1, H // q), max(1, W // q)
    Bq = cv2.resize(Bw, (wq, hq), interpolation=cv2.INTER_AREA)
    lx, ly = float(sun_dir[0]), float(sun_dir[1]) - 0.7
    ln = math.hypot(lx, ly) + 1e-9
    lx, ly = lx / ln, ly / ln
    # clump scale occlusion (how much blossom lies toward the sun), + mass scale
    s1 = (0.55 * clump_px if clump_px else 7.0 * px) / q
    s1 = float(np.clip(s1, 4.0 * px / q, 40.0 * px / q))
    s2 = max(22.0 * px / q, 2.5 * s1)
    b1 = cv2.GaussianBlur(Bq, (0, 0), s1)
    b2 = cv2.GaussianBlur(Bq, (0, 0), s2)
    e1 = _shift(b1, lx * 1.4 * s1, ly * 1.4 * s1)
    e2 = _shift(b2, lx * 1.2 * s2, ly * 1.2 * s2)
    occ = 0.75 * np.clip(e1 / np.maximum(b1, 0.08), 0, 1.6) + 0.25 * np.clip(e2 / np.maximum(b2, 0.08), 0, 1.6)
    # underside: blossom BELOW / away from the sun is exposed to nothing -> the opposite shift adds shade depth
    rows = np.nonzero(Bq.max(1) > 0.2)[0]
    yt, yb = (rows[0], rows[-1]) if len(rows) > 2 else (0, hq)
    yy = np.arange(hq, dtype=np.float32)[:, None]
    hgt = np.clip((yy - yt) / max(yb - yt, 1), 0, 1)
    v = -occ - 0.3 * hgt
    if clumps:
        # per-clump form: every clump is a lobed round mass lit from the sun (top-left, biased up); clumps are
        # laid top -> bottom so each lower clump's lit crown sits against the shaded underside of the one above
        Fv = np.zeros((hq, wq), np.float32)
        Fm = np.zeros((hq, wq), np.float32)
        yq, xq = np.mgrid[0:hq, 0:wq].astype(np.float32)
        L3 = np.array([lx, ly, 0.55])
        L3 = L3 / np.linalg.norm(L3)
        for (cx, cy, rc, p1, p2, nl) in sorted(clumps, key=lambda c: c[1]):
            cx, cy, rc = cx / q, cy / q, rc / q
            if rc < 1.0:
                continue
            x0, x1 = int(max(cx - 1.4 * rc, 0)), int(min(cx + 1.4 * rc + 1, wq))
            y0, y1 = int(max(cy - 1.4 * rc, 0)), int(min(cy + 1.4 * rc + 1, hq))
            if x1 <= x0 or y1 <= y0:
                continue
            dx = xq[y0:y1, x0:x1] - cx
            dy = yq[y0:y1, x0:x1] - cy
            th = np.arctan2(-dy, dx)
            rr = rc * (1.0 + 0.24 * np.cos(nl * th + p1) + 0.14 * np.cos((2 * nl + 1) * th + p2))
            d = np.hypot(dx, dy * 1.12) / rr
            m = np.clip((1.0 - d) * rc * 0.7 + 0.5, 0, 1)
            nx, ny = dx / rr, dy * 1.12 / rr
            nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
            lam = nx * L3[0] + ny * L3[1] + nz * L3[2]
            # flattened (painted planes): lit crown, body, a band of underside
            val = lam - 0.35 * np.clip(ny, 0, 1)
            val = 0.5 + 0.5 * np.tanh(3.0 * (val - 0.45))      # flattened into painted planes
            Fv[y0:y1, x0:x1] += (val - Fv[y0:y1, x0:x1]) * m
            Fm[y0:y1, x0:x1] = np.maximum(Fm[y0:y1, x0:x1], m)
        Fv = cv2.GaussianBlur(Fv, (0, 0), 0.6)
        vn = (v - float(np.median(v[Bq > 0.5])) if (Bq > 0.5).any() else v)
        v = vn * (1 - 0.6 * Fm) + (1.2 * (Fv - 0.5) - 0.4 * hgt) * 0.6 * Fm
    v = cv2.resize(v, (W, H), interpolation=cv2.INTER_CUBIC)
    sd = float(v[sel].std()) + 1e-4
    rng = np.random.default_rng(int(H * 7 + W))
    cell = int(6 * px + 1)
    dab = cv2.resize(rng.standard_normal((max(2, H // cell), max(2, W // cell))).astype(np.float32),
                     (W, H), interpolation=cv2.INTER_CUBIC)
    lb = cv2.GaussianBlur(lum, (0, 0), 2.2 * px)
    hp = lum - lb
    v = v + sd * (np.clip(hp, -0.15, 0.15) / 0.15 * 0.3 + 0.12 * dab)
    # rank -> 0..1 (area fractions fixed per card)
    qs = np.quantile(v[sel], np.linspace(0, 1, 33))
    u = np.interp(v, qs, np.linspace(0, 1, 33)).astype(np.float32)
    # painted dab mosaic: round dabs of varied size, each carrying one value offset; the continuous value is
    # stepped into NLEV levels through the dabs, so the lit -> body -> shade gradient is laid in as clustered
    # dabs (lost into a soft mass in the shade, see below) instead of posterised bands or an airbrush ramp
    dm = _dab_map(sel, px, rng)
    NLEV = 7.0
    uq = np.floor(u * NLEV + (dm - 0.5) * 1.1 + 0.5) / NLEV
    uq = cv2.GaussianBlur(uq.astype(np.float32), (0, 0), 0.45 * px)
    wq_ = _ss((u - 0.18) / 0.12)                       # stepped in the lit / body, smooth in the deep shade
    u = (u + (uq - u) * wq_).astype(np.float32)
    col = _ramp(np.clip(u, 0, 1))
    # deeper rose only in small pockets on the shade / body border
    cell2 = int(9 * px + 1)
    pk = cv2.resize(rng.random((max(2, H // cell2), max(2, W // cell2))).astype(np.float32), (W, H),
                    interpolation=cv2.INTER_CUBIC)
    rose = _ss((pk - 0.7) / 0.08) * np.exp(-((u - 0.38) / 0.14) ** 2) * (0.55 + 0.45 * _ss((dm - 0.3) / 0.2))
    col += (P_ROSE - col) * (0.7 * rose)[..., None]
    # warm transmitted glow on the thin sun-facing lit edge
    edge = np.clip(Bw - _shift(cv2.GaussianBlur(Bw, (0, 0), 1.5 * px), lx * 3 * px, ly * 3 * px), 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), 0.8 * px) * _ss((u - 0.6) / 0.2)
    col += (np.array([1.06, 0.92, 0.86], np.float32) - col) * (0.3 * edge)[..., None]
    if far:
        col = col * 0.85 + np.array([0.9, 0.8, 0.9], np.float32) * 0.15
    kth = max(3, int(round(4.5 * px)) | 1)
    w8 = (wood > 0.5).astype(np.uint8)
    thk = cv2.dilate(cv2.morphologyEx(w8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kth, kth))),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    thn = cv2.dilate(w8 & (1 - thk), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    wd_ = cv2.dilate(w8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    lumb = np.where(wd_ > 0, lb, lum)
    tex = np.clip(lumb - cv2.GaussianBlur(lumb, (0, 0), 2.2 * px), -0.12, 0.03)
    tex = cv2.GaussianBlur(tex, (0, 0), 0.5 * px)
    shade = 1.0 - _ss((u - 0.12) / 0.1)
    colt = col * (1.0 + (1.3 * (1 - 0.8 * shade))[..., None] * tex[..., None])
    kb = _ss(pink * 1.2)[..., None]
    Cn = C + (colt - C) * kb
    dens = cv2.GaussianBlur(Bw, (0, 0), 2.5 * px)
    kbur = np.clip(cv2.GaussianBlur(thn * np.clip((dens - 0.12) / 0.14, 0, 1), (0, 0), 0.5 * px) * 1.15, 0, 1)
    colb = cv2.GaussianBlur(col * Bw[..., None], (0, 0), 2.0 * px) / \
        np.maximum(cv2.GaussianBlur(Bw, (0, 0), 2.0 * px), 1e-3)[..., None]
    vis_w = (w8 > 0) & (kbur < 0.5)
    nlab, wl, stt, _ = cv2.connectedComponentsWithStats(vis_w.astype(np.uint8), connectivity=8)
    if nlab > 1:
        amin = 900.0 * px * px
        small = np.zeros(nlab, np.float32)
        small[1:] = (stt[1:, cv2.CC_STAT_AREA] < amin).astype(np.float32)
        frag = cv2.dilate(small[wl], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        frag = frag * np.clip((dens - 0.1) / 0.12, 0, 1)
        kbur = np.maximum(kbur, cv2.GaussianBlur(frag, (0, 0), 0.5 * px))
    Cn = Cn + (colb - Cn) * kbur[..., None]
    # ---- shade interior: one soft mass with lost edges (gaps closed, colour smoothed); crisp florets stay on
    # the silhouette and in the lit parts
    Ab = (A > 0.5).astype(np.uint8)
    kc = max(3, int(round(3.5 * px)) | 1)
    Acl = cv2.morphologyEx(Ab, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    ke = max(3, int(round(5.0 * px)) | 1)
    inner = cv2.erode(Acl, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke))).astype(np.float32)
    inner = cv2.GaussianBlur(inner, (0, 0), 1.5 * px)
    zs = np.clip(shade * 1.3, 0, 1) * inner * (1 - np.clip(wood * 2, 0, 1))
    zs = cv2.GaussianBlur(zs, (0, 0), 0.8 * px)
    Cp = Cn * A[..., None]
    sgm = 1.4 * px
    Cbl = cv2.GaussianBlur(Cp, (0, 0), sgm) / np.maximum(cv2.GaussianBlur(A, (0, 0), sgm), 1e-3)[..., None]
    Cn = Cn + (Cbl - Cn) * zs[..., None]
    A2 = A + (np.maximum(A, cv2.GaussianBlur(Acl.astype(np.float32), (0, 0), 0.8 * px)) - A) * zs
    # the lit / body interiors: gentle smoothing so the flower texture reads as dabs, not stipple
    Cs = cv2.GaussianBlur(Cn, (0, 0), 0.7 * px)
    out = np.dstack([Cn * A2[..., None], A2]).astype(np.float32)
    if trim:
        out = _trim25(out, Bw, wood, px)
    return out


def _trim25(pm, Bw, wood, px, bridge_keep=1.0, bridge_len=4.0, limb_len=8.0, term_keep=1.5, term_ramp=3.5):
    """R24 twig trim with shorter exposed stubs that taper (alpha + width) to a point"""
    old_k, old_r = 5.0, 5.0
    H, W = Bw.shape
    hull = (cv2.GaussianBlur(Bw, (0, 0), 3.0 * px) > 0.14).astype(np.uint8)
    kth = max(3, int(round(4.5 * px)) | 1)
    wood8 = (wood > 0.5).astype(np.uint8)
    thick = cv2.morphologyEx(wood8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kth, kth)))
    thick = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    thin = wood8 & (1 - thick)
    thin_soft = cv2.dilate(thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out_t = (thin & (1 - hull)).astype(np.uint8)
    if out_t.sum() == 0:
        return pm
    nt, lab = cv2.connectedComponents(out_t, connectivity=8)
    anchor = np.maximum(hull, thick)
    ring = cv2.dilate(out_t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) & anchor
    nc, clab = cv2.connectedComponents(ring.astype(np.uint8), connectivity=8)
    labd = cv2.dilate(lab.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    ys, xs = np.nonzero(ring)
    tl = labd[ys, xs].astype(np.int64)
    cl = clab[ys, xs].astype(np.int64)
    ok = tl > 0
    pairs = np.unique(tl[ok] * (nc + 1) + cl[ok])
    cnt = np.bincount(pairs // (nc + 1), minlength=nt)
    dist = cv2.distanceTransform((1 - anchor).astype(np.uint8), cv2.DIST_L2, 5)
    keep_len = term_keep * px
    ramp = term_ramp * px
    fade = np.clip(1.0 - (dist - keep_len) / ramp, 0, 1) ** 1.5
    kill = np.ones((H, W), np.float32)
    lab_s = cv2.dilate(lab.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.int64)
    term = cnt[lab_s] == 1
    bridge = cnt[lab_s] >= 2
    zero = (cnt[lab_s] == 0) & (lab_s > 0)
    m = (thin_soft > 0) & (1 - hull).astype(bool)
    # thin wood continuing a visible thick limb (not touching the blossom): the limb's own taper, faded to a
    # point over a longer run instead of being cut off at the thick / thin boundary (no blunt stubs)
    hl = cv2.connectedComponents(hull, connectivity=8)[1]
    ring_h = cv2.dilate(out_t, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) & hull
    ys2, xs2 = np.nonzero(ring_h)
    hit_h = np.zeros(nt, bool)
    hit_h[labd[ys2, xs2].astype(np.int64)] = True
    hit_h[0] = False
    dth = cv2.distanceTransform((1 - thick).astype(np.uint8), cv2.DIST_L2, 5)
    fade_t = np.clip(1.0 - (dth - limb_len * px) / (1.25 * limb_len * px), 0, 1) ** 1.2
    limb = (~hit_h[lab_s]) & (lab_s > 0)
    kill = np.where(m & term, fade, kill)
    kill = np.where(m & zero, 0.0, kill)
    if bridge_keep < 1.0:
        # round 26: most twigs spanning a sky gap between two masses are buried: they run a short tapering way
        # out of each mass and vanish; only a few (hash-picked) cross the gap
        hsh = (np.arange(nt + 1) * 0.6180339887 + 0.137) % 1.0
        fade_b = np.clip(1.0 - (dist - bridge_len * px) / (4.0 * px), 0, 1) ** 1.5
        kill = np.where(m & bridge, np.where(hsh[np.clip(lab_s, 0, nt)] < bridge_keep, 1.0, fade_b), kill)
    else:
        kill = np.where(m & bridge, 1.0, kill)
    kill = np.where(m & limb, fade_t, kill)
    kill = cv2.GaussianBlur(kill.astype(np.float32), (0, 0), 0.6 * px)
    kill = np.where(m | (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0), kill, 1.0)
    return (pm * kill[..., None]).astype(np.float32)
