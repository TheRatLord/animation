"""s05_sakura round 24 (reviewer FAIL on round 23: milky sun smear, flat bubble-gum canopy with white stipple,
hairy twig dashes past the silhouette, grey-purple mush canal).

  * regrade_card(): a painted value pass over every finished cherry card (premultiplied, supersampled):
      - the blossom is re-valued PER CLUMP and PER MASS from the sun direction (top-left, biased up): how much
        blossom lies between a pixel and the sun at clump scale and at mass scale -> pale near-white pink lit
        tips on the sun side, a pink mid-tone body, violet-grey shade clusters underneath / inside (~30% of the
        area, per-card quantiles) and a deeper violet-magenta core; band borders are broken at flower scale by the
        painter's own flower texture, so they read as clustered dabs, not smooth emboss;
      - the even white stipple is compressed (positive high-pass clamped), flower texture kept as darker gaps;
      - twigs: thin wood that leaves the blossom hull as a terminal twig is cut to a short tapered stub; bridges
        between two blossom masses are kept (continuous), floating dashes are removed.
  * Sun24: small hot white core, a clean thin tapered 6-point star, a tight soft halo (replaces the old
    starburst / rainbow ring / broad glow).
  * water24(): the canal repainted as clean broken horizontal reflection strokes (pink trees, sky cyan, sign
    colours) over a dark teal-blue base; few deliberate glitter sparkles on the sun path only.
"""
import math

import numpy as np
import cv2
from numba import njit, prange

from s05_sakura_raster import hash2, fbm2, sstep


# ============================================================================ canopy value pass
LIT = np.array([0.99, 0.85, 0.9], np.float32)        # pale near-white pink lit tips
LIT2 = np.array([0.97, 0.73, 0.84], np.float32)        # light pink (lit side of the body)
MID = np.array([0.93, 0.57, 0.75], np.float32)         # pink mid-tone body
SHADE = np.array([0.66, 0.52, 0.71], np.float32)      # violet-grey shade clusters (#b08fb8-ish)
DEEP = np.array([0.54, 0.38, 0.6], np.float32)       # violet-magenta core where the limbs pass through


def _ss(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _shift(a, dx, dy):
    h, w = a.shape[:2]
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])       # out(x) = a(x + d)
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def regrade_card(pm, sun_dir, px, far=False, trim=True, shade_q=0.32, lit_q=0.68):
    """pm: premultiplied (H, W, 4) supersampled card (px card pixels per screen pixel); sun_dir: unit vector
    toward the sun in image coordinates.  Returns a new premultiplied card."""
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

    # ---- fields at quarter resolution
    q = 4
    hq, wq = max(1, H // q), max(1, W // q)
    Bq = cv2.resize(Bw, (wq, hq), interpolation=cv2.INTER_AREA)
    lx, ly = float(sun_dir[0]), float(sun_dir[1]) - 0.7
    ln = math.hypot(lx, ly) + 1e-9
    lx, ly = lx / ln, ly / ln
    s1 = 9.0 * px / q
    s2 = 26.0 * px / q
    b1 = cv2.GaussianBlur(Bq, (0, 0), s1)
    b2 = cv2.GaussianBlur(Bq, (0, 0), s2)
    e1 = _shift(b1, lx * 1.3 * s1, ly * 1.3 * s1)
    e2 = _shift(b2, lx * 1.2 * s2, ly * 1.2 * s2)
    n1 = np.maximum(b1, 0.08)
    n2 = np.maximum(b2, 0.08)
    occ = 0.55 * np.clip(e1 / n1, 0, 1.6) + 0.45 * np.clip(e2 / n2, 0, 1.6)
    # crown height: upper crown lighter, lower crown sinks into shade
    rows = np.nonzero(Bq.max(1) > 0.2)[0]
    if len(rows) > 2:
        yt, yb = rows[0], rows[-1]
    else:
        yt, yb = 0, hq
    yy = np.arange(hq, dtype=np.float32)[:, None]
    hgt = np.clip((yy - yt) / max(yb - yt, 1), 0, 1)
    v = -occ - 0.35 * hgt
    # the painter's own mass values (smoothed) keep its intent
    lsm = cv2.GaussianBlur(cv2.resize(lum * Bw, (wq, hq), interpolation=cv2.INTER_AREA), (0, 0), s1) / \
        np.maximum(b1, 1e-3)
    v = v + 0.6 * (lsm - 0.75)
    v = cv2.resize(v, (W, H), interpolation=cv2.INTER_CUBIC)
    # flower-scale breakup of the band borders (clustered dabs, not a smooth emboss)
    lb = cv2.GaussianBlur(lum, (0, 0), 2.2 * px)
    hp = lum - lb
    sel = Bw > 0.5
    if sel.sum() < 50:
        return pm
    sd = float(v[sel].std()) + 1e-4
    rng = np.random.default_rng(int(H * 7 + W))
    dab = cv2.resize(rng.standard_normal((max(2, H // int(5 * px + 1)), max(2, W // int(5 * px + 1)))).astype(np.float32),
                     (W, H), interpolation=cv2.INTER_CUBIC)
    v = v + sd * (np.clip(hp, -0.15, 0.15) / 0.15 * 0.3 + 0.1 * dab)
    t_lo = float(np.quantile(v[sel], shade_q))
    t_hi = float(np.quantile(v[sel], lit_q))
    t_dp = float(np.quantile(v[sel], 0.1))
    t_top = float(np.quantile(v[sel], 0.92))
    w = 0.06 * sd
    k_mid = _ss((v - t_lo + w) / (2 * w))
    k_lit = _ss((v - t_hi + w) / (2 * w))
    k_top = _ss((v - t_top + w) / (2 * w))
    k_dp = 1 - _ss((v - t_dp + w) / (2 * w))
    col = DEEP[None, None] * k_dp[..., None] + SHADE[None, None] * (1 - k_dp)[..., None]
    col = col + (MID - col) * k_mid[..., None]
    col = col + (LIT2 - col) * k_lit[..., None]
    col = col + (LIT - col) * k_top[..., None]
    # warm peach transmitted glow on the thin sun-facing lit edge
    edge = np.clip(Bw - _shift(cv2.GaussianBlur(Bw, (0, 0), 1.5 * px), lx * 3 * px, ly * 3 * px), 0, 1)
    edge = cv2.GaussianBlur(edge, (0, 0), 0.8 * px) * k_lit
    col = col + (np.array([1.06, 0.9, 0.84], np.float32) - col) * (0.35 * edge)[..., None]
    if far:
        col = col * 0.85 + np.array([0.9, 0.8, 0.9], np.float32) * 0.15
    # thin wood (twigs) vs thick limbs
    kth = max(3, int(round(4.5 * px)) | 1)
    w8 = (wood > 0.5).astype(np.uint8)
    thk = cv2.dilate(cv2.morphologyEx(w8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kth, kth))),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    thn = cv2.dilate(w8 & (1 - thk), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    # flower texture (wood excluded): darker gaps kept, bright stipple compressed
    wd_ = cv2.dilate(w8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    lumb = np.where(wd_ > 0, lb, lum)
    tex = np.clip(lumb - cv2.GaussianBlur(lumb, (0, 0), 2.2 * px), -0.12, 0.03)
    tex = cv2.GaussianBlur(tex, (0, 0), 0.5 * px)
    colt = col * (1.0 + (1.3 + 0.5 * k_mid)[..., None] * tex[..., None])
    kb = _ss(pink * 1.2)[..., None]
    Cn = C + (colt - C) * kb
    # bury the thin wood inside dense blossom (only limbs cross the masses, no scratchy dashes)
    dens = cv2.GaussianBlur(Bw, (0, 0), 2.5 * px)
    kbur = np.clip(cv2.GaussianBlur(thn * np.clip((dens - 0.12) / 0.14, 0, 1), (0, 0), 0.5 * px) * 1.15, 0, 1)
    colb = cv2.GaussianBlur(col * Bw[..., None], (0, 0), 2.0 * px) /         np.maximum(cv2.GaussianBlur(Bw, (0, 0), 2.0 * px), 1e-3)[..., None]
    # small visible wood fragments left floating inside the blossom (limb pieces peeking through) are buried
    # too: only trunk-connected, sizeable wood stays visible
    vis_w = (w8 > 0) & (kbur < 0.5)
    nlab, wl, st, _ = cv2.connectedComponentsWithStats(vis_w.astype(np.uint8), connectivity=8)
    if nlab > 1:
        amin = 900.0 * px * px
        small = np.zeros(nlab, np.float32)
        small[1:] = (st[1:, cv2.CC_STAT_AREA] < amin).astype(np.float32)
        frag = small[wl]
        frag = cv2.dilate(frag, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        frag = frag * np.clip((dens - 0.1) / 0.12, 0, 1)
        kbur = np.maximum(kbur, cv2.GaussianBlur(frag, (0, 0), 0.5 * px))
    Cn = Cn + (colb - Cn) * kbur[..., None]
    # interior shade cluster edges a touch softer (lost edges underneath)
    Cs = cv2.GaussianBlur(Cn, (0, 0), 0.9 * px)
    ks = ((1 - k_mid) * kb[..., 0] * 0.6)[..., None]
    Cn = Cn + (Cs - Cn) * ks
    out = np.dstack([Cn * A[..., None], A]).astype(np.float32)
    if trim:
        out = _trim_twigs(out, Bw, wood, px)
    return out


def _trim_twigs(pm, Bw, wood, px):
    """terminal thin twigs leaving the blossom hull are cut to a short tapered stub; floating thin dashes
    are removed; bridges between two blossom regions are kept."""
    H, W = Bw.shape
    hull = (cv2.GaussianBlur(Bw, (0, 0), 3.0 * px) > 0.14).astype(np.uint8)
    kth = max(3, int(round(4.5 * px)) | 1)
    wood8 = (wood > 0.5).astype(np.uint8)
    thick = cv2.morphologyEx(wood8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kth, kth)))
    thick = cv2.dilate(thick, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    thin = wood8 & (1 - thick)
    # soft twig coverage (AA fringe included)
    thin_soft = cv2.dilate(thin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    out_t = (thin & (1 - hull)).astype(np.uint8)
    if out_t.sum() == 0:
        return pm
    nt, lab = cv2.connectedComponents(out_t, connectivity=8)
    # contacts with the hull (or the thick wood): label the contact pieces
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
    # distance from the anchor (hull / thick wood) along the image (a good proxy for the twig's exposed length)
    dist = cv2.distanceTransform((1 - anchor).astype(np.uint8), cv2.DIST_L2, 5)
    keep_len = 5.0 * px
    ramp = 5.0 * px
    fade = np.clip(1.0 - (dist - keep_len) / ramp, 0, 1)
    kill = np.ones(H * W, np.float32).reshape(H, W)
    lab_s = cv2.dilate(lab.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(np.int64)
    term = cnt[lab_s] == 1
    bridge = cnt[lab_s] >= 2
    zero = (cnt[lab_s] == 0) & (lab_s > 0)
    m = (thin_soft > 0) & (1 - hull).astype(bool)
    kill = np.where(m & term, fade, kill)
    kill = np.where(m & zero, 0.0, kill)
    kill = np.where(m & bridge, 1.0, kill)
    kill = cv2.GaussianBlur(kill.astype(np.float32), (0, 0), 0.6 * px)
    kill = np.where(m | (cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0), kill, 1.0)
    return (pm * kill[..., None]).astype(np.float32)


# ============================================================================ sun
class Sun24:
    """hot white core (~22 px radius, clipped), a clean tapered 6-point star (spikes 250-400 px) and a tight
    soft halo (~120 px across).  Star + halo and the core mask are precomputed sprites; per frame they are only
    rotated a hair (slow star drift) and added around the sun."""

    def __init__(self, W, H, base_deg=14.0, lens=None, lmax=400.0):
        self.W, self.H = W, H
        u = self.u = W / 1920.0
        self.base = math.radians(base_deg)
        self.lens = list(lens) if lens is not None else [1.0, 0.68, 0.9, 0.62, 0.95, 0.7]
        Lmax = lmax * u
        R = self.R = int(Lmax + 4)
        ys, xs = np.mgrid[-R:R, -R:R].astype(np.float32)
        dx, dy = xs + 0.5, ys + 0.5
        d = np.sqrt(dx * dx + dy * dy) + 1e-3
        star = np.zeros_like(d)
        for i in range(6):
            a = self.base + i * math.pi / 3
            ca, sa = math.cos(a), math.sin(a)
            along = dx * ca + dy * sa
            perp = np.abs(-dx * sa + dy * ca)
            L = Lmax * self.lens[i]
            tt = np.clip(along / L, 0, 1)
            wdt = (2.2 * u) * (1 - tt) ** 1.2 + 0.35 * u
            prof = np.exp(-(perp / wdt) ** 2) * (1 - tt) ** 1.6 * (along > 0)
            star = np.maximum(star, prof)
        star *= np.clip((d - 10 * u) / (14 * u), 0, 1) * 0.55
        halo = 0.55 * np.exp(-(d / (38 * u)) ** 2) + 0.12 * np.exp(-(d / (95 * u)) ** 2)
        warm = np.array([1.0, 0.95, 0.86], np.float32)
        self.spr = ((star * 0.9 + halo)[..., None] * warm).astype(np.float32)
        rc = 22 * u
        rq = int(rc * 3)
        ys, xs = np.mgrid[-rq:rq, -rq:rq].astype(np.float32)
        dc = np.sqrt((xs + 0.5) ** 2 + (ys + 0.5) ** 2)
        disc = np.clip((rc - dc) / 1.5 + 0.5, 0, 1)
        cor = 1.0 / (1.0 + ((dc - rc).clip(0) / (9 * u)) ** 2)
        self.ck = np.maximum(disc, 0.85 * cor * np.clip((dc - rc) / 2 + 1, 0, 1))[..., None].astype(np.float32)
        self.rq = rq

    def render(self, img, sx, sy, t, vis=1.0):
        W, H, R = self.W, self.H, self.R
        ang = math.degrees(0.015 * t)
        # sprite placed with its centre on the (sub-pixel) sun position
        x0, y0 = int(math.floor(sx)) - R, int(math.floor(sy)) - R
        fx, fy = sx - math.floor(sx), sy - math.floor(sy)
        M = cv2.getRotationMatrix2D((R - 0.5, R - 0.5), ang, 1.0)
        M[0, 2] += fx
        M[1, 2] += fy
        spr = cv2.warpAffine(self.spr, M, (2 * R, 2 * R), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        a0, a1 = max(x0, 0), min(x0 + 2 * R, W)
        b0, b1 = max(y0, 0), min(y0 + 2 * R, H)
        if a1 > a0 and b1 > b0:
            img[b0:b1, a0:a1] += spr[b0 - y0:b1 - y0, a0 - x0:a1 - x0] * vis
        return img

    def core(self, img, sx, sy):
        """after the tone shoulder: the hot core clipped white (never a grey disc)"""
        W, H, rq = self.W, self.H, self.rq
        x0, y0 = int(round(sx)) - rq, int(round(sy)) - rq
        a0, a1 = max(x0, 0), min(x0 + 2 * rq, W)
        b0, b1 = max(y0, 0), min(y0 + 2 * rq, H)
        if a1 <= a0 or b1 <= b0:
            return img
        k = self.ck[b0 - y0:b1 - y0, a0 - x0:a1 - x0]
        sub = img[b0:b1, a0:a1]
        hot = np.array([1.12, 1.12, 1.1], np.float32)
        sub[:] = sub + (np.maximum(sub, hot) - sub) * k
        return img


# ============================================================================ canal water
@njit(cache=True, parallel=True, fastmath=True)
def water24(out, wy, wx, wu, ws, wZ, wa, wry, refl, ywb, t, f, sunx, suny_m, deep, sun_col, invz, kf):
    """Clean broken horizontal reflection strokes over a dark teal base (premultiplied add by coverage)."""
    n = wy.shape[0]
    H, W = refl.shape[0], refl.shape[1]
    for k in prange(n):
        y = wy[k]
        x = wx[k]
        u = wu[k]
        s = ws[k]
        Z = wZ[k]
        fp = Z / f
        sf = s + 0.25 * t
        # stroke coordinates: ~screen-constant stroke size (length 60-200 px, height ~4-7 px)
        au = u * f / (150.0 * Z)
        bv = y / 5.5 - 0.35 * t
        n1 = fbm2(au + 0.13 * bv, bv, 91, 3)
        n2 = fbm2(au * 2.3 + 5.0, bv * 1.7 + 3.0, 93, 2)
        stroke = sstep(0.46, 0.53, n1 * 0.8 + n2 * 0.2 + 0.03)
        # per-stroke lateral shift (smooth, small): reflection sliced into bands that slip sideways
        dxs = (fbm2(au * 0.5 + 11.0, bv * 0.2, 94, 2) - 0.5) * (9.0 + 60.0 / Z)
        dys = (fbm2(au * 0.4, bv * 0.5, 95, 2) - 0.5) * 3.0
        ry_ = 2.0 * ywb[x] - y + dys
        iyo = min(max(int(2.0 * ywb[x] - y), 0), H - 1)
        rx_ = x + dxs - kf * (invz[y, x] - invz[iyo, x])
        if ry_ < 0:
            ry_ = 0.0
        if ry_ > H - 2:
            ry_ = H - 2.0
        if rx_ < 0:
            rx_ = 0.0
        if rx_ > W - 2:
            rx_ = W - 2.0
        iy = int(ry_)
        ix = int(rx_)
        fy = ry_ - iy
        fx = rx_ - ix
        cr = (refl[iy, ix, 0] * (1 - fx) + refl[iy, ix + 1, 0] * fx) * (1 - fy) + \
             (refl[iy + 1, ix, 0] * (1 - fx) + refl[iy + 1, ix + 1, 0] * fx) * fy
        cg = (refl[iy, ix, 1] * (1 - fx) + refl[iy, ix + 1, 1] * fx) * (1 - fy) + \
             (refl[iy + 1, ix, 1] * (1 - fx) + refl[iy + 1, ix + 1, 1] * fx) * fy
        cb = (refl[iy, ix, 2] * (1 - fx) + refl[iy, ix + 1, 2] * fx) * (1 - fy) + \
             (refl[iy + 1, ix, 2] * (1 - fx) + refl[iy + 1, ix + 1, 2] * fx) * fy
        # saturate the reflected colour a little (clean, not milky)
        lm = (cr + cg + cb) / 3.0
        cr = lm + (cr - lm) * 1.25
        cg = lm + (cg - lm) * 1.25
        cb = lm + (cb - lm) * 1.25
        cth = min(abs(wry[k]), 1.0)
        Fr = 0.05 + 0.95 * (1 - cth) ** 4
        Fr = min(0.72 + Fr * 0.5, 0.95)
        nb = sstep(-9.5, -5.0, u)
        rk = 1.0 - 0.28 * nb
        # dark teal-blue base between the strokes (carries a dim version of the reflection)
        br = deep[0] * (1 - 0.4 * nb) + cr * 0.22 * rk
        bg = deep[1] * (1 - 0.4 * nb) + cg * 0.24 * rk
        bb = deep[2] * (1 - 0.35 * nb) + cb * 0.26 * rk
        # far water (grazing) is almost all reflection; near water shows more base between the strokes
        cov = stroke * Fr + (1 - stroke) * sstep(0.004, 0.02, fp) * 0.0
        far = sstep(0.012, 0.03, fp)
        cov = cov + (Fr * 0.85 - cov) * far * 0.6
        sr = cr * 0.93 * rk
        sg = cg * 0.95 * rk
        sb = cb * 0.98 * rk
        r = br + (sr - br) * cov
        g = bg + (sg - bg) * cov
        b = bb + (sb - bb) * cov
        # a lighter sky tint on the upper part of each stroke (ripple crest catching the sky)
        hl = stroke * (1.0 - sstep(0.5, 0.6, n1)) * 0.08
        r += hl * 0.85
        g += hl * 0.95
        b += hl * 1.0
        # far wall's wet foot line
        wl = math.exp(-((y - ywb[x]) / (1.2 + 0.25 * f / Z)) ** 2) * 0.55
        r *= 1 - wl
        g *= 1 - wl * 0.9
        b *= 1 - wl * 0.75
        # sun glitter: a few deliberate sparkles on the sun path only
        gx = (x - sunx) / (0.05 * W)
        gy = (y - suny_m) / (0.2 * H)
        gw = math.exp(-gx * gx - gy * gy * 0.6)
        if gw > 0.02:
            cell = 0.35 + fp * 3.0
            cu = math.floor(u / cell)
            cs = math.floor(sf / (cell * 0.5))
            hh = hash2(int(cu), int(cs), 61)
            ph = hash2(int(cu), int(cs), 62) * 6.283
            tw = max(0.0, math.sin(t * (1.5 + 2.0 * hh) + ph)) ** 4
            pu = (cu + 0.5) * cell
            ps = (cs + 0.5) * cell * 0.5
            dd = ((u - pu) / (cell * 0.22)) ** 2 + ((sf - ps) / (cell * 0.06)) ** 2
            sp = math.exp(-dd * 1.5) * tw * (hh > 0.72)
            gl = gw * (sp * 1.1 + 0.12 * stroke)
            r += sun_col[0] * gl
            g += sun_col[1] * gl
            b += sun_col[2] * gl
        # petal rafts (hanaikada) along the banks
        sr_ = s + 0.35 * t
        d1 = fbm2(u * 0.3, sr_ * 0.045, 71, 3)
        d2 = fbm2(u * 1.1, sr_ * 0.25, 72, 2)
        w1 = 0.5 + 1.4 * fbm2(sr_ * 0.07, 4.1, 74, 2)
        c1 = -29.9 + 0.5 * w1 + (fbm2(sr_ * 0.05, 1.3, 73, 2) - 0.5) * 1.2
        w2 = 0.3 + 0.9 * fbm2(sr_ * 0.09, 7.7, 76, 2)
        c2 = -4.75 - 0.5 * w2 + (fbm2(sr_ * 0.06, 2.9, 77, 2) - 0.5) * 0.6
        rib = max(1.0 - sstep(0.6, 1.0, abs(u - c1) / w1), 1.0 - sstep(0.6, 1.0, abs(u - c2) / w2))
        brk = fbm2(u * 0.4, sr_ * 0.1, 75, 3)
        rib *= sstep(0.25, 0.36, brk)
        dens = max(sstep(0.3, 0.45, rib * (0.7 + 0.6 * d2)), sstep(0.66, 0.74, d1 + (d2 - 0.5) * 0.3))
        if dens > 0.0:
            pcov = dens * 0.8
            pr = 1.0 - 0.1 * dens
            r += (pr - r) * pcov
            g += (0.8 * pr - g) * pcov
            b += (0.87 * pr - b) * pcov
        a = wa[k]
        out[y, x, 0] += r * a
        out[y, x, 1] += g * a
        out[y, x, 2] += b * a
